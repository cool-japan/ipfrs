//! Tensor flow controller — manages backpressure, rate limiting, and priority-based admission
//! for tensor data flowing through a processing pipeline.

/// Operational state of the flow controller.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FlowState {
    /// Normal operation; items are accepted and processed at full rate.
    Running,
    /// Rate limit applied; some items may be delayed.
    Throttled,
    /// All flow halted; no items are processed.
    Paused,
    /// No new items accepted; existing items continue to drain.
    Draining,
}

/// Priority level assigned to a flow item.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum FlowPriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

/// A single item flowing through the pipeline.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FlowItem {
    pub item_id: u64,
    pub tensor_id: u64,
    pub priority: FlowPriority,
    pub size_bytes: u64,
    pub enqueued_at_tick: u64,
}

/// Configuration for the flow controller.
#[derive(Clone, Debug)]
pub struct FlowControllerConfig {
    /// Maximum number of items the queue may hold.
    pub max_queue_size: usize,
    /// Maximum bytes to process in a single tick.
    pub max_bytes_per_tick: u64,
    /// Queue fill ratio (0.0–1.0) at which the state transitions to `Throttled`.
    pub backpressure_threshold: f64,
}

impl Default for FlowControllerConfig {
    fn default() -> Self {
        Self {
            max_queue_size: 256,
            max_bytes_per_tick: 1_048_576,
            backpressure_threshold: 0.8,
        }
    }
}

/// Cumulative statistics for the flow controller.
#[derive(Clone, Debug, Default)]
pub struct FlowStats {
    pub total_admitted: u64,
    pub total_dropped: u64,
    pub total_processed: u64,
    pub total_bytes_processed: u64,
}

impl FlowStats {
    /// Drop rate: `dropped / (admitted + dropped)`. Returns `0.0` when both are zero.
    pub fn drop_rate(&self) -> f64 {
        let total = self.total_admitted + self.total_dropped;
        if total == 0 {
            0.0
        } else {
            self.total_dropped as f64 / total as f64
        }
    }
}

/// Controls the flow of tensor data through a processing pipeline.
///
/// The internal queue is kept sorted by priority descending (Critical first),
/// then by `enqueued_at_tick` ascending (FIFO within a priority level).
pub struct TensorFlowController {
    queue: Vec<FlowItem>,
    state: FlowState,
    config: FlowControllerConfig,
    stats: FlowStats,
}

impl TensorFlowController {
    /// Create a new controller with the given configuration, starting in `Running` state.
    pub fn new(config: FlowControllerConfig) -> Self {
        Self {
            queue: Vec::new(),
            state: FlowState::Running,
            config,
            stats: FlowStats::default(),
        }
    }

    /// Attempt to admit a new item into the queue.
    ///
    /// Returns `true` if the item was admitted, `false` if it was dropped.
    pub fn admit(&mut self, item: FlowItem) -> bool {
        // Paused and Draining states reject all new items.
        if self.state == FlowState::Paused || self.state == FlowState::Draining {
            self.stats.total_dropped += 1;
            return false;
        }

        // Reject when the queue is at capacity.
        if self.queue.len() >= self.config.max_queue_size {
            self.stats.total_dropped += 1;
            return false;
        }

        // Insert maintaining sort order: priority desc, then enqueued_at_tick asc.
        let pos = self.queue.partition_point(|existing| {
            existing.priority > item.priority
                || (existing.priority == item.priority
                    && existing.enqueued_at_tick <= item.enqueued_at_tick)
        });
        self.queue.insert(pos, item);
        self.stats.total_admitted += 1;

        // Update state based on queue fill ratio.
        self.state = self.compute_state_after_admit();
        true
    }

    /// Process one tick: drain items from the front of the queue up to the byte budget.
    ///
    /// Returns the items that were processed during this tick.
    pub fn process_tick(&mut self) -> Vec<FlowItem> {
        if self.state == FlowState::Paused {
            return Vec::new();
        }

        let mut processed = Vec::new();
        let mut bytes_remaining = self.config.max_bytes_per_tick;

        while let Some(front) = self.queue.first() {
            if front.size_bytes > bytes_remaining {
                break;
            }
            bytes_remaining -= front.size_bytes;
            let item = self.queue.remove(0);
            self.stats.total_processed += 1;
            self.stats.total_bytes_processed += item.size_bytes;
            processed.push(item);
        }

        // Update state after draining.
        self.state = self.compute_state_after_tick();
        processed
    }

    /// Halt all flow immediately.
    pub fn pause(&mut self) {
        self.state = FlowState::Paused;
    }

    /// Resume from a `Paused` state. The new state is determined by the current queue fill.
    pub fn resume(&mut self) {
        if self.state == FlowState::Paused {
            self.state = self.fill_based_state();
        }
    }

    /// Stop admitting new items; allow the queue to drain.
    pub fn drain(&mut self) {
        self.state = FlowState::Draining;
    }

    /// Current number of items waiting in the queue.
    pub fn queue_len(&self) -> usize {
        self.queue.len()
    }

    /// Reference to the current statistics.
    pub fn stats(&self) -> &FlowStats {
        &self.stats
    }

    // ── internal helpers ──────────────────────────────────────────────────────

    /// Derive the correct state based solely on queue fill ratio (used after `admit`).
    fn compute_state_after_admit(&self) -> FlowState {
        let fill = self.queue.len() as f64 / self.config.max_queue_size as f64;
        if fill >= self.config.backpressure_threshold {
            FlowState::Throttled
        } else {
            FlowState::Running
        }
    }

    /// Derive the correct state after a `process_tick` call.
    fn compute_state_after_tick(&self) -> FlowState {
        if self.state == FlowState::Draining && self.queue.is_empty() {
            return FlowState::Paused;
        }
        self.fill_based_state()
    }

    /// Running vs Throttled based on queue fill — does not consider Paused / Draining.
    fn fill_based_state(&self) -> FlowState {
        let fill = self.queue.len() as f64 / self.config.max_queue_size as f64;
        if fill >= self.config.backpressure_threshold {
            FlowState::Throttled
        } else {
            FlowState::Running
        }
    }
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_controller() -> TensorFlowController {
        TensorFlowController::new(FlowControllerConfig::default())
    }

    fn make_item(item_id: u64, priority: FlowPriority, size_bytes: u64, tick: u64) -> FlowItem {
        FlowItem {
            item_id,
            tensor_id: item_id * 10,
            priority,
            size_bytes,
            enqueued_at_tick: tick,
        }
    }

    // 1. new() starts Running with empty queue
    #[test]
    fn test_new_starts_running() {
        let ctrl = default_controller();
        assert_eq!(ctrl.state, FlowState::Running);
        assert_eq!(ctrl.queue_len(), 0);
    }

    // 2. admit in Running state succeeds
    #[test]
    fn test_admit_running_succeeds() {
        let mut ctrl = default_controller();
        let admitted = ctrl.admit(make_item(1, FlowPriority::Normal, 100, 0));
        assert!(admitted);
        assert_eq!(ctrl.queue_len(), 1);
    }

    // 3. admit when Paused drops item
    #[test]
    fn test_admit_paused_drops() {
        let mut ctrl = default_controller();
        ctrl.pause();
        let admitted = ctrl.admit(make_item(1, FlowPriority::Normal, 100, 0));
        assert!(!admitted);
        assert_eq!(ctrl.stats().total_dropped, 1);
        assert_eq!(ctrl.queue_len(), 0);
    }

    // 4. admit when Draining drops item
    #[test]
    fn test_admit_draining_drops() {
        let mut ctrl = default_controller();
        ctrl.drain();
        let admitted = ctrl.admit(make_item(1, FlowPriority::Normal, 100, 0));
        assert!(!admitted);
        assert_eq!(ctrl.stats().total_dropped, 1);
        assert_eq!(ctrl.queue_len(), 0);
    }

    // 5. admit at capacity drops item
    #[test]
    fn test_admit_at_capacity_drops() {
        let config = FlowControllerConfig {
            max_queue_size: 2,
            max_bytes_per_tick: 1_048_576,
            backpressure_threshold: 0.99,
        };
        let mut ctrl = TensorFlowController::new(config);
        assert!(ctrl.admit(make_item(1, FlowPriority::Normal, 100, 0)));
        assert!(ctrl.admit(make_item(2, FlowPriority::Normal, 100, 1)));
        let admitted = ctrl.admit(make_item(3, FlowPriority::Normal, 100, 2));
        assert!(!admitted);
        assert_eq!(ctrl.stats().total_dropped, 1);
    }

    // 6. backpressure_threshold triggers Throttled
    #[test]
    fn test_backpressure_threshold_throttled() {
        let config = FlowControllerConfig {
            max_queue_size: 10,
            max_bytes_per_tick: 1_048_576,
            backpressure_threshold: 0.8,
        };
        let mut ctrl = TensorFlowController::new(config);
        // Fill to 8 items (80% = threshold)
        for i in 0..8 {
            ctrl.admit(make_item(i, FlowPriority::Normal, 100, i));
        }
        assert_eq!(ctrl.state, FlowState::Throttled);
    }

    // 7. below threshold stays Running
    #[test]
    fn test_below_threshold_stays_running() {
        let config = FlowControllerConfig {
            max_queue_size: 10,
            max_bytes_per_tick: 1_048_576,
            backpressure_threshold: 0.8,
        };
        let mut ctrl = TensorFlowController::new(config);
        // 7 items = 70% < 80%
        for i in 0..7 {
            ctrl.admit(make_item(i, FlowPriority::Normal, 100, i));
        }
        assert_eq!(ctrl.state, FlowState::Running);
    }

    // 8. process_tick returns items up to byte budget
    #[test]
    fn test_process_tick_respects_byte_budget() {
        let config = FlowControllerConfig {
            max_queue_size: 256,
            max_bytes_per_tick: 300,
            backpressure_threshold: 0.8,
        };
        let mut ctrl = TensorFlowController::new(config);
        ctrl.admit(make_item(1, FlowPriority::Normal, 100, 0));
        ctrl.admit(make_item(2, FlowPriority::Normal, 100, 1));
        ctrl.admit(make_item(3, FlowPriority::Normal, 100, 2));
        ctrl.admit(make_item(4, FlowPriority::Normal, 100, 3));
        let processed = ctrl.process_tick();
        // 3 × 100 = 300, 4th would exceed budget
        assert_eq!(processed.len(), 3);
        assert_eq!(ctrl.queue_len(), 1);
    }

    // 9. process_tick returns empty when Paused
    #[test]
    fn test_process_tick_paused_returns_empty() {
        let mut ctrl = default_controller();
        ctrl.admit(make_item(1, FlowPriority::Normal, 100, 0));
        ctrl.pause();
        let processed = ctrl.process_tick();
        assert!(processed.is_empty());
        assert_eq!(ctrl.queue_len(), 1);
    }

    // 10. process_tick respects priority order (Critical before Low)
    #[test]
    fn test_process_tick_priority_order() {
        let config = FlowControllerConfig {
            max_queue_size: 256,
            max_bytes_per_tick: 200,
            backpressure_threshold: 0.8,
        };
        let mut ctrl = TensorFlowController::new(config);
        // Admit Low first, then Critical
        ctrl.admit(make_item(1, FlowPriority::Low, 100, 0));
        ctrl.admit(make_item(2, FlowPriority::Critical, 100, 1));
        let processed = ctrl.process_tick();
        assert_eq!(processed.len(), 2);
        // Critical must come first
        assert_eq!(processed[0].priority, FlowPriority::Critical);
        assert_eq!(processed[1].priority, FlowPriority::Low);
    }

    // 11. process_tick respects FIFO within same priority
    #[test]
    fn test_process_tick_fifo_within_priority() {
        let config = FlowControllerConfig {
            max_queue_size: 256,
            max_bytes_per_tick: 300,
            backpressure_threshold: 0.8,
        };
        let mut ctrl = TensorFlowController::new(config);
        ctrl.admit(make_item(10, FlowPriority::Normal, 100, 5));
        ctrl.admit(make_item(20, FlowPriority::Normal, 100, 3));
        ctrl.admit(make_item(30, FlowPriority::Normal, 100, 7));
        let processed = ctrl.process_tick();
        assert_eq!(processed.len(), 3);
        assert_eq!(processed[0].enqueued_at_tick, 3);
        assert_eq!(processed[1].enqueued_at_tick, 5);
        assert_eq!(processed[2].enqueued_at_tick, 7);
    }

    // 12. process_tick updates total_processed
    #[test]
    fn test_process_tick_updates_total_processed() {
        let mut ctrl = default_controller();
        ctrl.admit(make_item(1, FlowPriority::Normal, 100, 0));
        ctrl.admit(make_item(2, FlowPriority::Normal, 100, 1));
        ctrl.process_tick();
        assert_eq!(ctrl.stats().total_processed, 2);
    }

    // 13. process_tick updates total_bytes_processed
    #[test]
    fn test_process_tick_updates_total_bytes() {
        let mut ctrl = default_controller();
        ctrl.admit(make_item(1, FlowPriority::Normal, 400, 0));
        ctrl.admit(make_item(2, FlowPriority::Normal, 600, 1));
        ctrl.process_tick();
        assert_eq!(ctrl.stats().total_bytes_processed, 1000);
    }

    // 14. Draining + empty queue → Paused after process_tick
    #[test]
    fn test_draining_empty_queue_becomes_paused() {
        let mut ctrl = default_controller();
        ctrl.admit(make_item(1, FlowPriority::Normal, 100, 0));
        ctrl.drain();
        assert_eq!(ctrl.state, FlowState::Draining);
        ctrl.process_tick();
        assert_eq!(ctrl.state, FlowState::Paused);
    }

    // 15. pause() sets Paused
    #[test]
    fn test_pause_sets_paused() {
        let mut ctrl = default_controller();
        ctrl.pause();
        assert_eq!(ctrl.state, FlowState::Paused);
    }

    // 16. resume() from Paused sets Running (empty queue)
    #[test]
    fn test_resume_from_paused_sets_running() {
        let mut ctrl = default_controller();
        ctrl.pause();
        ctrl.resume();
        assert_eq!(ctrl.state, FlowState::Running);
    }

    // 17. resume() from Paused with heavy queue sets Throttled
    #[test]
    fn test_resume_from_paused_heavy_queue_sets_throttled() {
        let config = FlowControllerConfig {
            max_queue_size: 10,
            max_bytes_per_tick: 1_048_576,
            backpressure_threshold: 0.5,
        };
        let mut ctrl = TensorFlowController::new(config);
        // Admit 5 items (50% = threshold) → Throttled
        for i in 0..5 {
            ctrl.admit(make_item(i, FlowPriority::Normal, 100, i));
        }
        ctrl.pause();
        ctrl.resume();
        assert_eq!(ctrl.state, FlowState::Throttled);
    }

    // 18. drain() sets Draining
    #[test]
    fn test_drain_sets_draining() {
        let mut ctrl = default_controller();
        ctrl.drain();
        assert_eq!(ctrl.state, FlowState::Draining);
    }

    // 19. drop_rate() is 0.0 when nothing dropped
    #[test]
    fn test_drop_rate_zero_when_nothing_dropped() {
        let ctrl = default_controller();
        assert_eq!(ctrl.stats().drop_rate(), 0.0);
    }

    // 20. drop_rate() correct when items dropped
    #[test]
    fn test_drop_rate_correct() {
        let config = FlowControllerConfig {
            max_queue_size: 1,
            max_bytes_per_tick: 1_048_576,
            backpressure_threshold: 0.99,
        };
        let mut ctrl = TensorFlowController::new(config);
        ctrl.admit(make_item(1, FlowPriority::Normal, 100, 0)); // admitted
        ctrl.admit(make_item(2, FlowPriority::Normal, 100, 1)); // dropped (full)
                                                                // admitted=1, dropped=1 → drop_rate = 0.5
        let rate = ctrl.stats().drop_rate();
        assert!((rate - 0.5).abs() < f64::EPSILON);
    }

    // 21. total_admitted increments
    #[test]
    fn test_total_admitted_increments() {
        let mut ctrl = default_controller();
        ctrl.admit(make_item(1, FlowPriority::Normal, 100, 0));
        ctrl.admit(make_item(2, FlowPriority::Normal, 100, 1));
        ctrl.admit(make_item(3, FlowPriority::Normal, 100, 2));
        assert_eq!(ctrl.stats().total_admitted, 3);
    }

    // 22. total_dropped increments on drop
    #[test]
    fn test_total_dropped_increments() {
        let mut ctrl = default_controller();
        ctrl.pause();
        ctrl.admit(make_item(1, FlowPriority::Normal, 100, 0));
        ctrl.admit(make_item(2, FlowPriority::Normal, 100, 1));
        assert_eq!(ctrl.stats().total_dropped, 2);
    }

    // 23. queue_len reflects actual size
    #[test]
    fn test_queue_len_reflects_size() {
        let mut ctrl = default_controller();
        assert_eq!(ctrl.queue_len(), 0);
        ctrl.admit(make_item(1, FlowPriority::Normal, 100, 0));
        assert_eq!(ctrl.queue_len(), 1);
        ctrl.admit(make_item(2, FlowPriority::Normal, 100, 1));
        assert_eq!(ctrl.queue_len(), 2);
        ctrl.process_tick();
        assert_eq!(ctrl.queue_len(), 0);
    }

    // 24. Mixed priority ordering is preserved across admit calls
    #[test]
    fn test_mixed_priority_ordering() {
        let config = FlowControllerConfig {
            max_queue_size: 256,
            max_bytes_per_tick: 1_048_576,
            backpressure_threshold: 0.8,
        };
        let mut ctrl = TensorFlowController::new(config);
        ctrl.admit(make_item(1, FlowPriority::Low, 10, 0));
        ctrl.admit(make_item(2, FlowPriority::High, 10, 1));
        ctrl.admit(make_item(3, FlowPriority::Normal, 10, 2));
        ctrl.admit(make_item(4, FlowPriority::Critical, 10, 3));
        let processed = ctrl.process_tick();
        assert_eq!(processed[0].priority, FlowPriority::Critical);
        assert_eq!(processed[1].priority, FlowPriority::High);
        assert_eq!(processed[2].priority, FlowPriority::Normal);
        assert_eq!(processed[3].priority, FlowPriority::Low);
    }

    // 25. Draining state still processes existing items
    #[test]
    fn test_draining_processes_existing_items() {
        let mut ctrl = default_controller();
        ctrl.admit(make_item(1, FlowPriority::Normal, 100, 0));
        ctrl.admit(make_item(2, FlowPriority::Normal, 100, 1));
        ctrl.drain();
        let processed = ctrl.process_tick();
        assert_eq!(processed.len(), 2);
        assert_eq!(ctrl.stats().total_processed, 2);
    }
}
