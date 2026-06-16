//! I/O request scheduling with deadline and priority ordering.
//!
//! Provides a [`StorageIOScheduler`] that accepts read/write I/O requests,
//! orders them by priority (deadline-urgent first), and dispatches them
//! in an efficient sequence. Tracks completion statistics and supports
//! expiration of requests that miss their deadline.

// ---------------------------------------------------------------------------
// IOPriority
// ---------------------------------------------------------------------------

/// Priority level for an I/O request.
///
/// Higher numeric values are dispatched first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum IOPriority {
    /// Lowest priority — background maintenance, GC sweeps, etc.
    Background = 0,
    /// Default priority for user-initiated operations.
    Normal = 1,
    /// Elevated priority — latency-sensitive reads, replication.
    High = 2,
    /// Highest priority — real-time streaming, health probes.
    Realtime = 3,
}

// ---------------------------------------------------------------------------
// IODirection
// ---------------------------------------------------------------------------

/// Whether the I/O request is a read or a write.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IODirection {
    /// Read from storage.
    Read,
    /// Write to storage.
    Write,
}

// ---------------------------------------------------------------------------
// IORequest
// ---------------------------------------------------------------------------

/// A single I/O request submitted to the scheduler.
#[derive(Debug, Clone)]
pub struct IORequest {
    /// Unique identifier assigned by the scheduler.
    pub id: u64,
    /// CID of the block this request targets.
    pub block_cid: String,
    /// Read or write.
    pub direction: IODirection,
    /// Scheduling priority.
    pub priority: IOPriority,
    /// Payload size in bytes.
    pub size_bytes: u64,
    /// Optional tick by which the request must be dispatched.
    pub deadline_tick: Option<u64>,
    /// Tick at which the request was enqueued.
    pub enqueued_tick: u64,
}

// ---------------------------------------------------------------------------
// SchedulerConfig
// ---------------------------------------------------------------------------

/// Tuning knobs for the [`StorageIOScheduler`].
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    /// Maximum number of pending (un-dispatched) requests.
    pub max_pending: usize,
    /// Fraction of bandwidth allocated to reads (0.0–1.0).
    pub read_weight: f64,
    /// Fraction of bandwidth allocated to writes (0.0–1.0).
    pub write_weight: f64,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_pending: 1000,
            read_weight: 0.7,
            write_weight: 0.3,
        }
    }
}

// ---------------------------------------------------------------------------
// IOSchedulerStats
// ---------------------------------------------------------------------------

/// Snapshot of scheduler statistics.
#[derive(Debug, Clone)]
pub struct IOSchedulerStats {
    /// Number of requests currently pending dispatch.
    pub pending_count: usize,
    /// Total read requests successfully completed.
    pub completed_reads: u64,
    /// Total write requests successfully completed.
    pub completed_writes: u64,
    /// Cumulative bytes of completed requests.
    pub total_bytes_scheduled: u64,
    /// Number of pending requests whose deadline has passed.
    pub expired_count: usize,
}

// ---------------------------------------------------------------------------
// StorageIOScheduler
// ---------------------------------------------------------------------------

/// I/O request scheduler with deadline-aware, priority-based ordering.
///
/// # Ordering rules
///
/// 1. Requests whose `deadline_tick` is at or before the current tick are
///    dispatched first (earliest deadline first).
/// 2. Among non-deadline-urgent requests, higher [`IOPriority`] wins.
/// 3. Ties within the same priority level are broken by enqueue order (FIFO).
pub struct StorageIOScheduler {
    config: SchedulerConfig,
    pending: Vec<IORequest>,
    next_id: u64,
    current_tick: u64,
    completed_reads: u64,
    completed_writes: u64,
    total_bytes_scheduled: u64,
}

impl StorageIOScheduler {
    /// Create a new scheduler with the given configuration.
    pub fn new(config: SchedulerConfig) -> Self {
        Self {
            config,
            pending: Vec::new(),
            next_id: 0,
            current_tick: 0,
            completed_reads: 0,
            completed_writes: 0,
            total_bytes_scheduled: 0,
        }
    }

    /// Submit a new I/O request.
    ///
    /// Returns the unique request ID on success, or an error if the pending
    /// queue is at capacity.
    pub fn submit(
        &mut self,
        block_cid: &str,
        direction: IODirection,
        priority: IOPriority,
        size_bytes: u64,
        deadline_tick: Option<u64>,
    ) -> Result<u64, String> {
        if self.pending.len() >= self.config.max_pending {
            return Err(format!(
                "pending queue full ({} / {})",
                self.pending.len(),
                self.config.max_pending,
            ));
        }

        let id = self.next_id;
        self.next_id = self.next_id.saturating_add(1);

        self.pending.push(IORequest {
            id,
            block_cid: block_cid.to_string(),
            direction,
            priority,
            size_bytes,
            deadline_tick,
            enqueued_tick: self.current_tick,
        });

        Ok(id)
    }

    /// Pop and return the highest-priority pending request.
    ///
    /// Ordering:
    /// - Deadline-urgent requests (deadline ≤ current tick) come first,
    ///   ordered by earliest deadline, then FIFO.
    /// - Non-urgent requests follow, ordered by priority desc then FIFO.
    pub fn next_request(&mut self) -> Option<IORequest> {
        if self.pending.is_empty() {
            return None;
        }

        let tick = self.current_tick;

        // Find the best candidate index.
        let mut best_idx: usize = 0;
        let mut best_urgent = false;
        let mut best_deadline: u64 = u64::MAX;
        let mut best_priority = IOPriority::Background;
        let mut best_enqueued: u64 = u64::MAX;

        for (i, req) in self.pending.iter().enumerate() {
            let urgent = req.deadline_tick.map(|d| d <= tick).unwrap_or(false);

            let better = if urgent && !best_urgent {
                // Urgent beats non-urgent unconditionally.
                true
            } else if urgent && best_urgent {
                // Both urgent — earlier deadline wins, FIFO tiebreak.
                let dl = req.deadline_tick.unwrap_or(u64::MAX);
                dl < best_deadline || (dl == best_deadline && req.enqueued_tick < best_enqueued)
            } else if !urgent && best_urgent {
                false
            } else {
                // Neither urgent — higher priority wins, FIFO tiebreak.
                req.priority > best_priority
                    || (req.priority == best_priority && req.enqueued_tick < best_enqueued)
            };

            if better {
                best_idx = i;
                best_urgent = urgent;
                best_deadline = req.deadline_tick.unwrap_or(u64::MAX);
                best_priority = req.priority;
                best_enqueued = req.enqueued_tick;
            }
        }

        Some(self.pending.remove(best_idx))
    }

    /// Record the completion of a previously dispatched request.
    ///
    /// Updates internal counters for reads, writes, and bytes scheduled.
    pub fn complete(&mut self, request_id: u64, direction: IODirection, size_bytes: u64) {
        match direction {
            IODirection::Read => self.completed_reads = self.completed_reads.saturating_add(1),
            IODirection::Write => self.completed_writes = self.completed_writes.saturating_add(1),
        }
        self.total_bytes_scheduled = self.total_bytes_scheduled.saturating_add(size_bytes);
        // Also remove from pending if still there (idempotent).
        self.pending.retain(|r| r.id != request_id);
    }

    /// Cancel a pending request by ID.
    ///
    /// Returns `true` if the request was found and removed.
    pub fn cancel(&mut self, request_id: u64) -> bool {
        let before = self.pending.len();
        self.pending.retain(|r| r.id != request_id);
        self.pending.len() < before
    }

    /// Number of requests currently pending dispatch.
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Number of pending read requests.
    pub fn pending_reads(&self) -> usize {
        self.pending
            .iter()
            .filter(|r| r.direction == IODirection::Read)
            .count()
    }

    /// Number of pending write requests.
    pub fn pending_writes(&self) -> usize {
        self.pending
            .iter()
            .filter(|r| r.direction == IODirection::Write)
            .count()
    }

    /// Advance the internal clock by one tick.
    pub fn tick(&mut self) {
        self.current_tick = self.current_tick.saturating_add(1);
    }

    /// Return the current tick value.
    pub fn current_tick(&self) -> u64 {
        self.current_tick
    }

    /// References to pending requests whose deadline has passed.
    pub fn expired_requests(&self) -> Vec<&IORequest> {
        let tick = self.current_tick;
        self.pending
            .iter()
            .filter(|r| r.deadline_tick.map(|d| d < tick).unwrap_or(false))
            .collect()
    }

    /// Remove and return all pending requests whose deadline has passed.
    pub fn drain_expired(&mut self) -> Vec<IORequest> {
        let tick = self.current_tick;
        let mut expired = Vec::new();
        let mut kept = Vec::new();
        for req in self.pending.drain(..) {
            if req.deadline_tick.map(|d| d < tick).unwrap_or(false) {
                expired.push(req);
            } else {
                kept.push(req);
            }
        }
        self.pending = kept;
        expired
    }

    /// Snapshot of current scheduler statistics.
    pub fn stats(&self) -> IOSchedulerStats {
        IOSchedulerStats {
            pending_count: self.pending.len(),
            completed_reads: self.completed_reads,
            completed_writes: self.completed_writes,
            total_bytes_scheduled: self.total_bytes_scheduled,
            expired_count: self.expired_requests().len(),
        }
    }

    /// Borrow the scheduler configuration.
    pub fn config(&self) -> &SchedulerConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_scheduler() -> StorageIOScheduler {
        StorageIOScheduler::new(SchedulerConfig::default())
    }

    fn small_scheduler(max: usize) -> StorageIOScheduler {
        StorageIOScheduler::new(SchedulerConfig {
            max_pending: max,
            ..SchedulerConfig::default()
        })
    }

    // -- submit basics ------------------------------------------------------

    #[test]
    fn test_submit_returns_unique_ids() {
        let mut s = default_scheduler();
        let id0 = s.submit("cid0", IODirection::Read, IOPriority::Normal, 100, None);
        let id1 = s.submit("cid1", IODirection::Write, IOPriority::Normal, 200, None);
        assert!(id0.is_ok());
        assert!(id1.is_ok());
        assert_ne!(id0.ok(), id1.ok());
    }

    #[test]
    fn test_submit_increments_pending() {
        let mut s = default_scheduler();
        assert_eq!(s.pending_count(), 0);
        let _ = s.submit("c", IODirection::Read, IOPriority::Normal, 10, None);
        assert_eq!(s.pending_count(), 1);
        let _ = s.submit("c", IODirection::Write, IOPriority::High, 20, None);
        assert_eq!(s.pending_count(), 2);
    }

    #[test]
    fn test_submit_max_pending_enforced() {
        let mut s = small_scheduler(2);
        assert!(s
            .submit("a", IODirection::Read, IOPriority::Normal, 1, None)
            .is_ok());
        assert!(s
            .submit("b", IODirection::Read, IOPriority::Normal, 1, None)
            .is_ok());
        let res = s.submit("c", IODirection::Read, IOPriority::Normal, 1, None);
        assert!(res.is_err());
        assert!(res.err().unwrap_or_default().contains("full"));
    }

    // -- next_request ordering by priority ----------------------------------

    #[test]
    fn test_next_request_priority_ordering() {
        let mut s = default_scheduler();
        let _ = s.submit("lo", IODirection::Read, IOPriority::Background, 1, None);
        let _ = s.submit("hi", IODirection::Read, IOPriority::High, 1, None);
        let _ = s.submit("mid", IODirection::Read, IOPriority::Normal, 1, None);

        let r = s.next_request().expect("should have request");
        assert_eq!(r.block_cid, "hi");
        let r = s.next_request().expect("should have request");
        assert_eq!(r.block_cid, "mid");
        let r = s.next_request().expect("should have request");
        assert_eq!(r.block_cid, "lo");
        assert!(s.next_request().is_none());
    }

    #[test]
    fn test_next_request_realtime_highest() {
        let mut s = default_scheduler();
        let _ = s.submit("h", IODirection::Read, IOPriority::High, 1, None);
        let _ = s.submit("rt", IODirection::Read, IOPriority::Realtime, 1, None);

        let r = s.next_request().expect("should have request");
        assert_eq!(r.block_cid, "rt");
    }

    // -- FIFO within same priority ------------------------------------------

    #[test]
    fn test_fifo_within_same_priority() {
        let mut s = default_scheduler();
        let _ = s.submit("first", IODirection::Read, IOPriority::Normal, 1, None);
        let _ = s.submit("second", IODirection::Read, IOPriority::Normal, 1, None);
        let _ = s.submit("third", IODirection::Read, IOPriority::Normal, 1, None);

        assert_eq!(s.next_request().expect("r").block_cid, "first");
        assert_eq!(s.next_request().expect("r").block_cid, "second");
        assert_eq!(s.next_request().expect("r").block_cid, "third");
    }

    // -- deadline urgency ---------------------------------------------------

    #[test]
    fn test_deadline_urgent_dispatched_first() {
        let mut s = default_scheduler();
        // Advance tick to 5.
        for _ in 0..5 {
            s.tick();
        }
        // Submit a high-priority request with no deadline.
        let _ = s.submit("high", IODirection::Read, IOPriority::High, 1, None);
        // Submit a low-priority request whose deadline already passed.
        let _ = s.submit(
            "urgent",
            IODirection::Read,
            IOPriority::Background,
            1,
            Some(3),
        );

        let r = s.next_request().expect("r");
        assert_eq!(r.block_cid, "urgent");
    }

    #[test]
    fn test_deadline_urgent_earliest_first() {
        let mut s = default_scheduler();
        for _ in 0..10 {
            s.tick();
        }
        let _ = s.submit("dl5", IODirection::Read, IOPriority::Normal, 1, Some(5));
        let _ = s.submit("dl3", IODirection::Read, IOPriority::Normal, 1, Some(3));
        let _ = s.submit("dl7", IODirection::Read, IOPriority::Normal, 1, Some(7));

        assert_eq!(s.next_request().expect("r").block_cid, "dl3");
        assert_eq!(s.next_request().expect("r").block_cid, "dl5");
        assert_eq!(s.next_request().expect("r").block_cid, "dl7");
    }

    #[test]
    fn test_deadline_at_current_tick_is_urgent() {
        let mut s = default_scheduler();
        for _ in 0..5 {
            s.tick();
        }
        // deadline == current_tick (5) => urgent
        let _ = s.submit(
            "edge",
            IODirection::Read,
            IOPriority::Background,
            1,
            Some(5),
        );
        let _ = s.submit("nope", IODirection::Read, IOPriority::Realtime, 1, None);

        assert_eq!(s.next_request().expect("r").block_cid, "edge");
    }

    // -- cancel -------------------------------------------------------------

    #[test]
    fn test_cancel_existing() {
        let mut s = default_scheduler();
        let id = s
            .submit("c", IODirection::Read, IOPriority::Normal, 1, None)
            .expect("ok");
        assert_eq!(s.pending_count(), 1);
        assert!(s.cancel(id));
        assert_eq!(s.pending_count(), 0);
    }

    #[test]
    fn test_cancel_nonexistent() {
        let mut s = default_scheduler();
        assert!(!s.cancel(999));
    }

    #[test]
    fn test_cancel_double() {
        let mut s = default_scheduler();
        let id = s
            .submit("c", IODirection::Read, IOPriority::Normal, 1, None)
            .expect("ok");
        assert!(s.cancel(id));
        assert!(!s.cancel(id));
    }

    // -- complete / counting ------------------------------------------------

    #[test]
    fn test_complete_read_counting() {
        let mut s = default_scheduler();
        s.complete(0, IODirection::Read, 1024);
        s.complete(1, IODirection::Read, 2048);
        let st = s.stats();
        assert_eq!(st.completed_reads, 2);
        assert_eq!(st.completed_writes, 0);
        assert_eq!(st.total_bytes_scheduled, 3072);
    }

    #[test]
    fn test_complete_write_counting() {
        let mut s = default_scheduler();
        s.complete(0, IODirection::Write, 512);
        let st = s.stats();
        assert_eq!(st.completed_writes, 1);
        assert_eq!(st.completed_reads, 0);
        assert_eq!(st.total_bytes_scheduled, 512);
    }

    #[test]
    fn test_complete_mixed() {
        let mut s = default_scheduler();
        s.complete(0, IODirection::Read, 100);
        s.complete(1, IODirection::Write, 200);
        s.complete(2, IODirection::Read, 300);
        let st = s.stats();
        assert_eq!(st.completed_reads, 2);
        assert_eq!(st.completed_writes, 1);
        assert_eq!(st.total_bytes_scheduled, 600);
    }

    // -- pending reads / writes ---------------------------------------------

    #[test]
    fn test_pending_reads_writes() {
        let mut s = default_scheduler();
        let _ = s.submit("r1", IODirection::Read, IOPriority::Normal, 1, None);
        let _ = s.submit("r2", IODirection::Read, IOPriority::High, 1, None);
        let _ = s.submit("w1", IODirection::Write, IOPriority::Normal, 1, None);

        assert_eq!(s.pending_reads(), 2);
        assert_eq!(s.pending_writes(), 1);
        assert_eq!(s.pending_count(), 3);
    }

    // -- tick ---------------------------------------------------------------

    #[test]
    fn test_tick_advances_clock() {
        let mut s = default_scheduler();
        assert_eq!(s.current_tick(), 0);
        s.tick();
        assert_eq!(s.current_tick(), 1);
        for _ in 0..10 {
            s.tick();
        }
        assert_eq!(s.current_tick(), 11);
    }

    // -- expired_requests ---------------------------------------------------

    #[test]
    fn test_expired_requests_none_initially() {
        let mut s = default_scheduler();
        let _ = s.submit("a", IODirection::Read, IOPriority::Normal, 1, Some(5));
        assert!(s.expired_requests().is_empty());
    }

    #[test]
    fn test_expired_requests_after_ticks() {
        let mut s = default_scheduler();
        let _ = s.submit("a", IODirection::Read, IOPriority::Normal, 1, Some(2));
        let _ = s.submit("b", IODirection::Read, IOPriority::Normal, 1, Some(5));
        let _ = s.submit("c", IODirection::Read, IOPriority::Normal, 1, None);

        // Advance to tick 3 — deadline=2 is expired, deadline=5 is not.
        for _ in 0..3 {
            s.tick();
        }
        let expired = s.expired_requests();
        assert_eq!(expired.len(), 1);
        assert_eq!(expired[0].block_cid, "a");
    }

    #[test]
    fn test_expired_at_exact_tick_not_expired() {
        // deadline_tick < current_tick means expired. Equal is NOT expired.
        let mut s = default_scheduler();
        let _ = s.submit("a", IODirection::Read, IOPriority::Normal, 1, Some(3));
        for _ in 0..3 {
            s.tick();
        }
        // current_tick == 3, deadline == 3 => not expired (still serviceable).
        assert!(s.expired_requests().is_empty());
    }

    // -- drain_expired ------------------------------------------------------

    #[test]
    fn test_drain_expired() {
        let mut s = default_scheduler();
        let _ = s.submit("a", IODirection::Read, IOPriority::Normal, 1, Some(1));
        let _ = s.submit("b", IODirection::Read, IOPriority::Normal, 1, Some(2));
        let _ = s.submit("c", IODirection::Read, IOPriority::Normal, 1, Some(10));
        let _ = s.submit("d", IODirection::Read, IOPriority::Normal, 1, None);

        for _ in 0..5 {
            s.tick();
        }

        let drained = s.drain_expired();
        assert_eq!(drained.len(), 2);
        assert_eq!(s.pending_count(), 2); // c (deadline 10) + d (no deadline)
    }

    #[test]
    fn test_drain_expired_idempotent() {
        let mut s = default_scheduler();
        let _ = s.submit("a", IODirection::Read, IOPriority::Normal, 1, Some(1));
        for _ in 0..3 {
            s.tick();
        }
        let first = s.drain_expired();
        assert_eq!(first.len(), 1);
        let second = s.drain_expired();
        assert!(second.is_empty());
    }

    // -- stats --------------------------------------------------------------

    #[test]
    fn test_stats_accuracy() {
        let mut s = default_scheduler();
        let _ = s.submit("a", IODirection::Read, IOPriority::Normal, 100, Some(1));
        let _ = s.submit("b", IODirection::Write, IOPriority::High, 200, None);
        s.complete(10, IODirection::Read, 50);

        for _ in 0..3 {
            s.tick();
        }

        let st = s.stats();
        assert_eq!(st.pending_count, 2);
        assert_eq!(st.completed_reads, 1);
        assert_eq!(st.completed_writes, 0);
        assert_eq!(st.total_bytes_scheduled, 50);
        assert_eq!(st.expired_count, 1); // "a" expired (deadline 1, tick 3)
    }

    // -- mixed read/write ---------------------------------------------------

    #[test]
    fn test_mixed_read_write_ordering() {
        let mut s = default_scheduler();
        let _ = s.submit("w", IODirection::Write, IOPriority::High, 1, None);
        let _ = s.submit("r", IODirection::Read, IOPriority::Normal, 1, None);

        // High > Normal regardless of direction.
        assert_eq!(s.next_request().expect("r").block_cid, "w");
        assert_eq!(s.next_request().expect("r").block_cid, "r");
    }

    // -- empty scheduler ----------------------------------------------------

    #[test]
    fn test_next_request_empty() {
        let mut s = default_scheduler();
        assert!(s.next_request().is_none());
    }

    // -- submit after cancel frees slot -------------------------------------

    #[test]
    fn test_submit_after_cancel() {
        let mut s = small_scheduler(1);
        let id = s
            .submit("a", IODirection::Read, IOPriority::Normal, 1, None)
            .expect("ok");
        assert!(s
            .submit("b", IODirection::Read, IOPriority::Normal, 1, None)
            .is_err());
        s.cancel(id);
        assert!(s
            .submit("c", IODirection::Read, IOPriority::Normal, 1, None)
            .is_ok());
    }

    // -- large batch --------------------------------------------------------

    #[test]
    fn test_large_batch_submit_and_drain() {
        let mut s = default_scheduler();
        for i in 0..100 {
            let _ = s.submit(
                &format!("blk{i}"),
                IODirection::Read,
                IOPriority::Normal,
                64,
                Some(50),
            );
        }
        assert_eq!(s.pending_count(), 100);

        for _ in 0..60 {
            s.tick();
        }
        let drained = s.drain_expired();
        assert_eq!(drained.len(), 100);
        assert_eq!(s.pending_count(), 0);
    }

    // -- config accessor ----------------------------------------------------

    #[test]
    fn test_config_accessor() {
        let s = default_scheduler();
        assert_eq!(s.config().max_pending, 1000);
        assert!((s.config().read_weight - 0.7).abs() < 1e-9);
        assert!((s.config().write_weight - 0.3).abs() < 1e-9);
    }

    // -- default config -----------------------------------------------------

    #[test]
    fn test_default_config() {
        let cfg = SchedulerConfig::default();
        assert_eq!(cfg.max_pending, 1000);
        assert!((cfg.read_weight - 0.7).abs() < 1e-9);
        assert!((cfg.write_weight - 0.3).abs() < 1e-9);
    }

    // -- complete removes from pending if present ---------------------------

    #[test]
    fn test_complete_removes_from_pending() {
        let mut s = default_scheduler();
        let id = s
            .submit("x", IODirection::Read, IOPriority::Normal, 256, None)
            .expect("ok");
        assert_eq!(s.pending_count(), 1);
        s.complete(id, IODirection::Read, 256);
        assert_eq!(s.pending_count(), 0);
        assert_eq!(s.stats().completed_reads, 1);
    }

    // -- deadline fifo tiebreak ---------------------------------------------

    #[test]
    fn test_deadline_urgent_fifo_tiebreak() {
        let mut s = default_scheduler();
        // Both have same deadline and same priority.
        let _ = s.submit("first", IODirection::Read, IOPriority::Normal, 1, Some(2));
        let _ = s.submit("second", IODirection::Read, IOPriority::Normal, 1, Some(2));

        for _ in 0..3 {
            s.tick();
        }
        assert_eq!(s.next_request().expect("r").block_cid, "first");
        assert_eq!(s.next_request().expect("r").block_cid, "second");
    }

    // -- multiple priorities with deadlines ---------------------------------

    #[test]
    fn test_non_urgent_deadline_does_not_promote() {
        let mut s = default_scheduler();
        // deadline in the future — not urgent yet.
        let _ = s.submit(
            "dl_future",
            IODirection::Read,
            IOPriority::Background,
            1,
            Some(100),
        );
        let _ = s.submit("high", IODirection::Read, IOPriority::High, 1, None);

        // tick=0, deadline=100 is far away — "high" should come first.
        assert_eq!(s.next_request().expect("r").block_cid, "high");
    }
}
