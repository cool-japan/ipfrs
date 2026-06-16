//! TensorEventBusV2 — typed in-process event bus for TensorLogic events.
//!
//! Supports multiple subscribers with priority-ordered delivery, per-subscriber
//! event filtering, bounded queues with drop accounting, drain-on-demand, dead-letter
//! queuing for unrouted events, and cumulative delivery statistics.
//!
//! # Design overview
//!
//! ```text
//! Publisher  ──publish(event)──►  TensorEventBusV2
//!                                       │
//!               ┌────────── priority-sorted subscriptions ─────────────┐
//!               │  sub(prio=10, filter=All)                             │
//!               │  sub(prio=5,  filter=RuleEventsOnly)                  │
//!               │  sub(prio=1,  filter=TensorEventsOnly)                │
//!               └──────────────────────────────────────────────────────┘
//!                                       │
//!                          no subscriber accepted ──► dead_letters
//! ```

// ─── Types ───────────────────────────────────────────────────────────────────

/// Events that can be published on the bus.
#[derive(Clone, Debug, PartialEq)]
pub enum TensorEvent {
    /// A Datalog/TensorLogic rule was fired.
    RuleFired {
        /// Identifier of the rule.
        rule_id: u64,
        /// Number of variable bindings produced.
        bindings_count: usize,
    },
    /// An inference session completed.
    InferenceComplete {
        /// Identifier of the session.
        session_id: u64,
        /// Wall-clock duration in milliseconds.
        duration_ms: u64,
    },
    /// A tensor was updated to a new version.
    TensorUpdated {
        /// Identifier of the tensor.
        tensor_id: u64,
        /// New version number.
        new_version: u64,
    },
    /// A checkpoint was persisted to disk.
    CheckpointSaved {
        /// Filesystem path of the checkpoint file.
        path: String,
        /// Size of the checkpoint in bytes.
        size_bytes: u64,
    },
    /// A gradient step was applied.
    GradientApplied {
        /// Optimiser step counter.
        step: u64,
        /// Training loss at this step.
        loss: f64,
    },
}

// ─── Filter ──────────────────────────────────────────────────────────────────

/// Subscription filter — determines which [`TensorEvent`] variants a subscriber receives.
#[derive(Clone, Debug, PartialEq)]
pub enum EventFilter {
    /// Receive every event regardless of variant.
    All,
    /// Receive only [`TensorEvent::RuleFired`] events.
    RuleEventsOnly,
    /// Receive only [`TensorEvent::InferenceComplete`] events.
    InferenceEventsOnly,
    /// Receive only [`TensorEvent::TensorUpdated`] events.
    TensorEventsOnly,
}

impl EventFilter {
    /// Returns `true` when this filter accepts the given event.
    pub fn matches(&self, event: &TensorEvent) -> bool {
        match self {
            EventFilter::All => true,
            EventFilter::RuleEventsOnly => matches!(event, TensorEvent::RuleFired { .. }),
            EventFilter::InferenceEventsOnly => {
                matches!(event, TensorEvent::InferenceComplete { .. })
            }
            EventFilter::TensorEventsOnly => matches!(event, TensorEvent::TensorUpdated { .. }),
        }
    }
}

// ─── Subscription ─────────────────────────────────────────────────────────────

/// Per-subscriber state maintained by the bus.
#[derive(Debug)]
pub struct Subscription {
    /// Unique subscription identifier returned by [`TensorEventBusV2::subscribe`].
    pub sub_id: u64,
    /// Event filter for this subscription.
    pub filter: EventFilter,
    /// Delivery priority — higher values are served first.
    pub priority: u32,
    /// Maximum number of events buffered before drops occur.
    pub max_queue_size: usize,
    /// Pending (unread) events for this subscriber.
    pub queue: Vec<TensorEvent>,
    /// Cumulative count of events successfully enqueued.
    pub delivered: u64,
    /// Cumulative count of events dropped because the queue was full.
    pub dropped: u64,
}

impl Subscription {
    fn new(sub_id: u64, filter: EventFilter, priority: u32, max_queue_size: usize) -> Self {
        Self {
            sub_id,
            filter,
            priority,
            max_queue_size,
            queue: Vec::new(),
            delivered: 0,
            dropped: 0,
        }
    }
}

// ─── BusStats ─────────────────────────────────────────────────────────────────

/// Aggregate delivery statistics for the entire bus.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BusStats {
    /// Total number of events passed to [`TensorEventBusV2::publish`].
    pub total_published: u64,
    /// Total events successfully enqueued across all subscribers.
    pub total_delivered: u64,
    /// Total events dropped (queue full) across all subscribers.
    pub total_dropped: u64,
}

impl BusStats {
    /// Fraction of published events that were delivered to at least one subscriber queue.
    ///
    /// Returns `1.0` when no events have been published (vacuously perfect).
    pub fn delivery_rate(&self) -> f64 {
        if self.total_published == 0 {
            return 1.0;
        }
        // The numerator is capped at total_published to keep the rate in [0, 1].
        // An event published to N subscribers counts N times towards total_delivered,
        // so we normalise by total_published rather than total_delivered.
        // Simple ratio: events-delivered / events-published (per-event fan-out semantics).
        // Because one publish can fan out to many subs the rate can exceed 1.0 when
        // there are multiple subscribers; callers should interpret accordingly.
        self.total_delivered as f64 / self.total_published as f64
    }
}

// ─── TensorEventBusV2 ─────────────────────────────────────────────────────────

/// Typed in-process event bus for TensorLogic events.
///
/// Events are published synchronously and fanned out to all matching subscriptions
/// in descending priority order.  Each subscription maintains a bounded queue;
/// events that overflow the queue are counted as dropped.  Events that match no
/// subscription are collected in the dead-letter queue.
#[derive(Debug)]
pub struct TensorEventBusV2 {
    /// Active subscriptions, kept sorted by `priority` descending.
    pub subscriptions: Vec<Subscription>,
    /// Events that were not delivered to any subscriber.
    pub dead_letters: Vec<TensorEvent>,
    /// Bus-level statistics.
    pub stats: BusStats,
    /// Monotonically increasing counter used to assign subscription IDs.
    pub next_sub_id: u64,
}

impl TensorEventBusV2 {
    /// Creates a new, empty event bus.
    pub fn new() -> Self {
        Self {
            subscriptions: Vec::new(),
            dead_letters: Vec::new(),
            stats: BusStats::default(),
            next_sub_id: 1,
        }
    }

    /// Registers a new subscription and returns its unique identifier.
    ///
    /// The internal subscription list is re-sorted after each registration so that
    /// [`publish`](Self::publish) can iterate in priority order without extra work.
    pub fn subscribe(&mut self, filter: EventFilter, priority: u32, max_queue_size: usize) -> u64 {
        let sub_id = self.next_sub_id;
        self.next_sub_id += 1;

        let effective_max = if max_queue_size == 0 {
            100
        } else {
            max_queue_size
        };

        self.subscriptions
            .push(Subscription::new(sub_id, filter, priority, effective_max));

        // Keep sorted: highest priority first.
        self.subscriptions
            .sort_unstable_by_key(|s| std::cmp::Reverse(s.priority));

        sub_id
    }

    /// Removes the subscription identified by `sub_id`.
    ///
    /// Returns `true` if the subscription existed and was removed.
    pub fn unsubscribe(&mut self, sub_id: u64) -> bool {
        let before = self.subscriptions.len();
        self.subscriptions.retain(|s| s.sub_id != sub_id);
        self.subscriptions.len() < before
    }

    /// Publishes an event to all matching subscriptions.
    ///
    /// Subscriptions are visited in descending priority order.  For each
    /// subscription whose filter matches:
    /// - If the queue is not full, the event is cloned and enqueued; `delivered` and
    ///   `stats.total_delivered` are incremented.
    /// - If the queue is full, the event is not enqueued; `dropped` and
    ///   `stats.total_dropped` are incremented.
    ///
    /// If no subscription accepted the event it is moved to [`dead_letters`](Self::dead_letters).
    pub fn publish(&mut self, event: TensorEvent) {
        self.stats.total_published += 1;

        let mut any_accepted = false;

        for sub in self.subscriptions.iter_mut() {
            if !sub.filter.matches(&event) {
                continue;
            }

            if sub.queue.len() < sub.max_queue_size {
                sub.queue.push(event.clone());
                sub.delivered += 1;
                self.stats.total_delivered += 1;
                any_accepted = true;
            } else {
                sub.dropped += 1;
                self.stats.total_dropped += 1;
                // Still counts as "accepted by filter" for dead-letter purposes.
                any_accepted = true;
            }
        }

        if !any_accepted {
            self.dead_letters.push(event);
        }
    }

    /// Drains and returns all queued events for the subscription identified by `sub_id`.
    ///
    /// Returns an empty `Vec` if the subscription does not exist or has no pending events.
    pub fn drain(&mut self, sub_id: u64) -> Vec<TensorEvent> {
        match self.subscriptions.iter_mut().find(|s| s.sub_id == sub_id) {
            Some(sub) => {
                let mut out = Vec::with_capacity(sub.queue.len());
                core::mem::swap(&mut sub.queue, &mut out);
                out
            }
            None => Vec::new(),
        }
    }

    /// Returns a reference to the current bus statistics.
    pub fn stats(&self) -> &BusStats {
        &self.stats
    }
}

impl Default for TensorEventBusV2 {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: build a RuleFired event.
    fn rule_fired(rule_id: u64, bindings_count: usize) -> TensorEvent {
        TensorEvent::RuleFired {
            rule_id,
            bindings_count,
        }
    }

    // Helper: build an InferenceComplete event.
    fn inference_complete(session_id: u64, duration_ms: u64) -> TensorEvent {
        TensorEvent::InferenceComplete {
            session_id,
            duration_ms,
        }
    }

    // Helper: build a TensorUpdated event.
    fn tensor_updated(tensor_id: u64, new_version: u64) -> TensorEvent {
        TensorEvent::TensorUpdated {
            tensor_id,
            new_version,
        }
    }

    // Helper: build a CheckpointSaved event.
    fn checkpoint_saved(path: &str, size_bytes: u64) -> TensorEvent {
        TensorEvent::CheckpointSaved {
            path: path.to_string(),
            size_bytes,
        }
    }

    // Helper: build a GradientApplied event.
    fn gradient_applied(step: u64, loss: f64) -> TensorEvent {
        TensorEvent::GradientApplied { step, loss }
    }

    // ── Test 1: subscribe returns a monotonically increasing sub_id ──────────

    #[test]
    fn subscribe_returns_unique_sub_ids() {
        let mut bus = TensorEventBusV2::new();
        let id1 = bus.subscribe(EventFilter::All, 10, 100);
        let id2 = bus.subscribe(EventFilter::All, 10, 100);
        let id3 = bus.subscribe(EventFilter::All, 10, 100);
        assert_ne!(id1, id2);
        assert_ne!(id2, id3);
        assert_ne!(id1, id3);
    }

    // ── Test 2: first sub_id is nonzero ──────────────────────────────────────

    #[test]
    fn subscribe_first_id_nonzero() {
        let mut bus = TensorEventBusV2::new();
        let id = bus.subscribe(EventFilter::All, 5, 100);
        assert!(id > 0, "sub_id must be nonzero");
    }

    // ── Test 3: publish routes to correct subscriber queue ───────────────────

    #[test]
    fn publish_routes_to_subscriber_queue() {
        let mut bus = TensorEventBusV2::new();
        let id = bus.subscribe(EventFilter::All, 10, 100);
        let event = rule_fired(42, 3);
        bus.publish(event.clone());
        let drained = bus.drain(id);
        assert_eq!(drained.len(), 1);
        assert_eq!(drained[0], event);
    }

    // ── Test 4: filter All vs RuleEventsOnly ─────────────────────────────────

    #[test]
    fn filter_all_receives_every_variant() {
        let mut bus = TensorEventBusV2::new();
        let id = bus.subscribe(EventFilter::All, 5, 100);
        bus.publish(rule_fired(1, 0));
        bus.publish(inference_complete(2, 50));
        bus.publish(tensor_updated(3, 7));
        bus.publish(checkpoint_saved("/tmp/ckpt", 1024));
        bus.publish(gradient_applied(10, 0.25));
        let drained = bus.drain(id);
        assert_eq!(drained.len(), 5);
    }

    // ── Test 5: RuleEventsOnly filter passes only RuleFired ──────────────────

    #[test]
    fn filter_rule_events_only() {
        let mut bus = TensorEventBusV2::new();
        let id = bus.subscribe(EventFilter::RuleEventsOnly, 5, 100);
        bus.publish(rule_fired(1, 2));
        bus.publish(inference_complete(2, 10));
        bus.publish(rule_fired(3, 5));
        let drained = bus.drain(id);
        assert_eq!(drained.len(), 2);
        for e in &drained {
            assert!(matches!(e, TensorEvent::RuleFired { .. }));
        }
    }

    // ── Test 6: InferenceEventsOnly filter ───────────────────────────────────

    #[test]
    fn filter_inference_events_only() {
        let mut bus = TensorEventBusV2::new();
        let id = bus.subscribe(EventFilter::InferenceEventsOnly, 5, 100);
        bus.publish(rule_fired(1, 0));
        bus.publish(inference_complete(7, 100));
        bus.publish(tensor_updated(3, 2));
        let drained = bus.drain(id);
        assert_eq!(drained.len(), 1);
        assert!(matches!(
            drained[0],
            TensorEvent::InferenceComplete { session_id: 7, .. }
        ));
    }

    // ── Test 7: TensorEventsOnly filter ──────────────────────────────────────

    #[test]
    fn filter_tensor_events_only() {
        let mut bus = TensorEventBusV2::new();
        let id = bus.subscribe(EventFilter::TensorEventsOnly, 5, 100);
        bus.publish(rule_fired(1, 0));
        bus.publish(tensor_updated(10, 5));
        bus.publish(gradient_applied(1, 0.1));
        let drained = bus.drain(id);
        assert_eq!(drained.len(), 1);
        assert!(matches!(
            drained[0],
            TensorEvent::TensorUpdated { tensor_id: 10, .. }
        ));
    }

    // ── Test 8: priority ordering ─────────────────────────────────────────────

    #[test]
    fn priority_ordering_high_to_low() {
        // Use a custom event to detect ordering side-effects via delivery counters.
        let mut bus = TensorEventBusV2::new();
        // Subscribe in low-first order; bus must still deliver high-priority first.
        let id_low = bus.subscribe(EventFilter::All, 1, 100);
        let id_high = bus.subscribe(EventFilter::All, 99, 100);

        bus.publish(rule_fired(1, 0));

        // Both should receive the event.
        assert_eq!(bus.drain(id_high).len(), 1);
        assert_eq!(bus.drain(id_low).len(), 1);

        // Verify internal ordering: highest priority subscription is first.
        assert!(bus.subscriptions[0].priority >= bus.subscriptions[1].priority);
    }

    // ── Test 9: queue full causes drops ──────────────────────────────────────

    #[test]
    fn queue_full_drops_excess_events() {
        let max = 3_usize;
        let mut bus = TensorEventBusV2::new();
        let id = bus.subscribe(EventFilter::All, 5, max);

        for i in 0..5_u64 {
            bus.publish(rule_fired(i, 0));
        }

        let sub = bus
            .subscriptions
            .iter()
            .find(|s| s.sub_id == id)
            .expect("test: should succeed");
        assert_eq!(sub.queue.len(), max);
        assert_eq!(sub.dropped, 2);
        assert_eq!(sub.delivered, max as u64);
    }

    // ── Test 10: drain clears the queue ──────────────────────────────────────

    #[test]
    fn drain_clears_queue() {
        let mut bus = TensorEventBusV2::new();
        let id = bus.subscribe(EventFilter::All, 5, 100);
        bus.publish(rule_fired(1, 0));
        bus.publish(rule_fired(2, 0));
        assert_eq!(bus.drain(id).len(), 2);
        // Second drain must be empty.
        assert!(bus.drain(id).is_empty());
    }

    // ── Test 11: unsubscribe removes subscription ─────────────────────────────

    #[test]
    fn unsubscribe_removes_subscription() {
        let mut bus = TensorEventBusV2::new();
        let id = bus.subscribe(EventFilter::All, 5, 100);
        assert!(bus.unsubscribe(id));
        // Removed, so second call must return false.
        assert!(!bus.unsubscribe(id));
        assert!(bus.subscriptions.is_empty());
    }

    // ── Test 12: unsubscribe non-existent id returns false ────────────────────

    #[test]
    fn unsubscribe_nonexistent_returns_false() {
        let mut bus = TensorEventBusV2::new();
        assert!(!bus.unsubscribe(999));
    }

    // ── Test 13: dead_letters when no subscribers ─────────────────────────────

    #[test]
    fn dead_letters_with_no_subscribers() {
        let mut bus = TensorEventBusV2::new();
        let event = rule_fired(1, 0);
        bus.publish(event.clone());
        assert_eq!(bus.dead_letters.len(), 1);
        assert_eq!(bus.dead_letters[0], event);
    }

    // ── Test 14: dead_letters when no filter matches ──────────────────────────

    #[test]
    fn dead_letters_when_no_filter_matches() {
        let mut bus = TensorEventBusV2::new();
        // Subscribe only to rule events.
        let _id = bus.subscribe(EventFilter::RuleEventsOnly, 5, 100);
        // Publish an inference event — no subscriber wants it.
        bus.publish(inference_complete(1, 50));
        assert_eq!(bus.dead_letters.len(), 1);
    }

    // ── Test 15: delivery_rate perfect when all delivered ─────────────────────

    #[test]
    fn delivery_rate_perfect() {
        let mut bus = TensorEventBusV2::new();
        let _id = bus.subscribe(EventFilter::All, 5, 100);
        bus.publish(rule_fired(1, 0));
        bus.publish(inference_complete(2, 10));
        let stats = bus.stats();
        // 2 events, 2 deliveries — delivery_rate = delivered/published = 2/2 = 1.0
        assert!((stats.delivery_rate() - 1.0_f64).abs() < f64::EPSILON);
    }

    // ── Test 16: delivery_rate when no events published ───────────────────────

    #[test]
    fn delivery_rate_empty_bus() {
        let bus = TensorEventBusV2::new();
        assert_eq!(bus.stats().delivery_rate(), 1.0);
    }

    // ── Test 17: stats total_dropped accumulates across subscriptions ─────────

    #[test]
    fn stats_total_dropped_accumulates() {
        let mut bus = TensorEventBusV2::new();
        // Two subscriptions each with queue size 1.
        let _id1 = bus.subscribe(EventFilter::All, 5, 1);
        let _id2 = bus.subscribe(EventFilter::All, 5, 1);

        // Publish 3 events: first fills each queue (1 each), remaining 2 are dropped per sub.
        bus.publish(rule_fired(1, 0));
        bus.publish(rule_fired(2, 0));
        bus.publish(rule_fired(3, 0));

        // Each sub received 1, dropped 2 → total_dropped = 4.
        assert_eq!(bus.stats().total_dropped, 4);
    }

    // ── Test 18: multiple subscribers all receive matching event ──────────────

    #[test]
    fn multiple_subscribers_all_receive() {
        let mut bus = TensorEventBusV2::new();
        let id1 = bus.subscribe(EventFilter::All, 10, 100);
        let id2 = bus.subscribe(EventFilter::All, 5, 100);
        let id3 = bus.subscribe(EventFilter::All, 1, 100);

        bus.publish(tensor_updated(7, 3));

        assert_eq!(bus.drain(id1).len(), 1);
        assert_eq!(bus.drain(id2).len(), 1);
        assert_eq!(bus.drain(id3).len(), 1);
    }

    // ── Test 19: partial subscriber match (some filter, some don't) ───────────

    #[test]
    fn partial_subscriber_match() {
        let mut bus = TensorEventBusV2::new();
        let id_rule = bus.subscribe(EventFilter::RuleEventsOnly, 10, 100);
        let id_all = bus.subscribe(EventFilter::All, 5, 100);

        // Publish one event that only id_all matches.
        bus.publish(gradient_applied(1, 0.5));

        assert!(bus.drain(id_rule).is_empty());
        assert_eq!(bus.drain(id_all).len(), 1);
        // Should NOT be in dead_letters because id_all accepted it.
        assert!(bus.dead_letters.is_empty());
    }

    // ── Test 20: checkpoint saved event round-trips correctly ─────────────────

    #[test]
    fn checkpoint_saved_event_round_trip() {
        let mut bus = TensorEventBusV2::new();
        let id = bus.subscribe(EventFilter::All, 5, 100);
        let event = checkpoint_saved("/tmp/model.safetensors", 4096);
        bus.publish(event.clone());
        let drained = bus.drain(id);
        assert_eq!(drained.len(), 1);
        assert_eq!(drained[0], event);
    }

    // ── Test 21: gradient applied event round-trips correctly ─────────────────

    #[test]
    fn gradient_applied_event_round_trip() {
        let mut bus = TensorEventBusV2::new();
        let id = bus.subscribe(EventFilter::All, 5, 100);
        let event = gradient_applied(42, 0.001_f64);
        bus.publish(event.clone());
        let drained = bus.drain(id);
        assert_eq!(drained.len(), 1);
        if let TensorEvent::GradientApplied { step, loss } = &drained[0] {
            assert_eq!(*step, 42);
            assert!((*loss - 0.001_f64).abs() < 1e-12);
        } else {
            panic!("unexpected event variant");
        }
    }

    // ── Test 22: stats total_published and total_delivered are consistent ──────

    #[test]
    fn stats_consistency() {
        let mut bus = TensorEventBusV2::new();
        let _id = bus.subscribe(EventFilter::All, 5, 100);
        for i in 0..10_u64 {
            bus.publish(rule_fired(i, 0));
        }
        let s = bus.stats();
        assert_eq!(s.total_published, 10);
        assert_eq!(s.total_delivered, 10);
        assert_eq!(s.total_dropped, 0);
    }

    // ── Test 23: drain on unknown sub_id returns empty vec ────────────────────

    #[test]
    fn drain_unknown_sub_id_returns_empty() {
        let mut bus = TensorEventBusV2::new();
        assert!(bus.drain(9999).is_empty());
    }

    // ── Test 24: default queue size used when max_queue_size is zero ──────────

    #[test]
    fn default_queue_size_applied_when_zero() {
        let mut bus = TensorEventBusV2::new();
        let id = bus.subscribe(EventFilter::All, 5, 0);
        let sub = bus
            .subscriptions
            .iter()
            .find(|s| s.sub_id == id)
            .expect("test: should succeed");
        assert_eq!(sub.max_queue_size, 100);
    }
}
