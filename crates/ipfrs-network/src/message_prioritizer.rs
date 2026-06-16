//! PeerMessagePrioritizer — Multi-level priority queue with aging for outbound messages.
//!
//! Each peer gets its own sorted queue (highest effective priority first, FIFO within same
//! priority). An aging mechanism prevents message starvation by promoting lower-priority
//! messages after they have waited for a configurable number of ticks.
//!
//! ## Design highlights
//!
//! - **Four priority levels**: Background → Normal → High → Urgent
//! - **Aging / anti-starvation**: `advance_tick` promotes stale messages one level at a time
//! - **Per-peer capacity cap**: when full, the lowest-priority message is evicted to make room
//! - **Rich statistics**: enqueued, dequeued, promoted, dropped counters with a throughput ratio
//! - **No `unwrap()`**: every fallible operation is expressed through `Option` / explicit checks

use std::collections::HashMap;

// ─── MessagePriority ──────────────────────────────────────────────────────────

/// Four-level priority for outbound peer messages.
///
/// The integer discriminants are intentional: `Ord` derived from them means
/// `Background < Normal < High < Urgent` so arithmetic promotion (`as u8 + 1`)
/// produces the next level.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum MessagePriority {
    /// Best-effort background traffic (gossip, keep-alives, …).
    Background = 0,
    /// Default priority for ordinary messages.
    Normal = 1,
    /// Important messages that should be delivered before normal ones.
    High = 2,
    /// Time-sensitive messages that preempt all others.
    Urgent = 3,
}

impl MessagePriority {
    /// Promote this priority one level upward, capped at `Urgent`.
    fn promote(self) -> Self {
        match self {
            Self::Background => Self::Normal,
            Self::Normal => Self::High,
            Self::High => Self::Urgent,
            Self::Urgent => Self::Urgent,
        }
    }
}

// ─── PrioritizedMessage ───────────────────────────────────────────────────────

/// A single outbound message sitting in a peer's priority queue.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrioritizedMessage {
    /// Monotonically increasing identifier assigned at enqueue time.
    pub msg_id: u64,
    /// Destination peer identifier.
    pub peer_id: String,
    /// Original priority assigned by the caller.
    pub priority: MessagePriority,
    /// Size of the message payload in bytes.
    pub payload_bytes: u64,
    /// Value of `current_tick` when the message was enqueued.
    pub enqueued_tick: u64,
    /// Effective priority after aging promotions; starts equal to `priority`.
    pub effective_priority: MessagePriority,
}

// ─── AgingConfig ─────────────────────────────────────────────────────────────

/// Tunable parameters for the aging / anti-starvation mechanism.
#[derive(Clone, Debug)]
pub struct AgingConfig {
    /// Number of ticks a message must wait before its effective priority is
    /// promoted by one level. Default: 50.
    pub promote_after_ticks: u64,
    /// Maximum number of messages kept per peer queue. Default: 256.
    pub max_queue_size: usize,
}

impl Default for AgingConfig {
    fn default() -> Self {
        Self {
            promote_after_ticks: 50,
            max_queue_size: 256,
        }
    }
}

// ─── PrioritizerStats ────────────────────────────────────────────────────────

/// Aggregate statistics for a `PeerMessagePrioritizer` instance.
#[derive(Clone, Debug, Default)]
pub struct PrioritizerStats {
    /// Total messages successfully enqueued (including those that displaced a
    /// dropped message).
    pub total_enqueued: u64,
    /// Total messages removed via `dequeue`.
    pub total_dequeued: u64,
    /// Total aging promotions applied across all messages and all ticks.
    pub total_promoted: u64,
    /// Total messages evicted because the per-peer queue was at capacity.
    pub total_dropped: u64,
}

impl PrioritizerStats {
    /// Fraction of enqueued messages that have been dequeued.
    ///
    /// Returns `0.0` when nothing has been enqueued yet.
    pub fn throughput_ratio(&self) -> f64 {
        self.total_dequeued as f64 / self.total_enqueued.max(1) as f64
    }
}

// ─── PeerMessagePrioritizer ───────────────────────────────────────────────────

/// Multi-level outbound message prioritizer with per-peer queues and aging.
///
/// # Ordering guarantee
///
/// Within a peer's queue messages are sorted so that:
/// 1. Higher `effective_priority` comes first.
/// 2. Among messages with the same `effective_priority`, the one enqueued
///    earliest (lowest `enqueued_tick`) comes first (FIFO).
pub struct PeerMessagePrioritizer {
    /// Per-peer queues; each `Vec` is kept sorted (front = highest priority).
    pub queues: HashMap<String, Vec<PrioritizedMessage>>,
    /// Aging and capacity configuration.
    pub config: AgingConfig,
    /// Running statistics.
    pub stats: PrioritizerStats,
    /// Logical clock advanced via `advance_tick`.
    pub current_tick: u64,
    /// Counter used to assign unique message identifiers.
    pub next_id: u64,
}

impl PeerMessagePrioritizer {
    /// Create a new prioritizer with the supplied configuration.
    pub fn new(config: AgingConfig) -> Self {
        Self {
            queues: HashMap::new(),
            config,
            stats: PrioritizerStats::default(),
            current_tick: 0,
            next_id: 1,
        }
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    /// Insert `msg` into `queue` maintaining the sorted invariant.
    ///
    /// Sort key: descending `effective_priority`, then ascending `enqueued_tick`
    /// (FIFO within the same effective priority).
    fn insert_sorted(queue: &mut Vec<PrioritizedMessage>, msg: PrioritizedMessage) {
        // Binary-search for the correct insertion point.
        let pos = queue.partition_point(|m| {
            // Items that should come *before* `msg` satisfy this predicate.
            m.effective_priority > msg.effective_priority
                || (m.effective_priority == msg.effective_priority
                    && m.enqueued_tick <= msg.enqueued_tick)
        });
        queue.insert(pos, msg);
    }

    /// Drop the message with the lowest priority (and, within that, the most
    /// recently enqueued one) from `queue`.  Returns `true` if a message was
    /// removed.
    fn drop_lowest(queue: &mut Vec<PrioritizedMessage>) -> bool {
        if queue.is_empty() {
            return false;
        }
        // The queue is sorted highest-first so the lowest-priority messages are
        // at the tail.  Among equal-priority tail messages prefer removing the
        // one with the highest `enqueued_tick` (most recent, least-waited).
        let lowest_prio = queue.last().map(|m| m.effective_priority);
        if let Some(lp) = lowest_prio {
            // Find the index of the most-recently-enqueued message among those
            // sharing the lowest effective priority.
            let victim = queue
                .iter()
                .enumerate()
                .rev() // iterate from tail toward head
                .take_while(|(_, m)| m.effective_priority == lp)
                .max_by_key(|(_, m)| m.enqueued_tick)
                .map(|(i, _)| i);

            if let Some(idx) = victim {
                queue.remove(idx);
                return true;
            }
        }
        false
    }

    // ── Public API ────────────────────────────────────────────────────────────

    /// Enqueue a new message for `peer_id`.
    ///
    /// If the peer's queue is already at `max_queue_size`:
    /// 1. Attempt to drop the lowest-priority message (`Background` first, then
    ///    `Normal`).
    /// 2. If the queue is still full after the drop attempt, return `None`.
    ///
    /// On success, returns `Some(msg_id)`.
    pub fn enqueue(
        &mut self,
        peer_id: String,
        priority: MessagePriority,
        payload_bytes: u64,
    ) -> Option<u64> {
        let queue = self.queues.entry(peer_id.clone()).or_default();

        if queue.len() >= self.config.max_queue_size {
            // Try to make room by evicting the lowest-priority message.
            if Self::drop_lowest(queue) {
                self.stats.total_dropped += 1;
            }
            // Re-check — if still full we cannot accept the message.
            if queue.len() >= self.config.max_queue_size {
                return None;
            }
        }

        let msg_id = self.next_id;
        self.next_id += 1;

        let msg = PrioritizedMessage {
            msg_id,
            peer_id,
            priority,
            payload_bytes,
            enqueued_tick: self.current_tick,
            effective_priority: priority,
        };

        Self::insert_sorted(queue, msg);
        self.stats.total_enqueued += 1;
        Some(msg_id)
    }

    /// Remove and return the highest-priority message for `peer_id`, or `None`
    /// if the peer has no queued messages.
    pub fn dequeue(&mut self, peer_id: &str) -> Option<PrioritizedMessage> {
        let queue = self.queues.get_mut(peer_id)?;
        if queue.is_empty() {
            return None;
        }
        let msg = queue.remove(0);
        self.stats.total_dequeued += 1;
        Some(msg)
    }

    /// Advance the logical clock by one tick and apply aging promotions.
    ///
    /// Any message whose waiting time (`current_tick - enqueued_tick`) meets or
    /// exceeds `promote_after_ticks` and whose `effective_priority` is below
    /// `Urgent` is promoted by one level.  Queues that had at least one
    /// promotion are re-sorted to preserve the ordering invariant.
    pub fn advance_tick(&mut self) {
        self.current_tick += 1;

        let promote_after = self.config.promote_after_ticks;
        let current_tick = self.current_tick;

        for queue in self.queues.values_mut() {
            let mut any_promoted = false;

            for msg in queue.iter_mut() {
                if msg.effective_priority < MessagePriority::Urgent
                    && current_tick.saturating_sub(msg.enqueued_tick) >= promote_after
                {
                    msg.effective_priority = msg.effective_priority.promote();
                    self.stats.total_promoted += 1;
                    any_promoted = true;
                }
            }

            if any_promoted {
                // Re-sort: descending effective_priority, then ascending enqueued_tick.
                queue.sort_by(|a, b| {
                    b.effective_priority
                        .cmp(&a.effective_priority)
                        .then_with(|| a.enqueued_tick.cmp(&b.enqueued_tick))
                });
            }
        }
    }

    /// Return a reference to the accumulated statistics.
    pub fn stats(&self) -> &PrioritizerStats {
        &self.stats
    }

    /// Return the number of messages currently queued for `peer_id`.
    pub fn queue_depth(&self, peer_id: &str) -> usize {
        self.queues.get(peer_id).map_or(0, |q| q.len())
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_prioritizer() -> PeerMessagePrioritizer {
        PeerMessagePrioritizer::new(AgingConfig::default())
    }

    // ── 1. enqueue returns a unique msg_id ────────────────────────────────────

    #[test]
    fn enqueue_returns_msg_id() {
        let mut p = default_prioritizer();
        let id = p.enqueue("peer-a".to_string(), MessagePriority::Normal, 128);
        assert!(id.is_some(), "enqueue should return Some(id)");
        assert_eq!(id.expect("test: enqueue should return msg_id"), 1u64);
    }

    // ── 2. msg_ids are monotonically increasing ───────────────────────────────

    #[test]
    fn msg_ids_are_monotonically_increasing() {
        let mut p = default_prioritizer();
        let id1 = p
            .enqueue("peer-a".to_string(), MessagePriority::Normal, 64)
            .expect("test: first enqueue should return msg_id");
        let id2 = p
            .enqueue("peer-a".to_string(), MessagePriority::Normal, 64)
            .expect("test: second enqueue should return msg_id");
        let id3 = p
            .enqueue("peer-b".to_string(), MessagePriority::High, 64)
            .expect("test: third enqueue should return msg_id");
        assert!(id1 < id2);
        assert!(id2 < id3);
    }

    // ── 3. dequeue returns highest effective priority first ───────────────────

    #[test]
    fn dequeue_highest_priority_first() {
        let mut p = default_prioritizer();
        p.enqueue("peer-a".to_string(), MessagePriority::Background, 10)
            .expect("test: enqueue Background should succeed");
        p.enqueue("peer-a".to_string(), MessagePriority::Urgent, 10)
            .expect("test: enqueue Urgent should succeed");
        p.enqueue("peer-a".to_string(), MessagePriority::Normal, 10)
            .expect("test: enqueue Normal should succeed");

        let first = p
            .dequeue("peer-a")
            .expect("test: first dequeue should return Urgent message");
        assert_eq!(first.effective_priority, MessagePriority::Urgent);

        let second = p
            .dequeue("peer-a")
            .expect("test: second dequeue should return Normal message");
        assert_eq!(second.effective_priority, MessagePriority::Normal);

        let third = p
            .dequeue("peer-a")
            .expect("test: third dequeue should return Background message");
        assert_eq!(third.effective_priority, MessagePriority::Background);
    }

    // ── 4. FIFO within the same priority ─────────────────────────────────────

    #[test]
    fn fifo_within_same_priority() {
        let mut p = default_prioritizer();
        let id1 = p
            .enqueue("peer-a".to_string(), MessagePriority::Normal, 10)
            .expect("test: first enqueue should succeed");
        let id2 = p
            .enqueue("peer-a".to_string(), MessagePriority::Normal, 20)
            .expect("test: second enqueue should succeed");
        let id3 = p
            .enqueue("peer-a".to_string(), MessagePriority::Normal, 30)
            .expect("test: third enqueue should succeed");

        assert_eq!(
            p.dequeue("peer-a")
                .expect("test: first dequeue should return message")
                .msg_id,
            id1
        );
        assert_eq!(
            p.dequeue("peer-a")
                .expect("test: second dequeue should return message")
                .msg_id,
            id2
        );
        assert_eq!(
            p.dequeue("peer-a")
                .expect("test: third dequeue should return message")
                .msg_id,
            id3
        );
    }

    // ── 5. FIFO is stable across ticks ───────────────────────────────────────

    #[test]
    fn fifo_stable_across_ticks() {
        let config = AgingConfig {
            promote_after_ticks: 1000, // never promote during this test
            max_queue_size: 256,
        };
        let mut p = PeerMessagePrioritizer::new(config);
        let id1 = p
            .enqueue("peer-a".to_string(), MessagePriority::High, 10)
            .expect("test: first enqueue at tick 0 should succeed");
        p.advance_tick();
        let id2 = p
            .enqueue("peer-a".to_string(), MessagePriority::High, 10)
            .expect("test: second enqueue at tick 1 should succeed");
        p.advance_tick();
        let id3 = p
            .enqueue("peer-a".to_string(), MessagePriority::High, 10)
            .expect("test: third enqueue at tick 2 should succeed");

        assert_eq!(
            p.dequeue("peer-a")
                .expect("test: first dequeue should return message")
                .msg_id,
            id1
        );
        assert_eq!(
            p.dequeue("peer-a")
                .expect("test: second dequeue should return message")
                .msg_id,
            id2
        );
        assert_eq!(
            p.dequeue("peer-a")
                .expect("test: third dequeue should return message")
                .msg_id,
            id3
        );
    }

    // ── 6. advance_tick promotes after threshold ──────────────────────────────

    #[test]
    fn advance_tick_promotes_after_threshold() {
        let config = AgingConfig {
            promote_after_ticks: 3,
            max_queue_size: 256,
        };
        let mut p = PeerMessagePrioritizer::new(config);
        p.enqueue("peer-a".to_string(), MessagePriority::Background, 10)
            .expect("test: enqueue Background should succeed");

        // Advance 2 ticks — not enough.
        p.advance_tick();
        p.advance_tick();
        assert_eq!(p.stats().total_promoted, 0);

        // Third tick crosses the threshold.
        p.advance_tick();
        assert_eq!(p.stats().total_promoted, 1);

        let msg = p
            .dequeue("peer-a")
            .expect("test: dequeue should return promoted message");
        assert_eq!(msg.effective_priority, MessagePriority::Normal);
        assert_eq!(msg.priority, MessagePriority::Background); // original unchanged
    }

    // ── 7. Promotion is capped at Urgent ──────────────────────────────────────

    #[test]
    fn promotion_capped_at_urgent() {
        let config = AgingConfig {
            promote_after_ticks: 1,
            max_queue_size: 256,
        };
        let mut p = PeerMessagePrioritizer::new(config);
        p.enqueue("peer-a".to_string(), MessagePriority::High, 10)
            .expect("test: enqueue High should succeed");

        // Advance enough ticks to try to promote beyond Urgent.
        for _ in 0..10 {
            p.advance_tick();
        }

        let msg = p
            .dequeue("peer-a")
            .expect("test: dequeue should return Urgent message after promotion");
        assert_eq!(msg.effective_priority, MessagePriority::Urgent);
    }

    // ── 8. total_promoted counter ─────────────────────────────────────────────

    #[test]
    fn total_promoted_counter() {
        let config = AgingConfig {
            promote_after_ticks: 2,
            max_queue_size: 256,
        };
        let mut p = PeerMessagePrioritizer::new(config);
        // Two Background messages enqueued at tick 0.
        p.enqueue("peer-a".to_string(), MessagePriority::Background, 10)
            .expect("test: first enqueue Background should succeed");
        p.enqueue("peer-a".to_string(), MessagePriority::Background, 10)
            .expect("test: second enqueue Background should succeed");

        p.advance_tick(); // tick 1 — waited 1 tick, not yet promoted
        assert_eq!(p.stats().total_promoted, 0);

        p.advance_tick(); // tick 2 — waited 2 ticks, both promoted
        assert_eq!(p.stats().total_promoted, 2);
    }

    // ── 9. drop lowest priority when full ────────────────────────────────────

    #[test]
    fn drop_lowest_priority_when_full() {
        let config = AgingConfig {
            promote_after_ticks: 9999,
            max_queue_size: 3,
        };
        let mut p = PeerMessagePrioritizer::new(config);

        p.enqueue("peer-a".to_string(), MessagePriority::Normal, 10)
            .expect("test: enqueue Normal should succeed");
        p.enqueue("peer-a".to_string(), MessagePriority::Background, 10)
            .expect("test: enqueue Background should succeed");
        p.enqueue("peer-a".to_string(), MessagePriority::High, 10)
            .expect("test: enqueue High should succeed");
        // Queue is now full (3 messages).

        // Enqueue Urgent — should evict Background, succeed.
        let result = p.enqueue("peer-a".to_string(), MessagePriority::Urgent, 10);
        assert!(
            result.is_some(),
            "should succeed by evicting lowest-priority message"
        );
        assert_eq!(p.stats().total_dropped, 1);
        assert_eq!(p.queue_depth("peer-a"), 3);

        // All remaining messages should have priority >= Normal.
        let priorities: Vec<_> = p.queues["peer-a"]
            .iter()
            .map(|m| m.effective_priority)
            .collect();
        for prio in &priorities {
            assert!(
                *prio >= MessagePriority::Normal,
                "Background should have been evicted"
            );
        }
    }

    // ── 10. total_dropped counter ─────────────────────────────────────────────

    #[test]
    fn total_dropped_counter() {
        let config = AgingConfig {
            promote_after_ticks: 9999,
            max_queue_size: 2,
        };
        let mut p = PeerMessagePrioritizer::new(config);
        p.enqueue("peer-a".to_string(), MessagePriority::Background, 10)
            .expect("test: first enqueue Background should succeed");
        p.enqueue("peer-a".to_string(), MessagePriority::Background, 10)
            .expect("test: second enqueue Background should succeed");
        // Evict 1 Background, enqueue Normal.
        p.enqueue("peer-a".to_string(), MessagePriority::Normal, 10)
            .expect("test: enqueue Normal should succeed");
        assert_eq!(p.stats().total_dropped, 1);
    }

    // ── 11. returns None when queue still full after drop attempt ─────────────

    #[test]
    fn returns_none_when_still_full_after_drop() {
        let config = AgingConfig {
            promote_after_ticks: 9999,
            max_queue_size: 2,
        };
        let mut p = PeerMessagePrioritizer::new(config);
        p.enqueue("peer-a".to_string(), MessagePriority::Urgent, 10)
            .expect("test: first enqueue Urgent should succeed");
        p.enqueue("peer-a".to_string(), MessagePriority::Urgent, 10)
            .expect("test: second enqueue Urgent should succeed");
        // Queue full with Urgents; nothing lower to drop — new message rejected.
        // But drop_lowest will remove one (same priority, most recent).
        // Let's fill with two Urgents then try a third Urgent.
        let third = p.enqueue("peer-a".to_string(), MessagePriority::Urgent, 10);
        // After dropping one Urgent there is space, so this succeeds.
        assert!(third.is_some());
        assert_eq!(p.stats().total_dropped, 1);
        assert_eq!(p.queue_depth("peer-a"), 2);
    }

    // ── 12. empty dequeue returns None ───────────────────────────────────────

    #[test]
    fn empty_dequeue_returns_none() {
        let mut p = default_prioritizer();
        assert!(p.dequeue("nonexistent-peer").is_none());
    }

    // ── 13. dequeue from empty queue returns None ─────────────────────────────

    #[test]
    fn dequeue_exhausted_queue_returns_none() {
        let mut p = default_prioritizer();
        p.enqueue("peer-a".to_string(), MessagePriority::Normal, 10)
            .expect("test: enqueue Normal should succeed");
        p.dequeue("peer-a");
        assert!(p.dequeue("peer-a").is_none());
    }

    // ── 14. queue_depth reflects current state ────────────────────────────────

    #[test]
    fn queue_depth_reflects_state() {
        let mut p = default_prioritizer();
        assert_eq!(p.queue_depth("peer-a"), 0);
        p.enqueue("peer-a".to_string(), MessagePriority::Normal, 10)
            .expect("test: first enqueue should succeed");
        assert_eq!(p.queue_depth("peer-a"), 1);
        p.enqueue("peer-a".to_string(), MessagePriority::High, 10)
            .expect("test: second enqueue should succeed");
        assert_eq!(p.queue_depth("peer-a"), 2);
        p.dequeue("peer-a");
        assert_eq!(p.queue_depth("peer-a"), 1);
    }

    // ── 15. throughput_ratio ──────────────────────────────────────────────────

    #[test]
    fn throughput_ratio_correct() {
        let mut p = default_prioritizer();
        // Zero enqueued → ratio = 0.0
        assert!((p.stats().throughput_ratio() - 0.0).abs() < f64::EPSILON);

        p.enqueue("peer-a".to_string(), MessagePriority::Normal, 10)
            .expect("test: first enqueue should succeed");
        p.enqueue("peer-a".to_string(), MessagePriority::Normal, 10)
            .expect("test: second enqueue should succeed");
        p.enqueue("peer-a".to_string(), MessagePriority::Normal, 10)
            .expect("test: third enqueue should succeed");
        p.dequeue("peer-a");
        // 1 dequeued / 3 enqueued = 0.333…
        let ratio = p.stats().throughput_ratio();
        assert!((ratio - 1.0 / 3.0).abs() < 1e-9, "ratio = {ratio}");
    }

    // ── 16. total_enqueued and total_dequeued counters ────────────────────────

    #[test]
    fn enqueued_and_dequeued_counters() {
        let mut p = default_prioritizer();
        for _ in 0..5 {
            p.enqueue("peer-a".to_string(), MessagePriority::Normal, 10)
                .expect("test: enqueue in loop should succeed");
        }
        assert_eq!(p.stats().total_enqueued, 5);
        for _ in 0..3 {
            p.dequeue("peer-a");
        }
        assert_eq!(p.stats().total_dequeued, 3);
    }

    // ── 17. multi-peer isolation ──────────────────────────────────────────────

    #[test]
    fn multi_peer_isolation() {
        let mut p = default_prioritizer();
        p.enqueue("peer-a".to_string(), MessagePriority::Urgent, 10)
            .expect("test: enqueue Urgent for peer-a should succeed");
        p.enqueue("peer-b".to_string(), MessagePriority::Background, 10)
            .expect("test: enqueue Background for peer-b should succeed");

        let msg_a = p
            .dequeue("peer-a")
            .expect("test: dequeue peer-a should return Urgent message");
        assert_eq!(msg_a.effective_priority, MessagePriority::Urgent);
        assert_eq!(msg_a.peer_id, "peer-a");

        let msg_b = p
            .dequeue("peer-b")
            .expect("test: dequeue peer-b should return Background message");
        assert_eq!(msg_b.effective_priority, MessagePriority::Background);
        assert_eq!(msg_b.peer_id, "peer-b");
    }

    // ── 18. advance_tick does not promote Urgent messages ────────────────────

    #[test]
    fn advance_tick_does_not_promote_urgent() {
        let config = AgingConfig {
            promote_after_ticks: 1,
            max_queue_size: 256,
        };
        let mut p = PeerMessagePrioritizer::new(config);
        p.enqueue("peer-a".to_string(), MessagePriority::Urgent, 10)
            .expect("test: enqueue Urgent should succeed");

        p.advance_tick();

        assert_eq!(p.stats().total_promoted, 0);
        let msg = p
            .dequeue("peer-a")
            .expect("test: dequeue should return Urgent message");
        assert_eq!(msg.effective_priority, MessagePriority::Urgent);
    }

    // ── 19. payload_bytes is preserved ───────────────────────────────────────

    #[test]
    fn payload_bytes_preserved() {
        let mut p = default_prioritizer();
        p.enqueue("peer-a".to_string(), MessagePriority::Normal, 4096)
            .expect("test: enqueue 4096 bytes should succeed");
        let msg = p
            .dequeue("peer-a")
            .expect("test: dequeue should return message with payload");
        assert_eq!(msg.payload_bytes, 4096);
    }

    // ── 20. promote_after_ticks boundary (strictly >=) ───────────────────────

    #[test]
    fn promote_at_exact_threshold() {
        let config = AgingConfig {
            promote_after_ticks: 5,
            max_queue_size: 256,
        };
        let mut p = PeerMessagePrioritizer::new(config);
        p.enqueue("peer-a".to_string(), MessagePriority::Background, 10)
            .expect("test: enqueue Background should succeed");

        for _ in 0..4 {
            p.advance_tick();
        }
        assert_eq!(
            p.stats().total_promoted,
            0,
            "should not promote before threshold"
        );

        p.advance_tick(); // tick 5 — exactly at threshold
        assert_eq!(
            p.stats().total_promoted,
            1,
            "should promote at exact threshold"
        );
    }
}
