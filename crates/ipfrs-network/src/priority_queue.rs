//! Multi-level priority queue for outbound peer messages.
//!
//! Provides [`PeerPriorityQueue`] with three priority tiers (Urgent, Normal,
//! Background), per-peer fairness enforcement, and a global byte-budget cap.
//!
//! # Design
//!
//! Messages are placed into one of three FIFO sub-queues based on their
//! [`MessagePriority`].  On dequeue, the implementation drains the highest
//! non-empty tier first (Urgent → Normal → Background), guaranteeing strict
//! priority ordering while still allowing lower-priority traffic to be sent
//! whenever the higher tiers are empty.
//!
//! Backpressure is enforced in two independent dimensions:
//!
//! * **Global byte budget** — the sum of all `payload_bytes` across all
//!   enqueued messages must not exceed [`QueueConfig::max_total_bytes`].
//! * **Per-peer message cap** — a single peer may not have more than
//!   [`QueueConfig::max_per_peer_messages`] messages pending (across all
//!   priority levels).
//!
//! When either limit would be exceeded the incoming message is dropped and
//! [`QueueStats::total_dropped`] is incremented.

use std::collections::HashMap;

// ─── MessagePriority ──────────────────────────────────────────────────────────

/// Priority tier for a queued peer message.
///
/// The discriminants are chosen so that `Urgent < Normal < Background` via the
/// derived `Ord` implementation; however the *dequeue* logic treats `Urgent` as
/// the **highest** priority (it is drained first).  The ascending-integer
/// convention mirrors C-style enum tables where 0 == most important.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum MessagePriority {
    /// Highest priority — dequeued before all others.
    Urgent = 0,
    /// Mid-level priority.
    Normal = 1,
    /// Lowest priority — only sent when Urgent and Normal are empty.
    Background = 2,
}

// ─── QueuedMessage ────────────────────────────────────────────────────────────

/// A message waiting to be sent to a remote peer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QueuedMessage {
    /// Unique identifier for the message.
    pub message_id: u64,
    /// Destination peer identifier.
    pub peer_id: String,
    /// Priority tier that determines the sub-queue this message is placed in.
    pub priority: MessagePriority,
    /// Size of the message payload in bytes, used for byte-budget accounting.
    pub payload_bytes: u64,
    /// Logical clock value at the time the message was enqueued.
    pub enqueued_at_tick: u64,
}

// ─── QueueConfig ─────────────────────────────────────────────────────────────

/// Configuration for [`PeerPriorityQueue`].
#[derive(Clone, Debug)]
pub struct QueueConfig {
    /// Maximum total payload bytes that may be buffered at once (default 10 MB).
    pub max_total_bytes: u64,
    /// Maximum number of pending messages per peer across all priority levels
    /// (default 100).
    pub max_per_peer_messages: usize,
    /// Relative dequeue weight for the Urgent tier (default 4).
    pub urgent_weight: u32,
    /// Relative dequeue weight for the Normal tier (default 2).
    pub normal_weight: u32,
    /// Relative dequeue weight for the Background tier (default 1).
    pub background_weight: u32,
}

impl Default for QueueConfig {
    fn default() -> Self {
        Self {
            max_total_bytes: 10_485_760, // 10 MB
            max_per_peer_messages: 100,
            urgent_weight: 4,
            normal_weight: 2,
            background_weight: 1,
        }
    }
}

// ─── QueueStats ──────────────────────────────────────────────────────────────

/// Running statistics for a [`PeerPriorityQueue`].
#[derive(Clone, Debug, Default)]
pub struct QueueStats {
    /// Total messages successfully enqueued over the lifetime of the queue.
    pub total_enqueued: u64,
    /// Total messages successfully dequeued over the lifetime of the queue.
    pub total_dequeued: u64,
    /// Total messages dropped due to budget exhaustion or per-peer limits.
    pub total_dropped: u64,
    /// Current total payload bytes held across all sub-queues.
    pub current_bytes: u64,
    /// Current total message count across all sub-queues.
    pub current_message_count: usize,
    /// Current message count per priority tier.
    pub by_priority: HashMap<MessagePriority, usize>,
}

// ─── PeerPriorityQueue ───────────────────────────────────────────────────────

/// Multi-level priority queue for outbound peer messages.
///
/// See the [module documentation](self) for a full description of the
/// backpressure strategy and dequeue ordering.
pub struct PeerPriorityQueue {
    /// FIFO sub-queue for [`MessagePriority::Urgent`] messages.
    pub urgent: Vec<QueuedMessage>,
    /// FIFO sub-queue for [`MessagePriority::Normal`] messages.
    pub normal: Vec<QueuedMessage>,
    /// FIFO sub-queue for [`MessagePriority::Background`] messages.
    pub background: Vec<QueuedMessage>,
    /// Current pending message count keyed by peer identifier.
    pub peer_counts: HashMap<String, usize>,
    /// Sum of `payload_bytes` for all currently queued messages.
    pub current_bytes: u64,
    /// Configuration controlling limits and weights.
    pub config: QueueConfig,
    /// Cumulative statistics.
    pub stats: QueueStats,
}

impl PeerPriorityQueue {
    /// Create a new queue with the supplied configuration.
    pub fn new(config: QueueConfig) -> Self {
        Self {
            urgent: Vec::new(),
            normal: Vec::new(),
            background: Vec::new(),
            peer_counts: HashMap::new(),
            current_bytes: 0,
            config,
            stats: QueueStats::default(),
        }
    }

    /// Attempt to enqueue `msg`.
    ///
    /// Returns `true` when the message was accepted, `false` when it was
    /// dropped because either the global byte budget or the per-peer message
    /// cap would be exceeded.
    pub fn enqueue(&mut self, msg: QueuedMessage) -> bool {
        // ── backpressure checks ──────────────────────────────────────────────
        let would_exceed_bytes = self
            .current_bytes
            .checked_add(msg.payload_bytes)
            .is_none_or(|total| total > self.config.max_total_bytes);

        let peer_count = self
            .peer_counts
            .get(msg.peer_id.as_str())
            .copied()
            .unwrap_or(0);
        let would_exceed_peer = peer_count >= self.config.max_per_peer_messages;

        if would_exceed_bytes || would_exceed_peer {
            self.stats.total_dropped += 1;
            return false;
        }

        // ── accept ───────────────────────────────────────────────────────────
        self.current_bytes += msg.payload_bytes;
        *self.peer_counts.entry(msg.peer_id.clone()).or_insert(0) += 1;

        // Update by_priority counter.
        *self.stats.by_priority.entry(msg.priority).or_insert(0) += 1;

        self.stats.total_enqueued += 1;
        self.stats.current_bytes = self.current_bytes;
        self.stats.current_message_count += 1;

        match msg.priority {
            MessagePriority::Urgent => self.urgent.push(msg),
            MessagePriority::Normal => self.normal.push(msg),
            MessagePriority::Background => self.background.push(msg),
        }

        true
    }

    /// Dequeue the highest-priority available message.
    ///
    /// Priority order is Urgent → Normal → Background.  Returns `None` when
    /// all sub-queues are empty.
    pub fn dequeue(&mut self) -> Option<QueuedMessage> {
        // Pick the highest non-empty tier.
        let msg = if !self.urgent.is_empty() {
            self.urgent.remove(0)
        } else if !self.normal.is_empty() {
            self.normal.remove(0)
        } else if !self.background.is_empty() {
            self.background.remove(0)
        } else {
            return None;
        };

        // ── bookkeeping ──────────────────────────────────────────────────────
        self.current_bytes = self.current_bytes.saturating_sub(msg.payload_bytes);

        // Decrement peer count; remove the key when it hits zero to keep the
        // map tidy and let later limit checks work correctly.
        let peer_count = self.peer_counts.entry(msg.peer_id.clone()).or_insert(0);
        if *peer_count > 0 {
            *peer_count -= 1;
        }
        if *peer_count == 0 {
            self.peer_counts.remove(&msg.peer_id);
        }

        // Decrement by_priority counter.
        if let Some(cnt) = self.stats.by_priority.get_mut(&msg.priority) {
            *cnt = cnt.saturating_sub(1);
        }

        self.stats.total_dequeued += 1;
        self.stats.current_bytes = self.current_bytes;
        self.stats.current_message_count = self.stats.current_message_count.saturating_sub(1);

        Some(msg)
    }

    /// Dequeue up to `n` messages, returning however many were available.
    pub fn dequeue_n(&mut self, n: usize) -> Vec<QueuedMessage> {
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            match self.dequeue() {
                Some(msg) => out.push(msg),
                None => break,
            }
        }
        out
    }

    /// Total number of messages across all sub-queues.
    pub fn len(&self) -> usize {
        self.urgent.len() + self.normal.len() + self.background.len()
    }

    /// Returns `true` when all sub-queues are empty.
    pub fn is_empty(&self) -> bool {
        self.urgent.is_empty() && self.normal.is_empty() && self.background.is_empty()
    }

    /// Reference to the current queue statistics.
    pub fn stats(&self) -> &QueueStats {
        &self.stats
    }

    /// Remove all pending messages for `peer_id`.
    ///
    /// Returns the number of messages removed.  Updates `current_bytes`,
    /// `peer_counts`, and the relevant `stats` fields.
    pub fn remove_peer(&mut self, peer_id: &str) -> usize {
        let mut removed = 0usize;

        // Helper closure to drain a sub-queue and accumulate byte/priority stats.
        macro_rules! drain_queue {
            ($queue:expr) => {{
                let before = $queue.len();
                let mut freed_bytes = 0u64;
                let mut by_prio: HashMap<MessagePriority, usize> = HashMap::new();

                $queue.retain(|msg| {
                    if msg.peer_id == peer_id {
                        freed_bytes += msg.payload_bytes;
                        *by_prio.entry(msg.priority).or_insert(0) += 1;
                        false
                    } else {
                        true
                    }
                });

                let count = before - $queue.len();
                removed += count;
                self.current_bytes = self.current_bytes.saturating_sub(freed_bytes);

                for (prio, n) in by_prio {
                    if let Some(c) = self.stats.by_priority.get_mut(&prio) {
                        *c = c.saturating_sub(n);
                    }
                }
            }};
        }

        drain_queue!(self.urgent);
        drain_queue!(self.normal);
        drain_queue!(self.background);

        // Remove the peer from the per-peer map entirely.
        self.peer_counts.remove(peer_id);

        // Sync aggregate stats.
        self.stats.current_bytes = self.current_bytes;
        self.stats.current_message_count = self.stats.current_message_count.saturating_sub(removed);

        removed
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ──────────────────────────────────────────────────────────────

    fn default_queue() -> PeerPriorityQueue {
        PeerPriorityQueue::new(QueueConfig::default())
    }

    fn msg(id: u64, peer: &str, priority: MessagePriority, bytes: u64) -> QueuedMessage {
        QueuedMessage {
            message_id: id,
            peer_id: peer.to_string(),
            priority,
            payload_bytes: bytes,
            enqueued_at_tick: id,
        }
    }

    // ── enqueue / sub-queue placement ────────────────────────────────────────

    #[test]
    fn enqueue_urgent_goes_to_urgent_queue() {
        let mut q = default_queue();
        assert!(q.enqueue(msg(1, "peer-a", MessagePriority::Urgent, 100)));
        assert_eq!(q.urgent.len(), 1);
        assert_eq!(q.normal.len(), 0);
        assert_eq!(q.background.len(), 0);
    }

    #[test]
    fn enqueue_normal_goes_to_normal_queue() {
        let mut q = default_queue();
        assert!(q.enqueue(msg(1, "peer-a", MessagePriority::Normal, 100)));
        assert_eq!(q.normal.len(), 1);
        assert_eq!(q.urgent.len(), 0);
        assert_eq!(q.background.len(), 0);
    }

    #[test]
    fn enqueue_background_goes_to_background_queue() {
        let mut q = default_queue();
        assert!(q.enqueue(msg(1, "peer-a", MessagePriority::Background, 100)));
        assert_eq!(q.background.len(), 1);
        assert_eq!(q.urgent.len(), 0);
        assert_eq!(q.normal.len(), 0);
    }

    #[test]
    fn enqueue_increments_stats() {
        let mut q = default_queue();
        q.enqueue(msg(1, "peer-a", MessagePriority::Normal, 256));
        assert_eq!(q.stats().total_enqueued, 1);
        assert_eq!(q.stats().current_message_count, 1);
        assert_eq!(q.stats().current_bytes, 256);
    }

    // ── byte budget enforcement ───────────────────────────────────────────────

    #[test]
    fn enqueue_drops_when_byte_budget_exceeded() {
        let config = QueueConfig {
            max_total_bytes: 500,
            ..Default::default()
        };
        let mut q = PeerPriorityQueue::new(config);

        // Fill up to exactly 500 bytes.
        assert!(q.enqueue(msg(1, "peer-a", MessagePriority::Normal, 500)));
        // Next message would exceed budget.
        assert!(!q.enqueue(msg(2, "peer-a", MessagePriority::Normal, 1)));
        assert_eq!(q.stats().total_dropped, 1);
        assert_eq!(q.len(), 1);
    }

    #[test]
    fn enqueue_accepts_when_byte_budget_exactly_full() {
        let config = QueueConfig {
            max_total_bytes: 1000,
            ..Default::default()
        };
        let mut q = PeerPriorityQueue::new(config);
        // Two messages totalling exactly 1000 bytes should both be accepted.
        assert!(q.enqueue(msg(1, "peer-a", MessagePriority::Normal, 600)));
        assert!(q.enqueue(msg(2, "peer-a", MessagePriority::Normal, 400)));
        assert_eq!(q.len(), 2);
        assert_eq!(q.stats().total_dropped, 0);
    }

    #[test]
    fn enqueue_byte_budget_recovers_after_dequeue() {
        let config = QueueConfig {
            max_total_bytes: 500,
            ..Default::default()
        };
        let mut q = PeerPriorityQueue::new(config);
        assert!(q.enqueue(msg(1, "peer-a", MessagePriority::Normal, 400)));
        // Would exceed budget.
        assert!(!q.enqueue(msg(2, "peer-a", MessagePriority::Normal, 200)));
        // Dequeue to free space.
        let _ = q.dequeue();
        // Now there's room.
        assert!(q.enqueue(msg(3, "peer-a", MessagePriority::Normal, 200)));
        assert_eq!(q.len(), 1);
    }

    // ── per-peer message limit ────────────────────────────────────────────────

    #[test]
    fn enqueue_drops_when_per_peer_limit_exceeded() {
        let config = QueueConfig {
            max_per_peer_messages: 3,
            max_total_bytes: u64::MAX,
            ..Default::default()
        };
        let mut q = PeerPriorityQueue::new(config);
        assert!(q.enqueue(msg(1, "peer-a", MessagePriority::Normal, 10)));
        assert!(q.enqueue(msg(2, "peer-a", MessagePriority::Normal, 10)));
        assert!(q.enqueue(msg(3, "peer-a", MessagePriority::Normal, 10)));
        // 4th message for same peer should be dropped.
        assert!(!q.enqueue(msg(4, "peer-a", MessagePriority::Normal, 10)));
        assert_eq!(q.stats().total_dropped, 1);
    }

    #[test]
    fn per_peer_limit_is_per_peer_not_global() {
        let config = QueueConfig {
            max_per_peer_messages: 2,
            max_total_bytes: u64::MAX,
            ..Default::default()
        };
        let mut q = PeerPriorityQueue::new(config);
        assert!(q.enqueue(msg(1, "peer-a", MessagePriority::Normal, 10)));
        assert!(q.enqueue(msg(2, "peer-a", MessagePriority::Normal, 10)));
        // peer-a is at limit but peer-b is not.
        assert!(!q.enqueue(msg(3, "peer-a", MessagePriority::Normal, 10)));
        assert!(q.enqueue(msg(4, "peer-b", MessagePriority::Normal, 10)));
        assert_eq!(q.stats().total_dropped, 1);
        assert_eq!(q.len(), 3);
    }

    // ── dequeue priority ordering ─────────────────────────────────────────────

    #[test]
    fn dequeue_returns_urgent_before_normal() {
        let mut q = default_queue();
        q.enqueue(msg(1, "peer-a", MessagePriority::Normal, 10));
        q.enqueue(msg(2, "peer-a", MessagePriority::Urgent, 10));
        let first = q.dequeue().expect("should have a message");
        assert_eq!(first.priority, MessagePriority::Urgent);
    }

    #[test]
    fn dequeue_returns_urgent_before_background() {
        let mut q = default_queue();
        q.enqueue(msg(1, "peer-a", MessagePriority::Background, 10));
        q.enqueue(msg(2, "peer-a", MessagePriority::Urgent, 10));
        let first = q.dequeue().expect("should have a message");
        assert_eq!(first.priority, MessagePriority::Urgent);
    }

    #[test]
    fn dequeue_returns_normal_when_urgent_empty() {
        let mut q = default_queue();
        q.enqueue(msg(1, "peer-a", MessagePriority::Background, 10));
        q.enqueue(msg(2, "peer-a", MessagePriority::Normal, 10));
        let first = q.dequeue().expect("should have a message");
        assert_eq!(first.priority, MessagePriority::Normal);
    }

    #[test]
    fn dequeue_returns_background_last() {
        let mut q = default_queue();
        q.enqueue(msg(1, "peer-a", MessagePriority::Background, 10));
        q.enqueue(msg(2, "peer-a", MessagePriority::Normal, 10));
        q.enqueue(msg(3, "peer-a", MessagePriority::Urgent, 10));
        let m1 = q.dequeue().expect("msg");
        let m2 = q.dequeue().expect("msg");
        let m3 = q.dequeue().expect("msg");
        assert_eq!(m1.priority, MessagePriority::Urgent);
        assert_eq!(m2.priority, MessagePriority::Normal);
        assert_eq!(m3.priority, MessagePriority::Background);
    }

    #[test]
    fn dequeue_returns_none_when_all_empty() {
        let mut q = default_queue();
        assert!(q.dequeue().is_none());
    }

    #[test]
    fn dequeue_updates_stats() {
        let mut q = default_queue();
        q.enqueue(msg(1, "peer-a", MessagePriority::Normal, 100));
        let _ = q.dequeue();
        assert_eq!(q.stats().total_dequeued, 1);
        assert_eq!(q.stats().current_message_count, 0);
        assert_eq!(q.stats().current_bytes, 0);
    }

    // ── FIFO ordering within a priority tier ─────────────────────────────────

    #[test]
    fn fifo_ordering_within_urgent() {
        let mut q = default_queue();
        q.enqueue(msg(1, "peer-a", MessagePriority::Urgent, 10));
        q.enqueue(msg(2, "peer-a", MessagePriority::Urgent, 10));
        q.enqueue(msg(3, "peer-a", MessagePriority::Urgent, 10));
        assert_eq!(
            q.dequeue()
                .expect("test: dequeue should return urgent message with id 1")
                .message_id,
            1
        );
        assert_eq!(
            q.dequeue()
                .expect("test: dequeue should return urgent message with id 2")
                .message_id,
            2
        );
        assert_eq!(
            q.dequeue()
                .expect("test: dequeue should return urgent message with id 3")
                .message_id,
            3
        );
    }

    #[test]
    fn fifo_ordering_within_normal() {
        let mut q = default_queue();
        for id in [10u64, 20, 30] {
            q.enqueue(msg(id, "peer-b", MessagePriority::Normal, 5));
        }
        assert_eq!(
            q.dequeue()
                .expect("test: dequeue should return normal message with id 10")
                .message_id,
            10
        );
        assert_eq!(
            q.dequeue()
                .expect("test: dequeue should return normal message with id 20")
                .message_id,
            20
        );
        assert_eq!(
            q.dequeue()
                .expect("test: dequeue should return normal message with id 30")
                .message_id,
            30
        );
    }

    #[test]
    fn fifo_ordering_within_background() {
        let mut q = default_queue();
        for id in [100u64, 200, 300] {
            q.enqueue(msg(id, "peer-c", MessagePriority::Background, 5));
        }
        assert_eq!(
            q.dequeue()
                .expect("test: dequeue should return background message with id 100")
                .message_id,
            100
        );
        assert_eq!(
            q.dequeue()
                .expect("test: dequeue should return background message with id 200")
                .message_id,
            200
        );
        assert_eq!(
            q.dequeue()
                .expect("test: dequeue should return background message with id 300")
                .message_id,
            300
        );
    }

    // ── dequeue_n ────────────────────────────────────────────────────────────

    #[test]
    fn dequeue_n_returns_up_to_n() {
        let mut q = default_queue();
        for id in 1u64..=5 {
            q.enqueue(msg(id, "peer-a", MessagePriority::Normal, 10));
        }
        let drained = q.dequeue_n(3);
        assert_eq!(drained.len(), 3);
        assert_eq!(q.len(), 2);
    }

    #[test]
    fn dequeue_n_returns_fewer_when_queue_has_less() {
        let mut q = default_queue();
        q.enqueue(msg(1, "peer-a", MessagePriority::Normal, 10));
        q.enqueue(msg(2, "peer-a", MessagePriority::Normal, 10));
        let drained = q.dequeue_n(10);
        assert_eq!(drained.len(), 2);
        assert!(q.is_empty());
    }

    #[test]
    fn dequeue_n_zero_returns_empty_vec() {
        let mut q = default_queue();
        q.enqueue(msg(1, "peer-a", MessagePriority::Normal, 10));
        let drained = q.dequeue_n(0);
        assert!(drained.is_empty());
        assert_eq!(q.len(), 1);
    }

    // ── remove_peer ───────────────────────────────────────────────────────────

    #[test]
    fn remove_peer_removes_all_messages_for_peer() {
        let mut q = default_queue();
        q.enqueue(msg(1, "peer-a", MessagePriority::Urgent, 10));
        q.enqueue(msg(2, "peer-a", MessagePriority::Normal, 20));
        q.enqueue(msg(3, "peer-a", MessagePriority::Background, 30));
        q.enqueue(msg(4, "peer-b", MessagePriority::Normal, 40));

        let removed = q.remove_peer("peer-a");
        assert_eq!(removed, 3);
        assert_eq!(q.len(), 1);
        assert_eq!(
            q.dequeue()
                .expect("test: dequeue should return peer-b message after remove_peer")
                .peer_id,
            "peer-b"
        );
    }

    #[test]
    fn remove_peer_updates_current_bytes() {
        let mut q = default_queue();
        q.enqueue(msg(1, "peer-a", MessagePriority::Normal, 300));
        q.enqueue(msg(2, "peer-b", MessagePriority::Normal, 200));
        q.remove_peer("peer-a");
        assert_eq!(q.current_bytes, 200);
        assert_eq!(q.stats().current_bytes, 200);
    }

    #[test]
    fn remove_peer_for_unknown_peer_returns_zero() {
        let mut q = default_queue();
        q.enqueue(msg(1, "peer-a", MessagePriority::Normal, 10));
        let removed = q.remove_peer("peer-x");
        assert_eq!(removed, 0);
        assert_eq!(q.len(), 1);
    }

    // ── stats.by_priority ────────────────────────────────────────────────────

    #[test]
    fn by_priority_tracks_enqueued_counts() {
        let mut q = default_queue();
        q.enqueue(msg(1, "peer-a", MessagePriority::Urgent, 10));
        q.enqueue(msg(2, "peer-a", MessagePriority::Urgent, 10));
        q.enqueue(msg(3, "peer-a", MessagePriority::Normal, 10));
        let stats = q.stats();
        assert_eq!(
            stats
                .by_priority
                .get(&MessagePriority::Urgent)
                .copied()
                .unwrap_or(0),
            2
        );
        assert_eq!(
            stats
                .by_priority
                .get(&MessagePriority::Normal)
                .copied()
                .unwrap_or(0),
            1
        );
        assert_eq!(
            stats
                .by_priority
                .get(&MessagePriority::Background)
                .copied()
                .unwrap_or(0),
            0
        );
    }

    #[test]
    fn by_priority_decrements_on_dequeue() {
        let mut q = default_queue();
        q.enqueue(msg(1, "peer-a", MessagePriority::Urgent, 10));
        q.enqueue(msg(2, "peer-a", MessagePriority::Urgent, 10));
        q.dequeue();
        let stats = q.stats();
        assert_eq!(
            stats
                .by_priority
                .get(&MessagePriority::Urgent)
                .copied()
                .unwrap_or(0),
            1
        );
    }

    // ── current_bytes tracking ────────────────────────────────────────────────

    #[test]
    fn current_bytes_increases_on_enqueue() {
        let mut q = default_queue();
        q.enqueue(msg(1, "peer-a", MessagePriority::Normal, 128));
        q.enqueue(msg(2, "peer-a", MessagePriority::Normal, 256));
        assert_eq!(q.current_bytes, 384);
    }

    #[test]
    fn current_bytes_decreases_on_dequeue() {
        let mut q = default_queue();
        q.enqueue(msg(1, "peer-a", MessagePriority::Normal, 512));
        q.enqueue(msg(2, "peer-a", MessagePriority::Normal, 256));
        q.dequeue();
        assert_eq!(q.current_bytes, 256);
    }

    #[test]
    fn current_bytes_zero_after_all_dequeued() {
        let mut q = default_queue();
        q.enqueue(msg(1, "peer-a", MessagePriority::Urgent, 100));
        q.enqueue(msg(2, "peer-b", MessagePriority::Normal, 200));
        q.dequeue();
        q.dequeue();
        assert_eq!(q.current_bytes, 0);
    }

    // ── len / is_empty ───────────────────────────────────────────────────────

    #[test]
    fn len_sums_all_sub_queues() {
        let mut q = default_queue();
        q.enqueue(msg(1, "peer-a", MessagePriority::Urgent, 10));
        q.enqueue(msg(2, "peer-a", MessagePriority::Normal, 10));
        q.enqueue(msg(3, "peer-a", MessagePriority::Background, 10));
        assert_eq!(q.len(), 3);
    }

    #[test]
    fn is_empty_true_when_no_messages() {
        let q = default_queue();
        assert!(q.is_empty());
    }

    #[test]
    fn is_empty_false_after_enqueue() {
        let mut q = default_queue();
        q.enqueue(msg(1, "peer-a", MessagePriority::Normal, 10));
        assert!(!q.is_empty());
    }
}
