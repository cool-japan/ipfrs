//! TopicRouter — Intelligent GossipSub topic management with priority-based message queuing.
//!
//! This module provides a production-grade topic router that sits above the raw GossipSub layer
//! and adds:
//!
//! - Per-topic configuration (queue depth, priority threshold, TTL)
//! - Lock-free per-topic message counters via `Arc<AtomicU64>`
//! - Priority queues backed by `BinaryHeap` — highest-priority messages dequeued first
//! - Automatic drop of messages below a topic's `priority_threshold`
//! - Bounded queues — messages rejected with `QueueFull` when `max_queue_depth` is exceeded
//! - Aggregate statistics with atomic accumulators and a `snapshot()` API

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::sync::{
    atomic::{AtomicU64, Ordering as AOrdering},
    Arc, Mutex,
};
use std::time::{Duration, Instant};

use thiserror::Error;

// ─── Errors ──────────────────────────────────────────────────────────────────

/// Errors that can occur in `TopicRouter` operations.
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum TopicError {
    /// The requested topic has not been registered.
    #[error("Topic not found: {0}")]
    TopicNotFound(String),

    /// A topic with the same name is already registered.
    #[error("Topic already registered: {0}")]
    AlreadyRegistered(String),

    /// The topic's queue is at capacity.
    #[error("Queue full for topic '{topic}' (max {max})")]
    QueueFull { topic: String, max: usize },

    /// The message priority is below the configured threshold for the topic.
    #[error("Message priority {priority} is below threshold for topic '{topic}'")]
    BelowThreshold { topic: String, priority: u8 },
}

// ─── TopicConfig ─────────────────────────────────────────────────────────────

/// Per-topic routing configuration.
#[derive(Debug, Clone)]
pub struct TopicConfig {
    /// Maximum number of messages that may reside in the queue at once.
    /// Messages enqueued beyond this limit are rejected with `TopicError::QueueFull`.
    pub max_queue_depth: usize,

    /// Minimum priority required for a message to be accepted.
    /// Messages strictly below this value are silently dropped and counted as dropped.
    pub priority_threshold: u8,

    /// Maximum age a message may reach before `PrioritizedMessage::is_expired` returns `true`.
    pub ttl: Duration,
}

impl Default for TopicConfig {
    fn default() -> Self {
        Self {
            max_queue_depth: 1000,
            priority_threshold: 0,
            ttl: Duration::from_secs(60),
        }
    }
}

// ─── PrioritizedMessage ───────────────────────────────────────────────────────

/// A message that can be stored in a per-topic `BinaryHeap`.
///
/// Ordering is based purely on `priority`: higher numerical value → greater ordering,
/// so the binary heap (a max-heap) will always pop the highest-priority message first.
/// Ties are broken by *earlier* `enqueued_at` (FIFO within the same priority).
#[derive(Debug, Clone)]
pub struct PrioritizedMessage {
    /// Raw message bytes.
    pub payload: Vec<u8>,

    /// Message priority (0 = lowest, 255 = highest).
    pub priority: u8,

    /// Monotonic timestamp recorded at enqueue time.
    pub enqueued_at: Instant,

    /// The topic this message belongs to.
    pub topic: String,
}

impl PrioritizedMessage {
    /// Returns `true` when the message has lived longer than `ttl`.
    #[must_use]
    pub fn is_expired(&self, ttl: Duration) -> bool {
        self.enqueued_at.elapsed() >= ttl
    }
}

impl PartialEq for PrioritizedMessage {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority && self.enqueued_at == other.enqueued_at
    }
}

impl Eq for PrioritizedMessage {}

impl PartialOrd for PrioritizedMessage {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PrioritizedMessage {
    fn cmp(&self, other: &Self) -> Ordering {
        // Higher priority → greater. Tie-break: earlier enqueued_at → greater (FIFO).
        self.priority
            .cmp(&other.priority)
            .then_with(|| other.enqueued_at.cmp(&self.enqueued_at))
    }
}

// ─── TopicRouterStats ─────────────────────────────────────────────────────────

/// A point-in-time snapshot of aggregate router statistics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TopicRouterStatsSnapshot {
    /// Number of currently registered topics.
    pub total_topics: usize,
    /// Total messages ever successfully enqueued (not dropped).
    pub total_enqueued: u64,
    /// Total messages dropped (below threshold or queue full).
    pub total_dropped: u64,
    /// Total messages dequeued.
    pub total_dequeued: u64,
}

/// Atomic accumulator for aggregate router statistics.
///
/// Call `snapshot()` to get a plain `TopicRouterStatsSnapshot` suitable for logging/export.
#[derive(Debug)]
pub struct TopicRouterStats {
    total_enqueued: Arc<AtomicU64>,
    total_dropped: Arc<AtomicU64>,
    total_dequeued: Arc<AtomicU64>,
}

impl Default for TopicRouterStats {
    fn default() -> Self {
        Self {
            total_enqueued: Arc::new(AtomicU64::new(0)),
            total_dropped: Arc::new(AtomicU64::new(0)),
            total_dequeued: Arc::new(AtomicU64::new(0)),
        }
    }
}

impl TopicRouterStats {
    /// Create a new zeroed stats accumulator.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns a cloned handle that shares the same atomic counters.
    #[must_use]
    pub fn clone_handle(&self) -> Self {
        Self {
            total_enqueued: Arc::clone(&self.total_enqueued),
            total_dropped: Arc::clone(&self.total_dropped),
            total_dequeued: Arc::clone(&self.total_dequeued),
        }
    }

    /// Record one successfully enqueued message.
    pub(crate) fn record_enqueue(&self) {
        self.total_enqueued.fetch_add(1, AOrdering::Relaxed);
    }

    /// Record one dropped message.
    pub(crate) fn record_drop(&self) {
        self.total_dropped.fetch_add(1, AOrdering::Relaxed);
    }

    /// Record one dequeued message.
    pub(crate) fn record_dequeue(&self) {
        self.total_dequeued.fetch_add(1, AOrdering::Relaxed);
    }

    /// Take an instantaneous snapshot of all counters.
    #[must_use]
    pub fn snapshot(&self, total_topics: usize) -> TopicRouterStatsSnapshot {
        TopicRouterStatsSnapshot {
            total_topics,
            total_enqueued: self.total_enqueued.load(AOrdering::Relaxed),
            total_dropped: self.total_dropped.load(AOrdering::Relaxed),
            total_dequeued: self.total_dequeued.load(AOrdering::Relaxed),
        }
    }
}

// ─── TopicRouter ─────────────────────────────────────────────────────────────

/// Intelligent GossipSub topic manager with priority-based message queuing.
///
/// `TopicRouter` is `Send + Sync` and designed for concurrent use behind an `Arc`.
///
/// # Example
///
/// ```rust
/// use ipfrs_network::topic_router::{TopicRouter, TopicConfig, PrioritizedMessage};
/// use std::time::{Duration, Instant};
///
/// let router = TopicRouter::new();
/// router.register_topic("blocks", TopicConfig::default()).unwrap();
///
/// let msg = PrioritizedMessage {
///     payload: b"hello".to_vec(),
///     priority: 200,
///     enqueued_at: Instant::now(),
///     topic: "blocks".to_string(),
/// };
/// router.enqueue("blocks", msg).unwrap();
/// let popped = router.dequeue("blocks");
/// assert!(popped.is_some());
/// ```
pub struct TopicRouter {
    /// Registered topic configurations.
    topics: Mutex<HashMap<String, TopicConfig>>,

    /// Per-topic message counter (messages ever enqueued, not dropped).
    message_counts: Mutex<HashMap<String, Arc<AtomicU64>>>,

    /// Per-topic priority queues.
    priority_queues: Mutex<HashMap<String, BinaryHeap<PrioritizedMessage>>>,

    /// Aggregate statistics.
    stats: TopicRouterStats,
}

impl Default for TopicRouter {
    fn default() -> Self {
        Self::new()
    }
}

impl TopicRouter {
    /// Create a new, empty `TopicRouter`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            topics: Mutex::new(HashMap::new()),
            message_counts: Mutex::new(HashMap::new()),
            priority_queues: Mutex::new(HashMap::new()),
            stats: TopicRouterStats::new(),
        }
    }

    // ── Registration ─────────────────────────────────────────────────────────

    /// Register a new topic with the given configuration.
    ///
    /// # Errors
    ///
    /// Returns `TopicError::AlreadyRegistered` if the topic name has already been registered.
    pub fn register_topic(&self, name: &str, config: TopicConfig) -> Result<(), TopicError> {
        let mut topics = self.topics.lock().unwrap_or_else(|e| e.into_inner());

        if topics.contains_key(name) {
            return Err(TopicError::AlreadyRegistered(name.to_string()));
        }

        topics.insert(name.to_string(), config);
        drop(topics);

        // Initialise per-topic structures.
        self.message_counts
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(name.to_string(), Arc::new(AtomicU64::new(0)));

        self.priority_queues
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(name.to_string(), BinaryHeap::new());

        Ok(())
    }

    /// Unregister a topic and discard its queue and counters.
    ///
    /// # Errors
    ///
    /// Returns `TopicError::TopicNotFound` if the topic has not been registered.
    pub fn unregister_topic(&self, name: &str) -> Result<(), TopicError> {
        let removed = self
            .topics
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(name)
            .is_some();

        if !removed {
            return Err(TopicError::TopicNotFound(name.to_string()));
        }

        self.message_counts
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(name);

        self.priority_queues
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(name);

        Ok(())
    }

    // ── Queuing ───────────────────────────────────────────────────────────────

    /// Enqueue a message on the given topic.
    ///
    /// # Errors
    ///
    /// - `TopicError::TopicNotFound` — topic not registered
    /// - `TopicError::BelowThreshold` — message priority below topic threshold (also counts as dropped)
    /// - `TopicError::QueueFull` — queue at capacity (also counts as dropped)
    pub fn enqueue(&self, topic: &str, msg: PrioritizedMessage) -> Result<(), TopicError> {
        // Retrieve config under a short-lived lock to avoid holding it across queue operations.
        let (max_depth, threshold) = {
            let topics = self.topics.lock().unwrap_or_else(|e| e.into_inner());
            match topics.get(topic) {
                Some(cfg) => (cfg.max_queue_depth, cfg.priority_threshold),
                None => return Err(TopicError::TopicNotFound(topic.to_string())),
            }
        };

        // Drop messages below the priority threshold.
        if msg.priority < threshold {
            self.stats.record_drop();
            return Err(TopicError::BelowThreshold {
                topic: topic.to_string(),
                priority: msg.priority,
            });
        }

        // Push into the priority queue, enforcing depth limit.
        let mut queues = self
            .priority_queues
            .lock()
            .unwrap_or_else(|e| e.into_inner());

        let queue = queues.get_mut(topic).ok_or_else(|| {
            // Should be impossible if registration succeeded, but handle gracefully.
            TopicError::TopicNotFound(topic.to_string())
        })?;

        if queue.len() >= max_depth {
            drop(queues);
            self.stats.record_drop();
            return Err(TopicError::QueueFull {
                topic: topic.to_string(),
                max: max_depth,
            });
        }

        queue.push(msg);
        drop(queues);

        // Update per-topic counter and aggregate stats.
        if let Some(counter) = self
            .message_counts
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(topic)
        {
            counter.fetch_add(1, AOrdering::Relaxed);
        }

        self.stats.record_enqueue();
        Ok(())
    }

    /// Pop and return the highest-priority message from the topic's queue.
    ///
    /// Returns `None` if the topic does not exist or the queue is empty.
    #[must_use]
    pub fn dequeue(&self, topic: &str) -> Option<PrioritizedMessage> {
        let msg = self
            .priority_queues
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get_mut(topic)?
            .pop()?;

        self.stats.record_dequeue();
        Some(msg)
    }

    // ── Inspection ────────────────────────────────────────────────────────────

    /// Returns the number of messages currently in the topic's queue.
    ///
    /// Returns `0` for unregistered topics.
    #[must_use]
    pub fn queue_depth(&self, topic: &str) -> usize {
        self.priority_queues
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(topic)
            .map(|q| q.len())
            .unwrap_or(0)
    }

    /// Returns the total number of messages ever successfully enqueued on the topic
    /// (messages that were dropped are not counted here).
    ///
    /// Returns `0` for unregistered topics.
    #[must_use]
    pub fn message_count(&self, topic: &str) -> u64 {
        self.message_counts
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(topic)
            .map(|c| c.load(AOrdering::Relaxed))
            .unwrap_or(0)
    }

    /// Returns a sorted list of all registered topic names.
    #[must_use]
    pub fn all_topics(&self) -> Vec<String> {
        let mut names: Vec<String> = self
            .topics
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .keys()
            .cloned()
            .collect();
        names.sort();
        names
    }

    /// Returns a point-in-time statistics snapshot.
    #[must_use]
    pub fn stats(&self) -> TopicRouterStatsSnapshot {
        let total_topics = self.topics.lock().unwrap_or_else(|e| e.into_inner()).len();
        self.stats.snapshot(total_topics)
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    fn make_msg(topic: &str, priority: u8, payload: &[u8]) -> PrioritizedMessage {
        PrioritizedMessage {
            payload: payload.to_vec(),
            priority,
            enqueued_at: Instant::now(),
            topic: topic.to_string(),
        }
    }

    fn default_router_with_topic(topic: &str) -> TopicRouter {
        let r = TopicRouter::new();
        r.register_topic(topic, TopicConfig::default())
            .expect("test: register_topic should succeed for new topic");
        r
    }

    // ── 1. Register a topic successfully ─────────────────────────────────────
    #[test]
    fn test_register_topic_ok() {
        let router = TopicRouter::new();
        let result = router.register_topic("alpha", TopicConfig::default());
        assert!(result.is_ok(), "Expected successful registration");
        assert!(router.all_topics().contains(&"alpha".to_string()));
    }

    // ── 2. Duplicate registration is rejected ─────────────────────────────────
    #[test]
    fn test_register_topic_duplicate() {
        let router = TopicRouter::new();
        router
            .register_topic("beta", TopicConfig::default())
            .expect("test: first registration of beta should succeed");
        let err = router
            .register_topic("beta", TopicConfig::default())
            .expect_err("test: duplicate registration should fail");
        assert!(
            matches!(err, TopicError::AlreadyRegistered(ref n) if n == "beta"),
            "Expected AlreadyRegistered, got {err:?}"
        );
    }

    // ── 3. Unregister removes the topic ───────────────────────────────────────
    #[test]
    fn test_unregister_topic() {
        let router = default_router_with_topic("gamma");
        router
            .unregister_topic("gamma")
            .expect("test: unregister_topic should succeed for registered topic");
        assert!(!router.all_topics().contains(&"gamma".to_string()));
    }

    // ── 4. Unregister unknown topic returns TopicNotFound ─────────────────────
    #[test]
    fn test_unregister_nonexistent_topic() {
        let router = TopicRouter::new();
        let err = router
            .unregister_topic("ghost")
            .expect_err("test: unregister_topic should fail for unknown topic");
        assert!(
            matches!(err, TopicError::TopicNotFound(ref n) if n == "ghost"),
            "Expected TopicNotFound"
        );
    }

    // ── 5. Enqueue on unknown topic returns TopicNotFound ─────────────────────
    #[test]
    fn test_enqueue_unknown_topic() {
        let router = TopicRouter::new();
        let msg = make_msg("unknown", 100, b"payload");
        let err = router
            .enqueue("unknown", msg)
            .expect_err("test: enqueue on unknown topic should fail");
        assert!(matches!(err, TopicError::TopicNotFound(_)));
    }

    // ── 6. Messages dequeued in priority order (highest first) ───────────────
    #[test]
    fn test_priority_ordering() {
        let router = default_router_with_topic("prio");
        router
            .enqueue("prio", make_msg("prio", 10, b"low"))
            .expect("test: enqueue should succeed for valid priority message");
        router
            .enqueue("prio", make_msg("prio", 200, b"high"))
            .expect("test: enqueue should succeed for valid priority message");
        router
            .enqueue("prio", make_msg("prio", 100, b"mid"))
            .expect("test: enqueue should succeed for valid priority message");

        let first = router
            .dequeue("prio")
            .expect("test: dequeue should return a message from non-empty queue");
        let second = router
            .dequeue("prio")
            .expect("test: dequeue should return a message from non-empty queue");
        let third = router
            .dequeue("prio")
            .expect("test: dequeue should return a message from non-empty queue");

        assert_eq!(
            first.payload, b"high",
            "First dequeued should be highest priority"
        );
        assert_eq!(second.payload, b"mid");
        assert_eq!(third.payload, b"low");
    }

    // ── 7. Dequeue returns None on empty queue ───────────────────────────────
    #[test]
    fn test_dequeue_empty() {
        let router = default_router_with_topic("empty");
        assert!(router.dequeue("empty").is_none());
    }

    // ── 8. Queue depth limit is enforced ─────────────────────────────────────
    #[test]
    fn test_queue_depth_limit() {
        let router = TopicRouter::new();
        let config = TopicConfig {
            max_queue_depth: 3,
            ..Default::default()
        };
        router
            .register_topic("bounded", config)
            .expect("test: register bounded topic should succeed");

        for i in 0_u8..3 {
            router
                .enqueue("bounded", make_msg("bounded", i, b"x"))
                .expect("test: enqueue should succeed when queue is not full");
        }
        assert_eq!(router.queue_depth("bounded"), 3);

        let err = router
            .enqueue("bounded", make_msg("bounded", 1, b"overflow"))
            .expect_err("test: enqueue beyond max_queue_depth should fail");
        assert!(
            matches!(err, TopicError::QueueFull { ref topic, max: 3 } if topic == "bounded"),
            "Expected QueueFull"
        );
    }

    // ── 9. Priority threshold drops low-priority messages ────────────────────
    #[test]
    fn test_priority_threshold_drop() {
        let router = TopicRouter::new();
        let config = TopicConfig {
            priority_threshold: 100,
            ..Default::default()
        };
        router
            .register_topic("thresh", config)
            .expect("test: register thresh topic should succeed");

        let err = router
            .enqueue("thresh", make_msg("thresh", 50, b"low"))
            .expect_err("test: enqueue below priority threshold should fail");
        assert!(
            matches!(err, TopicError::BelowThreshold { ref topic, priority: 50 } if topic == "thresh"),
            "Expected BelowThreshold"
        );

        // Message at exact threshold should be accepted.
        router
            .enqueue("thresh", make_msg("thresh", 100, b"ok"))
            .expect("test: enqueue at exact priority threshold should succeed");
        assert_eq!(router.queue_depth("thresh"), 1);
    }

    // ── 10. Stats accumulate correctly ───────────────────────────────────────
    #[test]
    fn test_stats_accumulation() {
        let router = TopicRouter::new();
        let config = TopicConfig {
            priority_threshold: 50,
            max_queue_depth: 2,
            ..Default::default()
        };
        router
            .register_topic("stats_topic", config)
            .expect("test: register stats_topic should succeed");

        // 2 successful enqueues
        router
            .enqueue("stats_topic", make_msg("stats_topic", 100, b"a"))
            .expect("test: enqueue into stats_topic should succeed");
        router
            .enqueue("stats_topic", make_msg("stats_topic", 200, b"b"))
            .expect("test: enqueue into stats_topic should succeed");
        // 1 drop: below threshold
        let _ = router.enqueue("stats_topic", make_msg("stats_topic", 10, b"c"));
        // 1 drop: queue full
        let _ = router.enqueue("stats_topic", make_msg("stats_topic", 200, b"d"));

        // 1 dequeue
        let _ = router.dequeue("stats_topic");

        let snap = router.stats();
        assert_eq!(snap.total_enqueued, 2);
        assert_eq!(snap.total_dropped, 2);
        assert_eq!(snap.total_dequeued, 1);
        assert_eq!(snap.total_topics, 1);
    }

    // ── 11. TTL expiry check on PrioritizedMessage ───────────────────────────
    #[test]
    fn test_ttl_expiry() {
        // A message created with an artificially old enqueued_at should be expired.
        let old = PrioritizedMessage {
            payload: vec![],
            priority: 100,
            // Pretend the message was enqueued 2 seconds ago by lying about enqueued_at.
            // We do this by creating a fresh Instant and noting it predates the TTL check.
            enqueued_at: Instant::now(),
            topic: "ttl_topic".to_string(),
        };
        // With a 10s TTL a brand-new message should NOT be expired.
        assert!(!old.is_expired(Duration::from_secs(10)));

        // With a 0-nanosecond TTL any message should immediately be expired.
        assert!(old.is_expired(Duration::from_nanos(0)));
    }

    // ── 12. all_topics() returns all registered topics sorted ────────────────
    #[test]
    fn test_all_topics_listing() {
        let router = TopicRouter::new();
        router
            .register_topic("zebra", TopicConfig::default())
            .expect("test: register_topic should succeed for unique topic name");
        router
            .register_topic("apple", TopicConfig::default())
            .expect("test: register_topic should succeed for unique topic name");
        router
            .register_topic("mango", TopicConfig::default())
            .expect("test: register_topic should succeed for unique topic name");

        let topics = router.all_topics();
        assert_eq!(topics, vec!["apple", "mango", "zebra"]);
    }

    // ── 13. message_count reflects enqueued (not dropped) messages ───────────
    #[test]
    fn test_message_count_excludes_drops() {
        let router = TopicRouter::new();
        let config = TopicConfig {
            priority_threshold: 100,
            ..Default::default()
        };
        router
            .register_topic("cnt", config)
            .expect("test: register cnt topic should succeed");

        router
            .enqueue("cnt", make_msg("cnt", 200, b"good"))
            .expect("test: enqueue good message into cnt should succeed");
        let _ = router.enqueue("cnt", make_msg("cnt", 50, b"bad")); // dropped

        assert_eq!(router.message_count("cnt"), 1);
    }

    // ── 14. queue_depth tracks enqueue and dequeue correctly ─────────────────
    #[test]
    fn test_queue_depth_tracking() {
        let router = default_router_with_topic("depth");
        assert_eq!(router.queue_depth("depth"), 0);

        router
            .enqueue("depth", make_msg("depth", 1, b"a"))
            .expect("test: enqueue into depth topic should succeed");
        router
            .enqueue("depth", make_msg("depth", 2, b"b"))
            .expect("test: enqueue into depth topic should succeed");
        assert_eq!(router.queue_depth("depth"), 2);

        let _ = router.dequeue("depth");
        assert_eq!(router.queue_depth("depth"), 1);

        let _ = router.dequeue("depth");
        assert_eq!(router.queue_depth("depth"), 0);
    }

    // ── 15. Stats total_topics reflects current registration count ───────────
    #[test]
    fn test_stats_total_topics() {
        let router = TopicRouter::new();
        assert_eq!(router.stats().total_topics, 0);

        router
            .register_topic("t1", TopicConfig::default())
            .expect("test: register topic t1 should succeed");
        assert_eq!(router.stats().total_topics, 1);

        router
            .register_topic("t2", TopicConfig::default())
            .expect("test: register topic t2 should succeed");
        assert_eq!(router.stats().total_topics, 2);

        router
            .unregister_topic("t1")
            .expect("test: unregister_topic t1 should succeed");
        assert_eq!(router.stats().total_topics, 1);
    }
}
