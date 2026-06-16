//! MessageRouter — Priority-based message dispatch with dead-letter queue.
//!
//! Routes incoming [`RoutedMessage`]s to registered [`HandlerRegistration`]s using
//! exact-match or wildcard topic patterns.  Each handler maintains its own
//! `BinaryHeap`-backed priority queue.  Messages that match no handler are
//! forwarded to a bounded dead-letter queue.
//!
//! ## Design highlights
//!
//! - **Priority dispatch**: handlers are stored sorted by priority descending so
//!   higher-priority handlers are always evaluated first.
//! - **Wildcard matching**: supports exact match, prefix wildcard (`block.*`), and
//!   catch-all (`*`).
//! - **Dead-letter queue**: bounded at 100 items; oldest entries are evicted when
//!   full.
//! - **Atomic statistics**: all counters use `AtomicU64` for lock-free reads.
//! - **No `unwrap()`**: all fallible operations surface errors explicitly.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, VecDeque};
use std::sync::{
    atomic::{AtomicU64, Ordering as AOrdering},
    Arc, Mutex, RwLock,
};
use std::time::{Duration, Instant};

use thiserror::Error;

// ─── Constants ────────────────────────────────────────────────────────────────

/// Maximum number of items retained in the dead-letter queue.
const MAX_DEAD_LETTER: usize = 100;

/// Default message TTL.
const DEFAULT_TTL: Duration = Duration::from_secs(30);

// ─── Errors ───────────────────────────────────────────────────────────────────

/// Errors that can be returned by [`MessageRouter`].
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum RouterError {
    /// A handler with the same `handler_id` is already registered.
    #[error("Duplicate handler: {0}")]
    DuplicateHandler(String),

    /// No handler with the given `handler_id` exists.
    #[error("Handler not found: {0}")]
    HandlerNotFound(String),
}

// ─── MessagePriority ──────────────────────────────────────────────────────────

/// Message priority levels.
///
/// `Critical` is the highest priority.  The integer discriminants (3 > 2 > 1 > 0)
/// drive the derived `Ord` implementation so that `Critical > High > Normal > Low`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[repr(u8)]
pub enum MessagePriority {
    /// Lowest priority — background / best-effort.
    Low = 0,
    /// Normal priority — default.
    #[default]
    Normal = 1,
    /// High priority — time-sensitive.
    High = 2,
    /// Critical priority — must be processed immediately.
    Critical = 3,
}

// ─── RoutedMessage ────────────────────────────────────────────────────────────

/// A message travelling through the router.
#[derive(Debug, Clone)]
pub struct RoutedMessage {
    /// Monotonically increasing message identifier assigned by the router.
    pub id: u64,
    /// Originating peer identifier.
    pub from_peer: String,
    /// Topic this message belongs to.
    pub topic: String,
    /// Raw payload bytes.
    pub payload: Vec<u8>,
    /// Priority of this message.
    pub priority: MessagePriority,
    /// Wall-clock time at which the message was enqueued.
    pub enqueued_at: Instant,
    /// How long the message remains valid after `enqueued_at`.
    pub ttl: Duration,
}

impl RoutedMessage {
    /// Create a new [`RoutedMessage`] with the default TTL of 30 s.
    pub fn new(
        id: u64,
        from_peer: impl Into<String>,
        topic: impl Into<String>,
        payload: Vec<u8>,
        priority: MessagePriority,
    ) -> Self {
        Self {
            id,
            from_peer: from_peer.into(),
            topic: topic.into(),
            payload,
            priority,
            enqueued_at: Instant::now(),
            ttl: DEFAULT_TTL,
        }
    }

    /// Returns `true` when this message has exceeded its TTL relative to `now`.
    pub fn is_expired(&self, now: Instant) -> bool {
        now.saturating_duration_since(self.enqueued_at) >= self.ttl
    }
}

// ─── HandlerRegistration ─────────────────────────────────────────────────────

/// Describes a handler that the [`MessageRouter`] should dispatch messages to.
#[derive(Debug, Clone)]
pub struct HandlerRegistration {
    /// Topic pattern used for matching.
    ///
    /// - `"blocks"` — exact match only.
    /// - `"block.*"` — prefix match: matches any topic whose prefix before `*` is `"block."`.
    /// - `"*"` — matches every topic.
    pub topic_pattern: String,

    /// Handler priority; higher-priority handlers are stored first and called first
    /// when multiple handlers share a topic.
    pub priority: MessagePriority,

    /// Unique name for this handler.
    pub handler_id: String,
}

impl HandlerRegistration {
    /// Returns `true` if `topic` matches this registration's `topic_pattern`.
    pub fn matches(&self, topic: &str) -> bool {
        let pattern = &self.topic_pattern;

        if pattern == "*" {
            return true;
        }

        if let Some(prefix) = pattern.strip_suffix('*') {
            return topic.starts_with(prefix);
        }

        pattern == topic
    }
}

// ─── QueuedMessage ────────────────────────────────────────────────────────────

/// Private wrapper that adds `Ord` to [`RoutedMessage`] so it can live in a
/// `BinaryHeap`.
///
/// Ordering: higher [`MessagePriority`] → larger; for equal priority, *earlier*
/// `enqueued_at` → larger (FIFO within the same priority level).
#[derive(Debug, Clone)]
struct QueuedMessage(RoutedMessage);

impl PartialEq for QueuedMessage {
    fn eq(&self, other: &Self) -> bool {
        self.0.id == other.0.id
    }
}

impl Eq for QueuedMessage {}

impl PartialOrd for QueuedMessage {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for QueuedMessage {
    fn cmp(&self, other: &Self) -> Ordering {
        // Higher priority → greater
        let pri = self.0.priority.cmp(&other.0.priority);
        if pri != Ordering::Equal {
            return pri;
        }
        // Earlier enqueued_at → greater (FIFO within same priority)
        other.0.enqueued_at.cmp(&self.0.enqueued_at)
    }
}

// ─── RouterStats ──────────────────────────────────────────────────────────────

/// Atomic statistics counters for a [`MessageRouter`].
#[derive(Debug, Default)]
pub struct RouterStats {
    /// Total messages submitted to `route()`.
    pub total_routed: AtomicU64,
    /// Total messages matched to at least one handler.
    pub total_matched: AtomicU64,
    /// Total messages forwarded to the dead-letter queue.
    pub total_dead_lettered: AtomicU64,
    /// Total messages dropped because they were already expired on arrival.
    pub total_expired_dropped: AtomicU64,
}

/// Snapshot of [`RouterStats`] at a point in time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RouterStatsSnapshot {
    /// Total messages submitted to `route()`.
    pub total_routed: u64,
    /// Total messages matched to at least one handler.
    pub total_matched: u64,
    /// Total messages forwarded to the dead-letter queue.
    pub total_dead_lettered: u64,
    /// Total messages dropped because they were already expired on arrival.
    pub total_expired_dropped: u64,
}

impl RouterStats {
    /// Returns a consistent snapshot of all counters.
    pub fn snapshot(&self) -> RouterStatsSnapshot {
        RouterStatsSnapshot {
            total_routed: self.total_routed.load(AOrdering::Relaxed),
            total_matched: self.total_matched.load(AOrdering::Relaxed),
            total_dead_lettered: self.total_dead_lettered.load(AOrdering::Relaxed),
            total_expired_dropped: self.total_expired_dropped.load(AOrdering::Relaxed),
        }
    }
}

// ─── MessageRouter ────────────────────────────────────────────────────────────

/// Priority-based message router with per-handler queues and a dead-letter queue.
///
/// # Thread safety
///
/// All public methods are safe to call from multiple threads simultaneously.
/// Handlers are protected by a `RwLock`; per-handler queues and the dead-letter
/// queue are protected by a `Mutex`.
pub struct MessageRouter {
    /// Registered handlers, kept sorted by priority descending.
    handlers: RwLock<Vec<HandlerRegistration>>,

    /// Per-handler priority queues keyed by `handler_id`.
    queues: Mutex<HashMap<String, BinaryHeap<QueuedMessage>>>,

    /// Unmatched messages; bounded at [`MAX_DEAD_LETTER`].
    dead_letter: Mutex<VecDeque<RoutedMessage>>,

    /// Monotonically increasing message-ID source.
    next_id: AtomicU64,

    /// Aggregate statistics.
    pub stats: Arc<RouterStats>,
}

impl Default for MessageRouter {
    fn default() -> Self {
        Self::new()
    }
}

impl MessageRouter {
    /// Create a new, empty [`MessageRouter`].
    pub fn new() -> Self {
        Self {
            handlers: RwLock::new(Vec::new()),
            queues: Mutex::new(HashMap::new()),
            dead_letter: Mutex::new(VecDeque::new()),
            next_id: AtomicU64::new(0),
            stats: Arc::new(RouterStats::default()),
        }
    }

    /// Allocate the next monotonic message ID.
    pub fn next_id(&self) -> u64 {
        self.next_id.fetch_add(1, AOrdering::Relaxed)
    }

    // ── Handler management ───────────────────────────────────────────────────

    /// Register a new handler.
    ///
    /// Returns [`RouterError::DuplicateHandler`] if a handler with the same
    /// `handler_id` is already registered.
    pub fn register_handler(&self, reg: HandlerRegistration) -> Result<(), RouterError> {
        let mut handlers = self.handlers.write().unwrap_or_else(|e| e.into_inner());

        if handlers.iter().any(|h| h.handler_id == reg.handler_id) {
            return Err(RouterError::DuplicateHandler(reg.handler_id.clone()));
        }

        handlers.push(reg);
        // Keep sorted: highest priority first.
        // sort_by_key with Reverse gives descending priority order.
        handlers.sort_by_key(|h| std::cmp::Reverse(h.priority));

        // Ensure a queue slot exists for this handler.
        let mut queues = self.queues.lock().unwrap_or_else(|e| e.into_inner());
        queues
            .entry(
                handlers
                    .last()
                    .map(|h| h.handler_id.clone())
                    .unwrap_or_default(),
            )
            .or_default();

        // Re-insert after sort; the handler we just pushed may have moved.
        // Rebuild queue slots for all known handlers.
        for h in handlers.iter() {
            queues.entry(h.handler_id.clone()).or_default();
        }

        Ok(())
    }

    /// Unregister a handler by `handler_id`.
    ///
    /// This is a no-op if the handler does not exist (no error is returned so
    /// callers can safely call this during shutdown without worrying about race
    /// conditions with double-unregister).
    pub fn unregister_handler(&self, handler_id: &str) {
        let mut handlers = self.handlers.write().unwrap_or_else(|e| e.into_inner());

        handlers.retain(|h| h.handler_id != handler_id);

        // Leave the queue in place so in-flight messages are not lost; callers
        // should drain before unregistering if they care about ordering.
    }

    /// Return the number of currently registered handlers.
    pub fn handler_count(&self) -> usize {
        self.handlers
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .len()
    }

    // ── Routing ───────────────────────────────────────────────────────────────

    /// Route `msg` to all matching handlers.
    ///
    /// Returns the number of handlers the message was dispatched to.  When no
    /// handler matches, the message is pushed onto the dead-letter queue (evicting
    /// the oldest entry if the queue is full).
    ///
    /// Messages that are already expired on arrival are counted as
    /// `total_expired_dropped` and discarded immediately.
    pub fn route(&self, msg: RoutedMessage) -> usize {
        self.stats.total_routed.fetch_add(1, AOrdering::Relaxed);

        let now = Instant::now();
        if msg.is_expired(now) {
            self.stats
                .total_expired_dropped
                .fetch_add(1, AOrdering::Relaxed);
            return 0;
        }

        let handlers = self.handlers.read().unwrap_or_else(|e| e.into_inner());

        let matching: Vec<String> = handlers
            .iter()
            .filter(|h| h.matches(&msg.topic))
            .map(|h| h.handler_id.clone())
            .collect();

        let matched_count = matching.len();

        if matched_count == 0 {
            self.stats
                .total_dead_lettered
                .fetch_add(1, AOrdering::Relaxed);

            let mut dl = self.dead_letter.lock().unwrap_or_else(|e| e.into_inner());
            if dl.len() >= MAX_DEAD_LETTER {
                dl.pop_front();
            }
            dl.push_back(msg);
        } else {
            self.stats.total_matched.fetch_add(1, AOrdering::Relaxed);

            let mut queues = self.queues.lock().unwrap_or_else(|e| e.into_inner());

            for handler_id in &matching {
                let queue = queues.entry(handler_id.clone()).or_default();
                queue.push(QueuedMessage(msg.clone()));
            }
        }

        matched_count
    }

    // ── Message retrieval ────────────────────────────────────────────────────

    /// Drain all queued messages for `handler_id` in priority order (highest first).
    ///
    /// Returns an empty `Vec` if the handler has no queued messages or is unknown.
    pub fn drain_for_handler(&self, handler_id: &str) -> Vec<RoutedMessage> {
        let mut queues = self.queues.lock().unwrap_or_else(|e| e.into_inner());

        match queues.get_mut(handler_id) {
            None => Vec::new(),
            Some(heap) => {
                let mut out = Vec::with_capacity(heap.len());
                while let Some(qm) = heap.pop() {
                    out.push(qm.0);
                }
                out
            }
        }
    }

    // ── Dead-letter queue ────────────────────────────────────────────────────

    /// Return the current number of messages in the dead-letter queue.
    pub fn dead_letter_count(&self) -> usize {
        self.dead_letter
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .len()
    }

    /// Drain and return all messages from the dead-letter queue.
    pub fn drain_dead_letter(&self) -> Vec<RoutedMessage> {
        let mut dl = self.dead_letter.lock().unwrap_or_else(|e| e.into_inner());
        dl.drain(..).collect()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// PeerMessageRouter — subscription-aware peer message router
// ═══════════════════════════════════════════════════════════════════════════════

/// Classification of a message moving through the [`PeerMessageRouter`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MessageType {
    /// Request for a specific block.
    BlockRequest,
    /// Response carrying block data.
    BlockResponse,
    /// Request for a validity proof.
    ProofRequest,
    /// Response carrying proof data.
    ProofResponse,
    /// Gossip message on a named topic.
    Gossip { topic: String },
    /// Control-plane message (e.g., keep-alive, disconnect).
    Control,
}

/// A message that has been routed (or queued for routing) by [`PeerMessageRouter`].
#[derive(Debug, Clone)]
pub struct PeerRoutedMessage {
    /// Monotonically increasing message identifier assigned by the router.
    pub msg_id: u64,
    /// Type (and topic, for gossip) of the message.
    pub msg_type: MessageType,
    /// Originating peer identifier.
    pub from_peer: String,
    /// Destination peer identifier; `None` means the message was broadcast.
    pub to_peer: Option<String>,
    /// Size of the raw payload in bytes.
    pub payload_bytes: u64,
    /// Logical clock tick at which the message was created.
    pub created_at_tick: u64,
}

/// A rule that influences how the [`PeerMessageRouter`] dispatches messages.
///
/// Rules are evaluated in descending priority order.  The first matching rule
/// determines whether/how the message is forwarded.
#[derive(Debug, Clone)]
pub struct RouteRule {
    /// Unique identifier for this rule.
    pub rule_id: u64,
    /// The [`MessageType`] that this rule matches.
    ///
    /// For [`MessageType::Gossip`], the topic field is ignored — any gossip
    /// message matches regardless of topic.
    pub match_type: MessageType,
    /// When `Some`, the rule only applies when the destination peer equals this
    /// value.  `None` means the rule applies to any peer.
    pub target_peer: Option<String>,
    /// Higher-priority rules are evaluated first.
    pub priority: u32,
}

/// Aggregate statistics for a [`PeerMessageRouter`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MessageRouterStats {
    /// Total messages routed directly (to_peer was Some).
    pub total_routed: u64,
    /// Total messages broadcast (to_peer was None and at least one recipient
    /// was found).
    pub total_broadcast: u64,
    /// Total messages dropped because no recipient could be determined.
    pub total_dropped: u64,
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

/// Returns `true` when `a` and `b` are the same variant, ignoring inner data.
fn same_message_type_variant(a: &MessageType, b: &MessageType) -> bool {
    match (a, b) {
        (MessageType::BlockRequest, MessageType::BlockRequest) => true,
        (MessageType::BlockResponse, MessageType::BlockResponse) => true,
        (MessageType::ProofRequest, MessageType::ProofRequest) => true,
        (MessageType::ProofResponse, MessageType::ProofResponse) => true,
        // Gossip matches any Gossip regardless of topic.
        (MessageType::Gossip { .. }, MessageType::Gossip { .. }) => true,
        (MessageType::Control, MessageType::Control) => true,
        _ => false,
    }
}

// ─── PeerMessageRouter ────────────────────────────────────────────────────────

/// Routes messages to appropriate peer handlers based on message type and topic
/// subscriptions, supporting both direct and broadcast routing.
///
/// # Routing semantics
///
/// - **Direct** (`to_peer = Some`): a single [`PeerRoutedMessage`] is produced
///   addressed to the specified peer.
/// - **Broadcast** (`to_peer = None`):
///   - [`MessageType::Gossip`]: delivered to all peers subscribed to the gossip
///     topic, excluding `from_peer`.
///   - All other types: delivered to all peers that appear as keys in the
///     subscription map *and* match at least one rule with `target_peer = None`.
///   - If no recipients are found the message is counted as dropped.
pub struct PeerMessageRouter {
    /// Routing rules kept sorted by priority descending.
    pub rules: Vec<RouteRule>,
    /// Topic → list of subscribed peer IDs.
    pub subscriptions: HashMap<String, Vec<String>>,
    /// Next message ID to assign (post-increment semantics).
    pub next_msg_id: u64,
    /// Aggregate statistics.
    pub stats: MessageRouterStats,
}

impl Default for PeerMessageRouter {
    fn default() -> Self {
        Self::new()
    }
}

impl PeerMessageRouter {
    /// Create a new, empty [`PeerMessageRouter`].
    pub fn new() -> Self {
        Self {
            rules: Vec::new(),
            subscriptions: HashMap::new(),
            next_msg_id: 0,
            stats: MessageRouterStats::default(),
        }
    }

    // ── Rule management ───────────────────────────────────────────────────────

    /// Insert `rule`, maintaining the invariant that `self.rules` is sorted by
    /// `priority` descending.
    pub fn add_rule(&mut self, rule: RouteRule) {
        // Binary-search for the insertion position so we stay O(n) in the worst
        // case but avoid a full sort on every insertion.
        let pos = self.rules.partition_point(|r| r.priority >= rule.priority);
        self.rules.insert(pos, rule);
    }

    /// Remove the rule with the given `rule_id`.
    ///
    /// Returns `true` if a rule was found and removed, `false` otherwise.
    pub fn remove_rule(&mut self, rule_id: u64) -> bool {
        let before = self.rules.len();
        self.rules.retain(|r| r.rule_id != rule_id);
        self.rules.len() < before
    }

    // ── Subscription management ───────────────────────────────────────────────

    /// Subscribe `peer_id` to `topic`.
    ///
    /// If the peer is already subscribed this is a no-op.
    pub fn subscribe(&mut self, peer_id: &str, topic: &str) {
        let peers = self.subscriptions.entry(topic.to_string()).or_default();

        if !peers.iter().any(|p| p == peer_id) {
            peers.push(peer_id.to_string());
        }
    }

    /// Unsubscribe `peer_id` from `topic`.
    ///
    /// Returns `true` if the peer was subscribed, `false` if it was not.
    pub fn unsubscribe(&mut self, peer_id: &str, topic: &str) -> bool {
        if let Some(peers) = self.subscriptions.get_mut(topic) {
            let before = peers.len();
            peers.retain(|p| p != peer_id);
            return peers.len() < before;
        }
        false
    }

    // ── Routing ───────────────────────────────────────────────────────────────

    /// Route a message, returning one [`PeerRoutedMessage`] per recipient.
    ///
    /// `msg_id` is assigned from `next_msg_id` (pre-increment — the assigned
    /// value is used, then `next_msg_id` is incremented).
    pub fn route(
        &mut self,
        msg_type: MessageType,
        from_peer: &str,
        to_peer: Option<&str>,
        payload_bytes: u64,
        tick: u64,
    ) -> Vec<PeerRoutedMessage> {
        let msg_id = self.next_msg_id;
        self.next_msg_id = self.next_msg_id.saturating_add(1);

        match to_peer {
            // ── Direct routing ────────────────────────────────────────────────
            Some(dest) => {
                self.stats.total_routed = self.stats.total_routed.saturating_add(1);
                vec![PeerRoutedMessage {
                    msg_id,
                    msg_type,
                    from_peer: from_peer.to_string(),
                    to_peer: Some(dest.to_string()),
                    payload_bytes,
                    created_at_tick: tick,
                }]
            }

            // ── Broadcast routing ─────────────────────────────────────────────
            None => {
                let recipients: Vec<String> = match &msg_type {
                    MessageType::Gossip { topic } => {
                        // Route to all subscribers of the specific topic, excluding
                        // the originating peer.
                        self.subscriptions
                            .get(topic.as_str())
                            .map(|peers| {
                                peers
                                    .iter()
                                    .filter(|p| p.as_str() != from_peer)
                                    .cloned()
                                    .collect()
                            })
                            .unwrap_or_default()
                    }
                    other => {
                        // For non-gossip broadcasts: find all peers (keys of
                        // subscriptions) that match at least one rule with
                        // target_peer = None.
                        let has_wildcard_rule = self.rules.iter().any(|r| {
                            r.target_peer.is_none()
                                && same_message_type_variant(&r.match_type, other)
                        });

                        if has_wildcard_rule {
                            // Collect unique peer IDs across all topics.
                            let mut peers: Vec<String> = self
                                .subscriptions
                                .values()
                                .flat_map(|ps| ps.iter().cloned())
                                .collect();
                            peers.sort();
                            peers.dedup();
                            peers
                        } else {
                            Vec::new()
                        }
                    }
                };

                if recipients.is_empty() {
                    self.stats.total_dropped = self.stats.total_dropped.saturating_add(1);
                    Vec::new()
                } else {
                    self.stats.total_broadcast = self.stats.total_broadcast.saturating_add(1);
                    recipients
                        .into_iter()
                        .map(|peer| PeerRoutedMessage {
                            msg_id,
                            msg_type: msg_type.clone(),
                            from_peer: from_peer.to_string(),
                            to_peer: Some(peer),
                            payload_bytes,
                            created_at_tick: tick,
                        })
                        .collect()
                }
            }
        }
    }

    // ── Query helpers ─────────────────────────────────────────────────────────

    /// Return all peer IDs subscribed to `topic`, sorted lexicographically.
    pub fn subscribers(&self, topic: &str) -> Vec<&str> {
        let mut peers: Vec<&str> = self
            .subscriptions
            .get(topic)
            .map(|ps| ps.iter().map(String::as_str).collect())
            .unwrap_or_default();
        peers.sort_unstable();
        peers
    }

    /// Return a reference to the current statistics.
    pub fn stats(&self) -> &MessageRouterStats {
        &self.stats
    }
}

// ─── Tests (PeerMessageRouter) ────────────────────────────────────────────────

#[cfg(test)]
mod peer_router_tests {
    use super::{MessageRouterStats, MessageType, PeerMessageRouter, RouteRule};

    fn make_rule(rule_id: u64, match_type: MessageType, priority: u32) -> RouteRule {
        RouteRule {
            rule_id,
            match_type,
            target_peer: None,
            priority,
        }
    }

    // 1. new() starts empty
    #[test]
    fn test_new_starts_empty() {
        let router = PeerMessageRouter::new();
        assert!(router.rules.is_empty());
        assert!(router.subscriptions.is_empty());
        assert_eq!(router.next_msg_id, 0);
        assert_eq!(router.stats, MessageRouterStats::default());
    }

    // 2. add_rule maintains priority order (descending)
    #[test]
    fn test_add_rule_maintains_priority_order() {
        let mut router = PeerMessageRouter::new();
        router.add_rule(make_rule(1, MessageType::BlockRequest, 10));
        router.add_rule(make_rule(2, MessageType::BlockResponse, 50));
        router.add_rule(make_rule(3, MessageType::Control, 30));

        assert_eq!(router.rules[0].priority, 50);
        assert_eq!(router.rules[1].priority, 30);
        assert_eq!(router.rules[2].priority, 10);
    }

    // 3. remove_rule returns true when found
    #[test]
    fn test_remove_rule_found() {
        let mut router = PeerMessageRouter::new();
        router.add_rule(make_rule(42, MessageType::Control, 5));
        assert!(router.remove_rule(42));
        assert!(router.rules.is_empty());
    }

    // 4. remove_rule returns false when not found
    #[test]
    fn test_remove_rule_not_found() {
        let mut router = PeerMessageRouter::new();
        assert!(!router.remove_rule(99));
    }

    // 5. subscribe adds peer to topic
    #[test]
    fn test_subscribe_adds_peer() {
        let mut router = PeerMessageRouter::new();
        router.subscribe("peer-A", "news");
        let subs = router.subscribers("news");
        assert_eq!(subs, vec!["peer-A"]);
    }

    // 6. subscribe is idempotent (no duplicates)
    #[test]
    fn test_subscribe_idempotent() {
        let mut router = PeerMessageRouter::new();
        router.subscribe("peer-A", "news");
        router.subscribe("peer-A", "news");
        assert_eq!(router.subscribers("news").len(), 1);
    }

    // 7. unsubscribe removes peer and returns true
    #[test]
    fn test_unsubscribe_removes_peer() {
        let mut router = PeerMessageRouter::new();
        router.subscribe("peer-A", "news");
        let removed = router.unsubscribe("peer-A", "news");
        assert!(removed);
        assert!(router.subscribers("news").is_empty());
    }

    // 8. unsubscribe returns false when peer is not subscribed
    #[test]
    fn test_unsubscribe_not_subscribed() {
        let mut router = PeerMessageRouter::new();
        assert!(!router.unsubscribe("nobody", "news"));
    }

    // 9. unsubscribe returns false on unknown topic
    #[test]
    fn test_unsubscribe_unknown_topic() {
        let mut router = PeerMessageRouter::new();
        router.subscribe("peer-A", "news");
        assert!(!router.unsubscribe("peer-A", "sports"));
    }

    // 10. route direct (to_peer Some) creates exactly one message
    #[test]
    fn test_route_direct_single_message() {
        let mut router = PeerMessageRouter::new();
        let msgs = router.route(
            MessageType::BlockRequest,
            "sender",
            Some("receiver"),
            128,
            1,
        );
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].to_peer, Some("receiver".to_string()));
    }

    // 11. route direct increments total_routed
    #[test]
    fn test_route_direct_increments_total_routed() {
        let mut router = PeerMessageRouter::new();
        router.route(MessageType::BlockRequest, "s", Some("r"), 0, 0);
        router.route(MessageType::BlockResponse, "s", Some("r"), 0, 0);
        assert_eq!(router.stats().total_routed, 2);
        assert_eq!(router.stats().total_broadcast, 0);
    }

    // 12. route broadcast Gossip goes to topic subscribers
    #[test]
    fn test_route_broadcast_gossip_to_subscribers() {
        let mut router = PeerMessageRouter::new();
        router.subscribe("peer-A", "chain");
        router.subscribe("peer-B", "chain");

        let msgs = router.route(
            MessageType::Gossip {
                topic: "chain".to_string(),
            },
            "sender",
            None,
            64,
            5,
        );
        assert_eq!(msgs.len(), 2);
    }

    // 13. route broadcast Gossip excludes from_peer
    #[test]
    fn test_route_broadcast_gossip_excludes_sender() {
        let mut router = PeerMessageRouter::new();
        router.subscribe("peer-A", "chain");
        router.subscribe("origin", "chain");

        let msgs = router.route(
            MessageType::Gossip {
                topic: "chain".to_string(),
            },
            "origin",
            None,
            64,
            1,
        );
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].to_peer, Some("peer-A".to_string()));
    }

    // 14. route broadcast Gossip with no other subscribers drops message
    #[test]
    fn test_route_broadcast_gossip_no_subscribers_dropped() {
        let mut router = PeerMessageRouter::new();
        // Only the sender is subscribed; result should be empty.
        router.subscribe("origin", "chain");

        let msgs = router.route(
            MessageType::Gossip {
                topic: "chain".to_string(),
            },
            "origin",
            None,
            0,
            0,
        );
        assert!(msgs.is_empty());
        assert_eq!(router.stats().total_dropped, 1);
    }

    // 15. route broadcast no subscribers at all drops message
    #[test]
    fn test_route_broadcast_no_subscribers_drops() {
        let mut router = PeerMessageRouter::new();
        let msgs = router.route(MessageType::BlockRequest, "s", None, 0, 0);
        assert!(msgs.is_empty());
        assert_eq!(router.stats().total_dropped, 1);
    }

    // 16. route broadcast non-gossip with matching wildcard rule routes to all peers
    #[test]
    fn test_route_broadcast_non_gossip_with_rule() {
        let mut router = PeerMessageRouter::new();
        router.add_rule(make_rule(1, MessageType::BlockRequest, 10));
        router.subscribe("peer-A", "any-topic");
        router.subscribe("peer-B", "other-topic");

        let msgs = router.route(MessageType::BlockRequest, "s", None, 0, 0);
        assert_eq!(msgs.len(), 2);
        assert_eq!(router.stats().total_broadcast, 1);
    }

    // 17. route broadcast increments total_broadcast (not total_routed)
    #[test]
    fn test_route_broadcast_increments_total_broadcast() {
        let mut router = PeerMessageRouter::new();
        router.subscribe("peer-A", "t");

        router.route(
            MessageType::Gossip {
                topic: "t".to_string(),
            },
            "s",
            None,
            0,
            0,
        );
        assert_eq!(router.stats().total_broadcast, 1);
        assert_eq!(router.stats().total_routed, 0);
    }

    // 18. route dropped increments total_dropped
    #[test]
    fn test_route_dropped_increments_total_dropped() {
        let mut router = PeerMessageRouter::new();
        router.route(MessageType::Control, "s", None, 0, 0);
        router.route(MessageType::Control, "s", None, 0, 0);
        assert_eq!(router.stats().total_dropped, 2);
    }

    // 19. msg_id is monotonically increasing across calls
    #[test]
    fn test_msg_id_monotonically_increasing() {
        let mut router = PeerMessageRouter::new();
        let r1 = router.route(MessageType::BlockRequest, "s", Some("r"), 0, 0);
        let r2 = router.route(MessageType::BlockResponse, "s", Some("r"), 0, 0);
        let r3 = router.route(MessageType::Control, "s", Some("r"), 0, 0);
        assert!(r1[0].msg_id < r2[0].msg_id);
        assert!(r2[0].msg_id < r3[0].msg_id);
    }

    // 20. PeerRoutedMessage fields set correctly for direct route
    #[test]
    fn test_routed_message_fields_direct() {
        let mut router = PeerMessageRouter::new();
        let msgs = router.route(MessageType::ProofRequest, "alice", Some("bob"), 256, 42);
        let m = &msgs[0];
        assert_eq!(m.msg_id, 0);
        assert_eq!(m.msg_type, MessageType::ProofRequest);
        assert_eq!(m.from_peer, "alice");
        assert_eq!(m.to_peer, Some("bob".to_string()));
        assert_eq!(m.payload_bytes, 256);
        assert_eq!(m.created_at_tick, 42);
    }

    // 21. subscribers() returns sorted list
    #[test]
    fn test_subscribers_sorted() {
        let mut router = PeerMessageRouter::new();
        router.subscribe("charlie", "topic");
        router.subscribe("alice", "topic");
        router.subscribe("bob", "topic");

        let subs = router.subscribers("topic");
        assert_eq!(subs, vec!["alice", "bob", "charlie"]);
    }

    // 22. multiple subscribers all receive broadcast
    #[test]
    fn test_multiple_subscribers_all_receive_broadcast() {
        let mut router = PeerMessageRouter::new();
        for i in 0..5 {
            router.subscribe(&format!("peer-{}", i), "t");
        }
        let msgs = router.route(
            MessageType::Gossip {
                topic: "t".to_string(),
            },
            "outsider",
            None,
            10,
            1,
        );
        assert_eq!(msgs.len(), 5);
    }

    // 23. stats total_routed and total_broadcast are independent
    #[test]
    fn test_stats_independence() {
        let mut router = PeerMessageRouter::new();
        router.subscribe("p", "t");

        router.route(MessageType::BlockRequest, "s", Some("r"), 0, 0); // direct
        router.route(
            MessageType::Gossip {
                topic: "t".to_string(),
            },
            "s",
            None,
            0,
            0,
        ); // broadcast

        let s = router.stats();
        assert_eq!(s.total_routed, 1);
        assert_eq!(s.total_broadcast, 1);
        assert_eq!(s.total_dropped, 0);
    }

    // 24. remove_rule only removes the matching rule
    #[test]
    fn test_remove_rule_selective() {
        let mut router = PeerMessageRouter::new();
        router.add_rule(make_rule(1, MessageType::BlockRequest, 10));
        router.add_rule(make_rule(2, MessageType::BlockResponse, 20));
        router.add_rule(make_rule(3, MessageType::Control, 30));

        assert!(router.remove_rule(2));
        assert_eq!(router.rules.len(), 2);
        assert!(router.rules.iter().all(|r| r.rule_id != 2));
    }

    // 25. default() matches new()
    #[test]
    fn test_default_matches_new() {
        let a = PeerMessageRouter::new();
        let b = PeerMessageRouter::default();
        assert_eq!(a.rules.len(), b.rules.len());
        assert_eq!(a.next_msg_id, b.next_msg_id);
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn make_msg(topic: &str, priority: MessagePriority) -> RoutedMessage {
        RoutedMessage {
            id: 0,
            from_peer: "peer-1".to_string(),
            topic: topic.to_string(),
            payload: b"hello".to_vec(),
            priority,
            enqueued_at: Instant::now(),
            ttl: DEFAULT_TTL,
        }
    }

    fn reg(pattern: &str, priority: MessagePriority, id: &str) -> HandlerRegistration {
        HandlerRegistration {
            topic_pattern: pattern.to_string(),
            priority,
            handler_id: id.to_string(),
        }
    }

    // 1. Register and route exact match
    #[test]
    fn test_register_and_route_exact_match() {
        let router = MessageRouter::new();
        router
            .register_handler(reg("blocks", MessagePriority::Normal, "h1"))
            .expect("register failed");

        let matched = router.route(make_msg("blocks", MessagePriority::Normal));
        assert_eq!(matched, 1);
        assert_eq!(router.dead_letter_count(), 0);
    }

    // 2. Exact match does NOT match a different topic
    #[test]
    fn test_exact_match_no_false_positive() {
        let router = MessageRouter::new();
        router
            .register_handler(reg("blocks", MessagePriority::Normal, "h1"))
            .expect("register failed");

        let matched = router.route(make_msg("transactions", MessagePriority::Normal));
        assert_eq!(matched, 0);
        assert_eq!(router.dead_letter_count(), 1);
    }

    // 3. Prefix wildcard "block.*" matching
    #[test]
    fn test_wildcard_prefix_matching() {
        let router = MessageRouter::new();
        router
            .register_handler(reg("block.*", MessagePriority::Normal, "h1"))
            .expect("register failed");

        assert_eq!(
            router.route(make_msg("block.add", MessagePriority::Normal)),
            1
        );
        assert_eq!(
            router.route(make_msg("block.get", MessagePriority::Normal)),
            1
        );
        assert_eq!(
            router.route(make_msg("transaction.add", MessagePriority::Normal)),
            0
        );
    }

    // 4. Catch-all "*" matches every topic
    #[test]
    fn test_catch_all_wildcard() {
        let router = MessageRouter::new();
        router
            .register_handler(reg("*", MessagePriority::Normal, "h1"))
            .expect("register failed");

        assert_eq!(
            router.route(make_msg("anything", MessagePriority::Normal)),
            1
        );
        assert_eq!(
            router.route(make_msg("block.add", MessagePriority::Normal)),
            1
        );
        assert_eq!(router.route(make_msg("", MessagePriority::Normal)), 1);
        assert_eq!(router.dead_letter_count(), 0);
    }

    // 5. Unmatched message goes to dead-letter queue
    #[test]
    fn test_unmatched_goes_to_dead_letter() {
        let router = MessageRouter::new();
        router.route(make_msg("orphan", MessagePriority::Normal));
        assert_eq!(router.dead_letter_count(), 1);
        let dl = router.drain_dead_letter();
        assert_eq!(dl.len(), 1);
        assert_eq!(dl[0].topic, "orphan");
    }

    // 6. drain_for_handler returns correct messages
    #[test]
    fn test_drain_for_handler() {
        let router = MessageRouter::new();
        router
            .register_handler(reg("t", MessagePriority::Normal, "h1"))
            .expect("register failed");

        router.route(make_msg("t", MessagePriority::Normal));
        router.route(make_msg("t", MessagePriority::Normal));

        let msgs = router.drain_for_handler("h1");
        assert_eq!(msgs.len(), 2);

        // Queue is now empty
        assert!(router.drain_for_handler("h1").is_empty());
    }

    // 7. Priority ordering within handler queue
    #[test]
    fn test_priority_ordering_in_queue() {
        let router = MessageRouter::new();
        router
            .register_handler(reg("t", MessagePriority::Low, "h1"))
            .expect("register failed");

        let mut low = make_msg("t", MessagePriority::Low);
        low.id = 1;
        let mut high = make_msg("t", MessagePriority::High);
        high.id = 2;
        let mut critical = make_msg("t", MessagePriority::Critical);
        critical.id = 3;
        let mut normal = make_msg("t", MessagePriority::Normal);
        normal.id = 4;

        router.route(low);
        router.route(high);
        router.route(critical);
        router.route(normal);

        let msgs = router.drain_for_handler("h1");
        assert_eq!(msgs.len(), 4);
        // Should come out Critical, High, Normal, Low
        assert_eq!(msgs[0].priority, MessagePriority::Critical);
        assert_eq!(msgs[1].priority, MessagePriority::High);
        assert_eq!(msgs[2].priority, MessagePriority::Normal);
        assert_eq!(msgs[3].priority, MessagePriority::Low);
    }

    // 8. Multiple handlers for the same topic all receive the message
    #[test]
    fn test_multiple_handlers_all_receive() {
        let router = MessageRouter::new();
        router
            .register_handler(reg("t", MessagePriority::Normal, "h1"))
            .expect("register h1");
        router
            .register_handler(reg("t", MessagePriority::High, "h2"))
            .expect("register h2");
        router
            .register_handler(reg("t", MessagePriority::Low, "h3"))
            .expect("register h3");

        let matched = router.route(make_msg("t", MessagePriority::Normal));
        assert_eq!(matched, 3);

        assert_eq!(router.drain_for_handler("h1").len(), 1);
        assert_eq!(router.drain_for_handler("h2").len(), 1);
        assert_eq!(router.drain_for_handler("h3").len(), 1);
    }

    // 9. Duplicate handler registration returns an error
    #[test]
    fn test_duplicate_handler_error() {
        let router = MessageRouter::new();
        router
            .register_handler(reg("t", MessagePriority::Normal, "h1"))
            .expect("first register");

        let result = router.register_handler(reg("t", MessagePriority::High, "h1"));
        assert!(matches!(result, Err(RouterError::DuplicateHandler(ref id)) if id == "h1"));
    }

    // 10. drain_dead_letter clears the queue
    #[test]
    fn test_drain_dead_letter_clears_queue() {
        let router = MessageRouter::new();
        router.route(make_msg("orphan1", MessagePriority::Normal));
        router.route(make_msg("orphan2", MessagePriority::Normal));

        assert_eq!(router.dead_letter_count(), 2);
        let dl = router.drain_dead_letter();
        assert_eq!(dl.len(), 2);
        assert_eq!(router.dead_letter_count(), 0);
    }

    // 11. Stats accumulation
    #[test]
    fn test_stats_accumulation() {
        let router = MessageRouter::new();
        router
            .register_handler(reg("t", MessagePriority::Normal, "h1"))
            .expect("register");

        router.route(make_msg("t", MessagePriority::Normal)); // matched
        router.route(make_msg("other", MessagePriority::Normal)); // dead-lettered

        let snap = router.stats.snapshot();
        assert_eq!(snap.total_routed, 2);
        assert_eq!(snap.total_matched, 1);
        assert_eq!(snap.total_dead_lettered, 1);
        assert_eq!(snap.total_expired_dropped, 0);
    }

    // 12. Expired message detection
    #[test]
    fn test_expired_message_detection() {
        let msg = RoutedMessage {
            id: 0,
            from_peer: "p".to_string(),
            topic: "t".to_string(),
            payload: vec![],
            priority: MessagePriority::Normal,
            enqueued_at: Instant::now() - Duration::from_secs(60),
            ttl: Duration::from_secs(30),
        };
        assert!(msg.is_expired(Instant::now()));
    }

    // 13. Non-expired message
    #[test]
    fn test_non_expired_message() {
        let msg = make_msg("t", MessagePriority::Normal);
        assert!(!msg.is_expired(Instant::now()));
    }

    // 14. Expired message on arrival is dropped (not dead-lettered)
    #[test]
    fn test_expired_on_arrival_dropped() {
        let router = MessageRouter::new();
        let msg = RoutedMessage {
            id: 0,
            from_peer: "p".to_string(),
            topic: "orphan".to_string(),
            payload: vec![],
            priority: MessagePriority::Normal,
            enqueued_at: Instant::now() - Duration::from_secs(60),
            ttl: Duration::from_secs(30),
        };
        let matched = router.route(msg);
        assert_eq!(matched, 0);
        assert_eq!(router.dead_letter_count(), 0);

        let snap = router.stats.snapshot();
        assert_eq!(snap.total_expired_dropped, 1);
        assert_eq!(snap.total_dead_lettered, 0);
    }

    // 15. Dead-letter queue is bounded at MAX_DEAD_LETTER
    #[test]
    fn test_dead_letter_bounded() {
        let router = MessageRouter::new();

        for _ in 0..=MAX_DEAD_LETTER + 10 {
            router.route(make_msg("orphan", MessagePriority::Normal));
        }

        assert_eq!(router.dead_letter_count(), MAX_DEAD_LETTER);
    }

    // 16. handler_count reflects registrations and unregistrations
    #[test]
    fn test_handler_count() {
        let router = MessageRouter::new();
        assert_eq!(router.handler_count(), 0);

        router
            .register_handler(reg("a", MessagePriority::Normal, "h1"))
            .expect("register h1");
        router
            .register_handler(reg("b", MessagePriority::Normal, "h2"))
            .expect("register h2");
        assert_eq!(router.handler_count(), 2);

        router.unregister_handler("h1");
        assert_eq!(router.handler_count(), 1);
    }

    // 17. Priority ordering of handlers (higher-priority handler stored first)
    #[test]
    fn test_handlers_sorted_by_priority() {
        let router = MessageRouter::new();
        router
            .register_handler(reg("t", MessagePriority::Low, "low"))
            .expect("register low");
        router
            .register_handler(reg("t", MessagePriority::Critical, "critical"))
            .expect("register critical");
        router
            .register_handler(reg("t", MessagePriority::Normal, "normal"))
            .expect("register normal");

        let handlers = router.handlers.read().expect("lock");
        assert_eq!(handlers[0].handler_id, "critical");
        assert_eq!(handlers[1].handler_id, "normal");
        assert_eq!(handlers[2].handler_id, "low");
    }

    // 18. drain_for_handler on unknown handler returns empty vec
    #[test]
    fn test_drain_unknown_handler_empty() {
        let router = MessageRouter::new();
        assert!(router.drain_for_handler("nonexistent").is_empty());
    }

    // 19. MessagePriority ordering
    #[test]
    fn test_message_priority_ordering() {
        assert!(MessagePriority::Critical > MessagePriority::High);
        assert!(MessagePriority::High > MessagePriority::Normal);
        assert!(MessagePriority::Normal > MessagePriority::Low);
    }

    // 20. next_id is monotonically increasing
    #[test]
    fn test_next_id_monotonic() {
        let router = MessageRouter::new();
        let id0 = router.next_id();
        let id1 = router.next_id();
        let id2 = router.next_id();
        assert!(id0 < id1);
        assert!(id1 < id2);
    }
}
