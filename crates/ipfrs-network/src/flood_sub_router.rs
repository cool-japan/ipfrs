//! FloodSubRouter — topic-based message flooding router with subscription
//! management, message deduplication, TTL-based expiry, and per-peer forwarding
//! history to prevent routing loops.
//!
//! ## Design
//!
//! - **Topic subscriptions**: many peers → one topic via `FloodTopic`.
//! - **Deduplication**: every seen `FloodMessageId` is cached with its arrival
//!   timestamp for `dedup_window_secs`; duplicates are immediately dropped.
//! - **Loop prevention**: `forwarded_to` tracks which peers already received a
//!   given message so the router never re-sends to the same peer.
//! - **TTL enforcement**: messages whose TTL falls at or below `min_ttl` are
//!   dropped before any forwarding occurs.
//! - **No `unwrap()`**: all fallible operations are handled explicitly.

use std::collections::{HashMap, HashSet};

// ─── FNV-1a ──────────────────────────────────────────────────────────────────

/// FNV-1a 64-bit hash over an arbitrary byte slice.
///
/// Used for `FloodMessageId` computation without requiring any external crate.
#[inline]
fn fnv1a_64(data: &[u8]) -> u64 {
    let mut hash: u64 = 14_695_981_039_346_656_037;
    for &b in data {
        hash ^= b as u64;
        hash = hash.wrapping_mul(1_099_511_628_211);
    }
    hash
}

// ─── xorshift64 PRNG ─────────────────────────────────────────────────────────

/// xorshift64 PRNG — used wherever deterministic pseudo-randomness is needed
/// inside this module (e.g. test helpers) without pulling in external crates.
#[allow(dead_code)]
#[inline]
fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

// ═══════════════════════════════════════════════════════════════════════════════
// Core types
// ═══════════════════════════════════════════════════════════════════════════════

/// Newtype wrapper for a topic identifier string.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FloodTopic(pub String);

impl FloodTopic {
    /// Create a new `FloodTopic` from any string-like value.
    pub fn new(s: impl Into<String>) -> Self {
        FloodTopic(s.into())
    }

    /// Return the inner topic string as a `&str`.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for FloodTopic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

// ─────────────────────────────────────────────────────────────────────────────

/// 16-byte message ID derived from FNV-1a over content + timestamp + peer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FloodMessageId(pub [u8; 16]);

impl FloodMessageId {
    /// Return the raw bytes of this ID.
    pub fn as_bytes(&self) -> &[u8; 16] {
        &self.0
    }

    /// Return a hex-encoded representation.
    pub fn to_hex(&self) -> String {
        self.0.iter().map(|b| format!("{b:02x}")).collect()
    }
}

impl std::fmt::Display for FloodMessageId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_hex())
    }
}

// ─────────────────────────────────────────────────────────────────────────────

/// A flood-routed message with all routing metadata.
#[derive(Debug, Clone)]
pub struct FloodMessage {
    /// Unique message identifier (derived via FNV-1a).
    pub id: FloodMessageId,
    /// Topic this message belongs to.
    pub topic: FloodTopic,
    /// Application-level payload bytes.
    pub payload: Vec<u8>,
    /// Time-to-live in hops; decremented by the receiving router before
    /// forwarding.  When `ttl <= min_ttl` the message is dropped.
    pub ttl: u8,
    /// Peer ID of the originator of this message.
    pub origin_peer: String,
    /// Unix-epoch timestamp (seconds) at message creation.
    pub created_at: u64,
}

// ─────────────────────────────────────────────────────────────────────────────

/// Record of a peer's subscription to a topic.
#[derive(Debug, Clone)]
pub struct SubscriptionRecord {
    /// Peer that subscribed.
    pub peer_id: String,
    /// The topic subscribed to.
    pub topic: FloodTopic,
    /// Unix-epoch timestamp when the subscription was registered.
    pub subscribed_at: u64,
    /// Cumulative count of messages delivered to this subscriber.
    pub message_count: u64,
}

// ─────────────────────────────────────────────────────────────────────────────

/// Decision returned by [`FloodSubRouter::route`].
#[derive(Debug, Clone, PartialEq)]
pub enum ForwardDecision {
    /// Forward this message to the listed peers.
    Forward {
        /// Peers that should receive this message next.
        to_peers: Vec<String>,
    },
    /// Drop the message; `reason` explains why.
    Drop {
        /// Human-readable reason for dropping.
        reason: String,
    },
    /// The origin peer is the only subscriber — the message "loops back" to
    /// its sender without needing further forwarding.
    Loopback,
}

// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for a [`FloodSubRouter`] instance.
#[derive(Debug, Clone)]
pub struct RouterConfig {
    /// Maximum number of peers allowed per topic.
    pub max_peers_per_topic: usize,
    /// Maximum number of message IDs held in the dedup cache.
    pub max_message_cache: usize,
    /// Minimum TTL value; messages at or below this are dropped.
    pub min_ttl: u8,
    /// Duration (seconds) for which a message ID is kept in the dedup cache.
    pub dedup_window_secs: u64,
    /// Maximum number of distinct topics.
    pub max_topics: usize,
}

impl Default for RouterConfig {
    fn default() -> Self {
        RouterConfig {
            max_peers_per_topic: 100,
            max_message_cache: 10_000,
            min_ttl: 1,
            dedup_window_secs: 60,
            max_topics: 256,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────

/// Snapshot of operational statistics for a [`FloodSubRouter`].
#[derive(Debug, Clone, Default)]
pub struct FsrRouterStats {
    /// Total messages forwarded to at least one peer.
    pub messages_forwarded: u64,
    /// Total messages dropped (duplicate, TTL, no-subscribers, etc.).
    pub messages_dropped: u64,
    /// Total messages returned as `Loopback`.
    pub messages_looped: u64,
    /// Current number of active subscriptions (peer × topic pairs).
    pub subscriptions: usize,
    /// Current number of distinct active topics.
    pub topics: usize,
    /// Current number of entries in the dedup cache.
    pub cache_size: usize,
}

// ═══════════════════════════════════════════════════════════════════════════════
// FloodSubRouter
// ═══════════════════════════════════════════════════════════════════════════════

/// Topic-based message flooding router.
///
/// # Responsibilities
///
/// 1. Maintain per-peer topic subscriptions.
/// 2. Deduplicate messages using a bounded, TTL-based ID cache.
/// 3. Track which peers have already received a given message to prevent loops.
/// 4. Compute forward sets excluding the origin peer and previously-forwarded
///    peers.
///
/// # Thread safety
///
/// `FloodSubRouter` is *not* `Send + Sync` by default; callers should wrap it
/// in a `Mutex` or `RwLock` when sharing across threads.
pub struct FloodSubRouter {
    /// Runtime configuration.
    config: RouterConfig,
    /// Active subscriptions keyed by `"peer_id:topic"`.
    subscriptions: HashMap<String, SubscriptionRecord>,
    /// Dedup cache: message ID → arrival timestamp (seconds).
    message_cache: HashMap<FloodMessageId, u64>,
    /// Per-message forward log: message ID → set of peer IDs already forwarded.
    forwarded_to: HashMap<FloodMessageId, HashSet<String>>,
    /// Operational statistics.
    stats: FsrRouterStats,
}

impl FloodSubRouter {
    // ── Construction ─────────────────────────────────────────────────────────

    /// Create a new router with the supplied configuration.
    pub fn new(config: RouterConfig) -> Self {
        FloodSubRouter {
            config,
            subscriptions: HashMap::new(),
            message_cache: HashMap::new(),
            forwarded_to: HashMap::new(),
            stats: FsrRouterStats::default(),
        }
    }

    // ── Subscription management ───────────────────────────────────────────────

    /// Subscribe `peer_id` to `topic`.
    ///
    /// Returns `true` if this was a *new* subscription, `false` if the peer
    /// was already subscribed to this topic.
    ///
    /// The subscription is rejected (returns `false`) when either:
    /// - the per-topic peer limit (`max_peers_per_topic`) would be exceeded, or
    /// - the global topic limit (`max_topics`) would be exceeded by a *new*
    ///   topic.
    pub fn subscribe(&mut self, peer_id: &str, topic: &FloodTopic, now: u64) -> bool {
        let key = Self::sub_key(peer_id, topic);
        if self.subscriptions.contains_key(&key) {
            return false;
        }

        // Enforce per-topic peer cap.
        let current_peers = self.peers_for_topic(topic).len();
        if current_peers >= self.config.max_peers_per_topic {
            return false;
        }

        // Enforce max-topics limit for brand-new topics.
        let topic_exists = self.subscriptions.values().any(|r| &r.topic == topic);
        if !topic_exists {
            let distinct_topics: HashSet<&FloodTopic> =
                self.subscriptions.values().map(|r| &r.topic).collect();
            if distinct_topics.len() >= self.config.max_topics {
                return false;
            }
        }

        let record = SubscriptionRecord {
            peer_id: peer_id.to_owned(),
            topic: topic.clone(),
            subscribed_at: now,
            message_count: 0,
        };
        self.subscriptions.insert(key, record);
        self.refresh_stats();
        true
    }

    /// Unsubscribe `peer_id` from `topic`.
    ///
    /// Returns `true` if the subscription existed and was removed.
    pub fn unsubscribe(&mut self, peer_id: &str, topic: &FloodTopic) -> bool {
        let key = Self::sub_key(peer_id, topic);
        let removed = self.subscriptions.remove(&key).is_some();
        if removed {
            self.refresh_stats();
        }
        removed
    }

    /// Return all peer IDs subscribed to `topic`.
    pub fn peers_for_topic(&self, topic: &FloodTopic) -> Vec<String> {
        self.subscriptions
            .values()
            .filter(|r| &r.topic == topic)
            .map(|r| r.peer_id.clone())
            .collect()
    }

    /// Return all topics to which `peer_id` is subscribed.
    pub fn topics_for_peer(&self, peer_id: &str) -> Vec<FloodTopic> {
        self.subscriptions
            .values()
            .filter(|r| r.peer_id == peer_id)
            .map(|r| r.topic.clone())
            .collect()
    }

    // ── Routing ───────────────────────────────────────────────────────────────

    /// Decide how to forward `msg` and update all internal state accordingly.
    ///
    /// # Decision logic (in order)
    ///
    /// 1. If `msg.id` is already in the dedup cache → `Drop` (duplicate).
    /// 2. If `msg.ttl <= config.min_ttl` → `Drop` (TTL exhausted).
    /// 3. If no peer is subscribed to `msg.topic` → `Drop` (no subscribers).
    /// 4. Insert `msg.id` into the dedup cache with `now` as the timestamp.
    /// 5. Build the candidate forward set: all subscribers except the origin
    ///    peer and peers already recorded in `forwarded_to[msg.id]`.
    /// 6. Record the candidate set in `forwarded_to`.
    /// 7. If the candidate set is empty and the origin is the *only* subscriber
    ///    → `Loopback`.
    /// 8. If the candidate set is empty (all already forwarded) → `Drop`.
    /// 9. Otherwise → `Forward { to_peers }`.
    pub fn route(&mut self, msg: &FloodMessage, now: u64) -> ForwardDecision {
        // Step 1 — dedup.
        if self.message_cache.contains_key(&msg.id) {
            self.stats.messages_dropped += 1;
            return ForwardDecision::Drop {
                reason: "duplicate: already in dedup cache".to_owned(),
            };
        }

        // Step 2 — TTL.
        if msg.ttl <= self.config.min_ttl {
            self.stats.messages_dropped += 1;
            return ForwardDecision::Drop {
                reason: format!("ttl {} <= min_ttl {}", msg.ttl, self.config.min_ttl),
            };
        }

        // Step 3 — subscribers.
        let subscribers = self.peers_for_topic(&msg.topic);
        if subscribers.is_empty() {
            self.stats.messages_dropped += 1;
            return ForwardDecision::Drop {
                reason: format!("no subscribers for topic '{}'", msg.topic),
            };
        }

        // Step 4 — register in dedup cache (bounded eviction when full).
        if self.message_cache.len() >= self.config.max_message_cache {
            // Evict the oldest entry to maintain the cap.
            if let Some(oldest_key) = self
                .message_cache
                .iter()
                .min_by_key(|(_, &ts)| ts)
                .map(|(k, _)| *k)
            {
                self.message_cache.remove(&oldest_key);
                self.forwarded_to.remove(&oldest_key);
            }
        }
        self.message_cache.insert(msg.id, now);

        // Step 5 — compute forward set.
        let already_forwarded = self.forwarded_to.get(&msg.id).cloned().unwrap_or_default();

        let to_peers: Vec<String> = subscribers
            .iter()
            .filter(|p| *p != &msg.origin_peer && !already_forwarded.contains(*p))
            .cloned()
            .collect();

        // Step 6 — record forwarded peers.
        let entry = self.forwarded_to.entry(msg.id).or_default();
        for p in &to_peers {
            entry.insert(p.clone());
        }

        // Step 7 / 8 — empty candidate set.
        if to_peers.is_empty() {
            // Loopback: origin is the only subscriber.
            let non_origin_subs: Vec<_> = subscribers
                .iter()
                .filter(|p| *p != &msg.origin_peer)
                .collect();
            if non_origin_subs.is_empty() {
                self.stats.messages_looped += 1;
                return ForwardDecision::Loopback;
            }
            // All non-origin subscribers already received this message.
            self.stats.messages_dropped += 1;
            return ForwardDecision::Drop {
                reason: "all subscribers already received this message".to_owned(),
            };
        }

        // Step 9 — increment per-subscriber message counts.
        for peer in &to_peers {
            let key = Self::sub_key(peer, &msg.topic);
            if let Some(rec) = self.subscriptions.get_mut(&key) {
                rec.message_count += 1;
            }
        }

        self.stats.messages_forwarded += 1;
        self.refresh_stats();
        ForwardDecision::Forward { to_peers }
    }

    /// Record that `msg_id` was manually forwarded to `peer_id`.
    ///
    /// Useful when the caller forwards a message outside of `route` and wants
    /// to keep the loop-prevention state consistent.
    pub fn mark_forwarded(&mut self, id: &FloodMessageId, peer_id: &str) {
        self.forwarded_to
            .entry(*id)
            .or_default()
            .insert(peer_id.to_owned());
    }

    // ── Cache maintenance ─────────────────────────────────────────────────────

    /// Evict all dedup-cache entries older than `config.dedup_window_secs`.
    ///
    /// Should be called periodically (e.g. every 10–30 seconds) to bound memory
    /// consumption.
    pub fn expire_cache(&mut self, now: u64) {
        let window = self.config.dedup_window_secs;
        let cutoff = now.saturating_sub(window);

        // Collect keys to remove first to satisfy the borrow checker.
        let expired: Vec<FloodMessageId> = self
            .message_cache
            .iter()
            .filter(|(_, &ts)| ts <= cutoff)
            .map(|(id, _)| *id)
            .collect();

        for id in expired {
            self.message_cache.remove(&id);
            self.forwarded_to.remove(&id);
        }

        self.refresh_stats();
    }

    // ── Statistics ────────────────────────────────────────────────────────────

    /// Return a reference to the current operational statistics.
    pub fn stats(&self) -> &FsrRouterStats {
        &self.stats
    }

    // ── ID computation ────────────────────────────────────────────────────────

    /// Compute a `FloodMessageId` from `payload`, `topic`, `peer_id`, and a
    /// Unix-epoch timestamp.
    ///
    /// The ID is produced by running FNV-1a over all inputs concatenated in a
    /// deterministic order, then taking two sequential hashes to fill 16 bytes.
    pub fn compute_message_id(
        payload: &[u8],
        topic: &FloodTopic,
        peer_id: &str,
        now: u64,
    ) -> FloodMessageId {
        // Build a contiguous buffer: payload || topic_bytes || peer_bytes ||
        // now_le_bytes, separated by a null byte to prevent boundary ambiguity.
        let mut buf: Vec<u8> =
            Vec::with_capacity(payload.len() + topic.0.len() + peer_id.len() + 18);
        buf.extend_from_slice(payload);
        buf.push(0x00);
        buf.extend_from_slice(topic.0.as_bytes());
        buf.push(0x00);
        buf.extend_from_slice(peer_id.as_bytes());
        buf.push(0x00);
        buf.extend_from_slice(&now.to_le_bytes());

        let h1 = fnv1a_64(&buf);
        // Second hash: feed h1's bytes back through FNV-1a for the upper half.
        let h2 = fnv1a_64(&h1.to_le_bytes());

        let mut id = [0u8; 16];
        id[..8].copy_from_slice(&h1.to_le_bytes());
        id[8..].copy_from_slice(&h2.to_le_bytes());
        FloodMessageId(id)
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /// Canonical subscription map key: `"peer_id\x00topic"`.
    ///
    /// Using `\x00` as separator prevents collisions when peer IDs or topic
    /// names contain colons.
    #[inline]
    fn sub_key(peer_id: &str, topic: &FloodTopic) -> String {
        format!("{}\x00{}", peer_id, topic.0)
    }

    /// Refresh derived counter fields in `stats`.
    #[inline]
    fn refresh_stats(&mut self) {
        let distinct_topics: HashSet<&FloodTopic> =
            self.subscriptions.values().map(|r| &r.topic).collect();
        self.stats.subscriptions = self.subscriptions.len();
        self.stats.topics = distinct_topics.len();
        self.stats.cache_size = self.message_cache.len();
    }

    /// Return `true` when the dedup cache is at capacity.
    pub fn cache_full(&self) -> bool {
        self.message_cache.len() >= self.config.max_message_cache
    }

    /// Return the number of peers that have already been sent `id`.
    pub fn forwarded_count(&self, id: &FloodMessageId) -> usize {
        self.forwarded_to.get(id).map(|s| s.len()).unwrap_or(0)
    }

    /// Return `true` if `peer_id` is subscribed to `topic`.
    pub fn is_subscribed(&self, peer_id: &str, topic: &FloodTopic) -> bool {
        self.subscriptions
            .contains_key(&Self::sub_key(peer_id, topic))
    }

    /// Return the `SubscriptionRecord` for `peer_id` on `topic`, if any.
    pub fn subscription(&self, peer_id: &str, topic: &FloodTopic) -> Option<&SubscriptionRecord> {
        self.subscriptions.get(&Self::sub_key(peer_id, topic))
    }

    /// Return the total number of active subscriptions.
    pub fn subscription_count(&self) -> usize {
        self.subscriptions.len()
    }

    /// Return an iterator over all active `SubscriptionRecord`s.
    pub fn all_subscriptions(&self) -> impl Iterator<Item = &SubscriptionRecord> {
        self.subscriptions.values()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ────────────────────────────────────────────────────────────────

    fn default_router() -> FloodSubRouter {
        FloodSubRouter::new(RouterConfig::default())
    }

    fn topic(s: &str) -> FloodTopic {
        FloodTopic::new(s)
    }

    /// Build a `FloodMessage` with a freshly computed ID.
    fn make_msg(payload: &[u8], t: &FloodTopic, peer: &str, ttl: u8, now: u64) -> FloodMessage {
        let id = FloodSubRouter::compute_message_id(payload, t, peer, now);
        FloodMessage {
            id,
            topic: t.clone(),
            payload: payload.to_vec(),
            ttl,
            origin_peer: peer.to_owned(),
            created_at: now,
        }
    }

    // ── 1. Basic construction ──────────────────────────────────────────────────

    #[test]
    fn test_new_router_is_empty() {
        let r = default_router();
        let s = r.stats();
        assert_eq!(s.subscriptions, 0);
        assert_eq!(s.topics, 0);
        assert_eq!(s.cache_size, 0);
        assert_eq!(s.messages_forwarded, 0);
        assert_eq!(s.messages_dropped, 0);
        assert_eq!(s.messages_looped, 0);
    }

    // ── 2. Subscribe ───────────────────────────────────────────────────────────

    #[test]
    fn test_subscribe_new_returns_true() {
        let mut r = default_router();
        assert!(r.subscribe("peer1", &topic("news"), 100));
    }

    #[test]
    fn test_subscribe_duplicate_returns_false() {
        let mut r = default_router();
        r.subscribe("peer1", &topic("news"), 100);
        assert!(!r.subscribe("peer1", &topic("news"), 200));
    }

    #[test]
    fn test_subscribe_updates_stats() {
        let mut r = default_router();
        r.subscribe("p1", &topic("t1"), 1);
        r.subscribe("p2", &topic("t1"), 2);
        r.subscribe("p1", &topic("t2"), 3);
        assert_eq!(r.stats().subscriptions, 3);
        assert_eq!(r.stats().topics, 2);
    }

    #[test]
    fn test_subscribe_respects_max_peers_per_topic() {
        let config = RouterConfig {
            max_peers_per_topic: 2,
            ..Default::default()
        };
        let mut r = FloodSubRouter::new(config);
        let t = topic("flood");
        assert!(r.subscribe("p1", &t, 1));
        assert!(r.subscribe("p2", &t, 2));
        assert!(!r.subscribe("p3", &t, 3)); // over limit
    }

    #[test]
    fn test_subscribe_respects_max_topics() {
        let config = RouterConfig {
            max_topics: 2,
            ..Default::default()
        };
        let mut r = FloodSubRouter::new(config);
        assert!(r.subscribe("p1", &topic("t1"), 1));
        assert!(r.subscribe("p1", &topic("t2"), 2));
        assert!(!r.subscribe("p1", &topic("t3"), 3)); // would exceed max_topics
    }

    // ── 3. Unsubscribe ─────────────────────────────────────────────────────────

    #[test]
    fn test_unsubscribe_existing_returns_true() {
        let mut r = default_router();
        r.subscribe("p1", &topic("x"), 1);
        assert!(r.unsubscribe("p1", &topic("x")));
    }

    #[test]
    fn test_unsubscribe_missing_returns_false() {
        let mut r = default_router();
        assert!(!r.unsubscribe("p1", &topic("x")));
    }

    #[test]
    fn test_unsubscribe_decrements_stats() {
        let mut r = default_router();
        r.subscribe("p1", &topic("x"), 1);
        r.subscribe("p2", &topic("x"), 2);
        r.unsubscribe("p1", &topic("x"));
        assert_eq!(r.stats().subscriptions, 1);
        assert_eq!(r.stats().topics, 1); // topic still active via p2
    }

    #[test]
    fn test_unsubscribe_last_peer_removes_topic_from_stats() {
        let mut r = default_router();
        r.subscribe("p1", &topic("x"), 1);
        r.unsubscribe("p1", &topic("x"));
        assert_eq!(r.stats().topics, 0);
    }

    // ── 4. peers_for_topic ────────────────────────────────────────────────────

    #[test]
    fn test_peers_for_topic_empty_when_no_subs() {
        let r = default_router();
        assert!(r.peers_for_topic(&topic("news")).is_empty());
    }

    #[test]
    fn test_peers_for_topic_returns_all_subscribers() {
        let mut r = default_router();
        r.subscribe("p1", &topic("news"), 1);
        r.subscribe("p2", &topic("news"), 2);
        r.subscribe("p3", &topic("other"), 3);
        let mut peers = r.peers_for_topic(&topic("news"));
        peers.sort();
        assert_eq!(peers, vec!["p1", "p2"]);
    }

    // ── 5. topics_for_peer ────────────────────────────────────────────────────

    #[test]
    fn test_topics_for_peer_empty_when_not_subscribed() {
        let r = default_router();
        assert!(r.topics_for_peer("p1").is_empty());
    }

    #[test]
    fn test_topics_for_peer_returns_all_topics() {
        let mut r = default_router();
        r.subscribe("p1", &topic("t1"), 1);
        r.subscribe("p1", &topic("t2"), 2);
        r.subscribe("p2", &topic("t1"), 3);
        let mut topics = r
            .topics_for_peer("p1")
            .into_iter()
            .map(|t| t.0)
            .collect::<Vec<_>>();
        topics.sort();
        assert_eq!(topics, vec!["t1", "t2"]);
    }

    // ── 6. route — duplicate drop ─────────────────────────────────────────────

    #[test]
    fn test_route_drops_duplicate_message() {
        let mut r = default_router();
        r.subscribe("p2", &topic("t"), 1);
        let msg = make_msg(b"hello", &topic("t"), "p1", 5, 1000);
        let d1 = r.route(&msg, 1000);
        assert!(matches!(d1, ForwardDecision::Forward { .. }));
        let d2 = r.route(&msg, 1001);
        assert!(matches!(d2, ForwardDecision::Drop { .. }));
        if let ForwardDecision::Drop { reason } = d2 {
            assert!(reason.contains("duplicate"));
        }
    }

    // ── 7. route — TTL drop ───────────────────────────────────────────────────

    #[test]
    fn test_route_drops_on_min_ttl() {
        let mut r = default_router();
        r.subscribe("p2", &topic("t"), 1);
        let msg = make_msg(b"x", &topic("t"), "p1", 1, 1000); // ttl == min_ttl (1)
        let d = r.route(&msg, 1000);
        assert!(matches!(d, ForwardDecision::Drop { .. }));
        if let ForwardDecision::Drop { reason } = d {
            assert!(reason.contains("ttl"));
        }
    }

    #[test]
    fn test_route_drops_on_zero_ttl() {
        let mut r = default_router();
        r.subscribe("p2", &topic("t"), 1);
        let msg = make_msg(b"x", &topic("t"), "p1", 0, 1000);
        assert!(matches!(r.route(&msg, 1000), ForwardDecision::Drop { .. }));
    }

    #[test]
    fn test_route_forwards_with_ttl_above_min() {
        let mut r = default_router();
        r.subscribe("p2", &topic("t"), 1);
        let msg = make_msg(b"x", &topic("t"), "p1", 2, 1000); // ttl = 2 > min_ttl = 1
        assert!(matches!(
            r.route(&msg, 1000),
            ForwardDecision::Forward { .. }
        ));
    }

    // ── 8. route — no subscribers ─────────────────────────────────────────────

    #[test]
    fn test_route_drops_on_no_subscribers() {
        let mut r = default_router();
        let msg = make_msg(b"x", &topic("t"), "p1", 5, 1000);
        let d = r.route(&msg, 1000);
        assert!(matches!(d, ForwardDecision::Drop { .. }));
        if let ForwardDecision::Drop { reason } = d {
            assert!(reason.contains("no subscribers"));
        }
    }

    // ── 9. route — loopback ───────────────────────────────────────────────────

    #[test]
    fn test_route_loopback_when_origin_only_subscriber() {
        let mut r = default_router();
        r.subscribe("p1", &topic("t"), 1); // only p1 subscribed
        let msg = make_msg(b"x", &topic("t"), "p1", 5, 1000); // p1 is origin
        let d = r.route(&msg, 1000);
        assert_eq!(d, ForwardDecision::Loopback);
        assert_eq!(r.stats().messages_looped, 1);
    }

    // ── 10. route — forward excludes origin ───────────────────────────────────

    #[test]
    fn test_route_excludes_origin_from_forward_set() {
        let mut r = default_router();
        r.subscribe("p1", &topic("t"), 1);
        r.subscribe("p2", &topic("t"), 2);
        r.subscribe("p3", &topic("t"), 3);
        let msg = make_msg(b"x", &topic("t"), "p1", 5, 1000);
        if let ForwardDecision::Forward { to_peers } = r.route(&msg, 1000) {
            assert!(!to_peers.contains(&"p1".to_owned()));
            assert!(to_peers.contains(&"p2".to_owned()));
            assert!(to_peers.contains(&"p3".to_owned()));
        } else {
            panic!("expected Forward");
        }
    }

    // ── 11. route — loop prevention via forwarded_to ──────────────────────────

    #[test]
    fn test_route_excludes_already_forwarded_peers() {
        let mut r = default_router();
        r.subscribe("p2", &topic("t"), 1);
        r.subscribe("p3", &topic("t"), 2);
        let msg = make_msg(b"x", &topic("t"), "p1", 5, 1000);
        // First route: should forward to p2 and p3.
        if let ForwardDecision::Forward { to_peers } = r.route(&msg, 1000) {
            assert_eq!(to_peers.len(), 2);
        } else {
            panic!("expected Forward on first call");
        }
        // Second route with a different ID but same peers (simulate another msg).
        let msg2 = make_msg(b"y", &topic("t"), "p1", 5, 1001);
        // p2 and p3 have NOT been forwarded msg2 yet — should forward.
        if let ForwardDecision::Forward { to_peers } = r.route(&msg2, 1001) {
            assert!(to_peers.contains(&"p2".to_owned()) || to_peers.contains(&"p3".to_owned()));
        } else {
            panic!("expected Forward for second distinct message");
        }
    }

    // ── 12. mark_forwarded ────────────────────────────────────────────────────

    #[test]
    fn test_mark_forwarded_prevents_re_forward() {
        let mut r = default_router();
        r.subscribe("p2", &topic("t"), 1);
        r.subscribe("p3", &topic("t"), 2);
        let msg = make_msg(b"x", &topic("t"), "p1", 5, 1000);
        // Manually mark p2 as already forwarded.
        r.mark_forwarded(&msg.id, "p2");
        // Insert into cache manually so route() doesn't think it's a dup.
        // (We call route, expecting only p3.)
        if let ForwardDecision::Forward { to_peers } = r.route(&msg, 1000) {
            assert!(!to_peers.contains(&"p2".to_owned()));
            assert!(to_peers.contains(&"p3".to_owned()));
        } else {
            panic!("expected Forward");
        }
    }

    // ── 13. expire_cache ──────────────────────────────────────────────────────

    #[test]
    fn test_expire_cache_removes_old_entries() {
        let mut r = default_router();
        r.subscribe("p2", &topic("t"), 1);
        let msg = make_msg(b"x", &topic("t"), "p1", 5, 1000);
        r.route(&msg, 1000); // inserts into cache at t=1000
        assert_eq!(r.stats().cache_size, 1);
        // Expire with `now` = 1000 + dedup_window + 1 = 1061.
        r.expire_cache(1061);
        assert_eq!(r.stats().cache_size, 0);
    }

    #[test]
    fn test_expire_cache_keeps_recent_entries() {
        let mut r = default_router();
        r.subscribe("p2", &topic("t"), 1);
        let msg = make_msg(b"x", &topic("t"), "p1", 5, 1000);
        r.route(&msg, 1000);
        // Expire at t=1050 (only 50 s elapsed, window=60 s → not expired).
        r.expire_cache(1050);
        assert_eq!(r.stats().cache_size, 1);
    }

    #[test]
    fn test_expire_cache_clears_forwarded_to_map() {
        let mut r = default_router();
        r.subscribe("p2", &topic("t"), 1);
        let msg = make_msg(b"x", &topic("t"), "p1", 5, 1000);
        r.route(&msg, 1000);
        r.expire_cache(1061);
        // After expiry the forwarded_to entry should also be gone.
        assert_eq!(r.forwarded_count(&msg.id), 0);
    }

    // ── 14. stats counters ────────────────────────────────────────────────────

    #[test]
    fn test_stats_messages_forwarded_increments() {
        let mut r = default_router();
        r.subscribe("p2", &topic("t"), 1);
        let msg = make_msg(b"a", &topic("t"), "p1", 5, 1000);
        r.route(&msg, 1000);
        assert_eq!(r.stats().messages_forwarded, 1);
    }

    #[test]
    fn test_stats_messages_dropped_increments_on_dup() {
        let mut r = default_router();
        r.subscribe("p2", &topic("t"), 1);
        let msg = make_msg(b"a", &topic("t"), "p1", 5, 1000);
        r.route(&msg, 1000);
        r.route(&msg, 1001);
        assert_eq!(r.stats().messages_dropped, 1);
    }

    #[test]
    fn test_stats_messages_dropped_increments_on_ttl() {
        let mut r = default_router();
        r.subscribe("p2", &topic("t"), 1);
        let msg = make_msg(b"a", &topic("t"), "p1", 1, 1000);
        r.route(&msg, 1000);
        assert_eq!(r.stats().messages_dropped, 1);
    }

    // ── 15. compute_message_id ────────────────────────────────────────────────

    #[test]
    fn test_compute_message_id_is_deterministic() {
        let t = topic("x");
        let id1 = FloodSubRouter::compute_message_id(b"hello", &t, "peer1", 42);
        let id2 = FloodSubRouter::compute_message_id(b"hello", &t, "peer1", 42);
        assert_eq!(id1, id2);
    }

    #[test]
    fn test_compute_message_id_differs_on_different_payload() {
        let t = topic("x");
        let id1 = FloodSubRouter::compute_message_id(b"hello", &t, "peer1", 42);
        let id2 = FloodSubRouter::compute_message_id(b"world", &t, "peer1", 42);
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_compute_message_id_differs_on_different_topic() {
        let id1 = FloodSubRouter::compute_message_id(b"x", &topic("t1"), "peer1", 1);
        let id2 = FloodSubRouter::compute_message_id(b"x", &topic("t2"), "peer1", 1);
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_compute_message_id_differs_on_different_peer() {
        let t = topic("t");
        let id1 = FloodSubRouter::compute_message_id(b"x", &t, "p1", 1);
        let id2 = FloodSubRouter::compute_message_id(b"x", &t, "p2", 1);
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_compute_message_id_differs_on_different_timestamp() {
        let t = topic("t");
        let id1 = FloodSubRouter::compute_message_id(b"x", &t, "p1", 1);
        let id2 = FloodSubRouter::compute_message_id(b"x", &t, "p1", 2);
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_compute_message_id_is_16_bytes() {
        let id = FloodSubRouter::compute_message_id(b"test", &topic("t"), "p", 0);
        assert_eq!(id.as_bytes().len(), 16);
    }

    // ── 16. FloodMessageId helpers ────────────────────────────────────────────

    #[test]
    fn test_flood_message_id_to_hex_is_32_chars() {
        let id = FloodSubRouter::compute_message_id(b"test", &topic("t"), "p", 0);
        assert_eq!(id.to_hex().len(), 32);
    }

    #[test]
    fn test_flood_message_id_display() {
        let id = FloodMessageId([0u8; 16]);
        assert_eq!(id.to_string(), "0".repeat(32));
    }

    // ── 17. FloodTopic helpers ────────────────────────────────────────────────

    #[test]
    fn test_flood_topic_as_str() {
        let t = FloodTopic::new("sports");
        assert_eq!(t.as_str(), "sports");
    }

    #[test]
    fn test_flood_topic_display() {
        let t = FloodTopic::new("sports");
        assert_eq!(t.to_string(), "sports");
    }

    #[test]
    fn test_flood_topic_equality() {
        assert_eq!(FloodTopic::new("a"), FloodTopic::new("a"));
        assert_ne!(FloodTopic::new("a"), FloodTopic::new("b"));
    }

    // ── 18. Multi-topic routing ───────────────────────────────────────────────

    #[test]
    fn test_route_only_delivers_to_topic_subscribers() {
        let mut r = default_router();
        r.subscribe("p2", &topic("sports"), 1);
        r.subscribe("p3", &topic("news"), 2);
        let msg = make_msg(b"goal!", &topic("sports"), "p1", 5, 1000);
        if let ForwardDecision::Forward { to_peers } = r.route(&msg, 1000) {
            assert!(to_peers.contains(&"p2".to_owned()));
            assert!(!to_peers.contains(&"p3".to_owned()));
        } else {
            panic!("expected Forward");
        }
    }

    // ── 19. Subscription record message_count ─────────────────────────────────

    #[test]
    fn test_subscription_record_message_count_increments() {
        let mut r = default_router();
        r.subscribe("p2", &topic("t"), 1);
        let msg1 = make_msg(b"a", &topic("t"), "p1", 5, 1000);
        let msg2 = make_msg(b"b", &topic("t"), "p1", 5, 1001);
        r.route(&msg1, 1000);
        r.route(&msg2, 1001);
        let rec = r.subscription("p2", &topic("t")).expect("must exist");
        assert_eq!(rec.message_count, 2);
    }

    // ── 20. is_subscribed ─────────────────────────────────────────────────────

    #[test]
    fn test_is_subscribed_true_after_subscribe() {
        let mut r = default_router();
        r.subscribe("p1", &topic("t"), 1);
        assert!(r.is_subscribed("p1", &topic("t")));
    }

    #[test]
    fn test_is_subscribed_false_before_subscribe() {
        let r = default_router();
        assert!(!r.is_subscribed("p1", &topic("t")));
    }

    #[test]
    fn test_is_subscribed_false_after_unsubscribe() {
        let mut r = default_router();
        r.subscribe("p1", &topic("t"), 1);
        r.unsubscribe("p1", &topic("t"));
        assert!(!r.is_subscribed("p1", &topic("t")));
    }

    // ── 21. cache eviction under max_message_cache ────────────────────────────

    #[test]
    fn test_cache_does_not_exceed_max_message_cache() {
        let config = RouterConfig {
            max_message_cache: 3,
            ..Default::default()
        };
        let mut r = FloodSubRouter::new(config);
        r.subscribe("p2", &topic("t"), 1);
        for i in 0u64..5 {
            let msg = make_msg(
                format!("payload{i}").as_bytes(),
                &topic("t"),
                "p1",
                5,
                1000 + i,
            );
            r.route(&msg, 1000 + i);
        }
        // Cache should never grow beyond max_message_cache.
        assert!(r.stats().cache_size <= 3);
    }

    // ── 22. forwarded_count ───────────────────────────────────────────────────

    #[test]
    fn test_forwarded_count_zero_before_route() {
        let r = default_router();
        let id = FloodSubRouter::compute_message_id(b"x", &topic("t"), "p", 1);
        assert_eq!(r.forwarded_count(&FloodMessageId(id.0)), 0);
    }

    #[test]
    fn test_forwarded_count_after_route() {
        let mut r = default_router();
        r.subscribe("p2", &topic("t"), 1);
        r.subscribe("p3", &topic("t"), 2);
        let msg = make_msg(b"x", &topic("t"), "p1", 5, 1000);
        r.route(&msg, 1000);
        assert_eq!(r.forwarded_count(&msg.id), 2);
    }

    // ── 23. cache_full ────────────────────────────────────────────────────────

    #[test]
    fn test_cache_full_returns_false_when_empty() {
        let r = default_router();
        assert!(!r.cache_full());
    }

    #[test]
    fn test_cache_full_returns_true_at_capacity() {
        let config = RouterConfig {
            max_message_cache: 1,
            ..Default::default()
        };
        let mut r = FloodSubRouter::new(config);
        r.subscribe("p2", &topic("t"), 1);
        let msg = make_msg(b"x", &topic("t"), "p1", 5, 1000);
        r.route(&msg, 1000);
        assert!(r.cache_full());
    }

    // ── 24. subscription_count ────────────────────────────────────────────────

    #[test]
    fn test_subscription_count_matches_stats() {
        let mut r = default_router();
        r.subscribe("p1", &topic("a"), 1);
        r.subscribe("p2", &topic("a"), 2);
        r.subscribe("p1", &topic("b"), 3);
        assert_eq!(r.subscription_count(), r.stats().subscriptions);
        assert_eq!(r.subscription_count(), 3);
    }

    // ── 25. all_subscriptions iterator ───────────────────────────────────────

    #[test]
    fn test_all_subscriptions_iterator_count() {
        let mut r = default_router();
        r.subscribe("p1", &topic("t1"), 1);
        r.subscribe("p2", &topic("t1"), 2);
        r.subscribe("p1", &topic("t2"), 3);
        assert_eq!(r.all_subscriptions().count(), 3);
    }

    // ── 26. xorshift64 PRNG sanity ────────────────────────────────────────────

    #[test]
    fn test_xorshift64_produces_different_values() {
        let mut state: u64 = 0xDEAD_BEEF_CAFE_1234;
        let v1 = xorshift64(&mut state);
        let v2 = xorshift64(&mut state);
        let v3 = xorshift64(&mut state);
        // Values must differ (practically guaranteed for any non-zero seed).
        assert_ne!(v1, v2);
        assert_ne!(v2, v3);
    }

    #[test]
    fn test_xorshift64_is_deterministic() {
        let mut s1: u64 = 42;
        let mut s2: u64 = 42;
        assert_eq!(xorshift64(&mut s1), xorshift64(&mut s2));
    }

    // ── 27. Drop reason for all-forwarded peers ───────────────────────────────

    #[test]
    fn test_route_drops_when_all_peers_already_forwarded() {
        let mut r = default_router();
        r.subscribe("p2", &topic("t"), 1);
        let msg = make_msg(b"x", &topic("t"), "p1", 5, 1000);
        // Mark p2 as already forwarded before routing.
        r.mark_forwarded(&msg.id, "p2");
        // Also need msg.id to NOT be in dedup cache yet.
        // route() will see no new peers → Drop (all already forwarded).
        let d = r.route(&msg, 1000);
        assert!(matches!(d, ForwardDecision::Drop { .. }));
        if let ForwardDecision::Drop { reason } = d {
            assert!(reason.contains("already received") || reason.contains("already"));
        }
    }

    // ── 28. multiple expire rounds ────────────────────────────────────────────

    #[test]
    fn test_multiple_expire_rounds_clean_incrementally() {
        let mut r = default_router();
        r.subscribe("p2", &topic("t"), 1);
        let msg1 = make_msg(b"first", &topic("t"), "p1", 5, 1000);
        let msg2 = make_msg(b"second", &topic("t"), "p1", 5, 1050);
        r.route(&msg1, 1000);
        r.route(&msg2, 1050);
        assert_eq!(r.stats().cache_size, 2);
        // Expire at t=1061: only msg1 (created_at=1000, window=60, cutoff=1001) expires.
        r.expire_cache(1061);
        assert_eq!(r.stats().cache_size, 1);
        // Expire at t=1111: msg2 (created_at=1050, cutoff=1051) expires.
        r.expire_cache(1111);
        assert_eq!(r.stats().cache_size, 0);
    }

    // ── 29. RouterConfig default values ──────────────────────────────────────

    #[test]
    fn test_router_config_defaults() {
        let cfg = RouterConfig::default();
        assert_eq!(cfg.max_peers_per_topic, 100);
        assert_eq!(cfg.max_message_cache, 10_000);
        assert_eq!(cfg.min_ttl, 1);
        assert_eq!(cfg.dedup_window_secs, 60);
        assert_eq!(cfg.max_topics, 256);
    }

    // ── 30. FsrRouterStats default values ────────────────────────────────────

    #[test]
    fn test_fsr_router_stats_defaults() {
        let s = FsrRouterStats::default();
        assert_eq!(s.messages_forwarded, 0);
        assert_eq!(s.messages_dropped, 0);
        assert_eq!(s.messages_looped, 0);
        assert_eq!(s.subscriptions, 0);
        assert_eq!(s.topics, 0);
        assert_eq!(s.cache_size, 0);
    }

    // ── 31. Peer with many topics respects max_topics boundary ───────────────

    #[test]
    fn test_subscribe_at_exactly_max_topics() {
        let config = RouterConfig {
            max_topics: 3,
            ..Default::default()
        };
        let mut r = FloodSubRouter::new(config);
        assert!(r.subscribe("p1", &topic("t1"), 1));
        assert!(r.subscribe("p1", &topic("t2"), 2));
        assert!(r.subscribe("p1", &topic("t3"), 3));
        // Same peer, new topic — must be rejected because we are already at 3.
        assert!(!r.subscribe("p1", &topic("t4"), 4));
    }

    // ── 32. Resubscribe after unsubscribe is allowed ──────────────────────────

    #[test]
    fn test_resubscribe_after_unsubscribe() {
        let mut r = default_router();
        r.subscribe("p1", &topic("t"), 1);
        r.unsubscribe("p1", &topic("t"));
        assert!(r.subscribe("p1", &topic("t"), 2));
    }

    // ── 33. Stats cache_size reflects expire ──────────────────────────────────

    #[test]
    fn test_stats_cache_size_reflects_expire() {
        let mut r = default_router();
        r.subscribe("p2", &topic("t"), 1);
        for i in 0u64..5 {
            let msg = make_msg(format!("m{i}").as_bytes(), &topic("t"), "p1", 5, 1000 + i);
            r.route(&msg, 1000 + i);
        }
        assert_eq!(r.stats().cache_size, 5);
        r.expire_cache(1100);
        assert_eq!(r.stats().cache_size, 0);
    }

    // ── 34. Route after cache expiry allows re-routing same message ───────────

    #[test]
    fn test_can_reroute_message_after_cache_expiry() {
        let mut r = default_router();
        r.subscribe("p2", &topic("t"), 1);
        let msg = make_msg(b"x", &topic("t"), "p1", 5, 1000);
        r.route(&msg, 1000);
        r.expire_cache(1100); // Expire the entry.
                              // After expiry, the same message ID should be routable again.
        let d = r.route(&msg, 1100);
        assert!(matches!(d, ForwardDecision::Forward { .. }));
    }

    // ── 35. Forward to multiple peers updates all message_counts ─────────────

    #[test]
    fn test_forward_to_multiple_peers_increments_all_counts() {
        let mut r = default_router();
        for i in 1u8..=5 {
            r.subscribe(&format!("p{i}"), &topic("t"), i as u64);
        }
        let msg = make_msg(b"broadcast", &topic("t"), "p_origin", 5, 1000);
        r.route(&msg, 1000);
        for i in 1u8..=5 {
            let rec = r
                .subscription(&format!("p{i}"), &topic("t"))
                .expect("must exist");
            assert_eq!(rec.message_count, 1, "p{i} should have count 1");
        }
    }
}
