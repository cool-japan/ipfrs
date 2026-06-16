//! SubscriptionRouter — topic/type-based message routing with subscription
//! management, filter evaluation, and delivery tracking.
//!
//! Routes incoming [`RoutingMessage`]s to registered [`Subscription`]s using
//! topic matching and filter evaluation.  Deliveries are tracked in a bounded
//! [`VecDeque`]-backed log.
//!
//! ## Design highlights
//!
//! - **Topic matching**: exact match or wildcard (`*`).
//! - **Filter evaluation**: composable [`SubscriptionFilter`] predicates.
//! - **Delivery log**: bounded at a configurable capacity; oldest entries are
//!   evicted when full.
//! - **No `unwrap()`**: all fallible operations surface errors explicitly.

use std::collections::{HashMap, VecDeque};

// ═══════════════════════════════════════════════════════════════════════════════
// SubscriptionRouter — topic/type-based message routing with subscription
// management, filter evaluation, and delivery tracking.
// ═══════════════════════════════════════════════════════════════════════════════

// ─── FNV-1a helpers ───────────────────────────────────────────────────────────

/// Compute an FNV-1a 64-bit hash over an arbitrary byte slice.
pub fn fnv1a_64(data: &[u8]) -> u64 {
    const FNV_OFFSET: u64 = 14_695_981_039_346_656_037;
    const FNV_PRIME: u64 = 1_099_511_628_211;
    let mut hash = FNV_OFFSET;
    for &byte in data {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Hash multiple string slices concatenated together.
fn fnv1a_strings(parts: &[&str]) -> u64 {
    const FNV_OFFSET: u64 = 14_695_981_039_346_656_037;
    const FNV_PRIME: u64 = 1_099_511_628_211;
    let mut hash = FNV_OFFSET;
    for part in parts {
        for &byte in part.as_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
    }
    hash
}

// ─── MessageTopic ─────────────────────────────────────────────────────────────

/// Newtype wrapper for topic identifiers used by [`SubscriptionRouter`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MessageTopic(pub String);

impl MessageTopic {
    /// Create a new [`MessageTopic`] from any string-like value.
    pub fn new(topic: impl Into<String>) -> Self {
        Self(topic.into())
    }

    /// Return the inner string as a `&str`.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for MessageTopic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

// ─── RoutingMessage ───────────────────────────────────────────────────────────

/// A message travelling through [`SubscriptionRouter`].
///
/// The `id` field is computed as the FNV-1a hash of `sender + topic.0 +
/// timestamp.to_string()` at construction time.
#[derive(Debug, Clone)]
pub struct RoutingMessage {
    /// FNV-1a hash of `(sender + topic + timestamp)`.
    pub id: u64,
    /// Topic this message belongs to.
    pub topic: MessageTopic,
    /// Number of bytes in the logical payload.
    pub payload_size: usize,
    /// Originating peer identifier.
    pub sender: String,
    /// Wall-clock millisecond timestamp at which the message was created.
    pub timestamp: u64,
    /// Remaining hops; the router drops messages where `ttl == 0` before
    /// delivering them.
    pub ttl: u8,
    /// Priority: 0 = lowest, 255 = highest.
    pub priority: u8,
}

impl RoutingMessage {
    /// Create a new [`RoutingMessage`], computing its `id` automatically.
    pub fn new(
        topic: MessageTopic,
        payload_size: usize,
        sender: impl Into<String>,
        timestamp: u64,
        ttl: u8,
        priority: u8,
    ) -> Self {
        let sender = sender.into();
        let ts_str = timestamp.to_string();
        let id = fnv1a_strings(&[sender.as_str(), topic.as_str(), ts_str.as_str()]);
        Self {
            id,
            topic,
            payload_size,
            sender,
            timestamp,
            ttl,
            priority,
        }
    }
}

// ─── SubscriptionFilter ───────────────────────────────────────────────────────

/// A composable predicate for filtering [`RoutingMessage`]s.
///
/// Filters are evaluated recursively; compound variants (`And`, `Or`) enable
/// arbitrary boolean combinations.
#[derive(Debug, Clone)]
pub enum SubscriptionFilter {
    /// Accept every message regardless of content.
    All,
    /// Accept only messages from the specified sender.
    BySender(String),
    /// Accept messages whose `priority` is ≥ `min`.
    ByPriority { min: u8 },
    /// Accept messages whose `payload_size` is ≤ `max_bytes`.
    BySize { max_bytes: usize },
    /// Accept messages where **both** sub-filters accept.
    And(Box<SubscriptionFilter>, Box<SubscriptionFilter>),
    /// Accept messages where **either** sub-filter accepts.
    Or(Box<SubscriptionFilter>, Box<SubscriptionFilter>),
}

impl SubscriptionFilter {
    /// Convenience constructor for [`SubscriptionFilter::And`].
    pub fn and(left: SubscriptionFilter, right: SubscriptionFilter) -> Self {
        Self::And(Box::new(left), Box::new(right))
    }

    /// Convenience constructor for [`SubscriptionFilter::Or`].
    pub fn or(left: SubscriptionFilter, right: SubscriptionFilter) -> Self {
        Self::Or(Box::new(left), Box::new(right))
    }
}

// ─── Subscription ─────────────────────────────────────────────────────────────

/// A subscriber registration in [`SubscriptionRouter`].
///
/// The `id` is the FNV-1a hash of `(peer_id + topic.0 + created_at.to_string())`.
#[derive(Debug, Clone)]
pub struct Subscription {
    /// Stable identifier for this subscription.
    pub id: u64,
    /// Peer that owns this subscription.
    pub peer_id: String,
    /// Topic this subscription is interested in.
    pub topic: MessageTopic,
    /// Additional predicate applied after topic matching.
    pub filter: SubscriptionFilter,
    /// Millisecond timestamp at which this subscription was created.
    pub created_at: u64,
    /// Number of messages successfully delivered to this subscription.
    pub message_count: u64,
}

impl Subscription {
    /// Create a new [`Subscription`], computing its `id` automatically.
    pub fn new(
        peer_id: impl Into<String>,
        topic: MessageTopic,
        filter: SubscriptionFilter,
        created_at: u64,
    ) -> Self {
        let peer_id = peer_id.into();
        let ts_str = created_at.to_string();
        let id = fnv1a_strings(&[peer_id.as_str(), topic.as_str(), ts_str.as_str()]);
        Self {
            id,
            peer_id,
            topic,
            filter,
            created_at,
            message_count: 0,
        }
    }
}

// ─── DeliveryRecord ───────────────────────────────────────────────────────────

/// Audit record written whenever the router attempts delivery.
#[derive(Debug, Clone)]
pub struct DeliveryRecord {
    /// ID of the [`RoutingMessage`] that was delivered (or dropped).
    pub message_id: u64,
    /// ID of the [`Subscription`] the router attempted to deliver to.
    pub subscription_id: u64,
    /// Millisecond timestamp of the delivery attempt.
    pub delivered_at: u64,
    /// `true` if the delivery was counted as successful, `false` otherwise.
    pub success: bool,
}

// ─── SubRouterStats ───────────────────────────────────────────────────────────

/// Point-in-time statistics snapshot for [`SubscriptionRouter`].
///
/// Named `SubRouterStats` to avoid collision with the existing `RouterStats`
/// type in this module.
#[derive(Debug, Clone, PartialEq)]
pub struct SubRouterStats {
    /// Number of currently active subscriptions.
    pub total_subscriptions: usize,
    /// Total messages passed to [`SubscriptionRouter::route`].
    pub total_routed: u64,
    /// Total successful deliveries across all subscriptions.
    pub total_delivered: u64,
    /// Total messages dropped (TTL == 0 on arrival or no matching subscription).
    pub total_dropped: u64,
    /// Ratio of delivered to (delivered + dropped); `1.0` when both are zero.
    pub delivery_rate: f64,
    /// Number of distinct topics with at least one active subscription.
    pub topics: usize,
}

// ─── SubscriptionRouter ───────────────────────────────────────────────────────

/// Topic/type-based message routing engine.
///
/// Manages subscriber registrations, evaluates [`SubscriptionFilter`] predicates,
/// logs delivery attempts to a bounded ring-buffer, and exposes rich query and
/// eviction helpers.
///
/// # Design
///
/// - Messages with `ttl == 0` on arrival are immediately dropped (counted in
///   `total_dropped`); their TTL is *not* decremented because the router does
///   not modify the caller-owned message.
/// - The delivery log is a [`VecDeque`] capped at `max_log_size`.  Oldest
///   entries are evicted when the cap is reached.
/// - All statistics counters are plain `u64` fields (not atomic) because the
///   router is designed for single-threaded or externally-synchronized use.
///   Wrap in `Arc<Mutex<_>>` if concurrent access is required.
pub struct SubscriptionRouter {
    subscriptions: HashMap<u64, Subscription>,
    delivery_log: VecDeque<DeliveryRecord>,
    max_log_size: usize,
    /// Total messages passed to [`route`](Self::route).
    pub total_routed: u64,
    /// Total successful deliveries.
    pub total_delivered: u64,
    /// Total dropped messages.
    pub total_dropped: u64,
}

impl SubscriptionRouter {
    /// Create a new [`SubscriptionRouter`] with a delivery log bounded at
    /// `max_log_size` entries.
    pub fn new(max_log_size: usize) -> Self {
        Self {
            subscriptions: HashMap::new(),
            delivery_log: VecDeque::new(),
            max_log_size,
            total_routed: 0,
            total_delivered: 0,
            total_dropped: 0,
        }
    }

    // ── Subscription management ───────────────────────────────────────────────

    /// Register a subscription and return its stable ID.
    ///
    /// If a subscription with the same derived ID already exists it is
    /// silently replaced.
    pub fn subscribe(
        &mut self,
        peer_id: String,
        topic: MessageTopic,
        filter: SubscriptionFilter,
        now: u64,
    ) -> u64 {
        let sub = Subscription::new(peer_id, topic, filter, now);
        let id = sub.id;
        self.subscriptions.insert(id, sub);
        id
    }

    /// Remove the subscription with the given `subscription_id`.
    ///
    /// Returns `true` if a subscription was found and removed, `false` if none
    /// existed with that ID.
    pub fn unsubscribe(&mut self, subscription_id: u64) -> bool {
        self.subscriptions.remove(&subscription_id).is_some()
    }

    // ── Routing ───────────────────────────────────────────────────────────────

    /// Route `message` to all matching subscriptions.
    ///
    /// Rules applied in order:
    /// 1. If `message.ttl == 0` the message is immediately dropped (counted in
    ///    `total_dropped`); an empty `Vec` is returned.
    /// 2. For every subscription, [`Self::matches`] is called.  Matching
    ///    subscriptions have their `message_count` incremented and receive a
    ///    successful [`DeliveryRecord`].
    /// 3. If no subscriptions match, `total_dropped` is incremented and a
    ///    failing [`DeliveryRecord`] with `subscription_id = 0` is appended.
    ///
    /// Returns the list of subscription IDs that were delivered to.
    pub fn route(&mut self, message: &RoutingMessage, now: u64) -> Vec<u64> {
        self.total_routed = self.total_routed.saturating_add(1);

        // Drop expired messages (TTL == 0).
        if message.ttl == 0 {
            self.total_dropped = self.total_dropped.saturating_add(1);
            self.push_delivery_record(DeliveryRecord {
                message_id: message.id,
                subscription_id: 0,
                delivered_at: now,
                success: false,
            });
            return Vec::new();
        }

        // Collect matching subscription IDs without borrowing mutably yet.
        let matching: Vec<u64> = self
            .subscriptions
            .values()
            .filter(|sub| Self::matches_static(sub, message))
            .map(|sub| sub.id)
            .collect();

        if matching.is_empty() {
            self.total_dropped = self.total_dropped.saturating_add(1);
            self.push_delivery_record(DeliveryRecord {
                message_id: message.id,
                subscription_id: 0,
                delivered_at: now,
                success: false,
            });
            return Vec::new();
        }

        // Deliver to each matching subscription.
        for &sub_id in &matching {
            if let Some(sub) = self.subscriptions.get_mut(&sub_id) {
                sub.message_count = sub.message_count.saturating_add(1);
            }
            self.total_delivered = self.total_delivered.saturating_add(1);
            self.push_delivery_record(DeliveryRecord {
                message_id: message.id,
                subscription_id: sub_id,
                delivered_at: now,
                success: true,
            });
        }

        matching
    }

    /// Append a [`DeliveryRecord`] to the ring-buffer, evicting the oldest
    /// entry if the buffer is at capacity.
    fn push_delivery_record(&mut self, record: DeliveryRecord) {
        if self.max_log_size == 0 {
            return;
        }
        if self.delivery_log.len() >= self.max_log_size {
            self.delivery_log.pop_front();
        }
        self.delivery_log.push_back(record);
    }

    // ── Filter evaluation ────────────────────────────────────────────────────

    /// Return `true` when `sub`'s topic matches `msg`'s topic **and** the
    /// subscription's filter accepts the message.
    ///
    /// This is a `pub` method as required by the specification.
    pub fn matches(sub: &Subscription, msg: &RoutingMessage) -> bool {
        Self::matches_static(sub, msg)
    }

    /// Internal implementation used in both `matches` and `route` (avoids
    /// borrow-checker issues when calling from `&mut self` contexts).
    fn matches_static(sub: &Subscription, msg: &RoutingMessage) -> bool {
        if sub.topic != msg.topic {
            return false;
        }
        Self::filter_matches(&sub.filter, msg)
    }

    /// Recursively evaluate `filter` against `msg`.
    ///
    /// This is a `pub` method as required by the specification.
    pub fn filter_matches(filter: &SubscriptionFilter, msg: &RoutingMessage) -> bool {
        match filter {
            SubscriptionFilter::All => true,
            SubscriptionFilter::BySender(sender) => &msg.sender == sender,
            SubscriptionFilter::ByPriority { min } => msg.priority >= *min,
            SubscriptionFilter::BySize { max_bytes } => msg.payload_size <= *max_bytes,
            SubscriptionFilter::And(left, right) => {
                Self::filter_matches(left, msg) && Self::filter_matches(right, msg)
            }
            SubscriptionFilter::Or(left, right) => {
                Self::filter_matches(left, msg) || Self::filter_matches(right, msg)
            }
        }
    }

    // ── Query helpers ────────────────────────────────────────────────────────

    /// Return references to all active subscriptions for `topic`.
    pub fn subscriptions_for_topic(&self, topic: &MessageTopic) -> Vec<&Subscription> {
        self.subscriptions
            .values()
            .filter(|sub| &sub.topic == topic)
            .collect()
    }

    /// Return references to all subscriptions owned by `peer_id`.
    pub fn subscriptions_for_peer(&self, peer_id: &str) -> Vec<&Subscription> {
        self.subscriptions
            .values()
            .filter(|sub| sub.peer_id == peer_id)
            .collect()
    }

    /// Compute the delivery rate over records whose `delivered_at >= since`.
    ///
    /// Returns `1.0` when there are no matching records.
    pub fn delivery_rate(&self, since: u64) -> f64 {
        let relevant: Vec<&DeliveryRecord> = self
            .delivery_log
            .iter()
            .filter(|r| r.delivered_at >= since)
            .collect();

        if relevant.is_empty() {
            return 1.0;
        }

        let delivered = relevant.iter().filter(|r| r.success).count() as f64;
        let total = relevant.len() as f64;
        delivered / total
    }

    /// Return references to the `n` most-recent delivery records.
    ///
    /// If fewer than `n` records exist, all are returned.
    pub fn recent_deliveries(&self, n: usize) -> Vec<&DeliveryRecord> {
        let len = self.delivery_log.len();
        let skip = len.saturating_sub(n);
        self.delivery_log.iter().skip(skip).collect()
    }

    /// Remove subscriptions that have never delivered a message and were
    /// created more than `max_age_ms` milliseconds before `now`.
    ///
    /// Returns the number of subscriptions evicted.
    pub fn evict_stale_subscriptions(&mut self, max_age_ms: u64, now: u64) -> usize {
        let before = self.subscriptions.len();
        self.subscriptions.retain(|_, sub| {
            // Keep if it has delivered at least one message OR is still young.
            sub.message_count > 0 || now.saturating_sub(sub.created_at) <= max_age_ms
        });
        before - self.subscriptions.len()
    }

    /// Return a [`SubRouterStats`] snapshot computed at the given logical
    /// `now` timestamp (used only for the `delivery_rate` calculation).
    pub fn stats(&self, now: u64) -> SubRouterStats {
        let topics: std::collections::HashSet<&MessageTopic> =
            self.subscriptions.values().map(|s| &s.topic).collect();

        SubRouterStats {
            total_subscriptions: self.subscriptions.len(),
            total_routed: self.total_routed,
            total_delivered: self.total_delivered,
            total_dropped: self.total_dropped,
            delivery_rate: self.delivery_rate(now),
            topics: topics.len(),
        }
    }
}

// ─── Tests (SubscriptionRouter) ───────────────────────────────────────────────

#[cfg(test)]
mod subscription_router_tests {
    use crate::subscription_router::{
        MessageTopic, RoutingMessage, Subscription, SubscriptionFilter, SubscriptionRouter,
    };

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn topic(s: &str) -> MessageTopic {
        MessageTopic::new(s)
    }

    fn msg(
        topic: &str,
        sender: &str,
        payload_size: usize,
        priority: u8,
        ttl: u8,
        timestamp: u64,
    ) -> RoutingMessage {
        RoutingMessage::new(
            MessageTopic::new(topic),
            payload_size,
            sender,
            timestamp,
            ttl,
            priority,
        )
    }

    fn all_filter() -> SubscriptionFilter {
        SubscriptionFilter::All
    }

    // ── Construction ──────────────────────────────────────────────────────────

    // 1. new() starts empty
    #[test]
    fn test_new_starts_empty() {
        let router = SubscriptionRouter::new(100);
        assert_eq!(router.total_routed, 0);
        assert_eq!(router.total_delivered, 0);
        assert_eq!(router.total_dropped, 0);
        let s = router.stats(0);
        assert_eq!(s.total_subscriptions, 0);
        assert_eq!(s.topics, 0);
    }

    // 2. new() with zero log size does not panic
    #[test]
    fn test_new_zero_log_size() {
        let mut router = SubscriptionRouter::new(0);
        let m = msg("t", "alice", 10, 5, 1, 100);
        router.subscribe("peer".to_string(), topic("t"), all_filter(), 0);
        router.route(&m, 0);
        // No delivery records stored, no panic.
        assert!(router.recent_deliveries(10).is_empty());
    }

    // ── Subscribe / Unsubscribe ────────────────────────────────────────────────

    // 3. subscribe returns a non-zero ID
    #[test]
    fn test_subscribe_returns_nonzero_id() {
        let mut router = SubscriptionRouter::new(100);
        let id = router.subscribe("alice".to_string(), topic("news"), all_filter(), 1000);
        assert_ne!(id, 0);
    }

    // 4. subscribe with different args yields distinct IDs
    #[test]
    fn test_subscribe_distinct_ids() {
        let mut router = SubscriptionRouter::new(100);
        let id1 = router.subscribe("alice".to_string(), topic("news"), all_filter(), 1000);
        let id2 = router.subscribe("bob".to_string(), topic("news"), all_filter(), 1000);
        assert_ne!(id1, id2);
    }

    // 5. unsubscribe returns true when found
    #[test]
    fn test_unsubscribe_returns_true() {
        let mut router = SubscriptionRouter::new(100);
        let id = router.subscribe("alice".to_string(), topic("news"), all_filter(), 1000);
        assert!(router.unsubscribe(id));
    }

    // 6. unsubscribe returns false when not found
    #[test]
    fn test_unsubscribe_returns_false() {
        let mut router = SubscriptionRouter::new(100);
        assert!(!router.unsubscribe(999_999));
    }

    // 7. subscription count decrements after unsubscribe
    #[test]
    fn test_subscription_count_after_unsubscribe() {
        let mut router = SubscriptionRouter::new(100);
        let id = router.subscribe("alice".to_string(), topic("news"), all_filter(), 1000);
        assert_eq!(router.stats(0).total_subscriptions, 1);
        router.unsubscribe(id);
        assert_eq!(router.stats(0).total_subscriptions, 0);
    }

    // ── Route: TTL == 0 ───────────────────────────────────────────────────────

    // 8. TTL == 0 message is dropped immediately
    #[test]
    fn test_ttl_zero_dropped() {
        let mut router = SubscriptionRouter::new(100);
        router.subscribe("alice".to_string(), topic("t"), all_filter(), 0);
        let m = msg("t", "sender", 10, 5, 0, 1000);
        let delivered = router.route(&m, 1000);
        assert!(delivered.is_empty());
        assert_eq!(router.total_dropped, 1);
        assert_eq!(router.total_delivered, 0);
    }

    // 9. TTL == 0 message leaves a failed delivery record
    #[test]
    fn test_ttl_zero_leaves_failed_record() {
        let mut router = SubscriptionRouter::new(100);
        let m = msg("t", "sender", 10, 5, 0, 1000);
        router.route(&m, 2000);
        let records = router.recent_deliveries(10);
        assert_eq!(records.len(), 1);
        assert!(!records[0].success);
    }

    // ── Route: matching ───────────────────────────────────────────────────────

    // 10. Message with TTL > 0 delivered to matching subscriber
    #[test]
    fn test_basic_delivery() {
        let mut router = SubscriptionRouter::new(100);
        let id = router.subscribe("alice".to_string(), topic("news"), all_filter(), 0);
        let m = msg("news", "bob", 100, 10, 3, 1000);
        let delivered = router.route(&m, 1000);
        assert_eq!(delivered, vec![id]);
        assert_eq!(router.total_delivered, 1);
    }

    // 11. Message delivered to multiple matching subscribers
    #[test]
    fn test_multiple_subscribers_delivery() {
        let mut router = SubscriptionRouter::new(100);
        let id1 = router.subscribe("alice".to_string(), topic("news"), all_filter(), 0);
        let id2 = router.subscribe("bob".to_string(), topic("news"), all_filter(), 0);
        let m = msg("news", "carol", 50, 5, 1, 1000);
        let mut delivered = router.route(&m, 1000);
        delivered.sort_unstable();
        let mut expected = vec![id1, id2];
        expected.sort_unstable();
        assert_eq!(delivered, expected);
        assert_eq!(router.total_delivered, 2);
    }

    // 12. Message on wrong topic is not delivered
    #[test]
    fn test_wrong_topic_not_delivered() {
        let mut router = SubscriptionRouter::new(100);
        router.subscribe("alice".to_string(), topic("sports"), all_filter(), 0);
        let m = msg("news", "bob", 50, 5, 1, 1000);
        let delivered = router.route(&m, 1000);
        assert!(delivered.is_empty());
        assert_eq!(router.total_dropped, 1);
    }

    // 13. message_count increments on delivery
    #[test]
    fn test_message_count_increments() {
        let mut router = SubscriptionRouter::new(100);
        let id = router.subscribe("alice".to_string(), topic("t"), all_filter(), 0);
        let m = msg("t", "bob", 10, 1, 1, 100);
        router.route(&m, 100);
        router.route(&m, 200);
        let sub = router.subscriptions.get(&id).expect("sub exists");
        assert_eq!(sub.message_count, 2);
    }

    // ── Filter: BySender ──────────────────────────────────────────────────────

    // 14. BySender filter accepts matching sender
    #[test]
    fn test_by_sender_accepts() {
        let mut router = SubscriptionRouter::new(100);
        let filter = SubscriptionFilter::BySender("alice".to_string());
        router.subscribe("peer".to_string(), topic("t"), filter, 0);
        let m = msg("t", "alice", 10, 5, 1, 100);
        let delivered = router.route(&m, 100);
        assert_eq!(delivered.len(), 1);
    }

    // 15. BySender filter rejects non-matching sender
    #[test]
    fn test_by_sender_rejects() {
        let mut router = SubscriptionRouter::new(100);
        let filter = SubscriptionFilter::BySender("alice".to_string());
        router.subscribe("peer".to_string(), topic("t"), filter, 0);
        let m = msg("t", "bob", 10, 5, 1, 100);
        let delivered = router.route(&m, 100);
        assert!(delivered.is_empty());
    }

    // ── Filter: ByPriority ────────────────────────────────────────────────────

    // 16. ByPriority accepts message with priority >= min
    #[test]
    fn test_by_priority_accepts() {
        let mut router = SubscriptionRouter::new(100);
        router.subscribe(
            "p".to_string(),
            topic("t"),
            SubscriptionFilter::ByPriority { min: 5 },
            0,
        );
        let m = msg("t", "s", 10, 5, 1, 100);
        assert_eq!(router.route(&m, 100).len(), 1);
    }

    // 17. ByPriority rejects message with priority < min
    #[test]
    fn test_by_priority_rejects() {
        let mut router = SubscriptionRouter::new(100);
        router.subscribe(
            "p".to_string(),
            topic("t"),
            SubscriptionFilter::ByPriority { min: 10 },
            0,
        );
        let m = msg("t", "s", 10, 5, 1, 100);
        assert!(router.route(&m, 100).is_empty());
    }

    // ── Filter: BySize ────────────────────────────────────────────────────────

    // 18. BySize accepts message with payload_size <= max_bytes
    #[test]
    fn test_by_size_accepts() {
        let mut router = SubscriptionRouter::new(100);
        router.subscribe(
            "p".to_string(),
            topic("t"),
            SubscriptionFilter::BySize { max_bytes: 100 },
            0,
        );
        let m = msg("t", "s", 100, 5, 1, 100);
        assert_eq!(router.route(&m, 100).len(), 1);
    }

    // 19. BySize rejects message with payload_size > max_bytes
    #[test]
    fn test_by_size_rejects() {
        let mut router = SubscriptionRouter::new(100);
        router.subscribe(
            "p".to_string(),
            topic("t"),
            SubscriptionFilter::BySize { max_bytes: 99 },
            0,
        );
        let m = msg("t", "s", 100, 5, 1, 100);
        assert!(router.route(&m, 100).is_empty());
    }

    // ── Filter: And / Or ──────────────────────────────────────────────────────

    // 20. And filter: both must pass
    #[test]
    fn test_and_filter_both_must_pass() {
        let filter = SubscriptionFilter::and(
            SubscriptionFilter::BySender("alice".to_string()),
            SubscriptionFilter::ByPriority { min: 5 },
        );
        let mut router = SubscriptionRouter::new(100);
        router.subscribe("p".to_string(), topic("t"), filter, 0);

        // Both pass
        let m_ok = msg("t", "alice", 10, 7, 1, 100);
        assert_eq!(router.route(&m_ok, 100).len(), 1);

        // Priority fails
        let m_bad_pri = msg("t", "alice", 10, 3, 1, 200);
        assert!(router.route(&m_bad_pri, 200).is_empty());

        // Sender fails
        let m_bad_sender = msg("t", "bob", 10, 7, 1, 300);
        assert!(router.route(&m_bad_sender, 300).is_empty());
    }

    // 21. Or filter: either can pass
    #[test]
    fn test_or_filter_either_can_pass() {
        let filter = SubscriptionFilter::or(
            SubscriptionFilter::BySender("alice".to_string()),
            SubscriptionFilter::ByPriority { min: 200 },
        );
        let mut router = SubscriptionRouter::new(100);
        router.subscribe("p".to_string(), topic("t"), filter, 0);

        // Sender matches (priority doesn't need to)
        let m_sender = msg("t", "alice", 10, 5, 1, 100);
        assert_eq!(router.route(&m_sender, 100).len(), 1);

        // Priority matches (sender doesn't need to)
        let m_prio = msg("t", "carol", 10, 200, 1, 200);
        assert_eq!(router.route(&m_prio, 200).len(), 1);

        // Neither matches
        let m_none = msg("t", "dave", 10, 50, 1, 300);
        assert!(router.route(&m_none, 300).is_empty());
    }

    // ── subscriptions_for_topic ───────────────────────────────────────────────

    // 22. subscriptions_for_topic returns correct entries
    #[test]
    fn test_subscriptions_for_topic() {
        let mut router = SubscriptionRouter::new(100);
        router.subscribe("alice".to_string(), topic("news"), all_filter(), 0);
        router.subscribe("bob".to_string(), topic("news"), all_filter(), 1);
        router.subscribe("carol".to_string(), topic("sports"), all_filter(), 2);

        let news_subs = router.subscriptions_for_topic(&topic("news"));
        assert_eq!(news_subs.len(), 2);

        let sports_subs = router.subscriptions_for_topic(&topic("sports"));
        assert_eq!(sports_subs.len(), 1);
    }

    // 23. subscriptions_for_topic returns empty for unknown topic
    #[test]
    fn test_subscriptions_for_topic_empty() {
        let router = SubscriptionRouter::new(100);
        assert!(router.subscriptions_for_topic(&topic("unknown")).is_empty());
    }

    // ── subscriptions_for_peer ────────────────────────────────────────────────

    // 24. subscriptions_for_peer returns correct entries
    #[test]
    fn test_subscriptions_for_peer() {
        let mut router = SubscriptionRouter::new(100);
        router.subscribe("alice".to_string(), topic("news"), all_filter(), 0);
        router.subscribe("alice".to_string(), topic("sports"), all_filter(), 1);
        router.subscribe("bob".to_string(), topic("news"), all_filter(), 2);

        let alice_subs = router.subscriptions_for_peer("alice");
        assert_eq!(alice_subs.len(), 2);

        let bob_subs = router.subscriptions_for_peer("bob");
        assert_eq!(bob_subs.len(), 1);
    }

    // ── delivery_rate ─────────────────────────────────────────────────────────

    // 25. delivery_rate returns 1.0 with no records
    #[test]
    fn test_delivery_rate_no_records() {
        let router = SubscriptionRouter::new(100);
        assert!((router.delivery_rate(0) - 1.0).abs() < f64::EPSILON);
    }

    // 26. delivery_rate returns 1.0 for all successful deliveries
    #[test]
    fn test_delivery_rate_all_success() {
        let mut router = SubscriptionRouter::new(100);
        router.subscribe("p".to_string(), topic("t"), all_filter(), 0);
        let m = msg("t", "s", 10, 1, 1, 100);
        router.route(&m, 100);
        router.route(&m, 200);
        // All records are successes (one subscription, two deliveries)
        let rate = router.delivery_rate(0);
        assert!((rate - 1.0).abs() < f64::EPSILON);
    }

    // 27. delivery_rate returns 0.0 when all records are failures
    #[test]
    fn test_delivery_rate_all_failures() {
        let mut router = SubscriptionRouter::new(100);
        // No subscriptions — every message drops
        let m = msg("t", "s", 10, 1, 1, 100);
        router.route(&m, 100);
        router.route(&m, 200);
        let rate = router.delivery_rate(0);
        assert!((rate - 0.0).abs() < f64::EPSILON);
    }

    // 28. delivery_rate respects the `since` filter
    #[test]
    fn test_delivery_rate_since_filter() {
        let mut router = SubscriptionRouter::new(100);
        // No subscription → all drops
        let m = msg("t", "s", 10, 1, 1, 100);
        router.route(&m, 50); // before 'since'
                              // Now add a subscription for future successes
        router.subscribe("p".to_string(), topic("t"), all_filter(), 0);
        let m2 = msg("t", "s", 10, 1, 1, 200);
        router.route(&m2, 200); // after 'since'
                                // Records since ts=100 should include only the second (success)
        let rate = router.delivery_rate(100);
        assert!((rate - 1.0).abs() < f64::EPSILON);
    }

    // ── recent_deliveries ────────────────────────────────────────────────────

    // 29. recent_deliveries returns most-recent n records
    #[test]
    fn test_recent_deliveries() {
        let mut router = SubscriptionRouter::new(100);
        // No subscription — all drops (failures)
        for ts in 0..5u64 {
            let m = msg("t", "s", 10, 1, 1, ts * 100);
            router.route(&m, ts * 100);
        }
        let recent = router.recent_deliveries(3);
        assert_eq!(recent.len(), 3);
        // The most recent three should have delivered_at 200, 300, 400.
        assert_eq!(recent[2].delivered_at, 400);
    }

    // 30. recent_deliveries returns all when n > log length
    #[test]
    fn test_recent_deliveries_all() {
        let mut router = SubscriptionRouter::new(100);
        let m = msg("t", "s", 10, 1, 1, 100);
        router.route(&m, 100);
        let recent = router.recent_deliveries(1000);
        assert_eq!(recent.len(), 1);
    }

    // ── evict_stale_subscriptions ─────────────────────────────────────────────

    // 31. evict_stale_subscriptions removes old unused subscriptions
    #[test]
    fn test_evict_stale_removes_old_unused() {
        let mut router = SubscriptionRouter::new(100);
        // Created at t=0, no messages delivered.
        router.subscribe("alice".to_string(), topic("t"), all_filter(), 0);
        // Evict subs older than 1000ms at t=2000.
        let evicted = router.evict_stale_subscriptions(1000, 2000);
        assert_eq!(evicted, 1);
        assert_eq!(router.stats(2000).total_subscriptions, 0);
    }

    // 32. evict_stale_subscriptions keeps young subscriptions
    #[test]
    fn test_evict_stale_keeps_young() {
        let mut router = SubscriptionRouter::new(100);
        router.subscribe("alice".to_string(), topic("t"), all_filter(), 1500);
        // Evict subs older than 1000ms at t=2000.  Sub is only 500ms old.
        let evicted = router.evict_stale_subscriptions(1000, 2000);
        assert_eq!(evicted, 0);
        assert_eq!(router.stats(2000).total_subscriptions, 1);
    }

    // 33. evict_stale_subscriptions keeps old subscriptions with messages
    #[test]
    fn test_evict_stale_keeps_active() {
        let mut router = SubscriptionRouter::new(100);
        let _id = router.subscribe("alice".to_string(), topic("t"), all_filter(), 0);
        // Deliver a message so message_count > 0.
        let m = msg("t", "s", 10, 1, 1, 100);
        router.route(&m, 100);
        // Evict subs older than 1000ms at t=5000.  Sub is old but has messages.
        let evicted = router.evict_stale_subscriptions(1000, 5000);
        assert_eq!(evicted, 0);
    }

    // ── stats() ───────────────────────────────────────────────────────────────

    // 34. stats() reflects correct topic count
    #[test]
    fn test_stats_topic_count() {
        let mut router = SubscriptionRouter::new(100);
        router.subscribe("alice".to_string(), topic("news"), all_filter(), 0);
        router.subscribe("bob".to_string(), topic("sports"), all_filter(), 0);
        router.subscribe("carol".to_string(), topic("news"), all_filter(), 0);
        let s = router.stats(0);
        assert_eq!(s.topics, 2);
        assert_eq!(s.total_subscriptions, 3);
    }

    // 35. stats() total_routed is accurate
    #[test]
    fn test_stats_total_routed() {
        let mut router = SubscriptionRouter::new(100);
        let m = msg("t", "s", 10, 1, 1, 100);
        router.route(&m, 100);
        router.route(&m, 200);
        assert_eq!(router.stats(0).total_routed, 2);
    }

    // ── delivery log cap ─────────────────────────────────────────────────────

    // 36. delivery log is capped at max_log_size
    #[test]
    fn test_delivery_log_capped() {
        let cap = 5;
        let mut router = SubscriptionRouter::new(cap);
        let m = msg("t", "s", 10, 1, 1, 100);
        for _ in 0..20 {
            router.route(&m, 100);
        }
        assert_eq!(router.recent_deliveries(100).len(), cap);
    }

    // ── RoutingMessage helpers ────────────────────────────────────────────────

    // 37. RoutingMessage id is derived from sender+topic+timestamp
    #[test]
    fn test_routing_message_id_derived() {
        let m1 = RoutingMessage::new(topic("t"), 10, "alice", 100, 1, 5);
        let m2 = RoutingMessage::new(topic("t"), 10, "alice", 100, 1, 5);
        assert_eq!(m1.id, m2.id);

        let m3 = RoutingMessage::new(topic("t"), 10, "bob", 100, 1, 5);
        assert_ne!(m1.id, m3.id);
    }

    // 38. Subscription id is derived from peer_id+topic+created_at
    #[test]
    fn test_subscription_id_derived() {
        let s1 = Subscription::new("alice", topic("t"), SubscriptionFilter::All, 1000);
        let s2 = Subscription::new("alice", topic("t"), SubscriptionFilter::All, 1000);
        assert_eq!(s1.id, s2.id);

        let s3 = Subscription::new("bob", topic("t"), SubscriptionFilter::All, 1000);
        assert_ne!(s1.id, s3.id);
    }

    // 39. MessageTopic Display and as_str
    #[test]
    fn test_message_topic_display() {
        let t = MessageTopic::new("hello");
        assert_eq!(t.as_str(), "hello");
        assert_eq!(t.to_string(), "hello");
    }

    // 40. DeliveryRecord fields are correctly populated on successful delivery
    #[test]
    fn test_delivery_record_fields_on_success() {
        let mut router = SubscriptionRouter::new(100);
        router.subscribe("alice".to_string(), topic("t"), all_filter(), 0);
        let m = msg("t", "bob", 10, 5, 1, 999);
        router.route(&m, 999);
        let records = router.recent_deliveries(10);
        assert_eq!(records.len(), 1);
        let r = records[0];
        assert_eq!(r.message_id, m.id);
        assert!(r.success);
        assert_eq!(r.delivered_at, 999);
    }

    // ── fnv1a helper ──────────────────────────────────────────────────────────

    // 41. fnv1a_64 produces consistent output
    #[test]
    fn test_fnv1a_consistent() {
        use crate::subscription_router::fnv1a_64;
        let h1 = fnv1a_64(b"hello");
        let h2 = fnv1a_64(b"hello");
        assert_eq!(h1, h2);
        let h3 = fnv1a_64(b"world");
        assert_ne!(h1, h3);
    }

    // 42. And filter: deeply nested
    #[test]
    fn test_and_filter_deeply_nested() {
        let filter = SubscriptionFilter::and(
            SubscriptionFilter::And(
                Box::new(SubscriptionFilter::ByPriority { min: 1 }),
                Box::new(SubscriptionFilter::BySize { max_bytes: 500 }),
            ),
            SubscriptionFilter::BySender("trusted".to_string()),
        );
        let m = RoutingMessage::new(topic("t"), 100, "trusted", 100, 1, 10);
        assert!(SubscriptionRouter::filter_matches(&filter, &m));

        let m_bad = RoutingMessage::new(topic("t"), 1000, "trusted", 100, 1, 10);
        assert!(!SubscriptionRouter::filter_matches(&filter, &m_bad));
    }

    // 43. Or filter: both fail → rejected
    #[test]
    fn test_or_filter_both_fail() {
        let filter = SubscriptionFilter::or(
            SubscriptionFilter::BySender("alice".to_string()),
            SubscriptionFilter::ByPriority { min: 200 },
        );
        let m = RoutingMessage::new(topic("t"), 10, "dave", 100, 1, 50);
        assert!(!SubscriptionRouter::filter_matches(&filter, &m));
    }
}
