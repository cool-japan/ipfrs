//! GossipMessageFilter — multi-criteria filter for GossipSub messages.
//!
//! Deduplicates, validates, and rate-limits incoming gossip messages before
//! propagation.  Combines:
//!
//! - **Deduplication**: time-windowed seen-ID set with bounded memory (LRU eviction)
//! - **Rule-based filtering**: hop-count limits, data-size caps, topic allowlists,
//!   sender block-lists, per-topic minimum intervals
//! - **Statistics**: accept / reject / duplicate counters with accept-rate calculation
//!
//! # Example
//!
//! ```rust
//! use ipfrs_network::gossip_message_filter::{
//!     FilterConfig, FilterRule, FilterVerdict, GossipMessage, GossipMessageFilter,
//!     MessageId,
//! };
//!
//! let config = FilterConfig {
//!     rules: vec![FilterRule::MaxHopCount(4), FilterRule::MaxDataSize(65_536)],
//!     dedup_window_ms: 30_000,
//!     max_seen_ids: 10_000,
//! };
//! let mut filter = GossipMessageFilter::new(config);
//!
//! let data = b"hello world";
//! let msg = GossipMessage {
//!     id: MessageId::from_content(data),
//!     topic: "blocks".to_string(),
//!     sender: "peer-A".to_string(),
//!     data: data.to_vec(),
//!     received_at: 1_000,
//!     hop_count: 2,
//! };
//!
//! assert_eq!(filter.filter(&msg, 1_000), FilterVerdict::Accept);
//! assert_eq!(filter.filter(&msg, 1_001), FilterVerdict::Duplicate);
//! ```

use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// FNV-1a constants
// ─────────────────────────────────────────────────────────────────────────────

const FNV_OFFSET_BASIS: u64 = 14_695_981_039_346_656_037;
const FNV_PRIME: u64 = 1_099_511_628_211;

/// Compute a 64-bit FNV-1a hash over `bytes`.
#[inline]
pub fn fnv1a_64(bytes: &[u8]) -> u64 {
    let mut h = FNV_OFFSET_BASIS;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

// ─────────────────────────────────────────────────────────────────────────────
// MessageId
// ─────────────────────────────────────────────────────────────────────────────

/// A 32-byte message identifier derived from message content.
///
/// Layout:
/// - bytes  0– 7: FNV-1a hash
/// - bytes  8–15: FNV-1a XOR byte-reversed FNV-1a
/// - bytes 16–23: FNV-1a rotated left 32 bits
/// - bytes 24–31: XOR of the three preceding 8-byte words
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct MessageId(pub [u8; 32]);

impl std::fmt::Debug for MessageId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MessageId(")?;
        for b in &self.0 {
            write!(f, "{b:02x}")?;
        }
        write!(f, ")")
    }
}

impl MessageId {
    /// Derive a `MessageId` from raw content bytes using a 64-bit FNV-1a hash
    /// spread across four 8-byte words.
    pub fn from_content(data: &[u8]) -> Self {
        let hash = fnv1a_64(data);

        // Word 0: raw FNV-1a
        let w0 = hash;
        // Word 1: hash XOR byte-reversed hash
        let w1 = hash ^ hash.swap_bytes();
        // Word 2: hash rotated left by 32 bits
        let w2 = hash.rotate_left(32);
        // Word 3: XOR of the three preceding words
        let w3 = w0 ^ w1 ^ w2;

        let mut id = [0u8; 32];
        id[0..8].copy_from_slice(&w0.to_le_bytes());
        id[8..16].copy_from_slice(&w1.to_le_bytes());
        id[16..24].copy_from_slice(&w2.to_le_bytes());
        id[24..32].copy_from_slice(&w3.to_le_bytes());
        Self(id)
    }

    /// Return the raw 32-byte array.
    #[inline]
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl std::fmt::Display for MessageId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for b in &self.0 {
            write!(f, "{b:02x}")?;
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GossipMessage
// ─────────────────────────────────────────────────────────────────────────────

/// A gossip message received from the network.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GossipMessage {
    /// Unique identifier for the message.
    pub id: MessageId,
    /// Topic / channel the message was published on.
    pub topic: String,
    /// Peer ID (or other identifier) of the originating sender.
    pub sender: String,
    /// Raw payload bytes.
    pub data: Vec<u8>,
    /// Wall-clock timestamp in milliseconds when the message was received.
    pub received_at: u64,
    /// Number of hops the message has already traversed.
    pub hop_count: u8,
}

// ─────────────────────────────────────────────────────────────────────────────
// FilterRule
// ─────────────────────────────────────────────────────────────────────────────

/// A single filtering criterion applied during [`GossipMessageFilter::filter`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FilterRule {
    /// Reject messages whose `hop_count` exceeds `n`.
    MaxHopCount(u8),
    /// Reject messages whose `data` length exceeds `n` bytes.
    MaxDataSize(usize),
    /// Reject messages whose `topic` is not in the allowed list.
    AllowedTopics(Vec<String>),
    /// Reject messages whose `sender` appears in the block list.
    BlockedSenders(Vec<String>),
    /// Reject a message on `topic` if the same topic was received within `min_ms`
    /// milliseconds of `now`.
    MinInterval { topic: String, min_ms: u64 },
}

// ─────────────────────────────────────────────────────────────────────────────
// FilterVerdict
// ─────────────────────────────────────────────────────────────────────────────

/// Decision produced by [`GossipMessageFilter::filter`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FilterVerdict {
    /// The message passed all checks and should be propagated.
    Accept,
    /// The message failed a rule and should be dropped.
    Reject {
        /// Human-readable explanation of the rejection.
        reason: String,
    },
    /// The message was already seen within the deduplication window.
    Duplicate,
}

// ─────────────────────────────────────────────────────────────────────────────
// FilterConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for [`GossipMessageFilter`].
#[derive(Debug, Clone)]
pub struct FilterConfig {
    /// Ordered list of rules applied to each incoming message.
    pub rules: Vec<FilterRule>,
    /// Duration (milliseconds) during which a message ID is considered "seen".
    pub dedup_window_ms: u64,
    /// Maximum number of message IDs tracked in the seen-ID map before the
    /// oldest entries are evicted.
    pub max_seen_ids: usize,
}

impl Default for FilterConfig {
    fn default() -> Self {
        Self {
            rules: Vec::new(),
            dedup_window_ms: 60_000,
            max_seen_ids: 100_000,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FilterStats / GossipMessageFilter
// ─────────────────────────────────────────────────────────────────────────────

/// Snapshot of filter activity counters.
#[derive(Debug, Clone, PartialEq)]
pub struct FilterStats {
    /// Number of messages accepted since construction.
    pub accepted: u64,
    /// Number of messages rejected since construction.
    pub rejected: u64,
    /// Number of duplicate messages detected since construction.
    pub duplicates: u64,
    /// Current number of message IDs in the seen-ID map.
    pub seen_ids_count: usize,
    /// `accepted / (accepted + rejected + duplicates)`, or `0.0` when no
    /// messages have been processed.
    pub accept_rate: f64,
}

/// Multi-criteria filter for GossipSub messages.
///
/// Thread-safety: **not** `Send`/`Sync` by default.  Wrap in an `Arc<Mutex<_>>`
/// if shared across tasks.
pub struct GossipMessageFilter {
    /// Active configuration.
    pub config: FilterConfig,
    /// Map from message ID → timestamp (ms) of first observation.
    pub seen_ids: HashMap<MessageId, u64>,
    /// Map from topic → timestamp (ms) of the most recent accepted message.
    pub topic_last_seen: HashMap<String, u64>,
    /// Total messages accepted.
    pub accepted: u64,
    /// Total messages rejected.
    pub rejected: u64,
    /// Total duplicate messages detected.
    pub duplicates: u64,
}

impl GossipMessageFilter {
    /// Construct a new filter with the supplied configuration.
    pub fn new(config: FilterConfig) -> Self {
        let cap = config.max_seen_ids.min(1_024);
        Self {
            config,
            seen_ids: HashMap::with_capacity(cap),
            topic_last_seen: HashMap::new(),
            accepted: 0,
            rejected: 0,
            duplicates: 0,
        }
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Core filter logic
    // ──────────────────────────────────────────────────────────────────────────

    /// Evaluate `msg` at wall-clock `now` (milliseconds).
    ///
    /// Evaluation order:
    /// 1. Deduplication check — `Duplicate` if already seen within the window.
    /// 2. Rule evaluation — first failing rule produces `Reject { reason }`.
    /// 3. Accept — record the message ID and update per-topic timestamp.
    pub fn filter(&mut self, msg: &GossipMessage, now: u64) -> FilterVerdict {
        // 1. Deduplication
        if self.is_duplicate(&msg.id, now) {
            self.duplicates += 1;
            return FilterVerdict::Duplicate;
        }

        // 2. Rule evaluation
        let rules: Vec<FilterRule> = self.config.rules.clone();
        for rule in &rules {
            if let Some(reason) = self.apply_rule(rule, msg, now) {
                self.rejected += 1;
                return FilterVerdict::Reject { reason };
            }
        }

        // 3. Accept
        self.seen_ids.insert(msg.id, now);
        self.topic_last_seen.insert(msg.topic.clone(), now);
        self.accepted += 1;

        // Opportunistic eviction to bound memory usage
        if self.seen_ids.len() > self.config.max_seen_ids {
            self.evict_expired_seen(now);
        }

        FilterVerdict::Accept
    }

    /// Apply a single rule to `msg`.  Returns `Some(reason)` if the message
    /// should be rejected, `None` if it passes this rule.
    pub fn apply_rule(&self, rule: &FilterRule, msg: &GossipMessage, now: u64) -> Option<String> {
        match rule {
            FilterRule::MaxHopCount(max) => {
                if msg.hop_count > *max {
                    Some(format!(
                        "hop_count {} exceeds maximum {}",
                        msg.hop_count, max
                    ))
                } else {
                    None
                }
            }
            FilterRule::MaxDataSize(max) => {
                if msg.data.len() > *max {
                    Some(format!(
                        "data size {} exceeds maximum {}",
                        msg.data.len(),
                        max
                    ))
                } else {
                    None
                }
            }
            FilterRule::AllowedTopics(allowed) => {
                if !allowed.contains(&msg.topic) {
                    Some(format!("topic '{}' is not in the allowed list", msg.topic))
                } else {
                    None
                }
            }
            FilterRule::BlockedSenders(blocked) => {
                if blocked.contains(&msg.sender) {
                    Some(format!("sender '{}' is blocked", msg.sender))
                } else {
                    None
                }
            }
            FilterRule::MinInterval { topic, min_ms } => {
                if msg.topic != *topic {
                    return None;
                }
                if let Some(&last) = self.topic_last_seen.get(topic.as_str()) {
                    let elapsed = now.saturating_sub(last);
                    if elapsed < *min_ms {
                        return Some(format!(
                            "topic '{}' rate-limited: only {}ms since last message (minimum {}ms)",
                            topic, elapsed, min_ms
                        ));
                    }
                }
                None
            }
        }
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Seen-ID management
    // ──────────────────────────────────────────────────────────────────────────

    /// Returns `true` if `id` is present in the seen-ID map **and** the
    /// recorded timestamp is within the deduplication window.
    #[inline]
    pub fn is_duplicate(&self, id: &MessageId, now: u64) -> bool {
        if let Some(&seen_at) = self.seen_ids.get(id) {
            now.saturating_sub(seen_at) < self.config.dedup_window_ms
        } else {
            false
        }
    }

    /// Remove seen-ID entries that are older than `dedup_window_ms`.
    ///
    /// If the map still exceeds `max_seen_ids` after TTL eviction, the oldest
    /// `(current_len - max_seen_ids)` entries are removed.
    ///
    /// Returns the number of entries removed.
    pub fn evict_expired_seen(&mut self, now: u64) -> usize {
        let window = self.config.dedup_window_ms;
        let before = self.seen_ids.len();

        self.seen_ids
            .retain(|_, &mut ts| now.saturating_sub(ts) < window);

        // If still over capacity, drop the oldest entries
        let max = self.config.max_seen_ids;
        if self.seen_ids.len() > max {
            let overflow = self.seen_ids.len() - max;
            // Collect and sort by timestamp ascending (oldest first)
            let mut entries: Vec<(MessageId, u64)> =
                self.seen_ids.iter().map(|(&id, &ts)| (id, ts)).collect();
            entries.sort_unstable_by_key(|&(_, ts)| ts);
            for (id, _) in entries.iter().take(overflow) {
                self.seen_ids.remove(id);
            }
        }

        before - self.seen_ids.len()
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Rule management
    // ──────────────────────────────────────────────────────────────────────────

    /// Append a rule to the end of the rule list.
    pub fn add_rule(&mut self, rule: FilterRule) {
        self.config.rules.push(rule);
    }

    /// Remove the rule at position `idx`.  Returns `true` on success, `false`
    /// if `idx` is out of bounds.
    pub fn remove_rule_by_index(&mut self, idx: usize) -> bool {
        if idx < self.config.rules.len() {
            self.config.rules.remove(idx);
            true
        } else {
            false
        }
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Statistics
    // ──────────────────────────────────────────────────────────────────────────

    /// Return a snapshot of the filter's activity counters.
    pub fn filter_stats(&self) -> FilterStats {
        let total = self.accepted + self.rejected + self.duplicates;
        let accept_rate = if total == 0 {
            0.0
        } else {
            self.accepted as f64 / total as f64
        };
        FilterStats {
            accepted: self.accepted,
            rejected: self.rejected,
            duplicates: self.duplicates,
            seen_ids_count: self.seen_ids.len(),
            accept_rate,
        }
    }
}

impl std::fmt::Debug for GossipMessageFilter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GossipMessageFilter")
            .field("rules_count", &self.config.rules.len())
            .field("dedup_window_ms", &self.config.dedup_window_ms)
            .field("max_seen_ids", &self.config.max_seen_ids)
            .field("seen_ids_count", &self.seen_ids.len())
            .field("accepted", &self.accepted)
            .field("rejected", &self.rejected)
            .field("duplicates", &self.duplicates)
            .finish()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::{
        fnv1a_64, FilterConfig, FilterRule, FilterStats, FilterVerdict, GossipMessage,
        GossipMessageFilter, MessageId,
    };

    // ── Helpers ────────────────────────────────────────────────────────────────

    fn make_msg(topic: &str, sender: &str, data: &[u8], hop_count: u8) -> GossipMessage {
        GossipMessage {
            id: MessageId::from_content(data),
            topic: topic.to_string(),
            sender: sender.to_string(),
            data: data.to_vec(),
            received_at: 0,
            hop_count,
        }
    }

    fn default_filter() -> GossipMessageFilter {
        GossipMessageFilter::new(FilterConfig::default())
    }

    // ── MessageId ──────────────────────────────────────────────────────────────

    #[test]
    fn message_id_from_content_is_deterministic() {
        let a = MessageId::from_content(b"hello");
        let b = MessageId::from_content(b"hello");
        assert_eq!(a, b);
    }

    #[test]
    fn message_id_differs_for_different_content() {
        let a = MessageId::from_content(b"hello");
        let b = MessageId::from_content(b"world");
        assert_ne!(a, b);
    }

    #[test]
    fn message_id_is_32_bytes() {
        let id = MessageId::from_content(b"test");
        assert_eq!(id.0.len(), 32);
    }

    #[test]
    fn message_id_words_are_consistent() {
        let data = b"consistency-check";
        let hash = fnv1a_64(data);
        let w0 = hash;
        let w1 = hash ^ hash.swap_bytes();
        let w2 = hash.rotate_left(32);
        let w3 = w0 ^ w1 ^ w2;

        let id = MessageId::from_content(data);
        assert_eq!(id.0[0..8], w0.to_le_bytes());
        assert_eq!(id.0[8..16], w1.to_le_bytes());
        assert_eq!(id.0[16..24], w2.to_le_bytes());
        assert_eq!(id.0[24..32], w3.to_le_bytes());
    }

    #[test]
    fn message_id_debug_and_display_produce_hex() {
        let id = MessageId::from_content(b"hex");
        let s = format!("{id}");
        assert_eq!(s.len(), 64); // 32 bytes × 2 hex chars
        let d = format!("{id:?}");
        assert!(d.starts_with("MessageId("));
        assert!(d.ends_with(')'));
    }

    #[test]
    fn message_id_clone_and_eq() {
        let id = MessageId::from_content(b"clone");
        #[allow(clippy::clone_on_copy)]
        let cloned = id.clone();
        assert_eq!(id, cloned);
    }

    #[test]
    fn message_id_hash_consistency() {
        use std::collections::HashSet;
        let id = MessageId::from_content(b"hash");
        let mut set = HashSet::new();
        set.insert(id);
        assert!(set.contains(&id));
    }

    #[test]
    fn message_id_empty_content() {
        let id = MessageId::from_content(b"");
        // Should not panic; all bytes still form a valid 32-byte array
        assert_eq!(id.0.len(), 32);
    }

    // ── Basic accept / duplicate ───────────────────────────────────────────────

    #[test]
    fn accept_new_message() {
        let mut f = default_filter();
        let msg = make_msg("topic", "peer", b"payload", 0);
        assert_eq!(f.filter(&msg, 1_000), FilterVerdict::Accept);
    }

    #[test]
    fn duplicate_same_id_within_window() {
        let mut f = default_filter();
        let msg = make_msg("topic", "peer", b"payload", 0);
        assert_eq!(f.filter(&msg, 1_000), FilterVerdict::Accept);
        assert_eq!(f.filter(&msg, 1_500), FilterVerdict::Duplicate);
    }

    #[test]
    fn duplicate_counter_increments() {
        let mut f = default_filter();
        let msg = make_msg("t", "p", b"data", 0);
        f.filter(&msg, 1_000);
        f.filter(&msg, 1_001);
        f.filter(&msg, 1_002);
        assert_eq!(f.duplicates, 2);
    }

    #[test]
    fn accept_counter_increments() {
        let mut f = default_filter();
        f.filter(&make_msg("t", "p", b"a", 0), 1_000);
        f.filter(&make_msg("t", "p", b"b", 0), 1_001);
        assert_eq!(f.accepted, 2);
    }

    #[test]
    fn same_id_after_window_expiry_is_accepted() {
        let mut f = GossipMessageFilter::new(FilterConfig {
            dedup_window_ms: 500,
            ..Default::default()
        });
        let msg = make_msg("t", "p", b"data", 0);
        f.filter(&msg, 1_000);
        // Past the window
        assert_eq!(f.filter(&msg, 1_600), FilterVerdict::Accept);
    }

    // ── MaxHopCount rule ───────────────────────────────────────────────────────

    #[test]
    fn max_hop_count_accept_at_limit() {
        let mut f = GossipMessageFilter::new(FilterConfig {
            rules: vec![FilterRule::MaxHopCount(3)],
            ..Default::default()
        });
        let msg = make_msg("t", "p", b"d", 3);
        assert_eq!(f.filter(&msg, 0), FilterVerdict::Accept);
    }

    #[test]
    fn max_hop_count_reject_above_limit() {
        let mut f = GossipMessageFilter::new(FilterConfig {
            rules: vec![FilterRule::MaxHopCount(3)],
            ..Default::default()
        });
        let msg = make_msg("t", "p", b"d", 4);
        assert!(matches!(f.filter(&msg, 0), FilterVerdict::Reject { .. }));
    }

    #[test]
    fn max_hop_count_reason_contains_values() {
        let mut f = GossipMessageFilter::new(FilterConfig {
            rules: vec![FilterRule::MaxHopCount(2)],
            ..Default::default()
        });
        let msg = make_msg("t", "p", b"d", 5);
        if let FilterVerdict::Reject { reason } = f.filter(&msg, 0) {
            assert!(reason.contains('5'));
            assert!(reason.contains('2'));
        } else {
            panic!("expected Reject");
        }
    }

    // ── MaxDataSize rule ───────────────────────────────────────────────────────

    #[test]
    fn max_data_size_accept_at_limit() {
        let data = vec![0u8; 100];
        let mut f = GossipMessageFilter::new(FilterConfig {
            rules: vec![FilterRule::MaxDataSize(100)],
            ..Default::default()
        });
        let msg = make_msg("t", "p", &data, 0);
        assert_eq!(f.filter(&msg, 0), FilterVerdict::Accept);
    }

    #[test]
    fn max_data_size_reject_above_limit() {
        let data = vec![0u8; 101];
        let mut f = GossipMessageFilter::new(FilterConfig {
            rules: vec![FilterRule::MaxDataSize(100)],
            ..Default::default()
        });
        let msg = make_msg("t", "p", &data, 0);
        assert!(matches!(f.filter(&msg, 0), FilterVerdict::Reject { .. }));
    }

    #[test]
    fn max_data_size_zero_payload_always_accepted() {
        let mut f = GossipMessageFilter::new(FilterConfig {
            rules: vec![FilterRule::MaxDataSize(0)],
            ..Default::default()
        });
        let msg = make_msg("t", "p", b"", 0);
        assert_eq!(f.filter(&msg, 0), FilterVerdict::Accept);
    }

    // ── AllowedTopics rule ─────────────────────────────────────────────────────

    #[test]
    fn allowed_topics_accept_listed_topic() {
        let mut f = GossipMessageFilter::new(FilterConfig {
            rules: vec![FilterRule::AllowedTopics(vec!["blocks".to_string()])],
            ..Default::default()
        });
        let msg = make_msg("blocks", "p", b"d", 0);
        assert_eq!(f.filter(&msg, 0), FilterVerdict::Accept);
    }

    #[test]
    fn allowed_topics_reject_unlisted_topic() {
        let mut f = GossipMessageFilter::new(FilterConfig {
            rules: vec![FilterRule::AllowedTopics(vec!["blocks".to_string()])],
            ..Default::default()
        });
        let msg = make_msg("other", "p", b"d", 0);
        assert!(matches!(f.filter(&msg, 0), FilterVerdict::Reject { .. }));
    }

    // ── BlockedSenders rule ────────────────────────────────────────────────────

    #[test]
    fn blocked_senders_accept_unlisted_sender() {
        let mut f = GossipMessageFilter::new(FilterConfig {
            rules: vec![FilterRule::BlockedSenders(vec!["bad-peer".to_string()])],
            ..Default::default()
        });
        let msg = make_msg("t", "good-peer", b"d", 0);
        assert_eq!(f.filter(&msg, 0), FilterVerdict::Accept);
    }

    #[test]
    fn blocked_senders_reject_listed_sender() {
        let mut f = GossipMessageFilter::new(FilterConfig {
            rules: vec![FilterRule::BlockedSenders(vec!["bad-peer".to_string()])],
            ..Default::default()
        });
        let msg = make_msg("t", "bad-peer", b"d", 0);
        assert!(matches!(f.filter(&msg, 0), FilterVerdict::Reject { .. }));
    }

    // ── MinInterval rule ───────────────────────────────────────────────────────

    #[test]
    fn min_interval_accepts_first_message_on_topic() {
        let mut f = GossipMessageFilter::new(FilterConfig {
            rules: vec![FilterRule::MinInterval {
                topic: "t".to_string(),
                min_ms: 1_000,
            }],
            ..Default::default()
        });
        let msg = make_msg("t", "p", b"d1", 0);
        assert_eq!(f.filter(&msg, 5_000), FilterVerdict::Accept);
    }

    #[test]
    fn min_interval_rejects_message_within_interval() {
        let mut f = GossipMessageFilter::new(FilterConfig {
            rules: vec![FilterRule::MinInterval {
                topic: "t".to_string(),
                min_ms: 1_000,
            }],
            ..Default::default()
        });
        let msg1 = make_msg("t", "p", b"d1", 0);
        let msg2 = make_msg("t", "p", b"d2", 0);
        f.filter(&msg1, 5_000);
        assert!(matches!(
            f.filter(&msg2, 5_500),
            FilterVerdict::Reject { .. }
        ));
    }

    #[test]
    fn min_interval_accepts_message_after_interval() {
        let mut f = GossipMessageFilter::new(FilterConfig {
            rules: vec![FilterRule::MinInterval {
                topic: "t".to_string(),
                min_ms: 1_000,
            }],
            ..Default::default()
        });
        let msg1 = make_msg("t", "p", b"d1", 0);
        let msg2 = make_msg("t", "p", b"d2", 0);
        f.filter(&msg1, 5_000);
        assert_eq!(f.filter(&msg2, 6_001), FilterVerdict::Accept);
    }

    #[test]
    fn min_interval_only_affects_matching_topic() {
        let mut f = GossipMessageFilter::new(FilterConfig {
            rules: vec![FilterRule::MinInterval {
                topic: "restricted".to_string(),
                min_ms: 5_000,
            }],
            ..Default::default()
        });
        let r1 = make_msg("restricted", "p", b"r1", 0);
        let other = make_msg("open", "p", b"o1", 0);
        f.filter(&r1, 1_000);
        // "open" topic is unaffected
        assert_eq!(f.filter(&other, 1_001), FilterVerdict::Accept);
    }

    // ── Rule ordering ──────────────────────────────────────────────────────────

    #[test]
    fn first_failing_rule_wins() {
        let mut f = GossipMessageFilter::new(FilterConfig {
            rules: vec![FilterRule::MaxHopCount(2), FilterRule::MaxDataSize(10)],
            ..Default::default()
        });
        // hop_count > 2 should fail before data size check
        let msg = make_msg("t", "p", b"tiny", 5);
        if let FilterVerdict::Reject { reason } = f.filter(&msg, 0) {
            assert!(reason.contains("hop_count"));
        } else {
            panic!("expected Reject");
        }
    }

    // ── add_rule / remove_rule_by_index ────────────────────────────────────────

    #[test]
    fn add_rule_appends_to_list() {
        let mut f = default_filter();
        f.add_rule(FilterRule::MaxHopCount(5));
        assert_eq!(f.config.rules.len(), 1);
    }

    #[test]
    fn remove_rule_by_index_valid() {
        let mut f = default_filter();
        f.add_rule(FilterRule::MaxHopCount(5));
        f.add_rule(FilterRule::MaxDataSize(1024));
        assert!(f.remove_rule_by_index(0));
        assert_eq!(f.config.rules.len(), 1);
        assert!(matches!(f.config.rules[0], FilterRule::MaxDataSize(1024)));
    }

    #[test]
    fn remove_rule_by_index_out_of_bounds_returns_false() {
        let mut f = default_filter();
        assert!(!f.remove_rule_by_index(0));
    }

    // ── evict_expired_seen ─────────────────────────────────────────────────────

    #[test]
    fn evict_removes_expired_entries() {
        let mut f = GossipMessageFilter::new(FilterConfig {
            dedup_window_ms: 1_000,
            ..Default::default()
        });
        let msg = make_msg("t", "p", b"data", 0);
        f.filter(&msg, 0);
        assert_eq!(f.seen_ids.len(), 1);
        let evicted = f.evict_expired_seen(2_000);
        assert_eq!(evicted, 1);
        assert_eq!(f.seen_ids.len(), 0);
    }

    #[test]
    fn evict_keeps_unexpired_entries() {
        let mut f = GossipMessageFilter::new(FilterConfig {
            dedup_window_ms: 10_000,
            ..Default::default()
        });
        let msg = make_msg("t", "p", b"data", 0);
        f.filter(&msg, 0);
        let evicted = f.evict_expired_seen(5_000);
        assert_eq!(evicted, 0);
        assert_eq!(f.seen_ids.len(), 1);
    }

    #[test]
    fn evict_respects_max_seen_ids_cap() {
        let max = 3usize;
        let mut f = GossipMessageFilter::new(FilterConfig {
            dedup_window_ms: 60_000,
            max_seen_ids: max,
            ..Default::default()
        });
        // Insert 5 distinct messages at different timestamps
        for i in 0u64..5 {
            let data = i.to_le_bytes();
            let msg = make_msg("t", "p", &data, 0);
            f.seen_ids.insert(msg.id, i * 100);
        }
        assert_eq!(f.seen_ids.len(), 5);
        f.evict_expired_seen(1_000); // none expired yet (window = 60s)
        assert!(f.seen_ids.len() <= max);
    }

    // ── filter_stats ───────────────────────────────────────────────────────────

    #[test]
    fn filter_stats_zero_when_empty() {
        let f = default_filter();
        let s = f.filter_stats();
        assert_eq!(
            s,
            FilterStats {
                accepted: 0,
                rejected: 0,
                duplicates: 0,
                seen_ids_count: 0,
                accept_rate: 0.0,
            }
        );
    }

    #[test]
    fn filter_stats_accept_rate_all_accepted() {
        let mut f = default_filter();
        f.filter(&make_msg("t", "p", b"a", 0), 0);
        f.filter(&make_msg("t", "p", b"b", 0), 1);
        let s = f.filter_stats();
        assert!((s.accept_rate - 1.0).abs() < 1e-9);
    }

    #[test]
    fn filter_stats_accept_rate_mixed() {
        let mut f = GossipMessageFilter::new(FilterConfig {
            rules: vec![FilterRule::MaxHopCount(0)],
            ..Default::default()
        });
        // Accept one (hop = 0), reject one (hop = 1)
        f.filter(&make_msg("t", "p", b"ok", 0), 0);
        f.filter(&make_msg("t", "p", b"bad", 1), 1);
        let s = f.filter_stats();
        assert!((s.accept_rate - 0.5).abs() < 1e-9);
    }

    #[test]
    fn filter_stats_counts_duplicates_separately() {
        let mut f = default_filter();
        let msg = make_msg("t", "p", b"d", 0);
        f.filter(&msg, 0);
        f.filter(&msg, 1);
        let s = f.filter_stats();
        assert_eq!(s.accepted, 1);
        assert_eq!(s.duplicates, 1);
        assert_eq!(s.rejected, 0);
    }

    // ── is_duplicate ───────────────────────────────────────────────────────────

    #[test]
    fn is_duplicate_false_for_unknown_id() {
        let f = default_filter();
        let id = MessageId::from_content(b"unknown");
        assert!(!f.is_duplicate(&id, 1_000));
    }

    #[test]
    fn is_duplicate_true_within_window() {
        let mut f = GossipMessageFilter::new(FilterConfig {
            dedup_window_ms: 5_000,
            ..Default::default()
        });
        let msg = make_msg("t", "p", b"x", 0);
        f.filter(&msg, 1_000);
        assert!(f.is_duplicate(&msg.id, 3_000));
    }

    #[test]
    fn is_duplicate_false_after_window() {
        let mut f = GossipMessageFilter::new(FilterConfig {
            dedup_window_ms: 500,
            ..Default::default()
        });
        let msg = make_msg("t", "p", b"x", 0);
        f.filter(&msg, 1_000);
        assert!(!f.is_duplicate(&msg.id, 2_000));
    }

    // ── Debug impl ─────────────────────────────────────────────────────────────

    #[test]
    fn gossip_message_filter_debug() {
        let f = default_filter();
        let s = format!("{f:?}");
        assert!(s.contains("GossipMessageFilter"));
    }

    // ── Combined scenario ─────────────────────────────────────────────────────

    #[test]
    fn combined_rules_all_passing() {
        let mut f = GossipMessageFilter::new(FilterConfig {
            rules: vec![
                FilterRule::MaxHopCount(5),
                FilterRule::MaxDataSize(1024),
                FilterRule::AllowedTopics(vec!["blocks".to_string(), "peers".to_string()]),
                FilterRule::BlockedSenders(vec!["evil".to_string()]),
            ],
            dedup_window_ms: 10_000,
            max_seen_ids: 50_000,
        });
        let msg = make_msg("blocks", "good-peer", b"valid payload", 3);
        assert_eq!(f.filter(&msg, 1_000), FilterVerdict::Accept);
        let stats = f.filter_stats();
        assert_eq!(stats.accepted, 1);
        assert_eq!(stats.rejected, 0);
    }

    #[test]
    fn rejected_message_not_stored_in_seen_ids() {
        let mut f = GossipMessageFilter::new(FilterConfig {
            rules: vec![FilterRule::MaxHopCount(0)],
            ..Default::default()
        });
        let msg = make_msg("t", "p", b"d", 5);
        f.filter(&msg, 0);
        // Rejected messages must NOT be stored; re-sending should still be rejected
        // (not duplicated)
        let verdict = f.filter(&msg, 1);
        assert!(matches!(verdict, FilterVerdict::Reject { .. }));
    }

    #[test]
    fn min_interval_exact_boundary_is_accepted() {
        let mut f = GossipMessageFilter::new(FilterConfig {
            rules: vec![FilterRule::MinInterval {
                topic: "t".to_string(),
                min_ms: 1_000,
            }],
            ..Default::default()
        });
        let msg1 = make_msg("t", "p", b"d1", 0);
        let msg2 = make_msg("t", "p", b"d2", 0);
        f.filter(&msg1, 0);
        // elapsed == min_ms; should accept (strictly less-than check)
        assert_eq!(f.filter(&msg2, 1_000), FilterVerdict::Accept);
    }
}
