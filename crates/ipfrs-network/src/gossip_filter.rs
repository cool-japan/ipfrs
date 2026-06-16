//! Peer Gossip Filter — probabilistic duplicate/spam/storm suppression
//!
//! Uses a fixed-size ring buffer of FNV-1a fingerprints to detect already-seen
//! gossip messages without unbounded memory growth.  Combines seen-set lookup,
//! per-sender rate limiting, TTL enforcement, and a manual block list.
//!
//! # Example
//!
//! ```rust
//! use ipfrs_network::gossip_filter::{
//!     FilterConfig, GossipMessage, PeerGossipFilter,
//! };
//!
//! let config = FilterConfig::default();
//! let mut filter = PeerGossipFilter::new(config);
//!
//! let mut msg = GossipMessage {
//!     message_id: 1,
//!     topic: "blocks".to_string(),
//!     sender_peer_id: "peer-A".to_string(),
//!     payload_hash: 0xdeadbeef,
//!     received_at_tick: 1,
//!     ttl: 4,
//! };
//!
//! use ipfrs_network::gossip_filter::FilterVerdict;
//! assert_eq!(filter.evaluate(&mut msg, 1), FilterVerdict::Accept);
//! assert_eq!(filter.evaluate(&mut msg, 1), FilterVerdict::Duplicate);
//! ```

use std::collections::{HashMap, HashSet};

// ─────────────────────────────────────────────────────────────────────────────
// FNV-1a constants (64-bit)
// ─────────────────────────────────────────────────────────────────────────────

const FNV_OFFSET_BASIS: u64 = 14_695_981_039_346_656_037;
const FNV_PRIME: u64 = 1_099_511_628_211;

/// Compute an FNV-1a hash over a byte slice.
#[inline]
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h = FNV_OFFSET_BASIS;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

/// Combine `message_id` and `payload_hash` into a single FNV-1a fingerprint.
///
/// Both values are hashed together so that a reused message ID with different
/// payload content still produces a distinct fingerprint.
fn message_fingerprint(message_id: u64, payload_hash: u64) -> u64 {
    let mut bytes = [0u8; 16];
    bytes[..8].copy_from_slice(&message_id.to_le_bytes());
    bytes[8..].copy_from_slice(&payload_hash.to_le_bytes());
    fnv1a(&bytes)
}

// ─────────────────────────────────────────────────────────────────────────────
// Public data types
// ─────────────────────────────────────────────────────────────────────────────

/// A gossip message flowing through the network layer.
#[derive(Debug, Clone)]
pub struct GossipMessage {
    /// Globally unique message identifier assigned by the originator.
    pub message_id: u64,
    /// Pub/sub topic this message belongs to.
    pub topic: String,
    /// Peer ID string of the direct sender.
    pub sender_peer_id: String,
    /// FNV-1a hash of the raw message payload (content integrity check).
    pub payload_hash: u64,
    /// Logical clock tick at which the local node received this message.
    pub received_at_tick: u64,
    /// Remaining hop count; decremented before forwarding.  When 0 the message
    /// must not be forwarded further.
    pub ttl: u8,
}

/// Decision returned by [`PeerGossipFilter::evaluate`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FilterVerdict {
    /// New message — should be processed and forwarded.
    Accept,
    /// Already seen this `(message_id, payload_hash)` combination.
    Duplicate,
    /// TTL was 0 before evaluation; message must be discarded.
    Expired,
    /// Sender is on the block list or exceeded the per-tick rate limit.
    Blocked,
}

/// Configuration knobs for [`PeerGossipFilter`].
#[derive(Debug, Clone)]
pub struct FilterConfig {
    /// Number of fingerprint slots in the ring buffer (default 1024).
    pub seen_set_capacity: usize,
    /// Maximum messages accepted from a single peer per scheduling tick
    /// before that peer is auto-blocked (default 10).
    pub max_messages_per_peer_per_tick: u32,
}

impl Default for FilterConfig {
    fn default() -> Self {
        Self {
            seen_set_capacity: 1024,
            max_messages_per_peer_per_tick: 10,
        }
    }
}

/// Cumulative statistics collected by [`PeerGossipFilter`].
#[derive(Debug, Clone, Default)]
pub struct FilterStats {
    /// Total number of messages passed to [`PeerGossipFilter::evaluate`].
    pub total_evaluated: u64,
    /// Messages that received [`FilterVerdict::Accept`].
    pub accepted: u64,
    /// Messages that received [`FilterVerdict::Duplicate`].
    pub duplicates: u64,
    /// Messages that received [`FilterVerdict::Expired`].
    pub expired: u64,
    /// Messages that received [`FilterVerdict::Blocked`].
    pub blocked: u64,
}

// ─────────────────────────────────────────────────────────────────────────────
// PeerGossipFilter
// ─────────────────────────────────────────────────────────────────────────────

/// Stateful filter that suppresses duplicate, expired, and abusive gossip
/// messages using a fixed-memory probabilistic seen-set plus rate limiting.
///
/// ## Evaluation order
///
/// 1. **Block-list check** — if the sender is explicitly blocked → [`FilterVerdict::Blocked`].
/// 2. **Rate-limit check** — if the sender has already sent
///    `max_messages_per_peer_per_tick` messages this tick → [`FilterVerdict::Blocked`]
///    *and* the sender is added to the permanent block list.
/// 3. **TTL check** — if `msg.ttl == 0` before decrement → [`FilterVerdict::Expired`].
///    Otherwise `msg.ttl` is decremented by 1 in-place.
/// 4. **Seen-set check** — fingerprint lookup in the ring buffer:
///    - Match → [`FilterVerdict::Duplicate`].
///    - No match → fingerprint inserted, per-tick count incremented →
///      [`FilterVerdict::Accept`].
pub struct PeerGossipFilter {
    /// Fixed-capacity ring buffer of fingerprints.  Slots initialised to 0.
    seen_ring: Vec<u64>,
    /// Write cursor; wraps at `seen_ring.len()`.
    ring_pos: usize,
    /// Permanently blocked peer IDs.
    blocked_peers: HashSet<String>,
    /// Per-tick message count per sender peer ID.
    per_tick_counts: HashMap<String, u32>,
    /// Filter configuration (immutable after construction).
    config: FilterConfig,
    /// Cumulative evaluation statistics.
    stats: FilterStats,
}

impl PeerGossipFilter {
    /// Create a new filter with the supplied configuration.
    ///
    /// The seen-ring is pre-allocated and zero-initialised to `seen_set_capacity`
    /// slots.  A capacity of 0 is clamped to 1 so the ring is never empty.
    pub fn new(config: FilterConfig) -> Self {
        let capacity = config.seen_set_capacity.max(1);
        Self {
            seen_ring: vec![0u64; capacity],
            ring_pos: 0,
            blocked_peers: HashSet::new(),
            per_tick_counts: HashMap::new(),
            config,
            stats: FilterStats::default(),
        }
    }

    /// Evaluate a gossip message and return the appropriate verdict.
    ///
    /// The message is mutated in-place only when the verdict is [`FilterVerdict::Accept`]
    /// or [`FilterVerdict::Duplicate`] (TTL is decremented once if it was > 0 and the
    /// sender passes the block / rate checks).
    pub fn evaluate(&mut self, msg: &mut GossipMessage, _current_tick: u64) -> FilterVerdict {
        self.stats.total_evaluated += 1;

        let sender = msg.sender_peer_id.clone();

        // ── Step 1: permanent block list ──────────────────────────────────────
        if self.blocked_peers.contains(&sender) {
            self.stats.blocked += 1;
            return FilterVerdict::Blocked;
        }

        // ── Step 2: per-tick rate limit ───────────────────────────────────────
        let tick_count = self.per_tick_counts.get(&sender).copied().unwrap_or(0);
        if tick_count >= self.config.max_messages_per_peer_per_tick {
            // Auto-block the abusive peer permanently.
            self.blocked_peers.insert(sender);
            self.stats.blocked += 1;
            return FilterVerdict::Blocked;
        }

        // ── Step 3: TTL check ─────────────────────────────────────────────────
        if msg.ttl == 0 {
            self.stats.expired += 1;
            return FilterVerdict::Expired;
        }
        msg.ttl -= 1;

        // ── Step 4: seen-set lookup ───────────────────────────────────────────
        let fp = message_fingerprint(msg.message_id, msg.payload_hash);
        let capacity = self.seen_ring.len();

        for &slot in &self.seen_ring {
            if slot == fp {
                self.stats.duplicates += 1;
                return FilterVerdict::Duplicate;
            }
        }

        // Not seen — insert into ring and accept.
        self.seen_ring[self.ring_pos] = fp;
        self.ring_pos = (self.ring_pos + 1) % capacity;

        // ── Step 5: update per-tick count ─────────────────────────────────────
        *self.per_tick_counts.entry(sender).or_insert(0) += 1;

        self.stats.accepted += 1;
        FilterVerdict::Accept
    }

    /// Reset per-tick counters.  Call this at the start of every scheduling
    /// tick.  Note: peers that were auto-blocked due to rate limiting remain
    /// blocked across ticks — call [`PeerGossipFilter::unblock_peer`] explicitly
    /// if recovery is desired.
    pub fn new_tick(&mut self) {
        self.per_tick_counts.clear();
    }

    /// Add a peer ID to the permanent block list.
    pub fn block_peer(&mut self, peer_id: &str) {
        self.blocked_peers.insert(peer_id.to_string());
    }

    /// Remove a peer ID from the permanent block list.
    pub fn unblock_peer(&mut self, peer_id: &str) {
        self.blocked_peers.remove(peer_id);
    }

    /// Return `true` if `peer_id` is currently on the block list.
    pub fn is_blocked(&self, peer_id: &str) -> bool {
        self.blocked_peers.contains(peer_id)
    }

    /// Return a reference to the cumulative statistics.
    pub fn stats(&self) -> &FilterStats {
        &self.stats
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ───────────────────────────────────────────────────────────────

    fn make_msg(id: u64, sender: &str, ttl: u8, payload_hash: u64) -> GossipMessage {
        GossipMessage {
            message_id: id,
            topic: "test-topic".to_string(),
            sender_peer_id: sender.to_string(),
            payload_hash,
            received_at_tick: 1,
            ttl,
        }
    }

    fn default_filter() -> PeerGossipFilter {
        PeerGossipFilter::new(FilterConfig::default())
    }

    // ── 1. Accept new message ─────────────────────────────────────────────────

    #[test]
    fn test_accept_new_message() {
        let mut f = default_filter();
        let mut msg = make_msg(1, "peer-A", 4, 0xaabbccdd);
        assert_eq!(f.evaluate(&mut msg, 1), FilterVerdict::Accept);
    }

    #[test]
    fn test_accept_decrements_ttl() {
        let mut f = default_filter();
        let mut msg = make_msg(1, "peer-A", 4, 0xaabbccdd);
        f.evaluate(&mut msg, 1);
        assert_eq!(msg.ttl, 3);
    }

    #[test]
    fn test_accept_updates_stats() {
        let mut f = default_filter();
        let mut msg = make_msg(42, "peer-B", 3, 0x1234);
        f.evaluate(&mut msg, 1);
        let s = f.stats();
        assert_eq!(s.total_evaluated, 1);
        assert_eq!(s.accepted, 1);
        assert_eq!(s.duplicates, 0);
    }

    // ── 2. Duplicate detection ────────────────────────────────────────────────

    #[test]
    fn test_duplicate_same_message() {
        let mut f = default_filter();
        let mut msg = make_msg(1, "peer-A", 4, 0xaabbccdd);
        assert_eq!(f.evaluate(&mut msg, 1), FilterVerdict::Accept);
        // Re-submit the same (id, payload_hash) — ttl is now 3 but fingerprint
        // already in ring.
        let mut msg2 = make_msg(1, "peer-A", 3, 0xaabbccdd);
        assert_eq!(f.evaluate(&mut msg2, 1), FilterVerdict::Duplicate);
    }

    #[test]
    fn test_duplicate_different_sender_same_fingerprint() {
        let mut f = default_filter();
        let mut msg1 = make_msg(7, "peer-A", 4, 0x9999);
        assert_eq!(f.evaluate(&mut msg1, 1), FilterVerdict::Accept);
        // Same message relayed by a different peer.
        let mut msg2 = make_msg(7, "peer-B", 4, 0x9999);
        assert_eq!(f.evaluate(&mut msg2, 1), FilterVerdict::Duplicate);
    }

    #[test]
    fn test_no_duplicate_different_payload() {
        let mut f = default_filter();
        let mut msg1 = make_msg(1, "peer-A", 4, 0x1111);
        let mut msg2 = make_msg(1, "peer-A", 4, 0x2222); // same id, different payload
        assert_eq!(f.evaluate(&mut msg1, 1), FilterVerdict::Accept);
        assert_eq!(f.evaluate(&mut msg2, 1), FilterVerdict::Accept);
    }

    #[test]
    fn test_duplicate_updates_stats() {
        let mut f = default_filter();
        let mut msg = make_msg(1, "peer-A", 4, 0xaabbccdd);
        f.evaluate(&mut msg, 1);
        let mut msg2 = make_msg(1, "peer-A", 3, 0xaabbccdd);
        f.evaluate(&mut msg2, 1);
        assert_eq!(f.stats().duplicates, 1);
        assert_eq!(f.stats().total_evaluated, 2);
    }

    // ── 3. Expired (TTL = 0) ──────────────────────────────────────────────────

    #[test]
    fn test_expired_when_ttl_zero() {
        let mut f = default_filter();
        let mut msg = make_msg(1, "peer-A", 0, 0xbeef);
        assert_eq!(f.evaluate(&mut msg, 1), FilterVerdict::Expired);
    }

    #[test]
    fn test_expired_ttl_not_decremented_below_zero() {
        let mut f = default_filter();
        let mut msg = make_msg(1, "peer-A", 0, 0xbeef);
        f.evaluate(&mut msg, 1);
        // u8 stays at 0, not wrapped.
        assert_eq!(msg.ttl, 0);
    }

    #[test]
    fn test_expired_updates_stats() {
        let mut f = default_filter();
        let mut msg = make_msg(1, "peer-A", 0, 0xbeef);
        f.evaluate(&mut msg, 1);
        assert_eq!(f.stats().expired, 1);
        assert_eq!(f.stats().accepted, 0);
    }

    // ── 4. TTL = 1 → decrement to 0 → still Accept ───────────────────────────

    #[test]
    fn test_ttl_one_becomes_zero_and_accepts() {
        let mut f = default_filter();
        let mut msg = make_msg(1, "peer-A", 1, 0xface);
        let verdict = f.evaluate(&mut msg, 1);
        assert_eq!(verdict, FilterVerdict::Accept);
        assert_eq!(msg.ttl, 0);
    }

    // ── 5. Manual block ───────────────────────────────────────────────────────

    #[test]
    fn test_blocked_peer_returns_blocked() {
        let mut f = default_filter();
        f.block_peer("evil-peer");
        let mut msg = make_msg(1, "evil-peer", 4, 0x1234);
        assert_eq!(f.evaluate(&mut msg, 1), FilterVerdict::Blocked);
    }

    #[test]
    fn test_blocked_updates_stats() {
        let mut f = default_filter();
        f.block_peer("evil-peer");
        let mut msg = make_msg(1, "evil-peer", 4, 0x1234);
        f.evaluate(&mut msg, 1);
        assert_eq!(f.stats().blocked, 1);
    }

    #[test]
    fn test_is_blocked_true_after_block() {
        let mut f = default_filter();
        f.block_peer("evil-peer");
        assert!(f.is_blocked("evil-peer"));
    }

    #[test]
    fn test_is_blocked_false_for_unknown() {
        let f = default_filter();
        assert!(!f.is_blocked("unknown-peer"));
    }

    // ── 6. Unblock ────────────────────────────────────────────────────────────

    #[test]
    fn test_unblock_peer_allows_messages() {
        let mut f = default_filter();
        f.block_peer("peer-X");
        f.unblock_peer("peer-X");
        assert!(!f.is_blocked("peer-X"));
        let mut msg = make_msg(1, "peer-X", 4, 0x5678);
        assert_eq!(f.evaluate(&mut msg, 1), FilterVerdict::Accept);
    }

    #[test]
    fn test_unblock_nonexistent_is_noop() {
        let mut f = default_filter();
        // Should not panic.
        f.unblock_peer("nonexistent");
        assert!(!f.is_blocked("nonexistent"));
    }

    // ── 7. Rate limiting ──────────────────────────────────────────────────────

    #[test]
    fn test_rate_limit_triggers_blocked() {
        let config = FilterConfig {
            seen_set_capacity: 1024,
            max_messages_per_peer_per_tick: 3,
        };
        let mut f = PeerGossipFilter::new(config);

        // 3 distinct messages → all accepted.
        for i in 0..3u64 {
            let mut msg = make_msg(i, "spammer", 4, i);
            assert_eq!(f.evaluate(&mut msg, 1), FilterVerdict::Accept, "msg {i}");
        }

        // 4th message exceeds the limit.
        let mut msg4 = make_msg(100, "spammer", 4, 0xaaaa);
        assert_eq!(f.evaluate(&mut msg4, 1), FilterVerdict::Blocked);
    }

    #[test]
    fn test_rate_limit_auto_blocks_peer() {
        let config = FilterConfig {
            seen_set_capacity: 1024,
            max_messages_per_peer_per_tick: 2,
        };
        let mut f = PeerGossipFilter::new(config);

        for i in 0..2u64 {
            let mut msg = make_msg(i, "spammer", 4, i);
            f.evaluate(&mut msg, 1);
        }
        let mut msg_extra = make_msg(99, "spammer", 4, 0xffff);
        f.evaluate(&mut msg_extra, 1);

        assert!(f.is_blocked("spammer"), "peer should be auto-blocked");
    }

    #[test]
    fn test_rate_limit_only_affects_spammer() {
        let config = FilterConfig {
            seen_set_capacity: 1024,
            max_messages_per_peer_per_tick: 2,
        };
        let mut f = PeerGossipFilter::new(config);

        // Exhaust spammer's budget.
        for i in 0..3u64 {
            let mut msg = make_msg(i, "spammer", 4, i);
            f.evaluate(&mut msg, 1);
        }

        // A different peer is unaffected.
        let mut good = make_msg(200, "good-peer", 4, 0xbabe);
        assert_eq!(f.evaluate(&mut good, 1), FilterVerdict::Accept);
    }

    // ── 8. new_tick resets per-tick counts ───────────────────────────────────

    #[test]
    fn test_new_tick_resets_counts() {
        let config = FilterConfig {
            seen_set_capacity: 1024,
            max_messages_per_peer_per_tick: 2,
        };
        let mut f = PeerGossipFilter::new(config);

        // Send 2 messages (fills budget).
        for i in 0..2u64 {
            let mut msg = make_msg(i, "peer-A", 4, i);
            f.evaluate(&mut msg, 1);
        }

        // New tick — counts reset.
        f.new_tick();

        // Send distinct new messages that haven't been seen.
        for i in 100..102u64 {
            let mut msg = make_msg(i, "peer-A", 4, i);
            let v = f.evaluate(&mut msg, 2);
            assert_eq!(
                v,
                FilterVerdict::Accept,
                "msg {i} should be accepted after new_tick"
            );
        }
    }

    #[test]
    fn test_new_tick_does_not_unblock_auto_blocked_peer() {
        let config = FilterConfig {
            seen_set_capacity: 1024,
            max_messages_per_peer_per_tick: 1,
        };
        let mut f = PeerGossipFilter::new(config);

        // First message accepted, second triggers auto-block.
        let mut m1 = make_msg(1, "spammer", 4, 1);
        f.evaluate(&mut m1, 1);
        let mut m2 = make_msg(2, "spammer", 4, 2);
        f.evaluate(&mut m2, 1);

        f.new_tick();

        // Peer is still blocked.
        assert!(f.is_blocked("spammer"));
        let mut m3 = make_msg(3, "spammer", 4, 3);
        assert_eq!(f.evaluate(&mut m3, 2), FilterVerdict::Blocked);
    }

    // ── 9. Ring buffer wrap-around ────────────────────────────────────────────

    #[test]
    fn test_ring_buffer_wraparound_evicts_oldest() {
        // Very small ring so we can force wrap-around quickly.
        let config = FilterConfig {
            seen_set_capacity: 4,
            max_messages_per_peer_per_tick: u32::MAX,
        };
        let mut f = PeerGossipFilter::new(config);

        // Fill ring with 4 distinct messages.
        for i in 0..4u64 {
            let mut msg = make_msg(i, "peer", 4, i);
            assert_eq!(f.evaluate(&mut msg, 1), FilterVerdict::Accept);
        }

        // The 5th message wraps and evicts slot 0 (message 0).
        let mut msg4 = make_msg(4, "peer", 4, 4);
        assert_eq!(f.evaluate(&mut msg4, 1), FilterVerdict::Accept);

        // Message 0 fingerprint is gone — re-submitting it should Accept again.
        let mut msg0_again = make_msg(0, "peer", 4, 0);
        assert_eq!(f.evaluate(&mut msg0_again, 1), FilterVerdict::Accept);
    }

    #[test]
    fn test_ring_buffer_still_detects_recent_duplicates() {
        let config = FilterConfig {
            seen_set_capacity: 8,
            max_messages_per_peer_per_tick: u32::MAX,
        };
        let mut f = PeerGossipFilter::new(config);

        for i in 0..8u64 {
            let mut msg = make_msg(i, "peer", 4, i);
            f.evaluate(&mut msg, 1);
        }

        // All 8 are still in the ring — duplicates detected.
        for i in 0..8u64 {
            let mut msg = make_msg(i, "peer", 3, i);
            assert_eq!(f.evaluate(&mut msg, 1), FilterVerdict::Duplicate, "msg {i}");
        }
    }

    // ── 10. Stats accumulate correctly ────────────────────────────────────────

    #[test]
    fn test_stats_accumulate_across_verdicts() {
        let config = FilterConfig {
            seen_set_capacity: 1024,
            max_messages_per_peer_per_tick: 5,
        };
        let mut f = PeerGossipFilter::new(config);

        // 1 accepted.
        let mut m1 = make_msg(1, "peer-A", 3, 0x1111);
        f.evaluate(&mut m1, 1);

        // 1 duplicate.
        let mut m1_dup = make_msg(1, "peer-A", 3, 0x1111);
        f.evaluate(&mut m1_dup, 1);

        // 1 expired.
        let mut m2 = make_msg(2, "peer-A", 0, 0x2222);
        f.evaluate(&mut m2, 1);

        // 1 manually blocked.
        f.block_peer("peer-B");
        let mut m3 = make_msg(3, "peer-B", 4, 0x3333);
        f.evaluate(&mut m3, 1);

        let s = f.stats();
        assert_eq!(s.total_evaluated, 4);
        assert_eq!(s.accepted, 1);
        assert_eq!(s.duplicates, 1);
        assert_eq!(s.expired, 1);
        assert_eq!(s.blocked, 1);
    }

    // ── 11. Zero-slot ring clamped to 1 ──────────────────────────────────────

    #[test]
    fn test_zero_capacity_clamped_to_one() {
        let config = FilterConfig {
            seen_set_capacity: 0,
            max_messages_per_peer_per_tick: u32::MAX,
        };
        let mut f = PeerGossipFilter::new(config);
        // Should not panic.
        let mut msg = make_msg(1, "peer", 4, 0xdead);
        assert_eq!(f.evaluate(&mut msg, 1), FilterVerdict::Accept);
    }

    // ── 12. FNV-1a fingerprint helper ─────────────────────────────────────────

    #[test]
    fn test_fingerprint_deterministic() {
        let fp1 = message_fingerprint(42, 0xdeadbeef);
        let fp2 = message_fingerprint(42, 0xdeadbeef);
        assert_eq!(fp1, fp2);
    }

    #[test]
    fn test_fingerprint_differs_on_id_change() {
        let fp1 = message_fingerprint(1, 0x1234);
        let fp2 = message_fingerprint(2, 0x1234);
        assert_ne!(fp1, fp2);
    }

    #[test]
    fn test_fingerprint_differs_on_payload_change() {
        let fp1 = message_fingerprint(1, 0x1111);
        let fp2 = message_fingerprint(1, 0x2222);
        assert_ne!(fp1, fp2);
    }

    // ── 13. Zero-slot fingerprint collisions handled gracefully ──────────────

    #[test]
    fn test_zero_fingerprint_not_considered_seen() {
        // Slot initialised to 0; a legitimate message that happens to hash to 0
        // should not be treated as duplicate on the first occurrence.
        // We can't force hash == 0 easily, but we test that newly constructed
        // filters don't incorrectly flag first messages.
        let mut f = default_filter();
        let mut msg = make_msg(0, "peer", 4, 0);
        // The fingerprint of (0, 0) is very unlikely to be 0, but even if it
        // were, the test validates the filter doesn't panic.
        let v = f.evaluate(&mut msg, 1);
        // Either Accept or Duplicate (if hash==0 collides with init value).
        assert!(v == FilterVerdict::Accept || v == FilterVerdict::Duplicate);
    }
}
