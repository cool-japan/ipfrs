//! Multi-layer flood/DoS protection for P2P networks.
//!
//! Provides three integrated protection layers:
//! 1. **Message deduplication** – FNV-1a–hashed sliding-window cache rejects
//!    replayed messages within a configurable time window.
//! 2. **Per-peer rate limiting** – Token-bucket throttles each peer independently.
//! 3. **Global rate limiting** – A single global token bucket caps aggregate
//!    throughput regardless of how many peers contribute traffic.
//!
//! Peers that violate limits accumulate a violation counter; once they reach
//! [`FloodConfig::ban_threshold`] they are temporarily banned for
//! [`FloodConfig::ban_duration_ms`] milliseconds.
//!
//! ## Example
//!
//! ```rust
//! use ipfrs_network::flood_protection::{
//!     FloodConfig, FloodProtection, MessageId, fnv1a_message_id, CheckResult,
//! };
//!
//! let config = FloodConfig::default();
//! let mut fp = FloodProtection::new(config);
//! let msg_id = fnv1a_message_id(b"hello");
//! let now_ms: u64 = 1_000_000;
//!
//! match fp.check("peer-1", msg_id, now_ms) {
//!     CheckResult::Allow => {
//!         fp.record_allowed("peer-1", msg_id, now_ms);
//!     }
//!     other => eprintln!("blocked: {:?}", other),
//! }
//! ```

use std::collections::{HashMap, VecDeque};

// ── FNV-1a constants ───────────────────────────────────────────────────────────

const FNV_OFFSET_BASIS_64: u64 = 14_695_981_039_346_656_037;
const FNV_PRIME_64: u64 = 1_099_511_628_211;

// ═══════════════════════════════════════════════════════════════════════════════
// Public primitive types
// ═══════════════════════════════════════════════════════════════════════════════

/// A 64-bit message identifier derived via FNV-1a from message bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MessageId(pub u64);

/// Compute a [`MessageId`] from raw message bytes using FNV-1a hashing.
///
/// ```rust
/// use ipfrs_network::flood_protection::{fnv1a_message_id, MessageId};
/// let id = fnv1a_message_id(b"test message");
/// assert_ne!(id.0, 0);
/// ```
#[inline]
pub fn fnv1a_message_id(data: &[u8]) -> MessageId {
    let mut h: u64 = FNV_OFFSET_BASIS_64;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(FNV_PRIME_64);
    }
    MessageId(h)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Configuration
// ═══════════════════════════════════════════════════════════════════════════════

/// Configuration for [`FloodProtection`].
#[derive(Debug, Clone)]
pub struct FloodConfig {
    /// Global maximum requests per second (token bucket).
    pub global_rps: u64,
    /// Per-peer maximum requests per second (token bucket).
    pub per_peer_rps: u64,
    /// Sliding window for message deduplication, in milliseconds.
    pub dedup_window_ms: u64,
    /// Maximum number of entries in the deduplication cache.
    pub dedup_capacity: usize,
    /// Number of violations before a peer is banned.
    pub ban_threshold: u32,
    /// How long a ban lasts, in milliseconds.
    pub ban_duration_ms: u64,
    /// Token-bucket burst capacity = `rps * burst_multiplier`.
    pub burst_multiplier: f64,
}

impl Default for FloodConfig {
    fn default() -> Self {
        Self {
            global_rps: 1_000,
            per_peer_rps: 100,
            dedup_window_ms: 30_000,
            dedup_capacity: 10_000,
            ban_threshold: 10,
            ban_duration_ms: 300_000,
            burst_multiplier: 2.0,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Result / violation types
// ═══════════════════════════════════════════════════════════════════════════════

/// Outcome of a [`FloodProtection::check`] call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CheckResult {
    /// The message is allowed to pass through.
    Allow,
    /// The originating peer is currently banned.
    Banned {
        /// Timestamp (ms) when the ban expires.
        until: u64,
    },
    /// The global token bucket is exhausted.
    GlobalRateLimited,
    /// The per-peer token bucket for this peer is exhausted.
    PeerRateLimited,
    /// This message was seen recently and is a duplicate.
    Duplicate,
}

/// Categories of violations tracked by the flood-protection layer.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ViolationType {
    /// Peer exceeded its per-peer rate limit.
    RateLimitExceeded,
    /// Peer sent a message that was already seen.
    DuplicateMessage,
    /// Peer sent a message while banned.
    BannedPeer,
}

impl ViolationType {
    fn as_str(&self) -> &'static str {
        match self {
            Self::RateLimitExceeded => "RateLimitExceeded",
            Self::DuplicateMessage => "DuplicateMessage",
            Self::BannedPeer => "BannedPeer",
        }
    }
}

/// A single violation event.
#[derive(Debug, Clone)]
pub struct ViolationRecord {
    /// Peer that triggered the violation.
    pub peer_id: String,
    /// Kind of violation.
    pub violation_type: ViolationType,
    /// Wall-clock timestamp of the violation (ms since epoch / reference point).
    pub timestamp: u64,
    /// Optional message that triggered the violation.
    pub message_id: Option<MessageId>,
}

// ═══════════════════════════════════════════════════════════════════════════════
// Per-peer state
// ═══════════════════════════════════════════════════════════════════════════════

/// Runtime state maintained for each peer.
#[derive(Debug, Clone)]
pub struct PeerState {
    /// Peer identifier.
    pub peer_id: String,
    /// Current token count for the per-peer token bucket.
    pub tokens: f64,
    /// Timestamp (ms) of the last token-refill operation.
    pub last_refill: u64,
    /// Cumulative violation count since peer was first seen (or last unbanned).
    pub violation_count: u32,
    /// If `Some(t)`, the peer is banned until timestamp `t`.
    pub banned_until: Option<u64>,
}

impl PeerState {
    fn new(peer_id: &str, initial_tokens: f64, now: u64) -> Self {
        Self {
            peer_id: peer_id.to_string(),
            tokens: initial_tokens,
            last_refill: now,
            violation_count: 0,
            banned_until: None,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Aggregate statistics
// ═══════════════════════════════════════════════════════════════════════════════

/// Snapshot of flood-protection statistics.
#[derive(Debug, Clone)]
pub struct FloodStats {
    /// Number of peers with tracked state.
    pub total_peers: usize,
    /// Number of currently-banned peers.
    pub banned_peers: usize,
    /// Number of entries in the deduplication cache.
    pub dedup_cache_size: usize,
    /// Total messages allowed since creation.
    pub total_allowed: u64,
    /// Total messages blocked since creation.
    pub total_blocked: u64,
    /// Current global token count.
    pub global_tokens: f64,
}

// ═══════════════════════════════════════════════════════════════════════════════
// Main struct
// ═══════════════════════════════════════════════════════════════════════════════

/// Multi-layer flood/DoS protection for P2P networks.
///
/// See the [module-level documentation](self) for a detailed description of each
/// protection layer and usage examples.
pub struct FloodProtection {
    /// Immutable configuration.
    pub config: FloodConfig,
    /// Global token-bucket state.
    pub global_tokens: f64,
    /// Timestamp of last global token refill.
    pub global_last_refill: u64,
    /// Per-peer token-bucket and ban state.
    pub peers: HashMap<String, PeerState>,
    /// Sliding-window deduplication cache: `(message_id, insertion_timestamp_ms)`.
    pub seen_messages: VecDeque<(MessageId, u64)>,
    /// Ring buffer of recent violations (capped at 1 000).
    pub violations: VecDeque<ViolationRecord>,
    /// Monotonically increasing count of allowed messages.
    pub total_allowed: u64,
    /// Monotonically increasing count of blocked messages.
    pub total_blocked: u64,
}

// ── Internal helpers ───────────────────────────────────────────────────────────

impl FloodProtection {
    /// Initial token count for a new bucket = `rps * burst_multiplier`.
    #[inline]
    fn initial_tokens(rps: u64, burst: f64) -> f64 {
        rps as f64 * burst
    }

    /// Refill `tokens` according to elapsed time and return the updated value.
    ///
    /// ```text
    /// new_tokens = min(old + elapsed_ms / 1000 * rps, rps * burst)
    /// ```
    #[inline]
    fn refill(tokens: f64, last_refill: u64, now: u64, rps: u64, burst: f64) -> (f64, u64) {
        let elapsed_ms = now.saturating_sub(last_refill);
        let replenished = tokens + elapsed_ms as f64 / 1_000.0 * rps as f64;
        let cap = rps as f64 * burst;
        (replenished.min(cap), now)
    }

    /// Check whether a `MessageId` is already in the dedup cache.
    fn is_duplicate(&self, msg_id: MessageId) -> bool {
        self.seen_messages.iter().any(|(id, _)| *id == msg_id)
    }
}

// ── Public API ─────────────────────────────────────────────────────────────────

impl FloodProtection {
    /// Create a new [`FloodProtection`] instance with the given configuration.
    pub fn new(config: FloodConfig) -> Self {
        let burst = config.burst_multiplier;
        let g_rps = config.global_rps;
        Self {
            global_tokens: Self::initial_tokens(g_rps, burst),
            global_last_refill: 0,
            peers: HashMap::new(),
            seen_messages: VecDeque::new(),
            violations: VecDeque::new(),
            total_allowed: 0,
            total_blocked: 0,
            config,
        }
    }

    // ── Core check ──────────────────────────────────────────────────────────

    /// Run all protection checks in priority order.
    ///
    /// The checks are applied in this order:
    /// 1. Peer ban check.
    /// 2. Global rate limit.
    /// 3. Per-peer rate limit.
    /// 4. Message deduplication.
    ///
    /// This method is **read-only**: it does not consume tokens or register the
    /// message.  Call [`record_allowed`](Self::record_allowed) after receiving
    /// [`CheckResult::Allow`] to commit the state change.
    ///
    /// Violations triggered here are recorded automatically.
    pub fn check(&mut self, peer_id: &str, message_id: MessageId, now: u64) -> CheckResult {
        // ── 1. Ban check ─────────────────────────────────────────────────────
        if self.is_banned(peer_id, now) {
            let until = self
                .peers
                .get(peer_id)
                .and_then(|p| p.banned_until)
                .unwrap_or(now);
            self.record_violation(peer_id, ViolationType::BannedPeer, Some(message_id), now);
            self.total_blocked += 1;
            return CheckResult::Banned { until };
        }

        // ── 2. Global rate limit ─────────────────────────────────────────────
        let burst = self.config.burst_multiplier;
        let (new_global, new_global_ts) = Self::refill(
            self.global_tokens,
            self.global_last_refill,
            now,
            self.config.global_rps,
            burst,
        );
        self.global_tokens = new_global;
        self.global_last_refill = new_global_ts;

        if self.global_tokens < 1.0 {
            self.total_blocked += 1;
            return CheckResult::GlobalRateLimited;
        }

        // ── 3. Per-peer rate limit ───────────────────────────────────────────
        let peer_rps = self.config.per_peer_rps;
        let peer_burst = self.config.burst_multiplier;
        let initial_tokens = Self::initial_tokens(peer_rps, peer_burst);

        let peer = self
            .peers
            .entry(peer_id.to_string())
            .or_insert_with(|| PeerState::new(peer_id, initial_tokens, now));

        let (new_peer_tokens, new_peer_ts) =
            Self::refill(peer.tokens, peer.last_refill, now, peer_rps, peer_burst);
        peer.tokens = new_peer_tokens;
        peer.last_refill = new_peer_ts;

        if peer.tokens < 1.0 {
            self.record_violation(
                peer_id,
                ViolationType::RateLimitExceeded,
                Some(message_id),
                now,
            );
            self.total_blocked += 1;
            return CheckResult::PeerRateLimited;
        }

        // ── 4. Deduplication ─────────────────────────────────────────────────
        if self.is_duplicate(message_id) {
            self.record_violation(
                peer_id,
                ViolationType::DuplicateMessage,
                Some(message_id),
                now,
            );
            self.total_blocked += 1;
            return CheckResult::Duplicate;
        }

        CheckResult::Allow
    }

    // ── State mutation after allow ───────────────────────────────────────────

    /// Consume one token from the global and per-peer buckets and record the
    /// message in the deduplication cache.
    ///
    /// This **must** be called after [`check`](Self::check) returns
    /// [`CheckResult::Allow`] and before the next call to `check` for the same
    /// peer, otherwise the token counts will be stale.
    pub fn record_allowed(&mut self, peer_id: &str, message_id: MessageId, now: u64) {
        // Consume global token.
        self.global_tokens -= 1.0;

        // Consume per-peer token (create entry if somehow missing).
        let peer_rps = self.config.per_peer_rps;
        let burst = self.config.burst_multiplier;
        let initial_tokens = Self::initial_tokens(peer_rps, burst);

        let peer = self
            .peers
            .entry(peer_id.to_string())
            .or_insert_with(|| PeerState::new(peer_id, initial_tokens, now));
        peer.tokens -= 1.0;

        // Register in dedup cache, evicting overflow if at capacity.
        if self.seen_messages.len() >= self.config.dedup_capacity {
            self.seen_messages.pop_front();
        }
        self.seen_messages.push_back((message_id, now));

        self.total_allowed += 1;
    }

    // ── Violation tracking ───────────────────────────────────────────────────

    /// Record a violation for `peer_id` and ban them if they hit the threshold.
    ///
    /// The violations ring-buffer is capped at 1 000 entries; oldest entries are
    /// silently dropped when the buffer is full.
    pub fn record_violation(
        &mut self,
        peer_id: &str,
        vtype: ViolationType,
        msg_id: Option<MessageId>,
        now: u64,
    ) {
        // Push violation record, respecting the 1 000-entry cap.
        if self.violations.len() >= 1_000 {
            self.violations.pop_front();
        }
        self.violations.push_back(ViolationRecord {
            peer_id: peer_id.to_string(),
            violation_type: vtype,
            timestamp: now,
            message_id: msg_id,
        });

        // Bump the violation counter on the peer and potentially ban them.
        let ban_threshold = self.config.ban_threshold;
        let ban_duration = self.config.ban_duration_ms;
        let peer_rps = self.config.per_peer_rps;
        let burst = self.config.burst_multiplier;
        let initial_tokens = Self::initial_tokens(peer_rps, burst);

        let peer = self
            .peers
            .entry(peer_id.to_string())
            .or_insert_with(|| PeerState::new(peer_id, initial_tokens, now));

        peer.violation_count += 1;
        if peer.violation_count >= ban_threshold {
            peer.banned_until = Some(now + ban_duration);
        }
    }

    // ── Ban management ───────────────────────────────────────────────────────

    /// Return `true` if `peer_id` is currently banned.
    ///
    /// If a ban has expired at `now` the peer's `banned_until` field is cleared
    /// automatically (lazy unban).
    pub fn is_banned(&mut self, peer_id: &str, now: u64) -> bool {
        if let Some(peer) = self.peers.get_mut(peer_id) {
            if let Some(until) = peer.banned_until {
                if now >= until {
                    // Ban has expired – remove it.
                    peer.banned_until = None;
                    return false;
                }
                return true;
            }
        }
        false
    }

    /// Unconditionally lift the ban on `peer_id`.
    pub fn unban(&mut self, peer_id: &str) {
        if let Some(peer) = self.peers.get_mut(peer_id) {
            peer.banned_until = None;
        }
    }

    // ── Eviction ─────────────────────────────────────────────────────────────

    /// Remove deduplication entries older than [`FloodConfig::dedup_window_ms`].
    ///
    /// Returns the number of entries removed.
    pub fn evict_expired_dedup(&mut self, now: u64) -> usize {
        let window = self.config.dedup_window_ms;
        let cutoff = now.saturating_sub(window);
        let before = self.seen_messages.len();
        self.seen_messages.retain(|(_, ts)| *ts > cutoff);
        before - self.seen_messages.len()
    }

    /// Remove unbanned peers whose `last_refill` is older than `max_age_ms`.
    ///
    /// Banned peers are never evicted regardless of age.
    /// Returns the number of peer entries removed.
    pub fn evict_stale_peers(&mut self, max_age_ms: u64, now: u64) -> usize {
        let cutoff = now.saturating_sub(max_age_ms);
        let before = self.peers.len();
        self.peers.retain(|_, p| {
            // Keep if banned (regardless of staleness).
            if p.banned_until.is_some() {
                return true;
            }
            // Keep if recently active.
            p.last_refill > cutoff
        });
        before - self.peers.len()
    }

    // ── Reporting ────────────────────────────────────────────────────────────

    /// Summarise violations recorded since `since` (ms timestamp).
    ///
    /// Returns a map from violation-type name to occurrence count.
    pub fn violation_summary(&self, since: u64) -> HashMap<String, usize> {
        let mut summary: HashMap<String, usize> = HashMap::new();
        for v in &self.violations {
            if v.timestamp >= since {
                *summary
                    .entry(v.violation_type.as_str().to_string())
                    .or_insert(0) += 1;
            }
        }
        summary
    }

    /// Return a point-in-time statistics snapshot.
    pub fn stats(&self, now: u64) -> FloodStats {
        let banned_peers = self
            .peers
            .values()
            .filter(|p| p.banned_until.map(|u| u > now).unwrap_or(false))
            .count();

        FloodStats {
            total_peers: self.peers.len(),
            banned_peers,
            dedup_cache_size: self.seen_messages.len(),
            total_allowed: self.total_allowed,
            total_blocked: self.total_blocked,
            global_tokens: self.global_tokens,
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Tests
// ══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use crate::flood_protection::{
        fnv1a_message_id, CheckResult, FloodConfig, FloodProtection, MessageId, ViolationType,
    };

    // ── helpers ────────────────────────────────────────────────────────────

    fn default_fp() -> FloodProtection {
        FloodProtection::new(FloodConfig::default())
    }

    fn tight_config() -> FloodConfig {
        FloodConfig {
            global_rps: 10,
            per_peer_rps: 5,
            dedup_window_ms: 5_000,
            dedup_capacity: 100,
            ban_threshold: 3,
            ban_duration_ms: 60_000,
            burst_multiplier: 1.0, // no burst headroom
        }
    }

    // ── fnv1a_message_id ───────────────────────────────────────────────────

    #[test]
    fn test_fnv1a_empty_slice() {
        // FNV-1a of empty input equals the offset basis.
        let id = fnv1a_message_id(&[]);
        assert_eq!(id.0, 14_695_981_039_346_656_037_u64);
    }

    #[test]
    fn test_fnv1a_deterministic() {
        assert_eq!(fnv1a_message_id(b"hello"), fnv1a_message_id(b"hello"));
    }

    #[test]
    fn test_fnv1a_different_inputs() {
        assert_ne!(fnv1a_message_id(b"foo"), fnv1a_message_id(b"bar"));
    }

    #[test]
    fn test_fnv1a_single_byte() {
        let id = fnv1a_message_id(&[0x42]);
        assert_ne!(id.0, 0);
    }

    #[test]
    fn test_fnv1a_all_zeros_vs_empty() {
        assert_ne!(fnv1a_message_id(&[0u8]), fnv1a_message_id(&[]));
    }

    #[test]
    fn test_fnv1a_uniqueness_across_messages() {
        let msgs: Vec<&[u8]> = vec![b"msg1", b"msg2", b"msg3", b"hello world", b"ipfrs"];
        let ids: HashSet<u64> = msgs.iter().map(|m| fnv1a_message_id(m).0).collect();
        assert_eq!(ids.len(), msgs.len(), "all hashes should be unique");
    }

    // ── MessageId ─────────────────────────────────────────────────────────

    #[test]
    fn test_message_id_equality() {
        let a = MessageId(42);
        let b = MessageId(42);
        assert_eq!(a, b);
    }

    #[test]
    fn test_message_id_inequality() {
        assert_ne!(MessageId(1), MessageId(2));
    }

    #[test]
    fn test_message_id_copy() {
        let a = MessageId(99);
        let b = a; // copy
        assert_eq!(a, b);
    }

    // ── FloodConfig defaults ───────────────────────────────────────────────

    #[test]
    fn test_default_config() {
        let cfg = FloodConfig::default();
        assert_eq!(cfg.global_rps, 1_000);
        assert_eq!(cfg.per_peer_rps, 100);
        assert_eq!(cfg.dedup_window_ms, 30_000);
        assert_eq!(cfg.dedup_capacity, 10_000);
        assert_eq!(cfg.ban_threshold, 10);
        assert_eq!(cfg.ban_duration_ms, 300_000);
        assert!((cfg.burst_multiplier - 2.0).abs() < f64::EPSILON);
    }

    // ── Basic allow path ──────────────────────────────────────────────────

    #[test]
    fn test_fresh_message_allowed() {
        let mut fp = default_fp();
        let id = fnv1a_message_id(b"new message");
        let result = fp.check("peer-a", id, 1_000_000);
        assert_eq!(result, CheckResult::Allow);
    }

    #[test]
    fn test_record_allowed_increments_counter() {
        let mut fp = default_fp();
        let id = fnv1a_message_id(b"m1");
        fp.check("peer-a", id, 1_000);
        fp.record_allowed("peer-a", id, 1_000);
        assert_eq!(fp.total_allowed, 1);
    }

    #[test]
    fn test_multiple_distinct_messages_allowed() {
        let mut fp = default_fp();
        let now = 1_000_000_u64;
        for i in 0_u8..5 {
            let id = fnv1a_message_id(&[i]);
            assert_eq!(
                fp.check("peer-a", id, now + u64::from(i) * 10),
                CheckResult::Allow
            );
            fp.record_allowed("peer-a", id, now + u64::from(i) * 10);
        }
        assert_eq!(fp.total_allowed, 5);
    }

    // ── Deduplication ─────────────────────────────────────────────────────

    #[test]
    fn test_duplicate_message_blocked() {
        let mut fp = FloodProtection::new(tight_config());
        let id = fnv1a_message_id(b"dup");
        let now = 1_000_u64;
        assert_eq!(fp.check("peer-a", id, now), CheckResult::Allow);
        fp.record_allowed("peer-a", id, now);
        // Same message again – must be rejected.
        let result = fp.check("peer-a", id, now + 1);
        assert_eq!(result, CheckResult::Duplicate);
    }

    #[test]
    fn test_duplicate_from_different_peer_also_blocked() {
        let mut fp = FloodProtection::new(tight_config());
        let id = fnv1a_message_id(b"shared");
        fp.check("peer-a", id, 1_000);
        fp.record_allowed("peer-a", id, 1_000);
        // Different peer sending the same message.
        let result = fp.check("peer-b", id, 1_100);
        assert_eq!(result, CheckResult::Duplicate);
    }

    #[test]
    fn test_evict_expired_dedup_removes_old_entries() {
        let config = FloodConfig {
            dedup_window_ms: 1_000,
            global_rps: 10_000,
            per_peer_rps: 10_000,
            burst_multiplier: 2.0,
            ..FloodConfig::default()
        };
        let mut fp = FloodProtection::new(config);
        let id = fnv1a_message_id(b"old");
        fp.check("peer-a", id, 0);
        fp.record_allowed("peer-a", id, 0);
        assert_eq!(fp.seen_messages.len(), 1);

        // Time advances past the window.
        let removed = fp.evict_expired_dedup(2_000);
        assert_eq!(removed, 1);
        assert_eq!(fp.seen_messages.len(), 0);
    }

    #[test]
    fn test_evict_expired_dedup_keeps_fresh_entries() {
        let config = FloodConfig {
            dedup_window_ms: 60_000,
            global_rps: 10_000,
            per_peer_rps: 10_000,
            burst_multiplier: 2.0,
            ..FloodConfig::default()
        };
        let mut fp = FloodProtection::new(config);
        let id = fnv1a_message_id(b"fresh");
        fp.check("peer-a", id, 50_000);
        fp.record_allowed("peer-a", id, 50_000);
        let removed = fp.evict_expired_dedup(51_000); // only 1 s elapsed
        assert_eq!(removed, 0);
    }

    #[test]
    fn test_dedup_capacity_evicts_oldest() {
        let config = FloodConfig {
            dedup_capacity: 3,
            global_rps: 100_000,
            per_peer_rps: 100_000,
            burst_multiplier: 10.0,
            ..FloodConfig::default()
        };
        let mut fp = FloodProtection::new(config);
        for i in 0_u8..4 {
            let id = fnv1a_message_id(&[i]);
            assert_eq!(
                fp.check("peer-a", id, 1_000 + u64::from(i)),
                CheckResult::Allow
            );
            fp.record_allowed("peer-a", id, 1_000 + u64::from(i));
        }
        // Cache should still be capped at 3.
        assert_eq!(fp.seen_messages.len(), 3);
        // The oldest entry (byte 0) was evicted; sending it again should be allowed.
        let evicted_id = fnv1a_message_id(&[0_u8]);
        assert_eq!(fp.check("peer-a", evicted_id, 2_000), CheckResult::Allow);
    }

    // ── Per-peer rate limiting ─────────────────────────────────────────────

    #[test]
    fn test_per_peer_rate_limit_exhausted() {
        let config = FloodConfig {
            per_peer_rps: 2,
            global_rps: 1_000,
            burst_multiplier: 1.0, // tokens start at 2
            dedup_capacity: 10_000,
            dedup_window_ms: 30_000,
            ..FloodConfig::default()
        };
        let mut fp = FloodProtection::new(config);
        let now = 0_u64;

        // First two messages – consume all tokens.
        let id1 = fnv1a_message_id(b"a");
        assert_eq!(fp.check("p", id1, now), CheckResult::Allow);
        fp.record_allowed("p", id1, now);

        let id2 = fnv1a_message_id(b"b");
        assert_eq!(fp.check("p", id2, now), CheckResult::Allow);
        fp.record_allowed("p", id2, now);

        // Third message – peer bucket empty.
        let id3 = fnv1a_message_id(b"c");
        let result = fp.check("p", id3, now);
        assert_eq!(result, CheckResult::PeerRateLimited);
    }

    #[test]
    fn test_per_peer_token_refill_over_time() {
        let config = FloodConfig {
            per_peer_rps: 1,
            global_rps: 10_000,
            burst_multiplier: 1.0,
            dedup_capacity: 10_000,
            dedup_window_ms: 30_000,
            ..FloodConfig::default()
        };
        let mut fp = FloodProtection::new(config);

        let id1 = fnv1a_message_id(b"x");
        assert_eq!(fp.check("p", id1, 0), CheckResult::Allow);
        fp.record_allowed("p", id1, 0);

        // Immediately after – bucket empty.
        let id2 = fnv1a_message_id(b"y");
        assert_eq!(fp.check("p", id2, 0), CheckResult::PeerRateLimited);

        // After 1 second the token is replenished.
        let id3 = fnv1a_message_id(b"z");
        assert_eq!(fp.check("p", id3, 1_000), CheckResult::Allow);
    }

    #[test]
    fn test_different_peers_have_independent_buckets() {
        let config = FloodConfig {
            per_peer_rps: 1,
            global_rps: 10_000,
            burst_multiplier: 1.0,
            dedup_capacity: 10_000,
            dedup_window_ms: 30_000,
            ..FloodConfig::default()
        };
        let mut fp = FloodProtection::new(config);

        let id_a = fnv1a_message_id(b"pa");
        let id_b = fnv1a_message_id(b"pb");

        assert_eq!(fp.check("peer-a", id_a, 0), CheckResult::Allow);
        fp.record_allowed("peer-a", id_a, 0);
        // peer-a is exhausted but peer-b should still be fine.
        assert_eq!(fp.check("peer-b", id_b, 0), CheckResult::Allow);
    }

    // ── Global rate limiting ───────────────────────────────────────────────

    #[test]
    fn test_global_rate_limit_exhausted() {
        let config = FloodConfig {
            global_rps: 2,
            per_peer_rps: 1_000,
            burst_multiplier: 1.0,
            dedup_capacity: 10_000,
            dedup_window_ms: 30_000,
            ..FloodConfig::default()
        };
        let mut fp = FloodProtection::new(config);
        let now = 0_u64;

        let id1 = fnv1a_message_id(b"g1");
        assert_eq!(fp.check("p1", id1, now), CheckResult::Allow);
        fp.record_allowed("p1", id1, now);

        let id2 = fnv1a_message_id(b"g2");
        assert_eq!(fp.check("p2", id2, now), CheckResult::Allow);
        fp.record_allowed("p2", id2, now);

        // Global bucket exhausted.
        let id3 = fnv1a_message_id(b"g3");
        assert_eq!(fp.check("p3", id3, now), CheckResult::GlobalRateLimited);
    }

    #[test]
    fn test_global_token_refill() {
        let config = FloodConfig {
            global_rps: 1,
            per_peer_rps: 10_000,
            burst_multiplier: 1.0,
            dedup_capacity: 10_000,
            dedup_window_ms: 30_000,
            ..FloodConfig::default()
        };
        let mut fp = FloodProtection::new(config);

        let id1 = fnv1a_message_id(b"r1");
        assert_eq!(fp.check("p", id1, 0), CheckResult::Allow);
        fp.record_allowed("p", id1, 0);

        // No refill yet – blocked.
        let id2 = fnv1a_message_id(b"r2");
        assert_eq!(fp.check("p", id2, 0), CheckResult::GlobalRateLimited);

        // After 1 second – token refilled.
        let id3 = fnv1a_message_id(b"r3");
        assert_eq!(fp.check("p", id3, 1_000), CheckResult::Allow);
    }

    // ── Ban management ────────────────────────────────────────────────────

    #[test]
    fn test_violation_ban_after_threshold() {
        let config = FloodConfig {
            ban_threshold: 2,
            ban_duration_ms: 10_000,
            global_rps: 1_000_000,
            per_peer_rps: 1_000_000,
            burst_multiplier: 10.0,
            dedup_capacity: 10_000,
            dedup_window_ms: 30_000,
        };
        let mut fp = FloodProtection::new(config);
        let now = 1_000_u64;

        fp.record_violation("bad-peer", ViolationType::RateLimitExceeded, None, now);
        assert!(!fp.is_banned("bad-peer", now));
        fp.record_violation("bad-peer", ViolationType::RateLimitExceeded, None, now);
        assert!(fp.is_banned("bad-peer", now));
    }

    #[test]
    fn test_ban_expires_after_duration() {
        let config = FloodConfig {
            ban_threshold: 1,
            ban_duration_ms: 5_000,
            ..FloodConfig::default()
        };
        let mut fp = FloodProtection::new(config);
        let now = 1_000_u64;

        fp.record_violation("bad-peer", ViolationType::BannedPeer, None, now);
        assert!(fp.is_banned("bad-peer", now + 1_000)); // still banned
        assert!(!fp.is_banned("bad-peer", now + 6_000)); // expired
    }

    #[test]
    fn test_unban_clears_ban() {
        let config = FloodConfig {
            ban_threshold: 1,
            ban_duration_ms: 100_000,
            ..FloodConfig::default()
        };
        let mut fp = FloodProtection::new(config);
        fp.record_violation("bad-peer", ViolationType::RateLimitExceeded, None, 0);
        assert!(fp.is_banned("bad-peer", 100));
        fp.unban("bad-peer");
        assert!(!fp.is_banned("bad-peer", 100));
    }

    #[test]
    fn test_check_returns_banned_for_banned_peer() {
        let config = FloodConfig {
            ban_threshold: 1,
            ban_duration_ms: 100_000,
            global_rps: 100_000,
            per_peer_rps: 100_000,
            burst_multiplier: 2.0,
            dedup_capacity: 10_000,
            dedup_window_ms: 30_000,
        };
        let mut fp = FloodProtection::new(config);
        fp.record_violation("evil", ViolationType::RateLimitExceeded, None, 0);
        let id = fnv1a_message_id(b"anything");
        match fp.check("evil", id, 1_000) {
            CheckResult::Banned { .. } => {}
            other => panic!("expected Banned, got {other:?}"),
        }
    }

    #[test]
    fn test_unban_unknown_peer_is_noop() {
        let mut fp = default_fp();
        // Should not panic.
        fp.unban("unknown-peer");
    }

    // ── Stale peer eviction ───────────────────────────────────────────────

    #[test]
    fn test_evict_stale_peers_removes_inactive() {
        let mut fp = default_fp();
        let id = fnv1a_message_id(b"ev");
        fp.check("old-peer", id, 0);
        fp.record_allowed("old-peer", id, 0);
        assert_eq!(fp.peers.len(), 1);

        let removed = fp.evict_stale_peers(1_000, 10_000);
        assert_eq!(removed, 1);
        assert!(fp.peers.is_empty());
    }

    #[test]
    fn test_evict_stale_peers_keeps_banned() {
        let config = FloodConfig {
            ban_threshold: 1,
            ban_duration_ms: 1_000_000,
            ..FloodConfig::default()
        };
        let mut fp = FloodProtection::new(config);
        fp.record_violation("banned-peer", ViolationType::RateLimitExceeded, None, 0);

        // Even though last_refill is 0 and 10 000 ms have elapsed, banned peers stay.
        let removed = fp.evict_stale_peers(1_000, 10_000);
        assert_eq!(removed, 0);
        assert_eq!(fp.peers.len(), 1);
    }

    #[test]
    fn test_evict_stale_peers_keeps_recently_active() {
        let mut fp = default_fp();
        let id = fnv1a_message_id(b"recent");
        fp.check("active-peer", id, 9_500);
        fp.record_allowed("active-peer", id, 9_500);

        // Only 500 ms have elapsed – peer is still fresh.
        let removed = fp.evict_stale_peers(1_000, 10_000);
        assert_eq!(removed, 0);
    }

    // ── Violation summary ─────────────────────────────────────────────────

    #[test]
    fn test_violation_summary_counts_by_type() {
        let mut fp = default_fp();
        fp.record_violation("p", ViolationType::RateLimitExceeded, None, 1_000);
        fp.record_violation("p", ViolationType::DuplicateMessage, None, 2_000);
        fp.record_violation("p", ViolationType::RateLimitExceeded, None, 3_000);

        let summary = fp.violation_summary(0);
        assert_eq!(summary.get("RateLimitExceeded").copied().unwrap_or(0), 2);
        assert_eq!(summary.get("DuplicateMessage").copied().unwrap_or(0), 1);
    }

    #[test]
    fn test_violation_summary_respects_since_filter() {
        let mut fp = default_fp();
        fp.record_violation("p", ViolationType::RateLimitExceeded, None, 500);
        fp.record_violation("p", ViolationType::RateLimitExceeded, None, 1_500);

        // Only the second violation is >= 1_000.
        let summary = fp.violation_summary(1_000);
        assert_eq!(summary.get("RateLimitExceeded").copied().unwrap_or(0), 1);
    }

    #[test]
    fn test_violation_summary_empty_when_none() {
        let fp = default_fp();
        assert!(fp.violation_summary(0).is_empty());
    }

    #[test]
    fn test_violation_buffer_capped_at_1000() {
        let mut fp = default_fp();
        for i in 0_u64..1_100 {
            fp.record_violation("p", ViolationType::RateLimitExceeded, None, i);
        }
        assert_eq!(fp.violations.len(), 1_000);
    }

    // ── Stats ─────────────────────────────────────────────────────────────

    #[test]
    fn test_stats_initial_state() {
        let fp = FloodProtection::new(FloodConfig {
            global_rps: 100,
            burst_multiplier: 2.0,
            ..FloodConfig::default()
        });
        let s = fp.stats(0);
        assert_eq!(s.total_peers, 0);
        assert_eq!(s.banned_peers, 0);
        assert_eq!(s.dedup_cache_size, 0);
        assert_eq!(s.total_allowed, 0);
        assert_eq!(s.total_blocked, 0);
        assert!((s.global_tokens - 200.0).abs() < 1e-6);
    }

    #[test]
    fn test_stats_reflects_activity() {
        let mut fp = FloodProtection::new(FloodConfig {
            global_rps: 1_000,
            per_peer_rps: 1_000,
            burst_multiplier: 2.0,
            ..FloodConfig::default()
        });
        let id = fnv1a_message_id(b"stat-test");
        fp.check("p", id, 0);
        fp.record_allowed("p", id, 0);

        let s = fp.stats(0);
        assert_eq!(s.total_peers, 1);
        assert_eq!(s.total_allowed, 1);
        assert_eq!(s.dedup_cache_size, 1);
    }

    #[test]
    fn test_stats_banned_peers_counted() {
        let config = FloodConfig {
            ban_threshold: 1,
            ban_duration_ms: 100_000,
            ..FloodConfig::default()
        };
        let mut fp = FloodProtection::new(config);
        fp.record_violation("bad", ViolationType::RateLimitExceeded, None, 0);
        let s = fp.stats(500);
        assert_eq!(s.banned_peers, 1);
    }

    // ── Interaction / integration checks ─────────────────────────────────

    #[test]
    fn test_block_counts_increment_total_blocked() {
        let config = FloodConfig {
            global_rps: 1,
            burst_multiplier: 1.0,
            per_peer_rps: 10_000,
            dedup_capacity: 10_000,
            dedup_window_ms: 30_000,
            ..FloodConfig::default()
        };
        let mut fp = FloodProtection::new(config);

        // Use up the one global token.
        let id1 = fnv1a_message_id(b"first");
        assert_eq!(fp.check("p", id1, 0), CheckResult::Allow);
        fp.record_allowed("p", id1, 0);

        let id2 = fnv1a_message_id(b"second");
        fp.check("p", id2, 0); // GlobalRateLimited → blocked
        assert_eq!(fp.total_blocked, 1);
    }

    #[test]
    fn test_check_ordering_ban_before_global() {
        // If a peer is banned the ban response must come before the global-limit
        // check (which might also be triggered).
        let config = FloodConfig {
            global_rps: 1,
            burst_multiplier: 1.0,
            per_peer_rps: 100_000,
            ban_threshold: 1,
            ban_duration_ms: 50_000,
            dedup_capacity: 10_000,
            dedup_window_ms: 30_000,
        };
        let mut fp = FloodProtection::new(config);

        // Exhaust global bucket.
        let id1 = fnv1a_message_id(b"c1");
        fp.check("p", id1, 0);
        fp.record_allowed("p", id1, 0);

        // Ban the peer.
        fp.record_violation("p", ViolationType::RateLimitExceeded, None, 0);

        // Both global-limit and ban conditions hold; ban check fires first.
        let id2 = fnv1a_message_id(b"c2");
        match fp.check("p", id2, 0) {
            CheckResult::Banned { .. } => {}
            other => panic!("expected Banned, got {other:?}"),
        }
    }

    #[test]
    fn test_burst_multiplier_affects_initial_tokens() {
        let config = FloodConfig {
            global_rps: 10,
            burst_multiplier: 3.0,
            per_peer_rps: 10,
            dedup_capacity: 10_000,
            dedup_window_ms: 30_000,
            ..FloodConfig::default()
        };
        let fp = FloodProtection::new(config);
        // Initial global tokens = rps * burst = 10 * 3 = 30.
        assert!((fp.global_tokens - 30.0).abs() < 1e-6);
    }

    #[test]
    fn test_violation_with_message_id_stored() {
        let mut fp = default_fp();
        let msg = fnv1a_message_id(b"viol-msg");
        fp.record_violation("p", ViolationType::DuplicateMessage, Some(msg), 100);
        let v = fp.violations.back().expect("violation recorded");
        assert_eq!(v.message_id, Some(msg));
    }
}
