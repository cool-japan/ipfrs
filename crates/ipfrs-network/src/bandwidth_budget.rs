//! Per-peer bandwidth budget allocation and enforcement with token-bucket per peer.
//!
//! Each peer gets an independent token bucket for upload and download.  Tokens
//! refill over time at the configured rate (bytes/sec) and may briefly exceed
//! the nominal quota up to `quota * burst_factor` to allow short bursts.
//!
//! A global upload and download cap prevents any single peer's activity from
//! starving the whole node.
//!
//! # Example
//!
//! ```rust
//! use ipfrs_network::bandwidth_budget::{
//!     BandwidthBudgetManager, BandwidthQuota, BudgetConfig,
//! };
//!
//! let quota = BandwidthQuota {
//!     upload_bytes_per_sec: 1_000_000,
//!     download_bytes_per_sec: 2_000_000,
//!     burst_factor: 1.5,
//! };
//! let config = BudgetConfig {
//!     default_quota: quota,
//!     global_upload_cap: 100_000_000,
//!     global_download_cap: 200_000_000,
//!     min_tokens: 0.0,
//!     gc_threshold: 1000,
//! };
//! let mut mgr = BandwidthBudgetManager::new(config);
//!
//! // Register a peer with timestamp 0 (ms since epoch or any monotonic counter).
//! mgr.register_peer("peer-1", 0);
//!
//! // Try to use 512 KiB of upload bandwidth at t=1000 ms.
//! let allowed = mgr.try_consume_upload("peer-1", 512 * 1024, 1000);
//! println!("upload allowed: {allowed}");
//! ```

use std::collections::HashMap;

// ── BandwidthQuota ────────────────────────────────────────────────────────────

/// Steady-state rate limits for a single peer.
///
/// `burst_factor` must be ≥ 1.0; values below 1.0 are clamped to 1.0 at
/// token-bucket initialisation time.
#[derive(Debug, Clone)]
pub struct BandwidthQuota {
    /// Maximum outbound bytes per second for this peer.
    pub upload_bytes_per_sec: u64,
    /// Maximum inbound bytes per second for this peer.
    pub download_bytes_per_sec: u64,
    /// Burst multiplier: tokens can accumulate up to `quota * burst_factor`.
    pub burst_factor: f64,
}

impl BandwidthQuota {
    /// Returns the upload burst cap in bytes.
    #[inline]
    pub fn upload_burst_cap(&self) -> f64 {
        self.upload_bytes_per_sec as f64 * self.burst_factor.max(1.0)
    }

    /// Returns the download burst cap in bytes.
    #[inline]
    pub fn download_burst_cap(&self) -> f64 {
        self.download_bytes_per_sec as f64 * self.burst_factor.max(1.0)
    }
}

// ── PeerBucket ────────────────────────────────────────────────────────────────

/// Token-bucket state for a single peer.
///
/// Timestamps (`last_refill`) are in **milliseconds** and can be any
/// monotonically-increasing integer (e.g. milliseconds since UNIX epoch, or
/// from a test counter).
#[derive(Debug, Clone)]
pub struct PeerBucket {
    /// Unique peer identifier.
    pub peer_id: String,
    /// Available upload tokens (bytes).  May not exceed the burst cap.
    pub upload_tokens: f64,
    /// Available download tokens (bytes).  May not exceed the burst cap.
    pub download_tokens: f64,
    /// Timestamp (ms) of the last token-refill.
    pub last_refill: u64,
    /// Per-peer quota configuration.
    pub quota: BandwidthQuota,
    /// Cumulative upload bytes consumed (granted) since registration.
    pub bytes_used_up: u64,
    /// Cumulative download bytes consumed (granted) since registration.
    pub bytes_used_down: u64,
}

impl PeerBucket {
    /// Creates a new bucket initialised with one second's worth of tokens (capped
    /// at the burst cap).
    fn new(peer_id: String, quota: BandwidthQuota, now: u64) -> Self {
        let upload_tokens = quota.upload_burst_cap();
        let download_tokens = quota.download_burst_cap();
        Self {
            peer_id,
            upload_tokens,
            download_tokens,
            last_refill: now,
            quota,
            bytes_used_up: 0,
            bytes_used_down: 0,
        }
    }

    /// Refills tokens based on elapsed time since `last_refill`.
    ///
    /// Tokens are added proportionally to the elapsed milliseconds and capped at
    /// the per-quota burst ceiling.
    fn refill(&mut self, now: u64, min_tokens: f64) {
        if now <= self.last_refill {
            return;
        }
        let elapsed_ms = (now - self.last_refill) as f64;
        let elapsed_secs = elapsed_ms / 1_000.0;

        let up_cap = self.quota.upload_burst_cap();
        let down_cap = self.quota.download_burst_cap();

        self.upload_tokens = (self.upload_tokens
            + elapsed_secs * self.quota.upload_bytes_per_sec as f64)
            .min(up_cap)
            .max(min_tokens);

        self.download_tokens = (self.download_tokens
            + elapsed_secs * self.quota.download_bytes_per_sec as f64)
            .min(down_cap)
            .max(min_tokens);

        self.last_refill = now;
    }
}

// ── BudgetConfig ──────────────────────────────────────────────────────────────

/// Global configuration for [`BandwidthBudgetManager`].
#[derive(Debug, Clone)]
pub struct BudgetConfig {
    /// Quota applied to newly registered peers (unless overridden).
    pub default_quota: BandwidthQuota,
    /// Maximum total upload bytes that can be granted across **all** peers.
    /// `try_consume_upload` returns `false` once this is reached.
    pub global_upload_cap: u64,
    /// Maximum total download bytes across all peers.
    pub global_download_cap: u64,
    /// Floor value for token buckets; prevents tokens going below this.
    pub min_tokens: f64,
    /// Remove peers with no active tokens after this many accounting operations.
    /// Currently used as a hint for future GC; the manager tracks an internal
    /// operation counter and prunes fully-drained peers when the counter exceeds
    /// this threshold.
    pub gc_threshold: usize,
}

// ── BudgetStats ───────────────────────────────────────────────────────────────

/// Aggregate statistics collected by [`BandwidthBudgetManager`].
#[derive(Debug, Clone, Default)]
pub struct BudgetStats {
    /// Total upload bytes granted across all peers and all time.
    pub total_granted_up: u64,
    /// Total download bytes granted across all peers and all time.
    pub total_granted_down: u64,
    /// Total upload requests rejected (insufficient tokens or global cap).
    pub total_rejected_up: u64,
    /// Total download requests rejected.
    pub total_rejected_down: u64,
    /// Number of peers currently registered.
    pub active_peers: u64,
}

// ── BandwidthBudgetManager ────────────────────────────────────────────────────

/// Per-peer token-bucket bandwidth manager.
///
/// Call [`refill`] or [`refill_all`] periodically (or lazily before each
/// consume call) to add tokens, then use [`try_consume_upload`] /
/// [`try_consume_download`] to request bandwidth.
///
/// [`refill`]: BandwidthBudgetManager::refill
/// [`refill_all`]: BandwidthBudgetManager::refill_all
/// [`try_consume_upload`]: BandwidthBudgetManager::try_consume_upload
/// [`try_consume_download`]: BandwidthBudgetManager::try_consume_download
pub struct BandwidthBudgetManager {
    config: BudgetConfig,
    peers: HashMap<String, PeerBucket>,
    global_upload_used: u64,
    global_download_used: u64,
    stats: BudgetStats,
    /// Internal operation counter for GC triggering.
    op_count: usize,
}

impl BandwidthBudgetManager {
    // ── Construction ──────────────────────────────────────────────────────────

    /// Creates a new manager with the supplied configuration.
    pub fn new(config: BudgetConfig) -> Self {
        Self {
            config,
            peers: HashMap::new(),
            global_upload_used: 0,
            global_download_used: 0,
            stats: BudgetStats::default(),
            op_count: 0,
        }
    }

    // ── Peer registration ─────────────────────────────────────────────────────

    /// Registers a peer using the default quota.
    ///
    /// If the peer is already registered this is a no-op.
    pub fn register_peer(&mut self, peer_id: &str, now: u64) {
        if self.peers.contains_key(peer_id) {
            return;
        }
        let quota = self.config.default_quota.clone();
        self.register_peer_with_quota(peer_id, quota, now);
    }

    /// Registers a peer with a custom quota, overriding the default.
    ///
    /// If the peer already exists its quota is **replaced** and tokens are
    /// re-initialised.
    pub fn register_peer_with_quota(&mut self, peer_id: &str, quota: BandwidthQuota, now: u64) {
        let bucket = PeerBucket::new(peer_id.to_owned(), quota, now);
        self.peers.insert(peer_id.to_owned(), bucket);
        self.stats.active_peers = self.peers.len() as u64;
    }

    // ── Token refill ──────────────────────────────────────────────────────────

    /// Refills the token bucket for a single peer based on elapsed time.
    ///
    /// Does nothing if the peer is not registered or `now` is not later than
    /// the peer's `last_refill` timestamp.
    pub fn refill(&mut self, peer_id: &str, now: u64) {
        let min_tokens = self.config.min_tokens;
        if let Some(bucket) = self.peers.get_mut(peer_id) {
            bucket.refill(now, min_tokens);
        }
    }

    /// Refills token buckets for **all** registered peers.
    pub fn refill_all(&mut self, now: u64) {
        let min_tokens = self.config.min_tokens;
        for bucket in self.peers.values_mut() {
            bucket.refill(now, min_tokens);
        }
    }

    // ── Bandwidth consumption ─────────────────────────────────────────────────

    /// Attempts to consume `bytes` of upload bandwidth for `peer_id`.
    ///
    /// Returns `true` and deducts tokens when:
    /// 1. The peer is registered,
    /// 2. The peer has enough upload tokens,
    /// 3. The global upload cap has not been reached.
    ///
    /// Returns `false` (and does **not** deduct tokens) in all other cases.
    pub fn try_consume_upload(&mut self, peer_id: &str, bytes: u64, now: u64) -> bool {
        // Lazy refill before consumption.
        self.refill(peer_id, now);
        self.bump_op_count();

        let global_cap = self.config.global_upload_cap;
        let global_used = self.global_upload_used;

        let bucket = match self.peers.get_mut(peer_id) {
            Some(b) => b,
            None => {
                self.stats.total_rejected_up = self.stats.total_rejected_up.saturating_add(1);
                return false;
            }
        };

        let bytes_f = bytes as f64;
        if bucket.upload_tokens < bytes_f {
            self.stats.total_rejected_up = self.stats.total_rejected_up.saturating_add(1);
            return false;
        }

        if global_used.saturating_add(bytes) > global_cap {
            self.stats.total_rejected_up = self.stats.total_rejected_up.saturating_add(1);
            return false;
        }

        bucket.upload_tokens -= bytes_f;
        bucket.bytes_used_up = bucket.bytes_used_up.saturating_add(bytes);
        self.global_upload_used = self.global_upload_used.saturating_add(bytes);
        self.stats.total_granted_up = self.stats.total_granted_up.saturating_add(bytes);
        true
    }

    /// Attempts to consume `bytes` of download bandwidth for `peer_id`.
    ///
    /// Returns `true` and deducts tokens when:
    /// 1. The peer is registered,
    /// 2. The peer has enough download tokens,
    /// 3. The global download cap has not been reached.
    ///
    /// Returns `false` (and does **not** deduct tokens) in all other cases.
    pub fn try_consume_download(&mut self, peer_id: &str, bytes: u64, now: u64) -> bool {
        self.refill(peer_id, now);
        self.bump_op_count();

        let global_cap = self.config.global_download_cap;
        let global_used = self.global_download_used;

        let bucket = match self.peers.get_mut(peer_id) {
            Some(b) => b,
            None => {
                self.stats.total_rejected_down = self.stats.total_rejected_down.saturating_add(1);
                return false;
            }
        };

        let bytes_f = bytes as f64;
        if bucket.download_tokens < bytes_f {
            self.stats.total_rejected_down = self.stats.total_rejected_down.saturating_add(1);
            return false;
        }

        if global_used.saturating_add(bytes) > global_cap {
            self.stats.total_rejected_down = self.stats.total_rejected_down.saturating_add(1);
            return false;
        }

        bucket.download_tokens -= bytes_f;
        bucket.bytes_used_down = bucket.bytes_used_down.saturating_add(bytes);
        self.global_download_used = self.global_download_used.saturating_add(bytes);
        self.stats.total_granted_down = self.stats.total_granted_down.saturating_add(bytes);
        true
    }

    // ── Queries ───────────────────────────────────────────────────────────────

    /// Returns the number of remaining upload tokens for `peer_id`, floored to 0.
    ///
    /// Returns `0` for unknown peers.
    pub fn remaining_upload(&self, peer_id: &str) -> u64 {
        self.peers
            .get(peer_id)
            .map(|b| b.upload_tokens.max(0.0) as u64)
            .unwrap_or(0)
    }

    /// Returns the number of remaining download tokens for `peer_id`, floored to 0.
    ///
    /// Returns `0` for unknown peers.
    pub fn remaining_download(&self, peer_id: &str) -> u64 {
        self.peers
            .get(peer_id)
            .map(|b| b.download_tokens.max(0.0) as u64)
            .unwrap_or(0)
    }

    // ── Peer management ───────────────────────────────────────────────────────

    /// Removes a peer from the manager.
    ///
    /// Returns `true` if the peer existed, `false` otherwise.
    pub fn remove_peer(&mut self, peer_id: &str) -> bool {
        let removed = self.peers.remove(peer_id).is_some();
        if removed {
            self.stats.active_peers = self.peers.len() as u64;
        }
        removed
    }

    /// Returns the number of currently registered peers.
    pub fn peer_count(&self) -> usize {
        self.peers.len()
    }

    /// Returns a reference to the aggregate statistics.
    pub fn stats(&self) -> &BudgetStats {
        &self.stats
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    /// Increments the operation counter and runs GC when the threshold is met.
    fn bump_op_count(&mut self) {
        self.op_count = self.op_count.wrapping_add(1);
        if self.op_count >= self.config.gc_threshold {
            self.op_count = 0;
            self.gc_idle_peers();
        }
    }

    /// Removes peers whose token buckets are completely empty (both upload and
    /// download tokens at or below zero) and who have no accumulated usage.
    ///
    /// This is a heuristic GC pass meant to reclaim memory for ephemeral peers
    /// that were registered but never granted any bandwidth.
    fn gc_idle_peers(&mut self) {
        self.peers.retain(|_, bucket| {
            bucket.bytes_used_up > 0
                || bucket.bytes_used_down > 0
                || bucket.upload_tokens > 0.0
                || bucket.download_tokens > 0.0
        });
        self.stats.active_peers = self.peers.len() as u64;
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn default_quota() -> BandwidthQuota {
        BandwidthQuota {
            upload_bytes_per_sec: 1_000,   // 1 KB/s
            download_bytes_per_sec: 2_000, // 2 KB/s
            burst_factor: 2.0,
        }
    }

    fn make_manager() -> BandwidthBudgetManager {
        let config = BudgetConfig {
            default_quota: default_quota(),
            global_upload_cap: 1_000_000,
            global_download_cap: 2_000_000,
            min_tokens: 0.0,
            gc_threshold: 10_000,
        };
        BandwidthBudgetManager::new(config)
    }

    // T01 ── register_peer adds the peer ──────────────────────────────────────

    #[test]
    fn t01_register_peer_adds_peer() {
        let mut mgr = make_manager();
        assert_eq!(mgr.peer_count(), 0);
        mgr.register_peer("p1", 0);
        assert_eq!(mgr.peer_count(), 1);
    }

    // T02 ── register_peer is idempotent ──────────────────────────────────────

    #[test]
    fn t02_register_peer_idempotent() {
        let mut mgr = make_manager();
        mgr.register_peer("p1", 0);
        mgr.register_peer("p1", 100); // duplicate — should not add again
        assert_eq!(mgr.peer_count(), 1);
    }

    // T03 ── register_peer_with_quota sets custom quota ───────────────────────

    #[test]
    fn t03_register_peer_with_quota() {
        let mut mgr = make_manager();
        let quota = BandwidthQuota {
            upload_bytes_per_sec: 9_999,
            download_bytes_per_sec: 8_888,
            burst_factor: 1.0,
        };
        mgr.register_peer_with_quota("p1", quota.clone(), 0);
        // Tokens should be initialised at burst cap = quota (since burst_factor=1.0)
        assert_eq!(mgr.remaining_upload("p1"), 9_999);
        assert_eq!(mgr.remaining_download("p1"), 8_888);
    }

    // T04 ── initial tokens equal burst cap ───────────────────────────────────

    #[test]
    fn t04_initial_tokens_equal_burst_cap() {
        let mut mgr = make_manager();
        mgr.register_peer("p1", 0);
        // quota: upload=1000, burst_factor=2.0 → cap=2000
        assert_eq!(mgr.remaining_upload("p1"), 2_000);
        assert_eq!(mgr.remaining_download("p1"), 4_000);
    }

    // T05 ── try_consume_upload succeeds when tokens are available ─────────────

    #[test]
    fn t05_consume_upload_succeeds() {
        let mut mgr = make_manager();
        mgr.register_peer("p1", 0);
        let ok = mgr.try_consume_upload("p1", 500, 0);
        assert!(ok);
        assert_eq!(mgr.remaining_upload("p1"), 1_500);
    }

    // T06 ── try_consume_download succeeds when tokens are available ───────────

    #[test]
    fn t06_consume_download_succeeds() {
        let mut mgr = make_manager();
        mgr.register_peer("p1", 0);
        let ok = mgr.try_consume_download("p1", 1_000, 0);
        assert!(ok);
        assert_eq!(mgr.remaining_download("p1"), 3_000);
    }

    // T07 ── try_consume_upload fails when tokens are exhausted ───────────────

    #[test]
    fn t07_consume_upload_rejected_no_tokens() {
        let mut mgr = make_manager();
        mgr.register_peer("p1", 0);
        // Drain all upload tokens (burst cap = 2000)
        assert!(mgr.try_consume_upload("p1", 2_000, 0));
        // Now should fail
        let ok = mgr.try_consume_upload("p1", 1, 0);
        assert!(!ok);
    }

    // T08 ── try_consume_download fails when tokens are exhausted ─────────────

    #[test]
    fn t08_consume_download_rejected_no_tokens() {
        let mut mgr = make_manager();
        mgr.register_peer("p1", 0);
        assert!(mgr.try_consume_download("p1", 4_000, 0));
        let ok = mgr.try_consume_download("p1", 1, 0);
        assert!(!ok);
    }

    // T09 ── refill adds tokens over elapsed time ──────────────────────────────

    #[test]
    fn t09_refill_adds_tokens() {
        let mut mgr = make_manager();
        mgr.register_peer("p1", 0);
        // Drain all tokens
        assert!(mgr.try_consume_upload("p1", 2_000, 0));
        assert_eq!(mgr.remaining_upload("p1"), 0);
        // After 1000 ms: 1 KB/s × 1 s = 1000 tokens added (capped at burst=2000)
        mgr.refill("p1", 1_000);
        assert_eq!(mgr.remaining_upload("p1"), 1_000);
    }

    // T10 ── refill caps tokens at burst ceiling ───────────────────────────────

    #[test]
    fn t10_refill_caps_at_burst() {
        let mut mgr = make_manager();
        mgr.register_peer("p1", 0);
        // Partially drain: 1000 tokens remaining (out of 2000 burst cap)
        assert!(mgr.try_consume_upload("p1", 1_000, 0));
        // Refill 10 s → would add 10,000 tokens, but burst cap is 2000
        mgr.refill("p1", 10_000);
        assert_eq!(mgr.remaining_upload("p1"), 2_000);
    }

    // T11 ── burst_factor allows consuming beyond steady rate ─────────────────

    #[test]
    fn t11_burst_factor_allows_burst() {
        let quota = BandwidthQuota {
            upload_bytes_per_sec: 1_000,
            download_bytes_per_sec: 1_000,
            burst_factor: 3.0,
        };
        let config = BudgetConfig {
            default_quota: quota,
            global_upload_cap: u64::MAX,
            global_download_cap: u64::MAX,
            min_tokens: 0.0,
            gc_threshold: 10_000,
        };
        let mut mgr = BandwidthBudgetManager::new(config);
        mgr.register_peer("p1", 0);
        // Burst cap = 1000 × 3 = 3000; consuming 2500 should succeed.
        assert!(mgr.try_consume_upload("p1", 2_500, 0));
    }

    // T12 ── global_upload_cap blocks consumption ─────────────────────────────

    #[test]
    fn t12_global_upload_cap_enforced() {
        let config = BudgetConfig {
            default_quota: BandwidthQuota {
                upload_bytes_per_sec: 1_000_000,
                download_bytes_per_sec: 1_000_000,
                burst_factor: 10.0,
            },
            global_upload_cap: 100,
            global_download_cap: u64::MAX,
            min_tokens: 0.0,
            gc_threshold: 10_000,
        };
        let mut mgr = BandwidthBudgetManager::new(config);
        mgr.register_peer("p1", 0);
        // First consume uses 100 bytes (exactly at the cap).
        assert!(mgr.try_consume_upload("p1", 100, 0));
        // Second consume must fail because global cap is exhausted.
        assert!(!mgr.try_consume_upload("p1", 1, 0));
    }

    // T13 ── global_download_cap blocks consumption ────────────────────────────

    #[test]
    fn t13_global_download_cap_enforced() {
        let config = BudgetConfig {
            default_quota: BandwidthQuota {
                upload_bytes_per_sec: 1_000_000,
                download_bytes_per_sec: 1_000_000,
                burst_factor: 10.0,
            },
            global_upload_cap: u64::MAX,
            global_download_cap: 200,
            min_tokens: 0.0,
            gc_threshold: 10_000,
        };
        let mut mgr = BandwidthBudgetManager::new(config);
        mgr.register_peer("p1", 0);
        assert!(mgr.try_consume_download("p1", 200, 0));
        assert!(!mgr.try_consume_download("p1", 1, 0));
    }

    // T14 ── per-peer quota override is respected ──────────────────────────────

    #[test]
    fn t14_per_peer_quota_override() {
        let mut mgr = make_manager();
        let custom = BandwidthQuota {
            upload_bytes_per_sec: 50,
            download_bytes_per_sec: 50,
            burst_factor: 1.0,
        };
        mgr.register_peer_with_quota("slow", custom, 0);
        // Default quota allows 2000 upload, but "slow" can only do 50.
        assert!(mgr.try_consume_upload("slow", 50, 0));
        assert!(!mgr.try_consume_upload("slow", 1, 0));
    }

    // T15 ── remove_peer returns true for existing peer ────────────────────────

    #[test]
    fn t15_remove_peer_returns_true() {
        let mut mgr = make_manager();
        mgr.register_peer("p1", 0);
        assert!(mgr.remove_peer("p1"));
    }

    // T16 ── remove_peer returns false for unknown peer ────────────────────────

    #[test]
    fn t16_remove_peer_returns_false_unknown() {
        let mut mgr = make_manager();
        assert!(!mgr.remove_peer("ghost"));
    }

    // T17 ── peer_count decrements after remove ───────────────────────────────

    #[test]
    fn t17_peer_count_after_remove() {
        let mut mgr = make_manager();
        mgr.register_peer("p1", 0);
        mgr.register_peer("p2", 0);
        assert_eq!(mgr.peer_count(), 2);
        mgr.remove_peer("p1");
        assert_eq!(mgr.peer_count(), 1);
    }

    // T18 ── refill_all refreshes all peers ───────────────────────────────────

    #[test]
    fn t18_refill_all_refreshes_all_peers() {
        let mut mgr = make_manager();
        mgr.register_peer("p1", 0);
        mgr.register_peer("p2", 0);
        // Drain both
        assert!(mgr.try_consume_upload("p1", 2_000, 0));
        assert!(mgr.try_consume_upload("p2", 2_000, 0));
        assert_eq!(mgr.remaining_upload("p1"), 0);
        assert_eq!(mgr.remaining_upload("p2"), 0);
        // Refill all at t=2000 ms → 2 s × 1000 B/s = 2000 tokens (= burst cap)
        mgr.refill_all(2_000);
        assert_eq!(mgr.remaining_upload("p1"), 2_000);
        assert_eq!(mgr.remaining_upload("p2"), 2_000);
    }

    // T19 ── stats track granted bytes ────────────────────────────────────────

    #[test]
    fn t19_stats_granted_bytes() {
        let mut mgr = make_manager();
        mgr.register_peer("p1", 0);
        assert!(mgr.try_consume_upload("p1", 100, 0));
        assert!(mgr.try_consume_download("p1", 200, 0));
        let s = mgr.stats();
        assert_eq!(s.total_granted_up, 100);
        assert_eq!(s.total_granted_down, 200);
    }

    // T20 ── stats track rejected counts ──────────────────────────────────────

    #[test]
    fn t20_stats_rejected_counts() {
        let mut mgr = make_manager();
        mgr.register_peer("p1", 0);
        // Exhaust tokens then attempt another consume
        assert!(mgr.try_consume_upload("p1", 2_000, 0));
        assert!(!mgr.try_consume_upload("p1", 1, 0));
        assert!(mgr.try_consume_download("p1", 4_000, 0));
        assert!(!mgr.try_consume_download("p1", 1, 0));
        let s = mgr.stats();
        assert_eq!(s.total_rejected_up, 1);
        assert_eq!(s.total_rejected_down, 1);
    }

    // T21 ── stats active_peers reflects real count ────────────────────────────

    #[test]
    fn t21_stats_active_peers() {
        let mut mgr = make_manager();
        mgr.register_peer("a", 0);
        mgr.register_peer("b", 0);
        assert_eq!(mgr.stats().active_peers, 2);
        mgr.remove_peer("a");
        assert_eq!(mgr.stats().active_peers, 1);
    }

    // T22 ── zero-byte consume always succeeds ─────────────────────────────────

    #[test]
    fn t22_zero_byte_consume_always_succeeds() {
        let mut mgr = make_manager();
        mgr.register_peer("p1", 0);
        // Drain all tokens first
        assert!(mgr.try_consume_upload("p1", 2_000, 0));
        // Zero-byte consume must succeed even with empty bucket
        assert!(mgr.try_consume_upload("p1", 0, 0));
        assert!(mgr.try_consume_download("p1", 0, 0));
    }

    // T23 ── consume for unregistered peer returns false ───────────────────────

    #[test]
    fn t23_consume_unknown_peer_returns_false() {
        let mut mgr = make_manager();
        assert!(!mgr.try_consume_upload("nobody", 1, 0));
        assert!(!mgr.try_consume_download("nobody", 1, 0));
        assert_eq!(mgr.stats().total_rejected_up, 1);
        assert_eq!(mgr.stats().total_rejected_down, 1);
    }

    // T24 ── remaining_upload returns 0 for unknown peer ──────────────────────

    #[test]
    fn t24_remaining_upload_unknown_peer() {
        let mgr = make_manager();
        assert_eq!(mgr.remaining_upload("ghost"), 0);
        assert_eq!(mgr.remaining_download("ghost"), 0);
    }

    // T25 ── rapid fill-drain cycle maintains correctness ─────────────────────
    //
    // Quota: 1000 B/s, burst_factor=2.0 → burst cap = 2000 bytes.
    // Cycle: drain 500 bytes, then advance 1000 ms (refilling 1000 bytes).
    // Net gain per cycle = 1000 − 500 = +500, so tokens grow until they hit the cap.

    #[test]
    fn t25_rapid_fill_drain_cycle() {
        let mut mgr = make_manager();
        mgr.register_peer("p1", 0);

        let mut now: u64 = 0;
        for _ in 0..10 {
            // Consume 500 bytes — always affordable since refill >= 1000 B/s per second.
            assert!(
                mgr.try_consume_upload("p1", 500, now),
                "consume failed at t={now}"
            );
            now += 1_000; // advance 1 s → refill 1000 tokens (lazy refill on next consume)
            mgr.refill("p1", now);
        }
        // After 10 cycles tokens must be ≤ burst cap and ≥ 0.
        let remaining = mgr.remaining_upload("p1");
        assert!(remaining <= 2_000, "tokens exceeded burst cap: {remaining}");
    }

    // T26 ── refill does not rewind time (now <= last_refill is no-op) ─────────

    #[test]
    fn t26_refill_no_rewind() {
        let mut mgr = make_manager();
        mgr.register_peer("p1", 1_000);
        // Drain fully at t=1000
        assert!(mgr.try_consume_upload("p1", 2_000, 1_000));
        assert_eq!(mgr.remaining_upload("p1"), 0);
        // Refill with an older timestamp — should be a no-op
        mgr.refill("p1", 500);
        assert_eq!(mgr.remaining_upload("p1"), 0);
    }

    // T27 ── multiple peers do not share tokens ────────────────────────────────

    #[test]
    fn t27_peers_independent_buckets() {
        let mut mgr = make_manager();
        mgr.register_peer("a", 0);
        mgr.register_peer("b", 0);
        // Drain all of "a"
        assert!(mgr.try_consume_upload("a", 2_000, 0));
        // "b" should still have full tokens
        assert_eq!(mgr.remaining_upload("b"), 2_000);
        // "b" can still consume
        assert!(mgr.try_consume_upload("b", 2_000, 0));
    }

    // T28 ── global cap shared across peers ───────────────────────────────────

    #[test]
    fn t28_global_cap_shared_across_peers() {
        let config = BudgetConfig {
            default_quota: BandwidthQuota {
                upload_bytes_per_sec: 1_000_000,
                download_bytes_per_sec: 1_000_000,
                burst_factor: 100.0,
            },
            global_upload_cap: 300,
            global_download_cap: u64::MAX,
            min_tokens: 0.0,
            gc_threshold: 10_000,
        };
        let mut mgr = BandwidthBudgetManager::new(config);
        mgr.register_peer("a", 0);
        mgr.register_peer("b", 0);
        // "a" consumes 200, "b" consumes 100 → total = 300 = cap
        assert!(mgr.try_consume_upload("a", 200, 0));
        assert!(mgr.try_consume_upload("b", 100, 0));
        // Next consume (either peer) must fail
        assert!(!mgr.try_consume_upload("a", 1, 0));
        assert!(!mgr.try_consume_upload("b", 1, 0));
    }
}
