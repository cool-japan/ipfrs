//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::sync::Mutex;

use super::constants::TOKEN_SCALE;
use super::types::{AtomicRateLimiterStats, AtomicTokenBucket, RateLimitDecision};

/// Production rate limiter with per-peer buckets and a global bucket.
///
/// # Design
///
/// Every call to [`RateLimiter::check`] must pass **two** independent token
/// buckets:
/// 1. A **global** bucket shared across all peers.
/// 2. A **per-peer** bucket, created on demand with `default_capacity` /
///    `default_rate` if the peer has not been explicitly registered.
///
/// If either bucket cannot satisfy the request the decision is `Throttled`
/// (with a rough retry estimate) or `Rejected` (global bucket empty and peer
/// bucket also empty).
pub struct RateLimiter {
    /// Per-peer token buckets.
    pub(super) per_peer: Mutex<HashMap<String, AtomicTokenBucket>>,
    /// Global rate limit bucket.
    pub(super) global: AtomicTokenBucket,
    /// Default capacity for auto-created peer buckets.
    pub(super) default_capacity: u64,
    /// Default refill rate for auto-created peer buckets.
    pub(super) default_rate: u64,
    /// Counters.
    pub(super) stats: AtomicRateLimiterStats,
}
impl RateLimiter {
    /// Create a new `RateLimiter` with global capacity and rate.
    pub fn new(global_capacity: u64, global_rate: u64) -> Self {
        Self::with_defaults(global_capacity, global_rate, 1_000, 100)
    }
    /// Create a `RateLimiter` with explicit per-peer defaults.
    pub fn with_defaults(
        global_capacity: u64,
        global_rate: u64,
        default_capacity: u64,
        default_rate: u64,
    ) -> Self {
        Self {
            per_peer: Mutex::new(HashMap::new()),
            global: AtomicTokenBucket::new(global_capacity, global_rate),
            default_capacity,
            default_rate,
            stats: AtomicRateLimiterStats::default(),
        }
    }
    /// Check whether `peer_id` may consume `tokens`.
    ///
    /// Returns:
    /// - `Allowed` if both global and per-peer buckets have sufficient tokens.
    /// - `Throttled { retry_after_ms }` if the per-peer bucket is insufficient
    ///   (global was sufficient).
    /// - `Rejected` if the global bucket is insufficient.
    pub fn check(&self, peer_id: &str, tokens: u64) -> RateLimitDecision {
        if !self.global.try_acquire(tokens) {
            self.stats.total_throttled.fetch_add(1, Ordering::Relaxed);
            let available = self.global.available_tokens();
            let needed = tokens.saturating_sub(available);
            let retry_ms = if self.global.refill_rate > 0 {
                (needed as f64 / self.global.refill_rate as f64 * 1_000.0).ceil() as u64
            } else {
                u64::MAX
            };
            return RateLimitDecision::Throttled {
                retry_after_ms: retry_ms,
            };
        }
        let decision = {
            let mut peers = match self.per_peer.lock() {
                Ok(g) => g,
                Err(poisoned) => poisoned.into_inner(),
            };
            let bucket = peers.entry(peer_id.to_string()).or_insert_with(|| {
                AtomicTokenBucket::new(self.default_capacity, self.default_rate)
            });
            if bucket.try_acquire(tokens) {
                RateLimitDecision::Allowed
            } else {
                let available = bucket.available_tokens();
                let needed = tokens.saturating_sub(available);
                let retry_ms = if bucket.refill_rate > 0 {
                    (needed as f64 / bucket.refill_rate as f64 * 1_000.0).ceil() as u64
                } else {
                    u64::MAX
                };
                RateLimitDecision::Throttled {
                    retry_after_ms: retry_ms,
                }
            }
        };
        match &decision {
            RateLimitDecision::Allowed => {
                self.stats.total_allowed.fetch_add(1, Ordering::Relaxed);
            }
            RateLimitDecision::Throttled { .. } => {
                let cap_scaled = self.global.capacity.saturating_mul(TOKEN_SCALE);
                let refund = tokens.saturating_mul(TOKEN_SCALE);
                self.global
                    .tokens
                    .fetch_update(Ordering::Release, Ordering::Relaxed, |cur| {
                        Some(cur.saturating_add(refund).min(cap_scaled))
                    })
                    .ok();
                self.stats.total_throttled.fetch_add(1, Ordering::Relaxed);
            }
            RateLimitDecision::Rejected => {
                self.stats.total_rejected.fetch_add(1, Ordering::Relaxed);
            }
        }
        decision
    }
    /// Explicitly register a peer with custom capacity and rate.
    /// Overwrites any existing bucket for that peer.
    pub fn register_peer(&self, peer_id: &str, capacity: u64, rate: u64) {
        let mut peers = match self.per_peer.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        peers.insert(peer_id.to_string(), AtomicTokenBucket::new(capacity, rate));
    }
    /// Remove the per-peer bucket for `peer_id`.
    pub fn remove_peer(&self, peer_id: &str) {
        let mut peers = match self.per_peer.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        peers.remove(peer_id);
    }
    /// Return the number of registered per-peer buckets.
    pub fn peer_count(&self) -> usize {
        let peers = match self.per_peer.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        peers.len()
    }
    /// Return a snapshot of the stats counters.
    pub fn stats_snapshot(&self) -> RateLimiterStatsSnapshot {
        self.stats.snapshot()
    }
}
/// Aggregated statistics for the `PeerRateLimiter`.
#[derive(Debug, Clone)]
pub struct PeerRateLimiterStats {
    /// Total number of tracked peers.
    pub total_peers: usize,
    /// Number of currently blocked peers.
    pub blocked_peers: usize,
    /// Total requests allowed across all peers.
    pub total_allowed: u64,
    /// Total requests throttled across all peers.
    pub total_throttled: u64,
}
/// A snapshot of [`AtomicRateLimiterStats`] for reporting.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RateLimiterStatsSnapshot {
    /// Total allowed requests.
    pub total_allowed: u64,
    /// Total throttled requests.
    pub total_throttled: u64,
    /// Total rejected requests.
    pub total_rejected: u64,
}
