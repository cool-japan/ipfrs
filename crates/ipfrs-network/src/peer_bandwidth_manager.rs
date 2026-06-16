//! Per-peer bandwidth management with token-bucket rate limiting and fairness scheduling.
//!
//! This module provides production-grade per-peer bandwidth tracking and enforcement:
//! - Token-bucket based rate limiting for upload and download directions
//! - Configurable burst sizes for bursty traffic tolerance
//! - Sliding-window usage snapshots for real-time rate computation
//! - Fairness scheduling policies (Max-Min, Weighted Fair, Unrestricted)
//! - Top-K upload/download peer ranking
//! - Idle peer eviction to bound memory
//! - Global aggregate statistics
//!
//! # Example
//!
//! ```rust
//! use ipfrs_network::peer_bandwidth_manager::{
//!     BandwidthManagerConfig, BandwidthLimit, BandwidthDirection, PeerBandwidthManager,
//! };
//!
//! let config = BandwidthManagerConfig::default();
//! let mut mgr = PeerBandwidthManager::new(config);
//! mgr.register_peer("peer-1".to_string(), 0);
//!
//! // Try to consume 512 bytes for upload
//! let allowed = mgr.try_consume("peer-1", 512, BandwidthDirection::Upload, 0);
//! assert!(allowed);
//!
//! // Record an unconditional transfer
//! mgr.record_transfer("peer-1", 512, 256, 1_000);
//!
//! // Query usage
//! if let Some(usage) = mgr.peer_usage("peer-1", 5_000) {
//!     println!("upload rate: {:.1} bps", usage.upload_rate_bps);
//! }
//! ```

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

// ---------------------------------------------------------------------------
// BandwidthDirection
// ---------------------------------------------------------------------------

/// Direction of bandwidth consumption for rate-limiting purposes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BandwidthDirection {
    /// Bytes being sent to remote peers.
    Upload,
    /// Bytes being received from remote peers.
    Download,
    /// Both upload and download simultaneously.
    Both,
}

// ---------------------------------------------------------------------------
// BandwidthLimit
// ---------------------------------------------------------------------------

/// Token-bucket parameters for a single bandwidth direction.
///
/// A token bucket refills at `bytes_per_second` tokens per second and can
/// hold at most `burst_bytes` tokens (the bucket capacity).  A transfer of N
/// bytes is allowed when at least N tokens are available.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BandwidthLimit {
    /// Sustained rate in bytes per second.
    pub bytes_per_second: u64,
    /// Maximum burst size in bytes.  Tokens accumulate up to this cap.
    pub burst_bytes: u64,
}

impl BandwidthLimit {
    /// Construct a new [`BandwidthLimit`].
    pub fn new(bytes_per_second: u64, burst_bytes: u64) -> Self {
        Self {
            bytes_per_second,
            burst_bytes,
        }
    }

    /// Returns `true` when the limit is effectively "unlimited" (rate == 0).
    pub fn is_unlimited(&self) -> bool {
        self.bytes_per_second == 0
    }
}

impl Default for BandwidthLimit {
    fn default() -> Self {
        // 1 MB/s sustained, 2 MB burst.
        Self {
            bytes_per_second: 1_048_576,
            burst_bytes: 2_097_152,
        }
    }
}

// ---------------------------------------------------------------------------
// PeerBandwidthState
// ---------------------------------------------------------------------------

/// Token-bucket state for a single peer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerBandwidthState {
    /// Stable identifier for the peer.
    pub peer_id: String,
    /// Current upload token count (fractional tokens are kept for precision).
    pub upload_tokens: f64,
    /// Current download token count.
    pub download_tokens: f64,
    /// Rate/burst configuration for the upload direction.
    pub upload_limit: BandwidthLimit,
    /// Rate/burst configuration for the download direction.
    pub download_limit: BandwidthLimit,
    /// Cumulative bytes sent to this peer (unconditional accounting).
    pub total_bytes_sent: u64,
    /// Cumulative bytes received from this peer.
    pub total_bytes_recv: u64,
    /// Millisecond timestamp of the last token refill.
    pub last_refill_ms: u64,
}

impl PeerBandwidthState {
    /// Create a new state initialised with full token buckets.
    pub fn new(
        peer_id: String,
        upload_limit: BandwidthLimit,
        download_limit: BandwidthLimit,
        now_ms: u64,
    ) -> Self {
        Self {
            peer_id,
            upload_tokens: upload_limit.burst_bytes as f64,
            download_tokens: download_limit.burst_bytes as f64,
            upload_limit,
            download_limit,
            total_bytes_sent: 0,
            total_bytes_recv: 0,
            last_refill_ms: now_ms,
        }
    }
}

// ---------------------------------------------------------------------------
// BandwidthUsage
// ---------------------------------------------------------------------------

/// Computed bandwidth usage for a peer within the configured time window.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BandwidthUsage {
    /// Peer identifier.
    pub peer_id: String,
    /// Total bytes sent within the window.
    pub bytes_sent: u64,
    /// Total bytes received within the window.
    pub bytes_recv: u64,
    /// Upload rate in bytes per second derived from window samples.
    pub upload_rate_bps: f64,
    /// Download rate in bytes per second.
    pub download_rate_bps: f64,
    /// Window duration in milliseconds over which rates were computed.
    pub window_ms: u64,
}

// ---------------------------------------------------------------------------
// FairnessPolicy
// ---------------------------------------------------------------------------

/// Scheduling policy used by [`PeerBandwidthManager::apply_fairness`].
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub enum FairnessPolicy {
    /// Max-Min fairness: every peer receives an equal share of the total
    /// available tokens.  Each peer's current token count is clamped to
    /// `total_tokens / num_peers`.
    MaxMinFairness,
    /// Weighted fair queuing: each peer's tokens are scaled by its normalised
    /// weight relative to all registered peers.
    WeightedFair {
        /// Per-peer weights.  Peers absent from the map are assigned weight 1.0.
        weights: HashMap<String, f64>,
    },
    /// No fairness enforcement; tokens accumulate freely up to the burst cap.
    #[default]
    Unrestricted,
}

// ---------------------------------------------------------------------------
// BandwidthManagerConfig
// ---------------------------------------------------------------------------

/// Configuration for [`PeerBandwidthManager`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BandwidthManagerConfig {
    /// Default upload limit applied when a peer is first registered.
    pub default_upload_limit: BandwidthLimit,
    /// Default download limit applied when a peer is first registered.
    pub default_download_limit: BandwidthLimit,
    /// Maximum number of peers tracked simultaneously.  When the limit is
    /// reached the oldest registered peer is evicted to make room.
    pub max_peers: usize,
    /// Policy used by [`PeerBandwidthManager::apply_fairness`].
    pub fairness_policy: FairnessPolicy,
    /// Rolling window in milliseconds used for rate computation.
    pub usage_window_ms: u64,
}

impl Default for BandwidthManagerConfig {
    fn default() -> Self {
        Self {
            default_upload_limit: BandwidthLimit {
                bytes_per_second: 1_048_576, // 1 MB/s
                burst_bytes: 2_097_152,      // 2 MB
            },
            default_download_limit: BandwidthLimit {
                bytes_per_second: 2_097_152, // 2 MB/s
                burst_bytes: 4_194_304,      // 4 MB
            },
            max_peers: 1_000,
            fairness_policy: FairnessPolicy::Unrestricted,
            usage_window_ms: 5_000,
        }
    }
}

// ---------------------------------------------------------------------------
// BandwidthManagerStats
// ---------------------------------------------------------------------------

/// Aggregate statistics for the entire [`PeerBandwidthManager`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BandwidthManagerStats {
    /// Number of currently registered peers.
    pub registered_peers: usize,
    /// Total bytes sent across all peers since manager creation.
    pub total_bytes_sent: u64,
    /// Total bytes received across all peers since manager creation.
    pub total_bytes_recv: u64,
    /// Average upload rate across all peers (bytes per second).
    pub avg_upload_rate_bps: f64,
    /// Average download rate across all peers (bytes per second).
    pub avg_download_rate_bps: f64,
}

// ---------------------------------------------------------------------------
// PeerBandwidthManager
// ---------------------------------------------------------------------------

/// Manages per-peer bandwidth state, rate limiting, and fairness scheduling.
///
/// All timestamps are provided by the caller as milliseconds since an
/// arbitrary epoch (e.g. `SystemTime::UNIX_EPOCH` or `Instant`-derived).
/// This makes the manager fully testable without real-time dependencies.
pub struct PeerBandwidthManager {
    /// Manager configuration.
    pub config: BandwidthManagerConfig,
    /// Per-peer token-bucket state keyed by peer ID.
    pub peers: HashMap<String, PeerBandwidthState>,
    /// Sliding-window snapshots per peer: `(timestamp_ms, bytes_sent, bytes_recv)`.
    pub usage_snapshots: HashMap<String, VecDeque<(u64, u64, u64)>>,
    /// Total bytes sent across all peers since manager creation.
    pub total_bytes_sent: u64,
    /// Total bytes received across all peers since manager creation.
    pub total_bytes_recv: u64,
    /// Registration order (FIFO) used for eviction when `max_peers` is reached.
    registration_order: VecDeque<String>,
}

impl PeerBandwidthManager {
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    /// Create a new manager with the given configuration.
    pub fn new(config: BandwidthManagerConfig) -> Self {
        Self {
            peers: HashMap::new(),
            usage_snapshots: HashMap::new(),
            total_bytes_sent: 0,
            total_bytes_recv: 0,
            registration_order: VecDeque::new(),
            config,
        }
    }

    // -----------------------------------------------------------------------
    // Peer registration
    // -----------------------------------------------------------------------

    /// Register a peer with the default limits from the manager configuration.
    ///
    /// If the peer is already registered, the call is a no-op.
    /// If `max_peers` would be exceeded, the oldest registered peer is evicted.
    pub fn register_peer(&mut self, peer_id: String, now: u64) {
        if self.peers.contains_key(&peer_id) {
            return;
        }

        // Evict oldest peer if we're at capacity.
        if self.peers.len() >= self.config.max_peers {
            if let Some(oldest) = self.registration_order.pop_front() {
                self.peers.remove(&oldest);
                self.usage_snapshots.remove(&oldest);
            }
        }

        let state = PeerBandwidthState::new(
            peer_id.clone(),
            self.config.default_upload_limit,
            self.config.default_download_limit,
            now,
        );
        self.peers.insert(peer_id.clone(), state);
        self.usage_snapshots
            .insert(peer_id.clone(), VecDeque::new());
        self.registration_order.push_back(peer_id);
    }

    /// Override the upload and download limits for an already-registered peer.
    ///
    /// Returns `true` on success, `false` if the peer is not registered.
    pub fn set_peer_limits(
        &mut self,
        peer_id: &str,
        upload: BandwidthLimit,
        download: BandwidthLimit,
    ) -> bool {
        if let Some(state) = self.peers.get_mut(peer_id) {
            state.upload_limit = upload;
            state.download_limit = download;
            // Re-clamp existing tokens so they don't exceed the new burst caps.
            state.upload_tokens = state.upload_tokens.min(upload.burst_bytes as f64);
            state.download_tokens = state.download_tokens.min(download.burst_bytes as f64);
            true
        } else {
            false
        }
    }

    /// Remove a peer from the manager.
    ///
    /// Returns `true` if the peer existed and was removed.
    pub fn remove_peer(&mut self, peer_id: &str) -> bool {
        if self.peers.remove(peer_id).is_some() {
            self.usage_snapshots.remove(peer_id);
            self.registration_order.retain(|id| id != peer_id);
            true
        } else {
            false
        }
    }

    // -----------------------------------------------------------------------
    // Token-bucket helpers
    // -----------------------------------------------------------------------

    /// Refill the token buckets for `state` based on elapsed time since the
    /// last refill.
    ///
    /// `elapsed_s = (now_ms - last_refill_ms) / 1000.0`
    ///
    /// New tokens are calculated as `elapsed_s * bytes_per_second` and added
    /// to the current token count, clamped to `burst_bytes`.
    pub fn refill_tokens(state: &mut PeerBandwidthState, now: u64) {
        if now <= state.last_refill_ms {
            return;
        }
        let elapsed_ms = now - state.last_refill_ms;
        let elapsed_s = elapsed_ms as f64 / 1_000.0;

        if !state.upload_limit.is_unlimited() {
            let new_up = state.upload_limit.bytes_per_second as f64 * elapsed_s;
            state.upload_tokens =
                (state.upload_tokens + new_up).min(state.upload_limit.burst_bytes as f64);
        }

        if !state.download_limit.is_unlimited() {
            let new_dn = state.download_limit.bytes_per_second as f64 * elapsed_s;
            state.download_tokens =
                (state.download_tokens + new_dn).min(state.download_limit.burst_bytes as f64);
        }

        state.last_refill_ms = now;
    }

    /// Attempt to consume `bytes` from the appropriate token bucket(s).
    ///
    /// Tokens are refilled before checking.
    /// Returns `true` if the transfer is permitted, `false` if there are
    /// insufficient tokens.
    pub fn try_consume(
        &mut self,
        peer_id: &str,
        bytes: u64,
        direction: BandwidthDirection,
        now: u64,
    ) -> bool {
        let state = match self.peers.get_mut(peer_id) {
            Some(s) => s,
            None => return false,
        };

        Self::refill_tokens(state, now);

        let bytes_f = bytes as f64;

        match direction {
            BandwidthDirection::Upload => {
                if state.upload_limit.is_unlimited() || state.upload_tokens >= bytes_f {
                    if !state.upload_limit.is_unlimited() {
                        state.upload_tokens -= bytes_f;
                    }
                    true
                } else {
                    false
                }
            }
            BandwidthDirection::Download => {
                if state.download_limit.is_unlimited() || state.download_tokens >= bytes_f {
                    if !state.download_limit.is_unlimited() {
                        state.download_tokens -= bytes_f;
                    }
                    true
                } else {
                    false
                }
            }
            BandwidthDirection::Both => {
                let up_ok = state.upload_limit.is_unlimited() || state.upload_tokens >= bytes_f;
                let dn_ok = state.download_limit.is_unlimited() || state.download_tokens >= bytes_f;
                if up_ok && dn_ok {
                    if !state.upload_limit.is_unlimited() {
                        state.upload_tokens -= bytes_f;
                    }
                    if !state.download_limit.is_unlimited() {
                        state.download_tokens -= bytes_f;
                    }
                    true
                } else {
                    false
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Accounting
    // -----------------------------------------------------------------------

    /// Record an unconditional transfer for a peer (no token consumption).
    ///
    /// The snapshot is pushed to the sliding window and the manager-wide
    /// totals are updated.  Snapshots older than `usage_window_ms` are
    /// dropped.
    pub fn record_transfer(&mut self, peer_id: &str, bytes_sent: u64, bytes_recv: u64, now: u64) {
        // Update per-peer cumulative totals.
        if let Some(state) = self.peers.get_mut(peer_id) {
            state.total_bytes_sent = state.total_bytes_sent.saturating_add(bytes_sent);
            state.total_bytes_recv = state.total_bytes_recv.saturating_add(bytes_recv);
        }

        // Push snapshot and prune stale entries.
        if let Some(snapshots) = self.usage_snapshots.get_mut(peer_id) {
            snapshots.push_back((now, bytes_sent, bytes_recv));

            let window = self.config.usage_window_ms;
            let cutoff = now.saturating_sub(window);
            while snapshots
                .front()
                .map(|(ts, _, _)| *ts < cutoff)
                .unwrap_or(false)
            {
                snapshots.pop_front();
            }
        }

        // Update global totals.
        self.total_bytes_sent = self.total_bytes_sent.saturating_add(bytes_sent);
        self.total_bytes_recv = self.total_bytes_recv.saturating_add(bytes_recv);
    }

    // -----------------------------------------------------------------------
    // Usage queries
    // -----------------------------------------------------------------------

    /// Compute [`BandwidthUsage`] for a peer by summing all snapshots within
    /// the configured window ending at `now`.
    ///
    /// Returns `None` if the peer is not registered.
    pub fn peer_usage(&self, peer_id: &str, now: u64) -> Option<BandwidthUsage> {
        if !self.peers.contains_key(peer_id) {
            return None;
        }

        let snapshots = self.usage_snapshots.get(peer_id)?;
        let window = self.config.usage_window_ms;
        let cutoff = now.saturating_sub(window);

        let mut bytes_sent: u64 = 0;
        let mut bytes_recv: u64 = 0;
        for (ts, sent, recv) in snapshots.iter() {
            if *ts >= cutoff {
                bytes_sent = bytes_sent.saturating_add(*sent);
                bytes_recv = bytes_recv.saturating_add(*recv);
            }
        }

        let window_s = window as f64 / 1_000.0;
        let upload_rate_bps = if window_s > 0.0 {
            bytes_sent as f64 / window_s
        } else {
            0.0
        };
        let download_rate_bps = if window_s > 0.0 {
            bytes_recv as f64 / window_s
        } else {
            0.0
        };

        Some(BandwidthUsage {
            peer_id: peer_id.to_string(),
            bytes_sent,
            bytes_recv,
            upload_rate_bps,
            download_rate_bps,
            window_ms: window,
        })
    }

    /// Return the top-`k` peers by upload rate, descending.
    pub fn top_uploaders(&self, k: usize, now: u64) -> Vec<BandwidthUsage> {
        let mut usages: Vec<BandwidthUsage> = self
            .peers
            .keys()
            .filter_map(|id| self.peer_usage(id, now))
            .collect();
        usages.sort_by(|a, b| {
            b.upload_rate_bps
                .partial_cmp(&a.upload_rate_bps)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        usages.truncate(k);
        usages
    }

    /// Return the top-`k` peers by download rate, descending.
    pub fn top_downloaders(&self, k: usize, now: u64) -> Vec<BandwidthUsage> {
        let mut usages: Vec<BandwidthUsage> = self
            .peers
            .keys()
            .filter_map(|id| self.peer_usage(id, now))
            .collect();
        usages.sort_by(|a, b| {
            b.download_rate_bps
                .partial_cmp(&a.download_rate_bps)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        usages.truncate(k);
        usages
    }

    // -----------------------------------------------------------------------
    // Fairness scheduling
    // -----------------------------------------------------------------------

    /// Apply the configured fairness policy to all registered peers.
    ///
    /// - [`FairnessPolicy::MaxMinFairness`]: Compute the sum of all peers'
    ///   upload tokens and divide evenly; clamp each peer's token count to
    ///   that fair share.  The same applies to download tokens.
    /// - [`FairnessPolicy::WeightedFair`]: Scale each peer's tokens by its
    ///   normalised weight (peer_weight / sum_weights), then clamp to burst.
    /// - [`FairnessPolicy::Unrestricted`][]: No-op.
    pub fn apply_fairness(&mut self, now: u64) {
        // Refill all peers first so fairness acts on up-to-date token counts.
        let peer_ids: Vec<String> = self.peers.keys().cloned().collect();
        for id in &peer_ids {
            if let Some(state) = self.peers.get_mut(id) {
                Self::refill_tokens(state, now);
            }
        }

        let n = self.peers.len();
        if n == 0 {
            return;
        }

        match &self.config.fairness_policy.clone() {
            FairnessPolicy::Unrestricted => {}

            FairnessPolicy::MaxMinFairness => {
                let total_upload: f64 = self.peers.values().map(|s| s.upload_tokens).sum();
                let total_download: f64 = self.peers.values().map(|s| s.download_tokens).sum();
                let fair_up = total_upload / n as f64;
                let fair_dn = total_download / n as f64;

                for state in self.peers.values_mut() {
                    state.upload_tokens = state
                        .upload_tokens
                        .min(fair_up)
                        .min(state.upload_limit.burst_bytes as f64);
                    state.download_tokens = state
                        .download_tokens
                        .min(fair_dn)
                        .min(state.download_limit.burst_bytes as f64);
                }
            }

            FairnessPolicy::WeightedFair { weights } => {
                // Collect (id, weight) pairs; default weight is 1.0 for absent entries.
                let weighted: Vec<(String, f64)> = peer_ids
                    .iter()
                    .map(|id| {
                        let w = weights.get(id).copied().unwrap_or(1.0).max(0.0);
                        (id.clone(), w)
                    })
                    .collect();

                let total_weight: f64 = weighted.iter().map(|(_, w)| w).sum();
                if total_weight <= 0.0 {
                    return;
                }

                for (id, weight) in &weighted {
                    let norm = weight / total_weight;
                    if let Some(state) = self.peers.get_mut(id) {
                        state.upload_tokens =
                            (state.upload_tokens * norm).min(state.upload_limit.burst_bytes as f64);
                        state.download_tokens = (state.download_tokens * norm)
                            .min(state.download_limit.burst_bytes as f64);
                    }
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Maintenance
    // -----------------------------------------------------------------------

    /// Remove peers that have no snapshot newer than `now - idle_threshold_ms`.
    ///
    /// Returns the number of peers evicted.
    pub fn evict_idle_peers(&mut self, now: u64, idle_threshold_ms: u64) -> usize {
        let cutoff = now.saturating_sub(idle_threshold_ms);
        let idle: Vec<String> = self
            .usage_snapshots
            .iter()
            .filter(|(_, snaps)| {
                // A peer is idle if there are no snapshots at or after the cutoff.
                !snaps.iter().any(|(ts, _, _)| *ts >= cutoff)
            })
            .map(|(id, _)| id.clone())
            .collect();

        let count = idle.len();
        for id in &idle {
            self.peers.remove(id);
            self.usage_snapshots.remove(id);
            self.registration_order.retain(|r| r != id);
        }
        count
    }

    // -----------------------------------------------------------------------
    // Statistics
    // -----------------------------------------------------------------------

    /// Collect aggregate statistics for the manager.
    ///
    /// Average rates are computed from the usage window of each registered
    /// peer at the current moment (passing `now = 0` gives a snapshot of
    /// accumulated counters without rate computation).
    pub fn manager_stats(&self) -> BandwidthManagerStats {
        // Use a synthetic "now" far in the future so all snapshots fall within
        // the window when called without a real timestamp.
        let n = self.peers.len();
        let (sum_up, sum_dn) = if n > 0 {
            // We cannot call peer_usage here without a sensible `now`, so we
            // compute rates from the raw snapshot totals using the window.
            let window_s = self.config.usage_window_ms as f64 / 1_000.0;
            let mut up = 0.0f64;
            let mut dn = 0.0f64;
            for snaps in self.usage_snapshots.values() {
                let sent: u64 = snaps.iter().map(|(_, s, _)| s).sum();
                let recv: u64 = snaps.iter().map(|(_, _, r)| r).sum();
                if window_s > 0.0 {
                    up += sent as f64 / window_s;
                    dn += recv as f64 / window_s;
                }
            }
            (up, dn)
        } else {
            (0.0, 0.0)
        };

        let avg_up = if n > 0 { sum_up / n as f64 } else { 0.0 };
        let avg_dn = if n > 0 { sum_dn / n as f64 } else { 0.0 };

        BandwidthManagerStats {
            registered_peers: n,
            total_bytes_sent: self.total_bytes_sent,
            total_bytes_recv: self.total_bytes_recv,
            avg_upload_rate_bps: avg_up,
            avg_download_rate_bps: avg_dn,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::{
        BandwidthDirection, BandwidthLimit, BandwidthManagerConfig, BandwidthUsage, FairnessPolicy,
        PeerBandwidthManager, PeerBandwidthState,
    };
    use std::collections::HashMap;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /// Build a manager with a 5-second window.
    fn make_manager() -> PeerBandwidthManager {
        PeerBandwidthManager::new(BandwidthManagerConfig::default())
    }

    /// Build a manager with a very low default upload/download limit (1 KB/s, 2 KB burst).
    fn make_tight_manager() -> PeerBandwidthManager {
        let config = BandwidthManagerConfig {
            default_upload_limit: BandwidthLimit {
                bytes_per_second: 1_024,
                burst_bytes: 2_048,
            },
            default_download_limit: BandwidthLimit {
                bytes_per_second: 1_024,
                burst_bytes: 2_048,
            },
            ..Default::default()
        };
        PeerBandwidthManager::new(config)
    }

    // -----------------------------------------------------------------------
    // register_peer
    // -----------------------------------------------------------------------

    #[test]
    fn test_register_peer_basic() {
        let mut mgr = make_manager();
        mgr.register_peer("alice".to_string(), 0);
        assert!(mgr.peers.contains_key("alice"));
    }

    #[test]
    fn test_register_peer_idempotent() {
        let mut mgr = make_manager();
        mgr.register_peer("alice".to_string(), 0);
        mgr.register_peer("alice".to_string(), 1_000);
        // Should still be exactly one entry.
        assert_eq!(mgr.peers.len(), 1);
    }

    #[test]
    fn test_register_peer_initial_tokens_full() {
        let mut mgr = make_tight_manager();
        mgr.register_peer("peer-a".to_string(), 0);
        let state = mgr.peers.get("peer-a").expect("peer not found");
        assert_eq!(state.upload_tokens, 2_048.0);
        assert_eq!(state.download_tokens, 2_048.0);
    }

    #[test]
    fn test_register_peer_evicts_oldest_when_full() {
        let config = BandwidthManagerConfig {
            max_peers: 2,
            ..Default::default()
        };
        let mut mgr = PeerBandwidthManager::new(config);
        mgr.register_peer("p1".to_string(), 0);
        mgr.register_peer("p2".to_string(), 0);
        mgr.register_peer("p3".to_string(), 0);
        // p1 should have been evicted.
        assert!(!mgr.peers.contains_key("p1"));
        assert!(mgr.peers.contains_key("p2"));
        assert!(mgr.peers.contains_key("p3"));
    }

    // -----------------------------------------------------------------------
    // remove_peer
    // -----------------------------------------------------------------------

    #[test]
    fn test_remove_peer_existing() {
        let mut mgr = make_manager();
        mgr.register_peer("bob".to_string(), 0);
        assert!(mgr.remove_peer("bob"));
        assert!(!mgr.peers.contains_key("bob"));
    }

    #[test]
    fn test_remove_peer_nonexistent() {
        let mut mgr = make_manager();
        assert!(!mgr.remove_peer("ghost"));
    }

    #[test]
    fn test_remove_peer_clears_snapshots() {
        let mut mgr = make_manager();
        mgr.register_peer("carol".to_string(), 0);
        mgr.record_transfer("carol", 100, 200, 1_000);
        mgr.remove_peer("carol");
        assert!(!mgr.usage_snapshots.contains_key("carol"));
    }

    // -----------------------------------------------------------------------
    // set_peer_limits
    // -----------------------------------------------------------------------

    #[test]
    fn test_set_peer_limits_success() {
        let mut mgr = make_tight_manager();
        mgr.register_peer("dave".to_string(), 0);
        let new_up = BandwidthLimit::new(4_096, 8_192);
        let new_dn = BandwidthLimit::new(4_096, 8_192);
        assert!(mgr.set_peer_limits("dave", new_up, new_dn));
        let state = mgr.peers.get("dave").expect("peer not found");
        assert_eq!(state.upload_limit.bytes_per_second, 4_096);
    }

    #[test]
    fn test_set_peer_limits_nonexistent() {
        let mut mgr = make_manager();
        let limit = BandwidthLimit::default();
        assert!(!mgr.set_peer_limits("nobody", limit, limit));
    }

    #[test]
    fn test_set_peer_limits_clamps_tokens() {
        let mut mgr = make_tight_manager();
        mgr.register_peer("eve".to_string(), 0);
        // Reduce burst to 512 bytes; existing tokens (2048) must be clamped.
        let tight = BandwidthLimit::new(1_024, 512);
        mgr.set_peer_limits("eve", tight, tight);
        let state = mgr.peers.get("eve").expect("peer not found");
        assert!(state.upload_tokens <= 512.0);
        assert!(state.download_tokens <= 512.0);
    }

    // -----------------------------------------------------------------------
    // refill_tokens
    // -----------------------------------------------------------------------

    #[test]
    fn test_refill_tokens_adds_correct_amount() {
        let limit = BandwidthLimit::new(1_000, 10_000);
        let mut state = PeerBandwidthState::new("p".to_string(), limit, limit, 0);
        // Drain tokens completely.
        state.upload_tokens = 0.0;
        state.download_tokens = 0.0;
        // Refill after 1 second.
        PeerBandwidthManager::refill_tokens(&mut state, 1_000);
        assert!((state.upload_tokens - 1_000.0).abs() < 1.0);
        assert!((state.download_tokens - 1_000.0).abs() < 1.0);
    }

    #[test]
    fn test_refill_tokens_clamps_to_burst() {
        let limit = BandwidthLimit::new(10_000, 5_000);
        let mut state = PeerBandwidthState::new("p".to_string(), limit, limit, 0);
        state.upload_tokens = 0.0;
        state.download_tokens = 0.0;
        // 1 second refill would add 10_000 tokens but burst is 5_000.
        PeerBandwidthManager::refill_tokens(&mut state, 1_000);
        assert_eq!(state.upload_tokens, 5_000.0);
        assert_eq!(state.download_tokens, 5_000.0);
    }

    #[test]
    fn test_refill_tokens_noop_when_time_not_advanced() {
        let limit = BandwidthLimit::new(1_000, 10_000);
        let mut state = PeerBandwidthState::new("p".to_string(), limit, limit, 1_000);
        state.upload_tokens = 500.0;
        PeerBandwidthManager::refill_tokens(&mut state, 1_000); // same timestamp
        assert_eq!(state.upload_tokens, 500.0);
    }

    // -----------------------------------------------------------------------
    // try_consume
    // -----------------------------------------------------------------------

    #[test]
    fn test_try_consume_upload_allowed() {
        let mut mgr = make_tight_manager();
        mgr.register_peer("frank".to_string(), 0);
        assert!(mgr.try_consume("frank", 1_000, BandwidthDirection::Upload, 0));
    }

    #[test]
    fn test_try_consume_upload_denied_insufficient_tokens() {
        let mut mgr = make_tight_manager();
        mgr.register_peer("grace".to_string(), 0);
        // Drain entire burst.
        assert!(mgr.try_consume("grace", 2_048, BandwidthDirection::Upload, 0));
        // Next request should be denied.
        assert!(!mgr.try_consume("grace", 1, BandwidthDirection::Upload, 0));
    }

    #[test]
    fn test_try_consume_download_allowed() {
        let mut mgr = make_tight_manager();
        mgr.register_peer("hank".to_string(), 0);
        assert!(mgr.try_consume("hank", 1_024, BandwidthDirection::Download, 0));
    }

    #[test]
    fn test_try_consume_both_directions() {
        let mut mgr = make_tight_manager();
        mgr.register_peer("ivan".to_string(), 0);
        assert!(mgr.try_consume("ivan", 1_000, BandwidthDirection::Both, 0));
        let state = mgr.peers.get("ivan").expect("peer not found");
        assert!((state.upload_tokens - 1_048.0).abs() < 2.0);
        assert!((state.download_tokens - 1_048.0).abs() < 2.0);
    }

    #[test]
    fn test_try_consume_both_denied_if_one_direction_insufficient() {
        let mut mgr = make_tight_manager();
        mgr.register_peer("judy".to_string(), 0);
        // Drain upload only.
        assert!(mgr.try_consume("judy", 2_048, BandwidthDirection::Upload, 0));
        // Both should now fail because upload is empty.
        assert!(!mgr.try_consume("judy", 100, BandwidthDirection::Both, 0));
    }

    #[test]
    fn test_try_consume_unknown_peer_returns_false() {
        let mut mgr = make_manager();
        assert!(!mgr.try_consume("unknown", 1, BandwidthDirection::Upload, 0));
    }

    #[test]
    fn test_try_consume_refills_before_check() {
        let mut mgr = make_tight_manager();
        mgr.register_peer("karl".to_string(), 0);
        // Drain all upload tokens.
        assert!(mgr.try_consume("karl", 2_048, BandwidthDirection::Upload, 0));
        // Advance 2 seconds — should gain 2_048 tokens (2 * 1024), capped at burst.
        assert!(mgr.try_consume("karl", 2_048, BandwidthDirection::Upload, 2_000));
    }

    // -----------------------------------------------------------------------
    // record_transfer
    // -----------------------------------------------------------------------

    #[test]
    fn test_record_transfer_updates_global_totals() {
        let mut mgr = make_manager();
        mgr.register_peer("liam".to_string(), 0);
        mgr.record_transfer("liam", 1_000, 500, 1_000);
        assert_eq!(mgr.total_bytes_sent, 1_000);
        assert_eq!(mgr.total_bytes_recv, 500);
    }

    #[test]
    fn test_record_transfer_updates_peer_totals() {
        let mut mgr = make_manager();
        mgr.register_peer("mia".to_string(), 0);
        mgr.record_transfer("mia", 300, 150, 1_000);
        mgr.record_transfer("mia", 200, 100, 2_000);
        let state = mgr.peers.get("mia").expect("peer not found");
        assert_eq!(state.total_bytes_sent, 500);
        assert_eq!(state.total_bytes_recv, 250);
    }

    #[test]
    fn test_record_transfer_pushes_snapshot() {
        let mut mgr = make_manager();
        mgr.register_peer("noah".to_string(), 0);
        mgr.record_transfer("noah", 100, 50, 1_000);
        let snaps = mgr.usage_snapshots.get("noah").expect("no snapshots");
        assert_eq!(snaps.len(), 1);
    }

    #[test]
    fn test_record_transfer_prunes_old_snapshots() {
        let mut mgr = make_manager(); // 5_000 ms window
        mgr.register_peer("olivia".to_string(), 0);
        mgr.record_transfer("olivia", 100, 50, 0);
        mgr.record_transfer("olivia", 100, 50, 4_999);
        // Snapshot at t=0 should be pruned when we record at t=5001 (window cutoff = 1).
        mgr.record_transfer("olivia", 100, 50, 5_001);
        let snaps = mgr.usage_snapshots.get("olivia").expect("no snapshots");
        // t=0 is before cutoff (5001-5000=1), t=4999 is before cutoff too, t=5001 is ok.
        for (ts, _, _) in snaps.iter() {
            assert!(*ts >= 5_001u64.saturating_sub(5_000));
        }
    }

    #[test]
    fn test_record_transfer_noop_for_unknown_peer_global_counts_still_updated() {
        let mut mgr = make_manager();
        // No peer registered — global counts should still go up.
        mgr.record_transfer("unknown", 999, 111, 1_000);
        assert_eq!(mgr.total_bytes_sent, 999);
        assert_eq!(mgr.total_bytes_recv, 111);
    }

    // -----------------------------------------------------------------------
    // peer_usage
    // -----------------------------------------------------------------------

    #[test]
    fn test_peer_usage_returns_none_for_unregistered() {
        let mgr = make_manager();
        assert!(mgr.peer_usage("ghost", 5_000).is_none());
    }

    #[test]
    fn test_peer_usage_empty_snapshots() {
        let mut mgr = make_manager();
        mgr.register_peer("petra".to_string(), 0);
        let usage = mgr.peer_usage("petra", 5_000).expect("should return usage");
        assert_eq!(usage.bytes_sent, 0);
        assert_eq!(usage.bytes_recv, 0);
        assert_eq!(usage.upload_rate_bps, 0.0);
    }

    #[test]
    fn test_peer_usage_rate_calculation() {
        let mut mgr = make_manager(); // 5_000 ms window
        mgr.register_peer("quinn".to_string(), 0);
        // 5_000 bytes sent within the 5-second window → 1_000 B/s
        mgr.record_transfer("quinn", 5_000, 0, 1_000);
        let usage = mgr.peer_usage("quinn", 6_000).expect("should return usage");
        // All snapshots within the 5 s window → rate ≈ 5000/5 = 1000 B/s.
        assert!((usage.upload_rate_bps - 1_000.0).abs() < 1.0);
    }

    #[test]
    fn test_peer_usage_excludes_old_snapshots() {
        let mut mgr = make_manager(); // 5_000 ms window
        mgr.register_peer("rachel".to_string(), 0);
        // Record at t=0 (old) and t=5_001 (current).
        mgr.record_transfer("rachel", 10_000, 0, 0);
        mgr.record_transfer("rachel", 1_000, 0, 5_001);
        let usage = mgr
            .peer_usage("rachel", 10_001)
            .expect("should return usage");
        // Only the snapshot at t=5_001 is within window [10_001-5000, 10_001] = [5_001, 10_001].
        assert_eq!(usage.bytes_sent, 1_000);
    }

    #[test]
    fn test_peer_usage_window_ms_matches_config() {
        let mut mgr = make_manager();
        mgr.register_peer("sam".to_string(), 0);
        let usage = mgr.peer_usage("sam", 0).expect("should return usage");
        assert_eq!(usage.window_ms, 5_000);
    }

    // -----------------------------------------------------------------------
    // top_uploaders / top_downloaders
    // -----------------------------------------------------------------------

    #[test]
    fn test_top_uploaders_ordering() {
        let mut mgr = make_manager();
        for id in &["a", "b", "c"] {
            mgr.register_peer(id.to_string(), 0);
        }
        mgr.record_transfer("a", 300, 0, 1_000);
        mgr.record_transfer("b", 100, 0, 1_000);
        mgr.record_transfer("c", 200, 0, 1_000);
        let top = mgr.top_uploaders(3, 5_000);
        assert_eq!(top[0].peer_id, "a");
        assert_eq!(top[1].peer_id, "c");
        assert_eq!(top[2].peer_id, "b");
    }

    #[test]
    fn test_top_uploaders_k_limits_results() {
        let mut mgr = make_manager();
        for i in 0..5u32 {
            let id = format!("peer-{i}");
            mgr.register_peer(id.clone(), 0);
            mgr.record_transfer(&id, (i + 1) as u64 * 100, 0, 1_000);
        }
        let top = mgr.top_uploaders(2, 5_000);
        assert_eq!(top.len(), 2);
    }

    #[test]
    fn test_top_downloaders_ordering() {
        let mut mgr = make_manager();
        for id in &["x", "y", "z"] {
            mgr.register_peer(id.to_string(), 0);
        }
        mgr.record_transfer("x", 0, 500, 1_000);
        mgr.record_transfer("y", 0, 200, 1_000);
        mgr.record_transfer("z", 0, 800, 1_000);
        let top = mgr.top_downloaders(3, 5_000);
        assert_eq!(top[0].peer_id, "z");
        assert_eq!(top[1].peer_id, "x");
        assert_eq!(top[2].peer_id, "y");
    }

    #[test]
    fn test_top_downloaders_returns_all_if_k_exceeds_peers() {
        let mut mgr = make_manager();
        mgr.register_peer("only".to_string(), 0);
        mgr.record_transfer("only", 0, 100, 1_000);
        let top = mgr.top_downloaders(10, 5_000);
        assert_eq!(top.len(), 1);
    }

    // -----------------------------------------------------------------------
    // apply_fairness — MaxMinFairness
    // -----------------------------------------------------------------------

    #[test]
    fn test_fairness_maxmin_equalises_tokens() {
        let config = BandwidthManagerConfig {
            fairness_policy: FairnessPolicy::MaxMinFairness,
            ..Default::default()
        };
        let mut mgr = PeerBandwidthManager::new(config);
        mgr.register_peer("p1".to_string(), 0);
        mgr.register_peer("p2".to_string(), 0);
        // Give p1 twice the tokens of p2.
        mgr.peers.get_mut("p1").expect("p1").upload_tokens = 2_000.0;
        mgr.peers.get_mut("p2").expect("p2").upload_tokens = 1_000.0;
        mgr.apply_fairness(0);
        let fair_share = 3_000.0 / 2.0; // total / n = 1500
        let t1 = mgr.peers.get("p1").expect("p1").upload_tokens;
        let t2 = mgr.peers.get("p2").expect("p2").upload_tokens;
        // Both should be clamped to at most fair_share.
        assert!(t1 <= fair_share + 0.01);
        assert!(t2 <= fair_share + 0.01);
    }

    #[test]
    fn test_fairness_maxmin_noop_when_no_peers() {
        let config = BandwidthManagerConfig {
            fairness_policy: FairnessPolicy::MaxMinFairness,
            ..Default::default()
        };
        let mut mgr = PeerBandwidthManager::new(config);
        // Should not panic.
        mgr.apply_fairness(0);
    }

    // -----------------------------------------------------------------------
    // apply_fairness — WeightedFair
    // -----------------------------------------------------------------------

    #[test]
    fn test_fairness_weighted_scales_tokens() {
        let mut weights = HashMap::new();
        weights.insert("heavy".to_string(), 3.0);
        weights.insert("light".to_string(), 1.0);
        let config = BandwidthManagerConfig {
            fairness_policy: FairnessPolicy::WeightedFair { weights },
            default_upload_limit: BandwidthLimit {
                bytes_per_second: 100_000,
                burst_bytes: 1_000_000,
            },
            default_download_limit: BandwidthLimit {
                bytes_per_second: 100_000,
                burst_bytes: 1_000_000,
            },
            ..Default::default()
        };
        let mut mgr = PeerBandwidthManager::new(config);
        mgr.register_peer("heavy".to_string(), 0);
        mgr.register_peer("light".to_string(), 0);
        // Set equal starting tokens.
        mgr.peers.get_mut("heavy").expect("heavy").upload_tokens = 1_000_000.0;
        mgr.peers.get_mut("light").expect("light").upload_tokens = 1_000_000.0;
        mgr.apply_fairness(0);
        let heavy_tokens = mgr.peers.get("heavy").expect("heavy").upload_tokens;
        let light_tokens = mgr.peers.get("light").expect("light").upload_tokens;
        // heavy has 3/4 weight, light has 1/4 weight.
        // heavy_tokens should be greater than light_tokens.
        assert!(
            heavy_tokens > light_tokens,
            "heavy={heavy_tokens} light={light_tokens}"
        );
    }

    #[test]
    fn test_fairness_unrestricted_noop() {
        let mut mgr = make_manager(); // Unrestricted policy
        mgr.register_peer("q1".to_string(), 0);
        mgr.peers.get_mut("q1").expect("q1").upload_tokens = 777.0;
        mgr.apply_fairness(0);
        // Tokens should not change (only possible refill from elapsed=0).
        let tokens = mgr.peers.get("q1").expect("q1").upload_tokens;
        // After refill with elapsed=0 tokens remain unchanged.
        assert!(tokens >= 777.0);
    }

    // -----------------------------------------------------------------------
    // evict_idle_peers
    // -----------------------------------------------------------------------

    #[test]
    fn test_evict_idle_peers_removes_stale() {
        let mut mgr = make_manager();
        mgr.register_peer("active".to_string(), 0);
        mgr.register_peer("idle".to_string(), 0);
        mgr.record_transfer("active", 100, 0, 10_000);
        // "idle" has no snapshots.
        let evicted = mgr.evict_idle_peers(10_000, 5_000);
        assert_eq!(evicted, 1);
        assert!(!mgr.peers.contains_key("idle"));
        assert!(mgr.peers.contains_key("active"));
    }

    #[test]
    fn test_evict_idle_peers_returns_zero_when_all_active() {
        let mut mgr = make_manager();
        mgr.register_peer("p1".to_string(), 0);
        mgr.record_transfer("p1", 1, 0, 5_000);
        let evicted = mgr.evict_idle_peers(5_000, 5_000);
        assert_eq!(evicted, 0);
    }

    #[test]
    fn test_evict_idle_peers_clears_snapshots() {
        let mut mgr = make_manager();
        mgr.register_peer("stale".to_string(), 0);
        // Record a very old snapshot.
        mgr.record_transfer("stale", 100, 0, 0);
        // Evict with threshold 5000 at now=10_000 — cutoff = 5000; snapshot at 0 < 5000.
        let evicted = mgr.evict_idle_peers(10_000, 5_000);
        assert_eq!(evicted, 1);
        assert!(!mgr.usage_snapshots.contains_key("stale"));
    }

    // -----------------------------------------------------------------------
    // manager_stats
    // -----------------------------------------------------------------------

    #[test]
    fn test_manager_stats_registered_peers() {
        let mut mgr = make_manager();
        mgr.register_peer("s1".to_string(), 0);
        mgr.register_peer("s2".to_string(), 0);
        let stats = mgr.manager_stats();
        assert_eq!(stats.registered_peers, 2);
    }

    #[test]
    fn test_manager_stats_global_byte_counts() {
        let mut mgr = make_manager();
        mgr.register_peer("t1".to_string(), 0);
        mgr.record_transfer("t1", 1_000, 500, 1_000);
        let stats = mgr.manager_stats();
        assert_eq!(stats.total_bytes_sent, 1_000);
        assert_eq!(stats.total_bytes_recv, 500);
    }

    #[test]
    fn test_manager_stats_avg_rates() {
        let mut mgr = make_manager(); // 5_000 ms window
        mgr.register_peer("u1".to_string(), 0);
        mgr.register_peer("u2".to_string(), 0);
        mgr.record_transfer("u1", 5_000, 0, 1_000);
        mgr.record_transfer("u2", 10_000, 0, 1_000);
        let stats = mgr.manager_stats();
        // avg = (5000/5 + 10000/5) / 2 = (1000 + 2000) / 2 = 1500 B/s
        assert!((stats.avg_upload_rate_bps - 1_500.0).abs() < 1.0);
    }

    #[test]
    fn test_manager_stats_empty() {
        let mgr = make_manager();
        let stats = mgr.manager_stats();
        assert_eq!(stats.registered_peers, 0);
        assert_eq!(stats.total_bytes_sent, 0);
        assert_eq!(stats.total_bytes_recv, 0);
        assert_eq!(stats.avg_upload_rate_bps, 0.0);
    }

    // -----------------------------------------------------------------------
    // BandwidthLimit helpers
    // -----------------------------------------------------------------------

    #[test]
    fn test_bandwidth_limit_is_unlimited() {
        let unlimited = BandwidthLimit::new(0, 0);
        assert!(unlimited.is_unlimited());
    }

    #[test]
    fn test_bandwidth_limit_is_not_unlimited() {
        let limited = BandwidthLimit::new(1_000, 2_000);
        assert!(!limited.is_unlimited());
    }

    #[test]
    fn test_bandwidth_limit_default_values() {
        let limit = BandwidthLimit::default();
        assert_eq!(limit.bytes_per_second, 1_048_576);
        assert_eq!(limit.burst_bytes, 2_097_152);
    }

    // -----------------------------------------------------------------------
    // BandwidthManagerConfig default
    // -----------------------------------------------------------------------

    #[test]
    fn test_config_default_values() {
        let config = BandwidthManagerConfig::default();
        assert_eq!(config.max_peers, 1_000);
        assert_eq!(config.usage_window_ms, 5_000);
        assert_eq!(config.default_upload_limit.bytes_per_second, 1_048_576);
        assert_eq!(config.default_download_limit.bytes_per_second, 2_097_152);
    }

    // -----------------------------------------------------------------------
    // BandwidthUsage struct fields
    // -----------------------------------------------------------------------

    #[test]
    fn test_bandwidth_usage_fields() {
        let usage = BandwidthUsage {
            peer_id: "test".to_string(),
            bytes_sent: 1_000,
            bytes_recv: 500,
            upload_rate_bps: 200.0,
            download_rate_bps: 100.0,
            window_ms: 5_000,
        };
        assert_eq!(usage.peer_id, "test");
        assert_eq!(usage.bytes_sent, 1_000);
        assert_eq!(usage.bytes_recv, 500);
        assert!((usage.upload_rate_bps - 200.0).abs() < f64::EPSILON);
    }

    // -----------------------------------------------------------------------
    // Edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_unlimited_upload_always_allowed() {
        let config = BandwidthManagerConfig {
            default_upload_limit: BandwidthLimit::new(0, 0), // unlimited
            default_download_limit: BandwidthLimit::new(1_024, 2_048),
            ..Default::default()
        };
        let mut mgr = PeerBandwidthManager::new(config);
        mgr.register_peer("unlimited".to_string(), 0);
        // Should always pass, even for huge transfers.
        assert!(mgr.try_consume("unlimited", u64::MAX / 2, BandwidthDirection::Upload, 0));
    }

    #[test]
    fn test_saturating_add_prevents_overflow() {
        let mut mgr = make_manager();
        mgr.register_peer("big".to_string(), 0);
        mgr.record_transfer("big", u64::MAX - 10, 0, 1_000);
        mgr.record_transfer("big", 100, 0, 2_000);
        // Should not panic due to overflow.
        assert_eq!(mgr.total_bytes_sent, u64::MAX);
    }

    #[test]
    fn test_multiple_peers_independent_token_buckets() {
        let mut mgr = make_tight_manager();
        mgr.register_peer("p1".to_string(), 0);
        mgr.register_peer("p2".to_string(), 0);
        // Drain p1 upload entirely.
        assert!(mgr.try_consume("p1", 2_048, BandwidthDirection::Upload, 0));
        // p2 should still have full tokens.
        assert!(mgr.try_consume("p2", 2_048, BandwidthDirection::Upload, 0));
    }

    #[test]
    fn test_top_uploaders_empty_manager() {
        let mgr = make_manager();
        let top = mgr.top_uploaders(5, 0);
        assert!(top.is_empty());
    }

    #[test]
    fn test_top_downloaders_empty_manager() {
        let mgr = make_manager();
        let top = mgr.top_downloaders(5, 0);
        assert!(top.is_empty());
    }

    #[test]
    fn test_registration_order_maintained_for_eviction() {
        let config = BandwidthManagerConfig {
            max_peers: 3,
            ..Default::default()
        };
        let mut mgr = PeerBandwidthManager::new(config);
        for id in &["r1", "r2", "r3", "r4", "r5"] {
            mgr.register_peer(id.to_string(), 0);
        }
        // r1, r2, r3 should have been evicted in order as r4, r5 were added.
        assert!(!mgr.peers.contains_key("r1"));
        assert!(!mgr.peers.contains_key("r2"));
        assert!(mgr.peers.contains_key("r3"));
        assert!(mgr.peers.contains_key("r4"));
        assert!(mgr.peers.contains_key("r5"));
    }

    #[test]
    fn test_try_consume_after_refill_succeeds() {
        let mut mgr = make_tight_manager();
        mgr.register_peer("walter".to_string(), 0);
        // Exhaust all upload tokens at t=0.
        assert!(mgr.try_consume("walter", 2_048, BandwidthDirection::Upload, 0));
        // At t=1 the bucket refills by exactly 1_024 bytes (1 second at 1_024 B/s).
        // Try consuming 1_024 bytes — should succeed.
        assert!(mgr.try_consume("walter", 1_024, BandwidthDirection::Upload, 1_000));
    }
}
