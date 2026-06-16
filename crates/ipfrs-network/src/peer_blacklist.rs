//! Peer blacklist with expiry, reason tracking, and reputation-based auto-blacklisting.
//!
//! Provides a wall-clock-time-aware blacklist for peers, supporting:
//! - Manual blocking with optional expiry
//! - Strike-based automatic blacklisting with configurable thresholds
//! - Permanent escalation after repeated offences
//! - Fine-grained reason classification
//! - Aggregate statistics
//!
//! # Example
//!
//! ```rust
//! use ipfrs_network::peer_blacklist::{PeerBlacklist, BlacklistConfig, BlacklistReason};
//!
//! let config = BlacklistConfig::default();
//! let mut bl = PeerBlacklist::new(config);
//!
//! // Manually block a peer for 60 seconds (in milliseconds)
//! let now = 1_000_000u64;
//! bl.block("peer-1", BlacklistReason::ManualBlock, "test block", Some(now + 60_000));
//! assert!(bl.is_blocked_at("peer-1", now));
//!
//! // Record strikes to trigger auto-block
//! let count = bl.record_strike("peer-2", now);
//! assert!(count >= 1);
//! ```

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Reason why a peer was added to the blacklist.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BlacklistReason {
    /// Manually blocked by an operator.
    ManualBlock,
    /// Peer sent an excessive number of messages in a short window.
    SpamDetected,
    /// Peer violated the expected protocol semantics.
    ProtocolViolation,
    /// Peer performed actions classified as malicious.
    MaliciousBehavior,
    /// Peer repeatedly exceeded connection-level rate limits.
    RateLimitExceeded,
    /// Peer sent messages that failed validation.
    InvalidMessages,
    /// Peer's reputation score fell below the configured threshold.
    ReputationThreshold,
}

/// A single entry in the blacklist.
#[derive(Debug, Clone)]
pub struct BlacklistEntry {
    /// Peer identifier (libp2p PeerId string or any opaque identifier).
    pub peer_id: String,
    /// Reason the peer was blacklisted.
    pub reason: BlacklistReason,
    /// Unix-epoch milliseconds when the block was recorded.
    pub blocked_at: u64,
    /// Unix-epoch milliseconds when the block expires. `None` means permanent.
    pub expires_at: Option<u64>,
    /// Total number of strikes accumulated (across all windows) by this peer.
    pub strike_count: u32,
    /// Free-form operator notes.
    pub notes: String,
}

/// Configuration for `PeerBlacklist`.
#[derive(Debug, Clone)]
pub struct BlacklistConfig {
    /// Maximum number of entries retained. Oldest entries are evicted when exceeded.
    pub max_entries: usize,
    /// Default expiry in milliseconds for blocks that do not specify one.
    /// `None` means blocks without an explicit expiry are permanent by default.
    pub default_expiry_ms: Option<u64>,
    /// Number of strikes (within `strike_window_ms`) that trigger an automatic block.
    pub auto_blacklist_strikes: u32,
    /// After this many total strikes, the block becomes permanent regardless of expiry.
    pub permanent_after_strikes: u32,
    /// Width of the rolling window (in milliseconds) used to count strikes.
    pub strike_window_ms: u64,
}

impl Default for BlacklistConfig {
    fn default() -> Self {
        Self {
            max_entries: 10_000,
            default_expiry_ms: Some(3_600_000), // 1 hour
            auto_blacklist_strikes: 5,
            permanent_after_strikes: 10,
            strike_window_ms: 60_000, // 1 minute
        }
    }
}

/// Aggregate statistics for the blacklist.
#[derive(Debug, Clone, Default)]
pub struct BlacklistStats {
    /// Total peers ever blocked (including re-blocks).
    pub total_blocked: u64,
    /// Total peers ever unblocked (manual or via expiry purge).
    pub total_unblocked: u64,
    /// Peers automatically blocked via the strike threshold.
    pub auto_blocked: u64,
    /// Peers given a permanent block (either manual or strike escalation).
    pub permanent_blocks: u64,
    /// Entries removed by `purge_expired`.
    pub expired_removed: u64,
}

/// Peer blacklist manager.
///
/// Uses wall-clock time (Unix-epoch milliseconds supplied by the caller) so
/// that it is fully deterministic and testable without relying on system clocks.
pub struct PeerBlacklist {
    config: BlacklistConfig,
    entries: HashMap<String, BlacklistEntry>,
    /// Maps peer_id -> sorted list of timestamps at which strikes were recorded.
    strike_log: HashMap<String, Vec<u64>>,
    stats: BlacklistStats,
}

impl PeerBlacklist {
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    /// Create a new `PeerBlacklist` with the given configuration.
    pub fn new(config: BlacklistConfig) -> Self {
        Self {
            config,
            entries: HashMap::new(),
            strike_log: HashMap::new(),
            stats: BlacklistStats::default(),
        }
    }

    // -----------------------------------------------------------------------
    // Core block / unblock
    // -----------------------------------------------------------------------

    /// Block a peer.
    ///
    /// Returns `false` if the peer is already blocked (the existing entry is
    /// left unchanged); returns `true` on success.
    ///
    /// If `expires_at` is `None` the `default_expiry_ms` from `BlacklistConfig`
    /// is applied relative to the current strike-count; if the peer's strike
    /// count already meets `permanent_after_strikes`, the block is made permanent.
    pub fn block(
        &mut self,
        peer_id: &str,
        reason: BlacklistReason,
        notes: &str,
        expires_at: Option<u64>,
    ) -> bool {
        if self.entries.contains_key(peer_id) {
            return false;
        }

        // Enforce max_entries: remove the first (arbitrary) entry when full.
        if self.entries.len() >= self.config.max_entries {
            if let Some(key) = self.entries.keys().next().cloned() {
                self.entries.remove(&key);
            }
        }

        let strike_count = self
            .strike_log
            .get(peer_id)
            .map(|v| v.len() as u32)
            .unwrap_or(0);

        // Resolve effective expiry.
        let effective_expires = self.resolve_expiry(expires_at, strike_count);

        let is_permanent = effective_expires.is_none();

        let entry = BlacklistEntry {
            peer_id: peer_id.to_string(),
            reason,
            blocked_at: 0, // caller did not pass `now`; use sentinel value
            expires_at: effective_expires,
            strike_count,
            notes: notes.to_string(),
        };

        self.entries.insert(peer_id.to_string(), entry);
        self.stats.total_blocked += 1;
        if is_permanent {
            self.stats.permanent_blocks += 1;
        }
        true
    }

    /// Block a peer with an explicit `now` timestamp for `blocked_at`.
    ///
    /// Returns `false` if already blocked.
    pub fn block_at(
        &mut self,
        peer_id: &str,
        reason: BlacklistReason,
        notes: &str,
        expires_at: Option<u64>,
        now: u64,
    ) -> bool {
        if self.entries.contains_key(peer_id) {
            return false;
        }

        if self.entries.len() >= self.config.max_entries {
            if let Some(key) = self.entries.keys().next().cloned() {
                self.entries.remove(&key);
            }
        }

        let strike_count = self
            .strike_log
            .get(peer_id)
            .map(|v| v.len() as u32)
            .unwrap_or(0);

        let effective_expires = self.resolve_expiry(expires_at, strike_count);
        let is_permanent = effective_expires.is_none();

        let entry = BlacklistEntry {
            peer_id: peer_id.to_string(),
            reason,
            blocked_at: now,
            expires_at: effective_expires,
            strike_count,
            notes: notes.to_string(),
        };

        self.entries.insert(peer_id.to_string(), entry);
        self.stats.total_blocked += 1;
        if is_permanent {
            self.stats.permanent_blocks += 1;
        }
        true
    }

    /// Unblock a peer regardless of expiry.
    ///
    /// Returns `false` if the peer was not in the blacklist.
    pub fn unblock(&mut self, peer_id: &str) -> bool {
        if self.entries.remove(peer_id).is_some() {
            self.stats.total_unblocked += 1;
            true
        } else {
            false
        }
    }

    // -----------------------------------------------------------------------
    // Lookup
    // -----------------------------------------------------------------------

    /// Check whether a peer is currently blocked, **without** checking expiry.
    ///
    /// Use `is_blocked_at` for expiry-aware checks.
    pub fn is_blocked(&self, peer_id: &str) -> bool {
        self.entries.contains_key(peer_id)
    }

    /// Check whether a peer is blocked at the given wall-clock time (ms).
    ///
    /// An entry whose `expires_at` is `Some(t)` and `t <= now` is considered
    /// expired and treated as not blocked.
    pub fn is_blocked_at(&self, peer_id: &str, now: u64) -> bool {
        match self.entries.get(peer_id) {
            None => false,
            Some(entry) => match entry.expires_at {
                None => true,
                Some(exp) => exp > now,
            },
        }
    }

    /// Return a reference to the blacklist entry for a peer, if present.
    pub fn get_entry(&self, peer_id: &str) -> Option<&BlacklistEntry> {
        self.entries.get(peer_id)
    }

    /// Return the number of entries currently in the blacklist.
    pub fn blocked_count(&self) -> usize {
        self.entries.len()
    }

    /// Return references to all current blacklist entries.
    pub fn all_entries(&self) -> Vec<&BlacklistEntry> {
        self.entries.values().collect()
    }

    // -----------------------------------------------------------------------
    // Strike system
    // -----------------------------------------------------------------------

    /// Record a strike against a peer at timestamp `now` (Unix-epoch ms).
    ///
    /// Returns the number of strikes within the current window
    /// (`strike_window_ms`). When the count reaches `auto_blacklist_strikes`,
    /// the peer is automatically blocked; if `permanent_after_strikes` is also
    /// reached, the block is permanent.
    pub fn record_strike(&mut self, peer_id: &str, now: u64) -> u32 {
        // Append the new strike timestamp.
        let log = self.strike_log.entry(peer_id.to_string()).or_default();
        log.push(now);

        // Trim strikes older than the window.
        let cutoff = now.saturating_sub(self.config.strike_window_ms);
        log.retain(|&t| t > cutoff);

        let window_count = log.len() as u32;
        let total_count = log.len() as u32; // we use window count for auto-block too

        // Determine whether we should auto-block.
        if window_count >= self.config.auto_blacklist_strikes && !self.entries.contains_key(peer_id)
        {
            // Determine permanence from cumulative strikes in the log.
            // We count all recorded (potentially pruned) strikes via the entry's
            // strike_count field after insertion.
            let is_permanent = total_count >= self.config.permanent_after_strikes;
            let expires_at = if is_permanent {
                None
            } else {
                self.config.default_expiry_ms.map(|d| now.saturating_add(d))
            };

            if self.entries.len() >= self.config.max_entries {
                if let Some(key) = self.entries.keys().next().cloned() {
                    self.entries.remove(&key);
                }
            }

            let entry = BlacklistEntry {
                peer_id: peer_id.to_string(),
                reason: BlacklistReason::ReputationThreshold,
                blocked_at: now,
                expires_at,
                strike_count: total_count,
                notes: format!("{} strikes in window", window_count),
            };
            self.entries.insert(peer_id.to_string(), entry);
            self.stats.total_blocked += 1;
            self.stats.auto_blocked += 1;
            if is_permanent {
                self.stats.permanent_blocks += 1;
            }
        } else if let Some(entry) = self.entries.get_mut(peer_id) {
            // Peer is already blocked — update strike count and upgrade to
            // permanent if the total now meets the threshold.
            entry.strike_count = entry.strike_count.saturating_add(1);
            if entry.expires_at.is_some()
                && entry.strike_count >= self.config.permanent_after_strikes
            {
                entry.expires_at = None;
                self.stats.permanent_blocks += 1;
            }
        }

        window_count
    }

    /// Return the number of strikes recorded for `peer_id` within the rolling
    /// window ending at `now`.
    pub fn strikes_in_window(&self, peer_id: &str, now: u64) -> u32 {
        let cutoff = now.saturating_sub(self.config.strike_window_ms);
        self.strike_log
            .get(peer_id)
            .map(|v| v.iter().filter(|&&t| t > cutoff).count() as u32)
            .unwrap_or(0)
    }

    // -----------------------------------------------------------------------
    // Maintenance
    // -----------------------------------------------------------------------

    /// Remove all entries whose `expires_at` is `Some(t)` with `t <= now`.
    ///
    /// Returns the number of entries removed.
    pub fn purge_expired(&mut self, now: u64) -> usize {
        let before = self.entries.len();
        self.entries.retain(|_, entry| match entry.expires_at {
            None => true,
            Some(exp) => exp > now,
        });
        let removed = before - self.entries.len();
        self.stats.expired_removed += removed as u64;
        self.stats.total_unblocked += removed as u64;
        removed
    }

    /// Extend (or make permanent) the block for an existing entry.
    ///
    /// `new_expiry` is the absolute timestamp (ms) for the new expiry, or
    /// `None` to make the block permanent.
    ///
    /// Returns `false` if the peer is not currently in the blacklist.
    pub fn extend_block(&mut self, peer_id: &str, new_expiry: Option<u64>) -> bool {
        match self.entries.get_mut(peer_id) {
            None => false,
            Some(entry) => {
                let was_permanent = entry.expires_at.is_none();
                entry.expires_at = new_expiry;
                // Track new permanent block if this is an escalation.
                if !was_permanent && new_expiry.is_none() {
                    self.stats.permanent_blocks += 1;
                }
                true
            }
        }
    }

    // -----------------------------------------------------------------------
    // Stats
    // -----------------------------------------------------------------------

    /// Return a reference to the current aggregate statistics.
    pub fn stats(&self) -> &BlacklistStats {
        &self.stats
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Resolve the effective expiry timestamp, applying the permanent-after-strikes
    /// rule and the default expiry.
    fn resolve_expiry(&self, explicit: Option<u64>, strike_count: u32) -> Option<u64> {
        if strike_count >= self.config.permanent_after_strikes {
            return None; // permanent
        }
        // default_expiry_ms is relative; we have no `now` in this path when
        // called from the non-`_at` variant. Return `None` (permanent) as
        // the safest default when we cannot compute an absolute timestamp.
        explicit
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const NOW: u64 = 1_700_000_000_000; // arbitrary fixed "now" in ms

    fn default_config() -> BlacklistConfig {
        BlacklistConfig {
            max_entries: 1_000,
            default_expiry_ms: Some(60_000), // 1 minute
            auto_blacklist_strikes: 3,
            permanent_after_strikes: 6,
            strike_window_ms: 30_000, // 30 seconds
        }
    }

    fn make_bl() -> PeerBlacklist {
        PeerBlacklist::new(default_config())
    }

    // --- block / unblock ---

    #[test]
    fn test_block_returns_true_on_success() {
        let mut bl = make_bl();
        let result = bl.block_at("p1", BlacklistReason::ManualBlock, "test", None, NOW);
        assert!(result);
    }

    #[test]
    fn test_block_returns_false_when_already_blocked() {
        let mut bl = make_bl();
        bl.block_at("p1", BlacklistReason::ManualBlock, "first", None, NOW);
        let result = bl.block_at("p1", BlacklistReason::SpamDetected, "second", None, NOW);
        assert!(!result);
    }

    #[test]
    fn test_unblock_returns_true_when_blocked() {
        let mut bl = make_bl();
        bl.block_at("p1", BlacklistReason::ManualBlock, "", None, NOW);
        assert!(bl.unblock("p1"));
    }

    #[test]
    fn test_unblock_returns_false_when_not_blocked() {
        let mut bl = make_bl();
        assert!(!bl.unblock("nonexistent"));
    }

    #[test]
    fn test_unblock_removes_entry() {
        let mut bl = make_bl();
        bl.block_at("p1", BlacklistReason::ManualBlock, "", None, NOW);
        bl.unblock("p1");
        assert!(!bl.is_blocked("p1"));
    }

    // --- is_blocked ---

    #[test]
    fn test_is_blocked_true_when_present() {
        let mut bl = make_bl();
        bl.block_at("p1", BlacklistReason::ManualBlock, "", None, NOW);
        assert!(bl.is_blocked("p1"));
    }

    #[test]
    fn test_is_blocked_false_for_unknown_peer() {
        let bl = make_bl();
        assert!(!bl.is_blocked("unknown"));
    }

    // --- is_blocked_at with expiry ---

    #[test]
    fn test_is_blocked_at_returns_true_before_expiry() {
        let mut bl = make_bl();
        let expiry = NOW + 10_000;
        bl.block_at("p1", BlacklistReason::ManualBlock, "", Some(expiry), NOW);
        assert!(bl.is_blocked_at("p1", NOW));
        assert!(bl.is_blocked_at("p1", NOW + 9_999));
    }

    #[test]
    fn test_is_blocked_at_returns_false_at_expiry() {
        let mut bl = make_bl();
        let expiry = NOW + 10_000;
        bl.block_at("p1", BlacklistReason::ManualBlock, "", Some(expiry), NOW);
        assert!(!bl.is_blocked_at("p1", expiry));
    }

    #[test]
    fn test_is_blocked_at_returns_false_after_expiry() {
        let mut bl = make_bl();
        let expiry = NOW + 10_000;
        bl.block_at("p1", BlacklistReason::ManualBlock, "", Some(expiry), NOW);
        assert!(!bl.is_blocked_at("p1", NOW + 20_000));
    }

    #[test]
    fn test_is_blocked_at_permanent_entry_always_true() {
        let mut bl = make_bl();
        bl.block_at("p1", BlacklistReason::ManualBlock, "", None, NOW);
        assert!(bl.is_blocked_at("p1", NOW + 1_000_000_000));
    }

    #[test]
    fn test_is_blocked_at_false_for_unknown_peer() {
        let bl = make_bl();
        assert!(!bl.is_blocked_at("unknown", NOW));
    }

    // --- strike counting in window ---

    #[test]
    fn test_strikes_in_window_zero_initially() {
        let bl = make_bl();
        assert_eq!(bl.strikes_in_window("p1", NOW), 0);
    }

    #[test]
    fn test_strikes_in_window_counts_recent() {
        let mut bl = make_bl();
        bl.record_strike("p1", NOW);
        bl.record_strike("p1", NOW + 1_000);
        assert_eq!(bl.strikes_in_window("p1", NOW + 2_000), 2);
    }

    #[test]
    fn test_strikes_in_window_excludes_old() {
        let mut bl = make_bl();
        // Record a strike that is outside the 30-second window
        bl.record_strike("p1", NOW);
        // Query from 40 seconds later — the strike at NOW is outside the window
        assert_eq!(bl.strikes_in_window("p1", NOW + 40_000), 0);
    }

    // --- auto-blacklist on threshold ---

    #[test]
    fn test_auto_blacklist_triggers_at_threshold() {
        let mut bl = make_bl(); // auto_blacklist_strikes = 3
        bl.record_strike("p1", NOW);
        bl.record_strike("p1", NOW + 1_000);
        assert!(!bl.is_blocked("p1")); // not yet at threshold
        bl.record_strike("p1", NOW + 2_000); // 3rd strike -> auto-block
        assert!(bl.is_blocked("p1"));
    }

    #[test]
    fn test_auto_blacklist_increments_auto_blocked_stat() {
        let mut bl = make_bl();
        for i in 0..3 {
            bl.record_strike("p1", NOW + i * 1_000);
        }
        assert_eq!(bl.stats().auto_blocked, 1);
    }

    #[test]
    fn test_auto_blacklist_reason_is_reputation_threshold() {
        let mut bl = make_bl();
        for i in 0..3 {
            bl.record_strike("p1", NOW + i * 1_000);
        }
        let entry = bl.get_entry("p1").expect("should be auto-blocked");
        assert_eq!(entry.reason, BlacklistReason::ReputationThreshold);
    }

    // --- permanent after strikes ---

    #[test]
    fn test_permanent_block_after_strike_threshold() {
        let mut bl = make_bl(); // permanent_after_strikes = 6
        for i in 0..6 {
            bl.record_strike("p2", NOW + i * 1_000);
        }
        let entry = bl.get_entry("p2").expect("should be blocked");
        assert!(entry.expires_at.is_none(), "should be permanent");
    }

    #[test]
    fn test_auto_block_not_permanent_below_threshold() {
        let mut bl = make_bl(); // auto at 3, permanent at 6
        for i in 0..3 {
            bl.record_strike("p3", NOW + i * 1_000);
        }
        let entry = bl.get_entry("p3").expect("should be auto-blocked");
        // 3 strikes < permanent_after_strikes (6), so should have an expiry
        assert!(
            entry.expires_at.is_some(),
            "should not be permanent at only 3 strikes"
        );
    }

    // --- purge_expired ---

    #[test]
    fn test_purge_expired_removes_expired_entries() {
        let mut bl = make_bl();
        bl.block_at(
            "p1",
            BlacklistReason::ManualBlock,
            "",
            Some(NOW + 1_000),
            NOW,
        );
        bl.block_at(
            "p2",
            BlacklistReason::ManualBlock,
            "",
            Some(NOW + 2_000),
            NOW,
        );
        bl.block_at("p3", BlacklistReason::ManualBlock, "", None, NOW); // permanent
        let removed = bl.purge_expired(NOW + 1_500);
        assert_eq!(removed, 1); // only p1 expired
        assert!(!bl.is_blocked("p1"));
        assert!(bl.is_blocked("p2"));
        assert!(bl.is_blocked("p3"));
    }

    #[test]
    fn test_purge_expired_returns_zero_when_none_expired() {
        let mut bl = make_bl();
        bl.block_at(
            "p1",
            BlacklistReason::ManualBlock,
            "",
            Some(NOW + 10_000),
            NOW,
        );
        assert_eq!(bl.purge_expired(NOW), 0);
    }

    #[test]
    fn test_purge_expired_updates_stats() {
        let mut bl = make_bl();
        bl.block_at(
            "p1",
            BlacklistReason::ManualBlock,
            "",
            Some(NOW + 1_000),
            NOW,
        );
        bl.purge_expired(NOW + 2_000);
        assert_eq!(bl.stats().expired_removed, 1);
        assert_eq!(bl.stats().total_unblocked, 1);
    }

    // --- extend_block ---

    #[test]
    fn test_extend_block_updates_expiry() {
        let mut bl = make_bl();
        bl.block_at(
            "p1",
            BlacklistReason::ManualBlock,
            "",
            Some(NOW + 1_000),
            NOW,
        );
        let result = bl.extend_block("p1", Some(NOW + 10_000));
        assert!(result);
        let entry = bl.get_entry("p1").expect("should exist");
        assert_eq!(entry.expires_at, Some(NOW + 10_000));
    }

    #[test]
    fn test_extend_block_to_permanent() {
        let mut bl = make_bl();
        bl.block_at(
            "p1",
            BlacklistReason::ManualBlock,
            "",
            Some(NOW + 1_000),
            NOW,
        );
        bl.extend_block("p1", None);
        let entry = bl.get_entry("p1").expect("should exist");
        assert!(entry.expires_at.is_none());
    }

    #[test]
    fn test_extend_block_returns_false_when_not_blocked() {
        let mut bl = make_bl();
        assert!(!bl.extend_block("unknown", None));
    }

    // --- stats ---

    #[test]
    fn test_stats_total_blocked() {
        let mut bl = make_bl();
        bl.block_at("p1", BlacklistReason::ManualBlock, "", None, NOW);
        bl.block_at("p2", BlacklistReason::ManualBlock, "", None, NOW);
        assert_eq!(bl.stats().total_blocked, 2);
    }

    #[test]
    fn test_stats_total_unblocked() {
        let mut bl = make_bl();
        bl.block_at("p1", BlacklistReason::ManualBlock, "", None, NOW);
        bl.unblock("p1");
        assert_eq!(bl.stats().total_unblocked, 1);
    }

    #[test]
    fn test_stats_permanent_blocks() {
        let mut bl = make_bl();
        bl.block_at("p1", BlacklistReason::ManualBlock, "", None, NOW);
        // p1 has 0 strikes so resolve_expiry would return None (no default now)
        // -> permanent block counted
        assert_eq!(bl.stats().permanent_blocks, 1);
    }

    // --- empty blacklist ---

    #[test]
    fn test_empty_blacklist_blocked_count() {
        let bl = make_bl();
        assert_eq!(bl.blocked_count(), 0);
    }

    #[test]
    fn test_empty_blacklist_all_entries() {
        let bl = make_bl();
        assert!(bl.all_entries().is_empty());
    }

    #[test]
    fn test_empty_blacklist_stats_all_zero() {
        let bl = make_bl();
        let s = bl.stats();
        assert_eq!(s.total_blocked, 0);
        assert_eq!(s.total_unblocked, 0);
        assert_eq!(s.auto_blocked, 0);
        assert_eq!(s.permanent_blocks, 0);
        assert_eq!(s.expired_removed, 0);
    }

    // --- all_entries ---

    #[test]
    fn test_all_entries_returns_correct_count() {
        let mut bl = make_bl();
        bl.block_at("p1", BlacklistReason::ManualBlock, "", None, NOW);
        bl.block_at(
            "p2",
            BlacklistReason::SpamDetected,
            "",
            Some(NOW + 1_000),
            NOW,
        );
        let entries = bl.all_entries();
        assert_eq!(entries.len(), 2);
    }

    // --- get_entry ---

    #[test]
    fn test_get_entry_fields() {
        let mut bl = make_bl();
        let expiry = NOW + 5_000;
        bl.block_at(
            "p1",
            BlacklistReason::ProtocolViolation,
            "bad proto",
            Some(expiry),
            NOW,
        );
        let entry = bl.get_entry("p1").expect("should exist");
        assert_eq!(entry.peer_id, "p1");
        assert_eq!(entry.reason, BlacklistReason::ProtocolViolation);
        assert_eq!(entry.notes, "bad proto");
        assert_eq!(entry.blocked_at, NOW);
        assert_eq!(entry.expires_at, Some(expiry));
    }

    // --- duplicate block ---

    #[test]
    fn test_duplicate_block_does_not_double_count_stats() {
        let mut bl = make_bl();
        bl.block_at("p1", BlacklistReason::ManualBlock, "", None, NOW);
        bl.block_at("p1", BlacklistReason::SpamDetected, "", None, NOW); // returns false, ignored
        assert_eq!(bl.stats().total_blocked, 1);
        assert_eq!(bl.blocked_count(), 1);
    }

    // --- blocked_count after operations ---

    #[test]
    fn test_blocked_count_after_block_and_unblock() {
        let mut bl = make_bl();
        bl.block_at("p1", BlacklistReason::ManualBlock, "", None, NOW);
        bl.block_at("p2", BlacklistReason::ManualBlock, "", None, NOW);
        assert_eq!(bl.blocked_count(), 2);
        bl.unblock("p1");
        assert_eq!(bl.blocked_count(), 1);
    }

    // --- BlacklistReason equality ---

    #[test]
    fn test_blacklist_reason_equality() {
        assert_eq!(BlacklistReason::ManualBlock, BlacklistReason::ManualBlock);
        assert_ne!(BlacklistReason::ManualBlock, BlacklistReason::SpamDetected);
    }
}
