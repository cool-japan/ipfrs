//! Peer ban list for temporary and permanent banning of misbehaving peers.
//!
//! Provides a tick-based ban management system that supports both temporary bans
//! (which expire after a configurable TTL) and permanent bans (which persist until
//! explicitly removed). The ban list enforces a maximum capacity and tracks
//! statistics about ban/unban operations.
//!
//! # Example
//!
//! ```rust
//! use ipfrs_network::ban_list::{PeerBanList, BanConfig};
//!
//! let config = BanConfig::default();
//! let mut ban_list = PeerBanList::new(config);
//!
//! // Temporarily ban a peer with default TTL
//! ban_list.ban_temporary("peer-1", "spamming", None);
//! assert!(ban_list.is_banned("peer-1"));
//!
//! // Permanently ban a peer
//! ban_list.ban_permanent("peer-2", "protocol violation");
//! assert!(ban_list.is_banned("peer-2"));
//!
//! // Unban a peer
//! assert!(ban_list.unban("peer-1"));
//! assert!(!ban_list.is_banned("peer-1"));
//! ```

use std::collections::HashMap;

/// The kind of ban applied to a peer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BanKind {
    /// Temporary ban that expires after a TTL measured in ticks.
    Temporary,
    /// Permanent ban that never expires unless explicitly removed.
    Permanent,
}

/// An entry recording a ban against a specific peer.
#[derive(Debug, Clone)]
pub struct BanEntry {
    /// The identifier of the banned peer.
    pub peer_id: String,
    /// Whether the ban is temporary or permanent.
    pub kind: BanKind,
    /// Human-readable reason for the ban.
    pub reason: String,
    /// The tick at which the ban was issued.
    pub banned_tick: u64,
    /// The tick at which a temporary ban expires. `None` for permanent bans.
    pub expires_tick: Option<u64>,
}

/// Configuration for the peer ban list.
#[derive(Debug, Clone)]
pub struct BanConfig {
    /// Default TTL in ticks for temporary bans when no explicit TTL is provided.
    pub default_temp_ttl_ticks: u64,
    /// Maximum number of simultaneous bans allowed.
    pub max_bans: usize,
}

impl Default for BanConfig {
    fn default() -> Self {
        Self {
            default_temp_ttl_ticks: 500,
            max_bans: 10_000,
        }
    }
}

/// Aggregate statistics about the ban list.
#[derive(Debug, Clone)]
pub struct BanListStats {
    /// Number of currently active bans.
    pub active_bans: usize,
    /// Number of currently active permanent bans.
    pub permanent_bans: usize,
    /// Number of currently active temporary bans.
    pub temporary_bans: usize,
    /// Total number of bans ever issued.
    pub total_bans_issued: u64,
    /// Total number of unbans (manual or via expiry cleanup).
    pub total_unbans: u64,
}

/// Manages a list of banned peers with temporary and permanent ban support.
///
/// Bans are tracked using a logical tick counter rather than wall-clock time,
/// allowing deterministic testing and integration with tick-based event loops.
#[derive(Debug, Clone)]
pub struct PeerBanList {
    config: BanConfig,
    bans: HashMap<String, BanEntry>,
    current_tick: u64,
    total_bans_issued: u64,
    total_unbans: u64,
}

impl PeerBanList {
    /// Create a new `PeerBanList` with the given configuration.
    pub fn new(config: BanConfig) -> Self {
        Self {
            config,
            bans: HashMap::new(),
            current_tick: 0,
            total_bans_issued: 0,
            total_unbans: 0,
        }
    }

    /// Ban a peer temporarily. Uses `default_temp_ttl_ticks` when `ttl_ticks` is `None`.
    ///
    /// If the ban list is at capacity, the ban is silently dropped.
    /// If the peer is already banned, the entry is updated (re-banned).
    pub fn ban_temporary(&mut self, peer_id: &str, reason: &str, ttl_ticks: Option<u64>) {
        let ttl = ttl_ticks.unwrap_or(self.config.default_temp_ttl_ticks);
        let expires = self.current_tick.saturating_add(ttl);

        // If the peer is already banned, we update (re-ban) without checking max_bans again.
        let is_update = self.bans.contains_key(peer_id);
        if !is_update && self.bans.len() >= self.config.max_bans {
            return;
        }

        let entry = BanEntry {
            peer_id: peer_id.to_string(),
            kind: BanKind::Temporary,
            reason: reason.to_string(),
            banned_tick: self.current_tick,
            expires_tick: Some(expires),
        };
        self.bans.insert(peer_id.to_string(), entry);
        self.total_bans_issued += 1;
    }

    /// Ban a peer permanently. The ban will not expire until explicitly removed via `unban`.
    ///
    /// If the ban list is at capacity, the ban is silently dropped.
    /// If the peer is already banned, the entry is updated (re-banned).
    pub fn ban_permanent(&mut self, peer_id: &str, reason: &str) {
        let is_update = self.bans.contains_key(peer_id);
        if !is_update && self.bans.len() >= self.config.max_bans {
            return;
        }

        let entry = BanEntry {
            peer_id: peer_id.to_string(),
            kind: BanKind::Permanent,
            reason: reason.to_string(),
            banned_tick: self.current_tick,
            expires_tick: None,
        };
        self.bans.insert(peer_id.to_string(), entry);
        self.total_bans_issued += 1;
    }

    /// Remove a ban for the given peer. Returns `true` if the peer was banned.
    pub fn unban(&mut self, peer_id: &str) -> bool {
        if self.bans.remove(peer_id).is_some() {
            self.total_unbans += 1;
            true
        } else {
            false
        }
    }

    /// Check whether a peer is currently banned, respecting temporary ban expiry.
    ///
    /// A temporary ban is considered expired if `current_tick >= expires_tick`.
    /// Expired entries are lazily removed on access.
    pub fn is_banned(&mut self, peer_id: &str) -> bool {
        if let Some(entry) = self.bans.get(peer_id) {
            match entry.kind {
                BanKind::Permanent => true,
                BanKind::Temporary => {
                    if let Some(expires) = entry.expires_tick {
                        if self.current_tick >= expires {
                            // Expired: lazily remove
                            self.bans.remove(peer_id);
                            self.total_unbans += 1;
                            false
                        } else {
                            true
                        }
                    } else {
                        // Temporary with no expiry treated as expired (defensive)
                        true
                    }
                }
            }
        } else {
            false
        }
    }

    /// Get the ban entry for a peer, if it exists and is still active.
    ///
    /// Note: This does not perform lazy expiry cleanup. Use `is_banned` for
    /// an expiry-aware check, or call `tick_cleanup` periodically.
    pub fn get_ban(&self, peer_id: &str) -> Option<&BanEntry> {
        self.bans.get(peer_id)
    }

    /// Advance the current tick by one and remove all expired temporary bans.
    pub fn tick_cleanup(&mut self) {
        self.current_tick += 1;
        let current = self.current_tick;
        let before = self.bans.len();
        self.bans.retain(|_, entry| {
            if entry.kind == BanKind::Temporary {
                if let Some(expires) = entry.expires_tick {
                    return current < expires;
                }
            }
            true
        });
        let removed = before - self.bans.len();
        self.total_unbans += removed as u64;
    }

    /// Return the number of currently active bans (including not-yet-cleaned expired ones).
    pub fn banned_count(&self) -> usize {
        self.bans.len()
    }

    /// Return references to all current ban entries.
    pub fn list_banned(&self) -> Vec<&BanEntry> {
        self.bans.values().collect()
    }

    /// Return aggregate statistics about the ban list.
    pub fn stats(&self) -> BanListStats {
        let mut permanent = 0usize;
        let mut temporary = 0usize;
        for entry in self.bans.values() {
            match entry.kind {
                BanKind::Permanent => permanent += 1,
                BanKind::Temporary => temporary += 1,
            }
        }
        BanListStats {
            active_bans: self.bans.len(),
            permanent_bans: permanent,
            temporary_bans: temporary,
            total_bans_issued: self.total_bans_issued,
            total_unbans: self.total_unbans,
        }
    }

    /// Return the current logical tick.
    pub fn current_tick(&self) -> u64 {
        self.current_tick
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_list() -> PeerBanList {
        PeerBanList::new(BanConfig::default())
    }

    // --- Basic temporary ban ---

    #[test]
    fn test_temp_ban_is_banned() {
        let mut list = default_list();
        list.ban_temporary("peer-1", "spam", None);
        assert!(list.is_banned("peer-1"));
    }

    #[test]
    fn test_temp_ban_expires_after_ttl() {
        let mut list = default_list();
        list.ban_temporary("peer-1", "spam", Some(3));
        assert!(list.is_banned("peer-1"));
        // Advance 3 ticks (tick goes 1, 2, 3)
        list.tick_cleanup(); // tick=1
        list.tick_cleanup(); // tick=2
        assert!(list.is_banned("peer-1")); // expires_tick=3, current=2 => still banned
        list.tick_cleanup(); // tick=3
                             // Now current_tick == expires_tick == 3 => expired
        assert!(!list.is_banned("peer-1"));
    }

    #[test]
    fn test_temp_ban_with_default_ttl() {
        let mut list = default_list();
        list.ban_temporary("peer-1", "reason", None);
        let entry = list.get_ban("peer-1");
        assert!(entry.is_some());
        let entry = entry.expect("already checked");
        assert_eq!(entry.expires_tick, Some(500)); // default TTL
    }

    #[test]
    fn test_temp_ban_with_custom_ttl() {
        let mut list = default_list();
        list.ban_temporary("peer-1", "reason", Some(100));
        let entry = list.get_ban("peer-1").expect("should exist");
        assert_eq!(entry.expires_tick, Some(100));
        assert_eq!(entry.kind, BanKind::Temporary);
    }

    // --- Permanent ban ---

    #[test]
    fn test_permanent_ban_persists() {
        let mut list = default_list();
        list.ban_permanent("peer-1", "protocol violation");
        // Advance many ticks
        for _ in 0..1000 {
            list.tick_cleanup();
        }
        assert!(list.is_banned("peer-1"));
    }

    #[test]
    fn test_permanent_ban_has_no_expiry() {
        let mut list = default_list();
        list.ban_permanent("peer-1", "bad actor");
        let entry = list.get_ban("peer-1").expect("should exist");
        assert_eq!(entry.kind, BanKind::Permanent);
        assert!(entry.expires_tick.is_none());
    }

    // --- Unban ---

    #[test]
    fn test_unban_removes_temp_ban() {
        let mut list = default_list();
        list.ban_temporary("peer-1", "spam", None);
        assert!(list.unban("peer-1"));
        assert!(!list.is_banned("peer-1"));
    }

    #[test]
    fn test_unban_removes_permanent_ban() {
        let mut list = default_list();
        list.ban_permanent("peer-1", "bad");
        assert!(list.unban("peer-1"));
        assert!(!list.is_banned("peer-1"));
    }

    #[test]
    fn test_unban_nonexistent_returns_false() {
        let mut list = default_list();
        assert!(!list.unban("peer-nonexistent"));
    }

    // --- is_banned with expiry ---

    #[test]
    fn test_is_banned_lazily_removes_expired() {
        let mut list = default_list();
        list.ban_temporary("peer-1", "spam", Some(1));
        list.tick_cleanup(); // tick=1, expires_tick=1 => expired
                             // is_banned should return false and lazily remove
        assert!(!list.is_banned("peer-1"));
        assert_eq!(list.banned_count(), 0);
    }

    #[test]
    fn test_is_banned_false_for_unknown_peer() {
        let mut list = default_list();
        assert!(!list.is_banned("unknown"));
    }

    // --- tick_cleanup ---

    #[test]
    fn test_tick_cleanup_removes_expired_bans() {
        let mut list = default_list();
        list.ban_temporary("peer-1", "a", Some(2));
        list.ban_temporary("peer-2", "b", Some(5));
        list.ban_permanent("peer-3", "c");

        // Advance to tick 2 => peer-1 expires
        list.tick_cleanup(); // tick=1
        list.tick_cleanup(); // tick=2
        assert_eq!(list.banned_count(), 2); // peer-2, peer-3 remain
        assert!(list.get_ban("peer-1").is_none());
    }

    #[test]
    fn test_tick_cleanup_preserves_permanent_bans() {
        let mut list = default_list();
        list.ban_permanent("peer-1", "forever");
        for _ in 0..100 {
            list.tick_cleanup();
        }
        assert_eq!(list.banned_count(), 1);
    }

    #[test]
    fn test_tick_cleanup_advances_tick() {
        let mut list = default_list();
        assert_eq!(list.current_tick(), 0);
        list.tick_cleanup();
        assert_eq!(list.current_tick(), 1);
        list.tick_cleanup();
        assert_eq!(list.current_tick(), 2);
    }

    // --- max_bans enforcement ---

    #[test]
    fn test_max_bans_enforcement_temp() {
        let config = BanConfig {
            default_temp_ttl_ticks: 500,
            max_bans: 3,
        };
        let mut list = PeerBanList::new(config);
        list.ban_temporary("p1", "a", None);
        list.ban_temporary("p2", "b", None);
        list.ban_temporary("p3", "c", None);
        // This should be silently dropped
        list.ban_temporary("p4", "d", None);
        assert_eq!(list.banned_count(), 3);
        assert!(!list.is_banned("p4"));
    }

    #[test]
    fn test_max_bans_enforcement_permanent() {
        let config = BanConfig {
            default_temp_ttl_ticks: 500,
            max_bans: 2,
        };
        let mut list = PeerBanList::new(config);
        list.ban_permanent("p1", "a");
        list.ban_permanent("p2", "b");
        list.ban_permanent("p3", "c"); // dropped
        assert_eq!(list.banned_count(), 2);
        assert!(!list.is_banned("p3"));
    }

    #[test]
    fn test_max_bans_allows_reban_existing() {
        let config = BanConfig {
            default_temp_ttl_ticks: 500,
            max_bans: 2,
        };
        let mut list = PeerBanList::new(config);
        list.ban_temporary("p1", "a", None);
        list.ban_temporary("p2", "b", None);
        // Re-banning an existing peer should work even at capacity
        list.ban_temporary("p1", "updated reason", Some(100));
        assert_eq!(list.banned_count(), 2);
        let entry = list.get_ban("p1").expect("should exist");
        assert_eq!(entry.reason, "updated reason");
    }

    // --- Re-banning updates entry ---

    #[test]
    fn test_reban_updates_entry_temp_to_permanent() {
        let mut list = default_list();
        list.ban_temporary("peer-1", "spam", Some(10));
        list.ban_permanent("peer-1", "escalated");
        let entry = list.get_ban("peer-1").expect("should exist");
        assert_eq!(entry.kind, BanKind::Permanent);
        assert_eq!(entry.reason, "escalated");
    }

    #[test]
    fn test_reban_updates_entry_permanent_to_temp() {
        let mut list = default_list();
        list.ban_permanent("peer-1", "permanent");
        list.ban_temporary("peer-1", "downgraded", Some(50));
        let entry = list.get_ban("peer-1").expect("should exist");
        assert_eq!(entry.kind, BanKind::Temporary);
        assert_eq!(entry.reason, "downgraded");
    }

    #[test]
    fn test_reban_increments_total_bans() {
        let mut list = default_list();
        list.ban_temporary("peer-1", "first", None);
        list.ban_temporary("peer-1", "second", None);
        let stats = list.stats();
        assert_eq!(stats.total_bans_issued, 2);
        assert_eq!(stats.active_bans, 1);
    }

    // --- Stats ---

    #[test]
    fn test_stats_accuracy() {
        let mut list = default_list();
        list.ban_temporary("p1", "a", None);
        list.ban_temporary("p2", "b", None);
        list.ban_permanent("p3", "c");
        list.unban("p1");

        let stats = list.stats();
        assert_eq!(stats.active_bans, 2);
        assert_eq!(stats.temporary_bans, 1);
        assert_eq!(stats.permanent_bans, 1);
        assert_eq!(stats.total_bans_issued, 3);
        assert_eq!(stats.total_unbans, 1);
    }

    #[test]
    fn test_stats_after_tick_cleanup() {
        let mut list = default_list();
        list.ban_temporary("p1", "a", Some(1));
        list.ban_temporary("p2", "b", Some(3));
        list.ban_permanent("p3", "c");
        list.tick_cleanup(); // tick=1 => p1 expires
        let stats = list.stats();
        assert_eq!(stats.active_bans, 2);
        assert_eq!(stats.temporary_bans, 1);
        assert_eq!(stats.permanent_bans, 1);
        assert_eq!(stats.total_unbans, 1);
    }

    #[test]
    fn test_stats_empty_list() {
        let list = default_list();
        let stats = list.stats();
        assert_eq!(stats.active_bans, 0);
        assert_eq!(stats.permanent_bans, 0);
        assert_eq!(stats.temporary_bans, 0);
        assert_eq!(stats.total_bans_issued, 0);
        assert_eq!(stats.total_unbans, 0);
    }

    // --- list_banned ---

    #[test]
    fn test_list_banned_returns_all_entries() {
        let mut list = default_list();
        list.ban_temporary("p1", "a", None);
        list.ban_permanent("p2", "b");
        let entries = list.list_banned();
        assert_eq!(entries.len(), 2);
    }

    #[test]
    fn test_list_banned_empty() {
        let list = default_list();
        let entries = list.list_banned();
        assert!(entries.is_empty());
    }

    // --- banned_count ---

    #[test]
    fn test_banned_count() {
        let mut list = default_list();
        assert_eq!(list.banned_count(), 0);
        list.ban_temporary("p1", "a", None);
        assert_eq!(list.banned_count(), 1);
        list.ban_permanent("p2", "b");
        assert_eq!(list.banned_count(), 2);
        list.unban("p1");
        assert_eq!(list.banned_count(), 1);
    }

    // --- Edge cases ---

    #[test]
    fn test_ban_empty_peer_id() {
        let mut list = default_list();
        list.ban_temporary("", "empty id", None);
        assert!(list.is_banned(""));
    }

    #[test]
    fn test_ban_empty_reason() {
        let mut list = default_list();
        list.ban_temporary("p1", "", None);
        let entry = list.get_ban("p1").expect("should exist");
        assert_eq!(entry.reason, "");
    }

    #[test]
    fn test_zero_ttl_expires_immediately_on_cleanup() {
        let mut list = default_list();
        list.ban_temporary("p1", "zero", Some(0));
        // expires_tick = current_tick + 0 = 0, current_tick = 0 => expired
        assert!(!list.is_banned("p1"));
    }

    #[test]
    fn test_large_ttl_does_not_overflow() {
        let mut list = default_list();
        list.ban_temporary("p1", "big", Some(u64::MAX));
        let entry = list.get_ban("p1").expect("should exist");
        // saturating_add should cap at u64::MAX
        assert_eq!(entry.expires_tick, Some(u64::MAX));
    }

    #[test]
    fn test_multiple_bans_and_unbans_stats() {
        let mut list = default_list();
        for i in 0..10 {
            list.ban_temporary(&format!("p{}", i), "test", Some(5));
        }
        for i in 0..5 {
            list.unban(&format!("p{}", i));
        }
        let stats = list.stats();
        assert_eq!(stats.total_bans_issued, 10);
        assert_eq!(stats.total_unbans, 5);
        assert_eq!(stats.active_bans, 5);
    }

    #[test]
    fn test_tick_cleanup_counts_as_unban() {
        let mut list = default_list();
        list.ban_temporary("p1", "a", Some(1));
        list.ban_temporary("p2", "b", Some(1));
        list.tick_cleanup(); // both expire at tick=1
        let stats = list.stats();
        assert_eq!(stats.total_unbans, 2);
        assert_eq!(stats.active_bans, 0);
    }

    #[test]
    fn test_get_ban_returns_none_for_unknown() {
        let list = default_list();
        assert!(list.get_ban("unknown").is_none());
    }

    #[test]
    fn test_default_config() {
        let config = BanConfig::default();
        assert_eq!(config.default_temp_ttl_ticks, 500);
        assert_eq!(config.max_bans, 10_000);
    }

    #[test]
    fn test_ban_kind_equality() {
        assert_eq!(BanKind::Temporary, BanKind::Temporary);
        assert_eq!(BanKind::Permanent, BanKind::Permanent);
        assert_ne!(BanKind::Temporary, BanKind::Permanent);
    }
}
