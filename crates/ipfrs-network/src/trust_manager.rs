//! Peer trust management for selective content acceptance based on trust hierarchies.
//!
//! This module provides [`PeerTrustManager`] which tracks trust levels and attestations
//! for peers in the network.

use std::collections::HashMap;

/// Trust level assigned to a peer, from lowest to highest.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum TrustLevel {
    /// No trust; default for unknown peers.
    Untrusted = 0,
    /// Limited trust; restricted capabilities.
    Limited = 1,
    /// Standard trusted peer.
    Trusted = 2,
    /// Vouched for by another trusted peer.
    Vouched = 3,
    /// Highest trust level; network authority.
    Authority = 4,
}

/// An attestation that one peer vouches for another.
#[derive(Clone, Debug)]
pub struct TrustAttestation {
    /// The peer that issued this attestation.
    pub attester_peer_id: String,
    /// The peer being vouched for.
    pub target_peer_id: String,
    /// The trust level being granted.
    pub granted_level: TrustLevel,
    /// The tick at which this attestation was issued.
    pub issued_at_tick: u64,
    /// The tick at which this attestation expires.
    pub expiry_tick: u64,
}

impl TrustAttestation {
    /// Returns `true` if this attestation is still valid at `current_tick`.
    pub fn is_valid(&self, current_tick: u64) -> bool {
        current_tick < self.expiry_tick
    }
}

/// A record of trust information for a single peer.
#[derive(Clone, Debug)]
pub struct TrustRecord {
    /// The peer this record belongs to.
    pub peer_id: String,
    /// The base trust level set directly on this peer.
    pub level: TrustLevel,
    /// Attestations received by this peer from others.
    pub attestations: Vec<TrustAttestation>,
}

impl TrustRecord {
    fn new(peer_id: impl Into<String>, level: TrustLevel) -> Self {
        Self {
            peer_id: peer_id.into(),
            level,
            attestations: Vec::new(),
        }
    }

    /// Returns the effective trust level considering valid attestations.
    ///
    /// Starts with the base `level` and upgrades to the highest granted level
    /// among valid (non-expired) attestations.
    pub fn effective_level(&self, current_tick: u64) -> TrustLevel {
        let mut effective = self.level;
        for attestation in &self.attestations {
            if attestation.is_valid(current_tick) && attestation.granted_level > effective {
                effective = attestation.granted_level;
            }
        }
        effective
    }
}

/// Statistics about the current state of the trust manager.
#[derive(Clone, Debug)]
pub struct TrustManagerStats {
    /// Total number of peers tracked.
    pub total_peers: usize,
    /// Count of peers at each effective trust level.
    pub by_level: HashMap<TrustLevel, usize>,
    /// Total number of attestations across all peers.
    pub total_attestations: usize,
    /// Number of expired attestations across all peers.
    pub expired_attestations: usize,
}

/// Manages trust levels and attestations for peers in the network.
pub struct PeerTrustManager {
    /// Trust records keyed by peer_id.
    pub records: HashMap<String, TrustRecord>,
}

impl PeerTrustManager {
    /// Creates a new, empty `PeerTrustManager`.
    pub fn new() -> Self {
        Self {
            records: HashMap::new(),
        }
    }

    /// Sets the base trust level for a peer. Creates the record if it does not exist.
    pub fn set_trust(&mut self, peer_id: &str, level: TrustLevel) {
        let record = self
            .records
            .entry(peer_id.to_string())
            .or_insert_with(|| TrustRecord::new(peer_id, TrustLevel::Untrusted));
        record.level = level;
    }

    /// Adds an attestation. Auto-creates the target peer record at `Untrusted` if missing.
    pub fn add_attestation(&mut self, attestation: TrustAttestation) {
        let target = attestation.target_peer_id.clone();
        let record = self
            .records
            .entry(target.clone())
            .or_insert_with(|| TrustRecord::new(&target, TrustLevel::Untrusted));
        record.attestations.push(attestation);
    }

    /// Returns the effective trust level for a peer at `current_tick`.
    ///
    /// Returns `TrustLevel::Untrusted` if the peer is not tracked.
    pub fn effective_trust(&self, peer_id: &str, current_tick: u64) -> TrustLevel {
        self.records
            .get(peer_id)
            .map(|r| r.effective_level(current_tick))
            .unwrap_or(TrustLevel::Untrusted)
    }

    /// Returns peer IDs whose effective trust is at or above `min_level`, sorted alphabetically.
    pub fn peers_at_or_above(&self, min_level: TrustLevel, current_tick: u64) -> Vec<&str> {
        let mut result: Vec<&str> = self
            .records
            .values()
            .filter(|r| r.effective_level(current_tick) >= min_level)
            .map(|r| r.peer_id.as_str())
            .collect();
        result.sort_unstable();
        result
    }

    /// Removes all attestations issued by `attester_peer_id` across all peer records.
    pub fn revoke_attestations(&mut self, attester_peer_id: &str) {
        for record in self.records.values_mut() {
            record
                .attestations
                .retain(|a| a.attester_peer_id != attester_peer_id);
        }
    }

    /// Removes a peer record. Returns `true` if the peer was present.
    pub fn remove_peer(&mut self, peer_id: &str) -> bool {
        self.records.remove(peer_id).is_some()
    }

    /// Returns aggregate statistics for the trust manager at `current_tick`.
    pub fn stats(&self, current_tick: u64) -> TrustManagerStats {
        let mut by_level: HashMap<TrustLevel, usize> = HashMap::new();
        let mut total_attestations = 0usize;
        let mut expired_attestations = 0usize;

        for record in self.records.values() {
            let eff = record.effective_level(current_tick);
            *by_level.entry(eff).or_insert(0) += 1;

            for attestation in &record.attestations {
                total_attestations += 1;
                if !attestation.is_valid(current_tick) {
                    expired_attestations += 1;
                }
            }
        }

        TrustManagerStats {
            total_peers: self.records.len(),
            by_level,
            total_attestations,
            expired_attestations,
        }
    }
}

impl Default for PeerTrustManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    fn make_attestation(
        attester: &str,
        target: &str,
        level: TrustLevel,
        issued: u64,
        expiry: u64,
    ) -> TrustAttestation {
        TrustAttestation {
            attester_peer_id: attester.to_string(),
            target_peer_id: target.to_string(),
            granted_level: level,
            issued_at_tick: issued,
            expiry_tick: expiry,
        }
    }

    // ------------------------------------------------------------------
    // TrustLevel ordering
    // ------------------------------------------------------------------

    #[test]
    fn test_trust_level_ordering() {
        assert!(TrustLevel::Untrusted < TrustLevel::Limited);
        assert!(TrustLevel::Limited < TrustLevel::Trusted);
        assert!(TrustLevel::Trusted < TrustLevel::Vouched);
        assert!(TrustLevel::Vouched < TrustLevel::Authority);
    }

    // ------------------------------------------------------------------
    // is_valid
    // ------------------------------------------------------------------

    #[test]
    fn test_is_valid_before_expiry() {
        let a = make_attestation("A", "B", TrustLevel::Trusted, 0, 100);
        assert!(a.is_valid(99));
    }

    #[test]
    fn test_is_valid_at_expiry() {
        let a = make_attestation("A", "B", TrustLevel::Trusted, 0, 100);
        assert!(!a.is_valid(100));
    }

    #[test]
    fn test_is_valid_after_expiry() {
        let a = make_attestation("A", "B", TrustLevel::Trusted, 0, 100);
        assert!(!a.is_valid(200));
    }

    // ------------------------------------------------------------------
    // PeerTrustManager::new
    // ------------------------------------------------------------------

    #[test]
    fn test_new_starts_empty() {
        let mgr = PeerTrustManager::new();
        assert!(mgr.records.is_empty());
    }

    // ------------------------------------------------------------------
    // set_trust
    // ------------------------------------------------------------------

    #[test]
    fn test_set_trust_creates_record() {
        let mut mgr = PeerTrustManager::new();
        mgr.set_trust("peer-1", TrustLevel::Trusted);
        assert!(mgr.records.contains_key("peer-1"));
        assert_eq!(mgr.records["peer-1"].level, TrustLevel::Trusted);
    }

    #[test]
    fn test_set_trust_updates_existing_record() {
        let mut mgr = PeerTrustManager::new();
        mgr.set_trust("peer-1", TrustLevel::Limited);
        mgr.set_trust("peer-1", TrustLevel::Authority);
        assert_eq!(mgr.records["peer-1"].level, TrustLevel::Authority);
    }

    // ------------------------------------------------------------------
    // effective_trust
    // ------------------------------------------------------------------

    #[test]
    fn test_effective_trust_unknown_peer_returns_untrusted() {
        let mgr = PeerTrustManager::new();
        assert_eq!(mgr.effective_trust("nobody", 0), TrustLevel::Untrusted);
    }

    #[test]
    fn test_effective_trust_base_level_no_attestations() {
        let mut mgr = PeerTrustManager::new();
        mgr.set_trust("peer-1", TrustLevel::Trusted);
        assert_eq!(mgr.effective_trust("peer-1", 50), TrustLevel::Trusted);
    }

    #[test]
    fn test_effective_trust_upgraded_by_valid_attestation() {
        let mut mgr = PeerTrustManager::new();
        mgr.set_trust("peer-1", TrustLevel::Limited);
        mgr.add_attestation(make_attestation(
            "auth",
            "peer-1",
            TrustLevel::Vouched,
            0,
            100,
        ));
        assert_eq!(mgr.effective_trust("peer-1", 50), TrustLevel::Vouched);
    }

    #[test]
    fn test_effective_trust_not_upgraded_by_expired_attestation() {
        let mut mgr = PeerTrustManager::new();
        mgr.set_trust("peer-1", TrustLevel::Limited);
        mgr.add_attestation(make_attestation(
            "auth",
            "peer-1",
            TrustLevel::Vouched,
            0,
            100,
        ));
        // After expiry the base level should be returned
        assert_eq!(mgr.effective_trust("peer-1", 100), TrustLevel::Limited);
    }

    #[test]
    fn test_effective_trust_returns_max_of_base_and_attestations() {
        let mut mgr = PeerTrustManager::new();
        mgr.set_trust("peer-1", TrustLevel::Trusted);
        // Attestation granting a lower level than base should be ignored
        mgr.add_attestation(make_attestation(
            "auth",
            "peer-1",
            TrustLevel::Limited,
            0,
            100,
        ));
        assert_eq!(mgr.effective_trust("peer-1", 50), TrustLevel::Trusted);
    }

    #[test]
    fn test_effective_trust_multiple_attestations_highest_wins() {
        let mut mgr = PeerTrustManager::new();
        mgr.set_trust("peer-1", TrustLevel::Untrusted);
        mgr.add_attestation(make_attestation(
            "a1",
            "peer-1",
            TrustLevel::Limited,
            0,
            200,
        ));
        mgr.add_attestation(make_attestation(
            "a2",
            "peer-1",
            TrustLevel::Authority,
            0,
            200,
        ));
        mgr.add_attestation(make_attestation(
            "a3",
            "peer-1",
            TrustLevel::Trusted,
            0,
            200,
        ));
        assert_eq!(mgr.effective_trust("peer-1", 100), TrustLevel::Authority);
    }

    // ------------------------------------------------------------------
    // add_attestation
    // ------------------------------------------------------------------

    #[test]
    fn test_add_attestation_auto_creates_target() {
        let mut mgr = PeerTrustManager::new();
        mgr.add_attestation(make_attestation(
            "auth",
            "peer-new",
            TrustLevel::Trusted,
            0,
            50,
        ));
        assert!(mgr.records.contains_key("peer-new"));
        assert_eq!(mgr.records["peer-new"].level, TrustLevel::Untrusted);
    }

    #[test]
    fn test_add_attestation_appends_to_record() {
        let mut mgr = PeerTrustManager::new();
        mgr.set_trust("peer-1", TrustLevel::Untrusted);
        mgr.add_attestation(make_attestation("a1", "peer-1", TrustLevel::Limited, 0, 50));
        mgr.add_attestation(make_attestation("a2", "peer-1", TrustLevel::Trusted, 0, 50));
        assert_eq!(mgr.records["peer-1"].attestations.len(), 2);
    }

    // ------------------------------------------------------------------
    // peers_at_or_above
    // ------------------------------------------------------------------

    #[test]
    fn test_peers_at_or_above_filters_correctly() {
        let mut mgr = PeerTrustManager::new();
        mgr.set_trust("alice", TrustLevel::Trusted);
        mgr.set_trust("bob", TrustLevel::Limited);
        mgr.set_trust("carol", TrustLevel::Authority);

        let peers = mgr.peers_at_or_above(TrustLevel::Trusted, 0);
        assert!(peers.contains(&"alice"));
        assert!(peers.contains(&"carol"));
        assert!(!peers.contains(&"bob"));
    }

    #[test]
    fn test_peers_at_or_above_sorted_alphabetically() {
        let mut mgr = PeerTrustManager::new();
        mgr.set_trust("zara", TrustLevel::Trusted);
        mgr.set_trust("alice", TrustLevel::Trusted);
        mgr.set_trust("mike", TrustLevel::Trusted);

        let peers = mgr.peers_at_or_above(TrustLevel::Trusted, 0);
        assert_eq!(peers, vec!["alice", "mike", "zara"]);
    }

    // ------------------------------------------------------------------
    // revoke_attestations
    // ------------------------------------------------------------------

    #[test]
    fn test_revoke_attestations_removes_all_from_attester() {
        let mut mgr = PeerTrustManager::new();
        mgr.set_trust("peer-1", TrustLevel::Untrusted);
        mgr.set_trust("peer-2", TrustLevel::Untrusted);
        mgr.add_attestation(make_attestation(
            "bad-auth",
            "peer-1",
            TrustLevel::Vouched,
            0,
            200,
        ));
        mgr.add_attestation(make_attestation(
            "bad-auth",
            "peer-2",
            TrustLevel::Vouched,
            0,
            200,
        ));

        mgr.revoke_attestations("bad-auth");

        assert!(mgr.records["peer-1"].attestations.is_empty());
        assert!(mgr.records["peer-2"].attestations.is_empty());
    }

    #[test]
    fn test_revoke_attestations_leaves_other_attestations() {
        let mut mgr = PeerTrustManager::new();
        mgr.set_trust("peer-1", TrustLevel::Untrusted);
        mgr.add_attestation(make_attestation(
            "bad-auth",
            "peer-1",
            TrustLevel::Vouched,
            0,
            200,
        ));
        mgr.add_attestation(make_attestation(
            "good-auth",
            "peer-1",
            TrustLevel::Trusted,
            0,
            200,
        ));

        mgr.revoke_attestations("bad-auth");

        let attestations = &mgr.records["peer-1"].attestations;
        assert_eq!(attestations.len(), 1);
        assert_eq!(attestations[0].attester_peer_id, "good-auth");
    }

    // ------------------------------------------------------------------
    // remove_peer
    // ------------------------------------------------------------------

    #[test]
    fn test_remove_peer_returns_true_when_present() {
        let mut mgr = PeerTrustManager::new();
        mgr.set_trust("peer-1", TrustLevel::Trusted);
        assert!(mgr.remove_peer("peer-1"));
    }

    #[test]
    fn test_remove_peer_returns_false_when_absent() {
        let mut mgr = PeerTrustManager::new();
        assert!(!mgr.remove_peer("ghost"));
    }

    // ------------------------------------------------------------------
    // stats
    // ------------------------------------------------------------------

    #[test]
    fn test_stats_total_peers() {
        let mut mgr = PeerTrustManager::new();
        mgr.set_trust("p1", TrustLevel::Trusted);
        mgr.set_trust("p2", TrustLevel::Limited);
        let s = mgr.stats(0);
        assert_eq!(s.total_peers, 2);
    }

    #[test]
    fn test_stats_by_level_counts_effective_levels() {
        let mut mgr = PeerTrustManager::new();
        mgr.set_trust("p1", TrustLevel::Trusted);
        mgr.set_trust("p2", TrustLevel::Limited);
        // p2 gets a valid attestation upgrading to Vouched
        mgr.add_attestation(make_attestation("a", "p2", TrustLevel::Vouched, 0, 100));
        let s = mgr.stats(50);
        assert_eq!(
            s.by_level.get(&TrustLevel::Trusted).copied().unwrap_or(0),
            1
        );
        assert_eq!(
            s.by_level.get(&TrustLevel::Vouched).copied().unwrap_or(0),
            1
        );
    }

    #[test]
    fn test_stats_total_attestations() {
        let mut mgr = PeerTrustManager::new();
        mgr.set_trust("p1", TrustLevel::Untrusted);
        mgr.add_attestation(make_attestation("a1", "p1", TrustLevel::Limited, 0, 200));
        mgr.add_attestation(make_attestation("a2", "p1", TrustLevel::Trusted, 0, 200));
        let s = mgr.stats(50);
        assert_eq!(s.total_attestations, 2);
    }

    #[test]
    fn test_stats_expired_attestations_count() {
        let mut mgr = PeerTrustManager::new();
        mgr.set_trust("p1", TrustLevel::Untrusted);
        mgr.add_attestation(make_attestation("a1", "p1", TrustLevel::Limited, 0, 50)); // will expire at tick 100
        mgr.add_attestation(make_attestation("a2", "p1", TrustLevel::Trusted, 0, 200)); // still valid
        let s = mgr.stats(100); // tick 100: a1 is expired (100 < 50 is false), a2 is valid
        assert_eq!(s.expired_attestations, 1);
        assert_eq!(s.total_attestations, 2);
    }
}
