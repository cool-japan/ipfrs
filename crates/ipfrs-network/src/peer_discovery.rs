//! Peer Discovery Manager
//!
//! Manages multiple peer discovery mechanisms (mDNS, Bootstrap, DHT random walk, etc.)
//! and deduplicates discovered peers across all sources.

use std::collections::HashMap;

/// Source from which a peer was discovered.
#[derive(Clone, Debug, PartialEq)]
pub enum DiscoverySource {
    /// Local network multicast discovery via mDNS.
    Mdns,
    /// Discovered from a static bootstrap list.
    Bootstrap,
    /// Discovered via DHT iterative peer lookup (random walk).
    DhtRandomWalk,
    /// A connected peer advertised this peer (peer exchange).
    PeerExchange,
    /// Manually added by an operator.
    Manual,
}

/// A peer that has been discovered through one of the discovery mechanisms.
#[derive(Clone, Debug)]
pub struct DiscoveredPeer {
    /// Unique peer identifier (e.g., libp2p PeerId as string).
    pub peer_id: String,
    /// Known multiaddresses for this peer.
    pub addresses: Vec<String>,
    /// Which mechanism surfaced this peer.
    pub source: DiscoverySource,
    /// Unix timestamp (seconds) when this peer was first discovered.
    pub discovered_at_secs: u64,
    /// How many dial attempts have been made so far.
    pub dial_attempts: u32,
    /// Result of the most recent dial attempt.
    /// `Some(true)` = success, `Some(false)` = failure, `None` = not yet dialed.
    pub last_dial_result: Option<bool>,
}

impl DiscoveredPeer {
    /// Returns `true` when another dial should be attempted.
    ///
    /// A retry is warranted when we have not yet reached `max_attempts` *and*
    /// the peer has not already been reached successfully.
    pub fn should_retry(&self, max_attempts: u32) -> bool {
        self.dial_attempts < max_attempts && self.last_dial_result != Some(true)
    }
}

/// Aggregate statistics for the discovery manager.
#[derive(Clone, Debug, Default)]
pub struct DiscoveryStats {
    /// Total number of unique peers ever added (duplicates excluded).
    pub total_discovered: u64,
    /// Peers sourced from mDNS.
    pub from_mdns: u64,
    /// Peers sourced from a bootstrap list.
    pub from_bootstrap: u64,
    /// Peers sourced from DHT random walk.
    pub from_dht: u64,
    /// Peers sourced from peer exchange.
    pub from_peer_exchange: u64,
    /// Peers sourced from manual operator input.
    pub from_manual: u64,
    /// How many `add_peer` calls were rejected because the peer was already known.
    pub duplicates_skipped: u64,
    /// Cumulative successful dial results recorded.
    pub dial_successes: u64,
    /// Cumulative failed dial results recorded.
    pub dial_failures: u64,
}

/// Manages multiple peer discovery mechanisms and deduplicates discovered peers.
pub struct PeerDiscoveryManager {
    /// Map from peer_id to its discovery record.
    pub peers: HashMap<String, DiscoveredPeer>,
    /// Maximum number of dial attempts before a peer is considered permanently unreachable.
    pub max_dial_attempts: u32,
    /// Running statistics.
    pub stats: DiscoveryStats,
}

impl PeerDiscoveryManager {
    /// Create a new manager with the given dial-attempt limit.
    pub fn new(max_dial_attempts: u32) -> Self {
        Self {
            peers: HashMap::new(),
            max_dial_attempts,
            stats: DiscoveryStats::default(),
        }
    }

    /// Attempt to register a newly discovered peer.
    ///
    /// Returns `true` when the peer was inserted (first time seen).
    /// Returns `false` when the peer was already known; `stats.duplicates_skipped` is
    /// incremented in that case.
    pub fn add_peer(&mut self, peer: DiscoveredPeer) -> bool {
        if self.peers.contains_key(&peer.peer_id) {
            self.stats.duplicates_skipped += 1;
            return false;
        }

        // Update per-source counters.
        match &peer.source {
            DiscoverySource::Mdns => self.stats.from_mdns += 1,
            DiscoverySource::Bootstrap => self.stats.from_bootstrap += 1,
            DiscoverySource::DhtRandomWalk => self.stats.from_dht += 1,
            DiscoverySource::PeerExchange => self.stats.from_peer_exchange += 1,
            DiscoverySource::Manual => self.stats.from_manual += 1,
        }

        self.stats.total_discovered += 1;
        self.peers.insert(peer.peer_id.clone(), peer);
        true
    }

    /// Record the outcome of a dial attempt for the given peer.
    ///
    /// Increments `dial_attempts`, updates `last_dial_result`, and bumps the
    /// appropriate aggregate counter.  If `peer_id` is unknown this is a no-op.
    pub fn record_dial_result(&mut self, peer_id: &str, success: bool) {
        if let Some(peer) = self.peers.get_mut(peer_id) {
            peer.dial_attempts += 1;
            peer.last_dial_result = Some(success);
            if success {
                self.stats.dial_successes += 1;
            } else {
                self.stats.dial_failures += 1;
            }
        }
    }

    /// Return all peers that should still be dialed, sorted by ascending `dial_attempts`.
    ///
    /// A peer is a candidate when `should_retry(max_dial_attempts)` is true.
    pub fn candidates_to_dial(&self) -> Vec<&DiscoveredPeer> {
        let mut candidates: Vec<&DiscoveredPeer> = self
            .peers
            .values()
            .filter(|p| p.should_retry(self.max_dial_attempts))
            .collect();
        candidates.sort_by_key(|p| p.dial_attempts);
        candidates
    }

    /// Return all peers whose last dial was successful.
    pub fn connected_peers(&self) -> Vec<&DiscoveredPeer> {
        self.peers
            .values()
            .filter(|p| p.last_dial_result == Some(true))
            .collect()
    }

    /// Return all peers that have exhausted their dial budget without success.
    pub fn failed_peers(&self) -> Vec<&DiscoveredPeer> {
        self.peers
            .values()
            .filter(|p| {
                p.dial_attempts >= self.max_dial_attempts && p.last_dial_result != Some(true)
            })
            .collect()
    }

    /// Return all peers discovered via a specific source.
    pub fn peers_by_source(&self, source: DiscoverySource) -> Vec<&DiscoveredPeer> {
        self.peers.values().filter(|p| p.source == source).collect()
    }

    /// Remove a peer from the manager.
    ///
    /// Returns `true` if the peer was present and has been removed, `false` otherwise.
    pub fn remove_peer(&mut self, peer_id: &str) -> bool {
        self.peers.remove(peer_id).is_some()
    }

    /// Borrow the current discovery statistics.
    pub fn stats(&self) -> &DiscoveryStats {
        &self.stats
    }

    /// Merge additional addresses into a known peer's address list.
    ///
    /// Any address already recorded for that peer is silently skipped.
    /// If `peer_id` is unknown this is a no-op.
    pub fn merge_addresses(&mut self, peer_id: &str, new_addrs: &[String]) {
        if let Some(peer) = self.peers.get_mut(peer_id) {
            for addr in new_addrs {
                if !peer.addresses.contains(addr) {
                    peer.addresses.push(addr.clone());
                }
            }
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a minimal `DiscoveredPeer` with sensible defaults.
    fn make_peer(id: &str, source: DiscoverySource) -> DiscoveredPeer {
        DiscoveredPeer {
            peer_id: id.to_string(),
            addresses: vec![format!("/ip4/127.0.0.1/tcp/{}", id.len())],
            source,
            discovered_at_secs: 1_000_000,
            dial_attempts: 0,
            last_dial_result: None,
        }
    }

    // ── 1. new() produces empty state ─────────────────────────────────────────

    #[test]
    fn test_new_empty_state() {
        let mgr = PeerDiscoveryManager::new(3);
        assert_eq!(mgr.max_dial_attempts, 3);
        assert!(mgr.peers.is_empty());
        let s = mgr.stats();
        assert_eq!(s.total_discovered, 0);
        assert_eq!(s.duplicates_skipped, 0);
        assert_eq!(s.dial_successes, 0);
        assert_eq!(s.dial_failures, 0);
    }

    // ── 2. add_peer: new peer returns true, stats updated ────────────────────

    #[test]
    fn test_add_new_peer_returns_true() {
        let mut mgr = PeerDiscoveryManager::new(3);
        let result = mgr.add_peer(make_peer("peer1", DiscoverySource::Mdns));
        assert!(result);
        assert_eq!(mgr.stats().total_discovered, 1);
        assert_eq!(mgr.peers.len(), 1);
    }

    // ── 3. add_peer: duplicate returns false, duplicates_skipped++ ───────────

    #[test]
    fn test_add_duplicate_peer_returns_false() {
        let mut mgr = PeerDiscoveryManager::new(3);
        mgr.add_peer(make_peer("peer1", DiscoverySource::Mdns));
        let result = mgr.add_peer(make_peer("peer1", DiscoverySource::Bootstrap));
        assert!(!result);
        assert_eq!(mgr.stats().duplicates_skipped, 1);
        assert_eq!(mgr.stats().total_discovered, 1);
        assert_eq!(mgr.peers.len(), 1);
    }

    // ── 4-a. per-source stats: Mdns ──────────────────────────────────────────

    #[test]
    fn test_per_source_stats_mdns() {
        let mut mgr = PeerDiscoveryManager::new(3);
        mgr.add_peer(make_peer("p1", DiscoverySource::Mdns));
        assert_eq!(mgr.stats().from_mdns, 1);
    }

    // ── 4-b. per-source stats: Bootstrap ─────────────────────────────────────

    #[test]
    fn test_per_source_stats_bootstrap() {
        let mut mgr = PeerDiscoveryManager::new(3);
        mgr.add_peer(make_peer("p2", DiscoverySource::Bootstrap));
        assert_eq!(mgr.stats().from_bootstrap, 1);
    }

    // ── 4-c. per-source stats: DhtRandomWalk ─────────────────────────────────

    #[test]
    fn test_per_source_stats_dht() {
        let mut mgr = PeerDiscoveryManager::new(3);
        mgr.add_peer(make_peer("p3", DiscoverySource::DhtRandomWalk));
        assert_eq!(mgr.stats().from_dht, 1);
    }

    // ── 4-d. per-source stats: PeerExchange ──────────────────────────────────

    #[test]
    fn test_per_source_stats_peer_exchange() {
        let mut mgr = PeerDiscoveryManager::new(3);
        mgr.add_peer(make_peer("p4", DiscoverySource::PeerExchange));
        assert_eq!(mgr.stats().from_peer_exchange, 1);
    }

    // ── 4-e. per-source stats: Manual ────────────────────────────────────────

    #[test]
    fn test_per_source_stats_manual() {
        let mut mgr = PeerDiscoveryManager::new(3);
        mgr.add_peer(make_peer("p5", DiscoverySource::Manual));
        assert_eq!(mgr.stats().from_manual, 1);
    }

    // ── 5. record_dial_result: success ────────────────────────────────────────

    #[test]
    fn test_record_dial_result_success() {
        let mut mgr = PeerDiscoveryManager::new(3);
        mgr.add_peer(make_peer("peer1", DiscoverySource::Bootstrap));
        mgr.record_dial_result("peer1", true);
        let peer = mgr.peers.get("peer1").expect("peer must exist");
        assert_eq!(peer.dial_attempts, 1);
        assert_eq!(peer.last_dial_result, Some(true));
        assert_eq!(mgr.stats().dial_successes, 1);
        assert_eq!(mgr.stats().dial_failures, 0);
    }

    // ── 6. record_dial_result: failure ────────────────────────────────────────

    #[test]
    fn test_record_dial_result_failure() {
        let mut mgr = PeerDiscoveryManager::new(3);
        mgr.add_peer(make_peer("peer1", DiscoverySource::Bootstrap));
        mgr.record_dial_result("peer1", false);
        let peer = mgr.peers.get("peer1").expect("peer must exist");
        assert_eq!(peer.dial_attempts, 1);
        assert_eq!(peer.last_dial_result, Some(false));
        assert_eq!(mgr.stats().dial_successes, 0);
        assert_eq!(mgr.stats().dial_failures, 1);
    }

    // ── 7. record_dial_result: unknown peer is no-op ──────────────────────────

    #[test]
    fn test_record_dial_result_unknown_peer_noop() {
        let mut mgr = PeerDiscoveryManager::new(3);
        mgr.record_dial_result("ghost", true);
        assert_eq!(mgr.stats().dial_successes, 0);
        assert_eq!(mgr.stats().dial_failures, 0);
    }

    // ── 8. should_retry: under max_attempts, not succeeded → true ─────────────

    #[test]
    fn test_should_retry_under_max_attempts() {
        let peer = DiscoveredPeer {
            peer_id: "p".to_string(),
            addresses: vec![],
            source: DiscoverySource::Mdns,
            discovered_at_secs: 0,
            dial_attempts: 1,
            last_dial_result: Some(false),
        };
        assert!(peer.should_retry(3));
    }

    // ── 9. should_retry: at max_attempts → false ──────────────────────────────

    #[test]
    fn test_should_retry_at_max_attempts() {
        let peer = DiscoveredPeer {
            peer_id: "p".to_string(),
            addresses: vec![],
            source: DiscoverySource::Mdns,
            discovered_at_secs: 0,
            dial_attempts: 3,
            last_dial_result: Some(false),
        };
        assert!(!peer.should_retry(3));
    }

    // ── 10. should_retry: already succeeded → false ───────────────────────────

    #[test]
    fn test_should_retry_already_succeeded() {
        let peer = DiscoveredPeer {
            peer_id: "p".to_string(),
            addresses: vec![],
            source: DiscoverySource::Mdns,
            discovered_at_secs: 0,
            dial_attempts: 1,
            last_dial_result: Some(true),
        };
        assert!(!peer.should_retry(3));
    }

    // ── 11. candidates_to_dial sorted by dial_attempts ascending ──────────────

    #[test]
    fn test_candidates_to_dial_sorted_ascending() {
        let mut mgr = PeerDiscoveryManager::new(5);

        for (id, attempts) in [("pa", 2u32), ("pb", 0u32), ("pc", 1u32)] {
            let mut p = make_peer(id, DiscoverySource::Bootstrap);
            p.dial_attempts = attempts;
            mgr.peers.insert(id.to_string(), p);
        }

        let candidates = mgr.candidates_to_dial();
        assert_eq!(candidates.len(), 3);
        let attempt_counts: Vec<u32> = candidates.iter().map(|p| p.dial_attempts).collect();
        assert_eq!(attempt_counts, vec![0, 1, 2]);
    }

    // ── 12. connected_peers filtered correctly ────────────────────────────────

    #[test]
    fn test_connected_peers_filtered() {
        let mut mgr = PeerDiscoveryManager::new(3);
        mgr.add_peer(make_peer("pa", DiscoverySource::Mdns));
        mgr.add_peer(make_peer("pb", DiscoverySource::Mdns));
        mgr.add_peer(make_peer("pc", DiscoverySource::Mdns));
        mgr.record_dial_result("pa", true);
        mgr.record_dial_result("pb", false);

        let connected = mgr.connected_peers();
        assert_eq!(connected.len(), 1);
        assert_eq!(connected[0].peer_id, "pa");
    }

    // ── 13. failed_peers filtered correctly ───────────────────────────────────

    #[test]
    fn test_failed_peers_filtered() {
        let mut mgr = PeerDiscoveryManager::new(2);
        mgr.add_peer(make_peer("pa", DiscoverySource::Bootstrap));
        mgr.add_peer(make_peer("pb", DiscoverySource::Bootstrap));
        mgr.add_peer(make_peer("pc", DiscoverySource::Bootstrap));

        // Exhaust pa's attempts without success.
        mgr.record_dial_result("pa", false);
        mgr.record_dial_result("pa", false);

        // pb succeeds on first try.
        mgr.record_dial_result("pb", true);

        // pc has one failed attempt (still within budget).
        mgr.record_dial_result("pc", false);

        let failed = mgr.failed_peers();
        assert_eq!(failed.len(), 1);
        assert_eq!(failed[0].peer_id, "pa");
    }

    // ── 14. peers_by_source filters by source ─────────────────────────────────

    #[test]
    fn test_peers_by_source() {
        let mut mgr = PeerDiscoveryManager::new(3);
        mgr.add_peer(make_peer("p1", DiscoverySource::Mdns));
        mgr.add_peer(make_peer("p2", DiscoverySource::Bootstrap));
        mgr.add_peer(make_peer("p3", DiscoverySource::Mdns));
        mgr.add_peer(make_peer("p4", DiscoverySource::DhtRandomWalk));

        let mdns_peers = mgr.peers_by_source(DiscoverySource::Mdns);
        assert_eq!(mdns_peers.len(), 2);

        let bootstrap_peers = mgr.peers_by_source(DiscoverySource::Bootstrap);
        assert_eq!(bootstrap_peers.len(), 1);

        let dht_peers = mgr.peers_by_source(DiscoverySource::DhtRandomWalk);
        assert_eq!(dht_peers.len(), 1);

        let manual_peers = mgr.peers_by_source(DiscoverySource::Manual);
        assert_eq!(manual_peers.len(), 0);
    }

    // ── 15. remove_peer returns true / false ──────────────────────────────────

    #[test]
    fn test_remove_peer_returns_correct_bool() {
        let mut mgr = PeerDiscoveryManager::new(3);
        mgr.add_peer(make_peer("p1", DiscoverySource::Manual));

        assert!(mgr.remove_peer("p1"));
        assert!(!mgr.remove_peer("p1")); // already gone
        assert!(!mgr.remove_peer("ghost")); // never existed
        assert!(mgr.peers.is_empty());
    }

    // ── 16. merge_addresses adds new, deduplicates existing ──────────────────

    #[test]
    fn test_merge_addresses_deduplicates() {
        let mut mgr = PeerDiscoveryManager::new(3);
        let mut peer = make_peer("p1", DiscoverySource::Mdns);
        peer.addresses = vec!["/ip4/1.2.3.4/tcp/4001".to_string()];
        mgr.add_peer(peer);

        // First merge: one new address, one duplicate.
        mgr.merge_addresses(
            "p1",
            &[
                "/ip4/1.2.3.4/tcp/4001".to_string(), // duplicate
                "/ip4/5.6.7.8/tcp/4001".to_string(), // new
            ],
        );
        let p = mgr.peers.get("p1").expect("peer must exist");
        assert_eq!(p.addresses.len(), 2);

        // Second merge: same two addresses again → no growth.
        mgr.merge_addresses(
            "p1",
            &[
                "/ip4/1.2.3.4/tcp/4001".to_string(),
                "/ip4/5.6.7.8/tcp/4001".to_string(),
            ],
        );
        let p = mgr.peers.get("p1").expect("peer must exist");
        assert_eq!(p.addresses.len(), 2);

        // Unknown peer: no-op (must not panic).
        mgr.merge_addresses("ghost", &["/ip4/9.9.9.9/tcp/4001".to_string()]);
    }

    // ── 17. stats() totals correct after multiple operations ──────────────────

    #[test]
    fn test_stats_totals_after_multiple_operations() {
        let mut mgr = PeerDiscoveryManager::new(3);

        // Add five distinct peers from different sources.
        mgr.add_peer(make_peer("p1", DiscoverySource::Mdns));
        mgr.add_peer(make_peer("p2", DiscoverySource::Bootstrap));
        mgr.add_peer(make_peer("p3", DiscoverySource::DhtRandomWalk));
        mgr.add_peer(make_peer("p4", DiscoverySource::PeerExchange));
        mgr.add_peer(make_peer("p5", DiscoverySource::Manual));

        // Two duplicates.
        mgr.add_peer(make_peer("p1", DiscoverySource::Mdns));
        mgr.add_peer(make_peer("p3", DiscoverySource::Bootstrap));

        // Three dial results.
        mgr.record_dial_result("p1", true);
        mgr.record_dial_result("p2", false);
        mgr.record_dial_result("p3", true);

        let s = mgr.stats();
        assert_eq!(s.total_discovered, 5);
        assert_eq!(s.from_mdns, 1);
        assert_eq!(s.from_bootstrap, 1);
        assert_eq!(s.from_dht, 1);
        assert_eq!(s.from_peer_exchange, 1);
        assert_eq!(s.from_manual, 1);
        assert_eq!(s.duplicates_skipped, 2);
        assert_eq!(s.dial_successes, 2);
        assert_eq!(s.dial_failures, 1);
    }
}
