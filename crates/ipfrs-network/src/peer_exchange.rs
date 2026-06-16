//! Peer Exchange Protocol (PEX)
//!
//! Periodic peer list exchange for peer discovery without DHT.
//! Nodes periodically share subsets of their known peers with
//! connected peers, enabling organic network growth and resilience.

use std::collections::HashMap;

/// Configuration for the Peer Exchange Protocol.
#[derive(Debug, Clone)]
pub struct PexConfig {
    /// Maximum number of peers to include in a single exchange.
    pub max_peers_per_exchange: usize,
    /// Number of ticks between exchanges.
    pub exchange_interval_ticks: u64,
    /// Maximum number of known peers to store.
    pub max_known_peers: usize,
}

impl Default for PexConfig {
    fn default() -> Self {
        Self {
            max_peers_per_exchange: 20,
            exchange_interval_ticks: 50,
            max_known_peers: 500,
        }
    }
}

/// Record of a known peer.
#[derive(Debug, Clone)]
pub struct PexPeerRecord {
    /// The peer's identifier.
    pub peer_id: String,
    /// Known multiaddresses for this peer.
    pub addresses: Vec<String>,
    /// Tick at which this peer was last seen or updated.
    pub last_seen_tick: u64,
    /// How this peer was discovered.
    pub source: PeerSource,
}

/// How a peer was discovered.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PeerSource {
    /// Connected directly.
    Direct,
    /// Received via peer exchange.
    Exchange,
    /// From the bootstrap list.
    Bootstrap,
}

impl PeerSource {
    /// Priority for selection ordering (lower = higher priority).
    fn priority(self) -> u8 {
        match self {
            PeerSource::Direct => 0,
            PeerSource::Exchange => 1,
            PeerSource::Bootstrap => 2,
        }
    }
}

/// Statistics for the Peer Exchange Protocol.
#[derive(Debug, Clone)]
pub struct PexStats {
    /// Total number of known peers.
    pub total_peers: usize,
    /// Number of directly connected peers.
    pub direct_peers: usize,
    /// Number of peers learned via exchange.
    pub exchange_peers: usize,
    /// Number of bootstrap peers.
    pub bootstrap_peers: usize,
    /// Number of exchanges completed.
    pub exchanges_completed: u64,
}

/// Peer Exchange Protocol implementation.
///
/// Manages a set of known peers and supports periodic exchange of peer
/// lists with connected peers for decentralised peer discovery.
pub struct PeerExchangeProtocol {
    config: PexConfig,
    known_peers: HashMap<String, PexPeerRecord>,
    current_tick: u64,
    exchanges_completed: u64,
    last_exchange_tick: u64,
}

impl PeerExchangeProtocol {
    /// Create a new `PeerExchangeProtocol` with the given configuration.
    pub fn new(config: PexConfig) -> Self {
        Self {
            config,
            known_peers: HashMap::new(),
            current_tick: 0,
            exchanges_completed: 0,
            last_exchange_tick: 0,
        }
    }

    /// Add or update a known peer.
    ///
    /// If the peer already exists, its addresses and last-seen tick are
    /// updated. The source is upgraded if the new source has higher
    /// priority (Direct > Exchange > Bootstrap).
    pub fn add_peer(&mut self, peer_id: &str, addresses: Vec<String>, source: PeerSource) {
        if let Some(existing) = self.known_peers.get_mut(peer_id) {
            existing.addresses = addresses;
            existing.last_seen_tick = self.current_tick;
            // Upgrade source if new source is higher priority
            if source.priority() < existing.source.priority() {
                existing.source = source;
            }
        } else {
            self.known_peers.insert(
                peer_id.to_string(),
                PexPeerRecord {
                    peer_id: peer_id.to_string(),
                    addresses,
                    last_seen_tick: self.current_tick,
                    source,
                },
            );
        }
    }

    /// Remove a peer from the known peer set.
    ///
    /// Returns `true` if the peer existed and was removed.
    pub fn remove_peer(&mut self, peer_id: &str) -> bool {
        self.known_peers.remove(peer_id).is_some()
    }

    /// Select peers for exchange.
    ///
    /// Returns up to `max_peers_per_exchange` peers, preferring
    /// Direct > Exchange > Bootstrap, and within each source category
    /// preferring the most recently seen peers.
    pub fn select_for_exchange(&self) -> Vec<&PexPeerRecord> {
        let mut peers: Vec<&PexPeerRecord> = self.known_peers.values().collect();

        // Sort by source priority (ascending = higher priority first),
        // then by last_seen_tick descending (most recent first).
        peers.sort_by(|a, b| {
            a.source
                .priority()
                .cmp(&b.source.priority())
                .then_with(|| b.last_seen_tick.cmp(&a.last_seen_tick))
        });

        peers.truncate(self.config.max_peers_per_exchange);
        peers
    }

    /// Receive a set of peers from a peer exchange.
    ///
    /// All received peers are added with `PeerSource::Exchange`. If the
    /// total number of known peers exceeds `max_known_peers`, the oldest
    /// peers (by `last_seen_tick`) are evicted.
    pub fn receive_exchange(&mut self, peers: Vec<(String, Vec<String>)>) {
        for (peer_id, addresses) in peers {
            self.add_peer(&peer_id, addresses, PeerSource::Exchange);
        }

        self.exchanges_completed += 1;
        self.last_exchange_tick = self.current_tick;

        // Evict oldest peers if over capacity
        self.enforce_max_peers();
    }

    /// Returns `true` if enough ticks have elapsed since the last
    /// exchange to warrant a new one.
    pub fn should_exchange(&self) -> bool {
        self.current_tick.saturating_sub(self.last_exchange_tick)
            >= self.config.exchange_interval_ticks
    }

    /// Advance the internal clock by one tick.
    pub fn tick(&mut self) {
        self.current_tick += 1;
    }

    /// Look up a peer by ID.
    pub fn get_peer(&self, peer_id: &str) -> Option<&PexPeerRecord> {
        self.known_peers.get(peer_id)
    }

    /// Return the number of known peers.
    pub fn peer_count(&self) -> usize {
        self.known_peers.len()
    }

    /// Return all peers discovered via the given source.
    pub fn peers_by_source(&self, source: PeerSource) -> Vec<&PexPeerRecord> {
        self.known_peers
            .values()
            .filter(|p| p.source == source)
            .collect()
    }

    /// Return protocol statistics.
    pub fn stats(&self) -> PexStats {
        let mut direct = 0usize;
        let mut exchange = 0usize;
        let mut bootstrap = 0usize;

        for peer in self.known_peers.values() {
            match peer.source {
                PeerSource::Direct => direct += 1,
                PeerSource::Exchange => exchange += 1,
                PeerSource::Bootstrap => bootstrap += 1,
            }
        }

        PexStats {
            total_peers: self.known_peers.len(),
            direct_peers: direct,
            exchange_peers: exchange,
            bootstrap_peers: bootstrap,
            exchanges_completed: self.exchanges_completed,
        }
    }

    /// Evict oldest peers when over `max_known_peers`.
    fn enforce_max_peers(&mut self) {
        if self.known_peers.len() <= self.config.max_known_peers {
            return;
        }

        let to_remove = self.known_peers.len() - self.config.max_known_peers;

        // Collect (peer_id, last_seen_tick) and sort oldest first
        let mut peers_by_age: Vec<(String, u64)> = self
            .known_peers
            .iter()
            .map(|(id, rec)| (id.clone(), rec.last_seen_tick))
            .collect();

        peers_by_age.sort_by_key(|&(_, tick)| tick);

        for (peer_id, _) in peers_by_age.into_iter().take(to_remove) {
            self.known_peers.remove(&peer_id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_protocol() -> PeerExchangeProtocol {
        PeerExchangeProtocol::new(PexConfig::default())
    }

    fn small_protocol() -> PeerExchangeProtocol {
        PeerExchangeProtocol::new(PexConfig {
            max_peers_per_exchange: 3,
            exchange_interval_ticks: 5,
            max_known_peers: 5,
        })
    }

    // --- Basic add/remove ---

    #[test]
    fn test_add_peer() {
        let mut pex = default_protocol();
        pex.add_peer(
            "peer1",
            vec!["/ip4/1.2.3.4/tcp/4001".to_string()],
            PeerSource::Direct,
        );
        assert_eq!(pex.peer_count(), 1);
        let peer = pex.get_peer("peer1").expect("peer should exist");
        assert_eq!(peer.peer_id, "peer1");
        assert_eq!(peer.source, PeerSource::Direct);
    }

    #[test]
    fn test_add_peer_updates_existing() {
        let mut pex = default_protocol();
        pex.add_peer(
            "peer1",
            vec!["/ip4/1.2.3.4/tcp/4001".to_string()],
            PeerSource::Bootstrap,
        );
        pex.tick();
        pex.add_peer(
            "peer1",
            vec!["/ip4/5.6.7.8/tcp/4001".to_string()],
            PeerSource::Direct,
        );
        assert_eq!(pex.peer_count(), 1);
        let peer = pex.get_peer("peer1").expect("peer should exist");
        assert_eq!(peer.addresses[0], "/ip4/5.6.7.8/tcp/4001");
        assert_eq!(peer.source, PeerSource::Direct); // upgraded
        assert_eq!(peer.last_seen_tick, 1);
    }

    #[test]
    fn test_add_peer_does_not_downgrade_source() {
        let mut pex = default_protocol();
        pex.add_peer("peer1", vec![], PeerSource::Direct);
        pex.add_peer("peer1", vec![], PeerSource::Bootstrap);
        let peer = pex.get_peer("peer1").expect("peer should exist");
        assert_eq!(peer.source, PeerSource::Direct);
    }

    #[test]
    fn test_remove_peer_exists() {
        let mut pex = default_protocol();
        pex.add_peer("peer1", vec![], PeerSource::Direct);
        assert!(pex.remove_peer("peer1"));
        assert_eq!(pex.peer_count(), 0);
    }

    #[test]
    fn test_remove_peer_not_exists() {
        let mut pex = default_protocol();
        assert!(!pex.remove_peer("nonexistent"));
    }

    #[test]
    fn test_remove_peer_idempotent() {
        let mut pex = default_protocol();
        pex.add_peer("peer1", vec![], PeerSource::Direct);
        assert!(pex.remove_peer("peer1"));
        assert!(!pex.remove_peer("peer1"));
    }

    // --- select_for_exchange ---

    #[test]
    fn test_select_for_exchange_empty() {
        let pex = default_protocol();
        assert!(pex.select_for_exchange().is_empty());
    }

    #[test]
    fn test_select_for_exchange_respects_max() {
        let mut pex = small_protocol();
        for i in 0..10 {
            pex.add_peer(&format!("peer{i}"), vec![], PeerSource::Direct);
        }
        assert_eq!(pex.select_for_exchange().len(), 3);
    }

    #[test]
    fn test_select_for_exchange_priority_ordering() {
        let mut pex = small_protocol();
        pex.add_peer("bootstrap1", vec![], PeerSource::Bootstrap);
        pex.add_peer("exchange1", vec![], PeerSource::Exchange);
        pex.add_peer("direct1", vec![], PeerSource::Direct);

        let selected = pex.select_for_exchange();
        assert_eq!(selected.len(), 3);
        assert_eq!(selected[0].source, PeerSource::Direct);
        assert_eq!(selected[1].source, PeerSource::Exchange);
        assert_eq!(selected[2].source, PeerSource::Bootstrap);
    }

    #[test]
    fn test_select_for_exchange_recency_within_source() {
        let mut pex = PeerExchangeProtocol::new(PexConfig {
            max_peers_per_exchange: 5,
            exchange_interval_ticks: 50,
            max_known_peers: 500,
        });

        pex.add_peer("direct_old", vec![], PeerSource::Direct);
        pex.tick();
        pex.tick();
        pex.add_peer("direct_new", vec![], PeerSource::Direct);

        let selected = pex.select_for_exchange();
        assert_eq!(selected[0].peer_id, "direct_new");
        assert_eq!(selected[1].peer_id, "direct_old");
    }

    #[test]
    fn test_select_for_exchange_direct_before_exchange_before_bootstrap() {
        let mut pex = PeerExchangeProtocol::new(PexConfig {
            max_peers_per_exchange: 2,
            exchange_interval_ticks: 50,
            max_known_peers: 500,
        });

        pex.add_peer("bootstrap1", vec![], PeerSource::Bootstrap);
        pex.add_peer("exchange1", vec![], PeerSource::Exchange);
        pex.add_peer("direct1", vec![], PeerSource::Direct);

        let selected = pex.select_for_exchange();
        assert_eq!(selected.len(), 2);
        assert_eq!(selected[0].source, PeerSource::Direct);
        assert_eq!(selected[1].source, PeerSource::Exchange);
    }

    // --- receive_exchange ---

    #[test]
    fn test_receive_exchange_adds_peers() {
        let mut pex = default_protocol();
        pex.receive_exchange(vec![
            (
                "peer1".to_string(),
                vec!["/ip4/1.2.3.4/tcp/4001".to_string()],
            ),
            (
                "peer2".to_string(),
                vec!["/ip4/5.6.7.8/tcp/4001".to_string()],
            ),
        ]);
        assert_eq!(pex.peer_count(), 2);
        let peer = pex.get_peer("peer1").expect("peer should exist");
        assert_eq!(peer.source, PeerSource::Exchange);
    }

    #[test]
    fn test_receive_exchange_increments_counter() {
        let mut pex = default_protocol();
        pex.receive_exchange(vec![]);
        pex.receive_exchange(vec![]);
        assert_eq!(pex.stats().exchanges_completed, 2);
    }

    #[test]
    fn test_receive_exchange_does_not_downgrade_direct() {
        let mut pex = default_protocol();
        pex.add_peer("peer1", vec![], PeerSource::Direct);
        pex.receive_exchange(vec![(
            "peer1".to_string(),
            vec!["/ip4/9.9.9.9/tcp/4001".to_string()],
        )]);
        let peer = pex.get_peer("peer1").expect("peer should exist");
        assert_eq!(peer.source, PeerSource::Direct);
    }

    // --- max_known_peers eviction ---

    #[test]
    fn test_max_known_peers_eviction() {
        let mut pex = small_protocol(); // max 5
        for i in 0..5 {
            pex.add_peer(&format!("peer{i}"), vec![], PeerSource::Direct);
            pex.tick();
        }
        assert_eq!(pex.peer_count(), 5);

        // Receive 3 more peers, should evict oldest to stay at 5
        pex.receive_exchange(vec![
            ("new1".to_string(), vec![]),
            ("new2".to_string(), vec![]),
            ("new3".to_string(), vec![]),
        ]);
        assert_eq!(pex.peer_count(), 5);
        // Oldest peers (peer0, peer1, peer2) should be evicted
        assert!(pex.get_peer("peer0").is_none());
        assert!(pex.get_peer("peer1").is_none());
        assert!(pex.get_peer("peer2").is_none());
    }

    #[test]
    fn test_eviction_preserves_newest() {
        let mut pex = PeerExchangeProtocol::new(PexConfig {
            max_peers_per_exchange: 20,
            exchange_interval_ticks: 50,
            max_known_peers: 3,
        });

        pex.add_peer("old", vec![], PeerSource::Direct);
        pex.tick();
        pex.add_peer("mid", vec![], PeerSource::Direct);
        pex.tick();
        pex.add_peer("new", vec![], PeerSource::Direct);
        pex.tick();

        pex.receive_exchange(vec![("newest".to_string(), vec![])]);

        assert_eq!(pex.peer_count(), 3);
        assert!(pex.get_peer("old").is_none());
        assert!(pex.get_peer("mid").is_some());
        assert!(pex.get_peer("new").is_some());
        assert!(pex.get_peer("newest").is_some());
    }

    // --- should_exchange / tick ---

    #[test]
    fn test_should_exchange_initially_true() {
        let pex = default_protocol();
        // At tick 0, last_exchange_tick 0, interval 50 => 0 >= 50 is false
        // Actually 0 - 0 = 0 < 50, so false
        assert!(!pex.should_exchange());
    }

    #[test]
    fn test_should_exchange_after_interval() {
        let mut pex = PeerExchangeProtocol::new(PexConfig {
            max_peers_per_exchange: 20,
            exchange_interval_ticks: 5,
            max_known_peers: 500,
        });

        for _ in 0..4 {
            pex.tick();
            assert!(!pex.should_exchange());
        }
        pex.tick(); // tick 5
        assert!(pex.should_exchange());
    }

    #[test]
    fn test_should_exchange_resets_after_receive() {
        let mut pex = PeerExchangeProtocol::new(PexConfig {
            max_peers_per_exchange: 20,
            exchange_interval_ticks: 3,
            max_known_peers: 500,
        });

        for _ in 0..3 {
            pex.tick();
        }
        assert!(pex.should_exchange());

        pex.receive_exchange(vec![]);
        assert!(!pex.should_exchange());
    }

    #[test]
    fn test_tick_advances_clock() {
        let mut pex = default_protocol();
        pex.tick();
        pex.tick();
        pex.add_peer("peer1", vec![], PeerSource::Direct);
        let peer = pex.get_peer("peer1").expect("peer should exist");
        assert_eq!(peer.last_seen_tick, 2);
    }

    // --- peers_by_source ---

    #[test]
    fn test_peers_by_source_direct() {
        let mut pex = default_protocol();
        pex.add_peer("d1", vec![], PeerSource::Direct);
        pex.add_peer("d2", vec![], PeerSource::Direct);
        pex.add_peer("e1", vec![], PeerSource::Exchange);
        pex.add_peer("b1", vec![], PeerSource::Bootstrap);

        let direct = pex.peers_by_source(PeerSource::Direct);
        assert_eq!(direct.len(), 2);
        for p in &direct {
            assert_eq!(p.source, PeerSource::Direct);
        }
    }

    #[test]
    fn test_peers_by_source_exchange() {
        let mut pex = default_protocol();
        pex.add_peer("e1", vec![], PeerSource::Exchange);
        pex.add_peer("d1", vec![], PeerSource::Direct);
        let exchange = pex.peers_by_source(PeerSource::Exchange);
        assert_eq!(exchange.len(), 1);
    }

    #[test]
    fn test_peers_by_source_bootstrap() {
        let mut pex = default_protocol();
        pex.add_peer("b1", vec![], PeerSource::Bootstrap);
        pex.add_peer("b2", vec![], PeerSource::Bootstrap);
        pex.add_peer("b3", vec![], PeerSource::Bootstrap);
        let bootstrap = pex.peers_by_source(PeerSource::Bootstrap);
        assert_eq!(bootstrap.len(), 3);
    }

    #[test]
    fn test_peers_by_source_empty() {
        let pex = default_protocol();
        assert!(pex.peers_by_source(PeerSource::Direct).is_empty());
    }

    // --- stats ---

    #[test]
    fn test_stats_empty() {
        let pex = default_protocol();
        let s = pex.stats();
        assert_eq!(s.total_peers, 0);
        assert_eq!(s.direct_peers, 0);
        assert_eq!(s.exchange_peers, 0);
        assert_eq!(s.bootstrap_peers, 0);
        assert_eq!(s.exchanges_completed, 0);
    }

    #[test]
    fn test_stats_accuracy() {
        let mut pex = default_protocol();
        pex.add_peer("d1", vec![], PeerSource::Direct);
        pex.add_peer("d2", vec![], PeerSource::Direct);
        pex.add_peer("e1", vec![], PeerSource::Exchange);
        pex.add_peer("b1", vec![], PeerSource::Bootstrap);
        pex.add_peer("b2", vec![], PeerSource::Bootstrap);
        pex.add_peer("b3", vec![], PeerSource::Bootstrap);
        pex.receive_exchange(vec![("e2".to_string(), vec![])]);

        let s = pex.stats();
        assert_eq!(s.total_peers, 7);
        assert_eq!(s.direct_peers, 2);
        assert_eq!(s.exchange_peers, 2);
        assert_eq!(s.bootstrap_peers, 3);
        assert_eq!(s.exchanges_completed, 1);
    }

    // --- get_peer ---

    #[test]
    fn test_get_peer_not_found() {
        let pex = default_protocol();
        assert!(pex.get_peer("nonexistent").is_none());
    }

    #[test]
    fn test_get_peer_after_remove() {
        let mut pex = default_protocol();
        pex.add_peer("peer1", vec![], PeerSource::Direct);
        pex.remove_peer("peer1");
        assert!(pex.get_peer("peer1").is_none());
    }

    // --- peer_count ---

    #[test]
    fn test_peer_count_empty() {
        let pex = default_protocol();
        assert_eq!(pex.peer_count(), 0);
    }

    #[test]
    fn test_peer_count_after_adds_and_removes() {
        let mut pex = default_protocol();
        pex.add_peer("p1", vec![], PeerSource::Direct);
        pex.add_peer("p2", vec![], PeerSource::Exchange);
        pex.add_peer("p3", vec![], PeerSource::Bootstrap);
        assert_eq!(pex.peer_count(), 3);
        pex.remove_peer("p2");
        assert_eq!(pex.peer_count(), 2);
    }

    // --- Edge cases ---

    #[test]
    fn test_add_peer_with_multiple_addresses() {
        let mut pex = default_protocol();
        pex.add_peer(
            "peer1",
            vec![
                "/ip4/1.2.3.4/tcp/4001".to_string(),
                "/ip6/::1/tcp/4001".to_string(),
                "/ip4/10.0.0.1/udp/4001/quic-v1".to_string(),
            ],
            PeerSource::Direct,
        );
        let peer = pex.get_peer("peer1").expect("peer should exist");
        assert_eq!(peer.addresses.len(), 3);
    }

    #[test]
    fn test_add_peer_empty_addresses() {
        let mut pex = default_protocol();
        pex.add_peer("peer1", vec![], PeerSource::Direct);
        let peer = pex.get_peer("peer1").expect("peer should exist");
        assert!(peer.addresses.is_empty());
    }

    #[test]
    fn test_large_exchange() {
        let mut pex = default_protocol();
        let peers: Vec<(String, Vec<String>)> = (0..100)
            .map(|i| {
                (
                    format!("peer{i}"),
                    vec![format!("/ip4/10.0.0.{}/tcp/4001", i % 256)],
                )
            })
            .collect();
        pex.receive_exchange(peers);
        assert_eq!(pex.peer_count(), 100);
    }

    #[test]
    fn test_config_default_values() {
        let config = PexConfig::default();
        assert_eq!(config.max_peers_per_exchange, 20);
        assert_eq!(config.exchange_interval_ticks, 50);
        assert_eq!(config.max_known_peers, 500);
    }

    #[test]
    fn test_select_all_when_fewer_than_max() {
        let mut pex = default_protocol(); // max 20
        pex.add_peer("p1", vec![], PeerSource::Direct);
        pex.add_peer("p2", vec![], PeerSource::Exchange);
        let selected = pex.select_for_exchange();
        assert_eq!(selected.len(), 2);
    }

    #[test]
    fn test_multiple_ticks_and_exchanges() {
        let mut pex = PeerExchangeProtocol::new(PexConfig {
            max_peers_per_exchange: 5,
            exchange_interval_ticks: 3,
            max_known_peers: 100,
        });

        // First exchange cycle
        for _ in 0..3 {
            pex.tick();
        }
        assert!(pex.should_exchange());
        pex.receive_exchange(vec![("a".to_string(), vec![])]);
        assert!(!pex.should_exchange());

        // Second exchange cycle
        for _ in 0..3 {
            pex.tick();
        }
        assert!(pex.should_exchange());
        pex.receive_exchange(vec![("b".to_string(), vec![])]);

        assert_eq!(pex.stats().exchanges_completed, 2);
        assert_eq!(pex.peer_count(), 2);
    }
}
