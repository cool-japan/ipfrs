//! Peer Discovery Cache
//!
//! Caches peer discovery results from multiple sources (DHT, mDNS, bootstrap, relay)
//! to reduce redundant lookups and enable offline-resilient reconnection.

use std::collections::HashMap;

/// Source from which a peer was discovered.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DiscoverySource {
    /// Kademlia DHT discovery
    Dht,
    /// mDNS local discovery
    Mdns,
    /// Bootstrap node discovery
    Bootstrap,
    /// Relay-based discovery
    Relay,
    /// Manually added peer
    Manual,
}

/// A record representing a discovered peer and its connection history.
#[derive(Clone, Debug)]
pub struct PeerRecord {
    /// Peer identifier (libp2p PeerId as string)
    pub peer_id: String,
    /// Known multiaddrs for this peer
    pub addresses: Vec<String>,
    /// Source from which this peer was discovered
    pub source: DiscoverySource,
    /// Unix timestamp (seconds) when this peer was first discovered
    pub discovered_at_secs: u64,
    /// Unix timestamp (seconds) when this peer was last seen
    pub last_seen_secs: u64,
    /// Number of successful connection attempts
    pub successful_connections: u32,
    /// Number of failed connection attempts
    pub failed_connections: u32,
}

impl PeerRecord {
    /// Compute a reliability score in [0.0, 1.0].
    ///
    /// Returns 0.5 when no connection attempts have been made.
    /// Otherwise returns `successful / (successful + failed)`.
    pub fn reliability_score(&self) -> f64 {
        let total = self.successful_connections + self.failed_connections;
        if total == 0 {
            0.5
        } else {
            self.successful_connections as f64 / total as f64
        }
    }
}

/// Aggregate statistics for the discovery cache.
#[derive(Clone, Debug)]
pub struct DiscoveryCacheStats {
    /// Total number of peers currently cached
    pub total_peers: usize,
    /// Count of peers per discovery source
    pub by_source: HashMap<DiscoverySource, usize>,
    /// Average reliability score across all cached peers (0.0 if no peers)
    pub avg_reliability: f64,
}

/// Cache for peer discovery results with TTL and capacity management.
pub struct PeerDiscoveryCache {
    /// Peer records keyed by peer_id
    pub records: HashMap<String, PeerRecord>,
    /// Maximum number of records to retain
    pub max_size: usize,
    /// Time-to-live in seconds; records older than this are stale
    pub ttl_secs: u64,
}

impl PeerDiscoveryCache {
    /// Create a new empty cache.
    pub fn new(max_size: usize, ttl_secs: u64) -> Self {
        Self {
            records: HashMap::new(),
            max_size,
            ttl_secs,
        }
    }

    /// Insert or update a peer record.
    ///
    /// If the peer already exists, addresses are merged (deduped), `last_seen_secs` is
    /// updated if the incoming record is newer, and `source` is updated.
    ///
    /// If the cache is at capacity and the peer is new, the peer with the lowest
    /// `reliability_score` is evicted (ties broken by oldest `last_seen_secs`).
    pub fn upsert(&mut self, record: PeerRecord) {
        if let Some(existing) = self.records.get_mut(&record.peer_id) {
            // Merge addresses (dedup)
            for addr in &record.addresses {
                if !existing.addresses.contains(addr) {
                    existing.addresses.push(addr.clone());
                }
            }
            // Update last_seen_secs if incoming is newer
            if record.last_seen_secs > existing.last_seen_secs {
                existing.last_seen_secs = record.last_seen_secs;
            }
            // Update source
            existing.source = record.source;
            return;
        }

        // New peer — evict if at capacity
        if self.records.len() >= self.max_size {
            self.evict_lowest_reliability();
        }

        self.records.insert(record.peer_id.clone(), record);
    }

    /// Evict the peer with the lowest reliability score.
    /// Ties are broken by choosing the peer with the oldest `last_seen_secs`.
    fn evict_lowest_reliability(&mut self) {
        let victim = self
            .records
            .iter()
            .min_by(|a, b| {
                let score_a = a.1.reliability_score();
                let score_b = b.1.reliability_score();
                score_a
                    .partial_cmp(&score_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.1.last_seen_secs.cmp(&b.1.last_seen_secs))
            })
            .map(|(k, _)| k.clone());

        if let Some(key) = victim {
            self.records.remove(&key);
        }
    }

    /// Record a successful connection to a peer.
    ///
    /// Increments `successful_connections` and updates `last_seen_secs`.
    /// Returns `false` if the peer is not in the cache.
    pub fn record_success(&mut self, peer_id: &str, now_secs: u64) -> bool {
        match self.records.get_mut(peer_id) {
            Some(record) => {
                record.successful_connections += 1;
                record.last_seen_secs = now_secs;
                true
            }
            None => false,
        }
    }

    /// Record a failed connection attempt to a peer.
    ///
    /// Increments `failed_connections`.
    /// Returns `false` if the peer is not in the cache.
    pub fn record_failure(&mut self, peer_id: &str) -> bool {
        match self.records.get_mut(peer_id) {
            Some(record) => {
                record.failed_connections += 1;
                true
            }
            None => false,
        }
    }

    /// Remove all records where `last_seen_secs + ttl_secs < now_secs`.
    pub fn evict_stale(&mut self, now_secs: u64) {
        let ttl = self.ttl_secs;
        self.records
            .retain(|_, record| record.last_seen_secs + ttl >= now_secs);
    }

    /// Look up a peer record by peer_id.
    pub fn get(&self, peer_id: &str) -> Option<&PeerRecord> {
        self.records.get(peer_id)
    }

    /// Return all peer records sorted by reliability score descending.
    pub fn peers_by_reliability(&self) -> Vec<&PeerRecord> {
        let mut peers: Vec<&PeerRecord> = self.records.values().collect();
        peers.sort_by(|a, b| {
            b.reliability_score()
                .partial_cmp(&a.reliability_score())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        peers
    }

    /// Return all peers discovered from a specific source, sorted by `last_seen_secs` descending.
    pub fn peers_from_source(&self, source: DiscoverySource) -> Vec<&PeerRecord> {
        let mut peers: Vec<&PeerRecord> = self
            .records
            .values()
            .filter(|r| r.source == source)
            .collect();
        peers.sort_by_key(|b| std::cmp::Reverse(b.last_seen_secs));
        peers
    }

    /// Remove a peer from the cache.
    ///
    /// Returns `true` if the peer was present and removed, `false` otherwise.
    pub fn remove(&mut self, peer_id: &str) -> bool {
        self.records.remove(peer_id).is_some()
    }

    /// Compute and return aggregate cache statistics.
    pub fn stats(&self) -> DiscoveryCacheStats {
        let total_peers = self.records.len();

        let mut by_source: HashMap<DiscoverySource, usize> = HashMap::new();
        let mut reliability_sum = 0.0_f64;

        for record in self.records.values() {
            *by_source.entry(record.source).or_insert(0) += 1;
            reliability_sum += record.reliability_score();
        }

        let avg_reliability = if total_peers == 0 {
            0.0
        } else {
            reliability_sum / total_peers as f64
        };

        DiscoveryCacheStats {
            total_peers,
            by_source,
            avg_reliability,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_record(
        peer_id: &str,
        source: DiscoverySource,
        last_seen_secs: u64,
        successful: u32,
        failed: u32,
    ) -> PeerRecord {
        PeerRecord {
            peer_id: peer_id.to_string(),
            addresses: vec![format!("/ip4/127.0.0.1/tcp/{}", peer_id.len())],
            source,
            discovered_at_secs: 1000,
            last_seen_secs,
            successful_connections: successful,
            failed_connections: failed,
        }
    }

    // 1. new() starts empty
    #[test]
    fn test_new_starts_empty() {
        let cache = PeerDiscoveryCache::new(10, 300);
        assert_eq!(cache.records.len(), 0);
        assert_eq!(cache.max_size, 10);
        assert_eq!(cache.ttl_secs, 300);
    }

    // 2. upsert adds new record
    #[test]
    fn test_upsert_adds_new_record() {
        let mut cache = PeerDiscoveryCache::new(10, 300);
        let rec = make_record("peer1", DiscoverySource::Dht, 2000, 0, 0);
        cache.upsert(rec);
        assert!(cache.records.contains_key("peer1"));
    }

    // 3. upsert updates existing (merge addresses, update last_seen)
    #[test]
    fn test_upsert_updates_existing_merge_addresses() {
        let mut cache = PeerDiscoveryCache::new(10, 300);
        let rec1 = PeerRecord {
            peer_id: "peer1".to_string(),
            addresses: vec!["/ip4/1.2.3.4/tcp/4001".to_string()],
            source: DiscoverySource::Dht,
            discovered_at_secs: 1000,
            last_seen_secs: 2000,
            successful_connections: 0,
            failed_connections: 0,
        };
        cache.upsert(rec1);

        let rec2 = PeerRecord {
            peer_id: "peer1".to_string(),
            addresses: vec![
                "/ip4/1.2.3.4/tcp/4001".to_string(),
                "/ip4/5.6.7.8/tcp/4002".to_string(),
            ],
            source: DiscoverySource::Mdns,
            discovered_at_secs: 1500,
            last_seen_secs: 3000,
            successful_connections: 0,
            failed_connections: 0,
        };
        cache.upsert(rec2);

        let record = cache
            .records
            .get("peer1")
            .expect("test: peer1 should exist in cache after upsert");
        assert_eq!(record.addresses.len(), 2);
        assert_eq!(record.last_seen_secs, 3000);
        assert_eq!(record.source, DiscoverySource::Mdns);
    }

    // 4. upsert at capacity evicts lowest reliability
    #[test]
    fn test_upsert_at_capacity_evicts_lowest_reliability() {
        let mut cache = PeerDiscoveryCache::new(2, 300);
        // peer_a: reliability 1.0 (1 success, 0 fail)
        cache.upsert(make_record("peer_a", DiscoverySource::Dht, 2000, 1, 0));
        // peer_b: reliability 0.0 (0 success, 1 fail)
        cache.upsert(make_record("peer_b", DiscoverySource::Dht, 2000, 0, 1));
        // peer_c should evict peer_b (lowest reliability)
        cache.upsert(make_record("peer_c", DiscoverySource::Dht, 2000, 1, 1));

        assert!(cache.records.contains_key("peer_a"));
        assert!(!cache.records.contains_key("peer_b"));
        assert!(cache.records.contains_key("peer_c"));
    }

    // 5. upsert at capacity: tie broken by oldest last_seen_secs
    #[test]
    fn test_upsert_at_capacity_tie_broken_by_oldest_last_seen() {
        let mut cache = PeerDiscoveryCache::new(2, 300);
        // Both peers have default reliability (0.5 — no connections)
        // peer_a last seen at 1000 (older)
        cache.upsert(make_record("peer_a", DiscoverySource::Dht, 1000, 0, 0));
        // peer_b last seen at 2000 (newer)
        cache.upsert(make_record("peer_b", DiscoverySource::Dht, 2000, 0, 0));
        // peer_c should evict peer_a (tie in reliability, older last_seen)
        cache.upsert(make_record("peer_c", DiscoverySource::Dht, 3000, 0, 0));

        assert!(
            !cache.records.contains_key("peer_a"),
            "peer_a should have been evicted"
        );
        assert!(cache.records.contains_key("peer_b"));
        assert!(cache.records.contains_key("peer_c"));
    }

    // 6. record_success increments counter
    #[test]
    fn test_record_success_increments_counter() {
        let mut cache = PeerDiscoveryCache::new(10, 300);
        cache.upsert(make_record("peer1", DiscoverySource::Dht, 2000, 0, 0));
        assert!(cache.record_success("peer1", 3000));
        assert_eq!(cache.records["peer1"].successful_connections, 1);
    }

    // 7. record_success updates last_seen_secs
    #[test]
    fn test_record_success_updates_last_seen() {
        let mut cache = PeerDiscoveryCache::new(10, 300);
        cache.upsert(make_record("peer1", DiscoverySource::Dht, 2000, 0, 0));
        cache.record_success("peer1", 5000);
        assert_eq!(cache.records["peer1"].last_seen_secs, 5000);
    }

    // 8. record_success returns false for unknown peer
    #[test]
    fn test_record_success_false_if_unknown() {
        let mut cache = PeerDiscoveryCache::new(10, 300);
        assert!(!cache.record_success("unknown_peer", 5000));
    }

    // 9. record_failure increments counter
    #[test]
    fn test_record_failure_increments_counter() {
        let mut cache = PeerDiscoveryCache::new(10, 300);
        cache.upsert(make_record("peer1", DiscoverySource::Dht, 2000, 0, 0));
        assert!(cache.record_failure("peer1"));
        assert_eq!(cache.records["peer1"].failed_connections, 1);
    }

    // 10. record_failure returns false for unknown peer
    #[test]
    fn test_record_failure_false_if_unknown() {
        let mut cache = PeerDiscoveryCache::new(10, 300);
        assert!(!cache.record_failure("unknown_peer"));
    }

    // 11. reliability_score = 0.5 when no connections
    #[test]
    fn test_reliability_score_no_connections() {
        let rec = make_record("peer1", DiscoverySource::Dht, 2000, 0, 0);
        assert!((rec.reliability_score() - 0.5).abs() < f64::EPSILON);
    }

    // 12. reliability_score computed correctly
    #[test]
    fn test_reliability_score_computed_correctly() {
        let rec = make_record("peer1", DiscoverySource::Dht, 2000, 3, 1);
        assert!((rec.reliability_score() - 0.75).abs() < f64::EPSILON);

        let rec2 = make_record("peer2", DiscoverySource::Dht, 2000, 0, 4);
        assert!(rec2.reliability_score().abs() < f64::EPSILON);

        let rec3 = make_record("peer3", DiscoverySource::Dht, 2000, 5, 0);
        assert!((rec3.reliability_score() - 1.0).abs() < f64::EPSILON);
    }

    // 13. evict_stale removes old records
    #[test]
    fn test_evict_stale_removes_old() {
        let mut cache = PeerDiscoveryCache::new(10, 100);
        // last_seen = 500, ttl = 100; at now=700: 500+100=600 < 700 => stale
        cache.upsert(make_record("stale_peer", DiscoverySource::Dht, 500, 0, 0));
        cache.evict_stale(700);
        assert!(!cache.records.contains_key("stale_peer"));
    }

    // 14. evict_stale keeps fresh records
    #[test]
    fn test_evict_stale_keeps_fresh() {
        let mut cache = PeerDiscoveryCache::new(10, 100);
        // last_seen = 650, ttl = 100; at now=700: 650+100=750 >= 700 => fresh
        cache.upsert(make_record("fresh_peer", DiscoverySource::Dht, 650, 0, 0));
        cache.evict_stale(700);
        assert!(cache.records.contains_key("fresh_peer"));
    }

    // 15. peers_by_reliability sorted descending
    #[test]
    fn test_peers_by_reliability_sorted_desc() {
        let mut cache = PeerDiscoveryCache::new(10, 300);
        cache.upsert(make_record("low", DiscoverySource::Dht, 2000, 1, 9)); // 0.1
        cache.upsert(make_record("mid", DiscoverySource::Dht, 2000, 5, 5)); // 0.5
        cache.upsert(make_record("high", DiscoverySource::Dht, 2000, 9, 1)); // 0.9

        let sorted = cache.peers_by_reliability();
        assert_eq!(sorted.len(), 3);
        assert!(sorted[0].reliability_score() >= sorted[1].reliability_score());
        assert!(sorted[1].reliability_score() >= sorted[2].reliability_score());
        assert_eq!(sorted[0].peer_id, "high");
        assert_eq!(sorted[2].peer_id, "low");
    }

    // 16. peers_from_source filters correctly
    #[test]
    fn test_peers_from_source_filters_correctly() {
        let mut cache = PeerDiscoveryCache::new(10, 300);
        cache.upsert(make_record("dht1", DiscoverySource::Dht, 2000, 0, 0));
        cache.upsert(make_record("dht2", DiscoverySource::Dht, 2001, 0, 0));
        cache.upsert(make_record("mdns1", DiscoverySource::Mdns, 2002, 0, 0));

        let dht_peers = cache.peers_from_source(DiscoverySource::Dht);
        assert_eq!(dht_peers.len(), 2);
        for p in &dht_peers {
            assert_eq!(p.source, DiscoverySource::Dht);
        }

        let mdns_peers = cache.peers_from_source(DiscoverySource::Mdns);
        assert_eq!(mdns_peers.len(), 1);
        assert_eq!(mdns_peers[0].peer_id, "mdns1");
    }

    // 17. peers_from_source sorted by last_seen_secs descending
    #[test]
    fn test_peers_from_source_sorted_by_last_seen_desc() {
        let mut cache = PeerDiscoveryCache::new(10, 300);
        cache.upsert(make_record("a", DiscoverySource::Bootstrap, 1000, 0, 0));
        cache.upsert(make_record("b", DiscoverySource::Bootstrap, 3000, 0, 0));
        cache.upsert(make_record("c", DiscoverySource::Bootstrap, 2000, 0, 0));

        let peers = cache.peers_from_source(DiscoverySource::Bootstrap);
        assert_eq!(peers.len(), 3);
        assert_eq!(peers[0].last_seen_secs, 3000);
        assert_eq!(peers[1].last_seen_secs, 2000);
        assert_eq!(peers[2].last_seen_secs, 1000);
    }

    // 18. remove returns true when peer found and removed
    #[test]
    fn test_remove_returns_true_when_present() {
        let mut cache = PeerDiscoveryCache::new(10, 300);
        cache.upsert(make_record("peer1", DiscoverySource::Dht, 2000, 0, 0));
        assert!(cache.remove("peer1"));
        assert!(!cache.records.contains_key("peer1"));
    }

    // 19. remove returns false when peer not found
    #[test]
    fn test_remove_returns_false_when_absent() {
        let mut cache = PeerDiscoveryCache::new(10, 300);
        assert!(!cache.remove("nonexistent"));
    }

    // 20. get returns Some for known peer
    #[test]
    fn test_get_returns_some() {
        let mut cache = PeerDiscoveryCache::new(10, 300);
        cache.upsert(make_record("peer1", DiscoverySource::Relay, 2000, 2, 1));
        let rec = cache.get("peer1");
        assert!(rec.is_some());
        assert_eq!(
            rec.expect("test: peer1 record should be found by get()")
                .peer_id,
            "peer1"
        );
    }

    // 21. get returns None for unknown peer
    #[test]
    fn test_get_returns_none() {
        let cache = PeerDiscoveryCache::new(10, 300);
        assert!(cache.get("nobody").is_none());
    }

    // 22. stats total_peers
    #[test]
    fn test_stats_total_peers() {
        let mut cache = PeerDiscoveryCache::new(10, 300);
        assert_eq!(cache.stats().total_peers, 0);
        cache.upsert(make_record("p1", DiscoverySource::Dht, 2000, 0, 0));
        cache.upsert(make_record("p2", DiscoverySource::Dht, 2000, 0, 0));
        assert_eq!(cache.stats().total_peers, 2);
    }

    // 23. stats by_source counts
    #[test]
    fn test_stats_by_source_counts() {
        let mut cache = PeerDiscoveryCache::new(10, 300);
        cache.upsert(make_record("d1", DiscoverySource::Dht, 2000, 0, 0));
        cache.upsert(make_record("d2", DiscoverySource::Dht, 2000, 0, 0));
        cache.upsert(make_record("m1", DiscoverySource::Mdns, 2000, 0, 0));
        cache.upsert(make_record("b1", DiscoverySource::Bootstrap, 2000, 0, 0));

        let stats = cache.stats();
        assert_eq!(*stats.by_source.get(&DiscoverySource::Dht).unwrap_or(&0), 2);
        assert_eq!(
            *stats.by_source.get(&DiscoverySource::Mdns).unwrap_or(&0),
            1
        );
        assert_eq!(
            *stats
                .by_source
                .get(&DiscoverySource::Bootstrap)
                .unwrap_or(&0),
            1
        );
        assert_eq!(
            *stats.by_source.get(&DiscoverySource::Relay).unwrap_or(&0),
            0
        );
    }

    // 24. stats avg_reliability
    #[test]
    fn test_stats_avg_reliability() {
        let mut cache = PeerDiscoveryCache::new(10, 300);
        // Empty cache => 0.0
        assert!((cache.stats().avg_reliability - 0.0).abs() < f64::EPSILON);

        // Two peers: reliability 1.0 and 0.0 => avg 0.5
        cache.upsert(make_record("good", DiscoverySource::Dht, 2000, 4, 0));
        cache.upsert(make_record("bad", DiscoverySource::Dht, 2000, 0, 4));
        let avg = cache.stats().avg_reliability;
        assert!((avg - 0.5).abs() < f64::EPSILON);
    }

    // 25. upsert does not change last_seen_secs when incoming is older
    #[test]
    fn test_upsert_does_not_update_last_seen_if_older() {
        let mut cache = PeerDiscoveryCache::new(10, 300);
        cache.upsert(make_record("peer1", DiscoverySource::Dht, 5000, 0, 0));
        let older = PeerRecord {
            peer_id: "peer1".to_string(),
            addresses: vec![],
            source: DiscoverySource::Mdns,
            discovered_at_secs: 1000,
            last_seen_secs: 3000,
            successful_connections: 0,
            failed_connections: 0,
        };
        cache.upsert(older);
        assert_eq!(cache.records["peer1"].last_seen_secs, 5000);
    }

    // 26. Manual source entries work correctly
    #[test]
    fn test_manual_source() {
        let mut cache = PeerDiscoveryCache::new(10, 300);
        cache.upsert(make_record("manual1", DiscoverySource::Manual, 9000, 10, 0));
        let peers = cache.peers_from_source(DiscoverySource::Manual);
        assert_eq!(peers.len(), 1);
        assert!((peers[0].reliability_score() - 1.0).abs() < f64::EPSILON);
    }
}
