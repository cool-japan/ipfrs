//! HNSW-on-DHT Shard Balancing
//!
//! This module implements consistent-hashing based shard balancing for distributed
//! HNSW indices stored across DHT peers.  The design goals are:
//!
//! * **Predictable placement** – Knuth multiplicative hashing maps each `vector_id`
//!   to a shard deterministically.
//! * **Load awareness** – Atomic per-shard counters track live vector counts so that
//!   hot-spot detection and rebalancing decisions can be made without locks.
//! * **Peer coordination** – `DhtShardRouter` maintains the mapping from peer IDs to
//!   the shards they host, enabling routing of search and insert requests.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};

// ---------------------------------------------------------------------------
// ShardConfig
// ---------------------------------------------------------------------------

/// Configuration parameters for the shard balancer.
#[derive(Debug, Clone)]
pub struct ShardConfig {
    /// Total number of logical shards in the cluster.
    pub num_shards: usize,
    /// How many peers each shard should be replicated to.
    pub replication_factor: usize,
    /// Soft upper limit on vectors per shard before rebalancing is flagged.
    pub max_vectors_per_shard: usize,
}

impl Default for ShardConfig {
    fn default() -> Self {
        Self {
            num_shards: 8,
            replication_factor: 3,
            max_vectors_per_shard: 200_000,
        }
    }
}

// ---------------------------------------------------------------------------
// ShardAssignment
// ---------------------------------------------------------------------------

/// Describes which shard a vector belongs to and which peers host that shard.
#[derive(Debug, Clone)]
pub struct ShardAssignment {
    /// Logical shard index (0..num_shards).
    pub shard_id: usize,
    /// The primary peer responsible for writes.
    pub primary_peer: String,
    /// Additional peers that hold replicas.
    pub replica_peers: Vec<String>,
}

// ---------------------------------------------------------------------------
// ShardBalancer
// ---------------------------------------------------------------------------

/// Tracks per-shard load and makes placement / rebalancing decisions.
///
/// All load counters use `AtomicUsize` so concurrent updates from multiple
/// async tasks require no mutex.
pub struct ShardBalancer {
    config: ShardConfig,
    shard_loads: Arc<Vec<AtomicUsize>>,
}

impl ShardBalancer {
    /// Create a new `ShardBalancer` with the given configuration.
    ///
    /// # Panics
    ///
    /// Panics if `config.num_shards == 0`.
    pub fn new(config: ShardConfig) -> Self {
        assert!(config.num_shards > 0, "num_shards must be > 0");
        let mut loads = Vec::with_capacity(config.num_shards);
        for _ in 0..config.num_shards {
            loads.push(AtomicUsize::new(0));
        }
        Self {
            config,
            shard_loads: Arc::new(loads),
        }
    }

    /// Return the shard that should store the vector with the given id.
    ///
    /// Uses Knuth multiplicative hashing for a uniform distribution.
    pub fn assign_vector(&self, vector_id: u64) -> usize {
        // Knuth multiplicative hash (32-bit Fibonacci constant widened to 64-bit)
        let hash = vector_id.wrapping_mul(2_654_435_761_u64);
        (hash as usize) % self.config.num_shards
    }

    /// Return the shard index with the lowest current load.
    ///
    /// In the case of a tie the shard with the smaller index wins, giving
    /// stable, deterministic behaviour in tests.
    pub fn least_loaded_shard(&self) -> usize {
        let mut min_load = usize::MAX;
        let mut min_shard = 0usize;
        for (idx, counter) in self.shard_loads.iter().enumerate() {
            let load = counter.load(Ordering::Relaxed);
            if load < min_load {
                min_load = load;
                min_shard = idx;
            }
        }
        min_shard
    }

    /// Atomically increment the vector count for the given shard.
    pub fn increment_shard_load(&self, shard_id: usize) {
        if let Some(counter) = self.shard_loads.get(shard_id) {
            counter.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Atomically decrement the vector count for the given shard.
    ///
    /// Saturates at zero to avoid underflow.
    pub fn decrement_shard_load(&self, shard_id: usize) {
        if let Some(counter) = self.shard_loads.get(shard_id) {
            // Saturating decrement via compare-exchange loop
            let mut current = counter.load(Ordering::Relaxed);
            loop {
                if current == 0 {
                    break;
                }
                match counter.compare_exchange_weak(
                    current,
                    current - 1,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => break,
                    Err(actual) => current = actual,
                }
            }
        }
    }

    /// Return a point-in-time snapshot of all shard load counts.
    pub fn shard_loads_snapshot(&self) -> Vec<usize> {
        self.shard_loads
            .iter()
            .map(|c| c.load(Ordering::Relaxed))
            .collect()
    }

    /// Return `true` when the ratio of the most-loaded shard to the
    /// least-loaded shard exceeds 2.0 (ignoring empty shards with zero load).
    pub fn rebalance_needed(&self) -> bool {
        let snapshot = self.shard_loads_snapshot();
        // Only consider shards that have at least one vector.
        let non_zero: Vec<usize> = snapshot.into_iter().filter(|&v| v > 0).collect();
        if non_zero.len() < 2 {
            return false;
        }
        let max = non_zero.iter().copied().max().unwrap_or(0);
        let min = non_zero.iter().copied().min().unwrap_or(0);
        if min == 0 {
            return false;
        }
        (max as f64 / min as f64) > 2.0
    }

    /// Return the indices of shards whose load exceeds the average load.
    pub fn hotspot_shards(&self) -> Vec<usize> {
        let snapshot = self.shard_loads_snapshot();
        if snapshot.is_empty() {
            return vec![];
        }
        let total: usize = snapshot.iter().sum();
        let n = snapshot.len();
        // Use integer arithmetic: a shard is a hot-spot when its load * n > total.
        snapshot
            .iter()
            .enumerate()
            .filter(|&(_, &load)| load * n > total)
            .map(|(idx, _)| idx)
            .collect()
    }

    /// Expose configuration for inspection.
    pub fn config(&self) -> &ShardConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// DhtShardRouter
// ---------------------------------------------------------------------------

/// Routes insert / search operations to the appropriate peers based on
/// which shards they host.
pub struct DhtShardRouter {
    /// Shared shard balancer.
    pub balancer: Arc<ShardBalancer>,
    /// Maps peer_id → set of shard indices the peer hosts.
    peer_shard_map: Arc<RwLock<HashMap<String, HashSet<usize>>>>,
}

impl DhtShardRouter {
    /// Create a new `DhtShardRouter` backed by the given `ShardBalancer`.
    pub fn new(balancer: Arc<ShardBalancer>) -> Self {
        Self {
            balancer,
            peer_shard_map: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register (or replace) the set of shards hosted by a peer.
    pub fn register_peer_shards(&self, peer_id: &str, shards: Vec<usize>) {
        let mut map = self
            .peer_shard_map
            .write()
            .expect("peer_shard_map write lock poisoned");
        map.insert(peer_id.to_string(), shards.into_iter().collect());
    }

    /// Remove a peer and all of its shard registrations.
    pub fn unregister_peer(&self, peer_id: &str) {
        let mut map = self
            .peer_shard_map
            .write()
            .expect("peer_shard_map write lock poisoned");
        map.remove(peer_id);
    }

    /// Return the list of peers that host the given shard.
    pub fn peers_for_shard(&self, shard_id: usize) -> Vec<String> {
        let map = self
            .peer_shard_map
            .read()
            .expect("peer_shard_map read lock poisoned");
        map.iter()
            .filter(|(_, shards)| shards.contains(&shard_id))
            .map(|(peer_id, _)| peer_id.clone())
            .collect()
    }

    /// Return a snapshot of all peer → shard assignments.
    pub fn all_peer_assignments(&self) -> HashMap<String, Vec<usize>> {
        let map = self
            .peer_shard_map
            .read()
            .expect("peer_shard_map read lock poisoned");
        map.iter()
            .map(|(peer_id, shards)| {
                let mut shard_vec: Vec<usize> = shards.iter().copied().collect();
                shard_vec.sort_unstable();
                (peer_id.clone(), shard_vec)
            })
            .collect()
    }

    /// Return a map of shard_id → number of peers that host that shard.
    pub fn shard_coverage(&self) -> HashMap<usize, usize> {
        let map = self
            .peer_shard_map
            .read()
            .expect("peer_shard_map read lock poisoned");
        let num_shards = self.balancer.config().num_shards;
        let mut coverage: HashMap<usize, usize> = (0..num_shards).map(|s| (s, 0)).collect();
        for shards in map.values() {
            for &shard_id in shards {
                *coverage.entry(shard_id).or_insert(0) += 1;
            }
        }
        coverage
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_balancer(num_shards: usize) -> Arc<ShardBalancer> {
        Arc::new(ShardBalancer::new(ShardConfig {
            num_shards,
            replication_factor: 3,
            max_vectors_per_shard: 200_000,
        }))
    }

    fn make_router(num_shards: usize) -> DhtShardRouter {
        DhtShardRouter::new(make_balancer(num_shards))
    }

    // ------------------------------------------------------------------
    // 1. Consistency
    // ------------------------------------------------------------------
    #[test]
    fn test_shard_assignment_consistency() {
        let balancer = make_balancer(8);
        for vector_id in [0u64, 1, 42, 999, u64::MAX / 2, u64::MAX] {
            let first = balancer.assign_vector(vector_id);
            // Call many times – must be identical every time.
            for _ in 0..100 {
                assert_eq!(
                    balancer.assign_vector(vector_id),
                    first,
                    "assign_vector({vector_id}) is not consistent"
                );
            }
        }
    }

    // ------------------------------------------------------------------
    // 2. Load tracking
    // ------------------------------------------------------------------
    #[test]
    fn test_shard_load_tracking() {
        let balancer = make_balancer(4);

        // All zero at start.
        assert_eq!(balancer.shard_loads_snapshot(), vec![0, 0, 0, 0]);

        balancer.increment_shard_load(0);
        balancer.increment_shard_load(0);
        balancer.increment_shard_load(1);

        let snap = balancer.shard_loads_snapshot();
        assert_eq!(snap[0], 2);
        assert_eq!(snap[1], 1);
        assert_eq!(snap[2], 0);
        assert_eq!(snap[3], 0);

        balancer.decrement_shard_load(0);
        assert_eq!(balancer.shard_loads_snapshot()[0], 1);

        // Saturates at zero.
        balancer.decrement_shard_load(2);
        assert_eq!(balancer.shard_loads_snapshot()[2], 0);
    }

    // ------------------------------------------------------------------
    // 3. Rebalance detection
    // ------------------------------------------------------------------
    #[test]
    fn test_rebalance_detection() {
        let balancer = make_balancer(4);

        // Balanced – no rebalance needed.
        for shard in 0..4 {
            for _ in 0..10 {
                balancer.increment_shard_load(shard);
            }
        }
        assert!(
            !balancer.rebalance_needed(),
            "balanced shards should not need rebalance"
        );

        // Overload shard 0 to cause a ratio > 2.
        for _ in 0..30 {
            balancer.increment_shard_load(0);
        }
        assert!(
            balancer.rebalance_needed(),
            "highly skewed shards should need rebalance"
        );
    }

    // ------------------------------------------------------------------
    // 4. Hotspot detection
    // ------------------------------------------------------------------
    #[test]
    fn test_hotspot_shards() {
        let balancer = make_balancer(4);

        // shard 0 = 100, shards 1-3 = 10 each → avg = 32.5 → shard 0 is hotspot.
        for _ in 0..100 {
            balancer.increment_shard_load(0);
        }
        for shard in 1..4 {
            for _ in 0..10 {
                balancer.increment_shard_load(shard);
            }
        }

        let hotspots = balancer.hotspot_shards();
        assert!(hotspots.contains(&0), "shard 0 should be a hotspot");
        assert!(!hotspots.contains(&1), "shard 1 should not be a hotspot");
    }

    // ------------------------------------------------------------------
    // 5. Router – peer registration
    // ------------------------------------------------------------------
    #[test]
    fn test_dht_shard_router_registration() {
        let router = make_router(8);

        router.register_peer_shards("peer-A", vec![0, 1, 2]);
        router.register_peer_shards("peer-B", vec![3, 4, 5]);

        let assignments = router.all_peer_assignments();
        assert_eq!(assignments["peer-A"], vec![0, 1, 2]);
        assert_eq!(assignments["peer-B"], vec![3, 4, 5]);

        // Re-register overwrites.
        router.register_peer_shards("peer-A", vec![0, 7]);
        let assignments2 = router.all_peer_assignments();
        assert_eq!(assignments2["peer-A"], vec![0, 7]);
    }

    // ------------------------------------------------------------------
    // 6. Peers for shard
    // ------------------------------------------------------------------
    #[test]
    fn test_peers_for_shard() {
        let router = make_router(8);

        router.register_peer_shards("peer-X", vec![0, 1, 2]);
        router.register_peer_shards("peer-Y", vec![1, 2, 3]);
        router.register_peer_shards("peer-Z", vec![4, 5, 6]);

        let mut peers_for_1 = router.peers_for_shard(1);
        peers_for_1.sort();
        assert_eq!(peers_for_1, vec!["peer-X", "peer-Y"]);

        let peers_for_4 = router.peers_for_shard(4);
        assert_eq!(peers_for_4, vec!["peer-Z"]);

        let peers_for_7 = router.peers_for_shard(7);
        assert!(peers_for_7.is_empty(), "no peer hosts shard 7");
    }

    // ------------------------------------------------------------------
    // 7. Shard coverage
    // ------------------------------------------------------------------
    #[test]
    fn test_shard_coverage() {
        let router = make_router(4);

        router.register_peer_shards("peer-1", vec![0, 1, 2, 3]);
        router.register_peer_shards("peer-2", vec![0, 1, 2, 3]);
        router.register_peer_shards("peer-3", vec![0, 2]);

        let coverage = router.shard_coverage();
        assert_eq!(coverage[&0], 3);
        assert_eq!(coverage[&1], 2);
        assert_eq!(coverage[&2], 3);
        assert_eq!(coverage[&3], 2);
    }

    // ------------------------------------------------------------------
    // 8. Least loaded shard
    // ------------------------------------------------------------------
    #[test]
    fn test_least_loaded_shard() {
        let balancer = make_balancer(4);

        // All zeros → shard 0 wins (lowest index tie-break).
        assert_eq!(balancer.least_loaded_shard(), 0);

        balancer.increment_shard_load(0);
        balancer.increment_shard_load(0);
        balancer.increment_shard_load(1);

        // shard 2 and 3 are empty → shard 2 wins tie-break.
        assert_eq!(balancer.least_loaded_shard(), 2);

        balancer.increment_shard_load(2);
        balancer.increment_shard_load(2);
        balancer.increment_shard_load(2);

        // shard 3 is now the only zero.
        assert_eq!(balancer.least_loaded_shard(), 3);
    }

    // ------------------------------------------------------------------
    // 9. Consistent-hash distribution
    // ------------------------------------------------------------------
    #[test]
    fn test_consistent_hash_distribution() {
        let balancer = make_balancer(8);
        let mut counts = [0usize; 8];

        for id in 0u64..1000 {
            let shard = balancer.assign_vector(id);
            counts[shard] += 1;
        }

        let max = *counts.iter().max().expect("non-empty");
        let min = *counts.iter().min().expect("non-empty");
        // Require no shard is completely empty.
        assert!(
            min > 0,
            "every shard should receive at least one vector from 1000 IDs"
        );
        // Require the max/min ratio is less than 3.0 for good hashing.
        assert!(
            (max as f64 / min as f64) < 3.0,
            "hash distribution too skewed: max={max}, min={min}"
        );
    }

    // ------------------------------------------------------------------
    // 10. Unregister removes peer
    // ------------------------------------------------------------------
    #[test]
    fn test_unregister_removes_peer() {
        let router = make_router(8);

        router.register_peer_shards("peer-alpha", vec![0, 1, 2, 3]);
        router.register_peer_shards("peer-beta", vec![0, 1]);

        // Verify peer-alpha is present.
        assert!(router
            .peers_for_shard(0)
            .contains(&"peer-alpha".to_string()));

        // Unregister.
        router.unregister_peer("peer-alpha");

        // Should no longer appear for any shard.
        for shard in 0..8 {
            let peers = router.peers_for_shard(shard);
            assert!(
                !peers.contains(&"peer-alpha".to_string()),
                "peer-alpha still appears for shard {shard} after unregistration"
            );
        }

        // peer-beta should be unaffected.
        assert!(router.peers_for_shard(0).contains(&"peer-beta".to_string()));
    }
}
