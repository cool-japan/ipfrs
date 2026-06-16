//! Shard Coordinator — Consistent-Hash Distribution of Vectors Across Nodes
//!
//! At 1M+ vectors a single HNSW index is too large to fit on one node.
//! This module distributes vectors across logical *shards* using a consistent
//! hash ring so that:
//!
//! * Every `vector_id` maps deterministically to the same shard across the
//!   cluster without any central lookup table.
//! * Adding or removing a shard only re-hashes a minimal fraction of the
//!   keyspace (the usual consistent-hashing guarantee).
//! * The coordinator detects when shards are imbalanced and surfaces which
//!   shards are over- or under-loaded so the operator (or an auto-scaler) can
//!   trigger a rebalance.
//!
//! # Design Notes
//!
//! The hash ring uses **FNV-1a** (64-bit) because it is fast, has no
//! dependencies, and distributes keys uniformly for short byte strings.
//! Each physical shard is given `virtual_nodes` (default 150) positions on
//! the ring, which provides the load balance guarantee of consistent hashing.
//!
//! All mutable state inside [`ShardCoordinator`] is protected by
//! `std::sync::RwLock` so the struct is `Send + Sync` and can be shared
//! freely across async tasks.

use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

use thiserror::Error;

// ---------------------------------------------------------------------------
// Public Error type
// ---------------------------------------------------------------------------

/// Errors produced by the shard coordinator.
#[derive(Debug, Error)]
pub enum ShardError {
    /// No shard with the given numeric ID is registered.
    #[error("shard {0} not found")]
    ShardNotFound(u32),

    /// The target shard has reached its maximum capacity.
    #[error("shard {shard_id} is at capacity ({capacity} vectors)")]
    ShardAtCapacity {
        /// The numeric shard ID that is full.
        shard_id: u32,
        /// The capacity limit that was reached.
        capacity: u64,
    },
}

// ---------------------------------------------------------------------------
// ShardId newtype
// ---------------------------------------------------------------------------

/// Opaque identifier for a logical shard.
///
/// Internally a `u32`, but exposed as a newtype so callers cannot accidentally
/// mix raw integers with shard IDs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ShardId(pub u32);

impl fmt::Display for ShardId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "shard-{}", self.0)
    }
}

impl From<u32> for ShardId {
    fn from(v: u32) -> Self {
        ShardId(v)
    }
}

// ---------------------------------------------------------------------------
// VectorShard — per-shard metadata
// ---------------------------------------------------------------------------

/// Metadata and load information for one logical shard.
#[derive(Debug, Clone)]
pub struct VectorShard {
    /// Logical shard identifier.
    pub shard_id: ShardId,
    /// Network address / peer ID of the node that owns this shard.
    pub peer_id: String,
    /// Current number of vectors stored in this shard.
    pub vector_count: u64,
    /// Maximum number of vectors before this shard is considered full.
    pub capacity: u64,
    /// Embedding dimensionality stored in this shard.
    pub dimensions: u32,
}

impl VectorShard {
    /// Create a new shard with default capacity (`100_000`).
    pub fn new(shard_id: ShardId, peer_id: impl Into<String>, dimensions: u32) -> Self {
        Self {
            shard_id,
            peer_id: peer_id.into(),
            vector_count: 0,
            capacity: 100_000,
            dimensions,
        }
    }

    /// Fraction of capacity currently in use, in the range `[0.0, ∞)`.
    ///
    /// Values above `1.0` mean the shard has exceeded its configured capacity.
    pub fn utilization(&self) -> f64 {
        if self.capacity == 0 {
            return 0.0;
        }
        self.vector_count as f64 / self.capacity as f64
    }
}

// ---------------------------------------------------------------------------
// FNV-1a hash helpers
// ---------------------------------------------------------------------------

/// 64-bit FNV-1a offset basis.
const FNV_OFFSET_BASIS: u64 = 14_695_981_039_346_656_037;
/// FNV-1a prime.
const FNV_PRIME: u64 = 1_099_511_628_211;

/// Compute the FNV-1a 64-bit hash of a byte slice.
#[inline]
fn fnv1a_64(data: &[u8]) -> u64 {
    let mut hash = FNV_OFFSET_BASIS;
    for &byte in data {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Derive a deterministic virtual-node key from a shard ID and a replica index.
///
/// The key is `"shard-{shard_id}#{replica}"` encoded as UTF-8 bytes.
#[inline]
fn virtual_node_key(shard_id: ShardId, replica: usize) -> u64 {
    let label = format!("shard-{}#{}", shard_id.0, replica);
    fnv1a_64(label.as_bytes())
}

// ---------------------------------------------------------------------------
// ConsistentHashRing
// ---------------------------------------------------------------------------

/// A consistent-hash ring mapping arbitrary byte keys to [`ShardId`]s.
///
/// Each shard occupies `virtual_nodes` (default 150) positions on the ring,
/// providing excellent key distribution even with a small number of shards.
///
/// ## Lookup algorithm
///
/// 1. Hash the key with FNV-1a.
/// 2. Find the first ring position ≥ the hash (wrap-around to the minimum
///    position if none exists — standard consistent-hashing).
/// 3. Return the [`ShardId`] at that ring position.
#[derive(Debug, Clone)]
pub struct ConsistentHashRing {
    /// Sorted map: ring position (FNV-1a hash) → shard ID.
    ring: BTreeMap<u64, ShardId>,
    /// Number of virtual-node positions per physical shard.
    virtual_nodes: usize,
}

impl Default for ConsistentHashRing {
    fn default() -> Self {
        Self {
            ring: BTreeMap::new(),
            virtual_nodes: 150,
        }
    }
}

impl ConsistentHashRing {
    /// Create a new ring with the given number of virtual nodes per shard.
    pub fn new(virtual_nodes: usize) -> Self {
        Self {
            ring: BTreeMap::new(),
            virtual_nodes,
        }
    }

    /// Add `shard_id` to the ring, placing `virtual_nodes` replicas.
    ///
    /// `peer_id` is accepted for future extensibility (e.g., zone-aware
    /// placement) but is not stored inside the ring itself — it lives in
    /// [`VectorShard`].
    pub fn add_shard(&mut self, shard_id: ShardId, _peer_id: &str) {
        for replica in 0..self.virtual_nodes {
            let position = virtual_node_key(shard_id, replica);
            self.ring.insert(position, shard_id);
        }
    }

    /// Remove `shard_id` from the ring, deleting all its virtual nodes.
    pub fn remove_shard(&mut self, shard_id: ShardId) {
        for replica in 0..self.virtual_nodes {
            let position = virtual_node_key(shard_id, replica);
            self.ring.remove(&position);
        }
    }

    /// Find the shard responsible for `key`.
    ///
    /// Returns `None` only when the ring is empty.
    pub fn get_shard(&self, key: &[u8]) -> Option<ShardId> {
        if self.ring.is_empty() {
            return None;
        }
        let hash = fnv1a_64(key);
        // Walk clockwise from `hash` — wrap around to the minimum if needed.
        self.ring
            .range(hash..)
            .next()
            .or_else(|| self.ring.iter().next())
            .map(|(_, &shard)| shard)
    }

    /// Number of *distinct* shards currently on the ring.
    pub fn shard_count(&self) -> usize {
        // Collect the set of unique shard IDs.
        let mut seen = std::collections::HashSet::new();
        for shard_id in self.ring.values() {
            seen.insert(*shard_id);
        }
        seen.len()
    }
}

// ---------------------------------------------------------------------------
// ShardStats — lock-free counters
// ---------------------------------------------------------------------------

/// Atomic counters tracking lifetime activity of the coordinator.
#[derive(Debug, Default)]
pub struct ShardStats {
    /// Total number of vector → shard assignments performed.
    pub total_assignments: AtomicU64,
    /// Number of times [`ShardCoordinator::needs_rebalance`] returned `true`.
    pub total_rebalances_triggered: AtomicU64,
    /// Total number of shards registered via [`ShardCoordinator::register_shard`].
    pub total_shards_registered: AtomicU64,
}

/// A point-in-time snapshot of [`ShardStats`], for easy display / serialisation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShardStatsSnapshot {
    /// Total assignments at snapshot time.
    pub total_assignments: u64,
    /// Total rebalances triggered at snapshot time.
    pub total_rebalances_triggered: u64,
    /// Total shards registered at snapshot time.
    pub total_shards_registered: u64,
}

impl ShardStats {
    /// Take a consistent snapshot (using `SeqCst` loads).
    pub fn snapshot(&self) -> ShardStatsSnapshot {
        ShardStatsSnapshot {
            total_assignments: self.total_assignments.load(Ordering::SeqCst),
            total_rebalances_triggered: self.total_rebalances_triggered.load(Ordering::SeqCst),
            total_shards_registered: self.total_shards_registered.load(Ordering::SeqCst),
        }
    }
}

// ---------------------------------------------------------------------------
// ShardCoordinator — main entry-point
// ---------------------------------------------------------------------------

/// Coordinates the distribution of vectors across a cluster of shards.
///
/// # Thread safety
///
/// [`ShardCoordinator`] wraps its mutable state in `std::sync::RwLock` and
/// exposes only shared references (`&self`) from every public method.  It can
/// therefore be placed in an `Arc` and shared freely across async tasks:
///
/// ```rust,ignore
/// let coord = Arc::new(ShardCoordinator::new(0.2));
/// ```
pub struct ShardCoordinator {
    /// Map of numeric shard ID → shard metadata.
    shards: RwLock<HashMap<u32, VectorShard>>,
    /// Consistent hash ring.
    ring: RwLock<ConsistentHashRing>,
    /// Maximum allowed deviation from the mean utilization before a rebalance
    /// is flagged.  For example `0.2` means 20%.
    rebalance_threshold: f64,
    /// Lifetime statistics (lock-free).
    pub stats: Arc<ShardStats>,
}

impl ShardCoordinator {
    /// Create a coordinator with a custom rebalance threshold.
    ///
    /// `rebalance_threshold` is a fraction in `(0, 1)`.  A typical value is
    /// `0.2` (flag rebalance when any shard deviates > 20% from the mean).
    pub fn new(rebalance_threshold: f64) -> Self {
        Self {
            shards: RwLock::new(HashMap::new()),
            ring: RwLock::new(ConsistentHashRing::default()),
            rebalance_threshold,
            stats: Arc::new(ShardStats::default()),
        }
    }

    /// Create a coordinator with the default rebalance threshold of `0.2`.
    pub fn with_defaults() -> Self {
        Self::new(0.2)
    }

    // -----------------------------------------------------------------------
    // Shard lifecycle
    // -----------------------------------------------------------------------

    /// Register a new shard with the coordinator.
    ///
    /// This adds the shard to both the metadata map and the consistent hash
    /// ring.  If a shard with the same ID is already registered it is replaced.
    pub fn register_shard(&self, shard: VectorShard) {
        let shard_id = shard.shard_id;
        let peer_id = shard.peer_id.clone();
        {
            let mut shards = self
                .shards
                .write()
                .expect("shard registry write lock poisoned");
            shards.insert(shard_id.0, shard);
        }
        {
            let mut ring = self.ring.write().expect("ring write lock poisoned");
            ring.add_shard(shard_id, &peer_id);
        }
        self.stats
            .total_shards_registered
            .fetch_add(1, Ordering::Relaxed);
    }

    // -----------------------------------------------------------------------
    // Assignment
    // -----------------------------------------------------------------------

    /// Assign a vector to a shard using consistent hashing on `vector_id`.
    ///
    /// Returns `None` only when no shards have been registered yet.
    pub fn assign_vector(&self, vector_id: &str) -> Option<ShardId> {
        let ring = self.ring.read().expect("ring read lock poisoned");
        let result = ring.get_shard(vector_id.as_bytes());
        drop(ring);
        if result.is_some() {
            self.stats.total_assignments.fetch_add(1, Ordering::Relaxed);
        }
        result
    }

    /// Increment the vector counter for the given shard.
    ///
    /// # Errors
    ///
    /// * [`ShardError::ShardNotFound`] — the shard ID is not registered.
    /// * [`ShardError::ShardAtCapacity`] — the shard is already at its capacity
    ///   limit.
    pub fn increment_shard_count(&self, shard_id: ShardId) -> Result<(), ShardError> {
        let mut shards = self
            .shards
            .write()
            .expect("shard registry write lock poisoned");
        match shards.get_mut(&shard_id.0) {
            None => Err(ShardError::ShardNotFound(shard_id.0)),
            Some(shard) => {
                if shard.vector_count >= shard.capacity {
                    Err(ShardError::ShardAtCapacity {
                        shard_id: shard_id.0,
                        capacity: shard.capacity,
                    })
                } else {
                    shard.vector_count += 1;
                    Ok(())
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Rebalance detection
    // -----------------------------------------------------------------------

    /// Return `true` when at least one shard's utilization deviates from the
    /// mean by more than `rebalance_threshold`.
    ///
    /// When this returns `true` the `total_rebalances_triggered` counter is
    /// incremented.
    pub fn needs_rebalance(&self) -> bool {
        let shards = self
            .shards
            .read()
            .expect("shard registry read lock poisoned");
        if shards.len() < 2 {
            return false;
        }
        let utils: Vec<f64> = shards.values().map(|s| s.utilization()).collect();
        let mean = utils.iter().sum::<f64>() / utils.len() as f64;
        let diverges = utils
            .iter()
            .any(|&u| (u - mean).abs() > self.rebalance_threshold);
        if diverges {
            self.stats
                .total_rebalances_triggered
                .fetch_add(1, Ordering::Relaxed);
        }
        diverges
    }

    /// Return the IDs of shards whose utilization exceeds `mean + threshold`.
    pub fn overloaded_shards(&self) -> Vec<ShardId> {
        let shards = self
            .shards
            .read()
            .expect("shard registry read lock poisoned");
        if shards.is_empty() {
            return Vec::new();
        }
        let utils: Vec<(ShardId, f64)> = shards
            .values()
            .map(|s| (s.shard_id, s.utilization()))
            .collect();
        let mean = utils.iter().map(|(_, u)| u).sum::<f64>() / utils.len() as f64;
        utils
            .into_iter()
            .filter(|(_, u)| *u > mean + self.rebalance_threshold)
            .map(|(id, _)| id)
            .collect()
    }

    /// Return the IDs of shards whose utilization is below `mean - threshold`.
    pub fn underloaded_shards(&self) -> Vec<ShardId> {
        let shards = self
            .shards
            .read()
            .expect("shard registry read lock poisoned");
        if shards.is_empty() {
            return Vec::new();
        }
        let utils: Vec<(ShardId, f64)> = shards
            .values()
            .map(|s| (s.shard_id, s.utilization()))
            .collect();
        let mean = utils.iter().map(|(_, u)| u).sum::<f64>() / utils.len() as f64;
        utils
            .into_iter()
            .filter(|(_, u)| *u < mean - self.rebalance_threshold)
            .map(|(id, _)| id)
            .collect()
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /// Number of shards currently registered.
    pub fn shard_count(&self) -> usize {
        self.shards
            .read()
            .expect("shard registry read lock poisoned")
            .len()
    }

    /// Total vectors stored across all shards.
    pub fn total_vectors(&self) -> u64 {
        self.shards
            .read()
            .expect("shard registry read lock poisoned")
            .values()
            .map(|s| s.vector_count)
            .sum()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: build a coordinator pre-populated with `n` identical shards.
    fn make_coordinator_with_shards(n: u32, capacity: u64) -> ShardCoordinator {
        let coord = ShardCoordinator::with_defaults();
        for i in 0..n {
            let mut shard = VectorShard::new(ShardId(i), format!("peer-{}", i), 128);
            shard.capacity = capacity;
            coord.register_shard(shard);
        }
        coord
    }

    // -----------------------------------------------------------------------
    // ShardId
    // -----------------------------------------------------------------------

    #[test]
    fn test_shard_id_display() {
        let id = ShardId(42);
        assert_eq!(id.to_string(), "shard-42");
    }

    #[test]
    fn test_shard_id_from_u32() {
        let id: ShardId = 7_u32.into();
        assert_eq!(id.0, 7);
    }

    // -----------------------------------------------------------------------
    // VectorShard
    // -----------------------------------------------------------------------

    #[test]
    fn test_vector_shard_utilization() {
        let mut shard = VectorShard::new(ShardId(0), "peer-0", 128);
        shard.vector_count = 50_000;
        // 50_000 / 100_000 = 0.5
        let util = shard.utilization();
        assert!((util - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_vector_shard_utilization_zero_capacity() {
        let mut shard = VectorShard::new(ShardId(0), "peer-0", 128);
        shard.capacity = 0;
        assert_eq!(shard.utilization(), 0.0);
    }

    // -----------------------------------------------------------------------
    // ConsistentHashRing — basic operations
    // -----------------------------------------------------------------------

    #[test]
    fn test_ring_empty_returns_none() {
        let ring = ConsistentHashRing::default();
        assert!(ring.get_shard(b"anything").is_none());
    }

    #[test]
    fn test_ring_deterministic_assignment() {
        let mut ring = ConsistentHashRing::default();
        ring.add_shard(ShardId(0), "peer-0");
        ring.add_shard(ShardId(1), "peer-1");
        ring.add_shard(ShardId(2), "peer-2");

        let key = b"vector-12345";
        let first = ring.get_shard(key).expect("ring is not empty");
        // Calling get_shard again must return the exact same shard.
        for _ in 0..50 {
            assert_eq!(ring.get_shard(key), Some(first));
        }
    }

    #[test]
    fn test_ring_same_key_same_shard_after_rebuild() {
        let mut ring1 = ConsistentHashRing::new(150);
        ring1.add_shard(ShardId(10), "peer-10");
        ring1.add_shard(ShardId(20), "peer-20");
        let key = b"stable-key";
        let shard1 = ring1.get_shard(key);

        // Build an identical ring independently.
        let mut ring2 = ConsistentHashRing::new(150);
        ring2.add_shard(ShardId(10), "peer-10");
        ring2.add_shard(ShardId(20), "peer-20");
        let shard2 = ring2.get_shard(key);

        assert_eq!(shard1, shard2);
    }

    #[test]
    fn test_ring_remove_shard_redistributes() {
        let mut ring = ConsistentHashRing::default();
        ring.add_shard(ShardId(0), "peer-0");
        ring.add_shard(ShardId(1), "peer-1");

        // Collect 200 keys and their assignments before removal.
        let keys: Vec<Vec<u8>> = (0_u64..200)
            .map(|i| format!("key-{}", i).into_bytes())
            .collect();
        let before: Vec<ShardId> = keys
            .iter()
            .map(|k| {
                ring.get_shard(k)
                    .expect("test: ring is non-empty before removal")
            })
            .collect();

        // Remove shard 1.
        ring.remove_shard(ShardId(1));
        assert_eq!(ring.shard_count(), 1);

        // Every key should now map to shard 0.
        let after: Vec<ShardId> = keys
            .iter()
            .map(|k| {
                ring.get_shard(k)
                    .expect("test: ring still has shard 0 after removing shard 1")
            })
            .collect();

        for a in &after {
            assert_eq!(*a, ShardId(0));
        }

        // At least some keys must have changed shard.
        let changed = before
            .iter()
            .zip(after.iter())
            .filter(|(b, a)| b != a)
            .count();
        assert!(changed > 0, "expected some keys to be redistributed");
    }

    #[test]
    fn test_ring_shard_count() {
        let mut ring = ConsistentHashRing::default();
        assert_eq!(ring.shard_count(), 0);
        ring.add_shard(ShardId(0), "peer-0");
        assert_eq!(ring.shard_count(), 1);
        ring.add_shard(ShardId(1), "peer-1");
        assert_eq!(ring.shard_count(), 2);
        ring.remove_shard(ShardId(0));
        assert_eq!(ring.shard_count(), 1);
    }

    // -----------------------------------------------------------------------
    // ShardCoordinator — registration and assignment
    // -----------------------------------------------------------------------

    #[test]
    fn test_register_shard_and_assign_vector() {
        let coord = make_coordinator_with_shards(3, 100_000);
        assert_eq!(coord.shard_count(), 3);

        let shard = coord.assign_vector("my-vector-id");
        assert!(shard.is_some());
    }

    #[test]
    fn test_assign_vector_deterministic() {
        let coord = make_coordinator_with_shards(4, 100_000);
        let id = "deterministic-vector";
        let first = coord
            .assign_vector(id)
            .expect("test: coordinator has 4 shards so assignment cannot be None");
        for _ in 0..20 {
            assert_eq!(coord.assign_vector(id), Some(first));
        }
    }

    #[test]
    fn test_assign_vector_no_shards_returns_none() {
        let coord = ShardCoordinator::with_defaults();
        assert!(coord.assign_vector("v").is_none());
    }

    // -----------------------------------------------------------------------
    // increment_shard_count
    // -----------------------------------------------------------------------

    #[test]
    fn test_increment_shard_count_success() {
        let coord = make_coordinator_with_shards(2, 100_000);
        let result = coord.increment_shard_count(ShardId(0));
        assert!(result.is_ok());
        assert_eq!(coord.total_vectors(), 1);
    }

    #[test]
    fn test_increment_shard_count_not_found() {
        let coord = make_coordinator_with_shards(1, 100_000);
        let err = coord
            .increment_shard_count(ShardId(99))
            .expect_err("test: ShardId(99) is not registered so error is expected");
        assert!(matches!(err, ShardError::ShardNotFound(99)));
    }

    #[test]
    fn test_increment_shard_count_at_capacity() {
        // capacity = 2 so we can add exactly 2 vectors.
        let coord = make_coordinator_with_shards(1, 2);
        coord
            .increment_shard_count(ShardId(0))
            .expect("test: first increment is within capacity of 2");
        coord
            .increment_shard_count(ShardId(0))
            .expect("test: second increment is within capacity of 2");
        let err = coord
            .increment_shard_count(ShardId(0))
            .expect_err("test: third increment exceeds capacity of 2 so error is expected");
        assert!(
            matches!(
                err,
                ShardError::ShardAtCapacity {
                    shard_id: 0,
                    capacity: 2,
                }
            ),
            "expected ShardAtCapacity, got {:?}",
            err
        );
    }

    // -----------------------------------------------------------------------
    // needs_rebalance / overloaded / underloaded
    // -----------------------------------------------------------------------

    #[test]
    fn test_needs_rebalance_balanced() {
        let coord = make_coordinator_with_shards(3, 100_000);
        // All shards at 0 → perfectly balanced.
        assert!(!coord.needs_rebalance());
    }

    #[test]
    fn test_needs_rebalance_detects_imbalance() {
        let coord = ShardCoordinator::with_defaults();

        // Shard 0 heavily loaded, shard 1 empty.
        let mut s0 = VectorShard::new(ShardId(0), "peer-0", 128);
        s0.vector_count = 90_000;
        s0.capacity = 100_000;
        let s1 = VectorShard::new(ShardId(1), "peer-1", 128);
        // s1.vector_count = 0

        coord.register_shard(s0);
        coord.register_shard(s1);

        // Mean utilization ≈ (0.9 + 0.0) / 2 = 0.45
        // Shard 0 deviation: 0.9 - 0.45 = 0.45 > 0.2 threshold.
        assert!(coord.needs_rebalance());
    }

    #[test]
    fn test_overloaded_and_underloaded_shards() {
        let coord = ShardCoordinator::new(0.2);

        // Shard A: very full (0.9 utilization).
        let mut s_a = VectorShard::new(ShardId(0), "peer-0", 128);
        s_a.vector_count = 90_000;
        s_a.capacity = 100_000;

        // Shard B: medium (0.5 utilization) — will be the mean.
        let mut s_b = VectorShard::new(ShardId(1), "peer-1", 128);
        s_b.vector_count = 50_000;
        s_b.capacity = 100_000;

        // Shard C: almost empty (0.1 utilization).
        let mut s_c = VectorShard::new(ShardId(2), "peer-2", 128);
        s_c.vector_count = 10_000;
        s_c.capacity = 100_000;

        coord.register_shard(s_a);
        coord.register_shard(s_b);
        coord.register_shard(s_c);

        // mean ≈ (0.9 + 0.5 + 0.1) / 3 ≈ 0.5
        let overloaded = coord.overloaded_shards();
        let underloaded = coord.underloaded_shards();

        // Shard 0 is 0.4 above mean → overloaded.
        assert!(
            overloaded.contains(&ShardId(0)),
            "shard 0 should be overloaded"
        );
        // Shard 2 is 0.4 below mean → underloaded.
        assert!(
            underloaded.contains(&ShardId(2)),
            "shard 2 should be underloaded"
        );
        // Shard 1 is exactly at the mean — neither.
        assert!(!overloaded.contains(&ShardId(1)));
        assert!(!underloaded.contains(&ShardId(1)));
    }

    // -----------------------------------------------------------------------
    // Stats accumulation
    // -----------------------------------------------------------------------

    #[test]
    fn test_stats_accumulation() {
        let coord = make_coordinator_with_shards(3, 100_000);

        // 3 shards registered.
        assert_eq!(coord.stats.snapshot().total_shards_registered, 3);

        // 5 assignments.
        for i in 0..5 {
            coord.assign_vector(&format!("v-{}", i));
        }
        assert_eq!(coord.stats.snapshot().total_assignments, 5);

        // Trigger a rebalance by making shard 0 heavily loaded.
        {
            let mut shards = coord.shards.write().unwrap_or_else(|e| e.into_inner());
            if let Some(s) = shards.get_mut(&0) {
                s.vector_count = 90_000;
            }
        }
        coord.needs_rebalance();
        assert!(coord.stats.snapshot().total_rebalances_triggered >= 1);
    }

    // -----------------------------------------------------------------------
    // Virtual nodes provide balance
    // -----------------------------------------------------------------------

    #[test]
    fn test_virtual_nodes_balance() {
        // With 5 shards and 300 virtual nodes (1500 ring positions), a large
        // and diverse workload of 50_000 keys should give each shard roughly
        // 10_000 ± 50% keys.  Consistent hashing is *not* perfectly uniform
        // at small scale, but should be significantly better than putting all
        // keys on one shard.  We verify:
        //  1. Every shard receives at least one key.
        //  2. No single shard receives more than 60% of all keys (no runaway hot-spot).
        //  3. No single shard receives fewer than 5% of all keys (no starved cold-spot).
        let n_shards = 5_u32;
        let mut ring = ConsistentHashRing::new(300);
        for i in 0..n_shards {
            ring.add_shard(ShardId(i), &format!("peer-{}", i));
        }

        let n_keys = 50_000_usize;
        let mut counts: HashMap<ShardId, usize> = HashMap::new();
        for i in 0..n_keys {
            // Mix of numeric and string-like keys for good coverage.
            let key = format!("balance-test-key-{:08}", i);
            let shard = ring
                .get_shard(key.as_bytes())
                .expect("test: ring has 5 shards so lookup always returns Some");
            *counts.entry(shard).or_insert(0) += 1;
        }

        // All shards must be reached.
        assert_eq!(
            counts.len(),
            n_shards as usize,
            "every shard must receive at least one key"
        );

        let upper_bound = (n_keys as f64 * 0.60) as usize;
        let lower_bound = (n_keys as f64 * 0.05) as usize;

        for (shard_id, count) in &counts {
            assert!(
                *count <= upper_bound,
                "shard {:?} received {} keys — hot-spot detected (> 60% of {})",
                shard_id,
                count,
                n_keys
            );
            assert!(
                *count >= lower_bound,
                "shard {:?} received {} keys — starved (< 5% of {})",
                shard_id,
                count,
                n_keys
            );
        }
    }

    // -----------------------------------------------------------------------
    // total_vectors
    // -----------------------------------------------------------------------

    #[test]
    fn test_total_vectors() {
        let coord = make_coordinator_with_shards(3, 100_000);
        assert_eq!(coord.total_vectors(), 0);
        coord
            .increment_shard_count(ShardId(0))
            .expect("test: shard 0 exists and has capacity");
        coord
            .increment_shard_count(ShardId(1))
            .expect("test: shard 1 exists and has capacity");
        coord
            .increment_shard_count(ShardId(1))
            .expect("test: shard 1 still has capacity for a second vector");
        assert_eq!(coord.total_vectors(), 3);
    }
}
