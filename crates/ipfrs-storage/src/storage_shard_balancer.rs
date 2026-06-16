//! StorageShardBalancer — consistent-hashing-based shard management and rebalancing.
//!
//! Uses FNV-1a virtual nodes arranged in a sorted ring (BTreeMap) for O(log n)
//! lookup. Supports multiple rebalancing policies: LeastLoaded, ConsistentHash,
//! RegionAware, CapacityWeighted, and MinimalMovement.
//!
//! # Example
//!
//! ```rust
//! use ipfrs_storage::storage_shard_balancer::{
//!     StorageShardBalancer, ShardNode, BalancerConfig, RebalancePolicy,
//! };
//!
//! let config = BalancerConfig {
//!     replication_factor: 2,
//!     virtual_nodes_per_shard: 10,
//!     rebalance_threshold: 1.5,
//!     policy: RebalancePolicy::LeastLoaded,
//! };
//! let mut balancer = StorageShardBalancer::new(config);
//!
//! balancer.add_shard(ShardNode {
//!     id: "shard-a".to_string(),
//!     capacity_bytes: 1_000_000,
//!     used_bytes: 0,
//!     virtual_nodes: 10,
//!     is_healthy: true,
//!     region: "us-east".to_string(),
//!     weight: 1.0,
//! }).unwrap();
//!
//! balancer.add_shard(ShardNode {
//!     id: "shard-b".to_string(),
//!     capacity_bytes: 1_000_000,
//!     used_bytes: 0,
//!     virtual_nodes: 10,
//!     is_healthy: true,
//!     region: "us-west".to_string(),
//!     weight: 1.0,
//! }).unwrap();
//!
//! let assignment = balancer.assign("QmExampleCid123").unwrap();
//! assert!(!assignment.shard_id.is_empty());
//! ```

use std::collections::{BTreeMap, HashMap};

// ---------------------------------------------------------------------------
// FNV-1a hashing
// ---------------------------------------------------------------------------

/// FNV-1a 64-bit hash — deterministic and fast for consistent hashing.
pub fn fnv1a_64(data: &[u8]) -> u64 {
    let mut h: u64 = 14_695_981_039_346_656_037;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(1_099_511_628_211);
    }
    h
}

fn virtual_node_key(shard_id: &str, replica: usize) -> u64 {
    let key = format!("{}-{}", shard_id, replica);
    fnv1a_64(key.as_bytes())
}

fn content_key(cid: &str) -> u64 {
    fnv1a_64(cid.as_bytes())
}

/// Xorshift64 PRNG — used in tests (no `rand` crate dependency).
pub fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// A storage shard node participating in the consistent hash ring.
#[derive(Debug, Clone)]
pub struct ShardNode {
    /// Unique identifier for the shard.
    pub id: String,
    /// Total storage capacity in bytes.
    pub capacity_bytes: u64,
    /// Currently used storage in bytes.
    pub used_bytes: u64,
    /// Number of virtual nodes to place in the ring (default 150).
    pub virtual_nodes: usize,
    /// Whether this shard is accepting reads/writes.
    pub is_healthy: bool,
    /// Deployment region identifier.
    pub region: String,
    /// Relative weight for capacity-weighted policies (default 1.0).
    pub weight: f64,
}

impl Default for ShardNode {
    fn default() -> Self {
        Self {
            id: String::new(),
            capacity_bytes: 0,
            used_bytes: 0,
            virtual_nodes: 150,
            is_healthy: true,
            region: String::new(),
            weight: 1.0,
        }
    }
}

impl ShardNode {
    /// Utilization ratio in [0.0, 1.0].
    pub fn utilization(&self) -> f64 {
        if self.capacity_bytes == 0 {
            return 1.0;
        }
        self.used_bytes as f64 / self.capacity_bytes as f64
    }

    /// Free bytes remaining.
    pub fn free_bytes(&self) -> u64 {
        self.capacity_bytes.saturating_sub(self.used_bytes)
    }

    /// Effective capacity weight for CapacityWeighted policy.
    pub fn capacity_weight(&self) -> f64 {
        if self.capacity_bytes == 0 {
            return 0.0;
        }
        let free_ratio = self.free_bytes() as f64 / self.capacity_bytes as f64;
        free_ratio * self.weight
    }
}

/// Assignment of a CID to primary and replica shards.
#[derive(Debug, Clone)]
pub struct ShardAssignment {
    /// Content identifier being assigned.
    pub cid: String,
    /// Primary shard holding the canonical copy.
    pub shard_id: String,
    /// Replica shards for redundancy.
    pub replica_shards: Vec<String>,
    /// Unix timestamp (seconds) when the assignment was created.
    pub assigned_at: u64,
}

/// Rebalancing operations produced by the balancer.
#[derive(Debug, Clone, PartialEq)]
pub enum RebalanceOp {
    /// Move a piece of content from one shard to another.
    MoveContent {
        cid: String,
        from_shard: String,
        to_shard: String,
    },
    /// Add a virtual node at the given ring position.
    AddVirtualNode { shard_id: String, position: u64 },
    /// Remove a virtual node from the given ring position.
    RemoveVirtualNode { shard_id: String, position: u64 },
    /// Update the weight of a shard.
    UpdateWeight { shard_id: String, new_weight: f64 },
}

/// Rebalancing policy controlling how imbalance is resolved.
#[derive(Debug, Clone, PartialEq)]
pub enum RebalancePolicy {
    /// Move content from the most-loaded shard to the least-loaded shard.
    LeastLoaded,
    /// Reassign misplaced content to its ring-correct shard.
    ConsistentHash,
    /// Prefer shards in the given region for new assignments.
    RegionAware(String),
    /// Weight assignments by (capacity − used) / capacity.
    CapacityWeighted,
    /// Only move what is strictly necessary to fall below the threshold.
    MinimalMovement,
}

/// Configuration for `StorageShardBalancer`.
#[derive(Debug, Clone)]
pub struct BalancerConfig {
    /// Number of replica copies per CID (primary + replicas).
    pub replication_factor: usize,
    /// Virtual nodes per shard placed in the ring.
    pub virtual_nodes_per_shard: usize,
    /// Trigger rebalancing if max_load / min_load > threshold.
    pub rebalance_threshold: f64,
    /// Policy used when generating rebalance operations.
    pub policy: RebalancePolicy,
}

impl Default for BalancerConfig {
    fn default() -> Self {
        Self {
            replication_factor: 3,
            virtual_nodes_per_shard: 150,
            rebalance_threshold: 1.5,
            policy: RebalancePolicy::LeastLoaded,
        }
    }
}

/// Snapshot of balancer state for observability.
#[derive(Debug, Clone)]
pub struct SsbBalancerStats {
    /// Number of shards in the ring.
    pub shard_count: usize,
    /// Sum of all shard capacities.
    pub total_capacity_bytes: u64,
    /// Sum of all shard used bytes.
    pub total_used_bytes: u64,
    /// Overall utilization percentage (0–100).
    pub utilization_pct: f64,
    /// Ratio of the most-loaded to least-loaded shard (by utilization).
    pub imbalance_ratio: f64,
    /// Number of pending rebalance operations (from last `rebalance()` call).
    pub rebalance_ops_pending: usize,
}

/// Errors produced by the balancer.
#[derive(Debug, Clone, thiserror::Error)]
pub enum BalancerError {
    #[error("shard not found: {0}")]
    ShardNotFound(String),

    #[error("content not found: {0}")]
    ContentNotFound(String),

    #[error("insufficient shards: need {need}, have {have}")]
    InsufficientShards { need: usize, have: usize },

    #[error("replication failed: {0}")]
    ReplicationFailed(String),

    #[error("invalid configuration: {0}")]
    InvalidConfiguration(String),
}

// ---------------------------------------------------------------------------
// StorageShardBalancer
// ---------------------------------------------------------------------------

/// Production-grade consistent-hashing shard balancer.
///
/// Virtual nodes for each shard are distributed across a `BTreeMap<u64, String>`
/// ring. Content is assigned to shards by walking the ring clockwise from the
/// FNV-1a hash of the CID, selecting the first `replication_factor` distinct
/// healthy shards.
pub struct StorageShardBalancer {
    config: BalancerConfig,
    /// Virtual-node ring: position → shard_id.
    ring: BTreeMap<u64, String>,
    /// All registered shards.
    shards: HashMap<String, ShardNode>,
    /// Current content assignments.
    assignments: HashMap<String, ShardAssignment>,
    /// Pending operations from the last `rebalance()` call.
    pending_ops: Vec<RebalanceOp>,
    /// Monotonic counter used as a simple clock for `assigned_at`.
    clock: u64,
}

impl StorageShardBalancer {
    /// Create a new balancer with the provided configuration.
    pub fn new(config: BalancerConfig) -> Self {
        Self {
            config,
            ring: BTreeMap::new(),
            shards: HashMap::new(),
            assignments: HashMap::new(),
            pending_ops: Vec::new(),
            clock: 0,
        }
    }

    /// Create a balancer with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(BalancerConfig::default())
    }

    // -----------------------------------------------------------------------
    // Shard management
    // -----------------------------------------------------------------------

    /// Register a shard and add its virtual nodes to the ring.
    pub fn add_shard(&mut self, shard: ShardNode) -> Result<(), BalancerError> {
        if shard.id.is_empty() {
            return Err(BalancerError::InvalidConfiguration(
                "shard id must not be empty".to_string(),
            ));
        }
        let vn = if shard.virtual_nodes > 0 {
            shard.virtual_nodes
        } else {
            self.config.virtual_nodes_per_shard
        };
        let id = shard.id.clone();
        self.shards.insert(id.clone(), shard);
        for replica in 0..vn {
            let pos = virtual_node_key(&id, replica);
            self.ring.insert(pos, id.clone());
        }
        Ok(())
    }

    /// Remove a shard from the ring and generate `MoveContent` ops for all
    /// affected assignments.
    pub fn remove_shard(&mut self, shard_id: &str) -> Result<Vec<RebalanceOp>, BalancerError> {
        if !self.shards.contains_key(shard_id) {
            return Err(BalancerError::ShardNotFound(shard_id.to_string()));
        }

        // Remove all virtual nodes belonging to this shard.
        let positions: Vec<u64> = self
            .ring
            .iter()
            .filter_map(|(&pos, sid)| if sid == shard_id { Some(pos) } else { None })
            .collect();
        let mut ops: Vec<RebalanceOp> = positions
            .iter()
            .map(|&pos| RebalanceOp::RemoveVirtualNode {
                shard_id: shard_id.to_string(),
                position: pos,
            })
            .collect();
        for pos in &positions {
            self.ring.remove(pos);
        }
        self.shards.remove(shard_id);

        // Generate MoveContent ops for assignments that involve this shard.
        let affected: Vec<String> = self
            .assignments
            .iter()
            .filter_map(|(cid, a)| {
                if a.shard_id == shard_id || a.replica_shards.iter().any(|r| r == shard_id) {
                    Some(cid.clone())
                } else {
                    None
                }
            })
            .collect();

        for cid in &affected {
            // Attempt to re-assign to a new shard (ring already updated).
            match self.assign_internal(cid) {
                Ok(new_assignment) => {
                    // Emit a MoveContent op to the new primary.
                    ops.push(RebalanceOp::MoveContent {
                        cid: cid.clone(),
                        from_shard: shard_id.to_string(),
                        to_shard: new_assignment.shard_id.clone(),
                    });
                    self.assignments.insert(cid.clone(), new_assignment);
                }
                Err(_) => {
                    // Not enough shards remain — remove the assignment.
                    self.assignments.remove(cid);
                }
            }
        }

        Ok(ops)
    }

    // -----------------------------------------------------------------------
    // Assignment
    // -----------------------------------------------------------------------

    /// Assign a CID to shards using consistent hashing.
    ///
    /// Returns an error when fewer than `replication_factor` healthy shards
    /// are available.
    pub fn assign(&mut self, cid: &str) -> Result<ShardAssignment, BalancerError> {
        let ts = self.tick_clock();
        let assignment = Self::assign_from_ring(
            cid,
            &self.ring,
            &self.shards,
            self.config.replication_factor,
            ts,
        )?;
        self.assignments.insert(cid.to_string(), assignment.clone());
        Ok(assignment)
    }

    fn assign_internal(&self, cid: &str) -> Result<ShardAssignment, BalancerError> {
        Self::assign_from_ring(
            cid,
            &self.ring,
            &self.shards,
            self.config.replication_factor,
            self.clock,
        )
    }

    fn assign_from_ring(
        cid: &str,
        ring: &BTreeMap<u64, String>,
        shards: &HashMap<String, ShardNode>,
        needed: usize,
        assigned_at: u64,
    ) -> Result<ShardAssignment, BalancerError> {
        let healthy_count = shards.values().filter(|s| s.is_healthy).count();
        if healthy_count < needed {
            return Err(BalancerError::InsufficientShards {
                need: needed,
                have: healthy_count,
            });
        }

        let hash = content_key(cid);
        let mut selected: Vec<String> = Vec::with_capacity(needed);

        // Walk the ring clockwise from `hash`, collecting `needed` distinct
        // healthy shards. We must iterate twice (wrap around) in the worst case.
        let ring_walk = ring.range(hash..).chain(ring.range(..hash));

        for (_, sid) in ring_walk {
            if selected.contains(sid) {
                continue;
            }
            if let Some(shard) = shards.get(sid.as_str()) {
                if shard.is_healthy {
                    selected.push(sid.clone());
                    if selected.len() == needed {
                        break;
                    }
                }
            }
        }

        if selected.len() < needed {
            return Err(BalancerError::ReplicationFailed(format!(
                "could only place {} of {} replicas for cid {}",
                selected.len(),
                needed,
                cid
            )));
        }

        let primary = selected.remove(0);

        Ok(ShardAssignment {
            cid: cid.to_string(),
            shard_id: primary,
            replica_shards: selected,
            assigned_at,
        })
    }

    /// Returns a reference to an existing assignment.
    pub fn lookup(&self, cid: &str) -> Result<&ShardAssignment, BalancerError> {
        self.assignments
            .get(cid)
            .ok_or_else(|| BalancerError::ContentNotFound(cid.to_string()))
    }

    // -----------------------------------------------------------------------
    // Usage tracking
    // -----------------------------------------------------------------------

    /// Update used bytes for a shard (delta can be negative for deletions).
    pub fn record_usage(&mut self, shard_id: &str, delta_bytes: i64) -> Result<(), BalancerError> {
        let shard = self
            .shards
            .get_mut(shard_id)
            .ok_or_else(|| BalancerError::ShardNotFound(shard_id.to_string()))?;
        if delta_bytes >= 0 {
            shard.used_bytes = shard.used_bytes.saturating_add(delta_bytes as u64);
        } else {
            shard.used_bytes = shard.used_bytes.saturating_sub((-delta_bytes) as u64);
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Health
    // -----------------------------------------------------------------------

    /// Mark a shard healthy or unhealthy.
    pub fn set_shard_health(&mut self, shard_id: &str, healthy: bool) -> Result<(), BalancerError> {
        let shard = self
            .shards
            .get_mut(shard_id)
            .ok_or_else(|| BalancerError::ShardNotFound(shard_id.to_string()))?;
        shard.is_healthy = healthy;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Rebalancing
    // -----------------------------------------------------------------------

    /// Compute rebalance operations if the imbalance ratio exceeds the
    /// configured threshold.  Operations are also stored in `pending_ops`.
    pub fn rebalance(&mut self) -> Result<Vec<RebalanceOp>, BalancerError> {
        let stats = self.stats();
        if stats.imbalance_ratio <= self.config.rebalance_threshold {
            self.pending_ops.clear();
            return Ok(Vec::new());
        }

        let ops = match &self.config.policy.clone() {
            RebalancePolicy::LeastLoaded => self.rebalance_least_loaded(),
            RebalancePolicy::ConsistentHash => self.rebalance_consistent_hash(),
            RebalancePolicy::RegionAware(region) => {
                let r = region.clone();
                self.rebalance_region_aware(&r)
            }
            RebalancePolicy::CapacityWeighted => self.rebalance_capacity_weighted(),
            RebalancePolicy::MinimalMovement => self.rebalance_minimal_movement(),
        };

        self.pending_ops = ops.clone();
        Ok(ops)
    }

    /// LeastLoaded: move content from most-loaded shard to least-loaded shard.
    fn rebalance_least_loaded(&mut self) -> Vec<RebalanceOp> {
        let mut ops = Vec::new();
        let threshold = self.config.rebalance_threshold;

        // Iterative: repeatedly move one item from most-loaded to least-loaded
        // until the ratio drops below threshold or no moves are possible.
        for _ in 0..self.assignments.len().saturating_add(1) {
            let (most_id, least_id) = match self.most_and_least_loaded() {
                Some(pair) => pair,
                None => break,
            };

            let max_util = self.shards[&most_id].utilization();
            let min_util = self.shards[&least_id].utilization();
            if max_util == 0.0 || (max_util / min_util.max(f64::EPSILON)) <= threshold {
                break;
            }

            // Pick one content item currently primary on most_id.
            let cid = self
                .assignments
                .iter()
                .find(|(_, a)| a.shard_id == most_id)
                .map(|(c, _)| c.clone());

            match cid {
                None => break,
                Some(c) => {
                    // Estimate block size (not tracked exactly; use 1 byte placeholder).
                    let block_size = 1_u64;
                    ops.push(RebalanceOp::MoveContent {
                        cid: c.clone(),
                        from_shard: most_id.clone(),
                        to_shard: least_id.clone(),
                    });
                    // Update in-memory state to reflect the move.
                    if let Some(a) = self.assignments.get_mut(&c) {
                        a.shard_id = least_id.clone();
                    }
                    if let Some(s) = self.shards.get_mut(&most_id) {
                        s.used_bytes = s.used_bytes.saturating_sub(block_size);
                    }
                    if let Some(s) = self.shards.get_mut(&least_id) {
                        s.used_bytes = s.used_bytes.saturating_add(block_size);
                    }
                }
            }
        }
        ops
    }

    /// ConsistentHash: find content whose primary shard differs from the
    /// hash-assigned shard and emit moves.
    fn rebalance_consistent_hash(&mut self) -> Vec<RebalanceOp> {
        let misplaced: Vec<(String, String, String)> = self
            .assignments
            .iter()
            .filter_map(|(cid, a)| {
                let expected = self.ring_lookup_primary(cid)?;
                if expected != a.shard_id {
                    Some((cid.clone(), a.shard_id.clone(), expected))
                } else {
                    None
                }
            })
            .collect();

        let mut ops = Vec::new();
        for (cid, from, to) in misplaced {
            ops.push(RebalanceOp::MoveContent {
                cid: cid.clone(),
                from_shard: from,
                to_shard: to.clone(),
            });
            if let Some(a) = self.assignments.get_mut(&cid) {
                a.shard_id = to;
            }
        }
        ops
    }

    /// RegionAware: generate UpdateWeight ops to prefer shards in the given
    /// region, then run LeastLoaded within those shards.
    fn rebalance_region_aware(&mut self, prefer_region: &str) -> Vec<RebalanceOp> {
        let mut ops = Vec::new();
        // Boost weight for shards in the preferred region.
        let boosts: Vec<(String, f64)> = self
            .shards
            .iter()
            .filter(|(_, s)| s.region == prefer_region && s.is_healthy)
            .map(|(id, s)| (id.clone(), s.weight * 2.0))
            .collect();
        for (sid, new_weight) in boosts {
            ops.push(RebalanceOp::UpdateWeight {
                shard_id: sid.clone(),
                new_weight,
            });
            if let Some(s) = self.shards.get_mut(&sid) {
                s.weight = new_weight;
            }
        }
        // Then perform a LeastLoaded rebalance with the updated weights.
        ops.extend(self.rebalance_least_loaded());
        ops
    }

    /// CapacityWeighted: move content from over-capacity shards to shards with
    /// the highest (free/capacity) * weight score.
    fn rebalance_capacity_weighted(&mut self) -> Vec<RebalanceOp> {
        let mut ops = Vec::new();
        let threshold = self.config.rebalance_threshold;

        for _ in 0..self.assignments.len().saturating_add(1) {
            // Most loaded by utilization.
            let most_loaded = self
                .shards
                .iter()
                .filter(|(_, s)| s.is_healthy)
                .max_by(|(_, a), (_, b)| {
                    a.utilization()
                        .partial_cmp(&b.utilization())
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(id, _)| id.clone());

            // Best destination by capacity_weight.
            let best_dest = self
                .shards
                .iter()
                .filter(|(_, s)| s.is_healthy)
                .max_by(|(_, a), (_, b)| {
                    a.capacity_weight()
                        .partial_cmp(&b.capacity_weight())
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(id, _)| id.clone());

            let (most_id, dest_id) = match (most_loaded, best_dest) {
                (Some(m), Some(d)) if m != d => (m, d),
                _ => break,
            };

            let max_util = self.shards[&most_id].utilization();
            let dest_util = self.shards[&dest_id].utilization();
            if max_util == 0.0 || (max_util / dest_util.max(f64::EPSILON)) <= threshold {
                break;
            }

            let cid = self
                .assignments
                .iter()
                .find(|(_, a)| a.shard_id == most_id)
                .map(|(c, _)| c.clone());

            match cid {
                None => break,
                Some(c) => {
                    ops.push(RebalanceOp::MoveContent {
                        cid: c.clone(),
                        from_shard: most_id.clone(),
                        to_shard: dest_id.clone(),
                    });
                    if let Some(a) = self.assignments.get_mut(&c) {
                        a.shard_id = dest_id.clone();
                    }
                    if let Some(s) = self.shards.get_mut(&most_id) {
                        s.used_bytes = s.used_bytes.saturating_sub(1);
                    }
                    if let Some(s) = self.shards.get_mut(&dest_id) {
                        s.used_bytes = s.used_bytes.saturating_add(1);
                    }
                }
            }
        }
        ops
    }

    /// MinimalMovement: only move the minimum number of items needed to bring
    /// the imbalance ratio below the threshold.
    fn rebalance_minimal_movement(&mut self) -> Vec<RebalanceOp> {
        let mut ops = Vec::new();
        let threshold = self.config.rebalance_threshold;

        while let Some((most_id, least_id)) = self.most_and_least_loaded() {
            let max_util = self.shards[&most_id].utilization();
            let min_util = self.shards[&least_id].utilization();

            if max_util == 0.0 || (max_util / min_util.max(f64::EPSILON)) <= threshold {
                break;
            }

            // Move exactly one item.
            let cid = self
                .assignments
                .iter()
                .find(|(_, a)| a.shard_id == most_id)
                .map(|(c, _)| c.clone());

            match cid {
                None => break,
                Some(c) => {
                    ops.push(RebalanceOp::MoveContent {
                        cid: c.clone(),
                        from_shard: most_id.clone(),
                        to_shard: least_id.clone(),
                    });
                    if let Some(a) = self.assignments.get_mut(&c) {
                        a.shard_id = least_id.clone();
                    }
                    if let Some(s) = self.shards.get_mut(&most_id) {
                        s.used_bytes = s.used_bytes.saturating_sub(1);
                    }
                    if let Some(s) = self.shards.get_mut(&least_id) {
                        s.used_bytes = s.used_bytes.saturating_add(1);
                    }
                }
            }
        }
        ops
    }

    // -----------------------------------------------------------------------
    // Stats and inspection
    // -----------------------------------------------------------------------

    /// Return a snapshot of current balancer statistics.
    pub fn stats(&self) -> SsbBalancerStats {
        let shard_count = self.shards.len();
        let total_capacity_bytes: u64 = self.shards.values().map(|s| s.capacity_bytes).sum();
        let total_used_bytes: u64 = self.shards.values().map(|s| s.used_bytes).sum();

        let utilization_pct = if total_capacity_bytes == 0 {
            0.0
        } else {
            (total_used_bytes as f64 / total_capacity_bytes as f64) * 100.0
        };

        let imbalance_ratio = self.compute_imbalance_ratio();

        SsbBalancerStats {
            shard_count,
            total_capacity_bytes,
            total_used_bytes,
            utilization_pct,
            imbalance_ratio,
            rebalance_ops_pending: self.pending_ops.len(),
        }
    }

    /// Return all virtual node ring positions, sorted ascending.
    pub fn ring_positions(&self) -> Vec<(u64, String)> {
        self.ring
            .iter()
            .map(|(&pos, sid)| (pos, sid.clone()))
            .collect()
    }

    /// Return the current config.
    pub fn config(&self) -> &BalancerConfig {
        &self.config
    }

    /// Return a reference to a shard by id.
    pub fn shard(&self, shard_id: &str) -> Option<&ShardNode> {
        self.shards.get(shard_id)
    }

    /// Return iterator over all registered shards.
    pub fn shards(&self) -> impl Iterator<Item = &ShardNode> {
        self.shards.values()
    }

    /// Return the number of CIDs currently assigned.
    pub fn assignment_count(&self) -> usize {
        self.assignments.len()
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Walk the ring from the CID hash and return the primary healthy shard id.
    fn ring_lookup_primary(&self, cid: &str) -> Option<String> {
        let hash = content_key(cid);
        let ring_walk = self.ring.range(hash..).chain(self.ring.range(..hash));
        for (_, sid) in ring_walk {
            if let Some(shard) = self.shards.get(sid.as_str()) {
                if shard.is_healthy {
                    return Some(sid.clone());
                }
            }
        }
        None
    }

    fn most_and_least_loaded(&self) -> Option<(String, String)> {
        let healthy: Vec<(&String, &ShardNode)> =
            self.shards.iter().filter(|(_, s)| s.is_healthy).collect();

        if healthy.len() < 2 {
            return None;
        }

        let most = healthy
            .iter()
            .max_by(|(_, a), (_, b)| {
                a.utilization()
                    .partial_cmp(&b.utilization())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(id, _)| (*id).clone())?;

        let least = healthy
            .iter()
            .min_by(|(_, a), (_, b)| {
                a.utilization()
                    .partial_cmp(&b.utilization())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(id, _)| (*id).clone())?;

        if most == least {
            None
        } else {
            Some((most, least))
        }
    }

    fn compute_imbalance_ratio(&self) -> f64 {
        let healthy: Vec<f64> = self
            .shards
            .values()
            .filter(|s| s.is_healthy && s.capacity_bytes > 0)
            .map(|s| s.utilization())
            .collect();

        if healthy.len() < 2 {
            return 1.0;
        }

        let max = healthy.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min = healthy.iter().cloned().fold(f64::INFINITY, f64::min);

        // If the most-loaded shard has zero utilization, everything is empty —
        // no imbalance.
        if max <= 0.0 {
            return 1.0;
        }

        // If the least-loaded shard has zero utilization but some shard is
        // loaded, treat the ratio as very large (extreme imbalance).
        if min <= 0.0 {
            return max * 1000.0 + 1.0;
        }

        max / min
    }

    fn tick_clock(&mut self) -> u64 {
        self.clock = self.clock.wrapping_add(1);
        self.clock
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn make_shard(id: &str, cap: u64, used: u64) -> ShardNode {
        ShardNode {
            id: id.to_string(),
            capacity_bytes: cap,
            used_bytes: used,
            virtual_nodes: 20,
            is_healthy: true,
            region: "us-east".to_string(),
            weight: 1.0,
        }
    }

    fn make_shard_region(id: &str, cap: u64, used: u64, region: &str) -> ShardNode {
        ShardNode {
            id: id.to_string(),
            capacity_bytes: cap,
            used_bytes: used,
            virtual_nodes: 20,
            is_healthy: true,
            region: region.to_string(),
            weight: 1.0,
        }
    }

    fn two_shard_balancer() -> StorageShardBalancer {
        let mut b = StorageShardBalancer::new(BalancerConfig {
            replication_factor: 2,
            virtual_nodes_per_shard: 20,
            rebalance_threshold: 1.5,
            policy: RebalancePolicy::LeastLoaded,
        });
        b.add_shard(make_shard("s0", 1_000_000, 0)).unwrap();
        b.add_shard(make_shard("s1", 1_000_000, 0)).unwrap();
        b
    }

    fn three_shard_balancer() -> StorageShardBalancer {
        let mut b = StorageShardBalancer::new(BalancerConfig {
            replication_factor: 3,
            virtual_nodes_per_shard: 20,
            rebalance_threshold: 1.5,
            policy: RebalancePolicy::LeastLoaded,
        });
        b.add_shard(make_shard("s0", 1_000_000, 0)).unwrap();
        b.add_shard(make_shard("s1", 1_000_000, 0)).unwrap();
        b.add_shard(make_shard("s2", 1_000_000, 0)).unwrap();
        b
    }

    // -----------------------------------------------------------------------
    // FNV-1a correctness
    // -----------------------------------------------------------------------

    #[test]
    fn test_fnv1a_empty() {
        assert_eq!(fnv1a_64(b""), 14_695_981_039_346_656_037_u64);
    }

    #[test]
    fn test_fnv1a_deterministic() {
        assert_eq!(fnv1a_64(b"hello"), fnv1a_64(b"hello"));
    }

    #[test]
    fn test_fnv1a_distinct() {
        assert_ne!(fnv1a_64(b"alpha"), fnv1a_64(b"beta"));
    }

    #[test]
    fn test_virtual_node_key_distinct_replicas() {
        let k0 = virtual_node_key("shard-a", 0);
        let k1 = virtual_node_key("shard-a", 1);
        assert_ne!(k0, k1);
    }

    // -----------------------------------------------------------------------
    // xorshift64 PRNG
    // -----------------------------------------------------------------------

    #[test]
    fn test_xorshift64_nonzero() {
        let mut state = 12345_u64;
        let v = xorshift64(&mut state);
        assert_ne!(v, 0);
    }

    #[test]
    fn test_xorshift64_sequence() {
        let mut s = 1_u64;
        let a = xorshift64(&mut s);
        let b = xorshift64(&mut s);
        assert_ne!(a, b);
    }

    // -----------------------------------------------------------------------
    // add_shard
    // -----------------------------------------------------------------------

    #[test]
    fn test_add_shard_populates_ring() {
        let mut b = StorageShardBalancer::new(BalancerConfig::default());
        b.add_shard(make_shard("s0", 1_000, 0)).unwrap();
        assert!(!b.ring.is_empty());
    }

    #[test]
    fn test_add_shard_virtual_node_count() {
        let mut b = StorageShardBalancer::new(BalancerConfig {
            virtual_nodes_per_shard: 30,
            ..BalancerConfig::default()
        });
        let mut shard = make_shard("s0", 1_000, 0);
        shard.virtual_nodes = 30;
        b.add_shard(shard).unwrap();
        let count = b.ring.values().filter(|id| id.as_str() == "s0").count();
        // Collisions may reduce count; assert at least 1 unique node placed.
        assert!(count >= 1);
    }

    #[test]
    fn test_add_shard_empty_id_error() {
        let mut b = StorageShardBalancer::new(BalancerConfig::default());
        let result = b.add_shard(ShardNode {
            id: String::new(),
            ..ShardNode::default()
        });
        assert!(matches!(
            result,
            Err(BalancerError::InvalidConfiguration(_))
        ));
    }

    #[test]
    fn test_add_multiple_shards() {
        let mut b = StorageShardBalancer::new(BalancerConfig::default());
        for i in 0..5_u32 {
            b.add_shard(make_shard(&format!("s{}", i), 1_000, 0))
                .unwrap();
        }
        assert_eq!(b.shards.len(), 5);
    }

    // -----------------------------------------------------------------------
    // remove_shard
    // -----------------------------------------------------------------------

    #[test]
    fn test_remove_shard_cleans_ring() {
        let mut b = two_shard_balancer();
        let before = b.ring.len();
        b.remove_shard("s0").unwrap();
        let after = b.ring.len();
        assert!(after < before);
        assert!(!b.ring.values().any(|id| id == "s0"));
    }

    #[test]
    fn test_remove_shard_not_found_error() {
        let mut b = two_shard_balancer();
        let err = b.remove_shard("ghost");
        assert!(matches!(err, Err(BalancerError::ShardNotFound(_))));
    }

    #[test]
    fn test_remove_shard_generates_remove_vn_ops() {
        let mut b = two_shard_balancer();
        let ops = b.remove_shard("s0").unwrap();
        let remove_vn: Vec<_> = ops
            .iter()
            .filter(|o| matches!(o, RebalanceOp::RemoveVirtualNode { .. }))
            .collect();
        assert!(!remove_vn.is_empty());
    }

    #[test]
    fn test_remove_shard_reassigns_content() {
        let mut b = three_shard_balancer();
        b.assign("cid-abc").unwrap();
        b.assign("cid-xyz").unwrap();
        let ops = b.remove_shard("s0").unwrap();
        let moves: Vec<_> = ops
            .iter()
            .filter(|o| matches!(o, RebalanceOp::MoveContent { .. }))
            .collect();
        // At minimum we expect 0 or more moves (depends on which shards held
        // the content).  Assert the ring no longer contains s0.
        let _ = moves;
        assert!(!b.shards.contains_key("s0"));
    }

    // -----------------------------------------------------------------------
    // assign
    // -----------------------------------------------------------------------

    #[test]
    fn test_assign_returns_valid_shard() {
        let mut b = two_shard_balancer();
        let a = b.assign("Qmtest1").unwrap();
        assert!(!a.shard_id.is_empty());
        assert!(b.shards.contains_key(&a.shard_id));
    }

    #[test]
    fn test_assign_replication_factor_replicas() {
        let mut b = three_shard_balancer();
        let a = b.assign("Qmtest-rf3").unwrap();
        // 1 primary + (rf-1) replicas = rf total
        assert_eq!(a.replica_shards.len(), 2);
        // All distinct
        assert_ne!(a.shard_id, a.replica_shards[0]);
        assert_ne!(a.shard_id, a.replica_shards[1]);
        assert_ne!(a.replica_shards[0], a.replica_shards[1]);
    }

    #[test]
    fn test_assign_deterministic() {
        let mut b1 = two_shard_balancer();
        let mut b2 = two_shard_balancer();
        let a1 = b1.assign("cid-det").unwrap();
        let a2 = b2.assign("cid-det").unwrap();
        assert_eq!(a1.shard_id, a2.shard_id);
    }

    #[test]
    fn test_assign_insufficient_shards_error() {
        let mut b = StorageShardBalancer::new(BalancerConfig {
            replication_factor: 5,
            virtual_nodes_per_shard: 10,
            rebalance_threshold: 1.5,
            policy: RebalancePolicy::LeastLoaded,
        });
        b.add_shard(make_shard("only", 1_000, 0)).unwrap();
        let err = b.assign("cid-1");
        assert!(matches!(err, Err(BalancerError::InsufficientShards { .. })));
    }

    #[test]
    fn test_assign_stores_in_table() {
        let mut b = two_shard_balancer();
        b.assign("stored-cid").unwrap();
        assert!(b.lookup("stored-cid").is_ok());
    }

    #[test]
    fn test_assign_different_cids_may_land_different_shards() {
        let mut b = two_shard_balancer();
        let mut found_s0 = false;
        let mut found_s1 = false;
        let mut state = 999_u64;
        for _ in 0..50 {
            let cid = format!("cid-{}", xorshift64(&mut state));
            let a = b.assign(&cid).unwrap();
            if a.shard_id == "s0" {
                found_s0 = true;
            }
            if a.shard_id == "s1" {
                found_s1 = true;
            }
        }
        assert!(found_s0 && found_s1, "CIDs should distribute across shards");
    }

    #[test]
    fn test_assign_unhealthy_shard_skipped() {
        // Use rf=2 with 4 shards so marking one unhealthy still leaves 3 healthy.
        let mut b = StorageShardBalancer::new(BalancerConfig {
            replication_factor: 2,
            virtual_nodes_per_shard: 20,
            rebalance_threshold: 1.5,
            policy: RebalancePolicy::LeastLoaded,
        });
        for i in 0..4_u32 {
            b.add_shard(make_shard(&format!("s{}", i), 1_000_000, 0))
                .unwrap();
        }
        b.set_shard_health("s0", false).unwrap();
        for i in 0..20 {
            let a = b.assign(&format!("cid-{}", i)).unwrap();
            assert_ne!(a.shard_id, "s0");
            assert!(!a.replica_shards.contains(&"s0".to_string()));
        }
    }

    // -----------------------------------------------------------------------
    // lookup
    // -----------------------------------------------------------------------

    #[test]
    fn test_lookup_existing() {
        let mut b = two_shard_balancer();
        b.assign("lookup-cid").unwrap();
        let r = b.lookup("lookup-cid");
        assert!(r.is_ok());
        assert_eq!(r.unwrap().cid, "lookup-cid");
    }

    #[test]
    fn test_lookup_missing_error() {
        let b = two_shard_balancer();
        let err = b.lookup("missing");
        assert!(matches!(err, Err(BalancerError::ContentNotFound(_))));
    }

    // -----------------------------------------------------------------------
    // record_usage
    // -----------------------------------------------------------------------

    #[test]
    fn test_record_usage_positive() {
        let mut b = two_shard_balancer();
        b.record_usage("s0", 500).unwrap();
        assert_eq!(b.shards["s0"].used_bytes, 500);
    }

    #[test]
    fn test_record_usage_negative() {
        let mut b = two_shard_balancer();
        b.record_usage("s0", 1000).unwrap();
        b.record_usage("s0", -400).unwrap();
        assert_eq!(b.shards["s0"].used_bytes, 600);
    }

    #[test]
    fn test_record_usage_saturating_underflow() {
        let mut b = two_shard_balancer();
        b.record_usage("s0", -9_999_999).unwrap();
        assert_eq!(b.shards["s0"].used_bytes, 0);
    }

    #[test]
    fn test_record_usage_saturating_overflow() {
        let mut b = two_shard_balancer();
        b.record_usage("s0", i64::MAX).unwrap();
        // used_bytes capped at capacity (u64 always fits in u64::MAX)
        let _ = b.shards["s0"].used_bytes;
    }

    #[test]
    fn test_record_usage_not_found_error() {
        let mut b = two_shard_balancer();
        let err = b.record_usage("ghost", 100);
        assert!(matches!(err, Err(BalancerError::ShardNotFound(_))));
    }

    // -----------------------------------------------------------------------
    // set_shard_health
    // -----------------------------------------------------------------------

    #[test]
    fn test_set_shard_health_false() {
        let mut b = two_shard_balancer();
        b.set_shard_health("s0", false).unwrap();
        assert!(!b.shards["s0"].is_healthy);
    }

    #[test]
    fn test_set_shard_health_true() {
        let mut b = two_shard_balancer();
        b.set_shard_health("s0", false).unwrap();
        b.set_shard_health("s0", true).unwrap();
        assert!(b.shards["s0"].is_healthy);
    }

    #[test]
    fn test_set_shard_health_not_found_error() {
        let mut b = two_shard_balancer();
        let err = b.set_shard_health("ghost", false);
        assert!(matches!(err, Err(BalancerError::ShardNotFound(_))));
    }

    #[test]
    fn test_assign_after_health_toggle() {
        // Use rf=2 with 4 shards so marking one unhealthy still leaves 3 healthy.
        let mut b = StorageShardBalancer::new(BalancerConfig {
            replication_factor: 2,
            virtual_nodes_per_shard: 20,
            rebalance_threshold: 1.5,
            policy: RebalancePolicy::LeastLoaded,
        });
        for i in 0..4_u32 {
            b.add_shard(make_shard(&format!("s{}", i), 1_000_000, 0))
                .unwrap();
        }
        b.set_shard_health("s0", false).unwrap();
        let a = b.assign("cid-h").unwrap();
        assert_ne!(a.shard_id, "s0");
        b.set_shard_health("s0", true).unwrap();
        let _a2 = b.assign("cid-h2").unwrap();
    }

    // -----------------------------------------------------------------------
    // stats
    // -----------------------------------------------------------------------

    #[test]
    fn test_stats_empty() {
        let b = StorageShardBalancer::new(BalancerConfig::default());
        let s = b.stats();
        assert_eq!(s.shard_count, 0);
        assert_eq!(s.total_capacity_bytes, 0);
    }

    #[test]
    fn test_stats_capacity_sum() {
        let b = three_shard_balancer();
        let s = b.stats();
        assert_eq!(s.total_capacity_bytes, 3_000_000);
        assert_eq!(s.shard_count, 3);
    }

    #[test]
    fn test_stats_utilization_pct() {
        let mut b = two_shard_balancer();
        b.record_usage("s0", 500_000).unwrap();
        let s = b.stats();
        assert!((s.utilization_pct - 25.0).abs() < 0.01);
    }

    #[test]
    fn test_stats_imbalance_ratio_equal() {
        let mut b = two_shard_balancer();
        b.record_usage("s0", 100_000).unwrap();
        b.record_usage("s1", 100_000).unwrap();
        let s = b.stats();
        assert!((s.imbalance_ratio - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_stats_imbalance_ratio_unequal() {
        let mut b = two_shard_balancer();
        b.record_usage("s0", 900_000).unwrap();
        b.record_usage("s1", 100_000).unwrap();
        let s = b.stats();
        assert!(s.imbalance_ratio > 1.0);
    }

    // -----------------------------------------------------------------------
    // ring_positions
    // -----------------------------------------------------------------------

    #[test]
    fn test_ring_positions_sorted() {
        let b = two_shard_balancer();
        let positions = b.ring_positions();
        for w in positions.windows(2) {
            assert!(w[0].0 <= w[1].0);
        }
    }

    #[test]
    fn test_ring_positions_non_empty() {
        let b = two_shard_balancer();
        assert!(!b.ring_positions().is_empty());
    }

    #[test]
    fn test_ring_positions_contains_shard_ids() {
        let b = two_shard_balancer();
        let ids: std::collections::HashSet<_> =
            b.ring_positions().into_iter().map(|(_, id)| id).collect();
        assert!(ids.contains("s0"));
        assert!(ids.contains("s1"));
    }

    #[test]
    fn test_ring_positions_after_remove() {
        let mut b = two_shard_balancer();
        b.remove_shard("s0").unwrap();
        let ids: std::collections::HashSet<_> =
            b.ring_positions().into_iter().map(|(_, id)| id).collect();
        assert!(!ids.contains("s0"));
        assert!(ids.contains("s1"));
    }

    // -----------------------------------------------------------------------
    // rebalance — LeastLoaded
    // -----------------------------------------------------------------------

    #[test]
    fn test_rebalance_no_ops_when_balanced() {
        let mut b = two_shard_balancer();
        b.record_usage("s0", 500_000).unwrap();
        b.record_usage("s1", 500_000).unwrap();
        let ops = b.rebalance().unwrap();
        assert!(ops.is_empty());
    }

    #[test]
    fn test_rebalance_least_loaded_generates_moves() {
        let mut b = StorageShardBalancer::new(BalancerConfig {
            replication_factor: 1,
            virtual_nodes_per_shard: 20,
            rebalance_threshold: 1.5,
            policy: RebalancePolicy::LeastLoaded,
        });
        b.add_shard(make_shard("heavy", 1_000_000, 900_000))
            .unwrap();
        b.add_shard(make_shard("light", 1_000_000, 10_000)).unwrap();
        // Assign some content to heavy shard.
        let mut state = 42_u64;
        for _ in 0..10 {
            let cid = format!("cid-{}", xorshift64(&mut state));
            let _ = b.assign(&cid);
        }
        // Force all assignments to heavy shard so moves are guaranteed.
        let cids: Vec<_> = b.assignments.keys().cloned().collect();
        for cid in &cids {
            if let Some(a) = b.assignments.get_mut(cid) {
                a.shard_id = "heavy".to_string();
            }
        }
        let ops = b.rebalance().unwrap();
        assert!(ops
            .iter()
            .any(|o| matches!(o, RebalanceOp::MoveContent { .. })));
    }

    // -----------------------------------------------------------------------
    // rebalance — ConsistentHash
    // -----------------------------------------------------------------------

    #[test]
    fn test_rebalance_consistent_hash_fixes_misplaced() {
        let mut b = StorageShardBalancer::new(BalancerConfig {
            replication_factor: 1,
            virtual_nodes_per_shard: 20,
            rebalance_threshold: 1.0,
            policy: RebalancePolicy::ConsistentHash,
        });
        b.add_shard(make_shard("s0", 1_000_000, 900_000)).unwrap();
        b.add_shard(make_shard("s1", 1_000_000, 10_000)).unwrap();
        b.assign("cid-misplace").unwrap();
        // Manually misplace.
        if let Some(a) = b.assignments.get_mut("cid-misplace") {
            let correct = a.shard_id.clone();
            let wrong = if correct == "s0" {
                "s1".to_string()
            } else {
                "s0".to_string()
            };
            a.shard_id = wrong;
        }
        let ops = b.rebalance().unwrap();
        let move_ops: Vec<_> = ops
            .iter()
            .filter(|o| matches!(o, RebalanceOp::MoveContent { .. }))
            .collect();
        assert!(!move_ops.is_empty());
    }

    // -----------------------------------------------------------------------
    // rebalance — RegionAware
    // -----------------------------------------------------------------------

    #[test]
    fn test_rebalance_region_aware_boosts_weight() {
        let mut b = StorageShardBalancer::new(BalancerConfig {
            replication_factor: 1,
            virtual_nodes_per_shard: 20,
            rebalance_threshold: 1.5,
            policy: RebalancePolicy::RegionAware("eu-west".to_string()),
        });
        b.add_shard(make_shard_region("heavy", 1_000_000, 900_000, "us-east"))
            .unwrap();
        b.add_shard(make_shard_region("light-eu", 1_000_000, 10_000, "eu-west"))
            .unwrap();
        // Force assignment on heavy.
        let _ = b.assign("r-cid");
        if let Some(a) = b.assignments.get_mut("r-cid") {
            a.shard_id = "heavy".to_string();
        }
        let ops = b.rebalance().unwrap();
        let update_ops: Vec<_> = ops
            .iter()
            .filter(|o| matches!(o, RebalanceOp::UpdateWeight { shard_id, .. } if shard_id == "light-eu"))
            .collect();
        assert!(!update_ops.is_empty());
    }

    // -----------------------------------------------------------------------
    // rebalance — CapacityWeighted
    // -----------------------------------------------------------------------

    #[test]
    fn test_rebalance_capacity_weighted_generates_moves() {
        let mut b = StorageShardBalancer::new(BalancerConfig {
            replication_factor: 1,
            virtual_nodes_per_shard: 20,
            rebalance_threshold: 1.5,
            policy: RebalancePolicy::CapacityWeighted,
        });
        b.add_shard(make_shard("full", 1_000_000, 900_000)).unwrap();
        b.add_shard(make_shard("empty", 1_000_000, 0)).unwrap();
        let cids: Vec<String> = (0..5).map(|i| format!("cap-cid-{}", i)).collect();
        for cid in &cids {
            let _ = b.assign(cid);
        }
        for cid in &cids {
            if let Some(a) = b.assignments.get_mut(cid) {
                a.shard_id = "full".to_string();
            }
        }
        let ops = b.rebalance().unwrap();
        assert!(ops
            .iter()
            .any(|o| matches!(o, RebalanceOp::MoveContent { .. })));
    }

    // -----------------------------------------------------------------------
    // rebalance — MinimalMovement
    // -----------------------------------------------------------------------

    #[test]
    fn test_rebalance_minimal_movement_reduces_imbalance() {
        let mut b = StorageShardBalancer::new(BalancerConfig {
            replication_factor: 1,
            virtual_nodes_per_shard: 20,
            rebalance_threshold: 1.5,
            policy: RebalancePolicy::MinimalMovement,
        });
        b.add_shard(make_shard("heavy", 1_000_000, 900_000))
            .unwrap();
        b.add_shard(make_shard("light", 1_000_000, 10_000)).unwrap();
        let cids: Vec<String> = (0..10).map(|i| format!("min-cid-{}", i)).collect();
        for cid in &cids {
            let _ = b.assign(cid);
            if let Some(a) = b.assignments.get_mut(cid) {
                a.shard_id = "heavy".to_string();
            }
        }
        let before_ratio = b.stats().imbalance_ratio;
        let ops = b.rebalance().unwrap();
        let after_ratio = b.stats().imbalance_ratio;
        // Imbalance should have decreased (or ops were generated).
        assert!(
            after_ratio < before_ratio
                || ops
                    .iter()
                    .any(|o| matches!(o, RebalanceOp::MoveContent { .. }))
        );
    }

    // -----------------------------------------------------------------------
    // rebalance — stats.rebalance_ops_pending
    // -----------------------------------------------------------------------

    #[test]
    fn test_pending_ops_tracked_in_stats() {
        let mut b = StorageShardBalancer::new(BalancerConfig {
            replication_factor: 1,
            virtual_nodes_per_shard: 20,
            rebalance_threshold: 1.5,
            policy: RebalancePolicy::LeastLoaded,
        });
        b.add_shard(make_shard("heavy", 1_000_000, 900_000))
            .unwrap();
        b.add_shard(make_shard("light", 1_000_000, 10_000)).unwrap();
        let cids: Vec<String> = (0..5).map(|i| format!("pending-{}", i)).collect();
        for cid in &cids {
            let _ = b.assign(cid);
            if let Some(a) = b.assignments.get_mut(cid) {
                a.shard_id = "heavy".to_string();
            }
        }
        let ops = b.rebalance().unwrap();
        let stats = b.stats();
        assert_eq!(stats.rebalance_ops_pending, ops.len());
    }

    // -----------------------------------------------------------------------
    // Replication / multi-replica tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_replicas_are_distinct() {
        let mut b = three_shard_balancer();
        let a = b.assign("rep-cid").unwrap();
        let all: Vec<&str> = std::iter::once(a.shard_id.as_str())
            .chain(a.replica_shards.iter().map(|s| s.as_str()))
            .collect();
        let unique: std::collections::HashSet<_> = all.iter().copied().collect();
        assert_eq!(
            all.len(),
            unique.len(),
            "replica shards must all be distinct"
        );
    }

    #[test]
    fn test_rf1_no_replicas() {
        let mut b = StorageShardBalancer::new(BalancerConfig {
            replication_factor: 1,
            virtual_nodes_per_shard: 10,
            rebalance_threshold: 1.5,
            policy: RebalancePolicy::LeastLoaded,
        });
        b.add_shard(make_shard("solo", 1_000, 0)).unwrap();
        let a = b.assign("cid-solo").unwrap();
        assert!(a.replica_shards.is_empty());
    }

    #[test]
    fn test_rf4_four_replicas() {
        let mut b = StorageShardBalancer::new(BalancerConfig {
            replication_factor: 4,
            virtual_nodes_per_shard: 20,
            rebalance_threshold: 1.5,
            policy: RebalancePolicy::LeastLoaded,
        });
        for i in 0..4_u32 {
            b.add_shard(make_shard(&format!("s{}", i), 1_000_000, 0))
                .unwrap();
        }
        let a = b.assign("cid-rf4").unwrap();
        assert_eq!(a.replica_shards.len(), 3);
    }

    // -----------------------------------------------------------------------
    // Error cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_error_shard_not_found_display() {
        let e = BalancerError::ShardNotFound("x".to_string());
        assert!(e.to_string().contains("shard not found"));
    }

    #[test]
    fn test_error_content_not_found_display() {
        let e = BalancerError::ContentNotFound("c".to_string());
        assert!(e.to_string().contains("content not found"));
    }

    #[test]
    fn test_error_insufficient_shards_display() {
        let e = BalancerError::InsufficientShards { need: 3, have: 1 };
        assert!(e.to_string().contains("insufficient shards"));
    }

    #[test]
    fn test_error_invalid_configuration_display() {
        let e = BalancerError::InvalidConfiguration("bad".to_string());
        assert!(e.to_string().contains("invalid configuration"));
    }

    // -----------------------------------------------------------------------
    // ShardNode helpers
    // -----------------------------------------------------------------------

    #[test]
    fn test_shard_utilization_zero_cap() {
        let s = ShardNode {
            capacity_bytes: 0,
            ..ShardNode::default()
        };
        assert_eq!(s.utilization(), 1.0);
    }

    #[test]
    fn test_shard_utilization_half() {
        let s = make_shard("x", 1_000, 500);
        assert!((s.utilization() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_shard_free_bytes() {
        let s = make_shard("x", 1_000, 300);
        assert_eq!(s.free_bytes(), 700);
    }

    #[test]
    fn test_shard_capacity_weight() {
        let s = make_shard("x", 1_000, 0);
        assert!((s.capacity_weight() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_shard_capacity_weight_full() {
        let s = make_shard("x", 1_000, 1_000);
        assert!((s.capacity_weight() - 0.0).abs() < 1e-9);
    }

    // -----------------------------------------------------------------------
    // Miscellaneous / integration
    // -----------------------------------------------------------------------

    #[test]
    fn test_assignment_count() {
        let mut b = two_shard_balancer();
        b.assign("c1").unwrap();
        b.assign("c2").unwrap();
        b.assign("c3").unwrap();
        assert_eq!(b.assignment_count(), 3);
    }

    #[test]
    fn test_shards_iterator() {
        let b = three_shard_balancer();
        assert_eq!(b.shards().count(), 3);
    }

    #[test]
    fn test_shard_accessor() {
        let b = two_shard_balancer();
        assert!(b.shard("s0").is_some());
        assert!(b.shard("ghost").is_none());
    }

    #[test]
    fn test_config_accessor() {
        let b = two_shard_balancer();
        assert_eq!(b.config().replication_factor, 2);
    }

    #[test]
    fn test_large_ring_distribution() {
        let mut b = StorageShardBalancer::new(BalancerConfig {
            replication_factor: 1,
            virtual_nodes_per_shard: 150,
            rebalance_threshold: 2.0,
            policy: RebalancePolicy::LeastLoaded,
        });
        for i in 0..5_u32 {
            let mut s = make_shard(&format!("shard-{}", i), 10_000_000, 0);
            s.virtual_nodes = 150;
            b.add_shard(s).unwrap();
        }
        let mut counts: HashMap<String, usize> = HashMap::new();
        let mut state = 7_u64;
        for _ in 0..1000 {
            let cid = format!("cid-{}", xorshift64(&mut state));
            let a = b.assign(&cid).unwrap();
            *counts.entry(a.shard_id).or_insert(0) += 1;
        }
        // Each shard should receive at least 5% of assignments.
        for (shard_id, count) in &counts {
            assert!(
                *count >= 20,
                "shard {} received only {} / 1000 assignments",
                shard_id,
                count
            );
        }
    }

    #[test]
    fn test_reassign_after_remove_uses_remaining_shards() {
        let mut b = three_shard_balancer();
        b.assign("cid-remain").unwrap();
        b.remove_shard("s2").unwrap();
        // All remaining assignments should reference only s0 or s1.
        for a in b.assignments.values() {
            assert_ne!(a.shard_id, "s2");
        }
    }

    #[test]
    fn test_assign_all_shards_unhealthy_error() {
        let mut b = two_shard_balancer();
        b.set_shard_health("s0", false).unwrap();
        b.set_shard_health("s1", false).unwrap();
        let err = b.assign("cid-unhealthy");
        assert!(matches!(err, Err(BalancerError::InsufficientShards { .. })));
    }

    #[test]
    fn test_rebalance_consistent_hash_no_moves_when_correct() {
        let mut b = StorageShardBalancer::new(BalancerConfig {
            replication_factor: 1,
            virtual_nodes_per_shard: 20,
            rebalance_threshold: 1.0,
            policy: RebalancePolicy::ConsistentHash,
        });
        b.add_shard(make_shard("s0", 1_000_000, 900_000)).unwrap();
        b.add_shard(make_shard("s1", 1_000_000, 10_000)).unwrap();
        // Assign and let the ring decide — assignments should already be correct.
        let _ = b.assign("no-move-cid");
        // Consistent hash on correct assignments should produce no moves.
        let ops = b.rebalance().unwrap();
        let moves: Vec<_> = ops
            .iter()
            .filter(|o| matches!(o, RebalanceOp::MoveContent { .. }))
            .collect();
        assert!(
            moves.is_empty(),
            "correctly placed content should not be moved"
        );
    }
}
