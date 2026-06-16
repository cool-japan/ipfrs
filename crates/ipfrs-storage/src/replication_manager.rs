//! Storage Replication Manager
//!
//! Tracks and manages block replication across storage nodes with configurable
//! replication factors and health monitoring. Provides visibility into replica
//! health, selection of candidate nodes for new replicas, stale node eviction,
//! and comprehensive statistics.

use std::collections::HashMap;

// ── Error ─────────────────────────────────────────────────────────────────────

/// Errors produced by [`StorageReplicationManager`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReplicationError {
    /// A node with the given ID is already registered.
    NodeAlreadyRegistered(String),
    /// No node with the given ID is registered.
    NodeNotFound(String),
    /// No replicas are tracked for the given CID.
    CidNotFound(String),
}

impl std::fmt::Display for ReplicationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NodeAlreadyRegistered(id) => write!(f, "node already registered: {id}"),
            Self::NodeNotFound(id) => write!(f, "node not found: {id}"),
            Self::CidNotFound(cid) => write!(f, "CID not found: {cid}"),
        }
    }
}

impl std::error::Error for ReplicationError {}

// ── ReplicaNode ───────────────────────────────────────────────────────────────

/// A storage node that participates in the replication cluster.
#[derive(Debug, Clone)]
pub struct ReplicaNode {
    /// Unique identifier for the node.
    pub node_id: String,
    /// Network address of the node.
    pub address: String,
    /// Millisecond timestamp of when the node was last observed.
    pub last_seen: u64,
    /// Whether the node is currently considered healthy.
    pub healthy: bool,
    /// Total storage capacity in bytes.
    pub capacity_bytes: u64,
    /// Bytes currently in use.
    pub used_bytes: u64,
}

impl ReplicaNode {
    /// Returns the number of bytes available for new data.
    pub fn available_bytes(&self) -> u64 {
        self.capacity_bytes.saturating_sub(self.used_bytes)
    }

    /// Returns the fraction of capacity that is used, in `[0.0, 1.0]`.
    pub fn utilization(&self) -> f64 {
        self.used_bytes as f64 / self.capacity_bytes.max(1) as f64
    }
}

// ── ReplicaLocation ───────────────────────────────────────────────────────────

/// Records where a specific CID replica is stored.
#[derive(Debug, Clone)]
pub struct RmReplicaLocation {
    /// The node that holds this replica.
    pub node_id: String,
    /// Content identifier being stored.
    pub cid: String,
    /// Millisecond timestamp when the replica was stored.
    pub stored_at: u64,
    /// Millisecond timestamp of the last successful verification, if any.
    pub verified_at: Option<u64>,
    /// Checksum of the stored content.
    pub checksum: u64,
}

// ── ReplicationPolicy ─────────────────────────────────────────────────────────

/// Policy governing how many replicas are maintained and which nodes are chosen.
#[derive(Debug, Clone)]
pub struct RmReplicationPolicy {
    /// Target number of replicas per CID.
    pub replication_factor: u32,
    /// Minimum number of healthy replicas required for a CID to be `Healthy`.
    pub min_healthy_replicas: u32,
    /// Prefer nodes that are in the same locality.
    pub prefer_local: bool,
    /// Nodes above this utilization fraction are not eligible for new replicas.
    pub max_node_utilization: f64,
}

impl Default for RmReplicationPolicy {
    fn default() -> Self {
        Self {
            replication_factor: 3,
            min_healthy_replicas: 2,
            prefer_local: false,
            max_node_utilization: 0.85,
        }
    }
}

// ── ReplicationStatus ─────────────────────────────────────────────────────────

/// Health status of a CID's replication across the cluster.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RmReplicationStatus {
    /// Healthy replicas meet or exceed `min_healthy_replicas`.
    Healthy {
        /// Number of healthy replicas currently known.
        replicas: u32,
    },
    /// Healthy replicas are below `min_healthy_replicas`.
    UnderReplicated {
        /// Number of healthy replicas currently known.
        current: u32,
        /// Target replication factor.
        target: u32,
    },
    /// Healthy replicas exceed `replication_factor`.
    OverReplicated {
        /// Number of healthy replicas currently known.
        current: u32,
        /// Target replication factor.
        target: u32,
    },
    /// No replicas exist for this CID at all.
    Missing,
}

// ── ReplicationStats ──────────────────────────────────────────────────────────

/// Aggregate statistics for [`StorageReplicationManager`].
#[derive(Debug, Clone)]
pub struct RmReplicationStats {
    /// Total number of distinct CIDs being tracked.
    pub total_cids: usize,
    /// CIDs whose status is `Healthy`.
    pub healthy_cids: usize,
    /// CIDs whose status is `UnderReplicated`.
    pub under_replicated_cids: usize,
    /// CIDs whose status is `OverReplicated`.
    pub over_replicated_cids: usize,
    /// CIDs whose status is `Missing`.
    pub missing_cids: usize,
    /// Total number of nodes registered.
    pub total_nodes: usize,
    /// Number of nodes that are currently healthy.
    pub healthy_nodes: usize,
    /// Mean number of healthy replicas across all tracked CIDs.
    pub avg_replication_factor: f64,
}

// ── StorageReplicationManager ─────────────────────────────────────────────────

/// Manages block replication across a set of storage nodes.
///
/// Tracks the health and capacity of each node, records where each CID is
/// stored, and exposes methods to query replication status, select candidate
/// nodes for new replicas, and evict stale nodes.
pub struct StorageReplicationManager {
    /// Policy controlling replication factors and node selection.
    pub policy: RmReplicationPolicy,
    /// Registered nodes, keyed by `node_id`.
    pub nodes: HashMap<String, ReplicaNode>,
    /// Replica locations, keyed by CID. Each entry is a list of locations.
    pub replicas: HashMap<String, Vec<RmReplicaLocation>>,
}

impl StorageReplicationManager {
    /// Create a new manager with the supplied policy.
    pub fn new(policy: RmReplicationPolicy) -> Self {
        Self {
            policy,
            nodes: HashMap::new(),
            replicas: HashMap::new(),
        }
    }

    // ── Node management ───────────────────────────────────────────────

    /// Register a new storage node.
    ///
    /// Returns [`ReplicationError::NodeAlreadyRegistered`] if a node with the
    /// same `node_id` is already present.
    pub fn register_node(&mut self, node: ReplicaNode) -> Result<(), ReplicationError> {
        if self.nodes.contains_key(&node.node_id) {
            return Err(ReplicationError::NodeAlreadyRegistered(
                node.node_id.clone(),
            ));
        }
        self.nodes.insert(node.node_id.clone(), node);
        Ok(())
    }

    /// Remove a node and all replica records that reference it.
    ///
    /// Returns [`ReplicationError::NodeNotFound`] if the node is not present.
    pub fn deregister_node(&mut self, node_id: &str) -> Result<(), ReplicationError> {
        if self.nodes.remove(node_id).is_none() {
            return Err(ReplicationError::NodeNotFound(node_id.to_string()));
        }
        // Remove replica records that referenced the removed node.
        for locations in self.replicas.values_mut() {
            locations.retain(|loc| loc.node_id != node_id);
        }
        // Drop CID entries that are now empty.
        self.replicas.retain(|_, locs| !locs.is_empty());
        Ok(())
    }

    /// Mark a node as healthy and update its `last_seen` timestamp.
    ///
    /// Returns `true` if the node was found.
    pub fn mark_healthy(&mut self, node_id: &str, now: u64) -> bool {
        if let Some(node) = self.nodes.get_mut(node_id) {
            node.healthy = true;
            node.last_seen = now;
            true
        } else {
            false
        }
    }

    /// Mark a node as unhealthy.
    ///
    /// Returns `true` if the node was found.
    pub fn mark_unhealthy(&mut self, node_id: &str) -> bool {
        if let Some(node) = self.nodes.get_mut(node_id) {
            node.healthy = false;
            true
        } else {
            false
        }
    }

    // ── Replica management ────────────────────────────────────────────

    /// Record that a CID has been stored on a node.
    ///
    /// Returns [`ReplicationError::NodeNotFound`] if `node_id` is not registered.
    pub fn record_replica(
        &mut self,
        cid: String,
        node_id: &str,
        checksum: u64,
        now: u64,
    ) -> Result<(), ReplicationError> {
        if !self.nodes.contains_key(node_id) {
            return Err(ReplicationError::NodeNotFound(node_id.to_string()));
        }
        let location = RmReplicaLocation {
            node_id: node_id.to_string(),
            cid: cid.clone(),
            stored_at: now,
            verified_at: None,
            checksum,
        };
        self.replicas.entry(cid).or_default().push(location);
        Ok(())
    }

    /// Update the `verified_at` timestamp for a specific CID + node pair.
    ///
    /// Returns `true` if the replica record was found and updated.
    pub fn verify_replica(&mut self, cid: &str, node_id: &str, now: u64) -> bool {
        if let Some(locations) = self.replicas.get_mut(cid) {
            for loc in locations.iter_mut() {
                if loc.node_id == node_id {
                    loc.verified_at = Some(now);
                    return true;
                }
            }
        }
        false
    }

    /// Remove a specific replica location.
    ///
    /// Returns `true` if the location was found and removed.
    pub fn remove_replica(&mut self, cid: &str, node_id: &str) -> bool {
        if let Some(locations) = self.replicas.get_mut(cid) {
            let before = locations.len();
            locations.retain(|loc| loc.node_id != node_id);
            let removed = locations.len() < before;
            if locations.is_empty() {
                self.replicas.remove(cid);
            }
            removed
        } else {
            false
        }
    }

    // ── Status queries ────────────────────────────────────────────────

    /// Return the replication status for a CID.
    ///
    /// A "healthy replica" is one where both the replica record exists **and**
    /// the backing node is registered and marked healthy.
    pub fn replication_status(&self, cid: &str) -> RmReplicationStatus {
        let Some(locations) = self.replicas.get(cid) else {
            return RmReplicationStatus::Missing;
        };

        let healthy_count = self.count_healthy_replicas(locations);
        let target = self.policy.replication_factor;
        let min_healthy = self.policy.min_healthy_replicas;

        if healthy_count == 0 {
            // There are location records but no healthy nodes backing them.
            RmReplicationStatus::UnderReplicated { current: 0, target }
        } else if healthy_count > target {
            RmReplicationStatus::OverReplicated {
                current: healthy_count,
                target,
            }
        } else if healthy_count >= min_healthy {
            RmReplicationStatus::Healthy {
                replicas: healthy_count,
            }
        } else {
            RmReplicationStatus::UnderReplicated {
                current: healthy_count,
                target,
            }
        }
    }

    /// Return all CIDs that are `UnderReplicated` or `Missing`, sorted.
    pub fn under_replicated_cids(&self) -> Vec<&str> {
        let mut result: Vec<&str> = self
            .all_known_cids()
            .filter(|cid| {
                matches!(
                    self.replication_status(cid),
                    RmReplicationStatus::UnderReplicated { .. } | RmReplicationStatus::Missing
                )
            })
            .collect();
        result.sort_unstable();
        result
    }

    /// Return all CIDs that are `OverReplicated`, sorted.
    pub fn over_replicated_cids(&self) -> Vec<&str> {
        let mut result: Vec<&str> = self
            .all_known_cids()
            .filter(|cid| {
                matches!(
                    self.replication_status(cid),
                    RmReplicationStatus::OverReplicated { .. }
                )
            })
            .collect();
        result.sort_unstable();
        result
    }

    /// Select up to `count` healthy nodes suitable for storing a new replica of `cid`.
    ///
    /// Eligibility criteria (in order of filtering):
    /// 1. Node must be healthy.
    /// 2. Node must not already hold a replica of the given CID.
    /// 3. Node's utilization must be ≤ `max_node_utilization`.
    ///
    /// Among eligible nodes, those with the most `available_bytes` are preferred.
    pub fn select_nodes_for_replication(&self, cid: &str, count: usize) -> Vec<&ReplicaNode> {
        // Collect node IDs that already have a replica for this CID.
        let already_has: std::collections::HashSet<&str> = self
            .replicas
            .get(cid)
            .map(|locs| locs.iter().map(|l| l.node_id.as_str()).collect())
            .unwrap_or_default();

        let max_util = self.policy.max_node_utilization;

        let mut candidates: Vec<&ReplicaNode> = self
            .nodes
            .values()
            .filter(|n| {
                n.healthy
                    && !already_has.contains(n.node_id.as_str())
                    && n.utilization() <= max_util
            })
            .collect();

        // Sort descending by available bytes so we pick the most-available nodes first.
        candidates.sort_by_key(|n| std::cmp::Reverse(n.available_bytes()));
        candidates.truncate(count);
        candidates
    }

    // ── Maintenance ───────────────────────────────────────────────────

    /// Remove nodes whose `last_seen` is more than `max_age_ms` milliseconds
    /// before `now`, and clean up associated replica records.
    ///
    /// Returns the number of nodes evicted.
    pub fn evict_stale_nodes(&mut self, max_age_ms: u64, now: u64) -> usize {
        let stale_ids: Vec<String> = self
            .nodes
            .iter()
            .filter(|(_, n)| now.saturating_sub(n.last_seen) > max_age_ms)
            .map(|(id, _)| id.clone())
            .collect();

        let count = stale_ids.len();
        for id in &stale_ids {
            self.nodes.remove(id);
            for locations in self.replicas.values_mut() {
                locations.retain(|loc| &loc.node_id != id);
            }
        }
        self.replicas.retain(|_, locs| !locs.is_empty());
        count
    }

    // ── Counts ────────────────────────────────────────────────────────

    /// Returns the total number of distinct CIDs being tracked.
    pub fn cid_count(&self) -> usize {
        self.replicas.len()
    }

    /// Returns the total number of registered nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Returns the number of nodes currently marked as healthy.
    pub fn healthy_node_count(&self) -> usize {
        self.nodes.values().filter(|n| n.healthy).count()
    }

    // ── Statistics ────────────────────────────────────────────────────

    /// Compute aggregate statistics for the current state.
    pub fn stats(&self) -> RmReplicationStats {
        let total_cids = self.cid_count();
        let total_nodes = self.node_count();
        let healthy_nodes = self.healthy_node_count();

        let mut healthy_cids = 0usize;
        let mut under_replicated_cids = 0usize;
        let mut over_replicated_cids = 0usize;
        let mut missing_cids = 0usize;
        let mut total_healthy_replicas = 0u64;

        for cid in self.replicas.keys() {
            match self.replication_status(cid) {
                RmReplicationStatus::Healthy { replicas } => {
                    healthy_cids += 1;
                    total_healthy_replicas += u64::from(replicas);
                }
                RmReplicationStatus::UnderReplicated { current, .. } => {
                    under_replicated_cids += 1;
                    total_healthy_replicas += u64::from(current);
                }
                RmReplicationStatus::OverReplicated { current, .. } => {
                    over_replicated_cids += 1;
                    total_healthy_replicas += u64::from(current);
                }
                RmReplicationStatus::Missing => {
                    missing_cids += 1;
                }
            }
        }

        let avg_replication_factor = if total_cids == 0 {
            0.0
        } else {
            total_healthy_replicas as f64 / total_cids as f64
        };

        RmReplicationStats {
            total_cids,
            healthy_cids,
            under_replicated_cids,
            over_replicated_cids,
            missing_cids,
            total_nodes,
            healthy_nodes,
            avg_replication_factor,
        }
    }

    // ── Private helpers ───────────────────────────────────────────────

    fn count_healthy_replicas(&self, locations: &[RmReplicaLocation]) -> u32 {
        locations
            .iter()
            .filter(|loc| {
                self.nodes
                    .get(&loc.node_id)
                    .map(|n| n.healthy)
                    .unwrap_or(false)
            })
            .count() as u32
    }

    fn all_known_cids(&self) -> impl Iterator<Item = &str> {
        self.replicas.keys().map(|s| s.as_str())
    }
}

// ── Re-exports kept for backward compatibility ────────────────────────────────

/// Configuration for the replication manager (legacy name kept for backwards
/// compatibility; prefer [`RmReplicationPolicy`] for new code).
#[derive(Debug, Clone)]
pub struct ReplicationConfig {
    /// Default number of replicas desired per block.
    pub default_replica_count: usize,
    /// Maximum number of replication attempts before giving up.
    pub max_attempts: u32,
    /// Number of ticks to wait before retrying a failed replication.
    pub retry_interval_ticks: u64,
}

impl Default for ReplicationConfig {
    fn default() -> Self {
        Self {
            default_replica_count: 3,
            max_attempts: 5,
            retry_interval_ticks: 10,
        }
    }
}

/// State of a single replica of a block (legacy type).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplicationState {
    /// Replica has been registered but replication has not started.
    Pending,
    /// Replication is currently in progress.
    InProgress,
    /// Replication completed successfully.
    Completed,
    /// Replication failed.
    Failed,
}

/// Information about a single replica of a block on a specific peer (legacy type).
#[derive(Debug, Clone)]
pub struct ReplicaInfo {
    /// The peer holding (or targeted to hold) this replica.
    pub peer_id: String,
    /// Current state of replication to this peer.
    pub state: ReplicationState,
    /// The tick at which replication completed, if applicable.
    pub replicated_tick: Option<u64>,
    /// Number of replication attempts made.
    pub attempts: u32,
}

/// Tracks all replicas of a single block (legacy type).
#[derive(Debug, Clone)]
pub struct BlockReplicas {
    /// CID of the block being replicated.
    pub block_cid: String,
    /// Desired number of replicas.
    pub desired_replicas: usize,
    /// Current replicas (one per target peer).
    pub replicas: Vec<ReplicaInfo>,
}

/// Aggregate statistics for the replication manager (legacy type).
#[derive(Debug, Clone)]
pub struct ReplicationManagerStats {
    /// Number of blocks currently tracked.
    pub tracked_blocks: usize,
    /// Total number of successful replications since creation.
    pub total_replications: u64,
    /// Total number of replication failures since creation.
    pub total_failures: u64,
    /// Number of blocks that are under-replicated.
    pub under_replicated: usize,
    /// Number of blocks that are fully replicated.
    pub fully_replicated: usize,
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use crate::replication_manager::{
        BlockReplicas, ReplicaInfo, ReplicaNode, ReplicationConfig, ReplicationError,
        ReplicationManagerStats, ReplicationState, RmReplicaLocation, RmReplicationPolicy,
        RmReplicationStatus, StorageReplicationManager,
    };

    // ── helpers ───────────────────────────────────────────────────────

    fn default_policy() -> RmReplicationPolicy {
        RmReplicationPolicy::default()
    }

    fn make_manager() -> StorageReplicationManager {
        StorageReplicationManager::new(default_policy())
    }

    fn healthy_node(id: &str, capacity: u64, used: u64) -> ReplicaNode {
        ReplicaNode {
            node_id: id.to_string(),
            address: format!("192.168.0.1:{id}"),
            last_seen: 1000,
            healthy: true,
            capacity_bytes: capacity,
            used_bytes: used,
        }
    }

    fn unhealthy_node(id: &str) -> ReplicaNode {
        ReplicaNode {
            node_id: id.to_string(),
            address: format!("192.168.0.2:{id}"),
            last_seen: 500,
            healthy: false,
            capacity_bytes: 1_000_000,
            used_bytes: 100_000,
        }
    }

    // ── ReplicaNode ───────────────────────────────────────────────────

    #[test]
    fn test_replica_node_available_bytes() {
        let n = healthy_node("n1", 1_000, 300);
        assert_eq!(n.available_bytes(), 700);
    }

    #[test]
    fn test_replica_node_available_bytes_saturates() {
        let n = ReplicaNode {
            node_id: "n".into(),
            address: "a".into(),
            last_seen: 0,
            healthy: true,
            capacity_bytes: 100,
            used_bytes: 200,
        };
        assert_eq!(n.available_bytes(), 0);
    }

    #[test]
    fn test_replica_node_utilization() {
        let n = healthy_node("n1", 1_000, 500);
        assert!((n.utilization() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_replica_node_utilization_zero_capacity() {
        let n = ReplicaNode {
            node_id: "n".into(),
            address: "a".into(),
            last_seen: 0,
            healthy: true,
            capacity_bytes: 0,
            used_bytes: 0,
        };
        // capacity.max(1) == 1, used == 0 => 0.0
        assert!((n.utilization() - 0.0).abs() < f64::EPSILON);
    }

    // ── register_node / deregister_node ───────────────────────────────

    #[test]
    fn test_register_node_success() {
        let mut mgr = make_manager();
        let result = mgr.register_node(healthy_node("n1", 1_000_000, 0));
        assert!(result.is_ok());
        assert_eq!(mgr.node_count(), 1);
    }

    #[test]
    fn test_register_node_duplicate_error() {
        let mut mgr = make_manager();
        mgr.register_node(healthy_node("n1", 1_000_000, 0))
            .unwrap_or_default();
        let err = mgr
            .register_node(healthy_node("n1", 500_000, 0))
            .unwrap_err();
        assert_eq!(err, ReplicationError::NodeAlreadyRegistered("n1".into()));
    }

    #[test]
    fn test_deregister_node_success() {
        let mut mgr = make_manager();
        mgr.register_node(healthy_node("n1", 1_000_000, 0))
            .unwrap_or_default();
        let result = mgr.deregister_node("n1");
        assert!(result.is_ok());
        assert_eq!(mgr.node_count(), 0);
    }

    #[test]
    fn test_deregister_node_not_found() {
        let mut mgr = make_manager();
        let err = mgr.deregister_node("ghost").unwrap_err();
        assert_eq!(err, ReplicationError::NodeNotFound("ghost".into()));
    }

    #[test]
    fn test_deregister_node_removes_replicas() {
        let mut mgr = make_manager();
        mgr.register_node(healthy_node("n1", 1_000_000, 0))
            .unwrap_or_default();
        mgr.record_replica("cid1".into(), "n1", 42, 100)
            .unwrap_or_default();
        assert_eq!(mgr.cid_count(), 1);
        mgr.deregister_node("n1").unwrap_or_default();
        // CID entry should be removed because it has no remaining locations.
        assert_eq!(mgr.cid_count(), 0);
    }

    // ── mark_healthy / mark_unhealthy ─────────────────────────────────

    #[test]
    fn test_mark_healthy_updates_node() {
        let mut mgr = make_manager();
        mgr.register_node(unhealthy_node("n1")).unwrap_or_default();
        assert!(mgr.mark_healthy("n1", 9999));
        let n = mgr.nodes.get("n1").expect("node must exist");
        assert!(n.healthy);
        assert_eq!(n.last_seen, 9999);
    }

    #[test]
    fn test_mark_healthy_returns_false_unknown() {
        let mut mgr = make_manager();
        assert!(!mgr.mark_healthy("nope", 1));
    }

    #[test]
    fn test_mark_unhealthy() {
        let mut mgr = make_manager();
        mgr.register_node(healthy_node("n1", 1_000_000, 0))
            .unwrap_or_default();
        assert!(mgr.mark_unhealthy("n1"));
        let n = mgr.nodes.get("n1").expect("node must exist");
        assert!(!n.healthy);
    }

    #[test]
    fn test_mark_unhealthy_returns_false_unknown() {
        let mut mgr = make_manager();
        assert!(!mgr.mark_unhealthy("nope"));
    }

    // ── record_replica / verify_replica / remove_replica ─────────────

    #[test]
    fn test_record_replica_success() {
        let mut mgr = make_manager();
        mgr.register_node(healthy_node("n1", 1_000_000, 0))
            .unwrap_or_default();
        let r = mgr.record_replica("cid1".into(), "n1", 0xdeadbeef, 200);
        assert!(r.is_ok());
        assert_eq!(mgr.cid_count(), 1);
    }

    #[test]
    fn test_record_replica_node_not_found() {
        let mut mgr = make_manager();
        let err = mgr
            .record_replica("cid1".into(), "ghost", 0, 0)
            .unwrap_err();
        assert_eq!(err, ReplicationError::NodeNotFound("ghost".into()));
    }

    #[test]
    fn test_verify_replica_found() {
        let mut mgr = make_manager();
        mgr.register_node(healthy_node("n1", 1_000_000, 0))
            .unwrap_or_default();
        mgr.record_replica("cid1".into(), "n1", 0, 100)
            .unwrap_or_default();
        assert!(mgr.verify_replica("cid1", "n1", 300));
        let loc = &mgr.replicas["cid1"][0];
        assert_eq!(loc.verified_at, Some(300));
    }

    #[test]
    fn test_verify_replica_not_found() {
        let mut mgr = make_manager();
        assert!(!mgr.verify_replica("cid_missing", "n1", 0));
    }

    #[test]
    fn test_remove_replica_success() {
        let mut mgr = make_manager();
        mgr.register_node(healthy_node("n1", 1_000_000, 0))
            .unwrap_or_default();
        mgr.record_replica("cid1".into(), "n1", 0, 100)
            .unwrap_or_default();
        assert!(mgr.remove_replica("cid1", "n1"));
        assert_eq!(mgr.cid_count(), 0);
    }

    #[test]
    fn test_remove_replica_not_found() {
        let mut mgr = make_manager();
        assert!(!mgr.remove_replica("cid_missing", "n1"));
    }

    // ── replication_status ────────────────────────────────────────────

    #[test]
    fn test_status_missing_no_cid() {
        let mgr = make_manager();
        assert_eq!(mgr.replication_status("cid1"), RmReplicationStatus::Missing);
    }

    #[test]
    fn test_status_under_replicated_no_healthy_nodes() {
        let mut mgr = make_manager();
        mgr.register_node(unhealthy_node("n1")).unwrap_or_default();
        mgr.record_replica("cid1".into(), "n1", 0, 100)
            .unwrap_or_default();
        assert_eq!(
            mgr.replication_status("cid1"),
            RmReplicationStatus::UnderReplicated {
                current: 0,
                target: 3
            }
        );
    }

    #[test]
    fn test_status_healthy() {
        let mut mgr = make_manager();
        for i in 0..2u8 {
            let id = format!("n{i}");
            mgr.register_node(healthy_node(&id, 1_000_000, 0))
                .unwrap_or_default();
            mgr.record_replica("cid1".into(), &id, 0, 100)
                .unwrap_or_default();
        }
        assert_eq!(
            mgr.replication_status("cid1"),
            RmReplicationStatus::Healthy { replicas: 2 }
        );
    }

    #[test]
    fn test_status_over_replicated() {
        let mut mgr = StorageReplicationManager::new(RmReplicationPolicy {
            replication_factor: 2,
            min_healthy_replicas: 1,
            ..Default::default()
        });
        for i in 0..3u8 {
            let id = format!("n{i}");
            mgr.register_node(healthy_node(&id, 1_000_000, 0))
                .unwrap_or_default();
            mgr.record_replica("cid1".into(), &id, 0, 100)
                .unwrap_or_default();
        }
        assert_eq!(
            mgr.replication_status("cid1"),
            RmReplicationStatus::OverReplicated {
                current: 3,
                target: 2
            }
        );
    }

    // ── under_replicated_cids / over_replicated_cids ──────────────────

    #[test]
    fn test_under_replicated_cids_sorted() {
        let mut mgr = make_manager();
        // Register a single unhealthy node and record replicas on it — both CIDs
        // will be under-replicated.
        mgr.register_node(unhealthy_node("n1")).unwrap_or_default();
        mgr.record_replica("cid_b".into(), "n1", 0, 1)
            .unwrap_or_default();
        mgr.record_replica("cid_a".into(), "n1", 0, 1)
            .unwrap_or_default();
        let under = mgr.under_replicated_cids();
        assert_eq!(under, vec!["cid_a", "cid_b"]);
    }

    #[test]
    fn test_over_replicated_cids_sorted() {
        let mut mgr = StorageReplicationManager::new(RmReplicationPolicy {
            replication_factor: 1,
            min_healthy_replicas: 1,
            ..Default::default()
        });
        for suffix in ["z", "a", "m"] {
            let cid = format!("cid_{suffix}");
            for i in 0..2u8 {
                let node_id = format!("n{suffix}{i}");
                mgr.register_node(healthy_node(&node_id, 1_000_000, 0))
                    .unwrap_or_default();
                mgr.record_replica(cid.clone(), &node_id, 0, 1)
                    .unwrap_or_default();
            }
        }
        let over = mgr.over_replicated_cids();
        assert_eq!(over, vec!["cid_a", "cid_m", "cid_z"]);
    }

    // ── select_nodes_for_replication ──────────────────────────────────

    #[test]
    fn test_select_excludes_existing_replica_node() {
        let mut mgr = make_manager();
        mgr.register_node(healthy_node("n1", 1_000_000, 0))
            .unwrap_or_default();
        mgr.register_node(healthy_node("n2", 1_000_000, 0))
            .unwrap_or_default();
        mgr.record_replica("cid1".into(), "n1", 0, 1)
            .unwrap_or_default();
        let selected = mgr.select_nodes_for_replication("cid1", 10);
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].node_id, "n2");
    }

    #[test]
    fn test_select_excludes_unhealthy_nodes() {
        let mut mgr = make_manager();
        mgr.register_node(healthy_node("n1", 1_000_000, 0))
            .unwrap_or_default();
        mgr.register_node(unhealthy_node("n2")).unwrap_or_default();
        let selected = mgr.select_nodes_for_replication("cid1", 10);
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].node_id, "n1");
    }

    #[test]
    fn test_select_excludes_over_utilized_nodes() {
        let mut mgr = make_manager();
        // 90% utilization — exceeds default max of 0.85
        mgr.register_node(healthy_node("n1", 1_000_000, 900_000))
            .unwrap_or_default();
        mgr.register_node(healthy_node("n2", 1_000_000, 100_000))
            .unwrap_or_default();
        let selected = mgr.select_nodes_for_replication("cid1", 10);
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].node_id, "n2");
    }

    #[test]
    fn test_select_sorted_by_available_bytes_desc() {
        let mut mgr = make_manager();
        mgr.register_node(healthy_node("n1", 1_000_000, 800_000))
            .unwrap_or_default(); // 200k free
        mgr.register_node(healthy_node("n2", 1_000_000, 100_000))
            .unwrap_or_default(); // 900k free
        mgr.register_node(healthy_node("n3", 1_000_000, 500_000))
            .unwrap_or_default(); // 500k free
        let selected = mgr.select_nodes_for_replication("cid1", 3);
        assert_eq!(selected.len(), 3);
        assert_eq!(selected[0].node_id, "n2");
        assert_eq!(selected[1].node_id, "n3");
        assert_eq!(selected[2].node_id, "n1");
    }

    #[test]
    fn test_select_count_limits_result() {
        let mut mgr = make_manager();
        for i in 0..5u8 {
            mgr.register_node(healthy_node(&format!("n{i}"), 1_000_000, 0))
                .unwrap_or_default();
        }
        let selected = mgr.select_nodes_for_replication("cid1", 2);
        assert_eq!(selected.len(), 2);
    }

    #[test]
    fn test_select_returns_empty_when_no_eligible() {
        let mut mgr = make_manager();
        mgr.register_node(unhealthy_node("n1")).unwrap_or_default();
        let selected = mgr.select_nodes_for_replication("cid1", 5);
        assert!(selected.is_empty());
    }

    // ── evict_stale_nodes ─────────────────────────────────────────────

    #[test]
    fn test_evict_stale_nodes_removes_old() {
        let mut mgr = make_manager();
        mgr.register_node(ReplicaNode {
            node_id: "old".into(),
            address: "a".into(),
            last_seen: 100,
            healthy: true,
            capacity_bytes: 1_000_000,
            used_bytes: 0,
        })
        .unwrap_or_default();
        mgr.register_node(ReplicaNode {
            node_id: "fresh".into(),
            address: "b".into(),
            last_seen: 9_000,
            healthy: true,
            capacity_bytes: 1_000_000,
            used_bytes: 0,
        })
        .unwrap_or_default();
        // now=10_000, max_age=1_000 → "old" (last_seen=100, age=9900) evicted,
        // "fresh" (age=1000, NOT >1000) kept.
        let evicted = mgr.evict_stale_nodes(1_000, 10_000);
        assert_eq!(evicted, 1);
        assert_eq!(mgr.node_count(), 1);
        assert!(mgr.nodes.contains_key("fresh"));
    }

    #[test]
    fn test_evict_stale_cleans_replicas() {
        let mut mgr = make_manager();
        mgr.register_node(ReplicaNode {
            node_id: "old".into(),
            address: "a".into(),
            last_seen: 0,
            healthy: true,
            capacity_bytes: 1_000_000,
            used_bytes: 0,
        })
        .unwrap_or_default();
        mgr.record_replica("cid1".into(), "old", 0, 1)
            .unwrap_or_default();
        mgr.evict_stale_nodes(500, 10_000);
        assert_eq!(mgr.cid_count(), 0);
    }

    #[test]
    fn test_evict_stale_none_old() {
        let mut mgr = make_manager();
        mgr.register_node(healthy_node("n1", 1_000_000, 0))
            .unwrap_or_default();
        let evicted = mgr.evict_stale_nodes(10_000, 2_000);
        assert_eq!(evicted, 0);
        assert_eq!(mgr.node_count(), 1);
    }

    // ── cid_count / node_count / healthy_node_count ───────────────────

    #[test]
    fn test_node_count() {
        let mut mgr = make_manager();
        assert_eq!(mgr.node_count(), 0);
        mgr.register_node(healthy_node("n1", 1_000_000, 0))
            .unwrap_or_default();
        mgr.register_node(healthy_node("n2", 1_000_000, 0))
            .unwrap_or_default();
        assert_eq!(mgr.node_count(), 2);
    }

    #[test]
    fn test_healthy_node_count() {
        let mut mgr = make_manager();
        mgr.register_node(healthy_node("n1", 1_000_000, 0))
            .unwrap_or_default();
        mgr.register_node(unhealthy_node("n2")).unwrap_or_default();
        assert_eq!(mgr.healthy_node_count(), 1);
    }

    #[test]
    fn test_cid_count() {
        let mut mgr = make_manager();
        mgr.register_node(healthy_node("n1", 1_000_000, 0))
            .unwrap_or_default();
        mgr.record_replica("cid1".into(), "n1", 0, 1)
            .unwrap_or_default();
        mgr.record_replica("cid2".into(), "n1", 0, 1)
            .unwrap_or_default();
        assert_eq!(mgr.cid_count(), 2);
    }

    // ── stats ─────────────────────────────────────────────────────────

    #[test]
    fn test_stats_empty() {
        let mgr = make_manager();
        let s = mgr.stats();
        assert_eq!(s.total_cids, 0);
        assert_eq!(s.total_nodes, 0);
        assert_eq!(s.healthy_nodes, 0);
        assert!((s.avg_replication_factor - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_stats_all_healthy() {
        let mut mgr = make_manager();
        for i in 0..3u8 {
            let id = format!("n{i}");
            mgr.register_node(healthy_node(&id, 1_000_000, 0))
                .unwrap_or_default();
            mgr.record_replica("cid1".into(), &id, 0, 1)
                .unwrap_or_default();
        }
        let s = mgr.stats();
        assert_eq!(s.total_cids, 1);
        assert_eq!(s.healthy_cids, 1);
        assert_eq!(s.under_replicated_cids, 0);
        assert_eq!(s.over_replicated_cids, 0);
        assert_eq!(s.missing_cids, 0);
        assert!((s.avg_replication_factor - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_stats_under_and_over() {
        // policy: factor=2, min_healthy=2
        let policy = RmReplicationPolicy {
            replication_factor: 2,
            min_healthy_replicas: 2,
            ..Default::default()
        };
        let mut mgr = StorageReplicationManager::new(policy);
        // "cid_over": 3 healthy replicas on factor=2 → OverReplicated
        for i in 0..3u8 {
            let id = format!("over_n{i}");
            mgr.register_node(healthy_node(&id, 1_000_000, 0))
                .unwrap_or_default();
            mgr.record_replica("cid_over".into(), &id, 0, 1)
                .unwrap_or_default();
        }
        // "cid_under": 1 healthy replica → UnderReplicated
        mgr.register_node(healthy_node("under_n0", 1_000_000, 0))
            .unwrap_or_default();
        mgr.record_replica("cid_under".into(), "under_n0", 0, 1)
            .unwrap_or_default();

        let s = mgr.stats();
        assert_eq!(s.over_replicated_cids, 1);
        assert_eq!(s.under_replicated_cids, 1);
        assert_eq!(s.healthy_cids, 0);
    }

    // ── error display ─────────────────────────────────────────────────

    #[test]
    fn test_error_display_node_already_registered() {
        let e = ReplicationError::NodeAlreadyRegistered("n1".into());
        assert!(e.to_string().contains("n1"));
    }

    #[test]
    fn test_error_display_node_not_found() {
        let e = ReplicationError::NodeNotFound("n2".into());
        assert!(e.to_string().contains("n2"));
    }

    #[test]
    fn test_error_display_cid_not_found() {
        let e = ReplicationError::CidNotFound("cid_x".into());
        assert!(e.to_string().contains("cid_x"));
    }

    // ── policy defaults ───────────────────────────────────────────────

    #[test]
    fn test_policy_defaults() {
        let p = RmReplicationPolicy::default();
        assert_eq!(p.replication_factor, 3);
        assert_eq!(p.min_healthy_replicas, 2);
        assert!(!p.prefer_local);
        assert!((p.max_node_utilization - 0.85).abs() < f64::EPSILON);
    }

    // ── legacy type smoke-tests (kept for backwards compat) ───────────

    #[test]
    fn test_legacy_replication_config_default() {
        let cfg = ReplicationConfig::default();
        assert_eq!(cfg.default_replica_count, 3);
        assert_eq!(cfg.max_attempts, 5);
        assert_eq!(cfg.retry_interval_ticks, 10);
    }

    #[test]
    fn test_legacy_block_replicas_fields() {
        let br = BlockReplicas {
            block_cid: "cid".into(),
            desired_replicas: 2,
            replicas: vec![ReplicaInfo {
                peer_id: "p".into(),
                state: ReplicationState::Pending,
                replicated_tick: None,
                attempts: 0,
            }],
        };
        assert_eq!(br.desired_replicas, 2);
        assert_eq!(br.replicas[0].state, ReplicationState::Pending);
    }

    #[test]
    fn test_legacy_replication_manager_stats_fields() {
        let s = ReplicationManagerStats {
            tracked_blocks: 1,
            total_replications: 2,
            total_failures: 3,
            under_replicated: 4,
            fully_replicated: 5,
        };
        assert_eq!(s.tracked_blocks, 1);
        assert_eq!(s.fully_replicated, 5);
    }

    // ── deregister keeps other CIDs intact ───────────────────────────

    #[test]
    fn test_deregister_leaves_other_cid_replicas() {
        let mut mgr = make_manager();
        mgr.register_node(healthy_node("n1", 1_000_000, 0))
            .unwrap_or_default();
        mgr.register_node(healthy_node("n2", 1_000_000, 0))
            .unwrap_or_default();
        mgr.record_replica("cid1".into(), "n1", 0, 1)
            .unwrap_or_default();
        mgr.record_replica("cid1".into(), "n2", 0, 1)
            .unwrap_or_default();
        mgr.record_replica("cid2".into(), "n1", 0, 1)
            .unwrap_or_default();
        mgr.deregister_node("n1").unwrap_or_default();
        // cid1 still has n2; cid2 had only n1 → removed
        assert_eq!(mgr.cid_count(), 1);
        let locs = &mgr.replicas["cid1"];
        assert_eq!(locs.len(), 1);
        assert_eq!(locs[0].node_id, "n2");
    }

    // ── multiple replicas same CID ────────────────────────────────────

    #[test]
    fn test_multiple_replicas_same_cid() {
        let mut mgr = make_manager();
        for i in 0..5u8 {
            let id = format!("n{i}");
            mgr.register_node(healthy_node(&id, 1_000_000, 0))
                .unwrap_or_default();
            mgr.record_replica("cid1".into(), &id, i as u64, 1)
                .unwrap_or_default();
        }
        assert_eq!(mgr.replicas["cid1"].len(), 5);
    }

    // ── verify_replica updates correct entry ──────────────────────────

    #[test]
    fn test_verify_replica_updates_correct_location() {
        let mut mgr = make_manager();
        mgr.register_node(healthy_node("n1", 1_000_000, 0))
            .unwrap_or_default();
        mgr.register_node(healthy_node("n2", 1_000_000, 0))
            .unwrap_or_default();
        mgr.record_replica("cid1".into(), "n1", 0, 1)
            .unwrap_or_default();
        mgr.record_replica("cid1".into(), "n2", 0, 1)
            .unwrap_or_default();
        mgr.verify_replica("cid1", "n2", 999);
        let locs = &mgr.replicas["cid1"];
        let n1_loc = locs.iter().find(|l| l.node_id == "n1").expect("n1 loc");
        let n2_loc = locs.iter().find(|l| l.node_id == "n2").expect("n2 loc");
        assert_eq!(n1_loc.verified_at, None);
        assert_eq!(n2_loc.verified_at, Some(999));
    }

    // ── RmReplicaLocation fields ──────────────────────────────────────

    #[test]
    fn test_replica_location_fields() {
        let loc = RmReplicaLocation {
            node_id: "n1".into(),
            cid: "cid1".into(),
            stored_at: 100,
            verified_at: Some(200),
            checksum: 0xabcd,
        };
        assert_eq!(loc.checksum, 0xabcd);
        assert_eq!(loc.verified_at, Some(200));
    }
}
