//! Multi-Datacenter Support for Distributed Storage
//!
//! Provides datacenter-aware routing, cross-datacenter replication,
//! and latency-aware node selection for geo-distributed RAFT clusters.

use crate::raft::NodeId;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// Geographic region identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Region(String);

impl Region {
    /// Create a new region
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }

    /// Get the region name
    pub fn name(&self) -> &str {
        &self.0
    }
}

impl From<&str> for Region {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

impl std::fmt::Display for Region {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Datacenter identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DatacenterId(String);

impl DatacenterId {
    /// Create a new datacenter ID
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Get the datacenter ID string
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<&str> for DatacenterId {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

impl std::fmt::Display for DatacenterId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Datacenter metadata
#[derive(Debug, Clone)]
pub struct Datacenter {
    /// Datacenter ID
    pub id: DatacenterId,
    /// Geographic region
    pub region: Region,
    /// Nodes in this datacenter
    pub nodes: HashSet<NodeId>,
    /// Priority (higher = preferred for reads)
    pub priority: i32,
}

impl Datacenter {
    /// Create a new datacenter
    pub fn new(id: DatacenterId, region: Region) -> Self {
        Self {
            id,
            region,
            nodes: HashSet::new(),
            priority: 0,
        }
    }

    /// Add a node to this datacenter
    pub fn add_node(&mut self, node_id: NodeId) {
        self.nodes.insert(node_id);
    }

    /// Remove a node from this datacenter
    pub fn remove_node(&mut self, node_id: &NodeId) -> bool {
        self.nodes.remove(node_id)
    }

    /// Check if datacenter contains a node
    pub fn has_node(&self, node_id: &NodeId) -> bool {
        self.nodes.contains(node_id)
    }
}

/// Multi-datacenter coordinator
pub struct MultiDatacenterCoordinator {
    /// Datacenters indexed by ID
    datacenters: HashMap<DatacenterId, Datacenter>,
    /// Node to datacenter mapping
    node_to_dc: HashMap<NodeId, DatacenterId>,
    /// Cross-datacenter latency measurements (ms)
    latencies: HashMap<(DatacenterId, DatacenterId), u64>,
}

impl MultiDatacenterCoordinator {
    /// Create a new multi-datacenter coordinator
    pub fn new() -> Self {
        Self {
            datacenters: HashMap::new(),
            node_to_dc: HashMap::new(),
            latencies: HashMap::new(),
        }
    }

    /// Add a datacenter
    pub fn add_datacenter(&mut self, dc: Datacenter) {
        self.datacenters.insert(dc.id.clone(), dc);
    }

    /// Register a node in a datacenter
    pub fn register_node(&mut self, node_id: NodeId, dc_id: DatacenterId) -> Result<(), String> {
        let dc = self
            .datacenters
            .get_mut(&dc_id)
            .ok_or_else(|| format!("Datacenter {dc_id} not found"))?;

        dc.add_node(node_id);
        self.node_to_dc.insert(node_id, dc_id);
        Ok(())
    }

    /// Unregister a node
    pub fn unregister_node(&mut self, node_id: &NodeId) {
        if let Some(dc_id) = self.node_to_dc.remove(node_id) {
            if let Some(dc) = self.datacenters.get_mut(&dc_id) {
                dc.remove_node(node_id);
            }
        }
    }

    /// Get the datacenter for a node
    pub fn get_node_datacenter(&self, node_id: &NodeId) -> Option<&Datacenter> {
        self.node_to_dc
            .get(node_id)
            .and_then(|dc_id| self.datacenters.get(dc_id))
    }

    /// Record latency between two datacenters
    pub fn record_latency(&mut self, from: DatacenterId, to: DatacenterId, latency_ms: u64) {
        self.latencies
            .insert((from.clone(), to.clone()), latency_ms);
        // Also record reverse direction (assume symmetric)
        self.latencies.insert((to, from), latency_ms);
    }

    /// Get latency between two datacenters
    pub fn get_latency(&self, from: &DatacenterId, to: &DatacenterId) -> Option<u64> {
        self.latencies.get(&(from.clone(), to.clone())).copied()
    }

    /// Get all datacenters
    pub fn datacenters(&self) -> &HashMap<DatacenterId, Datacenter> {
        &self.datacenters
    }

    /// Get total number of nodes across all datacenters
    pub fn total_nodes(&self) -> usize {
        self.node_to_dc.len()
    }
}

impl Default for MultiDatacenterCoordinator {
    fn default() -> Self {
        Self::new()
    }
}

/// Replication policy for multi-datacenter setups
#[derive(Debug, Clone)]
pub enum ReplicationPolicy {
    /// Replicate to all datacenters
    AllDatacenters,
    /// Replicate to specific regions
    Regions(Vec<Region>),
    /// Replicate to N closest datacenters (by latency)
    NClosest(usize),
    /// Custom policy with specific datacenter IDs
    Custom(Vec<DatacenterId>),
}

impl ReplicationPolicy {
    /// Select datacenters based on the policy
    pub fn select_datacenters(
        &self,
        coordinator: &MultiDatacenterCoordinator,
        source_dc: &DatacenterId,
    ) -> Vec<DatacenterId> {
        match self {
            ReplicationPolicy::AllDatacenters => coordinator.datacenters.keys().cloned().collect(),
            ReplicationPolicy::Regions(regions) => coordinator
                .datacenters
                .values()
                .filter(|dc| regions.contains(&dc.region))
                .map(|dc| dc.id.clone())
                .collect(),
            ReplicationPolicy::NClosest(n) => {
                let mut dcs: Vec<_> = coordinator
                    .datacenters
                    .keys()
                    .filter(|dc_id| *dc_id != source_dc)
                    .cloned()
                    .collect();

                // Sort by latency
                dcs.sort_by_key(|dc_id| {
                    coordinator
                        .get_latency(source_dc, dc_id)
                        .unwrap_or(u64::MAX)
                });

                dcs.into_iter().take(*n).collect()
            }
            ReplicationPolicy::Custom(dcs) => dcs.clone(),
        }
    }
}

/// Node selector for latency-aware routing
pub struct LatencyAwareSelector {
    coordinator: Arc<MultiDatacenterCoordinator>,
    /// Prefer local datacenter reads
    local_preference: bool,
    /// Maximum acceptable latency (ms)
    max_latency_ms: Option<u64>,
}

impl LatencyAwareSelector {
    /// Create a new latency-aware selector
    pub fn new(coordinator: Arc<MultiDatacenterCoordinator>) -> Self {
        Self {
            coordinator,
            local_preference: true,
            max_latency_ms: None,
        }
    }

    /// Enable/disable local datacenter preference
    pub fn with_local_preference(mut self, enabled: bool) -> Self {
        self.local_preference = enabled;
        self
    }

    /// Set maximum acceptable latency
    pub fn with_max_latency(mut self, latency_ms: u64) -> Self {
        self.max_latency_ms = Some(latency_ms);
        self
    }

    /// Select best nodes for a read operation
    pub fn select_read_nodes(
        &self,
        available_nodes: &[NodeId],
        local_node: &NodeId,
    ) -> Vec<NodeId> {
        let local_dc = self.coordinator.get_node_datacenter(local_node);

        let mut candidates: Vec<_> = available_nodes
            .iter()
            .filter_map(|node_id| {
                let node_dc = self.coordinator.get_node_datacenter(node_id)?;

                // Calculate latency
                let latency = if let Some(local) = local_dc {
                    self.coordinator
                        .get_latency(&local.id, &node_dc.id)
                        .unwrap_or(0)
                } else {
                    0
                };

                // Apply max latency filter
                if let Some(max_lat) = self.max_latency_ms {
                    if latency > max_lat {
                        return None;
                    }
                }

                Some((node_id, node_dc, latency))
            })
            .collect();

        // Sort by local preference and latency
        candidates.sort_by(|(_, dc1, lat1), (_, dc2, lat2)| {
            if let (true, Some(local)) = (self.local_preference, local_dc) {
                match (dc1.id == local.id, dc2.id == local.id) {
                    (true, false) => std::cmp::Ordering::Less,
                    (false, true) => std::cmp::Ordering::Greater,
                    _ => lat1.cmp(lat2),
                }
            } else {
                lat1.cmp(lat2)
            }
        });

        candidates
            .into_iter()
            .map(|(node_id, _, _)| *node_id)
            .collect()
    }
}

/// Cross-datacenter statistics
#[derive(Debug, Clone, Default)]
pub struct CrossDcStats {
    /// Number of cross-datacenter requests
    pub cross_dc_requests: u64,
    /// Number of local datacenter requests
    pub local_requests: u64,
    /// Average cross-datacenter latency (ms)
    pub avg_cross_dc_latency_ms: f64,
}

impl CrossDcStats {
    /// Create new statistics
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a cross-datacenter request
    pub fn record_cross_dc(&mut self, latency_ms: u64) {
        let total_latency = self.avg_cross_dc_latency_ms * self.cross_dc_requests as f64;
        self.cross_dc_requests += 1;
        self.avg_cross_dc_latency_ms =
            (total_latency + latency_ms as f64) / self.cross_dc_requests as f64;
    }

    /// Record a local datacenter request
    pub fn record_local(&mut self) {
        self.local_requests += 1;
    }

    /// Get total requests
    pub fn total_requests(&self) -> u64 {
        self.cross_dc_requests + self.local_requests
    }

    /// Get percentage of cross-datacenter requests
    pub fn cross_dc_percentage(&self) -> f64 {
        let total = self.total_requests();
        if total == 0 {
            0.0
        } else {
            (self.cross_dc_requests as f64 / total as f64) * 100.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_datacenter_creation() {
        let dc = Datacenter::new(DatacenterId::new("us-east-1"), Region::new("us-east"));

        assert_eq!(dc.id.as_str(), "us-east-1");
        assert_eq!(dc.region.name(), "us-east");
        assert_eq!(dc.nodes.len(), 0);
    }

    #[test]
    fn test_datacenter_nodes() {
        let mut dc = Datacenter::new(DatacenterId::new("us-west-2"), Region::new("us-west"));

        let node1 = NodeId(1);
        let node2 = NodeId(2);

        dc.add_node(node1);
        dc.add_node(node2);
        assert_eq!(dc.nodes.len(), 2);
        assert!(dc.has_node(&node1));

        assert!(dc.remove_node(&node1));
        assert_eq!(dc.nodes.len(), 1);
        assert!(!dc.has_node(&node1));
    }

    #[test]
    fn test_multi_dc_coordinator() {
        let mut coord = MultiDatacenterCoordinator::new();

        let dc1 = Datacenter::new(DatacenterId::new("us-east-1"), Region::new("us-east"));
        let dc2 = Datacenter::new(DatacenterId::new("us-west-2"), Region::new("us-west"));

        coord.add_datacenter(dc1);
        coord.add_datacenter(dc2);

        let node1 = NodeId(1);
        let node2 = NodeId(2);

        coord
            .register_node(node1, DatacenterId::new("us-east-1"))
            .unwrap();
        coord
            .register_node(node2, DatacenterId::new("us-west-2"))
            .unwrap();

        assert_eq!(coord.total_nodes(), 2);

        let dc = coord.get_node_datacenter(&node1).unwrap();
        assert_eq!(dc.id.as_str(), "us-east-1");
    }

    #[test]
    fn test_latency_tracking() {
        let mut coord = MultiDatacenterCoordinator::new();

        let dc1_id = DatacenterId::new("us-east-1");
        let dc2_id = DatacenterId::new("us-west-2");

        coord.record_latency(dc1_id.clone(), dc2_id.clone(), 50);

        assert_eq!(coord.get_latency(&dc1_id, &dc2_id), Some(50));
        // Should be symmetric
        assert_eq!(coord.get_latency(&dc2_id, &dc1_id), Some(50));
    }

    #[test]
    fn test_replication_policy_all() {
        let mut coord = MultiDatacenterCoordinator::new();

        coord.add_datacenter(Datacenter::new(DatacenterId::new("dc1"), Region::new("r1")));
        coord.add_datacenter(Datacenter::new(DatacenterId::new("dc2"), Region::new("r2")));

        let policy = ReplicationPolicy::AllDatacenters;
        let dcs = policy.select_datacenters(&coord, &DatacenterId::new("dc1"));

        assert_eq!(dcs.len(), 2);
    }

    #[test]
    fn test_replication_policy_regions() {
        let mut coord = MultiDatacenterCoordinator::new();

        coord.add_datacenter(Datacenter::new(
            DatacenterId::new("us-east-1"),
            Region::new("us-east"),
        ));
        coord.add_datacenter(Datacenter::new(
            DatacenterId::new("us-west-2"),
            Region::new("us-west"),
        ));
        coord.add_datacenter(Datacenter::new(
            DatacenterId::new("eu-west-1"),
            Region::new("eu-west"),
        ));

        let policy =
            ReplicationPolicy::Regions(vec![Region::new("us-east"), Region::new("us-west")]);
        let dcs = policy.select_datacenters(&coord, &DatacenterId::new("us-east-1"));

        assert_eq!(dcs.len(), 2);
    }

    #[test]
    fn test_latency_aware_selector() {
        let mut coord = MultiDatacenterCoordinator::new();

        let dc1_id = DatacenterId::new("dc1");
        let dc2_id = DatacenterId::new("dc2");

        coord.add_datacenter(Datacenter::new(dc1_id.clone(), Region::new("r1")));
        coord.add_datacenter(Datacenter::new(dc2_id.clone(), Region::new("r2")));

        let node1 = NodeId(1);
        let node2 = NodeId(2);

        coord.register_node(node1, dc1_id.clone()).unwrap();
        coord.register_node(node2, dc2_id.clone()).unwrap();

        coord.record_latency(dc1_id.clone(), dc2_id.clone(), 100);

        let coord = Arc::new(coord);
        let selector = LatencyAwareSelector::new(coord);

        let nodes = selector.select_read_nodes(&[node1, node2], &node1);

        // Should prefer local node (1) over remote node (2)
        assert_eq!(nodes[0], node1);
    }

    #[test]
    fn test_cross_dc_stats() {
        let mut stats = CrossDcStats::new();

        stats.record_local();
        stats.record_local();
        stats.record_cross_dc(50);
        stats.record_cross_dc(100);

        assert_eq!(stats.local_requests, 2);
        assert_eq!(stats.cross_dc_requests, 2);
        assert_eq!(stats.total_requests(), 4);
        assert_eq!(stats.cross_dc_percentage(), 50.0);
        assert_eq!(stats.avg_cross_dc_latency_ms, 75.0);
    }
}
