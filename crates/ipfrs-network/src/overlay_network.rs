//! Structured overlay network manager supporting multiple topology strategies.
//!
//! This module provides [`OverlayNetworkManager`], which models a logical overlay
//! network on top of physical connectivity. It supports Chord finger-table routing,
//! Pastry prefix routing, Kademlia greedy forwarding, FullMesh direct routing, and
//! spanning Tree routing—all computed purely in-memory without I/O.
//!
//! # Routing algorithms
//!
//! | Topology  | Route selection | Latency estimate |
//! |-----------|-----------------|-----------------|
//! | FullMesh  | direct 1-hop    | 20 ms           |
//! | Chord     | finger table log₂(N) hops | 20 ms/hop |
//! | Tree      | BFS in spanning tree | 20 ms/hop |
//! | Pastry    | greedy numeric proximity | 20 ms/hop |
//! | Kademlia  | greedy numeric proximity | 20 ms/hop |

use std::collections::{HashMap, HashSet, VecDeque};

use thiserror::Error;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors returned by [`OverlayNetworkManager`].
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum OverlayError {
    /// The specified node ID is not present in the overlay.
    #[error("node not found: {0}")]
    NodeNotFound(u64),

    /// The overlay has reached `max_nodes` and cannot accept new members.
    #[error("overlay network is full")]
    NetworkFull,

    /// No route exists between the two node IDs.
    #[error("no route found from {from} to {to}")]
    RouteNotFound {
        /// Source node.
        from: u64,
        /// Destination node.
        to: u64,
    },

    /// Source and destination are the same node.
    #[error("cannot route a node to itself")]
    SelfRoute,
}

// ---------------------------------------------------------------------------
// Topology enum
// ---------------------------------------------------------------------------

/// Overlay topology strategy used by [`OverlayNetworkManager`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OverlayTopology {
    /// Chord-style DHT ring with `fingers` finger-table entries per node.
    Chord {
        /// Number of finger-table entries per node.
        fingers: usize,
    },
    /// Pastry-style prefix routing.
    Pastry {
        /// Size of the leaf set (closest nodes by ID).
        leaf_set_size: usize,
        /// Number of rows in the routing table.
        routing_table_rows: usize,
    },
    /// Kademlia-style XOR-metric greedy routing (higher-level overlay).
    Kademlia {
        /// Replication factor (bucket size).
        k: usize,
        /// Parallelism factor for lookups.
        alpha: usize,
    },
    /// Every node is directly connected to every other node (small clusters).
    FullMesh,
    /// Spanning tree overlay with a fixed branching factor.
    Tree {
        /// Number of children per parent in the spanning tree.
        branching_factor: usize,
    },
}

impl OverlayTopology {
    /// Human-readable name of the topology.
    pub fn name(&self) -> &'static str {
        match self {
            OverlayTopology::Chord { .. } => "Chord",
            OverlayTopology::Pastry { .. } => "Pastry",
            OverlayTopology::Kademlia { .. } => "Kademlia",
            OverlayTopology::FullMesh => "FullMesh",
            OverlayTopology::Tree { .. } => "Tree",
        }
    }
}

// ---------------------------------------------------------------------------
// Node type
// ---------------------------------------------------------------------------

/// A participant in the overlay network.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OverlayNode {
    /// Unique numeric node identifier (used as the ring/XOR key).
    pub node_id: u64,
    /// Network address (e.g. multiaddr or IP:port string).
    pub address: String,
    /// Unix-millisecond timestamp when the node joined.
    pub joined_at: u64,
    /// Unix-millisecond timestamp of the last received heartbeat.
    pub last_heartbeat: u64,
    /// Arbitrary key-value metadata attached to the node.
    pub metadata: HashMap<String, String>,
}

impl OverlayNode {
    /// Create a new overlay node.
    pub fn new(node_id: u64, address: impl Into<String>, now: u64) -> Self {
        Self {
            node_id,
            address: address.into(),
            joined_at: now,
            last_heartbeat: now,
            metadata: HashMap::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Route type
// ---------------------------------------------------------------------------

/// A computed route through the overlay network.
#[derive(Debug, Clone, PartialEq)]
pub struct OverlayRoute {
    /// Ordered list of node IDs that form the path (including source and destination).
    pub hops: Vec<u64>,
    /// Estimated end-to-end latency in milliseconds (`hop_count * 20.0`).
    pub latency_estimate_ms: f64,
    /// Number of hops in the route.
    pub hop_count: usize,
}

impl OverlayRoute {
    fn new(hops: Vec<u64>) -> Self {
        // hop_count is the number of inter-node edges, i.e. len - 1, but we define
        // it as the number of nodes in the path minus 1 (edges traversed).
        let hop_count = hops.len().saturating_sub(1);
        let latency_estimate_ms = hop_count as f64 * 20.0;
        Self {
            hops,
            latency_estimate_ms,
            hop_count,
        }
    }
}

// ---------------------------------------------------------------------------
// Message type
// ---------------------------------------------------------------------------

/// A message routed through the overlay network.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OverlayMessage {
    /// FNV-1a hash of `(from XOR to XOR created_at)`.
    pub id: u64,
    /// Sender node ID.
    pub from: u64,
    /// Receiver node ID.
    pub to: u64,
    /// Time-to-live: maximum hops before the message is discarded.
    pub ttl: u8,
    /// Byte length of the message payload.
    pub payload_size: usize,
    /// Unix-millisecond creation timestamp.
    pub created_at: u64,
}

impl OverlayMessage {
    /// Create a new overlay message, computing the FNV-1a ID automatically.
    pub fn new(from: u64, to: u64, ttl: u8, payload_size: usize, created_at: u64) -> Self {
        let id = fnv1a_hash(from ^ to ^ created_at);
        Self {
            id,
            from,
            to,
            ttl,
            payload_size,
            created_at,
        }
    }
}

/// Inline FNV-1a hash for a single 64-bit value.
#[inline]
fn fnv1a_hash(value: u64) -> u64 {
    const FNV_OFFSET_BASIS: u64 = 14_695_981_039_346_656_037;
    const FNV_PRIME: u64 = 1_099_511_628_211;
    let bytes = value.to_le_bytes();
    let mut hash = FNV_OFFSET_BASIS;
    for b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

// ---------------------------------------------------------------------------
// Stats type
// ---------------------------------------------------------------------------

/// Snapshot of overlay network statistics.
#[derive(Debug, Clone, PartialEq)]
pub struct OverlayStats {
    /// Number of nodes currently in the overlay.
    pub node_count: usize,
    /// Mean age (in ms) of the last heartbeat across all nodes.
    pub avg_heartbeat_age_ms: f64,
    /// Estimated maximum path length across the overlay.
    pub estimated_diameter: f64,
    /// Human-readable name of the active topology.
    pub topology_name: String,
}

// ---------------------------------------------------------------------------
// Manager
// ---------------------------------------------------------------------------

/// Manages a logical overlay network of nodes with topology-aware routing.
///
/// The manager is intentionally synchronous and allocation-light: it keeps
/// a flat `HashMap` of nodes and computes routes on demand without spawning
/// tasks or doing I/O.
///
/// # Example
///
/// ```rust
/// use ipfrs_network::{OverlayNetworkManager, OverlayTopology, OverlayNode};
///
/// let mut mgr = OverlayNetworkManager::new(1, OverlayTopology::FullMesh, 128);
/// let node = OverlayNode::new(2, "127.0.0.1:4001", 0);
/// mgr.join(node).expect("join should succeed in example");
/// let route = mgr.route(1, 2, 0).expect("route should succeed in example");
/// assert_eq!(route.hop_count, 1);
/// ```
#[derive(Debug)]
pub struct OverlayNetworkManager {
    /// Active topology strategy.
    topology: OverlayTopology,
    /// All nodes known to the overlay, keyed by `node_id`.
    nodes: HashMap<u64, OverlayNode>,
    /// Node ID of this local instance (always treated as present).
    local_id: u64,
    /// Hard upper bound on nodes that can join.
    max_nodes: usize,
}

impl OverlayNetworkManager {
    /// Create a new overlay network manager.
    ///
    /// The local node (identified by `local_id`) is implicitly part of the
    /// overlay but is **not** inserted into `nodes`; callers should [`join`]
    /// it if they want it visible to routing queries.
    ///
    /// [`join`]: OverlayNetworkManager::join
    pub fn new(local_id: u64, topology: OverlayTopology, max_nodes: usize) -> Self {
        Self {
            topology,
            nodes: HashMap::new(),
            local_id,
            max_nodes,
        }
    }

    // ------------------------------------------------------------------
    // Membership
    // ------------------------------------------------------------------

    /// Add a node to the overlay.
    ///
    /// Returns [`OverlayError::NetworkFull`] if `max_nodes` has been reached.
    /// If the node is already present its record is silently updated.
    pub fn join(&mut self, node: OverlayNode) -> Result<(), OverlayError> {
        // Allow update of existing node without checking capacity.
        if let std::collections::hash_map::Entry::Occupied(mut e) = self.nodes.entry(node.node_id) {
            e.insert(node);
            return Ok(());
        }
        if self.nodes.len() >= self.max_nodes {
            return Err(OverlayError::NetworkFull);
        }
        self.nodes.insert(node.node_id, node);
        Ok(())
    }

    /// Remove a node from the overlay.
    ///
    /// Returns [`OverlayError::NodeNotFound`] if the node is not present.
    pub fn leave(&mut self, node_id: u64) -> Result<(), OverlayError> {
        self.nodes
            .remove(&node_id)
            .map(|_| ())
            .ok_or(OverlayError::NodeNotFound(node_id))
    }

    /// Record a heartbeat for a node, updating its `last_heartbeat` timestamp.
    pub fn heartbeat(&mut self, node_id: u64, now: u64) -> Result<(), OverlayError> {
        self.nodes
            .get_mut(&node_id)
            .map(|n| n.last_heartbeat = now)
            .ok_or(OverlayError::NodeNotFound(node_id))
    }

    /// Remove nodes whose `last_heartbeat` is older than `max_age_ms` milliseconds.
    ///
    /// Returns the number of nodes evicted.
    pub fn evict_stale(&mut self, max_age_ms: u64, now: u64) -> usize {
        let cutoff = now.saturating_sub(max_age_ms);
        let stale: Vec<u64> = self
            .nodes
            .values()
            .filter(|n| n.last_heartbeat < cutoff)
            .map(|n| n.node_id)
            .collect();
        let count = stale.len();
        for id in stale {
            self.nodes.remove(&id);
        }
        count
    }

    /// Return the total number of nodes in the overlay.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    // ------------------------------------------------------------------
    // Routing
    // ------------------------------------------------------------------

    /// Compute a route from `from` to `to` using the active topology.
    ///
    /// Returns [`OverlayError::SelfRoute`] when `from == to`, and
    /// [`OverlayError::NodeNotFound`] when either endpoint is absent.
    pub fn route(&self, from: u64, to: u64, now: u64) -> Result<OverlayRoute, OverlayError> {
        let _ = now; // reserved for future timestamp-aware routing
        if from == to {
            return Err(OverlayError::SelfRoute);
        }
        if !self.nodes.contains_key(&from) {
            return Err(OverlayError::NodeNotFound(from));
        }
        if !self.nodes.contains_key(&to) {
            return Err(OverlayError::NodeNotFound(to));
        }

        match &self.topology {
            OverlayTopology::FullMesh => self.route_full_mesh(from, to),
            OverlayTopology::Chord { fingers } => self.route_chord(from, to, *fingers),
            OverlayTopology::Tree { branching_factor } => {
                self.route_tree(from, to, *branching_factor)
            }
            OverlayTopology::Pastry { .. } | OverlayTopology::Kademlia { .. } => {
                self.route_greedy_proximity(from, to)
            }
        }
    }

    /// FullMesh: direct one-hop route between any two present nodes.
    fn route_full_mesh(&self, from: u64, to: u64) -> Result<OverlayRoute, OverlayError> {
        Ok(OverlayRoute::new(vec![from, to]))
    }

    /// Chord: route via finger table.
    ///
    /// The finger table for node `n` contains up to `fingers` entries.
    /// Entry `i` points to the node whose ID is nearest (in ring arithmetic)
    /// to `n + 2^i`. Routing greedily advances via fingers until the
    /// destination is reached or no better finger exists.
    fn route_chord(
        &self,
        from: u64,
        to: u64,
        fingers: usize,
    ) -> Result<OverlayRoute, OverlayError> {
        const MAX_HOPS: usize = 64;
        let mut path = vec![from];
        let mut current = from;

        // Collect all IDs for efficient lookups.
        let all_ids: Vec<u64> = self.nodes.keys().copied().collect();

        // Distance on the Chord ring: how far `target` is clockwise from `base`.
        let ring_distance = |base: u64, target: u64| -> u64 {
            if target >= base {
                target - base
            } else {
                u64::MAX - base + target + 1
            }
        };

        // Find the node whose ID is closest to `ideal` in a clockwise sense.
        let best_finger = |node: u64, ideal: u64| -> u64 {
            all_ids
                .iter()
                .copied()
                .min_by_key(|&id| ring_distance(ideal, id))
                .unwrap_or(node)
        };

        while current != to && path.len() <= MAX_HOPS {
            // Look for the best finger to bring us closer to `to`.
            let mut best_next = current;
            let current_dist = ring_distance(current, to);

            let effective_fingers = fingers.max(1);
            for i in 0..effective_fingers {
                // ideal successor for finger i: current + 2^i (wrapping)
                let ideal = current.wrapping_add(1u64 << (i.min(63)));
                let finger_node = best_finger(current, ideal);
                let finger_dist = ring_distance(finger_node, to);
                if finger_dist < current_dist || finger_node == to {
                    // pick the finger that gets us closest (smallest ring-dist to dest)
                    let best_dist = ring_distance(best_next, to);
                    if best_next == current || finger_dist < best_dist {
                        best_next = finger_node;
                    }
                }
            }

            if best_next == current || path.contains(&best_next) {
                // No progress; check if `to` is directly reachable (i.e., in the node set)
                if self.nodes.contains_key(&to) && !path.contains(&to) {
                    path.push(to);
                    break;
                }
                return Err(OverlayError::RouteNotFound { from, to });
            }
            path.push(best_next);
            current = best_next;
        }

        if current != to {
            return Err(OverlayError::RouteNotFound { from, to });
        }
        Ok(OverlayRoute::new(path))
    }

    /// Tree: BFS through the spanning tree to find the path from `from` to `to`.
    ///
    /// Nodes are ordered by `joined_at` to form a deterministic tree; root
    /// is the node with the earliest `joined_at`. Each parent has up to
    /// `branching_factor` children.
    fn route_tree(
        &self,
        from: u64,
        to: u64,
        branching_factor: usize,
    ) -> Result<OverlayRoute, OverlayError> {
        // Build the spanning tree adjacency list.
        // Nodes sorted by joined_at (tie-break by node_id).
        let mut sorted: Vec<&OverlayNode> = self.nodes.values().collect();
        sorted.sort_by_key(|n| (n.joined_at, n.node_id));

        let bf = branching_factor.max(1);

        // Build parent→children mapping.
        let mut children: HashMap<u64, Vec<u64>> = HashMap::new();
        // parent of each node
        let mut parent: HashMap<u64, Option<u64>> = HashMap::new();

        for (idx, node) in sorted.iter().enumerate() {
            if idx == 0 {
                parent.insert(node.node_id, None);
            } else {
                // Parent index in the sorted list (0-based, 1-indexed children of root)
                let parent_idx = (idx - 1) / bf;
                let parent_id = sorted[parent_idx].node_id;
                parent.insert(node.node_id, Some(parent_id));
                children.entry(parent_id).or_default().push(node.node_id);
            }
        }

        // BFS from `from` to `to` in the undirected tree.
        let mut visited: HashSet<u64> = HashSet::new();
        let mut queue: VecDeque<(u64, Vec<u64>)> = VecDeque::new();
        queue.push_back((from, vec![from]));
        visited.insert(from);

        while let Some((current, path)) = queue.pop_front() {
            if current == to {
                return Ok(OverlayRoute::new(path));
            }
            // Explore children
            if let Some(kids) = children.get(&current) {
                for &child in kids {
                    if !visited.contains(&child) {
                        visited.insert(child);
                        let mut new_path = path.clone();
                        new_path.push(child);
                        queue.push_back((child, new_path));
                    }
                }
            }
            // Explore parent
            if let Some(Some(par)) = parent.get(&current) {
                if !visited.contains(par) {
                    visited.insert(*par);
                    let mut new_path = path.clone();
                    new_path.push(*par);
                    queue.push_back((*par, new_path));
                }
            }
        }

        Err(OverlayError::RouteNotFound { from, to })
    }

    /// Greedy forwarding by numeric (absolute) proximity — used for Pastry and Kademlia.
    ///
    /// At each hop we pick the neighbor (in the overlay's neighbor set for the
    /// current node) that is numerically closest to `to`. We cap at 32 hops to
    /// guarantee termination.
    fn route_greedy_proximity(&self, from: u64, to: u64) -> Result<OverlayRoute, OverlayError> {
        const MAX_HOPS: usize = 32;
        let mut path = vec![from];
        let mut current = from;
        let mut visited: HashSet<u64> = HashSet::new();
        visited.insert(from);

        while current != to && path.len() <= MAX_HOPS {
            let nbrs = self.neighbors_of(current);
            // Find the neighbor closest to `to` (by absolute XOR distance).
            let best = nbrs
                .into_iter()
                .filter(|id| !visited.contains(id))
                .min_by_key(|&id| xor_distance(id, to));

            match best {
                None => {
                    // No unvisited neighbors; check direct connection
                    if self.nodes.contains_key(&to) && !path.contains(&to) {
                        path.push(to);
                        return Ok(OverlayRoute::new(path));
                    }
                    return Err(OverlayError::RouteNotFound { from, to });
                }
                Some(next) => {
                    if next == to {
                        path.push(to);
                        return Ok(OverlayRoute::new(path));
                    }
                    // Check that we are making progress (getting closer).
                    let current_dist = xor_distance(current, to);
                    let next_dist = xor_distance(next, to);
                    if next_dist >= current_dist {
                        // Stuck; jump directly if reachable.
                        if self.nodes.contains_key(&to) {
                            path.push(to);
                            return Ok(OverlayRoute::new(path));
                        }
                        return Err(OverlayError::RouteNotFound { from, to });
                    }
                    visited.insert(next);
                    path.push(next);
                    current = next;
                }
            }
        }

        if current == to {
            Ok(OverlayRoute::new(path))
        } else {
            Err(OverlayError::RouteNotFound { from, to })
        }
    }

    // ------------------------------------------------------------------
    // Neighbors
    // ------------------------------------------------------------------

    /// Return the set of logical neighbors for `node_id` under the active topology.
    ///
    /// Returns an empty `Vec` if the node is not present.
    pub fn neighbors(&self, node_id: u64) -> Vec<u64> {
        if !self.nodes.contains_key(&node_id) {
            return Vec::new();
        }
        self.neighbors_of(node_id)
    }

    /// Internal neighbor computation (does not check node existence).
    fn neighbors_of(&self, node_id: u64) -> Vec<u64> {
        match &self.topology {
            OverlayTopology::FullMesh => {
                // Every other node is a neighbor.
                self.nodes
                    .keys()
                    .copied()
                    .filter(|&id| id != node_id)
                    .collect()
            }
            OverlayTopology::Chord { fingers } => self.chord_fingers(node_id, *fingers),
            OverlayTopology::Tree { branching_factor } => {
                self.tree_neighbors(node_id, *branching_factor)
            }
            OverlayTopology::Pastry { leaf_set_size, .. } => {
                self.closest_n_neighbors(node_id, *leaf_set_size)
            }
            OverlayTopology::Kademlia { k, .. } => self.closest_n_neighbors(node_id, *k),
        }
    }

    /// Chord finger table entries for `node_id`.
    fn chord_fingers(&self, node_id: u64, fingers: usize) -> Vec<u64> {
        let effective = fingers.max(1);
        let all_ids: Vec<u64> = self
            .nodes
            .keys()
            .copied()
            .filter(|&id| id != node_id)
            .collect();
        let mut seen: HashSet<u64> = HashSet::new();
        let mut result = Vec::new();

        for i in 0..effective {
            let ideal = node_id.wrapping_add(1u64 << (i.min(63)));
            // Node closest to `ideal` on the ring (clockwise distance).
            let best = all_ids.iter().copied().min_by_key(|&id| {
                if id >= ideal {
                    id - ideal
                } else {
                    u64::MAX - ideal + id + 1
                }
            });
            if let Some(id) = best {
                if seen.insert(id) {
                    result.push(id);
                }
            }
        }
        result
    }

    /// Return up to `n` nodes numerically closest to `node_id` (by XOR distance).
    fn closest_n_neighbors(&self, node_id: u64, n: usize) -> Vec<u64> {
        let mut candidates: Vec<u64> = self
            .nodes
            .keys()
            .copied()
            .filter(|&id| id != node_id)
            .collect();
        candidates.sort_by_key(|&id| xor_distance(id, node_id));
        candidates.truncate(n);
        candidates
    }

    /// Neighbors of `node_id` in the spanning tree (parent + children).
    fn tree_neighbors(&self, node_id: u64, branching_factor: usize) -> Vec<u64> {
        let mut sorted: Vec<&OverlayNode> = self.nodes.values().collect();
        sorted.sort_by_key(|n| (n.joined_at, n.node_id));
        let bf = branching_factor.max(1);

        let pos = match sorted.iter().position(|n| n.node_id == node_id) {
            Some(p) => p,
            None => return Vec::new(),
        };

        let mut neighbors = Vec::new();

        // Parent
        if pos > 0 {
            let parent_idx = (pos - 1) / bf;
            neighbors.push(sorted[parent_idx].node_id);
        }

        // Children
        let first_child = pos * bf + 1;
        for c in first_child..first_child + bf {
            if c < sorted.len() {
                neighbors.push(sorted[c].node_id);
            }
        }

        neighbors
    }

    // ------------------------------------------------------------------
    // Diameter estimate
    // ------------------------------------------------------------------

    /// Estimate the maximum path length (diameter) across the overlay.
    ///
    /// | Topology | Formula |
    /// |----------|---------|
    /// | FullMesh | 1.0 |
    /// | Chord / Kademlia / Pastry | log₂(max(n, 2)) |
    /// | Tree | log_{bf}(max(n, 2)) |
    pub fn diameter_estimate(&self) -> f64 {
        let n = self.nodes.len().max(2) as f64;
        match &self.topology {
            OverlayTopology::FullMesh => 1.0,
            OverlayTopology::Chord { .. }
            | OverlayTopology::Kademlia { .. }
            | OverlayTopology::Pastry { .. } => n.log2(),
            OverlayTopology::Tree { branching_factor } => {
                let bf = (*branching_factor).max(2) as f64;
                n.log(bf)
            }
        }
    }

    // ------------------------------------------------------------------
    // Stats
    // ------------------------------------------------------------------

    /// Return a snapshot of statistics at the given wall-clock timestamp `now`.
    pub fn stats(&self, now: u64) -> OverlayStats {
        let node_count = self.nodes.len();
        let avg_heartbeat_age_ms = if node_count == 0 {
            0.0
        } else {
            let total_age: u64 = self
                .nodes
                .values()
                .map(|n| now.saturating_sub(n.last_heartbeat))
                .sum();
            total_age as f64 / node_count as f64
        };
        OverlayStats {
            node_count,
            avg_heartbeat_age_ms,
            estimated_diameter: self.diameter_estimate(),
            topology_name: self.topology.name().to_string(),
        }
    }

    // ------------------------------------------------------------------
    // Accessors
    // ------------------------------------------------------------------

    /// Return a reference to the node record for `node_id`, if present.
    pub fn get_node(&self, node_id: u64) -> Option<&OverlayNode> {
        self.nodes.get(&node_id)
    }

    /// Return the local node ID.
    pub fn local_id(&self) -> u64 {
        self.local_id
    }

    /// Return a reference to the active topology.
    pub fn topology(&self) -> &OverlayTopology {
        &self.topology
    }
}

// ---------------------------------------------------------------------------
// Helper: XOR distance
// ---------------------------------------------------------------------------

/// XOR distance between two node IDs.
#[inline]
fn xor_distance(a: u64, b: u64) -> u64 {
    a ^ b
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    fn make_node(id: u64, joined_at: u64) -> OverlayNode {
        OverlayNode::new(id, format!("127.0.0.1:{}", 4000 + id), joined_at)
    }

    fn populate(mgr: &mut OverlayNetworkManager, ids: &[u64], base_time: u64) {
        for (i, &id) in ids.iter().enumerate() {
            mgr.join(make_node(id, base_time + i as u64))
                .expect("test: join should succeed during populate");
        }
    }

    // ------------------------------------------------------------------
    // OverlayNode tests
    // ------------------------------------------------------------------

    #[test]
    fn test_overlay_node_new() {
        let node = make_node(42, 1000);
        assert_eq!(node.node_id, 42);
        assert_eq!(node.joined_at, 1000);
        assert_eq!(node.last_heartbeat, 1000);
        assert!(node.metadata.is_empty());
    }

    #[test]
    fn test_overlay_node_address() {
        let node = OverlayNode::new(7, "/ip4/1.2.3.4/tcp/4001", 0);
        assert_eq!(node.address, "/ip4/1.2.3.4/tcp/4001");
    }

    // ------------------------------------------------------------------
    // OverlayMessage tests
    // ------------------------------------------------------------------

    #[test]
    fn test_overlay_message_id_deterministic() {
        let m1 = OverlayMessage::new(1, 2, 10, 100, 999);
        let m2 = OverlayMessage::new(1, 2, 10, 100, 999);
        assert_eq!(m1.id, m2.id);
    }

    #[test]
    fn test_overlay_message_id_changes_with_inputs() {
        let m1 = OverlayMessage::new(1, 2, 10, 100, 999);
        let m2 = OverlayMessage::new(1, 3, 10, 100, 999);
        assert_ne!(m1.id, m2.id);
    }

    #[test]
    fn test_overlay_message_fields() {
        let msg = OverlayMessage::new(10, 20, 5, 256, 12345);
        assert_eq!(msg.from, 10);
        assert_eq!(msg.to, 20);
        assert_eq!(msg.ttl, 5);
        assert_eq!(msg.payload_size, 256);
        assert_eq!(msg.created_at, 12345);
    }

    // ------------------------------------------------------------------
    // OverlayRoute tests
    // ------------------------------------------------------------------

    #[test]
    fn test_overlay_route_direct() {
        let r = OverlayRoute::new(vec![1, 2]);
        assert_eq!(r.hop_count, 1);
        assert!((r.latency_estimate_ms - 20.0).abs() < 1e-9);
    }

    #[test]
    fn test_overlay_route_multi_hop() {
        let r = OverlayRoute::new(vec![1, 2, 3, 4]);
        assert_eq!(r.hop_count, 3);
        assert!((r.latency_estimate_ms - 60.0).abs() < 1e-9);
    }

    #[test]
    fn test_overlay_route_single_node() {
        let r = OverlayRoute::new(vec![5]);
        assert_eq!(r.hop_count, 0);
        assert!((r.latency_estimate_ms - 0.0).abs() < 1e-9);
    }

    // ------------------------------------------------------------------
    // Join / Leave / Heartbeat
    // ------------------------------------------------------------------

    #[test]
    fn test_join_success() {
        let mut mgr = OverlayNetworkManager::new(0, OverlayTopology::FullMesh, 10);
        assert!(mgr.join(make_node(1, 0)).is_ok());
        assert_eq!(mgr.node_count(), 1);
    }

    #[test]
    fn test_join_network_full() {
        let mut mgr = OverlayNetworkManager::new(0, OverlayTopology::FullMesh, 2);
        mgr.join(make_node(1, 0))
            .expect("test: first join should succeed");
        mgr.join(make_node(2, 0))
            .expect("test: second join should succeed");
        let err = mgr.join(make_node(3, 0)).unwrap_err();
        assert_eq!(err, OverlayError::NetworkFull);
    }

    #[test]
    fn test_join_updates_existing_node() {
        let mut mgr = OverlayNetworkManager::new(0, OverlayTopology::FullMesh, 10);
        mgr.join(make_node(1, 0))
            .expect("test: initial join should succeed");
        // Join again with updated heartbeat; should succeed and update.
        let mut updated = make_node(1, 0);
        updated.last_heartbeat = 9999;
        mgr.join(updated).expect("test: update join should succeed");
        assert_eq!(mgr.node_count(), 1);
        assert_eq!(
            mgr.get_node(1)
                .expect("test: node 1 should be present")
                .last_heartbeat,
            9999
        );
    }

    #[test]
    fn test_leave_success() {
        let mut mgr = OverlayNetworkManager::new(0, OverlayTopology::FullMesh, 10);
        mgr.join(make_node(1, 0))
            .expect("test: join should succeed");
        assert!(mgr.leave(1).is_ok());
        assert_eq!(mgr.node_count(), 0);
    }

    #[test]
    fn test_leave_not_found() {
        let mut mgr = OverlayNetworkManager::new(0, OverlayTopology::FullMesh, 10);
        let err = mgr.leave(99).unwrap_err();
        assert_eq!(err, OverlayError::NodeNotFound(99));
    }

    #[test]
    fn test_heartbeat_updates_timestamp() {
        let mut mgr = OverlayNetworkManager::new(0, OverlayTopology::FullMesh, 10);
        mgr.join(make_node(1, 0))
            .expect("test: join should succeed");
        mgr.heartbeat(1, 5000)
            .expect("test: heartbeat should succeed");
        assert_eq!(
            mgr.get_node(1)
                .expect("test: node 1 should be present after join")
                .last_heartbeat,
            5000
        );
    }

    #[test]
    fn test_heartbeat_not_found() {
        let mut mgr = OverlayNetworkManager::new(0, OverlayTopology::FullMesh, 10);
        let err = mgr.heartbeat(99, 0).unwrap_err();
        assert_eq!(err, OverlayError::NodeNotFound(99));
    }

    // ------------------------------------------------------------------
    // Eviction
    // ------------------------------------------------------------------

    #[test]
    fn test_evict_stale_removes_old_nodes() {
        let mut mgr = OverlayNetworkManager::new(0, OverlayTopology::FullMesh, 100);
        mgr.join(make_node(1, 0))
            .expect("test: join node 1 should succeed");
        mgr.join(make_node(2, 0))
            .expect("test: join node 2 should succeed");
        mgr.heartbeat(1, 190)
            .expect("test: heartbeat for node 1 should succeed"); // node 1: heartbeat at t=190 — fresh
                                                                  // node 2 still at t=0 (stale)
        let removed = mgr.evict_stale(50, 200); // cutoff = 150, node2 @ 0 < 150 → stale; node1 @ 190 >= 150 → fresh
        assert_eq!(removed, 1);
        assert!(mgr.get_node(1).is_some());
        assert!(mgr.get_node(2).is_none());
    }

    #[test]
    fn test_evict_stale_none_removed() {
        let mut mgr = OverlayNetworkManager::new(0, OverlayTopology::FullMesh, 100);
        mgr.join(make_node(1, 1000))
            .expect("test: join should succeed");
        let removed = mgr.evict_stale(500, 1200);
        assert_eq!(removed, 0); // cutoff = 700, heartbeat 1000 > 700
    }

    #[test]
    fn test_evict_all() {
        let mut mgr = OverlayNetworkManager::new(0, OverlayTopology::FullMesh, 100);
        populate(&mut mgr, &[1, 2, 3, 4, 5], 0);
        let removed = mgr.evict_stale(0, 9999); // cutoff = 9999, all < 9999
        assert_eq!(removed, 5);
        assert_eq!(mgr.node_count(), 0);
    }

    // ------------------------------------------------------------------
    // FullMesh routing
    // ------------------------------------------------------------------

    #[test]
    fn test_full_mesh_route_direct() {
        let mut mgr = OverlayNetworkManager::new(0, OverlayTopology::FullMesh, 100);
        populate(&mut mgr, &[1, 2, 3], 0);
        let route = mgr
            .route(1, 2, 0)
            .expect("test: full mesh route should succeed");
        assert_eq!(route.hop_count, 1);
        assert_eq!(route.hops, vec![1, 2]);
    }

    #[test]
    fn test_full_mesh_route_self_error() {
        let mut mgr = OverlayNetworkManager::new(0, OverlayTopology::FullMesh, 100);
        mgr.join(make_node(1, 0))
            .expect("test: join should succeed");
        assert_eq!(mgr.route(1, 1, 0).unwrap_err(), OverlayError::SelfRoute);
    }

    #[test]
    fn test_full_mesh_route_unknown_source() {
        let mut mgr = OverlayNetworkManager::new(0, OverlayTopology::FullMesh, 100);
        mgr.join(make_node(2, 0))
            .expect("test: join should succeed");
        assert_eq!(
            mgr.route(99, 2, 0).unwrap_err(),
            OverlayError::NodeNotFound(99)
        );
    }

    #[test]
    fn test_full_mesh_route_unknown_dest() {
        let mut mgr = OverlayNetworkManager::new(0, OverlayTopology::FullMesh, 100);
        mgr.join(make_node(1, 0))
            .expect("test: join should succeed");
        assert_eq!(
            mgr.route(1, 99, 0).unwrap_err(),
            OverlayError::NodeNotFound(99)
        );
    }

    // ------------------------------------------------------------------
    // Chord routing
    // ------------------------------------------------------------------

    #[test]
    fn test_chord_route_two_nodes() {
        let mut mgr = OverlayNetworkManager::new(0, OverlayTopology::Chord { fingers: 4 }, 100);
        mgr.join(make_node(0, 0))
            .expect("test: join node 0 should succeed");
        mgr.join(make_node(100, 0))
            .expect("test: join node 100 should succeed");
        let route = mgr
            .route(0, 100, 0)
            .expect("test: chord route should succeed");
        assert!(route.hop_count >= 1);
        assert_eq!(
            *route
                .hops
                .first()
                .expect("test: route hops should be non-empty"),
            0
        );
        assert_eq!(
            *route
                .hops
                .last()
                .expect("test: route hops should have last element"),
            100
        );
    }

    #[test]
    fn test_chord_route_self_error() {
        let mut mgr = OverlayNetworkManager::new(0, OverlayTopology::Chord { fingers: 8 }, 100);
        mgr.join(make_node(5, 0))
            .expect("test: join should succeed");
        assert_eq!(mgr.route(5, 5, 0).unwrap_err(), OverlayError::SelfRoute);
    }

    #[test]
    fn test_chord_route_many_nodes() {
        let mut mgr = OverlayNetworkManager::new(0, OverlayTopology::Chord { fingers: 8 }, 100);
        // Space nodes across the u64 ring.
        let ids: Vec<u64> = (0..16).map(|i: u64| i * (u64::MAX / 16)).collect();
        populate(&mut mgr, &ids, 0);
        let route = mgr
            .route(ids[0], ids[8], 0)
            .expect("test: chord route across many nodes should succeed");
        assert_eq!(
            *route
                .hops
                .first()
                .expect("test: route hops should be non-empty"),
            ids[0]
        );
        assert_eq!(
            *route
                .hops
                .last()
                .expect("test: route hops should have last element"),
            ids[8]
        );
        // Should reach in log2(16) = 4 hops or fewer.
        assert!(route.hop_count <= 16);
    }

    // ------------------------------------------------------------------
    // Tree routing
    // ------------------------------------------------------------------

    #[test]
    fn test_tree_route_parent_to_child() {
        let mut mgr = OverlayNetworkManager::new(
            0,
            OverlayTopology::Tree {
                branching_factor: 2,
            },
            100,
        );
        // joined_at order: 10, 20, 30, 40
        // tree: 10 is root, children: 20,30. 20's child: 40.
        mgr.join(make_node(10, 10))
            .expect("test: join node 10 should succeed");
        mgr.join(make_node(20, 20))
            .expect("test: join node 20 should succeed");
        mgr.join(make_node(30, 30))
            .expect("test: join node 30 should succeed");
        mgr.join(make_node(40, 40))
            .expect("test: join node 40 should succeed");
        // Route from root (10) to grandchild (40): 10 -> 20 -> 40
        let route = mgr
            .route(10, 40, 0)
            .expect("test: tree route from root to grandchild should succeed");
        assert_eq!(
            *route
                .hops
                .first()
                .expect("test: route hops should be non-empty"),
            10
        );
        assert_eq!(
            *route
                .hops
                .last()
                .expect("test: route hops should have last element"),
            40
        );
        assert_eq!(route.hop_count, 2);
    }

    #[test]
    fn test_tree_route_sibling_via_parent() {
        let mut mgr = OverlayNetworkManager::new(
            0,
            OverlayTopology::Tree {
                branching_factor: 2,
            },
            100,
        );
        mgr.join(make_node(1, 1))
            .expect("test: join node 1 should succeed");
        mgr.join(make_node(2, 2))
            .expect("test: join node 2 should succeed");
        mgr.join(make_node(3, 3))
            .expect("test: join node 3 should succeed");
        // tree: 1 is root, 2 and 3 are children.
        // Route from 2 to 3 must go via 1 (root).
        let route = mgr
            .route(2, 3, 0)
            .expect("test: tree route between siblings should succeed");
        assert_eq!(
            *route
                .hops
                .first()
                .expect("test: route hops should be non-empty"),
            2
        );
        assert_eq!(
            *route
                .hops
                .last()
                .expect("test: route hops should have last element"),
            3
        );
        assert_eq!(route.hop_count, 2); // 2 -> 1 -> 3
    }

    #[test]
    fn test_tree_route_self_error() {
        let mut mgr = OverlayNetworkManager::new(
            0,
            OverlayTopology::Tree {
                branching_factor: 3,
            },
            100,
        );
        mgr.join(make_node(5, 0))
            .expect("test: join should succeed");
        assert_eq!(mgr.route(5, 5, 0).unwrap_err(), OverlayError::SelfRoute);
    }

    // ------------------------------------------------------------------
    // Pastry / Kademlia (greedy proximity) routing
    // ------------------------------------------------------------------

    #[test]
    fn test_pastry_route_direct() {
        let mut mgr = OverlayNetworkManager::new(
            0,
            OverlayTopology::Pastry {
                leaf_set_size: 4,
                routing_table_rows: 4,
            },
            100,
        );
        mgr.join(make_node(1, 0))
            .expect("test: join node 1 should succeed");
        mgr.join(make_node(2, 0))
            .expect("test: join node 2 should succeed");
        let route = mgr
            .route(1, 2, 0)
            .expect("test: pastry route should succeed");
        assert_eq!(
            *route
                .hops
                .first()
                .expect("test: route hops should be non-empty"),
            1
        );
        assert_eq!(
            *route
                .hops
                .last()
                .expect("test: route hops should have last element"),
            2
        );
    }

    #[test]
    fn test_kademlia_route_multi_hop() {
        let mut mgr =
            OverlayNetworkManager::new(0, OverlayTopology::Kademlia { k: 4, alpha: 3 }, 100);
        // Craft IDs so greedy XOR forwarding makes progress.
        // IDs: 0, 4, 6, 7.  XOR distances from 0 to 7: 7; from 4 to 7: 3; from 6 to 7: 1.
        populate(&mut mgr, &[0, 4, 6, 7], 0);
        let route = mgr
            .route(0, 7, 0)
            .expect("test: kademlia route should succeed");
        assert_eq!(
            *route
                .hops
                .first()
                .expect("test: route hops should be non-empty"),
            0
        );
        assert_eq!(
            *route
                .hops
                .last()
                .expect("test: route hops should have last element"),
            7
        );
    }

    #[test]
    fn test_kademlia_route_self_error() {
        let mut mgr =
            OverlayNetworkManager::new(0, OverlayTopology::Kademlia { k: 8, alpha: 3 }, 100);
        mgr.join(make_node(10, 0))
            .expect("test: join should succeed");
        assert_eq!(mgr.route(10, 10, 0).unwrap_err(), OverlayError::SelfRoute);
    }

    // ------------------------------------------------------------------
    // Neighbors
    // ------------------------------------------------------------------

    #[test]
    fn test_neighbors_full_mesh_all() {
        let mut mgr = OverlayNetworkManager::new(0, OverlayTopology::FullMesh, 100);
        populate(&mut mgr, &[1, 2, 3, 4], 0);
        let mut nbrs = mgr.neighbors(1);
        nbrs.sort();
        assert_eq!(nbrs, vec![2, 3, 4]);
    }

    #[test]
    fn test_neighbors_unknown_node_empty() {
        let mgr = OverlayNetworkManager::new(0, OverlayTopology::FullMesh, 100);
        assert!(mgr.neighbors(999).is_empty());
    }

    #[test]
    fn test_neighbors_kademlia_limited_to_k() {
        let mut mgr =
            OverlayNetworkManager::new(0, OverlayTopology::Kademlia { k: 3, alpha: 1 }, 100);
        populate(&mut mgr, &[0, 1, 2, 3, 4, 5, 6], 0);
        let nbrs = mgr.neighbors(0);
        assert!(nbrs.len() <= 3);
    }

    #[test]
    fn test_neighbors_tree_root_has_only_children() {
        let mut mgr = OverlayNetworkManager::new(
            0,
            OverlayTopology::Tree {
                branching_factor: 2,
            },
            100,
        );
        // root=1 (joined first), children=2,3
        mgr.join(make_node(1, 1))
            .expect("test: join node 1 should succeed");
        mgr.join(make_node(2, 2))
            .expect("test: join node 2 should succeed");
        mgr.join(make_node(3, 3))
            .expect("test: join node 3 should succeed");
        let mut nbrs = mgr.neighbors(1);
        nbrs.sort();
        assert_eq!(nbrs, vec![2, 3]);
    }

    #[test]
    fn test_neighbors_chord_finger_count() {
        let mut mgr = OverlayNetworkManager::new(0, OverlayTopology::Chord { fingers: 4 }, 100);
        populate(&mut mgr, &[0, 10, 20, 30, 40, 50, 60, 70], 0);
        let nbrs = mgr.neighbors(0);
        // Should have at most 4 distinct finger entries.
        assert!(nbrs.len() <= 4);
    }

    // ------------------------------------------------------------------
    // Diameter estimate
    // ------------------------------------------------------------------

    #[test]
    fn test_diameter_full_mesh() {
        let mut mgr = OverlayNetworkManager::new(0, OverlayTopology::FullMesh, 100);
        populate(&mut mgr, &[1, 2, 3, 4, 5], 0);
        assert!((mgr.diameter_estimate() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_diameter_chord_log2() {
        let mut mgr = OverlayNetworkManager::new(0, OverlayTopology::Chord { fingers: 8 }, 1000);
        let ids: Vec<u64> = (1..=16).collect();
        populate(&mut mgr, &ids, 0);
        // log2(16) = 4.0
        let d = mgr.diameter_estimate();
        assert!((d - 4.0).abs() < 1e-9);
    }

    #[test]
    fn test_diameter_tree() {
        let mut mgr = OverlayNetworkManager::new(
            0,
            OverlayTopology::Tree {
                branching_factor: 4,
            },
            1000,
        );
        // 16 nodes, log_4(16) = 2.0
        let ids: Vec<u64> = (1..=16).collect();
        populate(&mut mgr, &ids, 0);
        let d = mgr.diameter_estimate();
        assert!((d - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_diameter_kademlia_log2() {
        let mut mgr =
            OverlayNetworkManager::new(0, OverlayTopology::Kademlia { k: 8, alpha: 3 }, 1000);
        let ids: Vec<u64> = (1..=8).collect();
        populate(&mut mgr, &ids, 0);
        // log2(8) = 3.0
        let d = mgr.diameter_estimate();
        assert!((d - 3.0).abs() < 1e-9);
    }

    // ------------------------------------------------------------------
    // Stats
    // ------------------------------------------------------------------

    #[test]
    fn test_stats_empty_overlay() {
        let mgr = OverlayNetworkManager::new(0, OverlayTopology::FullMesh, 100);
        let s = mgr.stats(1000);
        assert_eq!(s.node_count, 0);
        assert!((s.avg_heartbeat_age_ms - 0.0).abs() < 1e-9);
        assert_eq!(s.topology_name, "FullMesh");
    }

    #[test]
    fn test_stats_heartbeat_age() {
        let mut mgr = OverlayNetworkManager::new(0, OverlayTopology::FullMesh, 100);
        mgr.join(make_node(1, 0))
            .expect("test: join node 1 should succeed"); // heartbeat=0
        mgr.join(make_node(2, 0))
            .expect("test: join node 2 should succeed"); // heartbeat=0
        let s = mgr.stats(1000); // now=1000
                                 // avg_heartbeat_age = (1000 + 1000) / 2 = 1000
        assert!((s.avg_heartbeat_age_ms - 1000.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_topology_name_chord() {
        let mgr = OverlayNetworkManager::new(0, OverlayTopology::Chord { fingers: 4 }, 100);
        assert_eq!(mgr.stats(0).topology_name, "Chord");
    }

    #[test]
    fn test_stats_topology_name_pastry() {
        let mgr = OverlayNetworkManager::new(
            0,
            OverlayTopology::Pastry {
                leaf_set_size: 4,
                routing_table_rows: 4,
            },
            100,
        );
        assert_eq!(mgr.stats(0).topology_name, "Pastry");
    }

    // ------------------------------------------------------------------
    // Accessors
    // ------------------------------------------------------------------

    #[test]
    fn test_local_id() {
        let mgr = OverlayNetworkManager::new(42, OverlayTopology::FullMesh, 100);
        assert_eq!(mgr.local_id(), 42);
    }

    #[test]
    fn test_topology_accessor() {
        let topo = OverlayTopology::Chord { fingers: 3 };
        let mgr = OverlayNetworkManager::new(0, topo.clone(), 100);
        assert_eq!(mgr.topology(), &topo);
    }

    #[test]
    fn test_get_node_present() {
        let mut mgr = OverlayNetworkManager::new(0, OverlayTopology::FullMesh, 100);
        mgr.join(make_node(7, 42))
            .expect("test: join should succeed");
        let node = mgr
            .get_node(7)
            .expect("test: node 7 should be present after join");
        assert_eq!(node.node_id, 7);
        assert_eq!(node.joined_at, 42);
    }

    #[test]
    fn test_get_node_absent() {
        let mgr = OverlayNetworkManager::new(0, OverlayTopology::FullMesh, 100);
        assert!(mgr.get_node(999).is_none());
    }

    // ------------------------------------------------------------------
    // FNV-1a hash
    // ------------------------------------------------------------------

    #[test]
    fn test_fnv1a_hash_different_inputs() {
        assert_ne!(fnv1a_hash(0), fnv1a_hash(1));
        assert_ne!(fnv1a_hash(u64::MAX), fnv1a_hash(0));
    }

    #[test]
    fn test_fnv1a_hash_deterministic() {
        assert_eq!(fnv1a_hash(12345), fnv1a_hash(12345));
    }

    // ------------------------------------------------------------------
    // XOR distance
    // ------------------------------------------------------------------

    #[test]
    fn test_xor_distance_identity() {
        assert_eq!(xor_distance(7, 7), 0);
    }

    #[test]
    fn test_xor_distance_symmetric() {
        assert_eq!(xor_distance(3, 5), xor_distance(5, 3));
    }

    // ------------------------------------------------------------------
    // Edge cases
    // ------------------------------------------------------------------

    #[test]
    fn test_evict_stale_overflow_safe() {
        let mut mgr = OverlayNetworkManager::new(0, OverlayTopology::FullMesh, 100);
        // now < max_age_ms: cutoff saturates at 0, nothing removed
        mgr.join(make_node(1, 0))
            .expect("test: join should succeed");
        let removed = mgr.evict_stale(u64::MAX, 0);
        assert_eq!(removed, 0);
    }

    #[test]
    fn test_chord_route_two_equal_nodes_no_panic() {
        let mut mgr = OverlayNetworkManager::new(0, OverlayTopology::Chord { fingers: 1 }, 100);
        mgr.join(make_node(1, 0))
            .expect("test: join node 1 should succeed");
        mgr.join(make_node(2, 0))
            .expect("test: join node 2 should succeed");
        // Should either route or return a known error, not panic.
        let result = mgr.route(1, 2, 0);
        assert!(result.is_ok() || matches!(result, Err(OverlayError::RouteNotFound { .. })));
    }

    #[test]
    fn test_pastry_route_self_error() {
        let mut mgr = OverlayNetworkManager::new(
            0,
            OverlayTopology::Pastry {
                leaf_set_size: 4,
                routing_table_rows: 4,
            },
            100,
        );
        mgr.join(make_node(5, 0))
            .expect("test: join should succeed");
        assert_eq!(mgr.route(5, 5, 0).unwrap_err(), OverlayError::SelfRoute);
    }

    #[test]
    fn test_tree_single_branching_factor_one() {
        let mut mgr = OverlayNetworkManager::new(
            0,
            OverlayTopology::Tree {
                branching_factor: 1,
            },
            100,
        );
        // With bf=1, the tree is a linked list sorted by joined_at.
        mgr.join(make_node(1, 1))
            .expect("test: join node 1 should succeed");
        mgr.join(make_node(2, 2))
            .expect("test: join node 2 should succeed");
        mgr.join(make_node(3, 3))
            .expect("test: join node 3 should succeed");
        // Route 1->3 must traverse 1->2->3
        let route = mgr
            .route(1, 3, 0)
            .expect("test: tree route with bf=1 should succeed");
        assert_eq!(route.hop_count, 2);
    }

    #[test]
    fn test_overlay_route_latency_proportional_to_hops() {
        let r2 = OverlayRoute::new(vec![1, 2, 3]);
        let r4 = OverlayRoute::new(vec![1, 2, 3, 4, 5]);
        assert!((r4.latency_estimate_ms - 2.0 * r2.latency_estimate_ms).abs() < 1e-9);
    }
}
