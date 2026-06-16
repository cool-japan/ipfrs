//! Network Topology Mapper
//!
//! Discovers and maps the P2P network topology, computing graph metrics and
//! detecting structural properties.  The mapper maintains an in-memory directed
//! graph of overlay nodes and edges, and exposes algorithms for path finding,
//! reachability, clustering, and component analysis.
//!
//! # Design
//!
//! * `NetworkTopologyMapper` owns the graph state.
//! * Nodes are keyed by opaque `String` identifiers (peer IDs or any label).
//! * Edges are directed (from → to) with latency and bandwidth annotations.
//! * All algorithms operate on the in-memory graph without I/O.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};

// ─── Core graph types ─────────────────────────────────────────────────────────

/// A node in the topology graph, representing a single peer.
#[derive(Debug, Clone)]
pub struct TopoNode {
    /// Unique identifier of this peer.
    pub node_id: String,
    /// Network address (multiaddr or socket string).
    pub address: String,
    /// Unix epoch milliseconds when this node was first observed.
    pub discovered_at: u64,
    /// BFS hop distance from the local node (0 = local node).
    pub hop_distance: u32,
}

/// A directed edge between two peers.
#[derive(Debug, Clone)]
pub struct TopoEdge {
    /// Source peer identifier.
    pub from: String,
    /// Destination peer identifier.
    pub to: String,
    /// One-way (or round-trip) latency estimate in milliseconds.
    pub latency_ms: f64,
    /// Measured bandwidth in bits per second.
    pub bandwidth_bps: u64,
    /// Unix epoch milliseconds when this edge was observed.
    pub discovered_at: u64,
}

/// A point-in-time snapshot of the full topology.
#[derive(Debug, Clone)]
pub struct TopologySnapshot {
    /// All nodes in the graph at snapshot time.
    pub nodes: Vec<TopoNode>,
    /// All directed edges in the graph at snapshot time.
    pub edges: Vec<TopoEdge>,
    /// Unix epoch milliseconds when the snapshot was taken.
    pub snapshot_at: u64,
    /// Identifier of the local node.
    pub local_node_id: String,
}

/// Result of a path query.
#[derive(Debug, Clone, PartialEq)]
pub struct PathResult {
    /// Ordered list of node IDs from source to destination (inclusive).
    pub path: Vec<String>,
    /// Sum of latency_ms across all traversed edges.
    pub total_latency_ms: f64,
    /// Number of hops (= `path.len() - 1`).
    pub hop_count: usize,
}

/// Aggregate statistics for the current topology graph.
#[derive(Debug, Clone, PartialEq)]
pub struct TopologyStats {
    /// Total number of nodes.
    pub node_count: usize,
    /// Total number of directed edges.
    pub edge_count: usize,
    /// Mean latency across all edges (0.0 if no edges).
    pub avg_latency_ms: f64,
    /// Maximum latency across all edges (0.0 if no edges).
    pub max_latency_ms: f64,
    /// Mean out-degree across all nodes (0.0 if no nodes).
    pub avg_out_degree: f64,
    /// Number of weakly connected components.
    pub component_count: usize,
    /// Graph diameter (max shortest-path latency), or `None` if < 2 nodes.
    pub diameter: Option<f64>,
}

// ─── Legacy compatibility aliases ────────────────────────────────────────────

/// Legacy: directed edge (alias for `TopoEdge`).
///
/// Retained so existing re-exports from `lib.rs` continue to compile.
pub type PeerEdge = TopoEdge;

/// Legacy: topology node (alias for `TopoNode`).
///
/// Retained so existing re-exports from `lib.rs` continue to compile.
pub type TopologyNode = TopoNode;

// ─── Main mapper ──────────────────────────────────────────────────────────────

/// Discovers and maps the P2P network topology.
///
/// # Example
///
/// ```rust
/// use ipfrs_network::topology_mapper::{NetworkTopologyMapper, TopoNode, TopoEdge};
///
/// let mut mapper = NetworkTopologyMapper::new("local".to_string());
/// mapper.add_node(TopoNode {
///     node_id: "peer-1".to_string(),
///     address: "/ip4/127.0.0.1/tcp/4001".to_string(),
///     discovered_at: 0,
///     hop_distance: 1,
/// });
/// mapper.add_edge(TopoEdge {
///     from: "local".to_string(),
///     to: "peer-1".to_string(),
///     latency_ms: 5.0,
///     bandwidth_bps: 10_000_000,
///     discovered_at: 0,
/// });
/// let path = mapper.shortest_path("local", "peer-1");
/// assert!(path.is_some());
/// ```
#[derive(Debug, Clone)]
pub struct NetworkTopologyMapper {
    /// Identifier of the local node (the observer).
    pub local_id: String,
    /// All known nodes keyed by `node_id`.
    pub nodes: HashMap<String, TopoNode>,
    /// All directed edges (flat list; may contain duplicate `from→to` pairs
    /// if the caller adds them; the last-added is preferred for path finding).
    pub edges: Vec<TopoEdge>,
    /// Adjacency list: `node_id → [neighbour_ids]` (outgoing only).
    pub adjacency: HashMap<String, Vec<String>>,
}

impl NetworkTopologyMapper {
    // ── Construction ──────────────────────────────────────────────────────────

    /// Create a new mapper with the given local node ID.
    pub fn new(local_id: String) -> Self {
        Self {
            local_id,
            nodes: HashMap::new(),
            edges: Vec::new(),
            adjacency: HashMap::new(),
        }
    }

    // ── Node management ───────────────────────────────────────────────────────

    /// Insert or update a node.  When a node is updated its hop_distance and
    /// address are refreshed; `discovered_at` is kept if the new value is 0.
    pub fn add_node(&mut self, node: TopoNode) {
        let id = node.node_id.clone();
        self.nodes.insert(id.clone(), node);
        // Ensure an adjacency entry exists (even if empty).
        self.adjacency.entry(id).or_default();
        // Rebuild adjacency from stored edges to keep it consistent.
        self.rebuild_adjacency_for_edges();
    }

    /// Remove a node and all edges that reference it (as source or destination).
    ///
    /// Returns `true` if the node existed, `false` otherwise.
    pub fn remove_node(&mut self, node_id: &str) -> bool {
        if self.nodes.remove(node_id).is_none() {
            return false;
        }
        self.edges.retain(|e| e.from != node_id && e.to != node_id);
        self.adjacency.remove(node_id);
        // Remove node_id from all adjacency lists where it appears as a neighbour.
        for neighbours in self.adjacency.values_mut() {
            neighbours.retain(|n| n != node_id);
        }
        true
    }

    // ── Edge management ───────────────────────────────────────────────────────

    /// Add a directed edge and update the adjacency list for `from`.
    ///
    /// Duplicate `from→to` edges are allowed; path queries always use the edge
    /// with the **lowest latency** between a pair.
    pub fn add_edge(&mut self, edge: TopoEdge) {
        let from = edge.from.clone();
        let to = edge.to.clone();
        self.edges.push(edge);
        // Update adjacency: add `to` to `from`'s neighbour list if not present.
        let neighbours = self.adjacency.entry(from).or_default();
        if !neighbours.contains(&to) {
            neighbours.push(to.clone());
        }
        // Ensure the destination has an adjacency entry.
        self.adjacency.entry(to).or_default();
    }

    /// Remove all edges from `from` to `to`.
    ///
    /// Returns `true` if at least one edge was removed.
    pub fn remove_edge(&mut self, from: &str, to: &str) -> bool {
        let before = self.edges.len();
        self.edges.retain(|e| !(e.from == from && e.to == to));
        let removed = self.edges.len() < before;
        if removed {
            // Rebuild adjacency for `from`.
            if let Some(nbrs) = self.adjacency.get_mut(from) {
                // Only remove `to` if no remaining edge from→to exists.
                let still_connected = self.edges.iter().any(|e| e.from == from && e.to == to);
                if !still_connected {
                    nbrs.retain(|n| n != to);
                }
            }
        }
        removed
    }

    // ── Path finding ──────────────────────────────────────────────────────────

    /// Dijkstra shortest path weighted by `latency_ms`.
    ///
    /// Returns `None` if either node is unknown or no path exists.
    pub fn shortest_path(&self, from: &str, to: &str) -> Option<PathResult> {
        if from == to {
            return None;
        }

        // Build a per-pair minimum-latency map from edges.
        let min_lat = self.min_latency_map();

        // dist[node] = current best (total_latency, prev_node)
        let mut dist: HashMap<&str, f64> = HashMap::new();
        let mut prev: HashMap<&str, &str> = HashMap::new();
        // BinaryHeap with Reverse for min-heap: (Reverse(latency_bits), node_id)
        let mut heap: BinaryHeap<(Reverse<u64>, &str)> = BinaryHeap::new();

        dist.insert(from, 0.0);
        heap.push((Reverse(0), from));

        while let Some((Reverse(cost_bits), node)) = heap.pop() {
            let cost = f64::from_bits(cost_bits);

            if node == to {
                // Reconstruct path.
                let mut path: Vec<String> = Vec::new();
                let mut cur = to;
                loop {
                    path.push(cur.to_string());
                    if let Some(&p) = prev.get(cur) {
                        cur = p;
                    } else {
                        break;
                    }
                }
                path.reverse();
                let hop_count = path.len().saturating_sub(1);
                return Some(PathResult {
                    path,
                    total_latency_ms: cost,
                    hop_count,
                });
            }

            // Skip stale heap entries.
            if let Some(&best) = dist.get(node) {
                if cost > best {
                    continue;
                }
            }

            // Explore neighbours.
            if let Some(neighbours) = self.adjacency.get(node) {
                for nb in neighbours {
                    let edge_lat = min_lat
                        .get(&(node, nb.as_str()))
                        .copied()
                        .unwrap_or(f64::INFINITY);
                    let new_cost = cost + edge_lat;
                    let better = dist.get(nb.as_str()).map(|&d| new_cost < d).unwrap_or(true);
                    if better {
                        dist.insert(nb.as_str(), new_cost);
                        prev.insert(nb.as_str(), node);
                        heap.push((Reverse(new_cost.to_bits()), nb.as_str()));
                    }
                }
            }
        }

        None
    }

    /// BFS shortest path (unweighted; minimises hop count).
    ///
    /// Returns `None` if either node is unknown or no path exists.
    pub fn hop_shortest_path(&self, from: &str, to: &str) -> Option<PathResult> {
        if from == to {
            return None;
        }

        let mut visited: HashSet<&str> = HashSet::new();
        let mut queue: VecDeque<(&str, Vec<&str>)> = VecDeque::new();

        visited.insert(from);
        queue.push_back((from, vec![from]));

        while let Some((node, path)) = queue.pop_front() {
            if let Some(neighbours) = self.adjacency.get(node) {
                for nb in neighbours {
                    let nb_str = nb.as_str();
                    if nb_str == to {
                        let mut full: Vec<String> = path.iter().map(|s| s.to_string()).collect();
                        full.push(to.to_string());
                        let hop_count = full.len().saturating_sub(1);
                        // Compute total latency along the hop path.
                        let min_lat = self.min_latency_map();
                        let total = full
                            .windows(2)
                            .map(|w| {
                                min_lat
                                    .get(&(w[0].as_str(), w[1].as_str()))
                                    .copied()
                                    .unwrap_or(0.0)
                            })
                            .sum();
                        return Some(PathResult {
                            path: full,
                            total_latency_ms: total,
                            hop_count,
                        });
                    }
                    if !visited.contains(nb_str) {
                        visited.insert(nb_str);
                        let mut new_path = path.clone();
                        new_path.push(nb_str);
                        queue.push_back((nb_str, new_path));
                    }
                }
            }
        }

        None
    }

    // ── Graph queries ─────────────────────────────────────────────────────────

    /// Return the outgoing neighbour IDs for `node_id`.
    pub fn neighbors(&self, node_id: &str) -> Vec<&str> {
        self.adjacency
            .get(node_id)
            .map(|v| v.iter().map(String::as_str).collect())
            .unwrap_or_default()
    }

    /// Number of edges pointing **into** `node_id`.
    pub fn in_degree(&self, node_id: &str) -> usize {
        self.edges.iter().filter(|e| e.to == node_id).count()
    }

    /// Number of outgoing edges from `node_id`.
    pub fn out_degree(&self, node_id: &str) -> usize {
        self.edges.iter().filter(|e| e.from == node_id).count()
    }

    /// BFS reachability check.
    pub fn is_reachable(&self, from: &str, to: &str) -> bool {
        if from == to {
            return true;
        }
        let mut visited: HashSet<&str> = HashSet::new();
        let mut queue: VecDeque<&str> = VecDeque::new();
        visited.insert(from);
        queue.push_back(from);
        while let Some(node) = queue.pop_front() {
            if let Some(nbrs) = self.adjacency.get(node) {
                for nb in nbrs {
                    let nb_str = nb.as_str();
                    if nb_str == to {
                        return true;
                    }
                    if visited.insert(nb_str) {
                        queue.push_back(nb_str);
                    }
                }
            }
        }
        false
    }

    /// Compute weakly connected components (treat directed edges as undirected).
    ///
    /// Each component is a sorted list of node IDs.  The returned list of
    /// components is sorted by the first element of each component.
    pub fn connected_components(&self) -> Vec<Vec<String>> {
        // Build an undirected adjacency set from all edges + node entries.
        let mut undirected: HashMap<&str, HashSet<&str>> = HashMap::new();

        // Seed every known node (even isolated ones).
        for id in self.nodes.keys() {
            undirected.entry(id.as_str()).or_default();
        }
        for id in self.adjacency.keys() {
            undirected.entry(id.as_str()).or_default();
        }

        for edge in &self.edges {
            undirected
                .entry(edge.from.as_str())
                .or_default()
                .insert(edge.to.as_str());
            undirected
                .entry(edge.to.as_str())
                .or_default()
                .insert(edge.from.as_str());
        }

        let mut visited: HashSet<&str> = HashSet::new();
        let mut components: Vec<Vec<String>> = Vec::new();

        for &start in undirected.keys() {
            if visited.contains(start) {
                continue;
            }
            // BFS over undirected graph.
            let mut component: Vec<String> = Vec::new();
            let mut queue: VecDeque<&str> = VecDeque::new();
            visited.insert(start);
            queue.push_back(start);
            while let Some(node) = queue.pop_front() {
                component.push(node.to_string());
                if let Some(nbrs) = undirected.get(node) {
                    for &nb in nbrs {
                        if visited.insert(nb) {
                            queue.push_back(nb);
                        }
                    }
                }
            }
            component.sort();
            components.push(component);
        }

        components.sort_by(|a, b| {
            a.first()
                .unwrap_or(&String::new())
                .cmp(b.first().unwrap_or(&String::new()))
        });
        components
    }

    /// Graph diameter: the maximum shortest-path latency over all reachable pairs.
    ///
    /// Returns `None` when there are fewer than two nodes.
    /// This is O(V × Dijkstra) and is expensive for large graphs.
    pub fn diameter(&self) -> Option<f64> {
        let ids: Vec<&str> = self
            .nodes
            .keys()
            .map(String::as_str)
            .chain(self.adjacency.keys().map(String::as_str))
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();

        if ids.len() < 2 {
            return None;
        }

        let mut max_lat: f64 = 0.0;
        let mut found_any = false;

        for &src in &ids {
            for &dst in &ids {
                if src == dst {
                    continue;
                }
                if let Some(res) = self.shortest_path(src, dst) {
                    found_any = true;
                    if res.total_latency_ms > max_lat {
                        max_lat = res.total_latency_ms;
                    }
                }
            }
        }

        if found_any {
            Some(max_lat)
        } else {
            None
        }
    }

    /// Mean clustering coefficient (undirected interpretation).
    ///
    /// For each node whose combined (in+out) neighbour set has size ≥ 2, the
    /// local clustering coefficient is:
    ///
    ///   C(v) = (actual triangles through v) / (k * (k-1) / 2)
    ///
    /// where `k` is the size of the undirected neighbourhood, and a "triangle"
    /// exists when two neighbours are themselves connected (in either direction).
    ///
    /// Returns 0.0 if no node has k ≥ 2.
    pub fn average_clustering_coefficient(&self) -> f64 {
        let mut total_cc = 0.0_f64;
        let mut count = 0_usize;

        // Build undirected neighbour sets for all nodes.
        let undirected_nbrs = self.undirected_neighbour_sets();
        // Build a fast edge-existence lookup (directed).
        let edge_set: HashSet<(&str, &str)> = self
            .edges
            .iter()
            .map(|e| (e.from.as_str(), e.to.as_str()))
            .collect();

        for (node, nbrs) in &undirected_nbrs {
            let k = nbrs.len();
            if k < 2 {
                continue;
            }
            let nbr_vec: Vec<&str> = nbrs.iter().copied().collect();
            let mut triangles = 0_usize;
            for i in 0..nbr_vec.len() {
                for j in (i + 1)..nbr_vec.len() {
                    let u = nbr_vec[i];
                    let v = nbr_vec[j];
                    // Check undirected connection between u and v.
                    if edge_set.contains(&(u, v)) || edge_set.contains(&(v, u)) {
                        triangles += 1;
                    }
                }
            }
            let possible = (k * (k - 1)) / 2;
            let cc = triangles as f64 / possible as f64;
            total_cc += cc;
            count += 1;
            let _ = node; // suppress unused-variable lint
        }

        if count == 0 {
            0.0
        } else {
            total_cc / count as f64
        }
    }

    // ── Snapshot / stats ──────────────────────────────────────────────────────

    /// Take a snapshot of the current topology.
    pub fn snapshot(&self) -> TopologySnapshot {
        use std::time::{SystemTime, UNIX_EPOCH};
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        TopologySnapshot {
            nodes: self.nodes.values().cloned().collect(),
            edges: self.edges.clone(),
            snapshot_at: now,
            local_node_id: self.local_id.clone(),
        }
    }

    /// Total number of known nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Total number of edges (including duplicates for the same pair).
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Remove nodes whose `discovered_at` is older than `max_age_ms` relative
    /// to `now`.  All edges referencing removed nodes are also cleaned up.
    ///
    /// Returns the number of nodes removed.
    pub fn evict_stale(&mut self, max_age_ms: u64, now: u64) -> usize {
        let stale_ids: Vec<String> = self
            .nodes
            .iter()
            .filter(|(_, n)| now.saturating_sub(n.discovered_at) > max_age_ms)
            .map(|(id, _)| id.clone())
            .collect();

        let removed = stale_ids.len();
        for id in &stale_ids {
            self.nodes.remove(id);
            self.edges.retain(|e| &e.from != id && &e.to != id);
            self.adjacency.remove(id);
            for nbrs in self.adjacency.values_mut() {
                nbrs.retain(|n| n != id);
            }
        }
        removed
    }

    /// Compute aggregate statistics for the current topology.
    pub fn stats(&self) -> TopologyStats {
        let node_count = self.node_count();
        let edge_count = self.edge_count();

        let (avg_latency_ms, max_latency_ms) = if edge_count == 0 {
            (0.0, 0.0)
        } else {
            let sum: f64 = self.edges.iter().map(|e| e.latency_ms).sum();
            let max = self
                .edges
                .iter()
                .map(|e| e.latency_ms)
                .fold(0.0_f64, f64::max);
            (sum / edge_count as f64, max)
        };

        let avg_out_degree = if node_count == 0 {
            0.0
        } else {
            edge_count as f64 / node_count as f64
        };

        let component_count = self.connected_components().len();
        let diameter = self.diameter();

        TopologyStats {
            node_count,
            edge_count,
            avg_latency_ms,
            max_latency_ms,
            avg_out_degree,
            component_count,
            diameter,
        }
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    /// Build a map `(from, to) → minimum latency` from the edge list.
    fn min_latency_map(&self) -> HashMap<(&str, &str), f64> {
        let mut map: HashMap<(&str, &str), f64> = HashMap::new();
        for edge in &self.edges {
            let key = (edge.from.as_str(), edge.to.as_str());
            let entry = map.entry(key).or_insert(f64::INFINITY);
            if edge.latency_ms < *entry {
                *entry = edge.latency_ms;
            }
        }
        map
    }

    /// Rebuild the full adjacency map from the current edge list.
    fn rebuild_adjacency_for_edges(&mut self) {
        // Keep existing entries but re-derive neighbour lists from edges.
        // Existing nodes that have no edges still get an empty entry.
        for id in self.nodes.keys() {
            self.adjacency.entry(id.clone()).or_default();
        }
        for edge in &self.edges {
            let from = edge.from.clone();
            let to = edge.to.clone();
            let nbrs = self.adjacency.entry(from).or_default();
            if !nbrs.contains(&to) {
                nbrs.push(to);
            }
            self.adjacency.entry(edge.to.clone()).or_default();
        }
    }

    /// Build per-node undirected neighbour sets (used for clustering coefficient).
    fn undirected_neighbour_sets(&self) -> HashMap<&str, HashSet<&str>> {
        let mut map: HashMap<&str, HashSet<&str>> = HashMap::new();
        for id in self.nodes.keys() {
            map.entry(id.as_str()).or_default();
        }
        for edge in &self.edges {
            map.entry(edge.from.as_str())
                .or_default()
                .insert(edge.to.as_str());
            map.entry(edge.to.as_str())
                .or_default()
                .insert(edge.from.as_str());
        }
        map
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use crate::topology_mapper::{
        NetworkTopologyMapper, PathResult, TopoEdge, TopoNode, TopologyStats,
    };

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn node(id: &str, hop: u32) -> TopoNode {
        TopoNode {
            node_id: id.to_string(),
            address: format!("/ip4/127.0.0.1/tcp/{}", 4000 + hop),
            discovered_at: 1_000,
            hop_distance: hop,
        }
    }

    fn edge(from: &str, to: &str, lat: f64) -> TopoEdge {
        TopoEdge {
            from: from.to_string(),
            to: to.to_string(),
            latency_ms: lat,
            bandwidth_bps: 1_000_000,
            discovered_at: 1_000,
        }
    }

    fn edge_ts(from: &str, to: &str, lat: f64, ts: u64) -> TopoEdge {
        TopoEdge {
            from: from.to_string(),
            to: to.to_string(),
            latency_ms: lat,
            bandwidth_bps: 1_000_000,
            discovered_at: ts,
        }
    }

    // Build a simple linear graph: local → A → B → C
    fn linear_mapper() -> NetworkTopologyMapper {
        let mut m = NetworkTopologyMapper::new("local".to_string());
        for (id, hop) in [("local", 0), ("A", 1), ("B", 2), ("C", 3)] {
            m.add_node(node(id, hop));
        }
        m.add_edge(edge("local", "A", 10.0));
        m.add_edge(edge("A", "B", 20.0));
        m.add_edge(edge("B", "C", 30.0));
        m
    }

    // ── 1. Constructor ────────────────────────────────────────────────────────

    #[test]
    fn test_new_empty() {
        let m = NetworkTopologyMapper::new("local".to_string());
        assert_eq!(m.local_id, "local");
        assert!(m.nodes.is_empty());
        assert!(m.edges.is_empty());
        assert!(m.adjacency.is_empty());
    }

    // ── 2. add_node ───────────────────────────────────────────────────────────

    #[test]
    fn test_add_node_creates_entry() {
        let mut m = NetworkTopologyMapper::new("local".to_string());
        m.add_node(node("A", 1));
        assert!(m.nodes.contains_key("A"));
        assert!(m.adjacency.contains_key("A"));
    }

    #[test]
    fn test_add_node_updates_existing() {
        let mut m = NetworkTopologyMapper::new("local".to_string());
        m.add_node(node("A", 1));
        m.add_node(TopoNode {
            node_id: "A".to_string(),
            address: "/ip4/1.2.3.4/tcp/9999".to_string(),
            discovered_at: 9999,
            hop_distance: 3,
        });
        let n = m.nodes.get("A").expect("A should exist");
        assert_eq!(n.address, "/ip4/1.2.3.4/tcp/9999");
        assert_eq!(n.hop_distance, 3);
    }

    // ── 3. remove_node ────────────────────────────────────────────────────────

    #[test]
    fn test_remove_node_returns_true_when_exists() {
        let mut m = NetworkTopologyMapper::new("local".to_string());
        m.add_node(node("A", 1));
        assert!(m.remove_node("A"));
    }

    #[test]
    fn test_remove_node_returns_false_when_missing() {
        let mut m = NetworkTopologyMapper::new("local".to_string());
        assert!(!m.remove_node("Z"));
    }

    #[test]
    fn test_remove_node_cleans_edges() {
        let mut m = linear_mapper();
        m.remove_node("A");
        assert!(!m.edges.iter().any(|e| e.from == "A" || e.to == "A"));
    }

    #[test]
    fn test_remove_node_cleans_adjacency() {
        let mut m = linear_mapper();
        m.remove_node("A");
        assert!(!m.adjacency.contains_key("A"));
        // "local" should no longer list "A" as a neighbour.
        let local_nbrs = m.adjacency.get("local").cloned().unwrap_or_default();
        assert!(!local_nbrs.contains(&"A".to_string()));
    }

    // ── 4. add_edge ───────────────────────────────────────────────────────────

    #[test]
    fn test_add_edge_updates_adjacency() {
        let mut m = NetworkTopologyMapper::new("local".to_string());
        m.add_edge(edge("X", "Y", 5.0));
        let nbrs = m.adjacency.get("X").expect("X should have adjacency");
        assert!(nbrs.contains(&"Y".to_string()));
    }

    #[test]
    fn test_add_edge_no_duplicate_adjacency() {
        let mut m = NetworkTopologyMapper::new("local".to_string());
        m.add_edge(edge("X", "Y", 5.0));
        m.add_edge(edge("X", "Y", 10.0));
        let nbrs = m.adjacency.get("X").expect("X should exist");
        assert_eq!(nbrs.iter().filter(|n| n.as_str() == "Y").count(), 1);
    }

    // ── 5. remove_edge ────────────────────────────────────────────────────────

    #[test]
    fn test_remove_edge_returns_true() {
        let mut m = linear_mapper();
        assert!(m.remove_edge("local", "A"));
    }

    #[test]
    fn test_remove_edge_returns_false_when_missing() {
        let mut m = linear_mapper();
        assert!(!m.remove_edge("local", "C"));
    }

    #[test]
    fn test_remove_edge_removes_from_edge_list() {
        let mut m = linear_mapper();
        m.remove_edge("local", "A");
        assert!(!m.edges.iter().any(|e| e.from == "local" && e.to == "A"));
    }

    // ── 6. shortest_path (Dijkstra) ───────────────────────────────────────────

    #[test]
    fn test_shortest_path_direct() {
        let m = linear_mapper();
        let r = m.shortest_path("local", "A").expect("path should exist");
        assert_eq!(r.path, vec!["local", "A"]);
        assert_eq!(r.hop_count, 1);
        assert!((r.total_latency_ms - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_shortest_path_multi_hop() {
        let m = linear_mapper();
        let r = m.shortest_path("local", "C").expect("path should exist");
        assert_eq!(r.path, vec!["local", "A", "B", "C"]);
        assert_eq!(r.hop_count, 3);
        assert!((r.total_latency_ms - 60.0).abs() < 1e-9);
    }

    #[test]
    fn test_shortest_path_prefers_low_latency() {
        let mut m = NetworkTopologyMapper::new("s".to_string());
        for id in ["s", "fast", "slow", "t"] {
            m.add_node(node(id, 0));
        }
        // s → fast → t (total 5)  vs  s → slow → t (total 1000)
        m.add_edge(edge("s", "fast", 2.0));
        m.add_edge(edge("fast", "t", 3.0));
        m.add_edge(edge("s", "slow", 500.0));
        m.add_edge(edge("slow", "t", 500.0));
        let r = m.shortest_path("s", "t").expect("path should exist");
        assert_eq!(r.path, vec!["s", "fast", "t"]);
        assert!((r.total_latency_ms - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_shortest_path_same_node_returns_none() {
        let m = linear_mapper();
        assert!(m.shortest_path("local", "local").is_none());
    }

    #[test]
    fn test_shortest_path_disconnected_returns_none() {
        let mut m = linear_mapper();
        m.add_node(node("island", 99));
        assert!(m.shortest_path("local", "island").is_none());
    }

    // ── 7. hop_shortest_path (BFS) ────────────────────────────────────────────

    #[test]
    fn test_hop_shortest_path_direct() {
        let m = linear_mapper();
        let r = m
            .hop_shortest_path("local", "A")
            .expect("path should exist");
        assert_eq!(r.hop_count, 1);
        assert_eq!(r.path, vec!["local", "A"]);
    }

    #[test]
    fn test_hop_shortest_path_prefers_fewer_hops() {
        let mut m = NetworkTopologyMapper::new("s".to_string());
        for id in ["s", "x", "y", "z", "t"] {
            m.add_node(node(id, 0));
        }
        // s → t (1 hop, high latency)  vs  s → x → y → z → t (4 hops, low latency)
        m.add_edge(edge("s", "t", 9999.0));
        m.add_edge(edge("s", "x", 1.0));
        m.add_edge(edge("x", "y", 1.0));
        m.add_edge(edge("y", "z", 1.0));
        m.add_edge(edge("z", "t", 1.0));
        let r = m.hop_shortest_path("s", "t").expect("path should exist");
        assert_eq!(r.hop_count, 1);
        assert_eq!(r.path, vec!["s", "t"]);
    }

    #[test]
    fn test_hop_shortest_path_same_node_none() {
        let m = linear_mapper();
        assert!(m.hop_shortest_path("A", "A").is_none());
    }

    // ── 8. neighbors ──────────────────────────────────────────────────────────

    #[test]
    fn test_neighbors_correct() {
        let m = linear_mapper();
        let mut nbrs = m.neighbors("local");
        nbrs.sort_unstable();
        assert_eq!(nbrs, vec!["A"]);
    }

    #[test]
    fn test_neighbors_unknown_node_empty() {
        let m = linear_mapper();
        assert!(m.neighbors("zzz").is_empty());
    }

    // ── 9. in_degree / out_degree ─────────────────────────────────────────────

    #[test]
    fn test_in_degree() {
        let m = linear_mapper();
        // Only "local → A" points to A.
        assert_eq!(m.in_degree("A"), 1);
        // "C" has one in-edge (B→C).
        assert_eq!(m.in_degree("C"), 1);
        // "local" has no in-edges.
        assert_eq!(m.in_degree("local"), 0);
    }

    #[test]
    fn test_out_degree() {
        let m = linear_mapper();
        assert_eq!(m.out_degree("local"), 1);
        assert_eq!(m.out_degree("A"), 1);
        assert_eq!(m.out_degree("C"), 0);
    }

    // ── 10. is_reachable ──────────────────────────────────────────────────────

    #[test]
    fn test_is_reachable_direct() {
        let m = linear_mapper();
        assert!(m.is_reachable("local", "A"));
    }

    #[test]
    fn test_is_reachable_transitive() {
        let m = linear_mapper();
        assert!(m.is_reachable("local", "C"));
    }

    #[test]
    fn test_is_reachable_same_node() {
        let m = linear_mapper();
        assert!(m.is_reachable("local", "local"));
    }

    #[test]
    fn test_is_reachable_unreachable() {
        let mut m = linear_mapper();
        m.add_node(node("island", 99));
        assert!(!m.is_reachable("local", "island"));
    }

    #[test]
    fn test_is_reachable_reverse_not_reachable() {
        // The graph is directed, so C cannot reach local.
        let m = linear_mapper();
        assert!(!m.is_reachable("C", "local"));
    }

    // ── 11. connected_components ──────────────────────────────────────────────

    #[test]
    fn test_connected_components_single() {
        let m = linear_mapper();
        let comps = m.connected_components();
        // All four nodes are weakly connected.
        assert_eq!(comps.len(), 1);
        let mut comp = comps[0].clone();
        comp.sort();
        assert_eq!(comp, vec!["A", "B", "C", "local"]);
    }

    #[test]
    fn test_connected_components_two_islands() {
        let mut m = NetworkTopologyMapper::new("L".to_string());
        for id in ["L", "M", "X", "Y"] {
            m.add_node(node(id, 0));
        }
        m.add_edge(edge("L", "M", 1.0));
        m.add_edge(edge("X", "Y", 1.0));
        let comps = m.connected_components();
        assert_eq!(comps.len(), 2);
    }

    #[test]
    fn test_connected_components_sorted() {
        let mut m = NetworkTopologyMapper::new("Z".to_string());
        for id in ["Z", "A", "M", "N"] {
            m.add_node(node(id, 0));
        }
        m.add_edge(edge("Z", "A", 1.0));
        m.add_edge(edge("M", "N", 1.0));
        let comps = m.connected_components();
        // Each component is sorted internally; components sorted by first element.
        for comp in &comps {
            let mut sorted = comp.clone();
            sorted.sort();
            assert_eq!(comp, &sorted, "component should be sorted");
        }
        // Components themselves are sorted by first element.
        for i in 1..comps.len() {
            assert!(comps[i - 1][0] <= comps[i][0]);
        }
    }

    // ── 12. diameter ──────────────────────────────────────────────────────────

    #[test]
    fn test_diameter_linear() {
        let m = linear_mapper();
        // local → C = 10+20+30 = 60, which should be the max.
        let d = m.diameter().expect("diameter should exist");
        assert!((d - 60.0).abs() < 1e-9);
    }

    #[test]
    fn test_diameter_none_single_node() {
        let mut m = NetworkTopologyMapper::new("x".to_string());
        m.add_node(node("x", 0));
        assert!(m.diameter().is_none());
    }

    #[test]
    fn test_diameter_none_disconnected_graph() {
        // Two fully isolated nodes — no reachable pairs → None.
        let mut m = NetworkTopologyMapper::new("a".to_string());
        m.add_node(node("a", 0));
        m.add_node(node("b", 0));
        // No edges; diameter is None because no reachable pair exists.
        assert!(m.diameter().is_none());
    }

    // ── 13. average_clustering_coefficient ────────────────────────────────────

    #[test]
    fn test_clustering_coefficient_triangle() {
        let mut m = NetworkTopologyMapper::new("A".to_string());
        for id in ["A", "B", "C"] {
            m.add_node(node(id, 0));
        }
        // Full triangle (directed, treated as undirected).
        m.add_edge(edge("A", "B", 1.0));
        m.add_edge(edge("B", "C", 1.0));
        m.add_edge(edge("A", "C", 1.0));
        let cc = m.average_clustering_coefficient();
        // Every node has k=2, one triangle → C = 1/1 = 1.0 each → mean = 1.0.
        assert!((cc - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_clustering_coefficient_no_triangles() {
        let m = linear_mapper();
        // Linear graph — no triangles → local CC = 0 for A (only neighbour).
        // Actually A has neighbours {local, B} (undirected); are local and B connected? No.
        let cc = m.average_clustering_coefficient();
        assert!((cc - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_clustering_coefficient_empty() {
        let m = NetworkTopologyMapper::new("x".to_string());
        assert_eq!(m.average_clustering_coefficient(), 0.0);
    }

    // ── 14. snapshot ──────────────────────────────────────────────────────────

    #[test]
    fn test_snapshot_counts() {
        let m = linear_mapper();
        let snap = m.snapshot();
        assert_eq!(snap.local_node_id, "local");
        assert_eq!(snap.nodes.len(), 4);
        assert_eq!(snap.edges.len(), 3);
    }

    // ── 15. node_count / edge_count ───────────────────────────────────────────

    #[test]
    fn test_node_count() {
        let m = linear_mapper();
        assert_eq!(m.node_count(), 4);
    }

    #[test]
    fn test_edge_count() {
        let m = linear_mapper();
        assert_eq!(m.edge_count(), 3);
    }

    // ── 16. evict_stale ───────────────────────────────────────────────────────

    #[test]
    fn test_evict_stale_removes_old_nodes() {
        let mut m = NetworkTopologyMapper::new("local".to_string());
        m.add_node(TopoNode {
            node_id: "old".to_string(),
            address: "".to_string(),
            discovered_at: 0,
            hop_distance: 1,
        });
        m.add_node(TopoNode {
            node_id: "fresh".to_string(),
            address: "".to_string(),
            discovered_at: 900,
            hop_distance: 1,
        });
        // now=1000, max_age=500 → age_of_old=1000>500 stale; age_of_fresh=100 fresh.
        let removed = m.evict_stale(500, 1000);
        assert_eq!(removed, 1);
        assert!(!m.nodes.contains_key("old"));
        assert!(m.nodes.contains_key("fresh"));
    }

    #[test]
    fn test_evict_stale_removes_associated_edges() {
        let mut m = NetworkTopologyMapper::new("local".to_string());
        m.add_node(TopoNode {
            node_id: "A".to_string(),
            address: "".to_string(),
            discovered_at: 0,
            hop_distance: 1,
        });
        m.add_node(TopoNode {
            node_id: "B".to_string(),
            address: "".to_string(),
            discovered_at: 999,
            hop_distance: 2,
        });
        m.add_edge(edge("A", "B", 5.0));
        m.evict_stale(500, 1000);
        // Edge A→B should be gone because A was evicted.
        assert!(!m.edges.iter().any(|e| e.from == "A" || e.to == "A"));
    }

    #[test]
    fn test_evict_stale_returns_zero_when_all_fresh() {
        let m = linear_mapper(); // all nodes at discovered_at=1000
        let mut m2 = m.clone();
        let removed = m2.evict_stale(5000, 2000); // max_age=5000, now=2000
        assert_eq!(removed, 0);
    }

    // ── 17. stats ─────────────────────────────────────────────────────────────

    #[test]
    fn test_stats_empty() {
        let m = NetworkTopologyMapper::new("local".to_string());
        let s = m.stats();
        assert_eq!(
            s,
            TopologyStats {
                node_count: 0,
                edge_count: 0,
                avg_latency_ms: 0.0,
                max_latency_ms: 0.0,
                avg_out_degree: 0.0,
                component_count: 0,
                diameter: None,
            }
        );
    }

    #[test]
    fn test_stats_linear_graph() {
        let m = linear_mapper();
        let s = m.stats();
        assert_eq!(s.node_count, 4);
        assert_eq!(s.edge_count, 3);
        // avg latency = (10+20+30)/3 = 20
        assert!((s.avg_latency_ms - 20.0).abs() < 1e-9);
        // max latency = 30
        assert!((s.max_latency_ms - 30.0).abs() < 1e-9);
        assert_eq!(s.component_count, 1);
    }

    // ── 18. PathResult equality ───────────────────────────────────────────────

    #[test]
    fn test_path_result_equality() {
        let p1 = PathResult {
            path: vec!["A".to_string(), "B".to_string()],
            total_latency_ms: 5.0,
            hop_count: 1,
        };
        let p2 = p1.clone();
        assert_eq!(p1, p2);
    }

    // ── 19. Multiple edges same pair ──────────────────────────────────────────

    #[test]
    fn test_shortest_path_uses_minimum_latency_edge() {
        let mut m = NetworkTopologyMapper::new("s".to_string());
        m.add_node(node("s", 0));
        m.add_node(node("t", 0));
        m.add_edge(edge("s", "t", 100.0));
        m.add_edge(edge("s", "t", 5.0));
        let r = m.shortest_path("s", "t").expect("path should exist");
        assert!((r.total_latency_ms - 5.0).abs() < 1e-9);
    }

    // ── 20. Edge timestamp field ──────────────────────────────────────────────

    #[test]
    fn test_edge_has_discovered_at() {
        let e = edge_ts("X", "Y", 1.0, 42_000);
        assert_eq!(e.discovered_at, 42_000);
    }

    // ── 21. Rebuild adjacency after add_node ──────────────────────────────────

    #[test]
    fn test_add_node_rebuilds_adjacency() {
        let mut m = NetworkTopologyMapper::new("local".to_string());
        m.add_edge(edge("local", "peer", 1.0));
        m.add_node(node("peer", 1));
        let nbrs = m.neighbors("local");
        assert!(nbrs.contains(&"peer"));
    }

    // ── 22. is_reachable after remove_edge ────────────────────────────────────

    #[test]
    fn test_is_reachable_after_remove_edge() {
        let mut m = linear_mapper();
        m.remove_edge("A", "B");
        assert!(!m.is_reachable("local", "B"));
    }

    // ── 23. evict_stale boundary ──────────────────────────────────────────────

    #[test]
    fn test_evict_stale_boundary_exact_age() {
        // discovered_at=500, now=1000, max_age=500 → age=500, which is NOT > 500 → not stale.
        let mut m = NetworkTopologyMapper::new("local".to_string());
        m.add_node(TopoNode {
            node_id: "A".to_string(),
            address: "".to_string(),
            discovered_at: 500,
            hop_distance: 1,
        });
        let removed = m.evict_stale(500, 1000);
        assert_eq!(removed, 0);
        assert!(m.nodes.contains_key("A"));
    }

    // ── 24. hop_shortest_path returns none for disconnected ───────────────────

    #[test]
    fn test_hop_shortest_path_disconnected_none() {
        let mut m = linear_mapper();
        m.add_node(node("isolated", 99));
        assert!(m.hop_shortest_path("local", "isolated").is_none());
    }

    // ── 25. in_degree multiple ────────────────────────────────────────────────

    #[test]
    fn test_in_degree_multiple_sources() {
        let mut m = NetworkTopologyMapper::new("hub".to_string());
        for id in ["hub", "s1", "s2", "s3"] {
            m.add_node(node(id, 0));
        }
        m.add_edge(edge("s1", "hub", 1.0));
        m.add_edge(edge("s2", "hub", 2.0));
        m.add_edge(edge("s3", "hub", 3.0));
        assert_eq!(m.in_degree("hub"), 3);
    }

    // ── 26. snapshot local_node_id ────────────────────────────────────────────

    #[test]
    fn test_snapshot_local_id() {
        let m = NetworkTopologyMapper::new("my-peer".to_string());
        assert_eq!(m.snapshot().local_node_id, "my-peer");
    }

    // ── 27. remove_node reduces node_count ────────────────────────────────────

    #[test]
    fn test_remove_node_reduces_count() {
        let mut m = linear_mapper();
        let before = m.node_count();
        m.remove_node("C");
        assert_eq!(m.node_count(), before - 1);
    }

    // ── 28. add_edge creates destination adjacency entry ─────────────────────

    #[test]
    fn test_add_edge_creates_destination_adjacency() {
        let mut m = NetworkTopologyMapper::new("s".to_string());
        m.add_edge(edge("s", "d", 1.0));
        assert!(m.adjacency.contains_key("d"));
    }

    // ── 29. clustering coefficient partial triangle ───────────────────────────

    #[test]
    fn test_clustering_partial_triangle() {
        let mut m = NetworkTopologyMapper::new("A".to_string());
        for id in ["A", "B", "C", "D"] {
            m.add_node(node(id, 0));
        }
        // Star: A → B, A → C, A → D (no edges between B, C, D).
        m.add_edge(edge("A", "B", 1.0));
        m.add_edge(edge("A", "C", 1.0));
        m.add_edge(edge("A", "D", 1.0));
        let cc = m.average_clustering_coefficient();
        // Only A has k ≥ 2 (k=3). No edges between its neighbours → cc(A)=0.
        assert!((cc - 0.0).abs() < 1e-9);
    }

    // ── 30. diameter with two separate reachable pairs ────────────────────────

    #[test]
    fn test_diameter_picks_max() {
        let mut m = NetworkTopologyMapper::new("A".to_string());
        for id in ["A", "B", "C"] {
            m.add_node(node(id, 0));
        }
        m.add_edge(edge("A", "B", 1.0));
        m.add_edge(edge("B", "C", 999.0));
        // A→B = 1, A→C = 1000, B→C = 999.  Max = 1000.
        let d = m.diameter().expect("diameter should exist");
        assert!((d - 1000.0).abs() < 1e-9);
    }

    // ── 31. stats component_count ─────────────────────────────────────────────

    #[test]
    fn test_stats_component_count() {
        let mut m = NetworkTopologyMapper::new("A".to_string());
        for id in ["A", "B", "C", "D"] {
            m.add_node(node(id, 0));
        }
        m.add_edge(edge("A", "B", 1.0));
        m.add_edge(edge("C", "D", 1.0));
        let s = m.stats();
        assert_eq!(s.component_count, 2);
    }

    // ── 32. TopoNode hop_distance field accessible ────────────────────────────

    #[test]
    fn test_topo_node_hop_distance() {
        let n = TopoNode {
            node_id: "peer".to_string(),
            address: "/ip4/1.2.3.4/tcp/1234".to_string(),
            discovered_at: 500,
            hop_distance: 7,
        };
        assert_eq!(n.hop_distance, 7);
    }
}
