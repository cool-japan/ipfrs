//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};
use thiserror::Error;

use super::functions::{best_candidate_index, canonical_key, default_link, reconstruct_path};

/// Edge entry stored in the adjacency list.
#[derive(Debug, Clone)]
pub(super) struct AdjEdge {
    pub(super) to: String,
    pub(super) latency_ms: u32,
    pub(super) bandwidth_bps: u64,
    pub(super) reliability: f64,
}
pub(super) struct UnionFind {
    pub(super) parent: HashMap<String, String>,
    pub(super) rank: HashMap<String, u32>,
}
impl UnionFind {
    pub(super) fn new(nodes: impl Iterator<Item = String>) -> Self {
        let mut parent = HashMap::new();
        let mut rank = HashMap::new();
        for n in nodes {
            parent.insert(n.clone(), n.clone());
            rank.insert(n, 0);
        }
        Self { parent, rank }
    }
    pub(super) fn find(&mut self, x: &str) -> String {
        if self.parent.get(x).map(|p| p.as_str()) == Some(x) {
            return x.to_owned();
        }
        let root = {
            let p = self.parent.get(x).cloned().unwrap_or_else(|| x.to_owned());
            self.find(&p)
        };
        if let Some(entry) = self.parent.get_mut(x) {
            *entry = root.clone();
        }
        root
    }
    pub(super) fn union(&mut self, a: &str, b: &str) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return;
        }
        let rank_a = self.rank.get(&ra).copied().unwrap_or(0);
        let rank_b = self.rank.get(&rb).copied().unwrap_or(0);
        match rank_a.cmp(&rank_b) {
            Ordering::Less => {
                self.parent.insert(ra, rb);
            }
            Ordering::Greater => {
                self.parent.insert(rb, ra);
            }
            Ordering::Equal => {
                self.parent.insert(rb, ra.clone());
                if let Some(r) = self.rank.get_mut(&ra) {
                    *r += 1;
                }
            }
        }
    }
}
/// Manages an in-memory virtual overlay network.
///
/// # Example
///
/// ```rust
/// use ipfrs_network::overlay_network_manager::{
///     OverlayNetworkManager, OverlayNode, OverlayLink, OverlayConfig, RoutingPolicy,
/// };
///
/// let cfg = OverlayConfig::default();
/// let mut mgr = OverlayNetworkManager::new(cfg);
///
/// let mut n1 = OverlayNode::new("a", "1.2.3.4:4001", "10.0.0.1", "us-east", 1000);
/// let mut n2 = OverlayNode::new("b", "1.2.3.5:4001", "10.0.0.2", "us-east", 1000);
/// mgr.add_node(n1).unwrap();
/// mgr.add_node(n2).unwrap();
///
/// let link = OverlayLink::new("a", "b", 10, 1_000_000);
/// mgr.add_link(link).unwrap();
///
/// let route = mgr.route("a", "b").unwrap();
/// assert_eq!(route.hops, vec!["a".to_owned(), "b".to_owned()]);
/// ```
pub struct OverlayNetworkManager {
    pub(super) config: OverlayConfig,
    /// All registered nodes, keyed by their ID.
    pub(super) nodes: HashMap<String, OverlayNode>,
    /// Adjacency list (undirected: each link is stored in both directions).
    pub(super) adj: HashMap<String, Vec<AdjEdge>>,
    /// Canonical link store for fast lookup (keyed by (from, to) canonical pair).
    pub(super) links: HashMap<(String, String), OverlayLink>,
}
impl OverlayNetworkManager {
    /// Create a new manager with the given configuration.
    pub fn new(config: OverlayConfig) -> Self {
        Self {
            config,
            nodes: HashMap::new(),
            adj: HashMap::new(),
            links: HashMap::new(),
        }
    }
    /// Add a node to the overlay.
    ///
    /// Fails if `max_nodes` would be exceeded.
    pub fn add_node(&mut self, node: OverlayNode) -> Result<(), OverlayError> {
        if self.nodes.len() >= self.config.max_nodes {
            return Err(OverlayError::MaxNodesExceeded);
        }
        let id = node.id.clone();
        self.nodes.insert(id.clone(), node);
        self.adj.entry(id).or_default();
        Ok(())
    }
    /// Remove a node and all of its incident links.
    ///
    /// Fails if the node is not found.
    pub fn remove_node(&mut self, id: &str) -> Result<(), OverlayError> {
        if !self.nodes.contains_key(id) {
            return Err(OverlayError::NodeNotFound(id.to_owned()));
        }
        let to_remove: Vec<(String, String)> = self
            .links
            .keys()
            .filter(|(a, b)| a == id || b == id)
            .cloned()
            .collect();
        for (a, b) in to_remove {
            self.links.remove(&(a.clone(), b.clone()));
            if let Some(edges) = self.adj.get_mut(&a) {
                edges.retain(|e| e.to != b);
            }
            if let Some(edges) = self.adj.get_mut(&b) {
                edges.retain(|e| e.to != a);
            }
        }
        self.adj.remove(id);
        self.nodes.remove(id);
        Ok(())
    }
    /// Add a bidirectional link between two existing nodes.
    ///
    /// Fails if either node is not found.
    pub fn add_link(&mut self, link: OverlayLink) -> Result<(), OverlayError> {
        if !self.nodes.contains_key(&link.from_id) {
            return Err(OverlayError::NodeNotFound(link.from_id.clone()));
        }
        if !self.nodes.contains_key(&link.to_id) {
            return Err(OverlayError::NodeNotFound(link.to_id.clone()));
        }
        let from = link.from_id.clone();
        let to = link.to_id.clone();
        let lat = link.latency_ms;
        let bw = link.bandwidth_bps;
        let rel = link.reliability;
        let key = canonical_key(&from, &to);
        self.links.insert(key, link);
        let edges_from = self.adj.entry(from.clone()).or_default();
        if let Some(e) = edges_from.iter_mut().find(|e| e.to == to) {
            e.latency_ms = lat;
            e.bandwidth_bps = bw;
            e.reliability = rel;
        } else {
            edges_from.push(AdjEdge {
                to: to.clone(),
                latency_ms: lat,
                bandwidth_bps: bw,
                reliability: rel,
            });
        }
        let edges_to = self.adj.entry(to.clone()).or_default();
        if let Some(e) = edges_to.iter_mut().find(|e| e.to == from) {
            e.latency_ms = lat;
            e.bandwidth_bps = bw;
            e.reliability = rel;
        } else {
            edges_to.push(AdjEdge {
                to: from,
                latency_ms: lat,
                bandwidth_bps: bw,
                reliability: rel,
            });
        }
        Ok(())
    }
    /// Remove the link between two nodes (fails if not found).
    pub fn remove_link(&mut self, from: &str, to: &str) -> Result<(), OverlayError> {
        let key = canonical_key(from, to);
        if self.links.remove(&key).is_none() {
            return Err(OverlayError::LinkNotFound {
                from: from.to_owned(),
                to: to.to_owned(),
            });
        }
        if let Some(edges) = self.adj.get_mut(from) {
            edges.retain(|e| e.to != to);
        }
        if let Some(edges) = self.adj.get_mut(to) {
            edges.retain(|e| e.to != from);
        }
        Ok(())
    }
    /// Compute the best route from `from` to `to` according to the configured
    /// routing policy.
    pub fn route(&self, from: &str, to: &str) -> Result<VirtualRoute, OverlayError> {
        self.route_with_policy(from, to, self.config.routing_policy)
    }
    /// Compute the best route using an explicitly supplied routing policy,
    /// ignoring nodes in `forbidden` (used internally by Yen's algorithm).
    pub(super) fn route_with_policy(
        &self,
        from: &str,
        to: &str,
        policy: RoutingPolicy,
    ) -> Result<VirtualRoute, OverlayError> {
        self.route_avoiding(from, to, policy, &HashSet::new(), &HashSet::new())
    }
    /// Core Dijkstra router with optional forbidden nodes / edges.
    pub(super) fn route_avoiding(
        &self,
        from: &str,
        to: &str,
        policy: RoutingPolicy,
        forbidden_nodes: &HashSet<String>,
        forbidden_edges: &HashSet<(String, String)>,
    ) -> Result<VirtualRoute, OverlayError> {
        if !self.nodes.contains_key(from) {
            return Err(OverlayError::NodeNotFound(from.to_owned()));
        }
        if !self.nodes.contains_key(to) {
            return Err(OverlayError::NodeNotFound(to.to_owned()));
        }
        if from == to {
            return Ok(VirtualRoute {
                hops: vec![from.to_owned()],
                total_latency_ms: 0,
                bottleneck_bandwidth_bps: u64::MAX,
                path_reliability: 1.0,
            });
        }
        match policy {
            RoutingPolicy::MaxBandwidth => {
                self.dijkstra_max_bandwidth(from, to, forbidden_nodes, forbidden_edges)
            }
            _ => self.dijkstra_min_cost(from, to, policy, forbidden_nodes, forbidden_edges),
        }
    }
    /// Dijkstra minimising cost (ShortestPath, MaxReliability, LoadBalanced,
    /// GeographicProximity).
    pub(super) fn dijkstra_min_cost(
        &self,
        from: &str,
        to: &str,
        policy: RoutingPolicy,
        forbidden_nodes: &HashSet<String>,
        forbidden_edges: &HashSet<(String, String)>,
    ) -> Result<VirtualRoute, OverlayError> {
        let mut dist: HashMap<String, f64> = HashMap::new();
        let mut prev: HashMap<String, String> = HashMap::new();
        let mut heap = BinaryHeap::new();
        dist.insert(from.to_owned(), 0.0);
        heap.push(DijkNode {
            cost: 0.0,
            id: from.to_owned(),
        });
        while let Some(DijkNode { cost, id }) = heap.pop() {
            if cost > *dist.get(&id).unwrap_or(&f64::INFINITY) {
                continue;
            }
            if id == to {
                break;
            }
            let edges = match self.adj.get(&id) {
                Some(e) => e,
                None => continue,
            };
            for edge in edges {
                if forbidden_nodes.contains(&edge.to) {
                    continue;
                }
                let fwd_edge = (id.clone(), edge.to.clone());
                let rev_edge = (edge.to.clone(), id.clone());
                if forbidden_edges.contains(&fwd_edge) || forbidden_edges.contains(&rev_edge) {
                    continue;
                }
                let dest_node = match self.nodes.get(&edge.to) {
                    Some(n) => n,
                    None => continue,
                };
                let step_cost = match policy {
                    RoutingPolicy::ShortestPath => 1.0,
                    RoutingPolicy::MaxReliability => {
                        let r = edge.reliability.max(1e-300);
                        -r.ln()
                    }
                    RoutingPolicy::LoadBalanced => {
                        let cap = dest_node.capacity.max(1) as f64;
                        (dest_node.load as f64 / cap) + 1.0
                    }
                    RoutingPolicy::GeographicProximity => {
                        let cur_region =
                            self.nodes.get(&id).map(|n| n.region.as_str()).unwrap_or("");
                        if dest_node.region == cur_region {
                            0.1
                        } else {
                            1.0
                        }
                    }
                    RoutingPolicy::MaxBandwidth => unreachable!(),
                };
                let new_cost = cost + step_cost;
                let best = dist.entry(edge.to.clone()).or_insert(f64::INFINITY);
                if new_cost < *best {
                    *best = new_cost;
                    prev.insert(edge.to.clone(), id.clone());
                    heap.push(DijkNode {
                        cost: new_cost,
                        id: edge.to.clone(),
                    });
                }
            }
        }
        if !dist.contains_key(to) {
            return Err(OverlayError::NoPathExists {
                from: from.to_owned(),
                to: to.to_owned(),
            });
        }
        let hops = reconstruct_path(&prev, from, to)?;
        Ok(self.build_route(&hops))
    }
    /// Dijkstra maximising bottleneck bandwidth (max-min path).
    pub(super) fn dijkstra_max_bandwidth(
        &self,
        from: &str,
        to: &str,
        forbidden_nodes: &HashSet<String>,
        forbidden_edges: &HashSet<(String, String)>,
    ) -> Result<VirtualRoute, OverlayError> {
        let mut best_bw: HashMap<String, f64> = HashMap::new();
        let mut prev: HashMap<String, String> = HashMap::new();
        let mut heap = BinaryHeap::new();
        let init_bw = f64::INFINITY;
        best_bw.insert(from.to_owned(), init_bw);
        heap.push(BwNode {
            bw: init_bw,
            id: from.to_owned(),
        });
        while let Some(BwNode { bw, id }) = heap.pop() {
            if bw < *best_bw.get(&id).unwrap_or(&0.0) {
                continue;
            }
            if id == to {
                break;
            }
            let edges = match self.adj.get(&id) {
                Some(e) => e,
                None => continue,
            };
            for edge in edges {
                if forbidden_nodes.contains(&edge.to) {
                    continue;
                }
                let fwd_edge = (id.clone(), edge.to.clone());
                let rev_edge = (edge.to.clone(), id.clone());
                if forbidden_edges.contains(&fwd_edge) || forbidden_edges.contains(&rev_edge) {
                    continue;
                }
                let new_bw = bw.min(edge.bandwidth_bps as f64);
                let best = best_bw.entry(edge.to.clone()).or_insert(0.0);
                if new_bw > *best {
                    *best = new_bw;
                    prev.insert(edge.to.clone(), id.clone());
                    heap.push(BwNode {
                        bw: new_bw,
                        id: edge.to.clone(),
                    });
                }
            }
        }
        if !best_bw.contains_key(to) {
            return Err(OverlayError::NoPathExists {
                from: from.to_owned(),
                to: to.to_owned(),
            });
        }
        let hops = reconstruct_path(&prev, from, to)?;
        Ok(self.build_route(&hops))
    }
    /// Build a [`VirtualRoute`] by summing link properties along `hops`.
    pub(super) fn build_route(&self, hops: &[String]) -> VirtualRoute {
        let mut total_latency_ms: u32 = 0;
        let mut bottleneck_bandwidth_bps: u64 = u64::MAX;
        let mut path_reliability: f64 = 1.0;
        for i in 0..hops.len().saturating_sub(1) {
            let a = &hops[i];
            let b = &hops[i + 1];
            let key = canonical_key(a, b);
            if let Some(link) = self.links.get(&key) {
                total_latency_ms = total_latency_ms.saturating_add(link.latency_ms);
                if link.bandwidth_bps < bottleneck_bandwidth_bps {
                    bottleneck_bandwidth_bps = link.bandwidth_bps;
                }
                path_reliability *= link.reliability;
            }
        }
        if hops.len() <= 1 {
            bottleneck_bandwidth_bps = u64::MAX;
        }
        VirtualRoute {
            hops: hops.to_vec(),
            total_latency_ms,
            bottleneck_bandwidth_bps,
            path_reliability,
        }
    }
    /// Return up to `max_k` distinct paths from `from` to `to` using Yen's
    /// algorithm with the configured routing policy.
    ///
    /// Returns at least the shortest path if one exists; fewer than `max_k`
    /// paths if the graph has fewer alternatives.
    pub fn all_routes(
        &self,
        from: &str,
        to: &str,
        max_k: usize,
    ) -> Result<Vec<VirtualRoute>, OverlayError> {
        self.all_routes_with_policy(from, to, max_k, self.config.routing_policy)
    }
    /// Like `all_routes` but with an explicit policy.
    pub fn all_routes_with_policy(
        &self,
        from: &str,
        to: &str,
        max_k: usize,
        policy: RoutingPolicy,
    ) -> Result<Vec<VirtualRoute>, OverlayError> {
        if max_k == 0 {
            return Ok(vec![]);
        }
        let mut k_paths: Vec<VirtualRoute> = Vec::with_capacity(max_k);
        let mut candidates: Vec<VirtualRoute> = Vec::new();
        let first = self.route_with_policy(from, to, policy)?;
        k_paths.push(first);
        while k_paths.len() < max_k {
            let last_path = match k_paths.last() {
                Some(p) => p.clone(),
                None => break,
            };
            for spur_idx in 0..last_path.hops.len().saturating_sub(1) {
                let spur_node = &last_path.hops[spur_idx];
                let root_path = &last_path.hops[..=spur_idx];
                let mut forbidden_edges: HashSet<(String, String)> = HashSet::new();
                for kp in &k_paths {
                    if kp.hops.len() > spur_idx
                        && kp.hops[..=spur_idx] == *root_path
                        && kp.hops.len() > spur_idx + 1
                    {
                        let a = kp.hops[spur_idx].clone();
                        let b = kp.hops[spur_idx + 1].clone();
                        forbidden_edges.insert((a.clone(), b.clone()));
                        forbidden_edges.insert((b, a));
                    }
                }
                for cp in &candidates {
                    if cp.hops.len() > spur_idx
                        && cp.hops[..=spur_idx] == *root_path
                        && cp.hops.len() > spur_idx + 1
                    {
                        let a = cp.hops[spur_idx].clone();
                        let b = cp.hops[spur_idx + 1].clone();
                        forbidden_edges.insert((a.clone(), b.clone()));
                        forbidden_edges.insert((b, a));
                    }
                }
                let forbidden_nodes: HashSet<String> =
                    root_path[..spur_idx].iter().cloned().collect();
                if let Ok(spur_route) =
                    self.route_avoiding(spur_node, to, policy, &forbidden_nodes, &forbidden_edges)
                {
                    let mut full_hops: Vec<String> = root_path.to_vec();
                    for h in spur_route.hops.iter().skip(1) {
                        full_hops.push(h.clone());
                    }
                    let candidate = self.build_route(&full_hops);
                    let is_new = !k_paths.iter().any(|p| p.hops == candidate.hops)
                        && !candidates.iter().any(|p| p.hops == candidate.hops);
                    if is_new {
                        candidates.push(candidate);
                    }
                }
            }
            if candidates.is_empty() {
                break;
            }
            let best_idx = best_candidate_index(&candidates);
            let next = candidates.remove(best_idx);
            k_paths.push(next);
        }
        Ok(k_paths)
    }
    /// Generate links for the given topology and add them to the manager.
    ///
    /// Returns the list of newly created links.
    /// For [`OverlayTopology::Custom`] this is a no-op (empty list).
    pub fn apply_topology(
        &mut self,
        topo: OverlayTopology,
    ) -> Result<Vec<OverlayLink>, OverlayError> {
        let node_ids: Vec<String> = self.nodes.keys().cloned().collect();
        let n = node_ids.len();
        let mut new_links: Vec<OverlayLink> = Vec::new();
        match &topo {
            OverlayTopology::FullMesh => {
                for i in 0..n {
                    for j in (i + 1)..n {
                        let link = default_link(&node_ids[i], &node_ids[j]);
                        new_links.push(link);
                    }
                }
            }
            OverlayTopology::Ring => {
                if n == 0 {
                    return Ok(new_links);
                }
                for i in 0..n {
                    let next = (i + 1) % n;
                    let link = default_link(&node_ids[i], &node_ids[next]);
                    new_links.push(link);
                }
            }
            OverlayTopology::Star { center_id } => {
                if !self.nodes.contains_key(center_id) {
                    return Err(OverlayError::TopologyError(format!(
                        "star centre node '{}' not found",
                        center_id
                    )));
                }
                for id in &node_ids {
                    if id != center_id {
                        let link = default_link(center_id, id);
                        new_links.push(link);
                    }
                }
            }
            OverlayTopology::Hypercube(dims) => {
                let dims = *dims as u32;
                let side = 1u32 << dims;
                if n < side as usize {
                    return Err(OverlayError::TopologyError(format!(
                        "hypercube({}) needs at least {} nodes, have {}",
                        dims, side, n
                    )));
                }
                for i in 0..side as usize {
                    for bit in 0..dims {
                        let j = i ^ (1 << bit);
                        if j > i {
                            let link = default_link(&node_ids[i], &node_ids[j]);
                            new_links.push(link);
                        }
                    }
                }
            }
            OverlayTopology::Custom => {
                return Ok(new_links);
            }
        }
        for link in &new_links {
            let _ = self.add_link(link.clone());
        }
        Ok(new_links)
    }
    /// Update the load value for a node.
    pub fn update_load(&mut self, node_id: &str, load: u32) -> Result<(), OverlayError> {
        let node = self
            .nodes
            .get_mut(node_id)
            .ok_or_else(|| OverlayError::NodeNotFound(node_id.to_owned()))?;
        node.load = load;
        Ok(())
    }
    /// Return references to all nodes marked as gateways.
    pub fn gateway_nodes(&self) -> Vec<&OverlayNode> {
        self.nodes.values().filter(|n| n.is_gateway).collect()
    }
    /// Compute connected components using union-find.
    ///
    /// Returns a list of components, each being a sorted list of node IDs.
    pub fn connected_components(&self) -> Vec<Vec<String>> {
        let mut uf = UnionFind::new(self.nodes.keys().cloned());
        for (a, b) in self.links.keys() {
            uf.union(a, b);
        }
        let mut components: HashMap<String, Vec<String>> = HashMap::new();
        for id in self.nodes.keys() {
            let root = uf.find(id);
            components.entry(root).or_default().push(id.clone());
        }
        let mut result: Vec<Vec<String>> = components.into_values().collect();
        for comp in &mut result {
            comp.sort();
        }
        result.sort_by(|a, b| b.len().cmp(&a.len()).then(a[0].cmp(&b[0])));
        result
    }
    /// Compute aggregate statistics for the current overlay state.
    ///
    /// BFS from every node to compute path lengths (O(V*(V+E))).
    pub fn stats(&self) -> OverlayStats {
        let node_ids: Vec<String> = self.nodes.keys().cloned().collect();
        let n = node_ids.len();
        if n == 0 {
            return OverlayStats {
                node_count: 0,
                link_count: self.links.len(),
                avg_path_length: 0.0,
                network_diameter: 0,
                connectivity: 0.0,
            };
        }
        if n == 1 {
            return OverlayStats {
                node_count: 1,
                link_count: self.links.len(),
                avg_path_length: 0.0,
                network_diameter: 0,
                connectivity: 1.0,
            };
        }
        let mut total_path_len: u64 = 0;
        let mut reachable_pairs: u64 = 0;
        let mut diameter: usize = 0;
        let total_pairs = (n as u64) * (n as u64 - 1);
        for src in &node_ids {
            let dist_map = self.bfs_distances(src);
            for dst in &node_ids {
                if src == dst {
                    continue;
                }
                if let Some(&d) = dist_map.get(dst) {
                    reachable_pairs += 1;
                    total_path_len += d as u64;
                    if d > diameter {
                        diameter = d;
                    }
                }
            }
        }
        let avg_path_length = if reachable_pairs == 0 {
            0.0
        } else {
            total_path_len as f64 / reachable_pairs as f64
        };
        let connectivity = if total_pairs == 0 {
            1.0
        } else {
            reachable_pairs as f64 / total_pairs as f64
        };
        OverlayStats {
            node_count: n,
            link_count: self.links.len(),
            avg_path_length,
            network_diameter: diameter,
            connectivity,
        }
    }
    /// BFS from `src`; returns map of node → hop distance.
    pub(super) fn bfs_distances(&self, src: &str) -> HashMap<String, usize> {
        let mut dist: HashMap<String, usize> = HashMap::new();
        let mut queue: VecDeque<String> = VecDeque::new();
        dist.insert(src.to_owned(), 0);
        queue.push_back(src.to_owned());
        while let Some(cur) = queue.pop_front() {
            let d = *dist.get(&cur).unwrap_or(&0);
            if let Some(edges) = self.adj.get(&cur) {
                for edge in edges {
                    if !dist.contains_key(&edge.to) {
                        dist.insert(edge.to.clone(), d + 1);
                        queue.push_back(edge.to.clone());
                    }
                }
            }
        }
        dist
    }
    /// Return the number of nodes currently registered.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
    /// Return the number of (canonical) links currently registered.
    pub fn link_count(&self) -> usize {
        self.links.len()
    }
    /// Return a reference to a node by ID, or `None`.
    pub fn get_node(&self, id: &str) -> Option<&OverlayNode> {
        self.nodes.get(id)
    }
}
/// Topology pattern for the virtual overlay network.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OverlayTopology {
    /// All nodes are directly connected to all other nodes.
    FullMesh,
    /// Nodes are arranged in a single ring.
    Ring,
    /// Hub-and-spoke: one centre node is connected to all others.
    Star {
        /// ID of the hub node.
        center_id: String,
    },
    /// Binary hypercube: each node connects to nodes differing by one bit.
    Hypercube(u8),
    /// User-defined topology; the manager does not generate links.
    Custom,
}
/// Errors returned by [`OverlayNetworkManager`].
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum OverlayError {
    /// A node with the given ID was not found.
    #[error("node not found: {0}")]
    NodeNotFound(String),
    /// No link exists between the two node IDs.
    #[error("link not found from {from} to {to}")]
    LinkNotFound {
        /// Source node ID.
        from: String,
        /// Destination node ID.
        to: String,
    },
    /// No path exists between the two node IDs.
    #[error("no path exists from {from} to {to}")]
    NoPathExists {
        /// Source node ID.
        from: String,
        /// Destination node ID.
        to: String,
    },
    /// A topology-level error (e.g. Star without a valid centre).
    #[error("topology error: {0}")]
    TopologyError(String),
    /// Adding a node would exceed `OverlayConfig::max_nodes`.
    #[error("maximum node count exceeded")]
    MaxNodesExceeded,
}
/// Policy used to select the optimal route.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoutingPolicy {
    /// Minimise the number of hops (BFS / unit-weight Dijkstra).
    ShortestPath,
    /// Maximise the bottleneck bandwidth (max-min bandwidth path).
    MaxBandwidth,
    /// Maximise the product of link reliabilities (min -ln(r) Dijkstra).
    MaxReliability,
    /// Prefer links to low-load nodes (weight = destination node's load).
    LoadBalanced,
    /// Prefer same-region hops over cross-region hops (weight = 0 same / 1 cross).
    GeographicProximity,
}
/// A computed virtual route through the overlay network.
#[derive(Debug, Clone, PartialEq)]
pub struct VirtualRoute {
    /// Ordered list of node IDs from source to destination (inclusive).
    pub hops: Vec<String>,
    /// Sum of per-link latencies along the path (ms).
    pub total_latency_ms: u32,
    /// Minimum bandwidth on any link in the path (bps) — the bottleneck.
    pub bottleneck_bandwidth_bps: u64,
    /// Product of per-link reliabilities (probability the whole path succeeds).
    pub path_reliability: f64,
}
impl VirtualRoute {
    /// Number of inter-node edges (hops) in this route.
    pub fn hop_count(&self) -> usize {
        self.hops.len().saturating_sub(1)
    }
}
/// A directional (but logically bidirectional) link between two overlay nodes.
#[derive(Debug, Clone, PartialEq)]
pub struct OverlayLink {
    /// Source node ID.
    pub from_id: String,
    /// Destination node ID.
    pub to_id: String,
    /// One-way latency in milliseconds.
    pub latency_ms: u32,
    /// Bandwidth in bits per second.
    pub bandwidth_bps: u64,
    /// Probability that the link delivers a packet (in [0, 1]).
    pub reliability: f64,
    /// Whether this is a tunnelled (encrypted) link.
    pub is_tunnel: bool,
}
impl OverlayLink {
    /// Create a new overlay link with full reliability and no tunnelling.
    pub fn new(
        from_id: impl Into<String>,
        to_id: impl Into<String>,
        latency_ms: u32,
        bandwidth_bps: u64,
    ) -> Self {
        Self {
            from_id: from_id.into(),
            to_id: to_id.into(),
            latency_ms,
            bandwidth_bps,
            reliability: 1.0,
            is_tunnel: false,
        }
    }
}
/// A participant in the virtual overlay network.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OverlayNode {
    /// Unique string identifier for this node (e.g. peer ID or hostname).
    pub id: String,
    /// Physical address (IP:port, multiaddr, …).
    pub physical_address: String,
    /// Virtual address assigned by the overlay (e.g. "10.0.1.5").
    pub overlay_address: String,
    /// Logical region label (e.g. "us-east", "eu-west").
    pub region: String,
    /// Maximum traffic capacity this node can handle (arbitrary units).
    pub capacity: u32,
    /// Current load on this node (arbitrary units, ≤ capacity).
    pub load: u32,
    /// Whether this node acts as a gateway between regions.
    pub is_gateway: bool,
}
impl OverlayNode {
    /// Create a new overlay node with sensible defaults for load/gateway.
    pub fn new(
        id: impl Into<String>,
        physical_address: impl Into<String>,
        overlay_address: impl Into<String>,
        region: impl Into<String>,
        capacity: u32,
    ) -> Self {
        Self {
            id: id.into(),
            physical_address: physical_address.into(),
            overlay_address: overlay_address.into(),
            region: region.into(),
            capacity,
            load: 0,
            is_gateway: false,
        }
    }
}
/// Generic cost node used in Dijkstra's algorithm.
/// We store an `f64` cost (lower is better) and use a min-heap.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct DijkNode {
    pub(super) cost: f64,
    pub(super) id: String,
}
#[derive(Debug, Clone, PartialEq)]
pub(super) struct BwNode {
    pub(super) bw: f64,
    pub(super) id: String,
}
/// Configuration for an [`OverlayNetworkManager`].
#[derive(Debug, Clone)]
pub struct OverlayConfig {
    /// Which topology pattern to use.
    pub topology: OverlayTopology,
    /// Upper bound on the number of nodes.
    pub max_nodes: usize,
    /// Maximum number of hops a single route may traverse.
    pub max_hops: u8,
    /// Default routing policy.
    pub routing_policy: RoutingPolicy,
    /// Whether to prefer same-region links.
    pub region_aware: bool,
}
/// Aggregate statistics for the current overlay topology.
#[derive(Debug, Clone, PartialEq)]
pub struct OverlayStats {
    /// Number of nodes in the overlay.
    pub node_count: usize,
    /// Number of (directed) links in the overlay.
    pub link_count: usize,
    /// Average length (in hops) of shortest paths between all connected pairs.
    pub avg_path_length: f64,
    /// Longest shortest path between any pair of connected nodes.
    pub network_diameter: usize,
    /// Fraction of ordered node pairs (u, v) for which a path exists.
    pub connectivity: f64,
}
