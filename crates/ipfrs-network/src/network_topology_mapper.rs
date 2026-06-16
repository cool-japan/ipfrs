//! Network Topology Mapper
//!
//! A live, production-quality topology mapper that tracks peer connections and
//! infers structural properties of the overlay network.
//!
//! # Features
//!
//! - Fixed-size `[u8; 32]` node identifiers and `u64` edge identifiers.
//! - Directed graph with per-edge latency and bandwidth annotations.
//! - Dijkstra shortest-path (minimising latency), BFS diameter, clustering
//!   coefficient, and Brandes betweenness centrality.
//! - Bounded snapshot history (max 20 entries, oldest dropped).
//! - Stale-entry pruning via configurable TTL.
//! - No external crates beyond those already in `ipfrs-network`.
//! - Zero `unwrap()` calls throughout.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};

// ─── Type Aliases ─────────────────────────────────────────────────────────────

/// Opaque 32-byte node identifier.
pub type NtmNodeId = [u8; 32];

/// Opaque 64-bit edge identifier (derived from src XOR dst XOR counter).
pub type NtmEdgeId = u64;

/// Convenient top-level alias for the mapper itself.
pub type NtmNetworkTopologyMapper = NetworkTopologyMapper;

// ─── PRNG helpers (pure inline, no external crates) ──────────────────────────

#[inline]
fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

#[inline]
fn fnv1a_64(data: &[u8]) -> u64 {
    let mut h: u64 = 14_695_981_039_346_656_037;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(1_099_511_628_211);
    }
    h
}

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for `NetworkTopologyMapper`.
#[derive(Debug, Clone)]
pub struct NtmMapperConfig {
    /// Maximum number of nodes the graph may hold.
    pub max_nodes: usize,
    /// Maximum number of edges the graph may hold.
    pub max_edges: usize,
    /// Minimum seconds between automatic snapshots (0 = never auto-snapshot).
    pub snapshot_interval_secs: u64,
    /// Seconds after which a node that has not been seen is pruned.
    pub prune_disconnected_after_secs: u64,
}

impl Default for NtmMapperConfig {
    fn default() -> Self {
        Self {
            max_nodes: 4_096,
            max_edges: 65_536,
            snapshot_interval_secs: 60,
            prune_disconnected_after_secs: 300,
        }
    }
}

// ─── Error type ───────────────────────────────────────────────────────────────

/// Errors produced by `NetworkTopologyMapper` operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NtmMapperError {
    /// Node or edge not found.
    NotFound(String),
    /// The graph is at capacity.
    CapacityExceeded(String),
    /// A duplicate key was inserted.
    Duplicate(String),
    /// Generic internal error.
    Internal(String),
}

impl std::fmt::Display for NtmMapperError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound(s) => write!(f, "not found: {s}"),
            Self::CapacityExceeded(s) => write!(f, "capacity exceeded: {s}"),
            Self::Duplicate(s) => write!(f, "duplicate: {s}"),
            Self::Internal(s) => write!(f, "internal error: {s}"),
        }
    }
}

impl std::error::Error for NtmMapperError {}

// ─── Core graph data types ────────────────────────────────────────────────────

/// A peer node in the topology graph.
#[derive(Debug, Clone)]
pub struct NtmNode {
    /// 32-byte opaque node identifier.
    pub id: NtmNodeId,
    /// Human-readable network address (multiaddr or socket string).
    pub addr: String,
    /// Optional geographic region hint.
    pub region: Option<String>,
    /// Latest round-trip time in milliseconds.
    pub rtt_ms: f64,
    /// Unix-epoch seconds when this node was last observed alive.
    pub last_seen: u64,
    /// Number of neighbours (undirected degree = out-degree + in-degree).
    pub degree: u32,
    /// Whether this node is a bootstrap / well-known anchor.
    pub is_bootstrap: bool,
}

/// A directed edge between two peers.
#[derive(Debug, Clone)]
pub struct NtmEdge {
    /// Unique edge identifier.
    pub id: NtmEdgeId,
    /// Source node identifier.
    pub src: NtmNodeId,
    /// Destination node identifier.
    pub dst: NtmNodeId,
    /// One-way (or round-trip) latency estimate in milliseconds.
    pub latency_ms: f64,
    /// Measured bandwidth in kbps.
    pub bandwidth_kbps: f64,
    /// Unix-epoch seconds when this edge was last observed.
    pub observed_at: u64,
}

/// Point-in-time snapshot of key topology metrics.
#[derive(Debug, Clone)]
pub struct NtmSnapshot {
    /// Unix-epoch seconds when the snapshot was taken.
    pub ts: u64,
    /// Number of nodes at snapshot time.
    pub node_count: usize,
    /// Number of edges at snapshot time.
    pub edge_count: usize,
    /// Average node degree at snapshot time.
    pub avg_degree: f64,
    /// Graph diameter (longest shortest path) at snapshot time.
    pub diameter: u32,
    /// Average clustering coefficient at snapshot time.
    pub clustering_coeff: f64,
}

/// Derived topology metrics computed on demand.
#[derive(Debug, Clone)]
pub struct NtmTopologyMetrics {
    /// Graph density: `|E| / (|V| * (|V| - 1))`.
    pub density: f64,
    /// Average of all pairwise shortest-path lengths.
    pub avg_path_length: f64,
    /// Betweenness centrality per node (Brandes algorithm).
    pub betweenness: HashMap<NtmNodeId, f64>,
    /// Degree centrality per node: `degree / (|V| - 1)`.
    pub centrality: HashMap<NtmNodeId, f64>,
}

// ─── Internal adjacency helpers ───────────────────────────────────────────────

/// Compact ordered pair used as the key for the adjacency set.
type EdgeKey = (NtmNodeId, NtmNodeId);

// ─── Main struct ──────────────────────────────────────────────────────────────

/// Live network topology mapper.
///
/// Tracks peer nodes and directed edges, computes graph algorithms, and
/// maintains a bounded ring-buffer of snapshots.
pub struct NetworkTopologyMapper {
    /// All known nodes keyed by their 32-byte id.
    nodes: HashMap<NtmNodeId, NtmNode>,
    /// All known edges keyed by their `NtmEdgeId`.
    edges: HashMap<NtmEdgeId, NtmEdge>,
    /// Adjacency index: `(src, dst) → edge_id` for fast lookup.
    adj: HashMap<EdgeKey, NtmEdgeId>,
    /// Out-edge list: `src → Vec<edge_id>` for fast neighbour iteration.
    out_edges: HashMap<NtmNodeId, Vec<NtmEdgeId>>,
    /// In-edge list: `dst → Vec<edge_id>` for efficient reverse traversal.
    in_edges: HashMap<NtmNodeId, Vec<NtmEdgeId>>,
    /// Bounded snapshot ring-buffer (capacity = 20).
    snapshots: VecDeque<NtmSnapshot>,
    /// Mapper configuration.
    config: NtmMapperConfig,
    /// Monotonic PRNG state for edge-id generation.
    prng_state: u64,
    /// Edge counter used to salt edge-id hashing.
    edge_counter: u64,
}

impl NetworkTopologyMapper {
    // ── Construction ──────────────────────────────────────────────────────────

    /// Create a new mapper with the given `config`.
    pub fn new(config: NtmMapperConfig) -> Self {
        // Seed the PRNG with the FNV-1a hash of a fixed string so it is
        // deterministic yet non-trivial.
        let seed = fnv1a_64(b"NetworkTopologyMapper:v1");
        Self {
            nodes: HashMap::new(),
            edges: HashMap::new(),
            adj: HashMap::new(),
            out_edges: HashMap::new(),
            in_edges: HashMap::new(),
            snapshots: VecDeque::with_capacity(20),
            config,
            prng_state: seed | 1, // ensure non-zero
            edge_counter: 0,
        }
    }

    /// Create a mapper with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(NtmMapperConfig::default())
    }

    // ── Node operations ───────────────────────────────────────────────────────

    /// Insert or update a node.
    ///
    /// If the node already exists its `addr`, `region`, `rtt_ms`, and
    /// `last_seen` fields are refreshed.  Returns `Err(CapacityExceeded)`
    /// when the node is new and the graph is full.
    pub fn add_node(
        &mut self,
        id: NtmNodeId,
        addr: impl Into<String>,
        region: Option<String>,
        rtt_ms: f64,
        last_seen: u64,
    ) -> Result<(), NtmMapperError> {
        if let Some(existing) = self.nodes.get_mut(&id) {
            existing.addr = addr.into();
            existing.region = region;
            existing.rtt_ms = rtt_ms;
            existing.last_seen = last_seen;
            return Ok(());
        }
        if self.nodes.len() >= self.config.max_nodes {
            return Err(NtmMapperError::CapacityExceeded(format!(
                "node limit {} reached",
                self.config.max_nodes
            )));
        }
        self.nodes.insert(
            id,
            NtmNode {
                id,
                addr: addr.into(),
                region,
                rtt_ms,
                last_seen,
                degree: 0,
                is_bootstrap: false,
            },
        );
        Ok(())
    }

    /// Remove a node and all edges incident to it.
    pub fn remove_node(&mut self, id: &NtmNodeId) -> Result<(), NtmMapperError> {
        if !self.nodes.contains_key(id) {
            return Err(NtmMapperError::NotFound(format!("{id:?}")));
        }
        // Collect all edge ids touching this node before mutating.
        let mut to_remove: Vec<NtmEdgeId> = Vec::new();
        if let Some(outs) = self.out_edges.get(id) {
            to_remove.extend_from_slice(outs);
        }
        if let Some(ins) = self.in_edges.get(id) {
            to_remove.extend_from_slice(ins);
        }
        for eid in to_remove {
            let _ = self.remove_edge(eid);
        }
        self.out_edges.remove(id);
        self.in_edges.remove(id);
        self.nodes.remove(id);
        Ok(())
    }

    /// Update the RTT measurement for an existing node.
    pub fn update_node_rtt(&mut self, id: &NtmNodeId, rtt_ms: f64) -> Result<(), NtmMapperError> {
        let node = self
            .nodes
            .get_mut(id)
            .ok_or_else(|| NtmMapperError::NotFound(format!("{id:?}")))?;
        node.rtt_ms = rtt_ms;
        Ok(())
    }

    /// Mark a node as a bootstrap / well-known anchor.
    pub fn set_bootstrap(&mut self, id: &NtmNodeId, flag: bool) -> Result<(), NtmMapperError> {
        let node = self
            .nodes
            .get_mut(id)
            .ok_or_else(|| NtmMapperError::NotFound(format!("{id:?}")))?;
        node.is_bootstrap = flag;
        Ok(())
    }

    /// Return an immutable reference to a node.
    pub fn get_node(&self, id: &NtmNodeId) -> Option<&NtmNode> {
        self.nodes.get(id)
    }

    /// Iterate over all nodes.
    pub fn nodes(&self) -> impl Iterator<Item = &NtmNode> {
        self.nodes.values()
    }

    // ── Edge operations ───────────────────────────────────────────────────────

    /// Insert a directed edge `src → dst`.
    ///
    /// If an edge with the same `(src, dst)` pair already exists, its
    /// `latency_ms`, `bandwidth_kbps`, and `observed_at` fields are updated.
    /// Returns the edge identifier.
    pub fn add_edge(
        &mut self,
        src: NtmNodeId,
        dst: NtmNodeId,
        latency_ms: f64,
        bandwidth_kbps: f64,
        observed_at: u64,
    ) -> Result<NtmEdgeId, NtmMapperError> {
        if !self.nodes.contains_key(&src) {
            return Err(NtmMapperError::NotFound(format!("src {:?}", src)));
        }
        if !self.nodes.contains_key(&dst) {
            return Err(NtmMapperError::NotFound(format!("dst {:?}", dst)));
        }

        // Update existing edge.
        if let Some(&eid) = self.adj.get(&(src, dst)) {
            if let Some(edge) = self.edges.get_mut(&eid) {
                edge.latency_ms = latency_ms;
                edge.bandwidth_kbps = bandwidth_kbps;
                edge.observed_at = observed_at;
            }
            return Ok(eid);
        }

        if self.edges.len() >= self.config.max_edges {
            return Err(NtmMapperError::CapacityExceeded(format!(
                "edge limit {} reached",
                self.config.max_edges
            )));
        }

        let eid = self.gen_edge_id(&src, &dst);
        let edge = NtmEdge {
            id: eid,
            src,
            dst,
            latency_ms,
            bandwidth_kbps,
            observed_at,
        };
        self.edges.insert(eid, edge);
        self.adj.insert((src, dst), eid);
        self.out_edges.entry(src).or_default().push(eid);
        self.in_edges.entry(dst).or_default().push(eid);

        // Update degrees.
        self.recompute_degree(&src);
        self.recompute_degree(&dst);

        Ok(eid)
    }

    /// Remove a directed edge by its identifier.
    pub fn remove_edge(&mut self, id: NtmEdgeId) -> Result<(), NtmMapperError> {
        let edge = self
            .edges
            .remove(&id)
            .ok_or_else(|| NtmMapperError::NotFound(format!("edge {id}")))?;
        self.adj.remove(&(edge.src, edge.dst));
        if let Some(list) = self.out_edges.get_mut(&edge.src) {
            list.retain(|&e| e != id);
        }
        if let Some(list) = self.in_edges.get_mut(&edge.dst) {
            list.retain(|&e| e != id);
        }
        self.recompute_degree(&edge.src);
        self.recompute_degree(&edge.dst);
        Ok(())
    }

    /// Update the latency measurement for an existing edge.
    pub fn update_edge_latency(
        &mut self,
        id: NtmEdgeId,
        latency_ms: f64,
    ) -> Result<(), NtmMapperError> {
        let edge = self
            .edges
            .get_mut(&id)
            .ok_or_else(|| NtmMapperError::NotFound(format!("edge {id}")))?;
        edge.latency_ms = latency_ms;
        Ok(())
    }

    /// Return an immutable reference to an edge.
    pub fn get_edge(&self, id: NtmEdgeId) -> Option<&NtmEdge> {
        self.edges.get(&id)
    }

    /// Return the edge identifier for the directed pair `(src, dst)`, if any.
    pub fn find_edge(&self, src: &NtmNodeId, dst: &NtmNodeId) -> Option<NtmEdgeId> {
        self.adj.get(&(*src, *dst)).copied()
    }

    /// Iterate over all edges.
    pub fn edges(&self) -> impl Iterator<Item = &NtmEdge> {
        self.edges.values()
    }

    // ── Graph algorithms ──────────────────────────────────────────────────────

    /// Return the neighbour node ids reachable via outgoing edges from `node_id`.
    pub fn neighbors(&self, node_id: &NtmNodeId) -> Vec<NtmNodeId> {
        match self.out_edges.get(node_id) {
            None => Vec::new(),
            Some(eids) => eids
                .iter()
                .filter_map(|eid| self.edges.get(eid).map(|e| e.dst))
                .collect(),
        }
    }

    /// Dijkstra shortest path from `src` to `dst` weighted by `latency_ms`.
    ///
    /// Returns `None` if either node is unknown or no path exists.
    pub fn shortest_path(&self, src: &NtmNodeId, dst: &NtmNodeId) -> Option<Vec<NtmNodeId>> {
        if !self.nodes.contains_key(src) || !self.nodes.contains_key(dst) {
            return None;
        }
        if src == dst {
            return Some(vec![*src]);
        }

        // dist stored as ordered bits of f64 (positive, so bit-cast works for
        // ordering): we use `u64` distances in the heap via `f64::to_bits`.
        let mut dist: HashMap<NtmNodeId, f64> = HashMap::new();
        let mut prev: HashMap<NtmNodeId, NtmNodeId> = HashMap::new();
        // BinaryHeap<Reverse<(dist_bits, node_id)>>
        let mut heap: BinaryHeap<Reverse<(u64, NtmNodeId)>> = BinaryHeap::new();

        dist.insert(*src, 0.0);
        heap.push(Reverse((0u64, *src)));

        while let Some(Reverse((d_bits, u))) = heap.pop() {
            let d = f64::from_bits(d_bits);
            // Skip stale entries.
            if let Some(&best) = dist.get(&u) {
                if d > best + f64::EPSILON {
                    continue;
                }
            }
            if &u == dst {
                // Reconstruct path.
                let mut path = vec![u];
                let mut cur = u;
                while let Some(&p) = prev.get(&cur) {
                    path.push(p);
                    cur = p;
                }
                path.reverse();
                return Some(path);
            }
            if let Some(eids) = self.out_edges.get(&u) {
                for &eid in eids {
                    if let Some(edge) = self.edges.get(&eid) {
                        let new_d = d + edge.latency_ms.max(0.0);
                        let better = match dist.get(&edge.dst) {
                            None => true,
                            Some(&old) => new_d < old - f64::EPSILON,
                        };
                        if better {
                            dist.insert(edge.dst, new_d);
                            prev.insert(edge.dst, u);
                            heap.push(Reverse((new_d.to_bits(), edge.dst)));
                        }
                    }
                }
            }
        }
        None
    }

    /// BFS-based shortest path counting hops (ignores weights).
    ///
    /// Returns `None` if no path exists.
    pub fn bfs_distance(&self, src: &NtmNodeId, dst: &NtmNodeId) -> Option<u32> {
        if !self.nodes.contains_key(src) || !self.nodes.contains_key(dst) {
            return None;
        }
        if src == dst {
            return Some(0);
        }
        let mut visited: HashSet<NtmNodeId> = HashSet::new();
        let mut queue: VecDeque<(NtmNodeId, u32)> = VecDeque::new();
        queue.push_back((*src, 0));
        visited.insert(*src);
        while let Some((cur, d)) = queue.pop_front() {
            for nb in self.neighbors(&cur) {
                if &nb == dst {
                    return Some(d + 1);
                }
                if visited.insert(nb) {
                    queue.push_back((nb, d + 1));
                }
            }
        }
        None
    }

    /// Graph diameter: maximum of all pairwise BFS distances.
    ///
    /// Returns 0 when the graph has fewer than two nodes.
    pub fn diameter(&self) -> u32 {
        let ids: Vec<NtmNodeId> = self.nodes.keys().copied().collect();
        if ids.len() < 2 {
            return 0;
        }
        let mut max_d = 0u32;
        for &start in &ids {
            // BFS from start.
            let mut dist: HashMap<NtmNodeId, u32> = HashMap::new();
            let mut queue: VecDeque<NtmNodeId> = VecDeque::new();
            dist.insert(start, 0);
            queue.push_back(start);
            while let Some(cur) = queue.pop_front() {
                let d = dist[&cur];
                for nb in self.neighbors(&cur) {
                    if let std::collections::hash_map::Entry::Vacant(e) = dist.entry(nb) {
                        e.insert(d + 1);
                        queue.push_back(nb);
                    }
                }
            }
            for &v in dist.values() {
                if v > max_d {
                    max_d = v;
                }
            }
        }
        max_d
    }

    /// Local clustering coefficient for `node_id`.
    ///
    /// `C(u) = triangles / (k * (k-1))` where `k = degree(u)`.
    /// Returns `0.0` for nodes with degree < 2.
    pub fn clustering_coefficient(&self, node_id: &NtmNodeId) -> f64 {
        let nbs: Vec<NtmNodeId> = self.neighbors(node_id);
        let k = nbs.len();
        if k < 2 {
            return 0.0;
        }
        let mut triangles = 0u64;
        for &nb in &nbs {
            for &nb2 in &nbs {
                if nb == nb2 {
                    continue;
                }
                if self.adj.contains_key(&(nb, nb2)) {
                    triangles += 1;
                }
            }
        }
        // Each triangle counted twice (both orderings of the pair).
        triangles as f64 / (k as f64 * (k as f64 - 1.0))
    }

    /// Brandes algorithm for betweenness centrality on the directed graph.
    ///
    /// Complexity O(V * E) — suitable for moderate-sized graphs (≤ 4096 nodes).
    pub fn compute_betweenness_centrality(&self) -> HashMap<NtmNodeId, f64> {
        let mut cb: HashMap<NtmNodeId, f64> = self.nodes.keys().map(|&id| (id, 0.0)).collect();

        let all_nodes: Vec<NtmNodeId> = self.nodes.keys().copied().collect();

        for &s in &all_nodes {
            // BFS from s.
            let mut stack: Vec<NtmNodeId> = Vec::new();
            let mut pred: HashMap<NtmNodeId, Vec<NtmNodeId>> = HashMap::new();
            let mut sigma: HashMap<NtmNodeId, f64> = HashMap::new();
            let mut dist_map: HashMap<NtmNodeId, i64> = HashMap::new();

            for &v in &all_nodes {
                pred.insert(v, Vec::new());
                sigma.insert(v, 0.0);
                dist_map.insert(v, -1);
            }
            if let Some(sig) = sigma.get_mut(&s) {
                *sig = 1.0;
            }
            if let Some(d) = dist_map.get_mut(&s) {
                *d = 0;
            }

            let mut queue: VecDeque<NtmNodeId> = VecDeque::new();
            queue.push_back(s);

            while let Some(v) = queue.pop_front() {
                stack.push(v);
                let dv = dist_map[&v];
                let sig_v = sigma[&v];
                if let Some(eids) = self.out_edges.get(&v) {
                    for &eid in eids {
                        if let Some(edge) = self.edges.get(&eid) {
                            let w = edge.dst;
                            if dist_map[&w] < 0 {
                                queue.push_back(w);
                                if let Some(d) = dist_map.get_mut(&w) {
                                    *d = dv + 1;
                                }
                            }
                            if dist_map[&w] == dv + 1 {
                                if let Some(sig) = sigma.get_mut(&w) {
                                    *sig += sig_v;
                                }
                                if let Some(p) = pred.get_mut(&w) {
                                    p.push(v);
                                }
                            }
                        }
                    }
                }
            }

            let mut delta: HashMap<NtmNodeId, f64> = all_nodes.iter().map(|&v| (v, 0.0)).collect();

            while let Some(w) = stack.pop() {
                let sig_w = sigma[&w];
                let delta_w = delta[&w];
                let preds = pred[&w].clone();
                for v in preds {
                    let coeff = (sigma[&v] / sig_w) * (1.0 + delta_w);
                    if let Some(d) = delta.get_mut(&v) {
                        *d += coeff;
                    }
                }
                if w != s {
                    if let Some(c) = cb.get_mut(&w) {
                        *c += delta_w;
                    }
                }
            }
        }

        // Normalise by (V-1)(V-2) for directed graphs.
        let n = all_nodes.len() as f64;
        if n > 2.0 {
            let norm = (n - 1.0) * (n - 2.0);
            for v in cb.values_mut() {
                *v /= norm;
            }
        }

        cb
    }

    // ── Snapshot ──────────────────────────────────────────────────────────────

    /// Capture the current state into a snapshot and store it.
    ///
    /// If the snapshot buffer is full (20 entries) the oldest entry is
    /// dropped before the new one is inserted.
    pub fn take_snapshot(&mut self, ts: u64) -> NtmSnapshot {
        let node_count = self.nodes.len();
        let edge_count = self.edges.len();
        let avg_degree = if node_count == 0 {
            0.0
        } else {
            self.nodes.values().map(|n| n.degree as f64).sum::<f64>() / node_count as f64
        };
        let diameter = self.diameter();
        let clustering_coeff = if node_count == 0 {
            0.0
        } else {
            let sum: f64 = self
                .nodes
                .keys()
                .map(|id| self.clustering_coefficient(id))
                .sum();
            sum / node_count as f64
        };
        let snap = NtmSnapshot {
            ts,
            node_count,
            edge_count,
            avg_degree,
            diameter,
            clustering_coeff,
        };
        if self.snapshots.len() == 20 {
            self.snapshots.pop_front();
        }
        self.snapshots.push_back(snap.clone());
        snap
    }

    /// Access the snapshot history (oldest first).
    pub fn snapshots(&self) -> &VecDeque<NtmSnapshot> {
        &self.snapshots
    }

    // ── Pruning ───────────────────────────────────────────────────────────────

    /// Remove all nodes (and their edges) whose `last_seen` is older than
    /// `now_ts - config.prune_disconnected_after_secs`.
    pub fn prune_stale(&mut self, now_ts: u64) {
        let threshold = now_ts.saturating_sub(self.config.prune_disconnected_after_secs);
        let stale: Vec<NtmNodeId> = self
            .nodes
            .values()
            .filter(|n| n.last_seen < threshold)
            .map(|n| n.id)
            .collect();
        for id in stale {
            let _ = self.remove_node(&id);
        }
    }

    // ── Derived metrics ───────────────────────────────────────────────────────

    /// Compute the full suite of topology metrics.
    pub fn topology_stats(&self) -> NtmTopologyMetrics {
        let n = self.nodes.len();
        let e = self.edges.len();

        let density = if n > 1 {
            e as f64 / (n as f64 * (n as f64 - 1.0))
        } else {
            0.0
        };

        let avg_path_length = self.compute_avg_path_length();
        let betweenness = self.compute_betweenness_centrality();

        let centrality: HashMap<NtmNodeId, f64> = if n > 1 {
            self.nodes
                .values()
                .map(|node| (node.id, node.degree as f64 / (n as f64 - 1.0)))
                .collect()
        } else {
            self.nodes.values().map(|n| (n.id, 0.0)).collect()
        };

        NtmTopologyMetrics {
            density,
            avg_path_length,
            betweenness,
            centrality,
        }
    }

    // ── Accessors ─────────────────────────────────────────────────────────────

    /// Return the number of nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Return the number of edges.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Return the current configuration.
    pub fn config(&self) -> &NtmMapperConfig {
        &self.config
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /// Generate a collision-resistant edge identifier.
    fn gen_edge_id(&mut self, src: &NtmNodeId, dst: &NtmNodeId) -> NtmEdgeId {
        self.edge_counter += 1;
        // Mix src bytes, dst bytes, and a fresh PRNG value.
        let src_h = fnv1a_64(src);
        let dst_h = fnv1a_64(dst);
        let rnd = xorshift64(&mut self.prng_state);
        src_h ^ dst_h.rotate_left(31) ^ rnd ^ self.edge_counter.wrapping_mul(0x9e3779b97f4a7c15)
    }

    /// Recompute the `degree` field (out-degree + in-degree) for `id`.
    fn recompute_degree(&mut self, id: &NtmNodeId) {
        let out = self.out_edges.get(id).map(|v| v.len()).unwrap_or(0);
        let inc = self.in_edges.get(id).map(|v| v.len()).unwrap_or(0);
        if let Some(node) = self.nodes.get_mut(id) {
            node.degree = (out + inc) as u32;
        }
    }

    /// Average pairwise shortest-path length via BFS (unweighted).
    fn compute_avg_path_length(&self) -> f64 {
        let ids: Vec<NtmNodeId> = self.nodes.keys().copied().collect();
        let n = ids.len();
        if n < 2 {
            return 0.0;
        }
        let mut total = 0u64;
        let mut pairs = 0u64;
        for &s in &ids {
            // BFS from s.
            let mut dist: HashMap<NtmNodeId, u32> = HashMap::new();
            let mut queue: VecDeque<NtmNodeId> = VecDeque::new();
            dist.insert(s, 0);
            queue.push_back(s);
            while let Some(cur) = queue.pop_front() {
                let d = dist[&cur];
                for nb in self.neighbors(&cur) {
                    if let std::collections::hash_map::Entry::Vacant(e) = dist.entry(nb) {
                        e.insert(d + 1);
                        queue.push_back(nb);
                    }
                }
            }
            for (&v, &d) in &dist {
                if v != s {
                    total += d as u64;
                    pairs += 1;
                }
            }
        }
        if pairs == 0 {
            0.0
        } else {
            total as f64 / pairs as f64
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ───────────────────────────────────────────────────────────────

    fn make_id(v: u8) -> NtmNodeId {
        let mut id = [0u8; 32];
        id[0] = v;
        id
    }

    fn make_mapper() -> NetworkTopologyMapper {
        NetworkTopologyMapper::with_defaults()
    }

    fn add_n(m: &mut NetworkTopologyMapper, v: u8) {
        m.add_node(make_id(v), format!("127.0.0.{v}:4001"), None, 1.0, 100)
            .expect("add_node failed");
    }

    fn add_e(m: &mut NetworkTopologyMapper, u: u8, v: u8, lat: f64) -> NtmEdgeId {
        m.add_edge(make_id(u), make_id(v), lat, 1000.0, 100)
            .expect("add_edge failed")
    }

    // ── Config ────────────────────────────────────────────────────────────────

    #[test]
    fn test_config_defaults() {
        let c = NtmMapperConfig::default();
        assert_eq!(c.max_nodes, 4_096);
        assert_eq!(c.max_edges, 65_536);
        assert_eq!(c.snapshot_interval_secs, 60);
        assert_eq!(c.prune_disconnected_after_secs, 300);
    }

    #[test]
    fn test_config_custom() {
        let c = NtmMapperConfig {
            max_nodes: 10,
            max_edges: 20,
            snapshot_interval_secs: 5,
            prune_disconnected_after_secs: 15,
        };
        assert_eq!(c.max_nodes, 10);
    }

    // ── Construction ──────────────────────────────────────────────────────────

    #[test]
    fn test_new_mapper_empty() {
        let m = make_mapper();
        assert_eq!(m.node_count(), 0);
        assert_eq!(m.edge_count(), 0);
    }

    #[test]
    fn test_with_defaults() {
        let m = NetworkTopologyMapper::with_defaults();
        assert_eq!(m.node_count(), 0);
    }

    // ── add_node ──────────────────────────────────────────────────────────────

    #[test]
    fn test_add_single_node() {
        let mut m = make_mapper();
        add_n(&mut m, 1);
        assert_eq!(m.node_count(), 1);
    }

    #[test]
    fn test_add_node_idempotent() {
        let mut m = make_mapper();
        add_n(&mut m, 1);
        add_n(&mut m, 1); // second call updates, not duplicates
        assert_eq!(m.node_count(), 1);
    }

    #[test]
    fn test_add_node_updates_rtt() {
        let mut m = make_mapper();
        m.add_node(make_id(1), "a:1", None, 5.0, 100)
            .expect("test: add_node for node 1 (initial)");
        m.add_node(make_id(1), "a:1", None, 99.0, 200)
            .expect("test: add_node for node 1 (update rtt)");
        assert!(
            (m.get_node(&make_id(1))
                .expect("test: get_node for node 1 to check rtt")
                .rtt_ms
                - 99.0)
                .abs()
                < f64::EPSILON
        );
    }

    #[test]
    fn test_add_node_capacity_exceeded() {
        let config = NtmMapperConfig {
            max_nodes: 2,
            ..Default::default()
        };
        let mut m = NetworkTopologyMapper::new(config);
        add_n(&mut m, 1);
        add_n(&mut m, 2);
        let r = m.add_node(make_id(3), "x", None, 1.0, 0);
        assert!(matches!(r, Err(NtmMapperError::CapacityExceeded(_))));
    }

    #[test]
    fn test_add_node_with_region() {
        let mut m = make_mapper();
        m.add_node(make_id(5), "addr", Some("eu-west".into()), 1.0, 0)
            .expect("test: add_node for node 5 with region");
        assert_eq!(
            m.get_node(&make_id(5))
                .expect("test: get_node for node 5 to check region")
                .region
                .as_deref(),
            Some("eu-west")
        );
    }

    // ── get_node ──────────────────────────────────────────────────────────────

    #[test]
    fn test_get_node_existing() {
        let mut m = make_mapper();
        add_n(&mut m, 7);
        let n = m.get_node(&make_id(7));
        assert!(n.is_some());
    }

    #[test]
    fn test_get_node_missing() {
        let m = make_mapper();
        assert!(m.get_node(&make_id(99)).is_none());
    }

    // ── remove_node ───────────────────────────────────────────────────────────

    #[test]
    fn test_remove_node_basic() {
        let mut m = make_mapper();
        add_n(&mut m, 1);
        m.remove_node(&make_id(1))
            .expect("test: remove_node should succeed for existing node 1");
        assert_eq!(m.node_count(), 0);
    }

    #[test]
    fn test_remove_node_missing() {
        let mut m = make_mapper();
        let r = m.remove_node(&make_id(42));
        assert!(matches!(r, Err(NtmMapperError::NotFound(_))));
    }

    #[test]
    fn test_remove_node_cascades_edges() {
        let mut m = make_mapper();
        add_n(&mut m, 1);
        add_n(&mut m, 2);
        add_e(&mut m, 1, 2, 5.0);
        m.remove_node(&make_id(1))
            .expect("test: remove_node should cascade edges for node 1");
        assert_eq!(m.edge_count(), 0);
    }

    // ── update_node_rtt ───────────────────────────────────────────────────────

    #[test]
    fn test_update_rtt_ok() {
        let mut m = make_mapper();
        add_n(&mut m, 3);
        m.update_node_rtt(&make_id(3), 42.0)
            .expect("test: update_node_rtt should succeed for existing node 3");
        assert!(
            (m.get_node(&make_id(3))
                .expect("test: get_node should return node 3 after rtt update")
                .rtt_ms
                - 42.0)
                .abs()
                < f64::EPSILON
        );
    }

    #[test]
    fn test_update_rtt_missing() {
        let mut m = make_mapper();
        let r = m.update_node_rtt(&make_id(99), 1.0);
        assert!(matches!(r, Err(NtmMapperError::NotFound(_))));
    }

    // ── set_bootstrap ─────────────────────────────────────────────────────────

    #[test]
    fn test_set_bootstrap_true() {
        let mut m = make_mapper();
        add_n(&mut m, 1);
        m.set_bootstrap(&make_id(1), true)
            .expect("test: set_bootstrap true should succeed for node 1");
        assert!(
            m.get_node(&make_id(1))
                .expect("test: get_node should return node 1 after set_bootstrap")
                .is_bootstrap
        );
    }

    #[test]
    fn test_set_bootstrap_false() {
        let mut m = make_mapper();
        add_n(&mut m, 1);
        m.set_bootstrap(&make_id(1), true)
            .expect("test: set_bootstrap true should succeed");
        m.set_bootstrap(&make_id(1), false)
            .expect("test: set_bootstrap false should succeed");
        assert!(
            !m.get_node(&make_id(1))
                .expect("test: get_node should return node 1 after set_bootstrap false")
                .is_bootstrap
        );
    }

    // ── add_edge ──────────────────────────────────────────────────────────────

    #[test]
    fn test_add_edge_basic() {
        let mut m = make_mapper();
        add_n(&mut m, 1);
        add_n(&mut m, 2);
        add_e(&mut m, 1, 2, 10.0);
        assert_eq!(m.edge_count(), 1);
    }

    #[test]
    fn test_add_edge_updates_latency() {
        let mut m = make_mapper();
        add_n(&mut m, 1);
        add_n(&mut m, 2);
        let eid = add_e(&mut m, 1, 2, 10.0);
        m.add_edge(make_id(1), make_id(2), 99.0, 500.0, 200)
            .expect("test: add_edge update should succeed");
        assert!(
            (m.get_edge(eid)
                .expect("test: get_edge should return edge after latency update")
                .latency_ms
                - 99.0)
                .abs()
                < f64::EPSILON
        );
    }

    #[test]
    fn test_add_edge_missing_src() {
        let mut m = make_mapper();
        add_n(&mut m, 2);
        let r = m.add_edge(make_id(1), make_id(2), 1.0, 1.0, 0);
        assert!(matches!(r, Err(NtmMapperError::NotFound(_))));
    }

    #[test]
    fn test_add_edge_missing_dst() {
        let mut m = make_mapper();
        add_n(&mut m, 1);
        let r = m.add_edge(make_id(1), make_id(2), 1.0, 1.0, 0);
        assert!(matches!(r, Err(NtmMapperError::NotFound(_))));
    }

    #[test]
    fn test_add_edge_capacity_exceeded() {
        let config = NtmMapperConfig {
            max_edges: 1,
            ..Default::default()
        };
        let mut m = NetworkTopologyMapper::new(config);
        add_n(&mut m, 1);
        add_n(&mut m, 2);
        add_n(&mut m, 3);
        add_e(&mut m, 1, 2, 1.0);
        let r = m.add_edge(make_id(2), make_id(3), 1.0, 1.0, 0);
        assert!(matches!(r, Err(NtmMapperError::CapacityExceeded(_))));
    }

    // ── remove_edge ───────────────────────────────────────────────────────────

    #[test]
    fn test_remove_edge_ok() {
        let mut m = make_mapper();
        add_n(&mut m, 1);
        add_n(&mut m, 2);
        let eid = add_e(&mut m, 1, 2, 1.0);
        m.remove_edge(eid)
            .expect("test: remove_edge should succeed for existing edge");
        assert_eq!(m.edge_count(), 0);
    }

    #[test]
    fn test_remove_edge_missing() {
        let mut m = make_mapper();
        let r = m.remove_edge(0xdeadbeef);
        assert!(matches!(r, Err(NtmMapperError::NotFound(_))));
    }

    #[test]
    fn test_remove_edge_updates_degree() {
        let mut m = make_mapper();
        add_n(&mut m, 1);
        add_n(&mut m, 2);
        let eid = add_e(&mut m, 1, 2, 1.0);
        assert_eq!(
            m.get_node(&make_id(1))
                .expect("test: get_node should return node 1 before remove_edge")
                .degree,
            1
        );
        m.remove_edge(eid)
            .expect("test: remove_edge should succeed when updating degree");
        assert_eq!(
            m.get_node(&make_id(1))
                .expect("test: get_node should return node 1 after remove_edge")
                .degree,
            0
        );
    }

    // ── update_edge_latency ───────────────────────────────────────────────────

    #[test]
    fn test_update_edge_latency_ok() {
        let mut m = make_mapper();
        add_n(&mut m, 1);
        add_n(&mut m, 2);
        let eid = add_e(&mut m, 1, 2, 5.0);
        m.update_edge_latency(eid, 77.0)
            .expect("test: update_edge_latency should succeed");
        assert!(
            (m.get_edge(eid)
                .expect("test: get_edge should return edge after latency update")
                .latency_ms
                - 77.0)
                .abs()
                < f64::EPSILON
        );
    }

    #[test]
    fn test_update_edge_latency_missing() {
        let mut m = make_mapper();
        let r = m.update_edge_latency(999, 1.0);
        assert!(matches!(r, Err(NtmMapperError::NotFound(_))));
    }

    // ── find_edge ─────────────────────────────────────────────────────────────

    #[test]
    fn test_find_edge_exists() {
        let mut m = make_mapper();
        add_n(&mut m, 1);
        add_n(&mut m, 2);
        let eid = add_e(&mut m, 1, 2, 1.0);
        assert_eq!(m.find_edge(&make_id(1), &make_id(2)), Some(eid));
    }

    #[test]
    fn test_find_edge_not_exists() {
        let m = make_mapper();
        assert!(m.find_edge(&make_id(1), &make_id(2)).is_none());
    }

    // ── neighbors ─────────────────────────────────────────────────────────────

    #[test]
    fn test_neighbors_empty() {
        let mut m = make_mapper();
        add_n(&mut m, 1);
        assert!(m.neighbors(&make_id(1)).is_empty());
    }

    #[test]
    fn test_neighbors_single() {
        let mut m = make_mapper();
        add_n(&mut m, 1);
        add_n(&mut m, 2);
        add_e(&mut m, 1, 2, 1.0);
        assert_eq!(m.neighbors(&make_id(1)), vec![make_id(2)]);
    }

    #[test]
    fn test_neighbors_multiple() {
        let mut m = make_mapper();
        for v in 1..=4 {
            add_n(&mut m, v);
        }
        add_e(&mut m, 1, 2, 1.0);
        add_e(&mut m, 1, 3, 2.0);
        add_e(&mut m, 1, 4, 3.0);
        let nbs = m.neighbors(&make_id(1));
        assert_eq!(nbs.len(), 3);
    }

    // ── shortest_path ─────────────────────────────────────────────────────────

    #[test]
    fn test_shortest_path_direct() {
        let mut m = make_mapper();
        add_n(&mut m, 1);
        add_n(&mut m, 2);
        add_e(&mut m, 1, 2, 5.0);
        let path = m
            .shortest_path(&make_id(1), &make_id(2))
            .expect("test: shortest_path should find direct path");
        assert_eq!(path, vec![make_id(1), make_id(2)]);
    }

    #[test]
    fn test_shortest_path_multi_hop() {
        let mut m = make_mapper();
        for v in 1..=4 {
            add_n(&mut m, v);
        }
        // 1→2 (100), 1→3 (1), 3→4 (1), 2→4 (1)
        add_e(&mut m, 1, 2, 100.0);
        add_e(&mut m, 1, 3, 1.0);
        add_e(&mut m, 3, 4, 1.0);
        add_e(&mut m, 2, 4, 1.0);
        let path = m
            .shortest_path(&make_id(1), &make_id(4))
            .expect("test: shortest_path should find multi-hop path");
        // Should go 1→3→4 (cost 2) not 1→2→4 (cost 101).
        assert_eq!(path, vec![make_id(1), make_id(3), make_id(4)]);
    }

    #[test]
    fn test_shortest_path_no_path() {
        let mut m = make_mapper();
        add_n(&mut m, 1);
        add_n(&mut m, 2);
        assert!(m.shortest_path(&make_id(1), &make_id(2)).is_none());
    }

    #[test]
    fn test_shortest_path_same_node() {
        let mut m = make_mapper();
        add_n(&mut m, 1);
        let path = m
            .shortest_path(&make_id(1), &make_id(1))
            .expect("test: shortest_path to self should return single node");
        assert_eq!(path, vec![make_id(1)]);
    }

    #[test]
    fn test_shortest_path_missing_node() {
        let m = make_mapper();
        assert!(m.shortest_path(&make_id(1), &make_id(2)).is_none());
    }

    // ── bfs_distance ──────────────────────────────────────────────────────────

    #[test]
    fn test_bfs_distance_direct() {
        let mut m = make_mapper();
        add_n(&mut m, 1);
        add_n(&mut m, 2);
        add_e(&mut m, 1, 2, 1.0);
        assert_eq!(m.bfs_distance(&make_id(1), &make_id(2)), Some(1));
    }

    #[test]
    fn test_bfs_distance_two_hops() {
        let mut m = make_mapper();
        add_n(&mut m, 1);
        add_n(&mut m, 2);
        add_n(&mut m, 3);
        add_e(&mut m, 1, 2, 1.0);
        add_e(&mut m, 2, 3, 1.0);
        assert_eq!(m.bfs_distance(&make_id(1), &make_id(3)), Some(2));
    }

    #[test]
    fn test_bfs_distance_self() {
        let mut m = make_mapper();
        add_n(&mut m, 1);
        assert_eq!(m.bfs_distance(&make_id(1), &make_id(1)), Some(0));
    }

    #[test]
    fn test_bfs_distance_unreachable() {
        let mut m = make_mapper();
        add_n(&mut m, 1);
        add_n(&mut m, 2);
        assert!(m.bfs_distance(&make_id(1), &make_id(2)).is_none());
    }

    // ── diameter ──────────────────────────────────────────────────────────────

    #[test]
    fn test_diameter_empty() {
        let m = make_mapper();
        assert_eq!(m.diameter(), 0);
    }

    #[test]
    fn test_diameter_single_node() {
        let mut m = make_mapper();
        add_n(&mut m, 1);
        assert_eq!(m.diameter(), 0);
    }

    #[test]
    fn test_diameter_two_connected() {
        let mut m = make_mapper();
        add_n(&mut m, 1);
        add_n(&mut m, 2);
        add_e(&mut m, 1, 2, 1.0);
        assert_eq!(m.diameter(), 1);
    }

    #[test]
    fn test_diameter_chain() {
        // 1→2→3→4: diameter = 3 (1 to 4)
        let mut m = make_mapper();
        for v in 1..=4 {
            add_n(&mut m, v);
        }
        add_e(&mut m, 1, 2, 1.0);
        add_e(&mut m, 2, 3, 1.0);
        add_e(&mut m, 3, 4, 1.0);
        assert_eq!(m.diameter(), 3);
    }

    // ── clustering_coefficient ────────────────────────────────────────────────

    #[test]
    fn test_clustering_low_degree() {
        let mut m = make_mapper();
        add_n(&mut m, 1);
        assert!((m.clustering_coefficient(&make_id(1)) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_clustering_triangle() {
        // Fully connected triangle: C = 1.0.
        let mut m = make_mapper();
        add_n(&mut m, 1);
        add_n(&mut m, 2);
        add_n(&mut m, 3);
        add_e(&mut m, 1, 2, 1.0);
        add_e(&mut m, 1, 3, 1.0);
        add_e(&mut m, 2, 3, 1.0); // 2→3
        add_e(&mut m, 3, 2, 1.0); // 3→2
                                  // Node 1 has neighbours 2 and 3; 2→3 and 3→2 exist → both pairs satisfied.
        let c = m.clustering_coefficient(&make_id(1));
        assert!(c > 0.0);
    }

    #[test]
    fn test_clustering_no_edges_between_neighbours() {
        let mut m = make_mapper();
        add_n(&mut m, 1);
        add_n(&mut m, 2);
        add_n(&mut m, 3);
        add_e(&mut m, 1, 2, 1.0);
        add_e(&mut m, 1, 3, 1.0);
        // No edge between 2 and 3.
        let c = m.clustering_coefficient(&make_id(1));
        assert!((c - 0.0).abs() < f64::EPSILON);
    }

    // ── take_snapshot ─────────────────────────────────────────────────────────

    #[test]
    fn test_snapshot_basic() {
        let mut m = make_mapper();
        add_n(&mut m, 1);
        add_n(&mut m, 2);
        let snap = m.take_snapshot(1000);
        assert_eq!(snap.node_count, 2);
        assert_eq!(snap.ts, 1000);
    }

    #[test]
    fn test_snapshot_stored() {
        let mut m = make_mapper();
        m.take_snapshot(1);
        assert_eq!(m.snapshots().len(), 1);
    }

    #[test]
    fn test_snapshot_ring_bounded() {
        let mut m = make_mapper();
        for i in 0..25u64 {
            m.take_snapshot(i);
        }
        assert_eq!(m.snapshots().len(), 20);
    }

    #[test]
    fn test_snapshot_oldest_dropped() {
        let mut m = make_mapper();
        for i in 0..21u64 {
            m.take_snapshot(i);
        }
        // The oldest snapshot (ts = 0) should have been dropped.
        assert_eq!(
            m.snapshots()
                .front()
                .expect("test: snapshots should have at least one entry after 21 takes")
                .ts,
            1
        );
    }

    #[test]
    fn test_snapshot_edge_count() {
        let mut m = make_mapper();
        add_n(&mut m, 1);
        add_n(&mut m, 2);
        add_e(&mut m, 1, 2, 1.0);
        let snap = m.take_snapshot(0);
        assert_eq!(snap.edge_count, 1);
    }

    // ── prune_stale ───────────────────────────────────────────────────────────

    #[test]
    fn test_prune_stale_removes_old() {
        let config = NtmMapperConfig {
            prune_disconnected_after_secs: 100,
            ..Default::default()
        };
        let mut m = NetworkTopologyMapper::new(config);
        m.add_node(make_id(1), "a", None, 1.0, 0)
            .expect("test: add_node for stale node 1 should succeed"); // last_seen = 0
        m.add_node(make_id(2), "b", None, 1.0, 500)
            .expect("test: add_node for fresh node 2 should succeed"); // last_seen = 500
        m.prune_stale(600); // threshold = 500
        assert_eq!(m.node_count(), 1);
        assert!(m.get_node(&make_id(2)).is_some());
    }

    #[test]
    fn test_prune_stale_keeps_recent() {
        let config = NtmMapperConfig {
            prune_disconnected_after_secs: 300,
            ..Default::default()
        };
        let mut m = NetworkTopologyMapper::new(config);
        m.add_node(make_id(1), "a", None, 1.0, 1000)
            .expect("test: add_node for recent node should succeed");
        m.prune_stale(1200); // threshold = 900; 1000 > 900 → kept
        assert_eq!(m.node_count(), 1);
    }

    #[test]
    fn test_prune_stale_cascades_edges() {
        let config = NtmMapperConfig {
            prune_disconnected_after_secs: 100,
            ..Default::default()
        };
        let mut m = NetworkTopologyMapper::new(config);
        m.add_node(make_id(1), "a", None, 1.0, 0)
            .expect("test: add_node for cascade edge test node 1 should succeed");
        m.add_node(make_id(2), "b", None, 1.0, 0)
            .expect("test: add_node for cascade edge test node 2 should succeed");
        add_e(&mut m, 1, 2, 1.0);
        m.prune_stale(1000);
        assert_eq!(m.edge_count(), 0);
    }

    // ── topology_stats ────────────────────────────────────────────────────────

    #[test]
    fn test_topology_stats_empty() {
        let m = make_mapper();
        let s = m.topology_stats();
        assert!((s.density - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_topology_stats_density() {
        // 2 nodes, 1 directed edge → density = 1/2.
        let mut m = make_mapper();
        add_n(&mut m, 1);
        add_n(&mut m, 2);
        add_e(&mut m, 1, 2, 1.0);
        let s = m.topology_stats();
        assert!((s.density - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_topology_stats_centrality_present() {
        let mut m = make_mapper();
        add_n(&mut m, 1);
        add_n(&mut m, 2);
        add_n(&mut m, 3);
        add_e(&mut m, 1, 2, 1.0);
        add_e(&mut m, 2, 3, 1.0);
        let s = m.topology_stats();
        assert!(s.centrality.contains_key(&make_id(1)));
        assert!(s.centrality.contains_key(&make_id(2)));
        assert!(s.centrality.contains_key(&make_id(3)));
    }

    // ── betweenness centrality ────────────────────────────────────────────────

    #[test]
    fn test_betweenness_single_node() {
        let mut m = make_mapper();
        add_n(&mut m, 1);
        let b = m.compute_betweenness_centrality();
        assert!((b[&make_id(1)] - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_betweenness_chain() {
        // In a directed chain 1→2→3, node 2 is the only intermediary.
        let mut m = make_mapper();
        add_n(&mut m, 1);
        add_n(&mut m, 2);
        add_n(&mut m, 3);
        add_e(&mut m, 1, 2, 1.0);
        add_e(&mut m, 2, 3, 1.0);
        let b = m.compute_betweenness_centrality();
        // Node 2 has higher betweenness than node 1 or 3.
        assert!(b[&make_id(2)] >= b[&make_id(1)]);
        assert!(b[&make_id(2)] >= b[&make_id(3)]);
    }

    // ── PRNG / hash helpers ───────────────────────────────────────────────────

    #[test]
    fn test_xorshift64_nonzero() {
        let mut state = 12345u64;
        let v = xorshift64(&mut state);
        assert_ne!(v, 0);
    }

    #[test]
    fn test_xorshift64_changes_state() {
        let mut state = 1u64;
        let first = xorshift64(&mut state);
        let second = xorshift64(&mut state);
        assert_ne!(first, second);
    }

    #[test]
    fn test_fnv1a_64_empty() {
        let h = fnv1a_64(b"");
        assert_eq!(h, 14_695_981_039_346_656_037u64);
    }

    #[test]
    fn test_fnv1a_64_known() {
        // "hello" FNV-1a 64-bit
        let h = fnv1a_64(b"hello");
        assert_ne!(h, 0);
        assert_ne!(h, 14_695_981_039_346_656_037u64);
    }

    #[test]
    fn test_fnv1a_64_different_inputs() {
        let h1 = fnv1a_64(b"foo");
        let h2 = fnv1a_64(b"bar");
        assert_ne!(h1, h2);
    }

    // ── edge id uniqueness ────────────────────────────────────────────────────

    #[test]
    fn test_edge_ids_unique() {
        let mut m = make_mapper();
        for v in 1..=5 {
            add_n(&mut m, v);
        }
        let e12 = add_e(&mut m, 1, 2, 1.0);
        let e13 = add_e(&mut m, 1, 3, 1.0);
        let e14 = add_e(&mut m, 1, 4, 1.0);
        assert_ne!(e12, e13);
        assert_ne!(e12, e14);
        assert_ne!(e13, e14);
    }

    // ── degree counting ───────────────────────────────────────────────────────

    #[test]
    fn test_degree_after_add_edge() {
        let mut m = make_mapper();
        add_n(&mut m, 1);
        add_n(&mut m, 2);
        add_e(&mut m, 1, 2, 1.0);
        // src gets out-degree 1, dst gets in-degree 1.
        assert_eq!(
            m.get_node(&make_id(1))
                .expect("test: get_node for node 1 degree after add_edge")
                .degree,
            1
        );
        assert_eq!(
            m.get_node(&make_id(2))
                .expect("test: get_node for node 2 degree after add_edge")
                .degree,
            1
        );
    }

    #[test]
    fn test_degree_bidirectional() {
        let mut m = make_mapper();
        add_n(&mut m, 1);
        add_n(&mut m, 2);
        add_e(&mut m, 1, 2, 1.0);
        add_e(&mut m, 2, 1, 1.0);
        assert_eq!(
            m.get_node(&make_id(1))
                .expect("test: get_node for node 1 degree after bidirectional edges")
                .degree,
            2
        );
        assert_eq!(
            m.get_node(&make_id(2))
                .expect("test: get_node for node 2 degree after bidirectional edges")
                .degree,
            2
        );
    }

    // ── nodes() / edges() iterators ───────────────────────────────────────────

    #[test]
    fn test_nodes_iterator() {
        let mut m = make_mapper();
        add_n(&mut m, 1);
        add_n(&mut m, 2);
        let ids: Vec<NtmNodeId> = m.nodes().map(|n| n.id).collect();
        assert_eq!(ids.len(), 2);
    }

    #[test]
    fn test_edges_iterator() {
        let mut m = make_mapper();
        add_n(&mut m, 1);
        add_n(&mut m, 2);
        add_e(&mut m, 1, 2, 1.0);
        let eids: Vec<NtmEdgeId> = m.edges().map(|e| e.id).collect();
        assert_eq!(eids.len(), 1);
    }

    // ── NtmMapperError display ────────────────────────────────────────────────

    #[test]
    fn test_error_display_not_found() {
        let e = NtmMapperError::NotFound("x".into());
        assert!(e.to_string().contains("not found"));
    }

    #[test]
    fn test_error_display_capacity() {
        let e = NtmMapperError::CapacityExceeded("full".into());
        assert!(e.to_string().contains("capacity exceeded"));
    }

    #[test]
    fn test_error_display_internal() {
        let e = NtmMapperError::Internal("oops".into());
        assert!(e.to_string().contains("internal error"));
    }

    // ── avg_path_length ───────────────────────────────────────────────────────

    #[test]
    fn test_avg_path_length_empty() {
        let m = make_mapper();
        let s = m.topology_stats();
        assert!((s.avg_path_length - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_avg_path_length_two_nodes_connected() {
        let mut m = make_mapper();
        add_n(&mut m, 1);
        add_n(&mut m, 2);
        add_e(&mut m, 1, 2, 1.0);
        let s = m.topology_stats();
        // Only path: 1→2 = 1 hop, 2→1 unreachable.
        assert!(s.avg_path_length >= 0.0);
    }

    // ── miscellaneous edge cases ───────────────────────────────────────────────

    #[test]
    fn test_remove_nonexistent_edge_error() {
        let mut m = make_mapper();
        assert!(m.remove_edge(42).is_err());
    }

    #[test]
    fn test_snapshot_avg_degree() {
        let mut m = make_mapper();
        add_n(&mut m, 1);
        add_n(&mut m, 2);
        add_e(&mut m, 1, 2, 1.0);
        // Both nodes have degree 1; avg = 1.0.
        let snap = m.take_snapshot(0);
        assert!((snap.avg_degree - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_many_nodes_and_edges() {
        let mut m = make_mapper();
        for v in 0..10u8 {
            add_n(&mut m, v);
        }
        for u in 0..9u8 {
            add_e(&mut m, u, u + 1, (u + 1) as f64);
        }
        assert_eq!(m.node_count(), 10);
        assert_eq!(m.edge_count(), 9);
        let path = m.shortest_path(&make_id(0), &make_id(9));
        assert!(path.is_some());
    }

    #[test]
    fn test_type_alias_ntm_network_topology_mapper() {
        let _m: NtmNetworkTopologyMapper = NtmNetworkTopologyMapper::with_defaults();
    }
}
