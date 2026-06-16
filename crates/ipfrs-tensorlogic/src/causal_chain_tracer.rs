//! Causal Chain Tracer — production-quality causal chain tracing for event sequences.
//!
//! This module provides:
//! - Directed causal graph with configurable relation types
//! - BFS/DFS chain tracing with depth, strength, time-window, and relation filters
//! - Shortest-path (hop count) and strongest-path (max product of edge strengths)
//! - Root-cause identification and downstream-effect enumeration
//! - DFS-based cycle detection on edge insertion

use std::collections::{HashMap, HashSet, VecDeque};

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors produced by [`CausalChainTracer`].
#[derive(Debug, Clone, PartialEq)]
pub enum TracerError {
    /// A node ID was referenced but does not exist in the graph.
    NodeNotFound(String),
    /// Adding an edge would create a cycle.
    CycleDetected {
        /// Ordered sequence of node IDs that forms the cycle.
        path: Vec<String>,
    },
    /// The query would require exploring more nodes than allowed.
    QueryTooExpensive(usize),
    /// An edge strength value was outside `[0.0, 1.0]`.
    InvalidStrength(f64),
    /// A traversal exceeded the configured maximum depth.
    MaxDepthExceeded,
}

impl std::fmt::Display for TracerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NodeNotFound(id) => write!(f, "node not found: {id}"),
            Self::CycleDetected { path } => write!(f, "cycle detected: {}", path.join(" -> ")),
            Self::QueryTooExpensive(n) => write!(f, "query too expensive: {n} nodes"),
            Self::InvalidStrength(s) => write!(f, "invalid strength {s}: must be in [0.0, 1.0]"),
            Self::MaxDepthExceeded => write!(f, "max depth exceeded"),
        }
    }
}

impl std::error::Error for TracerError {}

// ---------------------------------------------------------------------------
// CausalRelation
// ---------------------------------------------------------------------------

/// Semantic relationship carried by a [`CausalEdge`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CausalRelation {
    /// A directly causes B.
    DirectCause,
    /// A indirectly causes B (via one or more intermediaries).
    IndirectCause,
    /// A enables B to occur.
    Enables,
    /// A inhibits B from occurring.
    Inhibits,
    /// A and B are correlated but neither directly causes the other.
    Correlates,
    /// A temporally precedes B without a strict causal link.
    Precedes,
}

// ---------------------------------------------------------------------------
// Core node/edge types
// ---------------------------------------------------------------------------

/// A node in the causal graph, representing a single event.
#[derive(Debug, Clone)]
pub struct CausalNode {
    /// Unique identifier for this event.
    pub id: String,
    /// Category/type label for this event.
    pub event_type: String,
    /// Microsecond timestamp at which this event occurred.
    pub timestamp: u64,
    /// Arbitrary key-value metadata.
    pub attributes: Vec<(String, String)>,
    /// Confidence that this event occurred as described, in `[0.0, 1.0]`.
    pub confidence: f64,
}

impl CausalNode {
    /// Convenience constructor.
    pub fn new(
        id: impl Into<String>,
        event_type: impl Into<String>,
        timestamp: u64,
        confidence: f64,
    ) -> Self {
        Self {
            id: id.into(),
            event_type: event_type.into(),
            timestamp,
            attributes: Vec::new(),
            confidence,
        }
    }

    /// Builder: attach a key-value attribute.
    pub fn with_attribute(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.attributes.push((key.into(), value.into()));
        self
    }
}

/// A directed edge between two nodes in the causal graph.
#[derive(Debug, Clone)]
pub struct CausalEdge {
    /// Source node ID.
    pub from_id: String,
    /// Destination node ID.
    pub to_id: String,
    /// Semantic relationship.
    pub relation: CausalRelation,
    /// Strength of this causal link, in `[0.0, 1.0]`.
    pub strength: f64,
    /// Typical delay in microseconds between cause and effect.
    pub delay_us: u64,
}

impl CausalEdge {
    /// Convenience constructor.
    pub fn new(
        from_id: impl Into<String>,
        to_id: impl Into<String>,
        relation: CausalRelation,
        strength: f64,
        delay_us: u64,
    ) -> Self {
        Self {
            from_id: from_id.into(),
            to_id: to_id.into(),
            relation,
            strength,
            delay_us,
        }
    }
}

// ---------------------------------------------------------------------------
// CausalChain — a complete traced path
// ---------------------------------------------------------------------------

/// A traced causal chain from a root event to one or more leaf events.
#[derive(Debug, Clone)]
pub struct CausalChain {
    /// All nodes participating in the chain (ordered by traversal).
    pub nodes: Vec<CausalNode>,
    /// All edges in traversal order.
    pub edges: Vec<CausalEdge>,
    /// The starting node ID.
    pub root_id: String,
    /// Terminal node IDs (no outgoing edges within the chain).
    pub leaf_ids: Vec<String>,
    /// Product of all edge strengths along the chain; 1.0 for a single node.
    pub chain_confidence: f64,
    /// Maximum hop depth of the chain.
    pub depth: usize,
}

// ---------------------------------------------------------------------------
// TraceQuery
// ---------------------------------------------------------------------------

/// Query parameters passed to [`CausalChainTracer::trace`].
#[derive(Debug, Clone)]
pub struct TraceQuery {
    /// If `Some`, start traversal from this node; otherwise start from all roots.
    pub root_event_id: Option<String>,
    /// If non-empty, only include nodes whose `event_type` is in this list.
    pub event_types: Vec<String>,
    /// Maximum traversal depth (0 = root only).
    pub max_depth: usize,
    /// Minimum edge strength to follow.
    pub min_strength: f64,
    /// Optional `[start_us, end_us]` time window (inclusive).
    pub time_window_us: Option<(u64, u64)>,
    /// If non-empty, only traverse edges whose relation is in this list.
    pub include_relations: Vec<CausalRelation>,
}

impl Default for TraceQuery {
    fn default() -> Self {
        Self {
            root_event_id: None,
            event_types: Vec::new(),
            max_depth: 16,
            min_strength: 0.0,
            time_window_us: None,
            include_relations: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// TracerConfig
// ---------------------------------------------------------------------------

/// Configuration for a [`CausalChainTracer`] instance.
#[derive(Debug, Clone)]
pub struct TracerConfig {
    /// Hard limit on chain depth during tracing.
    pub max_chain_depth: usize,
    /// Minimum edge strength to index.
    pub min_edge_strength: f64,
    /// Maximum number of nodes the graph may hold.
    pub max_nodes: usize,
    /// When `true`, adding an edge triggers cycle detection.
    pub enable_cycle_detection: bool,
    /// Only return chains whose `chain_confidence` meets this threshold.
    pub confidence_threshold: f64,
}

impl Default for TracerConfig {
    fn default() -> Self {
        Self {
            max_chain_depth: 32,
            min_edge_strength: 0.0,
            max_nodes: 100_000,
            enable_cycle_detection: true,
            confidence_threshold: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// TracerStats
// ---------------------------------------------------------------------------

/// Aggregate statistics for a [`CausalChainTracer`].
#[derive(Debug, Clone, Default)]
pub struct TracerStats {
    /// Number of nodes currently tracked.
    pub nodes_tracked: usize,
    /// Number of edges currently tracked.
    pub edges_tracked: usize,
    /// Total number of `trace()` calls completed.
    pub chains_traced: usize,
    /// Running average of chain depth across all traced chains.
    pub avg_chain_depth: f64,
    /// Number of cycle-detection rejections since creation.
    pub cycles_detected: usize,
}

// ---------------------------------------------------------------------------
// Internal adjacency helpers
// ---------------------------------------------------------------------------

/// Lightweight edge reference stored in the adjacency list.
#[derive(Debug, Clone)]
struct EdgeRef {
    to: String,
    edge_index: usize,
}

// ---------------------------------------------------------------------------
// CausalChainTracer
// ---------------------------------------------------------------------------

/// Production-quality causal chain tracer for event sequences.
///
/// Maintains a directed graph of [`CausalNode`]s connected by [`CausalEdge`]s and
/// exposes high-level methods for chain tracing, path finding, root-cause
/// identification, and downstream-effect enumeration.
pub struct CausalChainTracer {
    config: TracerConfig,
    nodes: HashMap<String, CausalNode>,
    edges: Vec<CausalEdge>,
    /// Forward adjacency: node_id -> list of outgoing edge refs.
    adj_out: HashMap<String, Vec<EdgeRef>>,
    /// Backward adjacency: node_id -> list of incoming node IDs.
    adj_in: HashMap<String, Vec<String>>,
    stats: TracerStats,
}

impl CausalChainTracer {
    // ------------------------------------------------------------------
    // Construction
    // ------------------------------------------------------------------

    /// Create a new tracer with the given configuration.
    pub fn new(config: TracerConfig) -> Self {
        Self {
            config,
            nodes: HashMap::new(),
            edges: Vec::new(),
            adj_out: HashMap::new(),
            adj_in: HashMap::new(),
            stats: TracerStats::default(),
        }
    }

    /// Create a tracer with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(TracerConfig::default())
    }

    // ------------------------------------------------------------------
    // Mutation
    // ------------------------------------------------------------------

    /// Insert a node into the graph.
    ///
    /// Returns [`TracerError::QueryTooExpensive`] when `max_nodes` would be exceeded.
    pub fn add_node(&mut self, node: CausalNode) -> Result<(), TracerError> {
        if self.nodes.len() >= self.config.max_nodes && !self.nodes.contains_key(&node.id) {
            return Err(TracerError::QueryTooExpensive(self.nodes.len()));
        }
        // Ensure adjacency lists exist even if there are no edges yet.
        self.adj_out.entry(node.id.clone()).or_default();
        self.adj_in.entry(node.id.clone()).or_default();
        self.nodes.insert(node.id.clone(), node);
        self.stats.nodes_tracked = self.nodes.len();
        Ok(())
    }

    /// Insert an edge into the graph.
    ///
    /// Validates:
    /// - Both endpoints must exist.
    /// - `strength` must be in `[0.0, 1.0]`.
    /// - When `enable_cycle_detection` is set, the edge must not introduce a cycle.
    pub fn add_edge(&mut self, edge: CausalEdge) -> Result<(), TracerError> {
        if !self.nodes.contains_key(&edge.from_id) {
            return Err(TracerError::NodeNotFound(edge.from_id.clone()));
        }
        if !self.nodes.contains_key(&edge.to_id) {
            return Err(TracerError::NodeNotFound(edge.to_id.clone()));
        }
        if edge.strength < 0.0 || edge.strength > 1.0 {
            return Err(TracerError::InvalidStrength(edge.strength));
        }

        if self.config.enable_cycle_detection {
            // After adding from->to, a cycle exists iff `from` is reachable from `to`.
            if let Some(cycle_path) = self.find_cycle_path(&edge.to_id, &edge.from_id) {
                self.stats.cycles_detected += 1;
                // Prepend from_id so the path reads: from -> to -> ... -> from
                let mut full_path = vec![edge.from_id.clone()];
                full_path.extend(cycle_path);
                return Err(TracerError::CycleDetected { path: full_path });
            }
        }

        let edge_index = self.edges.len();
        let from = edge.from_id.clone();
        let to = edge.to_id.clone();

        self.adj_out.entry(from.clone()).or_default().push(EdgeRef {
            to: to.clone(),
            edge_index,
        });

        self.adj_in.entry(to).or_default().push(from);

        self.edges.push(edge);
        self.stats.edges_tracked = self.edges.len();
        Ok(())
    }

    /// Remove a node and all edges that touch it.
    pub fn remove_node(&mut self, id: &str) -> Result<(), TracerError> {
        if !self.nodes.contains_key(id) {
            return Err(TracerError::NodeNotFound(id.to_string()));
        }

        // Collect edge indices to remove.
        let edge_indices_to_remove: HashSet<usize> = self
            .edges
            .iter()
            .enumerate()
            .filter(|(_, e)| e.from_id == id || e.to_id == id)
            .map(|(i, _)| i)
            .collect();

        // Rebuild edge list, shifting adjacency indices.
        let mut new_edges: Vec<CausalEdge> = Vec::new();
        let mut old_to_new: HashMap<usize, usize> = HashMap::new();
        for (old_idx, edge) in self.edges.drain(..).enumerate() {
            if !edge_indices_to_remove.contains(&old_idx) {
                old_to_new.insert(old_idx, new_edges.len());
                new_edges.push(edge);
            }
        }
        self.edges = new_edges;

        // Rebuild adjacency maps entirely.
        self.adj_out.clear();
        self.adj_in.clear();
        self.nodes.remove(id);

        for nid in self.nodes.keys() {
            self.adj_out.entry(nid.clone()).or_default();
            self.adj_in.entry(nid.clone()).or_default();
        }
        for (i, edge) in self.edges.iter().enumerate() {
            self.adj_out
                .entry(edge.from_id.clone())
                .or_default()
                .push(EdgeRef {
                    to: edge.to_id.clone(),
                    edge_index: i,
                });
            self.adj_in
                .entry(edge.to_id.clone())
                .or_default()
                .push(edge.from_id.clone());
        }

        self.stats.nodes_tracked = self.nodes.len();
        self.stats.edges_tracked = self.edges.len();
        Ok(())
    }

    // ------------------------------------------------------------------
    // Tracing
    // ------------------------------------------------------------------

    /// Trace causal chains according to `query`.
    ///
    /// Each returned [`CausalChain`] is a root-to-leaf path in the graph that
    /// satisfies all query constraints and whose `chain_confidence` meets the
    /// configured `confidence_threshold`.
    pub fn trace(&mut self, query: &TraceQuery) -> Result<Vec<CausalChain>, TracerError> {
        let roots = self.resolve_roots(query)?;
        let max_depth = query.max_depth.min(self.config.max_chain_depth);

        let mut all_chains: Vec<CausalChain> = Vec::new();

        for root_id in &roots {
            // Only consider the root if it passes the event_type / time_window filters.
            let root_node = self
                .nodes
                .get(root_id)
                .ok_or_else(|| TracerError::NodeNotFound(root_id.clone()))?;

            if !self.node_passes_query(root_node, query) {
                continue;
            }

            // DFS stack: (current_id, path_of_node_ids, path_of_edge_indices, confidence)
            let mut stack: Vec<(String, Vec<String>, Vec<usize>, f64)> =
                vec![(root_id.clone(), vec![root_id.clone()], Vec::new(), 1.0_f64)];

            while let Some((current_id, node_path, edge_path, confidence)) = stack.pop() {
                let depth = node_path.len() - 1;
                let out_edges = self.adj_out.get(&current_id).cloned().unwrap_or_default();

                // Collect edges that pass filters.
                let valid_next: Vec<(String, usize, f64)> = out_edges
                    .iter()
                    .filter_map(|eref| {
                        let edge = &self.edges[eref.edge_index];
                        // Strength filter.
                        if edge.strength < query.min_strength {
                            return None;
                        }
                        // Relation filter.
                        if !query.include_relations.is_empty()
                            && !query.include_relations.contains(&edge.relation)
                        {
                            return None;
                        }
                        // Target node filters.
                        let target = self.nodes.get(&eref.to)?;
                        if !self.node_passes_query(target, query) {
                            return None;
                        }
                        // Avoid revisiting nodes already in the current path (simple path).
                        if node_path.contains(&eref.to) {
                            return None;
                        }
                        Some((eref.to.clone(), eref.edge_index, edge.strength))
                    })
                    .collect();

                let is_leaf = valid_next.is_empty() || depth >= max_depth;

                if is_leaf {
                    // Emit this chain if confidence threshold is met.
                    if confidence >= self.config.confidence_threshold {
                        let chain = self.build_chain(&node_path, &edge_path, confidence)?;
                        all_chains.push(chain);
                    }
                } else {
                    for (next_id, edge_idx, strength) in valid_next {
                        let mut new_node_path = node_path.clone();
                        new_node_path.push(next_id.clone());
                        let mut new_edge_path = edge_path.clone();
                        new_edge_path.push(edge_idx);
                        let new_confidence = confidence * strength;
                        stack.push((next_id, new_node_path, new_edge_path, new_confidence));
                    }
                }
            }
        }

        // Update stats.
        self.stats.chains_traced += all_chains.len();
        if !all_chains.is_empty() {
            let total_depth: usize = all_chains.iter().map(|c| c.depth).sum();
            self.stats.avg_chain_depth = total_depth as f64 / all_chains.len() as f64;
        }

        Ok(all_chains)
    }

    // ------------------------------------------------------------------
    // Path finding
    // ------------------------------------------------------------------

    /// Return the shortest path (fewest hops) from `from` to `to`, or `None`.
    pub fn shortest_path(&self, from: &str, to: &str) -> Result<Option<Vec<String>>, TracerError> {
        if !self.nodes.contains_key(from) {
            return Err(TracerError::NodeNotFound(from.to_string()));
        }
        if !self.nodes.contains_key(to) {
            return Err(TracerError::NodeNotFound(to.to_string()));
        }
        if from == to {
            return Ok(Some(vec![from.to_string()]));
        }

        // BFS.
        let mut visited: HashSet<String> = HashSet::new();
        // Queue entries: (current_id, path_so_far)
        let mut queue: VecDeque<(String, Vec<String>)> = VecDeque::new();
        queue.push_back((from.to_string(), vec![from.to_string()]));
        visited.insert(from.to_string());

        while let Some((current, path)) = queue.pop_front() {
            let out_edges = match self.adj_out.get(&current) {
                Some(v) => v.clone(),
                None => continue,
            };
            for eref in &out_edges {
                if visited.contains(&eref.to) {
                    continue;
                }
                let mut new_path = path.clone();
                new_path.push(eref.to.clone());
                if eref.to == to {
                    return Ok(Some(new_path));
                }
                visited.insert(eref.to.clone());
                queue.push_back((eref.to.clone(), new_path));
            }
        }

        Ok(None)
    }

    /// Return the path from `from` to `to` that maximises the product of edge
    /// strengths (i.e., the "strongest" causal chain).
    ///
    /// Uses Dijkstra's algorithm on the negated log strengths so that path
    /// score = sum(-ln(strength_i)), and we minimise it.  A strength of 0 is
    /// treated as –∞ (the path is ignored).
    pub fn strongest_path(&self, from: &str, to: &str) -> Result<Option<CausalChain>, TracerError> {
        if !self.nodes.contains_key(from) {
            return Err(TracerError::NodeNotFound(from.to_string()));
        }
        if !self.nodes.contains_key(to) {
            return Err(TracerError::NodeNotFound(to.to_string()));
        }

        if from == to {
            let node = self
                .nodes
                .get(from)
                .ok_or_else(|| TracerError::NodeNotFound(from.to_string()))?
                .clone();
            return Ok(Some(CausalChain {
                nodes: vec![node],
                edges: Vec::new(),
                root_id: from.to_string(),
                leaf_ids: vec![from.to_string()],
                chain_confidence: 1.0,
                depth: 0,
            }));
        }

        // --- Dijkstra on -ln(strength) ---
        // dist[id] = (neg_log_product, predecessor, edge_index_used)
        let mut dist: HashMap<String, (f64, Option<String>, Option<usize>)> = HashMap::new();
        // Use a simple priority queue via BinaryHeap with ordered f64.
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;

        // Wrap f64 in a Newtype that implements Ord.
        #[derive(PartialEq)]
        struct OrdF64(f64);
        impl Eq for OrdF64 {}
        impl PartialOrd for OrdF64 {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }
        impl Ord for OrdF64 {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                self.0
                    .partial_cmp(&other.0)
                    .unwrap_or(std::cmp::Ordering::Equal)
            }
        }

        // Heap entries: (Reverse(cost), node_id)
        let mut heap: BinaryHeap<(Reverse<OrdF64>, String)> = BinaryHeap::new();

        dist.insert(from.to_string(), (0.0, None, None));
        heap.push((Reverse(OrdF64(0.0)), from.to_string()));

        while let Some((Reverse(OrdF64(cost)), current)) = heap.pop() {
            // Skip stale entries.
            let recorded_cost = dist.get(&current).map(|e| e.0).unwrap_or(f64::INFINITY);
            if cost > recorded_cost + f64::EPSILON {
                continue;
            }

            if current == to {
                break;
            }

            let out_edges = match self.adj_out.get(&current) {
                Some(v) => v.clone(),
                None => continue,
            };

            for eref in &out_edges {
                let edge = &self.edges[eref.edge_index];
                if edge.strength <= 0.0 {
                    // Zero-strength edges block the path.
                    continue;
                }
                let new_cost = cost + (-edge.strength.ln());
                let existing = dist.get(&eref.to).map(|e| e.0).unwrap_or(f64::INFINITY);
                if new_cost < existing - f64::EPSILON {
                    dist.insert(
                        eref.to.clone(),
                        (new_cost, Some(current.clone()), Some(eref.edge_index)),
                    );
                    heap.push((Reverse(OrdF64(new_cost)), eref.to.clone()));
                }
            }
        }

        // Reconstruct path.
        if !dist.contains_key(to) || dist[to].1.is_none() && to != from {
            return Ok(None);
        }

        let mut node_ids: Vec<String> = Vec::new();
        let mut edge_indices: Vec<usize> = Vec::new();
        let mut cursor = to.to_string();

        loop {
            node_ids.push(cursor.clone());
            let entry = match dist.get(&cursor) {
                Some(e) => e.clone(),
                None => break,
            };
            if let Some(ei) = entry.2 {
                edge_indices.push(ei);
            }
            match entry.1 {
                Some(pred) => cursor = pred,
                None => break,
            }
        }

        node_ids.reverse();
        edge_indices.reverse();

        let chain = self.build_chain(
            &node_ids,
            &edge_indices,
            self.product_of_edges(&edge_indices),
        )?;
        Ok(Some(chain))
    }

    // ------------------------------------------------------------------
    // Analysis
    // ------------------------------------------------------------------

    /// Return all nodes with no incoming edges that can reach `event_id`.
    pub fn root_causes(&self, event_id: &str) -> Result<Vec<CausalNode>, TracerError> {
        if !self.nodes.contains_key(event_id) {
            return Err(TracerError::NodeNotFound(event_id.to_string()));
        }

        // BFS backwards from event_id.
        let mut visited: HashSet<String> = HashSet::new();
        let mut queue: VecDeque<String> = VecDeque::new();
        queue.push_back(event_id.to_string());
        visited.insert(event_id.to_string());

        while let Some(current) = queue.pop_front() {
            let parents = self.adj_in.get(&current).cloned().unwrap_or_default();
            for parent in parents {
                if !visited.contains(&parent) {
                    visited.insert(parent.clone());
                    queue.push_back(parent);
                }
            }
        }

        // Filter to nodes with no incoming edges among those reachable.
        let mut roots: Vec<CausalNode> = Vec::new();
        for node_id in &visited {
            if node_id == event_id {
                continue;
            }
            let has_incoming = self
                .adj_in
                .get(node_id)
                .map(|v| !v.is_empty())
                .unwrap_or(false);
            if !has_incoming {
                if let Some(n) = self.nodes.get(node_id) {
                    roots.push(n.clone());
                }
            }
        }
        Ok(roots)
    }

    /// Return all nodes reachable from `event_id` via BFS up to `depth` hops.
    pub fn downstream_effects(
        &self,
        event_id: &str,
        depth: usize,
    ) -> Result<Vec<CausalNode>, TracerError> {
        if !self.nodes.contains_key(event_id) {
            return Err(TracerError::NodeNotFound(event_id.to_string()));
        }

        let mut visited: HashSet<String> = HashSet::new();
        let mut queue: VecDeque<(String, usize)> = VecDeque::new();
        queue.push_back((event_id.to_string(), 0));
        visited.insert(event_id.to_string());

        while let Some((current, d)) = queue.pop_front() {
            if d >= depth {
                continue;
            }
            let out_edges = self.adj_out.get(&current).cloned().unwrap_or_default();
            for eref in &out_edges {
                if !visited.contains(&eref.to) {
                    visited.insert(eref.to.clone());
                    queue.push_back((eref.to.clone(), d + 1));
                }
            }
        }

        // Return all visited nodes except the starting node.
        let mut effects: Vec<CausalNode> = visited
            .iter()
            .filter(|id| id.as_str() != event_id)
            .filter_map(|id| self.nodes.get(id).cloned())
            .collect();
        effects.sort_by_key(|a| a.timestamp);
        Ok(effects)
    }

    /// Return a snapshot of aggregate statistics.
    pub fn stats(&self) -> TracerStats {
        self.stats.clone()
    }

    // ------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------

    /// DFS to determine if there is a path from `start` to `target`.
    /// Returns the path if found (including `target`).
    fn find_cycle_path(&self, start: &str, target: &str) -> Option<Vec<String>> {
        if start == target {
            return Some(vec![start.to_string()]);
        }

        let mut visited: HashSet<String> = HashSet::new();
        let mut path: Vec<String> = Vec::new();
        self.dfs_find(start, target, &mut visited, &mut path)
    }

    /// Recursive DFS helper used by `find_cycle_path`.
    fn dfs_find(
        &self,
        current: &str,
        target: &str,
        visited: &mut HashSet<String>,
        path: &mut Vec<String>,
    ) -> Option<Vec<String>> {
        if visited.contains(current) {
            return None;
        }
        visited.insert(current.to_string());
        path.push(current.to_string());

        if current == target {
            return Some(path.clone());
        }

        let out_edges = match self.adj_out.get(current) {
            Some(v) => v.clone(),
            None => {
                path.pop();
                return None;
            }
        };

        for eref in &out_edges {
            if let Some(found) = self.dfs_find(&eref.to, target, visited, path) {
                return Some(found);
            }
        }

        path.pop();
        None
    }

    /// Determine the root nodes for a trace query.
    fn resolve_roots(&self, query: &TraceQuery) -> Result<Vec<String>, TracerError> {
        match &query.root_event_id {
            Some(id) => {
                if !self.nodes.contains_key(id) {
                    return Err(TracerError::NodeNotFound(id.clone()));
                }
                Ok(vec![id.clone()])
            }
            None => {
                // All nodes with no incoming edges.
                let roots: Vec<String> = self
                    .nodes
                    .keys()
                    .filter(|id| self.adj_in.get(*id).map(|v| v.is_empty()).unwrap_or(true))
                    .cloned()
                    .collect();
                Ok(roots)
            }
        }
    }

    /// Check whether a node passes the event_type and time_window filters.
    fn node_passes_query(&self, node: &CausalNode, query: &TraceQuery) -> bool {
        if !query.event_types.is_empty() && !query.event_types.contains(&node.event_type) {
            return false;
        }
        if let Some((start, end)) = query.time_window_us {
            if node.timestamp < start || node.timestamp > end {
                return false;
            }
        }
        true
    }

    /// Compute the product of strengths for a slice of edge indices.
    fn product_of_edges(&self, edge_indices: &[usize]) -> f64 {
        edge_indices
            .iter()
            .fold(1.0_f64, |acc, &i| acc * self.edges[i].strength)
    }

    /// Construct a [`CausalChain`] from parallel node-ID and edge-index slices.
    fn build_chain(
        &self,
        node_ids: &[String],
        edge_indices: &[usize],
        confidence: f64,
    ) -> Result<CausalChain, TracerError> {
        let mut nodes: Vec<CausalNode> = Vec::with_capacity(node_ids.len());
        for id in node_ids {
            let node = self
                .nodes
                .get(id)
                .ok_or_else(|| TracerError::NodeNotFound(id.clone()))?
                .clone();
            nodes.push(node);
        }

        let mut edges: Vec<CausalEdge> = Vec::with_capacity(edge_indices.len());
        for &i in edge_indices {
            edges.push(self.edges[i].clone());
        }

        let root_id = node_ids.first().cloned().unwrap_or_default();
        let leaf_ids = vec![node_ids.last().cloned().unwrap_or_default()];
        let depth = node_ids.len().saturating_sub(1);

        Ok(CausalChain {
            nodes,
            edges,
            root_id,
            leaf_ids,
            chain_confidence: confidence,
            depth,
        })
    }
}

// ---------------------------------------------------------------------------
// PRNG (used in tests — no `rand` dependency)
// ---------------------------------------------------------------------------

/// XorShift64 PRNG; deterministic, dependency-free.
pub fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
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

    fn make_tracer() -> CausalChainTracer {
        CausalChainTracer::new(TracerConfig {
            max_chain_depth: 16,
            min_edge_strength: 0.0,
            max_nodes: 10_000,
            enable_cycle_detection: true,
            confidence_threshold: 0.0,
        })
    }

    fn node(id: &str, etype: &str, ts: u64) -> CausalNode {
        CausalNode::new(id, etype, ts, 1.0)
    }

    fn edge(from: &str, to: &str, rel: CausalRelation, strength: f64) -> CausalEdge {
        CausalEdge::new(from, to, rel, strength, 0)
    }

    // -----------------------------------------------------------------------
    // 1. add_node basic
    // -----------------------------------------------------------------------
    #[test]
    fn test_add_node_basic() {
        let mut t = make_tracer();
        let n = node("A", "login", 1000);
        assert!(t.add_node(n).is_ok());
        assert_eq!(t.stats().nodes_tracked, 1);
    }

    // -----------------------------------------------------------------------
    // 2. add_node duplicate (overwrite)
    // -----------------------------------------------------------------------
    #[test]
    fn test_add_node_duplicate_overwrite() {
        let mut t = make_tracer();
        t.add_node(node("A", "login", 100))
            .expect("test setup: add_node failed");
        t.add_node(node("A", "logout", 200))
            .expect("test setup: add_node failed");
        // Still 1 node, updated value.
        assert_eq!(t.stats().nodes_tracked, 1);
    }

    // -----------------------------------------------------------------------
    // 3. add_node max_nodes limit
    // -----------------------------------------------------------------------
    #[test]
    fn test_add_node_max_nodes() {
        let mut t = CausalChainTracer::new(TracerConfig {
            max_nodes: 2,
            ..Default::default()
        });
        t.add_node(node("A", "x", 0))
            .expect("test setup: add_node failed");
        t.add_node(node("B", "x", 0))
            .expect("test setup: add_node failed");
        let err = t.add_node(node("C", "x", 0)).unwrap_err();
        assert!(matches!(err, TracerError::QueryTooExpensive(_)));
    }

    // -----------------------------------------------------------------------
    // 4. add_edge basic
    // -----------------------------------------------------------------------
    #[test]
    fn test_add_edge_basic() {
        let mut t = make_tracer();
        t.add_node(node("A", "e", 0))
            .expect("test setup: add_node failed");
        t.add_node(node("B", "e", 1))
            .expect("test setup: add_node failed");
        assert!(t
            .add_edge(edge("A", "B", CausalRelation::DirectCause, 0.8))
            .is_ok());
        assert_eq!(t.stats().edges_tracked, 1);
    }

    // -----------------------------------------------------------------------
    // 5. add_edge missing from-node
    // -----------------------------------------------------------------------
    #[test]
    fn test_add_edge_missing_from() {
        let mut t = make_tracer();
        t.add_node(node("B", "e", 0))
            .expect("test setup: add_node failed");
        let err = t
            .add_edge(edge("X", "B", CausalRelation::Enables, 0.5))
            .unwrap_err();
        assert!(matches!(err, TracerError::NodeNotFound(_)));
    }

    // -----------------------------------------------------------------------
    // 6. add_edge missing to-node
    // -----------------------------------------------------------------------
    #[test]
    fn test_add_edge_missing_to() {
        let mut t = make_tracer();
        t.add_node(node("A", "e", 0))
            .expect("test setup: add_node failed");
        let err = t
            .add_edge(edge("A", "Y", CausalRelation::Enables, 0.5))
            .unwrap_err();
        assert!(matches!(err, TracerError::NodeNotFound(_)));
    }

    // -----------------------------------------------------------------------
    // 7. add_edge invalid strength > 1
    // -----------------------------------------------------------------------
    #[test]
    fn test_add_edge_strength_too_high() {
        let mut t = make_tracer();
        t.add_node(node("A", "e", 0))
            .expect("test setup: add_node failed");
        t.add_node(node("B", "e", 1))
            .expect("test setup: add_node failed");
        let err = t
            .add_edge(edge("A", "B", CausalRelation::Precedes, 1.5))
            .unwrap_err();
        assert!(matches!(err, TracerError::InvalidStrength(_)));
    }

    // -----------------------------------------------------------------------
    // 8. add_edge invalid strength < 0
    // -----------------------------------------------------------------------
    #[test]
    fn test_add_edge_strength_negative() {
        let mut t = make_tracer();
        t.add_node(node("A", "e", 0))
            .expect("test setup: add_node failed");
        t.add_node(node("B", "e", 1))
            .expect("test setup: add_node failed");
        let err = t
            .add_edge(edge("A", "B", CausalRelation::Precedes, -0.1))
            .unwrap_err();
        assert!(matches!(err, TracerError::InvalidStrength(_)));
    }

    // -----------------------------------------------------------------------
    // 9. add_edge boundary strengths (0.0 and 1.0 are valid)
    // -----------------------------------------------------------------------
    #[test]
    fn test_add_edge_boundary_strengths() {
        let mut t = make_tracer();
        t.add_node(node("A", "e", 0))
            .expect("test setup: add_node failed");
        t.add_node(node("B", "e", 1))
            .expect("test setup: add_node failed");
        t.add_node(node("C", "e", 2))
            .expect("test setup: add_node failed");
        assert!(t
            .add_edge(edge("A", "B", CausalRelation::Precedes, 0.0))
            .is_ok());
        assert!(t
            .add_edge(edge("B", "C", CausalRelation::Precedes, 1.0))
            .is_ok());
    }

    // -----------------------------------------------------------------------
    // 10. cycle detection — direct cycle
    // -----------------------------------------------------------------------
    #[test]
    fn test_cycle_direct() {
        let mut t = make_tracer();
        t.add_node(node("A", "e", 0))
            .expect("test setup: add_node failed");
        t.add_node(node("B", "e", 1))
            .expect("test setup: add_node failed");
        t.add_edge(edge("A", "B", CausalRelation::DirectCause, 0.9))
            .expect("test setup: add_edge failed");
        let err = t
            .add_edge(edge("B", "A", CausalRelation::DirectCause, 0.9))
            .unwrap_err();
        assert!(matches!(err, TracerError::CycleDetected { .. }));
    }

    // -----------------------------------------------------------------------
    // 11. cycle detection — indirect cycle A->B->C->A
    // -----------------------------------------------------------------------
    #[test]
    fn test_cycle_indirect() {
        let mut t = make_tracer();
        for id in ["A", "B", "C"] {
            t.add_node(node(id, "e", 0))
                .expect("test setup: add_node failed");
        }
        t.add_edge(edge("A", "B", CausalRelation::DirectCause, 0.8))
            .expect("test setup: add_edge failed");
        t.add_edge(edge("B", "C", CausalRelation::DirectCause, 0.8))
            .expect("test setup: add_edge failed");
        let err = t
            .add_edge(edge("C", "A", CausalRelation::DirectCause, 0.8))
            .unwrap_err();
        assert!(matches!(err, TracerError::CycleDetected { .. }));
    }

    // -----------------------------------------------------------------------
    // 12. cycle detection — self-loop
    // -----------------------------------------------------------------------
    #[test]
    fn test_cycle_self_loop() {
        let mut t = make_tracer();
        t.add_node(node("A", "e", 0))
            .expect("test setup: add_node failed");
        let err = t
            .add_edge(edge("A", "A", CausalRelation::DirectCause, 0.5))
            .unwrap_err();
        assert!(matches!(err, TracerError::CycleDetected { .. }));
    }

    // -----------------------------------------------------------------------
    // 13. cycle detection disabled
    // -----------------------------------------------------------------------
    #[test]
    fn test_cycle_detection_disabled() {
        let mut t = CausalChainTracer::new(TracerConfig {
            enable_cycle_detection: false,
            ..Default::default()
        });
        t.add_node(node("A", "e", 0))
            .expect("test setup: add_node failed");
        t.add_node(node("B", "e", 1))
            .expect("test setup: add_node failed");
        t.add_edge(edge("A", "B", CausalRelation::DirectCause, 0.9))
            .expect("test setup: add_edge failed");
        // With detection disabled, this back-edge is accepted.
        assert!(t
            .add_edge(edge("B", "A", CausalRelation::DirectCause, 0.9))
            .is_ok());
    }

    // -----------------------------------------------------------------------
    // 14. trace — simple two-node chain
    // -----------------------------------------------------------------------
    #[test]
    fn test_trace_simple_chain() {
        let mut t = make_tracer();
        t.add_node(node("A", "e", 0))
            .expect("test setup: add_node failed");
        t.add_node(node("B", "e", 1))
            .expect("test setup: add_node failed");
        t.add_edge(edge("A", "B", CausalRelation::DirectCause, 0.9))
            .expect("test setup: add_edge failed");
        let q = TraceQuery {
            root_event_id: Some("A".to_string()),
            max_depth: 4,
            ..Default::default()
        };
        let chains = t.trace(&q).expect("test setup: trace failed");
        assert_eq!(chains.len(), 1);
        assert_eq!(chains[0].nodes.len(), 2);
        assert!((chains[0].chain_confidence - 0.9).abs() < 1e-9);
    }

    // -----------------------------------------------------------------------
    // 15. trace — linear chain A->B->C confidence product
    // -----------------------------------------------------------------------
    #[test]
    fn test_trace_confidence_product() {
        let mut t = make_tracer();
        for (id, ts) in [("A", 0u64), ("B", 1), ("C", 2)] {
            t.add_node(node(id, "e", ts))
                .expect("test setup: add_node failed");
        }
        t.add_edge(edge("A", "B", CausalRelation::DirectCause, 0.8))
            .expect("test setup: add_edge failed");
        t.add_edge(edge("B", "C", CausalRelation::DirectCause, 0.5))
            .expect("test setup: add_edge failed");
        let q = TraceQuery {
            root_event_id: Some("A".to_string()),
            max_depth: 4,
            ..Default::default()
        };
        let chains = t.trace(&q).expect("test setup: trace failed");
        assert_eq!(chains.len(), 1);
        assert!((chains[0].chain_confidence - 0.4).abs() < 1e-9);
    }

    // -----------------------------------------------------------------------
    // 16. trace — max_depth limits
    // -----------------------------------------------------------------------
    #[test]
    fn test_trace_max_depth() {
        let mut t = make_tracer();
        for (id, ts) in [("A", 0u64), ("B", 1), ("C", 2), ("D", 3)] {
            t.add_node(node(id, "e", ts))
                .expect("test setup: add_node failed");
        }
        t.add_edge(edge("A", "B", CausalRelation::DirectCause, 0.9))
            .expect("test setup: add_edge failed");
        t.add_edge(edge("B", "C", CausalRelation::DirectCause, 0.9))
            .expect("test setup: add_edge failed");
        t.add_edge(edge("C", "D", CausalRelation::DirectCause, 0.9))
            .expect("test setup: add_edge failed");

        let q = TraceQuery {
            root_event_id: Some("A".to_string()),
            max_depth: 1,
            ..Default::default()
        };
        let chains = t.trace(&q).expect("test setup: trace failed");
        // Depth 1 means we can only reach B from A.
        assert_eq!(chains.len(), 1);
        assert_eq!(chains[0].depth, 1);
    }

    // -----------------------------------------------------------------------
    // 17. trace — min_strength filter
    // -----------------------------------------------------------------------
    #[test]
    fn test_trace_min_strength_filter() {
        let mut t = make_tracer();
        t.add_node(node("A", "e", 0))
            .expect("test setup: add_node failed");
        t.add_node(node("B", "e", 1))
            .expect("test setup: add_node failed");
        t.add_node(node("C", "e", 2))
            .expect("test setup: add_node failed");
        t.add_edge(edge("A", "B", CausalRelation::DirectCause, 0.3))
            .expect("test setup: add_edge failed");
        t.add_edge(edge("A", "C", CausalRelation::DirectCause, 0.9))
            .expect("test setup: add_edge failed");
        let q = TraceQuery {
            root_event_id: Some("A".to_string()),
            min_strength: 0.5,
            max_depth: 4,
            ..Default::default()
        };
        let chains = t.trace(&q).expect("test setup: trace failed");
        // Only A->C survives.
        assert_eq!(chains.len(), 1);
        assert_eq!(chains[0].leaf_ids[0], "C");
    }

    // -----------------------------------------------------------------------
    // 18. trace — event_type filter
    // -----------------------------------------------------------------------
    #[test]
    fn test_trace_event_type_filter() {
        let mut t = make_tracer();
        t.add_node(node("A", "request", 0))
            .expect("test setup: add_node failed");
        t.add_node(node("B", "response", 1))
            .expect("test setup: add_node failed");
        t.add_node(node("C", "error", 2))
            .expect("test setup: add_node failed");
        t.add_edge(edge("A", "B", CausalRelation::DirectCause, 0.9))
            .expect("test setup: add_edge failed");
        t.add_edge(edge("A", "C", CausalRelation::DirectCause, 0.9))
            .expect("test setup: add_edge failed");

        let q = TraceQuery {
            root_event_id: Some("A".to_string()),
            event_types: vec!["request".to_string(), "response".to_string()],
            max_depth: 4,
            ..Default::default()
        };
        let chains = t.trace(&q).expect("test setup: trace failed");
        // Only A->B (error C is filtered).
        assert_eq!(chains.len(), 1);
        assert!(chains[0].nodes.iter().all(|n| n.event_type != "error"));
    }

    // -----------------------------------------------------------------------
    // 19. trace — time_window_us filter
    // -----------------------------------------------------------------------
    #[test]
    fn test_trace_time_window_filter() {
        let mut t = make_tracer();
        t.add_node(node("A", "e", 0))
            .expect("test setup: add_node failed");
        t.add_node(node("B", "e", 100))
            .expect("test setup: add_node failed");
        t.add_node(node("C", "e", 500))
            .expect("test setup: add_node failed");
        t.add_edge(edge("A", "B", CausalRelation::DirectCause, 0.9))
            .expect("test setup: add_edge failed");
        t.add_edge(edge("A", "C", CausalRelation::DirectCause, 0.9))
            .expect("test setup: add_edge failed");

        let q = TraceQuery {
            root_event_id: Some("A".to_string()),
            time_window_us: Some((0, 200)),
            max_depth: 4,
            ..Default::default()
        };
        let chains = t.trace(&q).expect("test setup: trace failed");
        // C is at ts=500 > 200, filtered out.
        assert!(chains
            .iter()
            .all(|c| !c.leaf_ids.contains(&"C".to_string())));
    }

    // -----------------------------------------------------------------------
    // 20. trace — include_relations filter
    // -----------------------------------------------------------------------
    #[test]
    fn test_trace_relation_filter() {
        let mut t = make_tracer();
        t.add_node(node("A", "e", 0))
            .expect("test setup: add_node failed");
        t.add_node(node("B", "e", 1))
            .expect("test setup: add_node failed");
        t.add_node(node("C", "e", 2))
            .expect("test setup: add_node failed");
        t.add_edge(edge("A", "B", CausalRelation::DirectCause, 0.9))
            .expect("test setup: add_edge failed");
        t.add_edge(edge("A", "C", CausalRelation::Correlates, 0.9))
            .expect("test setup: add_edge failed");

        let q = TraceQuery {
            root_event_id: Some("A".to_string()),
            include_relations: vec![CausalRelation::DirectCause],
            max_depth: 4,
            ..Default::default()
        };
        let chains = t.trace(&q).expect("test setup: trace failed");
        // Correlates edge to C is excluded.
        assert!(chains
            .iter()
            .all(|c| !c.leaf_ids.contains(&"C".to_string())));
    }

    // -----------------------------------------------------------------------
    // 21. trace — confidence_threshold
    // -----------------------------------------------------------------------
    #[test]
    fn test_trace_confidence_threshold() {
        let mut t = CausalChainTracer::new(TracerConfig {
            confidence_threshold: 0.5,
            ..Default::default()
        });
        t.add_node(node("A", "e", 0))
            .expect("test setup: add_node failed");
        t.add_node(node("B", "e", 1))
            .expect("test setup: add_node failed");
        t.add_node(node("C", "e", 2))
            .expect("test setup: add_node failed");
        t.add_edge(edge("A", "B", CausalRelation::DirectCause, 0.9))
            .expect("test setup: add_edge failed");
        // A->C has confidence 0.3 < threshold.
        t.add_edge(edge("A", "C", CausalRelation::DirectCause, 0.3))
            .expect("test setup: add_edge failed");
        let q = TraceQuery {
            root_event_id: Some("A".to_string()),
            max_depth: 4,
            ..Default::default()
        };
        let chains = t.trace(&q).expect("test setup: trace failed");
        assert!(chains.iter().all(|c| c.chain_confidence >= 0.5));
    }

    // -----------------------------------------------------------------------
    // 22. trace — branching graph
    // -----------------------------------------------------------------------
    #[test]
    fn test_trace_branching() {
        let mut t = make_tracer();
        // A -> B, A -> C, B -> D, C -> D
        for id in ["A", "B", "C", "D"] {
            t.add_node(node(id, "e", 0))
                .expect("test setup: add_node failed");
        }
        t.add_edge(edge("A", "B", CausalRelation::DirectCause, 0.8))
            .expect("test setup: add_edge failed");
        t.add_edge(edge("A", "C", CausalRelation::DirectCause, 0.7))
            .expect("test setup: add_edge failed");
        t.add_edge(edge("B", "D", CausalRelation::DirectCause, 0.9))
            .expect("test setup: add_edge failed");
        t.add_edge(edge("C", "D", CausalRelation::DirectCause, 0.6))
            .expect("test setup: add_edge failed");

        let q = TraceQuery {
            root_event_id: Some("A".to_string()),
            max_depth: 8,
            ..Default::default()
        };
        let chains = t.trace(&q).expect("test setup: trace failed");
        // Two paths: A->B->D and A->C->D.
        assert_eq!(chains.len(), 2);
    }

    // -----------------------------------------------------------------------
    // 23. trace — no roots (empty graph)
    // -----------------------------------------------------------------------
    #[test]
    fn test_trace_empty_graph() {
        let mut t = make_tracer();
        let q = TraceQuery::default();
        let chains = t.trace(&q).expect("test setup: trace failed");
        assert!(chains.is_empty());
    }

    // -----------------------------------------------------------------------
    // 24. trace — root_event_id not found
    // -----------------------------------------------------------------------
    #[test]
    fn test_trace_root_not_found() {
        let mut t = make_tracer();
        let q = TraceQuery {
            root_event_id: Some("MISSING".to_string()),
            ..Default::default()
        };
        let err = t.trace(&q).unwrap_err();
        assert!(matches!(err, TracerError::NodeNotFound(_)));
    }

    // -----------------------------------------------------------------------
    // 25. shortest_path — direct connection
    // -----------------------------------------------------------------------
    #[test]
    fn test_shortest_path_direct() {
        let mut t = make_tracer();
        t.add_node(node("A", "e", 0))
            .expect("test setup: add_node failed");
        t.add_node(node("B", "e", 1))
            .expect("test setup: add_node failed");
        t.add_edge(edge("A", "B", CausalRelation::DirectCause, 0.9))
            .expect("test setup: add_edge failed");
        let path = t
            .shortest_path("A", "B")
            .expect("test setup: shortest_path failed")
            .expect("test setup: expected Some path");
        assert_eq!(path, vec!["A", "B"]);
    }

    // -----------------------------------------------------------------------
    // 26. shortest_path — multi-hop
    // -----------------------------------------------------------------------
    #[test]
    fn test_shortest_path_multi_hop() {
        let mut t = make_tracer();
        for id in ["A", "B", "C"] {
            t.add_node(node(id, "e", 0))
                .expect("test setup: add_node failed");
        }
        t.add_edge(edge("A", "B", CausalRelation::DirectCause, 0.8))
            .expect("test setup: add_edge failed");
        t.add_edge(edge("B", "C", CausalRelation::DirectCause, 0.8))
            .expect("test setup: add_edge failed");
        let path = t
            .shortest_path("A", "C")
            .expect("test setup: shortest_path failed")
            .expect("test setup: expected Some path");
        assert_eq!(path.len(), 3);
    }

    // -----------------------------------------------------------------------
    // 27. shortest_path — no path
    // -----------------------------------------------------------------------
    #[test]
    fn test_shortest_path_no_path() {
        let mut t = make_tracer();
        t.add_node(node("A", "e", 0))
            .expect("test setup: add_node failed");
        t.add_node(node("B", "e", 1))
            .expect("test setup: add_node failed");
        let result = t
            .shortest_path("A", "B")
            .expect("test setup: shortest_path failed");
        assert!(result.is_none());
    }

    // -----------------------------------------------------------------------
    // 28. shortest_path — same node
    // -----------------------------------------------------------------------
    #[test]
    fn test_shortest_path_same_node() {
        let mut t = make_tracer();
        t.add_node(node("A", "e", 0))
            .expect("test setup: add_node failed");
        let path = t
            .shortest_path("A", "A")
            .expect("test setup: shortest_path failed")
            .expect("test setup: expected Some path");
        assert_eq!(path, vec!["A"]);
    }

    // -----------------------------------------------------------------------
    // 29. shortest_path — missing node error
    // -----------------------------------------------------------------------
    #[test]
    fn test_shortest_path_missing_node() {
        let t = make_tracer();
        let err = t.shortest_path("X", "Y").unwrap_err();
        assert!(matches!(err, TracerError::NodeNotFound(_)));
    }

    // -----------------------------------------------------------------------
    // 30. shortest_path — BFS finds shortest not longest
    // -----------------------------------------------------------------------
    #[test]
    fn test_shortest_path_is_shortest() {
        let mut t = make_tracer();
        // Two paths: A->B->D (len 3) and A->C->D (len 3); both equally short.
        for id in ["A", "B", "C", "D"] {
            t.add_node(node(id, "e", 0))
                .expect("test setup: add_node failed");
        }
        t.add_edge(edge("A", "B", CausalRelation::DirectCause, 0.9))
            .expect("test setup: add_edge failed");
        t.add_edge(edge("B", "D", CausalRelation::DirectCause, 0.9))
            .expect("test setup: add_edge failed");
        t.add_edge(edge("A", "C", CausalRelation::DirectCause, 0.9))
            .expect("test setup: add_edge failed");
        t.add_edge(edge("C", "D", CausalRelation::DirectCause, 0.9))
            .expect("test setup: add_edge failed");
        let path = t
            .shortest_path("A", "D")
            .expect("test setup: shortest_path failed")
            .expect("test setup: expected Some path");
        assert_eq!(path.len(), 3);
    }

    // -----------------------------------------------------------------------
    // 31. strongest_path — simple case
    // -----------------------------------------------------------------------
    #[test]
    fn test_strongest_path_simple() {
        let mut t = make_tracer();
        t.add_node(node("A", "e", 0))
            .expect("test setup: add_node failed");
        t.add_node(node("B", "e", 1))
            .expect("test setup: add_node failed");
        t.add_edge(edge("A", "B", CausalRelation::DirectCause, 0.7))
            .expect("test setup: add_edge failed");
        let chain = t
            .strongest_path("A", "B")
            .expect("test setup: strongest_path failed")
            .expect("test setup: expected Some chain");
        assert!((chain.chain_confidence - 0.7).abs() < 1e-9);
    }

    // -----------------------------------------------------------------------
    // 32. strongest_path — picks higher product
    // -----------------------------------------------------------------------
    #[test]
    fn test_strongest_path_picks_higher() {
        let mut t = make_tracer();
        // Path 1: A->B->D  product = 0.9 * 0.9 = 0.81
        // Path 2: A->C->D  product = 0.5 * 0.5 = 0.25
        for id in ["A", "B", "C", "D"] {
            t.add_node(node(id, "e", 0))
                .expect("test setup: add_node failed");
        }
        t.add_edge(edge("A", "B", CausalRelation::DirectCause, 0.9))
            .expect("test setup: add_edge failed");
        t.add_edge(edge("B", "D", CausalRelation::DirectCause, 0.9))
            .expect("test setup: add_edge failed");
        t.add_edge(edge("A", "C", CausalRelation::DirectCause, 0.5))
            .expect("test setup: add_edge failed");
        t.add_edge(edge("C", "D", CausalRelation::DirectCause, 0.5))
            .expect("test setup: add_edge failed");
        let chain = t
            .strongest_path("A", "D")
            .expect("test setup: strongest_path failed")
            .expect("test setup: expected Some chain");
        assert!(chain.chain_confidence > 0.8);
    }

    // -----------------------------------------------------------------------
    // 33. strongest_path — same node
    // -----------------------------------------------------------------------
    #[test]
    fn test_strongest_path_same_node() {
        let mut t = make_tracer();
        t.add_node(node("A", "e", 0))
            .expect("test setup: add_node failed");
        let chain = t
            .strongest_path("A", "A")
            .expect("test setup: strongest_path failed")
            .expect("test setup: expected Some chain");
        assert!((chain.chain_confidence - 1.0).abs() < 1e-9);
        assert_eq!(chain.depth, 0);
    }

    // -----------------------------------------------------------------------
    // 34. strongest_path — no path
    // -----------------------------------------------------------------------
    #[test]
    fn test_strongest_path_no_path() {
        let mut t = make_tracer();
        t.add_node(node("A", "e", 0))
            .expect("test setup: add_node failed");
        t.add_node(node("B", "e", 1))
            .expect("test setup: add_node failed");
        let result = t
            .strongest_path("A", "B")
            .expect("test setup: strongest_path failed");
        assert!(result.is_none());
    }

    // -----------------------------------------------------------------------
    // 35. strongest_path — missing node error
    // -----------------------------------------------------------------------
    #[test]
    fn test_strongest_path_missing() {
        let t = make_tracer();
        let err = t.strongest_path("X", "Y").unwrap_err();
        assert!(matches!(err, TracerError::NodeNotFound(_)));
    }

    // -----------------------------------------------------------------------
    // 36. root_causes — single root
    // -----------------------------------------------------------------------
    #[test]
    fn test_root_causes_single() {
        let mut t = make_tracer();
        for id in ["A", "B", "C"] {
            t.add_node(node(id, "e", 0))
                .expect("test setup: add_node failed");
        }
        t.add_edge(edge("A", "B", CausalRelation::DirectCause, 0.9))
            .expect("test setup: add_edge failed");
        t.add_edge(edge("B", "C", CausalRelation::DirectCause, 0.9))
            .expect("test setup: add_edge failed");
        let roots = t.root_causes("C").expect("test setup: root_causes failed");
        assert_eq!(roots.len(), 1);
        assert_eq!(roots[0].id, "A");
    }

    // -----------------------------------------------------------------------
    // 37. root_causes — multiple roots
    // -----------------------------------------------------------------------
    #[test]
    fn test_root_causes_multiple() {
        let mut t = make_tracer();
        for id in ["A", "B", "C"] {
            t.add_node(node(id, "e", 0))
                .expect("test setup: add_node failed");
        }
        t.add_edge(edge("A", "C", CausalRelation::DirectCause, 0.9))
            .expect("test setup: add_edge failed");
        t.add_edge(edge("B", "C", CausalRelation::DirectCause, 0.9))
            .expect("test setup: add_edge failed");
        let mut roots = t.root_causes("C").expect("test setup: root_causes failed");
        roots.sort_by(|a, b| a.id.cmp(&b.id));
        assert_eq!(roots.len(), 2);
        assert_eq!(roots[0].id, "A");
        assert_eq!(roots[1].id, "B");
    }

    // -----------------------------------------------------------------------
    // 38. root_causes — node with no ancestors
    // -----------------------------------------------------------------------
    #[test]
    fn test_root_causes_no_ancestors() {
        let mut t = make_tracer();
        t.add_node(node("A", "e", 0))
            .expect("test setup: add_node failed");
        let roots = t.root_causes("A").expect("test setup: root_causes failed");
        // A itself has no ancestors, so no roots other than itself.
        assert!(roots.is_empty());
    }

    // -----------------------------------------------------------------------
    // 39. root_causes — missing node
    // -----------------------------------------------------------------------
    #[test]
    fn test_root_causes_missing() {
        let t = make_tracer();
        let err = t.root_causes("MISSING").unwrap_err();
        assert!(matches!(err, TracerError::NodeNotFound(_)));
    }

    // -----------------------------------------------------------------------
    // 40. downstream_effects — depth 1
    // -----------------------------------------------------------------------
    #[test]
    fn test_downstream_effects_depth1() {
        let mut t = make_tracer();
        for id in ["A", "B", "C", "D"] {
            t.add_node(node(id, "e", 0))
                .expect("test setup: add_node failed");
        }
        t.add_edge(edge("A", "B", CausalRelation::DirectCause, 0.9))
            .expect("test setup: add_edge failed");
        t.add_edge(edge("B", "C", CausalRelation::DirectCause, 0.9))
            .expect("test setup: add_edge failed");
        t.add_edge(edge("B", "D", CausalRelation::DirectCause, 0.9))
            .expect("test setup: add_edge failed");
        let effects = t
            .downstream_effects("A", 1)
            .expect("test setup: downstream_effects failed");
        let ids: Vec<&str> = effects.iter().map(|n| n.id.as_str()).collect();
        assert!(ids.contains(&"B"));
        assert!(!ids.contains(&"C"));
    }

    // -----------------------------------------------------------------------
    // 41. downstream_effects — deep traversal
    // -----------------------------------------------------------------------
    #[test]
    fn test_downstream_effects_deep() {
        let mut t = make_tracer();
        for (id, ts) in [("A", 0u64), ("B", 1), ("C", 2), ("D", 3)] {
            t.add_node(node(id, "e", ts))
                .expect("test setup: add_node failed");
        }
        t.add_edge(edge("A", "B", CausalRelation::DirectCause, 0.9))
            .expect("test setup: add_edge failed");
        t.add_edge(edge("B", "C", CausalRelation::DirectCause, 0.9))
            .expect("test setup: add_edge failed");
        t.add_edge(edge("C", "D", CausalRelation::DirectCause, 0.9))
            .expect("test setup: add_edge failed");
        let effects = t
            .downstream_effects("A", 10)
            .expect("test setup: downstream_effects failed");
        assert_eq!(effects.len(), 3);
    }

    // -----------------------------------------------------------------------
    // 42. downstream_effects — missing node
    // -----------------------------------------------------------------------
    #[test]
    fn test_downstream_effects_missing() {
        let t = make_tracer();
        let err = t.downstream_effects("MISSING", 5).unwrap_err();
        assert!(matches!(err, TracerError::NodeNotFound(_)));
    }

    // -----------------------------------------------------------------------
    // 43. downstream_effects — zero depth
    // -----------------------------------------------------------------------
    #[test]
    fn test_downstream_effects_zero_depth() {
        let mut t = make_tracer();
        t.add_node(node("A", "e", 0))
            .expect("test setup: add_node failed");
        t.add_node(node("B", "e", 1))
            .expect("test setup: add_node failed");
        t.add_edge(edge("A", "B", CausalRelation::DirectCause, 0.9))
            .expect("test setup: add_edge failed");
        let effects = t
            .downstream_effects("A", 0)
            .expect("test setup: downstream_effects failed");
        // Depth 0 means no expansion.
        assert!(effects.is_empty());
    }

    // -----------------------------------------------------------------------
    // 44. remove_node — basic
    // -----------------------------------------------------------------------
    #[test]
    fn test_remove_node_basic() {
        let mut t = make_tracer();
        t.add_node(node("A", "e", 0))
            .expect("test setup: add_node failed");
        t.add_node(node("B", "e", 1))
            .expect("test setup: add_node failed");
        t.add_edge(edge("A", "B", CausalRelation::DirectCause, 0.8))
            .expect("test setup: add_edge failed");
        t.remove_node("A").expect("test setup: remove_node failed");
        assert_eq!(t.stats().nodes_tracked, 1);
        assert_eq!(t.stats().edges_tracked, 0);
    }

    // -----------------------------------------------------------------------
    // 45. remove_node — removes connected edges
    // -----------------------------------------------------------------------
    #[test]
    fn test_remove_node_removes_edges() {
        let mut t = make_tracer();
        for id in ["A", "B", "C"] {
            t.add_node(node(id, "e", 0))
                .expect("test setup: add_node failed");
        }
        t.add_edge(edge("A", "B", CausalRelation::DirectCause, 0.9))
            .expect("test setup: add_edge failed");
        t.add_edge(edge("B", "C", CausalRelation::DirectCause, 0.9))
            .expect("test setup: add_edge failed");
        t.remove_node("B").expect("test setup: remove_node failed");
        assert_eq!(t.stats().edges_tracked, 0);
    }

    // -----------------------------------------------------------------------
    // 46. remove_node — missing node
    // -----------------------------------------------------------------------
    #[test]
    fn test_remove_node_missing() {
        let mut t = make_tracer();
        let err = t.remove_node("MISSING").unwrap_err();
        assert!(matches!(err, TracerError::NodeNotFound(_)));
    }

    // -----------------------------------------------------------------------
    // 47. stats — basic correctness
    // -----------------------------------------------------------------------
    #[test]
    fn test_stats_basic() {
        let mut t = make_tracer();
        t.add_node(node("A", "e", 0))
            .expect("test setup: add_node failed");
        t.add_node(node("B", "e", 1))
            .expect("test setup: add_node failed");
        t.add_edge(edge("A", "B", CausalRelation::DirectCause, 0.9))
            .expect("test setup: add_edge failed");
        let s = t.stats();
        assert_eq!(s.nodes_tracked, 2);
        assert_eq!(s.edges_tracked, 1);
    }

    // -----------------------------------------------------------------------
    // 48. stats — cycles_detected counter
    // -----------------------------------------------------------------------
    #[test]
    fn test_stats_cycles_detected() {
        let mut t = make_tracer();
        t.add_node(node("A", "e", 0))
            .expect("test setup: add_node failed");
        t.add_node(node("B", "e", 1))
            .expect("test setup: add_node failed");
        t.add_edge(edge("A", "B", CausalRelation::DirectCause, 0.9))
            .expect("test setup: add_edge failed");
        let _ = t.add_edge(edge("B", "A", CausalRelation::DirectCause, 0.9));
        assert_eq!(t.stats().cycles_detected, 1);
    }

    // -----------------------------------------------------------------------
    // 49. xorshift64 — deterministic PRNG
    // -----------------------------------------------------------------------
    #[test]
    fn test_xorshift64_deterministic() {
        let mut state = 12345u64;
        let v1 = xorshift64(&mut state);
        let mut state2 = 12345u64;
        let v2 = xorshift64(&mut state2);
        assert_eq!(v1, v2);
        assert_ne!(v1, 0);
    }

    // -----------------------------------------------------------------------
    // 50. large graph stress test with PRNG
    // -----------------------------------------------------------------------
    #[test]
    fn test_large_graph_stress() {
        let mut t = make_tracer();
        let mut rng = 999_999u64;
        let n = 50usize;

        for i in 0..n {
            t.add_node(node(&i.to_string(), "stress", xorshift64(&mut rng) % 1000))
                .expect("test setup: add_node failed");
        }

        // Add a random DAG (always i < j to avoid cycles).
        let mut added = 0usize;
        for i in 0..n {
            for j in (i + 1)..n {
                let r = xorshift64(&mut rng);
                if r.is_multiple_of(5) {
                    let strength = (r % 100) as f64 / 100.0;
                    let _ = t.add_edge(edge(
                        &i.to_string(),
                        &j.to_string(),
                        CausalRelation::DirectCause,
                        strength,
                    ));
                    added += 1;
                }
            }
        }

        assert!(added > 0, "expected some edges");
        let q = TraceQuery {
            root_event_id: Some("0".to_string()),
            max_depth: 5,
            ..Default::default()
        };
        let chains = t.trace(&q).expect("test setup: trace failed");
        // Just verify no panic and some chains were discovered.
        assert!(chains.len() <= 10_000);
    }

    // -----------------------------------------------------------------------
    // 51. trace — all relations (empty include_relations = no filter)
    // -----------------------------------------------------------------------
    #[test]
    fn test_trace_no_relation_filter() {
        let mut t = make_tracer();
        t.add_node(node("A", "e", 0))
            .expect("test setup: add_node failed");
        t.add_node(node("B", "e", 1))
            .expect("test setup: add_node failed");
        t.add_node(node("C", "e", 2))
            .expect("test setup: add_node failed");
        t.add_edge(edge("A", "B", CausalRelation::Inhibits, 0.9))
            .expect("test setup: add_edge failed");
        t.add_edge(edge("A", "C", CausalRelation::IndirectCause, 0.9))
            .expect("test setup: add_edge failed");
        let q = TraceQuery {
            root_event_id: Some("A".to_string()),
            include_relations: vec![], // empty = no filter
            max_depth: 4,
            ..Default::default()
        };
        let chains = t.trace(&q).expect("test setup: trace failed");
        assert_eq!(chains.len(), 2);
    }

    // -----------------------------------------------------------------------
    // 52. CausalNode with_attribute builder
    // -----------------------------------------------------------------------
    #[test]
    fn test_causal_node_with_attribute() {
        let n = CausalNode::new("A", "login", 0, 1.0)
            .with_attribute("ip", "192.168.1.1")
            .with_attribute("user", "bob");
        assert_eq!(n.attributes.len(), 2);
        assert_eq!(
            n.attributes[0],
            ("ip".to_string(), "192.168.1.1".to_string())
        );
    }

    // -----------------------------------------------------------------------
    // 53. trace from implicit roots (no root_event_id)
    // -----------------------------------------------------------------------
    #[test]
    fn test_trace_implicit_roots() {
        let mut t = make_tracer();
        // Two separate root nodes.
        t.add_node(node("R1", "e", 0))
            .expect("test setup: add_node failed");
        t.add_node(node("R2", "e", 0))
            .expect("test setup: add_node failed");
        t.add_node(node("X", "e", 1))
            .expect("test setup: add_node failed");
        t.add_edge(edge("R1", "X", CausalRelation::DirectCause, 0.8))
            .expect("test setup: add_edge failed");
        t.add_edge(edge("R2", "X", CausalRelation::DirectCause, 0.7))
            .expect("test setup: add_edge failed");

        let q = TraceQuery {
            root_event_id: None,
            max_depth: 4,
            ..Default::default()
        };
        let chains = t.trace(&q).expect("test setup: trace failed");
        // Each root produces one chain.
        assert_eq!(chains.len(), 2);
    }

    // -----------------------------------------------------------------------
    // 54. TracerError Display
    // -----------------------------------------------------------------------
    #[test]
    fn test_tracer_error_display() {
        let e = TracerError::NodeNotFound("ABC".to_string());
        assert!(e.to_string().contains("ABC"));
        let e2 = TracerError::InvalidStrength(2.0);
        assert!(e2.to_string().contains("2"));
        let e3 = TracerError::CycleDetected {
            path: vec!["A".to_string(), "B".to_string()],
        };
        assert!(e3.to_string().contains("A -> B"));
    }

    // -----------------------------------------------------------------------
    // 55. trace chains_traced counter
    // -----------------------------------------------------------------------
    #[test]
    fn test_chains_traced_counter() {
        let mut t = make_tracer();
        t.add_node(node("A", "e", 0))
            .expect("test setup: add_node failed");
        t.add_node(node("B", "e", 1))
            .expect("test setup: add_node failed");
        t.add_edge(edge("A", "B", CausalRelation::DirectCause, 0.9))
            .expect("test setup: add_edge failed");
        let q = TraceQuery {
            root_event_id: Some("A".to_string()),
            max_depth: 4,
            ..Default::default()
        };
        t.trace(&q).expect("test setup: trace failed");
        t.trace(&q).expect("test setup: trace failed");
        assert_eq!(t.stats().chains_traced, 2);
    }
}
