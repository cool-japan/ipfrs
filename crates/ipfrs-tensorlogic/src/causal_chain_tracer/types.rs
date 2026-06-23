//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::{HashMap, HashSet, VecDeque};

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
/// Production-quality causal chain tracer for event sequences.
///
/// Maintains a directed graph of [`CausalNode`]s connected by [`CausalEdge`]s and
/// exposes high-level methods for chain tracing, path finding, root-cause
/// identification, and downstream-effect enumeration.
pub struct CausalChainTracer {
    pub(super) config: TracerConfig,
    pub(super) nodes: HashMap<String, CausalNode>,
    pub(super) edges: Vec<CausalEdge>,
    /// Forward adjacency: node_id -> list of outgoing edge refs.
    pub(super) adj_out: HashMap<String, Vec<EdgeRef>>,
    /// Backward adjacency: node_id -> list of incoming node IDs.
    pub(super) adj_in: HashMap<String, Vec<String>>,
    pub(super) stats: TracerStats,
}
impl CausalChainTracer {
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
    /// Insert a node into the graph.
    ///
    /// Returns [`TracerError::QueryTooExpensive`] when `max_nodes` would be exceeded.
    pub fn add_node(&mut self, node: CausalNode) -> Result<(), TracerError> {
        if self.nodes.len() >= self.config.max_nodes && !self.nodes.contains_key(&node.id) {
            return Err(TracerError::QueryTooExpensive(self.nodes.len()));
        }
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
            if let Some(cycle_path) = self.find_cycle_path(&edge.to_id, &edge.from_id) {
                self.stats.cycles_detected += 1;
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
        let edge_indices_to_remove: HashSet<usize> = self
            .edges
            .iter()
            .enumerate()
            .filter(|(_, e)| e.from_id == id || e.to_id == id)
            .map(|(i, _)| i)
            .collect();
        let mut new_edges: Vec<CausalEdge> = Vec::new();
        let mut old_to_new: HashMap<usize, usize> = HashMap::new();
        for (old_idx, edge) in self.edges.drain(..).enumerate() {
            if !edge_indices_to_remove.contains(&old_idx) {
                old_to_new.insert(old_idx, new_edges.len());
                new_edges.push(edge);
            }
        }
        self.edges = new_edges;
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
            let root_node = self
                .nodes
                .get(root_id)
                .ok_or_else(|| TracerError::NodeNotFound(root_id.clone()))?;
            if !self.node_passes_query(root_node, query) {
                continue;
            }
            let mut stack: Vec<(String, Vec<String>, Vec<usize>, f64)> =
                vec![(root_id.clone(), vec![root_id.clone()], Vec::new(), 1.0_f64)];
            while let Some((current_id, node_path, edge_path, confidence)) = stack.pop() {
                let depth = node_path.len() - 1;
                let out_edges = self.adj_out.get(&current_id).cloned().unwrap_or_default();
                let valid_next: Vec<(String, usize, f64)> = out_edges
                    .iter()
                    .filter_map(|eref| {
                        let edge = &self.edges[eref.edge_index];
                        if edge.strength < query.min_strength {
                            return None;
                        }
                        if !query.include_relations.is_empty()
                            && !query.include_relations.contains(&edge.relation)
                        {
                            return None;
                        }
                        let target = self.nodes.get(&eref.to)?;
                        if !self.node_passes_query(target, query) {
                            return None;
                        }
                        if node_path.contains(&eref.to) {
                            return None;
                        }
                        Some((eref.to.clone(), eref.edge_index, edge.strength))
                    })
                    .collect();
                let is_leaf = valid_next.is_empty() || depth >= max_depth;
                if is_leaf {
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
        self.stats.chains_traced += all_chains.len();
        if !all_chains.is_empty() {
            let total_depth: usize = all_chains.iter().map(|c| c.depth).sum();
            self.stats.avg_chain_depth = total_depth as f64 / all_chains.len() as f64;
        }
        Ok(all_chains)
    }
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
        let mut visited: HashSet<String> = HashSet::new();
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
        let mut dist: HashMap<String, (f64, Option<String>, Option<usize>)> = HashMap::new();
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;
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
        let mut heap: BinaryHeap<(Reverse<OrdF64>, String)> = BinaryHeap::new();
        dist.insert(from.to_string(), (0.0, None, None));
        heap.push((Reverse(OrdF64(0.0)), from.to_string()));
        while let Some((Reverse(OrdF64(cost)), current)) = heap.pop() {
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
    /// Return all nodes with no incoming edges that can reach `event_id`.
    pub fn root_causes(&self, event_id: &str) -> Result<Vec<CausalNode>, TracerError> {
        if !self.nodes.contains_key(event_id) {
            return Err(TracerError::NodeNotFound(event_id.to_string()));
        }
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
    /// DFS to determine if there is a path from `start` to `target`.
    /// Returns the path if found (including `target`).
    pub(super) fn find_cycle_path(&self, start: &str, target: &str) -> Option<Vec<String>> {
        if start == target {
            return Some(vec![start.to_string()]);
        }
        let mut visited: HashSet<String> = HashSet::new();
        let mut path: Vec<String> = Vec::new();
        self.dfs_find(start, target, &mut visited, &mut path)
    }
    /// Recursive DFS helper used by `find_cycle_path`.
    pub(super) fn dfs_find(
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
    pub(super) fn resolve_roots(&self, query: &TraceQuery) -> Result<Vec<String>, TracerError> {
        match &query.root_event_id {
            Some(id) => {
                if !self.nodes.contains_key(id) {
                    return Err(TracerError::NodeNotFound(id.clone()));
                }
                Ok(vec![id.clone()])
            }
            None => {
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
    pub(super) fn node_passes_query(&self, node: &CausalNode, query: &TraceQuery) -> bool {
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
    pub(super) fn product_of_edges(&self, edge_indices: &[usize]) -> f64 {
        edge_indices
            .iter()
            .fold(1.0_f64, |acc, &i| acc * self.edges[i].strength)
    }
    /// Construct a [`CausalChain`] from parallel node-ID and edge-index slices.
    pub(super) fn build_chain(
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
/// Lightweight edge reference stored in the adjacency list.
#[derive(Debug, Clone)]
pub(super) struct EdgeRef {
    pub(super) to: String,
    pub(super) edge_index: usize,
}
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
