//! Causal Inference Engine — do-calculus, interventional distributions, and
//! counterfactual reasoning over Gaussian structural equation models.
//!
//! # Overview
//!
//! This module implements a full causal inference pipeline built on directed
//! acyclic graphs (DAGs) with Gaussian node semantics.  The design follows
//! Pearl's do-calculus framework:
//!
//! 1. **Structural Causal Model (SCM)** — each node is parameterised by a mean
//!    and a variance under a Gaussian structural equation model.
//! 2. **Interventional inference** (`do_calculus`) — cuts incoming edges to the
//!    intervened node and propagates the fixed value through all directed paths
//!    to the target, accumulating linear causal effects.
//! 3. **Counterfactual inference** (`counterfactual`) — applies an intervention
//!    and conditions on observed evidence by adding a weighted correction from
//!    each evidence node to the target.
//! 4. **Average Causal Effect** (`average_causal_effect`) — contrasts
//!    interventional distributions to quantify treatment effects.
//! 5. **d-separation** — checks whether two nodes are conditionally independent
//!    given a set of observed variables.
//! 6. **Backdoor paths** — enumerates confounding paths for identifiability
//!    analysis.

use std::collections::{HashMap, HashSet, VecDeque};

/// Opaque identifier for a node inside a [`CausalGraph`].
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct CausalNodeId(pub String);

impl CausalNodeId {
    /// Create a new identifier from any string-like value.
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }

    /// Borrow the inner string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for CausalNodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

// ── Edge ──────────────────────────────────────────────────────────────────────

/// Semantic classification for an edge in a [`CausalGraph`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CausalEdgeType {
    /// A direct causal pathway (X → Y).
    Direct,
    /// A common-cause (bi-directed) relationship.
    Confounded,
    /// An open backdoor path that may introduce confounding bias.
    Backdoor,
    /// An instrumental variable link (Z → X, Z ⊥ Y | X).
    Instrumental,
}

/// A directed edge in the causal graph carrying a linear strength coefficient.
#[derive(Debug, Clone)]
pub struct CausalEdge {
    /// Source node of the edge.
    pub from: CausalNodeId,
    /// Destination node of the edge.
    pub to: CausalNodeId,
    /// Signed linear coefficient representing causal strength (may be negative).
    pub strength: f64,
    /// Semantic category of the relationship.
    pub edge_type: CausalEdgeType,
}

impl CausalEdge {
    /// Construct a new edge with default type [`CausalEdgeType::Direct`].
    pub fn direct(from: impl Into<String>, to: impl Into<String>, strength: f64) -> Self {
        Self {
            from: CausalNodeId::new(from),
            to: CausalNodeId::new(to),
            strength,
            edge_type: CausalEdgeType::Direct,
        }
    }
}

// ── Node ──────────────────────────────────────────────────────────────────────

/// A variable in the structural causal model parameterised by a Gaussian prior.
#[derive(Debug, Clone)]
pub struct CausalNode {
    /// Unique identifier of this node.
    pub id: CausalNodeId,
    /// Direct causal parents (variables whose values influence this node).
    pub parents: Vec<CausalNodeId>,
    /// Direct causal children (variables this node influences).
    pub children: Vec<CausalNodeId>,
    /// Prior (or marginal) mean of the Gaussian distribution over this variable.
    pub mean: f64,
    /// Prior (or marginal) variance of the Gaussian distribution over this variable.
    pub variance: f64,
}

impl CausalNode {
    /// Create a standalone node (no edges yet) with the given Gaussian parameters.
    pub fn new(id: impl Into<String>, mean: f64, variance: f64) -> Self {
        Self {
            id: CausalNodeId::new(id),
            parents: Vec::new(),
            children: Vec::new(),
            mean,
            variance,
        }
    }
}

// ── Graph ─────────────────────────────────────────────────────────────────────

/// The underlying directed acyclic graph over causal nodes.
#[derive(Debug, Default, Clone)]
pub struct CausalGraph {
    /// All nodes, keyed by their identifier.
    pub nodes: HashMap<CausalNodeId, CausalNode>,
    /// All directed edges in the graph.
    pub edges: Vec<CausalEdge>,
}

// ── Do-calculus primitives ────────────────────────────────────────────────────

/// A hard intervention: set node `node` to exactly `value` (written do(X = value)).
#[derive(Debug, Clone)]
pub struct Intervention {
    /// The node being intervened upon.
    pub node: CausalNodeId,
    /// The value assigned by the intervention.
    pub value: f64,
}

impl Intervention {
    /// Construct a new intervention.
    pub fn new(node: impl Into<String>, value: f64) -> Self {
        Self {
            node: CausalNodeId::new(node),
            value,
        }
    }
}

/// A counterfactual query: what would `target` be if we had applied
/// `intervention`, given that we observed `evidence`?
#[derive(Debug, Clone)]
pub struct CounterfactualQuery {
    /// The variable whose counterfactual value we wish to estimate.
    pub target: CausalNodeId,
    /// The hypothetical intervention.
    pub intervention: Intervention,
    /// Observed values for conditioning variables.
    pub evidence: HashMap<CausalNodeId, f64>,
}

/// The result returned by an inference query.
#[derive(Debug, Clone)]
pub struct InferenceResult {
    /// The variable that was queried.
    pub target: CausalNodeId,
    /// Posterior mean under the intervention / evidence.
    pub mean: f64,
    /// Posterior variance under the intervention / evidence.
    pub variance: f64,
    /// Confidence in [0, 1] — fraction of target variance explained by causal paths.
    pub confidence: f64,
    /// All interventions that were applied to produce this result.
    pub interventions_applied: Vec<Intervention>,
}

// ── Error type ────────────────────────────────────────────────────────────────

/// Errors that can be produced by the [`CausalInferenceEngine`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CausalError {
    /// A node with the same identifier already exists in the graph.
    NodeAlreadyExists(String),
    /// A referenced node does not exist in the graph.
    NodeNotFound(String),
    /// Adding the requested edge would create a directed cycle.
    CycleDetected,
    /// The edge specification is otherwise invalid.
    InvalidEdge(String),
}

impl std::fmt::Display for CausalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NodeAlreadyExists(id) => write!(f, "node already exists: {id}"),
            Self::NodeNotFound(id) => write!(f, "node not found: {id}"),
            Self::CycleDetected => write!(f, "adding this edge would create a cycle"),
            Self::InvalidEdge(msg) => write!(f, "invalid edge: {msg}"),
        }
    }
}

impl std::error::Error for CausalError {}

// ── Statistics ────────────────────────────────────────────────────────────────

/// Summary statistics about the structure of a [`CausalGraph`].
#[derive(Debug, Clone)]
pub struct CausalStats {
    /// Total number of nodes in the graph.
    pub node_count: usize,
    /// Total number of edges in the graph.
    pub edge_count: usize,
    /// Mean number of children per node.
    pub avg_children: f64,
    /// Length of the longest path (in hops) from any root to any leaf.
    pub max_depth: usize,
}

// ── Engine ────────────────────────────────────────────────────────────────────

/// Production-grade causal inference engine.
///
/// # Example
///
/// ```
/// use ipfrs_tensorlogic::{
///     CausalInferenceEngine, CausalNode, CausalEdge, Intervention,
/// };
///
/// let mut engine = CausalInferenceEngine::new(10);
/// engine.add_node(CausalNode::new("X", 0.0, 1.0)).expect("example: should succeed in docs");
/// engine.add_node(CausalNode::new("Y", 0.0, 1.0)).expect("example: should succeed in docs");
/// engine.add_edge(CausalEdge::direct("X", "Y", 0.8)).expect("example: should succeed in docs");
///
/// let result = engine.do_calculus(
///     &Intervention::new("X", 1.0),
///     &ipfrs_tensorlogic::CausalNodeId::new("Y"),
/// );
/// assert!((result.mean - 0.8).abs() < 1e-9);
/// ```
#[derive(Debug)]
pub struct CausalInferenceEngine {
    /// The underlying causal graph.
    pub graph: CausalGraph,
    /// Maximum number of hops considered when enumerating paths.
    pub max_path_length: usize,
}

impl CausalInferenceEngine {
    // ── Construction ──────────────────────────────────────────────────────────

    /// Create a new engine with an empty graph and the given path-length limit.
    pub fn new(max_path_length: usize) -> Self {
        Self {
            graph: CausalGraph::default(),
            max_path_length,
        }
    }

    // ── Mutation ──────────────────────────────────────────────────────────────

    /// Add a node to the graph.
    ///
    /// Returns [`CausalError::NodeAlreadyExists`] if the identifier is taken.
    pub fn add_node(&mut self, node: CausalNode) -> Result<(), CausalError> {
        if self.graph.nodes.contains_key(&node.id) {
            return Err(CausalError::NodeAlreadyExists(node.id.0.clone()));
        }
        self.graph.nodes.insert(node.id.clone(), node);
        Ok(())
    }

    /// Add a directed edge to the graph.
    ///
    /// Both endpoints must already exist, and the edge must not introduce a
    /// directed cycle.  On success the parent/child lists of both endpoints are
    /// updated in-place.
    pub fn add_edge(&mut self, edge: CausalEdge) -> Result<(), CausalError> {
        if !self.graph.nodes.contains_key(&edge.from) {
            return Err(CausalError::NodeNotFound(edge.from.0.clone()));
        }
        if !self.graph.nodes.contains_key(&edge.to) {
            return Err(CausalError::NodeNotFound(edge.to.0.clone()));
        }
        if edge.from == edge.to {
            return Err(CausalError::InvalidEdge("self-loop is not allowed".into()));
        }

        // Cycle check: if there is already a path from `to` back to `from`,
        // adding from→to would create a cycle.
        if self.has_path(&edge.to, &edge.from) {
            return Err(CausalError::CycleDetected);
        }

        // Update adjacency lists.
        let from_id = edge.from.clone();
        let to_id = edge.to.clone();

        if let Some(from_node) = self.graph.nodes.get_mut(&from_id) {
            if !from_node.children.contains(&to_id) {
                from_node.children.push(to_id.clone());
            }
        }
        if let Some(to_node) = self.graph.nodes.get_mut(&to_id) {
            if !to_node.parents.contains(&from_id) {
                to_node.parents.push(from_id);
            }
        }

        self.graph.edges.push(edge);
        Ok(())
    }

    /// Remove a node and all edges that touch it.
    ///
    /// Returns `true` if the node existed and was removed, `false` otherwise.
    pub fn remove_node(&mut self, id: &CausalNodeId) -> bool {
        if !self.graph.nodes.contains_key(id) {
            return false;
        }

        // Remove all edges incident to this node.
        self.graph.edges.retain(|e| &e.from != id && &e.to != id);

        // Clean up parent/child references in remaining nodes.
        for node in self.graph.nodes.values_mut() {
            node.parents.retain(|p| p != id);
            node.children.retain(|c| c != id);
        }

        self.graph.nodes.remove(id);
        true
    }

    // ── Graph queries ─────────────────────────────────────────────────────────

    /// Return `true` if there is at least one directed path from `from` to `to`
    /// (depth-first search, respects `max_path_length`).
    pub fn has_path(&self, from: &CausalNodeId, to: &CausalNodeId) -> bool {
        if from == to {
            return true;
        }
        let mut visited: HashSet<&CausalNodeId> = HashSet::new();
        let mut stack: Vec<(&CausalNodeId, usize)> = vec![(from, 0)];
        while let Some((current, depth)) = stack.pop() {
            if current == to {
                return true;
            }
            if depth >= self.max_path_length {
                continue;
            }
            if !visited.insert(current) {
                continue;
            }
            if let Some(node) = self.graph.nodes.get(current) {
                for child in &node.children {
                    stack.push((child, depth + 1));
                }
            }
        }
        false
    }

    /// Return `true` if `ancestor` is a strict ancestor of `descendant`
    /// (i.e. there is a directed path from `ancestor` to `descendant`).
    pub fn is_ancestor(&self, ancestor: &CausalNodeId, descendant: &CausalNodeId) -> bool {
        if ancestor == descendant {
            return false;
        }
        self.has_path(ancestor, descendant)
    }

    /// Return all strict ancestors of `id` via BFS through parent pointers.
    pub fn ancestors(&self, id: &CausalNodeId) -> Vec<CausalNodeId> {
        let mut result: Vec<CausalNodeId> = Vec::new();
        let mut visited: HashSet<CausalNodeId> = HashSet::new();
        let mut queue: VecDeque<CausalNodeId> = VecDeque::new();

        if let Some(node) = self.graph.nodes.get(id) {
            for p in &node.parents {
                queue.push_back(p.clone());
            }
        }
        while let Some(current) = queue.pop_front() {
            if !visited.insert(current.clone()) {
                continue;
            }
            result.push(current.clone());
            if let Some(node) = self.graph.nodes.get(&current) {
                for p in &node.parents {
                    if !visited.contains(p) {
                        queue.push_back(p.clone());
                    }
                }
            }
        }
        result
    }

    /// Return all strict descendants of `id` via BFS through child pointers.
    pub fn descendants(&self, id: &CausalNodeId) -> Vec<CausalNodeId> {
        let mut result: Vec<CausalNodeId> = Vec::new();
        let mut visited: HashSet<CausalNodeId> = HashSet::new();
        let mut queue: VecDeque<CausalNodeId> = VecDeque::new();

        if let Some(node) = self.graph.nodes.get(id) {
            for c in &node.children {
                queue.push_back(c.clone());
            }
        }
        while let Some(current) = queue.pop_front() {
            if !visited.insert(current.clone()) {
                continue;
            }
            result.push(current.clone());
            if let Some(node) = self.graph.nodes.get(&current) {
                for c in &node.children {
                    if !visited.contains(c) {
                        queue.push_back(c.clone());
                    }
                }
            }
        }
        result
    }

    // ── Path enumeration ──────────────────────────────────────────────────────

    /// Enumerate all directed paths from `from` to `to` up to
    /// `max_path_length` hops (DFS with backtracking).
    pub fn all_directed_paths(
        &self,
        from: &CausalNodeId,
        to: &CausalNodeId,
    ) -> Vec<Vec<CausalNodeId>> {
        let mut paths: Vec<Vec<CausalNodeId>> = Vec::new();
        let mut current_path: Vec<CausalNodeId> = vec![from.clone()];
        self.dfs_paths(from, to, &mut current_path, &mut paths);
        paths
    }

    fn dfs_paths(
        &self,
        current: &CausalNodeId,
        target: &CausalNodeId,
        path: &mut Vec<CausalNodeId>,
        results: &mut Vec<Vec<CausalNodeId>>,
    ) {
        if path.len() > self.max_path_length + 1 {
            return;
        }
        if current == target && path.len() > 1 {
            results.push(path.clone());
            return;
        }
        if let Some(node) = self.graph.nodes.get(current) {
            for child in &node.children {
                // Avoid cycles in the traversal path
                if path.contains(child) {
                    continue;
                }
                path.push(child.clone());
                self.dfs_paths(child, target, path, results);
                path.pop();
            }
        }
    }

    /// Enumerate backdoor paths from `from` to `to`.
    ///
    /// A backdoor path is any path that arrives at `from` via an *incoming*
    /// edge (i.e. starts from a parent of `from` and eventually reaches `to`).
    /// Paths are limited to `max_path_length` hops.
    pub fn backdoor_paths(&self, from: &CausalNodeId, to: &CausalNodeId) -> Vec<Vec<CausalNodeId>> {
        let Some(from_node) = self.graph.nodes.get(from) else {
            return Vec::new();
        };
        let parents: Vec<CausalNodeId> = from_node.parents.clone();
        let mut all_paths: Vec<Vec<CausalNodeId>> = Vec::new();

        for parent in &parents {
            // Build paths from this parent to `to` using the undirected
            // skeleton but respecting the "starts by entering `from`" semantics.
            // We use a DFS that follows all adjacent edges (both directed
            // directions) while forbidding the use of the direct from→to
            // directed edge so as to enumerate only backdoor routes.
            let mut path: Vec<CausalNodeId> = vec![from.clone(), parent.clone()];
            self.backdoor_dfs(parent, to, from, &mut path, &mut all_paths);
        }
        all_paths
    }

    fn backdoor_dfs(
        &self,
        current: &CausalNodeId,
        target: &CausalNodeId,
        source: &CausalNodeId, // the original `from` — used to avoid re-entering it
        path: &mut Vec<CausalNodeId>,
        results: &mut Vec<Vec<CausalNodeId>>,
    ) {
        if path.len() > self.max_path_length + 1 {
            return;
        }
        if current == target && path.len() > 2 {
            results.push(path.clone());
            return;
        }
        // Walk neighbours in the undirected skeleton (both directions).
        let mut neighbours: Vec<CausalNodeId> = Vec::new();
        if let Some(node) = self.graph.nodes.get(current) {
            for c in &node.children {
                neighbours.push(c.clone());
            }
            for p in &node.parents {
                neighbours.push(p.clone());
            }
        }
        for neighbour in &neighbours {
            if neighbour == source {
                continue;
            }
            if path.contains(neighbour) {
                continue;
            }
            path.push(neighbour.clone());
            self.backdoor_dfs(neighbour, target, source, path, results);
            path.pop();
        }
    }

    // ── Edge-lookup helpers ───────────────────────────────────────────────────

    /// Look up the strength of the direct edge from `from` to `to`, returning
    /// `0.0` if no such edge exists.
    fn direct_edge_strength(&self, from: &CausalNodeId, to: &CausalNodeId) -> f64 {
        self.graph
            .edges
            .iter()
            .find(|e| &e.from == from && &e.to == to)
            .map(|e| e.strength)
            .unwrap_or(0.0)
    }

    /// Compute the product of edge strengths along a directed path.
    ///
    /// The path is given as a sequence of node ids.  For each consecutive pair
    /// the direct edge strength is looked up; if any hop has no direct edge the
    /// path effect is `0.0`.
    fn path_effect(&self, path: &[CausalNodeId]) -> f64 {
        if path.len() < 2 {
            return 0.0;
        }
        let mut product = 1.0_f64;
        for window in path.windows(2) {
            let strength = self.direct_edge_strength(&window[0], &window[1]);
            product *= strength;
        }
        product
    }

    // ── Interventional inference ──────────────────────────────────────────────

    /// Compute the interventional distribution P(target | do(intervention)).
    ///
    /// Uses a linear causal model: the mean of `target` under the intervention
    /// is the intervention value times the sum of all path effects from the
    /// intervened node to `target`.  The posterior variance shrinks by the
    /// total explained fraction (capped at 0.99 to avoid degenerate zero
    /// variance), and the confidence is the explained fraction clamped to
    /// [0, 1].
    pub fn do_calculus(
        &self,
        intervention: &Intervention,
        target: &CausalNodeId,
    ) -> InferenceResult {
        // All directed paths from the intervened node to the target.
        let paths = self.all_directed_paths(&intervention.node, target);

        let total_path_effect: f64 = paths.iter().map(|p| self.path_effect(p)).sum();
        let total_explained_variance: f64 = total_path_effect.powi(2).min(1.0);

        let target_variance = self
            .graph
            .nodes
            .get(target)
            .map(|n| n.variance)
            .unwrap_or(1.0);

        let mean = intervention.value * total_path_effect;
        let variance = target_variance * (1.0 - total_explained_variance.min(0.99));
        let confidence = total_explained_variance.clamp(0.0, 1.0);

        InferenceResult {
            target: target.clone(),
            mean,
            variance,
            confidence,
            interventions_applied: vec![intervention.clone()],
        }
    }

    // ── Counterfactual inference ──────────────────────────────────────────────

    /// Estimate the counterfactual distribution of `query.target` had
    /// `query.intervention` been applied, conditioned on `query.evidence`.
    ///
    /// The implementation uses do-calculus as a base and adds an evidence
    /// correction: for each observed variable in `evidence` we sum the product
    /// of its observed value and the direct edge strength to the target.
    pub fn counterfactual(&self, query: &CounterfactualQuery) -> InferenceResult {
        let mut base = self.do_calculus(&query.intervention, &query.target);

        // Evidence correction: weighted sum of evidence → target edge strengths.
        let evidence_correction: f64 = query
            .evidence
            .iter()
            .map(|(ev_node, &ev_value)| {
                let strength = self.direct_edge_strength(ev_node, &query.target);
                ev_value * strength
            })
            .sum();

        base.mean += evidence_correction;
        // Record all interventions that were implicitly applied via evidence.
        for (ev_node, &ev_value) in &query.evidence {
            base.interventions_applied.push(Intervention {
                node: ev_node.clone(),
                value: ev_value,
            });
        }
        base
    }

    // ── Average Causal Effect ─────────────────────────────────────────────────

    /// Compute the Average Causal Effect (ACE) of `from` on `to`.
    ///
    /// ACE = E[to | do(from = value2)] - E[to | do(from = value1)]
    pub fn average_causal_effect(
        &self,
        from: &CausalNodeId,
        to: &CausalNodeId,
        value1: f64,
        value2: f64,
    ) -> f64 {
        let int1 = Intervention {
            node: from.clone(),
            value: value1,
        };
        let int2 = Intervention {
            node: from.clone(),
            value: value2,
        };
        self.do_calculus(&int2, to).mean - self.do_calculus(&int1, to).mean
    }

    // ── Confounders ───────────────────────────────────────────────────────────

    /// Return the common ancestors of `x` and `y` (i.e. potential confounders).
    pub fn confounders(&self, x: &CausalNodeId, y: &CausalNodeId) -> Vec<CausalNodeId> {
        let anc_x: HashSet<CausalNodeId> = self.ancestors(x).into_iter().collect();
        let anc_y: HashSet<CausalNodeId> = self.ancestors(y).into_iter().collect();
        let mut common: Vec<CausalNodeId> = anc_x.intersection(&anc_y).cloned().collect();
        common.sort();
        common
    }

    // ── d-separation ─────────────────────────────────────────────────────────

    /// Simplified d-separation check.
    ///
    /// Returns `true` if `x` and `y` are d-separated given the conditioning
    /// set `given`.
    ///
    /// The implementation tests:
    /// 1. No direct edge from `x` to `y` (after removing given nodes).
    /// 2. No directed path from `x` to `y` that does not pass through a
    ///    given node.
    /// 3. No backdoor path from `x` to `y` that is not blocked by a given node.
    pub fn is_d_separated(
        &self,
        x: &CausalNodeId,
        y: &CausalNodeId,
        given: &[CausalNodeId],
    ) -> bool {
        let given_set: HashSet<&CausalNodeId> = given.iter().collect();

        // Check direct edge x → y not blocked.
        for edge in &self.graph.edges {
            if &edge.from == x && &edge.to == y && !given_set.contains(y) {
                return false;
            }
        }

        // Check all directed paths from x to y; a path is blocked if it
        // passes through a given node (chain / fork blocking).
        let directed_paths = self.all_directed_paths(x, y);
        for path in &directed_paths {
            let intermediate_nodes = &path[1..path.len().saturating_sub(1)];
            let blocked = intermediate_nodes.iter().any(|n| given_set.contains(n));
            if !blocked {
                return false;
            }
        }

        // Check backdoor paths from x to y.
        let bd_paths = self.backdoor_paths(x, y);
        for path in &bd_paths {
            let intermediate_nodes = if path.len() > 2 {
                &path[1..path.len() - 1]
            } else {
                &path[1..path.len()]
            };
            let blocked = intermediate_nodes.iter().any(|n| given_set.contains(n));
            if !blocked {
                return false;
            }
        }

        true
    }

    // ── Statistics ────────────────────────────────────────────────────────────

    /// Compute summary statistics for the current graph.
    pub fn stats(&self) -> CausalStats {
        let node_count = self.graph.nodes.len();
        let edge_count = self.graph.edges.len();

        let avg_children = if node_count == 0 {
            0.0
        } else {
            self.graph
                .nodes
                .values()
                .map(|n| n.children.len() as f64)
                .sum::<f64>()
                / node_count as f64
        };

        // BFS from each root (no parents) to find the maximum depth.
        let max_depth = self.compute_max_depth();

        CausalStats {
            node_count,
            edge_count,
            avg_children,
            max_depth,
        }
    }

    /// Compute the maximum path length from any root to any leaf.
    fn compute_max_depth(&self) -> usize {
        let roots: Vec<&CausalNodeId> = self
            .graph
            .nodes
            .values()
            .filter(|n| n.parents.is_empty())
            .map(|n| &n.id)
            .collect();

        let mut max_depth = 0usize;
        for root in roots {
            let depth = self.bfs_depth(root);
            if depth > max_depth {
                max_depth = depth;
            }
        }
        max_depth
    }

    fn bfs_depth(&self, root: &CausalNodeId) -> usize {
        let mut queue: VecDeque<(&CausalNodeId, usize)> = VecDeque::new();
        queue.push_back((root, 0));
        let mut max_depth = 0usize;
        while let Some((current, depth)) = queue.pop_front() {
            if depth > max_depth {
                max_depth = depth;
            }
            if let Some(node) = self.graph.nodes.get(current) {
                for child in &node.children {
                    queue.push_back((child, depth + 1));
                }
            }
        }
        max_depth
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::causal_inference::{
        CausalEdge, CausalEdgeType, CausalError, CausalInferenceEngine, CausalNode, CausalNodeId,
        CounterfactualQuery, Intervention,
    };

    // ── Helpers ────────────────────────────────────────────────────────────────

    /// Build a simple X → Y graph.
    fn simple_xy() -> CausalInferenceEngine {
        let mut engine = CausalInferenceEngine::new(10);
        engine
            .add_node(CausalNode::new("X", 0.0, 1.0))
            .expect("test setup: add_node should not fail for unique node");
        engine
            .add_node(CausalNode::new("Y", 0.0, 1.0))
            .expect("test setup: add_node should not fail for unique node");
        engine
            .add_edge(CausalEdge::direct("X", "Y", 0.5))
            .expect("test setup: add_edge should not fail for valid DAG edge");
        engine
    }

    /// Build X → M → Y (mediated chain).
    fn chain_xmy() -> CausalInferenceEngine {
        let mut engine = CausalInferenceEngine::new(10);
        engine
            .add_node(CausalNode::new("X", 0.0, 1.0))
            .expect("test setup: add_node should not fail for unique node");
        engine
            .add_node(CausalNode::new("M", 0.0, 1.0))
            .expect("test setup: add_node should not fail for unique node");
        engine
            .add_node(CausalNode::new("Y", 0.0, 1.0))
            .expect("test setup: add_node should not fail for unique node");
        engine
            .add_edge(CausalEdge::direct("X", "M", 0.6))
            .expect("test setup: add_edge should not fail for valid DAG edge");
        engine
            .add_edge(CausalEdge::direct("M", "Y", 0.8))
            .expect("test setup: add_edge should not fail for valid DAG edge");
        engine
    }

    // ── 1. Construction ────────────────────────────────────────────────────────

    #[test]
    fn test_new_engine_is_empty() {
        let engine = CausalInferenceEngine::new(5);
        assert_eq!(engine.graph.nodes.len(), 0);
        assert_eq!(engine.graph.edges.len(), 0);
        assert_eq!(engine.max_path_length, 5);
    }

    // ── 2. add_node ───────────────────────────────────────────────────────────

    #[test]
    fn test_add_node_success() {
        let mut engine = CausalInferenceEngine::new(10);
        let result = engine.add_node(CausalNode::new("A", 1.0, 2.0));
        assert!(result.is_ok());
        assert!(engine.graph.nodes.contains_key(&CausalNodeId::new("A")));
    }

    #[test]
    fn test_add_node_duplicate_returns_error() {
        let mut engine = CausalInferenceEngine::new(10);
        engine
            .add_node(CausalNode::new("A", 0.0, 1.0))
            .expect("test setup: add_node should not fail for unique node");
        let err = engine.add_node(CausalNode::new("A", 1.0, 2.0)).unwrap_err();
        assert_eq!(err, CausalError::NodeAlreadyExists("A".into()));
    }

    // ── 3. add_edge ───────────────────────────────────────────────────────────

    #[test]
    fn test_add_edge_success_updates_parent_children() {
        let engine = simple_xy();
        let x = engine
            .graph
            .nodes
            .get(&CausalNodeId::new("X"))
            .expect("test setup: node must exist in graph");
        let y = engine
            .graph
            .nodes
            .get(&CausalNodeId::new("Y"))
            .expect("test setup: node must exist in graph");
        assert!(x.children.contains(&CausalNodeId::new("Y")));
        assert!(y.parents.contains(&CausalNodeId::new("X")));
    }

    #[test]
    fn test_add_edge_missing_from_returns_error() {
        let mut engine = CausalInferenceEngine::new(10);
        engine
            .add_node(CausalNode::new("Y", 0.0, 1.0))
            .expect("test setup: add_node should not fail for unique node");
        let err = engine
            .add_edge(CausalEdge::direct("X", "Y", 1.0))
            .unwrap_err();
        assert_eq!(err, CausalError::NodeNotFound("X".into()));
    }

    #[test]
    fn test_add_edge_missing_to_returns_error() {
        let mut engine = CausalInferenceEngine::new(10);
        engine
            .add_node(CausalNode::new("X", 0.0, 1.0))
            .expect("test setup: add_node should not fail for unique node");
        let err = engine
            .add_edge(CausalEdge::direct("X", "Y", 1.0))
            .unwrap_err();
        assert_eq!(err, CausalError::NodeNotFound("Y".into()));
    }

    #[test]
    fn test_add_edge_self_loop_rejected() {
        let mut engine = CausalInferenceEngine::new(10);
        engine
            .add_node(CausalNode::new("X", 0.0, 1.0))
            .expect("test setup: add_node should not fail for unique node");
        let err = engine
            .add_edge(CausalEdge::direct("X", "X", 1.0))
            .unwrap_err();
        assert_eq!(
            err,
            CausalError::InvalidEdge("self-loop is not allowed".into())
        );
    }

    #[test]
    fn test_add_edge_cycle_rejected() {
        let mut engine = CausalInferenceEngine::new(10);
        engine
            .add_node(CausalNode::new("A", 0.0, 1.0))
            .expect("test setup: add_node should not fail for unique node");
        engine
            .add_node(CausalNode::new("B", 0.0, 1.0))
            .expect("test setup: add_node should not fail for unique node");
        engine
            .add_edge(CausalEdge::direct("A", "B", 1.0))
            .expect("test setup: add_edge should not fail for valid DAG edge");
        let err = engine
            .add_edge(CausalEdge::direct("B", "A", 1.0))
            .unwrap_err();
        assert_eq!(err, CausalError::CycleDetected);
    }

    // ── 4. remove_node ────────────────────────────────────────────────────────

    #[test]
    fn test_remove_node_removes_edges() {
        let mut engine = simple_xy();
        let removed = engine.remove_node(&CausalNodeId::new("X"));
        assert!(removed);
        assert!(!engine.graph.nodes.contains_key(&CausalNodeId::new("X")));
        assert!(engine.graph.edges.is_empty());
        // Y's parent list should be cleaned.
        let y = engine
            .graph
            .nodes
            .get(&CausalNodeId::new("Y"))
            .expect("test setup: node must exist in graph");
        assert!(y.parents.is_empty());
    }

    #[test]
    fn test_remove_nonexistent_node_returns_false() {
        let mut engine = CausalInferenceEngine::new(10);
        assert!(!engine.remove_node(&CausalNodeId::new("Ghost")));
    }

    // ── 5. has_path ───────────────────────────────────────────────────────────

    #[test]
    fn test_has_path_direct() {
        let engine = simple_xy();
        assert!(engine.has_path(&CausalNodeId::new("X"), &CausalNodeId::new("Y")));
    }

    #[test]
    fn test_has_path_no_reverse() {
        let engine = simple_xy();
        assert!(!engine.has_path(&CausalNodeId::new("Y"), &CausalNodeId::new("X")));
    }

    #[test]
    fn test_has_path_through_mediator() {
        let engine = chain_xmy();
        assert!(engine.has_path(&CausalNodeId::new("X"), &CausalNodeId::new("Y")));
    }

    #[test]
    fn test_has_path_same_node() {
        let engine = simple_xy();
        assert!(engine.has_path(&CausalNodeId::new("X"), &CausalNodeId::new("X")));
    }

    // ── 6. is_ancestor ────────────────────────────────────────────────────────

    #[test]
    fn test_is_ancestor_direct() {
        let engine = simple_xy();
        assert!(engine.is_ancestor(&CausalNodeId::new("X"), &CausalNodeId::new("Y")));
    }

    #[test]
    fn test_is_ancestor_not_self() {
        let engine = simple_xy();
        assert!(!engine.is_ancestor(&CausalNodeId::new("X"), &CausalNodeId::new("X")));
    }

    #[test]
    fn test_is_ancestor_transitive() {
        let engine = chain_xmy();
        assert!(engine.is_ancestor(&CausalNodeId::new("X"), &CausalNodeId::new("Y")));
    }

    // ── 7. ancestors / descendants ───────────────────────────────────────────

    #[test]
    fn test_ancestors_chain() {
        let engine = chain_xmy();
        let ancs = engine.ancestors(&CausalNodeId::new("Y"));
        assert!(ancs.contains(&CausalNodeId::new("M")));
        assert!(ancs.contains(&CausalNodeId::new("X")));
    }

    #[test]
    fn test_ancestors_root_has_none() {
        let engine = chain_xmy();
        assert!(engine.ancestors(&CausalNodeId::new("X")).is_empty());
    }

    #[test]
    fn test_descendants_chain() {
        let engine = chain_xmy();
        let descs = engine.descendants(&CausalNodeId::new("X"));
        assert!(descs.contains(&CausalNodeId::new("M")));
        assert!(descs.contains(&CausalNodeId::new("Y")));
    }

    #[test]
    fn test_descendants_leaf_has_none() {
        let engine = chain_xmy();
        assert!(engine.descendants(&CausalNodeId::new("Y")).is_empty());
    }

    // ── 8. do_calculus ────────────────────────────────────────────────────────

    #[test]
    fn test_do_calculus_direct_edge() {
        let engine = simple_xy();
        let result = engine.do_calculus(&Intervention::new("X", 2.0), &CausalNodeId::new("Y"));
        // Only one path X→Y with strength 0.5, so mean = 2.0 * 0.5 = 1.0
        assert!((result.mean - 1.0).abs() < 1e-9, "mean={}", result.mean);
    }

    #[test]
    fn test_do_calculus_chain() {
        let engine = chain_xmy();
        let result = engine.do_calculus(&Intervention::new("X", 1.0), &CausalNodeId::new("Y"));
        // Path X→M→Y: strength = 0.6 * 0.8 = 0.48
        assert!((result.mean - 0.48).abs() < 1e-9, "mean={}", result.mean);
    }

    #[test]
    fn test_do_calculus_no_path_gives_zero_mean() {
        let engine = simple_xy();
        // Intervene on Y, query X — no directed path Y→X
        let result = engine.do_calculus(&Intervention::new("Y", 5.0), &CausalNodeId::new("X"));
        assert!((result.mean).abs() < 1e-9);
    }

    #[test]
    fn test_do_calculus_target_variance_shrinks() {
        let engine = simple_xy();
        let base_var = engine
            .graph
            .nodes
            .get(&CausalNodeId::new("Y"))
            .expect("test setup: node must exist in graph")
            .variance;
        let result = engine.do_calculus(&Intervention::new("X", 1.0), &CausalNodeId::new("Y"));
        assert!(result.variance <= base_var);
    }

    #[test]
    fn test_do_calculus_confidence_bounded() {
        let engine = simple_xy();
        let result = engine.do_calculus(&Intervention::new("X", 1.0), &CausalNodeId::new("Y"));
        assert!((0.0..=1.0).contains(&result.confidence));
    }

    #[test]
    fn test_do_calculus_interventions_recorded() {
        let engine = simple_xy();
        let int = Intervention::new("X", 3.0);
        let result = engine.do_calculus(&int, &CausalNodeId::new("Y"));
        assert_eq!(result.interventions_applied.len(), 1);
        assert_eq!(result.interventions_applied[0].node, CausalNodeId::new("X"));
    }

    // ── 9. counterfactual ─────────────────────────────────────────────────────

    #[test]
    fn test_counterfactual_no_evidence_equals_do_calculus() {
        let engine = simple_xy();
        let int = Intervention::new("X", 1.0);
        let query = CounterfactualQuery {
            target: CausalNodeId::new("Y"),
            intervention: int.clone(),
            evidence: HashMap::new(),
        };
        let cf_result = engine.counterfactual(&query);
        let do_result = engine.do_calculus(&int, &CausalNodeId::new("Y"));
        assert!((cf_result.mean - do_result.mean).abs() < 1e-9);
    }

    #[test]
    fn test_counterfactual_with_evidence_adjusts_mean() {
        // Build Z → Y (direct), X → Y (direct)
        let mut engine = CausalInferenceEngine::new(10);
        engine
            .add_node(CausalNode::new("X", 0.0, 1.0))
            .expect("test setup: add_node should not fail for unique node");
        engine
            .add_node(CausalNode::new("Z", 0.0, 1.0))
            .expect("test setup: add_node should not fail for unique node");
        engine
            .add_node(CausalNode::new("Y", 0.0, 1.0))
            .expect("test setup: add_node should not fail for unique node");
        engine
            .add_edge(CausalEdge::direct("X", "Y", 0.5))
            .expect("test setup: add_edge should not fail for valid DAG edge");
        engine
            .add_edge(CausalEdge::direct("Z", "Y", 0.3))
            .expect("test setup: add_edge should not fail for valid DAG edge");

        let mut evidence = HashMap::new();
        evidence.insert(CausalNodeId::new("Z"), 2.0);

        let query = CounterfactualQuery {
            target: CausalNodeId::new("Y"),
            intervention: Intervention::new("X", 1.0),
            evidence,
        };
        let cf = engine.counterfactual(&query);
        // do-calculus mean = 1.0 * 0.5 = 0.5; correction = 2.0 * 0.3 = 0.6
        assert!((cf.mean - 1.1).abs() < 1e-9, "mean={}", cf.mean);
    }

    // ── 10. average_causal_effect ─────────────────────────────────────────────

    #[test]
    fn test_ace_linear() {
        let engine = simple_xy();
        // ACE = (2.0 - 0.0) * 0.5 = 1.0
        let ace = engine.average_causal_effect(
            &CausalNodeId::new("X"),
            &CausalNodeId::new("Y"),
            0.0,
            2.0,
        );
        assert!((ace - 1.0).abs() < 1e-9, "ace={ace}");
    }

    #[test]
    fn test_ace_zero_when_no_path() {
        let engine = simple_xy();
        let ace = engine.average_causal_effect(
            &CausalNodeId::new("Y"),
            &CausalNodeId::new("X"),
            0.0,
            1.0,
        );
        assert!((ace).abs() < 1e-9);
    }

    #[test]
    fn test_ace_chain() {
        let engine = chain_xmy();
        // Total path X→M→Y: 0.6*0.8 = 0.48; ACE = (1.0-0.0)*0.48 = 0.48
        let ace = engine.average_causal_effect(
            &CausalNodeId::new("X"),
            &CausalNodeId::new("Y"),
            0.0,
            1.0,
        );
        assert!((ace - 0.48).abs() < 1e-9, "ace={ace}");
    }

    // ── 11. confounders ───────────────────────────────────────────────────────

    #[test]
    fn test_confounders_common_cause() {
        // Z → X, Z → Y
        let mut engine = CausalInferenceEngine::new(10);
        engine
            .add_node(CausalNode::new("Z", 0.0, 1.0))
            .expect("test setup: add_node should not fail for unique node");
        engine
            .add_node(CausalNode::new("X", 0.0, 1.0))
            .expect("test setup: add_node should not fail for unique node");
        engine
            .add_node(CausalNode::new("Y", 0.0, 1.0))
            .expect("test setup: add_node should not fail for unique node");
        engine
            .add_edge(CausalEdge::direct("Z", "X", 1.0))
            .expect("test setup: add_edge should not fail for valid DAG edge");
        engine
            .add_edge(CausalEdge::direct("Z", "Y", 1.0))
            .expect("test setup: add_edge should not fail for valid DAG edge");

        let conf = engine.confounders(&CausalNodeId::new("X"), &CausalNodeId::new("Y"));
        assert!(conf.contains(&CausalNodeId::new("Z")));
    }

    #[test]
    fn test_confounders_no_common_cause() {
        let engine = simple_xy();
        let conf = engine.confounders(&CausalNodeId::new("X"), &CausalNodeId::new("Y"));
        assert!(conf.is_empty());
    }

    // ── 12. is_d_separated ───────────────────────────────────────────────────

    #[test]
    fn test_d_sep_blocked_by_given() {
        let engine = chain_xmy();
        // Conditioning on M blocks the path X → M → Y
        let given = vec![CausalNodeId::new("M")];
        assert!(engine.is_d_separated(&CausalNodeId::new("X"), &CausalNodeId::new("Y"), &given,));
    }

    #[test]
    fn test_d_sep_not_separated_without_given() {
        let engine = chain_xmy();
        assert!(!engine.is_d_separated(&CausalNodeId::new("X"), &CausalNodeId::new("Y"), &[],));
    }

    #[test]
    fn test_d_sep_direct_edge_blocks_without_given() {
        let engine = simple_xy();
        assert!(!engine.is_d_separated(&CausalNodeId::new("X"), &CausalNodeId::new("Y"), &[],));
    }

    // ── 13. backdoor_paths ────────────────────────────────────────────────────

    #[test]
    fn test_backdoor_paths_with_confounder() {
        // Z → X, Z → Y (fork)
        let mut engine = CausalInferenceEngine::new(10);
        engine
            .add_node(CausalNode::new("Z", 0.0, 1.0))
            .expect("test setup: add_node should not fail for unique node");
        engine
            .add_node(CausalNode::new("X", 0.0, 1.0))
            .expect("test setup: add_node should not fail for unique node");
        engine
            .add_node(CausalNode::new("Y", 0.0, 1.0))
            .expect("test setup: add_node should not fail for unique node");
        engine
            .add_edge(CausalEdge::direct("Z", "X", 1.0))
            .expect("test setup: add_edge should not fail for valid DAG edge");
        engine
            .add_edge(CausalEdge::direct("Z", "Y", 1.0))
            .expect("test setup: add_edge should not fail for valid DAG edge");

        let bd = engine.backdoor_paths(&CausalNodeId::new("X"), &CausalNodeId::new("Y"));
        assert!(!bd.is_empty(), "expected at least one backdoor path");
    }

    #[test]
    fn test_backdoor_paths_empty_for_chain() {
        // X → M → Y has no backdoor path into X
        let engine = chain_xmy();
        let bd = engine.backdoor_paths(&CausalNodeId::new("X"), &CausalNodeId::new("Y"));
        assert!(bd.is_empty());
    }

    // ── 14. stats ─────────────────────────────────────────────────────────────

    #[test]
    fn test_stats_empty_graph() {
        let engine = CausalInferenceEngine::new(5);
        let s = engine.stats();
        assert_eq!(s.node_count, 0);
        assert_eq!(s.edge_count, 0);
        assert_eq!(s.max_depth, 0);
    }

    #[test]
    fn test_stats_simple_xy() {
        let engine = simple_xy();
        let s = engine.stats();
        assert_eq!(s.node_count, 2);
        assert_eq!(s.edge_count, 1);
        assert!((s.avg_children - 0.5).abs() < 1e-9);
        assert_eq!(s.max_depth, 1);
    }

    #[test]
    fn test_stats_chain() {
        let engine = chain_xmy();
        let s = engine.stats();
        assert_eq!(s.node_count, 3);
        assert_eq!(s.edge_count, 2);
        assert_eq!(s.max_depth, 2);
    }

    // ── 15. CausalNodeId helpers ──────────────────────────────────────────────

    #[test]
    fn test_causal_node_id_display() {
        let id = CausalNodeId::new("foo");
        assert_eq!(format!("{id}"), "foo");
    }

    #[test]
    fn test_causal_node_id_as_str() {
        let id = CausalNodeId::new("bar");
        assert_eq!(id.as_str(), "bar");
    }

    #[test]
    fn test_causal_node_id_equality() {
        let a = CausalNodeId::new("x");
        let b = CausalNodeId::new("x");
        let c = CausalNodeId::new("y");
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    // ── 16. CausalError display ───────────────────────────────────────────────

    #[test]
    fn test_error_display_messages() {
        assert!(CausalError::NodeAlreadyExists("X".into())
            .to_string()
            .contains("X"));
        assert!(CausalError::NodeNotFound("Y".into())
            .to_string()
            .contains("Y"));
        assert!(!CausalError::CycleDetected.to_string().is_empty());
        assert!(CausalError::InvalidEdge("bad".into())
            .to_string()
            .contains("bad"));
    }

    // ── 17. Multiple paths ────────────────────────────────────────────────────

    #[test]
    fn test_do_calculus_multiple_paths() {
        // X → Y (0.3) and X → M → Y (0.5 * 0.4 = 0.2); total = 0.5
        let mut engine = CausalInferenceEngine::new(10);
        engine
            .add_node(CausalNode::new("X", 0.0, 1.0))
            .expect("test setup: add_node should not fail for unique node");
        engine
            .add_node(CausalNode::new("M", 0.0, 1.0))
            .expect("test setup: add_node should not fail for unique node");
        engine
            .add_node(CausalNode::new("Y", 0.0, 1.0))
            .expect("test setup: add_node should not fail for unique node");
        engine
            .add_edge(CausalEdge::direct("X", "Y", 0.3))
            .expect("test setup: add_edge should not fail for valid DAG edge");
        engine
            .add_edge(CausalEdge::direct("X", "M", 0.5))
            .expect("test setup: add_edge should not fail for valid DAG edge");
        engine
            .add_edge(CausalEdge::direct("M", "Y", 0.4))
            .expect("test setup: add_edge should not fail for valid DAG edge");

        let result = engine.do_calculus(&Intervention::new("X", 1.0), &CausalNodeId::new("Y"));
        // 0.3 + 0.2 = 0.5
        assert!((result.mean - 0.5).abs() < 1e-9, "mean={}", result.mean);
    }

    // ── 18. all_directed_paths ────────────────────────────────────────────────

    #[test]
    fn test_all_directed_paths_chain() {
        let engine = chain_xmy();
        let paths = engine.all_directed_paths(&CausalNodeId::new("X"), &CausalNodeId::new("Y"));
        assert_eq!(paths.len(), 1);
        assert_eq!(paths[0].len(), 3); // X, M, Y
    }

    #[test]
    fn test_all_directed_paths_no_path() {
        let engine = simple_xy();
        let paths = engine.all_directed_paths(&CausalNodeId::new("Y"), &CausalNodeId::new("X"));
        assert!(paths.is_empty());
    }

    // ── 19. path_length limit ─────────────────────────────────────────────────

    #[test]
    fn test_path_length_limit_blocks_long_paths() {
        let mut engine = CausalInferenceEngine::new(2);
        for name in ["A", "B", "C", "D"] {
            engine
                .add_node(CausalNode::new(name, 0.0, 1.0))
                .expect("test setup: add_node should not fail for unique node");
        }
        engine
            .add_edge(CausalEdge::direct("A", "B", 1.0))
            .expect("test setup: add_edge should not fail for valid DAG edge");
        engine
            .add_edge(CausalEdge::direct("B", "C", 1.0))
            .expect("test setup: add_edge should not fail for valid DAG edge");
        engine
            .add_edge(CausalEdge::direct("C", "D", 1.0))
            .expect("test setup: add_edge should not fail for valid DAG edge");

        // A→D requires 3 hops but max_path_length=2
        let paths = engine.all_directed_paths(&CausalNodeId::new("A"), &CausalNodeId::new("D"));
        assert!(paths.is_empty(), "expected no paths with limit=2");
    }

    // ── 20. negative strength ─────────────────────────────────────────────────

    #[test]
    fn test_do_calculus_negative_strength() {
        let mut engine = CausalInferenceEngine::new(10);
        engine
            .add_node(CausalNode::new("X", 0.0, 1.0))
            .expect("test setup: add_node should not fail for unique node");
        engine
            .add_node(CausalNode::new("Y", 0.0, 1.0))
            .expect("test setup: add_node should not fail for unique node");
        engine
            .add_edge(CausalEdge::direct("X", "Y", -0.4))
            .expect("test setup: add_edge should not fail for valid DAG edge");

        let result = engine.do_calculus(&Intervention::new("X", 2.0), &CausalNodeId::new("Y"));
        // mean = 2.0 * (-0.4) = -0.8
        assert!((result.mean - (-0.8)).abs() < 1e-9, "mean={}", result.mean);
    }

    // ── 21. CausalEdge construction helpers ───────────────────────────────────

    #[test]
    fn test_edge_direct_helper() {
        let edge = CausalEdge::direct("A", "B", 0.7);
        assert_eq!(edge.from, CausalNodeId::new("A"));
        assert_eq!(edge.to, CausalNodeId::new("B"));
        assert_eq!(edge.edge_type, CausalEdgeType::Direct);
        assert!((edge.strength - 0.7).abs() < 1e-9);
    }

    // ── 22. Intervention helper ───────────────────────────────────────────────

    #[test]
    fn test_intervention_new() {
        let int = Intervention::new("X", std::f64::consts::PI);
        assert_eq!(int.node, CausalNodeId::new("X"));
        assert!((int.value - std::f64::consts::PI).abs() < 1e-9);
    }

    // ── 23. large diamond graph ───────────────────────────────────────────────

    #[test]
    fn test_diamond_graph_two_paths() {
        // X → A → Y and X → B → Y
        let mut engine = CausalInferenceEngine::new(10);
        for name in ["X", "A", "B", "Y"] {
            engine
                .add_node(CausalNode::new(name, 0.0, 1.0))
                .expect("test setup: add_node should not fail for unique node");
        }
        engine
            .add_edge(CausalEdge::direct("X", "A", 0.5))
            .expect("test setup: add_edge should not fail for valid DAG edge");
        engine
            .add_edge(CausalEdge::direct("X", "B", 0.5))
            .expect("test setup: add_edge should not fail for valid DAG edge");
        engine
            .add_edge(CausalEdge::direct("A", "Y", 0.6))
            .expect("test setup: add_edge should not fail for valid DAG edge");
        engine
            .add_edge(CausalEdge::direct("B", "Y", 0.4))
            .expect("test setup: add_edge should not fail for valid DAG edge");

        let result = engine.do_calculus(&Intervention::new("X", 1.0), &CausalNodeId::new("Y"));
        // Path X→A→Y: 0.5*0.6=0.3; path X→B→Y: 0.5*0.4=0.2; total = 0.5
        assert!((result.mean - 0.5).abs() < 1e-9, "mean={}", result.mean);
    }

    // ── 24. counterfactual multiple evidence ─────────────────────────────────

    #[test]
    fn test_counterfactual_multiple_evidence_nodes() {
        let mut engine = CausalInferenceEngine::new(10);
        engine
            .add_node(CausalNode::new("X", 0.0, 1.0))
            .expect("test setup: add_node should not fail for unique node");
        engine
            .add_node(CausalNode::new("Z1", 0.0, 1.0))
            .expect("test setup: add_node should not fail for unique node");
        engine
            .add_node(CausalNode::new("Z2", 0.0, 1.0))
            .expect("test setup: add_node should not fail for unique node");
        engine
            .add_node(CausalNode::new("Y", 0.0, 1.0))
            .expect("test setup: add_node should not fail for unique node");
        engine
            .add_edge(CausalEdge::direct("X", "Y", 0.4))
            .expect("test setup: add_edge should not fail for valid DAG edge");
        engine
            .add_edge(CausalEdge::direct("Z1", "Y", 0.2))
            .expect("test setup: add_edge should not fail for valid DAG edge");
        engine
            .add_edge(CausalEdge::direct("Z2", "Y", 0.3))
            .expect("test setup: add_edge should not fail for valid DAG edge");

        let mut evidence = HashMap::new();
        evidence.insert(CausalNodeId::new("Z1"), 1.0);
        evidence.insert(CausalNodeId::new("Z2"), 1.0);

        let query = CounterfactualQuery {
            target: CausalNodeId::new("Y"),
            intervention: Intervention::new("X", 1.0),
            evidence,
        };
        let cf = engine.counterfactual(&query);
        // base = 1.0*0.4 = 0.4; correction = 1.0*0.2 + 1.0*0.3 = 0.5
        assert!((cf.mean - 0.9).abs() < 1e-9, "mean={}", cf.mean);
    }

    // ── 25. descendants of middle node ───────────────────────────────────────

    #[test]
    fn test_descendants_of_mediator() {
        let engine = chain_xmy();
        let descs = engine.descendants(&CausalNodeId::new("M"));
        assert_eq!(descs.len(), 1);
        assert!(descs.contains(&CausalNodeId::new("Y")));
    }

    // ── 26. confounders sorted ────────────────────────────────────────────────

    #[test]
    fn test_confounders_sorted() {
        // Build: Z1 → X, Z2 → X, Z1 → Y, Z2 → Y
        let mut engine = CausalInferenceEngine::new(10);
        for name in ["Z1", "Z2", "X", "Y"] {
            engine
                .add_node(CausalNode::new(name, 0.0, 1.0))
                .expect("test setup: add_node should not fail for unique node");
        }
        engine
            .add_edge(CausalEdge::direct("Z1", "X", 1.0))
            .expect("test setup: add_edge should not fail for valid DAG edge");
        engine
            .add_edge(CausalEdge::direct("Z2", "X", 1.0))
            .expect("test setup: add_edge should not fail for valid DAG edge");
        engine
            .add_edge(CausalEdge::direct("Z1", "Y", 1.0))
            .expect("test setup: add_edge should not fail for valid DAG edge");
        engine
            .add_edge(CausalEdge::direct("Z2", "Y", 1.0))
            .expect("test setup: add_edge should not fail for valid DAG edge");

        let conf = engine.confounders(&CausalNodeId::new("X"), &CausalNodeId::new("Y"));
        assert_eq!(conf.len(), 2);
        // They should be sorted
        let names: Vec<&str> = conf.iter().map(|id| id.as_str()).collect();
        let mut sorted_names = names.clone();
        sorted_names.sort_unstable();
        assert_eq!(names, sorted_names);
    }

    // ── 27. remove node cleans children of parent ────────────────────────────

    #[test]
    fn test_remove_child_updates_parent_children_list() {
        let mut engine = chain_xmy();
        engine.remove_node(&CausalNodeId::new("Y"));
        let m_node = engine
            .graph
            .nodes
            .get(&CausalNodeId::new("M"))
            .expect("test setup: node must exist in graph");
        assert!(m_node.children.is_empty());
    }

    // ── 28. CausalNode::new initialises empty edges ──────────────────────────

    #[test]
    fn test_causal_node_new_empty() {
        let node = CausalNode::new("test", 1.5, 2.5);
        assert_eq!(node.id, CausalNodeId::new("test"));
        assert!((node.mean - 1.5).abs() < 1e-9);
        assert!((node.variance - 2.5).abs() < 1e-9);
        assert!(node.parents.is_empty());
        assert!(node.children.is_empty());
    }

    // ── 29. stats avg_children ────────────────────────────────────────────────

    #[test]
    fn test_stats_avg_children_diamond() {
        let mut engine = CausalInferenceEngine::new(10);
        for name in ["X", "A", "B", "Y"] {
            engine
                .add_node(CausalNode::new(name, 0.0, 1.0))
                .expect("test setup: add_node should not fail for unique node");
        }
        engine
            .add_edge(CausalEdge::direct("X", "A", 1.0))
            .expect("test setup: add_edge should not fail for valid DAG edge");
        engine
            .add_edge(CausalEdge::direct("X", "B", 1.0))
            .expect("test setup: add_edge should not fail for valid DAG edge");
        engine
            .add_edge(CausalEdge::direct("A", "Y", 1.0))
            .expect("test setup: add_edge should not fail for valid DAG edge");
        engine
            .add_edge(CausalEdge::direct("B", "Y", 1.0))
            .expect("test setup: add_edge should not fail for valid DAG edge");

        let s = engine.stats();
        // X has 2 children, A has 1, B has 1, Y has 0 → avg = (2+1+1+0)/4 = 1.0
        assert!(
            (s.avg_children - 1.0).abs() < 1e-9,
            "avg_children={}",
            s.avg_children
        );
    }

    // ── 30. is_d_separated fork blocked at cause ─────────────────────────────

    #[test]
    fn test_d_sep_fork_blocked_at_common_cause() {
        // Z → X, Z → Y: conditioning on Z blocks the backdoor X←Z→Y
        let mut engine = CausalInferenceEngine::new(10);
        engine
            .add_node(CausalNode::new("Z", 0.0, 1.0))
            .expect("test setup: add_node should not fail for unique node");
        engine
            .add_node(CausalNode::new("X", 0.0, 1.0))
            .expect("test setup: add_node should not fail for unique node");
        engine
            .add_node(CausalNode::new("Y", 0.0, 1.0))
            .expect("test setup: add_node should not fail for unique node");
        engine
            .add_edge(CausalEdge::direct("Z", "X", 1.0))
            .expect("test setup: add_edge should not fail for valid DAG edge");
        engine
            .add_edge(CausalEdge::direct("Z", "Y", 1.0))
            .expect("test setup: add_edge should not fail for valid DAG edge");

        let given = vec![CausalNodeId::new("Z")];
        assert!(engine.is_d_separated(&CausalNodeId::new("X"), &CausalNodeId::new("Y"), &given,));
    }

    // ── 31. ACE scaling ──────────────────────────────────────────────────────

    #[test]
    fn test_ace_scales_linearly_with_delta() {
        let engine = simple_xy();
        let ace1 = engine.average_causal_effect(
            &CausalNodeId::new("X"),
            &CausalNodeId::new("Y"),
            0.0,
            1.0,
        );
        let ace2 = engine.average_causal_effect(
            &CausalNodeId::new("X"),
            &CausalNodeId::new("Y"),
            0.0,
            2.0,
        );
        // ace2 should be twice ace1
        assert!((ace2 - 2.0 * ace1).abs() < 1e-9, "ace1={ace1} ace2={ace2}");
    }

    // ── 32. all_directed_paths diamond ───────────────────────────────────────

    #[test]
    fn test_all_directed_paths_diamond_returns_two() {
        let mut engine = CausalInferenceEngine::new(10);
        for name in ["X", "A", "B", "Y"] {
            engine
                .add_node(CausalNode::new(name, 0.0, 1.0))
                .expect("test setup: add_node should not fail for unique node");
        }
        engine
            .add_edge(CausalEdge::direct("X", "A", 1.0))
            .expect("test setup: add_edge should not fail for valid DAG edge");
        engine
            .add_edge(CausalEdge::direct("X", "B", 1.0))
            .expect("test setup: add_edge should not fail for valid DAG edge");
        engine
            .add_edge(CausalEdge::direct("A", "Y", 1.0))
            .expect("test setup: add_edge should not fail for valid DAG edge");
        engine
            .add_edge(CausalEdge::direct("B", "Y", 1.0))
            .expect("test setup: add_edge should not fail for valid DAG edge");

        let paths = engine.all_directed_paths(&CausalNodeId::new("X"), &CausalNodeId::new("Y"));
        assert_eq!(paths.len(), 2);
    }
}
