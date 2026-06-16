//! Knowledge Graph Traversal for TensorLogic distributed reasoning.
//!
//! TensorLogic rules form an implicit knowledge graph. This module provides
//! efficient traversal, path-finding, and subgraph extraction for distributed
//! reasoning over predicate dependency graphs.

use std::collections::{HashMap, HashSet, VecDeque};
use thiserror::Error;

/// Error types for knowledge graph operations.
#[derive(Debug, Error)]
pub enum KgError {
    /// A node with the given ID was not found.
    #[error("node not found: {0}")]
    NodeNotFound(String),

    /// A node with the given ID already exists.
    #[error("duplicate node: {0}")]
    DuplicateNode(String),

    /// An edge references a node that does not exist.
    #[error("edge endpoint missing: from={from}, to={to}")]
    EdgeEndpointMissing { from: String, to: String },
}

/// Discriminates the semantic role of a knowledge graph node.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum NodeType {
    /// A predicate symbol (e.g. `parent/2`).
    Predicate,
    /// A ground constant (e.g. `"alice"`).
    Constant,
    /// A logical variable (e.g. `X`).
    Variable,
    /// A named rule.
    Rule,
}

/// A single node in the knowledge graph.
#[derive(Debug, Clone)]
pub struct KgNode {
    /// Unique identifier (predicate name, constant literal, etc.).
    pub id: String,
    /// Semantic role of the node.
    pub node_type: NodeType,
    /// Arbitrary key-value metadata.
    pub metadata: HashMap<String, String>,
}

impl KgNode {
    /// Construct a new node with empty metadata.
    pub fn new(id: impl Into<String>, node_type: NodeType) -> Self {
        Self {
            id: id.into(),
            node_type,
            metadata: HashMap::new(),
        }
    }

    /// Construct a new node with pre-populated metadata.
    pub fn with_metadata(
        id: impl Into<String>,
        node_type: NodeType,
        metadata: HashMap<String, String>,
    ) -> Self {
        Self {
            id: id.into(),
            node_type,
            metadata,
        }
    }
}

/// Discriminates the semantic relationship encoded by an edge.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum EdgeType {
    /// Node A uses node B inside a rule body / goal.
    UsesIn,
    /// Node A defines (introduces) node B.
    DefinesIn,
    /// Node A depends on node B for derivation.
    DependsOn,
    /// Node A and node B are logically contradictory.
    Contradicts,
    /// Node A subsumes (is more general than) node B.
    Subsumes,
}

/// A directed, weighted edge between two knowledge graph nodes.
#[derive(Debug, Clone)]
pub struct KgEdge {
    /// Source node ID.
    pub from_id: String,
    /// Target node ID.
    pub to_id: String,
    /// Semantic type of the relationship.
    pub edge_type: EdgeType,
    /// Edge weight (defaults to 1.0).
    pub weight: f32,
}

impl KgEdge {
    /// Create a new edge with default weight 1.0.
    pub fn new(from_id: impl Into<String>, to_id: impl Into<String>, edge_type: EdgeType) -> Self {
        Self {
            from_id: from_id.into(),
            to_id: to_id.into(),
            edge_type,
            weight: 1.0,
        }
    }

    /// Create a new edge with an explicit weight.
    pub fn with_weight(
        from_id: impl Into<String>,
        to_id: impl Into<String>,
        edge_type: EdgeType,
        weight: f32,
    ) -> Self {
        Self {
            from_id: from_id.into(),
            to_id: to_id.into(),
            edge_type,
            weight,
        }
    }
}

/// A directed knowledge graph whose nodes are predicates, constants, variables,
/// or rules, and whose edges encode semantic relationships among them.
///
/// Adjacency is maintained as a map from node ID to a list of indices into
/// the `edges` vector for O(1) lookup of outgoing edges.
#[derive(Debug, Clone, Default)]
pub struct KnowledgeGraph {
    /// All nodes, keyed by their unique identifier.
    pub nodes: HashMap<String, KgNode>,
    /// All edges stored in insertion order.
    pub edges: Vec<KgEdge>,
    /// node_id → indices into `edges` for edges leaving that node.
    pub adjacency: HashMap<String, Vec<usize>>,
}

impl KnowledgeGraph {
    /// Create an empty knowledge graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a node.  Returns `KgError::DuplicateNode` if the ID already exists.
    pub fn add_node(&mut self, node: KgNode) -> Result<(), KgError> {
        if self.nodes.contains_key(&node.id) {
            return Err(KgError::DuplicateNode(node.id));
        }
        // Pre-allocate an adjacency list entry so `neighbors` can always find it.
        self.adjacency.entry(node.id.clone()).or_default();
        self.nodes.insert(node.id.clone(), node);
        Ok(())
    }

    /// Insert an edge.  Both endpoints must already exist as nodes.
    ///
    /// Returns `KgError::EdgeEndpointMissing` if either endpoint is absent.
    pub fn add_edge(&mut self, edge: KgEdge) -> Result<(), KgError> {
        if !self.nodes.contains_key(&edge.from_id) || !self.nodes.contains_key(&edge.to_id) {
            return Err(KgError::EdgeEndpointMissing {
                from: edge.from_id.clone(),
                to: edge.to_id.clone(),
            });
        }
        let idx = self.edges.len();
        self.adjacency
            .entry(edge.from_id.clone())
            .or_default()
            .push(idx);
        self.edges.push(edge);
        Ok(())
    }

    /// Number of nodes in the graph.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of edges in the graph.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Return references to all nodes directly reachable from `node_id` via
    /// outgoing edges (i.e. the immediate successors).
    ///
    /// Returns an empty slice if the node has no outgoing edges or does not
    /// exist.
    pub fn neighbors(&self, node_id: &str) -> Vec<&KgNode> {
        let Some(edge_indices) = self.adjacency.get(node_id) else {
            return Vec::new();
        };
        edge_indices
            .iter()
            .filter_map(|&idx| {
                let edge = self.edges.get(idx)?;
                self.nodes.get(&edge.to_id)
            })
            .collect()
    }
}

/// Provides graph traversal, path-finding, and subgraph extraction over a
/// [`KnowledgeGraph`].
///
/// All algorithms are iterative (no recursion) to avoid stack-overflow on
/// large graphs and to give predictable performance for distributed workloads.
pub struct KnowledgeGraphTraverser {
    /// The underlying knowledge graph.
    pub graph: KnowledgeGraph,
}

impl KnowledgeGraphTraverser {
    /// Wrap a [`KnowledgeGraph`] in a traverser.
    pub fn new(graph: KnowledgeGraph) -> Self {
        Self { graph }
    }

    /// Breadth-first search starting from `start`.
    ///
    /// Returns the node IDs in BFS visit order.  Nodes deeper than
    /// `max_depth` hops from `start` are not visited.  A visited-set
    /// prevents cycles from causing duplicate visits.
    ///
    /// Returns an empty `Vec` if `start` does not exist in the graph.
    pub fn bfs(&self, start: &str, max_depth: usize) -> Vec<String> {
        if !self.graph.nodes.contains_key(start) {
            return Vec::new();
        }

        let mut visited: HashSet<String> = HashSet::new();
        let mut order: Vec<String> = Vec::new();
        // Queue stores (node_id, current_depth).
        let mut queue: VecDeque<(String, usize)> = VecDeque::new();

        visited.insert(start.to_string());
        queue.push_back((start.to_string(), 0));

        while let Some((current, depth)) = queue.pop_front() {
            order.push(current.clone());

            if depth >= max_depth {
                continue;
            }

            if let Some(edge_indices) = self.graph.adjacency.get(&current) {
                for &idx in edge_indices {
                    if let Some(edge) = self.graph.edges.get(idx) {
                        let neighbor = &edge.to_id;
                        if !visited.contains(neighbor) {
                            visited.insert(neighbor.clone());
                            queue.push_back((neighbor.clone(), depth + 1));
                        }
                    }
                }
            }
        }

        order
    }

    /// Depth-first search starting from `start`.
    ///
    /// Returns node IDs in DFS visit order (pre-order).  Nodes deeper than
    /// `max_depth` hops from `start` are not visited.  A visited-set
    /// prevents revisiting nodes in cyclic graphs.
    ///
    /// Returns an empty `Vec` if `start` does not exist in the graph.
    pub fn dfs(&self, start: &str, max_depth: usize) -> Vec<String> {
        if !self.graph.nodes.contains_key(start) {
            return Vec::new();
        }

        let mut visited: HashSet<String> = HashSet::new();
        let mut order: Vec<String> = Vec::new();
        // Stack stores (node_id, depth).
        let mut stack: Vec<(String, usize)> = vec![(start.to_string(), 0)];

        while let Some((current, depth)) = stack.pop() {
            if visited.contains(&current) {
                continue;
            }
            visited.insert(current.clone());
            order.push(current.clone());

            if depth < max_depth {
                // Collect neighbours and push them in reverse so that the
                // first listed neighbour is visited first.
                if let Some(edge_indices) = self.graph.adjacency.get(&current) {
                    let neighbours: Vec<String> = edge_indices
                        .iter()
                        .filter_map(|&idx| {
                            let edge = self.graph.edges.get(idx)?;
                            if !visited.contains(&edge.to_id) {
                                Some(edge.to_id.clone())
                            } else {
                                None
                            }
                        })
                        .collect();
                    for n in neighbours.into_iter().rev() {
                        stack.push((n, depth + 1));
                    }
                }
            }
        }

        order
    }

    /// Find the shortest path (fewest hops) from `from` to `to` using BFS.
    ///
    /// Returns `Some(path)` where `path` is the sequence of node IDs from
    /// `from` to `to` inclusive, or `None` if no path exists or either
    /// endpoint is absent.
    pub fn find_path(&self, from: &str, to: &str) -> Option<Vec<String>> {
        if !self.graph.nodes.contains_key(from) || !self.graph.nodes.contains_key(to) {
            return None;
        }
        if from == to {
            return Some(vec![from.to_string()]);
        }

        // BFS; track predecessor so we can reconstruct the path.
        let mut visited: HashSet<String> = HashSet::new();
        let mut predecessor: HashMap<String, String> = HashMap::new();
        let mut queue: VecDeque<String> = VecDeque::new();

        visited.insert(from.to_string());
        queue.push_back(from.to_string());

        let mut found = false;

        'outer: while let Some(current) = queue.pop_front() {
            if let Some(edge_indices) = self.graph.adjacency.get(&current) {
                for &idx in edge_indices {
                    if let Some(edge) = self.graph.edges.get(idx) {
                        let neighbor = &edge.to_id;
                        if !visited.contains(neighbor) {
                            visited.insert(neighbor.clone());
                            predecessor.insert(neighbor.clone(), current.clone());
                            if neighbor == to {
                                found = true;
                                break 'outer;
                            }
                            queue.push_back(neighbor.clone());
                        }
                    }
                }
            }
        }

        if !found {
            return None;
        }

        // Reconstruct path by walking predecessors backwards.
        let mut path = vec![to.to_string()];
        let mut node = to.to_string();
        while node != from {
            let prev = predecessor.get(&node)?.clone();
            path.push(prev.clone());
            node = prev;
        }
        path.reverse();
        Some(path)
    }

    /// Extract the subgraph reachable from any of the `roots` within `depth`
    /// hops (inclusive).
    ///
    /// The returned [`KnowledgeGraph`] contains all reachable nodes together
    /// with every edge whose both endpoints are in the reachable set.
    pub fn subgraph(&self, roots: &[String], depth: usize) -> KnowledgeGraph {
        // Collect all reachable node IDs via multi-source BFS.
        let mut reachable: HashSet<String> = HashSet::new();
        let mut queue: VecDeque<(String, usize)> = VecDeque::new();

        for root in roots {
            if self.graph.nodes.contains_key(root.as_str()) && !reachable.contains(root) {
                reachable.insert(root.clone());
                queue.push_back((root.clone(), 0));
            }
        }

        while let Some((current, d)) = queue.pop_front() {
            if d >= depth {
                continue;
            }
            if let Some(edge_indices) = self.graph.adjacency.get(&current) {
                for &idx in edge_indices {
                    if let Some(edge) = self.graph.edges.get(idx) {
                        let neighbor = &edge.to_id;
                        if !reachable.contains(neighbor) {
                            reachable.insert(neighbor.clone());
                            queue.push_back((neighbor.clone(), d + 1));
                        }
                    }
                }
            }
        }

        // Build the sub-graph.
        let mut sg = KnowledgeGraph::new();
        for id in &reachable {
            if let Some(node) = self.graph.nodes.get(id) {
                // Ignore duplicate errors — should never occur here since we
                // deduplicate via the `reachable` set.
                let _ = sg.add_node(node.clone());
            }
        }
        for edge in &self.graph.edges {
            if reachable.contains(&edge.from_id) && reachable.contains(&edge.to_id) {
                let _ = sg.add_edge(edge.clone());
            }
        }
        sg
    }

    /// Compute the connected components of the **undirected** version of the
    /// graph using union-find (path-compressed, union-by-rank).
    ///
    /// Returns one `Vec<String>` per component.  Each component's node IDs
    /// are sorted lexicographically.  Components are themselves ordered by
    /// their smallest member.
    pub fn connected_components(&self) -> Vec<Vec<String>> {
        // Collect all node IDs into a deterministic order.
        let mut ids: Vec<String> = self.graph.nodes.keys().cloned().collect();
        ids.sort();

        // Map each ID to a compact integer index.
        let index: HashMap<&str, usize> = ids
            .iter()
            .enumerate()
            .map(|(i, id)| (id.as_str(), i))
            .collect();
        let n = ids.len();

        // parent[i] initially points to itself; rank[i] starts at 0.
        let mut parent: Vec<usize> = (0..n).collect();
        let mut rank: Vec<usize> = vec![0; n];

        fn find(parent: &mut [usize], x: usize) -> usize {
            if parent[x] != x {
                parent[x] = find(parent, parent[x]); // path compression
            }
            parent[x]
        }

        fn union(parent: &mut [usize], rank: &mut [usize], x: usize, y: usize) {
            let rx = find(parent, x);
            let ry = find(parent, y);
            if rx == ry {
                return;
            }
            match rank[rx].cmp(&rank[ry]) {
                std::cmp::Ordering::Less => parent[rx] = ry,
                std::cmp::Ordering::Greater => parent[ry] = rx,
                std::cmp::Ordering::Equal => {
                    parent[ry] = rx;
                    rank[rx] += 1;
                }
            }
        }

        // Treat every edge as undirected: union from_id and to_id.
        for edge in &self.graph.edges {
            if let (Some(&ai), Some(&bi)) = (
                index.get(edge.from_id.as_str()),
                index.get(edge.to_id.as_str()),
            ) {
                union(&mut parent, &mut rank, ai, bi);
            }
        }

        // Group node IDs by their root representative.
        let mut components: HashMap<usize, Vec<String>> = HashMap::new();
        for (i, id) in ids.iter().enumerate() {
            let root = find(&mut parent, i);
            components.entry(root).or_default().push(id.clone());
        }

        // Each component is already in sorted order because `ids` was sorted.
        let mut result: Vec<Vec<String>> = components.into_values().collect();
        // Order components by their first (smallest) element.
        result.sort_by(|a, b| a[0].cmp(&b[0]));
        result
    }

    /// Detect whether the **directed** graph contains at least one cycle.
    ///
    /// Uses iterative DFS with three-colour marking:
    /// - White (0): unvisited
    /// - Gray  (1): on the current DFS path (recursion stack)
    /// - Black (2): fully processed
    pub fn has_cycle(&self) -> bool {
        // 0 = white, 1 = gray, 2 = black
        let mut color: HashMap<&str, u8> = HashMap::new();

        for start_id in self.graph.nodes.keys() {
            if color.get(start_id.as_str()).copied().unwrap_or(0) != 0 {
                continue;
            }
            // Iterative DFS: stack entries are (node_id, iterator_index_into_adjacency).
            // We use an explicit "call stack" of (node, edge_cursor) pairs.
            let mut dfs_stack: Vec<(&str, usize)> = vec![(start_id.as_str(), 0)];
            color.insert(start_id.as_str(), 1); // mark gray

            'dfs: while let Some((node, cursor)) = dfs_stack.last_mut() {
                let node: &str = node; // reborrow for lifetime
                let edge_indices = match self.graph.adjacency.get(node) {
                    Some(v) => v,
                    None => {
                        // No outgoing edges: mark black and pop.
                        color.insert(node, 2);
                        dfs_stack.pop();
                        continue 'dfs;
                    }
                };

                if *cursor < edge_indices.len() {
                    let idx = edge_indices[*cursor];
                    *cursor += 1;
                    if let Some(edge) = self.graph.edges.get(idx) {
                        let neighbor = edge.to_id.as_str();
                        let c = color.get(neighbor).copied().unwrap_or(0);
                        if c == 1 {
                            // Back-edge → cycle found.
                            return true;
                        }
                        if c == 0 {
                            color.insert(neighbor, 1);
                            dfs_stack.push((neighbor, 0));
                        }
                        // c == 2: already fully explored, skip.
                    }
                } else {
                    // All neighbours processed: mark black and pop.
                    color.insert(node, 2);
                    dfs_stack.pop();
                }
            }
        }

        false
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ──────────────────────────────────────────────────────────────

    /// Build a small diamond graph:
    ///   A → B → D
    ///   A → C → D
    fn diamond_graph() -> KnowledgeGraph {
        let mut g = KnowledgeGraph::new();
        for id in ["A", "B", "C", "D"] {
            g.add_node(KgNode::new(id, NodeType::Predicate))
                .expect("test: should succeed");
        }
        g.add_edge(KgEdge::new("A", "B", EdgeType::DependsOn))
            .expect("test: should succeed");
        g.add_edge(KgEdge::new("A", "C", EdgeType::DependsOn))
            .expect("test: should succeed");
        g.add_edge(KgEdge::new("B", "D", EdgeType::DependsOn))
            .expect("test: should succeed");
        g.add_edge(KgEdge::new("C", "D", EdgeType::DependsOn))
            .expect("test: should succeed");
        g
    }

    // ── node / edge basic operations ─────────────────────────────────────────

    #[test]
    fn test_add_node_and_count() {
        let mut g = KnowledgeGraph::new();
        g.add_node(KgNode::new("X", NodeType::Variable))
            .expect("test: should succeed");
        g.add_node(KgNode::new("Y", NodeType::Constant))
            .expect("test: should succeed");
        assert_eq!(g.node_count(), 2);
        assert_eq!(g.edge_count(), 0);
    }

    #[test]
    fn test_duplicate_node_error() {
        let mut g = KnowledgeGraph::new();
        g.add_node(KgNode::new("dup", NodeType::Rule))
            .expect("test: should succeed");
        let err = g.add_node(KgNode::new("dup", NodeType::Rule)).unwrap_err();
        assert!(matches!(err, KgError::DuplicateNode(ref s) if s == "dup"));
    }

    #[test]
    fn test_add_edge_and_count() {
        let mut g = KnowledgeGraph::new();
        g.add_node(KgNode::new("p", NodeType::Predicate))
            .expect("test: should succeed");
        g.add_node(KgNode::new("q", NodeType::Predicate))
            .expect("test: should succeed");
        g.add_edge(KgEdge::new("p", "q", EdgeType::UsesIn))
            .expect("test: should succeed");
        assert_eq!(g.edge_count(), 1);
    }

    #[test]
    fn test_edge_missing_from_endpoint_error() {
        let mut g = KnowledgeGraph::new();
        g.add_node(KgNode::new("exists", NodeType::Predicate))
            .expect("test: should succeed");
        let err = g
            .add_edge(KgEdge::new("missing", "exists", EdgeType::DefinesIn))
            .unwrap_err();
        assert!(matches!(err, KgError::EdgeEndpointMissing { .. }));
    }

    #[test]
    fn test_edge_missing_to_endpoint_error() {
        let mut g = KnowledgeGraph::new();
        g.add_node(KgNode::new("exists", NodeType::Predicate))
            .expect("test: should succeed");
        let err = g
            .add_edge(KgEdge::new("exists", "ghost", EdgeType::DefinesIn))
            .unwrap_err();
        assert!(matches!(err, KgError::EdgeEndpointMissing { .. }));
    }

    // ── neighbors ────────────────────────────────────────────────────────────

    #[test]
    fn test_neighbors_listing() {
        let g = diamond_graph();
        let mut nb: Vec<String> = g.neighbors("A").iter().map(|n| n.id.clone()).collect();
        nb.sort();
        assert_eq!(nb, vec!["B", "C"]);
    }

    #[test]
    fn test_neighbors_leaf_node_is_empty() {
        let g = diamond_graph();
        assert!(g.neighbors("D").is_empty());
    }

    // ── BFS ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_bfs_visit_order() {
        let g = diamond_graph();
        let t = KnowledgeGraphTraverser::new(g);
        let visited = t.bfs("A", 10);
        // A must come first; B and C before D.
        assert_eq!(visited[0], "A");
        let pos_b = visited
            .iter()
            .position(|x| x == "B")
            .expect("test: should succeed");
        let pos_c = visited
            .iter()
            .position(|x| x == "C")
            .expect("test: should succeed");
        let pos_d = visited
            .iter()
            .position(|x| x == "D")
            .expect("test: should succeed");
        assert!(pos_b < pos_d);
        assert!(pos_c < pos_d);
        assert_eq!(visited.len(), 4);
    }

    #[test]
    fn test_bfs_max_depth_cutoff() {
        let g = diamond_graph();
        let t = KnowledgeGraphTraverser::new(g);
        // depth 1: only A and its direct neighbours B, C
        let visited = t.bfs("A", 1);
        assert!(visited.contains(&"A".to_string()));
        assert!(visited.contains(&"B".to_string()));
        assert!(visited.contains(&"C".to_string()));
        assert!(!visited.contains(&"D".to_string()));
    }

    #[test]
    fn test_bfs_nonexistent_start_returns_empty() {
        let g = diamond_graph();
        let t = KnowledgeGraphTraverser::new(g);
        assert!(t.bfs("NOPE", 5).is_empty());
    }

    // ── DFS ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_dfs_visit_order_starts_with_root() {
        let g = diamond_graph();
        let t = KnowledgeGraphTraverser::new(g);
        let visited = t.dfs("A", 10);
        assert_eq!(visited[0], "A");
        assert_eq!(visited.len(), 4);
    }

    #[test]
    fn test_dfs_max_depth_cutoff() {
        let g = diamond_graph();
        let t = KnowledgeGraphTraverser::new(g);
        let visited = t.dfs("A", 1);
        assert!(visited.contains(&"A".to_string()));
        assert!(!visited.contains(&"D".to_string()));
    }

    // ── find_path ────────────────────────────────────────────────────────────

    #[test]
    fn test_find_path_direct() {
        let g = diamond_graph();
        let t = KnowledgeGraphTraverser::new(g);
        let path = t.find_path("A", "D").expect("test: should succeed");
        assert_eq!(path[0], "A");
        assert_eq!(*path.last().expect("test: should succeed"), "D");
        // Shortest path has exactly 3 hops (A→B→D or A→C→D).
        assert_eq!(path.len(), 3);
    }

    #[test]
    fn test_find_path_self() {
        let g = diamond_graph();
        let t = KnowledgeGraphTraverser::new(g);
        let path = t.find_path("A", "A").expect("test: should succeed");
        assert_eq!(path, vec!["A"]);
    }

    #[test]
    fn test_find_path_not_found() {
        let g = diamond_graph();
        let t = KnowledgeGraphTraverser::new(g);
        // D has no outgoing edges, so D→A is unreachable.
        assert!(t.find_path("D", "A").is_none());
    }

    #[test]
    fn test_find_path_missing_node() {
        let g = diamond_graph();
        let t = KnowledgeGraphTraverser::new(g);
        assert!(t.find_path("A", "NOPE").is_none());
    }

    // ── subgraph ─────────────────────────────────────────────────────────────

    #[test]
    fn test_subgraph_contains_correct_nodes() {
        let g = diamond_graph();
        let t = KnowledgeGraphTraverser::new(g);
        let sg = t.subgraph(&["A".to_string()], 1);
        // Depth 1 from A: A, B, C — but not D (which is depth 2).
        assert!(sg.nodes.contains_key("A"));
        assert!(sg.nodes.contains_key("B"));
        assert!(sg.nodes.contains_key("C"));
        assert!(!sg.nodes.contains_key("D"));
    }

    #[test]
    fn test_subgraph_full_depth_includes_all() {
        let g = diamond_graph();
        let t = KnowledgeGraphTraverser::new(g);
        let sg = t.subgraph(&["A".to_string()], 10);
        assert_eq!(sg.node_count(), 4);
    }

    // ── connected_components ─────────────────────────────────────────────────

    #[test]
    fn test_connected_components_disconnected() {
        let mut g = KnowledgeGraph::new();
        // Component 1: P1 → P2
        g.add_node(KgNode::new("P1", NodeType::Predicate))
            .expect("test: should succeed");
        g.add_node(KgNode::new("P2", NodeType::Predicate))
            .expect("test: should succeed");
        g.add_edge(KgEdge::new("P1", "P2", EdgeType::DependsOn))
            .expect("test: should succeed");
        // Component 2: isolated Q1
        g.add_node(KgNode::new("Q1", NodeType::Constant))
            .expect("test: should succeed");

        let t = KnowledgeGraphTraverser::new(g);
        let comps = t.connected_components();
        assert_eq!(comps.len(), 2);
        // Verify each node appears in exactly one component.
        let flat: Vec<String> = comps.iter().flatten().cloned().collect();
        assert_eq!(flat.len(), 3);
        assert!(flat.contains(&"P1".to_string()));
        assert!(flat.contains(&"Q1".to_string()));
    }

    #[test]
    fn test_connected_components_single_component() {
        let g = diamond_graph();
        let t = KnowledgeGraphTraverser::new(g);
        let comps = t.connected_components();
        assert_eq!(comps.len(), 1);
        let mut comp = comps[0].clone();
        comp.sort();
        assert_eq!(comp, vec!["A", "B", "C", "D"]);
    }

    // ── has_cycle ────────────────────────────────────────────────────────────

    #[test]
    fn test_has_cycle_detects_cycle() {
        let mut g = KnowledgeGraph::new();
        for id in ["X", "Y", "Z"] {
            g.add_node(KgNode::new(id, NodeType::Rule))
                .expect("test: should succeed");
        }
        g.add_edge(KgEdge::new("X", "Y", EdgeType::DependsOn))
            .expect("test: should succeed");
        g.add_edge(KgEdge::new("Y", "Z", EdgeType::DependsOn))
            .expect("test: should succeed");
        g.add_edge(KgEdge::new("Z", "X", EdgeType::DependsOn))
            .expect("test: should succeed"); // back-edge

        let t = KnowledgeGraphTraverser::new(g);
        assert!(t.has_cycle());
    }

    #[test]
    fn test_has_cycle_false_on_dag() {
        let g = diamond_graph(); // A→B→D, A→C→D — no back-edges
        let t = KnowledgeGraphTraverser::new(g);
        assert!(!t.has_cycle());
    }

    // ── metadata and edge weight ──────────────────────────────────────────────

    #[test]
    fn test_node_metadata_preserved() {
        let mut meta = HashMap::new();
        meta.insert("arity".to_string(), "2".to_string());
        let node = KgNode::with_metadata("parent", NodeType::Predicate, meta.clone());
        let mut g = KnowledgeGraph::new();
        g.add_node(node).expect("test: should succeed");
        let stored = g.nodes.get("parent").expect("test: should succeed");
        assert_eq!(stored.metadata.get("arity").map(String::as_str), Some("2"));
    }

    #[test]
    fn test_edge_weight_preserved() {
        let mut g = KnowledgeGraph::new();
        g.add_node(KgNode::new("a", NodeType::Predicate))
            .expect("test: should succeed");
        g.add_node(KgNode::new("b", NodeType::Predicate))
            .expect("test: should succeed");
        g.add_edge(KgEdge::with_weight("a", "b", EdgeType::Subsumes, 0.42))
            .expect("test: should succeed");
        let edge = &g.edges[0];
        assert!((edge.weight - 0.42).abs() < f32::EPSILON);
    }
}
