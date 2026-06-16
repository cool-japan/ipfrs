//! Tensor Dependency Graph — tracks data dependencies between tensors and rules,
//! enabling incremental recomputation when tensors are updated.

use std::collections::{HashMap, HashSet, VecDeque};

// ---------------------------------------------------------------------------
// DependencyKind
// ---------------------------------------------------------------------------

/// The semantic relationship represented by a dependency edge.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DependencyKind {
    /// The source tensor is an input to the destination rule.
    TensorInput,
    /// The destination tensor is produced by the source rule.
    TensorOutput,
    /// The source rule implies the destination rule must re-run.
    RuleImplication,
    /// Two rules share a fact dependency.
    SharedFact,
}

// ---------------------------------------------------------------------------
// DependencyEdge
// ---------------------------------------------------------------------------

/// A directed dependency edge between two graph nodes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DependencyEdge {
    /// Source node ID.
    pub from_id: u64,
    /// Target node ID.
    pub to_id: u64,
    /// The semantic kind of the dependency.
    pub kind: DependencyKind,
    /// Edge weight: 1 = normal, higher = stronger dependency.
    pub weight: u32,
}

// ---------------------------------------------------------------------------
// DirtySet
// ---------------------------------------------------------------------------

/// Set of node IDs that require recomputation.
#[derive(Clone, Debug, Default)]
pub struct DirtySet {
    pub dirty: HashSet<u64>,
}

impl DirtySet {
    /// Create a new, empty `DirtySet`.
    pub fn new() -> Self {
        Self {
            dirty: HashSet::new(),
        }
    }

    /// Mark `id` as needing recomputation.
    pub fn mark_dirty(&mut self, id: u64) {
        self.dirty.insert(id);
    }

    /// Returns `true` if `id` is currently dirty.
    pub fn is_dirty(&self, id: u64) -> bool {
        self.dirty.contains(&id)
    }

    /// Clear the dirty flag for `id`.
    pub fn clear_dirty(&mut self, id: u64) {
        self.dirty.remove(&id);
    }

    /// Return all dirty IDs, sorted ascending.
    pub fn all_dirty(&self) -> Vec<u64> {
        let mut v: Vec<u64> = self.dirty.iter().copied().collect();
        v.sort_unstable();
        v
    }
}

// ---------------------------------------------------------------------------
// GraphStats
// ---------------------------------------------------------------------------

/// Statistics snapshot for a [`TensorDependencyGraph`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GraphStats {
    pub node_count: usize,
    pub edge_count: usize,
    pub dirty_count: usize,
    pub max_in_degree: usize,
    pub max_out_degree: usize,
}

// ---------------------------------------------------------------------------
// TensorDependencyGraph
// ---------------------------------------------------------------------------

/// Tracks data dependencies between tensors and rules, enabling incremental
/// recomputation when tensors are updated.
pub struct TensorDependencyGraph {
    /// All registered node IDs.
    pub nodes: HashSet<u64>,
    /// All dependency edges.
    pub edges: Vec<DependencyEdge>,
    /// Set of dirty (stale) node IDs.
    pub dirty: DirtySet,
}

impl TensorDependencyGraph {
    /// Create a new, empty graph.
    pub fn new() -> Self {
        Self {
            nodes: HashSet::new(),
            edges: Vec::new(),
            dirty: DirtySet::new(),
        }
    }

    /// Register a node — idempotent.
    pub fn add_node(&mut self, id: u64) {
        self.nodes.insert(id);
    }

    /// Add a dependency edge.  Also registers the `from` and `to` nodes.
    pub fn add_edge(&mut self, edge: DependencyEdge) {
        self.nodes.insert(edge.from_id);
        self.nodes.insert(edge.to_id);
        self.edges.push(edge);
    }

    /// Remove a node and all edges that reference it.  Clears any dirty flag.
    pub fn remove_node(&mut self, id: u64) {
        self.nodes.remove(&id);
        self.edges.retain(|e| e.from_id != id && e.to_id != id);
        self.dirty.clear_dirty(id);
    }

    /// Mark `id` dirty and transitively propagate to all reachable dependents
    /// via `TensorOutput`, `RuleImplication`, and `SharedFact` edges where
    /// `from_id == id`.
    pub fn mark_dirty(&mut self, id: u64) {
        // BFS over outgoing edges with the three propagating kinds.
        let mut queue: VecDeque<u64> = VecDeque::new();
        self.dirty.mark_dirty(id);
        queue.push_back(id);

        while let Some(current) = queue.pop_front() {
            // Collect successors first to avoid borrow issues.
            let successors: Vec<u64> = self
                .edges
                .iter()
                .filter(|e| {
                    e.from_id == current
                        && matches!(
                            e.kind,
                            DependencyKind::TensorOutput
                                | DependencyKind::RuleImplication
                                | DependencyKind::SharedFact
                        )
                })
                .map(|e| e.to_id)
                .collect();

            for succ in successors {
                if !self.dirty.is_dirty(succ) {
                    self.dirty.mark_dirty(succ);
                    queue.push_back(succ);
                }
            }
        }
    }

    /// Topological sort (Kahn's algorithm) of dirty nodes only.
    /// Returns nodes in processing order (dependencies before dependents).
    /// On cycle detection, appends remaining nodes in arbitrary order.
    pub fn recompute_order(&self) -> Vec<u64> {
        let dirty: HashSet<u64> = self.dirty.dirty.clone();
        if dirty.is_empty() {
            return Vec::new();
        }

        // Build in-degree map considering only edges within the dirty subgraph.
        let mut in_degree: HashMap<u64, usize> = dirty.iter().map(|&n| (n, 0)).collect();
        // Adjacency list restricted to dirty nodes.
        let mut adj: HashMap<u64, Vec<u64>> = dirty.iter().map(|&n| (n, Vec::new())).collect();

        for edge in &self.edges {
            if dirty.contains(&edge.from_id) && dirty.contains(&edge.to_id) {
                adj.entry(edge.from_id).or_default().push(edge.to_id);
                *in_degree.entry(edge.to_id).or_insert(0) += 1;
            }
        }

        let mut queue: VecDeque<u64> = in_degree
            .iter()
            .filter(|(_, &deg)| deg == 0)
            .map(|(&n, _)| n)
            .collect();

        // Deterministic ordering within equal in-degree tiers.
        let mut sorted_queue: Vec<u64> = queue.drain(..).collect();
        sorted_queue.sort_unstable();
        queue.extend(sorted_queue);

        let mut result: Vec<u64> = Vec::with_capacity(dirty.len());

        while let Some(node) = queue.pop_front() {
            result.push(node);
            if let Some(neighbors) = adj.get(&node) {
                let mut next_batch: Vec<u64> = Vec::new();
                for &neighbor in neighbors {
                    if let Some(deg) = in_degree.get_mut(&neighbor) {
                        *deg = deg.saturating_sub(1);
                        if *deg == 0 {
                            next_batch.push(neighbor);
                        }
                    }
                }
                next_batch.sort_unstable();
                queue.extend(next_batch);
            }
        }

        // Handle cycle — append remaining nodes in sorted order.
        if result.len() < dirty.len() {
            let processed: HashSet<u64> = result.iter().copied().collect();
            let mut remaining: Vec<u64> = dirty
                .iter()
                .filter(|n| !processed.contains(n))
                .copied()
                .collect();
            remaining.sort_unstable();
            result.extend(remaining);
        }

        result
    }

    /// Direct successors of `id` (edges where `from_id == id`), sorted ascending.
    pub fn dependents_of(&self, id: u64) -> Vec<u64> {
        let mut out: Vec<u64> = self
            .edges
            .iter()
            .filter(|e| e.from_id == id)
            .map(|e| e.to_id)
            .collect();
        out.sort_unstable();
        out.dedup();
        out
    }

    /// Direct predecessors of `id` (edges where `to_id == id`), sorted ascending.
    pub fn dependencies_of(&self, id: u64) -> Vec<u64> {
        let mut out: Vec<u64> = self
            .edges
            .iter()
            .filter(|e| e.to_id == id)
            .map(|e| e.from_id)
            .collect();
        out.sort_unstable();
        out.dedup();
        out
    }

    /// Compute and return a statistics snapshot.
    pub fn stats(&self) -> GraphStats {
        let node_count = self.nodes.len();
        let edge_count = self.edges.len();
        let dirty_count = self.dirty.dirty.len();

        let mut in_degree: HashMap<u64, usize> = HashMap::new();
        let mut out_degree: HashMap<u64, usize> = HashMap::new();

        for node in &self.nodes {
            in_degree.entry(*node).or_insert(0);
            out_degree.entry(*node).or_insert(0);
        }

        for edge in &self.edges {
            *out_degree.entry(edge.from_id).or_insert(0) += 1;
            *in_degree.entry(edge.to_id).or_insert(0) += 1;
        }

        let max_in_degree = in_degree.values().copied().max().unwrap_or(0);
        let max_out_degree = out_degree.values().copied().max().unwrap_or(0);

        GraphStats {
            node_count,
            edge_count,
            dirty_count,
            max_in_degree,
            max_out_degree,
        }
    }

    /// Clear the entire dirty set.
    pub fn clear_all_dirty(&mut self) {
        self.dirty.dirty.clear();
    }
}

impl Default for TensorDependencyGraph {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn edge(from: u64, to: u64, kind: DependencyKind) -> DependencyEdge {
        DependencyEdge {
            from_id: from,
            to_id: to,
            kind,
            weight: 1,
        }
    }

    fn edge_w(from: u64, to: u64, kind: DependencyKind, weight: u32) -> DependencyEdge {
        DependencyEdge {
            from_id: from,
            to_id: to,
            kind,
            weight,
        }
    }

    // 1. Empty graph stats
    #[test]
    fn test_empty_graph_stats() {
        let g = TensorDependencyGraph::new();
        let s = g.stats();
        assert_eq!(s.node_count, 0);
        assert_eq!(s.edge_count, 0);
        assert_eq!(s.dirty_count, 0);
        assert_eq!(s.max_in_degree, 0);
        assert_eq!(s.max_out_degree, 0);
    }

    // 2. add_node idempotent
    #[test]
    fn test_add_node_idempotent() {
        let mut g = TensorDependencyGraph::new();
        g.add_node(1);
        g.add_node(1);
        g.add_node(1);
        assert_eq!(g.stats().node_count, 1);
    }

    // 3. add_edge registers nodes
    #[test]
    fn test_add_edge_registers_nodes() {
        let mut g = TensorDependencyGraph::new();
        g.add_edge(edge(10, 20, DependencyKind::TensorInput));
        assert!(g.nodes.contains(&10));
        assert!(g.nodes.contains(&20));
        assert_eq!(g.stats().node_count, 2);
    }

    // 4. remove_node cleans edges
    #[test]
    fn test_remove_node_cleans_edges() {
        let mut g = TensorDependencyGraph::new();
        g.add_edge(edge(1, 2, DependencyKind::TensorOutput));
        g.add_edge(edge(2, 3, DependencyKind::RuleImplication));
        g.remove_node(2);
        assert_eq!(g.edges.len(), 0);
        assert!(!g.nodes.contains(&2));
    }

    // 5. remove_node clears dirty
    #[test]
    fn test_remove_node_clears_dirty() {
        let mut g = TensorDependencyGraph::new();
        g.add_node(5);
        g.dirty.mark_dirty(5);
        assert!(g.dirty.is_dirty(5));
        g.remove_node(5);
        assert!(!g.dirty.is_dirty(5));
    }

    // 6. mark_dirty propagates transitively
    #[test]
    fn test_mark_dirty_propagates_transitively() {
        let mut g = TensorDependencyGraph::new();
        // chain: 1 ->TensorOutput-> 2 ->RuleImplication-> 3 ->SharedFact-> 4
        g.add_edge(edge(1, 2, DependencyKind::TensorOutput));
        g.add_edge(edge(2, 3, DependencyKind::RuleImplication));
        g.add_edge(edge(3, 4, DependencyKind::SharedFact));
        g.mark_dirty(1);
        assert!(g.dirty.is_dirty(1));
        assert!(g.dirty.is_dirty(2));
        assert!(g.dirty.is_dirty(3));
        assert!(g.dirty.is_dirty(4));
    }

    // 7. mark_dirty single node no propagation if no outgoing edges
    #[test]
    fn test_mark_dirty_no_outgoing() {
        let mut g = TensorDependencyGraph::new();
        g.add_node(42);
        g.mark_dirty(42);
        assert!(g.dirty.is_dirty(42));
        assert_eq!(g.dirty.dirty.len(), 1);
    }

    // 8. mark_dirty does NOT propagate over TensorInput edges
    #[test]
    fn test_mark_dirty_no_propagate_tensor_input() {
        let mut g = TensorDependencyGraph::new();
        g.add_edge(edge(1, 2, DependencyKind::TensorInput));
        g.mark_dirty(1);
        assert!(g.dirty.is_dirty(1));
        assert!(!g.dirty.is_dirty(2));
    }

    // 9. recompute_order empty dirty
    #[test]
    fn test_recompute_order_empty_dirty() {
        let g = TensorDependencyGraph::new();
        assert!(g.recompute_order().is_empty());
    }

    // 10. recompute_order single dirty node
    #[test]
    fn test_recompute_order_single_dirty() {
        let mut g = TensorDependencyGraph::new();
        g.add_node(7);
        g.mark_dirty(7);
        let order = g.recompute_order();
        assert_eq!(order, vec![7]);
    }

    // 11. recompute_order respects topo order (dependency before dependent)
    #[test]
    fn test_recompute_order_topo() {
        let mut g = TensorDependencyGraph::new();
        // A -> B -> C
        g.add_edge(edge(1, 2, DependencyKind::TensorOutput));
        g.add_edge(edge(2, 3, DependencyKind::TensorOutput));
        g.mark_dirty(1);
        let order = g.recompute_order();
        let pos = |n: u64| {
            order
                .iter()
                .position(|&x| x == n)
                .expect("test: should succeed")
        };
        assert!(pos(1) < pos(2));
        assert!(pos(2) < pos(3));
    }

    // 12. recompute_order handles diamond dependency
    #[test]
    fn test_recompute_order_diamond() {
        let mut g = TensorDependencyGraph::new();
        //        1
        //       / \
        //      2   3
        //       \ /
        //        4
        g.add_edge(edge(1, 2, DependencyKind::TensorOutput));
        g.add_edge(edge(1, 3, DependencyKind::TensorOutput));
        g.add_edge(edge(2, 4, DependencyKind::TensorOutput));
        g.add_edge(edge(3, 4, DependencyKind::TensorOutput));
        g.mark_dirty(1);
        let order = g.recompute_order();
        let pos = |n: u64| {
            order
                .iter()
                .position(|&x| x == n)
                .expect("test: should succeed")
        };
        assert!(pos(1) < pos(2));
        assert!(pos(1) < pos(3));
        assert!(pos(2) < pos(4));
        assert!(pos(3) < pos(4));
        assert_eq!(order.len(), 4);
    }

    // 13. dependents_of correct
    #[test]
    fn test_dependents_of() {
        let mut g = TensorDependencyGraph::new();
        g.add_edge(edge(1, 3, DependencyKind::TensorOutput));
        g.add_edge(edge(1, 2, DependencyKind::RuleImplication));
        g.add_edge(edge(5, 1, DependencyKind::SharedFact));
        let deps = g.dependents_of(1);
        assert_eq!(deps, vec![2, 3]);
    }

    // 14. dependencies_of correct
    #[test]
    fn test_dependencies_of() {
        let mut g = TensorDependencyGraph::new();
        g.add_edge(edge(2, 1, DependencyKind::TensorOutput));
        g.add_edge(edge(3, 1, DependencyKind::RuleImplication));
        g.add_edge(edge(1, 5, DependencyKind::SharedFact));
        let deps = g.dependencies_of(1);
        assert_eq!(deps, vec![2, 3]);
    }

    // 15. DirtySet mark/check/clear
    #[test]
    fn test_dirtyset_mark_check_clear() {
        let mut ds = DirtySet::new();
        assert!(!ds.is_dirty(1));
        ds.mark_dirty(1);
        assert!(ds.is_dirty(1));
        ds.clear_dirty(1);
        assert!(!ds.is_dirty(1));
    }

    // 16. DirtySet all_dirty sorted
    #[test]
    fn test_dirtyset_all_dirty_sorted() {
        let mut ds = DirtySet::new();
        ds.mark_dirty(5);
        ds.mark_dirty(2);
        ds.mark_dirty(8);
        ds.mark_dirty(1);
        assert_eq!(ds.all_dirty(), vec![1, 2, 5, 8]);
    }

    // 17. GraphStats node_count / edge_count
    #[test]
    fn test_stats_node_edge_count() {
        let mut g = TensorDependencyGraph::new();
        g.add_node(1);
        g.add_node(2);
        g.add_edge(edge(1, 2, DependencyKind::TensorInput));
        g.add_edge(edge(1, 2, DependencyKind::TensorOutput));
        let s = g.stats();
        assert_eq!(s.node_count, 2);
        assert_eq!(s.edge_count, 2);
    }

    // 18. max_in_degree computed correctly
    #[test]
    fn test_max_in_degree() {
        let mut g = TensorDependencyGraph::new();
        g.add_edge(edge(1, 3, DependencyKind::TensorOutput));
        g.add_edge(edge(2, 3, DependencyKind::TensorOutput));
        g.add_edge(edge(4, 3, DependencyKind::TensorOutput));
        let s = g.stats();
        assert_eq!(s.max_in_degree, 3);
    }

    // 19. max_out_degree computed correctly
    #[test]
    fn test_max_out_degree() {
        let mut g = TensorDependencyGraph::new();
        g.add_edge(edge(1, 2, DependencyKind::TensorOutput));
        g.add_edge(edge(1, 3, DependencyKind::TensorOutput));
        g.add_edge(edge(1, 4, DependencyKind::TensorOutput));
        let s = g.stats();
        assert_eq!(s.max_out_degree, 3);
    }

    // 20. clear_all_dirty empties dirty set
    #[test]
    fn test_clear_all_dirty() {
        let mut g = TensorDependencyGraph::new();
        g.add_node(1);
        g.add_node(2);
        g.mark_dirty(1);
        g.mark_dirty(2);
        assert_eq!(g.dirty.dirty.len(), 2);
        g.clear_all_dirty();
        assert!(g.dirty.dirty.is_empty());
    }

    // 21. cycle handling doesn't panic
    #[test]
    fn test_cycle_no_panic() {
        let mut g = TensorDependencyGraph::new();
        // A -> B -> A (cycle)
        g.add_edge(edge(1, 2, DependencyKind::RuleImplication));
        g.add_edge(edge(2, 1, DependencyKind::RuleImplication));
        g.mark_dirty(1);
        let order = g.recompute_order();
        // Both nodes should be in the result, no panic.
        assert_eq!(order.len(), 2);
        assert!(order.contains(&1));
        assert!(order.contains(&2));
    }

    // 22. DependencyKind variants accessible
    #[test]
    fn test_dependency_kind_variants() {
        let kinds = [
            DependencyKind::TensorInput,
            DependencyKind::TensorOutput,
            DependencyKind::RuleImplication,
            DependencyKind::SharedFact,
        ];
        // Each variant should be copy-able and comparable.
        for k in kinds {
            let k2 = k;
            assert_eq!(k, k2);
        }
    }

    // 23. weight preserved in edge
    #[test]
    fn test_weight_preserved() {
        let mut g = TensorDependencyGraph::new();
        g.add_edge(edge_w(1, 2, DependencyKind::TensorOutput, 42));
        let e = &g.edges[0];
        assert_eq!(e.weight, 42);
        assert_eq!(e.from_id, 1);
        assert_eq!(e.to_id, 2);
    }

    // 24. stats dirty_count reflects current dirty state
    #[test]
    fn test_stats_dirty_count() {
        let mut g = TensorDependencyGraph::new();
        g.add_node(1);
        g.add_node(2);
        g.add_node(3);
        g.mark_dirty(1);
        g.mark_dirty(3);
        let s = g.stats();
        assert_eq!(s.dirty_count, 2);
    }

    // 25. remove_node leaves unrelated edges intact
    #[test]
    fn test_remove_node_leaves_other_edges() {
        let mut g = TensorDependencyGraph::new();
        g.add_edge(edge(1, 2, DependencyKind::TensorOutput));
        g.add_edge(edge(3, 4, DependencyKind::TensorOutput));
        g.remove_node(1);
        assert_eq!(g.edges.len(), 1);
        assert_eq!(g.edges[0].from_id, 3);
    }

    // 26. mark_dirty does not re-enqueue already-dirty nodes (termination)
    #[test]
    fn test_mark_dirty_terminates_with_shared_targets() {
        let mut g = TensorDependencyGraph::new();
        g.add_edge(edge(1, 2, DependencyKind::TensorOutput));
        g.add_edge(edge(1, 2, DependencyKind::TensorOutput)); // duplicate edge
        g.mark_dirty(1);
        assert!(g.dirty.is_dirty(2));
        // Should not infinite-loop.
    }
}
