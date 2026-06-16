//! Tensor computation graph partitioner for distributed execution.
//!
//! Partitions a tensor computation graph into balanced subgraphs,
//! minimizing cross-partition communication (cut edges).

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

/// Weight associated with a single computation node in the tensor graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeWeight {
    /// Unique identifier for this node.
    pub node_id: u64,
    /// Estimated floating-point operations (flops) for this node.
    pub compute_cost: u64,
    /// Memory footprint of this node's output tensor in bytes.
    pub memory_bytes: u64,
}

/// A directed edge between two nodes in the tensor computation graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphEdge {
    /// Source node identifier.
    pub from: u64,
    /// Destination node identifier.
    pub to: u64,
    /// Number of bytes transferred across this edge when it is cut.
    pub data_bytes: u64,
}

/// A single partition produced by the partitioner.
#[derive(Debug, Clone)]
pub struct Partition {
    /// Zero-based partition index.
    pub partition_id: usize,
    /// Identifiers of all nodes assigned to this partition.
    pub node_ids: Vec<u64>,
    /// Sum of `compute_cost` for all nodes in this partition.
    pub total_compute: u64,
    /// Sum of `memory_bytes` for all nodes in this partition.
    pub total_memory: u64,
}

impl Partition {
    /// Returns the number of nodes in this partition.
    pub fn size(&self) -> usize {
        self.node_ids.len()
    }
}

/// Statistics describing a completed partitioning.
#[derive(Debug, Clone)]
pub struct PartitionStats {
    /// Number of partitions produced.
    pub num_partitions: usize,
    /// Number of edges whose endpoints are in different partitions.
    pub cut_edges: usize,
    /// Total bytes transferred across all cut edges.
    pub cut_bytes: u64,
    /// Imbalance metric:
    /// `(max_compute − min_compute) / avg_compute`, or `0.0` for one partition.
    pub compute_imbalance: f64,
}

impl PartitionStats {
    /// Returns `true` when `compute_imbalance < threshold`.
    pub fn is_balanced(&self, threshold: f64) -> bool {
        self.compute_imbalance < threshold
    }
}

// ---------------------------------------------------------------------------
// Partitioner
// ---------------------------------------------------------------------------

/// Partitions a tensor computation graph into balanced subgraphs for
/// distributed execution, minimising cross-partition edge traffic.
#[derive(Debug, Default)]
pub struct TensorGraphPartitioner {
    /// All nodes, keyed by their `node_id`.
    pub nodes: HashMap<u64, NodeWeight>,
    /// All directed edges in the graph.
    pub edges: Vec<GraphEdge>,
}

impl TensorGraphPartitioner {
    /// Creates an empty `TensorGraphPartitioner`.
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: Vec::new(),
        }
    }

    /// Inserts (or replaces) a node in the graph.
    pub fn add_node(&mut self, node: NodeWeight) {
        self.nodes.insert(node.node_id, node);
    }

    /// Appends an edge to the graph.
    pub fn add_edge(&mut self, edge: GraphEdge) {
        self.edges.push(edge);
    }

    /// Removes a node and returns `true` if it was present.
    ///
    /// Note: edges referencing the removed node are **not** automatically
    /// removed; callers that need edge consistency should manage this manually.
    pub fn remove_node(&mut self, node_id: u64) -> bool {
        self.nodes.remove(&node_id).is_some()
    }

    /// Partitions the graph into at most `k` parts.
    ///
    /// # Strategy
    ///
    /// 1. If `k == 0` or the graph has no nodes, return an empty vector.
    /// 2. If `k >= node count`, assign one node per partition (only non-empty
    ///    partitions are returned).
    /// 3. Otherwise, sort all nodes by `compute_cost` descending and assign
    ///    each node to the partition with the lowest current total compute cost
    ///    (greedy min-heap approximation via linear scan over `k` partitions).
    pub fn partition(&self, k: usize) -> Vec<Partition> {
        if k == 0 || self.nodes.is_empty() {
            return Vec::new();
        }

        // Collect nodes sorted by compute_cost descending (ties broken by node_id
        // for determinism).
        let mut sorted_nodes: Vec<&NodeWeight> = self.nodes.values().collect();
        sorted_nodes.sort_unstable_by(|a, b| {
            b.compute_cost
                .cmp(&a.compute_cost)
                .then_with(|| a.node_id.cmp(&b.node_id))
        });

        let num_partitions = k.min(sorted_nodes.len());

        // Initialise partitions.
        let mut partitions: Vec<Partition> = (0..num_partitions)
            .map(|pid| Partition {
                partition_id: pid,
                node_ids: Vec::new(),
                total_compute: 0,
                total_memory: 0,
            })
            .collect();

        if num_partitions == sorted_nodes.len() {
            // One node per partition – simple assignment.
            for (node, partition) in sorted_nodes.iter().zip(partitions.iter_mut()) {
                partition.node_ids.push(node.node_id);
                partition.total_compute = node.compute_cost;
                partition.total_memory = node.memory_bytes;
            }
        } else {
            // Greedy: assign each node to the partition with the smallest
            // current total_compute (linear scan, O(n·k)).
            for node in &sorted_nodes {
                let target = partitions
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, p)| p.total_compute)
                    .map(|(i, _)| i)
                    .unwrap_or(0); // safe: num_partitions >= 1

                partitions[target].node_ids.push(node.node_id);
                partitions[target].total_compute = partitions[target]
                    .total_compute
                    .saturating_add(node.compute_cost);
                partitions[target].total_memory = partitions[target]
                    .total_memory
                    .saturating_add(node.memory_bytes);
            }
        }

        // Return only non-empty partitions.
        partitions.retain(|p| !p.node_ids.is_empty());
        partitions
    }

    /// Computes statistics for the given set of partitions.
    pub fn stats(&self, partitions: &[Partition]) -> PartitionStats {
        // Build node → partition_id lookup.
        let mut node_to_partition: HashMap<u64, usize> = HashMap::new();
        for partition in partitions {
            for &nid in &partition.node_ids {
                node_to_partition.insert(nid, partition.partition_id);
            }
        }

        // Count cut edges.
        let mut cut_edges: usize = 0;
        let mut cut_bytes: u64 = 0;
        for edge in &self.edges {
            let from_part = node_to_partition.get(&edge.from);
            let to_part = node_to_partition.get(&edge.to);
            match (from_part, to_part) {
                (Some(fp), Some(tp)) if fp != tp => {
                    cut_edges += 1;
                    cut_bytes = cut_bytes.saturating_add(edge.data_bytes);
                }
                _ => {}
            }
        }

        // Compute imbalance.
        let num_partitions = partitions.len();
        let compute_imbalance = if num_partitions <= 1 {
            0.0_f64
        } else {
            let computes: Vec<u64> = partitions.iter().map(|p| p.total_compute).collect();
            let max_c = computes.iter().copied().max().unwrap_or(0);
            let min_c = computes.iter().copied().min().unwrap_or(0);
            let avg_c: f64 = computes.iter().copied().sum::<u64>() as f64 / num_partitions as f64;
            if avg_c == 0.0 {
                0.0
            } else {
                (max_c - min_c) as f64 / avg_c
            }
        };

        PartitionStats {
            num_partitions,
            cut_edges,
            cut_bytes,
            compute_imbalance,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: build a NodeWeight quickly.
    fn node(id: u64, compute: u64, mem: u64) -> NodeWeight {
        NodeWeight {
            node_id: id,
            compute_cost: compute,
            memory_bytes: mem,
        }
    }

    // Helper: build a GraphEdge quickly.
    fn edge(from: u64, to: u64, bytes: u64) -> GraphEdge {
        GraphEdge {
            from,
            to,
            data_bytes: bytes,
        }
    }

    // 1. Empty graph returns empty vec.
    #[test]
    fn test_empty_graph_returns_empty() {
        let p = TensorGraphPartitioner::new();
        assert!(p.partition(4).is_empty());
    }

    // 2. k == 0 returns empty vec even with nodes.
    #[test]
    fn test_k_zero_returns_empty() {
        let mut p = TensorGraphPartitioner::new();
        p.add_node(node(1, 100, 64));
        assert!(p.partition(0).is_empty());
    }

    // 3. k == 1 → single partition gets all nodes.
    #[test]
    fn test_single_partition_contains_all_nodes() {
        let mut p = TensorGraphPartitioner::new();
        for i in 1..=5_u64 {
            p.add_node(node(i, i * 10, i * 8));
        }
        let parts = p.partition(1);
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0].size(), 5);
        // total_compute should be 10+20+30+40+50 = 150
        assert_eq!(parts[0].total_compute, 150);
        // total_memory should be 8+16+24+32+40 = 120
        assert_eq!(parts[0].total_memory, 120);
    }

    // 4. k >= node count → one node per partition (only non-empty returned).
    #[test]
    fn test_k_greater_than_nodes_one_per_partition() {
        let mut p = TensorGraphPartitioner::new();
        p.add_node(node(10, 50, 8));
        p.add_node(node(20, 50, 8));
        p.add_node(node(30, 50, 8));
        let parts = p.partition(10);
        assert_eq!(parts.len(), 3);
        for part in &parts {
            assert_eq!(part.size(), 1);
        }
    }

    // 5. k == node count → one node per partition exactly.
    #[test]
    fn test_k_equals_node_count() {
        let mut p = TensorGraphPartitioner::new();
        p.add_node(node(1, 100, 10));
        p.add_node(node(2, 200, 20));
        let parts = p.partition(2);
        assert_eq!(parts.len(), 2);
        assert!(parts.iter().all(|pt| pt.size() == 1));
    }

    // 6. Greedy balancing: 2 partitions, nodes with varied compute.
    //    The greedy algorithm should distribute so that large nodes are spread.
    #[test]
    fn test_greedy_balancing_reduces_imbalance() {
        let mut p = TensorGraphPartitioner::new();
        // Four nodes: 100, 100, 50, 50 → ideal 2 parts of 150 each.
        p.add_node(node(1, 100, 0));
        p.add_node(node(2, 100, 0));
        p.add_node(node(3, 50, 0));
        p.add_node(node(4, 50, 0));
        let parts = p.partition(2);
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0].total_compute, 150);
        assert_eq!(parts[1].total_compute, 150);
    }

    // 7. Imbalance is 0 for a single partition.
    #[test]
    fn test_stats_single_partition_zero_imbalance() {
        let mut p = TensorGraphPartitioner::new();
        p.add_node(node(1, 500, 64));
        let parts = p.partition(1);
        let s = p.stats(&parts);
        assert_eq!(s.num_partitions, 1);
        assert!((s.compute_imbalance - 0.0).abs() < f64::EPSILON);
    }

    // 8. compute_imbalance formula verification.
    //    Two partitions: computes 300 and 100 → avg=200, imbalance=(300-100)/200=1.0
    #[test]
    fn test_compute_imbalance_formula() {
        let mut p = TensorGraphPartitioner::new();
        // Force a 2-partition result with known totals by using k >= node_count (1:1).
        p.add_node(node(1, 300, 0));
        p.add_node(node(2, 100, 0));
        let parts = p.partition(2);
        let s = p.stats(&parts);
        // (300-100)/200 = 1.0
        assert!(
            (s.compute_imbalance - 1.0).abs() < 1e-9,
            "imbalance={}",
            s.compute_imbalance
        );
    }

    // 9. is_balanced threshold check.
    #[test]
    fn test_is_balanced_threshold() {
        let mut p = TensorGraphPartitioner::new();
        p.add_node(node(1, 300, 0));
        p.add_node(node(2, 100, 0));
        let parts = p.partition(2);
        let s = p.stats(&parts);
        // imbalance == 1.0 → balanced at threshold 1.1, unbalanced at 0.9
        assert!(s.is_balanced(1.1));
        assert!(!s.is_balanced(0.9));
    }

    // 10. cut_edges count: edges within same partition are not cut.
    #[test]
    fn test_cut_edges_same_partition_not_cut() {
        let mut p = TensorGraphPartitioner::new();
        p.add_node(node(1, 100, 0));
        p.add_node(node(2, 50, 0));
        // k=1 → all in one partition → no cuts.
        p.add_edge(edge(1, 2, 1024));
        let parts = p.partition(1);
        let s = p.stats(&parts);
        assert_eq!(s.cut_edges, 0);
        assert_eq!(s.cut_bytes, 0);
    }

    // 11. cut_edges count across partitions.
    #[test]
    fn test_cut_edges_cross_partition() {
        let mut p = TensorGraphPartitioner::new();
        p.add_node(node(1, 100, 0));
        p.add_node(node(2, 100, 0));
        p.add_edge(edge(1, 2, 512));
        // k=2 → two separate partitions → edge (1→2) is cut.
        let parts = p.partition(2);
        let s = p.stats(&parts);
        assert_eq!(s.cut_edges, 1);
        assert_eq!(s.cut_bytes, 512);
    }

    // 12. cut_bytes sums all cut edge data_bytes.
    #[test]
    fn test_cut_bytes_sum() {
        let mut p = TensorGraphPartitioner::new();
        p.add_node(node(1, 100, 0));
        p.add_node(node(2, 100, 0));
        p.add_edge(edge(1, 2, 200));
        p.add_edge(edge(1, 2, 300));
        let parts = p.partition(2);
        let s = p.stats(&parts);
        // Both edges connect different partitions.
        assert_eq!(s.cut_bytes, 500);
    }

    // 13. Edges referencing unknown nodes are ignored (not counted as cut).
    #[test]
    fn test_edges_with_unknown_nodes_ignored() {
        let mut p = TensorGraphPartitioner::new();
        p.add_node(node(1, 100, 0));
        // node 99 does not exist.
        p.add_edge(edge(1, 99, 999));
        let parts = p.partition(1);
        let s = p.stats(&parts);
        assert_eq!(s.cut_edges, 0);
        assert_eq!(s.cut_bytes, 0);
    }

    // 14. remove_node returns true if present, false otherwise.
    #[test]
    fn test_remove_node_present() {
        let mut p = TensorGraphPartitioner::new();
        p.add_node(node(42, 10, 8));
        assert!(p.remove_node(42));
        assert!(!p.nodes.contains_key(&42));
    }

    // 15. remove_node returns false when node is absent.
    #[test]
    fn test_remove_node_absent() {
        let mut p = TensorGraphPartitioner::new();
        assert!(!p.remove_node(999));
    }

    // 16. add_edge appends correctly.
    #[test]
    fn test_add_edge_appended() {
        let mut p = TensorGraphPartitioner::new();
        p.add_edge(edge(1, 2, 64));
        p.add_edge(edge(2, 3, 128));
        assert_eq!(p.edges.len(), 2);
        assert_eq!(p.edges[0].data_bytes, 64);
        assert_eq!(p.edges[1].data_bytes, 128);
    }

    // 17. Partition size method reflects node_ids length.
    #[test]
    fn test_partition_size_method() {
        let part = Partition {
            partition_id: 0,
            node_ids: vec![1, 2, 3],
            total_compute: 300,
            total_memory: 48,
        };
        assert_eq!(part.size(), 3);
    }

    // 18. stats on multiple partitions with no edges gives zero cuts.
    #[test]
    fn test_stats_no_edges_zero_cuts() {
        let mut p = TensorGraphPartitioner::new();
        for i in 1..=4_u64 {
            p.add_node(node(i, i * 100, i * 8));
        }
        let parts = p.partition(2);
        let s = p.stats(&parts);
        assert_eq!(s.cut_edges, 0);
        assert_eq!(s.cut_bytes, 0);
    }

    // 19. Greedy assigns largest nodes first, checking monotone property.
    //     With nodes [1000,1,1,1] and k=2, the ideal split is [1000] vs [1,1,1]=3.
    //     The greedy produces partition-0: [1000], partition-1: [1,1,1].
    #[test]
    fn test_greedy_largest_first_assignment() {
        let mut p = TensorGraphPartitioner::new();
        p.add_node(node(1, 1000, 0));
        p.add_node(node(2, 1, 0));
        p.add_node(node(3, 1, 0));
        p.add_node(node(4, 1, 0));
        let parts = p.partition(2);
        assert_eq!(parts.len(), 2);
        let computes: Vec<u64> = parts.iter().map(|p| p.total_compute).collect();
        // One partition should have 1000 and the other 3.
        assert!(computes.contains(&1000));
        assert!(computes.contains(&3));
    }

    // 20. num_partitions in stats matches the slice length.
    #[test]
    fn test_stats_num_partitions_matches_slice() {
        let mut p = TensorGraphPartitioner::new();
        for i in 1..=6_u64 {
            p.add_node(node(i, 10, 8));
        }
        let parts = p.partition(3);
        let s = p.stats(&parts);
        assert_eq!(s.num_partitions, parts.len());
    }
}
