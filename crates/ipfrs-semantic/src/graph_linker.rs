//! Semantic Graph Linker — builds a semantic graph by linking embeddings above a similarity
//! threshold, enabling graph-based search and community detection.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// EdgeType
// ---------------------------------------------------------------------------

/// Classifies the semantic relationship between two linked nodes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EdgeType {
    /// Two nodes share similar (but not identical) content.
    SimilarContent,
    /// Two nodes are near-duplicate (cosine similarity ≥ duplicate_threshold).
    Duplicate,
    /// Two nodes are loosely related.
    Related,
    /// Two nodes express opposing/contradictory information (very low similarity).
    Contradictory,
}

// ---------------------------------------------------------------------------
// SemanticEdge
// ---------------------------------------------------------------------------

/// A directed (logically undirected) edge in the semantic graph.
#[derive(Clone, Debug)]
pub struct SemanticEdge {
    pub from_id: u64,
    pub to_id: u64,
    pub similarity: f32,
    pub edge_type: EdgeType,
}

// ---------------------------------------------------------------------------
// GraphNode
// ---------------------------------------------------------------------------

/// A node in the semantic graph, carrying its embedding vector and optional label.
#[derive(Clone, Debug)]
pub struct GraphNode {
    pub id: u64,
    pub embedding: Vec<f32>,
    pub label: Option<String>,
}

impl GraphNode {
    /// Returns the number of edges incident to this node (i.e., where `from_id` or
    /// `to_id` equals `self.id`).
    pub fn degree(&self, edges: &[SemanticEdge]) -> usize {
        edges
            .iter()
            .filter(|e| e.from_id == self.id || e.to_id == self.id)
            .count()
    }
}

// ---------------------------------------------------------------------------
// LinkerConfig
// ---------------------------------------------------------------------------

/// Configuration for `SemanticGraphLinker`.
#[derive(Clone, Debug)]
pub struct LinkerConfig {
    /// Minimum cosine similarity to create a `SimilarContent` edge (default 0.8).
    pub similarity_threshold: f32,
    /// Minimum cosine similarity to classify an edge as `Duplicate` (default 0.99).
    pub duplicate_threshold: f32,
    /// Maximum number of edges stored per node (default 20). Surplus edges are
    /// removed keeping only the highest-similarity ones.
    pub max_edges_per_node: usize,
    /// Maximum cosine similarity to classify a pair as `Contradictory` (default 0.1).
    pub contradiction_threshold: f32,
}

impl Default for LinkerConfig {
    fn default() -> Self {
        Self {
            similarity_threshold: 0.8,
            duplicate_threshold: 0.99,
            max_edges_per_node: 20,
            contradiction_threshold: 0.1,
        }
    }
}

// ---------------------------------------------------------------------------
// GraphLinkerStats
// ---------------------------------------------------------------------------

/// Aggregate statistics about the semantic graph.
#[derive(Clone, Debug, Default)]
pub struct GraphLinkerStats {
    pub node_count: usize,
    pub edge_count: usize,
    pub duplicate_count: usize,
}

impl GraphLinkerStats {
    /// Average degree = 2 * edge_count / max(node_count, 1) (each edge contributes
    /// to two nodes' degree counts).
    pub fn avg_degree(&self) -> f64 {
        (2 * self.edge_count) as f64 / self.node_count.max(1) as f64
    }
}

// ---------------------------------------------------------------------------
// SemanticGraphLinker
// ---------------------------------------------------------------------------

/// Builds and queries a semantic similarity graph over embedding vectors.
pub struct SemanticGraphLinker {
    pub nodes: HashMap<u64, GraphNode>,
    pub edges: Vec<SemanticEdge>,
    pub config: LinkerConfig,
}

impl SemanticGraphLinker {
    /// Creates a new linker with the provided configuration.
    pub fn new(config: LinkerConfig) -> Self {
        Self {
            nodes: HashMap::new(),
            edges: Vec::new(),
            config,
        }
    }

    /// Inserts a node into the graph.  Any existing node with the same `id` is
    /// replaced (its edges are not automatically removed; call `remove_node`
    /// first if you want a clean replacement).
    pub fn add_node(&mut self, node: GraphNode) {
        self.nodes.insert(node.id, node);
    }

    /// Links all node pairs whose cosine similarity exceeds the configured
    /// thresholds, then enforces the `max_edges_per_node` cap.
    ///
    /// Calling this method more than once is safe but will duplicate edges for
    /// pairs that were already linked; it is the caller's responsibility to
    /// clear edges first if a full rebuild is desired.
    pub fn link_all(&mut self) {
        let ids: Vec<u64> = self.nodes.keys().copied().collect();
        let n = ids.len();

        let sim_threshold = self.config.similarity_threshold;
        let dup_threshold = self.config.duplicate_threshold;
        let cont_threshold = self.config.contradiction_threshold;
        let related_threshold = sim_threshold * 0.8;

        for i in 0..n {
            for j in (i + 1)..n {
                let id_a = ids[i];
                let id_b = ids[j];

                let sim =
                    cosine_similarity(&self.nodes[&id_a].embedding, &self.nodes[&id_b].embedding);

                let edge_type = if sim >= dup_threshold {
                    EdgeType::Duplicate
                } else if sim >= sim_threshold {
                    EdgeType::SimilarContent
                } else if sim <= cont_threshold {
                    EdgeType::Contradictory
                } else if sim >= related_threshold {
                    EdgeType::Related
                } else {
                    // Between related_threshold and sim_threshold: skip.
                    continue;
                };

                self.edges.push(SemanticEdge {
                    from_id: id_a,
                    to_id: id_b,
                    similarity: sim,
                    edge_type,
                });
            }
        }

        // Enforce max_edges_per_node.
        self.trim_edges();
    }

    /// Returns the IDs of all nodes directly adjacent to `node_id`.
    pub fn neighbors(&self, node_id: u64) -> Vec<u64> {
        let mut result = Vec::new();
        for edge in &self.edges {
            if edge.from_id == node_id {
                result.push(edge.to_id);
            } else if edge.to_id == node_id {
                result.push(edge.from_id);
            }
        }
        result.sort_unstable();
        result.dedup();
        result
    }

    /// Computes connected components considering only `SimilarContent` and
    /// `Duplicate` edges (using union-find).
    pub fn connected_components(&self) -> Vec<Vec<u64>> {
        let ids: Vec<u64> = self.nodes.keys().copied().collect();
        if ids.is_empty() {
            return Vec::new();
        }

        // Build a mapping id -> index for union-find.
        let mut index_map: HashMap<u64, usize> = HashMap::with_capacity(ids.len());
        for (idx, &id) in ids.iter().enumerate() {
            index_map.insert(id, idx);
        }

        let mut parent: Vec<usize> = (0..ids.len()).collect();
        let mut rank: Vec<u8> = vec![0; ids.len()];

        for edge in &self.edges {
            if edge.edge_type != EdgeType::SimilarContent && edge.edge_type != EdgeType::Duplicate {
                continue;
            }
            if let (Some(&a), Some(&b)) = (index_map.get(&edge.from_id), index_map.get(&edge.to_id))
            {
                union(&mut parent, &mut rank, a, b);
            }
        }

        // Group by root.
        let mut groups: HashMap<usize, Vec<u64>> = HashMap::new();
        for (idx, &id) in ids.iter().enumerate() {
            let root = find(&mut parent, idx);
            groups.entry(root).or_default().push(id);
        }

        let mut components: Vec<Vec<u64>> = groups.into_values().collect();
        for comp in &mut components {
            comp.sort_unstable();
        }
        components.sort_by_key(|c| c[0]);
        components
    }

    /// Removes a node and all edges incident to it from the graph.
    pub fn remove_node(&mut self, node_id: u64) {
        self.nodes.remove(&node_id);
        self.edges
            .retain(|e| e.from_id != node_id && e.to_id != node_id);
    }

    /// Returns aggregate statistics for the current graph.
    pub fn stats(&self) -> GraphLinkerStats {
        let duplicate_count = self
            .edges
            .iter()
            .filter(|e| e.edge_type == EdgeType::Duplicate)
            .count();

        GraphLinkerStats {
            node_count: self.nodes.len(),
            edge_count: self.edges.len(),
            duplicate_count,
        }
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Trims edges so that no node participates in more than `max_edges_per_node`
    /// edges.  When a node exceeds the cap, its lowest-similarity edges are
    /// removed first.
    fn trim_edges(&mut self) {
        let max = self.config.max_edges_per_node;

        // Count edges per node and identify which edges need pruning.
        // We do this in a stable, deterministic way:
        //   1. Sort all edges by similarity (descending) so we prefer to keep
        //      the highest-similarity edges when trimming.
        //   2. Walk the sorted list and track how many edges each node has
        //      accumulated; mark as removed when the cap is hit.

        // Build a list of (original_index, similarity) sorted descending.
        let mut order: Vec<usize> = (0..self.edges.len()).collect();
        order.sort_by(|&a, &b| {
            self.edges[b]
                .similarity
                .partial_cmp(&self.edges[a].similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut degree: HashMap<u64, usize> = HashMap::new();
        let mut keep: Vec<bool> = vec![false; self.edges.len()];

        for idx in order {
            let edge = &self.edges[idx];
            let da = *degree.get(&edge.from_id).unwrap_or(&0);
            let db = *degree.get(&edge.to_id).unwrap_or(&0);
            if da < max && db < max {
                keep[idx] = true;
                *degree.entry(edge.from_id).or_insert(0) += 1;
                *degree.entry(edge.to_id).or_insert(0) += 1;
            }
        }

        let mut kept = Vec::with_capacity(self.edges.len());
        for (idx, edge) in self.edges.drain(..).enumerate() {
            if keep[idx] {
                kept.push(edge);
            }
        }
        self.edges = kept;
    }
}

// ---------------------------------------------------------------------------
// Cosine similarity
// ---------------------------------------------------------------------------

/// Computes the cosine similarity between two vectors.  Returns 0.0 if either
/// vector has zero magnitude.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    if len == 0 {
        return 0.0;
    }

    let mut dot = 0.0_f32;
    let mut mag_a = 0.0_f32;
    let mut mag_b = 0.0_f32;

    for i in 0..len {
        dot += a[i] * b[i];
        mag_a += a[i] * a[i];
        mag_b += b[i] * b[i];
    }

    let denom = mag_a.sqrt() * mag_b.sqrt();
    if denom < f32::EPSILON {
        0.0
    } else {
        (dot / denom).clamp(-1.0, 1.0)
    }
}

// ---------------------------------------------------------------------------
// Union-Find helpers
// ---------------------------------------------------------------------------

fn find(parent: &mut [usize], mut x: usize) -> usize {
    while parent[x] != x {
        parent[x] = parent[parent[x]]; // path compression (halving)
        x = parent[x];
    }
    x
}

fn union(parent: &mut [usize], rank: &mut [u8], x: usize, y: usize) {
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Helper builders
    // -----------------------------------------------------------------------

    fn make_node(id: u64, embedding: Vec<f32>) -> GraphNode {
        GraphNode {
            id,
            embedding,
            label: None,
        }
    }

    fn make_node_labeled(id: u64, embedding: Vec<f32>, label: &str) -> GraphNode {
        GraphNode {
            id,
            embedding,
            label: Some(label.to_string()),
        }
    }

    fn default_linker() -> SemanticGraphLinker {
        SemanticGraphLinker::new(LinkerConfig::default())
    }

    // Produce a unit vector with all entries equal.
    fn uniform_vec(dim: usize, value: f32) -> Vec<f32> {
        let norm = (dim as f32).sqrt();
        vec![value / norm; dim]
    }

    // Two orthogonal vectors (cosine = 0).
    fn orthogonal_pair() -> (Vec<f32>, Vec<f32>) {
        let mut a = vec![0.0_f32; 4];
        let mut b = vec![0.0_f32; 4];
        a[0] = 1.0;
        b[1] = 1.0;
        (a, b)
    }

    // -----------------------------------------------------------------------
    // Test 1: add_node stores the node
    // -----------------------------------------------------------------------
    #[test]
    fn test_add_node_stores_node() {
        let mut linker = default_linker();
        let node = make_node(1, vec![1.0, 0.0, 0.0]);
        linker.add_node(node);
        assert!(linker.nodes.contains_key(&1));
    }

    // -----------------------------------------------------------------------
    // Test 2: add_node with label
    // -----------------------------------------------------------------------
    #[test]
    fn test_add_node_with_label() {
        let mut linker = default_linker();
        let node = make_node_labeled(42, vec![0.5, 0.5], "hello");
        linker.add_node(node);
        assert_eq!(linker.nodes[&42].label.as_deref(), Some("hello"));
    }

    // -----------------------------------------------------------------------
    // Test 3: link_all creates SimilarContent edge for similar vectors
    // -----------------------------------------------------------------------
    #[test]
    fn test_link_all_similar_content() {
        let mut linker = default_linker();
        // Nearly identical vectors: cosine ~= 1.0 but let's make them slightly below
        // duplicate_threshold (0.99) and above similarity_threshold (0.8).
        // We do this by nudging one component slightly.
        let a = vec![1.0_f32, 0.0, 0.0, 0.0];
        let b = vec![0.97_f32, 0.24_f32, 0.0, 0.0]; // cos ≈ 0.97 / 1.0 = 0.97
        linker.add_node(make_node(1, a));
        linker.add_node(make_node(2, b));
        linker.link_all();
        let similar: Vec<_> = linker
            .edges
            .iter()
            .filter(|e| e.edge_type == EdgeType::SimilarContent)
            .collect();
        assert!(
            !similar.is_empty(),
            "expected at least one SimilarContent edge"
        );
    }

    // -----------------------------------------------------------------------
    // Test 4: link_all creates Duplicate edge above duplicate_threshold
    // -----------------------------------------------------------------------
    #[test]
    fn test_link_all_duplicate() {
        let mut linker = default_linker();
        // Two identical vectors → cosine = 1.0 ≥ 0.99.
        let v = vec![1.0_f32, 0.0, 0.0];
        linker.add_node(make_node(1, v.clone()));
        linker.add_node(make_node(2, v));
        linker.link_all();
        let dup: Vec<_> = linker
            .edges
            .iter()
            .filter(|e| e.edge_type == EdgeType::Duplicate)
            .collect();
        assert!(!dup.is_empty(), "expected at least one Duplicate edge");
    }

    // -----------------------------------------------------------------------
    // Test 5: link_all creates Contradictory edge below contradiction_threshold
    // -----------------------------------------------------------------------
    #[test]
    fn test_link_all_contradictory() {
        let mut linker = default_linker();
        let (a, b) = orthogonal_pair(); // cosine = 0.0 ≤ 0.1
        linker.add_node(make_node(1, a));
        linker.add_node(make_node(2, b));
        linker.link_all();
        let cont: Vec<_> = linker
            .edges
            .iter()
            .filter(|e| e.edge_type == EdgeType::Contradictory)
            .collect();
        assert!(!cont.is_empty(), "expected at least one Contradictory edge");
    }

    // -----------------------------------------------------------------------
    // Test 6: link_all creates Related edge in intermediate range
    // -----------------------------------------------------------------------
    #[test]
    fn test_link_all_related() {
        // similarity_threshold = 0.8, related threshold = 0.64.
        // We need cosine in [0.64, 0.80).
        // cos(a, b) = dot / (|a||b|).
        // a = [1,0,0,0], b = [0.7, 0.714, 0, 0] → dot = 0.7, |b| = sqrt(0.49+0.51) = 1.0
        // cosine ≈ 0.7 which is in [0.64, 0.80) ✓
        let mut linker = default_linker();
        let a = vec![1.0_f32, 0.0, 0.0, 0.0];
        let b = vec![0.7_f32, 0.71414_f32, 0.0, 0.0]; // |b| ≈ 1.0, dot ≈ 0.7
        linker.add_node(make_node(1, a));
        linker.add_node(make_node(2, b));
        linker.link_all();
        let related: Vec<_> = linker
            .edges
            .iter()
            .filter(|e| e.edge_type == EdgeType::Related)
            .collect();
        assert!(!related.is_empty(), "expected at least one Related edge");
    }

    // -----------------------------------------------------------------------
    // Test 7: neighbors returns correct adjacent node IDs
    // -----------------------------------------------------------------------
    #[test]
    fn test_neighbors() {
        let mut linker = default_linker();
        let v = vec![1.0_f32, 0.0];
        linker.add_node(make_node(1, v.clone()));
        linker.add_node(make_node(2, v.clone()));
        linker.add_node(make_node(3, vec![0.0, 1.0])); // orthogonal → Contradictory
        linker.link_all();
        // Node 1 and 2 are identical → Duplicate; 1↔3 and 2↔3 are Contradictory.
        let n1 = linker.neighbors(1);
        assert!(n1.contains(&2), "node 1 should be adjacent to node 2");
    }

    // -----------------------------------------------------------------------
    // Test 8: neighbors returns empty for isolated node
    // -----------------------------------------------------------------------
    #[test]
    fn test_neighbors_isolated() {
        let mut linker = SemanticGraphLinker::new(LinkerConfig {
            similarity_threshold: 0.8,
            duplicate_threshold: 0.99,
            max_edges_per_node: 20,
            // Set below -1.0 so even perfectly anti-parallel vectors (cos=-1) are
            // not classified as Contradictory, giving node 10 no edges.
            contradiction_threshold: -1.1,
        });
        // Nodes 1 and 2 are identical (Duplicate edge between them).
        linker.add_node(make_node(1, vec![1.0, 0.0]));
        linker.add_node(make_node(2, vec![1.0, 0.0]));
        // Node 10 is orthogonal to 1 and 2 (cosine = 0.0).
        // 0.0 is not >= similarity_threshold(0.8), not >= related_threshold(0.64),
        // and not <= contradiction_threshold(-1.1), so no edge is created.
        linker.add_node(make_node(10, vec![0.0, 1.0]));
        linker.link_all();
        let n10 = linker.neighbors(10);
        assert!(n10.is_empty(), "node 10 should have no neighbors");
    }

    // -----------------------------------------------------------------------
    // Test 9: connected_components separates two clusters
    // -----------------------------------------------------------------------
    #[test]
    fn test_connected_components_two_clusters() {
        let mut linker = SemanticGraphLinker::new(LinkerConfig {
            similarity_threshold: 0.8,
            duplicate_threshold: 0.99,
            max_edges_per_node: 20,
            contradiction_threshold: 0.0, // no contradictory
        });
        // Cluster A: nodes 1, 2 with identical vectors.
        linker.add_node(make_node(1, vec![1.0, 0.0]));
        linker.add_node(make_node(2, vec![1.0, 0.0]));
        // Cluster B: nodes 3, 4 with identical orthogonal vectors.
        linker.add_node(make_node(3, vec![0.0, 1.0]));
        linker.add_node(make_node(4, vec![0.0, 1.0]));
        linker.link_all();

        let comps = linker.connected_components();
        assert_eq!(comps.len(), 2, "expected 2 connected components");
        let flat: std::collections::HashSet<u64> = comps.iter().flatten().copied().collect();
        assert!(flat.contains(&1) && flat.contains(&2));
        assert!(flat.contains(&3) && flat.contains(&4));
    }

    // -----------------------------------------------------------------------
    // Test 10: connected_components with single node
    // -----------------------------------------------------------------------
    #[test]
    fn test_connected_components_single_node() {
        let mut linker = default_linker();
        linker.add_node(make_node(99, vec![1.0]));
        linker.link_all();
        let comps = linker.connected_components();
        assert_eq!(comps.len(), 1);
        assert_eq!(comps[0], vec![99]);
    }

    // -----------------------------------------------------------------------
    // Test 11: connected_components on empty graph
    // -----------------------------------------------------------------------
    #[test]
    fn test_connected_components_empty() {
        let linker = default_linker();
        let comps = linker.connected_components();
        assert!(comps.is_empty());
    }

    // -----------------------------------------------------------------------
    // Test 12: remove_node removes node and all incident edges
    // -----------------------------------------------------------------------
    #[test]
    fn test_remove_node_cleans_edges() {
        let mut linker = default_linker();
        let v = vec![1.0_f32, 0.0];
        linker.add_node(make_node(1, v.clone()));
        linker.add_node(make_node(2, v));
        linker.link_all();
        assert!(!linker.edges.is_empty(), "should have edges before removal");
        linker.remove_node(1);
        assert!(!linker.nodes.contains_key(&1));
        assert!(
            linker.edges.iter().all(|e| e.from_id != 1 && e.to_id != 1),
            "all edges involving node 1 should be removed"
        );
    }

    // -----------------------------------------------------------------------
    // Test 13: remove_node on non-existent node is a no-op
    // -----------------------------------------------------------------------
    #[test]
    fn test_remove_node_nonexistent() {
        let mut linker = default_linker();
        linker.add_node(make_node(1, vec![1.0, 0.0]));
        linker.link_all();
        let edge_count_before = linker.edges.len();
        linker.remove_node(999); // does not exist
        assert_eq!(linker.edges.len(), edge_count_before);
        assert!(linker.nodes.contains_key(&1));
    }

    // -----------------------------------------------------------------------
    // Test 14: max_edges_per_node cap is enforced
    // -----------------------------------------------------------------------
    #[test]
    fn test_max_edges_per_node_cap() {
        let max = 2_usize;
        let config = LinkerConfig {
            similarity_threshold: 0.0, // link everything
            duplicate_threshold: 0.99,
            max_edges_per_node: max,
            contradiction_threshold: -1.0, // never contradictory
        };
        let mut linker = SemanticGraphLinker::new(config);
        // Add 6 nodes. Every pair will be "similar" (sim_threshold = 0).
        for i in 0..6_u64 {
            linker.add_node(make_node(i, vec![1.0, 0.0]));
        }
        linker.link_all();

        // No node should participate in more than `max` edges.
        for id in linker.nodes.keys() {
            let deg = linker
                .edges
                .iter()
                .filter(|e| e.from_id == *id || e.to_id == *id)
                .count();
            assert!(
                deg <= max,
                "node {id} has degree {deg} which exceeds max {max}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Test 15: stats — node_count and edge_count
    // -----------------------------------------------------------------------
    #[test]
    fn test_stats_counts() {
        let mut linker = default_linker();
        linker.add_node(make_node(1, vec![1.0, 0.0]));
        linker.add_node(make_node(2, vec![1.0, 0.0]));
        linker.link_all();
        let s = linker.stats();
        assert_eq!(s.node_count, 2);
        assert!(s.edge_count >= 1);
    }

    // -----------------------------------------------------------------------
    // Test 16: stats — duplicate_count
    // -----------------------------------------------------------------------
    #[test]
    fn test_stats_duplicate_count() {
        let mut linker = default_linker();
        let v = vec![1.0_f32, 0.0];
        linker.add_node(make_node(1, v.clone()));
        linker.add_node(make_node(2, v));
        linker.link_all();
        let s = linker.stats();
        assert!(
            s.duplicate_count >= 1,
            "expected at least one duplicate edge"
        );
    }

    // -----------------------------------------------------------------------
    // Test 17: avg_degree correctness
    // -----------------------------------------------------------------------
    #[test]
    fn test_avg_degree() {
        let mut linker = default_linker();
        // 3 identical nodes → C(3,2)=3 Duplicate edges; avg_degree = 2*3/3 = 2.0
        let v = vec![1.0_f32, 0.0];
        linker.add_node(make_node(1, v.clone()));
        linker.add_node(make_node(2, v.clone()));
        linker.add_node(make_node(3, v));
        linker.link_all();
        let s = linker.stats();
        let expected = (2 * s.edge_count) as f64 / 3.0;
        let diff = (s.avg_degree() - expected).abs();
        assert!(
            diff < 1e-10,
            "avg_degree mismatch: {} vs {}",
            s.avg_degree(),
            expected
        );
    }

    // -----------------------------------------------------------------------
    // Test 18: avg_degree on empty graph (should not panic)
    // -----------------------------------------------------------------------
    #[test]
    fn test_avg_degree_empty() {
        let s = GraphLinkerStats::default();
        assert_eq!(s.avg_degree(), 0.0);
    }

    // -----------------------------------------------------------------------
    // Test 19: GraphNode::degree counts correctly
    // -----------------------------------------------------------------------
    #[test]
    fn test_graph_node_degree() {
        let node = make_node(5, vec![1.0, 0.0]);
        let edges = vec![
            SemanticEdge {
                from_id: 5,
                to_id: 1,
                similarity: 0.9,
                edge_type: EdgeType::SimilarContent,
            },
            SemanticEdge {
                from_id: 2,
                to_id: 5,
                similarity: 0.85,
                edge_type: EdgeType::SimilarContent,
            },
            SemanticEdge {
                from_id: 3,
                to_id: 4,
                similarity: 0.9,
                edge_type: EdgeType::SimilarContent,
            },
        ];
        assert_eq!(node.degree(&edges), 2);
    }

    // -----------------------------------------------------------------------
    // Test 20: EdgeType equality and copy
    // -----------------------------------------------------------------------
    #[test]
    fn test_edge_type_equality() {
        let a = EdgeType::Duplicate;
        let b = a; // Copy
        assert_eq!(a, b);
        assert_ne!(EdgeType::Related, EdgeType::Contradictory);
    }

    // -----------------------------------------------------------------------
    // Test 21: cosine_similarity zero-vector safety
    // -----------------------------------------------------------------------
    #[test]
    fn test_cosine_zero_vector() {
        let zero = vec![0.0_f32; 3];
        let v = vec![1.0_f32, 0.0, 0.0];
        assert_eq!(cosine_similarity(&zero, &v), 0.0);
        assert_eq!(cosine_similarity(&zero, &zero), 0.0);
    }

    // -----------------------------------------------------------------------
    // Test 22: link_all on empty graph is a no-op
    // -----------------------------------------------------------------------
    #[test]
    fn test_link_all_empty_graph() {
        let mut linker = default_linker();
        linker.link_all(); // should not panic
        assert!(linker.edges.is_empty());
    }

    // -----------------------------------------------------------------------
    // Test 23: uniform vectors produce high cosine similarity
    // -----------------------------------------------------------------------
    #[test]
    fn test_uniform_vectors_high_similarity() {
        let a = uniform_vec(128, 1.0);
        let b = uniform_vec(128, 1.0);
        let sim = cosine_similarity(&a, &b);
        assert!(
            (sim - 1.0).abs() < 1e-5,
            "uniform identical vectors should have cosine ≈ 1"
        );
    }
}
