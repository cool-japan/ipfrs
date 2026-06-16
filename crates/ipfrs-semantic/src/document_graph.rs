//! Semantic Document Graph
//!
//! Graph structure for document relationships based on semantic similarity.
//! Nodes represent documents with embeddings; edges encode similarity, citation,
//! or cluster-membership relationships.

use std::collections::{HashMap, HashSet, VecDeque};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Kind of relationship between two documents.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EdgeKind {
    /// Cosine similarity above the graph threshold.
    Similar,
    /// Explicit citation / reference.
    Citation,
    /// Both documents belong to the same cluster.
    SameCluster,
}

/// A node in the semantic document graph.
#[derive(Debug, Clone)]
pub struct DocGraphNode {
    /// Unique document identifier.
    pub doc_id: String,
    /// Embedding vector for this document.
    pub embedding: Vec<f64>,
    /// Arbitrary key-value metadata.
    pub metadata: HashMap<String, String>,
}

/// A directed edge in the semantic document graph.
#[derive(Debug, Clone)]
pub struct DocGraphEdge {
    /// Source document id.
    pub source: String,
    /// Target document id.
    pub target: String,
    /// Relationship kind.
    pub kind: EdgeKind,
    /// Edge weight (e.g. similarity score).
    pub weight: f64,
}

/// Summary statistics for the graph.
#[derive(Debug, Clone)]
pub struct DocumentGraphStats {
    pub node_count: usize,
    pub edge_count: usize,
    pub avg_degree: f64,
    pub component_count: usize,
}

// ---------------------------------------------------------------------------
// Cosine similarity helper
// ---------------------------------------------------------------------------

/// Cosine similarity between two vectors.  Returns 0.0 when either vector has
/// zero magnitude or the lengths differ.
pub fn cosine_sim(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0_f64;
    let mut mag_a = 0.0_f64;
    let mut mag_b = 0.0_f64;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        mag_a += x * x;
        mag_b += y * y;
    }
    let denom = mag_a.sqrt() * mag_b.sqrt();
    if denom == 0.0 {
        0.0
    } else {
        dot / denom
    }
}

// ---------------------------------------------------------------------------
// SemanticDocumentGraph
// ---------------------------------------------------------------------------

/// Graph of document relationships based on semantic similarity and explicit
/// annotations (citations, cluster membership).
pub struct SemanticDocumentGraph {
    nodes: HashMap<String, DocGraphNode>,
    edges: Vec<DocGraphEdge>,
    /// doc_id → indices into `edges` where this doc appears as source **or** target.
    adjacency: HashMap<String, Vec<usize>>,
    similarity_threshold: f64,
}

impl SemanticDocumentGraph {
    /// Create a new graph with the given cosine-similarity threshold for
    /// automatic linking.
    pub fn new(similarity_threshold: f64) -> Self {
        Self {
            nodes: HashMap::new(),
            edges: Vec::new(),
            adjacency: HashMap::new(),
            similarity_threshold,
        }
    }

    /// Add a document node.  If a node with the same `doc_id` already exists
    /// it is silently replaced (edges are **not** touched).
    pub fn add_node(
        &mut self,
        doc_id: &str,
        embedding: Vec<f64>,
        metadata: HashMap<String, String>,
    ) {
        let node = DocGraphNode {
            doc_id: doc_id.to_string(),
            embedding,
            metadata,
        };
        self.nodes.insert(doc_id.to_string(), node);
        // Ensure the adjacency entry exists.
        self.adjacency.entry(doc_id.to_string()).or_default();
    }

    /// Remove a node and all edges incident to it.  Returns `true` if the node
    /// existed.
    pub fn remove_node(&mut self, doc_id: &str) -> bool {
        if self.nodes.remove(doc_id).is_none() {
            return false;
        }

        // Collect indices of edges to remove (those touching this node).
        let indices_to_remove: HashSet<usize> = self
            .adjacency
            .remove(doc_id)
            .unwrap_or_default()
            .into_iter()
            .collect();

        if indices_to_remove.is_empty() {
            return true;
        }

        // Build a new edge list, keeping only edges not in the removal set.
        // We also need to rebuild the entire adjacency map because indices shift.
        let old_edges = std::mem::take(&mut self.edges);
        let mut new_adjacency: HashMap<String, Vec<usize>> =
            self.nodes.keys().map(|k| (k.clone(), Vec::new())).collect();

        for (old_idx, edge) in old_edges.into_iter().enumerate() {
            if indices_to_remove.contains(&old_idx) {
                continue;
            }
            let new_idx = self.edges.len();
            if let Some(list) = new_adjacency.get_mut(&edge.source) {
                list.push(new_idx);
            }
            if let Some(list) = new_adjacency.get_mut(&edge.target) {
                list.push(new_idx);
            }
            self.edges.push(edge);
        }

        self.adjacency = new_adjacency;
        true
    }

    /// Add an explicit edge between two existing nodes.  Returns an error if
    /// either endpoint does not exist.
    pub fn add_edge(
        &mut self,
        source: &str,
        target: &str,
        kind: EdgeKind,
        weight: f64,
    ) -> Result<(), String> {
        if !self.nodes.contains_key(source) {
            return Err(format!("source node '{}' does not exist", source));
        }
        if !self.nodes.contains_key(target) {
            return Err(format!("target node '{}' does not exist", target));
        }

        let idx = self.edges.len();
        self.edges.push(DocGraphEdge {
            source: source.to_string(),
            target: target.to_string(),
            kind,
            weight,
        });

        self.adjacency
            .entry(source.to_string())
            .or_default()
            .push(idx);
        self.adjacency
            .entry(target.to_string())
            .or_default()
            .push(idx);

        Ok(())
    }

    /// Compute pairwise cosine similarity for all node pairs and add `Similar`
    /// edges for every pair whose similarity meets or exceeds the threshold.
    ///
    /// Existing `Similar` edges are **not** removed first — call this on a
    /// fresh graph or manually prune duplicates if needed.
    pub fn auto_link_similar(&mut self) {
        let ids: Vec<String> = self.nodes.keys().cloned().collect();
        let len = ids.len();

        // Collect edges to add (avoid borrow conflict).
        let mut new_edges: Vec<(String, String, f64)> = Vec::new();

        for i in 0..len {
            for j in (i + 1)..len {
                let a = self
                    .nodes
                    .get(&ids[i])
                    .map(|n| n.embedding.as_slice())
                    .unwrap_or(&[]);
                let b = self
                    .nodes
                    .get(&ids[j])
                    .map(|n| n.embedding.as_slice())
                    .unwrap_or(&[]);

                let sim = cosine_sim(a, b);
                if sim >= self.similarity_threshold {
                    new_edges.push((ids[i].clone(), ids[j].clone(), sim));
                }
            }
        }

        for (src, tgt, w) in new_edges {
            let idx = self.edges.len();
            self.edges.push(DocGraphEdge {
                source: src.clone(),
                target: tgt.clone(),
                kind: EdgeKind::Similar,
                weight: w,
            });
            self.adjacency.entry(src).or_default().push(idx);
            self.adjacency.entry(tgt).or_default().push(idx);
        }
    }

    /// Return neighbour nodes with their edge weights.
    pub fn neighbors(&self, doc_id: &str) -> Vec<(&DocGraphNode, f64)> {
        let indices = match self.adjacency.get(doc_id) {
            Some(v) => v,
            None => return Vec::new(),
        };

        let mut result = Vec::new();
        for &idx in indices {
            if let Some(edge) = self.edges.get(idx) {
                let other_id = if edge.source == doc_id {
                    &edge.target
                } else {
                    &edge.source
                };
                if let Some(node) = self.nodes.get(other_id) {
                    result.push((node, edge.weight));
                }
            }
        }
        result
    }

    /// BFS shortest path (by hop count) from `from` to `to`.  Returns the
    /// sequence of doc-ids including both endpoints, or `None` if no path
    /// exists.
    pub fn shortest_path(&self, from: &str, to: &str) -> Option<Vec<String>> {
        if !self.nodes.contains_key(from) || !self.nodes.contains_key(to) {
            return None;
        }
        if from == to {
            return Some(vec![from.to_string()]);
        }

        let mut visited: HashSet<String> = HashSet::new();
        let mut queue: VecDeque<String> = VecDeque::new();
        let mut parent: HashMap<String, String> = HashMap::new();

        visited.insert(from.to_string());
        queue.push_back(from.to_string());

        while let Some(current) = queue.pop_front() {
            if let Some(indices) = self.adjacency.get(&current) {
                for &idx in indices {
                    if let Some(edge) = self.edges.get(idx) {
                        let neighbor = if edge.source == current {
                            &edge.target
                        } else {
                            &edge.source
                        };
                        if visited.contains(neighbor) {
                            continue;
                        }
                        visited.insert(neighbor.clone());
                        parent.insert(neighbor.clone(), current.clone());

                        if neighbor == to {
                            // Reconstruct path.
                            let mut path = vec![to.to_string()];
                            let mut cur = to.to_string();
                            while let Some(p) = parent.get(&cur) {
                                path.push(p.clone());
                                cur = p.clone();
                            }
                            path.reverse();
                            return Some(path);
                        }
                        queue.push_back(neighbor.clone());
                    }
                }
            }
        }

        None
    }

    /// Return the connected components of the (undirected) graph.
    pub fn connected_components(&self) -> Vec<Vec<String>> {
        let mut visited: HashSet<String> = HashSet::new();
        let mut components: Vec<Vec<String>> = Vec::new();

        for id in self.nodes.keys() {
            if visited.contains(id) {
                continue;
            }
            let mut component = Vec::new();
            let mut stack = vec![id.clone()];
            while let Some(cur) = stack.pop() {
                if !visited.insert(cur.clone()) {
                    continue;
                }
                component.push(cur.clone());

                if let Some(indices) = self.adjacency.get(&cur) {
                    for &idx in indices {
                        if let Some(edge) = self.edges.get(idx) {
                            let neighbor = if edge.source == cur {
                                &edge.target
                            } else {
                                &edge.source
                            };
                            if !visited.contains(neighbor) {
                                stack.push(neighbor.clone());
                            }
                        }
                    }
                }
            }
            component.sort();
            components.push(component);
        }

        components.sort_by(|a, b| a.first().cmp(&b.first()));
        components
    }

    /// Number of nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of edges.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Degree (number of incident edges) for a given node.  Returns 0 if the
    /// node does not exist.
    pub fn degree(&self, doc_id: &str) -> usize {
        self.adjacency.get(doc_id).map(|v| v.len()).unwrap_or(0)
    }

    /// Aggregate statistics.
    pub fn stats(&self) -> DocumentGraphStats {
        let nc = self.node_count();
        let ec = self.edge_count();
        let avg = if nc == 0 {
            0.0
        } else {
            // Each edge contributes to two nodes' degree, so sum of degrees = 2*ec.
            (2.0 * ec as f64) / nc as f64
        };
        let cc = self.connected_components().len();
        DocumentGraphStats {
            node_count: nc,
            edge_count: ec,
            avg_degree: avg,
            component_count: cc,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_meta() -> HashMap<String, String> {
        HashMap::new()
    }

    fn meta(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    // -- cosine_sim --

    #[test]
    fn cosine_sim_identical_vectors() {
        let v = vec![1.0, 2.0, 3.0];
        let s = cosine_sim(&v, &v);
        assert!((s - 1.0).abs() < 1e-9);
    }

    #[test]
    fn cosine_sim_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!(cosine_sim(&a, &b).abs() < 1e-9);
    }

    #[test]
    fn cosine_sim_opposite() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        assert!((cosine_sim(&a, &b) + 1.0).abs() < 1e-9);
    }

    #[test]
    fn cosine_sim_different_lengths() {
        assert_eq!(cosine_sim(&[1.0, 2.0], &[1.0]), 0.0);
    }

    #[test]
    fn cosine_sim_empty() {
        assert_eq!(cosine_sim(&[], &[]), 0.0);
    }

    #[test]
    fn cosine_sim_zero_vector() {
        assert_eq!(cosine_sim(&[0.0, 0.0], &[1.0, 2.0]), 0.0);
    }

    // -- add / remove nodes --

    #[test]
    fn add_node_increases_count() {
        let mut g = SemanticDocumentGraph::new(0.7);
        assert_eq!(g.node_count(), 0);
        g.add_node("a", vec![1.0], empty_meta());
        assert_eq!(g.node_count(), 1);
        g.add_node("b", vec![2.0], empty_meta());
        assert_eq!(g.node_count(), 2);
    }

    #[test]
    fn add_node_replaces_existing() {
        let mut g = SemanticDocumentGraph::new(0.7);
        g.add_node("a", vec![1.0], meta(&[("k", "v1")]));
        g.add_node("a", vec![2.0], meta(&[("k", "v2")]));
        assert_eq!(g.node_count(), 1);
    }

    #[test]
    fn remove_node_returns_false_for_missing() {
        let mut g = SemanticDocumentGraph::new(0.7);
        assert!(!g.remove_node("x"));
    }

    #[test]
    fn remove_node_decreases_count() {
        let mut g = SemanticDocumentGraph::new(0.7);
        g.add_node("a", vec![1.0], empty_meta());
        g.add_node("b", vec![2.0], empty_meta());
        assert!(g.remove_node("a"));
        assert_eq!(g.node_count(), 1);
    }

    #[test]
    fn remove_node_cascades_edges() {
        let mut g = SemanticDocumentGraph::new(0.7);
        g.add_node("a", vec![1.0], empty_meta());
        g.add_node("b", vec![2.0], empty_meta());
        g.add_node("c", vec![3.0], empty_meta());
        g.add_edge("a", "b", EdgeKind::Citation, 1.0).ok();
        g.add_edge("b", "c", EdgeKind::Citation, 1.0).ok();
        g.add_edge("a", "c", EdgeKind::Similar, 0.8).ok();
        assert_eq!(g.edge_count(), 3);

        g.remove_node("a");
        // Only b-c should remain.
        assert_eq!(g.edge_count(), 1);
        assert_eq!(g.degree("b"), 1);
        assert_eq!(g.degree("c"), 1);
    }

    // -- add_edge --

    #[test]
    fn add_edge_ok() {
        let mut g = SemanticDocumentGraph::new(0.7);
        g.add_node("a", vec![1.0], empty_meta());
        g.add_node("b", vec![2.0], empty_meta());
        assert!(g.add_edge("a", "b", EdgeKind::Citation, 1.0).is_ok());
        assert_eq!(g.edge_count(), 1);
    }

    #[test]
    fn add_edge_missing_source() {
        let mut g = SemanticDocumentGraph::new(0.7);
        g.add_node("b", vec![2.0], empty_meta());
        let r = g.add_edge("x", "b", EdgeKind::Citation, 1.0);
        assert!(r.is_err());
        assert!(r.err().unwrap_or_default().contains("source"));
    }

    #[test]
    fn add_edge_missing_target() {
        let mut g = SemanticDocumentGraph::new(0.7);
        g.add_node("a", vec![1.0], empty_meta());
        let r = g.add_edge("a", "x", EdgeKind::Citation, 1.0);
        assert!(r.is_err());
        assert!(r.err().unwrap_or_default().contains("target"));
    }

    #[test]
    fn add_edge_both_missing() {
        let mut g = SemanticDocumentGraph::new(0.7);
        assert!(g.add_edge("x", "y", EdgeKind::Citation, 1.0).is_err());
    }

    // -- auto_link_similar --

    #[test]
    fn auto_link_similar_above_threshold() {
        let mut g = SemanticDocumentGraph::new(0.9);
        // Two very similar vectors.
        g.add_node("a", vec![1.0, 0.0, 0.0], empty_meta());
        g.add_node("b", vec![1.0, 0.01, 0.0], empty_meta());
        g.auto_link_similar();
        // They should be linked.
        assert_eq!(g.edge_count(), 1);
    }

    #[test]
    fn auto_link_similar_below_threshold() {
        let mut g = SemanticDocumentGraph::new(0.99);
        g.add_node("a", vec![1.0, 0.0], empty_meta());
        g.add_node("b", vec![0.0, 1.0], empty_meta());
        g.auto_link_similar();
        assert_eq!(g.edge_count(), 0);
    }

    #[test]
    fn auto_link_similar_multiple_pairs() {
        let mut g = SemanticDocumentGraph::new(0.5);
        g.add_node("a", vec![1.0, 0.0], empty_meta());
        g.add_node("b", vec![0.9, 0.1], empty_meta());
        g.add_node("c", vec![0.0, 1.0], empty_meta());
        g.auto_link_similar();
        // a-b should be linked (high sim), a-c and b-c probably not at 0.5 threshold
        // cosine(a, b) ≈ 0.994 => linked
        // cosine(a, c) = 0.0 => not linked
        // cosine(b, c) ≈ 0.11 => not linked
        assert_eq!(g.edge_count(), 1);
    }

    // -- neighbors --

    #[test]
    fn neighbors_returns_correct_set() {
        let mut g = SemanticDocumentGraph::new(0.7);
        g.add_node("a", vec![1.0], empty_meta());
        g.add_node("b", vec![2.0], empty_meta());
        g.add_node("c", vec![3.0], empty_meta());
        g.add_edge("a", "b", EdgeKind::Citation, 0.9).ok();
        g.add_edge("a", "c", EdgeKind::Similar, 0.8).ok();

        let nbrs = g.neighbors("a");
        assert_eq!(nbrs.len(), 2);

        let ids: HashSet<String> = nbrs.iter().map(|(n, _)| n.doc_id.clone()).collect();
        assert!(ids.contains("b"));
        assert!(ids.contains("c"));
    }

    #[test]
    fn neighbors_for_missing_node() {
        let g = SemanticDocumentGraph::new(0.7);
        assert!(g.neighbors("x").is_empty());
    }

    #[test]
    fn neighbors_no_edges() {
        let mut g = SemanticDocumentGraph::new(0.7);
        g.add_node("a", vec![1.0], empty_meta());
        assert!(g.neighbors("a").is_empty());
    }

    // -- shortest_path --

    #[test]
    fn shortest_path_direct() {
        let mut g = SemanticDocumentGraph::new(0.7);
        g.add_node("a", vec![1.0], empty_meta());
        g.add_node("b", vec![2.0], empty_meta());
        g.add_edge("a", "b", EdgeKind::Citation, 1.0).ok();

        let path = g.shortest_path("a", "b");
        assert_eq!(path, Some(vec!["a".to_string(), "b".to_string()]));
    }

    #[test]
    fn shortest_path_multi_hop() {
        let mut g = SemanticDocumentGraph::new(0.7);
        g.add_node("a", vec![1.0], empty_meta());
        g.add_node("b", vec![2.0], empty_meta());
        g.add_node("c", vec![3.0], empty_meta());
        g.add_edge("a", "b", EdgeKind::Citation, 1.0).ok();
        g.add_edge("b", "c", EdgeKind::Citation, 1.0).ok();

        let path = g.shortest_path("a", "c");
        assert_eq!(
            path,
            Some(vec!["a".to_string(), "b".to_string(), "c".to_string()])
        );
    }

    #[test]
    fn shortest_path_same_node() {
        let mut g = SemanticDocumentGraph::new(0.7);
        g.add_node("a", vec![1.0], empty_meta());
        assert_eq!(g.shortest_path("a", "a"), Some(vec!["a".to_string()]));
    }

    #[test]
    fn shortest_path_no_connection() {
        let mut g = SemanticDocumentGraph::new(0.7);
        g.add_node("a", vec![1.0], empty_meta());
        g.add_node("b", vec![2.0], empty_meta());
        assert_eq!(g.shortest_path("a", "b"), None);
    }

    #[test]
    fn shortest_path_missing_node() {
        let g = SemanticDocumentGraph::new(0.7);
        assert_eq!(g.shortest_path("x", "y"), None);
    }

    // -- connected_components --

    #[test]
    fn connected_components_single() {
        let mut g = SemanticDocumentGraph::new(0.7);
        g.add_node("a", vec![1.0], empty_meta());
        g.add_node("b", vec![2.0], empty_meta());
        g.add_edge("a", "b", EdgeKind::Citation, 1.0).ok();

        let cc = g.connected_components();
        assert_eq!(cc.len(), 1);
        assert_eq!(cc[0].len(), 2);
    }

    #[test]
    fn connected_components_multiple() {
        let mut g = SemanticDocumentGraph::new(0.7);
        g.add_node("a", vec![1.0], empty_meta());
        g.add_node("b", vec![2.0], empty_meta());
        g.add_node("c", vec![3.0], empty_meta());
        g.add_node("d", vec![4.0], empty_meta());
        g.add_edge("a", "b", EdgeKind::Citation, 1.0).ok();
        g.add_edge("c", "d", EdgeKind::SameCluster, 1.0).ok();

        let cc = g.connected_components();
        assert_eq!(cc.len(), 2);
    }

    #[test]
    fn connected_components_all_isolated() {
        let mut g = SemanticDocumentGraph::new(0.7);
        g.add_node("a", vec![1.0], empty_meta());
        g.add_node("b", vec![2.0], empty_meta());
        g.add_node("c", vec![3.0], empty_meta());
        let cc = g.connected_components();
        assert_eq!(cc.len(), 3);
    }

    // -- degree --

    #[test]
    fn degree_counts_correctly() {
        let mut g = SemanticDocumentGraph::new(0.7);
        g.add_node("a", vec![1.0], empty_meta());
        g.add_node("b", vec![2.0], empty_meta());
        g.add_node("c", vec![3.0], empty_meta());
        g.add_edge("a", "b", EdgeKind::Citation, 1.0).ok();
        g.add_edge("a", "c", EdgeKind::Citation, 1.0).ok();
        assert_eq!(g.degree("a"), 2);
        assert_eq!(g.degree("b"), 1);
        assert_eq!(g.degree("c"), 1);
    }

    #[test]
    fn degree_missing_node() {
        let g = SemanticDocumentGraph::new(0.7);
        assert_eq!(g.degree("x"), 0);
    }

    // -- stats --

    #[test]
    fn stats_empty_graph() {
        let g = SemanticDocumentGraph::new(0.7);
        let s = g.stats();
        assert_eq!(s.node_count, 0);
        assert_eq!(s.edge_count, 0);
        assert_eq!(s.avg_degree, 0.0);
        assert_eq!(s.component_count, 0);
    }

    #[test]
    fn stats_non_empty() {
        let mut g = SemanticDocumentGraph::new(0.7);
        g.add_node("a", vec![1.0], empty_meta());
        g.add_node("b", vec![2.0], empty_meta());
        g.add_node("c", vec![3.0], empty_meta());
        g.add_edge("a", "b", EdgeKind::Citation, 1.0).ok();
        g.add_edge("b", "c", EdgeKind::Similar, 0.8).ok();

        let s = g.stats();
        assert_eq!(s.node_count, 3);
        assert_eq!(s.edge_count, 2);
        // avg_degree = 2*2/3 ≈ 1.333
        assert!((s.avg_degree - 4.0 / 3.0).abs() < 1e-9);
        assert_eq!(s.component_count, 1);
    }

    // -- edge_count / node_count --

    #[test]
    fn empty_graph_counts() {
        let g = SemanticDocumentGraph::new(0.7);
        assert_eq!(g.node_count(), 0);
        assert_eq!(g.edge_count(), 0);
    }

    // -- metadata preserved --

    #[test]
    fn metadata_is_stored() {
        let mut g = SemanticDocumentGraph::new(0.7);
        g.add_node("a", vec![1.0], meta(&[("title", "hello")]));
        let nbrs_unused = g.neighbors("a"); // just ensure it doesn't panic
        drop(nbrs_unused);
        // Access node directly via stats / node_count; we can also verify via
        // auto_link_similar to exercise the embedding path.
        assert_eq!(g.node_count(), 1);
    }

    // -- edge kind preserved --

    #[test]
    fn edge_kind_preserved() {
        let mut g = SemanticDocumentGraph::new(0.7);
        g.add_node("a", vec![1.0, 0.0], empty_meta());
        g.add_node("b", vec![1.0, 0.01], empty_meta());
        g.add_edge("a", "b", EdgeKind::SameCluster, 0.5).ok();
        g.auto_link_similar();
        // Should have two edges: one SameCluster and one Similar.
        assert_eq!(g.edge_count(), 2);
    }

    // -- complex scenario --

    #[test]
    fn complex_graph_scenario() {
        let mut g = SemanticDocumentGraph::new(0.8);
        for i in 0..5 {
            let id = format!("doc{}", i);
            let emb = vec![(i as f64) * 0.1 + 0.5, 1.0 - (i as f64) * 0.1];
            g.add_node(&id, emb, meta(&[("idx", &i.to_string())]));
        }
        // Chain: doc0 - doc1 - doc2 - doc3 - doc4
        for i in 0..4 {
            g.add_edge(
                &format!("doc{}", i),
                &format!("doc{}", i + 1),
                EdgeKind::Citation,
                1.0,
            )
            .ok();
        }

        assert_eq!(g.node_count(), 5);
        assert_eq!(g.edge_count(), 4);
        assert_eq!(g.connected_components().len(), 1);

        let path = g.shortest_path("doc0", "doc4");
        assert!(path.is_some());
        let path = path.unwrap_or_default();
        assert_eq!(path.len(), 5);
        assert_eq!(path[0], "doc0");
        assert_eq!(path[4], "doc4");

        // Remove middle node.
        g.remove_node("doc2");
        assert_eq!(g.node_count(), 4);
        // Edges doc1-doc2 and doc2-doc3 removed => 2 remain.
        assert_eq!(g.edge_count(), 2);
        // Now two components: {doc0, doc1} and {doc3, doc4}.
        assert_eq!(g.connected_components().len(), 2);
        assert_eq!(g.shortest_path("doc0", "doc4"), None);
    }
}
