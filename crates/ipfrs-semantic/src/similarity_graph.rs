//! # Semantic Similarity Graph
//!
//! A graph where nodes are semantic embeddings and edges represent similarity above a threshold.
//! Supports community detection, shortest-path traversal, subgraph extraction, and graph statistics.

use std::collections::{HashMap, HashSet, VecDeque};

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// A node in the semantic similarity graph.
#[derive(Debug, Clone)]
pub struct SgNode {
    /// Unique identifier for this node.
    pub id: String,
    /// Embedding vector.
    pub embedding: Vec<f64>,
    /// Optional human-readable label.
    pub label: Option<String>,
    /// Arbitrary key-value metadata.
    pub metadata: HashMap<String, String>,
}

impl SgNode {
    /// Create a new node with the given id and embedding.
    pub fn new(id: impl Into<String>, embedding: Vec<f64>) -> Self {
        Self {
            id: id.into(),
            embedding,
            label: None,
            metadata: HashMap::new(),
        }
    }

    /// Builder: set label.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Builder: insert a metadata entry.
    pub fn with_meta(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// An undirected edge between two nodes, keyed by `min_id:max_id`.
#[derive(Debug, Clone)]
pub struct SgEdge {
    /// ID of the first endpoint (lexicographically smaller).
    pub from_id: String,
    /// ID of the second endpoint (lexicographically larger).
    pub to_id: String,
    /// Cosine similarity in [−1, 1].
    pub similarity: f64,
    /// Unix-epoch milliseconds at creation time.
    pub created_at: u64,
}

impl SgEdge {
    /// Canonical edge key: `min_id:max_id`.
    pub fn key(a: &str, b: &str) -> String {
        if a <= b {
            format!("{}:{}", a, b)
        } else {
            format!("{}:{}", b, a)
        }
    }
}

/// A community (cluster) of nodes.
#[derive(Debug, Clone)]
pub struct SgCommunity {
    /// Unique index of this community.
    pub id: usize,
    /// Node IDs that belong to this community.
    pub members: Vec<String>,
    /// Mean embedding of all member nodes (centroid).
    pub centroid: Vec<f64>,
    /// Mean pairwise cosine similarity of members (1.0 if single member).
    pub cohesion: f64,
}

/// Configuration for `SemanticSimilarityGraph`.
#[derive(Debug, Clone)]
pub struct GraphConfig {
    /// Minimum cosine similarity for an edge to be created.
    pub similarity_threshold: f64,
    /// Maximum number of neighbors per node.
    pub max_edges_per_node: usize,
    /// If `true`, prune lowest-similarity edge when a node exceeds `max_edges_per_node`.
    pub auto_prune: bool,
}

impl Default for GraphConfig {
    fn default() -> Self {
        Self {
            similarity_threshold: 0.7,
            max_edges_per_node: 50,
            auto_prune: true,
        }
    }
}

/// Graph-level statistics snapshot.
#[derive(Debug, Clone)]
pub struct SgStats {
    /// Total number of nodes.
    pub node_count: usize,
    /// Total number of edges.
    pub edge_count: usize,
    /// Graph density: `edges / (n*(n-1)/2)`.
    pub density: f64,
    /// Mean degree.
    pub avg_degree: f64,
    /// Mean edge similarity across all edges.
    pub avg_similarity: f64,
    /// Number of nodes with no neighbors.
    pub isolated_nodes: usize,
}

// ---------------------------------------------------------------------------
// Main graph structure
// ---------------------------------------------------------------------------

/// A graph of semantic embeddings connected by cosine similarity.
#[derive(Debug, Clone)]
pub struct SemanticSimilarityGraph {
    /// Graph configuration.
    pub config: GraphConfig,
    /// All nodes, keyed by node ID.
    pub nodes: HashMap<String, SgNode>,
    /// All edges, keyed by `min_id:max_id`.
    pub edges: HashMap<String, SgEdge>,
    /// Adjacency list: node ID → list of neighbour IDs.
    pub adjacency: HashMap<String, Vec<String>>,
}

impl SemanticSimilarityGraph {
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    /// Create an empty graph with the given configuration.
    pub fn new(config: GraphConfig) -> Self {
        Self {
            config,
            nodes: HashMap::new(),
            edges: HashMap::new(),
            adjacency: HashMap::new(),
        }
    }

    /// Create a graph with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(GraphConfig::default())
    }

    // -----------------------------------------------------------------------
    // Cosine similarity
    // -----------------------------------------------------------------------

    /// Compute cosine similarity between two embedding vectors.
    ///
    /// Returns `0.0` if either vector is all-zero, empty, or dimensions differ.
    pub fn similarity(a: &[f64], b: &[f64]) -> f64 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }
        let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }
        (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
    }

    // -----------------------------------------------------------------------
    // Mutation
    // -----------------------------------------------------------------------

    /// Add a node to the graph.
    ///
    /// Computes similarity with every existing node and adds edges where the
    /// similarity is ≥ `config.similarity_threshold`.  When `auto_prune` is
    /// enabled the lowest-similarity edge of each over-full neighbour list is
    /// dropped.
    pub fn add_node(&mut self, node: SgNode) {
        let node_id = node.id.clone();

        // Collect similarity scores against all existing nodes first, before
        // we mutate `self.nodes`.
        let existing: Vec<(String, f64)> = self
            .nodes
            .iter()
            .map(|(id, existing_node)| {
                let sim = Self::similarity(&node.embedding, &existing_node.embedding);
                (id.clone(), sim)
            })
            .collect();

        // Insert the new node and initialise its adjacency list.
        self.nodes.insert(node_id.clone(), node);
        self.adjacency.entry(node_id.clone()).or_default();

        // Create edges where similarity meets the threshold.
        let threshold = self.config.similarity_threshold;
        let now = Self::unix_millis();

        for (other_id, sim) in existing {
            if sim < threshold {
                continue;
            }
            self.insert_edge_raw(&node_id, &other_id, sim, now);
        }
    }

    /// Remove a node and all edges connected to it.
    ///
    /// Returns `true` if the node existed.
    pub fn remove_node(&mut self, node_id: &str) -> bool {
        if self.nodes.remove(node_id).is_none() {
            return false;
        }

        // Collect neighbour list before mutating adjacency.
        let neighbours: Vec<String> = self.adjacency.remove(node_id).unwrap_or_default();

        for neighbour_id in &neighbours {
            // Remove the edge record.
            let key = SgEdge::key(node_id, neighbour_id);
            self.edges.remove(&key);

            // Remove from the neighbour's adjacency list.
            if let Some(adj) = self.adjacency.get_mut(neighbour_id.as_str()) {
                adj.retain(|id| id != node_id);
            }
        }

        true
    }

    // -----------------------------------------------------------------------
    // Queries
    // -----------------------------------------------------------------------

    /// Look up a node by ID.
    pub fn get_node(&self, node_id: &str) -> Option<&SgNode> {
        self.nodes.get(node_id)
    }

    /// Return all neighbours of `node_id`, sorted by similarity descending.
    pub fn neighbors(&self, node_id: &str) -> Vec<(&SgNode, f64)> {
        let adj = match self.adjacency.get(node_id) {
            Some(v) => v,
            None => return Vec::new(),
        };

        let mut result: Vec<(&SgNode, f64)> = adj
            .iter()
            .filter_map(|other_id| {
                let key = SgEdge::key(node_id, other_id);
                let edge = self.edges.get(&key)?;
                let node = self.nodes.get(other_id.as_str())?;
                Some((node, edge.similarity))
            })
            .collect();

        result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        result
    }

    /// Return the top-`n` most similar neighbours of `node_id`.
    pub fn most_similar(&self, node_id: &str, n: usize) -> Vec<(&SgNode, f64)> {
        let all = self.neighbors(node_id);
        all.into_iter().take(n).collect()
    }

    // -----------------------------------------------------------------------
    // Community detection
    // -----------------------------------------------------------------------

    /// Detect communities using BFS connected components, filtered by
    /// `min_community_size`.  Centroid and cohesion are computed for each
    /// returned community.
    pub fn find_communities(&self, min_community_size: usize) -> Vec<SgCommunity> {
        let mut visited: HashSet<&str> = HashSet::new();
        let mut communities: Vec<SgCommunity> = Vec::new();
        let mut community_id = 0usize;

        for start_id in self.nodes.keys() {
            if visited.contains(start_id.as_str()) {
                continue;
            }

            // BFS
            let mut queue: VecDeque<&str> = VecDeque::new();
            let mut component: Vec<String> = Vec::new();

            queue.push_back(start_id.as_str());
            visited.insert(start_id.as_str());

            while let Some(current) = queue.pop_front() {
                component.push(current.to_owned());

                if let Some(adj) = self.adjacency.get(current) {
                    for neighbour in adj {
                        if !visited.contains(neighbour.as_str()) {
                            visited.insert(neighbour.as_str());
                            queue.push_back(neighbour.as_str());
                        }
                    }
                }
            }

            if component.len() < min_community_size {
                continue;
            }

            let centroid = self.compute_centroid(&component);
            let cohesion = self.compute_cohesion(&component);

            communities.push(SgCommunity {
                id: community_id,
                members: component,
                centroid,
                cohesion,
            });
            community_id += 1;
        }

        communities
    }

    /// Return the community ID that contains `node_id`, if any.
    pub fn community_of(node_id: &str, communities: &[SgCommunity]) -> Option<usize> {
        for community in communities {
            if community.members.iter().any(|m| m == node_id) {
                return Some(community.id);
            }
        }
        None
    }

    // -----------------------------------------------------------------------
    // Path finding
    // -----------------------------------------------------------------------

    /// BFS shortest path from `from` to `to`.
    ///
    /// Returns `None` if either node does not exist or no path exists.
    pub fn path_between(&self, from: &str, to: &str) -> Option<Vec<String>> {
        if !self.nodes.contains_key(from) || !self.nodes.contains_key(to) {
            return None;
        }
        if from == to {
            return Some(vec![from.to_owned()]);
        }

        let mut visited: HashSet<&str> = HashSet::new();
        let mut queue: VecDeque<Vec<&str>> = VecDeque::new();

        visited.insert(from);
        queue.push_back(vec![from]);

        while let Some(path) = queue.pop_front() {
            let current = *path.last()?;

            if let Some(adj) = self.adjacency.get(current) {
                for neighbour in adj {
                    if neighbour == to {
                        let mut result: Vec<String> = path.iter().map(|s| s.to_string()).collect();
                        result.push(to.to_owned());
                        return Some(result);
                    }
                    if !visited.contains(neighbour.as_str()) {
                        visited.insert(neighbour.as_str());
                        let mut new_path = path.clone();
                        new_path.push(neighbour.as_str());
                        queue.push_back(new_path);
                    }
                }
            }
        }

        None
    }

    // -----------------------------------------------------------------------
    // Subgraph extraction
    // -----------------------------------------------------------------------

    /// Induce a subgraph from the given node IDs, including only edges whose
    /// both endpoints are in the set.
    pub fn subgraph(&self, node_ids: &[&str]) -> SemanticSimilarityGraph {
        let id_set: HashSet<&str> = node_ids.iter().copied().collect();

        let mut sub = SemanticSimilarityGraph::new(self.config.clone());

        for &nid in &id_set {
            if let Some(node) = self.nodes.get(nid) {
                sub.nodes.insert(nid.to_owned(), node.clone());
                sub.adjacency.entry(nid.to_owned()).or_default();
            }
        }

        for (key, edge) in &self.edges {
            if id_set.contains(edge.from_id.as_str()) && id_set.contains(edge.to_id.as_str()) {
                sub.edges.insert(key.clone(), edge.clone());
                sub.adjacency
                    .entry(edge.from_id.clone())
                    .or_default()
                    .push(edge.to_id.clone());
                sub.adjacency
                    .entry(edge.to_id.clone())
                    .or_default()
                    .push(edge.from_id.clone());
            }
        }

        sub
    }

    // -----------------------------------------------------------------------
    // Graph metrics
    // -----------------------------------------------------------------------

    /// Graph density: `|E| / (n*(n-1)/2)`.  Returns `0.0` for fewer than 2
    /// nodes.
    pub fn density(&self) -> f64 {
        let n = self.nodes.len();
        if n < 2 {
            return 0.0;
        }
        let max_edges = n * (n - 1) / 2;
        self.edges.len() as f64 / max_edges as f64
    }

    /// Mean number of neighbours per node.  Returns `0.0` if empty.
    pub fn avg_degree(&self) -> f64 {
        if self.nodes.is_empty() {
            return 0.0;
        }
        let total: usize = self.adjacency.values().map(|v| v.len()).sum();
        total as f64 / self.nodes.len() as f64
    }

    /// Number of nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of edges.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Compute aggregate graph statistics.
    pub fn stats(&self) -> SgStats {
        let node_count = self.node_count();
        let edge_count = self.edge_count();
        let density = self.density();
        let avg_degree = self.avg_degree();

        let avg_similarity = if edge_count == 0 {
            0.0
        } else {
            let total: f64 = self.edges.values().map(|e| e.similarity).sum();
            total / edge_count as f64
        };

        let isolated_nodes = self.adjacency.values().filter(|v| v.is_empty()).count();

        SgStats {
            node_count,
            edge_count,
            density,
            avg_degree,
            avg_similarity,
            isolated_nodes,
        }
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Insert a single undirected edge, respecting `auto_prune` / `max_edges_per_node`.
    fn insert_edge_raw(&mut self, a: &str, b: &str, sim: f64, now: u64) {
        let key = SgEdge::key(a, b);

        // Avoid duplicate edges.
        if self.edges.contains_key(&key) {
            return;
        }

        let (from_id, to_id) = if a <= b {
            (a.to_owned(), b.to_owned())
        } else {
            (b.to_owned(), a.to_owned())
        };

        let edge = SgEdge {
            from_id: from_id.clone(),
            to_id: to_id.clone(),
            similarity: sim,
            created_at: now,
        };

        self.edges.insert(key, edge);

        self.adjacency
            .entry(a.to_owned())
            .or_default()
            .push(b.to_owned());
        self.adjacency
            .entry(b.to_owned())
            .or_default()
            .push(a.to_owned());

        // Auto-prune for node `a`.
        self.maybe_prune(a);
        // Auto-prune for node `b`.
        self.maybe_prune(b);
    }

    /// If `node_id` exceeds `max_edges_per_node`, remove the edge with the
    /// lowest similarity.
    fn maybe_prune(&mut self, node_id: &str) {
        if !self.config.auto_prune {
            return;
        }
        let max = self.config.max_edges_per_node;
        let adj_len = self.adjacency.get(node_id).map(|v| v.len()).unwrap_or(0);
        if adj_len <= max {
            return;
        }

        // Find the weakest neighbour.
        let weakest: Option<(String, f64)> = self.adjacency.get(node_id).and_then(|adj| {
            adj.iter()
                .filter_map(|nid| {
                    let key = SgEdge::key(node_id, nid);
                    let sim = self.edges.get(&key).map(|e| e.similarity)?;
                    Some((nid.clone(), sim))
                })
                .min_by(|x, y| x.1.partial_cmp(&y.1).unwrap_or(std::cmp::Ordering::Equal))
        });

        if let Some((weak_id, _)) = weakest {
            let key = SgEdge::key(node_id, &weak_id);
            self.edges.remove(&key);

            if let Some(adj) = self.adjacency.get_mut(node_id) {
                adj.retain(|id| id != &weak_id);
            }
            if let Some(adj) = self.adjacency.get_mut(weak_id.as_str()) {
                adj.retain(|id| id != node_id);
            }
        }
    }

    /// Compute the centroid (mean embedding) for a set of node IDs.
    fn compute_centroid(&self, members: &[String]) -> Vec<f64> {
        if members.is_empty() {
            return Vec::new();
        }

        // Determine dimension from first member that exists.
        let dim = members
            .iter()
            .find_map(|id| self.nodes.get(id.as_str()).map(|n| n.embedding.len()))
            .unwrap_or(0);

        if dim == 0 {
            return Vec::new();
        }

        let mut centroid = vec![0.0f64; dim];
        let mut count = 0usize;

        for id in members {
            if let Some(node) = self.nodes.get(id.as_str()) {
                if node.embedding.len() == dim {
                    for (c, v) in centroid.iter_mut().zip(node.embedding.iter()) {
                        *c += v;
                    }
                    count += 1;
                }
            }
        }

        if count > 0 {
            for c in &mut centroid {
                *c /= count as f64;
            }
        }

        centroid
    }

    /// Compute mean pairwise cosine similarity for a set of node IDs.
    /// Returns `1.0` if fewer than 2 members.
    fn compute_cohesion(&self, members: &[String]) -> f64 {
        if members.len() < 2 {
            return 1.0;
        }

        let embeddings: Vec<&[f64]> = members
            .iter()
            .filter_map(|id| self.nodes.get(id.as_str()).map(|n| n.embedding.as_slice()))
            .collect();

        let n = embeddings.len();
        if n < 2 {
            return 1.0;
        }

        let mut sum = 0.0f64;
        let mut pairs = 0usize;

        for i in 0..n {
            for j in (i + 1)..n {
                sum += Self::similarity(embeddings[i], embeddings[j]);
                pairs += 1;
            }
        }

        if pairs == 0 {
            1.0
        } else {
            sum / pairs as f64
        }
    }

    /// Current time as Unix-epoch milliseconds.
    fn unix_millis() -> u64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::similarity_graph::{
        GraphConfig, SemanticSimilarityGraph, SgCommunity, SgEdge, SgNode,
    };

    // ---- helpers -----------------------------------------------------------

    fn unit(v: &[f64]) -> Vec<f64> {
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm == 0.0 {
            return v.to_vec();
        }
        v.iter().map(|x| x / norm).collect()
    }

    fn node(id: &str, emb: Vec<f64>) -> SgNode {
        SgNode::new(id, emb)
    }

    fn default_graph() -> SemanticSimilarityGraph {
        SemanticSimilarityGraph::with_defaults()
    }

    fn graph_with_threshold(t: f64) -> SemanticSimilarityGraph {
        SemanticSimilarityGraph::new(GraphConfig {
            similarity_threshold: t,
            max_edges_per_node: 50,
            auto_prune: true,
        })
    }

    // ---- similarity --------------------------------------------------------

    #[test]
    fn test_similarity_identical_vectors() {
        let v = vec![1.0, 0.0, 0.0];
        let s = SemanticSimilarityGraph::similarity(&v, &v);
        assert!((s - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_similarity_orthogonal_vectors() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let s = SemanticSimilarityGraph::similarity(&a, &b);
        assert!(s.abs() < 1e-10);
    }

    #[test]
    fn test_similarity_opposite_vectors() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![-1.0, 0.0, 0.0];
        let s = SemanticSimilarityGraph::similarity(&a, &b);
        assert!((s + 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_similarity_zero_vector_returns_zero() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 2.0, 3.0];
        assert_eq!(SemanticSimilarityGraph::similarity(&a, &b), 0.0);
    }

    #[test]
    fn test_similarity_mismatched_dims_returns_zero() {
        let a = vec![1.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert_eq!(SemanticSimilarityGraph::similarity(&a, &b), 0.0);
    }

    #[test]
    fn test_similarity_empty_returns_zero() {
        assert_eq!(SemanticSimilarityGraph::similarity(&[], &[]), 0.0);
    }

    #[test]
    fn test_similarity_known_value() {
        // 45 degree angle → cos(45°) ≈ 0.7071
        let a = unit(&[1.0, 1.0]);
        let b = unit(&[1.0, 0.0]);
        let s = SemanticSimilarityGraph::similarity(&a, &b);
        assert!((s - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-9);
    }

    // ---- node CRUD ---------------------------------------------------------

    #[test]
    fn test_add_single_node() {
        let mut g = default_graph();
        g.add_node(node("a", vec![1.0, 0.0]));
        assert_eq!(g.node_count(), 1);
        assert_eq!(g.edge_count(), 0);
    }

    #[test]
    fn test_get_node_exists() {
        let mut g = default_graph();
        g.add_node(node("a", vec![1.0, 0.0]));
        assert!(g.get_node("a").is_some());
    }

    #[test]
    fn test_get_node_missing() {
        let g = default_graph();
        assert!(g.get_node("x").is_none());
    }

    #[test]
    fn test_remove_existing_node_returns_true() {
        let mut g = default_graph();
        g.add_node(node("a", vec![1.0, 0.0]));
        assert!(g.remove_node("a"));
        assert_eq!(g.node_count(), 0);
    }

    #[test]
    fn test_remove_missing_node_returns_false() {
        let mut g = default_graph();
        assert!(!g.remove_node("ghost"));
    }

    #[test]
    fn test_remove_node_cleans_edges() {
        let mut g = graph_with_threshold(0.0); // threshold=0 → always add edges
        g.add_node(node("a", vec![1.0, 0.0]));
        g.add_node(node("b", vec![0.0, 1.0]));
        assert_eq!(g.edge_count(), 1);
        g.remove_node("a");
        assert_eq!(g.edge_count(), 0);
        assert!(g.adjacency.get("b").map(|v| v.is_empty()).unwrap_or(true));
    }

    // ---- edge creation -----------------------------------------------------

    #[test]
    fn test_edge_created_above_threshold() {
        let mut g = graph_with_threshold(0.5);
        // highly similar embeddings
        g.add_node(node("a", unit(&[1.0, 0.1])));
        g.add_node(node("b", unit(&[1.0, 0.2])));
        assert_eq!(g.edge_count(), 1);
    }

    #[test]
    fn test_no_edge_below_threshold() {
        let mut g = graph_with_threshold(0.99);
        g.add_node(node("a", vec![1.0, 0.0]));
        g.add_node(node("b", vec![0.0, 1.0])); // orthogonal → sim=0
        assert_eq!(g.edge_count(), 0);
    }

    #[test]
    fn test_edge_key_canonical_order() {
        assert_eq!(SgEdge::key("b", "a"), "a:b");
        assert_eq!(SgEdge::key("a", "b"), "a:b");
        assert_eq!(SgEdge::key("z", "a"), "a:z");
    }

    // ---- neighbors / most_similar ------------------------------------------

    #[test]
    fn test_neighbors_sorted_by_similarity_desc() {
        let mut g = graph_with_threshold(0.0);
        g.add_node(node("origin", unit(&[1.0, 0.0, 0.0])));
        g.add_node(node("close", unit(&[1.0, 0.1, 0.0])));
        g.add_node(node("far", unit(&[1.0, 1.0, 0.0])));

        let nbrs = g.neighbors("origin");
        assert_eq!(nbrs.len(), 2);
        assert!(nbrs[0].1 >= nbrs[1].1);
    }

    #[test]
    fn test_neighbors_unknown_node_empty() {
        let g = default_graph();
        assert!(g.neighbors("ghost").is_empty());
    }

    #[test]
    fn test_most_similar_respects_n() {
        let mut g = graph_with_threshold(0.0);
        for i in 0..10 {
            let v = unit(&[i as f64 + 1.0, 1.0]);
            g.add_node(node(&format!("n{}", i), v));
        }
        let top3 = g.most_similar("n0", 3);
        assert_eq!(top3.len(), 3);
    }

    #[test]
    fn test_most_similar_fewer_than_n() {
        let mut g = graph_with_threshold(0.0);
        g.add_node(node("a", unit(&[1.0, 0.0])));
        g.add_node(node("b", unit(&[1.0, 0.1])));
        let top10 = g.most_similar("a", 10);
        assert_eq!(top10.len(), 1);
    }

    // ---- auto-prune --------------------------------------------------------

    #[test]
    fn test_auto_prune_keeps_max_edges() {
        let config = GraphConfig {
            similarity_threshold: 0.0,
            max_edges_per_node: 3,
            auto_prune: true,
        };
        let mut g = SemanticSimilarityGraph::new(config);
        // Add 5 nodes; the center node should never exceed 3 edges.
        let center = unit(&[1.0, 0.0, 0.0]);
        g.add_node(node("center", center));
        for i in 0..5 {
            let v = unit(&[1.0, (i as f64 + 1.0) * 0.01, 0.0]);
            g.add_node(node(&format!("n{}", i), v));
        }
        let degree = g.adjacency.get("center").map(|v| v.len()).unwrap_or(0);
        assert!(degree <= 3, "degree={}", degree);
    }

    #[test]
    fn test_auto_prune_disabled_allows_many_edges() {
        let config = GraphConfig {
            similarity_threshold: 0.0,
            max_edges_per_node: 3,
            auto_prune: false,
        };
        let mut g = SemanticSimilarityGraph::new(config);
        g.add_node(node("center", unit(&[1.0, 0.0, 0.0])));
        for i in 0..5 {
            let v = unit(&[1.0, (i as f64 + 1.0) * 0.01, 0.0]);
            g.add_node(node(&format!("n{}", i), v));
        }
        let degree = g.adjacency.get("center").map(|v| v.len()).unwrap_or(0);
        assert_eq!(degree, 5);
    }

    // ---- path_between ------------------------------------------------------

    #[test]
    fn test_path_between_direct_neighbors() {
        let mut g = graph_with_threshold(0.0);
        g.add_node(node("a", unit(&[1.0, 0.0])));
        g.add_node(node("b", unit(&[1.0, 0.1])));
        let path = g.path_between("a", "b");
        assert_eq!(path, Some(vec!["a".to_owned(), "b".to_owned()]));
    }

    #[test]
    fn test_path_between_same_node() {
        let mut g = default_graph();
        g.add_node(node("a", vec![1.0]));
        let path = g.path_between("a", "a");
        assert_eq!(path, Some(vec!["a".to_owned()]));
    }

    #[test]
    fn test_path_between_no_path() {
        let mut g = graph_with_threshold(0.99);
        g.add_node(node("a", vec![1.0, 0.0]));
        g.add_node(node("b", vec![0.0, 1.0]));
        assert!(g.path_between("a", "b").is_none());
    }

    #[test]
    fn test_path_between_missing_node() {
        let g = default_graph();
        assert!(g.path_between("x", "y").is_none());
    }

    #[test]
    fn test_path_between_multi_hop() {
        let mut g = graph_with_threshold(0.0);
        g.add_node(node("a", unit(&[1.0, 0.0, 0.0])));
        g.add_node(node("b", unit(&[0.0, 1.0, 0.0])));
        g.add_node(node("c", unit(&[0.0, 0.0, 1.0])));
        // Connect a-b and b-c manually by using threshold=0 so all nodes
        // with non-zero similarity connect. Since a,b,c are orthogonal they
        // won't connect. Force connections by using similar vectors.
        let mut g2 = graph_with_threshold(0.0);
        g2.add_node(node("a", unit(&[1.0, 0.0, 0.0])));
        // b is close to a
        g2.add_node(node("b", unit(&[1.0, 0.5, 0.0])));
        // c is close to b but not necessarily a
        g2.add_node(node("c", unit(&[0.0, 1.0, 0.5])));
        // d is close to c
        g2.add_node(node("d", unit(&[0.0, 0.5, 1.0])));

        let path = g2.path_between("a", "d");
        assert!(path.is_some());
        let p = path.expect("test: path_between returned None");
        assert_eq!(p.first().map(|s| s.as_str()), Some("a"));
        assert_eq!(p.last().map(|s| s.as_str()), Some("d"));
    }

    // ---- subgraph ----------------------------------------------------------

    #[test]
    fn test_subgraph_contains_only_selected_nodes() {
        let mut g = graph_with_threshold(0.0);
        g.add_node(node("a", unit(&[1.0, 0.0])));
        g.add_node(node("b", unit(&[1.0, 0.1])));
        g.add_node(node("c", unit(&[0.0, 1.0])));
        let sub = g.subgraph(&["a", "b"]);
        assert_eq!(sub.node_count(), 2);
        assert!(sub.get_node("c").is_none());
    }

    #[test]
    fn test_subgraph_preserves_inter_edges() {
        let mut g = graph_with_threshold(0.0);
        g.add_node(node("a", unit(&[1.0, 0.0])));
        g.add_node(node("b", unit(&[1.0, 0.1])));
        g.add_node(node("c", unit(&[0.0, 1.0])));
        let sub = g.subgraph(&["a", "b"]);
        assert_eq!(sub.edge_count(), g.subgraph(&["a", "b"]).edge_count());
    }

    #[test]
    fn test_subgraph_excludes_cross_edges() {
        let mut g = graph_with_threshold(0.0);
        g.add_node(node("a", unit(&[1.0, 0.0])));
        g.add_node(node("b", unit(&[1.0, 0.1])));
        g.add_node(node("c", unit(&[1.0, 0.2])));
        // All three connect at threshold 0.
        let sub = g.subgraph(&["a", "b"]); // exclude c
                                           // The a-c edge should not appear in subgraph.
        assert!(!sub.edges.contains_key(&SgEdge::key("a", "c")));
    }

    // ---- density / avg_degree / stats --------------------------------------

    #[test]
    fn test_density_empty_graph() {
        assert_eq!(default_graph().density(), 0.0);
    }

    #[test]
    fn test_density_single_node() {
        let mut g = default_graph();
        g.add_node(node("a", vec![1.0]));
        assert_eq!(g.density(), 0.0);
    }

    #[test]
    fn test_density_complete_graph() {
        let mut g = graph_with_threshold(0.0);
        g.add_node(node("a", unit(&[1.0, 0.0])));
        g.add_node(node("b", unit(&[1.0, 0.1])));
        g.add_node(node("c", unit(&[1.0, 0.2])));
        // 3 edges out of max 3 → density = 1.0
        assert!((g.density() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_avg_degree_empty() {
        assert_eq!(default_graph().avg_degree(), 0.0);
    }

    #[test]
    fn test_avg_degree_single() {
        let mut g = default_graph();
        g.add_node(node("a", vec![1.0]));
        assert_eq!(g.avg_degree(), 0.0);
    }

    #[test]
    fn test_stats_fields() {
        let mut g = graph_with_threshold(0.0);
        g.add_node(node("a", unit(&[1.0, 0.0])));
        g.add_node(node("b", unit(&[1.0, 0.1])));
        let s = g.stats();
        assert_eq!(s.node_count, 2);
        assert_eq!(s.edge_count, 1);
        assert!(s.avg_similarity > 0.0);
        assert_eq!(s.isolated_nodes, 0);
    }

    // ---- community detection -----------------------------------------------

    #[test]
    fn test_find_communities_single_component() {
        let mut g = graph_with_threshold(0.0);
        g.add_node(node("a", unit(&[1.0, 0.0])));
        g.add_node(node("b", unit(&[1.0, 0.1])));
        g.add_node(node("c", unit(&[1.0, 0.2])));
        let communities = g.find_communities(1);
        assert_eq!(communities.len(), 1);
        assert_eq!(communities[0].members.len(), 3);
    }

    #[test]
    fn test_find_communities_two_isolated() {
        let mut g = graph_with_threshold(0.99);
        g.add_node(node("a", vec![1.0, 0.0]));
        g.add_node(node("b", vec![0.0, 1.0]));
        // No edges → two singletons.
        let communities = g.find_communities(1);
        assert_eq!(communities.len(), 2);
    }

    #[test]
    fn test_find_communities_min_size_filter() {
        let mut g = graph_with_threshold(0.99);
        g.add_node(node("a", vec![1.0, 0.0]));
        g.add_node(node("b", vec![0.0, 1.0]));
        // Both communities are singletons; min_size=2 → empty result.
        let communities = g.find_communities(2);
        assert!(communities.is_empty());
    }

    #[test]
    fn test_community_cohesion_single_member() {
        let mut g = default_graph();
        g.add_node(node("a", vec![1.0, 0.0]));
        let communities = g.find_communities(1);
        assert_eq!(communities[0].cohesion, 1.0);
    }

    #[test]
    fn test_community_of_returns_correct_id() {
        let mut g = graph_with_threshold(0.0);
        g.add_node(node("a", unit(&[1.0, 0.0])));
        g.add_node(node("b", unit(&[1.0, 0.1])));
        let communities = g.find_communities(1);
        let cid = SemanticSimilarityGraph::community_of("a", &communities);
        assert!(cid.is_some());
    }

    #[test]
    fn test_community_of_missing_node_returns_none() {
        let communities: Vec<SgCommunity> = vec![];
        assert!(SemanticSimilarityGraph::community_of("ghost", &communities).is_none());
    }

    // ---- SgNode builder ----------------------------------------------------

    #[test]
    fn test_node_builder_label() {
        let n = SgNode::new("x", vec![1.0]).with_label("hello");
        assert_eq!(n.label.as_deref(), Some("hello"));
    }

    #[test]
    fn test_node_builder_metadata() {
        let n = SgNode::new("x", vec![1.0]).with_meta("k", "v");
        assert_eq!(n.metadata.get("k").map(|s| s.as_str()), Some("v"));
    }

    #[test]
    fn test_node_default_no_label() {
        let n = SgNode::new("x", vec![1.0]);
        assert!(n.label.is_none());
    }

    #[test]
    fn test_node_default_empty_metadata() {
        let n = SgNode::new("x", vec![1.0]);
        assert!(n.metadata.is_empty());
    }

    // ---- GraphConfig default -----------------------------------------------

    #[test]
    fn test_graph_config_defaults() {
        let cfg = GraphConfig::default();
        assert_eq!(cfg.similarity_threshold, 0.7);
        assert_eq!(cfg.max_edges_per_node, 50);
        assert!(cfg.auto_prune);
    }

    // ---- edge key ordering -------------------------------------------------

    #[test]
    fn test_edge_key_same_regardless_of_order() {
        assert_eq!(SgEdge::key("foo", "bar"), SgEdge::key("bar", "foo"));
    }

    // ---- community centroid ------------------------------------------------

    #[test]
    fn test_community_centroid_not_empty() {
        let mut g = graph_with_threshold(0.0);
        g.add_node(node("a", unit(&[1.0, 0.0])));
        g.add_node(node("b", unit(&[0.5, 0.5])));
        let communities = g.find_communities(1);
        assert!(!communities[0].centroid.is_empty());
    }

    // ---- re-add node -------------------------------------------------------

    #[test]
    fn test_add_node_overwrites_existing() {
        let mut g = default_graph();
        g.add_node(node("a", vec![1.0, 0.0]));
        g.add_node(node("a", vec![0.0, 1.0])); // same id, different embedding
                                               // The node should have been replaced.
        let n = g.get_node("a").expect("node must exist");
        assert_eq!(n.embedding, vec![0.0, 1.0]);
    }

    // ---- stats avg_similarity edge-less graph ------------------------------

    #[test]
    fn test_stats_no_edges_avg_similarity_zero() {
        let mut g = graph_with_threshold(0.99);
        g.add_node(node("a", vec![1.0, 0.0]));
        g.add_node(node("b", vec![0.0, 1.0]));
        let s = g.stats();
        assert_eq!(s.avg_similarity, 0.0);
    }

    // ---- isolated node count -----------------------------------------------

    #[test]
    fn test_stats_isolated_nodes_counted() {
        let mut g = graph_with_threshold(0.99);
        g.add_node(node("a", vec![1.0, 0.0]));
        g.add_node(node("b", vec![0.0, 1.0]));
        let s = g.stats();
        assert_eq!(s.isolated_nodes, 2);
    }

    // ---- SgNode metadata with HashMap --------------------------------------

    #[test]
    fn test_node_with_multiple_metadata() {
        let mut meta = HashMap::new();
        meta.insert("key1".to_owned(), "val1".to_owned());
        meta.insert("key2".to_owned(), "val2".to_owned());
        let n = SgNode {
            id: "x".to_owned(),
            embedding: vec![1.0],
            label: None,
            metadata: meta,
        };
        assert_eq!(n.metadata.len(), 2);
    }
}
