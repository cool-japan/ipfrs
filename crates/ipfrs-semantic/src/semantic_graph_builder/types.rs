//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::{HashMap, HashSet, VecDeque};

use super::functions::{cosine_similarity, now_secs};

/// A node in the semantic knowledge graph.
///
/// Exported as `SgbGraphNode` to avoid conflict with the `GraphNode` type
/// already re-exported from `graph_linker`.
#[derive(Debug, Clone)]
pub struct SgbGraphNode {
    /// Unique identifier
    pub id: String,
    /// Human-readable label
    pub label: String,
    /// Semantic role
    pub node_type: NodeType,
    /// Optional vector embedding (f64)
    pub embedding: Option<Vec<f64>>,
    /// Arbitrary key-value attributes
    pub attributes: Vec<(String, String)>,
    /// Current degree (edges incident to this node)
    pub degree: usize,
}
impl SgbGraphNode {
    /// Convenience constructor.
    pub fn new(id: impl Into<String>, label: impl Into<String>, node_type: NodeType) -> Self {
        Self {
            id: id.into(),
            label: label.into(),
            node_type,
            embedding: None,
            attributes: Vec::new(),
            degree: 0,
        }
    }
    /// Set the embedding.
    pub fn with_embedding(mut self, emb: Vec<f64>) -> Self {
        self.embedding = Some(emb);
        self
    }
    /// Add an attribute key-value pair.
    pub fn with_attribute(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.attributes.push((key.into(), value.into()));
        self
    }
}
/// Production-quality builder and manager for semantic knowledge graphs.
///
/// # Example
/// ```rust
/// use ipfrs_semantic::{
///     SemanticGraphBuilder, BuilderConfig, SgbGraphNode, NodeType, SgbGraphEdge, EdgeRelation,
/// };
///
/// let config = BuilderConfig {
///     similarity_threshold: 0.9,
///     ..Default::default()
/// };
/// let mut builder = SemanticGraphBuilder::new(config);
///
/// let node = SgbGraphNode::new("n1", "Rust", NodeType::Concept)
///     .with_embedding(vec![1.0, 0.0, 0.0]);
/// builder.add_node(node).unwrap();
/// ```
pub struct SemanticGraphBuilder {
    /// Node storage, keyed by id
    pub(super) nodes: HashMap<String, SgbGraphNode>,
    /// All edges in the graph
    pub(super) edges: Vec<SgbGraphEdge>,
    /// Fast adjacency index: node_id → list of edge indices in `edges`
    pub(super) adj: HashMap<String, Vec<usize>>,
    /// Builder configuration
    pub(super) config: BuilderConfig,
}
impl SemanticGraphBuilder {
    /// Create a new builder with the given configuration.
    pub fn new(config: BuilderConfig) -> Self {
        Self {
            nodes: HashMap::new(),
            edges: Vec::new(),
            adj: HashMap::new(),
            config,
        }
    }
    /// Create a builder with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(BuilderConfig::default())
    }
    /// Add a node to the graph.
    ///
    /// Returns `BuilderError::DuplicateNode` if the id already exists.
    /// If the node carries an embedding, `SimilarTo` edges are automatically
    /// added to existing nodes whose cosine similarity exceeds the threshold.
    pub fn add_node(&mut self, node: SgbGraphNode) -> Result<(), BuilderError> {
        if self.nodes.contains_key(&node.id) {
            return Err(BuilderError::DuplicateNode(node.id.clone()));
        }
        let mut sim_edges: Vec<SgbGraphEdge> = Vec::new();
        if let Some(ref emb) = node.embedding {
            for (existing_id, existing_node) in &self.nodes {
                if let Some(ref other_emb) = existing_node.embedding {
                    let sim = cosine_similarity(emb, other_emb);
                    if sim >= self.config.similarity_threshold {
                        sim_edges.push(SgbGraphEdge::new(
                            node.id.clone(),
                            existing_id.clone(),
                            EdgeRelation::SimilarTo,
                            sim,
                        ));
                    }
                }
            }
        }
        let node_id = node.id.clone();
        self.nodes.insert(node_id.clone(), node);
        self.adj.entry(node_id).or_default();
        for edge in sim_edges {
            let _ = self.add_edge(edge);
        }
        Ok(())
    }
    /// Remove a node and all edges incident to it.
    ///
    /// Returns `BuilderError::NodeNotFound` if the id is unknown.
    pub fn remove_node(&mut self, id: &str) -> Result<(), BuilderError> {
        if !self.nodes.contains_key(id) {
            return Err(BuilderError::NodeNotFound(id.to_owned()));
        }
        let old_len = self.edges.len();
        self.edges.retain(|e| e.from_id != id && e.to_id != id);
        let removed = old_len - self.edges.len();
        self.adj.remove(id);
        for v in self.adj.values_mut() {
            v.clear();
        }
        for (idx, edge) in self.edges.iter().enumerate() {
            self.adj.entry(edge.from_id.clone()).or_default().push(idx);
            self.adj.entry(edge.to_id.clone()).or_default().push(idx);
        }
        for node in self.nodes.values_mut() {
            node.degree = self.adj.get(&node.id).map(|v| v.len()).unwrap_or(0);
        }
        self.nodes.remove(id);
        let _ = removed;
        Ok(())
    }
    /// Look up a node by id.
    pub fn get_node(&self, id: &str) -> Option<&SgbGraphNode> {
        self.nodes.get(id)
    }
    /// Add an edge to the graph.
    ///
    /// Errors:
    /// - `SelfLoop` if `from_id == to_id`
    /// - `InvalidWeight` if weight is outside [0.0, 1.0]
    /// - `NodeNotFound` if either endpoint is unknown
    ///
    /// If adding the edge would cause either endpoint to exceed
    /// `max_edges_per_node`, the lowest-weight edge at that endpoint is
    /// silently dropped first.
    pub fn add_edge(&mut self, edge: SgbGraphEdge) -> Result<(), BuilderError> {
        if edge.from_id == edge.to_id {
            return Err(BuilderError::SelfLoop(edge.from_id.clone()));
        }
        if edge.weight < 0.0 || edge.weight > 1.0 {
            return Err(BuilderError::InvalidWeight(edge.weight));
        }
        if !self.nodes.contains_key(&edge.from_id) {
            return Err(BuilderError::NodeNotFound(edge.from_id.clone()));
        }
        if !self.nodes.contains_key(&edge.to_id) {
            return Err(BuilderError::NodeNotFound(edge.to_id.clone()));
        }
        for endpoint in [&edge.from_id, &edge.to_id] {
            let ep = endpoint.clone();
            self.enforce_max_edges(&ep);
        }
        let idx = self.edges.len();
        let from = edge.from_id.clone();
        let to = edge.to_id.clone();
        self.edges.push(edge);
        self.adj.entry(from).or_default().push(idx);
        self.adj.entry(to).or_default().push(idx);
        let from2 = self.edges[idx].from_id.clone();
        let to2 = self.edges[idx].to_id.clone();
        if let Some(n) = self.nodes.get_mut(&from2) {
            n.degree = self.adj.get(&from2).map(|v| v.len()).unwrap_or(0);
        }
        if let Some(n) = self.nodes.get_mut(&to2) {
            n.degree = self.adj.get(&to2).map(|v| v.len()).unwrap_or(0);
        }
        Ok(())
    }
    /// Drop the lowest-weight edge at a node if it exceeds `max_edges_per_node`.
    pub(super) fn enforce_max_edges(&mut self, node_id: &str) {
        let max = self.config.max_edges_per_node;
        let deg = self.adj.get(node_id).map(|v| v.len()).unwrap_or(0);
        if deg < max {
            return;
        }
        let min_idx = self.adj.get(node_id).and_then(|idxs| {
            idxs.iter()
                .min_by(|&&a, &&b| {
                    self.edges[a]
                        .weight
                        .partial_cmp(&self.edges[b].weight)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .copied()
        });
        if let Some(drop_idx) = min_idx {
            let from = self.edges[drop_idx].from_id.clone();
            let to = self.edges[drop_idx].to_id.clone();
            self.edges.remove(drop_idx);
            self.rebuild_adj_fast(&from, &to, drop_idx);
        }
    }
    /// Rebuild the adjacency indices for two nodes after an edge removal.
    pub(super) fn rebuild_adj_fast(&mut self, from: &str, to: &str, removed_idx: usize) {
        for adj_list in self.adj.values_mut() {
            adj_list.retain(|&i| i != removed_idx);
            for i in adj_list.iter_mut() {
                if *i > removed_idx {
                    *i -= 1;
                }
            }
        }
        if let Some(n) = self.nodes.get_mut(from) {
            n.degree = self.adj.get(from).map(|v| v.len()).unwrap_or(0);
        }
        if let Some(n) = self.nodes.get_mut(to) {
            n.degree = self.adj.get(to).map(|v| v.len()).unwrap_or(0);
        }
    }
    /// Build `Term` nodes and `CoOccurs` edges from a raw text document.
    ///
    /// Tokenizes `text` into lowercase alpha-only words, creates a `Term`
    /// node for each unique token (keyed `{doc_id}::{word}`), and links
    /// pairs within `cooccurrence_window` positions with `CoOccurs` edges.
    ///
    /// Returns the list of new `SgbGraphNode`s created.
    pub fn build_from_text(
        &mut self,
        doc_id: &str,
        text: &str,
    ) -> Result<Vec<SgbGraphNode>, BuilderError> {
        let tokens: Vec<String> = text
            .split(|c: char| !c.is_alphabetic())
            .filter(|t| !t.is_empty())
            .map(|t| t.to_lowercase())
            .collect();
        let window = self.config.cooccurrence_window;
        let mut new_ids: Vec<String> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();
        for token in &tokens {
            let node_id = format!("{doc_id}::{token}");
            if seen.insert(node_id.clone()) && !self.nodes.contains_key(&node_id) {
                let node = SgbGraphNode::new(node_id.clone(), token.clone(), NodeType::Term);
                self.add_node(node)?;
                new_ids.push(node_id);
            }
        }
        for i in 0..tokens.len() {
            let id_i = format!("{doc_id}::{}", tokens[i]);
            let end = (i + 1 + window).min(tokens.len());
            for j in (i + 1)..end {
                if tokens[i] == tokens[j] {
                    continue;
                }
                let id_j = format!("{doc_id}::{}", tokens[j]);
                let already = self.edges.iter().any(|e| {
                    e.relation == EdgeRelation::CoOccurs
                        && ((e.from_id == id_i && e.to_id == id_j)
                            || (e.from_id == id_j && e.to_id == id_i))
                });
                if !already {
                    let dist = (j - i) as f64;
                    let weight = (1.0 / dist).clamp(0.0, 1.0);
                    let edge = SgbGraphEdge::new(
                        id_i.clone(),
                        id_j.clone(),
                        EdgeRelation::CoOccurs,
                        weight,
                    );
                    let _ = self.add_edge(edge);
                }
            }
        }
        let new_nodes: Vec<SgbGraphNode> = new_ids
            .iter()
            .filter_map(|id| self.nodes.get(id).cloned())
            .collect();
        Ok(new_nodes)
    }
    /// Scan all node pairs that have embeddings and add `SimilarTo` edges
    /// whose cosine similarity exceeds `threshold` (or `config.similarity_threshold`
    /// if `None`).
    ///
    /// Returns the number of edges added.
    pub fn build_similarity_edges(&mut self, threshold: Option<f64>) -> usize {
        let thresh = threshold.unwrap_or(self.config.similarity_threshold);
        let ids: Vec<String> = self.nodes.keys().cloned().collect();
        let mut to_add: Vec<SgbGraphEdge> = Vec::new();
        for i in 0..ids.len() {
            for j in (i + 1)..ids.len() {
                let emb_i = match self.nodes.get(&ids[i]).and_then(|n| n.embedding.as_deref()) {
                    Some(e) => e.to_vec(),
                    None => continue,
                };
                let emb_j = match self.nodes.get(&ids[j]).and_then(|n| n.embedding.as_deref()) {
                    Some(e) => e.to_vec(),
                    None => continue,
                };
                let sim = cosine_similarity(&emb_i, &emb_j);
                if sim >= thresh {
                    let already = self.edges.iter().any(|e| {
                        e.relation.variant_eq(&EdgeRelation::SimilarTo)
                            && ((e.from_id == ids[i] && e.to_id == ids[j])
                                || (e.from_id == ids[j] && e.to_id == ids[i]))
                    });
                    if !already {
                        to_add.push(SgbGraphEdge::new(
                            ids[i].clone(),
                            ids[j].clone(),
                            EdgeRelation::SimilarTo,
                            sim,
                        ));
                    }
                }
            }
        }
        let count = to_add.len();
        for edge in to_add {
            let _ = self.add_edge(edge);
        }
        count
    }
    /// If `enable_transitive_closure` is set, propagate `PartOf` edges:
    /// if A PartOf B and B PartOf C then add A PartOf C.
    pub fn apply_transitive_closure(&mut self) {
        if !self.config.enable_transitive_closure {
            return;
        }
        loop {
            let part_of_pairs: Vec<(String, String)> = self
                .edges
                .iter()
                .filter(|e| e.relation.variant_eq(&EdgeRelation::PartOf))
                .map(|e| (e.from_id.clone(), e.to_id.clone()))
                .collect();
            let mut new_edges: Vec<SgbGraphEdge> = Vec::new();
            for (a, b) in &part_of_pairs {
                for (b2, c) in &part_of_pairs {
                    if b == b2 && a != c {
                        let already = self.edges.iter().any(|e| {
                            e.relation.variant_eq(&EdgeRelation::PartOf)
                                && &e.from_id == a
                                && &e.to_id == c
                        });
                        if !already {
                            new_edges.push(SgbGraphEdge::new(
                                a.clone(),
                                c.clone(),
                                EdgeRelation::PartOf,
                                1.0,
                            ));
                        }
                    }
                }
            }
            if new_edges.is_empty() {
                break;
            }
            for e in new_edges {
                let _ = self.add_edge(e);
            }
        }
    }
    /// Return nodes reachable via BFS from `query.start_node` (or all nodes if
    /// `None`), filtered by node type and edge constraints.
    pub fn subgraph(&self, query: &SgbGraphQuery) -> Result<Vec<SgbGraphNode>, BuilderError> {
        if let Some(ref start) = query.start_node {
            if !self.nodes.contains_key(start.as_str()) {
                return Err(BuilderError::NodeNotFound(start.clone()));
            }
        }
        let seeds: Vec<String> = match &query.start_node {
            Some(s) => vec![s.clone()],
            None => self.nodes.keys().cloned().collect(),
        };
        let mut visited: HashSet<String> = HashSet::new();
        let mut result: Vec<SgbGraphNode> = Vec::new();
        for seed in seeds {
            if visited.contains(&seed) {
                continue;
            }
            let mut queue: VecDeque<(String, usize)> = VecDeque::new();
            queue.push_back((seed.clone(), 0));
            while let Some((current, depth)) = queue.pop_front() {
                if visited.contains(&current) {
                    continue;
                }
                if depth > query.max_depth {
                    continue;
                }
                visited.insert(current.clone());
                if let Some(node) = self.nodes.get(&current) {
                    let type_ok =
                        query.node_types.is_empty() || query.node_types.contains(&node.node_type);
                    if type_ok {
                        result.push(node.clone());
                    }
                }
                if depth < query.max_depth {
                    if let Some(idxs) = self.adj.get(&current) {
                        for &idx in idxs {
                            let edge = &self.edges[idx];
                            if edge.weight < query.min_edge_weight {
                                continue;
                            }
                            if !edge.relation.matches(&query.relation_filter) {
                                continue;
                            }
                            let neighbor = if edge.from_id == current {
                                &edge.to_id
                            } else {
                                &edge.from_id
                            };
                            if !visited.contains(neighbor.as_str()) {
                                queue.push_back((neighbor.clone(), depth + 1));
                            }
                        }
                    }
                }
            }
        }
        Ok(result)
    }
    /// Return all nodes reachable within `depth` hops from `node_id`.
    pub fn neighborhood(
        &self,
        node_id: &str,
        depth: usize,
    ) -> Result<Vec<SgbGraphNode>, BuilderError> {
        if !self.nodes.contains_key(node_id) {
            return Err(BuilderError::NodeNotFound(node_id.to_owned()));
        }
        let mut visited: HashSet<String> = HashSet::new();
        let mut result: Vec<SgbGraphNode> = Vec::new();
        let mut queue: VecDeque<(String, usize)> = VecDeque::new();
        queue.push_back((node_id.to_owned(), 0));
        while let Some((current, d)) = queue.pop_front() {
            if visited.contains(&current) {
                continue;
            }
            visited.insert(current.clone());
            if let Some(node) = self.nodes.get(&current) {
                result.push(node.clone());
            }
            if d < depth {
                if let Some(idxs) = self.adj.get(&current) {
                    for &idx in idxs {
                        let edge = &self.edges[idx];
                        let neighbor = if edge.from_id == current {
                            &edge.to_id
                        } else {
                            &edge.from_id
                        };
                        if !visited.contains(neighbor.as_str()) {
                            queue.push_back((neighbor.clone(), d + 1));
                        }
                    }
                }
            }
        }
        Ok(result)
    }
    /// Return the shortest path (by hop count) between two nodes as a
    /// sequence of node ids including both endpoints.
    ///
    /// Returns `BuilderError::NodeNotFound` if either id is absent.
    /// Returns an empty vector if no path exists.
    pub fn path(&self, from: &str, to: &str) -> Result<Vec<String>, BuilderError> {
        if !self.nodes.contains_key(from) {
            return Err(BuilderError::NodeNotFound(from.to_owned()));
        }
        if !self.nodes.contains_key(to) {
            return Err(BuilderError::NodeNotFound(to.to_owned()));
        }
        if from == to {
            return Ok(vec![from.to_owned()]);
        }
        let mut visited: HashSet<String> = HashSet::new();
        let mut prev: HashMap<String, String> = HashMap::new();
        let mut queue: VecDeque<String> = VecDeque::new();
        queue.push_back(from.to_owned());
        visited.insert(from.to_owned());
        'bfs: while let Some(current) = queue.pop_front() {
            if let Some(idxs) = self.adj.get(&current) {
                for &idx in idxs {
                    let edge = &self.edges[idx];
                    let neighbor = if edge.from_id == current {
                        &edge.to_id
                    } else {
                        &edge.from_id
                    };
                    if visited.contains(neighbor.as_str()) {
                        continue;
                    }
                    visited.insert(neighbor.clone());
                    prev.insert(neighbor.clone(), current.clone());
                    if neighbor == to {
                        break 'bfs;
                    }
                    queue.push_back(neighbor.clone());
                }
            }
        }
        if !prev.contains_key(to) {
            return Ok(Vec::new());
        }
        let mut path: Vec<String> = Vec::new();
        let mut cur = to.to_owned();
        while cur != from {
            path.push(cur.clone());
            cur = prev[&cur].clone();
        }
        path.push(from.to_owned());
        path.reverse();
        Ok(path)
    }
    /// Return all weakly-connected components as lists of node ids.
    pub fn connected_components(&self) -> Vec<Vec<String>> {
        let ids: Vec<String> = self.nodes.keys().cloned().collect();
        let n = ids.len();
        if n == 0 {
            return Vec::new();
        }
        let id_to_idx: HashMap<&str, usize> = ids
            .iter()
            .enumerate()
            .map(|(i, id)| (id.as_str(), i))
            .collect();
        let mut uf = UnionFind::new(n);
        for edge in &self.edges {
            if let (Some(&ai), Some(&bi)) = (
                id_to_idx.get(edge.from_id.as_str()),
                id_to_idx.get(edge.to_id.as_str()),
            ) {
                uf.union(ai, bi);
            }
        }
        let mut components: HashMap<usize, Vec<String>> = HashMap::new();
        for (i, id) in ids.iter().enumerate() {
            let root = uf.find(i);
            components.entry(root).or_default().push(id.clone());
        }
        components.into_values().collect()
    }
    /// Merge two nodes into a new node with a fresh id.
    ///
    /// - Attributes are combined (union).
    /// - Embeddings are averaged (if both have one).
    /// - All edges previously incident to either node are redirected to the
    ///   new node (self-loops resulting from the merge are dropped).
    ///
    /// Returns the newly created merged node.
    pub fn merge_nodes(
        &mut self,
        id_a: &str,
        id_b: &str,
        new_id: String,
    ) -> Result<SgbGraphNode, BuilderError> {
        if !self.nodes.contains_key(id_a) {
            return Err(BuilderError::NodeNotFound(id_a.to_owned()));
        }
        if !self.nodes.contains_key(id_b) {
            return Err(BuilderError::NodeNotFound(id_b.to_owned()));
        }
        if self.nodes.contains_key(&new_id) {
            return Err(BuilderError::DuplicateNode(new_id));
        }
        let node_a = self.nodes[id_a].clone();
        let node_b = self.nodes[id_b].clone();
        let mut attr_map: HashMap<String, String> = HashMap::new();
        for (k, v) in node_a.attributes.iter().chain(node_b.attributes.iter()) {
            attr_map.entry(k.clone()).or_insert_with(|| v.clone());
        }
        let attributes: Vec<(String, String)> = attr_map.into_iter().collect();
        let embedding = match (&node_a.embedding, &node_b.embedding) {
            (Some(ea), Some(eb)) => {
                if ea.len() == eb.len() {
                    Some(
                        ea.iter()
                            .zip(eb.iter())
                            .map(|(x, y)| (x + y) / 2.0)
                            .collect::<Vec<f64>>(),
                    )
                } else {
                    node_a.embedding.clone()
                }
            }
            (Some(e), None) | (None, Some(e)) => Some(e.clone()),
            (None, None) => None,
        };
        let merged = SgbGraphNode {
            id: new_id.clone(),
            label: format!("{}+{}", node_a.label, node_b.label),
            node_type: node_a.node_type.clone(),
            embedding,
            attributes,
            degree: 0,
        };
        let old_edges: Vec<SgbGraphEdge> = self.edges.clone();
        self.remove_node(id_a)?;
        self.remove_node(id_b)?;
        self.nodes.insert(new_id.clone(), merged.clone());
        self.adj.entry(new_id.clone()).or_default();
        for mut edge in old_edges {
            let touches_a = edge.from_id == id_a || edge.to_id == id_a;
            let touches_b = edge.from_id == id_b || edge.to_id == id_b;
            if !touches_a && !touches_b {
                continue;
            }
            if edge.from_id == id_a || edge.from_id == id_b {
                edge.from_id = new_id.clone();
            }
            if edge.to_id == id_a || edge.to_id == id_b {
                edge.to_id = new_id.clone();
            }
            if edge.from_id == edge.to_id {
                continue;
            }
            if self.nodes.contains_key(&edge.from_id) && self.nodes.contains_key(&edge.to_id) {
                let _ = self.add_edge(edge);
            }
        }
        let final_degree = self.adj.get(&new_id).map(|v| v.len()).unwrap_or(0);
        if let Some(n) = self.nodes.get_mut(&new_id) {
            n.degree = final_degree;
        }
        Ok(self.nodes[&new_id].clone())
    }
    /// Compute aggregate statistics for the current graph state.
    pub fn stats(&self) -> GraphStats {
        let node_count = self.nodes.len();
        let edge_count = self.edges.len();
        let avg_degree = if node_count == 0 {
            0.0
        } else {
            (2 * edge_count) as f64 / node_count as f64
        };
        let density = if node_count < 2 {
            0.0
        } else {
            edge_count as f64 / (node_count * (node_count - 1)) as f64
        };
        let components = self.connected_components();
        let connected_components = components.len();
        let largest_component_size = components.iter().map(|c| c.len()).max().unwrap_or(0);
        GraphStats {
            node_count,
            edge_count,
            avg_degree,
            density,
            connected_components,
            largest_component_size,
        }
    }
    /// Number of nodes currently in the graph.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
    /// Number of edges currently in the graph.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }
    /// Iterator over all nodes.
    pub fn iter_nodes(&self) -> impl Iterator<Item = &SgbGraphNode> {
        self.nodes.values()
    }
    /// Iterator over all edges.
    pub fn iter_edges(&self) -> impl Iterator<Item = &SgbGraphEdge> {
        self.edges.iter()
    }
}
/// The semantic relationship expressed by an edge.
#[derive(Debug, Clone, PartialEq)]
pub enum EdgeRelation {
    /// Nodes have high vector similarity
    SimilarTo,
    /// Generic bidirectional relation
    RelatedTo,
    /// `from` is part of `to`
    PartOf,
    /// `from` has the property `to`
    HasProperty,
    /// `from` and `to` co-occur in a window
    CoOccurs,
    /// `from` subsumes (generalises) `to`
    Subsumes,
    /// `from` and `to` are antonyms / opposites
    Opposes,
    /// User-defined relation label
    Defined(String),
}
impl EdgeRelation {
    pub(super) fn matches(&self, filter: &[EdgeRelation]) -> bool {
        if filter.is_empty() {
            return true;
        }
        filter.iter().any(|f| self.variant_eq(f))
    }
    pub(super) fn variant_eq(&self, other: &EdgeRelation) -> bool {
        match (self, other) {
            (EdgeRelation::SimilarTo, EdgeRelation::SimilarTo) => true,
            (EdgeRelation::RelatedTo, EdgeRelation::RelatedTo) => true,
            (EdgeRelation::PartOf, EdgeRelation::PartOf) => true,
            (EdgeRelation::HasProperty, EdgeRelation::HasProperty) => true,
            (EdgeRelation::CoOccurs, EdgeRelation::CoOccurs) => true,
            (EdgeRelation::Subsumes, EdgeRelation::Subsumes) => true,
            (EdgeRelation::Opposes, EdgeRelation::Opposes) => true,
            (EdgeRelation::Defined(a), EdgeRelation::Defined(b)) => a == b,
            _ => false,
        }
    }
}
/// A directed, weighted edge between two nodes.
///
/// Exported as `SgbGraphEdge` to avoid conflict with the `GraphEdge` type
/// already re-exported from `knowledge_graph`.
#[derive(Debug, Clone)]
pub struct SgbGraphEdge {
    /// Source node id
    pub from_id: String,
    /// Destination node id
    pub to_id: String,
    /// Semantic relation
    pub relation: EdgeRelation,
    /// Edge weight in [0, 1]
    pub weight: f64,
    /// Unix timestamp of creation
    pub created_at: u64,
}
impl SgbGraphEdge {
    /// Convenience constructor (sets `created_at` to the current wall-clock time).
    pub fn new(
        from_id: impl Into<String>,
        to_id: impl Into<String>,
        relation: EdgeRelation,
        weight: f64,
    ) -> Self {
        Self {
            from_id: from_id.into(),
            to_id: to_id.into(),
            relation,
            weight,
            created_at: now_secs(),
        }
    }
}
/// Aggregate statistics for the knowledge graph.
#[derive(Debug, Clone)]
pub struct GraphStats {
    /// Total number of nodes
    pub node_count: usize,
    /// Total number of edges
    pub edge_count: usize,
    /// Mean node degree
    pub avg_degree: f64,
    /// Graph density: `edges / (nodes * (nodes - 1))`
    pub density: f64,
    /// Number of weakly-connected components
    pub connected_components: usize,
    /// Node count of the largest component
    pub largest_component_size: usize,
}
/// Errors produced by [`SemanticGraphBuilder`].
#[derive(Debug, Clone, PartialEq)]
pub enum BuilderError {
    /// Referenced node id was not found
    NodeNotFound(String),
    /// A node with this id already exists
    DuplicateNode(String),
    /// A self-loop (from == to) was attempted
    SelfLoop(String),
    /// Edge weight is outside [0, 1]
    InvalidWeight(f64),
    /// Graph exceeds the configured maximum size
    GraphTooLarge(usize),
}
pub(super) struct UnionFind {
    pub(super) parent: Vec<usize>,
    pub(super) rank: Vec<usize>,
}
impl UnionFind {
    pub(super) fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }
    pub(super) fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }
    pub(super) fn union(&mut self, x: usize, y: usize) {
        let rx = self.find(x);
        let ry = self.find(y);
        if rx == ry {
            return;
        }
        match self.rank[rx].cmp(&self.rank[ry]) {
            std::cmp::Ordering::Less => self.parent[rx] = ry,
            std::cmp::Ordering::Greater => self.parent[ry] = rx,
            std::cmp::Ordering::Equal => {
                self.parent[ry] = rx;
                self.rank[rx] += 1;
            }
        }
    }
}
/// The semantic role of a node in the knowledge graph.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum NodeType {
    /// A real-world entity (person, place, organisation, …)
    Entity,
    /// An abstract concept
    Concept,
    /// A document / content item
    Document,
    /// A vocabulary term
    Term,
    /// A cluster of related items
    Cluster,
}
/// Configuration for [`SemanticGraphBuilder`].
#[derive(Debug, Clone)]
pub struct BuilderConfig {
    /// Cosine similarity threshold for auto-adding `SimilarTo` edges on `add_node`
    pub similarity_threshold: f64,
    /// Maximum number of edges per node; excess lowest-weight edges are dropped
    pub max_edges_per_node: usize,
    /// If `true`, `PartOf` edges are extended transitively
    pub enable_transitive_closure: bool,
    /// Number of adjacent words that produce a `CoOccurs` edge in text building
    pub cooccurrence_window: usize,
}
/// Query parameters for subgraph / BFS traversal.
///
/// Exported as `SgbGraphQuery` to avoid conflict with the `GraphQuery` type
/// already re-exported from `knowledge_graph`.
#[derive(Debug, Clone, Default)]
pub struct SgbGraphQuery {
    /// Only include nodes of these types (empty ⇒ all types)
    pub node_types: Vec<NodeType>,
    /// Minimum edge weight to traverse
    pub min_edge_weight: f64,
    /// Maximum BFS depth
    pub max_depth: usize,
    /// Optional starting node id (BFS root); `None` ⇒ all nodes
    pub start_node: Option<String>,
    /// Only traverse edges with these relations (empty ⇒ all)
    pub relation_filter: Vec<EdgeRelation>,
}
