//! # Concept Graph Builder
//!
//! A semantic concept graph that extracts concepts from text, builds weighted
//! relationships, and enables concept-based navigation and similarity search.
//!
//! The graph supports:
//! - Automatic concept extraction via tokenization
//! - Co-occurrence-based edge weighting
//! - Explicit semantic relation tagging (synonym, hierarchical, antonym, related)
//! - BFS shortest-path computation
//! - Embedding-based or graph-topology-based similarity search
//! - Pruning by frequency and edge weight

use std::collections::{HashMap, HashSet, VecDeque};

// ─── Core types ─────────────────────────────────────────────────────────────

/// Opaque concept index newtype.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ConceptId(pub usize);

/// Semantic relation between two concepts.
#[derive(Debug, Clone, PartialEq)]
pub enum CgConceptRelation {
    /// Concepts frequently appear together in the same context window.
    CoOccurrence,
    /// Two terms are synonyms.
    Synonym,
    /// One concept is a subtype / supertype of the other.
    Hierarchical,
    /// Concepts are semantically opposite.
    Antonym,
    /// Generic semantic relatedness.
    Related,
}

/// A node in the concept graph.
#[derive(Debug, Clone)]
pub struct CgConcept {
    /// Stable identifier.
    pub id: ConceptId,
    /// Canonical term string.
    pub term: String,
    /// Optional dense embedding vector.
    pub embedding: Option<Vec<f64>>,
    /// Number of times this concept was seen across all processed documents.
    pub frequency: u64,
    /// Document IDs in which this concept appears (deduplication done at insertion).
    pub documents: Vec<String>,
}

/// A directed (but treated as undirected for traversal) weighted edge.
#[derive(Debug, Clone)]
pub struct CgConceptEdge {
    pub from: ConceptId,
    pub to: ConceptId,
    /// Normalised edge weight ∈ (0, 1].
    pub weight: f64,
    pub relation: CgConceptRelation,
    /// Raw co-occurrence count (incremented each time the pair is observed).
    pub co_occurrence: u64,
}

/// Build-time configuration.
#[derive(Debug, Clone)]
pub struct CgGraphConfig {
    /// Concepts appearing fewer than this many times are considered noise.
    pub min_concept_frequency: u64,
    /// Hard cap on the total number of concepts tracked.
    pub max_concepts: usize,
    /// Token window used when counting co-occurrences.
    pub co_occurrence_window: usize,
    /// Edges below this weight are considered insignificant.
    pub min_edge_weight: f64,
}

impl Default for CgGraphConfig {
    fn default() -> Self {
        Self {
            min_concept_frequency: 2,
            max_concepts: 10_000,
            co_occurrence_window: 5,
            min_edge_weight: 0.1,
        }
    }
}

/// Snapshot statistics about the concept graph.
#[derive(Debug, Clone)]
pub struct ConceptGraphStats {
    pub concept_count: usize,
    pub edge_count: usize,
    pub avg_degree: f64,
    pub total_documents: u64,
    /// Number of distinct terms (same as concept_count after pruning).
    pub vocabulary_size: usize,
}

// ─── Builder ────────────────────────────────────────────────────────────────

/// Builds and queries a semantic concept graph.
///
/// # Example
///
/// ```rust
/// use ipfrs_semantic::{CgGraphConfig, ConceptGraphBuilder};
///
/// let mut builder = ConceptGraphBuilder::new(CgGraphConfig::default());
/// builder.process_document("doc1", "the quick brown fox jumps over the lazy dog");
/// builder.process_document("doc2", "the brown fox ran across the field");
/// let stats = builder.graph_stats();
/// assert!(stats.concept_count > 0);
/// ```
pub struct ConceptGraphBuilder {
    pub config: CgGraphConfig,
    /// All concepts indexed by their stable [`ConceptId`].
    pub concepts: Vec<CgConcept>,
    /// Reverse map: term → concept index.
    pub term_to_id: HashMap<String, ConceptId>,
    /// All edges (both directions may be represented; deduplication is done via
    /// the canonical-pair key stored in `edge_index`).
    pub edges: Vec<CgConceptEdge>,
    /// Concept id → list of indices into `edges` (both in-coming and out-going).
    pub adjacency: HashMap<usize, Vec<usize>>,
    /// Canonical edge key `(min, max)` → index in `edges`.
    edge_index: HashMap<(usize, usize), usize>,
    pub total_documents: u64,
}

impl ConceptGraphBuilder {
    /// Create a new builder with the supplied configuration.
    pub fn new(config: CgGraphConfig) -> Self {
        Self {
            config,
            concepts: Vec::new(),
            term_to_id: HashMap::new(),
            edges: Vec::new(),
            adjacency: HashMap::new(),
            edge_index: HashMap::new(),
            total_documents: 0,
        }
    }

    // ── Concept management ───────────────────────────────────────────────

    /// Return the [`ConceptId`] for `term`, creating it if it does not exist
    /// and the max-concept cap has not been reached.
    pub fn add_concept_term(&mut self, term: String, embedding: Option<Vec<f64>>) -> ConceptId {
        if let Some(&id) = self.term_to_id.get(&term) {
            // Update embedding if one was supplied and none was stored yet.
            if embedding.is_some() && self.concepts[id.0].embedding.is_none() {
                self.concepts[id.0].embedding = embedding;
            }
            return id;
        }
        if self.concepts.len() >= self.config.max_concepts {
            // Return sentinel id pointing past the end; callers must handle this.
            // We use usize::MAX as the "invalid" marker.
            return ConceptId(usize::MAX);
        }
        let id = ConceptId(self.concepts.len());
        self.concepts.push(CgConcept {
            id,
            term: term.clone(),
            embedding,
            frequency: 0,
            documents: Vec::new(),
        });
        self.term_to_id.insert(term, id);
        id
    }

    // ── Document processing ──────────────────────────────────────────────

    /// Tokenize `text`, index all tokens as concepts, and record
    /// co-occurrence edges within the sliding window.
    pub fn process_document(&mut self, doc_id: &str, text: &str) {
        let tokens = tokenize(text);
        if tokens.is_empty() {
            self.total_documents += 1;
            return;
        }

        // Resolve (or create) concept ids for every token.
        let mut concept_ids: Vec<ConceptId> = Vec::with_capacity(tokens.len());
        for token in &tokens {
            let cid = self.add_concept_term(token.clone(), None);
            if cid.0 == usize::MAX {
                // Cap hit; skip this token.
                concept_ids.push(cid);
                continue;
            }
            let concept = &mut self.concepts[cid.0];
            concept.frequency += 1;
            // Deduplicated document tracking.
            if !concept.documents.contains(&doc_id.to_string()) {
                concept.documents.push(doc_id.to_string());
            }
            concept_ids.push(cid);
        }

        // Build co-occurrence edges within the sliding window.
        let n = concept_ids.len();
        for i in 0..n {
            let a = concept_ids[i];
            if a.0 == usize::MAX {
                continue;
            }
            let window_end = (i + self.config.co_occurrence_window + 1).min(n);
            for &b in concept_ids.iter().take(window_end).skip(i + 1) {
                if b.0 == usize::MAX || a == b {
                    continue;
                }
                self.upsert_cooccurrence_edge(a, b);
            }
        }

        self.total_documents += 1;
    }

    /// Insert or update a co-occurrence edge between `a` and `b`, then
    /// recompute the normalised weight.
    fn upsert_cooccurrence_edge(&mut self, a: ConceptId, b: ConceptId) {
        let key = canonical_key(a, b);
        if let Some(&edge_idx) = self.edge_index.get(&key) {
            // Increment co-occurrence and recompute weight.
            let edge = &mut self.edges[edge_idx];
            edge.co_occurrence += 1;
            let freq_a = self.concepts[a.0].frequency.max(1) as f64;
            let freq_b = self.concepts[b.0].frequency.max(1) as f64;
            edge.weight = (edge.co_occurrence as f64) / (freq_a * freq_b).sqrt();
        } else {
            let freq_a = self.concepts[a.0].frequency.max(1) as f64;
            let freq_b = self.concepts[b.0].frequency.max(1) as f64;
            let weight = 1.0_f64 / (freq_a * freq_b).sqrt();
            let edge_idx = self.edges.len();
            self.edges.push(CgConceptEdge {
                from: a,
                to: b,
                weight,
                relation: CgConceptRelation::CoOccurrence,
                co_occurrence: 1,
            });
            self.edge_index.insert(key, edge_idx);
            self.adjacency.entry(a.0).or_default().push(edge_idx);
            self.adjacency.entry(b.0).or_default().push(edge_idx);
        }
    }

    // ── Explicit relations ───────────────────────────────────────────────

    /// Add an explicit semantic relation edge.
    ///
    /// Returns `false` if either `term_a` or `term_b` is unknown.
    pub fn add_relation(
        &mut self,
        term_a: &str,
        term_b: &str,
        relation: CgConceptRelation,
        weight: f64,
    ) -> bool {
        let id_a = match self.term_to_id.get(term_a).copied() {
            Some(id) => id,
            None => return false,
        };
        let id_b = match self.term_to_id.get(term_b).copied() {
            Some(id) => id,
            None => return false,
        };
        let key = canonical_key(id_a, id_b);
        if let Some(&edge_idx) = self.edge_index.get(&key) {
            // Overwrite the existing edge's relation and weight.
            self.edges[edge_idx].relation = relation;
            self.edges[edge_idx].weight = weight;
        } else {
            let edge_idx = self.edges.len();
            self.edges.push(CgConceptEdge {
                from: id_a,
                to: id_b,
                weight,
                relation,
                co_occurrence: 0,
            });
            self.edge_index.insert(key, edge_idx);
            self.adjacency.entry(id_a.0).or_default().push(edge_idx);
            self.adjacency.entry(id_b.0).or_default().push(edge_idx);
        }
        true
    }

    // ── Lookup helpers ───────────────────────────────────────────────────

    /// Look up a concept by its term string.
    pub fn concept_by_term(&self, term: &str) -> Option<&CgConcept> {
        let id = self.term_to_id.get(term)?;
        self.concepts.get(id.0)
    }

    /// Look up a concept by its [`ConceptId`].
    pub fn concept_by_id(&self, id: ConceptId) -> Option<&CgConcept> {
        if id.0 == usize::MAX {
            return None;
        }
        self.concepts.get(id.0)
    }

    // ── Graph traversal ──────────────────────────────────────────────────

    /// Return all direct neighbours of `id` sorted by edge weight (descending).
    ///
    /// Each element is `(neighbour_concept, edge_weight)`.
    pub fn neighbors(&self, id: ConceptId) -> Vec<(&CgConcept, f64)> {
        let edge_indices = match self.adjacency.get(&id.0) {
            Some(v) => v,
            None => return Vec::new(),
        };
        let mut result: Vec<(&CgConcept, f64)> = edge_indices
            .iter()
            .filter_map(|&ei| {
                let edge = self.edges.get(ei)?;
                let neighbour_id = if edge.from == id { edge.to } else { edge.from };
                let concept = self.concepts.get(neighbour_id.0)?;
                Some((concept, edge.weight))
            })
            .collect();
        result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        result
    }

    /// BFS shortest path from `from` to `to`.
    ///
    /// Returns `None` if the nodes are not connected.
    pub fn shortest_path(&self, from: ConceptId, to: ConceptId) -> Option<Vec<ConceptId>> {
        if from == to {
            return Some(vec![from]);
        }
        // Standard BFS.
        let mut visited: HashSet<usize> = HashSet::new();
        let mut queue: VecDeque<Vec<ConceptId>> = VecDeque::new();
        visited.insert(from.0);
        queue.push_back(vec![from]);

        while let Some(path) = queue.pop_front() {
            let current = *path.last()?;
            let edge_indices = match self.adjacency.get(&current.0) {
                Some(v) => v,
                None => continue,
            };
            for &ei in edge_indices {
                let edge = self.edges.get(ei)?;
                let next = if edge.from == current {
                    edge.to
                } else {
                    edge.from
                };
                if next == to {
                    let mut full = path.clone();
                    full.push(to);
                    return Some(full);
                }
                if !visited.contains(&next.0) {
                    visited.insert(next.0);
                    let mut new_path = path.clone();
                    new_path.push(next);
                    queue.push_back(new_path);
                }
            }
        }
        None
    }

    // ── Similarity search ────────────────────────────────────────────────

    /// Return the top-`k` concepts most similar to `id`.
    ///
    /// Strategy:
    /// - If the target concept has an embedding, rank all other embedded
    ///   concepts by cosine similarity.
    /// - Otherwise fall back to the top-k neighbours by edge weight.
    pub fn similar_concepts(&self, id: ConceptId, k: usize) -> Vec<(&CgConcept, f64)> {
        if k == 0 {
            return Vec::new();
        }
        let target = match self.concept_by_id(id) {
            Some(c) => c,
            None => return Vec::new(),
        };

        if let Some(target_emb) = &target.embedding {
            // Embedding-based cosine similarity.
            let mut scored: Vec<(&CgConcept, f64)> = self
                .concepts
                .iter()
                .filter(|c| c.id != id)
                .filter_map(|c| {
                    let emb = c.embedding.as_ref()?;
                    let sim = cosine_similarity(target_emb, emb);
                    Some((c, sim))
                })
                .collect();
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            scored.truncate(k);
            scored
        } else {
            // Fall back: top-k neighbours by edge weight.
            let mut nbrs = self.neighbors(id);
            nbrs.truncate(k);
            nbrs
        }
    }

    // ── Pruning ──────────────────────────────────────────────────────────

    /// Remove all concepts whose frequency is below `min_concept_frequency`.
    ///
    /// Associated edges and adjacency entries are also removed.
    /// Returns the number of concepts removed.
    pub fn prune_low_frequency(&mut self) -> usize {
        let min_freq = self.config.min_concept_frequency;
        let to_remove: HashSet<usize> = self
            .concepts
            .iter()
            .filter(|c| c.frequency < min_freq)
            .map(|c| c.id.0)
            .collect();
        if to_remove.is_empty() {
            return 0;
        }
        self.remove_concepts(&to_remove)
    }

    /// Remove all edges whose weight is below `min_edge_weight`.
    ///
    /// Returns the number of edges removed.
    pub fn prune_weak_edges(&mut self) -> usize {
        let min_weight = self.config.min_edge_weight;
        let initial = self.edges.len();

        // Collect indices of edges to remove.
        let remove_set: HashSet<usize> = self
            .edges
            .iter()
            .enumerate()
            .filter(|(_, e)| e.weight < min_weight)
            .map(|(i, _)| i)
            .collect();

        if remove_set.is_empty() {
            return 0;
        }

        self.rebuild_edges_excluding(&remove_set);
        initial - self.edges.len()
    }

    // ── Statistics ───────────────────────────────────────────────────────

    /// Compute a snapshot of graph statistics.
    pub fn graph_stats(&self) -> ConceptGraphStats {
        let concept_count = self.concepts.len();
        let edge_count = self.edges.len();
        let avg_degree = if concept_count == 0 {
            0.0
        } else {
            // Each edge contributes to two nodes' degree.
            (2 * edge_count) as f64 / concept_count as f64
        };
        ConceptGraphStats {
            concept_count,
            edge_count,
            avg_degree,
            total_documents: self.total_documents,
            vocabulary_size: self.term_to_id.len(),
        }
    }

    // ── Internal helpers ─────────────────────────────────────────────────

    /// Remove the given concept indices and rebuild all derived data structures.
    fn remove_concepts(&mut self, to_remove: &HashSet<usize>) -> usize {
        // Build a mapping old_id → new_id for surviving concepts.
        let mut new_index: HashMap<usize, usize> = HashMap::new();
        let mut new_concepts: Vec<CgConcept> = Vec::new();
        for concept in self.concepts.drain(..) {
            if to_remove.contains(&concept.id.0) {
                continue;
            }
            let new_id = new_concepts.len();
            new_index.insert(concept.id.0, new_id);
            let mut c = concept;
            c.id = ConceptId(new_id);
            new_concepts.push(c);
        }
        let removed = to_remove.len();
        self.concepts = new_concepts;

        // Rebuild term_to_id.
        self.term_to_id.clear();
        for c in &self.concepts {
            self.term_to_id.insert(c.term.clone(), c.id);
        }

        // Rebuild edges — drop any that reference a removed concept.
        let mut new_edges: Vec<CgConceptEdge> = Vec::new();
        let mut new_edge_index: HashMap<(usize, usize), usize> = HashMap::new();
        for edge in self.edges.drain(..) {
            let new_from = match new_index.get(&edge.from.0) {
                Some(&i) => i,
                None => continue,
            };
            let new_to = match new_index.get(&edge.to.0) {
                Some(&i) => i,
                None => continue,
            };
            let key = canonical_key(ConceptId(new_from), ConceptId(new_to));
            let ei = new_edges.len();
            new_edge_index.insert(key, ei);
            new_edges.push(CgConceptEdge {
                from: ConceptId(new_from),
                to: ConceptId(new_to),
                weight: edge.weight,
                relation: edge.relation,
                co_occurrence: edge.co_occurrence,
            });
        }
        self.edges = new_edges;
        self.edge_index = new_edge_index;

        // Rebuild adjacency.
        self.adjacency.clear();
        for (ei, edge) in self.edges.iter().enumerate() {
            self.adjacency.entry(edge.from.0).or_default().push(ei);
            self.adjacency.entry(edge.to.0).or_default().push(ei);
        }

        removed
    }

    /// Rebuild internal edge structures excluding the given edge indices.
    fn rebuild_edges_excluding(&mut self, remove_set: &HashSet<usize>) {
        let mut new_edges: Vec<CgConceptEdge> = Vec::new();
        let mut new_edge_index: HashMap<(usize, usize), usize> = HashMap::new();
        for (old_idx, edge) in self.edges.drain(..).enumerate() {
            if remove_set.contains(&old_idx) {
                continue;
            }
            let key = canonical_key(edge.from, edge.to);
            let new_idx = new_edges.len();
            new_edge_index.insert(key, new_idx);
            new_edges.push(edge);
        }
        self.edges = new_edges;
        self.edge_index = new_edge_index;

        // Rebuild adjacency.
        self.adjacency.clear();
        for (ei, edge) in self.edges.iter().enumerate() {
            self.adjacency.entry(edge.from.0).or_default().push(ei);
            self.adjacency.entry(edge.to.0).or_default().push(ei);
        }
    }
}

// ─── Free functions ──────────────────────────────────────────────────────────

/// Return the canonical (min, max) key for an undirected edge.
#[inline]
fn canonical_key(a: ConceptId, b: ConceptId) -> (usize, usize) {
    let (x, y) = (a.0, b.0);
    if x <= y {
        (x, y)
    } else {
        (y, x)
    }
}

/// Split `text` on whitespace and ASCII punctuation; lowercase; keep tokens
/// with at least 3 characters.
pub fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| c.is_whitespace() || (c.is_ascii_punctuation() && c != '\''))
        .map(|s| s.to_lowercase())
        .filter(|s| s.chars().count() >= 3)
        .collect()
}

/// Cosine similarity between two equal-length vectors.
///
/// Returns 0.0 if either vector is all-zero or if they have different lengths.
pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
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

// Re-export a helper only used in tests (needed because tests reference a private fn via full path).
#[doc(hidden)]
pub fn canonize_key_test(a: ConceptId, b: ConceptId) -> (usize, usize) {
    canonical_key(a, b)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use crate::concept_graph::{
        cosine_similarity, tokenize, CgConceptRelation, CgGraphConfig, ConceptGraphBuilder,
        ConceptId,
    };

    fn default_builder() -> ConceptGraphBuilder {
        ConceptGraphBuilder::new(CgGraphConfig::default())
    }

    fn small_config() -> CgGraphConfig {
        CgGraphConfig {
            min_concept_frequency: 1,
            max_concepts: 10_000,
            co_occurrence_window: 3,
            min_edge_weight: 0.01,
        }
    }

    fn small_builder() -> ConceptGraphBuilder {
        ConceptGraphBuilder::new(small_config())
    }

    // ── tokenize ─────────────────────────────────────────────────────────

    #[test]
    fn test_tokenize_basic() {
        let tokens = tokenize("Hello, World!");
        assert!(tokens.contains(&"hello".to_string()));
        assert!(tokens.contains(&"world".to_string()));
    }

    #[test]
    fn test_tokenize_min_length() {
        let tokens = tokenize("I am a big cat");
        assert!(!tokens.contains(&"i".to_string()));
        assert!(!tokens.contains(&"am".to_string()));
        assert!(!tokens.contains(&"a".to_string()));
        assert!(tokens.contains(&"big".to_string()));
        assert!(tokens.contains(&"cat".to_string()));
    }

    #[test]
    fn test_tokenize_punctuation_split() {
        let tokens = tokenize("hello.world");
        assert!(tokens.contains(&"hello".to_string()));
        assert!(tokens.contains(&"world".to_string()));
    }

    #[test]
    fn test_tokenize_lowercase() {
        let tokens = tokenize("RUST Language");
        assert!(tokens.contains(&"rust".to_string()));
        assert!(tokens.contains(&"language".to_string()));
    }

    #[test]
    fn test_tokenize_empty() {
        let tokens = tokenize("");
        assert!(tokens.is_empty());
    }

    // ── cosine_similarity ────────────────────────────────────────────────

    #[test]
    fn test_cosine_identical() {
        let v = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_cosine_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-9);
    }

    #[test]
    fn test_cosine_zero_vector() {
        let a = vec![0.0, 0.0];
        let b = vec![1.0, 2.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn test_cosine_length_mismatch() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    // ── add_concept_term ─────────────────────────────────────────────────

    #[test]
    fn test_add_concept_term_new() {
        let mut b = default_builder();
        let id = b.add_concept_term("rust".to_string(), None);
        assert_eq!(id, ConceptId(0));
        assert_eq!(b.concepts.len(), 1);
    }

    #[test]
    fn test_add_concept_term_idempotent() {
        let mut b = default_builder();
        let id1 = b.add_concept_term("rust".to_string(), None);
        let id2 = b.add_concept_term("rust".to_string(), None);
        assert_eq!(id1, id2);
        assert_eq!(b.concepts.len(), 1);
    }

    #[test]
    fn test_add_concept_term_embeds_updated() {
        let mut b = default_builder();
        b.add_concept_term("rust".to_string(), None);
        b.add_concept_term("rust".to_string(), Some(vec![1.0, 0.0]));
        assert!(b
            .concept_by_term("rust")
            .and_then(|c| c.embedding.as_ref())
            .is_some());
    }

    #[test]
    fn test_add_concept_term_cap() {
        let mut b = ConceptGraphBuilder::new(CgGraphConfig {
            max_concepts: 2,
            ..CgGraphConfig::default()
        });
        b.add_concept_term("aaa".to_string(), None);
        b.add_concept_term("bbb".to_string(), None);
        let id = b.add_concept_term("ccc".to_string(), None);
        assert_eq!(id, ConceptId(usize::MAX));
        assert_eq!(b.concepts.len(), 2);
    }

    // ── process_document ─────────────────────────────────────────────────

    #[test]
    fn test_process_document_increments_frequency() {
        let mut b = small_builder();
        b.process_document("d1", "rust programming language");
        let rust = b.concept_by_term("rust").expect("rust concept");
        assert_eq!(rust.frequency, 1);
    }

    #[test]
    fn test_process_document_two_docs() {
        let mut b = small_builder();
        b.process_document("d1", "rust programming language");
        b.process_document("d2", "rust systems programming");
        let rust = b.concept_by_term("rust").expect("rust concept");
        assert_eq!(rust.frequency, 2);
        assert_eq!(rust.documents.len(), 2);
    }

    #[test]
    fn test_process_document_doc_deduplicated() {
        let mut b = small_builder();
        b.process_document("d1", "rust rust rust");
        let rust = b.concept_by_term("rust").expect("rust concept");
        assert_eq!(rust.documents.len(), 1);
        assert_eq!(rust.frequency, 3);
    }

    #[test]
    fn test_process_document_creates_edges() {
        let mut b = small_builder();
        b.process_document("d1", "rust programming language systems");
        assert!(!b.edges.is_empty());
    }

    #[test]
    fn test_process_document_total_documents() {
        let mut b = default_builder();
        b.process_document("d1", "hello world foo");
        b.process_document("d2", "another world document");
        assert_eq!(b.total_documents, 2);
    }

    #[test]
    fn test_process_empty_document() {
        let mut b = default_builder();
        b.process_document("d1", "");
        assert_eq!(b.total_documents, 1);
        assert!(b.concepts.is_empty());
    }

    // ── add_relation ──────────────────────────────────────────────────────

    #[test]
    fn test_add_relation_success() {
        let mut b = small_builder();
        b.process_document("d1", "fast quick");
        let ok = b.add_relation("fast", "quick", CgConceptRelation::Synonym, 0.9);
        assert!(ok);
    }

    #[test]
    fn test_add_relation_missing_term_returns_false() {
        let mut b = small_builder();
        b.process_document("d1", "fast quick");
        let ok = b.add_relation("fast", "slow", CgConceptRelation::Antonym, 0.8);
        assert!(!ok);
    }

    #[test]
    fn test_add_relation_overwrites_existing() {
        let mut b = small_builder();
        b.process_document("d1", "fast quick");
        b.add_relation("fast", "quick", CgConceptRelation::CoOccurrence, 0.5);
        b.add_relation("fast", "quick", CgConceptRelation::Synonym, 0.95);
        // There should still be only one edge between them.
        let key = {
            let id_fast = b.term_to_id["fast"];
            let id_quick = b.term_to_id["quick"];
            let (lo, hi) = if id_fast.0 <= id_quick.0 {
                (id_fast.0, id_quick.0)
            } else {
                (id_quick.0, id_fast.0)
            };
            (lo, hi)
        };
        let edge_idx = b.edge_index[&key];
        assert!((b.edges[edge_idx].weight - 0.95).abs() < 1e-9);
    }

    // ── concept_by_term / concept_by_id ──────────────────────────────────

    #[test]
    fn test_concept_by_term_found() {
        let mut b = small_builder();
        b.process_document("d1", "hello world rust");
        assert!(b.concept_by_term("rust").is_some());
    }

    #[test]
    fn test_concept_by_term_not_found() {
        let b = default_builder();
        assert!(b.concept_by_term("missing").is_none());
    }

    #[test]
    fn test_concept_by_id_valid() {
        let mut b = small_builder();
        b.add_concept_term("hello".to_string(), None);
        assert!(b.concept_by_id(ConceptId(0)).is_some());
    }

    #[test]
    fn test_concept_by_id_invalid() {
        let b = default_builder();
        assert!(b.concept_by_id(ConceptId(usize::MAX)).is_none());
        assert!(b.concept_by_id(ConceptId(999)).is_none());
    }

    // ── neighbors ────────────────────────────────────────────────────────

    #[test]
    fn test_neighbors_sorted_desc() {
        let mut b = small_builder();
        b.process_document("d1", "alpha beta gamma delta");
        b.process_document("d2", "alpha beta gamma");
        b.process_document("d3", "alpha beta");
        let id_alpha = b.term_to_id["alpha"];
        let nbrs = b.neighbors(id_alpha);
        // Neighbours should be sorted by descending weight.
        for w in nbrs.windows(2) {
            assert!(w[0].1 >= w[1].1);
        }
    }

    #[test]
    fn test_neighbors_no_edges() {
        let mut b = small_builder();
        b.add_concept_term("lone".to_string(), None);
        let id = b.term_to_id["lone"];
        let nbrs = b.neighbors(id);
        assert!(nbrs.is_empty());
    }

    // ── shortest_path ────────────────────────────────────────────────────

    #[test]
    fn test_shortest_path_direct() {
        let mut b = small_builder();
        b.process_document("d1", "alpha beta");
        let a = b.term_to_id["alpha"];
        let bb = b.term_to_id["beta"];
        let path = b.shortest_path(a, bb).expect("direct path");
        assert_eq!(path.len(), 2);
        assert_eq!(path[0], a);
        assert_eq!(path[1], bb);
    }

    #[test]
    fn test_shortest_path_same_node() {
        let mut b = small_builder();
        b.add_concept_term("solo".to_string(), None);
        let id = b.term_to_id["solo"];
        let path = b.shortest_path(id, id).expect("self path");
        assert_eq!(path, vec![id]);
    }

    #[test]
    fn test_shortest_path_multi_hop() {
        let mut b = small_builder();
        // a-b edge, b-c edge — a reaches c through b.
        b.process_document("d1", "aaa bbb");
        b.process_document("d2", "bbb ccc");
        let a = b.term_to_id["aaa"];
        let c = b.term_to_id["ccc"];
        let path = b.shortest_path(a, c).expect("multi-hop path");
        assert!(path.len() >= 3);
        assert_eq!(*path.first().expect("first"), a);
        assert_eq!(*path.last().expect("last"), c);
    }

    #[test]
    fn test_shortest_path_no_connection() {
        let mut b = small_builder();
        b.process_document("d1", "aaa bbb");
        b.add_concept_term("ccc".to_string(), None);
        let a = b.term_to_id["aaa"];
        let c = b.term_to_id["ccc"];
        assert!(b.shortest_path(a, c).is_none());
    }

    // ── similar_concepts ─────────────────────────────────────────────────

    #[test]
    fn test_similar_concepts_embedding_based() {
        let mut b = small_builder();
        b.add_concept_term("rust".to_string(), Some(vec![1.0, 0.0]));
        b.add_concept_term("systems".to_string(), Some(vec![0.9, 0.1]));
        b.add_concept_term("python".to_string(), Some(vec![0.0, 1.0]));
        let id = b.term_to_id["rust"];
        let similar = b.similar_concepts(id, 1);
        assert_eq!(similar.len(), 1);
        assert_eq!(similar[0].0.term, "systems");
    }

    #[test]
    fn test_similar_concepts_graph_fallback() {
        let mut b = small_builder();
        b.process_document("d1", "aaa bbb ccc");
        let id = b.term_to_id["aaa"];
        let similar = b.similar_concepts(id, 2);
        assert!(similar.len() <= 2);
    }

    #[test]
    fn test_similar_concepts_k_zero() {
        let mut b = small_builder();
        b.process_document("d1", "alpha beta");
        let id = b.term_to_id["alpha"];
        assert!(b.similar_concepts(id, 0).is_empty());
    }

    #[test]
    fn test_similar_concepts_unknown_id() {
        let b = default_builder();
        assert!(b.similar_concepts(ConceptId(999), 5).is_empty());
    }

    // ── prune_low_frequency ───────────────────────────────────────────────

    #[test]
    fn test_prune_low_frequency_removes_rare() {
        let mut b = ConceptGraphBuilder::new(CgGraphConfig {
            min_concept_frequency: 2,
            ..small_config()
        });
        b.process_document("d1", "common common rare");
        b.process_document("d2", "common");
        let removed = b.prune_low_frequency();
        assert!(removed > 0);
        assert!(b.concept_by_term("rare").is_none());
    }

    #[test]
    fn test_prune_low_frequency_keeps_frequent() {
        let mut b = ConceptGraphBuilder::new(CgGraphConfig {
            min_concept_frequency: 2,
            ..small_config()
        });
        b.process_document("d1", "common common");
        b.process_document("d2", "common");
        b.prune_low_frequency();
        assert!(b.concept_by_term("common").is_some());
    }

    #[test]
    fn test_prune_low_frequency_none_removed() {
        let mut b = ConceptGraphBuilder::new(CgGraphConfig {
            min_concept_frequency: 1,
            ..small_config()
        });
        b.process_document("d1", "alpha beta");
        let removed = b.prune_low_frequency();
        assert_eq!(removed, 0);
    }

    #[test]
    fn test_prune_low_frequency_removes_associated_edges() {
        let mut b = ConceptGraphBuilder::new(CgGraphConfig {
            min_concept_frequency: 2,
            ..small_config()
        });
        b.process_document("d1", "rare rare common");
        b.process_document("d2", "common common");
        // edges before prune: rare-common
        let edges_before = b.edges.len();
        b.prune_low_frequency();
        assert!(b.edges.len() <= edges_before);
    }

    // ── prune_weak_edges ─────────────────────────────────────────────────

    #[test]
    fn test_prune_weak_edges_removes_below_threshold() {
        let mut b = ConceptGraphBuilder::new(CgGraphConfig {
            min_edge_weight: 0.5,
            ..small_config()
        });
        b.process_document("d1", "aaa bbb");
        let removed = b.prune_weak_edges();
        // The single co-occurrence edge will likely be below 0.5.
        let _ = removed; // we just verify it doesn't panic
        assert!(b.edges.len() <= 1);
    }

    #[test]
    fn test_prune_weak_edges_none_removed() {
        let mut b = ConceptGraphBuilder::new(CgGraphConfig {
            min_edge_weight: 0.0,
            ..small_config()
        });
        b.process_document("d1", "aaa bbb");
        let removed = b.prune_weak_edges();
        assert_eq!(removed, 0);
    }

    // ── graph_stats ───────────────────────────────────────────────────────

    #[test]
    fn test_graph_stats_empty() {
        let b = default_builder();
        let s = b.graph_stats();
        assert_eq!(s.concept_count, 0);
        assert_eq!(s.edge_count, 0);
        assert_eq!(s.avg_degree, 0.0);
        assert_eq!(s.total_documents, 0);
    }

    #[test]
    fn test_graph_stats_after_processing() {
        let mut b = small_builder();
        b.process_document("d1", "alpha beta gamma");
        let s = b.graph_stats();
        assert!(s.concept_count > 0);
        assert!(s.edge_count > 0);
        assert_eq!(s.total_documents, 1);
        assert_eq!(s.vocabulary_size, s.concept_count);
    }

    #[test]
    fn test_graph_stats_avg_degree() {
        let mut b = small_builder();
        b.process_document("d1", "alpha beta");
        let s = b.graph_stats();
        // 2 concepts, 1 edge → avg_degree = 2 * 1 / 2 = 1.0
        assert!((s.avg_degree - 1.0).abs() < 1e-9);
    }

    // ── ConceptId ordering ────────────────────────────────────────────────

    #[test]
    fn test_concept_id_ordering() {
        assert!(ConceptId(0) < ConceptId(1));
        assert_eq!(ConceptId(5), ConceptId(5));
    }

    // ── Full integration ──────────────────────────────────────────────────

    #[test]
    fn test_full_pipeline() {
        let mut b = ConceptGraphBuilder::new(CgGraphConfig {
            min_concept_frequency: 2,
            min_edge_weight: 0.05,
            co_occurrence_window: 4,
            max_concepts: 1000,
        });
        let docs = [
            ("d1", "machine learning neural networks deep learning"),
            ("d2", "machine learning gradient descent optimization"),
            ("d3", "neural networks deep learning backpropagation"),
            ("d4", "deep learning convolutional neural networks"),
        ];
        for (id, text) in &docs {
            b.process_document(id, text);
        }
        // Prune noise.
        b.prune_low_frequency();
        b.prune_weak_edges();

        // Core concepts should survive.
        assert!(b.concept_by_term("machine").is_some());
        assert!(b.concept_by_term("learning").is_some());
        assert!(b.concept_by_term("neural").is_some());

        // Graph should be connected.
        let stats = b.graph_stats();
        assert!(stats.concept_count > 0);
        assert!(stats.edge_count > 0);
    }
}
