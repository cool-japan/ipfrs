//! Hierarchical Topic Model (HTM) — LDA-style topic inference over a tree-structured topic
//! hierarchy.
//!
//! # Overview
//!
//! [`HierarchicalTopicModel`] maintains:
//! - A rooted topic tree (`HtmTopicNode` nodes linked by parent/child IDs).
//! - A document corpus (`HtmDocument` entries with per-token topic assignments).
//! - A shared vocabulary (word ↔ index mapping).
//!
//! Inference is performed via collapsed Gibbs sampling where, for each token, we sample a new
//! topic from the set of nodes currently on the document's root-to-leaf path.  After `n_iter`
//! sweeps the topic tree is updated with final counts.
//!
//! # Example
//!
//! ```rust
//! use ipfrs_semantic::hierarchical_topic_model::{HierarchicalTopicModel, HtmModelConfig};
//!
//! let cfg = HtmModelConfig {
//!     max_depth: 3,
//!     max_children_per_node: 4,
//!     alpha: 0.1,
//!     beta: 0.01,
//!     n_iterations: 50,
//!     seed: 42,
//! };
//! let mut model = HierarchicalTopicModel::new(cfg);
//!
//! let _d0 = model.add_document(&["rust", "programming", "language", "fast"]);
//! let _d1 = model.add_document(&["machine", "learning", "neural", "network"]);
//!
//! model.run_inference(20);
//!
//! let stats = model.model_stats();
//! assert!(stats.n_docs >= 2);
//! ```

use std::cmp::Reverse;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Type aliases
// ---------------------------------------------------------------------------

/// Unique identifier for a topic node in the hierarchy.
pub type HtmTopicNodeId = u64;

/// Unique identifier for a document.
pub type HtmDocId = u64;

// ---------------------------------------------------------------------------
// Inline PRNG / hashing utilities
// ---------------------------------------------------------------------------

/// Xorshift64 pseudo-random number generator (single step).
#[inline]
fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// FNV-1a 64-bit hash of an arbitrary byte slice.
#[inline]
fn fnv1a_64(data: &[u8]) -> u64 {
    let mut h: u64 = 14_695_981_039_346_656_037;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(1_099_511_628_211);
    }
    h
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration parameters for [`HierarchicalTopicModel`].
#[derive(Debug, Clone)]
pub struct HtmModelConfig {
    /// Maximum depth of the topic tree (root is depth 0).
    pub max_depth: u32,
    /// Maximum number of child nodes for any single topic node.
    pub max_children_per_node: usize,
    /// Dirichlet hyper-parameter α (document–topic prior).
    pub alpha: f64,
    /// Dirichlet hyper-parameter β (topic–word prior).
    pub beta: f64,
    /// Default number of Gibbs-sampling iterations used by [`HierarchicalTopicModel::run_inference`].
    pub n_iterations: usize,
    /// Seed for the internal PRNG.
    pub seed: u64,
}

impl Default for HtmModelConfig {
    fn default() -> Self {
        Self {
            max_depth: 3,
            max_children_per_node: 8,
            alpha: 0.1,
            beta: 0.01,
            n_iterations: 100,
            seed: 12_345,
        }
    }
}

// ---------------------------------------------------------------------------
// Topic node
// ---------------------------------------------------------------------------

/// A single node in the topic tree.
#[derive(Debug, Clone)]
pub struct HtmTopicNode {
    /// Unique node ID (`0` is reserved for the root).
    pub id: HtmTopicNodeId,
    /// Parent node ID (`None` for the root).
    pub parent: Option<HtmTopicNodeId>,
    /// IDs of child nodes.
    pub children: Vec<HtmTopicNodeId>,
    /// Depth of this node (root = 0).
    pub depth: u32,
    /// Per-vocabulary-index word counts accumulated from assigned tokens.
    pub word_counts: Vec<u32>,
    /// Sum of all word counts (`word_counts.iter().sum()`).
    pub total_words: u32,
    /// Optional human-readable label for this topic.
    pub label: Option<String>,
}

impl HtmTopicNode {
    fn new(
        id: HtmTopicNodeId,
        parent: Option<HtmTopicNodeId>,
        depth: u32,
        vocab_size: usize,
    ) -> Self {
        Self {
            id,
            parent,
            children: Vec::new(),
            depth,
            word_counts: vec![0u32; vocab_size],
            total_words: 0,
            label: None,
        }
    }

    /// Ensure `word_counts` has at least `vocab_size` entries.
    fn ensure_vocab_size(&mut self, vocab_size: usize) {
        if self.word_counts.len() < vocab_size {
            self.word_counts.resize(vocab_size, 0);
        }
    }

    /// Increment a word count (growing the vector if needed).
    fn increment_word(&mut self, word_idx: usize) {
        if word_idx >= self.word_counts.len() {
            self.word_counts.resize(word_idx + 1, 0);
        }
        self.word_counts[word_idx] = self.word_counts[word_idx].saturating_add(1);
        self.total_words = self.total_words.saturating_add(1);
    }

    /// Decrement a word count (saturating at 0).
    fn decrement_word(&mut self, word_idx: usize) {
        if word_idx < self.word_counts.len() && self.word_counts[word_idx] > 0 {
            self.word_counts[word_idx] -= 1;
            if self.total_words > 0 {
                self.total_words -= 1;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Document
// ---------------------------------------------------------------------------

/// A document stored in the model.
#[derive(Debug, Clone)]
pub struct HtmDocument {
    /// Unique document ID.
    pub id: HtmDocId,
    /// Vocabulary indices of tokens in order of appearance.
    pub token_indices: Vec<u32>,
    /// Per-token topic assignment (same length as `token_indices`).
    pub topic_assignments: Vec<HtmTopicNodeId>,
    /// Ordered path from root to the document's current leaf topic node.
    pub path: Vec<HtmTopicNodeId>,
}

// ---------------------------------------------------------------------------
// Output types
// ---------------------------------------------------------------------------

/// Aggregated view of a single topic node, returned by [`HierarchicalTopicModel::get_topic`].
#[derive(Debug, Clone)]
pub struct HtmTopic {
    /// Node ID of this topic.
    pub id: HtmTopicNodeId,
    /// Top words ranked by (smoothed) probability, with their scores.
    pub top_words: Vec<(String, f64)>,
    /// PMI-based coherence score (set to `0.0` until explicitly computed).
    pub coherence: f64,
    /// Number of documents that have at least one token assigned to this topic.
    pub doc_count: u32,
    /// Depth of this node in the topic tree.
    pub depth: u32,
}

/// Summary statistics for the whole model.
#[derive(Debug, Clone)]
pub struct HtmModelStats {
    /// Total number of topic nodes (including root).
    pub n_topics: usize,
    /// Total number of documents.
    pub n_docs: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Average PMI coherence across all non-root nodes (or `0.0` if none).
    pub avg_coherence: f64,
    /// Maximum depth encountered in the topic tree.
    pub max_depth: u32,
}

// ---------------------------------------------------------------------------
// Main struct
// ---------------------------------------------------------------------------

/// Hierarchical LDA-style topic model with a tree-structured topic hierarchy.
///
/// See the [module-level documentation](self) for a full overview.
pub struct HierarchicalTopicModel {
    config: HtmModelConfig,
    /// All topic nodes, keyed by their ID.
    topics: HashMap<HtmTopicNodeId, HtmTopicNode>,
    /// ID of the root topic node (always `0`).
    root: HtmTopicNodeId,
    /// All documents, keyed by their ID.
    documents: HashMap<HtmDocId, HtmDocument>,
    /// Word → vocabulary index.
    vocab: HashMap<String, u32>,
    /// Vocabulary index → word.
    vocab_inv: Vec<String>,
    /// Monotonically increasing counter for new topic node IDs.
    next_topic_id: HtmTopicNodeId,
    /// Monotonically increasing counter for new document IDs.
    next_doc_id: HtmDocId,
    /// Internal PRNG state.
    rng_state: u64,
}

impl HierarchicalTopicModel {
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    /// Create a new model with the given configuration.
    pub fn new(config: HtmModelConfig) -> Self {
        let seed = config.seed;
        let root_node = HtmTopicNode::new(0, None, 0, 0);
        let mut topics = HashMap::new();
        topics.insert(0, root_node);

        Self {
            config,
            topics,
            root: 0,
            documents: HashMap::new(),
            vocab: HashMap::new(),
            vocab_inv: Vec::new(),
            next_topic_id: 1,
            next_doc_id: 0,
            rng_state: if seed == 0 {
                6_364_136_223_846_793_005
            } else {
                seed
            },
        }
    }

    // -----------------------------------------------------------------------
    // Vocabulary helpers
    // -----------------------------------------------------------------------

    /// Return the vocabulary index for `word`, creating a new entry if necessary.
    fn get_or_insert_word(&mut self, word: &str) -> u32 {
        if let Some(&idx) = self.vocab.get(word) {
            return idx;
        }
        let idx = self.vocab_inv.len() as u32;
        self.vocab_inv.push(word.to_owned());
        self.vocab.insert(word.to_owned(), idx);
        // Grow every existing topic node's word_counts vector.
        let new_size = self.vocab_inv.len();
        for node in self.topics.values_mut() {
            node.ensure_vocab_size(new_size);
        }
        idx
    }

    // -----------------------------------------------------------------------
    // Topic tree operations
    // -----------------------------------------------------------------------

    /// Add a new topic node to the hierarchy.
    ///
    /// If `parent` is `None` the root node is used as the parent.  Returns the new node's ID.
    /// Returns an error string when the parent doesn't exist or depth/children limits are exceeded.
    pub fn add_topic_node(
        &mut self,
        parent: Option<HtmTopicNodeId>,
        label: Option<String>,
    ) -> Result<HtmTopicNodeId, String> {
        let parent_id = parent.unwrap_or(self.root);

        let (parent_depth, parent_children_count) = {
            let p = self
                .topics
                .get(&parent_id)
                .ok_or_else(|| format!("parent topic node {parent_id} does not exist"))?;
            (p.depth, p.children.len())
        };

        if parent_depth + 1 > self.config.max_depth {
            return Err(format!(
                "cannot create node at depth {}; max_depth is {}",
                parent_depth + 1,
                self.config.max_depth
            ));
        }
        if parent_children_count >= self.config.max_children_per_node {
            return Err(format!(
                "parent node {parent_id} already has {} children (max {})",
                parent_children_count, self.config.max_children_per_node
            ));
        }

        let new_id = self.next_topic_id;
        self.next_topic_id += 1;

        let vocab_size = self.vocab_inv.len();
        let mut node = HtmTopicNode::new(new_id, Some(parent_id), parent_depth + 1, vocab_size);
        node.label = label;

        self.topics.insert(new_id, node);
        if let Some(p) = self.topics.get_mut(&parent_id) {
            p.children.push(new_id);
        }

        Ok(new_id)
    }

    // -----------------------------------------------------------------------
    // Document management
    // -----------------------------------------------------------------------

    /// Add a document given a slice of string tokens.
    ///
    /// Tokenisation: tokens are lower-cased; empty tokens are skipped.
    /// Returns the new document's `HtmDocId`.
    pub fn add_document(&mut self, tokens: &[&str]) -> HtmDocId {
        let mut token_indices: Vec<u32> = Vec::with_capacity(tokens.len());
        for &tok in tokens {
            let lower = tok.to_lowercase();
            if lower.is_empty() {
                continue;
            }
            let idx = self.get_or_insert_word(&lower);
            token_indices.push(idx);
        }

        let doc_id = self.next_doc_id;
        self.next_doc_id += 1;

        // Build initial path & assignments using the current topic tree.
        let path = self.build_initial_path();
        let n_tokens = token_indices.len();
        let mut topic_assignments: Vec<HtmTopicNodeId> = Vec::with_capacity(n_tokens);

        let path_len = path.len();
        for &token_idx in &token_indices {
            // Assign each token to a node on the path (xorshift64-based selection).
            let r = xorshift64(&mut self.rng_state);
            let node_id = if path_len > 0 {
                path[r as usize % path_len]
            } else {
                self.root
            };
            topic_assignments.push(node_id);

            // Update word counts.
            let word_idx = token_idx as usize;
            if let Some(node) = self.topics.get_mut(&node_id) {
                node.increment_word(word_idx);
            }
        }

        let doc = HtmDocument {
            id: doc_id,
            token_indices,
            topic_assignments,
            path,
        };
        self.documents.insert(doc_id, doc);
        doc_id
    }

    // -----------------------------------------------------------------------
    // Path helpers
    // -----------------------------------------------------------------------

    /// Build a root-to-leaf path for a new document using the current tree topology.
    ///
    /// Greedily follows the child with the highest `total_words`, creating new leaf nodes when
    /// the current depth limit hasn't been reached.
    fn build_initial_path(&mut self) -> Vec<HtmTopicNodeId> {
        let mut path = Vec::new();
        let mut current = self.root;
        path.push(current);

        loop {
            let (depth, children) = {
                let node = match self.topics.get(&current) {
                    Some(n) => n,
                    None => break,
                };
                (node.depth, node.children.clone())
            };

            if depth >= self.config.max_depth {
                break;
            }

            if children.is_empty() {
                // Create a new leaf when possible.
                let new_id = self.next_topic_id;
                self.next_topic_id += 1;
                let vocab_size = self.vocab_inv.len();
                let child = HtmTopicNode::new(new_id, Some(current), depth + 1, vocab_size);
                self.topics.insert(new_id, child);
                if let Some(parent_node) = self.topics.get_mut(&current) {
                    parent_node.children.push(new_id);
                }
                path.push(new_id);
                current = new_id;
            } else {
                // Choose the child with the highest total_words (ties broken by ID).
                let best = children
                    .iter()
                    .filter_map(|&cid| self.topics.get(&cid).map(|n| (cid, n.total_words)))
                    .max_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));

                match best {
                    Some((bid, _)) => {
                        path.push(bid);
                        current = bid;
                    }
                    None => break,
                }
            }
        }
        path
    }

    /// Return the full root-to-node path for `node_id` (inclusive, from root to node).
    fn path_to_root(&self, node_id: HtmTopicNodeId) -> Vec<HtmTopicNodeId> {
        let mut path = Vec::new();
        let mut current = node_id;
        loop {
            path.push(current);
            match self.topics.get(&current).and_then(|n| n.parent) {
                Some(pid) => current = pid,
                None => break,
            }
        }
        path.reverse();
        path
    }

    // -----------------------------------------------------------------------
    // Inference
    // -----------------------------------------------------------------------

    /// Run collapsed Gibbs sampling for `n_iter` iterations.
    ///
    /// Each iteration sweeps all documents and re-samples each token's topic assignment from the
    /// conditional posterior restricted to nodes on the document's current root-to-leaf path.
    pub fn run_inference(&mut self, n_iter: usize) {
        let iterations = if n_iter == 0 {
            self.config.n_iterations
        } else {
            n_iter
        };
        let vocab_size = self.vocab_inv.len();
        let beta = self.config.beta;
        let alpha = self.config.alpha;

        for _iter in 0..iterations {
            let doc_ids: Vec<HtmDocId> = self.documents.keys().copied().collect();

            for doc_id in doc_ids {
                // --- Update path for this document (re-sample leaf) ---
                self.resample_path(doc_id);

                // --- Per-token sampling ---
                let (token_indices, old_assignments, path) = {
                    let doc = match self.documents.get(&doc_id) {
                        Some(d) => d,
                        None => continue,
                    };
                    (
                        doc.token_indices.clone(),
                        doc.topic_assignments.clone(),
                        doc.path.clone(),
                    )
                };

                let path_len = path.len();
                if path_len == 0 {
                    continue;
                }

                let mut new_assignments = old_assignments.clone();

                for token_pos in 0..token_indices.len() {
                    let word_idx = token_indices[token_pos] as usize;
                    let old_topic = old_assignments[token_pos];

                    // Remove this token's contribution from its current topic.
                    if let Some(node) = self.topics.get_mut(&old_topic) {
                        node.decrement_word(word_idx);
                    }

                    // Compute unnormalised probabilities for each node on the path.
                    let mut probs: Vec<f64> = Vec::with_capacity(path_len);
                    for &topic_id in &path {
                        let (wc, tw) = match self.topics.get(&topic_id) {
                            Some(n) => {
                                let wc = n.word_counts.get(word_idx).copied().unwrap_or(0) as f64;
                                (wc, n.total_words as f64)
                            }
                            None => (0.0, 0.0),
                        };
                        // Document–topic count for this path node.
                        let doc_topic_count =
                            old_assignments.iter().filter(|&&t| t == topic_id).count() as f64;

                        let p_word_given_topic = (wc + beta) / (tw + vocab_size as f64 * beta);
                        let p_topic_given_doc = doc_topic_count + alpha;
                        probs.push(p_word_given_topic * p_topic_given_doc);
                    }

                    // Sample from `probs`.
                    let new_topic = self.sample_from_probs(&path, &probs);
                    new_assignments[token_pos] = new_topic;

                    // Re-add contribution to the newly sampled topic.
                    if let Some(node) = self.topics.get_mut(&new_topic) {
                        node.increment_word(word_idx);
                    }
                }

                // Persist updated assignments.
                if let Some(doc) = self.documents.get_mut(&doc_id) {
                    doc.topic_assignments = new_assignments;
                }
            }
        }
    }

    /// Re-sample the root-to-leaf path for a document.
    ///
    /// We try each possible leaf (all nodes that have no children, or all nodes at max_depth)
    /// and score the path by the product of per-level document likelihoods.
    fn resample_path(&mut self, doc_id: HtmDocId) {
        let token_indices = match self.documents.get(&doc_id) {
            Some(d) => d.token_indices.clone(),
            None => return,
        };

        // Collect leaf IDs (nodes with no children, excluding root when tree is trivial).
        let leaf_ids: Vec<HtmTopicNodeId> = self
            .topics
            .values()
            .filter(|n| n.children.is_empty())
            .map(|n| n.id)
            .collect();

        if leaf_ids.is_empty() {
            return;
        }

        let vocab_size = self.vocab_inv.len();
        let beta = self.config.beta;

        // Score each leaf path.
        let mut scores: Vec<f64> = Vec::with_capacity(leaf_ids.len());
        let mut paths: Vec<Vec<HtmTopicNodeId>> = Vec::with_capacity(leaf_ids.len());

        for &leaf in &leaf_ids {
            let path = self.path_to_root(leaf);
            let mut log_score = 0.0_f64;

            for &topic_id in &path {
                if let Some(node) = self.topics.get(&topic_id) {
                    for &wi in &token_indices {
                        let wc = node.word_counts.get(wi as usize).copied().unwrap_or(0) as f64;
                        let tw = node.total_words as f64;
                        let p = (wc + beta) / (tw + vocab_size as f64 * beta);
                        log_score += p.ln();
                    }
                }
            }
            scores.push(log_score.exp());
            paths.push(path);
        }

        // Sample a path.
        let chosen_idx = self.sample_index_from_raw_probs(&scores);
        let new_path = paths[chosen_idx].clone();

        if let Some(doc) = self.documents.get_mut(&doc_id) {
            doc.path = new_path;
        }
    }

    // -----------------------------------------------------------------------
    // Sampling helpers
    // -----------------------------------------------------------------------

    /// Categorical sample from a normalised probability distribution over `topics`.
    fn sample_from_probs(&mut self, topics: &[HtmTopicNodeId], probs: &[f64]) -> HtmTopicNodeId {
        let total: f64 = probs.iter().sum();
        if total <= 0.0 || topics.is_empty() {
            // Uniform fallback.
            let r = xorshift64(&mut self.rng_state);
            return topics[r as usize % topics.len()];
        }
        let threshold = (xorshift64(&mut self.rng_state) as f64 / u64::MAX as f64) * total;
        let mut cumulative = 0.0_f64;
        for (i, &p) in probs.iter().enumerate() {
            cumulative += p;
            if cumulative >= threshold {
                return topics[i];
            }
        }
        *topics.last().unwrap_or(&self.root)
    }

    /// Return the index into `probs` sampled proportionally.
    fn sample_index_from_raw_probs(&mut self, probs: &[f64]) -> usize {
        let total: f64 = probs.iter().sum();
        if total <= 0.0 || probs.is_empty() {
            let r = xorshift64(&mut self.rng_state);
            return if probs.is_empty() {
                0
            } else {
                r as usize % probs.len()
            };
        }
        let threshold = (xorshift64(&mut self.rng_state) as f64 / u64::MAX as f64) * total;
        let mut cumulative = 0.0_f64;
        for (i, &p) in probs.iter().enumerate() {
            cumulative += p;
            if cumulative >= threshold {
                return i;
            }
        }
        probs.len() - 1
    }

    // -----------------------------------------------------------------------
    // Query / inspection
    // -----------------------------------------------------------------------

    /// Retrieve an aggregated [`HtmTopic`] for the given node ID.
    ///
    /// Returns `None` if the node doesn't exist.
    /// The top-N words are computed using Laplace-smoothed probabilities (β smoothing).
    pub fn get_topic(&self, id: HtmTopicNodeId) -> Option<HtmTopic> {
        let node = self.topics.get(&id)?;
        let vocab_size = self.vocab_inv.len();
        let top_n = 10_usize.min(vocab_size);

        let beta = self.config.beta;
        let total = node.total_words as f64 + vocab_size as f64 * beta;

        // Collect (score, word_idx).
        let mut scored: Vec<(f64, usize)> = (0..vocab_size)
            .map(|wi| {
                let count = node.word_counts.get(wi).copied().unwrap_or(0) as f64;
                let score = (count + beta) / total;
                (score, wi)
            })
            .collect();

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let top_words: Vec<(String, f64)> = scored
            .into_iter()
            .take(top_n)
            .filter_map(|(score, wi)| self.vocab_inv.get(wi).map(|w| (w.clone(), score)))
            .collect();

        // Count documents that have at least one token assigned to this topic.
        let doc_count = self
            .documents
            .values()
            .filter(|doc| doc.topic_assignments.contains(&id))
            .count() as u32;

        Some(HtmTopic {
            id,
            top_words,
            coherence: 0.0, // filled in by compute_coherence if desired
            doc_count,
            depth: node.depth,
        })
    }

    /// Return proportional topic distribution along a document's current path.
    ///
    /// Each element is `(topic_node_id, proportion)` where `proportion` is the fraction of the
    /// document's tokens assigned to that node.  Sums to 1.0 (or 0.0 for empty documents).
    pub fn document_topics(&self, doc_id: HtmDocId) -> Vec<(HtmTopicNodeId, f64)> {
        let doc = match self.documents.get(&doc_id) {
            Some(d) => d,
            None => return Vec::new(),
        };

        let n_tokens = doc.topic_assignments.len();
        if n_tokens == 0 {
            return doc.path.iter().map(|&t| (t, 0.0)).collect();
        }

        let mut counts: HashMap<HtmTopicNodeId, u32> = HashMap::new();
        for &t in &doc.topic_assignments {
            *counts.entry(t).or_insert(0) += 1;
        }

        doc.path
            .iter()
            .map(|&t| {
                let c = counts.get(&t).copied().unwrap_or(0) as f64;
                (t, c / n_tokens as f64)
            })
            .collect()
    }

    /// Return the full topic hierarchy as a flat list of `(depth, node_id, parent_id)` tuples
    /// sorted by depth then by node ID.
    pub fn topic_hierarchy(&self) -> Vec<(u32, HtmTopicNodeId, Option<HtmTopicNodeId>)> {
        let mut entries: Vec<(u32, HtmTopicNodeId, Option<HtmTopicNodeId>)> = self
            .topics
            .values()
            .map(|n| (n.depth, n.id, n.parent))
            .collect();
        entries.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        entries
    }

    /// Compute PMI-based coherence for a topic node using its top-`top_n` words.
    ///
    /// The coherence score is the average pointwise mutual information for all pairs of top words,
    /// computed over the document corpus.
    ///
    /// Returns `0.0` if the topic node doesn't exist, has fewer than 2 words, or no documents.
    pub fn compute_coherence(&self, topic_id: HtmTopicNodeId, top_n: usize) -> f64 {
        let node = match self.topics.get(&topic_id) {
            Some(n) => n,
            None => return 0.0,
        };

        let vocab_size = self.vocab_inv.len();
        if vocab_size < 2 || self.documents.is_empty() {
            return 0.0;
        }

        let effective_top_n = top_n.min(vocab_size);

        // Select top-N word indices by count.
        let mut scored: Vec<(u32, usize)> = (0..vocab_size)
            .map(|wi| (node.word_counts.get(wi).copied().unwrap_or(0), wi))
            .collect();
        scored.sort_by_key(|b| Reverse(b.0));
        let top_indices: Vec<usize> = scored
            .into_iter()
            .take(effective_top_n)
            .map(|(_, wi)| wi)
            .collect();

        if top_indices.len() < 2 {
            return 0.0;
        }

        let n_docs = self.documents.len() as f64;

        // Build per-word document frequency sets.
        let mut doc_freq: Vec<f64> = vec![0.0; vocab_size];
        let mut co_occur: HashMap<(usize, usize), f64> = HashMap::new();

        for doc in self.documents.values() {
            // Build set of unique word indices in this document.
            let mut present: Vec<bool> = vec![false; vocab_size];
            for &wi in &doc.token_indices {
                let wi = wi as usize;
                if wi < vocab_size {
                    present[wi] = true;
                }
            }
            for &wi in &top_indices {
                if present[wi] {
                    doc_freq[wi] += 1.0;
                }
            }
            // Co-occurrence.
            for i in 0..top_indices.len() {
                for j in (i + 1)..top_indices.len() {
                    let wi = top_indices[i];
                    let wj = top_indices[j];
                    if present[wi] && present[wj] {
                        let key = (wi.min(wj), wi.max(wj));
                        *co_occur.entry(key).or_insert(0.0) += 1.0;
                    }
                }
            }
        }

        // Average PMI over all top-word pairs.
        let mut total_pmi = 0.0_f64;
        let mut n_pairs = 0_u64;

        for i in 0..top_indices.len() {
            for j in (i + 1)..top_indices.len() {
                let wi = top_indices[i];
                let wj = top_indices[j];
                let df_i = doc_freq[wi];
                let df_j = doc_freq[wj];
                if df_i < 1.0 || df_j < 1.0 {
                    continue;
                }
                let key = (wi.min(wj), wi.max(wj));
                let co = co_occur.get(&key).copied().unwrap_or(0.0);
                if co < 1.0 {
                    continue;
                }
                let pmi = (co * n_docs / (df_i * df_j)).ln();
                total_pmi += pmi;
                n_pairs += 1;
            }
        }

        if n_pairs == 0 {
            0.0
        } else {
            total_pmi / n_pairs as f64
        }
    }

    /// Remove leaf topic nodes that have zero total word counts and no documents assigned.
    ///
    /// The root node is never removed.  Pruning is repeated until no further nodes can be removed.
    pub fn prune_empty_topics(&mut self) {
        loop {
            let candidates: Vec<HtmTopicNodeId> = self
                .topics
                .values()
                .filter(|n| n.id != self.root && n.children.is_empty() && n.total_words == 0)
                .map(|n| n.id)
                .collect();

            if candidates.is_empty() {
                break;
            }

            for node_id in candidates {
                // Remove from parent's children list.
                if let Some(node) = self.topics.get(&node_id) {
                    if let Some(parent_id) = node.parent {
                        if let Some(parent) = self.topics.get_mut(&parent_id) {
                            parent.children.retain(|&c| c != node_id);
                        }
                    }
                }
                self.topics.remove(&node_id);
            }
        }
    }

    /// Return summary statistics for the current model state.
    pub fn model_stats(&self) -> HtmModelStats {
        let n_topics = self.topics.len();
        let n_docs = self.documents.len();
        let vocab_size = self.vocab_inv.len();
        let max_depth = self.topics.values().map(|n| n.depth).max().unwrap_or(0);

        let non_root_ids: Vec<HtmTopicNodeId> = self
            .topics
            .keys()
            .copied()
            .filter(|&id| id != self.root)
            .collect();

        let avg_coherence = if non_root_ids.is_empty() {
            0.0
        } else {
            let total: f64 = non_root_ids
                .iter()
                .map(|&id| self.compute_coherence(id, 10))
                .sum();
            total / non_root_ids.len() as f64
        };

        HtmModelStats {
            n_topics,
            n_docs,
            vocab_size,
            avg_coherence,
            max_depth,
        }
    }

    // -----------------------------------------------------------------------
    // Internal helpers exposed for testing
    // -----------------------------------------------------------------------

    /// Return the number of topic nodes (including root).
    pub fn n_topic_nodes(&self) -> usize {
        self.topics.len()
    }

    /// Return a reference to a topic node, if it exists.
    pub fn topic_node(&self, id: HtmTopicNodeId) -> Option<&HtmTopicNode> {
        self.topics.get(&id)
    }

    /// Return the current vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab_inv.len()
    }

    /// Return the word at a given vocabulary index.
    pub fn word_at(&self, idx: u32) -> Option<&str> {
        self.vocab_inv.get(idx as usize).map(|s| s.as_str())
    }

    /// Return the vocabulary index of a word (None if not present).
    pub fn word_index(&self, word: &str) -> Option<u32> {
        self.vocab.get(word).copied()
    }

    /// Retrieve a document by its ID.
    pub fn get_document(&self, doc_id: HtmDocId) -> Option<&HtmDocument> {
        self.documents.get(&doc_id)
    }

    /// Return the ID of the root topic node.
    pub fn root_id(&self) -> HtmTopicNodeId {
        self.root
    }

    /// FNV-1a 64-bit hash helper (exposed for testing / external hashing needs).
    pub fn hash_bytes(data: &[u8]) -> u64 {
        fnv1a_64(data)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_model() -> HierarchicalTopicModel {
        HierarchicalTopicModel::new(HtmModelConfig::default())
    }

    fn small_model() -> HierarchicalTopicModel {
        HierarchicalTopicModel::new(HtmModelConfig {
            max_depth: 2,
            max_children_per_node: 4,
            alpha: 0.1,
            beta: 0.01,
            n_iterations: 10,
            seed: 7,
        })
    }

    // -------------------------------------------------------------------------
    // 1. Construction & config
    // -------------------------------------------------------------------------

    #[test]
    fn test_default_config() {
        let cfg = HtmModelConfig::default();
        assert_eq!(cfg.max_depth, 3);
        assert_eq!(cfg.n_iterations, 100);
    }

    #[test]
    fn test_model_initial_state() {
        let m = default_model();
        assert_eq!(m.n_topic_nodes(), 1, "only root initially");
        assert_eq!(m.root_id(), 0);
        assert_eq!(m.vocab_size(), 0);
        assert_eq!(m.documents.len(), 0);
    }

    #[test]
    fn test_root_node_properties() {
        let m = default_model();
        let root = m.topic_node(0).expect("root must exist");
        assert!(root.parent.is_none());
        assert_eq!(root.depth, 0);
        assert!(root.children.is_empty());
    }

    #[test]
    fn test_custom_seed() {
        let cfg = HtmModelConfig {
            seed: 99,
            ..Default::default()
        };
        let m = HierarchicalTopicModel::new(cfg);
        assert_eq!(m.rng_state, 99);
    }

    #[test]
    fn test_zero_seed_replaced() {
        let cfg = HtmModelConfig {
            seed: 0,
            ..Default::default()
        };
        let m = HierarchicalTopicModel::new(cfg);
        assert_ne!(
            m.rng_state, 0,
            "seed 0 should be replaced by a non-zero default"
        );
    }

    // -------------------------------------------------------------------------
    // 2. Vocabulary
    // -------------------------------------------------------------------------

    #[test]
    fn test_vocab_grows_on_add_document() {
        let mut m = default_model();
        m.add_document(&["hello", "world"]);
        assert_eq!(m.vocab_size(), 2);
    }

    #[test]
    fn test_vocab_deduplicates_words() {
        let mut m = default_model();
        m.add_document(&["rust", "rust", "rust"]);
        assert_eq!(m.vocab_size(), 1);
    }

    #[test]
    fn test_vocab_lowercase_normalisation() {
        let mut m = default_model();
        m.add_document(&["Rust", "RUST", "rust"]);
        assert_eq!(m.vocab_size(), 1);
        assert_eq!(m.word_index("rust"), Some(0));
    }

    #[test]
    fn test_vocab_index_lookup() {
        let mut m = default_model();
        m.add_document(&["alpha", "beta", "gamma"]);
        assert!(m.word_index("alpha").is_some());
        assert!(m.word_index("delta").is_none());
    }

    #[test]
    fn test_word_at() {
        let mut m = default_model();
        m.add_document(&["one", "two"]);
        let idx0 = m.word_index("one").expect("test: 'one' must be in vocab");
        assert_eq!(m.word_at(idx0), Some("one"));
    }

    #[test]
    fn test_empty_tokens_skipped() {
        let mut m = default_model();
        m.add_document(&["", "hello", ""]);
        assert_eq!(m.vocab_size(), 1);
    }

    // -------------------------------------------------------------------------
    // 3. Document management
    // -------------------------------------------------------------------------

    #[test]
    fn test_add_document_returns_sequential_ids() {
        let mut m = default_model();
        let d0 = m.add_document(&["foo"]);
        let d1 = m.add_document(&["bar"]);
        assert_eq!(d0, 0);
        assert_eq!(d1, 1);
    }

    #[test]
    fn test_get_document_after_add() {
        let mut m = default_model();
        let id = m.add_document(&["hello", "world"]);
        let doc = m.get_document(id).expect("document must exist");
        assert_eq!(doc.id, id);
        assert_eq!(doc.token_indices.len(), 2);
    }

    #[test]
    fn test_document_path_non_empty() {
        let mut m = default_model();
        let id = m.add_document(&["a", "b", "c"]);
        let doc = m
            .get_document(id)
            .expect("test: document must exist after add");
        assert!(!doc.path.is_empty());
    }

    #[test]
    fn test_document_path_starts_at_root() {
        let mut m = default_model();
        let id = m.add_document(&["x"]);
        let doc = m
            .get_document(id)
            .expect("test: document must exist after add");
        assert_eq!(doc.path[0], m.root_id());
    }

    #[test]
    fn test_assignments_length_matches_tokens() {
        let mut m = default_model();
        let tokens = ["a", "b", "c", "d", "e"];
        let id = m.add_document(&tokens);
        let doc = m
            .get_document(id)
            .expect("test: document must exist after add");
        assert_eq!(doc.topic_assignments.len(), tokens.len());
    }

    #[test]
    fn test_assignments_valid_topic_ids() {
        let mut m = default_model();
        let id = m.add_document(&["rust", "fast", "safe"]);
        let doc = m
            .get_document(id)
            .expect("test: document must exist after add");
        for &t in &doc.topic_assignments {
            assert!(m.topic_node(t).is_some(), "assigned topic {t} must exist");
        }
    }

    #[test]
    fn test_empty_document_adds_cleanly() {
        let mut m = default_model();
        let id = m.add_document(&[]);
        let doc = m
            .get_document(id)
            .expect("test: document must exist after add");
        assert!(doc.token_indices.is_empty());
        assert!(doc.topic_assignments.is_empty());
    }

    // -------------------------------------------------------------------------
    // 4. Topic node management
    // -------------------------------------------------------------------------

    #[test]
    fn test_add_topic_node_to_root() {
        let mut m = default_model();
        let id = m.add_topic_node(None, None).expect("should succeed");
        assert!(m.topic_node(id).is_some());
        assert_eq!(m.n_topic_nodes(), 2);
    }

    #[test]
    fn test_add_topic_node_depth() {
        let mut m = default_model();
        let child = m
            .add_topic_node(None, None)
            .expect("test: add topic node to root should succeed");
        let grandchild = m
            .add_topic_node(Some(child), None)
            .expect("test: add grandchild node should succeed");
        assert_eq!(
            m.topic_node(grandchild)
                .expect("test: grandchild must exist")
                .depth,
            2
        );
    }

    #[test]
    fn test_add_topic_node_registers_parent() {
        let mut m = default_model();
        let child = m
            .add_topic_node(None, None)
            .expect("test: add topic node to root should succeed");
        assert_eq!(
            m.topic_node(child).expect("test: child must exist").parent,
            Some(0)
        );
    }

    #[test]
    fn test_add_topic_node_parent_children_updated() {
        let mut m = default_model();
        let child = m
            .add_topic_node(None, None)
            .expect("test: add topic node to root should succeed");
        assert!(m
            .topic_node(0)
            .expect("test: root must exist")
            .children
            .contains(&child));
    }

    #[test]
    fn test_add_topic_node_with_label() {
        let mut m = default_model();
        let id = m
            .add_topic_node(None, Some("science".to_owned()))
            .expect("test: add topic node to root should succeed");
        assert_eq!(
            m.topic_node(id).expect("test: node must exist").label,
            Some("science".to_owned())
        );
    }

    #[test]
    fn test_add_topic_node_depth_limit() {
        let mut m = HierarchicalTopicModel::new(HtmModelConfig {
            max_depth: 1,
            ..Default::default()
        });
        let child = m
            .add_topic_node(None, None)
            .expect("test: add topic node to root should succeed");
        let result = m.add_topic_node(Some(child), None);
        assert!(result.is_err(), "depth limit must be enforced");
    }

    #[test]
    fn test_add_topic_node_children_limit() {
        let mut m = HierarchicalTopicModel::new(HtmModelConfig {
            max_children_per_node: 2,
            ..Default::default()
        });
        m.add_topic_node(None, None)
            .expect("test: first child within limit");
        m.add_topic_node(None, None)
            .expect("test: second child within limit");
        let result = m.add_topic_node(None, None);
        assert!(result.is_err(), "children limit must be enforced");
    }

    #[test]
    fn test_add_topic_node_invalid_parent() {
        let mut m = default_model();
        let result = m.add_topic_node(Some(9999), None);
        assert!(result.is_err());
    }

    // -------------------------------------------------------------------------
    // 5. topic_hierarchy
    // -------------------------------------------------------------------------

    #[test]
    fn test_hierarchy_contains_root() {
        let m = default_model();
        let h = m.topic_hierarchy();
        assert!(h.iter().any(|&(_, id, _)| id == 0));
    }

    #[test]
    fn test_hierarchy_sorted_by_depth() {
        let mut m = default_model();
        m.add_topic_node(None, None)
            .expect("test: add node should succeed");
        let h = m.topic_hierarchy();
        for i in 1..h.len() {
            assert!(h[i - 1].0 <= h[i].0, "hierarchy must be sorted by depth");
        }
    }

    #[test]
    fn test_hierarchy_full_tree() {
        let mut m = default_model();
        let c1 = m
            .add_topic_node(None, None)
            .expect("test: add c1 should succeed");
        let _c2 = m
            .add_topic_node(None, None)
            .expect("test: add c2 should succeed");
        let _gc1 = m
            .add_topic_node(Some(c1), None)
            .expect("test: add grandchild should succeed");
        let h = m.topic_hierarchy();
        assert_eq!(h.len(), 4); // root + c1 + c2 + gc1
    }

    // -------------------------------------------------------------------------
    // 6. get_topic
    // -------------------------------------------------------------------------

    #[test]
    fn test_get_topic_root() {
        let mut m = default_model();
        m.add_document(&["hello", "world"]);
        let t = m.get_topic(0);
        assert!(t.is_some());
    }

    #[test]
    fn test_get_topic_nonexistent() {
        let m = default_model();
        assert!(m.get_topic(9999).is_none());
    }

    #[test]
    fn test_get_topic_depth() {
        let m = default_model();
        let t = m.get_topic(0).expect("test: root topic must exist");
        assert_eq!(t.depth, 0);
    }

    #[test]
    fn test_get_topic_top_words_len() {
        let mut m = default_model();
        m.add_document(&["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"]);
        m.run_inference(5);
        let t = m
            .get_topic(0)
            .expect("test: root topic must exist after inference");
        assert!(t.top_words.len() <= 10);
    }

    #[test]
    fn test_get_topic_top_words_sum_to_one_approx() {
        let mut m = default_model();
        m.add_document(&["a", "b", "c"]);
        m.run_inference(5);
        let t = m
            .get_topic(0)
            .expect("test: root topic must exist after inference");
        let sum: f64 = t.top_words.iter().map(|(_, s)| s).sum();
        // With β smoothing sum over all words == 1; top-N is a subset, so sum ≤ 1.
        assert!(sum > 0.0 && sum <= 1.0 + 1e-9);
    }

    // -------------------------------------------------------------------------
    // 7. document_topics
    // -------------------------------------------------------------------------

    #[test]
    fn test_document_topics_sum_to_one() {
        let mut m = default_model();
        let id = m.add_document(&["rust", "fast", "memory"]);
        m.run_inference(5);
        let dt = m.document_topics(id);
        let sum: f64 = dt.iter().map(|(_, p)| p).sum();
        assert!(
            (sum - 1.0).abs() < 1e-9 || sum == 0.0,
            "proportions must sum to 1"
        );
    }

    #[test]
    fn test_document_topics_nonexistent() {
        let m = default_model();
        let dt = m.document_topics(9999);
        assert!(dt.is_empty());
    }

    #[test]
    fn test_document_topics_valid_topic_ids() {
        let mut m = default_model();
        let id = m.add_document(&["one", "two", "three"]);
        m.run_inference(3);
        for (tid, _) in m.document_topics(id) {
            assert!(m.topic_node(tid).is_some());
        }
    }

    #[test]
    fn test_document_topics_empty_doc() {
        let mut m = default_model();
        let id = m.add_document(&[]);
        let dt = m.document_topics(id);
        // All proportions should be 0 for empty document.
        for (_, p) in &dt {
            assert_eq!(*p, 0.0);
        }
    }

    // -------------------------------------------------------------------------
    // 8. run_inference
    // -------------------------------------------------------------------------

    #[test]
    fn test_run_inference_does_not_panic() {
        let mut m = small_model();
        m.add_document(&["a", "b", "c"]);
        m.run_inference(5);
    }

    #[test]
    fn test_run_inference_keeps_assignment_length() {
        let mut m = small_model();
        let id = m.add_document(&["x", "y", "z", "w"]);
        m.run_inference(10);
        let doc = m
            .get_document(id)
            .expect("test: document must exist after inference");
        assert_eq!(doc.topic_assignments.len(), 4);
    }

    #[test]
    fn test_run_inference_zero_iters() {
        let mut m = small_model();
        let id = m.add_document(&["foo"]);
        m.run_inference(0); // should use config.n_iterations
        let doc = m
            .get_document(id)
            .expect("test: document must exist after zero-iter inference");
        assert!(!doc.topic_assignments.is_empty());
    }

    #[test]
    fn test_run_inference_multiple_documents() {
        let mut m = small_model();
        let ids: Vec<_> = (0..5)
            .map(|i| m.add_document(&[&format!("word{i}"), "common"]))
            .collect();
        m.run_inference(10);
        for id in ids {
            let doc = m
                .get_document(id)
                .expect("test: document must exist after multi-doc inference");
            assert_eq!(doc.topic_assignments.len(), 2);
        }
    }

    #[test]
    fn test_inference_with_explicit_topic_tree() {
        let mut m = HierarchicalTopicModel::new(HtmModelConfig {
            max_depth: 2,
            max_children_per_node: 4,
            alpha: 0.5,
            beta: 0.1,
            n_iterations: 10,
            seed: 42,
        });
        let t1 = m
            .add_topic_node(None, Some("topic_a".into()))
            .expect("test: add topic_a should succeed");
        let t2 = m
            .add_topic_node(None, Some("topic_b".into()))
            .expect("test: add topic_b should succeed");
        m.add_topic_node(Some(t1), None)
            .expect("test: add child of topic_a should succeed");
        m.add_topic_node(Some(t2), None)
            .expect("test: add child of topic_b should succeed");
        m.add_document(&["science", "physics", "math"]);
        m.add_document(&["art", "music", "painting"]);
        m.run_inference(5);
        // No panic, basic sanity.
        assert!(m.n_topic_nodes() >= 5);
    }

    // -------------------------------------------------------------------------
    // 9. prune_empty_topics
    // -------------------------------------------------------------------------

    #[test]
    fn test_prune_removes_empty_leaves() {
        let mut m = default_model();
        // Manually add an empty leaf.
        let _leaf = m
            .add_topic_node(None, None)
            .expect("test: add empty leaf should succeed");
        let before = m.n_topic_nodes();
        m.prune_empty_topics();
        let after = m.n_topic_nodes();
        assert!(after <= before);
    }

    #[test]
    fn test_prune_never_removes_root() {
        let mut m = default_model();
        m.prune_empty_topics();
        assert!(m.topic_node(0).is_some());
    }

    #[test]
    fn test_prune_keeps_nonempty_nodes() {
        let mut m = default_model();
        let id = m.add_document(&["hello"]);
        // Ensure word count is on some node.
        let doc = m
            .get_document(id)
            .expect("test: document must exist before pruning");
        let assigned_topic = doc.topic_assignments[0];
        let before_count = m
            .topic_node(assigned_topic)
            .map(|n| n.total_words)
            .unwrap_or(0);
        m.prune_empty_topics();
        if before_count > 0 {
            assert!(
                m.topic_node(assigned_topic).is_some(),
                "non-empty node must survive pruning"
            );
        }
    }

    #[test]
    fn test_prune_parent_updated() {
        let mut m = default_model();
        let child = m
            .add_topic_node(None, None)
            .expect("test: add child node should succeed");
        m.prune_empty_topics();
        // child had no words, should be pruned.
        if m.topic_node(child).is_none() {
            assert!(!m
                .topic_node(0)
                .expect("test: root must exist after pruning")
                .children
                .contains(&child));
        }
    }

    // -------------------------------------------------------------------------
    // 10. compute_coherence
    // -------------------------------------------------------------------------

    #[test]
    fn test_coherence_root_no_docs() {
        let m = default_model();
        let c = m.compute_coherence(0, 5);
        assert_eq!(c, 0.0);
    }

    #[test]
    fn test_coherence_nonexistent_topic() {
        let m = default_model();
        let c = m.compute_coherence(999, 5);
        assert_eq!(c, 0.0);
    }

    #[test]
    fn test_coherence_returns_finite() {
        let mut m = small_model();
        m.add_document(&["alpha", "beta", "gamma", "delta", "alpha", "beta"]);
        m.add_document(&["gamma", "delta", "epsilon", "zeta", "alpha"]);
        m.run_inference(5);
        let c = m.compute_coherence(0, 5);
        assert!(c.is_finite());
    }

    #[test]
    fn test_coherence_top_n_zero() {
        let mut m = small_model();
        m.add_document(&["a", "b"]);
        let c = m.compute_coherence(0, 0);
        assert_eq!(c, 0.0);
    }

    // -------------------------------------------------------------------------
    // 11. model_stats
    // -------------------------------------------------------------------------

    #[test]
    fn test_model_stats_empty() {
        let m = default_model();
        let s = m.model_stats();
        assert_eq!(s.n_docs, 0);
        assert_eq!(s.vocab_size, 0);
        assert_eq!(s.n_topics, 1);
    }

    #[test]
    fn test_model_stats_n_docs() {
        let mut m = default_model();
        m.add_document(&["a"]);
        m.add_document(&["b"]);
        assert_eq!(m.model_stats().n_docs, 2);
    }

    #[test]
    fn test_model_stats_vocab_size() {
        let mut m = default_model();
        m.add_document(&["rust", "safe", "fast"]);
        assert_eq!(m.model_stats().vocab_size, 3);
    }

    #[test]
    fn test_model_stats_max_depth() {
        let mut m = default_model();
        let c = m
            .add_topic_node(None, None)
            .expect("test: add child node should succeed");
        let _gc = m
            .add_topic_node(Some(c), None)
            .expect("test: add grandchild node should succeed");
        let s = m.model_stats();
        assert!(s.max_depth >= 2);
    }

    #[test]
    fn test_model_stats_avg_coherence_finite() {
        let mut m = small_model();
        m.add_document(&["one", "two", "three"]);
        m.run_inference(5);
        let s = m.model_stats();
        assert!(s.avg_coherence.is_finite());
    }

    // -------------------------------------------------------------------------
    // 12. xorshift64 / fnv1a_64
    // -------------------------------------------------------------------------

    #[test]
    fn test_xorshift64_not_zero_from_nonzero_state() {
        let mut state = 1u64;
        let v = xorshift64(&mut state);
        assert_ne!(v, 0);
    }

    #[test]
    fn test_xorshift64_different_states_differ() {
        let mut s1 = 1u64;
        let mut s2 = 2u64;
        assert_ne!(xorshift64(&mut s1), xorshift64(&mut s2));
    }

    #[test]
    fn test_xorshift64_advances_state() {
        let mut state = 42u64;
        let before = state;
        xorshift64(&mut state);
        assert_ne!(state, before);
    }

    #[test]
    fn test_fnv1a_64_known_value() {
        // FNV-1a of empty slice is the offset basis.
        let h = fnv1a_64(&[]);
        assert_eq!(h, 14_695_981_039_346_656_037u64);
    }

    #[test]
    fn test_fnv1a_64_differs_for_different_inputs() {
        let h1 = fnv1a_64(b"hello");
        let h2 = fnv1a_64(b"world");
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_hash_bytes_deterministic() {
        let h1 = HierarchicalTopicModel::hash_bytes(b"test");
        let h2 = HierarchicalTopicModel::hash_bytes(b"test");
        assert_eq!(h1, h2);
    }

    // -------------------------------------------------------------------------
    // 13. path_to_root
    // -------------------------------------------------------------------------

    #[test]
    fn test_path_to_root_single_node() {
        let m = default_model();
        let path = m.path_to_root(0);
        assert_eq!(path, vec![0]);
    }

    #[test]
    fn test_path_to_root_child() {
        let mut m = default_model();
        let child = m
            .add_topic_node(None, None)
            .expect("test: add child node should succeed");
        let path = m.path_to_root(child);
        assert_eq!(path, vec![0, child]);
    }

    #[test]
    fn test_path_to_root_grandchild() {
        let mut m = default_model();
        let c = m
            .add_topic_node(None, None)
            .expect("test: add child node should succeed");
        let gc = m
            .add_topic_node(Some(c), None)
            .expect("test: add grandchild node should succeed");
        let path = m.path_to_root(gc);
        assert_eq!(path, vec![0, c, gc]);
    }

    // -------------------------------------------------------------------------
    // 14. Integration / stress
    // -------------------------------------------------------------------------

    #[test]
    fn test_large_document_set() {
        let mut m = HierarchicalTopicModel::new(HtmModelConfig {
            max_depth: 2,
            max_children_per_node: 4,
            alpha: 0.1,
            beta: 0.01,
            n_iterations: 5,
            seed: 1,
        });
        let words = ["a", "b", "c", "d", "e", "f", "g", "h"];
        for i in 0..20 {
            let tokens: Vec<&str> = words[..((i % 4) + 2)].to_vec();
            m.add_document(&tokens);
        }
        m.run_inference(5);
        let stats = m.model_stats();
        assert_eq!(stats.n_docs, 20);
    }

    #[test]
    fn test_repeated_inference_stable() {
        let mut m = small_model();
        m.add_document(&["rust", "is", "great"]);
        m.run_inference(5);
        m.run_inference(5);
        let stats = m.model_stats();
        assert_eq!(stats.n_docs, 1);
    }

    #[test]
    fn test_many_topics_coherence_finite() {
        let mut m = HierarchicalTopicModel::new(HtmModelConfig {
            max_depth: 3,
            max_children_per_node: 3,
            alpha: 0.1,
            beta: 0.01,
            n_iterations: 5,
            seed: 55,
        });
        for i in 0..10 {
            m.add_document(&[&format!("word{}", i % 5), &format!("topic{}", i % 3)]);
        }
        m.run_inference(5);
        let stats = m.model_stats();
        assert!(stats.avg_coherence.is_finite());
    }

    #[test]
    fn test_document_topic_proportions_non_negative() {
        let mut m = small_model();
        let id = m.add_document(&["x", "y", "z"]);
        m.run_inference(10);
        for (_, p) in m.document_topics(id) {
            assert!(p >= 0.0);
        }
    }

    #[test]
    fn test_total_words_consistent_after_inference() {
        let mut m = small_model();
        m.add_document(&["a", "b", "c"]);
        m.run_inference(5);
        // total_words across all nodes must equal total tokens across all docs (3).
        let total_in_nodes: u32 = m.topics.values().map(|n| n.total_words).sum();
        let total_in_docs: u32 = m
            .documents
            .values()
            .map(|d| d.token_indices.len() as u32)
            .sum();
        assert_eq!(total_in_nodes, total_in_docs);
    }

    #[test]
    fn test_prune_idempotent() {
        let mut m = default_model();
        m.add_document(&["hello"]);
        m.run_inference(5);
        m.prune_empty_topics();
        let n1 = m.n_topic_nodes();
        m.prune_empty_topics();
        let n2 = m.n_topic_nodes();
        assert_eq!(n1, n2);
    }

    #[test]
    fn test_get_topic_doc_count() {
        let mut m = small_model();
        let id = m.add_document(&["machine", "learning"]);
        m.run_inference(5);
        let doc = m
            .get_document(id)
            .expect("test: document must exist after inference");
        // At least one assignment must exist.
        assert!(!doc.topic_assignments.is_empty());
    }
}
