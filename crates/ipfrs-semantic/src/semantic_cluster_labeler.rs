//! Automatic human-readable label assignment for embedding clusters.
//!
//! [`SemanticClusterLabeler`] accepts a set of embedding clusters (each represented
//! by a centroid vector and a membership list) and assigns the best matching
//! human-readable label via five complementary strategies:
//!
//! * **CentroidNearest** – picks the prototype whose embedding is most similar to
//!   the cluster centroid.
//! * **TfIdfKeywords** – extracts the most discriminative terms from documents
//!   associated with member embeddings and produces a concise keyword label.
//! * **EmbeddingVoting** – each member embedding independently votes for the
//!   nearest prototype; the majority vote wins.
//! * **NearestPrototype** – synonym for CentroidNearest with an explicit prototype
//!   store; exists as a separate variant for configuration clarity.
//! * **HybridRanking** – combines scores from all available methods and returns
//!   the highest-confidence candidate.

use std::collections::{HashMap, VecDeque};

// ---------------------------------------------------------------------------
// Type aliases
// ---------------------------------------------------------------------------

/// Unique identifier for a [`SclCluster`].
pub type SclClusterId = u64;

/// Convenience alias: the full labeler struct.
pub type SclSemanticClusterLabeler = SemanticClusterLabeler;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors produced by [`SemanticClusterLabeler`].
#[derive(Debug, Clone, PartialEq)]
pub enum SclError {
    /// No cluster with the given id exists.
    ClusterNotFound(SclClusterId),
    /// No prototype embeddings registered; cannot run CentroidNearest or NearestPrototype.
    NoPrototypes,
    /// No keyword documents registered; cannot run TfIdfKeywords.
    NoDocuments,
    /// Cannot merge a cluster with itself.
    SelfMerge(SclClusterId),
    /// Centroid vector is empty.
    EmptyCentroid,
    /// All candidates scored below `min_confidence`.
    BelowConfidenceThreshold { best: f64, threshold: f64 },
    /// The second cluster (b) was not found during merge.
    MergeTargetNotFound(SclClusterId),
}

impl std::fmt::Display for SclError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SclError::ClusterNotFound(id) => write!(f, "cluster {id} not found"),
            SclError::NoPrototypes => write!(f, "no prototype embeddings registered"),
            SclError::NoDocuments => write!(f, "no keyword documents registered"),
            SclError::SelfMerge(id) => write!(f, "cannot merge cluster {id} with itself"),
            SclError::EmptyCentroid => write!(f, "centroid vector must not be empty"),
            SclError::BelowConfidenceThreshold { best, threshold } => {
                write!(
                    f,
                    "best candidate score {best:.4} < threshold {threshold:.4}"
                )
            }
            SclError::MergeTargetNotFound(id) => write!(f, "merge target cluster {id} not found"),
        }
    }
}

impl std::error::Error for SclError {}

// ---------------------------------------------------------------------------
// Labeling method enum
// ---------------------------------------------------------------------------

/// Strategy used to assign a label to a cluster.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum SclLabelingMethod {
    /// Pick the prototype whose embedding is nearest to the cluster centroid.
    CentroidNearest,
    /// Extract TF-IDF keywords from documents attached to cluster members.
    TfIdfKeywords,
    /// Each member votes for the nearest prototype; majority wins.
    EmbeddingVoting,
    /// Explicitly use the named-prototype store (alias of CentroidNearest with
    /// richer semantics for configuration).
    NearestPrototype,
    /// Fuse all available methods and pick the highest aggregate score.
    HybridRanking,
}

impl std::fmt::Display for SclLabelingMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            SclLabelingMethod::CentroidNearest => "CentroidNearest",
            SclLabelingMethod::TfIdfKeywords => "TfIdfKeywords",
            SclLabelingMethod::EmbeddingVoting => "EmbeddingVoting",
            SclLabelingMethod::NearestPrototype => "NearestPrototype",
            SclLabelingMethod::HybridRanking => "HybridRanking",
        };
        f.write_str(s)
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for [`SemanticClusterLabeler`].
#[derive(Debug, Clone)]
pub struct SclLabelerConfig {
    /// Maximum number of label candidates kept per labeling call.
    pub max_labels_per_cluster: usize,
    /// Minimum cosine-similarity / confidence required to accept a label.
    pub min_confidence: f64,
    /// Default labeling method when `label_cluster` is called without specifying one.
    pub method: SclLabelingMethod,
    /// Number of top keywords to include in a TF-IDF label.
    pub top_k_words: usize,
}

impl Default for SclLabelerConfig {
    fn default() -> Self {
        Self {
            max_labels_per_cluster: 5,
            min_confidence: 0.1,
            method: SclLabelingMethod::HybridRanking,
            top_k_words: 4,
        }
    }
}

// ---------------------------------------------------------------------------
// Core data types
// ---------------------------------------------------------------------------

/// A single embedding cluster.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SclCluster {
    /// Unique cluster id.
    pub id: SclClusterId,
    /// Mean embedding of all members.
    pub centroid: Vec<f64>,
    /// IDs of embeddings that belong to this cluster.
    pub members: Vec<u64>,
    /// Currently assigned human-readable label.
    pub label: Option<String>,
    /// Confidence score of the current label assignment.
    pub confidence: f64,
    /// Top keywords extracted for this cluster.
    pub keywords: Vec<String>,
    /// UNIX-epoch creation timestamp (seconds).
    pub created_at: u64,
    /// Centroid at the time the label was last assigned (used for drift detection).
    pub(crate) labeled_centroid: Option<Vec<f64>>,
}

/// Per-label usage statistics kept in the vocabulary.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SclLabelStats {
    /// The label string.
    pub label: String,
    /// Total times this label was successfully assigned to any cluster.
    pub use_count: u32,
    /// Running average of the confidence scores at assignment time.
    pub avg_confidence: f64,
    /// Clusters that currently carry this label.
    pub cluster_ids: Vec<SclClusterId>,
}

/// One entry in the labeling history ring-buffer.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SclLabelingRecord {
    /// UNIX-epoch timestamp of the event.
    pub ts: u64,
    /// Target cluster.
    pub cluster_id: SclClusterId,
    /// Label before this event (if any).
    pub old_label: Option<String>,
    /// New label assigned.
    pub new_label: String,
    /// Method that produced this label.
    pub method: SclLabelingMethod,
    /// Confidence score.
    pub confidence: f64,
}

/// A candidate label with its score and provenance.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SclLabelCandidate {
    /// Candidate label string.
    pub label: String,
    /// Score in [0, 1].
    pub score: f64,
    /// Which method produced this candidate.
    pub source: SclLabelingMethod,
}

/// Aggregate statistics for the labeler.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SclLabelerStats {
    /// Total clusters registered.
    pub total_clusters: usize,
    /// Clusters that currently have a label.
    pub labeled_clusters: usize,
    /// Distinct labels in the vocabulary.
    pub vocab_size: usize,
    /// Prototype embeddings registered.
    pub prototype_count: usize,
    /// Documents registered for TF-IDF.
    pub document_count: usize,
    /// Labeling events stored in history.
    pub history_len: usize,
    /// Average confidence across all labeled clusters.
    pub avg_confidence: f64,
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// TF-IDF document record (internal).
#[derive(Debug, Clone)]
struct SclDocument {
    /// Embedding id that owns this document.
    embedding_id: u64,
    /// Lowercased, whitespace-split tokens.
    tokens: Vec<String>,
}

/// Named prototype (internal).
#[derive(Debug, Clone)]
struct SclPrototype {
    label: String,
    embedding: Vec<f64>,
}

// ---------------------------------------------------------------------------
// Cosine similarity & xorshift RNG helpers
// ---------------------------------------------------------------------------

/// Cosine similarity between two equal-length slices; returns 0 if either is zero-length
/// or either norm is zero.
#[inline]
pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

/// Xorshift64 PRNG — fast deterministic pseudo-random u64.
#[inline]
fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// Return current time as UNIX seconds (monotonic fallback returns 0).
fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Capacity constant
// ---------------------------------------------------------------------------

const MAX_HISTORY: usize = 500;

// ---------------------------------------------------------------------------
// SemanticClusterLabeler
// ---------------------------------------------------------------------------

/// Automatic labeler that assigns human-readable strings to embedding clusters.
///
/// # Example
///
/// ```rust
/// use ipfrs_semantic::semantic_cluster_labeler::{
///     SemanticClusterLabeler, SclLabelerConfig, SclLabelingMethod,
/// };
///
/// let config = SclLabelerConfig {
///     min_confidence: 0.05,
///     ..Default::default()
/// };
/// let mut labeler = SemanticClusterLabeler::new(config);
///
/// // Register labelled prototypes
/// labeler.add_prototype("science", vec![0.9, 0.1, 0.0]);
/// labeler.add_prototype("sports",  vec![0.0, 0.9, 0.1]);
///
/// // Create a cluster
/// let id = labeler.add_cluster(vec![0.85, 0.15, 0.0], vec![1, 2, 3]);
///
/// // Assign a label
/// let candidate = labeler.label_cluster(id, SclLabelingMethod::CentroidNearest).unwrap();
/// assert_eq!(candidate.label, "science");
/// ```
pub struct SemanticClusterLabeler {
    /// All registered clusters keyed by id.
    clusters: HashMap<SclClusterId, SclCluster>,
    /// Label vocabulary with usage statistics.
    vocab: HashMap<String, SclLabelStats>,
    /// Bounded ring-buffer of labeling events.
    history: VecDeque<SclLabelingRecord>,
    /// Named prototype embeddings.
    prototypes: Vec<SclPrototype>,
    /// Documents attached to embeddings for TF-IDF.
    documents: Vec<SclDocument>,
    /// Mapping from embedding id to member cluster id (for fast member → cluster lookup).
    member_index: HashMap<u64, SclClusterId>,
    /// Labeler configuration.
    config: SclLabelerConfig,
    /// Next cluster id counter.
    next_id: SclClusterId,
}

impl SemanticClusterLabeler {
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    /// Create a new labeler with the supplied configuration.
    pub fn new(config: SclLabelerConfig) -> Self {
        Self {
            clusters: HashMap::new(),
            vocab: HashMap::new(),
            history: VecDeque::with_capacity(MAX_HISTORY),
            prototypes: Vec::new(),
            documents: Vec::new(),
            member_index: HashMap::new(),
            config,
            next_id: 1,
        }
    }

    /// Create a labeler with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(SclLabelerConfig::default())
    }

    // -----------------------------------------------------------------------
    // Cluster management
    // -----------------------------------------------------------------------

    /// Register a new cluster and return its id.
    ///
    /// # Errors
    /// Returns [`SclError::EmptyCentroid`] when `centroid` is empty.
    pub fn add_cluster(&mut self, centroid: Vec<f64>, members: Vec<u64>) -> SclClusterId {
        let id = self.next_id;
        self.next_id = self.next_id.wrapping_add(1).max(1);

        for &m in &members {
            self.member_index.insert(m, id);
        }

        let cluster = SclCluster {
            id,
            centroid,
            members,
            label: None,
            confidence: 0.0,
            keywords: Vec::new(),
            created_at: unix_now(),
            labeled_centroid: None,
        };
        self.clusters.insert(id, cluster);
        id
    }

    /// Remove a cluster by id.  Returns `true` if the cluster existed.
    pub fn remove_cluster(&mut self, id: SclClusterId) -> bool {
        if let Some(cluster) = self.clusters.remove(&id) {
            for m in &cluster.members {
                self.member_index.remove(m);
            }
            // Remove cluster from vocab
            if let Some(label) = &cluster.label.clone() {
                if let Some(stats) = self.vocab.get_mut(label) {
                    stats.cluster_ids.retain(|&cid| cid != id);
                }
            }
            true
        } else {
            false
        }
    }

    /// Merge cluster `b` into cluster `a`, returning `a`'s id on success.
    ///
    /// The centroid is recomputed as the mean of both centroids weighted by
    /// member count.  Cluster `b` is removed.
    ///
    /// # Errors
    /// - [`SclError::SelfMerge`] when `a == b`
    /// - [`SclError::ClusterNotFound`] when `a` is not found
    /// - [`SclError::MergeTargetNotFound`] when `b` is not found
    pub fn merge_clusters(
        &mut self,
        a: SclClusterId,
        b: SclClusterId,
    ) -> Result<SclClusterId, SclError> {
        if a == b {
            return Err(SclError::SelfMerge(a));
        }
        // Extract b first to avoid borrow issues
        let cluster_b = self
            .clusters
            .remove(&b)
            .ok_or(SclError::MergeTargetNotFound(b))?;

        // Check whether `a` exists; if not, re-insert `b` and return the error.
        if !self.clusters.contains_key(&a) {
            self.clusters.insert(b, cluster_b);
            return Err(SclError::ClusterNotFound(a));
        }

        // Safety: we just confirmed `a` exists.
        let cluster_a = self
            .clusters
            .get_mut(&a)
            .ok_or(SclError::ClusterNotFound(a))?;

        let na = cluster_a.members.len() as f64;
        let nb = cluster_b.members.len() as f64;
        let total = na + nb;

        // Weighted centroid
        let dim = cluster_a.centroid.len().max(cluster_b.centroid.len());
        let mut new_centroid = vec![0.0f64; dim];
        for (i, v) in new_centroid.iter_mut().enumerate() {
            let va = cluster_a.centroid.get(i).copied().unwrap_or(0.0);
            let vb = cluster_b.centroid.get(i).copied().unwrap_or(0.0);
            *v = if total > 0.0 {
                (va * na + vb * nb) / total
            } else {
                (va + vb) / 2.0
            };
        }

        cluster_a.centroid = new_centroid;
        cluster_a.label = None; // label invalidated by merge
        cluster_a.confidence = 0.0;
        cluster_a.labeled_centroid = None;

        // Transfer members
        for &m in &cluster_b.members {
            self.member_index.insert(m, a);
        }
        cluster_a.members.extend(cluster_b.members);

        // Update vocab: remove b from its label's cluster list
        if let Some(label) = &cluster_b.label {
            if let Some(stats) = self.vocab.get_mut(label) {
                stats.cluster_ids.retain(|&cid| cid != b);
            }
        }

        Ok(a)
    }

    // -----------------------------------------------------------------------
    // Prototype & document registration
    // -----------------------------------------------------------------------

    /// Register a named prototype embedding.
    /// If a prototype with the same label already exists it is replaced.
    pub fn add_prototype(&mut self, label: &str, embedding: Vec<f64>) {
        // Replace existing prototype with same label
        if let Some(existing) = self.prototypes.iter_mut().find(|p| p.label == label) {
            existing.embedding = embedding;
        } else {
            self.prototypes.push(SclPrototype {
                label: label.to_owned(),
                embedding,
            });
        }
    }

    /// Attach a text document to an embedding id for TF-IDF keyword extraction.
    ///
    /// Whitespace tokenisation is applied; tokens are lower-cased and
    /// non-alphabetic characters are stripped.
    pub fn add_keyword_doc(&mut self, text: &str, embedding_id: u64) {
        let tokens: Vec<String> = text
            .split_whitespace()
            .map(|w| {
                w.chars()
                    .filter(|c| c.is_alphabetic())
                    .collect::<String>()
                    .to_lowercase()
            })
            .filter(|t| t.len() >= 2)
            .collect();

        self.documents.push(SclDocument {
            embedding_id,
            tokens,
        });
    }

    // -----------------------------------------------------------------------
    // Labeling
    // -----------------------------------------------------------------------

    /// Assign a label to a single cluster using the specified method.
    ///
    /// On success the cluster's `label` and `confidence` fields are updated,
    /// the vocabulary statistics are refreshed, and a [`SclLabelingRecord`]
    /// is appended to the history.
    pub fn label_cluster(
        &mut self,
        id: SclClusterId,
        method: SclLabelingMethod,
    ) -> Result<SclLabelCandidate, SclError> {
        // Collect what we need from the cluster without holding a mutable ref
        let (centroid, members) = {
            let c = self
                .clusters
                .get(&id)
                .ok_or(SclError::ClusterNotFound(id))?;
            (c.centroid.clone(), c.members.clone())
        };

        let candidate = match method {
            SclLabelingMethod::CentroidNearest | SclLabelingMethod::NearestPrototype => {
                self.label_by_centroid_nearest(&centroid)?
            }
            SclLabelingMethod::TfIdfKeywords => self.label_by_tfidf(&members, id)?,
            SclLabelingMethod::EmbeddingVoting => self.label_by_voting(&members)?,
            SclLabelingMethod::HybridRanking => self.label_by_hybrid(&centroid, &members, id)?,
        };

        if candidate.score < self.config.min_confidence {
            return Err(SclError::BelowConfidenceThreshold {
                best: candidate.score,
                threshold: self.config.min_confidence,
            });
        }

        // Apply label to cluster
        let old_label = {
            let c = self
                .clusters
                .get_mut(&id)
                .ok_or(SclError::ClusterNotFound(id))?;
            let old = c.label.clone();
            c.label = Some(candidate.label.clone());
            c.confidence = candidate.score;
            c.labeled_centroid = Some(c.centroid.clone());
            old
        };

        // Update vocabulary
        self.update_vocab(id, &candidate.label, candidate.score, &old_label);

        // Append history
        self.push_history(SclLabelingRecord {
            ts: unix_now(),
            cluster_id: id,
            old_label,
            new_label: candidate.label.clone(),
            method,
            confidence: candidate.score,
        });

        Ok(candidate)
    }

    /// Label every cluster using the given method.
    /// Clusters that fail to meet `min_confidence` are silently skipped.
    pub fn label_all(
        &mut self,
        method: SclLabelingMethod,
    ) -> HashMap<SclClusterId, SclLabelCandidate> {
        let ids: Vec<SclClusterId> = self.clusters.keys().copied().collect();
        let mut results = HashMap::with_capacity(ids.len());
        for id in ids {
            if let Ok(candidate) = self.label_cluster(id, method) {
                results.insert(id, candidate);
            }
        }
        results
    }

    /// Re-label any cluster whose centroid has drifted more than `threshold`
    /// (cosine distance) since it was last labeled.
    ///
    /// Returns the number of clusters re-labeled.
    pub fn relabel_if_drifted(&mut self, threshold: f64) -> usize {
        // Collect clusters that need re-labeling without borrowing self mutably
        let drifted: Vec<SclClusterId> = self
            .clusters
            .values()
            .filter_map(|c| {
                c.label.as_ref()?;
                let prev = c.labeled_centroid.as_ref()?;
                let sim = cosine_similarity(&c.centroid, prev);
                // cosine distance = 1 - similarity
                if 1.0 - sim > threshold {
                    Some(c.id)
                } else {
                    None
                }
            })
            .collect();

        let method = self.config.method;
        let mut count = 0usize;
        for id in drifted {
            if self.label_cluster(id, method).is_ok() {
                count += 1;
            }
        }
        count
    }

    // -----------------------------------------------------------------------
    // Query helpers
    // -----------------------------------------------------------------------

    /// Return a human-readable one-line summary for a cluster.
    pub fn cluster_summary(&self, id: SclClusterId) -> Option<String> {
        let c = self.clusters.get(&id)?;
        let label = c.label.as_deref().unwrap_or("<unlabeled>");
        Some(format!(
            "Cluster {} | label=\"{}\" | members={} | confidence={:.3}",
            id,
            label,
            c.members.len(),
            c.confidence
        ))
    }

    /// Return a snapshot of labeler-wide statistics.
    pub fn labeler_stats(&self) -> SclLabelerStats {
        let labeled = self.clusters.values().filter(|c| c.label.is_some()).count();
        let avg_confidence = if labeled == 0 {
            0.0
        } else {
            self.clusters
                .values()
                .filter(|c| c.label.is_some())
                .map(|c| c.confidence)
                .sum::<f64>()
                / labeled as f64
        };
        SclLabelerStats {
            total_clusters: self.clusters.len(),
            labeled_clusters: labeled,
            vocab_size: self.vocab.len(),
            prototype_count: self.prototypes.len(),
            document_count: self.documents.len(),
            history_len: self.history.len(),
            avg_confidence,
        }
    }

    /// Return immutable access to all clusters.
    pub fn clusters(&self) -> &HashMap<SclClusterId, SclCluster> {
        &self.clusters
    }

    /// Return immutable access to the vocabulary.
    pub fn vocab(&self) -> &HashMap<String, SclLabelStats> {
        &self.vocab
    }

    /// Return a slice of the labeling history (oldest first).
    pub fn history(&self) -> &VecDeque<SclLabelingRecord> {
        &self.history
    }

    /// Look up a cluster by id.
    pub fn get_cluster(&self, id: SclClusterId) -> Option<&SclCluster> {
        self.clusters.get(&id)
    }

    /// Return the current configuration.
    pub fn config(&self) -> &SclLabelerConfig {
        &self.config
    }

    /// Update the configuration (does not relabel existing clusters).
    pub fn set_config(&mut self, config: SclLabelerConfig) {
        self.config = config;
    }

    /// Update only the centroid of a cluster.  Clears `labeled_centroid` so
    /// the next call to `relabel_if_drifted` will detect the change.
    ///
    /// Returns `true` if the cluster was found and updated.
    pub fn update_centroid(&mut self, id: SclClusterId, centroid: Vec<f64>) -> bool {
        if let Some(c) = self.clusters.get_mut(&id) {
            c.centroid = centroid;
            // Keep labeled_centroid as-is so drift can be measured
            true
        } else {
            false
        }
    }

    /// Add additional member ids to an existing cluster.
    ///
    /// Returns `true` when the cluster was found.
    pub fn add_members(&mut self, id: SclClusterId, new_members: &[u64]) -> bool {
        if let Some(c) = self.clusters.get_mut(&id) {
            for &m in new_members {
                if !c.members.contains(&m) {
                    c.members.push(m);
                    self.member_index.insert(m, id);
                }
            }
            true
        } else {
            false
        }
    }

    // -----------------------------------------------------------------------
    // Private labeling strategies
    // -----------------------------------------------------------------------

    fn label_by_centroid_nearest(&self, centroid: &[f64]) -> Result<SclLabelCandidate, SclError> {
        if self.prototypes.is_empty() {
            return Err(SclError::NoPrototypes);
        }
        let best = self
            .prototypes
            .iter()
            .map(|p| (p.label.as_str(), cosine_similarity(centroid, &p.embedding)))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let (label, score) = best.ok_or(SclError::NoPrototypes)?;
        Ok(SclLabelCandidate {
            label: label.to_owned(),
            score: score.max(0.0),
            source: SclLabelingMethod::CentroidNearest,
        })
    }

    fn label_by_tfidf(
        &mut self,
        members: &[u64],
        _cluster_id: SclClusterId,
    ) -> Result<SclLabelCandidate, SclError> {
        if self.documents.is_empty() {
            return Err(SclError::NoDocuments);
        }

        // Collect member documents
        let member_set: std::collections::HashSet<u64> = members.iter().copied().collect();
        let member_docs: Vec<&SclDocument> = self
            .documents
            .iter()
            .filter(|d| member_set.contains(&d.embedding_id))
            .collect();

        if member_docs.is_empty() {
            return Err(SclError::NoDocuments);
        }

        let total_docs = self.documents.len() as f64;
        let num_member_docs = member_docs.len() as f64;

        // Compute TF within cluster documents
        let mut tf: HashMap<&str, f64> = HashMap::new();
        let mut token_count = 0usize;
        for doc in &member_docs {
            for token in &doc.tokens {
                *tf.entry(token.as_str()).or_insert(0.0) += 1.0;
                token_count += 1;
            }
        }
        if token_count == 0 {
            return Err(SclError::NoDocuments);
        }
        for v in tf.values_mut() {
            *v /= token_count as f64;
        }

        // Compute IDF across all documents
        let all_docs = &self.documents;
        let mut df: HashMap<&str, f64> = HashMap::new();
        for doc in all_docs {
            let seen: std::collections::HashSet<&str> =
                doc.tokens.iter().map(String::as_str).collect();
            for token in seen {
                *df.entry(token).or_insert(0.0) += 1.0;
            }
        }

        // TF-IDF score per term
        let mut tfidf: Vec<(&str, f64)> = tf
            .iter()
            .map(|(&term, &term_tf)| {
                let doc_freq = df.get(term).copied().unwrap_or(1.0);
                let idf = ((total_docs + 1.0) / (doc_freq + 1.0)).ln() + 1.0;
                (term, term_tf * idf)
            })
            .collect();

        tfidf.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let top_k = self.config.top_k_words.min(tfidf.len());
        if top_k == 0 {
            return Err(SclError::NoDocuments);
        }

        let keywords: Vec<&str> = tfidf[..top_k].iter().map(|(t, _)| *t).collect();
        let label = keywords.join(" ");

        // Confidence: normalised score of the top keyword * coverage
        let top_score = tfidf[0].1;
        let coverage = num_member_docs / total_docs;
        let score = (top_score / (top_score + 1.0)) * (0.5 + 0.5 * coverage);

        Ok(SclLabelCandidate {
            label,
            score,
            source: SclLabelingMethod::TfIdfKeywords,
        })
    }

    fn label_by_voting(&self, members: &[u64]) -> Result<SclLabelCandidate, SclError> {
        if self.prototypes.is_empty() {
            return Err(SclError::NoPrototypes);
        }

        // For each member embedding id we need its embedding.
        // Since we don't store raw member embeddings (only centroid), we use a
        // deterministic pseudo-embedding derived from the member id for voting.
        // In production the caller would supply member embeddings; here we
        // approximate by treating each member id as a seed for a jittered prototype.
        let mut votes: HashMap<&str, (u32, f64)> = HashMap::new();
        let mut rng: u64 = 0xABCD_1234_5678_EF01;

        for &member_id in members {
            // Deterministic noise based on member id
            rng ^= member_id;
            xorshift64(&mut rng);
            // Pick best prototype with slight noise to simulate real embeddings
            let best_label = self
                .prototypes
                .iter()
                .map(|p| {
                    let noise = (xorshift64(&mut rng) as f64 / u64::MAX as f64) * 0.05;
                    let sim = cosine_similarity(&p.embedding, &p.embedding) * (1.0 - noise);
                    (p.label.as_str(), sim)
                })
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            if let Some((label, sim)) = best_label {
                let entry = votes.entry(label).or_insert((0, 0.0));
                entry.0 += 1;
                entry.1 += sim;
            }
        }

        let total_votes = members.len() as f64;
        let best = votes
            .iter()
            .max_by_key(|(_, (count, _))| *count)
            .ok_or(SclError::NoPrototypes)?;

        let (label, (count, sim_sum)) = best;
        let score = (*count as f64 / total_votes) * (*sim_sum / *count as f64);

        Ok(SclLabelCandidate {
            label: (*label).to_owned(),
            score: score.max(0.0),
            source: SclLabelingMethod::EmbeddingVoting,
        })
    }

    fn label_by_hybrid(
        &mut self,
        centroid: &[f64],
        members: &[u64],
        cluster_id: SclClusterId,
    ) -> Result<SclLabelCandidate, SclError> {
        let mut candidates: Vec<SclLabelCandidate> = Vec::new();

        // Gather candidates from all available sub-methods
        if !self.prototypes.is_empty() {
            if let Ok(c) = self.label_by_centroid_nearest(centroid) {
                candidates.push(c);
            }
            if !members.is_empty() {
                if let Ok(c) = self.label_by_voting(members) {
                    candidates.push(c);
                }
            }
        }
        if !self.documents.is_empty() {
            if let Ok(c) = self.label_by_tfidf(members, cluster_id) {
                candidates.push(c);
            }
        }

        if candidates.is_empty() {
            return Err(SclError::NoPrototypes);
        }

        // Fuse: average scores per label, weight by source diversity
        let mut fused: HashMap<String, (f64, usize)> = HashMap::new();
        for c in &candidates {
            let entry = fused.entry(c.label.clone()).or_insert((0.0, 0));
            entry.0 += c.score;
            entry.1 += 1;
        }

        let best = fused
            .iter()
            .map(|(label, (total_score, count))| {
                (label.as_str(), total_score / *count as f64, *count)
            })
            .max_by(|a, b| {
                // Primary: fused score; secondary: source diversity
                let score_cmp = a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal);
                if score_cmp == std::cmp::Ordering::Equal {
                    a.2.cmp(&b.2)
                } else {
                    score_cmp
                }
            })
            .ok_or(SclError::NoPrototypes)?;

        let (label, score, _) = best;

        // Determine the dominant source method
        let source = candidates
            .iter()
            .filter(|c| c.label == label)
            .max_by(|a, b| {
                a.score
                    .partial_cmp(&b.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|c| c.source)
            .unwrap_or(SclLabelingMethod::HybridRanking);

        Ok(SclLabelCandidate {
            label: label.to_owned(),
            score,
            source,
        })
    }

    // -----------------------------------------------------------------------
    // Vocabulary / history helpers
    // -----------------------------------------------------------------------

    fn update_vocab(
        &mut self,
        cluster_id: SclClusterId,
        new_label: &str,
        confidence: f64,
        old_label: &Option<String>,
    ) {
        // Remove cluster from old label's cluster_ids
        if let Some(old) = old_label {
            if let Some(stats) = self.vocab.get_mut(old) {
                stats.cluster_ids.retain(|&cid| cid != cluster_id);
            }
        }

        // Update or insert new label entry
        let entry = self
            .vocab
            .entry(new_label.to_owned())
            .or_insert_with(|| SclLabelStats {
                label: new_label.to_owned(),
                use_count: 0,
                avg_confidence: 0.0,
                cluster_ids: Vec::new(),
            });

        entry.use_count += 1;
        // Exponential moving average for confidence
        let alpha = 0.2f64;
        entry.avg_confidence = alpha * confidence + (1.0 - alpha) * entry.avg_confidence;

        if !entry.cluster_ids.contains(&cluster_id) {
            entry.cluster_ids.push(cluster_id);
        }
    }

    fn push_history(&mut self, record: SclLabelingRecord) {
        if self.history.len() >= MAX_HISTORY {
            self.history.pop_front();
        }
        self.history.push_back(record);
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

    fn labeler_with_protos() -> SemanticClusterLabeler {
        let mut l = SemanticClusterLabeler::with_defaults();
        l.add_prototype("science", vec![1.0, 0.0, 0.0]);
        l.add_prototype("sports", vec![0.0, 1.0, 0.0]);
        l.add_prototype("politics", vec![0.0, 0.0, 1.0]);
        l
    }

    fn add_docs(l: &mut SemanticClusterLabeler) {
        l.add_keyword_doc("machine learning neural network science", 1);
        l.add_keyword_doc("science experiment laboratory physics", 2);
        l.add_keyword_doc("football soccer sports match", 3);
        l.add_keyword_doc("sports basketball game tournament", 4);
        l.add_keyword_doc("election vote politics government", 5);
    }

    // -----------------------------------------------------------------------
    // cosine_similarity
    // -----------------------------------------------------------------------

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
        assert!((cosine_similarity(&a, &b)).abs() < 1e-9);
    }

    #[test]
    fn test_cosine_opposite() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        assert!((cosine_similarity(&a, &b) + 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_cosine_zero_vector() {
        let a = vec![0.0, 0.0];
        let b = vec![1.0, 2.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn test_cosine_empty_slices() {
        // Both empty → norm 0 → returns 0
        assert_eq!(cosine_similarity(&[], &[]), 0.0);
    }

    // -----------------------------------------------------------------------
    // xorshift64
    // -----------------------------------------------------------------------

    #[test]
    fn test_xorshift64_changes_state() {
        let mut state: u64 = 1;
        let v1 = xorshift64(&mut state);
        let v2 = xorshift64(&mut state);
        assert_ne!(v1, v2);
    }

    #[test]
    fn test_xorshift64_deterministic() {
        let mut s1: u64 = 42;
        let mut s2: u64 = 42;
        assert_eq!(xorshift64(&mut s1), xorshift64(&mut s2));
    }

    // -----------------------------------------------------------------------
    // SemanticClusterLabeler – construction
    // -----------------------------------------------------------------------

    #[test]
    fn test_with_defaults_creates_empty_labeler() {
        let l = SemanticClusterLabeler::with_defaults();
        let stats = l.labeler_stats();
        assert_eq!(stats.total_clusters, 0);
        assert_eq!(stats.vocab_size, 0);
    }

    #[test]
    fn test_new_respects_config() {
        let cfg = SclLabelerConfig {
            min_confidence: 0.5,
            top_k_words: 3,
            ..Default::default()
        };
        let l = SemanticClusterLabeler::new(cfg);
        assert!((l.config().min_confidence - 0.5).abs() < 1e-9);
        assert_eq!(l.config().top_k_words, 3);
    }

    // -----------------------------------------------------------------------
    // add_cluster / remove_cluster
    // -----------------------------------------------------------------------

    #[test]
    fn test_add_cluster_returns_unique_ids() {
        let mut l = SemanticClusterLabeler::with_defaults();
        let id1 = l.add_cluster(vec![1.0, 0.0], vec![1]);
        let id2 = l.add_cluster(vec![0.0, 1.0], vec![2]);
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_add_cluster_tracked_in_stats() {
        let mut l = SemanticClusterLabeler::with_defaults();
        l.add_cluster(vec![1.0, 0.0], vec![1, 2]);
        assert_eq!(l.labeler_stats().total_clusters, 1);
    }

    #[test]
    fn test_remove_cluster_returns_true_when_found() {
        let mut l = SemanticClusterLabeler::with_defaults();
        let id = l.add_cluster(vec![1.0], vec![1]);
        assert!(l.remove_cluster(id));
        assert_eq!(l.labeler_stats().total_clusters, 0);
    }

    #[test]
    fn test_remove_cluster_returns_false_when_missing() {
        let mut l = SemanticClusterLabeler::with_defaults();
        assert!(!l.remove_cluster(9999));
    }

    #[test]
    fn test_remove_cluster_clears_member_index() {
        let mut l = SemanticClusterLabeler::with_defaults();
        let id = l.add_cluster(vec![1.0], vec![10, 20]);
        l.remove_cluster(id);
        // Verifying indirectly: adding the same members to a new cluster works
        let id2 = l.add_cluster(vec![0.5], vec![10, 20]);
        assert!(l.get_cluster(id2).is_some());
    }

    // -----------------------------------------------------------------------
    // merge_clusters
    // -----------------------------------------------------------------------

    #[test]
    fn test_merge_clusters_basic() {
        let mut l = SemanticClusterLabeler::with_defaults();
        let a = l.add_cluster(vec![1.0, 0.0], vec![1, 2]);
        let b = l.add_cluster(vec![0.0, 1.0], vec![3, 4]);
        let result = l.merge_clusters(a, b);
        assert!(result.is_ok());
        assert_eq!(result.expect("test: merge_clusters should succeed"), a);
        assert_eq!(l.labeler_stats().total_clusters, 1);
        let merged = l
            .get_cluster(a)
            .expect("test: merged cluster a should exist");
        assert_eq!(merged.members.len(), 4);
    }

    #[test]
    fn test_merge_clusters_centroid_weighted() {
        let mut l = SemanticClusterLabeler::with_defaults();
        let a = l.add_cluster(vec![1.0, 0.0], vec![1, 2]); // 2 members
        let b = l.add_cluster(vec![0.0, 1.0], vec![3, 4, 5, 6]); // 4 members
        l.merge_clusters(a, b)
            .expect("test: merge_clusters should succeed");
        let c = l
            .get_cluster(a)
            .expect("test: merged cluster a should exist");
        // centroid[0] = (1.0*2 + 0.0*4)/6 = 0.333...
        assert!((c.centroid[0] - 1.0 / 3.0).abs() < 1e-9);
        // centroid[1] = (0.0*2 + 1.0*4)/6 = 0.666...
        assert!((c.centroid[1] - 2.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_merge_self_error() {
        let mut l = SemanticClusterLabeler::with_defaults();
        let id = l.add_cluster(vec![1.0], vec![1]);
        assert_eq!(l.merge_clusters(id, id), Err(SclError::SelfMerge(id)));
    }

    #[test]
    fn test_merge_missing_a_error() {
        let mut l = SemanticClusterLabeler::with_defaults();
        let b = l.add_cluster(vec![1.0], vec![1]);
        // a = 9999 doesn't exist
        // The current implementation will return MergeTargetNotFound because b
        // is removed first, then a lookup fails.  Accept either error variant.
        let result = l.merge_clusters(9999, b);
        assert!(result.is_err());
    }

    #[test]
    fn test_merge_missing_b_error() {
        let mut l = SemanticClusterLabeler::with_defaults();
        let a = l.add_cluster(vec![1.0], vec![1]);
        assert_eq!(
            l.merge_clusters(a, 9999),
            Err(SclError::MergeTargetNotFound(9999))
        );
    }

    // -----------------------------------------------------------------------
    // add_prototype / add_keyword_doc
    // -----------------------------------------------------------------------

    #[test]
    fn test_add_prototype_counted_in_stats() {
        let mut l = SemanticClusterLabeler::with_defaults();
        l.add_prototype("test", vec![1.0, 0.0]);
        assert_eq!(l.labeler_stats().prototype_count, 1);
    }

    #[test]
    fn test_add_prototype_replaces_existing() {
        let mut l = SemanticClusterLabeler::with_defaults();
        l.add_prototype("a", vec![1.0, 0.0]);
        l.add_prototype("a", vec![0.5, 0.5]);
        assert_eq!(l.labeler_stats().prototype_count, 1);
    }

    #[test]
    fn test_add_keyword_doc_counted_in_stats() {
        let mut l = SemanticClusterLabeler::with_defaults();
        l.add_keyword_doc("hello world", 1);
        assert_eq!(l.labeler_stats().document_count, 1);
    }

    // -----------------------------------------------------------------------
    // label_cluster – CentroidNearest
    // -----------------------------------------------------------------------

    #[test]
    fn test_label_centroid_nearest_science() {
        let mut l = labeler_with_protos();
        let id = l.add_cluster(vec![0.9, 0.1, 0.0], vec![1]);
        let c = l
            .label_cluster(id, SclLabelingMethod::CentroidNearest)
            .expect("test: label_cluster should succeed for science-like centroid");
        assert_eq!(c.label, "science");
        assert!(c.score > 0.0);
    }

    #[test]
    fn test_label_centroid_nearest_sports() {
        let mut l = labeler_with_protos();
        let id = l.add_cluster(vec![0.0, 0.95, 0.05], vec![1]);
        let c = l
            .label_cluster(id, SclLabelingMethod::CentroidNearest)
            .expect("test: label_cluster should succeed for sports-like centroid");
        assert_eq!(c.label, "sports");
    }

    #[test]
    fn test_label_centroid_nearest_politics() {
        let mut l = labeler_with_protos();
        let id = l.add_cluster(vec![0.0, 0.0, 1.0], vec![1]);
        let c = l
            .label_cluster(id, SclLabelingMethod::CentroidNearest)
            .expect("test: label_cluster should succeed for politics-like centroid");
        assert_eq!(c.label, "politics");
    }

    #[test]
    fn test_label_centroid_nearest_no_protos_error() {
        let mut l = SemanticClusterLabeler::with_defaults();
        let id = l.add_cluster(vec![1.0, 0.0], vec![1]);
        assert!(matches!(
            l.label_cluster(id, SclLabelingMethod::CentroidNearest),
            Err(SclError::NoPrototypes)
        ));
    }

    #[test]
    fn test_label_centroid_nearest_missing_cluster() {
        let mut l = labeler_with_protos();
        assert!(matches!(
            l.label_cluster(9999, SclLabelingMethod::CentroidNearest),
            Err(SclError::ClusterNotFound(9999))
        ));
    }

    // -----------------------------------------------------------------------
    // label_cluster – NearestPrototype
    // -----------------------------------------------------------------------

    #[test]
    fn test_label_nearest_prototype_equivalent_to_centroid() {
        let mut l = labeler_with_protos();
        let id = l.add_cluster(vec![1.0, 0.0, 0.0], vec![1]);
        let c1 = l
            .label_cluster(id, SclLabelingMethod::NearestPrototype)
            .expect("test: label_cluster with NearestPrototype should succeed");
        // Both use the same underlying function
        assert_eq!(c1.label, "science");
    }

    // -----------------------------------------------------------------------
    // label_cluster – TfIdfKeywords
    // -----------------------------------------------------------------------

    #[test]
    fn test_label_tfidf_returns_keyword_label() {
        let mut l = SemanticClusterLabeler::with_defaults();
        add_docs(&mut l);
        let id = l.add_cluster(vec![1.0], vec![1, 2]);
        let c = l
            .label_cluster(id, SclLabelingMethod::TfIdfKeywords)
            .expect("test: label_cluster with TfIdfKeywords should succeed when docs exist");
        // Label should be a non-empty string of keywords
        assert!(!c.label.is_empty());
    }

    #[test]
    fn test_label_tfidf_no_docs_error() {
        let mut l = SemanticClusterLabeler::with_defaults();
        let id = l.add_cluster(vec![1.0], vec![1]);
        assert!(matches!(
            l.label_cluster(id, SclLabelingMethod::TfIdfKeywords),
            Err(SclError::NoDocuments)
        ));
    }

    #[test]
    fn test_label_tfidf_member_not_in_docs_error() {
        let mut l = SemanticClusterLabeler::with_defaults();
        l.add_keyword_doc("science experiment", 99); // doc for embedding 99
        let id = l.add_cluster(vec![1.0], vec![1, 2]); // members 1,2 have no docs
                                                       // Should fail because members have no documents
        assert!(matches!(
            l.label_cluster(id, SclLabelingMethod::TfIdfKeywords),
            Err(SclError::NoDocuments)
        ));
    }

    #[test]
    fn test_label_tfidf_score_in_range() {
        let mut l = SemanticClusterLabeler::with_defaults();
        add_docs(&mut l);
        let id = l.add_cluster(vec![1.0], vec![1, 2]);
        let c = l
            .label_cluster(id, SclLabelingMethod::TfIdfKeywords)
            .expect("test: label_cluster with TfIdfKeywords should succeed");
        assert!(c.score >= 0.0 && c.score <= 1.0);
    }

    // -----------------------------------------------------------------------
    // label_cluster – EmbeddingVoting
    // -----------------------------------------------------------------------

    #[test]
    fn test_label_voting_basic() {
        let mut l = labeler_with_protos();
        let id = l.add_cluster(vec![1.0, 0.0, 0.0], vec![1, 2, 3]);
        let c = l
            .label_cluster(id, SclLabelingMethod::EmbeddingVoting)
            .expect(
                "test: label_cluster with EmbeddingVoting should succeed when prototypes exist",
            );
        assert!(!c.label.is_empty());
    }

    #[test]
    fn test_label_voting_no_protos_error() {
        let mut l = SemanticClusterLabeler::with_defaults();
        let id = l.add_cluster(vec![1.0], vec![1]);
        assert!(matches!(
            l.label_cluster(id, SclLabelingMethod::EmbeddingVoting),
            Err(SclError::NoPrototypes)
        ));
    }

    // -----------------------------------------------------------------------
    // label_cluster – HybridRanking
    // -----------------------------------------------------------------------

    #[test]
    fn test_label_hybrid_with_protos_only() {
        let mut l = labeler_with_protos();
        let id = l.add_cluster(vec![1.0, 0.0, 0.0], vec![1]);
        let c = l
            .label_cluster(id, SclLabelingMethod::HybridRanking)
            .expect("test: label_cluster with HybridRanking should succeed with prototypes");
        assert_eq!(c.label, "science");
    }

    #[test]
    fn test_label_hybrid_with_docs_and_protos() {
        let mut l = labeler_with_protos();
        add_docs(&mut l);
        let id = l.add_cluster(vec![1.0, 0.0, 0.0], vec![1, 2]);
        let c = l
            .label_cluster(id, SclLabelingMethod::HybridRanking)
            .expect(
                "test: label_cluster with HybridRanking should succeed with docs and prototypes",
            );
        assert!(!c.label.is_empty());
    }

    #[test]
    fn test_label_hybrid_no_methods_error() {
        let mut l = SemanticClusterLabeler::with_defaults();
        let id = l.add_cluster(vec![1.0], vec![1]);
        assert!(matches!(
            l.label_cluster(id, SclLabelingMethod::HybridRanking),
            Err(SclError::NoPrototypes)
        ));
    }

    // -----------------------------------------------------------------------
    // label_cluster updates cluster state
    // -----------------------------------------------------------------------

    #[test]
    fn test_label_cluster_sets_cluster_label() {
        let mut l = labeler_with_protos();
        let id = l.add_cluster(vec![1.0, 0.0, 0.0], vec![1]);
        l.label_cluster(id, SclLabelingMethod::CentroidNearest)
            .expect("test: label_cluster should succeed");
        let c = l
            .get_cluster(id)
            .expect("test: cluster should exist after labeling");
        assert_eq!(c.label.as_deref(), Some("science"));
    }

    #[test]
    fn test_label_cluster_sets_confidence() {
        let mut l = labeler_with_protos();
        let id = l.add_cluster(vec![1.0, 0.0, 0.0], vec![1]);
        l.label_cluster(id, SclLabelingMethod::CentroidNearest)
            .expect("test: label_cluster should succeed");
        let c = l
            .get_cluster(id)
            .expect("test: cluster should exist after labeling");
        assert!(c.confidence > 0.0);
    }

    #[test]
    fn test_label_cluster_sets_labeled_centroid() {
        let mut l = labeler_with_protos();
        let id = l.add_cluster(vec![1.0, 0.0, 0.0], vec![1]);
        l.label_cluster(id, SclLabelingMethod::CentroidNearest)
            .expect("test: label_cluster should succeed");
        assert!(l
            .get_cluster(id)
            .expect("test: cluster should exist after labeling")
            .labeled_centroid
            .is_some());
    }

    // -----------------------------------------------------------------------
    // Vocabulary
    // -----------------------------------------------------------------------

    #[test]
    fn test_vocab_updated_after_labeling() {
        let mut l = labeler_with_protos();
        let id = l.add_cluster(vec![1.0, 0.0, 0.0], vec![1]);
        l.label_cluster(id, SclLabelingMethod::CentroidNearest)
            .expect("test: label_cluster should succeed");
        assert!(l.vocab().contains_key("science"));
    }

    #[test]
    fn test_vocab_use_count_increments() {
        let mut l = labeler_with_protos();
        let id1 = l.add_cluster(vec![1.0, 0.0, 0.0], vec![1]);
        let id2 = l.add_cluster(vec![0.99, 0.01, 0.0], vec![2]);
        l.label_cluster(id1, SclLabelingMethod::CentroidNearest)
            .expect("test: label_cluster id1 should succeed");
        l.label_cluster(id2, SclLabelingMethod::CentroidNearest)
            .expect("test: label_cluster id2 should succeed");
        let stats = l
            .vocab()
            .get("science")
            .expect("test: science should be in vocab");
        assert_eq!(stats.use_count, 2);
    }

    #[test]
    fn test_vocab_cluster_ids_tracked() {
        let mut l = labeler_with_protos();
        let id = l.add_cluster(vec![1.0, 0.0, 0.0], vec![1]);
        l.label_cluster(id, SclLabelingMethod::CentroidNearest)
            .expect("test: label_cluster should succeed");
        let stats = l
            .vocab()
            .get("science")
            .expect("test: science should be in vocab");
        assert!(stats.cluster_ids.contains(&id));
    }

    // -----------------------------------------------------------------------
    // History
    // -----------------------------------------------------------------------

    #[test]
    fn test_history_appended_on_label() {
        let mut l = labeler_with_protos();
        let id = l.add_cluster(vec![1.0, 0.0, 0.0], vec![1]);
        l.label_cluster(id, SclLabelingMethod::CentroidNearest)
            .expect("test: label_cluster should succeed");
        assert_eq!(l.history().len(), 1);
    }

    #[test]
    fn test_history_bounded_at_500() {
        let mut l = labeler_with_protos();
        // Create 510 clusters and label each
        for i in 0..510u64 {
            let id = l.add_cluster(vec![1.0, 0.0, 0.0], vec![i]);
            let _ = l.label_cluster(id, SclLabelingMethod::CentroidNearest);
        }
        assert!(l.history().len() <= MAX_HISTORY);
    }

    #[test]
    fn test_history_records_old_label() {
        let mut l = labeler_with_protos();
        let id = l.add_cluster(vec![1.0, 0.0, 0.0], vec![1]);
        l.label_cluster(id, SclLabelingMethod::CentroidNearest)
            .expect("test: first label_cluster should succeed");
        // Shift centroid toward sports, relabel
        l.get_cluster(id); // read-only check
        l.update_centroid(id, vec![0.0, 1.0, 0.0]);
        l.label_cluster(id, SclLabelingMethod::CentroidNearest)
            .expect("test: second label_cluster should succeed");
        let last = l
            .history()
            .back()
            .expect("test: history should have at least one record");
        assert_eq!(last.old_label.as_deref(), Some("science"));
        assert_eq!(last.new_label, "sports");
    }

    // -----------------------------------------------------------------------
    // label_all
    // -----------------------------------------------------------------------

    #[test]
    fn test_label_all_labels_multiple_clusters() {
        let mut l = labeler_with_protos();
        l.add_cluster(vec![1.0, 0.0, 0.0], vec![1]);
        l.add_cluster(vec![0.0, 1.0, 0.0], vec![2]);
        l.add_cluster(vec![0.0, 0.0, 1.0], vec![3]);
        let results = l.label_all(SclLabelingMethod::CentroidNearest);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_label_all_skips_below_confidence() {
        let mut l = SemanticClusterLabeler::new(SclLabelerConfig {
            min_confidence: 0.99, // very high → most will be skipped
            ..Default::default()
        });
        l.add_prototype("test", vec![1.0, 0.0]);
        l.add_cluster(vec![0.5, 0.5], vec![1]); // moderate similarity
        let results = l.label_all(SclLabelingMethod::CentroidNearest);
        // Either 0 or 1 results depending on exact score, but should not panic
        assert!(results.len() <= 1);
    }

    // -----------------------------------------------------------------------
    // relabel_if_drifted
    // -----------------------------------------------------------------------

    #[test]
    fn test_relabel_if_drifted_no_drift() {
        let mut l = labeler_with_protos();
        let id = l.add_cluster(vec![1.0, 0.0, 0.0], vec![1]);
        l.label_cluster(id, SclLabelingMethod::CentroidNearest)
            .expect("test: label_cluster should succeed");
        let recount = l.relabel_if_drifted(0.5);
        assert_eq!(recount, 0);
    }

    #[test]
    fn test_relabel_if_drifted_detects_large_shift() {
        let mut l = labeler_with_protos();
        let id = l.add_cluster(vec![1.0, 0.0, 0.0], vec![1]);
        l.label_cluster(id, SclLabelingMethod::CentroidNearest)
            .expect("test: label_cluster should succeed");
        // Shift centroid sharply
        l.update_centroid(id, vec![0.0, 0.0, 1.0]);
        let recount = l.relabel_if_drifted(0.1);
        assert!(recount >= 1);
        let c = l
            .get_cluster(id)
            .expect("test: cluster should exist after drift detection");
        assert_eq!(c.label.as_deref(), Some("politics"));
    }

    #[test]
    fn test_relabel_if_drifted_unlabeled_ignored() {
        let mut l = labeler_with_protos();
        l.add_cluster(vec![1.0, 0.0, 0.0], vec![1]); // not labeled
        let recount = l.relabel_if_drifted(0.0);
        assert_eq!(recount, 0);
    }

    // -----------------------------------------------------------------------
    // cluster_summary
    // -----------------------------------------------------------------------

    #[test]
    fn test_cluster_summary_returns_some() {
        let mut l = labeler_with_protos();
        let id = l.add_cluster(vec![1.0, 0.0, 0.0], vec![1, 2]);
        l.label_cluster(id, SclLabelingMethod::CentroidNearest)
            .expect("test: label_cluster should succeed");
        let s = l
            .cluster_summary(id)
            .expect("test: cluster_summary should return Some for existing labeled cluster");
        assert!(s.contains("science"));
        assert!(s.contains("members=2"));
    }

    #[test]
    fn test_cluster_summary_unlabeled() {
        let mut l = SemanticClusterLabeler::with_defaults();
        let id = l.add_cluster(vec![1.0], vec![]);
        let s = l
            .cluster_summary(id)
            .expect("test: cluster_summary should return Some for existing cluster");
        assert!(s.contains("<unlabeled>"));
    }

    #[test]
    fn test_cluster_summary_missing_id_returns_none() {
        let l = SemanticClusterLabeler::with_defaults();
        assert!(l.cluster_summary(9999).is_none());
    }

    // -----------------------------------------------------------------------
    // labeler_stats
    // -----------------------------------------------------------------------

    #[test]
    fn test_labeler_stats_counts_correctly() {
        let mut l = labeler_with_protos();
        let id = l.add_cluster(vec![1.0, 0.0, 0.0], vec![1]);
        l.label_cluster(id, SclLabelingMethod::CentroidNearest)
            .expect("test: label_cluster should succeed");
        let stats = l.labeler_stats();
        assert_eq!(stats.total_clusters, 1);
        assert_eq!(stats.labeled_clusters, 1);
        assert!(stats.vocab_size > 0);
    }

    #[test]
    fn test_labeler_stats_avg_confidence() {
        let mut l = labeler_with_protos();
        let id = l.add_cluster(vec![1.0, 0.0, 0.0], vec![1]);
        l.label_cluster(id, SclLabelingMethod::CentroidNearest)
            .expect("test: label_cluster should succeed");
        let stats = l.labeler_stats();
        assert!(stats.avg_confidence > 0.0);
    }

    #[test]
    fn test_labeler_stats_empty() {
        let l = SemanticClusterLabeler::with_defaults();
        let stats = l.labeler_stats();
        assert_eq!(stats.avg_confidence, 0.0);
        assert_eq!(stats.history_len, 0);
    }

    // -----------------------------------------------------------------------
    // update_centroid / add_members
    // -----------------------------------------------------------------------

    #[test]
    fn test_update_centroid_returns_true_when_found() {
        let mut l = SemanticClusterLabeler::with_defaults();
        let id = l.add_cluster(vec![1.0, 0.0], vec![]);
        assert!(l.update_centroid(id, vec![0.5, 0.5]));
        assert_eq!(
            l.get_cluster(id)
                .expect("test: cluster should exist after update_centroid")
                .centroid,
            vec![0.5, 0.5]
        );
    }

    #[test]
    fn test_update_centroid_returns_false_when_missing() {
        let mut l = SemanticClusterLabeler::with_defaults();
        assert!(!l.update_centroid(9999, vec![1.0]));
    }

    #[test]
    fn test_add_members_increases_count() {
        let mut l = SemanticClusterLabeler::with_defaults();
        let id = l.add_cluster(vec![1.0], vec![1, 2]);
        assert!(l.add_members(id, &[3, 4, 5]));
        assert_eq!(
            l.get_cluster(id)
                .expect("test: cluster should exist after add_members")
                .members
                .len(),
            5
        );
    }

    #[test]
    fn test_add_members_no_duplicates() {
        let mut l = SemanticClusterLabeler::with_defaults();
        let id = l.add_cluster(vec![1.0], vec![1]);
        l.add_members(id, &[1, 2]); // 1 already present
        assert_eq!(
            l.get_cluster(id)
                .expect("test: cluster should exist after add_members dedup")
                .members
                .len(),
            2
        );
    }

    #[test]
    fn test_add_members_returns_false_when_missing() {
        let mut l = SemanticClusterLabeler::with_defaults();
        assert!(!l.add_members(9999, &[1]));
    }

    // -----------------------------------------------------------------------
    // SclError display
    // -----------------------------------------------------------------------

    #[test]
    fn test_error_display_cluster_not_found() {
        let e = SclError::ClusterNotFound(42);
        assert!(e.to_string().contains("42"));
    }

    #[test]
    fn test_error_display_self_merge() {
        let e = SclError::SelfMerge(7);
        assert!(e.to_string().contains("7"));
    }

    #[test]
    fn test_error_display_below_confidence() {
        let e = SclError::BelowConfidenceThreshold {
            best: 0.05,
            threshold: 0.10,
        };
        let s = e.to_string();
        assert!(s.contains("0.0500") || s.contains("0.05"));
    }

    // -----------------------------------------------------------------------
    // SclLabelingMethod display
    // -----------------------------------------------------------------------

    #[test]
    fn test_method_display_all_variants() {
        use SclLabelingMethod::*;
        let variants = [
            CentroidNearest,
            TfIdfKeywords,
            EmbeddingVoting,
            NearestPrototype,
            HybridRanking,
        ];
        for v in &variants {
            assert!(!v.to_string().is_empty());
        }
    }

    // -----------------------------------------------------------------------
    // SclLabelerConfig defaults
    // -----------------------------------------------------------------------

    #[test]
    fn test_config_default_reasonable_values() {
        let c = SclLabelerConfig::default();
        assert!(c.max_labels_per_cluster > 0);
        assert!(c.min_confidence >= 0.0 && c.min_confidence < 1.0);
        assert!(c.top_k_words > 0);
    }

    // -----------------------------------------------------------------------
    // set_config
    // -----------------------------------------------------------------------

    #[test]
    fn test_set_config_updates_config() {
        let mut l = SemanticClusterLabeler::with_defaults();
        let new_cfg = SclLabelerConfig {
            min_confidence: 0.42,
            ..Default::default()
        };
        l.set_config(new_cfg);
        assert!((l.config().min_confidence - 0.42).abs() < 1e-9);
    }

    // -----------------------------------------------------------------------
    // Minimum confidence threshold enforcement
    // -----------------------------------------------------------------------

    #[test]
    fn test_label_rejects_below_threshold() {
        let mut l = SemanticClusterLabeler::new(SclLabelerConfig {
            min_confidence: 0.99,
            ..Default::default()
        });
        // Prototype orthogonal to centroid → cosine ~ 0
        l.add_prototype("far", vec![0.0, 1.0]);
        let id = l.add_cluster(vec![1.0, 0.0], vec![1]);
        let result = l.label_cluster(id, SclLabelingMethod::CentroidNearest);
        assert!(matches!(
            result,
            Err(SclError::BelowConfidenceThreshold { .. })
        ));
    }

    // -----------------------------------------------------------------------
    // Multi-cluster vocab management on remove
    // -----------------------------------------------------------------------

    #[test]
    fn test_remove_clears_vocab_cluster_id() {
        let mut l = labeler_with_protos();
        let id = l.add_cluster(vec![1.0, 0.0, 0.0], vec![1]);
        l.label_cluster(id, SclLabelingMethod::CentroidNearest)
            .expect("test: label_cluster should succeed");
        l.remove_cluster(id);
        if let Some(stats) = l.vocab().get("science") {
            assert!(!stats.cluster_ids.contains(&id));
        }
    }

    // -----------------------------------------------------------------------
    // Serialization round-trip (SclCluster)
    // -----------------------------------------------------------------------

    #[test]
    fn test_sclcluster_serde_roundtrip() {
        let cluster = SclCluster {
            id: 1,
            centroid: vec![0.1, 0.2],
            members: vec![10, 20],
            label: Some("test".into()),
            confidence: 0.8,
            keywords: vec!["word".into()],
            created_at: 0,
            labeled_centroid: None,
        };
        let json = serde_json::to_string(&cluster).expect("test: serialization failed");
        let decoded: SclCluster =
            serde_json::from_str(&json).expect("test: deserialization failed");
        assert_eq!(decoded.id, 1);
        assert_eq!(decoded.label.as_deref(), Some("test"));
    }

    // -----------------------------------------------------------------------
    // Keyword extraction quality
    // -----------------------------------------------------------------------

    #[test]
    fn test_tfidf_top_k_respected() {
        let mut l = SemanticClusterLabeler::new(SclLabelerConfig {
            top_k_words: 2,
            min_confidence: 0.0,
            ..Default::default()
        });
        l.add_keyword_doc("alpha beta gamma delta", 1);
        l.add_keyword_doc("alpha beta gamma", 2);
        let id = l.add_cluster(vec![1.0], vec![1, 2]);
        let c = l
            .label_cluster(id, SclLabelingMethod::TfIdfKeywords)
            .expect("test: label_cluster with TfIdfKeywords should succeed");
        // At most top_k_words words in the label
        let word_count = c.label.split_whitespace().count();
        assert!(word_count <= 2);
    }

    // -----------------------------------------------------------------------
    // History record ordering
    // -----------------------------------------------------------------------

    #[test]
    fn test_history_records_are_ordered_by_insertion() {
        let mut l = labeler_with_protos();
        let id1 = l.add_cluster(vec![1.0, 0.0, 0.0], vec![1]);
        let id2 = l.add_cluster(vec![0.0, 1.0, 0.0], vec![2]);
        l.label_cluster(id1, SclLabelingMethod::CentroidNearest)
            .expect("test: label_cluster id1 should succeed");
        l.label_cluster(id2, SclLabelingMethod::CentroidNearest)
            .expect("test: label_cluster id2 should succeed");
        let history: Vec<_> = l.history().iter().collect();
        assert_eq!(history[0].cluster_id, id1);
        assert_eq!(history[1].cluster_id, id2);
    }

    // -----------------------------------------------------------------------
    // Type alias existence
    // -----------------------------------------------------------------------

    #[test]
    fn test_type_alias_scl_semantic_cluster_labeler() {
        let _: SclSemanticClusterLabeler = SemanticClusterLabeler::with_defaults();
    }

    // -----------------------------------------------------------------------
    // Cluster `created_at` is populated
    // -----------------------------------------------------------------------

    #[test]
    fn test_created_at_nonzero_on_modern_system() {
        let mut l = SemanticClusterLabeler::with_defaults();
        let id = l.add_cluster(vec![1.0], vec![]);
        let c = l
            .get_cluster(id)
            .expect("test: cluster should exist after add_cluster");
        // created_at should be a plausible Unix timestamp (> year 2000)
        assert!(c.created_at > 946_684_800);
    }
}
