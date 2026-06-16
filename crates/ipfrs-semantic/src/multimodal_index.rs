//! Unified multimodal index for cross-modal similarity search.
//!
//! This module implements `MultiModalIndex`, a production-grade index supporting
//! text, image, audio, video, and structured data embeddings with cross-modal
//! projection and flexible fusion strategies.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Modality
// ---------------------------------------------------------------------------

/// Supported content modalities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Modality {
    /// Text embeddings (e.g., from BERT, GPT).
    Text,
    /// Image embeddings (e.g., from ResNet, CLIP).
    Image,
    /// Audio embeddings (e.g., from Wav2Vec, CLAP).
    Audio,
    /// Video embeddings (e.g., from VideoMAE).
    Video,
    /// Structured/tabular data embeddings.
    Structured,
}

// ---------------------------------------------------------------------------
// ModalityEmbedding
// ---------------------------------------------------------------------------

/// An embedding vector associated with a specific modality.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModalityEmbedding {
    /// The modality this embedding belongs to.
    pub modality: Modality,
    /// Raw embedding vector (f64 for numeric precision).
    pub embedding: Vec<f64>,
    /// Declared dimensionality — must equal `embedding.len()`.
    pub dims: usize,
}

impl ModalityEmbedding {
    /// Create a new `ModalityEmbedding`, recording `dims` from the vector length.
    pub fn new(modality: Modality, embedding: Vec<f64>) -> Self {
        let dims = embedding.len();
        Self {
            modality,
            embedding,
            dims,
        }
    }
}

// ---------------------------------------------------------------------------
// MultiModalDocument
// ---------------------------------------------------------------------------

/// A document containing one or more modality embeddings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MultiModalDocument {
    /// Unique document identifier.
    pub id: String,
    /// Per-modality embeddings (at least one required).
    pub modalities: HashMap<Modality, ModalityEmbedding>,
    /// Arbitrary string metadata for post-filtering.
    pub metadata: HashMap<String, String>,
    /// Unix-epoch timestamp of indexing (seconds).
    pub indexed_at: u64,
}

impl MultiModalDocument {
    /// Construct a new document.  Returns `Err(MmiError::EmptyDocument)` if `modalities` is empty.
    pub fn new(
        id: impl Into<String>,
        modalities: HashMap<Modality, ModalityEmbedding>,
        metadata: HashMap<String, String>,
        indexed_at: u64,
    ) -> Result<Self, MmiError> {
        if modalities.is_empty() {
            return Err(MmiError::EmptyDocument);
        }
        Ok(Self {
            id: id.into(),
            modalities,
            metadata,
            indexed_at,
        })
    }
}

// ---------------------------------------------------------------------------
// CrossModalQuery
// ---------------------------------------------------------------------------

/// Query for cross-modal similarity search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossModalQuery {
    /// The modality the query embedding originates from.
    pub query_modality: Modality,
    /// Query vector.
    pub query_embedding: Vec<f64>,
    /// Which document modalities to search against.
    pub target_modalities: Vec<Modality>,
    /// Maximum number of results to return.
    pub top_k: usize,
    /// Minimum acceptable combined score (0.0 – 1.0).
    pub min_similarity: f64,
    /// Metadata key-value filters (all must match).
    pub filters: HashMap<String, String>,
}

impl CrossModalQuery {
    /// Construct a query with no filters and default similarity threshold.
    pub fn new(
        query_modality: Modality,
        query_embedding: Vec<f64>,
        target_modalities: Vec<Modality>,
        top_k: usize,
    ) -> Self {
        Self {
            query_modality,
            query_embedding,
            target_modalities,
            top_k,
            min_similarity: 0.0,
            filters: HashMap::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// CrossModalResult
// ---------------------------------------------------------------------------

/// A single search result with per-modality scores.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossModalResult {
    /// Document identifier.
    pub doc_id: String,
    /// Score for each matched modality.
    pub scores: HashMap<Modality, f64>,
    /// Fused combined score used for ranking.
    pub combined_score: f64,
    /// 1-based rank in the result list.
    pub rank: usize,
}

// ---------------------------------------------------------------------------
// FusionStrategy
// ---------------------------------------------------------------------------

/// Strategy for combining per-modality scores into a single combined score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FusionStrategy {
    /// Take the maximum score across all matched modalities.
    MaxScore,
    /// Take the unweighted mean of all matched modality scores.
    MeanScore,
    /// Weighted mean; missing modalities are excluded and weights renormalized.
    WeightedFusion {
        /// Per-modality weights (need not sum to 1; normalized internally).
        weights: HashMap<Modality, f64>,
    },
    /// Use Text score if available; otherwise fall back to `MaxScore`.
    TextPrimary,
}

impl FusionStrategy {
    /// Fuse a map of modality → score into a single scalar.
    /// Returns `None` if `scores` is empty.
    pub fn fuse(&self, scores: &HashMap<Modality, f64>) -> Option<f64> {
        if scores.is_empty() {
            return None;
        }
        match self {
            FusionStrategy::MaxScore => scores.values().copied().reduce(f64::max),
            FusionStrategy::MeanScore => {
                let sum: f64 = scores.values().sum();
                Some(sum / scores.len() as f64)
            }
            FusionStrategy::WeightedFusion { weights } => {
                let mut weighted_sum = 0.0_f64;
                let mut weight_total = 0.0_f64;
                for (modality, &score) in scores {
                    let w = weights.get(modality).copied().unwrap_or(0.0);
                    if w > 0.0 {
                        weighted_sum += score * w;
                        weight_total += w;
                    }
                }
                if weight_total == 0.0 {
                    // Fall back to mean when no weights match
                    let sum: f64 = scores.values().sum();
                    Some(sum / scores.len() as f64)
                } else {
                    Some(weighted_sum / weight_total)
                }
            }
            FusionStrategy::TextPrimary => {
                if let Some(&text_score) = scores.get(&Modality::Text) {
                    Some(text_score)
                } else {
                    scores.values().copied().reduce(f64::max)
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// MultiModalIndexConfig
// ---------------------------------------------------------------------------

/// Configuration for `MultiModalIndex`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiModalIndexConfig {
    /// Strategy for fusing per-modality scores.
    pub fusion_strategy: FusionStrategy,
    /// Target dimensionality for cross-modal projection output.
    pub cross_modal_dims: usize,
    /// If `true`, normalize all query and document embeddings to unit length
    /// before computing similarities.
    pub normalize_embeddings: bool,
}

impl Default for MultiModalIndexConfig {
    fn default() -> Self {
        Self {
            fusion_strategy: FusionStrategy::MeanScore,
            cross_modal_dims: 256,
            normalize_embeddings: true,
        }
    }
}

// ---------------------------------------------------------------------------
// MmiError
// ---------------------------------------------------------------------------

/// Error type for `MultiModalIndex` operations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MmiError {
    /// A document with the given ID already exists.
    DocumentAlreadyExists(String),
    /// No document with the given ID exists.
    DocumentNotFound(String),
    /// A document must contain at least one modality embedding.
    EmptyDocument,
    /// Cross-modal projection matrix dimensions are incompatible.
    ProjectionDimsMismatch,
    /// The query targeted modalities not found in any document.
    NoMatchingModality,
}

impl std::fmt::Display for MmiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MmiError::DocumentAlreadyExists(id) => {
                write!(f, "Document already exists: {id}")
            }
            MmiError::DocumentNotFound(id) => {
                write!(f, "Document not found: {id}")
            }
            MmiError::EmptyDocument => write!(f, "Document has no modality embeddings"),
            MmiError::ProjectionDimsMismatch => {
                write!(f, "Projection matrix dimensions do not match")
            }
            MmiError::NoMatchingModality => {
                write!(f, "No matching modality found in any document")
            }
        }
    }
}

impl std::error::Error for MmiError {}

// ---------------------------------------------------------------------------
// MmiStats
// ---------------------------------------------------------------------------

/// Aggregate statistics for a `MultiModalIndex`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MmiStats {
    /// Total number of indexed documents.
    pub doc_count: usize,
    /// Number of documents that contain each modality.
    pub modality_counts: HashMap<Modality, usize>,
    /// Average number of modalities per document.
    pub avg_modalities_per_doc: f64,
    /// Cumulative number of `search` calls since index creation.
    pub total_searches: u64,
}

// ---------------------------------------------------------------------------
// MultiModalIndex
// ---------------------------------------------------------------------------

/// A unified index for multimodal content supporting cross-modal similarity search.
///
/// # Overview
///
/// `MultiModalIndex` stores `MultiModalDocument` entries — each carrying one or
/// more modality embeddings — and provides:
///
/// * **Same-modality search** via `same_modality_search`
/// * **Cross-modal search** via `search` (projects query into each target modality
///   space when a projection matrix is registered)
/// * **Configurable score fusion** via `FusionStrategy`
/// * **Metadata filtering** on result sets
pub struct MultiModalIndex {
    /// Index configuration (fusion strategy, dims, normalization).
    pub config: MultiModalIndexConfig,
    /// Indexed documents keyed by their string ID.
    pub documents: HashMap<String, MultiModalDocument>,
    /// Learned cross-modal projection matrices.
    /// Key `(from, to)` maps query space `from` into document space `to`.
    pub projection_matrices: HashMap<(Modality, Modality), Vec<Vec<f64>>>,
    /// Atomic counter incremented on every `search` call.
    total_searches: Arc<AtomicU64>,
}

impl MultiModalIndex {
    /// Create a new empty `MultiModalIndex` with the given configuration.
    pub fn new(config: MultiModalIndexConfig) -> Self {
        Self {
            config,
            documents: HashMap::new(),
            projection_matrices: HashMap::new(),
            total_searches: Arc::new(AtomicU64::new(0)),
        }
    }

    // -----------------------------------------------------------------------
    // Document management
    // -----------------------------------------------------------------------

    /// Index a new document.
    ///
    /// Returns `Err(MmiError::DocumentAlreadyExists)` if a document with the
    /// same ID is already present.
    pub fn add_document(&mut self, doc: MultiModalDocument) -> Result<(), MmiError> {
        if self.documents.contains_key(&doc.id) {
            return Err(MmiError::DocumentAlreadyExists(doc.id.clone()));
        }
        self.documents.insert(doc.id.clone(), doc);
        Ok(())
    }

    /// Remove a document by ID.  Returns `true` if the document existed.
    pub fn remove_document(&mut self, doc_id: &str) -> bool {
        self.documents.remove(doc_id).is_some()
    }

    /// Total number of indexed documents.
    pub fn doc_count(&self) -> usize {
        self.documents.len()
    }

    /// Count the number of documents that contain each modality.
    pub fn modality_coverage(&self) -> HashMap<Modality, usize> {
        let mut counts: HashMap<Modality, usize> = HashMap::new();
        for doc in self.documents.values() {
            for modality in doc.modalities.keys() {
                *counts.entry(*modality).or_insert(0) += 1;
            }
        }
        counts
    }

    // -----------------------------------------------------------------------
    // Projection management
    // -----------------------------------------------------------------------

    /// Register a cross-modal projection matrix from modality `from` to `to`.
    ///
    /// `matrix[i][j]` is the weight from input dimension `j` to output dimension `i`.
    /// So `matrix` must be `cross_modal_dims × from_dims`.
    ///
    /// Returns `Err(MmiError::ProjectionDimsMismatch)` if the matrix has zero
    /// rows or inconsistent column counts.
    pub fn add_projection(
        &mut self,
        from: Modality,
        to: Modality,
        matrix: Vec<Vec<f64>>,
    ) -> Result<(), MmiError> {
        if matrix.is_empty() {
            return Err(MmiError::ProjectionDimsMismatch);
        }
        let expected_cols = matrix[0].len();
        if expected_cols == 0 {
            return Err(MmiError::ProjectionDimsMismatch);
        }
        for row in &matrix {
            if row.len() != expected_cols {
                return Err(MmiError::ProjectionDimsMismatch);
            }
        }
        self.projection_matrices.insert((from, to), matrix);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Math helpers
    // -----------------------------------------------------------------------

    /// Compute cosine similarity between two equal-length vectors.
    /// Returns 0.0 for zero-norm vectors.
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

    /// Project an embedding vector through a matrix.
    ///
    /// `result[i] = Σ_j matrix[i][j] * embedding[j]`
    ///
    /// Output length equals the number of rows in `matrix`.
    pub fn project(embedding: &[f64], matrix: &[Vec<f64>]) -> Vec<f64> {
        matrix
            .iter()
            .map(|row| {
                row.iter()
                    .zip(embedding.iter())
                    .map(|(w, x)| w * x)
                    .sum::<f64>()
            })
            .collect()
    }

    /// L2-normalize a vector.  Returns the zero vector if the norm is zero.
    pub fn l2_normalize(v: &[f64]) -> Vec<f64> {
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm == 0.0 {
            return v.to_vec();
        }
        v.iter().map(|x| x / norm).collect()
    }

    // -----------------------------------------------------------------------
    // Search internals
    // -----------------------------------------------------------------------

    /// Prepare the query embedding: optionally normalize it.
    fn prepare_query(&self, embedding: &[f64]) -> Vec<f64> {
        if self.config.normalize_embeddings {
            Self::l2_normalize(embedding)
        } else {
            embedding.to_vec()
        }
    }

    /// Prepare a document embedding: optionally normalize it.
    fn prepare_doc_embedding(&self, embedding: &[f64]) -> Vec<f64> {
        if self.config.normalize_embeddings {
            Self::l2_normalize(embedding)
        } else {
            embedding.to_vec()
        }
    }

    /// Check whether all metadata filters in the query match the document.
    fn matches_filters(doc: &MultiModalDocument, filters: &HashMap<String, String>) -> bool {
        for (key, value) in filters {
            match doc.metadata.get(key) {
                Some(doc_val) if doc_val == value => {}
                _ => return false,
            }
        }
        true
    }

    /// Compute per-modality scores for one document against a cross-modal query.
    ///
    /// For each `target_modality` in `query.target_modalities`:
    /// - If the document has that modality AND `query_modality == target_modality`:
    ///   compare directly using cosine similarity.
    /// - If `query_modality != target_modality`:
    ///   apply projection `(query_modality → target_modality)` if available;
    ///   otherwise skip.
    fn score_document(
        &self,
        query: &CrossModalQuery,
        prepared_query: &[f64],
        doc: &MultiModalDocument,
    ) -> HashMap<Modality, f64> {
        let mut scores: HashMap<Modality, f64> = HashMap::new();

        for &target in &query.target_modalities {
            let Some(doc_emb) = doc.modalities.get(&target) else {
                continue;
            };

            let prepared_doc = self.prepare_doc_embedding(&doc_emb.embedding);

            if query.query_modality == target {
                // Same-modality: direct cosine similarity.
                if prepared_query.len() == prepared_doc.len() {
                    let sim = Self::cosine_similarity(prepared_query, &prepared_doc);
                    scores.insert(target, sim);
                }
            } else {
                // Cross-modal: project query into target space if we have a matrix.
                if let Some(matrix) = self
                    .projection_matrices
                    .get(&(query.query_modality, target))
                {
                    let projected = Self::project(prepared_query, matrix);
                    // Only score if projected dims match document dims.
                    if projected.len() == prepared_doc.len() {
                        let sim = Self::cosine_similarity(&projected, &prepared_doc);
                        scores.insert(target, sim);
                    }
                }
                // No projection available → skip this modality for this document.
            }
        }

        scores
    }

    // -----------------------------------------------------------------------
    // Public search API
    // -----------------------------------------------------------------------

    /// Search the index using the given cross-modal query.
    ///
    /// For each document:
    /// 1. Compute per-modality cosine similarities (with optional cross-modal projection).
    /// 2. Fuse scores using `config.fusion_strategy`.
    /// 3. Filter by `query.min_similarity` and metadata filters.
    /// 4. Return the top-`query.top_k` results sorted by combined score descending,
    ///    with 1-based ranks assigned.
    pub fn search(&self, query: &CrossModalQuery) -> Vec<CrossModalResult> {
        self.total_searches.fetch_add(1, Ordering::Relaxed);

        let prepared_query = self.prepare_query(&query.query_embedding);

        let mut results: Vec<CrossModalResult> = self
            .documents
            .values()
            .filter(|doc| Self::matches_filters(doc, &query.filters))
            .filter_map(|doc| {
                let scores = self.score_document(query, &prepared_query, doc);
                if scores.is_empty() {
                    return None;
                }
                let combined_score = self.config.fusion_strategy.fuse(&scores)?;
                if combined_score < query.min_similarity {
                    return None;
                }
                Some(CrossModalResult {
                    doc_id: doc.id.clone(),
                    scores,
                    combined_score,
                    rank: 0, // assigned below
                })
            })
            .collect();

        // Sort descending by combined_score; use doc_id as tiebreaker for determinism.
        results.sort_by(|a, b| {
            b.combined_score
                .partial_cmp(&a.combined_score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.doc_id.cmp(&b.doc_id))
        });

        results.truncate(query.top_k);

        // Assign 1-based ranks.
        for (i, result) in results.iter_mut().enumerate() {
            result.rank = i + 1;
        }

        results
    }

    /// Convenience search restricted to `query.query_modality` only, with no
    /// cross-modal projection.
    ///
    /// This is equivalent to calling `search` with
    /// `target_modalities = [query_modality]` and is more efficient because it
    /// skips the projection lookup entirely.
    pub fn same_modality_search(&self, query: &CrossModalQuery) -> Vec<CrossModalResult> {
        self.total_searches.fetch_add(1, Ordering::Relaxed);

        let prepared_query = self.prepare_query(&query.query_embedding);
        let target = query.query_modality;

        let mut results: Vec<CrossModalResult> = self
            .documents
            .values()
            .filter(|doc| Self::matches_filters(doc, &query.filters))
            .filter_map(|doc| {
                let doc_emb = doc.modalities.get(&target)?;
                let prepared_doc = self.prepare_doc_embedding(&doc_emb.embedding);
                if prepared_query.len() != prepared_doc.len() {
                    return None;
                }
                let sim = Self::cosine_similarity(&prepared_query, &prepared_doc);
                if sim < query.min_similarity {
                    return None;
                }
                let mut scores = HashMap::new();
                scores.insert(target, sim);
                Some(CrossModalResult {
                    doc_id: doc.id.clone(),
                    scores,
                    combined_score: sim,
                    rank: 0,
                })
            })
            .collect();

        results.sort_by(|a, b| {
            b.combined_score
                .partial_cmp(&a.combined_score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.doc_id.cmp(&b.doc_id))
        });

        results.truncate(query.top_k);

        for (i, result) in results.iter_mut().enumerate() {
            result.rank = i + 1;
        }

        results
    }

    // -----------------------------------------------------------------------
    // Stats
    // -----------------------------------------------------------------------

    /// Return aggregate statistics for this index.
    pub fn stats(&self) -> MmiStats {
        let doc_count = self.documents.len();
        let modality_counts = self.modality_coverage();
        let total_mod: usize = self.documents.values().map(|d| d.modalities.len()).sum();
        let avg_modalities_per_doc = if doc_count == 0 {
            0.0
        } else {
            total_mod as f64 / doc_count as f64
        };
        MmiStats {
            doc_count,
            modality_counts,
            avg_modalities_per_doc,
            total_searches: self.total_searches.load(Ordering::Relaxed),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::multimodal_index::{
        CrossModalQuery, FusionStrategy, MmiError, MmiStats, Modality, ModalityEmbedding,
        MultiModalDocument, MultiModalIndex, MultiModalIndexConfig,
    };

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn make_config(strategy: FusionStrategy) -> MultiModalIndexConfig {
        MultiModalIndexConfig {
            fusion_strategy: strategy,
            cross_modal_dims: 4,
            normalize_embeddings: true,
        }
    }

    fn make_doc(id: &str, modalities: HashMap<Modality, ModalityEmbedding>) -> MultiModalDocument {
        MultiModalDocument::new(id, modalities, HashMap::new(), 0).expect("doc creation failed")
    }

    fn make_doc_with_meta(
        id: &str,
        modalities: HashMap<Modality, ModalityEmbedding>,
        meta: HashMap<String, String>,
    ) -> MultiModalDocument {
        MultiModalDocument::new(id, modalities, meta, 42).expect("doc creation failed")
    }

    fn text_emb(v: Vec<f64>) -> ModalityEmbedding {
        ModalityEmbedding::new(Modality::Text, v)
    }

    fn image_emb(v: Vec<f64>) -> ModalityEmbedding {
        ModalityEmbedding::new(Modality::Image, v)
    }

    fn audio_emb(v: Vec<f64>) -> ModalityEmbedding {
        ModalityEmbedding::new(Modality::Audio, v)
    }

    fn unit(n: usize, pos: usize) -> Vec<f64> {
        let mut v = vec![0.0; n];
        v[pos] = 1.0;
        v
    }

    fn query_same(modality: Modality, embedding: Vec<f64>, k: usize) -> CrossModalQuery {
        CrossModalQuery::new(modality, embedding, vec![modality], k)
    }

    // -----------------------------------------------------------------------
    // Modality tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_modality_hash_eq() {
        let mut map = HashMap::new();
        map.insert(Modality::Text, 1_usize);
        map.insert(Modality::Image, 2_usize);
        assert_eq!(map[&Modality::Text], 1);
        assert_eq!(map[&Modality::Image], 2);
        assert_ne!(Modality::Text, Modality::Image);
    }

    #[test]
    fn test_all_modalities_distinct() {
        let all = [
            Modality::Text,
            Modality::Image,
            Modality::Audio,
            Modality::Video,
            Modality::Structured,
        ];
        for i in 0..all.len() {
            for j in 0..all.len() {
                if i != j {
                    assert_ne!(all[i], all[j], "modalities {i} and {j} should differ");
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // ModalityEmbedding tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_modality_embedding_dims() {
        let emb = ModalityEmbedding::new(Modality::Audio, vec![1.0, 2.0, 3.0]);
        assert_eq!(emb.dims, 3);
        assert_eq!(emb.embedding.len(), 3);
        assert_eq!(emb.modality, Modality::Audio);
    }

    // -----------------------------------------------------------------------
    // MultiModalDocument tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_empty_document_rejected() {
        let result = MultiModalDocument::new("id", HashMap::new(), HashMap::new(), 0);
        assert_eq!(result, Err(MmiError::EmptyDocument));
    }

    #[test]
    fn test_document_creation_with_metadata() {
        let mut mods = HashMap::new();
        mods.insert(Modality::Text, text_emb(vec![0.5; 4]));
        let mut meta = HashMap::new();
        meta.insert("author".to_string(), "alice".to_string());
        let doc = MultiModalDocument::new("doc1", mods, meta, 1000).expect("should succeed");
        assert_eq!(doc.id, "doc1");
        assert_eq!(doc.indexed_at, 1000);
        assert_eq!(
            doc.metadata.get("author").map(String::as_str),
            Some("alice")
        );
    }

    // -----------------------------------------------------------------------
    // Index creation and basic management
    // -----------------------------------------------------------------------

    #[test]
    fn test_new_index_is_empty() {
        let idx = MultiModalIndex::new(MultiModalIndexConfig::default());
        assert_eq!(idx.doc_count(), 0);
        assert!(idx.modality_coverage().is_empty());
    }

    #[test]
    fn test_add_document_success() {
        let mut idx = MultiModalIndex::new(MultiModalIndexConfig::default());
        let mut mods = HashMap::new();
        mods.insert(Modality::Text, text_emb(vec![1.0, 0.0, 0.0]));
        let doc = make_doc("a", mods);
        idx.add_document(doc).expect("should succeed");
        assert_eq!(idx.doc_count(), 1);
    }

    #[test]
    fn test_add_duplicate_document_error() {
        let mut idx = MultiModalIndex::new(MultiModalIndexConfig::default());
        let mut mods = HashMap::new();
        mods.insert(Modality::Text, text_emb(vec![1.0, 0.0, 0.0]));
        let doc1 = make_doc("dup", mods.clone());
        let doc2 = make_doc("dup", mods);
        idx.add_document(doc1).expect("first insert ok");
        let err = idx.add_document(doc2).expect_err("second should fail");
        assert!(matches!(err, MmiError::DocumentAlreadyExists(ref id) if id == "dup"));
    }

    #[test]
    fn test_remove_document() {
        let mut idx = MultiModalIndex::new(MultiModalIndexConfig::default());
        let mut mods = HashMap::new();
        mods.insert(Modality::Text, text_emb(vec![1.0, 0.0]));
        let doc = make_doc("rem", mods);
        idx.add_document(doc).expect("ok");
        assert!(idx.remove_document("rem"));
        assert_eq!(idx.doc_count(), 0);
        assert!(!idx.remove_document("rem")); // idempotent second call
    }

    #[test]
    fn test_modality_coverage() {
        let mut idx = MultiModalIndex::new(MultiModalIndexConfig::default());

        let mut mods1 = HashMap::new();
        mods1.insert(Modality::Text, text_emb(vec![1.0, 0.0]));
        mods1.insert(Modality::Image, image_emb(vec![0.0, 1.0]));
        idx.add_document(make_doc("d1", mods1)).expect("ok");

        let mut mods2 = HashMap::new();
        mods2.insert(Modality::Text, text_emb(vec![0.5, 0.5]));
        idx.add_document(make_doc("d2", mods2)).expect("ok");

        let cov = idx.modality_coverage();
        assert_eq!(cov[&Modality::Text], 2);
        assert_eq!(cov[&Modality::Image], 1);
        assert!(!cov.contains_key(&Modality::Audio));
    }

    // -----------------------------------------------------------------------
    // Math helper tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_cosine_similarity_identical() {
        let v = vec![1.0, 2.0, 3.0];
        let sim = MultiModalIndex::cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let sim = MultiModalIndex::cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        let sim = MultiModalIndex::cosine_similarity(&a, &b);
        assert!((sim - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 2.0, 3.0];
        let sim = MultiModalIndex::cosine_similarity(&a, &b);
        assert_eq!(sim, 0.0);
    }

    #[test]
    fn test_cosine_similarity_mismatched_len() {
        let a = vec![1.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let sim = MultiModalIndex::cosine_similarity(&a, &b);
        assert_eq!(sim, 0.0);
    }

    #[test]
    fn test_l2_normalize() {
        let v = vec![3.0, 4.0];
        let n = MultiModalIndex::l2_normalize(&v);
        assert!((n[0] - 0.6).abs() < 1e-10);
        assert!((n[1] - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_l2_normalize_zero_vec() {
        let v = vec![0.0, 0.0];
        let n = MultiModalIndex::l2_normalize(&v);
        assert_eq!(n, vec![0.0, 0.0]);
    }

    #[test]
    fn test_project_identity() {
        // 2×2 identity matrix
        let matrix = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let emb = vec![3.0, 7.0];
        let out = MultiModalIndex::project(&emb, &matrix);
        assert_eq!(out, vec![3.0, 7.0]);
    }

    #[test]
    fn test_project_dimensions() {
        // 3-output × 2-input matrix
        let matrix = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        let emb = vec![2.0, 3.0];
        let out = MultiModalIndex::project(&emb, &matrix);
        assert_eq!(out.len(), 3);
        assert!((out[2] - 5.0).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // Projection matrix management
    // -----------------------------------------------------------------------

    #[test]
    fn test_add_projection_success() {
        let mut idx = MultiModalIndex::new(MultiModalIndexConfig::default());
        let matrix = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        assert!(idx
            .add_projection(Modality::Text, Modality::Image, matrix)
            .is_ok());
    }

    #[test]
    fn test_add_projection_empty_matrix() {
        let mut idx = MultiModalIndex::new(MultiModalIndexConfig::default());
        let err = idx
            .add_projection(Modality::Text, Modality::Image, vec![])
            .expect_err("empty should fail");
        assert_eq!(err, MmiError::ProjectionDimsMismatch);
    }

    #[test]
    fn test_add_projection_ragged_matrix() {
        let mut idx = MultiModalIndex::new(MultiModalIndexConfig::default());
        // Second row has different length
        let matrix = vec![vec![1.0, 0.0], vec![0.0]];
        let err = idx
            .add_projection(Modality::Text, Modality::Image, matrix)
            .expect_err("ragged should fail");
        assert_eq!(err, MmiError::ProjectionDimsMismatch);
    }

    #[test]
    fn test_add_projection_zero_cols() {
        let mut idx = MultiModalIndex::new(MultiModalIndexConfig::default());
        let matrix = vec![vec![]];
        let err = idx
            .add_projection(Modality::Text, Modality::Image, matrix)
            .expect_err("zero cols should fail");
        assert_eq!(err, MmiError::ProjectionDimsMismatch);
    }

    // -----------------------------------------------------------------------
    // Same-modality search
    // -----------------------------------------------------------------------

    #[test]
    fn test_same_modality_search_basic() {
        let mut idx = MultiModalIndex::new(make_config(FusionStrategy::MeanScore));
        let mut mods1 = HashMap::new();
        mods1.insert(Modality::Text, text_emb(unit(3, 0)));
        let mut mods2 = HashMap::new();
        mods2.insert(Modality::Text, text_emb(unit(3, 1)));
        let mut mods3 = HashMap::new();
        mods3.insert(Modality::Text, text_emb(unit(3, 2)));
        idx.add_document(make_doc("a", mods1)).expect("ok");
        idx.add_document(make_doc("b", mods2)).expect("ok");
        idx.add_document(make_doc("c", mods3)).expect("ok");

        let q = query_same(Modality::Text, unit(3, 0), 1);
        let results = idx.same_modality_search(&q);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].doc_id, "a");
        assert!((results[0].combined_score - 1.0).abs() < 1e-10);
        assert_eq!(results[0].rank, 1);
    }

    #[test]
    fn test_same_modality_search_top_k() {
        let mut idx = MultiModalIndex::new(make_config(FusionStrategy::MeanScore));
        for i in 0..10_usize {
            let mut mods = HashMap::new();
            // All embeddings have positive similarity to [1,0,...] but diminishing
            let mut v = vec![0.0; 4];
            v[0] = 1.0 - i as f64 * 0.05;
            v[1] = (i as f64 * 0.05).max(0.0);
            mods.insert(Modality::Text, text_emb(v));
            idx.add_document(make_doc(&i.to_string(), mods))
                .expect("ok");
        }
        let q = query_same(Modality::Text, unit(4, 0), 3);
        let results = idx.same_modality_search(&q);
        assert_eq!(results.len(), 3);
        // Ranks should be 1, 2, 3 in order
        assert_eq!(results[0].rank, 1);
        assert_eq!(results[1].rank, 2);
        assert_eq!(results[2].rank, 3);
        // Scores should be non-increasing
        assert!(results[0].combined_score >= results[1].combined_score);
        assert!(results[1].combined_score >= results[2].combined_score);
    }

    #[test]
    fn test_same_modality_search_min_similarity_filter() {
        let mut idx = MultiModalIndex::new(make_config(FusionStrategy::MeanScore));
        // d1 is very similar, d2 is orthogonal
        let mut mods1 = HashMap::new();
        mods1.insert(Modality::Text, text_emb(unit(3, 0)));
        let mut mods2 = HashMap::new();
        mods2.insert(Modality::Text, text_emb(unit(3, 1)));
        idx.add_document(make_doc("d1", mods1)).expect("ok");
        idx.add_document(make_doc("d2", mods2)).expect("ok");

        let mut q = query_same(Modality::Text, unit(3, 0), 10);
        q.min_similarity = 0.5;
        let results = idx.same_modality_search(&q);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].doc_id, "d1");
    }

    // -----------------------------------------------------------------------
    // Cross-modal search
    // -----------------------------------------------------------------------

    #[test]
    fn test_search_same_modality_in_search() {
        let mut idx = MultiModalIndex::new(make_config(FusionStrategy::MeanScore));
        let mut mods = HashMap::new();
        mods.insert(Modality::Text, text_emb(unit(3, 0)));
        idx.add_document(make_doc("t1", mods)).expect("ok");

        let q = CrossModalQuery::new(Modality::Text, unit(3, 0), vec![Modality::Text], 5);
        let results = idx.search(&q);
        assert_eq!(results.len(), 1);
        assert!((results[0].combined_score - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_search_cross_modal_with_projection() {
        let mut idx = MultiModalIndex::new(make_config(FusionStrategy::MeanScore));

        // Document has an Image embedding (dim=4)
        let mut mods = HashMap::new();
        mods.insert(Modality::Image, image_emb(unit(4, 0)));
        idx.add_document(make_doc("img1", mods)).expect("ok");

        // Add a Text→Image projection (identity in 4D)
        let identity = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
        ];
        idx.add_projection(Modality::Text, Modality::Image, identity)
            .expect("projection ok");

        // Query from Text, targeting Image
        let q = CrossModalQuery::new(Modality::Text, unit(4, 0), vec![Modality::Image], 5);
        let results = idx.search(&q);
        assert_eq!(results.len(), 1);
        assert!((results[0].combined_score - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_search_no_projection_skips_cross_modal() {
        let mut idx = MultiModalIndex::new(make_config(FusionStrategy::MeanScore));
        let mut mods = HashMap::new();
        mods.insert(Modality::Image, image_emb(unit(4, 0)));
        idx.add_document(make_doc("img1", mods)).expect("ok");

        // No projection registered → Text→Image cross-modal should return nothing
        let q = CrossModalQuery::new(Modality::Text, unit(4, 0), vec![Modality::Image], 5);
        let results = idx.search(&q);
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_multimodal_fusion_mean() {
        // MeanScore fusion: when two modalities score differently and both are
        // in the same modality space (same query_modality), the mean is taken.
        // Here both Text embeddings (dim=2) score against the Text query: d1=1.0, d2=0.0.
        // The combined score for d1 is mean(1.0) = 1.0.
        let mut idx = MultiModalIndex::new(make_config(FusionStrategy::MeanScore));

        let mut mods1 = HashMap::new();
        mods1.insert(Modality::Text, text_emb(unit(2, 0)));
        let mut mods2 = HashMap::new();
        mods2.insert(Modality::Text, text_emb(unit(2, 1)));
        idx.add_document(make_doc("d1", mods1)).expect("ok");
        idx.add_document(make_doc("d2", mods2)).expect("ok");

        let q = CrossModalQuery::new(Modality::Text, unit(2, 0), vec![Modality::Text], 5);
        let results = idx.search(&q);
        assert_eq!(results.len(), 2);
        // d1 has score 1.0, d2 has score 0.0
        let r1 = results.iter().find(|r| r.doc_id == "d1").expect("d1");
        let r2 = results.iter().find(|r| r.doc_id == "d2").expect("d2");
        assert!((r1.combined_score - 1.0).abs() < 1e-10);
        assert!(r2.combined_score.abs() < 1e-10);
        // Verify mean fusion: combined = mean of per-modality scores (just one here)
        assert!((r1.combined_score - r1.scores[&Modality::Text]).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // FusionStrategy tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_fusion_max_score() {
        let strategy = FusionStrategy::MaxScore;
        let mut scores = HashMap::new();
        scores.insert(Modality::Text, 0.3);
        scores.insert(Modality::Image, 0.8);
        scores.insert(Modality::Audio, 0.5);
        let fused = strategy.fuse(&scores).expect("some");
        assert!((fused - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_fusion_mean_score() {
        let strategy = FusionStrategy::MeanScore;
        let mut scores = HashMap::new();
        scores.insert(Modality::Text, 0.6);
        scores.insert(Modality::Image, 0.4);
        let fused = strategy.fuse(&scores).expect("some");
        assert!((fused - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_fusion_weighted() {
        let mut weights = HashMap::new();
        weights.insert(Modality::Text, 3.0);
        weights.insert(Modality::Image, 1.0);
        let strategy = FusionStrategy::WeightedFusion { weights };

        let mut scores = HashMap::new();
        scores.insert(Modality::Text, 1.0);
        scores.insert(Modality::Image, 0.0);
        let fused = strategy.fuse(&scores).expect("some");
        // 3/4 * 1.0 + 1/4 * 0.0 = 0.75
        assert!((fused - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_fusion_weighted_missing_modality_uses_mean_fallback() {
        // Weights only cover Audio, but scores only have Text/Image → weight_total=0 → mean fallback
        let mut weights = HashMap::new();
        weights.insert(Modality::Audio, 2.0);
        let strategy = FusionStrategy::WeightedFusion { weights };

        let mut scores = HashMap::new();
        scores.insert(Modality::Text, 0.4);
        scores.insert(Modality::Image, 0.6);
        let fused = strategy.fuse(&scores).expect("some");
        assert!((fused - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_fusion_text_primary_uses_text_when_present() {
        let strategy = FusionStrategy::TextPrimary;
        let mut scores = HashMap::new();
        scores.insert(Modality::Text, 0.7);
        scores.insert(Modality::Image, 0.9);
        let fused = strategy.fuse(&scores).expect("some");
        assert!((fused - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_fusion_text_primary_falls_back_to_max() {
        let strategy = FusionStrategy::TextPrimary;
        let mut scores = HashMap::new();
        scores.insert(Modality::Image, 0.6);
        scores.insert(Modality::Audio, 0.9);
        let fused = strategy.fuse(&scores).expect("some");
        assert!((fused - 0.9).abs() < 1e-10);
    }

    #[test]
    fn test_fusion_empty_scores_returns_none() {
        let strategy = FusionStrategy::MeanScore;
        assert!(strategy.fuse(&HashMap::new()).is_none());
    }

    // -----------------------------------------------------------------------
    // Metadata filter tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_metadata_filter_match() {
        let mut idx = MultiModalIndex::new(make_config(FusionStrategy::MeanScore));

        let mut meta1 = HashMap::new();
        meta1.insert("type".to_string(), "news".to_string());
        let mut mods1 = HashMap::new();
        mods1.insert(Modality::Text, text_emb(unit(3, 0)));
        idx.add_document(make_doc_with_meta("n1", mods1, meta1))
            .expect("ok");

        let mut meta2 = HashMap::new();
        meta2.insert("type".to_string(), "blog".to_string());
        let mut mods2 = HashMap::new();
        mods2.insert(Modality::Text, text_emb(unit(3, 0)));
        idx.add_document(make_doc_with_meta("b1", mods2, meta2))
            .expect("ok");

        let mut filters = HashMap::new();
        filters.insert("type".to_string(), "news".to_string());
        let mut q = query_same(Modality::Text, unit(3, 0), 10);
        q.filters = filters;

        let results = idx.same_modality_search(&q);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].doc_id, "n1");
    }

    #[test]
    fn test_metadata_filter_no_match_returns_empty() {
        let mut idx = MultiModalIndex::new(make_config(FusionStrategy::MeanScore));
        let mut mods = HashMap::new();
        mods.insert(Modality::Text, text_emb(unit(3, 0)));
        idx.add_document(make_doc("d1", mods)).expect("ok");

        let mut filters = HashMap::new();
        filters.insert("nonexistent".to_string(), "value".to_string());
        let mut q = query_same(Modality::Text, unit(3, 0), 10);
        q.filters = filters;

        let results = idx.search(&q);
        assert!(results.is_empty());
    }

    // -----------------------------------------------------------------------
    // Stats tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_stats_empty_index() {
        let idx = MultiModalIndex::new(MultiModalIndexConfig::default());
        let stats = idx.stats();
        assert_eq!(stats.doc_count, 0);
        assert_eq!(stats.total_searches, 0);
        assert_eq!(stats.avg_modalities_per_doc, 0.0);
    }

    #[test]
    fn test_stats_search_counter_increments() {
        let mut idx = MultiModalIndex::new(make_config(FusionStrategy::MeanScore));
        let mut mods = HashMap::new();
        mods.insert(Modality::Text, text_emb(unit(3, 0)));
        idx.add_document(make_doc("x", mods)).expect("ok");

        let q = query_same(Modality::Text, unit(3, 0), 5);
        idx.search(&q);
        idx.search(&q);
        idx.same_modality_search(&q);

        let stats = idx.stats();
        assert_eq!(stats.total_searches, 3);
    }

    #[test]
    fn test_stats_avg_modalities_per_doc() {
        let mut idx = MultiModalIndex::new(MultiModalIndexConfig::default());

        // doc1: 2 modalities
        let mut mods1 = HashMap::new();
        mods1.insert(Modality::Text, text_emb(vec![1.0]));
        mods1.insert(Modality::Image, image_emb(vec![1.0]));
        idx.add_document(make_doc("d1", mods1)).expect("ok");

        // doc2: 1 modality
        let mut mods2 = HashMap::new();
        mods2.insert(Modality::Audio, audio_emb(vec![1.0]));
        idx.add_document(make_doc("d2", mods2)).expect("ok");

        let stats = idx.stats();
        assert_eq!(stats.doc_count, 2);
        // avg = (2 + 1) / 2 = 1.5
        assert!((stats.avg_modalities_per_doc - 1.5).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // Edge-case and robustness tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_search_empty_index() {
        let idx = MultiModalIndex::new(MultiModalIndexConfig::default());
        let q = query_same(Modality::Text, unit(3, 0), 5);
        let results = idx.search(&q);
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_dim_mismatch_skips_document() {
        let mut idx = MultiModalIndex::new(make_config(FusionStrategy::MeanScore));
        // Document has 3D text embedding, query has 2D
        let mut mods = HashMap::new();
        mods.insert(Modality::Text, text_emb(unit(3, 0)));
        idx.add_document(make_doc("dim_mismatch", mods))
            .expect("ok");

        let q = query_same(Modality::Text, unit(2, 0), 5);
        let results = idx.same_modality_search(&q);
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_scores_contain_matched_modalities() {
        let mut idx = MultiModalIndex::new(make_config(FusionStrategy::MeanScore));
        let mut mods = HashMap::new();
        mods.insert(Modality::Text, text_emb(unit(3, 0)));
        mods.insert(Modality::Image, image_emb(unit(3, 1)));
        idx.add_document(make_doc("multi", mods)).expect("ok");

        let q = CrossModalQuery::new(
            Modality::Text,
            unit(3, 0),
            vec![Modality::Text, Modality::Image],
            5,
        );
        let results = idx.search(&q);
        assert_eq!(results.len(), 1);
        assert!(results[0].scores.contains_key(&Modality::Text));
        // Image modality is in the same document but different dim check doesn't interfere
    }

    #[test]
    fn test_search_multimodal_fusion_mean_cross_modal() {
        // Cross-modal mean: Text→Text (sim=1.0) and Text→Image via projection (sim=0.0).
        // Combined = mean(1.0, 0.0) = 0.5.
        let mut idx = MultiModalIndex::new(make_config(FusionStrategy::MeanScore));

        let mut mods = HashMap::new();
        mods.insert(Modality::Text, text_emb(unit(2, 0)));
        mods.insert(Modality::Image, image_emb(unit(2, 1)));
        idx.add_document(make_doc("mm", mods)).expect("ok");

        // 2×2 identity projection: Text space → Image space
        let identity = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        idx.add_projection(Modality::Text, Modality::Image, identity)
            .expect("ok");

        let q = CrossModalQuery::new(
            Modality::Text,
            unit(2, 0),
            vec![Modality::Text, Modality::Image],
            5,
        );
        let results = idx.search(&q);
        assert_eq!(results.len(), 1);
        // Text sim = 1.0, Image sim = cosine(unit(2,0), unit(2,1)) = 0.0 → mean = 0.5
        assert!((results[0].combined_score - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_cross_modal_result_rank_assignment() {
        let mut idx = MultiModalIndex::new(make_config(FusionStrategy::MeanScore));
        for i in 0..5_usize {
            let mut mods = HashMap::new();
            let mut v = unit(4, 0);
            v[1] = i as f64 * 0.01; // slight variation
            mods.insert(Modality::Text, text_emb(v));
            idx.add_document(make_doc(&format!("r{i}"), mods))
                .expect("ok");
        }
        let q = query_same(Modality::Text, unit(4, 0), 5);
        let results = idx.search(&q);
        for (i, r) in results.iter().enumerate() {
            assert_eq!(r.rank, i + 1);
        }
    }

    #[test]
    fn test_mmi_error_display() {
        let e1 = MmiError::DocumentAlreadyExists("id1".into());
        let e2 = MmiError::DocumentNotFound("id2".into());
        let e3 = MmiError::EmptyDocument;
        let e4 = MmiError::ProjectionDimsMismatch;
        let e5 = MmiError::NoMatchingModality;
        assert!(e1.to_string().contains("id1"));
        assert!(e2.to_string().contains("id2"));
        assert!(!e3.to_string().is_empty());
        assert!(!e4.to_string().is_empty());
        assert!(!e5.to_string().is_empty());
    }

    #[test]
    fn test_mmi_stats_modality_counts() {
        let mut idx = MultiModalIndex::new(MultiModalIndexConfig::default());
        for i in 0..3_usize {
            let mut mods = HashMap::new();
            mods.insert(Modality::Text, text_emb(vec![i as f64]));
            idx.add_document(make_doc(&i.to_string(), mods))
                .expect("ok");
        }
        let stats: MmiStats = idx.stats();
        assert_eq!(*stats.modality_counts.get(&Modality::Text).unwrap_or(&0), 3);
    }

    #[test]
    fn test_normalization_flag_off() {
        let config = MultiModalIndexConfig {
            fusion_strategy: FusionStrategy::MeanScore,
            cross_modal_dims: 4,
            normalize_embeddings: false,
        };
        let mut idx = MultiModalIndex::new(config);
        let mut mods = HashMap::new();
        mods.insert(Modality::Text, text_emb(vec![2.0, 0.0, 0.0]));
        idx.add_document(make_doc("nn", mods)).expect("ok");

        // Unnormalized query; cosine_similarity still computes correctly
        let q = query_same(Modality::Text, vec![2.0, 0.0, 0.0], 5);
        let results = idx.search(&q);
        assert_eq!(results.len(), 1);
        assert!((results[0].combined_score - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_structured_and_video_modalities() {
        let mut idx = MultiModalIndex::new(make_config(FusionStrategy::MaxScore));
        let mut mods = HashMap::new();
        mods.insert(
            Modality::Structured,
            ModalityEmbedding::new(Modality::Structured, unit(4, 0)),
        );
        mods.insert(
            Modality::Video,
            ModalityEmbedding::new(Modality::Video, unit(4, 1)),
        );
        idx.add_document(make_doc("sv", mods)).expect("ok");

        let q = CrossModalQuery::new(
            Modality::Structured,
            unit(4, 0),
            vec![Modality::Structured, Modality::Video],
            5,
        );
        let results = idx.search(&q);
        assert_eq!(results.len(), 1);
        // MaxScore of (1.0, 0.0) = 1.0
        assert!((results[0].combined_score - 1.0).abs() < 1e-10);
    }
}
