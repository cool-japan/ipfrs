//! Cross-encoder reranking for semantic search results.
//!
//! This module provides pairwise query-document scoring for reranking
//! candidate documents retrieved from an initial retrieval stage. Unlike
//! bi-encoder models that score query and document independently, cross-encoders
//! jointly score the (query, document) pair for higher precision.
//!
//! # Architecture
//!
//! The pipeline is:
//! 1. Initial retrieval (e.g., HNSW ANN search) yields `CandidateDoc` list with `initial_score`.
//! 2. `CrossEncoder::rerank` scores each (query, doc) pair via the configured `ScoringModel`.
//! 3. Results are sorted by `cross_encoder_score` and optionally min-max normalized.
//! 4. `RerankedDoc` carries the original rank metadata and `score_delta` for analysis.
//!
//! # Scoring Models
//!
//! - [`ScoringModel::DotProduct`] — raw inner product, fast, unnormalized.
//! - [`ScoringModel::Cosine`] — cosine similarity in `[-1, 1]`, direction-sensitive.
//! - [`ScoringModel::BilinearForm`] — diagonal bilinear `Σ w_i q_i d_i`, learned weights.
//! - [`ScoringModel::Linear`] — `dot(weights, q ⊙ d) + bias`, affine combination.
//!
//! # Example
//!
//! ```rust
//! use ipfrs_semantic::cross_encoder::{
//!     CrossEncoder, CrossEncoderConfig, ScoringModel, CandidateDoc,
//! };
//! use std::collections::HashMap;
//!
//! let config = CrossEncoderConfig {
//!     model: ScoringModel::Cosine,
//!     max_doc_length: 512,
//!     batch_size: 32,
//!     normalize_scores: true,
//! };
//! let mut encoder = CrossEncoder::new(config);
//!
//! let query = vec![1.0, 0.0, 0.0];
//! let candidates = vec![
//!     CandidateDoc {
//!         doc_id: "doc_a".to_string(),
//!         embedding: vec![0.9, 0.1, 0.0],
//!         initial_score: 0.8,
//!         metadata: HashMap::new(),
//!     },
//!     CandidateDoc {
//!         doc_id: "doc_b".to_string(),
//!         embedding: vec![0.0, 1.0, 0.0],
//!         initial_score: 0.9,
//!         metadata: HashMap::new(),
//!     },
//! ];
//!
//! let reranked = encoder.rerank(&query, candidates);
//! // doc_a should now rank #1 because it aligns better with query [1,0,0]
//! assert_eq!(reranked[0].doc_id, "doc_a");
//! ```

use std::collections::HashMap;

// ────────────────────────────────────────────────────────────────────────────
// Public types
// ────────────────────────────────────────────────────────────────────────────

/// Relevance scoring model used by the [`CrossEncoder`].
#[derive(Debug, Clone)]
pub enum ScoringModel {
    /// Raw dot product: `Σ q_i · d_i`.
    DotProduct,
    /// Cosine similarity: `dot(q, d) / (|q| · |d|)`.
    Cosine,
    /// Diagonal bilinear form: `Σ w_i · q_i · d_i`.
    ///
    /// `weights` is a diagonal weight vector of length `dim`.
    /// Weights are reused cyclically when the embedding dimension exceeds
    /// the weight length, so a length-1 weight vector acts as a global scalar.
    BilinearForm(Vec<f64>),
    /// Affine linear model: `dot(weights, q ⊙ d) + bias`.
    ///
    /// `q ⊙ d` is the element-wise product of query and document embeddings.
    /// `weights` length should match the embedding dimension; shorter weights
    /// are extended with zeros, longer weights are truncated.
    Linear { weights: Vec<f64>, bias: f64 },
}

/// Configuration for the [`CrossEncoder`].
#[derive(Debug, Clone)]
pub struct CrossEncoderConfig {
    /// Scoring model to use for pairwise relevance estimation.
    pub model: ScoringModel,
    /// Maximum document embedding length to consider (truncates longer ones).
    pub max_doc_length: usize,
    /// Number of (query, doc) pairs to score in a single batch during
    /// [`CrossEncoder::rerank_batch`].
    pub batch_size: usize,
    /// Whether to apply min-max normalization to `cross_encoder_score` values
    /// so they fall in `[0, 1]`.
    pub normalize_scores: bool,
}

impl Default for CrossEncoderConfig {
    fn default() -> Self {
        Self {
            model: ScoringModel::Cosine,
            max_doc_length: 512,
            batch_size: 64,
            normalize_scores: true,
        }
    }
}

/// A candidate document produced by an upstream retrieval system.
#[derive(Debug, Clone)]
pub struct CandidateDoc {
    /// Unique document identifier.
    pub doc_id: String,
    /// Dense embedding vector.
    pub embedding: Vec<f64>,
    /// Retrieval score assigned by the first-stage retriever.
    pub initial_score: f64,
    /// Arbitrary key-value metadata (e.g., title, source, date).
    pub metadata: HashMap<String, String>,
}

/// A reranked document with updated score and rank metadata.
#[derive(Debug, Clone)]
pub struct RerankedDoc {
    /// Document identifier (mirrors [`CandidateDoc::doc_id`]).
    pub doc_id: String,
    /// Score assigned by the cross-encoder.
    pub cross_encoder_score: f64,
    /// Original first-stage retrieval score.
    pub initial_score: f64,
    /// Final rank after reranking (1-indexed, best = 1).
    pub final_rank: usize,
    /// `cross_encoder_score - initial_score`; positive means reranking improved
    /// the document's perceived relevance.
    pub score_delta: f64,
}

/// Aggregate statistics collected across all reranking calls.
#[derive(Debug, Clone, Default)]
pub struct CrossEncoderStats {
    /// Total number of `rerank` or `rerank_batch` calls.
    pub total_reranks: u64,
    /// Total number of individual documents scored.
    pub total_docs_reranked: u64,
    /// Running average absolute rank change per reranking call.
    pub avg_rank_change: f64,
}

// ────────────────────────────────────────────────────────────────────────────
// CrossEncoder
// ────────────────────────────────────────────────────────────────────────────

/// Cross-encoder that jointly scores (query, document) pairs for reranking.
///
/// Construct with [`CrossEncoder::new`] and call [`CrossEncoder::rerank`] to
/// reorder a `Vec<CandidateDoc>` by cross-encoder relevance scores.
pub struct CrossEncoder {
    config: CrossEncoderConfig,
    stats: CrossEncoderStats,
}

impl CrossEncoder {
    // ── Construction ────────────────────────────────────────────────────────

    /// Create a new cross-encoder with the given configuration.
    pub fn new(config: CrossEncoderConfig) -> Self {
        Self {
            config,
            stats: CrossEncoderStats::default(),
        }
    }

    // ── Scoring ─────────────────────────────────────────────────────────────

    /// Compute the relevance score for a single (query, document) pair.
    ///
    /// The embedding lengths are independently capped at
    /// [`CrossEncoderConfig::max_doc_length`] before scoring.
    pub fn score_pair(&self, query: &[f64], doc: &[f64]) -> f64 {
        let max_len = self.config.max_doc_length;
        let q = if query.len() > max_len {
            &query[..max_len]
        } else {
            query
        };
        let d = if doc.len() > max_len {
            &doc[..max_len]
        } else {
            doc
        };

        match &self.config.model {
            ScoringModel::DotProduct => Self::dot_product(q, d),
            ScoringModel::Cosine => Self::cosine_similarity(q, d),
            ScoringModel::BilinearForm(weights) => Self::bilinear_score(q, d, weights),
            ScoringModel::Linear { weights, bias } => Self::linear_score(q, d, weights, *bias),
        }
    }

    // ── Reranking ────────────────────────────────────────────────────────────

    /// Rerank a list of candidate documents for the given query embedding.
    ///
    /// Returns documents sorted in descending order of `cross_encoder_score`.
    /// Stats are updated after each call.
    pub fn rerank(&mut self, query: &[f64], candidates: Vec<CandidateDoc>) -> Vec<RerankedDoc> {
        if candidates.is_empty() {
            self.stats.total_reranks += 1;
            return Vec::new();
        }

        // Capture original order (by doc_id) so we can count rank changes.
        let initial_order: Vec<String> = candidates.iter().map(|c| c.doc_id.clone()).collect();

        // Score every candidate.
        let mut scored: Vec<(f64, CandidateDoc)> = candidates
            .into_iter()
            .map(|c| {
                let score = self.score_pair(query, &c.embedding);
                (score, c)
            })
            .collect();

        // Sort descending by cross-encoder score, stable for ties.
        scored.sort_by(|(a, _), (b, _)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        // Build RerankedDoc list.
        let mut reranked: Vec<RerankedDoc> = scored
            .into_iter()
            .enumerate()
            .map(|(rank_idx, (ce_score, candidate))| RerankedDoc {
                doc_id: candidate.doc_id,
                cross_encoder_score: ce_score,
                initial_score: candidate.initial_score,
                final_rank: rank_idx + 1,
                score_delta: ce_score - candidate.initial_score,
            })
            .collect();

        // Optionally normalize scores in-place.
        if self.config.normalize_scores {
            Self::normalize_scores(&mut reranked);
        }

        // Update statistics.
        let initial_order_refs: Vec<&str> = initial_order.iter().map(String::as_str).collect();
        let rank_changes = Self::rank_changed(&initial_order_refs, &reranked);
        let doc_count = reranked.len() as u64;

        self.stats.total_reranks += 1;
        self.stats.total_docs_reranked += doc_count;

        // Incremental average: avg = avg + (new - avg) / n
        let n = self.stats.total_reranks as f64;
        let change_rate = rank_changes as f64 / doc_count.max(1) as f64;
        self.stats.avg_rank_change += (change_rate - self.stats.avg_rank_change) / n;

        reranked
    }

    /// Rerank multiple (query, candidates) pairs in a single call.
    ///
    /// Each element of the input slice is processed independently. The method
    /// is equivalent to calling [`rerank`](CrossEncoder::rerank) for each pair
    /// sequentially.
    pub fn rerank_batch(
        &mut self,
        queries_and_candidates: Vec<(Vec<f64>, Vec<CandidateDoc>)>,
    ) -> Vec<Vec<RerankedDoc>> {
        queries_and_candidates
            .into_iter()
            .map(|(query, candidates)| self.rerank(&query, candidates))
            .collect()
    }

    // ── Primitive scoring functions (static) ────────────────────────────────

    /// Raw dot product: `Σ q_i · d_i`.
    ///
    /// Stops at the shorter slice length.
    pub fn dot_product(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    /// Cosine similarity: `dot(a, b) / (|a| · |b|)`.
    ///
    /// Returns `0.0` when either vector has near-zero norm to avoid division
    /// by zero; the epsilon used is `1e-10`.
    pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
        const EPSILON: f64 = 1e-10;
        let dot = Self::dot_product(a, b);
        let norm_a = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm_b = b.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm_a < EPSILON || norm_b < EPSILON {
            0.0
        } else {
            dot / (norm_a * norm_b)
        }
    }

    /// Diagonal bilinear form: `Σ w_i · q_i · d_i`.
    ///
    /// Weights are cycled when the embedding dimension exceeds `weights.len()`.
    /// A zero-length weights slice returns `0.0`.
    pub fn bilinear_score(query: &[f64], doc: &[f64], weights: &[f64]) -> f64 {
        if weights.is_empty() {
            return 0.0;
        }
        let wlen = weights.len();
        query
            .iter()
            .zip(doc.iter())
            .enumerate()
            .map(|(i, (q, d))| weights[i % wlen] * q * d)
            .sum()
    }

    /// Affine linear model: `dot(weights, q ⊙ d) + bias`.
    ///
    /// `q ⊙ d` is the element-wise product, capped at the minimum length of
    /// `query` and `doc`. Weights beyond the element-wise product length are
    /// ignored; product components without a corresponding weight are treated
    /// as having weight `0.0`.
    pub fn linear_score(query: &[f64], doc: &[f64], weights: &[f64], bias: f64) -> f64 {
        let dot: f64 = query
            .iter()
            .zip(doc.iter())
            .enumerate()
            .map(|(i, (q, d))| {
                let w = weights.get(i).copied().unwrap_or(0.0);
                w * q * d
            })
            .sum();
        dot + bias
    }

    // ── Post-processing helpers ──────────────────────────────────────────────

    /// Min-max normalize `cross_encoder_score` values to `[0, 1]` in-place.
    ///
    /// When all scores are identical the method maps every score to `1.0` to
    /// avoid a degenerate zero-range normalization.
    pub fn normalize_scores(docs: &mut [RerankedDoc]) {
        if docs.is_empty() {
            return;
        }

        let min = docs
            .iter()
            .map(|d| d.cross_encoder_score)
            .fold(f64::INFINITY, f64::min);
        let max = docs
            .iter()
            .map(|d| d.cross_encoder_score)
            .fold(f64::NEG_INFINITY, f64::max);

        let range = max - min;
        if range < f64::EPSILON {
            // All scores are equal; map everything to 1.0 (perfectly relevant).
            for d in docs.iter_mut() {
                d.cross_encoder_score = 1.0;
            }
            return;
        }

        for d in docs.iter_mut() {
            d.cross_encoder_score = (d.cross_encoder_score - min) / range;
        }
    }

    /// Count the number of documents whose position changed after reranking.
    ///
    /// `initial_order` holds document IDs in the order returned by the first-
    /// stage retriever; `reranked` is the cross-encoder output. A document is
    /// counted as "changed" when its 0-indexed position in `reranked` differs
    /// from its position in `initial_order`. Documents present in only one list
    /// are not counted.
    pub fn rank_changed(initial_order: &[&str], reranked: &[RerankedDoc]) -> usize {
        // Build position map for initial order.
        let initial_positions: HashMap<&str, usize> = initial_order
            .iter()
            .enumerate()
            .map(|(i, id)| (*id, i))
            .collect();

        reranked
            .iter()
            .enumerate()
            .filter(|(new_pos, doc)| {
                initial_positions
                    .get(doc.doc_id.as_str())
                    .map(|old_pos| *old_pos != *new_pos)
                    .unwrap_or(false)
            })
            .count()
    }

    // ── Introspection ────────────────────────────────────────────────────────

    /// Return a reference to the accumulated runtime statistics.
    pub fn stats(&self) -> &CrossEncoderStats {
        &self.stats
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    // ── Helpers ──────────────────────────────────────────────────────────────

    fn make_candidate(id: &str, embedding: Vec<f64>, initial_score: f64) -> CandidateDoc {
        CandidateDoc {
            doc_id: id.to_string(),
            embedding,
            initial_score,
            metadata: HashMap::new(),
        }
    }

    fn encoder_with(model: ScoringModel, normalize: bool) -> CrossEncoder {
        CrossEncoder::new(CrossEncoderConfig {
            model,
            max_doc_length: 512,
            batch_size: 32,
            normalize_scores: normalize,
        })
    }

    // ── 1. DotProduct scoring ─────────────────────────────────────────────

    #[test]
    fn test_dot_product_basic() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let score = CrossEncoder::dot_product(&a, &b);
        assert!((score - 32.0).abs() < 1e-9, "expected 32, got {score}");
    }

    #[test]
    fn test_dot_product_model_score_pair() {
        let enc = encoder_with(ScoringModel::DotProduct, false);
        let query = vec![1.0, 0.0];
        let doc = vec![0.5, 0.5];
        // score_pair should forward to dot_product
        let s = enc.score_pair(&query, &doc);
        assert!((s - 0.5).abs() < 1e-9);
        // also exercise mutability guard (no mutation happens here)
        let _ = enc.stats();
    }

    #[test]
    fn test_dot_product_mismatched_lengths() {
        // zip stops at shorter
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 1.0];
        let score = CrossEncoder::dot_product(&a, &b);
        assert!((score - 3.0).abs() < 1e-9);
    }

    // ── 2. Cosine scoring ─────────────────────────────────────────────────

    #[test]
    fn test_cosine_identical_vectors() {
        let v = vec![1.0, 2.0, 3.0];
        let score = CrossEncoder::cosine_similarity(&v, &v);
        assert!((score - 1.0).abs() < 1e-9, "identical vectors => cos=1");
    }

    #[test]
    fn test_cosine_orthogonal_vectors() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let score = CrossEncoder::cosine_similarity(&a, &b);
        assert!(score.abs() < 1e-9, "orthogonal vectors => cos=0");
    }

    #[test]
    fn test_cosine_opposite_vectors() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        let score = CrossEncoder::cosine_similarity(&a, &b);
        assert!((score - (-1.0)).abs() < 1e-9, "opposite vectors => cos=-1");
    }

    #[test]
    fn test_cosine_zero_vector_query() {
        let zero = vec![0.0, 0.0, 0.0];
        let doc = vec![1.0, 2.0, 3.0];
        let score = CrossEncoder::cosine_similarity(&zero, &doc);
        assert_eq!(score, 0.0, "zero query vector => 0");
    }

    #[test]
    fn test_cosine_zero_vector_doc() {
        let query = vec![1.0, 2.0, 3.0];
        let zero = vec![0.0, 0.0, 0.0];
        let score = CrossEncoder::cosine_similarity(&query, &zero);
        assert_eq!(score, 0.0, "zero doc vector => 0");
    }

    #[test]
    fn test_cosine_both_zero_vectors() {
        let zero = vec![0.0, 0.0];
        let score = CrossEncoder::cosine_similarity(&zero, &zero);
        assert_eq!(score, 0.0, "both zero => 0");
    }

    #[test]
    fn test_cosine_model_score_pair() {
        let enc = encoder_with(ScoringModel::Cosine, false);
        let a = vec![1.0, 1.0];
        let b = vec![1.0, 1.0];
        let s = enc.score_pair(&a, &b);
        assert!((s - 1.0).abs() < 1e-9);
    }

    // ── 3. BilinearForm scoring ───────────────────────────────────────────

    #[test]
    fn test_bilinear_basic() {
        let q = vec![1.0, 2.0, 3.0];
        let d = vec![4.0, 5.0, 6.0];
        let w = vec![1.0, 2.0, 3.0];
        // 1*1*4 + 2*2*5 + 3*3*6 = 4 + 20 + 54 = 78
        let score = CrossEncoder::bilinear_score(&q, &d, &w);
        assert!((score - 78.0).abs() < 1e-9, "expected 78, got {score}");
    }

    #[test]
    fn test_bilinear_cyclic_weights() {
        let q = vec![1.0, 2.0, 3.0, 4.0];
        let d = vec![1.0, 1.0, 1.0, 1.0];
        let w = vec![2.0]; // single weight cycles over all dims
                           // 2*(1+2+3+4) = 20
        let score = CrossEncoder::bilinear_score(&q, &d, &w);
        assert!((score - 20.0).abs() < 1e-9, "expected 20, got {score}");
    }

    #[test]
    fn test_bilinear_empty_weights() {
        let q = vec![1.0, 2.0];
        let d = vec![3.0, 4.0];
        let score = CrossEncoder::bilinear_score(&q, &d, &[]);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_bilinear_model_score_pair() {
        let w = vec![1.0, 0.0]; // only first dimension contributes
        let enc = encoder_with(ScoringModel::BilinearForm(w), false);
        let q = vec![1.0, 999.0];
        let d = vec![0.5, 999.0];
        let s = enc.score_pair(&q, &d);
        assert!((s - 0.5).abs() < 1e-9, "only dim0 => 0.5");
    }

    // ── 4. Linear scoring ─────────────────────────────────────────────────

    #[test]
    fn test_linear_basic() {
        let q = vec![1.0, 2.0];
        let d = vec![3.0, 4.0];
        let w = vec![1.0, 1.0];
        let bias = 0.5;
        // dot([1,1], [3,8]) + 0.5 = (1*3 + 1*8) + 0.5 = 11.5
        let score = CrossEncoder::linear_score(&q, &d, &w, bias);
        assert!((score - 11.5).abs() < 1e-9, "expected 11.5, got {score}");
    }

    #[test]
    fn test_linear_bias_only() {
        let q = vec![1.0];
        let d = vec![1.0];
        let w = vec![0.0]; // zero weight => only bias contributes
        let score = CrossEncoder::linear_score(&q, &d, &w, 42.0);
        assert!((score - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_linear_shorter_weights() {
        let q = vec![1.0, 2.0, 3.0];
        let d = vec![1.0, 1.0, 1.0];
        let w = vec![2.0]; // only first element has weight
                           // 2*1*1 + 0*2*1 + 0*3*1 = 2
        let score = CrossEncoder::linear_score(&q, &d, &w, 0.0);
        assert!((score - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_linear_model_score_pair() {
        let enc = encoder_with(
            ScoringModel::Linear {
                weights: vec![1.0, 1.0],
                bias: 1.0,
            },
            false,
        );
        let q = vec![2.0, 3.0];
        let d = vec![4.0, 5.0];
        // dot([1,1],[8,15]) + 1 = 23 + 1 = 24
        let s = enc.score_pair(&q, &d);
        assert!((s - 24.0).abs() < 1e-9);
    }

    // ── 5. rerank changes order ───────────────────────────────────────────

    #[test]
    fn test_rerank_changes_order() {
        let mut enc = encoder_with(ScoringModel::Cosine, false);
        let query = vec![1.0, 0.0];
        // doc_b has higher initial_score but lower cosine with query
        let candidates = vec![
            make_candidate("doc_a", vec![0.99, 0.01], 0.5),
            make_candidate("doc_b", vec![0.0, 1.0], 0.9),
        ];
        let reranked = enc.rerank(&query, candidates);
        assert_eq!(
            reranked[0].doc_id, "doc_a",
            "doc_a aligns better with query"
        );
        assert_eq!(reranked[1].doc_id, "doc_b");
    }

    #[test]
    fn test_rerank_preserves_initial_score() {
        let mut enc = encoder_with(ScoringModel::DotProduct, false);
        let query = vec![1.0, 0.0];
        let candidates = vec![make_candidate("doc_x", vec![0.5, 0.5], 0.77)];
        let reranked = enc.rerank(&query, candidates);
        assert!((reranked[0].initial_score - 0.77).abs() < 1e-9);
    }

    #[test]
    fn test_rerank_final_rank_numbering() {
        let mut enc = encoder_with(ScoringModel::Cosine, false);
        let query = vec![1.0, 0.0, 0.0];
        let candidates = vec![
            make_candidate("a", vec![0.1, 0.9, 0.0], 0.3),
            make_candidate("b", vec![0.9, 0.1, 0.0], 0.7),
            make_candidate("c", vec![0.5, 0.5, 0.0], 0.5),
        ];
        let reranked = enc.rerank(&query, candidates);
        for (i, doc) in reranked.iter().enumerate() {
            assert_eq!(doc.final_rank, i + 1, "final_rank must be 1-indexed");
        }
    }

    #[test]
    fn test_rerank_empty_candidates() {
        let mut enc = encoder_with(ScoringModel::Cosine, false);
        let result = enc.rerank(&[1.0], vec![]);
        assert!(result.is_empty());
        assert_eq!(enc.stats().total_reranks, 1);
    }

    // ── 6. score_delta ────────────────────────────────────────────────────

    #[test]
    fn test_score_delta_equals_cross_minus_initial() {
        let mut enc = encoder_with(ScoringModel::DotProduct, false);
        let query = vec![1.0, 0.0];
        let candidates = vec![make_candidate("d1", vec![0.3, 0.7], 0.2)];
        let reranked = enc.rerank(&query, candidates);
        let doc = &reranked[0];
        let expected_delta = doc.cross_encoder_score - doc.initial_score;
        assert!(
            (doc.score_delta - expected_delta).abs() < 1e-9,
            "score_delta must equal cross_encoder_score - initial_score"
        );
    }

    // ── 7. normalize_scores ───────────────────────────────────────────────

    #[test]
    fn test_normalize_scores_range() {
        let mut docs = vec![
            RerankedDoc {
                doc_id: "a".into(),
                cross_encoder_score: 2.0,
                initial_score: 0.5,
                final_rank: 1,
                score_delta: 0.0,
            },
            RerankedDoc {
                doc_id: "b".into(),
                cross_encoder_score: 8.0,
                initial_score: 0.5,
                final_rank: 2,
                score_delta: 0.0,
            },
            RerankedDoc {
                doc_id: "c".into(),
                cross_encoder_score: 5.0,
                initial_score: 0.5,
                final_rank: 3,
                score_delta: 0.0,
            },
        ];
        CrossEncoder::normalize_scores(&mut docs);
        let scores: Vec<f64> = docs.iter().map(|d| d.cross_encoder_score).collect();
        assert!(
            scores.iter().all(|&s| (0.0..=1.0).contains(&s)),
            "all in [0,1]"
        );
        let min = scores.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(min.abs() < 1e-9, "min should be 0");
        assert!((max - 1.0).abs() < 1e-9, "max should be 1");
    }

    #[test]
    fn test_normalize_scores_all_equal() {
        let mut docs: Vec<RerankedDoc> = (0..3)
            .map(|i| RerankedDoc {
                doc_id: i.to_string(),
                cross_encoder_score: 0.5,
                initial_score: 0.5,
                final_rank: i + 1,
                score_delta: 0.0,
            })
            .collect();
        CrossEncoder::normalize_scores(&mut docs);
        for d in &docs {
            assert!(
                (d.cross_encoder_score - 1.0).abs() < 1e-9,
                "equal scores => 1.0"
            );
        }
    }

    #[test]
    fn test_normalize_scores_single_doc() {
        let mut docs = vec![RerankedDoc {
            doc_id: "solo".into(),
            cross_encoder_score: 0.42,
            initial_score: 0.1,
            final_rank: 1,
            score_delta: 0.32,
        }];
        CrossEncoder::normalize_scores(&mut docs);
        // Single doc: min == max => all-equal path => 1.0
        assert!((docs[0].cross_encoder_score - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_normalize_scores_empty_slice() {
        let mut docs: Vec<RerankedDoc> = vec![];
        // Should not panic
        CrossEncoder::normalize_scores(&mut docs);
    }

    // ── 8. rank_changed counting ──────────────────────────────────────────

    #[test]
    fn test_rank_changed_no_change() {
        let initial = vec!["a", "b", "c"];
        let reranked = vec![
            RerankedDoc {
                doc_id: "a".into(),
                cross_encoder_score: 0.9,
                initial_score: 0.9,
                final_rank: 1,
                score_delta: 0.0,
            },
            RerankedDoc {
                doc_id: "b".into(),
                cross_encoder_score: 0.7,
                initial_score: 0.7,
                final_rank: 2,
                score_delta: 0.0,
            },
            RerankedDoc {
                doc_id: "c".into(),
                cross_encoder_score: 0.5,
                initial_score: 0.5,
                final_rank: 3,
                score_delta: 0.0,
            },
        ];
        let changed = CrossEncoder::rank_changed(&initial, &reranked);
        assert_eq!(changed, 0, "order unchanged => 0 rank changes");
    }

    #[test]
    fn test_rank_changed_full_reversal() {
        let initial = vec!["a", "b", "c"];
        let reranked = vec![
            RerankedDoc {
                doc_id: "c".into(),
                cross_encoder_score: 0.9,
                initial_score: 0.5,
                final_rank: 1,
                score_delta: 0.4,
            },
            RerankedDoc {
                doc_id: "b".into(),
                cross_encoder_score: 0.7,
                initial_score: 0.7,
                final_rank: 2,
                score_delta: 0.0,
            },
            RerankedDoc {
                doc_id: "a".into(),
                cross_encoder_score: 0.3,
                initial_score: 0.9,
                final_rank: 3,
                score_delta: -0.6,
            },
        ];
        // 'a' moved 0->2, 'b' stayed, 'c' moved 2->0 => 2 changes
        let changed = CrossEncoder::rank_changed(&initial, &reranked);
        assert_eq!(changed, 2);
    }

    #[test]
    fn test_rank_changed_partial() {
        let initial = vec!["a", "b", "c", "d"];
        let reranked = vec![
            RerankedDoc {
                doc_id: "a".into(),
                cross_encoder_score: 0.9,
                initial_score: 0.8,
                final_rank: 1,
                score_delta: 0.1,
            },
            RerankedDoc {
                doc_id: "c".into(),
                cross_encoder_score: 0.7,
                initial_score: 0.6,
                final_rank: 2,
                score_delta: 0.1,
            },
            RerankedDoc {
                doc_id: "b".into(),
                cross_encoder_score: 0.5,
                initial_score: 0.7,
                final_rank: 3,
                score_delta: -0.2,
            },
            RerankedDoc {
                doc_id: "d".into(),
                cross_encoder_score: 0.3,
                initial_score: 0.3,
                final_rank: 4,
                score_delta: 0.0,
            },
        ];
        // 'a' stayed(0->0), 'b' moved(1->2), 'c' moved(2->1), 'd' stayed(3->3) => 2
        let changed = CrossEncoder::rank_changed(&initial, &reranked);
        assert_eq!(changed, 2);
    }

    // ── 9. batch reranking ────────────────────────────────────────────────

    #[test]
    fn test_rerank_batch_two_queries() {
        let mut enc = encoder_with(ScoringModel::Cosine, false);
        let q1 = vec![1.0, 0.0];
        let q2 = vec![0.0, 1.0];
        let c1 = vec![
            make_candidate("a", vec![0.9, 0.1], 0.5),
            make_candidate("b", vec![0.1, 0.9], 0.6),
        ];
        let c2 = vec![
            make_candidate("c", vec![0.1, 0.9], 0.5),
            make_candidate("d", vec![0.9, 0.1], 0.6),
        ];
        let results = enc.rerank_batch(vec![(q1, c1), (q2, c2)]);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0][0].doc_id, "a", "q1 should prefer a");
        assert_eq!(results[1][0].doc_id, "c", "q2 should prefer c");
    }

    #[test]
    fn test_rerank_batch_empty_input() {
        let mut enc = encoder_with(ScoringModel::DotProduct, false);
        let results = enc.rerank_batch(vec![]);
        assert!(results.is_empty());
    }

    #[test]
    fn test_rerank_batch_stats_accumulate() {
        let mut enc = encoder_with(ScoringModel::DotProduct, false);
        let inputs: Vec<(Vec<f64>, Vec<CandidateDoc>)> = (0..5)
            .map(|i| {
                let q = vec![1.0, 0.0];
                let docs = vec![make_candidate(&format!("d{i}"), vec![0.5, 0.5], 0.5)];
                (q, docs)
            })
            .collect();
        enc.rerank_batch(inputs);
        assert_eq!(enc.stats().total_reranks, 5);
        assert_eq!(enc.stats().total_docs_reranked, 5);
    }

    // ── 10. stats tracking ────────────────────────────────────────────────

    #[test]
    fn test_stats_initial_state() {
        let enc = encoder_with(ScoringModel::Cosine, false);
        let s = enc.stats();
        assert_eq!(s.total_reranks, 0);
        assert_eq!(s.total_docs_reranked, 0);
        assert_eq!(s.avg_rank_change, 0.0);
    }

    #[test]
    fn test_stats_after_single_rerank() {
        let mut enc = encoder_with(ScoringModel::Cosine, false);
        let query = vec![1.0, 0.0];
        let candidates = vec![
            make_candidate("x", vec![0.9, 0.1], 0.3),
            make_candidate("y", vec![0.1, 0.9], 0.8),
        ];
        enc.rerank(&query, candidates);
        assert_eq!(enc.stats().total_reranks, 1);
        assert_eq!(enc.stats().total_docs_reranked, 2);
    }

    #[test]
    fn test_stats_multiple_reranks() {
        let mut enc = encoder_with(ScoringModel::DotProduct, false);
        for i in 0..4u64 {
            let q = vec![1.0];
            let docs = vec![
                make_candidate(&format!("d{i}a"), vec![0.5], 0.5),
                make_candidate(&format!("d{i}b"), vec![0.3], 0.3),
            ];
            enc.rerank(&q, docs);
        }
        assert_eq!(enc.stats().total_reranks, 4);
        assert_eq!(enc.stats().total_docs_reranked, 8);
    }

    // ── 11. identical embeddings ──────────────────────────────────────────

    #[test]
    fn test_identical_embeddings_cosine() {
        let enc = encoder_with(ScoringModel::Cosine, false);
        let v = vec![0.3, 0.4, 0.5];
        let s = enc.score_pair(&v, &v);
        assert!((s - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_identical_embeddings_dot() {
        let enc = encoder_with(ScoringModel::DotProduct, false);
        let v = vec![1.0, 2.0];
        let s = enc.score_pair(&v, &v);
        assert!((s - 5.0).abs() < 1e-9, "1+4=5");
    }

    // ── 12. 1-D edge case ─────────────────────────────────────────────────

    #[test]
    fn test_1d_cosine() {
        let a = vec![3.0];
        let b = vec![5.0];
        let s = CrossEncoder::cosine_similarity(&a, &b);
        assert!((s - 1.0).abs() < 1e-9, "same-sign scalars => cos=1");
    }

    #[test]
    fn test_1d_dot_product() {
        let a = vec![7.0];
        let b = vec![3.0];
        let s = CrossEncoder::dot_product(&a, &b);
        assert!((s - 21.0).abs() < 1e-9);
    }

    // ── 13. max_doc_length truncation ─────────────────────────────────────

    #[test]
    fn test_max_doc_length_truncation() {
        let enc = CrossEncoder::new(CrossEncoderConfig {
            model: ScoringModel::DotProduct,
            max_doc_length: 2,
            batch_size: 32,
            normalize_scores: false,
        });
        // Only first 2 dims should count
        let q = vec![1.0, 1.0, 100.0];
        let d = vec![1.0, 1.0, 100.0];
        let s = enc.score_pair(&q, &d);
        // truncated to [1,1] · [1,1] = 2
        assert!((s - 2.0).abs() < 1e-9, "truncated dot should be 2, got {s}");
    }

    // ── 14. normalize_scores via rerank pipeline ──────────────────────────

    #[test]
    fn test_rerank_with_normalization() {
        let mut enc = encoder_with(ScoringModel::DotProduct, true);
        let query = vec![1.0, 0.0];
        let candidates = vec![
            make_candidate("a", vec![10.0, 0.0], 0.5),
            make_candidate("b", vec![5.0, 0.0], 0.3),
            make_candidate("c", vec![1.0, 0.0], 0.1),
        ];
        let reranked = enc.rerank(&query, candidates);
        for doc in &reranked {
            assert!(
                doc.cross_encoder_score >= 0.0 && doc.cross_encoder_score <= 1.0,
                "normalized score out of [0,1]: {}",
                doc.cross_encoder_score
            );
        }
        // Best doc should have score 1.0
        assert!((reranked[0].cross_encoder_score - 1.0).abs() < 1e-9);
    }

    // ── 15. large candidate set ───────────────────────────────────────────

    #[test]
    fn test_large_candidate_set_50_docs() {
        let mut enc = encoder_with(ScoringModel::Cosine, true);
        let dim = 16;
        let query: Vec<f64> = (0..dim).map(|i| i as f64).collect();

        // Generate 50 candidates with varying embeddings
        let candidates: Vec<CandidateDoc> = (0..50)
            .map(|i| {
                let embedding: Vec<f64> = (0..dim).map(|j| (i * dim + j) as f64 * 0.01).collect();
                make_candidate(&format!("doc_{i:02}"), embedding, i as f64 * 0.02)
            })
            .collect();

        let reranked = enc.rerank(&query, candidates);
        assert_eq!(reranked.len(), 50, "all 50 docs should be present");

        // Verify descending order of cross_encoder_score
        for window in reranked.windows(2) {
            assert!(
                window[0].cross_encoder_score >= window[1].cross_encoder_score,
                "output must be sorted descending"
            );
        }

        // All scores in [0, 1] due to normalize_scores=true
        for d in &reranked {
            assert!(d.cross_encoder_score >= 0.0 && d.cross_encoder_score <= 1.0);
        }

        assert_eq!(enc.stats().total_docs_reranked, 50);
    }

    // ── 16. score_delta sign ──────────────────────────────────────────────

    #[test]
    fn test_score_delta_positive_when_improved() {
        let mut enc = encoder_with(ScoringModel::DotProduct, false);
        let query = vec![1.0, 0.0];
        // initial_score=0.1, but dot-product will be 0.9 => positive delta
        let candidates = vec![make_candidate("high", vec![0.9, 0.0], 0.1)];
        let reranked = enc.rerank(&query, candidates);
        assert!(
            reranked[0].score_delta > 0.0,
            "score improved => positive delta"
        );
    }

    #[test]
    fn test_score_delta_negative_when_demoted() {
        let mut enc = encoder_with(ScoringModel::DotProduct, false);
        let query = vec![0.01, 0.0];
        // initial_score=0.9, but dot-product will be 0.009 => negative delta
        let candidates = vec![make_candidate("low", vec![0.9, 0.0], 0.9)];
        let reranked = enc.rerank(&query, candidates);
        assert!(
            reranked[0].score_delta < 0.0,
            "score demoted => negative delta"
        );
    }
}
