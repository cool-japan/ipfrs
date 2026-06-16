//! Cross-Modal Reranker — fuses text (BM25) and vector similarity signals
//! to produce a single ranked list from multi-modal retrieval candidates.
//!
//! # Overview
//!
//! [`CrossModalReranker`] is the primary entry point.  Call
//! [`CrossModalReranker::rerank`] with a candidate list and an optional query
//! text / query embedding.  The ranker:
//!
//! 1. Computes BM25 features against each candidate's `text_snippet`.
//! 2. Computes cosine / dot-product / L2 features against each candidate's
//!    `embedding`.
//! 3. Fuses the per-modality scores with the configured [`CmrFusionStrategy`].
//! 4. Optionally normalises scores to `[0, 1]`, filters by
//!    `min_score_threshold`, and keeps only the top-k results.
//!
//! # Example
//!
//! ```rust
//! use ipfrs_semantic::cross_modal_reranker::{
//!     CrossModalReranker, CmrFusionStrategy, RerankerConfig, RerankerCandidate,
//! };
//!
//! let mut reranker = CrossModalReranker::new(RerankerConfig::default());
//!
//! let candidates = vec![
//!     RerankerCandidate::new("doc1", Some("rust systems programming"), None),
//!     RerankerCandidate::new("doc2", Some("python machine learning"), None),
//! ];
//!
//! let results = reranker.rerank(candidates, Some("rust"), None).unwrap();
//! assert!(!results.is_empty());
//! ```

use std::collections::HashMap;
use std::fmt;

// ─────────────────────────────────────────────────────────────────────────────
// Error
// ─────────────────────────────────────────────────────────────────────────────

/// Errors produced by [`CrossModalReranker`].
#[derive(Debug, Clone, PartialEq)]
pub enum RerankerError {
    /// No candidates were supplied to [`CrossModalReranker::rerank`].
    NoCandidates,
    /// Query and candidate embedding dimension mismatch.
    IncompatibleDimensions {
        /// Expected dimension (query).
        expected: usize,
        /// Dimension received (candidate).
        got: usize,
    },
    /// A weight value is outside the valid range `[0, ∞)`.
    InvalidWeight(f64),
    /// General configuration problem.
    ConfigurationError(String),
}

impl fmt::Display for RerankerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RerankerError::NoCandidates => {
                write!(f, "no candidates provided for reranking")
            }
            RerankerError::IncompatibleDimensions { expected, got } => {
                write!(
                    f,
                    "embedding dimension mismatch: expected {expected}, got {got}"
                )
            }
            RerankerError::InvalidWeight(w) => {
                write!(f, "invalid weight value: {w}")
            }
            RerankerError::ConfigurationError(msg) => {
                write!(f, "configuration error: {msg}")
            }
        }
    }
}

impl std::error::Error for RerankerError {}

// ─────────────────────────────────────────────────────────────────────────────
// Score per modality
// ─────────────────────────────────────────────────────────────────────────────

/// Score contribution from a single retrieval modality.
#[derive(Debug, Clone)]
pub struct ModalityScore {
    /// Name of the modality (e.g. `"text"`, `"vector"`).
    pub modality: String,
    /// Raw, un-normalised score.
    pub raw_score: f64,
    /// Score after min-max normalisation across all candidates.
    pub normalized_score: f64,
    /// Weight applied to this modality during fusion.
    pub weight: f64,
}

impl ModalityScore {
    /// Create a new [`ModalityScore`] with `normalized_score` equal to `raw_score`.
    pub fn new(modality: impl Into<String>, raw_score: f64, weight: f64) -> Self {
        Self {
            modality: modality.into(),
            raw_score,
            normalized_score: raw_score,
            weight,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Candidate
// ─────────────────────────────────────────────────────────────────────────────

/// A single document / chunk that can be reranked.
#[derive(Debug, Clone)]
pub struct RerankerCandidate {
    /// Unique identifier.
    pub id: String,
    /// Optional text used for BM25 scoring.
    pub text_snippet: Option<String>,
    /// Optional dense embedding used for vector scoring.
    pub embedding: Option<Vec<f64>>,
    /// Per-modality breakdown (populated after `rerank`).
    pub modality_scores: Vec<ModalityScore>,
    /// Fused final score (populated after `rerank`).
    pub final_score: f64,
    /// 1-based rank in the result list (populated after `rerank`).
    pub rank: usize,
}

impl RerankerCandidate {
    /// Convenience constructor.
    pub fn new(
        id: impl Into<String>,
        text_snippet: Option<&str>,
        embedding: Option<Vec<f64>>,
    ) -> Self {
        Self {
            id: id.into(),
            text_snippet: text_snippet.map(str::to_owned),
            embedding,
            modality_scores: Vec::new(),
            final_score: 0.0,
            rank: 0,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// BM25 / text features
// ─────────────────────────────────────────────────────────────────────────────

/// Features derived from BM25-style keyword matching.
#[derive(Debug, Clone)]
pub struct TextFeatures {
    /// Per-term `(term, tf_idf_contribution)` pairs.
    pub term_frequency: Vec<(String, f64)>,
    /// Aggregate BM25 score.
    pub bm25_score: f64,
    /// `+0.5` bonus when the whole query is an exact sub-string of the text.
    pub exact_match_bonus: f64,
    /// Length penalty ∈ `(-∞, 1.0]`; penalises documents longer than average.
    pub length_penalty: f64,
}

// ─────────────────────────────────────────────────────────────────────────────
// Vector features
// ─────────────────────────────────────────────────────────────────────────────

/// Features derived from dense vector comparison.
#[derive(Debug, Clone)]
pub struct VectorFeatures {
    /// Cosine similarity ∈ `[-1, 1]`.
    pub cosine_similarity: f64,
    /// Raw dot product.
    pub dot_product: f64,
    /// Euclidean (L2) distance ∈ `[0, ∞)`.
    pub l2_distance: f64,
    /// `1 / (1 + l2_distance)` — bounded similarity in `(0, 1]`.
    pub euclidean_normalized: f64,
}

// ─────────────────────────────────────────────────────────────────────────────
// Fusion strategy
// ─────────────────────────────────────────────────────────────────────────────

/// How scores from different modalities are combined into a single ranking.
///
/// Note: This type is named `CmrFusionStrategy` in the public crate API to
/// avoid conflicting with `FusionStrategy` already exported from
/// `multimodal_search`.  Within this module it is also re-exported as the
/// canonical name.
#[derive(Debug, Clone)]
pub enum CmrFusionStrategy {
    /// Weighted linear combination: `Σ weight_i * score_i`.
    /// The `Vec` contains `(modality_name, weight)` pairs.
    LinearCombination(Vec<(String, f64)>),
    /// Reciprocal Rank Fusion: `score = Σ 1 / (k + rank_i)`.
    ReciprocalRankFusion(f64),
    /// Borda count — rank-based voting.
    Borda,
    /// Keep the maximum individual modality score.
    MaxScore,
    /// Pre-trained / supplied weight vector applied to ordered modality scores.
    LearnedWeights(Vec<f64>),
}

impl Default for CmrFusionStrategy {
    fn default() -> Self {
        CmrFusionStrategy::LinearCombination(vec![
            ("text".to_string(), 0.4),
            ("vector".to_string(), 0.6),
        ])
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Config
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for [`CrossModalReranker`].
#[derive(Debug, Clone)]
pub struct RerankerConfig {
    /// Score fusion strategy.
    pub fusion_strategy: CmrFusionStrategy,
    /// Global text-modality weight (used when `fusion_strategy` is not
    /// `LinearCombination`).
    pub text_weight: f64,
    /// Global vector-modality weight.
    pub vector_weight: f64,
    /// BM25 term-frequency saturation constant (typical: 1.2–2.0).
    pub bm25_k1: f64,
    /// BM25 length normalisation constant (typical: 0.75).
    pub bm25_b: f64,
    /// Normalise final scores to `[0, 1]`.
    pub normalize_scores: bool,
    /// Discard candidates whose final score is below this threshold.
    pub min_score_threshold: f64,
    /// Maximum number of results to return.  `0` means no limit.
    pub top_k: usize,
}

impl Default for RerankerConfig {
    fn default() -> Self {
        Self {
            fusion_strategy: CmrFusionStrategy::default(),
            text_weight: 0.4,
            vector_weight: 0.6,
            bm25_k1: 1.5,
            bm25_b: 0.75,
            normalize_scores: true,
            min_score_threshold: 0.0,
            top_k: 100,
        }
    }
}

impl RerankerConfig {
    /// Validate all weight fields, returning an error on invalid values.
    fn validate(&self) -> Result<(), RerankerError> {
        if self.text_weight < 0.0 || self.text_weight.is_nan() {
            return Err(RerankerError::InvalidWeight(self.text_weight));
        }
        if self.vector_weight < 0.0 || self.vector_weight.is_nan() {
            return Err(RerankerError::InvalidWeight(self.vector_weight));
        }
        if self.bm25_k1 < 0.0 || self.bm25_k1.is_nan() {
            return Err(RerankerError::ConfigurationError(
                "bm25_k1 must be non-negative".to_string(),
            ));
        }
        if !(0.0..=1.0).contains(&self.bm25_b) {
            return Err(RerankerError::ConfigurationError(
                "bm25_b must be in [0, 1]".to_string(),
            ));
        }
        // Validate LinearCombination weights
        if let CmrFusionStrategy::LinearCombination(ref pairs) = self.fusion_strategy {
            for (_, w) in pairs {
                if *w < 0.0 || w.is_nan() {
                    return Err(RerankerError::InvalidWeight(*w));
                }
            }
        }
        if let CmrFusionStrategy::ReciprocalRankFusion(k) = self.fusion_strategy {
            if k <= 0.0 || k.is_nan() {
                return Err(RerankerError::ConfigurationError(
                    "RRF k must be positive".to_string(),
                ));
            }
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Stats
// ─────────────────────────────────────────────────────────────────────────────

/// Operational statistics for [`CrossModalReranker`].
#[derive(Debug, Clone, Default)]
pub struct RerankerStats {
    /// Total candidates processed across all `rerank` calls.
    pub candidates_reranked: u64,
    /// Average displacement of rank position (abs(new_rank - old_rank)).
    pub avg_rank_displacement: f64,
    /// Distinct modality names observed.
    pub modalities_used: Vec<String>,
    /// Number of fusion operations performed.
    pub fusion_calls: u64,
}

// ─────────────────────────────────────────────────────────────────────────────
// Tokenizer
// ─────────────────────────────────────────────────────────────────────────────

/// Simple whitespace tokenizer with alphabetic filtering.
/// Uses the same semantics as the FNV-1a-compatible form in the spec.
fn tokenize(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|w| {
            w.to_lowercase()
                .trim_matches(|c: char| !c.is_alphabetic())
                .to_string()
        })
        .filter(|w| !w.is_empty())
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Pure PRNG (for tests, not used in production logic)
// ─────────────────────────────────────────────────────────────────────────────

/// XorShift-64 PRNG — integer step.
#[allow(dead_code)]
fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// XorShift-64 PRNG — float step returning a value in `[0, 1)`.
#[allow(dead_code)]
fn xorshift_f64(state: &mut u64) -> f64 {
    (xorshift64(state) >> 11) as f64 / (1u64 << 53) as f64
}

// ─────────────────────────────────────────────────────────────────────────────
// CrossModalReranker
// ─────────────────────────────────────────────────────────────────────────────

/// Cross-modal reranker that fuses BM25 text scores with dense vector scores.
pub struct CrossModalReranker {
    config: RerankerConfig,
    stats: RerankerStats,
}

impl CrossModalReranker {
    /// Create a new reranker with the supplied configuration.
    pub fn new(config: RerankerConfig) -> Self {
        Self {
            config,
            stats: RerankerStats::default(),
        }
    }

    /// Replace the current configuration.
    pub fn update_config(&mut self, config: RerankerConfig) {
        self.config = config;
    }

    /// Return a snapshot of the current operational statistics.
    pub fn stats(&self) -> RerankerStats {
        self.stats.clone()
    }

    // ──────────────────────────────────────────────────────────────────────
    // BM25 text features
    // ──────────────────────────────────────────────────────────────────────

    /// Compute BM25-derived text features for a single `(query, text)` pair.
    ///
    /// `avg_doc_len` is the average document token count across the corpus;
    /// pass `0.0` (or any non-positive value) to fall back to `1.0`.
    pub fn compute_text_features(&self, query: &str, text: &str, avg_doc_len: f64) -> TextFeatures {
        let avg_doc_len = if avg_doc_len > 0.0 { avg_doc_len } else { 1.0 };

        let query_tokens = tokenize(query);
        let doc_tokens = tokenize(text);
        let doc_len = doc_tokens.len() as f64;

        // Term frequency map for the document
        let mut tf_map: HashMap<String, f64> = HashMap::new();
        for tok in &doc_tokens {
            *tf_map.entry(tok.clone()).or_insert(0.0) += 1.0;
        }

        let k1 = self.config.bm25_k1;
        let b = self.config.bm25_b;

        // N = 1 (single-document estimate)
        // IDF = ln((N - df + 0.5) / (df + 0.5) + 1) where df = 1 if term present, 0 otherwise
        let n = 1.0_f64;
        let mut term_contributions: Vec<(String, f64)> = Vec::new();
        let mut bm25_total = 0.0_f64;

        for qt in &query_tokens {
            let freq = tf_map.get(qt).copied().unwrap_or(0.0);
            let df = if freq > 0.0 { 1.0 } else { 0.0 };

            let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();
            let tf_norm = (freq * (k1 + 1.0)) / (freq + k1 * (1.0 - b + b * doc_len / avg_doc_len));

            let contribution = idf * tf_norm;
            bm25_total += contribution;
            term_contributions.push((qt.clone(), contribution));
        }

        let exact_match_bonus =
            if !query.is_empty() && text.to_lowercase().contains(&query.to_lowercase()) {
                0.5
            } else {
                0.0
            };

        let length_penalty = 1.0 - 0.1 * ((doc_len / avg_doc_len) - 1.0).max(0.0);

        TextFeatures {
            term_frequency: term_contributions,
            bm25_score: bm25_total,
            exact_match_bonus,
            length_penalty,
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    // Vector features
    // ──────────────────────────────────────────────────────────────────────

    /// Compute vector-space features between a query embedding and a candidate
    /// embedding.
    ///
    /// Returns [`RerankerError::IncompatibleDimensions`] if the slices have
    /// different lengths.
    pub fn compute_vector_features(
        query: &[f64],
        candidate: &[f64],
    ) -> Result<VectorFeatures, RerankerError> {
        if query.len() != candidate.len() {
            return Err(RerankerError::IncompatibleDimensions {
                expected: query.len(),
                got: candidate.len(),
            });
        }

        let mut dot = 0.0_f64;
        let mut norm_q = 0.0_f64;
        let mut norm_c = 0.0_f64;
        let mut sq_diff = 0.0_f64;

        for (q, c) in query.iter().zip(candidate.iter()) {
            dot += q * c;
            norm_q += q * q;
            norm_c += c * c;
            let d = q - c;
            sq_diff += d * d;
        }

        let norm_q = norm_q.sqrt();
        let norm_c = norm_c.sqrt();
        let denom = norm_q * norm_c;

        let cosine_similarity = if denom > 0.0 { dot / denom } else { 0.0 };
        let l2_distance = sq_diff.sqrt();
        let euclidean_normalized = 1.0 / (1.0 + l2_distance);

        Ok(VectorFeatures {
            cosine_similarity,
            dot_product: dot,
            l2_distance,
            euclidean_normalized,
        })
    }

    // ──────────────────────────────────────────────────────────────────────
    // Reciprocal Rank Fusion
    // ──────────────────────────────────────────────────────────────────────

    /// Merge multiple ranked lists using Reciprocal Rank Fusion.
    ///
    /// `rank_lists` is a list of ranked ID lists (index 0 = rank 1).
    /// Returns `(id, rrf_score)` pairs sorted by descending score.
    pub fn reciprocal_rank_fusion(rank_lists: Vec<Vec<String>>, k: f64) -> Vec<(String, f64)> {
        let mut scores: HashMap<String, f64> = HashMap::new();

        for list in &rank_lists {
            for (rank_zero_based, id) in list.iter().enumerate() {
                let rank = (rank_zero_based + 1) as f64;
                *scores.entry(id.clone()).or_insert(0.0) += 1.0 / (k + rank);
            }
        }

        let mut result: Vec<(String, f64)> = scores.into_iter().collect();
        result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        result
    }

    // ──────────────────────────────────────────────────────────────────────
    // Main rerank entry point
    // ──────────────────────────────────────────────────────────────────────

    /// Rerank the given candidates by fusing text and/or vector similarity
    /// with the configured [`CmrFusionStrategy`].
    ///
    /// * `query_text`      — used to compute BM25 features (optional).
    /// * `query_embedding` — used to compute vector features (optional).
    ///
    /// The returned list is sorted descending by `final_score`, filtered
    /// by `min_score_threshold`, and limited to `top_k` entries.
    pub fn rerank(
        &mut self,
        mut candidates: Vec<RerankerCandidate>,
        query_text: Option<&str>,
        query_embedding: Option<&[f64]>,
    ) -> Result<Vec<RerankerCandidate>, RerankerError> {
        if candidates.is_empty() {
            return Err(RerankerError::NoCandidates);
        }

        self.config.validate()?;

        // Validate embedding dimensions up-front
        if let Some(qe) = query_embedding {
            for c in &candidates {
                if let Some(ce) = &c.embedding {
                    if ce.len() != qe.len() {
                        return Err(RerankerError::IncompatibleDimensions {
                            expected: qe.len(),
                            got: ce.len(),
                        });
                    }
                }
            }
        }

        // Compute average document length for BM25
        let avg_doc_len = {
            let texts: Vec<usize> = candidates
                .iter()
                .filter_map(|c| c.text_snippet.as_ref())
                .map(|t| tokenize(t).len())
                .collect();
            if texts.is_empty() {
                1.0
            } else {
                texts.iter().sum::<usize>() as f64 / texts.len() as f64
            }
        };

        // Assign modality scores to each candidate
        for cand in candidates.iter_mut() {
            cand.modality_scores.clear();

            // ── Text ──
            if let (Some(qt), Some(snippet)) = (query_text, cand.text_snippet.as_deref()) {
                let tf = self.compute_text_features(qt, snippet, avg_doc_len);
                let text_score = (tf.bm25_score + tf.exact_match_bonus) * tf.length_penalty;
                cand.modality_scores.push(ModalityScore::new(
                    "text",
                    text_score,
                    self.config.text_weight,
                ));
            }

            // ── Vector ──
            if let (Some(qe), Some(ce)) = (query_embedding, cand.embedding.as_deref()) {
                // Dimension already validated above
                let vf = Self::compute_vector_features(qe, ce)?;
                cand.modality_scores.push(ModalityScore::new(
                    "vector",
                    vf.cosine_similarity,
                    self.config.vector_weight,
                ));
            }
        }

        // Fuse scores
        self.apply_fusion(&mut candidates)?;

        // Sort descending
        candidates.sort_by(|a, b| {
            b.final_score
                .partial_cmp(&a.final_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Normalise if requested (before filtering / truncation)
        if self.config.normalize_scores {
            Self::normalize_scores(&mut candidates);
        }

        // Assign pre-filter ranks to compute displacement later
        let original_ranks: Vec<(String, usize)> = candidates
            .iter()
            .enumerate()
            .map(|(i, c)| (c.id.clone(), i + 1))
            .collect();

        // Filter by min_score_threshold
        candidates.retain(|c| c.final_score >= self.config.min_score_threshold);

        // Limit to top_k
        if self.config.top_k > 0 && candidates.len() > self.config.top_k {
            candidates.truncate(self.config.top_k);
        }

        // Assign final 1-based ranks
        for (i, c) in candidates.iter_mut().enumerate() {
            c.rank = i + 1;
        }

        // Update stats
        let total = candidates.len() as u64;
        let displacement: f64 = candidates
            .iter()
            .map(|c| {
                original_ranks
                    .iter()
                    .find(|(id, _)| id == &c.id)
                    .map(|(_, orig)| (c.rank as i64 - *orig as i64).unsigned_abs() as f64)
                    .unwrap_or(0.0)
            })
            .sum::<f64>()
            / total.max(1) as f64;

        self.stats.candidates_reranked += total;
        self.stats.fusion_calls += 1;
        // Rolling average of displacement
        if self.stats.fusion_calls == 1 {
            self.stats.avg_rank_displacement = displacement;
        } else {
            let n = self.stats.fusion_calls as f64;
            self.stats.avg_rank_displacement =
                (self.stats.avg_rank_displacement * (n - 1.0) + displacement) / n;
        }

        // Track modalities
        for c in &candidates {
            for ms in &c.modality_scores {
                if !self.stats.modalities_used.contains(&ms.modality) {
                    self.stats.modalities_used.push(ms.modality.clone());
                }
            }
        }

        Ok(candidates)
    }

    // ──────────────────────────────────────────────────────────────────────
    // Internal: per-strategy fusion
    // ──────────────────────────────────────────────────────────────────────

    fn apply_fusion(&self, candidates: &mut [RerankerCandidate]) -> Result<(), RerankerError> {
        match &self.config.fusion_strategy {
            CmrFusionStrategy::LinearCombination(pairs) => {
                let weight_map: HashMap<&str, f64> =
                    pairs.iter().map(|(k, v)| (k.as_str(), *v)).collect();

                for cand in candidates.iter_mut() {
                    let score: f64 = cand
                        .modality_scores
                        .iter()
                        .map(|ms| {
                            let w = weight_map
                                .get(ms.modality.as_str())
                                .copied()
                                .unwrap_or(ms.weight);
                            w * ms.raw_score
                        })
                        .sum();
                    cand.final_score = score;
                }
            }

            CmrFusionStrategy::ReciprocalRankFusion(k) => {
                let k = *k;
                // Build per-modality rank lists
                let modality_names: Vec<String> = {
                    let mut names: Vec<String> = Vec::new();
                    for c in candidates.iter() {
                        for ms in &c.modality_scores {
                            if !names.contains(&ms.modality) {
                                names.push(ms.modality.clone());
                            }
                        }
                    }
                    names
                };

                // Per-modality sorted ID lists (by raw_score desc)
                let rank_lists: Vec<Vec<String>> = modality_names
                    .iter()
                    .map(|m| {
                        let mut scored: Vec<(String, f64)> = candidates
                            .iter()
                            .filter_map(|c| {
                                c.modality_scores
                                    .iter()
                                    .find(|ms| &ms.modality == m)
                                    .map(|ms| (c.id.clone(), ms.raw_score))
                            })
                            .collect();
                        scored.sort_by(|a, b| {
                            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                        });
                        scored.into_iter().map(|(id, _)| id).collect()
                    })
                    .collect();

                let rrf_scores = Self::reciprocal_rank_fusion(rank_lists, k);
                let score_map: HashMap<&str, f64> =
                    rrf_scores.iter().map(|(id, s)| (id.as_str(), *s)).collect();

                for cand in candidates.iter_mut() {
                    cand.final_score = score_map.get(cand.id.as_str()).copied().unwrap_or(0.0);
                }
            }

            CmrFusionStrategy::Borda => {
                // Borda count: for each modality rank list, candidate at rank r
                // receives (N - r) points.
                let n = candidates.len();
                let modality_names: Vec<String> = {
                    let mut names: Vec<String> = Vec::new();
                    for c in candidates.iter() {
                        for ms in &c.modality_scores {
                            if !names.contains(&ms.modality) {
                                names.push(ms.modality.clone());
                            }
                        }
                    }
                    names
                };

                let mut borda_totals: HashMap<String, f64> =
                    candidates.iter().map(|c| (c.id.clone(), 0.0)).collect();

                for m in &modality_names {
                    let mut scored: Vec<(String, f64)> = candidates
                        .iter()
                        .filter_map(|c| {
                            c.modality_scores
                                .iter()
                                .find(|ms| &ms.modality == m)
                                .map(|ms| (c.id.clone(), ms.raw_score))
                        })
                        .collect();
                    scored
                        .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

                    for (rank_zero, (id, _)) in scored.iter().enumerate() {
                        let points = (n - rank_zero) as f64;
                        if let Some(total) = borda_totals.get_mut(id) {
                            *total += points;
                        }
                    }
                }

                for cand in candidates.iter_mut() {
                    cand.final_score = borda_totals.get(&cand.id).copied().unwrap_or(0.0);
                }
            }

            CmrFusionStrategy::MaxScore => {
                for cand in candidates.iter_mut() {
                    cand.final_score = cand
                        .modality_scores
                        .iter()
                        .map(|ms| ms.raw_score)
                        .fold(f64::NEG_INFINITY, f64::max);
                    if cand.final_score.is_infinite() {
                        cand.final_score = 0.0;
                    }
                }
            }

            CmrFusionStrategy::LearnedWeights(weights) => {
                for cand in candidates.iter_mut() {
                    let score: f64 = cand
                        .modality_scores
                        .iter()
                        .enumerate()
                        .map(|(i, ms)| {
                            let w = weights.get(i).copied().unwrap_or(1.0);
                            w * ms.raw_score
                        })
                        .sum();
                    cand.final_score = score;
                }
            }
        }

        Ok(())
    }

    // ──────────────────────────────────────────────────────────────────────
    // Internal: score normalisation
    // ──────────────────────────────────────────────────────────────────────

    fn normalize_scores(candidates: &mut [RerankerCandidate]) {
        if candidates.is_empty() {
            return;
        }
        let min_s = candidates
            .iter()
            .map(|c| c.final_score)
            .fold(f64::INFINITY, f64::min);
        let max_s = candidates
            .iter()
            .map(|c| c.final_score)
            .fold(f64::NEG_INFINITY, f64::max);

        let range = max_s - min_s;
        if range < f64::EPSILON {
            for c in candidates.iter_mut() {
                c.final_score = 1.0;
            }
            return;
        }
        for c in candidates.iter_mut() {
            c.final_score = (c.final_score - min_s) / range;
        }

        // Also normalise per-modality scores
        // (collect modality names first to avoid borrow conflicts)
        let modality_names: Vec<String> = {
            let mut names: Vec<String> = Vec::new();
            for c in candidates.iter() {
                for ms in &c.modality_scores {
                    if !names.contains(&ms.modality) {
                        names.push(ms.modality.clone());
                    }
                }
            }
            names
        };

        for m in &modality_names {
            let min_r = candidates
                .iter()
                .flat_map(|c| c.modality_scores.iter())
                .filter(|ms| &ms.modality == m)
                .map(|ms| ms.raw_score)
                .fold(f64::INFINITY, f64::min);
            let max_r = candidates
                .iter()
                .flat_map(|c| c.modality_scores.iter())
                .filter(|ms| &ms.modality == m)
                .map(|ms| ms.raw_score)
                .fold(f64::NEG_INFINITY, f64::max);

            let r = max_r - min_r;
            for c in candidates.iter_mut() {
                for ms in c.modality_scores.iter_mut() {
                    if &ms.modality == m {
                        ms.normalized_score = if r < f64::EPSILON {
                            1.0
                        } else {
                            (ms.raw_score - min_r) / r
                        };
                    }
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ──────────────────────────────────────────────────────────────

    fn make_text_candidate(id: &str, text: &str) -> RerankerCandidate {
        RerankerCandidate::new(id, Some(text), None)
    }

    fn make_vec_candidate(id: &str, embedding: Vec<f64>) -> RerankerCandidate {
        RerankerCandidate::new(id, None, Some(embedding))
    }

    fn make_full_candidate(id: &str, text: &str, embedding: Vec<f64>) -> RerankerCandidate {
        RerankerCandidate::new(id, Some(text), Some(embedding))
    }

    fn default_reranker() -> CrossModalReranker {
        CrossModalReranker::new(RerankerConfig::default())
    }

    // ── tokenizer ────────────────────────────────────────────────────────────

    #[test]
    fn test_tokenize_basic() {
        let tokens = tokenize("Hello, World!");
        assert_eq!(tokens, vec!["hello", "world"]);
    }

    #[test]
    fn test_tokenize_empty() {
        assert!(tokenize("").is_empty());
    }

    #[test]
    fn test_tokenize_punctuation_stripped() {
        let tokens = tokenize("rust, systems, programming.");
        assert_eq!(tokens, vec!["rust", "systems", "programming"]);
    }

    #[test]
    fn test_tokenize_lowercase() {
        let tokens = tokenize("Rust PROGRAMMING");
        assert!(tokens
            .iter()
            .all(|t| t.chars().all(|c| c.is_lowercase() || !c.is_alphabetic())));
    }

    // ── PRNG ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_xorshift64_not_zero_after_seed() {
        let mut state: u64 = 12345;
        let v = xorshift64(&mut state);
        assert_ne!(v, 0);
    }

    #[test]
    fn test_xorshift_f64_in_range() {
        let mut state: u64 = 99999;
        for _ in 0..1000 {
            let v = xorshift_f64(&mut state);
            assert!((0.0..1.0).contains(&v), "value out of range: {v}");
        }
    }

    #[test]
    fn test_xorshift_f64_deterministic() {
        let mut s1: u64 = 42;
        let mut s2: u64 = 42;
        assert_eq!(xorshift_f64(&mut s1), xorshift_f64(&mut s2));
    }

    // ── BM25 text features ────────────────────────────────────────────────────

    #[test]
    fn test_bm25_empty_query() {
        let r = default_reranker();
        let tf = r.compute_text_features("", "some text here", 4.0);
        assert_eq!(tf.bm25_score, 0.0);
    }

    #[test]
    fn test_bm25_empty_document() {
        let r = default_reranker();
        let tf = r.compute_text_features("rust", "", 4.0);
        assert_eq!(tf.bm25_score, 0.0);
    }

    #[test]
    fn test_bm25_term_present_vs_absent() {
        let r = default_reranker();
        let tf_present = r.compute_text_features("rust", "rust systems", 2.0);
        let tf_absent = r.compute_text_features("rust", "python systems", 2.0);
        assert!(tf_present.bm25_score > tf_absent.bm25_score);
    }

    #[test]
    fn test_bm25_exact_match_bonus() {
        let r = default_reranker();
        let tf_exact =
            r.compute_text_features("rust programming", "I love rust programming a lot", 5.0);
        let tf_partial =
            r.compute_text_features("rust programming", "I love rust and programming", 5.0);
        assert!(
            tf_exact.exact_match_bonus > tf_partial.exact_match_bonus,
            "exact match should have bonus: exact={}, partial={}",
            tf_exact.exact_match_bonus,
            tf_partial.exact_match_bonus
        );
    }

    #[test]
    fn test_bm25_exact_match_bonus_value() {
        let r = default_reranker();
        let tf = r.compute_text_features("hello world", "hello world this is a test", 5.0);
        assert!((tf.exact_match_bonus - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_bm25_no_exact_match_bonus() {
        let r = default_reranker();
        let tf = r.compute_text_features("hello world", "goodbye everyone", 5.0);
        assert_eq!(tf.exact_match_bonus, 0.0);
    }

    #[test]
    fn test_bm25_length_penalty_short_doc() {
        let r = default_reranker();
        // doc shorter than avg → no penalty
        let tf = r.compute_text_features("a", "a", 100.0);
        assert!((tf.length_penalty - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_bm25_length_penalty_long_doc() {
        let r = default_reranker();
        let long_text = "word ".repeat(100);
        let tf = r.compute_text_features("word", long_text.trim(), 10.0);
        assert!(tf.length_penalty < 1.0);
    }

    #[test]
    fn test_bm25_term_frequency_populated() {
        let r = default_reranker();
        let tf = r.compute_text_features("rust python", "rust is great", 3.0);
        assert!(!tf.term_frequency.is_empty());
    }

    #[test]
    fn test_bm25_zero_avg_doc_len_fallback() {
        let r = default_reranker();
        let tf = r.compute_text_features("hello", "hello world", 0.0);
        // should not panic, bm25_score should be finite
        assert!(tf.bm25_score.is_finite());
    }

    #[test]
    fn test_bm25_custom_k1_b() {
        let config = RerankerConfig {
            bm25_k1: 2.0,
            bm25_b: 0.5,
            ..Default::default()
        };
        let r = CrossModalReranker::new(config);
        let tf = r.compute_text_features("rust", "rust systems rust", 3.0);
        assert!(tf.bm25_score > 0.0);
    }

    // ── vector features ───────────────────────────────────────────────────────

    #[test]
    fn test_vector_features_identical() {
        let v = vec![1.0, 0.0, 0.0];
        let vf = CrossModalReranker::compute_vector_features(&v, &v)
            .expect("test: identical vectors should compute without error");
        assert!((vf.cosine_similarity - 1.0).abs() < 1e-10);
        assert!(vf.l2_distance.abs() < 1e-10);
        assert!((vf.euclidean_normalized - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_vector_features_orthogonal() {
        let q = vec![1.0, 0.0];
        let c = vec![0.0, 1.0];
        let vf = CrossModalReranker::compute_vector_features(&q, &c)
            .expect("test: orthogonal vectors should compute without error");
        assert!(vf.cosine_similarity.abs() < 1e-10);
    }

    #[test]
    fn test_vector_features_opposite() {
        let q = vec![1.0, 0.0];
        let c = vec![-1.0, 0.0];
        let vf = CrossModalReranker::compute_vector_features(&q, &c)
            .expect("test: opposite vectors should compute without error");
        assert!((vf.cosine_similarity + 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_vector_features_dimension_mismatch() {
        let q = vec![1.0, 2.0, 3.0];
        let c = vec![1.0, 2.0];
        let err = CrossModalReranker::compute_vector_features(&q, &c)
            .expect_err("test: dimension mismatch should return error");
        assert_eq!(
            err,
            RerankerError::IncompatibleDimensions {
                expected: 3,
                got: 2
            }
        );
    }

    #[test]
    fn test_vector_features_zero_vector() {
        let q = vec![0.0, 0.0];
        let c = vec![1.0, 0.0];
        let vf = CrossModalReranker::compute_vector_features(&q, &c)
            .expect("test: zero query vector should compute without error");
        // zero query → cosine = 0
        assert_eq!(vf.cosine_similarity, 0.0);
    }

    #[test]
    fn test_vector_features_dot_product() {
        let q = vec![1.0, 2.0, 3.0];
        let c = vec![4.0, 5.0, 6.0];
        let vf = CrossModalReranker::compute_vector_features(&q, &c)
            .expect("test: dot product computation should succeed");
        assert!((vf.dot_product - 32.0).abs() < 1e-10);
    }

    #[test]
    fn test_vector_features_l2_distance() {
        let q = vec![0.0, 0.0];
        let c = vec![3.0, 4.0];
        let vf = CrossModalReranker::compute_vector_features(&q, &c)
            .expect("test: L2 distance computation should succeed");
        assert!((vf.l2_distance - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_vector_features_euclidean_normalized_bounded() {
        let mut state: u64 = 777;
        let q: Vec<f64> = (0..8).map(|_| xorshift_f64(&mut state)).collect();
        let c: Vec<f64> = (0..8).map(|_| xorshift_f64(&mut state)).collect();
        let vf = CrossModalReranker::compute_vector_features(&q, &c)
            .expect("test: euclidean_normalized computation should succeed");
        assert!(vf.euclidean_normalized > 0.0);
        assert!(vf.euclidean_normalized <= 1.0);
    }

    // ── RRF ───────────────────────────────────────────────────────────────────

    #[test]
    fn test_rrf_single_list() {
        let lists = vec![vec!["a".to_string(), "b".to_string(), "c".to_string()]];
        let scores = CrossModalReranker::reciprocal_rank_fusion(lists, 60.0);
        // a at rank 1 → 1/61
        let a_score = scores
            .iter()
            .find(|(id, _)| id == "a")
            .expect("test: 'a' must be in RRF scores")
            .1;
        let b_score = scores
            .iter()
            .find(|(id, _)| id == "b")
            .expect("test: 'b' must be in RRF scores")
            .1;
        assert!(a_score > b_score);
    }

    #[test]
    fn test_rrf_two_lists_consensus() {
        let lists = vec![
            vec!["a".to_string(), "b".to_string()],
            vec!["a".to_string(), "b".to_string()],
        ];
        let scores = CrossModalReranker::reciprocal_rank_fusion(lists, 60.0);
        let a = scores
            .iter()
            .find(|(id, _)| id == "a")
            .expect("test: 'a' must be in RRF scores")
            .1;
        let b = scores
            .iter()
            .find(|(id, _)| id == "b")
            .expect("test: 'b' must be in RRF scores")
            .1;
        assert!(a > b);
    }

    #[test]
    fn test_rrf_rank_disagreement() {
        let lists = vec![
            vec!["a".to_string(), "b".to_string()],
            vec!["b".to_string(), "a".to_string()],
        ];
        let scores = CrossModalReranker::reciprocal_rank_fusion(lists, 60.0);
        // Equal ranking → scores should be equal
        let a = scores
            .iter()
            .find(|(id, _)| id == "a")
            .expect("test: 'a' must be in RRF scores")
            .1;
        let b = scores
            .iter()
            .find(|(id, _)| id == "b")
            .expect("test: 'b' must be in RRF scores")
            .1;
        assert!((a - b).abs() < 1e-10, "a={a}, b={b}");
    }

    #[test]
    fn test_rrf_custom_k() {
        let lists = vec![vec!["x".to_string()]];
        let s1 = CrossModalReranker::reciprocal_rank_fusion(lists.clone(), 10.0);
        let s2 = CrossModalReranker::reciprocal_rank_fusion(lists, 100.0);
        // smaller k → larger score
        let v1 = s1[0].1;
        let v2 = s2[0].1;
        assert!(v1 > v2);
    }

    #[test]
    fn test_rrf_empty_lists() {
        let scores = CrossModalReranker::reciprocal_rank_fusion(vec![], 60.0);
        assert!(scores.is_empty());
    }

    #[test]
    fn test_rrf_sorted_descending() {
        let lists = vec![
            vec!["c".to_string(), "b".to_string(), "a".to_string()],
            vec!["a".to_string(), "b".to_string(), "c".to_string()],
        ];
        let scores = CrossModalReranker::reciprocal_rank_fusion(lists, 60.0);
        for w in scores.windows(2) {
            assert!(w[0].1 >= w[1].1);
        }
    }

    // ── text-only rerank ──────────────────────────────────────────────────────

    #[test]
    fn test_text_only_rerank_ordering() {
        let mut r = default_reranker();
        let candidates = vec![
            make_text_candidate("doc1", "python machine learning"),
            make_text_candidate("doc2", "rust systems programming rust"),
        ];
        let results = r
            .rerank(candidates, Some("rust"), None)
            .expect("test: rerank should succeed");
        assert_eq!(results[0].id, "doc2");
    }

    #[test]
    fn test_text_only_rerank_ranks_assigned() {
        let mut r = default_reranker();
        let candidates = vec![
            make_text_candidate("a", "foo"),
            make_text_candidate("b", "foo bar"),
            make_text_candidate("c", "foo bar baz"),
        ];
        let results = r
            .rerank(candidates, Some("foo"), None)
            .expect("test: rerank should succeed");
        for (i, res) in results.iter().enumerate() {
            assert_eq!(res.rank, i + 1);
        }
    }

    #[test]
    fn test_text_only_empty_query_still_returns() {
        let mut r = default_reranker();
        let candidates = vec![make_text_candidate("a", "hello world")];
        let results = r
            .rerank(candidates, Some(""), None)
            .expect("test: rerank should succeed");
        assert_eq!(results.len(), 1);
    }

    // ── vector-only rerank ────────────────────────────────────────────────────

    #[test]
    fn test_vector_only_rerank_ordering() {
        let mut r = default_reranker();
        let query = vec![1.0_f64, 0.0];
        let close = make_vec_candidate("close", vec![0.99, 0.14]);
        let far = make_vec_candidate("far", vec![0.0, 1.0]);
        let results = r
            .rerank(vec![far, close], None, Some(&query))
            .expect("test: rerank should succeed");
        assert_eq!(results[0].id, "close");
    }

    #[test]
    fn test_vector_only_rerank_scores_finite() {
        let mut r = default_reranker();
        let mut state: u64 = 1234;
        let q: Vec<f64> = (0..16).map(|_| xorshift_f64(&mut state)).collect();
        let candidates: Vec<RerankerCandidate> = (0..5)
            .map(|i| {
                let emb: Vec<f64> = (0..16).map(|_| xorshift_f64(&mut state)).collect();
                make_vec_candidate(&format!("doc{i}"), emb)
            })
            .collect();
        let results = r
            .rerank(candidates, None, Some(&q))
            .expect("test: rerank should succeed");
        for res in &results {
            assert!(res.final_score.is_finite());
        }
    }

    #[test]
    fn test_vector_only_dimension_mismatch_error() {
        let mut r = default_reranker();
        let q = vec![1.0, 2.0, 3.0];
        let cand = make_vec_candidate("bad", vec![1.0, 2.0]);
        let err = r
            .rerank(vec![cand], None, Some(&q))
            .expect_err("test: rerank should return error for dimension mismatch");
        assert!(matches!(err, RerankerError::IncompatibleDimensions { .. }));
    }

    // ── cross-modal fusion ────────────────────────────────────────────────────

    #[test]
    fn test_cross_modal_fusion_both_modalities_present() {
        let mut r = default_reranker();
        let q_text = "rust programming";
        let q_emb = vec![1.0_f64, 0.0];
        let candidates = vec![
            make_full_candidate("doc1", "rust programming language", vec![0.99, 0.14]),
            make_full_candidate("doc2", "python data science", vec![0.0, 1.0]),
        ];
        let results = r
            .rerank(candidates, Some(q_text), Some(&q_emb))
            .expect("test: rerank should succeed");
        assert_eq!(results[0].id, "doc1");
    }

    #[test]
    fn test_cross_modal_modality_scores_populated() {
        let mut r = default_reranker();
        let candidates = vec![make_full_candidate("doc1", "hello world", vec![1.0, 0.0])];
        let q_emb = vec![1.0, 0.0];
        let results = r
            .rerank(candidates, Some("hello"), Some(&q_emb))
            .expect("test: rerank should succeed");
        assert!(!results[0].modality_scores.is_empty());
        let has_text = results[0]
            .modality_scores
            .iter()
            .any(|ms| ms.modality == "text");
        let has_vec = results[0]
            .modality_scores
            .iter()
            .any(|ms| ms.modality == "vector");
        assert!(has_text);
        assert!(has_vec);
    }

    // ── FusionStrategy: LinearCombination ────────────────────────────────────

    #[test]
    fn test_linear_combination_weights() {
        let config = RerankerConfig {
            fusion_strategy: CmrFusionStrategy::LinearCombination(vec![
                ("text".to_string(), 0.9),
                ("vector".to_string(), 0.1),
            ]),
            normalize_scores: false,
            ..Default::default()
        };
        let mut r = CrossModalReranker::new(config);
        let candidates = vec![
            make_full_candidate("doc1", "rust is great", vec![0.0, 1.0]),
            make_full_candidate("doc2", "python is fine", vec![1.0, 0.0]),
        ];
        let q_emb = vec![1.0_f64, 0.0];
        // doc2 has better vector score, doc1 has better text score
        // heavy text weight → doc1 should win
        let results = r
            .rerank(candidates, Some("rust"), Some(&q_emb))
            .expect("test: rerank should succeed");
        assert_eq!(results[0].id, "doc1");
    }

    // ── FusionStrategy: ReciprocalRankFusion ─────────────────────────────────

    #[test]
    fn test_rrf_fusion_strategy() {
        let config = RerankerConfig {
            fusion_strategy: CmrFusionStrategy::ReciprocalRankFusion(60.0),
            normalize_scores: false,
            ..Default::default()
        };
        let mut r = CrossModalReranker::new(config);
        let candidates = vec![
            make_full_candidate("doc1", "rust rust rust", vec![0.9, 0.0]),
            make_full_candidate("doc2", "python", vec![0.1, 0.0]),
        ];
        let q_emb = vec![1.0_f64, 0.0];
        let results = r
            .rerank(candidates, Some("rust"), Some(&q_emb))
            .expect("test: rerank should succeed");
        assert!(!results.is_empty());
    }

    // ── FusionStrategy: Borda ─────────────────────────────────────────────────

    #[test]
    fn test_borda_fusion_strategy() {
        let config = RerankerConfig {
            fusion_strategy: CmrFusionStrategy::Borda,
            normalize_scores: false,
            ..Default::default()
        };
        let mut r = CrossModalReranker::new(config);
        let candidates = vec![
            make_full_candidate("doc1", "rust rust", vec![0.8, 0.2]),
            make_full_candidate("doc2", "python", vec![0.2, 0.8]),
        ];
        let q_emb = vec![1.0_f64, 0.0];
        let results = r
            .rerank(candidates, Some("rust"), Some(&q_emb))
            .expect("test: rerank should succeed");
        // doc1 ranks first in text; doc2 first in vector. Borda should give
        // each 2 points in one list and 1 in the other → equal Borda score.
        // Just verify no panic and scores are non-negative.
        for res in &results {
            assert!(res.final_score >= 0.0);
        }
    }

    #[test]
    fn test_borda_scores_non_negative() {
        let config = RerankerConfig {
            fusion_strategy: CmrFusionStrategy::Borda,
            normalize_scores: false,
            ..Default::default()
        };
        let mut r = CrossModalReranker::new(config);
        let candidates = (0..5)
            .map(|i| make_text_candidate(&format!("d{i}"), &"word ".repeat(i + 1)))
            .collect();
        let results = r
            .rerank(candidates, Some("word"), None)
            .expect("test: rerank should succeed");
        for res in &results {
            assert!(res.final_score >= 0.0);
        }
    }

    // ── FusionStrategy: MaxScore ──────────────────────────────────────────────

    #[test]
    fn test_max_score_fusion() {
        let config = RerankerConfig {
            fusion_strategy: CmrFusionStrategy::MaxScore,
            normalize_scores: false,
            ..Default::default()
        };
        let mut r = CrossModalReranker::new(config);
        let candidates = vec![
            make_full_candidate("doc1", "rust rust rust rust", vec![0.2, 0.0]),
            make_full_candidate("doc2", "python", vec![0.99, 0.0]),
        ];
        let q_emb = vec![1.0_f64, 0.0];
        let results = r
            .rerank(candidates, Some("rust"), Some(&q_emb))
            .expect("test: rerank should succeed");
        // max score: doc1 should win if BM25 is higher; doc2 if vector is higher
        assert!(!results.is_empty());
        for res in &results {
            assert!(res.final_score.is_finite());
        }
    }

    // ── FusionStrategy: LearnedWeights ───────────────────────────────────────

    #[test]
    fn test_learned_weights_fusion() {
        let config = RerankerConfig {
            fusion_strategy: CmrFusionStrategy::LearnedWeights(vec![2.0, 1.0]),
            normalize_scores: false,
            ..Default::default()
        };
        let mut r = CrossModalReranker::new(config);
        let candidates = vec![
            make_full_candidate("doc1", "rust programming", vec![0.5, 0.0]),
            make_full_candidate("doc2", "java programming", vec![0.8, 0.0]),
        ];
        let q_emb = vec![1.0_f64, 0.0];
        let results = r
            .rerank(candidates, Some("rust"), Some(&q_emb))
            .expect("test: rerank should succeed");
        assert!(!results.is_empty());
    }

    // ── normalization ─────────────────────────────────────────────────────────

    #[test]
    fn test_normalize_scores_in_range() {
        let config = RerankerConfig {
            normalize_scores: true,
            ..Default::default()
        };
        let mut r = CrossModalReranker::new(config);
        let mut state: u64 = 54321;
        let q: Vec<f64> = (0..4).map(|_| xorshift_f64(&mut state)).collect();
        let candidates: Vec<RerankerCandidate> = (0..8)
            .map(|i| {
                let emb: Vec<f64> = (0..4).map(|_| xorshift_f64(&mut state)).collect();
                make_full_candidate(&format!("d{i}"), "some text here", emb)
            })
            .collect();
        let results = r
            .rerank(candidates, Some("text"), Some(&q))
            .expect("test: rerank should succeed");
        for res in &results {
            assert!(
                (0.0..=1.0).contains(&res.final_score),
                "score={}",
                res.final_score
            );
        }
    }

    #[test]
    fn test_normalize_scores_disabled() {
        let config = RerankerConfig {
            normalize_scores: false,
            ..Default::default()
        };
        let mut r = CrossModalReranker::new(config);
        let candidates = vec![
            make_text_candidate("d1", "hello world hello"),
            make_text_candidate("d2", "foo bar"),
        ];
        let results = r
            .rerank(candidates, Some("hello"), None)
            .expect("test: rerank should succeed");
        // Scores may exceed 1
        for res in &results {
            assert!(res.final_score.is_finite());
        }
    }

    #[test]
    fn test_normalize_all_equal_scores() {
        let config = RerankerConfig {
            normalize_scores: true,
            ..Default::default()
        };
        let mut r = CrossModalReranker::new(config);
        let candidates = vec![
            make_vec_candidate("d1", vec![1.0, 0.0]),
            make_vec_candidate("d2", vec![1.0, 0.0]),
        ];
        let q = vec![1.0, 0.0];
        let results = r
            .rerank(candidates, None, Some(&q))
            .expect("test: rerank should succeed");
        for res in &results {
            assert!((0.0..=1.0).contains(&res.final_score));
        }
    }

    // ── top_k filtering ───────────────────────────────────────────────────────

    #[test]
    fn test_top_k_limits_results() {
        let config = RerankerConfig {
            top_k: 3,
            normalize_scores: false,
            ..Default::default()
        };
        let mut r = CrossModalReranker::new(config);
        let candidates = (0..10)
            .map(|i| make_text_candidate(&format!("d{i}"), &format!("word {i}")))
            .collect();
        let results = r
            .rerank(candidates, Some("word"), None)
            .expect("test: rerank should succeed");
        assert!(results.len() <= 3);
    }

    #[test]
    fn test_top_k_zero_returns_all() {
        let config = RerankerConfig {
            top_k: 0,
            normalize_scores: false,
            min_score_threshold: 0.0,
            ..Default::default()
        };
        let mut r = CrossModalReranker::new(config);
        let candidates = (0..5)
            .map(|i| make_text_candidate(&format!("d{i}"), &format!("word {i}")))
            .collect();
        let results = r
            .rerank(candidates, Some("word"), None)
            .expect("test: rerank should succeed");
        assert_eq!(results.len(), 5);
    }

    // ── min_score_threshold ───────────────────────────────────────────────────

    #[test]
    fn test_min_score_threshold_filters_low() {
        let config = RerankerConfig {
            normalize_scores: true,
            min_score_threshold: 0.5,
            top_k: 0,
            ..Default::default()
        };
        let mut r = CrossModalReranker::new(config);
        let candidates = vec![
            make_text_candidate("high", "rust rust rust rust"),
            make_text_candidate("low", "java"),
        ];
        let results = r
            .rerank(candidates, Some("rust"), None)
            .expect("test: rerank should succeed");
        for res in &results {
            assert!(
                res.final_score >= 0.5,
                "score below threshold: {}",
                res.final_score
            );
        }
    }

    #[test]
    fn test_min_score_threshold_zero_keeps_all() {
        let config = RerankerConfig {
            normalize_scores: false,
            min_score_threshold: 0.0,
            top_k: 0,
            ..Default::default()
        };
        let mut r = CrossModalReranker::new(config);
        let candidates = (0..4)
            .map(|i| make_text_candidate(&format!("d{i}"), &format!("text {i}")))
            .collect();
        let results = r
            .rerank(candidates, Some("text"), None)
            .expect("test: rerank should succeed");
        assert_eq!(results.len(), 4);
    }

    // ── error cases ───────────────────────────────────────────────────────────

    #[test]
    fn test_error_no_candidates() {
        let mut r = default_reranker();
        let err = r
            .rerank(vec![], Some("query"), None)
            .expect_err("test: rerank should return error for empty candidates");
        assert_eq!(err, RerankerError::NoCandidates);
    }

    #[test]
    fn test_error_incompatible_dimensions() {
        let mut r = default_reranker();
        let cand = make_vec_candidate("d1", vec![1.0, 2.0]);
        let q = vec![1.0, 2.0, 3.0];
        let err = r
            .rerank(vec![cand], None, Some(&q))
            .expect_err("test: rerank should return error for incompatible dimensions");
        assert!(matches!(err, RerankerError::IncompatibleDimensions { .. }));
    }

    #[test]
    fn test_error_invalid_weight_negative() {
        let config = RerankerConfig {
            text_weight: -1.0,
            ..Default::default()
        };
        let mut r = CrossModalReranker::new(config);
        let cand = make_text_candidate("d1", "hello");
        let err = r
            .rerank(vec![cand], Some("hello"), None)
            .expect_err("test: rerank should return error for negative text_weight");
        assert!(matches!(err, RerankerError::InvalidWeight(_)));
    }

    #[test]
    fn test_error_invalid_linear_weight() {
        let config = RerankerConfig {
            fusion_strategy: CmrFusionStrategy::LinearCombination(vec![("text".to_string(), -0.1)]),
            ..Default::default()
        };
        let mut r = CrossModalReranker::new(config);
        let cand = make_text_candidate("d1", "hello");
        let err = r
            .rerank(vec![cand], Some("hello"), None)
            .expect_err("test: rerank should return error for negative linear combination weight");
        assert!(matches!(err, RerankerError::InvalidWeight(_)));
    }

    #[test]
    fn test_error_invalid_rrf_k() {
        let config = RerankerConfig {
            fusion_strategy: CmrFusionStrategy::ReciprocalRankFusion(-1.0),
            ..Default::default()
        };
        let mut r = CrossModalReranker::new(config);
        let cand = make_text_candidate("d1", "hello");
        let err = r
            .rerank(vec![cand], Some("hello"), None)
            .expect_err("test: rerank should return error for invalid RRF k value");
        assert!(matches!(err, RerankerError::ConfigurationError(_)));
    }

    #[test]
    fn test_error_display() {
        let e = RerankerError::NoCandidates;
        assert!(!format!("{e}").is_empty());
        let e2 = RerankerError::IncompatibleDimensions {
            expected: 4,
            got: 3,
        };
        assert!(format!("{e2}").contains("4"));
    }

    // ── stats ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_stats_fusion_calls_incremented() {
        let mut r = default_reranker();
        assert_eq!(r.stats().fusion_calls, 0);
        let _ = r.rerank(vec![make_text_candidate("a", "hi")], Some("hi"), None);
        assert_eq!(r.stats().fusion_calls, 1);
        let _ = r.rerank(vec![make_text_candidate("b", "bye")], Some("bye"), None);
        assert_eq!(r.stats().fusion_calls, 2);
    }

    #[test]
    fn test_stats_candidates_reranked_accumulates() {
        let mut r = default_reranker();
        let c1 = vec![make_text_candidate("a", "a"), make_text_candidate("b", "b")];
        let c2 = vec![make_text_candidate("c", "c")];
        let _ = r.rerank(c1, Some("q"), None);
        let _ = r.rerank(c2, Some("q"), None);
        assert_eq!(r.stats().candidates_reranked, 3);
    }

    #[test]
    fn test_stats_modalities_tracked() {
        let mut r = default_reranker();
        let cands = vec![make_full_candidate("d1", "hello", vec![1.0, 0.0])];
        let q_emb = vec![1.0, 0.0];
        let _ = r.rerank(cands, Some("hello"), Some(&q_emb));
        let stats = r.stats();
        assert!(stats.modalities_used.contains(&"text".to_string()));
        assert!(stats.modalities_used.contains(&"vector".to_string()));
    }

    // ── update_config ─────────────────────────────────────────────────────────

    #[test]
    fn test_update_config() {
        let mut r = default_reranker();
        let new_cfg = RerankerConfig {
            top_k: 5,
            ..Default::default()
        };
        r.update_config(new_cfg.clone());
        assert_eq!(r.config.top_k, 5);
    }

    // ── ModalityScore ─────────────────────────────────────────────────────────

    #[test]
    fn test_modality_score_new() {
        let ms = ModalityScore::new("text", 0.8, 0.4);
        assert_eq!(ms.modality, "text");
        assert!((ms.raw_score - 0.8).abs() < 1e-10);
        assert!((ms.normalized_score - 0.8).abs() < 1e-10);
        assert!((ms.weight - 0.4).abs() < 1e-10);
    }

    // ── single candidate ──────────────────────────────────────────────────────

    #[test]
    fn test_single_candidate_gets_rank_one() {
        let mut r = default_reranker();
        let cand = make_text_candidate("solo", "only document");
        let results = r
            .rerank(vec![cand], Some("document"), None)
            .expect("test: rerank should succeed");
        assert_eq!(results[0].rank, 1);
    }

    #[test]
    fn test_single_candidate_score_normalised_to_one() {
        let config = RerankerConfig {
            normalize_scores: true,
            ..Default::default()
        };
        let mut r = CrossModalReranker::new(config);
        let cand = make_text_candidate("solo", "only document");
        let results = r
            .rerank(vec![cand], Some("document"), None)
            .expect("test: rerank should succeed");
        assert!((results[0].final_score - 1.0).abs() < 1e-10);
    }

    // ── misc edge cases ───────────────────────────────────────────────────────

    #[test]
    fn test_candidates_with_no_matching_modality_get_zero_score() {
        // No query text or embedding → all candidates should have final_score 0
        let config = RerankerConfig {
            normalize_scores: false,
            ..Default::default()
        };
        let mut r = CrossModalReranker::new(config);
        let candidates = vec![
            make_text_candidate("d1", "hello"),
            make_text_candidate("d2", "world"),
        ];
        // Neither query_text nor query_embedding provided
        let results = r
            .rerank(candidates, None, None)
            .expect("test: rerank should succeed");
        for res in &results {
            // All modality_scores will be empty → linear combo sums to 0
            assert_eq!(res.final_score, 0.0);
        }
    }

    #[test]
    fn test_reranker_candidate_new() {
        let c = RerankerCandidate::new("id", Some("text"), Some(vec![1.0]));
        assert_eq!(c.id, "id");
        assert_eq!(c.text_snippet.as_deref(), Some("text"));
        assert_eq!(c.embedding, Some(vec![1.0]));
        assert!(c.modality_scores.is_empty());
        assert_eq!(c.rank, 0);
    }

    #[test]
    fn test_large_candidate_set_no_panic() {
        let mut state: u64 = 314159;
        let mut r = default_reranker();
        let q: Vec<f64> = (0..32).map(|_| xorshift_f64(&mut state)).collect();
        let candidates: Vec<RerankerCandidate> = (0..200)
            .map(|i| {
                let emb: Vec<f64> = (0..32).map(|_| xorshift_f64(&mut state)).collect();
                make_full_candidate(&format!("d{i}"), "some query terms here", emb)
            })
            .collect();
        let results = r
            .rerank(candidates, Some("query terms"), Some(&q))
            .expect("test: rerank should succeed");
        assert!(results.len() <= 100); // default top_k = 100
    }
}
