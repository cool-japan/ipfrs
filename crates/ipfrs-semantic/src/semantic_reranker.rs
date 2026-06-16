//! Semantic Reranker — cross-encoder-style query-document pair scoring.
//!
//! This module provides a production-grade reranking engine that combines
//! multiple feature signals (embedding similarity, keyword overlap, length
//! penalty, position prior, title boost) using configurable weighted fusion.
//! After initial retrieval (e.g., via HNSW or DiskANN), `SemanticReranker`
//! rescores each candidate against the query to yield a refined ranking with
//! substantially higher precision.
//!
//! # Pipeline
//!
//! 1. Retrieve an initial candidate set with rough scores.
//! 2. Call [`SemanticReranker::rerank`] → sorted [`RerankResult`] list.
//! 3. Optionally slice with [`SemanticReranker::top_k`].
//! 4. Evaluate ranking quality via [`SemanticReranker::precision_at_k`] /
//!    [`SemanticReranker::ndcg_at_k`].
//!
//! # Example
//!
//! ```rust
//! use ipfrs_semantic::semantic_reranker::{
//!     SemanticReranker, RerankConfig, RerankCandidate, RerankQuery, RerankFeature,
//! };
//! use std::collections::HashMap;
//!
//! let config = RerankConfig::default();
//! let mut reranker = SemanticReranker::new(config);
//!
//! let query = RerankQuery {
//!     text: "rust programming language".to_string(),
//!     embedding: Some(vec![0.1, 0.9, 0.0]),
//!     context: vec![],
//! };
//!
//! let candidates = vec![
//!     RerankCandidate {
//!         id: "doc1".to_string(),
//!         initial_score: 0.9,
//!         content: "Rust is a systems programming language focused on safety and performance."
//!             .to_string(),
//!         embedding: Some(vec![0.15, 0.85, 0.05]),
//!         metadata: HashMap::new(),
//!     },
//! ];
//!
//! let results = reranker.rerank(&query, &candidates);
//! assert!(!results.is_empty());
//! ```

use std::collections::{HashMap, HashSet};

// ─────────────────────────────────────────────────────────────────────────────
// Core types
// ─────────────────────────────────────────────────────────────────────────────

/// A single candidate document to be reranked.
#[derive(Debug, Clone)]
pub struct RerankCandidate {
    /// Unique document identifier.
    pub id: String,
    /// Score from the initial retrieval stage.
    pub initial_score: f64,
    /// Raw text content used for keyword-based features.
    pub content: String,
    /// Optional dense embedding vector.
    pub embedding: Option<Vec<f64>>,
    /// Arbitrary key-value metadata (e.g., `"title"` → document title).
    pub metadata: HashMap<String, String>,
}

/// Query object supplied to the reranker.
#[derive(Debug, Clone)]
pub struct RerankQuery {
    /// Query text used for keyword-based features.
    pub text: String,
    /// Optional dense query embedding.
    pub embedding: Option<Vec<f64>>,
    /// Additional context sentences that may supplement scoring.
    pub context: Vec<String>,
}

/// Reranking feature variants.
#[derive(Debug, Clone)]
pub enum RerankFeature {
    /// Cosine similarity between `query.embedding` and `candidate.embedding`.
    EmbeddingScore,
    /// Jaccard similarity between tokenised query and content term sets.
    KeywordOverlap,
    /// Penalty for very short (<50 chars) or very long (>5000 chars) content.
    /// `score = 1.0 - |optimal - len| / optimal`, optimal = 500.
    LengthPenalty,
    /// Multiply score by `boost` when the title metadata contains a query term.
    TitleBoost {
        /// Multiplicative factor applied when the title matches (e.g., 1.5).
        boost: f64,
    },
    /// Weighted initial score decay based on candidate rank in initial list.
    /// `score = initial_score * (1 - decay * rank_fraction)`
    PositionPrior {
        /// Decay factor controlling how quickly position degrades score.
        decay: f64,
    },
}

impl RerankFeature {
    /// Human-readable feature name used as key in [`RerankResult::feature_scores`].
    pub fn name(&self) -> &'static str {
        match self {
            RerankFeature::EmbeddingScore => "embedding_score",
            RerankFeature::KeywordOverlap => "keyword_overlap",
            RerankFeature::LengthPenalty => "length_penalty",
            RerankFeature::TitleBoost { .. } => "title_boost",
            RerankFeature::PositionPrior { .. } => "position_prior",
        }
    }
}

/// Configuration for [`SemanticReranker`].
#[derive(Debug, Clone)]
pub struct RerankConfig {
    /// Weighted feature list.  Weights are normalised internally.
    pub features: Vec<(RerankFeature, f64)>,
    /// Whether to min-max normalise final scores across the candidate set.
    pub normalize_scores: bool,
    /// Candidates scoring below this threshold (after normalisation if enabled)
    /// are dropped from the output.
    pub min_rerank_score: f64,
}

impl Default for RerankConfig {
    fn default() -> Self {
        Self {
            features: vec![
                (RerankFeature::EmbeddingScore, 0.5),
                (RerankFeature::KeywordOverlap, 0.3),
                (RerankFeature::LengthPenalty, 0.1),
                (RerankFeature::PositionPrior { decay: 0.1 }, 0.1),
            ],
            normalize_scores: true,
            min_rerank_score: 0.0,
        }
    }
}

/// Result for a single reranked candidate.
#[derive(Debug, Clone)]
pub struct RerankResult {
    /// Candidate identifier.
    pub candidate_id: String,
    /// Final combined rerank score.
    pub rerank_score: f64,
    /// Score from the initial retrieval stage.
    pub initial_score: f64,
    /// Per-feature scores (feature name → score).
    pub feature_scores: HashMap<String, f64>,
    /// 1-based rank in the final sorted list.
    pub rank: usize,
}

/// Aggregate statistics produced by [`SemanticReranker`].
#[derive(Debug, Clone)]
pub struct RerankStats {
    /// Total number of times [`SemanticReranker::rerank`] has been called.
    pub total_rerankings: u64,
    /// Average number of candidates processed per reranking call.
    pub avg_candidates_per_reranking: f64,
    /// Mean difference (rerank_score − initial_score) across all results.
    pub avg_score_improvement: f64,
}

// ─────────────────────────────────────────────────────────────────────────────
// SemanticReranker
// ─────────────────────────────────────────────────────────────────────────────

/// Internal per-call tracking record.
#[derive(Debug, Default)]
struct CallRecord {
    candidate_count: usize,
    total_improvement: f64,
    result_count: usize,
}

/// Cross-encoder-style reranking engine.
pub struct SemanticReranker {
    /// Reranking configuration.
    pub config: RerankConfig,
    /// Total number of `rerank` invocations.
    pub total_rerankings: u64,
    /// Accumulated per-call statistics for aggregate reporting.
    call_records: Vec<CallRecord>,
}

impl SemanticReranker {
    /// Create a new reranker with the supplied configuration.
    pub fn new(config: RerankConfig) -> Self {
        Self {
            config,
            total_rerankings: 0,
            call_records: Vec::new(),
        }
    }

    /// Rerank `candidates` with respect to `query`.
    ///
    /// Steps:
    /// 1. Compute weighted feature scores for every candidate.
    /// 2. Optionally normalise scores across the candidate set.
    /// 3. Filter by `min_rerank_score`.
    /// 4. Sort descending and assign 1-based ranks.
    pub fn rerank(
        &mut self,
        query: &RerankQuery,
        candidates: &[RerankCandidate],
    ) -> Vec<RerankResult> {
        let total = candidates.len();
        if total == 0 {
            self.total_rerankings += 1;
            self.call_records.push(CallRecord::default());
            return Vec::new();
        }

        // Normalised feature weights (guard against all-zero sum).
        let weight_sum: f64 = self.config.features.iter().map(|(_, w)| w.abs()).sum();
        let weight_sum = if weight_sum < f64::EPSILON {
            1.0
        } else {
            weight_sum
        };

        // Score every candidate.
        let mut raw: Vec<(RerankResult, f64)> = candidates
            .iter()
            .enumerate()
            .map(|(rank_idx, candidate)| {
                let feature_scores = self.score_candidate(query, candidate, rank_idx, total);
                let combined: f64 = self
                    .config
                    .features
                    .iter()
                    .map(|(feat, weight)| {
                        let score = feature_scores.get(feat.name()).copied().unwrap_or(0.0);
                        score * weight / weight_sum
                    })
                    .sum();
                let result = RerankResult {
                    candidate_id: candidate.id.clone(),
                    rerank_score: combined,
                    initial_score: candidate.initial_score,
                    feature_scores,
                    rank: 0, // assigned later
                };
                (result, combined)
            })
            .collect();

        // Optional min-max normalisation.
        if self.config.normalize_scores && raw.len() > 1 {
            let min_score = raw.iter().map(|(_, s)| *s).fold(f64::INFINITY, f64::min);
            let max_score = raw
                .iter()
                .map(|(_, s)| *s)
                .fold(f64::NEG_INFINITY, f64::max);
            let range = max_score - min_score;
            if range > f64::EPSILON {
                for (result, score) in raw.iter_mut() {
                    let normalised = (*score - min_score) / range;
                    *score = normalised;
                    result.rerank_score = normalised;
                }
            }
        }

        // Filter by threshold.
        let threshold = self.config.min_rerank_score;
        let mut filtered: Vec<RerankResult> = raw
            .into_iter()
            .filter(|(_, s)| *s >= threshold)
            .map(|(mut r, s)| {
                r.rerank_score = s;
                r
            })
            .collect();

        // Sort descending.
        filtered.sort_by(|a, b| {
            b.rerank_score
                .partial_cmp(&a.rerank_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Assign 1-based ranks.
        for (i, result) in filtered.iter_mut().enumerate() {
            result.rank = i + 1;
        }

        // Track statistics.
        let record = CallRecord {
            candidate_count: total,
            total_improvement: filtered
                .iter()
                .map(|r| r.rerank_score - r.initial_score)
                .sum(),
            result_count: filtered.len(),
        };
        self.call_records.push(record);
        self.total_rerankings += 1;

        filtered
    }

    /// Compute per-feature scores for a single candidate.
    ///
    /// Returns a map from feature name to score in `[0.0, 1.0]` (approximately).
    pub fn score_candidate(
        &self,
        query: &RerankQuery,
        candidate: &RerankCandidate,
        rank: usize,
        total: usize,
    ) -> HashMap<String, f64> {
        let mut scores = HashMap::new();
        for (feature, _) in &self.config.features {
            let score = self.compute_feature(feature, query, candidate, rank, total);
            scores.insert(feature.name().to_string(), score);
        }
        scores
    }

    /// Compute the score for a single feature.
    pub fn compute_feature(
        &self,
        feature: &RerankFeature,
        query: &RerankQuery,
        candidate: &RerankCandidate,
        rank: usize,
        total: usize,
    ) -> f64 {
        match feature {
            RerankFeature::EmbeddingScore => match (&query.embedding, &candidate.embedding) {
                (Some(qe), Some(ce)) => Self::cosine_similarity(qe, ce),
                _ => 0.0,
            },

            RerankFeature::KeywordOverlap => {
                let query_terms = Self::tokenize(&query.text);
                let content_terms = Self::tokenize(&candidate.content);
                Self::jaccard_similarity(&query_terms, &content_terms)
            }

            RerankFeature::LengthPenalty => {
                const OPTIMAL: f64 = 500.0;
                let len = candidate.content.len();
                let deviation = (OPTIMAL - len as f64).abs() / OPTIMAL;
                (1.0 - deviation).max(0.0)
            }

            RerankFeature::TitleBoost { boost } => {
                let title = candidate
                    .metadata
                    .get("title")
                    .map(|s| s.to_lowercase())
                    .unwrap_or_default();
                if title.is_empty() {
                    1.0
                } else {
                    let query_terms = Self::tokenize(&query.text);
                    let has_match = query_terms.iter().any(|term| title.contains(term.as_str()));
                    if has_match {
                        *boost
                    } else {
                        1.0
                    }
                }
            }

            RerankFeature::PositionPrior { decay } => {
                if total == 0 {
                    return candidate.initial_score;
                }
                let rank_fraction = rank as f64 / total as f64;
                candidate.initial_score * (1.0 - decay * rank_fraction)
            }
        }
    }

    /// Cosine similarity between two vectors.  Returns 0.0 on zero vectors or
    /// dimension mismatch.
    pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }
        let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm_a < f64::EPSILON || norm_b < f64::EPSILON {
            return 0.0;
        }
        (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
    }

    /// Jaccard similarity: |A ∩ B| / |A ∪ B|.
    /// Returns 0.0 when both sets are empty.
    pub fn jaccard_similarity(a: &[String], b: &[String]) -> f64 {
        let set_a: HashSet<&str> = a.iter().map(|s| s.as_str()).collect();
        let set_b: HashSet<&str> = b.iter().map(|s| s.as_str()).collect();
        let intersection = set_a.intersection(&set_b).count();
        let union = set_a.union(&set_b).count();
        if union == 0 {
            0.0
        } else {
            intersection as f64 / union as f64
        }
    }

    /// Tokenise text: lowercase, keep only alphanumeric characters, split on
    /// whitespace/punctuation, deduplicate, and return sorted.
    pub fn tokenize(text: &str) -> Vec<String> {
        let mut terms: HashSet<String> = HashSet::new();
        for word in text.split(|c: char| !c.is_alphanumeric()) {
            let token: String = word
                .chars()
                .filter(|c| c.is_alphanumeric())
                .map(|c| c.to_lowercase().next().unwrap_or(c))
                .collect();
            if !token.is_empty() {
                terms.insert(token);
            }
        }
        let mut sorted: Vec<String> = terms.into_iter().collect();
        sorted.sort_unstable();
        sorted
    }

    /// Return the top-`k` results by `rerank_score`.
    pub fn top_k<'a>(&self, results: &'a [RerankResult], k: usize) -> Vec<&'a RerankResult> {
        // Results are assumed to already be sorted descending from `rerank`.
        results.iter().take(k).collect()
    }

    /// Fraction of the top-`k` results whose `candidate_id` appears in
    /// `relevant_ids`.  Returns 0.0 when `k == 0`.
    pub fn precision_at_k(
        &self,
        results: &[RerankResult],
        k: usize,
        relevant_ids: &[String],
    ) -> f64 {
        if k == 0 {
            return 0.0;
        }
        let relevant_set: HashSet<&str> = relevant_ids.iter().map(|s| s.as_str()).collect();
        let top = results.iter().take(k);
        let hits = top
            .filter(|r| relevant_set.contains(r.candidate_id.as_str()))
            .count();
        hits as f64 / k as f64
    }

    /// Normalised Discounted Cumulative Gain at depth `k` with binary relevance.
    /// Returns 0.0 when `k == 0` or IDCG == 0.
    pub fn ndcg_at_k(&self, results: &[RerankResult], k: usize, relevant_ids: &[String]) -> f64 {
        if k == 0 {
            return 0.0;
        }
        let relevant_set: HashSet<&str> = relevant_ids.iter().map(|s| s.as_str()).collect();

        // Actual DCG.
        let dcg: f64 = results
            .iter()
            .take(k)
            .enumerate()
            .filter(|(_, r)| relevant_set.contains(r.candidate_id.as_str()))
            .map(|(i, _)| 1.0 / (i as f64 + 2.0).log2()) // rel=1 → rel / log2(pos+1)
            .sum();

        // Ideal DCG: place all relevant docs first.
        let num_relevant = relevant_set.len().min(k);
        let idcg: f64 = (0..num_relevant)
            .map(|i| 1.0 / (i as f64 + 2.0).log2())
            .sum();

        if idcg < f64::EPSILON {
            0.0
        } else {
            dcg / idcg
        }
    }

    /// Return accumulated statistics.
    pub fn stats(&self) -> RerankStats {
        let total = self.total_rerankings;
        if total == 0 {
            return RerankStats {
                total_rerankings: 0,
                avg_candidates_per_reranking: 0.0,
                avg_score_improvement: 0.0,
            };
        }
        let total_candidates: usize = self.call_records.iter().map(|r| r.candidate_count).sum();
        let total_improvement: f64 = self.call_records.iter().map(|r| r.total_improvement).sum();
        let total_results: usize = self.call_records.iter().map(|r| r.result_count).sum();

        RerankStats {
            total_rerankings: total,
            avg_candidates_per_reranking: total_candidates as f64 / total as f64,
            avg_score_improvement: if total_results == 0 {
                0.0
            } else {
                total_improvement / total_results as f64
            },
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::semantic_reranker::{
        RerankCandidate, RerankConfig, RerankFeature, RerankQuery, SemanticReranker,
    };

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn make_candidate(id: &str, score: f64, content: &str) -> RerankCandidate {
        RerankCandidate {
            id: id.to_string(),
            initial_score: score,
            content: content.to_string(),
            embedding: None,
            metadata: HashMap::new(),
        }
    }

    fn make_candidate_with_embedding(
        id: &str,
        score: f64,
        content: &str,
        emb: Vec<f64>,
    ) -> RerankCandidate {
        RerankCandidate {
            id: id.to_string(),
            initial_score: score,
            content: content.to_string(),
            embedding: Some(emb),
            metadata: HashMap::new(),
        }
    }

    fn make_query(text: &str) -> RerankQuery {
        RerankQuery {
            text: text.to_string(),
            embedding: None,
            context: vec![],
        }
    }

    fn make_query_with_embedding(text: &str, emb: Vec<f64>) -> RerankQuery {
        RerankQuery {
            text: text.to_string(),
            embedding: Some(emb),
            context: vec![],
        }
    }

    // ── cosine_similarity ─────────────────────────────────────────────────────

    #[test]
    fn test_cosine_identical_vectors() {
        let v = vec![1.0, 2.0, 3.0];
        let sim = SemanticReranker::cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_cosine_orthogonal_vectors() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = SemanticReranker::cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-9);
    }

    #[test]
    fn test_cosine_opposite_vectors() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        let sim = SemanticReranker::cosine_similarity(&a, &b);
        assert!((sim - (-1.0)).abs() < 1e-9);
    }

    #[test]
    fn test_cosine_zero_vector_returns_zero() {
        let a = vec![0.0, 0.0];
        let b = vec![1.0, 2.0];
        let sim = SemanticReranker::cosine_similarity(&a, &b);
        assert_eq!(sim, 0.0);
    }

    #[test]
    fn test_cosine_dimension_mismatch_returns_zero() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0];
        let sim = SemanticReranker::cosine_similarity(&a, &b);
        assert_eq!(sim, 0.0);
    }

    #[test]
    fn test_cosine_empty_vectors_returns_zero() {
        let sim = SemanticReranker::cosine_similarity(&[], &[]);
        assert_eq!(sim, 0.0);
    }

    #[test]
    fn test_cosine_near_parallel() {
        let a = vec![1.0, 0.001];
        let b = vec![1.0, 0.001];
        let sim = SemanticReranker::cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    // ── jaccard_similarity ────────────────────────────────────────────────────

    #[test]
    fn test_jaccard_identical_sets() {
        let terms = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let sim = SemanticReranker::jaccard_similarity(&terms, &terms);
        assert!((sim - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_jaccard_disjoint_sets() {
        let a = vec!["a".to_string()];
        let b = vec!["b".to_string()];
        let sim = SemanticReranker::jaccard_similarity(&a, &b);
        assert_eq!(sim, 0.0);
    }

    #[test]
    fn test_jaccard_partial_overlap() {
        let a = vec!["a".to_string(), "b".to_string()];
        let b = vec!["b".to_string(), "c".to_string()];
        let sim = SemanticReranker::jaccard_similarity(&a, &b);
        // |{b}| / |{a,b,c}| = 1/3
        assert!((sim - 1.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_jaccard_empty_sets() {
        let sim = SemanticReranker::jaccard_similarity(&[], &[]);
        assert_eq!(sim, 0.0);
    }

    #[test]
    fn test_jaccard_one_empty() {
        let a = vec!["rust".to_string()];
        let sim = SemanticReranker::jaccard_similarity(&a, &[]);
        assert_eq!(sim, 0.0);
    }

    // ── tokenize ──────────────────────────────────────────────────────────────

    #[test]
    fn test_tokenize_basic() {
        let tokens = SemanticReranker::tokenize("Hello, World!");
        assert!(tokens.contains(&"hello".to_string()));
        assert!(tokens.contains(&"world".to_string()));
    }

    #[test]
    fn test_tokenize_deduplicates() {
        let tokens = SemanticReranker::tokenize("rust rust RUST");
        assert_eq!(tokens, vec!["rust".to_string()]);
    }

    #[test]
    fn test_tokenize_sorted() {
        let tokens = SemanticReranker::tokenize("zebra apple mango");
        assert_eq!(tokens, vec!["apple", "mango", "zebra"]);
    }

    #[test]
    fn test_tokenize_strips_punctuation() {
        let tokens = SemanticReranker::tokenize("hello-world foo.bar");
        assert!(
            tokens.contains(&"hello".to_string()) || tokens.contains(&"helloworld".to_string())
        );
        // The key assertion: no punctuation characters in any token.
        for t in &tokens {
            assert!(
                t.chars().all(|c| c.is_alphanumeric()),
                "token '{t}' contains non-alphanumeric"
            );
        }
    }

    #[test]
    fn test_tokenize_empty_string() {
        let tokens = SemanticReranker::tokenize("");
        assert!(tokens.is_empty());
    }

    // ── rerank – empty candidates ─────────────────────────────────────────────

    #[test]
    fn test_rerank_empty_candidates() {
        let mut reranker = SemanticReranker::new(RerankConfig::default());
        let query = make_query("test");
        let results = reranker.rerank(&query, &[]);
        assert!(results.is_empty());
        assert_eq!(reranker.total_rerankings, 1);
    }

    // ── rerank – rank assignment ──────────────────────────────────────────────

    #[test]
    fn test_rerank_ranks_are_1_based_and_sequential() {
        let mut reranker = SemanticReranker::new(RerankConfig {
            normalize_scores: false,
            min_rerank_score: f64::NEG_INFINITY,
            ..Default::default()
        });
        let query = make_query("rust language");
        let candidates = vec![
            make_candidate("d1", 0.8, "rust systems language"),
            make_candidate("d2", 0.6, "python scripting"),
            make_candidate("d3", 0.7, "rust memory safety"),
        ];
        let results = reranker.rerank(&query, &candidates);
        let ranks: Vec<usize> = results.iter().map(|r| r.rank).collect();
        assert_eq!(ranks, vec![1, 2, 3]);
    }

    #[test]
    fn test_rerank_sorted_descending() {
        let mut reranker = SemanticReranker::new(RerankConfig::default());
        let query = make_query("rust programming");
        let candidates = vec![
            make_candidate("d1", 0.5, "unrelated topic about cooking"),
            make_candidate("d2", 0.9, "rust programming language systems"),
        ];
        let results = reranker.rerank(&query, &candidates);
        // First result should have highest score.
        assert!(results[0].rerank_score >= results[results.len() - 1].rerank_score);
    }

    #[test]
    fn test_rerank_preserves_initial_score() {
        let mut reranker = SemanticReranker::new(RerankConfig::default());
        let query = make_query("test query");
        let candidates = vec![make_candidate("d1", 0.75, "some content here")];
        let results = reranker.rerank(&query, &candidates);
        assert!(!results.is_empty());
        assert!((results[0].initial_score - 0.75).abs() < 1e-9);
    }

    #[test]
    fn test_rerank_feature_scores_populated() {
        let mut reranker = SemanticReranker::new(RerankConfig::default());
        let query = make_query("rust programming");
        let candidates = vec![make_candidate("d1", 0.5, "rust programming language")];
        let results = reranker.rerank(&query, &candidates);
        assert!(!results.is_empty());
        // Default features produce 4 named keys.
        assert!(results[0].feature_scores.contains_key("keyword_overlap"));
        assert!(results[0].feature_scores.contains_key("length_penalty"));
        assert!(results[0].feature_scores.contains_key("position_prior"));
    }

    // ── min_rerank_score filter ───────────────────────────────────────────────

    #[test]
    fn test_rerank_min_score_filter() {
        let config = RerankConfig {
            features: vec![(RerankFeature::KeywordOverlap, 1.0)],
            normalize_scores: false,
            min_rerank_score: 0.5,
        };
        let mut reranker = SemanticReranker::new(config);
        let query = make_query("rust");
        // d1: overlap with "rust" → 1.0; d2: no overlap → 0.0
        let candidates = vec![
            make_candidate("d1", 0.9, "rust systems programming"),
            make_candidate("d2", 0.8, "python machine learning"),
        ];
        let results = reranker.rerank(&query, &candidates);
        // Only d1 should pass the threshold.
        assert!(results.iter().all(|r| r.rerank_score >= 0.5));
    }

    // ── EmbeddingScore feature ────────────────────────────────────────────────

    #[test]
    fn test_embedding_feature_present_both() {
        let config = RerankConfig {
            features: vec![(RerankFeature::EmbeddingScore, 1.0)],
            normalize_scores: false,
            min_rerank_score: f64::NEG_INFINITY,
        };
        let mut reranker = SemanticReranker::new(config);
        let query = make_query_with_embedding("query", vec![1.0, 0.0]);
        let candidates = vec![
            make_candidate_with_embedding("d1", 0.5, "doc", vec![1.0, 0.0]),
            make_candidate_with_embedding("d2", 0.5, "doc", vec![0.0, 1.0]),
        ];
        let results = reranker.rerank(&query, &candidates);
        // d1 is parallel → cosine=1.0, d2 is orthogonal → cosine=0.0
        assert_eq!(results[0].candidate_id, "d1");
        assert!(results[0].rerank_score > results[1].rerank_score);
    }

    #[test]
    fn test_embedding_feature_missing_embedding_returns_zero() {
        let config = RerankConfig {
            features: vec![(RerankFeature::EmbeddingScore, 1.0)],
            normalize_scores: false,
            min_rerank_score: f64::NEG_INFINITY,
        };
        let mut reranker = SemanticReranker::new(config);
        let query = make_query("no embedding");
        let candidates = vec![make_candidate("d1", 0.5, "content")];
        let results = reranker.rerank(&query, &candidates);
        // No embedding → score 0.
        let score = *results[0]
            .feature_scores
            .get("embedding_score")
            .unwrap_or(&-1.0);
        assert_eq!(score, 0.0);
    }

    // ── LengthPenalty feature ─────────────────────────────────────────────────

    #[test]
    fn test_length_penalty_optimal_length() {
        let config = RerankConfig {
            features: vec![(RerankFeature::LengthPenalty, 1.0)],
            normalize_scores: false,
            min_rerank_score: f64::NEG_INFINITY,
        };
        let mut reranker = SemanticReranker::new(config);
        let query = make_query("anything");
        // Build a string of exactly 500 chars.
        let content_500 = "x".repeat(500);
        let candidates = vec![make_candidate("d1", 0.5, &content_500)];
        let results = reranker.rerank(&query, &candidates);
        let score = *results[0]
            .feature_scores
            .get("length_penalty")
            .unwrap_or(&-1.0);
        assert!((score - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_length_penalty_very_short_content() {
        let config = RerankConfig {
            features: vec![(RerankFeature::LengthPenalty, 1.0)],
            normalize_scores: false,
            min_rerank_score: f64::NEG_INFINITY,
        };
        let mut reranker = SemanticReranker::new(config);
        let query = make_query("anything");
        // Very short content (10 chars) → large deviation.
        let candidates = vec![make_candidate("d1", 0.5, "short txt.")];
        let results = reranker.rerank(&query, &candidates);
        let score = *results[0]
            .feature_scores
            .get("length_penalty")
            .unwrap_or(&-1.0);
        assert!(score < 1.0);
        assert!(score >= 0.0);
    }

    // ── TitleBoost feature ────────────────────────────────────────────────────

    #[test]
    fn test_title_boost_match() {
        let config = RerankConfig {
            features: vec![(RerankFeature::TitleBoost { boost: 2.0 }, 1.0)],
            normalize_scores: false,
            min_rerank_score: f64::NEG_INFINITY,
        };
        let mut reranker = SemanticReranker::new(config);
        let query = make_query("rust programming");
        let mut meta = HashMap::new();
        meta.insert(
            "title".to_string(),
            "Introduction to Rust Programming".to_string(),
        );
        let candidate = RerankCandidate {
            id: "d1".to_string(),
            initial_score: 0.5,
            content: "content".to_string(),
            embedding: None,
            metadata: meta,
        };
        let results = reranker.rerank(&query, &[candidate]);
        let score = *results[0]
            .feature_scores
            .get("title_boost")
            .unwrap_or(&-1.0);
        assert!((score - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_title_boost_no_match() {
        let config = RerankConfig {
            features: vec![(RerankFeature::TitleBoost { boost: 2.0 }, 1.0)],
            normalize_scores: false,
            min_rerank_score: f64::NEG_INFINITY,
        };
        let mut reranker = SemanticReranker::new(config);
        let query = make_query("python");
        let mut meta = HashMap::new();
        meta.insert("title".to_string(), "Introduction to Rust".to_string());
        let candidate = RerankCandidate {
            id: "d1".to_string(),
            initial_score: 0.5,
            content: "content".to_string(),
            embedding: None,
            metadata: meta,
        };
        let results = reranker.rerank(&query, &[candidate]);
        let score = *results[0]
            .feature_scores
            .get("title_boost")
            .unwrap_or(&-1.0);
        assert!((score - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_title_boost_missing_title_returns_one() {
        let config = RerankConfig {
            features: vec![(RerankFeature::TitleBoost { boost: 3.0 }, 1.0)],
            normalize_scores: false,
            min_rerank_score: f64::NEG_INFINITY,
        };
        let mut reranker = SemanticReranker::new(config);
        let query = make_query("anything");
        let candidates = vec![make_candidate("d1", 0.5, "content")]; // no metadata
        let results = reranker.rerank(&query, &candidates);
        let score = *results[0]
            .feature_scores
            .get("title_boost")
            .unwrap_or(&-1.0);
        assert!((score - 1.0).abs() < 1e-9);
    }

    // ── PositionPrior feature ─────────────────────────────────────────────────

    #[test]
    fn test_position_prior_first_rank() {
        let config = RerankConfig {
            features: vec![(RerankFeature::PositionPrior { decay: 0.5 }, 1.0)],
            normalize_scores: false,
            min_rerank_score: f64::NEG_INFINITY,
        };
        let reranker = SemanticReranker::new(config);
        let query = make_query("q");
        let candidate = make_candidate("d1", 0.8, "content");
        // rank=0, total=5 → rank_fraction=0.0 → score = 0.8 * (1 - 0) = 0.8
        let score = reranker.compute_feature(
            &RerankFeature::PositionPrior { decay: 0.5 },
            &query,
            &candidate,
            0,
            5,
        );
        assert!((score - 0.8).abs() < 1e-9);
    }

    #[test]
    fn test_position_prior_last_rank() {
        let config = RerankConfig {
            features: vec![(RerankFeature::PositionPrior { decay: 1.0 }, 1.0)],
            normalize_scores: false,
            min_rerank_score: f64::NEG_INFINITY,
        };
        let reranker = SemanticReranker::new(config);
        let query = make_query("q");
        let candidate = make_candidate("d1", 1.0, "content");
        // rank=4, total=5 → rank_fraction=0.8 → score = 1.0 * (1 - 1.0*0.8) = 0.2
        let score = reranker.compute_feature(
            &RerankFeature::PositionPrior { decay: 1.0 },
            &query,
            &candidate,
            4,
            5,
        );
        assert!((score - 0.2).abs() < 1e-9);
    }

    // ── top_k ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_top_k_returns_correct_count() {
        let mut reranker = SemanticReranker::new(RerankConfig::default());
        let query = make_query("rust");
        let candidates: Vec<RerankCandidate> = (0..10)
            .map(|i| make_candidate(&format!("d{i}"), i as f64 / 10.0, "rust content"))
            .collect();
        let results = reranker.rerank(&query, &candidates);
        let top3 = reranker.top_k(&results, 3);
        assert_eq!(top3.len(), 3);
    }

    #[test]
    fn test_top_k_larger_than_results() {
        let mut reranker = SemanticReranker::new(RerankConfig::default());
        let query = make_query("rust");
        let candidates = vec![
            make_candidate("d1", 0.9, "rust lang"),
            make_candidate("d2", 0.5, "python"),
        ];
        let results = reranker.rerank(&query, &candidates);
        let top10 = reranker.top_k(&results, 10);
        assert_eq!(top10.len(), results.len());
    }

    #[test]
    fn test_top_k_zero() {
        let reranker = SemanticReranker::new(RerankConfig::default());
        let results: Vec<crate::semantic_reranker::RerankResult> = vec![];
        let top = reranker.top_k(&results, 0);
        assert!(top.is_empty());
    }

    // ── precision_at_k ────────────────────────────────────────────────────────

    #[test]
    fn test_precision_at_k_all_relevant() {
        let mut reranker = SemanticReranker::new(RerankConfig::default());
        let query = make_query("rust");
        let candidates = vec![
            make_candidate("d1", 0.9, "rust lang"),
            make_candidate("d2", 0.8, "rust systems"),
        ];
        let results = reranker.rerank(&query, &candidates);
        let relevant = vec!["d1".to_string(), "d2".to_string()];
        let p = reranker.precision_at_k(&results, 2, &relevant);
        assert!((p - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_precision_at_k_none_relevant() {
        let mut reranker = SemanticReranker::new(RerankConfig::default());
        let query = make_query("rust");
        let candidates = vec![make_candidate("d1", 0.9, "rust lang")];
        let results = reranker.rerank(&query, &candidates);
        let relevant: Vec<String> = vec![];
        let p = reranker.precision_at_k(&results, 1, &relevant);
        assert_eq!(p, 0.0);
    }

    #[test]
    fn test_precision_at_k_zero_k() {
        let reranker = SemanticReranker::new(RerankConfig::default());
        let p = reranker.precision_at_k(&[], 0, &[]);
        assert_eq!(p, 0.0);
    }

    #[test]
    fn test_precision_at_k_partial() {
        let mut reranker = SemanticReranker::new(RerankConfig::default());
        let query = make_query("rust");
        let candidates = vec![
            make_candidate("d1", 0.9, "rust lang"),
            make_candidate("d2", 0.8, "python"),
            make_candidate("d3", 0.7, "rust sys"),
            make_candidate("d4", 0.6, "java"),
        ];
        let results = reranker.rerank(&query, &candidates);
        // Mark 2 of 4 as relevant.
        let relevant = vec!["d1".to_string(), "d3".to_string()];
        let p = reranker.precision_at_k(&results, 4, &relevant);
        assert!((p - 0.5).abs() < 1e-9);
    }

    // ── ndcg_at_k ─────────────────────────────────────────────────────────────

    #[test]
    fn test_ndcg_perfect_ranking() {
        let config = RerankConfig {
            features: vec![(RerankFeature::KeywordOverlap, 1.0)],
            normalize_scores: false,
            min_rerank_score: f64::NEG_INFINITY,
        };
        let mut reranker = SemanticReranker::new(config);
        let query = make_query("rust lang");
        let candidates = vec![
            make_candidate("d1", 0.9, "rust lang systems"),
            make_candidate("d2", 0.5, "python scripting"),
        ];
        let results = reranker.rerank(&query, &candidates);
        let relevant = vec!["d1".to_string()];
        let ndcg = reranker.ndcg_at_k(&results, 2, &relevant);
        // Perfect: relevant doc at rank 1 → NDCG = 1.0
        assert!((ndcg - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_ndcg_zero_k() {
        let reranker = SemanticReranker::new(RerankConfig::default());
        let ndcg = reranker.ndcg_at_k(&[], 0, &[]);
        assert_eq!(ndcg, 0.0);
    }

    #[test]
    fn test_ndcg_no_relevant_docs() {
        let mut reranker = SemanticReranker::new(RerankConfig::default());
        let query = make_query("rust");
        let candidates = vec![make_candidate("d1", 0.9, "rust lang")];
        let results = reranker.rerank(&query, &candidates);
        let ndcg = reranker.ndcg_at_k(&results, 1, &[]);
        assert_eq!(ndcg, 0.0);
    }

    #[test]
    fn test_ndcg_worst_case_ordering() {
        // Two candidates, relevant one placed last.
        let config = RerankConfig {
            features: vec![(RerankFeature::PositionPrior { decay: 0.0 }, 1.0)],
            normalize_scores: false,
            min_rerank_score: f64::NEG_INFINITY,
        };
        let mut reranker = SemanticReranker::new(config);
        let query = make_query("q");
        let candidates = vec![
            make_candidate("irrelevant", 0.9, "unrelated content"),
            make_candidate("relevant", 0.1, "matching content"),
        ];
        let results = reranker.rerank(&query, &candidates);
        let relevant = vec!["relevant".to_string()];
        let ndcg = reranker.ndcg_at_k(&results, 2, &relevant);
        // Relevant doc at rank 2, not rank 1 → NDCG < 1.0
        assert!(ndcg < 1.0);
    }

    // ── stats ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_stats_initial_zero() {
        let reranker = SemanticReranker::new(RerankConfig::default());
        let stats = reranker.stats();
        assert_eq!(stats.total_rerankings, 0);
        assert_eq!(stats.avg_candidates_per_reranking, 0.0);
    }

    #[test]
    fn test_stats_after_rerankings() {
        let mut reranker = SemanticReranker::new(RerankConfig::default());
        let query = make_query("rust");
        let c1 = vec![make_candidate("d1", 0.9, "rust lang")];
        let c2 = vec![
            make_candidate("d2", 0.7, "rust sys"),
            make_candidate("d3", 0.5, "python"),
        ];
        reranker.rerank(&query, &c1);
        reranker.rerank(&query, &c2);
        let stats = reranker.stats();
        assert_eq!(stats.total_rerankings, 2);
        // Average: (1 + 2) / 2 = 1.5
        assert!((stats.avg_candidates_per_reranking - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_stats_total_rerankings_increments() {
        let mut reranker = SemanticReranker::new(RerankConfig::default());
        let query = make_query("test");
        for _ in 0..5 {
            reranker.rerank(&query, &[]);
        }
        assert_eq!(reranker.total_rerankings, 5);
    }

    // ── normalize_scores ──────────────────────────────────────────────────────

    #[test]
    fn test_normalize_scores_range() {
        let config = RerankConfig {
            features: vec![(RerankFeature::KeywordOverlap, 1.0)],
            normalize_scores: true,
            min_rerank_score: f64::NEG_INFINITY,
        };
        let mut reranker = SemanticReranker::new(config);
        let query = make_query("rust lang");
        let candidates: Vec<RerankCandidate> = (0..5)
            .map(|i| make_candidate(&format!("d{i}"), 0.5, &format!("rust lang doc {i}")))
            .collect();
        let results = reranker.rerank(&query, &candidates);
        if results.len() > 1 {
            let max = results
                .iter()
                .map(|r| r.rerank_score)
                .fold(f64::NEG_INFINITY, f64::max);
            let min = results
                .iter()
                .map(|r| r.rerank_score)
                .fold(f64::INFINITY, f64::min);
            // After normalisation, max should be 1.0 and min should be 0.0
            // unless all scores are identical (in which case no normalisation occurs).
            assert!(max <= 1.0 + 1e-9);
            assert!(min >= -1e-9);
        }
    }

    // ── default config ────────────────────────────────────────────────────────

    #[test]
    fn test_default_config_has_four_features() {
        let config = RerankConfig::default();
        assert_eq!(config.features.len(), 4);
    }

    #[test]
    fn test_default_config_weights_sum_to_one() {
        let config = RerankConfig::default();
        let total: f64 = config.features.iter().map(|(_, w)| w).sum();
        assert!((total - 1.0).abs() < 1e-9);
    }

    // ── score_candidate keys ──────────────────────────────────────────────────

    #[test]
    fn test_score_candidate_all_feature_keys_present() {
        let config = RerankConfig {
            features: vec![
                (RerankFeature::EmbeddingScore, 0.25),
                (RerankFeature::KeywordOverlap, 0.25),
                (RerankFeature::LengthPenalty, 0.25),
                (RerankFeature::PositionPrior { decay: 0.1 }, 0.25),
            ],
            normalize_scores: false,
            min_rerank_score: f64::NEG_INFINITY,
        };
        let reranker = SemanticReranker::new(config);
        let query = make_query("test");
        let candidate = make_candidate("d1", 0.5, "some content here");
        let scores = reranker.score_candidate(&query, &candidate, 0, 1);
        assert!(scores.contains_key("embedding_score"));
        assert!(scores.contains_key("keyword_overlap"));
        assert!(scores.contains_key("length_penalty"));
        assert!(scores.contains_key("position_prior"));
    }

    // ── single candidate edge case ────────────────────────────────────────────

    #[test]
    fn test_single_candidate_rank_is_one() {
        let mut reranker = SemanticReranker::new(RerankConfig::default());
        let query = make_query("test");
        let candidates = vec![make_candidate("d1", 0.5, "some content")];
        let results = reranker.rerank(&query, &candidates);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].rank, 1);
    }

    // ── keyword overlap with context ──────────────────────────────────────────

    #[test]
    fn test_keyword_overlap_case_insensitive() {
        let a = SemanticReranker::tokenize("Rust LANG");
        let b = SemanticReranker::tokenize("rust lang");
        // Both should yield the same tokens after lowercasing.
        assert_eq!(a, b);
    }

    // ── multiple feature weights are normalised ───────────────────────────────

    #[test]
    fn test_unequal_weights_still_produce_valid_scores() {
        let config = RerankConfig {
            features: vec![
                (RerankFeature::KeywordOverlap, 10.0),
                (RerankFeature::LengthPenalty, 5.0),
            ],
            normalize_scores: false,
            min_rerank_score: f64::NEG_INFINITY,
        };
        let mut reranker = SemanticReranker::new(config);
        let query = make_query("rust lang");
        let candidates = vec![make_candidate("d1", 0.9, "rust lang systems")];
        let results = reranker.rerank(&query, &candidates);
        assert!(!results.is_empty());
        // Score should be finite.
        assert!(results[0].rerank_score.is_finite());
    }
}
