//! Search quality evaluation for HNSW-based retrieval systems.
//!
//! This module provides standard information retrieval metrics for evaluating
//! the quality of approximate nearest neighbour search results against a
//! ground-truth relevance set.
//!
//! ## Metrics
//!
//! | Metric | Description |
//! |--------|-------------|
//! | Recall\@K | Fraction of relevant documents retrieved in top-K |
//! | Precision\@K | Fraction of top-K results that are relevant |
//! | NDCG\@K | Normalised Discounted Cumulative Gain (binary relevance) |
//! | Average Precision | Area under the precision-recall curve |
//! | Reciprocal Rank | `1 / rank` of the first relevant result |
//!
//! ## Example
//!
//! ```rust
//! use ipfrs_semantic::search_quality::{
//!     GroundTruth, SearchResultSet, SearchQualityEvaluator,
//! };
//!
//! let gt = GroundTruth {
//!     query_id: "q1".to_string(),
//!     relevant_ids: vec!["a".to_string(), "b".to_string(), "c".to_string()],
//!     top_k: 3,
//! };
//!
//! let results = SearchResultSet {
//!     query_id: "q1".to_string(),
//!     result_ids: vec!["a".to_string(), "b".to_string(), "d".to_string()],
//! };
//!
//! let evaluator = SearchQualityEvaluator::new();
//! let metrics = evaluator.evaluate(&gt, &results).unwrap();
//!
//! assert!((metrics.recall_at_k - 2.0 / 3.0).abs() < 1e-9);
//! assert!((metrics.precision_at_k - 2.0 / 3.0).abs() < 1e-9);
//! ```

use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use thiserror::Error;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors returned by [`SearchQualityEvaluator`].
#[derive(Debug, Error, PartialEq, Eq, Clone)]
pub enum EvalError {
    /// The `query_id` fields of the ground-truth and result set do not match.
    #[error("query-id mismatch: ground truth is '{ground_truth}', results are '{results}'")]
    QueryIdMismatch {
        /// Query ID from the ground-truth.
        ground_truth: String,
        /// Query ID from the result set.
        results: String,
    },

    /// The result set contains no items.
    #[error("result set is empty")]
    EmptyResults,

    /// The ground-truth contains no relevant documents.
    #[error("ground truth contains no relevant documents")]
    EmptyGroundTruth,
}

// ─────────────────────────────────────────────────────────────────────────────
// Data structures
// ─────────────────────────────────────────────────────────────────────────────

/// Ground-truth specification for a single query.
///
/// The `relevant_ids` list is **ordered by decreasing relevance**, which is
/// used when computing NDCG ideal rankings.  For binary-relevance metrics
/// (Recall, Precision, AP, RR) the ordering does not matter.
#[derive(Debug, Clone)]
pub struct GroundTruth {
    /// Unique identifier for the query.
    pub query_id: String,
    /// Document IDs that are relevant for this query, ordered by relevance.
    pub relevant_ids: Vec<String>,
    /// Evaluation cut-off (K).
    pub top_k: usize,
}

/// System output for a single query.
#[derive(Debug, Clone)]
pub struct SearchResultSet {
    /// Unique identifier for the query (must match the corresponding [`GroundTruth`]).
    pub query_id: String,
    /// Retrieved document IDs in rank order (highest-scoring first).
    pub result_ids: Vec<String>,
}

/// All quality metrics for a single query evaluation.
#[derive(Debug, Clone, PartialEq)]
pub struct QualityMetrics {
    /// |relevant ∩ results\@K| / min(|relevant|, K)
    pub recall_at_k: f64,
    /// |relevant ∩ results\@K| / K
    pub precision_at_k: f64,
    /// Normalised Discounted Cumulative Gain at K (binary relevance).
    pub ndcg_at_k: f64,
    /// Average Precision — area under the precision-recall curve up to K.
    pub average_precision: f64,
    /// Reciprocal Rank — `1 / rank` of the first relevant result (0 if none).
    pub reciprocal_rank: f64,
}

impl QualityMetrics {
    /// Returns a zero-valued metrics instance (used as additive identity when
    /// computing macro-averages).
    fn zero() -> Self {
        Self {
            recall_at_k: 0.0,
            precision_at_k: 0.0,
            ndcg_at_k: 0.0,
            average_precision: 0.0,
            reciprocal_rank: 0.0,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Atomic statistics
// ─────────────────────────────────────────────────────────────────────────────

/// A point-in-time snapshot of [`EvaluatorStats`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvaluatorStatsSnapshot {
    /// Total number of individual query evaluations performed.
    pub total_evaluated: u64,
    /// Total number of [`SearchQualityEvaluator::batch_evaluate`] calls.
    pub total_batches: u64,
}

/// Lock-free atomic counters for the evaluator.
#[derive(Debug)]
pub struct EvaluatorStats {
    /// Monotonically increasing count of single-query evaluations.
    pub total_evaluated: AtomicU64,
    /// Monotonically increasing count of batch evaluation calls.
    pub total_batches: AtomicU64,
}

impl EvaluatorStats {
    fn new() -> Self {
        Self {
            total_evaluated: AtomicU64::new(0),
            total_batches: AtomicU64::new(0),
        }
    }

    /// Returns a consistent snapshot of the current counter values.
    pub fn snapshot(&self) -> EvaluatorStatsSnapshot {
        EvaluatorStatsSnapshot {
            total_evaluated: self.total_evaluated.load(Ordering::Relaxed),
            total_batches: self.total_batches.load(Ordering::Relaxed),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Evaluator
// ─────────────────────────────────────────────────────────────────────────────

/// Computes information-retrieval quality metrics for HNSW search results.
///
/// The evaluator is cheap to clone because all mutable state lives behind an
/// [`Arc`].  Multiple threads can share a single evaluator instance.
#[derive(Debug, Clone)]
pub struct SearchQualityEvaluator {
    /// Operational statistics.
    pub stats: Arc<EvaluatorStats>,
}

impl Default for SearchQualityEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

impl SearchQualityEvaluator {
    /// Creates a new evaluator with zeroed statistics.
    pub fn new() -> Self {
        Self {
            stats: Arc::new(EvaluatorStats::new()),
        }
    }

    // ── Public API ────────────────────────────────────────────────────────────

    /// Evaluates a single query, computing all five quality metrics.
    ///
    /// # Errors
    ///
    /// Returns [`EvalError::QueryIdMismatch`] when the `query_id` fields
    /// differ, [`EvalError::EmptyResults`] when the result set is empty, or
    /// [`EvalError::EmptyGroundTruth`] when the ground-truth has no relevant
    /// documents.
    pub fn evaluate(
        &self,
        ground_truth: &GroundTruth,
        results: &SearchResultSet,
    ) -> Result<QualityMetrics, EvalError> {
        // ── Guard checks ──────────────────────────────────────────────────────
        if ground_truth.query_id != results.query_id {
            return Err(EvalError::QueryIdMismatch {
                ground_truth: ground_truth.query_id.clone(),
                results: results.query_id.clone(),
            });
        }

        if results.result_ids.is_empty() {
            return Err(EvalError::EmptyResults);
        }

        if ground_truth.relevant_ids.is_empty() {
            return Err(EvalError::EmptyGroundTruth);
        }

        let k = ground_truth.top_k;
        let relevant = &ground_truth.relevant_ids;
        let retrieved = &results.result_ids;

        let recall = Self::recall_at_k(relevant, retrieved, k);
        let precision = Self::precision_at_k(relevant, retrieved, k);
        let ndcg = Self::ndcg_at_k(relevant, retrieved, k);
        let ap = Self::average_precision_at_k(relevant, retrieved, k);
        let rr = Self::reciprocal_rank(relevant, retrieved);

        self.stats.total_evaluated.fetch_add(1, Ordering::Relaxed);

        Ok(QualityMetrics {
            recall_at_k: recall,
            precision_at_k: precision,
            ndcg_at_k: ndcg,
            average_precision: ap,
            reciprocal_rank: rr,
        })
    }

    /// Evaluates a batch of (ground-truth, result-set) pairs in order.
    ///
    /// Each entry is evaluated independently; errors are captured per-entry
    /// rather than aborting the batch.
    pub fn batch_evaluate(
        &self,
        pairs: &[(GroundTruth, SearchResultSet)],
    ) -> Vec<Result<QualityMetrics, EvalError>> {
        self.stats.total_batches.fetch_add(1, Ordering::Relaxed);
        pairs.iter().map(|(gt, rs)| self.evaluate(gt, rs)).collect()
    }

    /// Computes the macro-average of a slice of [`QualityMetrics`].
    ///
    /// Returns zero metrics when the slice is empty.
    pub fn mean_metrics(&self, metrics: &[QualityMetrics]) -> QualityMetrics {
        if metrics.is_empty() {
            return QualityMetrics::zero();
        }

        let n = metrics.len() as f64;
        let sum = metrics
            .iter()
            .fold(QualityMetrics::zero(), |acc, m| QualityMetrics {
                recall_at_k: acc.recall_at_k + m.recall_at_k,
                precision_at_k: acc.precision_at_k + m.precision_at_k,
                ndcg_at_k: acc.ndcg_at_k + m.ndcg_at_k,
                average_precision: acc.average_precision + m.average_precision,
                reciprocal_rank: acc.reciprocal_rank + m.reciprocal_rank,
            });

        QualityMetrics {
            recall_at_k: sum.recall_at_k / n,
            precision_at_k: sum.precision_at_k / n,
            ndcg_at_k: sum.ndcg_at_k / n,
            average_precision: sum.average_precision / n,
            reciprocal_rank: sum.reciprocal_rank / n,
        }
    }

    // ── Static helpers ────────────────────────────────────────────────────────

    /// Computes Recall\@K.
    ///
    /// `|relevant ∩ results[..k]| / min(|relevant|, k)`
    pub fn recall_at_k(relevant: &[String], results: &[String], k: usize) -> f64 {
        if relevant.is_empty() || k == 0 {
            return 0.0;
        }

        let relevant_set: HashSet<&str> = relevant.iter().map(String::as_str).collect();
        let top_k = &results[..k.min(results.len())];
        let hits = top_k
            .iter()
            .filter(|id| relevant_set.contains(id.as_str()))
            .count();

        let denominator = relevant.len().min(k);
        hits as f64 / denominator as f64
    }

    /// Computes Precision\@K.
    ///
    /// `|relevant ∩ results[..k]| / k`
    pub fn precision_at_k(relevant: &[String], results: &[String], k: usize) -> f64 {
        if k == 0 || results.is_empty() {
            return 0.0;
        }

        let relevant_set: HashSet<&str> = relevant.iter().map(String::as_str).collect();
        let top_k = &results[..k.min(results.len())];
        let hits = top_k
            .iter()
            .filter(|id| relevant_set.contains(id.as_str()))
            .count();

        hits as f64 / k as f64
    }

    /// Computes binary-relevance NDCG\@K.
    ///
    /// DCG  = Σ `rel_i / log2(i + 2)` for `i` in `0..k` (0-indexed rank).
    /// IDCG = Σ `1 / log2(i + 2)`     for `i` in `0..min(|relevant|, k)`.
    /// NDCG = DCG / IDCG.
    pub fn ndcg_at_k(relevant: &[String], results: &[String], k: usize) -> f64 {
        if relevant.is_empty() || k == 0 {
            return 0.0;
        }

        let relevant_set: HashSet<&str> = relevant.iter().map(String::as_str).collect();
        let top_k = &results[..k.min(results.len())];

        let dcg: f64 = top_k
            .iter()
            .enumerate()
            .map(|(i, id)| {
                let rel = if relevant_set.contains(id.as_str()) {
                    1.0_f64
                } else {
                    0.0_f64
                };
                rel / (i as f64 + 2.0_f64).log2()
            })
            .sum();

        let ideal_hits = relevant.len().min(k);
        let idcg: f64 = (0..ideal_hits)
            .map(|i| 1.0_f64 / (i as f64 + 2.0_f64).log2())
            .sum();

        if idcg < f64::EPSILON {
            return 0.0;
        }

        dcg / idcg
    }

    /// Computes Average Precision\@K (AP\@K).
    ///
    /// AP = (1 / |relevant|) × Σ P\@i × rel_i  for i in 1..=k.
    fn average_precision_at_k(relevant: &[String], results: &[String], k: usize) -> f64 {
        if relevant.is_empty() || k == 0 || results.is_empty() {
            return 0.0;
        }

        let relevant_set: HashSet<&str> = relevant.iter().map(String::as_str).collect();
        let top_k = &results[..k.min(results.len())];

        let mut hits = 0usize;
        let mut sum = 0.0_f64;

        for (i, id) in top_k.iter().enumerate() {
            if relevant_set.contains(id.as_str()) {
                hits += 1;
                // P@(i+1)
                sum += hits as f64 / (i + 1) as f64;
            }
        }

        if relevant.is_empty() {
            return 0.0;
        }

        sum / relevant.len() as f64
    }

    /// Computes Reciprocal Rank (RR).
    ///
    /// Returns `1 / rank` of the first relevant result in `results` (1-indexed),
    /// or `0.0` when no relevant document appears in the list.
    fn reciprocal_rank(relevant: &[String], results: &[String]) -> f64 {
        if relevant.is_empty() || results.is_empty() {
            return 0.0;
        }

        let relevant_set: HashSet<&str> = relevant.iter().map(String::as_str).collect();
        for (i, id) in results.iter().enumerate() {
            if relevant_set.contains(id.as_str()) {
                return 1.0 / (i + 1) as f64;
            }
        }
        0.0
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helper ────────────────────────────────────────────────────────────────

    fn sv(ids: &[&str]) -> Vec<String> {
        ids.iter().map(|s| s.to_string()).collect()
    }

    fn gt(query_id: &str, relevant: &[&str], k: usize) -> GroundTruth {
        GroundTruth {
            query_id: query_id.to_string(),
            relevant_ids: sv(relevant),
            top_k: k,
        }
    }

    fn rs(query_id: &str, results: &[&str]) -> SearchResultSet {
        SearchResultSet {
            query_id: query_id.to_string(),
            result_ids: sv(results),
        }
    }

    // ── 1. Perfect recall ─────────────────────────────────────────────────────

    #[test]
    fn test_perfect_recall() {
        let recall =
            SearchQualityEvaluator::recall_at_k(&sv(&["a", "b", "c"]), &sv(&["a", "b", "c"]), 3);
        assert!(
            (recall - 1.0).abs() < 1e-9,
            "perfect recall should be 1.0, got {recall}"
        );
    }

    // ── 2. Zero recall ────────────────────────────────────────────────────────

    #[test]
    fn test_zero_recall() {
        let recall =
            SearchQualityEvaluator::recall_at_k(&sv(&["a", "b", "c"]), &sv(&["x", "y", "z"]), 3);
        assert!(
            (recall - 0.0).abs() < 1e-9,
            "zero recall should be 0.0, got {recall}"
        );
    }

    // ── 3. Partial recall ─────────────────────────────────────────────────────

    #[test]
    fn test_partial_recall() {
        // 2 out of 3 relevant returned in top-3, denominator = min(3,3) = 3
        let recall =
            SearchQualityEvaluator::recall_at_k(&sv(&["a", "b", "c"]), &sv(&["a", "b", "x"]), 3);
        let expected = 2.0 / 3.0;
        assert!(
            (recall - expected).abs() < 1e-9,
            "partial recall {recall} ≠ {expected}"
        );
    }

    // ── 4. Precision@K ────────────────────────────────────────────────────────

    #[test]
    fn test_precision_at_k() {
        // 2 hits in top-4, P@4 = 2/4 = 0.5
        let prec = SearchQualityEvaluator::precision_at_k(
            &sv(&["a", "b", "c"]),
            &sv(&["a", "x", "b", "y"]),
            4,
        );
        let expected = 2.0 / 4.0;
        assert!((prec - expected).abs() < 1e-9, "P@4 {prec} ≠ {expected}");
    }

    #[test]
    fn test_precision_at_k_perfect() {
        let prec = SearchQualityEvaluator::precision_at_k(
            &sv(&["a", "b", "c"]),
            &sv(&["a", "b", "c", "d"]),
            3,
        );
        assert!(
            (prec - 1.0).abs() < 1e-9,
            "perfect P@3 should be 1.0, got {prec}"
        );
    }

    // ── 5. NDCG@K perfect ranking ─────────────────────────────────────────────

    #[test]
    fn test_ndcg_perfect_ranking() {
        // All top-k are relevant → NDCG = 1.0
        let ndcg =
            SearchQualityEvaluator::ndcg_at_k(&sv(&["a", "b", "c"]), &sv(&["a", "b", "c"]), 3);
        assert!(
            (ndcg - 1.0).abs() < 1e-9,
            "perfect NDCG should be 1.0, got {ndcg}"
        );
    }

    // ── 6. NDCG@K worst ranking ───────────────────────────────────────────────

    #[test]
    fn test_ndcg_worst_ranking() {
        // No relevant documents in top-3 → NDCG = 0.0
        let ndcg =
            SearchQualityEvaluator::ndcg_at_k(&sv(&["a", "b", "c"]), &sv(&["x", "y", "z"]), 3);
        assert!(
            (ndcg - 0.0).abs() < 1e-9,
            "worst NDCG should be 0.0, got {ndcg}"
        );
    }

    #[test]
    fn test_ndcg_partial_ranking() {
        // Only first position is relevant.
        // DCG  = 1/log2(2) = 1.0
        // IDCG = 1/log2(2) + 1/log2(3) = 1.0 + 0.6309… = 1.6309…
        // NDCG = 1.0 / 1.6309… ≈ 0.6131
        let ndcg = SearchQualityEvaluator::ndcg_at_k(&sv(&["a", "b"]), &sv(&["a", "x", "y"]), 3);
        let idcg = 1.0_f64 / 2.0_f64.log2() + 1.0_f64 / 3.0_f64.log2();
        let expected = (1.0_f64 / 2.0_f64.log2()) / idcg;
        assert!(
            (ndcg - expected).abs() < 1e-9,
            "partial NDCG {ndcg} ≠ {expected}"
        );
    }

    // ── 7. Average Precision ──────────────────────────────────────────────────

    #[test]
    fn test_average_precision_perfect() {
        let evaluator = SearchQualityEvaluator::new();
        let m = evaluator
            .evaluate(&gt("q", &["a", "b", "c"], 3), &rs("q", &["a", "b", "c"]))
            .expect("test: evaluate should succeed for valid perfect-match inputs");
        // P@1=1, P@2=1, P@3=1 → AP = (1+1+1)/3 = 1.0
        assert!(
            (m.average_precision - 1.0).abs() < 1e-9,
            "AP={}",
            m.average_precision
        );
    }

    #[test]
    fn test_average_precision_interleaved() {
        // relevant = [a, b, c], results = [a, x, b, y, c]
        // hits at ranks 1, 3, 5 → P@1=1/1, P@3=2/3, P@5=3/5
        // AP = (1 + 2/3 + 3/5) / 3
        let evaluator = SearchQualityEvaluator::new();
        let m = evaluator
            .evaluate(
                &gt("q", &["a", "b", "c"], 5),
                &rs("q", &["a", "x", "b", "y", "c"]),
            )
            .expect("test: evaluate should succeed for valid interleaved inputs");
        let expected = (1.0_f64 + 2.0 / 3.0 + 3.0 / 5.0) / 3.0;
        assert!(
            (m.average_precision - expected).abs() < 1e-9,
            "AP={} ≠ {expected}",
            m.average_precision
        );
    }

    // ── 8. Reciprocal Rank ────────────────────────────────────────────────────

    #[test]
    fn test_reciprocal_rank_first_hit_rank1() {
        let evaluator = SearchQualityEvaluator::new();
        let m = evaluator
            .evaluate(&gt("q", &["a"], 3), &rs("q", &["a", "b", "c"]))
            .expect("test: evaluate should succeed when first result is the only relevant doc");
        assert!(
            (m.reciprocal_rank - 1.0).abs() < 1e-9,
            "RR={}",
            m.reciprocal_rank
        );
    }

    #[test]
    fn test_reciprocal_rank_first_hit_rank3() {
        let evaluator = SearchQualityEvaluator::new();
        let m = evaluator
            .evaluate(&gt("q", &["c"], 3), &rs("q", &["a", "b", "c"]))
            .expect("test: evaluate should succeed when relevant doc appears at rank 3");
        assert!(
            (m.reciprocal_rank - 1.0 / 3.0).abs() < 1e-9,
            "RR={}",
            m.reciprocal_rank
        );
    }

    #[test]
    fn test_reciprocal_rank_no_hit() {
        let evaluator = SearchQualityEvaluator::new();
        let m = evaluator
            .evaluate(&gt("q", &["z"], 3), &rs("q", &["a", "b", "c"]))
            .expect("test: evaluate should succeed even when no relevant doc appears in results");
        assert!(
            (m.reciprocal_rank - 0.0).abs() < 1e-9,
            "RR={}",
            m.reciprocal_rank
        );
    }

    // ── 9. batch_evaluate ─────────────────────────────────────────────────────

    #[test]
    fn test_batch_evaluate_returns_all() {
        let evaluator = SearchQualityEvaluator::new();

        let pairs = vec![
            (gt("q1", &["a", "b"], 2), rs("q1", &["a", "b"])),
            (gt("q2", &["x"], 2), rs("q2", &["x", "y"])),
            (gt("q3", &["p"], 2), rs("q3", &["z", "w"])),
        ];

        let results = evaluator.batch_evaluate(&pairs);
        assert_eq!(results.len(), 3, "batch should return one result per pair");

        // All should be Ok
        for r in &results {
            assert!(r.is_ok(), "unexpected error: {r:?}");
        }

        let snap = evaluator.stats.snapshot();
        assert_eq!(snap.total_batches, 1);
        assert_eq!(snap.total_evaluated, 3);
    }

    // ── 10. mean_metrics ──────────────────────────────────────────────────────

    #[test]
    fn test_mean_metrics_averages_correctly() {
        let evaluator = SearchQualityEvaluator::new();

        let m1 = QualityMetrics {
            recall_at_k: 1.0,
            precision_at_k: 1.0,
            ndcg_at_k: 1.0,
            average_precision: 1.0,
            reciprocal_rank: 1.0,
        };
        let m2 = QualityMetrics {
            recall_at_k: 0.0,
            precision_at_k: 0.0,
            ndcg_at_k: 0.0,
            average_precision: 0.0,
            reciprocal_rank: 0.0,
        };

        let mean = evaluator.mean_metrics(&[m1, m2]);
        assert!((mean.recall_at_k - 0.5).abs() < 1e-9);
        assert!((mean.precision_at_k - 0.5).abs() < 1e-9);
        assert!((mean.ndcg_at_k - 0.5).abs() < 1e-9);
        assert!((mean.average_precision - 0.5).abs() < 1e-9);
        assert!((mean.reciprocal_rank - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_mean_metrics_empty_slice() {
        let evaluator = SearchQualityEvaluator::new();
        let mean = evaluator.mean_metrics(&[]);
        assert_eq!(mean, QualityMetrics::zero());
    }

    // ── 11. QueryId mismatch error ────────────────────────────────────────────

    #[test]
    fn test_query_id_mismatch_error() {
        let evaluator = SearchQualityEvaluator::new();
        let err = evaluator
            .evaluate(&gt("q1", &["a"], 3), &rs("q2", &["a"]))
            .unwrap_err();

        assert_eq!(
            err,
            EvalError::QueryIdMismatch {
                ground_truth: "q1".to_string(),
                results: "q2".to_string(),
            }
        );
    }

    // ── 12. Empty results error ───────────────────────────────────────────────

    #[test]
    fn test_empty_results_error() {
        let evaluator = SearchQualityEvaluator::new();
        let err = evaluator
            .evaluate(&gt("q", &["a"], 3), &rs("q", &[]))
            .unwrap_err();

        assert_eq!(err, EvalError::EmptyResults);
    }

    // ── 13. Empty ground truth error ──────────────────────────────────────────

    #[test]
    fn test_empty_ground_truth_error() {
        let evaluator = SearchQualityEvaluator::new();
        let err = evaluator
            .evaluate(&gt("q", &[], 3), &rs("q", &["a"]))
            .unwrap_err();

        assert_eq!(err, EvalError::EmptyGroundTruth);
    }

    // ── 14. Stats increments ──────────────────────────────────────────────────

    #[test]
    fn test_stats_increment_on_evaluate() {
        let evaluator = SearchQualityEvaluator::new();

        for _ in 0..5 {
            let _ = evaluator.evaluate(&gt("q", &["a"], 1), &rs("q", &["a"]));
        }

        assert_eq!(evaluator.stats.snapshot().total_evaluated, 5);
    }

    // ── 15. recall_at_k with k > len(results) ────────────────────────────────

    #[test]
    fn test_recall_k_larger_than_results() {
        // k=10 but only 3 results available; 2 are relevant out of 4 ground-truth
        // hits = 2, denominator = min(4, 10) = 4 → recall = 0.5
        let recall = SearchQualityEvaluator::recall_at_k(
            &sv(&["a", "b", "c", "d"]),
            &sv(&["a", "b", "x"]),
            10,
        );
        let expected = 2.0 / 4.0;
        assert!(
            (recall - expected).abs() < 1e-9,
            "recall {recall} ≠ {expected}"
        );
    }

    // ── 16. Full pipeline: evaluate + batch + mean ────────────────────────────

    #[test]
    fn test_full_pipeline() {
        let evaluator = SearchQualityEvaluator::new();

        let pairs = vec![
            // Perfect results
            (gt("q1", &["a", "b", "c"], 3), rs("q1", &["a", "b", "c"])),
            // No relevant results
            (gt("q2", &["a", "b", "c"], 3), rs("q2", &["x", "y", "z"])),
        ];

        let batch = evaluator.batch_evaluate(&pairs);
        let metrics: Vec<QualityMetrics> = batch
            .into_iter()
            .map(|r| r.expect("test: each batch entry should evaluate without error"))
            .collect();

        let mean = evaluator.mean_metrics(&metrics);

        // Mean of 1.0 and 0.0 = 0.5 for all metrics involving pure hits
        assert!(
            (mean.recall_at_k - 0.5).abs() < 1e-9,
            "mean recall={}",
            mean.recall_at_k
        );
        assert!((mean.precision_at_k - 0.5).abs() < 1e-9);
        assert!((mean.ndcg_at_k - 0.5).abs() < 1e-9);
    }
}
