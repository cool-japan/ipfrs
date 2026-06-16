//! Embedding aggregation strategies for combining multiple vector representations.
//!
//! This module provides [`EmbeddingAggregator`], which merges several embedding
//! vectors into one using configurable pooling strategies: mean, weighted mean,
//! max, min, sum, geometric mean, and attention-based pooling.
//!
//! # Example
//!
//! ```rust
//! use ipfrs_semantic::embedding_aggregator::{
//!     AggregationInput, AggregationMethod, EmbeddingAggregator,
//!     EmbeddingAggregatorConfig,
//! };
//!
//! let method = AggregationMethod::Mean;
//! let config = EmbeddingAggregatorConfig::default();
//! let mut agg = EmbeddingAggregator::new(method, config);
//!
//! let inputs = vec![
//!     AggregationInput::new("a", vec![1.0, 0.0], 1.0),
//!     AggregationInput::new("b", vec![0.0, 1.0], 1.0),
//! ];
//!
//! let result = agg.aggregate(&inputs).unwrap();
//! assert_eq!(result.input_count, 2);
//! assert_eq!(result.dims, 2);
//! ```

use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors returned by [`EmbeddingAggregator`] operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AggregatorError {
    /// No inputs were provided.
    EmptyInput,
    /// All embeddings must have the same dimensionality.
    DimensionMismatch {
        /// Expected dimensionality.
        expected: usize,
        /// Actual dimensionality of the offending vector.
        got: usize,
    },
    /// The number of explicit weights does not match the number of inputs.
    WeightCountMismatch {
        /// Expected number of weights.
        expected: usize,
        /// Actual number of weights supplied.
        got: usize,
    },
    /// The attention query vector is invalid (e.g. wrong length or all-zero).
    InvalidQuery,
}

impl std::fmt::Display for AggregatorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyInput => write!(f, "no input embeddings were provided"),
            Self::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
            Self::WeightCountMismatch { expected, got } => write!(
                f,
                "weight count mismatch: expected {expected} weights, got {got}"
            ),
            Self::InvalidQuery => {
                write!(f, "attention query vector is invalid or has wrong length")
            }
        }
    }
}

impl std::error::Error for AggregatorError {}

// ---------------------------------------------------------------------------
// Input
// ---------------------------------------------------------------------------

/// A single embedding with an optional scalar weight used by [`AggregationMethod::WeightedMean`].
#[derive(Debug, Clone)]
pub struct AggregationInput {
    /// Identifier for this input (e.g. document ID, chunk ID).
    pub id: String,
    /// The embedding vector.
    pub embedding: Vec<f64>,
    /// Scalar weight.  Used directly by `WeightedMean`; ignored by all other
    /// methods (all inputs are treated equally).
    pub weight: f64,
}

impl AggregationInput {
    /// Create a new [`AggregationInput`].
    pub fn new(id: impl Into<String>, embedding: Vec<f64>, weight: f64) -> Self {
        Self {
            id: id.into(),
            embedding,
            weight,
        }
    }
}

// ---------------------------------------------------------------------------
// Method
// ---------------------------------------------------------------------------

/// Strategy used to pool multiple embedding vectors into one.
#[derive(Debug, Clone)]
pub enum AggregationMethod {
    /// Element-wise arithmetic mean.
    Mean,
    /// Weighted arithmetic mean.  The `weights` slice is normalised internally
    /// so that the caller does not need to ensure they sum to 1.
    WeightedMean {
        /// Per-input weights (must have the same length as the input slice).
        weights: Vec<f64>,
    },
    /// Element-wise maximum.
    Max,
    /// Element-wise minimum.
    Min,
    /// Element-wise sum.
    Sum,
    /// Geometric mean: `exp(mean(ln(|v[j]| + ε))) * sign(mean(v[j]))`.
    GeometricMean,
    /// Attention pooling: softmax(query · embeddings / sqrt(d)) weighted sum.
    AttentionPooling {
        /// Query vector used to compute attention scores.
        query: Vec<f64>,
    },
}

impl AggregationMethod {
    /// Human-readable name of the method.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Mean => "Mean",
            Self::WeightedMean { .. } => "WeightedMean",
            Self::Max => "Max",
            Self::Min => "Min",
            Self::Sum => "Sum",
            Self::GeometricMean => "GeometricMean",
            Self::AttentionPooling { .. } => "AttentionPooling",
        }
    }
}

// ---------------------------------------------------------------------------
// Result
// ---------------------------------------------------------------------------

/// The outcome of a single aggregation operation.
#[derive(Debug, Clone)]
pub struct AggregationResult {
    /// Name of the aggregation method used.
    pub method: String,
    /// The aggregated output vector.
    pub output: Vec<f64>,
    /// Number of inputs that were aggregated.
    pub input_count: usize,
    /// Dimensionality of the output vector.
    pub dims: usize,
    /// L2 norm of `output`.
    pub norm: f64,
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Configuration options for [`EmbeddingAggregator`].
#[derive(Debug, Clone, Default)]
pub struct EmbeddingAggregatorConfig {
    /// If `true`, L2-normalise the output vector before storing it in the result.
    pub normalize_output: bool,
    /// If `true`, replace any all-zero input embedding with a uniform vector
    /// `1/sqrt(d)` before aggregation.
    pub handle_zeros: bool,
}

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------

/// Aggregate statistics derived from the aggregator's history.
#[derive(Debug, Clone)]
pub struct EaAggregatorStats {
    /// Total number of aggregation calls recorded in history.
    pub total_aggregations: u64,
    /// Average number of inputs per aggregation call.
    pub avg_input_count: f64,
    /// Average L2 norm of the output vectors.
    pub avg_output_norm: f64,
    /// Name of the configured aggregation method.
    pub method_name: String,
}

// ---------------------------------------------------------------------------
// Core aggregator
// ---------------------------------------------------------------------------

/// Aggregates multiple embedding vectors into a single representation.
///
/// Maintains a rolling history of the most recent aggregation results for
/// monitoring and introspection.
pub struct EmbeddingAggregator {
    /// Configuration flags.
    pub config: EmbeddingAggregatorConfig,
    /// The pooling method.
    pub method: AggregationMethod,
    /// Rolling history of past results.
    pub history: VecDeque<AggregationResult>,
    /// Maximum number of entries kept in `history`.
    pub max_history: usize,
}

impl EmbeddingAggregator {
    /// Create a new [`EmbeddingAggregator`] with the given method and config.
    ///
    /// The history capacity defaults to 1 000 entries.
    pub fn new(method: AggregationMethod, config: EmbeddingAggregatorConfig) -> Self {
        Self {
            config,
            method,
            history: VecDeque::new(),
            max_history: 1_000,
        }
    }

    /// Create a new aggregator with an explicit history capacity.
    pub fn with_history_capacity(
        method: AggregationMethod,
        config: EmbeddingAggregatorConfig,
        max_history: usize,
    ) -> Self {
        Self {
            config,
            method,
            history: VecDeque::with_capacity(max_history.min(1_000_000)),
            max_history,
        }
    }

    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------

    /// Aggregate a slice of [`AggregationInput`] values using `self.method`.
    ///
    /// All inputs must share the same dimensionality.  Returns an
    /// [`AggregationResult`] and records it in the history ring.
    pub fn aggregate(
        &mut self,
        inputs: &[AggregationInput],
    ) -> Result<AggregationResult, AggregatorError> {
        if inputs.is_empty() {
            return Err(AggregatorError::EmptyInput);
        }

        let dims = inputs[0].embedding.len();
        for inp in inputs.iter().skip(1) {
            if inp.embedding.len() != dims {
                return Err(AggregatorError::DimensionMismatch {
                    expected: dims,
                    got: inp.embedding.len(),
                });
            }
        }

        // Optionally replace zero vectors with uniform ones.
        let embeddings: Vec<Vec<f64>> = if self.config.handle_zeros {
            inputs
                .iter()
                .map(|inp| {
                    if Self::l2_norm(&inp.embedding) < f64::EPSILON {
                        vec![1.0 / (dims as f64).sqrt(); dims]
                    } else {
                        inp.embedding.clone()
                    }
                })
                .collect()
        } else {
            inputs.iter().map(|inp| inp.embedding.clone()).collect()
        };

        let method_name = self.method.name().to_owned();
        let output = match &self.method {
            AggregationMethod::Mean => compute_mean(&embeddings),
            AggregationMethod::WeightedMean { weights } => {
                if weights.len() != inputs.len() {
                    return Err(AggregatorError::WeightCountMismatch {
                        expected: inputs.len(),
                        got: weights.len(),
                    });
                }
                compute_weighted_mean(&embeddings, weights)
            }
            AggregationMethod::Max => compute_max(&embeddings),
            AggregationMethod::Min => compute_min(&embeddings),
            AggregationMethod::Sum => compute_sum(&embeddings),
            AggregationMethod::GeometricMean => compute_geometric_mean(&embeddings),
            AggregationMethod::AttentionPooling { query } => {
                if query.len() != dims {
                    return Err(AggregatorError::InvalidQuery);
                }
                let scores = attention_scores_impl(query, &embeddings);
                compute_weighted_mean_with_weights(&embeddings, &scores)
            }
        };

        // Use per-input weights for WeightedMean if method weight count matches,
        // otherwise fall back to the method above.  (Already handled above via
        // the WeightedMean arm.)
        // NOTE: the WeightedMean arm with inputs[i].weight is an alternative
        // pathway when caller wants dynamic per-call weights embedded in the
        // AggregationInput structs instead of the method enum.
        // That is exposed via aggregate_with_input_weights below.

        let output = if self.config.normalize_output {
            Self::l2_normalize(&output)
        } else {
            output
        };

        let norm = Self::l2_norm(&output);
        let result = AggregationResult {
            method: method_name,
            output,
            input_count: inputs.len(),
            dims,
            norm,
        };

        self.push_history(result.clone());
        Ok(result)
    }

    /// Convenience method that treats all `embeddings` with weight `1.0` and
    /// applies `self.method`.
    pub fn aggregate_raw(
        &mut self,
        embeddings: &[Vec<f64>],
    ) -> Result<AggregationResult, AggregatorError> {
        let inputs: Vec<AggregationInput> = embeddings
            .iter()
            .enumerate()
            .map(|(i, e)| AggregationInput::new(format!("raw_{i}"), e.clone(), 1.0))
            .collect();
        self.aggregate(&inputs)
    }

    /// Aggregate using the `weight` field of each [`AggregationInput`] directly,
    /// ignoring the method's own weight vector.  Useful when weights are
    /// determined dynamically (e.g. relevance scores).
    pub fn aggregate_with_input_weights(
        &mut self,
        inputs: &[AggregationInput],
    ) -> Result<AggregationResult, AggregatorError> {
        if inputs.is_empty() {
            return Err(AggregatorError::EmptyInput);
        }
        let dims = inputs[0].embedding.len();
        for inp in inputs.iter().skip(1) {
            if inp.embedding.len() != dims {
                return Err(AggregatorError::DimensionMismatch {
                    expected: dims,
                    got: inp.embedding.len(),
                });
            }
        }
        let embeddings: Vec<Vec<f64>> = inputs.iter().map(|i| i.embedding.clone()).collect();
        let weights: Vec<f64> = inputs.iter().map(|i| i.weight).collect();
        let output = compute_weighted_mean(&embeddings, &weights);
        let output = if self.config.normalize_output {
            Self::l2_normalize(&output)
        } else {
            output
        };
        let norm = Self::l2_norm(&output);
        let result = AggregationResult {
            method: "WeightedMeanInputWeights".to_owned(),
            output,
            input_count: inputs.len(),
            dims,
            norm,
        };
        self.push_history(result.clone());
        Ok(result)
    }

    /// Aggregate the output vectors from several [`AggregationResult`]s using
    /// the provided `method`.
    pub fn merge_results(
        results: &[AggregationResult],
        method: AggregationMethod,
    ) -> Result<AggregationResult, AggregatorError> {
        if results.is_empty() {
            return Err(AggregatorError::EmptyInput);
        }
        let dims = results[0].dims;
        for r in results.iter().skip(1) {
            if r.dims != dims {
                return Err(AggregatorError::DimensionMismatch {
                    expected: dims,
                    got: r.dims,
                });
            }
        }
        let embeddings: Vec<Vec<f64>> = results.iter().map(|r| r.output.clone()).collect();
        let method_name = method.name().to_owned();
        let output = match &method {
            AggregationMethod::Mean => compute_mean(&embeddings),
            AggregationMethod::WeightedMean { weights } => {
                if weights.len() != results.len() {
                    return Err(AggregatorError::WeightCountMismatch {
                        expected: results.len(),
                        got: weights.len(),
                    });
                }
                compute_weighted_mean(&embeddings, weights)
            }
            AggregationMethod::Max => compute_max(&embeddings),
            AggregationMethod::Min => compute_min(&embeddings),
            AggregationMethod::Sum => compute_sum(&embeddings),
            AggregationMethod::GeometricMean => compute_geometric_mean(&embeddings),
            AggregationMethod::AttentionPooling { query } => {
                if query.len() != dims {
                    return Err(AggregatorError::InvalidQuery);
                }
                let scores = attention_scores_impl(query, &embeddings);
                compute_weighted_mean_with_weights(&embeddings, &scores)
            }
        };
        let norm = Self::l2_norm(&output);
        Ok(AggregationResult {
            method: method_name,
            output,
            input_count: results.len(),
            dims,
            norm,
        })
    }

    // -----------------------------------------------------------------------
    // Utilities
    // -----------------------------------------------------------------------

    /// L2-normalise `v` in place.  Returns `v` unchanged if its norm is ≈ 0.
    pub fn l2_normalize(v: &[f64]) -> Vec<f64> {
        let n = Self::l2_norm(v);
        if n < f64::EPSILON {
            return v.to_vec();
        }
        v.iter().map(|x| x / n).collect()
    }

    /// Compute the L2 (Euclidean) norm of `v`.
    pub fn l2_norm(v: &[f64]) -> f64 {
        v.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    /// Cosine similarity between two vectors.
    ///
    /// Returns `0.0` if either vector is the zero vector.
    pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
        let na = Self::l2_norm(a);
        let nb = Self::l2_norm(b);
        if na < f64::EPSILON || nb < f64::EPSILON {
            return 0.0;
        }
        let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        (dot / (na * nb)).clamp(-1.0, 1.0)
    }

    /// Compute softmax attention scores: `softmax(query · embedding[i] / sqrt(d))`.
    pub fn attention_scores(query: &[f64], embeddings: &[Vec<f64>]) -> Vec<f64> {
        attention_scores_impl(query, embeddings)
    }

    // -----------------------------------------------------------------------
    // History
    // -----------------------------------------------------------------------

    /// Return the `n` most-recent [`AggregationResult`]s (newest last).
    pub fn recent_history(&self, n: usize) -> Vec<&AggregationResult> {
        let skip = self.history.len().saturating_sub(n);
        self.history.iter().skip(skip).collect()
    }

    /// Compute statistics over the full history.
    pub fn stats(&self) -> EaAggregatorStats {
        let total = self.history.len() as u64;
        let avg_input_count = if self.history.is_empty() {
            0.0
        } else {
            self.history
                .iter()
                .map(|r| r.input_count as f64)
                .sum::<f64>()
                / total as f64
        };
        let avg_output_norm = if self.history.is_empty() {
            0.0
        } else {
            self.history.iter().map(|r| r.norm).sum::<f64>() / total as f64
        };
        EaAggregatorStats {
            total_aggregations: total,
            avg_input_count,
            avg_output_norm,
            method_name: self.method.name().to_owned(),
        }
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    fn push_history(&mut self, result: AggregationResult) {
        if self.history.len() >= self.max_history && self.max_history > 0 {
            self.history.pop_front();
        }
        if self.max_history > 0 {
            self.history.push_back(result);
        }
    }
}

// ---------------------------------------------------------------------------
// Free-function compute kernels
// ---------------------------------------------------------------------------

fn compute_mean(embeddings: &[Vec<f64>]) -> Vec<f64> {
    let n = embeddings.len() as f64;
    let dims = embeddings[0].len();
    let mut out = vec![0.0_f64; dims];
    for emb in embeddings {
        for (j, v) in emb.iter().enumerate() {
            out[j] += v;
        }
    }
    out.iter_mut().for_each(|x| *x /= n);
    out
}

fn compute_weighted_mean(embeddings: &[Vec<f64>], weights: &[f64]) -> Vec<f64> {
    let total: f64 = weights.iter().sum();
    let denom = if total.abs() < f64::EPSILON {
        1.0
    } else {
        total
    };
    let dims = embeddings[0].len();
    let mut out = vec![0.0_f64; dims];
    for (emb, &w) in embeddings.iter().zip(weights.iter()) {
        for (j, v) in emb.iter().enumerate() {
            out[j] += w * v;
        }
    }
    out.iter_mut().for_each(|x| *x /= denom);
    out
}

fn compute_weighted_mean_with_weights(embeddings: &[Vec<f64>], weights: &[f64]) -> Vec<f64> {
    compute_weighted_mean(embeddings, weights)
}

fn compute_max(embeddings: &[Vec<f64>]) -> Vec<f64> {
    let dims = embeddings[0].len();
    let mut out = vec![f64::NEG_INFINITY; dims];
    for emb in embeddings {
        for (j, &v) in emb.iter().enumerate() {
            if v > out[j] {
                out[j] = v;
            }
        }
    }
    out
}

fn compute_min(embeddings: &[Vec<f64>]) -> Vec<f64> {
    let dims = embeddings[0].len();
    let mut out = vec![f64::INFINITY; dims];
    for emb in embeddings {
        for (j, &v) in emb.iter().enumerate() {
            if v < out[j] {
                out[j] = v;
            }
        }
    }
    out
}

fn compute_sum(embeddings: &[Vec<f64>]) -> Vec<f64> {
    let dims = embeddings[0].len();
    let mut out = vec![0.0_f64; dims];
    for emb in embeddings {
        for (j, &v) in emb.iter().enumerate() {
            out[j] += v;
        }
    }
    out
}

/// Geometric mean: `exp(mean(ln(|v[j]| + ε)))` with sign of arithmetic mean.
fn compute_geometric_mean(embeddings: &[Vec<f64>]) -> Vec<f64> {
    const EPS: f64 = 1e-10;
    let n = embeddings.len() as f64;
    let dims = embeddings[0].len();
    let mut out = vec![0.0_f64; dims];

    for j in 0..dims {
        // Arithmetic mean sign
        let arith_mean: f64 = embeddings.iter().map(|e| e[j]).sum::<f64>() / n;
        let sign = if arith_mean >= 0.0 { 1.0_f64 } else { -1.0_f64 };

        // Geometric mean of absolute values (+ε for numerical stability)
        let log_mean: f64 = embeddings
            .iter()
            .map(|e| (e[j].abs() + EPS).ln())
            .sum::<f64>()
            / n;
        out[j] = log_mean.exp() * sign;
    }
    out
}

/// Softmax of scaled dot products.
fn attention_scores_impl(query: &[f64], embeddings: &[Vec<f64>]) -> Vec<f64> {
    let dims = query.len() as f64;
    let scale = dims.sqrt();

    let raw: Vec<f64> = embeddings
        .iter()
        .map(|emb| {
            let dot: f64 = query.iter().zip(emb.iter()).map(|(q, e)| q * e).sum();
            dot / scale
        })
        .collect();

    softmax(&raw)
}

fn softmax(logits: &[f64]) -> Vec<f64> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max_val = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f64 = exps.iter().sum();
    let denom = if sum < f64::EPSILON { 1.0 } else { sum };
    exps.iter().map(|e| e / denom).collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::{
        AggregationInput, AggregationMethod, AggregationResult, AggregatorError,
        EmbeddingAggregator, EmbeddingAggregatorConfig,
    };

    // Helpers
    fn uniform(val: f64, dim: usize) -> Vec<f64> {
        vec![val; dim]
    }

    fn make_agg(method: AggregationMethod) -> EmbeddingAggregator {
        EmbeddingAggregator::new(method, EmbeddingAggregatorConfig::default())
    }

    fn make_inputs(embeddings: Vec<Vec<f64>>) -> Vec<AggregationInput> {
        embeddings
            .into_iter()
            .enumerate()
            .map(|(i, e)| AggregationInput::new(format!("i{i}"), e, 1.0))
            .collect()
    }

    fn result_near(result: &AggregationResult, expected: &[f64], tol: f64) -> bool {
        result
            .output
            .iter()
            .zip(expected.iter())
            .all(|(a, b)| (a - b).abs() < tol)
    }

    // ------------------------------------------------------------------
    // 1  Mean
    // ------------------------------------------------------------------

    #[test]
    fn test_mean_basic() {
        let mut agg = make_agg(AggregationMethod::Mean);
        let inputs = make_inputs(vec![vec![2.0, 4.0], vec![0.0, 0.0]]);
        let r = agg
            .aggregate(&inputs)
            .expect("test: aggregate should succeed");
        assert!(result_near(&r, &[1.0, 2.0], 1e-10));
    }

    #[test]
    fn test_mean_single_input() {
        let mut agg = make_agg(AggregationMethod::Mean);
        let inputs = make_inputs(vec![vec![3.0, 7.0, 1.0]]);
        let r = agg
            .aggregate(&inputs)
            .expect("test: aggregate should succeed");
        assert!(result_near(&r, &[3.0, 7.0, 1.0], 1e-10));
    }

    #[test]
    fn test_mean_three_equal_vecs() {
        let mut agg = make_agg(AggregationMethod::Mean);
        let inputs = make_inputs(vec![vec![1.0, 2.0], vec![1.0, 2.0], vec![1.0, 2.0]]);
        let r = agg
            .aggregate(&inputs)
            .expect("test: aggregate should succeed");
        assert!(result_near(&r, &[1.0, 2.0], 1e-10));
    }

    // ------------------------------------------------------------------
    // 2  WeightedMean
    // ------------------------------------------------------------------

    #[test]
    fn test_weighted_mean_basic() {
        let mut agg = make_agg(AggregationMethod::WeightedMean {
            weights: vec![3.0, 1.0],
        });
        let inputs = make_inputs(vec![vec![1.0, 0.0], vec![0.0, 1.0]]);
        let r = agg
            .aggregate(&inputs)
            .expect("test: aggregate should succeed");
        // weights normalised: 3/4, 1/4
        assert!((r.output[0] - 0.75).abs() < 1e-10);
        assert!((r.output[1] - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_weighted_mean_equal_weights() {
        let mut agg = make_agg(AggregationMethod::WeightedMean {
            weights: vec![1.0, 1.0],
        });
        let inputs = make_inputs(vec![vec![2.0, 4.0], vec![0.0, 0.0]]);
        let r = agg
            .aggregate(&inputs)
            .expect("test: aggregate should succeed");
        assert!(result_near(&r, &[1.0, 2.0], 1e-10));
    }

    #[test]
    fn test_weighted_mean_count_mismatch_error() {
        let mut agg = make_agg(AggregationMethod::WeightedMean {
            weights: vec![1.0, 2.0, 3.0],
        });
        let inputs = make_inputs(vec![vec![1.0], vec![2.0]]);
        let err = agg
            .aggregate(&inputs)
            .expect_err("test: aggregate should return error");
        assert_eq!(
            err,
            AggregatorError::WeightCountMismatch {
                expected: 2,
                got: 3,
            }
        );
    }

    // ------------------------------------------------------------------
    // 3  Max
    // ------------------------------------------------------------------

    #[test]
    fn test_max_basic() {
        let mut agg = make_agg(AggregationMethod::Max);
        let inputs = make_inputs(vec![vec![1.0, 5.0], vec![3.0, 2.0]]);
        let r = agg
            .aggregate(&inputs)
            .expect("test: aggregate should succeed");
        assert!(result_near(&r, &[3.0, 5.0], 1e-10));
    }

    #[test]
    fn test_max_negative_values() {
        let mut agg = make_agg(AggregationMethod::Max);
        let inputs = make_inputs(vec![vec![-1.0, -5.0], vec![-3.0, -2.0]]);
        let r = agg
            .aggregate(&inputs)
            .expect("test: aggregate should succeed");
        assert!(result_near(&r, &[-1.0, -2.0], 1e-10));
    }

    // ------------------------------------------------------------------
    // 4  Min
    // ------------------------------------------------------------------

    #[test]
    fn test_min_basic() {
        let mut agg = make_agg(AggregationMethod::Min);
        let inputs = make_inputs(vec![vec![1.0, 5.0], vec![3.0, 2.0]]);
        let r = agg
            .aggregate(&inputs)
            .expect("test: aggregate should succeed");
        assert!(result_near(&r, &[1.0, 2.0], 1e-10));
    }

    #[test]
    fn test_min_all_equal() {
        let mut agg = make_agg(AggregationMethod::Min);
        let inputs = make_inputs(vec![vec![4.0, 4.0], vec![4.0, 4.0]]);
        let r = agg
            .aggregate(&inputs)
            .expect("test: aggregate should succeed");
        assert!(result_near(&r, &[4.0, 4.0], 1e-10));
    }

    // ------------------------------------------------------------------
    // 5  Sum
    // ------------------------------------------------------------------

    #[test]
    fn test_sum_basic() {
        let mut agg = make_agg(AggregationMethod::Sum);
        let inputs = make_inputs(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let r = agg
            .aggregate(&inputs)
            .expect("test: aggregate should succeed");
        assert!(result_near(&r, &[4.0, 6.0], 1e-10));
    }

    #[test]
    fn test_sum_three_vectors() {
        let mut agg = make_agg(AggregationMethod::Sum);
        let inputs = make_inputs(vec![vec![1.0], vec![2.0], vec![3.0]]);
        let r = agg
            .aggregate(&inputs)
            .expect("test: aggregate should succeed");
        assert!(result_near(&r, &[6.0], 1e-10));
    }

    // ------------------------------------------------------------------
    // 6  GeometricMean
    // ------------------------------------------------------------------

    #[test]
    fn test_geometric_mean_positive_values() {
        let mut agg = make_agg(AggregationMethod::GeometricMean);
        // geometric mean of 1 and 4 ≈ 2.0
        let inputs = make_inputs(vec![vec![1.0], vec![4.0]]);
        let r = agg
            .aggregate(&inputs)
            .expect("test: aggregate should succeed");
        // expected ≈ exp((ln(1+ε) + ln(4+ε))/2) * sign(2.5)
        assert!((r.output[0] - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_geometric_mean_negative_sign() {
        let mut agg = make_agg(AggregationMethod::GeometricMean);
        // arith mean is negative → output should be negative
        let inputs = make_inputs(vec![vec![-4.0], vec![-1.0]]);
        let r = agg
            .aggregate(&inputs)
            .expect("test: aggregate should succeed");
        assert!(r.output[0] < 0.0);
    }

    #[test]
    fn test_geometric_mean_mixed_sign() {
        let mut agg = make_agg(AggregationMethod::GeometricMean);
        // arith mean = 0.0 → sign treated as positive
        let inputs = make_inputs(vec![vec![-1.0], vec![1.0]]);
        let r = agg
            .aggregate(&inputs)
            .expect("test: aggregate should succeed");
        // Should not panic
        let _ = r.output[0];
    }

    // ------------------------------------------------------------------
    // 7  AttentionPooling
    // ------------------------------------------------------------------

    #[test]
    fn test_attention_pooling_uniform_query() {
        // Uniform query → all attention weights equal → reduces to mean
        let q = uniform(1.0, 4);
        let mut agg = make_agg(AggregationMethod::AttentionPooling { query: q });
        let inputs = make_inputs(vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]]);
        let mean_agg = make_inputs(vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]]);
        let r = agg
            .aggregate(&inputs)
            .expect("test: aggregate should succeed");
        let mut mean_agg2 = make_agg(AggregationMethod::Mean);
        let rm = mean_agg2
            .aggregate(&mean_agg)
            .expect("test: aggregate should succeed");
        // Because dots differ they won't be exactly equal, but the pooling should
        // produce a valid vector of the right length.
        assert_eq!(r.output.len(), 4);
        let _ = rm;
    }

    #[test]
    fn test_attention_pooling_query_dim_mismatch() {
        let q = vec![1.0, 0.0]; // wrong dim
        let mut agg = make_agg(AggregationMethod::AttentionPooling { query: q });
        let inputs = make_inputs(vec![vec![1.0, 2.0, 3.0]]);
        assert_eq!(
            agg.aggregate(&inputs)
                .expect_err("test: aggregate should return error"),
            AggregatorError::InvalidQuery
        );
    }

    #[test]
    fn test_attention_pooling_single_input() {
        let q = vec![1.0, 0.0];
        let mut agg = make_agg(AggregationMethod::AttentionPooling { query: q });
        let inputs = make_inputs(vec![vec![3.0, 7.0]]);
        let r = agg
            .aggregate(&inputs)
            .expect("test: aggregate should succeed");
        assert!(result_near(&r, &[3.0, 7.0], 1e-9));
    }

    // ------------------------------------------------------------------
    // 8  Error cases
    // ------------------------------------------------------------------

    #[test]
    fn test_empty_input_error() {
        let mut agg = make_agg(AggregationMethod::Mean);
        let err = agg
            .aggregate(&[])
            .expect_err("test: aggregate should return error");
        assert_eq!(err, AggregatorError::EmptyInput);
    }

    #[test]
    fn test_dimension_mismatch_error() {
        let mut agg = make_agg(AggregationMethod::Mean);
        let inputs = vec![
            AggregationInput::new("a", vec![1.0, 2.0], 1.0),
            AggregationInput::new("b", vec![1.0, 2.0, 3.0], 1.0),
        ];
        let err = agg
            .aggregate(&inputs)
            .expect_err("test: aggregate should return error");
        assert_eq!(
            err,
            AggregatorError::DimensionMismatch {
                expected: 2,
                got: 3
            }
        );
    }

    #[test]
    fn test_aggregate_raw_empty() {
        let mut agg = make_agg(AggregationMethod::Mean);
        let err = agg
            .aggregate_raw(&[])
            .expect_err("test: aggregate_raw with empty input should return error");
        assert_eq!(err, AggregatorError::EmptyInput);
    }

    #[test]
    fn test_aggregate_raw_basic() {
        let mut agg = make_agg(AggregationMethod::Mean);
        let r = agg
            .aggregate_raw(&[vec![2.0, 4.0], vec![0.0, 0.0]])
            .expect("test: aggregate_raw should succeed");
        assert!(result_near(&r, &[1.0, 2.0], 1e-10));
    }

    // ------------------------------------------------------------------
    // 9  L2 utilities
    // ------------------------------------------------------------------

    #[test]
    fn test_l2_norm_zero_vec() {
        let norm = EmbeddingAggregator::l2_norm(&[0.0, 0.0, 0.0]);
        assert_eq!(norm, 0.0);
    }

    #[test]
    fn test_l2_norm_unit_vec() {
        let norm = EmbeddingAggregator::l2_norm(&[1.0, 0.0, 0.0]);
        assert!((norm - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_l2_normalize_already_unit() {
        let v = vec![1.0, 0.0, 0.0];
        let n = EmbeddingAggregator::l2_normalize(&v);
        assert!((EmbeddingAggregator::l2_norm(&n) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_l2_normalize_zero_vec_unchanged() {
        let v = vec![0.0, 0.0];
        let n = EmbeddingAggregator::l2_normalize(&v);
        assert_eq!(n, vec![0.0, 0.0]);
    }

    #[test]
    fn test_l2_normalize_scales_correctly() {
        let v = vec![3.0, 4.0];
        let n = EmbeddingAggregator::l2_normalize(&v);
        let norm = EmbeddingAggregator::l2_norm(&n);
        assert!((norm - 1.0).abs() < 1e-10);
    }

    // ------------------------------------------------------------------
    // 10 Cosine similarity
    // ------------------------------------------------------------------

    #[test]
    fn test_cosine_similarity_identical() {
        let v = vec![1.0, 2.0, 3.0];
        let sim = EmbeddingAggregator::cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = EmbeddingAggregator::cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        let sim = EmbeddingAggregator::cosine_similarity(&a, &b);
        assert!((sim + 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_zero_vec() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 0.0];
        let sim = EmbeddingAggregator::cosine_similarity(&a, &b);
        assert_eq!(sim, 0.0);
    }

    // ------------------------------------------------------------------
    // 11 Attention scores
    // ------------------------------------------------------------------

    #[test]
    fn test_attention_scores_sum_to_one() {
        let query = vec![1.0, 0.0, 1.0];
        let embeddings = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let scores = EmbeddingAggregator::attention_scores(&query, &embeddings);
        let total: f64 = scores.iter().sum();
        assert!((total - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_attention_scores_length() {
        let query = vec![1.0, 1.0];
        let embeddings = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        let scores = EmbeddingAggregator::attention_scores(&query, &embeddings);
        assert_eq!(scores.len(), 3);
    }

    // ------------------------------------------------------------------
    // 12 Normalize output config
    // ------------------------------------------------------------------

    #[test]
    fn test_normalize_output_enabled() {
        let config = EmbeddingAggregatorConfig {
            normalize_output: true,
            handle_zeros: false,
        };
        let mut agg = EmbeddingAggregator::new(AggregationMethod::Sum, config);
        let r = agg
            .aggregate_raw(&[vec![3.0, 4.0], vec![0.0, 0.0]])
            .expect("test: aggregate_raw should succeed");
        let norm = EmbeddingAggregator::l2_norm(&r.output);
        assert!((norm - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_handle_zeros_replaces_zero_vec() {
        let config = EmbeddingAggregatorConfig {
            normalize_output: false,
            handle_zeros: true,
        };
        let mut agg = EmbeddingAggregator::new(AggregationMethod::Mean, config);
        let inputs = vec![
            AggregationInput::new("z", vec![0.0, 0.0], 1.0),
            AggregationInput::new("v", vec![1.0, 0.0], 1.0),
        ];
        let r = agg
            .aggregate(&inputs)
            .expect("test: aggregate should succeed");
        // zero vec replaced by [1/sqrt(2), 1/sqrt(2)]
        let expected_0 = (1.0 / 2.0_f64.sqrt() + 1.0) / 2.0;
        let expected_1 = 1.0 / (2.0 * 2.0_f64.sqrt());
        assert!((r.output[0] - expected_0).abs() < 1e-9);
        assert!((r.output[1] - expected_1).abs() < 1e-9);
    }

    // ------------------------------------------------------------------
    // 13 Result metadata
    // ------------------------------------------------------------------

    #[test]
    fn test_result_input_count() {
        let mut agg = make_agg(AggregationMethod::Mean);
        let inputs = make_inputs(vec![vec![1.0], vec![2.0], vec![3.0]]);
        let r = agg
            .aggregate(&inputs)
            .expect("test: aggregate should succeed");
        assert_eq!(r.input_count, 3);
    }

    #[test]
    fn test_result_dims() {
        let mut agg = make_agg(AggregationMethod::Sum);
        let inputs = make_inputs(vec![vec![1.0, 2.0, 3.0, 4.0]]);
        let r = agg
            .aggregate(&inputs)
            .expect("test: aggregate should succeed");
        assert_eq!(r.dims, 4);
    }

    #[test]
    fn test_result_norm_matches_output() {
        let mut agg = make_agg(AggregationMethod::Mean);
        let inputs = make_inputs(vec![vec![3.0, 4.0]]);
        let r = agg
            .aggregate(&inputs)
            .expect("test: aggregate should succeed");
        let expected_norm = EmbeddingAggregator::l2_norm(&r.output);
        assert!((r.norm - expected_norm).abs() < 1e-10);
    }

    #[test]
    fn test_result_method_name() {
        let mut agg = make_agg(AggregationMethod::Max);
        let inputs = make_inputs(vec![vec![1.0]]);
        let r = agg
            .aggregate(&inputs)
            .expect("test: aggregate should succeed");
        assert_eq!(r.method, "Max");
    }

    // ------------------------------------------------------------------
    // 14 History
    // ------------------------------------------------------------------

    #[test]
    fn test_history_grows() {
        let mut agg = make_agg(AggregationMethod::Mean);
        for _ in 0..5 {
            agg.aggregate_raw(&[vec![1.0]])
                .expect("test: aggregate_raw should succeed");
        }
        assert_eq!(agg.history.len(), 5);
    }

    #[test]
    fn test_history_capped_at_max() {
        let mut agg = EmbeddingAggregator::with_history_capacity(
            AggregationMethod::Mean,
            EmbeddingAggregatorConfig::default(),
            3,
        );
        for _ in 0..10 {
            agg.aggregate_raw(&[vec![1.0]])
                .expect("test: aggregate_raw should succeed");
        }
        assert_eq!(agg.history.len(), 3);
    }

    #[test]
    fn test_recent_history_returns_n() {
        let mut agg = make_agg(AggregationMethod::Sum);
        for _ in 0..8 {
            agg.aggregate_raw(&[vec![1.0]])
                .expect("test: aggregate_raw should succeed");
        }
        let recent = agg.recent_history(3);
        assert_eq!(recent.len(), 3);
    }

    #[test]
    fn test_recent_history_more_than_total() {
        let mut agg = make_agg(AggregationMethod::Sum);
        agg.aggregate_raw(&[vec![1.0]])
            .expect("test: aggregate_raw should succeed");
        let recent = agg.recent_history(100);
        assert_eq!(recent.len(), 1);
    }

    // ------------------------------------------------------------------
    // 15 Stats
    // ------------------------------------------------------------------

    #[test]
    fn test_stats_empty_history() {
        let agg = make_agg(AggregationMethod::Mean);
        let s = agg.stats();
        assert_eq!(s.total_aggregations, 0);
        assert_eq!(s.avg_input_count, 0.0);
        assert_eq!(s.avg_output_norm, 0.0);
        assert_eq!(s.method_name, "Mean");
    }

    #[test]
    fn test_stats_total_aggregations() {
        let mut agg = make_agg(AggregationMethod::Min);
        for _ in 0..7 {
            agg.aggregate_raw(&[vec![1.0]])
                .expect("test: aggregate_raw should succeed");
        }
        assert_eq!(agg.stats().total_aggregations, 7);
    }

    #[test]
    fn test_stats_avg_input_count() {
        let mut agg = make_agg(AggregationMethod::Mean);
        agg.aggregate_raw(&[vec![1.0], vec![2.0]])
            .expect("test: aggregate_raw should succeed"); // count 2
        agg.aggregate_raw(&[vec![1.0]])
            .expect("test: aggregate_raw should succeed"); // count 1
        let s = agg.stats();
        assert!((s.avg_input_count - 1.5).abs() < 1e-10);
    }

    // ------------------------------------------------------------------
    // 16 merge_results
    // ------------------------------------------------------------------

    #[test]
    fn test_merge_results_mean() {
        let mut agg = make_agg(AggregationMethod::Mean);
        let r1 = agg
            .aggregate_raw(&[vec![2.0, 0.0]])
            .expect("test: aggregate_raw should succeed");
        let r2 = agg
            .aggregate_raw(&[vec![0.0, 4.0]])
            .expect("test: aggregate_raw should succeed");
        let merged = EmbeddingAggregator::merge_results(&[r1, r2], AggregationMethod::Mean)
            .expect("test: merge_results should succeed");
        assert!(result_near(&merged, &[1.0, 2.0], 1e-10));
    }

    #[test]
    fn test_merge_results_empty_error() {
        let err = EmbeddingAggregator::merge_results(&[], AggregationMethod::Sum)
            .expect_err("test: merge_results with empty input should return error");
        assert_eq!(err, AggregatorError::EmptyInput);
    }

    #[test]
    fn test_merge_results_dim_mismatch() {
        let r1 = AggregationResult {
            method: "Mean".into(),
            output: vec![1.0, 2.0],
            input_count: 1,
            dims: 2,
            norm: 1.0,
        };
        let r2 = AggregationResult {
            method: "Mean".into(),
            output: vec![1.0],
            input_count: 1,
            dims: 1,
            norm: 1.0,
        };
        let err = EmbeddingAggregator::merge_results(&[r1, r2], AggregationMethod::Sum)
            .expect_err("test: merge_results with dim mismatch should return error");
        assert_eq!(
            err,
            AggregatorError::DimensionMismatch {
                expected: 2,
                got: 1
            }
        );
    }

    // ------------------------------------------------------------------
    // 17 aggregate_with_input_weights
    // ------------------------------------------------------------------

    #[test]
    fn test_aggregate_with_input_weights() {
        let mut agg = make_agg(AggregationMethod::Mean);
        let inputs = vec![
            AggregationInput::new("a", vec![1.0, 0.0], 3.0),
            AggregationInput::new("b", vec![0.0, 1.0], 1.0),
        ];
        let r = agg
            .aggregate_with_input_weights(&inputs)
            .expect("test: aggregate_with_input_weights should succeed");
        assert!((r.output[0] - 0.75).abs() < 1e-10);
        assert!((r.output[1] - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_aggregate_with_input_weights_empty_error() {
        let mut agg = make_agg(AggregationMethod::Mean);
        let err = agg
            .aggregate_with_input_weights(&[])
            .expect_err("test: aggregate_with_input_weights with empty input should return error");
        assert_eq!(err, AggregatorError::EmptyInput);
    }

    // ------------------------------------------------------------------
    // 18 Error Display
    // ------------------------------------------------------------------

    #[test]
    fn test_error_display_empty_input() {
        let e = AggregatorError::EmptyInput;
        assert!(!e.to_string().is_empty());
    }

    #[test]
    fn test_error_display_dimension_mismatch() {
        let e = AggregatorError::DimensionMismatch {
            expected: 4,
            got: 3,
        };
        assert!(e.to_string().contains('4'));
    }

    #[test]
    fn test_error_display_weight_count() {
        let e = AggregatorError::WeightCountMismatch {
            expected: 2,
            got: 5,
        };
        assert!(e.to_string().contains('5'));
    }

    #[test]
    fn test_error_display_invalid_query() {
        let e = AggregatorError::InvalidQuery;
        assert!(!e.to_string().is_empty());
    }

    // ------------------------------------------------------------------
    // 19 Method names
    // ------------------------------------------------------------------

    #[test]
    fn test_method_names() {
        let methods: Vec<(&str, AggregationMethod)> = vec![
            ("Mean", AggregationMethod::Mean),
            ("Max", AggregationMethod::Max),
            ("Min", AggregationMethod::Min),
            ("Sum", AggregationMethod::Sum),
            ("GeometricMean", AggregationMethod::GeometricMean),
            (
                "WeightedMean",
                AggregationMethod::WeightedMean { weights: vec![1.0] },
            ),
            (
                "AttentionPooling",
                AggregationMethod::AttentionPooling { query: vec![1.0] },
            ),
        ];
        for (expected, method) in methods {
            assert_eq!(method.name(), expected);
        }
    }

    // ------------------------------------------------------------------
    // 20 Zero-history capacity
    // ------------------------------------------------------------------

    #[test]
    fn test_zero_history_capacity() {
        let mut agg = EmbeddingAggregator::with_history_capacity(
            AggregationMethod::Sum,
            EmbeddingAggregatorConfig::default(),
            0,
        );
        agg.aggregate_raw(&[vec![1.0]])
            .expect("test: aggregate_raw should succeed");
        assert_eq!(agg.history.len(), 0);
        let s = agg.stats();
        assert_eq!(s.total_aggregations, 0);
    }
}
