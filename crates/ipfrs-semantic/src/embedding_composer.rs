//! Embedding Composer — compose multiple embeddings into a single representation
//! using various late-fusion strategies.
//!
//! Supports Concatenate, Average, WeightedAverage, MaxPooling, and HadamardProduct
//! fusion strategies for building multi-modal or multi-view late-fusion pipelines.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// CompositionStrategy
// ---------------------------------------------------------------------------

/// The fusion strategy to apply when composing multiple embeddings.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CompositionStrategy {
    /// Concatenate all vectors in order.  Output dim = sum of all input dims.
    Concatenate,
    /// Element-wise arithmetic mean.  All inputs must share the same dimension.
    Average,
    /// Weighted element-wise average.  All inputs must share the same dimension.
    /// Each [`EmbeddingInput::weight`] is used; weights are normalised to sum to 1.
    WeightedAverage,
    /// Element-wise maximum.  All inputs must share the same dimension.
    MaxPooling,
    /// Element-wise product (Hadamard product).  All inputs must share the same dimension.
    HadamardProduct,
}

// ---------------------------------------------------------------------------
// EmbeddingInput
// ---------------------------------------------------------------------------

/// A single embedding source together with its metadata.
#[derive(Debug, Clone)]
pub struct EmbeddingInput {
    /// Opaque identifier for the originating source (e.g. modality id, model id).
    pub source_id: u64,
    /// The embedding vector.
    pub vector: Vec<f32>,
    /// Weight used by [`CompositionStrategy::WeightedAverage`].
    /// Ignored by other strategies.
    pub weight: f64,
}

impl EmbeddingInput {
    /// Convenience constructor.
    pub fn new(source_id: u64, vector: Vec<f32>, weight: f64) -> Self {
        Self {
            source_id,
            vector,
            weight,
        }
    }
}

// ---------------------------------------------------------------------------
// CompositionResult
// ---------------------------------------------------------------------------

/// The outcome of a composition operation.
#[derive(Debug, Clone)]
pub struct CompositionResult {
    /// The composed embedding vector.
    pub composed: Vec<f32>,
    /// The strategy that produced this result.
    pub strategy: CompositionStrategy,
    /// Number of input embeddings that were fused.
    pub input_count: usize,
    /// Dimensionality of the composed vector.
    pub output_dim: usize,
}

impl CompositionResult {
    /// Compute the L2 (Euclidean) norm of the composed vector.
    pub fn l2_norm(&self) -> f32 {
        self.composed.iter().map(|v| v * v).sum::<f32>().sqrt()
    }

    /// Return an L2-normalised copy of the composed vector.
    /// If the norm is zero (or very close to zero), returns a zero vector of the same length.
    pub fn normalize(&self) -> Vec<f32> {
        let norm = self.l2_norm();
        if norm == 0.0 {
            return vec![0.0_f32; self.composed.len()];
        }
        self.composed.iter().map(|v| v / norm).collect()
    }
}

// ---------------------------------------------------------------------------
// ComposerStats
// ---------------------------------------------------------------------------

/// Accumulated statistics for an [`EmbeddingComposer`] instance.
#[derive(Debug, Clone, Default)]
pub struct ComposerStats {
    /// Total number of successful composition operations.
    pub total_composed: u64,
    /// Per-strategy success counts.
    pub strategy_counts: HashMap<CompositionStrategy, u64>,
}

impl ComposerStats {
    /// Returns the strategy that has been used most often, or `None` if no
    /// compositions have been performed yet.
    pub fn most_used_strategy(&self) -> Option<CompositionStrategy> {
        self.strategy_counts
            .iter()
            .max_by_key(|(_, count)| *count)
            .map(|(strategy, _)| *strategy)
    }
}

// ---------------------------------------------------------------------------
// EmbeddingComposer
// ---------------------------------------------------------------------------

/// Composes multiple embeddings into a single representation using a chosen
/// [`CompositionStrategy`].
///
/// # Example
///
/// ```rust
/// use ipfrs_semantic::embedding_composer::{
///     EmbeddingComposer, EmbeddingInput, CompositionStrategy,
/// };
///
/// let mut composer = EmbeddingComposer::new();
///
/// let inputs = vec![
///     EmbeddingInput::new(1, vec![1.0, 2.0], 1.0),
///     EmbeddingInput::new(2, vec![3.0, 4.0], 1.0),
/// ];
///
/// let result = composer
///     .compose(&inputs, CompositionStrategy::Concatenate)
///     .unwrap();
/// assert_eq!(result.composed, vec![1.0, 2.0, 3.0, 4.0]);
/// ```
#[derive(Debug, Default)]
pub struct EmbeddingComposer {
    /// Accumulated statistics across all compose calls.
    pub stats: ComposerStats,
}

impl EmbeddingComposer {
    /// Create a new composer with zeroed statistics.
    pub fn new() -> Self {
        Self {
            stats: ComposerStats::default(),
        }
    }

    /// Compose `inputs` using `strategy` and return the result.
    ///
    /// # Errors
    ///
    /// - Returns `Err("no inputs")` when `inputs` is empty.
    /// - Returns `Err("dimension mismatch")` when a strategy that requires
    ///   equal dimensions receives inputs of differing dimensionality.
    pub fn compose(
        &mut self,
        inputs: &[EmbeddingInput],
        strategy: CompositionStrategy,
    ) -> Result<CompositionResult, String> {
        if inputs.is_empty() {
            return Err("no inputs".to_string());
        }

        let composed = match strategy {
            CompositionStrategy::Concatenate => Self::apply_concatenate(inputs),
            CompositionStrategy::Average => Self::apply_average(inputs)?,
            CompositionStrategy::WeightedAverage => Self::apply_weighted_average(inputs)?,
            CompositionStrategy::MaxPooling => Self::apply_max_pooling(inputs)?,
            CompositionStrategy::HadamardProduct => Self::apply_hadamard(inputs)?,
        };

        let output_dim = composed.len();
        let input_count = inputs.len();

        // Update stats.
        self.stats.total_composed += 1;
        *self.stats.strategy_counts.entry(strategy).or_insert(0) += 1;

        Ok(CompositionResult {
            composed,
            strategy,
            input_count,
            output_dim,
        })
    }

    /// Compose multiple independent batches in a single call.
    ///
    /// Each element of `batches` is a `(inputs, strategy)` pair.  Results are
    /// returned in the same order as the input batches.
    pub fn batch_compose(
        &mut self,
        batches: Vec<(Vec<EmbeddingInput>, CompositionStrategy)>,
    ) -> Vec<Result<CompositionResult, String>> {
        batches
            .into_iter()
            .map(|(inputs, strategy)| self.compose(&inputs, strategy))
            .collect()
    }

    /// Return a reference to the accumulated statistics.
    pub fn stats(&self) -> &ComposerStats {
        &self.stats
    }

    // -----------------------------------------------------------------------
    // Private strategy implementations
    // -----------------------------------------------------------------------

    fn apply_concatenate(inputs: &[EmbeddingInput]) -> Vec<f32> {
        inputs
            .iter()
            .flat_map(|inp| inp.vector.iter().copied())
            .collect()
    }

    fn check_same_dim(inputs: &[EmbeddingInput]) -> Result<usize, String> {
        let dim = inputs[0].vector.len();
        for inp in inputs.iter().skip(1) {
            if inp.vector.len() != dim {
                return Err("dimension mismatch".to_string());
            }
        }
        Ok(dim)
    }

    fn apply_average(inputs: &[EmbeddingInput]) -> Result<Vec<f32>, String> {
        let dim = Self::check_same_dim(inputs)?;
        let n = inputs.len() as f32;
        let mut result = vec![0.0_f32; dim];
        for inp in inputs {
            for (r, v) in result.iter_mut().zip(inp.vector.iter()) {
                *r += v;
            }
        }
        for r in &mut result {
            *r /= n;
        }
        Ok(result)
    }

    fn apply_weighted_average(inputs: &[EmbeddingInput]) -> Result<Vec<f32>, String> {
        let dim = Self::check_same_dim(inputs)?;

        // Normalise weights: if all weights sum to zero, treat all as equal.
        let weight_sum: f64 = inputs.iter().map(|inp| inp.weight).sum();
        let weights: Vec<f64> = if weight_sum == 0.0 {
            let equal = 1.0 / inputs.len() as f64;
            vec![equal; inputs.len()]
        } else {
            inputs.iter().map(|inp| inp.weight / weight_sum).collect()
        };

        let mut result = vec![0.0_f32; dim];
        for (inp, w) in inputs.iter().zip(weights.iter()) {
            let wf = *w as f32;
            for (r, v) in result.iter_mut().zip(inp.vector.iter()) {
                *r += wf * v;
            }
        }
        Ok(result)
    }

    fn apply_max_pooling(inputs: &[EmbeddingInput]) -> Result<Vec<f32>, String> {
        let dim = Self::check_same_dim(inputs)?;
        let mut result = inputs[0].vector.clone();
        for inp in inputs.iter().skip(1) {
            for (r, v) in result.iter_mut().zip(inp.vector.iter()) {
                if *v > *r {
                    *r = *v;
                }
            }
        }
        // Suppress unused variable warning from `dim` when result is taken from clone.
        let _ = dim;
        Ok(result)
    }

    fn apply_hadamard(inputs: &[EmbeddingInput]) -> Result<Vec<f32>, String> {
        let dim = Self::check_same_dim(inputs)?;
        let mut result = inputs[0].vector.clone();
        for inp in inputs.iter().skip(1) {
            for (r, v) in result.iter_mut().zip(inp.vector.iter()) {
                *r *= v;
            }
        }
        let _ = dim;
        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_input(source_id: u64, vector: Vec<f32>, weight: f64) -> EmbeddingInput {
        EmbeddingInput::new(source_id, vector, weight)
    }

    // -------
    // Concatenate
    // -------

    #[test]
    fn test_concatenate_joins_vectors() {
        let mut c = EmbeddingComposer::new();
        let inputs = vec![
            make_input(1, vec![1.0, 2.0], 1.0),
            make_input(2, vec![3.0, 4.0], 1.0),
        ];
        let res = c
            .compose(&inputs, CompositionStrategy::Concatenate)
            .expect("test: compose concatenate failed");
        assert_eq!(res.composed, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_concatenate_three_inputs() {
        let mut c = EmbeddingComposer::new();
        let inputs = vec![
            make_input(1, vec![1.0], 1.0),
            make_input(2, vec![2.0], 1.0),
            make_input(3, vec![3.0], 1.0),
        ];
        let res = c
            .compose(&inputs, CompositionStrategy::Concatenate)
            .expect("test: compose concatenate three inputs failed");
        assert_eq!(res.composed, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_concatenate_output_dim_is_sum() {
        let mut c = EmbeddingComposer::new();
        let inputs = vec![
            make_input(1, vec![0.0; 3], 1.0),
            make_input(2, vec![0.0; 5], 1.0),
        ];
        let res = c
            .compose(&inputs, CompositionStrategy::Concatenate)
            .expect("test: compose concatenate output dim failed");
        assert_eq!(res.output_dim, 8);
    }

    #[test]
    fn test_concatenate_different_dims_allowed() {
        let mut c = EmbeddingComposer::new();
        let inputs = vec![
            make_input(1, vec![1.0, 2.0, 3.0], 1.0),
            make_input(2, vec![4.0], 1.0),
        ];
        let res = c
            .compose(&inputs, CompositionStrategy::Concatenate)
            .expect("test: compose concatenate different dims failed");
        assert_eq!(res.composed, vec![1.0, 2.0, 3.0, 4.0]);
    }

    // -------
    // Average
    // -------

    #[test]
    fn test_average_is_element_mean() {
        let mut c = EmbeddingComposer::new();
        let inputs = vec![
            make_input(1, vec![0.0, 4.0], 1.0),
            make_input(2, vec![2.0, 0.0], 1.0),
        ];
        let res = c
            .compose(&inputs, CompositionStrategy::Average)
            .expect("test: compose average failed");
        assert!((res.composed[0] - 1.0).abs() < 1e-6);
        assert!((res.composed[1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_average_three_inputs() {
        let mut c = EmbeddingComposer::new();
        let inputs = vec![
            make_input(1, vec![3.0], 1.0),
            make_input(2, vec![6.0], 1.0),
            make_input(3, vec![9.0], 1.0),
        ];
        let res = c
            .compose(&inputs, CompositionStrategy::Average)
            .expect("test: compose average three inputs failed");
        assert!((res.composed[0] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_average_output_dim_same_as_input() {
        let mut c = EmbeddingComposer::new();
        let inputs = vec![
            make_input(1, vec![0.0; 4], 1.0),
            make_input(2, vec![0.0; 4], 1.0),
        ];
        let res = c
            .compose(&inputs, CompositionStrategy::Average)
            .expect("test: compose average output dim failed");
        assert_eq!(res.output_dim, 4);
    }

    // -------
    // WeightedAverage
    // -------

    #[test]
    fn test_weighted_average_applies_weights() {
        let mut c = EmbeddingComposer::new();
        // weight 1 (0.25) and weight 3 (0.75)
        let inputs = vec![make_input(1, vec![0.0], 1.0), make_input(2, vec![4.0], 3.0)];
        let res = c
            .compose(&inputs, CompositionStrategy::WeightedAverage)
            .expect("test: compose weighted average failed");
        // Expected: 0*0.25 + 4*0.75 = 3.0
        assert!((res.composed[0] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_weighted_average_zero_weight_sum_is_equal() {
        let mut c = EmbeddingComposer::new();
        let inputs = vec![make_input(1, vec![0.0], 0.0), make_input(2, vec![4.0], 0.0)];
        let res = c
            .compose(&inputs, CompositionStrategy::WeightedAverage)
            .expect("test: compose weighted average zero weights failed");
        // Equal weights → mean
        assert!((res.composed[0] - 2.0).abs() < 1e-5);
    }

    // -------
    // MaxPooling
    // -------

    #[test]
    fn test_max_pooling_takes_element_max() {
        let mut c = EmbeddingComposer::new();
        let inputs = vec![
            make_input(1, vec![1.0, 5.0, 2.0], 1.0),
            make_input(2, vec![3.0, 2.0, 7.0], 1.0),
        ];
        let res = c
            .compose(&inputs, CompositionStrategy::MaxPooling)
            .expect("test: compose max pooling failed");
        assert_eq!(res.composed, vec![3.0, 5.0, 7.0]);
    }

    #[test]
    fn test_max_pooling_three_inputs() {
        let mut c = EmbeddingComposer::new();
        let inputs = vec![
            make_input(1, vec![1.0, 9.0], 1.0),
            make_input(2, vec![5.0, 2.0], 1.0),
            make_input(3, vec![3.0, 7.0], 1.0),
        ];
        let res = c
            .compose(&inputs, CompositionStrategy::MaxPooling)
            .expect("test: compose max pooling three inputs failed");
        assert_eq!(res.composed, vec![5.0, 9.0]);
    }

    // -------
    // HadamardProduct
    // -------

    #[test]
    fn test_hadamard_product_multiplies() {
        let mut c = EmbeddingComposer::new();
        let inputs = vec![
            make_input(1, vec![2.0, 3.0], 1.0),
            make_input(2, vec![4.0, 5.0], 1.0),
        ];
        let res = c
            .compose(&inputs, CompositionStrategy::HadamardProduct)
            .expect("test: compose hadamard product failed");
        assert_eq!(res.composed, vec![8.0, 15.0]);
    }

    #[test]
    fn test_hadamard_product_three_inputs() {
        let mut c = EmbeddingComposer::new();
        let inputs = vec![
            make_input(1, vec![2.0], 1.0),
            make_input(2, vec![3.0], 1.0),
            make_input(3, vec![4.0], 1.0),
        ];
        let res = c
            .compose(&inputs, CompositionStrategy::HadamardProduct)
            .expect("test: compose hadamard product three inputs failed");
        assert_eq!(res.composed, vec![24.0]);
    }

    // -------
    // Error cases
    // -------

    #[test]
    fn test_empty_inputs_returns_error() {
        let mut c = EmbeddingComposer::new();
        let err = c
            .compose(&[], CompositionStrategy::Average)
            .expect_err("test: expected error for empty inputs");
        assert_eq!(err, "no inputs");
    }

    #[test]
    fn test_dimension_mismatch_average() {
        let mut c = EmbeddingComposer::new();
        let inputs = vec![
            make_input(1, vec![1.0, 2.0], 1.0),
            make_input(2, vec![3.0], 1.0),
        ];
        let err = c
            .compose(&inputs, CompositionStrategy::Average)
            .expect_err("test: expected dimension mismatch error for average");
        assert_eq!(err, "dimension mismatch");
    }

    #[test]
    fn test_dimension_mismatch_max_pooling() {
        let mut c = EmbeddingComposer::new();
        let inputs = vec![
            make_input(1, vec![1.0, 2.0], 1.0),
            make_input(2, vec![3.0, 4.0, 5.0], 1.0),
        ];
        let err = c
            .compose(&inputs, CompositionStrategy::MaxPooling)
            .expect_err("test: expected dimension mismatch error for max pooling");
        assert_eq!(err, "dimension mismatch");
    }

    #[test]
    fn test_dimension_mismatch_hadamard() {
        let mut c = EmbeddingComposer::new();
        let inputs = vec![
            make_input(1, vec![1.0], 1.0),
            make_input(2, vec![1.0, 2.0], 1.0),
        ];
        let err = c
            .compose(&inputs, CompositionStrategy::HadamardProduct)
            .expect_err("test: expected dimension mismatch error for hadamard");
        assert_eq!(err, "dimension mismatch");
    }

    // -------
    // l2_norm and normalize
    // -------

    #[test]
    fn test_l2_norm_correct() {
        let mut c = EmbeddingComposer::new();
        let inputs = vec![make_input(1, vec![3.0, 4.0], 1.0)];
        let res = c
            .compose(&inputs, CompositionStrategy::Concatenate)
            .expect("test: compose for l2 norm failed");
        assert!((res.l2_norm() - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_normalize_unit_length() {
        let mut c = EmbeddingComposer::new();
        let inputs = vec![make_input(1, vec![0.0, 3.0, 4.0], 1.0)];
        let res = c
            .compose(&inputs, CompositionStrategy::Concatenate)
            .expect("test: compose for normalize failed");
        let norm_vec = res.normalize();
        let sq_sum: f32 = norm_vec.iter().map(|v| v * v).sum();
        assert!((sq_sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_normalize_zero_vector_returns_zeros() {
        let mut c = EmbeddingComposer::new();
        let inputs = vec![make_input(1, vec![0.0, 0.0], 1.0)];
        let res = c
            .compose(&inputs, CompositionStrategy::Concatenate)
            .expect("test: compose for normalize zero vector failed");
        assert_eq!(res.normalize(), vec![0.0, 0.0]);
    }

    // -------
    // batch_compose
    // -------

    #[test]
    fn test_batch_compose_returns_correct_count() {
        let mut c = EmbeddingComposer::new();
        let batches = vec![
            (
                vec![
                    make_input(1, vec![1.0, 2.0], 1.0),
                    make_input(2, vec![3.0, 4.0], 1.0),
                ],
                CompositionStrategy::Concatenate,
            ),
            (
                vec![make_input(3, vec![1.0], 1.0), make_input(4, vec![1.0], 1.0)],
                CompositionStrategy::Average,
            ),
        ];
        let results = c.batch_compose(batches);
        assert_eq!(results.len(), 2);
        assert!(results[0].is_ok());
        assert!(results[1].is_ok());
    }

    #[test]
    fn test_batch_compose_with_error() {
        let mut c = EmbeddingComposer::new();
        let batches = vec![
            // Valid
            (
                vec![make_input(1, vec![1.0], 1.0), make_input(2, vec![2.0], 1.0)],
                CompositionStrategy::Average,
            ),
            // Invalid: empty
            (vec![], CompositionStrategy::MaxPooling),
        ];
        let results = c.batch_compose(batches);
        assert!(results[0].is_ok());
        assert!(results[1].is_err());
    }

    // -------
    // Stats
    // -------

    #[test]
    fn test_stats_total_composed() {
        let mut c = EmbeddingComposer::new();
        let inputs = vec![make_input(1, vec![1.0], 1.0)];
        c.compose(&inputs, CompositionStrategy::Concatenate)
            .expect("test: first compose for stats failed");
        c.compose(&inputs, CompositionStrategy::Concatenate)
            .expect("test: second compose for stats failed");
        assert_eq!(c.stats().total_composed, 2);
    }

    #[test]
    fn test_stats_most_used_strategy() {
        let mut c = EmbeddingComposer::new();
        let inputs = vec![make_input(1, vec![1.0], 1.0), make_input(2, vec![1.0], 1.0)];
        c.compose(&inputs, CompositionStrategy::Average)
            .expect("test: compose average for stats failed");
        c.compose(&inputs, CompositionStrategy::Average)
            .expect("test: compose second average for stats failed");
        c.compose(&inputs, CompositionStrategy::MaxPooling)
            .expect("test: compose max pooling for stats failed");
        assert_eq!(
            c.stats().most_used_strategy(),
            Some(CompositionStrategy::Average)
        );
    }

    #[test]
    fn test_stats_most_used_strategy_none_when_empty() {
        let c = EmbeddingComposer::new();
        assert_eq!(c.stats().most_used_strategy(), None);
    }

    // -------
    // input_count field
    // -------

    #[test]
    fn test_input_count_field() {
        let mut c = EmbeddingComposer::new();
        let inputs = vec![
            make_input(1, vec![1.0], 1.0),
            make_input(2, vec![2.0], 1.0),
            make_input(3, vec![3.0], 1.0),
        ];
        let res = c
            .compose(&inputs, CompositionStrategy::Concatenate)
            .expect("test: compose for input_count failed");
        assert_eq!(res.input_count, 3);
    }
}
