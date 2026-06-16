//! Batch Normalization layer with running statistics tracking.
//!
//! Provides [`TensorBatchNorm`] — a full batch normalisation layer that
//! supports both *training* mode (batch statistics, running stat updates) and
//! *inference* mode (uses pre-accumulated running statistics).  Learnable
//! affine parameters γ (gamma / scale) and β (beta / shift) are optional.

/// Execution mode for batch normalisation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NormMode {
    /// Use batch statistics and update running statistics.
    Training,
    /// Use accumulated running statistics only (no update).
    Inference,
}

/// Configuration for [`TensorBatchNorm`].
#[derive(Clone, Debug)]
pub struct BatchNormConfig {
    /// Number of channels / features (C in an [N, C] input).
    pub num_features: usize,
    /// Small constant added to the variance for numerical stability.
    pub epsilon: f64,
    /// Momentum for the exponential moving average of running statistics.
    pub momentum: f64,
    /// Whether to apply learnable affine parameters γ and β.
    pub affine: bool,
}

impl BatchNormConfig {
    /// Create a `BatchNormConfig` with sensible defaults for `num_features`.
    pub fn default_for(num_features: usize) -> Self {
        Self {
            num_features,
            epsilon: 1e-5,
            momentum: 0.1,
            affine: true,
        }
    }
}

/// Aggregate statistics collected across [`TensorBatchNorm::forward`] calls.
#[derive(Clone, Debug, Default)]
pub struct BatchNormStats {
    /// Total number of `forward` invocations (training + inference).
    pub total_forward_passes: u64,
    /// Number of successful `forward` calls in [`NormMode::Training`].
    pub training_passes: u64,
    /// Number of successful `forward` calls in [`NormMode::Inference`].
    pub inference_passes: u64,
}

/// Batch normalisation layer.
///
/// # Layout
///
/// Input is expected to have shape `[batch_size, num_features]`.
///
/// # Example
///
/// ```rust
/// use ipfrs_tensorlogic::{TensorBatchNorm, BatchNormConfig, NormMode};
///
/// let config = BatchNormConfig::default_for(4);
/// let mut bn = TensorBatchNorm::new(config);
///
/// let batch = vec![
///     vec![1.0_f64, 2.0, 3.0, 4.0],
///     vec![5.0_f64, 6.0, 7.0, 8.0],
/// ];
/// let output = bn.forward(&batch).expect("forward failed");
/// assert_eq!(output.len(), 2);
/// assert_eq!(output[0].len(), 4);
/// ```
pub struct TensorBatchNorm {
    /// Layer configuration.
    pub config: BatchNormConfig,
    /// Learnable scale parameters, one per feature (γ, initialised to 1.0).
    pub gamma: Vec<f64>,
    /// Learnable shift parameters, one per feature (β, initialised to 0.0).
    pub beta: Vec<f64>,
    /// Exponential moving average of per-feature batch means.
    pub running_mean: Vec<f64>,
    /// Exponential moving average of per-feature batch variances.
    pub running_var: Vec<f64>,
    /// Current execution mode.
    pub mode: NormMode,
    /// Collected statistics.
    pub stats: BatchNormStats,
}

impl TensorBatchNorm {
    /// Construct a new `TensorBatchNorm` from a [`BatchNormConfig`].
    pub fn new(config: BatchNormConfig) -> Self {
        let n = config.num_features;
        Self {
            gamma: vec![1.0; n],
            beta: vec![0.0; n],
            running_mean: vec![0.0; n],
            running_var: vec![1.0; n],
            mode: NormMode::Training,
            stats: BatchNormStats::default(),
            config,
        }
    }

    /// Switch the layer between training and inference mode.
    pub fn set_mode(&mut self, mode: NormMode) {
        self.mode = mode;
    }

    /// Run a forward pass through the batch normalisation layer.
    ///
    /// * `input` — a slice of rows, each of length `num_features`.
    ///
    /// Returns `None` when:
    /// - `input` is empty, or
    /// - any row does not have exactly `num_features` elements.
    pub fn forward(&mut self, input: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
        let n = self.config.num_features;
        let batch = input.len();

        if batch == 0 {
            return None;
        }
        for row in input {
            if row.len() != n {
                return None;
            }
        }

        let output = match self.mode {
            NormMode::Training => self.forward_training(input, batch, n),
            NormMode::Inference => self.forward_inference(input, batch, n),
        };

        // Update stats regardless of mode.
        self.stats.total_forward_passes += 1;
        match self.mode {
            NormMode::Training => self.stats.training_passes += 1,
            NormMode::Inference => self.stats.inference_passes += 1,
        }

        Some(output)
    }

    // ------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------

    fn forward_training(&mut self, input: &[Vec<f64>], batch: usize, n: usize) -> Vec<Vec<f64>> {
        let batch_f = batch as f64;
        let momentum = self.config.momentum;
        let epsilon = self.config.epsilon;

        // Per-feature mean and population variance over the batch.
        let mut batch_mean = vec![0.0_f64; n];
        let mut batch_var = vec![0.0_f64; n];

        for row in input {
            for f in 0..n {
                batch_mean[f] += row[f];
            }
        }
        for m in batch_mean.iter_mut() {
            *m /= batch_f;
        }

        for row in input {
            for f in 0..n {
                let diff = row[f] - batch_mean[f];
                batch_var[f] += diff * diff;
            }
        }
        for v in batch_var.iter_mut() {
            *v /= batch_f; // population variance
        }

        // Update running statistics (exponential moving average).
        for f in 0..n {
            self.running_mean[f] =
                (1.0 - momentum) * self.running_mean[f] + momentum * batch_mean[f];
            self.running_var[f] = (1.0 - momentum) * self.running_var[f] + momentum * batch_var[f];
        }

        // Normalise and apply optional affine transform.
        self.normalise(input, batch, n, &batch_mean, &batch_var, epsilon)
    }

    fn forward_inference(&self, input: &[Vec<f64>], batch: usize, n: usize) -> Vec<Vec<f64>> {
        let epsilon = self.config.epsilon;
        self.normalise(
            input,
            batch,
            n,
            &self.running_mean,
            &self.running_var,
            epsilon,
        )
    }

    /// Shared normalisation kernel.
    fn normalise(
        &self,
        input: &[Vec<f64>],
        batch: usize,
        n: usize,
        mean: &[f64],
        var: &[f64],
        epsilon: f64,
    ) -> Vec<Vec<f64>> {
        let mut output = vec![vec![0.0_f64; n]; batch];
        for b in 0..batch {
            for f in 0..n {
                let x_hat = (input[b][f] - mean[f]) / (var[f] + epsilon).sqrt();
                output[b][f] = if self.config.affine {
                    self.gamma[f] * x_hat + self.beta[f]
                } else {
                    x_hat
                };
            }
        }
        output
    }

    /// Replace the gamma (scale) parameters.
    ///
    /// Returns `false` and leaves `gamma` unchanged when `gamma.len() !=
    /// num_features`.
    pub fn set_gamma(&mut self, gamma: Vec<f64>) -> bool {
        if gamma.len() != self.config.num_features {
            return false;
        }
        self.gamma = gamma;
        true
    }

    /// Replace the beta (shift) parameters.
    ///
    /// Returns `false` and leaves `beta` unchanged when `beta.len() !=
    /// num_features`.
    pub fn set_beta(&mut self, beta: Vec<f64>) -> bool {
        if beta.len() != self.config.num_features {
            return false;
        }
        self.beta = beta;
        true
    }

    /// Borrow the accumulated statistics.
    pub fn stats(&self) -> &BatchNormStats {
        &self.stats
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn make_bn(num_features: usize) -> TensorBatchNorm {
        TensorBatchNorm::new(BatchNormConfig::default_for(num_features))
    }

    /// Compute mean of a flat f64 slice.
    fn mean(v: &[f64]) -> f64 {
        v.iter().sum::<f64>() / v.len() as f64
    }

    /// Compute population variance of a flat f64 slice.
    fn variance(v: &[f64]) -> f64 {
        let m = mean(v);
        v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64
    }

    // -----------------------------------------------------------------------
    // 1. Basic construction
    // -----------------------------------------------------------------------

    #[test]
    fn test_new_initialisation() {
        let bn = make_bn(3);
        assert_eq!(bn.gamma, vec![1.0, 1.0, 1.0]);
        assert_eq!(bn.beta, vec![0.0, 0.0, 0.0]);
        assert_eq!(bn.running_mean, vec![0.0, 0.0, 0.0]);
        assert_eq!(bn.running_var, vec![1.0, 1.0, 1.0]);
        assert_eq!(bn.mode, NormMode::Training);
    }

    // -----------------------------------------------------------------------
    // 2. forward — Training mode normalises batch
    // -----------------------------------------------------------------------

    #[test]
    fn test_training_output_shape() {
        let mut bn = make_bn(4);
        let batch = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![5.0, 6.0, 7.0, 8.0],
            vec![9.0, 10.0, 11.0, 12.0],
        ];
        let out = bn.forward(&batch).expect("forward returned None");
        assert_eq!(out.len(), 3);
        for row in &out {
            assert_eq!(row.len(), 4);
        }
    }

    #[test]
    fn test_training_output_mean_near_zero() {
        let mut bn = make_bn(2);
        // Affine=false so we get raw normalised values.
        bn.config.affine = false;
        let batch = vec![
            vec![1.0, 10.0],
            vec![2.0, 20.0],
            vec![3.0, 30.0],
            vec![4.0, 40.0],
        ];
        let out = bn.forward(&batch).expect("forward None");

        // For feature 0: collect the 4 outputs.
        let f0: Vec<f64> = out.iter().map(|r| r[0]).collect();
        let f1: Vec<f64> = out.iter().map(|r| r[1]).collect();

        assert!(mean(&f0).abs() < 1e-10, "mean of feature 0 ≈ 0");
        assert!(mean(&f1).abs() < 1e-10, "mean of feature 1 ≈ 0");
    }

    #[test]
    fn test_training_output_var_near_one() {
        let mut bn = make_bn(2);
        bn.config.affine = false;
        let batch = vec![
            vec![1.0, 10.0],
            vec![2.0, 20.0],
            vec![3.0, 30.0],
            vec![4.0, 40.0],
        ];
        let out = bn.forward(&batch).expect("forward None");
        let f0: Vec<f64> = out.iter().map(|r| r[0]).collect();
        let f1: Vec<f64> = out.iter().map(|r| r[1]).collect();

        // Population variance of the normalised values should be ≈ 1.
        let var0 = variance(&f0);
        let var1 = variance(&f1);
        assert!((var0 - 1.0).abs() < 1e-4, "var feature 0 ≈ 1, got {var0}");
        assert!((var1 - 1.0).abs() < 1e-4, "var feature 1 ≈ 1, got {var1}");
    }

    // -----------------------------------------------------------------------
    // 3. Running statistics updated with momentum
    // -----------------------------------------------------------------------

    #[test]
    fn test_running_mean_updated() {
        let mut bn = make_bn(1);
        // With momentum=0.1: new_running = 0.9*0 + 0.1*batch_mean
        // batch_mean for [3.0, 7.0] = 5.0  → running_mean = 0.5
        let batch = vec![vec![3.0], vec![7.0]];
        bn.forward(&batch);
        let expected = 0.1 * 5.0; // 0.9*0 + 0.1*5
        assert!(
            (bn.running_mean[0] - expected).abs() < 1e-12,
            "running_mean = {}, expected {expected}",
            bn.running_mean[0]
        );
    }

    #[test]
    fn test_running_var_updated() {
        let mut bn = make_bn(1);
        // batch_var for [3, 7] (population) = ((3-5)^2 + (7-5)^2) / 2 = 4
        // running_var starts at 1.0
        // new_running_var = 0.9*1.0 + 0.1*4.0 = 0.9 + 0.4 = 1.3
        let batch = vec![vec![3.0], vec![7.0]];
        bn.forward(&batch);
        let expected = 0.9 * 1.0 + 0.1 * 4.0;
        assert!(
            (bn.running_var[0] - expected).abs() < 1e-12,
            "running_var = {}, expected {expected}",
            bn.running_var[0]
        );
    }

    #[test]
    fn test_running_stats_accumulate_over_multiple_passes() {
        let mut bn = make_bn(1);
        let batch1 = vec![vec![0.0], vec![2.0]]; // mean=1, var=1
        let batch2 = vec![vec![4.0], vec![8.0]]; // mean=6, var=4

        bn.forward(&batch1);
        let rm1 = bn.running_mean[0]; // 0.9*0 + 0.1*1 = 0.1
        let rv1 = bn.running_var[0]; // 0.9*1 + 0.1*1 = 1.0

        bn.forward(&batch2);
        let rm2_expected = 0.9 * rm1 + 0.1 * 6.0;
        let rv2_expected = 0.9 * rv1 + 0.1 * 4.0;

        assert!((bn.running_mean[0] - rm2_expected).abs() < 1e-12);
        assert!((bn.running_var[0] - rv2_expected).abs() < 1e-12);
    }

    // -----------------------------------------------------------------------
    // 4. Inference mode uses running stats
    // -----------------------------------------------------------------------

    #[test]
    fn test_inference_uses_running_stats() {
        let mut bn = make_bn(1);

        // Manually set running stats so the result is predictable.
        bn.running_mean[0] = 5.0;
        bn.running_var[0] = 4.0; // std=2
        bn.config.affine = false;
        bn.set_mode(NormMode::Inference);

        let batch = vec![vec![7.0]];
        let out = bn.forward(&batch).expect("None");
        // x_hat = (7 - 5) / sqrt(4 + 1e-5) ≈ 1.0
        let expected = (7.0 - 5.0) / (4.0_f64 + 1e-5).sqrt();
        assert!((out[0][0] - expected).abs() < 1e-9);
    }

    #[test]
    fn test_inference_running_stats_not_updated() {
        let mut bn = make_bn(1);
        bn.running_mean[0] = 3.0;
        bn.running_var[0] = 2.0;
        bn.set_mode(NormMode::Inference);

        bn.forward(&[vec![10.0], vec![20.0]]);
        // Running stats should not change.
        assert!((bn.running_mean[0] - 3.0).abs() < 1e-15);
        assert!((bn.running_var[0] - 2.0).abs() < 1e-15);
    }

    // -----------------------------------------------------------------------
    // 5. Affine transform
    // -----------------------------------------------------------------------

    #[test]
    fn test_affine_true_applies_gamma_beta() {
        let mut bn = make_bn(1);
        bn.gamma[0] = 2.0;
        bn.beta[0] = 1.0;
        bn.config.affine = true;

        // batch: constant  → x_hat = 0 for both (var → epsilon only)
        // Actually use a proper batch with variance.
        let batch = vec![vec![1.0], vec![3.0]]; // mean=2, var=1
        let out = bn.forward(&batch).expect("None");

        // x_hat[0] = (1-2)/sqrt(1+eps) ≈ -1,  y = 2*(-1)+1 = -1
        // x_hat[1] = (3-2)/sqrt(1+eps) ≈  1,  y = 2*( 1)+1 =  3
        let eps: f64 = 1e-5;
        let expected0 = 2.0 * ((1.0 - 2.0) / (1.0_f64 + eps).sqrt()) + 1.0;
        let expected1 = 2.0 * ((3.0 - 2.0) / (1.0_f64 + eps).sqrt()) + 1.0;
        assert!((out[0][0] - expected0).abs() < 1e-9);
        assert!((out[1][0] - expected1).abs() < 1e-9);
    }

    #[test]
    fn test_affine_false_no_gamma_beta() {
        let mut bn = make_bn(1);
        bn.gamma[0] = 99.0; // should NOT be applied
        bn.beta[0] = 99.0;
        bn.config.affine = false;

        let batch = vec![vec![1.0], vec![3.0]]; // mean=2, var=1
        let out = bn.forward(&batch).expect("None");
        let eps: f64 = 1e-5;
        let expected0 = (1.0 - 2.0) / (1.0_f64 + eps).sqrt();
        let expected1 = (3.0 - 2.0) / (1.0_f64 + eps).sqrt();
        assert!((out[0][0] - expected0).abs() < 1e-9);
        assert!((out[1][0] - expected1).abs() < 1e-9);
    }

    // -----------------------------------------------------------------------
    // 6. set_gamma / set_beta
    // -----------------------------------------------------------------------

    #[test]
    fn test_set_gamma_correct_size() {
        let mut bn = make_bn(3);
        assert!(bn.set_gamma(vec![2.0, 3.0, 4.0]));
        assert_eq!(bn.gamma, vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_set_gamma_wrong_size_returns_false() {
        let mut bn = make_bn(3);
        let old = bn.gamma.clone();
        assert!(!bn.set_gamma(vec![1.0, 2.0]));
        assert_eq!(bn.gamma, old, "gamma must be unchanged after rejection");
    }

    #[test]
    fn test_set_beta_correct_size() {
        let mut bn = make_bn(2);
        assert!(bn.set_beta(vec![0.5, -0.5]));
        assert_eq!(bn.beta, vec![0.5, -0.5]);
    }

    #[test]
    fn test_set_beta_wrong_size_returns_false() {
        let mut bn = make_bn(2);
        let old = bn.beta.clone();
        assert!(!bn.set_beta(vec![1.0, 2.0, 3.0]));
        assert_eq!(bn.beta, old);
    }

    #[test]
    fn test_set_gamma_empty_wrong_size() {
        let mut bn = make_bn(2);
        assert!(!bn.set_gamma(vec![]));
    }

    #[test]
    fn test_set_beta_empty_wrong_size() {
        let mut bn = make_bn(2);
        assert!(!bn.set_beta(vec![]));
    }

    // -----------------------------------------------------------------------
    // 7. forward returns None for invalid input
    // -----------------------------------------------------------------------

    #[test]
    fn test_forward_none_empty_batch() {
        let mut bn = make_bn(3);
        assert!(bn.forward(&[]).is_none());
    }

    #[test]
    fn test_forward_none_wrong_feature_count() {
        let mut bn = make_bn(3);
        let bad = vec![vec![1.0, 2.0]]; // only 2 features, expected 3
        assert!(bn.forward(&bad).is_none());
    }

    #[test]
    fn test_forward_none_one_row_wrong_features() {
        let mut bn = make_bn(4);
        let batch = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![5.0, 6.0, 7.0], // wrong!
        ];
        assert!(bn.forward(&batch).is_none());
    }

    // -----------------------------------------------------------------------
    // 8. Stats counting
    // -----------------------------------------------------------------------

    #[test]
    fn test_stats_training_pass_count() {
        let mut bn = make_bn(2);
        let batch = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        bn.forward(&batch);
        bn.forward(&batch);
        assert_eq!(bn.stats().training_passes, 2);
        assert_eq!(bn.stats().inference_passes, 0);
        assert_eq!(bn.stats().total_forward_passes, 2);
    }

    #[test]
    fn test_stats_inference_pass_count() {
        let mut bn = make_bn(2);
        bn.set_mode(NormMode::Inference);
        let batch = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        bn.forward(&batch);
        assert_eq!(bn.stats().inference_passes, 1);
        assert_eq!(bn.stats().training_passes, 0);
        assert_eq!(bn.stats().total_forward_passes, 1);
    }

    #[test]
    fn test_stats_mixed_modes() {
        let mut bn = make_bn(1);
        let b = vec![vec![1.0], vec![2.0]];
        bn.set_mode(NormMode::Training);
        bn.forward(&b);
        bn.forward(&b);
        bn.set_mode(NormMode::Inference);
        bn.forward(&b);
        assert_eq!(bn.stats().training_passes, 2);
        assert_eq!(bn.stats().inference_passes, 1);
        assert_eq!(bn.stats().total_forward_passes, 3);
    }

    #[test]
    fn test_stats_not_incremented_on_none() {
        let mut bn = make_bn(2);
        bn.forward(&[]); // returns None
        assert_eq!(bn.stats().total_forward_passes, 0);
    }

    // -----------------------------------------------------------------------
    // 9. Epsilon / momentum config
    // -----------------------------------------------------------------------

    #[test]
    fn test_custom_epsilon() {
        let config = BatchNormConfig {
            num_features: 1,
            epsilon: 1.0,
            momentum: 0.1,
            affine: false,
        };
        let mut bn = TensorBatchNorm::new(config);
        // batch: [1, 3], mean=2, var=1
        // x_hat[0] = (1-2)/sqrt(1+1) = -1/sqrt(2)
        let batch = vec![vec![1.0], vec![3.0]];
        let out = bn.forward(&batch).expect("None");
        let expected = (1.0 - 2.0) / (1.0_f64 + 1.0).sqrt();
        assert!((out[0][0] - expected).abs() < 1e-9);
    }

    #[test]
    fn test_custom_momentum() {
        let config = BatchNormConfig {
            num_features: 1,
            epsilon: 1e-5,
            momentum: 0.5, // high momentum
            affine: false,
        };
        let mut bn = TensorBatchNorm::new(config);
        // batch_mean = 2, running_mean starts at 0
        // new_running_mean = 0.5*0 + 0.5*2 = 1.0
        let batch = vec![vec![1.0], vec![3.0]];
        bn.forward(&batch);
        assert!((bn.running_mean[0] - 1.0).abs() < 1e-12);
    }

    // -----------------------------------------------------------------------
    // 10. set_mode
    // -----------------------------------------------------------------------

    #[test]
    fn test_set_mode_switches_correctly() {
        let mut bn = make_bn(1);
        assert_eq!(bn.mode, NormMode::Training);
        bn.set_mode(NormMode::Inference);
        assert_eq!(bn.mode, NormMode::Inference);
        bn.set_mode(NormMode::Training);
        assert_eq!(bn.mode, NormMode::Training);
    }

    // -----------------------------------------------------------------------
    // 11. default_for
    // -----------------------------------------------------------------------

    #[test]
    fn test_default_for_values() {
        let cfg = BatchNormConfig::default_for(8);
        assert_eq!(cfg.num_features, 8);
        assert!((cfg.epsilon - 1e-5).abs() < 1e-15);
        assert!((cfg.momentum - 0.1).abs() < 1e-15);
        assert!(cfg.affine);
    }

    // -----------------------------------------------------------------------
    // 12. Multi-feature forward correctness
    // -----------------------------------------------------------------------

    #[test]
    fn test_multi_feature_independent_normalisation() {
        let mut bn = make_bn(2);
        bn.config.affine = false;
        // Feature 0: values [0, 4] → mean=2, var=4, std=2
        // Feature 1: values [10, 10] → mean=10, var=0, std≈0
        let batch = vec![vec![0.0, 10.0], vec![4.0, 10.0]];
        let out = bn.forward(&batch).expect("None");

        // Feature 0: normalised outputs
        let eps = 1e-5;
        let exp0_0 = (0.0 - 2.0) / (4.0_f64 + eps).sqrt();
        let exp0_1 = (4.0 - 2.0) / (4.0_f64 + eps).sqrt();
        assert!((out[0][0] - exp0_0).abs() < 1e-9);
        assert!((out[1][0] - exp0_1).abs() < 1e-9);

        // Feature 1: var≈0, so x_hat ≈ 0 for both rows.
        let exp1 = (10.0 - 10.0) / (0.0_f64 + eps).sqrt();
        assert!((out[0][1] - exp1).abs() < 1e-9);
        assert!((out[1][1] - exp1).abs() < 1e-9);
    }

    // -----------------------------------------------------------------------
    // 13. NormMode Clone/Copy/Eq
    // -----------------------------------------------------------------------

    #[test]
    fn test_norm_mode_copy_clone() {
        let m = NormMode::Training;
        let m2 = m; // Copy
        let m3 = m; // Clone
        assert_eq!(m, m2);
        assert_eq!(m, m3);
        assert_ne!(m, NormMode::Inference);
    }

    // -----------------------------------------------------------------------
    // 14. Large single-feature batch — statistical accuracy
    // -----------------------------------------------------------------------

    #[test]
    fn test_large_batch_normalisation_accuracy() {
        let n = 1;
        let mut bn = TensorBatchNorm::new(BatchNormConfig {
            num_features: n,
            epsilon: 1e-8,
            momentum: 0.1,
            affine: false,
        });

        // 100 samples from [1..=100]
        let batch: Vec<Vec<f64>> = (1..=100).map(|i| vec![i as f64]).collect();
        let out = bn.forward(&batch).expect("None");

        let vals: Vec<f64> = out.iter().map(|r| r[0]).collect();
        let m = mean(&vals);
        let v = variance(&vals);

        assert!(m.abs() < 1e-10, "mean ≈ 0, got {m}");
        assert!((v - 1.0).abs() < 1e-6, "var ≈ 1, got {v}");
    }
}
