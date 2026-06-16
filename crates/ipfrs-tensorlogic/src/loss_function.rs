//! TensorLossFunction — common loss functions for tensor computations.
//!
//! Provides MSE, MAE, Cross-Entropy, Huber, and Hinge loss with configurable
//! reduction modes (Mean, Sum, None) and per-element gradient computation.

/// Type of loss function to compute.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LossType {
    /// Mean Squared Error: (pred - target)^2
    MSE,
    /// Mean Absolute Error: |pred - target|
    MAE,
    /// Binary cross-entropy: -(t*ln(p+eps) + (1-t)*ln(1-p+eps))
    CrossEntropy,
    /// Huber loss (smooth L1): quadratic near zero, linear far from zero
    Huber,
    /// Hinge loss for SVM: max(0, 1 - target*pred)
    Hinge,
}

/// How to aggregate per-element losses into a scalar.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reduction {
    /// Average of all element losses.
    Mean,
    /// Sum of all element losses.
    Sum,
    /// No reduction — return per-element losses (used by `compute`).
    None,
}

/// Configuration for a [`TensorLossFunction`].
#[derive(Debug, Clone)]
pub struct LossConfig {
    /// Which loss formula to use.
    pub loss_type: LossType,
    /// Delta threshold for Huber loss (default 1.0).
    pub huber_delta: f64,
    /// Small constant for numerical stability (default 1e-7).
    pub epsilon: f64,
    /// Reduction mode (default [`Reduction::Mean`]).
    pub reduction: Reduction,
}

impl Default for LossConfig {
    fn default() -> Self {
        Self {
            loss_type: LossType::MSE,
            huber_delta: 1.0,
            epsilon: 1e-7,
            reduction: Reduction::Mean,
        }
    }
}

/// Statistics snapshot for a [`TensorLossFunction`].
#[derive(Debug, Clone)]
pub struct LossFunctionStats {
    /// The configured loss type.
    pub loss_type: LossType,
    /// Total number of forward/compute/gradient calls (element-level operations).
    pub computations: u64,
}

/// Common loss functions for tensor computations with gradient support.
///
/// # Examples
///
/// ```
/// use ipfrs_tensorlogic::loss_function::{TensorLossFunction, LossConfig, LossType, Reduction};
///
/// let config = LossConfig {
///     loss_type: LossType::MSE,
///     reduction: Reduction::Mean,
///     ..LossConfig::default()
/// };
/// let mut loss_fn = TensorLossFunction::new(config);
///
/// let preds = vec![1.0, 2.0, 3.0];
/// let targets = vec![1.5, 2.5, 3.5];
///
/// let loss = loss_fn.forward(&preds, &targets).expect("example: should succeed in docs");
/// assert!((loss - 0.25).abs() < 1e-10); // mean of [0.25, 0.25, 0.25]
/// ```
pub struct TensorLossFunction {
    config: LossConfig,
    computations: u64,
}

impl TensorLossFunction {
    /// Create a new loss function with the given configuration.
    pub fn new(config: LossConfig) -> Self {
        Self {
            config,
            computations: 0,
        }
    }

    /// Compute per-element loss values.
    ///
    /// Returns an error if `predictions` and `targets` have different lengths.
    pub fn compute(&mut self, predictions: &[f64], targets: &[f64]) -> Result<Vec<f64>, String> {
        if predictions.len() != targets.len() {
            return Err(format!(
                "length mismatch: predictions={} vs targets={}",
                predictions.len(),
                targets.len()
            ));
        }

        let eps = self.config.epsilon;
        let delta = self.config.huber_delta;

        let losses: Vec<f64> = predictions
            .iter()
            .zip(targets.iter())
            .map(|(&p, &t)| match self.config.loss_type {
                LossType::MSE => {
                    let d = p - t;
                    d * d
                }
                LossType::MAE => (p - t).abs(),
                LossType::CrossEntropy => -(t * (p + eps).ln() + (1.0 - t) * (1.0 - p + eps).ln()),
                LossType::Huber => {
                    let d = (p - t).abs();
                    if d <= delta {
                        0.5 * d * d
                    } else {
                        delta * (d - 0.5 * delta)
                    }
                }
                LossType::Hinge => {
                    let margin = 1.0 - t * p;
                    if margin > 0.0 {
                        margin
                    } else {
                        0.0
                    }
                }
            })
            .collect();

        self.computations += losses.len() as u64;
        Ok(losses)
    }

    /// Apply the configured reduction to a slice of per-element losses.
    pub fn reduce(&self, losses: &[f64]) -> f64 {
        match self.config.reduction {
            Reduction::Sum => losses.iter().sum(),
            Reduction::Mean => {
                if losses.is_empty() {
                    0.0
                } else {
                    let sum: f64 = losses.iter().sum();
                    sum / losses.len() as f64
                }
            }
            Reduction::None => {
                // When "None" reduction is used in a scalar context, return sum
                // (the caller should use `compute` directly for per-element values).
                losses.iter().sum()
            }
        }
    }

    /// Compute loss and reduce in one call.
    pub fn forward(&mut self, predictions: &[f64], targets: &[f64]) -> Result<f64, String> {
        let losses = self.compute(predictions, targets)?;
        Ok(self.reduce(&losses))
    }

    /// Compute dL/d(prediction) per element.
    ///
    /// Returns an error if `predictions` and `targets` have different lengths.
    pub fn gradient(&mut self, predictions: &[f64], targets: &[f64]) -> Result<Vec<f64>, String> {
        if predictions.len() != targets.len() {
            return Err(format!(
                "length mismatch: predictions={} vs targets={}",
                predictions.len(),
                targets.len()
            ));
        }

        let eps = self.config.epsilon;
        let delta = self.config.huber_delta;

        let grads: Vec<f64> = predictions
            .iter()
            .zip(targets.iter())
            .map(|(&p, &t)| match self.config.loss_type {
                LossType::MSE => 2.0 * (p - t),
                LossType::MAE => {
                    let d = p - t;
                    if d > 0.0 {
                        1.0
                    } else if d < 0.0 {
                        -1.0
                    } else {
                        0.0
                    }
                }
                LossType::CrossEntropy => -(t / (p + eps)) + (1.0 - t) / (1.0 - p + eps),
                LossType::Huber => {
                    let d = p - t;
                    let abs_d = d.abs();
                    if abs_d <= delta {
                        d
                    } else if d > 0.0 {
                        delta
                    } else {
                        -delta
                    }
                }
                LossType::Hinge => {
                    if t * p < 1.0 {
                        -t
                    } else {
                        0.0
                    }
                }
            })
            .collect();

        self.computations += grads.len() as u64;
        Ok(grads)
    }

    /// Static helper: compute mean squared error between two slices.
    ///
    /// Returns 0.0 if slices are empty or have different lengths.
    pub fn mse(a: &[f64], b: &[f64]) -> f64 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }
        let sum: f64 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum();
        sum / a.len() as f64
    }

    /// Static helper: compute mean absolute error between two slices.
    ///
    /// Returns 0.0 if slices are empty or have different lengths.
    pub fn mae(a: &[f64], b: &[f64]) -> f64 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }
        let sum: f64 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum();
        sum / a.len() as f64
    }

    /// Return a snapshot of accumulated statistics.
    pub fn stats(&self) -> LossFunctionStats {
        LossFunctionStats {
            loss_type: self.config.loss_type,
            computations: self.computations,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(loss_type: LossType) -> LossConfig {
        LossConfig {
            loss_type,
            ..LossConfig::default()
        }
    }

    fn cfg_with_reduction(loss_type: LossType, reduction: Reduction) -> LossConfig {
        LossConfig {
            loss_type,
            reduction,
            ..LossConfig::default()
        }
    }

    // ---- MSE ----

    #[test]
    fn mse_basic() {
        let mut f = TensorLossFunction::new(cfg(LossType::MSE));
        let losses = f.compute(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]).expect("ok");
        assert!(losses.iter().all(|&v| v.abs() < 1e-15));
    }

    #[test]
    fn mse_nonzero() {
        let mut f = TensorLossFunction::new(cfg(LossType::MSE));
        let losses = f.compute(&[1.0, 2.0], &[2.0, 4.0]).expect("ok");
        assert!((losses[0] - 1.0).abs() < 1e-15);
        assert!((losses[1] - 4.0).abs() < 1e-15);
    }

    #[test]
    fn mse_forward_mean() {
        let mut f = TensorLossFunction::new(cfg(LossType::MSE));
        let val = f.forward(&[1.0, 2.0, 3.0], &[1.5, 2.5, 3.5]).expect("ok");
        assert!((val - 0.25).abs() < 1e-10);
    }

    #[test]
    fn mse_gradient() {
        let mut f = TensorLossFunction::new(cfg(LossType::MSE));
        let g = f.gradient(&[3.0], &[1.0]).expect("ok");
        assert!((g[0] - 4.0).abs() < 1e-15); // 2*(3-1)=4
    }

    // ---- MAE ----

    #[test]
    fn mae_basic() {
        let mut f = TensorLossFunction::new(cfg(LossType::MAE));
        let losses = f.compute(&[1.0, 5.0], &[3.0, 2.0]).expect("ok");
        assert!((losses[0] - 2.0).abs() < 1e-15);
        assert!((losses[1] - 3.0).abs() < 1e-15);
    }

    #[test]
    fn mae_gradient_positive() {
        let mut f = TensorLossFunction::new(cfg(LossType::MAE));
        let g = f.gradient(&[5.0], &[3.0]).expect("ok");
        assert!((g[0] - 1.0).abs() < 1e-15);
    }

    #[test]
    fn mae_gradient_negative() {
        let mut f = TensorLossFunction::new(cfg(LossType::MAE));
        let g = f.gradient(&[1.0], &[3.0]).expect("ok");
        assert!((g[0] - (-1.0)).abs() < 1e-15);
    }

    #[test]
    fn mae_gradient_zero() {
        let mut f = TensorLossFunction::new(cfg(LossType::MAE));
        let g = f.gradient(&[3.0], &[3.0]).expect("ok");
        assert!(g[0].abs() < 1e-15);
    }

    // ---- CrossEntropy ----

    #[test]
    fn cross_entropy_perfect_prediction() {
        let mut f = TensorLossFunction::new(cfg(LossType::CrossEntropy));
        // pred≈1 for target=1 should give loss≈0
        let losses = f.compute(&[0.9999999], &[1.0]).expect("ok");
        assert!(losses[0] < 0.001);
    }

    #[test]
    fn cross_entropy_bad_prediction() {
        let mut f = TensorLossFunction::new(cfg(LossType::CrossEntropy));
        // pred≈0 for target=1 should give large loss
        let losses = f.compute(&[0.01], &[1.0]).expect("ok");
        assert!(losses[0] > 1.0);
    }

    #[test]
    fn cross_entropy_gradient() {
        let mut f = TensorLossFunction::new(cfg(LossType::CrossEntropy));
        let eps = 1e-7;
        let p = 0.7;
        let t = 1.0;
        let g = f.gradient(&[p], &[t]).expect("ok");
        let expected = -(t / (p + eps)) + (1.0 - t) / (1.0 - p + eps);
        assert!((g[0] - expected).abs() < 1e-10);
    }

    #[test]
    fn cross_entropy_symmetry() {
        // For target=0, loss should behave symmetrically
        let mut f = TensorLossFunction::new(cfg(LossType::CrossEntropy));
        let losses = f.compute(&[0.01], &[0.0]).expect("ok");
        assert!(losses[0] < 0.02); // small loss for correct prediction
    }

    // ---- Huber ----

    #[test]
    fn huber_quadratic_region() {
        let mut f = TensorLossFunction::new(cfg(LossType::Huber));
        // |d|=0.5 <= delta=1.0 => 0.5 * 0.5^2 = 0.125
        let losses = f.compute(&[1.5], &[1.0]).expect("ok");
        assert!((losses[0] - 0.125).abs() < 1e-15);
    }

    #[test]
    fn huber_linear_region() {
        let mut f = TensorLossFunction::new(cfg(LossType::Huber));
        // |d|=2.0 > delta=1.0 => 1.0*(2.0 - 0.5) = 1.5
        let losses = f.compute(&[3.0], &[1.0]).expect("ok");
        assert!((losses[0] - 1.5).abs() < 1e-15);
    }

    #[test]
    fn huber_transition_at_delta() {
        // At exactly |d|=delta, both branches should give same result
        let mut f = TensorLossFunction::new(cfg(LossType::Huber));
        let losses = f.compute(&[2.0], &[1.0]).expect("ok");
        // quadratic: 0.5 * 1^2 = 0.5
        // linear:    1*(1 - 0.5) = 0.5
        assert!((losses[0] - 0.5).abs() < 1e-15);
    }

    #[test]
    fn huber_custom_delta() {
        let config = LossConfig {
            loss_type: LossType::Huber,
            huber_delta: 0.5,
            ..LossConfig::default()
        };
        let mut f = TensorLossFunction::new(config);
        // |d|=1.0 > delta=0.5 => 0.5*(1.0 - 0.25) = 0.375
        let losses = f.compute(&[2.0], &[1.0]).expect("ok");
        assert!((losses[0] - 0.375).abs() < 1e-15);
    }

    #[test]
    fn huber_gradient_quadratic() {
        let mut f = TensorLossFunction::new(cfg(LossType::Huber));
        let g = f.gradient(&[1.3], &[1.0]).expect("ok");
        assert!((g[0] - 0.3).abs() < 1e-14);
    }

    #[test]
    fn huber_gradient_linear_positive() {
        let mut f = TensorLossFunction::new(cfg(LossType::Huber));
        let g = f.gradient(&[5.0], &[1.0]).expect("ok");
        assert!((g[0] - 1.0).abs() < 1e-15); // delta * sign(d) = 1.0
    }

    #[test]
    fn huber_gradient_linear_negative() {
        let mut f = TensorLossFunction::new(cfg(LossType::Huber));
        let g = f.gradient(&[1.0], &[5.0]).expect("ok");
        assert!((g[0] - (-1.0)).abs() < 1e-15);
    }

    // ---- Hinge ----

    #[test]
    fn hinge_correct_large_margin() {
        let mut f = TensorLossFunction::new(cfg(LossType::Hinge));
        // target=1, pred=2 => max(0, 1-2)=0
        let losses = f.compute(&[2.0], &[1.0]).expect("ok");
        assert!(losses[0].abs() < 1e-15);
    }

    #[test]
    fn hinge_violation() {
        let mut f = TensorLossFunction::new(cfg(LossType::Hinge));
        // target=1, pred=0.5 => max(0, 1-0.5)=0.5
        let losses = f.compute(&[0.5], &[1.0]).expect("ok");
        assert!((losses[0] - 0.5).abs() < 1e-15);
    }

    #[test]
    fn hinge_negative_target() {
        let mut f = TensorLossFunction::new(cfg(LossType::Hinge));
        // target=-1, pred=-2 => max(0, 1-(-1*-2))=max(0,-1)=0
        let losses = f.compute(&[-2.0], &[-1.0]).expect("ok");
        assert!(losses[0].abs() < 1e-15);
    }

    #[test]
    fn hinge_gradient_active() {
        let mut f = TensorLossFunction::new(cfg(LossType::Hinge));
        let g = f.gradient(&[0.5], &[1.0]).expect("ok");
        assert!((g[0] - (-1.0)).abs() < 1e-15); // -target
    }

    #[test]
    fn hinge_gradient_inactive() {
        let mut f = TensorLossFunction::new(cfg(LossType::Hinge));
        let g = f.gradient(&[2.0], &[1.0]).expect("ok");
        assert!(g[0].abs() < 1e-15);
    }

    // ---- Reduction modes ----

    #[test]
    fn reduction_sum() {
        let mut f = TensorLossFunction::new(cfg_with_reduction(LossType::MSE, Reduction::Sum));
        let val = f.forward(&[1.0, 2.0], &[2.0, 4.0]).expect("ok");
        assert!((val - 5.0).abs() < 1e-15); // 1+4=5
    }

    #[test]
    fn reduction_mean() {
        let mut f = TensorLossFunction::new(cfg_with_reduction(LossType::MSE, Reduction::Mean));
        let val = f.forward(&[1.0, 2.0], &[2.0, 4.0]).expect("ok");
        assert!((val - 2.5).abs() < 1e-15); // (1+4)/2=2.5
    }

    #[test]
    fn reduction_none_returns_sum_in_forward() {
        let mut f = TensorLossFunction::new(cfg_with_reduction(LossType::MSE, Reduction::None));
        let val = f.forward(&[1.0, 2.0], &[2.0, 4.0]).expect("ok");
        // None reduction in scalar context falls back to sum
        assert!((val - 5.0).abs() < 1e-15);
    }

    #[test]
    fn reduce_empty() {
        let f = TensorLossFunction::new(cfg(LossType::MSE));
        assert!(f.reduce(&[]).abs() < 1e-15);
    }

    // ---- Length mismatch ----

    #[test]
    fn compute_length_mismatch() {
        let mut f = TensorLossFunction::new(cfg(LossType::MSE));
        let res = f.compute(&[1.0, 2.0], &[1.0]);
        assert!(res.is_err());
        let msg = res.expect_err("should fail");
        assert!(msg.contains("length mismatch"));
    }

    #[test]
    fn gradient_length_mismatch() {
        let mut f = TensorLossFunction::new(cfg(LossType::MSE));
        let res = f.gradient(&[1.0], &[1.0, 2.0]);
        assert!(res.is_err());
    }

    #[test]
    fn forward_length_mismatch() {
        let mut f = TensorLossFunction::new(cfg(LossType::MSE));
        let res = f.forward(&[1.0, 2.0, 3.0], &[1.0]);
        assert!(res.is_err());
    }

    // ---- Edge cases ----

    #[test]
    fn all_zeros() {
        let mut f = TensorLossFunction::new(cfg(LossType::MSE));
        let losses = f.compute(&[0.0, 0.0], &[0.0, 0.0]).expect("ok");
        assert!(losses.iter().all(|&v| v.abs() < 1e-15));
    }

    #[test]
    fn all_ones_mse() {
        let mut f = TensorLossFunction::new(cfg(LossType::MSE));
        let losses = f.compute(&[1.0, 1.0], &[1.0, 1.0]).expect("ok");
        assert!(losses.iter().all(|&v| v.abs() < 1e-15));
    }

    #[test]
    fn empty_inputs() {
        let mut f = TensorLossFunction::new(cfg(LossType::MSE));
        let losses = f.compute(&[], &[]).expect("ok");
        assert!(losses.is_empty());
    }

    #[test]
    fn single_element() {
        let mut f = TensorLossFunction::new(cfg(LossType::MAE));
        let val = f.forward(&[5.0], &[3.0]).expect("ok");
        assert!((val - 2.0).abs() < 1e-15);
    }

    // ---- Static helpers ----

    #[test]
    fn static_mse() {
        let val = TensorLossFunction::mse(&[1.0, 2.0], &[2.0, 4.0]);
        assert!((val - 2.5).abs() < 1e-15);
    }

    #[test]
    fn static_mae() {
        let val = TensorLossFunction::mae(&[1.0, 2.0], &[3.0, 5.0]);
        assert!((val - 2.5).abs() < 1e-15);
    }

    #[test]
    fn static_mse_length_mismatch() {
        let val = TensorLossFunction::mse(&[1.0], &[1.0, 2.0]);
        assert!(val.abs() < 1e-15);
    }

    #[test]
    fn static_mae_empty() {
        let val = TensorLossFunction::mae(&[], &[]);
        assert!(val.abs() < 1e-15);
    }

    // ---- Stats ----

    #[test]
    fn stats_initial() {
        let f = TensorLossFunction::new(cfg(LossType::Huber));
        let s = f.stats();
        assert_eq!(s.loss_type, LossType::Huber);
        assert_eq!(s.computations, 0);
    }

    #[test]
    fn stats_after_compute() {
        let mut f = TensorLossFunction::new(cfg(LossType::MSE));
        let _ = f.compute(&[1.0, 2.0, 3.0], &[0.0, 0.0, 0.0]);
        assert_eq!(f.stats().computations, 3);
    }

    #[test]
    fn stats_accumulate() {
        let mut f = TensorLossFunction::new(cfg(LossType::MSE));
        let _ = f.compute(&[1.0, 2.0], &[0.0, 0.0]);
        let _ = f.gradient(&[1.0, 2.0, 3.0], &[0.0, 0.0, 0.0]);
        assert_eq!(f.stats().computations, 5); // 2 + 3
    }

    // ---- Numerical gradient verification ----

    #[test]
    fn numerical_gradient_mse() {
        verify_numerical_gradient(LossType::MSE, &[1.5, 2.5, 0.3], &[1.0, 3.0, 0.1]);
    }

    #[test]
    fn numerical_gradient_mae() {
        // MAE gradient is discontinuous at 0, so avoid exact matches
        verify_numerical_gradient(LossType::MAE, &[1.5, 2.5, 0.3], &[1.0, 3.0, 0.1]);
    }

    #[test]
    fn numerical_gradient_cross_entropy() {
        verify_numerical_gradient(LossType::CrossEntropy, &[0.7, 0.3, 0.9], &[1.0, 0.0, 1.0]);
    }

    #[test]
    fn numerical_gradient_huber() {
        verify_numerical_gradient(LossType::Huber, &[1.5, 4.0, 0.3], &[1.0, 1.0, 0.1]);
    }

    /// Verify analytical gradient against finite-difference approximation.
    fn verify_numerical_gradient(loss_type: LossType, preds: &[f64], targets: &[f64]) {
        let config = cfg(loss_type);
        let mut f = TensorLossFunction::new(config.clone());
        let analytical = f.gradient(preds, targets).expect("ok");

        let h = 1e-5;
        for i in 0..preds.len() {
            let mut p_plus = preds.to_vec();
            let mut p_minus = preds.to_vec();
            p_plus[i] += h;
            p_minus[i] -= h;

            let mut f1 = TensorLossFunction::new(config.clone());
            let mut f2 = TensorLossFunction::new(config.clone());
            let l_plus = f1.compute(&p_plus, targets).expect("ok");
            let l_minus = f2.compute(&p_minus, targets).expect("ok");

            let numerical = (l_plus[i] - l_minus[i]) / (2.0 * h);
            let tol = 1e-4;
            assert!(
                (analytical[i] - numerical).abs() < tol,
                "{:?} grad[{}]: analytical={}, numerical={}",
                loss_type,
                i,
                analytical[i],
                numerical
            );
        }
    }
}
