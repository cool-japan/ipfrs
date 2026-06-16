//! L1/L2/ElasticNet regularization for tensor parameters.
//!
//! Provides regularization penalty and gradient computation to prevent
//! overfitting during tensor-based optimization and training loops.

/// Type of regularization to apply.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegularizerType {
    /// L1 (Lasso): penalty = lambda * sum(|w|)
    L1,
    /// L2 (Ridge): penalty = lambda * sum(w^2)
    L2,
    /// ElasticNet: alpha * L1 + (1 - alpha) * L2
    ElasticNet,
}

/// Configuration for a `TensorRegularizer`.
#[derive(Debug, Clone)]
pub struct RegularizerConfig {
    /// Which regularization strategy to use.
    pub reg_type: RegularizerType,
    /// Regularization strength (default 0.01).
    pub lambda: f64,
    /// L1 ratio for ElasticNet (default 0.5).
    /// 0.0 = pure L2, 1.0 = pure L1.
    pub elastic_alpha: f64,
}

impl Default for RegularizerConfig {
    fn default() -> Self {
        Self {
            reg_type: RegularizerType::L2,
            lambda: 0.01,
            elastic_alpha: 0.5,
        }
    }
}

/// Runtime statistics for a `TensorRegularizer`.
#[derive(Debug, Clone)]
pub struct RegularizerStats {
    /// The regularizer type in use.
    pub reg_type: RegularizerType,
    /// The lambda (regularization strength).
    pub lambda: f64,
    /// Total number of penalty/gradient computations performed.
    pub computations: u64,
}

/// L1/L2/ElasticNet regularization for tensor weight vectors.
///
/// # Examples
///
/// ```
/// use ipfrs_tensorlogic::regularizer::{TensorRegularizer, RegularizerConfig, RegularizerType};
///
/// let config = RegularizerConfig {
///     reg_type: RegularizerType::L1,
///     lambda: 0.1,
///     elastic_alpha: 0.5,
/// };
/// let mut reg = TensorRegularizer::new(config);
///
/// let weights = vec![1.0, -2.0, 3.0];
/// let penalty = reg.penalty(&weights);
/// assert!((penalty - 0.6).abs() < 1e-12); // 0.1 * (1 + 2 + 3)
/// ```
pub struct TensorRegularizer {
    config: RegularizerConfig,
    computations: u64,
}

impl TensorRegularizer {
    /// Create a new regularizer from the given configuration.
    pub fn new(config: RegularizerConfig) -> Self {
        Self {
            config,
            computations: 0,
        }
    }

    // ------------------------------------------------------------------
    // Dispatched methods (delegate to the configured type)
    // ------------------------------------------------------------------

    /// Compute the regularization penalty for the given weight vector.
    pub fn penalty(&mut self, weights: &[f64]) -> f64 {
        self.computations += 1;
        match self.config.reg_type {
            RegularizerType::L1 => self.l1_penalty(weights),
            RegularizerType::L2 => self.l2_penalty(weights),
            RegularizerType::ElasticNet => self.elastic_penalty(weights),
        }
    }

    /// Compute the regularization gradient for the given weight vector.
    pub fn gradient(&mut self, weights: &[f64]) -> Vec<f64> {
        self.computations += 1;
        match self.config.reg_type {
            RegularizerType::L1 => self.l1_gradient(weights),
            RegularizerType::L2 => self.l2_gradient(weights),
            RegularizerType::ElasticNet => self.elastic_gradient(weights),
        }
    }

    // ------------------------------------------------------------------
    // L1
    // ------------------------------------------------------------------

    /// L1 penalty: `lambda * sum(|w_i|)`.
    pub fn l1_penalty(&self, weights: &[f64]) -> f64 {
        let sum_abs: f64 = weights.iter().map(|w| w.abs()).sum();
        self.config.lambda * sum_abs
    }

    /// L1 gradient: `lambda * sign(w_i)`.
    ///
    /// The sub-gradient at zero is defined as 0.0.
    pub fn l1_gradient(&self, weights: &[f64]) -> Vec<f64> {
        weights
            .iter()
            .map(|&w| {
                if w > 0.0 {
                    self.config.lambda
                } else if w < 0.0 {
                    -self.config.lambda
                } else {
                    0.0
                }
            })
            .collect()
    }

    // ------------------------------------------------------------------
    // L2
    // ------------------------------------------------------------------

    /// L2 penalty: `lambda * sum(w_i^2)`.
    pub fn l2_penalty(&self, weights: &[f64]) -> f64 {
        let sum_sq: f64 = weights.iter().map(|w| w * w).sum();
        self.config.lambda * sum_sq
    }

    /// L2 gradient: `2 * lambda * w_i`.
    pub fn l2_gradient(&self, weights: &[f64]) -> Vec<f64> {
        weights
            .iter()
            .map(|&w| 2.0 * self.config.lambda * w)
            .collect()
    }

    // ------------------------------------------------------------------
    // ElasticNet
    // ------------------------------------------------------------------

    /// ElasticNet penalty: `alpha * L1(w) + (1 - alpha) * L2(w)`.
    pub fn elastic_penalty(&self, weights: &[f64]) -> f64 {
        let alpha = self.config.elastic_alpha;
        alpha * self.l1_penalty(weights) + (1.0 - alpha) * self.l2_penalty(weights)
    }

    /// ElasticNet gradient: `alpha * L1_grad(w) + (1 - alpha) * L2_grad(w)`.
    pub fn elastic_gradient(&self, weights: &[f64]) -> Vec<f64> {
        let alpha = self.config.elastic_alpha;
        let l1 = self.l1_gradient(weights);
        let l2 = self.l2_gradient(weights);
        l1.iter()
            .zip(l2.iter())
            .map(|(&a, &b)| alpha * a + (1.0 - alpha) * b)
            .collect()
    }

    // ------------------------------------------------------------------
    // Stats
    // ------------------------------------------------------------------

    /// Return runtime statistics.
    pub fn stats(&self) -> RegularizerStats {
        RegularizerStats {
            reg_type: self.config.reg_type,
            lambda: self.config.lambda,
            computations: self.computations,
        }
    }
}

// ======================================================================
// Tests
// ======================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make(reg_type: RegularizerType, lambda: f64, elastic_alpha: f64) -> TensorRegularizer {
        TensorRegularizer::new(RegularizerConfig {
            reg_type,
            lambda,
            elastic_alpha,
        })
    }

    // ------------------------------------------------------------------
    // L1 penalty
    // ------------------------------------------------------------------

    #[test]
    fn l1_penalty_positive_weights() {
        let mut r = make(RegularizerType::L1, 0.1, 0.5);
        let p = r.penalty(&[1.0, 2.0, 3.0]);
        assert!((p - 0.6).abs() < 1e-12);
    }

    #[test]
    fn l1_penalty_negative_weights() {
        let mut r = make(RegularizerType::L1, 0.1, 0.5);
        let p = r.penalty(&[-1.0, -2.0, -3.0]);
        assert!((p - 0.6).abs() < 1e-12);
    }

    #[test]
    fn l1_penalty_mixed_weights() {
        let mut r = make(RegularizerType::L1, 0.5, 0.5);
        let p = r.penalty(&[1.0, -2.0, 0.0]);
        assert!((p - 1.5).abs() < 1e-12); // 0.5 * (1+2+0)
    }

    #[test]
    fn l1_penalty_zero_weights() {
        let mut r = make(RegularizerType::L1, 0.1, 0.5);
        let p = r.penalty(&[0.0, 0.0, 0.0]);
        assert!((p - 0.0).abs() < 1e-12);
    }

    #[test]
    fn l1_penalty_single_weight() {
        let mut r = make(RegularizerType::L1, 0.2, 0.5);
        let p = r.penalty(&[5.0]);
        assert!((p - 1.0).abs() < 1e-12); // 0.2 * 5
    }

    #[test]
    fn l1_penalty_empty_weights() {
        let mut r = make(RegularizerType::L1, 1.0, 0.5);
        let p = r.penalty(&[]);
        assert!((p - 0.0).abs() < 1e-12);
    }

    // ------------------------------------------------------------------
    // L1 gradient
    // ------------------------------------------------------------------

    #[test]
    fn l1_gradient_correctness() {
        let mut r = make(RegularizerType::L1, 0.1, 0.5);
        let g = r.gradient(&[3.0, -2.0, 0.0]);
        assert!((g[0] - 0.1).abs() < 1e-12);
        assert!((g[1] - (-0.1)).abs() < 1e-12);
        assert!((g[2] - 0.0).abs() < 1e-12);
    }

    #[test]
    fn l1_gradient_all_positive() {
        let r = make(RegularizerType::L1, 0.5, 0.5);
        let g = r.l1_gradient(&[1.0, 2.0, 3.0]);
        assert!(g.iter().all(|&v| (v - 0.5).abs() < 1e-12));
    }

    #[test]
    fn l1_gradient_all_negative() {
        let r = make(RegularizerType::L1, 0.5, 0.5);
        let g = r.l1_gradient(&[-1.0, -2.0, -3.0]);
        assert!(g.iter().all(|&v| (v - (-0.5)).abs() < 1e-12));
    }

    // ------------------------------------------------------------------
    // L2 penalty
    // ------------------------------------------------------------------

    #[test]
    fn l2_penalty_positive_weights() {
        let mut r = make(RegularizerType::L2, 0.1, 0.5);
        let p = r.penalty(&[1.0, 2.0, 3.0]);
        // 0.1 * (1 + 4 + 9) = 1.4
        assert!((p - 1.4).abs() < 1e-12);
    }

    #[test]
    fn l2_penalty_negative_weights() {
        let mut r = make(RegularizerType::L2, 0.1, 0.5);
        let p = r.penalty(&[-1.0, -2.0, -3.0]);
        assert!((p - 1.4).abs() < 1e-12);
    }

    #[test]
    fn l2_penalty_zero_weights() {
        let mut r = make(RegularizerType::L2, 0.1, 0.5);
        let p = r.penalty(&[0.0, 0.0]);
        assert!((p - 0.0).abs() < 1e-12);
    }

    #[test]
    fn l2_penalty_single_weight() {
        let mut r = make(RegularizerType::L2, 0.5, 0.5);
        let p = r.penalty(&[4.0]);
        assert!((p - 8.0).abs() < 1e-12); // 0.5 * 16
    }

    // ------------------------------------------------------------------
    // L2 gradient
    // ------------------------------------------------------------------

    #[test]
    fn l2_gradient_correctness() {
        let mut r = make(RegularizerType::L2, 0.1, 0.5);
        let g = r.gradient(&[1.0, -2.0, 3.0]);
        // 2 * 0.1 * w
        assert!((g[0] - 0.2).abs() < 1e-12);
        assert!((g[1] - (-0.4)).abs() < 1e-12);
        assert!((g[2] - 0.6).abs() < 1e-12);
    }

    #[test]
    fn l2_gradient_zero() {
        let r = make(RegularizerType::L2, 0.5, 0.5);
        let g = r.l2_gradient(&[0.0, 0.0]);
        assert!(g.iter().all(|&v| v.abs() < 1e-12));
    }

    // ------------------------------------------------------------------
    // ElasticNet penalty
    // ------------------------------------------------------------------

    #[test]
    fn elastic_penalty_balanced() {
        let mut r = make(RegularizerType::ElasticNet, 0.1, 0.5);
        let w = [1.0, -2.0, 3.0];
        let expected = 0.5 * r.l1_penalty(&w) + 0.5 * r.l2_penalty(&w);
        let p = r.penalty(&w);
        assert!((p - expected).abs() < 1e-12);
    }

    #[test]
    fn elastic_penalty_pure_l1() {
        // alpha = 1.0 => pure L1
        let mut elastic = make(RegularizerType::ElasticNet, 0.1, 1.0);
        let mut l1 = make(RegularizerType::L1, 0.1, 0.5);
        let w = [2.0, -3.0, 0.5];
        let pe = elastic.penalty(&w);
        let pl = l1.penalty(&w);
        assert!((pe - pl).abs() < 1e-12);
    }

    #[test]
    fn elastic_penalty_pure_l2() {
        // alpha = 0.0 => pure L2
        let mut elastic = make(RegularizerType::ElasticNet, 0.1, 0.0);
        let mut l2 = make(RegularizerType::L2, 0.1, 0.5);
        let w = [2.0, -3.0, 0.5];
        let pe = elastic.penalty(&w);
        let pl = l2.penalty(&w);
        assert!((pe - pl).abs() < 1e-12);
    }

    #[test]
    fn elastic_penalty_zero_weights() {
        let mut r = make(RegularizerType::ElasticNet, 0.5, 0.3);
        let p = r.penalty(&[0.0, 0.0]);
        assert!((p - 0.0).abs() < 1e-12);
    }

    // ------------------------------------------------------------------
    // ElasticNet gradient
    // ------------------------------------------------------------------

    #[test]
    fn elastic_gradient_balanced() {
        let mut r = make(RegularizerType::ElasticNet, 0.1, 0.5);
        let w = [1.0, -2.0, 3.0];
        let l1g = r.l1_gradient(&w);
        let l2g = r.l2_gradient(&w);
        let expected: Vec<f64> = l1g
            .iter()
            .zip(l2g.iter())
            .map(|(&a, &b)| 0.5 * a + 0.5 * b)
            .collect();
        let g = r.gradient(&w);
        for (i, (&got, &exp)) in g.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-12,
                "mismatch at index {i}: got {got}, expected {exp}"
            );
        }
    }

    #[test]
    fn elastic_gradient_pure_l1() {
        let mut elastic = make(RegularizerType::ElasticNet, 0.2, 1.0);
        let mut l1 = make(RegularizerType::L1, 0.2, 0.5);
        let w = [1.0, -1.0, 0.0];
        let ge = elastic.gradient(&w);
        let gl = l1.gradient(&w);
        for (a, b) in ge.iter().zip(gl.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn elastic_gradient_pure_l2() {
        let mut elastic = make(RegularizerType::ElasticNet, 0.2, 0.0);
        let mut l2 = make(RegularizerType::L2, 0.2, 0.5);
        let w = [1.0, -1.0, 0.5];
        let ge = elastic.gradient(&w);
        let gl = l2.gradient(&w);
        for (a, b) in ge.iter().zip(gl.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    // ------------------------------------------------------------------
    // Lambda scaling
    // ------------------------------------------------------------------

    #[test]
    fn lambda_scaling_l1() {
        let r1 = make(RegularizerType::L1, 0.1, 0.5);
        let r2 = make(RegularizerType::L1, 0.2, 0.5);
        let w = [1.0, 2.0, 3.0];
        let p1 = r1.l1_penalty(&w);
        let p2 = r2.l1_penalty(&w);
        assert!((p2 / p1 - 2.0).abs() < 1e-12);
    }

    #[test]
    fn lambda_scaling_l2() {
        let r1 = make(RegularizerType::L2, 0.1, 0.5);
        let r2 = make(RegularizerType::L2, 0.3, 0.5);
        let w = [1.0, 2.0];
        let p1 = r1.l2_penalty(&w);
        let p2 = r2.l2_penalty(&w);
        assert!((p2 / p1 - 3.0).abs() < 1e-12);
    }

    // ------------------------------------------------------------------
    // Stats tracking
    // ------------------------------------------------------------------

    #[test]
    fn stats_initial() {
        let r = make(RegularizerType::L2, 0.01, 0.5);
        let s = r.stats();
        assert_eq!(s.reg_type, RegularizerType::L2);
        assert!((s.lambda - 0.01).abs() < 1e-12);
        assert_eq!(s.computations, 0);
    }

    #[test]
    fn stats_after_operations() {
        let mut r = make(RegularizerType::L1, 0.1, 0.5);
        r.penalty(&[1.0]);
        r.penalty(&[2.0]);
        r.gradient(&[3.0]);
        let s = r.stats();
        assert_eq!(s.computations, 3);
    }

    // ------------------------------------------------------------------
    // Default config
    // ------------------------------------------------------------------

    #[test]
    fn default_config() {
        let cfg = RegularizerConfig::default();
        assert_eq!(cfg.reg_type, RegularizerType::L2);
        assert!((cfg.lambda - 0.01).abs() < 1e-12);
        assert!((cfg.elastic_alpha - 0.5).abs() < 1e-12);
    }

    // ------------------------------------------------------------------
    // Edge cases
    // ------------------------------------------------------------------

    #[test]
    fn large_weight_l2() {
        let mut r = make(RegularizerType::L2, 1.0, 0.5);
        let p = r.penalty(&[1000.0]);
        assert!((p - 1_000_000.0).abs() < 1e-6);
    }

    #[test]
    fn very_small_lambda() {
        let mut r = make(RegularizerType::L1, 1e-10, 0.5);
        let p = r.penalty(&[1.0, 2.0, 3.0]);
        assert!((p - 6e-10).abs() < 1e-20);
    }

    #[test]
    fn elastic_alpha_0_25() {
        let mut r = make(RegularizerType::ElasticNet, 1.0, 0.25);
        let w = [2.0];
        // 0.25 * 1.0 * 2.0 + 0.75 * 1.0 * 4.0 = 0.5 + 3.0 = 3.5
        let p = r.penalty(&w);
        assert!((p - 3.5).abs() < 1e-12);
    }

    #[test]
    fn gradient_empty_weights() {
        let mut r = make(RegularizerType::L2, 0.1, 0.5);
        let g = r.gradient(&[]);
        assert!(g.is_empty());
    }
}
