//! Activation functions with forward and backward passes for tensor computations.
//!
//! Provides element-wise activation functions (ReLU, LeakyReLU, Sigmoid, Tanh,
//! Softmax, GELU, Swish) with both forward evaluation and gradient (backward)
//! computation.

use std::f64::consts::PI;

/// Type of activation function.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationType {
    /// Rectified Linear Unit: max(0, x)
    ReLU,
    /// Leaky ReLU: x if x > 0, alpha * x otherwise
    LeakyReLU,
    /// Logistic sigmoid: 1 / (1 + exp(-x))
    Sigmoid,
    /// Hyperbolic tangent
    Tanh,
    /// Softmax (applied across all elements)
    Softmax,
    /// Gaussian Error Linear Unit (approximate form)
    GELU,
    /// Swish / SiLU: x * sigmoid(x)
    Swish,
}

/// Configuration for an activation function.
#[derive(Debug, Clone)]
pub struct ActivationConfig {
    /// Which activation function to use.
    pub activation_type: ActivationType,
    /// Alpha parameter for LeakyReLU (default 0.01).
    pub leaky_alpha: f64,
}

impl Default for ActivationConfig {
    fn default() -> Self {
        Self {
            activation_type: ActivationType::ReLU,
            leaky_alpha: 0.01,
        }
    }
}

/// Runtime statistics for a `TensorActivation` instance.
#[derive(Debug, Clone)]
pub struct ActivationStats {
    /// The activation type in use.
    pub activation_type: ActivationType,
    /// Number of forward passes executed.
    pub forward_calls: u64,
    /// Number of backward passes executed.
    pub backward_calls: u64,
}

/// Activation layer with forward/backward support and call statistics.
pub struct TensorActivation {
    config: ActivationConfig,
    forward_calls: u64,
    backward_calls: u64,
}

impl TensorActivation {
    /// Create a new activation layer from the given configuration.
    pub fn new(config: ActivationConfig) -> Self {
        Self {
            config,
            forward_calls: 0,
            backward_calls: 0,
        }
    }

    // ------------------------------------------------------------------
    // Static helpers
    // ------------------------------------------------------------------

    /// ReLU activation: max(0, x).
    #[inline]
    pub fn relu(x: f64) -> f64 {
        if x > 0.0 {
            x
        } else {
            0.0
        }
    }

    /// Sigmoid activation: 1 / (1 + exp(-x)).
    #[inline]
    pub fn sigmoid(x: f64) -> f64 {
        if x >= 0.0 {
            let e = (-x).exp();
            1.0 / (1.0 + e)
        } else {
            let e = x.exp();
            e / (1.0 + e)
        }
    }

    /// Tanh activation (wrapper around `f64::tanh`).
    #[inline]
    pub fn tanh_act(x: f64) -> f64 {
        x.tanh()
    }

    /// GELU activation (approximate):
    /// 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    #[inline]
    pub fn gelu(x: f64) -> f64 {
        let sqrt_2_over_pi = (2.0 / PI).sqrt();
        let inner = sqrt_2_over_pi * (x + 0.044715 * x * x * x);
        0.5 * x * (1.0 + inner.tanh())
    }

    /// Swish activation: x * sigmoid(x).
    #[inline]
    pub fn swish(x: f64) -> f64 {
        x * Self::sigmoid(x)
    }

    /// Softmax over a slice of values, using the log-sum-exp trick for
    /// numerical stability.
    pub fn softmax(input: &[f64]) -> Vec<f64> {
        if input.is_empty() {
            return Vec::new();
        }
        let max_val = input.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = input.iter().map(|&x| (x - max_val).exp()).collect();
        let sum: f64 = exps.iter().sum();
        if sum == 0.0 {
            // Degenerate case – return uniform.
            let n = input.len() as f64;
            return vec![1.0 / n; input.len()];
        }
        exps.iter().map(|&e| e / sum).collect()
    }

    // ------------------------------------------------------------------
    // Forward pass
    // ------------------------------------------------------------------

    /// Apply the configured activation element-wise (Softmax across all
    /// elements).
    pub fn forward(&mut self, input: &[f64]) -> Vec<f64> {
        self.forward_calls += 1;
        match self.config.activation_type {
            ActivationType::ReLU => input.iter().map(|&x| Self::relu(x)).collect(),
            ActivationType::LeakyReLU => {
                let alpha = self.config.leaky_alpha;
                input
                    .iter()
                    .map(|&x| if x > 0.0 { x } else { alpha * x })
                    .collect()
            }
            ActivationType::Sigmoid => input.iter().map(|&x| Self::sigmoid(x)).collect(),
            ActivationType::Tanh => input.iter().map(|&x| Self::tanh_act(x)).collect(),
            ActivationType::Softmax => Self::softmax(input),
            ActivationType::GELU => input.iter().map(|&x| Self::gelu(x)).collect(),
            ActivationType::Swish => input.iter().map(|&x| Self::swish(x)).collect(),
        }
    }

    // ------------------------------------------------------------------
    // Backward pass
    // ------------------------------------------------------------------

    /// Compute the gradient of the loss w.r.t. the activation input.
    ///
    /// * `input`       – the original pre-activation values.
    /// * `grad_output` – the upstream gradient (dL/d(output)).
    ///
    /// Returns dL/d(input).
    pub fn backward(&mut self, input: &[f64], grad_output: &[f64]) -> Vec<f64> {
        self.backward_calls += 1;

        match self.config.activation_type {
            ActivationType::ReLU => input
                .iter()
                .zip(grad_output.iter())
                .map(|(&x, &g)| if x > 0.0 { g } else { 0.0 })
                .collect(),
            ActivationType::LeakyReLU => {
                let alpha = self.config.leaky_alpha;
                input
                    .iter()
                    .zip(grad_output.iter())
                    .map(|(&x, &g)| if x > 0.0 { g } else { alpha * g })
                    .collect()
            }
            ActivationType::Sigmoid => input
                .iter()
                .zip(grad_output.iter())
                .map(|(&x, &g)| {
                    let s = Self::sigmoid(x);
                    g * s * (1.0 - s)
                })
                .collect(),
            ActivationType::Tanh => input
                .iter()
                .zip(grad_output.iter())
                .map(|(&x, &g)| {
                    let t = x.tanh();
                    g * (1.0 - t * t)
                })
                .collect(),
            ActivationType::GELU => input
                .iter()
                .zip(grad_output.iter())
                .map(|(&x, &g)| g * Self::gelu_derivative(x))
                .collect(),
            ActivationType::Swish => input
                .iter()
                .zip(grad_output.iter())
                .map(|(&x, &g)| {
                    let sw = Self::swish(x);
                    let sig = Self::sigmoid(x);
                    g * (sw + sig * (1.0 - sw))
                })
                .collect(),
            ActivationType::Softmax => {
                // Jacobian-vector product for softmax:
                // dL/dx_i = s_i * (g_i - dot(g, s))
                let s = Self::softmax(input);
                let dot: f64 = grad_output
                    .iter()
                    .zip(s.iter())
                    .map(|(&g, &si)| g * si)
                    .sum();
                s.iter()
                    .zip(grad_output.iter())
                    .map(|(&si, &gi)| si * (gi - dot))
                    .collect()
            }
        }
    }

    // ------------------------------------------------------------------
    // Stats
    // ------------------------------------------------------------------

    /// Return runtime statistics.
    pub fn stats(&self) -> ActivationStats {
        ActivationStats {
            activation_type: self.config.activation_type,
            forward_calls: self.forward_calls,
            backward_calls: self.backward_calls,
        }
    }

    // ------------------------------------------------------------------
    // Private helpers
    // ------------------------------------------------------------------

    /// Approximate derivative of GELU.
    ///
    /// d/dx GELU(x) ≈ 0.5 * (1 + tanh(u)) + 0.5 * x * sech²(u) * u'
    /// where u = sqrt(2/π) * (x + 0.044715 x³), u' = sqrt(2/π) * (1 + 3*0.044715 x²)
    #[inline]
    fn gelu_derivative(x: f64) -> f64 {
        let sqrt_2_over_pi = (2.0 / PI).sqrt();
        let x3 = x * x * x;
        let u = sqrt_2_over_pi * (x + 0.044715 * x3);
        let tanh_u = u.tanh();
        let sech2_u = 1.0 - tanh_u * tanh_u;
        let u_prime = sqrt_2_over_pi * (1.0 + 3.0 * 0.044715 * x * x);
        0.5 * (1.0 + tanh_u) + 0.5 * x * sech2_u * u_prime
    }
}

// ======================================================================
// Tests
// ======================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Helper to build a TensorActivation quickly.
    fn make(ty: ActivationType) -> TensorActivation {
        TensorActivation::new(ActivationConfig {
            activation_type: ty,
            leaky_alpha: 0.01,
        })
    }

    // ------------------------------------------------------------------
    // ReLU
    // ------------------------------------------------------------------

    #[test]
    fn relu_forward_positive() {
        let mut act = make(ActivationType::ReLU);
        let out = act.forward(&[1.0, 2.0, 3.0]);
        assert_eq!(out, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn relu_forward_zeros_negatives() {
        let mut act = make(ActivationType::ReLU);
        let out = act.forward(&[-1.0, -0.5, 0.0, 0.5]);
        assert_eq!(out, vec![0.0, 0.0, 0.0, 0.5]);
    }

    #[test]
    fn relu_backward() {
        let mut act = make(ActivationType::ReLU);
        let grad = act.backward(&[-1.0, 0.0, 1.0], &[1.0, 1.0, 1.0]);
        assert_eq!(grad, vec![0.0, 0.0, 1.0]);
    }

    #[test]
    fn relu_static_helper() {
        assert_eq!(TensorActivation::relu(5.0), 5.0);
        assert_eq!(TensorActivation::relu(-3.0), 0.0);
        assert_eq!(TensorActivation::relu(0.0), 0.0);
    }

    // ------------------------------------------------------------------
    // LeakyReLU
    // ------------------------------------------------------------------

    #[test]
    fn leaky_relu_forward() {
        let mut act = make(ActivationType::LeakyReLU);
        let out = act.forward(&[-10.0, 0.0, 5.0]);
        assert!((out[0] - (-0.1)).abs() < 1e-12);
        assert_eq!(out[1], 0.0);
        assert_eq!(out[2], 5.0);
    }

    #[test]
    fn leaky_relu_backward() {
        let mut act = make(ActivationType::LeakyReLU);
        let grad = act.backward(&[-2.0, 3.0], &[1.0, 1.0]);
        assert!((grad[0] - 0.01).abs() < 1e-12);
        assert_eq!(grad[1], 1.0);
    }

    // ------------------------------------------------------------------
    // Sigmoid
    // ------------------------------------------------------------------

    #[test]
    fn sigmoid_forward_range() {
        let mut act = make(ActivationType::Sigmoid);
        let out = act.forward(&[-100.0, -1.0, 0.0, 1.0, 100.0]);
        for &v in &out {
            assert!((0.0..=1.0).contains(&v), "sigmoid out of range: {v}");
        }
        assert!((out[2] - 0.5).abs() < 1e-12, "sigmoid(0) should be 0.5");
    }

    #[test]
    fn sigmoid_backward() {
        let mut act = make(ActivationType::Sigmoid);
        let grad = act.backward(&[0.0], &[1.0]);
        // sigmoid(0) = 0.5, derivative = 0.5 * 0.5 = 0.25
        assert!((grad[0] - 0.25).abs() < 1e-12);
    }

    #[test]
    fn sigmoid_static_helper() {
        assert!((TensorActivation::sigmoid(0.0) - 0.5).abs() < 1e-12);
        assert!(TensorActivation::sigmoid(100.0) > 0.999);
        assert!(TensorActivation::sigmoid(-100.0) < 0.001);
    }

    // ------------------------------------------------------------------
    // Tanh
    // ------------------------------------------------------------------

    #[test]
    fn tanh_forward_range() {
        let mut act = make(ActivationType::Tanh);
        let out = act.forward(&[-100.0, -1.0, 0.0, 1.0, 100.0]);
        for &v in &out {
            assert!((-1.0..=1.0).contains(&v), "tanh out of range: {v}");
        }
        assert!(out[2].abs() < 1e-12, "tanh(0) should be 0");
    }

    #[test]
    fn tanh_backward() {
        let mut act = make(ActivationType::Tanh);
        let grad = act.backward(&[0.0], &[1.0]);
        // tanh(0) = 0, derivative = 1 - 0^2 = 1
        assert!((grad[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn tanh_static_helper() {
        assert!(TensorActivation::tanh_act(0.0).abs() < 1e-12);
        assert!((TensorActivation::tanh_act(1.0) - 1.0_f64.tanh()).abs() < 1e-14);
    }

    // ------------------------------------------------------------------
    // Softmax
    // ------------------------------------------------------------------

    #[test]
    fn softmax_sums_to_one() {
        let mut act = make(ActivationType::Softmax);
        let out = act.forward(&[1.0, 2.0, 3.0, 4.0]);
        let sum: f64 = out.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-12,
            "softmax sum should be 1, got {sum}"
        );
    }

    #[test]
    fn softmax_monotonicity() {
        let out = TensorActivation::softmax(&[1.0, 2.0, 3.0]);
        assert!(out[0] < out[1] && out[1] < out[2]);
    }

    #[test]
    fn softmax_backward() {
        let mut act = make(ActivationType::Softmax);
        let input = vec![1.0, 2.0, 3.0];
        let grad_out = vec![1.0, 0.0, 0.0];
        let grad = act.backward(&input, &grad_out);
        // Sum of softmax backward gradients should be 0
        let grad_sum: f64 = grad.iter().sum();
        assert!(
            grad_sum.abs() < 1e-12,
            "softmax grad sum should be ~0, got {grad_sum}"
        );
    }

    #[test]
    fn softmax_static_helper() {
        let out = TensorActivation::softmax(&[0.0, 0.0, 0.0]);
        for &v in &out {
            assert!((v - 1.0 / 3.0).abs() < 1e-12);
        }
    }

    // ------------------------------------------------------------------
    // GELU
    // ------------------------------------------------------------------

    #[test]
    fn gelu_forward_approximation() {
        let mut act = make(ActivationType::GELU);
        let out = act.forward(&[0.0, 1.0, -1.0]);
        // GELU(0) = 0
        assert!(out[0].abs() < 1e-12);
        // GELU(1) ≈ 0.8412
        assert!((out[1] - 0.8412).abs() < 0.001);
        // GELU(-1) ≈ -0.1588
        assert!((out[2] - (-0.1588)).abs() < 0.001);
    }

    #[test]
    fn gelu_backward() {
        let mut act = make(ActivationType::GELU);
        let grad = act.backward(&[0.0], &[1.0]);
        // GELU'(0) = 0.5
        assert!((grad[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn gelu_static_helper() {
        assert!(TensorActivation::gelu(0.0).abs() < 1e-12);
    }

    // ------------------------------------------------------------------
    // Swish
    // ------------------------------------------------------------------

    #[test]
    fn swish_forward() {
        let mut act = make(ActivationType::Swish);
        let out = act.forward(&[0.0, 1.0, -1.0]);
        // swish(0) = 0 * 0.5 = 0
        assert!(out[0].abs() < 1e-12);
        // swish(1) = 1 * sigmoid(1)
        let expected = TensorActivation::sigmoid(1.0);
        assert!((out[1] - expected).abs() < 1e-12);
    }

    #[test]
    fn swish_backward() {
        let mut act = make(ActivationType::Swish);
        let grad = act.backward(&[0.0], &[1.0]);
        // swish(0)=0, sigmoid(0)=0.5 => derivative = 0 + 0.5*(1-0) = 0.5
        assert!((grad[0] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn swish_static_helper() {
        assert!(TensorActivation::swish(0.0).abs() < 1e-12);
        let s5 = TensorActivation::swish(5.0);
        assert!((s5 - 5.0 * TensorActivation::sigmoid(5.0)).abs() < 1e-12);
    }

    // ------------------------------------------------------------------
    // Empty input
    // ------------------------------------------------------------------

    #[test]
    fn empty_input_forward() {
        let mut act = make(ActivationType::ReLU);
        assert!(act.forward(&[]).is_empty());
    }

    #[test]
    fn empty_input_backward() {
        let mut act = make(ActivationType::Sigmoid);
        assert!(act.backward(&[], &[]).is_empty());
    }

    #[test]
    fn empty_softmax() {
        assert!(TensorActivation::softmax(&[]).is_empty());
    }

    // ------------------------------------------------------------------
    // Stats tracking
    // ------------------------------------------------------------------

    #[test]
    fn stats_tracking() {
        let mut act = make(ActivationType::GELU);
        let s = act.stats();
        assert_eq!(s.forward_calls, 0);
        assert_eq!(s.backward_calls, 0);
        assert_eq!(s.activation_type, ActivationType::GELU);

        act.forward(&[1.0]);
        act.forward(&[2.0]);
        act.backward(&[1.0], &[1.0]);

        let s = act.stats();
        assert_eq!(s.forward_calls, 2);
        assert_eq!(s.backward_calls, 1);
    }

    // ------------------------------------------------------------------
    // Numerical gradient checks
    // ------------------------------------------------------------------

    fn numerical_gradient(act: &mut TensorActivation, x: f64, eps: f64) -> f64 {
        let mut act_plus = make(act.stats().activation_type);
        let mut act_minus = make(act.stats().activation_type);
        let f_plus = act_plus.forward(&[x + eps])[0];
        let f_minus = act_minus.forward(&[x - eps])[0];
        (f_plus - f_minus) / (2.0 * eps)
    }

    #[test]
    fn numerical_grad_relu() {
        let mut act = make(ActivationType::ReLU);
        let analytic = act.backward(&[1.0], &[1.0])[0];
        let numeric = numerical_gradient(&mut act, 1.0, 1e-5);
        assert!((analytic - numeric).abs() < 1e-4);
    }

    #[test]
    fn numerical_grad_sigmoid() {
        let mut act = make(ActivationType::Sigmoid);
        for &x in &[-2.0, -0.5, 0.0, 0.5, 2.0] {
            let analytic = act.backward(&[x], &[1.0])[0];
            let numeric = numerical_gradient(&mut act, x, 1e-5);
            assert!(
                (analytic - numeric).abs() < 1e-4,
                "sigmoid grad mismatch at x={x}: analytic={analytic}, numeric={numeric}"
            );
        }
    }

    #[test]
    fn numerical_grad_tanh() {
        let mut act = make(ActivationType::Tanh);
        for &x in &[-2.0, 0.0, 1.5] {
            let analytic = act.backward(&[x], &[1.0])[0];
            let numeric = numerical_gradient(&mut act, x, 1e-5);
            assert!(
                (analytic - numeric).abs() < 1e-4,
                "tanh grad mismatch at x={x}"
            );
        }
    }

    #[test]
    fn numerical_grad_gelu() {
        let mut act = make(ActivationType::GELU);
        for &x in &[-2.0, -0.5, 0.0, 0.5, 2.0] {
            let analytic = act.backward(&[x], &[1.0])[0];
            let numeric = numerical_gradient(&mut act, x, 1e-5);
            assert!(
                (analytic - numeric).abs() < 1e-3,
                "gelu grad mismatch at x={x}: analytic={analytic}, numeric={numeric}"
            );
        }
    }

    #[test]
    fn numerical_grad_swish() {
        let mut act = make(ActivationType::Swish);
        for &x in &[-2.0, -0.5, 0.0, 0.5, 2.0] {
            let analytic = act.backward(&[x], &[1.0])[0];
            let numeric = numerical_gradient(&mut act, x, 1e-5);
            assert!(
                (analytic - numeric).abs() < 1e-4,
                "swish grad mismatch at x={x}: analytic={analytic}, numeric={numeric}"
            );
        }
    }

    #[test]
    fn numerical_grad_leaky_relu() {
        let mut act = make(ActivationType::LeakyReLU);
        for &x in &[-2.0, 2.0] {
            let analytic = act.backward(&[x], &[1.0])[0];
            let numeric = numerical_gradient(&mut act, x, 1e-5);
            assert!(
                (analytic - numeric).abs() < 1e-4,
                "leaky_relu grad mismatch at x={x}"
            );
        }
    }

    // ------------------------------------------------------------------
    // Default config
    // ------------------------------------------------------------------

    #[test]
    fn default_config() {
        let cfg = ActivationConfig::default();
        assert_eq!(cfg.activation_type, ActivationType::ReLU);
        assert!((cfg.leaky_alpha - 0.01).abs() < 1e-14);
    }
}
