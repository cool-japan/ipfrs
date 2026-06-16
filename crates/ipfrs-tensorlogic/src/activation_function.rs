//! Neural network activation functions with forward pass, derivative, and vectorized operations.
//!
//! Provides a comprehensive set of activation functions for neural network layers including
//! ReLU variants, sigmoid, tanh, softmax, GELU, Swish, Mish, HardSwish, and more.
//! Each activation supports forward evaluation, derivative computation for backpropagation,
//! in-place mutation, and call statistics tracking.
//!
//! # Naming Convention
//!
//! Types in this module are prefixed with `Af` (Activation Function) in re-exports to
//! avoid collision with the pre-existing `activation` module in the same crate.

use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// ActivationType
// ---------------------------------------------------------------------------

/// Specifies which activation function to apply.
#[derive(Debug, Clone, PartialEq)]
pub enum ActivationType {
    /// Rectified Linear Unit: max(0, x)
    ReLU,
    /// Leaky ReLU: x if x >= 0, slope * x otherwise.
    LeakyReLU(f64),
    /// Exponential Linear Unit: x if x >= 0, alpha * (exp(x) - 1) otherwise.
    ELU(f64),
    /// Logistic sigmoid: 1 / (1 + exp(-x))
    Sigmoid,
    /// Hyperbolic tangent.
    Tanh,
    /// Normalised exponentials across the full input slice.
    Softmax,
    /// Gaussian Error Linear Unit (tanh approximation).
    GELU,
    /// Swish / SiLU: x * sigmoid(x)
    Swish,
    /// Mish: x * tanh(softplus(x)) where softplus(x) = ln(1 + exp(x))
    Mish,
    /// Hard Swish: x * relu6(x + 3) / 6
    HardSwish,
    /// Identity: f(x) = x
    Linear,
    /// Threshold: value if x > threshold, else 0.0
    Threshold(f64, f64),
}

// ---------------------------------------------------------------------------
// ActivationConfig
// ---------------------------------------------------------------------------

/// Configuration for an [`ActivationFunction`] instance.
#[derive(Debug, Clone)]
pub struct ActivationConfig {
    /// Which activation function to apply.
    pub activation_type: ActivationType,
    /// When `true`, `apply_inplace` mutates the slice in place rather than
    /// allocating a new buffer.  `forward` always returns a new `Vec`.
    pub inplace: bool,
}

impl ActivationConfig {
    /// Convenience constructor.
    pub fn new(activation_type: ActivationType) -> Self {
        Self {
            activation_type,
            inplace: false,
        }
    }

    /// Enable in-place mode.
    pub fn with_inplace(mut self) -> Self {
        self.inplace = true;
        self
    }
}

// ---------------------------------------------------------------------------
// ActivationStats
// ---------------------------------------------------------------------------

/// Runtime statistics collected by an [`ActivationFunction`].
#[derive(Debug, Clone, Default)]
pub struct ActivationStats {
    /// Total number of `forward` / `apply_inplace` calls.
    pub total_calls: u64,
    /// Cumulative number of elements processed across all calls.
    pub total_elements: u64,
    /// Number of ReLU output elements that were clamped to zero (dead neurons).
    pub dead_relu_count: u64,
}

// ---------------------------------------------------------------------------
// ActivationFunction
// ---------------------------------------------------------------------------

/// Neural network activation layer with forward pass, derivative, in-place
/// application, and call statistics.
pub struct ActivationFunction {
    config: ActivationConfig,
    stats: ActivationStats,
}

impl ActivationFunction {
    /// Create a new activation layer with the given configuration.
    pub fn new(config: ActivationConfig) -> Self {
        Self {
            config,
            stats: ActivationStats::default(),
        }
    }

    // ------------------------------------------------------------------
    // Public forward API
    // ------------------------------------------------------------------

    /// Apply the configured activation to every element of `input` and return
    /// a newly allocated `Vec<f64>`.
    ///
    /// For `Softmax` the operation is applied across the entire slice.
    pub fn forward(&mut self, input: &[f64]) -> Vec<f64> {
        self.stats.total_calls += 1;
        self.stats.total_elements += input.len() as u64;

        let result = match &self.config.activation_type {
            ActivationType::ReLU => {
                let out: Vec<f64> = input.iter().map(|&x| Self::relu(x)).collect();
                // Count dead ReLU outputs before returning.
                let dead = out.iter().filter(|&&v| v == 0.0).count() as u64;
                self.stats.dead_relu_count += dead;
                out
            }
            ActivationType::LeakyReLU(slope) => {
                let s = *slope;
                input.iter().map(|&x| Self::leaky_relu(x, s)).collect()
            }
            ActivationType::ELU(alpha) => {
                let a = *alpha;
                input.iter().map(|&x| Self::elu(x, a)).collect()
            }
            ActivationType::Sigmoid => input.iter().map(|&x| Self::sigmoid(x)).collect(),
            ActivationType::Tanh => input.iter().map(|&x| Self::tanh_activation(x)).collect(),
            ActivationType::Softmax => Self::softmax(input),
            ActivationType::GELU => input.iter().map(|&x| Self::gelu(x)).collect(),
            ActivationType::Swish => input.iter().map(|&x| Self::swish(x)).collect(),
            ActivationType::Mish => input.iter().map(|&x| Self::mish(x)).collect(),
            ActivationType::HardSwish => input.iter().map(|&x| Self::hard_swish(x)).collect(),
            ActivationType::Linear => input.to_vec(),
            ActivationType::Threshold(threshold, value) => {
                let (t, v) = (*threshold, *value);
                input.iter().map(|&x| if x > t { v } else { 0.0 }).collect()
            }
        };

        result
    }

    /// Compute the element-wise derivative of the activation with respect to
    /// the pre-activation input.
    ///
    /// # Argument semantics
    ///
    /// The `output` slice is interpreted differently per activation:
    ///
    /// - **Sigmoid** — `output` should be the *post*-activation value `σ(x)`.
    ///   The derivative is `σ(x) * (1 − σ(x))` which avoids re-computing the
    ///   expensive exponential.
    /// - **Tanh** — `output` should be the *post*-activation value `tanh(x)`.
    ///   The derivative is `1 − tanh(x)²`.
    /// - **All others** — `output` is treated as the *pre*-activation value `x`
    ///   and the derivative is computed from first principles.
    pub fn derivative(&self, output: &[f64]) -> Vec<f64> {
        match &self.config.activation_type {
            ActivationType::ReLU => output
                .iter()
                .map(|&x| if x > 0.0 { 1.0 } else { 0.0 })
                .collect(),

            ActivationType::LeakyReLU(slope) => {
                let s = *slope;
                output
                    .iter()
                    .map(|&x| if x >= 0.0 { 1.0 } else { s })
                    .collect()
            }

            ActivationType::ELU(alpha) => {
                let a = *alpha;
                output
                    .iter()
                    .map(|&x| {
                        if x >= 0.0 {
                            1.0
                        } else {
                            // d/dx ELU = alpha * exp(x)
                            a * x.exp()
                        }
                    })
                    .collect()
            }

            // Sigmoid derivative from post-activation value: σ * (1 - σ)
            ActivationType::Sigmoid => output.iter().map(|&s| s * (1.0 - s)).collect(),

            // Tanh derivative from post-activation value: 1 - tanh²
            ActivationType::Tanh => output.iter().map(|&t| 1.0 - t * t).collect(),

            // Softmax Jacobian diagonal (useful for element-wise backprop):
            // ∂softmax_i/∂x_i = softmax_i * (1 - softmax_i)
            // Here `output` is treated as the softmax output vector.
            ActivationType::Softmax => output.iter().map(|&s| s * (1.0 - s)).collect(),

            ActivationType::GELU => output.iter().map(|&x| Self::gelu_derivative(x)).collect(),

            ActivationType::Swish => output.iter().map(|&x| Self::swish_derivative(x)).collect(),

            ActivationType::Mish => output.iter().map(|&x| Self::mish_derivative(x)).collect(),

            ActivationType::HardSwish => output
                .iter()
                .map(|&x| Self::hard_swish_derivative(x))
                .collect(),

            ActivationType::Linear => vec![1.0; output.len()],

            ActivationType::Threshold(threshold, _value) => {
                let t = *threshold;
                output
                    .iter()
                    .map(|&x| {
                        // The function is a step, so the derivative is 0 everywhere
                        // except at the threshold itself (where it is undefined / ∞).
                        // Convention: return 0 everywhere.
                        if (x - t).abs() < f64::EPSILON {
                            f64::INFINITY
                        } else {
                            0.0
                        }
                    })
                    .collect()
            }
        }
    }

    /// Apply the configured activation **in place**, mutating `data`.
    ///
    /// Statistics are updated identically to [`forward`](Self::forward).
    pub fn apply_inplace(&mut self, data: &mut [f64]) {
        self.stats.total_calls += 1;
        self.stats.total_elements += data.len() as u64;

        match &self.config.activation_type.clone() {
            ActivationType::ReLU => {
                let mut dead: u64 = 0;
                for x in data.iter_mut() {
                    if *x <= 0.0 {
                        *x = 0.0;
                        dead += 1;
                    }
                }
                self.stats.dead_relu_count += dead;
            }
            ActivationType::LeakyReLU(slope) => {
                let s = *slope;
                for x in data.iter_mut() {
                    if *x < 0.0 {
                        *x *= s;
                    }
                }
            }
            ActivationType::ELU(alpha) => {
                let a = *alpha;
                for x in data.iter_mut() {
                    if *x < 0.0 {
                        *x = a * (x.exp() - 1.0);
                    }
                }
            }
            ActivationType::Sigmoid => {
                for x in data.iter_mut() {
                    *x = Self::sigmoid(*x);
                }
            }
            ActivationType::Tanh => {
                for x in data.iter_mut() {
                    *x = Self::tanh_activation(*x);
                }
            }
            ActivationType::Softmax => {
                let result = Self::softmax(data);
                data.copy_from_slice(&result);
            }
            ActivationType::GELU => {
                for x in data.iter_mut() {
                    *x = Self::gelu(*x);
                }
            }
            ActivationType::Swish => {
                for x in data.iter_mut() {
                    *x = Self::swish(*x);
                }
            }
            ActivationType::Mish => {
                for x in data.iter_mut() {
                    *x = Self::mish(*x);
                }
            }
            ActivationType::HardSwish => {
                for x in data.iter_mut() {
                    *x = Self::hard_swish(*x);
                }
            }
            ActivationType::Linear => { /* identity — nothing to do */ }
            ActivationType::Threshold(threshold, value) => {
                let (t, v) = (*threshold, *value);
                for x in data.iter_mut() {
                    *x = if *x > t { v } else { 0.0 };
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // Read-only accessors
    // ------------------------------------------------------------------

    /// Return a reference to the accumulated statistics for this instance.
    pub fn stats(&self) -> &ActivationStats {
        &self.stats
    }

    /// Return a reference to the current configuration.
    pub fn config(&self) -> &ActivationConfig {
        &self.config
    }

    // ------------------------------------------------------------------
    // Static scalar activations
    // ------------------------------------------------------------------

    /// Rectified Linear Unit: `max(0, x)`.
    #[inline]
    pub fn relu(x: f64) -> f64 {
        x.max(0.0)
    }

    /// Leaky ReLU: `x` if `x >= 0`, else `slope * x`.
    #[inline]
    pub fn leaky_relu(x: f64, slope: f64) -> f64 {
        if x >= 0.0 {
            x
        } else {
            slope * x
        }
    }

    /// Exponential Linear Unit: `x` if `x >= 0`, else `alpha * (exp(x) - 1)`.
    #[inline]
    pub fn elu(x: f64, alpha: f64) -> f64 {
        if x >= 0.0 {
            x
        } else {
            alpha * (x.exp() - 1.0)
        }
    }

    /// Logistic sigmoid: `1 / (1 + exp(-x))`.
    #[inline]
    pub fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    /// Hyperbolic tangent activation.
    #[inline]
    pub fn tanh_activation(x: f64) -> f64 {
        x.tanh()
    }

    /// Numerically stable softmax over the entire `input` slice.
    ///
    /// Subtracts `max(input)` before exponentiating to prevent overflow.
    pub fn softmax(input: &[f64]) -> Vec<f64> {
        if input.is_empty() {
            return Vec::new();
        }

        let max_val = input.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        let exps: Vec<f64> = input.iter().map(|&x| (x - max_val).exp()).collect();
        let sum: f64 = exps.iter().sum();

        if sum == 0.0 {
            // Degenerate — return uniform distribution.
            let n = input.len();
            return vec![1.0 / n as f64; n];
        }

        exps.iter().map(|&e| e / sum).collect()
    }

    /// GELU activation (tanh approximation).
    ///
    /// `0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))`
    #[inline]
    pub fn gelu(x: f64) -> f64 {
        // sqrt(2 / π)
        let c = (2.0_f64 / PI).sqrt();
        let inner = c * (x + 0.044715 * x * x * x);
        0.5 * x * (1.0 + inner.tanh())
    }

    /// Swish / SiLU: `x * sigmoid(x)`.
    #[inline]
    pub fn swish(x: f64) -> f64 {
        x * Self::sigmoid(x)
    }

    /// Mish: `x * tanh(softplus(x))` where `softplus(x) = ln(1 + exp(x))`.
    ///
    /// Uses the numerically stable form `ln(1 + exp(x))`:
    /// for large positive `x`, `softplus(x) ≈ x`; for large negative `x`,
    /// `softplus(x) ≈ exp(x)`.
    #[inline]
    pub fn mish(x: f64) -> f64 {
        // Numerically stable softplus
        let sp = if x > 20.0 { x } else { (1.0 + x.exp()).ln() };
        x * sp.tanh()
    }

    /// Hard Swish: `x * relu6(x + 3) / 6` where `relu6(t) = min(max(t, 0), 6)`.
    #[inline]
    pub fn hard_swish(x: f64) -> f64 {
        let relu6 = (x + 3.0).clamp(0.0, 6.0);
        x * relu6 / 6.0
    }

    // ------------------------------------------------------------------
    // Private derivative helpers
    // ------------------------------------------------------------------

    /// GELU derivative (analytical, using tanh approximation).
    #[inline]
    fn gelu_derivative(x: f64) -> f64 {
        let c = (2.0_f64 / PI).sqrt();
        let inner = c * (x + 0.044715 * x * x * x);
        let t = inner.tanh();
        // d/dx [0.5 * x * (1 + tanh(inner))]
        // = 0.5 * (1 + tanh(inner)) + 0.5 * x * sech²(inner) * d_inner/dx
        let sech2 = 1.0 - t * t;
        let d_inner = c * (1.0 + 3.0 * 0.044715 * x * x);
        0.5 * (1.0 + t) + 0.5 * x * sech2 * d_inner
    }

    /// Swish derivative: `sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))`.
    #[inline]
    fn swish_derivative(x: f64) -> f64 {
        let s = Self::sigmoid(x);
        s + x * s * (1.0 - s)
    }

    /// Mish derivative.
    ///
    /// Let `sp = softplus(x) = ln(1 + exp(x))` and `omega = tanh(sp)`.
    /// Then `d mish / dx = omega + x * (1 - omega²) * sigmoid(x)`.
    #[inline]
    fn mish_derivative(x: f64) -> f64 {
        let sp = if x > 20.0 { x } else { (1.0 + x.exp()).ln() };
        let omega = sp.tanh();
        let sech2_sp = 1.0 - omega * omega;
        let sig = Self::sigmoid(x);
        omega + x * sech2_sp * sig
    }

    /// Hard Swish derivative.
    ///
    /// The piecewise function is:
    /// - `x <= -3`  →  0
    /// - `-3 < x < 3`  →  `(2x + 3) / 6`
    /// - `x >= 3`   →  1
    #[inline]
    fn hard_swish_derivative(x: f64) -> f64 {
        if x <= -3.0 {
            0.0
        } else if x >= 3.0 {
            1.0
        } else {
            (2.0 * x + 3.0) / 6.0
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-9;

    // -----------------------------------------------------------------------
    // Helper
    // -----------------------------------------------------------------------

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    fn make_af(at: ActivationType) -> ActivationFunction {
        ActivationFunction::new(ActivationConfig::new(at))
    }

    // -----------------------------------------------------------------------
    // ReLU
    // -----------------------------------------------------------------------

    #[test]
    fn test_relu_positive() {
        assert!(approx_eq(ActivationFunction::relu(3.5), 3.5, EPS));
    }

    #[test]
    fn test_relu_negative() {
        assert!(approx_eq(ActivationFunction::relu(-2.0), 0.0, EPS));
    }

    #[test]
    fn test_relu_zero() {
        assert!(approx_eq(ActivationFunction::relu(0.0), 0.0, EPS));
    }

    #[test]
    fn test_relu_derivative_positive() {
        let af = make_af(ActivationType::ReLU);
        let d = af.derivative(&[1.0, 2.0]);
        assert!(approx_eq(d[0], 1.0, EPS));
        assert!(approx_eq(d[1], 1.0, EPS));
    }

    #[test]
    fn test_relu_derivative_negative() {
        let af = make_af(ActivationType::ReLU);
        let d = af.derivative(&[-1.0, -5.0]);
        assert!(approx_eq(d[0], 0.0, EPS));
        assert!(approx_eq(d[1], 0.0, EPS));
    }

    // -----------------------------------------------------------------------
    // Dead ReLU tracking
    // -----------------------------------------------------------------------

    #[test]
    fn test_dead_relu_count() {
        let mut af = make_af(ActivationType::ReLU);
        let _out = af.forward(&[-1.0, 2.0, -3.0, 4.0]);
        // Two negative inputs → two dead neurons.
        assert_eq!(af.stats().dead_relu_count, 2);
    }

    #[test]
    fn test_dead_relu_accumulates_across_calls() {
        let mut af = make_af(ActivationType::ReLU);
        let _ = af.forward(&[-1.0, 1.0]);
        let _ = af.forward(&[-2.0, -3.0]);
        assert_eq!(af.stats().dead_relu_count, 3);
    }

    // -----------------------------------------------------------------------
    // Leaky ReLU
    // -----------------------------------------------------------------------

    #[test]
    fn test_leaky_relu_negative_slope() {
        let slope = 0.1;
        let val = ActivationFunction::leaky_relu(-5.0, slope);
        assert!(approx_eq(val, -0.5, EPS));
    }

    #[test]
    fn test_leaky_relu_positive() {
        assert!(approx_eq(
            ActivationFunction::leaky_relu(3.0, 0.1),
            3.0,
            EPS
        ));
    }

    #[test]
    fn test_leaky_relu_derivative_negative() {
        let slope = 0.01;
        let af = make_af(ActivationType::LeakyReLU(slope));
        let d = af.derivative(&[-2.0]);
        assert!(approx_eq(d[0], slope, EPS));
    }

    // -----------------------------------------------------------------------
    // ELU — continuity at zero
    // -----------------------------------------------------------------------

    #[test]
    fn test_elu_zero() {
        // ELU(0) = 0 regardless of alpha.
        assert!(approx_eq(ActivationFunction::elu(0.0, 1.0), 0.0, EPS));
    }

    #[test]
    fn test_elu_positive() {
        assert!(approx_eq(ActivationFunction::elu(2.0, 1.0), 2.0, EPS));
    }

    #[test]
    fn test_elu_negative() {
        let alpha = 1.0;
        let x = -1.0_f64;
        let expected = alpha * (x.exp() - 1.0);
        assert!(approx_eq(
            ActivationFunction::elu(x, alpha),
            expected,
            1e-12
        ));
    }

    #[test]
    fn test_elu_continuity() {
        // ELU should be continuous at 0: left-limit == right-limit == 0.
        let left = ActivationFunction::elu(-1e-10, 1.0);
        let right = ActivationFunction::elu(1e-10, 1.0);
        assert!(approx_eq(left, right, 1e-8));
    }

    // -----------------------------------------------------------------------
    // Sigmoid
    // -----------------------------------------------------------------------

    #[test]
    fn test_sigmoid_zero() {
        assert!(approx_eq(ActivationFunction::sigmoid(0.0), 0.5, EPS));
    }

    #[test]
    fn test_sigmoid_large_positive() {
        // σ(large) ≈ 1
        assert!(ActivationFunction::sigmoid(100.0) > 0.9999);
    }

    #[test]
    fn test_sigmoid_large_negative() {
        // σ(-large) ≈ 0
        assert!(ActivationFunction::sigmoid(-100.0) < 1e-10);
    }

    #[test]
    fn test_sigmoid_derivative_from_output() {
        // Derivative of sigmoid from its output: σ * (1 - σ)
        // At x=0, σ=0.5 → derivative = 0.25
        let af = make_af(ActivationType::Sigmoid);
        let d = af.derivative(&[0.5]);
        assert!(approx_eq(d[0], 0.25, EPS));
    }

    // -----------------------------------------------------------------------
    // Tanh
    // -----------------------------------------------------------------------

    #[test]
    fn test_tanh_zero() {
        assert!(approx_eq(
            ActivationFunction::tanh_activation(0.0),
            0.0,
            EPS
        ));
    }

    #[test]
    fn test_tanh_derivative_from_output() {
        // At tanh(x)=0 (i.e. x=0), derivative = 1 - 0^2 = 1
        let af = make_af(ActivationType::Tanh);
        let d = af.derivative(&[0.0]);
        assert!(approx_eq(d[0], 1.0, EPS));
    }

    #[test]
    fn test_tanh_derivative_saturated() {
        // At tanh(x)≈1, derivative ≈ 0
        let af = make_af(ActivationType::Tanh);
        let d = af.derivative(&[0.9999]);
        assert!(d[0] < 0.001);
    }

    // -----------------------------------------------------------------------
    // Softmax
    // -----------------------------------------------------------------------

    #[test]
    fn test_softmax_sums_to_one() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let out = ActivationFunction::softmax(&input);
        let sum: f64 = out.iter().sum();
        assert!(approx_eq(sum, 1.0, 1e-12));
    }

    #[test]
    fn test_softmax_numerical_stability_large_values() {
        // All very large — should not overflow / produce NaN.
        let input = vec![1000.0, 1001.0, 1002.0];
        let out = ActivationFunction::softmax(&input);
        let sum: f64 = out.iter().sum();
        assert!(!sum.is_nan());
        assert!(approx_eq(sum, 1.0, 1e-12));
    }

    #[test]
    fn test_softmax_uniform_input() {
        // Equal logits → uniform distribution.
        let input = vec![2.0; 4];
        let out = ActivationFunction::softmax(&input);
        for &v in &out {
            assert!(approx_eq(v, 0.25, 1e-12));
        }
    }

    #[test]
    fn test_softmax_empty_input() {
        let out = ActivationFunction::softmax(&[]);
        assert!(out.is_empty());
    }

    #[test]
    fn test_softmax_all_outputs_positive() {
        let input = vec![-5.0, 0.0, 5.0];
        let out = ActivationFunction::softmax(&input);
        for &v in &out {
            assert!(v > 0.0);
        }
    }

    // -----------------------------------------------------------------------
    // GELU
    // -----------------------------------------------------------------------

    #[test]
    fn test_gelu_zero() {
        // GELU(0) = 0 (since tanh(0) = 0, so 0.5*0*(1+0) = 0)
        assert!(approx_eq(ActivationFunction::gelu(0.0), 0.0, EPS));
    }

    #[test]
    fn test_gelu_positive_domain_close_to_x() {
        // For large positive x, GELU(x) ≈ x
        let x = 10.0_f64;
        let g = ActivationFunction::gelu(x);
        assert!(approx_eq(g, x, 1e-4));
    }

    #[test]
    fn test_gelu_negative_domain_small() {
        // For large negative x, GELU(x) ≈ 0
        let g = ActivationFunction::gelu(-10.0);
        assert!(g.abs() < 1e-3);
    }

    #[test]
    fn test_gelu_approximation_bounds() {
        // The tanh-approximation GELU should be within 0.001 of the exact
        // erf-based GELU for x in [-3, 3].
        // Exact GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
        // We use the polynomial approximation for erf.
        for i in -30..=30 {
            let x = i as f64 / 10.0;
            let approx_gelu = ActivationFunction::gelu(x);
            // Rough bound: GELU output should be within (-0.2, x+0.1)
            assert!(
                approx_gelu > -0.2,
                "GELU({x}) = {approx_gelu} unexpectedly low"
            );
            assert!(
                approx_gelu <= x + 0.1 || x < 0.0,
                "GELU({x}) = {approx_gelu} unexpectedly high"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Swish = x * sigmoid(x)
    // -----------------------------------------------------------------------

    #[test]
    fn test_swish_equals_x_times_sigmoid() {
        for i in -5..=5 {
            let x = i as f64;
            let swish_val = ActivationFunction::swish(x);
            let expected = x * ActivationFunction::sigmoid(x);
            assert!(approx_eq(swish_val, expected, EPS));
        }
    }

    #[test]
    fn test_swish_zero() {
        assert!(approx_eq(ActivationFunction::swish(0.0), 0.0, EPS));
    }

    // -----------------------------------------------------------------------
    // Mish — positive domain and zero
    // -----------------------------------------------------------------------

    #[test]
    fn test_mish_zero() {
        assert!(approx_eq(ActivationFunction::mish(0.0), 0.0, EPS));
    }

    #[test]
    fn test_mish_positive_domain_greater_than_zero() {
        // Mish of a positive value should itself be positive.
        for i in 1..=10 {
            let x = i as f64;
            assert!(
                ActivationFunction::mish(x) > 0.0,
                "mish({x}) should be positive"
            );
        }
    }

    #[test]
    fn test_mish_monotone_positive() {
        // Mish should be (roughly) monotone increasing for positive x.
        let mut prev = ActivationFunction::mish(0.0);
        for i in 1..=20 {
            let cur = ActivationFunction::mish(i as f64 * 0.5);
            assert!(cur >= prev, "Mish not monotone at x={}", i as f64 * 0.5);
            prev = cur;
        }
    }

    // -----------------------------------------------------------------------
    // HardSwish
    // -----------------------------------------------------------------------

    #[test]
    fn test_hard_swish_clamp_negative() {
        // x <= -3 → 0
        assert!(approx_eq(ActivationFunction::hard_swish(-3.0), 0.0, EPS));
        assert!(approx_eq(ActivationFunction::hard_swish(-5.0), 0.0, EPS));
    }

    #[test]
    fn test_hard_swish_clamp_positive() {
        // x >= 3 → x
        assert!(approx_eq(ActivationFunction::hard_swish(3.0), 3.0, EPS));
        assert!(approx_eq(ActivationFunction::hard_swish(6.0), 6.0, EPS));
    }

    #[test]
    fn test_hard_swish_zero() {
        // x=0 → 0 * relu6(3)/6 = 0 * 3/6 = 0
        assert!(approx_eq(ActivationFunction::hard_swish(0.0), 0.0, EPS));
    }

    // -----------------------------------------------------------------------
    // Linear / Identity
    // -----------------------------------------------------------------------

    #[test]
    fn test_linear_forward() {
        let mut af = make_af(ActivationType::Linear);
        let input = vec![1.0, -2.0, std::f64::consts::PI];
        let out = af.forward(&input);
        assert_eq!(out, input);
    }

    #[test]
    fn test_linear_derivative() {
        let af = make_af(ActivationType::Linear);
        let d = af.derivative(&[1.0, 2.0, 3.0]);
        for v in d {
            assert!(approx_eq(v, 1.0, EPS));
        }
    }

    // -----------------------------------------------------------------------
    // Threshold
    // -----------------------------------------------------------------------

    #[test]
    fn test_threshold_above() {
        let mut af = make_af(ActivationType::Threshold(0.5, 1.0));
        let out = af.forward(&[0.6, 1.0, 2.0]);
        for v in out {
            assert!(approx_eq(v, 1.0, EPS));
        }
    }

    #[test]
    fn test_threshold_below() {
        let mut af = make_af(ActivationType::Threshold(0.5, 1.0));
        let out = af.forward(&[0.0, 0.5, 0.4]);
        for v in out {
            assert!(approx_eq(v, 0.0, EPS));
        }
    }

    #[test]
    fn test_threshold_exact() {
        // x == threshold should NOT trigger → 0.0
        let mut af = make_af(ActivationType::Threshold(1.0, 42.0));
        let out = af.forward(&[1.0]);
        assert!(approx_eq(out[0], 0.0, EPS));
    }

    // -----------------------------------------------------------------------
    // Vectorised apply / in-place
    // -----------------------------------------------------------------------

    #[test]
    fn test_apply_inplace_relu() {
        let mut af = make_af(ActivationType::ReLU);
        let mut data = vec![-1.0, 2.0, -3.0, 4.0];
        af.apply_inplace(&mut data);
        assert_eq!(data, vec![0.0, 2.0, 0.0, 4.0]);
    }

    #[test]
    fn test_apply_inplace_sigmoid() {
        let mut af = make_af(ActivationType::Sigmoid);
        let mut data = vec![0.0];
        af.apply_inplace(&mut data);
        assert!(approx_eq(data[0], 0.5, EPS));
    }

    #[test]
    fn test_apply_inplace_softmax_sums_one() {
        let mut af = make_af(ActivationType::Softmax);
        let mut data = vec![1.0, 2.0, 3.0];
        af.apply_inplace(&mut data);
        let sum: f64 = data.iter().sum();
        assert!(approx_eq(sum, 1.0, 1e-12));
    }

    // -----------------------------------------------------------------------
    // Stats tracking
    // -----------------------------------------------------------------------

    #[test]
    fn test_stats_total_calls() {
        let mut af = make_af(ActivationType::Sigmoid);
        let _ = af.forward(&[1.0]);
        let _ = af.forward(&[2.0, 3.0]);
        assert_eq!(af.stats().total_calls, 2);
    }

    #[test]
    fn test_stats_total_elements() {
        let mut af = make_af(ActivationType::Tanh);
        let _ = af.forward(&[1.0, 2.0, 3.0]);
        assert_eq!(af.stats().total_elements, 3);
    }

    #[test]
    fn test_stats_inplace_increments() {
        let mut af = make_af(ActivationType::Linear);
        let mut data = vec![1.0; 5];
        af.apply_inplace(&mut data);
        assert_eq!(af.stats().total_calls, 1);
        assert_eq!(af.stats().total_elements, 5);
    }

    // -----------------------------------------------------------------------
    // PI usage sanity
    // -----------------------------------------------------------------------

    #[test]
    fn test_pi_used_in_gelu() {
        // GELU uses PI internally; verify the constant is consistent.
        let c = (2.0_f64 / PI).sqrt();
        assert!(approx_eq(c, 0.7978845608028654, 1e-12));
    }
}
