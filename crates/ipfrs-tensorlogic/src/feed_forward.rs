//! Feedforward network layer for transformer blocks.
//!
//! This module implements the position-wise feedforward sub-layer that appears
//! in every transformer encoder/decoder block (Vaswani et al. 2017).  The
//! standard two-layer structure is:
//!
//! ```text
//! output = Linear2( Activation( Linear1( input ) ) )
//! ```
//!
//! where `Linear1` projects from `input_dim` → `hidden_dim` (typically
//! `4 × input_dim`) and `Linear2` projects back to `output_dim`.
//!
//! # Supported activations
//!
//! | Variant   | Description                                    |
//! |-----------|------------------------------------------------|
//! | `ReLU`    | max(0, x)                                      |
//! | `GELU`    | Gaussian Error Linear Unit (tanh approximation)|
//! | `SiLU`    | x · σ(x)  (also known as Swish)               |
//! | `Linear`  | Identity — no activation applied               |
//!
//! # Weight initialisation
//!
//! Weights are initialised with **He (Kaiming) normal** initialisation using a
//! custom xorshift64 PRNG with Box-Muller transform — zero external crate
//! dependency.
//!
//! # Example
//!
//! ```rust
//! use ipfrs_tensorlogic::{FeedForwardConfig, FeedForwardActivation, FeedForwardNetwork};
//!
//! let cfg = FeedForwardConfig {
//!     input_dim: 8,
//!     hidden_dim: 32,
//!     output_dim: 8,
//!     activation: FeedForwardActivation::GELU,
//!     use_bias: true,
//!     dropout_rate: 0.1,
//! };
//!
//! let mut net = FeedForwardNetwork::new(cfg, 42);
//!
//! let token = vec![1.0_f64; 8];
//! let out = net.forward(&token);
//! assert_eq!(out.len(), 8);
//! ```

use std::f64::consts::PI;

// ── Activation enum ───────────────────────────────────────────────────────────

/// Activation function applied between the two linear projections.
#[derive(Debug, Clone, PartialEq)]
pub enum FeedForwardActivation {
    /// Rectified Linear Unit: max(0, x).
    ReLU,
    /// Gaussian Error Linear Unit (tanh approximation).
    GELU,
    /// Sigmoid Linear Unit / Swish: x · σ(x).
    SiLU,
    /// No activation — passes the pre-activation values through unchanged.
    Linear,
}

// ── Configuration ─────────────────────────────────────────────────────────────

/// Configuration for [`FeedForwardNetwork`].
#[derive(Debug, Clone)]
pub struct FeedForwardConfig {
    /// Dimensionality of the input vector (and typically the output as well).
    pub input_dim: usize,
    /// Width of the hidden (intermediate) projection; commonly `4 × input_dim`.
    pub hidden_dim: usize,
    /// Dimensionality of the network output.
    pub output_dim: usize,
    /// Activation function applied after the first linear transformation.
    pub activation: FeedForwardActivation,
    /// When `true` bias vectors are allocated and applied in each layer.
    pub use_bias: bool,
    /// Conceptual dropout rate; stored for reference — no stochastic drop is
    /// applied during deterministic inference or the current forward pass.
    pub dropout_rate: f64,
}

// ── Single layer ──────────────────────────────────────────────────────────────

/// A single affine (linear) layer: weight matrix and optional bias.
///
/// `weights` has shape `out_dim × in_dim` — each row is the weight vector
/// for one output neuron.
#[derive(Debug, Clone)]
pub struct FFLayer {
    /// Weight matrix stored row-major: `weights[o][i]` = weight from input `i`
    /// to output neuron `o`.  Shape: `out_dim × in_dim`.
    pub weights: Vec<Vec<f64>>,
    /// Bias vector of length `out_dim`.  All zeros when `use_bias` is `false`.
    pub bias: Vec<f64>,
}

// ── Running statistics ────────────────────────────────────────────────────────

/// Lightweight counters accumulated across [`FeedForwardNetwork::forward`] calls.
#[derive(Debug, Clone, Default)]
pub struct FFStats {
    /// Total number of times `forward` or `forward_batch` has been invoked.
    pub total_forward_calls: u64,
    /// Cumulative number of individual tokens (vectors) that have passed
    /// through the network.
    pub total_tokens_processed: u64,
}

// ── Network ───────────────────────────────────────────────────────────────────

/// Two-layer feedforward network suitable for use as the FFN sub-layer of a
/// transformer block.
pub struct FeedForwardNetwork {
    config: FeedForwardConfig,
    /// First projection: `input_dim` → `hidden_dim`.
    layer1: FFLayer,
    /// Second projection: `hidden_dim` → `output_dim`.
    layer2: FFLayer,
    /// Internal xorshift64 PRNG state (retained for reproducible re-init).
    rng_state: u64,
    stats: FFStats,
}

// ── Implementation ────────────────────────────────────────────────────────────

impl FeedForwardNetwork {
    /// Construct a new network from `config`, initialising weights with He
    /// normal initialisation seeded by `seed`.
    pub fn new(config: FeedForwardConfig, seed: u64) -> Self {
        let mut rng = if seed == 0 { 0x853c49e6748fea9b } else { seed };

        let layer1 = Self::init_layer(
            config.input_dim,
            config.hidden_dim,
            config.use_bias,
            &mut rng,
        );
        let layer2 = Self::init_layer(
            config.hidden_dim,
            config.output_dim,
            config.use_bias,
            &mut rng,
        );

        Self {
            config,
            layer1,
            layer2,
            rng_state: rng,
            stats: FFStats::default(),
        }
    }

    /// Run a single token (flat `f64` slice of length `input_dim`) through the
    /// network and return the output vector of length `output_dim`.
    ///
    /// If the input length does not match `input_dim` the network applies
    /// whatever it can and pads/truncates gracefully — no panic.
    pub fn forward(&mut self, input: &[f64]) -> Vec<f64> {
        self.stats.total_forward_calls += 1;
        self.stats.total_tokens_processed += 1;

        // Layer 1: input_dim → hidden_dim
        let mut hidden = Self::linear_transform(input, &self.layer1);

        // Apply activation element-wise
        for v in hidden.iter_mut() {
            *v = self.apply_activation(*v);
        }

        // Layer 2: hidden_dim → output_dim
        Self::linear_transform(&hidden, &self.layer2)
    }

    /// Run a batch of tokens through the network.
    ///
    /// Returns one output vector per input token; empty input yields an empty
    /// result with no panic.
    pub fn forward_batch(&mut self, inputs: &[Vec<f64>]) -> Vec<Vec<f64>> {
        self.stats.total_forward_calls += 1;
        let token_count = inputs.len() as u64;
        self.stats.total_tokens_processed += token_count;

        inputs
            .iter()
            .map(|token| {
                // Layer 1
                let mut hidden = Self::linear_transform(token, &self.layer1);
                for v in hidden.iter_mut() {
                    *v = self.apply_activation(*v);
                }
                // Layer 2
                Self::linear_transform(&hidden, &self.layer2)
            })
            .collect()
    }

    /// Affine transformation: `output[o] = bias[o] + Σ_i weights[o][i] * input[i]`.
    ///
    /// Dimension mismatches are handled gracefully: the dot-product iterates
    /// over `min(in_dim, input.len())` elements and missing bias values default
    /// to `0.0`.
    pub fn linear_transform(input: &[f64], layer: &FFLayer) -> Vec<f64> {
        layer
            .weights
            .iter()
            .enumerate()
            .map(|(o, row)| {
                let dot: f64 = row.iter().zip(input.iter()).map(|(w, x)| w * x).sum();
                let b = layer.bias.get(o).copied().unwrap_or(0.0);
                dot + b
            })
            .collect()
    }

    /// Apply the configured activation function to a single scalar.
    #[inline]
    pub fn apply_activation(&self, x: f64) -> f64 {
        match self.config.activation {
            FeedForwardActivation::ReLU => Self::relu(x),
            FeedForwardActivation::GELU => Self::gelu(x),
            FeedForwardActivation::SiLU => Self::silu(x),
            FeedForwardActivation::Linear => x,
        }
    }

    /// Rectified Linear Unit.
    #[inline]
    pub fn relu(x: f64) -> f64 {
        x.max(0.0)
    }

    /// GELU using the tanh approximation (Hendrycks & Gimpel 2016):
    ///
    /// ```text
    /// GELU(x) ≈ 0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715 · x³)))
    /// ```
    pub fn gelu(x: f64) -> f64 {
        let c = (2.0_f64 / PI).sqrt();
        let inner = c * (x + 0.044715 * x * x * x);
        0.5 * x * (1.0 + inner.tanh())
    }

    /// Sigmoid Linear Unit (Swish): `x · σ(x)`.
    #[inline]
    pub fn silu(x: f64) -> f64 {
        x * Self::sigmoid(x)
    }

    /// Logistic sigmoid: `σ(x) = 1 / (1 + e^{−x})`.
    #[inline]
    pub fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    /// Initialise a single [`FFLayer`] with He (Kaiming) normal weights.
    ///
    /// He init draws weights from N(0, σ²) where σ = √(2 / in_dim).
    /// Bias is always zero-initialised.
    pub fn init_layer(in_dim: usize, out_dim: usize, use_bias: bool, rng: &mut u64) -> FFLayer {
        // He-init standard deviation: sqrt(2 / fan_in)
        let std_dev = if in_dim > 0 {
            (2.0_f64 / in_dim as f64).sqrt()
        } else {
            1.0
        };

        let weights: Vec<Vec<f64>> = (0..out_dim)
            .map(|_| {
                (0..in_dim)
                    .map(|_| Self::next_normal(rng) * std_dev)
                    .collect()
            })
            .collect();

        let bias = if use_bias {
            vec![0.0_f64; out_dim]
        } else {
            // Even when use_bias is false we keep a zero vector so that
            // `linear_transform` never needs to branch on this field.
            vec![0.0_f64; out_dim]
        };

        FFLayer { weights, bias }
    }

    /// Draw a standard-normal sample using the xorshift64 PRNG (Marsaglia 2003)
    /// and Box-Muller transform.
    ///
    /// Two uniform samples `u1`, `u2` ∈ (0, 1] are generated; one standard-
    /// normal deviate is returned.
    pub fn next_normal(rng: &mut u64) -> f64 {
        let u1 = Self::xorshift64(rng);
        let u2 = Self::xorshift64(rng);

        // Box-Muller: Z = √(-2 ln u1) · cos(2π u2)
        let r = (-2.0 * u1.ln()).sqrt();
        r * (2.0 * PI * u2).cos()
    }

    /// xorshift64 PRNG step (Marsaglia 2003) — returns a uniform f64 in (0, 1].
    #[inline]
    fn xorshift64(state: &mut u64) -> f64 {
        let mut x = *state;
        // Guard against zero state (would produce all-zero sequence)
        if x == 0 {
            x = 0x853c49e6748fea9b;
        }
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *state = x;

        // Map to (0, 1] — divide by 2^64, add tiny epsilon to exclude 0
        (x as f64) / (u64::MAX as f64) + f64::EPSILON
    }

    /// Reinitialise both layers using the stored `rng_state`, effectively
    /// resetting weights to a new He-normal draw that continues from where the
    /// original seed sequence left off.  Useful for experimentation without
    /// constructing a brand-new network.
    pub fn reinit_weights(&mut self) {
        let mut rng = self.rng_state;
        self.layer1 = Self::init_layer(
            self.config.input_dim,
            self.config.hidden_dim,
            self.config.use_bias,
            &mut rng,
        );
        self.layer2 = Self::init_layer(
            self.config.hidden_dim,
            self.config.output_dim,
            self.config.use_bias,
            &mut rng,
        );
        self.rng_state = rng;
    }

    /// Reference to the accumulated runtime statistics.
    pub fn stats(&self) -> &FFStats {
        &self.stats
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn make_net(
        in_dim: usize,
        hidden: usize,
        out_dim: usize,
        act: FeedForwardActivation,
    ) -> FeedForwardNetwork {
        let cfg = FeedForwardConfig {
            input_dim: in_dim,
            hidden_dim: hidden,
            output_dim: out_dim,
            activation: act,
            use_bias: true,
            dropout_rate: 0.0,
        };
        FeedForwardNetwork::new(cfg, 12345)
    }

    fn make_net_no_bias(in_dim: usize, hidden: usize, out_dim: usize) -> FeedForwardNetwork {
        let cfg = FeedForwardConfig {
            input_dim: in_dim,
            hidden_dim: hidden,
            output_dim: out_dim,
            activation: FeedForwardActivation::Linear,
            use_bias: false,
            dropout_rate: 0.0,
        };
        FeedForwardNetwork::new(cfg, 99)
    }

    // ── 1. Output shape — single token ────────────────────────────────────────

    #[test]
    fn test_forward_output_shape() {
        let mut net = make_net(8, 32, 8, FeedForwardActivation::ReLU);
        let out = net.forward(&[1.0; 8]);
        assert_eq!(out.len(), 8);
    }

    // ── 2. Output shape — batch ───────────────────────────────────────────────

    #[test]
    fn test_forward_batch_shape() {
        let mut net = make_net(4, 16, 4, FeedForwardActivation::GELU);
        let batch: Vec<Vec<f64>> = (0..5).map(|_| vec![1.0; 4]).collect();
        let out = net.forward_batch(&batch);
        assert_eq!(out.len(), 5);
        for row in &out {
            assert_eq!(row.len(), 4);
        }
    }

    // ── 3. Linear transform — known values ───────────────────────────────────

    #[test]
    fn test_linear_transform_known_values() {
        // weights = [[1, 0], [0, 1]], bias = [10, 20]
        let layer = FFLayer {
            weights: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            bias: vec![10.0, 20.0],
        };
        let input = vec![3.0, 7.0];
        let out = FeedForwardNetwork::linear_transform(&input, &layer);
        assert!((out[0] - 13.0).abs() < 1e-12, "expected 13, got {}", out[0]);
        assert!((out[1] - 27.0).abs() < 1e-12, "expected 27, got {}", out[1]);
    }

    // ── 4. Linear transform — scaling ────────────────────────────────────────

    #[test]
    fn test_linear_transform_scaling() {
        let layer = FFLayer {
            weights: vec![vec![2.0, 3.0]],
            bias: vec![0.0],
        };
        let input = vec![4.0, 5.0];
        let out = FeedForwardNetwork::linear_transform(&input, &layer);
        assert!((out[0] - 23.0).abs() < 1e-12);
    }

    // ── 5. ReLU: positive ────────────────────────────────────────────────────

    #[test]
    fn test_relu_positive() {
        assert!((FeedForwardNetwork::relu(3.5) - 3.5).abs() < 1e-12);
    }

    // ── 6. ReLU: negative ────────────────────────────────────────────────────

    #[test]
    fn test_relu_negative() {
        assert!((FeedForwardNetwork::relu(-2.0)).abs() < 1e-12);
    }

    // ── 7. ReLU: zero ────────────────────────────────────────────────────────

    #[test]
    fn test_relu_zero() {
        assert!((FeedForwardNetwork::relu(0.0)).abs() < 1e-12);
    }

    // ── 8. GELU: zero ────────────────────────────────────────────────────────

    #[test]
    fn test_gelu_zero() {
        // GELU(0) = 0
        assert!(FeedForwardNetwork::gelu(0.0).abs() < 1e-10);
    }

    // ── 9. GELU: large positive ───────────────────────────────────────────────

    #[test]
    fn test_gelu_large_positive() {
        // For large x, GELU(x) ≈ x
        let x = 10.0_f64;
        let g = FeedForwardNetwork::gelu(x);
        assert!((g - x).abs() < 1e-4, "GELU({}) = {} expected ≈ {}", x, g, x);
    }

    // ── 10. GELU: large negative ──────────────────────────────────────────────

    #[test]
    fn test_gelu_large_negative() {
        // For large negative x, GELU(x) ≈ 0
        let g = FeedForwardNetwork::gelu(-10.0);
        assert!(g.abs() < 1e-4, "GELU(-10) = {} expected ≈ 0", g);
    }

    // ── 11. SiLU: zero ───────────────────────────────────────────────────────

    #[test]
    fn test_silu_zero() {
        // SiLU(0) = 0 · σ(0) = 0 · 0.5 = 0
        assert!(FeedForwardNetwork::silu(0.0).abs() < 1e-12);
    }

    // ── 12. SiLU: positive ───────────────────────────────────────────────────

    #[test]
    fn test_silu_positive() {
        // SiLU(1) = 1 · σ(1) ≈ 0.7310585786300049
        let s = FeedForwardNetwork::silu(1.0);
        assert!((s - 0.7310585786300049).abs() < 1e-9);
    }

    // ── 13. SiLU: large positive ≈ identity ──────────────────────────────────

    #[test]
    fn test_silu_large_positive() {
        // SiLU(x) → x as x → +∞
        let x = 20.0_f64;
        let s = FeedForwardNetwork::silu(x);
        assert!((s - x).abs() < 1e-4);
    }

    // ── 14. Linear activation ─────────────────────────────────────────────────

    #[test]
    fn test_linear_activation_identity() {
        let net = make_net(4, 8, 4, FeedForwardActivation::Linear);
        // apply_activation should be identity for Linear
        assert!((net.apply_activation(3.7) - 3.7).abs() < 1e-12);
        assert!((net.apply_activation(-1.23) - (-1.23)).abs() < 1e-12);
    }

    // ── 15. Bias addition ─────────────────────────────────────────────────────

    #[test]
    fn test_bias_addition() {
        let layer = FFLayer {
            weights: vec![vec![0.0, 0.0], vec![0.0, 0.0]],
            bias: vec![5.0, -3.0],
        };
        let input = vec![1.0, 2.0];
        let out = FeedForwardNetwork::linear_transform(&input, &layer);
        assert!((out[0] - 5.0).abs() < 1e-12);
        assert!((out[1] - (-3.0)).abs() < 1e-12);
    }

    // ── 16. He init — weights non-zero ───────────────────────────────────────

    #[test]
    fn test_he_init_weights_nonzero() {
        let mut rng = 42_u64;
        let layer = FeedForwardNetwork::init_layer(8, 16, true, &mut rng);
        let nonzero = layer.weights.iter().flatten().any(|&w| w.abs() > 1e-12);
        assert!(nonzero, "All weights were zero — He init failed");
    }

    // ── 17. He init — bias is zero ────────────────────────────────────────────

    #[test]
    fn test_he_init_bias_zero() {
        let mut rng = 42_u64;
        let layer = FeedForwardNetwork::init_layer(8, 16, true, &mut rng);
        for b in &layer.bias {
            assert!(b.abs() < 1e-30, "Bias should be zero-initialised");
        }
    }

    // ── 18. He init — weight variance ─────────────────────────────────────────

    #[test]
    fn test_he_init_variance_property() {
        // Variance of He-normal weights ≈ 2 / fan_in
        let fan_in = 64_usize;
        let fan_out = 256_usize;
        let mut rng = 7654321_u64;
        let layer = FeedForwardNetwork::init_layer(fan_in, fan_out, true, &mut rng);

        let all: Vec<f64> = layer.weights.into_iter().flatten().collect();
        let n = all.len() as f64;
        let mean = all.iter().sum::<f64>() / n;
        let variance = all.iter().map(|w| (w - mean).powi(2)).sum::<f64>() / n;
        let expected_var = 2.0 / fan_in as f64;

        // Allow 50 % relative tolerance — empirical sampling noise
        assert!(
            (variance - expected_var).abs() / expected_var < 0.5,
            "Variance {:.4} too far from expected {:.4}",
            variance,
            expected_var
        );
    }

    // ── 19. Single token forward — output is finite ───────────────────────────

    #[test]
    fn test_single_token_forward_finite() {
        let mut net = make_net(16, 64, 16, FeedForwardActivation::GELU);
        let token: Vec<f64> = (0..16).map(|i| i as f64 * 0.1).collect();
        let out = net.forward(&token);
        for (i, v) in out.iter().enumerate() {
            assert!(v.is_finite(), "output[{}] = {} is not finite", i, v);
        }
    }

    // ── 20. Sequential batch — each result matches individual forward ─────────

    #[test]
    fn test_sequential_batch_matches_individual() {
        // Use a fresh net for each mode to ensure identical RNG state influence.
        // Because forward() updates stats but not weights, results must be equal.
        let mut net = make_net(4, 8, 4, FeedForwardActivation::SiLU);
        let tokens: Vec<Vec<f64>> = vec![
            vec![1.0, 0.0, -1.0, 0.5],
            vec![0.0, 1.0, 0.0, -1.0],
            vec![0.5, 0.5, 0.5, 0.5],
        ];

        let batch_out = net.forward_batch(&tokens);

        // Reset stats to compare only outputs
        let mut net2 = make_net(4, 8, 4, FeedForwardActivation::SiLU);
        let individual: Vec<Vec<f64>> = tokens.iter().map(|t| net2.forward(t)).collect();

        for (b, ind) in batch_out.iter().zip(individual.iter()) {
            for (bv, iv) in b.iter().zip(ind.iter()) {
                assert!(
                    (bv - iv).abs() < 1e-12,
                    "batch vs individual mismatch: {} vs {}",
                    bv,
                    iv
                );
            }
        }
    }

    // ── 21. Stats tracking — forward_calls ───────────────────────────────────

    #[test]
    fn test_stats_forward_calls() {
        let mut net = make_net(4, 8, 4, FeedForwardActivation::ReLU);
        assert_eq!(net.stats().total_forward_calls, 0);
        net.forward(&[0.0; 4]);
        net.forward(&[1.0; 4]);
        assert_eq!(net.stats().total_forward_calls, 2);
    }

    // ── 22. Stats tracking — tokens processed ────────────────────────────────

    #[test]
    fn test_stats_tokens_processed() {
        let mut net = make_net(4, 8, 4, FeedForwardActivation::ReLU);
        let batch: Vec<Vec<f64>> = (0..7).map(|_| vec![0.0; 4]).collect();
        net.forward_batch(&batch);
        // forward_batch counts all tokens
        assert_eq!(net.stats().total_tokens_processed, 7);
    }

    // ── 23. Stats tracking — mixed forward and batch ──────────────────────────

    #[test]
    fn test_stats_mixed_forward_and_batch() {
        let mut net = make_net(4, 8, 4, FeedForwardActivation::Linear);
        net.forward(&[0.0; 4]); // +1 call, +1 token
        net.forward_batch(&vec![vec![0.0; 4]; 3]); // +1 call, +3 tokens
        assert_eq!(net.stats().total_forward_calls, 2);
        assert_eq!(net.stats().total_tokens_processed, 4);
    }

    // ── 24. Zero input ────────────────────────────────────────────────────────

    #[test]
    fn test_zero_input_with_bias() {
        // For zero input, output = activation( bias1 ) projected through layer2.
        // With zero-initialised biases the hidden layer is all-zero, so after any
        // activation the output should also be all-zero.
        let mut net = make_net(4, 8, 4, FeedForwardActivation::ReLU);
        let out = net.forward(&[0.0; 4]);
        // All biases are zero, so hidden = 0 after ReLU = 0, output = all zeros.
        for v in &out {
            assert!((v).abs() < 1e-30, "expected 0, got {}", v);
        }
    }

    // ── 25. Unit input ────────────────────────────────────────────────────────

    #[test]
    fn test_unit_input_finite() {
        let mut net = make_net(8, 32, 8, FeedForwardActivation::SiLU);
        let out = net.forward(&[1.0; 8]);
        for v in &out {
            assert!(v.is_finite());
        }
    }

    // ── 26. Dimension mismatch — short input (graceful) ───────────────────────

    #[test]
    fn test_short_input_graceful() {
        // Provide a shorter-than-expected input; should not panic.
        let mut net = make_net(8, 16, 8, FeedForwardActivation::ReLU);
        let out = net.forward(&[1.0; 4]); // only 4 of 8 expected inputs
        assert_eq!(out.len(), 8, "output dim should still be 8");
        for v in &out {
            assert!(v.is_finite());
        }
    }

    // ── 27. Dimension mismatch — zero-dim input ───────────────────────────────

    #[test]
    fn test_empty_input_graceful() {
        let mut net = make_net(4, 8, 4, FeedForwardActivation::GELU);
        let out = net.forward(&[]);
        // With empty input all dot products are zero, so output == biases == 0
        assert_eq!(out.len(), 4);
        for v in &out {
            assert!(v.is_finite());
        }
    }

    // ── 28. No-bias layer ─────────────────────────────────────────────────────

    #[test]
    fn test_no_bias_zero_input_is_zero() {
        let mut net = make_net_no_bias(4, 8, 4);
        let out = net.forward(&[0.0; 4]);
        for v in &out {
            assert!(v.abs() < 1e-30);
        }
    }

    // ── 29. Sigmoid basic properties ──────────────────────────────────────────

    #[test]
    fn test_sigmoid_properties() {
        assert!((FeedForwardNetwork::sigmoid(0.0) - 0.5).abs() < 1e-12);
        assert!(FeedForwardNetwork::sigmoid(100.0) > 0.999);
        assert!(FeedForwardNetwork::sigmoid(-100.0) < 0.001);
    }

    // ── 30. FeedForwardActivation equality ───────────────────────────────────

    #[test]
    fn test_activation_enum_equality() {
        assert_eq!(FeedForwardActivation::ReLU, FeedForwardActivation::ReLU);
        assert_ne!(FeedForwardActivation::ReLU, FeedForwardActivation::GELU);
        assert_ne!(FeedForwardActivation::SiLU, FeedForwardActivation::Linear);
    }

    // ── 31. Empty batch ───────────────────────────────────────────────────────

    #[test]
    fn test_empty_batch() {
        let mut net = make_net(4, 8, 4, FeedForwardActivation::ReLU);
        let out = net.forward_batch(&[]);
        assert!(out.is_empty());
    }

    // ── 32. Weight matrix shape from init_layer ───────────────────────────────

    #[test]
    fn test_init_layer_shape() {
        let mut rng = 1_u64;
        let layer = FeedForwardNetwork::init_layer(6, 12, true, &mut rng);
        assert_eq!(layer.weights.len(), 12, "out_dim rows");
        for row in &layer.weights {
            assert_eq!(row.len(), 6, "in_dim cols");
        }
        assert_eq!(layer.bias.len(), 12);
    }
}
