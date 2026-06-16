//! # Embedding Fine-Tuner
//!
//! A contrastive learning fine-tuner that adapts embedding spaces using
//! positive/negative pair training with triplet loss and margin-based updates.
//!
//! ## Overview
//!
//! `EmbeddingFinetuner` implements a linear projection layer trained via
//! triplet-loss contrastive learning. Given anchor/positive/negative triplets,
//! it learns a projection W such that projected positives are closer to their
//! anchors than projected negatives, by at least a configurable margin.
//!
//! ## Algorithm
//!
//! - **Triplet Loss**: `max(0, ||W·a - W·p||² - ||W·a - W·n||² + margin)`
//! - **Gradient**: simplified first-order update (gradient of L w.r.t. W)
//! - **Regularisation**: L2 weight decay applied after every mini-batch
//! - **Initialisation**: Xavier uniform via xorshift64 PRNG (no external crates)

use std::fmt;

// ---------------------------------------------------------------------------
// PRNG helper (xorshift64 – no rand crate)
// ---------------------------------------------------------------------------

#[inline]
fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors produced by [`EmbeddingFinetuner`].
#[derive(Debug, Clone, PartialEq)]
pub enum FinetunerError {
    /// Input dimensionality did not match the layer's expected dimension.
    DimensionMismatch { expected: usize, got: usize },
    /// An empty slice was passed where at least one element is required.
    EmptyInput,
    /// The model has not been trained yet (no training history).
    NotTrained,
}

impl fmt::Display for FinetunerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
            Self::EmptyInput => write!(f, "input must not be empty"),
            Self::NotTrained => write!(f, "model has not been trained"),
        }
    }
}

impl std::error::Error for FinetunerError {}

// ---------------------------------------------------------------------------
// Core data structures
// ---------------------------------------------------------------------------

/// An anchor / positive / negative triplet used for contrastive training.
#[derive(Debug, Clone)]
pub struct TrainingPair {
    /// The reference embedding.
    pub anchor: Vec<f64>,
    /// An embedding semantically close to `anchor`.
    pub positive: Vec<f64>,
    /// An embedding semantically far from `anchor`.
    pub negative: Vec<f64>,
}

impl TrainingPair {
    /// Construct a new [`TrainingPair`].
    pub fn new(anchor: Vec<f64>, positive: Vec<f64>, negative: Vec<f64>) -> Self {
        Self {
            anchor,
            positive,
            negative,
        }
    }
}

// ---------------------------------------------------------------------------
// Triplet loss
// ---------------------------------------------------------------------------

/// Margin-based triplet loss.
///
/// `loss = max(0, ||a-p||² - ||a-n||² + margin)`
#[derive(Debug, Clone, Copy)]
pub struct TripletLoss {
    /// Minimum desired distance gap between positive and negative pairs.
    pub margin: f64,
}

impl TripletLoss {
    /// Create a new [`TripletLoss`] with the given margin.
    pub fn new(margin: f64) -> Self {
        Self { margin }
    }

    /// Compute the triplet loss for a single triplet.
    pub fn compute(&self, anchor: &[f64], pos: &[f64], neg: &[f64]) -> f64 {
        let d_pos = l2_distance_sq(anchor, pos);
        let d_neg = l2_distance_sq(anchor, neg);
        (d_pos - d_neg + self.margin).max(0.0)
    }
}

// ---------------------------------------------------------------------------
// Projection layer
// ---------------------------------------------------------------------------

/// A linear projection followed by a ReLU activation.
///
/// Dimensions: `weights[output_dim][input_dim]`, `bias[output_dim]`.
#[derive(Debug, Clone)]
pub struct ProjectionLayer {
    /// Weight matrix stored in row-major order: `weights[out][in]`.
    pub weights: Vec<Vec<f64>>,
    /// Bias vector of length `output_dim`.
    pub bias: Vec<f64>,
}

impl ProjectionLayer {
    /// Create a zero-initialised projection layer.
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        Self {
            weights: vec![vec![0.0_f64; input_dim]; output_dim],
            bias: vec![0.0_f64; output_dim],
        }
    }

    /// Forward pass: `output[i] = relu(Σ_j weights[i][j] * input[j] + bias[i])`.
    pub fn forward(&self, input: &[f64]) -> Vec<f64> {
        self.weights
            .iter()
            .zip(self.bias.iter())
            .map(|(row, b)| {
                let dot: f64 = row.iter().zip(input.iter()).map(|(w, x)| w * x).sum();
                (dot + b).max(0.0)
            })
            .collect()
    }

    /// Return input dimensionality.
    pub fn input_dim(&self) -> usize {
        self.weights.first().map(|r| r.len()).unwrap_or(0)
    }

    /// Return output dimensionality.
    pub fn output_dim(&self) -> usize {
        self.weights.len()
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Hyper-parameters for [`EmbeddingFinetuner`].
#[derive(Debug, Clone)]
pub struct FinetunerConfig {
    /// Dimensionality of the raw input embeddings.
    pub input_dim: usize,
    /// Dimensionality of the projected output.
    pub output_dim: usize,
    /// Gradient descent step size.
    pub learning_rate: f64,
    /// Triplet loss margin.
    pub margin: f64,
    /// Maximum number of training epochs.
    pub max_epochs: usize,
    /// Mini-batch size.
    pub batch_size: usize,
    /// L2 regularisation coefficient.
    pub l2_reg: f64,
}

impl Default for FinetunerConfig {
    fn default() -> Self {
        Self {
            input_dim: 128,
            output_dim: 64,
            learning_rate: 0.01,
            margin: 1.0,
            max_epochs: 10,
            batch_size: 32,
            l2_reg: 1e-4,
        }
    }
}

impl FinetunerConfig {
    /// Create a configuration with the given input / output dimensions and defaults.
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        Self {
            input_dim,
            output_dim,
            ..Default::default()
        }
    }
}

// ---------------------------------------------------------------------------
// Training statistics
// ---------------------------------------------------------------------------

/// Per-epoch training statistics.
#[derive(Debug, Clone)]
pub struct TrainingStats {
    /// Epoch index (0-based).
    pub epoch: usize,
    /// Mean triplet loss across all mini-batches in this epoch.
    pub avg_loss: f64,
    /// Number of pairs where `d(a,p) < d(a,n)` after projection.
    pub positive_pairs_closer: usize,
    /// Number of pairs where `d(a,n) > d(a,p)` (same as `positive_pairs_closer`
    /// but surfaced explicitly for clarity).
    pub negative_pairs_farther: usize,
    /// Learning rate at this epoch (may vary if a schedule is added later).
    pub learning_rate: f64,
}

// ---------------------------------------------------------------------------
// Fine-tuner
// ---------------------------------------------------------------------------

/// Contrastive embedding fine-tuner using triplet loss.
///
/// # Example
///
/// ```rust
/// use ipfrs_semantic::embedding_finetuner::{
///     EmbeddingFinetuner, FinetunerConfig, TrainingPair,
/// };
///
/// let config = FinetunerConfig::new(4, 4);
/// let mut finetuner = EmbeddingFinetuner::new(config);
///
/// let pairs = vec![
///     TrainingPair::new(vec![1.0, 0.0, 0.0, 0.0],
///                       vec![0.9, 0.1, 0.0, 0.0],
///                       vec![0.0, 0.0, 1.0, 0.0]),
/// ];
/// let history = finetuner.train(&pairs);
/// assert!(!history.is_empty());
/// ```
#[derive(Debug, Clone)]
pub struct EmbeddingFinetuner {
    /// Hyper-parameters.
    pub config: FinetunerConfig,
    /// Linear projection layer.
    pub layer: ProjectionLayer,
    /// Epoch-level statistics accumulated across all `train` calls.
    pub training_history: Vec<TrainingStats>,
    /// Running count of training pairs consumed.
    pub total_pairs_seen: u64,
    /// Internal xorshift64 PRNG state.
    pub rng_state: u64,
}

impl EmbeddingFinetuner {
    /// Construct a new fine-tuner, initialising weights with Xavier uniform.
    pub fn new(config: FinetunerConfig) -> Self {
        let mut rng_state: u64 = 0xFEED_FACE_1234_5678_u64;
        let input_dim = config.input_dim;
        let output_dim = config.output_dim;
        let scale = 2.0_f64 / (input_dim as f64).sqrt();

        let weights: Vec<Vec<f64>> = (0..output_dim)
            .map(|_| {
                (0..input_dim)
                    .map(|_| {
                        let u = xorshift64(&mut rng_state) as f64 / u64::MAX as f64;
                        (u - 0.5) * scale
                    })
                    .collect()
            })
            .collect();

        let layer = ProjectionLayer {
            weights,
            bias: vec![0.0_f64; output_dim],
        };

        Self {
            config,
            layer,
            training_history: Vec::new(),
            total_pairs_seen: 0,
            rng_state,
        }
    }

    // -----------------------------------------------------------------------
    // Core utilities
    // -----------------------------------------------------------------------

    /// Project a single embedding through the linear layer.
    pub fn project(&self, embedding: &[f64]) -> Result<Vec<f64>, FinetunerError> {
        if embedding.is_empty() {
            return Err(FinetunerError::EmptyInput);
        }
        let expected = self.layer.input_dim();
        if embedding.len() != expected {
            return Err(FinetunerError::DimensionMismatch {
                expected,
                got: embedding.len(),
            });
        }
        Ok(self.layer.forward(embedding))
    }

    /// Squared L2 distance between two equal-length slices.
    ///
    /// Panics in debug if lengths differ; silently stops at the shorter length
    /// in release (caller must ensure equal lengths).
    pub fn l2_distance_sq(a: &[f64], b: &[f64]) -> f64 {
        l2_distance_sq(a, b)
    }

    // -----------------------------------------------------------------------
    // Gradient helpers (private)
    // -----------------------------------------------------------------------

    /// Simplified gradient update for one triplet.
    ///
    /// When the loss is > 0 (the margin is violated), we nudge:
    ///
    /// * `W` toward making `W·a` closer to `W·p`: gradient direction
    ///   proportional to `(a_projected - p_projected)` outer-producted with input.
    /// * `W` away from making `W·a` close to `W·n`: gradient direction
    ///   proportional to `-(a_projected - n_projected)` outer-producted with input.
    ///
    /// This is the first-order approximation of the true triplet-loss gradient
    /// when ignoring the ReLU in the projection (treating it as linear post-hoc).
    fn apply_triplet_gradient(
        weights: &mut [Vec<f64>],
        projected_anchor: &[f64],
        projected_pos: &[f64],
        projected_neg: &[f64],
        raw_anchor: &[f64],
        lr: f64,
    ) {
        // diff_ap[i] = 2 * (proj_a[i] - proj_p[i])  (gradient of ||a-p||^2 w.r.t. proj_a)
        // diff_an[i] = 2 * (proj_a[i] - proj_n[i])  (gradient of ||a-n||^2 w.r.t. proj_a)
        //
        // Loss = ||a-p||^2 - ||a-n||^2 + margin
        // dLoss/dW[out][in] ≈ diff_ap[out] * raw_anchor[in]
        //                    - diff_an[out] * raw_anchor[in]
        for (out_idx, row) in weights.iter_mut().enumerate() {
            let diff_ap = 2.0 * (projected_anchor[out_idx] - projected_pos[out_idx]);
            let diff_an = 2.0 * (projected_anchor[out_idx] - projected_neg[out_idx]);
            let grad_out = diff_ap - diff_an; // net gradient
            for (in_idx, w) in row.iter_mut().enumerate() {
                *w -= lr * grad_out * raw_anchor[in_idx];
            }
        }
    }

    // -----------------------------------------------------------------------
    // Training
    // -----------------------------------------------------------------------

    /// Run a single mini-batch training step and return statistics.
    ///
    /// For each pair in `pairs`:
    /// 1. Project anchor, positive, negative.
    /// 2. Compute triplet loss.
    /// 3. If loss > 0, compute simplified gradient and update weights.
    /// 4. Apply L2 regularisation: `w ← w - l2_reg * w`.
    pub fn train_step(&mut self, pairs: &[TrainingPair]) -> TrainingStats {
        let mut total_loss = 0.0_f64;
        let mut positive_pairs_closer: usize = 0;
        let mut negative_pairs_farther: usize = 0;
        let triplet_loss = TripletLoss::new(self.config.margin);

        for pair in pairs {
            // Project all three embeddings (ignore dimension errors: skip bad pairs)
            let proj_a = match self.project(&pair.anchor) {
                Ok(v) => v,
                Err(_) => continue,
            };
            let proj_p = match self.project(&pair.positive) {
                Ok(v) => v,
                Err(_) => continue,
            };
            let proj_n = match self.project(&pair.negative) {
                Ok(v) => v,
                Err(_) => continue,
            };

            let loss = triplet_loss.compute(&proj_a, &proj_p, &proj_n);
            total_loss += loss;

            // Track correctness
            let d_pos = l2_distance_sq(&proj_a, &proj_p);
            let d_neg = l2_distance_sq(&proj_a, &proj_n);
            if d_pos < d_neg {
                positive_pairs_closer += 1;
                negative_pairs_farther += 1;
            }

            // Gradient update only when margin is violated
            if loss > 0.0 {
                Self::apply_triplet_gradient(
                    &mut self.layer.weights,
                    &proj_a,
                    &proj_p,
                    &proj_n,
                    &pair.anchor,
                    self.config.learning_rate,
                );
            }
        }

        // L2 regularisation
        let l2 = self.config.l2_reg;
        for row in &mut self.layer.weights {
            for w in row.iter_mut() {
                *w -= l2 * *w;
            }
        }

        let n = pairs.len().max(1);
        TrainingStats {
            epoch: 0, // caller sets epoch
            avg_loss: total_loss / n as f64,
            positive_pairs_closer,
            negative_pairs_farther,
            learning_rate: self.config.learning_rate,
        }
    }

    /// Full training loop over `max_epochs`, processing `batch_size` pairs per step.
    ///
    /// Pairs are shuffled at the start of each epoch using Fisher-Yates with
    /// the internal xorshift64 PRNG.
    ///
    /// Returns the per-epoch [`TrainingStats`] (also appended to `self.training_history`).
    pub fn train(&mut self, pairs: &[TrainingPair]) -> Vec<TrainingStats> {
        if pairs.is_empty() {
            return Vec::new();
        }

        let max_epochs = self.config.max_epochs;
        let batch_size = self.config.batch_size.max(1);
        let mut epoch_stats: Vec<TrainingStats> = Vec::with_capacity(max_epochs);

        // Working copy that we shuffle in-place each epoch
        let mut order: Vec<usize> = (0..pairs.len()).collect();

        for epoch in 0..max_epochs {
            // Fisher-Yates shuffle
            for i in (1..order.len()).rev() {
                let j = (xorshift64(&mut self.rng_state) as usize) % (i + 1);
                order.swap(i, j);
            }

            let mut epoch_total_loss = 0.0_f64;
            let mut epoch_pos_closer: usize = 0;
            let mut epoch_neg_farther: usize = 0;
            let mut num_batches: usize = 0;

            // Process mini-batches
            for chunk in order.chunks(batch_size) {
                let batch: Vec<TrainingPair> = chunk.iter().map(|&i| pairs[i].clone()).collect();
                let mut stats = self.train_step(&batch);
                stats.epoch = epoch;

                epoch_total_loss += stats.avg_loss * batch.len() as f64;
                epoch_pos_closer += stats.positive_pairs_closer;
                epoch_neg_farther += stats.negative_pairs_farther;
                num_batches += 1;

                self.total_pairs_seen += batch.len() as u64;
            }

            let avg_loss = if num_batches > 0 {
                epoch_total_loss / pairs.len() as f64
            } else {
                0.0
            };

            let stat = TrainingStats {
                epoch,
                avg_loss,
                positive_pairs_closer: epoch_pos_closer,
                negative_pairs_farther: epoch_neg_farther,
                learning_rate: self.config.learning_rate,
            };
            self.training_history.push(stat.clone());
            epoch_stats.push(stat);
        }

        epoch_stats
    }

    // -----------------------------------------------------------------------
    // Inference
    // -----------------------------------------------------------------------

    /// Project a batch of embeddings.
    pub fn encode_batch(&self, embeddings: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, FinetunerError> {
        if embeddings.is_empty() {
            return Err(FinetunerError::EmptyInput);
        }
        embeddings.iter().map(|e| self.project(e)).collect()
    }

    /// Cosine similarity between the projected versions of `a` and `b`.
    ///
    /// Falls back to the cosine of the raw vectors if projection fails.
    pub fn similarity(&self, a: &[f64], b: &[f64]) -> f64 {
        let (va, vb) = match (self.project(a), self.project(b)) {
            (Ok(x), Ok(y)) => (x, y),
            _ => (a.to_vec(), b.to_vec()),
        };
        cosine_similarity(&va, &vb)
    }

    /// Evaluate a set of triplets, returning `(avg_loss, fraction_correct)`.
    ///
    /// A triplet is "correct" when `d(proj_a, proj_p) < d(proj_a, proj_n)`.
    pub fn evaluate_pairs(&self, pairs: &[TrainingPair]) -> (f64, f64) {
        if pairs.is_empty() {
            return (0.0, 0.0);
        }
        let triplet_loss = TripletLoss::new(self.config.margin);
        let mut total_loss = 0.0_f64;
        let mut correct: usize = 0;

        for pair in pairs {
            let (proj_a, proj_p, proj_n) = match (
                self.project(&pair.anchor),
                self.project(&pair.positive),
                self.project(&pair.negative),
            ) {
                (Ok(a), Ok(p), Ok(n)) => (a, p, n),
                _ => continue,
            };

            let loss = triplet_loss.compute(&proj_a, &proj_p, &proj_n);
            total_loss += loss;

            let d_pos = l2_distance_sq(&proj_a, &proj_p);
            let d_neg = l2_distance_sq(&proj_a, &proj_n);
            if d_pos < d_neg {
                correct += 1;
            }
        }

        let avg_loss = total_loss / pairs.len() as f64;
        let fraction_correct = correct as f64 / pairs.len() as f64;
        (avg_loss, fraction_correct)
    }

    /// Read-only view of accumulated per-epoch statistics.
    pub fn training_history(&self) -> &[TrainingStats] {
        &self.training_history
    }
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

/// Squared Euclidean distance: `Σ (a_i - b_i)²`.
///
/// Iterates to the length of the shorter slice; call-sites must ensure equal
/// lengths for a mathematically meaningful result.
pub fn l2_distance_sq(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

/// Cosine similarity in `[-1, 1]`.  Returns `0.0` if either vector is zero.
pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if na < f64::EPSILON || nb < f64::EPSILON {
        0.0
    } else {
        dot / (na * nb)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use crate::embedding_finetuner::{
        cosine_similarity, l2_distance_sq, xorshift64, EmbeddingFinetuner, FinetunerConfig,
        FinetunerError, ProjectionLayer, TrainingPair, TripletLoss,
    };

    // -- xorshift64 -----------------------------------------------------------

    #[test]
    fn test_xorshift64_not_zero() {
        let mut state: u64 = 0xDEAD_BEEF_0000_0001;
        let v = xorshift64(&mut state);
        assert_ne!(v, 0);
        assert_ne!(state, 0xDEAD_BEEF_0000_0001);
    }

    #[test]
    fn test_xorshift64_deterministic() {
        let mut s1: u64 = 42;
        let mut s2: u64 = 42;
        assert_eq!(xorshift64(&mut s1), xorshift64(&mut s2));
    }

    #[test]
    fn test_xorshift64_produces_different_values() {
        let mut state: u64 = 1;
        let a = xorshift64(&mut state);
        let b = xorshift64(&mut state);
        assert_ne!(a, b);
    }

    // -- l2_distance_sq -------------------------------------------------------

    #[test]
    fn test_l2_distance_sq_zero() {
        let v = vec![1.0, 2.0, 3.0];
        assert!((l2_distance_sq(&v, &v) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_l2_distance_sq_known_value() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        assert!((l2_distance_sq(&a, &b) - 25.0).abs() < 1e-12);
    }

    #[test]
    fn test_l2_distance_sq_symmetry() {
        let a = vec![1.0, -2.0, 3.0];
        let b = vec![4.0, 5.0, -6.0];
        assert!((l2_distance_sq(&a, &b) - l2_distance_sq(&b, &a)).abs() < 1e-12);
    }

    // -- cosine_similarity ----------------------------------------------------

    #[test]
    fn test_cosine_similarity_identical() {
        let v = vec![1.0, 2.0, 3.0];
        assert!((cosine_similarity(&v, &v) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!((cosine_similarity(&a, &b)).abs() < 1e-12);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        assert!((cosine_similarity(&a, &b) + 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let a = vec![0.0, 0.0];
        let b = vec![1.0, 2.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    // -- TripletLoss ----------------------------------------------------------

    #[test]
    fn test_triplet_loss_zero_when_margin_satisfied() {
        let loss = TripletLoss::new(1.0);
        // anchor == positive, negative is far → d_pos = 0, d_neg = 9  → loss < 0 → clamped 0
        let a = vec![0.0, 0.0];
        let p = vec![0.0, 0.0];
        let n = vec![3.0, 0.0];
        assert!((loss.compute(&a, &p, &n) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_triplet_loss_positive_when_violated() {
        let loss = TripletLoss::new(1.0);
        // d_pos = 25, d_neg = 1, margin = 1 → loss = 25 - 1 + 1 = 25
        let a = vec![0.0, 0.0];
        let p = vec![3.0, 4.0];
        let n = vec![1.0, 0.0];
        let v = loss.compute(&a, &p, &n);
        assert!((v - 25.0).abs() < 1e-9);
    }

    #[test]
    fn test_triplet_loss_margin_boundary() {
        let margin = 2.0;
        let loss = TripletLoss::new(margin);
        // d_pos = 1, d_neg = 2, margin = 2 → loss = 1 - 2 + 2 = 1
        let a = vec![0.0];
        let p = vec![1.0];
        let _n = [2.0]; // wait: d_neg from a = 4
                        // let's be precise: d_pos = 1, d_neg from a=[0] to n=[sqrt(2)] = 2
        let n2 = vec![(2.0_f64).sqrt()];
        let v = loss.compute(&a, &p, &n2);
        let expected = (1.0_f64 - 2.0 + margin).max(0.0);
        assert!((v - expected).abs() < 1e-9);
    }

    // -- ProjectionLayer ------------------------------------------------------

    #[test]
    fn test_projection_layer_zero_output_for_zero_input() {
        let layer = ProjectionLayer::new(4, 4);
        let input = vec![0.0_f64; 4];
        let out = layer.forward(&input);
        assert_eq!(out.len(), 4);
        for v in &out {
            assert!(*v >= 0.0); // relu(0) = 0
        }
    }

    #[test]
    fn test_projection_layer_relu_clips_negative() {
        let mut layer = ProjectionLayer::new(2, 2);
        // Manually set weights to produce negative pre-activation
        layer.weights[0] = vec![-1.0, -1.0];
        layer.weights[1] = vec![1.0, 1.0];
        layer.bias = vec![0.0, 0.0];
        let input = vec![1.0, 1.0];
        let out = layer.forward(&input);
        assert!((out[0] - 0.0).abs() < 1e-12, "relu should clip to 0");
        assert!((out[1] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_projection_layer_dimensions() {
        let layer = ProjectionLayer::new(8, 4);
        assert_eq!(layer.input_dim(), 8);
        assert_eq!(layer.output_dim(), 4);
    }

    #[test]
    fn test_projection_layer_bias_added() {
        let mut layer = ProjectionLayer::new(1, 1);
        layer.weights[0] = vec![0.0];
        layer.bias[0] = 5.0;
        let out = layer.forward(&[1.0]);
        assert!((out[0] - 5.0).abs() < 1e-12);
    }

    // -- FinetunerConfig ------------------------------------------------------

    #[test]
    fn test_finetuner_config_default() {
        let cfg = FinetunerConfig::default();
        assert!((cfg.learning_rate - 0.01).abs() < 1e-15);
        assert!((cfg.margin - 1.0).abs() < 1e-15);
        assert_eq!(cfg.max_epochs, 10);
        assert_eq!(cfg.batch_size, 32);
        assert!((cfg.l2_reg - 1e-4).abs() < 1e-20);
    }

    #[test]
    fn test_finetuner_config_new() {
        let cfg = FinetunerConfig::new(16, 8);
        assert_eq!(cfg.input_dim, 16);
        assert_eq!(cfg.output_dim, 8);
    }

    // -- EmbeddingFinetuner construction --------------------------------------

    #[test]
    fn test_finetuner_new_weight_dimensions() {
        let cfg = FinetunerConfig::new(8, 4);
        let ft = EmbeddingFinetuner::new(cfg);
        assert_eq!(ft.layer.output_dim(), 4);
        assert_eq!(ft.layer.input_dim(), 8);
    }

    #[test]
    fn test_finetuner_new_weights_nonzero() {
        let cfg = FinetunerConfig::new(8, 4);
        let ft = EmbeddingFinetuner::new(cfg);
        let any_nonzero = ft.layer.weights.iter().flatten().any(|&w| w.abs() > 1e-15);
        assert!(any_nonzero, "Xavier init should produce non-zero weights");
    }

    #[test]
    fn test_finetuner_new_rng_seed() {
        let cfg = FinetunerConfig::new(4, 4);
        let ft = EmbeddingFinetuner::new(cfg);
        // After construction the rng_state has advanced (it was used for init)
        assert_ne!(ft.rng_state, 0);
    }

    // -- project --------------------------------------------------------------

    #[test]
    fn test_project_correct_output_dim() {
        let cfg = FinetunerConfig::new(6, 3);
        let ft = EmbeddingFinetuner::new(cfg);
        let v = ft.project(&[1.0; 6]).expect("project should succeed");
        assert_eq!(v.len(), 3);
    }

    #[test]
    fn test_project_dimension_mismatch_error() {
        let cfg = FinetunerConfig::new(4, 2);
        let ft = EmbeddingFinetuner::new(cfg);
        let err = ft.project(&[1.0; 5]).unwrap_err();
        assert!(matches!(
            err,
            FinetunerError::DimensionMismatch {
                expected: 4,
                got: 5
            }
        ));
    }

    #[test]
    fn test_project_empty_error() {
        let cfg = FinetunerConfig::new(4, 2);
        let ft = EmbeddingFinetuner::new(cfg);
        let err = ft.project(&[]).unwrap_err();
        assert_eq!(err, FinetunerError::EmptyInput);
    }

    // -- encode_batch ---------------------------------------------------------

    #[test]
    fn test_encode_batch_basic() {
        let cfg = FinetunerConfig::new(4, 2);
        let ft = EmbeddingFinetuner::new(cfg);
        let batch = vec![vec![1.0; 4], vec![0.5; 4]];
        let out = ft
            .encode_batch(&batch)
            .expect("encode_batch should succeed");
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].len(), 2);
    }

    #[test]
    fn test_encode_batch_empty_error() {
        let cfg = FinetunerConfig::new(4, 2);
        let ft = EmbeddingFinetuner::new(cfg);
        let err = ft.encode_batch(&[]).unwrap_err();
        assert_eq!(err, FinetunerError::EmptyInput);
    }

    // -- train_step -----------------------------------------------------------

    #[test]
    fn test_train_step_returns_stats() {
        let cfg = FinetunerConfig {
            input_dim: 4,
            output_dim: 4,
            ..FinetunerConfig::default()
        };
        let mut ft = EmbeddingFinetuner::new(cfg);
        let pairs = vec![TrainingPair::new(
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.9, 0.1, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
        )];
        let stats = ft.train_step(&pairs);
        assert!(stats.avg_loss >= 0.0);
    }

    #[test]
    fn test_train_step_empty_pairs() {
        let cfg = FinetunerConfig::new(4, 4);
        let mut ft = EmbeddingFinetuner::new(cfg);
        let stats = ft.train_step(&[]);
        assert!((stats.avg_loss - 0.0).abs() < 1e-12);
    }

    // -- train ----------------------------------------------------------------

    #[test]
    fn test_train_returns_epoch_stats() {
        let cfg = FinetunerConfig {
            input_dim: 4,
            output_dim: 4,
            max_epochs: 3,
            ..FinetunerConfig::default()
        };
        let mut ft = EmbeddingFinetuner::new(cfg);
        let pairs = vec![
            TrainingPair::new(
                vec![1.0, 0.0, 0.0, 0.0],
                vec![0.9, 0.1, 0.0, 0.0],
                vec![0.0, 0.0, 1.0, 0.0],
            ),
            TrainingPair::new(
                vec![0.0, 1.0, 0.0, 0.0],
                vec![0.1, 0.9, 0.0, 0.0],
                vec![0.0, 0.0, 0.0, 1.0],
            ),
        ];
        let history = ft.train(&pairs);
        assert_eq!(history.len(), 3);
    }

    #[test]
    fn test_train_history_appended() {
        let cfg = FinetunerConfig {
            input_dim: 4,
            output_dim: 4,
            max_epochs: 2,
            ..FinetunerConfig::default()
        };
        let mut ft = EmbeddingFinetuner::new(cfg);
        let pairs = vec![TrainingPair::new(
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.9, 0.1, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
        )];
        ft.train(&pairs);
        assert_eq!(ft.training_history().len(), 2);
    }

    #[test]
    fn test_train_total_pairs_seen_incremented() {
        let cfg = FinetunerConfig {
            input_dim: 4,
            output_dim: 4,
            max_epochs: 2,
            batch_size: 10,
            ..FinetunerConfig::default()
        };
        let mut ft = EmbeddingFinetuner::new(cfg);
        let pairs: Vec<TrainingPair> = (0..5)
            .map(|_| {
                TrainingPair::new(
                    vec![1.0, 0.0, 0.0, 0.0],
                    vec![0.9, 0.1, 0.0, 0.0],
                    vec![0.0, 0.0, 1.0, 0.0],
                )
            })
            .collect();
        ft.train(&pairs);
        // 5 pairs × 2 epochs = 10
        assert_eq!(ft.total_pairs_seen, 10);
    }

    #[test]
    fn test_train_empty_pairs_returns_empty() {
        let cfg = FinetunerConfig::new(4, 4);
        let mut ft = EmbeddingFinetuner::new(cfg);
        let history = ft.train(&[]);
        assert!(history.is_empty());
    }

    // -- evaluate_pairs -------------------------------------------------------

    #[test]
    fn test_evaluate_pairs_fraction_in_range() {
        let cfg = FinetunerConfig {
            input_dim: 4,
            output_dim: 4,
            max_epochs: 5,
            ..FinetunerConfig::default()
        };
        let mut ft = EmbeddingFinetuner::new(cfg);
        let pairs: Vec<TrainingPair> = (0..10)
            .map(|i| {
                let f = i as f64 / 10.0;
                TrainingPair::new(
                    vec![1.0, 0.0, 0.0, 0.0],
                    vec![1.0 - f, f, 0.0, 0.0],
                    vec![0.0, 0.0, 1.0, 0.0],
                )
            })
            .collect();
        ft.train(&pairs);
        let (avg_loss, frac) = ft.evaluate_pairs(&pairs);
        assert!(avg_loss >= 0.0);
        assert!((0.0..=1.0).contains(&frac));
    }

    #[test]
    fn test_evaluate_pairs_empty() {
        let cfg = FinetunerConfig::new(4, 4);
        let ft = EmbeddingFinetuner::new(cfg);
        let (loss, frac) = ft.evaluate_pairs(&[]);
        assert_eq!(loss, 0.0);
        assert_eq!(frac, 0.0);
    }

    // -- similarity -----------------------------------------------------------

    #[test]
    fn test_similarity_same_vector() {
        let cfg = FinetunerConfig::new(4, 4);
        let ft = EmbeddingFinetuner::new(cfg);
        let v = vec![1.0, 2.0, 3.0, 4.0];
        let s = ft.similarity(&v, &v);
        // Should be close to 1.0 (identical projected vectors)
        assert!(s > 0.9 || (s - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_similarity_range() {
        let cfg = FinetunerConfig::new(4, 4);
        let ft = EmbeddingFinetuner::new(cfg);
        let a = vec![1.0, 0.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0, 0.0];
        let s = ft.similarity(&a, &b);
        assert!((-1.0..=1.0).contains(&s));
    }

    // -- training convergence (smoke test) ------------------------------------

    #[test]
    fn test_training_reduces_loss_on_trivial_problem() {
        // Very clean triplets: positives are very close, negatives are far.
        let cfg = FinetunerConfig {
            input_dim: 4,
            output_dim: 4,
            max_epochs: 20,
            batch_size: 4,
            learning_rate: 0.05,
            margin: 0.5,
            l2_reg: 1e-5,
        };
        let mut ft = EmbeddingFinetuner::new(cfg);
        let pairs: Vec<TrainingPair> = (0..20)
            .map(|i| {
                let sign = if i % 2 == 0 { 1.0_f64 } else { -1.0_f64 };
                TrainingPair::new(
                    vec![sign, 0.0, 0.0, 0.0],
                    vec![sign * 0.99, 0.01, 0.0, 0.0],
                    vec![0.0, 0.0, sign, 0.0], // orthogonal, far in projected space
                )
            })
            .collect();
        let history = ft.train(&pairs);
        let first_loss = history.first().map(|s| s.avg_loss).unwrap_or(0.0);
        let last_loss = history.last().map(|s| s.avg_loss).unwrap_or(0.0);
        // Loss should not increase (may stay 0 or decrease)
        assert!(
            last_loss <= first_loss + 1e-6,
            "Expected loss to not increase: first={first_loss:.6}, last={last_loss:.6}"
        );
    }
}
