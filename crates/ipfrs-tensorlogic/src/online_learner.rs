//! Online / incremental learning algorithms for streaming data.
//!
//! Implements three production-grade online learning algorithms:
//!
//! * **Perceptron** — classic binary classifier; updates weights only on mispredictions.
//! * **Passive-Aggressive (PA-I)** — margin-based update with a soft constraint
//!   (`C` parameter) that controls the trade-off between aggressiveness and
//!   passiveness.
//! * **SGD with Momentum** — stochastic gradient descent with configurable
//!   momentum, learning rate, and L2 regularisation.
//!
//! All algorithms share a unified [`OnlineLearner`] interface that tracks
//! running statistics (total updates, accuracy, average loss, weight norm).
//!
//! # Examples
//!
//! ```rust
//! use ipfrs_tensorlogic::online_learner::{
//!     OnlineLearner, OnlineAlgorithm, OlLossFunction, TrainingSample,
//! };
//!
//! let mut learner = OnlineLearner::new(
//!     OnlineAlgorithm::Perceptron,
//!     2,
//!     OlLossFunction::Hinge,
//! );
//!
//! let sample = TrainingSample { features: vec![1.0, 0.5], label: 1.0 };
//! learner.update(&sample).expect("example: should succeed in docs");
//! let class = learner.predict_class(&[1.0, 0.5]).expect("example: should succeed in docs");
//! assert!(class == 1 || class == -1);
//! ```

use std::fmt;
use thiserror::Error;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can be raised by [`OnlineLearner`] operations.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum LearnerError {
    /// Feature vector dimensionality does not match the learner.
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    /// An empty input (zero-length feature vector or empty sample slice) was
    /// provided where non-empty input is required.
    #[error("empty input")]
    EmptyInput,

    /// A label value was provided that is invalid for the chosen algorithm
    /// (e.g. a value other than ±1.0 for binary classification).
    #[error("invalid label: {label} — binary classifiers expect +1.0 or -1.0")]
    InvalidLabel { label: f64 },
}

// ---------------------------------------------------------------------------
// Core enumerations
// ---------------------------------------------------------------------------

/// Online learning algorithm selection.
#[derive(Debug, Clone, PartialEq)]
pub enum OnlineAlgorithm {
    /// Classic Perceptron binary classifier.
    ///
    /// Update rule (on misprediction only):
    /// ```text
    /// w[i] += label * x[i]
    /// bias  += label
    /// ```
    Perceptron,

    /// Passive-Aggressive PA-I update.
    ///
    /// ```text
    /// loss = max(0, 1 - label * score)
    /// tau  = loss / (||x||² + 1 / (2 * C))
    /// w[i] += tau * label * x[i]
    /// bias  += tau * label
    /// ```
    PassiveAggressive {
        /// Aggressiveness parameter.  Larger values → more aggressive updates.
        c: f64,
    },

    /// Stochastic gradient descent with momentum and L2 regularisation.
    ///
    /// ```text
    /// velocity[i] = momentum * velocity[i] - lr * (grad[i] + l2_reg * w[i])
    /// w[i]       += velocity[i]
    /// bias       -= lr * (-label)
    /// ```
    SgdMomentum {
        /// Learning rate (step size).
        lr: f64,
        /// Momentum coefficient ∈ [0, 1).
        momentum: f64,
        /// L2 weight-decay coefficient.
        l2_reg: f64,
    },
}

/// Loss function used for computing per-sample losses and SGD gradients.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OlLossFunction {
    /// `max(0, 1 − label · score)`
    Hinge,
    /// `max(0, 1 − label · score)²`
    SquaredHinge,
    /// `ln(1 + exp(−label · score))` — numerically stable via log-sum-exp.
    LogLoss,
}

impl fmt::Display for OlLossFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Hinge => write!(f, "Hinge"),
            Self::SquaredHinge => write!(f, "SquaredHinge"),
            Self::LogLoss => write!(f, "LogLoss"),
        }
    }
}

// ---------------------------------------------------------------------------
// Training sample
// ---------------------------------------------------------------------------

/// A single labelled training example for online learning.
///
/// For binary classification the label **must** be `+1.0` or `−1.0`.
/// For regression the label may be any finite `f64`.
#[derive(Debug, Clone, PartialEq)]
pub struct TrainingSample {
    /// Input feature vector.
    pub features: Vec<f64>,
    /// Target label.  Binary classifiers expect ±1.0.
    pub label: f64,
}

impl TrainingSample {
    /// Construct a new training sample.
    pub fn new(features: Vec<f64>, label: f64) -> Self {
        Self { features, label }
    }

    /// Return `true` if the label is a valid binary classification label (±1.0).
    pub fn is_valid_binary_label(&self) -> bool {
        (self.label - 1.0).abs() < f64::EPSILON || (self.label + 1.0).abs() < f64::EPSILON
    }
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// Running statistics tracked by [`OnlineLearner`] across all updates and
/// predictions.
#[derive(Debug, Clone, PartialEq)]
pub struct OnlineLearnerStats {
    /// Total number of `update()` calls performed.
    pub total_updates: u64,
    /// Number of `predict_class()` calls that returned the correct label.
    pub correct_predictions: u64,
    /// Total number of `predict_class()` calls.
    pub total_predictions: u64,
    /// Running average of per-update losses.
    pub avg_loss: f64,
    /// L2 norm of the weight vector at the time `stats()` was called.
    pub weight_norm: f64,
}

impl Default for OnlineLearnerStats {
    fn default() -> Self {
        Self {
            total_updates: 0,
            correct_predictions: 0,
            total_predictions: 0,
            avg_loss: 0.0,
            weight_norm: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Internal running-average accumulator (Welford online algorithm)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default)]
struct RunningMean {
    count: u64,
    mean: f64,
}

impl RunningMean {
    fn update(&mut self, value: f64) {
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
    }

    fn value(&self) -> f64 {
        self.mean
    }
}

// ---------------------------------------------------------------------------
// Main learner struct
// ---------------------------------------------------------------------------

/// Online / incremental learner supporting Perceptron, Passive-Aggressive, and
/// SGD-with-Momentum algorithms.
///
/// The learner maintains a weight vector `w ∈ ℝᵈ` and a scalar `bias`, updated
/// sample-by-sample via the selected [`OnlineAlgorithm`].
#[derive(Debug, Clone)]
pub struct OnlineLearner {
    /// The update algorithm in use.
    pub algorithm: OnlineAlgorithm,
    /// Current weight vector.
    pub weights: Vec<f64>,
    /// Scalar bias term.
    pub bias: f64,
    /// Dimensionality (number of features).
    pub dims: usize,
    /// Loss function for computing per-sample losses.
    pub loss_fn: OlLossFunction,
    /// Velocity buffer for SGD-with-Momentum (zero for other algorithms).
    pub velocity: Vec<f64>,

    // Internal stats tracking
    running_loss: RunningMean,
    total_updates: u64,
    correct_predictions: u64,
    total_predictions: u64,
}

impl OnlineLearner {
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    /// Create a new [`OnlineLearner`] with zero-initialised weights.
    ///
    /// # Arguments
    ///
    /// * `algorithm` — update rule to apply on each `update()` call.
    /// * `dims` — feature dimensionality; all input vectors must have
    ///   exactly `dims` elements.
    /// * `loss_fn` — loss function used for reporting and SGD gradient
    ///   computation.
    ///
    /// # Panics
    ///
    /// Does not panic; returns a well-formed `OnlineLearner` even for `dims == 0`.
    pub fn new(algorithm: OnlineAlgorithm, dims: usize, loss_fn: OlLossFunction) -> Self {
        Self {
            algorithm,
            weights: vec![0.0_f64; dims],
            bias: 0.0,
            dims,
            loss_fn,
            velocity: vec![0.0_f64; dims],
            running_loss: RunningMean::default(),
            total_updates: 0,
            correct_predictions: 0,
            total_predictions: 0,
        }
    }

    // -----------------------------------------------------------------------
    // Prediction
    // -----------------------------------------------------------------------

    /// Compute the raw decision score: `dot(weights, features) + bias`.
    ///
    /// # Errors
    ///
    /// Returns [`LearnerError::EmptyInput`] if `features` is empty when
    /// `dims > 0`, or [`LearnerError::DimensionMismatch`] if
    /// `features.len() != dims`.
    pub fn predict(&self, features: &[f64]) -> Result<f64, LearnerError> {
        self.check_dims(features)?;
        Ok(dot(&self.weights, features) + self.bias)
    }

    /// Return the predicted class (`+1` or `−1`) for `features`.
    ///
    /// The class is the sign of [`predict`](Self::predict).  A score of
    /// exactly zero is classified as `+1`.
    ///
    /// This method also updates the internal prediction statistics.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`predict`](Self::predict).
    pub fn predict_class(&mut self, features: &[f64]) -> Result<i32, LearnerError> {
        let score = self.predict(features)?;
        self.total_predictions += 1;
        Ok(if score >= 0.0 { 1 } else { -1 })
    }

    /// A non-mutating variant of [`predict_class`](Self::predict_class) that
    /// does **not** update internal prediction statistics.
    ///
    /// Useful for evaluation loops where you want to call `accuracy()` later
    /// without double-counting.
    pub fn classify(&self, features: &[f64]) -> Result<i32, LearnerError> {
        let score = self.predict(features)?;
        Ok(if score >= 0.0 { 1 } else { -1 })
    }

    // -----------------------------------------------------------------------
    // Loss computation
    // -----------------------------------------------------------------------

    /// Compute the loss for a given `(score, label)` pair using the learner's
    /// configured [`OlLossFunction`].
    ///
    /// | Loss          | Formula                                        |
    /// |---------------|------------------------------------------------|
    /// | Hinge         | `max(0, 1 − label · score)`                    |
    /// | SquaredHinge  | `max(0, 1 − label · score)²`                   |
    /// | LogLoss       | `ln(1 + exp(−label · score))` (stable)         |
    pub fn loss(&self, score: f64, label: f64) -> f64 {
        compute_loss(self.loss_fn, score, label)
    }

    // -----------------------------------------------------------------------
    // Online update
    // -----------------------------------------------------------------------

    /// Perform a single online update for `sample` and return the pre-update
    /// loss.
    ///
    /// # Errors
    ///
    /// * [`LearnerError::EmptyInput`] — `sample.features` is empty but
    ///   `dims > 0`.
    /// * [`LearnerError::DimensionMismatch`] — feature length ≠ `dims`.
    /// * [`LearnerError::InvalidLabel`] — label is not ±1.0 for Perceptron or
    ///   Passive-Aggressive (binary classifiers).
    pub fn update(&mut self, sample: &TrainingSample) -> Result<f64, LearnerError> {
        self.check_dims(&sample.features)?;

        // Binary classifiers require ±1.0 labels.
        match &self.algorithm {
            OnlineAlgorithm::Perceptron | OnlineAlgorithm::PassiveAggressive { .. } => {
                if !is_binary_label(sample.label) {
                    return Err(LearnerError::InvalidLabel {
                        label: sample.label,
                    });
                }
            }
            OnlineAlgorithm::SgdMomentum { .. } => {}
        }

        let score = dot(&self.weights, &sample.features) + self.bias;
        let loss = compute_loss(self.loss_fn, score, sample.label);

        // Clone algorithm to avoid borrow issues
        let algo = self.algorithm.clone();
        match algo {
            OnlineAlgorithm::Perceptron => {
                self.update_perceptron(sample.label, &sample.features, score);
            }
            OnlineAlgorithm::PassiveAggressive { c } => {
                self.update_pa(sample.label, &sample.features, score, c);
            }
            OnlineAlgorithm::SgdMomentum {
                lr,
                momentum,
                l2_reg,
            } => {
                self.update_sgd(sample.label, &sample.features, score, lr, momentum, l2_reg);
            }
        }

        self.running_loss.update(loss);
        self.total_updates += 1;
        Ok(loss)
    }

    /// Perform online updates for a batch of samples, returning the per-sample
    /// losses in the same order as `samples`.
    ///
    /// Equivalent to calling [`update`](Self::update) in sequence.
    ///
    /// # Errors
    ///
    /// Returns the first error encountered, if any.
    pub fn batch_update(&mut self, samples: &[TrainingSample]) -> Result<Vec<f64>, LearnerError> {
        if samples.is_empty() {
            return Err(LearnerError::EmptyInput);
        }
        let mut losses = Vec::with_capacity(samples.len());
        for sample in samples {
            losses.push(self.update(sample)?);
        }
        Ok(losses)
    }

    // -----------------------------------------------------------------------
    // Evaluation
    // -----------------------------------------------------------------------

    /// Compute the fraction of `samples` correctly classified without updating
    /// weights.
    ///
    /// Classification is performed via [`classify`](Self::classify) so the
    /// internal `total_predictions` counter is **not** incremented.
    ///
    /// # Errors
    ///
    /// * [`LearnerError::EmptyInput`] — `samples` is empty.
    /// * Propagates dimension/label errors from `classify`.
    pub fn accuracy(&self, samples: &[TrainingSample]) -> Result<f64, LearnerError> {
        if samples.is_empty() {
            return Err(LearnerError::EmptyInput);
        }
        let mut correct = 0usize;
        for s in samples {
            let predicted = self.classify(&s.features)?;
            let expected = if s.label >= 0.0 { 1_i32 } else { -1_i32 };
            if predicted == expected {
                correct += 1;
            }
        }
        Ok(correct as f64 / samples.len() as f64)
    }

    // -----------------------------------------------------------------------
    // Maintenance
    // -----------------------------------------------------------------------

    /// Reset weights, bias, velocity, and all accumulated statistics to zero.
    pub fn reset(&mut self) {
        self.weights.fill(0.0);
        self.bias = 0.0;
        self.velocity.fill(0.0);
        self.running_loss = RunningMean::default();
        self.total_updates = 0;
        self.correct_predictions = 0;
        self.total_predictions = 0;
    }

    /// Compute the L2 norm of the weight vector: `√(Σ wᵢ²)`.
    pub fn l2_norm(&self) -> f64 {
        self.weights.iter().map(|w| w * w).sum::<f64>().sqrt()
    }

    /// Snapshot current training statistics.
    pub fn stats(&self) -> OnlineLearnerStats {
        OnlineLearnerStats {
            total_updates: self.total_updates,
            correct_predictions: self.correct_predictions,
            total_predictions: self.total_predictions,
            avg_loss: self.running_loss.value(),
            weight_norm: self.l2_norm(),
        }
    }

    // -----------------------------------------------------------------------
    // Private update helpers
    // -----------------------------------------------------------------------

    fn update_perceptron(&mut self, label: f64, features: &[f64], score: f64) {
        // Only update on misprediction: label * score ≤ 0
        if label * score <= 0.0 {
            for (w, &x) in self.weights.iter_mut().zip(features.iter()) {
                *w += label * x;
            }
            self.bias += label;
        }
    }

    fn update_pa(&mut self, label: f64, features: &[f64], score: f64, c: f64) {
        // Hinge loss (always used for PA update regardless of loss_fn setting)
        let margin = label * score;
        let hinge = (1.0 - margin).max(0.0);

        if hinge == 0.0 {
            // Already in the margin — passive (no update)
            return;
        }

        let sq_norm: f64 = features.iter().map(|x| x * x).sum();
        // PA-I: tau = hinge / (||x||² + 1/(2C))
        let denom = sq_norm + 1.0 / (2.0 * c);
        let tau = hinge / denom;

        for (w, &x) in self.weights.iter_mut().zip(features.iter()) {
            *w += tau * label * x;
        }
        self.bias += tau * label;
    }

    fn update_sgd(
        &mut self,
        label: f64,
        features: &[f64],
        score: f64,
        lr: f64,
        momentum: f64,
        l2_reg: f64,
    ) {
        // Subgradient of hinge loss w.r.t. score:
        //   if margin < 1  →  -label  (we're inside the margin)
        //   if margin >= 1 →   0.0    (correct & outside margin — no grad)
        // For LogLoss, use the logistic gradient: -label * sigmoid(-label*score)
        let grad_score = match self.loss_fn {
            OlLossFunction::Hinge | OlLossFunction::SquaredHinge => {
                let margin = label * score;
                if margin < 1.0 {
                    -label
                } else {
                    0.0
                }
            }
            OlLossFunction::LogLoss => {
                // d/d_score ln(1 + exp(-y*s)) = -y * sigma(-y*s)
                let neg_margin = -(label * score);
                let sigma = stable_sigmoid(neg_margin);
                -label * sigma
            }
        };

        // Update weight velocity and weights
        for (i, &xi) in features.iter().enumerate().take(self.dims) {
            let grad_w = grad_score * xi + l2_reg * self.weights[i];
            self.velocity[i] = momentum * self.velocity[i] - lr * grad_w;
            self.weights[i] += self.velocity[i];
        }

        // Bias does not get L2 regularisation (standard practice)
        self.bias -= lr * grad_score;
    }

    // -----------------------------------------------------------------------
    // Validation helper
    // -----------------------------------------------------------------------

    fn check_dims(&self, features: &[f64]) -> Result<(), LearnerError> {
        if self.dims == 0 && features.is_empty() {
            return Ok(());
        }
        if features.is_empty() {
            return Err(LearnerError::EmptyInput);
        }
        if features.len() != self.dims {
            return Err(LearnerError::DimensionMismatch {
                expected: self.dims,
                got: features.len(),
            });
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Additional evaluation helpers
    // -----------------------------------------------------------------------

    /// Compute the average loss over a slice of samples without updating weights.
    ///
    /// # Errors
    ///
    /// Returns [`LearnerError::EmptyInput`] if `samples` is empty, or
    /// propagates dimension errors.
    pub fn average_loss(&self, samples: &[TrainingSample]) -> Result<f64, LearnerError> {
        if samples.is_empty() {
            return Err(LearnerError::EmptyInput);
        }
        let total: f64 = samples
            .iter()
            .map(|s| {
                let score = dot(&self.weights, &s.features) + self.bias;
                compute_loss(self.loss_fn, score, s.label)
            })
            .sum();
        Ok(total / samples.len() as f64)
    }

    /// Compute per-sample losses over `samples` without updating weights.
    ///
    /// # Errors
    ///
    /// Returns [`LearnerError::EmptyInput`] if `samples` is empty.
    pub fn evaluate_losses(&self, samples: &[TrainingSample]) -> Result<Vec<f64>, LearnerError> {
        if samples.is_empty() {
            return Err(LearnerError::EmptyInput);
        }
        samples
            .iter()
            .map(|s| {
                self.check_dims(&s.features)?;
                let score = dot(&self.weights, &s.features) + self.bias;
                Ok(compute_loss(self.loss_fn, score, s.label))
            })
            .collect()
    }

    /// Record a correct/incorrect prediction result into the running stats.
    ///
    /// This is used internally when `predict_class` is called.  Exposed
    /// publicly for external evaluation loops that use `classify()` and wish
    /// to manually feed outcomes back.
    pub fn record_prediction(&mut self, was_correct: bool) {
        self.total_predictions += 1;
        if was_correct {
            self.correct_predictions += 1;
        }
    }

    /// Return a reference to the current weight vector.
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// Return the current bias value.
    pub fn bias(&self) -> f64 {
        self.bias
    }

    /// Return the number of features this learner was constructed for.
    pub fn dims(&self) -> usize {
        self.dims
    }
}

// ---------------------------------------------------------------------------
// Free-function helpers (module-private)
// ---------------------------------------------------------------------------

/// Dot product of two equal-length slices.
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Numerically stable sigmoid: σ(x) = 1/(1 + exp(-x)).
///
/// Uses the standard trick of branching on the sign of x to avoid overflow.
fn stable_sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        let e = (-x).exp();
        1.0 / (1.0 + e)
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Numerically stable log-sigmoid loss: ln(1 + exp(-margin)).
///
/// Uses log-sum-exp trick for numerical stability.
fn log_loss_stable(margin: f64) -> f64 {
    // ln(1 + exp(-margin))
    if margin >= 0.0 {
        // margin >= 0 → exp(-margin) ≤ 1 → no overflow
        (-margin).exp().ln_1p()
    } else {
        // margin < 0 → -margin > 0 → exp(-margin) can overflow
        // Use: ln(1 + exp(-margin)) = -margin + ln(1 + exp(margin))
        -margin + margin.exp().ln_1p()
    }
}

/// Compute loss for the given function variant.
fn compute_loss(loss_fn: OlLossFunction, score: f64, label: f64) -> f64 {
    let margin = label * score;
    match loss_fn {
        OlLossFunction::Hinge => (1.0 - margin).max(0.0),
        OlLossFunction::SquaredHinge => {
            let h = (1.0 - margin).max(0.0);
            h * h
        }
        OlLossFunction::LogLoss => log_loss_stable(margin),
    }
}

/// Return `true` iff `label` is ±1.0 (up to floating-point precision).
fn is_binary_label(label: f64) -> bool {
    (label - 1.0).abs() < 1e-9 || (label + 1.0).abs() < 1e-9
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::{
        compute_loss, dot, is_binary_label, log_loss_stable, stable_sigmoid, LearnerError,
        OlLossFunction, OnlineAlgorithm, OnlineLearner, TrainingSample,
    };

    // -----------------------------------------------------------------------
    // Helper builders
    // -----------------------------------------------------------------------

    fn perceptron(dims: usize) -> OnlineLearner {
        OnlineLearner::new(OnlineAlgorithm::Perceptron, dims, OlLossFunction::Hinge)
    }

    fn pa(dims: usize, c: f64) -> OnlineLearner {
        OnlineLearner::new(
            OnlineAlgorithm::PassiveAggressive { c },
            dims,
            OlLossFunction::Hinge,
        )
    }

    fn sgd(dims: usize, lr: f64, momentum: f64, l2_reg: f64) -> OnlineLearner {
        OnlineLearner::new(
            OnlineAlgorithm::SgdMomentum {
                lr,
                momentum,
                l2_reg,
            },
            dims,
            OlLossFunction::Hinge,
        )
    }

    fn sample(features: Vec<f64>, label: f64) -> TrainingSample {
        TrainingSample::new(features, label)
    }

    // -----------------------------------------------------------------------
    // Test 1: construction initialises to zero
    // -----------------------------------------------------------------------
    #[test]
    fn test_construction_zero_init() {
        let learner = perceptron(4);
        assert_eq!(learner.dims(), 4);
        assert_eq!(learner.bias(), 0.0);
        assert!(learner.weights().iter().all(|&w| w == 0.0));
        assert!(learner.velocity.iter().all(|&v| v == 0.0));
    }

    // -----------------------------------------------------------------------
    // Test 2: predict on zero weights returns bias (0)
    // -----------------------------------------------------------------------
    #[test]
    fn test_predict_zero_weights() {
        let learner = perceptron(3);
        let score = learner
            .predict(&[1.0, 2.0, 3.0])
            .expect("test: should succeed");
        assert_eq!(score, 0.0);
    }

    // -----------------------------------------------------------------------
    // Test 3: dimension mismatch error
    // -----------------------------------------------------------------------
    #[test]
    fn test_dimension_mismatch() {
        let learner = perceptron(3);
        let err = learner.predict(&[1.0, 2.0]).unwrap_err();
        assert!(matches!(
            err,
            LearnerError::DimensionMismatch {
                expected: 3,
                got: 2
            }
        ));
    }

    // -----------------------------------------------------------------------
    // Test 4: empty input error
    // -----------------------------------------------------------------------
    #[test]
    fn test_empty_input() {
        let learner = perceptron(3);
        let err = learner.predict(&[]).unwrap_err();
        assert_eq!(err, LearnerError::EmptyInput);
    }

    // -----------------------------------------------------------------------
    // Test 5: invalid label for perceptron
    // -----------------------------------------------------------------------
    #[test]
    fn test_invalid_label_perceptron() {
        let mut learner = perceptron(2);
        let s = sample(vec![1.0, 0.0], 0.5);
        let err = learner.update(&s).unwrap_err();
        assert!(matches!(err, LearnerError::InvalidLabel { .. }));
    }

    // -----------------------------------------------------------------------
    // Test 6: invalid label for PA
    // -----------------------------------------------------------------------
    #[test]
    fn test_invalid_label_pa() {
        let mut learner = pa(2, 1.0);
        let s = sample(vec![1.0, 0.0], 0.0);
        let err = learner.update(&s).unwrap_err();
        assert!(matches!(err, LearnerError::InvalidLabel { .. }));
    }

    // -----------------------------------------------------------------------
    // Test 7: SGD accepts non-binary labels
    // -----------------------------------------------------------------------
    #[test]
    fn test_sgd_non_binary_label() {
        let mut learner = sgd(2, 0.1, 0.9, 0.0);
        let s = sample(vec![1.0, 0.5], 2.5);
        // Should not error
        learner.update(&s).expect("test: TD update should succeed");
    }

    // -----------------------------------------------------------------------
    // Test 8: Perceptron updates on misprediction
    // -----------------------------------------------------------------------
    #[test]
    fn test_perceptron_updates_on_misprediction() {
        let mut learner = perceptron(2);
        // Zero weights → score=0 → label*score=0 ≤ 0 → misprediction for label=1
        let s = sample(vec![1.0, 1.0], 1.0);
        learner.update(&s).expect("test: TD update should succeed");
        // Weights should be updated: w += label * x → [1, 1]
        assert_eq!(learner.weights()[0], 1.0);
        assert_eq!(learner.weights()[1], 1.0);
        assert_eq!(learner.bias(), 1.0);
    }

    // -----------------------------------------------------------------------
    // Test 9: Perceptron no update when correctly classified
    // -----------------------------------------------------------------------
    #[test]
    fn test_perceptron_no_update_correct() {
        let mut learner = perceptron(2);
        // Give it correct weights first
        learner.weights[0] = 2.0;
        learner.bias = 1.0;
        // score = 2.0 * 1.0 + 1.0 = 3.0 → label*score = 3 > 0 → correct
        let s = sample(vec![1.0, 0.0], 1.0);
        learner.update(&s).expect("test: TD update should succeed");
        assert_eq!(learner.weights()[0], 2.0); // unchanged
        assert_eq!(learner.bias(), 1.0); // unchanged
    }

    // -----------------------------------------------------------------------
    // Test 10: Perceptron converges on linearly separable data
    // -----------------------------------------------------------------------
    #[test]
    fn test_perceptron_convergence() {
        let mut learner = perceptron(2);
        let positives: Vec<_> = (0..5)
            .map(|i| sample(vec![i as f64 + 1.0, 0.5], 1.0))
            .collect();
        let negatives: Vec<_> = (0..5)
            .map(|i| sample(vec![-(i as f64 + 1.0), -0.5], -1.0))
            .collect();

        let mut all: Vec<TrainingSample> = Vec::new();
        all.extend(positives);
        all.extend(negatives);

        for _ in 0..20 {
            for s in &all {
                let _ = learner.update(s);
            }
        }
        let acc = learner.accuracy(&all).expect("test: should succeed");
        assert!(acc > 0.9, "Expected accuracy > 0.9, got {acc}");
    }

    // -----------------------------------------------------------------------
    // Test 11: PA-I update reduces loss on positive example
    // -----------------------------------------------------------------------
    #[test]
    fn test_pa_update_reduces_loss() {
        let mut learner = pa(2, 1.0);
        let s = sample(vec![1.0, 0.0], 1.0);
        let pre_loss = learner.update(&s).expect("test: TD update should succeed");
        let post_score = learner.predict(&s.features).expect("test: should succeed");
        let post_loss = compute_loss(OlLossFunction::Hinge, post_score, 1.0);
        // Loss should decrease or stay zero
        assert!(post_loss <= pre_loss + 1e-10);
    }

    // -----------------------------------------------------------------------
    // Test 12: PA-I passive on already correct prediction
    // -----------------------------------------------------------------------
    #[test]
    fn test_pa_passive_when_correct() {
        let mut learner = pa(2, 1.0);
        // Set large weights so sample is correctly classified with large margin
        learner.weights[0] = 10.0;
        let s = sample(vec![1.0, 0.0], 1.0); // score = 10 → margin = 10 > 1
        let w_before = learner.weights()[0];
        learner.update(&s).expect("test: TD update should succeed");
        assert_eq!(learner.weights()[0], w_before); // no update
    }

    // -----------------------------------------------------------------------
    // Test 13: PA-I tau computation is correct
    // -----------------------------------------------------------------------
    #[test]
    fn test_pa_tau_formula() {
        let mut learner = pa(1, 1.0);
        // x = [1.0], label = 1.0, initial score = 0
        // loss = max(0, 1 - 1*0) = 1
        // ||x||^2 = 1
        // tau = 1 / (1 + 1/(2*1)) = 1 / 1.5 = 2/3
        let s = sample(vec![1.0], 1.0);
        learner.update(&s).expect("test: TD update should succeed");
        let expected = 2.0 / 3.0;
        assert!((learner.weights()[0] - expected).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // Test 14: SGD momentum velocity accumulates
    // -----------------------------------------------------------------------
    #[test]
    fn test_sgd_velocity_accumulates() {
        let mut learner = sgd(2, 0.1, 0.9, 0.0);
        let s = sample(vec![1.0, 1.0], 1.0);
        learner.update(&s).expect("test: TD update should succeed");
        // velocity should be non-zero after first update
        let v_sum: f64 = learner.velocity.iter().sum();
        assert_ne!(v_sum, 0.0);
    }

    // -----------------------------------------------------------------------
    // Test 15: SGD L2 regularisation shrinks weights
    // -----------------------------------------------------------------------
    #[test]
    fn test_sgd_l2_shrinks_weights() {
        let mut learner = OnlineLearner::new(
            OnlineAlgorithm::SgdMomentum {
                lr: 0.01,
                momentum: 0.0,
                l2_reg: 0.1,
            },
            2,
            OlLossFunction::Hinge,
        );
        // Give it some weights
        learner.weights[0] = 5.0;
        learner.weights[1] = 5.0;

        // Correctly classified sample (no gradient from loss, only L2)
        // score = 5.0 * 0.0 = 0 → loss grad = -label = -1 (in margin)
        // But let's use large weights so the score will be large enough
        learner.weights[0] = 5.0;
        learner.weights[1] = 0.0;
        // score = 5.0*1.0 + 0.0*0.0 = 5.0, margin = 5 > 1 → grad_score = 0
        // Only L2 acts: grad_w = l2_reg * w[0] = 0.1 * 5 = 0.5
        // velocity = 0 - 0.01 * 0.5 = -0.005
        // w[0] = 5.0 - 0.005 = 4.995
        let s = sample(vec![1.0, 0.0], 1.0);
        learner.update(&s).expect("test: TD update should succeed");
        assert!(learner.weights()[0] < 5.0);
    }

    // -----------------------------------------------------------------------
    // Test 16: predict_class returns +1 or -1
    // -----------------------------------------------------------------------
    #[test]
    fn test_predict_class_values() {
        let mut learner = perceptron(2);
        learner.weights[0] = 1.0;
        let c1 = learner
            .predict_class(&[1.0, 0.0])
            .expect("test: should succeed");
        let c2 = learner
            .predict_class(&[-1.0, 0.0])
            .expect("test: should succeed");
        assert_eq!(c1, 1);
        assert_eq!(c2, -1);
    }

    // -----------------------------------------------------------------------
    // Test 17: predict_class updates total_predictions
    // -----------------------------------------------------------------------
    #[test]
    fn test_predict_class_updates_stats() {
        let mut learner = perceptron(2);
        learner
            .predict_class(&[1.0, 0.0])
            .expect("test: should succeed");
        learner
            .predict_class(&[1.0, 0.0])
            .expect("test: should succeed");
        assert_eq!(learner.stats().total_predictions, 2);
    }

    // -----------------------------------------------------------------------
    // Test 18: batch_update returns per-sample losses
    // -----------------------------------------------------------------------
    #[test]
    fn test_batch_update_returns_losses() {
        let mut learner = perceptron(2);
        let samples = vec![sample(vec![1.0, 0.0], 1.0), sample(vec![0.0, 1.0], -1.0)];
        let losses = learner
            .batch_update(&samples)
            .expect("test: should succeed");
        assert_eq!(losses.len(), 2);
        assert!(losses.iter().all(|&l| l >= 0.0));
    }

    // -----------------------------------------------------------------------
    // Test 19: batch_update on empty slice returns EmptyInput
    // -----------------------------------------------------------------------
    #[test]
    fn test_batch_update_empty() {
        let mut learner = perceptron(2);
        let err = learner.batch_update(&[]).unwrap_err();
        assert_eq!(err, LearnerError::EmptyInput);
    }

    // -----------------------------------------------------------------------
    // Test 20: accuracy on perfectly learned data is 1.0
    // -----------------------------------------------------------------------
    #[test]
    fn test_accuracy_perfect() {
        let mut learner = perceptron(1);
        let samples = vec![sample(vec![3.0], 1.0), sample(vec![-3.0], -1.0)];
        // Train multiple epochs
        for _ in 0..10 {
            for s in &samples {
                let _ = learner.update(s);
            }
        }
        let acc = learner.accuracy(&samples).expect("test: should succeed");
        assert_eq!(acc, 1.0);
    }

    // -----------------------------------------------------------------------
    // Test 21: accuracy on empty returns EmptyInput
    // -----------------------------------------------------------------------
    #[test]
    fn test_accuracy_empty() {
        let learner = perceptron(2);
        let err = learner.accuracy(&[]).unwrap_err();
        assert_eq!(err, LearnerError::EmptyInput);
    }

    // -----------------------------------------------------------------------
    // Test 22: reset zeroes everything
    // -----------------------------------------------------------------------
    #[test]
    fn test_reset() {
        let mut learner = perceptron(3);
        let s = sample(vec![1.0, 1.0, 1.0], 1.0);
        learner.update(&s).expect("test: TD update should succeed");
        learner.reset();
        assert!(learner.weights().iter().all(|&w| w == 0.0));
        assert_eq!(learner.bias(), 0.0);
        assert_eq!(learner.stats().total_updates, 0);
        assert_eq!(learner.stats().avg_loss, 0.0);
    }

    // -----------------------------------------------------------------------
    // Test 23: l2_norm of zero vector is 0
    // -----------------------------------------------------------------------
    #[test]
    fn test_l2_norm_zero() {
        let learner = perceptron(4);
        assert_eq!(learner.l2_norm(), 0.0);
    }

    // -----------------------------------------------------------------------
    // Test 24: l2_norm computation is correct
    // -----------------------------------------------------------------------
    #[test]
    fn test_l2_norm_value() {
        let mut learner = perceptron(2);
        learner.weights[0] = 3.0;
        learner.weights[1] = 4.0;
        assert!((learner.l2_norm() - 5.0).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // Test 25: stats() reports correct total_updates
    // -----------------------------------------------------------------------
    #[test]
    fn test_stats_total_updates() {
        let mut learner = perceptron(2);
        for _ in 0..5 {
            learner
                .update(&sample(vec![1.0, 0.0], 1.0))
                .expect("test: should succeed");
        }
        assert_eq!(learner.stats().total_updates, 5);
    }

    // -----------------------------------------------------------------------
    // Test 26: avg_loss increases on hard examples
    // -----------------------------------------------------------------------
    #[test]
    fn test_stats_avg_loss_non_negative() {
        let mut learner = perceptron(2);
        let samples = vec![sample(vec![1.0, 0.0], 1.0), sample(vec![-1.0, 0.0], -1.0)];
        let _ = learner
            .batch_update(&samples)
            .expect("test: should succeed");
        assert!(learner.stats().avg_loss >= 0.0);
    }

    // -----------------------------------------------------------------------
    // Test 27: Hinge loss computation
    // -----------------------------------------------------------------------
    #[test]
    fn test_hinge_loss() {
        // margin = 1 → loss = 0
        assert_eq!(compute_loss(OlLossFunction::Hinge, 1.0, 1.0), 0.0);
        // margin = 0.5 → loss = 0.5
        assert!((compute_loss(OlLossFunction::Hinge, 0.5, 1.0) - 0.5).abs() < 1e-10);
        // margin = -1 → loss = 2
        assert!((compute_loss(OlLossFunction::Hinge, -1.0, 1.0) - 2.0).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // Test 28: SquaredHinge loss computation
    // -----------------------------------------------------------------------
    #[test]
    fn test_squared_hinge_loss() {
        // margin = 1 → loss = 0
        assert_eq!(compute_loss(OlLossFunction::SquaredHinge, 1.0, 1.0), 0.0);
        // margin = 0.5 → hinge = 0.5, loss = 0.25
        assert!((compute_loss(OlLossFunction::SquaredHinge, 0.5, 1.0) - 0.25).abs() < 1e-10);
        // margin = -1 → hinge = 2, loss = 4
        assert!((compute_loss(OlLossFunction::SquaredHinge, -1.0, 1.0) - 4.0).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // Test 29: LogLoss computation and numerical stability
    // -----------------------------------------------------------------------
    #[test]
    fn test_log_loss_stability() {
        // At score=0, margin=0 → ln(2) ≈ 0.693
        let l = compute_loss(OlLossFunction::LogLoss, 0.0, 1.0);
        assert!((l - std::f64::consts::LN_2).abs() < 1e-10);

        // Large positive margin → very small loss
        let l_large = compute_loss(OlLossFunction::LogLoss, 100.0, 1.0);
        assert!(l_large < 1e-10);

        // Large negative margin → approximately equal to |margin|
        let l_neg = compute_loss(OlLossFunction::LogLoss, -100.0, 1.0);
        assert!((l_neg - 100.0).abs() < 1.0);

        // Always non-negative
        for s in [-10.0_f64, -1.0, 0.0, 1.0, 10.0] {
            for y in [-1.0_f64, 1.0] {
                assert!(compute_loss(OlLossFunction::LogLoss, s, y) >= 0.0);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Test 30: stable_sigmoid is in (0,1) and symmetric
    // -----------------------------------------------------------------------
    #[test]
    fn test_stable_sigmoid() {
        assert!((stable_sigmoid(0.0) - 0.5).abs() < 1e-10);
        assert!(stable_sigmoid(100.0) > 0.999);
        assert!(stable_sigmoid(-100.0) < 0.001);
        // Symmetry: sigma(x) = 1 - sigma(-x)
        for x in [-5.0_f64, -1.0, 0.0, 1.0, 5.0] {
            assert!((stable_sigmoid(x) + stable_sigmoid(-x) - 1.0).abs() < 1e-12);
        }
    }

    // -----------------------------------------------------------------------
    // Test 31: log_loss_stable equals ln(2) at margin=0
    // -----------------------------------------------------------------------
    #[test]
    fn test_log_loss_stable_fn() {
        let at_zero = log_loss_stable(0.0);
        assert!((at_zero - std::f64::consts::LN_2).abs() < 1e-12);
        // Positive margin → decreasing loss
        assert!(log_loss_stable(1.0) < log_loss_stable(0.0));
        assert!(log_loss_stable(5.0) < log_loss_stable(1.0));
    }

    // -----------------------------------------------------------------------
    // Test 32: dot product correctness
    // -----------------------------------------------------------------------
    #[test]
    fn test_dot() {
        assert_eq!(dot(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]), 32.0);
        assert_eq!(dot(&[], &[]), 0.0);
    }

    // -----------------------------------------------------------------------
    // Test 33: is_binary_label helper
    // -----------------------------------------------------------------------
    #[test]
    fn test_is_binary_label() {
        assert!(is_binary_label(1.0));
        assert!(is_binary_label(-1.0));
        assert!(!is_binary_label(0.0));
        assert!(!is_binary_label(2.0));
        assert!(!is_binary_label(0.5));
    }

    // -----------------------------------------------------------------------
    // Test 34: TrainingSample::is_valid_binary_label
    // -----------------------------------------------------------------------
    #[test]
    fn test_training_sample_valid_binary_label() {
        let pos = sample(vec![1.0], 1.0);
        let neg = sample(vec![1.0], -1.0);
        let bad = sample(vec![1.0], 0.0);
        assert!(pos.is_valid_binary_label());
        assert!(neg.is_valid_binary_label());
        assert!(!bad.is_valid_binary_label());
    }

    // -----------------------------------------------------------------------
    // Test 35: classify does not modify prediction stats
    // -----------------------------------------------------------------------
    #[test]
    fn test_classify_no_stats_change() {
        let learner = perceptron(2);
        learner.classify(&[1.0, 0.0]).expect("test: should succeed");
        assert_eq!(learner.stats().total_predictions, 0);
    }

    // -----------------------------------------------------------------------
    // Test 36: evaluate_losses returns correct count
    // -----------------------------------------------------------------------
    #[test]
    fn test_evaluate_losses() {
        let learner = perceptron(2);
        let samples = vec![
            sample(vec![1.0, 0.0], 1.0),
            sample(vec![0.0, 1.0], -1.0),
            sample(vec![1.0, 1.0], 1.0),
        ];
        let losses = learner
            .evaluate_losses(&samples)
            .expect("test: should succeed");
        assert_eq!(losses.len(), 3);
        assert!(losses.iter().all(|&l| l >= 0.0));
    }

    // -----------------------------------------------------------------------
    // Test 37: average_loss empty returns error
    // -----------------------------------------------------------------------
    #[test]
    fn test_average_loss_empty() {
        let learner = perceptron(2);
        let err = learner.average_loss(&[]).unwrap_err();
        assert_eq!(err, LearnerError::EmptyInput);
    }

    // -----------------------------------------------------------------------
    // Test 38: SGD with LogLoss converges on simple data
    // -----------------------------------------------------------------------
    #[test]
    fn test_sgd_logloss_convergence() {
        let mut learner = OnlineLearner::new(
            OnlineAlgorithm::SgdMomentum {
                lr: 0.1,
                momentum: 0.9,
                l2_reg: 0.001,
            },
            1,
            OlLossFunction::LogLoss,
        );
        // Trivially separable 1-D data
        let pos = sample(vec![3.0], 1.0);
        let neg = sample(vec![-3.0], -1.0);

        for _ in 0..200 {
            let _ = learner.update(&pos);
            let _ = learner.update(&neg);
        }
        assert_eq!(learner.classify(&[3.0]).expect("test: should succeed"), 1);
        assert_eq!(learner.classify(&[-3.0]).expect("test: should succeed"), -1);
    }

    // -----------------------------------------------------------------------
    // Test 39: PA-I with different C values
    // -----------------------------------------------------------------------
    #[test]
    fn test_pa_c_parameter_effect() {
        // Larger C → more aggressive update → larger weight change per step
        let mut learner_low_c = pa(1, 0.1);
        let mut learner_high_c = pa(1, 100.0);

        let s = sample(vec![1.0], 1.0);
        learner_low_c
            .update(&s)
            .expect("test: TD update should succeed");
        learner_high_c
            .update(&s)
            .expect("test: TD update should succeed");

        // High C should produce a larger (or equal) weight update
        assert!(learner_high_c.weights()[0] >= learner_low_c.weights()[0]);
    }

    // -----------------------------------------------------------------------
    // Test 40: OnlineLearnerStats weight_norm matches l2_norm
    // -----------------------------------------------------------------------
    #[test]
    fn test_stats_weight_norm() {
        let mut learner = perceptron(3);
        learner
            .update(&sample(vec![3.0, 0.0, 4.0], 1.0))
            .expect("test: should succeed");
        let stats = learner.stats();
        assert!((stats.weight_norm - learner.l2_norm()).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // Test 41: record_prediction increments counts
    // -----------------------------------------------------------------------
    #[test]
    fn test_record_prediction() {
        let mut learner = perceptron(2);
        learner.record_prediction(true);
        learner.record_prediction(false);
        learner.record_prediction(true);
        let s = learner.stats();
        assert_eq!(s.total_predictions, 3);
        assert_eq!(s.correct_predictions, 2);
    }

    // -----------------------------------------------------------------------
    // Test 42: LearnerError display messages
    // -----------------------------------------------------------------------
    #[test]
    fn test_error_display() {
        let e1 = LearnerError::DimensionMismatch {
            expected: 3,
            got: 2,
        };
        let e2 = LearnerError::EmptyInput;
        let e3 = LearnerError::InvalidLabel { label: 0.5 };
        assert!(e1.to_string().contains("3"));
        assert!(e2.to_string().contains("empty"));
        assert!(e3.to_string().contains("0.5"));
    }

    // -----------------------------------------------------------------------
    // Test 43: OlLossFunction display
    // -----------------------------------------------------------------------
    #[test]
    fn test_loss_function_display() {
        assert_eq!(OlLossFunction::Hinge.to_string(), "Hinge");
        assert_eq!(OlLossFunction::SquaredHinge.to_string(), "SquaredHinge");
        assert_eq!(OlLossFunction::LogLoss.to_string(), "LogLoss");
    }

    // -----------------------------------------------------------------------
    // Test 44: perceptron correctly handles negative class features
    // -----------------------------------------------------------------------
    #[test]
    fn test_perceptron_negative_class() {
        let mut learner = perceptron(2);
        let s = sample(vec![-1.0, -1.0], -1.0);
        // Initial score = 0, label*score = 0 ≤ 0 → update
        learner.update(&s).expect("test: TD update should succeed");
        // w += label * x = -1 * [-1, -1] = [1, 1] wait that's wrong
        // w += (-1) * (-1, -1) = (1, 1)
        assert_eq!(learner.weights()[0], 1.0);
        assert_eq!(learner.bias(), -1.0);
    }

    // -----------------------------------------------------------------------
    // Test 45: multiple reset cycles
    // -----------------------------------------------------------------------
    #[test]
    fn test_multiple_reset_cycles() {
        let mut learner = perceptron(3);
        for _ in 0..3 {
            for _ in 0..5 {
                let _ = learner.update(&sample(vec![1.0, 0.0, 0.5], 1.0));
            }
            learner.reset();
            assert_eq!(learner.stats().total_updates, 0);
            assert!(learner.weights().iter().all(|&w| w == 0.0));
        }
    }

    // -----------------------------------------------------------------------
    // Test 46: SGD with zero momentum behaves like vanilla SGD
    // -----------------------------------------------------------------------
    #[test]
    fn test_sgd_zero_momentum() {
        let mut learner = OnlineLearner::new(
            OnlineAlgorithm::SgdMomentum {
                lr: 0.5,
                momentum: 0.0,
                l2_reg: 0.0,
            },
            1,
            OlLossFunction::Hinge,
        );
        // x=[1], label=1, score=0, margin=0 < 1 → grad_score = -1
        // grad_w = -1 * 1 + 0 * 0 = -1
        // velocity = 0*0 - 0.5*(-1) = 0.5
        // w = 0 + 0.5 = 0.5
        let s = sample(vec![1.0], 1.0);
        learner.update(&s).expect("test: TD update should succeed");
        assert!((learner.weights()[0] - 0.5).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // Test 47: SquaredHinge SGD gradient at margin boundary
    // -----------------------------------------------------------------------
    #[test]
    fn test_squared_hinge_sgd_boundary() {
        let mut learner = OnlineLearner::new(
            OnlineAlgorithm::SgdMomentum {
                lr: 0.1,
                momentum: 0.0,
                l2_reg: 0.0,
            },
            1,
            OlLossFunction::SquaredHinge,
        );
        // margin=1 → grad_score = 0 (outside margin)
        learner.weights[0] = 1.0;
        // score = 1.0, margin = 1 → grad_score = 0
        let w_before = learner.weights()[0];
        let s = sample(vec![1.0], 1.0);
        learner.update(&s).expect("test: TD update should succeed");
        assert_eq!(learner.weights()[0], w_before); // no update
    }

    // -----------------------------------------------------------------------
    // Test 48: evaluate_losses empty returns EmptyInput
    // -----------------------------------------------------------------------
    #[test]
    fn test_evaluate_losses_empty() {
        let learner = perceptron(2);
        let err = learner.evaluate_losses(&[]).unwrap_err();
        assert_eq!(err, LearnerError::EmptyInput);
    }

    // -----------------------------------------------------------------------
    // Test 49: batch_update increments total_updates correctly
    // -----------------------------------------------------------------------
    #[test]
    fn test_batch_update_stats_total_updates() {
        let mut learner = perceptron(2);
        let samples = vec![
            sample(vec![1.0, 0.0], 1.0),
            sample(vec![0.0, 1.0], -1.0),
            sample(vec![1.0, 1.0], 1.0),
        ];
        learner
            .batch_update(&samples)
            .expect("test: should succeed");
        assert_eq!(learner.stats().total_updates, 3);
    }

    // -----------------------------------------------------------------------
    // Test 50: PA converges on 2-D linearly separable data
    // -----------------------------------------------------------------------
    #[test]
    fn test_pa_convergence_2d() {
        let mut learner = pa(2, 1.0);
        let samples: Vec<TrainingSample> = vec![
            sample(vec![2.0, 1.0], 1.0),
            sample(vec![1.0, 2.0], 1.0),
            sample(vec![-2.0, -1.0], -1.0),
            sample(vec![-1.0, -2.0], -1.0),
        ];
        for _ in 0..30 {
            for s in &samples {
                let _ = learner.update(s);
            }
        }
        let acc = learner.accuracy(&samples).expect("test: should succeed");
        assert_eq!(acc, 1.0);
    }
}
