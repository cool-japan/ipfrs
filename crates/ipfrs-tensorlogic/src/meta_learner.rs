//! MetaLearner — MAML-inspired meta-learning system.
//!
//! Maintains task-specific adaptations and a shared meta-representation.
//! The outer loop (meta-update) aggregates task adaptations to improve the
//! shared initialisation; the inner loop (adapt_to_task) fine-tunes the shared
//! weights for a single task in a small number of gradient-descent steps.

use std::collections::HashMap;
use std::fmt;

// ─── Error ────────────────────────────────────────────────────────────────────

/// Errors produced by [`MetaLearner`] operations.
#[derive(Debug, Clone, PartialEq)]
pub enum MetaError {
    /// The support set of a task is empty; cannot compute gradients.
    EmptySupportSet,
    /// The query set of a task is empty; cannot evaluate query loss.
    EmptyQuerySet,
    /// Feature vector has the wrong number of dimensions.
    DimensionMismatch {
        /// Expected number of dimensions.
        expected: usize,
        /// Actual number of dimensions encountered.
        got: usize,
    },
    /// `meta_update` was called with an empty slice of adaptations.
    NoAdaptations,
}

impl fmt::Display for MetaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MetaError::EmptySupportSet => write!(f, "support set must not be empty"),
            MetaError::EmptyQuerySet => write!(f, "query set must not be empty"),
            MetaError::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
            MetaError::NoAdaptations => write!(f, "no task adaptations provided for meta-update"),
        }
    }
}

impl std::error::Error for MetaError {}

// ─── Core domain types ────────────────────────────────────────────────────────

/// Newtype wrapper around a task identifier string.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TaskId(pub String);

impl TaskId {
    /// Create a new `TaskId` from any `Into<String>` value.
    pub fn new(id: impl Into<String>) -> Self {
        TaskId(id.into())
    }
}

impl fmt::Display for TaskId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// A single labeled training example.
#[derive(Debug, Clone)]
pub struct TaskExample {
    /// Input feature vector.
    pub features: Vec<f64>,
    /// Ground-truth label.
    pub label: f64,
}

impl TaskExample {
    /// Construct a new [`TaskExample`].
    pub fn new(features: Vec<f64>, label: f64) -> Self {
        TaskExample { features, label }
    }
}

/// Discriminates the learning objective of a meta-task.
#[derive(Debug, Clone, PartialEq)]
pub enum TaskType {
    /// Multi-class classification with a fixed number of categories.
    Classification {
        /// Number of classes.
        n_classes: usize,
    },
    /// Continuous-valued regression.
    Regression,
    /// Pairwise ranking (query/document relevance).
    Ranking,
}

/// A meta-learning task consisting of a support set (used for inner-loop
/// adaptation) and a query set (used to evaluate the adapted model).
#[derive(Debug, Clone)]
pub struct MetaTask {
    /// Unique task identifier.
    pub id: TaskId,
    /// Examples used during the inner-loop gradient-descent adaptation.
    pub support_set: Vec<TaskExample>,
    /// Examples used to evaluate the quality of the adaptation.
    pub query_set: Vec<TaskExample>,
    /// Discriminator for the learning objective.
    pub task_type: TaskType,
}

impl MetaTask {
    /// Construct a new [`MetaTask`].
    pub fn new(
        id: TaskId,
        support_set: Vec<TaskExample>,
        query_set: Vec<TaskExample>,
        task_type: TaskType,
    ) -> Self {
        MetaTask {
            id,
            support_set,
            query_set,
            task_type,
        }
    }
}

// ─── Model parameters ─────────────────────────────────────────────────────────

/// Shared meta-parameters: a linear model with `dims` weights and one bias.
#[derive(Debug, Clone)]
pub struct MetaParameters {
    /// Weight vector of length `dims`.
    pub weights: Vec<f64>,
    /// Scalar bias term.
    pub bias: f64,
    /// Dimensionality of the input feature space.
    pub dims: usize,
}

impl MetaParameters {
    /// Create zero-initialised meta-parameters with the given dimensionality.
    pub fn zeros(dims: usize) -> Self {
        MetaParameters {
            weights: vec![0.0; dims],
            bias: 0.0,
            dims,
        }
    }
}

/// The result of adapting the meta-parameters to a specific task.
#[derive(Debug, Clone)]
pub struct TaskAdaptation {
    /// Which task these weights were adapted for.
    pub task_id: TaskId,
    /// Weights after inner-loop gradient descent.
    pub adapted_weights: Vec<f64>,
    /// Bias after inner-loop gradient descent.
    pub adapted_bias: f64,
    /// Mean loss on the support set after the final inner-loop step.
    pub support_loss: f64,
    /// Mean loss on the query set evaluated with the adapted weights.
    pub query_loss: f64,
    /// Number of inner-loop gradient-descent steps taken.
    pub steps: u32,
}

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for [`MetaLearner`].
#[derive(Debug, Clone)]
pub struct MetaLearnerConfig {
    /// Learning rate used in the inner-loop adaptation.
    pub inner_lr: f64,
    /// Learning rate used for the outer meta-update.
    pub meta_lr: f64,
    /// Number of inner-loop gradient-descent steps per task.
    pub inner_steps: u32,
    /// Dimensionality of the input feature space.
    pub dims: usize,
    /// Random seed used to initialise meta-parameters via xorshift64.
    pub seed: u64,
}

impl Default for MetaLearnerConfig {
    fn default() -> Self {
        MetaLearnerConfig {
            inner_lr: 0.01,
            meta_lr: 0.001,
            inner_steps: 5,
            dims: 10,
            seed: 42,
        }
    }
}

// ─── Aggregate statistics ─────────────────────────────────────────────────────

/// Aggregate statistics for a [`MetaLearner`] instance.
#[derive(Debug, Clone)]
pub struct MetaLearnerStats {
    /// Total number of tasks stored in the history.
    pub total_tasks: usize,
    /// Number of outer meta-update steps that have been performed.
    pub meta_steps: u64,
    /// Mean support-set loss across all stored task adaptations.
    pub avg_support_loss: f64,
    /// Mean query-set loss across all stored task adaptations.
    pub avg_query_loss: f64,
    /// Lowest query-set loss found in the task history.
    pub best_query_loss: f64,
}

// ─── xorshift64 ───────────────────────────────────────────────────────────────

/// Inline xorshift64 PRNG step.  The state must be non-zero.
#[inline]
fn xorshift64(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

// ─── MetaLearner ─────────────────────────────────────────────────────────────

/// MAML-inspired meta-learner.
///
/// # Overview
///
/// * [`MetaLearner::adapt_to_task`] runs the **inner loop**: starting from the
///   shared `meta_params` it takes `config.inner_steps` gradient-descent steps
///   on the support set of the given task and evaluates the adapted model on
///   the query set.
/// * [`MetaLearner::meta_update`] runs the **outer loop**: given a batch of
///   `TaskAdaptation`s it averages the adapted weights, computes a meta-gradient
///   (direction from current meta-weights to the average), and updates
///   `meta_params` with `config.meta_lr`.
/// * [`MetaLearner::predict`] performs linear inference.
pub struct MetaLearner {
    /// Hyper-parameters that control the learning algorithm.
    pub config: MetaLearnerConfig,
    /// Shared meta-parameters (the "good initialisation" learned by MAML).
    pub meta_params: MetaParameters,
    /// Per-task adaptation results keyed by [`TaskId`].
    pub task_history: HashMap<TaskId, TaskAdaptation>,
    /// Count of completed outer meta-update steps.
    pub meta_step: u64,
}

impl MetaLearner {
    /// Create a new [`MetaLearner`] with weights initialised using xorshift64.
    ///
    /// Each weight is set to `xorshift64(state) as f64 / u64::MAX as f64 * 0.01`
    /// so that the initial values are small random numbers in `[0, 0.01)`.
    pub fn new(config: MetaLearnerConfig) -> Self {
        let mut state = config.seed.max(1); // xorshift64 state must be non-zero
        let dims = config.dims;
        let weights: Vec<f64> = (0..dims)
            .map(|_| {
                let raw = xorshift64(&mut state);
                (raw as f64 / u64::MAX as f64) * 0.01
            })
            .collect();

        let meta_params = MetaParameters {
            weights,
            bias: 0.0,
            dims,
        };

        MetaLearner {
            config,
            meta_params,
            task_history: HashMap::new(),
            meta_step: 0,
        }
    }

    // ── Loss helpers ────────────────────────────────────────────────────────

    /// Compute the scalar loss for a single prediction/label pair.
    ///
    /// | `task_type`       | loss formula                                           |
    /// |-------------------|--------------------------------------------------------|
    /// | Classification    | `max(0, 1 - label * tanh(prediction))`                 |
    /// | Regression        | `(prediction - label)²`                                |
    /// | Ranking           | `max(0, 1 - prediction * label)`                       |
    pub fn loss(prediction: f64, label: f64, task_type: &TaskType) -> f64 {
        match task_type {
            TaskType::Classification { .. } => (1.0 - label * prediction.tanh()).max(0.0),
            TaskType::Regression => (prediction - label).powi(2),
            TaskType::Ranking => (1.0 - prediction * label).max(0.0),
        }
    }

    // ── Gradient helpers ────────────────────────────────────────────────────

    /// Compute the gradient of the **MSE** loss with respect to the linear
    /// model parameters (weights and bias).
    ///
    /// Returns `(dL/dw, dL/db)` where `dL/dw[i] = 2*(prediction-label)*features[i]`
    /// and `dL/db = 2*(prediction-label)`.
    pub fn gradient(features: &[f64], prediction: f64, label: f64) -> (Vec<f64>, f64) {
        let residual = 2.0 * (prediction - label);
        let dw: Vec<f64> = features.iter().map(|&x| residual * x).collect();
        let db = residual;
        (dw, db)
    }

    /// Compute the average gradient over a batch of examples.
    fn batch_gradient(
        weights: &[f64],
        bias: f64,
        examples: &[TaskExample],
        dims: usize,
    ) -> Result<(Vec<f64>, f64), MetaError> {
        if examples.is_empty() {
            return Err(MetaError::EmptySupportSet);
        }
        let n = examples.len() as f64;
        let mut dw_sum = vec![0.0f64; dims];
        let mut db_sum = 0.0f64;

        for ex in examples {
            if ex.features.len() != dims {
                return Err(MetaError::DimensionMismatch {
                    expected: dims,
                    got: ex.features.len(),
                });
            }
            let pred = Self::linear_predict(weights, bias, &ex.features);
            let (dw, db) = Self::gradient(&ex.features, pred, ex.label);
            for (acc, g) in dw_sum.iter_mut().zip(dw.iter()) {
                *acc += g;
            }
            db_sum += db;
        }

        let dw_avg: Vec<f64> = dw_sum.iter().map(|&v| v / n).collect();
        Ok((dw_avg, db_sum / n))
    }

    // ── Linear prediction ───────────────────────────────────────────────────

    /// Dot-product linear prediction: `weights · features + bias`.
    fn linear_predict(weights: &[f64], bias: f64, features: &[f64]) -> f64 {
        weights
            .iter()
            .zip(features.iter())
            .map(|(&w, &x)| w * x)
            .sum::<f64>()
            + bias
    }

    // ── Public API ──────────────────────────────────────────────────────────

    /// Perform a linear prediction using either the meta-parameters or, if
    /// `adaptation` is `Some`, the task-adapted parameters.
    ///
    /// # Errors
    ///
    /// Returns [`MetaError::DimensionMismatch`] if `features.len() != config.dims`.
    pub fn predict(
        &self,
        features: &[f64],
        adaptation: Option<&TaskAdaptation>,
    ) -> Result<f64, MetaError> {
        if features.len() != self.config.dims {
            return Err(MetaError::DimensionMismatch {
                expected: self.config.dims,
                got: features.len(),
            });
        }
        let (weights, bias) = match adaptation {
            Some(a) => (a.adapted_weights.as_slice(), a.adapted_bias),
            None => (self.meta_params.weights.as_slice(), self.meta_params.bias),
        };
        Ok(Self::linear_predict(weights, bias, features))
    }

    /// Run the **inner loop** for `task`: adapt from `meta_params` using the
    /// support set, then evaluate on the query set.
    ///
    /// The resulting [`TaskAdaptation`] is stored in `task_history` and
    /// returned to the caller.
    ///
    /// # Errors
    ///
    /// * [`MetaError::EmptySupportSet`] — if the task support set is empty.
    /// * [`MetaError::EmptyQuerySet`] — if the task query set is empty.
    /// * [`MetaError::DimensionMismatch`] — if any example has the wrong
    ///   number of features.
    pub fn adapt_to_task(&mut self, task: &MetaTask) -> Result<TaskAdaptation, MetaError> {
        if task.support_set.is_empty() {
            return Err(MetaError::EmptySupportSet);
        }
        if task.query_set.is_empty() {
            return Err(MetaError::EmptyQuerySet);
        }

        let dims = self.config.dims;
        let inner_lr = self.config.inner_lr;
        let inner_steps = self.config.inner_steps;

        // Start from the meta-parameters (clone so we don't touch meta_params)
        let mut w = self.meta_params.weights.clone();
        let mut b = self.meta_params.bias;

        for _ in 0..inner_steps {
            // Compute batch gradient on support set
            let (dw, db) = Self::batch_gradient(&w, b, &task.support_set, dims)?;

            // Gradient descent step
            for (wi, &gi) in w.iter_mut().zip(dw.iter()) {
                *wi -= inner_lr * gi;
            }
            b -= inner_lr * db;
        }

        // Compute final support loss with the adapted weights
        let support_loss = self.mean_loss_raw(&w, b, &task.support_set, &task.task_type)?;

        // Evaluate on query set
        let query_loss = self.mean_loss_raw(&w, b, &task.query_set, &task.task_type)?;

        let adaptation = TaskAdaptation {
            task_id: task.id.clone(),
            adapted_weights: w,
            adapted_bias: b,
            support_loss,
            query_loss,
            steps: inner_steps,
        };

        self.task_history
            .insert(task.id.clone(), adaptation.clone());
        Ok(adaptation)
    }

    /// Helper: compute mean loss without the extra `self` borrow on `dims`.
    fn mean_loss_raw(
        &self,
        weights: &[f64],
        bias: f64,
        examples: &[TaskExample],
        task_type: &TaskType,
    ) -> Result<f64, MetaError> {
        if examples.is_empty() {
            return Err(MetaError::EmptySupportSet);
        }
        let mut total = 0.0;
        for ex in examples {
            if ex.features.len() != self.config.dims {
                return Err(MetaError::DimensionMismatch {
                    expected: self.config.dims,
                    got: ex.features.len(),
                });
            }
            let pred = Self::linear_predict(weights, bias, &ex.features);
            total += Self::loss(pred, ex.label, task_type);
        }
        Ok(total / examples.len() as f64)
    }

    /// Run the **outer loop** (meta-update).
    ///
    /// Averages `adapted_weights` across all provided adaptations, then
    /// computes a meta-gradient as the direction from current meta-weights to
    /// that average and applies a single gradient-descent step with `meta_lr`.
    ///
    /// # Errors
    ///
    /// * [`MetaError::NoAdaptations`] — if `adaptations` is empty.
    /// * [`MetaError::DimensionMismatch`] — if any adaptation has the wrong
    ///   number of weight dimensions.
    pub fn meta_update(&mut self, adaptations: &[TaskAdaptation]) -> Result<(), MetaError> {
        if adaptations.is_empty() {
            return Err(MetaError::NoAdaptations);
        }

        let dims = self.config.dims;
        let meta_lr = self.config.meta_lr;
        let n = adaptations.len() as f64;

        // Average adapted weights and biases
        let mut avg_w = vec![0.0f64; dims];
        let mut avg_b = 0.0f64;

        for a in adaptations {
            if a.adapted_weights.len() != dims {
                return Err(MetaError::DimensionMismatch {
                    expected: dims,
                    got: a.adapted_weights.len(),
                });
            }
            for (acc, &v) in avg_w.iter_mut().zip(a.adapted_weights.iter()) {
                *acc += v;
            }
            avg_b += a.adapted_bias;
        }
        for v in avg_w.iter_mut() {
            *v /= n;
        }
        avg_b /= n;

        // Meta-gradient = (meta_weights - adapted_avg)
        // Update: meta_weights += meta_lr * (adapted_avg - meta_weights)
        for (mw, &aw) in self.meta_params.weights.iter_mut().zip(avg_w.iter()) {
            let meta_grad = *mw - aw; // gradient pointing back toward meta
            *mw -= meta_lr * meta_grad;
        }
        self.meta_params.bias -= meta_lr * (self.meta_params.bias - avg_b);

        self.meta_step += 1;
        Ok(())
    }

    /// Compute the cosine similarity between the `adapted_weights` of two
    /// task adaptations.  Returns `0.0` if either weight vector is all-zero.
    pub fn task_similarity(a: &TaskAdaptation, b: &TaskAdaptation) -> f64 {
        let dot: f64 = a
            .adapted_weights
            .iter()
            .zip(b.adapted_weights.iter())
            .map(|(&x, &y)| x * y)
            .sum();
        let norm_a: f64 = a.adapted_weights.iter().map(|&x| x * x).sum::<f64>().sqrt();
        let norm_b: f64 = b.adapted_weights.iter().map(|&x| x * x).sum::<f64>().sqrt();
        let denom = norm_a * norm_b;
        if denom == 0.0 {
            0.0
        } else {
            (dot / denom).clamp(-1.0, 1.0)
        }
    }

    /// Return the task with the lowest query loss from the history, or `None`
    /// if the history is empty.
    pub fn best_task(&self) -> Option<(&TaskId, &TaskAdaptation)> {
        self.task_history.iter().min_by(|(_, a), (_, b)| {
            a.query_loss
                .partial_cmp(&b.query_loss)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Remove the stored adaptation for `task_id` from the history.
    pub fn reset_task(&mut self, task_id: &TaskId) {
        self.task_history.remove(task_id);
    }

    /// Compute aggregate statistics over the current task history.
    pub fn stats(&self) -> MetaLearnerStats {
        let total_tasks = self.task_history.len();
        if total_tasks == 0 {
            return MetaLearnerStats {
                total_tasks: 0,
                meta_steps: self.meta_step,
                avg_support_loss: 0.0,
                avg_query_loss: 0.0,
                best_query_loss: f64::INFINITY,
            };
        }

        let mut sum_support = 0.0;
        let mut sum_query = 0.0;
        let mut best = f64::INFINITY;

        for a in self.task_history.values() {
            sum_support += a.support_loss;
            sum_query += a.query_loss;
            if a.query_loss < best {
                best = a.query_loss;
            }
        }

        MetaLearnerStats {
            total_tasks,
            meta_steps: self.meta_step,
            avg_support_loss: sum_support / total_tasks as f64,
            avg_query_loss: sum_query / total_tasks as f64,
            best_query_loss: best,
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use crate::meta_learner::{
        xorshift64, MetaError, MetaLearner, MetaLearnerConfig, MetaParameters, MetaTask,
        TaskAdaptation, TaskExample, TaskId, TaskType,
    };

    // ── Helpers ──────────────────────────────────────────────────────────────

    fn simple_config(dims: usize) -> MetaLearnerConfig {
        MetaLearnerConfig {
            inner_lr: 0.1,
            meta_lr: 0.01,
            inner_steps: 3,
            dims,
            seed: 7,
        }
    }

    fn make_regression_task(id: &str, dims: usize, n_support: usize, n_query: usize) -> MetaTask {
        let support_set: Vec<TaskExample> = (0..n_support)
            .map(|i| {
                let features: Vec<f64> = (0..dims).map(|j| (i + j) as f64 * 0.1).collect();
                let label = features.iter().sum::<f64>(); // sum of features as target
                TaskExample::new(features, label)
            })
            .collect();
        let query_set: Vec<TaskExample> = (n_support..n_support + n_query)
            .map(|i| {
                let features: Vec<f64> = (0..dims).map(|j| (i + j) as f64 * 0.1).collect();
                let label = features.iter().sum::<f64>();
                TaskExample::new(features, label)
            })
            .collect();
        MetaTask::new(
            TaskId::new(id),
            support_set,
            query_set,
            TaskType::Regression,
        )
    }

    fn make_classification_task(id: &str, dims: usize) -> MetaTask {
        let make_ex = |v: f64| TaskExample::new(vec![v; dims], if v > 0.0 { 1.0 } else { -1.0 });
        MetaTask::new(
            TaskId::new(id),
            vec![make_ex(0.5), make_ex(-0.5)],
            vec![make_ex(0.3), make_ex(-0.3)],
            TaskType::Classification { n_classes: 2 },
        )
    }

    fn make_ranking_task(id: &str, dims: usize) -> MetaTask {
        let make_ex = |v: f64| TaskExample::new(vec![v; dims], if v > 0.5 { 1.0 } else { -1.0 });
        MetaTask::new(
            TaskId::new(id),
            vec![make_ex(0.9), make_ex(0.1)],
            vec![make_ex(0.8), make_ex(0.2)],
            TaskType::Ranking,
        )
    }

    // ── TaskId tests ─────────────────────────────────────────────────────────

    #[test]
    fn test_task_id_new() {
        let id = TaskId::new("task_1");
        assert_eq!(id.0, "task_1");
    }

    #[test]
    fn test_task_id_display() {
        let id = TaskId::new("hello");
        assert_eq!(format!("{id}"), "hello");
    }

    #[test]
    fn test_task_id_equality() {
        assert_eq!(TaskId::new("a"), TaskId::new("a"));
        assert_ne!(TaskId::new("a"), TaskId::new("b"));
    }

    #[test]
    fn test_task_id_hash_in_map() {
        let mut map = std::collections::HashMap::new();
        map.insert(TaskId::new("k"), 42u32);
        assert_eq!(map[&TaskId::new("k")], 42);
    }

    // ── TaskExample tests ────────────────────────────────────────────────────

    #[test]
    fn test_task_example_new() {
        let ex = TaskExample::new(vec![1.0, 2.0], 3.0);
        assert_eq!(ex.features.len(), 2);
        assert!((ex.label - 3.0).abs() < 1e-12);
    }

    // ── MetaParameters tests ─────────────────────────────────────────────────

    #[test]
    fn test_meta_parameters_zeros() {
        let p = MetaParameters::zeros(5);
        assert_eq!(p.dims, 5);
        assert_eq!(p.weights.len(), 5);
        assert!(p.weights.iter().all(|&w| w == 0.0));
        assert_eq!(p.bias, 0.0);
    }

    // ── MetaLearnerConfig defaults ───────────────────────────────────────────

    #[test]
    fn test_config_defaults() {
        let cfg = MetaLearnerConfig::default();
        assert!((cfg.inner_lr - 0.01).abs() < 1e-12);
        assert!((cfg.meta_lr - 0.001).abs() < 1e-12);
        assert_eq!(cfg.inner_steps, 5);
        assert_eq!(cfg.dims, 10);
        assert_eq!(cfg.seed, 42);
    }

    // ── xorshift64 ───────────────────────────────────────────────────────────

    #[test]
    fn test_xorshift64_non_zero() {
        let mut state = 1u64;
        for _ in 0..100 {
            let v = xorshift64(&mut state);
            assert_ne!(v, 0, "xorshift64 must never produce 0");
        }
    }

    #[test]
    fn test_xorshift64_deterministic() {
        let mut s1 = 42u64;
        let mut s2 = 42u64;
        for _ in 0..50 {
            assert_eq!(xorshift64(&mut s1), xorshift64(&mut s2));
        }
    }

    // ── MetaLearner::new ─────────────────────────────────────────────────────

    #[test]
    fn test_new_initialises_weights() {
        let cfg = simple_config(4);
        let ml = MetaLearner::new(cfg);
        assert_eq!(ml.meta_params.weights.len(), 4);
        // All weights should be small positive values in [0, 0.01)
        for &w in &ml.meta_params.weights {
            assert!((0.0..0.01).contains(&w), "weight out of range: {w}");
        }
    }

    #[test]
    fn test_new_bias_is_zero() {
        let ml = MetaLearner::new(simple_config(3));
        assert_eq!(ml.meta_params.bias, 0.0);
    }

    #[test]
    fn test_new_history_empty() {
        let ml = MetaLearner::new(simple_config(3));
        assert!(ml.task_history.is_empty());
    }

    #[test]
    fn test_new_meta_step_zero() {
        let ml = MetaLearner::new(simple_config(3));
        assert_eq!(ml.meta_step, 0);
    }

    #[test]
    fn test_new_seed_determines_weights() {
        let cfg1 = MetaLearnerConfig {
            seed: 99,
            ..simple_config(5)
        };
        let cfg2 = MetaLearnerConfig {
            seed: 99,
            ..simple_config(5)
        };
        let ml1 = MetaLearner::new(cfg1);
        let ml2 = MetaLearner::new(cfg2);
        assert_eq!(ml1.meta_params.weights, ml2.meta_params.weights);
    }

    // ── Loss function ────────────────────────────────────────────────────────

    #[test]
    fn test_loss_regression_zero_residual() {
        let l = MetaLearner::loss(2.0, 2.0, &TaskType::Regression);
        assert!(l.abs() < 1e-12);
    }

    #[test]
    fn test_loss_regression_positive() {
        let l = MetaLearner::loss(3.0, 1.0, &TaskType::Regression);
        assert!((l - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_loss_classification_correct_sign() {
        // prediction strongly positive, label positive → low loss
        let l = MetaLearner::loss(5.0, 1.0, &TaskType::Classification { n_classes: 2 });
        assert!(l < 0.01);
    }

    #[test]
    fn test_loss_classification_wrong_sign() {
        // prediction strongly positive, label negative → high loss
        let l = MetaLearner::loss(5.0, -1.0, &TaskType::Classification { n_classes: 2 });
        assert!(l > 0.5);
    }

    #[test]
    fn test_loss_classification_non_negative() {
        for pred in [-2.0, 0.0, 2.0] {
            for label in [-1.0, 1.0] {
                let l = MetaLearner::loss(pred, label, &TaskType::Classification { n_classes: 3 });
                assert!(l >= 0.0, "loss was {l}");
            }
        }
    }

    #[test]
    fn test_loss_ranking_margin_satisfied() {
        // prediction and label same sign → margin satisfied → loss = 0
        let l = MetaLearner::loss(2.0, 1.0, &TaskType::Ranking);
        assert_eq!(l, 0.0);
    }

    #[test]
    fn test_loss_ranking_margin_violated() {
        // prediction and label opposite sign → loss > 0
        let l = MetaLearner::loss(-1.0, 1.0, &TaskType::Ranking);
        assert!(l > 0.0);
    }

    // ── gradient ────────────────────────────────────────────────────────────

    #[test]
    fn test_gradient_zero_residual() {
        let (dw, db) = MetaLearner::gradient(&[1.0, 2.0], 3.0, 3.0);
        assert!(dw.iter().all(|&g| g.abs() < 1e-12));
        assert!(db.abs() < 1e-12);
    }

    #[test]
    fn test_gradient_direction() {
        // prediction > label → gradient positive → weight should decrease
        let (dw, db) = MetaLearner::gradient(&[1.0], 2.0, 1.0);
        assert!(dw[0] > 0.0);
        assert!(db > 0.0);
    }

    #[test]
    fn test_gradient_length_matches_features() {
        let features = vec![0.1, 0.2, 0.3, 0.4];
        let (dw, _) = MetaLearner::gradient(&features, 1.0, 0.0);
        assert_eq!(dw.len(), features.len());
    }

    // ── predict ─────────────────────────────────────────────────────────────

    #[test]
    fn test_predict_zero_weights() {
        let ml = MetaLearner::new(MetaLearnerConfig {
            seed: 1,
            dims: 3,
            ..MetaLearnerConfig::default()
        });
        // override weights to zero for a predictable result
        let features = vec![1.0, 2.0, 3.0];
        // We cannot easily zero out weights without a setter, but we can
        // test through adaptation with known weights.
        let adaptation = TaskAdaptation {
            task_id: TaskId::new("t"),
            adapted_weights: vec![0.0, 0.0, 0.0],
            adapted_bias: 5.0,
            support_loss: 0.0,
            query_loss: 0.0,
            steps: 0,
        };
        let result = ml
            .predict(&features, Some(&adaptation))
            .expect("predict should succeed");
        assert!((result - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_predict_uses_adaptation_weights() {
        let ml = MetaLearner::new(simple_config(2));
        let adaptation = TaskAdaptation {
            task_id: TaskId::new("t"),
            adapted_weights: vec![1.0, 2.0],
            adapted_bias: 0.5,
            support_loss: 0.0,
            query_loss: 0.0,
            steps: 0,
        };
        // dot([1, 2], [3, 4]) + 0.5 = 3 + 8 + 0.5 = 11.5
        let result = ml.predict(&[3.0, 4.0], Some(&adaptation)).expect("ok");
        assert!((result - 11.5).abs() < 1e-10);
    }

    #[test]
    fn test_predict_dimension_mismatch() {
        let ml = MetaLearner::new(simple_config(3));
        let err = ml.predict(&[1.0, 2.0], None).unwrap_err();
        assert_eq!(
            err,
            MetaError::DimensionMismatch {
                expected: 3,
                got: 2
            }
        );
    }

    // ── adapt_to_task ────────────────────────────────────────────────────────

    #[test]
    fn test_adapt_regression_task_stores_history() {
        let mut ml = MetaLearner::new(simple_config(3));
        let task = make_regression_task("t1", 3, 4, 2);
        ml.adapt_to_task(&task).expect("adapt should succeed");
        assert!(ml.task_history.contains_key(&TaskId::new("t1")));
    }

    #[test]
    fn test_adapt_returns_correct_task_id() {
        let mut ml = MetaLearner::new(simple_config(3));
        let task = make_regression_task("my_task", 3, 3, 2);
        let adaptation = ml.adapt_to_task(&task).expect("ok");
        assert_eq!(adaptation.task_id, TaskId::new("my_task"));
    }

    #[test]
    fn test_adapt_steps_count() {
        let cfg = MetaLearnerConfig {
            inner_steps: 7,
            ..simple_config(3)
        };
        let mut ml = MetaLearner::new(cfg);
        let task = make_regression_task("t", 3, 3, 2);
        let a = ml.adapt_to_task(&task).expect("ok");
        assert_eq!(a.steps, 7);
    }

    #[test]
    fn test_adapt_empty_support_set_error() {
        let mut ml = MetaLearner::new(simple_config(3));
        let task = MetaTask::new(
            TaskId::new("empty"),
            vec![],
            vec![TaskExample::new(vec![0.0, 0.0, 0.0], 0.0)],
            TaskType::Regression,
        );
        assert_eq!(
            ml.adapt_to_task(&task).unwrap_err(),
            MetaError::EmptySupportSet
        );
    }

    #[test]
    fn test_adapt_empty_query_set_error() {
        let mut ml = MetaLearner::new(simple_config(3));
        let task = MetaTask::new(
            TaskId::new("empty_q"),
            vec![TaskExample::new(vec![0.0, 0.0, 0.0], 0.0)],
            vec![],
            TaskType::Regression,
        );
        assert_eq!(
            ml.adapt_to_task(&task).unwrap_err(),
            MetaError::EmptyQuerySet
        );
    }

    #[test]
    fn test_adapt_dimension_mismatch_error() {
        let mut ml = MetaLearner::new(simple_config(3));
        let task = MetaTask::new(
            TaskId::new("bad_dim"),
            vec![TaskExample::new(vec![1.0, 2.0], 0.0)], // dims=2, expected 3
            vec![TaskExample::new(vec![1.0, 2.0, 3.0], 0.0)],
            TaskType::Regression,
        );
        assert!(matches!(
            ml.adapt_to_task(&task).unwrap_err(),
            MetaError::DimensionMismatch {
                expected: 3,
                got: 2
            }
        ));
    }

    #[test]
    fn test_adapt_classification_task() {
        let mut ml = MetaLearner::new(simple_config(2));
        let task = make_classification_task("cls", 2);
        let a = ml.adapt_to_task(&task).expect("ok");
        assert!(a.support_loss >= 0.0);
        assert!(a.query_loss >= 0.0);
    }

    #[test]
    fn test_adapt_ranking_task() {
        let mut ml = MetaLearner::new(simple_config(2));
        let task = make_ranking_task("rnk", 2);
        let a = ml.adapt_to_task(&task).expect("ok");
        assert!(a.support_loss >= 0.0);
        assert!(a.query_loss >= 0.0);
    }

    #[test]
    fn test_adapt_does_not_change_meta_params() {
        let mut ml = MetaLearner::new(simple_config(3));
        let before = ml.meta_params.weights.clone();
        let task = make_regression_task("t", 3, 3, 2);
        ml.adapt_to_task(&task).expect("ok");
        assert_eq!(ml.meta_params.weights, before);
    }

    // ── meta_update ──────────────────────────────────────────────────────────

    #[test]
    fn test_meta_update_increments_step() {
        let mut ml = MetaLearner::new(simple_config(3));
        let task = make_regression_task("t", 3, 3, 2);
        let a = ml.adapt_to_task(&task).expect("ok");
        ml.meta_update(&[a]).expect("ok");
        assert_eq!(ml.meta_step, 1);
    }

    #[test]
    fn test_meta_update_empty_error() {
        let mut ml = MetaLearner::new(simple_config(3));
        assert_eq!(ml.meta_update(&[]).unwrap_err(), MetaError::NoAdaptations);
    }

    #[test]
    fn test_meta_update_moves_weights_toward_adapted() {
        let cfg = MetaLearnerConfig {
            inner_lr: 0.1,
            meta_lr: 1.0, // large lr so movement is obvious
            dims: 2,
            ..MetaLearnerConfig::default()
        };
        let mut ml = MetaLearner::new(cfg);
        // Force meta weights to zero for predictability
        ml.meta_params.weights = vec![0.0; 2];
        ml.meta_params.bias = 0.0;

        let adaptation = TaskAdaptation {
            task_id: TaskId::new("t"),
            adapted_weights: vec![1.0, 1.0],
            adapted_bias: 1.0,
            support_loss: 0.0,
            query_loss: 0.0,
            steps: 1,
        };
        ml.meta_update(&[adaptation]).expect("ok");
        // With meta_lr=1 the update is full: meta_w += 1*(avg - meta_w) = avg
        assert!((ml.meta_params.weights[0] - 1.0).abs() < 1e-10);
        assert!((ml.meta_params.bias - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_meta_update_dimension_mismatch() {
        let mut ml = MetaLearner::new(simple_config(3));
        let bad = TaskAdaptation {
            task_id: TaskId::new("bad"),
            adapted_weights: vec![1.0, 2.0], // wrong dims
            adapted_bias: 0.0,
            support_loss: 0.0,
            query_loss: 0.0,
            steps: 1,
        };
        assert!(matches!(
            ml.meta_update(&[bad]).unwrap_err(),
            MetaError::DimensionMismatch {
                expected: 3,
                got: 2
            }
        ));
    }

    // ── task_similarity ──────────────────────────────────────────────────────

    #[test]
    fn test_task_similarity_identical() {
        let a = TaskAdaptation {
            task_id: TaskId::new("a"),
            adapted_weights: vec![1.0, 0.0, 1.0],
            adapted_bias: 0.0,
            support_loss: 0.0,
            query_loss: 0.0,
            steps: 1,
        };
        let sim = MetaLearner::task_similarity(&a, &a);
        assert!((sim - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_task_similarity_orthogonal() {
        let a = TaskAdaptation {
            task_id: TaskId::new("a"),
            adapted_weights: vec![1.0, 0.0],
            adapted_bias: 0.0,
            support_loss: 0.0,
            query_loss: 0.0,
            steps: 1,
        };
        let b = TaskAdaptation {
            task_id: TaskId::new("b"),
            adapted_weights: vec![0.0, 1.0],
            adapted_bias: 0.0,
            support_loss: 0.0,
            query_loss: 0.0,
            steps: 1,
        };
        let sim = MetaLearner::task_similarity(&a, &b);
        assert!(sim.abs() < 1e-10);
    }

    #[test]
    fn test_task_similarity_opposite() {
        let a = TaskAdaptation {
            task_id: TaskId::new("a"),
            adapted_weights: vec![1.0, 0.0],
            adapted_bias: 0.0,
            support_loss: 0.0,
            query_loss: 0.0,
            steps: 1,
        };
        let b = TaskAdaptation {
            task_id: TaskId::new("b"),
            adapted_weights: vec![-1.0, 0.0],
            adapted_bias: 0.0,
            support_loss: 0.0,
            query_loss: 0.0,
            steps: 1,
        };
        let sim = MetaLearner::task_similarity(&a, &b);
        assert!((sim + 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_task_similarity_zero_vector() {
        let a = TaskAdaptation {
            task_id: TaskId::new("a"),
            adapted_weights: vec![0.0, 0.0],
            adapted_bias: 0.0,
            support_loss: 0.0,
            query_loss: 0.0,
            steps: 1,
        };
        let sim = MetaLearner::task_similarity(&a, &a);
        assert_eq!(sim, 0.0);
    }

    // ── best_task ────────────────────────────────────────────────────────────

    #[test]
    fn test_best_task_empty_history() {
        let ml = MetaLearner::new(simple_config(3));
        assert!(ml.best_task().is_none());
    }

    #[test]
    fn test_best_task_returns_lowest_query_loss() {
        let mut ml = MetaLearner::new(simple_config(3));
        for (id, ql) in [("t1", 0.5), ("t2", 0.1), ("t3", 0.8)] {
            ml.task_history.insert(
                TaskId::new(id),
                TaskAdaptation {
                    task_id: TaskId::new(id),
                    adapted_weights: vec![0.0; 3],
                    adapted_bias: 0.0,
                    support_loss: 0.0,
                    query_loss: ql,
                    steps: 1,
                },
            );
        }
        let (best_id, best_a) = ml.best_task().expect("should have a best task");
        assert_eq!(best_id, &TaskId::new("t2"));
        assert!((best_a.query_loss - 0.1).abs() < 1e-10);
    }

    // ── reset_task ───────────────────────────────────────────────────────────

    #[test]
    fn test_reset_task_removes_entry() {
        let mut ml = MetaLearner::new(simple_config(3));
        let task = make_regression_task("to_remove", 3, 3, 2);
        ml.adapt_to_task(&task).expect("ok");
        assert!(ml.task_history.contains_key(&TaskId::new("to_remove")));
        ml.reset_task(&TaskId::new("to_remove"));
        assert!(!ml.task_history.contains_key(&TaskId::new("to_remove")));
    }

    #[test]
    fn test_reset_task_nonexistent_is_noop() {
        let mut ml = MetaLearner::new(simple_config(3));
        ml.reset_task(&TaskId::new("ghost")); // should not panic
    }

    // ── stats ────────────────────────────────────────────────────────────────

    #[test]
    fn test_stats_empty() {
        let ml = MetaLearner::new(simple_config(3));
        let s = ml.stats();
        assert_eq!(s.total_tasks, 0);
        assert_eq!(s.meta_steps, 0);
        assert_eq!(s.avg_support_loss, 0.0);
        assert_eq!(s.avg_query_loss, 0.0);
        assert!(s.best_query_loss.is_infinite());
    }

    #[test]
    fn test_stats_after_adapt() {
        let mut ml = MetaLearner::new(simple_config(3));
        let t1 = make_regression_task("t1", 3, 4, 2);
        let t2 = make_regression_task("t2", 3, 4, 2);
        ml.adapt_to_task(&t1).expect("ok");
        ml.adapt_to_task(&t2).expect("ok");
        let s = ml.stats();
        assert_eq!(s.total_tasks, 2);
    }

    #[test]
    fn test_stats_best_query_loss_decreases() {
        let mut ml = MetaLearner::new(simple_config(3));
        ml.task_history.insert(
            TaskId::new("t1"),
            TaskAdaptation {
                task_id: TaskId::new("t1"),
                adapted_weights: vec![0.0; 3],
                adapted_bias: 0.0,
                support_loss: 1.0,
                query_loss: 0.3,
                steps: 1,
            },
        );
        let s = ml.stats();
        assert!((s.best_query_loss - 0.3).abs() < 1e-10);
    }

    // ── MetaError display ────────────────────────────────────────────────────

    #[test]
    fn test_meta_error_display() {
        assert!(!MetaError::EmptySupportSet.to_string().is_empty());
        assert!(!MetaError::EmptyQuerySet.to_string().is_empty());
        assert!(!MetaError::NoAdaptations.to_string().is_empty());
        assert!(!MetaError::DimensionMismatch {
            expected: 3,
            got: 2
        }
        .to_string()
        .is_empty());
    }

    // ── End-to-end: full adapt + meta_update cycle ──────────────────────────

    #[test]
    fn test_full_maml_cycle() {
        let mut ml = MetaLearner::new(simple_config(4));
        let tasks: Vec<MetaTask> = (0..3)
            .map(|i| make_regression_task(&format!("task_{i}"), 4, 5, 3))
            .collect();

        let adaptations: Vec<_> = tasks
            .iter()
            .map(|t| ml.adapt_to_task(t).expect("adapt ok"))
            .collect();

        ml.meta_update(&adaptations).expect("meta_update ok");

        assert_eq!(ml.meta_step, 1);
        assert_eq!(ml.task_history.len(), 3);

        let s = ml.stats();
        assert_eq!(s.total_tasks, 3);
        assert_eq!(s.meta_steps, 1);
        assert!(s.best_query_loss < f64::INFINITY);
    }
}
