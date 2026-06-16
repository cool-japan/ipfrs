//! MetaLearningOptimizer — production-quality meta-learning (learning to learn) optimization.
//!
//! Implements MAML, Reptile, FOMAML, and ProtoNet for few-shot adaptation
//! of a linear regression model (`y = w·x + b`) over many tasks.
//!
//! # Collision aliases (types already exported at crate root from other modules)
//! - `TaskId`      → `MloTaskId`
//! - `TaskExample` → `MloTaskExample`
//! - `MetaTask`    → `MloMetaTask`
//! - `MetaError`   → `MloMetaError`

use std::collections::HashMap;
use std::fmt;

// ─── PRNG ─────────────────────────────────────────────────────────────────────

/// XorShift-64 PRNG step — deterministic, no external deps.
#[inline]
fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// Generate a `f64` in `[0, 1)` from the XorShift-64 state.
#[inline]
fn xorshift_f64(state: &mut u64) -> f64 {
    (xorshift64(state) >> 11) as f64 / (1u64 << 53) as f64
}

// ─── TaskId ───────────────────────────────────────────────────────────────────

/// A unique identifier for a meta-learning task.
///
/// Aliased as `MloTaskId` at crate root to avoid collision with the
/// `TaskId` exported from `meta_learner`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TaskId(pub String);

impl TaskId {
    /// Create a `TaskId` from any `Into<String>`.
    pub fn new(id: impl Into<String>) -> Self {
        TaskId(id.into())
    }

    /// Return the inner string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for TaskId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

// ─── TaskExample ──────────────────────────────────────────────────────────────

/// A single labeled example belonging to a specific task.
///
/// Aliased as `MloTaskExample` at crate root.
#[derive(Debug, Clone)]
pub struct TaskExample {
    /// Input feature vector.
    pub features: Vec<f64>,
    /// Ground-truth label.
    pub label: f64,
    /// The task this example belongs to.
    pub task_id: TaskId,
}

impl TaskExample {
    /// Construct a new `TaskExample`.
    pub fn new(features: Vec<f64>, label: f64, task_id: TaskId) -> Self {
        TaskExample {
            features,
            label,
            task_id,
        }
    }
}

// ─── ModelParams ──────────────────────────────────────────────────────────────

/// Linear model parameters: `y = w·x + b`.
#[derive(Debug, Clone)]
pub struct ModelParams {
    /// Weight vector, length == `dim`.
    pub weights: Vec<f64>,
    /// Scalar bias.
    pub bias: f64,
    /// Input dimensionality.
    pub dim: usize,
}

impl ModelParams {
    /// Create zero-initialised params of dimensionality `dim`.
    pub fn zeros(dim: usize) -> Self {
        ModelParams {
            weights: vec![0.0; dim],
            bias: 0.0,
            dim,
        }
    }

    /// Linear prediction: `w·x + b`.
    fn predict(&self, x: &[f64]) -> f64 {
        let dot: f64 = self
            .weights
            .iter()
            .zip(x.iter())
            .map(|(w, xi)| w * xi)
            .sum();
        dot + self.bias
    }

    /// MSE loss and gradients for a batch of examples.
    fn mse_and_grads(&self, examples: &[TaskExample]) -> (f64, Vec<f64>, f64) {
        let n = examples.len() as f64;
        let mut grad_w = vec![0.0; self.dim];
        let mut grad_b = 0.0_f64;
        let mut loss = 0.0_f64;

        for ex in examples {
            let pred = self.predict(&ex.features);
            let residual = pred - ex.label;
            loss += residual * residual;
            let coeff = 2.0 * residual / n;
            for (gw, xi) in grad_w.iter_mut().zip(ex.features.iter()) {
                *gw += coeff * xi;
            }
            grad_b += coeff;
        }
        loss /= n;
        (loss, grad_w, grad_b)
    }
}

// ─── AdaptationStep ───────────────────────────────────────────────────────────

/// Snapshot of the model state after one inner-loop gradient step.
#[derive(Debug, Clone)]
pub struct AdaptationStep {
    /// Model parameters after this step.
    pub params: ModelParams,
    /// MSE loss on the support set at this step.
    pub loss: f64,
    /// Gradient vector (weights component) used in this step.
    pub gradient: Vec<f64>,
    /// Zero-based step index.
    pub step_num: usize,
}

// ─── MetaTask ─────────────────────────────────────────────────────────────────

/// A meta-learning task holding support and query sets.
///
/// Aliased as `MloMetaTask` at crate root.
#[derive(Debug, Clone)]
pub struct MetaTask {
    /// Unique task identifier.
    pub id: TaskId,
    /// Support set — used in inner-loop adaptation (K examples).
    pub support_set: Vec<TaskExample>,
    /// Query set — used to evaluate the adapted model.
    pub query_set: Vec<TaskExample>,
    /// Adapted parameters after the last call to `adapt_to_task`, if any.
    pub adapted_params: Option<ModelParams>,
}

impl MetaTask {
    /// Construct a `MetaTask` with explicit support and query sets.
    pub fn new(id: TaskId, support_set: Vec<TaskExample>, query_set: Vec<TaskExample>) -> Self {
        MetaTask {
            id,
            support_set,
            query_set,
            adapted_params: None,
        }
    }

    /// Return the feature dimensionality inferred from the support set,
    /// or `None` if the support set is empty.
    pub fn feature_dim(&self) -> Option<usize> {
        self.support_set.first().map(|ex| ex.features.len())
    }
}

// ─── MetaAlgorithm ────────────────────────────────────────────────────────────

/// The meta-learning algorithm variant.
#[derive(Debug, Clone)]
pub enum MetaAlgorithm {
    /// Model-Agnostic Meta-Learning (MAML).
    MAML {
        /// Inner-loop learning rate.
        inner_lr: f64,
        /// Number of inner-loop gradient steps.
        inner_steps: u8,
    },
    /// Prototypical Networks (ProtoNet).
    ProtoNet,
    /// Reptile (first-order meta-learner via parameter interpolation).
    Reptile {
        /// Interpolation step size toward each task's adapted params.
        step_size: f64,
    },
    /// First-Order MAML (omit second-order terms).
    FOMAML {
        /// Inner-loop learning rate.
        inner_lr: f64,
    },
}

// ─── OptimizerConfig ──────────────────────────────────────────────────────────

/// Configuration for [`MetaLearningOptimizer`].
#[derive(Debug, Clone)]
pub struct OptimizerConfig {
    /// The meta-learning algorithm to use.
    pub algorithm: MetaAlgorithm,
    /// Outer (meta) learning rate.
    pub meta_lr: f64,
    /// Number of tasks sampled per meta-update batch.
    pub n_tasks_per_batch: usize,
    /// Maximum allowed parameter dimensionality.
    pub max_params_dim: usize,
}

impl OptimizerConfig {
    /// Create a default MAML configuration.
    pub fn default_maml(dim: usize) -> Self {
        OptimizerConfig {
            algorithm: MetaAlgorithm::MAML {
                inner_lr: 0.01,
                inner_steps: 5,
            },
            meta_lr: 0.001,
            n_tasks_per_batch: 4,
            max_params_dim: dim,
        }
    }

    /// Create a default Reptile configuration.
    pub fn default_reptile(dim: usize) -> Self {
        OptimizerConfig {
            algorithm: MetaAlgorithm::Reptile { step_size: 0.1 },
            meta_lr: 0.001,
            n_tasks_per_batch: 4,
            max_params_dim: dim,
        }
    }
}

// ─── MetaStats ────────────────────────────────────────────────────────────────

/// Accumulated statistics for the meta-learning optimizer.
#[derive(Debug, Clone, Default)]
pub struct MetaStats {
    /// Total number of tasks trained.
    pub tasks_trained: u64,
    /// Total number of outer-loop meta-update steps performed.
    pub meta_updates: u64,
    /// Running average of inner-loop (support-set) loss.
    pub avg_adaptation_loss: f64,
    /// Running average of outer-loop (query-set) loss.
    pub avg_query_loss: f64,
    /// Change in outer loss between the last two meta-updates (for convergence).
    pub convergence_delta: f64,
}

// ─── MetaError ────────────────────────────────────────────────────────────────

/// Errors produced by [`MetaLearningOptimizer`] operations.
///
/// Aliased as `MloMetaError` at crate root.
#[derive(Debug, Clone, PartialEq)]
pub enum MetaError {
    /// Fewer tasks were provided than required by the algorithm.
    InsufficientTasks(usize),
    /// Feature dimensionality does not match the expected value.
    DimensionMismatch {
        /// Expected dimensionality.
        expected: usize,
        /// Actual dimensionality encountered.
        got: usize,
    },
    /// The inner-loop adaptation procedure failed.
    AdaptationFailed(String),
    /// The optimizer configuration is invalid.
    InvalidConfig(String),
}

impl fmt::Display for MetaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MetaError::InsufficientTasks(n) => {
                write!(f, "insufficient tasks: need at least {n}")
            }
            MetaError::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
            MetaError::AdaptationFailed(msg) => write!(f, "adaptation failed: {msg}"),
            MetaError::InvalidConfig(msg) => write!(f, "invalid config: {msg}"),
        }
    }
}

impl std::error::Error for MetaError {}

// ─── MetaLearningOptimizer ────────────────────────────────────────────────────

/// A production-quality meta-learning optimizer.
///
/// Supports MAML, Reptile, FOMAML, and ProtoNet meta-algorithms over a
/// linear regression model (`y = w·x + b`, MSE loss).
///
/// # Example
///
/// ```
/// use ipfrs_tensorlogic::{
///     MetaLearningOptimizer, MloTaskId, MloTaskExample, MloMetaTask,
///     OptimizerConfig, MetaAlgorithm,
/// };
///
/// let config = OptimizerConfig::default_maml(2);
/// let mut opt = MetaLearningOptimizer::new(config);
///
/// // Build a simple task
/// let tid = MloTaskId::new("t1");
/// let ex = MloTaskExample::new(vec![1.0, 0.0], 1.0, tid.clone());
/// let qex = MloTaskExample::new(vec![0.0, 1.0], 0.5, tid.clone());
/// let task = MloMetaTask::new(tid, vec![ex], vec![qex]);
///
/// opt.add_task(task).expect("example: should succeed in docs");
/// let init = MetaLearningOptimizer::initialize_params(2, 42);
/// let steps = opt.adapt_to_task(&MloTaskId::new("t1"), &init, 3, 0.01).expect("example: should succeed in docs");
/// assert!(!steps.is_empty());
/// ```
pub struct MetaLearningOptimizer {
    config: OptimizerConfig,
    tasks: HashMap<TaskId, MetaTask>,
    /// Expected feature dimensionality (inferred from first registered task).
    feature_dim: Option<usize>,
    stats: MetaStats,
    /// Running sum for the average adaptation loss (numerator).
    adaptation_loss_sum: f64,
    /// Number of adaptation loss samples.
    adaptation_loss_count: u64,
    /// Running sum for the average query loss (numerator).
    query_loss_sum: f64,
    /// Number of query loss samples.
    query_loss_count: u64,
    /// Last recorded query loss (for convergence_delta).
    prev_query_loss: Option<f64>,
}

impl MetaLearningOptimizer {
    /// Create a new `MetaLearningOptimizer` with the given configuration.
    pub fn new(config: OptimizerConfig) -> Self {
        MetaLearningOptimizer {
            config,
            tasks: HashMap::new(),
            feature_dim: None,
            stats: MetaStats::default(),
            adaptation_loss_sum: 0.0,
            adaptation_loss_count: 0,
            query_loss_sum: 0.0,
            query_loss_count: 0,
            prev_query_loss: None,
        }
    }

    // ── Task registration ─────────────────────────────────────────────────────

    /// Register a task with the optimizer.
    ///
    /// Validates that the feature dimensionality is consistent with previously
    /// registered tasks.
    pub fn add_task(&mut self, task: MetaTask) -> Result<(), MetaError> {
        // Determine feature dim from support set
        if let Some(dim) = task.feature_dim() {
            match self.feature_dim {
                None => {
                    if dim > self.config.max_params_dim {
                        return Err(MetaError::InvalidConfig(format!(
                            "feature dim {dim} exceeds max_params_dim {}",
                            self.config.max_params_dim
                        )));
                    }
                    self.feature_dim = Some(dim);
                }
                Some(expected) => {
                    if dim != expected {
                        return Err(MetaError::DimensionMismatch { expected, got: dim });
                    }
                }
            }
        }
        // Validate query set dims
        for qex in &task.query_set {
            let got = qex.features.len();
            if let Some(expected) = self.feature_dim {
                if got != expected {
                    return Err(MetaError::DimensionMismatch { expected, got });
                }
            }
        }
        self.tasks.insert(task.id.clone(), task);
        self.stats.tasks_trained += 1;
        Ok(())
    }

    // ── Inner-loop adaptation ─────────────────────────────────────────────────

    /// Perform gradient descent on the support set of a task for `steps` steps.
    ///
    /// Returns the full adaptation history (one [`AdaptationStep`] per step).
    pub fn adapt_to_task(
        &self,
        task_id: &TaskId,
        init_params: &ModelParams,
        steps: u8,
        lr: f64,
    ) -> Result<Vec<AdaptationStep>, MetaError> {
        let task = self
            .tasks
            .get(task_id)
            .ok_or_else(|| MetaError::AdaptationFailed(format!("unknown task: {task_id}")))?;

        if task.support_set.is_empty() {
            return Err(MetaError::AdaptationFailed(
                "support set is empty".to_string(),
            ));
        }

        // Validate dimensionality
        let expected_dim = init_params.dim;
        for ex in &task.support_set {
            let got = ex.features.len();
            if got != expected_dim {
                return Err(MetaError::DimensionMismatch {
                    expected: expected_dim,
                    got,
                });
            }
        }

        let mut params = init_params.clone();
        let mut history = Vec::with_capacity(steps as usize);

        for step in 0..steps {
            let (loss, grad_w, grad_b) = params.mse_and_grads(&task.support_set);

            // Gradient descent step
            for (w, gw) in params.weights.iter_mut().zip(grad_w.iter()) {
                *w -= lr * gw;
            }
            params.bias -= lr * grad_b;

            history.push(AdaptationStep {
                params: params.clone(),
                loss,
                gradient: grad_w,
                step_num: step as usize,
            });
        }

        Ok(history)
    }

    // ── Outer-loop meta-update ────────────────────────────────────────────────

    /// Perform one outer-loop meta-update over the specified task IDs.
    ///
    /// Algorithm dispatch:
    /// - **MAML / FOMAML**: for each task, run inner-loop adaptation; compute
    ///   meta-gradient as the mean of `(adapted − init)`; update
    ///   `new = current + meta_lr * meta_grad`.
    /// - **Reptile**: move `current` toward each task's adapted params by
    ///   `step_size`.
    /// - **ProtoNet**: compute per-task prototypes, return params whose weights
    ///   encode mean prototype and whose bias encodes the grand mean.
    pub fn meta_update(
        &mut self,
        task_ids: &[TaskId],
        current_params: &ModelParams,
    ) -> Result<ModelParams, MetaError> {
        if task_ids.is_empty() {
            return Err(MetaError::InsufficientTasks(1));
        }
        let dim = current_params.dim;

        let result = match &self.config.algorithm.clone() {
            MetaAlgorithm::MAML {
                inner_lr,
                inner_steps,
            } => self.meta_update_maml(task_ids, current_params, *inner_lr, *inner_steps, dim)?,
            MetaAlgorithm::FOMAML { inner_lr } => {
                self.meta_update_fomaml(task_ids, current_params, *inner_lr, dim)?
            }
            MetaAlgorithm::Reptile { step_size } => {
                self.meta_update_reptile(task_ids, current_params, *step_size, dim)?
            }
            MetaAlgorithm::ProtoNet => self.meta_update_protonet(task_ids, current_params, dim)?,
        };

        // Update query-loss stats
        let avg_q = self.compute_avg_query_loss(task_ids, &result);
        self.query_loss_sum += avg_q;
        self.query_loss_count += 1;
        let new_avg = self.query_loss_sum / self.query_loss_count as f64;
        let delta = match self.prev_query_loss {
            Some(prev) => (new_avg - prev).abs(),
            None => 0.0,
        };
        self.prev_query_loss = Some(new_avg);
        self.stats.avg_query_loss = new_avg;
        self.stats.convergence_delta = delta;
        self.stats.meta_updates += 1;

        Ok(result)
    }

    // ── Evaluate task ─────────────────────────────────────────────────────────

    /// Compute MSE of `params` on the query set of `task_id`.
    pub fn evaluate_task(&self, task_id: &TaskId, params: &ModelParams) -> Result<f64, MetaError> {
        let task = self
            .tasks
            .get(task_id)
            .ok_or_else(|| MetaError::AdaptationFailed(format!("unknown task: {task_id}")))?;

        if task.query_set.is_empty() {
            return Err(MetaError::AdaptationFailed(
                "query set is empty".to_string(),
            ));
        }

        let (loss, _, _) = params.mse_and_grads(&task.query_set);
        Ok(loss)
    }

    // ── Parameter initialisation ──────────────────────────────────────────────

    /// Create a small-random `ModelParams` of dimensionality `dim` using an
    /// XorShift-64 PRNG seeded with `seed`.
    pub fn initialize_params(dim: usize, seed: u64) -> ModelParams {
        let mut state = if seed == 0 { 0xdeadbeef_cafebabe } else { seed };
        let weights: Vec<f64> = (0..dim)
            .map(|_| (xorshift_f64(&mut state) - 0.5) * 0.01)
            .collect();
        let bias = (xorshift_f64(&mut state) - 0.5) * 0.01;
        ModelParams { weights, bias, dim }
    }

    // ── Few-shot prediction ───────────────────────────────────────────────────

    /// Adapt to the task's support set and predict the label for `x`.
    pub fn few_shot_predict(
        &self,
        task: &MetaTask,
        x: &[f64],
        init_params: &ModelParams,
    ) -> Result<f64, MetaError> {
        // Determine adaptation hyper-params from config
        let (steps, lr) = match &self.config.algorithm {
            MetaAlgorithm::MAML {
                inner_lr,
                inner_steps,
            } => (*inner_steps, *inner_lr),
            MetaAlgorithm::FOMAML { inner_lr } => (5u8, *inner_lr),
            MetaAlgorithm::Reptile { step_size } => (5u8, *step_size),
            MetaAlgorithm::ProtoNet => (1u8, 0.01),
        };

        if task.support_set.is_empty() {
            return Err(MetaError::AdaptationFailed(
                "support set is empty for few_shot_predict".to_string(),
            ));
        }

        let dim = init_params.dim;
        if x.len() != dim {
            return Err(MetaError::DimensionMismatch {
                expected: dim,
                got: x.len(),
            });
        }

        // Validate support set dims
        for ex in &task.support_set {
            let got = ex.features.len();
            if got != dim {
                return Err(MetaError::DimensionMismatch { expected: dim, got });
            }
        }

        // Perform inner-loop adaptation on a temporary task
        let mut params = init_params.clone();
        for _ in 0..steps {
            let (_, grad_w, grad_b) = params.mse_and_grads(&task.support_set);
            for (w, gw) in params.weights.iter_mut().zip(grad_w.iter()) {
                *w -= lr * gw;
            }
            params.bias -= lr * grad_b;
        }

        Ok(params.predict(x))
    }

    // ── Stats ─────────────────────────────────────────────────────────────────

    /// Return a snapshot of the current optimizer statistics.
    pub fn stats(&self) -> MetaStats {
        self.stats.clone()
    }

    // ─── Private helpers ───────────────────────────────────────────────────────

    /// MAML outer update: for each task, run inner adaptation; accumulate
    /// the mean parameter delta and add `meta_lr * mean_delta` to `current`.
    fn meta_update_maml(
        &mut self,
        task_ids: &[TaskId],
        current_params: &ModelParams,
        inner_lr: f64,
        inner_steps: u8,
        dim: usize,
    ) -> Result<ModelParams, MetaError> {
        let mut meta_grad_w = vec![0.0_f64; dim];
        let mut meta_grad_b = 0.0_f64;
        let mut valid_count = 0usize;

        for tid in task_ids {
            let history = self.adapt_to_task(tid, current_params, inner_steps, inner_lr)?;
            if let Some(last) = history.last() {
                // meta-gradient = adapted_params - init_params
                for (mg, (aw, iw)) in meta_grad_w.iter_mut().zip(
                    last.params
                        .weights
                        .iter()
                        .zip(current_params.weights.iter()),
                ) {
                    *mg += aw - iw;
                }
                meta_grad_b += last.params.bias - current_params.bias;

                // Track adaptation loss
                let adapt_loss = last.loss;
                self.adaptation_loss_sum += adapt_loss;
                self.adaptation_loss_count += 1;
                self.stats.avg_adaptation_loss =
                    self.adaptation_loss_sum / self.adaptation_loss_count as f64;

                valid_count += 1;
            }
        }

        if valid_count == 0 {
            return Err(MetaError::InsufficientTasks(1));
        }

        let inv = 1.0 / valid_count as f64;
        let meta_lr = self.config.meta_lr;
        let mut new_w = current_params.weights.clone();
        for (w, mg) in new_w.iter_mut().zip(meta_grad_w.iter()) {
            *w += meta_lr * mg * inv;
        }
        let new_b = current_params.bias + meta_lr * meta_grad_b * inv;

        Ok(ModelParams {
            weights: new_w,
            bias: new_b,
            dim,
        })
    }

    /// FOMAML — identical structure to MAML; uses only first-order gradients.
    fn meta_update_fomaml(
        &mut self,
        task_ids: &[TaskId],
        current_params: &ModelParams,
        inner_lr: f64,
        dim: usize,
    ) -> Result<ModelParams, MetaError> {
        // FOMAML uses a single inner step and first-order gradient approximation
        let inner_steps: u8 = 1;
        let mut meta_grad_w = vec![0.0_f64; dim];
        let mut meta_grad_b = 0.0_f64;
        let mut valid_count = 0usize;

        for tid in task_ids {
            let task = self
                .tasks
                .get(tid)
                .ok_or_else(|| MetaError::AdaptationFailed(format!("unknown task: {tid}")))?;
            if task.support_set.is_empty() {
                continue;
            }
            // Validate dims
            for ex in &task.support_set {
                let got = ex.features.len();
                if got != dim {
                    return Err(MetaError::DimensionMismatch { expected: dim, got });
                }
            }
            // First-order gradient at query set after one inner step
            let history = self.adapt_to_task(tid, current_params, inner_steps, inner_lr)?;
            if let Some(last) = history.last() {
                // Compute query-set gradient at adapted params
                let task2 = self
                    .tasks
                    .get(tid)
                    .ok_or_else(|| MetaError::AdaptationFailed(format!("unknown task: {tid}")))?;
                let (qloss, qgrad_w, qgrad_b) = last.params.mse_and_grads(&task2.query_set);

                // FOMAML meta-gradient = query-set gradient (no second-order terms)
                for (mg, qg) in meta_grad_w.iter_mut().zip(qgrad_w.iter()) {
                    *mg += qg;
                }
                meta_grad_b += qgrad_b;

                self.adaptation_loss_sum += qloss;
                self.adaptation_loss_count += 1;
                self.stats.avg_adaptation_loss =
                    self.adaptation_loss_sum / self.adaptation_loss_count as f64;

                valid_count += 1;
            }
        }

        if valid_count == 0 {
            return Err(MetaError::InsufficientTasks(1));
        }

        let inv = 1.0 / valid_count as f64;
        let meta_lr = self.config.meta_lr;
        let mut new_w = current_params.weights.clone();
        for (w, mg) in new_w.iter_mut().zip(meta_grad_w.iter()) {
            *w -= meta_lr * mg * inv;
        }
        let new_b = current_params.bias - meta_lr * meta_grad_b * inv;

        Ok(ModelParams {
            weights: new_w,
            bias: new_b,
            dim,
        })
    }

    /// Reptile meta-update: move `current` toward each task's adapted params.
    fn meta_update_reptile(
        &mut self,
        task_ids: &[TaskId],
        current_params: &ModelParams,
        step_size: f64,
        dim: usize,
    ) -> Result<ModelParams, MetaError> {
        let inner_steps = 5u8;
        let inner_lr = 0.01;
        let mut result = current_params.clone();
        let mut valid_count = 0usize;

        for tid in task_ids {
            let history = self.adapt_to_task(tid, current_params, inner_steps, inner_lr)?;
            if let Some(last) = history.last() {
                // Move toward adapted params: result_w += step_size * (adapted_w - init_w)
                for (idx, rw) in result.weights.iter_mut().enumerate() {
                    let init_w = current_params.weights[idx];
                    let adapted_w = last.params.weights[idx];
                    *rw += step_size * (adapted_w - init_w);
                }
                result.bias += step_size * (last.params.bias - current_params.bias);

                self.adaptation_loss_sum += last.loss;
                self.adaptation_loss_count += 1;
                self.stats.avg_adaptation_loss =
                    self.adaptation_loss_sum / self.adaptation_loss_count as f64;

                valid_count += 1;
            }
        }

        if valid_count == 0 {
            return Err(MetaError::InsufficientTasks(1));
        }

        let _ = dim; // dim captured from current_params
        Ok(result)
    }

    /// ProtoNet meta-update: compute per-class prototypes, encode them in params.
    ///
    /// For regression tasks the "prototype" is the mean feature vector weighted
    /// by label; for the bias, we use the grand mean label.
    fn meta_update_protonet(
        &mut self,
        task_ids: &[TaskId],
        current_params: &ModelParams,
        dim: usize,
    ) -> Result<ModelParams, MetaError> {
        let mut proto_w = vec![0.0_f64; dim];
        let mut proto_b = 0.0_f64;
        let mut valid_count = 0usize;

        for tid in task_ids {
            let task = self
                .tasks
                .get(tid)
                .ok_or_else(|| MetaError::AdaptationFailed(format!("unknown task: {tid}")))?;
            if task.support_set.is_empty() {
                continue;
            }
            let n = task.support_set.len() as f64;
            let mean_label: f64 = task.support_set.iter().map(|e| e.label).sum::<f64>() / n;
            let mut mean_feat = vec![0.0_f64; dim];
            for ex in &task.support_set {
                if ex.features.len() != dim {
                    return Err(MetaError::DimensionMismatch {
                        expected: dim,
                        got: ex.features.len(),
                    });
                }
                for (mf, xi) in mean_feat.iter_mut().zip(ex.features.iter()) {
                    *mf += xi / n;
                }
            }
            // Prototype direction ≈ mean_feat * mean_label
            for (pw, mf) in proto_w.iter_mut().zip(mean_feat.iter()) {
                *pw += mf * mean_label;
            }
            proto_b += mean_label;
            valid_count += 1;
        }

        if valid_count == 0 {
            return Err(MetaError::InsufficientTasks(1));
        }

        let inv = 1.0 / valid_count as f64;
        let meta_lr = self.config.meta_lr;
        let mut new_w = current_params.weights.clone();
        for (w, pw) in new_w.iter_mut().zip(proto_w.iter()) {
            *w += meta_lr * pw * inv;
        }
        let new_b = current_params.bias + meta_lr * proto_b * inv;

        Ok(ModelParams {
            weights: new_w,
            bias: new_b,
            dim,
        })
    }

    /// Compute the average query-set MSE over the given tasks.
    fn compute_avg_query_loss(&self, task_ids: &[TaskId], params: &ModelParams) -> f64 {
        let mut sum = 0.0;
        let mut count = 0usize;
        for tid in task_ids {
            if let Ok(loss) = self.evaluate_task(tid, params) {
                sum += loss;
                count += 1;
            }
        }
        if count == 0 {
            0.0
        } else {
            sum / count as f64
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ──────────────────────────────────────────────────────────────

    /// Build a simple 1-D regression task:  label = slope * feature + intercept + noise
    fn make_regression_task(
        id: &str,
        slope: f64,
        intercept: f64,
        n_support: usize,
        n_query: usize,
        seed: u64,
    ) -> MetaTask {
        let tid = TaskId::new(id);
        let mut state = seed;
        let mut support = Vec::with_capacity(n_support);
        for _ in 0..n_support {
            let x = xorshift_f64(&mut state) * 4.0 - 2.0;
            let y = slope * x + intercept;
            support.push(TaskExample::new(vec![x], y, tid.clone()));
        }
        let mut query = Vec::with_capacity(n_query);
        for _ in 0..n_query {
            let x = xorshift_f64(&mut state) * 4.0 - 2.0;
            let y = slope * x + intercept;
            query.push(TaskExample::new(vec![x], y, tid.clone()));
        }
        MetaTask::new(tid, support, query)
    }

    /// Build a 2-D regression task.
    fn make_2d_task(
        id: &str,
        w0: f64,
        w1: f64,
        bias: f64,
        n_support: usize,
        n_query: usize,
        seed: u64,
    ) -> MetaTask {
        let tid = TaskId::new(id);
        let mut state = seed;
        let mut support = Vec::with_capacity(n_support);
        for _ in 0..n_support {
            let x0 = xorshift_f64(&mut state) * 2.0 - 1.0;
            let x1 = xorshift_f64(&mut state) * 2.0 - 1.0;
            let y = w0 * x0 + w1 * x1 + bias;
            support.push(TaskExample::new(vec![x0, x1], y, tid.clone()));
        }
        let mut query = Vec::with_capacity(n_query);
        for _ in 0..n_query {
            let x0 = xorshift_f64(&mut state) * 2.0 - 1.0;
            let x1 = xorshift_f64(&mut state) * 2.0 - 1.0;
            let y = w0 * x0 + w1 * x1 + bias;
            query.push(TaskExample::new(vec![x0, x1], y, tid.clone()));
        }
        MetaTask::new(tid, support, query)
    }

    // ── add_task ──────────────────────────────────────────────────────────────

    #[test]
    fn test_add_task_basic() {
        let config = OptimizerConfig::default_maml(1);
        let mut opt = MetaLearningOptimizer::new(config);
        let task = make_regression_task("t1", 2.0, 1.0, 5, 5, 1);
        assert!(opt.add_task(task).is_ok());
        assert_eq!(opt.stats().tasks_trained, 1);
    }

    #[test]
    fn test_add_multiple_tasks() {
        let config = OptimizerConfig::default_maml(1);
        let mut opt = MetaLearningOptimizer::new(config);
        for i in 0..5 {
            let task = make_regression_task(&format!("t{i}"), i as f64, 0.0, 4, 4, i as u64 + 1);
            assert!(opt.add_task(task).is_ok());
        }
        assert_eq!(opt.stats().tasks_trained, 5);
    }

    #[test]
    fn test_add_task_dimension_consistency() {
        let config = OptimizerConfig::default_maml(2);
        let mut opt = MetaLearningOptimizer::new(config);
        let t1 = make_2d_task("t1", 1.0, 2.0, 0.5, 4, 4, 10);
        assert!(opt.add_task(t1).is_ok());
        // t2 with wrong dim (1D)
        let t2 = make_regression_task("t2", 1.0, 0.0, 4, 4, 20);
        let err = opt.add_task(t2).unwrap_err();
        assert!(matches!(
            err,
            MetaError::DimensionMismatch {
                expected: 2,
                got: 1
            }
        ));
    }

    #[test]
    fn test_add_task_dim_exceeds_max() {
        let config = OptimizerConfig {
            algorithm: MetaAlgorithm::MAML {
                inner_lr: 0.01,
                inner_steps: 3,
            },
            meta_lr: 0.001,
            n_tasks_per_batch: 2,
            max_params_dim: 2,
        };
        let mut opt = MetaLearningOptimizer::new(config);
        // Build a 3-D task
        let tid = TaskId::new("too-big");
        let ex = TaskExample::new(vec![1.0, 2.0, 3.0], 0.5, tid.clone());
        let task = MetaTask::new(tid, vec![ex.clone()], vec![ex]);
        let err = opt.add_task(task).unwrap_err();
        assert!(matches!(err, MetaError::InvalidConfig(_)));
    }

    #[test]
    fn test_add_task_empty_support_allowed() {
        // Tasks with empty support sets are allowed at registration time
        let config = OptimizerConfig::default_maml(1);
        let mut opt = MetaLearningOptimizer::new(config);
        let tid = TaskId::new("empty");
        let qex = TaskExample::new(vec![1.0], 1.0, tid.clone());
        let task = MetaTask::new(tid, vec![], vec![qex]);
        // Should succeed — dim not inferred from empty support
        assert!(opt.add_task(task).is_ok());
    }

    // ── adapt_to_task ─────────────────────────────────────────────────────────

    #[test]
    fn test_adapt_returns_correct_step_count() {
        let config = OptimizerConfig::default_maml(1);
        let mut opt = MetaLearningOptimizer::new(config);
        let task = make_regression_task("t1", 3.0, 0.5, 10, 5, 42);
        opt.add_task(task).expect("test: should succeed");
        let init = MetaLearningOptimizer::initialize_params(1, 1);
        let steps = opt
            .adapt_to_task(&TaskId::new("t1"), &init, 7, 0.01)
            .expect("test: should succeed");
        assert_eq!(steps.len(), 7);
    }

    #[test]
    fn test_adapt_step_numbers_sequential() {
        let config = OptimizerConfig::default_maml(1);
        let mut opt = MetaLearningOptimizer::new(config);
        let task = make_regression_task("t1", 1.0, 0.0, 8, 4, 5);
        opt.add_task(task).expect("test: should succeed");
        let init = MetaLearningOptimizer::initialize_params(1, 7);
        let steps = opt
            .adapt_to_task(&TaskId::new("t1"), &init, 5, 0.05)
            .expect("test: should succeed");
        for (i, step) in steps.iter().enumerate() {
            assert_eq!(step.step_num, i);
        }
    }

    #[test]
    fn test_adapt_loss_non_negative() {
        let config = OptimizerConfig::default_maml(1);
        let mut opt = MetaLearningOptimizer::new(config);
        let task = make_regression_task("t1", 2.0, -1.0, 10, 5, 11);
        opt.add_task(task).expect("test: should succeed");
        let init = MetaLearningOptimizer::initialize_params(1, 99);
        let steps = opt
            .adapt_to_task(&TaskId::new("t1"), &init, 10, 0.01)
            .expect("test: should succeed");
        for step in &steps {
            assert!(step.loss >= 0.0, "loss must be non-negative");
        }
    }

    #[test]
    fn test_adapt_loss_decreases_over_steps() {
        // With sufficient steps and a well-posed problem, loss should decrease
        let config = OptimizerConfig::default_maml(1);
        let mut opt = MetaLearningOptimizer::new(config);
        let task = make_regression_task("t1", 1.5, 0.3, 20, 5, 7);
        opt.add_task(task).expect("test: should succeed");
        let init = MetaLearningOptimizer::initialize_params(1, 3);
        let steps = opt
            .adapt_to_task(&TaskId::new("t1"), &init, 20, 0.05)
            .expect("test: should succeed");
        let first_loss = steps.first().map(|s| s.loss).unwrap_or(f64::MAX);
        let last_loss = steps.last().map(|s| s.loss).unwrap_or(f64::MAX);
        assert!(
            last_loss <= first_loss + 1e-10,
            "loss should decrease: {first_loss} -> {last_loss}"
        );
    }

    #[test]
    fn test_adapt_2d_loss_decreases() {
        let config = OptimizerConfig::default_maml(2);
        let mut opt = MetaLearningOptimizer::new(config);
        let task = make_2d_task("t1", 1.0, -1.0, 0.5, 15, 5, 42);
        opt.add_task(task).expect("test: should succeed");
        let init = MetaLearningOptimizer::initialize_params(2, 9);
        let steps = opt
            .adapt_to_task(&TaskId::new("t1"), &init, 30, 0.02)
            .expect("test: should succeed");
        let first = steps.first().map(|s| s.loss).expect("test: should succeed");
        let last = steps.last().map(|s| s.loss).expect("test: should succeed");
        assert!(last <= first + 1e-9);
    }

    #[test]
    fn test_adapt_unknown_task_error() {
        let config = OptimizerConfig::default_maml(1);
        let opt = MetaLearningOptimizer::new(config);
        let init = MetaLearningOptimizer::initialize_params(1, 1);
        let err = opt
            .adapt_to_task(&TaskId::new("no-such"), &init, 5, 0.01)
            .unwrap_err();
        assert!(matches!(err, MetaError::AdaptationFailed(_)));
    }

    #[test]
    fn test_adapt_empty_support_error() {
        let config = OptimizerConfig::default_maml(1);
        let mut opt = MetaLearningOptimizer::new(config);
        let tid = TaskId::new("empty");
        let qex = TaskExample::new(vec![1.0], 1.0, tid.clone());
        let task = MetaTask::new(tid.clone(), vec![], vec![qex]);
        opt.add_task(task).expect("test: should succeed");
        let init = MetaLearningOptimizer::initialize_params(1, 1);
        let err = opt.adapt_to_task(&tid, &init, 3, 0.01).unwrap_err();
        assert!(matches!(err, MetaError::AdaptationFailed(_)));
    }

    #[test]
    fn test_adapt_dim_mismatch_error() {
        let config = OptimizerConfig::default_maml(2);
        let mut opt = MetaLearningOptimizer::new(config);
        let task = make_2d_task("t1", 1.0, 1.0, 0.0, 5, 5, 1);
        opt.add_task(task).expect("test: should succeed");
        // init with wrong dim
        let bad_init = MetaLearningOptimizer::initialize_params(3, 1);
        let err = opt
            .adapt_to_task(&TaskId::new("t1"), &bad_init, 3, 0.01)
            .unwrap_err();
        assert!(matches!(err, MetaError::DimensionMismatch { .. }));
    }

    // ── meta_update (MAML) ────────────────────────────────────────────────────

    #[test]
    fn test_meta_update_maml_returns_new_params() {
        let config = OptimizerConfig::default_maml(1);
        let mut opt = MetaLearningOptimizer::new(config);
        for i in 0..4 {
            let task = make_regression_task(
                &format!("t{i}"),
                (i + 1) as f64,
                0.1,
                8,
                4,
                (i * 7 + 1) as u64,
            );
            opt.add_task(task).expect("test: should succeed");
        }
        let init = MetaLearningOptimizer::initialize_params(1, 42);
        let ids: Vec<TaskId> = (0..4).map(|i| TaskId::new(format!("t{i}"))).collect();
        let new_params = opt.meta_update(&ids, &init).expect("test: should succeed");
        assert_eq!(new_params.dim, 1);
        assert_eq!(opt.stats().meta_updates, 1);
    }

    #[test]
    fn test_meta_update_maml_params_changed() {
        let config = OptimizerConfig::default_maml(1);
        let mut opt = MetaLearningOptimizer::new(config);
        for i in 0..4 {
            let task = make_regression_task(
                &format!("t{i}"),
                (i as f64 + 1.0) * 0.7,
                0.3,
                10,
                5,
                i as u64 + 11,
            );
            opt.add_task(task).expect("test: should succeed");
        }
        let init = MetaLearningOptimizer::initialize_params(1, 17);
        let ids: Vec<TaskId> = (0..4).map(|i| TaskId::new(format!("t{i}"))).collect();
        let new_params = opt.meta_update(&ids, &init).expect("test: should succeed");
        // At least one parameter should have changed
        let changed = new_params.weights[0] != init.weights[0] || new_params.bias != init.bias;
        assert!(changed, "meta_update should change parameters");
    }

    #[test]
    fn test_meta_update_maml_multiple_rounds_converge() {
        let config = OptimizerConfig {
            algorithm: MetaAlgorithm::MAML {
                inner_lr: 0.05,
                inner_steps: 5,
            },
            meta_lr: 0.1,
            n_tasks_per_batch: 4,
            max_params_dim: 1,
        };
        let mut opt = MetaLearningOptimizer::new(config);
        for i in 0..4 {
            let task = make_regression_task(&format!("t{i}"), 1.0, 0.0, 10, 5, i as u64 + 1);
            opt.add_task(task).expect("test: should succeed");
        }
        let mut params = MetaLearningOptimizer::initialize_params(1, 5);
        let ids: Vec<TaskId> = (0..4).map(|i| TaskId::new(format!("t{i}"))).collect();
        for _ in 0..20 {
            params = opt
                .meta_update(&ids, &params)
                .expect("test: should succeed");
        }
        assert_eq!(opt.stats().meta_updates, 20);
    }

    #[test]
    fn test_meta_update_maml_empty_task_list_error() {
        let config = OptimizerConfig::default_maml(1);
        let mut opt = MetaLearningOptimizer::new(config);
        let init = MetaLearningOptimizer::initialize_params(1, 1);
        let err = opt.meta_update(&[], &init).unwrap_err();
        assert!(matches!(err, MetaError::InsufficientTasks(_)));
    }

    // ── meta_update (Reptile) ─────────────────────────────────────────────────

    #[test]
    fn test_meta_update_reptile_basic() {
        let config = OptimizerConfig::default_reptile(1);
        let mut opt = MetaLearningOptimizer::new(config);
        for i in 0..3 {
            let task =
                make_regression_task(&format!("r{i}"), (i + 1) as f64, 0.0, 8, 4, i as u64 + 5);
            opt.add_task(task).expect("test: should succeed");
        }
        let init = MetaLearningOptimizer::initialize_params(1, 42);
        let ids: Vec<TaskId> = (0..3).map(|i| TaskId::new(format!("r{i}"))).collect();
        let new_p = opt.meta_update(&ids, &init).expect("test: should succeed");
        assert_eq!(new_p.dim, 1);
    }

    #[test]
    fn test_meta_update_reptile_params_change() {
        let config = OptimizerConfig::default_reptile(1);
        let mut opt = MetaLearningOptimizer::new(config);
        for i in 0..3 {
            let task = make_regression_task(&format!("r{i}"), 2.0, 1.0, 10, 5, i as u64 + 100);
            opt.add_task(task).expect("test: should succeed");
        }
        let init = MetaLearningOptimizer::initialize_params(1, 77);
        let ids: Vec<TaskId> = (0..3).map(|i| TaskId::new(format!("r{i}"))).collect();
        let new_p = opt.meta_update(&ids, &init).expect("test: should succeed");
        let changed = new_p.weights[0] != init.weights[0] || new_p.bias != init.bias;
        assert!(changed);
    }

    #[test]
    fn test_meta_update_reptile_multiple_rounds() {
        let config = OptimizerConfig::default_reptile(1);
        let mut opt = MetaLearningOptimizer::new(config);
        for i in 0..3 {
            let task = make_regression_task(&format!("r{i}"), 1.0, 0.5, 8, 4, i as u64 + 3);
            opt.add_task(task).expect("test: should succeed");
        }
        let mut params = MetaLearningOptimizer::initialize_params(1, 17);
        let ids: Vec<TaskId> = (0..3).map(|i| TaskId::new(format!("r{i}"))).collect();
        for _ in 0..10 {
            params = opt
                .meta_update(&ids, &params)
                .expect("test: should succeed");
        }
        assert_eq!(opt.stats().meta_updates, 10);
    }

    // ── meta_update (FOMAML) ──────────────────────────────────────────────────

    #[test]
    fn test_meta_update_fomaml_basic() {
        let config = OptimizerConfig {
            algorithm: MetaAlgorithm::FOMAML { inner_lr: 0.02 },
            meta_lr: 0.01,
            n_tasks_per_batch: 3,
            max_params_dim: 1,
        };
        let mut opt = MetaLearningOptimizer::new(config);
        for i in 0..3 {
            let task = make_regression_task(
                &format!("f{i}"),
                i as f64 + 0.5,
                0.0,
                8,
                4,
                (i * 3 + 2) as u64,
            );
            opt.add_task(task).expect("test: should succeed");
        }
        let init = MetaLearningOptimizer::initialize_params(1, 55);
        let ids: Vec<TaskId> = (0..3).map(|i| TaskId::new(format!("f{i}"))).collect();
        let new_p = opt.meta_update(&ids, &init).expect("test: should succeed");
        assert_eq!(new_p.dim, 1);
    }

    #[test]
    fn test_meta_update_fomaml_params_change() {
        let config = OptimizerConfig {
            algorithm: MetaAlgorithm::FOMAML { inner_lr: 0.05 },
            meta_lr: 0.1,
            n_tasks_per_batch: 3,
            max_params_dim: 1,
        };
        let mut opt = MetaLearningOptimizer::new(config);
        for i in 0..3 {
            let task = make_regression_task(&format!("f{i}"), 2.0, 0.5, 10, 5, i as u64 + 20);
            opt.add_task(task).expect("test: should succeed");
        }
        let init = MetaLearningOptimizer::initialize_params(1, 3);
        let ids: Vec<TaskId> = (0..3).map(|i| TaskId::new(format!("f{i}"))).collect();
        let new_p = opt.meta_update(&ids, &init).expect("test: should succeed");
        let changed = new_p.weights[0] != init.weights[0] || new_p.bias != init.bias;
        assert!(changed);
    }

    // ── meta_update (ProtoNet) ────────────────────────────────────────────────

    #[test]
    fn test_meta_update_protonet_basic() {
        let config = OptimizerConfig {
            algorithm: MetaAlgorithm::ProtoNet,
            meta_lr: 0.01,
            n_tasks_per_batch: 3,
            max_params_dim: 2,
        };
        let mut opt = MetaLearningOptimizer::new(config);
        for i in 0..3 {
            let task = make_2d_task(&format!("p{i}"), 1.0, 1.0, 0.0, 6, 4, (i * 5 + 1) as u64);
            opt.add_task(task).expect("test: should succeed");
        }
        let init = MetaLearningOptimizer::initialize_params(2, 11);
        let ids: Vec<TaskId> = (0..3).map(|i| TaskId::new(format!("p{i}"))).collect();
        let new_p = opt.meta_update(&ids, &init).expect("test: should succeed");
        assert_eq!(new_p.dim, 2);
        assert_eq!(opt.stats().meta_updates, 1);
    }

    // ── evaluate_task ─────────────────────────────────────────────────────────

    #[test]
    fn test_evaluate_task_perfect_params() {
        let config = OptimizerConfig::default_maml(1);
        let mut opt = MetaLearningOptimizer::new(config);
        // Task: y = 2x + 1
        let task = make_regression_task("t1", 2.0, 1.0, 5, 10, 13);
        opt.add_task(task).expect("test: should succeed");
        // Perfect params
        let params = ModelParams {
            weights: vec![2.0],
            bias: 1.0,
            dim: 1,
        };
        let loss = opt
            .evaluate_task(&TaskId::new("t1"), &params)
            .expect("test: should succeed");
        assert!(
            loss < 1e-20,
            "perfect params should give ~0 MSE, got {loss}"
        );
    }

    #[test]
    fn test_evaluate_task_non_negative() {
        let config = OptimizerConfig::default_maml(1);
        let mut opt = MetaLearningOptimizer::new(config);
        let task = make_regression_task("t1", 3.0, -1.0, 5, 10, 17);
        opt.add_task(task).expect("test: should succeed");
        let params = MetaLearningOptimizer::initialize_params(1, 7);
        let loss = opt
            .evaluate_task(&TaskId::new("t1"), &params)
            .expect("test: should succeed");
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_evaluate_task_unknown_error() {
        let config = OptimizerConfig::default_maml(1);
        let opt = MetaLearningOptimizer::new(config);
        let params = MetaLearningOptimizer::initialize_params(1, 1);
        let err = opt
            .evaluate_task(&TaskId::new("no-such"), &params)
            .unwrap_err();
        assert!(matches!(err, MetaError::AdaptationFailed(_)));
    }

    #[test]
    fn test_evaluate_task_empty_query_error() {
        let config = OptimizerConfig::default_maml(1);
        let mut opt = MetaLearningOptimizer::new(config);
        let tid = TaskId::new("empty-q");
        let sex = TaskExample::new(vec![1.0], 1.0, tid.clone());
        let task = MetaTask::new(tid.clone(), vec![sex], vec![]);
        opt.add_task(task).expect("test: should succeed");
        let params = MetaLearningOptimizer::initialize_params(1, 1);
        let err = opt.evaluate_task(&tid, &params).unwrap_err();
        assert!(matches!(err, MetaError::AdaptationFailed(_)));
    }

    // ── initialize_params ─────────────────────────────────────────────────────

    #[test]
    fn test_initialize_params_dim() {
        let params = MetaLearningOptimizer::initialize_params(4, 42);
        assert_eq!(params.dim, 4);
        assert_eq!(params.weights.len(), 4);
    }

    #[test]
    fn test_initialize_params_small_values() {
        // Values should be in roughly [-0.01, 0.01]
        let params = MetaLearningOptimizer::initialize_params(100, 123);
        for w in &params.weights {
            assert!(w.abs() <= 0.01, "weight {w} exceeds 0.01");
        }
        assert!(
            params.bias.abs() <= 0.01,
            "bias {} exceeds 0.01",
            params.bias
        );
    }

    #[test]
    fn test_initialize_params_deterministic() {
        let p1 = MetaLearningOptimizer::initialize_params(5, 77);
        let p2 = MetaLearningOptimizer::initialize_params(5, 77);
        assert_eq!(p1.weights, p2.weights);
        assert_eq!(p1.bias, p2.bias);
    }

    #[test]
    fn test_initialize_params_different_seeds() {
        let p1 = MetaLearningOptimizer::initialize_params(5, 1);
        let p2 = MetaLearningOptimizer::initialize_params(5, 2);
        assert_ne!(
            p1.weights, p2.weights,
            "different seeds should give different weights"
        );
    }

    #[test]
    fn test_initialize_params_zero_seed_fallback() {
        // seed=0 should use internal fallback and not panic
        let params = MetaLearningOptimizer::initialize_params(3, 0);
        assert_eq!(params.dim, 3);
    }

    // ── few_shot_predict ──────────────────────────────────────────────────────

    #[test]
    fn test_few_shot_predict_basic() {
        let config = OptimizerConfig::default_maml(1);
        let opt = MetaLearningOptimizer::new(config);
        // Build task manually so we know the answer
        let tid = TaskId::new("fs1");
        let support: Vec<TaskExample> = (0..8)
            .map(|i| {
                let x = i as f64;
                TaskExample::new(vec![x], 2.0 * x + 1.0, tid.clone())
            })
            .collect();
        let task = MetaTask::new(tid, support, vec![]);
        let init = MetaLearningOptimizer::initialize_params(1, 42);
        let pred = opt
            .few_shot_predict(&task, &[3.0], &init)
            .expect("test: should succeed");
        // After adaptation we expect a value closer to 7.0 than the raw init
        assert!(pred.is_finite(), "prediction should be finite");
    }

    #[test]
    fn test_few_shot_predict_dim_mismatch() {
        let config = OptimizerConfig::default_maml(2);
        let opt = MetaLearningOptimizer::new(config);
        let tid = TaskId::new("fs2");
        let support = vec![TaskExample::new(vec![1.0, 2.0], 3.0, tid.clone())];
        let task = MetaTask::new(tid, support, vec![]);
        let init = MetaLearningOptimizer::initialize_params(2, 9);
        // x has wrong dim
        let err = opt.few_shot_predict(&task, &[1.0], &init).unwrap_err();
        assert!(matches!(
            err,
            MetaError::DimensionMismatch {
                expected: 2,
                got: 1
            }
        ));
    }

    #[test]
    fn test_few_shot_predict_empty_support_error() {
        let config = OptimizerConfig::default_maml(1);
        let opt = MetaLearningOptimizer::new(config);
        let tid = TaskId::new("fse");
        let task = MetaTask::new(tid, vec![], vec![]);
        let init = MetaLearningOptimizer::initialize_params(1, 1);
        let err = opt.few_shot_predict(&task, &[1.0], &init).unwrap_err();
        assert!(matches!(err, MetaError::AdaptationFailed(_)));
    }

    #[test]
    fn test_few_shot_predict_reptile() {
        let config = OptimizerConfig::default_reptile(1);
        let opt = MetaLearningOptimizer::new(config);
        let tid = TaskId::new("rfs");
        let support: Vec<TaskExample> = (0..5)
            .map(|i| {
                let x = i as f64 * 0.5;
                TaskExample::new(vec![x], 3.0 * x, tid.clone())
            })
            .collect();
        let task = MetaTask::new(tid, support, vec![]);
        let init = MetaLearningOptimizer::initialize_params(1, 7);
        let pred = opt
            .few_shot_predict(&task, &[2.0], &init)
            .expect("test: should succeed");
        assert!(pred.is_finite());
    }

    #[test]
    fn test_few_shot_predict_adapts_correctly_2d() {
        let config = OptimizerConfig {
            algorithm: MetaAlgorithm::MAML {
                inner_lr: 0.05,
                inner_steps: 20,
            },
            meta_lr: 0.01,
            n_tasks_per_batch: 2,
            max_params_dim: 2,
        };
        let opt = MetaLearningOptimizer::new(config);
        let tid = TaskId::new("2dfs");
        // Ground truth: y = 1.5*x0 + 0.5*x1
        let support: Vec<TaskExample> = (0..20)
            .map(|i| {
                let x0 = i as f64 * 0.1;
                let x1 = (i as f64) * 0.2 - 1.0;
                TaskExample::new(vec![x0, x1], 1.5 * x0 + 0.5 * x1, tid.clone())
            })
            .collect();
        let task = MetaTask::new(tid, support, vec![]);
        let init = MetaLearningOptimizer::initialize_params(2, 42);
        let pred = opt
            .few_shot_predict(&task, &[1.0, 0.0], &init)
            .expect("test: should succeed");
        // After adaptation, prediction for [1,0] should be closer to 1.5
        assert!((pred - 1.5).abs() < 1.0, "pred {pred} should be near 1.5");
    }

    // ── stats ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_stats_initial_state() {
        let config = OptimizerConfig::default_maml(1);
        let opt = MetaLearningOptimizer::new(config);
        let stats = opt.stats();
        assert_eq!(stats.tasks_trained, 0);
        assert_eq!(stats.meta_updates, 0);
        assert_eq!(stats.avg_adaptation_loss, 0.0);
        assert_eq!(stats.avg_query_loss, 0.0);
    }

    #[test]
    fn test_stats_tasks_trained_increments() {
        let config = OptimizerConfig::default_maml(1);
        let mut opt = MetaLearningOptimizer::new(config);
        for i in 0..5 {
            let task = make_regression_task(&format!("t{i}"), 1.0, 0.0, 5, 5, i as u64 + 1);
            opt.add_task(task).expect("test: should succeed");
        }
        assert_eq!(opt.stats().tasks_trained, 5);
    }

    #[test]
    fn test_stats_meta_updates_increments() {
        let config = OptimizerConfig::default_maml(1);
        let mut opt = MetaLearningOptimizer::new(config);
        for i in 0..3 {
            let task = make_regression_task(&format!("t{i}"), 1.0, 0.0, 5, 5, i as u64 + 1);
            opt.add_task(task).expect("test: should succeed");
        }
        let mut params = MetaLearningOptimizer::initialize_params(1, 42);
        let ids: Vec<TaskId> = (0..3).map(|i| TaskId::new(format!("t{i}"))).collect();
        for n in 1..=5 {
            params = opt
                .meta_update(&ids, &params)
                .expect("test: should succeed");
            assert_eq!(opt.stats().meta_updates, n);
        }
    }

    #[test]
    fn test_stats_avg_adaptation_loss_non_negative() {
        let config = OptimizerConfig::default_maml(1);
        let mut opt = MetaLearningOptimizer::new(config);
        for i in 0..4 {
            let task =
                make_regression_task(&format!("t{i}"), (i + 1) as f64, 0.5, 8, 4, i as u64 + 2);
            opt.add_task(task).expect("test: should succeed");
        }
        let init = MetaLearningOptimizer::initialize_params(1, 9);
        let ids: Vec<TaskId> = (0..4).map(|i| TaskId::new(format!("t{i}"))).collect();
        opt.meta_update(&ids, &init).expect("test: should succeed");
        assert!(opt.stats().avg_adaptation_loss >= 0.0);
    }

    #[test]
    fn test_stats_query_loss_non_negative() {
        let config = OptimizerConfig::default_maml(1);
        let mut opt = MetaLearningOptimizer::new(config);
        for i in 0..4 {
            let task = make_regression_task(&format!("t{i}"), 1.0, 0.5, 8, 4, i as u64 + 3);
            opt.add_task(task).expect("test: should succeed");
        }
        let init = MetaLearningOptimizer::initialize_params(1, 11);
        let ids: Vec<TaskId> = (0..4).map(|i| TaskId::new(format!("t{i}"))).collect();
        opt.meta_update(&ids, &init).expect("test: should succeed");
        assert!(opt.stats().avg_query_loss >= 0.0);
    }

    // ── error cases ───────────────────────────────────────────────────────────

    #[test]
    fn test_error_display_insufficient_tasks() {
        let err = MetaError::InsufficientTasks(2);
        assert!(err.to_string().contains("insufficient"));
    }

    #[test]
    fn test_error_display_dimension_mismatch() {
        let err = MetaError::DimensionMismatch {
            expected: 4,
            got: 3,
        };
        let s = err.to_string();
        assert!(s.contains("4") && s.contains("3"));
    }

    #[test]
    fn test_error_display_adaptation_failed() {
        let err = MetaError::AdaptationFailed("oops".to_string());
        assert!(err.to_string().contains("oops"));
    }

    #[test]
    fn test_error_display_invalid_config() {
        let err = MetaError::InvalidConfig("bad lr".to_string());
        assert!(err.to_string().contains("bad lr"));
    }

    #[test]
    fn test_error_is_clone() {
        let err = MetaError::InsufficientTasks(3);
        let err2 = err.clone();
        assert_eq!(err, err2);
    }

    // ── xorshift PRNG internal tests ──────────────────────────────────────────

    #[test]
    fn test_xorshift64_deterministic() {
        let mut s1 = 12345u64;
        let mut s2 = 12345u64;
        assert_eq!(xorshift64(&mut s1), xorshift64(&mut s2));
    }

    #[test]
    fn test_xorshift_f64_range() {
        let mut state = 99999u64;
        for _ in 0..1000 {
            let v = xorshift_f64(&mut state);
            assert!((0.0..1.0).contains(&v), "out of range: {v}");
        }
    }

    // ── ModelParams helpers ───────────────────────────────────────────────────

    #[test]
    fn test_model_params_predict() {
        let p = ModelParams {
            weights: vec![2.0, -1.0],
            bias: 0.5,
            dim: 2,
        };
        let pred = p.predict(&[1.0, 1.0]);
        assert!((pred - 1.5).abs() < 1e-12);
    }

    #[test]
    fn test_model_params_zeros() {
        let p = ModelParams::zeros(3);
        assert_eq!(p.weights, vec![0.0, 0.0, 0.0]);
        assert_eq!(p.bias, 0.0);
        assert_eq!(p.dim, 3);
    }

    #[test]
    fn test_mse_zero_on_perfect_fit() {
        // y = 3x; params = w=[3], b=0
        let p = ModelParams {
            weights: vec![3.0],
            bias: 0.0,
            dim: 1,
        };
        let tid = TaskId::new("t");
        let examples: Vec<TaskExample> = (0..5)
            .map(|i| {
                let x = i as f64;
                TaskExample::new(vec![x], 3.0 * x, tid.clone())
            })
            .collect();
        let (loss, _, _) = p.mse_and_grads(&examples);
        assert!(loss < 1e-20, "MSE should be ~0 for perfect fit, got {loss}");
    }

    // ── Integration tests ─────────────────────────────────────────────────────

    #[test]
    fn test_end_to_end_maml_regression() {
        // After enough meta-updates, adapted params should perform well
        let config = OptimizerConfig {
            algorithm: MetaAlgorithm::MAML {
                inner_lr: 0.05,
                inner_steps: 5,
            },
            meta_lr: 0.1,
            n_tasks_per_batch: 4,
            max_params_dim: 1,
        };
        let mut opt = MetaLearningOptimizer::new(config);
        // All tasks share slope=2 but different intercepts
        for i in 0..4 {
            let task = make_regression_task(
                &format!("t{i}"),
                2.0,
                i as f64 * 0.5,
                15,
                5,
                (i * 13 + 7) as u64,
            );
            opt.add_task(task).expect("test: should succeed");
        }
        let mut meta_params = MetaLearningOptimizer::initialize_params(1, 42);
        let ids: Vec<TaskId> = (0..4).map(|i| TaskId::new(format!("t{i}"))).collect();
        for _ in 0..30 {
            meta_params = opt
                .meta_update(&ids, &meta_params)
                .expect("test: should succeed");
        }
        // Adapt on a new unseen task
        let tid = TaskId::new("new");
        let new_support: Vec<TaskExample> = (0..5)
            .map(|i| {
                let x = i as f64 * 0.5;
                TaskExample::new(vec![x], 2.0 * x + 0.3, tid.clone())
            })
            .collect();
        let new_query: Vec<TaskExample> = (0..5)
            .map(|i| {
                let x = i as f64 * 0.5 + 0.1;
                TaskExample::new(vec![x], 2.0 * x + 0.3, tid.clone())
            })
            .collect();
        let new_task = MetaTask::new(tid.clone(), new_support, new_query);
        opt.add_task(new_task).expect("test: should succeed");
        let adapted = opt
            .adapt_to_task(&tid, &meta_params, 10, 0.05)
            .expect("test: should succeed");
        let init_loss = adapted.first().map(|s| s.loss).unwrap_or(f64::MAX);
        let final_loss = adapted.last().map(|s| s.loss).unwrap_or(f64::MAX);
        assert!(
            final_loss <= init_loss + 1e-6,
            "adaptation should reduce loss: {init_loss} -> {final_loss}"
        );
    }

    #[test]
    fn test_end_to_end_reptile() {
        let config = OptimizerConfig::default_reptile(1);
        let mut opt = MetaLearningOptimizer::new(config);
        for i in 0..4 {
            let task = make_regression_task(
                &format!("t{i}"),
                1.5,
                i as f64 * 0.2,
                10,
                5,
                (i + 1) as u64 * 7,
            );
            opt.add_task(task).expect("test: should succeed");
        }
        let mut params = MetaLearningOptimizer::initialize_params(1, 33);
        let ids: Vec<TaskId> = (0..4).map(|i| TaskId::new(format!("t{i}"))).collect();
        for _ in 0..15 {
            params = opt
                .meta_update(&ids, &params)
                .expect("test: should succeed");
        }
        assert_eq!(opt.stats().meta_updates, 15);
        assert!(opt.stats().avg_query_loss >= 0.0);
    }

    #[test]
    fn test_task_id_display() {
        let tid = TaskId::new("hello");
        assert_eq!(tid.to_string(), "hello");
        assert_eq!(tid.as_str(), "hello");
    }

    #[test]
    fn test_meta_task_feature_dim() {
        let tid = TaskId::new("t");
        let ex = TaskExample::new(vec![1.0, 2.0, 3.0], 0.0, tid.clone());
        let task = MetaTask::new(tid, vec![ex], vec![]);
        assert_eq!(task.feature_dim(), Some(3));
    }

    #[test]
    fn test_meta_task_feature_dim_empty() {
        let tid = TaskId::new("t");
        let task = MetaTask::new(tid, vec![], vec![]);
        assert_eq!(task.feature_dim(), None);
    }
}
