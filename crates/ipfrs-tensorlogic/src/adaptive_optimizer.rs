//! AdaptiveOptimizer — Adam, AdaGrad, RMSProp, and AdamW optimizers for
//! distributed gradient descent.
//!
//! # Overview
//!
//! This module implements four widely-used adaptive gradient optimizers:
//!
//! * **Adam** — first- and second-moment estimates with bias correction.
//! * **AdaGrad** — cumulative squared-gradient denominator, good for sparse gradients.
//! * **RMSProp** — exponentially-decaying squared-gradient estimate with optional momentum.
//! * **AdamW** — Adam with decoupled weight-decay regularisation (Loshchilov & Hutter 2019).
//!
//! All optimizers share a common [`AdaptiveOptimizer`] driver that
//! manages per-group [`OptimizerState`] lazily and exposes helpers for
//! gradient clipping, norm computation, and statistics.

use std::collections::HashMap;
use thiserror::Error;

// ─────────────────────────────── errors ─────────────────────────────────────

/// Errors produced by the adaptive optimizer.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum OptimizerError {
    /// Parameter tensor and gradient tensor have incompatible sizes.
    #[error("dimension mismatch in group '{name}': params len={params}, grad len={grad}")]
    DimensionMismatch {
        /// Name of the parameter group.
        name: String,
        /// Length of the parameter vector.
        params: usize,
        /// Length of the gradient vector.
        grad: usize,
    },

    /// A parameter group is empty (zero parameters).
    #[error("parameter group '{0}' is empty")]
    EmptyGroup(String),
}

// ────────────────────────────── algorithm ───────────────────────────────────

/// Choice of adaptive gradient algorithm and its hyper-parameters.
#[derive(Debug, Clone, PartialEq)]
pub enum OptimizerAlgorithm {
    /// Adam optimiser (Kingma & Ba, 2015).
    Adam {
        /// Learning rate. Default 0.001.
        lr: f64,
        /// Exponential decay for first-moment estimates. Default 0.9.
        beta1: f64,
        /// Exponential decay for second-moment estimates. Default 0.999.
        beta2: f64,
        /// Numerical stability constant. Default 1e-8.
        epsilon: f64,
    },

    /// Adaptive gradient descent (Duchi et al., 2011).
    AdaGrad {
        /// Learning rate. Default 0.01.
        lr: f64,
        /// Numerical stability constant. Default 1e-8.
        epsilon: f64,
    },

    /// Root-mean-square propagation (Hinton, 2012).
    RmsProp {
        /// Learning rate. Default 0.01.
        lr: f64,
        /// Smoothing constant (decay for squared-gradient average). Default 0.99.
        alpha: f64,
        /// Numerical stability constant. Default 1e-8.
        epsilon: f64,
        /// Momentum coefficient. Default 0.0 (no momentum).
        momentum: f64,
    },

    /// Adam with decoupled weight decay (Loshchilov & Hutter, 2019).
    AdamW {
        /// Learning rate. Default 0.001.
        lr: f64,
        /// Exponential decay for first-moment estimates. Default 0.9.
        beta1: f64,
        /// Exponential decay for second-moment estimates. Default 0.999.
        beta2: f64,
        /// Numerical stability constant. Default 1e-8.
        epsilon: f64,
        /// Decoupled weight-decay coefficient. Default 0.01.
        weight_decay: f64,
    },
}

impl OptimizerAlgorithm {
    /// Construct an [`Adam`][Self::Adam] instance with default hyper-parameters.
    #[must_use]
    pub fn adam_default() -> Self {
        Self::Adam {
            lr: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        }
    }

    /// Construct an [`AdaGrad`][Self::AdaGrad] instance with default hyper-parameters.
    #[must_use]
    pub fn adagrad_default() -> Self {
        Self::AdaGrad {
            lr: 0.01,
            epsilon: 1e-8,
        }
    }

    /// Construct an [`RmsProp`][Self::RmsProp] instance with default hyper-parameters.
    #[must_use]
    pub fn rmsprop_default() -> Self {
        Self::RmsProp {
            lr: 0.01,
            alpha: 0.99,
            epsilon: 1e-8,
            momentum: 0.0,
        }
    }

    /// Construct an [`AdamW`][Self::AdamW] instance with default hyper-parameters.
    #[must_use]
    pub fn adamw_default() -> Self {
        Self::AdamW {
            lr: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.01,
        }
    }
}

// ──────────────────────────── parameter group ───────────────────────────────

/// A named group of parameters together with their current gradients.
///
/// Both `params` and `grad` must have equal length when passed to
/// [`AdaptiveOptimizer::step_group`].
#[derive(Debug, Clone)]
pub struct ParameterGroup {
    /// Unique identifier used as the key in the optimizer state map.
    pub name: String,
    /// Current parameter values (updated in-place by the optimizer step).
    pub params: Vec<f64>,
    /// Gradient of the loss with respect to each parameter.
    pub grad: Vec<f64>,
}

impl ParameterGroup {
    /// Create a new parameter group with all-zero gradients.
    #[must_use]
    pub fn new(name: impl Into<String>, params: Vec<f64>) -> Self {
        let n = params.len();
        Self {
            name: name.into(),
            params,
            grad: vec![0.0; n],
        }
    }

    /// Create a parameter group with explicit gradients.
    #[must_use]
    pub fn with_grad(name: impl Into<String>, params: Vec<f64>, grad: Vec<f64>) -> Self {
        Self {
            name: name.into(),
            params,
            grad,
        }
    }
}

// ────────────────────────────── optimizer state ─────────────────────────────

/// Per-parameter-group optimizer state (first moment, second moment, step counter).
///
/// This is the *internal* accumulator state maintained by [`AdaptiveOptimizer`].
/// It corresponds to the PyTorch `state_dict` entries for a parameter group.
#[derive(Debug, Clone)]
pub struct OptimizerState {
    /// First-moment (mean) vector; length equals the number of parameters.
    pub m: Vec<f64>,
    /// Second-moment (uncentered variance) vector; same length as `m`.
    pub v: Vec<f64>,
    /// Number of optimizer steps taken for this group (1-indexed when used).
    pub step: u64,
}

impl OptimizerState {
    /// Construct a zeroed state for `n` parameters.
    #[must_use]
    pub fn zeros(n: usize) -> Self {
        Self {
            m: vec![0.0; n],
            v: vec![0.0; n],
            step: 0,
        }
    }

    /// Reset state back to zeros without reallocating.
    pub fn reset(&mut self) {
        self.m.iter_mut().for_each(|x| *x = 0.0);
        self.v.iter_mut().for_each(|x| *x = 0.0);
        self.step = 0;
    }
}

// ───────────────────────── statistics snapshot ──────────────────────────────

/// A lightweight statistics snapshot returned by [`AdaptiveOptimizer::stats`].
#[derive(Debug, Clone, PartialEq)]
pub struct OptimizerStats {
    /// Total number of global optimizer steps executed.
    pub total_steps: u64,
    /// Number of parameter groups currently tracked.
    pub parameter_groups: usize,
    /// Total number of scalar parameters across all groups.
    pub total_parameters: usize,
    /// L2 norm of the gradient vector measured at the most recent step.
    pub last_grad_norm: f64,
}

// ─────────────────────────── main optimizer ─────────────────────────────────

/// Adaptive gradient optimizer supporting Adam, AdaGrad, RMSProp, and AdamW.
///
/// # Example
///
/// ```
/// use ipfrs_tensorlogic::adaptive_optimizer::{
///     AdaptiveOptimizer, OptimizerAlgorithm, ParameterGroup,
/// };
///
/// let mut opt = AdaptiveOptimizer::new(OptimizerAlgorithm::adam_default());
/// let mut groups = vec![
///     ParameterGroup::with_grad("w", vec![0.5, -0.3], vec![0.1, -0.2]),
/// ];
/// let norm = opt.step(&mut groups).expect("example: should succeed in docs");
/// assert!(norm > 0.0);
/// ```
#[derive(Debug, Clone)]
pub struct AdaptiveOptimizer {
    /// The algorithm (and its hyper-parameters) used for all steps.
    pub algorithm: OptimizerAlgorithm,
    /// Per-group optimizer states; keyed by [`ParameterGroup::name`].
    pub states: HashMap<String, OptimizerState>,
    /// Global step counter (incremented once per call to [`step`][Self::step]).
    pub global_step: u64,
    /// Cached gradient norm from the last [`step`][Self::step] call.
    last_grad_norm: f64,
}

impl AdaptiveOptimizer {
    /// Create a new optimizer wrapping the given algorithm.
    #[must_use]
    pub fn new(algorithm: OptimizerAlgorithm) -> Self {
        Self {
            algorithm,
            states: HashMap::new(),
            global_step: 0,
            last_grad_norm: 0.0,
        }
    }

    // ─── public API ──────────────────────────────────────────────────────────

    /// Perform one optimizer step across all `groups`.
    ///
    /// Returns the global L2 gradient norm computed *before* any update.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::DimensionMismatch`] if `params.len() != grad.len()`
    /// for any group, or [`OptimizerError::EmptyGroup`] if a group has zero parameters.
    pub fn step(&mut self, groups: &mut [ParameterGroup]) -> Result<f64, OptimizerError> {
        // Validate all groups first — fail fast before mutating any state.
        for g in groups.iter() {
            Self::validate_group(g)?;
        }

        // Compute global gradient norm before the update.
        let norm = Self::global_grad_norm(groups);
        self.last_grad_norm = norm;
        self.global_step += 1;

        // Update each group.
        for g in groups.iter_mut() {
            self.step_group(g)?;
        }

        Ok(norm)
    }

    /// Perform one optimizer step for a *single* parameter group.
    ///
    /// This is the low-level primitive used by [`step`][Self::step].
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::DimensionMismatch`] or [`OptimizerError::EmptyGroup`].
    pub fn step_group(&mut self, group: &mut ParameterGroup) -> Result<(), OptimizerError> {
        Self::validate_group(group)?;
        let n = group.params.len();
        let key = group.name.clone();

        // Lazily initialise state.
        let state = self
            .states
            .entry(key)
            .or_insert_with(|| OptimizerState::zeros(n));

        // Ensure state vector lengths match (handles groups that grew).
        if state.m.len() != n {
            *state = OptimizerState::zeros(n);
        }

        match &self.algorithm.clone() {
            OptimizerAlgorithm::Adam {
                lr,
                beta1,
                beta2,
                epsilon,
            } => Self::apply_adam(group, state, *lr, *beta1, *beta2, *epsilon),
            OptimizerAlgorithm::AdaGrad { lr, epsilon } => {
                Self::apply_adagrad(group, state, *lr, *epsilon);
            }
            OptimizerAlgorithm::RmsProp {
                lr,
                alpha,
                epsilon,
                momentum,
            } => Self::apply_rmsprop(group, state, *lr, *alpha, *epsilon, *momentum),
            OptimizerAlgorithm::AdamW {
                lr,
                beta1,
                beta2,
                epsilon,
                weight_decay,
            } => Self::apply_adamw(group, state, *lr, *beta1, *beta2, *epsilon, *weight_decay),
        }

        Ok(())
    }

    /// Zero-out all gradients in `groups`.
    pub fn zero_grad(groups: &mut [ParameterGroup]) {
        for g in groups.iter_mut() {
            g.grad.iter_mut().for_each(|x| *x = 0.0);
        }
    }

    /// Compute the global L2 norm of all gradients across `groups`.
    #[must_use]
    pub fn global_grad_norm(groups: &[ParameterGroup]) -> f64 {
        let sum_sq: f64 = groups
            .iter()
            .flat_map(|g| g.grad.iter())
            .map(|&x| x * x)
            .sum();
        sum_sq.sqrt()
    }

    /// Scale all gradients so the global norm does not exceed `max_norm`.
    ///
    /// If the current global norm is ≤ `max_norm` or is not finite (e.g. NaN/inf),
    /// gradients are left unchanged.
    pub fn clip_grad_norm(groups: &mut [ParameterGroup], max_norm: f64) {
        let norm = Self::global_grad_norm(groups);
        if norm > max_norm && norm.is_finite() && max_norm > 0.0 {
            let scale = max_norm / norm;
            for g in groups.iter_mut() {
                g.grad.iter_mut().for_each(|x| *x *= scale);
            }
        }
    }

    /// Clear the optimizer state for a specific group (by name).
    pub fn reset_state(&mut self, group_name: &str) {
        self.states.remove(group_name);
    }

    /// Clear all optimizer states and reset the global step counter.
    pub fn reset_all(&mut self) {
        self.states.clear();
        self.global_step = 0;
        self.last_grad_norm = 0.0;
    }

    /// Return a statistics snapshot.
    #[must_use]
    pub fn stats(&self, groups: &[ParameterGroup]) -> OptimizerStats {
        let total_parameters = groups.iter().map(|g| g.params.len()).sum();
        OptimizerStats {
            total_steps: self.global_step,
            parameter_groups: groups.len(),
            total_parameters,
            last_grad_norm: self.last_grad_norm,
        }
    }

    // ─── private update kernels ──────────────────────────────────────────────

    /// Validate that a group is non-empty and that `params` and `grad` agree in length.
    fn validate_group(g: &ParameterGroup) -> Result<(), OptimizerError> {
        if g.params.is_empty() {
            return Err(OptimizerError::EmptyGroup(g.name.clone()));
        }
        if g.params.len() != g.grad.len() {
            return Err(OptimizerError::DimensionMismatch {
                name: g.name.clone(),
                params: g.params.len(),
                grad: g.grad.len(),
            });
        }
        Ok(())
    }

    /// Adam update rule.
    ///
    /// ```text
    /// state.step += 1
    /// for i in 0..n:
    ///     m[i] = β₁·m[i] + (1−β₁)·g
    ///     v[i] = β₂·v[i] + (1−β₂)·g²
    ///     m̂ = m[i] / (1 − β₁^t)
    ///     v̂ = v[i] / (1 − β₂^t)
    ///     θ[i] -= lr · m̂ / (√v̂ + ε)
    /// ```
    fn apply_adam(
        group: &mut ParameterGroup,
        state: &mut OptimizerState,
        lr: f64,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
    ) {
        state.step += 1;
        let t = state.step as f64;
        let bc1 = 1.0 - beta1.powf(t);
        let bc2 = 1.0 - beta2.powf(t);

        for i in 0..group.params.len() {
            let g = group.grad[i];
            state.m[i] = beta1 * state.m[i] + (1.0 - beta1) * g;
            state.v[i] = beta2 * state.v[i] + (1.0 - beta2) * g * g;
            let m_hat = state.m[i] / bc1;
            let v_hat = state.v[i] / bc2;
            group.params[i] -= lr * m_hat / (v_hat.sqrt() + epsilon);
        }
    }

    /// AdaGrad update rule.
    ///
    /// ```text
    /// v[i] += g²
    /// θ[i] -= lr · g / (√v[i] + ε)
    /// ```
    /// The first-moment slot `m` is not used.
    fn apply_adagrad(
        group: &mut ParameterGroup,
        state: &mut OptimizerState,
        lr: f64,
        epsilon: f64,
    ) {
        state.step += 1;
        for i in 0..group.params.len() {
            let g = group.grad[i];
            state.v[i] += g * g;
            group.params[i] -= lr * g / (state.v[i].sqrt() + epsilon);
        }
    }

    /// RMSProp update rule (with optional momentum).
    ///
    /// ```text
    /// v[i] = α·v[i] + (1−α)·g²
    /// m[i] = momentum·m[i] + lr·g / √(v[i] + ε)
    /// θ[i] -= m[i]
    /// ```
    fn apply_rmsprop(
        group: &mut ParameterGroup,
        state: &mut OptimizerState,
        lr: f64,
        alpha: f64,
        epsilon: f64,
        momentum: f64,
    ) {
        state.step += 1;
        for i in 0..group.params.len() {
            let g = group.grad[i];
            state.v[i] = alpha * state.v[i] + (1.0 - alpha) * g * g;
            let delta = lr * g / (state.v[i] + epsilon).sqrt();
            state.m[i] = momentum * state.m[i] + delta;
            group.params[i] -= state.m[i];
        }
    }

    /// AdamW update rule (Adam + decoupled weight decay).
    ///
    /// ```text
    /// θ[i] -= lr·wd·θ[i]     ← weight decay applied first
    /// then Adam update
    /// ```
    fn apply_adamw(
        group: &mut ParameterGroup,
        state: &mut OptimizerState,
        lr: f64,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
        weight_decay: f64,
    ) {
        state.step += 1;
        let t = state.step as f64;
        let bc1 = 1.0 - beta1.powf(t);
        let bc2 = 1.0 - beta2.powf(t);

        for i in 0..group.params.len() {
            // Decoupled weight decay.
            group.params[i] -= lr * weight_decay * group.params[i];

            let g = group.grad[i];
            state.m[i] = beta1 * state.m[i] + (1.0 - beta1) * g;
            state.v[i] = beta2 * state.v[i] + (1.0 - beta2) * g * g;
            let m_hat = state.m[i] / bc1;
            let v_hat = state.v[i] / bc2;
            group.params[i] -= lr * m_hat / (v_hat.sqrt() + epsilon);
        }
    }
}

// ═══════════════════════════════ tests ═══════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::{
        AdaptiveOptimizer, OptimizerAlgorithm, OptimizerError, OptimizerState, ParameterGroup,
    };

    // ── helpers ──────────────────────────────────────────────────────────────

    fn adam_opt() -> AdaptiveOptimizer {
        AdaptiveOptimizer::new(OptimizerAlgorithm::adam_default())
    }

    fn adagrad_opt() -> AdaptiveOptimizer {
        AdaptiveOptimizer::new(OptimizerAlgorithm::adagrad_default())
    }

    fn rmsprop_opt() -> AdaptiveOptimizer {
        AdaptiveOptimizer::new(OptimizerAlgorithm::rmsprop_default())
    }

    fn adamw_opt() -> AdaptiveOptimizer {
        AdaptiveOptimizer::new(OptimizerAlgorithm::adamw_default())
    }

    fn simple_group(name: &str, p: f64, g: f64) -> ParameterGroup {
        ParameterGroup::with_grad(name, vec![p], vec![g])
    }

    // ── construction ─────────────────────────────────────────────────────────

    #[test]
    fn test_new_optimizer_initial_state() {
        let opt = adam_opt();
        assert_eq!(opt.global_step, 0);
        assert!(opt.states.is_empty());
    }

    #[test]
    fn test_parameter_group_new_zeros_grad() {
        let g = ParameterGroup::new("layer", vec![1.0, 2.0, 3.0]);
        assert_eq!(g.grad, vec![0.0, 0.0, 0.0]);
        assert_eq!(g.params.len(), 3);
    }

    #[test]
    fn test_parameter_group_with_grad() {
        let g = ParameterGroup::with_grad("w", vec![1.0], vec![0.5]);
        assert_eq!(g.params[0], 1.0);
        assert_eq!(g.grad[0], 0.5);
    }

    #[test]
    fn test_optimizer_state_zeros() {
        let s = OptimizerState::zeros(4);
        assert_eq!(s.m, vec![0.0; 4]);
        assert_eq!(s.v, vec![0.0; 4]);
        assert_eq!(s.step, 0);
    }

    #[test]
    fn test_optimizer_state_reset() {
        let mut s = OptimizerState {
            m: vec![1.0, 2.0],
            v: vec![3.0, 4.0],
            step: 10,
        };
        s.reset();
        assert_eq!(s.m, vec![0.0, 0.0]);
        assert_eq!(s.v, vec![0.0, 0.0]);
        assert_eq!(s.step, 0);
    }

    // ── validation errors ────────────────────────────────────────────────────

    #[test]
    fn test_step_dimension_mismatch_error() {
        let mut opt = adam_opt();
        let mut groups = vec![ParameterGroup {
            name: "bad".to_string(),
            params: vec![1.0, 2.0],
            grad: vec![0.1],
        }];
        let err = opt.step(&mut groups).unwrap_err();
        assert!(matches!(err, OptimizerError::DimensionMismatch { .. }));
    }

    #[test]
    fn test_step_empty_group_error() {
        let mut opt = adam_opt();
        let mut groups = vec![ParameterGroup {
            name: "empty".to_string(),
            params: vec![],
            grad: vec![],
        }];
        let err = opt.step(&mut groups).unwrap_err();
        assert!(matches!(err, OptimizerError::EmptyGroup(_)));
    }

    #[test]
    fn test_step_group_dimension_mismatch() {
        let mut opt = adam_opt();
        let mut g = ParameterGroup {
            name: "x".to_string(),
            params: vec![1.0],
            grad: vec![0.1, 0.2],
        };
        assert!(opt.step_group(&mut g).is_err());
    }

    // ── Adam ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_adam_step_reduces_param_toward_zero() {
        let mut opt = adam_opt();
        let mut groups = vec![simple_group("w", 1.0, 1.0)];
        opt.step(&mut groups).expect("test: should succeed");
        // With positive gradient the parameter should decrease.
        assert!(groups[0].params[0] < 1.0);
    }

    #[test]
    fn test_adam_global_step_increments() {
        let mut opt = adam_opt();
        let mut groups = vec![simple_group("w", 1.0, 0.1)];
        opt.step(&mut groups).expect("test: should succeed");
        opt.step(&mut groups).expect("test: should succeed");
        assert_eq!(opt.global_step, 2);
    }

    #[test]
    fn test_adam_state_step_increments_per_group() {
        let mut opt = adam_opt();
        let mut groups = vec![simple_group("a", 0.5, 0.2)];
        opt.step(&mut groups).expect("test: should succeed");
        opt.step(&mut groups).expect("test: should succeed");
        assert_eq!(opt.states["a"].step, 2);
    }

    #[test]
    fn test_adam_moment_vectors_are_nonzero_after_step() {
        let mut opt = adam_opt();
        let mut groups = vec![simple_group("m", 0.0, 1.0)];
        opt.step(&mut groups).expect("test: should succeed");
        let s = &opt.states["m"];
        assert_ne!(s.m[0], 0.0);
        assert_ne!(s.v[0], 0.0);
    }

    #[test]
    fn test_adam_multiple_params() {
        let mut opt = adam_opt();
        let mut groups = vec![ParameterGroup::with_grad(
            "layer",
            vec![1.0, -1.0, 0.0],
            vec![0.5, -0.5, 1.0],
        )];
        let norm = opt.step(&mut groups).expect("test: should succeed");
        assert!(norm > 0.0);
        assert_eq!(groups[0].params.len(), 3);
    }

    #[test]
    fn test_adam_zero_gradient_leaves_param_almost_unchanged() {
        let mut opt = AdaptiveOptimizer::new(OptimizerAlgorithm::Adam {
            lr: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        });
        let initial = 5.0_f64;
        let mut groups = vec![simple_group("p", initial, 0.0)];
        opt.step(&mut groups).expect("test: should succeed");
        // Zero gradient → m and v remain zero → m_hat and v_hat → 0/bc → 0.
        // Update: 0 / (sqrt(0) + eps) = 0 → param unchanged.
        assert!((groups[0].params[0] - initial).abs() < 1e-12);
    }

    // ── AdaGrad ───────────────────────────────────────────────────────────────

    #[test]
    fn test_adagrad_step_lowers_param_for_positive_grad() {
        let mut opt = adagrad_opt();
        let mut groups = vec![simple_group("w", 2.0, 1.0)];
        opt.step(&mut groups).expect("test: should succeed");
        assert!(groups[0].params[0] < 2.0);
    }

    #[test]
    fn test_adagrad_accumulates_squared_grad_in_v() {
        let mut opt = adagrad_opt();
        let mut groups = vec![simple_group("a", 0.0, 3.0)];
        opt.step(&mut groups).expect("test: should succeed");
        // v[0] should be 3^2 = 9.
        assert!((opt.states["a"].v[0] - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_adagrad_large_gradient_decays_lr() {
        // After many steps the effective lr → 0.
        let mut opt = adagrad_opt();
        let mut groups = vec![simple_group("w", 1.0, 100.0)];
        let p_after_1 = {
            opt.step(&mut groups).expect("test: should succeed");
            groups[0].params[0]
        };
        // Reset param but keep state (simulating large cumulative gradient).
        groups[0].params[0] = 1.0;
        groups[0].grad[0] = 100.0;
        opt.step(&mut groups).expect("test: should succeed");
        let p_after_2 = groups[0].params[0];
        // The second step should move the parameter less than the first
        // because v is larger.
        let delta1 = (1.0 - p_after_1).abs();
        let delta2 = (1.0 - p_after_2).abs();
        assert!(delta2 < delta1);
    }

    // ── RMSProp ───────────────────────────────────────────────────────────────

    #[test]
    fn test_rmsprop_step_moves_param() {
        let mut opt = rmsprop_opt();
        let mut groups = vec![simple_group("w", 1.0, 1.0)];
        let before = groups[0].params[0];
        opt.step(&mut groups).expect("test: should succeed");
        assert_ne!(groups[0].params[0], before);
    }

    #[test]
    fn test_rmsprop_with_momentum() {
        let mut opt = AdaptiveOptimizer::new(OptimizerAlgorithm::RmsProp {
            lr: 0.01,
            alpha: 0.99,
            epsilon: 1e-8,
            momentum: 0.9,
        });
        let mut groups = vec![simple_group("w", 1.0, 1.0)];
        opt.step(&mut groups).expect("test: should succeed");
        // m should be non-zero because momentum is active.
        assert_ne!(opt.states["w"].m[0], 0.0);
    }

    #[test]
    fn test_rmsprop_v_decays_toward_zero_on_zero_grad() {
        let mut opt = AdaptiveOptimizer::new(OptimizerAlgorithm::RmsProp {
            lr: 0.01,
            alpha: 0.9,
            epsilon: 1e-8,
            momentum: 0.0,
        });
        // Warm up v with a non-zero gradient.
        let mut groups = vec![simple_group("w", 1.0, 1.0)];
        opt.step(&mut groups).expect("test: should succeed");
        let v_after_1 = opt.states["w"].v[0];

        // Now apply zero gradient — v should decay.
        groups[0].grad[0] = 0.0;
        opt.step(&mut groups).expect("test: should succeed");
        let v_after_2 = opt.states["w"].v[0];
        assert!(v_after_2 < v_after_1);
    }

    // ── AdamW ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_adamw_applies_weight_decay() {
        // Compare AdamW with wd > 0 vs Adam (wd=0) on same initial state.
        let mut opt_adamw = adamw_opt(); // wd = 0.01
        let mut opt_adam = AdaptiveOptimizer::new(OptimizerAlgorithm::Adam {
            lr: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        });

        let init_param = 2.0_f64;
        let grad_val = 0.1_f64;

        let mut groups_wd = vec![simple_group("p", init_param, grad_val)];
        let mut groups_no_wd = vec![simple_group("p", init_param, grad_val)];

        opt_adamw
            .step(&mut groups_wd)
            .expect("test: should succeed");
        opt_adam
            .step(&mut groups_no_wd)
            .expect("test: should succeed");

        // Weight decay shrinks the parameter more.
        assert!(groups_wd[0].params[0] < groups_no_wd[0].params[0]);
    }

    #[test]
    fn test_adamw_zero_weight_decay_equals_adam() {
        let mut opt_adamw = AdaptiveOptimizer::new(OptimizerAlgorithm::AdamW {
            lr: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.0,
        });
        let mut opt_adam = AdaptiveOptimizer::new(OptimizerAlgorithm::Adam {
            lr: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        });

        let mut g1 = vec![simple_group("w", 1.0, 0.5)];
        let mut g2 = vec![simple_group("w", 1.0, 0.5)];

        opt_adamw.step(&mut g1).expect("test: should succeed");
        opt_adam.step(&mut g2).expect("test: should succeed");

        let diff = (g1[0].params[0] - g2[0].params[0]).abs();
        assert!(diff < 1e-14, "expected Adam≈AdamW(wd=0), diff={diff}");
    }

    // ── gradient utilities ────────────────────────────────────────────────────

    #[test]
    fn test_global_grad_norm_single_value() {
        let groups = vec![simple_group("w", 0.0, 3.0)];
        let norm = AdaptiveOptimizer::global_grad_norm(&groups);
        assert!((norm - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_global_grad_norm_two_groups() {
        let groups = vec![simple_group("a", 0.0, 3.0), simple_group("b", 0.0, 4.0)];
        let norm = AdaptiveOptimizer::global_grad_norm(&groups);
        // sqrt(9 + 16) = 5
        assert!((norm - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_global_grad_norm_zero_gradients() {
        let groups = vec![ParameterGroup::new("w", vec![1.0, 2.0])];
        let norm = AdaptiveOptimizer::global_grad_norm(&groups);
        assert_eq!(norm, 0.0);
    }

    #[test]
    fn test_clip_grad_norm_scales_down() {
        let mut groups = vec![simple_group("a", 0.0, 3.0), simple_group("b", 0.0, 4.0)];
        AdaptiveOptimizer::clip_grad_norm(&mut groups, 1.0);
        let new_norm = AdaptiveOptimizer::global_grad_norm(&groups);
        assert!((new_norm - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_clip_grad_norm_no_op_when_below_max() {
        let mut groups = vec![simple_group("w", 0.0, 0.3)];
        AdaptiveOptimizer::clip_grad_norm(&mut groups, 5.0);
        assert!((groups[0].grad[0] - 0.3).abs() < 1e-14);
    }

    #[test]
    fn test_clip_grad_norm_preserves_direction() {
        let mut groups = vec![ParameterGroup::with_grad(
            "w",
            vec![0.0, 0.0],
            vec![3.0, 4.0],
        )];
        AdaptiveOptimizer::clip_grad_norm(&mut groups, 1.0);
        // Ratio should be preserved.
        let ratio = groups[0].grad[0] / groups[0].grad[1];
        assert!((ratio - 0.75).abs() < 1e-10, "ratio={ratio}");
    }

    #[test]
    fn test_zero_grad_clears_all() {
        let mut groups = vec![simple_group("a", 1.0, 2.0), simple_group("b", 3.0, 4.0)];
        AdaptiveOptimizer::zero_grad(&mut groups);
        for g in &groups {
            for &v in &g.grad {
                assert_eq!(v, 0.0);
            }
        }
    }

    // ── state management ──────────────────────────────────────────────────────

    #[test]
    fn test_reset_state_clears_single_group() {
        let mut opt = adam_opt();
        let mut groups = vec![simple_group("w", 1.0, 0.5)];
        opt.step(&mut groups).expect("test: should succeed");
        assert!(opt.states.contains_key("w"));
        opt.reset_state("w");
        assert!(!opt.states.contains_key("w"));
    }

    #[test]
    fn test_reset_all_clears_everything() {
        let mut opt = adam_opt();
        let mut groups = vec![simple_group("a", 1.0, 0.1), simple_group("b", 2.0, 0.2)];
        opt.step(&mut groups).expect("test: should succeed");
        opt.reset_all();
        assert!(opt.states.is_empty());
        assert_eq!(opt.global_step, 0);
    }

    #[test]
    fn test_reset_state_nonexistent_key_is_noop() {
        let mut opt = adam_opt();
        opt.reset_state("nonexistent"); // Should not panic.
        assert!(opt.states.is_empty());
    }

    // ── statistics ────────────────────────────────────────────────────────────

    #[test]
    fn test_stats_initial() {
        let opt = adam_opt();
        let groups = vec![
            ParameterGroup::new("a", vec![1.0, 2.0]),
            ParameterGroup::new("b", vec![3.0]),
        ];
        let s = opt.stats(&groups);
        assert_eq!(s.total_steps, 0);
        assert_eq!(s.parameter_groups, 2);
        assert_eq!(s.total_parameters, 3);
        assert_eq!(s.last_grad_norm, 0.0);
    }

    #[test]
    fn test_stats_after_step() {
        let mut opt = adam_opt();
        let mut groups = vec![ParameterGroup::with_grad(
            "w",
            vec![1.0, 2.0],
            vec![3.0, 4.0],
        )];
        opt.step(&mut groups).expect("test: should succeed");
        let s = opt.stats(&groups);
        assert_eq!(s.total_steps, 1);
        assert!((s.last_grad_norm - 5.0).abs() < 1e-10);
    }

    // ── step returns gradient norm ────────────────────────────────────────────

    #[test]
    fn test_step_returns_correct_grad_norm() {
        let mut opt = adam_opt();
        let mut groups = vec![ParameterGroup::with_grad(
            "w",
            vec![0.0, 0.0],
            vec![3.0, 4.0],
        )];
        let norm = opt.step(&mut groups).expect("test: should succeed");
        assert!((norm - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_step_returns_zero_norm_for_zero_grads() {
        let mut opt = adam_opt();
        let mut groups = vec![ParameterGroup::new("w", vec![1.0, 2.0])];
        let norm = opt.step(&mut groups).expect("test: should succeed");
        assert_eq!(norm, 0.0);
    }

    // ── multi-group steps ─────────────────────────────────────────────────────

    #[test]
    fn test_multiple_groups_each_have_independent_state() {
        let mut opt = adam_opt();
        let mut groups = vec![
            simple_group("layer1", 1.0, 0.1),
            simple_group("layer2", -1.0, -0.1),
        ];
        opt.step(&mut groups).expect("test: should succeed");
        // Both groups should have their own state.
        assert!(opt.states.contains_key("layer1"));
        assert!(opt.states.contains_key("layer2"));
    }

    #[test]
    fn test_step_group_individually_matches_bulk_step() {
        // Run both paths on the same initial conditions and verify they agree.
        let mut opt_bulk = adam_opt();
        let mut opt_single = adam_opt();

        let mut groups_bulk = vec![simple_group("w1", 1.0, 0.5), simple_group("w2", -0.5, -0.3)];
        let mut groups_single = groups_bulk.clone();

        opt_bulk
            .step(&mut groups_bulk)
            .expect("test: should succeed");
        for g in groups_single.iter_mut() {
            opt_single.step_group(g).expect("test: should succeed");
        }

        for (gb, gs) in groups_bulk.iter().zip(groups_single.iter()) {
            let diff = (gb.params[0] - gs.params[0]).abs();
            assert!(diff < 1e-14, "param mismatch for {}: {diff}", gb.name);
        }
    }

    // ── convergence smoke test ────────────────────────────────────────────────

    #[test]
    fn test_adam_converges_simple_quadratic() {
        // Minimise f(x) = x^2/2 ⟹ grad = x, optimum at x=0.
        // Use a higher learning rate and enough steps to converge reliably.
        let mut opt = AdaptiveOptimizer::new(OptimizerAlgorithm::Adam {
            lr: 0.1,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        });
        let mut groups = vec![ParameterGroup::new("x", vec![5.0])];
        for _ in 0..2000 {
            groups[0].grad[0] = groups[0].params[0]; // grad of x²/2
            opt.step(&mut groups).expect("test: should succeed");
        }
        assert!(
            groups[0].params[0].abs() < 0.01,
            "did not converge: x={}",
            groups[0].params[0]
        );
    }

    #[test]
    fn test_adagrad_converges_simple_quadratic() {
        // AdaGrad with a larger initial lr converges for a quadratic.
        let mut opt = AdaptiveOptimizer::new(OptimizerAlgorithm::AdaGrad {
            lr: 1.0,
            epsilon: 1e-8,
        });
        let mut groups = vec![ParameterGroup::new("x", vec![3.0])];
        for _ in 0..500 {
            groups[0].grad[0] = groups[0].params[0];
            opt.step(&mut groups).expect("test: should succeed");
        }
        assert!(
            groups[0].params[0].abs() < 0.1,
            "did not converge: x={}",
            groups[0].params[0]
        );
    }

    #[test]
    fn test_rmsprop_converges_simple_quadratic() {
        let mut opt = rmsprop_opt();
        let mut groups = vec![ParameterGroup::new("x", vec![3.0])];
        for _ in 0..3000 {
            groups[0].grad[0] = groups[0].params[0];
            opt.step(&mut groups).expect("test: should succeed");
        }
        assert!(
            groups[0].params[0].abs() < 0.1,
            "did not converge: x={}",
            groups[0].params[0]
        );
    }

    #[test]
    fn test_adamw_converges_simple_quadratic() {
        let mut opt = AdaptiveOptimizer::new(OptimizerAlgorithm::AdamW {
            lr: 0.01,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.001,
        });
        let mut groups = vec![ParameterGroup::new("x", vec![3.0])];
        for _ in 0..5000 {
            groups[0].grad[0] = groups[0].params[0];
            opt.step(&mut groups).expect("test: should succeed");
        }
        assert!(
            groups[0].params[0].abs() < 0.1,
            "did not converge: x={}",
            groups[0].params[0]
        );
    }

    // ── default constructors ──────────────────────────────────────────────────

    #[test]
    fn test_algorithm_default_constructors() {
        let adam = OptimizerAlgorithm::adam_default();
        assert!(matches!(adam, OptimizerAlgorithm::Adam { lr, .. } if (lr - 0.001).abs() < 1e-15));

        let adagrad = OptimizerAlgorithm::adagrad_default();
        assert!(
            matches!(adagrad, OptimizerAlgorithm::AdaGrad { lr, .. } if (lr - 0.01).abs() < 1e-15)
        );

        let rmsprop = OptimizerAlgorithm::rmsprop_default();
        assert!(
            matches!(rmsprop, OptimizerAlgorithm::RmsProp { lr, .. } if (lr - 0.01).abs() < 1e-15)
        );

        let adamw = OptimizerAlgorithm::adamw_default();
        assert!(
            matches!(adamw, OptimizerAlgorithm::AdamW { weight_decay, .. } if (weight_decay - 0.01).abs() < 1e-15)
        );
    }

    // ── lazy state initialisation ─────────────────────────────────────────────

    #[test]
    fn test_state_lazily_initialised_on_first_step() {
        let mut opt = adam_opt();
        assert!(opt.states.is_empty());
        let mut groups = vec![simple_group("w", 0.0, 1.0)];
        opt.step(&mut groups).expect("test: should succeed");
        assert!(opt.states.contains_key("w"));
    }

    // ── error message content ─────────────────────────────────────────────────

    #[test]
    fn test_dimension_mismatch_error_contains_name() {
        let mut opt = adam_opt();
        let mut groups = vec![ParameterGroup {
            name: "my_layer".to_string(),
            params: vec![1.0],
            grad: vec![0.1, 0.2],
        }];
        let err = opt.step(&mut groups).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("my_layer"), "error message: {msg}");
    }

    #[test]
    fn test_empty_group_error_contains_name() {
        let mut opt = adam_opt();
        let mut groups = vec![ParameterGroup {
            name: "empty_layer".to_string(),
            params: vec![],
            grad: vec![],
        }];
        let err = opt.step(&mut groups).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("empty_layer"), "error message: {msg}");
    }

    // ── clone / debug ─────────────────────────────────────────────────────────

    #[test]
    fn test_optimizer_clone_is_independent() {
        let mut opt = adam_opt();
        let mut groups = vec![simple_group("w", 1.0, 0.5)];
        opt.step(&mut groups).expect("test: should succeed");
        let mut opt2 = opt.clone();
        opt2.reset_all();
        // Original should be unaffected.
        assert_eq!(opt.global_step, 1);
    }

    #[test]
    fn test_optimizer_debug_does_not_panic() {
        let opt = adam_opt();
        let _ = format!("{opt:?}");
    }
}
