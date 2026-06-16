//! SGD Optimizer variants for tensor parameter optimization.
//!
//! Provides stochastic gradient descent with optional momentum, Nesterov
//! acceleration, weight decay, and dampening.  Designed for training loops
//! inside the TensorLogic subsystem.
//!
//! # Supported variants
//!
//! | Variant        | Update rule |
//! |----------------|-------------|
//! | **SGD**        | `p -= lr * (g + wd * p)` |
//! | **SGDMomentum**| `v = m*v + (1-d)*g; p -= lr*(v + wd*p)` |
//! | **SGDNesterov**| `v = m*v + g; p -= lr*(g + m*v + wd*p)` |
//!
//! # Example
//!
//! ```
//! use ipfrs_tensorlogic::sgd_optimizer::{SGDConfig, SGDOptimizer, OptimizerType};
//! use std::collections::HashMap;
//!
//! let config = SGDConfig::default();
//! let mut opt = SGDOptimizer::new(config);
//! opt.register_parameter("w", vec![1.0, 2.0, 3.0]);
//!
//! let mut grads = HashMap::new();
//! grads.insert("w".to_string(), vec![0.1, 0.2, 0.3]);
//! opt.step(&grads).expect("example: should succeed in docs");
//!
//! let w = opt.get_parameter("w").expect("example: should succeed in docs");
//! assert!(w[0] < 1.0); // parameter decreased
//! ```

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Which SGD variant to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizerType {
    /// Vanilla SGD (no momentum buffer).
    SGD,
    /// Classical momentum SGD.
    SGDMomentum,
    /// Nesterov accelerated gradient.
    SGDNesterov,
}

impl std::fmt::Display for OptimizerType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SGD => write!(f, "SGD"),
            Self::SGDMomentum => write!(f, "SGDMomentum"),
            Self::SGDNesterov => write!(f, "SGDNesterov"),
        }
    }
}

/// Configuration for an [`SGDOptimizer`].
#[derive(Debug, Clone)]
pub struct SGDConfig {
    /// Optimizer variant.
    pub optimizer_type: OptimizerType,
    /// Step size (default `0.01`).
    pub learning_rate: f64,
    /// Momentum coefficient (used by Momentum/Nesterov, default `0.9`).
    pub momentum: f64,
    /// L2 weight-decay coefficient (default `0.0`).
    pub weight_decay: f64,
    /// Dampening factor for momentum (default `0.0`).
    pub dampening: f64,
}

impl Default for SGDConfig {
    fn default() -> Self {
        Self {
            optimizer_type: OptimizerType::SGD,
            learning_rate: 0.01,
            momentum: 0.9,
            weight_decay: 0.0,
            dampening: 0.0,
        }
    }
}

/// Per-parameter mutable state tracked by the optimizer.
#[derive(Debug, Clone)]
pub struct ParameterState {
    /// Human-readable name.
    pub name: String,
    /// Current parameter values.
    pub values: Vec<f64>,
    /// Momentum buffer (velocity).
    pub velocity: Vec<f64>,
}

/// Summary statistics returned by [`SGDOptimizer::stats`].
#[derive(Debug, Clone)]
pub struct SGDOptimizerStats {
    /// Which variant is active.
    pub optimizer_type: OptimizerType,
    /// Current learning rate.
    pub learning_rate: f64,
    /// Total number of scalar parameters.
    pub parameter_count: usize,
    /// Number of optimiser steps executed so far.
    pub step_count: u64,
}

// ---------------------------------------------------------------------------
// Optimizer
// ---------------------------------------------------------------------------

/// Stochastic gradient descent optimizer with momentum, Nesterov, and weight
/// decay support.
#[derive(Debug, Clone)]
pub struct SGDOptimizer {
    config: SGDConfig,
    parameters: HashMap<String, ParameterState>,
    step_count: u64,
}

impl SGDOptimizer {
    /// Create a new optimizer with the given configuration.
    pub fn new(config: SGDConfig) -> Self {
        Self {
            config,
            parameters: HashMap::new(),
            step_count: 0,
        }
    }

    /// Register a named parameter vector.  The velocity buffer is initialised
    /// to zeros with the same length as `initial_values`.
    pub fn register_parameter(&mut self, name: &str, initial_values: Vec<f64>) {
        let len = initial_values.len();
        self.parameters.insert(
            name.to_string(),
            ParameterState {
                name: name.to_string(),
                values: initial_values,
                velocity: vec![0.0; len],
            },
        );
    }

    /// Execute one optimiser step.
    ///
    /// `gradients` must contain exactly the same keys as the registered
    /// parameters, and each gradient vector must match the corresponding
    /// parameter length.
    pub fn step(&mut self, gradients: &HashMap<String, Vec<f64>>) -> Result<(), String> {
        // Validate gradient keys match parameters.
        for key in gradients.keys() {
            if !self.parameters.contains_key(key) {
                return Err(format!(
                    "gradient key '{}' does not match any registered parameter",
                    key
                ));
            }
        }
        for key in self.parameters.keys() {
            if !gradients.contains_key(key) {
                return Err(format!(
                    "missing gradient for registered parameter '{}'",
                    key
                ));
            }
        }

        // Validate sizes.
        for (key, grad) in gradients {
            let param = self
                .parameters
                .get(key)
                .ok_or_else(|| format!("parameter '{}' not found", key))?;
            if grad.len() != param.values.len() {
                return Err(format!(
                    "gradient length {} for '{}' does not match parameter length {}",
                    grad.len(),
                    key,
                    param.values.len(),
                ));
            }
        }

        let lr = self.config.learning_rate;
        let wd = self.config.weight_decay;
        let mom = self.config.momentum;
        let damp = self.config.dampening;

        // Collect keys to avoid borrow issues.
        let keys: Vec<String> = self.parameters.keys().cloned().collect();

        for key in &keys {
            let grad = gradients
                .get(key)
                .ok_or_else(|| format!("missing gradient for '{}'", key))?;
            let state = self
                .parameters
                .get_mut(key)
                .ok_or_else(|| format!("parameter '{}' disappeared", key))?;

            match self.config.optimizer_type {
                OptimizerType::SGD => {
                    for (p, g) in state.values.iter_mut().zip(grad.iter()) {
                        let effective_grad = g + wd * *p;
                        *p -= lr * effective_grad;
                    }
                }
                OptimizerType::SGDMomentum => {
                    for ((p, v), g) in state
                        .values
                        .iter_mut()
                        .zip(state.velocity.iter_mut())
                        .zip(grad.iter())
                    {
                        *v = mom * *v + (1.0 - damp) * g;
                        let effective = *v + wd * *p;
                        *p -= lr * effective;
                    }
                }
                OptimizerType::SGDNesterov => {
                    for ((p, v), g) in state
                        .values
                        .iter_mut()
                        .zip(state.velocity.iter_mut())
                        .zip(grad.iter())
                    {
                        *v = mom * *v + g;
                        let effective = g + mom * *v + wd * *p;
                        *p -= lr * effective;
                    }
                }
            }
        }

        self.step_count += 1;
        Ok(())
    }

    /// Return a slice of the current parameter values, if the name exists.
    pub fn get_parameter(&self, name: &str) -> Option<&[f64]> {
        self.parameters.get(name).map(|s| s.values.as_slice())
    }

    /// Return a slice of the velocity buffer, if the name exists.
    pub fn get_velocity(&self, name: &str) -> Option<&[f64]> {
        self.parameters.get(name).map(|s| s.velocity.as_slice())
    }

    /// Dynamically change the learning rate.
    pub fn set_learning_rate(&mut self, lr: f64) {
        self.config.learning_rate = lr;
    }

    /// Total number of registered scalar parameter values.
    pub fn parameter_count(&self) -> usize {
        self.parameters.values().map(|s| s.values.len()).sum()
    }

    /// How many optimiser steps have been performed.
    pub fn step_count(&self) -> u64 {
        self.step_count
    }

    /// Reset all velocity buffers to zero.
    pub fn zero_velocities(&mut self) {
        for state in self.parameters.values_mut() {
            for v in &mut state.velocity {
                *v = 0.0;
            }
        }
    }

    /// Return a snapshot of the optimizer statistics.
    pub fn stats(&self) -> SGDOptimizerStats {
        SGDOptimizerStats {
            optimizer_type: self.config.optimizer_type,
            learning_rate: self.config.learning_rate,
            parameter_count: self.parameter_count(),
            step_count: self.step_count,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_grads(name: &str, vals: Vec<f64>) -> HashMap<String, Vec<f64>> {
        let mut m = HashMap::new();
        m.insert(name.to_string(), vals);
        m
    }

    // ---- SGD basic ----

    #[test]
    fn sgd_basic_step() {
        let mut opt = SGDOptimizer::new(SGDConfig::default());
        opt.register_parameter("w", vec![1.0, 2.0, 3.0]);
        let grads = make_grads("w", vec![0.1, 0.2, 0.3]);
        opt.step(&grads).expect("step should succeed");
        let w = opt.get_parameter("w").expect("param exists");
        // p -= lr * grad => 1.0 - 0.01*0.1 = 0.999
        assert!((w[0] - 0.999).abs() < 1e-12);
        assert!((w[1] - 1.998).abs() < 1e-12);
        assert!((w[2] - 2.997).abs() < 1e-12);
    }

    #[test]
    fn sgd_step_count_increments() {
        let mut opt = SGDOptimizer::new(SGDConfig::default());
        opt.register_parameter("w", vec![1.0]);
        assert_eq!(opt.step_count(), 0);
        opt.step(&make_grads("w", vec![0.1]))
            .expect("step should succeed");
        assert_eq!(opt.step_count(), 1);
        opt.step(&make_grads("w", vec![0.1]))
            .expect("step should succeed");
        assert_eq!(opt.step_count(), 2);
    }

    #[test]
    fn sgd_parameter_count() {
        let mut opt = SGDOptimizer::new(SGDConfig::default());
        opt.register_parameter("a", vec![1.0, 2.0]);
        opt.register_parameter("b", vec![3.0]);
        assert_eq!(opt.parameter_count(), 3);
    }

    #[test]
    fn sgd_zero_gradient_no_change() {
        let mut opt = SGDOptimizer::new(SGDConfig::default());
        opt.register_parameter("w", vec![5.0, 10.0]);
        opt.step(&make_grads("w", vec![0.0, 0.0]))
            .expect("step should succeed");
        let w = opt.get_parameter("w").expect("param exists");
        assert!((w[0] - 5.0).abs() < 1e-12);
        assert!((w[1] - 10.0).abs() < 1e-12);
    }

    // ---- Weight decay ----

    #[test]
    fn sgd_weight_decay() {
        let config = SGDConfig {
            optimizer_type: OptimizerType::SGD,
            learning_rate: 0.1,
            weight_decay: 0.01,
            ..SGDConfig::default()
        };
        let mut opt = SGDOptimizer::new(config);
        opt.register_parameter("w", vec![10.0]);
        opt.step(&make_grads("w", vec![0.0]))
            .expect("step should succeed");
        let w = opt.get_parameter("w").expect("param exists");
        // p -= lr * (0 + 0.01 * 10) = 10 - 0.1 * 0.1 = 9.99
        assert!((w[0] - 9.99).abs() < 1e-12);
    }

    #[test]
    fn sgd_weight_decay_with_gradient() {
        let config = SGDConfig {
            optimizer_type: OptimizerType::SGD,
            learning_rate: 0.01,
            weight_decay: 0.1,
            ..SGDConfig::default()
        };
        let mut opt = SGDOptimizer::new(config);
        opt.register_parameter("w", vec![2.0]);
        opt.step(&make_grads("w", vec![1.0]))
            .expect("step should succeed");
        let w = opt.get_parameter("w").expect("param exists");
        // p -= 0.01 * (1.0 + 0.1*2.0) = 2.0 - 0.01*1.2 = 1.988
        assert!((w[0] - 1.988).abs() < 1e-12);
    }

    // ---- Momentum ----

    #[test]
    fn momentum_accumulation() {
        let config = SGDConfig {
            optimizer_type: OptimizerType::SGDMomentum,
            learning_rate: 0.01,
            momentum: 0.9,
            dampening: 0.0,
            weight_decay: 0.0,
        };
        let mut opt = SGDOptimizer::new(config);
        opt.register_parameter("w", vec![1.0]);

        // Step 1: v = 0.9*0 + 1.0*0.5 = 0.5; p = 1.0 - 0.01*0.5 = 0.995
        opt.step(&make_grads("w", vec![0.5]))
            .expect("step should succeed");
        let v1 = opt.get_velocity("w").expect("vel exists")[0];
        assert!((v1 - 0.5).abs() < 1e-12);
        let w1 = opt.get_parameter("w").expect("param exists")[0];
        assert!((w1 - 0.995).abs() < 1e-12);

        // Step 2: v = 0.9*0.5 + 1.0*0.5 = 0.95; p = 0.995 - 0.01*0.95 = 0.9855
        opt.step(&make_grads("w", vec![0.5]))
            .expect("step should succeed");
        let v2 = opt.get_velocity("w").expect("vel exists")[0];
        assert!((v2 - 0.95).abs() < 1e-12);
    }

    #[test]
    fn momentum_with_dampening() {
        let config = SGDConfig {
            optimizer_type: OptimizerType::SGDMomentum,
            learning_rate: 0.1,
            momentum: 0.9,
            dampening: 0.5,
            weight_decay: 0.0,
        };
        let mut opt = SGDOptimizer::new(config);
        opt.register_parameter("w", vec![1.0]);

        // v = 0.9*0 + 0.5*1.0 = 0.5; p = 1.0 - 0.1*0.5 = 0.95
        opt.step(&make_grads("w", vec![1.0]))
            .expect("step should succeed");
        let v = opt.get_velocity("w").expect("vel exists")[0];
        assert!((v - 0.5).abs() < 1e-12);
        let w = opt.get_parameter("w").expect("param exists")[0];
        assert!((w - 0.95).abs() < 1e-12);
    }

    #[test]
    fn momentum_with_weight_decay() {
        let config = SGDConfig {
            optimizer_type: OptimizerType::SGDMomentum,
            learning_rate: 0.1,
            momentum: 0.9,
            dampening: 0.0,
            weight_decay: 0.01,
        };
        let mut opt = SGDOptimizer::new(config);
        opt.register_parameter("w", vec![10.0]);

        // v = 0.9*0 + 1.0*0.0 = 0.0 (grad=0)
        // effective = 0.0 + 0.01*10 = 0.1
        // p = 10.0 - 0.1*0.1 = 9.99
        opt.step(&make_grads("w", vec![0.0]))
            .expect("step should succeed");
        let w = opt.get_parameter("w").expect("param exists")[0];
        assert!((w - 9.99).abs() < 1e-12);
    }

    // ---- Nesterov ----

    #[test]
    fn nesterov_lookahead() {
        let config = SGDConfig {
            optimizer_type: OptimizerType::SGDNesterov,
            learning_rate: 0.01,
            momentum: 0.9,
            dampening: 0.0,
            weight_decay: 0.0,
        };
        let mut opt = SGDOptimizer::new(config);
        opt.register_parameter("w", vec![1.0]);

        // v = 0.9*0 + 0.5 = 0.5
        // effective = 0.5 + 0.9*0.5 = 0.95
        // p = 1.0 - 0.01*0.95 = 0.9905
        opt.step(&make_grads("w", vec![0.5]))
            .expect("step should succeed");
        let w = opt.get_parameter("w").expect("param exists")[0];
        assert!((w - 0.9905).abs() < 1e-12);
        let v = opt.get_velocity("w").expect("vel exists")[0];
        assert!((v - 0.5).abs() < 1e-12);
    }

    #[test]
    fn nesterov_two_steps() {
        let config = SGDConfig {
            optimizer_type: OptimizerType::SGDNesterov,
            learning_rate: 0.01,
            momentum: 0.9,
            dampening: 0.0,
            weight_decay: 0.0,
        };
        let mut opt = SGDOptimizer::new(config);
        opt.register_parameter("w", vec![1.0]);

        opt.step(&make_grads("w", vec![1.0]))
            .expect("step should succeed");
        // v1 = 0 + 1.0 = 1.0; eff = 1.0 + 0.9*1.0 = 1.9; p = 1.0 - 0.019 = 0.981
        let w1 = opt.get_parameter("w").expect("param exists")[0];
        assert!((w1 - 0.981).abs() < 1e-12);

        opt.step(&make_grads("w", vec![1.0]))
            .expect("step should succeed");
        // v2 = 0.9*1.0 + 1.0 = 1.9; eff = 1.0 + 0.9*1.9 = 2.71
        // p = 0.981 - 0.01*2.71 = 0.9539
        let w2 = opt.get_parameter("w").expect("param exists")[0];
        assert!((w2 - 0.9539).abs() < 1e-12);
    }

    #[test]
    fn nesterov_with_weight_decay() {
        let config = SGDConfig {
            optimizer_type: OptimizerType::SGDNesterov,
            learning_rate: 0.1,
            momentum: 0.9,
            dampening: 0.0,
            weight_decay: 0.01,
        };
        let mut opt = SGDOptimizer::new(config);
        opt.register_parameter("w", vec![10.0]);

        // v = 0 + 0 = 0
        // effective = 0 + 0.9*0 + 0.01*10 = 0.1
        // p = 10.0 - 0.1*0.1 = 9.99
        opt.step(&make_grads("w", vec![0.0]))
            .expect("step should succeed");
        let w = opt.get_parameter("w").expect("param exists")[0];
        assert!((w - 9.99).abs() < 1e-12);
    }

    // ---- Error handling ----

    #[test]
    fn gradient_name_mismatch_error() {
        let mut opt = SGDOptimizer::new(SGDConfig::default());
        opt.register_parameter("w", vec![1.0]);
        let grads = make_grads("wrong_name", vec![0.1]);
        let result = opt.step(&grads);
        assert!(result.is_err());
    }

    #[test]
    fn gradient_size_mismatch_error() {
        let mut opt = SGDOptimizer::new(SGDConfig::default());
        opt.register_parameter("w", vec![1.0, 2.0]);
        let grads = make_grads("w", vec![0.1]);
        let result = opt.step(&grads);
        assert!(result.is_err());
    }

    #[test]
    fn missing_gradient_error() {
        let mut opt = SGDOptimizer::new(SGDConfig::default());
        opt.register_parameter("a", vec![1.0]);
        opt.register_parameter("b", vec![2.0]);
        // Only provide gradient for "a"
        let grads = make_grads("a", vec![0.1]);
        let result = opt.step(&grads);
        assert!(result.is_err());
    }

    // ---- zero_velocities ----

    #[test]
    fn zero_velocities_resets() {
        let config = SGDConfig {
            optimizer_type: OptimizerType::SGDMomentum,
            learning_rate: 0.01,
            momentum: 0.9,
            ..SGDConfig::default()
        };
        let mut opt = SGDOptimizer::new(config);
        opt.register_parameter("w", vec![1.0]);
        opt.step(&make_grads("w", vec![1.0]))
            .expect("step should succeed");
        let v = opt.get_velocity("w").expect("vel exists")[0];
        assert!(v.abs() > 0.0);

        opt.zero_velocities();
        let v2 = opt.get_velocity("w").expect("vel exists")[0];
        assert!((v2).abs() < 1e-15);
    }

    // ---- Multiple parameters ----

    #[test]
    fn multiple_parameters() {
        let mut opt = SGDOptimizer::new(SGDConfig {
            learning_rate: 0.1,
            ..SGDConfig::default()
        });
        opt.register_parameter("w1", vec![1.0, 2.0]);
        opt.register_parameter("w2", vec![3.0]);

        let mut grads = HashMap::new();
        grads.insert("w1".to_string(), vec![0.1, 0.2]);
        grads.insert("w2".to_string(), vec![0.3]);

        opt.step(&grads).expect("step should succeed");
        let w1 = opt.get_parameter("w1").expect("param exists");
        assert!((w1[0] - 0.99).abs() < 1e-12);
        assert!((w1[1] - 1.98).abs() < 1e-12);
        let w2 = opt.get_parameter("w2").expect("param exists");
        assert!((w2[0] - 2.97).abs() < 1e-12);
    }

    // ---- Learning rate schedule ----

    #[test]
    fn learning_rate_schedule() {
        let mut opt = SGDOptimizer::new(SGDConfig {
            learning_rate: 0.1,
            ..SGDConfig::default()
        });
        opt.register_parameter("w", vec![10.0]);

        opt.step(&make_grads("w", vec![1.0]))
            .expect("step should succeed");
        let w1 = opt.get_parameter("w").expect("param exists")[0];
        assert!((w1 - 9.9).abs() < 1e-12);

        // Halve the LR
        opt.set_learning_rate(0.05);
        opt.step(&make_grads("w", vec![1.0]))
            .expect("step should succeed");
        let w2 = opt.get_parameter("w").expect("param exists")[0];
        assert!((w2 - 9.85).abs() < 1e-12);
    }

    // ---- Convergence test ----

    #[test]
    fn multiple_steps_convergence() {
        // Minimise f(x) = 0.5 * x^2  =>  grad = x
        let mut opt = SGDOptimizer::new(SGDConfig {
            optimizer_type: OptimizerType::SGD,
            learning_rate: 0.1,
            ..SGDConfig::default()
        });
        opt.register_parameter("x", vec![10.0]);

        for _ in 0..200 {
            let x = opt.get_parameter("x").expect("param exists")[0];
            opt.step(&make_grads("x", vec![x]))
                .expect("step should succeed");
        }
        let x_final = opt.get_parameter("x").expect("param exists")[0];
        assert!(
            x_final.abs() < 1e-6,
            "should converge near zero, got {}",
            x_final
        );
    }

    #[test]
    fn momentum_convergence_faster() {
        // Compare SGD vs Momentum on f(x) = 0.5*x^2
        let steps = 30;
        let lr = 0.01;

        // Plain SGD
        let mut sgd = SGDOptimizer::new(SGDConfig {
            optimizer_type: OptimizerType::SGD,
            learning_rate: lr,
            ..SGDConfig::default()
        });
        sgd.register_parameter("x", vec![10.0]);
        for _ in 0..steps {
            let x = sgd.get_parameter("x").expect("param exists")[0];
            sgd.step(&make_grads("x", vec![x]))
                .expect("step should succeed");
        }

        // Momentum SGD
        let mut mom = SGDOptimizer::new(SGDConfig {
            optimizer_type: OptimizerType::SGDMomentum,
            learning_rate: lr,
            momentum: 0.9,
            ..SGDConfig::default()
        });
        mom.register_parameter("x", vec![10.0]);
        for _ in 0..steps {
            let x = mom.get_parameter("x").expect("param exists")[0];
            mom.step(&make_grads("x", vec![x]))
                .expect("step should succeed");
        }

        let sgd_x = sgd.get_parameter("x").expect("param exists")[0].abs();
        let mom_x = mom.get_parameter("x").expect("param exists")[0].abs();
        assert!(
            mom_x < sgd_x,
            "momentum should converge faster: sgd={}, mom={}",
            sgd_x,
            mom_x
        );
    }

    // ---- Stats ----

    #[test]
    fn stats_accuracy() {
        let config = SGDConfig {
            optimizer_type: OptimizerType::SGDNesterov,
            learning_rate: 0.05,
            ..SGDConfig::default()
        };
        let mut opt = SGDOptimizer::new(config);
        opt.register_parameter("a", vec![1.0, 2.0, 3.0]);
        opt.register_parameter("b", vec![4.0, 5.0]);

        let stats = opt.stats();
        assert_eq!(stats.optimizer_type, OptimizerType::SGDNesterov);
        assert!((stats.learning_rate - 0.05).abs() < 1e-15);
        assert_eq!(stats.parameter_count, 5);
        assert_eq!(stats.step_count, 0);
    }

    #[test]
    fn stats_after_steps() {
        let mut opt = SGDOptimizer::new(SGDConfig::default());
        opt.register_parameter("w", vec![1.0]);
        for _ in 0..5 {
            opt.step(&make_grads("w", vec![0.1]))
                .expect("step should succeed");
        }
        assert_eq!(opt.stats().step_count, 5);
    }

    // ---- get_parameter / get_velocity on missing ----

    #[test]
    fn get_missing_parameter_returns_none() {
        let opt = SGDOptimizer::new(SGDConfig::default());
        assert!(opt.get_parameter("nonexistent").is_none());
    }

    #[test]
    fn get_missing_velocity_returns_none() {
        let opt = SGDOptimizer::new(SGDConfig::default());
        assert!(opt.get_velocity("nonexistent").is_none());
    }

    // ---- Display for OptimizerType ----

    #[test]
    fn optimizer_type_display() {
        assert_eq!(format!("{}", OptimizerType::SGD), "SGD");
        assert_eq!(format!("{}", OptimizerType::SGDMomentum), "SGDMomentum");
        assert_eq!(format!("{}", OptimizerType::SGDNesterov), "SGDNesterov");
    }

    // ---- Default config ----

    #[test]
    fn default_config_values() {
        let cfg = SGDConfig::default();
        assert_eq!(cfg.optimizer_type, OptimizerType::SGD);
        assert!((cfg.learning_rate - 0.01).abs() < 1e-15);
        assert!((cfg.momentum - 0.9).abs() < 1e-15);
        assert!((cfg.weight_decay).abs() < 1e-15);
        assert!((cfg.dampening).abs() < 1e-15);
    }

    // ---- Register replaces existing ----

    #[test]
    fn register_replaces_existing_parameter() {
        let mut opt = SGDOptimizer::new(SGDConfig::default());
        opt.register_parameter("w", vec![1.0, 2.0]);
        opt.register_parameter("w", vec![10.0]);
        assert_eq!(opt.parameter_count(), 1);
        let w = opt.get_parameter("w").expect("param exists");
        assert!((w[0] - 10.0).abs() < 1e-12);
    }

    // ---- Negative gradients ----

    #[test]
    fn negative_gradient_increases_parameter() {
        let mut opt = SGDOptimizer::new(SGDConfig {
            learning_rate: 0.1,
            ..SGDConfig::default()
        });
        opt.register_parameter("w", vec![0.0]);
        opt.step(&make_grads("w", vec![-1.0]))
            .expect("step should succeed");
        let w = opt.get_parameter("w").expect("param exists")[0];
        // p -= 0.1 * (-1.0) = 0.0 + 0.1 = 0.1
        assert!((w - 0.1).abs() < 1e-12);
    }

    // ---- Large gradient ----

    #[test]
    fn large_gradient_large_step() {
        let mut opt = SGDOptimizer::new(SGDConfig {
            learning_rate: 1.0,
            ..SGDConfig::default()
        });
        opt.register_parameter("w", vec![100.0]);
        opt.step(&make_grads("w", vec![100.0]))
            .expect("step should succeed");
        let w = opt.get_parameter("w").expect("param exists")[0];
        assert!((w - 0.0).abs() < 1e-12);
    }

    // ---- Nesterov convergence ----

    #[test]
    fn nesterov_convergence() {
        let mut opt = SGDOptimizer::new(SGDConfig {
            optimizer_type: OptimizerType::SGDNesterov,
            learning_rate: 0.01,
            momentum: 0.9,
            ..SGDConfig::default()
        });
        opt.register_parameter("x", vec![10.0]);

        for _ in 0..200 {
            let x = opt.get_parameter("x").expect("param exists")[0];
            opt.step(&make_grads("x", vec![x]))
                .expect("step should succeed");
        }
        let x_final = opt.get_parameter("x").expect("param exists")[0];
        assert!(
            x_final.abs() < 1e-4,
            "should converge near zero, got {}",
            x_final
        );
    }

    // ---- Empty step ----

    #[test]
    fn step_with_no_parameters() {
        let mut opt = SGDOptimizer::new(SGDConfig::default());
        let grads: HashMap<String, Vec<f64>> = HashMap::new();
        opt.step(&grads).expect("empty step should succeed");
        assert_eq!(opt.step_count(), 1);
    }
}
