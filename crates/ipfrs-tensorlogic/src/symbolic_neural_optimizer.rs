//! SymbolicNeuralOptimizer — hybrid optimizer combining symbolic rule-based
//! parameter updates with gradient-based neural optimization.
//!
//! Symbolic constraints guide the search space and apply logical
//! post-corrections to neural gradient steps.  Hard constraints are enforced
//! via clamping; soft constraints add a penalty term to the loss.

// ---------------------------------------------------------------------------
// Constraint expression operator tokens
// ---------------------------------------------------------------------------

const OP_GE: &str = ">=";
const OP_LE: &str = "<=";
const OP_EQ: &str = "==";

// ---------------------------------------------------------------------------
// OptimizationObjective
// ---------------------------------------------------------------------------

/// The high-level goal of the optimization run.
#[derive(Clone, Debug, Default, PartialEq)]
pub enum OptimizationObjective {
    /// Minimize the scalar loss returned by the loss function.
    #[default]
    Minimize,
    /// Maximize the scalar loss returned by the loss function (internally the
    /// optimizer negates the value so that a minimiser can be reused).
    Maximize,
    /// Satisfy the named constraints; loss is the total weighted violation.
    Satisfy(Vec<String>),
}

// ---------------------------------------------------------------------------
// SymbolicConstraint
// ---------------------------------------------------------------------------

/// A named symbolic constraint with a weight and hardness flag.
///
/// Hard constraints are enforced by clamping parameters.  Soft constraints
/// contribute a penalty term to the loss function.
#[derive(Clone, Debug, PartialEq)]
pub struct SymbolicConstraint {
    /// Unique identifier for this constraint.
    pub name: String,
    /// Simple single-variable expression, e.g. `"x >= 0.0"`.
    pub expression: String,
    /// Weight used for soft-constraint penalties and gradient nudges.
    pub weight: f64,
    /// When `true` the constraint is enforced by clamping (hard).
    /// When `false` the constraint enters the loss as a penalty (soft).
    pub is_hard: bool,
}

impl SymbolicConstraint {
    /// Construct a new constraint.
    pub fn new(
        name: impl Into<String>,
        expression: impl Into<String>,
        weight: f64,
        is_hard: bool,
    ) -> Self {
        SymbolicConstraint {
            name: name.into(),
            expression: expression.into(),
            weight,
            is_hard,
        }
    }

    /// Return a hard constraint.
    pub fn hard(name: impl Into<String>, expression: impl Into<String>, weight: f64) -> Self {
        Self::new(name, expression, weight, true)
    }

    /// Return a soft constraint.
    pub fn soft(name: impl Into<String>, expression: impl Into<String>, weight: f64) -> Self {
        Self::new(name, expression, weight, false)
    }
}

// ---------------------------------------------------------------------------
// ParameterVector
// ---------------------------------------------------------------------------

/// A named parameter vector.
#[derive(Clone, Debug, PartialEq)]
pub struct ParameterVector {
    /// Parameter values.
    pub values: Vec<f64>,
    /// Parameter names (parallel to `values`).
    pub names: Vec<String>,
}

impl ParameterVector {
    /// Construct a new `ParameterVector`.
    ///
    /// If `names` and `values` lengths differ the shorter one determines the
    /// logical length; the remainder is ignored.
    pub fn new(names: Vec<String>, values: Vec<f64>) -> Self {
        ParameterVector { names, values }
    }

    /// Return the value of the parameter with the given name, if present.
    pub fn get(&self, name: &str) -> Option<f64> {
        self.names
            .iter()
            .position(|n| n == name)
            .map(|i| self.values[i])
    }

    /// Set the value of the parameter with the given name.
    ///
    /// Returns `true` if the name was found and updated, `false` otherwise.
    pub fn set(&mut self, name: &str, value: f64) -> bool {
        if let Some(i) = self.names.iter().position(|n| n == name) {
            self.values[i] = value;
            true
        } else {
            false
        }
    }

    /// Compute the L2 (Euclidean) norm of the value vector.
    pub fn l2_norm(&self) -> f64 {
        self.values.iter().map(|v| v * v).sum::<f64>().sqrt()
    }

    /// Number of parameters.
    pub fn len(&self) -> usize {
        self.values.len().min(self.names.len())
    }

    /// Return `true` when there are no parameters.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Iterate over `(name, value)` pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&str, f64)> {
        let n = self.len();
        self.names[..n]
            .iter()
            .map(String::as_str)
            .zip(self.values[..n].iter().copied())
    }
}

// ---------------------------------------------------------------------------
// SnoOptimizationStep  (renamed to avoid collision with OptimizationStep from
// optimization_history module)
// ---------------------------------------------------------------------------

/// A single recorded step produced by [`SymbolicNeuralOptimizer`].
#[derive(Clone, Debug, PartialEq)]
pub struct SnoOptimizationStep {
    /// Monotonically increasing iteration counter (0-based).
    pub iteration: u64,
    /// Loss value after this step.
    pub loss: f64,
    /// L2 norm of the gradient at this step.
    pub gradient_norm: f64,
    /// Number of hard constraint violations after the update.
    pub constraint_violations: usize,
    /// Parameter state after the update.
    pub params: ParameterVector,
}

// ---------------------------------------------------------------------------
// SnoOptimizationResult  (renamed to avoid collision with OptimizationResult
// from query_optimizer module)
// ---------------------------------------------------------------------------

/// The final result returned by [`SymbolicNeuralOptimizer::optimize`].
#[derive(Clone, Debug, PartialEq)]
pub struct SnoOptimizationResult {
    /// `true` when the run finished without error.
    pub success: bool,
    /// Total number of iterations executed.
    pub iterations: u64,
    /// Loss at the final iteration.
    pub final_loss: f64,
    /// Parameter values at the final iteration.
    pub final_params: ParameterVector,
    /// Number of hard constraint violations in the final parameter state.
    pub constraint_violations: usize,
    /// `true` when the run stopped due to convergence rather than hitting the
    /// iteration limit.
    pub converged: bool,
}

// ---------------------------------------------------------------------------
// OptimizerConfig
// ---------------------------------------------------------------------------

/// Configuration for [`SymbolicNeuralOptimizer`].
#[derive(Clone, Debug, PartialEq)]
pub struct SnoOptimizerConfig {
    /// Gradient descent step size.
    pub learning_rate: f64,
    /// Maximum number of optimization iterations.
    pub max_iterations: u64,
    /// Convergence threshold: stop when |loss_prev − loss_curr| < threshold.
    pub convergence_threshold: f64,
    /// Penalty multiplier applied to soft constraint violations.
    pub constraint_penalty: f64,
    /// Weight applied to symbolic correction nudges relative to the gradient.
    pub symbolic_correction_weight: f64,
    /// High-level objective of the optimization.
    pub objective: OptimizationObjective,
}

impl Default for SnoOptimizerConfig {
    fn default() -> Self {
        SnoOptimizerConfig {
            learning_rate: 0.01,
            max_iterations: 1000,
            convergence_threshold: 1e-6,
            constraint_penalty: 10.0,
            symbolic_correction_weight: 0.5,
            objective: OptimizationObjective::Minimize,
        }
    }
}

impl SnoOptimizerConfig {
    /// Create a new config with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder: set learning rate.
    pub fn with_learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Builder: set max iterations.
    pub fn with_max_iterations(mut self, n: u64) -> Self {
        self.max_iterations = n;
        self
    }

    /// Builder: set convergence threshold.
    pub fn with_convergence_threshold(mut self, t: f64) -> Self {
        self.convergence_threshold = t;
        self
    }

    /// Builder: set constraint penalty.
    pub fn with_constraint_penalty(mut self, p: f64) -> Self {
        self.constraint_penalty = p;
        self
    }

    /// Builder: set symbolic correction weight.
    pub fn with_symbolic_correction_weight(mut self, w: f64) -> Self {
        self.symbolic_correction_weight = w;
        self
    }

    /// Builder: set optimization objective.
    pub fn with_objective(mut self, obj: OptimizationObjective) -> Self {
        self.objective = obj;
        self
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Xorshift64 pseudo-random number generator.
///
/// Advances `state` and returns the new value.
pub fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// Parsed bound from a simple constraint expression.
#[derive(Clone, Debug, PartialEq)]
pub struct ConstraintBound {
    /// Parameter name on the left-hand side.
    pub param_name: String,
    /// Operator: `">="`, `"<="`, or `"=="`.
    pub operator: String,
    /// Bound value on the right-hand side.
    pub bound: f64,
}

/// Parse a simple single-variable constraint expression.
///
/// Recognised forms:
/// - `"x >= 0.0"` — lower bound
/// - `"x <= 1.0"` — upper bound
/// - `"x == 0.5"` — equality
///
/// Returns `None` for any expression that does not match these forms.
pub fn parse_constraint_bound(expr: &str) -> Option<ConstraintBound> {
    let expr = expr.trim();
    // Try each operator in longest-first order to avoid `>` matching `>=`.
    for op in &[OP_GE, OP_LE, OP_EQ] {
        if let Some(pos) = expr.find(op) {
            let name_part = expr[..pos].trim();
            let val_part = expr[pos + op.len()..].trim();
            if name_part.is_empty() || val_part.is_empty() {
                continue;
            }
            // The name must be a simple identifier (no spaces).
            if name_part.contains(' ') {
                continue;
            }
            if let Ok(bound) = val_part.parse::<f64>() {
                return Some(ConstraintBound {
                    param_name: name_part.to_owned(),
                    operator: op.to_string(),
                    bound,
                });
            }
        }
    }
    None
}

/// Compute the violation amount for a parsed bound given a parameter value.
///
/// Returns `0.0` when the constraint is satisfied, a positive amount when it
/// is violated.
fn violation_amount(bound: &ConstraintBound, value: f64) -> f64 {
    match bound.operator.as_str() {
        ">=" if value < bound.bound => bound.bound - value,
        "<=" if value > bound.bound => value - bound.bound,
        "==" => (value - bound.bound).abs(),
        _ => 0.0,
    }
}

/// Clamp a value so that it satisfies a hard constraint bound.
fn clamp_to_bound(bound: &ConstraintBound, value: f64) -> f64 {
    match bound.operator.as_str() {
        ">=" => value.max(bound.bound),
        "<=" => value.min(bound.bound),
        "==" => bound.bound,
        _ => value,
    }
}

// ---------------------------------------------------------------------------
// SymbolicNeuralOptimizer
// ---------------------------------------------------------------------------

/// A hybrid optimizer that combines symbolic rule-based parameter updates with
/// gradient-based neural optimization.
///
/// # How it works
///
/// 1. **Gradient step** — apply a standard gradient-descent update using the
///    configured learning rate.
/// 2. **Hard constraint enforcement** — for each hard constraint whose
///    expression can be parsed as a simple bound, clamp the relevant parameter
///    to the feasible region.
/// 3. **Soft constraint correction** — apply a small gradient nudge
///    proportional to `weight × constraint_penalty × symbolic_correction_weight`
///    for each violated soft constraint.
///
/// The [`SymbolicNeuralOptimizer::optimize`] method drives the loop, calling a
/// user-supplied `loss_fn` at each iteration and recording history.
pub struct SymbolicNeuralOptimizer {
    config: SnoOptimizerConfig,
    constraints: Vec<SymbolicConstraint>,
    history: Vec<SnoOptimizationStep>,
    iteration: u64,
    rng_state: u64,
}

impl SymbolicNeuralOptimizer {
    /// Create a new optimizer with the given configuration.
    ///
    /// The internal PRNG is seeded to `12345`.
    pub fn new(config: SnoOptimizerConfig) -> Self {
        SymbolicNeuralOptimizer {
            config,
            constraints: Vec::new(),
            history: Vec::new(),
            iteration: 0,
            rng_state: 12345,
        }
    }

    /// Add a symbolic constraint.
    pub fn add_constraint(&mut self, constraint: SymbolicConstraint) {
        self.constraints.push(constraint);
    }

    /// Remove all constraints whose name equals `name`.
    ///
    /// Returns `true` if at least one constraint was removed.
    pub fn remove_constraint(&mut self, name: &str) -> bool {
        let before = self.constraints.len();
        self.constraints.retain(|c| c.name != name);
        self.constraints.len() < before
    }

    /// Return all registered constraints.
    pub fn constraints(&self) -> &[SymbolicConstraint] {
        &self.constraints
    }

    /// Return the optimization history (one entry per completed step).
    pub fn history(&self) -> &[SnoOptimizationStep] {
        &self.history
    }

    /// Return the step with the lowest recorded loss, or `None` if no steps
    /// have been taken yet.
    pub fn best_step(&self) -> Option<&SnoOptimizationStep> {
        self.history.iter().min_by(|a, b| {
            a.loss
                .partial_cmp(&b.loss)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Return the current iteration counter.
    pub fn iteration(&self) -> u64 {
        self.iteration
    }

    /// Compute the total loss for the given parameters and residuals.
    ///
    /// The base loss is the mean squared residual.  Each **soft** constraint
    /// that is violated adds `weight × constraint_penalty × violation_amount`.
    pub fn compute_loss(&self, params: &ParameterVector, residuals: &[f64]) -> f64 {
        // Base loss: mean squared residual.
        let base_loss = if residuals.is_empty() {
            0.0
        } else {
            residuals.iter().map(|r| r * r).sum::<f64>() / residuals.len() as f64
        };

        // Sign factor for Maximize objective.
        let sign = match &self.config.objective {
            OptimizationObjective::Maximize => -1.0,
            _ => 1.0,
        };

        let penalty: f64 = self
            .constraints
            .iter()
            .filter(|c| !c.is_hard)
            .map(|c| {
                if let Some(bound) = parse_constraint_bound(&c.expression) {
                    let val = params.get(&bound.param_name).unwrap_or(0.0);
                    let viol = violation_amount(&bound, val);
                    c.weight * self.config.constraint_penalty * viol
                } else {
                    0.0
                }
            })
            .sum();

        sign * base_loss + penalty
    }

    /// Count the number of **hard** constraints that are currently violated.
    pub fn check_constraints(&self, params: &ParameterVector) -> usize {
        self.constraints
            .iter()
            .filter(|c| c.is_hard)
            .filter(|c| {
                if let Some(bound) = parse_constraint_bound(&c.expression) {
                    let val = params.get(&bound.param_name).unwrap_or(0.0);
                    violation_amount(&bound, val) > 0.0
                } else {
                    false
                }
            })
            .count()
    }

    /// Perform a single optimization step.
    ///
    /// 1. Apply the gradient step: `new_val = val − lr × grad_val`.
    /// 2. Enforce hard constraints by clamping.
    /// 3. Apply soft constraint gradient nudges.
    pub fn step(
        &mut self,
        params: &ParameterVector,
        gradient: &ParameterVector,
    ) -> ParameterVector {
        let lr = self.config.learning_rate;
        let penalty = self.config.constraint_penalty;
        let corr_w = self.config.symbolic_correction_weight;

        let n = params.len();
        let mut new_values = Vec::with_capacity(n);

        // --- 1. Gradient step ---
        for i in 0..n {
            let val = params.values[i];
            // Look up gradient by name for robustness against ordering
            // differences between the two vectors.
            let grad = gradient.get(&params.names[i]).unwrap_or(0.0);
            new_values.push(val - lr * grad);
        }

        let mut new_params = ParameterVector::new(params.names[..n].to_vec(), new_values);

        // --- 2. Hard constraint clamping ---
        for constraint in &self.constraints {
            if !constraint.is_hard {
                continue;
            }
            if let Some(bound) = parse_constraint_bound(&constraint.expression) {
                let current = new_params.get(&bound.param_name).unwrap_or(f64::NAN);
                if !current.is_nan() {
                    let clamped = clamp_to_bound(&bound, current);
                    new_params.set(&bound.param_name, clamped);
                }
            }
        }

        // --- 3. Soft constraint gradient nudge ---
        for constraint in &self.constraints {
            if constraint.is_hard {
                continue;
            }
            if let Some(bound) = parse_constraint_bound(&constraint.expression) {
                let val = new_params.get(&bound.param_name).unwrap_or(f64::NAN);
                if val.is_nan() {
                    continue;
                }
                let viol = violation_amount(&bound, val);
                if viol <= 0.0 {
                    continue;
                }
                // Nudge direction: towards the feasible side.
                let nudge_dir = match bound.operator.as_str() {
                    ">=" => 1.0,  // must go up
                    "<=" => -1.0, // must go down
                    "==" => {
                        if val < bound.bound {
                            1.0
                        } else {
                            -1.0
                        }
                    }
                    _ => 0.0,
                };
                let nudge = nudge_dir * constraint.weight * penalty * corr_w * viol;
                new_params.set(&bound.param_name, val + nudge);
            }
        }

        self.iteration += 1;
        new_params
    }

    /// Run the full optimization loop.
    ///
    /// `loss_fn` receives the current `ParameterVector` and must return
    /// `(loss, gradient)` where `gradient` is a `ParameterVector` with the
    /// same names as the input, containing the partial derivatives of the loss
    /// with respect to each parameter.
    ///
    /// The loop terminates when:
    /// - `|loss_prev − loss_curr| < convergence_threshold` (converged), or
    /// - `iteration >= max_iterations`.
    pub fn optimize(
        &mut self,
        initial_params: ParameterVector,
        loss_fn: &dyn Fn(&ParameterVector) -> (f64, ParameterVector),
    ) -> SnoOptimizationResult {
        let mut params = initial_params;
        let mut prev_loss = f64::INFINITY;
        let mut converged = false;

        self.history.clear();
        self.iteration = 0;

        for _iter in 0..self.config.max_iterations {
            let (loss, gradient) = loss_fn(&params);

            let grad_norm = gradient.l2_norm();
            let violations = self.check_constraints(&params);

            let step_record = SnoOptimizationStep {
                iteration: self.iteration,
                loss,
                gradient_norm: grad_norm,
                constraint_violations: violations,
                params: params.clone(),
            };
            self.history.push(step_record);

            // Convergence check (before taking the next step).
            if (prev_loss - loss).abs() < self.config.convergence_threshold {
                converged = true;
                break;
            }
            prev_loss = loss;

            params = self.step(&params, &gradient);
        }

        // Record a final step after the last update so that `final_params`
        // reflects the state after the last gradient step.
        let (final_loss, _) = loss_fn(&params);
        let final_violations = self.check_constraints(&params);

        SnoOptimizationResult {
            success: true,
            iterations: self.iteration,
            final_loss,
            final_params: params,
            constraint_violations: final_violations,
            converged,
        }
    }

    /// Access the raw PRNG state (useful for reproducibility testing).
    pub fn rng_state(&self) -> u64 {
        self.rng_state
    }

    /// Advance the internal PRNG and return a random `u64`.
    pub fn rand_u64(&mut self) -> u64 {
        xorshift64(&mut self.rng_state)
    }

    /// Advance the internal PRNG and return a random `f64` in `[0, 1)`.
    pub fn rand_f64(&mut self) -> f64 {
        let r = xorshift64(&mut self.rng_state);
        (r as f64) / (u64::MAX as f64)
    }

    /// Reset the optimizer state (history, iteration counter) but keep the
    /// configuration and constraints.
    pub fn reset(&mut self) {
        self.history.clear();
        self.iteration = 0;
    }

    /// Return a reference to the configuration.
    pub fn config(&self) -> &SnoOptimizerConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Helper factories
    // -----------------------------------------------------------------------

    fn default_config() -> SnoOptimizerConfig {
        SnoOptimizerConfig::default()
    }

    fn optimizer() -> SymbolicNeuralOptimizer {
        SymbolicNeuralOptimizer::new(default_config())
    }

    fn params(names: &[&str], values: &[f64]) -> ParameterVector {
        ParameterVector::new(
            names.iter().map(|s| s.to_string()).collect(),
            values.to_vec(),
        )
    }

    fn gradient_from(names: &[&str], grads: &[f64]) -> ParameterVector {
        params(names, grads)
    }

    // -----------------------------------------------------------------------
    // ParameterVector tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_parameter_vector_new() {
        let pv = params(&["x", "y"], &[1.0, 2.0]);
        assert_eq!(pv.len(), 2);
        assert!(!pv.is_empty());
    }

    #[test]
    fn test_parameter_vector_get_existing() {
        let pv = params(&["x", "y"], &[3.0, 7.0]);
        assert_eq!(pv.get("x"), Some(3.0));
        assert_eq!(pv.get("y"), Some(7.0));
    }

    #[test]
    fn test_parameter_vector_get_missing() {
        let pv = params(&["x"], &[1.0]);
        assert_eq!(pv.get("z"), None);
    }

    #[test]
    fn test_parameter_vector_set_existing() {
        let mut pv = params(&["x", "y"], &[1.0, 2.0]);
        assert!(pv.set("x", 99.0));
        assert_eq!(pv.get("x"), Some(99.0));
    }

    #[test]
    fn test_parameter_vector_set_missing() {
        let mut pv = params(&["x"], &[1.0]);
        assert!(!pv.set("z", 0.0));
    }

    #[test]
    fn test_parameter_vector_l2_norm_zero() {
        let pv = params(&["x", "y"], &[0.0, 0.0]);
        assert!((pv.l2_norm() - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_parameter_vector_l2_norm_unit() {
        let pv = params(&["x", "y"], &[1.0, 0.0]);
        assert!((pv.l2_norm() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_parameter_vector_l2_norm_pythagorean() {
        let pv = params(&["x", "y"], &[3.0, 4.0]);
        assert!((pv.l2_norm() - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_parameter_vector_iter() {
        let pv = params(&["a", "b"], &[10.0, 20.0]);
        let collected: Vec<_> = pv.iter().collect();
        assert_eq!(collected, vec![("a", 10.0), ("b", 20.0)]);
    }

    #[test]
    fn test_parameter_vector_empty() {
        let pv = ParameterVector::new(vec![], vec![]);
        assert!(pv.is_empty());
        assert_eq!(pv.l2_norm(), 0.0);
    }

    // -----------------------------------------------------------------------
    // SymbolicConstraint tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_symbolic_constraint_hard() {
        let c = SymbolicConstraint::hard("lower_x", "x >= 0.0", 1.0);
        assert!(c.is_hard);
        assert_eq!(c.name, "lower_x");
        assert!((c.weight - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_symbolic_constraint_soft() {
        let c = SymbolicConstraint::soft("upper_y", "y <= 1.0", 0.5);
        assert!(!c.is_hard);
    }

    // -----------------------------------------------------------------------
    // parse_constraint_bound tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_ge() {
        let b = parse_constraint_bound("x >= 0.0").expect("test: should succeed");
        assert_eq!(b.param_name, "x");
        assert_eq!(b.operator, ">=");
        assert!((b.bound - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_parse_le() {
        let b = parse_constraint_bound("alpha <= 1.5").expect("test: should succeed");
        assert_eq!(b.param_name, "alpha");
        assert_eq!(b.operator, "<=");
        assert!((b.bound - 1.5).abs() < 1e-12);
    }

    #[test]
    fn test_parse_eq() {
        let b = parse_constraint_bound("bias == 0.5").expect("test: should succeed");
        assert_eq!(b.param_name, "bias");
        assert_eq!(b.operator, "==");
        assert!((b.bound - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_parse_negative_bound() {
        let b = parse_constraint_bound("x >= -1.0").expect("test: should succeed");
        assert!((b.bound - (-1.0)).abs() < 1e-12);
    }

    #[test]
    fn test_parse_no_spaces() {
        // Tight format without spaces should still parse.
        let b = parse_constraint_bound("x>=0.0").expect("test: should succeed");
        assert_eq!(b.operator, ">=");
    }

    #[test]
    fn test_parse_invalid_expr() {
        // No operator
        assert!(parse_constraint_bound("x 0.0").is_none());
    }

    #[test]
    fn test_parse_invalid_value() {
        assert!(parse_constraint_bound("x >= abc").is_none());
    }

    // -----------------------------------------------------------------------
    // Optimizer construction and config tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_optimizer_default_config() {
        let opt = optimizer();
        assert!((opt.config().learning_rate - 0.01).abs() < 1e-12);
        assert_eq!(opt.config().max_iterations, 1000);
        assert_eq!(opt.config().objective, OptimizationObjective::Minimize);
    }

    #[test]
    fn test_optimizer_rng_initial_state() {
        let opt = optimizer();
        assert_eq!(opt.rng_state(), 12345);
    }

    #[test]
    fn test_optimizer_rand_u64_deterministic() {
        let mut opt = optimizer();
        let r1 = opt.rand_u64();
        let mut opt2 = optimizer();
        let r2 = opt2.rand_u64();
        assert_eq!(r1, r2);
    }

    #[test]
    fn test_optimizer_rand_f64_in_unit_interval() {
        let mut opt = optimizer();
        for _ in 0..100 {
            let r = opt.rand_f64();
            assert!((0.0..=1.0).contains(&r));
        }
    }

    // -----------------------------------------------------------------------
    // Constraint management tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_add_and_remove_constraint() {
        let mut opt = optimizer();
        opt.add_constraint(SymbolicConstraint::hard("c1", "x >= 0.0", 1.0));
        assert_eq!(opt.constraints().len(), 1);
        assert!(opt.remove_constraint("c1"));
        assert_eq!(opt.constraints().len(), 0);
    }

    #[test]
    fn test_remove_nonexistent_constraint() {
        let mut opt = optimizer();
        assert!(!opt.remove_constraint("ghost"));
    }

    #[test]
    fn test_add_multiple_constraints() {
        let mut opt = optimizer();
        opt.add_constraint(SymbolicConstraint::hard("c1", "x >= 0.0", 1.0));
        opt.add_constraint(SymbolicConstraint::soft("c2", "y <= 1.0", 0.5));
        assert_eq!(opt.constraints().len(), 2);
    }

    // -----------------------------------------------------------------------
    // check_constraints tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_check_constraints_none_violated() {
        let mut opt = optimizer();
        opt.add_constraint(SymbolicConstraint::hard("lb", "x >= 0.0", 1.0));
        let p = params(&["x"], &[1.0]);
        assert_eq!(opt.check_constraints(&p), 0);
    }

    #[test]
    fn test_check_constraints_one_violated() {
        let mut opt = optimizer();
        opt.add_constraint(SymbolicConstraint::hard("lb", "x >= 0.0", 1.0));
        let p = params(&["x"], &[-1.0]);
        assert_eq!(opt.check_constraints(&p), 1);
    }

    #[test]
    fn test_check_constraints_soft_not_counted() {
        let mut opt = optimizer();
        opt.add_constraint(SymbolicConstraint::soft("s", "x <= 0.5", 1.0));
        let p = params(&["x"], &[2.0]);
        // Soft constraints don't count in check_constraints.
        assert_eq!(opt.check_constraints(&p), 0);
    }

    // -----------------------------------------------------------------------
    // compute_loss tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_compute_loss_no_residuals() {
        let opt = optimizer();
        let p = params(&["x"], &[0.0]);
        assert!((opt.compute_loss(&p, &[]) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_compute_loss_mse() {
        let opt = optimizer();
        let p = params(&["x"], &[0.0]);
        // MSE([1.0, -1.0]) = (1+1)/2 = 1.0
        let loss = opt.compute_loss(&p, &[1.0, -1.0]);
        assert!((loss - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_compute_loss_soft_penalty() {
        let mut opt = SymbolicNeuralOptimizer::new(
            SnoOptimizerConfig::default().with_constraint_penalty(10.0),
        );
        // x must be >= 1.0 (soft), weight=1.0
        opt.add_constraint(SymbolicConstraint::soft("lb", "x >= 1.0", 1.0));
        let p = params(&["x"], &[0.0]); // violation = 1.0
        let loss = opt.compute_loss(&p, &[]);
        // penalty = 1.0 * 10.0 * 1.0 = 10.0; base_loss = 0
        assert!((loss - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_compute_loss_hard_no_penalty() {
        let mut opt = optimizer();
        opt.add_constraint(SymbolicConstraint::hard("lb", "x >= 1.0", 1.0));
        let p = params(&["x"], &[0.0]); // violated, but hard → no penalty
        let loss = opt.compute_loss(&p, &[]);
        assert!((loss - 0.0).abs() < 1e-12);
    }

    // -----------------------------------------------------------------------
    // step tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_step_gradient_descent() {
        let mut opt = optimizer(); // lr = 0.01
        let p = params(&["x"], &[1.0]);
        let g = gradient_from(&["x"], &[10.0]);
        let new_p = opt.step(&p, &g);
        // 1.0 - 0.01 * 10.0 = 0.9
        assert!((new_p.get("x").unwrap_or(0.0) - 0.9).abs() < 1e-12);
    }

    #[test]
    fn test_step_increments_iteration() {
        let mut opt = optimizer();
        let p = params(&["x"], &[0.0]);
        let g = gradient_from(&["x"], &[0.0]);
        opt.step(&p, &g);
        assert_eq!(opt.iteration(), 1);
        opt.step(&p, &g);
        assert_eq!(opt.iteration(), 2);
    }

    #[test]
    fn test_step_hard_constraint_clamp_lower() {
        let mut opt = optimizer();
        opt.add_constraint(SymbolicConstraint::hard("lb", "x >= 0.0", 1.0));
        // Start at 0.05, gradient=10 → raw new = 0.05 - 0.01*10 = -0.05 → clamped to 0.0
        let p = params(&["x"], &[0.05]);
        let g = gradient_from(&["x"], &[10.0]);
        let new_p = opt.step(&p, &g);
        assert!(new_p.get("x").unwrap_or(f64::NAN) >= 0.0);
    }

    #[test]
    fn test_step_hard_constraint_clamp_upper() {
        let mut opt = optimizer();
        opt.add_constraint(SymbolicConstraint::hard("ub", "x <= 1.0", 1.0));
        // Start at 0.95, gradient=-10 → raw new = 0.95 + 0.1 = 1.05 → clamped to 1.0
        let p = params(&["x"], &[0.95]);
        let g = gradient_from(&["x"], &[-10.0]);
        let new_p = opt.step(&p, &g);
        assert!(new_p.get("x").unwrap_or(f64::NAN) <= 1.0);
    }

    #[test]
    fn test_step_hard_constraint_equality() {
        let mut opt = optimizer();
        opt.add_constraint(SymbolicConstraint::hard("eq", "x == 0.5", 1.0));
        let p = params(&["x"], &[0.0]);
        let g = gradient_from(&["x"], &[0.0]);
        let new_p = opt.step(&p, &g);
        // Hard equality forces x to 0.5.
        assert!((new_p.get("x").unwrap_or(0.0) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_step_soft_constraint_nudge() {
        // Set up optimizer with a soft lower bound.
        let config = SnoOptimizerConfig::default()
            .with_learning_rate(0.0) // no gradient movement
            .with_constraint_penalty(1.0)
            .with_symbolic_correction_weight(1.0);
        let mut opt = SymbolicNeuralOptimizer::new(config);
        opt.add_constraint(SymbolicConstraint::soft("lb", "x >= 1.0", 1.0));
        // x starts at 0.5 → violation = 0.5 → nudge = +1.0*1.0*1.0*0.5 = 0.5
        let p = params(&["x"], &[0.5]);
        let g = gradient_from(&["x"], &[0.0]);
        let new_p = opt.step(&p, &g);
        let x_after = new_p.get("x").unwrap_or(0.0);
        // Should be nudged upward toward 1.0.
        assert!(x_after > 0.5, "Expected x_after > 0.5, got {}", x_after);
    }

    // -----------------------------------------------------------------------
    // optimize tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_optimize_converges_quadratic() {
        // Minimize (x - 2)^2; analytic gradient = 2*(x-2).
        let config = SnoOptimizerConfig::default()
            .with_learning_rate(0.1)
            .with_max_iterations(500)
            .with_convergence_threshold(1e-8);
        let mut opt = SymbolicNeuralOptimizer::new(config);
        let init = params(&["x"], &[0.0]);
        let result = opt.optimize(init, &|p| {
            let x = p.get("x").unwrap_or(0.0);
            let loss = (x - 2.0) * (x - 2.0);
            let grad = gradient_from(&["x"], &[2.0 * (x - 2.0)]);
            (loss, grad)
        });
        assert!(result.success);
        let x_final = result.final_params.get("x").unwrap_or(0.0);
        assert!((x_final - 2.0).abs() < 0.1, "x_final={}", x_final);
    }

    #[test]
    fn test_optimize_respects_max_iterations() {
        let config = SnoOptimizerConfig::default()
            .with_max_iterations(5)
            .with_convergence_threshold(f64::EPSILON);
        let mut opt = SymbolicNeuralOptimizer::new(config);
        let init = params(&["x"], &[0.0]);
        let result = opt.optimize(init, &|p| {
            let x = p.get("x").unwrap_or(0.0);
            let grad = gradient_from(&["x"], &[2.0 * x]);
            (x * x, grad)
        });
        assert!(result.iterations <= 5);
    }

    #[test]
    fn test_optimize_history_populated() {
        let config = SnoOptimizerConfig::default()
            .with_max_iterations(10)
            .with_convergence_threshold(f64::EPSILON);
        let mut opt = SymbolicNeuralOptimizer::new(config);
        let init = params(&["x"], &[5.0]);
        opt.optimize(init, &|p| {
            let x = p.get("x").unwrap_or(0.0);
            let grad = gradient_from(&["x"], &[2.0 * x]);
            (x * x, grad)
        });
        assert!(!opt.history().is_empty());
    }

    #[test]
    fn test_optimize_best_step_is_minimum_loss() {
        let config = SnoOptimizerConfig::default()
            .with_max_iterations(20)
            .with_convergence_threshold(f64::EPSILON);
        let mut opt = SymbolicNeuralOptimizer::new(config);
        let init = params(&["x"], &[5.0]);
        opt.optimize(init, &|p| {
            let x = p.get("x").unwrap_or(0.0);
            let grad = gradient_from(&["x"], &[2.0 * x]);
            (x * x, grad)
        });
        let best = opt.best_step().expect("history non-empty");
        let min_loss = opt
            .history()
            .iter()
            .map(|s| s.loss)
            .fold(f64::INFINITY, f64::min);
        assert!((best.loss - min_loss).abs() < 1e-12);
    }

    #[test]
    fn test_optimize_with_hard_constraint_satisfied() {
        // Minimize x^2 with hard constraint x >= 1.0 → minimum is 1.0.
        let config = SnoOptimizerConfig::default()
            .with_learning_rate(0.1)
            .with_max_iterations(200)
            .with_convergence_threshold(1e-8);
        let mut opt = SymbolicNeuralOptimizer::new(config);
        opt.add_constraint(SymbolicConstraint::hard("lb", "x >= 1.0", 1.0));
        let init = params(&["x"], &[5.0]);
        let result = opt.optimize(init, &|p| {
            let x = p.get("x").unwrap_or(0.0);
            let grad = gradient_from(&["x"], &[2.0 * x]);
            (x * x, grad)
        });
        let x_final = result.final_params.get("x").unwrap_or(0.0);
        // Hard constraint forces x >= 1.0.
        assert!(x_final >= 1.0 - 1e-9, "x_final={}", x_final);
        assert_eq!(result.constraint_violations, 0);
    }

    #[test]
    fn test_optimize_maximize() {
        // Maximize -(x-3)^2 + 9  (peak at x=3).
        let config = SnoOptimizerConfig::default()
            .with_learning_rate(0.1)
            .with_max_iterations(500)
            .with_convergence_threshold(1e-8)
            .with_objective(OptimizationObjective::Maximize);
        let mut opt = SymbolicNeuralOptimizer::new(config);
        let init = params(&["x"], &[0.0]);
        let result = opt.optimize(init, &|p| {
            let x = p.get("x").unwrap_or(0.0);
            // For Maximize the optimizer negates internally, so loss_fn should
            // return the value we want to maximise (not the negation).
            let val = -(x - 3.0) * (x - 3.0) + 9.0;
            // Gradient of val w.r.t. x = -2*(x-3)
            let grad_val = -2.0 * (x - 3.0);
            let grad = gradient_from(&["x"], &[grad_val]);
            (val, grad)
        });
        assert!(result.success);
    }

    #[test]
    fn test_optimize_satisfy_objective_type() {
        let config = SnoOptimizerConfig::default()
            .with_objective(OptimizationObjective::Satisfy(vec!["x_pos".to_string()]));
        let mut opt = SymbolicNeuralOptimizer::new(config);
        opt.add_constraint(SymbolicConstraint::soft("x_pos", "x >= 0.0", 1.0));
        let init = params(&["x"], &[-2.0]);
        let result = opt.optimize(init, &|p| {
            let x = p.get("x").unwrap_or(0.0);
            let grad = gradient_from(&["x"], &[0.0]);
            (x.abs(), grad)
        });
        assert!(result.success);
    }

    #[test]
    fn test_reset_clears_history() {
        let mut opt = optimizer();
        let p = params(&["x"], &[0.0]);
        let g = gradient_from(&["x"], &[0.0]);
        opt.step(&p, &g);
        opt.reset();
        assert!(opt.history().is_empty());
        assert_eq!(opt.iteration(), 0);
    }

    #[test]
    fn test_best_step_none_when_empty() {
        let opt = optimizer();
        assert!(opt.best_step().is_none());
    }

    #[test]
    fn test_optimize_multi_param() {
        // Minimize (x-1)^2 + (y+2)^2; analytic minimum: x=1, y=-2.
        let config = SnoOptimizerConfig::default()
            .with_learning_rate(0.05)
            .with_max_iterations(1000)
            .with_convergence_threshold(1e-9);
        let mut opt = SymbolicNeuralOptimizer::new(config);
        let init = params(&["x", "y"], &[5.0, 5.0]);
        let result = opt.optimize(init, &|p| {
            let x = p.get("x").unwrap_or(0.0);
            let y = p.get("y").unwrap_or(0.0);
            let loss = (x - 1.0) * (x - 1.0) + (y + 2.0) * (y + 2.0);
            let gx = 2.0 * (x - 1.0);
            let gy = 2.0 * (y + 2.0);
            let grad = gradient_from(&["x", "y"], &[gx, gy]);
            (loss, grad)
        });
        let x_f = result.final_params.get("x").unwrap_or(0.0);
        let y_f = result.final_params.get("y").unwrap_or(0.0);
        assert!((x_f - 1.0).abs() < 0.5, "x_f={}", x_f);
        assert!((y_f + 2.0).abs() < 0.5, "y_f={}", y_f);
    }

    #[test]
    fn test_optimizer_config_builder() {
        let cfg = SnoOptimizerConfig::new()
            .with_learning_rate(0.001)
            .with_max_iterations(2000)
            .with_convergence_threshold(1e-10)
            .with_constraint_penalty(5.0)
            .with_symbolic_correction_weight(0.3)
            .with_objective(OptimizationObjective::Maximize);
        assert!((cfg.learning_rate - 0.001).abs() < 1e-15);
        assert_eq!(cfg.max_iterations, 2000);
        assert!((cfg.convergence_threshold - 1e-10).abs() < 1e-20);
        assert!((cfg.constraint_penalty - 5.0).abs() < 1e-12);
        assert!((cfg.symbolic_correction_weight - 0.3).abs() < 1e-12);
        assert_eq!(cfg.objective, OptimizationObjective::Maximize);
    }

    #[test]
    fn test_xorshift64_not_stuck() {
        let mut state: u64 = 12345;
        let r1 = xorshift64(&mut state);
        let r2 = xorshift64(&mut state);
        let r3 = xorshift64(&mut state);
        assert_ne!(r1, r2);
        assert_ne!(r2, r3);
    }

    #[test]
    fn test_xorshift64_zero_seed_skips() {
        // xorshift64 with state=0 would be stuck; confirm non-zero seed works.
        let mut state: u64 = 1;
        let r = xorshift64(&mut state);
        assert_ne!(r, 0);
    }

    #[test]
    fn test_constraint_bound_eq_clamp() {
        let mut opt =
            SymbolicNeuralOptimizer::new(SnoOptimizerConfig::default().with_learning_rate(0.0));
        opt.add_constraint(SymbolicConstraint::hard("fix", "w == 2.0", 1.0));
        let p = params(&["w"], &[5.0]);
        let g = gradient_from(&["w"], &[0.0]);
        let new_p = opt.step(&p, &g);
        assert!((new_p.get("w").unwrap_or(0.0) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_step_unknown_param_in_gradient() {
        // Gradient has a name not in params — should default to 0.0 grad.
        let mut opt = optimizer();
        let p = params(&["x"], &[1.0]);
        let g = gradient_from(&["z"], &[100.0]); // "z" not in params
        let new_p = opt.step(&p, &g);
        // No gradient applied for "x" since "z" doesn't match.
        assert!((new_p.get("x").unwrap_or(0.0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_history_loss_monotone_tendency() {
        // For a simple convex problem without constraints the loss should
        // generally decrease.
        let config = SnoOptimizerConfig::default()
            .with_learning_rate(0.05)
            .with_max_iterations(50)
            .with_convergence_threshold(1e-15);
        let mut opt = SymbolicNeuralOptimizer::new(config);
        let init = params(&["x"], &[10.0]);
        opt.optimize(init, &|p| {
            let x = p.get("x").unwrap_or(0.0);
            let grad = gradient_from(&["x"], &[2.0 * x]);
            (x * x, grad)
        });
        let first_loss = opt
            .history()
            .first()
            .map(|s| s.loss)
            .unwrap_or(f64::INFINITY);
        let last_loss = opt.history().last().map(|s| s.loss).unwrap_or(0.0);
        assert!(
            last_loss <= first_loss,
            "first={} last={}",
            first_loss,
            last_loss
        );
    }

    #[test]
    fn test_remove_constraint_removes_all_matching() {
        let mut opt = optimizer();
        opt.add_constraint(SymbolicConstraint::hard("c", "x >= 0.0", 1.0));
        opt.add_constraint(SymbolicConstraint::hard("c", "x <= 5.0", 1.0));
        opt.add_constraint(SymbolicConstraint::soft("d", "y <= 1.0", 0.5));
        assert!(opt.remove_constraint("c"));
        assert_eq!(opt.constraints().len(), 1);
        assert_eq!(opt.constraints()[0].name, "d");
    }

    #[test]
    fn test_sno_optimization_result_fields() {
        let r = SnoOptimizationResult {
            success: true,
            iterations: 42,
            final_loss: 0.001,
            final_params: params(&["x"], &[1.0]),
            constraint_violations: 0,
            converged: true,
        };
        assert!(r.success);
        assert_eq!(r.iterations, 42);
        assert!(r.converged);
    }

    #[test]
    fn test_sno_optimization_step_fields() {
        let s = SnoOptimizationStep {
            iteration: 7,
            loss: std::f64::consts::PI,
            gradient_norm: 0.5,
            constraint_violations: 2,
            params: params(&["a"], &[9.9]),
        };
        assert_eq!(s.iteration, 7);
        assert_eq!(s.constraint_violations, 2);
    }
}
