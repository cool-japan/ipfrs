//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::HashMap;
use thiserror::Error;

use super::functions::{
    bisector, centroid, consequent_set_name, dominant_set_name, expr_targets_var,
    largest_of_maxima, mean_of_maxima, smallest_of_maxima, universe_bounds,
};

/// Errors produced by [`FuzzyLogicEngine`] and related operations.
#[derive(Debug, Clone, Error)]
pub enum FuzzyError {
    /// A referenced variable does not exist in the configuration.
    #[error("variable not found: {0}")]
    VariableNotFound(String),
    /// A referenced set does not exist inside a variable.
    #[error("set '{set}' not found in variable '{var}'")]
    SetNotFound {
        /// Variable name.
        var: String,
        /// Set name.
        set: String,
    },
    /// All rules produced zero activation; defuzzification is undefined.
    #[error("no rules produced non-zero activation")]
    NoRulesActivated,
    /// Defuzzification failed (e.g., all aggregate membership values are zero).
    #[error("defuzzification failed: {0}")]
    DefuzzFailed(String),
    /// Engine configuration is invalid.
    #[error("configuration error: {0}")]
    ConfigurationError(String),
}
/// Complete configuration for a [`FuzzyLogicEngine`] instance.
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Input linguistic variables.
    pub input_vars: Vec<FuzzyVariable>,
    /// Output linguistic variables.
    pub output_vars: Vec<FuzzyVariable>,
    /// Fuzzy IF-THEN rules.
    pub rules: Vec<FuzzyRule>,
    /// Defuzzification strategy.
    pub defuzz_method: DefuzzMethod,
    /// Number of discretisation points for output universe (default 100).
    pub resolution: usize,
}
impl EngineConfig {
    /// Create a new engine configuration with default resolution (100).
    #[must_use]
    pub fn new(
        input_vars: Vec<FuzzyVariable>,
        output_vars: Vec<FuzzyVariable>,
        rules: Vec<FuzzyRule>,
        defuzz_method: DefuzzMethod,
    ) -> Self {
        Self {
            input_vars,
            output_vars,
            rules,
            defuzz_method,
            resolution: 100,
        }
    }
}
/// A linguistic variable defined over a numeric universe.
#[derive(Debug, Clone)]
pub struct FuzzyVariable {
    /// Variable name, e.g. "temperature".
    pub name: String,
    /// Named fuzzy sets partitioning this variable's universe.
    pub sets: Vec<FuzzySet>,
    /// Current crisp input value (set by [`FuzzyLogicEngine::set_input`]).
    pub current_value: Option<f64>,
}
impl FuzzyVariable {
    /// Create a new fuzzy variable.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            sets: Vec::new(),
            current_value: None,
        }
    }
    /// Look up a set by name.
    #[must_use]
    pub fn get_set(&self, name: &str) -> Option<&FuzzySet> {
        self.sets.iter().find(|s| s.name == name)
    }
    /// Return the membership degree of `x` in the named set, if it exists.
    #[must_use]
    pub fn membership(&self, set_name: &str, x: f64) -> Option<f64> {
        self.get_set(set_name).map(|s| s.degree(x))
    }
}
/// A recursive expression tree for fuzzy antecedents and consequents.
#[derive(Debug, Clone)]
pub enum FuzzyExpr {
    /// Atomic proposition: variable IS set.
    Is {
        /// Variable name.
        var: String,
        /// Set name.
        set: String,
    },
    /// Logical AND (Zadeh min T-norm).
    And(Box<FuzzyExpr>, Box<FuzzyExpr>),
    /// Logical OR (Zadeh max S-norm).
    Or(Box<FuzzyExpr>, Box<FuzzyExpr>),
    /// Logical NOT (standard complement: `1 - x`).
    Not(Box<FuzzyExpr>),
    /// Concentration / "very": squares the membership degree (`x²`).
    Very(Box<FuzzyExpr>),
    /// Dilation / "somewhat": square-root of the membership degree (`√x`).
    Somewhat(Box<FuzzyExpr>),
}
/// Result of running inference for a single output variable.
#[derive(Debug, Clone)]
pub struct InferenceResult {
    /// Name of the output variable this result belongs to.
    pub output_var: String,
    /// Defuzzified crisp output value.
    pub crisp_value: f64,
    /// Activation level of each rule: `(rule_id, activation)`.
    pub activation_map: Vec<(String, f64)>,
    /// Name of the output fuzzy set with the highest post-aggregation membership.
    pub dominant_set: String,
}
/// A membership function mapping a crisp scalar to a degree in \[0, 1\].
#[derive(Debug, Clone, PartialEq)]
pub enum MembershipFunction {
    /// Triangular MF — zero at `a` and `c`, peak 1.0 at `b`.
    Triangle {
        /// Left foot.
        a: f64,
        /// Peak.
        b: f64,
        /// Right foot.
        c: f64,
    },
    /// Trapezoidal MF — slopes on \[a,b\] and \[c,d\], flat top on \[b,c\].
    Trapezoid {
        /// Left foot.
        a: f64,
        /// Left shoulder.
        b: f64,
        /// Right shoulder.
        c: f64,
        /// Right foot.
        d: f64,
    },
    /// Gaussian (bell-shaped) MF: `exp(-0.5 * ((x-mean)/sigma)^2)`.
    Gaussian {
        /// Centre.
        mean: f64,
        /// Standard deviation.
        sigma: f64,
    },
    /// Generalised bell MF: `1 / (1 + |((x-c)/a)|^(2b))`.
    Bell {
        /// Width parameter.
        a: f64,
        /// Slope parameter.
        b: f64,
        /// Centre.
        c: f64,
    },
    /// Sigmoid MF: `1 / (1 + exp(-a*(x-c)))`.
    Sigmoid {
        /// Slope (positive = rising, negative = falling).
        a: f64,
        /// Inflection point.
        c: f64,
    },
    /// Singleton — full membership only at exactly one point.
    Singleton(f64),
    /// Linear interpolation between `(x0,y0)` and `(x1,y1)`; clamped to
    /// \[0,1\] outside the segment.
    Linear {
        /// Left x.
        x0: f64,
        /// Right x.
        x1: f64,
        /// Left y (membership at x0).
        y0: f64,
        /// Right y (membership at x1).
        y1: f64,
    },
}
impl MembershipFunction {
    /// Evaluate the membership degree of `x`.
    #[must_use]
    pub fn evaluate(&self, x: f64) -> f64 {
        match self {
            MembershipFunction::Triangle { a, b, c } => {
                if x <= *a || x >= *c {
                    0.0
                } else if x <= *b {
                    let denom = b - a;
                    if denom.abs() < f64::EPSILON {
                        1.0
                    } else {
                        (x - a) / denom
                    }
                } else {
                    let denom = c - b;
                    if denom.abs() < f64::EPSILON {
                        1.0
                    } else {
                        (c - x) / denom
                    }
                }
            }
            MembershipFunction::Trapezoid { a, b, c, d } => {
                if x <= *a || x >= *d {
                    0.0
                } else if x <= *b {
                    let denom = b - a;
                    if denom.abs() < f64::EPSILON {
                        1.0
                    } else {
                        (x - a) / denom
                    }
                } else if x <= *c {
                    1.0
                } else {
                    let denom = d - c;
                    if denom.abs() < f64::EPSILON {
                        1.0
                    } else {
                        (d - x) / denom
                    }
                }
            }
            MembershipFunction::Gaussian { mean, sigma } => {
                if sigma.abs() < f64::EPSILON {
                    if (x - mean).abs() < 1e-10 {
                        1.0
                    } else {
                        0.0
                    }
                } else {
                    let z = (x - mean) / sigma;
                    (-0.5 * z * z).exp()
                }
            }
            MembershipFunction::Bell { a, b, c } => {
                if a.abs() < f64::EPSILON {
                    if (x - c).abs() < 1e-10 {
                        1.0
                    } else {
                        0.0
                    }
                } else {
                    let base = ((x - c) / a).abs();
                    let exp = 2.0 * b;
                    1.0 / (1.0 + base.powf(exp))
                }
            }
            MembershipFunction::Sigmoid { a, c } => 1.0 / (1.0 + (-a * (x - c)).exp()),
            MembershipFunction::Singleton(v) => {
                if (x - v).abs() < 1e-10 {
                    1.0
                } else {
                    0.0
                }
            }
            MembershipFunction::Linear { x0, x1, y0, y1 } => {
                let dx = x1 - x0;
                if dx.abs() < f64::EPSILON {
                    return ((*y0 + *y1) / 2.0).clamp(0.0, 1.0);
                }
                let t = (x - x0) / dx;
                (y0 + t * (y1 - y0)).clamp(0.0, 1.0)
            }
        }
    }
}
/// A named fuzzy set with an associated membership function and universe bounds.
#[derive(Debug, Clone)]
pub struct FuzzySet {
    /// Human-readable name, e.g. "low", "medium", "high".
    pub name: String,
    /// Membership function defining the set.
    pub mf: MembershipFunction,
    /// Lower bound of the universe of discourse.
    pub universe_min: f64,
    /// Upper bound of the universe of discourse.
    pub universe_max: f64,
}
impl FuzzySet {
    /// Create a new fuzzy set.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        mf: MembershipFunction,
        universe_min: f64,
        universe_max: f64,
    ) -> Self {
        Self {
            name: name.into(),
            mf,
            universe_min,
            universe_max,
        }
    }
    /// Return the membership degree of `x` in this set.
    #[must_use]
    pub fn degree(&self, x: f64) -> f64 {
        self.mf.evaluate(x)
    }
}
/// Defuzzification method used to convert the aggregated output MF to a crisp value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DefuzzMethod {
    /// Centre-of-gravity (centroid) numerical integration.
    #[default]
    Centroid,
    /// Bisector — point dividing the area under the aggregated MF in half.
    Bisector,
    /// Mean of all x-values where the aggregated MF reaches its maximum.
    MeanOfMaxima,
    /// Largest (rightmost) x where the aggregated MF reaches its maximum.
    LargestOfMaxima,
    /// Smallest (leftmost) x where the aggregated MF reaches its maximum.
    SmallestOfMaxima,
}
/// A weighted IF-THEN fuzzy rule.
#[derive(Debug, Clone)]
pub struct FuzzyRule {
    /// Unique rule identifier.
    pub id: String,
    /// Antecedent expression (the IF part).
    pub antecedent: FuzzyExpr,
    /// Consequent expression (the THEN part; typically `FuzzyExpr::Is { .. }`).
    pub consequent: FuzzyExpr,
    /// Importance weight in \[0, 1\].
    pub weight: f64,
}
impl FuzzyRule {
    /// Create a new fuzzy rule.
    #[must_use]
    pub fn new(
        id: impl Into<String>,
        antecedent: FuzzyExpr,
        consequent: FuzzyExpr,
        weight: f64,
    ) -> Self {
        Self {
            id: id.into(),
            antecedent,
            consequent,
            weight,
        }
    }
}
/// Cumulative runtime statistics for a [`FuzzyLogicEngine`].
#[derive(Debug, Clone, Default)]
pub struct EngineStats {
    /// Total number of rules evaluated across all [`infer`](FuzzyLogicEngine::infer) calls.
    pub rules_evaluated: u64,
    /// Average rule activation across all evaluations.
    pub avg_activation: f64,
    /// Number of times defuzzification has been called.
    pub defuzz_calls: u64,
}
/// Full Mamdani fuzzy inference engine.
///
/// # Workflow
///
/// 1. Build an [`EngineConfig`] describing inputs, outputs, and rules.
/// 2. Create the engine with [`FuzzyLogicEngine::new`].
/// 3. Set crisp input values via [`set_input`](FuzzyLogicEngine::set_input).
/// 4. Call [`infer`](FuzzyLogicEngine::infer) or
///    [`infer_single`](FuzzyLogicEngine::infer_single).
pub struct FuzzyLogicEngine {
    /// Input variables (name → variable).
    pub(super) input_vars: HashMap<String, FuzzyVariable>,
    /// Output variables (name → variable).
    pub(super) output_vars: HashMap<String, FuzzyVariable>,
    /// Active rule set.
    pub(super) rules: Vec<FuzzyRule>,
    /// Defuzzification method.
    pub(super) defuzz_method: DefuzzMethod,
    /// Discretisation resolution.
    pub(super) resolution: usize,
    /// Runtime statistics.
    pub(super) stats: EngineStats,
    /// Running sum of activations for average calculation.
    pub(super) activation_sum: f64,
    /// Total individual rule evaluations for average.
    pub(super) activation_count: u64,
}
impl FuzzyLogicEngine {
    /// Create a new engine, validating that all rule references resolve.
    ///
    /// # Errors
    /// Returns [`FuzzyError::ConfigurationError`] if `resolution` is zero.
    /// Returns [`FuzzyError::VariableNotFound`] / [`FuzzyError::SetNotFound`]
    /// if any rule references a non-existent variable or set.
    pub fn new(config: EngineConfig) -> Result<Self, FuzzyError> {
        if config.resolution == 0 {
            return Err(FuzzyError::ConfigurationError(
                "resolution must be >= 1".to_string(),
            ));
        }
        let input_vars: HashMap<String, FuzzyVariable> = config
            .input_vars
            .into_iter()
            .map(|v| (v.name.clone(), v))
            .collect();
        let output_vars: HashMap<String, FuzzyVariable> = config
            .output_vars
            .into_iter()
            .map(|v| (v.name.clone(), v))
            .collect();
        for rule in &config.rules {
            Self::validate_expr_static(&rule.antecedent, &input_vars, &output_vars)?;
            Self::validate_expr_static(&rule.consequent, &input_vars, &output_vars)?;
        }
        Ok(Self {
            input_vars,
            output_vars,
            rules: config.rules,
            defuzz_method: config.defuzz_method,
            resolution: config.resolution,
            stats: EngineStats::default(),
            activation_sum: 0.0,
            activation_count: 0,
        })
    }
    /// Set the current crisp value for a named input variable.
    ///
    /// # Errors
    /// Returns [`FuzzyError::VariableNotFound`] if the variable is not registered.
    pub fn set_input(&mut self, var_name: &str, value: f64) -> Result<(), FuzzyError> {
        let var = self
            .input_vars
            .get_mut(var_name)
            .ok_or_else(|| FuzzyError::VariableNotFound(var_name.to_string()))?;
        var.current_value = Some(value);
        Ok(())
    }
    /// Return the current membership degree of the current value of `var_name`
    /// in `set_name`.
    ///
    /// # Errors
    /// Returns [`FuzzyError::VariableNotFound`] if `var_name` does not exist.
    /// Returns [`FuzzyError::SetNotFound`] if `set_name` does not exist.
    pub fn membership(&self, var_name: &str, set_name: &str) -> Result<f64, FuzzyError> {
        let var = self
            .input_vars
            .get(var_name)
            .or_else(|| self.output_vars.get(var_name))
            .ok_or_else(|| FuzzyError::VariableNotFound(var_name.to_string()))?;
        let x = var.current_value.unwrap_or(0.0);
        var.membership(set_name, x)
            .ok_or_else(|| FuzzyError::SetNotFound {
                var: var_name.to_string(),
                set: set_name.to_string(),
            })
    }
    /// Append a rule after validating all references.
    ///
    /// # Errors
    /// Returns [`FuzzyError::VariableNotFound`] / [`FuzzyError::SetNotFound`]
    /// if any expression references a non-existent variable or set.
    pub fn add_rule(&mut self, rule: FuzzyRule) -> Result<(), FuzzyError> {
        Self::validate_expr_static(&rule.antecedent, &self.input_vars, &self.output_vars)?;
        Self::validate_expr_static(&rule.consequent, &self.input_vars, &self.output_vars)?;
        self.rules.push(rule);
        Ok(())
    }
    /// Run Mamdani inference for **all** output variables.
    ///
    /// # Errors
    /// Returns [`FuzzyError::NoRulesActivated`] if every rule fires at zero.
    pub fn infer(&mut self) -> Result<Vec<InferenceResult>, FuzzyError> {
        let output_names: Vec<String> = self.output_vars.keys().cloned().collect();
        let mut results = Vec::with_capacity(output_names.len());
        for name in output_names {
            results.push(self.infer_single(&name)?);
        }
        Ok(results)
    }
    /// Run Mamdani inference for a single named output variable.
    ///
    /// # Errors
    /// Returns [`FuzzyError::VariableNotFound`] if `output_var` is not registered.
    /// Returns [`FuzzyError::NoRulesActivated`] if no rule fires with non-zero activation.
    /// Returns [`FuzzyError::DefuzzFailed`] if defuzzification cannot produce a value.
    pub fn infer_single(&mut self, output_var: &str) -> Result<InferenceResult, FuzzyError> {
        if !self.output_vars.contains_key(output_var) {
            return Err(FuzzyError::VariableNotFound(output_var.to_string()));
        }
        let rule_data: Vec<(String, Option<String>, f64)> = {
            let mut out = Vec::new();
            for rule in &self.rules {
                if !expr_targets_var(&rule.consequent, output_var) {
                    continue;
                }
                let alpha = self.eval_expr(&rule.antecedent)?;
                let weighted = (alpha * rule.weight).clamp(0.0, 1.0);
                let cset = consequent_set_name(&rule.consequent, output_var);
                out.push((rule.id.clone(), cset, weighted));
            }
            out
        };
        let any_active = rule_data.iter().any(|(_, _, a)| *a > 0.0);
        if !any_active {
            return Err(FuzzyError::NoRulesActivated);
        }
        let activation_map: Vec<(String, f64)> = rule_data
            .iter()
            .map(|(id, _, a)| (id.clone(), *a))
            .collect();
        let (u_min, u_max, resolution) = {
            let out_var = self
                .output_vars
                .get(output_var)
                .ok_or_else(|| FuzzyError::VariableNotFound(output_var.to_string()))?;
            let (mn, mx) = universe_bounds(out_var);
            (mn, mx, self.resolution)
        };
        let step = (u_max - u_min) / (resolution - 1).max(1) as f64;
        let mut agg: Vec<f64> = vec![0.0; resolution];
        for (_, cset_opt, alpha) in &rule_data {
            if *alpha <= 0.0 {
                continue;
            }
            if let Some(set_name) = cset_opt {
                let mf_clone = {
                    let out_var_ref = self
                        .output_vars
                        .get(output_var)
                        .ok_or_else(|| FuzzyError::VariableNotFound(output_var.to_string()))?;
                    let fset =
                        out_var_ref
                            .get_set(set_name)
                            .ok_or_else(|| FuzzyError::SetNotFound {
                                var: output_var.to_string(),
                                set: set_name.clone(),
                            })?;
                    fset.mf.clone()
                };
                for (i, agg_val) in agg.iter_mut().enumerate() {
                    let x = u_min + i as f64 * step;
                    let clipped = mf_clone.evaluate(x).min(*alpha);
                    *agg_val = agg_val.max(clipped);
                }
            }
        }
        let crisp_value = self.defuzzify(&agg, u_min, step)?;
        let dominant_set = {
            let out_var_ref = self
                .output_vars
                .get(output_var)
                .ok_or_else(|| FuzzyError::VariableNotFound(output_var.to_string()))?;
            dominant_set_name(out_var_ref, crisp_value)
        };
        let n_activated = rule_data.iter().filter(|(_, _, a)| *a > 0.0).count() as u64;
        self.stats.rules_evaluated += rule_data.len() as u64;
        self.stats.defuzz_calls += 1;
        let total_act: f64 = rule_data.iter().map(|(_, _, a)| a).sum();
        self.activation_sum += total_act;
        self.activation_count += n_activated.max(1);
        self.stats.avg_activation = if self.activation_count > 0 {
            self.activation_sum / self.activation_count as f64
        } else {
            0.0
        };
        Ok(InferenceResult {
            output_var: output_var.to_string(),
            crisp_value,
            activation_map,
            dominant_set,
        })
    }
    /// Evaluate a single rule's antecedent activation (before weighting).
    ///
    /// # Errors
    /// Returns [`FuzzyError::VariableNotFound`] / [`FuzzyError::SetNotFound`]
    /// if any expression reference is invalid.
    pub fn evaluate_rule(&self, rule: &FuzzyRule) -> Result<f64, FuzzyError> {
        let alpha = self.eval_expr(&rule.antecedent)?;
        Ok((alpha * rule.weight).clamp(0.0, 1.0))
    }
    /// Return a snapshot of the current runtime statistics.
    #[must_use]
    pub fn stats(&self) -> EngineStats {
        self.stats.clone()
    }
    /// Recursively evaluate a `FuzzyExpr` using the current input values.
    pub(super) fn eval_expr(&self, expr: &FuzzyExpr) -> Result<f64, FuzzyError> {
        match expr {
            FuzzyExpr::Is { var, set } => {
                let variable = self
                    .input_vars
                    .get(var.as_str())
                    .ok_or_else(|| FuzzyError::VariableNotFound(var.clone()))?;
                let x = variable.current_value.unwrap_or(0.0);
                variable
                    .membership(set, x)
                    .ok_or_else(|| FuzzyError::SetNotFound {
                        var: var.clone(),
                        set: set.clone(),
                    })
            }
            FuzzyExpr::And(l, r) => Ok(self.eval_expr(l)?.min(self.eval_expr(r)?)),
            FuzzyExpr::Or(l, r) => Ok(self.eval_expr(l)?.max(self.eval_expr(r)?)),
            FuzzyExpr::Not(inner) => Ok(1.0 - self.eval_expr(inner)?),
            FuzzyExpr::Very(inner) => {
                let v = self.eval_expr(inner)?;
                Ok(v * v)
            }
            FuzzyExpr::Somewhat(inner) => Ok(self.eval_expr(inner)?.sqrt()),
        }
    }
    /// Perform defuzzification on the discretised aggregate array.
    pub(super) fn defuzzify(&self, agg: &[f64], u_min: f64, step: f64) -> Result<f64, FuzzyError> {
        match self.defuzz_method {
            DefuzzMethod::Centroid => centroid(agg, u_min, step),
            DefuzzMethod::Bisector => bisector(agg, u_min, step),
            DefuzzMethod::MeanOfMaxima => mean_of_maxima(agg, u_min, step),
            DefuzzMethod::LargestOfMaxima => largest_of_maxima(agg, u_min, step),
            DefuzzMethod::SmallestOfMaxima => smallest_of_maxima(agg, u_min, step),
        }
    }
    /// Static expression validator used during construction and `add_rule`.
    pub(super) fn validate_expr_static(
        expr: &FuzzyExpr,
        input_vars: &HashMap<String, FuzzyVariable>,
        output_vars: &HashMap<String, FuzzyVariable>,
    ) -> Result<(), FuzzyError> {
        match expr {
            FuzzyExpr::Is { var, set } => {
                let variable = input_vars
                    .get(var.as_str())
                    .or_else(|| output_vars.get(var.as_str()))
                    .ok_or_else(|| FuzzyError::VariableNotFound(var.clone()))?;
                if variable.get_set(set).is_none() {
                    return Err(FuzzyError::SetNotFound {
                        var: var.clone(),
                        set: set.clone(),
                    });
                }
                Ok(())
            }
            FuzzyExpr::And(l, r) | FuzzyExpr::Or(l, r) => {
                Self::validate_expr_static(l, input_vars, output_vars)?;
                Self::validate_expr_static(r, input_vars, output_vars)
            }
            FuzzyExpr::Not(inner) | FuzzyExpr::Very(inner) | FuzzyExpr::Somewhat(inner) => {
                Self::validate_expr_static(inner, input_vars, output_vars)
            }
        }
    }
}
