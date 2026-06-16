//! Fuzzy Logic Engine for approximate reasoning.
//!
//! Provides membership functions, fuzzy variables, fuzzy rules, inference
//! (Mamdani / Sugeno), and defuzzification (Centroid, MeanOfMax, LargestOfMax).

use std::collections::HashMap;
use thiserror::Error;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors produced by [`FuzzyLogicEngine`] and related types.
#[derive(Debug, Clone, Error)]
pub enum FuzzyError {
    /// A referenced variable does not exist in the engine.
    #[error("fuzzy variable not found: {0}")]
    VariableNotFound(String),

    /// A referenced set does not exist inside a variable.
    #[error("fuzzy set '{set}' not found in variable '{variable}'")]
    SetNotFound { variable: String, set: String },

    /// Attempt to add a variable that is already registered.
    #[error("duplicate fuzzy variable: {0}")]
    DuplicateVariable(String),

    /// `infer` was called but no rule targets the requested output variable.
    #[error("no fuzzy rules target output variable '{0}'")]
    NoRulesForOutput(String),

    /// A numerical issue prevented completing the computation (e.g. division by
    /// zero in defuzzification when all membership degrees are zero).
    #[error("numerical error in fuzzy inference: {0}")]
    NumericalError(String),
}

// ---------------------------------------------------------------------------
// Membership functions
// ---------------------------------------------------------------------------

/// A membership function that maps a crisp value `x` to a degree in `[0, 1]`.
#[derive(Debug, Clone, PartialEq)]
pub enum MembershipFunction {
    /// Triangular MF — zero at `a` and `c`, peak 1.0 at `b`.
    ///
    /// `μ(x) = max(0, min((x-a)/(b-a), (c-x)/(c-b)))`
    Triangular { a: f64, b: f64, c: f64 },

    /// Trapezoidal MF — zero below `a` and above `d`, full membership on `[b,c]`.
    ///
    /// `μ(x) = max(0, min((x-a)/(b-a), 1.0, (d-x)/(d-c)))`
    Trapezoidal { a: f64, b: f64, c: f64, d: f64 },

    /// Gaussian MF — bell-shaped curve.
    ///
    /// `μ(x) = exp(-0.5 * ((x - mean) / sigma)^2)`
    Gaussian { mean: f64, sigma: f64 },

    /// Singleton MF — full membership at exactly one point, zero elsewhere.
    Singleton { value: f64 },

    /// Universe of discourse — full membership everywhere.
    Universe,
}

impl MembershipFunction {
    /// Evaluate the membership degree of `x`.
    #[must_use]
    pub fn evaluate(&self, x: f64) -> f64 {
        match self {
            MembershipFunction::Triangular { a, b, c } => {
                let left = if (b - a).abs() < f64::EPSILON {
                    if (x - a).abs() < f64::EPSILON {
                        1.0
                    } else {
                        0.0
                    }
                } else {
                    (x - a) / (b - a)
                };
                let right = if (c - b).abs() < f64::EPSILON {
                    if (x - b).abs() < f64::EPSILON {
                        1.0
                    } else {
                        0.0
                    }
                } else {
                    (c - x) / (c - b)
                };
                left.min(right).max(0.0)
            }
            MembershipFunction::Trapezoidal { a, b, c, d } => {
                let left = if (b - a).abs() < f64::EPSILON {
                    if x >= *a {
                        1.0
                    } else {
                        0.0
                    }
                } else {
                    (x - a) / (b - a)
                };
                let right = if (d - c).abs() < f64::EPSILON {
                    if x <= *d {
                        1.0
                    } else {
                        0.0
                    }
                } else {
                    (d - x) / (d - c)
                };
                left.min(1.0_f64).min(right).max(0.0)
            }
            MembershipFunction::Gaussian { mean, sigma } => {
                if sigma.abs() < f64::EPSILON {
                    if (x - mean).abs() < f64::EPSILON {
                        1.0
                    } else {
                        0.0
                    }
                } else {
                    let z = (x - mean) / sigma;
                    (-0.5 * z * z).exp()
                }
            }
            MembershipFunction::Singleton { value } => {
                if (x - value).abs() < f64::EPSILON {
                    1.0
                } else {
                    0.0
                }
            }
            MembershipFunction::Universe => 1.0,
        }
    }
}

// ---------------------------------------------------------------------------
// FuzzySet
// ---------------------------------------------------------------------------

/// A named fuzzy set with an associated membership function.
#[derive(Debug, Clone)]
pub struct FuzzySet {
    /// Human-readable name (e.g. "low", "medium", "high").
    pub name: String,
    /// Membership function that defines the set.
    pub mf: MembershipFunction,
}

impl FuzzySet {
    /// Create a new fuzzy set.
    #[must_use]
    pub fn new(name: impl Into<String>, mf: MembershipFunction) -> Self {
        Self {
            name: name.into(),
            mf,
        }
    }

    /// Return the membership degree of `x` in this set.
    #[must_use]
    pub fn degree(&self, x: f64) -> f64 {
        self.mf.evaluate(x)
    }
}

// ---------------------------------------------------------------------------
// FuzzyVariable
// ---------------------------------------------------------------------------

/// A linguistic variable defined over a numeric range `[min_val, max_val]`.
#[derive(Debug, Clone)]
pub struct FuzzyVariable {
    /// Name of the variable (e.g. "temperature").
    pub name: String,
    /// The named fuzzy sets that partition this variable's universe.
    pub sets: Vec<FuzzySet>,
    /// Lower bound of the universe of discourse.
    pub min_val: f64,
    /// Upper bound of the universe of discourse.
    pub max_val: f64,
}

impl FuzzyVariable {
    /// Create a new fuzzy variable.
    #[must_use]
    pub fn new(name: impl Into<String>, min_val: f64, max_val: f64) -> Self {
        Self {
            name: name.into(),
            sets: Vec::new(),
            min_val,
            max_val,
        }
    }

    /// Add a fuzzy set to this variable.
    pub fn add_set(&mut self, set: FuzzySet) {
        self.sets.push(set);
    }

    /// Look up a set by name.
    #[must_use]
    pub fn get_set(&self, name: &str) -> Option<&FuzzySet> {
        self.sets.iter().find(|s| s.name == name)
    }
}

// ---------------------------------------------------------------------------
// FuzzyProposition
// ---------------------------------------------------------------------------

/// An atomic fuzzy proposition: "*variable* IS *set*".
#[derive(Debug, Clone)]
pub struct FuzzyProposition {
    /// Name of the linguistic variable.
    pub variable: String,
    /// Name of the fuzzy set.
    pub set: String,
}

impl FuzzyProposition {
    /// Create a new proposition.
    #[must_use]
    pub fn new(variable: impl Into<String>, set: impl Into<String>) -> Self {
        Self {
            variable: variable.into(),
            set: set.into(),
        }
    }
}

// ---------------------------------------------------------------------------
// FuzzyRule
// ---------------------------------------------------------------------------

/// A fuzzy IF-THEN rule.
///
/// Multiple antecedents are combined with AND (minimum T-norm).
/// The overall activation is multiplied by `weight` before being applied
/// to the consequent.
#[derive(Debug, Clone)]
pub struct FuzzyRule {
    /// Antecedent propositions (all combined with AND / minimum).
    pub antecedents: Vec<FuzzyProposition>,
    /// The consequent proposition (output linguistic variable and set).
    pub consequent: FuzzyProposition,
    /// Rule importance weight in `[0, 1]`.
    pub weight: f64,
}

impl FuzzyRule {
    /// Convenience constructor.
    #[must_use]
    pub fn new(
        antecedents: Vec<FuzzyProposition>,
        consequent: FuzzyProposition,
        weight: f64,
    ) -> Self {
        Self {
            antecedents,
            consequent,
            weight,
        }
    }
}

// ---------------------------------------------------------------------------
// Inference and defuzzification method enums
// ---------------------------------------------------------------------------

/// Fuzzy inference strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InferenceMethod {
    /// Mamdani min-of-max inference with aggregation and defuzzification.
    Mamdani,
    /// Takagi–Sugeno–Kang (TSK / Sugeno) weighted centroid aggregation.
    Sugeno,
}

/// Defuzzification method (used with Mamdani inference).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DefuzzMethod {
    /// Centre-of-gravity (centroid) numerical integration.
    Centroid,
    /// Mean of the x-values where the aggregate membership is maximal.
    MeanOfMax,
    /// Rightmost x where the aggregate membership is maximal.
    LargestOfMax,
}

// ---------------------------------------------------------------------------
// FuzzyStats
// ---------------------------------------------------------------------------

/// Snapshot of engine configuration statistics.
#[derive(Debug, Clone)]
pub struct FuzzyStats {
    /// Number of registered linguistic variables.
    pub variable_count: usize,
    /// Number of registered rules.
    pub rule_count: usize,
    /// Name of the inference method in use.
    pub inference: String,
    /// Name of the defuzzification method in use.
    pub defuzz: String,
}

// ---------------------------------------------------------------------------
// FuzzyLogicEngine
// ---------------------------------------------------------------------------

/// Number of integration steps used in centroid defuzzification.
const CENTROID_STEPS: usize = 100;

/// A complete fuzzy logic inference system.
///
/// # Usage
///
/// 1. Create an engine with [`FuzzyLogicEngine::new`].
/// 2. Register linguistic variables with [`FuzzyLogicEngine::add_variable`].
/// 3. Add IF-THEN rules with [`FuzzyLogicEngine::add_rule`].
/// 4. Call [`FuzzyLogicEngine::infer`] or [`FuzzyLogicEngine::evaluate`] to
///    obtain a crisp output for a given set of crisp inputs.
pub struct FuzzyLogicEngine {
    variables: HashMap<String, FuzzyVariable>,
    rules: Vec<FuzzyRule>,
    inference: InferenceMethod,
    defuzz: DefuzzMethod,
}

impl FuzzyLogicEngine {
    // ------------------------------------------------------------------
    // Construction
    // ------------------------------------------------------------------

    /// Create a new engine with the given inference and defuzzification strategies.
    #[must_use]
    pub fn new(inference: InferenceMethod, defuzz: DefuzzMethod) -> Self {
        Self {
            variables: HashMap::new(),
            rules: Vec::new(),
            inference,
            defuzz,
        }
    }

    // ------------------------------------------------------------------
    // Variable and rule registration
    // ------------------------------------------------------------------

    /// Register a linguistic variable.
    ///
    /// # Errors
    /// Returns [`FuzzyError::DuplicateVariable`] if a variable with the same
    /// name already exists.
    pub fn add_variable(&mut self, var: FuzzyVariable) -> Result<(), FuzzyError> {
        if self.variables.contains_key(&var.name) {
            return Err(FuzzyError::DuplicateVariable(var.name.clone()));
        }
        self.variables.insert(var.name.clone(), var);
        Ok(())
    }

    /// Register a fuzzy rule.
    ///
    /// Validates that every variable and set referenced by the rule exists.
    ///
    /// # Errors
    /// Returns [`FuzzyError::VariableNotFound`] or [`FuzzyError::SetNotFound`]
    /// if any reference is invalid.
    pub fn add_rule(&mut self, rule: FuzzyRule) -> Result<(), FuzzyError> {
        // Validate antecedents
        for prop in &rule.antecedents {
            self.validate_proposition(prop)?;
        }
        // Validate consequent
        self.validate_proposition(&rule.consequent)?;
        self.rules.push(rule);
        Ok(())
    }

    // ------------------------------------------------------------------
    // Query helpers
    // ------------------------------------------------------------------

    /// Return the number of registered rules.
    #[must_use]
    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }

    /// Return the names of all registered variables.
    #[must_use]
    pub fn variable_names(&self) -> Vec<&str> {
        self.variables.keys().map(String::as_str).collect()
    }

    /// Return the membership degree of `x` in a specific set of a variable.
    ///
    /// # Errors
    /// Returns [`FuzzyError::VariableNotFound`] or [`FuzzyError::SetNotFound`].
    pub fn membership_at(&self, variable: &str, set: &str, x: f64) -> Result<f64, FuzzyError> {
        let var = self.get_variable(variable)?;
        let fuzzy_set = var.get_set(set).ok_or_else(|| FuzzyError::SetNotFound {
            variable: variable.to_string(),
            set: set.to_string(),
        })?;
        Ok(fuzzy_set.degree(x))
    }

    /// Fuzzify a crisp value for a named variable.
    ///
    /// Returns a vector of `(set_name, degree)` pairs for every set defined
    /// on the variable.
    ///
    /// # Errors
    /// Returns [`FuzzyError::VariableNotFound`] if the variable is unknown.
    pub fn fuzzify(&self, variable: &str, crisp: f64) -> Result<Vec<(String, f64)>, FuzzyError> {
        let var = self.get_variable(variable)?;
        Ok(var
            .sets
            .iter()
            .map(|s| (s.name.clone(), s.degree(crisp)))
            .collect())
    }

    /// Evaluate the antecedent of `rule` given a map of crisp input values.
    ///
    /// Returns the weighted activation degree `α * rule.weight`.
    ///
    /// # Errors
    /// Returns [`FuzzyError::VariableNotFound`] or [`FuzzyError::SetNotFound`].
    pub fn fire_rule(
        &self,
        rule: &FuzzyRule,
        inputs: &HashMap<String, f64>,
    ) -> Result<f64, FuzzyError> {
        let mut activation: f64 = 1.0;
        for prop in &rule.antecedents {
            let crisp = inputs.get(&prop.variable).copied().unwrap_or(0.0);
            let degree = self.membership_at(&prop.variable, &prop.set, crisp)?;
            activation = activation.min(degree);
        }
        Ok(activation * rule.weight)
    }

    /// Run the complete fuzzy inference pipeline and return a crisp output
    /// for `output_var`.
    ///
    /// # Errors
    /// * [`FuzzyError::VariableNotFound`] — `output_var` or any input variable is missing.
    /// * [`FuzzyError::SetNotFound`] — a rule references an undefined set.
    /// * [`FuzzyError::NoRulesForOutput`] — no rules target `output_var`.
    /// * [`FuzzyError::NumericalError`] — all aggregate membership values are zero.
    pub fn infer(
        &self,
        inputs: &HashMap<String, f64>,
        output_var: &str,
    ) -> Result<f64, FuzzyError> {
        // Collect rules that address the requested output variable
        let relevant: Vec<(&FuzzyRule, f64)> = self
            .rules
            .iter()
            .filter(|r| r.consequent.variable == output_var)
            .map(|r| {
                let alpha = self.fire_rule(r, inputs)?;
                Ok((r, alpha))
            })
            .collect::<Result<Vec<_>, FuzzyError>>()?;

        if relevant.is_empty() {
            return Err(FuzzyError::NoRulesForOutput(output_var.to_string()));
        }

        match self.inference {
            InferenceMethod::Mamdani => self.infer_mamdani(output_var, &relevant),
            InferenceMethod::Sugeno => self.infer_sugeno(output_var, &relevant),
        }
    }

    /// Alias for [`infer`](FuzzyLogicEngine::infer); accepts `&mut self` for
    /// future stateful extensions.
    ///
    /// # Errors
    /// Same as [`infer`](FuzzyLogicEngine::infer).
    pub fn evaluate(
        &mut self,
        inputs: &HashMap<String, f64>,
        output_var: &str,
    ) -> Result<f64, FuzzyError> {
        self.infer(inputs, output_var)
    }

    /// Return a snapshot of engine statistics.
    #[must_use]
    pub fn stats(&self) -> FuzzyStats {
        FuzzyStats {
            variable_count: self.variables.len(),
            rule_count: self.rules.len(),
            inference: format!("{:?}", self.inference),
            defuzz: format!("{:?}", self.defuzz),
        }
    }

    // ------------------------------------------------------------------
    // Private helpers
    // ------------------------------------------------------------------

    fn get_variable(&self, name: &str) -> Result<&FuzzyVariable, FuzzyError> {
        self.variables
            .get(name)
            .ok_or_else(|| FuzzyError::VariableNotFound(name.to_string()))
    }

    fn validate_proposition(&self, prop: &FuzzyProposition) -> Result<(), FuzzyError> {
        let var = self.get_variable(&prop.variable)?;
        if var.get_set(&prop.set).is_none() {
            return Err(FuzzyError::SetNotFound {
                variable: prop.variable.clone(),
                set: prop.set.clone(),
            });
        }
        Ok(())
    }

    /// Mamdani inference: clip each consequent set at its activation α,
    /// aggregate with pointwise maximum, then defuzzify.
    fn infer_mamdani(
        &self,
        output_var: &str,
        relevant: &[(&FuzzyRule, f64)],
    ) -> Result<f64, FuzzyError> {
        let out_var = self.get_variable(output_var)?;
        let steps = CENTROID_STEPS;
        let range = out_var.max_val - out_var.min_val;
        let step_size = range / steps as f64;

        // Build the aggregate membership function via pointwise max of clipped sets.
        // We represent it as a sampled array over `steps` uniformly spaced points.
        let x_values: Vec<f64> = (0..=steps)
            .map(|i| out_var.min_val + i as f64 * step_size)
            .collect();

        let mut aggregate: Vec<f64> = vec![0.0; x_values.len()];
        for (rule, alpha) in relevant {
            let set =
                out_var
                    .get_set(&rule.consequent.set)
                    .ok_or_else(|| FuzzyError::SetNotFound {
                        variable: output_var.to_string(),
                        set: rule.consequent.set.clone(),
                    })?;
            for (i, &x) in x_values.iter().enumerate() {
                let mu = set.degree(x).min(*alpha);
                if mu > aggregate[i] {
                    aggregate[i] = mu;
                }
            }
        }

        self.defuzzify(&x_values, &aggregate)
    }

    /// Sugeno inference: weighted sum of set centroids divided by total weight.
    fn infer_sugeno(
        &self,
        output_var: &str,
        relevant: &[(&FuzzyRule, f64)],
    ) -> Result<f64, FuzzyError> {
        let out_var = self.get_variable(output_var)?;
        let steps = CENTROID_STEPS;
        let range = out_var.max_val - out_var.min_val;
        let step_size = range / steps as f64;

        let x_values: Vec<f64> = (0..=steps)
            .map(|i| out_var.min_val + i as f64 * step_size)
            .collect();

        let mut weighted_sum: f64 = 0.0;
        let mut weight_total: f64 = 0.0;

        for (rule, alpha) in relevant {
            if *alpha <= 0.0 {
                continue;
            }
            let set =
                out_var
                    .get_set(&rule.consequent.set)
                    .ok_or_else(|| FuzzyError::SetNotFound {
                        variable: output_var.to_string(),
                        set: rule.consequent.set.clone(),
                    })?;
            // Compute centroid of this set over the universe
            let centroid = centroid_of_set(set, &x_values);
            weighted_sum += alpha * centroid;
            weight_total += alpha;
        }

        if weight_total < f64::EPSILON {
            return Err(FuzzyError::NumericalError(
                "Sugeno total activation weight is zero".to_string(),
            ));
        }
        Ok(weighted_sum / weight_total)
    }

    /// Apply the configured defuzzification method to a sampled aggregate MF.
    fn defuzzify(&self, x: &[f64], mu: &[f64]) -> Result<f64, FuzzyError> {
        match self.defuzz {
            DefuzzMethod::Centroid => defuzz_centroid(x, mu),
            DefuzzMethod::MeanOfMax => defuzz_mean_of_max(x, mu),
            DefuzzMethod::LargestOfMax => defuzz_largest_of_max(x, mu),
        }
    }
}

// ---------------------------------------------------------------------------
// Stand-alone defuzzification helpers
// ---------------------------------------------------------------------------

/// Centre-of-gravity defuzzification.
fn defuzz_centroid(x: &[f64], mu: &[f64]) -> Result<f64, FuzzyError> {
    let sum_xmu: f64 = x.iter().zip(mu.iter()).map(|(xi, mi)| xi * mi).sum();
    let sum_mu: f64 = mu.iter().sum();
    if sum_mu < f64::EPSILON {
        return Err(FuzzyError::NumericalError(
            "centroid defuzzification: aggregate membership is all-zero".to_string(),
        ));
    }
    Ok(sum_xmu / sum_mu)
}

/// Mean-of-maximum defuzzification — average of all x where μ is maximum.
fn defuzz_mean_of_max(x: &[f64], mu: &[f64]) -> Result<f64, FuzzyError> {
    let max_mu = mu.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if max_mu < f64::EPSILON {
        return Err(FuzzyError::NumericalError(
            "mean-of-max defuzzification: aggregate membership is all-zero".to_string(),
        ));
    }
    let max_x: Vec<f64> = x
        .iter()
        .zip(mu.iter())
        .filter(|(_, &m)| (m - max_mu).abs() < 1e-10)
        .map(|(&xi, _)| xi)
        .collect();
    if max_x.is_empty() {
        return Err(FuzzyError::NumericalError(
            "mean-of-max defuzzification: no maximum found".to_string(),
        ));
    }
    Ok(max_x.iter().sum::<f64>() / max_x.len() as f64)
}

/// Largest-of-maximum defuzzification — rightmost x where μ is maximum.
fn defuzz_largest_of_max(x: &[f64], mu: &[f64]) -> Result<f64, FuzzyError> {
    let max_mu = mu.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if max_mu < f64::EPSILON {
        return Err(FuzzyError::NumericalError(
            "largest-of-max defuzzification: aggregate membership is all-zero".to_string(),
        ));
    }
    let largest_x = x
        .iter()
        .zip(mu.iter())
        .filter(|(_, &m)| (m - max_mu).abs() < 1e-10)
        .map(|(&xi, _)| xi)
        .fold(f64::NEG_INFINITY, f64::max);
    if largest_x == f64::NEG_INFINITY {
        return Err(FuzzyError::NumericalError(
            "largest-of-max defuzzification: no maximum found".to_string(),
        ));
    }
    Ok(largest_x)
}

/// Compute the centroid of a fuzzy set sampled over `x_values`.
fn centroid_of_set(set: &FuzzySet, x_values: &[f64]) -> f64 {
    let sum_xmu: f64 = x_values.iter().map(|&x| x * set.degree(x)).sum();
    let sum_mu: f64 = x_values.iter().map(|&x| set.degree(x)).sum();
    if sum_mu < f64::EPSILON {
        // Degenerate: return midpoint of first and last x
        let first = x_values.first().copied().unwrap_or(0.0);
        let last = x_values.last().copied().unwrap_or(0.0);
        (first + last) / 2.0
    } else {
        sum_xmu / sum_mu
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use crate::fuzzy_logic::{
        DefuzzMethod, FuzzyError, FuzzyLogicEngine, FuzzyProposition, FuzzyRule, FuzzySet,
        FuzzyVariable, InferenceMethod, MembershipFunction,
    };
    use std::collections::HashMap;

    // -----------------------------------------------------------------------
    // MembershipFunction tests
    // -----------------------------------------------------------------------

    #[test]
    fn triangular_mf_peak() {
        let mf = MembershipFunction::Triangular {
            a: 0.0,
            b: 5.0,
            c: 10.0,
        };
        assert!((mf.evaluate(5.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn triangular_mf_left_edge() {
        let mf = MembershipFunction::Triangular {
            a: 0.0,
            b: 5.0,
            c: 10.0,
        };
        assert!((mf.evaluate(0.0)).abs() < 1e-10);
    }

    #[test]
    fn triangular_mf_right_edge() {
        let mf = MembershipFunction::Triangular {
            a: 0.0,
            b: 5.0,
            c: 10.0,
        };
        assert!((mf.evaluate(10.0)).abs() < 1e-10);
    }

    #[test]
    fn triangular_mf_outside_range() {
        let mf = MembershipFunction::Triangular {
            a: 0.0,
            b: 5.0,
            c: 10.0,
        };
        assert_eq!(mf.evaluate(-1.0), 0.0);
        assert_eq!(mf.evaluate(11.0), 0.0);
    }

    #[test]
    fn triangular_mf_midpoint_left() {
        let mf = MembershipFunction::Triangular {
            a: 0.0,
            b: 4.0,
            c: 8.0,
        };
        // At x=2, μ = (2-0)/(4-0) = 0.5
        assert!((mf.evaluate(2.0) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn trapezoidal_mf_flat_top() {
        let mf = MembershipFunction::Trapezoidal {
            a: 0.0,
            b: 2.0,
            c: 8.0,
            d: 10.0,
        };
        // Inside flat top [2,8]
        assert!((mf.evaluate(5.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn trapezoidal_mf_rising_slope() {
        let mf = MembershipFunction::Trapezoidal {
            a: 0.0,
            b: 4.0,
            c: 8.0,
            d: 10.0,
        };
        // At x=2 on rising slope: (2-0)/(4-0) = 0.5
        assert!((mf.evaluate(2.0) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn trapezoidal_mf_falling_slope() {
        let mf = MembershipFunction::Trapezoidal {
            a: 0.0,
            b: 2.0,
            c: 8.0,
            d: 10.0,
        };
        // At x=9 on falling slope: (10-9)/(10-8) = 0.5
        assert!((mf.evaluate(9.0) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn trapezoidal_mf_outside() {
        let mf = MembershipFunction::Trapezoidal {
            a: 2.0,
            b: 4.0,
            c: 6.0,
            d: 8.0,
        };
        assert_eq!(mf.evaluate(1.0), 0.0);
        assert_eq!(mf.evaluate(9.0), 0.0);
    }

    #[test]
    fn gaussian_mf_at_mean() {
        let mf = MembershipFunction::Gaussian {
            mean: 5.0,
            sigma: 1.0,
        };
        assert!((mf.evaluate(5.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn gaussian_mf_sigma_symmetry() {
        let mf = MembershipFunction::Gaussian {
            mean: 0.0,
            sigma: 2.0,
        };
        // Symmetric around mean
        let left = mf.evaluate(-1.0);
        let right = mf.evaluate(1.0);
        assert!((left - right).abs() < 1e-10);
    }

    #[test]
    fn gaussian_mf_decay() {
        let mf = MembershipFunction::Gaussian {
            mean: 0.0,
            sigma: 1.0,
        };
        // At x=1, μ = exp(-0.5) ≈ 0.6065
        assert!((mf.evaluate(1.0) - std::f64::consts::E.powf(-0.5)).abs() < 1e-10);
    }

    #[test]
    fn singleton_mf_exact() {
        let mf = MembershipFunction::Singleton { value: 3.0 };
        assert!((mf.evaluate(3.0) - 1.0).abs() < 1e-10);
        assert_eq!(mf.evaluate(3.1), 0.0);
    }

    #[test]
    fn universe_mf_always_one() {
        let mf = MembershipFunction::Universe;
        assert!((mf.evaluate(-100.0) - 1.0).abs() < 1e-10);
        assert!((mf.evaluate(0.0) - 1.0).abs() < 1e-10);
        assert!((mf.evaluate(100.0) - 1.0).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // FuzzySet tests
    // -----------------------------------------------------------------------

    #[test]
    fn fuzzy_set_degree_delegates_to_mf() {
        let set = FuzzySet::new(
            "hot",
            MembershipFunction::Triangular {
                a: 5.0,
                b: 10.0,
                c: 15.0,
            },
        );
        assert!((set.degree(10.0) - 1.0).abs() < 1e-10);
        assert_eq!(set.degree(5.0), 0.0);
    }

    // -----------------------------------------------------------------------
    // FuzzyVariable tests
    // -----------------------------------------------------------------------

    #[test]
    fn fuzzy_variable_add_and_get_set() {
        let mut var = FuzzyVariable::new("temp", 0.0, 100.0);
        var.add_set(FuzzySet::new(
            "low",
            MembershipFunction::Triangular {
                a: 0.0,
                b: 0.0,
                c: 50.0,
            },
        ));
        var.add_set(FuzzySet::new(
            "high",
            MembershipFunction::Triangular {
                a: 50.0,
                b: 100.0,
                c: 100.0,
            },
        ));
        assert!(var.get_set("low").is_some());
        assert!(var.get_set("medium").is_none());
    }

    // -----------------------------------------------------------------------
    // Engine construction and registration
    // -----------------------------------------------------------------------

    fn build_temperature_engine() -> FuzzyLogicEngine {
        let mut engine = FuzzyLogicEngine::new(InferenceMethod::Mamdani, DefuzzMethod::Centroid);

        // Input variable: temperature [0, 100]
        // Use Trapezoidal for cold/hot so the endpoints have a full-membership plateau.
        let mut temp = FuzzyVariable::new("temperature", 0.0, 100.0);
        temp.add_set(FuzzySet::new(
            "cold",
            MembershipFunction::Trapezoidal {
                a: 0.0,
                b: 0.0,
                c: 20.0,
                d: 40.0,
            },
        ));
        temp.add_set(FuzzySet::new(
            "warm",
            MembershipFunction::Triangular {
                a: 30.0,
                b: 50.0,
                c: 70.0,
            },
        ));
        temp.add_set(FuzzySet::new(
            "hot",
            MembershipFunction::Trapezoidal {
                a: 60.0,
                b: 80.0,
                c: 100.0,
                d: 100.0,
            },
        ));
        engine.add_variable(temp).expect("add temperature");

        // Output variable: fan_speed [0, 100]
        let mut fan = FuzzyVariable::new("fan_speed", 0.0, 100.0);
        fan.add_set(FuzzySet::new(
            "slow",
            MembershipFunction::Trapezoidal {
                a: 0.0,
                b: 0.0,
                c: 20.0,
                d: 40.0,
            },
        ));
        fan.add_set(FuzzySet::new(
            "medium",
            MembershipFunction::Triangular {
                a: 30.0,
                b: 50.0,
                c: 70.0,
            },
        ));
        fan.add_set(FuzzySet::new(
            "fast",
            MembershipFunction::Trapezoidal {
                a: 60.0,
                b: 80.0,
                c: 100.0,
                d: 100.0,
            },
        ));
        engine.add_variable(fan).expect("add fan_speed");

        // Rules
        engine
            .add_rule(FuzzyRule::new(
                vec![FuzzyProposition::new("temperature", "cold")],
                FuzzyProposition::new("fan_speed", "slow"),
                1.0,
            ))
            .expect("add rule cold->slow");
        engine
            .add_rule(FuzzyRule::new(
                vec![FuzzyProposition::new("temperature", "warm")],
                FuzzyProposition::new("fan_speed", "medium"),
                1.0,
            ))
            .expect("add rule warm->medium");
        engine
            .add_rule(FuzzyRule::new(
                vec![FuzzyProposition::new("temperature", "hot")],
                FuzzyProposition::new("fan_speed", "fast"),
                1.0,
            ))
            .expect("add rule hot->fast");

        engine
    }

    #[test]
    fn engine_duplicate_variable_error() {
        let mut engine = FuzzyLogicEngine::new(InferenceMethod::Mamdani, DefuzzMethod::Centroid);
        let var = FuzzyVariable::new("x", 0.0, 1.0);
        engine.add_variable(var.clone()).expect("first add");
        let err = engine.add_variable(var);
        assert!(matches!(err, Err(FuzzyError::DuplicateVariable(_))));
    }

    #[test]
    fn engine_add_rule_missing_variable_error() {
        let mut engine = FuzzyLogicEngine::new(InferenceMethod::Mamdani, DefuzzMethod::Centroid);
        let rule = FuzzyRule::new(
            vec![FuzzyProposition::new("nonexistent", "set")],
            FuzzyProposition::new("out", "s"),
            1.0,
        );
        let err = engine.add_rule(rule);
        assert!(matches!(err, Err(FuzzyError::VariableNotFound(_))));
    }

    #[test]
    fn engine_add_rule_missing_set_error() {
        let mut engine = FuzzyLogicEngine::new(InferenceMethod::Mamdani, DefuzzMethod::Centroid);
        let mut var = FuzzyVariable::new("x", 0.0, 10.0);
        var.add_set(FuzzySet::new("low", MembershipFunction::Universe));
        engine.add_variable(var).expect("add var");
        // Reference a set that doesn't exist on "x"
        let rule = FuzzyRule::new(
            vec![FuzzyProposition::new("x", "nonexistent_set")],
            FuzzyProposition::new("x", "low"),
            1.0,
        );
        let err = engine.add_rule(rule);
        assert!(matches!(err, Err(FuzzyError::SetNotFound { .. })));
    }

    #[test]
    fn engine_rule_count() {
        let engine = build_temperature_engine();
        assert_eq!(engine.rule_count(), 3);
    }

    #[test]
    fn engine_variable_names_count() {
        let engine = build_temperature_engine();
        assert_eq!(engine.variable_names().len(), 2);
    }

    // -----------------------------------------------------------------------
    // Fuzzify tests
    // -----------------------------------------------------------------------

    #[test]
    fn fuzzify_returns_all_sets() {
        let engine = build_temperature_engine();
        let result = engine.fuzzify("temperature", 50.0).expect("fuzzify");
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn fuzzify_unknown_variable() {
        let engine = build_temperature_engine();
        let err = engine.fuzzify("humidity", 50.0);
        assert!(matches!(err, Err(FuzzyError::VariableNotFound(_))));
    }

    // -----------------------------------------------------------------------
    // Fire rule tests
    // -----------------------------------------------------------------------

    #[test]
    fn fire_rule_full_activation() {
        let engine = build_temperature_engine();
        let mut inputs = HashMap::new();
        inputs.insert("temperature".to_string(), 0.0_f64); // fully cold
        let rule = &engine.rules[0]; // cold -> slow
        let alpha = engine.fire_rule(rule, &inputs).expect("fire rule");
        assert!((alpha - 1.0).abs() < 1e-10);
    }

    #[test]
    fn fire_rule_zero_activation() {
        let engine = build_temperature_engine();
        let mut inputs = HashMap::new();
        inputs.insert("temperature".to_string(), 100.0_f64); // fully hot, not cold
        let rule = &engine.rules[0]; // cold -> slow (should fire at 0)
        let alpha = engine.fire_rule(rule, &inputs).expect("fire rule");
        assert!(alpha < 1e-10);
    }

    #[test]
    fn fire_rule_weight_applied() {
        let engine = build_temperature_engine();
        // Build a rule with weight 0.5
        let mut inputs = HashMap::new();
        inputs.insert("temperature".to_string(), 0.0_f64);
        let rule = FuzzyRule::new(
            vec![FuzzyProposition::new("temperature", "cold")],
            FuzzyProposition::new("fan_speed", "slow"),
            0.5,
        );
        let alpha = engine.fire_rule(&rule, &inputs).expect("fire rule");
        assert!((alpha - 0.5).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // Mamdani inference tests
    // -----------------------------------------------------------------------

    #[test]
    fn mamdani_cold_input_gives_low_speed() {
        let engine = build_temperature_engine();
        let mut inputs = HashMap::new();
        inputs.insert("temperature".to_string(), 5.0_f64);
        let speed = engine.infer(&inputs, "fan_speed").expect("infer");
        // Cold input should yield slow (low) fan speed
        assert!(speed < 40.0, "expected slow speed, got {speed}");
    }

    #[test]
    fn mamdani_hot_input_gives_high_speed() {
        let engine = build_temperature_engine();
        let mut inputs = HashMap::new();
        inputs.insert("temperature".to_string(), 95.0_f64);
        let speed = engine.infer(&inputs, "fan_speed").expect("infer");
        // Hot input should yield fast fan speed
        assert!(speed > 60.0, "expected fast speed, got {speed}");
    }

    #[test]
    fn mamdani_no_rules_for_output_error() {
        let engine = build_temperature_engine();
        let inputs = HashMap::new();
        let err = engine.infer(&inputs, "temperature");
        assert!(matches!(err, Err(FuzzyError::NoRulesForOutput(_))));
    }

    // -----------------------------------------------------------------------
    // Sugeno inference tests
    // -----------------------------------------------------------------------

    #[test]
    fn sugeno_cold_input_gives_low_speed() {
        let mut engine = build_temperature_engine();
        // Switch to Sugeno
        engine.inference = InferenceMethod::Sugeno;
        let mut inputs = HashMap::new();
        inputs.insert("temperature".to_string(), 5.0_f64);
        let speed = engine.infer(&inputs, "fan_speed").expect("infer sugeno");
        assert!(speed < 40.0, "expected slow speed, got {speed}");
    }

    #[test]
    fn sugeno_hot_input_gives_high_speed() {
        let mut engine = build_temperature_engine();
        engine.inference = InferenceMethod::Sugeno;
        let mut inputs = HashMap::new();
        inputs.insert("temperature".to_string(), 95.0_f64);
        let speed = engine.infer(&inputs, "fan_speed").expect("infer sugeno");
        assert!(speed > 60.0, "expected fast speed, got {speed}");
    }

    // -----------------------------------------------------------------------
    // Defuzzification method tests
    // -----------------------------------------------------------------------

    #[test]
    fn mean_of_max_defuzz() {
        let mut engine = build_temperature_engine();
        engine.defuzz = DefuzzMethod::MeanOfMax;
        let mut inputs = HashMap::new();
        inputs.insert("temperature".to_string(), 50.0_f64);
        let speed = engine.infer(&inputs, "fan_speed").expect("infer mom");
        assert!((0.0..=100.0).contains(&speed));
    }

    #[test]
    fn largest_of_max_defuzz() {
        let mut engine = build_temperature_engine();
        engine.defuzz = DefuzzMethod::LargestOfMax;
        let mut inputs = HashMap::new();
        inputs.insert("temperature".to_string(), 50.0_f64);
        let speed = engine.infer(&inputs, "fan_speed").expect("infer lom");
        assert!((0.0..=100.0).contains(&speed));
    }

    // -----------------------------------------------------------------------
    // Membership at helper test
    // -----------------------------------------------------------------------

    #[test]
    fn membership_at_correct_value() {
        let engine = build_temperature_engine();
        let mu = engine
            .membership_at("temperature", "warm", 50.0)
            .expect("membership_at");
        assert!((mu - 1.0).abs() < 1e-10);
    }

    #[test]
    fn membership_at_unknown_set() {
        let engine = build_temperature_engine();
        let err = engine.membership_at("temperature", "boiling", 50.0);
        assert!(matches!(err, Err(FuzzyError::SetNotFound { .. })));
    }

    #[test]
    fn membership_at_unknown_variable() {
        let engine = build_temperature_engine();
        let err = engine.membership_at("pressure", "low", 50.0);
        assert!(matches!(err, Err(FuzzyError::VariableNotFound(_))));
    }

    // -----------------------------------------------------------------------
    // evaluate() aliasing test
    // -----------------------------------------------------------------------

    #[test]
    fn evaluate_same_as_infer() {
        let mut engine = build_temperature_engine();
        let mut inputs = HashMap::new();
        inputs.insert("temperature".to_string(), 60.0_f64);
        let via_infer = engine.infer(&inputs, "fan_speed").expect("infer");
        let via_evaluate = engine
            .evaluate(&inputs.clone(), "fan_speed")
            .expect("evaluate");
        assert!((via_infer - via_evaluate).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // Stats test
    // -----------------------------------------------------------------------

    #[test]
    fn stats_reports_correct_counts() {
        let engine = build_temperature_engine();
        let stats = engine.stats();
        assert_eq!(stats.variable_count, 2);
        assert_eq!(stats.rule_count, 3);
        assert!(stats.inference.contains("Mamdani"));
        assert!(stats.defuzz.contains("Centroid"));
    }

    // -----------------------------------------------------------------------
    // Multi-antecedent AND rule test
    // -----------------------------------------------------------------------

    #[test]
    fn multi_antecedent_and_rule() {
        let mut engine = FuzzyLogicEngine::new(InferenceMethod::Mamdani, DefuzzMethod::Centroid);
        // Two inputs: temperature and humidity
        let mut temp = FuzzyVariable::new("temperature", 0.0, 100.0);
        temp.add_set(FuzzySet::new(
            "hot",
            MembershipFunction::Trapezoidal {
                a: 50.0,
                b: 75.0,
                c: 100.0,
                d: 100.0,
            },
        ));
        engine.add_variable(temp).expect("temp");

        let mut hum = FuzzyVariable::new("humidity", 0.0, 100.0);
        hum.add_set(FuzzySet::new(
            "high",
            MembershipFunction::Trapezoidal {
                a: 50.0,
                b: 75.0,
                c: 100.0,
                d: 100.0,
            },
        ));
        engine.add_variable(hum).expect("hum");

        let mut comfort = FuzzyVariable::new("discomfort", 0.0, 100.0);
        comfort.add_set(FuzzySet::new(
            "severe",
            MembershipFunction::Trapezoidal {
                a: 50.0,
                b: 75.0,
                c: 100.0,
                d: 100.0,
            },
        ));
        engine.add_variable(comfort).expect("comfort");

        // IF temperature IS hot AND humidity IS high THEN discomfort IS severe
        engine
            .add_rule(FuzzyRule::new(
                vec![
                    FuzzyProposition::new("temperature", "hot"),
                    FuzzyProposition::new("humidity", "high"),
                ],
                FuzzyProposition::new("discomfort", "severe"),
                1.0,
            ))
            .expect("add multi rule");

        let mut inputs = HashMap::new();
        inputs.insert("temperature".to_string(), 90.0_f64);
        inputs.insert("humidity".to_string(), 90.0_f64);
        let result = engine.infer(&inputs, "discomfort").expect("multi infer");
        // Both inputs are hot/high — should yield some discomfort
        assert!(result > 50.0, "expected high discomfort, got {result}");
    }

    #[test]
    fn multi_antecedent_and_takes_min() {
        let mut engine = FuzzyLogicEngine::new(InferenceMethod::Mamdani, DefuzzMethod::Centroid);

        let mut a = FuzzyVariable::new("a", 0.0, 1.0);
        a.add_set(FuzzySet::new("s", MembershipFunction::Universe));
        engine.add_variable(a).expect("a");

        let mut b = FuzzyVariable::new("b", 0.0, 1.0);
        b.add_set(FuzzySet::new(
            "s",
            MembershipFunction::Triangular {
                a: 0.0,
                b: 0.0,
                c: 1.0,
            },
        ));
        engine.add_variable(b).expect("b");

        let mut out = FuzzyVariable::new("out", 0.0, 1.0);
        out.add_set(FuzzySet::new("s", MembershipFunction::Universe));
        engine.add_variable(out).expect("out");

        engine
            .add_rule(FuzzyRule::new(
                vec![
                    FuzzyProposition::new("a", "s"), // degree = 1 always
                    FuzzyProposition::new("b", "s"), // degree = 1 - b_val
                ],
                FuzzyProposition::new("out", "s"),
                1.0,
            ))
            .expect("rule");

        let mut inputs = HashMap::new();
        inputs.insert("a".to_string(), 0.5_f64);
        inputs.insert("b".to_string(), 0.0_f64); // b at left edge => degree = 1.0 (triangular 0-0-1 at x=0)

        // a IS Universe (degree=1), b IS Triangular at x=0 => degree=1 (peak at 0)
        // min(1, 1) = 1 → out IS Universe clipped at 1 → centroid = 0.5
        let result = engine.infer(&inputs, "out").expect("and min");
        assert!((result - 0.5).abs() < 0.1, "expected ~0.5, got {result}");
    }

    // -----------------------------------------------------------------------
    // Gaussian MF integration test
    // -----------------------------------------------------------------------

    #[test]
    fn infer_with_gaussian_mf() {
        let mut engine = FuzzyLogicEngine::new(InferenceMethod::Mamdani, DefuzzMethod::Centroid);

        let mut inp = FuzzyVariable::new("signal", 0.0, 10.0);
        inp.add_set(FuzzySet::new(
            "mid",
            MembershipFunction::Gaussian {
                mean: 5.0,
                sigma: 1.0,
            },
        ));
        engine.add_variable(inp).expect("inp");

        let mut out = FuzzyVariable::new("output", 0.0, 10.0);
        out.add_set(FuzzySet::new(
            "mid",
            MembershipFunction::Gaussian {
                mean: 5.0,
                sigma: 1.0,
            },
        ));
        engine.add_variable(out).expect("out");

        engine
            .add_rule(FuzzyRule::new(
                vec![FuzzyProposition::new("signal", "mid")],
                FuzzyProposition::new("output", "mid"),
                1.0,
            ))
            .expect("gaussian rule");

        let mut inputs = HashMap::new();
        inputs.insert("signal".to_string(), 5.0_f64); // exactly at mean
        let result = engine.infer(&inputs, "output").expect("gaussian infer");
        // Rule fires at degree 1 → centroid of gaussian centred at 5 ≈ 5
        assert!((result - 5.0).abs() < 0.5, "expected ~5.0, got {result}");
    }

    // -----------------------------------------------------------------------
    // Sugeno no-activation numerical error
    // -----------------------------------------------------------------------

    #[test]
    fn sugeno_zero_activation_numerical_error() {
        let mut engine = FuzzyLogicEngine::new(InferenceMethod::Sugeno, DefuzzMethod::Centroid);

        let mut inp = FuzzyVariable::new("x", 0.0, 10.0);
        inp.add_set(FuzzySet::new(
            "low",
            MembershipFunction::Triangular {
                a: 0.0,
                b: 0.0,
                c: 5.0,
            },
        ));
        engine.add_variable(inp).expect("x");

        let mut out = FuzzyVariable::new("y", 0.0, 10.0);
        out.add_set(FuzzySet::new("s", MembershipFunction::Universe));
        engine.add_variable(out).expect("y");

        engine
            .add_rule(FuzzyRule::new(
                vec![FuzzyProposition::new("x", "low")],
                FuzzyProposition::new("y", "s"),
                1.0,
            ))
            .expect("rule");

        // At x=10, "low" has zero degree → Sugeno total weight is 0
        let mut inputs = HashMap::new();
        inputs.insert("x".to_string(), 10.0_f64);
        let err = engine.infer(&inputs, "y");
        assert!(matches!(err, Err(FuzzyError::NumericalError(_))));
    }
}
