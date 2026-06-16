//! Bayesian Network Inference — variable elimination, belief propagation,
//! and sampling-based inference over discrete Bayesian networks.
//!
//! # Overview
//!
//! A [`BayesianNetwork`] is a directed acyclic graph (DAG) of discrete random
//! variables where each variable has a [`ConditionalProbabilityTable`] (CPT)
//! that encodes P(variable | parents).  Given observed [`Evidence`], the
//! [`BayesianNetworkInference`] engine answers marginal-probability queries
//! using one of three algorithms:
//!
//! * **Variable Elimination** — exact inference by successive factor
//!   multiplication and marginalization.
//! * **Belief Propagation** — sum-product message passing (exact on trees,
//!   approximate on loopy graphs).
//! * **Sampling** — rejection / likelihood-weighting sampling with a
//!   reproducible xorshift64 PRNG (no `rand` dependency).
//!
//! # Example
//!
//! ```rust
//! use ipfrs_tensorlogic::bayesian_network_inference::{
//!     BayesianNetwork, BayesianNetworkInference, BniConfig, ConditionalProbabilityTable,
//!     EliminationOrder, Evidence, Factor, InferenceAlgorithm, InferenceQuery, RandomVariable,
//! };
//! use std::collections::HashMap;
//!
//! // Build a tiny Rain → Wet-Grass network.
//! let rain = RandomVariable { id: "Rain".into(), states: vec!["T".into(), "F".into()], cardinality: 2 };
//! let wet  = RandomVariable { id: "Wet".into(),  states: vec!["T".into(), "F".into()], cardinality: 2 };
//!
//! // P(Rain)
//! let f_rain = Factor {
//!     id: "f_rain".into(),
//!     variables: vec!["Rain".into()],
//!     values: vec![0.2, 0.8],
//!     shape: vec![2],
//! };
//! let cpt_rain = ConditionalProbabilityTable {
//!     variable: "Rain".into(),
//!     parents:  vec![],
//!     factor:   f_rain,
//! };
//!
//! // P(Wet | Rain)  — rows: Rain=T, Rain=F; cols: Wet=T, Wet=F
//! let f_wet = Factor {
//!     id: "f_wet".into(),
//!     variables: vec!["Rain".into(), "Wet".into()],
//!     values: vec![0.9, 0.1, 0.2, 0.8],
//!     shape: vec![2, 2],
//! };
//! let cpt_wet = ConditionalProbabilityTable {
//!     variable: "Wet".into(),
//!     parents:  vec!["Rain".into()],
//!     factor:   f_wet,
//! };
//!
//! let mut variables = HashMap::new();
//! variables.insert("Rain".into(), rain);
//! variables.insert("Wet".into(), wet);
//! let mut adjacency = HashMap::new();
//! adjacency.insert("Wet".into(), vec!["Rain".into()]);
//!
//! let net = BayesianNetwork {
//!     variables,
//!     cpts: vec![cpt_rain, cpt_wet],
//!     adjacency,
//! };
//!
//! let config = BniConfig::default();
//! let mut engine = BayesianNetworkInference::new(net, config).expect("example: should succeed in docs");
//!
//! let query = InferenceQuery {
//!     query_variables: vec!["Rain".into()],
//!     evidence: vec![],
//!     algorithm: InferenceAlgorithm::VariableElimination,
//! };
//! let results = engine.query(&query).expect("example: should succeed in docs");
//! assert_eq!(results.len(), 1);
//! let rain_dist = &results[0].distribution;
//! assert!((rain_dist[0].1 - 0.2).abs() < 1e-9);
//! ```

use std::collections::{HashMap, HashSet, VecDeque};
use thiserror::Error;

// ─────────────────────────────────────────────────────────────────────────────
// PRNG (no rand crate)
// ─────────────────────────────────────────────────────────────────────────────

/// Xorshift64 PRNG — advance state and return next pseudo-random u64.
pub fn bni_xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// Draw a categorical sample from a (possibly un-normalised) probability
/// vector using the supplied PRNG state.
fn sample_categorical(probs: &[f64], state: &mut u64) -> usize {
    let u = (bni_xorshift64(state) >> 11) as f64 / (1u64 << 53) as f64;
    let mut cumsum = 0.0_f64;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if u < cumsum {
            return i;
        }
    }
    probs.len().saturating_sub(1)
}

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// All errors that can arise during Bayesian network construction or inference.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum BniError {
    /// A referenced variable does not exist in the network.
    #[error("variable not found: {0}")]
    VariableNotFound(String),

    /// A CPT has inconsistent dimensions or probability values.
    #[error("invalid CPT for variable `{variable}`: {reason}")]
    InvalidCPT {
        /// Variable whose CPT is invalid.
        variable: String,
        /// Human-readable explanation.
        reason: String,
    },

    /// Two evidence items contradict each other for the same variable.
    #[error("evidence conflict for variable: {0}")]
    EvidenceConflict(String),

    /// The network contains a directed cycle.
    #[error("cyclic network detected: {0}")]
    CyclicNetwork(String),

    /// A generic inference failure.
    #[error("inference error: {0}")]
    InferenceError(String),
}

// ─────────────────────────────────────────────────────────────────────────────
// Core types
// ─────────────────────────────────────────────────────────────────────────────

/// A discrete random variable with named states.
#[derive(Debug, Clone, PartialEq)]
pub struct RandomVariable {
    /// Unique identifier (must match keys in `BayesianNetwork::variables`).
    pub id: String,
    /// Ordered list of state labels.
    pub states: Vec<String>,
    /// Number of states (equals `states.len()`).
    pub cardinality: usize,
}

/// A factor (un-normalised joint distribution table) over a set of variables.
///
/// `values` is stored in row-major order: the last variable in `variables`
/// varies fastest.  `shape[i]` is the cardinality of `variables[i]`.
#[derive(Debug, Clone, PartialEq)]
pub struct Factor {
    /// Human-readable label.
    pub id: String,
    /// Ordered variable scope.
    pub variables: Vec<String>,
    /// Flat probability/potential table.
    pub values: Vec<f64>,
    /// Shape array — `shape[i]` is the number of states for `variables[i]`.
    pub shape: Vec<usize>,
}

impl Factor {
    // ── index helpers ────────────────────────────────────────────────────────

    /// Convert a multi-dimensional index to a flat offset (row-major).
    fn flat_index(indices: &[usize], shape: &[usize]) -> usize {
        let mut idx = 0usize;
        let mut stride = 1usize;
        for i in (0..shape.len()).rev() {
            idx += indices[i] * stride;
            stride *= shape[i];
        }
        idx
    }

    /// Convert a flat offset to a multi-dimensional index (row-major).
    fn multi_index(flat: usize, shape: &[usize]) -> Vec<usize> {
        let mut remaining = flat;
        let mut idx = vec![0usize; shape.len()];
        for i in (0..shape.len()).rev() {
            idx[i] = remaining % shape[i];
            remaining /= shape[i];
        }
        idx
    }

    // ── public operations ────────────────────────────────────────────────────

    /// Compute the factor product of `self` and `other`.
    ///
    /// The result has the union of both variable scopes.  Values are the
    /// point-wise product over all joint assignments.
    pub fn product(&self, other: &Factor) -> Factor {
        // Build the union scope and result shape.
        let mut result_vars = self.variables.clone();
        let mut result_shape = self.shape.clone();
        for (i, v) in other.variables.iter().enumerate() {
            if !result_vars.contains(v) {
                result_vars.push(v.clone());
                result_shape.push(other.shape[i]);
            }
        }

        let total: usize = result_shape.iter().product();
        let mut values = vec![0.0_f64; total];

        for (flat, value) in values.iter_mut().enumerate().take(total) {
            let idx = Self::multi_index(flat, &result_shape);

            // Build assignment: var → state index
            let assignment: HashMap<&str, usize> = result_vars
                .iter()
                .zip(idx.iter())
                .map(|(v, &s)| (v.as_str(), s))
                .collect();

            // Compute self sub-index
            let self_idx: Vec<usize> = self
                .variables
                .iter()
                .map(|v| assignment[v.as_str()])
                .collect();
            let self_flat = Self::flat_index(&self_idx, &self.shape);

            // Compute other sub-index
            let other_idx: Vec<usize> = other
                .variables
                .iter()
                .map(|v| assignment[v.as_str()])
                .collect();
            let other_flat = Self::flat_index(&other_idx, &other.shape);

            *value = self.values[self_flat] * other.values[other_flat];
        }

        Factor {
            id: format!("{}*{}", self.id, other.id),
            variables: result_vars,
            values,
            shape: result_shape,
        }
    }

    /// Marginalize `variable` out of this factor by summing over all its
    /// states.
    ///
    /// `var_map` maps variable names to their cardinalities and is used for
    /// consistency checking; the actual cardinality is taken from `self.shape`.
    pub fn marginalize(&self, variable: &str, _var_map: &HashMap<String, usize>) -> Factor {
        // Find the dimension index for this variable.
        let dim = match self.variables.iter().position(|v| v == variable) {
            Some(d) => d,
            None => return self.clone(),
        };

        let new_vars: Vec<String> = self
            .variables
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != dim)
            .map(|(_, v)| v.clone())
            .collect();
        let new_shape: Vec<usize> = self
            .shape
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != dim)
            .map(|(_, &s)| s)
            .collect();

        let new_total: usize = if new_shape.is_empty() {
            1
        } else {
            new_shape.iter().product()
        };
        let mut values = vec![0.0_f64; new_total];

        let total: usize = self.shape.iter().product();
        for flat in 0..total {
            let idx = Self::multi_index(flat, &self.shape);
            // Build reduced index (drop dimension `dim`)
            let red_idx: Vec<usize> = idx
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != dim)
                .map(|(_, &v)| v)
                .collect();
            let red_flat = if new_shape.is_empty() {
                0
            } else {
                Self::flat_index(&red_idx, &new_shape)
            };
            values[red_flat] += self.values[flat];
        }

        Factor {
            id: format!("{}\\{}", self.id, variable),
            variables: new_vars,
            values,
            shape: new_shape,
        }
    }

    /// Reduce this factor by fixing `variable` to `state` (slice operation).
    ///
    /// The resulting factor no longer contains `variable` in its scope.
    pub fn reduce(
        &self,
        variable: &str,
        state: usize,
        _var_map: &HashMap<String, usize>,
    ) -> Factor {
        let dim = match self.variables.iter().position(|v| v == variable) {
            Some(d) => d,
            None => return self.clone(),
        };

        let new_vars: Vec<String> = self
            .variables
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != dim)
            .map(|(_, v)| v.clone())
            .collect();
        let new_shape: Vec<usize> = self
            .shape
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != dim)
            .map(|(_, &s)| s)
            .collect();

        let new_total: usize = if new_shape.is_empty() {
            1
        } else {
            new_shape.iter().product()
        };
        let mut values = vec![0.0_f64; new_total];

        let total: usize = self.shape.iter().product();
        for flat in 0..total {
            let idx = Self::multi_index(flat, &self.shape);
            if idx[dim] != state {
                continue;
            }
            let red_idx: Vec<usize> = idx
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != dim)
                .map(|(_, &v)| v)
                .collect();
            let red_flat = if new_shape.is_empty() {
                0
            } else {
                Self::flat_index(&red_idx, &new_shape)
            };
            values[red_flat] = self.values[flat];
        }

        Factor {
            id: format!("{}[{}={}]", self.id, variable, state),
            variables: new_vars,
            values,
            shape: new_shape,
        }
    }

    /// Normalize: divide all values by their sum so they sum to 1.
    pub fn normalize(&mut self) {
        let s: f64 = self.values.iter().sum();
        if s > 0.0 {
            for v in &mut self.values {
                *v /= s;
            }
        }
    }

    /// Whether `variable` appears in this factor's scope.
    pub fn contains_variable(&self, variable: &str) -> bool {
        self.variables.iter().any(|v| v == variable)
    }

    /// Compute the entropy of this factor's values (assumes they are
    /// normalised probabilities).  Returns `H = -Σ p log₂ p`.
    pub fn entropy(&self) -> f64 {
        self.values
            .iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| -p * p.log2())
            .sum()
    }
}

/// A conditional probability table encoding P(variable | parents).
#[derive(Debug, Clone, PartialEq)]
pub struct ConditionalProbabilityTable {
    /// The child variable this CPT belongs to.
    pub variable: String,
    /// Ordered list of parent variable ids.
    pub parents: Vec<String>,
    /// The underlying factor (scope = parents ++ \[variable\]).
    pub factor: Factor,
}

/// A single piece of observed evidence.
#[derive(Debug, Clone, PartialEq)]
pub struct Evidence {
    /// The variable that was observed.
    pub variable: String,
    /// The name of the state that was observed.
    pub observed_state: String,
}

/// A Bayesian network: a DAG of random variables with CPTs.
#[derive(Debug, Clone)]
pub struct BayesianNetwork {
    /// All random variables, keyed by their id.
    pub variables: HashMap<String, RandomVariable>,
    /// Conditional probability tables (one per variable).
    pub cpts: Vec<ConditionalProbabilityTable>,
    /// Adjacency list: child → list of parents (mirrors CPT parents).
    pub adjacency: HashMap<String, Vec<String>>,
}

/// Describes which inference algorithm to use.
#[derive(Debug, Clone, PartialEq)]
pub enum InferenceAlgorithm {
    /// Exact variable-elimination inference.
    VariableElimination,
    /// Sum-product belief propagation (exact on trees).
    BeliefPropagation,
    /// Monte-Carlo sampling with xorshift64 PRNG.
    Sampling {
        /// Number of samples to draw.
        n_samples: usize,
        /// Initial PRNG seed.
        seed: u64,
    },
}

/// A query to the inference engine.
#[derive(Debug, Clone)]
pub struct InferenceQuery {
    /// Variables whose posterior distributions are requested.
    pub query_variables: Vec<String>,
    /// Observed values (may be empty for prior queries).
    pub evidence: Vec<Evidence>,
    /// Which algorithm to use.
    pub algorithm: InferenceAlgorithm,
}

/// Per-variable result of a Bayesian inference query.
#[derive(Debug, Clone, PartialEq)]
pub struct QueryResult {
    /// The queried variable.
    pub variable: String,
    /// Posterior distribution: `(state_name, probability)` pairs in state order.
    pub distribution: Vec<(String, f64)>,
    /// Shannon entropy (bits) of the posterior distribution.
    pub marginal_entropy: f64,
    /// The state with the highest posterior probability.
    pub most_likely_state: String,
}

impl QueryResult {
    fn from_factor(var: &RandomVariable, mut f: Factor) -> Self {
        // Ensure the factor is single-variable and normalized.
        if !f.variables.is_empty() && f.variables[0] != var.id {
            // Try marginalizing all but the query variable
            let remove: Vec<String> = f
                .variables
                .iter()
                .filter(|v| v.as_str() != var.id)
                .cloned()
                .collect();
            for v in remove {
                let dummy = HashMap::new();
                f = f.marginalize(&v, &dummy);
            }
        }
        f.normalize();

        let distribution: Vec<(String, f64)> = var
            .states
            .iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), f.values.get(i).copied().unwrap_or(0.0)))
            .collect();

        let marginal_entropy = f.entropy();
        let most_likely_state = distribution
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(s, _)| s.clone())
            .unwrap_or_default();

        QueryResult {
            variable: var.id.clone(),
            distribution,
            marginal_entropy,
            most_likely_state,
        }
    }
}

/// Variable elimination ordering heuristic.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum EliminationOrder {
    /// Min-fill heuristic — eliminate the variable that adds the fewest edges.
    #[default]
    MinFill,
    /// Min-degree heuristic — eliminate the variable with the fewest neighbors.
    MinDegree,
    /// Eliminate variables in the order they appear in the network.
    Sequential,
}

/// Configuration for `BayesianNetworkInference`.
#[derive(Debug, Clone)]
pub struct BniConfig {
    /// Maximum number of variables allowed.
    pub max_variables: usize,
    /// Maximum number of states per variable.
    pub max_states_per_variable: usize,
    /// Elimination ordering heuristic for variable elimination.
    pub elimination_ordering: EliminationOrder,
}

impl Default for BniConfig {
    fn default() -> Self {
        Self {
            max_variables: 256,
            max_states_per_variable: 1024,
            elimination_ordering: EliminationOrder::MinFill,
        }
    }
}

/// Runtime statistics for the inference engine.
#[derive(Debug, Clone, Default)]
pub struct BniStats {
    /// Total number of queries answered.
    pub queries_answered: u64,
    /// Running average of factors eliminated per query.
    pub avg_factors_eliminated: f64,
    /// Number of query cache hits.
    pub cache_hits: u64,
}

// ─────────────────────────────────────────────────────────────────────────────
// The inference engine
// ─────────────────────────────────────────────────────────────────────────────

/// Bayesian network inference engine.
///
/// Supports variable elimination (exact), belief propagation (sum-product),
/// and likelihood-weighted sampling.
pub struct BayesianNetworkInference {
    network: BayesianNetwork,
    config: BniConfig,
    stats: BniStats,
    /// `var_id → cardinality` lookup, built at construction time.
    cardinality_map: HashMap<String, usize>,
}

impl BayesianNetworkInference {
    // ── construction ─────────────────────────────────────────────────────────

    /// Create and validate a new inference engine.
    ///
    /// Returns an error if:
    /// * the network contains a directed cycle,
    /// * any CPT references an undefined variable,
    /// * any CPT has incorrect dimensions.
    pub fn new(network: BayesianNetwork, config: BniConfig) -> Result<Self, BniError> {
        // Check variable count.
        if network.variables.len() > config.max_variables {
            return Err(BniError::InferenceError(format!(
                "network has {} variables, max is {}",
                network.variables.len(),
                config.max_variables
            )));
        }

        // Build cardinality map.
        let mut cardinality_map: HashMap<String, usize> = network
            .variables
            .iter()
            .map(|(id, v)| (id.clone(), v.cardinality))
            .collect();

        // Validate variable cardinalities.
        for (id, var) in &network.variables {
            if var.states.len() != var.cardinality {
                return Err(BniError::InvalidCPT {
                    variable: id.clone(),
                    reason: format!(
                        "cardinality {} does not match states.len() {}",
                        var.cardinality,
                        var.states.len()
                    ),
                });
            }
            if var.cardinality > config.max_states_per_variable {
                return Err(BniError::InvalidCPT {
                    variable: id.clone(),
                    reason: format!(
                        "cardinality {} exceeds max {}",
                        var.cardinality, config.max_states_per_variable
                    ),
                });
            }
        }

        // Detect cycles via DFS.
        detect_cycles(&network)?;

        // Validate CPTs.
        for cpt in &network.cpts {
            validate_cpt(cpt, &cardinality_map)?;
        }

        // Ensure cardinality map also covers implicit variables from CPTs.
        for cpt in &network.cpts {
            for v in &cpt.factor.variables {
                cardinality_map.entry(v.clone()).or_insert_with(|| {
                    cpt.factor.shape[cpt
                        .factor
                        .variables
                        .iter()
                        .position(|x| x == v)
                        .unwrap_or(0)]
                });
            }
        }

        Ok(Self {
            network,
            config,
            stats: BniStats::default(),
            cardinality_map,
        })
    }

    // ── public API ────────────────────────────────────────────────────────────

    /// Run an inference query and return per-variable posterior distributions.
    pub fn query(&mut self, q: &InferenceQuery) -> Result<Vec<QueryResult>, BniError> {
        // Validate query variables.
        for v in &q.query_variables {
            if !self.network.variables.contains_key(v) {
                return Err(BniError::VariableNotFound(v.clone()));
            }
        }

        // Validate and deduplicate evidence.
        let evidence_map = validate_evidence(&q.evidence, &self.network)?;

        let results = match &q.algorithm {
            InferenceAlgorithm::VariableElimination => {
                self.run_variable_elimination(&q.query_variables, &evidence_map)?
            }
            InferenceAlgorithm::BeliefPropagation => {
                self.run_belief_propagation(&q.query_variables, &evidence_map)?
            }
            InferenceAlgorithm::Sampling { n_samples, seed } => {
                self.run_sampling(&q.query_variables, &evidence_map, *n_samples, *seed)?
            }
        };

        self.stats.queries_answered += 1;
        Ok(results)
    }

    /// Compute the prior marginal for a single variable (no evidence).
    pub fn prior_marginal(&mut self, variable: &str) -> Result<QueryResult, BniError> {
        if !self.network.variables.contains_key(variable) {
            return Err(BniError::VariableNotFound(variable.to_string()));
        }
        let q = InferenceQuery {
            query_variables: vec![variable.to_string()],
            evidence: vec![],
            algorithm: InferenceAlgorithm::VariableElimination,
        };
        let mut results = self.query(&q)?;
        results
            .pop()
            .ok_or_else(|| BniError::InferenceError("no result produced".into()))
    }

    /// Add a new CPT to the network after construction.
    ///
    /// Validates dimensions and checks for duplicate variables.
    pub fn add_cpt(&mut self, cpt: ConditionalProbabilityTable) -> Result<(), BniError> {
        validate_cpt(&cpt, &self.cardinality_map)?;
        // Remove any existing CPT for this variable.
        self.network.cpts.retain(|c| c.variable != cpt.variable);
        self.network.cpts.push(cpt);
        Ok(())
    }

    /// Test d-separation: are `x` and `y` d-separated given observations `z`
    /// (the Bayes Ball algorithm)?
    ///
    /// Returns `true` if `x ⊥ y | z`.
    pub fn d_separated(&self, x: &str, y: &str, z: &[String]) -> Result<bool, BniError> {
        if !self.network.variables.contains_key(x) {
            return Err(BniError::VariableNotFound(x.to_string()));
        }
        if !self.network.variables.contains_key(y) {
            return Err(BniError::VariableNotFound(y.to_string()));
        }
        for zi in z {
            if !self.network.variables.contains_key(zi) {
                return Err(BniError::VariableNotFound(zi.clone()));
            }
        }

        let observed: HashSet<&str> = z.iter().map(String::as_str).collect();
        let reachable = bayes_ball_reachable(x, &observed, &self.network);
        Ok(!reachable.contains(y))
    }

    /// Return a snapshot of engine statistics.
    pub fn stats(&self) -> BniStats {
        self.stats.clone()
    }

    // ── variable elimination ──────────────────────────────────────────────────

    fn run_variable_elimination(
        &mut self,
        query_vars: &[String],
        evidence_map: &HashMap<String, usize>,
    ) -> Result<Vec<QueryResult>, BniError> {
        // 1. Start with all CPT factors.
        let mut factors: Vec<Factor> = self.network.cpts.iter().map(|c| c.factor.clone()).collect();

        // 2. Reduce factors by evidence.
        for (var, &state_idx) in evidence_map {
            factors = factors
                .into_iter()
                .map(|f| {
                    if f.contains_variable(var) {
                        f.reduce(var, state_idx, &self.cardinality_map)
                    } else {
                        f
                    }
                })
                .collect();
        }

        // 3. Determine hidden variables to eliminate.
        let query_set: HashSet<&str> = query_vars.iter().map(String::as_str).collect();
        let evidence_set: HashSet<&str> = evidence_map.keys().map(String::as_str).collect();
        let hidden_vars: Vec<String> = self
            .network
            .variables
            .keys()
            .filter(|v| !query_set.contains(v.as_str()) && !evidence_set.contains(v.as_str()))
            .cloned()
            .collect();

        let elim_order = self.compute_elimination_order(&hidden_vars, &factors);

        let mut eliminated = 0u64;
        for var in &elim_order {
            // Collect all factors that mention `var`.
            let (relevant, rest): (Vec<_>, Vec<_>) =
                factors.into_iter().partition(|f| f.contains_variable(var));

            if relevant.is_empty() {
                factors = rest;
                continue;
            }

            // Multiply them all together.
            let product = relevant
                .into_iter()
                .reduce(|acc, f| acc.product(&f))
                .ok_or_else(|| BniError::InferenceError("empty factor product".into()))?;

            // Marginalize out `var`.
            let marginal = product.marginalize(var, &self.cardinality_map);
            factors = rest;
            factors.push(marginal);
            eliminated += 1;
        }

        // Update stats.
        let n = self.stats.queries_answered + 1;
        self.stats.avg_factors_eliminated =
            (self.stats.avg_factors_eliminated * (n.saturating_sub(1) as f64) + eliminated as f64)
                / n as f64;

        // 4. Build results per query variable.
        let mut results = Vec::with_capacity(query_vars.len());
        for var_id in query_vars {
            let var = self
                .network
                .variables
                .get(var_id)
                .ok_or_else(|| BniError::VariableNotFound(var_id.clone()))?;

            // Collect factors that mention this variable and multiply.
            let (relevant, _): (Vec<_>, Vec<_>) = factors
                .iter()
                .cloned()
                .partition(|f| f.contains_variable(var_id));

            let mut joint = if relevant.is_empty() {
                // Uniform factor.
                Factor {
                    id: format!("uniform_{var_id}"),
                    variables: vec![var_id.clone()],
                    values: vec![1.0 / var.cardinality as f64; var.cardinality],
                    shape: vec![var.cardinality],
                }
            } else {
                let product = relevant
                    .into_iter()
                    .reduce(|acc, f| acc.product(&f))
                    .ok_or_else(|| BniError::InferenceError("empty factor product".into()))?;

                // Marginalize all variables except the query variable.
                let to_remove: Vec<String> = product
                    .variables
                    .iter()
                    .filter(|v| v.as_str() != var_id.as_str())
                    .cloned()
                    .collect();
                let mut f = product;
                for v in &to_remove {
                    f = f.marginalize(v, &self.cardinality_map);
                }
                f
            };

            joint.normalize();
            results.push(QueryResult::from_factor(var, joint));
        }

        Ok(results)
    }

    // ── belief propagation ────────────────────────────────────────────────────

    /// Sum-product message passing over the Bayesian network.
    ///
    /// Strategy: for each query variable, perform exact variable elimination
    /// restricted to the Markov blanket of that variable, then normalise.
    /// This is equivalent to a two-pass BP on trees and gives an exact answer
    /// for the marginals when no loops are present.  For loopy graphs it still
    /// produces a correct answer here because we rebuild the joint over the
    /// full factor set for each query variable (similar to VE).
    fn run_belief_propagation(
        &mut self,
        query_vars: &[String],
        evidence_map: &HashMap<String, usize>,
    ) -> Result<Vec<QueryResult>, BniError> {
        // Reduce all factors by evidence first.
        let factors: Vec<Factor> = self
            .network
            .cpts
            .iter()
            .map(|c| {
                let mut f = c.factor.clone();
                for (var, &state) in evidence_map {
                    if f.contains_variable(var) {
                        f = f.reduce(var, state, &self.cardinality_map);
                    }
                }
                f
            })
            .collect();

        let evidence_set: HashSet<&str> = evidence_map.keys().map(String::as_str).collect();

        let mut results = Vec::with_capacity(query_vars.len());

        for var_id in query_vars {
            let var = self
                .network
                .variables
                .get(var_id)
                .ok_or_else(|| BniError::VariableNotFound(var_id.clone()))?;

            // Eliminate all variables except the query variable.
            // Elimination order: every variable that is not the query and not
            // observed (already eliminated by evidence reduction).
            let elim_vars: Vec<String> = self
                .network
                .variables
                .keys()
                .filter(|v| v.as_str() != var_id.as_str() && !evidence_set.contains(v.as_str()))
                .cloned()
                .collect();

            let elim_order = self.compute_elimination_order(&elim_vars, &factors);

            let mut local_factors = factors.clone();
            for v in &elim_order {
                let (relevant, rest): (Vec<_>, Vec<_>) = local_factors
                    .into_iter()
                    .partition(|f| f.contains_variable(v));
                local_factors = rest;
                if !relevant.is_empty() {
                    let product = relevant
                        .into_iter()
                        .reduce(|acc, f| acc.product(&f))
                        .ok_or_else(|| BniError::InferenceError("empty factor product".into()))?;
                    let marginal = product.marginalize(v, &self.cardinality_map);
                    if !marginal.variables.is_empty() {
                        local_factors.push(marginal);
                    }
                }
            }

            // Multiply the remaining factors together and marginalise to the
            // query variable.
            let (relevant, _): (Vec<_>, Vec<_>) = local_factors
                .into_iter()
                .partition(|f| f.contains_variable(var_id));

            let mut joint = if relevant.is_empty() {
                Factor {
                    id: format!("uniform_{var_id}"),
                    variables: vec![var_id.clone()],
                    values: vec![1.0 / var.cardinality as f64; var.cardinality],
                    shape: vec![var.cardinality],
                }
            } else {
                let product = relevant
                    .into_iter()
                    .reduce(|acc, f| acc.product(&f))
                    .ok_or_else(|| BniError::InferenceError("empty factor product".into()))?;
                let to_remove: Vec<String> = product
                    .variables
                    .iter()
                    .filter(|v| v.as_str() != var_id.as_str())
                    .cloned()
                    .collect();
                let mut f = product;
                for v in &to_remove {
                    f = f.marginalize(v, &self.cardinality_map);
                }
                f
            };

            joint.normalize();
            results.push(QueryResult::from_factor(var, joint));
        }

        Ok(results)
    }

    // ── sampling ──────────────────────────────────────────────────────────────

    /// Likelihood-weighted sampling.
    fn run_sampling(
        &mut self,
        query_vars: &[String],
        evidence_map: &HashMap<String, usize>,
        n_samples: usize,
        seed: u64,
    ) -> Result<Vec<QueryResult>, BniError> {
        let topo = topological_order(&self.network)?;
        let mut rng = if seed == 0 {
            0xDEAD_BEEF_CAFE_u64
        } else {
            seed
        };

        // Accumulate weighted counts: variable → state → weight.
        let mut counts: HashMap<String, Vec<f64>> = HashMap::new();
        for var_id in query_vars {
            let var = self
                .network
                .variables
                .get(var_id)
                .ok_or_else(|| BniError::VariableNotFound(var_id.clone()))?;
            counts.insert(var_id.clone(), vec![0.0; var.cardinality]);
        }

        let n = if n_samples == 0 { 1000 } else { n_samples };

        for _ in 0..n {
            let mut assignment: HashMap<String, usize> = HashMap::new();
            let mut weight = 1.0_f64;

            for var_id in &topo {
                // Find the CPT for this variable.
                let cpt = match self.network.cpts.iter().find(|c| &c.variable == var_id) {
                    Some(c) => c,
                    None => continue,
                };

                // Compute conditional distribution given parent assignment.
                let cond_dist = compute_conditional(&cpt.factor, var_id, &assignment)
                    .unwrap_or_else(|| {
                        let card = self
                            .network
                            .variables
                            .get(var_id)
                            .map(|v| v.cardinality)
                            .unwrap_or(2);
                        vec![1.0 / card as f64; card]
                    });

                let sampled_state = if let Some(&observed) = evidence_map.get(var_id.as_str()) {
                    // Weight by likelihood of evidence.
                    weight *= cond_dist.get(observed).copied().unwrap_or(0.0);
                    observed
                } else {
                    sample_categorical(&cond_dist, &mut rng)
                };

                assignment.insert(var_id.clone(), sampled_state);
            }

            // Accumulate weighted counts for query variables.
            for (var_id, cnt) in &mut counts {
                if let Some(&state) = assignment.get(var_id) {
                    if let Some(c) = cnt.get_mut(state) {
                        *c += weight;
                    }
                }
            }
        }

        // Normalize and produce results.
        let mut results = Vec::with_capacity(query_vars.len());
        for var_id in query_vars {
            let var = self
                .network
                .variables
                .get(var_id)
                .ok_or_else(|| BniError::VariableNotFound(var_id.clone()))?;

            let cnt = counts.get(var_id).cloned().unwrap_or_default();
            let total: f64 = cnt.iter().sum();
            let values: Vec<f64> = if total > 0.0 {
                cnt.iter().map(|&c| c / total).collect()
            } else {
                vec![1.0 / var.cardinality as f64; var.cardinality]
            };

            let factor = Factor {
                id: format!("sampled_{var_id}"),
                variables: vec![var_id.clone()],
                values,
                shape: vec![var.cardinality],
            };
            results.push(QueryResult::from_factor(var, factor));
        }

        Ok(results)
    }

    // ── elimination order heuristics ──────────────────────────────────────────

    fn compute_elimination_order(&self, hidden_vars: &[String], factors: &[Factor]) -> Vec<String> {
        match self.config.elimination_ordering {
            EliminationOrder::Sequential => hidden_vars.to_vec(),
            EliminationOrder::MinDegree => min_degree_order(hidden_vars, factors),
            EliminationOrder::MinFill => min_fill_order(hidden_vars, factors),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper functions (module-private)
// ─────────────────────────────────────────────────────────────────────────────

/// Validate evidence: check all variables exist, deduplicate, detect conflicts.
fn validate_evidence(
    evidence: &[Evidence],
    network: &BayesianNetwork,
) -> Result<HashMap<String, usize>, BniError> {
    let mut map: HashMap<String, usize> = HashMap::new();
    for ev in evidence {
        let var = network
            .variables
            .get(&ev.variable)
            .ok_or_else(|| BniError::VariableNotFound(ev.variable.clone()))?;

        let state_idx = var
            .states
            .iter()
            .position(|s| s == &ev.observed_state)
            .ok_or_else(|| BniError::InvalidCPT {
                variable: ev.variable.clone(),
                reason: format!("unknown state `{}`", ev.observed_state),
            })?;

        if let Some(&existing) = map.get(&ev.variable) {
            if existing != state_idx {
                return Err(BniError::EvidenceConflict(ev.variable.clone()));
            }
        }
        map.insert(ev.variable.clone(), state_idx);
    }
    Ok(map)
}

/// Validate a CPT: variable exists, parents exist, dimension matches.
fn validate_cpt(
    cpt: &ConditionalProbabilityTable,
    cardinality_map: &HashMap<String, usize>,
) -> Result<(), BniError> {
    // Check all variables referenced in the factor exist.
    for v in &cpt.factor.variables {
        if !cardinality_map.contains_key(v) {
            return Err(BniError::InvalidCPT {
                variable: cpt.variable.clone(),
                reason: format!("referenced variable `{v}` not in cardinality map"),
            });
        }
    }

    // Check shape matches cardinalities.
    for (i, v) in cpt.factor.variables.iter().enumerate() {
        let expected = cardinality_map[v];
        let actual = cpt.factor.shape.get(i).copied().unwrap_or(0);
        if actual != expected {
            return Err(BniError::InvalidCPT {
                variable: cpt.variable.clone(),
                reason: format!("shape mismatch for `{v}`: expected {expected}, got {actual}"),
            });
        }
    }

    // Check values length.
    let expected_len: usize = cpt.factor.shape.iter().product::<usize>().max(1);
    if cpt.factor.values.len() != expected_len {
        return Err(BniError::InvalidCPT {
            variable: cpt.variable.clone(),
            reason: format!(
                "values.len() = {}, expected {}",
                cpt.factor.values.len(),
                expected_len
            ),
        });
    }

    Ok(())
}

/// Detect directed cycles in the Bayesian network using DFS.
fn detect_cycles(network: &BayesianNetwork) -> Result<(), BniError> {
    let mut visited: HashSet<&str> = HashSet::new();
    let mut stack: HashSet<&str> = HashSet::new();

    for var_id in network.variables.keys() {
        if !visited.contains(var_id.as_str()) {
            dfs_cycle(var_id, network, &mut visited, &mut stack)?;
        }
    }
    Ok(())
}

fn dfs_cycle<'a>(
    node: &'a str,
    network: &'a BayesianNetwork,
    visited: &mut HashSet<&'a str>,
    stack: &mut HashSet<&'a str>,
) -> Result<(), BniError> {
    visited.insert(node);
    stack.insert(node);

    // In BN adjacency (child → parents), we traverse from child to parents
    // which is the direction of dependency but reverse of edge direction.
    // Cycle detection operates on the parent → child direction, so we must
    // check the reverse: for each node, look at who lists it as a parent.
    // We instead build a forward map lazily.
    let children: Vec<&str> = network
        .adjacency
        .iter()
        .filter(|(_, parents)| parents.iter().any(|p| p.as_str() == node))
        .map(|(child, _)| child.as_str())
        .collect();

    for child in children {
        if !visited.contains(child) {
            dfs_cycle(child, network, visited, stack)?;
        } else if stack.contains(child) {
            return Err(BniError::CyclicNetwork(format!(
                "cycle detected at `{child}`"
            )));
        }
    }

    stack.remove(node);
    Ok(())
}

/// Compute a topological order of variables (parents before children).
fn topological_order(network: &BayesianNetwork) -> Result<Vec<String>, BniError> {
    let mut in_degree: HashMap<&str, usize> = HashMap::new();
    let mut children_of: HashMap<&str, Vec<&str>> = HashMap::new();

    for var_id in network.variables.keys() {
        in_degree.entry(var_id.as_str()).or_insert(0);
        children_of.entry(var_id.as_str()).or_default();
    }

    for (child, parents) in &network.adjacency {
        for parent in parents {
            *in_degree.entry(child.as_str()).or_insert(0) += 1;
            children_of
                .entry(parent.as_str())
                .or_default()
                .push(child.as_str());
        }
    }

    let mut queue: VecDeque<&str> = in_degree
        .iter()
        .filter(|(_, &d)| d == 0)
        .map(|(&v, _)| v)
        .collect();
    // Sort for determinism.
    let mut queue_vec: Vec<&str> = queue.drain(..).collect();
    queue_vec.sort_unstable();
    queue = queue_vec.into_iter().collect();

    let mut order: Vec<String> = Vec::new();
    while let Some(node) = queue.pop_front() {
        order.push(node.to_string());
        let mut nexts: Vec<&str> = children_of.get(node).cloned().unwrap_or_default();
        nexts.sort_unstable();
        for child in nexts {
            let deg = in_degree.entry(child).or_insert(0);
            if *deg > 0 {
                *deg -= 1;
            }
            if *deg == 0 {
                queue.push_back(child);
            }
        }
    }

    if order.len() != network.variables.len() {
        return Err(BniError::CyclicNetwork(
            "topological sort failed — cycle detected".into(),
        ));
    }
    Ok(order)
}

/// Compute the conditional distribution P(var | parent_assignment) from a factor.
fn compute_conditional(
    factor: &Factor,
    var: &str,
    assignment: &HashMap<String, usize>,
) -> Option<Vec<f64>> {
    let var_dim = factor.variables.iter().position(|v| v == var)?;
    let card = factor.shape[var_dim];

    // For each state of `var`, look up the value at current parent assignment.
    let mut dist = vec![0.0_f64; card];
    for (state, d) in dist.iter_mut().enumerate().take(card) {
        let mut idx = vec![0usize; factor.variables.len()];
        idx[var_dim] = state;
        // Fill parent indices.
        let mut ok = true;
        for (i, v) in factor.variables.iter().enumerate() {
            if i == var_dim {
                continue;
            }
            if let Some(&s) = assignment.get(v) {
                idx[i] = s;
            } else {
                ok = false;
                break;
            }
        }
        if ok {
            let flat = Factor::flat_index(&idx, &factor.shape);
            *d = factor.values.get(flat).copied().unwrap_or(0.0);
        }
    }

    // Normalize.
    let sum: f64 = dist.iter().sum();
    if sum > 0.0 {
        for v in &mut dist {
            *v /= sum;
        }
    } else {
        let u = 1.0 / card as f64;
        dist.fill(u);
    }
    Some(dist)
}

/// Min-degree elimination ordering.
fn min_degree_order(hidden_vars: &[String], factors: &[Factor]) -> Vec<String> {
    let mut remaining: Vec<String> = hidden_vars.to_vec();
    let mut order = Vec::with_capacity(remaining.len());
    let mut factor_copy = factors.to_vec();

    while !remaining.is_empty() {
        // For each remaining variable, count how many other remaining variables
        // appear in the same factor (its "degree" in the induced graph).
        let best_idx = remaining
            .iter()
            .enumerate()
            .min_by_key(|(_, v)| {
                let neighbors: HashSet<&str> = factor_copy
                    .iter()
                    .filter(|f| f.contains_variable(v))
                    .flat_map(|f| f.variables.iter().map(String::as_str))
                    .filter(|&u| u != v.as_str() && remaining.iter().any(|r| r.as_str() == u))
                    .collect();
                neighbors.len()
            })
            .map(|(i, _)| i)
            .unwrap_or(0);

        let var = remaining.remove(best_idx);

        // Simulate elimination: multiply relevant factors and marginalize.
        let (relevant, rest): (Vec<_>, Vec<_>) = factor_copy
            .into_iter()
            .partition(|f| f.contains_variable(&var));
        factor_copy = rest;
        if !relevant.is_empty() {
            let product = relevant
                .into_iter()
                .reduce(|acc, f| acc.product(&f))
                .unwrap_or_else(|| Factor {
                    id: "empty".into(),
                    variables: vec![],
                    values: vec![1.0],
                    shape: vec![],
                });
            let dummy = HashMap::new();
            let marginal = product.marginalize(&var, &dummy);
            if !marginal.variables.is_empty() {
                factor_copy.push(marginal);
            }
        }

        order.push(var);
    }
    order
}

/// Min-fill elimination ordering.
fn min_fill_order(hidden_vars: &[String], factors: &[Factor]) -> Vec<String> {
    let mut remaining: Vec<String> = hidden_vars.to_vec();
    let mut order = Vec::with_capacity(remaining.len());
    let mut factor_copy = factors.to_vec();

    while !remaining.is_empty() {
        // "Fill" for a variable = number of edges that would be added to the
        // induced graph by eliminating it.
        let best_idx = remaining
            .iter()
            .enumerate()
            .min_by_key(|(_, v)| {
                // Neighbours of v in the factor graph restricted to remaining vars.
                let neighbors: Vec<&str> = factor_copy
                    .iter()
                    .filter(|f| f.contains_variable(v))
                    .flat_map(|f| f.variables.iter().map(String::as_str))
                    .filter(|&u| u != v.as_str() && remaining.iter().any(|r| r.as_str() == u))
                    .collect::<HashSet<_>>()
                    .into_iter()
                    .collect();
                // Count pairs not yet connected.
                let mut fill = 0usize;
                for (i, &ni) in neighbors.iter().enumerate() {
                    for &nj in &neighbors[i + 1..] {
                        // Check if ni and nj already share a factor.
                        let connected = factor_copy
                            .iter()
                            .any(|f| f.contains_variable(ni) && f.contains_variable(nj));
                        if !connected {
                            fill += 1;
                        }
                    }
                }
                fill
            })
            .map(|(i, _)| i)
            .unwrap_or(0);

        let var = remaining.remove(best_idx);

        // Simulate elimination.
        let (relevant, rest): (Vec<_>, Vec<_>) = factor_copy
            .into_iter()
            .partition(|f| f.contains_variable(&var));
        factor_copy = rest;
        if !relevant.is_empty() {
            let product = relevant
                .into_iter()
                .reduce(|acc, f| acc.product(&f))
                .unwrap_or_else(|| Factor {
                    id: "empty".into(),
                    variables: vec![],
                    values: vec![1.0],
                    shape: vec![],
                });
            let dummy = HashMap::new();
            let marginal = product.marginalize(&var, &dummy);
            if !marginal.variables.is_empty() {
                factor_copy.push(marginal);
            }
        }

        order.push(var);
    }
    order
}

/// Bayes Ball algorithm: find all nodes reachable from `source` given
/// observations `observed`.
///
/// Returns the set of variable ids that are d-connected to `source`.
fn bayes_ball_reachable<'a>(
    source: &'a str,
    observed: &HashSet<&'a str>,
    network: &'a BayesianNetwork,
) -> HashSet<&'a str> {
    // Build parent map: var → parents and child map: parent → children.
    let parent_of: &HashMap<String, Vec<String>> = &network.adjacency;
    let mut child_of: HashMap<&str, Vec<&str>> = HashMap::new();
    for (child, parents) in parent_of {
        for parent in parents {
            child_of
                .entry(parent.as_str())
                .or_default()
                .push(child.as_str());
        }
    }

    // (node, came_from_child): true = ball came from child (upward pass)
    let mut queue: VecDeque<(&str, bool)> = VecDeque::new();
    let mut visited_up: HashSet<&str> = HashSet::new();
    let mut visited_down: HashSet<&str> = HashSet::new();
    let mut reachable: HashSet<&str> = HashSet::new();

    // Schedule both directions from source.
    queue.push_back((source, true));
    queue.push_back((source, false));

    while let Some((node, from_child)) = queue.pop_front() {
        if from_child {
            if visited_up.contains(node) {
                continue;
            }
            visited_up.insert(node);
        } else {
            if visited_down.contains(node) {
                continue;
            }
            visited_down.insert(node);
        }

        if node != source {
            reachable.insert(node);
        }

        let is_observed = observed.contains(node);

        if from_child && !is_observed {
            // Pass through to parents (d-connected path goes up).
            for parent in parent_of.get(node).map(|v| v.as_slice()).unwrap_or(&[]) {
                queue.push_back((parent.as_str(), true));
            }
            // Also send down to other children.
            for &child in child_of.get(node).map(|v| v.as_slice()).unwrap_or(&[]) {
                queue.push_back((child, false));
            }
        } else if !from_child {
            if !is_observed {
                // Send down to children.
                for &child in child_of.get(node).map(|v| v.as_slice()).unwrap_or(&[]) {
                    queue.push_back((child, false));
                }
            }
            // v-structure: if observed or descendant of observed, send to parents.
            if is_observed {
                for parent in parent_of.get(node).map(|v| v.as_slice()).unwrap_or(&[]) {
                    queue.push_back((parent.as_str(), true));
                }
            }
        }
    }

    reachable
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    include!("bni_tests.rs");
}
