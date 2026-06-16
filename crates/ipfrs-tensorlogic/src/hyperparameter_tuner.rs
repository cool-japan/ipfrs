//! Hyperparameter Tuner — Bayesian optimization and random/grid search.
//!
//! Provides [`HyperparameterTuner`] supporting:
//! - **Random Search** — sample random configurations with xorshift64 PRNG
//! - **Grid Search** — enumerate all discrete/categorical/continuous combinations
//! - **Bayesian Optimization** — Gaussian process surrogate with UCB acquisition
//!
//! # Example
//!
//! ```
//! use ipfrs_tensorlogic::{
//!     HyperparameterTuner, TunerConfig, HpSpec, HpType, TuningStrategy,
//! };
//!
//! let spec = HpSpec {
//!     name: "lr".to_string(),
//!     hp_type: HpType::Continuous { lo: 1e-4, hi: 1e-1 },
//!     log_scale: true,
//! };
//! let config = TunerConfig {
//!     specs: vec![spec],
//!     maximize: false,
//!     seed: 42,
//! };
//! let mut tuner = HyperparameterTuner::new(config);
//! let mut rng = 42u64;
//! let results = tuner.run_random_search(5, |cfg| {
//!     // Simulated scorer: extract lr, return a dummy loss
//!     use ipfrs_tensorlogic::HpValue;
//!     if let Some(HpValue::Float(lr)) = cfg.get("lr") {
//!         lr.abs()
//!     } else {
//!         f64::MAX
//!     }
//! }, &mut rng);
//! assert_eq!(results.len(), 5);
//! ```

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// xorshift64 PRNG
// ---------------------------------------------------------------------------

#[inline]
fn xorshift64(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

/// Generate a float in [0.0, 1.0).
#[inline]
fn rng_f64(state: &mut u64) -> f64 {
    // Use upper 53 bits for f64 mantissa precision.
    let bits = xorshift64(state) >> 11;
    bits as f64 / (1u64 << 53) as f64
}

/// Generate an integer in [lo, hi] (inclusive).
#[inline]
fn rng_i64_range(state: &mut u64, lo: i64, hi: i64) -> i64 {
    if lo >= hi {
        return lo;
    }
    let span = (hi - lo + 1) as u64;
    lo + (xorshift64(state) % span) as i64
}

/// Generate a usize in [lo, hi).
#[inline]
fn rng_usize_range(state: &mut u64, lo: usize, hi: usize) -> usize {
    if lo >= hi {
        return lo;
    }
    lo + (xorshift64(state) as usize % (hi - lo))
}

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

/// Errors that can arise during hyperparameter tuning.
#[derive(Debug, Clone, PartialEq)]
pub enum HpTunerError {
    /// No hyperparameter specs have been added.
    NoSpecs,
    /// The history is empty (no trials have been recorded).
    NoHistory,
    /// A specification is malformed (e.g. inverted bounds, empty categories).
    InvalidSpec(String),
}

impl fmt::Display for HpTunerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HpTunerError::NoSpecs => write!(f, "no hyperparameter specs defined"),
            HpTunerError::NoHistory => write!(f, "no trial history available"),
            HpTunerError::InvalidSpec(msg) => write!(f, "invalid spec: {}", msg),
        }
    }
}

impl std::error::Error for HpTunerError {}

// ---------------------------------------------------------------------------
// HpType
// ---------------------------------------------------------------------------

/// The domain type of a single hyperparameter.
#[derive(Debug, Clone, PartialEq)]
pub enum HpType {
    /// Real-valued parameter in `[lo, hi]`.
    Continuous { lo: f64, hi: f64 },
    /// Integer-valued parameter in `[lo, hi]` (inclusive).
    Discrete { lo: i64, hi: i64 },
    /// One of a fixed set of string choices.
    Categorical { choices: Vec<String> },
}

// ---------------------------------------------------------------------------
// HpSpec
// ---------------------------------------------------------------------------

/// Specification for a single hyperparameter.
#[derive(Debug, Clone)]
pub struct HpSpec {
    /// Name of the hyperparameter (used as the key in [`HpConfig`]).
    pub name: String,
    /// Domain type.
    pub hp_type: HpType,
    /// If `true` and the type is `Continuous`, sample in log space
    /// (transform lo/hi with `ln`, sample uniformly, then `exp`).
    pub log_scale: bool,
}

impl HpSpec {
    /// Validate the spec, returning an error description on failure.
    pub fn validate(&self) -> Result<(), HpTunerError> {
        match &self.hp_type {
            HpType::Continuous { lo, hi } => {
                if lo >= hi {
                    return Err(HpTunerError::InvalidSpec(format!(
                        "Continuous spec '{}': lo ({}) must be < hi ({})",
                        self.name, lo, hi
                    )));
                }
                if self.log_scale && *lo <= 0.0 {
                    return Err(HpTunerError::InvalidSpec(format!(
                        "Continuous spec '{}': log_scale requires lo > 0, got {}",
                        self.name, lo
                    )));
                }
            }
            HpType::Discrete { lo, hi } => {
                if lo > hi {
                    return Err(HpTunerError::InvalidSpec(format!(
                        "Discrete spec '{}': lo ({}) must be <= hi ({})",
                        self.name, lo, hi
                    )));
                }
            }
            HpType::Categorical { choices } => {
                if choices.is_empty() {
                    return Err(HpTunerError::InvalidSpec(format!(
                        "Categorical spec '{}': choices must not be empty",
                        self.name
                    )));
                }
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// HpValue
// ---------------------------------------------------------------------------

/// The value of a single hyperparameter in a concrete configuration.
#[derive(Debug, Clone, PartialEq)]
pub enum HpValue {
    /// Real-valued (Continuous or Continuous log-scale).
    Float(f64),
    /// Integer-valued (Discrete).
    Int(i64),
    /// One of a categorical set.
    Choice(String),
}

impl fmt::Display for HpValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HpValue::Float(v) => write!(f, "{:.6e}", v),
            HpValue::Int(v) => write!(f, "{}", v),
            HpValue::Choice(s) => write!(f, "{}", s),
        }
    }
}

// ---------------------------------------------------------------------------
// HpConfig
// ---------------------------------------------------------------------------

/// A complete hyperparameter configuration (map from name to value).
#[derive(Debug, Clone, Default)]
pub struct HpConfig(pub HashMap<String, HpValue>);

impl HpConfig {
    /// Create an empty config.
    pub fn new() -> Self {
        HpConfig(HashMap::new())
    }

    /// Get the value for a parameter by name.
    pub fn get(&self, name: &str) -> Option<&HpValue> {
        self.0.get(name)
    }

    /// Insert a name→value pair.
    pub fn insert(&mut self, name: String, value: HpValue) {
        self.0.insert(name, value);
    }

    /// Number of parameters in this config.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns true if the config has no parameters.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Encode continuous parameters as a vector for distance calculations.
    /// Discrete values are cast to f64; categorical values are ignored.
    pub fn continuous_vec(&self, specs: &[HpSpec]) -> Vec<f64> {
        let mut result = Vec::with_capacity(specs.len());
        for spec in specs {
            match self.0.get(&spec.name) {
                Some(HpValue::Float(v)) => result.push(*v),
                Some(HpValue::Int(v)) => result.push(*v as f64),
                _ => {}
            }
        }
        result
    }
}

// ---------------------------------------------------------------------------
// TuningResult
// ---------------------------------------------------------------------------

/// The result of a single hyperparameter tuning trial.
#[derive(Debug, Clone)]
pub struct TuningResult {
    /// Unique sequential trial identifier (starts at 0).
    pub trial_id: u64,
    /// The hyperparameter configuration that was evaluated.
    pub config: HpConfig,
    /// The score returned by the objective function (higher = better when `maximize = true`).
    pub score: f64,
    /// Caller-supplied timestamp (e.g. seconds since epoch, or step number).
    pub timestamp: u64,
}

// ---------------------------------------------------------------------------
// TuningStrategy
// ---------------------------------------------------------------------------

/// How the tuner explores the hyperparameter space.
#[derive(Debug, Clone)]
pub enum TuningStrategy {
    /// Draw `n_trials` random configurations independently.
    RandomSearch { n_trials: u32 },
    /// Enumerate every combination.  Continuous parameters get 5 evenly-spaced values.
    GridSearch,
    /// Gaussian process surrogate with UCB acquisition.
    BayesianOptimization {
        n_trials: u32,
        /// Number of random trials used to warm-start the surrogate.
        n_initial: u32,
        /// Weight on the standard-deviation term in the UCB formula (exploration factor).
        exploration_weight: f64,
    },
}

// ---------------------------------------------------------------------------
// TunerConfig
// ---------------------------------------------------------------------------

/// Configuration for [`HyperparameterTuner`].
#[derive(Debug, Clone)]
pub struct TunerConfig {
    /// Hyperparameter specifications.
    pub specs: Vec<HpSpec>,
    /// If `true`, higher score = better.  If `false`, lower score = better.
    pub maximize: bool,
    /// Seed for the xorshift64 PRNG (initialised once; individual calls receive
    /// a mutable reference so callers can control reproducibility).
    pub seed: u64,
}

// ---------------------------------------------------------------------------
// TunerStats
// ---------------------------------------------------------------------------

/// Aggregate statistics over all recorded trials.
#[derive(Debug, Clone, PartialEq)]
pub struct TunerStats {
    pub total_trials: usize,
    pub best_score: f64,
    pub worst_score: f64,
    pub avg_score: f64,
    /// Fraction of trials that strictly improved on the running best.
    pub improvement_rate: f64,
}

// ---------------------------------------------------------------------------
// HyperparameterTuner
// ---------------------------------------------------------------------------

/// Hyperparameter tuner supporting random search, grid search, and simplified
/// Bayesian optimization with UCB acquisition.
#[derive(Debug)]
pub struct HyperparameterTuner {
    pub config: TunerConfig,
    pub history: Vec<TuningResult>,
    pub next_trial_id: u64,
}

impl HyperparameterTuner {
    /// Create a new tuner with the given configuration.
    pub fn new(config: TunerConfig) -> Self {
        HyperparameterTuner {
            config,
            history: Vec::new(),
            next_trial_id: 0,
        }
    }

    /// Add a hyperparameter specification to the tuner (builder-style).
    pub fn add_spec(&mut self, spec: HpSpec) -> &mut Self {
        self.config.specs.push(spec);
        self
    }

    // -----------------------------------------------------------------------
    // Sampling helpers
    // -----------------------------------------------------------------------

    /// Sample a single value for one spec using the xorshift64 PRNG.
    pub fn sample_value(spec: &HpSpec, rng: &mut u64) -> HpValue {
        match &spec.hp_type {
            HpType::Continuous { lo, hi } => {
                if spec.log_scale {
                    let log_lo = lo.ln();
                    let log_hi = hi.ln();
                    let log_val = log_lo + rng_f64(rng) * (log_hi - log_lo);
                    HpValue::Float(log_val.exp())
                } else {
                    HpValue::Float(lo + rng_f64(rng) * (hi - lo))
                }
            }
            HpType::Discrete { lo, hi } => HpValue::Int(rng_i64_range(rng, *lo, *hi)),
            HpType::Categorical { choices } => {
                let idx = rng_usize_range(rng, 0, choices.len());
                HpValue::Choice(choices[idx].clone())
            }
        }
    }

    /// Sample a complete configuration (one value per spec).
    pub fn sample_config(&self, rng: &mut u64) -> HpConfig {
        let mut cfg = HpConfig::new();
        for spec in &self.config.specs {
            let val = Self::sample_value(spec, rng);
            cfg.insert(spec.name.clone(), val);
        }
        cfg
    }

    // -----------------------------------------------------------------------
    // Recording
    // -----------------------------------------------------------------------

    /// Record an evaluation result.  Returns the assigned trial_id.
    pub fn record_result(&mut self, config: HpConfig, score: f64, now: u64) -> u64 {
        let id = self.next_trial_id;
        self.history.push(TuningResult {
            trial_id: id,
            config,
            score,
            timestamp: now,
        });
        self.next_trial_id += 1;
        id
    }

    // -----------------------------------------------------------------------
    // Best result
    // -----------------------------------------------------------------------

    /// Return a reference to the best trial in history, or `None` if empty.
    pub fn best_config(&self) -> Option<&TuningResult> {
        if self.history.is_empty() {
            return None;
        }
        if self.config.maximize {
            self.history.iter().max_by(|a, b| {
                a.score
                    .partial_cmp(&b.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
        } else {
            self.history.iter().min_by(|a, b| {
                a.score
                    .partial_cmp(&b.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
        }
    }

    // -----------------------------------------------------------------------
    // UCB-based suggest_next (used by Bayesian strategy)
    // -----------------------------------------------------------------------

    /// Evaluate the GP-UCB acquisition for a candidate config given the
    /// current history.  Returns `(mean, std_dev, ucb)`.
    fn ucb_for_candidate(&self, candidate: &HpConfig, exploration_weight: f64) -> (f64, f64, f64) {
        if self.history.is_empty() {
            return (0.0, 1.0, exploration_weight);
        }

        // Compute Euclidean distances from candidate to each history point.
        let specs = &self.config.specs;
        let cand_vec = candidate.continuous_vec(specs);

        // Collect (distance, score) pairs.
        let mut weighted_scores: Vec<(f64, f64)> = self
            .history
            .iter()
            .map(|r| {
                let hist_vec = r.config.continuous_vec(specs);
                let dist = euclidean_dist(&cand_vec, &hist_vec);
                (dist, r.score)
            })
            .collect();

        // Sort by distance so nearest neighbors come first.
        weighted_scores.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        // Use up to k=5 nearest neighbors.
        let k = weighted_scores.len().min(5);
        let neighbors = &weighted_scores[..k];

        let scores: Vec<f64> = neighbors.iter().map(|(_, s)| *s).collect();
        let mean = scores.iter().sum::<f64>() / scores.len() as f64;

        let variance = scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / scores.len() as f64;
        let std_dev = variance.sqrt();

        let epsilon = 1e-8;
        let ucb = mean + exploration_weight * std_dev + epsilon;
        (mean, std_dev, ucb)
    }

    /// Suggest the next configuration to evaluate.
    ///
    /// - For `RandomSearch` / no strategy context: sample a random config.
    /// - For Bayesian: draw 10 random candidates, evaluate UCB for each,
    ///   and return the one with the highest UCB.
    pub fn suggest_next(&self, rng: &mut u64) -> HpConfig {
        // We always have access to history; choose UCB-guided selection if we
        // have prior results and a meaningful number of specs, otherwise random.
        if self.history.is_empty() || self.config.specs.is_empty() {
            return self.sample_config(rng);
        }

        // Default exploration weight when called outside a BayesianOptimization run.
        let exploration_weight = 1.0_f64;

        let n_candidates = 10_usize;
        let mut best_cfg = self.sample_config(rng);
        let (_, _, mut best_ucb) = self.ucb_for_candidate(&best_cfg, exploration_weight);

        for _ in 1..n_candidates {
            let candidate = self.sample_config(rng);
            let (_, _, ucb) = self.ucb_for_candidate(&candidate, exploration_weight);
            if ucb > best_ucb {
                best_ucb = ucb;
                best_cfg = candidate;
            }
        }
        best_cfg
    }

    // -----------------------------------------------------------------------
    // Random search
    // -----------------------------------------------------------------------

    /// Run `n_trials` random evaluations using the provided scorer, record all
    /// results, and return them sorted by score (best first).
    pub fn run_random_search(
        &mut self,
        n_trials: u32,
        mut scorer: impl FnMut(&HpConfig) -> f64,
        rng: &mut u64,
    ) -> Vec<TuningResult> {
        for _ in 0..n_trials {
            let cfg = self.sample_config(rng);
            let score = scorer(&cfg);
            self.record_result(cfg, score, 0);
        }
        let mut results = self.history.clone();
        sort_results(&mut results, self.config.maximize);
        results
    }

    // -----------------------------------------------------------------------
    // Grid search
    // -----------------------------------------------------------------------

    /// Enumerate all grid configurations.
    ///
    /// - `Continuous` → 5 evenly-spaced values across `[lo, hi]`
    /// - `Discrete`   → every integer in `[lo, hi]`
    /// - `Categorical` → every choice
    pub fn grid_configs(&self) -> Vec<HpConfig> {
        if self.config.specs.is_empty() {
            return Vec::new();
        }

        // Build a list of candidate values per spec.
        let value_lists: Vec<Vec<HpValue>> =
            self.config.specs.iter().map(spec_grid_values).collect();

        // Compute Cartesian product iteratively.
        let mut configs: Vec<HpConfig> = vec![HpConfig::new()];
        for (spec, values) in self.config.specs.iter().zip(value_lists.iter()) {
            let mut next_configs: Vec<HpConfig> = Vec::new();
            for existing in &configs {
                for val in values {
                    let mut new_cfg = existing.clone();
                    new_cfg.insert(spec.name.clone(), val.clone());
                    next_configs.push(new_cfg);
                }
            }
            configs = next_configs;
        }
        configs
    }

    /// Enumerate all grid configurations, evaluate each with `scorer`, record
    /// results, and return them sorted best-first.
    pub fn run_grid_search(
        &mut self,
        mut scorer: impl FnMut(&HpConfig) -> f64,
    ) -> Vec<TuningResult> {
        let configs = self.grid_configs();
        for cfg in configs {
            let score = scorer(&cfg);
            self.record_result(cfg, score, 0);
        }
        let mut results = self.history.clone();
        sort_results(&mut results, self.config.maximize);
        results
    }

    // -----------------------------------------------------------------------
    // Bayesian optimization
    // -----------------------------------------------------------------------

    /// Run Bayesian (UCB) optimization.  Draws `n_initial` random configs to
    /// warm-start, then proposes `n_trials - n_initial` UCB-guided candidates.
    pub fn run_bayesian(
        &mut self,
        n_trials: u32,
        n_initial: u32,
        exploration_weight: f64,
        mut scorer: impl FnMut(&HpConfig) -> f64,
        rng: &mut u64,
    ) -> Vec<TuningResult> {
        let initial = n_initial.min(n_trials);

        // Warm-start phase.
        for _ in 0..initial {
            let cfg = self.sample_config(rng);
            let score = scorer(&cfg);
            self.record_result(cfg, score, 0);
        }

        // Bayesian exploitation/exploration phase.
        for _ in initial..n_trials {
            let cfg = self.suggest_next_bayesian(rng, exploration_weight);
            let score = scorer(&cfg);
            self.record_result(cfg, score, 0);
        }

        let mut results = self.history.clone();
        sort_results(&mut results, self.config.maximize);
        results
    }

    /// Suggest the next config using UCB with a given exploration weight.
    fn suggest_next_bayesian(&self, rng: &mut u64, exploration_weight: f64) -> HpConfig {
        if self.history.is_empty() || self.config.specs.is_empty() {
            return self.sample_config(rng);
        }

        let n_candidates = 10_usize;
        let mut best_cfg = self.sample_config(rng);
        let (_, _, mut best_ucb) = self.ucb_for_candidate(&best_cfg, exploration_weight);

        for _ in 1..n_candidates {
            let candidate = self.sample_config(rng);
            let (_, _, ucb) = self.ucb_for_candidate(&candidate, exploration_weight);
            if self.config.maximize {
                if ucb > best_ucb {
                    best_ucb = ucb;
                    best_cfg = candidate;
                }
            } else {
                // For minimization, prefer low mean − exploration * std.
                let (mean, std_dev, _) = self.ucb_for_candidate(&candidate, exploration_weight);
                let lcb = mean - exploration_weight * std_dev;
                let (best_mean, best_std, _) =
                    self.ucb_for_candidate(&best_cfg, exploration_weight);
                let best_lcb = best_mean - exploration_weight * best_std;
                if lcb < best_lcb {
                    best_ucb = ucb;
                    best_cfg = candidate;
                }
            }
        }
        best_cfg
    }

    // -----------------------------------------------------------------------
    // Importance scores
    // -----------------------------------------------------------------------

    /// Compute a simple variance-of-scores importance score for each parameter.
    ///
    /// For each parameter, group history trials into equal-width buckets across
    /// the parameter's range and compute the variance of per-bucket mean scores.
    /// Returns `0.0` for any parameter if there are fewer than 2 trials.
    pub fn importance_scores(&self) -> HashMap<String, f64> {
        let mut result = HashMap::new();
        if self.history.len() < 2 {
            for spec in &self.config.specs {
                result.insert(spec.name.clone(), 0.0);
            }
            return result;
        }

        for spec in &self.config.specs {
            let importance = compute_importance(spec, &self.history);
            result.insert(spec.name.clone(), importance);
        }
        result
    }

    // -----------------------------------------------------------------------
    // Stats
    // -----------------------------------------------------------------------

    /// Aggregate statistics over all recorded trials.
    pub fn stats(&self) -> TunerStats {
        if self.history.is_empty() {
            return TunerStats {
                total_trials: 0,
                best_score: 0.0,
                worst_score: 0.0,
                avg_score: 0.0,
                improvement_rate: 0.0,
            };
        }

        let scores: Vec<f64> = self.history.iter().map(|r| r.score).collect();
        let best_score = if self.config.maximize {
            scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        } else {
            scores.iter().cloned().fold(f64::INFINITY, f64::min)
        };
        let worst_score = if self.config.maximize {
            scores.iter().cloned().fold(f64::INFINITY, f64::min)
        } else {
            scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        };
        let avg_score = scores.iter().sum::<f64>() / scores.len() as f64;

        let improvement_rate = compute_improvement_rate(&scores, self.config.maximize);

        TunerStats {
            total_trials: self.history.len(),
            best_score,
            worst_score,
            avg_score,
            improvement_rate,
        }
    }
}

// ---------------------------------------------------------------------------
// Free helper functions
// ---------------------------------------------------------------------------

/// Euclidean distance between two vectors (handles unequal lengths via zip).
fn euclidean_dist(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Sort results best-first (descending if maximize, ascending if minimize).
fn sort_results(results: &mut [TuningResult], maximize: bool) {
    if maximize {
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    } else {
        results.sort_by(|a, b| {
            a.score
                .partial_cmp(&b.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }
}

/// Build the grid of candidate values for a single spec.
fn spec_grid_values(spec: &HpSpec) -> Vec<HpValue> {
    const N_CONTINUOUS: usize = 5;
    match &spec.hp_type {
        HpType::Continuous { lo, hi } => (0..N_CONTINUOUS)
            .map(|i| {
                let t = i as f64 / (N_CONTINUOUS - 1) as f64;
                if spec.log_scale && *lo > 0.0 {
                    let log_lo = lo.ln();
                    let log_hi = hi.ln();
                    HpValue::Float((log_lo + t * (log_hi - log_lo)).exp())
                } else {
                    HpValue::Float(lo + t * (hi - lo))
                }
            })
            .collect(),
        HpType::Discrete { lo, hi } => (*lo..=*hi).map(HpValue::Int).collect(),
        HpType::Categorical { choices } => {
            choices.iter().map(|c| HpValue::Choice(c.clone())).collect()
        }
    }
}

/// Compute the variance-of-bucket-means importance score for a single parameter.
fn compute_importance(spec: &HpSpec, history: &[TuningResult]) -> f64 {
    const N_BUCKETS: usize = 5;

    // Collect (bucket_index, score) pairs.
    let mut bucket_scores: Vec<Vec<f64>> = vec![Vec::new(); N_BUCKETS];

    for result in history {
        let bucket_idx = match result.config.get(&spec.name) {
            Some(HpValue::Float(v)) => {
                // Bucket by value range.
                if let HpType::Continuous { lo, hi } = &spec.hp_type {
                    let range = hi - lo;
                    if range <= 0.0 {
                        0
                    } else {
                        let normalized = (v - lo) / range;
                        let idx = (normalized * N_BUCKETS as f64) as usize;
                        idx.min(N_BUCKETS - 1)
                    }
                } else {
                    0
                }
            }
            Some(HpValue::Int(v)) => {
                if let HpType::Discrete { lo, hi } = &spec.hp_type {
                    let range = (hi - lo) as f64;
                    if range <= 0.0 {
                        0
                    } else {
                        let normalized = (v - lo) as f64 / range;
                        let idx = (normalized * N_BUCKETS as f64) as usize;
                        idx.min(N_BUCKETS - 1)
                    }
                } else {
                    0
                }
            }
            Some(HpValue::Choice(c)) => {
                if let HpType::Categorical { choices } = &spec.hp_type {
                    choices.iter().position(|ch| ch == c).unwrap_or(0) % N_BUCKETS
                } else {
                    0
                }
            }
            None => continue,
        };
        bucket_scores[bucket_idx].push(result.score);
    }

    // Compute per-bucket means (skip empty buckets).
    let means: Vec<f64> = bucket_scores
        .iter()
        .filter(|b| !b.is_empty())
        .map(|b| b.iter().sum::<f64>() / b.len() as f64)
        .collect();

    if means.len() < 2 {
        return 0.0;
    }

    // Variance of bucket means.
    let mean_of_means = means.iter().sum::<f64>() / means.len() as f64;
    means
        .iter()
        .map(|m| (m - mean_of_means).powi(2))
        .sum::<f64>()
        / means.len() as f64
}

/// Compute the fraction of trials that strictly improved the running best.
fn compute_improvement_rate(scores: &[f64], maximize: bool) -> f64 {
    if scores.is_empty() {
        return 0.0;
    }
    let mut improvements = 0usize;
    let mut running_best = scores[0];
    // First trial is never counted as an improvement over nothing.
    for &s in scores.iter().skip(1) {
        let improved = if maximize {
            s > running_best
        } else {
            s < running_best
        };
        if improved {
            improvements += 1;
            running_best = s;
        }
    }
    improvements as f64 / scores.len() as f64
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use crate::hyperparameter_tuner::{
        compute_improvement_rate, euclidean_dist, rng_f64, rng_i64_range, rng_usize_range,
        sort_results, spec_grid_values, xorshift64, HpConfig, HpSpec, HpTunerError, HpType,
        HpValue, HyperparameterTuner, TunerConfig, TuningResult,
    };

    // ------------------------------------------------------------------
    // PRNG tests
    // ------------------------------------------------------------------

    #[test]
    fn test_xorshift64_non_zero() {
        let mut state = 1u64;
        let v = xorshift64(&mut state);
        assert_ne!(v, 0);
    }

    #[test]
    fn test_xorshift64_different_values() {
        let mut state = 12345u64;
        let a = xorshift64(&mut state);
        let b = xorshift64(&mut state);
        assert_ne!(a, b);
    }

    #[test]
    fn test_rng_f64_in_range() {
        let mut state = 99u64;
        for _ in 0..1000 {
            let v = rng_f64(&mut state);
            assert!((0.0..1.0).contains(&v), "out of [0,1): {}", v);
        }
    }

    #[test]
    fn test_rng_i64_range_bounds() {
        let mut state = 7u64;
        for _ in 0..500 {
            let v = rng_i64_range(&mut state, 3, 7);
            assert!((3..=7).contains(&v), "out of [3,7]: {}", v);
        }
    }

    #[test]
    fn test_rng_i64_range_equal_bounds() {
        let mut state = 1u64;
        assert_eq!(rng_i64_range(&mut state, 5, 5), 5);
    }

    #[test]
    fn test_rng_usize_range_bounds() {
        let mut state = 42u64;
        for _ in 0..500 {
            let v = rng_usize_range(&mut state, 0, 4);
            assert!(v < 4, "out of [0,4): {}", v);
        }
    }

    // ------------------------------------------------------------------
    // HpSpec validation
    // ------------------------------------------------------------------

    #[test]
    fn test_spec_validate_continuous_ok() {
        let spec = HpSpec {
            name: "lr".into(),
            hp_type: HpType::Continuous { lo: 1e-4, hi: 1e-1 },
            log_scale: false,
        };
        assert!(spec.validate().is_ok());
    }

    #[test]
    fn test_spec_validate_continuous_inverted_bounds() {
        let spec = HpSpec {
            name: "lr".into(),
            hp_type: HpType::Continuous { lo: 1.0, hi: 0.0 },
            log_scale: false,
        };
        assert!(matches!(spec.validate(), Err(HpTunerError::InvalidSpec(_))));
    }

    #[test]
    fn test_spec_validate_log_scale_nonpositive_lo() {
        let spec = HpSpec {
            name: "lr".into(),
            hp_type: HpType::Continuous { lo: 0.0, hi: 1.0 },
            log_scale: true,
        };
        assert!(matches!(spec.validate(), Err(HpTunerError::InvalidSpec(_))));
    }

    #[test]
    fn test_spec_validate_discrete_ok() {
        let spec = HpSpec {
            name: "layers".into(),
            hp_type: HpType::Discrete { lo: 1, hi: 5 },
            log_scale: false,
        };
        assert!(spec.validate().is_ok());
    }

    #[test]
    fn test_spec_validate_discrete_inverted() {
        let spec = HpSpec {
            name: "layers".into(),
            hp_type: HpType::Discrete { lo: 5, hi: 1 },
            log_scale: false,
        };
        assert!(matches!(spec.validate(), Err(HpTunerError::InvalidSpec(_))));
    }

    #[test]
    fn test_spec_validate_categorical_ok() {
        let spec = HpSpec {
            name: "optim".into(),
            hp_type: HpType::Categorical {
                choices: vec!["adam".into(), "sgd".into()],
            },
            log_scale: false,
        };
        assert!(spec.validate().is_ok());
    }

    #[test]
    fn test_spec_validate_categorical_empty() {
        let spec = HpSpec {
            name: "optim".into(),
            hp_type: HpType::Categorical { choices: vec![] },
            log_scale: false,
        };
        assert!(matches!(spec.validate(), Err(HpTunerError::InvalidSpec(_))));
    }

    // ------------------------------------------------------------------
    // HpConfig
    // ------------------------------------------------------------------

    #[test]
    fn test_hp_config_insert_and_get() {
        let mut cfg = HpConfig::new();
        cfg.insert("lr".into(), HpValue::Float(0.01));
        assert_eq!(cfg.get("lr"), Some(&HpValue::Float(0.01)));
        assert_eq!(cfg.get("missing"), None);
    }

    #[test]
    fn test_hp_config_len_is_empty() {
        let cfg = HpConfig::new();
        assert!(cfg.is_empty());
        assert_eq!(cfg.len(), 0);
        let mut cfg2 = HpConfig::new();
        cfg2.insert("x".into(), HpValue::Int(1));
        assert!(!cfg2.is_empty());
        assert_eq!(cfg2.len(), 1);
    }

    // ------------------------------------------------------------------
    // sample_value
    // ------------------------------------------------------------------

    #[test]
    fn test_sample_continuous_in_range() {
        let spec = HpSpec {
            name: "lr".into(),
            hp_type: HpType::Continuous { lo: 0.0, hi: 1.0 },
            log_scale: false,
        };
        let mut rng = 1234u64;
        for _ in 0..100 {
            if let HpValue::Float(v) = HyperparameterTuner::sample_value(&spec, &mut rng) {
                assert!((0.0..=1.0).contains(&v), "out of range: {}", v);
            } else {
                panic!("expected Float");
            }
        }
    }

    #[test]
    fn test_sample_continuous_log_scale() {
        let spec = HpSpec {
            name: "lr".into(),
            hp_type: HpType::Continuous { lo: 1e-4, hi: 1e-1 },
            log_scale: true,
        };
        let mut rng = 77u64;
        for _ in 0..200 {
            if let HpValue::Float(v) = HyperparameterTuner::sample_value(&spec, &mut rng) {
                assert!(
                    (1e-4..=1e-1 + 1e-10).contains(&v),
                    "log sample out of range: {}",
                    v
                );
            } else {
                panic!("expected Float");
            }
        }
    }

    #[test]
    fn test_sample_discrete_in_range() {
        let spec = HpSpec {
            name: "layers".into(),
            hp_type: HpType::Discrete { lo: 2, hi: 8 },
            log_scale: false,
        };
        let mut rng = 55u64;
        for _ in 0..200 {
            if let HpValue::Int(v) = HyperparameterTuner::sample_value(&spec, &mut rng) {
                assert!((2..=8).contains(&v), "discrete out of range: {}", v);
            } else {
                panic!("expected Int");
            }
        }
    }

    #[test]
    fn test_sample_categorical() {
        let choices = vec!["adam".to_string(), "sgd".to_string(), "rmsprop".to_string()];
        let spec = HpSpec {
            name: "opt".into(),
            hp_type: HpType::Categorical {
                choices: choices.clone(),
            },
            log_scale: false,
        };
        let mut rng = 11u64;
        for _ in 0..300 {
            if let HpValue::Choice(s) = HyperparameterTuner::sample_value(&spec, &mut rng) {
                assert!(choices.contains(&s), "unexpected choice: {}", s);
            } else {
                panic!("expected Choice");
            }
        }
    }

    // ------------------------------------------------------------------
    // sample_config
    // ------------------------------------------------------------------

    #[test]
    fn test_sample_config_keys_match_specs() {
        let config = TunerConfig {
            specs: vec![
                HpSpec {
                    name: "lr".into(),
                    hp_type: HpType::Continuous { lo: 1e-4, hi: 1e-1 },
                    log_scale: false,
                },
                HpSpec {
                    name: "layers".into(),
                    hp_type: HpType::Discrete { lo: 1, hi: 5 },
                    log_scale: false,
                },
                HpSpec {
                    name: "opt".into(),
                    hp_type: HpType::Categorical {
                        choices: vec!["adam".into()],
                    },
                    log_scale: false,
                },
            ],
            maximize: true,
            seed: 42,
        };
        let tuner = HyperparameterTuner::new(config);
        let mut rng = 42u64;
        let cfg = tuner.sample_config(&mut rng);
        assert!(cfg.get("lr").is_some());
        assert!(cfg.get("layers").is_some());
        assert!(cfg.get("opt").is_some());
    }

    // ------------------------------------------------------------------
    // record_result / best_config
    // ------------------------------------------------------------------

    #[test]
    fn test_record_and_best_maximize() {
        let config = TunerConfig {
            specs: vec![],
            maximize: true,
            seed: 0,
        };
        let mut tuner = HyperparameterTuner::new(config);
        tuner.record_result(HpConfig::new(), 0.5, 0);
        tuner.record_result(HpConfig::new(), 0.9, 1);
        tuner.record_result(HpConfig::new(), 0.2, 2);
        let best = tuner.best_config().expect("best must exist");
        assert!((best.score - 0.9).abs() < 1e-10);
    }

    #[test]
    fn test_record_and_best_minimize() {
        let config = TunerConfig {
            specs: vec![],
            maximize: false,
            seed: 0,
        };
        let mut tuner = HyperparameterTuner::new(config);
        tuner.record_result(HpConfig::new(), 0.5, 0);
        tuner.record_result(HpConfig::new(), 0.1, 1);
        tuner.record_result(HpConfig::new(), 0.8, 2);
        let best = tuner.best_config().expect("best must exist");
        assert!((best.score - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_best_config_empty_returns_none() {
        let config = TunerConfig {
            specs: vec![],
            maximize: true,
            seed: 0,
        };
        let tuner = HyperparameterTuner::new(config);
        assert!(tuner.best_config().is_none());
    }

    #[test]
    fn test_trial_id_sequential() {
        let config = TunerConfig {
            specs: vec![],
            maximize: true,
            seed: 0,
        };
        let mut tuner = HyperparameterTuner::new(config);
        let id0 = tuner.record_result(HpConfig::new(), 1.0, 0);
        let id1 = tuner.record_result(HpConfig::new(), 2.0, 0);
        assert_eq!(id0, 0);
        assert_eq!(id1, 1);
    }

    // ------------------------------------------------------------------
    // grid_configs
    // ------------------------------------------------------------------

    #[test]
    fn test_grid_configs_continuous_gives_5_values() {
        let spec = HpSpec {
            name: "lr".into(),
            hp_type: HpType::Continuous { lo: 0.0, hi: 1.0 },
            log_scale: false,
        };
        let vals = spec_grid_values(&spec);
        assert_eq!(vals.len(), 5);
        // Check endpoints.
        if let HpValue::Float(lo) = &vals[0] {
            assert!((*lo - 0.0).abs() < 1e-10);
        }
        if let HpValue::Float(hi) = &vals[4] {
            assert!((*hi - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_grid_configs_discrete() {
        let spec = HpSpec {
            name: "n".into(),
            hp_type: HpType::Discrete { lo: 1, hi: 4 },
            log_scale: false,
        };
        let vals = spec_grid_values(&spec);
        assert_eq!(vals.len(), 4);
        assert_eq!(vals[0], HpValue::Int(1));
        assert_eq!(vals[3], HpValue::Int(4));
    }

    #[test]
    fn test_grid_configs_categorical() {
        let spec = HpSpec {
            name: "opt".into(),
            hp_type: HpType::Categorical {
                choices: vec!["a".into(), "b".into(), "c".into()],
            },
            log_scale: false,
        };
        let vals = spec_grid_values(&spec);
        assert_eq!(vals.len(), 3);
        assert_eq!(vals[1], HpValue::Choice("b".into()));
    }

    #[test]
    fn test_grid_configs_cartesian_product() {
        let config = TunerConfig {
            specs: vec![
                HpSpec {
                    name: "a".into(),
                    hp_type: HpType::Discrete { lo: 0, hi: 1 },
                    log_scale: false,
                },
                HpSpec {
                    name: "b".into(),
                    hp_type: HpType::Categorical {
                        choices: vec!["x".into(), "y".into()],
                    },
                    log_scale: false,
                },
            ],
            maximize: true,
            seed: 0,
        };
        let tuner = HyperparameterTuner::new(config);
        let cfgs = tuner.grid_configs();
        // 2 discrete * 2 categorical = 4 configs
        assert_eq!(cfgs.len(), 4);
    }

    #[test]
    fn test_grid_configs_empty_specs() {
        let config = TunerConfig {
            specs: vec![],
            maximize: true,
            seed: 0,
        };
        let tuner = HyperparameterTuner::new(config);
        assert!(tuner.grid_configs().is_empty());
    }

    // ------------------------------------------------------------------
    // run_random_search
    // ------------------------------------------------------------------

    #[test]
    fn test_run_random_search_count() {
        let config = TunerConfig {
            specs: vec![HpSpec {
                name: "x".into(),
                hp_type: HpType::Continuous { lo: 0.0, hi: 1.0 },
                log_scale: false,
            }],
            maximize: true,
            seed: 1,
        };
        let mut tuner = HyperparameterTuner::new(config);
        let mut rng = 1u64;
        let results = tuner.run_random_search(10, |_| 0.5, &mut rng);
        assert_eq!(results.len(), 10);
    }

    #[test]
    fn test_run_random_search_sorted_maximize() {
        let config = TunerConfig {
            specs: vec![HpSpec {
                name: "x".into(),
                hp_type: HpType::Continuous { lo: 0.0, hi: 1.0 },
                log_scale: false,
            }],
            maximize: true,
            seed: 42,
        };
        let mut tuner = HyperparameterTuner::new(config);
        let mut rng = 42u64;
        let mut counter = 0.0f64;
        let results = tuner.run_random_search(
            5,
            |_| {
                counter += 1.0;
                counter
            },
            &mut rng,
        );
        // Should be sorted descending.
        for w in results.windows(2) {
            assert!(
                w[0].score >= w[1].score,
                "not sorted: {} < {}",
                w[0].score,
                w[1].score
            );
        }
    }

    #[test]
    fn test_run_random_search_sorted_minimize() {
        let config = TunerConfig {
            specs: vec![HpSpec {
                name: "x".into(),
                hp_type: HpType::Continuous { lo: 0.0, hi: 1.0 },
                log_scale: false,
            }],
            maximize: false,
            seed: 7,
        };
        let mut tuner = HyperparameterTuner::new(config);
        let mut rng = 7u64;
        let mut counter = 5.0f64;
        let results = tuner.run_random_search(
            5,
            |_| {
                counter -= 1.0;
                counter
            },
            &mut rng,
        );
        for w in results.windows(2) {
            assert!(w[0].score <= w[1].score, "not sorted ascending");
        }
    }

    // ------------------------------------------------------------------
    // run_grid_search
    // ------------------------------------------------------------------

    #[test]
    fn test_run_grid_search_all_evaluated() {
        let config = TunerConfig {
            specs: vec![HpSpec {
                name: "a".into(),
                hp_type: HpType::Discrete { lo: 1, hi: 3 },
                log_scale: false,
            }],
            maximize: true,
            seed: 0,
        };
        let mut tuner = HyperparameterTuner::new(config);
        let results = tuner.run_grid_search(|cfg| {
            if let Some(HpValue::Int(v)) = cfg.get("a") {
                *v as f64
            } else {
                0.0
            }
        });
        // Discrete 1..=3 → 3 values.
        assert_eq!(results.len(), 3);
        // Sorted descending (maximize).
        assert_eq!(results[0].score, 3.0);
    }

    // ------------------------------------------------------------------
    // importance_scores
    // ------------------------------------------------------------------

    #[test]
    fn test_importance_scores_no_history() {
        let config = TunerConfig {
            specs: vec![HpSpec {
                name: "x".into(),
                hp_type: HpType::Continuous { lo: 0.0, hi: 1.0 },
                log_scale: false,
            }],
            maximize: true,
            seed: 0,
        };
        let tuner = HyperparameterTuner::new(config);
        let scores = tuner.importance_scores();
        assert_eq!(scores.get("x"), Some(&0.0));
    }

    #[test]
    fn test_importance_scores_returns_all_specs() {
        let config = TunerConfig {
            specs: vec![
                HpSpec {
                    name: "lr".into(),
                    hp_type: HpType::Continuous { lo: 0.0, hi: 1.0 },
                    log_scale: false,
                },
                HpSpec {
                    name: "layers".into(),
                    hp_type: HpType::Discrete { lo: 1, hi: 5 },
                    log_scale: false,
                },
            ],
            maximize: true,
            seed: 0,
        };
        let mut tuner = HyperparameterTuner::new(config);
        // Add enough history.
        for i in 0..10 {
            let mut cfg = HpConfig::new();
            cfg.insert("lr".into(), HpValue::Float(i as f64 * 0.1));
            cfg.insert("layers".into(), HpValue::Int(i % 5 + 1));
            tuner.record_result(cfg, i as f64, 0);
        }
        let scores = tuner.importance_scores();
        assert!(scores.contains_key("lr"));
        assert!(scores.contains_key("layers"));
    }

    // ------------------------------------------------------------------
    // stats
    // ------------------------------------------------------------------

    #[test]
    fn test_stats_empty() {
        let config = TunerConfig {
            specs: vec![],
            maximize: true,
            seed: 0,
        };
        let tuner = HyperparameterTuner::new(config);
        let s = tuner.stats();
        assert_eq!(s.total_trials, 0);
        assert_eq!(s.improvement_rate, 0.0);
    }

    #[test]
    fn test_stats_correct_values() {
        let config = TunerConfig {
            specs: vec![],
            maximize: true,
            seed: 0,
        };
        let mut tuner = HyperparameterTuner::new(config);
        tuner.record_result(HpConfig::new(), 1.0, 0);
        tuner.record_result(HpConfig::new(), 3.0, 0);
        tuner.record_result(HpConfig::new(), 2.0, 0);
        let s = tuner.stats();
        assert_eq!(s.total_trials, 3);
        assert!((s.best_score - 3.0).abs() < 1e-10);
        assert!((s.worst_score - 1.0).abs() < 1e-10);
        assert!((s.avg_score - 2.0).abs() < 1e-10);
        // Improvement rate: trial 1 (3.0 > 1.0 → improves) = 1/3.
        assert!((s.improvement_rate - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_stats_minimize() {
        let config = TunerConfig {
            specs: vec![],
            maximize: false,
            seed: 0,
        };
        let mut tuner = HyperparameterTuner::new(config);
        tuner.record_result(HpConfig::new(), 10.0, 0);
        tuner.record_result(HpConfig::new(), 5.0, 0);
        tuner.record_result(HpConfig::new(), 7.0, 0);
        let s = tuner.stats();
        assert!((s.best_score - 5.0).abs() < 1e-10);
        assert!((s.worst_score - 10.0).abs() < 1e-10);
    }

    // ------------------------------------------------------------------
    // suggest_next
    // ------------------------------------------------------------------

    #[test]
    fn test_suggest_next_returns_config_with_all_specs() {
        let config = TunerConfig {
            specs: vec![HpSpec {
                name: "lr".into(),
                hp_type: HpType::Continuous { lo: 1e-4, hi: 1.0 },
                log_scale: false,
            }],
            maximize: true,
            seed: 1,
        };
        let mut tuner = HyperparameterTuner::new(config);
        let mut rng = 1u64;
        // Add some history first.
        for i in 0..5 {
            let mut cfg = HpConfig::new();
            cfg.insert("lr".into(), HpValue::Float(0.1 * i as f64));
            tuner.record_result(cfg, i as f64, 0);
        }
        let next = tuner.suggest_next(&mut rng);
        assert!(next.get("lr").is_some());
    }

    #[test]
    fn test_suggest_next_no_history_still_works() {
        let config = TunerConfig {
            specs: vec![HpSpec {
                name: "lr".into(),
                hp_type: HpType::Continuous { lo: 0.01, hi: 0.1 },
                log_scale: false,
            }],
            maximize: true,
            seed: 5,
        };
        let tuner = HyperparameterTuner::new(config);
        let mut rng = 5u64;
        let next = tuner.suggest_next(&mut rng);
        assert!(next.get("lr").is_some());
    }

    // ------------------------------------------------------------------
    // add_spec builder
    // ------------------------------------------------------------------

    #[test]
    fn test_add_spec_builder() {
        let config = TunerConfig {
            specs: vec![],
            maximize: true,
            seed: 0,
        };
        let mut tuner = HyperparameterTuner::new(config);
        tuner.add_spec(HpSpec {
            name: "lr".into(),
            hp_type: HpType::Continuous { lo: 1e-4, hi: 1.0 },
            log_scale: true,
        });
        assert_eq!(tuner.config.specs.len(), 1);
    }

    // ------------------------------------------------------------------
    // Bayesian optimization
    // ------------------------------------------------------------------

    #[test]
    fn test_bayesian_optimization_count() {
        let config = TunerConfig {
            specs: vec![HpSpec {
                name: "x".into(),
                hp_type: HpType::Continuous { lo: 0.0, hi: 1.0 },
                log_scale: false,
            }],
            maximize: false,
            seed: 3,
        };
        let mut tuner = HyperparameterTuner::new(config);
        let mut rng = 3u64;
        let results = tuner.run_bayesian(
            10,
            3,
            1.0,
            |cfg| {
                if let Some(HpValue::Float(x)) = cfg.get("x") {
                    (*x - 0.3).powi(2)
                } else {
                    1.0
                }
            },
            &mut rng,
        );
        assert_eq!(results.len(), 10);
    }

    #[test]
    fn test_bayesian_optimization_sorted() {
        let config = TunerConfig {
            specs: vec![HpSpec {
                name: "x".into(),
                hp_type: HpType::Continuous { lo: 0.0, hi: 1.0 },
                log_scale: false,
            }],
            maximize: true,
            seed: 17,
        };
        let mut tuner = HyperparameterTuner::new(config);
        let mut rng = 17u64;
        let results = tuner.run_bayesian(
            8,
            3,
            1.5,
            |cfg| {
                if let Some(HpValue::Float(x)) = cfg.get("x") {
                    *x
                } else {
                    0.0
                }
            },
            &mut rng,
        );
        for w in results.windows(2) {
            assert!(w[0].score >= w[1].score, "Bayesian results not sorted");
        }
    }

    // ------------------------------------------------------------------
    // Euclidean distance helper
    // ------------------------------------------------------------------

    #[test]
    fn test_euclidean_dist() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        assert!((euclidean_dist(&a, &b) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_euclidean_dist_same_point() {
        let a = vec![1.0, 2.0, 3.0];
        assert!((euclidean_dist(&a, &a) - 0.0).abs() < 1e-10);
    }

    // ------------------------------------------------------------------
    // sort_results helper
    // ------------------------------------------------------------------

    #[test]
    fn test_sort_results_maximize() {
        let mut results = vec![
            TuningResult {
                trial_id: 0,
                config: HpConfig::new(),
                score: 0.2,
                timestamp: 0,
            },
            TuningResult {
                trial_id: 1,
                config: HpConfig::new(),
                score: 0.8,
                timestamp: 0,
            },
            TuningResult {
                trial_id: 2,
                config: HpConfig::new(),
                score: 0.5,
                timestamp: 0,
            },
        ];
        sort_results(&mut results, true);
        assert!((results[0].score - 0.8).abs() < 1e-10);
        assert!((results[2].score - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_sort_results_minimize() {
        let mut results = vec![
            TuningResult {
                trial_id: 0,
                config: HpConfig::new(),
                score: 0.8,
                timestamp: 0,
            },
            TuningResult {
                trial_id: 1,
                config: HpConfig::new(),
                score: 0.2,
                timestamp: 0,
            },
        ];
        sort_results(&mut results, false);
        assert!((results[0].score - 0.2).abs() < 1e-10);
    }

    // ------------------------------------------------------------------
    // compute_improvement_rate
    // ------------------------------------------------------------------

    #[test]
    fn test_improvement_rate_monotone_increase_maximize() {
        let scores = vec![1.0, 2.0, 3.0, 4.0];
        // Improvements at indices 1, 2, 3 → 3 improvements / 4 total = 0.75
        let rate = compute_improvement_rate(&scores, true);
        assert!((rate - 0.75).abs() < 1e-10, "expected 0.75, got {}", rate);
    }

    #[test]
    fn test_improvement_rate_no_improvement() {
        let scores = vec![5.0, 4.0, 3.0]; // descending, maximize → no improvement
        let rate = compute_improvement_rate(&scores, true);
        assert_eq!(rate, 0.0);
    }

    #[test]
    fn test_improvement_rate_minimize() {
        let scores = vec![10.0, 8.0, 6.0]; // decreasing, minimize → improvements at 1,2 → 2/3
        let rate = compute_improvement_rate(&scores, false);
        assert!((rate - 2.0 / 3.0).abs() < 1e-10);
    }

    // ------------------------------------------------------------------
    // HpTunerError display
    // ------------------------------------------------------------------

    #[test]
    fn test_error_display_no_specs() {
        let e = HpTunerError::NoSpecs;
        assert!(!format!("{}", e).is_empty());
    }

    #[test]
    fn test_error_display_invalid_spec() {
        let e = HpTunerError::InvalidSpec("bad range".into());
        assert!(format!("{}", e).contains("bad range"));
    }

    // ------------------------------------------------------------------
    // HpValue display
    // ------------------------------------------------------------------

    #[test]
    fn test_hp_value_display() {
        assert!(!format!("{}", HpValue::Float(0.001)).is_empty());
        assert!(!format!("{}", HpValue::Int(42)).is_empty());
        assert!(!format!("{}", HpValue::Choice("relu".into())).is_empty());
    }

    // ------------------------------------------------------------------
    // Grid search log scale
    // ------------------------------------------------------------------

    #[test]
    fn test_grid_continuous_log_scale_endpoints() {
        let spec = HpSpec {
            name: "lr".into(),
            hp_type: HpType::Continuous { lo: 1e-4, hi: 1e-1 },
            log_scale: true,
        };
        let vals = spec_grid_values(&spec);
        assert_eq!(vals.len(), 5);
        if let HpValue::Float(lo_val) = &vals[0] {
            assert!(
                (lo_val - 1e-4).abs() < 1e-10,
                "log-scale lo wrong: {}",
                lo_val
            );
        }
        if let HpValue::Float(hi_val) = &vals[4] {
            assert!(
                (hi_val - 1e-1).abs() < 1e-10,
                "log-scale hi wrong: {}",
                hi_val
            );
        }
    }

    // ------------------------------------------------------------------
    // Mixed spec types in sample_config
    // ------------------------------------------------------------------

    #[test]
    fn test_sample_config_all_spec_types() {
        let config = TunerConfig {
            specs: vec![
                HpSpec {
                    name: "lr".into(),
                    hp_type: HpType::Continuous { lo: 1e-4, hi: 0.1 },
                    log_scale: true,
                },
                HpSpec {
                    name: "n".into(),
                    hp_type: HpType::Discrete { lo: 2, hi: 10 },
                    log_scale: false,
                },
                HpSpec {
                    name: "act".into(),
                    hp_type: HpType::Categorical {
                        choices: vec!["relu".into(), "tanh".into()],
                    },
                    log_scale: false,
                },
            ],
            maximize: true,
            seed: 99,
        };
        let tuner = HyperparameterTuner::new(config);
        let mut rng = 99u64;
        let cfg = tuner.sample_config(&mut rng);
        assert!(matches!(cfg.get("lr"), Some(HpValue::Float(_))));
        assert!(matches!(cfg.get("n"), Some(HpValue::Int(_))));
        assert!(matches!(cfg.get("act"), Some(HpValue::Choice(_))));
    }
}
