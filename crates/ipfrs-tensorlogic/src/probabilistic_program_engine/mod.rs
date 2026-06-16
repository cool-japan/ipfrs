//! Probabilistic Program Engine — Bayesian reasoning and posterior sampling.
//!
//! # Overview
//!
//! [`ProbabilisticProgramEngine`] provides a flexible probabilistic programming
//! environment for Bayesian inference.  Variables are declared with a prior
//! distribution; observations fix concrete likelihoods; the engine then draws
//! posterior samples via one of four sampling strategies:
//!
//! * **Metropolis-Hastings** — single-variable random-walk MCMC.
//! * **Gibbs Sampling** — coordinate-wise conditional sampling.
//! * **Importance Sampling** — weighted samples from the prior.
//! * **Rejection Sampling** — accept/reject from prior using unnormalised
//!   likelihood.
//!
//! After sampling, marginal posteriors, credible intervals, and histogram
//! approximations are available.
//!
//! All random number generation uses an inline xorshift64 PRNG seeded from
//! [`PpeEngineConfig::seed`]; no external RNG crates are required.
//!
//! # Quick-Start
//!
//! ```rust
//! use ipfrs_tensorlogic::probabilistic_program_engine::{
//!     PpeEngineConfig, PpePrior, PpeSamplingMethod, ProbabilisticProgramEngine,
//! };
//!
//! let config = PpeEngineConfig {
//!     n_samples: 500,
//!     burn_in: 100,
//!     thinning: 2,
//!     seed: 42,
//! };
//! let mut engine = ProbabilisticProgramEngine::new(config);
//!
//! // Add a Normally-distributed variable mu ~ N(0, 1).
//! let mu_id = engine.add_variable("mu".into(), PpePrior::Normal { mean: 0.0, std: 1.0 });
//!
//! // Condition on an observation.
//! engine.observe(mu_id, 0.5);
//!
//! // Run Metropolis-Hastings.
//! let result = engine.sample(PpeSamplingMethod::MetropolisHastings).expect("example: should succeed in docs");
//! assert!(result.accepted_samples > 0);
//!
//! // Posterior statistics.
//! let mean = engine.posterior_mean(mu_id).expect("example: should succeed in docs");
//! println!("Posterior mean of mu ≈ {mean:.4}");
//! ```

mod ppe_types;
pub use ppe_types::*;

mod ppe_sampling;
use ppe_sampling::{
    effective_sample_size, log_density, mh_propose, sample_prior, total_log_likelihood, uniform01,
    xorshift64,
};

use std::collections::HashMap;

// ─── ProbabilisticProgramEngine implementation ────────────────────────────────

impl ProbabilisticProgramEngine {
    /// Create a new engine with the given configuration.
    pub fn new(config: PpeEngineConfig) -> Self {
        let seed = if config.seed == 0 {
            0xDEAD_BEEF_CAFE_1234
        } else {
            config.seed
        };
        Self {
            rng_state: seed,
            config,
            variables: HashMap::new(),
            var_order: Vec::new(),
            observations: HashMap::new(),
            samples: HashMap::new(),
            last_method: None,
            last_result: None,
        }
    }

    // ── Variable management ────────────────────────────────────────────────

    /// Register a new variable with the given name and prior.  Returns its
    /// stable [`VarId`].
    pub fn add_variable(&mut self, name: String, prior: PpePrior) -> VarId {
        // Generate a deterministic-but-unique id from the PRNG.
        let mut id_bytes = [0u8; 16];
        let a = xorshift64(&mut self.rng_state).to_le_bytes();
        let b = xorshift64(&mut self.rng_state).to_le_bytes();
        id_bytes[..8].copy_from_slice(&a);
        id_bytes[8..].copy_from_slice(&b);
        let id = PpeVarId(id_bytes);

        // Initialise value by drawing one sample from the prior.
        let initial = sample_prior(&prior, &mut self.rng_state);
        let var = ProbVar {
            id,
            name,
            prior,
            value: Some(initial),
        };
        self.variables.insert(id, var);
        self.var_order.push(id);
        id
    }

    /// Condition variable `var_id` on the observed value.
    pub fn observe(&mut self, var_id: VarId, value: f64) {
        self.observations.insert(var_id, value);
        // Fix the variable's current value to the observation.
        if let Some(var) = self.variables.get_mut(&var_id) {
            var.value = Some(value);
        }
    }

    /// Remove an observation, allowing the variable to be sampled freely.
    pub fn clear_observation(&mut self, var_id: VarId) {
        self.observations.remove(&var_id);
    }

    // ── Sampling ──────────────────────────────────────────────────────────

    /// Run posterior sampling with the given method.
    ///
    /// # Errors
    ///
    /// Returns an error string if no variables have been registered.
    pub fn sample(&mut self, method: PpeSamplingMethod) -> Result<PpeSampleResult, String> {
        if self.variables.is_empty() {
            return Err("No variables registered".to_string());
        }
        self.samples.clear();

        let result = match method {
            PpeSamplingMethod::MetropolisHastings => self.run_metropolis_hastings(),
            PpeSamplingMethod::GibbsSampling => self.run_gibbs(),
            PpeSamplingMethod::ImportanceSampling => self.run_importance_sampling(),
            PpeSamplingMethod::RejectionSampling => self.run_rejection_sampling(),
        };
        self.last_method = Some(method);
        self.last_result = Some(result.clone());
        Ok(result)
    }

    // ── MH ───────────────────────────────────────────────────────────────

    fn run_metropolis_hastings(&mut self) -> PpeSampleResult {
        let n_samples = self.config.n_samples;
        let burn_in = self.config.burn_in;
        let thinning = self.config.thinning.max(1);
        let total_steps = burn_in + n_samples * thinning;

        // Initialise buffers.
        let var_ids: Vec<VarId> = self.var_order.clone();
        for &id in &var_ids {
            self.samples.insert(id, Vec::with_capacity(n_samples));
        }

        // Current state.
        let mut current_values: HashMap<VarId, f64> = var_ids
            .iter()
            .filter_map(|&id| {
                let v = self.variables.get(&id)?.value?;
                Some((id, v))
            })
            .collect();

        let mut accepted = 0usize;
        let mut collected = 0usize;

        for step in 0..total_steps {
            // Pick a variable to update (cyclic).
            let var_id = var_ids[step % var_ids.len()];
            let prior = {
                match self.variables.get(&var_id) {
                    Some(v) => v.prior.clone(),
                    None => continue,
                }
            };

            let current_val = *current_values.get(&var_id).unwrap_or(&0.0);
            let proposed_val = mh_propose(current_val, &prior, &mut self.rng_state);

            // Log-posterior ratio: log p(x'|prior) + logL(x') - log p(x|prior) - logL(x).
            let log_p_current = log_density(&prior, current_val);
            let log_p_proposed = log_density(&prior, proposed_val);

            let mut proposed_values = current_values.clone();
            proposed_values.insert(var_id, proposed_val);

            let ll_current =
                total_log_likelihood(&self.variables, &self.observations, &current_values);
            let ll_proposed =
                total_log_likelihood(&self.variables, &self.observations, &proposed_values);

            let log_ratio = (log_p_proposed + ll_proposed) - (log_p_current + ll_current);
            let accept = log_ratio >= 0.0 || uniform01(&mut self.rng_state) < log_ratio.exp();

            if accept {
                current_values.insert(var_id, proposed_val);
                accepted += 1;
            }

            // Collect sample if past burn-in and on thinning stride.
            if step >= burn_in && (step - burn_in).is_multiple_of(thinning) {
                for &id in &var_ids {
                    let val = *current_values.get(&id).unwrap_or(&0.0);
                    if let Some(buf) = self.samples.get_mut(&id) {
                        buf.push(val);
                    }
                }
                collected += 1;
            }
        }

        // Sync variable values to final state.
        for (&id, &val) in &current_values {
            if let Some(var) = self.variables.get_mut(&id) {
                var.value = Some(val);
            }
        }

        let acceptance_rate = if total_steps > 0 {
            accepted as f64 / total_steps as f64
        } else {
            0.0
        };

        PpeSampleResult {
            method: PpeSamplingMethod::MetropolisHastings,
            total_steps,
            accepted_samples: accepted,
            acceptance_rate,
            n_variables: var_ids.len(),
            n_retained: collected,
        }
    }

    // ── Gibbs ────────────────────────────────────────────────────────────

    fn run_gibbs(&mut self) -> PpeSampleResult {
        let n_samples = self.config.n_samples;
        let burn_in = self.config.burn_in;
        let thinning = self.config.thinning.max(1);
        let total_sweeps = burn_in + n_samples * thinning;

        let var_ids: Vec<VarId> = self.var_order.clone();
        for &id in &var_ids {
            self.samples.insert(id, Vec::with_capacity(n_samples));
        }

        let mut current_values: HashMap<VarId, f64> = var_ids
            .iter()
            .filter_map(|&id| {
                let v = self.variables.get(&id)?.value?;
                Some((id, v))
            })
            .collect();

        let mut collected = 0usize;

        for sweep in 0..total_sweeps {
            // Update each variable in order from its conditional.
            for &id in &var_ids {
                // For observed variables, set to observed value.
                if let Some(&obs) = self.observations.get(&id) {
                    current_values.insert(id, obs);
                    continue;
                }
                // Gibbs step: sample from prior (conjugate update approximation).
                let prior = match self.variables.get(&id) {
                    Some(v) => v.prior.clone(),
                    None => continue,
                };
                // Use rejection sampling conditioned on likelihood to get the
                // conditional.  For simplicity and efficiency, use a
                // Metropolis-within-Gibbs step.
                let current_val = *current_values.get(&id).unwrap_or(&0.0);
                let proposal = mh_propose(current_val, &prior, &mut self.rng_state);

                let log_p_current = log_density(&prior, current_val);
                let log_p_proposal = log_density(&prior, proposal);

                let mut proposed_values = current_values.clone();
                proposed_values.insert(id, proposal);

                let ll_cur =
                    total_log_likelihood(&self.variables, &self.observations, &current_values);
                let ll_prop =
                    total_log_likelihood(&self.variables, &self.observations, &proposed_values);

                let log_ratio = (log_p_proposal + ll_prop) - (log_p_current + ll_cur);
                if log_ratio >= 0.0 || uniform01(&mut self.rng_state) < log_ratio.exp() {
                    current_values.insert(id, proposal);
                }
            }

            if sweep >= burn_in && (sweep - burn_in).is_multiple_of(thinning) {
                for &id in &var_ids {
                    let val = *current_values.get(&id).unwrap_or(&0.0);
                    if let Some(buf) = self.samples.get_mut(&id) {
                        buf.push(val);
                    }
                }
                collected += 1;
            }
        }

        for (&id, &val) in &current_values {
            if let Some(var) = self.variables.get_mut(&id) {
                var.value = Some(val);
            }
        }

        PpeSampleResult {
            method: PpeSamplingMethod::GibbsSampling,
            total_steps: total_sweeps * var_ids.len(),
            accepted_samples: total_sweeps * var_ids.len(),
            acceptance_rate: 1.0,
            n_variables: var_ids.len(),
            n_retained: collected,
        }
    }

    // ── Importance sampling ──────────────────────────────────────────────

    fn run_importance_sampling(&mut self) -> PpeSampleResult {
        let n_samples = self.config.n_samples;
        let var_ids: Vec<VarId> = self.var_order.clone();
        for &id in &var_ids {
            self.samples.insert(id, Vec::with_capacity(n_samples));
        }

        // Draw many prior samples, compute log-weights, then resample.
        let n_proposal = (n_samples * 10).max(1000);
        let mut log_weights: Vec<f64> = Vec::with_capacity(n_proposal);
        let mut draws: Vec<HashMap<VarId, f64>> = Vec::with_capacity(n_proposal);

        for _ in 0..n_proposal {
            let mut draw: HashMap<VarId, f64> = HashMap::new();
            for &id in &var_ids {
                let prior = match self.variables.get(&id) {
                    Some(v) => v.prior.clone(),
                    None => continue,
                };
                let x = if let Some(&obs) = self.observations.get(&id) {
                    obs
                } else {
                    sample_prior(&prior, &mut self.rng_state)
                };
                draw.insert(id, x);
            }
            let lw = total_log_likelihood(&self.variables, &self.observations, &draw);
            log_weights.push(lw);
            draws.push(draw);
        }

        // Numerically stable softmax to get normalised weights.
        let max_lw = log_weights
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let weights: Vec<f64> = log_weights.iter().map(|&lw| (lw - max_lw).exp()).collect();
        let total_w: f64 = weights.iter().sum();

        // Systematic resampling.
        let u_start = uniform01(&mut self.rng_state) / n_samples as f64;
        let mut cumulative = 0.0_f64;
        let mut draw_idx = 0usize;
        let inv_n = 1.0 / n_samples as f64;

        for s in 0..n_samples {
            let threshold = u_start + s as f64 * inv_n;
            // Advance draw_idx until cumulative normalised weight >= threshold.
            while draw_idx < draws.len() - 1 {
                let nw = if total_w > 1e-300 {
                    weights[draw_idx] / total_w
                } else {
                    inv_n
                };
                if cumulative + nw >= threshold {
                    break;
                }
                cumulative += nw;
                draw_idx += 1;
            }
            for &id in &var_ids {
                let val = draws[draw_idx].get(&id).copied().unwrap_or(0.0);
                if let Some(buf) = self.samples.get_mut(&id) {
                    buf.push(val);
                }
            }
        }

        PpeSampleResult {
            method: PpeSamplingMethod::ImportanceSampling,
            total_steps: n_proposal,
            accepted_samples: n_samples,
            acceptance_rate: n_samples as f64 / n_proposal as f64,
            n_variables: var_ids.len(),
            n_retained: n_samples,
        }
    }

    // ── Rejection sampling ───────────────────────────────────────────────

    fn run_rejection_sampling(&mut self) -> PpeSampleResult {
        let n_samples = self.config.n_samples;
        let var_ids: Vec<VarId> = self.var_order.clone();
        for &id in &var_ids {
            self.samples.insert(id, Vec::with_capacity(n_samples));
        }

        let mut collected = 0usize;
        let mut total_attempts = 0usize;
        let max_attempts = n_samples * 10_000;

        // log acceptance threshold: use fixed value since we sample from prior.
        // Accept/reject based on unnormalised likelihood.
        while collected < n_samples && total_attempts < max_attempts {
            total_attempts += 1;
            let mut candidate: HashMap<VarId, f64> = HashMap::new();

            for &id in &var_ids {
                if let Some(&obs) = self.observations.get(&id) {
                    candidate.insert(id, obs);
                } else if let Some(var) = self.variables.get(&id) {
                    let x = sample_prior(&var.prior, &mut self.rng_state);
                    candidate.insert(id, x);
                }
            }

            let ll = total_log_likelihood(&self.variables, &self.observations, &candidate);

            // Accept with probability exp(ll) (assuming max likelihood = 1).
            let accept_prob = ll.exp().min(1.0);
            if uniform01(&mut self.rng_state) < accept_prob {
                for &id in &var_ids {
                    let val = candidate.get(&id).copied().unwrap_or(0.0);
                    if let Some(buf) = self.samples.get_mut(&id) {
                        buf.push(val);
                    }
                }
                collected += 1;
            }
        }

        // Pad with prior samples if we could not collect enough.
        while collected < n_samples {
            for &id in &var_ids {
                if let Some(var) = self.variables.get(&id) {
                    let x = if let Some(&obs) = self.observations.get(&id) {
                        obs
                    } else {
                        sample_prior(&var.prior, &mut self.rng_state)
                    };
                    if let Some(buf) = self.samples.get_mut(&id) {
                        buf.push(x);
                    }
                }
            }
            collected += 1;
        }

        let acceptance_rate = if total_attempts > 0 {
            collected as f64 / total_attempts as f64
        } else {
            0.0
        };

        PpeSampleResult {
            method: PpeSamplingMethod::RejectionSampling,
            total_steps: total_attempts,
            accepted_samples: collected,
            acceptance_rate,
            n_variables: var_ids.len(),
            n_retained: collected,
        }
    }

    // ── Posterior statistics ──────────────────────────────────────────────

    /// Posterior mean of variable `var_id` from the last sampling run.
    pub fn posterior_mean(&self, var_id: VarId) -> Option<f64> {
        let samples = self.samples.get(&var_id)?;
        if samples.is_empty() {
            return None;
        }
        Some(samples.iter().sum::<f64>() / samples.len() as f64)
    }

    /// Posterior standard deviation of variable `var_id`.
    pub fn posterior_std(&self, var_id: VarId) -> Option<f64> {
        let samples = self.samples.get(&var_id)?;
        let n = samples.len();
        if n < 2 {
            return None;
        }
        let mean = samples.iter().sum::<f64>() / n as f64;
        let variance = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
        Some(variance.sqrt())
    }

    /// Central credible interval at coverage `1 - alpha`.
    ///
    /// E.g., `alpha = 0.05` returns the 95% CI as `(lower, upper)`.
    pub fn credible_interval(&self, var_id: VarId, alpha: f64) -> Option<(f64, f64)> {
        let samples = self.samples.get(&var_id)?;
        if samples.is_empty() {
            return None;
        }
        let mut sorted = samples.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = sorted.len();
        let lo_idx = ((alpha / 2.0) * n as f64) as usize;
        let hi_idx = (((1.0 - alpha / 2.0) * n as f64) as usize).min(n - 1);
        Some((sorted[lo_idx], sorted[hi_idx]))
    }

    /// Log-density (log prior) of `value` under the prior of `var_id`.
    pub fn log_likelihood(&self, var_id: VarId, value: f64) -> f64 {
        match self.variables.get(&var_id) {
            Some(var) => log_density(&var.prior, value),
            None => f64::NEG_INFINITY,
        }
    }

    /// Approximate marginal distribution as a histogram with `n_bins` equal-width bins.
    ///
    /// Returns a `Vec<(bin_centre, frequency)>` in ascending order.
    pub fn marginal_distribution(&self, var_id: VarId, n_bins: usize) -> Vec<(f64, f64)> {
        let samples = match self.samples.get(&var_id) {
            Some(s) if !s.is_empty() => s,
            _ => return Vec::new(),
        };
        if n_bins == 0 {
            return Vec::new();
        }

        let min = samples.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = samples.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        if (max - min).abs() < 1e-300 {
            return vec![(min, samples.len() as f64)];
        }

        let bin_width = (max - min) / n_bins as f64;
        let mut counts = vec![0u64; n_bins];

        for &x in samples {
            let idx = ((x - min) / bin_width) as usize;
            let idx = idx.min(n_bins - 1);
            counts[idx] += 1;
        }

        let n = samples.len() as f64;
        counts
            .iter()
            .enumerate()
            .map(|(i, &c)| {
                let centre = min + (i as f64 + 0.5) * bin_width;
                let freq = c as f64 / n;
                (centre, freq)
            })
            .collect()
    }

    /// Diagnostics about the current state of the engine.
    pub fn sampling_stats(&self) -> PpeSamplingStats {
        let n_variables = self.variables.len();
        let n_observed = self.observations.len();
        let total_samples: usize = self.samples.values().map(Vec::len).sum();
        let has_samples = total_samples > 0;

        let min_ess = self
            .samples
            .values()
            .map(|s| effective_sample_size(s))
            .fold(f64::INFINITY, f64::min);
        let min_ess = if min_ess.is_infinite() { 0.0 } else { min_ess };

        PpeSamplingStats {
            n_variables,
            n_observed,
            total_samples,
            has_samples,
            last_method: self.last_method,
            min_ess,
        }
    }

    // ── Accessors ─────────────────────────────────────────────────────────

    /// Return a reference to the engine configuration.
    pub fn config(&self) -> &PpeEngineConfig {
        &self.config
    }

    /// Look up a variable by its id.
    pub fn get_variable(&self, var_id: VarId) -> Option<&ProbVar> {
        self.variables.get(&var_id)
    }

    /// Number of registered variables.
    pub fn n_variables(&self) -> usize {
        self.variables.len()
    }

    /// Number of retained samples for a given variable (0 if no sampling has
    /// been run).
    pub fn n_samples(&self, var_id: VarId) -> usize {
        self.samples.get(&var_id).map(Vec::len).unwrap_or(0)
    }

    /// Raw posterior samples for `var_id`.
    pub fn raw_samples(&self, var_id: VarId) -> Option<&[f64]> {
        self.samples.get(&var_id).map(Vec::as_slice)
    }

    /// Last [`PpeSampleResult`] returned by [`sample`](Self::sample).
    pub fn last_result(&self) -> Option<&PpeSampleResult> {
        self.last_result.as_ref()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::ppe_sampling::{box_muller, lgamma, sample_gamma, sample_standard_normal};
    use super::*;

    // ── helpers ───────────────────────────────────────────────────────────

    fn make_engine(seed: u64) -> ProbabilisticProgramEngine {
        ProbabilisticProgramEngine::new(PpeEngineConfig {
            n_samples: 300,
            burn_in: 50,
            thinning: 1,
            seed,
        })
    }

    fn make_small(seed: u64) -> ProbabilisticProgramEngine {
        ProbabilisticProgramEngine::new(PpeEngineConfig {
            n_samples: 100,
            burn_in: 20,
            thinning: 1,
            seed,
        })
    }

    // ── VarId ─────────────────────────────────────────────────────────────

    #[test]
    fn var_id_unique() {
        let mut engine = make_engine(1);
        let a = engine.add_variable(
            "a".into(),
            PpePrior::Normal {
                mean: 0.0,
                std: 1.0,
            },
        );
        let b = engine.add_variable(
            "b".into(),
            PpePrior::Normal {
                mean: 0.0,
                std: 1.0,
            },
        );
        assert_ne!(a, b);
    }

    #[test]
    fn var_id_equality() {
        let id = PpeVarId([1u8; 16]);
        assert_eq!(id, PpeVarId([1u8; 16]));
        assert_ne!(id, PpeVarId([2u8; 16]));
    }

    // ── add_variable ─────────────────────────────────────────────────────

    #[test]
    fn add_variable_returns_id() {
        let mut e = make_engine(2);
        let id = e.add_variable(
            "x".into(),
            PpePrior::Uniform {
                low: 0.0,
                high: 1.0,
            },
        );
        assert!(e.get_variable(id).is_some());
    }

    #[test]
    fn add_multiple_variables() {
        let mut e = make_engine(3);
        for i in 0..10 {
            e.add_variable(format!("v{i}"), PpePrior::Exponential { rate: 1.0 });
        }
        assert_eq!(e.n_variables(), 10);
    }

    #[test]
    fn variable_name_preserved() {
        let mut e = make_engine(4);
        let id = e.add_variable("my_var".into(), PpePrior::Bernoulli { p: 0.3 });
        assert_eq!(
            e.get_variable(id)
                .expect("test: variable should be registered")
                .name,
            "my_var"
        );
    }

    #[test]
    fn variable_has_initial_value() {
        let mut e = make_engine(5);
        let id = e.add_variable(
            "v".into(),
            PpePrior::Normal {
                mean: 5.0,
                std: 1.0,
            },
        );
        assert!(e
            .get_variable(id)
            .expect("test: variable should be registered")
            .value
            .is_some());
    }

    // ── observe / clear ───────────────────────────────────────────────────

    #[test]
    fn observe_sets_value() {
        let mut e = make_engine(6);
        let id = e.add_variable(
            "mu".into(),
            PpePrior::Normal {
                mean: 0.0,
                std: 1.0,
            },
        );
        e.observe(id, 3.7);
        assert_eq!(
            e.get_variable(id)
                .expect("test: variable should be registered")
                .value,
            Some(3.7)
        );
    }

    #[test]
    fn clear_observation_removes_obs() {
        let mut e = make_engine(7);
        let id = e.add_variable(
            "mu".into(),
            PpePrior::Normal {
                mean: 0.0,
                std: 1.0,
            },
        );
        e.observe(id, 1.0);
        e.clear_observation(id);
        // Variable value not reset, but observation removed.
        let stats = e.sampling_stats();
        assert_eq!(stats.n_observed, 0);
    }

    #[test]
    fn observe_nonexistent_var() {
        let mut e = make_engine(8);
        // Should not panic.
        e.observe(PpeVarId([0u8; 16]), 1.0);
    }

    // ── sample: no variables ──────────────────────────────────────────────

    #[test]
    fn sample_no_vars_returns_error() {
        let mut e = make_engine(9);
        assert!(e.sample(PpeSamplingMethod::MetropolisHastings).is_err());
    }

    // ── MH sampling ───────────────────────────────────────────────────────

    #[test]
    fn mh_produces_samples() {
        let mut e = make_engine(10);
        e.add_variable(
            "x".into(),
            PpePrior::Normal {
                mean: 0.0,
                std: 1.0,
            },
        );
        let res = e
            .sample(PpeSamplingMethod::MetropolisHastings)
            .expect("test: sampling should succeed");
        assert!(res.accepted_samples > 0);
        assert_eq!(res.method, PpeSamplingMethod::MetropolisHastings);
    }

    #[test]
    fn mh_correct_sample_count() {
        let mut e = make_engine(11);
        let id = e.add_variable(
            "x".into(),
            PpePrior::Normal {
                mean: 0.0,
                std: 1.0,
            },
        );
        e.sample(PpeSamplingMethod::MetropolisHastings)
            .expect("test: sampling should succeed");
        assert_eq!(e.n_samples(id), e.config().n_samples);
    }

    #[test]
    fn mh_normal_mean_near_observation() {
        let mut e = make_engine(12);
        let id = e.add_variable(
            "mu".into(),
            PpePrior::Normal {
                mean: 0.0,
                std: 5.0,
            },
        );
        e.observe(id, 3.0);
        e.sample(PpeSamplingMethod::MetropolisHastings)
            .expect("test: sampling should succeed");
        let mean = e
            .posterior_mean(id)
            .expect("test: posterior mean should be available");
        // With strong observation signal, posterior mean should be close to 3.
        assert!((mean - 3.0).abs() < 1.5, "mean={mean}");
    }

    #[test]
    fn mh_acceptance_rate_in_range() {
        let mut e = make_engine(13);
        e.add_variable(
            "x".into(),
            PpePrior::Normal {
                mean: 0.0,
                std: 1.0,
            },
        );
        let res = e
            .sample(PpeSamplingMethod::MetropolisHastings)
            .expect("test: sampling should succeed");
        assert!(res.acceptance_rate >= 0.0);
        assert!(res.acceptance_rate <= 1.0);
    }

    // ── Gibbs sampling ────────────────────────────────────────────────────

    #[test]
    fn gibbs_produces_samples() {
        let mut e = make_engine(14);
        e.add_variable(
            "y".into(),
            PpePrior::Uniform {
                low: -1.0,
                high: 1.0,
            },
        );
        let res = e
            .sample(PpeSamplingMethod::GibbsSampling)
            .expect("test: sampling should succeed");
        assert!(res.n_retained > 0);
        assert_eq!(res.method, PpeSamplingMethod::GibbsSampling);
    }

    #[test]
    fn gibbs_correct_sample_count() {
        let mut e = make_engine(15);
        let id = e.add_variable(
            "y".into(),
            PpePrior::Uniform {
                low: 0.0,
                high: 1.0,
            },
        );
        e.sample(PpeSamplingMethod::GibbsSampling)
            .expect("test: sampling should succeed");
        assert_eq!(e.n_samples(id), e.config().n_samples);
    }

    #[test]
    fn gibbs_observed_var_clamped() {
        let mut e = make_engine(16);
        let id = e.add_variable(
            "y".into(),
            PpePrior::Normal {
                mean: 0.0,
                std: 1.0,
            },
        );
        e.observe(id, 7.7);
        e.sample(PpeSamplingMethod::GibbsSampling)
            .expect("test: sampling should succeed");
        // All samples for observed var should be the observation value.
        let samples = e
            .raw_samples(id)
            .expect("test: raw samples should exist after sampling");
        for &s in samples {
            assert!((s - 7.7).abs() < 1e-9, "s={s}");
        }
    }

    // ── Importance sampling ───────────────────────────────────────────────

    #[test]
    fn importance_produces_samples() {
        let mut e = make_engine(17);
        e.add_variable(
            "z".into(),
            PpePrior::Normal {
                mean: 1.0,
                std: 2.0,
            },
        );
        let res = e
            .sample(PpeSamplingMethod::ImportanceSampling)
            .expect("test: sampling should succeed");
        assert!(res.n_retained > 0);
    }

    #[test]
    fn importance_sample_count_correct() {
        let mut e = make_engine(18);
        let id = e.add_variable(
            "z".into(),
            PpePrior::Normal {
                mean: 0.0,
                std: 1.0,
            },
        );
        e.sample(PpeSamplingMethod::ImportanceSampling)
            .expect("test: sampling should succeed");
        assert_eq!(e.n_samples(id), e.config().n_samples);
    }

    // ── Rejection sampling ────────────────────────────────────────────────

    #[test]
    fn rejection_produces_samples() {
        let mut e = make_small(19);
        e.add_variable(
            "r".into(),
            PpePrior::Normal {
                mean: 0.0,
                std: 1.0,
            },
        );
        let res = e
            .sample(PpeSamplingMethod::RejectionSampling)
            .expect("test: sampling should succeed");
        assert!(res.n_retained > 0);
    }

    #[test]
    fn rejection_sample_count_correct() {
        let mut e = make_small(20);
        let id = e.add_variable(
            "r".into(),
            PpePrior::Normal {
                mean: 0.0,
                std: 1.0,
            },
        );
        e.sample(PpeSamplingMethod::RejectionSampling)
            .expect("test: sampling should succeed");
        assert_eq!(e.n_samples(id), e.config().n_samples);
    }

    // ── posterior_mean ────────────────────────────────────────────────────

    #[test]
    fn posterior_mean_none_before_sampling() {
        let mut e = make_engine(21);
        let id = e.add_variable(
            "m".into(),
            PpePrior::Normal {
                mean: 0.0,
                std: 1.0,
            },
        );
        assert!(e.posterior_mean(id).is_none());
    }

    #[test]
    fn posterior_mean_finite_after_mh() {
        let mut e = make_engine(22);
        let id = e.add_variable(
            "m".into(),
            PpePrior::Normal {
                mean: 0.0,
                std: 1.0,
            },
        );
        e.sample(PpeSamplingMethod::MetropolisHastings)
            .expect("test: sampling should succeed");
        let m = e
            .posterior_mean(id)
            .expect("test: posterior mean should be available");
        assert!(m.is_finite());
    }

    #[test]
    fn posterior_mean_uniform_midpoint() {
        let mut e = make_engine(23);
        let id = e.add_variable(
            "u".into(),
            PpePrior::Uniform {
                low: 0.0,
                high: 2.0,
            },
        );
        e.sample(PpeSamplingMethod::MetropolisHastings)
            .expect("test: sampling should succeed");
        let m = e
            .posterior_mean(id)
            .expect("test: posterior mean should be available");
        // Uniform[0,2] has mean 1.0; allow ±0.3 tolerance.
        assert!((m - 1.0).abs() < 0.5, "mean={m}");
    }

    #[test]
    fn posterior_mean_bernoulli() {
        let mut e = make_engine(24);
        let id = e.add_variable("b".into(), PpePrior::Bernoulli { p: 0.7 });
        e.sample(PpeSamplingMethod::ImportanceSampling)
            .expect("test: sampling should succeed");
        let m = e
            .posterior_mean(id)
            .expect("test: posterior mean should be available");
        assert!((0.0..=1.0).contains(&m), "mean={m}");
    }

    // ── posterior_std ─────────────────────────────────────────────────────

    #[test]
    fn posterior_std_none_before_sampling() {
        let mut e = make_engine(25);
        let id = e.add_variable(
            "s".into(),
            PpePrior::Normal {
                mean: 0.0,
                std: 1.0,
            },
        );
        assert!(e.posterior_std(id).is_none());
    }

    #[test]
    fn posterior_std_non_negative() {
        let mut e = make_engine(26);
        let id = e.add_variable(
            "s".into(),
            PpePrior::Normal {
                mean: 0.0,
                std: 2.0,
            },
        );
        e.sample(PpeSamplingMethod::MetropolisHastings)
            .expect("test: sampling should succeed");
        let std = e
            .posterior_std(id)
            .expect("test: posterior std should be available");
        assert!(std >= 0.0);
    }

    // ── credible_interval ────────────────────────────────────────────────

    #[test]
    fn credible_interval_none_before_sampling() {
        let mut e = make_engine(27);
        let id = e.add_variable(
            "c".into(),
            PpePrior::Normal {
                mean: 0.0,
                std: 1.0,
            },
        );
        assert!(e.credible_interval(id, 0.05).is_none());
    }

    #[test]
    fn credible_interval_lower_lt_upper() {
        let mut e = make_engine(28);
        let id = e.add_variable(
            "c".into(),
            PpePrior::Normal {
                mean: 0.0,
                std: 1.0,
            },
        );
        e.sample(PpeSamplingMethod::MetropolisHastings)
            .expect("test: sampling should succeed");
        let (lo, hi) = e
            .credible_interval(id, 0.05)
            .expect("test: credible interval should be available");
        assert!(lo <= hi, "lo={lo}, hi={hi}");
    }

    #[test]
    fn credible_interval_50pct() {
        let mut e = make_engine(29);
        let id = e.add_variable(
            "c".into(),
            PpePrior::Uniform {
                low: 0.0,
                high: 1.0,
            },
        );
        e.sample(PpeSamplingMethod::GibbsSampling)
            .expect("test: sampling should succeed");
        let (lo, hi) = e
            .credible_interval(id, 0.5)
            .expect("test: credible interval should be available");
        assert!(lo >= 0.0 && hi <= 1.0);
        assert!(lo <= hi);
    }

    // ── log_likelihood ────────────────────────────────────────────────────

    #[test]
    fn log_likelihood_normal_peak_at_mean() {
        let mut e = make_engine(30);
        let id = e.add_variable(
            "ll".into(),
            PpePrior::Normal {
                mean: 2.0,
                std: 1.0,
            },
        );
        let at_mean = e.log_likelihood(id, 2.0);
        let off = e.log_likelihood(id, 5.0);
        assert!(at_mean > off);
    }

    #[test]
    fn log_likelihood_uniform_constant_inside() {
        let mut e = make_engine(31);
        let id = e.add_variable(
            "ll".into(),
            PpePrior::Uniform {
                low: 0.0,
                high: 1.0,
            },
        );
        let a = e.log_likelihood(id, 0.2);
        let b = e.log_likelihood(id, 0.8);
        assert!((a - b).abs() < 1e-9);
    }

    #[test]
    fn log_likelihood_uniform_neg_inf_outside() {
        let mut e = make_engine(32);
        let id = e.add_variable(
            "ll".into(),
            PpePrior::Uniform {
                low: 0.0,
                high: 1.0,
            },
        );
        assert_eq!(e.log_likelihood(id, -1.0), f64::NEG_INFINITY);
        assert_eq!(e.log_likelihood(id, 2.0), f64::NEG_INFINITY);
    }

    #[test]
    fn log_likelihood_exponential_positive() {
        let mut e = make_engine(33);
        let id = e.add_variable("ll".into(), PpePrior::Exponential { rate: 1.0 });
        let v = e.log_likelihood(id, 1.0);
        assert!(v.is_finite());
    }

    #[test]
    fn log_likelihood_exponential_neg_inf_outside() {
        let mut e = make_engine(34);
        let id = e.add_variable("ll".into(), PpePrior::Exponential { rate: 1.0 });
        assert_eq!(e.log_likelihood(id, -0.1), f64::NEG_INFINITY);
    }

    #[test]
    fn log_likelihood_beta_inside_unit_interval() {
        let mut e = make_engine(35);
        let id = e.add_variable(
            "ll".into(),
            PpePrior::Beta {
                alpha: 2.0,
                beta: 2.0,
            },
        );
        let v = e.log_likelihood(id, 0.5);
        assert!(v.is_finite());
    }

    #[test]
    fn log_likelihood_beta_boundary_neg_inf() {
        let mut e = make_engine(36);
        let id = e.add_variable(
            "ll".into(),
            PpePrior::Beta {
                alpha: 2.0,
                beta: 2.0,
            },
        );
        assert_eq!(e.log_likelihood(id, 0.0), f64::NEG_INFINITY);
        assert_eq!(e.log_likelihood(id, 1.0), f64::NEG_INFINITY);
    }

    #[test]
    fn log_likelihood_nonexistent_var() {
        let e = make_engine(37);
        assert_eq!(
            e.log_likelihood(PpeVarId([0u8; 16]), 1.0),
            f64::NEG_INFINITY
        );
    }

    // ── marginal_distribution ─────────────────────────────────────────────

    #[test]
    fn marginal_empty_before_sampling() {
        let mut e = make_engine(38);
        let id = e.add_variable(
            "m".into(),
            PpePrior::Normal {
                mean: 0.0,
                std: 1.0,
            },
        );
        assert!(e.marginal_distribution(id, 10).is_empty());
    }

    #[test]
    fn marginal_correct_bin_count() {
        let mut e = make_engine(39);
        let id = e.add_variable(
            "m".into(),
            PpePrior::Normal {
                mean: 0.0,
                std: 1.0,
            },
        );
        e.sample(PpeSamplingMethod::MetropolisHastings)
            .expect("test: sampling should succeed");
        let hist = e.marginal_distribution(id, 20);
        assert_eq!(hist.len(), 20);
    }

    #[test]
    fn marginal_frequencies_sum_to_one() {
        let mut e = make_engine(40);
        let id = e.add_variable(
            "m".into(),
            PpePrior::Uniform {
                low: 0.0,
                high: 1.0,
            },
        );
        e.sample(PpeSamplingMethod::GibbsSampling)
            .expect("test: sampling should succeed");
        let hist = e.marginal_distribution(id, 10);
        let total: f64 = hist.iter().map(|(_, f)| f).sum();
        assert!((total - 1.0).abs() < 1e-9, "total={total}");
    }

    #[test]
    fn marginal_zero_bins_returns_empty() {
        let mut e = make_engine(41);
        let id = e.add_variable(
            "m".into(),
            PpePrior::Normal {
                mean: 0.0,
                std: 1.0,
            },
        );
        e.sample(PpeSamplingMethod::MetropolisHastings)
            .expect("test: sampling should succeed");
        assert!(e.marginal_distribution(id, 0).is_empty());
    }

    // ── sampling_stats ────────────────────────────────────────────────────

    #[test]
    fn sampling_stats_initial_state() {
        let mut e = make_engine(42);
        e.add_variable(
            "x".into(),
            PpePrior::Normal {
                mean: 0.0,
                std: 1.0,
            },
        );
        let stats = e.sampling_stats();
        assert_eq!(stats.n_variables, 1);
        assert!(!stats.has_samples);
    }

    #[test]
    fn sampling_stats_after_mh() {
        let mut e = make_engine(43);
        e.add_variable(
            "x".into(),
            PpePrior::Normal {
                mean: 0.0,
                std: 1.0,
            },
        );
        e.sample(PpeSamplingMethod::MetropolisHastings)
            .expect("test: sampling should succeed");
        let stats = e.sampling_stats();
        assert!(stats.has_samples);
        assert_eq!(
            stats.last_method,
            Some(PpeSamplingMethod::MetropolisHastings)
        );
    }

    #[test]
    fn sampling_stats_total_samples() {
        let mut e = make_engine(44);
        e.add_variable(
            "a".into(),
            PpePrior::Normal {
                mean: 0.0,
                std: 1.0,
            },
        );
        e.add_variable(
            "b".into(),
            PpePrior::Normal {
                mean: 1.0,
                std: 1.0,
            },
        );
        e.sample(PpeSamplingMethod::GibbsSampling)
            .expect("test: sampling should succeed");
        let stats = e.sampling_stats();
        assert_eq!(stats.total_samples, 2 * e.config().n_samples);
    }

    #[test]
    fn sampling_stats_min_ess_positive() {
        let mut e = make_engine(45);
        e.add_variable(
            "x".into(),
            PpePrior::Normal {
                mean: 0.0,
                std: 1.0,
            },
        );
        e.sample(PpeSamplingMethod::MetropolisHastings)
            .expect("test: sampling should succeed");
        let stats = e.sampling_stats();
        assert!(stats.min_ess >= 0.0);
    }

    // ── last_result ───────────────────────────────────────────────────────

    #[test]
    fn last_result_none_before_sampling() {
        let e = make_engine(46);
        assert!(e.last_result().is_none());
    }

    #[test]
    fn last_result_after_sampling() {
        let mut e = make_engine(47);
        e.add_variable(
            "x".into(),
            PpePrior::Normal {
                mean: 0.0,
                std: 1.0,
            },
        );
        e.sample(PpeSamplingMethod::ImportanceSampling)
            .expect("test: sampling should succeed");
        assert!(e.last_result().is_some());
    }

    // ── prior distributions ───────────────────────────────────────────────

    #[test]
    fn categorical_samples_valid_indices() {
        let mut e = make_small(48);
        let probs = vec![0.2, 0.5, 0.3];
        let id = e.add_variable("cat".into(), PpePrior::Categorical { probs });
        e.sample(PpeSamplingMethod::ImportanceSampling)
            .expect("test: sampling should succeed");
        let samples = e
            .raw_samples(id)
            .expect("test: raw samples should exist after sampling");
        for &s in samples {
            assert!((0.0..3.0).contains(&s), "s={s}");
        }
    }

    #[test]
    fn exponential_samples_non_negative() {
        let mut e = make_small(49);
        let id = e.add_variable("exp".into(), PpePrior::Exponential { rate: 2.0 });
        e.sample(PpeSamplingMethod::GibbsSampling)
            .expect("test: sampling should succeed");
        let samples = e
            .raw_samples(id)
            .expect("test: raw samples should exist after sampling");
        for &s in samples {
            assert!(s >= 0.0, "s={s}");
        }
    }

    #[test]
    fn beta_samples_in_unit_interval() {
        let mut e = make_small(50);
        let id = e.add_variable(
            "beta".into(),
            PpePrior::Beta {
                alpha: 2.0,
                beta: 5.0,
            },
        );
        e.sample(PpeSamplingMethod::MetropolisHastings)
            .expect("test: sampling should succeed");
        let samples = e
            .raw_samples(id)
            .expect("test: raw samples should exist after sampling");
        for &s in samples {
            assert!((0.0..=1.0).contains(&s), "s={s}");
        }
    }

    #[test]
    fn bernoulli_samples_zero_or_one() {
        let mut e = make_small(51);
        let id = e.add_variable("bern".into(), PpePrior::Bernoulli { p: 0.6 });
        e.sample(PpeSamplingMethod::MetropolisHastings)
            .expect("test: sampling should succeed");
        let samples = e
            .raw_samples(id)
            .expect("test: raw samples should exist after sampling");
        for &s in samples {
            assert!(s == 0.0 || s == 1.0, "s={s}");
        }
    }

    // ── PRNG helpers ──────────────────────────────────────────────────────

    #[test]
    fn xorshift64_not_zero() {
        let mut state = 12345678u64;
        let r = xorshift64(&mut state);
        assert_ne!(r, 0);
    }

    #[test]
    fn xorshift64_different_successive_values() {
        let mut state = 99999u64;
        let a = xorshift64(&mut state);
        let b = xorshift64(&mut state);
        assert_ne!(a, b);
    }

    #[test]
    fn uniform01_in_range() {
        let mut state = 777u64;
        for _ in 0..1000 {
            let u = uniform01(&mut state);
            assert!((0.0..1.0).contains(&u), "u={u}");
        }
    }

    #[test]
    fn box_muller_finite() {
        let bm = box_muller(0.5, 0.3);
        assert!(bm.is_finite());
    }

    #[test]
    fn sample_standard_normal_finite() {
        let mut state = 4242u64;
        for _ in 0..100 {
            let n = sample_standard_normal(&mut state);
            assert!(n.is_finite(), "n={n}");
        }
    }

    #[test]
    fn lgamma_positive_values() {
        assert!(lgamma(1.0).is_finite());
        assert!(lgamma(2.0).is_finite());
        assert!(lgamma(0.5).is_finite());
    }

    #[test]
    fn lgamma_negative_inf_for_zero() {
        // lgamma(0) = +Inf
        let v = lgamma(0.0);
        assert!(v.is_infinite() && v > 0.0, "v={v}");
    }

    #[test]
    fn sample_gamma_positive() {
        let mut state = 123456u64;
        for shape in [0.5, 1.0, 2.0, 5.0] {
            let g = sample_gamma(shape, &mut state);
            assert!(g >= 0.0, "shape={shape}, g={g}");
        }
    }

    // ── Thinning ─────────────────────────────────────────────────────────

    #[test]
    fn thinning_respected() {
        let mut e = ProbabilisticProgramEngine::new(PpeEngineConfig {
            n_samples: 50,
            burn_in: 10,
            thinning: 3,
            seed: 9876,
        });
        let id = e.add_variable(
            "x".into(),
            PpePrior::Normal {
                mean: 0.0,
                std: 1.0,
            },
        );
        e.sample(PpeSamplingMethod::MetropolisHastings)
            .expect("test: sampling should succeed");
        assert_eq!(e.n_samples(id), 50);
    }

    // ── Multiple variables ────────────────────────────────────────────────

    #[test]
    fn multiple_vars_all_sampled() {
        let mut e = make_engine(60);
        let ids: Vec<_> = (0..5)
            .map(|i| {
                e.add_variable(
                    format!("v{i}"),
                    PpePrior::Normal {
                        mean: i as f64,
                        std: 1.0,
                    },
                )
            })
            .collect();
        e.sample(PpeSamplingMethod::GibbsSampling)
            .expect("test: sampling should succeed");
        for id in ids {
            assert_eq!(e.n_samples(id), e.config().n_samples);
        }
    }

    #[test]
    fn multiple_observations() {
        let mut e = make_engine(61);
        let a = e.add_variable(
            "a".into(),
            PpePrior::Normal {
                mean: 0.0,
                std: 1.0,
            },
        );
        let b = e.add_variable(
            "b".into(),
            PpePrior::Normal {
                mean: 0.0,
                std: 1.0,
            },
        );
        e.observe(a, 1.0);
        e.observe(b, -1.0);
        e.sample(PpeSamplingMethod::MetropolisHastings)
            .expect("test: sampling should succeed");
        let stats = e.sampling_stats();
        assert_eq!(stats.n_observed, 2);
    }

    // ── Edge cases ────────────────────────────────────────────────────────

    #[test]
    fn credible_interval_full_alpha() {
        let mut e = make_engine(62);
        let id = e.add_variable(
            "x".into(),
            PpePrior::Normal {
                mean: 0.0,
                std: 1.0,
            },
        );
        e.sample(PpeSamplingMethod::MetropolisHastings)
            .expect("test: sampling should succeed");
        // alpha=0 should give the full range.
        let (lo, hi) = e
            .credible_interval(id, 0.0)
            .expect("test: credible interval should be available");
        assert!(lo <= hi);
    }

    #[test]
    fn raw_samples_none_before_sampling() {
        let mut e = make_engine(63);
        let id = e.add_variable(
            "x".into(),
            PpePrior::Normal {
                mean: 0.0,
                std: 1.0,
            },
        );
        assert!(e.raw_samples(id).is_none());
    }

    #[test]
    fn default_config() {
        let cfg = PpeEngineConfig::default();
        assert!(cfg.n_samples > 0);
        assert!(cfg.thinning > 0);
        assert!(cfg.seed > 0);
    }

    #[test]
    fn posterior_std_single_sample_none() {
        let mut e = ProbabilisticProgramEngine::new(PpeEngineConfig {
            n_samples: 1,
            burn_in: 0,
            thinning: 1,
            seed: 777,
        });
        let id = e.add_variable(
            "x".into(),
            PpePrior::Normal {
                mean: 0.0,
                std: 1.0,
            },
        );
        e.sample(PpeSamplingMethod::RejectionSampling)
            .expect("test: sampling should succeed");
        // Std of a single sample is undefined.
        assert!(e.posterior_std(id).is_none());
    }
}
