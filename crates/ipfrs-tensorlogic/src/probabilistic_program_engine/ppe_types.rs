//! Types for the Probabilistic Program Engine.

use std::collections::HashMap;

// ─── Type aliases ────────────────────────────────────────────────────────────

/// Unique identifier for a probabilistic variable: 16 opaque bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PpeVarId(pub [u8; 16]);

/// Re-export: `PpeVarId` as the canonical `VarId` used internally.
pub type VarId = PpeVarId;

/// Alias: public name for [`ProbVar`].
pub type PpeProbVar = ProbVar;

// ─── Prior distributions ─────────────────────────────────────────────────────

/// Prior probability distribution for a probabilistic variable.
#[derive(Debug, Clone)]
pub enum PpePrior {
    /// Gaussian: N(mean, std²).
    Normal { mean: f64, std: f64 },
    /// Continuous uniform on [low, high].
    Uniform { low: f64, high: f64 },
    /// Beta distribution parameterised by α and β (support \[0,1\]).
    Beta { alpha: f64, beta: f64 },
    /// Exponential distribution with given rate λ (support [0,∞)).
    Exponential { rate: f64 },
    /// Bernoulli: takes value 1.0 with probability `p`, 0.0 otherwise.
    Bernoulli { p: f64 },
    /// Categorical: takes index `k` (as f64) with probability `probs[k]`.
    Categorical { probs: Vec<f64> },
}

// ─── Variable descriptor ─────────────────────────────────────────────────────

/// A named random variable with a prior and an optional current value.
#[derive(Debug, Clone)]
pub struct ProbVar {
    /// Stable identifier.
    pub id: VarId,
    /// Human-readable name.
    pub name: String,
    /// Prior distribution.
    pub prior: PpePrior,
    /// Current/proposed value during MCMC, or `None` before initialisation.
    pub value: Option<f64>,
}

// ─── Engine configuration ────────────────────────────────────────────────────

/// Configuration knobs for [`ProbabilisticProgramEngine`].
#[derive(Debug, Clone)]
pub struct PpeEngineConfig {
    /// Number of posterior samples to retain (after burn-in and thinning).
    pub n_samples: usize,
    /// Number of initial MCMC steps discarded as burn-in.
    pub burn_in: usize,
    /// Keep every `thinning`-th sample (1 = keep all).
    pub thinning: usize,
    /// Seed for the internal xorshift64 PRNG.
    pub seed: u64,
}

impl Default for PpeEngineConfig {
    fn default() -> Self {
        Self {
            n_samples: 1_000,
            burn_in: 200,
            thinning: 2,
            seed: 12_345_678_901_234_567,
        }
    }
}

// ─── Sampling method ─────────────────────────────────────────────────────────

/// Sampling algorithm to use when calling [`ProbabilisticProgramEngine::sample`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PpeSamplingMethod {
    /// Single-variable random-walk Metropolis-Hastings.
    MetropolisHastings,
    /// Coordinate-wise Gibbs sampling (resample each variable from its
    /// conditional distribution given the current state of all others).
    GibbsSampling,
    /// Prior-weighted importance sampling.
    ImportanceSampling,
    /// Rejection sampling from the prior.
    RejectionSampling,
}

// ─── Result types ────────────────────────────────────────────────────────────

/// Summary returned by [`ProbabilisticProgramEngine::sample`].
#[derive(Debug, Clone)]
pub struct PpeSampleResult {
    /// Method that was used.
    pub method: PpeSamplingMethod,
    /// Total MCMC steps attempted (incl. burn-in and thinning).
    pub total_steps: usize,
    /// Number of samples accepted (MH only; equals `total_steps` for other
    /// methods).
    pub accepted_samples: usize,
    /// MH acceptance rate (NaN for non-MH methods).
    pub acceptance_rate: f64,
    /// Number of variables sampled.
    pub n_variables: usize,
    /// Retained samples per variable.
    pub n_retained: usize,
}

/// Diagnostics returned by [`ProbabilisticProgramEngine::sampling_stats`].
#[derive(Debug, Clone)]
pub struct PpeSamplingStats {
    /// Number of variables registered.
    pub n_variables: usize,
    /// Number of observed (conditioned) variables.
    pub n_observed: usize,
    /// Total retained samples across all variables (sum).
    pub total_samples: usize,
    /// Whether any sampling run has been executed.
    pub has_samples: bool,
    /// Last method used, if any.
    pub last_method: Option<PpeSamplingMethod>,
    /// Effective sample size estimate (min ESS across all variables).
    pub min_ess: f64,
}

/// The main engine struct — declared here so all modules share one definition.
pub struct ProbabilisticProgramEngine {
    pub(crate) config: PpeEngineConfig,
    pub(crate) variables: HashMap<VarId, ProbVar>,
    /// Order in which variables were added (for deterministic iteration).
    pub(crate) var_order: Vec<VarId>,
    pub(crate) observations: HashMap<VarId, f64>,
    pub(crate) samples: HashMap<VarId, Vec<f64>>,
    pub(crate) rng_state: u64,
    pub(crate) last_method: Option<PpeSamplingMethod>,
    pub(crate) last_result: Option<PpeSampleResult>,
}
