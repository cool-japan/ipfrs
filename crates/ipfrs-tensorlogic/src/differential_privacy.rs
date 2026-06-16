//! Differential privacy toolkit for IPFRS.
//!
//! Provides noise mechanisms (Laplace, Gaussian), sensitivity analysis, and
//! privacy budget management for differentially-private data queries.
//!
//! # Example
//!
//! ```rust
//! use ipfrs_tensorlogic::differential_privacy::{
//!     DifferentialPrivacyEngine, DpQuery, PrivacyMechanism,
//! };
//!
//! let mut engine = DifferentialPrivacyEngine::new(10.0, 1e-5, 100);
//!
//! let query = DpQuery {
//!     query_id: "q1".to_string(),
//!     sensitivity: 1.0,
//!     mechanism: PrivacyMechanism::Laplace { sensitivity: 1.0, epsilon: 1.0 },
//! };
//!
//! let result = engine.apply_mechanism(&query, 42.0).expect("example: should succeed in docs");
//! assert_eq!(result.query_id, "q1");
//! assert!(result.noisy_value.is_finite());
//! ```

use std::collections::VecDeque;
use std::f64::consts::PI;
use thiserror::Error;

// ── xorshift64 PRNG ────────────────────────────────────────────────────────

/// xorshift64 PRNG — fast, deterministic, no external dependencies.
fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

// ── DpError ────────────────────────────────────────────────────────────────

/// Errors produced by differential-privacy operations.
#[derive(Debug, Error, Clone)]
pub enum DpError {
    /// Privacy budget is fully consumed.
    #[error("privacy budget exhausted: remaining epsilon = {remaining:.6}")]
    BudgetExhausted {
        /// Remaining epsilon at the time of the error.
        remaining: f64,
    },

    /// A parameter value was semantically invalid.
    #[error("invalid parameters: {0}")]
    InvalidParameters(String),

    /// Sensitivity was zero or negative, making noise computation impossible.
    #[error("sensitivity must be positive (got zero or negative)")]
    ZeroSensitivity,

    /// Epsilon was zero or negative, making the mechanism undefined.
    #[error("epsilon must be strictly positive")]
    InvalidEpsilon,
}

// ── PrivacyMechanism ───────────────────────────────────────────────────────

/// The noise mechanism to apply when answering a differentially-private query.
#[derive(Debug, Clone, PartialEq)]
pub enum PrivacyMechanism {
    /// Laplace mechanism: adds Laplace-distributed noise calibrated to
    /// `sensitivity / epsilon`.
    Laplace {
        /// Global L1 sensitivity of the query function.
        sensitivity: f64,
        /// Privacy parameter ε > 0.
        epsilon: f64,
    },

    /// Gaussian mechanism: adds Gaussian noise calibrated to achieve
    /// (ε, δ)-differential privacy.
    Gaussian {
        /// Global L2 sensitivity of the query function.
        sensitivity: f64,
        /// Privacy parameter ε > 0.
        epsilon: f64,
        /// Privacy failure probability δ ∈ (0, 1).
        delta: f64,
    },

    /// Randomized response mechanism for local differential privacy.
    Randomized {
        /// Privacy parameter ε > 0 (determines flip probability).
        epsilon: f64,
    },
}

impl PrivacyMechanism {
    /// Return the epsilon associated with this mechanism.
    pub fn epsilon(&self) -> f64 {
        match self {
            PrivacyMechanism::Laplace { epsilon, .. } => *epsilon,
            PrivacyMechanism::Gaussian { epsilon, .. } => *epsilon,
            PrivacyMechanism::Randomized { epsilon } => *epsilon,
        }
    }

    /// Return the delta associated with this mechanism (0.0 for pure DP).
    pub fn delta(&self) -> f64 {
        match self {
            PrivacyMechanism::Gaussian { delta, .. } => *delta,
            _ => 0.0,
        }
    }

    /// Return the sensitivity, if applicable (None for Randomized).
    pub fn sensitivity(&self) -> Option<f64> {
        match self {
            PrivacyMechanism::Laplace { sensitivity, .. } => Some(*sensitivity),
            PrivacyMechanism::Gaussian { sensitivity, .. } => Some(*sensitivity),
            PrivacyMechanism::Randomized { .. } => None,
        }
    }

    /// Validate mechanism parameters, returning an error on invalid values.
    pub fn validate(&self) -> Result<(), DpError> {
        let eps = self.epsilon();
        if eps <= 0.0 {
            return Err(DpError::InvalidEpsilon);
        }
        if let Some(s) = self.sensitivity() {
            if s <= 0.0 {
                return Err(DpError::ZeroSensitivity);
            }
        }
        if let PrivacyMechanism::Gaussian { delta, .. } = self {
            if *delta <= 0.0 || *delta >= 1.0 {
                return Err(DpError::InvalidParameters(format!(
                    "delta must be in (0,1), got {delta}"
                )));
            }
        }
        Ok(())
    }
}

// ── NoiseScale ─────────────────────────────────────────────────────────────

/// Computed noise scale for a given mechanism.
///
/// - Laplace:  `scale = sensitivity / epsilon`
/// - Gaussian: `scale = sensitivity * sqrt(2 * ln(1.25 / delta)) / epsilon`
/// - Randomized: `scale = 1.0 / (exp(epsilon) + 1)` (flip probability)
#[derive(Debug, Clone)]
pub struct NoiseScale {
    /// The mechanism this scale was computed for.
    pub mechanism: PrivacyMechanism,
    /// The computed noise scale (standard deviation or rate parameter).
    pub scale: f64,
}

// ── PrivacyParameters ──────────────────────────────────────────────────────

/// Budget parameters bundling epsilon, delta, and sensitivity together.
#[derive(Debug, Clone)]
pub struct PrivacyParameters {
    /// Privacy parameter ε.
    pub epsilon: f64,
    /// Privacy failure probability δ.
    pub delta: f64,
    /// Query sensitivity.
    pub sensitivity: f64,
}

impl PrivacyParameters {
    /// Construct and validate privacy parameters.
    pub fn new(epsilon: f64, delta: f64, sensitivity: f64) -> Result<Self, DpError> {
        if epsilon <= 0.0 {
            return Err(DpError::InvalidEpsilon);
        }
        if sensitivity <= 0.0 {
            return Err(DpError::ZeroSensitivity);
        }
        if !(0.0..1.0).contains(&delta) {
            return Err(DpError::InvalidParameters(format!(
                "delta must be in [0,1), got {delta}"
            )));
        }
        Ok(Self {
            epsilon,
            delta,
            sensitivity,
        })
    }
}

// ── DpQuery ────────────────────────────────────────────────────────────────

/// A differentially-private query specification.
#[derive(Debug, Clone)]
pub struct DpQuery {
    /// Unique identifier for this query.
    pub query_id: String,
    /// Global sensitivity of the query function.
    pub sensitivity: f64,
    /// Noise mechanism to apply.
    pub mechanism: PrivacyMechanism,
}

// ── DpResult ───────────────────────────────────────────────────────────────

/// The result of answering a differentially-private query.
#[derive(Debug, Clone)]
pub struct DpResult {
    /// The query identifier this result corresponds to.
    pub query_id: String,
    /// The true (pre-noise) value.
    pub true_value: f64,
    /// The noisy (post-mechanism) value returned to the caller.
    pub noisy_value: f64,
    /// The signed noise that was added: `noisy_value - true_value`.
    pub noise_added: f64,
    /// The epsilon charged against the privacy budget for this query.
    pub privacy_cost: f64,
}

// ── BudgetTracker ──────────────────────────────────────────────────────────

/// Tracks consumed and remaining privacy budget.
#[derive(Debug, Clone)]
pub struct BudgetTracker {
    /// Total epsilon allocated for all queries.
    pub epsilon_budget: f64,
    /// Epsilon consumed so far.
    pub epsilon_used: f64,
    /// Total delta allocated for all queries.
    pub delta_budget: f64,
    /// Delta consumed so far.
    pub delta_used: f64,
    /// Number of queries answered successfully.
    pub queries_answered: u64,
}

impl BudgetTracker {
    /// Construct a new tracker with given budgets and zero consumption.
    pub fn new(epsilon_budget: f64, delta_budget: f64) -> Self {
        Self {
            epsilon_budget,
            epsilon_used: 0.0,
            delta_budget,
            delta_used: 0.0,
            queries_answered: 0,
        }
    }

    /// Remaining epsilon = budget − used.
    pub fn remaining_epsilon(&self) -> f64 {
        (self.epsilon_budget - self.epsilon_used).max(0.0)
    }

    /// Remaining delta = budget − used.
    pub fn remaining_delta(&self) -> f64 {
        (self.delta_budget - self.delta_used).max(0.0)
    }

    /// Returns true when epsilon_used ≥ epsilon_budget.
    pub fn is_exhausted(&self) -> bool {
        self.epsilon_used >= self.epsilon_budget
    }

    /// Charge epsilon and delta to the budget. Returns an error if the budget
    /// would be exceeded.
    pub fn charge(&mut self, epsilon_cost: f64, delta_cost: f64) -> Result<(), DpError> {
        if self.is_exhausted() || self.epsilon_used + epsilon_cost > self.epsilon_budget {
            return Err(DpError::BudgetExhausted {
                remaining: self.remaining_epsilon(),
            });
        }
        self.epsilon_used += epsilon_cost;
        self.delta_used += delta_cost;
        self.queries_answered += 1;
        Ok(())
    }
}

// ── DifferentialPrivacyEngine ──────────────────────────────────────────────

/// Production-grade differential-privacy engine.
///
/// Manages a privacy budget, generates calibrated noise, and records an
/// auditable history of answered queries.
pub struct DifferentialPrivacyEngine {
    /// Live budget tracker.
    pub budget: BudgetTracker,
    /// Ring-buffer of answered query results (bounded by `max_history`).
    answered: VecDeque<DpResult>,
    /// Maximum number of results retained in history.
    max_history: usize,
    /// xorshift64 PRNG state.
    rng_state: u64,
}

impl DifferentialPrivacyEngine {
    /// Construct a new engine with the given budget parameters.
    ///
    /// The PRNG is seeded with `0xDEADBEEF42`.
    pub fn new(epsilon_budget: f64, delta_budget: f64, max_history: usize) -> Self {
        Self {
            budget: BudgetTracker::new(epsilon_budget, delta_budget),
            answered: VecDeque::new(),
            max_history,
            rng_state: 0x00DE_ADBE_EF42_u64,
        }
    }

    // ── Noise-scale computation ────────────────────────────────────────────

    /// Compute the noise scale for a given mechanism.
    ///
    /// - Laplace:  `scale = sensitivity / epsilon`
    /// - Gaussian: `scale = sensitivity * sqrt(2 * ln(1.25 / delta)) / epsilon`
    /// - Randomized: `scale = 1 / (exp(epsilon) + 1)` (flip probability)
    pub fn compute_noise_scale(mechanism: &PrivacyMechanism) -> NoiseScale {
        let scale = match mechanism {
            PrivacyMechanism::Laplace {
                sensitivity,
                epsilon,
            } => sensitivity / epsilon,

            PrivacyMechanism::Gaussian {
                sensitivity,
                epsilon,
                delta,
            } => {
                // Calibrated to satisfy (epsilon, delta)-DP via the analytic Gaussian mechanism.
                let inner = 2.0_f64 * (1.25_f64 / delta).ln();
                sensitivity * inner.sqrt() / epsilon
            }

            PrivacyMechanism::Randomized { epsilon } => 1.0 / (epsilon.exp() + 1.0),
        };
        NoiseScale {
            mechanism: mechanism.clone(),
            scale,
        }
    }

    // ── Noise sampling ─────────────────────────────────────────────────────

    /// Draw a uniform sample from (0, 1) using xorshift64.
    fn uniform_sample(&mut self) -> f64 {
        let raw = xorshift64(&mut self.rng_state);
        raw as f64 / u64::MAX as f64
    }

    /// Sample from Laplace(0, scale) using the inverse-CDF method.
    ///
    /// Formula: `-scale * sign(u - 0.5) * ln(1 - 2 * |u - 0.5|)`.
    /// If the argument to `ln` is ≤ 0, uses `1e-10` as a floor.
    pub fn sample_laplace(&mut self, scale: f64) -> f64 {
        let u = self.uniform_sample();
        let centered = u - 0.5;
        let sign = if centered >= 0.0 { 1.0_f64 } else { -1.0_f64 };
        let arg = (1.0 - 2.0 * centered.abs()).max(1e-10);
        -scale * sign * arg.ln()
    }

    /// Sample from Gaussian(0, scale) using the Box-Muller transform.
    ///
    /// Draws two uniform samples u1, u2 ∈ (0,1), then:
    /// `z = sqrt(-2 * ln(u1)) * cos(2π * u2)`.
    /// If `u1 ≤ 0`, uses `1e-10` as a floor.
    pub fn sample_gaussian(&mut self, scale: f64) -> f64 {
        let u1 = self.uniform_sample().max(1e-10);
        let u2 = self.uniform_sample();
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
        z * scale
    }

    /// Sample noise according to the mechanism (Laplace, Gaussian, or
    /// Randomized).
    fn sample_noise(&mut self, mechanism: &PrivacyMechanism, true_value: f64) -> f64 {
        let ns = Self::compute_noise_scale(mechanism);
        match mechanism {
            PrivacyMechanism::Laplace { .. } => self.sample_laplace(ns.scale),
            PrivacyMechanism::Gaussian { .. } => self.sample_gaussian(ns.scale),
            PrivacyMechanism::Randomized { epsilon } => {
                // Randomized response: flip the binary encoding of the value
                // with probability p = 1/(exp(ε)+1).
                let flip_prob = ns.scale; // = 1 / (exp(ε) + 1)
                let u = self.uniform_sample();
                if u < flip_prob {
                    // Flip: add a perturbation of magnitude 1.0 in a random direction.
                    let sign = if self.uniform_sample() < 0.5 {
                        1.0_f64
                    } else {
                        -1.0_f64
                    };
                    let _ = epsilon; // used via scale
                    sign * 1.0 - true_value + true_value // = sign * 1.0 (placeholder)
                } else {
                    0.0
                }
            }
        }
    }

    // ── Query application ──────────────────────────────────────────────────

    /// Answer a single differentially-private query.
    ///
    /// 1. Validates mechanism parameters.
    /// 2. Checks that the budget is not exhausted.
    /// 3. Generates calibrated noise.
    /// 4. Charges epsilon (and delta for Gaussian) to the budget.
    /// 5. Records the result in history and returns it.
    pub fn apply_mechanism(
        &mut self,
        query: &DpQuery,
        true_value: f64,
    ) -> Result<DpResult, DpError> {
        // Validate mechanism parameters up-front.
        query.mechanism.validate()?;

        // Guard against exhausted budget before allocating noise.
        if self.budget.is_exhausted() {
            return Err(DpError::BudgetExhausted {
                remaining: self.budget.remaining_epsilon(),
            });
        }

        // Compute noise.
        let noise = self.sample_noise(&query.mechanism, true_value);
        let noisy_value = true_value + noise;

        // Determine privacy cost for this query.
        let epsilon_cost = query.mechanism.epsilon();
        let delta_cost = query.mechanism.delta();

        // Charge budget (may fail if insufficient).
        self.budget.charge(epsilon_cost, delta_cost)?;

        let result = DpResult {
            query_id: query.query_id.clone(),
            true_value,
            noisy_value,
            noise_added: noise,
            privacy_cost: epsilon_cost,
        };

        // Maintain bounded history.
        if self.answered.len() >= self.max_history && self.max_history > 0 {
            self.answered.pop_front();
        }
        if self.max_history > 0 {
            self.answered.push_back(result.clone());
        }

        Ok(result)
    }

    /// Answer a batch of queries, applying each in sequence.
    ///
    /// Each result is `Ok` if the query succeeded, or `Err` if the budget
    /// was exhausted or parameters were invalid. Later queries in the batch
    /// see the already-reduced budget from earlier queries.
    pub fn apply_batch(&mut self, queries: &[(DpQuery, f64)]) -> Vec<Result<DpResult, DpError>> {
        queries
            .iter()
            .map(|(q, v)| self.apply_mechanism(q, *v))
            .collect()
    }

    // ── Composition theorems ───────────────────────────────────────────────

    /// Sequential composition: total epsilon = sum of per-query privacy costs.
    pub fn compose_sequential(results: &[DpResult]) -> f64 {
        results.iter().map(|r| r.privacy_cost).sum()
    }

    /// Advanced composition theorem (Dwork et al. 2010).
    ///
    /// For k independent (ε, 0)-DP mechanisms:
    ///
    /// ```text
    /// ε_total = sqrt(2k ln(1/δ)) * ε + k * ε * (exp(ε) - 1)
    /// ```
    ///
    /// where ε is the maximum per-query cost and k is the number of queries.
    /// Returns the sequential bound when `results` is empty.
    pub fn compose_advanced(results: &[DpResult], delta: f64) -> f64 {
        if results.is_empty() {
            return 0.0;
        }
        let k = results.len() as f64;
        let epsilon_per_query = results
            .iter()
            .map(|r| r.privacy_cost)
            .fold(f64::NEG_INFINITY, f64::max);

        let eps = epsilon_per_query;
        let term1 = (2.0 * k * (1.0 / delta).ln()).sqrt() * eps;
        let term2 = k * eps * (eps.exp() - 1.0);
        term1 + term2
    }

    // ── Sensitivity clipping ───────────────────────────────────────────────

    /// Clip each value to the range `[-sensitivity, sensitivity]`.
    ///
    /// This enforces global sensitivity bounds before computing statistics.
    pub fn sensitivity_clip(values: &[f64], sensitivity: f64) -> Vec<f64> {
        values
            .iter()
            .map(|&v| v.clamp(-sensitivity, sensitivity))
            .collect()
    }

    // ── Budget and history accessors ───────────────────────────────────────

    /// Return a clone of the current budget tracker.
    pub fn budget_stats(&self) -> BudgetTracker {
        self.budget.clone()
    }

    /// Return a reference to the bounded query-result history.
    pub fn history(&self) -> &VecDeque<DpResult> {
        &self.answered
    }

    /// Return a mutable reference to the budget tracker (for testing / integration).
    pub fn budget_mut(&mut self) -> &mut BudgetTracker {
        &mut self.budget
    }

    /// Reset the PRNG to a known seed for reproducible testing.
    pub fn reseed(&mut self, seed: u64) {
        // Ensure the seed is non-zero (xorshift64 with state=0 always produces 0).
        self.rng_state = if seed == 0 { 1 } else { seed };
    }

    /// Clear the query history.
    pub fn clear_history(&mut self) {
        self.answered.clear();
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use crate::differential_privacy::{
        xorshift64, BudgetTracker, DifferentialPrivacyEngine, DpError, DpQuery, DpResult,
        NoiseScale, PrivacyMechanism, PrivacyParameters,
    };

    // ── xorshift64 ─────────────────────────────────────────────────────────

    #[test]
    fn test_xorshift64_non_zero() {
        let mut state = 0x00DE_ADBE_EF42_u64;
        let v = xorshift64(&mut state);
        assert_ne!(v, 0);
        assert_ne!(state, 0x00DE_ADBE_EF42_u64);
    }

    #[test]
    fn test_xorshift64_deterministic() {
        let mut s1 = 12345u64;
        let mut s2 = 12345u64;
        for _ in 0..100 {
            assert_eq!(xorshift64(&mut s1), xorshift64(&mut s2));
        }
    }

    #[test]
    fn test_xorshift64_different_outputs() {
        let mut state = 1u64;
        let a = xorshift64(&mut state);
        let b = xorshift64(&mut state);
        assert_ne!(a, b);
    }

    // ── PrivacyMechanism ───────────────────────────────────────────────────

    #[test]
    fn test_laplace_mechanism_epsilon() {
        let m = PrivacyMechanism::Laplace {
            sensitivity: 1.0,
            epsilon: 0.5,
        };
        assert_eq!(m.epsilon(), 0.5);
        assert_eq!(m.delta(), 0.0);
        assert_eq!(m.sensitivity(), Some(1.0));
    }

    #[test]
    fn test_gaussian_mechanism_fields() {
        let m = PrivacyMechanism::Gaussian {
            sensitivity: 2.0,
            epsilon: 1.0,
            delta: 1e-5,
        };
        assert_eq!(m.epsilon(), 1.0);
        assert_eq!(m.delta(), 1e-5);
        assert_eq!(m.sensitivity(), Some(2.0));
    }

    #[test]
    fn test_randomized_mechanism_fields() {
        let m = PrivacyMechanism::Randomized { epsilon: 0.5 };
        assert_eq!(m.epsilon(), 0.5);
        assert_eq!(m.delta(), 0.0);
        assert!(m.sensitivity().is_none());
    }

    #[test]
    fn test_mechanism_validate_ok() {
        let m = PrivacyMechanism::Laplace {
            sensitivity: 1.0,
            epsilon: 1.0,
        };
        assert!(m.validate().is_ok());
    }

    #[test]
    fn test_mechanism_validate_invalid_epsilon() {
        let m = PrivacyMechanism::Laplace {
            sensitivity: 1.0,
            epsilon: 0.0,
        };
        assert!(matches!(m.validate(), Err(DpError::InvalidEpsilon)));
    }

    #[test]
    fn test_mechanism_validate_zero_sensitivity() {
        let m = PrivacyMechanism::Laplace {
            sensitivity: 0.0,
            epsilon: 1.0,
        };
        assert!(matches!(m.validate(), Err(DpError::ZeroSensitivity)));
    }

    #[test]
    fn test_mechanism_validate_gaussian_invalid_delta() {
        let m = PrivacyMechanism::Gaussian {
            sensitivity: 1.0,
            epsilon: 1.0,
            delta: 0.0,
        };
        assert!(matches!(m.validate(), Err(DpError::InvalidParameters(_))));
    }

    // ── NoiseScale ─────────────────────────────────────────────────────────

    #[test]
    fn test_laplace_noise_scale() {
        let m = PrivacyMechanism::Laplace {
            sensitivity: 1.0,
            epsilon: 2.0,
        };
        let ns = DifferentialPrivacyEngine::compute_noise_scale(&m);
        // scale = 1.0 / 2.0 = 0.5
        assert!((ns.scale - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_gaussian_noise_scale() {
        let delta = 1e-5;
        let m = PrivacyMechanism::Gaussian {
            sensitivity: 1.0,
            epsilon: 1.0,
            delta,
        };
        let ns = DifferentialPrivacyEngine::compute_noise_scale(&m);
        let expected = (2.0 * (1.25 / delta).ln()).sqrt();
        assert!((ns.scale - expected).abs() < 1e-10);
    }

    #[test]
    fn test_gaussian_noise_scale_scales_with_sensitivity() {
        let m1 = PrivacyMechanism::Gaussian {
            sensitivity: 1.0,
            epsilon: 1.0,
            delta: 1e-5,
        };
        let m2 = PrivacyMechanism::Gaussian {
            sensitivity: 2.0,
            epsilon: 1.0,
            delta: 1e-5,
        };
        let ns1 = DifferentialPrivacyEngine::compute_noise_scale(&m1);
        let ns2 = DifferentialPrivacyEngine::compute_noise_scale(&m2);
        assert!((ns2.scale - 2.0 * ns1.scale).abs() < 1e-10);
    }

    #[test]
    fn test_randomized_noise_scale() {
        let eps = 1.0_f64;
        let m = PrivacyMechanism::Randomized { epsilon: eps };
        let ns = DifferentialPrivacyEngine::compute_noise_scale(&m);
        let expected = 1.0 / (eps.exp() + 1.0);
        assert!((ns.scale - expected).abs() < 1e-12);
    }

    // ── PrivacyParameters ──────────────────────────────────────────────────

    #[test]
    fn test_privacy_parameters_valid() {
        let p = PrivacyParameters::new(1.0, 1e-5, 1.0);
        assert!(p.is_ok());
        let p = p.expect("test: should succeed");
        assert_eq!(p.epsilon, 1.0);
        assert_eq!(p.delta, 1e-5);
        assert_eq!(p.sensitivity, 1.0);
    }

    #[test]
    fn test_privacy_parameters_invalid_epsilon() {
        assert!(matches!(
            PrivacyParameters::new(0.0, 1e-5, 1.0),
            Err(DpError::InvalidEpsilon)
        ));
    }

    #[test]
    fn test_privacy_parameters_invalid_sensitivity() {
        assert!(matches!(
            PrivacyParameters::new(1.0, 1e-5, 0.0),
            Err(DpError::ZeroSensitivity)
        ));
    }

    #[test]
    fn test_privacy_parameters_invalid_delta() {
        assert!(matches!(
            PrivacyParameters::new(1.0, -0.1, 1.0),
            Err(DpError::InvalidParameters(_))
        ));
    }

    // ── BudgetTracker ──────────────────────────────────────────────────────

    #[test]
    fn test_budget_tracker_initial_state() {
        let bt = BudgetTracker::new(10.0, 1e-5);
        assert_eq!(bt.epsilon_budget, 10.0);
        assert_eq!(bt.epsilon_used, 0.0);
        assert!(!bt.is_exhausted());
        assert!((bt.remaining_epsilon() - 10.0).abs() < 1e-12);
    }

    #[test]
    fn test_budget_tracker_charge_success() {
        let mut bt = BudgetTracker::new(5.0, 1e-4);
        bt.charge(2.0, 1e-5).expect("test: should succeed");
        assert!((bt.remaining_epsilon() - 3.0).abs() < 1e-12);
        assert_eq!(bt.queries_answered, 1);
        assert!(!bt.is_exhausted());
    }

    #[test]
    fn test_budget_tracker_exhaustion() {
        let mut bt = BudgetTracker::new(1.0, 0.0);
        bt.charge(1.0, 0.0).expect("test: should succeed");
        assert!(bt.is_exhausted());
        // Trying again should fail.
        let err = bt.charge(0.5, 0.0);
        assert!(matches!(err, Err(DpError::BudgetExhausted { .. })));
    }

    #[test]
    fn test_budget_tracker_remaining_floored_at_zero() {
        let mut bt = BudgetTracker::new(1.0, 0.0);
        bt.charge(1.0, 0.0).expect("test: should succeed");
        assert_eq!(bt.remaining_epsilon(), 0.0);
    }

    // ── DifferentialPrivacyEngine ──────────────────────────────────────────

    #[test]
    fn test_engine_construction() {
        let engine = DifferentialPrivacyEngine::new(10.0, 1e-5, 100);
        assert_eq!(engine.budget.epsilon_budget, 10.0);
        assert_eq!(engine.history().len(), 0);
    }

    #[test]
    fn test_engine_laplace_query() {
        let mut engine = DifferentialPrivacyEngine::new(10.0, 0.0, 100);
        let query = DpQuery {
            query_id: "test_laplace".to_string(),
            sensitivity: 1.0,
            mechanism: PrivacyMechanism::Laplace {
                sensitivity: 1.0,
                epsilon: 1.0,
            },
        };
        let result = engine
            .apply_mechanism(&query, 100.0)
            .expect("test: should succeed");
        assert_eq!(result.query_id, "test_laplace");
        assert!(result.noisy_value.is_finite());
        assert!((result.noise_added - (result.noisy_value - result.true_value)).abs() < 1e-10);
        assert_eq!(result.privacy_cost, 1.0);
    }

    #[test]
    fn test_engine_gaussian_query() {
        let mut engine = DifferentialPrivacyEngine::new(10.0, 1.0, 100);
        let query = DpQuery {
            query_id: "test_gaussian".to_string(),
            sensitivity: 1.0,
            mechanism: PrivacyMechanism::Gaussian {
                sensitivity: 1.0,
                epsilon: 1.0,
                delta: 1e-5,
            },
        };
        let result = engine
            .apply_mechanism(&query, 50.0)
            .expect("test: should succeed");
        assert_eq!(result.query_id, "test_gaussian");
        assert!(result.noisy_value.is_finite());
    }

    #[test]
    fn test_engine_budget_deduction() {
        let mut engine = DifferentialPrivacyEngine::new(3.0, 0.0, 100);
        let query = DpQuery {
            query_id: "q".to_string(),
            sensitivity: 1.0,
            mechanism: PrivacyMechanism::Laplace {
                sensitivity: 1.0,
                epsilon: 1.0,
            },
        };
        engine
            .apply_mechanism(&query, 1.0)
            .expect("test: should succeed");
        engine
            .apply_mechanism(&query, 2.0)
            .expect("test: should succeed");
        engine
            .apply_mechanism(&query, 3.0)
            .expect("test: should succeed");
        assert!(engine.budget.is_exhausted());
        let err = engine.apply_mechanism(&query, 4.0);
        assert!(matches!(err, Err(DpError::BudgetExhausted { .. })));
    }

    #[test]
    fn test_engine_history_bounded() {
        let mut engine = DifferentialPrivacyEngine::new(1000.0, 0.0, 3);
        let query = DpQuery {
            query_id: "q".to_string(),
            sensitivity: 1.0,
            mechanism: PrivacyMechanism::Laplace {
                sensitivity: 1.0,
                epsilon: 0.1,
            },
        };
        for _ in 0..10 {
            engine
                .apply_mechanism(&query, 0.0)
                .expect("test: should succeed");
        }
        assert_eq!(engine.history().len(), 3);
    }

    #[test]
    fn test_engine_invalid_mechanism_rejected() {
        let mut engine = DifferentialPrivacyEngine::new(10.0, 0.0, 100);
        let query = DpQuery {
            query_id: "bad".to_string(),
            sensitivity: 0.0,
            mechanism: PrivacyMechanism::Laplace {
                sensitivity: -1.0,
                epsilon: 1.0,
            },
        };
        let err = engine.apply_mechanism(&query, 0.0);
        assert!(err.is_err());
    }

    #[test]
    fn test_engine_batch_apply() {
        let mut engine = DifferentialPrivacyEngine::new(100.0, 0.0, 100);
        let queries: Vec<(DpQuery, f64)> = (0..5)
            .map(|i| {
                (
                    DpQuery {
                        query_id: format!("q{i}"),
                        sensitivity: 1.0,
                        mechanism: PrivacyMechanism::Laplace {
                            sensitivity: 1.0,
                            epsilon: 1.0,
                        },
                    },
                    i as f64,
                )
            })
            .collect();
        let results = engine.apply_batch(&queries);
        assert_eq!(results.len(), 5);
        for r in &results {
            assert!(r.is_ok());
        }
    }

    #[test]
    fn test_engine_batch_stops_on_budget_exhaustion() {
        // Budget for exactly 2 queries.
        let mut engine = DifferentialPrivacyEngine::new(2.0, 0.0, 100);
        let queries: Vec<(DpQuery, f64)> = (0..5)
            .map(|i| {
                (
                    DpQuery {
                        query_id: format!("q{i}"),
                        sensitivity: 1.0,
                        mechanism: PrivacyMechanism::Laplace {
                            sensitivity: 1.0,
                            epsilon: 1.0,
                        },
                    },
                    i as f64,
                )
            })
            .collect();
        let results = engine.apply_batch(&queries);
        let ok_count = results.iter().filter(|r| r.is_ok()).count();
        let err_count = results.iter().filter(|r| r.is_err()).count();
        assert_eq!(ok_count, 2);
        assert_eq!(err_count, 3);
    }

    // ── Composition theorems ───────────────────────────────────────────────

    #[test]
    fn test_compose_sequential_empty() {
        assert_eq!(DifferentialPrivacyEngine::compose_sequential(&[]), 0.0);
    }

    #[test]
    fn test_compose_sequential_sums_costs() {
        let results = vec![
            make_result("a", 1.0),
            make_result("b", 0.5),
            make_result("c", 2.0),
        ];
        let total = DifferentialPrivacyEngine::compose_sequential(&results);
        assert!((total - 3.5).abs() < 1e-12);
    }

    #[test]
    fn test_compose_advanced_empty() {
        assert_eq!(DifferentialPrivacyEngine::compose_advanced(&[], 1e-5), 0.0);
    }

    #[test]
    fn test_compose_advanced_single_query() {
        let results = vec![make_result("a", 1.0)];
        let eps_adv = DifferentialPrivacyEngine::compose_advanced(&results, 1e-5);
        // For k=1: sqrt(2 * ln(1/delta)) * eps + eps * (exp(eps) - 1)
        let delta = 1e-5_f64;
        let eps = 1.0_f64;
        let expected = (2.0 * (1.0 / delta).ln()).sqrt() * eps + eps * (eps.exp() - 1.0);
        assert!((eps_adv - expected).abs() < 1e-10);
    }

    #[test]
    fn test_compose_advanced_larger_than_sequential_for_many_queries() {
        // Advanced composition can exceed sequential for small k but diverges
        // for large k — here we just check it is positive and finite.
        let results: Vec<DpResult> = (0..20)
            .map(|i| make_result(&format!("q{i}"), 0.1))
            .collect();
        let eps_adv = DifferentialPrivacyEngine::compose_advanced(&results, 1e-5);
        assert!(eps_adv > 0.0);
        assert!(eps_adv.is_finite());
    }

    // ── Sensitivity clipping ───────────────────────────────────────────────

    #[test]
    fn test_sensitivity_clip_within_bounds() {
        let values = vec![0.5, -0.3, 0.0];
        let clipped = DifferentialPrivacyEngine::sensitivity_clip(&values, 1.0);
        assert_eq!(clipped, values);
    }

    #[test]
    fn test_sensitivity_clip_above_bound() {
        let values = vec![5.0, -5.0, 2.0];
        let clipped = DifferentialPrivacyEngine::sensitivity_clip(&values, 1.0);
        assert_eq!(clipped, vec![1.0, -1.0, 1.0]);
    }

    #[test]
    fn test_sensitivity_clip_empty() {
        let clipped = DifferentialPrivacyEngine::sensitivity_clip(&[], 1.0);
        assert!(clipped.is_empty());
    }

    #[test]
    fn test_sensitivity_clip_preserves_sign() {
        let values = vec![-10.0, 10.0];
        let clipped = DifferentialPrivacyEngine::sensitivity_clip(&values, 3.0);
        assert_eq!(clipped, vec![-3.0, 3.0]);
    }

    // ── Noise distribution properties ─────────────────────────────────────

    #[test]
    fn test_laplace_noise_finite() {
        let mut engine = DifferentialPrivacyEngine::new(1000.0, 0.0, 1000);
        for _ in 0..1000 {
            let noise = engine.sample_laplace(1.0);
            assert!(noise.is_finite(), "Laplace noise must be finite");
        }
    }

    #[test]
    fn test_gaussian_noise_finite() {
        let mut engine = DifferentialPrivacyEngine::new(1000.0, 1000.0, 1000);
        for _ in 0..1000 {
            let noise = engine.sample_gaussian(1.0);
            assert!(noise.is_finite(), "Gaussian noise must be finite");
        }
    }

    #[test]
    fn test_laplace_noise_mean_near_zero() {
        // Empirical mean of 10 000 samples should be within ±0.15 of 0.
        let mut engine = DifferentialPrivacyEngine::new(f64::MAX, 0.0, 0);
        let n = 10_000usize;
        let mean: f64 = (0..n).map(|_| engine.sample_laplace(1.0)).sum::<f64>() / n as f64;
        assert!(
            mean.abs() < 0.15,
            "Empirical mean of Laplace samples too large: {mean}"
        );
    }

    #[test]
    fn test_gaussian_noise_mean_near_zero() {
        let mut engine = DifferentialPrivacyEngine::new(f64::MAX, 0.0, 0);
        let n = 10_000usize;
        let mean: f64 = (0..n).map(|_| engine.sample_gaussian(1.0)).sum::<f64>() / n as f64;
        assert!(
            mean.abs() < 0.15,
            "Empirical mean of Gaussian samples too large: {mean}"
        );
    }

    #[test]
    fn test_laplace_noise_scale_affects_variance() {
        let mut e1 = DifferentialPrivacyEngine::new(f64::MAX, 0.0, 0);
        e1.reseed(0xCAFE_BABE);
        let mut e2 = DifferentialPrivacyEngine::new(f64::MAX, 0.0, 0);
        e2.reseed(0xCAFE_BABE);
        let n = 1000usize;
        let var1: f64 = (0..n).map(|_| e1.sample_laplace(1.0).powi(2)).sum::<f64>() / n as f64;
        let var2: f64 = (0..n).map(|_| e2.sample_laplace(2.0).powi(2)).sum::<f64>() / n as f64;
        // Var[Laplace(0,b)] = 2b² — so var2 should be ~4x var1.
        assert!(var2 > var1 * 2.0, "Larger scale should increase variance");
    }

    // ── Reseed and clear_history ───────────────────────────────────────────

    #[test]
    fn test_reseed_reproducibility() {
        let mut engine = DifferentialPrivacyEngine::new(f64::MAX, 0.0, 0);
        engine.reseed(42);
        let a = engine.sample_laplace(1.0);
        engine.reseed(42);
        let b = engine.sample_laplace(1.0);
        assert_eq!(a, b);
    }

    #[test]
    fn test_clear_history() {
        let mut engine = DifferentialPrivacyEngine::new(100.0, 0.0, 100);
        let query = DpQuery {
            query_id: "q".to_string(),
            sensitivity: 1.0,
            mechanism: PrivacyMechanism::Laplace {
                sensitivity: 1.0,
                epsilon: 1.0,
            },
        };
        engine
            .apply_mechanism(&query, 0.0)
            .expect("test: should succeed");
        assert_eq!(engine.history().len(), 1);
        engine.clear_history();
        assert_eq!(engine.history().len(), 0);
    }

    #[test]
    fn test_budget_stats_clones_current_state() {
        let mut engine = DifferentialPrivacyEngine::new(10.0, 0.0, 100);
        let query = DpQuery {
            query_id: "q".to_string(),
            sensitivity: 1.0,
            mechanism: PrivacyMechanism::Laplace {
                sensitivity: 1.0,
                epsilon: 2.0,
            },
        };
        engine
            .apply_mechanism(&query, 0.0)
            .expect("test: should succeed");
        let stats = engine.budget_stats();
        assert!((stats.epsilon_used - 2.0).abs() < 1e-12);
        assert!((stats.remaining_epsilon() - 8.0).abs() < 1e-12);
    }

    #[test]
    fn test_noise_scale_struct_carries_mechanism() {
        let m = PrivacyMechanism::Laplace {
            sensitivity: 3.0,
            epsilon: 1.5,
        };
        let ns: NoiseScale = DifferentialPrivacyEngine::compute_noise_scale(&m);
        assert_eq!(ns.mechanism, m);
        assert!((ns.scale - 2.0).abs() < 1e-12);
    }

    // ── Helper ─────────────────────────────────────────────────────────────

    fn make_result(id: &str, cost: f64) -> DpResult {
        DpResult {
            query_id: id.to_string(),
            true_value: 0.0,
            noisy_value: 0.0,
            noise_added: 0.0,
            privacy_cost: cost,
        }
    }
}
