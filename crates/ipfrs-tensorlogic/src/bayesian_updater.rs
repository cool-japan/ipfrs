//! Bayesian belief updating with conjugate priors, likelihood functions,
//! and posterior inference.
//!
//! This module provides a production-grade [`BayesianUpdateEngine`] that
//! supports the four canonical conjugate-prior / likelihood pairs:
//!
//! | Prior      | Likelihood  |
//! |------------|-------------|
//! | Beta       | Bernoulli   |
//! | Gaussian   | Gaussian    |
//! | Dirichlet  | Categorical |
//! | Gamma      | Poisson     |
//!
//! # Example
//!
//! ```
//! use ipfrs_tensorlogic::bayesian_updater::{
//!     BayesianUpdateEngine, Prior, Observation,
//! };
//!
//! let mut engine = BayesianUpdateEngine::new(64);
//!
//! // Start with a uniform Beta(1,1) prior and observe 7 successes in 10 trials.
//! let prior = Prior::Beta { alpha: 1.0, beta: 1.0 };
//! let obs   = Observation::Bernoulli { successes: 7, trials: 10 };
//!
//! let posterior = engine.update(prior, &obs).expect("example: should succeed in docs");
//! // Posterior should be Beta(8, 4)
//! if let Prior::Beta { alpha, beta } = &posterior.updated {
//!     assert!((alpha - 8.0).abs() < 1e-10);
//!     assert!((beta  - 4.0).abs() < 1e-10);
//! }
//! ```

use std::collections::VecDeque;
use thiserror::Error;

// ──────────────────────────────────────────────────────────────────────────────
// Error type
// ──────────────────────────────────────────────────────────────────────────────

/// Errors that can arise during Bayesian updating.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum BayesError {
    /// The prior distribution family does not match the observation type.
    #[error(
        "prior/observation mismatch: cannot update {prior_type} prior with {obs_type} observation"
    )]
    PriorObservationMismatch {
        /// Name of the prior family (e.g. "Beta").
        prior_type: String,
        /// Name of the observation type (e.g. "Gaussian").
        obs_type: String,
    },

    /// One or more parameter values are invalid (e.g. non-positive concentration).
    #[error("invalid parameters: {0}")]
    InvalidParameters(String),

    /// The requested operation is not supported for this combination of types.
    #[error("unsupported operation: {0}")]
    UnsupportedOperation(String),

    /// A numerical error occurred (e.g. NaN or infinity).
    #[error("numerical error: {0}")]
    NumericalError(String),
}

// ──────────────────────────────────────────────────────────────────────────────
// Core enums and structs
// ──────────────────────────────────────────────────────────────────────────────

/// A conjugate prior distribution.
#[derive(Debug, Clone, PartialEq)]
pub enum Prior {
    /// Beta(α, β) — conjugate prior for Bernoulli/Binomial likelihoods.
    Beta {
        /// Pseudo-successes (must be > 0).
        alpha: f64,
        /// Pseudo-failures (must be > 0).
        beta: f64,
    },
    /// Normal(μ, σ²) — conjugate prior for Gaussian likelihoods with known
    /// observation variance.
    Gaussian {
        /// Prior mean.
        mean: f64,
        /// Prior variance (must be > 0).
        variance: f64,
    },
    /// Dirichlet(α₁, …, αₖ) — conjugate prior for Categorical likelihoods.
    Dirichlet {
        /// Concentration parameters (all must be > 0).
        alphas: Vec<f64>,
    },
    /// Gamma(shape, rate) — conjugate prior for Poisson likelihoods.
    Gamma {
        /// Shape parameter (must be > 0).
        shape: f64,
        /// Rate parameter (must be > 0).
        rate: f64,
    },
}

impl Prior {
    /// Human-readable name for error messages.
    fn type_name(&self) -> &'static str {
        match self {
            Prior::Beta { .. } => "Beta",
            Prior::Gaussian { .. } => "Gaussian",
            Prior::Dirichlet { .. } => "Dirichlet",
            Prior::Gamma { .. } => "Gamma",
        }
    }

    /// Validate that all parameters are in their legal ranges.
    fn validate(&self) -> Result<(), BayesError> {
        match self {
            Prior::Beta { alpha, beta } => {
                if *alpha <= 0.0 || alpha.is_nan() {
                    return Err(BayesError::InvalidParameters(format!(
                        "Beta alpha must be > 0, got {alpha}"
                    )));
                }
                if *beta <= 0.0 || beta.is_nan() {
                    return Err(BayesError::InvalidParameters(format!(
                        "Beta beta must be > 0, got {beta}"
                    )));
                }
            }
            Prior::Gaussian { variance, .. } => {
                if *variance <= 0.0 || variance.is_nan() {
                    return Err(BayesError::InvalidParameters(format!(
                        "Gaussian variance must be > 0, got {variance}"
                    )));
                }
            }
            Prior::Dirichlet { alphas } => {
                if alphas.is_empty() {
                    return Err(BayesError::InvalidParameters(
                        "Dirichlet alphas must be non-empty".to_string(),
                    ));
                }
                for (i, &a) in alphas.iter().enumerate() {
                    if a <= 0.0 || a.is_nan() {
                        return Err(BayesError::InvalidParameters(format!(
                            "Dirichlet alpha[{i}] must be > 0, got {a}"
                        )));
                    }
                }
            }
            Prior::Gamma { shape, rate } => {
                if *shape <= 0.0 || shape.is_nan() {
                    return Err(BayesError::InvalidParameters(format!(
                        "Gamma shape must be > 0, got {shape}"
                    )));
                }
                if *rate <= 0.0 || rate.is_nan() {
                    return Err(BayesError::InvalidParameters(format!(
                        "Gamma rate must be > 0, got {rate}"
                    )));
                }
            }
        }
        Ok(())
    }
}

/// An observed datum used to update a prior.
#[derive(Debug, Clone, PartialEq)]
pub enum Observation {
    /// Outcome of a series of Bernoulli trials.
    Bernoulli {
        /// Number of successes observed.
        successes: u64,
        /// Total number of trials.
        trials: u64,
    },
    /// Sufficient statistics from a Gaussian-distributed sample.
    Gaussian {
        /// Sample mean.
        sample_mean: f64,
        /// Sample variance (must be > 0).
        sample_variance: f64,
        /// Sample size.
        n: u64,
    },
    /// Category counts from a Categorical/Multinomial sample.
    Categorical {
        /// Observed count for each category (length must equal Dirichlet dim).
        counts: Vec<u64>,
    },
    /// Sufficient statistics from a Poisson process.
    Poisson {
        /// Total events observed.
        total_events: u64,
        /// Total elapsed time (must be > 0).
        total_time: f64,
    },
}

impl Observation {
    /// Human-readable name for error messages.
    fn type_name(&self) -> &'static str {
        match self {
            Observation::Bernoulli { .. } => "Bernoulli",
            Observation::Gaussian { .. } => "Gaussian",
            Observation::Categorical { .. } => "Categorical",
            Observation::Poisson { .. } => "Poisson",
        }
    }
}

/// The result of a single Bayesian update step.
#[derive(Debug, Clone)]
pub struct Posterior {
    /// The prior used in this update.
    pub prior: Prior,
    /// Human-readable description of the likelihood model.
    pub likelihood_type: String,
    /// The updated (posterior) distribution.
    pub updated: Prior,
    /// Log marginal likelihood ln p(observation | model).
    pub log_marginal: f64,
}

/// A credible interval `[lower, upper]` at the stated probability level.
#[derive(Debug, Clone, PartialEq)]
pub struct CredibleInterval {
    /// Lower bound of the interval.
    pub lower: f64,
    /// Upper bound of the interval.
    pub upper: f64,
    /// Nominal probability mass contained (e.g. `0.95` for a 95% CI).
    pub probability: f64,
}

// ──────────────────────────────────────────────────────────────────────────────
// Pure-Rust math helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Stirling-series approximation for ln Γ(x), accurate to ~1e-13 for x > 0.5.
/// Uses the recurrence Γ(x) = (x-1)·Γ(x-1) to shift small arguments up.
fn ln_gamma(x: f64) -> f64 {
    // Reflect small values upward via the recurrence ln Γ(x) = ln Γ(x+1) – ln x
    if x < 0.5 {
        // Use reflection: ln Γ(x) = ln π - ln sin(πx) - ln Γ(1-x)
        // But for our use-cases x is always positive, so just recurse.
        return ln_gamma(x + 1.0) - x.ln();
    }
    if x < 7.0 {
        // Shift x into the Stirling-series regime
        return ln_gamma(x + 1.0) - x.ln();
    }
    // Stirling's series: ln Γ(x) ≈ (x-0.5)·ln x - x + 0.5·ln(2π) + 1/(12x) - 1/(360x³) + …
    let half_ln_two_pi = 0.918_938_533_204_672_8_f64; // 0.5*ln(2π)
    let inv_x = 1.0 / x;
    let inv_x2 = inv_x * inv_x;
    (x - 0.5) * x.ln() - x
        + half_ln_two_pi
        + inv_x * (1.0 / 12.0 - inv_x2 * (1.0 / 360.0 - inv_x2 / 1260.0))
}

/// ln B(a, b) = ln Γ(a) + ln Γ(b) – ln Γ(a+b).
fn log_beta(a: f64, b: f64) -> f64 {
    ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b)
}

/// ln normalisation constant of the Dirichlet distribution:
/// Σᵢ ln Γ(αᵢ) – ln Γ(Σᵢ αᵢ).
fn log_dirichlet_norm(alphas: &[f64]) -> f64 {
    let sum: f64 = alphas.iter().sum();
    let sum_lg: f64 = alphas.iter().map(|&a| ln_gamma(a)).sum();
    sum_lg - ln_gamma(sum)
}

/// Digamma function ψ(x) = d/dx ln Γ(x).
///
/// Uses the asymptotic series for x > 6 and the recurrence ψ(x) = ψ(x+1) – 1/x
/// for smaller arguments.
fn digamma(x: f64) -> f64 {
    if x < 6.0 {
        // Recurrence: ψ(x) = ψ(x+1) - 1/x
        return digamma(x + 1.0) - 1.0 / x;
    }
    // Asymptotic expansion: ψ(x) ≈ ln x - 1/(2x) - 1/(12x²) + 1/(120x⁴) - 1/(252x⁶)
    let inv_x = 1.0 / x;
    let inv_x2 = inv_x * inv_x;
    x.ln() - 0.5 * inv_x - inv_x2 * (1.0 / 12.0 - inv_x2 * (1.0 / 120.0 - inv_x2 / 252.0))
}

/// Z-score for a given two-tailed credible probability using a rational
/// approximation to the probit function (accurate to ~1e-4).
///
/// Reference: Abramowitz & Stegun 26.2.17.
fn z_score(probability: f64) -> f64 {
    // We need the upper tail quantile for (1 + p) / 2
    let p = (1.0 + probability) / 2.0;
    // Rational approximation to Φ⁻¹(p) for 0.5 < p < 1
    if (p - 0.5).abs() < 1e-10 {
        return 0.0;
    }
    let t = (-2.0 * (1.0 - p).ln()).sqrt();
    let c0 = 2.515_517;
    let c1 = 0.802_853;
    let c2 = 0.010_328;
    let d1 = 1.432_788;
    let d2 = 0.189_269;
    let d3 = 0.001_308;
    let numer = c0 + c1 * t + c2 * t * t;
    let denom = 1.0 + d1 * t + d2 * t * t + d3 * t * t * t;
    t - numer / denom
}

/// Guard against NaN/Inf in a computed f64.
fn check_finite(val: f64, context: &str) -> Result<f64, BayesError> {
    if val.is_finite() {
        Ok(val)
    } else {
        Err(BayesError::NumericalError(format!(
            "{context}: computed non-finite value {val}"
        )))
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Engine
// ──────────────────────────────────────────────────────────────────────────────

/// Bayesian belief updating engine.
///
/// Maintains a bounded history of [`Posterior`] results and supports both
/// single-step updates and sequential folding of multiple observations.
#[derive(Debug)]
pub struct BayesianUpdateEngine {
    /// Ordered history of posteriors, newest at the back.
    history: VecDeque<Posterior>,
    /// Maximum number of posteriors to retain.
    max_history: usize,
}

impl BayesianUpdateEngine {
    /// Create a new engine with the given history capacity.
    pub fn new(max_history: usize) -> Self {
        Self {
            history: VecDeque::with_capacity(max_history.min(1024)),
            max_history,
        }
    }

    // ── Core update ──────────────────────────────────────────────────────────

    /// Perform a single Bayesian update, returning the posterior.
    ///
    /// The method validates parameters and dispatches to the appropriate
    /// conjugate-update formula.
    pub fn update(
        &mut self,
        prior: Prior,
        observation: &Observation,
    ) -> Result<Posterior, BayesError> {
        prior.validate()?;

        let posterior = match (&prior, observation) {
            // ── Beta–Bernoulli ────────────────────────────────────────────────
            (Prior::Beta { alpha, beta }, Observation::Bernoulli { successes, trials }) => {
                if successes > trials {
                    return Err(BayesError::InvalidParameters(format!(
                        "successes ({successes}) cannot exceed trials ({trials})"
                    )));
                }
                let s = *successes as f64;
                let f = (*trials - *successes) as f64;
                let alpha_post = alpha + s;
                let beta_post = beta + f;
                let log_marginal = check_finite(
                    log_beta(alpha_post, beta_post) - log_beta(*alpha, *beta),
                    "Beta-Bernoulli log_marginal",
                )?;
                Posterior {
                    prior: prior.clone(),
                    likelihood_type: "Bernoulli".to_string(),
                    updated: Prior::Beta {
                        alpha: alpha_post,
                        beta: beta_post,
                    },
                    log_marginal,
                }
            }

            // ── Gaussian–Gaussian (normal-normal conjugate) ───────────────────
            (
                Prior::Gaussian {
                    mean: prior_mean,
                    variance: prior_var,
                },
                Observation::Gaussian {
                    sample_mean,
                    sample_variance,
                    n,
                },
            ) => {
                if *sample_variance <= 0.0 || sample_variance.is_nan() {
                    return Err(BayesError::InvalidParameters(format!(
                        "sample_variance must be > 0, got {sample_variance}"
                    )));
                }
                if *n == 0 {
                    return Err(BayesError::InvalidParameters(
                        "n must be > 0 for Gaussian observation".to_string(),
                    ));
                }
                let n_f = *n as f64;
                // Posterior precision = 1/σ²_0 + n/σ²_obs
                let post_prec = 1.0 / prior_var + n_f / sample_variance;
                let post_var = 1.0 / post_prec;
                let post_mean =
                    post_var * (prior_mean / prior_var + n_f * sample_mean / sample_variance);

                // ln p(x̄ | model) ≈ -0.5 * ln(2π * (σ²_0 + σ²_obs / n))
                let effective_var = prior_var + sample_variance / n_f;
                let log_marginal = check_finite(
                    -0.5 * (std::f64::consts::TAU * effective_var).ln(),
                    "Gaussian-Gaussian log_marginal",
                )?;

                Posterior {
                    prior: prior.clone(),
                    likelihood_type: "Gaussian".to_string(),
                    updated: Prior::Gaussian {
                        mean: post_mean,
                        variance: post_var,
                    },
                    log_marginal,
                }
            }

            // ── Dirichlet–Categorical ─────────────────────────────────────────
            (Prior::Dirichlet { alphas }, Observation::Categorical { counts }) => {
                if alphas.len() != counts.len() {
                    return Err(BayesError::InvalidParameters(format!(
                        "Dirichlet dim {} != Categorical counts dim {}",
                        alphas.len(),
                        counts.len()
                    )));
                }
                let alphas_post: Vec<f64> = alphas
                    .iter()
                    .zip(counts.iter())
                    .map(|(&a, &c)| a + c as f64)
                    .collect();

                let log_marginal = check_finite(
                    log_dirichlet_norm(&alphas_post) - log_dirichlet_norm(alphas),
                    "Dirichlet-Categorical log_marginal",
                )?;

                Posterior {
                    prior: prior.clone(),
                    likelihood_type: "Categorical".to_string(),
                    updated: Prior::Dirichlet {
                        alphas: alphas_post,
                    },
                    log_marginal,
                }
            }

            // ── Gamma–Poisson ─────────────────────────────────────────────────
            (
                Prior::Gamma { shape, rate },
                Observation::Poisson {
                    total_events,
                    total_time,
                },
            ) => {
                if *total_time <= 0.0 || total_time.is_nan() {
                    return Err(BayesError::InvalidParameters(format!(
                        "total_time must be > 0, got {total_time}"
                    )));
                }
                let k = *total_events as f64;
                let shape_post = shape + k;
                let rate_post = rate + total_time;

                // ln p(data | model) = lgamma(shape') – lgamma(shape)
                //                    + shape * ln(rate) – shape' * ln(rate')
                let log_marginal = check_finite(
                    ln_gamma(shape_post) - ln_gamma(*shape) + shape * rate.ln()
                        - shape_post * rate_post.ln(),
                    "Gamma-Poisson log_marginal",
                )?;

                Posterior {
                    prior: prior.clone(),
                    likelihood_type: "Poisson".to_string(),
                    updated: Prior::Gamma {
                        shape: shape_post,
                        rate: rate_post,
                    },
                    log_marginal,
                }
            }

            // ── Mismatch ──────────────────────────────────────────────────────
            _ => {
                return Err(BayesError::PriorObservationMismatch {
                    prior_type: prior.type_name().to_string(),
                    obs_type: observation.type_name().to_string(),
                });
            }
        };

        // Store in history
        if self.history.len() >= self.max_history && self.max_history > 0 {
            self.history.pop_front();
        }
        if self.max_history > 0 {
            self.history.push_back(posterior.clone());
        }

        Ok(posterior)
    }

    // ── Sequential update ────────────────────────────────────────────────────

    /// Apply a sequence of observations left-to-right, using each posterior
    /// as the prior for the next update.
    ///
    /// Returns the final posterior, or an error at the first failing update.
    pub fn sequential_update(
        &mut self,
        prior: Prior,
        observations: &[Observation],
    ) -> Result<Posterior, BayesError> {
        if observations.is_empty() {
            return Err(BayesError::InvalidParameters(
                "observations slice must not be empty".to_string(),
            ));
        }

        let mut current_prior = prior;
        let mut last_posterior: Option<Posterior> = None;

        for obs in observations {
            let posterior = self.update(current_prior, obs)?;
            current_prior = posterior.updated.clone();
            last_posterior = Some(posterior);
        }

        // SAFETY: observations is non-empty, so last_posterior is Some.
        last_posterior.ok_or_else(|| {
            BayesError::NumericalError("unexpected empty observation sequence".to_string())
        })
    }

    // ── Credible interval ────────────────────────────────────────────────────

    /// Compute a symmetric credible interval for a posterior distribution.
    ///
    /// Uses normal / Wilson approximations — suitable for moderate-to-large
    /// concentration parameters.
    ///
    /// # Arguments
    /// * `posterior` – the posterior distribution.
    /// * `probability` – the desired probability mass (e.g. `0.95`).
    pub fn credible_interval(
        posterior: &Prior,
        probability: f64,
    ) -> Result<CredibleInterval, BayesError> {
        if !(0.0 < probability && probability < 1.0) {
            return Err(BayesError::InvalidParameters(format!(
                "probability must be in (0, 1), got {probability}"
            )));
        }
        posterior.validate()?;

        let z = z_score(probability);

        match posterior {
            Prior::Beta { alpha, beta } => {
                let n = alpha + beta;
                let center = alpha / n;
                let half_width = z * (center * (1.0 - center) / n).sqrt();
                let lower = (center - half_width).max(0.0);
                let upper = (center + half_width).min(1.0);
                Ok(CredibleInterval {
                    lower,
                    upper,
                    probability,
                })
            }

            Prior::Gaussian { mean, variance } => {
                let half_width = z * variance.sqrt();
                Ok(CredibleInterval {
                    lower: mean - half_width,
                    upper: mean + half_width,
                    probability,
                })
            }

            Prior::Gamma { shape, rate } => {
                // Normal approximation: mean = shape/rate, var = shape/rate²
                let mean = shape / rate;
                let std_dev = (shape / (rate * rate)).sqrt();
                let half_width = z * std_dev;
                let lower = (mean - half_width).max(0.0);
                let upper = mean + half_width;
                Ok(CredibleInterval {
                    lower,
                    upper,
                    probability,
                })
            }

            Prior::Dirichlet { alphas } => {
                // Return interval for the category with the highest concentration
                let sum: f64 = alphas.iter().sum();
                let max_alpha = alphas.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let center = max_alpha / sum;
                let half_width = z * (center * (1.0 - center) / (sum + 1.0)).sqrt();
                let lower = (center - half_width).max(0.0);
                let upper = (center + half_width).min(1.0);
                Ok(CredibleInterval {
                    lower,
                    upper,
                    probability,
                })
            }
        }
    }

    // ── MAP estimate ─────────────────────────────────────────────────────────

    /// Return the maximum a posteriori (MAP) estimate for a distribution.
    ///
    /// | Distribution | MAP |
    /// |---|---|
    /// | Beta(α,β) | (α-1)/(α+β-2) if α>1 && β>1, else α/(α+β) |
    /// | Gaussian(μ,σ²) | μ |
    /// | Gamma(k,r) | (k-1)/r if k>1, else 0 |
    /// | Dirichlet(α) | argmax(αᵢ) / Σαᵢ |
    pub fn map_estimate(posterior: &Prior) -> f64 {
        match posterior {
            Prior::Beta { alpha, beta } => {
                if *alpha > 1.0 && *beta > 1.0 {
                    (alpha - 1.0) / (alpha + beta - 2.0)
                } else {
                    alpha / (alpha + beta)
                }
            }
            Prior::Gaussian { mean, .. } => *mean,
            Prior::Gamma { shape, rate } => {
                if *shape > 1.0 {
                    (shape - 1.0) / rate
                } else {
                    0.0
                }
            }
            Prior::Dirichlet { alphas } => {
                let sum: f64 = alphas.iter().sum();
                let max_alpha = alphas.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                max_alpha / sum
            }
        }
    }

    // ── KL divergence ────────────────────────────────────────────────────────

    /// Compute the KL divergence KL(p ‖ q) between two distributions of the
    /// same family.
    ///
    /// Supported pairs: Beta vs Beta, Gaussian vs Gaussian.
    /// All other combinations return [`BayesError::UnsupportedOperation`].
    pub fn kl_divergence(p: &Prior, q: &Prior) -> Result<f64, BayesError> {
        match (p, q) {
            (
                Prior::Beta {
                    alpha: ap,
                    beta: bp,
                },
                Prior::Beta {
                    alpha: aq,
                    beta: bq,
                },
            ) => {
                // KL(Beta(αp,βp) ‖ Beta(αq,βq))
                // = log B(αq,βq) - log B(αp,βp)
                //   + (αp - αq) ψ(αp) + (βp - βq) ψ(βp)
                //   - (αp + βp - αq - βq) ψ(αp + βp)
                let psi_ap = digamma(*ap);
                let psi_bp = digamma(*bp);
                let psi_ap_bp = digamma(ap + bp);
                let kl = log_beta(*aq, *bq) - log_beta(*ap, *bp)
                    + (ap - aq) * psi_ap
                    + (bp - bq) * psi_bp
                    - ((ap + bp) - (aq + bq)) * psi_ap_bp;
                check_finite(kl, "KL(Beta‖Beta)")
            }

            (
                Prior::Gaussian {
                    mean: mp,
                    variance: vp,
                },
                Prior::Gaussian {
                    mean: mq,
                    variance: vq,
                },
            ) => {
                // KL(N(μp,σp²) ‖ N(μq,σq²))
                // = 0.5 * (ln(σq²/σp²) + σp²/σq² + (μp-μq)²/σq² - 1)
                let kl = 0.5 * ((vq / vp).ln() + vp / vq + (mp - mq) * (mp - mq) / vq - 1.0);
                check_finite(kl, "KL(Gaussian‖Gaussian)")
            }

            _ => Err(BayesError::UnsupportedOperation(format!(
                "KL divergence not implemented for {} vs {}",
                p.type_name(),
                q.type_name()
            ))),
        }
    }

    // ── History accessors ────────────────────────────────────────────────────

    /// Immutable reference to the update history (oldest first).
    pub fn history(&self) -> &VecDeque<Posterior> {
        &self.history
    }

    /// Clear the update history.
    pub fn clear_history(&mut self) {
        self.history.clear();
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::{
        digamma, ln_gamma, log_beta, log_dirichlet_norm, z_score, BayesError, BayesianUpdateEngine,
        Observation, Prior,
    };

    // ── Math helper tests ────────────────────────────────────────────────────

    #[test]
    fn ln_gamma_integer_values() {
        // Γ(n) = (n-1)! for positive integers
        // ln Γ(1) = 0
        assert!((ln_gamma(1.0)).abs() < 1e-8);
        // ln Γ(2) = 0
        assert!((ln_gamma(2.0)).abs() < 1e-8);
        // ln Γ(3) = ln 2 ≈ 0.6931
        assert!((ln_gamma(3.0) - 2.0_f64.ln()).abs() < 1e-8);
        // ln Γ(5) = ln 24
        assert!((ln_gamma(5.0) - 24.0_f64.ln()).abs() < 1e-7);
    }

    #[test]
    fn ln_gamma_half() {
        // Γ(1/2) = √π, so ln Γ(0.5) = 0.5 ln π
        let expected = 0.5 * std::f64::consts::PI.ln();
        assert!((ln_gamma(0.5) - expected).abs() < 1e-6);
    }

    #[test]
    fn log_beta_symmetry() {
        // B(a,b) == B(b,a)
        let diff = (log_beta(2.0, 5.0) - log_beta(5.0, 2.0)).abs();
        assert!(diff < 1e-12);
    }

    #[test]
    fn log_beta_known_value() {
        // B(1,1) = 1  →  ln B(1,1) = 0  (Stirling approximation, tolerance 1e-6)
        assert!(
            log_beta(1.0, 1.0).abs() < 1e-6,
            "log_beta(1,1) = {}",
            log_beta(1.0, 1.0)
        );
    }

    #[test]
    fn log_dirichlet_norm_two_dim_equals_log_beta() {
        // For k=2, Dirichlet norm = ln B(a1, a2)
        let a = 3.0_f64;
        let b = 7.0_f64;
        let dir = log_dirichlet_norm(&[a, b]);
        let lb = log_beta(a, b);
        assert!((dir - lb).abs() < 1e-10);
    }

    #[test]
    fn digamma_known_value() {
        // ψ(1) = -γ ≈ -0.5772156649
        let expected = -0.577_215_664_9_f64;
        assert!((digamma(1.0) - expected).abs() < 1e-4);
    }

    #[test]
    fn digamma_recurrence_property() {
        // ψ(x+1) - ψ(x) = 1/x
        let x = 4.5_f64;
        let diff = digamma(x + 1.0) - digamma(x);
        assert!((diff - 1.0 / x).abs() < 1e-8);
    }

    #[test]
    fn z_score_95_percent() {
        // 95% CI should give z ≈ 1.96
        let z = z_score(0.95);
        assert!((z - 1.96).abs() < 0.01);
    }

    #[test]
    fn z_score_99_percent() {
        // 99% CI should give z ≈ 2.576
        let z = z_score(0.99);
        assert!((z - 2.576).abs() < 0.01);
    }

    // ── Beta–Bernoulli update ────────────────────────────────────────────────

    #[test]
    fn beta_bernoulli_uniform_prior() {
        let mut engine = BayesianUpdateEngine::new(10);
        let prior = Prior::Beta {
            alpha: 1.0,
            beta: 1.0,
        };
        let obs = Observation::Bernoulli {
            successes: 7,
            trials: 10,
        };
        let post = engine
            .update(prior, &obs)
            .expect("test: TD update should succeed");
        match &post.updated {
            Prior::Beta { alpha, beta } => {
                assert!((alpha - 8.0).abs() < 1e-10);
                assert!((beta - 4.0).abs() < 1e-10);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn beta_bernoulli_all_successes() {
        let mut engine = BayesianUpdateEngine::new(10);
        let prior = Prior::Beta {
            alpha: 2.0,
            beta: 3.0,
        };
        let obs = Observation::Bernoulli {
            successes: 5,
            trials: 5,
        };
        let post = engine
            .update(prior, &obs)
            .expect("test: TD update should succeed");
        match post.updated {
            Prior::Beta { alpha, beta } => {
                assert!((alpha - 7.0).abs() < 1e-10);
                assert!((beta - 3.0).abs() < 1e-10);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn beta_bernoulli_zero_successes() {
        let mut engine = BayesianUpdateEngine::new(10);
        let prior = Prior::Beta {
            alpha: 1.0,
            beta: 1.0,
        };
        let obs = Observation::Bernoulli {
            successes: 0,
            trials: 5,
        };
        let post = engine
            .update(prior, &obs)
            .expect("test: TD update should succeed");
        match post.updated {
            Prior::Beta { alpha, beta } => {
                assert!((alpha - 1.0).abs() < 1e-10);
                assert!((beta - 6.0).abs() < 1e-10);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn beta_bernoulli_log_marginal_finite() {
        let mut engine = BayesianUpdateEngine::new(10);
        let prior = Prior::Beta {
            alpha: 2.0,
            beta: 2.0,
        };
        let obs = Observation::Bernoulli {
            successes: 3,
            trials: 6,
        };
        let post = engine
            .update(prior, &obs)
            .expect("test: TD update should succeed");
        assert!(post.log_marginal.is_finite());
    }

    #[test]
    fn beta_bernoulli_successes_exceed_trials_error() {
        let mut engine = BayesianUpdateEngine::new(10);
        let prior = Prior::Beta {
            alpha: 1.0,
            beta: 1.0,
        };
        let obs = Observation::Bernoulli {
            successes: 11,
            trials: 10,
        };
        let result = engine.update(prior, &obs);
        assert!(matches!(result, Err(BayesError::InvalidParameters(_))));
    }

    // ── Gaussian–Gaussian update ─────────────────────────────────────────────

    #[test]
    fn gaussian_gaussian_update_basic() {
        let mut engine = BayesianUpdateEngine::new(10);
        let prior = Prior::Gaussian {
            mean: 0.0,
            variance: 1.0,
        };
        let obs = Observation::Gaussian {
            sample_mean: 2.0,
            sample_variance: 1.0,
            n: 1,
        };
        let post = engine
            .update(prior, &obs)
            .expect("test: TD update should succeed");
        match post.updated {
            Prior::Gaussian { mean, variance } => {
                // post_var = 1/(1+1) = 0.5, post_mean = 0.5*(0+2) = 1.0
                assert!((variance - 0.5).abs() < 1e-10);
                assert!((mean - 1.0).abs() < 1e-10);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn gaussian_gaussian_large_n_pulls_to_sample() {
        let mut engine = BayesianUpdateEngine::new(10);
        let prior = Prior::Gaussian {
            mean: 0.0,
            variance: 100.0,
        };
        let obs = Observation::Gaussian {
            sample_mean: 5.0,
            sample_variance: 1.0,
            n: 1000,
        };
        let post = engine
            .update(prior, &obs)
            .expect("test: TD update should succeed");
        match post.updated {
            Prior::Gaussian { mean, .. } => {
                // With large n, posterior mean ≈ sample mean
                assert!((mean - 5.0).abs() < 0.1);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn gaussian_gaussian_log_marginal_finite() {
        let mut engine = BayesianUpdateEngine::new(10);
        let prior = Prior::Gaussian {
            mean: 1.0,
            variance: 2.0,
        };
        let obs = Observation::Gaussian {
            sample_mean: 1.5,
            sample_variance: 0.5,
            n: 10,
        };
        let post = engine
            .update(prior, &obs)
            .expect("test: TD update should succeed");
        assert!(post.log_marginal.is_finite());
    }

    #[test]
    fn gaussian_gaussian_zero_n_error() {
        let mut engine = BayesianUpdateEngine::new(10);
        let prior = Prior::Gaussian {
            mean: 0.0,
            variance: 1.0,
        };
        let obs = Observation::Gaussian {
            sample_mean: 1.0,
            sample_variance: 1.0,
            n: 0,
        };
        let result = engine.update(prior, &obs);
        assert!(matches!(result, Err(BayesError::InvalidParameters(_))));
    }

    // ── Dirichlet–Categorical update ─────────────────────────────────────────

    #[test]
    fn dirichlet_categorical_update_basic() {
        let mut engine = BayesianUpdateEngine::new(10);
        let prior = Prior::Dirichlet {
            alphas: vec![1.0, 1.0, 1.0],
        };
        let obs = Observation::Categorical {
            counts: vec![3, 2, 5],
        };
        let post = engine
            .update(prior, &obs)
            .expect("test: TD update should succeed");
        match post.updated {
            Prior::Dirichlet { alphas } => {
                assert!((alphas[0] - 4.0).abs() < 1e-10);
                assert!((alphas[1] - 3.0).abs() < 1e-10);
                assert!((alphas[2] - 6.0).abs() < 1e-10);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn dirichlet_categorical_dim_mismatch_error() {
        let mut engine = BayesianUpdateEngine::new(10);
        let prior = Prior::Dirichlet {
            alphas: vec![1.0, 1.0],
        };
        let obs = Observation::Categorical {
            counts: vec![1, 2, 3],
        };
        let result = engine.update(prior, &obs);
        assert!(matches!(result, Err(BayesError::InvalidParameters(_))));
    }

    #[test]
    fn dirichlet_categorical_log_marginal_finite() {
        let mut engine = BayesianUpdateEngine::new(10);
        let prior = Prior::Dirichlet {
            alphas: vec![2.0, 3.0, 5.0],
        };
        let obs = Observation::Categorical {
            counts: vec![10, 15, 25],
        };
        let post = engine
            .update(prior, &obs)
            .expect("test: TD update should succeed");
        assert!(post.log_marginal.is_finite());
    }

    #[test]
    fn dirichlet_categorical_zero_counts_no_change() {
        let mut engine = BayesianUpdateEngine::new(10);
        let prior = Prior::Dirichlet {
            alphas: vec![2.0, 3.0],
        };
        let obs = Observation::Categorical { counts: vec![0, 0] };
        let post = engine
            .update(prior, &obs)
            .expect("test: TD update should succeed");
        match post.updated {
            Prior::Dirichlet { alphas } => {
                assert!((alphas[0] - 2.0).abs() < 1e-10);
                assert!((alphas[1] - 3.0).abs() < 1e-10);
            }
            _ => panic!("wrong variant"),
        }
    }

    // ── Gamma–Poisson update ─────────────────────────────────────────────────

    #[test]
    fn gamma_poisson_update_basic() {
        let mut engine = BayesianUpdateEngine::new(10);
        let prior = Prior::Gamma {
            shape: 1.0,
            rate: 1.0,
        };
        let obs = Observation::Poisson {
            total_events: 5,
            total_time: 2.0,
        };
        let post = engine
            .update(prior, &obs)
            .expect("test: TD update should succeed");
        match post.updated {
            Prior::Gamma { shape, rate } => {
                assert!((shape - 6.0).abs() < 1e-10);
                assert!((rate - 3.0).abs() < 1e-10);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn gamma_poisson_log_marginal_finite() {
        let mut engine = BayesianUpdateEngine::new(10);
        let prior = Prior::Gamma {
            shape: 2.0,
            rate: 0.5,
        };
        let obs = Observation::Poisson {
            total_events: 10,
            total_time: 5.0,
        };
        let post = engine
            .update(prior, &obs)
            .expect("test: TD update should succeed");
        assert!(post.log_marginal.is_finite());
    }

    #[test]
    fn gamma_poisson_zero_time_error() {
        let mut engine = BayesianUpdateEngine::new(10);
        let prior = Prior::Gamma {
            shape: 1.0,
            rate: 1.0,
        };
        let obs = Observation::Poisson {
            total_events: 5,
            total_time: 0.0,
        };
        let result = engine.update(prior, &obs);
        assert!(matches!(result, Err(BayesError::InvalidParameters(_))));
    }

    // ── Mismatch errors ──────────────────────────────────────────────────────

    #[test]
    fn mismatch_beta_gaussian_obs() {
        let mut engine = BayesianUpdateEngine::new(10);
        let prior = Prior::Beta {
            alpha: 1.0,
            beta: 1.0,
        };
        let obs = Observation::Gaussian {
            sample_mean: 0.5,
            sample_variance: 1.0,
            n: 10,
        };
        let result = engine.update(prior, &obs);
        assert!(matches!(
            result,
            Err(BayesError::PriorObservationMismatch { .. })
        ));
    }

    #[test]
    fn mismatch_gaussian_bernoulli_obs() {
        let mut engine = BayesianUpdateEngine::new(10);
        let prior = Prior::Gaussian {
            mean: 0.0,
            variance: 1.0,
        };
        let obs = Observation::Bernoulli {
            successes: 3,
            trials: 5,
        };
        let result = engine.update(prior, &obs);
        assert!(matches!(
            result,
            Err(BayesError::PriorObservationMismatch { .. })
        ));
    }

    // ── Sequential update ────────────────────────────────────────────────────

    #[test]
    fn sequential_update_equivalent_to_batch() {
        // For Beta-Bernoulli, two sequential updates should equal one combined update
        let mut engine = BayesianUpdateEngine::new(64);
        let prior = Prior::Beta {
            alpha: 1.0,
            beta: 1.0,
        };
        let obs = vec![
            Observation::Bernoulli {
                successes: 3,
                trials: 5,
            },
            Observation::Bernoulli {
                successes: 2,
                trials: 4,
            },
        ];
        let seq_post = engine
            .sequential_update(prior.clone(), &obs)
            .expect("test: should succeed");

        // Batch equivalent: alpha' = 1 + 5, beta' = 1 + 4
        let mut engine2 = BayesianUpdateEngine::new(64);
        let batch_obs = Observation::Bernoulli {
            successes: 5,
            trials: 9,
        };
        let batch_post = engine2
            .update(prior, &batch_obs)
            .expect("test: TD update should succeed");

        match (&seq_post.updated, &batch_post.updated) {
            (
                Prior::Beta {
                    alpha: a1,
                    beta: b1,
                },
                Prior::Beta {
                    alpha: a2,
                    beta: b2,
                },
            ) => {
                assert!((a1 - a2).abs() < 1e-10);
                assert!((b1 - b2).abs() < 1e-10);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn sequential_update_empty_error() {
        let mut engine = BayesianUpdateEngine::new(10);
        let prior = Prior::Beta {
            alpha: 1.0,
            beta: 1.0,
        };
        let result = engine.sequential_update(prior, &[]);
        assert!(matches!(result, Err(BayesError::InvalidParameters(_))));
    }

    // ── Credible interval ────────────────────────────────────────────────────

    #[test]
    fn credible_interval_beta_bounds() {
        let post = Prior::Beta {
            alpha: 8.0,
            beta: 4.0,
        };
        let ci =
            BayesianUpdateEngine::credible_interval(&post, 0.95).expect("test: should succeed");
        assert!(ci.lower >= 0.0);
        assert!(ci.upper <= 1.0);
        assert!(ci.lower < ci.upper);
        assert!((ci.probability - 0.95).abs() < 1e-10);
    }

    #[test]
    fn credible_interval_gaussian_symmetric() {
        let post = Prior::Gaussian {
            mean: 5.0,
            variance: 1.0,
        };
        let ci =
            BayesianUpdateEngine::credible_interval(&post, 0.95).expect("test: should succeed");
        let center = (ci.lower + ci.upper) / 2.0;
        assert!((center - 5.0).abs() < 1e-8);
        // Half-width ≈ 1.96
        let hw = (ci.upper - ci.lower) / 2.0;
        assert!((hw - 1.96).abs() < 0.01);
    }

    #[test]
    fn credible_interval_invalid_probability() {
        let post = Prior::Gaussian {
            mean: 0.0,
            variance: 1.0,
        };
        assert!(BayesianUpdateEngine::credible_interval(&post, 0.0).is_err());
        assert!(BayesianUpdateEngine::credible_interval(&post, 1.0).is_err());
        assert!(BayesianUpdateEngine::credible_interval(&post, -0.1).is_err());
    }

    #[test]
    fn credible_interval_gamma() {
        let post = Prior::Gamma {
            shape: 9.0,
            rate: 3.0,
        };
        let ci =
            BayesianUpdateEngine::credible_interval(&post, 0.95).expect("test: should succeed");
        // mean = 3.0; lower should be positive
        assert!(ci.lower >= 0.0);
        assert!(ci.upper > ci.lower);
    }

    #[test]
    fn credible_interval_dirichlet() {
        let post = Prior::Dirichlet {
            alphas: vec![10.0, 2.0, 3.0],
        };
        let ci =
            BayesianUpdateEngine::credible_interval(&post, 0.90).expect("test: should succeed");
        assert!(ci.lower >= 0.0);
        assert!(ci.upper <= 1.0);
    }

    // ── MAP estimate ─────────────────────────────────────────────────────────

    #[test]
    fn map_beta_mode() {
        // Beta(3,3): mode = (3-1)/(3+3-2) = 2/4 = 0.5
        let p = Prior::Beta {
            alpha: 3.0,
            beta: 3.0,
        };
        let map = BayesianUpdateEngine::map_estimate(&p);
        assert!((map - 0.5).abs() < 1e-10);
    }

    #[test]
    fn map_beta_uniform_fallback() {
        // Beta(1,1) → alpha <= 1, use alpha/(alpha+beta) = 0.5
        let p = Prior::Beta {
            alpha: 1.0,
            beta: 1.0,
        };
        let map = BayesianUpdateEngine::map_estimate(&p);
        assert!((map - 0.5).abs() < 1e-10);
    }

    #[test]
    fn map_gaussian_is_mean() {
        let p = Prior::Gaussian {
            mean: 3.7,
            variance: 2.0,
        };
        assert!((BayesianUpdateEngine::map_estimate(&p) - 3.7).abs() < 1e-10);
    }

    #[test]
    fn map_gamma_mode() {
        // Gamma(5, 2): mode = (5-1)/2 = 2.0
        let p = Prior::Gamma {
            shape: 5.0,
            rate: 2.0,
        };
        assert!((BayesianUpdateEngine::map_estimate(&p) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn map_gamma_shape_one_gives_zero() {
        let p = Prior::Gamma {
            shape: 1.0,
            rate: 2.0,
        };
        assert!((BayesianUpdateEngine::map_estimate(&p) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn map_dirichlet_argmax_proportion() {
        let p = Prior::Dirichlet {
            alphas: vec![1.0, 5.0, 2.0],
        };
        // max alpha = 5.0, sum = 8.0 → map = 5/8
        let expected = 5.0 / 8.0;
        assert!((BayesianUpdateEngine::map_estimate(&p) - expected).abs() < 1e-10);
    }

    // ── KL divergence ────────────────────────────────────────────────────────

    #[test]
    fn kl_beta_self_is_zero() {
        let p = Prior::Beta {
            alpha: 3.0,
            beta: 5.0,
        };
        let kl = BayesianUpdateEngine::kl_divergence(&p, &p).expect("test: should succeed");
        assert!(kl.abs() < 1e-8);
    }

    #[test]
    fn kl_gaussian_self_is_zero() {
        let p = Prior::Gaussian {
            mean: 2.0,
            variance: 3.0,
        };
        let kl = BayesianUpdateEngine::kl_divergence(&p, &p).expect("test: should succeed");
        assert!(kl.abs() < 1e-10);
    }

    #[test]
    fn kl_gaussian_asymmetry() {
        let p = Prior::Gaussian {
            mean: 0.0,
            variance: 1.0,
        };
        let q = Prior::Gaussian {
            mean: 1.0,
            variance: 2.0,
        };
        let kl_pq = BayesianUpdateEngine::kl_divergence(&p, &q).expect("test: should succeed");
        let kl_qp = BayesianUpdateEngine::kl_divergence(&q, &p).expect("test: should succeed");
        // KL is asymmetric in general
        assert!((kl_pq - kl_qp).abs() > 1e-6);
        // Both should be non-negative
        assert!(kl_pq >= 0.0);
        assert!(kl_qp >= 0.0);
    }

    #[test]
    fn kl_known_gaussian_value() {
        // KL(N(0,1) ‖ N(1,1)) = 0.5 * (0 + 1 + 1 - 1) = 0.5
        let p = Prior::Gaussian {
            mean: 0.0,
            variance: 1.0,
        };
        let q = Prior::Gaussian {
            mean: 1.0,
            variance: 1.0,
        };
        let kl = BayesianUpdateEngine::kl_divergence(&p, &q).expect("test: should succeed");
        assert!((kl - 0.5).abs() < 1e-10);
    }

    #[test]
    fn kl_unsupported_pair_error() {
        let p = Prior::Beta {
            alpha: 1.0,
            beta: 1.0,
        };
        let q = Prior::Gamma {
            shape: 1.0,
            rate: 1.0,
        };
        let result = BayesianUpdateEngine::kl_divergence(&p, &q);
        assert!(matches!(result, Err(BayesError::UnsupportedOperation(_))));
    }

    // ── History management ───────────────────────────────────────────────────

    #[test]
    fn history_bounded_by_max() {
        let mut engine = BayesianUpdateEngine::new(3);
        for i in 0..5_u64 {
            let prior = Prior::Beta {
                alpha: 1.0,
                beta: 1.0,
            };
            let obs = Observation::Bernoulli {
                successes: i % 3,
                trials: 5,
            };
            engine
                .update(prior, &obs)
                .expect("test: TD update should succeed");
        }
        assert_eq!(engine.history().len(), 3);
    }

    #[test]
    fn history_clear() {
        let mut engine = BayesianUpdateEngine::new(10);
        let prior = Prior::Beta {
            alpha: 1.0,
            beta: 1.0,
        };
        let obs = Observation::Bernoulli {
            successes: 3,
            trials: 5,
        };
        engine
            .update(prior, &obs)
            .expect("test: TD update should succeed");
        assert!(!engine.history().is_empty());
        engine.clear_history();
        assert!(engine.history().is_empty());
    }

    #[test]
    fn history_zero_capacity_no_store() {
        let mut engine = BayesianUpdateEngine::new(0);
        let prior = Prior::Beta {
            alpha: 1.0,
            beta: 1.0,
        };
        let obs = Observation::Bernoulli {
            successes: 3,
            trials: 5,
        };
        engine
            .update(prior, &obs)
            .expect("test: TD update should succeed");
        assert!(engine.history().is_empty());
    }

    // ── Prior validation ─────────────────────────────────────────────────────

    #[test]
    fn invalid_beta_alpha_zero() {
        let mut engine = BayesianUpdateEngine::new(10);
        let prior = Prior::Beta {
            alpha: 0.0,
            beta: 1.0,
        };
        let obs = Observation::Bernoulli {
            successes: 1,
            trials: 2,
        };
        assert!(matches!(
            engine.update(prior, &obs),
            Err(BayesError::InvalidParameters(_))
        ));
    }

    #[test]
    fn invalid_gamma_rate_negative() {
        let mut engine = BayesianUpdateEngine::new(10);
        let prior = Prior::Gamma {
            shape: 1.0,
            rate: -1.0,
        };
        let obs = Observation::Poisson {
            total_events: 5,
            total_time: 1.0,
        };
        assert!(matches!(
            engine.update(prior, &obs),
            Err(BayesError::InvalidParameters(_))
        ));
    }

    // ── likelihood_type field ────────────────────────────────────────────────

    #[test]
    fn likelihood_type_labels() {
        let mut engine = BayesianUpdateEngine::new(10);

        let p1 = engine
            .update(
                Prior::Beta {
                    alpha: 1.0,
                    beta: 1.0,
                },
                &Observation::Bernoulli {
                    successes: 1,
                    trials: 2,
                },
            )
            .expect("test: should succeed");
        assert_eq!(p1.likelihood_type, "Bernoulli");

        let p2 = engine
            .update(
                Prior::Gaussian {
                    mean: 0.0,
                    variance: 1.0,
                },
                &Observation::Gaussian {
                    sample_mean: 1.0,
                    sample_variance: 1.0,
                    n: 5,
                },
            )
            .expect("test: should succeed");
        assert_eq!(p2.likelihood_type, "Gaussian");

        let p3 = engine
            .update(
                Prior::Dirichlet {
                    alphas: vec![1.0, 1.0],
                },
                &Observation::Categorical { counts: vec![3, 2] },
            )
            .expect("test: should succeed");
        assert_eq!(p3.likelihood_type, "Categorical");

        let p4 = engine
            .update(
                Prior::Gamma {
                    shape: 1.0,
                    rate: 1.0,
                },
                &Observation::Poisson {
                    total_events: 3,
                    total_time: 1.0,
                },
            )
            .expect("test: should succeed");
        assert_eq!(p4.likelihood_type, "Poisson");
    }
}
