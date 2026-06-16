//! PRNG helpers and sampling/density functions for the Probabilistic Program Engine.

use std::collections::HashMap;

use super::ppe_types::{PpePrior, ProbVar, VarId};

// ─── PRNG helpers ────────────────────────────────────────────────────────────

/// xorshift64 PRNG — no external crate dependencies.
#[inline(always)]
pub(super) fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// Uniform float in [0, 1).
#[inline(always)]
pub(super) fn uniform01(state: &mut u64) -> f64 {
    let bits = xorshift64(state);
    // Use upper 53 bits for precision.
    (bits >> 11) as f64 * (1.0_f64 / (1u64 << 53) as f64)
}

/// Box-Muller transform — produces a standard Normal sample from two
/// independent U(0,1) values.
#[inline(always)]
pub(super) fn box_muller(u1: f64, u2: f64) -> f64 {
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Draw a standard Normal sample.
#[inline]
pub(super) fn sample_standard_normal(state: &mut u64) -> f64 {
    let u1 = uniform01(state).max(1e-300);
    let u2 = uniform01(state);
    box_muller(u1, u2)
}

// ─── Prior sampling ──────────────────────────────────────────────────────────

/// Draw a single sample from a prior distribution.
pub(super) fn sample_prior(prior: &PpePrior, state: &mut u64) -> f64 {
    match prior {
        PpePrior::Normal { mean, std } => mean + std * sample_standard_normal(state),
        PpePrior::Uniform { low, high } => low + (high - low) * uniform01(state),
        PpePrior::Beta { alpha, beta } => sample_beta(*alpha, *beta, state),
        PpePrior::Exponential { rate } => {
            let u = uniform01(state).max(1e-300);
            -u.ln() / rate
        }
        PpePrior::Bernoulli { p } => {
            if uniform01(state) < *p {
                1.0
            } else {
                0.0
            }
        }
        PpePrior::Categorical { probs } => sample_categorical(probs, state),
    }
}

/// Approximate Beta(α, β) sampler using the ratio-of-gammas method (via
/// exponential / gamma approximation).  For moderate α, β (> 1) a
/// Johnk/Cheng approximation is used; for small shape parameters the naive
/// two-gamma ratio is used.
pub(super) fn sample_beta(alpha: f64, beta: f64, state: &mut u64) -> f64 {
    let x = sample_gamma(alpha, state);
    let y = sample_gamma(beta, state);
    if x + y < 1e-300 {
        0.5
    } else {
        (x / (x + y)).clamp(0.0, 1.0)
    }
}

/// Simple Gamma(shape, 1) sampler using Marsaglia-Tsang's method (pure Rust).
/// Falls back to exponential for shape == 1.
pub(super) fn sample_gamma(shape: f64, state: &mut u64) -> f64 {
    if shape <= 0.0 {
        return 0.0;
    }
    if (shape - 1.0).abs() < 1e-12 {
        // Gamma(1) = Exp(1)
        let u = uniform01(state).max(1e-300);
        return -u.ln();
    }
    if shape < 1.0 {
        // Gamma(shape) = Gamma(shape+1) * U^(1/shape)
        let g = sample_gamma(shape + 1.0, state);
        let u = uniform01(state).max(1e-300);
        return g * u.powf(1.0 / shape);
    }
    // Marsaglia-Tsang for shape >= 1.
    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let z = sample_standard_normal(state);
        let v_raw = 1.0 + c * z;
        if v_raw <= 0.0 {
            continue;
        }
        let v = v_raw * v_raw * v_raw;
        let u = uniform01(state).max(1e-300);
        if u < 1.0 - 0.0331 * (z * z) * (z * z) {
            return d * v;
        }
        if u.ln() < 0.5 * z * z + d * (1.0 - v + v.ln()) {
            return d * v;
        }
    }
}

/// Draw a categorical sample (returns index as f64).
pub(super) fn sample_categorical(probs: &[f64], state: &mut u64) -> f64 {
    if probs.is_empty() {
        return 0.0;
    }
    let total: f64 = probs.iter().sum();
    let u = uniform01(state) * total;
    let mut cumulative = 0.0;
    for (i, p) in probs.iter().enumerate() {
        cumulative += p;
        if u < cumulative {
            return i as f64;
        }
    }
    (probs.len() - 1) as f64
}

// ─── Log-density ─────────────────────────────────────────────────────────────

/// Compute log p(x | prior).
pub(super) fn log_density(prior: &PpePrior, x: f64) -> f64 {
    match prior {
        PpePrior::Normal { mean, std } => {
            if *std <= 0.0 {
                return f64::NEG_INFINITY;
            }
            let z = (x - mean) / std;
            -0.5 * z * z - std.ln() - 0.5 * (2.0 * std::f64::consts::PI).ln()
        }
        PpePrior::Uniform { low, high } => {
            if x >= *low && x <= *high && high > low {
                -((high - low).ln())
            } else {
                f64::NEG_INFINITY
            }
        }
        PpePrior::Beta { alpha, beta } => {
            if x <= 0.0 || x >= 1.0 {
                return f64::NEG_INFINITY;
            }
            (alpha - 1.0) * x.ln() + (beta - 1.0) * (1.0 - x).ln() - log_beta_fn(*alpha, *beta)
        }
        PpePrior::Exponential { rate } => {
            if x < 0.0 {
                f64::NEG_INFINITY
            } else {
                rate.ln() - rate * x
            }
        }
        PpePrior::Bernoulli { p } => {
            let p = p.clamp(1e-15, 1.0 - 1e-15);
            if (x - 1.0).abs() < 0.5 {
                p.ln()
            } else if x.abs() < 0.5 {
                (1.0 - p).ln()
            } else {
                f64::NEG_INFINITY
            }
        }
        PpePrior::Categorical { probs } => {
            let k = x.round() as usize;
            if k < probs.len() {
                let p = probs[k].max(1e-300);
                p.ln()
            } else {
                f64::NEG_INFINITY
            }
        }
    }
}

/// Natural log of the Beta function ln B(α, β) = lnΓ(α) + lnΓ(β) - lnΓ(α+β).
pub(super) fn log_beta_fn(alpha: f64, beta: f64) -> f64 {
    lgamma(alpha) + lgamma(beta) - lgamma(alpha + beta)
}

/// Stirling-series approximation of ln Γ(x) for x > 0.
pub(super) fn lgamma(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::INFINITY;
    }
    // Use Lanczos approximation (g=5, n=7).
    let c = [
        76.18009172947146_f64,
        -86.50532032941677,
        24.01409824083091,
        -1.231739572450155,
        1.208650973866179e-3,
        -5.395239384953e-6,
    ];
    let mut y = x;
    let mut tmp = x + 5.5;
    tmp -= (x + 0.5) * tmp.ln();
    let mut ser = 1.000000000190015_f64;
    for ci in &c {
        y += 1.0;
        ser += ci / y;
    }
    -tmp + (2.5066282746310005 * ser / x).ln()
}

// ─── Proposal helpers for MH ─────────────────────────────────────────────────

/// Width of the random-walk proposal for MH (one std dev of proposal noise).
pub(super) const MH_PROPOSAL_STD: f64 = 0.3;

/// Draw a MH proposal from the random-walk kernel.
pub(super) fn mh_propose(current: f64, prior: &PpePrior, state: &mut u64) -> f64 {
    match prior {
        PpePrior::Bernoulli { .. } => {
            // Flip with probability 0.5.
            if uniform01(state) < 0.5 {
                1.0 - current
            } else {
                current
            }
        }
        PpePrior::Categorical { probs } => {
            // Uniform random neighbour.
            let k = probs.len().max(1);
            (xorshift64(state) % k as u64) as f64
        }
        _ => current + MH_PROPOSAL_STD * sample_standard_normal(state),
    }
}

// ─── Log-likelihood of observations ──────────────────────────────────────────

/// Compute total log-likelihood of all observations given current variable
/// values.  `values` maps VarId → proposed value for unclamped variables.
pub(super) fn total_log_likelihood(
    variables: &HashMap<VarId, ProbVar>,
    observations: &HashMap<VarId, f64>,
    values: &HashMap<VarId, f64>,
) -> f64 {
    let mut ll = 0.0_f64;
    for (id, &obs) in observations {
        if variables.contains_key(id) {
            let proposed = values.get(id).copied().unwrap_or(obs);
            // Gaussian likelihood: obs ~ N(proposed, sigma=1).
            let diff = obs - proposed;
            ll += -0.5 * diff * diff;
        }
    }
    ll
}

// ─── Effective Sample Size ────────────────────────────────────────────────────

/// Estimate ESS using the initial monotone sequence estimator (truncated
/// autocorrelation).
pub(super) fn effective_sample_size(samples: &[f64]) -> f64 {
    let n = samples.len();
    if n < 4 {
        return n as f64;
    }
    let mean = samples.iter().sum::<f64>() / n as f64;
    let var = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    if var < 1e-300 {
        return n as f64;
    }
    // Estimate sum of autocorrelations up to first negative lag.
    let max_lag = (n / 2).min(200);
    let mut rho_sum = 1.0_f64;
    for lag in 1..max_lag {
        let mut ac = 0.0_f64;
        for i in 0..(n - lag) {
            ac += (samples[i] - mean) * (samples[i + lag] - mean);
        }
        ac /= (n as f64) * var;
        if ac <= 0.0 {
            break;
        }
        rho_sum += 2.0 * ac;
    }
    ((n as f64) / rho_sum).max(1.0)
}
