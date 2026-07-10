//! Gradient noise: training **regularization** and differentially-private SGD.
//!
//! This module contains two clearly separated tools that must not be confused:
//!
//! 1. [`GradientNoiseInjector`] — a *regularization* utility. It perturbs gradient
//!    tensors with configurable noise (Gaussian, Uniform, Laplacian, or a decaying
//!    Scheduled Gaussian) to smooth optimization and improve generalization. It is
//!    **not** a differential-privacy mechanism: it does not clip per-example
//!    gradient sensitivity, does not calibrate its scale to a privacy budget, and
//!    its optional `clip_value` clamps the *noise sample*, not the gradient. Its
//!    PRNG is seeded deterministically from `config.seed` purely for reproducible
//!    debugging.
//!
//! 2. [`DpSgdPrivatizer`] — a correct **DP-SGD** mechanism (Abadi et al., 2016). It
//!    performs the two operations that actually provide a privacy guarantee, in the
//!    only sound order: it **clips the gradient's global L2 norm first** (bounding
//!    the sensitivity to `C`), **then adds Gaussian noise** with standard deviation
//!    `σ = z · C`. Its noise source is seeded from OS entropy via a
//!    cryptographically-secure PRNG.
//!
//! # Regularization example
//!
//! ```
//! use ipfrs_tensorlogic::gradient_noise::{
//!     GradientNoiseConfig, GradientNoiseInjector, NoiseType,
//! };
//!
//! let config = GradientNoiseConfig {
//!     noise_type: NoiseType::Gaussian,
//!     initial_scale: 0.01,
//!     decay_rate: 0.0,
//!     clip_value: Some(0.05),
//!     seed: 42,
//! };
//!
//! let mut injector = GradientNoiseInjector::new(config);
//! let mut gradients = vec![1.0, 2.0, 3.0, 4.0];
//! injector.inject(&mut gradients);
//! // gradients now contain added noise
//! ```
//!
//! # DP-SGD example
//!
//! ```
//! use ipfrs_tensorlogic::gradient_noise::DpSgdPrivatizer;
//!
//! // Clip bound C = 1.0, noise multiplier z = 1.1 → σ = 1.1.
//! let mut dp = DpSgdPrivatizer::new(1.0, 1.1).expect("valid DP-SGD parameters");
//! let mut gradient = vec![3.0, 4.0]; // L2 norm 5.0, will be clipped to 1.0 first
//! dp.privatize(&mut gradient);
//! assert_eq!(dp.noise_std(), 1.1);
//! ```

use crate::gradient_clipper::{ClippingStrategy, GradientTensor, TensorGradientClipper};
use rand::distr::Open01;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use std::f64::consts::PI;
use thiserror::Error;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// The type of noise distribution to inject into gradients.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NoiseType {
    /// Standard Gaussian (normal) noise with mean 0 and configurable scale.
    Gaussian,
    /// Uniform noise in the range `[-scale, +scale]`.
    Uniform,
    /// Laplacian noise centred at 0 with configurable scale (heavier tails).
    Laplacian,
    /// Gaussian noise whose scale decays over training steps according to
    /// `scale = initial_scale / (1 + decay_rate * step)`.
    ScheduledGaussian,
}

/// Configuration for gradient noise injection.
#[derive(Debug, Clone)]
pub struct GradientNoiseConfig {
    /// The distribution family to sample noise from.
    pub noise_type: NoiseType,
    /// Initial noise magnitude (standard deviation for Gaussian, half-width
    /// for Uniform, scale parameter for Laplacian).
    pub initial_scale: f64,
    /// Decay rate applied to `ScheduledGaussian` noise. Ignored for other
    /// noise types.
    pub decay_rate: f64,
    /// If set, clamps each noise sample to `[-clip_value, +clip_value]`.
    ///
    /// Note: this is a *regularization* knob that bounds the magnitude of the
    /// injected noise. It is **not** DP-SGD gradient clipping (which bounds the
    /// gradient's L2 norm before noise is added); see [`DpSgdPrivatizer`] for that.
    pub clip_value: Option<f64>,
    /// Seed for the internal PRNG, enabling reproducible debugging runs. Any
    /// `u64` value is valid (including `0`).
    pub seed: u64,
}

/// Aggregate statistics about noise injections performed by an injector.
#[derive(Debug, Clone, Default)]
pub struct NoiseStats {
    /// Number of times `inject` has been called.
    pub total_injections: u64,
    /// Cumulative number of gradient elements that received noise.
    pub total_elements: u64,
    /// Running average of absolute noise magnitude across all elements.
    pub avg_noise_magnitude: f64,
    /// Largest absolute noise value ever applied.
    pub max_noise_applied: f64,
    /// Current noise scale (accounts for decay in `ScheduledGaussian`).
    pub current_scale: f64,
}

/// A batch of noise samples together with summary statistics.
#[derive(Debug, Clone)]
pub struct NoiseSample {
    /// The sampled noise values.
    pub values: Vec<f64>,
    /// Arithmetic mean of `values`.
    pub mean: f64,
    /// Sample standard deviation of `values`.
    pub std_dev: f64,
    /// Training step at which the sample was drawn.
    pub step: u64,
}

// ---------------------------------------------------------------------------
// GradientNoiseInjector
// ---------------------------------------------------------------------------

/// Injects configurable noise into gradient arrays for training
/// **regularization**.
///
/// This is deliberately *not* a differential-privacy mechanism. It smooths the
/// optimization landscape by perturbing gradients, but it does **not** bound
/// per-example gradient sensitivity and its scale is not tied to any privacy
/// budget. For a mechanism with an actual (ε, δ) guarantee, use
/// [`DpSgdPrivatizer`].
pub struct GradientNoiseInjector {
    config: GradientNoiseConfig,
    rng: StdRng,
    step: u64,
    stats: NoiseStats,
}

impl GradientNoiseInjector {
    /// Create a new injector from the given configuration.
    ///
    /// The internal PRNG is seeded deterministically from `config.seed`, giving
    /// reproducible noise sequences — a legitimate debugging convenience for a
    /// regularizer. `StdRng` accepts any `u64` seed (including `0`), so no
    /// non-zero fallback is required.
    pub fn new(config: GradientNoiseConfig) -> Self {
        let current_scale = config.initial_scale;
        let rng = StdRng::seed_from_u64(config.seed);
        Self {
            config,
            rng,
            step: 0,
            stats: NoiseStats {
                current_scale,
                ..NoiseStats::default()
            },
        }
    }

    // -- PRNG helpers -------------------------------------------------------

    /// Uniform `f64` in the **half-open** interval `[0, 1)` (`StandardUniform`).
    fn next_unit(&mut self) -> f64 {
        self.rng.random::<f64>()
    }

    /// Uniform `f64` in the **open** interval `(0, 1)` ([`rand::distr::Open01`]).
    ///
    /// Never returns `0.0`, so it is safe to feed directly to `ln` — the
    /// `ln(0) = -inf` hazard cannot occur and no rejection loop or epsilon guard
    /// is needed.
    fn next_unit_open(&mut self) -> f64 {
        self.rng.sample::<f64, _>(Open01)
    }

    /// Sample from a standard Gaussian (mean 0, std 1) using the Box-Muller
    /// transform.
    pub fn next_gaussian(&mut self) -> f64 {
        // `u1` is drawn from the open interval (0, 1) so that `ln(u1)` is always
        // finite; `u2` only feeds a cosine, so `[0, 1)` is fine there.
        let u1 = self.next_unit_open();
        let u2 = self.next_unit();
        let r = (-2.0 * u1.ln()).sqrt();
        r * (2.0 * PI * u2).cos()
    }

    /// Sample from a uniform distribution on `[low, high)`.
    pub fn next_uniform(&mut self, low: f64, high: f64) -> f64 {
        let u = self.next_unit();
        low + u * (high - low)
    }

    /// Sample from a Laplacian distribution with location 0 and the given
    /// `scale`, using the inverse-CDF method.
    pub fn next_laplacian(&mut self, scale: f64) -> f64 {
        // Drawing `u` from the open interval keeps `1 - 2|u - 0.5| > 0`, so the
        // `ln` argument is strictly positive and no clamping is required.
        let u = self.next_unit_open() - 0.5;
        let abs_u = u.abs();
        -scale * (1.0 - 2.0 * abs_u).ln() * u.signum()
    }

    /// Clip `value` to `[-clip, +clip]` if a clip bound is configured,
    /// otherwise return `value` unchanged.
    ///
    /// This clamps the **noise sample** as a regularization safeguard. It is
    /// **not** DP-SGD gradient clipping: DP-SGD clips the *gradient's* L2 norm
    /// *before* noise is added (see [`DpSgdPrivatizer`]); clamping the noise
    /// afterwards provides no differential-privacy guarantee whatsoever.
    pub fn clip_noise(&self, value: f64) -> f64 {
        match self.config.clip_value {
            Some(clip) => value.clamp(-clip, clip),
            None => value,
        }
    }

    // -- Public API ---------------------------------------------------------

    /// Compute the current effective noise scale, accounting for decay when
    /// using `ScheduledGaussian`.
    pub fn current_scale(&self) -> f64 {
        match self.config.noise_type {
            NoiseType::ScheduledGaussian => {
                self.config.initial_scale / (1.0 + self.config.decay_rate * self.step as f64)
            }
            _ => self.config.initial_scale,
        }
    }

    /// Sample a single noise value according to the configured distribution
    /// and scale, then clip it.
    fn sample_one(&mut self) -> f64 {
        let scale = self.current_scale();
        let raw = match self.config.noise_type {
            NoiseType::Gaussian => self.next_gaussian() * scale,
            NoiseType::Uniform => self.next_uniform(-scale, scale),
            NoiseType::Laplacian => self.next_laplacian(scale),
            NoiseType::ScheduledGaussian => self.next_gaussian() * scale,
        };
        self.clip_noise(raw)
    }

    /// Inject noise into `gradients` in-place.
    ///
    /// Each element of the slice receives an independent noise sample drawn
    /// from the configured distribution.  Statistics are updated accordingly.
    pub fn inject(&mut self, gradients: &mut [f64]) {
        let n = gradients.len() as u64;
        let mut sum_abs: f64 = 0.0;
        let mut local_max: f64 = 0.0;

        for g in gradients.iter_mut() {
            let noise = self.sample_one();
            *g += noise;
            let abs_noise = noise.abs();
            sum_abs += abs_noise;
            if abs_noise > local_max {
                local_max = abs_noise;
            }
        }

        // Update running statistics.
        let prev_total = self.stats.total_elements;
        self.stats.total_injections += 1;
        self.stats.total_elements += n;
        if self.stats.max_noise_applied < local_max {
            self.stats.max_noise_applied = local_max;
        }
        // Incremental average update.
        if n > 0 {
            let new_avg = sum_abs / n as f64;
            let total = prev_total + n;
            self.stats.avg_noise_magnitude = (self.stats.avg_noise_magnitude * prev_total as f64
                + new_avg * n as f64)
                / total as f64;
        }
        self.stats.current_scale = self.current_scale();
    }

    /// Generate `count` noise samples without applying them to any gradient
    /// array.  Returns a [`NoiseSample`] with summary statistics.
    pub fn sample_noise(&mut self, count: usize) -> NoiseSample {
        let mut values = Vec::with_capacity(count);
        for _ in 0..count {
            values.push(self.sample_one());
        }

        let (mean, std_dev) = compute_mean_std(&values);

        NoiseSample {
            values,
            mean,
            std_dev,
            step: self.step,
        }
    }

    /// Advance the training step counter by one.  This affects the noise
    /// scale when using `ScheduledGaussian`.
    pub fn step(&mut self) {
        self.step += 1;
        self.stats.current_scale = self.current_scale();
    }

    /// Reset the step counter, statistics, and re-seed the PRNG from
    /// `config.seed`, restoring the reproducible noise sequence from the start.
    pub fn reset(&mut self) {
        self.step = 0;
        self.rng = StdRng::seed_from_u64(self.config.seed);
        self.stats = NoiseStats {
            current_scale: self.config.initial_scale,
            ..NoiseStats::default()
        };
    }

    /// Read-only access to the accumulated statistics.
    pub fn stats(&self) -> &NoiseStats {
        &self.stats
    }

    /// Override the current noise scale (applies to all non-Scheduled types;
    /// for `ScheduledGaussian` this sets the *initial* scale used in decay).
    pub fn set_scale(&mut self, scale: f64) {
        self.config.initial_scale = scale;
        self.stats.current_scale = self.current_scale();
    }
}

// ---------------------------------------------------------------------------
// DP-SGD (Abadi et al., 2016)
// ---------------------------------------------------------------------------

/// Errors returned when constructing a [`DpSgdPrivatizer`].
#[derive(Debug, Clone, PartialEq, Error)]
pub enum DpSgdError {
    /// The clipping bound `C` was not strictly positive.
    ///
    /// `C` bounds the L2 sensitivity of a single example's gradient, so it must
    /// be `> 0` for the noise scale `σ = z · C` to be well defined. (`NaN` is
    /// also rejected here.)
    #[error("clip_bound must be strictly positive, got {0}")]
    InvalidClipBound(f64),

    /// The noise multiplier `z` was negative (or `NaN`).
    ///
    /// `z` may be `0.0` (clipping only, no noise — useful for tests), but never
    /// negative, since it scales a standard deviation.
    #[error("noise_multiplier must be non-negative, got {0}")]
    InvalidNoiseMultiplier(f64),
}

/// Differentially-private SGD gradient privatizer (Abadi et al., 2016).
///
/// Applies the Gaussian mechanism for DP-SGD in the **only** sound order:
///
/// 1. **Clip first.** The gradient's global L2 norm is clipped to `clip_bound`
///    (`C`), bounding the L2 sensitivity of the contribution to `C`.
/// 2. **Then add noise.** Independent Gaussian noise with standard deviation
///    `σ = noise_multiplier · clip_bound = z · C` is added to every coordinate.
///
/// # Why the order and the derived σ matter
///
/// Adding noise before clipping — or clipping the *noise* rather than the
/// *gradient* — destroys the guarantee, because the sensitivity would no longer
/// be bounded by `C`. Deriving `σ` **from** `C` (rather than accepting an
/// independent `σ`) means a caller **cannot understate the sensitivity**: the
/// same `C` that bounds sensitivity also sets the noise magnitude, so the
/// signal-to-noise ratio is honest by construction.
///
/// # Per-example vs. aggregate contract
///
/// The `(ε, δ)` guarantee of DP-SGD is defined **per example**: clip each
/// individual example's gradient to L2 norm `C`, *sum* the clipped per-example
/// gradients, then add noise once with `σ = z · C`. The L2 sensitivity of that
/// sum with respect to adding or removing one example is then exactly `C`. If
/// instead you pass an already-aggregated (e.g. mean) gradient,
/// [`privatize`](Self::privatize) still enforces the norm bound `C` on the slice
/// it is given, but only the caller knows the aggregation semantics and is
/// responsible for making the effective sensitivity match `C` (for a mean over
/// `B` examples, scale `C` or the noise by `1 / B`).
///
/// # Relation to Rényi DP accounting
///
/// One Gaussian step of this mechanism satisfies `(α, α / (2 z²))`-Rényi DP for
/// every order `α > 1` (Mironov, 2017). To account a sequence of steps, feed
/// that per-step RDP into [`crate::privacy_budget::RenyiAccountant`]. This type
/// deliberately does not embed a budget, so accounting stays explicit and
/// composable.
///
/// # Noise source
///
/// Seeded from OS entropy via a cryptographically-secure `StdRng` in
/// [`new`](Self::new); a predictable stream would let an adversary subtract the
/// noise. Use [`with_deterministic_seed_for_testing`](Self::with_deterministic_seed_for_testing)
/// only for reproducible tests.
pub struct DpSgdPrivatizer {
    clip_bound: f64,
    noise_multiplier: f64,
    rng: StdRng,
}

impl DpSgdPrivatizer {
    /// Construct a privatizer, seeding the noise source from OS entropy.
    ///
    /// Rejects `clip_bound <= 0.0` (or `NaN`) and `noise_multiplier < 0.0`
    /// (or `NaN`).
    ///
    /// # Panics
    ///
    /// Panics only if the OS entropy source fails (see [`rand::make_rng`]).
    pub fn new(clip_bound: f64, noise_multiplier: f64) -> Result<Self, DpSgdError> {
        Self::validate(clip_bound, noise_multiplier)?;
        Ok(Self {
            clip_bound,
            noise_multiplier,
            rng: rand::make_rng::<StdRng>(),
        })
    }

    /// Construct a privatizer with a caller-chosen, reproducible noise stream.
    ///
    /// **TEST ONLY.** A fixed seed makes the noise predictable; never use this
    /// for a real privacy-sensitive deployment.
    pub fn with_deterministic_seed_for_testing(
        clip_bound: f64,
        noise_multiplier: f64,
        seed: u64,
    ) -> Result<Self, DpSgdError> {
        Self::validate(clip_bound, noise_multiplier)?;
        Ok(Self {
            clip_bound,
            noise_multiplier,
            rng: StdRng::seed_from_u64(seed),
        })
    }

    /// Validate the DP-SGD parameters shared by both constructors.
    ///
    /// The explicit `is_nan()` checks reject `NaN` alongside the out-of-range
    /// values (a bare `x <= 0.0` would silently accept `NaN`, since every
    /// comparison with `NaN` is false).
    fn validate(clip_bound: f64, noise_multiplier: f64) -> Result<(), DpSgdError> {
        if clip_bound <= 0.0 || clip_bound.is_nan() {
            return Err(DpSgdError::InvalidClipBound(clip_bound));
        }
        if noise_multiplier < 0.0 || noise_multiplier.is_nan() {
            return Err(DpSgdError::InvalidNoiseMultiplier(noise_multiplier));
        }
        Ok(())
    }

    /// The Gaussian noise standard deviation `σ = z · C` used by
    /// [`privatize`](Self::privatize).
    pub fn noise_std(&self) -> f64 {
        self.noise_multiplier * self.clip_bound
    }

    /// The L2 clipping bound `C`.
    pub fn clip_bound(&self) -> f64 {
        self.clip_bound
    }

    /// The noise multiplier `z`.
    pub fn noise_multiplier(&self) -> f64 {
        self.noise_multiplier
    }

    /// Draw one standard normal (mean 0, std 1) via a `ln(0)`-guarded Box-Muller
    /// transform.
    ///
    /// `u1` is drawn from the **open** interval `(0, 1)` (via [`Open01`]) so that
    /// `ln(u1)` is always finite — the `ln(0) = -inf` hazard cannot occur.
    fn standard_normal(&mut self) -> f64 {
        let u1 = self.rng.sample::<f64, _>(Open01);
        let u2 = self.rng.sample::<f64, _>(Open01);
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }

    /// Privatize a gradient in place: **clip its global L2 norm to `clip_bound`,
    /// then add Gaussian noise** with std `σ = noise_multiplier · clip_bound`.
    ///
    /// The clipping step reuses [`TensorGradientClipper`] with
    /// [`ClippingStrategy::GlobalNorm`] — the same L2-rescaling routine used
    /// elsewhere in the crate — so the norm bound is enforced identically and is
    /// not re-implemented here.
    ///
    /// After this call the pre-noise gradient is guaranteed to have global L2
    /// norm `≤ clip_bound`; the added Gaussian noise then provides the privacy
    /// guarantee. An empty slice and a `noise_multiplier` of `0.0` are both
    /// handled gracefully (the latter clips without adding noise).
    pub fn privatize(&mut self, gradient: &mut [f64]) {
        // (a) Clip the global L2 norm to `clip_bound` by reusing the crate's
        //     GlobalNorm clipper. This scales the whole vector by
        //     `clip_bound / norm` when `norm > clip_bound`, and leaves it
        //     untouched otherwise.
        let mut clipper = TensorGradientClipper::new(ClippingStrategy::GlobalNorm {
            max_norm: self.clip_bound,
        });
        let mut tensors = vec![GradientTensor {
            tensor_id: 0,
            values: gradient.to_vec(),
        }];
        clipper.clip(&mut tensors);
        gradient.copy_from_slice(&tensors[0].values);

        // (b) Add Gaussian noise with std σ = z · C to every coordinate.
        let sigma = self.noise_std();
        for g in gradient.iter_mut() {
            *g += self.standard_normal() * sigma;
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute arithmetic mean and sample standard deviation for a slice.
fn compute_mean_std(values: &[f64]) -> (f64, f64) {
    if values.is_empty() {
        return (0.0, 0.0);
    }
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    if values.len() == 1 {
        return (mean, 0.0);
    }
    let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
    (mean, var.sqrt())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn gaussian_config(seed: u64) -> GradientNoiseConfig {
        GradientNoiseConfig {
            noise_type: NoiseType::Gaussian,
            initial_scale: 0.1,
            decay_rate: 0.0,
            clip_value: None,
            seed,
        }
    }

    fn uniform_config(seed: u64) -> GradientNoiseConfig {
        GradientNoiseConfig {
            noise_type: NoiseType::Uniform,
            initial_scale: 1.0,
            decay_rate: 0.0,
            clip_value: None,
            seed,
        }
    }

    fn laplacian_config(seed: u64) -> GradientNoiseConfig {
        GradientNoiseConfig {
            noise_type: NoiseType::Laplacian,
            initial_scale: 0.5,
            decay_rate: 0.0,
            clip_value: None,
            seed,
        }
    }

    fn scheduled_config(seed: u64) -> GradientNoiseConfig {
        GradientNoiseConfig {
            noise_type: NoiseType::ScheduledGaussian,
            initial_scale: 1.0,
            decay_rate: 0.1,
            clip_value: None,
            seed,
        }
    }

    // -- Gaussian distribution tests ----------------------------------------

    #[test]
    fn gaussian_noise_has_zero_mean_approximately() {
        let mut inj = GradientNoiseInjector::new(gaussian_config(123));
        let sample = inj.sample_noise(10_000);
        // With 10k samples and scale 0.1, mean should be near 0.
        assert!(
            sample.mean.abs() < 0.01,
            "mean = {} is too far from 0",
            sample.mean
        );
    }

    #[test]
    fn gaussian_noise_std_approximates_scale() {
        let mut inj = GradientNoiseInjector::new(gaussian_config(456));
        let sample = inj.sample_noise(10_000);
        // std should be close to the configured scale (0.1).
        assert!(
            (sample.std_dev - 0.1).abs() < 0.02,
            "std_dev = {} not close to 0.1",
            sample.std_dev
        );
    }

    #[test]
    fn gaussian_noise_values_are_finite() {
        let mut inj = GradientNoiseInjector::new(gaussian_config(789));
        let sample = inj.sample_noise(1000);
        for v in &sample.values {
            assert!(v.is_finite(), "non-finite value: {}", v);
        }
    }

    // -- Uniform distribution tests -----------------------------------------

    #[test]
    fn uniform_noise_within_bounds() {
        let mut inj = GradientNoiseInjector::new(uniform_config(111));
        let sample = inj.sample_noise(5000);
        for v in &sample.values {
            assert!(*v >= -1.0 && *v < 1.0, "value {} out of [-1, 1) range", v);
        }
    }

    #[test]
    fn uniform_noise_mean_near_zero() {
        let mut inj = GradientNoiseInjector::new(uniform_config(222));
        let sample = inj.sample_noise(10_000);
        assert!(
            sample.mean.abs() < 0.05,
            "mean = {} is too far from 0",
            sample.mean
        );
    }

    #[test]
    fn uniform_noise_spreads_across_range() {
        let mut inj = GradientNoiseInjector::new(uniform_config(333));
        let sample = inj.sample_noise(5000);
        let min = sample.values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = sample
            .values
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(min < -0.8, "min {} not spread enough", min);
        assert!(max > 0.8, "max {} not spread enough", max);
    }

    // -- Laplacian distribution tests ---------------------------------------

    #[test]
    fn laplacian_noise_mean_near_zero() {
        let mut inj = GradientNoiseInjector::new(laplacian_config(444));
        let sample = inj.sample_noise(10_000);
        assert!(
            sample.mean.abs() < 0.05,
            "mean = {} too far from 0",
            sample.mean
        );
    }

    #[test]
    fn laplacian_noise_values_are_finite() {
        let mut inj = GradientNoiseInjector::new(laplacian_config(555));
        let sample = inj.sample_noise(1000);
        for v in &sample.values {
            assert!(v.is_finite(), "non-finite Laplacian value: {}", v);
        }
    }

    #[test]
    fn laplacian_has_heavier_tails_than_gaussian() {
        // Compare the 99th percentile of Laplacian vs Gaussian at same scale.
        let mut g_inj = GradientNoiseInjector::new(GradientNoiseConfig {
            noise_type: NoiseType::Gaussian,
            initial_scale: 0.5,
            decay_rate: 0.0,
            clip_value: None,
            seed: 666,
        });
        let mut l_inj = GradientNoiseInjector::new(laplacian_config(666));
        let mut g_vals: Vec<f64> = g_inj.sample_noise(10_000).values;
        let mut l_vals: Vec<f64> = l_inj.sample_noise(10_000).values;
        g_vals.sort_by(|a, b| {
            a.abs()
                .partial_cmp(&b.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        l_vals.sort_by(|a, b| {
            a.abs()
                .partial_cmp(&b.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let g99 = g_vals[9900].abs();
        let l99 = l_vals[9900].abs();
        assert!(
            l99 > g99,
            "Laplacian 99th pct {} should exceed Gaussian {}",
            l99,
            g99
        );
    }

    // -- Scheduled Gaussian decay tests -------------------------------------

    #[test]
    fn scheduled_gaussian_decays_over_steps() {
        let mut inj = GradientNoiseInjector::new(scheduled_config(777));
        let scale0 = inj.current_scale();
        assert!((scale0 - 1.0).abs() < f64::EPSILON);

        inj.step();
        let scale1 = inj.current_scale();
        assert!(scale1 < scale0, "scale should decay");

        for _ in 0..10 {
            inj.step();
        }
        let scale11 = inj.current_scale();
        assert!(
            scale11 < scale1,
            "scale should keep decaying: {} vs {}",
            scale11,
            scale1
        );
    }

    #[test]
    fn scheduled_gaussian_scale_formula_correct() {
        let config = scheduled_config(888);
        let mut inj = GradientNoiseInjector::new(config);
        for _ in 0..5 {
            inj.step();
        }
        let expected = 1.0 / (1.0 + 0.1 * 5.0);
        let actual = inj.current_scale();
        assert!(
            (actual - expected).abs() < 1e-12,
            "expected {}, got {}",
            expected,
            actual
        );
    }

    #[test]
    fn scheduled_gaussian_noise_magnitude_decreases() {
        let mut inj = GradientNoiseInjector::new(scheduled_config(999));
        let s0 = inj.sample_noise(5000);
        for _ in 0..50 {
            inj.step();
        }
        let s50 = inj.sample_noise(5000);
        assert!(
            s50.std_dev < s0.std_dev,
            "later std {} should be less than initial {}",
            s50.std_dev,
            s0.std_dev
        );
    }

    // -- Clipping tests -----------------------------------------------------

    #[test]
    fn clipping_limits_noise_magnitude() {
        let config = GradientNoiseConfig {
            noise_type: NoiseType::Gaussian,
            initial_scale: 10.0,
            decay_rate: 0.0,
            clip_value: Some(0.5),
            seed: 1010,
        };
        let mut inj = GradientNoiseInjector::new(config);
        let sample = inj.sample_noise(5000);
        for v in &sample.values {
            assert!(
                v.abs() <= 0.5 + f64::EPSILON,
                "clipped value {} exceeds 0.5",
                v
            );
        }
    }

    #[test]
    fn clip_noise_returns_unchanged_without_config() {
        let config = GradientNoiseConfig {
            noise_type: NoiseType::Gaussian,
            initial_scale: 1.0,
            decay_rate: 0.0,
            clip_value: None,
            seed: 1111,
        };
        let inj = GradientNoiseInjector::new(config);
        assert!((inj.clip_noise(999.0) - 999.0).abs() < f64::EPSILON);
    }

    #[test]
    fn clip_noise_clamps_symmetric() {
        let config = GradientNoiseConfig {
            noise_type: NoiseType::Gaussian,
            initial_scale: 1.0,
            decay_rate: 0.0,
            clip_value: Some(2.0),
            seed: 1212,
        };
        let inj = GradientNoiseInjector::new(config);
        assert!((inj.clip_noise(5.0) - 2.0).abs() < f64::EPSILON);
        assert!((inj.clip_noise(-5.0) - (-2.0)).abs() < f64::EPSILON);
        assert!((inj.clip_noise(1.5) - 1.5).abs() < f64::EPSILON);
    }

    // -- Step advancement ---------------------------------------------------

    #[test]
    fn step_increments_counter() {
        let mut inj = GradientNoiseInjector::new(gaussian_config(1313));
        assert_eq!(inj.step, 0);
        inj.step();
        assert_eq!(inj.step, 1);
        inj.step();
        assert_eq!(inj.step, 2);
    }

    // -- Seed reproducibility -----------------------------------------------

    #[test]
    fn same_seed_produces_same_sequence() {
        let mut a = GradientNoiseInjector::new(gaussian_config(4242));
        let mut b = GradientNoiseInjector::new(gaussian_config(4242));
        let sa = a.sample_noise(100);
        let sb = b.sample_noise(100);
        assert_eq!(sa.values, sb.values);
    }

    #[test]
    fn different_seeds_produce_different_sequences() {
        let mut a = GradientNoiseInjector::new(gaussian_config(1));
        let mut b = GradientNoiseInjector::new(gaussian_config(2));
        let sa = a.sample_noise(100);
        let sb = b.sample_noise(100);
        // Extremely unlikely to be identical.
        assert_ne!(sa.values, sb.values);
    }

    // -- Stats tracking -----------------------------------------------------

    #[test]
    fn stats_updated_after_inject() {
        let mut inj = GradientNoiseInjector::new(gaussian_config(1414));
        let mut grads = vec![0.0; 50];
        inj.inject(&mut grads);
        let s = inj.stats();
        assert_eq!(s.total_injections, 1);
        assert_eq!(s.total_elements, 50);
        assert!(s.avg_noise_magnitude > 0.0);
    }

    #[test]
    fn stats_accumulate_across_injections() {
        let mut inj = GradientNoiseInjector::new(gaussian_config(1515));
        let mut g1 = vec![0.0; 100];
        let mut g2 = vec![0.0; 200];
        inj.inject(&mut g1);
        inj.inject(&mut g2);
        let s = inj.stats();
        assert_eq!(s.total_injections, 2);
        assert_eq!(s.total_elements, 300);
    }

    #[test]
    fn max_noise_tracked() {
        let config = GradientNoiseConfig {
            noise_type: NoiseType::Gaussian,
            initial_scale: 5.0,
            decay_rate: 0.0,
            clip_value: None,
            seed: 1616,
        };
        let mut inj = GradientNoiseInjector::new(config);
        let mut grads = vec![0.0; 1000];
        inj.inject(&mut grads);
        assert!(inj.stats().max_noise_applied > 0.0);
    }

    // -- inject modifies gradients ------------------------------------------

    #[test]
    fn inject_modifies_gradients() {
        let mut inj = GradientNoiseInjector::new(gaussian_config(1717));
        let original = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut grads = original.clone();
        inj.inject(&mut grads);
        assert_ne!(grads, original, "gradients should be modified by noise");
    }

    #[test]
    fn inject_preserves_length() {
        let mut inj = GradientNoiseInjector::new(gaussian_config(1818));
        let mut grads = vec![0.5; 37];
        inj.inject(&mut grads);
        assert_eq!(grads.len(), 37);
    }

    // -- Large gradient arrays ----------------------------------------------

    #[test]
    fn inject_large_array() {
        let mut inj = GradientNoiseInjector::new(gaussian_config(1919));
        let mut grads = vec![0.0; 100_000];
        inj.inject(&mut grads);
        let nonzero = grads.iter().filter(|v| v.abs() > f64::EPSILON).count();
        assert!(nonzero > 99_000, "almost all elements should receive noise");
    }

    // -- Zero-scale produces no noise ---------------------------------------

    #[test]
    fn zero_scale_produces_no_noise() {
        let config = GradientNoiseConfig {
            noise_type: NoiseType::Gaussian,
            initial_scale: 0.0,
            decay_rate: 0.0,
            clip_value: None,
            seed: 2020,
        };
        let mut inj = GradientNoiseInjector::new(config);
        let mut grads = vec![1.0, 2.0, 3.0];
        inj.inject(&mut grads);
        // With scale 0, noise samples are 0 * gaussian = 0.
        assert!((grads[0] - 1.0).abs() < f64::EPSILON);
        assert!((grads[1] - 2.0).abs() < f64::EPSILON);
        assert!((grads[2] - 3.0).abs() < f64::EPSILON);
    }

    // -- Reset behaviour ----------------------------------------------------

    #[test]
    fn reset_clears_stats_and_step() {
        let mut inj = GradientNoiseInjector::new(gaussian_config(2121));
        let mut grads = vec![0.0; 10];
        inj.inject(&mut grads);
        inj.step();
        inj.step();
        inj.reset();
        assert_eq!(inj.step, 0);
        assert_eq!(inj.stats().total_injections, 0);
        assert_eq!(inj.stats().total_elements, 0);
    }

    #[test]
    fn reset_reproduces_sequence() {
        let config = gaussian_config(2222);
        let mut inj = GradientNoiseInjector::new(config);
        let first = inj.sample_noise(50);
        inj.reset();
        let second = inj.sample_noise(50);
        assert_eq!(first.values, second.values);
    }

    // -- set_scale ----------------------------------------------------------

    #[test]
    fn set_scale_changes_output_magnitude() {
        let mut inj = GradientNoiseInjector::new(gaussian_config(2323));
        inj.set_scale(10.0);
        let sample = inj.sample_noise(5000);
        // std should now be close to 10.
        assert!(
            sample.std_dev > 5.0,
            "std_dev {} should reflect new scale 10",
            sample.std_dev
        );
    }

    // -- NoiseSample statistics ---------------------------------------------

    #[test]
    fn sample_noise_reports_correct_step() {
        let mut inj = GradientNoiseInjector::new(gaussian_config(2424));
        inj.step();
        inj.step();
        inj.step();
        let sample = inj.sample_noise(10);
        assert_eq!(sample.step, 3);
    }

    #[test]
    fn sample_noise_empty_returns_defaults() {
        let mut inj = GradientNoiseInjector::new(gaussian_config(2525));
        let sample = inj.sample_noise(0);
        assert!(sample.values.is_empty());
        assert!((sample.mean).abs() < f64::EPSILON);
        assert!((sample.std_dev).abs() < f64::EPSILON);
    }

    // -- Zero seed is valid (no fallback needed) ----------------------------

    #[test]
    fn zero_seed_is_valid_and_produces_finite_values() {
        // `StdRng` accepts seed 0 natively — there is no non-zero fallback.
        let config = GradientNoiseConfig {
            noise_type: NoiseType::Gaussian,
            initial_scale: 0.1,
            decay_rate: 0.0,
            clip_value: None,
            seed: 0,
        };
        let mut inj = GradientNoiseInjector::new(config);
        let sample = inj.sample_noise(10);
        assert_eq!(sample.values.len(), 10);
        for v in &sample.values {
            assert!(v.is_finite(), "seed 0 must yield finite values, got {v}");
        }
    }

    // -- current_scale for non-scheduled types ------------------------------

    #[test]
    fn current_scale_constant_for_non_scheduled() {
        let mut inj = GradientNoiseInjector::new(gaussian_config(2626));
        let s0 = inj.current_scale();
        for _ in 0..100 {
            inj.step();
        }
        let s100 = inj.current_scale();
        assert!(
            (s0 - s100).abs() < f64::EPSILON,
            "non-scheduled scale should not change"
        );
    }

    // -- inject empty slice -------------------------------------------------

    #[test]
    fn inject_empty_slice_no_panic() {
        let mut inj = GradientNoiseInjector::new(gaussian_config(2727));
        let mut grads: Vec<f64> = vec![];
        inj.inject(&mut grads);
        assert_eq!(inj.stats().total_injections, 1);
        assert_eq!(inj.stats().total_elements, 0);
    }

    // -- DpSgdPrivatizer ----------------------------------------------------

    fn l2_norm(values: &[f64]) -> f64 {
        values.iter().map(|v| v * v).sum::<f64>().sqrt()
    }

    #[test]
    fn dp_sgd_rejects_invalid_params() {
        assert!(matches!(
            DpSgdPrivatizer::new(0.0, 1.0),
            Err(DpSgdError::InvalidClipBound(_))
        ));
        assert!(matches!(
            DpSgdPrivatizer::new(-1.0, 1.0),
            Err(DpSgdError::InvalidClipBound(_))
        ));
        assert!(matches!(
            DpSgdPrivatizer::new(f64::NAN, 1.0),
            Err(DpSgdError::InvalidClipBound(_))
        ));
        assert!(matches!(
            DpSgdPrivatizer::new(1.0, -0.1),
            Err(DpSgdError::InvalidNoiseMultiplier(_))
        ));
        assert!(matches!(
            DpSgdPrivatizer::new(1.0, f64::NAN),
            Err(DpSgdError::InvalidNoiseMultiplier(_))
        ));
        // z = 0.0 (clipping only) is valid.
        assert!(DpSgdPrivatizer::new(1.0, 0.0).is_ok());
    }

    #[test]
    fn clips_before_noise() {
        // With noise_multiplier = 0.0 there is no noise, so the output is exactly
        // the clipped gradient and its L2 norm must equal the clip bound.
        let mut dp = DpSgdPrivatizer::with_deterministic_seed_for_testing(1.0, 0.0, 42)
            .expect("valid DP-SGD parameters");
        let mut grad = vec![3.0, 4.0]; // L2 norm 5.0 > clip_bound 1.0
        dp.privatize(&mut grad);
        let norm = l2_norm(&grad);
        assert!(
            norm <= 1.0 + 1e-9,
            "clipped L2 norm {norm} must not exceed clip_bound"
        );
        assert!(
            (norm - 1.0).abs() < 1e-9,
            "over-threshold gradient must be scaled to exactly clip_bound, got {norm}"
        );
    }

    #[test]
    fn clips_before_noise_leaves_small_gradient_unchanged() {
        // A gradient already within the clip bound is not rescaled (z = 0).
        let mut dp = DpSgdPrivatizer::with_deterministic_seed_for_testing(10.0, 0.0, 1)
            .expect("valid DP-SGD parameters");
        let mut grad = vec![3.0, 4.0]; // norm 5.0 < clip_bound 10.0
        dp.privatize(&mut grad);
        assert!((grad[0] - 3.0).abs() < 1e-12);
        assert!((grad[1] - 4.0).abs() < 1e-12);
    }

    #[test]
    fn noise_std_tracks_clip_bound() {
        // σ = z · C.
        let dp = DpSgdPrivatizer::new(2.0, 1.1).expect("valid DP-SGD parameters");
        assert!((dp.noise_std() - 2.2).abs() < 1e-12);
        assert!((dp.clip_bound() - 2.0).abs() < 1e-12);
        assert!((dp.noise_multiplier() - 1.1).abs() < 1e-12);

        let dp2 = DpSgdPrivatizer::new(0.5, 4.0).expect("valid DP-SGD parameters");
        assert!((dp2.noise_std() - 2.0).abs() < 1e-12);
    }

    #[test]
    fn deterministic_seed_reproduces() {
        let base = vec![5.0, -3.0, 2.0, 8.0, -1.0];
        let mut a = DpSgdPrivatizer::with_deterministic_seed_for_testing(1.0, 1.5, 99)
            .expect("valid DP-SGD parameters");
        let mut b = DpSgdPrivatizer::with_deterministic_seed_for_testing(1.0, 1.5, 99)
            .expect("valid DP-SGD parameters");
        let mut ga = base.clone();
        let mut gb = base.clone();
        a.privatize(&mut ga);
        b.privatize(&mut gb);
        assert_eq!(ga, gb, "same seed + same input must reproduce output");

        let mut c = DpSgdPrivatizer::with_deterministic_seed_for_testing(1.0, 1.5, 100)
            .expect("valid DP-SGD parameters");
        let mut gc = base.clone();
        c.privatize(&mut gc);
        assert_ne!(ga, gc, "a different seed must diverge");
    }

    #[test]
    fn added_noise_std_approximates_sigma() {
        // A zero gradient has L2 norm 0 ≤ C, so clipping is a no-op and the whole
        // output IS the added Gaussian noise. Its empirical std must approximate
        // σ = z · C. Deterministic seed → reproducible; the 0.05 tolerance is many
        // sigma for n = 50 000 samples (σ_of_std ≈ σ / sqrt(2n) ≈ 0.003).
        let clip_bound = 1.0_f64;
        let noise_multiplier = 1.0_f64;
        let sigma = noise_multiplier * clip_bound;
        let n = 50_000usize;
        let mut dp = DpSgdPrivatizer::with_deterministic_seed_for_testing(
            clip_bound,
            noise_multiplier,
            2024,
        )
        .expect("valid DP-SGD parameters");
        let mut grad = vec![0.0; n];
        dp.privatize(&mut grad);

        for v in &grad {
            assert!(v.is_finite(), "noise must be finite, got {v}");
        }
        let mean = grad.iter().sum::<f64>() / n as f64;
        let var = grad.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0);
        let std = var.sqrt();
        assert!(
            (std - sigma).abs() < 0.05,
            "empirical noise std {std} should approximate σ = {sigma}"
        );
        assert!(mean.abs() < 0.05, "noise mean {mean} should be near 0");
    }

    #[test]
    fn privatize_empty_gradient_no_panic() {
        let mut dp = DpSgdPrivatizer::with_deterministic_seed_for_testing(1.0, 1.0, 5)
            .expect("valid DP-SGD parameters");
        let mut grad: Vec<f64> = vec![];
        dp.privatize(&mut grad);
        assert!(grad.is_empty());
    }
}
