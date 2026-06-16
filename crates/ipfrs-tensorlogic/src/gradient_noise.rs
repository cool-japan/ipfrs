//! Gradient noise injection for training regularization.
//!
//! This module provides configurable noise injection into gradient tensors,
//! useful as a regularization technique during neural network training.
//! Supported noise types include Gaussian, Uniform, Laplacian, and
//! Scheduled Gaussian (where noise magnitude decays over training steps).
//!
//! # Example
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

use std::f64::consts::PI;

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
    pub clip_value: Option<f64>,
    /// Seed for the internal xorshift64 PRNG, enabling reproducibility.
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
/// regularization.
pub struct GradientNoiseInjector {
    config: GradientNoiseConfig,
    rng_state: u64,
    step: u64,
    stats: NoiseStats,
}

impl GradientNoiseInjector {
    /// Create a new injector from the given configuration.
    ///
    /// The internal PRNG is seeded with `config.seed` (or a fallback value if
    /// the seed is zero, since xorshift64 requires a non-zero state).
    pub fn new(config: GradientNoiseConfig) -> Self {
        let seed = if config.seed == 0 {
            0xDEAD_BEEF_CAFE_BABE
        } else {
            config.seed
        };
        let current_scale = config.initial_scale;
        Self {
            config,
            rng_state: seed,
            step: 0,
            stats: NoiseStats {
                current_scale,
                ..NoiseStats::default()
            },
        }
    }

    // -- PRNG helpers -------------------------------------------------------

    /// Advance the xorshift64 state and return a `u64`.
    fn xorshift64(&mut self) -> u64 {
        let mut s = self.rng_state;
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        self.rng_state = s;
        s
    }

    /// Return a uniform `f64` in `[0, 1)`.
    fn uniform_01(&mut self) -> f64 {
        // Use 53 bits of the state for a double in [0,1).
        let bits = self.xorshift64() >> 11;
        (bits as f64) / ((1u64 << 53) as f64)
    }

    /// Sample from a standard Gaussian (mean 0, std 1) using the Box-Muller
    /// transform.
    pub fn next_gaussian(&mut self) -> f64 {
        loop {
            let u1 = self.uniform_01();
            let u2 = self.uniform_01();
            // Guard against log(0).
            if u1 <= f64::EPSILON {
                continue;
            }
            let r = (-2.0 * u1.ln()).sqrt();
            return r * (2.0 * PI * u2).cos();
        }
    }

    /// Sample from a uniform distribution on `[low, high)`.
    pub fn next_uniform(&mut self, low: f64, high: f64) -> f64 {
        let u = self.uniform_01();
        low + u * (high - low)
    }

    /// Sample from a Laplacian distribution with location 0 and the given
    /// `scale`, using the inverse-CDF method.
    pub fn next_laplacian(&mut self, scale: f64) -> f64 {
        let u = self.uniform_01() - 0.5;
        // Avoid log(0) by clamping |u| away from 0.5.
        let abs_u = u.abs().min(0.5 - f64::EPSILON);
        -scale * (1.0 - 2.0 * abs_u).ln() * u.signum()
    }

    /// Clip `value` to `[-clip, +clip]` if a clip bound is configured,
    /// otherwise return `value` unchanged.
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

    /// Reset the step counter, statistics, and re-seed the PRNG.
    pub fn reset(&mut self) {
        self.step = 0;
        self.rng_state = if self.config.seed == 0 {
            0xDEAD_BEEF_CAFE_BABE
        } else {
            self.config.seed
        };
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

    // -- Zero seed fallback -------------------------------------------------

    #[test]
    fn zero_seed_uses_fallback() {
        let config = GradientNoiseConfig {
            noise_type: NoiseType::Gaussian,
            initial_scale: 0.1,
            decay_rate: 0.0,
            clip_value: None,
            seed: 0,
        };
        let mut inj = GradientNoiseInjector::new(config);
        // Should not panic; the fallback seed is non-zero.
        let sample = inj.sample_noise(10);
        assert_eq!(sample.values.len(), 10);
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
}
