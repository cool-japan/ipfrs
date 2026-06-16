//! LossScaler -- dynamic loss scaling for mixed-precision training.
//!
//! Mixed-precision training uses reduced-precision floats (FP16/BF16) to cut
//! memory and compute costs, but reduced dynamic range makes small gradients
//! underflow to zero.  Loss scaling multiplies the loss (and thus all
//! gradients) by a large scalar before the backward pass, then divides back
//! before the optimizer step.  When an overflow is detected the scale is
//! reduced; after a run of clean steps it is increased again.
//!
//! ## Policies
//!
//! | Policy    | On success streak         | On overflow           |
//! |-----------|---------------------------|-----------------------|
//! | Static    | no change                 | no change             |
//! | Dynamic   | double every N steps      | halve immediately     |
//! | Gradual   | multiply by `scale_up_factor` every `scale_up_interval` steps | multiply by `scale_down_factor` |
//!
//! # Examples
//!
//! ```
//! use ipfrs_tensorlogic::{LossScaler, LossScalerConfig, ScaleUpdatePolicy};
//!
//! let config = LossScalerConfig {
//!     policy: ScaleUpdatePolicy::Dynamic,
//!     initial_scale: 65536.0,
//!     scale_up_interval: 2000,
//!     ..LossScalerConfig::default()
//! };
//! let mut scaler = LossScaler::new(config);
//!
//! // Forward pass: scale the loss before backward.
//! let scaled = scaler.scale_loss(0.5);
//! assert_eq!(scaled, 0.5 * 65536.0);
//!
//! // After backward: check and unscale gradients.
//! let mut grads = vec![1.0_f64, 2.0, 3.0];
//! scaler.unscale_gradients(&mut grads);
//!
//! // Update scale based on whether an overflow was detected.
//! let overflow = LossScaler::has_overflow(&grads);
//! scaler.update(overflow);
//! ```

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Determines how the loss scale is adjusted over training.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ScaleUpdatePolicy {
    /// Fixed scale -- never adjusted.
    Static,
    /// Classic dynamic loss scaling: double after `scale_up_interval` clean
    /// steps, halve immediately on overflow.
    Dynamic,
    /// Gradual policy: multiply by `scale_up_factor` after
    /// `scale_up_interval` clean steps, multiply by `scale_down_factor` on
    /// overflow (both factors should be in (0, ∞), down_factor < 1).
    Gradual,
}

impl std::fmt::Display for ScaleUpdatePolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Static => write!(f, "Static"),
            Self::Dynamic => write!(f, "Dynamic"),
            Self::Gradual => write!(f, "Gradual"),
        }
    }
}

/// Configuration parameters for a [`LossScaler`].
#[derive(Debug, Clone)]
pub struct LossScalerConfig {
    /// Which update policy to use.
    pub policy: ScaleUpdatePolicy,
    /// Starting loss scale (default 65536.0 = 2^16).
    pub initial_scale: f64,
    /// Floor for the loss scale (default 1.0).
    pub min_scale: f64,
    /// Ceiling for the loss scale (default 2^24 = 16_777_216.0).
    pub max_scale: f64,
    /// Multiplicative factor applied to the scale on a successful streak
    /// (Dynamic policy uses 2.0 to double; Gradual uses a configurable
    /// value close to 1, e.g. 1.001).
    pub scale_up_factor: f64,
    /// Multiplicative factor applied on overflow (should be < 1, e.g. 0.5).
    pub scale_down_factor: f64,
    /// Number of clean (non-overflow) steps required before a scale-up.
    /// For Dynamic this is often 2000; for Gradual it can be smaller.
    pub scale_up_interval: u64,
}

impl Default for LossScalerConfig {
    fn default() -> Self {
        Self {
            policy: ScaleUpdatePolicy::Dynamic,
            initial_scale: 65_536.0,
            min_scale: 1.0,
            max_scale: 16_777_216.0, // 2^24
            scale_up_factor: 2.0,
            scale_down_factor: 0.5,
            scale_up_interval: 2000,
        }
    }
}

/// Snapshot of scaler run-time statistics.
#[derive(Debug, Clone, Default)]
pub struct ScalerStats {
    /// Total number of `update()` calls.
    pub total_steps: u64,
    /// Number of steps on which overflow was detected.
    pub overflow_events: u64,
    /// Number of times the scale was increased.
    pub scale_ups: u64,
    /// Number of times the scale was decreased.
    pub scale_downs: u64,
    /// Current loss scale at the time the snapshot was taken.
    pub current_scale: f64,
}

/// Dynamic loss scaler for mixed-precision training.
///
/// See the [module-level documentation](self) for an overview and examples.
#[derive(Debug, Clone)]
pub struct LossScaler {
    config: LossScalerConfig,
    current_scale: f64,
    /// Steps elapsed since the last overflow (or since creation).
    steps_since_overflow: u64,
    /// Total overflow events counted (mirrors stats, kept separately for
    /// clarity).
    overflow_count: u64,
    stats: ScalerStats,
}

impl LossScaler {
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    /// Create a new [`LossScaler`] from `config`.  The initial scale is taken
    /// from `config.initial_scale` and clamped into `[min_scale, max_scale]`.
    pub fn new(config: LossScalerConfig) -> Self {
        let scale = config
            .initial_scale
            .clamp(config.min_scale, config.max_scale);
        let mut scaler = Self {
            config,
            current_scale: scale,
            steps_since_overflow: 0,
            overflow_count: 0,
            stats: ScalerStats::default(),
        };
        scaler.stats.current_scale = scaler.current_scale;
        scaler
    }

    // -----------------------------------------------------------------------
    // Core operations
    // -----------------------------------------------------------------------

    /// Multiply `loss` by the current scale.  Call this *before* the backward
    /// pass.
    #[inline]
    pub fn scale_loss(&self, loss: f64) -> f64 {
        loss * self.current_scale
    }

    /// Divide every element of `grads` by the current scale in-place.  Call
    /// this *after* the backward pass and *before* passing gradients to the
    /// optimizer.
    ///
    /// A zero current scale is handled gracefully: if the scale is exactly 0,
    /// the gradients are left unchanged (division by zero is avoided).
    pub fn unscale_gradients(&self, grads: &mut [f64]) {
        if self.current_scale == 0.0 {
            return;
        }
        let inv = 1.0 / self.current_scale;
        for g in grads.iter_mut() {
            *g *= inv;
        }
    }

    /// Return `true` if any element of `grads` is `NaN` or `±∞`.
    pub fn has_overflow(grads: &[f64]) -> bool {
        grads.iter().any(|&g| !Self::is_finite(g))
    }

    /// Update the loss scale based on whether an overflow occurred this step.
    ///
    /// For [`ScaleUpdatePolicy::Static`] this is a no-op (stats are still
    /// updated).
    pub fn update(&mut self, overflow: bool) {
        self.stats.total_steps += 1;

        if overflow {
            self.overflow_count += 1;
            self.stats.overflow_events += 1;
            self.steps_since_overflow = 0;

            match self.config.policy {
                ScaleUpdatePolicy::Static => {}
                ScaleUpdatePolicy::Dynamic => {
                    self.current_scale *= 0.5;
                    self.clamp_scale();
                    self.stats.scale_downs += 1;
                }
                ScaleUpdatePolicy::Gradual => {
                    self.current_scale *= self.config.scale_down_factor;
                    self.clamp_scale();
                    self.stats.scale_downs += 1;
                }
            }
        } else {
            self.steps_since_overflow += 1;

            match self.config.policy {
                ScaleUpdatePolicy::Static => {}
                ScaleUpdatePolicy::Dynamic => {
                    if self.steps_since_overflow >= self.config.scale_up_interval {
                        self.current_scale *= self.config.scale_up_factor;
                        self.clamp_scale();
                        self.stats.scale_ups += 1;
                        self.steps_since_overflow = 0;
                    }
                }
                ScaleUpdatePolicy::Gradual => {
                    if self.steps_since_overflow >= self.config.scale_up_interval {
                        self.current_scale *= self.config.scale_up_factor;
                        self.clamp_scale();
                        self.stats.scale_ups += 1;
                        self.steps_since_overflow = 0;
                    }
                }
            }
        }

        self.stats.current_scale = self.current_scale;
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /// Return the current loss scale.
    #[inline]
    pub fn current_scale(&self) -> f64 {
        self.current_scale
    }

    /// Return a reference to the accumulated statistics.
    #[inline]
    pub fn stats(&self) -> &ScalerStats {
        &self.stats
    }

    /// Read-only access to the configuration.
    #[inline]
    pub fn config(&self) -> &LossScalerConfig {
        &self.config
    }

    // -----------------------------------------------------------------------
    // Mutations
    // -----------------------------------------------------------------------

    /// Reset the scaler to its initial state (scale, counters, stats).
    pub fn reset(&mut self) {
        self.current_scale = self
            .config
            .initial_scale
            .clamp(self.config.min_scale, self.config.max_scale);
        self.steps_since_overflow = 0;
        self.overflow_count = 0;
        self.stats = ScalerStats {
            current_scale: self.current_scale,
            ..ScalerStats::default()
        };
    }

    // -----------------------------------------------------------------------
    // Helper utilities (public so callers can reuse them)
    // -----------------------------------------------------------------------

    /// Return `true` iff `x` is neither `NaN` nor `±∞`.
    #[inline]
    pub fn is_finite(x: f64) -> bool {
        x.is_finite()
    }

    /// Clamp `current_scale` into `[config.min_scale, config.max_scale]`.
    pub fn clamp_scale(&mut self) {
        self.current_scale = self
            .current_scale
            .clamp(self.config.min_scale, self.config.max_scale);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---------------------------------------------------------------- helpers

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    fn dynamic_scaler() -> LossScaler {
        LossScaler::new(LossScalerConfig {
            policy: ScaleUpdatePolicy::Dynamic,
            initial_scale: 1024.0,
            min_scale: 1.0,
            max_scale: 1_048_576.0,
            scale_up_factor: 2.0,
            scale_down_factor: 0.5,
            scale_up_interval: 5,
        })
    }

    fn static_scaler() -> LossScaler {
        LossScaler::new(LossScalerConfig {
            policy: ScaleUpdatePolicy::Static,
            initial_scale: 512.0,
            ..LossScalerConfig::default()
        })
    }

    fn gradual_scaler() -> LossScaler {
        LossScaler::new(LossScalerConfig {
            policy: ScaleUpdatePolicy::Gradual,
            initial_scale: 256.0,
            min_scale: 1.0,
            max_scale: 1_048_576.0,
            scale_up_factor: 1.1,
            scale_down_factor: 0.8,
            scale_up_interval: 3,
        })
    }

    // --------------------------------------------------------- scale_loss

    #[test]
    fn scale_loss_multiplies_by_scale() {
        let scaler = dynamic_scaler();
        let scaled = scaler.scale_loss(2.5);
        assert!(approx_eq(scaled, 2.5 * 1024.0, 1e-10));
    }

    #[test]
    fn scale_loss_zero_loss_remains_zero() {
        let scaler = dynamic_scaler();
        assert_eq!(scaler.scale_loss(0.0), 0.0);
    }

    #[test]
    fn scale_loss_negative_loss() {
        let scaler = dynamic_scaler();
        let scaled = scaler.scale_loss(-1.0);
        assert!(approx_eq(scaled, -1024.0, 1e-10));
    }

    // ------------------------------------------------------ unscale_gradients

    #[test]
    fn unscale_gradients_divides_by_scale() {
        let scaler = dynamic_scaler();
        let mut grads = vec![1024.0, 2048.0, 512.0];
        scaler.unscale_gradients(&mut grads);
        assert!(approx_eq(grads[0], 1.0, 1e-10));
        assert!(approx_eq(grads[1], 2.0, 1e-10));
        assert!(approx_eq(grads[2], 0.5, 1e-10));
    }

    #[test]
    fn unscale_gradients_empty_slice_ok() {
        let scaler = dynamic_scaler();
        let mut grads: Vec<f64> = vec![];
        scaler.unscale_gradients(&mut grads); // must not panic
    }

    #[test]
    fn unscale_gradients_zero_scale_noop() {
        // Construct a scaler whose scale is forced to zero via min_scale = 0.
        let mut scaler = LossScaler::new(LossScalerConfig {
            policy: ScaleUpdatePolicy::Static,
            initial_scale: 0.0,
            min_scale: 0.0,
            max_scale: 1.0,
            ..LossScalerConfig::default()
        });
        scaler.current_scale = 0.0; // bypass clamp by direct mutation
        let mut grads = vec![3.0, 4.0];
        scaler.unscale_gradients(&mut grads);
        // Values must be unchanged (division-by-zero guard).
        assert!(approx_eq(grads[0], 3.0, 1e-15));
        assert!(approx_eq(grads[1], 4.0, 1e-15));
    }

    #[test]
    fn unscale_zero_gradient_stays_zero() {
        let scaler = dynamic_scaler();
        let mut grads = vec![0.0_f64];
        scaler.unscale_gradients(&mut grads);
        assert_eq!(grads[0], 0.0);
    }

    // -------------------------------------------------------- has_overflow

    #[test]
    fn has_overflow_clean_gradients_false() {
        let grads = vec![0.1, 0.2, -0.3, 0.0];
        assert!(!LossScaler::has_overflow(&grads));
    }

    #[test]
    fn has_overflow_nan_detected() {
        let grads = vec![1.0, f64::NAN, 3.0];
        assert!(LossScaler::has_overflow(&grads));
    }

    #[test]
    fn has_overflow_positive_inf_detected() {
        let grads = vec![1.0, f64::INFINITY];
        assert!(LossScaler::has_overflow(&grads));
    }

    #[test]
    fn has_overflow_negative_inf_detected() {
        let grads = vec![f64::NEG_INFINITY, 0.0];
        assert!(LossScaler::has_overflow(&grads));
    }

    #[test]
    fn has_overflow_empty_slice_false() {
        assert!(!LossScaler::has_overflow(&[]));
    }

    // ----------------------------------------- Dynamic: scale-up after streak

    #[test]
    fn dynamic_scale_up_after_interval() {
        let mut scaler = dynamic_scaler(); // interval = 5
        let initial = scaler.current_scale();
        for _ in 0..5 {
            scaler.update(false);
        }
        // After 5 clean steps the scale should have doubled once.
        assert!(approx_eq(scaler.current_scale(), initial * 2.0, 1e-10));
        assert_eq!(scaler.stats().scale_ups, 1);
    }

    #[test]
    fn dynamic_scale_up_resets_streak_counter() {
        let mut scaler = dynamic_scaler(); // interval = 5
                                           // First scale-up at step 5.
        for _ in 0..5 {
            scaler.update(false);
        }
        let after_first = scaler.current_scale();
        // Five more clean steps → second scale-up.
        for _ in 0..5 {
            scaler.update(false);
        }
        assert!(approx_eq(scaler.current_scale(), after_first * 2.0, 1e-10));
        assert_eq!(scaler.stats().scale_ups, 2);
    }

    #[test]
    fn dynamic_no_scale_up_before_interval() {
        let mut scaler = dynamic_scaler(); // interval = 5
        let initial = scaler.current_scale();
        for _ in 0..4 {
            scaler.update(false);
        }
        // 4 clean steps -- still below threshold.
        assert!(approx_eq(scaler.current_scale(), initial, 1e-10));
        assert_eq!(scaler.stats().scale_ups, 0);
    }

    // --------------------------------------- Dynamic: scale-down on overflow

    #[test]
    fn dynamic_scale_down_on_overflow() {
        let mut scaler = dynamic_scaler();
        let initial = scaler.current_scale();
        scaler.update(true);
        assert!(approx_eq(scaler.current_scale(), initial * 0.5, 1e-10));
        assert_eq!(scaler.stats().scale_downs, 1);
    }

    #[test]
    fn dynamic_consecutive_overflows_halve_repeatedly() {
        let mut scaler = dynamic_scaler();
        let initial = scaler.current_scale();
        for i in 1..=4 {
            scaler.update(true);
            let expected = initial * 0.5_f64.powi(i);
            assert!(
                approx_eq(scaler.current_scale(), expected, 1e-8),
                "after {i} overflows: expected {expected}, got {}",
                scaler.current_scale()
            );
        }
        assert_eq!(scaler.stats().scale_downs, 4);
        assert_eq!(scaler.stats().overflow_events, 4);
    }

    #[test]
    fn dynamic_overflow_resets_streak() {
        let mut scaler = dynamic_scaler(); // interval = 5
                                           // Build a streak of 4.
        for _ in 0..4 {
            scaler.update(false);
        }
        // Overflow resets the streak.
        scaler.update(true);
        // Four more clean -- still below interval of 5.
        for _ in 0..4 {
            scaler.update(false);
        }
        assert_eq!(scaler.stats().scale_ups, 0);
    }

    // --------------------------------------------------- min/max clamping

    #[test]
    fn scale_does_not_exceed_max() {
        let mut scaler = LossScaler::new(LossScalerConfig {
            policy: ScaleUpdatePolicy::Dynamic,
            initial_scale: 512.0,
            max_scale: 1024.0,
            scale_up_factor: 4.0,
            scale_up_interval: 1,
            ..LossScalerConfig::default()
        });
        scaler.update(false); // would multiply by 4 → 2048 > max
        assert!(scaler.current_scale() <= 1024.0);
    }

    #[test]
    fn scale_does_not_fall_below_min() {
        let mut scaler = LossScaler::new(LossScalerConfig {
            policy: ScaleUpdatePolicy::Dynamic,
            initial_scale: 4.0,
            min_scale: 2.0,
            scale_down_factor: 0.1,
            scale_up_interval: 100,
            ..LossScalerConfig::default()
        });
        scaler.update(true); // 4 * 0.5 = 2 → equals min
        assert!(scaler.current_scale() >= 2.0);
        scaler.update(true); // further halve → would be 1, but min = 2
        assert!(scaler.current_scale() >= 2.0);
    }

    #[test]
    fn clamp_scale_direct_call() {
        let mut scaler = dynamic_scaler();
        scaler.current_scale = 1e18; // manually exceed max
        scaler.clamp_scale();
        assert!(scaler.current_scale() <= scaler.config().max_scale);

        scaler.current_scale = -5.0; // below min
        scaler.clamp_scale();
        assert!(scaler.current_scale() >= scaler.config().min_scale);
    }

    // -------------------------------------------------- Static policy

    #[test]
    fn static_policy_never_changes_on_success() {
        let mut scaler = static_scaler();
        let initial = scaler.current_scale();
        for _ in 0..1000 {
            scaler.update(false);
        }
        assert!(approx_eq(scaler.current_scale(), initial, 1e-10));
        assert_eq!(scaler.stats().scale_ups, 0);
    }

    #[test]
    fn static_policy_never_changes_on_overflow() {
        let mut scaler = static_scaler();
        let initial = scaler.current_scale();
        for _ in 0..50 {
            scaler.update(true);
        }
        assert!(approx_eq(scaler.current_scale(), initial, 1e-10));
        assert_eq!(scaler.stats().scale_downs, 0);
        // Stats for events should still be tracked.
        assert_eq!(scaler.stats().overflow_events, 50);
    }

    // -------------------------------------------------- Gradual policy

    #[test]
    fn gradual_scale_up_after_interval() {
        let mut scaler = gradual_scaler(); // interval = 3, factor = 1.1
        let initial = scaler.current_scale();
        for _ in 0..3 {
            scaler.update(false);
        }
        let expected = initial * 1.1;
        assert!(
            approx_eq(scaler.current_scale(), expected, 1e-8),
            "gradual scale_up: expected {expected}, got {}",
            scaler.current_scale()
        );
        assert_eq!(scaler.stats().scale_ups, 1);
    }

    #[test]
    fn gradual_scale_down_on_overflow() {
        let mut scaler = gradual_scaler(); // down_factor = 0.8
        let initial = scaler.current_scale();
        scaler.update(true);
        let expected = initial * 0.8;
        assert!(
            approx_eq(scaler.current_scale(), expected, 1e-8),
            "gradual scale_down: expected {expected}, got {}",
            scaler.current_scale()
        );
        assert_eq!(scaler.stats().scale_downs, 1);
    }

    #[test]
    fn gradual_multiple_up_cycles() {
        let mut scaler = gradual_scaler(); // interval = 3, factor = 1.1
        let mut expected = scaler.current_scale();
        for _ in 0..3 {
            for _ in 0..3 {
                scaler.update(false);
            }
            expected *= 1.1;
        }
        assert!(
            approx_eq(scaler.current_scale(), expected, 1e-6),
            "after 3 cycles: expected {expected}, got {}",
            scaler.current_scale()
        );
        assert_eq!(scaler.stats().scale_ups, 3);
    }

    // ------------------------------------------------------- reset

    #[test]
    fn reset_restores_initial_scale() {
        let mut scaler = dynamic_scaler();
        for _ in 0..10 {
            scaler.update(true);
        }
        scaler.reset();
        assert!(approx_eq(scaler.current_scale(), 1024.0, 1e-10));
    }

    #[test]
    fn reset_clears_stats() {
        let mut scaler = dynamic_scaler();
        for _ in 0..20 {
            scaler.update(false);
        }
        scaler.update(true);
        scaler.reset();
        let s = scaler.stats();
        assert_eq!(s.total_steps, 0);
        assert_eq!(s.overflow_events, 0);
        assert_eq!(s.scale_ups, 0);
        assert_eq!(s.scale_downs, 0);
        assert!(approx_eq(s.current_scale, 1024.0, 1e-10));
    }

    // -------------------------------------------- stats tracking

    #[test]
    fn stats_total_steps_increments() {
        let mut scaler = dynamic_scaler();
        for i in 1..=10_u64 {
            scaler.update(false);
            assert_eq!(scaler.stats().total_steps, i);
        }
    }

    #[test]
    fn stats_current_scale_reflects_latest() {
        let mut scaler = dynamic_scaler(); // interval = 5
        for _ in 0..5 {
            scaler.update(false);
        }
        assert!(approx_eq(
            scaler.stats().current_scale,
            scaler.current_scale(),
            1e-15
        ));
    }

    // -------------------------------------------- is_finite helper

    #[test]
    fn is_finite_normal_values() {
        assert!(LossScaler::is_finite(0.0));
        assert!(LossScaler::is_finite(1.0));
        assert!(LossScaler::is_finite(-1e300));
    }

    #[test]
    fn is_finite_nan_false() {
        assert!(!LossScaler::is_finite(f64::NAN));
    }

    #[test]
    fn is_finite_inf_false() {
        assert!(!LossScaler::is_finite(f64::INFINITY));
        assert!(!LossScaler::is_finite(f64::NEG_INFINITY));
    }

    // -------------------------------------------- scale factor math

    #[test]
    fn scale_up_factor_math_exact() {
        // Dynamic doubles: 1024 * 2 = 2048
        let mut scaler = dynamic_scaler();
        for _ in 0..5 {
            scaler.update(false);
        }
        assert!(approx_eq(scaler.current_scale(), 2048.0, 1e-10));
    }

    #[test]
    fn scale_down_factor_math_exact() {
        // Dynamic halves: 1024 * 0.5 = 512
        let mut scaler = dynamic_scaler();
        scaler.update(true);
        assert!(approx_eq(scaler.current_scale(), 512.0, 1e-10));
    }

    // -------------------------------------------- mixed overflow / clean

    #[test]
    fn mixed_pattern_tracks_correctly() {
        let mut scaler = dynamic_scaler(); // interval = 5
                                           // 3 clean, 1 overflow, 2 clean -- no scale-up yet.
        for _ in 0..3 {
            scaler.update(false);
        }
        scaler.update(true);
        for _ in 0..2 {
            scaler.update(false);
        }
        let s = scaler.stats();
        assert_eq!(s.total_steps, 6);
        assert_eq!(s.overflow_events, 1);
        assert_eq!(s.scale_ups, 0);
        assert_eq!(s.scale_downs, 1);
    }
}
