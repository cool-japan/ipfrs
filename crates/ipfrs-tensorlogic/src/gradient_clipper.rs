//! Gradient clipping strategies for distributed tensor learning.
//!
//! Provides norm clipping and value clipping to prevent gradient explosion
//! during distributed tensor training.

/// Strategy for clipping gradients.
#[derive(Clone, Debug, PartialEq)]
pub enum ClippingStrategy {
    /// Clip all gradients so that their global L2 norm is at most `max_norm`.
    GlobalNorm {
        /// Maximum allowed global L2 norm.
        max_norm: f64,
    },
    /// Clip each gradient tensor independently so its L2 norm is at most `max_norm`.
    PerTensorNorm {
        /// Maximum allowed per-tensor L2 norm.
        max_norm: f64,
    },
    /// Clamp every scalar value in every tensor to the range `[min, max]`.
    ValueClip {
        /// Minimum allowed value.
        min: f64,
        /// Maximum allowed value.
        max: f64,
    },
    /// Running EMA of the global norm; clip when current norm > EMA * 1.5.
    ///
    /// The EMA is updated as: `ema = momentum * ema + (1 - momentum) * global_norm`.
    /// On the very first call the EMA is bootstrapped to the current global norm so
    /// no spurious clipping occurs on the initial step.
    Adaptive {
        /// Target norm used to scale clipping (the clip threshold is `ema * 1.5`).
        target_norm: f64,
        /// EMA momentum coefficient (should be in `[0, 1)`).
        momentum: f64,
    },
}

// ─── GradientTensor ──────────────────────────────────────────────────────────

/// A single gradient tensor identified by a unique id.
#[derive(Clone, Debug)]
pub struct GradientTensor {
    /// Unique identifier for this tensor.
    pub tensor_id: u64,
    /// The gradient values.
    pub values: Vec<f64>,
}

impl GradientTensor {
    /// Compute the L2 norm (Euclidean length) of the gradient values.
    ///
    /// Returns `0.0` for an empty tensor.
    pub fn l2_norm(&self) -> f64 {
        if self.values.is_empty() {
            return 0.0;
        }
        let sum_sq: f64 = self.values.iter().map(|v| v * v).sum();
        sum_sq.sqrt()
    }

    /// Return the maximum absolute value among all elements.
    ///
    /// Returns `0.0` for an empty tensor.
    pub fn max_abs_value(&self) -> f64 {
        self.values.iter().map(|v| v.abs()).fold(0.0_f64, f64::max)
    }
}

// ─── ClippingResult ──────────────────────────────────────────────────────────

/// The result of a clipping operation on a single tensor.
#[derive(Clone, Debug)]
pub struct ClippingResult {
    /// The id of the tensor that was (possibly) clipped.
    pub tensor_id: u64,
    /// L2 norm of the tensor **before** clipping.
    pub original_norm: f64,
    /// L2 norm of the tensor **after** clipping.
    pub clipped_norm: f64,
    /// `true` if any values were actually changed by the clipper.
    pub was_clipped: bool,
}

// ─── ClipperStats ────────────────────────────────────────────────────────────

/// Cumulative statistics for a [`TensorGradientClipper`].
#[derive(Clone, Debug, Default)]
pub struct ClipperStats {
    /// Number of times [`TensorGradientClipper::clip`] has been called.
    pub total_clip_calls: u64,
    /// Total number of tensors processed across all clip calls.
    pub total_tensors_processed: u64,
    /// Number of tensors for which clipping was actually applied.
    pub total_clipped: u64,
    /// Running mean of `clipped_norm / original_norm` for clipped tensors.
    ///
    /// `1.0` when no tensor has been clipped yet.
    pub avg_clip_ratio: f64,
}

// ─── TensorGradientClipper ───────────────────────────────────────────────────

/// Applies gradient-clipping strategies to collections of [`GradientTensor`]s.
pub struct TensorGradientClipper {
    /// The clipping strategy in use.
    pub strategy: ClippingStrategy,
    /// Cumulative statistics.
    pub stats: ClipperStats,
    /// EMA of the global norm (used only by [`ClippingStrategy::Adaptive`]).
    pub ema_norm: f64,
}

impl TensorGradientClipper {
    /// Create a new clipper with the given strategy and zeroed statistics.
    pub fn new(strategy: ClippingStrategy) -> Self {
        Self {
            strategy,
            stats: ClipperStats {
                avg_clip_ratio: 1.0,
                ..ClipperStats::default()
            },
            ema_norm: 0.0,
        }
    }

    /// Apply the configured clipping strategy to `tensors` in-place.
    ///
    /// Returns one [`ClippingResult`] per input tensor.
    pub fn clip(&mut self, tensors: &mut [GradientTensor]) -> Vec<ClippingResult> {
        self.stats.total_clip_calls += 1;
        self.stats.total_tensors_processed += tensors.len() as u64;

        let results = match &self.strategy.clone() {
            ClippingStrategy::GlobalNorm { max_norm } => self.apply_global_norm(tensors, *max_norm),
            ClippingStrategy::PerTensorNorm { max_norm } => {
                self.apply_per_tensor_norm(tensors, *max_norm)
            }
            ClippingStrategy::ValueClip { min, max } => self.apply_value_clip(tensors, *min, *max),
            ClippingStrategy::Adaptive { momentum, .. } => {
                let momentum = *momentum;
                self.apply_adaptive(tensors, momentum)
            }
        };

        // Update stats for each result
        for result in &results {
            if result.was_clipped {
                self.stats.total_clipped += 1;
                let ratio = if result.original_norm > 0.0 {
                    result.clipped_norm / result.original_norm
                } else {
                    1.0
                };
                // Update running mean of clip ratio for clipped tensors
                let n = self.stats.total_clipped as f64;
                self.stats.avg_clip_ratio =
                    self.stats.avg_clip_ratio + (ratio - self.stats.avg_clip_ratio) / n;
            }
        }

        results
    }

    /// Reset all statistics and the EMA norm to their initial state.
    pub fn reset_stats(&mut self) {
        self.stats = ClipperStats {
            avg_clip_ratio: 1.0,
            ..ClipperStats::default()
        };
        self.ema_norm = 0.0;
    }

    /// Return a reference to the current statistics.
    pub fn stats(&self) -> &ClipperStats {
        &self.stats
    }

    // ── private helpers ──────────────────────────────────────────────────────

    fn apply_global_norm(
        &self,
        tensors: &mut [GradientTensor],
        max_norm: f64,
    ) -> Vec<ClippingResult> {
        // Compute global norm = sqrt(sum of all per-tensor squared norms)
        let sum_sq: f64 = tensors.iter().map(|t| t.l2_norm().powi(2)).sum();
        let global_norm = sum_sq.sqrt();

        if global_norm > max_norm && global_norm > 0.0 {
            let scale = max_norm / global_norm;
            tensors.iter_mut().for_each(|t| {
                t.values.iter_mut().for_each(|v| *v *= scale);
            });
            tensors
                .iter()
                .map(|t| {
                    // After scaling: tensor_norm * scale
                    let original = t.l2_norm() / scale; // reverse-engineer pre-clip norm
                    let clipped = t.l2_norm();
                    ClippingResult {
                        tensor_id: t.tensor_id,
                        original_norm: original,
                        clipped_norm: clipped,
                        was_clipped: true,
                    }
                })
                .collect()
        } else {
            tensors
                .iter()
                .map(|t| {
                    let norm = t.l2_norm();
                    ClippingResult {
                        tensor_id: t.tensor_id,
                        original_norm: norm,
                        clipped_norm: norm,
                        was_clipped: false,
                    }
                })
                .collect()
        }
    }

    fn apply_per_tensor_norm(
        &self,
        tensors: &mut [GradientTensor],
        max_norm: f64,
    ) -> Vec<ClippingResult> {
        tensors
            .iter_mut()
            .map(|t| {
                let original_norm = t.l2_norm();
                if original_norm > max_norm && original_norm > 0.0 {
                    let scale = max_norm / original_norm;
                    t.values.iter_mut().for_each(|v| *v *= scale);
                    let clipped_norm = t.l2_norm();
                    ClippingResult {
                        tensor_id: t.tensor_id,
                        original_norm,
                        clipped_norm,
                        was_clipped: true,
                    }
                } else {
                    ClippingResult {
                        tensor_id: t.tensor_id,
                        original_norm,
                        clipped_norm: original_norm,
                        was_clipped: false,
                    }
                }
            })
            .collect()
    }

    fn apply_value_clip(
        &self,
        tensors: &mut [GradientTensor],
        min: f64,
        max: f64,
    ) -> Vec<ClippingResult> {
        tensors
            .iter_mut()
            .map(|t| {
                let original_norm = t.l2_norm();
                let mut any_changed = false;
                t.values.iter_mut().for_each(|v| {
                    let clamped = v.clamp(min, max);
                    if clamped != *v {
                        any_changed = true;
                        *v = clamped;
                    }
                });
                let clipped_norm = t.l2_norm();
                ClippingResult {
                    tensor_id: t.tensor_id,
                    original_norm,
                    clipped_norm,
                    was_clipped: any_changed,
                }
            })
            .collect()
    }

    fn apply_adaptive(
        &mut self,
        tensors: &mut [GradientTensor],
        momentum: f64,
    ) -> Vec<ClippingResult> {
        const SPIKE_THRESHOLD: f64 = 1.5;

        // Compute current global norm
        let sum_sq: f64 = tensors.iter().map(|t| t.l2_norm().powi(2)).sum();
        let global_norm = sum_sq.sqrt();

        // Bootstrap EMA on first call
        if self.ema_norm == 0.0 {
            self.ema_norm = global_norm;
        } else {
            self.ema_norm = momentum * self.ema_norm + (1.0 - momentum) * global_norm;
        }

        let clip_threshold = self.ema_norm * SPIKE_THRESHOLD;

        if global_norm > clip_threshold && global_norm > 0.0 {
            // Apply global norm clip to `clip_threshold`
            let scale = clip_threshold / global_norm;
            tensors.iter_mut().for_each(|t| {
                t.values.iter_mut().for_each(|v| *v *= scale);
            });
            tensors
                .iter()
                .map(|t| {
                    let clipped_norm = t.l2_norm();
                    let original_norm = clipped_norm / scale;
                    ClippingResult {
                        tensor_id: t.tensor_id,
                        original_norm,
                        clipped_norm,
                        was_clipped: true,
                    }
                })
                .collect()
        } else {
            tensors
                .iter()
                .map(|t| {
                    let norm = t.l2_norm();
                    ClippingResult {
                        tensor_id: t.tensor_id,
                        original_norm: norm,
                        clipped_norm: norm,
                        was_clipped: false,
                    }
                })
                .collect()
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-9;

    fn make_tensor(id: u64, values: Vec<f64>) -> GradientTensor {
        GradientTensor {
            tensor_id: id,
            values,
        }
    }

    // ── GradientTensor helpers ────────────────────────────────────────────────

    #[test]
    fn test_l2_norm_empty() {
        let t = make_tensor(0, vec![]);
        assert!((t.l2_norm() - 0.0).abs() < EPS);
    }

    #[test]
    fn test_l2_norm_single() {
        let t = make_tensor(1, vec![3.0]);
        assert!((t.l2_norm() - 3.0).abs() < EPS);
    }

    #[test]
    fn test_l2_norm_pythagorean() {
        // 3-4-5 triple
        let t = make_tensor(2, vec![3.0, 4.0]);
        assert!((t.l2_norm() - 5.0).abs() < EPS);
    }

    #[test]
    fn test_l2_norm_negative_values() {
        let t = make_tensor(3, vec![-3.0, -4.0]);
        assert!((t.l2_norm() - 5.0).abs() < EPS);
    }

    #[test]
    fn test_max_abs_value_empty() {
        let t = make_tensor(4, vec![]);
        assert!((t.max_abs_value() - 0.0).abs() < EPS);
    }

    #[test]
    fn test_max_abs_value_mixed() {
        let t = make_tensor(5, vec![-10.0, 5.0, 3.0]);
        assert!((t.max_abs_value() - 10.0).abs() < EPS);
    }

    #[test]
    fn test_max_abs_value_all_negative() {
        let t = make_tensor(6, vec![-1.0, -2.0, -0.5]);
        assert!((t.max_abs_value() - 2.0).abs() < EPS);
    }

    // ── GlobalNorm ────────────────────────────────────────────────────────────

    #[test]
    fn test_global_norm_no_clip() {
        let mut clipper =
            TensorGradientClipper::new(ClippingStrategy::GlobalNorm { max_norm: 10.0 });
        let mut tensors = vec![make_tensor(1, vec![3.0, 4.0])]; // norm = 5
        let results = clipper.clip(&mut tensors);
        assert_eq!(results.len(), 1);
        assert!(!results[0].was_clipped);
        assert!((results[0].original_norm - 5.0).abs() < EPS);
        assert!((results[0].clipped_norm - 5.0).abs() < EPS);
    }

    #[test]
    fn test_global_norm_clip_proportionally() {
        let mut clipper =
            TensorGradientClipper::new(ClippingStrategy::GlobalNorm { max_norm: 1.0 });
        // global norm = sqrt(9+16) = 5; scale = 1/5
        let mut tensors = vec![make_tensor(1, vec![3.0, 4.0])];
        let results = clipper.clip(&mut tensors);
        assert!(results[0].was_clipped);
        // After clip, norm should be 1.0
        let norm_after = tensors[0].l2_norm();
        assert!((norm_after - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_global_norm_clip_multi_tensor_proportional() {
        let mut clipper =
            TensorGradientClipper::new(ClippingStrategy::GlobalNorm { max_norm: 5.0 });
        // global norm = sqrt(9+16+25) = sqrt(50) ≈ 7.071; scale = 5/7.071
        let mut tensors = vec![make_tensor(1, vec![3.0, 4.0]), make_tensor(2, vec![5.0])];
        let results = clipper.clip(&mut tensors);
        assert!(results[0].was_clipped);
        assert!(results[1].was_clipped);
        // Global norm after clip should equal max_norm
        let new_global: f64 = tensors
            .iter()
            .map(|t| t.l2_norm().powi(2))
            .sum::<f64>()
            .sqrt();
        assert!((new_global - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_global_norm_exactly_at_threshold() {
        let mut clipper =
            TensorGradientClipper::new(ClippingStrategy::GlobalNorm { max_norm: 5.0 });
        let mut tensors = vec![make_tensor(1, vec![3.0, 4.0])]; // norm = 5 exactly
        let results = clipper.clip(&mut tensors);
        assert!(!results[0].was_clipped);
    }

    #[test]
    fn test_global_norm_empty_tensor_list() {
        let mut clipper =
            TensorGradientClipper::new(ClippingStrategy::GlobalNorm { max_norm: 1.0 });
        let mut tensors: Vec<GradientTensor> = vec![];
        let results = clipper.clip(&mut tensors);
        assert!(results.is_empty());
    }

    // ── PerTensorNorm ─────────────────────────────────────────────────────────

    #[test]
    fn test_per_tensor_norm_clips_independently() {
        let mut clipper =
            TensorGradientClipper::new(ClippingStrategy::PerTensorNorm { max_norm: 3.0 });
        let mut tensors = vec![
            make_tensor(1, vec![3.0, 4.0]), // norm=5, will be clipped
            make_tensor(2, vec![1.0, 2.0]), // norm≈2.24, will not be clipped
        ];
        let results = clipper.clip(&mut tensors);
        assert!(results[0].was_clipped);
        assert!(!results[1].was_clipped);
        // Tensor 1 norm should be 3.0
        assert!((tensors[0].l2_norm() - 3.0).abs() < 1e-9);
        // Tensor 2 unchanged
        assert!((tensors[1].values[0] - 1.0).abs() < EPS);
    }

    #[test]
    fn test_per_tensor_norm_no_clip_when_under() {
        let mut clipper =
            TensorGradientClipper::new(ClippingStrategy::PerTensorNorm { max_norm: 10.0 });
        let mut tensors = vec![make_tensor(1, vec![1.0, 1.0])];
        let results = clipper.clip(&mut tensors);
        assert!(!results[0].was_clipped);
    }

    #[test]
    fn test_per_tensor_norm_scale_correctness() {
        let mut clipper =
            TensorGradientClipper::new(ClippingStrategy::PerTensorNorm { max_norm: 1.0 });
        let mut tensors = vec![make_tensor(1, vec![0.0, 5.0])]; // norm=5
        clipper.clip(&mut tensors);
        // After clip, values should be [0.0, 1.0]
        assert!((tensors[0].values[0] - 0.0).abs() < EPS);
        assert!((tensors[0].values[1] - 1.0).abs() < 1e-9);
    }

    // ── ValueClip ─────────────────────────────────────────────────────────────

    #[test]
    fn test_value_clip_clamps_values() {
        let mut clipper = TensorGradientClipper::new(ClippingStrategy::ValueClip {
            min: -1.0,
            max: 1.0,
        });
        let mut tensors = vec![make_tensor(1, vec![-5.0, 0.5, 3.0])];
        let results = clipper.clip(&mut tensors);
        assert!(results[0].was_clipped);
        assert!((tensors[0].values[0] - (-1.0)).abs() < EPS);
        assert!((tensors[0].values[1] - 0.5).abs() < EPS);
        assert!((tensors[0].values[2] - 1.0).abs() < EPS);
    }

    #[test]
    fn test_value_clip_not_clipped_when_in_range() {
        let mut clipper = TensorGradientClipper::new(ClippingStrategy::ValueClip {
            min: -5.0,
            max: 5.0,
        });
        let mut tensors = vec![make_tensor(1, vec![-1.0, 0.0, 2.5])];
        let results = clipper.clip(&mut tensors);
        assert!(!results[0].was_clipped);
    }

    #[test]
    fn test_value_clip_norm_changes() {
        let mut clipper =
            TensorGradientClipper::new(ClippingStrategy::ValueClip { min: 0.0, max: 1.0 });
        let mut tensors = vec![make_tensor(1, vec![2.0, 2.0])];
        let results = clipper.clip(&mut tensors);
        // original norm = sqrt(8) ≈ 2.828
        assert!((results[0].original_norm - 8_f64.sqrt()).abs() < 1e-9);
        // clipped norm = sqrt(2) ≈ 1.414
        assert!((results[0].clipped_norm - 2_f64.sqrt()).abs() < 1e-9);
    }

    #[test]
    fn test_value_clip_empty_list() {
        let mut clipper = TensorGradientClipper::new(ClippingStrategy::ValueClip {
            min: -1.0,
            max: 1.0,
        });
        let mut tensors: Vec<GradientTensor> = vec![];
        let results = clipper.clip(&mut tensors);
        assert!(results.is_empty());
    }

    // ── Adaptive ──────────────────────────────────────────────────────────────

    #[test]
    fn test_adaptive_no_clip_on_first_call() {
        let mut clipper = TensorGradientClipper::new(ClippingStrategy::Adaptive {
            target_norm: 5.0,
            momentum: 0.9,
        });
        let mut tensors = vec![make_tensor(1, vec![3.0, 4.0])]; // norm=5
        let results = clipper.clip(&mut tensors);
        // First call: EMA bootstrapped to global_norm; threshold = global_norm*1.5 > global_norm
        assert!(!results[0].was_clipped, "First call should never clip");
    }

    #[test]
    fn test_adaptive_clips_on_spike() {
        let mut clipper = TensorGradientClipper::new(ClippingStrategy::Adaptive {
            target_norm: 5.0,
            momentum: 0.9,
        });
        // First call: normal gradient, EMA ~ 1.0
        let mut tensors1 = vec![make_tensor(1, vec![1.0])];
        clipper.clip(&mut tensors1);

        // Second call: spike at 3.0 (> 1.0 * 1.5 = 1.5)
        let mut tensors2 = vec![make_tensor(2, vec![3.0])];
        let results2 = clipper.clip(&mut tensors2);
        assert!(results2[0].was_clipped, "Spike should be clipped");
        // After clip, norm should be <= ema_norm * 1.5
    }

    #[test]
    fn test_adaptive_ema_is_updated() {
        let mut clipper = TensorGradientClipper::new(ClippingStrategy::Adaptive {
            target_norm: 5.0,
            momentum: 0.5,
        });
        let mut tensors = vec![make_tensor(1, vec![2.0])]; // norm=2
        clipper.clip(&mut tensors);
        // EMA should be bootstrapped to 2.0
        assert!((clipper.ema_norm - 2.0).abs() < EPS);

        let mut tensors2 = vec![make_tensor(2, vec![4.0])]; // norm=4
        clipper.clip(&mut tensors2);
        // EMA = 0.5*2 + 0.5*4 = 3.0 (uses norm from second call, which was not clipped because 4 <= 2*1.5=3 is false -- 4>3, so it IS clipped)
        // Actually 4 > 2.0*1.5=3.0, so it clips to 3.0; new global_norm passed to EMA update is 4.0 (before clip)
        // EMA = 0.5*2 + 0.5*4 = 3.0
        assert!((clipper.ema_norm - 3.0).abs() < EPS);
    }

    #[test]
    fn test_adaptive_no_clip_when_below_threshold() {
        let mut clipper = TensorGradientClipper::new(ClippingStrategy::Adaptive {
            target_norm: 5.0,
            momentum: 0.9,
        });
        // Bootstrap EMA to 10
        let mut tensors1 = vec![make_tensor(1, vec![10.0])];
        clipper.clip(&mut tensors1);

        // Second call: norm=5 (< 10*1.5=15), should not clip
        let mut tensors2 = vec![make_tensor(2, vec![5.0])];
        let results = clipper.clip(&mut tensors2);
        assert!(!results[0].was_clipped);
    }

    // ── Stats ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_stats_total_clip_calls() {
        let mut clipper =
            TensorGradientClipper::new(ClippingStrategy::GlobalNorm { max_norm: 10.0 });
        let mut t = vec![make_tensor(1, vec![1.0])];
        clipper.clip(&mut t);
        clipper.clip(&mut t);
        assert_eq!(clipper.stats().total_clip_calls, 2);
    }

    #[test]
    fn test_stats_total_tensors_processed() {
        let mut clipper =
            TensorGradientClipper::new(ClippingStrategy::GlobalNorm { max_norm: 10.0 });
        let mut tensors = vec![make_tensor(1, vec![1.0]), make_tensor(2, vec![2.0])];
        clipper.clip(&mut tensors);
        assert_eq!(clipper.stats().total_tensors_processed, 2);
        clipper.clip(&mut tensors);
        assert_eq!(clipper.stats().total_tensors_processed, 4);
    }

    #[test]
    fn test_stats_total_clipped_counts_correctly() {
        let mut clipper =
            TensorGradientClipper::new(ClippingStrategy::PerTensorNorm { max_norm: 3.0 });
        let mut tensors = vec![
            make_tensor(1, vec![3.0, 4.0]), // norm=5, clipped
            make_tensor(2, vec![1.0]),      // norm=1, not clipped
        ];
        clipper.clip(&mut tensors);
        assert_eq!(clipper.stats().total_clipped, 1);
    }

    #[test]
    fn test_stats_avg_clip_ratio_when_no_clipping() {
        let mut clipper =
            TensorGradientClipper::new(ClippingStrategy::GlobalNorm { max_norm: 100.0 });
        let mut tensors = vec![make_tensor(1, vec![1.0])];
        clipper.clip(&mut tensors);
        // No clipping => avg_clip_ratio stays 1.0
        assert!((clipper.stats().avg_clip_ratio - 1.0).abs() < EPS);
    }

    #[test]
    fn test_stats_avg_clip_ratio_running_mean() {
        let mut clipper =
            TensorGradientClipper::new(ClippingStrategy::PerTensorNorm { max_norm: 1.0 });
        // First clipped tensor: original_norm=5, clipped_norm=1 => ratio=0.2
        let mut t1 = vec![make_tensor(1, vec![0.0, 5.0])];
        clipper.clip(&mut t1);
        assert!((clipper.stats().avg_clip_ratio - 0.2).abs() < 1e-6);

        // Second clipped tensor: original_norm=10, clipped_norm=1 => ratio=0.1
        // running mean = (0.2 + 0.1)/2 = 0.15
        let mut t2 = vec![make_tensor(2, vec![0.0, 10.0])];
        clipper.clip(&mut t2);
        assert!((clipper.stats().avg_clip_ratio - 0.15).abs() < 1e-6);
    }

    #[test]
    fn test_reset_stats() {
        let mut clipper =
            TensorGradientClipper::new(ClippingStrategy::GlobalNorm { max_norm: 1.0 });
        let mut tensors = vec![make_tensor(1, vec![5.0])];
        clipper.clip(&mut tensors);
        clipper.reset_stats();
        assert_eq!(clipper.stats().total_clip_calls, 0);
        assert_eq!(clipper.stats().total_tensors_processed, 0);
        assert_eq!(clipper.stats().total_clipped, 0);
        assert!((clipper.stats().avg_clip_ratio - 1.0).abs() < EPS);
        assert!((clipper.ema_norm - 0.0).abs() < EPS);
    }

    #[test]
    fn test_empty_tensor_values_l2_norm() {
        let t = make_tensor(99, vec![]);
        assert!((t.l2_norm() - 0.0).abs() < EPS);
    }

    #[test]
    fn test_global_norm_single_zero_tensor() {
        let mut clipper =
            TensorGradientClipper::new(ClippingStrategy::GlobalNorm { max_norm: 1.0 });
        let mut tensors = vec![make_tensor(1, vec![0.0, 0.0])];
        let results = clipper.clip(&mut tensors);
        // global_norm=0, no scaling
        assert!(!results[0].was_clipped);
    }

    #[test]
    fn test_per_tensor_norm_zero_norm_no_clip() {
        let mut clipper =
            TensorGradientClipper::new(ClippingStrategy::PerTensorNorm { max_norm: 1.0 });
        let mut tensors = vec![make_tensor(1, vec![0.0])];
        let results = clipper.clip(&mut tensors);
        assert!(!results[0].was_clipped);
    }

    #[test]
    fn test_value_clip_boundary_values_not_clipped() {
        let mut clipper = TensorGradientClipper::new(ClippingStrategy::ValueClip {
            min: -1.0,
            max: 1.0,
        });
        let mut tensors = vec![make_tensor(1, vec![-1.0, 1.0])];
        let results = clipper.clip(&mut tensors);
        assert!(!results[0].was_clipped);
    }

    #[test]
    fn test_adaptive_multiple_stable_calls_no_clip() {
        let mut clipper = TensorGradientClipper::new(ClippingStrategy::Adaptive {
            target_norm: 5.0,
            momentum: 0.9,
        });
        for i in 0..5 {
            let mut tensors = vec![make_tensor(i, vec![1.0, 1.0])]; // norm≈1.414 each time
            let results = clipper.clip(&mut tensors);
            assert!(
                !results[0].was_clipped,
                "Stable gradients should not be clipped (call {i})"
            );
        }
    }

    #[test]
    fn test_clipping_result_fields() {
        let mut clipper =
            TensorGradientClipper::new(ClippingStrategy::GlobalNorm { max_norm: 5.0 });
        let mut tensors = vec![make_tensor(42, vec![3.0, 4.0])]; // norm=5, no clip
        let results = clipper.clip(&mut tensors);
        assert_eq!(results[0].tensor_id, 42);
    }
}
