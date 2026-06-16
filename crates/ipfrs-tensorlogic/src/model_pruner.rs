//! Model weight pruning with magnitude, structured, and gradual scheduling strategies.
//!
//! This module implements several classical and modern neural-network pruning
//! approaches:
//!
//! * **Magnitude pruning** — zero out individual weights whose absolute value
//!   falls below a fixed threshold.
//! * **Percentile-magnitude pruning** — zero out the bottom *X%* of weights
//!   ranked by absolute magnitude.
//! * **Structured L1 pruning** — remove entire neurons / output channels whose
//!   mean L1 norm is below a threshold (structured sparsity that directly
//!   speeds up inference on most hardware).
//! * **Random pruning** — stochastically mask out *X%* of weights using a
//!   deterministic xorshift64 PRNG seeded from [`PrunerConfig::seed`].
//! * **Gradual pruning** — linearly ramp sparsity from an initial value to a
//!   final value over a user-specified step window (Zhu & Gupta 2018 style).
//!
//! An optional binary mask is maintained alongside each [`LayerWeights`]
//! tensor so that sparse structure can be preserved across optimiser updates.
//!
//! # Example
//!
//! ```rust
//! use ipfrs_tensorlogic::{
//!     ModelPruner, PrunerConfig, PruningStrategy, LayerWeights,
//! };
//!
//! let cfg = PrunerConfig {
//!     strategy: PruningStrategy::Magnitude(0.1),
//!     seed: 42,
//!     update_mask: true,
//! };
//! let mut pruner = ModelPruner::new(cfg);
//!
//! let mut layer = LayerWeights {
//!     name: "fc1".to_string(),
//!     weights: vec![0.05, -0.2, 0.0, 0.3, -0.08],
//!     mask: None,
//! };
//!
//! let result = pruner.prune_layer(&mut layer);
//! assert!(result.sparsity > 0.0);
//! ```

// ── Pruning strategy ─────────────────────────────────────────────────────────

/// Selects the algorithm used to decide which weights to prune.
#[derive(Debug, Clone, PartialEq)]
pub enum PruningStrategy {
    /// Zero out every weight whose absolute value is strictly below `threshold`.
    Magnitude(f64),
    /// Zero out the bottom `percentile`% of weights ranked by absolute
    /// magnitude.  `percentile` must be in [0, 100].
    PercentileMagnitude(f64),
    /// Prune entire neurons (rows) whose mean L1 norm is below `threshold`.
    StructuredL1(f64),
    /// Randomly zero out `fraction`% of weights using the pruner's seeded PRNG.
    /// `fraction` must be in [0, 1].
    RandomPruning(f64),
    /// Linearly increase sparsity from `initial_sparsity` to `final_sparsity`
    /// between `begin_step` and `end_step`.
    GradualPruning {
        /// Starting sparsity (fraction in [0, 1]).
        initial_sparsity: f64,
        /// Target sparsity (fraction in [0, 1]).
        final_sparsity: f64,
        /// Step at which ramping begins.
        begin_step: usize,
        /// Step at which ramping ends (and `final_sparsity` is held).
        end_step: usize,
    },
}

// ── Configuration ─────────────────────────────────────────────────────────────

/// Configuration bundle passed to [`ModelPruner::new`].
#[derive(Debug, Clone)]
pub struct PrunerConfig {
    /// Which pruning algorithm to apply.
    pub strategy: PruningStrategy,
    /// Seed for the internal xorshift64 PRNG (used by
    /// [`PruningStrategy::RandomPruning`]).
    pub seed: u64,
    /// When `true` the pruner maintains and updates a boolean mask on each
    /// [`LayerWeights`]; when `false` the mask field is left as `None`.
    pub update_mask: bool,
}

// ── Data types ────────────────────────────────────────────────────────────────

/// A named layer's weight tensor together with an optional sparsity mask.
#[derive(Debug, Clone)]
pub struct LayerWeights {
    /// Human-readable name, e.g. `"encoder.layer.0.attention.weight"`.
    pub name: String,
    /// Flat weight values (row-major or column-major — the pruner is
    /// layout-agnostic).
    pub weights: Vec<f64>,
    /// Binary mask parallel to `weights`.  `true` = keep, `false` = pruned.
    /// Populated / updated by the pruner only when
    /// [`PrunerConfig::update_mask`] is `true`.
    pub mask: Option<Vec<bool>>,
}

/// Per-layer summary returned by each call to [`ModelPruner::prune_layer`].
#[derive(Debug, Clone)]
pub struct PruningResult {
    /// Name of the layer that was pruned.
    pub layer_name: String,
    /// Total number of weights in the layer (before pruning this step).
    pub weights_before: usize,
    /// Number of newly-zeroed weights introduced by this pruning step.
    pub weights_pruned: usize,
    /// Fraction of all weights that are zero after pruning.
    pub sparsity: f64,
    /// The pruner's internal step counter at the time of pruning.
    pub step: usize,
}

// ── Statistics ────────────────────────────────────────────────────────────────

/// Cumulative statistics tracked by a [`ModelPruner`] across all layers and
/// all pruning steps.
#[derive(Debug, Clone, Default)]
pub struct PrunerStats {
    /// How many times [`ModelPruner::prune_layer`] has been called.
    pub total_pruning_steps: u64,
    /// Total number of weight-zeroing operations performed.
    pub total_weights_pruned: u64,
    /// Running mean sparsity across every [`PruningResult`] produced.
    pub avg_sparsity: f64,
}

// ── Core pruner ───────────────────────────────────────────────────────────────

/// Stateful weight pruner.  Advance the step counter with
/// [`ModelPruner::advance_step`] between training iterations.
pub struct ModelPruner {
    config: PrunerConfig,
    /// Monotonically increasing iteration counter.
    step: usize,
    /// Current xorshift64 state (non-zero initialised from `config.seed`).
    rng_state: u64,
    stats: PrunerStats,
}

impl ModelPruner {
    // ── Construction ─────────────────────────────────────────────────────

    /// Create a new pruner from the supplied configuration.
    ///
    /// The PRNG seed is initialised to `config.seed`, falling back to `1` if
    /// the seed is zero (xorshift64 must not start from zero).
    pub fn new(config: PrunerConfig) -> Self {
        let rng_state = if config.seed == 0 { 1 } else { config.seed };
        Self {
            config,
            step: 0,
            rng_state,
            stats: PrunerStats::default(),
        }
    }

    // ── Public API ────────────────────────────────────────────────────────

    /// Prune a single layer in-place according to the configured strategy.
    ///
    /// The `layer.weights` vector is mutated directly; the mask (if enabled)
    /// is created or updated.
    pub fn prune_layer(&mut self, layer: &mut LayerWeights) -> PruningResult {
        let n = layer.weights.len();
        let zeros_before = layer.weights.iter().filter(|&&w| w == 0.0).count();

        match self.config.strategy.clone() {
            PruningStrategy::Magnitude(threshold) => {
                self.apply_magnitude(layer, threshold);
            }
            PruningStrategy::PercentileMagnitude(pct) => {
                let threshold = Self::compute_threshold(&layer.weights, pct);
                self.apply_magnitude(layer, threshold);
            }
            PruningStrategy::StructuredL1(threshold) => {
                self.apply_structured_l1(layer, threshold);
            }
            PruningStrategy::RandomPruning(fraction) => {
                self.apply_random(layer, fraction);
            }
            PruningStrategy::GradualPruning { .. } => {
                let target = self.current_sparsity_target();
                // Convert fraction to percentile for threshold computation.
                let pct = target * 100.0;
                let threshold = Self::compute_threshold(&layer.weights, pct);
                self.apply_magnitude(layer, threshold);
            }
        }

        if self.config.update_mask {
            Self::rebuild_mask(layer);
        }

        let zeros_after = layer.weights.iter().filter(|&&w| w == 0.0).count();
        let newly_pruned = zeros_after.saturating_sub(zeros_before);
        let sparsity = Self::compute_sparsity(&layer.weights);

        // Update cumulative stats.
        self.stats.total_pruning_steps += 1;
        self.stats.total_weights_pruned += newly_pruned as u64;
        let n_steps = self.stats.total_pruning_steps as f64;
        self.stats.avg_sparsity =
            self.stats.avg_sparsity * (n_steps - 1.0) / n_steps + sparsity / n_steps;

        PruningResult {
            layer_name: layer.name.clone(),
            weights_before: n,
            weights_pruned: newly_pruned,
            sparsity,
            step: self.step,
        }
    }

    /// Prune every layer in `layers` and return one result per layer.
    pub fn prune_all(&mut self, layers: &mut [LayerWeights]) -> Vec<PruningResult> {
        layers.iter_mut().map(|l| self.prune_layer(l)).collect()
    }

    /// Compute the sparsity target for the current step.
    ///
    /// For [`PruningStrategy::GradualPruning`] this linearly interpolates
    /// between `initial_sparsity` and `final_sparsity`.  For all other
    /// strategies it returns the equivalent fixed fraction.
    pub fn current_sparsity_target(&self) -> f64 {
        match &self.config.strategy {
            PruningStrategy::Magnitude(t) => *t,
            PruningStrategy::PercentileMagnitude(pct) => pct / 100.0,
            PruningStrategy::StructuredL1(t) => *t,
            PruningStrategy::RandomPruning(frac) => *frac,
            PruningStrategy::GradualPruning {
                initial_sparsity,
                final_sparsity,
                begin_step,
                end_step,
            } => {
                let s = self.step;
                if s <= *begin_step {
                    *initial_sparsity
                } else if s >= *end_step {
                    *final_sparsity
                } else {
                    let progress = (s - begin_step) as f64 / (end_step - begin_step).max(1) as f64;
                    initial_sparsity + progress * (final_sparsity - initial_sparsity)
                }
            }
        }
    }

    /// Return the weight value at the given `percentile` (0–100) of the
    /// absolute-magnitude distribution.
    ///
    /// Uses a partial sort to avoid allocating a fully sorted copy when only
    /// the boundary value is needed.  Returns `0.0` for empty slices.
    pub fn compute_threshold(weights: &[f64], percentile: f64) -> f64 {
        if weights.is_empty() {
            return 0.0;
        }
        let pct = percentile.clamp(0.0, 100.0);
        let mut magnitudes: Vec<f64> = weights.iter().map(|w| w.abs()).collect();
        magnitudes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = ((pct / 100.0) * magnitudes.len() as f64) as usize;
        let idx = idx.min(magnitudes.len().saturating_sub(1));
        magnitudes[idx]
    }

    /// Fraction of elements in `weights` that are exactly zero.
    pub fn compute_sparsity(weights: &[f64]) -> f64 {
        if weights.is_empty() {
            return 0.0;
        }
        let zeros = weights.iter().filter(|&&w| w == 0.0).count();
        zeros as f64 / weights.len() as f64
    }

    /// Sum of absolute values of `weights`.
    pub fn compute_l1_norm(weights: &[f64]) -> f64 {
        weights.iter().map(|w| w.abs()).sum()
    }

    /// Advance the step counter by one.
    pub fn advance_step(&mut self) {
        self.step += 1;
    }

    /// Generate the next pseudo-random float in [0, 1) using xorshift64.
    ///
    /// The internal state is updated in-place so successive calls yield
    /// independent values.
    pub fn next_uniform_prng(&mut self) -> f64 {
        // xorshift64 — never produces zero so the state invariant is preserved.
        let mut x = self.rng_state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng_state = x;
        // Map to [0, 1) by dividing by 2^64.
        (x as f64) / (u64::MAX as f64 + 1.0)
    }

    /// Zero out entries in `layer.weights` where the corresponding mask entry
    /// is `false`.  If no mask is present this is a no-op.
    pub fn apply_mask(layer: &mut LayerWeights) {
        if let Some(mask) = &layer.mask {
            let mask_clone: Vec<bool> = mask.clone();
            for (w, &keep) in layer.weights.iter_mut().zip(mask_clone.iter()) {
                if !keep {
                    *w = 0.0;
                }
            }
        }
    }

    /// Immutable access to the accumulated statistics.
    pub fn stats(&self) -> &PrunerStats {
        &self.stats
    }

    // ── Internal helpers ──────────────────────────────────────────────────

    /// Zero out all weights with absolute value strictly below `threshold`.
    fn apply_magnitude(&self, layer: &mut LayerWeights, threshold: f64) {
        for w in layer.weights.iter_mut() {
            if w.abs() < threshold {
                *w = 0.0;
            }
        }
    }

    /// Prune entire rows (neurons) whose mean absolute value is below
    /// `threshold`.  The weights tensor is assumed to be laid out as
    /// `num_neurons × neuron_size`, with rows of equal length.  If the tensor
    /// has fewer than two elements we fall back to element-wise magnitude
    /// pruning.
    fn apply_structured_l1(&self, layer: &mut LayerWeights, threshold: f64) {
        let n = layer.weights.len();
        if n < 2 {
            self.apply_magnitude(layer, threshold);
            return;
        }
        // Heuristic: treat the tensor as a 2-D matrix where each "neuron" is
        // a row of `row_len` weights.  We choose the largest divisor of `n`
        // that is at most √n so that we get the most "square" layout.
        let row_len = Self::choose_row_len(n);
        let num_rows = n / row_len;

        for row_idx in 0..num_rows {
            let start = row_idx * row_len;
            let end = start + row_len;
            let row = &layer.weights[start..end];
            let l1_mean = Self::compute_l1_norm(row) / row_len as f64;
            if l1_mean < threshold {
                for w in layer.weights[start..end].iter_mut() {
                    *w = 0.0;
                }
            }
        }
    }

    /// Randomly zero out `fraction` of the weights using the internal PRNG.
    fn apply_random(&mut self, layer: &mut LayerWeights, fraction: f64) {
        let frac = fraction.clamp(0.0, 1.0);
        for w in layer.weights.iter_mut() {
            if self.next_uniform_prng() < frac {
                *w = 0.0;
            }
        }
    }

    /// Rebuild the binary mask for `layer` to reflect its current zero pattern.
    fn rebuild_mask(layer: &mut LayerWeights) {
        let mask: Vec<bool> = layer.weights.iter().map(|&w| w != 0.0).collect();
        layer.mask = Some(mask);
    }

    /// Choose a "row length" for structured pruning by finding the largest
    /// divisor of `n` that is ≤ √n.  Falls back to 1 if none found.
    fn choose_row_len(n: usize) -> usize {
        let sqrt_n = (n as f64).sqrt() as usize;
        for d in (1..=sqrt_n).rev() {
            if n.is_multiple_of(d) {
                return n / d; // row_len = n / d gives num_rows = d
            }
        }
        1
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ───────────────────────────────────────────────────────────

    fn make_layer(name: &str, weights: Vec<f64>) -> LayerWeights {
        LayerWeights {
            name: name.to_string(),
            weights,
            mask: None,
        }
    }

    fn pruner(strategy: PruningStrategy) -> ModelPruner {
        ModelPruner::new(PrunerConfig {
            strategy,
            seed: 42,
            update_mask: true,
        })
    }

    // ── Magnitude pruning ─────────────────────────────────────────────────

    #[test]
    fn magnitude_removes_below_threshold() {
        let mut p = pruner(PruningStrategy::Magnitude(0.1));
        let mut layer = make_layer("l", vec![0.05, -0.2, 0.0, 0.3, -0.08]);
        p.prune_layer(&mut layer);
        // 0.05, 0.0, -0.08 should all be zeroed (abs < 0.1)
        assert_eq!(layer.weights[0], 0.0);
        assert_ne!(layer.weights[1], 0.0); // -0.2 kept
        assert_eq!(layer.weights[2], 0.0);
        assert_ne!(layer.weights[3], 0.0); // 0.3 kept
        assert_eq!(layer.weights[4], 0.0);
    }

    #[test]
    fn magnitude_threshold_zero_prunes_nothing() {
        let mut p = pruner(PruningStrategy::Magnitude(0.0));
        let weights = vec![0.1, -0.2, 0.3];
        let mut layer = make_layer("l", weights.clone());
        p.prune_layer(&mut layer);
        assert_eq!(layer.weights, weights);
    }

    #[test]
    fn magnitude_threshold_high_prunes_all() {
        let mut p = pruner(PruningStrategy::Magnitude(1e9));
        let mut layer = make_layer("l", vec![1.0, -2.0, 3.0]);
        p.prune_layer(&mut layer);
        assert!(layer.weights.iter().all(|&w| w == 0.0));
    }

    #[test]
    fn magnitude_result_fields_correct() {
        let mut p = pruner(PruningStrategy::Magnitude(0.1));
        let mut layer = make_layer("fc1", vec![0.05, -0.2, 0.3]);
        let res = p.prune_layer(&mut layer);
        assert_eq!(res.layer_name, "fc1");
        assert_eq!(res.weights_before, 3);
        assert_eq!(res.weights_pruned, 1);
        assert!(res.sparsity > 0.0 && res.sparsity <= 1.0);
        assert_eq!(res.step, 0);
    }

    // ── Percentile-magnitude pruning ──────────────────────────────────────

    #[test]
    fn percentile_prunes_bottom_fraction() {
        let weights: Vec<f64> = (1..=10).map(|i| i as f64 * 0.1).collect();
        let mut p = pruner(PruningStrategy::PercentileMagnitude(50.0));
        let mut layer = make_layer("l", weights);
        p.prune_layer(&mut layer);
        let sparsity = ModelPruner::compute_sparsity(&layer.weights);
        // Bottom 50 % → roughly 50 % zeros (may be slightly off at boundaries)
        assert!((0.4..=0.6).contains(&sparsity));
    }

    #[test]
    fn percentile_zero_prunes_nothing() {
        let weights = vec![0.1, 0.2, 0.3];
        let mut p = pruner(PruningStrategy::PercentileMagnitude(0.0));
        let mut layer = make_layer("l", weights.clone());
        p.prune_layer(&mut layer);
        // threshold = abs value at 0th percentile = minimum = 0.1, so nothing < 0.1
        // (strictly less than — 0.1 itself is kept)
        assert_eq!(layer.weights, weights);
    }

    #[test]
    fn percentile_hundred_prunes_all_nonzero() {
        let mut p = pruner(PruningStrategy::PercentileMagnitude(100.0));
        let mut layer = make_layer("l", vec![1.0, 2.0, 3.0]);
        p.prune_layer(&mut layer);
        // threshold == max value; all values are < threshold except the maximum
        // (which equals threshold, not strictly less).  So only values strictly
        // below 3.0 are pruned.
        assert_eq!(layer.weights[0], 0.0);
        assert_eq!(layer.weights[1], 0.0);
        // 3.0 == threshold so it is *not* strictly below — kept.
        assert_eq!(layer.weights[2], 3.0);
    }

    // ── Structured L1 pruning ─────────────────────────────────────────────

    #[test]
    fn structured_l1_prunes_weak_neurons() {
        // choose_row_len(9) → sqrt(9)=3, d=3, row_len=9/3=3, num_rows=3
        // So 3 neurons of 3 weights each.
        let mut weights = vec![0.01f64, 0.01, 0.01]; // neuron 0 — weak, mean L1 ≈ 0.01
        weights.extend_from_slice(&[1.0, 2.0, 3.0]); // neuron 1 — strong, mean L1 = 2.0
        weights.extend_from_slice(&[0.5, 0.6, 0.7]); // neuron 2 — strong, mean L1 = 0.6
        let mut p = pruner(PruningStrategy::StructuredL1(0.5));
        let mut layer = make_layer("l", weights);
        p.prune_layer(&mut layer);
        // Neuron 0 mean L1 ≈ 0.01 < 0.5 → pruned
        assert_eq!(layer.weights[0], 0.0);
        assert_eq!(layer.weights[1], 0.0);
        assert_eq!(layer.weights[2], 0.0);
        // Neuron 1 mean L1 = 2.0 > 0.5 → kept
        assert_ne!(layer.weights[3], 0.0);
    }

    #[test]
    fn structured_l1_single_element_falls_back_to_magnitude() {
        let mut p = pruner(PruningStrategy::StructuredL1(0.5));
        let mut layer = make_layer("l", vec![0.1]);
        p.prune_layer(&mut layer);
        // 0.1 < 0.5 → zeroed by magnitude fallback
        assert_eq!(layer.weights[0], 0.0);
    }

    #[test]
    fn structured_l1_no_pruning_when_all_strong() {
        let weights = vec![10.0f64; 9];
        let mut p = pruner(PruningStrategy::StructuredL1(0.1));
        let mut layer = make_layer("l", weights);
        p.prune_layer(&mut layer);
        assert!(layer.weights.iter().all(|&w| w != 0.0));
    }

    // ── Random pruning ────────────────────────────────────────────────────

    #[test]
    fn random_pruning_deterministic_with_seed() {
        let cfg1 = PrunerConfig {
            strategy: PruningStrategy::RandomPruning(0.5),
            seed: 12345,
            update_mask: false,
        };
        let cfg2 = PrunerConfig {
            strategy: PruningStrategy::RandomPruning(0.5),
            seed: 12345,
            update_mask: false,
        };
        let mut p1 = ModelPruner::new(cfg1);
        let mut p2 = ModelPruner::new(cfg2);
        let weights: Vec<f64> = (1..=20).map(|i| i as f64).collect();
        let mut l1 = make_layer("a", weights.clone());
        let mut l2 = make_layer("a", weights);
        p1.prune_layer(&mut l1);
        p2.prune_layer(&mut l2);
        assert_eq!(l1.weights, l2.weights);
    }

    #[test]
    fn random_pruning_different_seeds_differ() {
        let mut p1 = ModelPruner::new(PrunerConfig {
            strategy: PruningStrategy::RandomPruning(0.5),
            seed: 1,
            update_mask: false,
        });
        let mut p2 = ModelPruner::new(PrunerConfig {
            strategy: PruningStrategy::RandomPruning(0.5),
            seed: 999999,
            update_mask: false,
        });
        let weights: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        let mut l1 = make_layer("a", weights.clone());
        let mut l2 = make_layer("a", weights);
        p1.prune_layer(&mut l1);
        p2.prune_layer(&mut l2);
        assert_ne!(l1.weights, l2.weights);
    }

    #[test]
    fn random_pruning_zero_fraction_prunes_nothing() {
        let weights: Vec<f64> = vec![1.0, 2.0, 3.0];
        let mut p = pruner(PruningStrategy::RandomPruning(0.0));
        let mut layer = make_layer("l", weights.clone());
        p.prune_layer(&mut layer);
        assert_eq!(layer.weights, weights);
    }

    // ── Gradual pruning ───────────────────────────────────────────────────

    #[test]
    fn gradual_pruning_interpolates_between_steps() {
        let strategy = PruningStrategy::GradualPruning {
            initial_sparsity: 0.0,
            final_sparsity: 1.0,
            begin_step: 0,
            end_step: 10,
        };
        let mut p = pruner(strategy);
        // Step 0 → target = 0.0
        assert!((p.current_sparsity_target() - 0.0).abs() < 1e-9);
        p.advance_step(); // step 1
        let t1 = p.current_sparsity_target();
        assert!(t1 > 0.0 && t1 < 1.0);
    }

    #[test]
    fn gradual_pruning_clamps_to_final_after_end_step() {
        let strategy = PruningStrategy::GradualPruning {
            initial_sparsity: 0.0,
            final_sparsity: 0.9,
            begin_step: 2,
            end_step: 5,
        };
        let mut p = pruner(strategy);
        for _ in 0..10 {
            p.advance_step();
        }
        assert!((p.current_sparsity_target() - 0.9).abs() < 1e-9);
    }

    #[test]
    fn gradual_pruning_holds_initial_before_begin_step() {
        let strategy = PruningStrategy::GradualPruning {
            initial_sparsity: 0.1,
            final_sparsity: 0.8,
            begin_step: 5,
            end_step: 10,
        };
        let p = pruner(strategy);
        // step = 0 < begin_step = 5 → initial_sparsity
        assert!((p.current_sparsity_target() - 0.1).abs() < 1e-9);
    }

    #[test]
    fn gradual_pruning_midpoint_is_correct() {
        let strategy = PruningStrategy::GradualPruning {
            initial_sparsity: 0.0,
            final_sparsity: 1.0,
            begin_step: 0,
            end_step: 10,
        };
        let mut p = pruner(strategy);
        for _ in 0..5 {
            p.advance_step();
        }
        let target = p.current_sparsity_target();
        assert!((target - 0.5).abs() < 1e-9);
    }

    // ── advance_step ──────────────────────────────────────────────────────

    #[test]
    fn advance_step_increments_counter() {
        let strategy = PruningStrategy::GradualPruning {
            initial_sparsity: 0.0,
            final_sparsity: 1.0,
            begin_step: 0,
            end_step: 100,
        };
        let mut p = pruner(strategy);
        let t0 = p.current_sparsity_target();
        p.advance_step();
        let t1 = p.current_sparsity_target();
        assert!(t1 > t0);
    }

    // ── compute_threshold ─────────────────────────────────────────────────

    #[test]
    fn compute_threshold_median() {
        let weights = vec![-3.0, -2.0, -1.0, 1.0, 2.0, 3.0];
        let t = ModelPruner::compute_threshold(&weights, 50.0);
        // Sorted magnitudes: [1,1,2,2,3,3], median index = 3 → 2.0
        assert!((t - 2.0).abs() < 1e-9);
    }

    #[test]
    fn compute_threshold_zero_percentile() {
        let weights = vec![1.0, 2.0, 3.0];
        let t = ModelPruner::compute_threshold(&weights, 0.0);
        assert!((t - 1.0).abs() < 1e-9);
    }

    #[test]
    fn compute_threshold_hundred_percentile() {
        let weights = vec![1.0, 2.0, 3.0];
        let t = ModelPruner::compute_threshold(&weights, 100.0);
        assert!((t - 3.0).abs() < 1e-9);
    }

    #[test]
    fn compute_threshold_empty() {
        assert_eq!(ModelPruner::compute_threshold(&[], 50.0), 0.0);
    }

    // ── compute_sparsity ──────────────────────────────────────────────────

    #[test]
    fn compute_sparsity_all_nonzero() {
        assert_eq!(ModelPruner::compute_sparsity(&[1.0, 2.0, 3.0]), 0.0);
    }

    #[test]
    fn compute_sparsity_all_zero() {
        assert_eq!(ModelPruner::compute_sparsity(&[0.0, 0.0, 0.0]), 1.0);
    }

    #[test]
    fn compute_sparsity_half() {
        assert!((ModelPruner::compute_sparsity(&[0.0, 1.0]) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn compute_sparsity_empty() {
        assert_eq!(ModelPruner::compute_sparsity(&[]), 0.0);
    }

    // ── apply_mask ────────────────────────────────────────────────────────

    #[test]
    fn apply_mask_zeros_false_entries() {
        let mut layer = LayerWeights {
            name: "l".to_string(),
            weights: vec![1.0, 2.0, 3.0],
            mask: Some(vec![true, false, true]),
        };
        ModelPruner::apply_mask(&mut layer);
        assert_eq!(layer.weights, vec![1.0, 0.0, 3.0]);
    }

    #[test]
    fn apply_mask_no_mask_noop() {
        let mut layer = LayerWeights {
            name: "l".to_string(),
            weights: vec![1.0, 2.0, 3.0],
            mask: None,
        };
        ModelPruner::apply_mask(&mut layer);
        assert_eq!(layer.weights, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn apply_mask_all_false_zeroes_all() {
        let mut layer = LayerWeights {
            name: "l".to_string(),
            weights: vec![5.0, 6.0, 7.0],
            mask: Some(vec![false, false, false]),
        };
        ModelPruner::apply_mask(&mut layer);
        assert!(layer.weights.iter().all(|&w| w == 0.0));
    }

    // ── prune_all ─────────────────────────────────────────────────────────

    #[test]
    fn prune_all_returns_one_result_per_layer() {
        let mut p = pruner(PruningStrategy::Magnitude(0.1));
        let mut layers = vec![
            make_layer("a", vec![0.05, 0.5]),
            make_layer("b", vec![0.05, 0.5, -0.5]),
            make_layer("c", vec![1.0, 2.0]),
        ];
        let results = p.prune_all(&mut layers);
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].layer_name, "a");
        assert_eq!(results[1].layer_name, "b");
        assert_eq!(results[2].layer_name, "c");
    }

    #[test]
    fn prune_all_mutates_all_layers() {
        let mut p = pruner(PruningStrategy::Magnitude(1e9));
        let mut layers = vec![
            make_layer("a", vec![0.1, 0.2]),
            make_layer("b", vec![0.3, 0.4]),
        ];
        p.prune_all(&mut layers);
        for layer in &layers {
            assert!(layer.weights.iter().all(|&w| w == 0.0));
        }
    }

    // ── Mask update ───────────────────────────────────────────────────────

    #[test]
    fn mask_updated_after_pruning() {
        let mut p = pruner(PruningStrategy::Magnitude(0.5));
        let mut layer = make_layer("l", vec![0.1, 1.0, 0.2, 2.0]);
        p.prune_layer(&mut layer);
        let mask = layer.mask.expect("test: should succeed");
        // 0.1 and 0.2 are pruned → false
        assert!(!mask[0]);
        assert!(mask[1]);
        assert!(!mask[2]);
        assert!(mask[3]);
    }

    #[test]
    fn no_mask_update_when_disabled() {
        let cfg = PrunerConfig {
            strategy: PruningStrategy::Magnitude(0.5),
            seed: 0,
            update_mask: false,
        };
        let mut p = ModelPruner::new(cfg);
        let mut layer = make_layer("l", vec![0.1, 1.0]);
        p.prune_layer(&mut layer);
        assert!(layer.mask.is_none());
    }

    // ── Stats tracking ────────────────────────────────────────────────────

    #[test]
    fn stats_total_pruning_steps_increments() {
        let mut p = pruner(PruningStrategy::Magnitude(0.5));
        assert_eq!(p.stats().total_pruning_steps, 0);
        p.prune_layer(&mut make_layer("a", vec![0.1, 1.0]));
        assert_eq!(p.stats().total_pruning_steps, 1);
        p.prune_layer(&mut make_layer("b", vec![0.1, 1.0]));
        assert_eq!(p.stats().total_pruning_steps, 2);
    }

    #[test]
    fn stats_total_weights_pruned_accumulates() {
        let mut p = pruner(PruningStrategy::Magnitude(0.5));
        p.prune_layer(&mut make_layer("a", vec![0.1, 0.2, 1.0])); // 2 pruned
        p.prune_layer(&mut make_layer("b", vec![0.3, 0.4, 2.0])); // 2 pruned
        assert_eq!(p.stats().total_weights_pruned, 4);
    }

    #[test]
    fn stats_avg_sparsity_is_non_negative() {
        let mut p = pruner(PruningStrategy::Magnitude(0.5));
        p.prune_layer(&mut make_layer("a", vec![0.1, 1.0]));
        assert!(p.stats().avg_sparsity >= 0.0);
        assert!(p.stats().avg_sparsity <= 1.0);
    }

    // ── Edge cases ────────────────────────────────────────────────────────

    #[test]
    fn full_zero_weights_remain_zero() {
        let mut p = pruner(PruningStrategy::Magnitude(0.1));
        let mut layer = make_layer("l", vec![0.0, 0.0, 0.0]);
        let result = p.prune_layer(&mut layer);
        assert_eq!(result.sparsity, 1.0);
        assert_eq!(result.weights_pruned, 0); // already zero, nothing *newly* pruned
    }

    #[test]
    fn empty_layer_produces_valid_result() {
        let mut p = pruner(PruningStrategy::Magnitude(0.1));
        let mut layer = make_layer("empty", vec![]);
        let result = p.prune_layer(&mut layer);
        assert_eq!(result.weights_before, 0);
        assert_eq!(result.weights_pruned, 0);
        assert_eq!(result.sparsity, 0.0);
    }

    #[test]
    fn compute_l1_norm_sum_of_abs() {
        let weights = vec![-1.0, 2.0, -3.0, 4.0];
        assert!((ModelPruner::compute_l1_norm(&weights) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn next_uniform_prng_in_range() {
        let cfg = PrunerConfig {
            strategy: PruningStrategy::Magnitude(0.0),
            seed: 7,
            update_mask: false,
        };
        let mut p = ModelPruner::new(cfg);
        for _ in 0..1000 {
            let v = p.next_uniform_prng();
            assert!((0.0..1.0).contains(&v), "PRNG out of range: {}", v);
        }
    }
}
