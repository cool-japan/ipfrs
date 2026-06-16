//! Gradient accumulation across mini-batches before optimizer step.
//!
//! [`TensorGradAccumulator`] buffers gradient contributions over a configurable
//! number of accumulation steps and produces the final (optionally clipped)
//! gradient when the accumulation window is complete.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// AccumulationMode
// ---------------------------------------------------------------------------

/// How accumulated gradients are combined when the accumulation window closes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccumulationMode {
    /// Accumulate by summing — the result is the raw sum of all contributions.
    Sum,
    /// Accumulate by averaging — the result is the sum divided by the number
    /// of accumulation steps.
    Mean,
}

// ---------------------------------------------------------------------------
// AccumulatorConfig
// ---------------------------------------------------------------------------

/// Configuration for a [`TensorGradAccumulator`].
#[derive(Debug, Clone)]
pub struct AccumulatorConfig {
    /// Number of mini-batch steps to accumulate before an optimizer step.
    pub accumulation_steps: usize,
    /// Combination mode applied at the end of the accumulation window.
    pub mode: AccumulationMode,
    /// If set, gradient vectors whose L2 norm exceeds this value are
    /// rescaled so their norm equals `max_grad_norm`.
    pub max_grad_norm: Option<f64>,
}

impl Default for AccumulatorConfig {
    fn default() -> Self {
        Self {
            accumulation_steps: 4,
            mode: AccumulationMode::Sum,
            max_grad_norm: None,
        }
    }
}

// ---------------------------------------------------------------------------
// GradBuffer
// ---------------------------------------------------------------------------

/// Per-parameter gradient buffer that accumulates contributions across steps.
#[derive(Debug, Clone)]
pub struct GradBuffer {
    /// Name of the parameter this buffer belongs to.
    pub name: String,
    /// Running sum of gradient values.
    pub values: Vec<f64>,
    /// Number of times gradients have been added to this buffer since the
    /// last reset.
    pub steps_accumulated: usize,
}

// ---------------------------------------------------------------------------
// AccumulatorStats
// ---------------------------------------------------------------------------

/// Summary statistics for a [`TensorGradAccumulator`].
#[derive(Debug, Clone)]
pub struct AccumulatorStats {
    /// Number of parameter buffers currently tracked.
    pub buffer_count: usize,
    /// Current accumulation step (0-based within the current window).
    pub current_step: usize,
    /// Configured number of accumulation steps per window.
    pub accumulation_steps: usize,
    /// Total number of successful [`TensorGradAccumulator::step`] calls
    /// over the lifetime of the accumulator.
    pub total_accumulations: u64,
    /// Total number of gradient vectors that were clipped during
    /// [`TensorGradAccumulator::step`].
    pub total_clips: u64,
}

// ---------------------------------------------------------------------------
// TensorGradAccumulator
// ---------------------------------------------------------------------------

/// Accumulates gradients across mini-batches before an optimizer step.
///
/// # Typical usage
///
/// ```
/// use ipfrs_tensorlogic::grad_accumulator::{
///     TensorGradAccumulator, AccumulatorConfig, AccumulationMode,
/// };
///
/// let config = AccumulatorConfig {
///     accumulation_steps: 2,
///     mode: AccumulationMode::Mean,
///     max_grad_norm: Some(1.0),
/// };
/// let mut acc = TensorGradAccumulator::new(config);
///
/// // First mini-batch
/// acc.accumulate("weight", &[0.5, -0.3]).expect("example: should succeed in docs");
/// assert!(!acc.is_ready());
///
/// // Second mini-batch
/// acc.accumulate("weight", &[0.7, 0.1]).expect("example: should succeed in docs");
/// assert!(acc.is_ready());
///
/// // Retrieve accumulated gradients and reset
/// let grads = acc.step().expect("example: should succeed in docs");
/// assert!(grads.contains_key("weight"));
/// ```
pub struct TensorGradAccumulator {
    config: AccumulatorConfig,
    buffers: HashMap<String, GradBuffer>,
    current_step: usize,
    total_accumulations: u64,
    total_clips: u64,
}

impl TensorGradAccumulator {
    /// Create a new accumulator with the given configuration.
    pub fn new(config: AccumulatorConfig) -> Self {
        Self {
            config,
            buffers: HashMap::new(),
            current_step: 0,
            total_accumulations: 0,
            total_clips: 0,
        }
    }

    /// Add gradients for the named parameter to the accumulation buffer.
    ///
    /// If the buffer for `name` already exists its length must match
    /// `gradients.len()`, otherwise an error is returned.  On the first
    /// call for a given name the buffer is created with the supplied length.
    ///
    /// After all parameter gradients for a mini-batch have been added the
    /// caller should increment the internal step counter by calling
    /// [`accumulate`](Self::accumulate) for every parameter in each
    /// mini-batch.
    pub fn accumulate(&mut self, name: &str, gradients: &[f64]) -> Result<(), String> {
        if let Some(buf) = self.buffers.get_mut(name) {
            if buf.values.len() != gradients.len() {
                return Err(format!(
                    "gradient size mismatch for '{}': expected {}, got {}",
                    name,
                    buf.values.len(),
                    gradients.len()
                ));
            }
            for (dst, src) in buf.values.iter_mut().zip(gradients.iter()) {
                *dst += *src;
            }
            buf.steps_accumulated += 1;
        } else {
            self.buffers.insert(
                name.to_string(),
                GradBuffer {
                    name: name.to_string(),
                    values: gradients.to_vec(),
                    steps_accumulated: 1,
                },
            );
        }
        // Track how many mini-batch steps we have seen.  We use the
        // maximum `steps_accumulated` across all buffers as the canonical
        // step count — this handles the common case where every parameter
        // is accumulated once per mini-batch.
        self.current_step = self
            .buffers
            .values()
            .map(|b| b.steps_accumulated)
            .max()
            .unwrap_or(0);
        Ok(())
    }

    /// Returns `true` when every buffer has accumulated at least
    /// `accumulation_steps` contributions.
    pub fn is_ready(&self) -> bool {
        if self.buffers.is_empty() {
            return false;
        }
        self.buffers
            .values()
            .all(|b| b.steps_accumulated >= self.config.accumulation_steps)
    }

    /// Consume the accumulated gradients and return the final values.
    ///
    /// - In [`AccumulationMode::Mean`] mode each gradient vector is divided
    ///   by the number of steps that were accumulated.
    /// - If `max_grad_norm` is configured, gradient vectors whose L2 norm
    ///   exceeds it are rescaled.
    /// - All buffers are cleared after a successful step.
    ///
    /// Returns an error if the accumulator is not ready (see
    /// [`is_ready`](Self::is_ready)).
    pub fn step(&mut self) -> Result<HashMap<String, Vec<f64>>, String> {
        if !self.is_ready() {
            return Err("accumulator is not ready: not all buffers have enough steps".to_string());
        }

        let mut result = HashMap::new();

        for (name, buf) in &self.buffers {
            let mut grad = buf.values.clone();

            // Apply mean scaling if configured.
            if self.config.mode == AccumulationMode::Mean && buf.steps_accumulated > 0 {
                let scale = 1.0 / buf.steps_accumulated as f64;
                for v in &mut grad {
                    *v *= scale;
                }
            }

            // Apply gradient clipping if configured.
            if let Some(max_norm) = self.config.max_grad_norm {
                let original_norm = Self::clip_grad_norm(&mut grad, max_norm);
                if original_norm > max_norm {
                    self.total_clips += 1;
                }
            }

            result.insert(name.clone(), grad);
        }

        self.total_accumulations += 1;
        self.buffers.clear();
        self.current_step = 0;

        Ok(result)
    }

    /// Clip a gradient vector in-place so that its L2 norm does not exceed
    /// `max_norm`.
    ///
    /// Returns the **original** L2 norm (before any scaling).
    pub fn clip_grad_norm(gradients: &mut [f64], max_norm: f64) -> f64 {
        let norm_sq: f64 = gradients.iter().map(|x| x * x).sum();
        let norm = norm_sq.sqrt();
        if norm > max_norm && norm > 0.0 {
            let scale = max_norm / norm;
            for v in gradients.iter_mut() {
                *v *= scale;
            }
        }
        norm
    }

    /// Borrow the buffer for the named parameter, if it exists.
    pub fn get_buffer(&self, name: &str) -> Option<&GradBuffer> {
        self.buffers.get(name)
    }

    /// Number of parameter buffers currently tracked.
    pub fn buffer_count(&self) -> usize {
        self.buffers.len()
    }

    /// Clear all buffers and reset the step counter.  Lifetime statistics
    /// (`total_accumulations`, `total_clips`) are preserved.
    pub fn reset(&mut self) {
        self.buffers.clear();
        self.current_step = 0;
    }

    /// Current accumulation step (0-based within the current window).
    pub fn current_step(&self) -> usize {
        self.current_step
    }

    /// Return a snapshot of the accumulator's statistics.
    pub fn stats(&self) -> AccumulatorStats {
        AccumulatorStats {
            buffer_count: self.buffers.len(),
            current_step: self.current_step,
            accumulation_steps: self.config.accumulation_steps,
            total_accumulations: self.total_accumulations,
            total_clips: self.total_clips,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> AccumulatorConfig {
        AccumulatorConfig::default()
    }

    fn sum_config(steps: usize) -> AccumulatorConfig {
        AccumulatorConfig {
            accumulation_steps: steps,
            mode: AccumulationMode::Sum,
            max_grad_norm: None,
        }
    }

    fn mean_config(steps: usize) -> AccumulatorConfig {
        AccumulatorConfig {
            accumulation_steps: steps,
            mode: AccumulationMode::Mean,
            max_grad_norm: None,
        }
    }

    // -----------------------------------------------------------------------
    // 1. Basic construction
    // -----------------------------------------------------------------------

    #[test]
    fn test_new_default_config() {
        let cfg = default_config();
        assert_eq!(cfg.accumulation_steps, 4);
        assert_eq!(cfg.mode, AccumulationMode::Sum);
        assert!(cfg.max_grad_norm.is_none());
    }

    #[test]
    fn test_new_accumulator_empty() {
        let acc = TensorGradAccumulator::new(default_config());
        assert_eq!(acc.buffer_count(), 0);
        assert_eq!(acc.current_step(), 0);
        assert!(!acc.is_ready());
    }

    // -----------------------------------------------------------------------
    // 2. Accumulate — Sum mode
    // -----------------------------------------------------------------------

    #[test]
    fn test_accumulate_sum_single_step() {
        let mut acc = TensorGradAccumulator::new(sum_config(1));
        acc.accumulate("w", &[1.0, 2.0, 3.0]).ok();
        assert!(acc.is_ready());
        let grads = acc.step().expect("step should succeed");
        let w = &grads["w"];
        assert!((w[0] - 1.0).abs() < 1e-12);
        assert!((w[1] - 2.0).abs() < 1e-12);
        assert!((w[2] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_accumulate_sum_two_steps() {
        let mut acc = TensorGradAccumulator::new(sum_config(2));
        acc.accumulate("w", &[1.0, 2.0]).ok();
        assert!(!acc.is_ready());
        acc.accumulate("w", &[3.0, 4.0]).ok();
        assert!(acc.is_ready());
        let grads = acc.step().expect("step should succeed");
        let w = &grads["w"];
        assert!((w[0] - 4.0).abs() < 1e-12);
        assert!((w[1] - 6.0).abs() < 1e-12);
    }

    #[test]
    fn test_accumulate_sum_four_steps() {
        let mut acc = TensorGradAccumulator::new(sum_config(4));
        for i in 0..4 {
            acc.accumulate("w", &[i as f64]).ok();
        }
        assert!(acc.is_ready());
        let grads = acc.step().expect("step should succeed");
        // 0 + 1 + 2 + 3 = 6
        assert!((grads["w"][0] - 6.0).abs() < 1e-12);
    }

    // -----------------------------------------------------------------------
    // 3. Accumulate — Mean mode
    // -----------------------------------------------------------------------

    #[test]
    fn test_accumulate_mean_single_step() {
        let mut acc = TensorGradAccumulator::new(mean_config(1));
        acc.accumulate("w", &[4.0]).ok();
        let grads = acc.step().expect("step");
        assert!((grads["w"][0] - 4.0).abs() < 1e-12);
    }

    #[test]
    fn test_accumulate_mean_two_steps() {
        let mut acc = TensorGradAccumulator::new(mean_config(2));
        acc.accumulate("w", &[2.0, 6.0]).ok();
        acc.accumulate("w", &[4.0, 8.0]).ok();
        let grads = acc.step().expect("step");
        // mean of (2+4)/2 = 3, (6+8)/2 = 7
        assert!((grads["w"][0] - 3.0).abs() < 1e-12);
        assert!((grads["w"][1] - 7.0).abs() < 1e-12);
    }

    #[test]
    fn test_accumulate_mean_four_steps() {
        let mut acc = TensorGradAccumulator::new(mean_config(4));
        for _ in 0..4 {
            acc.accumulate("w", &[8.0]).ok();
        }
        let grads = acc.step().expect("step");
        // sum=32, mean=32/4=8
        assert!((grads["w"][0] - 8.0).abs() < 1e-12);
    }

    // -----------------------------------------------------------------------
    // 4. Gradient clipping
    // -----------------------------------------------------------------------

    #[test]
    fn test_clip_grad_norm_scales_down() {
        let mut g = vec![3.0, 4.0]; // norm = 5
        let original = TensorGradAccumulator::clip_grad_norm(&mut g, 1.0);
        assert!((original - 5.0).abs() < 1e-12);
        let clipped_norm: f64 = g.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((clipped_norm - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_clip_grad_norm_no_change_when_within() {
        let mut g = vec![0.3, 0.4]; // norm = 0.5
        let original = TensorGradAccumulator::clip_grad_norm(&mut g, 1.0);
        assert!((original - 0.5).abs() < 1e-12);
        assert!((g[0] - 0.3).abs() < 1e-12);
        assert!((g[1] - 0.4).abs() < 1e-12);
    }

    #[test]
    fn test_clip_grad_norm_exact_boundary() {
        let mut g = vec![3.0, 4.0]; // norm = 5
        let original = TensorGradAccumulator::clip_grad_norm(&mut g, 5.0);
        assert!((original - 5.0).abs() < 1e-12);
        assert!((g[0] - 3.0).abs() < 1e-12);
        assert!((g[1] - 4.0).abs() < 1e-12);
    }

    #[test]
    fn test_clip_grad_norm_zero_vector() {
        let mut g = vec![0.0, 0.0];
        let original = TensorGradAccumulator::clip_grad_norm(&mut g, 1.0);
        assert!((original).abs() < 1e-12);
        assert!((g[0]).abs() < 1e-12);
        assert!((g[1]).abs() < 1e-12);
    }

    #[test]
    fn test_step_with_clipping() {
        let config = AccumulatorConfig {
            accumulation_steps: 1,
            mode: AccumulationMode::Sum,
            max_grad_norm: Some(1.0),
        };
        let mut acc = TensorGradAccumulator::new(config);
        acc.accumulate("w", &[3.0, 4.0]).ok(); // norm=5, will be clipped
        let grads = acc.step().expect("step");
        let clipped_norm: f64 = grads["w"].iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((clipped_norm - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_step_clips_increments_total_clips() {
        let config = AccumulatorConfig {
            accumulation_steps: 1,
            mode: AccumulationMode::Sum,
            max_grad_norm: Some(1.0),
        };
        let mut acc = TensorGradAccumulator::new(config);
        acc.accumulate("w", &[3.0, 4.0]).ok(); // will clip
        acc.step().ok();
        assert_eq!(acc.stats().total_clips, 1);
    }

    #[test]
    fn test_step_no_clip_no_increment() {
        let config = AccumulatorConfig {
            accumulation_steps: 1,
            mode: AccumulationMode::Sum,
            max_grad_norm: Some(10.0),
        };
        let mut acc = TensorGradAccumulator::new(config);
        acc.accumulate("w", &[0.1, 0.2]).ok(); // norm~0.22, well within
        acc.step().ok();
        assert_eq!(acc.stats().total_clips, 0);
    }

    // -----------------------------------------------------------------------
    // 5. is_ready logic
    // -----------------------------------------------------------------------

    #[test]
    fn test_is_ready_empty() {
        let acc = TensorGradAccumulator::new(sum_config(2));
        assert!(!acc.is_ready());
    }

    #[test]
    fn test_is_ready_partial() {
        let mut acc = TensorGradAccumulator::new(sum_config(3));
        acc.accumulate("w", &[1.0]).ok();
        acc.accumulate("w", &[2.0]).ok();
        assert!(!acc.is_ready());
    }

    #[test]
    fn test_is_ready_exact() {
        let mut acc = TensorGradAccumulator::new(sum_config(2));
        acc.accumulate("w", &[1.0]).ok();
        acc.accumulate("w", &[2.0]).ok();
        assert!(acc.is_ready());
    }

    #[test]
    fn test_is_ready_multi_param_partial() {
        let mut acc = TensorGradAccumulator::new(sum_config(2));
        acc.accumulate("w", &[1.0]).ok();
        acc.accumulate("w", &[2.0]).ok();
        acc.accumulate("b", &[0.5]).ok(); // only 1 step for "b"
        assert!(!acc.is_ready());
    }

    #[test]
    fn test_is_ready_multi_param_all_ready() {
        let mut acc = TensorGradAccumulator::new(sum_config(2));
        acc.accumulate("w", &[1.0]).ok();
        acc.accumulate("b", &[0.5]).ok();
        acc.accumulate("w", &[2.0]).ok();
        acc.accumulate("b", &[0.7]).ok();
        assert!(acc.is_ready());
    }

    // -----------------------------------------------------------------------
    // 6. step() returns correct values
    // -----------------------------------------------------------------------

    #[test]
    fn test_step_returns_all_params() {
        let mut acc = TensorGradAccumulator::new(sum_config(1));
        acc.accumulate("w", &[1.0, 2.0]).ok();
        acc.accumulate("b", &[0.5]).ok();
        let grads = acc.step().expect("step");
        assert_eq!(grads.len(), 2);
        assert!(grads.contains_key("w"));
        assert!(grads.contains_key("b"));
    }

    #[test]
    fn test_step_clears_buffers() {
        let mut acc = TensorGradAccumulator::new(sum_config(1));
        acc.accumulate("w", &[1.0]).ok();
        acc.step().ok();
        assert_eq!(acc.buffer_count(), 0);
        assert_eq!(acc.current_step(), 0);
    }

    #[test]
    fn test_step_error_when_not_ready() {
        let mut acc = TensorGradAccumulator::new(sum_config(3));
        acc.accumulate("w", &[1.0]).ok();
        let err = acc.step().expect_err("should fail");
        assert!(err.contains("not ready"));
    }

    // -----------------------------------------------------------------------
    // 7. Size mismatch error
    // -----------------------------------------------------------------------

    #[test]
    fn test_accumulate_size_mismatch() {
        let mut acc = TensorGradAccumulator::new(sum_config(2));
        acc.accumulate("w", &[1.0, 2.0]).ok();
        let err = acc
            .accumulate("w", &[1.0, 2.0, 3.0])
            .expect_err("should fail");
        assert!(err.contains("size mismatch"));
    }

    #[test]
    fn test_accumulate_size_mismatch_shorter() {
        let mut acc = TensorGradAccumulator::new(sum_config(2));
        acc.accumulate("w", &[1.0, 2.0, 3.0]).ok();
        let err = acc.accumulate("w", &[1.0]).expect_err("should fail");
        assert!(err.contains("size mismatch"));
    }

    // -----------------------------------------------------------------------
    // 8. Reset
    // -----------------------------------------------------------------------

    #[test]
    fn test_reset_clears_buffers() {
        let mut acc = TensorGradAccumulator::new(sum_config(2));
        acc.accumulate("w", &[1.0]).ok();
        acc.accumulate("b", &[2.0]).ok();
        acc.reset();
        assert_eq!(acc.buffer_count(), 0);
        assert_eq!(acc.current_step(), 0);
        assert!(!acc.is_ready());
    }

    #[test]
    fn test_reset_preserves_lifetime_stats() {
        let mut acc = TensorGradAccumulator::new(sum_config(1));
        acc.accumulate("w", &[1.0]).ok();
        acc.step().ok();
        let accums_before = acc.stats().total_accumulations;
        acc.reset();
        assert_eq!(acc.stats().total_accumulations, accums_before);
    }

    // -----------------------------------------------------------------------
    // 9. Multiple parameters
    // -----------------------------------------------------------------------

    #[test]
    fn test_multiple_params_independent() {
        let mut acc = TensorGradAccumulator::new(mean_config(2));
        acc.accumulate("w1", &[2.0, 4.0]).ok();
        acc.accumulate("w2", &[10.0]).ok();
        acc.accumulate("w1", &[6.0, 8.0]).ok();
        acc.accumulate("w2", &[20.0]).ok();

        let grads = acc.step().expect("step");
        // w1: mean of (2+6)/2=4, (4+8)/2=6
        assert!((grads["w1"][0] - 4.0).abs() < 1e-12);
        assert!((grads["w1"][1] - 6.0).abs() < 1e-12);
        // w2: mean of (10+20)/2=15
        assert!((grads["w2"][0] - 15.0).abs() < 1e-12);
    }

    // -----------------------------------------------------------------------
    // 10. get_buffer
    // -----------------------------------------------------------------------

    #[test]
    fn test_get_buffer_existing() {
        let mut acc = TensorGradAccumulator::new(sum_config(2));
        acc.accumulate("w", &[1.0, 2.0]).ok();
        let buf = acc.get_buffer("w").expect("should exist");
        assert_eq!(buf.name, "w");
        assert_eq!(buf.values.len(), 2);
        assert_eq!(buf.steps_accumulated, 1);
    }

    #[test]
    fn test_get_buffer_missing() {
        let acc = TensorGradAccumulator::new(sum_config(2));
        assert!(acc.get_buffer("nonexistent").is_none());
    }

    // -----------------------------------------------------------------------
    // 11. Stats tracking
    // -----------------------------------------------------------------------

    #[test]
    fn test_stats_initial() {
        let acc = TensorGradAccumulator::new(sum_config(4));
        let s = acc.stats();
        assert_eq!(s.buffer_count, 0);
        assert_eq!(s.current_step, 0);
        assert_eq!(s.accumulation_steps, 4);
        assert_eq!(s.total_accumulations, 0);
        assert_eq!(s.total_clips, 0);
    }

    #[test]
    fn test_stats_after_step() {
        let mut acc = TensorGradAccumulator::new(sum_config(1));
        acc.accumulate("w", &[1.0]).ok();
        acc.step().ok();
        let s = acc.stats();
        assert_eq!(s.total_accumulations, 1);
        assert_eq!(s.buffer_count, 0); // cleared after step
    }

    #[test]
    fn test_stats_multiple_steps() {
        let mut acc = TensorGradAccumulator::new(sum_config(1));
        for _ in 0..5 {
            acc.accumulate("w", &[1.0]).ok();
            acc.step().ok();
        }
        assert_eq!(acc.stats().total_accumulations, 5);
    }

    // -----------------------------------------------------------------------
    // 12. Empty accumulator edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_step_empty_accumulator() {
        let mut acc = TensorGradAccumulator::new(sum_config(1));
        assert!(acc.step().is_err());
    }

    #[test]
    fn test_buffer_count_empty() {
        let acc = TensorGradAccumulator::new(sum_config(1));
        assert_eq!(acc.buffer_count(), 0);
    }

    #[test]
    fn test_current_step_tracks_max() {
        let mut acc = TensorGradAccumulator::new(sum_config(3));
        acc.accumulate("w", &[1.0]).ok();
        assert_eq!(acc.current_step(), 1);
        acc.accumulate("w", &[2.0]).ok();
        assert_eq!(acc.current_step(), 2);
    }

    // -----------------------------------------------------------------------
    // 13. Direction preservation after clipping
    // -----------------------------------------------------------------------

    #[test]
    fn test_clip_preserves_direction() {
        let mut g = vec![6.0, 8.0]; // norm = 10, direction = (0.6, 0.8)
        TensorGradAccumulator::clip_grad_norm(&mut g, 5.0);
        // after clip: (3.0, 4.0) — direction preserved
        assert!((g[0] - 3.0).abs() < 1e-10);
        assert!((g[1] - 4.0).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // 14. Accumulate then reset then re-use
    // -----------------------------------------------------------------------

    #[test]
    fn test_reset_then_reuse() {
        let mut acc = TensorGradAccumulator::new(sum_config(1));
        acc.accumulate("w", &[1.0]).ok();
        acc.step().ok();
        acc.reset();
        // Should be able to accumulate again with different size
        acc.accumulate("w", &[1.0, 2.0, 3.0]).ok();
        let grads = acc.step().expect("step");
        assert_eq!(grads["w"].len(), 3);
    }

    // -----------------------------------------------------------------------
    // 15. Mean mode with clipping
    // -----------------------------------------------------------------------

    #[test]
    fn test_mean_with_clipping() {
        let config = AccumulatorConfig {
            accumulation_steps: 2,
            mode: AccumulationMode::Mean,
            max_grad_norm: Some(1.0),
        };
        let mut acc = TensorGradAccumulator::new(config);
        acc.accumulate("w", &[6.0, 8.0]).ok(); // sum will be 12, 16
        acc.accumulate("w", &[6.0, 8.0]).ok();
        let grads = acc.step().expect("step");
        // mean: (12/2, 16/2) = (6, 8), norm = 10, clipped to 1.0
        let clipped_norm: f64 = grads["w"].iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((clipped_norm - 1.0).abs() < 1e-10);
        assert_eq!(acc.stats().total_clips, 1);
    }
}
