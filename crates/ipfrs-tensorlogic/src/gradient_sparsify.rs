//! Gradient sparsification and delta encoding for federated learning.
//!
//! This module provides bandwidth-efficient gradient transmission primitives:
//!
//! - [`SparsityConfig`] — policy for top-k selection and threshold filtering
//! - [`SparseGradient`] — compact index/value representation with residual support
//! - [`GradientSparsifier`] — stateful sparsifier with residual accumulation
//! - [`GradientDelta`] — delta-encoded gradient relative to the previous round
//! - [`DeltaEncoder`] — stateful encoder that tracks the previously sent gradient
//!
//! ## Design rationale
//!
//! In bandwidth-constrained federated learning scenarios, transmitting the full
//! gradient vector each round wastes network capacity. Two complementary
//! techniques address this:
//!
//! 1. **Sparsification** — keep only the top-k elements (by absolute value) or
//!    those exceeding a magnitude threshold, accumulating the dropped portion in
//!    a residual buffer so that no information is permanently lost.
//!
//! 2. **Delta encoding** — transmit the element-wise difference from the
//!    previous round instead of the full gradient; the receiver reconstructs
//!    the current gradient by adding the delta to its locally cached copy.

use serde::{Deserialize, Serialize};

// ── SparsityConfig ──────────────────────────────────────────────────────────

/// Configuration for the [`GradientSparsifier`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparsityConfig {
    /// Keep only the top-k elements by absolute value. `None` means no limit.
    pub top_k: Option<usize>,
    /// Drop elements whose absolute value is below this threshold. `None` means
    /// no threshold filtering.
    pub threshold: Option<f32>,
    /// When `true` (default), dropped elements are accumulated into a residual
    /// buffer and added back to the gradient on the next call to [`GradientSparsifier::sparsify`].
    pub accumulate_residuals: bool,
}

impl Default for SparsityConfig {
    fn default() -> Self {
        Self {
            top_k: None,
            threshold: None,
            accumulate_residuals: true,
        }
    }
}

// ── SparsifierStats ─────────────────────────────────────────────────────────

/// Cumulative statistics for a [`GradientSparsifier`].
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SparsifierStats {
    /// Total number of sparsification rounds completed.
    pub total_rounds: u64,
    /// Total number of gradient elements that were kept (sent).
    pub total_elements_kept: u64,
    /// Total number of gradient elements that were dropped (deferred to residual).
    pub total_elements_dropped: u64,
    /// Total number of residual elements that were re-applied to the gradient.
    pub total_residual_applied: u64,
}

// ── SparseGradient (sparsify module variant) ────────────────────────────────

/// A compact sparse representation of a gradient vector.
///
/// Indices use `u32` to halve storage compared with `usize` on 64-bit targets,
/// which is safe for gradient lengths that fit within 4 billion elements.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseGradient {
    /// Positions of kept (non-zero) elements in the original flat gradient.
    pub indices: Vec<u32>,
    /// Values corresponding to each kept index.
    pub values: Vec<f32>,
    /// Length of the original flat gradient from which this was derived.
    pub original_len: usize,
}

impl SparseGradient {
    /// Fraction of elements that were *not* kept.
    ///
    /// A value of 1.0 means all elements were dropped; 0.0 means all were kept.
    pub fn sparsity_ratio(&self) -> f64 {
        if self.original_len == 0 {
            return 0.0;
        }
        1.0 - (self.indices.len() as f64 / self.original_len as f64)
    }

    /// Reconstruct the full dense gradient vector (zeros at dropped positions).
    pub fn to_dense(&self) -> Vec<f32> {
        let mut dense = vec![0.0_f32; self.original_len];
        for (&idx, &val) in self.indices.iter().zip(self.values.iter()) {
            let pos = idx as usize;
            if pos < self.original_len {
                dense[pos] = val;
            }
        }
        dense
    }
}

// ── GradientSparsifier ──────────────────────────────────────────────────────

/// Stateful gradient sparsifier with optional residual accumulation.
///
/// Residual accumulation ensures that gradient information dropped in one round
/// is carried forward and injected in subsequent rounds, preventing systematic
/// bias toward always-large parameters.
pub struct GradientSparsifier {
    /// Sparsification policy.
    pub config: SparsityConfig,
    /// Accumulated residual from previously dropped gradient elements.
    pub residual: Vec<f32>,
    /// Cumulative statistics.
    pub stats: SparsifierStats,
}

impl GradientSparsifier {
    /// Create a new sparsifier.
    ///
    /// `gradient_len` is used to pre-allocate the residual buffer. If the
    /// gradient length changes between calls, the residual is silently
    /// zero-extended or truncated to match.
    pub fn new(config: SparsityConfig, gradient_len: usize) -> Self {
        Self {
            config,
            residual: vec![0.0_f32; gradient_len],
            stats: SparsifierStats::default(),
        }
    }

    /// Sparsify a gradient vector.
    ///
    /// Steps performed:
    /// 1. If `accumulate_residuals`, add the stored residual to `gradient`
    ///    element-wise (extending or truncating the residual as needed).
    /// 2. Apply top-k and/or threshold selection.
    /// 3. Update the residual with the dropped elements.
    /// 4. Update statistics.
    ///
    /// Returns a [`SparseGradient`] containing only the kept elements.
    pub fn sparsify(&mut self, gradient: &[f32]) -> SparseGradient {
        let len = gradient.len();

        // Ensure residual buffer matches current gradient length.
        if self.residual.len() != len {
            self.residual.resize(len, 0.0);
        }

        // Step 1: build the working vector (gradient + residual).
        let mut working: Vec<f32> = if self.config.accumulate_residuals {
            let residual_applied = self.residual.iter().filter(|&&v| v != 0.0).count() as u64;
            self.stats.total_residual_applied += residual_applied;

            gradient
                .iter()
                .zip(self.residual.iter())
                .map(|(&g, &r)| g + r)
                .collect()
        } else {
            gradient.to_vec()
        };

        // Step 2a: threshold filtering — zero out below-threshold elements.
        if let Some(thresh) = self.config.threshold {
            for v in working.iter_mut() {
                if v.abs() < thresh {
                    *v = 0.0;
                }
            }
        }

        // Step 2b: top-k selection.
        // Collect candidate (index, absolute_value) pairs for all non-zero elements.
        let mut candidates: Vec<(usize, f32)> = working
            .iter()
            .enumerate()
            .filter(|(_, &v)| v != 0.0)
            .map(|(i, &v)| (i, v.abs()))
            .collect();

        let keep_count = match self.config.top_k {
            Some(k) => k.min(candidates.len()),
            None => candidates.len(),
        };

        // Partial sort: bring the top `keep_count` elements to the front.
        // We use a partial selection sort variant via `select_nth_unstable_by`
        // on the candidates slice.
        let kept_indices: std::collections::HashSet<usize> = if keep_count < candidates.len() {
            // Partition so that the largest `keep_count` items (by abs value) are at [0..keep_count].
            candidates.select_nth_unstable_by(keep_count, |a, b| {
                // Descending order: larger absolute values first.
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            candidates[..keep_count].iter().map(|&(i, _)| i).collect()
        } else {
            candidates.iter().map(|&(i, _)| i).collect()
        };

        // Step 3: build sparse output and update residual.
        let mut indices: Vec<u32> = Vec::with_capacity(keep_count);
        let mut values: Vec<f32> = Vec::with_capacity(keep_count);

        // We iterate over `working` in order so that the output is index-sorted.
        for (i, &val) in working.iter().enumerate() {
            if kept_indices.contains(&i) {
                indices.push(i as u32);
                values.push(val);
                self.residual[i] = 0.0; // sent — clear residual
            } else {
                // Accumulate the working value (original gradient + previous residual)
                // into the residual for the next round.
                if self.config.accumulate_residuals {
                    self.residual[i] = val;
                } else {
                    self.residual[i] = 0.0;
                }
            }
        }

        // Step 4: update statistics.
        let kept = indices.len() as u64;
        let dropped = (len as u64).saturating_sub(kept);
        self.stats.total_rounds += 1;
        self.stats.total_elements_kept += kept;
        self.stats.total_elements_dropped += dropped;

        SparseGradient {
            indices,
            values,
            original_len: len,
        }
    }

    /// Clear the residual buffer (set all entries to zero).
    pub fn reset_residual(&mut self) {
        for v in self.residual.iter_mut() {
            *v = 0.0;
        }
    }
}

// ── DeltaStats ──────────────────────────────────────────────────────────────

/// Cumulative statistics for a [`DeltaEncoder`].
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DeltaStats {
    /// Total number of encode calls.
    pub total_encoded: u64,
    /// Number of times the full gradient was sent (no previous available).
    pub total_full_sends: u64,
    /// Number of times a delta was sent instead of the full gradient.
    pub total_delta_sends: u64,
}

// ── GradientDelta (sparsify module variant) ─────────────────────────────────

/// A gradient update that is either a complete gradient or a delta from the
/// previous round.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientDelta {
    /// Either the full gradient vector (`is_full = true`) or the element-wise
    /// delta from the previous round's gradient.
    pub values: Vec<f32>,
    /// `true` when `values` contains the full gradient rather than a delta.
    pub is_full: bool,
    /// The federated learning round number this update belongs to.
    pub round: u64,
}

impl GradientDelta {
    /// Compression ratio expressed as mean absolute delta divided by the
    /// maximum possible delta magnitude.
    ///
    /// For full gradients, returns `1.0` (no compression). For delta updates,
    /// a lower value indicates a smaller delta and therefore better effective
    /// compression relative to sending the full gradient.
    ///
    /// `original_len` is the number of elements in the underlying gradient
    /// (used for normalisation when `values.len() != original_len`).
    pub fn compression_ratio(&self, original_len: usize) -> f64 {
        if self.is_full || self.values.is_empty() || original_len == 0 {
            return 1.0;
        }
        let mean_abs_delta: f64 =
            self.values.iter().map(|&v| v.abs() as f64).sum::<f64>() / self.values.len() as f64;

        let max_possible: f64 = self
            .values
            .iter()
            .map(|&v| v.abs() as f64)
            .fold(0.0_f64, f64::max);

        if max_possible == 0.0 {
            return 0.0;
        }
        mean_abs_delta / max_possible
    }
}

// ── DeltaEncoder ────────────────────────────────────────────────────────────

/// Stateful encoder that computes element-wise deltas between successive
/// gradient rounds.
///
/// On the first call (or after [`DeltaEncoder::reset`]), the full gradient is
/// returned. Subsequent calls return the element-wise difference from the
/// previously stored gradient.
pub struct DeltaEncoder {
    /// The gradient that was sent in the previous round.
    pub previous: Option<Vec<f32>>,
    /// Cumulative statistics.
    pub stats: DeltaStats,
    /// Internal round counter, incremented on every [`encode_delta`] call.
    round_counter: u64,
}

impl DeltaEncoder {
    /// Create a new delta encoder with no prior state.
    pub fn new() -> Self {
        Self {
            previous: None,
            stats: DeltaStats::default(),
            round_counter: 0,
        }
    }

    /// Encode `current` as either a full gradient or a delta.
    ///
    /// - If no previous gradient is stored, the full gradient is returned and
    ///   `is_full` is set to `true`.
    /// - Otherwise, the element-wise delta `current[i] - previous[i]` is
    ///   returned.
    ///
    /// The internal round counter is incremented on every call. If the length
    /// of `current` differs from the stored previous gradient, the previous
    /// state is discarded and a full send is performed.
    pub fn encode_delta(&mut self, current: &[f32]) -> GradientDelta {
        let round = self.round_counter;
        self.round_counter += 1;
        self.stats.total_encoded += 1;

        match &self.previous {
            None => {
                self.stats.total_full_sends += 1;
                let values = current.to_vec();
                self.previous = Some(values.clone());
                GradientDelta {
                    values,
                    is_full: true,
                    round,
                }
            }
            Some(prev) if prev.len() != current.len() => {
                // Shape mismatch: treat as a fresh start.
                self.stats.total_full_sends += 1;
                let values = current.to_vec();
                self.previous = Some(values.clone());
                GradientDelta {
                    values,
                    is_full: true,
                    round,
                }
            }
            Some(prev) => {
                let delta: Vec<f32> = current
                    .iter()
                    .zip(prev.iter())
                    .map(|(&c, &p)| c - p)
                    .collect();
                self.stats.total_delta_sends += 1;
                self.previous = Some(current.to_vec());
                GradientDelta {
                    values: delta,
                    is_full: false,
                    round,
                }
            }
        }
    }

    /// Reconstruct a full gradient from a `base` vector and a [`GradientDelta`].
    ///
    /// - If `delta.is_full`, returns a clone of `delta.values`.
    /// - Otherwise, adds `delta.values[i]` to `base[i]` element-wise.
    ///
    /// If the lengths of `base` and `delta.values` disagree, the shorter length
    /// is used (extra elements in the longer slice are silently ignored).
    pub fn decode_delta(&self, base: &[f32], delta: &GradientDelta) -> Vec<f32> {
        if delta.is_full {
            return delta.values.clone();
        }
        let len = base.len().min(delta.values.len());
        let mut result = base.to_vec();
        result.truncate(len);
        for (r, &d) in result.iter_mut().zip(delta.values.iter()) {
            *r += d;
        }
        result
    }

    /// Clear the stored previous gradient so that the next `encode_delta`
    /// call performs a full send.
    pub fn reset(&mut self) {
        self.previous = None;
    }
}

impl Default for DeltaEncoder {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── SparseGradient ────────────────────────────────────────────────────

    #[test]
    fn test_sparse_gradient_sparsity_ratio_all_kept() {
        let sg = SparseGradient {
            indices: vec![0, 1, 2, 3],
            values: vec![1.0, 2.0, 3.0, 4.0],
            original_len: 4,
        };
        let ratio = sg.sparsity_ratio();
        assert!((ratio - 0.0).abs() < 1e-9, "expected 0.0, got {}", ratio);
    }

    #[test]
    fn test_sparse_gradient_sparsity_ratio_half() {
        let sg = SparseGradient {
            indices: vec![1, 3],
            values: vec![0.5, 1.5],
            original_len: 4,
        };
        let ratio = sg.sparsity_ratio();
        assert!((ratio - 0.5).abs() < 1e-9, "expected 0.5, got {}", ratio);
    }

    #[test]
    fn test_sparse_gradient_to_dense_basic() {
        let sg = SparseGradient {
            indices: vec![0, 2, 4],
            values: vec![1.0, 3.0, 5.0],
            original_len: 6,
        };
        let dense = sg.to_dense();
        assert_eq!(dense, vec![1.0, 0.0, 3.0, 0.0, 5.0, 0.0]);
    }

    #[test]
    fn test_sparse_gradient_to_dense_empty() {
        let sg = SparseGradient {
            indices: vec![],
            values: vec![],
            original_len: 5,
        };
        let dense = sg.to_dense();
        assert_eq!(dense, vec![0.0; 5]);
    }

    // ── GradientSparsifier: top-k ─────────────────────────────────────────

    #[test]
    fn test_sparsifier_top_k_keeps_largest() {
        let config = SparsityConfig {
            top_k: Some(2),
            threshold: None,
            accumulate_residuals: false,
        };
        let mut sparsifier = GradientSparsifier::new(config, 5);
        let gradient = vec![0.1_f32, 5.0, 0.2, 8.0, 0.3];
        let sparse = sparsifier.sparsify(&gradient);

        // Should keep 8.0 (index 3) and 5.0 (index 1).
        assert_eq!(sparse.indices.len(), 2);
        assert!(sparse.values.contains(&8.0), "8.0 must be kept");
        assert!(sparse.values.contains(&5.0), "5.0 must be kept");
    }

    #[test]
    fn test_sparsifier_top_k_respects_absolute_value() {
        let config = SparsityConfig {
            top_k: Some(2),
            threshold: None,
            accumulate_residuals: false,
        };
        let mut sparsifier = GradientSparsifier::new(config, 4);
        // -9.0 has the largest absolute value, followed by 7.0.
        let gradient = vec![1.0_f32, -9.0, 7.0, 0.5];
        let sparse = sparsifier.sparsify(&gradient);

        assert_eq!(sparse.indices.len(), 2);
        assert!(sparse.values.contains(&-9.0), "-9.0 must be kept");
        assert!(sparse.values.contains(&7.0), "7.0 must be kept");
    }

    // ── GradientSparsifier: threshold ─────────────────────────────────────

    #[test]
    fn test_sparsifier_threshold_drops_small() {
        let config = SparsityConfig {
            top_k: None,
            threshold: Some(1.0),
            accumulate_residuals: false,
        };
        let mut sparsifier = GradientSparsifier::new(config, 5);
        let gradient = vec![0.1_f32, 5.0, 0.2, 8.0, 0.3];
        let sparse = sparsifier.sparsify(&gradient);

        // Only 5.0 and 8.0 exceed the threshold.
        assert_eq!(sparse.indices.len(), 2);
        let dense = sparse.to_dense();
        assert_eq!(dense[1], 5.0);
        assert_eq!(dense[3], 8.0);
        assert_eq!(dense[0], 0.0);
        assert_eq!(dense[2], 0.0);
        assert_eq!(dense[4], 0.0);
    }

    #[test]
    fn test_sparsifier_threshold_keeps_all_above() {
        let config = SparsityConfig {
            top_k: None,
            threshold: Some(0.0),
            accumulate_residuals: false,
        };
        let mut sparsifier = GradientSparsifier::new(config, 3);
        let gradient = vec![1.0_f32, 2.0, 3.0];
        let sparse = sparsifier.sparsify(&gradient);

        // threshold=0.0 means values with |v| < 0 are dropped, so all are kept.
        assert_eq!(sparse.indices.len(), 3);
    }

    // ── Residual accumulation ─────────────────────────────────────────────

    #[test]
    fn test_residual_accumulation_carries_forward() {
        let config = SparsityConfig {
            top_k: Some(1),
            threshold: None,
            accumulate_residuals: true,
        };
        let mut sparsifier = GradientSparsifier::new(config, 3);

        // Round 1: [0.5, 0.4, 0.3] — only top-1 (0.5 at index 0) is kept.
        let g1 = vec![0.5_f32, 0.4, 0.3];
        let _s1 = sparsifier.sparsify(&g1);

        // Residual should now be [0.0, 0.4, 0.3].
        assert!((sparsifier.residual[0] - 0.0).abs() < 1e-6);
        assert!((sparsifier.residual[1] - 0.4).abs() < 1e-6);
        assert!((sparsifier.residual[2] - 0.3).abs() < 1e-6);

        // Round 2: [0.1, 0.1, 0.1] — working = [0.1, 0.5, 0.4].
        // Top-1 is now index 1 (0.5).
        let g2 = vec![0.1_f32, 0.1, 0.1];
        let s2 = sparsifier.sparsify(&g2);

        assert_eq!(s2.indices.len(), 1);
        assert_eq!(s2.indices[0], 1, "index 1 should be kept in round 2");
        // Value should be the combined working value 0.1 + 0.4 = 0.5.
        assert!(
            (s2.values[0] - 0.5).abs() < 1e-5,
            "expected 0.5, got {}",
            s2.values[0]
        );
    }

    #[test]
    fn test_residual_reset_clears_buffer() {
        let config = SparsityConfig {
            top_k: Some(1),
            threshold: None,
            accumulate_residuals: true,
        };
        let mut sparsifier = GradientSparsifier::new(config, 3);

        sparsifier.sparsify(&[0.5_f32, 0.4, 0.3]);
        // Residuals are non-zero after the first round.
        assert!(sparsifier.residual.iter().any(|&v| v != 0.0));

        sparsifier.reset_residual();
        assert!(sparsifier.residual.iter().all(|&v| v == 0.0));
    }

    // ── Stats accumulation ────────────────────────────────────────────────

    #[test]
    fn test_sparsifier_stats_accumulation() {
        let config = SparsityConfig {
            top_k: Some(2),
            threshold: None,
            accumulate_residuals: false,
        };
        let mut sparsifier = GradientSparsifier::new(config, 4);

        sparsifier.sparsify(&[1.0_f32, 2.0, 3.0, 4.0]);
        sparsifier.sparsify(&[0.1_f32, 0.2, 0.3, 0.4]);

        assert_eq!(sparsifier.stats.total_rounds, 2);
        // Each round keeps 2 out of 4.
        assert_eq!(sparsifier.stats.total_elements_kept, 4);
        assert_eq!(sparsifier.stats.total_elements_dropped, 4);
    }

    // ── DeltaEncoder ──────────────────────────────────────────────────────

    #[test]
    fn test_delta_encoder_first_call_is_full() {
        let mut encoder = DeltaEncoder::new();
        let g = vec![1.0_f32, 2.0, 3.0];
        let delta = encoder.encode_delta(&g);

        assert!(delta.is_full, "first call must be a full send");
        assert_eq!(delta.values, g);
        assert_eq!(delta.round, 0);
    }

    #[test]
    fn test_delta_encoder_subsequent_call_is_delta() {
        let mut encoder = DeltaEncoder::new();
        let g1 = vec![1.0_f32, 2.0, 3.0];
        let g2 = vec![1.5_f32, 2.5, 3.5];

        encoder.encode_delta(&g1);
        let delta = encoder.encode_delta(&g2);

        assert!(!delta.is_full, "second call must be a delta");
        assert_eq!(delta.values, vec![0.5, 0.5, 0.5]);
        assert_eq!(delta.round, 1);
    }

    #[test]
    fn test_delta_encoder_decode_full() {
        let encoder = DeltaEncoder::new();
        let base = vec![0.0_f32; 3];
        let delta = GradientDelta {
            values: vec![1.0, 2.0, 3.0],
            is_full: true,
            round: 0,
        };
        let result = encoder.decode_delta(&base, &delta);
        assert_eq!(result, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_delta_encoder_decode_reconstructs_correctly() {
        let mut encoder = DeltaEncoder::new();
        let g1 = vec![1.0_f32, 2.0, 3.0];
        let g2 = vec![1.5_f32, 2.0, 4.0];

        let _full = encoder.encode_delta(&g1);
        let delta = encoder.encode_delta(&g2);

        // Reconstruct from g1 (base) + delta.
        let encoder2 = DeltaEncoder::new();
        let reconstructed = encoder2.decode_delta(&g1, &delta);
        assert_eq!(reconstructed.len(), g2.len());
        for (r, &expected) in reconstructed.iter().zip(g2.iter()) {
            assert!(
                (r - expected).abs() < 1e-5,
                "mismatch: {} vs {}",
                r,
                expected
            );
        }
    }

    #[test]
    fn test_delta_encoder_reset_forces_full_send() {
        let mut encoder = DeltaEncoder::new();
        encoder.encode_delta(&[1.0_f32, 2.0]);
        encoder.reset();

        let delta = encoder.encode_delta(&[3.0_f32, 4.0]);
        assert!(delta.is_full, "after reset, send must be full");
        assert_eq!(delta.values, vec![3.0, 4.0]);
    }

    #[test]
    fn test_delta_encoder_stats() {
        let mut encoder = DeltaEncoder::new();
        encoder.encode_delta(&[1.0_f32, 2.0]);
        encoder.encode_delta(&[1.5_f32, 2.5]);
        encoder.encode_delta(&[2.0_f32, 3.0]);

        assert_eq!(encoder.stats.total_encoded, 3);
        assert_eq!(encoder.stats.total_full_sends, 1);
        assert_eq!(encoder.stats.total_delta_sends, 2);
    }

    #[test]
    fn test_gradient_delta_compression_ratio_full() {
        let delta = GradientDelta {
            values: vec![1.0, 2.0, 3.0],
            is_full: true,
            round: 0,
        };
        assert!(
            (delta.compression_ratio(3) - 1.0).abs() < 1e-9,
            "full gradient compression ratio must be 1.0"
        );
    }

    #[test]
    fn test_gradient_delta_compression_ratio_delta() {
        // A delta with small changes: mean abs = (0.1+0.1+0.1)/3 = 0.1, max = 0.1 → ratio = 1.0
        let delta = GradientDelta {
            values: vec![0.1_f32, 0.1, 0.1],
            is_full: false,
            round: 1,
        };
        let ratio = delta.compression_ratio(3);
        assert!(
            (ratio - 1.0).abs() < 1e-5,
            "uniform delta should give ratio 1.0, got {}",
            ratio
        );
    }

    #[test]
    fn test_sparsity_ratio_zero_len() {
        let sg = SparseGradient {
            indices: vec![],
            values: vec![],
            original_len: 0,
        };
        assert_eq!(sg.sparsity_ratio(), 0.0);
    }

    #[test]
    fn test_sparsifier_top_k_combined_with_threshold() {
        // Both top_k and threshold are active: threshold filters first, then top_k.
        let config = SparsityConfig {
            top_k: Some(1),
            threshold: Some(2.0),
            accumulate_residuals: false,
        };
        let mut sparsifier = GradientSparsifier::new(config, 5);
        // After threshold(2.0): only 5.0 and 8.0 survive; top_k(1) keeps 8.0.
        let sparse = sparsifier.sparsify(&[0.1_f32, 5.0, 0.2, 8.0, 0.3]);

        assert_eq!(sparse.indices.len(), 1);
        assert_eq!(sparse.values[0], 8.0);
    }
}
