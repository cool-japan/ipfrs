//! Quantization error tracking for INT8/binary quantization.
//!
//! This module measures and bounds quantization error introduced by INT8 or binary
//! quantization, enabling monitoring of search quality degradation over time.
//!
//! ## Overview
//!
//! Quantization reduces memory usage and speeds up distance computations, but at the
//! cost of precision. The [`QuantizationErrorTracker`] accumulates per-vector error
//! measurements and exposes rolling statistics (MSE, MAE, p99 MSE, SNR) to help
//! operators decide when re-indexing or parameter tuning is necessary.
//!
//! ## Example
//!
//! ```rust
//! use ipfrs_semantic::quantization_error::{QuantizationErrorTracker, QErrorError};
//!
//! let mut tracker = QuantizationErrorTracker::new();
//!
//! let original  = vec![0.1_f32, 0.2, 0.3, 0.4];
//! let quantized = vec![0.1_f32, 0.2, 0.3, 0.4]; // perfect quantization
//!
//! let err = tracker.compute_error(&original, &quantized).unwrap();
//! assert!(err.mse < 1e-9);
//!
//! tracker.record(err);
//! assert_eq!(tracker.history_len(), 1);
//! ```

use std::collections::VecDeque;

use thiserror::Error;

// ──────────────────────────────────────────────────────────────────────────────
// Error type
// ──────────────────────────────────────────────────────────────────────────────

/// Errors that can be returned from [`QuantizationErrorTracker`] operations.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum QErrorError {
    /// Original and quantized vectors have different lengths.
    #[error("vector length mismatch: expected {expected}, got {got}")]
    LengthMismatch {
        /// Length of the original vector.
        expected: usize,
        /// Length of the quantized vector.
        got: usize,
    },
    /// Both vectors are empty; error metrics are undefined.
    #[error("cannot compute quantization error for empty vectors")]
    EmptyVector,
}

// ──────────────────────────────────────────────────────────────────────────────
// QuantizationError
// ──────────────────────────────────────────────────────────────────────────────

/// Quantization error metrics for a single vector pair.
#[derive(Debug, Clone, PartialEq)]
pub struct QuantizationError {
    /// Mean squared error between original and quantized vectors.
    pub mse: f64,
    /// Mean absolute error between original and quantized vectors.
    pub mae: f64,
    /// Maximum element-wise absolute error.
    pub max_error: f64,
    /// Signal-to-noise ratio in decibels.
    ///
    /// Defined as `10 * log10(signal_power / noise_power)` where
    /// `signal_power = mean(orig²)` and `noise_power = mse`.
    /// Returns `-∞` when signal power is zero and noise power is also zero
    /// (perfect quantization of a zero vector), or `+∞` when MSE is zero
    /// and signal power is non-zero.
    pub snr_db: f64,
}

impl QuantizationError {
    /// Returns `true` when this error measurement is within an acceptable threshold.
    ///
    /// # Arguments
    /// * `max_mse` – the maximum allowable MSE.
    #[inline]
    pub fn is_acceptable(&self, max_mse: f64) -> bool {
        self.mse <= max_mse
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// QuantizationErrorTracker
// ──────────────────────────────────────────────────────────────────────────────

/// Capacity of the rolling history buffer.
const HISTORY_CAPACITY: usize = 256;

/// Tracks quantization error across a rolling window of vectors.
///
/// The tracker maintains the last `HISTORY_CAPACITY` (256) measurements and
/// exposes rolling statistics that let callers detect when quantization error
/// is accumulating to unacceptable levels.
#[derive(Debug, Clone)]
pub struct QuantizationErrorTracker {
    /// Rolling window of recent error measurements (capped at 256).
    pub history: VecDeque<QuantizationError>,
    /// Total number of vectors recorded since creation (monotonically increasing).
    pub total_vectors: u64,
}

impl QuantizationErrorTracker {
    /// Create a new, empty tracker.
    pub fn new() -> Self {
        Self {
            history: VecDeque::with_capacity(HISTORY_CAPACITY),
            total_vectors: 0,
        }
    }

    /// Compute [`QuantizationError`] between an original and a quantized vector.
    ///
    /// # Errors
    /// - [`QErrorError::EmptyVector`] – if `original` is empty.
    /// - [`QErrorError::LengthMismatch`] – if lengths differ.
    pub fn compute_error(
        &self,
        original: &[f32],
        quantized: &[f32],
    ) -> Result<QuantizationError, QErrorError> {
        let n = original.len();

        if n == 0 {
            return Err(QErrorError::EmptyVector);
        }
        if quantized.len() != n {
            return Err(QErrorError::LengthMismatch {
                expected: n,
                got: quantized.len(),
            });
        }

        let n_f64 = n as f64;
        let mut sum_sq_err = 0.0_f64;
        let mut sum_abs_err = 0.0_f64;
        let mut max_abs_err = 0.0_f64;
        let mut sum_sq_orig = 0.0_f64;

        for (&o, &q) in original.iter().zip(quantized.iter()) {
            let o64 = o as f64;
            let q64 = q as f64;
            let diff = o64 - q64;
            let abs_diff = diff.abs();

            sum_sq_err += diff * diff;
            sum_abs_err += abs_diff;
            if abs_diff > max_abs_err {
                max_abs_err = abs_diff;
            }
            sum_sq_orig += o64 * o64;
        }

        let mse = sum_sq_err / n_f64;
        let mae = sum_abs_err / n_f64;
        let max_error = max_abs_err;
        let signal_power = sum_sq_orig / n_f64;

        // SNR in dB: 10 * log10(signal / noise)
        // Edge cases handled explicitly so the result is always well-defined:
        //   - signal == 0 and mse == 0  → 0 dB (both are zero vectors, no error)
        //   - signal == 0 and mse  > 0  → -∞ dB (noise with no signal)
        //   - signal  > 0 and mse == 0  → +∞ dB (perfect reconstruction)
        let snr_db = if signal_power == 0.0 && mse == 0.0 {
            0.0
        } else if mse == 0.0 {
            f64::INFINITY
        } else {
            10.0 * (signal_power / mse).log10()
        };

        Ok(QuantizationError {
            mse,
            mae,
            max_error,
            snr_db,
        })
    }

    /// Record a [`QuantizationError`] measurement into the rolling history.
    ///
    /// When the history reaches 256 entries the oldest entry is evicted.
    pub fn record(&mut self, error: QuantizationError) {
        if self.history.len() == HISTORY_CAPACITY {
            self.history.pop_front();
        }
        self.history.push_back(error);
        self.total_vectors += 1;
    }

    /// Average MSE over all entries in the rolling history.
    ///
    /// Returns `0.0` when the history is empty.
    pub fn rolling_mse(&self) -> f64 {
        if self.history.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.history.iter().map(|e| e.mse).sum();
        sum / self.history.len() as f64
    }

    /// Average MAE over all entries in the rolling history.
    ///
    /// Returns `0.0` when the history is empty.
    pub fn rolling_mae(&self) -> f64 {
        if self.history.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.history.iter().map(|e| e.mae).sum();
        sum / self.history.len() as f64
    }

    /// 99th-percentile MSE across the rolling history.
    ///
    /// Uses the nearest-rank method: the index is `ceil(0.99 * n) - 1` (0-based)
    /// on a sorted copy of the MSE values.
    ///
    /// Returns `0.0` when the history is empty.
    pub fn p99_mse(&self) -> f64 {
        if self.history.is_empty() {
            return 0.0;
        }
        let mut values: Vec<f64> = self.history.iter().map(|e| e.mse).collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n = values.len();
        // Nearest-rank: index = ceil(p * n) - 1, clamped to [0, n-1]
        let rank = ((0.99_f64 * n as f64).ceil() as usize)
            .saturating_sub(1)
            .min(n - 1);
        values[rank]
    }

    /// Returns `true` when the rolling MSE is within `max_mse`.
    pub fn is_quality_acceptable(&self, max_mse: f64) -> bool {
        self.rolling_mse() <= max_mse
    }

    /// Returns the number of entries currently in the rolling history.
    pub fn history_len(&self) -> usize {
        self.history.len()
    }

    /// Clear all history entries.  `total_vectors` is not reset.
    pub fn clear_history(&mut self) {
        self.history.clear();
    }
}

impl Default for QuantizationErrorTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ──────────────────────────────────────────────────────────────

    fn make_tracker() -> QuantizationErrorTracker {
        QuantizationErrorTracker::new()
    }

    // ── 1. identical vectors → mse=0, mae=0 ─────────────────────────────────
    #[test]
    fn test_compute_error_identical_vectors() {
        let t = make_tracker();
        let v = vec![0.1_f32, 0.5, -0.3, 0.9, 0.0];
        let err = t.compute_error(&v, &v).expect("should succeed");
        assert!(
            err.mse < 1e-12,
            "MSE for identical vectors must be ~0, got {}",
            err.mse
        );
        assert!(
            err.mae < 1e-12,
            "MAE for identical vectors must be ~0, got {}",
            err.mae
        );
        assert!(
            err.max_error < 1e-12,
            "max_error for identical vectors must be ~0, got {}",
            err.max_error
        );
    }

    // ── 2. SNR is +∞ for identical non-zero vectors ──────────────────────────
    #[test]
    fn test_compute_error_identical_snr_infinite() {
        let t = make_tracker();
        let v = vec![1.0_f32, 2.0, 3.0];
        let err = t.compute_error(&v, &v).expect("should succeed");
        assert!(
            err.snr_db.is_infinite() && err.snr_db.is_sign_positive(),
            "SNR should be +∞ for perfect reconstruction, got {}",
            err.snr_db
        );
    }

    // ── 3. known constant offset ─────────────────────────────────────────────
    #[test]
    fn test_compute_error_constant_offset() {
        let t = make_tracker();
        let offset = 0.5_f32;
        let original = vec![1.0_f32, 2.0, 3.0, 4.0];
        let quantized: Vec<f32> = original.iter().map(|x| x + offset).collect();

        let err = t
            .compute_error(&original, &quantized)
            .expect("should succeed");

        let expected_mse = (offset as f64).powi(2);
        let expected_mae = offset as f64;

        assert!(
            (err.mse - expected_mse).abs() < 1e-9,
            "MSE = {}, expected {}",
            err.mse,
            expected_mse
        );
        assert!(
            (err.mae - expected_mae).abs() < 1e-9,
            "MAE = {}, expected {}",
            err.mae,
            expected_mae
        );
        assert!(
            (err.max_error - offset as f64).abs() < 1e-9,
            "max_error = {}, expected {}",
            err.max_error,
            offset
        );
    }

    // ── 4. length mismatch returns error ─────────────────────────────────────
    #[test]
    fn test_compute_error_length_mismatch() {
        let t = make_tracker();
        let original = vec![1.0_f32, 2.0, 3.0];
        let quantized = vec![1.0_f32, 2.0];

        match t.compute_error(&original, &quantized) {
            Err(QErrorError::LengthMismatch {
                expected: 3,
                got: 2,
            }) => {}
            other => panic!("expected LengthMismatch(3, 2), got {:?}", other),
        }
    }

    // ── 5. empty vector returns error ────────────────────────────────────────
    #[test]
    fn test_compute_error_empty_vector() {
        let t = make_tracker();
        match t.compute_error(&[], &[]) {
            Err(QErrorError::EmptyVector) => {}
            other => panic!("expected EmptyVector, got {:?}", other),
        }
    }

    // ── 6. snr_db is positive for low-noise quantization ─────────────────────
    #[test]
    fn test_snr_positive_for_low_noise() {
        let t = make_tracker();
        // Large signal, tiny noise
        let original: Vec<f32> = (0..64).map(|i| (i as f32) * 0.1).collect();
        let quantized: Vec<f32> = original.iter().map(|&x| x + 1e-4).collect();

        let err = t
            .compute_error(&original, &quantized)
            .expect("should succeed");
        assert!(
            err.snr_db > 0.0,
            "SNR should be positive for low-noise quantization, got {}",
            err.snr_db
        );
    }

    // ── 7. record appends to history ─────────────────────────────────────────
    #[test]
    fn test_record_appends_to_history() {
        let mut tracker = make_tracker();
        assert_eq!(tracker.history_len(), 0);

        let t = make_tracker();
        let v = vec![0.5_f32; 4];
        let err = t.compute_error(&v, &v).expect("should succeed");

        tracker.record(err);
        assert_eq!(tracker.history_len(), 1);
        assert_eq!(tracker.total_vectors, 1);
    }

    // ── 8. history is capped at 256 ───────────────────────────────────────────
    #[test]
    fn test_record_capped_at_256() {
        let mut tracker = make_tracker();
        let t = make_tracker();
        let v = vec![1.0_f32; 8];

        for _ in 0..300 {
            let err = t.compute_error(&v, &v).expect("should succeed");
            tracker.record(err);
        }

        assert_eq!(tracker.history_len(), 256, "history must be capped at 256");
        assert_eq!(
            tracker.total_vectors, 300,
            "total_vectors must reflect all recordings"
        );
    }

    // ── 9. rolling_mse average correctness ───────────────────────────────────
    #[test]
    fn test_rolling_mse_average() {
        let mut tracker = make_tracker();
        let t = make_tracker();

        // Create errors with known MSE values by using constant-offset vectors.
        // offset=1 → mse=1, offset=2 → mse=4, offset=3 → mse=9
        for offset in [1.0_f32, 2.0, 3.0] {
            let original = vec![0.0_f32; 4];
            let quantized = vec![offset; 4];
            let err = t
                .compute_error(&original, &quantized)
                .expect("should succeed");
            tracker.record(err);
        }

        // Expected average = (1 + 4 + 9) / 3 = 14/3 ≈ 4.6667
        let expected = (1.0_f64 + 4.0 + 9.0) / 3.0;
        let got = tracker.rolling_mse();
        assert!(
            (got - expected).abs() < 1e-9,
            "rolling_mse = {}, expected {}",
            got,
            expected
        );
    }

    // ── 10. rolling_mae average correctness ──────────────────────────────────
    #[test]
    fn test_rolling_mae_average() {
        let mut tracker = make_tracker();
        let t = make_tracker();

        for offset in [1.0_f32, 3.0] {
            let original = vec![0.0_f32; 4];
            let quantized = vec![offset; 4];
            let err = t
                .compute_error(&original, &quantized)
                .expect("should succeed");
            tracker.record(err);
        }

        // MAE values: 1 and 3 → average = 2
        let got = tracker.rolling_mae();
        assert!(
            (got - 2.0).abs() < 1e-9,
            "rolling_mae = {}, expected 2.0",
            got
        );
    }

    // ── 11. p99_mse correctness ────────────────────────────────────────────────
    #[test]
    fn test_p99_mse_correctness() {
        let mut tracker = make_tracker();
        let t = make_tracker();
        let original = vec![0.0_f32; 1];

        // Insert 100 measurements with MSE = i² (i in 1..=100)
        for i in 1_u32..=100 {
            let offset = i as f32;
            let quantized = vec![offset];
            let err = t.compute_error(&original, &quantized).expect("ok");
            tracker.record(err);
        }

        // Sorted MSE values: 1, 4, 9, …, 10000
        // p99 nearest-rank index = ceil(0.99 * 100) - 1 = 99 - 1 = 98 (0-based)
        // → 99th value in sorted order = 99² = 9801
        let got = tracker.p99_mse();
        let expected = 99.0_f64 * 99.0;
        assert!(
            (got - expected).abs() < 1e-6,
            "p99_mse = {}, expected {}",
            got,
            expected
        );
    }

    // ── 12. p99_mse on empty history ──────────────────────────────────────────
    #[test]
    fn test_p99_mse_empty_history() {
        let tracker = make_tracker();
        assert_eq!(tracker.p99_mse(), 0.0);
    }

    // ── 13. is_quality_acceptable threshold ──────────────────────────────────
    #[test]
    fn test_is_quality_acceptable_threshold() {
        let mut tracker = make_tracker();
        let t = make_tracker();

        let original = vec![0.0_f32; 4];
        let quantized = vec![1.0_f32; 4]; // mse = 1.0

        let err = t.compute_error(&original, &quantized).expect("ok");
        tracker.record(err);

        assert!(
            tracker.is_quality_acceptable(1.0),
            "rolling_mse == max_mse should be acceptable"
        );
        assert!(
            tracker.is_quality_acceptable(2.0),
            "rolling_mse < max_mse should be acceptable"
        );
        assert!(
            !tracker.is_quality_acceptable(0.5),
            "rolling_mse > max_mse should be unacceptable"
        );
    }

    // ── 14. clear_history empties the buffer ─────────────────────────────────
    #[test]
    fn test_clear_history_empties() {
        let mut tracker = make_tracker();
        let t = make_tracker();
        let v = vec![1.0_f32; 4];

        for _ in 0..10 {
            let err = t.compute_error(&v, &v).expect("ok");
            tracker.record(err);
        }

        assert_eq!(tracker.history_len(), 10);
        tracker.clear_history();
        assert_eq!(
            tracker.history_len(),
            0,
            "clear_history must empty the buffer"
        );
        // total_vectors must NOT be reset
        assert_eq!(
            tracker.total_vectors, 10,
            "total_vectors must survive clear_history"
        );
    }

    // ── 15. is_acceptable on QuantizationError ───────────────────────────────
    #[test]
    fn test_quantization_error_is_acceptable() {
        let err = QuantizationError {
            mse: 0.01,
            mae: 0.05,
            max_error: 0.1,
            snr_db: 20.0,
        };
        assert!(err.is_acceptable(0.01));
        assert!(err.is_acceptable(0.05));
        assert!(!err.is_acceptable(0.009));
    }

    // ── 16. snr_db = 0 for zero-vector pair ─────────────────────────────────
    #[test]
    fn test_snr_zero_vector_pair() {
        let t = make_tracker();
        let zeros = vec![0.0_f32; 4];
        let err = t.compute_error(&zeros, &zeros).expect("ok");
        // Both signal and noise are zero → defined as 0 dB
        assert_eq!(err.snr_db, 0.0);
    }

    // ── 17. default() and new() are equivalent ───────────────────────────────
    #[test]
    fn test_default_equals_new() {
        let a = QuantizationErrorTracker::new();
        let b = QuantizationErrorTracker::default();
        assert_eq!(a.history_len(), b.history_len());
        assert_eq!(a.total_vectors, b.total_vectors);
    }
}
