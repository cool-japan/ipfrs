//! TensorQuantizer — symmetric and asymmetric quantization for tensor values.
//!
//! Provides calibration-based quantization with INT8 and INT4 support,
//! including quantize/dequantize roundtrips and MSE error measurement.
//!
//! # Examples
//!
//! ```
//! use ipfrs_tensorlogic::quantizer::{TensorQuantizer, QuantMode, QuantBits};
//!
//! let mut quantizer = TensorQuantizer::new();
//! quantizer.calibrate(&[1.0, -0.5, 0.3, -1.0]);
//!
//! let params = quantizer.compute_params(QuantMode::Symmetric, QuantBits::Int8).expect("example: should succeed in docs");
//! let quantized = TensorQuantizer::quantize(&[0.5, -0.5], &params);
//! let restored = TensorQuantizer::dequantize(&quantized, &params);
//! ```

/// Quantization mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantMode {
    /// Symmetric: zero_point = 0, scale = max(|min|, |max|) / (2^(bits-1) - 1)
    Symmetric,
    /// Asymmetric: scale = (max - min) / (2^bits - 1), zero_point = round(-min / scale)
    Asymmetric,
}

/// Quantization bit width.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantBits {
    /// 8-bit: symmetric range [-128, 127], asymmetric range [0, 255]
    Int8,
    /// 4-bit: symmetric range [-8, 7], asymmetric range [0, 15]
    Int4,
}

impl QuantBits {
    /// Number of bits for this quantization level.
    fn num_bits(self) -> u32 {
        match self {
            Self::Int8 => 8,
            Self::Int4 => 4,
        }
    }

    /// Signed symmetric range: [-(2^(bits-1)), 2^(bits-1) - 1]
    fn symmetric_range(self) -> (i32, i32) {
        let half = 1_i32 << (self.num_bits() - 1);
        (-half, half - 1)
    }

    /// Unsigned asymmetric range: [0, 2^bits - 1]
    fn asymmetric_range(self) -> (i32, i32) {
        let max = (1_i32 << self.num_bits()) - 1;
        (0, max)
    }
}

/// Parameters computed from calibration data for quantization/dequantization.
#[derive(Debug, Clone)]
pub struct QuantParams {
    /// Scale factor: maps floating-point range to integer range.
    pub scale: f64,
    /// Zero point offset (0 for symmetric mode).
    pub zero_point: i32,
    /// Quantization mode used.
    pub mode: QuantMode,
    /// Bit width used.
    pub bits: QuantBits,
}

/// Statistics about the quantizer's calibration state.
#[derive(Debug, Clone)]
pub struct QuantizerStats {
    /// Number of samples observed during calibration.
    pub samples_seen: u64,
    /// Minimum value observed.
    pub calibration_min: f64,
    /// Maximum value observed.
    pub calibration_max: f64,
}

/// Calibration-based tensor quantizer.
///
/// Usage: call [`calibrate`](TensorQuantizer::calibrate) with representative
/// data, then [`compute_params`](TensorQuantizer::compute_params) to derive
/// quantization parameters.
pub struct TensorQuantizer {
    calibration_min: f64,
    calibration_max: f64,
    samples_seen: u64,
}

impl TensorQuantizer {
    /// Create a new uncalibrated quantizer.
    pub fn new() -> Self {
        Self {
            calibration_min: f64::MAX,
            calibration_max: f64::MIN,
            samples_seen: 0,
        }
    }

    /// Update calibration range from observed values.
    ///
    /// Can be called multiple times; min/max accumulate across calls.
    pub fn calibrate(&mut self, values: &[f64]) {
        for &v in values {
            if v < self.calibration_min {
                self.calibration_min = v;
            }
            if v > self.calibration_max {
                self.calibration_max = v;
            }
        }
        self.samples_seen += values.len() as u64;
    }

    /// Compute quantization parameters from calibration data.
    ///
    /// Returns an error if no samples have been observed.
    pub fn compute_params(&self, mode: QuantMode, bits: QuantBits) -> Result<QuantParams, String> {
        if self.samples_seen == 0 {
            return Err("No calibration data: call calibrate() first".to_string());
        }

        let (scale, zero_point) = match mode {
            QuantMode::Symmetric => {
                let abs_max = self.calibration_min.abs().max(self.calibration_max.abs());
                let (_qmin, qmax) = bits.symmetric_range();
                let s = if abs_max == 0.0 {
                    1.0
                } else {
                    abs_max / qmax as f64
                };
                (s, 0)
            }
            QuantMode::Asymmetric => {
                let range = self.calibration_max - self.calibration_min;
                let (_qmin, qmax) = bits.asymmetric_range();
                let s = if range == 0.0 {
                    1.0
                } else {
                    range / qmax as f64
                };
                let zp = (-self.calibration_min / s).round() as i32;
                (s, zp)
            }
        };

        Ok(QuantParams {
            scale,
            zero_point,
            mode,
            bits,
        })
    }

    /// Quantize floating-point values to integers using the given parameters.
    ///
    /// Each value is mapped as: clamp(round(value / scale) + zero_point, qmin, qmax)
    pub fn quantize(values: &[f64], params: &QuantParams) -> Vec<i32> {
        let (qmin, qmax) = match params.mode {
            QuantMode::Symmetric => params.bits.symmetric_range(),
            QuantMode::Asymmetric => params.bits.asymmetric_range(),
        };

        values
            .iter()
            .map(|&v| {
                let q = (v / params.scale).round() as i32 + params.zero_point;
                q.clamp(qmin, qmax)
            })
            .collect()
    }

    /// Dequantize integer values back to floating-point.
    ///
    /// Each value is mapped as: (q - zero_point) * scale
    pub fn dequantize(quantized: &[i32], params: &QuantParams) -> Vec<f64> {
        quantized
            .iter()
            .map(|&q| (q - params.zero_point) as f64 * params.scale)
            .collect()
    }

    /// Compute mean squared error between original values and their
    /// quantize→dequantize roundtrip.
    pub fn quantization_error(original: &[f64], params: &QuantParams) -> f64 {
        if original.is_empty() {
            return 0.0;
        }
        let quantized = Self::quantize(original, params);
        let dequantized = Self::dequantize(&quantized, params);
        let sum_sq: f64 = original
            .iter()
            .zip(dequantized.iter())
            .map(|(&o, &d)| {
                let diff = o - d;
                diff * diff
            })
            .sum();
        sum_sq / original.len() as f64
    }

    /// Reset calibration state.
    pub fn reset_calibration(&mut self) {
        self.calibration_min = f64::MAX;
        self.calibration_max = f64::MIN;
        self.samples_seen = 0;
    }

    /// Return current calibration statistics.
    pub fn stats(&self) -> QuantizerStats {
        QuantizerStats {
            samples_seen: self.samples_seen,
            calibration_min: self.calibration_min,
            calibration_max: self.calibration_max,
        }
    }
}

impl Default for TensorQuantizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Symmetric INT8 ──────────────────────────────────────────────

    #[test]
    fn symmetric_int8_basic() {
        let mut q = TensorQuantizer::new();
        q.calibrate(&[-1.0, 0.0, 1.0]);
        let params = q
            .compute_params(QuantMode::Symmetric, QuantBits::Int8)
            .expect("params");
        assert_eq!(params.zero_point, 0);
        // scale = 1.0 / 127
        let expected_scale = 1.0 / 127.0;
        assert!((params.scale - expected_scale).abs() < 1e-12);
    }

    #[test]
    fn symmetric_int8_quantize_dequantize() {
        let mut q = TensorQuantizer::new();
        q.calibrate(&[-1.0, 1.0]);
        let params = q
            .compute_params(QuantMode::Symmetric, QuantBits::Int8)
            .expect("params");

        let values = vec![0.0, 0.5, -0.5, 1.0, -1.0];
        let quantized = TensorQuantizer::quantize(&values, &params);
        let dequantized = TensorQuantizer::dequantize(&quantized, &params);

        for (&orig, &deq) in values.iter().zip(dequantized.iter()) {
            assert!((orig - deq).abs() < 0.01, "orig={orig} deq={deq}");
        }
    }

    #[test]
    fn symmetric_int8_zero_point_is_zero() {
        let mut q = TensorQuantizer::new();
        q.calibrate(&[-5.0, 3.0]);
        let params = q
            .compute_params(QuantMode::Symmetric, QuantBits::Int8)
            .expect("params");
        assert_eq!(params.zero_point, 0);
    }

    #[test]
    fn symmetric_int8_large_range() {
        let mut q = TensorQuantizer::new();
        q.calibrate(&[-100.0, 100.0]);
        let params = q
            .compute_params(QuantMode::Symmetric, QuantBits::Int8)
            .expect("params");
        let expected_scale = 100.0 / 127.0;
        assert!((params.scale - expected_scale).abs() < 1e-10);
    }

    // ── Asymmetric INT8 ─────────────────────────────────────────────

    #[test]
    fn asymmetric_int8_basic() {
        let mut q = TensorQuantizer::new();
        q.calibrate(&[0.0, 1.0]);
        let params = q
            .compute_params(QuantMode::Asymmetric, QuantBits::Int8)
            .expect("params");
        // scale = 1.0 / 255
        let expected_scale = 1.0 / 255.0;
        assert!((params.scale - expected_scale).abs() < 1e-12);
        // zero_point = round(0 / scale) = 0
        assert_eq!(params.zero_point, 0);
    }

    #[test]
    fn asymmetric_int8_negative_range() {
        let mut q = TensorQuantizer::new();
        q.calibrate(&[-2.0, 2.0]);
        let params = q
            .compute_params(QuantMode::Asymmetric, QuantBits::Int8)
            .expect("params");
        // scale = 4.0 / 255
        let expected_scale = 4.0 / 255.0;
        assert!((params.scale - expected_scale).abs() < 1e-10);
        // zero_point = round(2.0 / scale) = round(127.5) = 128
        let expected_zp = (2.0 / expected_scale).round() as i32;
        assert_eq!(params.zero_point, expected_zp);
    }

    #[test]
    fn asymmetric_int8_roundtrip() {
        let mut q = TensorQuantizer::new();
        q.calibrate(&[-1.0, 3.0]);
        let params = q
            .compute_params(QuantMode::Asymmetric, QuantBits::Int8)
            .expect("params");
        let values = vec![0.0, 1.0, 2.0, 3.0, -1.0];
        let quantized = TensorQuantizer::quantize(&values, &params);
        let dequantized = TensorQuantizer::dequantize(&quantized, &params);
        for (&orig, &deq) in values.iter().zip(dequantized.iter()) {
            assert!((orig - deq).abs() < 0.05, "orig={orig} deq={deq}");
        }
    }

    // ── INT4 quantization ───────────────────────────────────────────

    #[test]
    fn symmetric_int4_basic() {
        let mut q = TensorQuantizer::new();
        q.calibrate(&[-1.0, 1.0]);
        let params = q
            .compute_params(QuantMode::Symmetric, QuantBits::Int4)
            .expect("params");
        assert_eq!(params.zero_point, 0);
        // scale = 1.0 / 7
        let expected_scale = 1.0 / 7.0;
        assert!((params.scale - expected_scale).abs() < 1e-12);
    }

    #[test]
    fn symmetric_int4_clamping() {
        let mut q = TensorQuantizer::new();
        q.calibrate(&[-1.0, 1.0]);
        let params = q
            .compute_params(QuantMode::Symmetric, QuantBits::Int4)
            .expect("params");
        // Value well outside range should clamp
        let quantized = TensorQuantizer::quantize(&[10.0], &params);
        assert_eq!(quantized[0], 7); // clamped to qmax
        let quantized_neg = TensorQuantizer::quantize(&[-10.0], &params);
        assert_eq!(quantized_neg[0], -8); // clamped to qmin
    }

    #[test]
    fn asymmetric_int4_basic() {
        let mut q = TensorQuantizer::new();
        q.calibrate(&[0.0, 1.0]);
        let params = q
            .compute_params(QuantMode::Asymmetric, QuantBits::Int4)
            .expect("params");
        // scale = 1.0 / 15
        let expected_scale = 1.0 / 15.0;
        assert!((params.scale - expected_scale).abs() < 1e-12);
        assert_eq!(params.zero_point, 0);
    }

    #[test]
    fn asymmetric_int4_roundtrip() {
        let mut q = TensorQuantizer::new();
        q.calibrate(&[-2.0, 2.0]);
        let params = q
            .compute_params(QuantMode::Asymmetric, QuantBits::Int4)
            .expect("params");
        let values = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let quantized = TensorQuantizer::quantize(&values, &params);
        let dequantized = TensorQuantizer::dequantize(&quantized, &params);
        for (&orig, &deq) in values.iter().zip(dequantized.iter()) {
            // INT4 has coarser granularity
            assert!((orig - deq).abs() < 0.5, "orig={orig} deq={deq}");
        }
    }

    // ── Calibration accumulation ────────────────────────────────────

    #[test]
    fn calibration_accumulates_across_calls() {
        let mut q = TensorQuantizer::new();
        q.calibrate(&[0.0, 1.0]);
        q.calibrate(&[-2.0, 0.5]);
        q.calibrate(&[0.0, 3.0]);
        let stats = q.stats();
        assert_eq!(stats.samples_seen, 6);
        assert!((stats.calibration_min - (-2.0)).abs() < 1e-15);
        assert!((stats.calibration_max - 3.0).abs() < 1e-15);
    }

    #[test]
    fn calibration_single_value() {
        let mut q = TensorQuantizer::new();
        q.calibrate(&[5.0]);
        let stats = q.stats();
        assert_eq!(stats.samples_seen, 1);
        assert!((stats.calibration_min - 5.0).abs() < 1e-15);
        assert!((stats.calibration_max - 5.0).abs() < 1e-15);
    }

    // ── Roundtrip error ─────────────────────────────────────────────

    #[test]
    fn roundtrip_error_is_small_int8() {
        let mut q = TensorQuantizer::new();
        let values: Vec<f64> = (0..100).map(|i| (i as f64 - 50.0) / 50.0).collect();
        q.calibrate(&values);
        let params = q
            .compute_params(QuantMode::Symmetric, QuantBits::Int8)
            .expect("params");
        let mse = TensorQuantizer::quantization_error(&values, &params);
        assert!(mse < 0.001, "MSE too large: {mse}");
    }

    #[test]
    fn roundtrip_error_larger_for_int4() {
        let mut q = TensorQuantizer::new();
        let values: Vec<f64> = (0..100).map(|i| (i as f64 - 50.0) / 50.0).collect();
        q.calibrate(&values);
        let params_8 = q
            .compute_params(QuantMode::Symmetric, QuantBits::Int8)
            .expect("params8");
        let params_4 = q
            .compute_params(QuantMode::Symmetric, QuantBits::Int4)
            .expect("params4");
        let mse_8 = TensorQuantizer::quantization_error(&values, &params_8);
        let mse_4 = TensorQuantizer::quantization_error(&values, &params_4);
        assert!(mse_4 > mse_8, "INT4 error should exceed INT8 error");
    }

    // ── quantization_error MSE correctness ──────────────────────────

    #[test]
    fn quantization_error_manual_check() {
        let mut q = TensorQuantizer::new();
        q.calibrate(&[-1.0, 1.0]);
        let params = q
            .compute_params(QuantMode::Symmetric, QuantBits::Int8)
            .expect("params");
        let original = vec![0.5];
        let quantized = TensorQuantizer::quantize(&original, &params);
        let dequantized = TensorQuantizer::dequantize(&quantized, &params);
        let diff = original[0] - dequantized[0];
        let expected_mse = diff * diff;
        let mse = TensorQuantizer::quantization_error(&original, &params);
        assert!((mse - expected_mse).abs() < 1e-15);
    }

    #[test]
    fn quantization_error_empty_input() {
        let mut q = TensorQuantizer::new();
        q.calibrate(&[1.0]);
        let params = q
            .compute_params(QuantMode::Symmetric, QuantBits::Int8)
            .expect("params");
        let mse = TensorQuantizer::quantization_error(&[], &params);
        assert!((mse - 0.0).abs() < 1e-15);
    }

    // ── Edge cases ──────────────────────────────────────────────────

    #[test]
    fn all_zeros() {
        let mut q = TensorQuantizer::new();
        q.calibrate(&[0.0, 0.0, 0.0]);
        let params = q
            .compute_params(QuantMode::Symmetric, QuantBits::Int8)
            .expect("params");
        // scale should be 1.0 (fallback for zero range)
        assert!((params.scale - 1.0).abs() < 1e-15);
        let quantized = TensorQuantizer::quantize(&[0.0, 0.0], &params);
        assert!(quantized.iter().all(|&q| q == 0));
    }

    #[test]
    fn all_zeros_asymmetric() {
        let mut q = TensorQuantizer::new();
        q.calibrate(&[0.0, 0.0]);
        let params = q
            .compute_params(QuantMode::Asymmetric, QuantBits::Int8)
            .expect("params");
        assert!((params.scale - 1.0).abs() < 1e-15);
    }

    #[test]
    fn single_value_symmetric() {
        let mut q = TensorQuantizer::new();
        q.calibrate(&[5.0]);
        let params = q
            .compute_params(QuantMode::Symmetric, QuantBits::Int8)
            .expect("params");
        let expected_scale = 5.0 / 127.0;
        assert!((params.scale - expected_scale).abs() < 1e-12);
    }

    #[test]
    fn negative_only_values() {
        let mut q = TensorQuantizer::new();
        q.calibrate(&[-3.0, -1.0, -2.0]);
        let params = q
            .compute_params(QuantMode::Symmetric, QuantBits::Int8)
            .expect("params");
        assert_eq!(params.zero_point, 0);
        let expected_scale = 3.0 / 127.0;
        assert!((params.scale - expected_scale).abs() < 1e-12);
    }

    #[test]
    fn negative_only_asymmetric() {
        let mut q = TensorQuantizer::new();
        q.calibrate(&[-3.0, -1.0]);
        let params = q
            .compute_params(QuantMode::Asymmetric, QuantBits::Int8)
            .expect("params");
        // scale = 2.0 / 255
        let expected_scale = 2.0 / 255.0;
        assert!((params.scale - expected_scale).abs() < 1e-10);
        // zero_point = round(3.0 / scale)
        let expected_zp = (3.0 / expected_scale).round() as i32;
        assert_eq!(params.zero_point, expected_zp);
    }

    // ── Reset calibration ───────────────────────────────────────────

    #[test]
    fn reset_calibration_clears_state() {
        let mut q = TensorQuantizer::new();
        q.calibrate(&[1.0, 2.0, 3.0]);
        q.reset_calibration();
        let stats = q.stats();
        assert_eq!(stats.samples_seen, 0);
        assert_eq!(stats.calibration_min, f64::MAX);
        assert_eq!(stats.calibration_max, f64::MIN);
    }

    #[test]
    fn reset_then_recalibrate() {
        let mut q = TensorQuantizer::new();
        q.calibrate(&[-10.0, 10.0]);
        q.reset_calibration();
        q.calibrate(&[-1.0, 1.0]);
        let params = q
            .compute_params(QuantMode::Symmetric, QuantBits::Int8)
            .expect("params");
        let expected_scale = 1.0 / 127.0;
        assert!((params.scale - expected_scale).abs() < 1e-12);
    }

    // ── Uncalibrated error ──────────────────────────────────────────

    #[test]
    fn error_on_uncalibrated() {
        let q = TensorQuantizer::new();
        let result = q.compute_params(QuantMode::Symmetric, QuantBits::Int8);
        assert!(result.is_err());
        assert!(result.expect_err("should be err").contains("calibration"));
    }

    // ── Clamping at boundaries ──────────────────────────────────────

    #[test]
    fn clamping_symmetric_int8() {
        let mut q = TensorQuantizer::new();
        q.calibrate(&[-1.0, 1.0]);
        let params = q
            .compute_params(QuantMode::Symmetric, QuantBits::Int8)
            .expect("params");
        // Values far beyond calibration range should clamp
        let quantized = TensorQuantizer::quantize(&[1000.0, -1000.0], &params);
        assert_eq!(quantized[0], 127);
        assert_eq!(quantized[1], -128);
    }

    #[test]
    fn clamping_asymmetric_int8() {
        let mut q = TensorQuantizer::new();
        q.calibrate(&[0.0, 1.0]);
        let params = q
            .compute_params(QuantMode::Asymmetric, QuantBits::Int8)
            .expect("params");
        let quantized = TensorQuantizer::quantize(&[1000.0, -1000.0], &params);
        assert_eq!(quantized[0], 255);
        assert_eq!(quantized[1], 0);
    }

    // ── Default trait ───────────────────────────────────────────────

    #[test]
    fn default_trait_works() {
        let q = TensorQuantizer::default();
        let stats = q.stats();
        assert_eq!(stats.samples_seen, 0);
    }

    // ── QuantBits helpers ───────────────────────────────────────────

    #[test]
    fn quant_bits_ranges() {
        assert_eq!(QuantBits::Int8.symmetric_range(), (-128, 127));
        assert_eq!(QuantBits::Int8.asymmetric_range(), (0, 255));
        assert_eq!(QuantBits::Int4.symmetric_range(), (-8, 7));
        assert_eq!(QuantBits::Int4.asymmetric_range(), (0, 15));
    }

    // ── Stats ───────────────────────────────────────────────────────

    #[test]
    fn stats_reflect_calibration() {
        let mut q = TensorQuantizer::new();
        q.calibrate(&[1.0, 2.0]);
        q.calibrate(&[3.0]);
        let stats = q.stats();
        assert_eq!(stats.samples_seen, 3);
        assert!((stats.calibration_min - 1.0).abs() < 1e-15);
        assert!((stats.calibration_max - 3.0).abs() < 1e-15);
    }

    // ── Mixed mode comparison ───────────────────────────────────────

    #[test]
    fn symmetric_vs_asymmetric_error() {
        let mut q = TensorQuantizer::new();
        // Asymmetric data: all positive — asymmetric should use range better
        let values: Vec<f64> = (0..50).map(|i| i as f64 / 50.0).collect();
        q.calibrate(&values);
        let sym_params = q
            .compute_params(QuantMode::Symmetric, QuantBits::Int8)
            .expect("sym");
        let asym_params = q
            .compute_params(QuantMode::Asymmetric, QuantBits::Int8)
            .expect("asym");
        let sym_err = TensorQuantizer::quantization_error(&values, &sym_params);
        let asym_err = TensorQuantizer::quantization_error(&values, &asym_params);
        // For all-positive data, asymmetric should have equal or less error
        assert!(
            asym_err <= sym_err + 1e-10,
            "asym_err={asym_err} sym_err={sym_err}"
        );
    }
}
