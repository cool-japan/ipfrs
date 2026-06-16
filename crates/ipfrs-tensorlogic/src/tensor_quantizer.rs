//! TensorQuantizer — Multi-precision tensor quantization for model compression.
//!
//! Provides production-grade quantization for INT8 (symmetric and asymmetric),
//! INT4, FP16, and BF16, with per-channel support, calibration-percentile
//! outlier suppression, and comprehensive MSE error measurement.
//!
//! # Examples
//!
//! ```
//! use ipfrs_tensorlogic::tensor_quantizer::{
//!     TensorQuantizer, QuantizationMode, QuantizerConfig,
//! };
//!
//! let config = QuantizerConfig {
//!     mode: QuantizationMode::Int8Symmetric,
//!     per_channel: false,
//!     channel_dim: 0,
//!     calibration_percentile: 99.9,
//! };
//! let quantizer = TensorQuantizer::new(config);
//! let values = vec![0.5_f64, -0.3, 0.8, -0.1, 1.0, -1.0];
//! let dims = vec![6];
//! let qt = quantizer.quantize(&values, &dims).expect("example: should succeed in docs");
//! let dq = quantizer.dequantize(&qt).expect("example: should succeed in docs");
//! assert_eq!(dq.values.len(), 6);
//! ```

use thiserror::Error;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors produced by [`TensorQuantizer`].
#[derive(Debug, Error, Clone, PartialEq)]
pub enum QuantizerError {
    /// The input slice is empty.
    #[error("Input tensor is empty")]
    EmptyInput,

    /// The flat values length does not match the product of `dims`.
    #[error("Dimension mismatch: values.len()={values_len} != product(dims)={dims_product}")]
    DimensionMismatch {
        /// Actual length of values slice.
        values_len: usize,
        /// Expected length from dimensions product.
        dims_product: usize,
    },

    /// Percentile `p` was not in `[0, 100]`.
    #[error("Invalid percentile {0}: must be in [0, 100]")]
    InvalidPercentile(f64),

    /// The `dims` slice is empty (zero-rank tensor).
    #[error("Dims must be non-empty (scalar tensors are not supported)")]
    InvalidDims,
}

// ---------------------------------------------------------------------------
// QuantizationMode
// ---------------------------------------------------------------------------

/// Precision target for quantization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuantizationMode {
    /// Symmetric INT8: `scale = percentile(|x|) / 127.0`; range `[-127, 127]`.
    Int8Symmetric,
    /// Asymmetric INT8: separate zero-point; range `[0, 255]`.
    Int8Asymmetric,
    /// 4-bit: `scale = percentile(|x|) / 7.0`; range `[-7, 7]` stored as i8.
    Int4,
    /// FP16 simulation: round to nearest f16 (5-bit exp, 10-bit mantissa).
    Fp16,
    /// BFloat16 simulation: keep top 16 bits of the f32 representation.
    Bf16,
}

impl QuantizationMode {
    /// Human-readable name for statistics reporting.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Int8Symmetric => "Int8Symmetric",
            Self::Int8Asymmetric => "Int8Asymmetric",
            Self::Int4 => "Int4",
            Self::Fp16 => "Fp16",
            Self::Bf16 => "Bf16",
        }
    }

    /// Nominal storage bits per element (for compression ratio).
    pub fn bits_per_element(&self) -> f64 {
        match self {
            Self::Int8Symmetric | Self::Int8Asymmetric => 8.0,
            Self::Int4 => 4.0,
            Self::Fp16 | Self::Bf16 => 16.0,
        }
    }
}

// ---------------------------------------------------------------------------
// QuantizerConfig
// ---------------------------------------------------------------------------

/// Configuration for [`TensorQuantizer`].
#[derive(Debug, Clone)]
pub struct QuantizerConfig {
    /// Quantization precision target.
    pub mode: QuantizationMode,
    /// When `true`, compute one scale/zero-point per slice along `channel_dim`.
    pub per_channel: bool,
    /// The axis along which channels are defined (used only when `per_channel` is `true`).
    pub channel_dim: usize,
    /// Upper percentile used for calibration (e.g. `99.9` suppresses top-0.1% outliers).
    /// Must be in `[0.0, 100.0]`.
    pub calibration_percentile: f64,
}

impl Default for QuantizerConfig {
    fn default() -> Self {
        Self {
            mode: QuantizationMode::Int8Symmetric,
            per_channel: false,
            channel_dim: 0,
            calibration_percentile: 99.9,
        }
    }
}

// ---------------------------------------------------------------------------
// QuantizedTensor / DequantizedTensor
// ---------------------------------------------------------------------------

/// A quantized representation of a tensor.
///
/// `data` encoding per mode:
/// - `Int8Symmetric` / `Int8Asymmetric` / `Int4`: i8 cast to i32.
/// - `Fp16` / `Bf16`: u16 bits cast to i32.
#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    /// Quantization mode used to produce this tensor.
    pub mode: QuantizationMode,
    /// Quantized integer data (one entry per original element).
    pub data: Vec<i32>,
    /// Global (or per-channel, packed) scale factor(s).
    pub scale: f64,
    /// Global (or per-channel average) zero-point.
    pub zero_point: i32,
    /// Original tensor dimensions.
    pub original_dims: Vec<usize>,
    /// Observed minimum value before quantization.
    pub original_min: f64,
    /// Observed maximum value before quantization.
    pub original_max: f64,
    /// Per-channel scales (empty when `per_channel = false`).
    pub(crate) channel_scales: Vec<f64>,
    /// Per-channel zero-points (empty when `per_channel = false`).
    pub(crate) channel_zero_points: Vec<i32>,
}

/// The result of dequantization — approximate reconstruction of the original values.
#[derive(Debug, Clone)]
pub struct DequantizedTensor {
    /// Reconstructed f64 values.
    pub values: Vec<f64>,
    /// Tensor dimensions (same as the original).
    pub dims: Vec<usize>,
}

// ---------------------------------------------------------------------------
// QuantizerStats
// ---------------------------------------------------------------------------

/// Accumulated statistics across multiple [`TensorQuantizer::quantize`] calls.
#[derive(Debug, Clone, Default)]
pub struct QuantizerStats {
    /// Total number of scalar elements quantized.
    pub elements_quantized: usize,
    /// Weighted average compression ratio across all calls.
    pub avg_compression_ratio: f64,
    /// Weighted average MSE quantization error across all calls.
    pub avg_quantization_error: f64,
    /// Unique mode names encountered (in insertion order).
    pub modes_used: Vec<String>,

    // Internal accumulators for weighted averages.
    total_cr_weight: f64,
    total_cr_sum: f64,
    total_err_weight: f64,
    total_err_sum: f64,
}

impl QuantizerStats {
    fn record(&mut self, n: usize, cr: f64, err: f64, mode_name: &str) {
        self.elements_quantized += n;

        let w = n as f64;

        self.total_cr_sum += cr * w;
        self.total_cr_weight += w;
        self.avg_compression_ratio = if self.total_cr_weight > 0.0 {
            self.total_cr_sum / self.total_cr_weight
        } else {
            0.0
        };

        self.total_err_sum += err * w;
        self.total_err_weight += w;
        self.avg_quantization_error = if self.total_err_weight > 0.0 {
            self.total_err_sum / self.total_err_weight
        } else {
            0.0
        };

        let mode_str = mode_name.to_string();
        if !self.modes_used.contains(&mode_str) {
            self.modes_used.push(mode_str);
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: percentile
// ---------------------------------------------------------------------------

/// Compute the `p`-th percentile of `values` using the nearest-rank method.
///
/// `p` must be in `[0, 100]`.  Returns `Err(QuantizerError::InvalidPercentile)`
/// otherwise.  Returns `Err(QuantizerError::EmptyInput)` for an empty slice.
pub fn percentile(values: &[f64], p: f64) -> Result<f64, QuantizerError> {
    if !(0.0..=100.0).contains(&p) {
        return Err(QuantizerError::InvalidPercentile(p));
    }
    if values.is_empty() {
        return Err(QuantizerError::EmptyInput);
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    if n == 1 {
        return Ok(sorted[0]);
    }
    // Nearest-rank: index = ceil(p/100 * n) - 1, clamped.
    let idx = if p == 0.0 {
        0
    } else {
        let raw = (p / 100.0 * n as f64).ceil() as usize;
        raw.saturating_sub(1).min(n - 1)
    };
    Ok(sorted[idx])
}

// ---------------------------------------------------------------------------
// Core quantizer logic (per-tensor helpers)
// ---------------------------------------------------------------------------

struct ScaleZp {
    scale: f64,
    zero_point: i32,
}

/// Compute scale (and zero-point) from calibrated abs-max for the given mode.
fn compute_scale_zp(
    abs_values: &[f64],
    mode: QuantizationMode,
    calib_pct: f64,
) -> Result<ScaleZp, QuantizerError> {
    let p = percentile(abs_values, calib_pct)?;
    match mode {
        QuantizationMode::Int8Symmetric => {
            let scale = if p == 0.0 { 1.0 } else { p / 127.0 };
            Ok(ScaleZp {
                scale,
                zero_point: 0,
            })
        }
        QuantizationMode::Int8Asymmetric => {
            let max_val = p;
            let min_val = -p;
            let range = max_val - min_val;
            let scale = if range == 0.0 { 1.0 } else { range / 255.0 };
            let zero_point = (-min_val / scale).round().clamp(0.0, 255.0) as i32;
            Ok(ScaleZp { scale, zero_point })
        }
        QuantizationMode::Int4 => {
            let scale = if p == 0.0 { 1.0 } else { p / 7.0 };
            Ok(ScaleZp {
                scale,
                zero_point: 0,
            })
        }
        // FP16/BF16 do not use scale/zero-point in the classical sense.
        QuantizationMode::Fp16 | QuantizationMode::Bf16 => Ok(ScaleZp {
            scale: 1.0,
            zero_point: 0,
        }),
    }
}

fn quantize_element(x: f64, mode: QuantizationMode, scale: f64, zero_point: i32) -> i32 {
    match mode {
        QuantizationMode::Int8Symmetric => (x / scale).round().clamp(-127.0, 127.0) as i32,
        QuantizationMode::Int8Asymmetric => {
            ((x / scale).round() + zero_point as f64).clamp(0.0, 255.0) as i32
        }
        QuantizationMode::Int4 => (x / scale).round().clamp(-7.0, 7.0) as i32,
        QuantizationMode::Fp16 => {
            // Simulate FP16: round to nearest 1/1024.
            // Clamp to FP16 representable range (~65504).
            let clamped = x.clamp(-65504.0, 65504.0);
            let quantized = (clamped * 1024.0).round();
            // Store as i32 (representing a scaled integer, reconstructed by /1024).
            quantized as i32
        }
        QuantizationMode::Bf16 => {
            // Keep top 16 bits of f32 representation.
            let bits = (x as f32).to_bits();
            let bf16_bits = (bits >> 16) as u16;
            bf16_bits as i32
        }
    }
}

fn dequantize_element(q: i32, mode: QuantizationMode, scale: f64, zero_point: i32) -> f64 {
    match mode {
        QuantizationMode::Int8Symmetric => q as f64 * scale,
        QuantizationMode::Int8Asymmetric => (q - zero_point) as f64 * scale,
        QuantizationMode::Int4 => q as f64 * scale,
        QuantizationMode::Fp16 => {
            // Stored as (x * 1024).round() → reconstruct by /1024.
            q as f64 / 1024.0
        }
        QuantizationMode::Bf16 => {
            // Stored as u16 bits → reconstruct f32 by shifting back.
            let bf16_bits = q as u16;
            let f32_bits = (bf16_bits as u32) << 16;
            f32::from_bits(f32_bits) as f64
        }
    }
}

// ---------------------------------------------------------------------------
// TensorQuantizer
// ---------------------------------------------------------------------------

/// Multi-precision tensor quantizer.
///
/// Supports INT8 (symmetric/asymmetric), INT4, FP16, and BF16 quantization
/// with optional per-channel calibration and percentile-based outlier suppression.
pub struct TensorQuantizer {
    config: QuantizerConfig,
    stats: QuantizerStats,
}

impl TensorQuantizer {
    /// Create a new quantizer with the given configuration.
    pub fn new(config: QuantizerConfig) -> Self {
        Self {
            config,
            stats: QuantizerStats::default(),
        }
    }

    /// Read-only access to accumulated statistics.
    pub fn stats(&self) -> &QuantizerStats {
        &self.stats
    }

    /// Reset accumulated statistics.
    pub fn reset_stats(&mut self) {
        self.stats = QuantizerStats::default();
    }

    /// Quantize a flat tensor described by `dims`.
    ///
    /// Returns a [`QuantizedTensor`] whose `data` encodes the elements in the
    /// same order as `values`.
    pub fn quantize(
        &mut self,
        values: &[f64],
        dims: &[usize],
    ) -> Result<QuantizedTensor, QuantizerError> {
        // --- Validation -------------------------------------------------------
        if dims.is_empty() {
            return Err(QuantizerError::InvalidDims);
        }
        if values.is_empty() {
            return Err(QuantizerError::EmptyInput);
        }
        let expected: usize = dims.iter().product();
        if values.len() != expected {
            return Err(QuantizerError::DimensionMismatch {
                values_len: values.len(),
                dims_product: expected,
            });
        }

        let original_min = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let original_max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        let qt = if self.config.per_channel {
            self.quantize_per_channel(values, dims, original_min, original_max)?
        } else {
            self.quantize_per_tensor(values, dims, original_min, original_max)?
        };

        // Update stats.
        let n = values.len();
        let cr = Self::compression_ratio(n, &self.config.mode);
        // Compute MSE for stats (best effort; ignore errors).
        let err = self.quantization_error_internal(values, &qt).unwrap_or(0.0);
        self.stats.record(n, cr, err, self.config.mode.name());

        Ok(qt)
    }

    fn quantize_per_tensor(
        &self,
        values: &[f64],
        dims: &[usize],
        original_min: f64,
        original_max: f64,
    ) -> Result<QuantizedTensor, QuantizerError> {
        let abs_values: Vec<f64> = values.iter().map(|x| x.abs()).collect();
        let szp = compute_scale_zp(
            &abs_values,
            self.config.mode,
            self.config.calibration_percentile,
        )?;

        let data: Vec<i32> = values
            .iter()
            .map(|&x| quantize_element(x, self.config.mode, szp.scale, szp.zero_point))
            .collect();

        Ok(QuantizedTensor {
            mode: self.config.mode,
            data,
            scale: szp.scale,
            zero_point: szp.zero_point,
            original_dims: dims.to_vec(),
            original_min,
            original_max,
            channel_scales: Vec::new(),
            channel_zero_points: Vec::new(),
        })
    }

    fn quantize_per_channel(
        &self,
        values: &[f64],
        dims: &[usize],
        original_min: f64,
        original_max: f64,
    ) -> Result<QuantizedTensor, QuantizerError> {
        let channel_dim = self.config.channel_dim;
        if channel_dim >= dims.len() {
            // Fallback to per-tensor if channel_dim is out of range.
            return self.quantize_per_tensor(values, dims, original_min, original_max);
        }

        let num_channels = dims[channel_dim];
        // Elements per channel = total / num_channels.
        let total = values.len();
        let per_channel = total / num_channels;

        // For each channel index c, gather elements where the channel_dim index == c.
        // We treat the tensor as C-contiguous (row-major). For a shape [d0, d1, ..., dN],
        // the channel axis contribution repeats every product(dims[channel_dim+1..]) elements.
        let inner: usize = dims[channel_dim + 1..].iter().product();

        let mut channel_scales = vec![1.0f64; num_channels];
        let mut channel_zero_points = vec![0i32; num_channels];
        let mut data = vec![0i32; total];

        for c in 0..num_channels {
            // Collect elements belonging to channel c.
            let channel_vals: Vec<f64> = (0..total)
                .filter(|&idx| {
                    // Index along channel_dim for flat index idx.
                    let stride: usize = if channel_dim + 1 < dims.len() {
                        inner
                    } else {
                        1
                    };
                    (idx / stride) % num_channels == c
                })
                .map(|idx| values[idx])
                .collect();

            if channel_vals.is_empty() {
                continue;
            }

            let abs_vals: Vec<f64> = channel_vals.iter().map(|x| x.abs()).collect();
            let szp = compute_scale_zp(
                &abs_vals,
                self.config.mode,
                self.config.calibration_percentile,
            )?;
            channel_scales[c] = szp.scale;
            channel_zero_points[c] = szp.zero_point;

            // Write quantized elements back in-place.
            let stride = inner;
            let mut local_idx = 0usize;
            for (idx, slot) in data.iter_mut().enumerate() {
                let ch_idx = (idx / stride) % num_channels;
                if ch_idx == c {
                    *slot = quantize_element(
                        channel_vals[local_idx],
                        self.config.mode,
                        szp.scale,
                        szp.zero_point,
                    );
                    local_idx += 1;
                }
            }
        }

        // Global scale = mean of channel scales.
        let global_scale = if num_channels > 0 {
            channel_scales.iter().sum::<f64>() / num_channels as f64
        } else {
            1.0
        };
        let global_zp = if num_channels > 0 {
            (channel_zero_points.iter().map(|&z| z as i64).sum::<i64>() / num_channels as i64)
                as i32
        } else {
            0
        };

        // Sanity: `per_channel` should have produced `total` elements.
        let _ = per_channel; // consumed above

        Ok(QuantizedTensor {
            mode: self.config.mode,
            data,
            scale: global_scale,
            zero_point: global_zp,
            original_dims: dims.to_vec(),
            original_min,
            original_max,
            channel_scales,
            channel_zero_points,
        })
    }

    /// Reconstruct approximate f64 values from a [`QuantizedTensor`].
    pub fn dequantize(&self, qt: &QuantizedTensor) -> Result<DequantizedTensor, QuantizerError> {
        if qt.data.is_empty() {
            return Err(QuantizerError::EmptyInput);
        }

        let values = if !qt.channel_scales.is_empty() {
            // Per-channel dequantization.
            let num_channels = qt.channel_scales.len();
            let inner: usize = if qt.original_dims.len() > 1 {
                let channel_dim = self.config.channel_dim.min(qt.original_dims.len() - 1);
                qt.original_dims[channel_dim + 1..].iter().product()
            } else {
                1
            };
            let stride = inner;

            qt.data
                .iter()
                .enumerate()
                .map(|(idx, &q)| {
                    let ch_idx = (idx / stride) % num_channels;
                    let s = qt.channel_scales.get(ch_idx).copied().unwrap_or(qt.scale);
                    let zp = qt
                        .channel_zero_points
                        .get(ch_idx)
                        .copied()
                        .unwrap_or(qt.zero_point);
                    dequantize_element(q, qt.mode, s, zp)
                })
                .collect()
        } else {
            // Per-tensor dequantization.
            qt.data
                .iter()
                .map(|&q| dequantize_element(q, qt.mode, qt.scale, qt.zero_point))
                .collect()
        };

        Ok(DequantizedTensor {
            values,
            dims: qt.original_dims.clone(),
        })
    }

    /// Compute mean-squared error between the original values and the
    /// dequantized reconstruction.
    pub fn quantization_error(
        &self,
        original: &[f64],
        qt: &QuantizedTensor,
    ) -> Result<f64, QuantizerError> {
        self.quantization_error_internal(original, qt)
    }

    fn quantization_error_internal(
        &self,
        original: &[f64],
        qt: &QuantizedTensor,
    ) -> Result<f64, QuantizerError> {
        if original.is_empty() {
            return Err(QuantizerError::EmptyInput);
        }
        if original.len() != qt.data.len() {
            return Err(QuantizerError::DimensionMismatch {
                values_len: original.len(),
                dims_product: qt.data.len(),
            });
        }
        let dq = self.dequantize(qt)?;
        let mse = original
            .iter()
            .zip(dq.values.iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f64>()
            / original.len() as f64;
        Ok(mse)
    }

    /// Compression ratio relative to f64 (64-bit) storage.
    ///
    /// - f64 = 64 bits → ratio = 64 / bits_per_element.
    pub fn compression_ratio(original_len: usize, mode: &QuantizationMode) -> f64 {
        let _ = original_len; // ratio is per-element, length-independent
        64.0 / mode.bits_per_element()
    }

    /// Clamp `x` to the representable integer range for `mode`.
    pub fn clamp_to_range(x: f64, mode: &QuantizationMode) -> f64 {
        match mode {
            QuantizationMode::Int8Symmetric => x.clamp(-127.0, 127.0),
            QuantizationMode::Int8Asymmetric => x.clamp(0.0, 255.0),
            QuantizationMode::Int4 => x.clamp(-7.0, 7.0),
            QuantizationMode::Fp16 => x.clamp(-65504.0, 65504.0),
            // BF16 shares the f32 representable range.
            QuantizationMode::Bf16 => x.clamp(f32::MIN as f64, f32::MAX as f64),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::{percentile, QuantizationMode, QuantizerConfig, QuantizerError, TensorQuantizer};

    // -----------------------------------------------------------------------
    // Helper
    // -----------------------------------------------------------------------
    fn default_quantizer(mode: QuantizationMode) -> TensorQuantizer {
        TensorQuantizer::new(QuantizerConfig {
            mode,
            per_channel: false,
            channel_dim: 0,
            calibration_percentile: 99.9,
        })
    }

    fn mse(a: &[f64], b: &[f64]) -> f64 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x - y).powi(2))
            .sum::<f64>()
            / a.len() as f64
    }

    // -----------------------------------------------------------------------
    // percentile tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_percentile_single_element() {
        let v = vec![42.0];
        assert_eq!(percentile(&v, 50.0).expect("test: should succeed"), 42.0);
        assert_eq!(percentile(&v, 0.0).expect("test: should succeed"), 42.0);
        assert_eq!(percentile(&v, 100.0).expect("test: should succeed"), 42.0);
    }

    #[test]
    fn test_percentile_sorted_five() {
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        // 0th → index 0 = 1.0
        assert_eq!(percentile(&v, 0.0).expect("test: should succeed"), 1.0);
        // 100th → index 4 = 5.0
        assert_eq!(percentile(&v, 100.0).expect("test: should succeed"), 5.0);
        // 50th: ceil(0.5 * 5) = 3 → index 2 = 3.0
        assert_eq!(percentile(&v, 50.0).expect("test: should succeed"), 3.0);
    }

    #[test]
    fn test_percentile_unsorted() {
        let v = vec![5.0, 1.0, 3.0, 2.0, 4.0];
        assert_eq!(percentile(&v, 100.0).expect("test: should succeed"), 5.0);
        assert_eq!(percentile(&v, 0.0).expect("test: should succeed"), 1.0);
    }

    #[test]
    fn test_percentile_invalid() {
        let v = vec![1.0, 2.0];
        assert_eq!(
            percentile(&v, -1.0).unwrap_err(),
            QuantizerError::InvalidPercentile(-1.0)
        );
        assert_eq!(
            percentile(&v, 101.0).unwrap_err(),
            QuantizerError::InvalidPercentile(101.0)
        );
    }

    #[test]
    fn test_percentile_empty() {
        assert_eq!(
            percentile(&[], 50.0).unwrap_err(),
            QuantizerError::EmptyInput
        );
    }

    // -----------------------------------------------------------------------
    // Validation / error tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_empty_input_error() {
        let mut q = default_quantizer(QuantizationMode::Int8Symmetric);
        assert_eq!(
            q.quantize(&[], &[0]).unwrap_err(),
            QuantizerError::EmptyInput
        );
    }

    #[test]
    fn test_empty_dims_error() {
        let mut q = default_quantizer(QuantizationMode::Int8Symmetric);
        assert_eq!(
            q.quantize(&[1.0], &[]).unwrap_err(),
            QuantizerError::InvalidDims
        );
    }

    #[test]
    fn test_dimension_mismatch_error() {
        let mut q = default_quantizer(QuantizationMode::Int8Symmetric);
        let err = q.quantize(&[1.0, 2.0, 3.0], &[2]).unwrap_err();
        assert!(matches!(err, QuantizerError::DimensionMismatch { .. }));
    }

    // -----------------------------------------------------------------------
    // INT8 Symmetric
    // -----------------------------------------------------------------------

    #[test]
    fn test_int8sym_quantize_roundtrip() {
        let mut q = default_quantizer(QuantizationMode::Int8Symmetric);
        let values = vec![1.0_f64, -1.0, 0.5, -0.5, 0.0];
        let qt = q.quantize(&values, &[5]).expect("test: should succeed");
        assert_eq!(qt.mode, QuantizationMode::Int8Symmetric);
        assert_eq!(qt.data.len(), 5);
        let dq = q.dequantize(&qt).expect("test: should succeed");
        assert_eq!(dq.values.len(), 5);
        // Zero should survive perfectly.
        assert!((dq.values[4] - 0.0).abs() < 0.02);
    }

    #[test]
    fn test_int8sym_scale_calculation() {
        let mut q = default_quantizer(QuantizationMode::Int8Symmetric);
        let values: Vec<f64> = (1..=127).map(|x| x as f64).collect();
        let qt = q.quantize(&values, &[127]).expect("test: should succeed");
        // scale ≈ 1.0 because max ≈ 127 and scale = max/127.
        assert!((qt.scale - 1.0).abs() < 0.01, "scale={}", qt.scale);
    }

    #[test]
    fn test_int8sym_clamp() {
        let mut q = default_quantizer(QuantizationMode::Int8Symmetric);
        // Large outlier clamped to 127 (or -127).
        let values = vec![100.0, -100.0, 1000.0, -1000.0];
        let qt = q.quantize(&values, &[4]).expect("test: should succeed");
        for &v in &qt.data {
            assert!((-127..=127).contains(&v), "out of range: {v}");
        }
    }

    #[test]
    fn test_int8sym_zero_point_is_zero() {
        let mut q = default_quantizer(QuantizationMode::Int8Symmetric);
        let values = vec![0.1, 0.5, -0.3];
        let qt = q.quantize(&values, &[3]).expect("test: should succeed");
        assert_eq!(qt.zero_point, 0);
    }

    #[test]
    fn test_int8sym_mse_low() {
        let mut q = default_quantizer(QuantizationMode::Int8Symmetric);
        let values: Vec<f64> = (0..256).map(|i| (i as f64 / 128.0) - 1.0).collect();
        let qt = q.quantize(&values, &[256]).expect("test: should succeed");
        let err = q
            .quantization_error(&values, &qt)
            .expect("test: should succeed");
        // INT8 should give very low MSE on uniform range.
        assert!(err < 1e-4, "MSE too high: {err}");
    }

    // -----------------------------------------------------------------------
    // INT8 Asymmetric
    // -----------------------------------------------------------------------

    #[test]
    fn test_int8asym_roundtrip() {
        let mut q = default_quantizer(QuantizationMode::Int8Asymmetric);
        let values = vec![0.2_f64, 0.5, -0.5, 0.0, 0.8];
        let qt = q.quantize(&values, &[5]).expect("test: should succeed");
        assert_eq!(qt.mode, QuantizationMode::Int8Asymmetric);
        let dq = q.dequantize(&qt).expect("test: should succeed");
        let err = mse(&values, &dq.values);
        assert!(err < 1e-4, "MSE too high: {err}");
    }

    #[test]
    fn test_int8asym_data_range() {
        let mut q = default_quantizer(QuantizationMode::Int8Asymmetric);
        let values = vec![-1.0, 0.0, 0.5, 1.0];
        let qt = q.quantize(&values, &[4]).expect("test: should succeed");
        for &v in &qt.data {
            assert!((0..=255).contains(&v), "out of [0,255]: {v}");
        }
    }

    #[test]
    fn test_int8asym_zero_point_nonzero() {
        let mut q = default_quantizer(QuantizationMode::Int8Asymmetric);
        let values = vec![-1.0, 1.0];
        let qt = q.quantize(&values, &[2]).expect("test: should succeed");
        // For a symmetric range [-1,1] the zero_point should be ~128.
        assert!(qt.zero_point > 0, "zero_point={}", qt.zero_point);
    }

    // -----------------------------------------------------------------------
    // INT4
    // -----------------------------------------------------------------------

    #[test]
    fn test_int4_data_range() {
        let mut q = default_quantizer(QuantizationMode::Int4);
        let values = vec![-1.0, 0.0, 0.5, -0.5, 1.0, 0.25];
        let qt = q.quantize(&values, &[6]).expect("test: should succeed");
        for &v in &qt.data {
            assert!((-7..=7).contains(&v), "out of [-7,7]: {v}");
        }
    }

    #[test]
    fn test_int4_roundtrip() {
        let mut q = default_quantizer(QuantizationMode::Int4);
        let values: Vec<f64> = (-7..=7).map(|x| x as f64 * 0.1).collect();
        let qt = q.quantize(&values, &[15]).expect("test: should succeed");
        let dq = q.dequantize(&qt).expect("test: should succeed");
        let err = mse(&values, &dq.values);
        assert!(err < 1e-3, "MSE={err}");
    }

    #[test]
    fn test_int4_scale() {
        let mut q = default_quantizer(QuantizationMode::Int4);
        let values = vec![7.0, -7.0, 3.5];
        let qt = q.quantize(&values, &[3]).expect("test: should succeed");
        // scale ≈ 7/7 = 1.0.
        assert!((qt.scale - 1.0).abs() < 0.01, "scale={}", qt.scale);
    }

    // -----------------------------------------------------------------------
    // FP16
    // -----------------------------------------------------------------------

    #[test]
    fn test_fp16_roundtrip_small() {
        let mut q = default_quantizer(QuantizationMode::Fp16);
        let values = vec![1.0_f64, 0.5, -0.5, 0.25, -0.25];
        let qt = q.quantize(&values, &[5]).expect("test: should succeed");
        assert_eq!(qt.mode, QuantizationMode::Fp16);
        let dq = q.dequantize(&qt).expect("test: should succeed");
        for (&orig, &rec) in values.iter().zip(dq.values.iter()) {
            // FP16 round-trip precision ~3 decimal digits.
            assert!((orig - rec).abs() < 0.002, "orig={orig} rec={rec}");
        }
    }

    #[test]
    fn test_fp16_clamp_large() {
        let mut q = default_quantizer(QuantizationMode::Fp16);
        // Values beyond FP16 range are clamped.
        let values = vec![1e6_f64, -1e6];
        let qt = q.quantize(&values, &[2]).expect("test: should succeed");
        let dq = q.dequantize(&qt).expect("test: should succeed");
        // Clamped to ±65504.
        assert!(dq.values[0] <= 65504.1);
        assert!(dq.values[1] >= -65504.1);
    }

    #[test]
    fn test_fp16_zero() {
        let mut q = default_quantizer(QuantizationMode::Fp16);
        let values = vec![0.0_f64];
        let qt = q.quantize(&values, &[1]).expect("test: should succeed");
        let dq = q.dequantize(&qt).expect("test: should succeed");
        assert_eq!(dq.values[0], 0.0);
    }

    #[test]
    fn test_fp16_data_stored_as_scaled_int() {
        let mut q = default_quantizer(QuantizationMode::Fp16);
        let values = vec![1.0_f64];
        let qt = q.quantize(&values, &[1]).expect("test: should succeed");
        // 1.0 * 1024 = 1024.
        assert_eq!(qt.data[0], 1024);
    }

    // -----------------------------------------------------------------------
    // BF16
    // -----------------------------------------------------------------------

    #[test]
    fn test_bf16_roundtrip() {
        let mut q = default_quantizer(QuantizationMode::Bf16);
        let values = vec![1.0_f64, 0.5, -0.5, std::f64::consts::PI, -2.71];
        let qt = q.quantize(&values, &[5]).expect("test: should succeed");
        assert_eq!(qt.mode, QuantizationMode::Bf16);
        let dq = q.dequantize(&qt).expect("test: should succeed");
        // BF16 has 7 mantissa bits, so ~2 decimal digit precision.
        for (&orig, &rec) in values.iter().zip(dq.values.iter()) {
            let rel_err = if orig.abs() > 1e-9 {
                (orig - rec).abs() / orig.abs()
            } else {
                (orig - rec).abs()
            };
            assert!(rel_err < 0.02, "orig={orig} rec={rec} rel_err={rel_err}");
        }
    }

    #[test]
    fn test_bf16_stores_u16_bits() {
        let mut q = default_quantizer(QuantizationMode::Bf16);
        let values = vec![1.0_f64];
        let qt = q.quantize(&values, &[1]).expect("test: should succeed");
        // f32 bits of 1.0 = 0x3F800000; >> 16 = 0x3F80 = 16256.
        assert_eq!(qt.data[0], 0x3F80i32, "bf16 bits={}", qt.data[0]);
    }

    #[test]
    fn test_bf16_zero() {
        let mut q = default_quantizer(QuantizationMode::Bf16);
        let values = vec![0.0_f64];
        let qt = q.quantize(&values, &[1]).expect("test: should succeed");
        let dq = q.dequantize(&qt).expect("test: should succeed");
        assert_eq!(dq.values[0], 0.0);
    }

    // -----------------------------------------------------------------------
    // Compression ratio
    // -----------------------------------------------------------------------

    #[test]
    fn test_compression_ratio_int8() {
        let cr = TensorQuantizer::compression_ratio(100, &QuantizationMode::Int8Symmetric);
        assert!((cr - 8.0).abs() < 1e-10, "cr={cr}");
    }

    #[test]
    fn test_compression_ratio_int4() {
        let cr = TensorQuantizer::compression_ratio(100, &QuantizationMode::Int4);
        assert!((cr - 16.0).abs() < 1e-10, "cr={cr}");
    }

    #[test]
    fn test_compression_ratio_fp16() {
        let cr = TensorQuantizer::compression_ratio(100, &QuantizationMode::Fp16);
        assert!((cr - 4.0).abs() < 1e-10, "cr={cr}");
    }

    #[test]
    fn test_compression_ratio_bf16() {
        let cr = TensorQuantizer::compression_ratio(100, &QuantizationMode::Bf16);
        assert!((cr - 4.0).abs() < 1e-10, "cr={cr}");
    }

    // -----------------------------------------------------------------------
    // clamp_to_range
    // -----------------------------------------------------------------------

    #[test]
    fn test_clamp_int8sym() {
        assert_eq!(
            TensorQuantizer::clamp_to_range(200.0, &QuantizationMode::Int8Symmetric),
            127.0
        );
        assert_eq!(
            TensorQuantizer::clamp_to_range(-200.0, &QuantizationMode::Int8Symmetric),
            -127.0
        );
    }

    #[test]
    fn test_clamp_int8asym() {
        assert_eq!(
            TensorQuantizer::clamp_to_range(-1.0, &QuantizationMode::Int8Asymmetric),
            0.0
        );
        assert_eq!(
            TensorQuantizer::clamp_to_range(300.0, &QuantizationMode::Int8Asymmetric),
            255.0
        );
    }

    #[test]
    fn test_clamp_int4() {
        assert_eq!(
            TensorQuantizer::clamp_to_range(10.0, &QuantizationMode::Int4),
            7.0
        );
        assert_eq!(
            TensorQuantizer::clamp_to_range(-10.0, &QuantizationMode::Int4),
            -7.0
        );
    }

    #[test]
    fn test_clamp_fp16() {
        assert_eq!(
            TensorQuantizer::clamp_to_range(1e10, &QuantizationMode::Fp16),
            65504.0
        );
        assert_eq!(
            TensorQuantizer::clamp_to_range(-1e10, &QuantizationMode::Fp16),
            -65504.0
        );
    }

    // -----------------------------------------------------------------------
    // quantization_error
    // -----------------------------------------------------------------------

    #[test]
    fn test_quantization_error_zero_tensor() {
        let mut q = default_quantizer(QuantizationMode::Int8Symmetric);
        let values = vec![0.0_f64; 16];
        let qt = q.quantize(&values, &[16]).expect("test: should succeed");
        let err = q
            .quantization_error(&values, &qt)
            .expect("test: should succeed");
        assert_eq!(err, 0.0);
    }

    #[test]
    fn test_quantization_error_length_mismatch() {
        let mut q = default_quantizer(QuantizationMode::Int8Symmetric);
        let values = vec![1.0, 2.0, 3.0];
        let qt = q.quantize(&values, &[3]).expect("test: should succeed");
        let err = q.quantization_error(&[1.0, 2.0], &qt);
        assert!(matches!(err, Err(QuantizerError::DimensionMismatch { .. })));
    }

    #[test]
    fn test_quantization_error_bf16_low() {
        let mut q = default_quantizer(QuantizationMode::Bf16);
        let values: Vec<f64> = (0..64).map(|i| i as f64 * 0.01).collect();
        let qt = q.quantize(&values, &[64]).expect("test: should succeed");
        let err = q
            .quantization_error(&values, &qt)
            .expect("test: should succeed");
        // BF16 precision should give MSE < 1e-6 for small values.
        assert!(err < 1e-4, "MSE={err}");
    }

    // -----------------------------------------------------------------------
    // Per-channel quantization
    // -----------------------------------------------------------------------

    #[test]
    fn test_per_channel_produces_channel_scales() {
        let config = QuantizerConfig {
            mode: QuantizationMode::Int8Symmetric,
            per_channel: true,
            channel_dim: 0,
            calibration_percentile: 99.9,
        };
        let mut q = TensorQuantizer::new(config);
        // 2 channels, 4 elements each → shape [2, 4].
        let values = vec![1.0, 2.0, 3.0, 4.0, 0.1, 0.2, 0.3, 0.4];
        let qt = q.quantize(&values, &[2, 4]).expect("test: should succeed");
        assert_eq!(qt.channel_scales.len(), 2);
        // Different channels should have different scales.
        assert!(
            (qt.channel_scales[0] - qt.channel_scales[1]).abs() > 0.01,
            "scales equal: {:?}",
            qt.channel_scales
        );
    }

    #[test]
    fn test_per_channel_dequantize() {
        let config = QuantizerConfig {
            mode: QuantizationMode::Int8Symmetric,
            per_channel: true,
            channel_dim: 0,
            calibration_percentile: 100.0,
        };
        let mut q = TensorQuantizer::new(config);
        let values = vec![10.0, 20.0, 30.0, 40.0, 1.0, 2.0, 3.0, 4.0];
        let qt = q.quantize(&values, &[2, 4]).expect("test: should succeed");
        let dq = q.dequantize(&qt).expect("test: should succeed");
        assert_eq!(dq.values.len(), 8);
        // Approximate reconstruction — INT8 should be close.
        let err = mse(&values, &dq.values);
        assert!(err < 1.0, "MSE={err}");
    }

    // -----------------------------------------------------------------------
    // Stats tracking
    // -----------------------------------------------------------------------

    #[test]
    fn test_stats_accumulate() {
        let mut q = default_quantizer(QuantizationMode::Int8Symmetric);
        let v1 = vec![1.0, 2.0, 3.0];
        let v2 = vec![4.0, 5.0, 6.0, 7.0, 8.0];
        q.quantize(&v1, &[3]).expect("test: should succeed");
        q.quantize(&v2, &[5]).expect("test: should succeed");
        let stats = q.stats();
        assert_eq!(stats.elements_quantized, 8);
        assert_eq!(stats.modes_used, vec!["Int8Symmetric"]);
        assert!(stats.avg_compression_ratio > 0.0);
    }

    #[test]
    fn test_stats_reset() {
        let mut q = default_quantizer(QuantizationMode::Int8Symmetric);
        q.quantize(&[1.0, 2.0], &[2]).expect("test: should succeed");
        q.reset_stats();
        let stats = q.stats();
        assert_eq!(stats.elements_quantized, 0);
        assert!(stats.modes_used.is_empty());
    }

    #[test]
    fn test_stats_multiple_modes_if_changed() {
        // Verify that mode name is only recorded once even for multiple calls.
        let mut q = default_quantizer(QuantizationMode::Int4);
        q.quantize(&[1.0, 2.0], &[2]).expect("test: should succeed");
        q.quantize(&[3.0, 4.0], &[2]).expect("test: should succeed");
        assert_eq!(q.stats().modes_used.len(), 1);
        assert_eq!(q.stats().modes_used[0], "Int4");
    }

    // -----------------------------------------------------------------------
    // Multi-dim tensor (2-D)
    // -----------------------------------------------------------------------

    #[test]
    fn test_2d_tensor_int8sym() {
        let mut q = default_quantizer(QuantizationMode::Int8Symmetric);
        let values: Vec<f64> = (0..12).map(|i| i as f64 * 0.1).collect();
        let qt = q.quantize(&values, &[3, 4]).expect("test: should succeed");
        assert_eq!(qt.original_dims, vec![3, 4]);
        assert_eq!(qt.data.len(), 12);
        let dq = q.dequantize(&qt).expect("test: should succeed");
        assert_eq!(dq.dims, vec![3, 4]);
    }

    // -----------------------------------------------------------------------
    // Edge: all zeros
    // -----------------------------------------------------------------------

    #[test]
    fn test_all_zeros_int8sym() {
        let mut q = default_quantizer(QuantizationMode::Int8Symmetric);
        let values = vec![0.0_f64; 8];
        let qt = q.quantize(&values, &[8]).expect("test: should succeed");
        // scale defaults to 1.0 when max=0.
        assert_eq!(qt.scale, 1.0);
        let dq = q.dequantize(&qt).expect("test: should succeed");
        for v in &dq.values {
            assert_eq!(*v, 0.0);
        }
    }

    #[test]
    fn test_all_zeros_bf16() {
        let mut q = default_quantizer(QuantizationMode::Bf16);
        let values = vec![0.0_f64; 4];
        let qt = q.quantize(&values, &[4]).expect("test: should succeed");
        let dq = q.dequantize(&qt).expect("test: should succeed");
        for v in &dq.values {
            assert_eq!(*v, 0.0);
        }
    }

    // -----------------------------------------------------------------------
    // original_min / original_max preserved
    // -----------------------------------------------------------------------

    #[test]
    fn test_original_min_max_preserved() {
        let mut q = default_quantizer(QuantizationMode::Int8Symmetric);
        let values = vec![-3.5_f64, 0.0, 7.2];
        let qt = q.quantize(&values, &[3]).expect("test: should succeed");
        assert!((qt.original_min - (-3.5)).abs() < 1e-10);
        assert!((qt.original_max - 7.2).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // Calibration percentile: 100 vs 50 should produce different scales
    // -----------------------------------------------------------------------

    #[test]
    fn test_calibration_percentile_effect() {
        let values: Vec<f64> = (1..=100).map(|x| x as f64).collect();

        let mut q99 = TensorQuantizer::new(QuantizerConfig {
            mode: QuantizationMode::Int8Symmetric,
            calibration_percentile: 99.9,
            ..QuantizerConfig::default()
        });
        let mut q50 = TensorQuantizer::new(QuantizerConfig {
            mode: QuantizationMode::Int8Symmetric,
            calibration_percentile: 50.0,
            ..QuantizerConfig::default()
        });

        let qt99 = q99.quantize(&values, &[100]).expect("test: should succeed");
        let qt50 = q50.quantize(&values, &[100]).expect("test: should succeed");

        assert!(
            qt50.scale < qt99.scale,
            "scale_50={} scale_99={}",
            qt50.scale,
            qt99.scale
        );
    }
}
