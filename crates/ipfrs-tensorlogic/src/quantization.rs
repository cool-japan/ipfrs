//! Model Quantization Support
//!
//! This module provides comprehensive quantization support for ML models, enabling
//! efficient deployment on edge devices and reducing model size and inference latency.
//!
//! ## Supported Quantization Schemes
//!
//! - **INT8 Quantization**: 8-bit integer quantization with configurable ranges
//! - **INT4 Quantization**: 4-bit integer quantization for extreme compression
//! - **Per-Tensor Quantization**: Single scale/zero-point for entire tensor
//! - **Per-Channel Quantization**: Independent scale/zero-point per channel
//! - **Symmetric Quantization**: Zero-point = 0 (centered around zero)
//! - **Asymmetric Quantization**: Arbitrary zero-point for full range coverage
//! - **Dynamic Quantization**: Runtime quantization of activations
//!
//! ## Examples
//!
//! ```
//! use ipfrs_tensorlogic::{QuantizedTensor, QuantizationScheme, QuantizationConfig};
//!
//! // Per-tensor INT8 symmetric quantization
//! let weights = vec![0.5, -0.3, 0.8, -0.1];
//! let config = QuantizationConfig::int8_symmetric();
//! let quantized = QuantizedTensor::quantize_per_tensor(&weights, vec![4], config).expect("example: should succeed in docs");
//!
//! // Dequantize back to f32
//! let dequantized = quantized.dequantize();
//! assert_eq!(dequantized.len(), 4);
//!
//! // Per-channel quantization for Conv2D weights
//! let weights = vec![0.5, 0.3, -0.2, -0.4, 0.1, 0.6, -0.5, 0.2]; // 2 channels, 4 elements each
//! let config = QuantizationConfig::int8_per_channel(2);
//! let quantized = QuantizedTensor::quantize_per_channel(&weights, vec![2, 4], config).expect("example: should succeed in docs");
//! ```

use serde::{Deserialize, Serialize};
use std::fmt;
use thiserror::Error;

/// Errors that can occur during quantization operations
#[derive(Debug, Error)]
pub enum QuantizationError {
    #[error("Invalid quantization bit width: {0}")]
    InvalidBitWidth(u8),

    #[error("Invalid shape: {0}")]
    InvalidShape(String),

    #[error("Invalid number of channels: expected {expected}, got {got}")]
    InvalidChannelCount { expected: usize, got: usize },

    #[error("Empty tensor cannot be quantized")]
    EmptyTensor,

    #[error("Calibration data required for dynamic quantization")]
    CalibrationRequired,

    #[error("Unsupported quantization scheme: {0}")]
    UnsupportedScheme(String),
}

/// Quantization scheme
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantizationScheme {
    /// 8-bit integer quantization (INT8)
    Int8,
    /// 4-bit integer quantization (INT4)
    Int4,
    /// 16-bit integer quantization (INT16)
    Int16,
}

impl QuantizationScheme {
    /// Get the bit width for this scheme
    pub fn bit_width(&self) -> u8 {
        match self {
            QuantizationScheme::Int4 => 4,
            QuantizationScheme::Int8 => 8,
            QuantizationScheme::Int16 => 16,
        }
    }

    /// Get the quantization range (min, max) for this scheme
    pub fn range(&self, symmetric: bool) -> (i32, i32) {
        match (self, symmetric) {
            (QuantizationScheme::Int4, true) => (-8, 7),
            (QuantizationScheme::Int4, false) => (0, 15),
            (QuantizationScheme::Int8, true) => (-128, 127),
            (QuantizationScheme::Int8, false) => (0, 255),
            (QuantizationScheme::Int16, true) => (-32768, 32767),
            (QuantizationScheme::Int16, false) => (0, 65535),
        }
    }

    /// Calculate compression ratio compared to f32
    pub fn compression_ratio(&self) -> f32 {
        32.0 / self.bit_width() as f32
    }
}

/// Quantization granularity
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantizationGranularity {
    /// Per-tensor quantization (single scale/zero-point)
    PerTensor,
    /// Per-channel quantization (scale/zero-point per output channel)
    PerChannel { num_channels: usize },
    /// Per-group quantization (scale/zero-point per group)
    PerGroup { group_size: usize },
}

/// Quantization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationConfig {
    /// Quantization scheme (INT4, INT8, etc.)
    pub scheme: QuantizationScheme,
    /// Quantization granularity
    pub granularity: QuantizationGranularity,
    /// Use symmetric quantization (zero_point = 0)
    pub symmetric: bool,
    /// Calibration method for determining scale/zero-point
    pub calibration: CalibrationMethod,
}

impl QuantizationConfig {
    /// Create INT8 symmetric per-tensor quantization config
    pub fn int8_symmetric() -> Self {
        Self {
            scheme: QuantizationScheme::Int8,
            granularity: QuantizationGranularity::PerTensor,
            symmetric: true,
            calibration: CalibrationMethod::MinMax,
        }
    }

    /// Create INT8 asymmetric per-tensor quantization config
    pub fn int8_asymmetric() -> Self {
        Self {
            scheme: QuantizationScheme::Int8,
            granularity: QuantizationGranularity::PerTensor,
            symmetric: false,
            calibration: CalibrationMethod::MinMax,
        }
    }

    /// Create INT8 per-channel quantization config
    pub fn int8_per_channel(num_channels: usize) -> Self {
        Self {
            scheme: QuantizationScheme::Int8,
            granularity: QuantizationGranularity::PerChannel { num_channels },
            symmetric: true,
            calibration: CalibrationMethod::MinMax,
        }
    }

    /// Create INT4 symmetric per-tensor quantization config
    pub fn int4_symmetric() -> Self {
        Self {
            scheme: QuantizationScheme::Int4,
            granularity: QuantizationGranularity::PerTensor,
            symmetric: true,
            calibration: CalibrationMethod::MinMax,
        }
    }

    /// Create INT4 per-channel quantization config
    pub fn int4_per_channel(num_channels: usize) -> Self {
        Self {
            scheme: QuantizationScheme::Int4,
            granularity: QuantizationGranularity::PerChannel { num_channels },
            symmetric: true,
            calibration: CalibrationMethod::MinMax,
        }
    }
}

/// Calibration method for determining quantization parameters
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CalibrationMethod {
    /// Min-max calibration (uses actual min/max of data)
    MinMax,
    /// Percentile-based calibration (clips outliers)
    Percentile { lower: u8, upper: u8 },
    /// Entropy-based calibration (minimizes KL divergence)
    Entropy,
    /// MSE-based calibration (minimizes mean squared error)
    Mse,
}

/// Quantization parameters for a single channel/tensor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationParams {
    /// Scale factor: real_value = (quantized_value - zero_point) * scale
    pub scale: f32,
    /// Zero point (quantized value corresponding to real 0.0)
    pub zero_point: i32,
    /// Min value in quantized range
    pub qmin: i32,
    /// Max value in quantized range
    pub qmax: i32,
}

impl QuantizationParams {
    /// Create quantization parameters from min/max values
    pub fn from_min_max(
        min_val: f32,
        max_val: f32,
        scheme: QuantizationScheme,
        symmetric: bool,
    ) -> Self {
        let (qmin, qmax) = scheme.range(symmetric);

        let (scale, zero_point) = if symmetric {
            // Symmetric: zero_point = 0, scale based on max absolute value
            let abs_max = min_val.abs().max(max_val.abs());
            let scale = if abs_max > 0.0 {
                abs_max / (qmax as f32)
            } else {
                1.0
            };
            (scale, 0)
        } else {
            // Asymmetric: use full range
            let scale = if (max_val - min_val).abs() > 0.0 {
                (max_val - min_val) / ((qmax - qmin) as f32)
            } else {
                1.0
            };
            let zero_point = qmin - (min_val / scale).round() as i32;
            let zero_point = zero_point.clamp(qmin, qmax);
            (scale, zero_point)
        };

        Self {
            scale,
            zero_point,
            qmin,
            qmax,
        }
    }

    /// Quantize a floating-point value
    #[inline]
    pub fn quantize(&self, value: f32) -> i32 {
        let quantized = (value / self.scale).round() as i32 + self.zero_point;
        quantized.clamp(self.qmin, self.qmax)
    }

    /// Dequantize a quantized value
    #[inline]
    pub fn dequantize(&self, quantized: i32) -> f32 {
        (quantized - self.zero_point) as f32 * self.scale
    }
}

/// Quantized tensor representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizedTensor {
    /// Quantized data (stored as i32 for all schemes)
    pub data: Vec<i32>,
    /// Tensor shape
    pub shape: Vec<usize>,
    /// Quantization parameters (one per channel for per-channel quantization)
    pub params: Vec<QuantizationParams>,
    /// Quantization configuration
    pub config: QuantizationConfig,
}

impl QuantizedTensor {
    /// Quantize a tensor with per-tensor quantization
    pub fn quantize_per_tensor(
        data: &[f32],
        shape: Vec<usize>,
        config: QuantizationConfig,
    ) -> Result<Self, QuantizationError> {
        if data.is_empty() {
            return Err(QuantizationError::EmptyTensor);
        }

        // Ensure config is per-tensor
        if !matches!(config.granularity, QuantizationGranularity::PerTensor) {
            return Err(QuantizationError::UnsupportedScheme(
                "Expected per-tensor granularity".to_string(),
            ));
        }

        // Calculate min/max
        let (min_val, max_val) = Self::calculate_min_max(data, &config.calibration)?;

        // Create quantization parameters
        let params =
            QuantizationParams::from_min_max(min_val, max_val, config.scheme, config.symmetric);

        // Quantize data
        let quantized_data: Vec<i32> = data.iter().map(|&v| params.quantize(v)).collect();

        Ok(Self {
            data: quantized_data,
            shape,
            params: vec![params],
            config,
        })
    }

    /// Quantize a tensor with per-channel quantization
    pub fn quantize_per_channel(
        data: &[f32],
        shape: Vec<usize>,
        config: QuantizationConfig,
    ) -> Result<Self, QuantizationError> {
        if data.is_empty() {
            return Err(QuantizationError::EmptyTensor);
        }

        let num_channels = match config.granularity {
            QuantizationGranularity::PerChannel { num_channels } => num_channels,
            _ => {
                return Err(QuantizationError::UnsupportedScheme(
                    "Expected per-channel granularity".to_string(),
                ))
            }
        };

        if shape.is_empty() {
            return Err(QuantizationError::InvalidShape("Empty shape".to_string()));
        }

        // First dimension is typically the output channel dimension
        if shape[0] != num_channels {
            return Err(QuantizationError::InvalidChannelCount {
                expected: num_channels,
                got: shape[0],
            });
        }

        let channel_size = data.len() / num_channels;

        // Calculate parameters for each channel
        let mut params = Vec::with_capacity(num_channels);
        for i in 0..num_channels {
            let start = i * channel_size;
            let end = start + channel_size;
            let channel_data = &data[start..end];

            let (min_val, max_val) = Self::calculate_min_max(channel_data, &config.calibration)?;
            let channel_params =
                QuantizationParams::from_min_max(min_val, max_val, config.scheme, config.symmetric);
            params.push(channel_params);
        }

        // Quantize each channel
        let mut quantized_data = Vec::with_capacity(data.len());
        for (i, chunk) in data.chunks(channel_size).enumerate() {
            for &value in chunk {
                quantized_data.push(params[i].quantize(value));
            }
        }

        Ok(Self {
            data: quantized_data,
            shape,
            params,
            config,
        })
    }

    /// Calculate min/max values based on calibration method
    fn calculate_min_max(
        data: &[f32],
        calibration: &CalibrationMethod,
    ) -> Result<(f32, f32), QuantizationError> {
        match calibration {
            CalibrationMethod::MinMax => {
                let min_val = data.iter().copied().fold(f32::INFINITY, f32::min);
                let max_val = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                Ok((min_val, max_val))
            }
            CalibrationMethod::Percentile { lower, upper } => {
                let mut sorted = data.to_vec();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

                let lower_idx = (sorted.len() as f32 * (*lower as f32 / 100.0)) as usize;
                let upper_idx = (sorted.len() as f32 * (*upper as f32 / 100.0)) as usize;

                let min_val = sorted[lower_idx.min(sorted.len() - 1)];
                let max_val = sorted[upper_idx.min(sorted.len() - 1)];
                Ok((min_val, max_val))
            }
            _ => {
                // Entropy and MSE not yet implemented, fall back to MinMax
                let min_val = data.iter().copied().fold(f32::INFINITY, f32::min);
                let max_val = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                Ok((min_val, max_val))
            }
        }
    }

    /// Dequantize the tensor back to f32
    pub fn dequantize(&self) -> Vec<f32> {
        match self.config.granularity {
            QuantizationGranularity::PerTensor => {
                let params = &self.params[0];
                self.data.iter().map(|&q| params.dequantize(q)).collect()
            }
            QuantizationGranularity::PerChannel { num_channels } => {
                let channel_size = self.data.len() / num_channels;
                let mut result = Vec::with_capacity(self.data.len());

                for (i, chunk) in self.data.chunks(channel_size).enumerate() {
                    for &q in chunk {
                        result.push(self.params[i].dequantize(q));
                    }
                }
                result
            }
            QuantizationGranularity::PerGroup { .. } => {
                // Not yet implemented, fall back to per-tensor
                let params = &self.params[0];
                self.data.iter().map(|&q| params.dequantize(q)).collect()
            }
        }
    }

    /// Get the compression ratio compared to f32 storage
    pub fn compression_ratio(&self) -> f32 {
        let original_bytes = self.data.len() * 4; // f32 = 4 bytes
        let quantized_bytes = self.size_bytes();
        original_bytes as f32 / quantized_bytes as f32
    }

    /// Calculate size in bytes for quantized representation
    pub fn size_bytes(&self) -> usize {
        match self.config.scheme {
            QuantizationScheme::Int4 => {
                // INT4 packs 2 values per byte
                self.data.len().div_ceil(2)
                    + self.params.len() * std::mem::size_of::<QuantizationParams>()
            }
            QuantizationScheme::Int8 => {
                self.data.len() + self.params.len() * std::mem::size_of::<QuantizationParams>()
            }
            QuantizationScheme::Int16 => {
                self.data.len() * 2 + self.params.len() * std::mem::size_of::<QuantizationParams>()
            }
        }
    }

    /// Pack INT4 data into bytes (2 values per byte)
    pub fn pack_int4(&self) -> Result<Vec<u8>, QuantizationError> {
        if self.config.scheme != QuantizationScheme::Int4 {
            return Err(QuantizationError::InvalidBitWidth(
                self.config.scheme.bit_width(),
            ));
        }

        let mut packed = Vec::with_capacity(self.data.len().div_ceil(2));
        for chunk in self.data.chunks(2) {
            let high = (chunk[0] & 0xF) as u8;
            let low = if chunk.len() > 1 {
                (chunk[1] & 0xF) as u8
            } else {
                0
            };
            packed.push((high << 4) | low);
        }

        Ok(packed)
    }

    /// Unpack INT4 data from bytes
    pub fn unpack_int4(packed: &[u8], length: usize) -> Vec<i32> {
        let mut unpacked = Vec::with_capacity(length);
        for &byte in packed {
            let high = ((byte >> 4) & 0xF) as i32;
            let low = (byte & 0xF) as i32;
            unpacked.push(high);
            if unpacked.len() < length {
                unpacked.push(low);
            }
        }
        unpacked.truncate(length);
        unpacked
    }

    /// Calculate quantization error (MSE)
    pub fn quantization_error(&self, original: &[f32]) -> f32 {
        let dequantized = self.dequantize();
        let mse: f32 = original
            .iter()
            .zip(dequantized.iter())
            .map(|(o, d)| {
                let diff = o - d;
                diff * diff
            })
            .sum::<f32>()
            / original.len() as f32;
        mse
    }
}

impl fmt::Display for QuantizedTensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "QuantizedTensor({:?}, shape={:?}, scheme={:?}, params={})",
            self.config.granularity,
            self.shape,
            self.config.scheme,
            self.params.len()
        )
    }
}

/// Dynamic quantization configuration for runtime quantization
#[derive(Debug, Clone)]
pub struct DynamicQuantizer {
    /// Target quantization scheme
    scheme: QuantizationScheme,
    /// Use symmetric quantization
    symmetric: bool,
    /// Calibration method
    calibration: CalibrationMethod,
}

impl DynamicQuantizer {
    /// Create a new dynamic quantizer
    pub fn new(scheme: QuantizationScheme, symmetric: bool) -> Self {
        Self {
            scheme,
            symmetric,
            calibration: CalibrationMethod::MinMax,
        }
    }

    /// Quantize activation tensor at runtime
    pub fn quantize_activation(
        &self,
        data: &[f32],
        shape: Vec<usize>,
    ) -> Result<QuantizedTensor, QuantizationError> {
        let config = QuantizationConfig {
            scheme: self.scheme,
            granularity: QuantizationGranularity::PerTensor,
            symmetric: self.symmetric,
            calibration: self.calibration,
        };

        QuantizedTensor::quantize_per_tensor(data, shape, config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantization_scheme_ranges() {
        assert_eq!(QuantizationScheme::Int8.range(true), (-128, 127));
        assert_eq!(QuantizationScheme::Int8.range(false), (0, 255));
        assert_eq!(QuantizationScheme::Int4.range(true), (-8, 7));
        assert_eq!(QuantizationScheme::Int4.range(false), (0, 15));
    }

    #[test]
    fn test_quantization_params_symmetric() {
        let params = QuantizationParams::from_min_max(-1.0, 1.0, QuantizationScheme::Int8, true);
        assert_eq!(params.zero_point, 0);
        assert!(params.scale > 0.0);

        // Test quantization
        assert_eq!(params.quantize(0.0), 0);
        assert!(params.quantize(1.0) > 0);
        assert!(params.quantize(-1.0) < 0);
    }

    #[test]
    fn test_quantization_params_asymmetric() {
        // Use a range that doesn't start at zero to ensure non-zero zero_point
        let params = QuantizationParams::from_min_max(0.5, 1.5, QuantizationScheme::Int8, false);
        // For asymmetric quantization, zero_point should be calculated
        assert!(params.scale > 0.0);

        // Test another case with negative values
        let params2 = QuantizationParams::from_min_max(-1.0, 0.5, QuantizationScheme::Int8, false);
        assert!(params2.scale > 0.0);
        assert!(params2.zero_point >= params2.qmin && params2.zero_point <= params2.qmax);
    }

    #[test]
    fn test_per_tensor_quantization() {
        let data = vec![0.5, -0.3, 0.8, -0.1];
        let config = QuantizationConfig::int8_symmetric();
        let quantized = QuantizedTensor::quantize_per_tensor(&data, vec![4], config)
            .expect("test: should succeed");

        assert_eq!(quantized.data.len(), 4);
        assert_eq!(quantized.params.len(), 1);

        // Dequantize and check
        let dequantized = quantized.dequantize();
        assert_eq!(dequantized.len(), 4);

        // Should be close to original
        for (orig, deq) in data.iter().zip(dequantized.iter()) {
            assert!((orig - deq).abs() < 0.01);
        }
    }

    #[test]
    fn test_per_channel_quantization() {
        // 2 channels, 4 elements each
        let data = vec![0.5, 0.3, -0.2, -0.4, 0.1, 0.6, -0.5, 0.2];
        let config = QuantizationConfig::int8_per_channel(2);
        let quantized = QuantizedTensor::quantize_per_channel(&data, vec![2, 4], config)
            .expect("test: should succeed");

        assert_eq!(quantized.data.len(), 8);
        assert_eq!(quantized.params.len(), 2);

        // Each channel should have its own parameters
        assert_ne!(quantized.params[0].scale, quantized.params[1].scale);
    }

    #[test]
    fn test_int4_quantization() {
        let data = vec![0.1, 0.2, 0.3, 0.4];
        let config = QuantizationConfig::int4_symmetric();
        let quantized = QuantizedTensor::quantize_per_tensor(&data, vec![4], config)
            .expect("test: should succeed");

        // INT4 range is -8 to 7
        for &q in &quantized.data {
            assert!((-8..=7).contains(&q));
        }

        // Test packing
        let packed = quantized.pack_int4().expect("test: should succeed");
        assert_eq!(packed.len(), 2); // 4 values packed into 2 bytes

        // Test unpacking
        let unpacked = QuantizedTensor::unpack_int4(&packed, 4);
        assert_eq!(unpacked, quantized.data);
    }

    #[test]
    fn test_compression_ratio() {
        let data = vec![1.0; 100];
        let config = QuantizationConfig::int8_symmetric();
        let quantized = QuantizedTensor::quantize_per_tensor(&data, vec![100], config)
            .expect("test: should succeed");

        let ratio = quantized.compression_ratio();
        assert!(ratio > 1.0); // Should be compressed
    }

    #[test]
    fn test_quantization_error() {
        let data = vec![0.1, 0.5, 0.9, 0.3];
        let config = QuantizationConfig::int8_symmetric();
        let quantized = QuantizedTensor::quantize_per_tensor(&data, vec![4], config)
            .expect("test: should succeed");

        let error = quantized.quantization_error(&data);
        assert!(error < 0.001); // Error should be small for INT8
    }

    #[test]
    fn test_dynamic_quantizer() {
        let quantizer = DynamicQuantizer::new(QuantizationScheme::Int8, true);
        let data = vec![1.0, 2.0, 3.0, 4.0];

        let quantized = quantizer
            .quantize_activation(&data, vec![4])
            .expect("test: should succeed");
        assert_eq!(quantized.data.len(), 4);

        let dequantized = quantized.dequantize();
        for (orig, deq) in data.iter().zip(dequantized.iter()) {
            assert!((orig - deq).abs() < 0.1);
        }
    }

    #[test]
    fn test_percentile_calibration() {
        let mut data = vec![0.0; 100];
        // Add outliers
        data[0] = -100.0;
        data[99] = 100.0;
        // Normal data
        for (i, val) in data.iter_mut().enumerate().take(99).skip(1) {
            *val = (i as f32 - 50.0) / 50.0; // Range roughly -1 to 1
        }

        let config = QuantizationConfig {
            scheme: QuantizationScheme::Int8,
            granularity: QuantizationGranularity::PerTensor,
            symmetric: true,
            calibration: CalibrationMethod::Percentile {
                lower: 1,
                upper: 99,
            },
        };

        let quantized = QuantizedTensor::quantize_per_tensor(&data, vec![100], config)
            .expect("test: should succeed");

        // The outliers should be clipped in the calibration
        let params = &quantized.params[0];
        assert!(params.scale < 1.0); // Scale should be much less than if we used min/max with outliers
    }

    #[test]
    fn test_empty_tensor_error() {
        let data: Vec<f32> = vec![];
        let config = QuantizationConfig::int8_symmetric();
        let result = QuantizedTensor::quantize_per_tensor(&data, vec![0], config);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_channel_count() {
        let data = vec![1.0; 8];
        let config = QuantizationConfig::int8_per_channel(3); // Wrong number of channels
        let result = QuantizedTensor::quantize_per_channel(&data, vec![2, 4], config);
        assert!(result.is_err());
    }
}
