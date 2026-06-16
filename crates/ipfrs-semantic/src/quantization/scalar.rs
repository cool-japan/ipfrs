//! Scalar (INT8/UINT8) vector quantization.

use ipfrs_core::{Error, Result};
use serde::{Deserialize, Serialize};

/// Scalar quantization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalarQuantizerConfig {
    /// Number of bits for quantization (8 for int8/uint8)
    pub bits: u8,
    /// Whether to use signed quantization (int8 vs uint8)
    pub signed: bool,
    /// Per-dimension min values
    pub min_values: Vec<f32>,
    /// Per-dimension max values
    pub max_values: Vec<f32>,
}

/// Scalar quantizer for vector compression
///
/// Quantizes floating-point vectors to int8 or uint8, achieving 4x compression
/// with typically < 5% accuracy loss.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalarQuantizer {
    /// Quantization configuration
    config: ScalarQuantizerConfig,
    /// Vector dimension
    dimension: usize,
    /// Whether the quantizer has been trained
    trained: bool,
}

impl ScalarQuantizer {
    /// Create a new scalar quantizer for the given dimension
    ///
    /// # Arguments
    /// * `dimension` - Vector dimension
    /// * `signed` - Use int8 (true) or uint8 (false)
    pub fn new(dimension: usize, signed: bool) -> Self {
        Self {
            config: ScalarQuantizerConfig {
                bits: 8,
                signed,
                min_values: vec![f32::MAX; dimension],
                max_values: vec![f32::MIN; dimension],
            },
            dimension,
            trained: false,
        }
    }

    /// Create a quantizer with uint8 (unsigned) quantization
    pub fn uint8(dimension: usize) -> Self {
        Self::new(dimension, false)
    }

    /// Create a quantizer with int8 (signed) quantization
    pub fn int8(dimension: usize) -> Self {
        Self::new(dimension, true)
    }

    /// Train the quantizer on a set of vectors
    ///
    /// Learns min/max values for each dimension to establish
    /// the quantization range.
    ///
    /// # Arguments
    /// * `vectors` - Training vectors
    pub fn train(&mut self, vectors: &[Vec<f32>]) -> Result<()> {
        if vectors.is_empty() {
            return Err(Error::InvalidInput(
                "Cannot train on empty vector set".to_string(),
            ));
        }

        // Validate dimensions
        for (i, vec) in vectors.iter().enumerate() {
            if vec.len() != self.dimension {
                return Err(Error::InvalidInput(format!(
                    "Vector {} has dimension {}, expected {}",
                    i,
                    vec.len(),
                    self.dimension
                )));
            }
        }

        // Reset min/max values
        self.config.min_values = vec![f32::MAX; self.dimension];
        self.config.max_values = vec![f32::MIN; self.dimension];

        // Compute per-dimension min/max
        for vec in vectors {
            for (i, &val) in vec.iter().enumerate() {
                if val < self.config.min_values[i] {
                    self.config.min_values[i] = val;
                }
                if val > self.config.max_values[i] {
                    self.config.max_values[i] = val;
                }
            }
        }

        // Add small margin to avoid edge cases
        for i in 0..self.dimension {
            let range = self.config.max_values[i] - self.config.min_values[i];
            if range < 1e-6 {
                // Constant dimension, set arbitrary range
                self.config.min_values[i] -= 0.5;
                self.config.max_values[i] += 0.5;
            } else {
                let margin = range * 0.01;
                self.config.min_values[i] -= margin;
                self.config.max_values[i] += margin;
            }
        }

        self.trained = true;
        Ok(())
    }

    /// Train on a single vector (incremental training)
    pub fn train_incremental(&mut self, vector: &[f32]) -> Result<()> {
        if vector.len() != self.dimension {
            return Err(Error::InvalidInput(format!(
                "Vector has dimension {}, expected {}",
                vector.len(),
                self.dimension
            )));
        }

        for (i, &val) in vector.iter().enumerate() {
            if val < self.config.min_values[i] {
                self.config.min_values[i] = val;
            }
            if val > self.config.max_values[i] {
                self.config.max_values[i] = val;
            }
        }

        self.trained = true;
        Ok(())
    }

    /// Check if the quantizer has been trained
    pub fn is_trained(&self) -> bool {
        self.trained
    }

    /// Quantize a vector to uint8
    pub fn quantize(&self, vector: &[f32]) -> Result<QuantizedVector> {
        if !self.trained {
            return Err(Error::InvalidInput(
                "Quantizer must be trained before use".to_string(),
            ));
        }

        if vector.len() != self.dimension {
            return Err(Error::InvalidInput(format!(
                "Vector has dimension {}, expected {}",
                vector.len(),
                self.dimension
            )));
        }

        let mut quantized = Vec::with_capacity(self.dimension);

        for (i, &val) in vector.iter().enumerate() {
            let min = self.config.min_values[i];
            let max = self.config.max_values[i];
            let range = max - min;

            // Normalize to [0, 1]
            let normalized = if range > 1e-6 {
                ((val - min) / range).clamp(0.0, 1.0)
            } else {
                0.5
            };

            // Quantize
            let q = if self.config.signed {
                // int8: [-128, 127]
                ((normalized * 255.0 - 128.0).round() as i8) as u8
            } else {
                // uint8: [0, 255]
                (normalized * 255.0).round() as u8
            };

            quantized.push(q);
        }

        Ok(QuantizedVector {
            data: quantized,
            signed: self.config.signed,
        })
    }

    /// Dequantize a vector back to f32
    pub fn dequantize(&self, quantized: &QuantizedVector) -> Result<Vec<f32>> {
        if !self.trained {
            return Err(Error::InvalidInput(
                "Quantizer must be trained before use".to_string(),
            ));
        }

        if quantized.data.len() != self.dimension {
            return Err(Error::InvalidInput(format!(
                "Quantized vector has dimension {}, expected {}",
                quantized.data.len(),
                self.dimension
            )));
        }

        let mut result = Vec::with_capacity(self.dimension);

        for (i, &q) in quantized.data.iter().enumerate() {
            let min = self.config.min_values[i];
            let max = self.config.max_values[i];
            let range = max - min;

            // Dequantize
            let normalized = if self.config.signed {
                // int8: [-128, 127] -> [0, 1]
                ((q as i8) as f32 + 128.0) / 255.0
            } else {
                // uint8: [0, 255] -> [0, 1]
                q as f32 / 255.0
            };

            let val = min + normalized * range;
            result.push(val);
        }

        Ok(result)
    }

    /// Compute L2 distance between two quantized vectors (approximate)
    ///
    /// Uses integer arithmetic for fast computation
    pub fn distance_l2_quantized(&self, a: &QuantizedVector, b: &QuantizedVector) -> Result<f32> {
        if a.data.len() != b.data.len() {
            return Err(Error::InvalidInput(
                "Vectors must have same dimension".to_string(),
            ));
        }

        let mut sum_sq: i64 = 0;

        for (qa, qb) in a.data.iter().zip(b.data.iter()) {
            let diff = if a.signed {
                (*qa as i8 as i64) - (*qb as i8 as i64)
            } else {
                (*qa as i64) - (*qb as i64)
            };
            sum_sq += diff * diff;
        }

        // Scale back to approximate original distance
        // This is an approximation since we lose per-dimension scaling info
        Ok((sum_sq as f32).sqrt() / 255.0)
    }

    /// Compute dot product between two quantized vectors (approximate)
    pub fn dot_product_quantized(&self, a: &QuantizedVector, b: &QuantizedVector) -> Result<f32> {
        if a.data.len() != b.data.len() {
            return Err(Error::InvalidInput(
                "Vectors must have same dimension".to_string(),
            ));
        }

        let mut sum: i64 = 0;

        for (qa, qb) in a.data.iter().zip(b.data.iter()) {
            if a.signed {
                sum += (*qa as i8 as i64) * (*qb as i8 as i64);
            } else {
                sum += (*qa as i64) * (*qb as i64);
            }
        }

        // Normalize
        Ok(sum as f32 / (255.0 * 255.0))
    }

    /// Get the dimension
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Get compression ratio (always 4x for 8-bit quantization)
    pub fn compression_ratio(&self) -> f32 {
        4.0 // f32 (4 bytes) -> u8 (1 byte)
    }

    /// Get memory usage estimate for a given number of vectors
    pub fn memory_estimate(&self, num_vectors: usize) -> usize {
        // Per-vector: dimension bytes
        // Plus overhead: 2 * dimension * 4 bytes for min/max values
        num_vectors * self.dimension + 2 * self.dimension * 4
    }
}

/// Quantized vector storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizedVector {
    /// Quantized data (uint8 storage)
    pub data: Vec<u8>,
    /// Whether values are signed
    pub signed: bool,
}

impl QuantizedVector {
    /// Create a new quantized vector
    pub fn new(data: Vec<u8>, signed: bool) -> Self {
        Self { data, signed }
    }

    /// Get the dimension
    pub fn dimension(&self) -> usize {
        self.data.len()
    }

    /// Get the raw bytes
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    /// Get memory size in bytes
    pub fn size_bytes(&self) -> usize {
        self.data.len()
    }
}
