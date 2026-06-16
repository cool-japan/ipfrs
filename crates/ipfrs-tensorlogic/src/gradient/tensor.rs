//! Core gradient tensor types and operations.
//!
//! This module provides the fundamental gradient data structures:
//! - [`SparseGradient`] — sparse representation with index/value pairs
//! - [`QuantizedGradient`] — int8-quantized representation for compression
//! - [`LayerGradient`] — enum over dense / sparse / quantized
//! - [`GradientDelta`] — per-layer gradient bundle linked to a base model CID
//! - [`GradientCompressor`] — top-k, threshold, random, and quantization compression
//! - [`GradientAggregator`] — average, weighted-average, and momentum helpers
//! - [`GradientVerifier`] — shape, finiteness, outlier, clipping utilities

use crate::arrow::{TensorDtype, TensorMetadata};
use ipfrs_core::Cid;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::GradientError;

// ── SparseGradient ─────────────────────────────────────────────────────────

/// Sparse gradient representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseGradient {
    /// Indices of non-zero elements (flattened)
    pub indices: Vec<usize>,
    /// Non-zero gradient values
    pub values: Vec<f32>,
    /// Original tensor shape
    pub shape: Vec<usize>,
    /// Metadata
    pub metadata: TensorMetadata,
}

impl SparseGradient {
    /// Create a new sparse gradient
    pub fn new(indices: Vec<usize>, values: Vec<f32>, shape: Vec<usize>) -> Self {
        let metadata = TensorMetadata {
            name: "sparse_gradient".to_string(),
            shape: shape.clone(),
            dtype: TensorDtype::Float32,
            strides: None,
            custom: HashMap::new(),
        };

        Self {
            indices,
            values,
            shape,
            metadata,
        }
    }

    /// Get the number of non-zero elements
    pub fn nnz(&self) -> usize {
        self.indices.len()
    }

    /// Get the total number of elements
    pub fn total_elements(&self) -> usize {
        self.shape.iter().product()
    }

    /// Get the sparsity ratio (0.0 = dense, 1.0 = all zeros)
    pub fn sparsity_ratio(&self) -> f32 {
        1.0 - (self.nnz() as f32 / self.total_elements() as f32)
    }

    /// Convert to dense representation
    pub fn to_dense(&self) -> Vec<f32> {
        let total = self.total_elements();
        let mut dense = vec![0.0; total];

        for (&idx, &val) in self.indices.iter().zip(&self.values) {
            if idx < total {
                dense[idx] = val;
            }
        }

        dense
    }

    /// Verify shape consistency
    pub fn verify_shape(&self) -> Result<(), GradientError> {
        let total = self.total_elements();

        for &idx in &self.indices {
            if idx >= total {
                return Err(GradientError::InvalidGradient(format!(
                    "Index {} out of bounds for shape {:?}",
                    idx, self.shape
                )));
            }
        }

        if self.indices.len() != self.values.len() {
            return Err(GradientError::InvalidGradient(format!(
                "Indices length {} != values length {}",
                self.indices.len(),
                self.values.len()
            )));
        }

        Ok(())
    }
}

// ── QuantizedGradient ──────────────────────────────────────────────────────

/// Quantized gradient (reduced precision)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizedGradient {
    /// Quantized values (e.g., int8)
    pub quantized_values: Vec<i8>,
    /// Scale factor for dequantization
    pub scale: f32,
    /// Minimum value for dequantization
    pub min_val: f32,
    /// Original tensor shape
    pub shape: Vec<usize>,
    /// Metadata
    pub metadata: TensorMetadata,
}

impl QuantizedGradient {
    /// Quantize a dense gradient to int8
    pub fn from_dense(values: &[f32], shape: Vec<usize>) -> Self {
        let (quantized_values, scale, min_val) = Self::quantize_i8(values);

        let metadata = TensorMetadata {
            name: "quantized_gradient".to_string(),
            shape: shape.clone(),
            dtype: TensorDtype::Int8,
            strides: None,
            custom: HashMap::new(),
        };

        Self {
            quantized_values,
            scale,
            min_val,
            shape,
            metadata,
        }
    }

    /// Quantize f32 values to i8
    fn quantize_i8(values: &[f32]) -> (Vec<i8>, f32, f32) {
        if values.is_empty() {
            return (Vec::new(), 1.0, 0.0);
        }

        let min_val = values.iter().copied().fold(f32::INFINITY, f32::min);
        let max_val = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        // Avoid division by zero
        let scale = if (max_val - min_val).abs() < 1e-8 {
            1.0
        } else {
            (max_val - min_val) / 255.0
        };

        let quantized = values
            .iter()
            .map(|&v| {
                // Map [min_val, max_val] to [0, 255], then shift to [-128, 127]
                let normalized = (v - min_val) / scale;
                (normalized - 128.0).round().clamp(-128.0, 127.0) as i8
            })
            .collect();

        (quantized, scale, min_val)
    }

    /// Dequantize to f32 values
    pub fn to_dense(&self) -> Vec<f32> {
        self.quantized_values
            .iter()
            .map(|&q| {
                // Shift from [-128, 127] to [0, 255], then scale back
                let normalized = (q as f32) + 128.0;
                normalized * self.scale + self.min_val
            })
            .collect()
    }

    /// Get compression ratio
    pub fn compression_ratio(&self) -> f32 {
        // f32 = 4 bytes, i8 = 1 byte, plus scale and min_val
        let original_size = self.quantized_values.len() * 4;
        let compressed_size = self.quantized_values.len() + 8; // 4 bytes scale + 4 bytes min_val
        original_size as f32 / compressed_size as f32
    }
}

// ── LayerGradient ──────────────────────────────────────────────────────────

/// Gradient for a single layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LayerGradient {
    /// Dense gradient
    Dense { values: Vec<f32>, shape: Vec<usize> },
    /// Sparse gradient
    Sparse(SparseGradient),
    /// Quantized gradient
    Quantized(QuantizedGradient),
}

impl LayerGradient {
    /// Get the shape of the gradient
    pub fn shape(&self) -> &[usize] {
        match self {
            LayerGradient::Dense { shape, .. } => shape,
            LayerGradient::Sparse(sg) => &sg.shape,
            LayerGradient::Quantized(qg) => &qg.shape,
        }
    }

    /// Convert to dense representation
    pub fn to_dense(&self) -> Vec<f32> {
        match self {
            LayerGradient::Dense { values, .. } => values.clone(),
            LayerGradient::Sparse(sg) => sg.to_dense(),
            LayerGradient::Quantized(qg) => qg.to_dense(),
        }
    }

    /// Get memory size in bytes
    pub fn memory_size(&self) -> usize {
        match self {
            LayerGradient::Dense { values, .. } => values.len() * 4,
            LayerGradient::Sparse(sg) => sg.indices.len() * 4 + sg.values.len() * 4,
            LayerGradient::Quantized(qg) => qg.quantized_values.len() + 8,
        }
    }
}

// ── GradientDelta ──────────────────────────────────────────────────────────

/// Gradient delta (difference from base model)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientDelta {
    /// Base model CID
    #[serde(serialize_with = "crate::serialize_cid")]
    #[serde(deserialize_with = "crate::deserialize_cid")]
    pub base_model: Cid,
    /// Layer name to gradient mapping
    pub layer_gradients: HashMap<String, LayerGradient>,
    /// Checksum for verification
    pub checksum: u64,
    /// Timestamp
    pub timestamp: i64,
}

impl GradientDelta {
    /// Create a new gradient delta
    pub fn new(base_model: Cid) -> Self {
        Self {
            base_model,
            layer_gradients: HashMap::new(),
            checksum: 0,
            timestamp: chrono::Utc::now().timestamp(),
        }
    }

    /// Add a dense gradient for a layer
    pub fn add_dense_gradient(&mut self, layer_name: String, values: Vec<f32>, shape: Vec<usize>) {
        self.layer_gradients
            .insert(layer_name, LayerGradient::Dense { values, shape });
        self.update_checksum();
    }

    /// Add a sparse gradient for a layer
    pub fn add_sparse_gradient(&mut self, layer_name: String, gradient: SparseGradient) {
        self.layer_gradients
            .insert(layer_name, LayerGradient::Sparse(gradient));
        self.update_checksum();
    }

    /// Add a quantized gradient for a layer
    pub fn add_quantized_gradient(&mut self, layer_name: String, gradient: QuantizedGradient) {
        self.layer_gradients
            .insert(layer_name, LayerGradient::Quantized(gradient));
        self.update_checksum();
    }

    /// Compute checksum for verification
    fn update_checksum(&mut self) {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();

        // Hash layer count
        self.layer_gradients.len().hash(&mut hasher);

        // Hash each layer's data
        let mut sorted_layers: Vec<_> = self.layer_gradients.iter().collect();
        sorted_layers.sort_by_key(|(name, _)| *name);

        for (name, gradient) in sorted_layers {
            name.hash(&mut hasher);
            gradient.shape().hash(&mut hasher);

            // Hash a sample of values for efficiency
            let dense = gradient.to_dense();
            let sample_size = dense.len().min(100);
            for &v in dense.iter().take(sample_size) {
                v.to_bits().hash(&mut hasher);
            }
        }

        self.checksum = hasher.finish();
    }

    /// Verify checksum
    pub fn verify_checksum(&self) -> Result<(), GradientError> {
        let mut temp = self.clone();
        temp.update_checksum();

        if temp.checksum == self.checksum {
            Ok(())
        } else {
            Err(GradientError::ChecksumFailed)
        }
    }

    /// Get total memory size in bytes
    pub fn total_memory_size(&self) -> usize {
        self.layer_gradients.values().map(|g| g.memory_size()).sum()
    }
}

// ── GradientCompressor ─────────────────────────────────────────────────────

/// Gradient compression utilities
pub struct GradientCompressor;

impl GradientCompressor {
    /// Compress gradient using top-k sparsification
    pub fn top_k(
        values: &[f32],
        shape: Vec<usize>,
        k: usize,
    ) -> Result<SparseGradient, GradientError> {
        if k == 0 || k > values.len() {
            return Err(GradientError::InvalidCompressionRatio(
                k as f32 / values.len() as f32,
            ));
        }

        // Get indices of top-k absolute values
        let mut indexed_values: Vec<(usize, f32)> = values
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v.abs()))
            .collect();

        indexed_values.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed_values.truncate(k);

        let mut indices = Vec::with_capacity(k);
        let mut sparse_values = Vec::with_capacity(k);

        for (idx, _) in indexed_values {
            indices.push(idx);
            sparse_values.push(values[idx]);
        }

        Ok(SparseGradient::new(indices, sparse_values, shape))
    }

    /// Compress gradient using threshold-based sparsification
    pub fn threshold(values: &[f32], shape: Vec<usize>, threshold: f32) -> SparseGradient {
        let mut indices = Vec::new();
        let mut sparse_values = Vec::new();

        for (i, &v) in values.iter().enumerate() {
            if v.abs() >= threshold {
                indices.push(i);
                sparse_values.push(v);
            }
        }

        SparseGradient::new(indices, sparse_values, shape)
    }

    /// Compress gradient using quantization
    pub fn quantize(values: &[f32], shape: Vec<usize>) -> QuantizedGradient {
        QuantizedGradient::from_dense(values, shape)
    }

    /// Compress gradient using random sparsification
    pub fn random_sparsification(
        values: &[f32],
        shape: Vec<usize>,
        keep_ratio: f32,
    ) -> Result<SparseGradient, GradientError> {
        use rand::RngExt;

        if keep_ratio <= 0.0 || keep_ratio > 1.0 {
            return Err(GradientError::InvalidCompressionRatio(keep_ratio));
        }

        let mut rng = rand::rng();
        let mut indices = Vec::new();
        let mut sparse_values = Vec::new();

        for (i, &v) in values.iter().enumerate() {
            if rng.random::<f32>() < keep_ratio {
                indices.push(i);
                sparse_values.push(v / keep_ratio); // Compensate for dropout
            }
        }

        Ok(SparseGradient::new(indices, sparse_values, shape))
    }
}

// ── GradientAggregator ─────────────────────────────────────────────────────

/// Gradient aggregation for federated learning
pub struct GradientAggregator;

impl GradientAggregator {
    /// Average multiple gradients (unweighted)
    pub fn average(gradients: &[Vec<f32>]) -> Result<Vec<f32>, GradientError> {
        if gradients.is_empty() {
            return Err(GradientError::EmptyGradientSet);
        }

        let len = gradients[0].len();

        // Verify all gradients have the same length
        for g in gradients.iter() {
            if g.len() != len {
                return Err(GradientError::ShapeMismatch {
                    expected: vec![len],
                    actual: vec![g.len()],
                });
            }
        }

        let mut result = vec![0.0; len];
        let count = gradients.len() as f32;

        for gradient in gradients {
            for (i, &v) in gradient.iter().enumerate() {
                result[i] += v / count;
            }
        }

        Ok(result)
    }

    /// Weighted average of gradients
    pub fn weighted_average(
        gradients: &[Vec<f32>],
        weights: &[f32],
    ) -> Result<Vec<f32>, GradientError> {
        if gradients.is_empty() {
            return Err(GradientError::EmptyGradientSet);
        }

        if gradients.len() != weights.len() {
            return Err(GradientError::InvalidGradient(format!(
                "Gradient count {} != weight count {}",
                gradients.len(),
                weights.len()
            )));
        }

        let len = gradients[0].len();

        // Verify all gradients have the same length
        for g in gradients.iter() {
            if g.len() != len {
                return Err(GradientError::ShapeMismatch {
                    expected: vec![len],
                    actual: vec![g.len()],
                });
            }
        }

        let weight_sum: f32 = weights.iter().sum();
        if weight_sum == 0.0 {
            return Err(GradientError::InvalidGradient(
                "Sum of weights is zero".to_string(),
            ));
        }

        let mut result = vec![0.0; len];

        for (gradient, &weight) in gradients.iter().zip(weights) {
            let normalized_weight = weight / weight_sum;
            for (i, &v) in gradient.iter().enumerate() {
                result[i] += v * normalized_weight;
            }
        }

        Ok(result)
    }

    /// Apply momentum to gradient
    pub fn apply_momentum(
        current_gradient: &[f32],
        previous_momentum: &[f32],
        momentum_factor: f32,
    ) -> Result<Vec<f32>, GradientError> {
        if current_gradient.len() != previous_momentum.len() {
            return Err(GradientError::ShapeMismatch {
                expected: vec![previous_momentum.len()],
                actual: vec![current_gradient.len()],
            });
        }

        let result = current_gradient
            .iter()
            .zip(previous_momentum)
            .map(|(&g, &m)| momentum_factor * m + g)
            .collect();

        Ok(result)
    }
}

// ── GradientVerifier ───────────────────────────────────────────────────────

/// Gradient verification utilities
pub struct GradientVerifier;

impl GradientVerifier {
    /// Verify gradient shape matches expected shape
    pub fn verify_shape(gradient: &[f32], expected_shape: &[usize]) -> Result<(), GradientError> {
        let expected_size: usize = expected_shape.iter().product();

        if gradient.len() != expected_size {
            return Err(GradientError::ShapeMismatch {
                expected: expected_shape.to_vec(),
                actual: vec![gradient.len()],
            });
        }

        Ok(())
    }

    /// Detect outliers in gradient (values beyond threshold standard deviations)
    pub fn detect_outliers(gradient: &[f32], std_threshold: f32) -> Result<(), GradientError> {
        if gradient.is_empty() {
            return Ok(());
        }

        // Calculate mean
        let mean = gradient.iter().sum::<f32>() / gradient.len() as f32;

        // Calculate standard deviation
        let variance =
            gradient.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / gradient.len() as f32;
        let std_dev = variance.sqrt();

        // Check for outliers
        for (i, &v) in gradient.iter().enumerate() {
            let z_score = (v - mean).abs() / std_dev;
            if z_score > std_threshold {
                return Err(GradientError::OutlierDetected { index: i, value: v });
            }
        }

        Ok(())
    }

    /// Verify gradient is not NaN or Inf
    pub fn verify_finite(gradient: &[f32]) -> Result<(), GradientError> {
        for (i, &v) in gradient.iter().enumerate() {
            if !v.is_finite() {
                return Err(GradientError::InvalidGradient(format!(
                    "Non-finite value at index {}: {}",
                    i, v
                )));
            }
        }

        Ok(())
    }

    /// Compute L2 norm of gradient
    pub fn l2_norm(gradient: &[f32]) -> f32 {
        gradient.iter().map(|&v| v * v).sum::<f32>().sqrt()
    }

    /// Clip gradient by norm
    pub fn clip_by_norm(gradient: &mut [f32], max_norm: f32) {
        let norm = Self::l2_norm(gradient);

        if norm > max_norm {
            let scale = max_norm / norm;
            for v in gradient.iter_mut() {
                *v *= scale;
            }
        }
    }
}
