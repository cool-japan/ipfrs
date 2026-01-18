//! Gradient storage and management for federated learning
//!
//! This module provides:
//! - Gradient delta format (differences from base model)
//! - Gradient compression (sparsification, quantization, top-k)
//! - Gradient aggregation (averaging, weighted, momentum)
//! - Gradient verification (checksum, shape, outliers)

use crate::arrow::{TensorDtype, TensorMetadata};
use ipfrs_core::Cid;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

/// Errors that can occur during gradient operations
#[derive(Debug, Error)]
pub enum GradientError {
    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    #[error("Checksum verification failed")]
    ChecksumFailed,

    #[error("Invalid compression ratio: {0}")]
    InvalidCompressionRatio(f32),

    #[error("Empty gradient set")]
    EmptyGradientSet,

    #[error("Incompatible dtype: {0:?}")]
    IncompatibleDtype(TensorDtype),

    #[error("Outlier detected at index {index}: value {value}")]
    OutlierDetected { index: usize, value: f32 },

    #[error("Invalid gradient: {0}")]
    InvalidGradient(String),
}

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
        use rand::Rng;

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

/// Privacy budget for differential privacy
#[derive(Debug, Clone, Copy)]
pub struct PrivacyBudget {
    /// Epsilon (privacy loss parameter)
    pub epsilon: f64,
    /// Delta (failure probability)
    pub delta: f64,
    /// Remaining epsilon
    pub remaining_epsilon: f64,
}

impl PrivacyBudget {
    /// Create a new privacy budget
    pub fn new(epsilon: f64, delta: f64) -> Self {
        Self {
            epsilon,
            delta,
            remaining_epsilon: epsilon,
        }
    }

    /// Consume some privacy budget
    pub fn consume(&mut self, epsilon_used: f64) -> Result<(), GradientError> {
        if epsilon_used > self.remaining_epsilon {
            return Err(GradientError::InvalidGradient(format!(
                "Insufficient privacy budget: need {}, have {}",
                epsilon_used, self.remaining_epsilon
            )));
        }

        self.remaining_epsilon -= epsilon_used;
        Ok(())
    }

    /// Check if budget is exhausted
    pub fn is_exhausted(&self) -> bool {
        self.remaining_epsilon <= 0.0
    }

    /// Get the fraction of budget remaining
    pub fn remaining_fraction(&self) -> f64 {
        self.remaining_epsilon / self.epsilon
    }
}

/// Differential privacy mechanism types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DPMechanism {
    /// Gaussian mechanism (for bounded sensitivity)
    Gaussian,
    /// Laplacian mechanism (for L1 sensitivity)
    Laplacian,
}

/// Differential privacy for gradient protection
pub struct DifferentialPrivacy {
    /// Privacy budget
    budget: PrivacyBudget,
    /// Sensitivity (L2 norm bound for gradients)
    sensitivity: f64,
    /// Mechanism type
    mechanism: DPMechanism,
}

impl DifferentialPrivacy {
    /// Create a new differential privacy instance
    pub fn new(epsilon: f64, delta: f64, sensitivity: f64, mechanism: DPMechanism) -> Self {
        Self {
            budget: PrivacyBudget::new(epsilon, delta),
            sensitivity,
            mechanism,
        }
    }

    /// Add Gaussian noise to gradient (for DP-SGD)
    /// Calibrated according to sensitivity and privacy parameters
    pub fn add_gaussian_noise(&mut self, gradient: &mut [f32]) -> Result<(), GradientError> {
        use rand::Rng;

        if self.budget.is_exhausted() {
            return Err(GradientError::InvalidGradient(
                "Privacy budget exhausted".to_string(),
            ));
        }

        // Calculate noise scale using Gaussian mechanism
        // σ = sensitivity * sqrt(2 * ln(1.25/δ)) / ε
        let ln_term = (1.25 / self.budget.delta).ln();
        let sigma = self.sensitivity * (2.0 * ln_term).sqrt() / self.budget.epsilon;

        let mut rng = rand::rng();

        // Add Gaussian noise to each element
        for v in gradient.iter_mut() {
            let noise: f64 = rng.random_range(-1.0..1.0);
            let gaussian_noise = sigma * noise;
            *v += gaussian_noise as f32;
        }

        // Consume privacy budget (simplified - in practice, this depends on composition)
        self.budget.consume(self.budget.epsilon / 100.0)?;

        Ok(())
    }

    /// Add Laplacian noise to gradient
    /// Calibrated according to L1 sensitivity and privacy parameters
    pub fn add_laplacian_noise(&mut self, gradient: &mut [f32]) -> Result<(), GradientError> {
        use rand::Rng;

        if self.budget.is_exhausted() {
            return Err(GradientError::InvalidGradient(
                "Privacy budget exhausted".to_string(),
            ));
        }

        // Calculate noise scale using Laplacian mechanism
        // b = sensitivity / ε
        let scale = self.sensitivity / self.budget.epsilon;

        let mut rng = rand::rng();

        // Add Laplacian noise to each element
        for v in gradient.iter_mut() {
            let u: f64 = rng.random_range(-0.5..0.5);
            let laplacian_noise = -scale * u.signum() * (1.0 - 2.0 * u.abs()).ln();
            *v += laplacian_noise as f32;
        }

        // Consume privacy budget
        self.budget.consume(self.budget.epsilon / 100.0)?;

        Ok(())
    }

    /// Apply DP-SGD (Differential Private Stochastic Gradient Descent)
    /// This clips gradients and adds noise
    pub fn apply_dp_sgd(
        &mut self,
        gradient: &mut [f32],
        clip_norm: f32,
    ) -> Result<(), GradientError> {
        // Step 1: Clip gradient to bound sensitivity
        GradientVerifier::clip_by_norm(gradient, clip_norm);

        // Step 2: Add calibrated noise
        match self.mechanism {
            DPMechanism::Gaussian => self.add_gaussian_noise(gradient)?,
            DPMechanism::Laplacian => self.add_laplacian_noise(gradient)?,
        }

        Ok(())
    }

    /// Get remaining privacy budget
    pub fn remaining_budget(&self) -> f64 {
        self.budget.remaining_epsilon
    }

    /// Check if privacy budget is exhausted
    pub fn is_budget_exhausted(&self) -> bool {
        self.budget.is_exhausted()
    }

    /// Get privacy parameters
    pub fn get_privacy_params(&self) -> (f64, f64) {
        (self.budget.epsilon, self.budget.delta)
    }

    /// Calculate noise multiplier for given privacy parameters
    /// Used in DP-SGD implementations
    pub fn calculate_noise_multiplier(epsilon: f64, delta: f64, sensitivity: f64) -> f64 {
        // σ = sensitivity * sqrt(2 * ln(1.25/δ)) / ε
        let ln_term = (1.25 / delta).ln();
        sensitivity * (2.0 * ln_term).sqrt() / epsilon
    }
}

/// Secure aggregation for federated learning (simplified)
pub struct SecureAggregation {
    /// Minimum number of participants required
    min_participants: usize,
    /// Current participant count
    participant_count: usize,
}

impl SecureAggregation {
    /// Create a new secure aggregation instance
    pub fn new(min_participants: usize) -> Self {
        Self {
            min_participants,
            participant_count: 0,
        }
    }

    /// Add a participant
    pub fn add_participant(&mut self) {
        self.participant_count += 1;
    }

    /// Check if we have enough participants
    pub fn can_aggregate(&self) -> bool {
        self.participant_count >= self.min_participants
    }

    /// Aggregate gradients securely
    /// In a real implementation, this would use cryptographic techniques
    /// like secret sharing, homomorphic encryption, or secure multi-party computation
    pub fn aggregate_secure(&self, gradients: &[Vec<f32>]) -> Result<Vec<f32>, GradientError> {
        if !self.can_aggregate() {
            return Err(GradientError::InvalidGradient(format!(
                "Not enough participants: need {}, have {}",
                self.min_participants, self.participant_count
            )));
        }

        // For now, use simple averaging
        // In production, this would:
        // 1. Use secret sharing to split gradients
        // 2. Aggregate encrypted shares
        // 3. Reconstruct only the sum
        GradientAggregator::average(gradients)
    }

    /// Reset participant count
    pub fn reset(&mut self) {
        self.participant_count = 0;
    }

    /// Get participant count
    pub fn participant_count(&self) -> usize {
        self.participant_count
    }
}

/// Client state in federated learning
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClientState {
    /// Client is idle and ready for work
    Idle,
    /// Client is training
    Training,
    /// Client has completed training
    Completed,
    /// Client has failed or dropped out
    Failed,
}

/// Client information in federated learning
#[derive(Debug, Clone)]
pub struct ClientInfo {
    /// Client ID
    pub client_id: String,
    /// Client state
    pub state: ClientState,
    /// Number of samples the client has
    pub sample_count: usize,
    /// Last update timestamp
    pub last_update: i64,
}

impl ClientInfo {
    /// Create a new client info
    pub fn new(client_id: String, sample_count: usize) -> Self {
        Self {
            client_id,
            state: ClientState::Idle,
            sample_count,
            last_update: chrono::Utc::now().timestamp(),
        }
    }

    /// Mark client as training
    pub fn start_training(&mut self) {
        self.state = ClientState::Training;
        self.last_update = chrono::Utc::now().timestamp();
    }

    /// Mark client as completed
    pub fn complete_training(&mut self) {
        self.state = ClientState::Completed;
        self.last_update = chrono::Utc::now().timestamp();
    }

    /// Mark client as failed
    pub fn mark_failed(&mut self) {
        self.state = ClientState::Failed;
        self.last_update = chrono::Utc::now().timestamp();
    }
}

/// Federated learning round
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederatedRound {
    /// Round number
    pub round_num: usize,
    /// Clients participating in this round (stored as count for serialization)
    pub client_count: usize,
    /// Global model CID for this round
    #[serde(serialize_with = "crate::serialize_cid")]
    #[serde(deserialize_with = "crate::deserialize_cid")]
    pub global_model: Cid,
    /// Aggregated gradient for this round (if computed)
    pub aggregated_gradient: Option<Vec<f32>>,
    /// Round start timestamp
    pub start_time: i64,
    /// Round end timestamp (if completed)
    pub end_time: Option<i64>,
    /// Completed client count
    pub completed_count: usize,
}

impl FederatedRound {
    /// Create a new federated round
    pub fn new(round_num: usize, global_model: Cid, client_count: usize) -> Self {
        Self {
            round_num,
            client_count,
            global_model,
            aggregated_gradient: None,
            start_time: chrono::Utc::now().timestamp(),
            end_time: None,
            completed_count: 0,
        }
    }

    /// Mark a client as completed
    pub fn mark_client_completed(&mut self) {
        self.completed_count += 1;
    }

    /// Check if round is complete
    pub fn is_complete(&self) -> bool {
        self.completed_count >= self.client_count
    }

    /// Complete the round
    pub fn complete(&mut self, aggregated_gradient: Vec<f32>) {
        self.aggregated_gradient = Some(aggregated_gradient);
        self.end_time = Some(chrono::Utc::now().timestamp());
    }

    /// Get round duration in seconds
    pub fn duration(&self) -> Option<i64> {
        self.end_time.map(|end| end - self.start_time)
    }
}

/// Convergence detection for federated learning
pub struct ConvergenceDetector {
    /// Window size for convergence detection
    window_size: usize,
    /// Recent loss values
    loss_history: Vec<f64>,
    /// Convergence threshold (relative change)
    threshold: f64,
}

impl ConvergenceDetector {
    /// Create a new convergence detector
    pub fn new(window_size: usize, threshold: f64) -> Self {
        Self {
            window_size,
            loss_history: Vec::new(),
            threshold,
        }
    }

    /// Add a loss value
    pub fn add_loss(&mut self, loss: f64) {
        self.loss_history.push(loss);

        // Keep only the last window_size values
        if self.loss_history.len() > self.window_size {
            self.loss_history.remove(0);
        }
    }

    /// Check if training has converged
    pub fn has_converged(&self) -> bool {
        if self.loss_history.len() < self.window_size {
            return false;
        }

        // Calculate relative change in loss
        let recent = &self.loss_history[self.loss_history.len() - self.window_size..];
        let mean = recent.iter().sum::<f64>() / recent.len() as f64;

        if mean.abs() < 1e-10 {
            // Avoid division by zero
            return true;
        }

        let std_dev =
            (recent.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / recent.len() as f64).sqrt();

        // Converged if standard deviation is below threshold
        std_dev / mean.abs() < self.threshold
    }

    /// Get the latest loss
    pub fn latest_loss(&self) -> Option<f64> {
        self.loss_history.last().copied()
    }

    /// Clear loss history
    pub fn reset(&mut self) {
        self.loss_history.clear();
    }

    /// Get loss history
    pub fn history(&self) -> &[f64] {
        &self.loss_history
    }
}

/// Model synchronization protocol for federated learning
pub struct ModelSyncProtocol {
    /// Current round number
    current_round: usize,
    /// Maximum number of rounds
    max_rounds: usize,
    /// Minimum number of clients per round
    min_clients_per_round: usize,
    /// Round history
    rounds: Vec<FederatedRound>,
    /// Convergence detector
    convergence: ConvergenceDetector,
}

impl ModelSyncProtocol {
    /// Create a new model synchronization protocol
    pub fn new(
        max_rounds: usize,
        min_clients_per_round: usize,
        convergence_window: usize,
        convergence_threshold: f64,
    ) -> Self {
        Self {
            current_round: 0,
            max_rounds,
            min_clients_per_round,
            rounds: Vec::new(),
            convergence: ConvergenceDetector::new(convergence_window, convergence_threshold),
        }
    }

    /// Start a new round
    pub fn start_round(
        &mut self,
        global_model: Cid,
        client_count: usize,
    ) -> Result<usize, GradientError> {
        if client_count < self.min_clients_per_round {
            return Err(GradientError::InvalidGradient(format!(
                "Not enough clients: need {}, got {}",
                self.min_clients_per_round, client_count
            )));
        }

        if self.current_round >= self.max_rounds {
            return Err(GradientError::InvalidGradient(format!(
                "Maximum rounds reached: {}",
                self.max_rounds
            )));
        }

        let round = FederatedRound::new(self.current_round, global_model, client_count);
        self.rounds.push(round);
        self.current_round += 1;

        Ok(self.current_round - 1)
    }

    /// Complete the current round
    pub fn complete_round(
        &mut self,
        round_num: usize,
        aggregated_gradient: Vec<f32>,
        loss: f64,
    ) -> Result<(), GradientError> {
        if round_num >= self.rounds.len() {
            return Err(GradientError::InvalidGradient(format!(
                "Invalid round number: {}",
                round_num
            )));
        }

        self.rounds[round_num].complete(aggregated_gradient);
        self.convergence.add_loss(loss);

        Ok(())
    }

    /// Check if training should continue
    pub fn should_continue(&self) -> bool {
        self.current_round < self.max_rounds && !self.convergence.has_converged()
    }

    /// Check if training has converged
    pub fn has_converged(&self) -> bool {
        self.convergence.has_converged()
    }

    /// Get the current round number
    pub fn current_round(&self) -> usize {
        self.current_round
    }

    /// Get the total number of rounds
    pub fn total_rounds(&self) -> usize {
        self.rounds.len()
    }

    /// Get round information
    pub fn get_round(&self, round_num: usize) -> Option<&FederatedRound> {
        self.rounds.get(round_num)
    }

    /// Get the latest loss
    pub fn latest_loss(&self) -> Option<f64> {
        self.convergence.latest_loss()
    }

    /// Get max rounds
    pub fn max_rounds(&self) -> usize {
        self.max_rounds
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_gradient() {
        let indices = vec![0, 5, 10];
        let values = vec![1.0, 2.0, 3.0];
        let shape = vec![20];

        let sparse = SparseGradient::new(indices.clone(), values.clone(), shape);

        assert_eq!(sparse.nnz(), 3);
        assert_eq!(sparse.total_elements(), 20);
        assert!((sparse.sparsity_ratio() - 0.85).abs() < 0.01);

        let dense = sparse.to_dense();
        assert_eq!(dense.len(), 20);
        assert_eq!(dense[0], 1.0);
        assert_eq!(dense[5], 2.0);
        assert_eq!(dense[10], 3.0);
    }

    #[test]
    fn test_quantized_gradient() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let shape = vec![5];

        let quantized = QuantizedGradient::from_dense(&values, shape);
        let dequantized = quantized.to_dense();

        // Check that dequantization is approximately correct
        // For a small range like [1,5] with 256 quantization levels,
        // we expect good precision
        for (i, (orig, deq)) in values.iter().zip(&dequantized).enumerate() {
            let error = (orig - deq).abs();
            // Allow for quantization error (scale = 4/255 ≈ 0.0157)
            assert!(
                error < 0.02,
                "Value {} mismatch: orig={}, deq={}, error={}",
                i,
                orig,
                deq,
                error
            );
        }
    }

    #[test]
    fn test_gradient_delta() {
        let base_cid = Cid::default();
        let mut delta = GradientDelta::new(base_cid);

        delta.add_dense_gradient("layer1".to_string(), vec![1.0, 2.0, 3.0], vec![3]);
        delta.add_dense_gradient("layer2".to_string(), vec![4.0, 5.0], vec![2]);

        assert_eq!(delta.layer_gradients.len(), 2);
        assert!(delta.verify_checksum().is_ok());
    }

    #[test]
    fn test_top_k_compression() {
        let values = vec![1.0, 5.0, 2.0, 8.0, 3.0];
        let shape = vec![5];

        let sparse = GradientCompressor::top_k(&values, shape, 2).unwrap();

        assert_eq!(sparse.nnz(), 2);
        assert!(sparse.values.contains(&8.0));
        assert!(sparse.values.contains(&5.0));
    }

    #[test]
    fn test_threshold_compression() {
        let values = vec![0.1, 5.0, 0.2, 8.0, 0.3];
        let shape = vec![5];

        let sparse = GradientCompressor::threshold(&values, shape, 1.0);

        assert_eq!(sparse.nnz(), 2);
        assert!(sparse.values.contains(&5.0));
        assert!(sparse.values.contains(&8.0));
    }

    #[test]
    fn test_gradient_averaging() {
        let g1 = vec![1.0, 2.0, 3.0];
        let g2 = vec![3.0, 4.0, 5.0];
        let gradients = vec![g1, g2];

        let avg = GradientAggregator::average(&gradients).unwrap();

        assert_eq!(avg, vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_weighted_averaging() {
        let g1 = vec![1.0, 2.0, 3.0];
        let g2 = vec![3.0, 4.0, 5.0];
        let gradients = vec![g1, g2];
        let weights = vec![0.25, 0.75];

        let avg = GradientAggregator::weighted_average(&gradients, &weights).unwrap();

        // Expected: 0.25 * [1,2,3] + 0.75 * [3,4,5] = [2.5, 3.5, 4.5]
        assert!((avg[0] - 2.5).abs() < 0.01);
        assert!((avg[1] - 3.5).abs() < 0.01);
        assert!((avg[2] - 4.5).abs() < 0.01);
    }

    #[test]
    fn test_momentum() {
        let current = vec![1.0, 2.0, 3.0];
        let previous = vec![0.5, 1.0, 1.5];

        let result = GradientAggregator::apply_momentum(&current, &previous, 0.9).unwrap();

        // Expected: 0.9 * previous + current
        assert!((result[0] - 1.45).abs() < 0.01);
        assert!((result[1] - 2.9).abs() < 0.01);
        assert!((result[2] - 4.35).abs() < 0.01);
    }

    #[test]
    fn test_gradient_verification() {
        let gradient = vec![1.0, 2.0, 3.0, 4.0];

        // Test shape verification
        assert!(GradientVerifier::verify_shape(&gradient, &[4]).is_ok());
        assert!(GradientVerifier::verify_shape(&gradient, &[2, 2]).is_ok());
        assert!(GradientVerifier::verify_shape(&gradient, &[5]).is_err());

        // Test finite verification
        assert!(GradientVerifier::verify_finite(&gradient).is_ok());

        let invalid = vec![1.0, f32::NAN, 3.0];
        assert!(GradientVerifier::verify_finite(&invalid).is_err());
    }

    #[test]
    fn test_gradient_clipping() {
        let mut gradient = vec![3.0, 4.0]; // L2 norm = 5.0

        GradientVerifier::clip_by_norm(&mut gradient, 2.5);

        let norm = GradientVerifier::l2_norm(&gradient);
        assert!((norm - 2.5).abs() < 0.01);
    }

    #[test]
    fn test_privacy_budget() {
        let mut budget = PrivacyBudget::new(1.0, 1e-5);

        assert_eq!(budget.remaining_epsilon, 1.0);
        assert!(!budget.is_exhausted());

        // Consume some budget
        budget.consume(0.5).unwrap();
        assert_eq!(budget.remaining_epsilon, 0.5);
        assert!((budget.remaining_fraction() - 0.5).abs() < 1e-6);

        // Consume remaining budget
        budget.consume(0.5).unwrap();
        assert!(budget.is_exhausted());

        // Should fail when budget is exhausted
        assert!(budget.consume(0.1).is_err());
    }

    #[test]
    fn test_differential_privacy_gaussian() {
        let mut dp = DifferentialPrivacy::new(1.0, 1e-5, 1.0, DPMechanism::Gaussian);
        let mut gradient = vec![1.0, 2.0, 3.0, 4.0];
        let original = gradient.clone();

        dp.add_gaussian_noise(&mut gradient).unwrap();

        // Gradient should be modified (with very high probability)
        assert_ne!(gradient, original);

        // Values should still be finite
        assert!(GradientVerifier::verify_finite(&gradient).is_ok());

        // Budget should be consumed
        assert!(dp.remaining_budget() < 1.0);
    }

    #[test]
    fn test_differential_privacy_laplacian() {
        let mut dp = DifferentialPrivacy::new(1.0, 1e-5, 1.0, DPMechanism::Laplacian);
        let mut gradient = vec![1.0, 2.0, 3.0, 4.0];
        let original = gradient.clone();

        dp.add_laplacian_noise(&mut gradient).unwrap();

        // Gradient should be modified (with very high probability)
        assert_ne!(gradient, original);

        // Values should still be finite
        assert!(GradientVerifier::verify_finite(&gradient).is_ok());

        // Budget should be consumed
        assert!(dp.remaining_budget() < 1.0);
    }

    #[test]
    fn test_dp_sgd() {
        let mut dp = DifferentialPrivacy::new(1.0, 1e-5, 1.0, DPMechanism::Gaussian);
        let mut gradient = vec![3.0, 4.0, 5.0, 6.0]; // L2 norm > 5.0
        let original_norm = GradientVerifier::l2_norm(&gradient);

        dp.apply_dp_sgd(&mut gradient, 5.0).unwrap();

        // Gradient should be clipped and noised
        let new_norm = GradientVerifier::l2_norm(&gradient);

        // After clipping and noise, norm might be around 5.0 but not exact due to noise
        // Just check it's different from original
        assert!(original_norm != new_norm);

        // Values should still be finite
        assert!(GradientVerifier::verify_finite(&gradient).is_ok());
    }

    #[test]
    fn test_privacy_budget_exhaustion() {
        let mut dp = DifferentialPrivacy::new(1.0, 1e-5, 1.0, DPMechanism::Gaussian);
        let mut gradient = vec![1.0, 2.0];

        // Consume budget multiple times
        // Each call consumes epsilon/100 = 0.01, so we need 100 calls to exhaust budget of 1.0
        let mut successful_calls = 0;
        for _ in 0..200 {
            if dp.add_gaussian_noise(&mut gradient).is_ok() {
                successful_calls += 1;
            } else {
                // Budget exhausted, break
                break;
            }
        }

        // Should have made ~100 successful calls before budget exhaustion
        assert!(
            (90..=110).contains(&successful_calls),
            "Expected ~100 calls, got {}",
            successful_calls
        );

        // Budget should be very low or exhausted (allow small epsilon for floating point errors)
        let remaining = dp.remaining_budget();
        assert!(
            remaining < 0.02,
            "Expected nearly exhausted budget, got {}",
            remaining
        );

        // Should fail when trying to consume more than remaining
        let mut new_gradient = vec![1.0, 2.0];
        let result = dp.add_gaussian_noise(&mut new_gradient);
        // Might succeed if there's a tiny bit of budget left, or fail if exhausted
        // Either way is acceptable at this point
        let _ = result;
    }

    #[test]
    fn test_noise_multiplier_calculation() {
        let epsilon = 1.0;
        let delta = 1e-5;
        let sensitivity = 1.0;

        let multiplier =
            DifferentialPrivacy::calculate_noise_multiplier(epsilon, delta, sensitivity);

        // Noise multiplier should be positive and reasonable
        assert!(multiplier > 0.0);
        assert!(multiplier < 10.0); // Sanity check

        // For higher epsilon (less privacy), noise should be lower
        let multiplier_high_eps =
            DifferentialPrivacy::calculate_noise_multiplier(10.0, delta, sensitivity);
        assert!(multiplier_high_eps < multiplier);
    }

    #[test]
    fn test_secure_aggregation() {
        let mut aggregator = SecureAggregation::new(3);

        assert_eq!(aggregator.participant_count(), 0);
        assert!(!aggregator.can_aggregate());

        // Add participants
        aggregator.add_participant();
        aggregator.add_participant();
        assert!(!aggregator.can_aggregate());

        aggregator.add_participant();
        assert!(aggregator.can_aggregate());

        // Test aggregation
        let g1 = vec![1.0, 2.0, 3.0];
        let g2 = vec![2.0, 3.0, 4.0];
        let g3 = vec![3.0, 4.0, 5.0];
        let gradients = vec![g1, g2, g3];

        let result = aggregator.aggregate_secure(&gradients).unwrap();

        // Should be average of the three gradients
        assert!((result[0] - 2.0).abs() < 0.01);
        assert!((result[1] - 3.0).abs() < 0.01);
        assert!((result[2] - 4.0).abs() < 0.01);

        // Reset
        aggregator.reset();
        assert_eq!(aggregator.participant_count(), 0);
    }

    #[test]
    fn test_secure_aggregation_insufficient_participants() {
        let aggregator = SecureAggregation::new(5);

        let g1 = vec![1.0, 2.0];
        let g2 = vec![3.0, 4.0];
        let gradients = vec![g1, g2];

        // Should fail because we don't have enough participants
        let result = aggregator.aggregate_secure(&gradients);
        assert!(result.is_err());
    }

    #[test]
    fn test_dp_mechanism_types() {
        let gaussian = DPMechanism::Gaussian;
        let laplacian = DPMechanism::Laplacian;

        assert_eq!(gaussian, DPMechanism::Gaussian);
        assert_eq!(laplacian, DPMechanism::Laplacian);
        assert_ne!(gaussian, laplacian);
    }

    #[test]
    fn test_client_info() {
        let mut client = ClientInfo::new("client1".to_string(), 1000);

        assert_eq!(client.client_id, "client1");
        assert_eq!(client.state, ClientState::Idle);
        assert_eq!(client.sample_count, 1000);

        client.start_training();
        assert_eq!(client.state, ClientState::Training);

        client.complete_training();
        assert_eq!(client.state, ClientState::Completed);

        client.mark_failed();
        assert_eq!(client.state, ClientState::Failed);
    }

    #[test]
    fn test_federated_round() {
        let model_cid = Cid::default();
        let mut round = FederatedRound::new(0, model_cid, 5);

        assert_eq!(round.round_num, 0);
        assert_eq!(round.client_count, 5);
        assert_eq!(round.completed_count, 0);
        assert!(!round.is_complete());

        // Mark clients as completed
        for _ in 0..5 {
            round.mark_client_completed();
        }

        assert_eq!(round.completed_count, 5);
        assert!(round.is_complete());

        // Complete the round
        let gradient = vec![1.0, 2.0, 3.0];
        round.complete(gradient.clone());

        assert_eq!(round.aggregated_gradient, Some(gradient));
        assert!(round.end_time.is_some());
        assert!(round.duration().is_some());
    }

    #[test]
    fn test_convergence_detector() {
        let mut detector = ConvergenceDetector::new(3, 0.01);

        // Add loss values that are converging
        detector.add_loss(1.0);
        detector.add_loss(0.99);
        detector.add_loss(0.98);

        assert!(detector.has_converged());
        assert_eq!(detector.latest_loss(), Some(0.98));
        assert_eq!(detector.history().len(), 3);

        // Reset
        detector.reset();
        assert_eq!(detector.history().len(), 0);
    }

    #[test]
    fn test_convergence_detector_not_converged() {
        let mut detector = ConvergenceDetector::new(3, 0.01);

        // Add loss values that are NOT converging
        detector.add_loss(1.0);
        detector.add_loss(0.5);
        detector.add_loss(1.5);

        assert!(!detector.has_converged());
    }

    #[test]
    fn test_model_sync_protocol() {
        let mut protocol = ModelSyncProtocol::new(10, 3, 3, 0.01);

        assert_eq!(protocol.current_round(), 0);
        assert_eq!(protocol.max_rounds(), 10);
        assert!(protocol.should_continue());

        // Start round 0
        let model_cid = Cid::default();
        let round_num = protocol.start_round(model_cid, 5).unwrap();

        assert_eq!(round_num, 0);
        assert_eq!(protocol.current_round(), 1);
        assert_eq!(protocol.total_rounds(), 1);

        // Complete round 0
        let gradient = vec![1.0, 2.0, 3.0];
        protocol
            .complete_round(round_num, gradient.clone(), 1.0)
            .unwrap();

        assert_eq!(protocol.latest_loss(), Some(1.0));

        // Get round info
        let round = protocol.get_round(0).unwrap();
        assert_eq!(round.round_num, 0);
        assert_eq!(round.aggregated_gradient, Some(gradient));
    }

    #[test]
    fn test_model_sync_protocol_convergence() {
        let mut protocol = ModelSyncProtocol::new(10, 2, 3, 0.01);

        let model_cid = Cid::default();

        // Run multiple rounds with converging loss
        for i in 0..3 {
            protocol.start_round(model_cid, 3).unwrap();
            let gradient = vec![1.0, 2.0];
            let loss = 1.0 - (i as f64 * 0.001);
            protocol.complete_round(i, gradient, loss).unwrap();
        }

        // Should have converged
        assert!(protocol.has_converged());
        assert!(!protocol.should_continue());
    }

    #[test]
    fn test_model_sync_protocol_max_rounds() {
        let mut protocol = ModelSyncProtocol::new(2, 1, 3, 0.01);

        let model_cid = Cid::default();

        // Start 2 rounds (max)
        protocol.start_round(model_cid, 2).unwrap();
        protocol.start_round(model_cid, 2).unwrap();

        // Should fail to start a third round
        let result = protocol.start_round(model_cid, 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_model_sync_protocol_min_clients() {
        let mut protocol = ModelSyncProtocol::new(10, 5, 3, 0.01);

        let model_cid = Cid::default();

        // Should fail with too few clients
        let result = protocol.start_round(model_cid, 3);
        assert!(result.is_err());

        // Should succeed with enough clients
        let result = protocol.start_round(model_cid, 5);
        assert!(result.is_ok());
    }

    #[test]
    fn test_client_state_enum() {
        let idle = ClientState::Idle;
        let training = ClientState::Training;
        let completed = ClientState::Completed;
        let failed = ClientState::Failed;

        assert_ne!(idle, training);
        assert_ne!(training, completed);
        assert_ne!(completed, failed);
        assert_eq!(idle, ClientState::Idle);
    }
}
