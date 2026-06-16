//! GradientCheckpointer — gradient accumulation, checkpointing, and replay for
//! distributed training with fault tolerance.
//!
//! This module implements a production-grade gradient checkpointing system that
//! supports configurable accumulation modes (Sum, Mean, WeightedMean), L2-norm
//! gradient clipping, FNV-1a content checksums, and bounded checkpoint history
//! with automatic eviction.

use std::collections::{HashMap, VecDeque};
use thiserror::Error;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors produced by [`GradientCheckpointer`].
#[derive(Debug, Error, Clone, PartialEq)]
pub enum GradientCheckpointerError {
    /// No accumulated gradients to flush.
    #[error("no accumulated gradients available")]
    NoAccumulatedGradients,

    /// The requested checkpoint identifier does not exist.
    #[error("checkpoint not found")]
    CheckpointNotFound,

    /// The gradient tensor for `layer` has dimension `got` but the accumulation
    /// buffer for that layer expects `expected` elements.
    #[error("dimension mismatch for layer '{layer}': expected {expected}, got {got}")]
    DimensionMismatch {
        layer: String,
        expected: usize,
        got: usize,
    },
}

// ---------------------------------------------------------------------------
// Core data types
// ---------------------------------------------------------------------------

/// A gradient tensor associated with a named layer at a particular training step.
///
/// `norm` is always the L2 norm of `values` and is stored pre-computed to
/// avoid redundant recomputation during statistics gathering.
#[derive(Debug, Clone, PartialEq)]
pub struct GcGradientTensor {
    /// Identifier for the model layer this gradient belongs to.
    pub layer_id: String,
    /// Raw gradient values (one entry per parameter).
    pub values: Vec<f64>,
    /// The global training step at which this gradient was computed.
    pub step: u64,
    /// Pre-computed L2 norm of `values`.
    pub norm: f64,
}

impl GcGradientTensor {
    /// Construct a new `GcGradientTensor`, computing the L2 norm automatically.
    pub fn new(layer_id: impl Into<String>, values: Vec<f64>, step: u64) -> Self {
        let norm = Self::compute_norm(&values);
        Self {
            layer_id: layer_id.into(),
            values,
            step,
            norm,
        }
    }

    /// Compute the L2 (Euclidean) norm of a gradient value slice.
    ///
    /// Returns `sqrt(sum(v^2))`.
    #[inline]
    pub fn compute_norm(values: &[f64]) -> f64 {
        values.iter().map(|v| v * v).sum::<f64>().sqrt()
    }
}

// ---------------------------------------------------------------------------
// Checkpoint identifier
// ---------------------------------------------------------------------------

/// Newtype wrapper for checkpoint identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct CheckpointId(pub u64);

impl std::fmt::Display for CheckpointId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ckpt:{}", self.0)
    }
}

// ---------------------------------------------------------------------------
// Checkpoint
// ---------------------------------------------------------------------------

/// A snapshot of accumulated and (optionally clipped) gradients at a
/// particular training step.
#[derive(Debug, Clone)]
pub struct GcGradientCheckpoint {
    /// Unique identifier for this checkpoint.
    pub id: CheckpointId,
    /// The global step counter at the time `flush` was called.
    pub step: u64,
    /// Per-layer gradient tensors after accumulation and clipping.
    pub gradients: HashMap<String, GcGradientTensor>,
    /// Wall-clock timestamp (seconds since Unix epoch) at creation time,
    /// supplied by the caller of `flush`.
    pub created_at: u64,
    /// Approximate compressed size (in bytes) of the serialised gradient data.
    /// Currently approximated as `8 * total_values`.
    pub compressed_size: usize,
    /// FNV-1a hash over all gradient values (layer order sorted by layer_id).
    pub checksum: u64,
}

// ---------------------------------------------------------------------------
// Accumulation mode
// ---------------------------------------------------------------------------

/// Controls how multiple gradient tensors for the same layer are combined
/// during a `flush`.
#[derive(Debug, Clone)]
pub enum GcAccumulationMode {
    /// Element-wise sum of all accumulated tensors.
    Sum,
    /// Element-wise mean across all accumulated tensors.
    Mean,
    /// Weighted element-wise mean; weights must be the same length as the
    /// number of accumulated tensors for each layer.  If the layer has fewer
    /// tensors than weights, only the matching prefix of weights is used.
    WeightedMean {
        /// Per-tensor weights used during accumulation.
        weights: Vec<f64>,
    },
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for a [`GradientCheckpointer`].
#[derive(Debug, Clone)]
pub struct CheckpointerConfig {
    /// Maximum number of checkpoints to retain.  When a new checkpoint is
    /// created and this limit is already reached, the oldest checkpoint is
    /// evicted.
    pub max_checkpoints: usize,
    /// Value count threshold above which the compressed_size estimate uses a
    /// reduced coefficient.  Currently used only for the size approximation.
    pub compression_threshold: usize,
    /// Accumulation strategy applied during `flush`.
    pub accumulation_mode: GcAccumulationMode,
    /// If `Some(max_norm)`, gradients whose global L2 norm exceeds `max_norm`
    /// are rescaled so that the global norm equals `max_norm`.
    pub clip_norm: Option<f64>,
}

impl Default for CheckpointerConfig {
    fn default() -> Self {
        Self {
            max_checkpoints: 10,
            compression_threshold: 1000,
            accumulation_mode: GcAccumulationMode::Sum,
            clip_norm: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// Snapshot of checkpointer statistics.
#[derive(Debug, Clone, PartialEq)]
pub struct GcCheckpointerStats {
    /// Number of checkpoints currently stored.
    pub total_checkpoints: usize,
    /// Cumulative global step counter (incremented once per `accumulate` call).
    pub total_steps: u64,
    /// Mean global L2 norm across all stored checkpoints.
    pub avg_checkpoint_norm: f64,
    /// Maximum global L2 norm observed across all stored checkpoints.
    pub max_checkpoint_norm: f64,
    /// Total number of pending (not-yet-flushed) gradient tensors.
    pub pending_tensors: usize,
}

// ---------------------------------------------------------------------------
// Checkpointer
// ---------------------------------------------------------------------------

/// Production-grade gradient checkpointing engine.
///
/// Collects [`GcGradientTensor`] objects via [`accumulate`][Self::accumulate],
/// combines them into a [`GcGradientCheckpoint`] via [`flush`][Self::flush],
/// and maintains a bounded history of the most recent checkpoints.
pub struct GradientCheckpointer {
    /// Runtime configuration.
    pub config: CheckpointerConfig,
    /// Pending gradients, keyed by layer id.
    accumulated: HashMap<String, Vec<GcGradientTensor>>,
    /// Bounded history of flushed checkpoints.
    checkpoints: VecDeque<GcGradientCheckpoint>,
    /// Monotonically increasing checkpoint identifier source.
    next_checkpoint_id: u64,
    /// Cumulative step counter.  Incremented once per `accumulate` call.
    pub global_step: u64,
}

impl GradientCheckpointer {
    /// Create a new checkpointer with the given configuration.
    pub fn new(config: CheckpointerConfig) -> Self {
        Self {
            config,
            accumulated: HashMap::new(),
            checkpoints: VecDeque::new(),
            next_checkpoint_id: 0,
            global_step: 0,
        }
    }

    // -----------------------------------------------------------------------
    // Accumulation
    // -----------------------------------------------------------------------

    /// Push a gradient tensor into the accumulation buffer for its layer.
    ///
    /// The global step counter is incremented once per call.
    pub fn accumulate(&mut self, gradient: GcGradientTensor) {
        self.global_step += 1;
        self.accumulated
            .entry(gradient.layer_id.clone())
            .or_default()
            .push(gradient);
    }

    // -----------------------------------------------------------------------
    // Flush
    // -----------------------------------------------------------------------

    /// Flush accumulated gradients into a new checkpoint.
    ///
    /// Applies the configured accumulation mode over all pending tensors for
    /// each layer, optionally clips the resulting global norm, computes a
    /// FNV-1a checksum, and stores the checkpoint.  If the checkpoint count
    /// exceeds `max_checkpoints`, the oldest entry is evicted.
    ///
    /// # Errors
    ///
    /// Returns [`GradientCheckpointerError::NoAccumulatedGradients`] when
    /// there are no pending tensors.
    pub fn flush(&mut self, now: u64) -> Result<GcGradientCheckpoint, GradientCheckpointerError> {
        if self.accumulated.is_empty() {
            return Err(GradientCheckpointerError::NoAccumulatedGradients);
        }

        // Step 1 — accumulate per layer
        let mut merged: HashMap<String, GcGradientTensor> = HashMap::new();
        for (layer_id, tensors) in &self.accumulated {
            let merged_values = self.combine_tensors(layer_id, tensors)?;
            let step = tensors.last().map(|t| t.step).unwrap_or(self.global_step);
            let tensor = GcGradientTensor::new(layer_id.clone(), merged_values, step);
            merged.insert(layer_id.clone(), tensor);
        }

        // Step 2 — optional gradient clipping
        if let Some(max_norm) = self.config.clip_norm {
            let global_norm = Self::compute_global_norm_map(&merged);
            if global_norm > max_norm && global_norm > 0.0 {
                let scale = max_norm / global_norm;
                for tensor in merged.values_mut() {
                    for v in &mut tensor.values {
                        *v *= scale;
                    }
                    tensor.norm = GcGradientTensor::compute_norm(&tensor.values);
                }
            }
        }

        // Step 3 — checksum
        let checksum = compute_checksum(&merged);

        // Step 4 — compressed size estimate
        let total_values: usize = merged.values().map(|t| t.values.len()).sum();
        let compressed_size = if total_values > self.config.compression_threshold {
            (total_values as f64 * 8.0 * 0.6) as usize
        } else {
            total_values * 8
        };

        // Step 5 — build checkpoint
        let id = CheckpointId(self.next_checkpoint_id);
        self.next_checkpoint_id += 1;

        let checkpoint = GcGradientCheckpoint {
            id,
            step: self.global_step,
            gradients: merged,
            created_at: now,
            compressed_size,
            checksum,
        };

        // Step 6 — store with eviction
        if self.checkpoints.len() >= self.config.max_checkpoints {
            self.checkpoints.pop_front();
        }
        self.checkpoints.push_back(checkpoint.clone());

        // Step 7 — clear accumulation buffer
        self.accumulated.clear();

        Ok(checkpoint)
    }

    // -----------------------------------------------------------------------
    // Replay
    // -----------------------------------------------------------------------

    /// Return gradient tensors from `checkpoint` sorted by `layer_id`.
    pub fn replay(&self, checkpoint: &GcGradientCheckpoint) -> Vec<GcGradientTensor> {
        let mut tensors: Vec<GcGradientTensor> = checkpoint.gradients.values().cloned().collect();
        tensors.sort_by(|a, b| a.layer_id.cmp(&b.layer_id));
        tensors
    }

    // -----------------------------------------------------------------------
    // Lookup
    // -----------------------------------------------------------------------

    /// Return a reference to the most recently flushed checkpoint, if any.
    pub fn latest_checkpoint(&self) -> Option<&GcGradientCheckpoint> {
        self.checkpoints.back()
    }

    /// Return a reference to the checkpoint with the given `id`, if it still
    /// resides in the bounded history.
    pub fn checkpoint_by_id(&self, id: CheckpointId) -> Option<&GcGradientCheckpoint> {
        self.checkpoints.iter().find(|c| c.id == id)
    }

    // -----------------------------------------------------------------------
    // Diff / norms
    // -----------------------------------------------------------------------

    /// Compute the per-layer L2 distance between two checkpoints.
    ///
    /// For each layer present in either checkpoint the function returns the L2
    /// distance between the corresponding gradient value vectors.  If a layer
    /// is absent in one of the checkpoints its contribution is `0.0`.
    pub fn diff(&self, a: &GcGradientCheckpoint, b: &GcGradientCheckpoint) -> HashMap<String, f64> {
        let mut result: HashMap<String, f64> = HashMap::new();

        // Collect all layer ids from both checkpoints
        let mut all_layers: Vec<String> = a
            .gradients
            .keys()
            .chain(b.gradients.keys())
            .cloned()
            .collect();
        all_layers.sort();
        all_layers.dedup();

        for layer in all_layers {
            let dist = match (a.gradients.get(&layer), b.gradients.get(&layer)) {
                (Some(ta), Some(tb)) => l2_distance(&ta.values, &tb.values),
                (Some(ta), None) => GcGradientTensor::compute_norm(&ta.values),
                (None, Some(tb)) => GcGradientTensor::compute_norm(&tb.values),
                (None, None) => 0.0,
            };
            result.insert(layer, dist);
        }

        result
    }

    /// Compute the global L2 norm of a checkpoint (sqrt of sum of squared
    /// per-layer norms).
    pub fn global_norm(&self, checkpoint: &GcGradientCheckpoint) -> f64 {
        Self::compute_global_norm_map(&checkpoint.gradients)
    }

    // -----------------------------------------------------------------------
    // Pending state
    // -----------------------------------------------------------------------

    /// Return a sorted list of layer IDs that have pending (unflushed) gradients.
    pub fn pending_layers(&self) -> Vec<&str> {
        let mut ids: Vec<&str> = self.accumulated.keys().map(String::as_str).collect();
        ids.sort();
        ids
    }

    /// Return the total count of pending gradient tensors across all layers.
    pub fn pending_count(&self) -> usize {
        self.accumulated.values().map(Vec::len).sum()
    }

    // -----------------------------------------------------------------------
    // Statistics
    // -----------------------------------------------------------------------

    /// Return a snapshot of current checkpointer statistics.
    pub fn stats(&self) -> GcCheckpointerStats {
        let norms: Vec<f64> = self
            .checkpoints
            .iter()
            .map(|c| Self::compute_global_norm_map(&c.gradients))
            .collect();

        let total = norms.len();
        let avg_checkpoint_norm = if total == 0 {
            0.0
        } else {
            norms.iter().sum::<f64>() / total as f64
        };
        let max_checkpoint_norm = norms.iter().cloned().fold(0.0_f64, f64::max);

        GcCheckpointerStats {
            total_checkpoints: total,
            total_steps: self.global_step,
            avg_checkpoint_norm,
            max_checkpoint_norm,
            pending_tensors: self.pending_count(),
        }
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    fn combine_tensors(
        &self,
        layer_id: &str,
        tensors: &[GcGradientTensor],
    ) -> Result<Vec<f64>, GradientCheckpointerError> {
        debug_assert!(
            !tensors.is_empty(),
            "combine_tensors called with empty slice"
        );

        let dim = tensors[0].values.len();

        // Validate that all tensors have the same dimensionality
        for t in tensors.iter().skip(1) {
            if t.values.len() != dim {
                return Err(GradientCheckpointerError::DimensionMismatch {
                    layer: layer_id.to_string(),
                    expected: dim,
                    got: t.values.len(),
                });
            }
        }

        match &self.config.accumulation_mode {
            GcAccumulationMode::Sum => {
                let mut result = vec![0.0f64; dim];
                for t in tensors {
                    for (r, v) in result.iter_mut().zip(t.values.iter()) {
                        *r += v;
                    }
                }
                Ok(result)
            }

            GcAccumulationMode::Mean => {
                let mut result = vec![0.0f64; dim];
                for t in tensors {
                    for (r, v) in result.iter_mut().zip(t.values.iter()) {
                        *r += v;
                    }
                }
                let n = tensors.len() as f64;
                for r in &mut result {
                    *r /= n;
                }
                Ok(result)
            }

            GcAccumulationMode::WeightedMean { weights } => {
                let mut result = vec![0.0f64; dim];
                let mut weight_sum = 0.0f64;

                for (i, t) in tensors.iter().enumerate() {
                    let w = weights.get(i).copied().unwrap_or(1.0);
                    weight_sum += w;
                    for (r, v) in result.iter_mut().zip(t.values.iter()) {
                        *r += v * w;
                    }
                }

                if weight_sum != 0.0 {
                    for r in &mut result {
                        *r /= weight_sum;
                    }
                }

                Ok(result)
            }
        }
    }

    fn compute_global_norm_map(gradients: &HashMap<String, GcGradientTensor>) -> f64 {
        gradients
            .values()
            .map(|t| t.norm * t.norm)
            .sum::<f64>()
            .sqrt()
    }
}

// ---------------------------------------------------------------------------
// Free-standing helpers
// ---------------------------------------------------------------------------

/// Compute FNV-1a hash over a slice of `f64` values.
///
/// This is the standard 64-bit FNV-1a hash applied byte-by-byte over the
/// little-endian bit pattern of each float.
#[inline]
pub fn fnv1a_f64_slice(values: &[f64]) -> u64 {
    let mut h: u64 = 14695981039346656037;
    for v in values {
        for b in v.to_bits().to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(1099511628211);
        }
    }
    h
}

/// Compute the L2 distance between two equal-length slices.
///
/// Returns `0.0` when both slices are empty.
#[inline]
fn l2_distance(a: &[f64], b: &[f64]) -> f64 {
    let min_len = a.len().min(b.len());
    let mut sum = 0.0f64;
    for i in 0..min_len {
        let d = a[i] - b[i];
        sum += d * d;
    }
    // If one slice is longer, treat the excess elements as 0 in the other
    for &v in a.iter().skip(min_len) {
        sum += v * v;
    }
    for &v in b.iter().skip(min_len) {
        sum += v * v;
    }
    sum.sqrt()
}

/// Compute the FNV-1a checksum of all gradient values in a checkpoint, with
/// layers processed in sorted order.
fn compute_checksum(gradients: &HashMap<String, GcGradientTensor>) -> u64 {
    let mut layer_ids: Vec<&String> = gradients.keys().collect();
    layer_ids.sort();

    let mut all_values: Vec<f64> = Vec::new();
    for id in layer_ids {
        if let Some(t) = gradients.get(id) {
            all_values.extend_from_slice(&t.values);
        }
    }

    fnv1a_f64_slice(&all_values)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::{
        fnv1a_f64_slice, CheckpointId, CheckpointerConfig, GcAccumulationMode, GcGradientTensor,
        GradientCheckpointer, GradientCheckpointerError,
    };

    // -----------------------------------------------------------------------
    // Helper constructors
    // -----------------------------------------------------------------------

    fn default_checkpointer() -> GradientCheckpointer {
        GradientCheckpointer::new(CheckpointerConfig::default())
    }

    fn tensor(layer: &str, values: Vec<f64>, step: u64) -> GcGradientTensor {
        GcGradientTensor::new(layer, values, step)
    }

    // -----------------------------------------------------------------------
    // GcGradientTensor
    // -----------------------------------------------------------------------

    #[test]
    fn test_gradient_tensor_norm_zero() {
        let t = tensor("l0", vec![0.0, 0.0, 0.0], 1);
        assert_eq!(t.norm, 0.0);
    }

    #[test]
    fn test_gradient_tensor_norm_unit() {
        let t = tensor("l0", vec![1.0, 0.0, 0.0], 1);
        assert!((t.norm - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_gradient_tensor_norm_3_4_5() {
        let t = tensor("l0", vec![3.0, 4.0], 1);
        assert!((t.norm - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_gradient_tensor_compute_norm_static() {
        let norm = GcGradientTensor::compute_norm(&[1.0, 2.0, 2.0]);
        assert!((norm - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_gradient_tensor_layer_id_stored() {
        let t = tensor("my_layer", vec![1.0], 42);
        assert_eq!(t.layer_id, "my_layer");
        assert_eq!(t.step, 42);
    }

    // -----------------------------------------------------------------------
    // CheckpointId
    // -----------------------------------------------------------------------

    #[test]
    fn test_checkpoint_id_display() {
        let id = CheckpointId(7);
        assert_eq!(id.to_string(), "ckpt:7");
    }

    #[test]
    fn test_checkpoint_id_ordering() {
        assert!(CheckpointId(1) < CheckpointId(2));
        assert_eq!(CheckpointId(5), CheckpointId(5));
    }

    // -----------------------------------------------------------------------
    // Accumulate & global_step
    // -----------------------------------------------------------------------

    #[test]
    fn test_accumulate_increments_global_step() {
        let mut cp = default_checkpointer();
        assert_eq!(cp.global_step, 0);
        cp.accumulate(tensor("l0", vec![1.0], 1));
        assert_eq!(cp.global_step, 1);
        cp.accumulate(tensor("l0", vec![2.0], 2));
        assert_eq!(cp.global_step, 2);
    }

    #[test]
    fn test_pending_count_and_layers() {
        let mut cp = default_checkpointer();
        cp.accumulate(tensor("layer_a", vec![1.0], 1));
        cp.accumulate(tensor("layer_b", vec![2.0], 2));
        cp.accumulate(tensor("layer_a", vec![3.0], 3));

        assert_eq!(cp.pending_count(), 3);
        let layers = cp.pending_layers();
        assert_eq!(layers, vec!["layer_a", "layer_b"]);
    }

    #[test]
    fn test_pending_layers_sorted() {
        let mut cp = default_checkpointer();
        cp.accumulate(tensor("z_layer", vec![1.0], 1));
        cp.accumulate(tensor("a_layer", vec![1.0], 2));
        cp.accumulate(tensor("m_layer", vec![1.0], 3));
        let layers = cp.pending_layers();
        assert_eq!(layers, vec!["a_layer", "m_layer", "z_layer"]);
    }

    #[test]
    fn test_flush_clears_accumulated() {
        let mut cp = default_checkpointer();
        cp.accumulate(tensor("l0", vec![1.0], 1));
        let _ckpt = cp.flush(100).expect("flush failed");
        assert_eq!(cp.pending_count(), 0);
        assert!(cp.pending_layers().is_empty());
    }

    // -----------------------------------------------------------------------
    // Flush with no accumulated gradients
    // -----------------------------------------------------------------------

    #[test]
    fn test_flush_empty_returns_error() {
        let mut cp = default_checkpointer();
        let result = cp.flush(0);
        assert!(matches!(
            result,
            Err(GradientCheckpointerError::NoAccumulatedGradients)
        ));
    }

    // -----------------------------------------------------------------------
    // Accumulation modes
    // -----------------------------------------------------------------------

    #[test]
    fn test_flush_sum_mode() {
        let mut cp = default_checkpointer(); // default mode = Sum
        cp.accumulate(tensor("l0", vec![1.0, 2.0], 1));
        cp.accumulate(tensor("l0", vec![3.0, 4.0], 2));
        let ckpt = cp.flush(0).expect("flush failed");
        let t = ckpt.gradients.get("l0").expect("layer missing");
        assert!((t.values[0] - 4.0).abs() < 1e-12);
        assert!((t.values[1] - 6.0).abs() < 1e-12);
    }

    #[test]
    fn test_flush_mean_mode() {
        let config = CheckpointerConfig {
            accumulation_mode: GcAccumulationMode::Mean,
            ..Default::default()
        };
        let mut cp = GradientCheckpointer::new(config);
        cp.accumulate(tensor("l0", vec![2.0, 4.0], 1));
        cp.accumulate(tensor("l0", vec![4.0, 8.0], 2));
        let ckpt = cp.flush(0).expect("flush failed");
        let t = ckpt.gradients.get("l0").expect("layer missing");
        assert!((t.values[0] - 3.0).abs() < 1e-12);
        assert!((t.values[1] - 6.0).abs() < 1e-12);
    }

    #[test]
    fn test_flush_weighted_mean_mode() {
        let config = CheckpointerConfig {
            accumulation_mode: GcAccumulationMode::WeightedMean {
                weights: vec![1.0, 3.0],
            },
            ..Default::default()
        };
        let mut cp = GradientCheckpointer::new(config);
        cp.accumulate(tensor("l0", vec![0.0, 0.0], 1)); // weight 1
        cp.accumulate(tensor("l0", vec![4.0, 8.0], 2)); // weight 3
        let ckpt = cp.flush(0).expect("flush failed");
        let t = ckpt.gradients.get("l0").expect("layer missing");
        // (0*1 + 4*3) / 4 = 3.0
        assert!((t.values[0] - 3.0).abs() < 1e-12);
        // (0*1 + 8*3) / 4 = 6.0
        assert!((t.values[1] - 6.0).abs() < 1e-12);
    }

    #[test]
    fn test_flush_weighted_mean_uses_unit_weights_for_missing() {
        // Only one weight provided; second tensor should fall back to weight 1
        let config = CheckpointerConfig {
            accumulation_mode: GcAccumulationMode::WeightedMean { weights: vec![2.0] },
            ..Default::default()
        };
        let mut cp = GradientCheckpointer::new(config);
        cp.accumulate(tensor("l0", vec![2.0], 1)); // weight 2
        cp.accumulate(tensor("l0", vec![2.0], 2)); // weight 1 (fallback)
        let ckpt = cp.flush(0).expect("flush failed");
        let t = ckpt.gradients.get("l0").expect("layer missing");
        // (2*2 + 2*1) / 3 = 2.0
        assert!((t.values[0] - 2.0).abs() < 1e-12);
    }

    // -----------------------------------------------------------------------
    // Gradient clipping
    // -----------------------------------------------------------------------

    #[test]
    fn test_clip_norm_scales_down() {
        let config = CheckpointerConfig {
            clip_norm: Some(1.0),
            ..Default::default()
        };
        let mut cp = GradientCheckpointer::new(config);
        // norm = 5.0 (3-4-5 right triangle)
        cp.accumulate(tensor("l0", vec![3.0, 4.0], 1));
        let ckpt = cp.flush(0).expect("flush failed");
        let global = cp.global_norm(&ckpt);
        assert!(global <= 1.0 + 1e-9, "global norm {} > 1.0", global);
    }

    #[test]
    fn test_clip_norm_no_change_when_below_threshold() {
        let config = CheckpointerConfig {
            clip_norm: Some(10.0),
            ..Default::default()
        };
        let mut cp = GradientCheckpointer::new(config);
        cp.accumulate(tensor("l0", vec![1.0, 1.0], 1));
        let ckpt = cp.flush(0).expect("flush failed");
        let t = ckpt.gradients.get("l0").expect("layer missing");
        // Values should be unchanged (norm ≈ 1.414 < 10)
        assert!((t.values[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_clip_norm_multi_layer() {
        // Two layers each with norm 3.0 → global norm = sqrt(18) ≈ 4.24
        let config = CheckpointerConfig {
            clip_norm: Some(2.0),
            ..Default::default()
        };
        let mut cp = GradientCheckpointer::new(config);
        cp.accumulate(tensor("l0", vec![3.0, 0.0], 1));
        cp.accumulate(tensor("l1", vec![0.0, 3.0], 1));
        let ckpt = cp.flush(0).expect("flush failed");
        let global = cp.global_norm(&ckpt);
        assert!(global <= 2.0 + 1e-9, "global norm {} > 2.0", global);
    }

    // -----------------------------------------------------------------------
    // Checkpoint eviction
    // -----------------------------------------------------------------------

    #[test]
    fn test_max_checkpoints_eviction() {
        let config = CheckpointerConfig {
            max_checkpoints: 3,
            ..Default::default()
        };
        let mut cp = GradientCheckpointer::new(config);
        let mut ids = Vec::new();
        for i in 0..5u64 {
            cp.accumulate(tensor("l0", vec![i as f64], i));
            let ckpt = cp.flush(i).expect("flush failed");
            ids.push(ckpt.id);
        }
        // Only the last 3 checkpoints should remain
        assert!(cp.checkpoint_by_id(ids[0]).is_none());
        assert!(cp.checkpoint_by_id(ids[1]).is_none());
        assert!(cp.checkpoint_by_id(ids[2]).is_some());
        assert!(cp.checkpoint_by_id(ids[3]).is_some());
        assert!(cp.checkpoint_by_id(ids[4]).is_some());
    }

    #[test]
    fn test_latest_checkpoint_returns_last_flushed() {
        let mut cp = default_checkpointer();
        cp.accumulate(tensor("l0", vec![1.0], 1));
        let first = cp.flush(10).expect("flush failed");
        cp.accumulate(tensor("l0", vec![2.0], 2));
        let second = cp.flush(20).expect("flush failed");
        let latest = cp.latest_checkpoint().expect("no latest");
        assert_eq!(latest.id, second.id);
        assert_ne!(latest.id, first.id);
    }

    #[test]
    fn test_latest_checkpoint_none_initially() {
        let cp = default_checkpointer();
        assert!(cp.latest_checkpoint().is_none());
    }

    #[test]
    fn test_checkpoint_by_id_found() {
        let mut cp = default_checkpointer();
        cp.accumulate(tensor("l0", vec![1.0], 1));
        let ckpt = cp.flush(0).expect("flush failed");
        let found = cp.checkpoint_by_id(ckpt.id).expect("not found");
        assert_eq!(found.id, ckpt.id);
    }

    #[test]
    fn test_checkpoint_by_id_not_found() {
        let cp = default_checkpointer();
        assert!(cp.checkpoint_by_id(CheckpointId(9999)).is_none());
    }

    // -----------------------------------------------------------------------
    // Replay
    // -----------------------------------------------------------------------

    #[test]
    fn test_replay_sorted_by_layer_id() {
        let mut cp = default_checkpointer();
        cp.accumulate(tensor("z_layer", vec![3.0], 1));
        cp.accumulate(tensor("a_layer", vec![1.0], 2));
        cp.accumulate(tensor("m_layer", vec![2.0], 3));
        let ckpt = cp.flush(0).expect("flush failed");
        let replayed = cp.replay(&ckpt);
        let ids: Vec<&str> = replayed.iter().map(|t| t.layer_id.as_str()).collect();
        assert_eq!(ids, vec!["a_layer", "m_layer", "z_layer"]);
    }

    #[test]
    fn test_replay_values_preserved() {
        let mut cp = default_checkpointer();
        cp.accumulate(tensor("l0", vec![1.5, 2.5], 1));
        let ckpt = cp.flush(0).expect("flush failed");
        let replayed = cp.replay(&ckpt);
        assert_eq!(replayed.len(), 1);
        assert!((replayed[0].values[0] - 1.5).abs() < 1e-12);
        assert!((replayed[0].values[1] - 2.5).abs() < 1e-12);
    }

    // -----------------------------------------------------------------------
    // Diff
    // -----------------------------------------------------------------------

    #[test]
    fn test_diff_same_checkpoint_is_zero() {
        let mut cp = default_checkpointer();
        cp.accumulate(tensor("l0", vec![1.0, 2.0], 1));
        let ckpt = cp.flush(0).expect("flush failed");
        let diff = cp.diff(&ckpt, &ckpt);
        let d = diff["l0"];
        assert!(d.abs() < 1e-12, "expected 0 distance, got {}", d);
    }

    #[test]
    fn test_diff_known_distance() {
        let mut cp = default_checkpointer();
        cp.accumulate(tensor("l0", vec![0.0, 0.0], 1));
        let a = cp.flush(0).expect("flush a");
        cp.accumulate(tensor("l0", vec![3.0, 4.0], 2));
        let b = cp.flush(1).expect("flush b");
        let diff = cp.diff(&a, &b);
        assert!(
            (diff["l0"] - 5.0).abs() < 1e-12,
            "expected 5.0, got {}",
            diff["l0"]
        );
    }

    #[test]
    fn test_diff_missing_layer_returns_norm() {
        let mut cp = default_checkpointer();
        cp.accumulate(tensor("l0", vec![3.0, 4.0], 1));
        let a = cp.flush(0).expect("flush a");
        // b has l1 but not l0
        cp.accumulate(tensor("l1", vec![1.0], 2));
        let b = cp.flush(1).expect("flush b");
        let diff = cp.diff(&a, &b);
        // l0 is only in a — distance = norm(l0) = 5.0
        assert!(
            (diff["l0"] - 5.0).abs() < 1e-12,
            "expected 5.0, got {}",
            diff["l0"]
        );
        // l1 is only in b — distance = norm(l1) = 1.0
        assert!(
            (diff["l1"] - 1.0).abs() < 1e-12,
            "expected 1.0, got {}",
            diff["l1"]
        );
    }

    // -----------------------------------------------------------------------
    // Global norm
    // -----------------------------------------------------------------------

    #[test]
    fn test_global_norm_single_layer() {
        let mut cp = default_checkpointer();
        cp.accumulate(tensor("l0", vec![3.0, 4.0], 1));
        let ckpt = cp.flush(0).expect("flush failed");
        let gnorm = cp.global_norm(&ckpt);
        assert!((gnorm - 5.0).abs() < 1e-12, "expected 5.0, got {}", gnorm);
    }

    #[test]
    fn test_global_norm_multi_layer() {
        let mut cp = default_checkpointer();
        // layer_0 norm = 3, layer_1 norm = 4 → global = 5
        cp.accumulate(tensor("l0", vec![3.0, 0.0], 1));
        cp.accumulate(tensor("l1", vec![0.0, 4.0], 2));
        let ckpt = cp.flush(0).expect("flush failed");
        let gnorm = cp.global_norm(&ckpt);
        assert!((gnorm - 5.0).abs() < 1e-12, "expected 5.0, got {}", gnorm);
    }

    // -----------------------------------------------------------------------
    // Stats
    // -----------------------------------------------------------------------

    #[test]
    fn test_stats_empty() {
        let cp = default_checkpointer();
        let s = cp.stats();
        assert_eq!(s.total_checkpoints, 0);
        assert_eq!(s.total_steps, 0);
        assert_eq!(s.avg_checkpoint_norm, 0.0);
        assert_eq!(s.max_checkpoint_norm, 0.0);
        assert_eq!(s.pending_tensors, 0);
    }

    #[test]
    fn test_stats_after_flush() {
        let mut cp = default_checkpointer();
        cp.accumulate(tensor("l0", vec![3.0, 4.0], 1));
        let _ = cp.flush(0).expect("flush failed");
        let s = cp.stats();
        assert_eq!(s.total_checkpoints, 1);
        assert_eq!(s.total_steps, 1);
        // global norm = 5.0
        assert!((s.avg_checkpoint_norm - 5.0).abs() < 1e-9);
        assert!((s.max_checkpoint_norm - 5.0).abs() < 1e-9);
        assert_eq!(s.pending_tensors, 0);
    }

    #[test]
    fn test_stats_pending_tensors() {
        let mut cp = default_checkpointer();
        cp.accumulate(tensor("l0", vec![1.0], 1));
        cp.accumulate(tensor("l1", vec![2.0], 2));
        let s = cp.stats();
        assert_eq!(s.pending_tensors, 2);
    }

    // -----------------------------------------------------------------------
    // Checksum
    // -----------------------------------------------------------------------

    #[test]
    fn test_checksum_deterministic() {
        let mut cp = default_checkpointer();
        cp.accumulate(tensor("l0", vec![1.0, 2.0, 3.0], 1));
        let a = cp.flush(0).expect("flush a");
        let mut cp2 = default_checkpointer();
        cp2.accumulate(tensor("l0", vec![1.0, 2.0, 3.0], 1));
        let b = cp2.flush(0).expect("flush b");
        assert_eq!(a.checksum, b.checksum);
    }

    #[test]
    fn test_checksum_differs_for_different_values() {
        let mut cp = default_checkpointer();
        cp.accumulate(tensor("l0", vec![1.0], 1));
        let a = cp.flush(0).expect("flush a");
        cp.accumulate(tensor("l0", vec![2.0], 2));
        let b = cp.flush(1).expect("flush b");
        assert_ne!(a.checksum, b.checksum);
    }

    #[test]
    fn test_fnv1a_empty_slice() {
        let h = fnv1a_f64_slice(&[]);
        assert_eq!(h, 14695981039346656037u64);
    }

    #[test]
    fn test_fnv1a_known_value() {
        let h1 = fnv1a_f64_slice(&[1.0]);
        let h2 = fnv1a_f64_slice(&[1.0]);
        assert_eq!(h1, h2);
        // A different value should produce a different hash
        let h3 = fnv1a_f64_slice(&[2.0]);
        assert_ne!(h1, h3);
    }

    // -----------------------------------------------------------------------
    // Dimension mismatch
    // -----------------------------------------------------------------------

    #[test]
    fn test_dimension_mismatch_error() {
        let mut cp = default_checkpointer();
        cp.accumulate(tensor("l0", vec![1.0, 2.0], 1));
        cp.accumulate(tensor("l0", vec![1.0, 2.0, 3.0], 2)); // wrong dim
        let result = cp.flush(0);
        assert!(matches!(
            result,
            Err(GradientCheckpointerError::DimensionMismatch { .. })
        ));
    }

    // -----------------------------------------------------------------------
    // Checkpoint metadata
    // -----------------------------------------------------------------------

    #[test]
    fn test_checkpoint_created_at() {
        let mut cp = default_checkpointer();
        cp.accumulate(tensor("l0", vec![1.0], 1));
        let ckpt = cp.flush(12345).expect("flush failed");
        assert_eq!(ckpt.created_at, 12345);
    }

    #[test]
    fn test_compressed_size_positive() {
        let mut cp = default_checkpointer();
        cp.accumulate(tensor("l0", vec![1.0; 10], 1));
        let ckpt = cp.flush(0).expect("flush failed");
        assert!(ckpt.compressed_size > 0);
    }

    #[test]
    fn test_checkpoint_ids_are_monotonically_increasing() {
        let mut cp = default_checkpointer();
        let mut prev = CheckpointId(0);
        for i in 0..5u64 {
            cp.accumulate(tensor("l0", vec![i as f64], i));
            let ckpt = cp.flush(i).expect("flush failed");
            if i > 0 {
                assert!(ckpt.id > prev, "id {:?} not > {:?}", ckpt.id, prev);
            }
            prev = ckpt.id;
        }
    }

    #[test]
    fn test_step_recorded_in_checkpoint() {
        let mut cp = default_checkpointer();
        cp.accumulate(tensor("l0", vec![1.0], 1));
        cp.accumulate(tensor("l0", vec![2.0], 2));
        let ckpt = cp.flush(0).expect("flush failed");
        // global_step was incremented twice
        assert_eq!(ckpt.step, 2);
    }

    // -----------------------------------------------------------------------
    // Large-scale / integration
    // -----------------------------------------------------------------------

    #[test]
    fn test_multiple_layers_flush_and_replay() {
        let mut cp = default_checkpointer();
        for layer in ["encoder", "decoder", "classifier"] {
            for step in 0..5u64 {
                cp.accumulate(tensor(layer, vec![step as f64, (step + 1) as f64], step));
            }
        }
        let ckpt = cp.flush(999).expect("flush failed");
        assert_eq!(ckpt.gradients.len(), 3);
        let replayed = cp.replay(&ckpt);
        assert_eq!(replayed.len(), 3);
        // Verify sorted order
        assert_eq!(replayed[0].layer_id, "classifier");
        assert_eq!(replayed[1].layer_id, "decoder");
        assert_eq!(replayed[2].layer_id, "encoder");
    }

    #[test]
    fn test_multiple_flush_cycles() {
        let mut cp = default_checkpointer();
        for cycle in 0..5u64 {
            cp.accumulate(tensor("l0", vec![cycle as f64], cycle));
            let ckpt = cp.flush(cycle).expect("flush failed");
            assert_eq!(ckpt.id, CheckpointId(cycle));
        }
        let s = cp.stats();
        assert_eq!(s.total_checkpoints, 5);
    }

    #[test]
    fn test_no_unwrap_path_checkpoint_not_found_error() {
        let err = GradientCheckpointerError::CheckpointNotFound;
        assert_eq!(err.to_string(), "checkpoint not found");
    }

    #[test]
    fn test_dimension_mismatch_error_message() {
        let err = GradientCheckpointerError::DimensionMismatch {
            layer: "l0".to_string(),
            expected: 4,
            got: 3,
        };
        let msg = err.to_string();
        assert!(msg.contains("l0"));
        assert!(msg.contains('4'));
        assert!(msg.contains('3'));
    }
}
