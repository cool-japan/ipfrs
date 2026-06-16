//! Embedding Pipeline — preprocess raw content into normalized vectors for HNSW insertion.
//!
//! Accepts bytes, text, structured data, or pre-computed embeddings, applies
//! truncation/padding and a normalization strategy, and returns a `Vec<f32>` ready
//! for indexing.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use thiserror::Error;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors produced by [`EmbeddingPipeline`].
#[derive(Debug, Error)]
pub enum PipelineError {
    /// Input was empty and cannot produce a meaningful embedding.
    #[error("empty input")]
    EmptyInput,

    /// The supplied embedding has the wrong number of dimensions and
    /// `truncate_to_dims` is disabled.
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    /// The vector contains non-finite values (NaN / infinity).
    #[error("invalid vector: {0}")]
    InvalidVector(String),
}

// ---------------------------------------------------------------------------
// Input enum
// ---------------------------------------------------------------------------

/// The kind of content that the pipeline accepts.
#[derive(Debug, Clone)]
pub enum EmbeddingInput {
    /// Raw bytes with an associated MIME type hint.
    RawBytes { data: Vec<u8>, mime_type: String },
    /// UTF-8 text, optionally tagged with a BCP-47 language code.
    Text {
        content: String,
        language: Option<String>,
    },
    /// Key/value structured data (e.g. document fields).
    Structured { fields: HashMap<String, String> },
    /// A pre-computed embedding — passed through dimensionality adjustment only.
    Embedding { vector: Vec<f32> },
}

// ---------------------------------------------------------------------------
// Normalization strategy
// ---------------------------------------------------------------------------

/// How the raw float vector is normalised before being returned.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum NormalizationStrategy {
    /// Do not modify values.
    None,
    /// Divide each element by the L2 norm so the result has unit length.
    #[default]
    L2,
    /// Scale the vector to the range [0, 1] using min and max.
    MinMax,
    /// Subtract the mean and divide by the standard deviation.
    ZScore,
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Configuration for an [`EmbeddingPipeline`].
#[derive(Debug, Clone)]
pub struct EmbeddingPipelineConfig {
    /// Target dimensionality for all output vectors.
    pub dimensions: usize,
    /// Which normalization strategy to apply.
    pub normalization: NormalizationStrategy,
    /// Whether to truncate or pad vectors to exactly `dimensions`.
    pub truncate_to_dims: bool,
    /// Scalar used to pad vectors that are shorter than `dimensions`.
    pub pad_value: f32,
}

impl Default for EmbeddingPipelineConfig {
    fn default() -> Self {
        Self {
            dimensions: 128,
            normalization: NormalizationStrategy::L2,
            truncate_to_dims: true,
            pad_value: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------

/// Atomic counters tracking pipeline activity.
#[derive(Debug, Default)]
pub struct PipelineStats {
    /// Number of inputs that produced a successful embedding.
    pub total_processed: AtomicU64,
    /// Total number of bytes that flowed through `RawBytes` inputs.
    pub total_bytes_processed: AtomicU64,
    /// Number of inputs that resulted in a [`PipelineError`].
    pub total_errors: AtomicU64,
}

impl PipelineStats {
    /// Return a point-in-time snapshot of the counters.
    pub fn snapshot(&self) -> PipelineStatsSnapshot {
        PipelineStatsSnapshot {
            total_processed: self.total_processed.load(Ordering::Relaxed),
            total_bytes_processed: self.total_bytes_processed.load(Ordering::Relaxed),
            total_errors: self.total_errors.load(Ordering::Relaxed),
        }
    }
}

/// Non-atomic snapshot of [`PipelineStats`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PipelineStatsSnapshot {
    pub total_processed: u64,
    pub total_bytes_processed: u64,
    pub total_errors: u64,
}

// ---------------------------------------------------------------------------
// FNV-1a helper
// ---------------------------------------------------------------------------

/// Deterministically hash `data` into a fixed-length `f32` vector.
///
/// The output length equals `target_len`. Bytes are consumed in non-overlapping
/// groups whose size is `data.len() / target_len` (minimum 1). Within each
/// group the bytes are XOR-folded into a single byte, then divided by 255.0 to
/// produce a value in [0, 1].
pub fn fnv1a_hash_f32(data: &[u8], target_len: usize) -> Vec<f32> {
    if data.is_empty() || target_len == 0 {
        return vec![0.0_f32; target_len];
    }

    let mut result = Vec::with_capacity(target_len);

    // FNV-1a 64-bit constants
    const FNV_OFFSET_BASIS: u64 = 14_695_981_039_346_656_037;
    const FNV_PRIME: u64 = 1_099_511_628_211;

    for i in 0..target_len {
        // Seed the hash with the bucket index so each bucket is distinct.
        let mut hash: u64 = FNV_OFFSET_BASIS;
        hash ^= i as u64;
        hash = hash.wrapping_mul(FNV_PRIME);

        // Spread data bytes across buckets using modular indexing.
        // Every byte contributes to every bucket, weighted by position.
        let step = data.len().max(1);
        let start = (i * step) % data.len();
        for j in 0..step {
            let byte = data[(start + j) % data.len()];
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(FNV_PRIME);
        }

        // Map the 64-bit hash into [0, 1].
        let normalized = (hash % 256) as f32 / 255.0_f32;
        result.push(normalized);
    }

    result
}

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

/// Preprocessing pipeline that converts heterogeneous inputs into fixed-length
/// normalised float vectors suitable for HNSW indexing.
#[derive(Debug)]
pub struct EmbeddingPipeline {
    /// Configuration driving all pipeline behaviour.
    pub config: EmbeddingPipelineConfig,
    /// Operational statistics.
    pub stats: Arc<PipelineStats>,
}

impl EmbeddingPipeline {
    /// Create a new pipeline from the supplied configuration.
    pub fn new(config: EmbeddingPipelineConfig) -> Self {
        Self {
            config,
            stats: Arc::new(PipelineStats::default()),
        }
    }

    /// Create a pipeline with default settings (128-dim, L2 normalisation).
    pub fn with_defaults() -> Self {
        Self::new(EmbeddingPipelineConfig::default())
    }

    // ------------------------------------------------------------------
    // Public API
    // ------------------------------------------------------------------

    /// Process a single input and return the resulting embedding.
    pub fn process(&self, input: EmbeddingInput) -> Result<Vec<f32>, PipelineError> {
        let result = self.process_inner(input);
        match &result {
            Ok(_) => {
                self.stats.total_processed.fetch_add(1, Ordering::Relaxed);
            }
            Err(_) => {
                self.stats.total_errors.fetch_add(1, Ordering::Relaxed);
            }
        }
        result
    }

    /// Process a batch of inputs, returning one `Result` per input.
    pub fn batch_process(
        &self,
        inputs: Vec<EmbeddingInput>,
    ) -> Vec<Result<Vec<f32>, PipelineError>> {
        inputs.into_iter().map(|inp| self.process(inp)).collect()
    }

    // ------------------------------------------------------------------
    // Normalisation
    // ------------------------------------------------------------------

    /// Apply the configured normalisation strategy **in place**.
    pub fn normalize(&self, v: &mut [f32]) {
        match self.config.normalization {
            NormalizationStrategy::None => {}
            NormalizationStrategy::L2 => normalize_l2(v),
            NormalizationStrategy::MinMax => normalize_minmax(v),
            NormalizationStrategy::ZScore => normalize_zscore(v),
        }
    }

    // ------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------

    fn process_inner(&self, input: EmbeddingInput) -> Result<Vec<f32>, PipelineError> {
        let mut vec = match input {
            EmbeddingInput::RawBytes { data, .. } => {
                if data.is_empty() {
                    return Err(PipelineError::EmptyInput);
                }
                let byte_count = data.len() as u64;
                self.stats
                    .total_bytes_processed
                    .fetch_add(byte_count, Ordering::Relaxed);
                fnv1a_hash_f32(&data, self.config.dimensions)
            }

            EmbeddingInput::Text { content, .. } => {
                if content.is_empty() {
                    return Err(PipelineError::EmptyInput);
                }
                // Unicode code point → f32, normalised to [0, 1] by dividing by U+10FFFF.
                const MAX_CP: f32 = 0x10FFFF as f32;
                content
                    .chars()
                    .map(|c| c as u32 as f32 / MAX_CP)
                    .collect::<Vec<f32>>()
            }

            EmbeddingInput::Structured { fields } => {
                if fields.is_empty() {
                    return Err(PipelineError::EmptyInput);
                }
                // Sort by key for determinism, then concatenate as "key=value " pairs.
                let mut pairs: Vec<(&String, &String)> = fields.iter().collect();
                pairs.sort_by_key(|(k, _)| k.as_str());
                let combined: String = pairs.iter().map(|(k, v)| format!("{}={} ", k, v)).collect();
                // Reuse the Text path — recursion is safe because combined is non-empty.
                const MAX_CP: f32 = 0x10FFFF as f32;
                combined
                    .chars()
                    .map(|c| c as u32 as f32 / MAX_CP)
                    .collect::<Vec<f32>>()
            }

            EmbeddingInput::Embedding { vector } => {
                if vector.is_empty() {
                    return Err(PipelineError::EmptyInput);
                }
                vector
            }
        };

        // Validate (no NaN/inf before we try to normalise).
        validate_finite(&vec)?;

        // Truncate or pad.
        if self.config.truncate_to_dims {
            adjust_dimensions(&mut vec, self.config.dimensions, self.config.pad_value);
        } else if vec.len() != self.config.dimensions {
            return Err(PipelineError::DimensionMismatch {
                expected: self.config.dimensions,
                got: vec.len(),
            });
        }

        // Normalise.
        self.normalize(&mut vec);

        Ok(vec)
    }
}

// ---------------------------------------------------------------------------
// Private utility functions
// ---------------------------------------------------------------------------

fn adjust_dimensions(v: &mut Vec<f32>, target: usize, pad: f32) {
    match v.len().cmp(&target) {
        std::cmp::Ordering::Greater => v.truncate(target),
        std::cmp::Ordering::Less => v.resize(target, pad),
        std::cmp::Ordering::Equal => {}
    }
}

fn validate_finite(v: &[f32]) -> Result<(), PipelineError> {
    for (i, &val) in v.iter().enumerate() {
        if !val.is_finite() {
            return Err(PipelineError::InvalidVector(format!(
                "non-finite value at index {i}: {val}"
            )));
        }
    }
    Ok(())
}

fn normalize_l2(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm >= 1e-10 {
        let inv = 1.0 / norm;
        for x in v.iter_mut() {
            *x *= inv;
        }
    }
}

fn normalize_minmax(v: &mut [f32]) {
    let min = v.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let range = max - min;
    if range.abs() >= f32::EPSILON {
        for x in v.iter_mut() {
            *x = (*x - min) / range;
        }
    }
}

fn normalize_zscore(v: &mut [f32]) {
    let n = v.len() as f32;
    if n < 1.0 {
        return;
    }
    let mean = v.iter().sum::<f32>() / n;
    let variance = v.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
    let std_dev = variance.sqrt();
    if std_dev >= 1e-10 {
        for x in v.iter_mut() {
            *x = (*x - mean) / std_dev;
        }
    }
}

// ---------------------------------------------------------------------------
// SemanticEmbeddingPipeline — multi-stage pipeline for normalized embeddings
// ---------------------------------------------------------------------------

/// A single transformation stage in a [`SemanticEmbeddingPipeline`].
#[derive(Clone, Debug, PartialEq)]
pub enum PipelineStage {
    /// L2-normalize the embedding vector.
    Normalize,
    /// Multiply all components by a scalar factor.
    Scale { factor: f32 },
    /// Clamp each component to the range [min, max].
    Clamp { min: f32, max: f32 },
    /// Pad with 0.0 or truncate the vector to exactly `target_dim` elements.
    PadOrTruncate { target_dim: usize },
    /// Element-wise add `bias` to the vector.
    ///
    /// If `bias` is shorter than the vector the remaining elements get +0.0;
    /// if `bias` is longer it is silently truncated.
    AddBias { bias: Vec<f32> },
}

/// The result produced by [`SemanticEmbeddingPipeline::process`].
#[derive(Clone, Debug)]
pub struct PipelineResult {
    /// The transformed embedding vector.
    pub output: Vec<f32>,
    /// Number of pipeline stages that were applied.
    pub stages_applied: usize,
    /// Dimensionality of the input that was fed to the pipeline.
    pub input_dim: usize,
    /// Dimensionality of the output embedding.
    pub output_dim: usize,
}

impl PipelineResult {
    /// Returns `true` when the pipeline changed the vector's dimensionality.
    pub fn dimension_changed(&self) -> bool {
        self.input_dim != self.output_dim
    }
}

/// Aggregated statistics for a [`SemanticEmbeddingPipeline`].
#[derive(Clone, Debug, Default)]
pub struct SemanticPipelineStats {
    /// Total number of vectors processed via [`SemanticEmbeddingPipeline::process`].
    pub total_processed: u64,
    /// Total stage-application count across all processed vectors.
    pub total_stage_applications: u64,
    /// Running average of the output dimensionality.
    ///
    /// Returns `0.0` when no vectors have been processed yet.
    pub avg_output_dim: f64,
}

/// Multi-stage pipeline for transforming raw input vectors into
/// normalized embeddings.
///
/// Stages are applied in insertion order.  The pipeline tracks lightweight
/// statistics so callers can monitor throughput and dimensionality trends
/// without a separate monitoring side-channel.
///
/// # Example
/// ```
/// use ipfrs_semantic::embedding_pipeline::{SemanticEmbeddingPipeline, PipelineStage};
///
/// let mut pipeline = SemanticEmbeddingPipeline::new();
/// pipeline
///     .add_stage(PipelineStage::Clamp { min: -1.0, max: 1.0 })
///     .add_stage(PipelineStage::Normalize);
///
/// let result = pipeline.process(vec![3.0, -5.0, 2.0]);
/// assert_eq!(result.stages_applied, 2);
/// ```
#[derive(Debug)]
pub struct SemanticEmbeddingPipeline {
    /// Ordered list of stages that are applied to every input.
    pub stages: Vec<PipelineStage>,
    // Internal accumulators — not exposed directly; use `.stats()` instead.
    total_processed: u64,
    total_stage_applications: u64,
    total_output_dims: u64,
}

impl SemanticEmbeddingPipeline {
    /// Create a new pipeline with no stages.
    pub fn new() -> Self {
        Self {
            stages: Vec::new(),
            total_processed: 0,
            total_stage_applications: 0,
            total_output_dims: 0,
        }
    }

    /// Append a stage to the pipeline and return `&mut self` for chaining.
    pub fn add_stage(&mut self, stage: PipelineStage) -> &mut Self {
        self.stages.push(stage);
        self
    }

    /// Remove all stages from the pipeline.
    pub fn clear_stages(&mut self) {
        self.stages.clear();
    }

    /// Return the number of stages currently in the pipeline.
    pub fn stage_count(&self) -> usize {
        self.stages.len()
    }

    /// Apply every stage in order to `input` and return a [`PipelineResult`].
    pub fn process(&mut self, input: Vec<f32>) -> PipelineResult {
        let input_dim = input.len();
        let mut vec = input;

        for stage in &self.stages {
            Self::apply_stage(stage, &mut vec);
        }

        let output_dim = vec.len();
        let stages_applied = self.stages.len();

        self.total_processed += 1;
        self.total_stage_applications += stages_applied as u64;
        self.total_output_dims += output_dim as u64;

        PipelineResult {
            output: vec,
            stages_applied,
            input_dim,
            output_dim,
        }
    }

    /// Process a batch of input vectors, returning one [`PipelineResult`] each.
    pub fn process_batch(&mut self, inputs: Vec<Vec<f32>>) -> Vec<PipelineResult> {
        inputs.into_iter().map(|v| self.process(v)).collect()
    }

    /// Return a point-in-time snapshot of pipeline statistics.
    pub fn stats(&self) -> SemanticPipelineStats {
        let avg_output_dim = if self.total_processed == 0 {
            0.0
        } else {
            self.total_output_dims as f64 / self.total_processed as f64
        };
        SemanticPipelineStats {
            total_processed: self.total_processed,
            total_stage_applications: self.total_stage_applications,
            avg_output_dim,
        }
    }

    // ------------------------------------------------------------------
    // Private stage application
    // ------------------------------------------------------------------

    fn apply_stage(stage: &PipelineStage, vec: &mut Vec<f32>) {
        match stage {
            PipelineStage::Normalize => {
                let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm >= 1e-8 {
                    let inv = 1.0 / norm;
                    for x in vec.iter_mut() {
                        *x *= inv;
                    }
                }
            }

            PipelineStage::Scale { factor } => {
                for x in vec.iter_mut() {
                    *x *= factor;
                }
            }

            PipelineStage::Clamp { min, max } => {
                for x in vec.iter_mut() {
                    *x = x.clamp(*min, *max);
                }
            }

            PipelineStage::PadOrTruncate { target_dim } => {
                let current = vec.len();
                match current.cmp(target_dim) {
                    std::cmp::Ordering::Less => {
                        vec.resize(*target_dim, 0.0_f32);
                    }
                    std::cmp::Ordering::Greater => {
                        vec.truncate(*target_dim);
                    }
                    std::cmp::Ordering::Equal => {}
                }
            }

            PipelineStage::AddBias { bias } => {
                for (i, x) in vec.iter_mut().enumerate() {
                    let b = if i < bias.len() { bias[i] } else { 0.0 };
                    *x += b;
                }
            }
        }
    }
}

impl Default for SemanticEmbeddingPipeline {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_pipeline() -> EmbeddingPipeline {
        EmbeddingPipeline::with_defaults()
    }

    fn pipeline_with(
        dims: usize,
        norm: NormalizationStrategy,
        truncate: bool,
    ) -> EmbeddingPipeline {
        EmbeddingPipeline::new(EmbeddingPipelineConfig {
            dimensions: dims,
            normalization: norm,
            truncate_to_dims: truncate,
            pad_value: 0.0,
        })
    }

    // 1. RawBytes produces vector of correct length
    #[test]
    fn raw_bytes_correct_length() {
        let p = default_pipeline();
        let input = EmbeddingInput::RawBytes {
            data: vec![1u8, 2, 3, 4, 5, 6, 7, 8],
            mime_type: "application/octet-stream".to_string(),
        };
        let v = p.process(input).expect("should succeed");
        assert_eq!(v.len(), 128);
    }

    // 2. Text produces vector of correct length
    #[test]
    fn text_correct_length() {
        let p = default_pipeline();
        let input = EmbeddingInput::Text {
            content: "hello world".to_string(),
            language: None,
        };
        let v = p.process(input).expect("should succeed");
        assert_eq!(v.len(), 128);
    }

    // 3. Structured produces deterministic output
    #[test]
    fn structured_deterministic() {
        let p = default_pipeline();
        let mut fields = HashMap::new();
        fields.insert("name".to_string(), "alice".to_string());
        fields.insert("age".to_string(), "30".to_string());

        let v1 = p
            .process(EmbeddingInput::Structured {
                fields: fields.clone(),
            })
            .expect("first call");
        let v2 = p
            .process(EmbeddingInput::Structured { fields })
            .expect("second call");
        assert_eq!(v1, v2, "structured input must be deterministic");
    }

    // 4. Embedding pass-through (correct length preserved)
    #[test]
    fn embedding_passthrough() {
        let p = default_pipeline();
        let vec: Vec<f32> = (0..128).map(|i| i as f32 / 128.0).collect();
        let input = EmbeddingInput::Embedding {
            vector: vec.clone(),
        };
        let result = p.process(input).expect("should succeed");
        assert_eq!(result.len(), 128);
    }

    // 5. L2 normalization: result has unit norm
    #[test]
    fn l2_unit_norm() {
        let p = pipeline_with(16, NormalizationStrategy::L2, true);
        let input = EmbeddingInput::Embedding {
            vector: vec![1.0_f32; 16],
        };
        let v = p.process(input).expect("should succeed");
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-6,
            "L2 norm should be 1.0, got {norm}"
        );
    }

    // 6. MinMax normalization: result in [0, 1]
    #[test]
    fn minmax_in_range() {
        let p = pipeline_with(16, NormalizationStrategy::MinMax, true);
        let input = EmbeddingInput::Embedding {
            vector: (0..16).map(|i| i as f32 * 3.7 - 10.0).collect(),
        };
        let v = p.process(input).expect("should succeed");
        for &val in &v {
            assert!(
                (0.0..=1.0 + 1e-6).contains(&val),
                "minmax value out of range: {val}"
            );
        }
    }

    // 7. ZScore normalization: approximately zero mean
    #[test]
    fn zscore_zero_mean() {
        let p = pipeline_with(32, NormalizationStrategy::ZScore, true);
        let input = EmbeddingInput::Embedding {
            vector: (0..32).map(|i| i as f32).collect(),
        };
        let v = p.process(input).expect("should succeed");
        let mean: f32 = v.iter().sum::<f32>() / v.len() as f32;
        assert!(mean.abs() < 1e-5, "zscore mean should be ~0, got {mean}");
    }

    // 8. Truncation to configured dimensions
    #[test]
    fn truncation_to_dims() {
        let p = pipeline_with(8, NormalizationStrategy::None, true);
        let input = EmbeddingInput::Embedding {
            vector: vec![0.1_f32; 64],
        };
        let v = p.process(input).expect("should succeed");
        assert_eq!(v.len(), 8, "should be truncated to 8");
    }

    // 9. Padding to configured dimensions
    #[test]
    fn padding_to_dims() {
        let p = EmbeddingPipeline::new(EmbeddingPipelineConfig {
            dimensions: 32,
            normalization: NormalizationStrategy::None,
            truncate_to_dims: true,
            pad_value: -1.0,
        });
        let input = EmbeddingInput::Embedding {
            vector: vec![0.5_f32; 4],
        };
        let v = p.process(input).expect("should succeed");
        assert_eq!(v.len(), 32);
        // Padded portion should be -1.0 (before any normalisation — but None is used).
        for &val in &v[4..] {
            assert!((val - (-1.0)).abs() < 1e-7, "pad value mismatch: {val}");
        }
    }

    // 10. batch_process handles multiple inputs
    #[test]
    fn batch_process_multiple() {
        let p = default_pipeline();
        let inputs = vec![
            EmbeddingInput::Text {
                content: "alpha".to_string(),
                language: None,
            },
            EmbeddingInput::RawBytes {
                data: vec![42u8; 20],
                mime_type: "application/octet-stream".to_string(),
            },
            EmbeddingInput::Embedding {
                vector: vec![0.3_f32; 128],
            },
        ];
        let results = p.batch_process(inputs);
        assert_eq!(results.len(), 3);
        for r in &results {
            assert!(r.is_ok(), "batch result should be Ok: {r:?}");
        }
    }

    // 11. Stats accumulate
    #[test]
    fn stats_accumulate() {
        let p = default_pipeline();
        for _ in 0..5 {
            let _ = p.process(EmbeddingInput::Text {
                content: "test".to_string(),
                language: None,
            });
        }
        // Trigger one error
        let _ = p.process(EmbeddingInput::Text {
            content: String::new(),
            language: None,
        });
        let snap = p.stats.snapshot();
        assert_eq!(snap.total_processed, 5);
        assert_eq!(snap.total_errors, 1);
    }

    // 12. Empty Text input returns error
    #[test]
    fn empty_text_error() {
        let p = default_pipeline();
        let result = p.process(EmbeddingInput::Text {
            content: String::new(),
            language: None,
        });
        assert!(matches!(result, Err(PipelineError::EmptyInput)));
    }

    // 13. Empty RawBytes returns error
    #[test]
    fn empty_bytes_error() {
        let p = default_pipeline();
        let result = p.process(EmbeddingInput::RawBytes {
            data: vec![],
            mime_type: "application/octet-stream".to_string(),
        });
        assert!(matches!(result, Err(PipelineError::EmptyInput)));
    }

    // 14. DimensionMismatch when truncate_to_dims is false
    #[test]
    fn dimension_mismatch_error() {
        let p = pipeline_with(64, NormalizationStrategy::None, false);
        let result = p.process(EmbeddingInput::Embedding {
            vector: vec![1.0_f32; 32],
        });
        assert!(
            matches!(
                result,
                Err(PipelineError::DimensionMismatch {
                    expected: 64,
                    got: 32
                })
            ),
            "expected DimensionMismatch, got {result:?}"
        );
    }

    // 15. RawBytes stats counter increments correctly
    #[test]
    fn raw_bytes_stats_counter() {
        let p = default_pipeline();
        let _ = p.process(EmbeddingInput::RawBytes {
            data: vec![0xAA; 100],
            mime_type: "application/octet-stream".to_string(),
        });
        let snap = p.stats.snapshot();
        assert_eq!(snap.total_bytes_processed, 100);
    }

    // 16. None normalization preserves values (after pad/truncate)
    #[test]
    fn none_normalization_preserves_values() {
        let p = pipeline_with(4, NormalizationStrategy::None, true);
        let vals = vec![2.0_f32, 4.0, 6.0, 8.0];
        let input = EmbeddingInput::Embedding {
            vector: vals.clone(),
        };
        let v = p.process(input).expect("should succeed");
        for (a, b) in vals.iter().zip(v.iter()) {
            assert!((a - b).abs() < 1e-7);
        }
    }

    // -----------------------------------------------------------------------
    // SemanticEmbeddingPipeline tests
    // -----------------------------------------------------------------------

    // SEP-1: new() starts empty
    #[test]
    fn sep_new_starts_empty() {
        let p = SemanticEmbeddingPipeline::new();
        assert_eq!(p.stage_count(), 0);
        assert!(p.stages.is_empty());
    }

    // SEP-2: add_stage builder returns &mut Self (chain works)
    #[test]
    fn sep_add_stage_builder() {
        let mut p = SemanticEmbeddingPipeline::new();
        p.add_stage(PipelineStage::Normalize)
            .add_stage(PipelineStage::Scale { factor: 2.0 });
        assert_eq!(p.stage_count(), 2);
    }

    // SEP-3: stage_count correct
    #[test]
    fn sep_stage_count_correct() {
        let mut p = SemanticEmbeddingPipeline::new();
        assert_eq!(p.stage_count(), 0);
        p.add_stage(PipelineStage::Normalize);
        assert_eq!(p.stage_count(), 1);
        p.add_stage(PipelineStage::Scale { factor: 1.0 });
        assert_eq!(p.stage_count(), 2);
    }

    // SEP-4: clear_stages resets
    #[test]
    fn sep_clear_stages_resets() {
        let mut p = SemanticEmbeddingPipeline::new();
        p.add_stage(PipelineStage::Normalize)
            .add_stage(PipelineStage::Scale { factor: 2.0 });
        p.clear_stages();
        assert_eq!(p.stage_count(), 0);
    }

    // SEP-5: Normalize produces unit vector
    #[test]
    fn sep_normalize_unit_vector() {
        let mut p = SemanticEmbeddingPipeline::new();
        p.add_stage(PipelineStage::Normalize);
        let result = p.process(vec![3.0, 4.0]);
        let norm: f32 = result.output.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6, "expected unit norm, got {norm}");
    }

    // SEP-6: Normalize zero vector unchanged
    #[test]
    fn sep_normalize_zero_vector_unchanged() {
        let mut p = SemanticEmbeddingPipeline::new();
        p.add_stage(PipelineStage::Normalize);
        let result = p.process(vec![0.0, 0.0, 0.0]);
        assert_eq!(result.output, vec![0.0_f32, 0.0, 0.0]);
    }

    // SEP-7: Scale multiplies correctly
    #[test]
    fn sep_scale_multiplies_correctly() {
        let mut p = SemanticEmbeddingPipeline::new();
        p.add_stage(PipelineStage::Scale { factor: 3.0 });
        let result = p.process(vec![1.0, 2.0, 3.0]);
        assert_eq!(result.output, vec![3.0_f32, 6.0, 9.0]);
    }

    // SEP-8: Clamp restricts values
    #[test]
    fn sep_clamp_restricts_values() {
        let mut p = SemanticEmbeddingPipeline::new();
        p.add_stage(PipelineStage::Clamp {
            min: -1.0,
            max: 1.0,
        });
        let result = p.process(vec![-5.0, 0.5, 3.0]);
        assert_eq!(result.output, vec![-1.0_f32, 0.5, 1.0]);
    }

    // SEP-9: PadOrTruncate pads shorter vector
    #[test]
    fn sep_pad_or_truncate_pads_shorter() {
        let mut p = SemanticEmbeddingPipeline::new();
        p.add_stage(PipelineStage::PadOrTruncate { target_dim: 6 });
        let result = p.process(vec![1.0, 2.0, 3.0]);
        assert_eq!(result.output.len(), 6);
        assert_eq!(&result.output[3..], &[0.0_f32, 0.0, 0.0]);
    }

    // SEP-10: PadOrTruncate truncates longer vector
    #[test]
    fn sep_pad_or_truncate_truncates_longer() {
        let mut p = SemanticEmbeddingPipeline::new();
        p.add_stage(PipelineStage::PadOrTruncate { target_dim: 2 });
        let result = p.process(vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(result.output, vec![1.0_f32, 2.0]);
    }

    // SEP-11: PadOrTruncate exact length unchanged
    #[test]
    fn sep_pad_or_truncate_exact_unchanged() {
        let mut p = SemanticEmbeddingPipeline::new();
        p.add_stage(PipelineStage::PadOrTruncate { target_dim: 3 });
        let result = p.process(vec![1.0, 2.0, 3.0]);
        assert_eq!(result.output, vec![1.0_f32, 2.0, 3.0]);
    }

    // SEP-12: AddBias adds element-wise
    #[test]
    fn sep_add_bias_element_wise() {
        let mut p = SemanticEmbeddingPipeline::new();
        p.add_stage(PipelineStage::AddBias {
            bias: vec![0.1, 0.2, 0.3],
        });
        let result = p.process(vec![1.0, 1.0, 1.0]);
        let expected = [1.1_f32, 1.2, 1.3];
        for (a, b) in result.output.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6, "got {a}, expected {b}");
        }
    }

    // SEP-13: AddBias handles bias shorter than vec
    #[test]
    fn sep_add_bias_shorter_bias() {
        let mut p = SemanticEmbeddingPipeline::new();
        p.add_stage(PipelineStage::AddBias {
            bias: vec![1.0], // shorter
        });
        let result = p.process(vec![0.0, 0.0, 0.0]);
        assert!((result.output[0] - 1.0).abs() < 1e-6);
        assert!((result.output[1] - 0.0).abs() < 1e-6);
        assert!((result.output[2] - 0.0).abs() < 1e-6);
    }

    // SEP-14: AddBias handles bias longer than vec (truncated)
    #[test]
    fn sep_add_bias_longer_bias_truncated() {
        let mut p = SemanticEmbeddingPipeline::new();
        p.add_stage(PipelineStage::AddBias {
            bias: vec![1.0, 2.0, 3.0, 4.0, 5.0], // longer
        });
        let result = p.process(vec![0.0, 0.0]);
        // Only first two bias values applied
        assert_eq!(result.output.len(), 2);
        assert!((result.output[0] - 1.0).abs() < 1e-6);
        assert!((result.output[1] - 2.0).abs() < 1e-6);
    }

    // SEP-15: Multi-stage pipeline applies in order
    #[test]
    fn sep_multi_stage_order() {
        let mut p = SemanticEmbeddingPipeline::new();
        // Scale by 2, then clamp to [0, 3]
        p.add_stage(PipelineStage::Scale { factor: 2.0 })
            .add_stage(PipelineStage::Clamp { min: 0.0, max: 3.0 });
        let result = p.process(vec![-1.0, 1.0, 2.0]);
        // After Scale: [-2.0, 2.0, 4.0]
        // After Clamp: [0.0, 2.0, 3.0]
        assert_eq!(result.output, vec![0.0_f32, 2.0, 3.0]);
    }

    // SEP-16: PipelineResult stages_applied correct
    #[test]
    fn sep_result_stages_applied_correct() {
        let mut p = SemanticEmbeddingPipeline::new();
        p.add_stage(PipelineStage::Normalize)
            .add_stage(PipelineStage::Scale { factor: 1.0 })
            .add_stage(PipelineStage::Clamp {
                min: -1.0,
                max: 1.0,
            });
        let result = p.process(vec![1.0, 0.0]);
        assert_eq!(result.stages_applied, 3);
    }

    // SEP-17: PipelineResult input_dim / output_dim correct (no dim change)
    #[test]
    fn sep_result_dims_no_change() {
        let mut p = SemanticEmbeddingPipeline::new();
        p.add_stage(PipelineStage::Normalize);
        let result = p.process(vec![1.0, 2.0, 3.0]);
        assert_eq!(result.input_dim, 3);
        assert_eq!(result.output_dim, 3);
    }

    // SEP-18: dimension_changed true when PadOrTruncate changes size
    #[test]
    fn sep_dimension_changed_true() {
        let mut p = SemanticEmbeddingPipeline::new();
        p.add_stage(PipelineStage::PadOrTruncate { target_dim: 8 });
        let result = p.process(vec![1.0, 2.0]);
        assert!(result.dimension_changed());
        assert_eq!(result.input_dim, 2);
        assert_eq!(result.output_dim, 8);
    }

    // SEP-19: dimension_changed false when same size
    #[test]
    fn sep_dimension_changed_false() {
        let mut p = SemanticEmbeddingPipeline::new();
        p.add_stage(PipelineStage::Normalize);
        let result = p.process(vec![1.0, 0.0, 0.0]);
        assert!(!result.dimension_changed());
    }

    // SEP-20: process_batch returns correct count
    #[test]
    fn sep_process_batch_count() {
        let mut p = SemanticEmbeddingPipeline::new();
        p.add_stage(PipelineStage::Normalize);
        let inputs = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![2.0, 3.0],
        ];
        let results = p.process_batch(inputs);
        assert_eq!(results.len(), 4);
    }

    // SEP-21: stats total_processed increments
    #[test]
    fn sep_stats_total_processed_increments() {
        let mut p = SemanticEmbeddingPipeline::new();
        p.add_stage(PipelineStage::Normalize);
        for _ in 0..7 {
            p.process(vec![1.0, 2.0]);
        }
        let s = p.stats();
        assert_eq!(s.total_processed, 7);
        assert_eq!(s.total_stage_applications, 7);
    }

    // SEP-22: stats avg_output_dim computed correctly
    #[test]
    fn sep_stats_avg_output_dim_correct() {
        let mut p = SemanticEmbeddingPipeline::new();
        // First: PadOrTruncate to 4 => output_dim = 4
        // Second: PadOrTruncate to 4 => output_dim = 4
        p.add_stage(PipelineStage::PadOrTruncate { target_dim: 4 });
        p.process(vec![1.0, 2.0]); // output_dim = 4
        p.process(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]); // output_dim = 4
        let s = p.stats();
        assert_eq!(s.total_processed, 2);
        assert!((s.avg_output_dim - 4.0).abs() < 1e-9);
    }

    // SEP-23: stats avg_output_dim is 0.0 when nothing processed
    #[test]
    fn sep_stats_avg_output_dim_zero_when_empty() {
        let p = SemanticEmbeddingPipeline::new();
        let s = p.stats();
        assert_eq!(s.total_processed, 0);
        assert_eq!(s.avg_output_dim, 0.0);
    }
}
