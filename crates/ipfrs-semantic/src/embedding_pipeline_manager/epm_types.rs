//! Public type definitions for the Embedding Pipeline Manager.

use thiserror::Error;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors produced by [`super::EmbeddingPipelineManager`].
///
/// The `Epm` prefix disambiguates from `embedding_pipeline::PipelineError`.
#[derive(Debug, Error, Clone)]
pub enum EpmPipelineError {
    /// Configuration is invalid.
    #[error("invalid config: {0}")]
    InvalidConfig(String),

    /// Dimension mismatch between consecutive stages.
    #[error("dimension error: expected {expected}, got {got}")]
    DimensionError { expected: usize, got: usize },

    /// Input batch was empty.
    #[error("empty input")]
    EmptyInput,

    /// Generic processing failure.
    #[error("processing failed: {0}")]
    ProcessingFailed(String),

    /// A specific stage produced an error.
    #[error("stage '{stage}' failed: {reason}")]
    StageError { stage: String, reason: String },
}

// ---------------------------------------------------------------------------
// ReductionMethod
// ---------------------------------------------------------------------------

/// Algorithm used by [`EpmPipelineStage::DimensionReduce`].
///
/// The `Epm` prefix disambiguates from `dimension_reducer::ReductionMethod`.
#[derive(Debug, Clone, PartialEq)]
pub enum EpmReductionMethod {
    /// Simplified PCA via per-dimension mean centering and variance scaling.
    PCA,
    /// Deterministic random Gaussian projection seeded with `seed`.
    RandomProjection(u64),
    /// Keep only the first `target_dim` components.
    TruncateDims,
    /// Pool all components into `target_dim` by local mean-pooling.
    MeanPooling,
}

// ---------------------------------------------------------------------------
// PipelineStage
// ---------------------------------------------------------------------------

/// A single processing stage in an [`super::EmbeddingPipelineManager`].
///
/// The `Epm` prefix disambiguates from `embedding_pipeline::PipelineStage`.
#[derive(Debug, Clone)]
pub enum EpmPipelineStage {
    /// Tokenise text with optional case folding and punctuation stripping.
    Tokenize { lowercase: bool, strip_punct: bool },
    /// Remove tokens that appear in the stop-word list.
    StopWordFilter(Vec<String>),
    /// Replace individual tokens with n-gram strings (`"a_b"` for bigrams etc.).
    NGram { n: usize },
    /// Compute TF-IDF weighted embedding over the vocabulary induced from the batch.
    TfIdfWeighting,
    /// L2-normalise the embedding vector.
    L2Normalize,
    /// Reduce dimensionality.
    DimensionReduce {
        target_dim: usize,
        method: EpmReductionMethod,
    },
    /// Scale each component to `[0, 255]` and round to an integer, stored as `f64`.
    QuantizeToByte,
    /// Add sinusoidal positional encoding scaled to the embedding length.
    AddPositionalEncoding { max_len: usize },
}

impl EpmPipelineStage {
    /// Human-readable name used in metrics.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Tokenize { .. } => "Tokenize",
            Self::StopWordFilter(_) => "StopWordFilter",
            Self::NGram { .. } => "NGram",
            Self::TfIdfWeighting => "TfIdfWeighting",
            Self::L2Normalize => "L2Normalize",
            Self::DimensionReduce { .. } => "DimensionReduce",
            Self::QuantizeToByte => "QuantizeToByte",
            Self::AddPositionalEncoding { .. } => "AddPositionalEncoding",
        }
    }

    /// Returns `true` if the stage requires tokenised text input rather than a
    /// numeric embedding.
    pub(super) fn requires_tokens(&self) -> bool {
        matches!(
            self,
            Self::Tokenize { .. }
                | Self::StopWordFilter(_)
                | Self::NGram { .. }
                | Self::TfIdfWeighting
        )
    }
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Full configuration for an [`super::EmbeddingPipelineManager`].
///
/// The `Epm` prefix disambiguates from `query_pipeline::PipelineConfig`.
#[derive(Debug, Clone)]
pub struct EpmPipelineConfig {
    /// Ordered pipeline stages.
    pub stages: Vec<EpmPipelineStage>,
    /// Expected output dimensionality (used for validation).
    pub output_dim: usize,
    /// Number of inputs to process together in one call.
    pub batch_size: usize,
}

impl EpmPipelineConfig {
    /// Construct a minimal config with the given output dimension.
    pub fn new(output_dim: usize, batch_size: usize) -> Self {
        Self {
            stages: Vec::new(),
            output_dim,
            batch_size,
        }
    }
}

// ---------------------------------------------------------------------------
// EmbeddingBatch
// ---------------------------------------------------------------------------

/// The result of running one batch through the pipeline.
#[derive(Debug, Clone)]
pub struct EmbeddingBatch {
    /// Caller-supplied opaque identifiers, one per input.
    pub ids: Vec<String>,
    /// Original text strings (present only for text-input batches).
    pub texts: Option<Vec<String>>,
    /// Pre-computed embeddings as supplied by the caller (may be `None` for text input).
    pub raw_embeddings: Option<Vec<Vec<f64>>>,
    /// Final embeddings after all pipeline stages.
    pub output_embeddings: Vec<Vec<f64>>,
    /// Wall-clock time in microseconds for the entire batch.
    pub processing_time_us: u64,
}

// ---------------------------------------------------------------------------
// StageTiming / PipelineStats
// ---------------------------------------------------------------------------

/// Aggregate timing for a single pipeline stage.
#[derive(Debug, Clone)]
pub struct StageTiming {
    /// Stage name (from [`EpmPipelineStage::name`]).
    pub stage_name: String,
    /// Average microseconds per stage invocation.
    pub avg_time_us: f64,
    /// Total number of individual inputs processed by this stage.
    pub total_processed: u64,
}

/// Cumulative statistics for an [`super::EmbeddingPipelineManager`] instance.
///
/// The `Epm` prefix disambiguates from `embedding_pipeline::PipelineStats` and
/// `query_pipeline::PipelineStats`.
#[derive(Debug, Clone, Default)]
pub struct EpmPipelineStats {
    /// Number of batches processed.
    pub batches_processed: u64,
    /// Total individual inputs processed across all batches.
    pub total_inputs: u64,
    /// Running average batch wall-clock time in microseconds.
    pub avg_batch_time_us: f64,
    /// Per-stage timing details.
    pub stage_timings: Vec<StageTiming>,
    /// Current output dimensionality of the pipeline.
    pub output_dim: usize,
}
