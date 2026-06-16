//! Type definitions for the document summarizer.
//!
//! This module holds all public error types, enums, structs, and their
//! implementations that are shared across the summarizer implementation.

// ── Error ─────────────────────────────────────────────────────────────────────

/// Errors returned by [`crate::document_summarizer::DocumentSummarizer`].
#[derive(Debug, Clone, PartialEq)]
pub enum SummarizerError {
    /// The input document was empty or contained only whitespace.
    EmptyDocument,
    /// Fewer sentences were found than the algorithm requires.
    InsufficientSentences {
        /// Minimum number of sentences needed.
        needed: usize,
        /// Actual number of sentences found.
        got: usize,
    },
    /// Provided sentence embeddings had a different dimension from each other.
    EmbeddingDimensionMismatch {
        /// Expected embedding dimension.
        expected: usize,
        /// Actual dimension encountered.
        got: usize,
    },
    /// A configuration parameter was invalid.
    ConfigurationError(String),
}

impl std::fmt::Display for SummarizerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyDocument => write!(f, "document is empty"),
            Self::InsufficientSentences { needed, got } => {
                write!(f, "need at least {needed} sentences, got {got}")
            }
            Self::EmbeddingDimensionMismatch { expected, got } => {
                write!(
                    f,
                    "embedding dimension mismatch: expected {expected}, got {got}"
                )
            }
            Self::ConfigurationError(msg) => write!(f, "configuration error: {msg}"),
        }
    }
}

impl std::error::Error for SummarizerError {}

// ── SummaryStyle ──────────────────────────────────────────────────────────────

/// Strategy used to produce a summary.
#[derive(Debug, Clone, PartialEq)]
pub enum SummaryStyle {
    /// Return the top-`num_sentences` sentences in original document order.
    Extractive {
        /// Number of sentences to include.
        num_sentences: usize,
    },
    /// Return the top-`num_phrases` scored n-gram keyphrases.
    Keyphrase {
        /// Number of phrases to return.
        num_phrases: usize,
    },
    /// Return the single most important sentence, truncated to `max_chars`.
    Headline {
        /// Maximum character length of the headline.
        max_chars: usize,
    },
    /// Concatenate the top-3 sentences (transition words stripped) trimmed to
    /// approximately `target_words` words.
    Abstractive {
        /// Approximate word budget for the output.
        target_words: usize,
    },
    /// Cluster sentences into `levels` groups and return one sentence per cluster.
    Hierarchical {
        /// Number of clusters / output sentences.
        levels: usize,
    },
}

// ── Core types ────────────────────────────────────────────────────────────────

/// Per-sentence feature scores computed during summarization.
#[derive(Debug, Clone)]
pub struct SentenceScore {
    /// Raw text of the sentence.
    pub sentence: String,
    /// Zero-based position in the source document.
    pub index: usize,
    /// TF-IDF based importance score.
    pub tf_idf_score: f64,
    /// Score derived from sentence position (higher at start/end).
    pub position_score: f64,
    /// Score penalising very short or very long sentences.
    pub length_score: f64,
    /// Mean cosine similarity to all other sentence embeddings (0.0 when no embeddings).
    pub embedding_centrality: f64,
    /// Weighted combination of the four signals.
    pub final_score: f64,
}

/// The output produced by a single summarization call.
#[derive(Debug, Clone)]
pub struct SummaryResult {
    /// Number of characters in the original document.
    pub original_length: usize,
    /// Number of characters in the summary.
    pub summary_length: usize,
    /// `summary_length / original_length`; 0.0 when original is empty.
    pub compression_ratio: f64,
    /// Selected / generated sentences forming the summary.
    pub sentences: Vec<String>,
    /// Top keyphrases extracted from the document.
    pub keyphrases: Vec<String>,
    /// Summarization strategy used.
    pub style: SummaryStyle,
    /// Fraction of original keyphrases that appear in the summary; in \[0, 1\].
    pub quality_score: f64,
}

/// A text chunk with optional embedding and metadata.
#[derive(Debug, Clone)]
pub struct DocumentChunk {
    /// Text content of the chunk.
    pub text: String,
    /// Optional dense embedding for this chunk.
    pub embedding: Option<Vec<f64>>,
    /// Optional heading of the section this chunk belongs to.
    pub section_title: Option<String>,
    /// Zero-based index of this chunk in the sequence.
    pub chunk_index: usize,
}

/// Configuration for [`crate::document_summarizer::DocumentSummarizer`].
#[derive(Debug, Clone)]
pub struct SummarizerConfig {
    /// Target summarization style.
    pub style: SummaryStyle,
    /// Sentences shorter than this (in characters) are excluded.
    pub min_sentence_length: usize,
    /// Sentences longer than this (in characters) are excluded from scoring.
    pub max_sentence_length: usize,
    /// Words to ignore when computing TF-IDF and keyphrases.
    pub stop_words: Vec<String>,
    /// When `true`, use embedding centrality as a scoring signal.
    pub use_embeddings: bool,
    /// How strongly to favour sentences at the beginning or end of the document.
    /// A value of 0.0 disables position bias; 1.0 gives maximum weight.
    pub position_bias: f64,
}

impl Default for SummarizerConfig {
    fn default() -> Self {
        Self {
            style: SummaryStyle::Extractive { num_sentences: 3 },
            min_sentence_length: 20,
            max_sentence_length: 1000,
            stop_words: super::default_stop_words(),
            use_embeddings: false,
            position_bias: 0.5,
        }
    }
}

/// Cumulative statistics for a [`crate::document_summarizer::DocumentSummarizer`] instance.
#[derive(Debug, Clone, Default)]
pub struct SummarizerStats {
    /// Total number of documents summarized.
    pub documents_processed: u64,
    /// Running average compression ratio across all documents.
    pub avg_compression_ratio: f64,
    /// Running average quality score across all documents.
    pub avg_quality_score: f64,
    /// Approximate total number of tokens (whitespace-split words) processed.
    pub total_tokens_processed: u64,
}
