//! # Document Chunker
//!
//! A production-grade document chunking engine that splits text into semantically
//! coherent chunks for embedding and retrieval. Supports multiple chunking strategies
//! including fixed-size windows, sentence boundaries, paragraph-based, and semantic
//! grouping.
//!
//! ## Example
//!
//! ```rust
//! use ipfrs_semantic::{DocumentChunker, DocumentChunkerConfig, ChunkStrategy};
//!
//! let config = DocumentChunkerConfig {
//!     strategy: ChunkStrategy::SentenceBoundary {
//!         max_chunk_chars: 512,
//!         overlap_sentences: 1,
//!     },
//!     preserve_whitespace: false,
//!     min_chunk_chars: 10,
//! };
//!
//! let mut chunker = DocumentChunker::new(config);
//! let chunks = chunker.chunk_text("doc-001", "Hello world. This is a test.");
//! assert!(!chunks.is_empty());
//! ```

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Strategy that controls how a document is split into chunks.
#[derive(Debug, Clone, PartialEq)]
pub enum ChunkStrategy {
    /// Split by character count with configurable overlap.
    FixedSize {
        /// Window size in characters.
        size: usize,
        /// Number of characters to overlap between consecutive chunks.
        overlap: usize,
    },
    /// Split at sentence boundaries (`.`, `!`, `?` followed by whitespace).
    SentenceBoundary {
        /// Maximum chunk length in characters.
        max_chunk_chars: usize,
        /// Number of sentences from the end of the previous chunk that are
        /// prepended to the start of the next chunk as overlap.
        overlap_sentences: usize,
    },
    /// Split at double-newline paragraph boundaries.
    Paragraph {
        /// Maximum chunk length in characters; oversized paragraphs are
        /// further split at sentence boundaries.
        max_chunk_chars: usize,
    },
    /// Group sentences whose cumulative length stays within `max_chunk_chars`.
    /// `similarity_threshold` is reserved for future embedding-based splitting.
    Semantic {
        /// Maximum chunk length in characters.
        max_chunk_chars: usize,
        /// Reserved — similarity threshold for future embedding-based grouping.
        similarity_threshold: f64,
    },
}

/// A single text chunk produced by [`DocumentChunker`].
#[derive(Debug, Clone, PartialEq)]
pub struct TextChunk {
    /// Unique chunk identifier in the form `{doc_id}-{index}`.
    pub id: String,
    /// The text content of this chunk.
    pub content: String,
    /// Byte offset of the first character in the original document.
    pub start_offset: usize,
    /// Byte offset one past the last character in the original document.
    pub end_offset: usize,
    /// Zero-based position of this chunk in the output sequence.
    pub chunk_index: usize,
    /// Arbitrary metadata key-value pairs attached to this chunk.
    pub metadata: HashMap<String, String>,
}

/// Aggregate statistics over a collection of [`TextChunk`]s.
#[derive(Debug, Clone, PartialEq)]
pub struct ChunkStats {
    /// Total number of chunks.
    pub total_chunks: usize,
    /// Total characters across all chunks.
    pub total_chars: usize,
    /// Mean chunk length in characters.
    pub avg_chunk_chars: f64,
    /// Minimum chunk length.
    pub min_chunk_chars: usize,
    /// Maximum chunk length.
    pub max_chunk_chars: usize,
    /// Estimated number of overlapping characters (only meaningful for
    /// `FixedSize` and `SentenceBoundary` strategies).
    pub overlap_chars: usize,
}

/// Configuration for [`DocumentChunker`].
#[derive(Debug, Clone, PartialEq)]
pub struct DocumentChunkerConfig {
    /// The chunking strategy to apply.
    pub strategy: ChunkStrategy,
    /// When `true`, leading/trailing whitespace is preserved in each chunk;
    /// when `false` it is stripped.
    pub preserve_whitespace: bool,
    /// Chunks whose content is shorter than this threshold are discarded
    /// (after whitespace trimming if applicable).
    pub min_chunk_chars: usize,
}

impl Default for DocumentChunkerConfig {
    fn default() -> Self {
        Self {
            strategy: ChunkStrategy::SentenceBoundary {
                max_chunk_chars: 512,
                overlap_sentences: 1,
            },
            preserve_whitespace: false,
            min_chunk_chars: 10,
        }
    }
}

/// Document chunking engine.
///
/// Maintains running counters for chunks produced and documents processed
/// across the lifetime of the instance. All heavy work is done in the
/// `chunk_*` family of methods.
#[derive(Debug, Clone)]
pub struct DocumentChunker {
    /// Active configuration.
    pub config: DocumentChunkerConfig,
    /// Total number of [`TextChunk`]s emitted since construction.
    pub chunks_produced: u64,
    /// Total number of documents submitted to [`DocumentChunker::chunk_text`].
    pub documents_processed: u64,
}

// ---------------------------------------------------------------------------
// Implementation
// ---------------------------------------------------------------------------

impl DocumentChunker {
    /// Construct a new `DocumentChunker` with the supplied configuration.
    pub fn new(config: DocumentChunkerConfig) -> Self {
        Self {
            config,
            chunks_produced: 0,
            documents_processed: 0,
        }
    }

    /// Chunk `text` using the strategy stored in `self.config`, filter chunks
    /// shorter than `min_chunk_chars`, and update internal counters.
    pub fn chunk_text(&mut self, doc_id: &str, text: &str) -> Vec<TextChunk> {
        self.documents_processed += 1;

        let min = self.config.min_chunk_chars;
        let preserve = self.config.preserve_whitespace;

        let mut chunks = match &self.config.strategy.clone() {
            ChunkStrategy::FixedSize { size, overlap } => {
                self.chunk_fixed_size(doc_id, text, *size, *overlap)
            }
            ChunkStrategy::SentenceBoundary {
                max_chunk_chars,
                overlap_sentences,
            } => self.chunk_sentence_boundary(doc_id, text, *max_chunk_chars, *overlap_sentences),
            ChunkStrategy::Paragraph { max_chunk_chars } => {
                self.chunk_paragraph(doc_id, text, *max_chunk_chars)
            }
            ChunkStrategy::Semantic {
                max_chunk_chars,
                similarity_threshold,
            } => self.chunk_semantic(doc_id, text, *max_chunk_chars, *similarity_threshold),
        };

        // Optionally strip whitespace and filter undersized chunks.
        if !preserve {
            for chunk in &mut chunks {
                let trimmed = chunk.content.trim().to_string();
                chunk.content = trimmed;
            }
        }
        chunks.retain(|c| c.content.len() >= min);

        // Re-number chunk indices after filtering.
        for (i, chunk) in chunks.iter_mut().enumerate() {
            chunk.chunk_index = i;
            chunk.id = format!("{}-{}", doc_id, i);
        }

        self.chunks_produced += chunks.len() as u64;
        chunks
    }

    /// Split `text` into fixed-size windows of `size` characters stepping by
    /// `size - overlap` characters between each window.
    ///
    /// If `overlap >= size` it is clamped to `size.saturating_sub(1)` to
    /// guarantee progress.
    pub fn chunk_fixed_size(
        &self,
        doc_id: &str,
        text: &str,
        size: usize,
        overlap: usize,
    ) -> Vec<TextChunk> {
        if size == 0 || text.is_empty() {
            return Vec::new();
        }

        let overlap = overlap.min(size.saturating_sub(1));
        let step = size - overlap;

        // Work on the character sequence to avoid splitting UTF-8 scalars.
        let chars: Vec<char> = text.chars().collect();
        let total = chars.len();

        let mut chunks = Vec::new();
        let mut start = 0usize;
        let mut index = 0usize;

        while start < total {
            let end = (start + size).min(total);
            let content: String = chars[start..end].iter().collect();

            // Map character indices back to byte offsets.
            let byte_start = char_idx_to_byte(text, start);
            let byte_end = char_idx_to_byte(text, end);

            chunks.push(TextChunk {
                id: format!("{}-{}", doc_id, index),
                content,
                start_offset: byte_start,
                end_offset: byte_end,
                chunk_index: index,
                metadata: HashMap::new(),
            });

            index += 1;
            start += step;
        }

        chunks
    }

    /// Split `text` at sentence boundaries and group sentences so that each
    /// chunk stays within `max_chars` characters. The last `overlap_sentences`
    /// sentences of each chunk are prepended to the next chunk.
    pub fn chunk_sentence_boundary(
        &self,
        doc_id: &str,
        text: &str,
        max_chars: usize,
        overlap_sentences: usize,
    ) -> Vec<TextChunk> {
        let sentences = split_into_sentences(text);
        if sentences.is_empty() {
            return Vec::new();
        }
        group_sentences_into_chunks(doc_id, text, &sentences, max_chars, overlap_sentences)
    }

    /// Split `text` at `\n\n` paragraph boundaries. Paragraphs that exceed
    /// `max_chars` are further split at sentence boundaries.
    pub fn chunk_paragraph(&self, doc_id: &str, text: &str, max_chars: usize) -> Vec<TextChunk> {
        let paragraphs: Vec<&str> = text.split("\n\n").collect();

        let mut chunks = Vec::new();
        let mut global_byte_offset = 0usize;
        let mut index = 0usize;

        for para in &paragraphs {
            let para_len = para.len();

            if para.len() <= max_chars {
                // Emit as a single chunk, but only if non-empty.
                if !para.trim().is_empty() {
                    chunks.push(TextChunk {
                        id: format!("{}-{}", doc_id, index),
                        content: para.to_string(),
                        start_offset: global_byte_offset,
                        end_offset: global_byte_offset + para_len,
                        chunk_index: index,
                        metadata: HashMap::new(),
                    });
                    index += 1;
                }
            } else {
                // Oversized paragraph — sub-split at sentence boundaries.
                let sub = group_sentences_into_chunks(
                    doc_id,
                    para,
                    &split_into_sentences(para),
                    max_chars,
                    0,
                );
                for mut sc in sub {
                    sc.start_offset += global_byte_offset;
                    sc.end_offset += global_byte_offset;
                    sc.chunk_index = index;
                    sc.id = format!("{}-{}", doc_id, index);
                    chunks.push(sc);
                    index += 1;
                }
            }

            // Advance past the paragraph text plus the "\n\n" separator.
            global_byte_offset += para_len + 2;
        }

        chunks
    }

    /// Group sentences into chunks of at most `max_chars` characters.
    /// `_threshold` is a placeholder for future embedding-based similarity.
    pub fn chunk_semantic(
        &self,
        doc_id: &str,
        text: &str,
        max_chars: usize,
        _threshold: f64,
    ) -> Vec<TextChunk> {
        // For now the implementation is identical to sentence_boundary with
        // overlap = 0. When embedding support lands the similarity_threshold
        // will gate merging adjacent sentences.
        let sentences = split_into_sentences(text);
        group_sentences_into_chunks(doc_id, text, &sentences, max_chars, 0)
    }

    /// Merge chunks shorter than `min_chars` into the following chunk.
    /// The last chunk absorbs any trailing short chunks. Offsets and indices
    /// are updated in place.
    pub fn merge_small_chunks(chunks: Vec<TextChunk>, min_chars: usize) -> Vec<TextChunk> {
        if chunks.is_empty() {
            return Vec::new();
        }

        let mut result: Vec<TextChunk> = Vec::new();
        let mut pending: Option<TextChunk> = None;

        for chunk in chunks {
            match pending.take() {
                None => {
                    if chunk.content.len() < min_chars {
                        pending = Some(chunk);
                    } else {
                        result.push(chunk);
                    }
                }
                Some(mut prev) => {
                    // Merge `prev` (the short chunk) into `chunk`.
                    prev.content.push(' ');
                    prev.content.push_str(&chunk.content);
                    prev.end_offset = chunk.end_offset;

                    if prev.content.len() < min_chars {
                        // Still short — keep accumulating.
                        pending = Some(prev);
                    } else {
                        result.push(prev);
                    }
                }
            }
        }

        // Flush any remaining pending chunk (even if still short).
        if let Some(last) = pending {
            // Try to merge into the last emitted chunk.
            if let Some(prev) = result.last_mut() {
                prev.content.push(' ');
                prev.content.push_str(&last.content);
                prev.end_offset = last.end_offset;
            } else {
                result.push(last);
            }
        }

        // Re-number indices.
        for (i, chunk) in result.iter_mut().enumerate() {
            chunk.chunk_index = i;
        }

        result
    }

    /// Compute aggregate statistics over a slice of chunks.
    pub fn stats(chunks: &[TextChunk]) -> ChunkStats {
        if chunks.is_empty() {
            return ChunkStats {
                total_chunks: 0,
                total_chars: 0,
                avg_chunk_chars: 0.0,
                min_chunk_chars: 0,
                max_chunk_chars: 0,
                overlap_chars: 0,
            };
        }

        let total_chars: usize = chunks.iter().map(|c| c.content.len()).sum();
        let min_chunk_chars = chunks.iter().map(|c| c.content.len()).min().unwrap_or(0);
        let max_chunk_chars = chunks.iter().map(|c| c.content.len()).max().unwrap_or(0);

        // Estimate overlap as the sum of overlapping byte ranges between
        // consecutive chunks.
        let overlap_chars: usize = chunks
            .windows(2)
            .map(|w| {
                let prev_end = w[0].end_offset;
                let next_start = w[1].start_offset;
                prev_end.saturating_sub(next_start)
            })
            .sum();

        ChunkStats {
            total_chunks: chunks.len(),
            total_chars,
            avg_chunk_chars: total_chars as f64 / chunks.len() as f64,
            min_chunk_chars,
            max_chunk_chars,
            overlap_chars,
        }
    }

    /// Temporarily override the strategy on this chunker, produce chunks,
    /// then restore the original strategy.
    pub fn rechunk_with_strategy(
        &mut self,
        doc_id: &str,
        text: &str,
        strategy: ChunkStrategy,
    ) -> Vec<TextChunk> {
        let saved = self.config.strategy.clone();
        self.config.strategy = strategy;
        let chunks = self.chunk_text(doc_id, text);
        self.config.strategy = saved;
        chunks
    }

    /// Add `key = value` metadata to every chunk in `chunks`.
    pub fn set_metadata(chunks: &mut [TextChunk], key: &str, value: &str) {
        for chunk in chunks.iter_mut() {
            chunk.metadata.insert(key.to_string(), value.to_string());
        }
    }

    /// Return `(chunks_produced, documents_processed)` counters.
    pub fn chunker_stats(&self) -> (u64, u64) {
        (self.chunks_produced, self.documents_processed)
    }
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Convert a character-level index into the corresponding byte offset within
/// the UTF-8 string `s`.  Returns `s.len()` when `char_idx >= s.chars().count()`.
fn char_idx_to_byte(s: &str, char_idx: usize) -> usize {
    s.char_indices()
        .nth(char_idx)
        .map(|(b, _)| b)
        .unwrap_or(s.len())
}

/// Split `text` into sentences at `. `, `! `, and `? ` delimiters, preserving
/// the terminating punctuation on each sentence.
fn split_into_sentences(text: &str) -> Vec<String> {
    if text.is_empty() {
        return Vec::new();
    }

    let mut sentences: Vec<String> = Vec::new();
    let mut current = String::new();
    let chars: Vec<char> = text.chars().collect();
    let len = chars.len();
    let mut i = 0;

    while i < len {
        let ch = chars[i];
        current.push(ch);

        if (ch == '.' || ch == '!' || ch == '?') && i + 1 < len && chars[i + 1] == ' ' {
            sentences.push(current.trim_end().to_string());
            current = String::new();
            // Skip the trailing space.
            i += 2;
            continue;
        }
        i += 1;
    }

    // Push any remaining text as the final (possibly unterminated) sentence.
    let tail = current.trim_end().to_string();
    if !tail.is_empty() {
        sentences.push(tail);
    }

    sentences
}

/// Group `sentences` into chunks whose character length stays within
/// `max_chars`, with `overlap_sentences` sentences copied from the end of the
/// previous chunk to the start of the next.
///
/// Byte offsets are computed relative to `full_text`.
fn group_sentences_into_chunks(
    doc_id: &str,
    full_text: &str,
    sentences: &[String],
    max_chars: usize,
    overlap_sentences: usize,
) -> Vec<TextChunk> {
    if sentences.is_empty() {
        return Vec::new();
    }

    let effective_max = max_chars.max(1);
    let mut chunks: Vec<TextChunk> = Vec::new();
    let mut chunk_index = 0usize;
    let mut sent_idx = 0usize; // First sentence of the current chunk (in `sentences`).

    while sent_idx < sentences.len() {
        let mut group: Vec<&str> = Vec::new();
        let mut char_count = 0usize;
        let mut i = sent_idx;

        while i < sentences.len() {
            let s = sentences[i].as_str();
            // Account for the space separator that will be inserted.
            let needed = if group.is_empty() {
                s.len()
            } else {
                s.len() + 1
            };

            if !group.is_empty() && char_count + needed > effective_max {
                break;
            }

            group.push(s);
            char_count += needed;
            i += 1;
        }

        // If a single sentence already exceeds max_chars, include it anyway to
        // avoid an infinite loop.
        if group.is_empty() {
            group.push(sentences[i].as_str());
            i += 1;
        }

        let content = group.join(" ");

        // Locate this chunk's byte offsets inside `full_text`.
        let (start_offset, end_offset) = find_chunk_offsets(&content, full_text, chunks.last());

        chunks.push(TextChunk {
            id: format!("{}-{}", doc_id, chunk_index),
            content,
            start_offset,
            end_offset,
            chunk_index,
            metadata: HashMap::new(),
        });
        chunk_index += 1;

        // Advance, but keep the last `overlap_sentences` sentences.
        let advance = (i - sent_idx).saturating_sub(overlap_sentences).max(1);
        sent_idx += advance;
    }

    chunks
}

/// Find the byte range of `chunk_content` within `full_text`.
///
/// The search starts after the previous chunk's start offset (if any) to
/// prefer later occurrences over earlier duplicates.  Returns `(0, 0)` if the
/// content cannot be located (e.g. after whitespace normalisation).
fn find_chunk_offsets(
    chunk_content: &str,
    full_text: &str,
    prev_chunk: Option<&TextChunk>,
) -> (usize, usize) {
    // Determine the search start position.
    let search_from = prev_chunk.map(|c| c.start_offset).unwrap_or(0);

    // Try an exact substring match first.
    if let Some(rel) = full_text[search_from..].find(chunk_content) {
        let start = search_from + rel;
        return (start, start + chunk_content.len());
    }

    // Fallback: try from the very beginning (handles duplicated overlap text).
    if let Some(rel) = full_text.find(chunk_content) {
        return (rel, rel + chunk_content.len());
    }

    // Could not locate; return sentinel values.
    (0, chunk_content.len())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use crate::document_chunker::{
        char_idx_to_byte, group_sentences_into_chunks, split_into_sentences, ChunkStrategy,
        DocumentChunker, DocumentChunkerConfig, TextChunk,
    };
    use std::collections::HashMap;

    fn default_chunker() -> DocumentChunker {
        DocumentChunker::new(DocumentChunkerConfig::default())
    }

    // -----------------------------------------------------------------------
    // Helper / unit tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_char_idx_to_byte_ascii() {
        let s = "hello";
        assert_eq!(char_idx_to_byte(s, 0), 0);
        assert_eq!(char_idx_to_byte(s, 3), 3);
        assert_eq!(char_idx_to_byte(s, 5), 5); // == s.len()
        assert_eq!(char_idx_to_byte(s, 99), 5); // out of range → s.len()
    }

    #[test]
    fn test_char_idx_to_byte_utf8() {
        let s = "héllo"; // 'é' = 2 bytes
                         // 'h' at byte 0, 'é' at byte 1, 'l' at byte 3, 'l' at 4, 'o' at 5
        assert_eq!(char_idx_to_byte(s, 0), 0);
        assert_eq!(char_idx_to_byte(s, 1), 1);
        assert_eq!(char_idx_to_byte(s, 2), 3);
    }

    #[test]
    fn test_split_into_sentences_basic() {
        let text = "Hello world. How are you? I am fine!";
        let sents = split_into_sentences(text);
        assert_eq!(sents.len(), 3);
        assert_eq!(sents[0], "Hello world.");
        assert_eq!(sents[1], "How are you?");
        assert_eq!(sents[2], "I am fine!");
    }

    #[test]
    fn test_split_into_sentences_no_terminator() {
        let text = "No terminator here";
        let sents = split_into_sentences(text);
        assert_eq!(sents.len(), 1);
        assert_eq!(sents[0], "No terminator here");
    }

    #[test]
    fn test_split_into_sentences_empty() {
        assert!(split_into_sentences("").is_empty());
    }

    #[test]
    fn test_split_into_sentences_single_sentence() {
        let text = "Just one sentence.";
        let sents = split_into_sentences(text);
        assert_eq!(sents.len(), 1);
    }

    // -----------------------------------------------------------------------
    // Fixed-size chunking
    // -----------------------------------------------------------------------

    #[test]
    fn test_fixed_size_basic() {
        let chunker = default_chunker();
        let text = "abcdefghij"; // 10 chars
        let chunks = chunker.chunk_fixed_size("doc", text, 5, 0);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].content, "abcde");
        assert_eq!(chunks[1].content, "fghij");
    }

    #[test]
    fn test_fixed_size_with_overlap() {
        let chunker = default_chunker();
        let text = "abcdefghij"; // 10 chars
        let chunks = chunker.chunk_fixed_size("doc", text, 6, 2);
        // step = 4; starts: 0 → "abcdef", 4 → "efghij", 8 → "ij"
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].content, "abcdef");
        assert_eq!(chunks[1].content, "efghij");
        assert_eq!(chunks[2].content, "ij");
    }

    #[test]
    fn test_fixed_size_overlap_clamp() {
        // overlap >= size → clamped to size-1=2, step=1
        // "abcde": starts 0,1,2,3,4 → 5 chunks
        let chunker = default_chunker();
        let text = "abcde";
        let chunks = chunker.chunk_fixed_size("doc", text, 3, 10);
        assert_eq!(chunks.len(), 5);
        assert_eq!(chunks[0].content, "abc");
        assert_eq!(chunks[4].content, "e");
    }

    #[test]
    fn test_fixed_size_zero_size_returns_empty() {
        let chunker = default_chunker();
        let chunks = chunker.chunk_fixed_size("doc", "hello", 0, 0);
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_fixed_size_empty_text() {
        let chunker = default_chunker();
        let chunks = chunker.chunk_fixed_size("doc", "", 5, 0);
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_fixed_size_chunk_ids() {
        let chunker = default_chunker();
        let chunks = chunker.chunk_fixed_size("my-doc", "abcdefghij", 5, 0);
        assert_eq!(chunks[0].id, "my-doc-0");
        assert_eq!(chunks[1].id, "my-doc-1");
    }

    #[test]
    fn test_fixed_size_byte_offsets() {
        let chunker = default_chunker();
        let text = "abcde";
        let chunks = chunker.chunk_fixed_size("d", text, 3, 0);
        assert_eq!(chunks[0].start_offset, 0);
        assert_eq!(chunks[0].end_offset, 3);
        assert_eq!(chunks[1].start_offset, 3);
        assert_eq!(chunks[1].end_offset, 5);
    }

    // -----------------------------------------------------------------------
    // Sentence-boundary chunking
    // -----------------------------------------------------------------------

    #[test]
    fn test_sentence_boundary_basic() {
        let chunker = default_chunker();
        let text = "Hello world. How are you? I am fine!";
        let chunks = chunker.chunk_sentence_boundary("doc", text, 100, 0);
        assert_eq!(chunks.len(), 1); // all fit in 100 chars
        assert!(chunks[0].content.contains("Hello world"));
    }

    #[test]
    fn test_sentence_boundary_splits_on_max_chars() {
        let chunker = default_chunker();
        let text = "Hello world. How are you? I am fine!";
        // Force split by using a small max.
        let chunks = chunker.chunk_sentence_boundary("doc", text, 15, 0);
        assert!(chunks.len() >= 2);
    }

    #[test]
    fn test_sentence_boundary_overlap() {
        let chunker = default_chunker();
        let text = "Sentence one. Sentence two. Sentence three. Sentence four.";
        let chunks = chunker.chunk_sentence_boundary("doc", text, 30, 1);
        // With overlap_sentences=1, the last sentence of each chunk is
        // carried into the next.
        assert!(chunks.len() >= 2);
        if chunks.len() >= 2 {
            // The second chunk should start with the overlapping sentence.
            let c0_last_sentence = chunks[0]
                .content
                .split(". ")
                .last()
                .unwrap_or("")
                .to_string();
            assert!(
                chunks[1].content.starts_with(&c0_last_sentence)
                    || chunks[1].content.contains(&c0_last_sentence)
            );
        }
    }

    #[test]
    fn test_sentence_boundary_empty_text() {
        let chunker = default_chunker();
        let chunks = chunker.chunk_sentence_boundary("doc", "", 512, 1);
        assert!(chunks.is_empty());
    }

    // -----------------------------------------------------------------------
    // Paragraph chunking
    // -----------------------------------------------------------------------

    #[test]
    fn test_paragraph_basic() {
        let chunker = default_chunker();
        let text = "First paragraph.\n\nSecond paragraph.";
        let chunks = chunker.chunk_paragraph("doc", text, 200);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].content, "First paragraph.");
        assert_eq!(chunks[1].content, "Second paragraph.");
    }

    #[test]
    fn test_paragraph_oversized() {
        let chunker = default_chunker();
        // Create a paragraph longer than max_chunk_chars.
        let long = "This is sentence one. This is sentence two. This is sentence three.";
        let text = format!("{}\n\nShort.", long);
        let chunks = chunker.chunk_paragraph("doc", &text, 30);
        // The long paragraph should be split further.
        assert!(chunks.len() >= 3);
    }

    #[test]
    fn test_paragraph_empty_paragraphs_skipped() {
        let chunker = default_chunker();
        let text = "Hello.\n\n\n\nWorld.";
        let chunks = chunker.chunk_paragraph("doc", text, 200);
        // The empty paragraph between the two real ones should be skipped.
        assert_eq!(chunks.len(), 2);
    }

    #[test]
    fn test_paragraph_single_paragraph() {
        let chunker = default_chunker();
        let text = "Single paragraph without any double newline.";
        let chunks = chunker.chunk_paragraph("doc", text, 200);
        assert_eq!(chunks.len(), 1);
    }

    // -----------------------------------------------------------------------
    // Semantic chunking
    // -----------------------------------------------------------------------

    #[test]
    fn test_semantic_basic() {
        let chunker = default_chunker();
        let text = "First sentence. Second sentence. Third sentence.";
        let chunks = chunker.chunk_semantic("doc", text, 200, 0.8);
        assert_eq!(chunks.len(), 1); // all fit in 200 chars
    }

    #[test]
    fn test_semantic_splits() {
        let chunker = default_chunker();
        let text = "First sentence here. Second sentence here. Third sentence here.";
        let chunks = chunker.chunk_semantic("doc", text, 25, 0.8);
        assert!(chunks.len() >= 2);
    }

    // -----------------------------------------------------------------------
    // chunk_text dispatch and counters
    // -----------------------------------------------------------------------

    #[test]
    fn test_chunk_text_dispatches_fixed_size() {
        let mut chunker = DocumentChunker::new(DocumentChunkerConfig {
            strategy: ChunkStrategy::FixedSize {
                size: 5,
                overlap: 0,
            },
            preserve_whitespace: false,
            min_chunk_chars: 1,
        });
        let chunks = chunker.chunk_text("d", "abcdefghij");
        assert_eq!(chunks.len(), 2);
    }

    #[test]
    fn test_chunk_text_counters() {
        let mut chunker = default_chunker();
        chunker.chunk_text("d1", "Hello world. How are you?");
        chunker.chunk_text("d2", "Another document.");
        let (produced, processed) = chunker.chunker_stats();
        assert_eq!(processed, 2);
        assert!(produced >= 2);
    }

    #[test]
    fn test_chunk_text_min_chunk_filter() {
        let mut chunker = DocumentChunker::new(DocumentChunkerConfig {
            strategy: ChunkStrategy::FixedSize {
                size: 3,
                overlap: 0,
            },
            preserve_whitespace: false,
            min_chunk_chars: 5,
        });
        // All chunks (size=3) will be smaller than min_chunk_chars=5 → filtered out.
        let chunks = chunker.chunk_text("d", "abcdefghi");
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_chunk_text_whitespace_stripped_by_default() {
        let mut chunker = DocumentChunker::new(DocumentChunkerConfig {
            strategy: ChunkStrategy::FixedSize {
                size: 10,
                overlap: 0,
            },
            preserve_whitespace: false,
            min_chunk_chars: 1,
        });
        let text = "   hello   ";
        let chunks = chunker.chunk_text("d", text);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].content.trim(), "hello");
    }

    #[test]
    fn test_chunk_text_preserve_whitespace() {
        let mut chunker = DocumentChunker::new(DocumentChunkerConfig {
            strategy: ChunkStrategy::FixedSize {
                size: 11,
                overlap: 0,
            },
            preserve_whitespace: true,
            min_chunk_chars: 1,
        });
        let text = "   hello   ";
        let chunks = chunker.chunk_text("d", text);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].content, "   hello   ");
    }

    // -----------------------------------------------------------------------
    // merge_small_chunks
    // -----------------------------------------------------------------------

    fn make_chunk(id: &str, content: &str, idx: usize) -> TextChunk {
        TextChunk {
            id: id.to_string(),
            content: content.to_string(),
            start_offset: 0,
            end_offset: content.len(),
            chunk_index: idx,
            metadata: HashMap::new(),
        }
    }

    #[test]
    fn test_merge_small_chunks_no_merge_needed() {
        let chunks = vec![
            make_chunk("d-0", "long enough content", 0),
            make_chunk("d-1", "also long enough", 1),
        ];
        let merged = DocumentChunker::merge_small_chunks(chunks, 5);
        assert_eq!(merged.len(), 2);
    }

    #[test]
    fn test_merge_small_chunks_merges_short() {
        let chunks = vec![
            make_chunk("d-0", "hi", 0),
            make_chunk("d-1", "there friend of mine", 1),
        ];
        let merged = DocumentChunker::merge_small_chunks(chunks, 5);
        assert_eq!(merged.len(), 1);
        assert!(merged[0].content.contains("hi"));
        assert!(merged[0].content.contains("there friend of mine"));
    }

    #[test]
    fn test_merge_small_chunks_empty() {
        let merged = DocumentChunker::merge_small_chunks(vec![], 5);
        assert!(merged.is_empty());
    }

    #[test]
    fn test_merge_small_chunks_re_indices() {
        let chunks = vec![
            make_chunk("d-0", "hi", 0),
            make_chunk("d-1", "there friend of mine", 1),
            make_chunk("d-2", "another long chunk here", 2),
        ];
        let merged = DocumentChunker::merge_small_chunks(chunks, 5);
        for (i, c) in merged.iter().enumerate() {
            assert_eq!(c.chunk_index, i);
        }
    }

    #[test]
    fn test_merge_small_chunks_all_short() {
        // All chunks are short → they cascade into a single merged chunk.
        let chunks = vec![
            make_chunk("d-0", "a", 0),
            make_chunk("d-1", "b", 1),
            make_chunk("d-2", "c", 2),
        ];
        let merged = DocumentChunker::merge_small_chunks(chunks, 10);
        assert_eq!(merged.len(), 1);
    }

    // -----------------------------------------------------------------------
    // stats
    // -----------------------------------------------------------------------

    #[test]
    fn test_stats_empty() {
        let s = DocumentChunker::stats(&[]);
        assert_eq!(s.total_chunks, 0);
        assert_eq!(s.total_chars, 0);
        assert_eq!(s.avg_chunk_chars, 0.0);
    }

    #[test]
    fn test_stats_single_chunk() {
        let chunks = vec![make_chunk("d-0", "hello world", 0)];
        let s = DocumentChunker::stats(&chunks);
        assert_eq!(s.total_chunks, 1);
        assert_eq!(s.total_chars, 11);
        assert!((s.avg_chunk_chars - 11.0).abs() < 1e-9);
        assert_eq!(s.min_chunk_chars, 11);
        assert_eq!(s.max_chunk_chars, 11);
    }

    #[test]
    fn test_stats_multiple_chunks() {
        let chunks = vec![
            make_chunk("d-0", "hello", 0),
            make_chunk("d-1", "world!!", 1),
        ];
        let s = DocumentChunker::stats(&chunks);
        assert_eq!(s.total_chunks, 2);
        assert_eq!(s.total_chars, 12);
        assert_eq!(s.min_chunk_chars, 5);
        assert_eq!(s.max_chunk_chars, 7);
    }

    // -----------------------------------------------------------------------
    // rechunk_with_strategy
    // -----------------------------------------------------------------------

    #[test]
    fn test_rechunk_with_strategy() {
        // Use min_chunk_chars=1 so FixedSize chunks of size 5 are not filtered.
        let mut chunker = DocumentChunker::new(DocumentChunkerConfig {
            strategy: ChunkStrategy::SentenceBoundary {
                max_chunk_chars: 512,
                overlap_sentences: 1,
            },
            preserve_whitespace: false,
            min_chunk_chars: 1,
        });
        let text = "abcdefghijklmnopqrstuvwxyz";
        let chunks = chunker.rechunk_with_strategy(
            "doc",
            text,
            ChunkStrategy::FixedSize {
                size: 5,
                overlap: 0,
            },
        );
        assert!(!chunks.is_empty());
        // Strategy should be restored afterwards.
        assert_eq!(
            chunker.config.strategy,
            ChunkStrategy::SentenceBoundary {
                max_chunk_chars: 512,
                overlap_sentences: 1
            }
        );
    }

    #[test]
    fn test_rechunk_strategy_preserved_on_empty() {
        let mut chunker = default_chunker();
        let original = chunker.config.strategy.clone();
        let _ = chunker.rechunk_with_strategy(
            "doc",
            "",
            ChunkStrategy::Paragraph {
                max_chunk_chars: 100,
            },
        );
        assert_eq!(chunker.config.strategy, original);
    }

    // -----------------------------------------------------------------------
    // set_metadata
    // -----------------------------------------------------------------------

    #[test]
    fn test_set_metadata() {
        let mut chunks = vec![make_chunk("d-0", "hello", 0), make_chunk("d-1", "world", 1)];
        DocumentChunker::set_metadata(&mut chunks, "source", "test");
        for chunk in &chunks {
            assert_eq!(
                chunk.metadata.get("source").map(|s| s.as_str()),
                Some("test")
            );
        }
    }

    #[test]
    fn test_set_metadata_overwrites() {
        let mut chunks = vec![make_chunk("d-0", "hello", 0)];
        DocumentChunker::set_metadata(&mut chunks, "k", "v1");
        DocumentChunker::set_metadata(&mut chunks, "k", "v2");
        assert_eq!(chunks[0].metadata["k"], "v2");
    }

    #[test]
    fn test_set_metadata_empty_slice() {
        let mut chunks: Vec<TextChunk> = Vec::new();
        // Should not panic.
        DocumentChunker::set_metadata(&mut chunks, "k", "v");
        assert!(chunks.is_empty());
    }

    // -----------------------------------------------------------------------
    // group_sentences_into_chunks (low-level)
    // -----------------------------------------------------------------------

    #[test]
    fn test_group_sentences_empty() {
        let result = group_sentences_into_chunks("doc", "", &[], 100, 0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_group_sentences_single_oversized() {
        // A sentence larger than max_chars should still produce one chunk.
        let sentences = vec!["This is a rather long sentence.".to_string()];
        let result =
            group_sentences_into_chunks("doc", "This is a rather long sentence.", &sentences, 5, 0);
        assert_eq!(result.len(), 1);
    }

    // -----------------------------------------------------------------------
    // End-to-end / integration tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_chunk_stats_roundtrip() {
        let mut chunker = default_chunker();
        let text = "The quick brown fox. Jumped over the lazy dog. And ran away quickly.";
        let chunks = chunker.chunk_text("doc", text);
        let s = DocumentChunker::stats(&chunks);
        assert_eq!(s.total_chunks, chunks.len());
        assert_eq!(
            s.total_chars,
            chunks.iter().map(|c| c.content.len()).sum::<usize>()
        );
    }

    #[test]
    fn test_chunk_indices_sequential() {
        let mut chunker = DocumentChunker::new(DocumentChunkerConfig {
            strategy: ChunkStrategy::FixedSize {
                size: 3,
                overlap: 0,
            },
            preserve_whitespace: false,
            min_chunk_chars: 1,
        });
        let chunks = chunker.chunk_text("d", "abcdefghi");
        for (i, c) in chunks.iter().enumerate() {
            assert_eq!(c.chunk_index, i);
        }
    }

    #[test]
    fn test_paragraph_strategy_via_chunk_text() {
        let mut chunker = DocumentChunker::new(DocumentChunkerConfig {
            strategy: ChunkStrategy::Paragraph {
                max_chunk_chars: 200,
            },
            preserve_whitespace: false,
            min_chunk_chars: 1,
        });
        let text = "Alpha beta gamma.\n\nDelta epsilon zeta.";
        let chunks = chunker.chunk_text("doc", text);
        assert_eq!(chunks.len(), 2);
    }

    #[test]
    fn test_semantic_strategy_via_chunk_text() {
        let mut chunker = DocumentChunker::new(DocumentChunkerConfig {
            strategy: ChunkStrategy::Semantic {
                max_chunk_chars: 200,
                similarity_threshold: 0.9,
            },
            preserve_whitespace: false,
            min_chunk_chars: 1,
        });
        let text = "Sentence alpha. Sentence beta. Sentence gamma.";
        let chunks = chunker.chunk_text("doc", text);
        assert!(!chunks.is_empty());
        // All sentences fit in 200 chars → single chunk.
        assert_eq!(chunks.len(), 1);
    }

    #[test]
    fn test_chunker_stats_increments() {
        let mut chunker = default_chunker();
        assert_eq!(chunker.chunker_stats(), (0, 0));
        let _ = chunker.chunk_text("d1", "One sentence.");
        let (produced1, processed1) = chunker.chunker_stats();
        assert_eq!(processed1, 1);
        assert!(produced1 >= 1);
        let _ = chunker.chunk_text("d2", "Two sentences. Right here.");
        let (produced2, processed2) = chunker.chunker_stats();
        assert_eq!(processed2, 2);
        assert!(produced2 >= produced1);
    }

    #[test]
    fn test_default_config() {
        let cfg = DocumentChunkerConfig::default();
        assert_eq!(cfg.min_chunk_chars, 10);
        assert!(!cfg.preserve_whitespace);
        match cfg.strategy {
            ChunkStrategy::SentenceBoundary {
                max_chunk_chars,
                overlap_sentences,
            } => {
                assert_eq!(max_chunk_chars, 512);
                assert_eq!(overlap_sentences, 1);
            }
            _ => panic!("Expected SentenceBoundary"),
        }
    }

    #[test]
    fn test_chunk_text_returns_correct_ids() {
        let mut chunker = DocumentChunker::new(DocumentChunkerConfig {
            strategy: ChunkStrategy::FixedSize {
                size: 5,
                overlap: 0,
            },
            preserve_whitespace: false,
            min_chunk_chars: 1,
        });
        let chunks = chunker.chunk_text("myid", "abcdefghij");
        assert_eq!(chunks[0].id, "myid-0");
        assert_eq!(chunks[1].id, "myid-1");
    }

    #[test]
    fn test_stats_overlap_detection() {
        // For fixed-size chunks with overlap the offset ranges overlap.
        let chunker = default_chunker();
        let text = "abcdefghij";
        let chunks = chunker.chunk_fixed_size("d", text, 6, 2);
        // step=4: chunks start at 0, 4, 8
        // pairs: (0..6, 4..10) → 2 overlapping; (4..10, 8..10) → 2 overlapping
        // total overlap_chars = 4
        let s = DocumentChunker::stats(&chunks);
        assert_eq!(s.overlap_chars, 4);
    }
}
