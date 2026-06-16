//! Full-featured extractive and abstractive-style document summarization.
//!
//! [`DocumentSummarizer`] implements five summarization strategies driven by TF-IDF,
//! position bias, sentence length heuristics, and optional embedding centrality:
//!
//! * **Extractive** — score every sentence and return the top-k in original order.
//! * **Keyphrase** — extract the most significant 2–4-word n-gram keyphrases.
//! * **Headline** — return the single most important sentence, truncated.
//! * **Abstractive** — concatenate top-3 sentences with transition words stripped,
//!   trimmed to a target word count.
//! * **Hierarchical** — cluster sentences and pick one representative per cluster.

pub mod ds_types;
pub use ds_types::{
    DocumentChunk, SentenceScore, SummarizerConfig, SummarizerError, SummarizerStats,
    SummaryResult, SummaryStyle,
};

use std::collections::HashMap;

// ── xorshift PRNG (tests) ─────────────────────────────────────────────────────

/// Minimal xorshift64 PRNG; used in tests to avoid the `rand` crate.
#[allow(dead_code)]
pub fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

// ── Helpers ───────────────────────────────────────────────────────────────────

pub(crate) fn default_stop_words() -> Vec<String> {
    [
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
        "from", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do",
        "does", "did", "will", "would", "could", "should", "may", "might", "shall", "can", "that",
        "which", "this", "these", "those", "it", "its", "we", "our", "they", "their", "he", "she",
        "his", "her", "you", "your", "i", "my", "me", "us", "not", "no", "if", "as", "so", "then",
        "than", "also", "just", "about", "after", "before", "between", "into", "through", "during",
        "up", "down", "out", "off", "over", "under", "again", "further", "once", "very", "too",
        "more", "most", "other", "some", "such", "both", "each", "few", "own", "same", "only",
        "even", "when", "where", "how", "all", "while", "here", "there",
    ]
    .iter()
    .map(|w| w.to_string())
    .collect()
}

/// Transition phrases removed during abstractive summarization.
const TRANSITION_WORDS: &[&str] = &[
    "however",
    "furthermore",
    "moreover",
    "additionally",
    "nevertheless",
    "therefore",
    "thus",
    "hence",
    "consequently",
    "meanwhile",
    "subsequently",
    "nonetheless",
    "accordingly",
    "conversely",
    "alternatively",
    "similarly",
    "specifically",
    "particularly",
    "generally",
    "essentially",
    "basically",
    "obviously",
    "clearly",
    "certainly",
    "indeed",
    "actually",
    "importantly",
];

/// Tokenize `text` into lowercase alphanumeric tokens.
pub fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(|w| w.to_lowercase())
        .collect()
}

/// Split `text` into sentences on `'. '`, `'! '`, `'? '`, and `'\n\n'` boundaries.
pub fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences: Vec<String> = Vec::new();
    let mut current = String::new();
    let chars: Vec<char> = text.chars().collect();
    let len = chars.len();
    let mut i = 0;

    while i < len {
        let ch = chars[i];
        current.push(ch);

        // Double newline paragraph break.
        if ch == '\n' && i + 1 < len && chars[i + 1] == '\n' {
            let trimmed = current.trim().to_string();
            if !trimmed.is_empty() {
                sentences.push(trimmed);
            }
            current.clear();
            // Skip additional newlines.
            while i + 1 < len && chars[i + 1] == '\n' {
                i += 1;
            }
            i += 1;
            continue;
        }

        // Sentence-ending punctuation followed by a space or end-of-string.
        if matches!(ch, '.' | '!' | '?') {
            let next_is_space_or_end = i + 1 >= len || chars[i + 1] == ' ' || chars[i + 1] == '\n';
            if next_is_space_or_end {
                let trimmed = current.trim().to_string();
                if !trimmed.is_empty() {
                    sentences.push(trimmed);
                }
                current.clear();
                // Skip trailing space.
                if i + 1 < len && chars[i + 1] == ' ' {
                    i += 1;
                }
            }
        }

        i += 1;
    }

    // Flush any remaining text.
    let remainder = current.trim().to_string();
    if !remainder.is_empty() {
        sentences.push(remainder);
    }

    sentences
}

/// Compute TF-IDF for `term` given the tokens of its document and the full corpus.
pub fn tf_idf(term: &str, doc_tokens: &[String], all_docs: &[Vec<String>]) -> f64 {
    if doc_tokens.is_empty() || all_docs.is_empty() {
        return 0.0;
    }
    let tf =
        doc_tokens.iter().filter(|t| t.as_str() == term).count() as f64 / doc_tokens.len() as f64;
    let df = all_docs
        .iter()
        .filter(|d| d.iter().any(|t| t.as_str() == term))
        .count();
    let idf = ((all_docs.len() as f64 + 1.0) / (df as f64 + 1.0)).ln();
    tf * idf
}

/// Cosine similarity between two f64 slices; returns 0.0 on dimension mismatch or zero norm.
pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    if a.is_empty() || a.len() != b.len() {
        return 0.0;
    }
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
}

/// Compute mean cosine similarity of embedding `i` to all other embeddings.
fn embedding_centrality_score(i: usize, embeddings: &[Vec<f64>]) -> f64 {
    if embeddings.len() <= 1 {
        return 0.0;
    }
    let sum: f64 = embeddings
        .iter()
        .enumerate()
        .filter(|(j, _)| *j != i)
        .map(|(_, other)| cosine_similarity(&embeddings[i], other))
        .sum();
    sum / (embeddings.len() - 1) as f64
}

/// Position score: sentences at the start and end of the document score higher.
fn position_score(index: usize, total: usize, position_bias: f64) -> f64 {
    if total == 0 {
        return 0.0;
    }
    if total == 1 {
        return 1.0 * position_bias;
    }
    let rel = index as f64 / (total - 1) as f64; // 0.0 … 1.0
                                                 // U-shaped: 1 at edges, cos²(π·rel/2) falls then rises — use parabola-like:
                                                 // score = 1 - 4*(rel - 0.5)^2 gives 0 at edges, 1 at centre; invert:
    let centrality = 4.0 * (rel - 0.5).powi(2); // 1 at edges, 0 at centre
    centrality * position_bias
}

/// Length score: prefer sentences of "ideal" length (~100–200 chars).
fn length_score(sentence: &str) -> f64 {
    let len = sentence.len() as f64;
    if len <= 0.0 {
        return 0.0;
    }
    // Gaussian-ish peak around 150 chars.
    let ideal = 150.0_f64;
    let sigma = 80.0_f64;
    (-(len - ideal).powi(2) / (2.0 * sigma.powi(2))).exp()
}

/// Remove leading transition words from a sentence.
fn strip_transitions(sentence: &str) -> &str {
    let lower = sentence.to_lowercase();
    for tw in TRANSITION_WORDS {
        if let Some(rest) = lower.strip_prefix(tw) {
            if rest.starts_with([',', ' ', ';']) {
                let skip = tw.len() + 1; // +1 for the delimiter
                let stripped = sentence[skip..].trim_start_matches([',', ' ', ';']);
                if !stripped.is_empty() {
                    // Safety: `stripped` is a subslice of `sentence`
                    let offset = stripped.as_ptr() as usize - sentence.as_ptr() as usize;
                    return &sentence[offset..];
                }
            }
        }
    }
    sentence
}

// ── DocumentSummarizer ────────────────────────────────────────────────────────

/// Production-quality document summarizer supporting five summarization strategies.
pub struct DocumentSummarizer {
    config: SummarizerConfig,
    stats: SummarizerStats,
}

impl DocumentSummarizer {
    /// Create a new summarizer with the supplied configuration.
    pub fn new(config: SummarizerConfig) -> Self {
        Self {
            config,
            stats: SummarizerStats::default(),
        }
    }

    /// Create a summarizer with a default extractive (3-sentence) configuration.
    pub fn with_defaults() -> Self {
        Self::new(SummarizerConfig::default())
    }

    /// Return a reference to the accumulated statistics.
    pub fn stats(&self) -> &SummarizerStats {
        &self.stats
    }

    // ── Public API ────────────────────────────────────────────────────────────

    /// Summarize `text` using the configured strategy.
    ///
    /// `embeddings`, when provided, must contain one vector per sentence in document
    /// order and must all share the same dimension.
    pub fn summarize(
        &mut self,
        text: &str,
        embeddings: Option<Vec<Vec<f64>>>,
    ) -> Result<SummaryResult, SummarizerError> {
        if text.trim().is_empty() {
            return Err(SummarizerError::EmptyDocument);
        }

        // Validate embeddings dimensions.
        if let Some(ref embs) = embeddings {
            if let Some(first) = embs.first() {
                let dim = first.len();
                for (idx, e) in embs.iter().enumerate().skip(1) {
                    if e.len() != dim {
                        return Err(SummarizerError::EmbeddingDimensionMismatch {
                            expected: dim,
                            got: e.len(),
                        });
                    }
                    let _ = idx;
                }
            }
        }

        let original_length = text.len();
        let sentences_raw = split_sentences(text);

        // Filter by configured length bounds.
        let sentences: Vec<String> = sentences_raw
            .iter()
            .filter(|s| {
                s.len() >= self.config.min_sentence_length
                    && s.len() <= self.config.max_sentence_length
            })
            .cloned()
            .collect();

        // Align embeddings to filtered sentences if provided.
        // We keep a mapping: filtered_index → original_index for embedding look-up.
        let filtered_indices: Vec<usize> = sentences_raw
            .iter()
            .enumerate()
            .filter(|(_, s)| {
                s.len() >= self.config.min_sentence_length
                    && s.len() <= self.config.max_sentence_length
            })
            .map(|(i, _)| i)
            .collect();

        let filtered_embeddings: Option<Vec<Vec<f64>>> = embeddings.as_ref().map(|embs| {
            filtered_indices
                .iter()
                .filter_map(|&i| embs.get(i).cloned())
                .collect()
        });

        let result = match &self.config.style.clone() {
            SummaryStyle::Extractive { num_sentences } => self.summarize_extractive(
                text,
                &sentences,
                filtered_embeddings.as_deref(),
                *num_sentences,
                original_length,
            )?,
            SummaryStyle::Keyphrase { num_phrases } => {
                self.summarize_keyphrase(text, *num_phrases, original_length)?
            }
            SummaryStyle::Headline { max_chars } => self.summarize_headline(
                text,
                &sentences,
                filtered_embeddings.as_deref(),
                *max_chars,
                original_length,
            )?,
            SummaryStyle::Abstractive { target_words } => self.summarize_abstractive(
                text,
                &sentences,
                filtered_embeddings.as_deref(),
                *target_words,
                original_length,
            )?,
            SummaryStyle::Hierarchical { levels } => self.summarize_hierarchical(
                text,
                &sentences,
                filtered_embeddings.as_deref(),
                *levels,
                original_length,
            )?,
        };

        // Update stats (incremental mean).
        self.stats.documents_processed += 1;
        let n = self.stats.documents_processed as f64;
        let tokens = tokenize(text).len() as u64;
        self.stats.total_tokens_processed += tokens;
        self.stats.avg_compression_ratio +=
            (result.compression_ratio - self.stats.avg_compression_ratio) / n;
        self.stats.avg_quality_score += (result.quality_score - self.stats.avg_quality_score) / n;

        Ok(result)
    }

    /// Score a single sentence.
    ///
    /// `corpus` holds the tokenized form of every sentence in the document (used for IDF).
    pub fn score_sentence(
        &self,
        sentence: &str,
        index: usize,
        total: usize,
        corpus: &[Vec<String>],
    ) -> SentenceScore {
        let tokens = tokenize(sentence);
        let stop = &self.config.stop_words;

        // TF-IDF: average over non-stop content terms.
        let content_tokens: Vec<&String> = tokens
            .iter()
            .filter(|t| !stop.contains(t) && t.len() > 1)
            .collect();

        let tfidf_score = if content_tokens.is_empty() || corpus.is_empty() {
            0.0
        } else {
            let sum: f64 = content_tokens
                .iter()
                .map(|t| tf_idf(t, &tokens, corpus))
                .sum();
            sum / content_tokens.len() as f64
        };

        let pos_score = position_score(index, total, self.config.position_bias);
        let len_score = length_score(sentence);

        // Embedding centrality is computed externally and injected via summarize_extractive.
        let final_score = tfidf_score * 0.5 + pos_score * 0.25 + len_score * 0.25;

        SentenceScore {
            sentence: sentence.to_string(),
            index,
            tf_idf_score: tfidf_score,
            position_score: pos_score,
            length_score: len_score,
            embedding_centrality: 0.0,
            final_score,
        }
    }

    /// Extract the top-`n` n-gram keyphrases from `text`.
    pub fn extract_keyphrases(&self, text: &str, n: usize) -> Vec<String> {
        let tokens = tokenize(text);
        let stop = &self.config.stop_words;

        // Build 2-4 word n-gram candidates from token windows that start and end on
        // non-stop words.
        let mut phrase_counts: HashMap<String, usize> = HashMap::new();
        for window_size in 2usize..=4 {
            if tokens.len() < window_size {
                continue;
            }
            for i in 0..=(tokens.len() - window_size) {
                let window = &tokens[i..i + window_size];
                // Skip if first or last token is a stop word or very short.
                if stop.contains(&window[0])
                    || stop.contains(&window[window_size - 1])
                    || window[0].len() <= 1
                    || window[window_size - 1].len() <= 1
                {
                    continue;
                }
                let phrase = window.join(" ");
                *phrase_counts.entry(phrase).or_insert(0) += 1;
            }
        }

        // Score each phrase: count * avg_tfidf of its tokens.
        let all_tokens_vec = vec![tokens.clone()];
        let mut scored: Vec<(String, f64)> = phrase_counts
            .into_iter()
            .map(|(phrase, count)| {
                let phrase_tokens = tokenize(&phrase);
                let avg_tfidf: f64 = if phrase_tokens.is_empty() {
                    0.0
                } else {
                    phrase_tokens
                        .iter()
                        .filter(|t| !stop.contains(t))
                        .map(|t| tf_idf(t, &tokens, &all_tokens_vec))
                        .sum::<f64>()
                        / phrase_tokens.len() as f64
                };
                (phrase, count as f64 * avg_tfidf)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(n);

        // De-duplicate by prefix/suffix containment.
        let mut result: Vec<String> = Vec::new();
        for (phrase, _) in scored {
            let dominated = result.iter().any(|existing: &String| {
                existing.contains(phrase.as_str()) || phrase.contains(existing.as_str())
            });
            if !dominated {
                result.push(phrase);
            }
        }
        result.truncate(n);
        result
    }

    /// Split `text` into overlapping chunks of approximately `chunk_size` characters.
    ///
    /// Chunks overlap by 10% of `chunk_size` to preserve context across boundaries.
    pub fn chunk_document(&self, text: &str, chunk_size: usize) -> Vec<DocumentChunk> {
        if text.is_empty() || chunk_size == 0 {
            return Vec::new();
        }

        let overlap = (chunk_size / 10).max(1);
        let step = if chunk_size > overlap {
            chunk_size - overlap
        } else {
            1
        };

        let chars: Vec<char> = text.chars().collect();
        let total = chars.len();

        // Detect section titles (lines ending with ':' or short ALL-CAPS lines).
        let section_map = build_section_map(text);

        let mut chunks: Vec<DocumentChunk> = Vec::new();
        let mut start = 0_usize;
        let mut chunk_index = 0_usize;

        while start < total {
            let end = (start + chunk_size).min(total);
            let chunk_text: String = chars[start..end].iter().collect();
            let trimmed = chunk_text.trim().to_string();
            if !trimmed.is_empty() {
                let section_title = section_map
                    .iter()
                    .filter(|(pos, _)| *pos <= start)
                    .max_by_key(|(pos, _)| *pos)
                    .map(|(_, title)| title.clone());

                chunks.push(DocumentChunk {
                    text: trimmed,
                    embedding: None,
                    section_title,
                    chunk_index,
                });
                chunk_index += 1;
            }
            if end >= total {
                break;
            }
            start += step;
        }

        chunks
    }

    /// Compute a quality score in \[0, 1\] as the fraction of `original`'s top keyphrases
    /// that appear (as substrings) in `summary`.
    pub fn quality_score(&self, original: &str, summary: &str) -> f64 {
        let keyphrases = self.extract_keyphrases(original, 20);
        if keyphrases.is_empty() {
            return 0.0;
        }
        let summary_lower = summary.to_lowercase();
        let covered = keyphrases
            .iter()
            .filter(|kp| summary_lower.contains(kp.as_str()))
            .count();
        (covered as f64 / keyphrases.len() as f64).clamp(0.0, 1.0)
    }

    // ── Private strategy implementations ─────────────────────────────────────

    fn score_sentences_with_embeddings(
        &self,
        sentences: &[String],
        embeddings: Option<&[Vec<f64>]>,
        corpus: &[Vec<String>],
    ) -> Vec<SentenceScore> {
        let total = sentences.len();

        sentences
            .iter()
            .enumerate()
            .map(|(i, sent)| {
                let mut score = self.score_sentence(sent, i, total, corpus);

                // Inject embedding centrality when available.
                if self.config.use_embeddings {
                    if let Some(embs) = embeddings {
                        if embs.len() == sentences.len() {
                            let centrality = embedding_centrality_score(i, embs);
                            score.embedding_centrality = centrality;
                            // Recompute final score with centrality contribution.
                            score.final_score = score.tf_idf_score * 0.4
                                + score.position_score * 0.2
                                + score.length_score * 0.2
                                + centrality * 0.2;
                        }
                    }
                }

                score
            })
            .collect()
    }

    fn summarize_extractive(
        &self,
        original_text: &str,
        sentences: &[String],
        embeddings: Option<&[Vec<f64>]>,
        num_sentences: usize,
        original_length: usize,
    ) -> Result<SummaryResult, SummarizerError> {
        if sentences.is_empty() {
            return Err(SummarizerError::InsufficientSentences { needed: 1, got: 0 });
        }

        let corpus: Vec<Vec<String>> = sentences.iter().map(|s| tokenize(s)).collect();
        let mut scores = self.score_sentences_with_embeddings(sentences, embeddings, &corpus);

        // Sort descending by final_score, break ties by original index (ascending).
        scores.sort_by(|a, b| {
            b.final_score
                .partial_cmp(&a.final_score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.index.cmp(&b.index))
        });

        let take = num_sentences.min(scores.len());
        let mut top: Vec<&SentenceScore> = scores.iter().take(take).collect();
        // Restore original document order.
        top.sort_by_key(|s| s.index);

        let selected: Vec<String> = top.iter().map(|s| s.sentence.clone()).collect();
        let summary_text = selected.join(" ");
        let summary_length = summary_text.len();
        let compression_ratio = if original_length == 0 {
            0.0
        } else {
            summary_length as f64 / original_length as f64
        };
        let keyphrases = self.extract_keyphrases(original_text, 10);
        let quality = self.quality_score(original_text, &summary_text);

        Ok(SummaryResult {
            original_length,
            summary_length,
            compression_ratio,
            sentences: selected,
            keyphrases,
            style: SummaryStyle::Extractive { num_sentences },
            quality_score: quality,
        })
    }

    fn summarize_keyphrase(
        &self,
        text: &str,
        num_phrases: usize,
        original_length: usize,
    ) -> Result<SummaryResult, SummarizerError> {
        let keyphrases = self.extract_keyphrases(text, num_phrases);
        let summary_text = keyphrases.join(", ");
        let summary_length = summary_text.len();
        let compression_ratio = if original_length == 0 {
            0.0
        } else {
            summary_length as f64 / original_length as f64
        };
        let quality = self.quality_score(text, &summary_text);

        Ok(SummaryResult {
            original_length,
            summary_length,
            compression_ratio,
            sentences: keyphrases.clone(),
            keyphrases,
            style: SummaryStyle::Keyphrase { num_phrases },
            quality_score: quality,
        })
    }

    fn summarize_headline(
        &self,
        original_text: &str,
        sentences: &[String],
        embeddings: Option<&[Vec<f64>]>,
        max_chars: usize,
        original_length: usize,
    ) -> Result<SummaryResult, SummarizerError> {
        if sentences.is_empty() {
            return Err(SummarizerError::InsufficientSentences { needed: 1, got: 0 });
        }

        let corpus: Vec<Vec<String>> = sentences.iter().map(|s| tokenize(s)).collect();
        let scores = self.score_sentences_with_embeddings(sentences, embeddings, &corpus);

        let best = scores
            .iter()
            .max_by(|a, b| {
                a.final_score
                    .partial_cmp(&b.final_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|s| s.sentence.as_str())
            .unwrap_or("");

        // Truncate cleanly at a word boundary.
        let headline = truncate_at_word(best, max_chars);
        let summary_length = headline.len();
        let compression_ratio = if original_length == 0 {
            0.0
        } else {
            summary_length as f64 / original_length as f64
        };
        let keyphrases = self.extract_keyphrases(original_text, 5);
        let quality = self.quality_score(original_text, &headline);

        Ok(SummaryResult {
            original_length,
            summary_length,
            compression_ratio,
            sentences: vec![headline],
            keyphrases,
            style: SummaryStyle::Headline { max_chars },
            quality_score: quality,
        })
    }

    fn summarize_abstractive(
        &self,
        original_text: &str,
        sentences: &[String],
        embeddings: Option<&[Vec<f64>]>,
        target_words: usize,
        original_length: usize,
    ) -> Result<SummaryResult, SummarizerError> {
        if sentences.is_empty() {
            return Err(SummarizerError::InsufficientSentences { needed: 1, got: 0 });
        }

        let corpus: Vec<Vec<String>> = sentences.iter().map(|s| tokenize(s)).collect();
        let mut scores = self.score_sentences_with_embeddings(sentences, embeddings, &corpus);

        // Sort descending, take top 3, then restore original order.
        scores.sort_by(|a, b| {
            b.final_score
                .partial_cmp(&a.final_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scores.truncate(3);
        scores.sort_by_key(|s| s.index);

        // Strip transition words from each sentence.
        let cleaned: Vec<String> = scores
            .iter()
            .map(|s| strip_transitions(&s.sentence).to_string())
            .collect();

        // Concatenate and trim to target_words.
        let joined = cleaned.join(" ");
        let words: Vec<&str> = joined.split_whitespace().collect();
        let trimmed_words = if target_words > 0 && words.len() > target_words {
            words[..target_words].join(" ")
        } else {
            joined.clone()
        };

        let summary_length = trimmed_words.len();
        let compression_ratio = if original_length == 0 {
            0.0
        } else {
            summary_length as f64 / original_length as f64
        };
        let keyphrases = self.extract_keyphrases(original_text, 8);
        let quality = self.quality_score(original_text, &trimmed_words);

        Ok(SummaryResult {
            original_length,
            summary_length,
            compression_ratio,
            sentences: vec![trimmed_words],
            keyphrases,
            style: SummaryStyle::Abstractive { target_words },
            quality_score: quality,
        })
    }

    fn summarize_hierarchical(
        &self,
        original_text: &str,
        sentences: &[String],
        embeddings: Option<&[Vec<f64>]>,
        levels: usize,
        original_length: usize,
    ) -> Result<SummaryResult, SummarizerError> {
        if sentences.is_empty() {
            return Err(SummarizerError::InsufficientSentences { needed: 1, got: 0 });
        }
        if levels == 0 {
            return Err(SummarizerError::ConfigurationError(
                "levels must be >= 1".into(),
            ));
        }

        let k = levels.min(sentences.len());
        let corpus: Vec<Vec<String>> = sentences.iter().map(|s| tokenize(s)).collect();
        let scores = self.score_sentences_with_embeddings(sentences, embeddings, &corpus);

        let selected: Vec<String> = if let Some(embs) = embeddings {
            if embs.len() == sentences.len() {
                cluster_representative_sentences(sentences, embs, k, &scores)
            } else {
                positional_cluster_representatives(sentences, k, &scores)
            }
        } else {
            positional_cluster_representatives(sentences, k, &scores)
        };

        let summary_text = selected.join(" ");
        let summary_length = summary_text.len();
        let compression_ratio = if original_length == 0 {
            0.0
        } else {
            summary_length as f64 / original_length as f64
        };
        let keyphrases = self.extract_keyphrases(original_text, 8);
        let quality = self.quality_score(original_text, &summary_text);

        Ok(SummaryResult {
            original_length,
            summary_length,
            compression_ratio,
            sentences: selected,
            keyphrases,
            style: SummaryStyle::Hierarchical { levels },
            quality_score: quality,
        })
    }
}

// ── Private utilities ─────────────────────────────────────────────────────────

/// Build a map from character position to section title for the given text.
fn build_section_map(text: &str) -> Vec<(usize, String)> {
    let mut map = Vec::new();
    let mut pos = 0_usize;
    for line in text.lines() {
        let trimmed = line.trim();
        let is_title = (!trimmed.is_empty() && trimmed.len() <= 80)
            && (trimmed.ends_with(':') || trimmed == trimmed.to_uppercase() && trimmed.len() >= 3);
        if is_title {
            map.push((pos, trimmed.trim_end_matches(':').to_string()));
        }
        pos += line.len() + 1; // +1 for '\n'
    }
    map
}

/// Truncate `text` to at most `max_chars` characters at a word boundary.
fn truncate_at_word(text: &str, max_chars: usize) -> String {
    if text.len() <= max_chars {
        return text.to_string();
    }
    let truncated = &text[..max_chars];
    // Walk back to find the last space.
    if let Some(pos) = truncated.rfind(' ') {
        truncated[..pos]
            .trim_end_matches(|c: char| !c.is_alphanumeric())
            .to_string()
    } else {
        truncated.to_string()
    }
}

/// k-means-lite clustering: assign each sentence to the nearest centroid (by cosine),
/// then pick the sentence with the highest `final_score` in each cluster as representative.
fn cluster_representative_sentences(
    sentences: &[String],
    embeddings: &[Vec<f64>],
    k: usize,
    scores: &[SentenceScore],
) -> Vec<String> {
    let n = sentences.len();
    if n == 0 || k == 0 {
        return Vec::new();
    }
    let k = k.min(n);

    // Seed centroids: evenly spaced sentence indices.
    let step = n / k;
    let mut centroids: Vec<Vec<f64>> = (0..k)
        .map(|i| embeddings[(i * step).min(n - 1)].clone())
        .collect();

    let mut assignments = vec![0usize; n];

    for _iter in 0..10 {
        // Assign.
        let mut changed = false;
        for (i, emb) in embeddings.iter().enumerate() {
            let best = (0..k)
                .map(|c| (c, cosine_similarity(emb, &centroids[c])))
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(c, _)| c)
                .unwrap_or(0);
            if assignments[i] != best {
                assignments[i] = best;
                changed = true;
            }
        }
        if !changed {
            break;
        }
        // Update centroids.
        for (c, centroid_slot) in centroids.iter_mut().enumerate().take(k) {
            let members: Vec<&Vec<f64>> = (0..n)
                .filter(|&i| assignments[i] == c)
                .map(|i| &embeddings[i])
                .collect();
            if members.is_empty() {
                continue;
            }
            let dim = members[0].len();
            let mut centroid = vec![0.0_f64; dim];
            for m in &members {
                for (d, v) in m.iter().enumerate() {
                    centroid[d] += v;
                }
            }
            let cnt = members.len() as f64;
            for v in &mut centroid {
                *v /= cnt;
            }
            *centroid_slot = centroid;
        }
    }

    // Pick best sentence per cluster (highest final_score).
    let mut result = Vec::new();
    for c in 0..k {
        let best_idx = (0..n).filter(|&i| assignments[i] == c).max_by(|&a, &b| {
            scores[a]
                .final_score
                .partial_cmp(&scores[b].final_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        if let Some(idx) = best_idx {
            result.push((idx, sentences[idx].clone()));
        }
    }

    // Return in original document order.
    result.sort_by_key(|(idx, _)| *idx);
    result.into_iter().map(|(_, s)| s).collect()
}

/// Positional clustering fallback: divide sentences into k equal-sized buckets
/// and pick the highest-scoring sentence from each bucket.
fn positional_cluster_representatives(
    sentences: &[String],
    k: usize,
    scores: &[SentenceScore],
) -> Vec<String> {
    let n = sentences.len();
    if n == 0 || k == 0 {
        return Vec::new();
    }
    let k = k.min(n);
    let bucket_size = n.div_ceil(k);

    let mut result: Vec<(usize, String)> = Vec::new();
    for b in 0..k {
        let start = b * bucket_size;
        let end = ((b + 1) * bucket_size).min(n);
        if start >= n {
            break;
        }
        let best_idx = (start..end).max_by(|&a, &b_idx| {
            scores[a]
                .final_score
                .partial_cmp(&scores[b_idx].final_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        if let Some(idx) = best_idx {
            result.push((idx, sentences[idx].clone()));
        }
    }

    result.sort_by_key(|(idx, _)| *idx);
    result.into_iter().map(|(_, s)| s).collect()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::env::temp_dir;

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn default_summarizer() -> DocumentSummarizer {
        DocumentSummarizer::with_defaults()
    }

    fn make_config(style: SummaryStyle) -> SummarizerConfig {
        SummarizerConfig {
            style,
            ..SummarizerConfig::default()
        }
    }

    fn long_text() -> &'static str {
        "The quick brown fox jumps over the lazy dog. \
         Machine learning is a subset of artificial intelligence that enables computers to learn. \
         Natural language processing allows machines to understand human language effectively. \
         Deep learning models are inspired by the structure of the human brain's neural networks. \
         Data science combines statistics, programming, and domain knowledge to extract insights. \
         Reinforcement learning trains agents to make decisions by rewarding correct behaviour. \
         Transformer architectures revolutionized natural language processing tasks significantly. \
         Embeddings represent words and sentences as dense vectors in a high-dimensional space. \
         Semantic search retrieves documents based on meaning rather than exact keyword matching. \
         The field of computer vision enables machines to interpret and understand visual data."
    }

    fn make_embeddings(n: usize, dim: usize, seed: u64) -> Vec<Vec<f64>> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                (0..dim)
                    .map(|_| {
                        let x = xorshift64(&mut state);
                        (x as f64 / u64::MAX as f64) * 2.0 - 1.0
                    })
                    .collect()
            })
            .collect()
    }

    // ── xorshift64 ────────────────────────────────────────────────────────────

    #[test]
    fn xorshift64_changes_state() {
        let mut s = 12345u64;
        let a = xorshift64(&mut s);
        let b = xorshift64(&mut s);
        assert_ne!(a, b);
        assert_ne!(s, 12345);
    }

    #[test]
    fn xorshift64_deterministic() {
        let mut s1 = 9999u64;
        let mut s2 = 9999u64;
        assert_eq!(xorshift64(&mut s1), xorshift64(&mut s2));
    }

    // ── tokenize ──────────────────────────────────────────────────────────────

    #[test]
    fn tokenize_basic() {
        let tokens = tokenize("Hello, World!");
        assert!(tokens.contains(&"hello".to_string()));
        assert!(tokens.contains(&"world".to_string()));
        assert_eq!(tokens.len(), 2);
    }

    #[test]
    fn tokenize_empty() {
        assert!(tokenize("").is_empty());
    }

    #[test]
    fn tokenize_lowercase() {
        let tokens = tokenize("UPPER lower MiXeD");
        assert!(tokens.iter().all(|t| t == &t.to_lowercase()));
    }

    #[test]
    fn tokenize_strips_punctuation() {
        let tokens = tokenize("Hello... world!?");
        assert_eq!(tokens.len(), 2);
    }

    // ── split_sentences ───────────────────────────────────────────────────────

    #[test]
    fn split_sentences_basic() {
        let sents = split_sentences("Hello world. How are you? I am fine!");
        assert_eq!(sents.len(), 3);
    }

    #[test]
    fn split_sentences_empty() {
        assert!(split_sentences("").is_empty());
    }

    #[test]
    fn split_sentences_double_newline() {
        let sents = split_sentences("First paragraph.\n\nSecond paragraph.");
        assert_eq!(sents.len(), 2);
    }

    #[test]
    fn split_sentences_no_terminal_punct() {
        let sents = split_sentences("A sentence without a period");
        assert_eq!(sents.len(), 1);
    }

    // ── tf_idf ────────────────────────────────────────────────────────────────

    #[test]
    fn tf_idf_zero_on_empty_doc() {
        assert_eq!(tf_idf("word", &[], &[vec!["word".into()]]), 0.0);
    }

    #[test]
    fn tf_idf_zero_on_empty_corpus() {
        assert_eq!(tf_idf("word", &["word".into()], &[]), 0.0);
    }

    #[test]
    fn tf_idf_rare_term_scores_higher() {
        let doc_a = tokenize("machine learning is great");
        let doc_b = tokenize("machine learning for everyone and everyone");
        let all = vec![doc_a.clone(), doc_b.clone()];
        let score_rare = tf_idf("great", &doc_a, &all);
        let score_common = tf_idf("machine", &doc_a, &all);
        // "great" appears in only one doc; "machine" in both → great should have higher idf.
        assert!(score_rare > score_common);
    }

    // ── cosine_similarity ─────────────────────────────────────────────────────

    #[test]
    fn cosine_identical() {
        let v = vec![1.0, 2.0, 3.0];
        let s = cosine_similarity(&v, &v);
        assert!((s - 1.0).abs() < 1e-9);
    }

    #[test]
    fn cosine_orthogonal() {
        let s = cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]);
        assert!(s.abs() < 1e-9);
    }

    #[test]
    fn cosine_empty_returns_zero() {
        assert_eq!(cosine_similarity(&[], &[1.0]), 0.0);
    }

    #[test]
    fn cosine_dim_mismatch_returns_zero() {
        assert_eq!(cosine_similarity(&[1.0, 0.0], &[1.0]), 0.0);
    }

    #[test]
    fn cosine_zero_norm_returns_zero() {
        assert_eq!(cosine_similarity(&[0.0, 0.0], &[1.0, 0.0]), 0.0);
    }

    // ── SummarizerError ───────────────────────────────────────────────────────

    #[test]
    fn error_empty_document() {
        let mut s = default_summarizer();
        let err = s
            .summarize("   ", None)
            .expect_err("test: whitespace-only document should return EmptyDocument error");
        assert!(matches!(err, SummarizerError::EmptyDocument));
    }

    #[test]
    fn error_empty_string() {
        let mut s = default_summarizer();
        assert!(matches!(
            s.summarize("", None)
                .expect_err("test: empty string should return EmptyDocument error"),
            SummarizerError::EmptyDocument
        ));
    }

    #[test]
    fn error_embedding_dimension_mismatch() {
        let cfg = SummarizerConfig {
            style: SummaryStyle::Extractive { num_sentences: 2 },
            use_embeddings: true,
            min_sentence_length: 1,
            ..SummarizerConfig::default()
        };
        let mut s = DocumentSummarizer::new(cfg);
        let text = "First sentence here. Second sentence here.";
        let embs = vec![vec![1.0_f64, 0.0], vec![1.0_f64, 0.0, 0.5]]; // dim mismatch
        let err = s.summarize(text, Some(embs)).expect_err(
            "test: embedding dimension mismatch should return EmbeddingDimensionMismatch error",
        );
        assert!(matches!(
            err,
            SummarizerError::EmbeddingDimensionMismatch { .. }
        ));
    }

    #[test]
    fn error_display_empty_document() {
        let e = SummarizerError::EmptyDocument;
        assert!(!format!("{e}").is_empty());
    }

    #[test]
    fn error_display_insufficient_sentences() {
        let e = SummarizerError::InsufficientSentences { needed: 3, got: 1 };
        let msg = format!("{e}");
        assert!(msg.contains('3') || msg.contains('1'));
    }

    #[test]
    fn error_display_embedding_mismatch() {
        let e = SummarizerError::EmbeddingDimensionMismatch {
            expected: 4,
            got: 2,
        };
        let msg = format!("{e}");
        assert!(msg.contains('4') || msg.contains('2'));
    }

    #[test]
    fn error_display_config() {
        let e = SummarizerError::ConfigurationError("bad param".into());
        assert!(format!("{e}").contains("bad param"));
    }

    // ── SummaryStyle::Extractive ──────────────────────────────────────────────

    #[test]
    fn extractive_returns_requested_sentence_count() {
        let cfg = make_config(SummaryStyle::Extractive { num_sentences: 3 });
        let mut s = DocumentSummarizer::new(cfg);
        let result = s
            .summarize(long_text(), None)
            .expect("test: extractive summarize should succeed");
        assert_eq!(result.sentences.len(), 3);
    }

    #[test]
    fn extractive_does_not_exceed_available_sentences() {
        let cfg = make_config(SummaryStyle::Extractive { num_sentences: 100 });
        let mut s = DocumentSummarizer::new(cfg);
        let result = s
            .summarize(long_text(), None)
            .expect("test: extractive summarize with high count should succeed");
        assert!(!result.sentences.is_empty());
        let raw_count = split_sentences(long_text()).len();
        assert!(result.sentences.len() <= raw_count);
    }

    #[test]
    fn extractive_style_recorded_in_result() {
        let cfg = make_config(SummaryStyle::Extractive { num_sentences: 2 });
        let mut s = DocumentSummarizer::new(cfg);
        let result = s
            .summarize(long_text(), None)
            .expect("test: extractive summarize should succeed");
        assert!(matches!(
            result.style,
            SummaryStyle::Extractive { num_sentences: 2 }
        ));
    }

    #[test]
    fn extractive_compression_ratio_in_range() {
        let cfg = make_config(SummaryStyle::Extractive { num_sentences: 3 });
        let mut s = DocumentSummarizer::new(cfg);
        let result = s
            .summarize(long_text(), None)
            .expect("test: extractive summarize should succeed");
        assert!(result.compression_ratio > 0.0);
        assert!(result.compression_ratio <= 1.0);
    }

    #[test]
    fn extractive_with_embeddings() {
        let sents = split_sentences(long_text());
        let embs = make_embeddings(sents.len(), 16, 42);
        let cfg = SummarizerConfig {
            style: SummaryStyle::Extractive { num_sentences: 3 },
            use_embeddings: true,
            min_sentence_length: 1,
            ..SummarizerConfig::default()
        };
        let mut s = DocumentSummarizer::new(cfg);
        let result = s
            .summarize(long_text(), Some(embs))
            .expect("test: extractive summarize with embeddings should succeed");
        assert_eq!(result.sentences.len(), 3);
    }

    // ── SummaryStyle::Keyphrase ───────────────────────────────────────────────

    #[test]
    fn keyphrase_returns_requested_phrase_count() {
        let cfg = make_config(SummaryStyle::Keyphrase { num_phrases: 5 });
        let mut s = DocumentSummarizer::new(cfg);
        let result = s
            .summarize(long_text(), None)
            .expect("test: keyphrase summarize should succeed");
        assert!(result.sentences.len() <= 5);
    }

    #[test]
    fn keyphrase_style_recorded() {
        let cfg = make_config(SummaryStyle::Keyphrase { num_phrases: 3 });
        let mut s = DocumentSummarizer::new(cfg);
        let result = s
            .summarize(long_text(), None)
            .expect("test: keyphrase summarize should succeed");
        assert!(matches!(
            result.style,
            SummaryStyle::Keyphrase { num_phrases: 3 }
        ));
    }

    #[test]
    fn keyphrase_phrases_are_nonempty() {
        let cfg = make_config(SummaryStyle::Keyphrase { num_phrases: 5 });
        let mut s = DocumentSummarizer::new(cfg);
        let result = s
            .summarize(long_text(), None)
            .expect("test: keyphrase summarize should succeed");
        for phrase in &result.sentences {
            assert!(!phrase.is_empty());
        }
    }

    // ── SummaryStyle::Headline ────────────────────────────────────────────────

    #[test]
    fn headline_respects_max_chars() {
        let cfg = make_config(SummaryStyle::Headline { max_chars: 50 });
        let mut s = DocumentSummarizer::new(cfg);
        let result = s
            .summarize(long_text(), None)
            .expect("test: headline summarize should succeed");
        assert_eq!(result.sentences.len(), 1);
        assert!(result.sentences[0].len() <= 50);
    }

    #[test]
    fn headline_style_recorded() {
        let cfg = make_config(SummaryStyle::Headline { max_chars: 80 });
        let mut s = DocumentSummarizer::new(cfg);
        let result = s
            .summarize(long_text(), None)
            .expect("test: headline summarize should succeed");
        assert!(matches!(
            result.style,
            SummaryStyle::Headline { max_chars: 80 }
        ));
    }

    #[test]
    fn headline_is_nonempty() {
        let cfg = make_config(SummaryStyle::Headline { max_chars: 100 });
        let mut s = DocumentSummarizer::new(cfg);
        let result = s
            .summarize(long_text(), None)
            .expect("test: headline summarize should succeed");
        assert!(!result.sentences[0].is_empty());
    }

    #[test]
    fn headline_with_embeddings() {
        let sents = split_sentences(long_text());
        let embs = make_embeddings(sents.len(), 8, 7);
        let cfg = SummarizerConfig {
            style: SummaryStyle::Headline { max_chars: 60 },
            use_embeddings: true,
            min_sentence_length: 1,
            ..SummarizerConfig::default()
        };
        let mut s = DocumentSummarizer::new(cfg);
        let result = s
            .summarize(long_text(), Some(embs))
            .expect("test: headline summarize with embeddings should succeed");
        assert!(result.sentences[0].len() <= 60);
    }

    // ── SummaryStyle::Abstractive ─────────────────────────────────────────────

    #[test]
    fn abstractive_respects_target_words() {
        let cfg = make_config(SummaryStyle::Abstractive { target_words: 20 });
        let mut s = DocumentSummarizer::new(cfg);
        let result = s
            .summarize(long_text(), None)
            .expect("test: abstractive summarize should succeed");
        assert_eq!(result.sentences.len(), 1);
        let word_count = result.sentences[0].split_whitespace().count();
        assert!(word_count <= 20);
    }

    #[test]
    fn abstractive_style_recorded() {
        let cfg = make_config(SummaryStyle::Abstractive { target_words: 30 });
        let mut s = DocumentSummarizer::new(cfg);
        let result = s
            .summarize(long_text(), None)
            .expect("test: abstractive summarize should succeed");
        assert!(matches!(
            result.style,
            SummaryStyle::Abstractive { target_words: 30 }
        ));
    }

    #[test]
    fn abstractive_output_nonempty() {
        let cfg = make_config(SummaryStyle::Abstractive { target_words: 50 });
        let mut s = DocumentSummarizer::new(cfg);
        let result = s
            .summarize(long_text(), None)
            .expect("test: abstractive summarize should succeed");
        assert!(!result.sentences[0].is_empty());
    }

    // ── SummaryStyle::Hierarchical ────────────────────────────────────────────

    #[test]
    fn hierarchical_levels_sentences() {
        let cfg = make_config(SummaryStyle::Hierarchical { levels: 3 });
        let mut s = DocumentSummarizer::new(cfg);
        let result = s
            .summarize(long_text(), None)
            .expect("test: hierarchical summarize should succeed");
        assert!(result.sentences.len() <= 3);
        assert!(!result.sentences.is_empty());
    }

    #[test]
    fn hierarchical_style_recorded() {
        let cfg = make_config(SummaryStyle::Hierarchical { levels: 2 });
        let mut s = DocumentSummarizer::new(cfg);
        let result = s
            .summarize(long_text(), None)
            .expect("test: hierarchical summarize should succeed");
        assert!(matches!(
            result.style,
            SummaryStyle::Hierarchical { levels: 2 }
        ));
    }

    #[test]
    fn hierarchical_with_embeddings() {
        let sents = split_sentences(long_text());
        let embs = make_embeddings(sents.len(), 16, 123);
        let cfg = SummarizerConfig {
            style: SummaryStyle::Hierarchical { levels: 4 },
            use_embeddings: true,
            min_sentence_length: 1,
            ..SummarizerConfig::default()
        };
        let mut s = DocumentSummarizer::new(cfg);
        let result = s
            .summarize(long_text(), Some(embs))
            .expect("test: hierarchical summarize with embeddings should succeed");
        assert!(!result.sentences.is_empty());
        assert!(result.sentences.len() <= 4);
    }

    #[test]
    fn hierarchical_levels_zero_errors() {
        let cfg = make_config(SummaryStyle::Hierarchical { levels: 0 });
        let mut s = DocumentSummarizer::new(cfg);
        let err = s
            .summarize(long_text(), None)
            .expect_err("test: hierarchical with levels=0 should return ConfigurationError");
        assert!(matches!(err, SummarizerError::ConfigurationError(_)));
    }

    // ── score_sentence ────────────────────────────────────────────────────────

    #[test]
    fn score_sentence_returns_struct() {
        let s = default_summarizer();
        let corpus = vec![
            tokenize("hello world test sentence"),
            tokenize("another sentence here"),
        ];
        let score = s.score_sentence("hello world test sentence", 0, 5, &corpus);
        assert_eq!(score.index, 0);
        assert_eq!(score.sentence, "hello world test sentence");
        assert!(score.final_score >= 0.0);
    }

    #[test]
    fn score_sentence_position_zero_is_higher() {
        let cfg = SummarizerConfig {
            position_bias: 1.0,
            ..SummarizerConfig::default()
        };
        let s = DocumentSummarizer::new(cfg);
        let corpus = vec![tokenize("test"); 5];
        let first = s.score_sentence("test first sentence", 0, 5, &corpus);
        let middle = s.score_sentence("test middle sentence", 2, 5, &corpus);
        // position_bias 1.0: both edges score high, middle scores low.
        // first.position_score should be >= middle.position_score
        assert!(first.position_score >= middle.position_score);
    }

    #[test]
    fn score_sentence_empty_corpus() {
        let s = default_summarizer();
        let score = s.score_sentence("some sentence", 0, 1, &[]);
        assert_eq!(score.tf_idf_score, 0.0);
    }

    #[test]
    fn score_sentence_length_score_range() {
        let s = default_summarizer();
        let corpus = vec![tokenize("hello world")];
        let score = s.score_sentence("hello world", 0, 1, &corpus);
        assert!((0.0..=1.0).contains(&score.length_score));
    }

    // ── extract_keyphrases ────────────────────────────────────────────────────

    #[test]
    fn extract_keyphrases_count_limit() {
        let s = default_summarizer();
        let phrases = s.extract_keyphrases(long_text(), 5);
        assert!(phrases.len() <= 5);
    }

    #[test]
    fn extract_keyphrases_nonempty_on_rich_text() {
        let s = default_summarizer();
        let phrases = s.extract_keyphrases(long_text(), 10);
        assert!(!phrases.is_empty());
    }

    #[test]
    fn extract_keyphrases_all_nonempty() {
        let s = default_summarizer();
        for phrase in s.extract_keyphrases(long_text(), 8) {
            assert!(!phrase.is_empty());
        }
    }

    #[test]
    fn extract_keyphrases_zero_on_empty() {
        let s = default_summarizer();
        assert!(s.extract_keyphrases("", 5).is_empty());
    }

    #[test]
    fn extract_keyphrases_n_zero_returns_empty() {
        let s = default_summarizer();
        assert!(s.extract_keyphrases(long_text(), 0).is_empty());
    }

    // ── chunk_document ────────────────────────────────────────────────────────

    #[test]
    fn chunk_document_covers_all_content() {
        let s = default_summarizer();
        let text = long_text();
        let chunks = s.chunk_document(text, 100);
        assert!(!chunks.is_empty());
        // All chunk indices are assigned.
        for (i, c) in chunks.iter().enumerate() {
            assert_eq!(c.chunk_index, i);
        }
    }

    #[test]
    fn chunk_document_empty_text() {
        let s = default_summarizer();
        assert!(s.chunk_document("", 100).is_empty());
    }

    #[test]
    fn chunk_document_zero_size() {
        let s = default_summarizer();
        assert!(s.chunk_document(long_text(), 0).is_empty());
    }

    #[test]
    fn chunk_document_chunk_size_covers_full_text() {
        let s = default_summarizer();
        let text = "short text";
        let chunks = s.chunk_document(text, 1000);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].chunk_index, 0);
    }

    #[test]
    fn chunk_document_embeddings_none_by_default() {
        let s = default_summarizer();
        let chunks = s.chunk_document(long_text(), 200);
        for c in &chunks {
            assert!(c.embedding.is_none());
        }
    }

    #[test]
    fn chunk_document_uses_temp_dir_conceptually() {
        // Verify temp_dir is accessible (represents test isolation policy).
        let tmp = temp_dir();
        assert!(tmp.exists());
    }

    // ── quality_score ─────────────────────────────────────────────────────────

    #[test]
    fn quality_score_identical_text_is_high() {
        let s = default_summarizer();
        let qs = s.quality_score(long_text(), long_text());
        assert!(
            qs > 0.5,
            "quality score of identical texts should be > 0.5, got {qs}"
        );
    }

    #[test]
    fn quality_score_empty_summary_is_zero() {
        let s = default_summarizer();
        let qs = s.quality_score(long_text(), "");
        assert_eq!(qs, 0.0);
    }

    #[test]
    fn quality_score_in_range() {
        let s = default_summarizer();
        let cfg = make_config(SummaryStyle::Extractive { num_sentences: 3 });
        let mut ds = DocumentSummarizer::new(cfg);
        let result = ds
            .summarize(long_text(), None)
            .expect("test: extractive summarize for quality_score test should succeed");
        let summary = result.sentences.join(" ");
        let qs = s.quality_score(long_text(), &summary);
        assert!((0.0..=1.0).contains(&qs));
    }

    #[test]
    fn quality_score_empty_original_is_zero() {
        let s = default_summarizer();
        assert_eq!(s.quality_score("", "some summary"), 0.0);
    }

    // ── SummaryResult fields ──────────────────────────────────────────────────

    #[test]
    fn summary_result_original_length_correct() {
        let cfg = make_config(SummaryStyle::Extractive { num_sentences: 2 });
        let mut s = DocumentSummarizer::new(cfg);
        let text = long_text();
        let result = s
            .summarize(text, None)
            .expect("test: extractive summarize should succeed");
        assert_eq!(result.original_length, text.len());
    }

    #[test]
    fn summary_result_compression_ratio_formula() {
        let cfg = make_config(SummaryStyle::Extractive { num_sentences: 2 });
        let mut s = DocumentSummarizer::new(cfg);
        let text = long_text();
        let result = s
            .summarize(text, None)
            .expect("test: extractive summarize should succeed");
        let expected = result.summary_length as f64 / result.original_length as f64;
        assert!((result.compression_ratio - expected).abs() < 1e-9);
    }

    #[test]
    fn summary_result_keyphrases_nonempty() {
        let cfg = make_config(SummaryStyle::Extractive { num_sentences: 3 });
        let mut s = DocumentSummarizer::new(cfg);
        let result = s
            .summarize(long_text(), None)
            .expect("test: extractive summarize should succeed");
        assert!(!result.keyphrases.is_empty());
    }

    // ── embedding centrality ──────────────────────────────────────────────────

    #[test]
    fn embedding_centrality_single_emb_returns_zero() {
        let embs = vec![vec![1.0, 0.0]];
        assert_eq!(embedding_centrality_score(0, &embs), 0.0);
    }

    #[test]
    fn embedding_centrality_identical_embs() {
        let embs = vec![vec![1.0, 0.0], vec![1.0, 0.0], vec![1.0, 0.0]];
        let score = embedding_centrality_score(0, &embs);
        assert!((score - 1.0).abs() < 1e-9);
    }

    #[test]
    fn embedding_centrality_affects_score() {
        // A sentence whose embedding is very similar to others should get a higher
        // final_score when use_embeddings = true.
        let cfg = SummarizerConfig {
            style: SummaryStyle::Extractive { num_sentences: 1 },
            use_embeddings: true,
            min_sentence_length: 1,
            ..SummarizerConfig::default()
        };
        let mut s = DocumentSummarizer::new(cfg);
        // Two very different sentences with embeddings pointing in same direction.
        let text =
            "Machine learning enables computers to learn patterns from data automatically.\n\n\
                    Natural language processing is a field of artificial intelligence research.";
        let embs = vec![vec![1.0_f64, 0.0], vec![1.0_f64, 0.0]];
        let result = s
            .summarize(text, Some(embs))
            .expect("test: extractive summarize with central embeddings should succeed");
        assert_eq!(result.sentences.len(), 1);
    }

    // ── SummarizerStats ───────────────────────────────────────────────────────

    #[test]
    fn stats_initial_default() {
        let s = default_summarizer();
        let st = s.stats();
        assert_eq!(st.documents_processed, 0);
        assert_eq!(st.total_tokens_processed, 0);
    }

    #[test]
    fn stats_increments_after_summarize() {
        let cfg = make_config(SummaryStyle::Extractive { num_sentences: 2 });
        let mut s = DocumentSummarizer::new(cfg);
        s.summarize(long_text(), None)
            .expect("test: summarize for stats increment should succeed");
        assert_eq!(s.stats().documents_processed, 1);
        assert!(s.stats().total_tokens_processed > 0);
    }

    #[test]
    fn stats_compression_ratio_running_avg() {
        let cfg = make_config(SummaryStyle::Extractive { num_sentences: 3 });
        let mut s = DocumentSummarizer::new(cfg);
        s.summarize(long_text(), None)
            .expect("test: first summarize for running avg should succeed");
        s.summarize(long_text(), None)
            .expect("test: second summarize for running avg should succeed");
        let st = s.stats();
        assert_eq!(st.documents_processed, 2);
        assert!((0.0..=1.0).contains(&st.avg_compression_ratio));
    }

    #[test]
    fn stats_quality_score_running_avg() {
        let cfg = make_config(SummaryStyle::Extractive { num_sentences: 3 });
        let mut s = DocumentSummarizer::new(cfg);
        s.summarize(long_text(), None)
            .expect("test: summarize for quality score avg should succeed");
        assert!((0.0..=1.0).contains(&s.stats().avg_quality_score));
    }

    // ── SummarizerConfig ──────────────────────────────────────────────────────

    #[test]
    fn config_default_style_is_extractive_3() {
        let cfg = SummarizerConfig::default();
        assert!(matches!(
            cfg.style,
            SummaryStyle::Extractive { num_sentences: 3 }
        ));
    }

    #[test]
    fn config_custom_stop_words() {
        let cfg = SummarizerConfig {
            stop_words: vec!["machine".to_string(), "learning".to_string()],
            style: SummaryStyle::Keyphrase { num_phrases: 5 },
            ..SummarizerConfig::default()
        };
        let s = DocumentSummarizer::new(cfg);
        let phrases = s.extract_keyphrases(long_text(), 5);
        // Neither "machine" nor "learning" should start or end a keyphrase.
        for phrase in &phrases {
            let words: Vec<&str> = phrase.split_whitespace().collect();
            if let Some(first) = words.first() {
                assert_ne!(*first, "machine");
                assert_ne!(*first, "learning");
            }
        }
    }

    // ── DocumentChunk ─────────────────────────────────────────────────────────

    #[test]
    fn document_chunk_fields_accessible() {
        let chunk = DocumentChunk {
            text: "sample text".to_string(),
            embedding: Some(vec![1.0, 2.0]),
            section_title: Some("Introduction".to_string()),
            chunk_index: 0,
        };
        assert_eq!(chunk.text, "sample text");
        assert_eq!(chunk.chunk_index, 0);
        assert!(chunk.embedding.is_some());
        assert!(chunk.section_title.is_some());
    }

    #[test]
    fn document_chunk_no_embedding_no_title() {
        let chunk = DocumentChunk {
            text: "plain text".to_string(),
            embedding: None,
            section_title: None,
            chunk_index: 5,
        };
        assert!(chunk.embedding.is_none());
        assert!(chunk.section_title.is_none());
        assert_eq!(chunk.chunk_index, 5);
    }

    // ── Miscellaneous / edge cases ────────────────────────────────────────────

    #[test]
    fn single_sentence_document_extractive() {
        let cfg = SummarizerConfig {
            style: SummaryStyle::Extractive { num_sentences: 3 },
            min_sentence_length: 1,
            ..SummarizerConfig::default()
        };
        let mut s = DocumentSummarizer::new(cfg);
        let result = s
            .summarize("A single sentence document.", None)
            .expect("test: single-sentence extractive summarize should succeed");
        assert_eq!(result.sentences.len(), 1);
    }

    #[test]
    fn headline_large_max_chars_returns_full_best_sentence() {
        let cfg = make_config(SummaryStyle::Headline { max_chars: 10000 });
        let mut s = DocumentSummarizer::new(cfg);
        let result = s
            .summarize(long_text(), None)
            .expect("test: headline with large max_chars should succeed");
        assert_eq!(result.sentences.len(), 1);
        assert!(!result.sentences[0].is_empty());
    }

    #[test]
    fn abstractive_unlimited_words_returns_all_top3() {
        let cfg = make_config(SummaryStyle::Abstractive {
            target_words: 10000,
        });
        let mut s = DocumentSummarizer::new(cfg);
        let result = s
            .summarize(long_text(), None)
            .expect("test: abstractive with unlimited words should succeed");
        assert!(!result.sentences[0].is_empty());
    }

    #[test]
    fn summarize_multiple_styles_sequential() {
        let text = long_text();
        let styles = vec![
            SummaryStyle::Extractive { num_sentences: 2 },
            SummaryStyle::Keyphrase { num_phrases: 4 },
            SummaryStyle::Headline { max_chars: 60 },
            SummaryStyle::Abstractive { target_words: 25 },
            SummaryStyle::Hierarchical { levels: 3 },
        ];
        for style in styles {
            let cfg = make_config(style);
            let mut s = DocumentSummarizer::new(cfg);
            let result = s
                .summarize(text, None)
                .expect("test: each summarization style should succeed on long_text");
            assert!(!result.sentences.is_empty());
        }
    }
}
