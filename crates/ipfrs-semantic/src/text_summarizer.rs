//! Extractive text summarization using TF-IDF sentence scoring and TextRank graph-based ranking.
//!
//! [`TextSummarizer`] implements multiple summarization strategies:
//!
//! * **TF-IDF** — score each sentence by the sum of TF-IDF weights of its content words and
//!   return the top-N sentences in original order.
//! * **TextRank** — build a sentence similarity graph (cosine similarity of TF-IDF vectors) and
//!   run PageRank-style iteration; return top-N sentences in original order.
//! * **Lead** — baseline; return the first N sentences unchanged.
//! * **Hybrid** — weighted combination of TF-IDF and TextRank scores.
//!
//! The summarizer also maintains an optional external corpus (via [`TextSummarizer::add_to_corpus`])
//! to improve IDF estimates across multiple documents.

use std::collections::HashMap;

// ── Error ─────────────────────────────────────────────────────────────────────

/// Errors that can be returned by [`TextSummarizer`].
#[derive(Debug, Clone, PartialEq)]
pub enum SummarizerError {
    /// The text contained fewer sentences than the algorithm requires.
    InsufficientSentences {
        /// Minimum number required.
        min: usize,
        /// Actual number found.
        got: usize,
    },
    /// The input string was empty or contained only whitespace.
    EmptyText,
    /// A configuration parameter was invalid.
    InvalidConfig(String),
}

impl std::fmt::Display for SummarizerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InsufficientSentences { min, got } => {
                write!(f, "insufficient sentences: need at least {min}, got {got}")
            }
            Self::EmptyText => write!(f, "input text is empty"),
            Self::InvalidConfig(msg) => write!(f, "invalid configuration: {msg}"),
        }
    }
}

impl std::error::Error for SummarizerError {}

// ── Summarization method ──────────────────────────────────────────────────────

/// Which summarization algorithm [`TextSummarizer`] should use.
#[derive(Debug, Clone, PartialEq)]
pub enum SummarizationMethod {
    /// Score sentences by sum of TF-IDF weights; return top `top_n` in original order.
    TfIdf {
        /// Number of sentences to include in the summary.
        top_n: usize,
    },
    /// PageRank-style sentence graph ranking; return top `top_n` in original order.
    TextRank {
        /// Number of sentences to include in the summary.
        top_n: usize,
        /// Damping factor (typically 0.85).
        damping: f64,
        /// Maximum number of PageRank iterations.
        max_iter: u32,
    },
    /// Return the first `n_sentences` sentences (baseline).
    Lead {
        /// Number of leading sentences to return.
        n_sentences: usize,
    },
    /// Weighted combination of TF-IDF and TextRank scores.
    Hybrid {
        /// Number of sentences to include in the summary.
        top_n: usize,
        /// Weight applied to the TF-IDF score component (should sum to 1.0 with `textrank_weight`).
        tfidf_weight: f64,
        /// Weight applied to the TextRank score component.
        textrank_weight: f64,
    },
}

impl SummarizationMethod {
    /// Returns the maximum number of sentences this method can return, or `None` for Lead.
    fn top_n(&self) -> Option<usize> {
        match self {
            Self::TfIdf { top_n } => Some(*top_n),
            Self::TextRank { top_n, .. } => Some(*top_n),
            Self::Lead { n_sentences } => Some(*n_sentences),
            Self::Hybrid { top_n, .. } => Some(*top_n),
        }
    }

    fn name(&self) -> &'static str {
        match self {
            Self::TfIdf { .. } => "tfidf",
            Self::TextRank { .. } => "textrank",
            Self::Lead { .. } => "lead",
            Self::Hybrid { .. } => "hybrid",
        }
    }
}

// ── Config ────────────────────────────────────────────────────────────────────

/// Configuration for [`TextSummarizer`].
#[derive(Debug, Clone)]
pub struct SummarizerConfig {
    /// Which algorithm to use.
    pub method: SummarizationMethod,
    /// Minimum character length for a sentence to be considered (inclusive).
    pub min_sentence_length: usize,
    /// Maximum character length for a sentence to be considered (inclusive).
    pub max_sentence_length: usize,
    /// Words to ignore during tokenization.
    pub stop_words: Vec<String>,
}

impl Default for SummarizerConfig {
    fn default() -> Self {
        Self {
            method: SummarizationMethod::TfIdf { top_n: 3 },
            min_sentence_length: 10,
            max_sentence_length: 1000,
            stop_words: default_stop_words(),
        }
    }
}

/// Returns the default English stop-word list.
fn default_stop_words() -> Vec<String> {
    [
        "the", "a", "an", "is", "it", "in", "on", "at", "to", "of", "and", "or", "but", "for",
        "with", "this", "that", "are", "was", "were", "be", "been", "have", "has", "had", "do",
        "does", "did", "will", "would", "could", "should",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

// ── Output types ──────────────────────────────────────────────────────────────

/// A sentence together with its relevance scores.
#[derive(Debug, Clone)]
pub struct SentenceScore {
    /// Zero-based index of this sentence in the original text.
    pub sentence_index: usize,
    /// Raw text of the sentence.
    pub text: String,
    /// Final composite score used for ranking.
    pub score: f64,
    /// Per-method component scores (e.g. `"tfidf"`, `"textrank"`).
    pub method_scores: HashMap<String, f64>,
}

/// The result of a summarization run.
#[derive(Debug, Clone)]
pub struct TextSummaryResult {
    /// Number of sentences in the source text (after filtering by length).
    pub original_sentence_count: usize,
    /// Selected sentences, ordered as they appear in the source text.
    pub summary_sentences: Vec<SentenceScore>,
    /// Fraction of original sentences retained (0.0–1.0).
    pub compression_ratio: f64,
    /// Name of the method used (e.g. `"tfidf"`, `"textrank"`, `"lead"`, `"hybrid"`).
    pub method: String,
}

/// Usage statistics exposed by [`TextSummarizer::stats`].
#[derive(Debug, Clone)]
pub struct TextSummarizerStats {
    /// Number of documents added to the corpus via [`TextSummarizer::add_to_corpus`].
    pub documents_in_corpus: u32,
    /// Number of distinct terms in the IDF vocabulary.
    pub vocabulary_size: usize,
    /// Average number of sentences per document seen (across `summarize` calls).
    pub avg_sentences_per_doc: f64,
}

// ── Core summarizer ───────────────────────────────────────────────────────────

/// Extractive text summarizer combining TF-IDF and TextRank.
///
/// ```
/// use ipfrs_semantic::text_summarizer::{TextSummarizer, SummarizerConfig, SummarizationMethod};
///
/// let config = SummarizerConfig {
///     method: SummarizationMethod::TfIdf { top_n: 2 },
///     ..SummarizerConfig::default()
/// };
/// let mut summarizer = TextSummarizer::new(config);
/// let result = summarizer.summarize(
///     "The sky is blue. The ocean is also blue. Grass is green. Mountains are tall."
/// ).unwrap();
/// assert_eq!(result.summary_sentences.len(), 2);
/// ```
#[derive(Debug, Clone)]
pub struct TextSummarizer {
    /// Active configuration.
    pub config: SummarizerConfig,
    /// Corpus-level document frequencies: term → count of documents containing that term.
    pub document_frequencies: HashMap<String, u32>,
    /// Total number of documents added to the corpus.
    pub total_documents: u32,
    /// Number of `summarize` calls made.
    summarize_calls: u32,
    /// Total sentences seen across all `summarize` calls.
    total_sentences_seen: u64,
}

impl TextSummarizer {
    /// Create a new summarizer with the given configuration.
    pub fn new(config: SummarizerConfig) -> Self {
        Self {
            config,
            document_frequencies: HashMap::new(),
            total_documents: 0,
            summarize_calls: 0,
            total_sentences_seen: 0,
        }
    }

    // ── Public API ────────────────────────────────────────────────────────────

    /// Summarize `text` using the configured method.
    ///
    /// # Errors
    ///
    /// Returns [`SummarizerError::EmptyText`] when `text` is blank, or
    /// [`SummarizerError::InsufficientSentences`] when fewer sentences are
    /// found than the algorithm requires.
    pub fn summarize(&mut self, text: &str) -> Result<TextSummaryResult, SummarizerError> {
        if text.trim().is_empty() {
            return Err(SummarizerError::EmptyText);
        }

        self.validate_config()?;

        let sentences = self.split_sentences(text);
        let sentences = self.filter_by_length(sentences);
        let n = sentences.len();

        self.summarize_calls += 1;
        self.total_sentences_seen += n as u64;

        let top_n = self.config.method.top_n().unwrap_or(n);

        if n == 0 {
            return Err(SummarizerError::InsufficientSentences { min: 1, got: 0 });
        }

        // Tokenize every sentence once
        let tokens_per_sentence: Vec<Vec<String>> = sentences
            .iter()
            .map(|s| self.tokenize_sentence(s))
            .collect();

        let method_name = self.config.method.name().to_string();

        let scored = match &self.config.method.clone() {
            SummarizationMethod::TfIdf { top_n } => {
                self.score_tfidf(&sentences, &tokens_per_sentence, *top_n)?
            }
            SummarizationMethod::TextRank {
                top_n,
                damping,
                max_iter,
            } => self.score_textrank(
                &sentences,
                &tokens_per_sentence,
                *top_n,
                *damping,
                *max_iter,
            )?,
            SummarizationMethod::Lead { n_sentences } => {
                self.score_lead(&sentences, *n_sentences)?
            }
            SummarizationMethod::Hybrid {
                top_n,
                tfidf_weight,
                textrank_weight,
            } => self.score_hybrid(
                &sentences,
                &tokens_per_sentence,
                *top_n,
                *tfidf_weight,
                *textrank_weight,
            )?,
        };

        let compression_ratio = if n == 0 {
            0.0
        } else {
            scored.len() as f64 / n as f64
        };

        let _ = top_n; // already consumed

        Ok(TextSummaryResult {
            original_sentence_count: n,
            summary_sentences: scored,
            compression_ratio,
            method: method_name,
        })
    }

    /// Add `text` to the external IDF corpus.
    ///
    /// Each sentence in `text` is treated as a document for IDF purposes.
    /// Calling this before `summarize` improves IDF estimates, especially for
    /// domain-specific vocabulary.
    pub fn add_to_corpus(&mut self, text: &str) {
        let sentences = self.split_sentences(text);
        for sentence in &sentences {
            let tokens = self.tokenize_sentence(sentence);
            // Collect unique tokens per sentence (each sentence = one document)
            let mut seen = std::collections::HashSet::new();
            for token in tokens {
                if seen.insert(token.clone()) {
                    *self.document_frequencies.entry(token).or_insert(0) += 1;
                }
            }
            self.total_documents += 1;
        }
    }

    /// Return usage statistics.
    pub fn stats(&self) -> TextSummarizerStats {
        let avg_sentences_per_doc = if self.summarize_calls == 0 {
            0.0
        } else {
            self.total_sentences_seen as f64 / self.summarize_calls as f64
        };
        TextSummarizerStats {
            documents_in_corpus: self.total_documents,
            vocabulary_size: self.document_frequencies.len(),
            avg_sentences_per_doc,
        }
    }

    // ── Sentence splitting & tokenization ─────────────────────────────────────

    /// Split `text` into sentences on `.`, `!`, or `?` followed by whitespace or end of string.
    pub fn split_sentences(&self, text: &str) -> Vec<String> {
        let mut sentences = Vec::new();
        let mut current = String::new();

        let chars: Vec<char> = text.chars().collect();
        let len = chars.len();
        let mut i = 0;

        while i < len {
            let ch = chars[i];
            current.push(ch);

            if matches!(ch, '.' | '!' | '?') {
                // Look ahead: next char must be whitespace, or we are at end
                let at_end = i + 1 >= len;
                let next_is_space = chars.get(i + 1).map(|c| c.is_whitespace()).unwrap_or(false);

                if at_end || next_is_space {
                    let trimmed = current.trim().to_string();
                    if !trimmed.is_empty() {
                        sentences.push(trimmed);
                    }
                    current = String::new();
                    // Skip leading whitespace for the next sentence
                    i += 1;
                    while i < len && chars[i].is_whitespace() {
                        i += 1;
                    }
                    continue;
                }
            }
            i += 1;
        }

        // Any remaining text that did not end with a terminator
        let remaining = current.trim().to_string();
        if !remaining.is_empty() {
            sentences.push(remaining);
        }

        sentences
    }

    /// Tokenize a sentence: lowercase, keep alphanumeric chars, drop stop words.
    pub fn tokenize_sentence(&self, sentence: &str) -> Vec<String> {
        let stop_words: std::collections::HashSet<&str> =
            self.config.stop_words.iter().map(|s| s.as_str()).collect();

        sentence
            .split_whitespace()
            .flat_map(|word| {
                // Keep only alphanumeric characters within each word
                let cleaned: String = word
                    .chars()
                    .filter(|c| c.is_alphanumeric())
                    .collect::<String>()
                    .to_lowercase();
                if cleaned.is_empty() {
                    None
                } else {
                    Some(cleaned)
                }
            })
            .filter(|token| !stop_words.contains(token.as_str()))
            .collect()
    }

    // ── TF-IDF helpers ────────────────────────────────────────────────────────

    /// Compute the TF-IDF vector for one sentence relative to the given sentence corpus.
    ///
    /// IDF is smoothed: `ln((1 + N) / (1 + df)) + 1` where N is the number of sentences
    /// used as corpus. If the summarizer has an external corpus its frequencies are
    /// blended in (additive).
    pub fn tfidf_vector(
        &self,
        tokens: &[String],
        all_sentences_tokens: &[Vec<String>],
    ) -> HashMap<String, f64> {
        if tokens.is_empty() {
            return HashMap::new();
        }

        let n_docs = all_sentences_tokens.len() as f64;

        // Term frequency within this sentence
        let mut tf: HashMap<&str, f64> = HashMap::new();
        for token in tokens {
            *tf.entry(token.as_str()).or_insert(0.0) += 1.0;
        }
        let token_count = tokens.len() as f64;

        // Document frequency per term from the local corpus
        let mut df_local: HashMap<&str, u32> = HashMap::new();
        for sent_tokens in all_sentences_tokens {
            let mut seen = std::collections::HashSet::new();
            for token in sent_tokens {
                if seen.insert(token.as_str()) {
                    *df_local.entry(token.as_str()).or_insert(0) += 1;
                }
            }
        }

        let mut result = HashMap::new();
        for (term, &raw_tf) in &tf {
            let normalized_tf = raw_tf / token_count;

            let local_df = *df_local.get(term).unwrap_or(&0) as f64;
            let corpus_df = self.document_frequencies.get(*term).copied().unwrap_or(0) as f64;
            let corpus_n = self.total_documents as f64;

            // Blend local + external corpus
            let combined_df = local_df + corpus_df;
            let combined_n = n_docs + corpus_n;

            // Smoothed IDF
            let idf = ((1.0 + combined_n) / (1.0 + combined_df)).ln() + 1.0;

            result.insert(term.to_string(), normalized_tf * idf);
        }
        result
    }

    /// Cosine similarity between two TF-IDF vectors.
    ///
    /// Returns `0.0` when either vector is empty or has zero magnitude.
    pub fn cosine_similarity(a: &HashMap<String, f64>, b: &HashMap<String, f64>) -> f64 {
        if a.is_empty() || b.is_empty() {
            return 0.0;
        }

        let dot: f64 = a
            .iter()
            .filter_map(|(k, va)| b.get(k).map(|vb| va * vb))
            .sum();

        let norm_a: f64 = a.values().map(|v| v * v).sum::<f64>().sqrt();
        let norm_b: f64 = b.values().map(|v| v * v).sum::<f64>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot / (norm_a * norm_b)
        }
    }

    /// Run PageRank-style iteration on the sentence similarity matrix.
    ///
    /// Initialises scores uniformly and iterates until the maximum per-node delta
    /// is less than `1e-6` or `max_iter` is reached.
    pub fn textrank_scores(
        similarity_matrix: &[Vec<f64>],
        damping: f64,
        max_iter: u32,
    ) -> Vec<f64> {
        let n = similarity_matrix.len();
        if n == 0 {
            return Vec::new();
        }

        // Normalise each row so that outgoing weights sum to 1
        let mut transition: Vec<Vec<f64>> = similarity_matrix
            .iter()
            .map(|row| {
                let total: f64 = row.iter().sum();
                if total == 0.0 {
                    vec![1.0 / n as f64; n]
                } else {
                    row.iter().map(|v| v / total).collect()
                }
            })
            .collect();

        // Zero out self-loops
        for (i, row) in transition.iter_mut().enumerate() {
            row[i] = 0.0;
            // Re-normalise after zeroing self-loop
            let total: f64 = row.iter().sum();
            if total > 0.0 {
                for v in row.iter_mut() {
                    *v /= total;
                }
            } else {
                // Dangling node: distribute uniformly
                for v in row.iter_mut() {
                    *v = 1.0 / n as f64;
                }
            }
        }

        let mut scores = vec![1.0 / n as f64; n];

        for _ in 0..max_iter {
            let mut new_scores = vec![(1.0 - damping) / n as f64; n];
            for j in 0..n {
                // Sum contributions from all nodes pointing to j
                let incoming: f64 = (0..n).map(|i| transition[i][j] * scores[i]).sum();
                new_scores[j] += damping * incoming;
            }

            // Check convergence
            let max_delta = scores
                .iter()
                .zip(new_scores.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);

            scores = new_scores;
            if max_delta < 1e-6 {
                break;
            }
        }

        scores
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    fn validate_config(&self) -> Result<(), SummarizerError> {
        if self.config.min_sentence_length > self.config.max_sentence_length {
            return Err(SummarizerError::InvalidConfig(format!(
                "min_sentence_length ({}) must not exceed max_sentence_length ({})",
                self.config.min_sentence_length, self.config.max_sentence_length
            )));
        }
        match &self.config.method {
            SummarizationMethod::TextRank { damping, .. } if *damping <= 0.0 || *damping >= 1.0 => {
                return Err(SummarizerError::InvalidConfig(format!(
                    "TextRank damping must be in (0, 1), got {damping}"
                )));
            }
            SummarizationMethod::Hybrid {
                tfidf_weight,
                textrank_weight,
                ..
            } if *tfidf_weight < 0.0 || *textrank_weight < 0.0 => {
                return Err(SummarizerError::InvalidConfig(
                    "Hybrid weights must be non-negative".to_string(),
                ));
            }
            _ => {}
        }
        Ok(())
    }

    fn filter_by_length(&self, sentences: Vec<String>) -> Vec<String> {
        sentences
            .into_iter()
            .filter(|s| {
                s.len() >= self.config.min_sentence_length
                    && s.len() <= self.config.max_sentence_length
            })
            .collect()
    }

    /// Build TF-IDF vectors for all sentences.
    fn build_tfidf_vectors(
        &self,
        tokens_per_sentence: &[Vec<String>],
    ) -> Vec<HashMap<String, f64>> {
        tokens_per_sentence
            .iter()
            .map(|tokens| self.tfidf_vector(tokens, tokens_per_sentence))
            .collect()
    }

    /// Score each sentence by the sum of its TF-IDF weights.
    fn tfidf_sentence_scores(&self, tokens_per_sentence: &[Vec<String>]) -> Vec<f64> {
        let vectors = self.build_tfidf_vectors(tokens_per_sentence);
        vectors.iter().map(|v| v.values().sum::<f64>()).collect()
    }

    fn score_tfidf(
        &self,
        sentences: &[String],
        tokens_per_sentence: &[Vec<String>],
        top_n: usize,
    ) -> Result<Vec<SentenceScore>, SummarizerError> {
        let raw_scores = self.tfidf_sentence_scores(tokens_per_sentence);
        let scored = self.top_n_in_order(sentences, &raw_scores, top_n, "tfidf");
        Ok(scored)
    }

    fn score_textrank(
        &self,
        sentences: &[String],
        tokens_per_sentence: &[Vec<String>],
        top_n: usize,
        damping: f64,
        max_iter: u32,
    ) -> Result<Vec<SentenceScore>, SummarizerError> {
        let n = sentences.len();
        let vectors = self.build_tfidf_vectors(tokens_per_sentence);

        // Build N×N similarity matrix
        let mut matrix = vec![vec![0.0_f64; n]; n];
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    matrix[i][j] = Self::cosine_similarity(&vectors[i], &vectors[j]);
                }
            }
        }

        let tr_scores = Self::textrank_scores(&matrix, damping, max_iter);
        let scored = self.top_n_in_order(sentences, &tr_scores, top_n, "textrank");
        Ok(scored)
    }

    fn score_lead(
        &self,
        sentences: &[String],
        n_sentences: usize,
    ) -> Result<Vec<SentenceScore>, SummarizerError> {
        let take = n_sentences.min(sentences.len());
        let result = sentences[..take]
            .iter()
            .enumerate()
            .map(|(i, text)| {
                let mut method_scores = HashMap::new();
                method_scores.insert("lead".to_string(), 1.0);
                SentenceScore {
                    sentence_index: i,
                    text: text.clone(),
                    score: 1.0,
                    method_scores,
                }
            })
            .collect();
        Ok(result)
    }

    fn score_hybrid(
        &self,
        sentences: &[String],
        tokens_per_sentence: &[Vec<String>],
        top_n: usize,
        tfidf_weight: f64,
        textrank_weight: f64,
    ) -> Result<Vec<SentenceScore>, SummarizerError> {
        let n = sentences.len();
        let tfidf_scores = self.tfidf_sentence_scores(tokens_per_sentence);

        // TextRank component
        let vectors = self.build_tfidf_vectors(tokens_per_sentence);
        let mut matrix = vec![vec![0.0_f64; n]; n];
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    matrix[i][j] = Self::cosine_similarity(&vectors[i], &vectors[j]);
                }
            }
        }
        // Use default TextRank params for hybrid
        let tr_scores = Self::textrank_scores(&matrix, 0.85, 100);

        // Normalise each component to [0, 1] before blending
        let norm_tfidf = Self::normalise(&tfidf_scores);
        let norm_tr = Self::normalise(&tr_scores);

        let total_weight = tfidf_weight + textrank_weight;
        let combined: Vec<f64> = norm_tfidf
            .iter()
            .zip(norm_tr.iter())
            .map(|(tf, tr)| {
                if total_weight == 0.0 {
                    0.0
                } else {
                    (tfidf_weight * tf + textrank_weight * tr) / total_weight
                }
            })
            .collect();

        // Build scored output keeping per-method details
        let top_n_capped = top_n.min(n);
        let mut indexed: Vec<(usize, f64)> = combined.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.truncate(top_n_capped);
        indexed.sort_by_key(|&(i, _)| i);

        let result = indexed
            .into_iter()
            .map(|(i, score)| {
                let mut method_scores = HashMap::new();
                method_scores.insert("tfidf".to_string(), norm_tfidf[i]);
                method_scores.insert("textrank".to_string(), norm_tr[i]);
                SentenceScore {
                    sentence_index: i,
                    text: sentences[i].clone(),
                    score,
                    method_scores,
                }
            })
            .collect();

        Ok(result)
    }

    /// Select up to `top_n` highest-scoring sentence indices and return them in original order.
    fn top_n_in_order(
        &self,
        sentences: &[String],
        scores: &[f64],
        top_n: usize,
        method_name: &str,
    ) -> Vec<SentenceScore> {
        let n = sentences.len();
        let take = top_n.min(n);

        let mut indexed: Vec<(usize, f64)> = scores.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.truncate(take);
        indexed.sort_by_key(|&(i, _)| i); // restore original order

        indexed
            .into_iter()
            .map(|(i, score)| {
                let mut method_scores = HashMap::new();
                method_scores.insert(method_name.to_string(), score);
                SentenceScore {
                    sentence_index: i,
                    text: sentences[i].clone(),
                    score,
                    method_scores,
                }
            })
            .collect()
    }

    /// Linearly normalise `values` to [0, 1].  All-equal inputs map to 0.0.
    fn normalise(values: &[f64]) -> Vec<f64> {
        if values.is_empty() {
            return Vec::new();
        }
        let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = max - min;
        if range == 0.0 {
            return vec![0.0; values.len()];
        }
        values.iter().map(|v| (v - min) / range).collect()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use crate::text_summarizer::{
        SummarizationMethod, SummarizerConfig, SummarizerError, TextSummarizer,
    };
    use std::collections::HashMap;

    // Helper: build a summarizer with TF-IDF top_n
    fn tfidf_summarizer(top_n: usize) -> TextSummarizer {
        TextSummarizer::new(SummarizerConfig {
            method: SummarizationMethod::TfIdf { top_n },
            ..SummarizerConfig::default()
        })
    }

    fn textrank_summarizer(top_n: usize) -> TextSummarizer {
        TextSummarizer::new(SummarizerConfig {
            method: SummarizationMethod::TextRank {
                top_n,
                damping: 0.85,
                max_iter: 100,
            },
            ..SummarizerConfig::default()
        })
    }

    fn lead_summarizer(n: usize) -> TextSummarizer {
        TextSummarizer::new(SummarizerConfig {
            method: SummarizationMethod::Lead { n_sentences: n },
            ..SummarizerConfig::default()
        })
    }

    fn hybrid_summarizer(top_n: usize, tw: f64, rw: f64) -> TextSummarizer {
        TextSummarizer::new(SummarizerConfig {
            method: SummarizationMethod::Hybrid {
                top_n,
                tfidf_weight: tw,
                textrank_weight: rw,
            },
            ..SummarizerConfig::default()
        })
    }

    const SAMPLE: &str = "The quick brown fox jumps over the lazy dog. \
         Artificial intelligence is transforming many industries. \
         Machine learning enables computers to learn from data. \
         The weather today is sunny and warm. \
         Deep learning models require large amounts of training data.";

    // ── 1. split_sentences ────────────────────────────────────────────────────

    #[test]
    fn test_split_sentences_basic() {
        let s = TextSummarizer::new(SummarizerConfig::default());
        let sents = s.split_sentences("Hello world. Foo bar! Baz qux?");
        assert_eq!(sents.len(), 3);
    }

    #[test]
    fn test_split_sentences_empty_string() {
        let s = TextSummarizer::new(SummarizerConfig::default());
        let sents = s.split_sentences("");
        assert!(sents.is_empty());
    }

    #[test]
    fn test_split_sentences_no_terminator() {
        let s = TextSummarizer::new(SummarizerConfig::default());
        let sents = s.split_sentences("No terminator here");
        assert_eq!(sents.len(), 1);
        assert_eq!(sents[0], "No terminator here");
    }

    #[test]
    fn test_split_sentences_multiple_spaces() {
        let s = TextSummarizer::new(SummarizerConfig::default());
        let sents = s.split_sentences("Hello.   World.");
        assert_eq!(sents.len(), 2);
    }

    #[test]
    fn test_split_sentences_trims_whitespace() {
        let s = TextSummarizer::new(SummarizerConfig::default());
        let sents = s.split_sentences("  Leading spaces.  Trailing spaces.  ");
        assert!(sents.iter().all(|s| s == s.trim()));
    }

    // ── 2. tokenize_sentence ──────────────────────────────────────────────────

    #[test]
    fn test_tokenize_lowercases() {
        let s = TextSummarizer::new(SummarizerConfig::default());
        let tokens = s.tokenize_sentence("Hello WORLD");
        assert!(tokens.contains(&"hello".to_string()));
        assert!(tokens.contains(&"world".to_string()));
    }

    #[test]
    fn test_tokenize_removes_stop_words() {
        let s = TextSummarizer::new(SummarizerConfig::default());
        let tokens = s.tokenize_sentence("the quick brown fox");
        assert!(!tokens.contains(&"the".to_string()));
        assert!(tokens.contains(&"quick".to_string()));
    }

    #[test]
    fn test_tokenize_removes_punctuation() {
        let s = TextSummarizer::new(SummarizerConfig::default());
        let tokens = s.tokenize_sentence("Hello, world!");
        // punctuation stripped from tokens
        assert!(tokens
            .iter()
            .all(|t| t.chars().all(|c| c.is_alphanumeric())));
    }

    #[test]
    fn test_tokenize_empty_sentence() {
        let s = TextSummarizer::new(SummarizerConfig::default());
        let tokens = s.tokenize_sentence("");
        assert!(tokens.is_empty());
    }

    #[test]
    fn test_tokenize_all_stop_words() {
        let s = TextSummarizer::new(SummarizerConfig::default());
        let tokens = s.tokenize_sentence("the a an is it in on at");
        assert!(tokens.is_empty());
    }

    // ── 3. cosine_similarity ──────────────────────────────────────────────────

    #[test]
    fn test_cosine_identical_vectors() {
        let mut v: HashMap<String, f64> = HashMap::new();
        v.insert("foo".to_string(), 1.0);
        v.insert("bar".to_string(), 2.0);
        let sim = TextSummarizer::cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_cosine_orthogonal_vectors() {
        let mut a: HashMap<String, f64> = HashMap::new();
        a.insert("foo".to_string(), 1.0);
        let mut b: HashMap<String, f64> = HashMap::new();
        b.insert("bar".to_string(), 1.0);
        let sim = TextSummarizer::cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-9);
    }

    #[test]
    fn test_cosine_empty_vector() {
        let a: HashMap<String, f64> = HashMap::new();
        let mut b: HashMap<String, f64> = HashMap::new();
        b.insert("foo".to_string(), 1.0);
        assert_eq!(TextSummarizer::cosine_similarity(&a, &b), 0.0);
        assert_eq!(TextSummarizer::cosine_similarity(&b, &a), 0.0);
    }

    #[test]
    fn test_cosine_partial_overlap() {
        let mut a: HashMap<String, f64> = HashMap::new();
        a.insert("foo".to_string(), 1.0);
        a.insert("bar".to_string(), 1.0);
        let mut b: HashMap<String, f64> = HashMap::new();
        b.insert("foo".to_string(), 1.0);
        b.insert("baz".to_string(), 1.0);
        let sim = TextSummarizer::cosine_similarity(&a, &b);
        assert!(sim > 0.0 && sim < 1.0);
    }

    // ── 4. tfidf_vector ───────────────────────────────────────────────────────

    #[test]
    fn test_tfidf_vector_non_empty() {
        let s = TextSummarizer::new(SummarizerConfig::default());
        let tokens = vec!["machine".to_string(), "learning".to_string()];
        let corpus = vec![
            tokens.clone(),
            vec!["deep".to_string(), "learning".to_string()],
        ];
        let vec = s.tfidf_vector(&tokens, &corpus);
        assert!(!vec.is_empty());
    }

    #[test]
    fn test_tfidf_vector_empty_tokens() {
        let s = TextSummarizer::new(SummarizerConfig::default());
        let vec = s.tfidf_vector(&[], &[]);
        assert!(vec.is_empty());
    }

    #[test]
    fn test_tfidf_rare_term_higher_idf() {
        let s = TextSummarizer::new(SummarizerConfig::default());
        let rare = vec!["uniqueterm".to_string()];
        let common = vec!["shared".to_string()];
        let corpus = vec![rare.clone(), common.clone(), common.clone(), common.clone()];
        let rare_vec = s.tfidf_vector(&rare, &corpus);
        let common_vec = s.tfidf_vector(&common, &corpus);
        let rare_score = rare_vec.values().sum::<f64>();
        let common_score = common_vec.values().sum::<f64>();
        // Rare term should have higher IDF, but tf is equal, so rare_score >= common_score
        assert!(rare_score >= common_score);
    }

    // ── 5. textrank_scores ────────────────────────────────────────────────────

    #[test]
    fn test_textrank_scores_uniform_matrix() {
        // All sentences equally similar to each other → uniform scores
        let n = 4;
        let sim = vec![vec![1.0; n]; n];
        let scores = TextSummarizer::textrank_scores(&sim, 0.85, 200);
        assert_eq!(scores.len(), n);
        let expected = 1.0 / n as f64;
        for &s in &scores {
            assert!((s - expected).abs() < 1e-3, "score {s} vs {expected}");
        }
    }

    #[test]
    fn test_textrank_scores_empty_matrix() {
        let scores = TextSummarizer::textrank_scores(&[], 0.85, 100);
        assert!(scores.is_empty());
    }

    #[test]
    fn test_textrank_scores_single_sentence() {
        let sim = vec![vec![0.0]];
        let scores = TextSummarizer::textrank_scores(&sim, 0.85, 100);
        assert_eq!(scores.len(), 1);
    }

    #[test]
    fn test_textrank_scores_convergence() {
        let _n = 3;
        let sim = vec![
            vec![0.0, 0.8, 0.2],
            vec![0.8, 0.0, 0.5],
            vec![0.2, 0.5, 0.0],
        ];
        let scores_100 = TextSummarizer::textrank_scores(&sim, 0.85, 100);
        let scores_1000 = TextSummarizer::textrank_scores(&sim, 0.85, 1000);
        // Scores should be close after 100 iterations
        for (a, b) in scores_100.iter().zip(scores_1000.iter()) {
            assert!((a - b).abs() < 1e-4);
        }
    }

    // ── 6. summarize — error cases ────────────────────────────────────────────

    #[test]
    fn test_summarize_empty_text_error() {
        let mut s = tfidf_summarizer(2);
        let err = s
            .summarize("")
            .expect_err("test: empty string should return an error");
        assert_eq!(err, SummarizerError::EmptyText);
    }

    #[test]
    fn test_summarize_whitespace_only_error() {
        let mut s = tfidf_summarizer(2);
        let err = s
            .summarize("   \n\t  ")
            .expect_err("test: whitespace-only string should return an error");
        assert_eq!(err, SummarizerError::EmptyText);
    }

    #[test]
    fn test_summarize_invalid_config_length_bounds() {
        let cfg = SummarizerConfig {
            method: SummarizationMethod::TfIdf { top_n: 2 },
            min_sentence_length: 500,
            max_sentence_length: 10,
            stop_words: vec![],
        };
        let mut s = TextSummarizer::new(cfg);
        let err = s
            .summarize("Hello world. Foo bar.")
            .expect_err("test: invalid config (min > max sentence length) should return an error");
        matches!(err, SummarizerError::InvalidConfig(_));
    }

    #[test]
    fn test_summarize_invalid_textrank_damping() {
        let cfg = SummarizerConfig {
            method: SummarizationMethod::TextRank {
                top_n: 2,
                damping: 1.5,
                max_iter: 100,
            },
            ..SummarizerConfig::default()
        };
        let mut s = TextSummarizer::new(cfg);
        let err = s
            .summarize(SAMPLE)
            .expect_err("test: invalid damping factor should return an error");
        matches!(err, SummarizerError::InvalidConfig(_));
    }

    // ── 7. summarize — TF-IDF ─────────────────────────────────────────────────

    #[test]
    fn test_tfidf_returns_correct_count() {
        let mut s = tfidf_summarizer(2);
        let result = s
            .summarize(SAMPLE)
            .expect("test: TF-IDF summarize on valid SAMPLE should succeed");
        assert_eq!(result.summary_sentences.len(), 2);
    }

    #[test]
    fn test_tfidf_preserves_original_order() {
        let mut s = tfidf_summarizer(3);
        let result = s
            .summarize(SAMPLE)
            .expect("test: TF-IDF summarize on valid SAMPLE should succeed");
        let indices: Vec<usize> = result
            .summary_sentences
            .iter()
            .map(|ss| ss.sentence_index)
            .collect();
        let mut sorted = indices.clone();
        sorted.sort_unstable();
        assert_eq!(indices, sorted);
    }

    #[test]
    fn test_tfidf_method_name() {
        let mut s = tfidf_summarizer(2);
        let result = s
            .summarize(SAMPLE)
            .expect("test: TF-IDF summarize on valid SAMPLE should succeed");
        assert_eq!(result.method, "tfidf");
    }

    #[test]
    fn test_tfidf_compression_ratio() {
        let mut s = tfidf_summarizer(2);
        let result = s
            .summarize(SAMPLE)
            .expect("test: TF-IDF summarize on valid SAMPLE should succeed");
        assert!(result.compression_ratio > 0.0 && result.compression_ratio <= 1.0);
    }

    #[test]
    fn test_tfidf_top_n_capped_at_sentence_count() {
        let mut s = tfidf_summarizer(100);
        let result = s
            .summarize("Only two sentences here. Second one follows.")
            .expect("test: TF-IDF summarize with top_n larger than sentence count should succeed");
        assert!(result.summary_sentences.len() <= result.original_sentence_count);
    }

    // ── 8. summarize — TextRank ───────────────────────────────────────────────

    #[test]
    fn test_textrank_returns_correct_count() {
        let mut s = textrank_summarizer(2);
        let result = s
            .summarize(SAMPLE)
            .expect("test: TextRank summarize on valid SAMPLE should succeed");
        assert_eq!(result.summary_sentences.len(), 2);
    }

    #[test]
    fn test_textrank_preserves_original_order() {
        let mut s = textrank_summarizer(3);
        let result = s
            .summarize(SAMPLE)
            .expect("test: TextRank summarize on valid SAMPLE should succeed");
        let indices: Vec<usize> = result
            .summary_sentences
            .iter()
            .map(|ss| ss.sentence_index)
            .collect();
        let mut sorted = indices.clone();
        sorted.sort_unstable();
        assert_eq!(indices, sorted);
    }

    #[test]
    fn test_textrank_method_name() {
        let mut s = textrank_summarizer(2);
        let result = s
            .summarize(SAMPLE)
            .expect("test: TextRank summarize on valid SAMPLE should succeed");
        assert_eq!(result.method, "textrank");
    }

    #[test]
    fn test_textrank_scores_are_non_negative() {
        let mut s = textrank_summarizer(3);
        let result = s
            .summarize(SAMPLE)
            .expect("test: TextRank summarize on valid SAMPLE should succeed");
        for ss in &result.summary_sentences {
            assert!(ss.score >= 0.0);
        }
    }

    // ── 9. summarize — Lead ───────────────────────────────────────────────────

    #[test]
    fn test_lead_returns_first_n() {
        let mut s = lead_summarizer(2);
        let result = s
            .summarize(SAMPLE)
            .expect("test: Lead summarize on valid SAMPLE should succeed");
        assert_eq!(result.summary_sentences.len(), 2);
        assert_eq!(result.summary_sentences[0].sentence_index, 0);
        assert_eq!(result.summary_sentences[1].sentence_index, 1);
    }

    #[test]
    fn test_lead_method_name() {
        let mut s = lead_summarizer(2);
        let result = s
            .summarize(SAMPLE)
            .expect("test: Lead summarize on valid SAMPLE should succeed");
        assert_eq!(result.method, "lead");
    }

    #[test]
    fn test_lead_capped_at_available_sentences() {
        let mut s = lead_summarizer(100);
        // Sentences long enough to survive the min_sentence_length=10 filter
        let result = s
            .summarize("First sentence here. Second sentence here. Third sentence here.")
            .expect("test: Lead summarize with top_n larger than sentence count should succeed");
        assert!(result.summary_sentences.len() <= 3);
    }

    // ── 10. summarize — Hybrid ────────────────────────────────────────────────

    #[test]
    fn test_hybrid_returns_correct_count() {
        let mut s = hybrid_summarizer(2, 0.5, 0.5);
        let result = s
            .summarize(SAMPLE)
            .expect("test: Hybrid summarize on valid SAMPLE should succeed");
        assert_eq!(result.summary_sentences.len(), 2);
    }

    #[test]
    fn test_hybrid_method_name() {
        let mut s = hybrid_summarizer(2, 0.5, 0.5);
        let result = s
            .summarize(SAMPLE)
            .expect("test: Hybrid summarize on valid SAMPLE should succeed");
        assert_eq!(result.method, "hybrid");
    }

    #[test]
    fn test_hybrid_method_scores_contain_both_keys() {
        let mut s = hybrid_summarizer(2, 0.5, 0.5);
        let result = s
            .summarize(SAMPLE)
            .expect("test: summarize SAMPLE for hybrid method_scores keys");
        for ss in &result.summary_sentences {
            assert!(ss.method_scores.contains_key("tfidf"));
            assert!(ss.method_scores.contains_key("textrank"));
        }
    }

    #[test]
    fn test_hybrid_preserves_original_order() {
        let mut s = hybrid_summarizer(3, 0.6, 0.4);
        let result = s
            .summarize(SAMPLE)
            .expect("test: summarize SAMPLE for hybrid sentence order");
        let indices: Vec<usize> = result
            .summary_sentences
            .iter()
            .map(|ss| ss.sentence_index)
            .collect();
        let mut sorted = indices.clone();
        sorted.sort_unstable();
        assert_eq!(indices, sorted);
    }

    // ── 11. add_to_corpus ─────────────────────────────────────────────────────

    #[test]
    fn test_add_to_corpus_increases_vocab() {
        let mut s = tfidf_summarizer(2);
        assert_eq!(s.document_frequencies.len(), 0);
        s.add_to_corpus("Machine learning is powerful. Deep learning too.");
        assert!(!s.document_frequencies.is_empty());
    }

    #[test]
    fn test_add_to_corpus_increases_total_documents() {
        let mut s = tfidf_summarizer(2);
        s.add_to_corpus("First sentence. Second sentence.");
        assert!(s.total_documents >= 1);
    }

    #[test]
    fn test_corpus_influences_idf() {
        // With a corpus, terms appearing in many corpus docs should have lower IDF
        let mut s = tfidf_summarizer(2);
        // Add "common" word many times
        for _ in 0..10 {
            s.add_to_corpus("common word appears everywhere.");
        }
        let tokens_common = vec!["common".to_string()];
        let tokens_rare = vec!["xyzrare".to_string()];
        let corpus_local = vec![tokens_common.clone(), tokens_rare.clone()];
        let v_common = s.tfidf_vector(&tokens_common, &corpus_local);
        let v_rare = s.tfidf_vector(&tokens_rare, &corpus_local);
        let score_common: f64 = v_common.values().sum();
        let score_rare: f64 = v_rare.values().sum();
        assert!(score_rare > score_common);
    }

    // ── 12. stats ─────────────────────────────────────────────────────────────

    #[test]
    fn test_stats_initial_state() {
        let s = tfidf_summarizer(2);
        let stats = s.stats();
        assert_eq!(stats.documents_in_corpus, 0);
        assert_eq!(stats.vocabulary_size, 0);
        assert_eq!(stats.avg_sentences_per_doc, 0.0);
    }

    #[test]
    fn test_stats_after_summarize() {
        let mut s = tfidf_summarizer(2);
        s.summarize(SAMPLE)
            .expect("test: summarize SAMPLE to update stats");
        let stats = s.stats();
        assert!(stats.avg_sentences_per_doc > 0.0);
    }

    #[test]
    fn test_stats_after_corpus() {
        let mut s = tfidf_summarizer(2);
        s.add_to_corpus(SAMPLE);
        let stats = s.stats();
        assert!(stats.vocabulary_size > 0);
        assert!(stats.documents_in_corpus > 0);
    }

    // ── 13. SentenceScore fields ──────────────────────────────────────────────

    #[test]
    fn test_sentence_score_text_matches_original() {
        let mut s = tfidf_summarizer(2);
        let result = s
            .summarize(SAMPLE)
            .expect("test: summarize SAMPLE to access summary sentences");
        let original_sentences = s.split_sentences(SAMPLE);
        for ss in &result.summary_sentences {
            let orig = &original_sentences[ss.sentence_index];
            // Text should match (modulo length filtering)
            assert_eq!(&ss.text, orig);
        }
    }

    #[test]
    fn test_sentence_score_has_tfidf_method_score() {
        let mut s = tfidf_summarizer(2);
        let result = s
            .summarize(SAMPLE)
            .expect("test: summarize SAMPLE to check tfidf method score");
        for ss in &result.summary_sentences {
            assert!(ss.method_scores.contains_key("tfidf"));
        }
    }

    // ── 14. edge cases ────────────────────────────────────────────────────────

    #[test]
    fn test_single_sentence_tfidf() {
        let mut s = tfidf_summarizer(1);
        let result = s
            .summarize("Just one sentence here with content words.")
            .expect("test: summarize single sentence with tfidf");
        assert_eq!(result.summary_sentences.len(), 1);
    }

    #[test]
    fn test_single_sentence_textrank() {
        let mut s = textrank_summarizer(1);
        let result = s
            .summarize("Just one sentence here with content words.")
            .expect("test: summarize single sentence with textrank");
        assert_eq!(result.summary_sentences.len(), 1);
    }

    #[test]
    fn test_min_sentence_length_filter() {
        let cfg = SummarizerConfig {
            method: SummarizationMethod::TfIdf { top_n: 5 },
            min_sentence_length: 50,
            max_sentence_length: 1000,
            stop_words: vec![],
        };
        let mut s = TextSummarizer::new(cfg);
        // Short sentences should be filtered out
        let long =
            "This is a much longer sentence with plenty of content words to pass the filter.";
        let text = format!("Hi. Bye. {long}");
        let result = s
            .summarize(&text)
            .expect("test: summarize text with min_sentence_length filter");
        // Only the long sentence should survive
        assert!(result.original_sentence_count <= 1);
    }

    #[test]
    fn test_compression_ratio_never_exceeds_one() {
        let mut s = tfidf_summarizer(10);
        let result = s
            .summarize(SAMPLE)
            .expect("test: summarize SAMPLE for compression ratio check");
        assert!(result.compression_ratio <= 1.0);
    }

    #[test]
    fn test_summarize_increases_call_count() {
        let mut s = tfidf_summarizer(2);
        s.summarize(SAMPLE)
            .expect("test: first summarize call for stats check");
        s.summarize(SAMPLE)
            .expect("test: second summarize call for stats check");
        let stats = s.stats();
        // avg_sentences_per_doc should reflect 2 calls
        assert!(stats.avg_sentences_per_doc > 0.0);
    }
}
