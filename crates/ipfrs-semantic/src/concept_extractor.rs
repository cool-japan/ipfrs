//! # Concept and Keyword Extraction
//!
//! Provides TF-IDF-based concept and keyword extraction from text, supporting
//! n-gram phrases, named entity detection, technical term identification, and
//! corpus-level IDF scoring for multi-document analysis.
//!
//! ## Overview
//!
//! - **TF-IDF scoring** with configurable smoothing
//! - **N-gram phrase extraction** (bigrams, trigrams, etc.)
//! - **Entity detection** (capitalized / all-caps terms)
//! - **Technical term detection** (camelCase, snake_case)
//! - **Stop-word filtering** with configurable word list
//! - **Multi-document corpus statistics** for accurate IDF

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// ConceptType
// ---------------------------------------------------------------------------

/// Classifies the semantic role of an extracted concept.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConceptType {
    /// A plain keyword term.
    Keyword,
    /// A multi-word phrase (n-gram, n ≥ 2).
    Phrase,
    /// A named entity — starts with an upper-case letter or is fully capitalised.
    Entity,
    /// A technical identifier — camelCase or snake_case.
    Technical,
}

// ---------------------------------------------------------------------------
// Concept
// ---------------------------------------------------------------------------

/// A single extracted concept with scoring metadata.
#[derive(Debug, Clone)]
pub struct Concept {
    /// Normalised surface form of the concept.
    pub term: String,
    /// Combined TF-IDF score.
    pub score: f64,
    /// Raw count of the term in the current document.
    pub frequency: u32,
    /// Number of corpus documents that contain this term.
    pub doc_frequency: u32,
    /// Semantic category of the concept.
    pub concept_type: ConceptType,
}

// ---------------------------------------------------------------------------
// ExtractorConfig
// ---------------------------------------------------------------------------

/// Configuration for [`ConceptExtractor`].
#[derive(Debug, Clone)]
pub struct ExtractorConfig {
    /// Maximum number of concepts returned per document.
    pub max_concepts: usize,
    /// Minimum raw term frequency required for inclusion.
    pub min_term_frequency: u32,
    /// Minimum character length for a single token to be kept.
    pub min_term_length: usize,
    /// Maximum n for n-gram phrase extraction (1 = unigrams only).
    pub max_ngram: usize,
    /// Stop words to filter before scoring.
    pub stop_words: Vec<String>,
    /// Additive smoothing constant applied to IDF calculation.
    pub idf_smoothing: f64,
}

impl Default for ExtractorConfig {
    fn default() -> Self {
        Self {
            max_concepts: 20,
            min_term_frequency: 1,
            min_term_length: 3,
            max_ngram: 3,
            stop_words: default_stop_words(),
            idf_smoothing: 1.0,
        }
    }
}

// ---------------------------------------------------------------------------
// ExtractorStats
// ---------------------------------------------------------------------------

/// Cumulative statistics collected by [`ConceptExtractor`].
#[derive(Debug, Clone, Default)]
pub struct ExtractorStats {
    /// Total number of documents processed since construction.
    pub documents_processed: u64,
    /// Total concepts extracted across all documents.
    pub total_concepts_extracted: u64,
    /// Rolling average of concepts extracted per document.
    pub avg_concepts_per_doc: f64,
}

// ---------------------------------------------------------------------------
// ConceptExtractor
// ---------------------------------------------------------------------------

/// Extracts concepts and keywords from text using TF-IDF and frequency analysis.
///
/// The extractor maintains corpus-level document-frequency statistics so that
/// IDF weights improve as more documents are processed.
///
/// # Example
///
/// ```rust
/// use ipfrs_semantic::concept_extractor::{ConceptExtractor, ExtractorConfig};
///
/// let config = ExtractorConfig::default();
/// let mut extractor = ConceptExtractor::new(config);
///
/// let concepts = extractor.extract("The quick brown fox jumps over the lazy dog.");
/// for c in &concepts {
///     println!("{:?}: score={:.4}", c.term, c.score);
/// }
/// ```
pub struct ConceptExtractor {
    config: ExtractorConfig,
    /// Per-term document frequency across all previously seen documents.
    corpus_stats: HashMap<String, u32>,
    /// Total documents ingested (including the current one during `extract`).
    doc_count: usize,
    stats: ExtractorStats,
}

impl ConceptExtractor {
    /// Creates a new extractor with the given configuration.
    pub fn new(config: ExtractorConfig) -> Self {
        Self {
            config,
            corpus_stats: HashMap::new(),
            doc_count: 0,
            stats: ExtractorStats::default(),
        }
    }

    // ------------------------------------------------------------------
    // Public API
    // ------------------------------------------------------------------

    /// Extracts and returns the top concepts from `text`, updating internal
    /// corpus statistics so subsequent calls benefit from improved IDF weights.
    pub fn extract(&mut self, text: &str) -> Vec<Concept> {
        // 1. Tokenize preserving original casing for entity/technical detection.
        let raw_tokens: Vec<String> = Self::tokenize_raw(text);
        // 2. Lower-cased tokens for TF / stop-word / IDF lookups.
        let lc_tokens: Vec<String> = raw_tokens.iter().map(|t| t.to_lowercase()).collect();

        // 3. Update corpus with this document's unique terms (unigrams only for IDF).
        self.update_corpus(&lc_tokens);
        self.doc_count += 1;

        // 4. Compute unigram TF.
        let tf_map = Self::compute_tf(&lc_tokens);

        // 5. Build candidate concepts across all n-gram sizes.
        let mut concepts: Vec<Concept> = Vec::new();
        let doc_len = lc_tokens.len().max(1);

        // Unigrams.
        for (term, &tf) in &tf_map {
            if self.is_stop_word(term) {
                continue;
            }
            if tf < self.config.min_term_frequency {
                continue;
            }
            if term.len() < self.config.min_term_length {
                continue;
            }
            let score = self.compute_tfidf(term, tf, doc_len);
            let doc_frequency = self.corpus_stats.get(term).copied().unwrap_or(1);
            let concept_type = detect_concept_type_from_raw(term, &raw_tokens, &lc_tokens);
            concepts.push(Concept {
                term: term.clone(),
                score,
                frequency: tf,
                doc_frequency,
                concept_type,
            });
        }

        // N-grams (n = 2 .. max_ngram).
        for n in 2..=self.config.max_ngram {
            let ngrams = Self::extract_ngrams(&lc_tokens, n);
            // TF for n-grams.
            let ngram_tf = Self::compute_tf(&ngrams);
            let raw_ngrams = Self::extract_ngrams(&raw_tokens, n);
            let raw_ngram_tf = Self::compute_tf(&raw_ngrams);

            for (ngram, &tf) in &ngram_tf {
                if tf < self.config.min_term_frequency {
                    continue;
                }
                // Filter n-grams that are entirely stop words.
                let parts: Vec<&str> = ngram.split(' ').collect();
                let all_stop = parts.iter().all(|p| self.is_stop_word(p));
                if all_stop {
                    continue;
                }
                let score = self.compute_tfidf(ngram, tf, doc_len);
                // Use doc_frequency of the full phrase from corpus (may be 0 if
                // unseen — treat as 1 for IDF stability).
                let doc_frequency = self.corpus_stats.get(ngram).copied().unwrap_or(1);
                // For phrase concept type, check if any raw variant is entity/technical.
                let raw_phrase = raw_ngram_tf
                    .keys()
                    .find(|k| k.to_lowercase() == *ngram)
                    .cloned()
                    .unwrap_or_else(|| ngram.clone());
                let concept_type = detect_phrase_type(&raw_phrase);
                concepts.push(Concept {
                    term: ngram.clone(),
                    score,
                    frequency: tf,
                    doc_frequency,
                    concept_type,
                });
            }
        }

        // 6. Sort by score descending, deduplicate by term, take top N.
        let result = Self::top_concepts(&mut concepts, self.config.max_concepts);

        // 7. Update statistics.
        let extracted = result.len() as u64;
        self.stats.documents_processed += 1;
        self.stats.total_concepts_extracted += extracted;
        self.stats.avg_concepts_per_doc =
            self.stats.total_concepts_extracted as f64 / self.stats.documents_processed as f64;

        result
    }

    /// Tokenizes `text` into lower-cased tokens, splitting on whitespace and
    /// ASCII punctuation, dropping tokens shorter than `min_term_length`.
    ///
    /// This is the public lower-cased variant used for TF/IDF computation.
    pub fn tokenize(text: &str) -> Vec<String> {
        Self::tokenize_raw(text)
            .into_iter()
            .map(|t| t.to_lowercase())
            .collect()
    }

    /// Computes term frequency (raw count) for each token in `tokens`.
    pub fn compute_tf(tokens: &[String]) -> HashMap<String, u32> {
        let mut map: HashMap<String, u32> = HashMap::new();
        for tok in tokens {
            *map.entry(tok.clone()).or_insert(0) += 1;
        }
        map
    }

    /// Returns the TF-IDF score for a term given its raw frequency `tf` and
    /// the document length `doc_len`.
    ///
    /// Uses augmented TF (0.5 + 0.5 × tf / max_tf) to prevent bias towards
    /// long documents, combined with smoothed IDF.
    pub fn compute_tfidf(&self, term: &str, tf: u32, doc_len: usize) -> f64 {
        let max_tf = (doc_len as f64).max(1.0);
        // Augmented TF normalisation.
        let tf_norm = 0.5 + 0.5 * (tf as f64 / max_tf);
        // Smoothed IDF: log((N + smooth) / (df + smooth)) + 1
        let n = (self.doc_count as f64).max(1.0);
        let df = self.corpus_stats.get(term).copied().unwrap_or(0) as f64;
        let smooth = self.config.idf_smoothing;
        let idf = ((n + smooth) / (df + smooth)).ln() + 1.0;
        tf_norm * idf
    }

    /// Extracts all n-grams of size `n` from `tokens`, joining tokens with a
    /// single space to form the phrase string.
    pub fn extract_ngrams(tokens: &[String], n: usize) -> Vec<String> {
        if n == 0 || tokens.len() < n {
            return Vec::new();
        }
        tokens.windows(n).map(|w| w.join(" ")).collect()
    }

    /// Classifies a term into its [`ConceptType`].
    ///
    /// - **Entity** — starts with an ASCII upper-case letter *or* every ASCII
    ///   letter in the term is upper-case (acronym).
    /// - **Technical** — contains `_` (snake_case) or an interior upper-case
    ///   letter (camelCase).
    /// - **Phrase** — contains a space (multi-word).
    /// - **Keyword** — everything else.
    pub fn detect_concept_type(term: &str) -> ConceptType {
        if term.contains(' ') {
            return ConceptType::Phrase;
        }
        // All-caps acronym (e.g. "API", "HTTP").
        let ascii_letters: Vec<char> = term.chars().filter(|c| c.is_ascii_alphabetic()).collect();
        if !ascii_letters.is_empty() && ascii_letters.iter().all(|c| c.is_ascii_uppercase()) {
            return ConceptType::Entity;
        }
        // Starts with upper-case → named entity.
        if term
            .chars()
            .next()
            .map(|c| c.is_ascii_uppercase())
            .unwrap_or(false)
        {
            return ConceptType::Entity;
        }
        // snake_case.
        if term.contains('_') {
            return ConceptType::Technical;
        }
        // camelCase — interior uppercase letter.
        let mut chars = term.chars();
        // Skip the first character.
        let _ = chars.next();
        if chars.any(|c| c.is_ascii_uppercase()) {
            return ConceptType::Technical;
        }
        ConceptType::Keyword
    }

    /// Returns `true` if `word` appears in the configured stop-word list
    /// (case-insensitive comparison).
    pub fn is_stop_word(&self, word: &str) -> bool {
        let lower = word.to_lowercase();
        self.config.stop_words.iter().any(|sw| sw == &lower)
    }

    /// Updates corpus document-frequency statistics for the unique terms found
    /// in `tokens`.  Each unique term in the token slice counts as appearing
    /// in one additional document.
    pub fn update_corpus(&mut self, tokens: &[String]) {
        let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();
        for tok in tokens {
            if seen.insert(tok.as_str()) {
                *self.corpus_stats.entry(tok.clone()).or_insert(0) += 1;
            }
        }
    }

    /// Returns the top `n` concepts from `concepts` sorted by descending score.
    /// Deduplicates by term (keeps the highest-scoring entry).
    pub fn top_concepts(concepts: &mut Vec<Concept>, n: usize) -> Vec<Concept> {
        // Deduplicate: keep highest score per term.
        let mut best: HashMap<String, Concept> = HashMap::new();
        for c in concepts.drain(..) {
            let entry = best.entry(c.term.clone()).or_insert_with(|| c.clone());
            if c.score > entry.score {
                *entry = c;
            }
        }
        let mut sorted: Vec<Concept> = best.into_values().collect();
        sorted.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted.truncate(n);
        sorted
    }

    /// Returns a reference to the cumulative extraction statistics.
    pub fn stats(&self) -> &ExtractorStats {
        &self.stats
    }

    // ------------------------------------------------------------------
    // Private helpers
    // ------------------------------------------------------------------

    /// Tokenizes `text` preserving original casing for entity/technical detection.
    fn tokenize_raw(text: &str) -> Vec<String> {
        // Split on whitespace, then strip leading/trailing ASCII punctuation from
        // each token (commas, periods, brackets, quotes, etc.).
        text.split_whitespace()
            .flat_map(|word| {
                // Split further on common punctuation that may be embedded
                // (e.g. "foo/bar", "key:value", "term(s)").
                split_on_punctuation(word)
            })
            .filter(|t| !t.is_empty())
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Module-level helpers (not pub — internal implementation detail)
// ---------------------------------------------------------------------------

/// Splits a word on embedded punctuation characters that are not part of
/// identifiers (slash, colon, parentheses, brackets, etc.) while preserving
/// underscores and hyphens because they are meaningful in technical terms.
fn split_on_punctuation(word: &str) -> Vec<String> {
    // Strip wrapping punctuation first.
    let trimmed = word.trim_matches(|c: char| {
        matches!(
            c,
            '.' | ','
                | '!'
                | '?'
                | ';'
                | ':'
                | '"'
                | '\''
                | '('
                | ')'
                | '['
                | ']'
                | '{'
                | '}'
                | '<'
                | '>'
                | '/'
                | '\\'
                | '|'
                | '*'
                | '&'
                | '#'
                | '@'
                | '%'
                | '^'
                | '~'
                | '`'
        )
    });
    if trimmed.is_empty() {
        return Vec::new();
    }
    // Split on characters that cannot appear in a meaningful token.
    let parts: Vec<String> = trimmed
        .split(|c: char| {
            matches!(
                c,
                '/' | '\\'
                    | '('
                    | ')'
                    | '['
                    | ']'
                    | '{'
                    | '}'
                    | '<'
                    | '>'
                    | '|'
                    | ';'
                    | ','
                    | '"'
                    | '`'
            )
        })
        .map(str::to_owned)
        .filter(|s| !s.is_empty())
        .collect();
    parts
}

/// Detects the concept type from the original-cased raw token list.
/// Falls back to the lower-cased term when no match is found in raw tokens.
fn detect_concept_type_from_raw(
    lc_term: &str,
    raw_tokens: &[String],
    lc_tokens: &[String],
) -> ConceptType {
    // Try to find the original-cased version of the term.
    let raw = lc_tokens
        .iter()
        .zip(raw_tokens.iter())
        .find_map(|(lc, raw)| {
            if lc == lc_term {
                Some(raw.as_str())
            } else {
                None
            }
        })
        .unwrap_or(lc_term);
    ConceptExtractor::detect_concept_type(raw)
}

/// Determines the concept type of a multi-word phrase from its raw (original-cased)
/// surface form.  If any token looks like an entity or technical term, the whole
/// phrase gets that classification; otherwise it is [`ConceptType::Phrase`].
fn detect_phrase_type(raw_phrase: &str) -> ConceptType {
    // If the phrase contains a space it is by definition a Phrase, but we still
    // check whether it qualifies as an Entity (e.g. "New York") or Technical
    // (e.g. "get_item size").
    let mut has_entity = false;
    let mut has_technical = false;
    for part in raw_phrase.split(' ') {
        match ConceptExtractor::detect_concept_type(part) {
            ConceptType::Entity => has_entity = true,
            ConceptType::Technical => has_technical = true,
            _ => {}
        }
    }
    if has_entity {
        ConceptType::Entity
    } else if has_technical {
        ConceptType::Technical
    } else {
        ConceptType::Phrase
    }
}

/// Returns a curated English stop-word list.
fn default_stop_words() -> Vec<String> {
    [
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
        "from", "as", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
        "do", "does", "did", "will", "would", "could", "should", "may", "might", "shall", "can",
        "need", "dare", "ought", "used", "it", "its", "this", "that", "these", "those", "i", "me",
        "my", "we", "our", "you", "your", "he", "she", "they", "them", "their", "his", "her",
        "who", "which", "what", "when", "where", "why", "how", "all", "any", "both", "each", "few",
        "more", "most", "other", "some", "such", "no", "not", "only", "own", "same", "than", "too",
        "very", "just", "because", "if", "then", "so", "up", "out", "about", "into", "through",
        "during", "before", "after", "above", "below", "between", "each", "every", "also", "get",
        "got", "let",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_extractor() -> ConceptExtractor {
        ConceptExtractor::new(ExtractorConfig::default())
    }

    // -----------------------------------------------------------------------
    // Tokenization
    // -----------------------------------------------------------------------

    #[test]
    fn test_tokenize_basic() {
        let tokens = ConceptExtractor::tokenize("Hello, world! This is a test.");
        assert!(tokens.contains(&"hello".to_string()));
        assert!(tokens.contains(&"world".to_string()));
        assert!(tokens.contains(&"test".to_string()));
    }

    #[test]
    fn test_tokenize_punctuation_stripped() {
        let tokens = ConceptExtractor::tokenize("foo, bar; baz.");
        assert!(tokens.contains(&"foo".to_string()));
        assert!(tokens.contains(&"bar".to_string()));
        assert!(tokens.contains(&"baz".to_string()));
        // Punctuation characters must not appear as standalone tokens.
        assert!(!tokens.iter().any(|t| t == "," || t == ";" || t == "."));
    }

    #[test]
    fn test_tokenize_short_tokens_included() {
        // Default min_term_length = 3; "a" and "is" should be filtered in
        // concept extraction but tokenize itself does not filter length.
        let tokens = ConceptExtractor::tokenize("a cat");
        assert!(tokens.contains(&"cat".to_string()));
    }

    #[test]
    fn test_tokenize_embedded_slash() {
        let tokens = ConceptExtractor::tokenize("foo/bar baz");
        assert!(tokens.contains(&"foo".to_string()));
        assert!(tokens.contains(&"bar".to_string()));
    }

    #[test]
    fn test_tokenize_empty_string() {
        let tokens = ConceptExtractor::tokenize("");
        assert!(tokens.is_empty());
    }

    #[test]
    fn test_tokenize_whitespace_only() {
        let tokens = ConceptExtractor::tokenize("   \t\n  ");
        assert!(tokens.is_empty());
    }

    // -----------------------------------------------------------------------
    // TF computation
    // -----------------------------------------------------------------------

    #[test]
    fn test_compute_tf_basic() {
        let tokens: Vec<String> = ["apple", "banana", "apple", "cherry"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let tf = ConceptExtractor::compute_tf(&tokens);
        assert_eq!(tf.get("apple"), Some(&2));
        assert_eq!(tf.get("banana"), Some(&1));
        assert_eq!(tf.get("cherry"), Some(&1));
    }

    #[test]
    fn test_compute_tf_empty() {
        let tf = ConceptExtractor::compute_tf(&[]);
        assert!(tf.is_empty());
    }

    #[test]
    fn test_compute_tf_single_token() {
        let tokens = vec!["rust".to_string()];
        let tf = ConceptExtractor::compute_tf(&tokens);
        assert_eq!(tf.get("rust"), Some(&1));
    }

    // -----------------------------------------------------------------------
    // TF-IDF scoring
    // -----------------------------------------------------------------------

    #[test]
    fn test_tfidf_increases_with_frequency() {
        let mut extractor = make_extractor();
        // Give the corpus some content so IDF is non-trivial.
        extractor.update_corpus(&["rust".to_string(), "programming".to_string()]);
        extractor.doc_count = 1;
        let score_low = extractor.compute_tfidf("rust", 1, 100);
        let score_high = extractor.compute_tfidf("rust", 10, 100);
        assert!(
            score_high > score_low,
            "Higher TF should yield higher TF-IDF"
        );
    }

    #[test]
    fn test_tfidf_rare_term_scores_higher() {
        let mut extractor = make_extractor();
        // "common" appears in all 10 docs; "rare" appears in only 1.
        for _ in 0..10 {
            extractor.update_corpus(&["common".to_string()]);
            extractor.doc_count += 1;
        }
        extractor.update_corpus(&["rare".to_string()]);
        extractor.doc_count += 1;

        let score_common = extractor.compute_tfidf("common", 3, 50);
        let score_rare = extractor.compute_tfidf("rare", 3, 50);
        assert!(
            score_rare > score_common,
            "Rare term (high IDF) should score higher than common term"
        );
    }

    #[test]
    fn test_tfidf_zero_frequency_term() {
        let extractor = make_extractor();
        // Term not in corpus — df defaults to 0.
        let score = extractor.compute_tfidf("unknown", 1, 10);
        assert!(score > 0.0, "Score must be positive even for unknown terms");
    }

    // -----------------------------------------------------------------------
    // N-gram extraction
    // -----------------------------------------------------------------------

    #[test]
    fn test_extract_bigrams() {
        let tokens: Vec<String> = ["machine", "learning", "model"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let bigrams = ConceptExtractor::extract_ngrams(&tokens, 2);
        assert_eq!(bigrams, vec!["machine learning", "learning model"]);
    }

    #[test]
    fn test_extract_trigrams() {
        let tokens: Vec<String> = ["deep", "neural", "network", "architecture"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let trigrams = ConceptExtractor::extract_ngrams(&tokens, 3);
        assert_eq!(
            trigrams,
            vec!["deep neural network", "neural network architecture"]
        );
    }

    #[test]
    fn test_extract_ngrams_too_short() {
        let tokens: Vec<String> = ["only", "two"].iter().map(|s| s.to_string()).collect();
        let trigrams = ConceptExtractor::extract_ngrams(&tokens, 3);
        assert!(trigrams.is_empty());
    }

    #[test]
    fn test_extract_ngrams_n_zero() {
        let tokens: Vec<String> = ["hello", "world"].iter().map(|s| s.to_string()).collect();
        let result = ConceptExtractor::extract_ngrams(&tokens, 0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_extract_unigrams_as_ngrams() {
        let tokens: Vec<String> = ["foo", "bar"].iter().map(|s| s.to_string()).collect();
        let unigrams = ConceptExtractor::extract_ngrams(&tokens, 1);
        assert_eq!(unigrams, vec!["foo", "bar"]);
    }

    // -----------------------------------------------------------------------
    // Entity detection
    // -----------------------------------------------------------------------

    #[test]
    fn test_detect_entity_starts_uppercase() {
        assert_eq!(
            ConceptExtractor::detect_concept_type("London"),
            ConceptType::Entity
        );
    }

    #[test]
    fn test_detect_entity_all_caps() {
        assert_eq!(
            ConceptExtractor::detect_concept_type("API"),
            ConceptType::Entity
        );
        assert_eq!(
            ConceptExtractor::detect_concept_type("HTTP"),
            ConceptType::Entity
        );
    }

    #[test]
    fn test_detect_entity_single_capital() {
        assert_eq!(
            ConceptExtractor::detect_concept_type("Rust"),
            ConceptType::Entity
        );
    }

    // -----------------------------------------------------------------------
    // Technical term detection
    // -----------------------------------------------------------------------

    #[test]
    fn test_detect_technical_camel_case() {
        assert_eq!(
            ConceptExtractor::detect_concept_type("camelCase"),
            ConceptType::Technical
        );
        assert_eq!(
            ConceptExtractor::detect_concept_type("myVariable"),
            ConceptType::Technical
        );
    }

    #[test]
    fn test_detect_technical_snake_case() {
        assert_eq!(
            ConceptExtractor::detect_concept_type("snake_case"),
            ConceptType::Technical
        );
        assert_eq!(
            ConceptExtractor::detect_concept_type("get_value"),
            ConceptType::Technical
        );
    }

    #[test]
    fn test_detect_keyword_lowercase() {
        assert_eq!(
            ConceptExtractor::detect_concept_type("keyword"),
            ConceptType::Keyword
        );
    }

    #[test]
    fn test_detect_phrase_with_space() {
        assert_eq!(
            ConceptExtractor::detect_concept_type("machine learning"),
            ConceptType::Phrase
        );
    }

    // -----------------------------------------------------------------------
    // Stop word filtering
    // -----------------------------------------------------------------------

    #[test]
    fn test_stop_word_filtered() {
        let extractor = make_extractor();
        assert!(extractor.is_stop_word("the"));
        assert!(extractor.is_stop_word("and"));
        assert!(extractor.is_stop_word("is"));
    }

    #[test]
    fn test_stop_word_case_insensitive() {
        let extractor = make_extractor();
        assert!(extractor.is_stop_word("The"));
        assert!(extractor.is_stop_word("AND"));
    }

    #[test]
    fn test_non_stop_word() {
        let extractor = make_extractor();
        assert!(!extractor.is_stop_word("algorithm"));
        assert!(!extractor.is_stop_word("neural"));
    }

    // -----------------------------------------------------------------------
    // Minimum frequency filter
    // -----------------------------------------------------------------------

    #[test]
    fn test_min_frequency_filter() {
        let config = ExtractorConfig {
            min_term_frequency: 2,
            max_concepts: 100,
            ..ExtractorConfig::default()
        };
        let mut extractor = ConceptExtractor::new(config);
        // "algorithm" appears once; "network" appears twice.
        let concepts = extractor.extract("neural network deep network algorithm");
        let terms: Vec<&str> = concepts.iter().map(|c| c.term.as_str()).collect();
        assert!(
            terms.contains(&"network"),
            "network (freq=2) should be included"
        );
        assert!(
            !terms.contains(&"algorithm"),
            "algorithm (freq=1) should be excluded"
        );
    }

    // -----------------------------------------------------------------------
    // max_concepts limit
    // -----------------------------------------------------------------------

    #[test]
    fn test_max_concepts_limit() {
        let config = ExtractorConfig {
            max_concepts: 3,
            min_term_frequency: 1,
            ..ExtractorConfig::default()
        };
        let mut extractor = ConceptExtractor::new(config);
        let concepts = extractor.extract(
            "machine learning artificial intelligence deep neural network computer vision",
        );
        assert!(concepts.len() <= 3, "Should not exceed max_concepts limit");
    }

    // -----------------------------------------------------------------------
    // Corpus IDF updates
    // -----------------------------------------------------------------------

    #[test]
    fn test_update_corpus_increments_doc_freq() {
        let mut extractor = make_extractor();
        let tokens: Vec<String> = ["rust", "programming"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        extractor.update_corpus(&tokens);
        assert_eq!(extractor.corpus_stats.get("rust"), Some(&1));
        assert_eq!(extractor.corpus_stats.get("programming"), Some(&1));
        // Same doc: repeated token should not double-count.
        let tokens2: Vec<String> = ["rust", "rust", "memory"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        extractor.update_corpus(&tokens2);
        assert_eq!(
            extractor.corpus_stats.get("rust"),
            Some(&2),
            "rust should appear in 2 documents"
        );
        assert_eq!(extractor.corpus_stats.get("memory"), Some(&1));
    }

    // -----------------------------------------------------------------------
    // Multi-document IDF
    // -----------------------------------------------------------------------

    #[test]
    fn test_multi_document_idf() {
        let mut extractor = make_extractor();
        // Doc 1: "vector search"
        extractor.extract("vector search database system architecture");
        // Doc 2: "vector embedding"
        extractor.extract("vector embedding representation learning");
        // "vector" appears in 2 docs → lower IDF than "embedding" (1 doc).
        let idf_vector = extractor.corpus_stats.get("vector").copied().unwrap_or(0);
        let idf_embedding = extractor
            .corpus_stats
            .get("embedding")
            .copied()
            .unwrap_or(0);
        assert_eq!(idf_vector, 2, "vector should appear in 2 documents");
        assert_eq!(idf_embedding, 1, "embedding should appear in 1 document");
    }

    // -----------------------------------------------------------------------
    // Empty text
    // -----------------------------------------------------------------------

    #[test]
    fn test_extract_empty_text() {
        let mut extractor = make_extractor();
        let concepts = extractor.extract("");
        assert!(concepts.is_empty(), "Empty text should yield no concepts");
    }

    #[test]
    fn test_extract_only_stop_words() {
        let mut extractor = make_extractor();
        let concepts = extractor.extract("the and or but is are was");
        assert!(
            concepts.is_empty(),
            "Stop-word-only text should yield no concepts"
        );
    }

    // -----------------------------------------------------------------------
    // Stats tracking
    // -----------------------------------------------------------------------

    #[test]
    fn test_stats_documents_processed() {
        let mut extractor = make_extractor();
        assert_eq!(extractor.stats().documents_processed, 0);
        extractor.extract("semantic vector search");
        assert_eq!(extractor.stats().documents_processed, 1);
        extractor.extract("deep learning embeddings");
        assert_eq!(extractor.stats().documents_processed, 2);
    }

    #[test]
    fn test_stats_total_concepts_extracted() {
        let mut extractor = make_extractor();
        extractor.extract("neural network deep learning architecture");
        let after_first = extractor.stats().total_concepts_extracted;
        extractor.extract("vector database approximate search retrieval");
        let after_second = extractor.stats().total_concepts_extracted;
        assert!(
            after_second >= after_first,
            "Total concepts should be non-decreasing"
        );
    }

    #[test]
    fn test_stats_avg_concepts_per_doc() {
        let mut extractor = make_extractor();
        extractor.extract("machine learning model training pipeline");
        extractor.extract("vector similarity search index retrieval");
        let stats = extractor.stats();
        assert!(
            stats.avg_concepts_per_doc > 0.0,
            "avg_concepts_per_doc should be positive after processing docs"
        );
        assert_eq!(stats.documents_processed, 2);
    }

    // -----------------------------------------------------------------------
    // Concept type classification
    // -----------------------------------------------------------------------

    #[test]
    fn test_extract_includes_entities() {
        let mut extractor = ConceptExtractor::new(ExtractorConfig {
            stop_words: Vec::new(),
            min_term_length: 2,
            ..ExtractorConfig::default()
        });
        let concepts = extractor.extract("Python Rust Go programming languages");
        let entity_terms: Vec<&str> = concepts
            .iter()
            .filter(|c| c.concept_type == ConceptType::Entity)
            .map(|c| c.term.as_str())
            .collect();
        // At least one of the language names should be classified as Entity.
        assert!(
            !entity_terms.is_empty(),
            "Should detect capitalised language names as entities"
        );
    }

    #[test]
    fn test_extract_technical_terms() {
        let mut extractor = make_extractor();
        // snake_case and camelCase identifiers in technical prose.
        let concepts =
            extractor.extract("the function get_embedding uses camelCase internally for indexing");
        let technical: Vec<&str> = concepts
            .iter()
            .filter(|c| c.concept_type == ConceptType::Technical)
            .map(|c| c.term.as_str())
            .collect();
        assert!(
            !technical.is_empty(),
            "Technical terms should be detected: {:?}",
            technical
        );
    }

    #[test]
    fn test_phrase_extraction_bigram() {
        let mut extractor = ConceptExtractor::new(ExtractorConfig {
            max_ngram: 2,
            min_term_frequency: 1,
            max_concepts: 50,
            stop_words: Vec::new(),
            min_term_length: 2,
            idf_smoothing: 1.0,
        });
        let concepts = extractor.extract("machine learning machine learning algorithm");
        let phrase_terms: Vec<&str> = concepts
            .iter()
            .filter(|c| c.concept_type == ConceptType::Phrase)
            .map(|c| c.term.as_str())
            .collect();
        assert!(
            phrase_terms.contains(&"machine learning"),
            "Should extract 'machine learning' bigram, got: {:?}",
            phrase_terms
        );
    }

    #[test]
    fn test_concept_score_positive() {
        let mut extractor = make_extractor();
        let concepts = extractor
            .extract("information retrieval semantic search vector embeddings neural network");
        for c in &concepts {
            assert!(c.score > 0.0, "All concepts must have positive scores");
        }
    }

    #[test]
    fn test_concept_frequency_matches_occurrence() {
        let mut extractor = make_extractor();
        let concepts = extractor.extract("vector vector vector search search");
        let vector_c = concepts.iter().find(|c| c.term == "vector");
        let search_c = concepts.iter().find(|c| c.term == "search");
        if let Some(vc) = vector_c {
            assert_eq!(vc.frequency, 3, "vector should have frequency 3");
        }
        if let Some(sc) = search_c {
            assert_eq!(sc.frequency, 2, "search should have frequency 2");
        }
    }

    #[test]
    fn test_top_concepts_deduplication() {
        // Build a duplicate list and check dedup logic.
        let mut concepts = vec![
            Concept {
                term: "rust".to_string(),
                score: 0.5,
                frequency: 1,
                doc_frequency: 1,
                concept_type: ConceptType::Keyword,
            },
            Concept {
                term: "rust".to_string(),
                score: 0.9,
                frequency: 2,
                doc_frequency: 1,
                concept_type: ConceptType::Keyword,
            },
        ];
        let result = ConceptExtractor::top_concepts(&mut concepts, 10);
        assert_eq!(result.len(), 1, "Deduplication should collapse duplicates");
        assert!(
            (result[0].score - 0.9).abs() < 1e-9,
            "Should keep the higher-scoring entry"
        );
    }

    #[test]
    fn test_top_concepts_sorted_by_score() {
        let mut concepts = vec![
            Concept {
                term: "alpha".to_string(),
                score: 0.3,
                frequency: 1,
                doc_frequency: 1,
                concept_type: ConceptType::Keyword,
            },
            Concept {
                term: "beta".to_string(),
                score: 0.8,
                frequency: 2,
                doc_frequency: 1,
                concept_type: ConceptType::Keyword,
            },
            Concept {
                term: "gamma".to_string(),
                score: 0.5,
                frequency: 1,
                doc_frequency: 1,
                concept_type: ConceptType::Keyword,
            },
        ];
        let result = ConceptExtractor::top_concepts(&mut concepts, 10);
        assert_eq!(result[0].term, "beta");
        assert_eq!(result[1].term, "gamma");
        assert_eq!(result[2].term, "alpha");
    }
}
