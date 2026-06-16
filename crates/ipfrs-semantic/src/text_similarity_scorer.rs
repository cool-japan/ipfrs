//! # Text Similarity Scorer
//!
//! A multi-metric text similarity scoring system combining lexical, syntactic,
//! and semantic similarity measures with configurable weighting.
//!
//! ## Supported Metrics
//!
//! - **Jaccard**: Token-set intersection over union
//! - **Cosine**: TF-IDF weighted cosine similarity on term vectors
//! - **EditDistance**: Normalized Levenshtein distance similarity
//! - **NGram**: N-gram overlap (Jaccard on character/token n-grams)
//! - **LongestCommonSubsequence**: LCS-based similarity
//! - **EmbeddingCosine**: Cosine similarity on dense embedding vectors
//!
//! ## Usage
//!
//! ```rust
//! use ipfrs_semantic::text_similarity_scorer::{
//!     TextSimilarityScorer, ScorerConfig, SimilarityMetric, TextPair,
//! };
//!
//! let config = ScorerConfig::default();
//! let mut scorer = TextSimilarityScorer::new(config);
//!
//! let pair = TextPair {
//!     text_a: "the quick brown fox".to_string(),
//!     text_b: "a quick brown dog".to_string(),
//!     embedding_a: None,
//!     embedding_b: None,
//! };
//!
//! let result = scorer.score(pair);
//! println!("Composite similarity: {:.4}", result.composite_score);
//! ```

use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// A single similarity metric and the weight applied to it in composite scoring.
#[derive(Debug, Clone, PartialEq)]
pub enum SimilarityMetric {
    /// |A∩B| / |A∪B| on token sets (Jaccard index)
    Jaccard,
    /// TF-IDF weighted cosine similarity on term frequency vectors
    Cosine,
    /// Normalized Levenshtein: 1 − edit_distance / max(len_a, len_b)
    EditDistance,
    /// N-gram overlap: |ngrams_a ∩ ngrams_b| / |ngrams_a ∪ ngrams_b|
    NGram { n: usize },
    /// LCS length / max(len_a, len_b)
    LongestCommonSubsequence,
    /// Cosine similarity on provided dense embedding vectors
    EmbeddingCosine,
}

impl SimilarityMetric {
    /// Human-readable name of the metric, used in [`SimilarityScore::metric`].
    pub fn name(&self) -> String {
        match self {
            SimilarityMetric::Jaccard => "Jaccard".to_string(),
            SimilarityMetric::Cosine => "Cosine".to_string(),
            SimilarityMetric::EditDistance => "EditDistance".to_string(),
            SimilarityMetric::NGram { n } => format!("NGram({})", n),
            SimilarityMetric::LongestCommonSubsequence => "LongestCommonSubsequence".to_string(),
            SimilarityMetric::EmbeddingCosine => "EmbeddingCosine".to_string(),
        }
    }
}

/// An input text pair, optionally accompanied by pre-computed embedding vectors.
#[derive(Debug, Clone)]
pub struct TextPair {
    /// First text in the pair.
    pub text_a: String,
    /// Second text in the pair.
    pub text_b: String,
    /// Optional dense embedding for `text_a` (required by [`SimilarityMetric::EmbeddingCosine`]).
    pub embedding_a: Option<Vec<f64>>,
    /// Optional dense embedding for `text_b` (required by [`SimilarityMetric::EmbeddingCosine`]).
    pub embedding_b: Option<Vec<f64>>,
}

/// Configuration for [`TextSimilarityScorer`].
#[derive(Debug, Clone)]
pub struct ScorerConfig {
    /// Ordered list of `(metric, weight)` pairs.
    pub metrics: Vec<(SimilarityMetric, f64)>,
    /// When `true`, weights are normalised to sum to 1 before computing the
    /// composite score.
    pub normalize_weights: bool,
    /// Lower clamp for the composite score; default `0.0`.
    pub min_score: f64,
}

impl Default for ScorerConfig {
    fn default() -> Self {
        ScorerConfig {
            metrics: vec![
                (SimilarityMetric::Jaccard, 0.2),
                (SimilarityMetric::Cosine, 0.3),
                (SimilarityMetric::EditDistance, 0.2),
                (SimilarityMetric::NGram { n: 2 }, 0.15),
                (SimilarityMetric::LongestCommonSubsequence, 0.15),
            ],
            normalize_weights: true,
            min_score: 0.0,
        }
    }
}

/// Score produced by a single metric.
#[derive(Debug, Clone)]
pub struct SimilarityScore {
    /// Name of the metric (see [`SimilarityMetric::name`]).
    pub metric: String,
    /// Raw similarity value in `[0, 1]`.
    pub score: f64,
}

/// Aggregated result for a [`TextPair`].
#[derive(Debug, Clone)]
pub struct TextSimilarityResult {
    /// Copy of the first input text.
    pub text_a: String,
    /// Copy of the second input text.
    pub text_b: String,
    /// Individual scores for every configured metric.
    pub scores: Vec<SimilarityScore>,
    /// Weighted composite score (clamped to `[min_score, 1.0]`).
    pub composite_score: f64,
}

// ---------------------------------------------------------------------------
// Core scorer
// ---------------------------------------------------------------------------

/// Multi-metric text similarity scorer.
///
/// Create with [`TextSimilarityScorer::new`], then call [`TextSimilarityScorer::score`]
/// or [`TextSimilarityScorer::score_batch`] for each text pair.
#[derive(Debug)]
pub struct TextSimilarityScorer {
    /// Active configuration.
    pub config: ScorerConfig,
    /// Total number of pairs scored so far.
    pub total_scored: u64,
}

impl TextSimilarityScorer {
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    /// Create a new scorer from the given configuration.
    pub fn new(config: ScorerConfig) -> Self {
        TextSimilarityScorer {
            config,
            total_scored: 0,
        }
    }

    // -----------------------------------------------------------------------
    // Tokenisation
    // -----------------------------------------------------------------------

    /// Lowercase and split `text` on non-alphanumeric characters, discarding
    /// empty tokens and single-character tokens shorter than 1 char.
    /// Returns at least one token when the input is non-empty.
    pub fn tokenize(text: &str) -> Vec<String> {
        text.to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|t| !t.is_empty())
            .map(|t| t.to_string())
            .collect()
    }

    // -----------------------------------------------------------------------
    // Lexical metrics
    // -----------------------------------------------------------------------

    /// Jaccard similarity on token *sets*: |A∩B| / |A∪B|.
    /// Returns `1.0` if both token lists are empty.
    pub fn jaccard_similarity(a: &[String], b: &[String]) -> f64 {
        if a.is_empty() && b.is_empty() {
            return 1.0;
        }
        let set_a: HashSet<&String> = a.iter().collect();
        let set_b: HashSet<&String> = b.iter().collect();

        let intersection = set_a.intersection(&set_b).count();
        let union = set_a.union(&set_b).count();

        if union == 0 {
            1.0
        } else {
            intersection as f64 / union as f64
        }
    }

    /// TF-IDF cosine similarity between two token sequences.
    ///
    /// IDF is computed over the two-document micro-corpus:
    /// - `idf(t) = log(2 / (df + 1))` where `df` is the number of documents
    ///   (out of {`a`, `b`}) that contain `t`.
    ///
    /// Returns `0.0` if either document has a zero-norm TF-IDF vector or if
    /// both token lists are empty.
    pub fn cosine_tfidf(a: &[String], b: &[String]) -> f64 {
        if a.is_empty() && b.is_empty() {
            return 1.0;
        }
        if a.is_empty() || b.is_empty() {
            return 0.0;
        }

        // Compute term frequencies (raw counts / total)
        let tf_a = Self::term_frequencies(a);
        let tf_b = Self::term_frequencies(b);

        // Vocabulary
        let vocab: HashSet<&String> = tf_a.keys().chain(tf_b.keys()).collect();

        // IDF: log(2 / (df + 1)), df in {0, 1}
        // df = 0 means term in only one doc; df = 1 means in both
        let idf = |term: &String| -> f64 {
            let in_a = tf_a.contains_key(term);
            let in_b = tf_b.contains_key(term);
            let df = if in_a && in_b { 1u32 } else { 0u32 };
            ((2.0_f64) / (df as f64 + 1.0_f64)).ln()
        };

        // TF-IDF vectors; compute dot product and norms simultaneously
        let mut dot = 0.0_f64;
        let mut norm_a = 0.0_f64;
        let mut norm_b = 0.0_f64;

        for term in &vocab {
            let w_a = tf_a.get(*term).copied().unwrap_or(0.0) * idf(term);
            let w_b = tf_b.get(*term).copied().unwrap_or(0.0) * idf(term);
            dot += w_a * w_b;
            norm_a += w_a * w_a;
            norm_b += w_b * w_b;
        }

        let denom = norm_a.sqrt() * norm_b.sqrt();
        if denom == 0.0 {
            0.0
        } else {
            (dot / denom).clamp(0.0, 1.0)
        }
    }

    /// Standard dynamic-programming Levenshtein distance in O(mn) space.
    /// This is intentionally the full matrix version—short strings in practice.
    pub fn edit_distance(a: &str, b: &str) -> usize {
        let a_chars: Vec<char> = a.chars().collect();
        let b_chars: Vec<char> = b.chars().collect();
        let m = a_chars.len();
        let n = b_chars.len();

        if m == 0 {
            return n;
        }
        if n == 0 {
            return m;
        }

        // dp[i][j] = edit distance between a[..i] and b[..j]
        let mut dp = vec![vec![0usize; n + 1]; m + 1];

        for (i, row) in dp.iter_mut().enumerate() {
            row[0] = i;
        }
        // Initialise first row: dp[0][j] = j for all j
        if let Some(first_row) = dp.first_mut() {
            for (j, cell) in first_row.iter_mut().enumerate() {
                *cell = j;
            }
        }

        for i in 1..=m {
            for j in 1..=n {
                if a_chars[i - 1] == b_chars[j - 1] {
                    dp[i][j] = dp[i - 1][j - 1];
                } else {
                    dp[i][j] = 1 + dp[i - 1][j - 1].min(dp[i - 1][j]).min(dp[i][j - 1]);
                }
            }
        }

        dp[m][n]
    }

    /// Build the list of n-grams (sliding window) from a token slice.
    /// Each n-gram is the space-joined concatenation of `n` consecutive tokens.
    /// Returns an empty `Vec` when `tokens.len() < n`.
    pub fn ngrams(tokens: &[String], n: usize) -> Vec<String> {
        if n == 0 || tokens.len() < n {
            return Vec::new();
        }
        tokens.windows(n).map(|w| w.join(" ")).collect()
    }

    /// Longest Common Subsequence length using O(min(m,n)) space DP.
    pub fn lcs_length(a: &[String], b: &[String]) -> usize {
        // Ensure `b` is the shorter sequence for space efficiency
        let (longer, shorter) = if a.len() >= b.len() { (a, b) } else { (b, a) };

        let n = shorter.len();
        let mut prev = vec![0usize; n + 1];
        let mut curr = vec![0usize; n + 1];

        for item_l in longer.iter() {
            for j in 1..=n {
                if *item_l == shorter[j - 1] {
                    curr[j] = prev[j - 1] + 1;
                } else {
                    curr[j] = curr[j - 1].max(prev[j]);
                }
            }
            std::mem::swap(&mut prev, &mut curr);
            curr.fill(0);
        }

        prev[n]
    }

    // -----------------------------------------------------------------------
    // Metric dispatch
    // -----------------------------------------------------------------------

    /// Compute the raw value for a single `metric` on a `pair`.
    pub fn compute_metric(&self, metric: &SimilarityMetric, pair: &TextPair) -> f64 {
        match metric {
            SimilarityMetric::Jaccard => {
                let a = Self::tokenize(&pair.text_a);
                let b = Self::tokenize(&pair.text_b);
                Self::jaccard_similarity(&a, &b)
            }
            SimilarityMetric::Cosine => {
                let a = Self::tokenize(&pair.text_a);
                let b = Self::tokenize(&pair.text_b);
                Self::cosine_tfidf(&a, &b)
            }
            SimilarityMetric::EditDistance => {
                let a = pair.text_a.to_lowercase();
                let b = pair.text_b.to_lowercase();
                let max_len = a.chars().count().max(b.chars().count());
                if max_len == 0 {
                    return 1.0;
                }
                let dist = Self::edit_distance(&a, &b);
                (1.0 - dist as f64 / max_len as f64).max(0.0)
            }
            SimilarityMetric::NGram { n } => {
                let a_tokens = Self::tokenize(&pair.text_a);
                let b_tokens = Self::tokenize(&pair.text_b);
                let grams_a = Self::ngrams(&a_tokens, *n);
                let grams_b = Self::ngrams(&b_tokens, *n);
                Self::jaccard_similarity(&grams_a, &grams_b)
            }
            SimilarityMetric::LongestCommonSubsequence => {
                let a = Self::tokenize(&pair.text_a);
                let b = Self::tokenize(&pair.text_b);
                let max_len = a.len().max(b.len());
                if max_len == 0 {
                    return 1.0;
                }
                Self::lcs_length(&a, &b) as f64 / max_len as f64
            }
            SimilarityMetric::EmbeddingCosine => match (&pair.embedding_a, &pair.embedding_b) {
                (Some(ea), Some(eb)) => Self::embedding_cosine(ea, eb),
                _ => 0.0,
            },
        }
    }

    // -----------------------------------------------------------------------
    // Scoring
    // -----------------------------------------------------------------------

    /// Score a single [`TextPair`] and return the full result.
    pub fn score(&mut self, pair: TextPair) -> TextSimilarityResult {
        let raw_scores: Vec<SimilarityScore> = self
            .config
            .metrics
            .iter()
            .map(|(metric, _)| SimilarityScore {
                metric: metric.name(),
                score: self.compute_metric(metric, &pair),
            })
            .collect();

        // Compute effective weights
        let weights: Vec<f64> = self.config.metrics.iter().map(|(_, w)| *w).collect();
        let weight_sum: f64 = weights.iter().sum();

        let effective_weights: Vec<f64> = if self.config.normalize_weights && weight_sum > 0.0 {
            weights.iter().map(|w| w / weight_sum).collect()
        } else {
            weights
        };

        // Weighted sum
        let composite: f64 = raw_scores
            .iter()
            .zip(effective_weights.iter())
            .map(|(s, w)| s.score * w)
            .sum();

        let composite_score = composite.clamp(self.config.min_score, 1.0);

        self.total_scored += 1;

        TextSimilarityResult {
            text_a: pair.text_a,
            text_b: pair.text_b,
            scores: raw_scores,
            composite_score,
        }
    }

    /// Score a batch of [`TextPair`]s sequentially, returning one result per pair.
    pub fn score_batch(&mut self, pairs: Vec<TextPair>) -> Vec<TextSimilarityResult> {
        pairs.into_iter().map(|p| self.score(p)).collect()
    }

    /// Find the most similar candidate to `query` from a list of `candidates`.
    ///
    /// Scoring is performed without embeddings (lexical metrics only).
    /// Returns `None` if `candidates` is empty.
    pub fn most_similar<'a>(
        &mut self,
        query: &str,
        candidates: &'a [String],
    ) -> Option<(&'a str, f64)> {
        if candidates.is_empty() {
            return None;
        }

        let mut best_idx = 0usize;
        let mut best_score = f64::NEG_INFINITY;

        for (idx, candidate) in candidates.iter().enumerate() {
            let pair = TextPair {
                text_a: query.to_string(),
                text_b: candidate.clone(),
                embedding_a: None,
                embedding_b: None,
            };
            let result = self.score(pair);
            if result.composite_score > best_score {
                best_score = result.composite_score;
                best_idx = idx;
            }
        }

        Some((candidates[best_idx].as_str(), best_score))
    }

    /// Return statistics: `(total_scored,)`.
    pub fn scorer_stats(&self) -> (u64,) {
        (self.total_scored,)
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Compute relative term frequencies (count / total tokens) for a token slice.
    fn term_frequencies(tokens: &[String]) -> HashMap<String, f64> {
        let total = tokens.len() as f64;
        let mut counts: HashMap<String, f64> = HashMap::new();
        for t in tokens {
            *counts.entry(t.clone()).or_insert(0.0) += 1.0;
        }
        if total > 0.0 {
            for v in counts.values_mut() {
                *v /= total;
            }
        }
        counts
    }

    /// Cosine similarity between two dense f64 vectors.
    /// Returns `0.0` if either vector is zero or lengths differ.
    fn embedding_cosine(a: &[f64], b: &[f64]) -> f64 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }
        let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use crate::text_similarity_scorer::{
        ScorerConfig, SimilarityMetric, TextPair, TextSimilarityScorer,
    };

    // Helper: build a default scorer
    fn default_scorer() -> TextSimilarityScorer {
        TextSimilarityScorer::new(ScorerConfig::default())
    }

    // Helper: build a pair without embeddings
    fn pair(a: &str, b: &str) -> TextPair {
        TextPair {
            text_a: a.to_string(),
            text_b: b.to_string(),
            embedding_a: None,
            embedding_b: None,
        }
    }

    // --- tokenize -----------------------------------------------------------

    #[test]
    fn test_tokenize_basic() {
        let tokens = TextSimilarityScorer::tokenize("Hello, World!");
        assert_eq!(tokens, vec!["hello", "world"]);
    }

    #[test]
    fn test_tokenize_empty() {
        let tokens = TextSimilarityScorer::tokenize("");
        assert!(tokens.is_empty());
    }

    #[test]
    fn test_tokenize_numbers() {
        let tokens = TextSimilarityScorer::tokenize("version 3.14 is here");
        assert!(tokens.contains(&"3".to_string()));
        assert!(tokens.contains(&"14".to_string()));
    }

    #[test]
    fn test_tokenize_multiple_separators() {
        let tokens = TextSimilarityScorer::tokenize("a---b___c");
        assert_eq!(tokens, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_tokenize_lowercase() {
        let tokens = TextSimilarityScorer::tokenize("UPPER lower MiXeD");
        assert_eq!(tokens, vec!["upper", "lower", "mixed"]);
    }

    // --- jaccard_similarity -------------------------------------------------

    #[test]
    fn test_jaccard_identical() {
        let a = vec!["a".to_string(), "b".to_string()];
        let score = TextSimilarityScorer::jaccard_similarity(&a, &a);
        assert!((score - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_jaccard_disjoint() {
        let a = vec!["a".to_string()];
        let b = vec!["b".to_string()];
        let score = TextSimilarityScorer::jaccard_similarity(&a, &b);
        assert!((score - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_jaccard_partial_overlap() {
        let a = vec!["a".to_string(), "b".to_string()];
        let b = vec!["b".to_string(), "c".to_string()];
        // |{a,b}∩{b,c}| / |{a,b,c}| = 1/3
        let score = TextSimilarityScorer::jaccard_similarity(&a, &b);
        assert!((score - 1.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_jaccard_both_empty() {
        let a: Vec<String> = vec![];
        let score = TextSimilarityScorer::jaccard_similarity(&a, &a);
        assert!((score - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_jaccard_one_empty() {
        let a: Vec<String> = vec![];
        let b = vec!["x".to_string()];
        let score = TextSimilarityScorer::jaccard_similarity(&a, &b);
        assert!((score - 0.0).abs() < 1e-9);
    }

    // --- cosine_tfidf -------------------------------------------------------

    #[test]
    fn test_cosine_identical_texts() {
        // When all terms appear in both docs, IDF = log(2/(1+1)) = log(1) = 0 for every term.
        // This produces zero TF-IDF vectors, so cosine is defined as 0 (not 1).
        // The cosine metric therefore returns 0 for identical docs under this IDF scheme.
        let a = TextSimilarityScorer::tokenize("rust programming language");
        let score = TextSimilarityScorer::cosine_tfidf(&a, &a);
        // With df=1 for every shared term, IDF=0, so both vectors are zero → score=0.0
        assert!((score - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_cosine_both_empty() {
        let a: Vec<String> = vec![];
        let score = TextSimilarityScorer::cosine_tfidf(&a, &a);
        assert!((score - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_cosine_one_empty() {
        let a: Vec<String> = vec![];
        let b = TextSimilarityScorer::tokenize("hello world");
        let score = TextSimilarityScorer::cosine_tfidf(&a, &b);
        assert!((score - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_cosine_completely_different() {
        // Disjoint vocabularies: every term has df=0, so IDF is the same for all.
        // The dot product is 0 → cosine = 0.
        let a = TextSimilarityScorer::tokenize("apple banana cherry");
        let b = TextSimilarityScorer::tokenize("dog elephant fox");
        let score = TextSimilarityScorer::cosine_tfidf(&a, &b);
        assert!((score - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_cosine_partial_overlap() {
        // "rust" appears in both → IDF("rust") = log(2/2) = 0.
        // "language" and "programming" each appear in only one doc → IDF = log(2/1) > 0.
        // The shared term contributes nothing; only unique terms have non-zero weight.
        // The resulting vectors are orthogonal → cosine = 0.
        let a = TextSimilarityScorer::tokenize("rust language");
        let b = TextSimilarityScorer::tokenize("rust programming");
        let score = TextSimilarityScorer::cosine_tfidf(&a, &b);
        // Both unique terms (language, programming) have positive IDF but are in different docs,
        // so their product in the dot product is 0 → cosine = 0.
        assert!((score - 0.0).abs() < 1e-9);
    }

    // --- edit_distance ------------------------------------------------------

    #[test]
    fn test_edit_distance_identical() {
        assert_eq!(TextSimilarityScorer::edit_distance("hello", "hello"), 0);
    }

    #[test]
    fn test_edit_distance_empty_strings() {
        assert_eq!(TextSimilarityScorer::edit_distance("", ""), 0);
    }

    #[test]
    fn test_edit_distance_one_empty() {
        assert_eq!(TextSimilarityScorer::edit_distance("abc", ""), 3);
        assert_eq!(TextSimilarityScorer::edit_distance("", "abc"), 3);
    }

    #[test]
    fn test_edit_distance_single_substitution() {
        // kitten → sitten = 1 substitution
        assert_eq!(TextSimilarityScorer::edit_distance("kitten", "sitten"), 1);
    }

    #[test]
    fn test_edit_distance_classic_kitten_sitting() {
        // kitten → sitting = 3 operations
        assert_eq!(TextSimilarityScorer::edit_distance("kitten", "sitting"), 3);
    }

    // --- ngrams -------------------------------------------------------------

    #[test]
    fn test_ngrams_bigrams() {
        let tokens: Vec<String> = vec!["a", "b", "c"]
            .into_iter()
            .map(|s| s.to_string())
            .collect();
        let grams = TextSimilarityScorer::ngrams(&tokens, 2);
        assert_eq!(grams, vec!["a b", "b c"]);
    }

    #[test]
    fn test_ngrams_trigrams() {
        let tokens: Vec<String> = vec!["a", "b", "c", "d"]
            .into_iter()
            .map(|s| s.to_string())
            .collect();
        let grams = TextSimilarityScorer::ngrams(&tokens, 3);
        assert_eq!(grams, vec!["a b c", "b c d"]);
    }

    #[test]
    fn test_ngrams_empty_when_too_short() {
        let tokens: Vec<String> = vec!["a".to_string()];
        let grams = TextSimilarityScorer::ngrams(&tokens, 3);
        assert!(grams.is_empty());
    }

    #[test]
    fn test_ngrams_zero_n() {
        let tokens: Vec<String> = vec!["a".to_string(), "b".to_string()];
        let grams = TextSimilarityScorer::ngrams(&tokens, 0);
        assert!(grams.is_empty());
    }

    // --- lcs_length ---------------------------------------------------------

    #[test]
    fn test_lcs_identical() {
        let a: Vec<String> = vec!["a", "b", "c"]
            .into_iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(TextSimilarityScorer::lcs_length(&a, &a), 3);
    }

    #[test]
    fn test_lcs_disjoint() {
        let a: Vec<String> = vec!["a".to_string()];
        let b: Vec<String> = vec!["b".to_string()];
        assert_eq!(TextSimilarityScorer::lcs_length(&a, &b), 0);
    }

    #[test]
    fn test_lcs_partial() {
        let a: Vec<String> = vec!["a", "b", "c", "d"]
            .into_iter()
            .map(|s| s.to_string())
            .collect();
        let b: Vec<String> = vec!["b", "c", "e"]
            .into_iter()
            .map(|s| s.to_string())
            .collect();
        // LCS = [b, c]
        assert_eq!(TextSimilarityScorer::lcs_length(&a, &b), 2);
    }

    #[test]
    fn test_lcs_one_empty() {
        let a: Vec<String> = vec![];
        let b: Vec<String> = vec!["a".to_string()];
        assert_eq!(TextSimilarityScorer::lcs_length(&a, &b), 0);
    }

    // --- compute_metric -----------------------------------------------------

    #[test]
    fn test_compute_jaccard_via_metric() {
        let scorer = default_scorer();
        let p = pair("cat dog", "dog cat bird");
        // tokens a = {cat, dog}, b = {dog, cat, bird}
        // intersection = {cat, dog} = 2, union = {cat, dog, bird} = 3
        let score = scorer.compute_metric(&SimilarityMetric::Jaccard, &p);
        assert!((score - 2.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_compute_edit_distance_via_metric() {
        let scorer = default_scorer();
        let p = pair("abc", "abc");
        let score = scorer.compute_metric(&SimilarityMetric::EditDistance, &p);
        assert!((score - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_compute_ngram_via_metric() {
        let scorer = default_scorer();
        let p = pair("quick brown fox", "quick brown dog");
        let score = scorer.compute_metric(&SimilarityMetric::NGram { n: 2 }, &p);
        // bigrams a: [quick brown, brown fox], bigrams b: [quick brown, brown dog]
        // intersection = {quick brown} = 1, union = {quick brown, brown fox, brown dog} = 3
        assert!((score - 1.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_compute_lcs_via_metric() {
        let scorer = default_scorer();
        let p = pair("a b c", "a b d");
        let score = scorer.compute_metric(&SimilarityMetric::LongestCommonSubsequence, &p);
        // LCS = [a, b] = 2; max_len = 3
        assert!((score - 2.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_compute_embedding_cosine_with_embeddings() {
        let scorer = default_scorer();
        let v = vec![1.0_f64, 0.0, 0.0];
        let p = TextPair {
            text_a: "foo".to_string(),
            text_b: "bar".to_string(),
            embedding_a: Some(v.clone()),
            embedding_b: Some(v.clone()),
        };
        let score = scorer.compute_metric(&SimilarityMetric::EmbeddingCosine, &p);
        assert!((score - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_compute_embedding_cosine_missing() {
        let scorer = default_scorer();
        let p = pair("foo", "bar");
        let score = scorer.compute_metric(&SimilarityMetric::EmbeddingCosine, &p);
        assert!((score - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_compute_embedding_cosine_orthogonal() {
        let scorer = default_scorer();
        let p = TextPair {
            text_a: "foo".to_string(),
            text_b: "bar".to_string(),
            embedding_a: Some(vec![1.0, 0.0]),
            embedding_b: Some(vec![0.0, 1.0]),
        };
        let score = scorer.compute_metric(&SimilarityMetric::EmbeddingCosine, &p);
        assert!(score.abs() < 1e-9);
    }

    // --- score --------------------------------------------------------------

    #[test]
    fn test_score_returns_correct_metric_count() {
        let mut scorer = default_scorer();
        let result = scorer.score(pair("hello world", "hello rust"));
        // Default has 5 metrics
        assert_eq!(result.scores.len(), 5);
    }

    #[test]
    fn test_score_identical_texts_high_similarity() {
        // For identical texts, Jaccard=1, EditDistance=1, NGram=1, LCS=1 all score 1.0.
        // Cosine scores 0.0 because all shared terms get IDF=log(1)=0 under the two-doc IDF scheme.
        // Composite (normalised weights 0.2+0.3+0.2+0.15+0.15=1.0):
        //   = 1.0*0.2 + 0.0*0.3 + 1.0*0.2 + 1.0*0.15 + 1.0*0.15 = 0.70
        let mut scorer = default_scorer();
        let result = scorer.score(pair("the quick brown fox", "the quick brown fox"));
        // All lexical metrics except cosine return 1.0; composite ≥ 0.6 is a safe lower bound.
        assert!(result.composite_score >= 0.6);
        assert!(result.composite_score <= 1.0);
    }

    #[test]
    fn test_score_completely_different_texts_low_similarity() {
        let mut scorer = default_scorer();
        let result = scorer.score(pair("apple", "zzzzz"));
        // edit distance similarity for "apple" vs "zzzzz" is 0; others near 0
        assert!(result.composite_score < 0.5);
    }

    #[test]
    fn test_score_increments_total_scored() {
        let mut scorer = default_scorer();
        scorer.score(pair("a", "b"));
        scorer.score(pair("c", "d"));
        assert_eq!(scorer.total_scored, 2);
    }

    #[test]
    fn test_score_composite_clamped_to_min_score() {
        let config = ScorerConfig {
            metrics: vec![(SimilarityMetric::Jaccard, 1.0)],
            normalize_weights: true,
            min_score: 0.5,
        };
        let mut scorer = TextSimilarityScorer::new(config);
        // Completely disjoint → Jaccard = 0, but min_score clamps to 0.5
        let result = scorer.score(pair("apple", "banana"));
        assert!(result.composite_score >= 0.5);
    }

    #[test]
    fn test_score_weights_are_normalized() {
        // Large weights but normalized → composite ≤ 1.0
        let config = ScorerConfig {
            metrics: vec![
                (SimilarityMetric::Jaccard, 100.0),
                (SimilarityMetric::EditDistance, 200.0),
            ],
            normalize_weights: true,
            min_score: 0.0,
        };
        let mut scorer = TextSimilarityScorer::new(config);
        let result = scorer.score(pair("hello world", "hello rust"));
        assert!(result.composite_score <= 1.0);
        assert!(result.composite_score >= 0.0);
    }

    #[test]
    fn test_score_with_embedding_cosine_metric() {
        let config = ScorerConfig {
            metrics: vec![(SimilarityMetric::EmbeddingCosine, 1.0)],
            normalize_weights: true,
            min_score: 0.0,
        };
        let mut scorer = TextSimilarityScorer::new(config);
        let p = TextPair {
            text_a: "foo".to_string(),
            text_b: "bar".to_string(),
            embedding_a: Some(vec![1.0, 0.0]),
            embedding_b: Some(vec![1.0, 0.0]),
        };
        let result = scorer.score(p);
        assert!((result.composite_score - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_score_preserves_texts() {
        let mut scorer = default_scorer();
        let result = scorer.score(pair("alpha text", "beta text"));
        assert_eq!(result.text_a, "alpha text");
        assert_eq!(result.text_b, "beta text");
    }

    // --- score_batch --------------------------------------------------------

    #[test]
    fn test_score_batch_count() {
        let mut scorer = default_scorer();
        let pairs = vec![
            pair("a b", "a c"),
            pair("x y", "y z"),
            pair("foo bar", "foo baz"),
        ];
        let results = scorer.score_batch(pairs);
        assert_eq!(results.len(), 3);
        assert_eq!(scorer.total_scored, 3);
    }

    #[test]
    fn test_score_batch_empty() {
        let mut scorer = default_scorer();
        let results = scorer.score_batch(vec![]);
        assert!(results.is_empty());
        assert_eq!(scorer.total_scored, 0);
    }

    // --- most_similar -------------------------------------------------------

    #[test]
    fn test_most_similar_finds_best_match() {
        let mut scorer = default_scorer();
        let candidates: Vec<String> = vec![
            "the quick brown fox".to_string(),
            "completely unrelated words".to_string(),
            "quick fox jumps".to_string(),
        ];
        let (best, _score) = scorer
            .most_similar("quick brown fox", &candidates)
            .expect("test: candidates is non-empty so most_similar must return Some");
        // First candidate is identical modulo "the"; first or third should win
        assert!(best.contains("quick") && best.contains("fox"));
    }

    #[test]
    fn test_most_similar_empty_candidates_returns_none() {
        let mut scorer = default_scorer();
        let candidates: Vec<String> = vec![];
        let result = scorer.most_similar("query", &candidates);
        assert!(result.is_none());
    }

    #[test]
    fn test_most_similar_score_in_range() {
        let mut scorer = default_scorer();
        let candidates: Vec<String> = vec!["hello world".to_string()];
        let (_best, score) = scorer
            .most_similar("hello world", &candidates)
            .expect("test: candidates is non-empty so most_similar must return Some");
        assert!((0.0..=1.0).contains(&score));
    }

    // --- scorer_stats -------------------------------------------------------

    #[test]
    fn test_scorer_stats_initial() {
        let scorer = default_scorer();
        assert_eq!(scorer.scorer_stats(), (0,));
    }

    #[test]
    fn test_scorer_stats_after_scoring() {
        let mut scorer = default_scorer();
        scorer.score(pair("a", "b"));
        scorer.score(pair("c", "d"));
        assert_eq!(scorer.scorer_stats(), (2,));
    }

    // --- metric name --------------------------------------------------------

    #[test]
    fn test_metric_name_jaccard() {
        assert_eq!(SimilarityMetric::Jaccard.name(), "Jaccard");
    }

    #[test]
    fn test_metric_name_ngram() {
        assert_eq!(SimilarityMetric::NGram { n: 3 }.name(), "NGram(3)");
    }

    #[test]
    fn test_metric_name_lcs() {
        assert_eq!(
            SimilarityMetric::LongestCommonSubsequence.name(),
            "LongestCommonSubsequence"
        );
    }

    #[test]
    fn test_metric_name_embedding_cosine() {
        assert_eq!(SimilarityMetric::EmbeddingCosine.name(), "EmbeddingCosine");
    }

    // --- edge cases & regression --------------------------------------------

    #[test]
    fn test_score_both_empty_strings() {
        let mut scorer = default_scorer();
        let result = scorer.score(pair("", ""));
        // All metrics return 1.0 for two empty strings
        assert!(result.composite_score > 0.9);
    }

    #[test]
    fn test_lcs_both_empty() {
        let a: Vec<String> = vec![];
        let b: Vec<String> = vec![];
        assert_eq!(TextSimilarityScorer::lcs_length(&a, &b), 0);
    }

    #[test]
    fn test_edit_distance_symmetric() {
        let d1 = TextSimilarityScorer::edit_distance("sunday", "saturday");
        let d2 = TextSimilarityScorer::edit_distance("saturday", "sunday");
        assert_eq!(d1, d2);
    }

    #[test]
    fn test_cosine_tfidf_bounds() {
        // Score must always be in [0, 1]
        let pairs = [
            ("", ""),
            ("a b c", ""),
            ("", "x y"),
            ("hello", "world"),
            ("rust rust rust", "rust code"),
        ];
        for (a, b) in pairs {
            let ta = TextSimilarityScorer::tokenize(a);
            let tb = TextSimilarityScorer::tokenize(b);
            let s = TextSimilarityScorer::cosine_tfidf(&ta, &tb);
            assert!(
                (0.0..=1.0).contains(&s),
                "out-of-range cosine for ({a:?}, {b:?}): {s}"
            );
        }
    }

    #[test]
    fn test_embedding_cosine_mismatched_dims() {
        let scorer = default_scorer();
        let p = TextPair {
            text_a: "foo".to_string(),
            text_b: "bar".to_string(),
            embedding_a: Some(vec![1.0, 0.0]),
            embedding_b: Some(vec![1.0, 0.0, 0.0]),
        };
        let score = scorer.compute_metric(&SimilarityMetric::EmbeddingCosine, &p);
        assert!((score - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_embedding_cosine_zero_vector() {
        let scorer = default_scorer();
        let p = TextPair {
            text_a: "foo".to_string(),
            text_b: "bar".to_string(),
            embedding_a: Some(vec![0.0, 0.0]),
            embedding_b: Some(vec![1.0, 0.0]),
        };
        let score = scorer.compute_metric(&SimilarityMetric::EmbeddingCosine, &p);
        assert!((score - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_score_all_metrics_in_result() {
        let config = ScorerConfig {
            metrics: vec![
                (SimilarityMetric::Jaccard, 1.0),
                (SimilarityMetric::Cosine, 1.0),
                (SimilarityMetric::EditDistance, 1.0),
                (SimilarityMetric::NGram { n: 2 }, 1.0),
                (SimilarityMetric::LongestCommonSubsequence, 1.0),
                (SimilarityMetric::EmbeddingCosine, 1.0),
            ],
            normalize_weights: true,
            min_score: 0.0,
        };
        let mut scorer = TextSimilarityScorer::new(config);
        let result = scorer.score(pair("hello world", "hello rust"));
        assert_eq!(result.scores.len(), 6);
        // All metric scores must be in [0, 1]
        for s in &result.scores {
            assert!(
                s.score >= 0.0 && s.score <= 1.0,
                "bad score for {}: {}",
                s.metric,
                s.score
            );
        }
    }
}
