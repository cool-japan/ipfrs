//! Lexicon-based sentiment analysis engine with aspect-level sentiment detection.
//!
//! This module provides `SentimentAnalyzer` which performs document-level and
//! aspect-level sentiment scoring using a customisable lexicon. Negation handling,
//! intensifier/diminisher scaling, and batch-processing utilities are included.
//!
//! # Example
//!
//! ```rust
//! use ipfrs_semantic::sentiment_analyzer::{SentimentAnalyzer, SentimentConfig, SentimentPolarity};
//!
//! let config = SentimentConfig::default();
//! let analyzer = SentimentAnalyzer::new(config);
//! let result = analyzer.analyze("doc1".to_string(), "The service is absolutely fantastic!");
//! assert_eq!(result.overall.polarity(), SentimentPolarity::Positive);
//! ```

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Overall sentiment polarity of a piece of text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SentimentPolarity {
    /// Compound score > 0.05.
    Positive,
    /// Compound score < -0.05.
    Negative,
    /// Both positive and negative raw scores > 0.1 (ambivalent text).
    Mixed,
    /// Compound score in [-0.05, 0.05] and not Mixed.
    Neutral,
}

/// Raw sentiment scores for a piece of text, normalised to [0, 1] for the
/// positive/negative/neutral axes. `compound` is in [-1, 1].
#[derive(Debug, Clone, PartialEq)]
pub struct SentimentScore {
    /// Proportion of positive sentiment signals.
    pub positive: f64,
    /// Proportion of negative sentiment signals.
    pub negative: f64,
    /// Proportion of neutral (non-sentiment) content.
    pub neutral: f64,
    /// Aggregate polarity in [-1, 1].
    pub compound: f64,
}

impl SentimentScore {
    /// Create a zero-valued score (used as an identity element for aggregation).
    pub fn zero() -> Self {
        Self {
            positive: 0.0,
            negative: 0.0,
            neutral: 1.0,
            compound: 0.0,
        }
    }

    /// Determine the overall polarity from the component scores.
    ///
    /// Mixed takes priority: when both `positive` and `negative` exceed 0.1 the
    /// text is classified as `Mixed` regardless of the compound score.
    pub fn polarity(&self) -> SentimentPolarity {
        if self.positive > 0.1 && self.negative > 0.1 {
            SentimentPolarity::Mixed
        } else if self.compound > 0.05 {
            SentimentPolarity::Positive
        } else if self.compound < -0.05 {
            SentimentPolarity::Negative
        } else {
            SentimentPolarity::Neutral
        }
    }
}

/// Sentiment score for a particular aspect (topic) within a document.
#[derive(Debug, Clone)]
pub struct AspectSentiment {
    /// The aspect keyword that was matched.
    pub aspect: String,
    /// Sentiment score computed within the context window around this aspect.
    pub sentiment: SentimentScore,
    /// Words near the aspect that contributed to the sentiment score.
    pub mentions: Vec<String>,
}

/// Full sentiment analysis result for one document.
#[derive(Debug, Clone)]
pub struct SentimentResult {
    /// User-supplied identifier for the analysed text.
    pub text_id: String,
    /// Document-level sentiment.
    pub overall: SentimentScore,
    /// Per-aspect breakdown (one entry per occurrence of each aspect keyword).
    pub aspects: Vec<AspectSentiment>,
    /// Total number of tokens in the document.
    pub word_count: usize,
    /// Number of tokens that matched a lexicon entry (sentiment words).
    pub sentiment_word_count: usize,
}

/// A single entry in the sentiment lexicon.
#[derive(Debug, Clone)]
pub struct LexiconEntry {
    /// The canonical (lowercase) form of the word.
    pub word: String,
    /// Positive sentiment contribution [0, ∞).
    pub positive_score: f64,
    /// Negative sentiment contribution [0, ∞).
    pub negative_score: f64,
    /// Multiplier applied to sentiment scores of *nearby* words.
    ///
    /// - `1.0` — neutral (ordinary word)
    /// - `> 1.0` — intensifier (e.g. "very", "extremely")
    /// - `< 1.0` — diminisher (e.g. "slightly", "barely")
    pub intensifier: f64,
}

impl LexiconEntry {
    /// Convenience constructor for a purely positive word.
    pub fn positive(word: impl Into<String>, score: f64) -> Self {
        Self {
            word: word.into(),
            positive_score: score,
            negative_score: 0.0,
            intensifier: 1.0,
        }
    }

    /// Convenience constructor for a purely negative word.
    pub fn negative(word: impl Into<String>, score: f64) -> Self {
        Self {
            word: word.into(),
            positive_score: 0.0,
            negative_score: score,
            intensifier: 1.0,
        }
    }

    /// Convenience constructor for an intensifier or diminisher (no sentiment of its own).
    pub fn modifier(word: impl Into<String>, intensifier: f64) -> Self {
        Self {
            word: word.into(),
            positive_score: 0.0,
            negative_score: 0.0,
            intensifier,
        }
    }
}

/// Configuration for `SentimentAnalyzer`.
#[derive(Debug, Clone)]
pub struct SentimentConfig {
    /// Number of tokens either side of an aspect keyword to consider when
    /// computing aspect-level sentiment.
    pub window_size: usize,
    /// Number of tokens *before* a sentiment word to look for negators.
    pub negation_window: usize,
    /// Domain-specific aspect keywords to track.
    pub aspect_keywords: Vec<String>,
}

impl Default for SentimentConfig {
    fn default() -> Self {
        Self {
            window_size: 5,
            negation_window: 3,
            aspect_keywords: vec![
                "quality".to_string(),
                "price".to_string(),
                "service".to_string(),
                "speed".to_string(),
                "reliability".to_string(),
                "performance".to_string(),
            ],
        }
    }
}

/// Aggregate statistics over a collection of `SentimentResult`s.
#[derive(Debug, Clone)]
pub struct SentimentAnalyzerStats {
    /// Total number of documents analysed.
    pub total_analyzed: usize,
    /// Documents classified as Positive.
    pub positive_count: usize,
    /// Documents classified as Negative.
    pub negative_count: usize,
    /// Documents classified as Neutral.
    pub neutral_count: usize,
    /// Documents classified as Mixed.
    pub mixed_count: usize,
    /// Mean compound score across all documents.
    pub avg_compound: f64,
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Negation words that flip the polarity of sentiment within their window.
const NEGATIONS: &[&str] = &[
    "not", "never", "no", "isn't", "wasn't", "aren't", "weren't", "doesn't", "didn't", "don't",
    "nor", "neither", "without", "lacks", "lack",
];

/// Split `text` into lowercase tokens, keeping apostrophes so that contractions
/// such as "isn't" remain intact.
fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric() && c != '\'')
        .filter(|s| !s.is_empty())
        .map(|s| s.to_lowercase())
        .collect()
}

/// Check whether any token in `window` is a negation word.
fn contains_negation(window: &[String]) -> bool {
    window.iter().any(|w| NEGATIONS.contains(&w.as_str()))
}

/// Find the maximum intensifier multiplier in a slice of tokens, given the lexicon.
/// Returns 1.0 if no intensifier or diminisher is present.
fn find_intensity_multiplier(window: &[String], lexicon: &HashMap<String, LexiconEntry>) -> f64 {
    let mut multiplier = 1.0_f64;
    for token in window {
        if let Some(entry) = lexicon.get(token) {
            // Only apply if this token is a modifier (no raw sentiment of its own)
            if entry.positive_score == 0.0
                && entry.negative_score == 0.0
                && entry.intensifier != 1.0
            {
                // Compose multipliers: use the one that moves furthest from 1.0
                if (entry.intensifier - 1.0).abs() > (multiplier - 1.0).abs() {
                    multiplier = entry.intensifier;
                }
            }
        }
    }
    multiplier
}

// ---------------------------------------------------------------------------
// SentimentAnalyzer
// ---------------------------------------------------------------------------

/// Lexicon-based sentiment analysis engine with aspect-level detection.
///
/// The analyser operates entirely in-process; no external model or network call
/// is required. Accuracy is limited by the quality of the built-in (or
/// user-supplied) lexicon — for high-stakes applications consider augmenting
/// the lexicon via `with_lexicon_entry`.
pub struct SentimentAnalyzer {
    /// Runtime configuration.
    pub config: SentimentConfig,
    /// Word-to-entry lookup table.
    pub lexicon: HashMap<String, LexiconEntry>,
}

impl SentimentAnalyzer {
    /// Build a new analyser seeded with the built-in lexicon.
    pub fn new(config: SentimentConfig) -> Self {
        let mut analyzer = Self {
            config,
            lexicon: HashMap::new(),
        };
        analyzer.populate_builtin_lexicon();
        analyzer
    }

    /// Add or replace a single lexicon entry (builder pattern).
    pub fn with_lexicon_entry(mut self, entry: LexiconEntry) -> Self {
        self.lexicon.insert(entry.word.clone(), entry);
        self
    }

    // ------------------------------------------------------------------
    // Core analysis
    // ------------------------------------------------------------------

    /// Analyse a single document and return a `SentimentResult`.
    pub fn analyze(&self, text_id: String, text: &str) -> SentimentResult {
        let tokens = tokenize(text);
        let word_count = tokens.len();

        // --- document-level accumulation ---
        let (raw_pos, raw_neg, sentiment_word_count) = self.accumulate_scores(&tokens);

        let overall = self.build_score(raw_pos, raw_neg, word_count);

        // --- aspect detection ---
        let aspect_keywords: Vec<String> = self
            .config
            .aspect_keywords
            .iter()
            .map(|k| k.to_lowercase())
            .collect();

        let mut aspects = Vec::new();
        for (idx, token) in tokens.iter().enumerate() {
            if aspect_keywords.contains(token) {
                let start = idx.saturating_sub(self.config.window_size);
                let end = (idx + self.config.window_size + 1).min(tokens.len());
                let window: Vec<String> = tokens[start..end].to_vec();

                let (a_pos, a_neg, _) = self.accumulate_scores(&window);
                let a_score = self.build_score(a_pos, a_neg, window.len());

                // Collect the tokens that had actual sentiment signal as "mentions"
                let mentions: Vec<String> = window
                    .iter()
                    .filter(|t| {
                        self.lexicon
                            .get(*t)
                            .is_some_and(|e| e.positive_score > 0.0 || e.negative_score > 0.0)
                    })
                    .cloned()
                    .collect();

                aspects.push(AspectSentiment {
                    aspect: token.clone(),
                    sentiment: a_score,
                    mentions,
                });
            }
        }

        SentimentResult {
            text_id,
            overall,
            aspects,
            word_count,
            sentiment_word_count,
        }
    }

    /// Analyse a batch of `(text_id, text)` pairs.
    pub fn batch_analyze(&self, texts: &[(String, String)]) -> Vec<SentimentResult> {
        texts
            .iter()
            .map(|(id, text)| self.analyze(id.clone(), text))
            .collect()
    }

    // ------------------------------------------------------------------
    // Aggregation / ranking helpers
    // ------------------------------------------------------------------

    /// Return the top `n` results ordered by compound score (most positive first).
    pub fn top_positive<'a>(
        &self,
        results: &'a [SentimentResult],
        n: usize,
    ) -> Vec<&'a SentimentResult> {
        let mut sorted: Vec<&SentimentResult> = results.iter().collect();
        sorted.sort_by(|a, b| {
            b.overall
                .compound
                .partial_cmp(&a.overall.compound)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted.truncate(n);
        sorted
    }

    /// Return the `n` most negative results (lowest compound score first).
    pub fn top_negative<'a>(
        &self,
        results: &'a [SentimentResult],
        n: usize,
    ) -> Vec<&'a SentimentResult> {
        let mut sorted: Vec<&SentimentResult> = results.iter().collect();
        sorted.sort_by(|a, b| {
            a.overall
                .compound
                .partial_cmp(&b.overall.compound)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted.truncate(n);
        sorted
    }

    /// Compute the mean `SentimentScore` across a slice of results.
    ///
    /// Returns `SentimentScore::zero()` for an empty slice.
    pub fn aggregate_sentiment(&self, results: &[SentimentResult]) -> SentimentScore {
        if results.is_empty() {
            return SentimentScore::zero();
        }
        let n = results.len() as f64;
        let pos = results.iter().map(|r| r.overall.positive).sum::<f64>() / n;
        let neg = results.iter().map(|r| r.overall.negative).sum::<f64>() / n;
        let neu = results.iter().map(|r| r.overall.neutral).sum::<f64>() / n;
        let cmp = results.iter().map(|r| r.overall.compound).sum::<f64>() / n;
        SentimentScore {
            positive: pos,
            negative: neg,
            neutral: neu,
            compound: cmp,
        }
    }

    /// Compute aggregate statistics for a collection of results.
    pub fn stats(&self, results: &[SentimentResult]) -> SentimentAnalyzerStats {
        let total_analyzed = results.len();
        let mut positive_count = 0usize;
        let mut negative_count = 0usize;
        let mut neutral_count = 0usize;
        let mut mixed_count = 0usize;
        let mut compound_sum = 0.0_f64;

        for r in results {
            compound_sum += r.overall.compound;
            match r.overall.polarity() {
                SentimentPolarity::Positive => positive_count += 1,
                SentimentPolarity::Negative => negative_count += 1,
                SentimentPolarity::Neutral => neutral_count += 1,
                SentimentPolarity::Mixed => mixed_count += 1,
            }
        }

        let avg_compound = if total_analyzed > 0 {
            compound_sum / total_analyzed as f64
        } else {
            0.0
        };

        SentimentAnalyzerStats {
            total_analyzed,
            positive_count,
            negative_count,
            neutral_count,
            mixed_count,
            avg_compound,
        }
    }

    // ------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------

    /// Walk through all tokens and accumulate raw positive/negative scores,
    /// applying negation and intensity scaling. Returns `(raw_pos, raw_neg,
    /// sentiment_word_count)`.
    fn accumulate_scores(&self, tokens: &[String]) -> (f64, f64, usize) {
        let mut raw_pos = 0.0_f64;
        let mut raw_neg = 0.0_f64;
        let mut sentiment_word_count = 0usize;

        for (i, token) in tokens.iter().enumerate() {
            let entry = match self.lexicon.get(token) {
                Some(e) => e,
                None => continue,
            };

            // Skip pure modifiers (intensifiers / negations) — they are
            // accounted for through context windows of *other* tokens.
            if entry.positive_score == 0.0 && entry.negative_score == 0.0 {
                continue;
            }

            sentiment_word_count += 1;

            // Context window *before* this token for negation / intensity.
            let ctx_start = i.saturating_sub(self.config.negation_window);
            let preceding: Vec<String> = tokens[ctx_start..i].to_vec();

            let negated = contains_negation(&preceding);
            let intensity = find_intensity_multiplier(&preceding, &self.lexicon);

            let mut pos = entry.positive_score * intensity;
            let mut neg = entry.negative_score * intensity;

            if negated {
                // Flip: positive becomes negative and vice-versa.
                std::mem::swap(&mut pos, &mut neg);
            }

            raw_pos += pos;
            raw_neg += neg;
        }

        (raw_pos, raw_neg, sentiment_word_count)
    }

    /// Convert raw accumulated scores into a normalised `SentimentScore`.
    fn build_score(&self, raw_pos: f64, raw_neg: f64, word_count: usize) -> SentimentScore {
        let wc = word_count.max(1) as f64;

        // Normalise by word count so scores are comparable across document lengths.
        let pos = (raw_pos / wc).min(1.0_f64);
        let neg = (raw_neg / wc).min(1.0_f64);
        let neu = (1.0_f64 - pos - neg).max(0.0_f64);

        let denom = (pos + neg + neu + 0.001_f64).max(0.001_f64);
        let compound = (pos - neg) / denom;

        SentimentScore {
            positive: pos,
            negative: neg,
            neutral: neu,
            compound,
        }
    }

    /// Populate the built-in lexicon covering common positive/negative words,
    /// intensifiers, diminishers, and negation words.
    fn populate_builtin_lexicon(&mut self) {
        // ---- Positive words ----
        let positive_words = [
            ("good", 0.7),
            ("great", 0.85),
            ("excellent", 0.95),
            ("amazing", 0.90),
            ("wonderful", 0.90),
            ("fantastic", 0.95),
            ("love", 0.85),
            ("best", 0.90),
            ("perfect", 1.00),
            ("outstanding", 0.95),
            ("brilliant", 0.90),
            ("superb", 0.90),
            ("incredible", 0.90),
            ("awesome", 0.85),
            ("positive", 0.65),
            ("helpful", 0.70),
            ("fast", 0.60),
            ("reliable", 0.70),
            ("efficient", 0.70),
            ("clear", 0.55),
            ("smooth", 0.60),
            ("easy", 0.60),
            ("simple", 0.55),
            ("pleasant", 0.65),
            ("satisfied", 0.70),
            ("happy", 0.80),
            ("delighted", 0.85),
            ("impressed", 0.75),
            ("accurate", 0.65),
            ("responsive", 0.65),
            ("innovative", 0.70),
            ("intuitive", 0.65),
        ];

        for (word, score) in &positive_words {
            self.lexicon
                .insert(word.to_string(), LexiconEntry::positive(*word, *score));
        }

        // ---- Negative words ----
        let negative_words = [
            ("bad", 0.70),
            ("terrible", 0.90),
            ("awful", 0.90),
            ("horrible", 0.95),
            ("hate", 0.85),
            ("worst", 0.95),
            ("poor", 0.65),
            ("broken", 0.80),
            ("slow", 0.60),
            ("difficult", 0.55),
            ("frustrating", 0.80),
            ("disappointing", 0.75),
            ("unreliable", 0.75),
            ("complex", 0.50),
            ("confusing", 0.65),
            ("annoying", 0.70),
            ("useless", 0.85),
            ("failed", 0.80),
            ("error", 0.65),
            ("problem", 0.60),
            ("issue", 0.50),
            ("bug", 0.65),
            ("crash", 0.85),
            ("delay", 0.60),
            ("expensive", 0.60),
            ("lacking", 0.55),
            ("outdated", 0.55),
            ("clunky", 0.65),
        ];

        for (word, score) in &negative_words {
            self.lexicon
                .insert(word.to_string(), LexiconEntry::negative(*word, *score));
        }

        // ---- Intensifiers ----
        let intensifiers = [
            ("very", 1.5),
            ("extremely", 1.8),
            ("incredibly", 1.7),
            ("absolutely", 1.8),
            ("totally", 1.6),
            ("highly", 1.5),
            ("really", 1.4),
            ("deeply", 1.5),
            ("utterly", 1.7),
            ("truly", 1.4),
        ];

        for (word, mult) in &intensifiers {
            self.lexicon
                .insert(word.to_string(), LexiconEntry::modifier(*word, *mult));
        }

        // ---- Diminishers ----
        let diminishers = [
            ("slightly", 0.5),
            ("somewhat", 0.6),
            ("barely", 0.3),
            ("hardly", 0.3),
            ("rarely", 0.4),
            ("mildly", 0.5),
            ("partially", 0.6),
            ("almost", 0.7),
        ];

        for (word, mult) in &diminishers {
            self.lexicon
                .insert(word.to_string(), LexiconEntry::modifier(*word, *mult));
        }

        // ---- Negations (stored as pure modifiers with intensifier = 1.0;
        //      negation logic is handled separately via NEGATIONS constant) ----
        let negations = [
            "not", "never", "no", "isn't", "wasn't", "aren't", "weren't", "doesn't", "didn't",
            "don't", "nor", "neither", "without",
        ];
        for word in &negations {
            self.lexicon
                .entry(word.to_string())
                .or_insert_with(|| LexiconEntry::modifier(*word, 1.0));
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::{
        tokenize, AspectSentiment, LexiconEntry, SentimentAnalyzer, SentimentConfig,
        SentimentPolarity, SentimentResult, SentimentScore,
    };

    // ------ helpers -------------------------------------------------------

    fn default_analyzer() -> SentimentAnalyzer {
        SentimentAnalyzer::new(SentimentConfig::default())
    }

    // ------ SentimentScore unit tests ------------------------------------

    #[test]
    fn test_sentiment_score_zero_is_neutral() {
        let s = SentimentScore::zero();
        assert_eq!(s.polarity(), SentimentPolarity::Neutral);
    }

    #[test]
    fn test_polarity_positive_compound() {
        let s = SentimentScore {
            positive: 0.8,
            negative: 0.0,
            neutral: 0.2,
            compound: 0.8,
        };
        assert_eq!(s.polarity(), SentimentPolarity::Positive);
    }

    #[test]
    fn test_polarity_negative_compound() {
        let s = SentimentScore {
            positive: 0.0,
            negative: 0.8,
            neutral: 0.2,
            compound: -0.8,
        };
        assert_eq!(s.polarity(), SentimentPolarity::Negative);
    }

    #[test]
    fn test_polarity_mixed_both_significant() {
        let s = SentimentScore {
            positive: 0.4,
            negative: 0.3,
            neutral: 0.3,
            compound: 0.1,
        };
        // Both pos and neg > 0.1 → Mixed
        assert_eq!(s.polarity(), SentimentPolarity::Mixed);
    }

    #[test]
    fn test_polarity_neutral_small_compound() {
        let s = SentimentScore {
            positive: 0.02,
            negative: 0.01,
            neutral: 0.97,
            compound: 0.02,
        };
        assert_eq!(s.polarity(), SentimentPolarity::Neutral);
    }

    #[test]
    fn test_compound_range_positive() {
        let s = SentimentScore {
            positive: 0.9,
            negative: 0.0,
            neutral: 0.1,
            compound: 0.9,
        };
        assert!(s.compound > 0.05, "should be positive");
    }

    #[test]
    fn test_compound_range_negative() {
        let s = SentimentScore {
            positive: 0.0,
            negative: 0.9,
            neutral: 0.1,
            compound: -0.9,
        };
        assert!(s.compound < -0.05, "should be negative");
    }

    // ------ tokenize unit tests ------------------------------------------

    #[test]
    fn test_tokenize_basic() {
        let tokens = tokenize("Hello, World!");
        assert_eq!(tokens, vec!["hello", "world"]);
    }

    #[test]
    fn test_tokenize_contractions_preserved() {
        let tokens = tokenize("it isn't broken");
        assert!(tokens.contains(&"isn't".to_string()));
    }

    #[test]
    fn test_tokenize_empty_string() {
        assert!(tokenize("").is_empty());
    }

    #[test]
    fn test_tokenize_punctuation_only() {
        assert!(tokenize("... --- !!!").is_empty());
    }

    #[test]
    fn test_tokenize_lowercases() {
        let tokens = tokenize("GOOD BAD");
        assert_eq!(tokens, vec!["good", "bad"]);
    }

    // ------ LexiconEntry constructors ------------------------------------

    #[test]
    fn test_lexicon_entry_positive() {
        let e = LexiconEntry::positive("good", 0.7);
        assert_eq!(e.word, "good");
        assert!(e.positive_score > 0.0);
        assert_eq!(e.negative_score, 0.0);
        assert_eq!(e.intensifier, 1.0);
    }

    #[test]
    fn test_lexicon_entry_negative() {
        let e = LexiconEntry::negative("bad", 0.7);
        assert_eq!(e.positive_score, 0.0);
        assert!(e.negative_score > 0.0);
    }

    #[test]
    fn test_lexicon_entry_modifier_intensifier() {
        let e = LexiconEntry::modifier("very", 1.5);
        assert_eq!(e.positive_score, 0.0);
        assert_eq!(e.negative_score, 0.0);
        assert_eq!(e.intensifier, 1.5);
    }

    #[test]
    fn test_lexicon_entry_modifier_diminisher() {
        let e = LexiconEntry::modifier("slightly", 0.5);
        assert_eq!(e.intensifier, 0.5);
    }

    // ------ SentimentAnalyzer::new & lexicon coverage --------------------

    #[test]
    fn test_new_has_positive_words() {
        let a = default_analyzer();
        assert!(a.lexicon.contains_key("good"));
        assert!(a.lexicon.contains_key("excellent"));
    }

    #[test]
    fn test_new_has_negative_words() {
        let a = default_analyzer();
        assert!(a.lexicon.contains_key("bad"));
        assert!(a.lexicon.contains_key("terrible"));
    }

    #[test]
    fn test_new_has_intensifiers() {
        let a = default_analyzer();
        let e = a.lexicon.get("very").expect("very must exist");
        assert!(e.intensifier > 1.0);
    }

    #[test]
    fn test_new_has_diminishers() {
        let a = default_analyzer();
        let e = a.lexicon.get("slightly").expect("slightly must exist");
        assert!(e.intensifier < 1.0);
    }

    #[test]
    fn test_lexicon_has_at_least_60_entries() {
        let a = default_analyzer();
        assert!(
            a.lexicon.len() >= 60,
            "expected ≥60 entries, got {}",
            a.lexicon.len()
        );
    }

    // ------ with_lexicon_entry -------------------------------------------

    #[test]
    fn test_with_lexicon_entry_adds_word() {
        let a = default_analyzer().with_lexicon_entry(LexiconEntry::positive("stellar", 0.9));
        assert!(a.lexicon.contains_key("stellar"));
    }

    #[test]
    fn test_with_lexicon_entry_overrides_existing() {
        let a = default_analyzer().with_lexicon_entry(LexiconEntry::positive("good", 0.999));
        let e = a.lexicon.get("good").expect("good must exist");
        assert!((e.positive_score - 0.999).abs() < 1e-9);
    }

    // ------ analyze: polarity classification ----------------------------

    #[test]
    fn test_analyze_positive_text() {
        let a = default_analyzer();
        let r = a.analyze(
            "t1".to_string(),
            "This is absolutely excellent and amazing!",
        );
        assert_eq!(r.overall.polarity(), SentimentPolarity::Positive);
    }

    #[test]
    fn test_analyze_negative_text() {
        let a = default_analyzer();
        let r = a.analyze("t1".to_string(), "The service is terrible and frustrating.");
        assert_eq!(r.overall.polarity(), SentimentPolarity::Negative);
    }

    #[test]
    fn test_analyze_neutral_text() {
        let a = default_analyzer();
        let r = a.analyze("t1".to_string(), "The document is a plain text file.");
        assert_eq!(r.overall.polarity(), SentimentPolarity::Neutral);
    }

    // ------ analyze: word counts ----------------------------------------

    #[test]
    fn test_analyze_word_count() {
        let a = default_analyzer();
        let r = a.analyze("t1".to_string(), "good bad");
        assert_eq!(r.word_count, 2);
    }

    #[test]
    fn test_analyze_sentiment_word_count_non_zero_for_sentiment_text() {
        let a = default_analyzer();
        let r = a.analyze("t1".to_string(), "excellent");
        assert!(r.sentiment_word_count > 0);
    }

    #[test]
    fn test_analyze_empty_text() {
        let a = default_analyzer();
        let r = a.analyze("t1".to_string(), "");
        assert_eq!(r.word_count, 0);
        assert_eq!(r.sentiment_word_count, 0);
    }

    // ------ negation handling -------------------------------------------

    #[test]
    fn test_negation_flips_positive_to_negative() {
        let a = default_analyzer();
        let pos = a.analyze("pos".to_string(), "excellent");
        let neg = a.analyze("neg".to_string(), "not excellent");
        // After negation the compound should be lower (more negative)
        assert!(
            neg.overall.compound < pos.overall.compound,
            "negation should reduce compound: {} vs {}",
            neg.overall.compound,
            pos.overall.compound
        );
    }

    #[test]
    fn test_negation_flips_negative_to_positive() {
        let a = default_analyzer();
        let base = a.analyze("base".to_string(), "terrible");
        let neg = a.analyze("neg".to_string(), "not terrible");
        assert!(neg.overall.compound > base.overall.compound);
    }

    #[test]
    fn test_contraction_negation() {
        let a = default_analyzer();
        let r = a.analyze("t1".to_string(), "isn't broken");
        // "broken" is negative; "isn't" negates → positive compound
        assert!(r.overall.compound >= 0.0);
    }

    // ------ intensifier / diminisher handling ---------------------------

    #[test]
    fn test_intensifier_boosts_positive() {
        // Compare with equal word counts so normalisation is comparable.
        // "extremely good word" vs "neutral good word" — the intensifier should
        // produce a higher raw positive contribution even after dividing by wc=3.
        let a = default_analyzer();
        let base = a.analyze("base".to_string(), "the good thing");
        let boosted = a.analyze("boosted".to_string(), "extremely good thing");
        // The boosted text uses an intensifier, so its positive score must be higher.
        assert!(
            boosted.overall.positive >= base.overall.positive,
            "intensifier should boost positive: base={}, boosted={}",
            base.overall.positive,
            boosted.overall.positive
        );
    }

    #[test]
    fn test_diminisher_reduces_sentiment() {
        let a = default_analyzer();
        let base = a.analyze("base".to_string(), "good");
        let reduced = a.analyze("reduced".to_string(), "slightly good");
        assert!(reduced.overall.positive <= base.overall.positive);
    }

    // ------ aspect detection --------------------------------------------

    #[test]
    fn test_aspect_detected_for_keyword() {
        let a = default_analyzer();
        let r = a.analyze("t1".to_string(), "The service is excellent and fast.");
        let service_aspects: Vec<&AspectSentiment> = r
            .aspects
            .iter()
            .filter(|asp| asp.aspect == "service")
            .collect();
        assert!(!service_aspects.is_empty(), "expected a 'service' aspect");
    }

    #[test]
    fn test_aspect_not_detected_for_missing_keyword() {
        let a = default_analyzer();
        let r = a.analyze("t1".to_string(), "Everything is fine.");
        assert!(r.aspects.is_empty());
    }

    #[test]
    fn test_aspect_mentions_non_empty() {
        let a = default_analyzer();
        let r = a.analyze("t1".to_string(), "The quality is great and reliable.");
        let aspect = r.aspects.iter().find(|a| a.aspect == "quality");
        assert!(aspect.is_some());
        if let Some(asp) = aspect {
            // At least "great" or "reliable" should be in mentions
            assert!(!asp.mentions.is_empty(), "mentions should not be empty");
        }
    }

    #[test]
    fn test_aspect_multiple_occurrences() {
        let a = default_analyzer();
        let r = a.analyze(
            "t1".to_string(),
            "The performance is great. Performance is also reliable.",
        );
        let perf_count = r
            .aspects
            .iter()
            .filter(|a| a.aspect == "performance")
            .count();
        assert_eq!(perf_count, 2, "two occurrences of 'performance'");
    }

    // ------ batch_analyze -----------------------------------------------

    #[test]
    fn test_batch_analyze_returns_correct_count() {
        let a = default_analyzer();
        let texts = vec![
            ("t1".to_string(), "great product".to_string()),
            ("t2".to_string(), "terrible service".to_string()),
            ("t3".to_string(), "average experience".to_string()),
        ];
        let results = a.batch_analyze(&texts);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_batch_analyze_empty() {
        let a = default_analyzer();
        let results = a.batch_analyze(&[]);
        assert!(results.is_empty());
    }

    #[test]
    fn test_batch_analyze_preserves_text_ids() {
        let a = default_analyzer();
        let texts = vec![
            ("id_one".to_string(), "good".to_string()),
            ("id_two".to_string(), "bad".to_string()),
        ];
        let results = a.batch_analyze(&texts);
        assert_eq!(results[0].text_id, "id_one");
        assert_eq!(results[1].text_id, "id_two");
    }

    // ------ top_positive / top_negative ---------------------------------

    #[test]
    fn test_top_positive_ordering() {
        let a = default_analyzer();
        let texts = vec![
            ("neg".to_string(), "horrible terrible awful".to_string()),
            ("pos".to_string(), "excellent amazing perfect".to_string()),
            ("neu".to_string(), "the cat sat on a mat".to_string()),
        ];
        let results = a.batch_analyze(&texts);
        let top = a.top_positive(&results, 1);
        assert_eq!(top[0].text_id, "pos");
    }

    #[test]
    fn test_top_negative_ordering() {
        let a = default_analyzer();
        let texts = vec![
            ("neg".to_string(), "horrible terrible awful".to_string()),
            ("pos".to_string(), "excellent amazing perfect".to_string()),
        ];
        let results = a.batch_analyze(&texts);
        let bottom = a.top_negative(&results, 1);
        assert_eq!(bottom[0].text_id, "neg");
    }

    #[test]
    fn test_top_positive_n_larger_than_results() {
        let a = default_analyzer();
        let texts = vec![("t1".to_string(), "good".to_string())];
        let results = a.batch_analyze(&texts);
        let top = a.top_positive(&results, 100);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_top_negative_n_larger_than_results() {
        let a = default_analyzer();
        let texts = vec![("t1".to_string(), "bad".to_string())];
        let results = a.batch_analyze(&texts);
        let bottom = a.top_negative(&results, 100);
        assert_eq!(bottom.len(), 1);
    }

    // ------ aggregate_sentiment -----------------------------------------

    #[test]
    fn test_aggregate_empty_returns_zero() {
        let a = default_analyzer();
        let agg = a.aggregate_sentiment(&[]);
        assert_eq!(agg.polarity(), SentimentPolarity::Neutral);
    }

    #[test]
    fn test_aggregate_single_equals_result() {
        let a = default_analyzer();
        let r = a.analyze("t1".to_string(), "excellent");
        let agg = a.aggregate_sentiment(std::slice::from_ref(&r));
        assert!((agg.compound - r.overall.compound).abs() < 1e-9);
    }

    #[test]
    fn test_aggregate_mixed_batch() {
        let a = default_analyzer();
        let texts = vec![
            (
                "pos".to_string(),
                "excellent perfect outstanding".to_string(),
            ),
            ("neg".to_string(), "terrible awful horrible".to_string()),
        ];
        let results = a.batch_analyze(&texts);
        let agg = a.aggregate_sentiment(&results);
        // The average should be near zero but the components exist
        assert!(agg.positive > 0.0);
        assert!(agg.negative > 0.0);
    }

    // ------ stats -------------------------------------------------------

    #[test]
    fn test_stats_empty() {
        let a = default_analyzer();
        let s = a.stats(&[]);
        assert_eq!(s.total_analyzed, 0);
        assert_eq!(s.avg_compound, 0.0);
    }

    #[test]
    fn test_stats_counts_positive() {
        let a = default_analyzer();
        let texts = vec![
            ("t1".to_string(), "excellent amazing".to_string()),
            ("t2".to_string(), "great wonderful".to_string()),
        ];
        let results = a.batch_analyze(&texts);
        let s = a.stats(&results);
        assert!(s.positive_count > 0);
    }

    #[test]
    fn test_stats_counts_negative() {
        let a = default_analyzer();
        let texts = vec![("t1".to_string(), "terrible awful".to_string())];
        let results = a.batch_analyze(&texts);
        let s = a.stats(&results);
        assert!(s.negative_count > 0);
    }

    #[test]
    fn test_stats_total_analyzed() {
        let a = default_analyzer();
        let texts: Vec<(String, String)> = (0..7)
            .map(|i| (format!("t{}", i), "good".to_string()))
            .collect();
        let results = a.batch_analyze(&texts);
        let s = a.stats(&results);
        assert_eq!(s.total_analyzed, 7);
    }

    #[test]
    fn test_stats_avg_compound_single() {
        let a = default_analyzer();
        let r = a.analyze("t1".to_string(), "excellent");
        let compound = r.overall.compound;
        let s = a.stats(&[r]);
        assert!((s.avg_compound - compound).abs() < 1e-9);
    }

    // ------ SentimentResult fields --------------------------------------

    #[test]
    fn test_result_text_id_preserved() {
        let a = default_analyzer();
        let r = a.analyze("my-unique-id".to_string(), "good");
        assert_eq!(r.text_id, "my-unique-id");
    }

    #[test]
    fn test_result_compound_in_valid_range() {
        let a = default_analyzer();
        let texts = [
            "absolutely fantastic amazing wonderful",
            "horrible terrible awful crash",
            "the quick brown fox jumps",
        ];
        for text in &texts {
            let r = a.analyze("t".to_string(), text);
            assert!(
                r.overall.compound >= -1.0 && r.overall.compound <= 1.0,
                "compound out of range for {:?}: {}",
                text,
                r.overall.compound
            );
        }
    }

    // ------ custom config -----------------------------------------------

    #[test]
    fn test_custom_aspect_keywords() {
        let config = SentimentConfig {
            aspect_keywords: vec!["battery".to_string(), "camera".to_string()],
            ..SentimentConfig::default()
        };
        let a = SentimentAnalyzer::new(config);
        let r = a.analyze(
            "t1".to_string(),
            "The battery is amazing but the camera is slow.",
        );
        let aspects: Vec<&str> = r.aspects.iter().map(|a| a.aspect.as_str()).collect();
        assert!(aspects.contains(&"battery"));
        assert!(aspects.contains(&"camera"));
    }

    #[test]
    fn test_custom_window_size_zero() {
        let config = SentimentConfig {
            window_size: 0,
            ..SentimentConfig::default()
        };
        let a = SentimentAnalyzer::new(config);
        let r = a.analyze("t1".to_string(), "quality is great");
        // aspect should still be found (zero window = only the aspect token itself)
        assert!(!r.aspects.is_empty());
    }

    #[test]
    fn test_stats_mixed_counted_correctly() {
        // Build a text whose score will be Mixed
        let a = default_analyzer();
        // Craft a result manually to control scores precisely
        let result = SentimentResult {
            text_id: "m1".to_string(),
            overall: SentimentScore {
                positive: 0.3,
                negative: 0.3,
                neutral: 0.4,
                compound: 0.0,
            },
            aspects: vec![],
            word_count: 10,
            sentiment_word_count: 4,
        };
        let s = a.stats(&[result]);
        assert_eq!(s.mixed_count, 1);
    }

    #[test]
    fn test_stats_neutral_counted_correctly() {
        let a = default_analyzer();
        let result = SentimentResult {
            text_id: "n1".to_string(),
            overall: SentimentScore {
                positive: 0.02,
                negative: 0.01,
                neutral: 0.97,
                compound: 0.01,
            },
            aspects: vec![],
            word_count: 5,
            sentiment_word_count: 0,
        };
        let s = a.stats(&[result]);
        assert_eq!(s.neutral_count, 1);
    }
}
