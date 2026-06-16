//! # Multilingual Text Normalizer
//!
//! Production-grade text normalization for multilingual content supporting:
//! - Unicode normalization (combining character removal)
//! - Script detection across Latin, Cyrillic, Arabic, CJK, and Devanagari scripts
//! - Language hint detection from script analysis
//! - Script-aware tokenization strategies
//! - Configurable normalization pipelines

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

// ─────────────────────────────────────────────────────────────────────────────
// Script enum
// ─────────────────────────────────────────────────────────────────────────────

/// Unicode script classification for a run of characters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Script {
    /// Latin script (A-Z, a-z, extended Latin U+00C0-U+024F)
    Latin,
    /// Cyrillic script (U+0400-U+04FF)
    Cyrillic,
    /// Arabic script (U+0600-U+06FF)
    Arabic,
    /// CJK unified ideographs and CJK punctuation
    CJK,
    /// Devanagari script (U+0900-U+097F)
    Devanagari,
    /// Unknown / not classified
    Unknown,
}

impl Script {
    /// Human-readable name for display purposes.
    pub fn name(&self) -> &'static str {
        match self {
            Script::Latin => "Latin",
            Script::Cyrillic => "Cyrillic",
            Script::Arabic => "Arabic",
            Script::CJK => "CJK",
            Script::Devanagari => "Devanagari",
            Script::Unknown => "Unknown",
        }
    }
}

/// Classify a single Unicode code point into a [`Script`].
fn char_script(c: char) -> Script {
    let cp = c as u32;
    match cp {
        0x41..=0x7A | 0xC0..=0x24F => Script::Latin,
        0x400..=0x4FF => Script::Cyrillic,
        0x600..=0x6FF => Script::Arabic,
        0x4E00..=0x9FFF | 0x3000..=0x303F => Script::CJK,
        0x900..=0x97F => Script::Devanagari,
        _ => Script::Unknown,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// LanguageHint enum
// ─────────────────────────────────────────────────────────────────────────────

/// High-level language hint inferred from script analysis.
///
/// This is a heuristic approximation — it should not be used as a
/// production-quality language identifier for all use cases.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LanguageHint {
    English,
    Russian,
    Arabic,
    Chinese,
    Japanese,
    Hindi,
    Unknown,
}

impl LanguageHint {
    /// BCP-47 language tag approximation.
    pub fn bcp47(&self) -> &'static str {
        match self {
            LanguageHint::English => "en",
            LanguageHint::Russian => "ru",
            LanguageHint::Arabic => "ar",
            LanguageHint::Chinese => "zh",
            LanguageHint::Japanese => "ja",
            LanguageHint::Hindi => "hi",
            LanguageHint::Unknown => "und",
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// NormalizationOptions
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration flags that control the normalization pipeline.
#[derive(Debug, Clone)]
pub struct NormalizationOptions {
    /// Convert all characters to lowercase.
    pub lowercase: bool,
    /// Remove combining diacritical marks (U+0300-U+036F).
    pub strip_accents: bool,
    /// Collapse multiple consecutive whitespace characters to a single space
    /// and trim leading/trailing whitespace.
    pub normalize_whitespace: bool,
    /// Remove non-alphanumeric, non-space characters.
    pub remove_punctuation: bool,
    /// Truncate output to at most this many Unicode scalar values.
    pub max_length: Option<usize>,
    /// When `Some(script)`, retain only characters belonging to `script` plus
    /// ASCII space (U+0020).
    pub script_filter: Option<Script>,
}

impl Default for NormalizationOptions {
    fn default() -> Self {
        Self {
            lowercase: true,
            strip_accents: false,
            normalize_whitespace: true,
            remove_punctuation: false,
            max_length: None,
            script_filter: None,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TokenizationStrategy
// ─────────────────────────────────────────────────────────────────────────────

/// Strategy that controls how normalized text is split into tokens.
#[derive(Debug, Clone)]
pub enum TokenizationStrategy {
    /// Split on Unicode whitespace boundaries; empty tokens are discarded.
    Whitespace,
    /// Sliding window of `n` characters with a step size of 1.
    CharacterNgram {
        /// Width of each n-gram window.
        n: usize,
    },
    /// Simplified BPE approximation: whitespace split, then tokens longer than
    /// 10 characters are chunked into pieces of 5 characters each.
    Subword {
        /// Target vocabulary size (used for future extensions; currently
        /// determines the chunk length = max(3, vocab_size / 1000)).
        vocab_size: usize,
    },
    /// Contiguous runs of the same [`Script`] form individual tokens;
    /// whitespace acts as an additional boundary.
    ScriptAware,
}

// ─────────────────────────────────────────────────────────────────────────────
// NormalizedText
// ─────────────────────────────────────────────────────────────────────────────

/// The output produced by [`MultilingualNormalizer::normalize`].
#[derive(Debug, Clone)]
pub struct NormalizedText {
    /// Verbatim copy of the input string.
    pub original: String,
    /// Text after the normalization pipeline has been applied.
    pub normalized: String,
    /// Dominant script detected in `original`.
    pub detected_script: Script,
    /// Language inferred from the dominant script.
    pub language_hint: LanguageHint,
    /// Tokens produced by the configured [`TokenizationStrategy`].
    pub tokens: Vec<String>,
    /// Number of Unicode scalar values in `normalized`.
    pub char_count: usize,
    /// Number of tokens in `tokens`.
    pub token_count: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// NormalizerRunStats (internal atomic counters)
// ─────────────────────────────────────────────────────────────────────────────

/// Thread-safe, atomically updated runtime statistics.
#[derive(Debug, Default)]
struct NormalizerRunStats {
    total_processed: AtomicU64,
    total_chars_removed: AtomicU64,
    total_tokens_generated: AtomicU64,
}

impl NormalizerRunStats {
    fn record(&self, original_chars: u64, normalized_chars: u64, tokens: u64) {
        self.total_processed.fetch_add(1, Ordering::Relaxed);
        let removed = original_chars.saturating_sub(normalized_chars);
        self.total_chars_removed
            .fetch_add(removed, Ordering::Relaxed);
        self.total_tokens_generated
            .fetch_add(tokens, Ordering::Relaxed);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// NormalizerStats (public snapshot)
// ─────────────────────────────────────────────────────────────────────────────

/// A point-in-time snapshot of normalizer run statistics.
#[derive(Debug, Clone)]
pub struct NormalizerStats {
    /// Total number of `normalize` calls that have completed.
    pub total_processed: u64,
    /// Cumulative number of characters removed by the normalization pipeline.
    pub total_chars_removed: u64,
    /// Cumulative number of tokens produced across all `normalize` calls.
    pub total_tokens_generated: u64,
    /// Ratio of removed characters to original characters, averaged over all
    /// processed inputs.  Returns `0.0` when nothing has been processed yet.
    pub avg_compression_ratio: f64,
}

// ─────────────────────────────────────────────────────────────────────────────
// MultilingualNormalizer
// ─────────────────────────────────────────────────────────────────────────────

/// Production-grade multilingual text normalizer.
///
/// `MultilingualNormalizer` executes a configurable normalization pipeline on
/// arbitrary Unicode text and tokenizes the result according to the chosen
/// [`TokenizationStrategy`].  It is internally thread-safe via atomic counters
/// wrapped in an [`Arc`], so a single instance may be shared across threads.
///
/// # Example
///
/// ```rust
/// use ipfrs_semantic::multilingual_normalizer::{
///     MultilingualNormalizer, NormalizationOptions, TokenizationStrategy,
/// };
///
/// let opts  = NormalizationOptions { lowercase: true, ..Default::default() };
/// let norm  = MultilingualNormalizer::new(opts, TokenizationStrategy::Whitespace);
/// let result = norm.normalize("Hello World");
/// assert_eq!(result.normalized, "hello world");
/// assert_eq!(result.token_count, 2);
/// ```
pub struct MultilingualNormalizer {
    /// Active normalization options.
    pub options: NormalizationOptions,
    /// Active tokenization strategy.
    pub tokenization: TokenizationStrategy,
    stats: Arc<NormalizerRunStats>,
}

impl std::fmt::Debug for MultilingualNormalizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MultilingualNormalizer")
            .field("options", &self.options)
            .field("tokenization", &self.tokenization)
            .finish()
    }
}

impl MultilingualNormalizer {
    // ── Construction ───────────────────────────────────────────────────────

    /// Create a new normalizer with the given options and tokenization strategy.
    pub fn new(options: NormalizationOptions, tokenization: TokenizationStrategy) -> Self {
        Self {
            options,
            tokenization,
            stats: Arc::new(NormalizerRunStats::default()),
        }
    }

    /// Create a normalizer with default options and `Whitespace` tokenization.
    pub fn with_defaults() -> Self {
        Self::new(
            NormalizationOptions::default(),
            TokenizationStrategy::Whitespace,
        )
    }

    // ── Public API ─────────────────────────────────────────────────────────

    /// Normalize a single text string and tokenize it.
    ///
    /// The pipeline is:
    /// 1. Detect dominant script.
    /// 2. Derive a language hint from the script.
    /// 3. Apply all enabled normalization options in order.
    /// 4. Tokenize according to the configured strategy.
    /// 5. Update internal statistics.
    pub fn normalize(&self, text: &str) -> NormalizedText {
        let original_chars = text.chars().count() as u64;

        let detected_script = self.detect_script(text);
        let language_hint = script_to_language_hint(detected_script);

        let normalized = self.apply_options(text);
        let normalized_chars = normalized.chars().count() as u64;
        let char_count = normalized_chars as usize;

        let tokens = self.tokenize(&normalized);
        let token_count = tokens.len();

        self.stats
            .record(original_chars, normalized_chars, token_count as u64);

        NormalizedText {
            original: text.to_owned(),
            normalized,
            detected_script,
            language_hint,
            tokens,
            char_count,
            token_count,
        }
    }

    /// Normalize a batch of text strings.
    pub fn normalize_batch(&self, texts: &[&str]) -> Vec<NormalizedText> {
        texts.iter().map(|t| self.normalize(t)).collect()
    }

    /// Detect the dominant script in `text` by counting characters per script.
    ///
    /// Returns [`Script::Unknown`] when the text is empty or when two scripts
    /// tie for the highest count.
    pub fn detect_script(&self, text: &str) -> Script {
        detect_script_impl(text)
    }

    /// Derive a language hint from the dominant script in `text`.
    pub fn detect_language(&self, text: &str) -> LanguageHint {
        let script = self.detect_script(text);
        script_to_language_hint(script)
    }

    /// Apply all enabled normalization options to `text` and return the result.
    ///
    /// Options are applied in this fixed order:
    /// 1. `lowercase`
    /// 2. `strip_accents`
    /// 3. `normalize_whitespace`
    /// 4. `remove_punctuation`
    /// 5. `max_length`
    /// 6. `script_filter`
    pub fn apply_options(&self, text: &str) -> String {
        let mut s: String = text.to_owned();

        if self.options.lowercase {
            s = s.to_lowercase();
        }

        if self.options.strip_accents {
            s = strip_combining_chars(&s);
        }

        if self.options.normalize_whitespace {
            s = normalize_whitespace_impl(&s);
        }

        if self.options.remove_punctuation {
            s = remove_punctuation_impl(&s);
        }

        if let Some(max) = self.options.max_length {
            if s.chars().count() > max {
                s = s.chars().take(max).collect();
            }
        }

        if let Some(ref script) = self.options.script_filter {
            s = script_filter_impl(&s, *script);
        }

        s
    }

    /// Tokenize `text` according to the configured [`TokenizationStrategy`].
    pub fn tokenize(&self, text: &str) -> Vec<String> {
        tokenize_impl(text, &self.tokenization)
    }

    /// Remove Unicode combining diacritical marks (U+0300-U+036F) from `text`.
    pub fn strip_combining_chars(&self, text: &str) -> String {
        strip_combining_chars(text)
    }

    /// Return a point-in-time snapshot of accumulated statistics.
    pub fn stats(&self) -> NormalizerStats {
        let total_processed = self.stats.total_processed.load(Ordering::Relaxed);
        let total_chars_removed = self.stats.total_chars_removed.load(Ordering::Relaxed);
        let total_tokens_generated = self.stats.total_tokens_generated.load(Ordering::Relaxed);

        // avg_compression_ratio = mean fraction of chars removed per document
        let avg_compression_ratio = if total_processed == 0 {
            0.0
        } else {
            total_chars_removed as f64 / total_processed as f64
        };

        NormalizerStats {
            total_processed,
            total_chars_removed,
            total_tokens_generated,
            avg_compression_ratio,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Free-function helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Pure function for script detection (also used by the public method).
fn detect_script_impl(text: &str) -> Script {
    let mut counts: HashMap<Script, usize> = HashMap::new();

    for c in text.chars() {
        let s = char_script(c);
        if s != Script::Unknown {
            *counts.entry(s).or_insert(0) += 1;
        }
    }

    if counts.is_empty() {
        return Script::Unknown;
    }

    // Find maximum count
    let max_count = *counts.values().max().unwrap_or(&0);

    // Collect scripts that share the maximum count
    let leaders: Vec<Script> = counts
        .iter()
        .filter(|(_, &v)| v == max_count)
        .map(|(&k, _)| k)
        .collect();

    // Tie → Unknown
    if leaders.len() == 1 {
        leaders[0]
    } else {
        Script::Unknown
    }
}

/// Map a dominant script to the most likely language hint.
fn script_to_language_hint(script: Script) -> LanguageHint {
    match script {
        Script::Latin => LanguageHint::English,
        Script::Cyrillic => LanguageHint::Russian,
        Script::Arabic => LanguageHint::Arabic,
        Script::CJK => LanguageHint::Chinese,
        Script::Devanagari => LanguageHint::Hindi,
        Script::Unknown => LanguageHint::Unknown,
    }
}

/// Remove Unicode combining diacritical marks (U+0300-U+036F).
fn strip_combining_chars(text: &str) -> String {
    text.chars()
        .filter(|&c| {
            let cp = c as u32;
            !(0x300..=0x36F).contains(&cp)
        })
        .collect()
}

/// Collapse runs of whitespace to a single space and trim.
fn normalize_whitespace_impl(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut prev_was_space = false;

    for c in text.chars() {
        if c.is_whitespace() {
            if !prev_was_space {
                result.push(' ');
            }
            prev_was_space = true;
        } else {
            result.push(c);
            prev_was_space = false;
        }
    }

    // Trim leading/trailing
    result.trim().to_owned()
}

/// Remove characters that are neither alphanumeric nor ASCII space.
fn remove_punctuation_impl(text: &str) -> String {
    text.chars()
        .filter(|c| c.is_alphanumeric() || *c == ' ')
        .collect()
}

/// Retain only characters from `script` and ASCII space.
fn script_filter_impl(text: &str, script: Script) -> String {
    text.chars()
        .filter(|&c| c == ' ' || char_script(c) == script)
        .collect()
}

/// Dispatch tokenization to the appropriate strategy.
fn tokenize_impl(text: &str, strategy: &TokenizationStrategy) -> Vec<String> {
    match strategy {
        TokenizationStrategy::Whitespace => tokenize_whitespace(text),
        TokenizationStrategy::CharacterNgram { n } => tokenize_char_ngram(text, *n),
        TokenizationStrategy::Subword { vocab_size } => tokenize_subword(text, *vocab_size),
        TokenizationStrategy::ScriptAware => tokenize_script_aware(text),
    }
}

/// Whitespace tokenization: split on whitespace, discard empty strings.
fn tokenize_whitespace(text: &str) -> Vec<String> {
    text.split_whitespace().map(|s| s.to_owned()).collect()
}

/// Sliding-window character n-gram tokenization (step = 1).
fn tokenize_char_ngram(text: &str, n: usize) -> Vec<String> {
    if n == 0 {
        return Vec::new();
    }

    let chars: Vec<char> = text.chars().collect();
    if chars.len() < n {
        return Vec::new();
    }

    (0..=(chars.len() - n))
        .map(|i| chars[i..i + n].iter().collect())
        .collect()
}

/// Simplified subword tokenization (BPE approximation).
///
/// Whitespace-split words longer than 10 characters are chunked into pieces
/// of `chunk_len` characters.  `chunk_len` is derived as
/// `max(3, vocab_size / 1_000)` capped at 5.
fn tokenize_subword(text: &str, vocab_size: usize) -> Vec<String> {
    let chunk_len = (vocab_size / 1_000).clamp(3, 5);
    let mut tokens = Vec::new();

    for word in text.split_whitespace() {
        let chars: Vec<char> = word.chars().collect();
        if chars.len() > 10 {
            // Chunk into pieces
            let mut start = 0;
            while start < chars.len() {
                let end = (start + chunk_len).min(chars.len());
                tokens.push(chars[start..end].iter().collect::<String>());
                start += chunk_len;
            }
        } else {
            tokens.push(word.to_owned());
        }
    }

    tokens
}

/// Script-aware tokenization: contiguous runs of the same script become tokens.
///
/// Whitespace acts as an additional boundary (whitespace is not emitted as a
/// token).
fn tokenize_script_aware(text: &str) -> Vec<String> {
    let mut tokens: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut current_script: Option<Script> = None;

    for c in text.chars() {
        if c.is_whitespace() {
            if !current.is_empty() {
                tokens.push(current.clone());
                current.clear();
                current_script = None;
            }
            continue;
        }

        let script = char_script(c);

        match current_script {
            None => {
                current.push(c);
                current_script = Some(script);
            }
            Some(cs) if cs == script => {
                current.push(c);
            }
            Some(_) => {
                // Script boundary
                tokens.push(current.clone());
                current.clear();
                current.push(c);
                current_script = Some(script);
            }
        }
    }

    if !current.is_empty() {
        tokens.push(current);
    }

    tokens
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use crate::multilingual_normalizer::{
        char_script, detect_script_impl, normalize_whitespace_impl, remove_punctuation_impl,
        script_filter_impl, script_to_language_hint, strip_combining_chars, tokenize_char_ngram,
        tokenize_script_aware, tokenize_subword, tokenize_whitespace, LanguageHint,
        MultilingualNormalizer, NormalizationOptions, Script, TokenizationStrategy,
    };

    // ── Helper ────────────────────────────────────────────────────────────

    fn default_norm() -> MultilingualNormalizer {
        MultilingualNormalizer::with_defaults()
    }

    fn norm_with(
        options: NormalizationOptions,
        strategy: TokenizationStrategy,
    ) -> MultilingualNormalizer {
        MultilingualNormalizer::new(options, strategy)
    }

    // ── char_script ───────────────────────────────────────────────────────

    #[test]
    fn test_char_script_latin_ascii() {
        assert_eq!(char_script('A'), Script::Latin);
        assert_eq!(char_script('z'), Script::Latin);
        assert_eq!(char_script('M'), Script::Latin);
    }

    #[test]
    fn test_char_script_latin_extended() {
        assert_eq!(char_script('À'), Script::Latin); // U+00C0
        assert_eq!(char_script('ñ'), Script::Latin); // U+00F1
    }

    #[test]
    fn test_char_script_cyrillic() {
        assert_eq!(char_script('А'), Script::Cyrillic); // U+0410
        assert_eq!(char_script('я'), Script::Cyrillic); // U+044F
    }

    #[test]
    fn test_char_script_arabic() {
        assert_eq!(char_script('ع'), Script::Arabic); // U+0639
        assert_eq!(char_script('م'), Script::Arabic); // U+0645
    }

    #[test]
    fn test_char_script_cjk() {
        assert_eq!(char_script('中'), Script::CJK); // U+4E2D
        assert_eq!(char_script('文'), Script::CJK); // U+6587
    }

    #[test]
    fn test_char_script_devanagari() {
        assert_eq!(char_script('अ'), Script::Devanagari); // U+0905
        assert_eq!(char_script('ह'), Script::Devanagari); // U+0939
    }

    #[test]
    fn test_char_script_unknown() {
        assert_eq!(char_script(' '), Script::Unknown);
        assert_eq!(char_script('1'), Script::Unknown);
        assert_eq!(char_script('!'), Script::Unknown);
    }

    // ── detect_script_impl ────────────────────────────────────────────────

    #[test]
    fn test_detect_script_empty() {
        assert_eq!(detect_script_impl(""), Script::Unknown);
    }

    #[test]
    fn test_detect_script_latin_dominant() {
        assert_eq!(detect_script_impl("Hello World"), Script::Latin);
    }

    #[test]
    fn test_detect_script_cyrillic_dominant() {
        assert_eq!(detect_script_impl("Привет мир"), Script::Cyrillic);
    }

    #[test]
    fn test_detect_script_arabic_dominant() {
        assert_eq!(detect_script_impl("مرحبا"), Script::Arabic);
    }

    #[test]
    fn test_detect_script_cjk_dominant() {
        assert_eq!(detect_script_impl("你好世界"), Script::CJK);
    }

    #[test]
    fn test_detect_script_devanagari_dominant() {
        assert_eq!(detect_script_impl("नमस्ते"), Script::Devanagari);
    }

    #[test]
    fn test_detect_script_tie_returns_unknown() {
        // Equal counts of Latin and Cyrillic → Unknown
        let text = "ABcd АБвг"; // 4 Latin, 4 Cyrillic (ignoring space)
        let result = detect_script_impl(text);
        assert_eq!(result, Script::Unknown);
    }

    #[test]
    fn test_detect_script_only_numbers() {
        // Only Unknown chars
        assert_eq!(detect_script_impl("12345"), Script::Unknown);
    }

    // ── script_to_language_hint ───────────────────────────────────────────

    #[test]
    fn test_language_hint_from_scripts() {
        assert_eq!(
            script_to_language_hint(Script::Latin),
            LanguageHint::English
        );
        assert_eq!(
            script_to_language_hint(Script::Cyrillic),
            LanguageHint::Russian
        );
        assert_eq!(
            script_to_language_hint(Script::Arabic),
            LanguageHint::Arabic
        );
        assert_eq!(script_to_language_hint(Script::CJK), LanguageHint::Chinese);
        assert_eq!(
            script_to_language_hint(Script::Devanagari),
            LanguageHint::Hindi
        );
        assert_eq!(
            script_to_language_hint(Script::Unknown),
            LanguageHint::Unknown
        );
    }

    // ── strip_combining_chars ─────────────────────────────────────────────

    #[test]
    fn test_strip_combining_chars_basic() {
        // 'e' followed by combining grave accent U+0300
        let s = "e\u{0300}";
        assert_eq!(strip_combining_chars(s), "e");
    }

    #[test]
    fn test_strip_combining_chars_no_combining() {
        let s = "hello";
        assert_eq!(strip_combining_chars(s), "hello");
    }

    #[test]
    fn test_strip_combining_chars_multiple() {
        let s = "a\u{0301}b\u{0302}c";
        assert_eq!(strip_combining_chars(s), "abc");
    }

    // ── normalize_whitespace_impl ─────────────────────────────────────────

    #[test]
    fn test_normalize_whitespace_collapses_spaces() {
        assert_eq!(normalize_whitespace_impl("a  b   c"), "a b c");
    }

    #[test]
    fn test_normalize_whitespace_trims() {
        assert_eq!(normalize_whitespace_impl("  hello  "), "hello");
    }

    #[test]
    fn test_normalize_whitespace_tabs_and_newlines() {
        assert_eq!(normalize_whitespace_impl("a\t\nb"), "a b");
    }

    // ── remove_punctuation_impl ───────────────────────────────────────────

    #[test]
    fn test_remove_punctuation_basic() {
        assert_eq!(remove_punctuation_impl("hello, world!"), "hello world");
    }

    #[test]
    fn test_remove_punctuation_keeps_alphanumeric() {
        assert_eq!(remove_punctuation_impl("abc123"), "abc123");
    }

    // ── script_filter_impl ────────────────────────────────────────────────

    #[test]
    fn test_script_filter_latin_only() {
        let text = "Hello Привет";
        let result = script_filter_impl(text, Script::Latin);
        // Keeps Latin chars and spaces; drops Cyrillic
        assert!(result.contains("Hello"));
        assert!(!result.contains('П'));
    }

    #[test]
    fn test_script_filter_cjk_only() {
        let text = "你好 hello";
        let result = script_filter_impl(text, Script::CJK);
        assert!(result.contains('你'));
        assert!(!result.contains('h'));
    }

    // ── tokenize_whitespace ───────────────────────────────────────────────

    #[test]
    fn test_tokenize_whitespace_basic() {
        let tokens = tokenize_whitespace("hello world foo");
        assert_eq!(tokens, vec!["hello", "world", "foo"]);
    }

    #[test]
    fn test_tokenize_whitespace_empty() {
        assert!(tokenize_whitespace("   ").is_empty());
    }

    // ── tokenize_char_ngram ───────────────────────────────────────────────

    #[test]
    fn test_tokenize_char_ngram_bigrams() {
        let tokens = tokenize_char_ngram("abcd", 2);
        assert_eq!(tokens, vec!["ab", "bc", "cd"]);
    }

    #[test]
    fn test_tokenize_char_ngram_too_short() {
        assert!(tokenize_char_ngram("ab", 3).is_empty());
    }

    #[test]
    fn test_tokenize_char_ngram_zero_n() {
        assert!(tokenize_char_ngram("hello", 0).is_empty());
    }

    #[test]
    fn test_tokenize_char_ngram_exact_length() {
        let tokens = tokenize_char_ngram("abc", 3);
        assert_eq!(tokens, vec!["abc"]);
    }

    // ── tokenize_subword ──────────────────────────────────────────────────

    #[test]
    fn test_tokenize_subword_short_word() {
        // "hello" is 5 chars — not chunked (≤10)
        let tokens = tokenize_subword("hello", 5000);
        assert_eq!(tokens, vec!["hello"]);
    }

    #[test]
    fn test_tokenize_subword_long_word() {
        // "abcdefghijk" is 11 chars — chunked into 5-char pieces
        let tokens = tokenize_subword("abcdefghijk", 5000);
        // chunk_len = (5000/1000).clamp(3,5) = 5
        assert_eq!(tokens[0], "abcde");
        assert_eq!(tokens[1], "fghij");
        assert_eq!(tokens[2], "k");
    }

    #[test]
    fn test_tokenize_subword_mixed() {
        let tokens = tokenize_subword("hi abcdefghijk", 5000);
        assert_eq!(tokens[0], "hi");
        assert_eq!(tokens[1], "abcde");
    }

    // ── tokenize_script_aware ─────────────────────────────────────────────

    #[test]
    fn test_tokenize_script_aware_latin_only() {
        let tokens = tokenize_script_aware("hello world");
        assert_eq!(tokens, vec!["hello", "world"]);
    }

    #[test]
    fn test_tokenize_script_aware_mixed_scripts() {
        let text = "hello Привет";
        let tokens = tokenize_script_aware(text);
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0], "hello");
        assert_eq!(tokens[1], "Привет");
    }

    #[test]
    fn test_tokenize_script_aware_script_boundary_no_space() {
        // Latin then Cyrillic without space
        let text = "ABСа"; // AB = Latin, Са = Cyrillic
        let tokens = tokenize_script_aware(text);
        assert_eq!(tokens.len(), 2);
    }

    // ── MultilingualNormalizer full pipeline ──────────────────────────────

    #[test]
    fn test_normalize_lowercase() {
        let opts = NormalizationOptions {
            lowercase: true,
            normalize_whitespace: false,
            ..Default::default()
        };
        let norm = norm_with(opts, TokenizationStrategy::Whitespace);
        let result = norm.normalize("Hello World");
        assert_eq!(result.normalized, "hello world");
    }

    #[test]
    fn test_normalize_detected_script_and_language() {
        let norm = default_norm();
        let result = norm.normalize("Hello World");
        assert_eq!(result.detected_script, Script::Latin);
        assert_eq!(result.language_hint, LanguageHint::English);
    }

    #[test]
    fn test_normalize_cyrillic_language_hint() {
        let norm = default_norm();
        let result = norm.normalize("Привет мир");
        assert_eq!(result.detected_script, Script::Cyrillic);
        assert_eq!(result.language_hint, LanguageHint::Russian);
    }

    #[test]
    fn test_normalize_tokens_whitespace_strategy() {
        let opts = NormalizationOptions {
            lowercase: true,
            ..Default::default()
        };
        let norm = norm_with(opts, TokenizationStrategy::Whitespace);
        let result = norm.normalize("foo bar baz");
        assert_eq!(result.token_count, 3);
        assert_eq!(result.tokens, vec!["foo", "bar", "baz"]);
    }

    #[test]
    fn test_normalize_char_count() {
        let opts = NormalizationOptions {
            lowercase: false,
            normalize_whitespace: false,
            ..Default::default()
        };
        let norm = norm_with(opts, TokenizationStrategy::Whitespace);
        let result = norm.normalize("hello");
        assert_eq!(result.char_count, 5);
    }

    #[test]
    fn test_normalize_max_length() {
        let opts = NormalizationOptions {
            lowercase: false,
            normalize_whitespace: false,
            max_length: Some(3),
            ..Default::default()
        };
        let norm = norm_with(opts, TokenizationStrategy::Whitespace);
        let result = norm.normalize("hello");
        assert_eq!(result.normalized, "hel");
        assert_eq!(result.char_count, 3);
    }

    #[test]
    fn test_normalize_remove_punctuation() {
        let opts = NormalizationOptions {
            lowercase: false,
            normalize_whitespace: false,
            remove_punctuation: true,
            ..Default::default()
        };
        let norm = norm_with(opts, TokenizationStrategy::Whitespace);
        let result = norm.normalize("hello, world!");
        assert_eq!(result.normalized, "hello world");
    }

    #[test]
    fn test_normalize_strip_accents() {
        let opts = NormalizationOptions {
            lowercase: false,
            strip_accents: true,
            normalize_whitespace: false,
            ..Default::default()
        };
        let norm = norm_with(opts, TokenizationStrategy::Whitespace);
        let result = norm.normalize("e\u{0300}");
        assert_eq!(result.normalized, "e");
    }

    #[test]
    fn test_normalize_batch() {
        let norm = default_norm();
        let results = norm.normalize_batch(&["hello", "world", "foo"]);
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].normalized, "hello");
        assert_eq!(results[1].normalized, "world");
    }

    #[test]
    fn test_normalize_stats_increment() {
        let norm = default_norm();
        norm.normalize("hello world");
        norm.normalize("foo bar");
        let stats = norm.stats();
        assert_eq!(stats.total_processed, 2);
        assert!(stats.total_tokens_generated >= 4);
    }

    #[test]
    fn test_stats_initial_zero() {
        let norm = default_norm();
        let stats = norm.stats();
        assert_eq!(stats.total_processed, 0);
        assert_eq!(stats.total_chars_removed, 0);
        assert_eq!(stats.total_tokens_generated, 0);
        assert_eq!(stats.avg_compression_ratio, 0.0);
    }

    #[test]
    fn test_normalize_original_preserved() {
        let norm = default_norm();
        let result = norm.normalize("Hello World");
        assert_eq!(result.original, "Hello World");
    }

    #[test]
    fn test_normalize_ngram_strategy() {
        let opts = NormalizationOptions {
            lowercase: false,
            normalize_whitespace: false,
            ..Default::default()
        };
        let norm = norm_with(opts, TokenizationStrategy::CharacterNgram { n: 2 });
        let result = norm.normalize("abc");
        assert_eq!(result.tokens, vec!["ab", "bc"]);
    }

    #[test]
    fn test_normalize_subword_strategy() {
        let opts = NormalizationOptions {
            lowercase: false,
            normalize_whitespace: false,
            ..Default::default()
        };
        let norm = norm_with(opts, TokenizationStrategy::Subword { vocab_size: 5000 });
        let result = norm.normalize("abcdefghijk");
        // 11 chars → chunked
        assert!(result.token_count > 1);
    }

    #[test]
    fn test_normalize_script_aware_strategy() {
        let opts = NormalizationOptions {
            lowercase: false,
            ..Default::default()
        };
        let norm = norm_with(opts, TokenizationStrategy::ScriptAware);
        let result = norm.normalize("hello Привет");
        assert_eq!(result.token_count, 2);
    }

    #[test]
    fn test_normalize_script_filter() {
        let opts = NormalizationOptions {
            lowercase: false,
            normalize_whitespace: false,
            script_filter: Some(Script::Latin),
            ..Default::default()
        };
        let norm = norm_with(opts, TokenizationStrategy::Whitespace);
        let result = norm.normalize("hello Привет");
        // Only Latin chars + spaces retained
        assert!(!result.normalized.contains('П'));
        assert!(result.normalized.contains("hello"));
    }

    #[test]
    fn test_detect_language_method() {
        let norm = default_norm();
        assert_eq!(norm.detect_language("Hello"), LanguageHint::English);
        assert_eq!(norm.detect_language("Привет"), LanguageHint::Russian);
        assert_eq!(norm.detect_language("مرحبا"), LanguageHint::Arabic);
        assert_eq!(norm.detect_language("你好"), LanguageHint::Chinese);
    }

    #[test]
    fn test_detect_script_method() {
        let norm = default_norm();
        assert_eq!(norm.detect_script("नमस्ते"), Script::Devanagari);
    }

    #[test]
    fn test_language_hint_bcp47() {
        assert_eq!(LanguageHint::English.bcp47(), "en");
        assert_eq!(LanguageHint::Russian.bcp47(), "ru");
        assert_eq!(LanguageHint::Arabic.bcp47(), "ar");
        assert_eq!(LanguageHint::Chinese.bcp47(), "zh");
        assert_eq!(LanguageHint::Japanese.bcp47(), "ja");
        assert_eq!(LanguageHint::Hindi.bcp47(), "hi");
        assert_eq!(LanguageHint::Unknown.bcp47(), "und");
    }

    #[test]
    fn test_script_name() {
        assert_eq!(Script::Latin.name(), "Latin");
        assert_eq!(Script::CJK.name(), "CJK");
        assert_eq!(Script::Unknown.name(), "Unknown");
    }

    #[test]
    fn test_normalize_empty_string() {
        let norm = default_norm();
        let result = norm.normalize("");
        assert_eq!(result.normalized, "");
        assert_eq!(result.token_count, 0);
        assert_eq!(result.char_count, 0);
        assert_eq!(result.detected_script, Script::Unknown);
    }

    #[test]
    fn test_normalize_numbers_only() {
        let norm = default_norm();
        let result = norm.normalize("12345");
        assert_eq!(result.detected_script, Script::Unknown);
        assert_eq!(result.language_hint, LanguageHint::Unknown);
    }

    #[test]
    fn test_normalize_full_pipeline_order() {
        // lowercase → strip_accents → normalize_whitespace → remove_punctuation → max_length
        let opts = NormalizationOptions {
            lowercase: true,
            strip_accents: true,
            normalize_whitespace: true,
            remove_punctuation: true,
            max_length: Some(5),
            script_filter: None,
        };
        let norm = norm_with(opts, TokenizationStrategy::Whitespace);
        let result = norm.normalize("  He\u{0300}llo,  World!  ");
        // After lowercase: "  he\u{0300}llo,  world!  "
        // After strip_accents: "  hello,  world!  "
        // After normalize_whitespace: "hello, world!"
        // After remove_punctuation: "hello world"
        // After max_length(5): "hello"
        assert_eq!(result.normalized, "hello");
    }

    #[test]
    fn test_stats_chars_removed_tracked() {
        let opts = NormalizationOptions {
            lowercase: false,
            normalize_whitespace: false,
            strip_accents: false,
            remove_punctuation: true,
            max_length: None,
            script_filter: None,
        };
        let norm = norm_with(opts, TokenizationStrategy::Whitespace);
        norm.normalize("hello!!!");
        let stats = norm.stats();
        // "!!!" = 3 chars removed
        assert_eq!(stats.total_chars_removed, 3);
        assert!(stats.avg_compression_ratio > 0.0);
    }
}
