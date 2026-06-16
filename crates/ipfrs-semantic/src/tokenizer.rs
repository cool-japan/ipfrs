//! Semantic Tokenizer — text tokenization for semantic search indexing.
//!
//! Provides configurable tokenization with whitespace, word boundary, and n-gram modes,
//! stop-word filtering, length constraints, and normalization.

use std::fmt;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Tokenization strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenizerMode {
    /// Split on whitespace characters.
    Whitespace,
    /// Split on non-alphanumeric boundaries.
    WordBoundary,
    /// Generate character n-grams of a fixed size.
    NGram,
}

impl fmt::Display for TokenizerMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Whitespace => write!(f, "Whitespace"),
            Self::WordBoundary => write!(f, "WordBoundary"),
            Self::NGram => write!(f, "NGram"),
        }
    }
}

/// Configuration for [`SemanticTokenizer`].
#[derive(Debug, Clone)]
pub struct TokenizerConfig {
    /// Tokenization mode.
    pub mode: TokenizerMode,
    /// Whether to lowercase all tokens (default `true`).
    pub lowercase: bool,
    /// Minimum token length to keep (default `1`).
    pub min_token_length: usize,
    /// Maximum token length to keep (default `100`).
    pub max_token_length: usize,
    /// Character n-gram size (only used with [`TokenizerMode::NGram`], default `3`).
    pub ngram_size: usize,
    /// Words to exclude from output.
    pub stop_words: Vec<String>,
}

impl Default for TokenizerConfig {
    fn default() -> Self {
        Self {
            mode: TokenizerMode::Whitespace,
            lowercase: true,
            min_token_length: 1,
            max_token_length: 100,
            ngram_size: 3,
            stop_words: Vec::new(),
        }
    }
}

/// A single token produced by the tokenizer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Token {
    /// The (possibly lowercased) token text.
    pub text: String,
    /// Zero-based position among emitted tokens in the source text.
    pub position: usize,
    /// Byte offset where this token begins in the original text.
    pub byte_offset: usize,
}

/// Runtime statistics for a [`SemanticTokenizer`].
#[derive(Debug, Clone)]
pub struct TokenizerStats {
    /// Active tokenization mode.
    pub mode: TokenizerMode,
    /// Cumulative number of tokens produced.
    pub tokens_produced: u64,
    /// Number of texts that have been tokenized.
    pub texts_tokenized: u64,
    /// Current size of the stop-word list.
    pub stop_words_count: usize,
}

// ---------------------------------------------------------------------------
// SemanticTokenizer
// ---------------------------------------------------------------------------

/// A configurable tokenizer for semantic search indexing.
///
/// Supports whitespace splitting, word-boundary splitting, and character n-gram
/// generation. Tokens can be lowercased, length-filtered, and stop-word filtered.
pub struct SemanticTokenizer {
    config: TokenizerConfig,
    tokens_produced: u64,
    texts_tokenized: u64,
}

impl SemanticTokenizer {
    /// Create a new tokenizer with the given configuration.
    pub fn new(config: TokenizerConfig) -> Self {
        Self {
            config,
            tokens_produced: 0,
            texts_tokenized: 0,
        }
    }

    /// Tokenize `text` according to the current mode and filters.
    pub fn tokenize(&mut self, text: &str) -> Vec<Token> {
        let raw = self.raw_segments(text);
        let tokens = self.apply_filters(raw);
        self.texts_tokenized += 1;
        self.tokens_produced += tokens.len() as u64;
        tokens
    }

    /// Tokenize a batch of texts.
    pub fn tokenize_batch(&mut self, texts: &[&str]) -> Vec<Vec<Token>> {
        texts.iter().map(|t| self.tokenize(t)).collect()
    }

    /// Return the number of tokens that would be produced **without** allocating
    /// full `Token` structs.
    pub fn token_count(&self, text: &str) -> usize {
        let raw = self.raw_segments(text);
        self.count_filtered(&raw)
    }

    /// Add a stop word.
    pub fn add_stop_word(&mut self, word: &str) {
        let w = if self.config.lowercase {
            word.to_lowercase()
        } else {
            word.to_string()
        };
        if !self.config.stop_words.contains(&w) {
            self.config.stop_words.push(w);
        }
    }

    /// Remove a stop word. Returns `true` if it was present.
    pub fn remove_stop_word(&mut self, word: &str) -> bool {
        let w = if self.config.lowercase {
            word.to_lowercase()
        } else {
            word.to_string()
        };
        if let Some(pos) = self.config.stop_words.iter().position(|s| s == &w) {
            self.config.stop_words.remove(pos);
            true
        } else {
            false
        }
    }

    /// Check whether `word` is in the stop-word list.
    pub fn is_stop_word(&self, word: &str) -> bool {
        let w = if self.config.lowercase {
            word.to_lowercase()
        } else {
            word.to_string()
        };
        self.config.stop_words.contains(&w)
    }

    /// Normalize `text` by applying lowercasing (if configured).
    pub fn normalize(&self, text: &str) -> String {
        if self.config.lowercase {
            text.to_lowercase()
        } else {
            text.to_string()
        }
    }

    /// Return current tokenizer statistics.
    pub fn stats(&self) -> TokenizerStats {
        TokenizerStats {
            mode: self.config.mode,
            tokens_produced: self.tokens_produced,
            texts_tokenized: self.texts_tokenized,
            stop_words_count: self.config.stop_words.len(),
        }
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// A raw segment: (start_byte_offset, text_slice).
    fn raw_segments<'a>(&self, text: &'a str) -> Vec<(usize, &'a str)> {
        match self.config.mode {
            TokenizerMode::Whitespace => self.split_whitespace(text),
            TokenizerMode::WordBoundary => self.split_word_boundary(text),
            TokenizerMode::NGram => self.split_ngrams(text),
        }
    }

    fn split_whitespace<'a>(&self, text: &'a str) -> Vec<(usize, &'a str)> {
        let mut segments = Vec::new();
        let mut in_word = false;
        let mut start = 0;
        for (i, ch) in text.char_indices() {
            if ch.is_whitespace() {
                if in_word {
                    segments.push((start, &text[start..i]));
                    in_word = false;
                }
            } else {
                if !in_word {
                    start = i;
                    in_word = true;
                }
            }
        }
        if in_word {
            segments.push((start, &text[start..]));
        }
        segments
    }

    fn split_word_boundary<'a>(&self, text: &'a str) -> Vec<(usize, &'a str)> {
        let mut segments = Vec::new();
        let mut in_word = false;
        let mut start = 0;
        for (i, ch) in text.char_indices() {
            if ch.is_alphanumeric() || ch == '_' {
                if !in_word {
                    start = i;
                    in_word = true;
                }
            } else {
                if in_word {
                    segments.push((start, &text[start..i]));
                    in_word = false;
                }
            }
        }
        if in_word {
            segments.push((start, &text[start..]));
        }
        segments
    }

    fn split_ngrams<'a>(&self, text: &'a str) -> Vec<(usize, &'a str)> {
        let n = self.config.ngram_size;
        if n == 0 {
            return Vec::new();
        }
        let chars: Vec<(usize, char)> = text.char_indices().collect();
        if chars.len() < n {
            return Vec::new();
        }
        let mut segments = Vec::new();
        for window_start in 0..=(chars.len() - n) {
            let byte_start = chars[window_start].0;
            let byte_end = if window_start + n < chars.len() {
                chars[window_start + n].0
            } else {
                text.len()
            };
            segments.push((byte_start, &text[byte_start..byte_end]));
        }
        segments
    }

    /// Apply lowercase, length, and stop-word filters and assign positions.
    fn apply_filters(&self, raw: Vec<(usize, &str)>) -> Vec<Token> {
        let mut tokens = Vec::new();
        let mut position = 0usize;
        for (byte_offset, slice) in raw {
            let text = if self.config.lowercase {
                slice.to_lowercase()
            } else {
                slice.to_string()
            };
            let len = text.len();
            if len < self.config.min_token_length || len > self.config.max_token_length {
                continue;
            }
            if self.config.stop_words.contains(&text) {
                continue;
            }
            tokens.push(Token {
                text,
                position,
                byte_offset,
            });
            position += 1;
        }
        tokens
    }

    /// Count tokens that pass filters without allocating Token structs.
    fn count_filtered(&self, raw: &[(usize, &str)]) -> usize {
        let mut count = 0usize;
        for (_offset, slice) in raw {
            let text = if self.config.lowercase {
                slice.to_lowercase()
            } else {
                (*slice).to_string()
            };
            let len = text.len();
            if len < self.config.min_token_length || len > self.config.max_token_length {
                continue;
            }
            if self.config.stop_words.contains(&text) {
                continue;
            }
            count += 1;
        }
        count
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_tokenizer() -> SemanticTokenizer {
        SemanticTokenizer::new(TokenizerConfig::default())
    }

    // -- Whitespace mode ---------------------------------------------------

    #[test]
    fn whitespace_basic() {
        let mut tok = default_tokenizer();
        let tokens = tok.tokenize("hello world");
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].text, "hello");
        assert_eq!(tokens[1].text, "world");
    }

    #[test]
    fn whitespace_multiple_spaces() {
        let mut tok = default_tokenizer();
        let tokens = tok.tokenize("a   b\t\nc");
        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[0].text, "a");
        assert_eq!(tokens[1].text, "b");
        assert_eq!(tokens[2].text, "c");
    }

    #[test]
    fn whitespace_preserves_punctuation() {
        let mut tok = default_tokenizer();
        let tokens = tok.tokenize("hello, world!");
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].text, "hello,");
        assert_eq!(tokens[1].text, "world!");
    }

    #[test]
    fn whitespace_leading_trailing() {
        let mut tok = default_tokenizer();
        let tokens = tok.tokenize("  hello  ");
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].text, "hello");
        assert_eq!(tokens[0].byte_offset, 2);
    }

    // -- Word boundary mode ------------------------------------------------

    #[test]
    fn word_boundary_basic() {
        let mut tok = SemanticTokenizer::new(TokenizerConfig {
            mode: TokenizerMode::WordBoundary,
            ..Default::default()
        });
        let tokens = tok.tokenize("hello, world!");
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].text, "hello");
        assert_eq!(tokens[1].text, "world");
    }

    #[test]
    fn word_boundary_underscore() {
        let mut tok = SemanticTokenizer::new(TokenizerConfig {
            mode: TokenizerMode::WordBoundary,
            ..Default::default()
        });
        let tokens = tok.tokenize("foo_bar baz");
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].text, "foo_bar");
        assert_eq!(tokens[1].text, "baz");
    }

    #[test]
    fn word_boundary_numbers() {
        let mut tok = SemanticTokenizer::new(TokenizerConfig {
            mode: TokenizerMode::WordBoundary,
            ..Default::default()
        });
        let tokens = tok.tokenize("v2.0-alpha");
        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[0].text, "v2");
        assert_eq!(tokens[1].text, "0");
        assert_eq!(tokens[2].text, "alpha");
    }

    #[test]
    fn word_boundary_all_punctuation() {
        let mut tok = SemanticTokenizer::new(TokenizerConfig {
            mode: TokenizerMode::WordBoundary,
            ..Default::default()
        });
        let tokens = tok.tokenize("...---!!!");
        assert!(tokens.is_empty());
    }

    // -- NGram mode --------------------------------------------------------

    #[test]
    fn ngram_basic() {
        let mut tok = SemanticTokenizer::new(TokenizerConfig {
            mode: TokenizerMode::NGram,
            ngram_size: 3,
            ..Default::default()
        });
        let tokens = tok.tokenize("hello");
        // 5 chars => 3 trigrams: hel, ell, llo
        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[0].text, "hel");
        assert_eq!(tokens[1].text, "ell");
        assert_eq!(tokens[2].text, "llo");
    }

    #[test]
    fn ngram_too_short() {
        let mut tok = SemanticTokenizer::new(TokenizerConfig {
            mode: TokenizerMode::NGram,
            ngram_size: 5,
            ..Default::default()
        });
        let tokens = tok.tokenize("hi");
        assert!(tokens.is_empty());
    }

    #[test]
    fn ngram_exact_length() {
        let mut tok = SemanticTokenizer::new(TokenizerConfig {
            mode: TokenizerMode::NGram,
            ngram_size: 3,
            ..Default::default()
        });
        let tokens = tok.tokenize("abc");
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].text, "abc");
    }

    #[test]
    fn ngram_size_one() {
        let mut tok = SemanticTokenizer::new(TokenizerConfig {
            mode: TokenizerMode::NGram,
            ngram_size: 1,
            ..Default::default()
        });
        let tokens = tok.tokenize("hi");
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].text, "h");
        assert_eq!(tokens[1].text, "i");
    }

    #[test]
    fn ngram_size_zero() {
        let mut tok = SemanticTokenizer::new(TokenizerConfig {
            mode: TokenizerMode::NGram,
            ngram_size: 0,
            ..Default::default()
        });
        let tokens = tok.tokenize("anything");
        assert!(tokens.is_empty());
    }

    #[test]
    fn ngram_unicode() {
        let mut tok = SemanticTokenizer::new(TokenizerConfig {
            mode: TokenizerMode::NGram,
            ngram_size: 2,
            lowercase: false,
            ..Default::default()
        });
        let tokens = tok.tokenize("café");
        assert_eq!(tokens.len(), 3); // ca, af, fé
        assert_eq!(tokens[0].text, "ca");
        assert_eq!(tokens[2].text, "fé");
    }

    // -- Lowercase ---------------------------------------------------------

    #[test]
    fn lowercase_normalization() {
        let mut tok = default_tokenizer();
        let tokens = tok.tokenize("Hello WORLD FoO");
        assert_eq!(tokens[0].text, "hello");
        assert_eq!(tokens[1].text, "world");
        assert_eq!(tokens[2].text, "foo");
    }

    #[test]
    fn lowercase_disabled() {
        let mut tok = SemanticTokenizer::new(TokenizerConfig {
            lowercase: false,
            ..Default::default()
        });
        let tokens = tok.tokenize("Hello WORLD");
        assert_eq!(tokens[0].text, "Hello");
        assert_eq!(tokens[1].text, "WORLD");
    }

    // -- Stop words --------------------------------------------------------

    #[test]
    fn stop_words_filtering() {
        let mut tok = SemanticTokenizer::new(TokenizerConfig {
            stop_words: vec!["the".into(), "a".into(), "is".into()],
            ..Default::default()
        });
        let tokens = tok.tokenize("the cat is a friend");
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].text, "cat");
        assert_eq!(tokens[1].text, "friend");
    }

    #[test]
    fn stop_words_case_insensitive() {
        let mut tok = SemanticTokenizer::new(TokenizerConfig {
            stop_words: vec!["the".into()],
            ..Default::default()
        });
        let tokens = tok.tokenize("The THE thE dog");
        // all "the" variants lowercased => filtered
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].text, "dog");
    }

    #[test]
    fn add_remove_stop_word() {
        let mut tok = default_tokenizer();
        assert!(!tok.is_stop_word("foo"));
        tok.add_stop_word("foo");
        assert!(tok.is_stop_word("foo"));
        assert!(tok.is_stop_word("FOO")); // case insensitive lookup
        assert!(tok.remove_stop_word("foo"));
        assert!(!tok.is_stop_word("foo"));
        assert!(!tok.remove_stop_word("nonexistent"));
    }

    #[test]
    fn add_stop_word_no_duplicate() {
        let mut tok = default_tokenizer();
        tok.add_stop_word("hello");
        tok.add_stop_word("hello");
        tok.add_stop_word("HELLO"); // same after lowercase
        assert_eq!(tok.stats().stop_words_count, 1);
    }

    // -- Min / max token length --------------------------------------------

    #[test]
    fn min_token_length() {
        let mut tok = SemanticTokenizer::new(TokenizerConfig {
            min_token_length: 3,
            ..Default::default()
        });
        let tokens = tok.tokenize("a ab abc abcd");
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].text, "abc");
        assert_eq!(tokens[1].text, "abcd");
    }

    #[test]
    fn max_token_length() {
        let mut tok = SemanticTokenizer::new(TokenizerConfig {
            max_token_length: 4,
            ..Default::default()
        });
        let tokens = tok.tokenize("ab abcd abcdef");
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].text, "ab");
        assert_eq!(tokens[1].text, "abcd");
    }

    // -- Byte offset correctness -------------------------------------------

    #[test]
    fn byte_offset_whitespace() {
        let mut tok = default_tokenizer();
        let tokens = tok.tokenize("hello world");
        assert_eq!(tokens[0].byte_offset, 0);
        assert_eq!(tokens[1].byte_offset, 6);
    }

    #[test]
    fn byte_offset_word_boundary() {
        let mut tok = SemanticTokenizer::new(TokenizerConfig {
            mode: TokenizerMode::WordBoundary,
            lowercase: false,
            ..Default::default()
        });
        let tokens = tok.tokenize("foo--bar");
        assert_eq!(tokens[0].byte_offset, 0);
        assert_eq!(tokens[0].text, "foo");
        assert_eq!(tokens[1].byte_offset, 5);
        assert_eq!(tokens[1].text, "bar");
    }

    #[test]
    fn byte_offset_ngram() {
        let mut tok = SemanticTokenizer::new(TokenizerConfig {
            mode: TokenizerMode::NGram,
            ngram_size: 2,
            lowercase: false,
            ..Default::default()
        });
        let tokens = tok.tokenize("abcd");
        assert_eq!(tokens[0].byte_offset, 0); // ab
        assert_eq!(tokens[1].byte_offset, 1); // bc
        assert_eq!(tokens[2].byte_offset, 2); // cd
    }

    // -- Position tracking -------------------------------------------------

    #[test]
    fn position_sequential() {
        let mut tok = default_tokenizer();
        let tokens = tok.tokenize("a b c d");
        for (i, t) in tokens.iter().enumerate() {
            assert_eq!(t.position, i);
        }
    }

    #[test]
    fn position_with_stop_words() {
        let mut tok = SemanticTokenizer::new(TokenizerConfig {
            stop_words: vec!["b".into()],
            ..Default::default()
        });
        let tokens = tok.tokenize("a b c");
        // "b" filtered, positions renumbered
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].position, 0);
        assert_eq!(tokens[0].text, "a");
        assert_eq!(tokens[1].position, 1);
        assert_eq!(tokens[1].text, "c");
    }

    // -- Batch tokenization ------------------------------------------------

    #[test]
    fn tokenize_batch_basic() {
        let mut tok = default_tokenizer();
        let results = tok.tokenize_batch(&["hello world", "foo bar baz"]);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].len(), 2);
        assert_eq!(results[1].len(), 3);
    }

    #[test]
    fn tokenize_batch_empty_list() {
        let mut tok = default_tokenizer();
        let results = tok.tokenize_batch(&[]);
        assert!(results.is_empty());
    }

    // -- token_count -------------------------------------------------------

    #[test]
    fn token_count_matches_tokenize() {
        let mut tok = SemanticTokenizer::new(TokenizerConfig {
            stop_words: vec!["the".into()],
            min_token_length: 2,
            ..Default::default()
        });
        let text = "the quick brown fox jumps over the lazy dog a";
        let count = tok.token_count(text);
        let tokens = tok.tokenize(text);
        assert_eq!(count, tokens.len());
    }

    // -- Empty text --------------------------------------------------------

    #[test]
    fn empty_text_whitespace() {
        let mut tok = default_tokenizer();
        assert!(tok.tokenize("").is_empty());
    }

    #[test]
    fn empty_text_word_boundary() {
        let mut tok = SemanticTokenizer::new(TokenizerConfig {
            mode: TokenizerMode::WordBoundary,
            ..Default::default()
        });
        assert!(tok.tokenize("").is_empty());
    }

    #[test]
    fn empty_text_ngram() {
        let mut tok = SemanticTokenizer::new(TokenizerConfig {
            mode: TokenizerMode::NGram,
            ..Default::default()
        });
        assert!(tok.tokenize("").is_empty());
    }

    // -- Punctuation handling ----------------------------------------------

    #[test]
    fn punctuation_only_whitespace_mode() {
        let mut tok = default_tokenizer();
        let tokens = tok.tokenize("!!! ???");
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].text, "!!!");
        assert_eq!(tokens[1].text, "???");
    }

    #[test]
    fn punctuation_only_word_boundary_mode() {
        let mut tok = SemanticTokenizer::new(TokenizerConfig {
            mode: TokenizerMode::WordBoundary,
            ..Default::default()
        });
        let tokens = tok.tokenize("!!! ???");
        assert!(tokens.is_empty());
    }

    // -- Normalize ---------------------------------------------------------

    #[test]
    fn normalize_lowercase() {
        let tok = SemanticTokenizer::new(TokenizerConfig {
            lowercase: true,
            ..Default::default()
        });
        assert_eq!(tok.normalize("Hello WORLD"), "hello world");
    }

    #[test]
    fn normalize_no_lowercase() {
        let tok = SemanticTokenizer::new(TokenizerConfig {
            lowercase: false,
            ..Default::default()
        });
        assert_eq!(tok.normalize("Hello WORLD"), "Hello WORLD");
    }

    // -- Stats accuracy ----------------------------------------------------

    #[test]
    fn stats_initial() {
        let tok = default_tokenizer();
        let s = tok.stats();
        assert_eq!(s.mode, TokenizerMode::Whitespace);
        assert_eq!(s.tokens_produced, 0);
        assert_eq!(s.texts_tokenized, 0);
        assert_eq!(s.stop_words_count, 0);
    }

    #[test]
    fn stats_after_tokenization() {
        let mut tok = default_tokenizer();
        tok.tokenize("a b c");
        tok.tokenize("d e");
        let s = tok.stats();
        assert_eq!(s.texts_tokenized, 2);
        assert_eq!(s.tokens_produced, 5);
    }

    #[test]
    fn stats_stop_words_count() {
        let mut tok = SemanticTokenizer::new(TokenizerConfig {
            stop_words: vec!["x".into(), "y".into()],
            ..Default::default()
        });
        assert_eq!(tok.stats().stop_words_count, 2);
        tok.add_stop_word("z");
        assert_eq!(tok.stats().stop_words_count, 3);
        tok.remove_stop_word("x");
        assert_eq!(tok.stats().stop_words_count, 2);
    }

    // -- Display for TokenizerMode -----------------------------------------

    #[test]
    fn tokenizer_mode_display() {
        assert_eq!(format!("{}", TokenizerMode::Whitespace), "Whitespace");
        assert_eq!(format!("{}", TokenizerMode::WordBoundary), "WordBoundary");
        assert_eq!(format!("{}", TokenizerMode::NGram), "NGram");
    }

    // -- Unicode / multibyte -----------------------------------------------

    #[test]
    fn unicode_whitespace_mode() {
        let mut tok = default_tokenizer();
        let tokens = tok.tokenize("héllo wörld");
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].text, "héllo");
        assert_eq!(tokens[1].text, "wörld");
    }

    #[test]
    fn byte_offset_unicode() {
        let mut tok = SemanticTokenizer::new(TokenizerConfig {
            lowercase: false,
            ..Default::default()
        });
        // "é" is 2 bytes in UTF-8, so "héllo" = 6 bytes, then space = 1
        let tokens = tok.tokenize("héllo w");
        assert_eq!(tokens[0].byte_offset, 0);
        assert_eq!(tokens[1].byte_offset, 7); // 6 bytes for héllo + 1 space
    }
}
