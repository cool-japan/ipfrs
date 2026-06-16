//! Semantic Vocabulary Index — maps tokens to numeric IDs with frequency and
//! document-frequency tracking, TF-IDF helpers, and automatic pruning.

use std::collections::HashMap;

/// A single vocabulary entry storing a token, its numeric ID, and usage statistics.
#[derive(Debug, Clone)]
pub struct VocabEntry {
    /// The canonical (possibly case-folded) token string.
    pub token: String,
    /// Unique numeric identifier assigned to this token.
    pub id: u32,
    /// Total number of times this token has been observed.
    pub frequency: u64,
    /// Number of distinct documents that contain this token.
    pub document_frequency: u64,
}

/// Configuration for [`SemanticVocabIndex`].
#[derive(Debug, Clone)]
pub struct VocabConfig {
    /// Maximum number of tokens the index will retain after pruning.
    pub max_vocab_size: usize,
    /// Minimum frequency a token must reach to survive pruning.
    pub min_frequency: u64,
    /// When `false` all tokens are lower-cased before storage / lookup.
    pub case_sensitive: bool,
}

impl Default for VocabConfig {
    fn default() -> Self {
        Self {
            max_vocab_size: 100_000,
            min_frequency: 1,
            case_sensitive: false,
        }
    }
}

/// Aggregate statistics about the vocabulary index.
#[derive(Debug, Clone)]
pub struct VocabIndexStats {
    /// Number of unique tokens currently stored.
    pub vocab_size: usize,
    /// Cumulative count of all token observations.
    pub total_tokens_seen: u64,
    /// Highest frequency among stored tokens (0 when empty).
    pub max_frequency: u64,
    /// Lowest frequency among stored tokens (0 when empty).
    pub min_frequency: u64,
}

/// Vocabulary index mapping tokens to IDs with frequency tracking.
///
/// Provides TF-IDF helpers, top-k retrieval, and automatic pruning by
/// minimum frequency and maximum vocabulary size.
pub struct SemanticVocabIndex {
    config: VocabConfig,
    token_to_id: HashMap<String, u32>,
    id_to_entry: HashMap<u32, VocabEntry>,
    next_id: u32,
    total_tokens_seen: u64,
}

impl SemanticVocabIndex {
    /// Create a new index with the given configuration.
    pub fn new(config: VocabConfig) -> Self {
        Self {
            config,
            token_to_id: HashMap::new(),
            id_to_entry: HashMap::new(),
            next_id: 0,
            total_tokens_seen: 0,
        }
    }

    // ------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------

    /// Normalise a token according to the case-sensitivity setting.
    fn normalise<'a>(&self, token: &'a str) -> std::borrow::Cow<'a, str> {
        if self.config.case_sensitive {
            std::borrow::Cow::Borrowed(token)
        } else {
            std::borrow::Cow::Owned(token.to_lowercase())
        }
    }

    // ------------------------------------------------------------------
    // Public API
    // ------------------------------------------------------------------

    /// Add a single token (or increment its frequency). Returns the token's
    /// numeric ID.
    pub fn add_token(&mut self, token: &str) -> u32 {
        let norm = self.normalise(token).into_owned();
        self.total_tokens_seen += 1;

        if let Some(&id) = self.token_to_id.get(&norm) {
            if let Some(entry) = self.id_to_entry.get_mut(&id) {
                entry.frequency += 1;
            }
            id
        } else {
            let id = self.next_id;
            self.next_id = self.next_id.wrapping_add(1);
            self.token_to_id.insert(norm.clone(), id);
            self.id_to_entry.insert(
                id,
                VocabEntry {
                    token: norm,
                    id,
                    frequency: 1,
                    document_frequency: 0,
                },
            );
            id
        }
    }

    /// Add all tokens from a single document.
    ///
    /// Token frequencies are incremented for every occurrence while
    /// `document_frequency` is incremented at most once per unique token in
    /// the document.
    pub fn add_document(&mut self, tokens: &[&str]) {
        let mut seen_in_doc: HashMap<String, bool> = HashMap::new();

        for &tok in tokens {
            let id = self.add_token(tok);
            let norm = self.normalise(tok).into_owned();
            if let std::collections::hash_map::Entry::Vacant(e) = seen_in_doc.entry(norm) {
                e.insert(true);
                if let Some(entry) = self.id_to_entry.get_mut(&id) {
                    entry.document_frequency += 1;
                }
            }
        }
    }

    /// Look up the numeric ID for a token.
    pub fn get_id(&self, token: &str) -> Option<u32> {
        let norm = self.normalise(token);
        self.token_to_id.get(norm.as_ref()).copied()
    }

    /// Look up the token string for a numeric ID.
    pub fn get_token(&self, id: u32) -> Option<&str> {
        self.id_to_entry.get(&id).map(|e| e.token.as_str())
    }

    /// Return the full [`VocabEntry`] for a token.
    pub fn get_entry(&self, token: &str) -> Option<&VocabEntry> {
        let norm = self.normalise(token);
        self.token_to_id
            .get(norm.as_ref())
            .and_then(|id| self.id_to_entry.get(id))
    }

    /// Return the frequency count for a token (0 if unknown).
    pub fn frequency(&self, token: &str) -> u64 {
        self.get_entry(token).map_or(0, |e| e.frequency)
    }

    /// Compute the inverse document frequency: `ln(total_docs / (1 + df))`.
    ///
    /// Returns `0.0` when the token is not in the index.
    pub fn idf(&self, token: &str, total_docs: u64) -> f64 {
        match self.get_entry(token) {
            Some(entry) => (total_docs as f64 / (1 + entry.document_frequency) as f64).ln(),
            None => 0.0,
        }
    }

    /// Return the top `k` entries ordered by frequency (descending).
    pub fn top_k(&self, k: usize) -> Vec<&VocabEntry> {
        let mut entries: Vec<&VocabEntry> = self.id_to_entry.values().collect();
        entries.sort_by(|a, b| b.frequency.cmp(&a.frequency).then_with(|| a.id.cmp(&b.id)));
        entries.truncate(k);
        entries
    }

    /// Prune the index:
    ///
    /// 1. Remove tokens whose frequency is below `config.min_frequency`.
    /// 2. If the remaining vocabulary exceeds `config.max_vocab_size`, keep
    ///    only the most frequent tokens.
    ///
    /// Returns the number of entries removed.
    pub fn prune(&mut self) -> usize {
        let before = self.id_to_entry.len();

        // Step 1: remove below min_frequency
        let min_freq = self.config.min_frequency;
        let to_remove: Vec<u32> = self
            .id_to_entry
            .iter()
            .filter(|(_, e)| e.frequency < min_freq)
            .map(|(&id, _)| id)
            .collect();

        for id in &to_remove {
            if let Some(entry) = self.id_to_entry.remove(id) {
                self.token_to_id.remove(&entry.token);
            }
        }

        // Step 2: enforce max_vocab_size by keeping most frequent
        let max_size = self.config.max_vocab_size;
        if self.id_to_entry.len() > max_size {
            let mut entries: Vec<(u32, u64)> = self
                .id_to_entry
                .iter()
                .map(|(&id, e)| (id, e.frequency))
                .collect();
            // Sort descending by frequency, break ties by id ascending (oldest first)
            entries.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

            let ids_to_keep: std::collections::HashSet<u32> =
                entries.iter().take(max_size).map(|(id, _)| *id).collect();

            let excess: Vec<u32> = self
                .id_to_entry
                .keys()
                .filter(|id| !ids_to_keep.contains(id))
                .copied()
                .collect();

            for id in &excess {
                if let Some(entry) = self.id_to_entry.remove(id) {
                    self.token_to_id.remove(&entry.token);
                }
            }
        }

        before - self.id_to_entry.len()
    }

    /// Number of unique tokens currently in the index.
    pub fn vocab_size(&self) -> usize {
        self.id_to_entry.len()
    }

    /// Check whether the index contains the given token.
    pub fn contains(&self, token: &str) -> bool {
        let norm = self.normalise(token);
        self.token_to_id.contains_key(norm.as_ref())
    }

    /// Aggregate statistics about the index.
    pub fn stats(&self) -> VocabIndexStats {
        let (max_freq, min_freq) = if self.id_to_entry.is_empty() {
            (0, 0)
        } else {
            let mut max_f = 0u64;
            let mut min_f = u64::MAX;
            for entry in self.id_to_entry.values() {
                if entry.frequency > max_f {
                    max_f = entry.frequency;
                }
                if entry.frequency < min_f {
                    min_f = entry.frequency;
                }
            }
            (max_f, min_f)
        };

        VocabIndexStats {
            vocab_size: self.id_to_entry.len(),
            total_tokens_seen: self.total_tokens_seen,
            max_frequency: max_freq,
            min_frequency: min_freq,
        }
    }
}

// ======================================================================
// Tests
// ======================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn default_index() -> SemanticVocabIndex {
        SemanticVocabIndex::new(VocabConfig::default())
    }

    // --- add_token basics ---

    #[test]
    fn add_token_assigns_unique_ids() {
        let mut idx = default_index();
        let id_a = idx.add_token("alpha");
        let id_b = idx.add_token("beta");
        assert_ne!(id_a, id_b);
    }

    #[test]
    fn add_token_returns_same_id_for_same_token() {
        let mut idx = default_index();
        let id1 = idx.add_token("hello");
        let id2 = idx.add_token("hello");
        assert_eq!(id1, id2);
    }

    #[test]
    fn add_token_increments_frequency() {
        let mut idx = default_index();
        idx.add_token("foo");
        idx.add_token("foo");
        idx.add_token("foo");
        assert_eq!(idx.frequency("foo"), 3);
    }

    #[test]
    fn add_token_increments_total_tokens_seen() {
        let mut idx = default_index();
        idx.add_token("a");
        idx.add_token("b");
        idx.add_token("a");
        assert_eq!(idx.stats().total_tokens_seen, 3);
    }

    // --- case sensitivity ---

    #[test]
    fn case_insensitive_folding() {
        let mut idx = default_index();
        let id1 = idx.add_token("Hello");
        let id2 = idx.add_token("HELLO");
        let id3 = idx.add_token("hello");
        assert_eq!(id1, id2);
        assert_eq!(id2, id3);
        assert_eq!(idx.frequency("hElLo"), 3);
    }

    #[test]
    fn case_sensitive_mode() {
        let mut idx = SemanticVocabIndex::new(VocabConfig {
            case_sensitive: true,
            ..VocabConfig::default()
        });
        let id1 = idx.add_token("Hello");
        let id2 = idx.add_token("hello");
        assert_ne!(id1, id2);
    }

    // --- add_document ---

    #[test]
    fn add_document_updates_doc_frequency_once_per_unique_token() {
        let mut idx = default_index();
        idx.add_document(&["the", "the", "cat"]);
        assert_eq!(idx.get_entry("the").map(|e| e.document_frequency), Some(1));
        assert_eq!(idx.get_entry("cat").map(|e| e.document_frequency), Some(1));
        assert_eq!(idx.frequency("the"), 2);
    }

    #[test]
    fn add_document_multiple_docs() {
        let mut idx = default_index();
        idx.add_document(&["alpha", "beta"]);
        idx.add_document(&["beta", "gamma"]);
        assert_eq!(
            idx.get_entry("alpha").map(|e| e.document_frequency),
            Some(1)
        );
        assert_eq!(idx.get_entry("beta").map(|e| e.document_frequency), Some(2));
        assert_eq!(
            idx.get_entry("gamma").map(|e| e.document_frequency),
            Some(1)
        );
    }

    #[test]
    fn add_document_case_insensitive() {
        let mut idx = default_index();
        idx.add_document(&["Dog", "DOG", "dog"]);
        assert_eq!(idx.frequency("dog"), 3);
        assert_eq!(idx.get_entry("dog").map(|e| e.document_frequency), Some(1));
    }

    // --- get_id / get_token round-trip ---

    #[test]
    fn get_id_roundtrip() {
        let mut idx = default_index();
        let id = idx.add_token("roundtrip");
        assert_eq!(idx.get_id("roundtrip"), Some(id));
        assert_eq!(idx.get_token(id), Some("roundtrip"));
    }

    #[test]
    fn get_id_unknown_returns_none() {
        let idx = default_index();
        assert_eq!(idx.get_id("nonexistent"), None);
    }

    #[test]
    fn get_token_unknown_id_returns_none() {
        let idx = default_index();
        assert_eq!(idx.get_token(999), None);
    }

    // --- get_entry ---

    #[test]
    fn get_entry_returns_correct_data() {
        let mut idx = default_index();
        idx.add_document(&["word", "word"]);
        let entry = idx.get_entry("word").expect("entry should exist");
        assert_eq!(entry.token, "word");
        assert_eq!(entry.frequency, 2);
        assert_eq!(entry.document_frequency, 1);
    }

    // --- idf ---

    #[test]
    fn idf_calculation() {
        let mut idx = default_index();
        idx.add_document(&["common"]);
        idx.add_document(&["common"]);
        idx.add_document(&["rare"]);
        // idf("common", 3) = ln(3 / (1+2)) = ln(1) = 0
        let idf_common = idx.idf("common", 3);
        assert!((idf_common - 0.0).abs() < 1e-12);
        // idf("rare", 3) = ln(3 / (1+1)) = ln(1.5)
        let idf_rare = idx.idf("rare", 3);
        assert!((idf_rare - (1.5_f64).ln()).abs() < 1e-12);
    }

    #[test]
    fn idf_unknown_token_returns_zero() {
        let idx = default_index();
        assert_eq!(idx.idf("missing", 100), 0.0);
    }

    // --- top_k ---

    #[test]
    fn top_k_ordering() {
        let mut idx = default_index();
        for _ in 0..5 {
            idx.add_token("high");
        }
        for _ in 0..3 {
            idx.add_token("mid");
        }
        idx.add_token("low");

        let top = idx.top_k(2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].token, "high");
        assert_eq!(top[1].token, "mid");
    }

    #[test]
    fn top_k_larger_than_vocab() {
        let mut idx = default_index();
        idx.add_token("only");
        let top = idx.top_k(10);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn top_k_empty_index() {
        let idx = default_index();
        assert!(idx.top_k(5).is_empty());
    }

    // --- prune ---

    #[test]
    fn prune_removes_below_min_frequency() {
        let mut idx = SemanticVocabIndex::new(VocabConfig {
            min_frequency: 3,
            ..VocabConfig::default()
        });
        for _ in 0..5 {
            idx.add_token("keep");
        }
        idx.add_token("drop");
        let removed = idx.prune();
        assert_eq!(removed, 1);
        assert!(idx.contains("keep"));
        assert!(!idx.contains("drop"));
    }

    #[test]
    fn prune_enforces_max_vocab_size() {
        let mut idx = SemanticVocabIndex::new(VocabConfig {
            max_vocab_size: 2,
            min_frequency: 1,
            ..VocabConfig::default()
        });
        for _ in 0..10 {
            idx.add_token("top");
        }
        for _ in 0..5 {
            idx.add_token("mid");
        }
        idx.add_token("bottom");

        let removed = idx.prune();
        assert_eq!(removed, 1);
        assert_eq!(idx.vocab_size(), 2);
        assert!(idx.contains("top"));
        assert!(idx.contains("mid"));
        assert!(!idx.contains("bottom"));
    }

    #[test]
    fn prune_combined_min_freq_and_max_size() {
        let mut idx = SemanticVocabIndex::new(VocabConfig {
            max_vocab_size: 1,
            min_frequency: 2,
            ..VocabConfig::default()
        });
        for _ in 0..10 {
            idx.add_token("best");
        }
        for _ in 0..5 {
            idx.add_token("good");
        }
        idx.add_token("once"); // freq=1, below min_frequency

        let removed = idx.prune();
        // "once" removed by min_freq, "good" removed by max_size
        assert_eq!(removed, 2);
        assert_eq!(idx.vocab_size(), 1);
        assert!(idx.contains("best"));
    }

    #[test]
    fn prune_on_empty_index() {
        let mut idx = default_index();
        assert_eq!(idx.prune(), 0);
    }

    // --- contains ---

    #[test]
    fn contains_known_token() {
        let mut idx = default_index();
        idx.add_token("exists");
        assert!(idx.contains("exists"));
        assert!(idx.contains("EXISTS")); // case insensitive
    }

    #[test]
    fn contains_unknown_token() {
        let idx = default_index();
        assert!(!idx.contains("nope"));
    }

    // --- vocab_size ---

    #[test]
    fn vocab_size_tracks_unique_tokens() {
        let mut idx = default_index();
        idx.add_token("a");
        idx.add_token("b");
        idx.add_token("a");
        assert_eq!(idx.vocab_size(), 2);
    }

    // --- stats ---

    #[test]
    fn stats_accuracy() {
        let mut idx = default_index();
        for _ in 0..7 {
            idx.add_token("hot");
        }
        for _ in 0..2 {
            idx.add_token("cold");
        }
        let s = idx.stats();
        assert_eq!(s.vocab_size, 2);
        assert_eq!(s.total_tokens_seen, 9);
        assert_eq!(s.max_frequency, 7);
        assert_eq!(s.min_frequency, 2);
    }

    #[test]
    fn stats_empty_index() {
        let idx = default_index();
        let s = idx.stats();
        assert_eq!(s.vocab_size, 0);
        assert_eq!(s.total_tokens_seen, 0);
        assert_eq!(s.max_frequency, 0);
        assert_eq!(s.min_frequency, 0);
    }

    // --- edge cases ---

    #[test]
    fn add_empty_string_token() {
        let mut idx = default_index();
        let id = idx.add_token("");
        assert_eq!(idx.get_id(""), Some(id));
        assert_eq!(idx.frequency(""), 1);
    }

    #[test]
    fn add_document_empty_slice() {
        let mut idx = default_index();
        idx.add_document(&[]);
        assert_eq!(idx.vocab_size(), 0);
    }

    #[test]
    fn frequency_unknown_token_returns_zero() {
        let idx = default_index();
        assert_eq!(idx.frequency("ghost"), 0);
    }

    #[test]
    fn idf_with_zero_total_docs() {
        let mut idx = default_index();
        idx.add_document(&["x"]);
        // ln(0 / 2) is -inf; just verify it doesn't panic
        let val = idx.idf("x", 0);
        assert!(val.is_finite() || val.is_infinite());
    }

    #[test]
    fn large_vocab_prune_stress() {
        let mut idx = SemanticVocabIndex::new(VocabConfig {
            max_vocab_size: 10,
            min_frequency: 1,
            ..VocabConfig::default()
        });
        for i in 0..100u32 {
            let tok = format!("tok_{i}");
            for _ in 0..((i + 1) as usize) {
                idx.add_token(&tok);
            }
        }
        assert_eq!(idx.vocab_size(), 100);
        let removed = idx.prune();
        assert_eq!(removed, 90);
        assert_eq!(idx.vocab_size(), 10);
        // The top-10 by frequency should be tok_90..tok_99
        for i in 91..100u32 {
            assert!(idx.contains(&format!("tok_{i}")));
        }
    }
}
