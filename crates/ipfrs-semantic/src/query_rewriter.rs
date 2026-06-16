//! Semantic query rewriting and expansion system.
//!
//! Provides query transformation, synonym expansion, stemming normalization,
//! stop word removal, phrase detection, and weighted boosting for semantic search.

use std::collections::{HashMap, HashSet};
use std::time::Instant;

/// Rewrite rule types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RewriteRuleType {
    SynonymExpansion,
    StemmingNormalization,
    PhraseDetection,
    StopWordRemoval,
    SpellingCorrection,
    ConceptExpansion,
    WeightBoosting,
}

/// A single rewrite rule
#[derive(Debug, Clone)]
pub struct RewriteRule {
    pub rule_type: RewriteRuleType,
    pub pattern: String,
    pub replacement: String,
    pub boost: f64,
    pub enabled: bool,
    pub priority: u32,
}

/// Rewritten query term
#[derive(Debug, Clone)]
pub struct RewrittenTerm {
    pub original: String,
    pub rewritten: String,
    pub boost: f64,
    pub source_rule: RewriteRuleType,
}

/// Full rewrite result
#[derive(Debug, Clone)]
pub struct RewriteResult {
    pub original_query: String,
    pub rewritten_query: String,
    pub terms: Vec<RewrittenTerm>,
    pub expansions: Vec<String>,
    pub rules_applied: usize,
    pub processing_time_us: u64,
}

/// Configuration for the query rewriter
#[derive(Debug, Clone)]
pub struct QueryRewriterConfig {
    pub max_expansions: usize,
    pub min_term_length: usize,
    pub enable_stemming: bool,
    pub enable_stop_word_removal: bool,
    pub default_boost: f64,
    pub max_rules: usize,
}

impl Default for QueryRewriterConfig {
    fn default() -> Self {
        Self {
            max_expansions: 10,
            min_term_length: 2,
            enable_stemming: true,
            enable_stop_word_removal: true,
            default_boost: 1.0,
            max_rules: 1000,
        }
    }
}

/// Statistics for the query rewriter
#[derive(Debug, Clone, Default)]
pub struct QueryRewriterStats {
    pub queries_processed: u64,
    pub rules_applied: u64,
    pub expansions_generated: u64,
    pub avg_processing_time_us: f64,
}

/// Query rewriter engine
pub struct QueryRewriter {
    config: QueryRewriterConfig,
    rules: Vec<RewriteRule>,
    stop_words: HashSet<String>,
    synonym_map: HashMap<String, Vec<(String, f64)>>,
    stats: QueryRewriterStats,
}

impl QueryRewriter {
    /// Create a new query rewriter with the given configuration.
    pub fn new(config: QueryRewriterConfig) -> Self {
        Self {
            config,
            rules: Vec::new(),
            stop_words: HashSet::new(),
            synonym_map: HashMap::new(),
            stats: QueryRewriterStats::default(),
        }
    }

    /// Add a rewrite rule. Returns an error if the max rules limit is reached.
    pub fn add_rule(&mut self, rule: RewriteRule) -> Result<(), String> {
        if self.rules.len() >= self.config.max_rules {
            return Err(format!(
                "Maximum number of rules ({}) reached",
                self.config.max_rules
            ));
        }
        self.rules.push(rule);
        // Keep rules sorted by priority (lower number = higher priority)
        self.rules.sort_by_key(|r| r.priority);
        Ok(())
    }

    /// Add a synonym mapping with a boost factor.
    pub fn add_synonym(&mut self, word: &str, synonym: &str, boost: f64) {
        let key = word.to_lowercase();
        let entry = self.synonym_map.entry(key).or_default();
        entry.push((synonym.to_lowercase(), boost));
    }

    /// Add a stop word.
    pub fn add_stop_word(&mut self, word: &str) {
        self.stop_words.insert(word.to_lowercase());
    }

    /// Rewrite a query, applying all enabled rules, synonym expansion,
    /// stemming, and stop word removal.
    pub fn rewrite(&mut self, query: &str) -> RewriteResult {
        let start = Instant::now();
        let normalized = Self::normalize_query(query);
        let original_query = normalized.clone();

        if normalized.is_empty() {
            let elapsed = start.elapsed().as_micros() as u64;
            self.update_stats(0, 0, elapsed);
            return RewriteResult {
                original_query,
                rewritten_query: String::new(),
                terms: Vec::new(),
                expansions: Vec::new(),
                rules_applied: 0,
                processing_time_us: elapsed,
            };
        }

        let mut tokens = Self::tokenize(&normalized);
        let mut all_terms: Vec<RewrittenTerm> = Vec::new();
        let mut all_expansions: Vec<String> = Vec::new();
        let mut total_rules_applied: usize = 0;

        // Phase 1: Apply explicit rewrite rules (pattern-based)
        let mut rewritten_tokens: Vec<String> = Vec::new();
        for token in &tokens {
            let mut current = token.clone();
            for rule in &self.rules {
                if !rule.enabled {
                    continue;
                }
                if rule.pattern.to_lowercase() == current {
                    all_terms.push(RewrittenTerm {
                        original: token.clone(),
                        rewritten: rule.replacement.clone(),
                        boost: rule.boost,
                        source_rule: rule.rule_type.clone(),
                    });
                    current = rule.replacement.clone();
                    total_rules_applied += 1;
                    break; // Only apply highest-priority matching rule per token
                }
            }
            rewritten_tokens.push(current);
        }
        tokens = rewritten_tokens;

        // Phase 2: Stop word removal
        if self.config.enable_stop_word_removal {
            let before_len = tokens.len();
            tokens.retain(|t| !self.is_stop_word(t));
            let removed = before_len - tokens.len();
            if removed > 0 {
                total_rules_applied += removed;
            }
        }

        // Phase 3: Min term length filtering
        tokens.retain(|t| t.len() >= self.config.min_term_length);

        // Phase 4: Stemming
        if self.config.enable_stemming {
            let mut stemmed_tokens = Vec::new();
            for token in &tokens {
                let stemmed = Self::apply_stemming(token);
                if stemmed != *token {
                    all_terms.push(RewrittenTerm {
                        original: token.clone(),
                        rewritten: stemmed.clone(),
                        boost: self.config.default_boost,
                        source_rule: RewriteRuleType::StemmingNormalization,
                    });
                    total_rules_applied += 1;
                }
                stemmed_tokens.push(stemmed);
            }
            tokens = stemmed_tokens;
        }

        // Phase 5: Synonym expansion
        let mut expansion_count: usize = 0;
        for token in &tokens {
            if expansion_count >= self.config.max_expansions {
                break;
            }
            let synonyms = self.expand_synonyms(token);
            for (syn, boost) in synonyms {
                if expansion_count >= self.config.max_expansions {
                    break;
                }
                all_terms.push(RewrittenTerm {
                    original: token.clone(),
                    rewritten: syn.clone(),
                    boost,
                    source_rule: RewriteRuleType::SynonymExpansion,
                });
                all_expansions.push(syn);
                expansion_count += 1;
                total_rules_applied += 1;
            }
        }

        let rewritten_query = tokens.join(" ");
        let elapsed = start.elapsed().as_micros() as u64;

        self.update_stats(total_rules_applied, expansion_count, elapsed);

        RewriteResult {
            original_query,
            rewritten_query,
            terms: all_terms,
            expansions: all_expansions,
            rules_applied: total_rules_applied,
            processing_time_us: elapsed,
        }
    }

    /// Tokenize a query string: split on whitespace, lowercase, remove non-alphanumeric chars.
    pub fn tokenize(query: &str) -> Vec<String> {
        query
            .split_whitespace()
            .map(|w| {
                w.chars()
                    .filter(|c| c.is_alphanumeric())
                    .collect::<String>()
                    .to_lowercase()
            })
            .filter(|w| !w.is_empty())
            .collect()
    }

    /// Simple suffix-stripping stemmer (Porter-like).
    /// Handles: -ing, -ed, -ly, -tion, -ness, -ment
    pub fn apply_stemming(word: &str) -> String {
        if word.len() < 4 {
            return word.to_string();
        }

        // Try suffixes from longest to shortest to avoid partial matches
        let suffixes: &[&str] = &["tion", "ness", "ment", "ing", "ed", "ly"];
        for suffix in suffixes {
            if let Some(stem) = word.strip_suffix(suffix) {
                // Only accept stems of reasonable length
                if stem.len() >= 2 {
                    return stem.to_string();
                }
            }
        }

        word.to_string()
    }

    /// Check if a word is a stop word.
    pub fn is_stop_word(&self, word: &str) -> bool {
        self.stop_words.contains(&word.to_lowercase())
    }

    /// Expand a term into its synonyms with associated boost values.
    pub fn expand_synonyms(&self, term: &str) -> Vec<(String, f64)> {
        self.synonym_map
            .get(&term.to_lowercase())
            .cloned()
            .unwrap_or_default()
    }

    /// Return the number of rules.
    pub fn rules_count(&self) -> usize {
        self.rules.len()
    }

    /// Remove all rules matching the given pattern. Returns true if any were removed.
    pub fn remove_rule(&mut self, pattern: &str) -> bool {
        let before = self.rules.len();
        self.rules
            .retain(|r| r.pattern.to_lowercase() != pattern.to_lowercase());
        self.rules.len() < before
    }

    /// Clear all rules.
    pub fn clear_rules(&mut self) {
        self.rules.clear();
    }

    /// Get current statistics.
    pub fn stats(&self) -> &QueryRewriterStats {
        &self.stats
    }

    /// Normalize a query: lowercase, trim, collapse multiple whitespace into single space.
    pub fn normalize_query(query: &str) -> String {
        let trimmed = query.trim().to_lowercase();
        let mut result = String::with_capacity(trimmed.len());
        let mut prev_was_space = false;
        for ch in trimmed.chars() {
            if ch.is_whitespace() {
                if !prev_was_space {
                    result.push(' ');
                    prev_was_space = true;
                }
            } else {
                result.push(ch);
                prev_was_space = false;
            }
        }
        result
    }

    /// Update internal statistics after processing a query.
    fn update_stats(&mut self, rules_applied: usize, expansions: usize, elapsed_us: u64) {
        let n = self.stats.queries_processed;
        self.stats.queries_processed += 1;
        self.stats.rules_applied += rules_applied as u64;
        self.stats.expansions_generated += expansions as u64;

        // Running average
        let new_n = self.stats.queries_processed as f64;
        self.stats.avg_processing_time_us =
            (self.stats.avg_processing_time_us * n as f64 + elapsed_us as f64) / new_n;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_rewriter() -> QueryRewriter {
        QueryRewriter::new(QueryRewriterConfig::default())
    }

    #[test]
    fn test_basic_rewrite_single_rule() {
        let mut rw = default_rewriter();
        rw.add_rule(RewriteRule {
            rule_type: RewriteRuleType::SpellingCorrection,
            pattern: "teh".to_string(),
            replacement: "the".to_string(),
            boost: 1.0,
            enabled: true,
            priority: 1,
        })
        .expect("should add rule");

        let result = rw.rewrite("teh cat");
        assert_eq!(result.rewritten_query, "the cat");
        assert!(result.rules_applied >= 1);
    }

    #[test]
    fn test_multiple_rules_priority_order() {
        let mut rw = default_rewriter();
        // Lower priority number = applied first
        rw.add_rule(RewriteRule {
            rule_type: RewriteRuleType::SpellingCorrection,
            pattern: "foo".to_string(),
            replacement: "bar".to_string(),
            boost: 1.0,
            enabled: true,
            priority: 10,
        })
        .expect("add rule");
        rw.add_rule(RewriteRule {
            rule_type: RewriteRuleType::SpellingCorrection,
            pattern: "foo".to_string(),
            replacement: "baz".to_string(),
            boost: 1.0,
            enabled: true,
            priority: 1,
        })
        .expect("add rule");

        // The priority=1 rule should be first and applied
        let result = rw.rewrite("foo");
        assert_eq!(result.rewritten_query, "baz");
    }

    #[test]
    fn test_synonym_expansion() {
        let mut rw = default_rewriter();
        rw.add_synonym("fast", "quick", 0.9);
        rw.add_synonym("fast", "rapid", 0.8);

        let result = rw.rewrite("fast car");
        assert!(result.expansions.contains(&"quick".to_string()));
        assert!(result.expansions.contains(&"rapid".to_string()));
    }

    #[test]
    fn test_stop_word_removal() {
        let mut rw = default_rewriter();
        rw.add_stop_word("the");
        rw.add_stop_word("is");
        rw.add_stop_word("a");

        let result = rw.rewrite("the cat is a pet");
        // "the", "is", "a" removed; "cat" and "pet" remain
        assert!(!result.rewritten_query.contains("the"));
        assert!(!result.rewritten_query.contains(" is "));
        assert!(result.rewritten_query.contains("cat"));
        assert!(result.rewritten_query.contains("pet"));
    }

    #[test]
    fn test_stemming_ing() {
        let stemmed = QueryRewriter::apply_stemming("running");
        assert_eq!(stemmed, "runn");
    }

    #[test]
    fn test_stemming_ed() {
        let stemmed = QueryRewriter::apply_stemming("walked");
        assert_eq!(stemmed, "walk");
    }

    #[test]
    fn test_stemming_ly() {
        let stemmed = QueryRewriter::apply_stemming("quickly");
        assert_eq!(stemmed, "quick");
    }

    #[test]
    fn test_stemming_tion() {
        let stemmed = QueryRewriter::apply_stemming("creation");
        assert_eq!(stemmed, "crea");
    }

    #[test]
    fn test_stemming_ness() {
        let stemmed = QueryRewriter::apply_stemming("darkness");
        assert_eq!(stemmed, "dark");
    }

    #[test]
    fn test_stemming_ment() {
        let stemmed = QueryRewriter::apply_stemming("agreement");
        assert_eq!(stemmed, "agree");
    }

    #[test]
    fn test_stemming_short_word_unchanged() {
        let stemmed = QueryRewriter::apply_stemming("go");
        assert_eq!(stemmed, "go");
    }

    #[test]
    fn test_phrase_detection_rule() {
        let mut rw = default_rewriter();
        rw.add_rule(RewriteRule {
            rule_type: RewriteRuleType::PhraseDetection,
            pattern: "machine".to_string(),
            replacement: "machine_learning".to_string(),
            boost: 1.5,
            enabled: true,
            priority: 1,
        })
        .expect("add rule");

        let result = rw.rewrite("machine algorithms");
        // The term "machine" should be rewritten to "machine_learning"
        let has_phrase = result
            .terms
            .iter()
            .any(|t| t.rewritten == "machine_learning");
        assert!(has_phrase);
    }

    #[test]
    fn test_boost_application() {
        let mut rw = default_rewriter();
        rw.add_synonym("search", "lookup", 2.5);

        let result = rw.rewrite("search");
        let syn_term = result
            .terms
            .iter()
            .find(|t| t.rewritten == "lookup")
            .expect("should have synonym term");
        assert!((syn_term.boost - 2.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_empty_query() {
        let mut rw = default_rewriter();
        let result = rw.rewrite("");
        assert!(result.rewritten_query.is_empty());
        assert!(result.terms.is_empty());
        assert_eq!(result.rules_applied, 0);
    }

    #[test]
    fn test_whitespace_only_query() {
        let mut rw = default_rewriter();
        let result = rw.rewrite("   \t\n  ");
        assert!(result.rewritten_query.is_empty());
    }

    #[test]
    fn test_very_long_query() {
        let mut rw = default_rewriter();
        let long_query = "word ".repeat(500);
        let result = rw.rewrite(&long_query);
        assert!(!result.rewritten_query.is_empty());
    }

    #[test]
    fn test_rule_add_remove_clear() {
        let mut rw = default_rewriter();
        assert_eq!(rw.rules_count(), 0);

        rw.add_rule(RewriteRule {
            rule_type: RewriteRuleType::SpellingCorrection,
            pattern: "abc".to_string(),
            replacement: "def".to_string(),
            boost: 1.0,
            enabled: true,
            priority: 1,
        })
        .expect("add");
        assert_eq!(rw.rules_count(), 1);

        let removed = rw.remove_rule("abc");
        assert!(removed);
        assert_eq!(rw.rules_count(), 0);

        let not_removed = rw.remove_rule("nonexistent");
        assert!(!not_removed);

        rw.add_rule(RewriteRule {
            rule_type: RewriteRuleType::SpellingCorrection,
            pattern: "x".to_string(),
            replacement: "y".to_string(),
            boost: 1.0,
            enabled: true,
            priority: 1,
        })
        .expect("add");
        rw.clear_rules();
        assert_eq!(rw.rules_count(), 0);
    }

    #[test]
    fn test_stats_tracking() {
        let mut rw = default_rewriter();
        assert_eq!(rw.stats().queries_processed, 0);

        rw.rewrite("hello world");
        assert_eq!(rw.stats().queries_processed, 1);

        rw.rewrite("another query");
        assert_eq!(rw.stats().queries_processed, 2);
    }

    #[test]
    fn test_max_expansions_limit() {
        let mut rw = QueryRewriter::new(QueryRewriterConfig {
            max_expansions: 2,
            ..Default::default()
        });
        rw.add_synonym("test", "exam", 1.0);
        rw.add_synonym("test", "trial", 1.0);
        rw.add_synonym("test", "assessment", 1.0);
        rw.add_synonym("test", "evaluation", 1.0);

        let result = rw.rewrite("test");
        assert!(result.expansions.len() <= 2);
    }

    #[test]
    fn test_min_term_length_filtering() {
        let mut rw = QueryRewriter::new(QueryRewriterConfig {
            min_term_length: 3,
            enable_stop_word_removal: false,
            enable_stemming: false,
            ..Default::default()
        });

        let result = rw.rewrite("I am a big cat");
        // "I", "am", "a" should be filtered (length < 3)
        assert!(!result.rewritten_query.contains(" i "));
        assert!(result.rewritten_query.contains("big"));
        assert!(result.rewritten_query.contains("cat"));
    }

    #[test]
    fn test_combined_stemming_synonyms_stopwords() {
        let mut rw = default_rewriter();
        rw.add_stop_word("the");
        rw.add_synonym("run", "jog", 0.9);

        // "the" removed, "running" stemmed to "runn" (simple stemmer)
        let result = rw.rewrite("the running dog");
        assert!(!result.rewritten_query.contains("the"));
    }

    #[test]
    fn test_tokenization_correctness() {
        let tokens = QueryRewriter::tokenize("Hello, World! Test-case 123");
        assert_eq!(tokens, vec!["hello", "world", "testcase", "123"]);
    }

    #[test]
    fn test_tokenization_empty() {
        let tokens = QueryRewriter::tokenize("");
        assert!(tokens.is_empty());
    }

    #[test]
    fn test_query_normalization_whitespace() {
        let normalized = QueryRewriter::normalize_query("  hello   world  ");
        assert_eq!(normalized, "hello world");
    }

    #[test]
    fn test_query_normalization_case() {
        let normalized = QueryRewriter::normalize_query("Hello WORLD");
        assert_eq!(normalized, "hello world");
    }

    #[test]
    fn test_duplicate_rule_handling() {
        let mut rw = default_rewriter();
        // Adding two rules with same pattern but different replacements
        rw.add_rule(RewriteRule {
            rule_type: RewriteRuleType::SpellingCorrection,
            pattern: "test".to_string(),
            replacement: "exam".to_string(),
            boost: 1.0,
            enabled: true,
            priority: 2,
        })
        .expect("add");
        rw.add_rule(RewriteRule {
            rule_type: RewriteRuleType::SpellingCorrection,
            pattern: "test".to_string(),
            replacement: "quiz".to_string(),
            boost: 1.0,
            enabled: true,
            priority: 1,
        })
        .expect("add");

        // Both stored, but only highest priority (priority=1) applied
        assert_eq!(rw.rules_count(), 2);
        let result = rw.rewrite("test");
        assert_eq!(result.rewritten_query, "quiz");
    }

    #[test]
    fn test_rule_priority_ordering() {
        let mut rw = default_rewriter();
        rw.add_rule(RewriteRule {
            rule_type: RewriteRuleType::WeightBoosting,
            pattern: "alpha".to_string(),
            replacement: "beta".to_string(),
            boost: 1.0,
            enabled: true,
            priority: 100,
        })
        .expect("add");
        rw.add_rule(RewriteRule {
            rule_type: RewriteRuleType::WeightBoosting,
            pattern: "gamma".to_string(),
            replacement: "delta".to_string(),
            boost: 1.0,
            enabled: true,
            priority: 1,
        })
        .expect("add");

        // Internal ordering: priority=1 first, then priority=100
        assert_eq!(rw.rules[0].pattern, "gamma");
        assert_eq!(rw.rules[1].pattern, "alpha");
    }

    #[test]
    fn test_processing_time_tracked() {
        let mut rw = default_rewriter();
        let result = rw.rewrite("some query");
        // processing_time_us should be set (at least 0)
        assert!(result.processing_time_us < 1_000_000); // less than 1 second
        assert!(rw.stats().avg_processing_time_us >= 0.0);
    }

    #[test]
    fn test_config_defaults() {
        let cfg = QueryRewriterConfig::default();
        assert_eq!(cfg.max_expansions, 10);
        assert_eq!(cfg.min_term_length, 2);
        assert!(cfg.enable_stemming);
        assert!(cfg.enable_stop_word_removal);
        assert!((cfg.default_boost - 1.0).abs() < f64::EPSILON);
        assert_eq!(cfg.max_rules, 1000);
    }

    #[test]
    fn test_disabled_rule_not_applied() {
        let mut rw = default_rewriter();
        rw.add_rule(RewriteRule {
            rule_type: RewriteRuleType::SpellingCorrection,
            pattern: "hello".to_string(),
            replacement: "hi".to_string(),
            boost: 1.0,
            enabled: false,
            priority: 1,
        })
        .expect("add");

        let result = rw.rewrite("hello");
        // Rule is disabled, so "hello" should not become "hi"
        // (stemming may still apply, but the rule replacement should not)
        let has_rule_term = result
            .terms
            .iter()
            .any(|t| t.source_rule == RewriteRuleType::SpellingCorrection);
        assert!(!has_rule_term);
    }

    #[test]
    fn test_max_rules_limit() {
        let mut rw = QueryRewriter::new(QueryRewriterConfig {
            max_rules: 2,
            ..Default::default()
        });

        rw.add_rule(RewriteRule {
            rule_type: RewriteRuleType::SpellingCorrection,
            pattern: "a".to_string(),
            replacement: "b".to_string(),
            boost: 1.0,
            enabled: true,
            priority: 1,
        })
        .expect("first");

        rw.add_rule(RewriteRule {
            rule_type: RewriteRuleType::SpellingCorrection,
            pattern: "c".to_string(),
            replacement: "d".to_string(),
            boost: 1.0,
            enabled: true,
            priority: 2,
        })
        .expect("second");

        let res = rw.add_rule(RewriteRule {
            rule_type: RewriteRuleType::SpellingCorrection,
            pattern: "e".to_string(),
            replacement: "f".to_string(),
            boost: 1.0,
            enabled: true,
            priority: 3,
        });
        assert!(res.is_err());
    }

    #[test]
    fn test_concept_expansion_rule_type() {
        let mut rw = default_rewriter();
        rw.add_rule(RewriteRule {
            rule_type: RewriteRuleType::ConceptExpansion,
            pattern: "ml".to_string(),
            replacement: "machine_learning".to_string(),
            boost: 1.2,
            enabled: true,
            priority: 1,
        })
        .expect("add");

        let result = rw.rewrite("ml models");
        let term = result
            .terms
            .iter()
            .find(|t| t.source_rule == RewriteRuleType::ConceptExpansion);
        assert!(term.is_some());
    }

    #[test]
    fn test_stats_rules_applied_count() {
        let mut rw = default_rewriter();
        rw.add_synonym("dog", "canine", 1.0);
        rw.add_stop_word("the");

        rw.rewrite("the dog");
        assert!(rw.stats().rules_applied > 0);
        assert!(rw.stats().expansions_generated > 0);
    }

    #[test]
    fn test_special_characters_in_query() {
        let mut rw = default_rewriter();
        let result = rw.rewrite("hello! @world #test");
        // Special chars stripped by tokenizer
        let tokens = QueryRewriter::tokenize("hello! @world #test");
        assert_eq!(tokens, vec!["hello", "world", "test"]);
        assert!(!result.rewritten_query.is_empty());
    }

    #[test]
    fn test_synonym_case_insensitive() {
        let mut rw = default_rewriter();
        rw.add_synonym("Dog", "canine", 1.0);

        let result = rw.rewrite("DOG");
        // Synonym lookup is case-insensitive
        assert!(result.expansions.contains(&"canine".to_string()));
    }

    #[test]
    fn test_remove_rule_case_insensitive() {
        let mut rw = default_rewriter();
        rw.add_rule(RewriteRule {
            rule_type: RewriteRuleType::SpellingCorrection,
            pattern: "Hello".to_string(),
            replacement: "hi".to_string(),
            boost: 1.0,
            enabled: true,
            priority: 1,
        })
        .expect("add");

        let removed = rw.remove_rule("hello");
        assert!(removed);
        assert_eq!(rw.rules_count(), 0);
    }
}
