//! # Query Expansion Engine
//!
//! Enriches search queries with synonyms, related terms, and contextual expansions
//! to improve recall in semantic search pipelines.
//!
//! ## Overview
//!
//! The [`QueryExpansionEngine`] accepts a raw query string, tokenises it, and for
//! each token looks up a [`SynonymEntry`] registry and a context-term map.  It
//! returns a [`QeExpandedQuery`] containing the original query, a deduplicated /
//! weighted set of [`QeExpansionTerm`]s, and an optional pre-computed query
//! embedding.
//!
//! ## Example
//!
//! ```rust
//! use ipfrs_semantic::query_expansion::{
//!     ExpansionConfig, QueryExpansionEngine, SynonymEntry,
//! };
//!
//! let config = ExpansionConfig::default();
//! let mut engine = QueryExpansionEngine::new(config);
//!
//! engine.add_synonym_entry(SynonymEntry {
//!     term: "car".to_string(),
//!     synonyms: vec!["automobile".to_string(), "vehicle".to_string()],
//!     hypernyms: vec!["transport".to_string()],
//!     hyponyms: vec!["sedan".to_string(), "suv".to_string()],
//! });
//!
//! let expanded = engine.expand_query("fast car", None);
//! assert!(!expanded.terms.is_empty());
//! ```

use std::collections::HashMap;

// ──────────────────────────────────────────────────────────────────────────────
// Core data types
// ──────────────────────────────────────────────────────────────────────────────

/// The origin of an expansion term.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ExpansionSource {
    /// Direct synonym of the original token.
    Synonym,
    /// A more general / broader term (hypernym).
    Hypernym,
    /// A more specific / narrower term (hyponym).
    Hyponym,
    /// A contextually related term from the context map.
    RelatedTerm,
    /// Contextual expansion (also used for the original tokens themselves).
    ContextualExpansion,
}

impl ExpansionSource {
    /// Human-readable name used in statistics maps.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Synonym => "Synonym",
            Self::Hypernym => "Hypernym",
            Self::Hyponym => "Hyponym",
            Self::RelatedTerm => "RelatedTerm",
            Self::ContextualExpansion => "ContextualExpansion",
        }
    }
}

/// A single term in an expanded query, annotated with its weight and origin.
#[derive(Clone, Debug)]
pub struct QeExpansionTerm {
    /// The term string (always lowercased).
    pub term: String,
    /// Relevance weight in `(0.0, 1.0]` for expansion terms, or `>1.0` for
    /// boosted original tokens.
    pub weight: f64,
    /// Where this term came from.
    pub source: ExpansionSource,
}

/// A lexical entry stored in the synonym database.
#[derive(Clone, Debug)]
pub struct SynonymEntry {
    /// Canonical term (will be lowercased when stored).
    pub term: String,
    /// Direct synonyms (e.g., "car" → ["automobile", "vehicle"]).
    pub synonyms: Vec<String>,
    /// More general / broader terms (hypernyms).
    pub hypernyms: Vec<String>,
    /// More specific / narrower terms (hyponyms).
    pub hyponyms: Vec<String>,
}

/// Tuning knobs for [`QueryExpansionEngine`].
#[derive(Clone, Debug)]
pub struct ExpansionConfig {
    /// Maximum number of expansion terms added **per original token**.
    pub max_expansions_per_term: usize,
    /// Minimum weight an expansion must have to be kept.
    pub min_weight: f64,
    /// Whether to add synonyms.
    pub include_synonyms: bool,
    /// Whether to add hypernyms.
    pub include_hypernyms: bool,
    /// Whether to add hyponyms.
    pub include_hyponyms: bool,
    /// Whether to add context-map related terms.
    pub include_related: bool,
    /// Boost factor applied to the original query tokens (weight = this value).
    pub boost_exact_match: f64,
}

impl Default for ExpansionConfig {
    fn default() -> Self {
        Self {
            max_expansions_per_term: 5,
            min_weight: 0.3,
            include_synonyms: true,
            include_hypernyms: true,
            include_hyponyms: true,
            include_related: true,
            boost_exact_match: 2.0,
        }
    }
}

/// The result of expanding a query string.
#[derive(Clone, Debug)]
pub struct QeExpandedQuery {
    /// The raw, unmodified query string.
    pub original: String,
    /// All expansion terms (including the originals, boosted).
    pub terms: Vec<QeExpansionTerm>,
    /// An optional pre-computed embedding for the full query.
    pub query_embedding: Option<Vec<f64>>,
}

/// Summary statistics about how much a query was expanded.
#[derive(Clone, Debug)]
pub struct ExpansionStats {
    /// Number of unique tokens in the original query.
    pub original_term_count: usize,
    /// Total number of terms in the expanded query (including originals).
    pub expanded_term_count: usize,
    /// `expanded_term_count / original_term_count` (or `0.0` if there are no
    /// original terms).
    pub expansion_ratio: f64,
    /// Number of terms contributed by each [`ExpansionSource`] variant.
    pub sources: HashMap<String, usize>,
}

// ──────────────────────────────────────────────────────────────────────────────
// Engine
// ──────────────────────────────────────────────────────────────────────────────

/// Production-grade query expansion engine.
///
/// # Thread Safety
///
/// The engine holds no shared state — callers that need concurrent access should
/// wrap it in a `Mutex` / `RwLock` (write lock only for `expand_query` because
/// it mutates the counters).
pub struct QueryExpansionEngine {
    /// User-supplied configuration.
    pub config: ExpansionConfig,
    /// Synonym / hypernym / hyponym registry, keyed by lowercased term.
    pub synonym_db: HashMap<String, SynonymEntry>,
    /// Context-term map: trigger word → list of related terms.
    pub context_terms: HashMap<String, Vec<String>>,
    /// Total number of `expand_query` calls.
    pub expansions_performed: u64,
    /// Cumulative count of expansion terms added across all calls.
    pub total_terms_added: u64,
}

impl QueryExpansionEngine {
    // ── Construction ──────────────────────────────────────────────────────────

    /// Create a new engine with the given configuration.
    pub fn new(config: ExpansionConfig) -> Self {
        Self {
            config,
            synonym_db: HashMap::new(),
            context_terms: HashMap::new(),
            expansions_performed: 0,
            total_terms_added: 0,
        }
    }

    // ── Registry mutations ─────────────────────────────────────────────────────

    /// Insert or replace a [`SynonymEntry`] in the synonym database.
    ///
    /// The `entry.term` is lowercased before use as the map key so that
    /// lookups are case-insensitive.
    pub fn add_synonym_entry(&mut self, entry: SynonymEntry) {
        let key = entry.term.to_lowercase();
        self.synonym_db.insert(key, entry);
    }

    /// Register a list of related terms that fire when `trigger` appears in a
    /// query.
    ///
    /// If an entry for `trigger` already exists the new terms are **appended**
    /// (deduplication is left to the caller).
    pub fn add_context_terms(&mut self, trigger: String, related: Vec<String>) {
        let key = trigger.to_lowercase();
        self.context_terms.entry(key).or_default().extend(related);
    }

    // ── Per-token expansion ───────────────────────────────────────────────────

    /// Return expansion terms for a single `term`, respecting the current
    /// [`ExpansionConfig`].
    ///
    /// The returned list is sorted by weight descending and capped at
    /// `config.max_expansions_per_term`.  Terms below `config.min_weight` are
    /// excluded.
    pub fn expand_term(&self, term: &str) -> Vec<QeExpansionTerm> {
        let lower = term.to_lowercase();
        let mut candidates: Vec<QeExpansionTerm> = Vec::new();

        // ── Synonym-database lookup ──────────────────────────────────────────
        if let Some(entry) = self.synonym_db.get(&lower) {
            if self.config.include_synonyms {
                for s in &entry.synonyms {
                    candidates.push(QeExpansionTerm {
                        term: s.to_lowercase(),
                        weight: 0.9,
                        source: ExpansionSource::Synonym,
                    });
                }
            }

            if self.config.include_hypernyms {
                for h in &entry.hypernyms {
                    candidates.push(QeExpansionTerm {
                        term: h.to_lowercase(),
                        weight: 0.7,
                        source: ExpansionSource::Hypernym,
                    });
                }
            }

            if self.config.include_hyponyms {
                for h in &entry.hyponyms {
                    candidates.push(QeExpansionTerm {
                        term: h.to_lowercase(),
                        weight: 0.6,
                        source: ExpansionSource::Hyponym,
                    });
                }
            }
        }

        // ── Context-term lookup ──────────────────────────────────────────────
        if self.config.include_related {
            if let Some(related) = self.context_terms.get(&lower) {
                for r in related {
                    candidates.push(QeExpansionTerm {
                        term: r.to_lowercase(),
                        weight: 0.5,
                        source: ExpansionSource::RelatedTerm,
                    });
                }
            }
        }

        // ── Filter, sort, cap ────────────────────────────────────────────────
        candidates.retain(|t| t.weight >= self.config.min_weight);
        candidates.sort_by(|a, b| {
            b.weight
                .partial_cmp(&a.weight)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        candidates.truncate(self.config.max_expansions_per_term);
        candidates
    }

    // ── Full query expansion ──────────────────────────────────────────────────

    /// Tokenise `query`, expand every token, and return a [`QeExpandedQuery`].
    ///
    /// Tokenisation splits on ASCII whitespace and the punctuation characters
    /// `.`, `,`, `!`, `?`, `;`, `:`.  Empty tokens are discarded.
    ///
    /// The `embedding` argument, if provided, is stored verbatim in the result
    /// (e.g. a pre-computed dense vector for the full query string).
    pub fn expand_query(&mut self, query: &str, embedding: Option<Vec<f64>>) -> QeExpandedQuery {
        // ── Tokenise ─────────────────────────────────────────────────────────
        let tokens: Vec<String> = query
            .split(|c: char| {
                c.is_ascii_whitespace() || matches!(c, '.' | ',' | '!' | '?' | ';' | ':')
            })
            .filter(|s| !s.is_empty())
            .map(|s| s.to_lowercase())
            .collect();

        // Unique tokens (preserve first-occurrence order).
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        let unique_tokens: Vec<String> = tokens
            .into_iter()
            .filter(|t| seen.insert(t.clone()))
            .collect();

        // ── Collect all terms ─────────────────────────────────────────────────
        // term → best weight seen so far
        let mut term_map: HashMap<String, QeExpansionTerm> = HashMap::new();

        // 1. Original tokens (boosted)
        for tok in &unique_tokens {
            let candidate = QeExpansionTerm {
                term: tok.clone(),
                weight: self.config.boost_exact_match,
                source: ExpansionSource::ContextualExpansion,
            };
            Self::upsert_term(&mut term_map, candidate);
        }

        // 2. Expanded terms
        for tok in &unique_tokens {
            for exp in self.expand_term(tok) {
                Self::upsert_term(&mut term_map, exp);
            }
        }

        // ── Build final sorted term list ──────────────────────────────────────
        let mut terms: Vec<QeExpansionTerm> = term_map.into_values().collect();
        terms.sort_by(|a, b| {
            b.weight
                .partial_cmp(&a.weight)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // ── Update counters ───────────────────────────────────────────────────
        let expansion_count = terms
            .iter()
            .filter(|t| !matches!(t.source, ExpansionSource::ContextualExpansion))
            .count() as u64;

        self.expansions_performed += 1;
        self.total_terms_added += expansion_count;

        QeExpandedQuery {
            original: query.to_string(),
            terms,
            query_embedding: embedding,
        }
    }

    // ── Post-expansion utilities ──────────────────────────────────────────────

    /// Build a space-separated search string from all terms whose weight is at
    /// or above `config.min_weight`.
    ///
    /// The original tokens (source = `ContextualExpansion`) are placed first,
    /// followed by expansion terms, both groups sorted by weight descending.
    pub fn build_search_string(&self, expanded: &QeExpandedQuery) -> String {
        let min_w = self.config.min_weight;

        let mut originals: Vec<&QeExpansionTerm> = expanded
            .terms
            .iter()
            .filter(|t| {
                t.weight >= min_w && matches!(t.source, ExpansionSource::ContextualExpansion)
            })
            .collect();

        let mut expansions: Vec<&QeExpansionTerm> = expanded
            .terms
            .iter()
            .filter(|t| {
                t.weight >= min_w && !matches!(t.source, ExpansionSource::ContextualExpansion)
            })
            .collect();

        originals.sort_by(|a, b| {
            b.weight
                .partial_cmp(&a.weight)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        expansions.sort_by(|a, b| {
            b.weight
                .partial_cmp(&a.weight)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        originals
            .iter()
            .chain(expansions.iter())
            .map(|t| t.term.as_str())
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Return `(term, weight)` pairs sorted by weight descending.
    pub fn weighted_terms<'a>(&self, expanded: &'a QeExpandedQuery) -> Vec<(&'a str, f64)> {
        let mut pairs: Vec<(&str, f64)> = expanded
            .terms
            .iter()
            .map(|t| (t.term.as_str(), t.weight))
            .collect();
        pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        pairs
    }

    /// Compute expansion statistics for a [`QeExpandedQuery`].
    pub fn coverage_stats(&self, expanded: &QeExpandedQuery) -> ExpansionStats {
        // "Original" tokens are those with source == ContextualExpansion and
        // weight == boost_exact_match.  We count unique such terms.
        let original_term_count = expanded
            .terms
            .iter()
            .filter(|t| matches!(t.source, ExpansionSource::ContextualExpansion))
            .count();

        let expanded_term_count = expanded.terms.len();

        let expansion_ratio = if original_term_count == 0 {
            0.0
        } else {
            expanded_term_count as f64 / original_term_count as f64
        };

        let mut sources: HashMap<String, usize> = HashMap::new();
        for t in &expanded.terms {
            *sources.entry(t.source.as_str().to_string()).or_insert(0) += 1;
        }

        ExpansionStats {
            original_term_count,
            expanded_term_count,
            expansion_ratio,
            sources,
        }
    }

    /// Return `(expansions_performed, total_terms_added)`.
    pub fn engine_stats(&self) -> (u64, u64) {
        (self.expansions_performed, self.total_terms_added)
    }

    // ── Private helpers ────────────────────────────────────────────────────────

    /// Insert `candidate` into `map`, keeping the highest weight when a term
    /// already exists.
    fn upsert_term(map: &mut HashMap<String, QeExpansionTerm>, candidate: QeExpansionTerm) {
        map.entry(candidate.term.clone())
            .and_modify(|existing| {
                if candidate.weight > existing.weight {
                    *existing = candidate.clone();
                }
            })
            .or_insert(candidate);
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::{
        ExpansionConfig, ExpansionSource, QeExpandedQuery, QueryExpansionEngine, SynonymEntry,
    };
    use std::collections::HashMap;

    // ── Helpers ────────────────────────────────────────────────────────────────

    fn car_entry() -> SynonymEntry {
        SynonymEntry {
            term: "car".to_string(),
            synonyms: vec!["automobile".to_string(), "vehicle".to_string()],
            hypernyms: vec!["transport".to_string()],
            hyponyms: vec!["sedan".to_string(), "suv".to_string()],
        }
    }

    fn engine_with_car() -> QueryExpansionEngine {
        let mut e = QueryExpansionEngine::new(ExpansionConfig::default());
        e.add_synonym_entry(car_entry());
        e
    }

    fn count_source(expanded: &QeExpandedQuery, src: &ExpansionSource) -> usize {
        expanded.terms.iter().filter(|t| &t.source == src).count()
    }

    // ── 1. Construction ────────────────────────────────────────────────────────

    #[test]
    fn test_new_engine_starts_empty() {
        let e = QueryExpansionEngine::new(ExpansionConfig::default());
        assert!(e.synonym_db.is_empty());
        assert!(e.context_terms.is_empty());
        assert_eq!(e.expansions_performed, 0);
        assert_eq!(e.total_terms_added, 0);
    }

    #[test]
    fn test_default_config_values() {
        let cfg = ExpansionConfig::default();
        assert_eq!(cfg.max_expansions_per_term, 5);
        assert!((cfg.min_weight - 0.3).abs() < f64::EPSILON);
        assert!(cfg.include_synonyms);
        assert!(cfg.include_hypernyms);
        assert!(cfg.include_hyponyms);
        assert!(cfg.include_related);
        assert!((cfg.boost_exact_match - 2.0).abs() < f64::EPSILON);
    }

    // ── 2. add_synonym_entry ───────────────────────────────────────────────────

    #[test]
    fn test_add_synonym_entry_lowercases_key() {
        let mut e = QueryExpansionEngine::new(ExpansionConfig::default());
        e.add_synonym_entry(SynonymEntry {
            term: "CAR".to_string(),
            synonyms: vec!["auto".to_string()],
            hypernyms: vec![],
            hyponyms: vec![],
        });
        assert!(e.synonym_db.contains_key("car"));
        assert!(!e.synonym_db.contains_key("CAR"));
    }

    #[test]
    fn test_add_synonym_entry_replaces_existing() {
        let mut e = QueryExpansionEngine::new(ExpansionConfig::default());
        e.add_synonym_entry(SynonymEntry {
            term: "car".to_string(),
            synonyms: vec!["auto".to_string()],
            hypernyms: vec![],
            hyponyms: vec![],
        });
        e.add_synonym_entry(SynonymEntry {
            term: "car".to_string(),
            synonyms: vec!["vehicle".to_string()],
            hypernyms: vec![],
            hyponyms: vec![],
        });
        let entry = e.synonym_db.get("car").expect("entry missing");
        assert_eq!(entry.synonyms, vec!["vehicle"]);
    }

    // ── 3. add_context_terms ───────────────────────────────────────────────────

    #[test]
    fn test_add_context_terms_appends() {
        let mut e = QueryExpansionEngine::new(ExpansionConfig::default());
        e.add_context_terms("rust".to_string(), vec!["memory".to_string()]);
        e.add_context_terms("rust".to_string(), vec!["safety".to_string()]);
        let terms = e.context_terms.get("rust").expect("missing");
        assert_eq!(terms.len(), 2);
    }

    #[test]
    fn test_add_context_terms_lowercases_trigger() {
        let mut e = QueryExpansionEngine::new(ExpansionConfig::default());
        e.add_context_terms("RUST".to_string(), vec!["memory".to_string()]);
        assert!(e.context_terms.contains_key("rust"));
    }

    // ── 4. expand_term ────────────────────────────────────────────────────────

    #[test]
    fn test_expand_term_unknown_returns_empty() {
        let e = engine_with_car();
        assert!(e.expand_term("bicycle").is_empty());
    }

    #[test]
    fn test_expand_term_synonyms() {
        let e = engine_with_car();
        let terms = e.expand_term("car");
        let syns: Vec<_> = terms
            .iter()
            .filter(|t| t.source == ExpansionSource::Synonym)
            .collect();
        assert!(!syns.is_empty(), "expected at least one synonym");
        assert!(syns.iter().all(|t| (t.weight - 0.9).abs() < f64::EPSILON));
    }

    #[test]
    fn test_expand_term_hypernyms() {
        let e = engine_with_car();
        let terms = e.expand_term("car");
        let hypers: Vec<_> = terms
            .iter()
            .filter(|t| t.source == ExpansionSource::Hypernym)
            .collect();
        assert!(!hypers.is_empty());
        assert!(hypers.iter().all(|t| (t.weight - 0.7).abs() < f64::EPSILON));
    }

    #[test]
    fn test_expand_term_hyponyms() {
        let e = engine_with_car();
        let terms = e.expand_term("car");
        let hypos: Vec<_> = terms
            .iter()
            .filter(|t| t.source == ExpansionSource::Hyponym)
            .collect();
        assert!(!hypos.is_empty());
        assert!(hypos.iter().all(|t| (t.weight - 0.6).abs() < f64::EPSILON));
    }

    #[test]
    fn test_expand_term_case_insensitive() {
        let e = engine_with_car();
        let lower = e.expand_term("car");
        let upper = e.expand_term("CAR");
        assert_eq!(lower.len(), upper.len());
    }

    #[test]
    fn test_expand_term_respects_max_expansions() {
        let cfg = ExpansionConfig {
            max_expansions_per_term: 2,
            ..Default::default()
        };
        let mut e = QueryExpansionEngine::new(cfg);
        e.add_synonym_entry(car_entry());
        let terms = e.expand_term("car");
        assert!(terms.len() <= 2);
    }

    #[test]
    fn test_expand_term_sorted_desc() {
        let e = engine_with_car();
        let terms = e.expand_term("car");
        for pair in terms.windows(2) {
            assert!(pair[0].weight >= pair[1].weight);
        }
    }

    #[test]
    fn test_expand_term_min_weight_filter() {
        let cfg = ExpansionConfig {
            min_weight: 0.8,
            ..Default::default()
        };
        let mut e = QueryExpansionEngine::new(cfg);
        e.add_synonym_entry(car_entry());
        // Only synonyms (0.9) should survive; hypernyms (0.7) and hyponyms (0.6)
        // and related terms (0.5) should be filtered.
        let terms = e.expand_term("car");
        assert!(terms.iter().all(|t| t.weight >= 0.8));
    }

    #[test]
    fn test_expand_term_context_related() {
        // Use a larger cap so the related term (weight 0.5) is not crowded out
        // by the 5 synonym/hypernym/hyponym terms from car_entry().
        let cfg = ExpansionConfig {
            max_expansions_per_term: 10,
            ..Default::default()
        };
        let mut e = QueryExpansionEngine::new(cfg);
        e.add_synonym_entry(car_entry());
        e.add_context_terms("car".to_string(), vec!["driving".to_string()]);
        let terms = e.expand_term("car");
        let related: Vec<_> = terms
            .iter()
            .filter(|t| t.source == ExpansionSource::RelatedTerm)
            .collect();
        assert!(
            !related.is_empty(),
            "expected related terms, got: {terms:?}"
        );
        assert!(related
            .iter()
            .all(|t| (t.weight - 0.5).abs() < f64::EPSILON));
    }

    #[test]
    fn test_expand_term_include_synonyms_false() {
        let cfg = ExpansionConfig {
            include_synonyms: false,
            ..Default::default()
        };
        let mut e = QueryExpansionEngine::new(cfg);
        e.add_synonym_entry(car_entry());
        let terms = e.expand_term("car");
        assert!(terms.iter().all(|t| t.source != ExpansionSource::Synonym));
    }

    #[test]
    fn test_expand_term_include_hypernyms_false() {
        let cfg = ExpansionConfig {
            include_hypernyms: false,
            ..Default::default()
        };
        let mut e = QueryExpansionEngine::new(cfg);
        e.add_synonym_entry(car_entry());
        let terms = e.expand_term("car");
        assert!(terms.iter().all(|t| t.source != ExpansionSource::Hypernym));
    }

    #[test]
    fn test_expand_term_include_hyponyms_false() {
        let cfg = ExpansionConfig {
            include_hyponyms: false,
            ..Default::default()
        };
        let mut e = QueryExpansionEngine::new(cfg);
        e.add_synonym_entry(car_entry());
        let terms = e.expand_term("car");
        assert!(terms.iter().all(|t| t.source != ExpansionSource::Hyponym));
    }

    #[test]
    fn test_expand_term_include_related_false() {
        let cfg = ExpansionConfig {
            include_related: false,
            ..Default::default()
        };
        let mut e = QueryExpansionEngine::new(cfg);
        e.add_synonym_entry(car_entry());
        e.add_context_terms("car".to_string(), vec!["road".to_string()]);
        let terms = e.expand_term("car");
        assert!(terms
            .iter()
            .all(|t| t.source != ExpansionSource::RelatedTerm));
    }

    // ── 5. expand_query ───────────────────────────────────────────────────────

    #[test]
    fn test_expand_query_empty_string() {
        let mut e = QueryExpansionEngine::new(ExpansionConfig::default());
        let eq = e.expand_query("", None);
        assert_eq!(eq.original, "");
        assert!(eq.terms.is_empty());
    }

    #[test]
    fn test_expand_query_original_preserved() {
        let mut e = engine_with_car();
        let eq = e.expand_query("fast car", None);
        assert_eq!(eq.original, "fast car");
    }

    #[test]
    fn test_expand_query_original_tokens_boosted() {
        let mut e = engine_with_car();
        let eq = e.expand_query("car", None);
        let original_term = eq
            .terms
            .iter()
            .find(|t| t.term == "car" && t.source == ExpansionSource::ContextualExpansion);
        assert!(original_term.is_some());
        let w = original_term.map(|t| t.weight).unwrap_or(0.0);
        assert!((w - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_expand_query_no_duplicates() {
        let mut e = engine_with_car();
        let eq = e.expand_query("car car car", None);
        let terms: Vec<_> = eq.terms.iter().map(|t| &t.term).collect();
        let unique: std::collections::HashSet<_> = terms.iter().collect();
        assert_eq!(terms.len(), unique.len(), "duplicate terms found");
    }

    #[test]
    fn test_expand_query_punctuation_tokenisation() {
        let mut e = engine_with_car();
        // Punctuation separators should produce the same tokens as spaces.
        let with_punct = e.expand_query("fast,car!speed", None);
        let terms_punct: std::collections::HashSet<_> =
            with_punct.terms.iter().map(|t| t.term.clone()).collect();
        assert!(terms_punct.contains("car"));
        assert!(terms_punct.contains("fast"));
        assert!(terms_punct.contains("speed"));
    }

    #[test]
    fn test_expand_query_with_embedding() {
        let mut e = engine_with_car();
        let emb = vec![0.1, 0.2, 0.3];
        let eq = e.expand_query("car", Some(emb.clone()));
        assert_eq!(eq.query_embedding, Some(emb));
    }

    #[test]
    fn test_expand_query_no_embedding() {
        let mut e = engine_with_car();
        let eq = e.expand_query("car", None);
        assert!(eq.query_embedding.is_none());
    }

    #[test]
    fn test_expand_query_terms_sorted_desc() {
        let mut e = engine_with_car();
        let eq = e.expand_query("car", None);
        for pair in eq.terms.windows(2) {
            assert!(
                pair[0].weight >= pair[1].weight,
                "terms not sorted: {} < {}",
                pair[0].weight,
                pair[1].weight
            );
        }
    }

    #[test]
    fn test_expand_query_has_synonyms() {
        let mut e = engine_with_car();
        let eq = e.expand_query("car", None);
        assert!(count_source(&eq, &ExpansionSource::Synonym) > 0);
    }

    #[test]
    fn test_expand_query_has_hypernyms() {
        let mut e = engine_with_car();
        let eq = e.expand_query("car", None);
        assert!(count_source(&eq, &ExpansionSource::Hypernym) > 0);
    }

    #[test]
    fn test_expand_query_has_hyponyms() {
        let mut e = engine_with_car();
        let eq = e.expand_query("car", None);
        assert!(count_source(&eq, &ExpansionSource::Hyponym) > 0);
    }

    #[test]
    fn test_expand_query_counters_increment() {
        let mut e = engine_with_car();
        e.expand_query("car", None);
        e.expand_query("car", None);
        assert_eq!(e.expansions_performed, 2);
        assert!(e.total_terms_added > 0);
    }

    #[test]
    fn test_expand_query_multi_token() {
        let mut e = QueryExpansionEngine::new(ExpansionConfig::default());
        e.add_synonym_entry(SynonymEntry {
            term: "fast".to_string(),
            synonyms: vec!["quick".to_string()],
            hypernyms: vec![],
            hyponyms: vec![],
        });
        e.add_synonym_entry(car_entry());
        let eq = e.expand_query("fast car", None);
        let term_strs: Vec<_> = eq.terms.iter().map(|t| t.term.as_str()).collect();
        assert!(term_strs.contains(&"quick"));
        assert!(term_strs.contains(&"automobile"));
    }

    // ── 6. build_search_string ────────────────────────────────────────────────

    #[test]
    fn test_build_search_string_contains_original() {
        let mut e = engine_with_car();
        let eq = e.expand_query("car", None);
        let s = e.build_search_string(&eq);
        assert!(s.contains("car"), "search string missing original: {s}");
    }

    #[test]
    fn test_build_search_string_no_duplicates() {
        let mut e = engine_with_car();
        let eq = e.expand_query("car", None);
        let s = e.build_search_string(&eq);
        let words: Vec<_> = s.split_whitespace().collect();
        let unique: std::collections::HashSet<_> = words.iter().collect();
        assert_eq!(
            words.len(),
            unique.len(),
            "duplicates in search string: {s}"
        );
    }

    #[test]
    fn test_build_search_string_original_first() {
        let mut e = engine_with_car();
        let eq = e.expand_query("car", None);
        let s = e.build_search_string(&eq);
        // "car" is the original token so it must appear before any expansion.
        let words: Vec<_> = s.split_whitespace().collect();
        let car_pos = words.iter().position(|w| *w == "car");
        assert_eq!(car_pos, Some(0), "original not first in: {s}");
    }

    #[test]
    fn test_build_search_string_respects_min_weight() {
        let cfg = ExpansionConfig {
            min_weight: 0.95,
            boost_exact_match: 2.0,
            ..Default::default()
        };
        let mut e = QueryExpansionEngine::new(cfg);
        e.add_synonym_entry(car_entry());
        let eq = e.expand_query("car", None);
        let s = e.build_search_string(&eq);
        // Only the boosted original ("car", weight=2.0) clears 0.95.
        // synonyms (0.9) are below threshold → not included.
        assert!(
            !s.contains("automobile"),
            "automobile should be filtered: {s}"
        );
    }

    // ── 7. weighted_terms ─────────────────────────────────────────────────────

    #[test]
    fn test_weighted_terms_sorted_desc() {
        let mut e = engine_with_car();
        let eq = e.expand_query("car", None);
        let pairs = e.weighted_terms(&eq);
        for w in pairs.windows(2) {
            assert!(w[0].1 >= w[1].1);
        }
    }

    #[test]
    fn test_weighted_terms_count_matches_terms() {
        let mut e = engine_with_car();
        let eq = e.expand_query("car", None);
        assert_eq!(e.weighted_terms(&eq).len(), eq.terms.len());
    }

    // ── 8. coverage_stats ─────────────────────────────────────────────────────

    #[test]
    fn test_coverage_stats_empty_query() {
        let mut e = QueryExpansionEngine::new(ExpansionConfig::default());
        let eq = e.expand_query("", None);
        let stats = e.coverage_stats(&eq);
        assert_eq!(stats.original_term_count, 0);
        assert_eq!(stats.expanded_term_count, 0);
        assert!((stats.expansion_ratio - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_coverage_stats_ratio() {
        let mut e = engine_with_car();
        let eq = e.expand_query("car", None);
        let stats = e.coverage_stats(&eq);
        assert_eq!(stats.original_term_count, 1);
        assert!(stats.expanded_term_count >= 1);
        assert!((stats.expansion_ratio - stats.expanded_term_count as f64).abs() < f64::EPSILON);
    }

    #[test]
    fn test_coverage_stats_sources_populated() {
        let mut e = engine_with_car();
        let eq = e.expand_query("car", None);
        let stats = e.coverage_stats(&eq);
        assert!(!stats.sources.is_empty());
        assert!(stats.sources.contains_key("ContextualExpansion"));
    }

    #[test]
    fn test_coverage_stats_source_counts_sum_to_total() {
        let mut e = engine_with_car();
        let eq = e.expand_query("car", None);
        let stats = e.coverage_stats(&eq);
        let sum: usize = stats.sources.values().sum();
        assert_eq!(sum, stats.expanded_term_count);
    }

    // ── 9. engine_stats ───────────────────────────────────────────────────────

    #[test]
    fn test_engine_stats_initial_zero() {
        let e = QueryExpansionEngine::new(ExpansionConfig::default());
        assert_eq!(e.engine_stats(), (0, 0));
    }

    #[test]
    fn test_engine_stats_after_expansion() {
        let mut e = engine_with_car();
        e.expand_query("car", None);
        let (performed, added) = e.engine_stats();
        assert_eq!(performed, 1);
        assert!(added > 0);
    }

    #[test]
    fn test_engine_stats_accumulate() {
        let mut e = engine_with_car();
        e.expand_query("car", None);
        e.expand_query("car", None);
        let (performed, _) = e.engine_stats();
        assert_eq!(performed, 2);
    }

    // ── 10. Edge cases ─────────────────────────────────────────────────────────

    #[test]
    fn test_expand_query_all_punctuation() {
        let mut e = QueryExpansionEngine::new(ExpansionConfig::default());
        let eq = e.expand_query("...,,,!!!", None);
        assert!(eq.terms.is_empty());
    }

    #[test]
    fn test_deduplication_keeps_highest_weight() {
        // "automobile" might appear as a synonym with weight 0.9 and also as a
        // separate context term with weight 0.5.  The engine should keep 0.9.
        let mut e = QueryExpansionEngine::new(ExpansionConfig::default());
        e.add_synonym_entry(SynonymEntry {
            term: "car".to_string(),
            synonyms: vec!["automobile".to_string()],
            hypernyms: vec![],
            hyponyms: vec![],
        });
        e.add_context_terms("car".to_string(), vec!["automobile".to_string()]);
        let eq = e.expand_query("car", None);
        let auto_terms: Vec<_> = eq.terms.iter().filter(|t| t.term == "automobile").collect();
        assert_eq!(auto_terms.len(), 1, "duplicate automobile terms");
        assert!((auto_terms[0].weight - 0.9).abs() < f64::EPSILON);
    }

    #[test]
    fn test_expand_term_output_lowercased() {
        let mut e = QueryExpansionEngine::new(ExpansionConfig::default());
        e.add_synonym_entry(SynonymEntry {
            term: "car".to_string(),
            synonyms: vec!["Automobile".to_string()],
            hypernyms: vec![],
            hyponyms: vec![],
        });
        let terms = e.expand_term("car");
        assert!(terms.iter().all(|t| t.term == t.term.to_lowercase()));
    }

    #[test]
    fn test_multiple_entries_independent() {
        let mut e = QueryExpansionEngine::new(ExpansionConfig::default());
        e.add_synonym_entry(SynonymEntry {
            term: "car".to_string(),
            synonyms: vec!["vehicle".to_string()],
            hypernyms: vec![],
            hyponyms: vec![],
        });
        e.add_synonym_entry(SynonymEntry {
            term: "dog".to_string(),
            synonyms: vec!["canine".to_string()],
            hypernyms: vec![],
            hyponyms: vec![],
        });
        let car_terms = e.expand_term("car");
        let dog_terms = e.expand_term("dog");
        assert!(car_terms.iter().any(|t| t.term == "vehicle"));
        assert!(dog_terms.iter().any(|t| t.term == "canine"));
        // No cross-contamination
        assert!(!car_terms.iter().any(|t| t.term == "canine"));
        assert!(!dog_terms.iter().any(|t| t.term == "vehicle"));
    }

    #[test]
    fn test_expansion_stats_multi_token() {
        let mut e = QueryExpansionEngine::new(ExpansionConfig::default());
        e.add_synonym_entry(SynonymEntry {
            term: "fast".to_string(),
            synonyms: vec!["quick".to_string()],
            hypernyms: vec![],
            hyponyms: vec![],
        });
        e.add_synonym_entry(car_entry());
        let eq = e.expand_query("fast car", None);
        let stats = e.coverage_stats(&eq);
        assert_eq!(stats.original_term_count, 2, "two original tokens");
        assert!(stats.expanded_term_count > 2);
        // sources should have "Synonym" from both entries
        assert!(stats.sources.get("Synonym").copied().unwrap_or(0) >= 3);
    }

    #[test]
    fn test_expand_query_colon_semicolon_separators() {
        let mut e = engine_with_car();
        let eq = e.expand_query("keyword:car;fast", None);
        let term_set: std::collections::HashSet<_> =
            eq.terms.iter().map(|t| t.term.as_str()).collect();
        assert!(term_set.contains("keyword"));
        assert!(term_set.contains("car"));
        assert!(term_set.contains("fast"));
    }

    #[test]
    fn test_weighted_terms_references_are_valid() {
        let mut e = engine_with_car();
        let eq = e.expand_query("car", None);
        let pairs = e.weighted_terms(&eq);
        // All term references must be non-empty strings.
        for (term, weight) in &pairs {
            assert!(!term.is_empty());
            assert!(*weight > 0.0);
        }
    }

    #[test]
    fn test_large_synonym_list_capped() {
        let cfg = ExpansionConfig {
            max_expansions_per_term: 3,
            ..Default::default()
        };
        let mut e = QueryExpansionEngine::new(cfg);
        e.add_synonym_entry(SynonymEntry {
            term: "thing".to_string(),
            synonyms: vec![
                "a".to_string(),
                "b".to_string(),
                "c".to_string(),
                "d".to_string(),
                "e".to_string(),
            ],
            hypernyms: vec!["object".to_string()],
            hyponyms: vec!["widget".to_string()],
        });
        let terms = e.expand_term("thing");
        assert_eq!(terms.len(), 3);
    }

    #[test]
    fn test_expansion_source_as_str_variants() {
        // Ensure every variant produces a non-empty string, avoiding future regressions.
        let sources = [
            ExpansionSource::Synonym,
            ExpansionSource::Hypernym,
            ExpansionSource::Hyponym,
            ExpansionSource::RelatedTerm,
            ExpansionSource::ContextualExpansion,
        ];
        let names: Vec<_> = sources.iter().map(|s| s.as_str()).collect();
        let unique: std::collections::HashSet<_> = names.iter().collect();
        assert_eq!(names.len(), unique.len(), "duplicate source names");
        assert!(names.iter().all(|n| !n.is_empty()));
    }

    #[test]
    fn test_build_search_string_empty_query() {
        let mut e = QueryExpansionEngine::new(ExpansionConfig::default());
        let eq = e.expand_query("", None);
        let s = e.build_search_string(&eq);
        assert!(s.is_empty(), "expected empty string, got: {s}");
    }

    #[test]
    fn test_coverage_stats_sources_no_extra_keys() {
        let valid_keys = [
            "Synonym",
            "Hypernym",
            "Hyponym",
            "RelatedTerm",
            "ContextualExpansion",
        ];
        let mut e = engine_with_car();
        let eq = e.expand_query("car", None);
        let stats = e.coverage_stats(&eq);
        for key in stats.sources.keys() {
            assert!(
                valid_keys.contains(&key.as_str()),
                "unexpected source key: {key}"
            );
        }
    }

    #[test]
    fn test_engine_stats_type_returns_tuple() {
        let e = QueryExpansionEngine::new(ExpansionConfig::default());
        let (a, b): (u64, u64) = e.engine_stats();
        // Just verifying the return type destructures correctly.
        let _ = a + b;
    }

    // Verify that HashMap used internally is std
    #[test]
    fn test_expansion_stats_sources_is_hashmap() {
        let stats = super::ExpansionStats {
            original_term_count: 1,
            expanded_term_count: 3,
            expansion_ratio: 3.0,
            sources: HashMap::from([("Synonym".to_string(), 2_usize)]),
        };
        assert_eq!(*stats.sources.get("Synonym").unwrap_or(&0), 2);
    }
}
