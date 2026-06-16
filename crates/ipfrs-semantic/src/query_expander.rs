//! # Semantic Query Expander
//!
//! Expands semantic search queries by generating related query variants —
//! synonyms, paraphrases, sub-queries — to improve recall without sacrificing precision.
//!
//! ## Overview
//!
//! The [`SemanticQueryExpander`] accepts a query string and an [`ExpansionStrategy`],
//! then produces an [`ExpandedQuery`] containing the original query plus generated variants.
//! Variants are drawn from a [`TermEntry`] registry that encodes lexical relations
//! ([`TermRelation`]) between terms, each annotated with a relevance weight.
//!
//! ## Example
//!
//! ```rust
//! use ipfrs_semantic::query_expander::{
//!     ExpansionStrategy, SemanticQueryExpander, TermEntry, TermRelation,
//! };
//!
//! let mut expander = SemanticQueryExpander::new(5);
//! expander.register_term(TermEntry {
//!     term: "car".to_string(),
//!     related_term: "automobile".to_string(),
//!     relation: TermRelation::Synonym,
//!     weight: 0.9,
//! });
//!
//! let expanded = expander.expand("car", ExpansionStrategy::Synonyms);
//! assert_eq!(expanded.original, "car");
//! assert!(!expanded.expansions.is_empty());
//! ```

/// Expansion strategy controlling how query variants are generated.
#[derive(Clone, Debug, PartialEq)]
pub enum ExpansionStrategy {
    /// Replace keywords with synonyms from the registry.
    Synonyms,
    /// Add specificity terms (hyponyms) to the query.
    Narrowing,
    /// Remove specificity terms, generalising via hypernyms.
    Broadening,
    /// Add "NOT" exclusion terms derived from antonyms.
    Negation,
    /// Combine [`ExpansionStrategy::Synonyms`] and [`ExpansionStrategy::Narrowing`], then dedup.
    Combination,
}

/// Lexical / semantic relation between two terms.
#[derive(Clone, Debug, PartialEq)]
pub enum TermRelation {
    /// The related term is a synonym of the source term.
    Synonym,
    /// The related term is a broader concept (e.g. "vehicle" for "car").
    Hypernym,
    /// The related term is a narrower concept (e.g. "sedan" for "car").
    Hyponym,
    /// The related term is an antonym of the source term.
    Antonym,
}

/// A single entry in the term-relation registry.
#[derive(Clone, Debug, PartialEq)]
pub struct TermEntry {
    /// The source term.
    pub term: String,
    /// The term related to [`TermEntry::term`] via [`TermEntry::relation`].
    pub related_term: String,
    /// The type of relation between the two terms.
    pub relation: TermRelation,
    /// Relevance weight in the range \[0.0, 1.0\].
    pub weight: f32,
}

/// Result of a query expansion operation.
#[derive(Clone, Debug, PartialEq)]
pub struct ExpandedQuery {
    /// The original query string, unchanged.
    pub original: String,
    /// Generated query variants (excluding the original).
    pub expansions: Vec<String>,
    /// The strategy used to produce the expansions.
    pub strategy: ExpansionStrategy,
}

impl ExpandedQuery {
    /// Total number of query variants including the original.
    ///
    /// Always ≥ 1 because the original is always counted.
    #[must_use]
    pub fn total_variants(&self) -> usize {
        self.expansions.len() + 1
    }
}

/// Runtime statistics for a [`SemanticQueryExpander`].
#[derive(Clone, Debug, PartialEq)]
pub struct ExpanderStats {
    /// Total number of `expand()` calls made so far.
    pub total_expansions: u64,
    /// Average number of variant strings generated per `expand()` call.
    ///
    /// Returns `0.0` when `total_expansions == 0`.
    pub avg_variants_per_query: f64,
    /// Number of entries currently in the term registry.
    pub registry_size: usize,
}

/// Expands semantic search queries using a registry of term relations.
///
/// The expander maintains a list of [`TermEntry`] records that encode relations
/// (synonym, hypernym, hyponym, antonym) between pairs of terms.  Given a raw
/// query string and an [`ExpansionStrategy`], [`SemanticQueryExpander::expand`]
/// produces an [`ExpandedQuery`] with up to `max_expansions` variants.
pub struct SemanticQueryExpander {
    /// All known term-relation entries.
    pub registry: Vec<TermEntry>,
    /// Maximum number of expansion variants returned per call (default 5).
    pub max_expansions: usize,
    /// Cumulative count of successful `expand()` calls.
    pub total_expansions: u64,
    /// Cumulative count of expansion *variants* generated (not counting the original).
    pub total_variants_generated: u64,
}

impl SemanticQueryExpander {
    /// Create a new expander with an empty registry.
    ///
    /// # Arguments
    ///
    /// * `max_expansions` – maximum number of variant strings returned by each
    ///   [`SemanticQueryExpander::expand`] call.  Must be ≥ 1; a value of 0
    ///   means no variants are ever produced.
    #[must_use]
    pub fn new(max_expansions: usize) -> Self {
        Self {
            registry: Vec::new(),
            max_expansions,
            total_expansions: 0,
            total_variants_generated: 0,
        }
    }

    /// Append a new [`TermEntry`] to the registry.
    pub fn register_term(&mut self, entry: TermEntry) {
        self.registry.push(entry);
    }

    /// Expand `query` according to `strategy`, returning an [`ExpandedQuery`].
    ///
    /// Internally, the method:
    /// 1. Looks up matching entries in the registry (case-insensitive term match).
    /// 2. Formats candidate strings according to the strategy.
    /// 3. Sorts candidates by weight (descending), then deduplicates.
    /// 4. Caps the result at [`SemanticQueryExpander::max_expansions`].
    /// 5. Updates cumulative statistics.
    pub fn expand(&mut self, query: &str, strategy: ExpansionStrategy) -> ExpandedQuery {
        let expansions = match &strategy {
            ExpansionStrategy::Synonyms => self.expand_synonyms(query),
            ExpansionStrategy::Narrowing => self.expand_narrowing(query),
            ExpansionStrategy::Broadening => self.expand_broadening(query),
            ExpansionStrategy::Negation => self.expand_negation(query),
            ExpansionStrategy::Combination => self.expand_combination(query),
        };

        self.total_expansions += 1;
        self.total_variants_generated += expansions.len() as u64;

        ExpandedQuery {
            original: query.to_string(),
            expansions,
            strategy,
        }
    }

    /// Return a snapshot of the current runtime statistics.
    #[must_use]
    pub fn stats(&self) -> ExpanderStats {
        let avg_variants_per_query = if self.total_expansions == 0 {
            0.0
        } else {
            self.total_variants_generated as f64 / self.total_expansions as f64
        };

        ExpanderStats {
            total_expansions: self.total_expansions,
            avg_variants_per_query,
            registry_size: self.registry.len(),
        }
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Collect registry entries whose `term` matches `query` (case-insensitive)
    /// and whose `relation` equals `target_relation`, sorted by weight descending.
    fn matching_entries<'a>(
        &'a self,
        query: &str,
        target_relation: &TermRelation,
    ) -> Vec<&'a TermEntry> {
        let query_lower = query.to_lowercase();
        let mut entries: Vec<&TermEntry> = self
            .registry
            .iter()
            .filter(|e| e.term.to_lowercase() == query_lower && &e.relation == target_relation)
            .collect();

        // Sort descending by weight; NaN weights sort to the end.
        entries.sort_by(|a, b| {
            b.weight
                .partial_cmp(&a.weight)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        entries
    }

    /// Format candidates and cap at `max_expansions`, deduplicating.
    fn cap_and_dedup(&self, candidates: Vec<String>) -> Vec<String> {
        let mut seen = std::collections::HashSet::new();
        candidates
            .into_iter()
            .filter(|s| seen.insert(s.clone()))
            .take(self.max_expansions)
            .collect()
    }

    fn expand_synonyms(&self, query: &str) -> Vec<String> {
        let entries = self.matching_entries(query, &TermRelation::Synonym);
        let candidates: Vec<String> = entries
            .into_iter()
            .map(|e| format!("{} \u{2192} {}", query, e.related_term))
            .collect();
        self.cap_and_dedup(candidates)
    }

    fn expand_narrowing(&self, query: &str) -> Vec<String> {
        let entries = self.matching_entries(query, &TermRelation::Hyponym);
        let candidates: Vec<String> = entries
            .into_iter()
            .map(|e| format!("{} {}", query, e.related_term))
            .collect();
        self.cap_and_dedup(candidates)
    }

    fn expand_broadening(&self, query: &str) -> Vec<String> {
        let entries = self.matching_entries(query, &TermRelation::Hypernym);
        let candidates: Vec<String> = entries
            .into_iter()
            .map(|e| e.related_term.clone())
            .collect();
        self.cap_and_dedup(candidates)
    }

    fn expand_negation(&self, query: &str) -> Vec<String> {
        let entries = self.matching_entries(query, &TermRelation::Antonym);
        let candidates: Vec<String> = entries
            .into_iter()
            .map(|e| format!("{} NOT {}", query, e.related_term))
            .collect();
        self.cap_and_dedup(candidates)
    }

    fn expand_combination(&self, query: &str) -> Vec<String> {
        // Run Synonyms and Narrowing internally without touching stats.
        let syn = self.expand_synonyms(query);
        let narrow = self.expand_narrowing(query);

        // Merge in order: synonyms first, then narrowing, then dedup + cap.
        let merged: Vec<String> = syn.into_iter().chain(narrow).collect();
        self.cap_and_dedup(merged)
    }
}

// ===========================================================================
// Vector-based Semantic Query Expander
// ===========================================================================

use std::collections::HashMap;

/// Configuration for the [`VectorQueryExpander`].
#[derive(Debug, Clone)]
pub struct VectorExpanderConfig {
    /// Maximum number of expansion terms per query (default 5).
    pub max_expansions: usize,
    /// Minimum cosine similarity for a synonym to be included (default 0.7).
    pub similarity_threshold: f64,
    /// Multiplicative decay factor applied per rank position (default 0.8).
    pub weight_decay: f64,
}

impl Default for VectorExpanderConfig {
    fn default() -> Self {
        Self {
            max_expansions: 5,
            similarity_threshold: 0.7,
            weight_decay: 0.8,
        }
    }
}

/// Result of a vector-based query expansion.
#[derive(Debug, Clone)]
pub struct VectorExpandedQuery {
    /// The original query vector, unchanged.
    pub original: Vec<f64>,
    /// Individual expansion terms with their vectors and weights.
    pub expansions: Vec<VectorQueryExpansion>,
    /// Weighted combination of original + expansions, L2-normalised.
    pub combined: Vec<f64>,
}

/// A single expansion term produced by [`VectorQueryExpander`].
#[derive(Debug, Clone)]
pub struct VectorQueryExpansion {
    /// The synonym term string.
    pub term: String,
    /// The embedding vector for this term.
    pub vector: Vec<f64>,
    /// The computed weight (similarity × decay^rank).
    pub weight: f64,
    /// Cosine similarity between this term's vector and the original query vector.
    pub similarity_to_original: f64,
}

/// Runtime statistics for a [`VectorQueryExpander`].
#[derive(Debug, Clone)]
pub struct VectorExpanderStats {
    /// Number of distinct terms that have at least one synonym registered.
    pub total_terms: usize,
    /// Total number of synonym entries across all terms.
    pub total_synonyms: usize,
    /// Cumulative count of `expand_query` calls.
    pub expansions_performed: u64,
}

/// Vector-based semantic query expander.
///
/// Maintains a synonym registry where each term maps to a list of
/// `(synonym_string, embedding_vector)` pairs.  Given a query term and its
/// embedding, the expander finds synonyms whose cosine similarity exceeds
/// [`VectorExpanderConfig::similarity_threshold`], weights them by
/// `similarity × decay^rank`, and returns a combined (L2-normalised) vector.
pub struct VectorQueryExpander {
    config: VectorExpanderConfig,
    synonym_map: HashMap<String, Vec<(String, Vec<f64>)>>,
    expansions_performed: u64,
}

impl VectorQueryExpander {
    /// Create a new expander with the given configuration and an empty synonym
    /// registry.
    #[must_use]
    pub fn new(config: VectorExpanderConfig) -> Self {
        Self {
            config,
            synonym_map: HashMap::new(),
            expansions_performed: 0,
        }
    }

    /// Register a synonym for `term` together with its embedding vector.
    pub fn add_synonym(&mut self, term: &str, synonym: &str, vector: Vec<f64>) {
        self.synonym_map
            .entry(term.to_string())
            .or_default()
            .push((synonym.to_string(), vector));
    }

    /// Expand a query by finding matching synonyms, weighting them, and
    /// producing a combined vector.
    ///
    /// Steps:
    /// 1. Look up synonyms for `query_term`.
    /// 2. Compute cosine similarity of each synonym vector to `query_vector`.
    /// 3. Filter by `similarity_threshold`.
    /// 4. Sort descending by similarity.
    /// 5. Take at most `max_expansions`.
    /// 6. Weight each expansion by `similarity × decay^rank` (rank starts at 0).
    /// 7. Combine vectors via weighted average and L2-normalise.
    pub fn expand_query(&mut self, query_term: &str, query_vector: &[f64]) -> VectorExpandedQuery {
        self.expansions_performed += 1;

        let synonyms = match self.synonym_map.get(query_term) {
            Some(s) => s,
            None => {
                let combined = Self::l2_normalize(query_vector);
                return VectorExpandedQuery {
                    original: query_vector.to_vec(),
                    expansions: Vec::new(),
                    combined,
                };
            }
        };

        // Compute similarities and filter.
        let mut candidates: Vec<(String, Vec<f64>, f64)> = synonyms
            .iter()
            .filter_map(|(name, vec)| {
                let sim = Self::cosine_similarity(query_vector, vec);
                if sim >= self.config.similarity_threshold {
                    Some((name.clone(), vec.clone(), sim))
                } else {
                    None
                }
            })
            .collect();

        // Sort descending by similarity.
        candidates.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

        // Cap at max_expansions.
        candidates.truncate(self.config.max_expansions);

        // Build expansion entries with decay-weighted weights.
        let expansions: Vec<VectorQueryExpansion> = candidates
            .iter()
            .enumerate()
            .map(|(rank, (term, vec, sim))| {
                let weight = sim * self.config.weight_decay.powi(rank as i32);
                VectorQueryExpansion {
                    term: term.clone(),
                    vector: vec.clone(),
                    weight,
                    similarity_to_original: *sim,
                }
            })
            .collect();

        // Build weighted-vector pairs for combine_vectors.
        let weighted: Vec<(Vec<f64>, f64)> = expansions
            .iter()
            .map(|e| (e.vector.clone(), e.weight))
            .collect();

        let combined = Self::combine_vectors(query_vector, &weighted);

        VectorExpandedQuery {
            original: query_vector.to_vec(),
            expansions,
            combined,
        }
    }

    /// Weighted average of `original` and `expansions`, then L2-normalise.
    ///
    /// Formula: `(original + Σ(weight_i × expansion_i)) / (1 + Σ weight_i)`
    #[must_use]
    pub fn combine_vectors(original: &[f64], expansions: &[(Vec<f64>, f64)]) -> Vec<f64> {
        let dim = original.len();
        let mut result = original.to_vec();
        let mut total_weight: f64 = 1.0;

        for (vec, w) in expansions {
            let len = vec.len().min(dim);
            for i in 0..len {
                result[i] += w * vec[i];
            }
            total_weight += w;
        }

        if total_weight.abs() > f64::EPSILON {
            for v in &mut result {
                *v /= total_weight;
            }
        }

        Self::l2_normalize(&result)
    }

    /// Cosine similarity between two vectors.
    ///
    /// Returns 0.0 when either vector has zero magnitude.
    #[must_use]
    pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
        let len = a.len().min(b.len());
        let mut dot = 0.0_f64;
        let mut norm_a = 0.0_f64;
        let mut norm_b = 0.0_f64;
        for i in 0..len {
            dot += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }
        let denom = norm_a.sqrt() * norm_b.sqrt();
        if denom < f64::EPSILON {
            0.0
        } else {
            dot / denom
        }
    }

    /// Remove a specific synonym for `term`. Returns `true` if found and removed.
    pub fn remove_synonym(&mut self, term: &str, synonym: &str) -> bool {
        if let Some(synonyms) = self.synonym_map.get_mut(term) {
            let before = synonyms.len();
            synonyms.retain(|(s, _)| s != synonym);
            let removed = synonyms.len() < before;
            // Clean up empty entries.
            if synonyms.is_empty() {
                self.synonym_map.remove(term);
            }
            removed
        } else {
            false
        }
    }

    /// Number of synonyms registered for `term` (0 if unknown).
    #[must_use]
    pub fn synonym_count(&self, term: &str) -> usize {
        self.synonym_map.get(term).map_or(0, |v| v.len())
    }

    /// Remove all synonym entries.
    pub fn clear_synonyms(&mut self) {
        self.synonym_map.clear();
    }

    /// Return a snapshot of the current runtime statistics.
    #[must_use]
    pub fn stats(&self) -> VectorExpanderStats {
        let total_synonyms: usize = self.synonym_map.values().map(|v| v.len()).sum();
        VectorExpanderStats {
            total_terms: self.synonym_map.len(),
            total_synonyms,
            expansions_performed: self.expansions_performed,
        }
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// L2-normalise a vector.  Returns a zero vector when the norm is ≈ 0.
    fn l2_normalize(v: &[f64]) -> Vec<f64> {
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < f64::EPSILON {
            vec![0.0; v.len()]
        } else {
            v.iter().map(|x| x / norm).collect()
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn make_entry(term: &str, related: &str, relation: TermRelation, weight: f32) -> TermEntry {
        TermEntry {
            term: term.to_string(),
            related_term: related.to_string(),
            relation,
            weight,
        }
    }

    fn expander_with_car_data(max: usize) -> SemanticQueryExpander {
        let mut e = SemanticQueryExpander::new(max);
        // Synonyms for "car"
        e.register_term(make_entry("car", "automobile", TermRelation::Synonym, 0.9));
        e.register_term(make_entry("car", "vehicle", TermRelation::Synonym, 0.7));
        e.register_term(make_entry("car", "motorcar", TermRelation::Synonym, 0.5));
        // Hyponyms (narrowing)
        e.register_term(make_entry("car", "sedan", TermRelation::Hyponym, 0.85));
        e.register_term(make_entry("car", "coupe", TermRelation::Hyponym, 0.80));
        e.register_term(make_entry("car", "suv", TermRelation::Hyponym, 0.75));
        // Hypernyms (broadening)
        e.register_term(make_entry("car", "transport", TermRelation::Hypernym, 0.8));
        e.register_term(make_entry("car", "machine", TermRelation::Hypernym, 0.6));
        // Antonyms (negation)
        e.register_term(make_entry("car", "bicycle", TermRelation::Antonym, 0.7));
        e.register_term(make_entry("car", "train", TermRelation::Antonym, 0.65));
        e
    }

    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    #[test]
    fn test_new_starts_empty_registry() {
        let e = SemanticQueryExpander::new(5);
        assert!(e.registry.is_empty());
    }

    #[test]
    fn test_new_sets_max_expansions() {
        let e = SemanticQueryExpander::new(10);
        assert_eq!(e.max_expansions, 10);
    }

    #[test]
    fn test_new_counters_zero() {
        let e = SemanticQueryExpander::new(5);
        assert_eq!(e.total_expansions, 0);
        assert_eq!(e.total_variants_generated, 0);
    }

    // -----------------------------------------------------------------------
    // Registry
    // -----------------------------------------------------------------------

    #[test]
    fn test_register_term_adds_entry() {
        let mut e = SemanticQueryExpander::new(5);
        e.register_term(make_entry("dog", "canine", TermRelation::Synonym, 0.9));
        assert_eq!(e.registry.len(), 1);
    }

    #[test]
    fn test_register_multiple_terms() {
        let mut e = SemanticQueryExpander::new(5);
        e.register_term(make_entry("dog", "canine", TermRelation::Synonym, 0.9));
        e.register_term(make_entry("cat", "feline", TermRelation::Synonym, 0.9));
        assert_eq!(e.registry.len(), 2);
    }

    // -----------------------------------------------------------------------
    // Synonyms
    // -----------------------------------------------------------------------

    #[test]
    fn test_expand_synonyms_finds_related_terms() {
        let mut e = expander_with_car_data(5);
        let result = e.expand("car", ExpansionStrategy::Synonyms);
        assert!(!result.expansions.is_empty());
        // All expansions should contain the arrow separator
        for exp in &result.expansions {
            assert!(exp.contains('\u{2192}'), "expansion missing arrow: {exp}");
        }
    }

    #[test]
    fn test_expand_synonyms_respects_max_cap() {
        // Only allow 2 expansions
        let mut e = expander_with_car_data(2);
        let result = e.expand("car", ExpansionStrategy::Synonyms);
        assert!(
            result.expansions.len() <= 2,
            "got {} expansions, expected ≤ 2",
            result.expansions.len()
        );
    }

    #[test]
    fn test_expand_synonyms_sorts_by_weight_desc() {
        let mut e = expander_with_car_data(5);
        let result = e.expand("car", ExpansionStrategy::Synonyms);
        // First expansion should correspond to the highest-weight synonym (automobile, 0.9)
        assert!(
            result.expansions[0].contains("automobile"),
            "first synonym should be 'automobile' (weight 0.9), got: {}",
            result.expansions[0]
        );
    }

    // -----------------------------------------------------------------------
    // Narrowing
    // -----------------------------------------------------------------------

    #[test]
    fn test_expand_narrowing_adds_hyponyms() {
        let mut e = expander_with_car_data(5);
        let result = e.expand("car", ExpansionStrategy::Narrowing);
        assert!(!result.expansions.is_empty());
        // Format is "{query} {related_term}"
        for exp in &result.expansions {
            assert!(
                exp.starts_with("car "),
                "narrowing expansion should start with 'car ': {exp}"
            );
        }
    }

    #[test]
    fn test_expand_narrowing_sorts_by_weight() {
        let mut e = expander_with_car_data(5);
        let result = e.expand("car", ExpansionStrategy::Narrowing);
        // sedan has weight 0.85 — highest hyponym
        assert!(
            result.expansions[0].contains("sedan"),
            "first narrowing expansion should be 'car sedan', got: {}",
            result.expansions[0]
        );
    }

    // -----------------------------------------------------------------------
    // Broadening
    // -----------------------------------------------------------------------

    #[test]
    fn test_expand_broadening_uses_hypernyms() {
        let mut e = expander_with_car_data(5);
        let result = e.expand("car", ExpansionStrategy::Broadening);
        assert!(!result.expansions.is_empty());
        // Broadening returns the hypernym alone (no query prefix)
        assert!(
            result.expansions.iter().any(|s| s == "transport"),
            "expected 'transport' in broadening expansions: {:?}",
            result.expansions
        );
    }

    #[test]
    fn test_expand_broadening_no_query_prefix() {
        let mut e = expander_with_car_data(5);
        let result = e.expand("car", ExpansionStrategy::Broadening);
        for exp in &result.expansions {
            assert!(
                !exp.starts_with("car"),
                "broadening expansion must not start with query: {exp}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Negation
    // -----------------------------------------------------------------------

    #[test]
    fn test_expand_negation_formats_not_correctly() {
        let mut e = expander_with_car_data(5);
        let result = e.expand("car", ExpansionStrategy::Negation);
        assert!(!result.expansions.is_empty());
        for exp in &result.expansions {
            assert!(
                exp.contains(" NOT "),
                "negation expansion missing 'NOT': {exp}"
            );
        }
    }

    #[test]
    fn test_expand_negation_contains_antonym() {
        let mut e = expander_with_car_data(5);
        let result = e.expand("car", ExpansionStrategy::Negation);
        assert!(
            result.expansions.iter().any(|s| s.contains("bicycle")),
            "expected 'bicycle' antonym in negation: {:?}",
            result.expansions
        );
    }

    // -----------------------------------------------------------------------
    // Combination
    // -----------------------------------------------------------------------

    #[test]
    fn test_expand_combination_merges_synonyms_and_narrowing() {
        let mut e = expander_with_car_data(10);
        let result = e.expand("car", ExpansionStrategy::Combination);
        // Should contain both a synonym-style expansion and a narrowing-style expansion
        let has_arrow = result.expansions.iter().any(|s| s.contains('\u{2192}'));
        let has_space = result
            .expansions
            .iter()
            .any(|s| s.starts_with("car ") && !s.contains('\u{2192}'));
        assert!(has_arrow, "combination should include synonym expansions");
        assert!(has_space, "combination should include narrowing expansions");
    }

    #[test]
    fn test_expand_combination_deduplicates() {
        let mut e = SemanticQueryExpander::new(10);
        // Register the same related term via two strategies
        e.register_term(make_entry("dog", "canine", TermRelation::Synonym, 0.9));
        e.register_term(make_entry("dog", "canine", TermRelation::Synonym, 0.8));
        let result = e.expand("dog", ExpansionStrategy::Combination);
        // Both produce identical strings → should be deduped
        let count = result
            .expansions
            .iter()
            .filter(|s| s.contains("canine"))
            .count();
        assert_eq!(count, 1, "duplicate expansions not removed");
    }

    #[test]
    fn test_expand_combination_caps_at_max_expansions() {
        let mut e = expander_with_car_data(3);
        let result = e.expand("car", ExpansionStrategy::Combination);
        assert!(
            result.expansions.len() <= 3,
            "combination exceeded max_expansions cap"
        );
    }

    // -----------------------------------------------------------------------
    // total_variants
    // -----------------------------------------------------------------------

    #[test]
    fn test_total_variants_includes_original() {
        let mut e = expander_with_car_data(5);
        let result = e.expand("car", ExpansionStrategy::Synonyms);
        assert_eq!(
            result.total_variants(),
            result.expansions.len() + 1,
            "total_variants should be expansions.len() + 1"
        );
    }

    #[test]
    fn test_total_variants_minimum_one() {
        let mut e = SemanticQueryExpander::new(5);
        let result = e.expand("unknown_term", ExpansionStrategy::Synonyms);
        assert_eq!(result.total_variants(), 1);
    }

    // -----------------------------------------------------------------------
    // Unknown / empty queries
    // -----------------------------------------------------------------------

    #[test]
    fn test_unknown_query_returns_empty_expansions() {
        let mut e = expander_with_car_data(5);
        let result = e.expand("zzzunknown", ExpansionStrategy::Synonyms);
        assert!(
            result.expansions.is_empty(),
            "unknown term should produce no expansions"
        );
    }

    // -----------------------------------------------------------------------
    // Case-insensitive matching
    // -----------------------------------------------------------------------

    #[test]
    fn test_case_insensitive_matching() {
        let mut e = expander_with_car_data(5);
        let lower = e.expand("car", ExpansionStrategy::Synonyms);
        let upper = e.expand("CAR", ExpansionStrategy::Synonyms);
        assert_eq!(
            lower.expansions.len(),
            upper.expansions.len(),
            "case should not affect number of matches"
        );
    }

    // -----------------------------------------------------------------------
    // Stats
    // -----------------------------------------------------------------------

    #[test]
    fn test_stats_total_expansions_increments() {
        let mut e = expander_with_car_data(5);
        e.expand("car", ExpansionStrategy::Synonyms);
        e.expand("car", ExpansionStrategy::Narrowing);
        assert_eq!(e.stats().total_expansions, 2);
    }

    #[test]
    fn test_stats_avg_variants_zero_when_no_expansions() {
        let e = SemanticQueryExpander::new(5);
        assert_eq!(e.stats().avg_variants_per_query, 0.0);
    }

    #[test]
    fn test_stats_avg_variants_computed() {
        let mut e = expander_with_car_data(5);
        let r1 = e.expand("car", ExpansionStrategy::Synonyms);
        let r2 = e.expand("car", ExpansionStrategy::Narrowing);
        let expected = (r1.expansions.len() + r2.expansions.len()) as f64 / 2.0;
        let stats = e.stats();
        assert!(
            (stats.avg_variants_per_query - expected).abs() < 1e-9,
            "avg mismatch: got {}, expected {expected}",
            stats.avg_variants_per_query
        );
    }

    #[test]
    fn test_stats_registry_size_correct() {
        let e = expander_with_car_data(5);
        // expander_with_car_data registers 10 entries
        assert_eq!(e.stats().registry_size, 10);
    }

    #[test]
    fn test_multiple_calls_accumulate_stats() {
        let mut e = expander_with_car_data(5);
        for _ in 0..5 {
            e.expand("car", ExpansionStrategy::Synonyms);
        }
        assert_eq!(e.stats().total_expansions, 5);
    }

    // -----------------------------------------------------------------------
    // Weight ordering
    // -----------------------------------------------------------------------

    #[test]
    fn test_weight_ordering_respected() {
        let mut e = SemanticQueryExpander::new(5);
        e.register_term(make_entry("fruit", "berry", TermRelation::Hyponym, 0.4));
        e.register_term(make_entry("fruit", "apple", TermRelation::Hyponym, 0.95));
        e.register_term(make_entry("fruit", "mango", TermRelation::Hyponym, 0.7));
        let result = e.expand("fruit", ExpansionStrategy::Narrowing);
        assert!(
            result.expansions[0].contains("apple"),
            "highest-weight hyponym 'apple' should be first: {:?}",
            result.expansions
        );
    }

    // -----------------------------------------------------------------------
    // Strategy / original stored in ExpandedQuery
    // -----------------------------------------------------------------------

    #[test]
    fn test_strategy_stored_in_expanded_query() {
        let mut e = expander_with_car_data(5);
        let result = e.expand("car", ExpansionStrategy::Broadening);
        assert_eq!(result.strategy, ExpansionStrategy::Broadening);
    }

    #[test]
    fn test_original_stored_in_expanded_query() {
        let mut e = expander_with_car_data(5);
        let result = e.expand("car", ExpansionStrategy::Synonyms);
        assert_eq!(result.original, "car");
    }

    // =======================================================================
    // VectorQueryExpander tests
    // =======================================================================

    fn default_vec_config() -> VectorExpanderConfig {
        VectorExpanderConfig::default()
    }

    /// Helper: create a unit vector along dimension `dim` of length `len`.
    fn unit_vec(dim: usize, len: usize) -> Vec<f64> {
        let mut v = vec![0.0; len];
        if dim < len {
            v[dim] = 1.0;
        }
        v
    }

    /// Helper: create a vector with given values (already a simple wrapper).
    fn vec_of(vals: &[f64]) -> Vec<f64> {
        vals.to_vec()
    }

    // -- cosine_similarity --------------------------------------------------

    #[test]
    fn test_vec_cosine_parallel_vectors() {
        let a = vec_of(&[1.0, 0.0, 0.0]);
        let b = vec_of(&[2.0, 0.0, 0.0]);
        let sim = VectorQueryExpander::cosine_similarity(&a, &b);
        assert!(
            (sim - 1.0).abs() < 1e-9,
            "parallel vectors should have similarity 1.0, got {sim}"
        );
    }

    #[test]
    fn test_vec_cosine_orthogonal_vectors() {
        let a = unit_vec(0, 3);
        let b = unit_vec(1, 3);
        let sim = VectorQueryExpander::cosine_similarity(&a, &b);
        assert!(
            sim.abs() < 1e-9,
            "orthogonal vectors should have similarity 0.0, got {sim}"
        );
    }

    #[test]
    fn test_vec_cosine_antiparallel() {
        let a = vec_of(&[1.0, 0.0]);
        let b = vec_of(&[-1.0, 0.0]);
        let sim = VectorQueryExpander::cosine_similarity(&a, &b);
        assert!(
            (sim + 1.0).abs() < 1e-9,
            "anti-parallel vectors should have similarity -1.0, got {sim}"
        );
    }

    #[test]
    fn test_vec_cosine_zero_vector() {
        let a = vec_of(&[0.0, 0.0]);
        let b = vec_of(&[1.0, 1.0]);
        let sim = VectorQueryExpander::cosine_similarity(&a, &b);
        assert!(
            sim.abs() < 1e-9,
            "zero vector should produce similarity 0.0, got {sim}"
        );
    }

    #[test]
    fn test_vec_cosine_identical_vectors() {
        let a = vec_of(&[0.3, 0.4, 0.5]);
        let sim = VectorQueryExpander::cosine_similarity(&a, &a);
        assert!(
            (sim - 1.0).abs() < 1e-9,
            "identical vectors should have similarity 1.0, got {sim}"
        );
    }

    // -- expand_query: known synonyms ---------------------------------------

    #[test]
    fn test_vec_expand_with_known_synonyms() {
        let mut exp = VectorQueryExpander::new(default_vec_config());
        let qv = vec_of(&[1.0, 0.0, 0.0]);
        // Add a synonym with high similarity
        exp.add_synonym("cat", "feline", vec_of(&[0.9, 0.1, 0.0]));
        let result = exp.expand_query("cat", &qv);
        assert_eq!(result.expansions.len(), 1);
        assert_eq!(result.expansions[0].term, "feline");
    }

    #[test]
    fn test_vec_expand_no_synonyms_returns_original() {
        let mut exp = VectorQueryExpander::new(default_vec_config());
        let qv = vec_of(&[1.0, 0.0, 0.0]);
        let result = exp.expand_query("unknown", &qv);
        assert!(result.expansions.is_empty());
        // Combined should be L2-normalised original
        let norm: f64 = result.combined.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-9,
            "combined vector should be normalised, norm = {norm}"
        );
    }

    // -- similarity threshold filtering -------------------------------------

    #[test]
    fn test_vec_similarity_threshold_filters_low() {
        let cfg = VectorExpanderConfig {
            similarity_threshold: 0.9,
            ..default_vec_config()
        };
        let mut exp = VectorQueryExpander::new(cfg);
        let qv = vec_of(&[1.0, 0.0, 0.0]);
        // This synonym has moderate similarity (~0.707)
        exp.add_synonym("x", "y", vec_of(&[1.0, 1.0, 0.0]));
        let result = exp.expand_query("x", &qv);
        assert!(
            result.expansions.is_empty(),
            "synonym below threshold should be filtered"
        );
    }

    #[test]
    fn test_vec_similarity_threshold_allows_high() {
        let cfg = VectorExpanderConfig {
            similarity_threshold: 0.5,
            ..default_vec_config()
        };
        let mut exp = VectorQueryExpander::new(cfg);
        let qv = vec_of(&[1.0, 0.0, 0.0]);
        // Similarity ~0.707, above 0.5
        exp.add_synonym("x", "y", vec_of(&[1.0, 1.0, 0.0]));
        let result = exp.expand_query("x", &qv);
        assert_eq!(result.expansions.len(), 1);
    }

    // -- weight decay ordering ----------------------------------------------

    #[test]
    fn test_vec_weight_decay_ordering() {
        let cfg = VectorExpanderConfig {
            similarity_threshold: 0.0,
            weight_decay: 0.5,
            max_expansions: 10,
        };
        let mut exp = VectorQueryExpander::new(cfg);
        let qv = vec_of(&[1.0, 0.0, 0.0]);
        // Two synonyms: first has higher similarity
        exp.add_synonym("q", "a", vec_of(&[0.95, 0.05, 0.0]));
        exp.add_synonym("q", "b", vec_of(&[0.8, 0.2, 0.0]));
        let result = exp.expand_query("q", &qv);
        assert!(result.expansions.len() >= 2);
        // Rank 0 should have higher weight than rank 1 (decay applied)
        assert!(
            result.expansions[0].weight > result.expansions[1].weight,
            "rank-0 weight {} should exceed rank-1 weight {}",
            result.expansions[0].weight,
            result.expansions[1].weight
        );
    }

    #[test]
    fn test_vec_weight_decay_values() {
        let cfg = VectorExpanderConfig {
            similarity_threshold: 0.0,
            weight_decay: 0.8,
            max_expansions: 10,
        };
        let mut exp = VectorQueryExpander::new(cfg);
        let qv = unit_vec(0, 3);
        exp.add_synonym("q", "a", vec_of(&[1.0, 0.0, 0.0])); // sim=1.0
        exp.add_synonym("q", "b", vec_of(&[1.0, 0.0, 0.0])); // sim=1.0

        let result = exp.expand_query("q", &qv);
        assert!(result.expansions.len() >= 2);
        // rank 0: weight = 1.0 * 0.8^0 = 1.0
        assert!(
            (result.expansions[0].weight - 1.0).abs() < 1e-9,
            "rank-0 weight should be 1.0, got {}",
            result.expansions[0].weight
        );
        // rank 1: weight = 1.0 * 0.8^1 = 0.8
        assert!(
            (result.expansions[1].weight - 0.8).abs() < 1e-9,
            "rank-1 weight should be 0.8, got {}",
            result.expansions[1].weight
        );
    }

    // -- combined vector normalisation --------------------------------------

    #[test]
    fn test_vec_combined_is_normalized() {
        let mut exp = VectorQueryExpander::new(default_vec_config());
        let qv = vec_of(&[3.0, 4.0, 0.0]);
        exp.add_synonym("t", "s1", vec_of(&[3.0, 4.0, 1.0]));
        let result = exp.expand_query("t", &qv);
        let norm: f64 = result.combined.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-9,
            "combined vector L2 norm should be ~1.0, got {norm}"
        );
    }

    #[test]
    fn test_vec_combined_no_expansions_still_normalized() {
        let mut exp = VectorQueryExpander::new(default_vec_config());
        let qv = vec_of(&[3.0, 4.0]);
        let result = exp.expand_query("none", &qv);
        let norm: f64 = result.combined.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-9,
            "combined vector should be normalised even with 0 expansions, norm = {norm}"
        );
    }

    // -- max_expansions cap -------------------------------------------------

    #[test]
    fn test_vec_max_expansions_cap() {
        let cfg = VectorExpanderConfig {
            max_expansions: 2,
            similarity_threshold: 0.0,
            ..default_vec_config()
        };
        let mut exp = VectorQueryExpander::new(cfg);
        let qv = unit_vec(0, 3);
        for i in 0..5 {
            exp.add_synonym("t", &format!("s{i}"), vec_of(&[1.0, 0.01 * i as f64, 0.0]));
        }
        let result = exp.expand_query("t", &qv);
        assert!(
            result.expansions.len() <= 2,
            "should cap at 2, got {}",
            result.expansions.len()
        );
    }

    // -- add / remove synonyms ----------------------------------------------

    #[test]
    fn test_vec_add_synonym() {
        let mut exp = VectorQueryExpander::new(default_vec_config());
        exp.add_synonym("a", "b", vec![1.0]);
        assert_eq!(exp.synonym_count("a"), 1);
    }

    #[test]
    fn test_vec_add_multiple_synonyms() {
        let mut exp = VectorQueryExpander::new(default_vec_config());
        exp.add_synonym("a", "b", vec![1.0]);
        exp.add_synonym("a", "c", vec![2.0]);
        assert_eq!(exp.synonym_count("a"), 2);
    }

    #[test]
    fn test_vec_remove_synonym_exists() {
        let mut exp = VectorQueryExpander::new(default_vec_config());
        exp.add_synonym("a", "b", vec![1.0]);
        assert!(exp.remove_synonym("a", "b"));
        assert_eq!(exp.synonym_count("a"), 0);
    }

    #[test]
    fn test_vec_remove_synonym_not_found() {
        let mut exp = VectorQueryExpander::new(default_vec_config());
        assert!(!exp.remove_synonym("a", "b"));
    }

    #[test]
    fn test_vec_remove_synonym_partial() {
        let mut exp = VectorQueryExpander::new(default_vec_config());
        exp.add_synonym("a", "b", vec![1.0]);
        exp.add_synonym("a", "c", vec![2.0]);
        assert!(exp.remove_synonym("a", "b"));
        assert_eq!(exp.synonym_count("a"), 1);
    }

    // -- empty query --------------------------------------------------------

    #[test]
    fn test_vec_empty_query_term() {
        let mut exp = VectorQueryExpander::new(default_vec_config());
        let qv = vec_of(&[1.0, 0.0]);
        let result = exp.expand_query("", &qv);
        assert!(result.expansions.is_empty());
    }

    // -- clear_synonyms -----------------------------------------------------

    #[test]
    fn test_vec_clear_synonyms() {
        let mut exp = VectorQueryExpander::new(default_vec_config());
        exp.add_synonym("a", "b", vec![1.0]);
        exp.add_synonym("c", "d", vec![2.0]);
        exp.clear_synonyms();
        assert_eq!(exp.synonym_count("a"), 0);
        assert_eq!(exp.synonym_count("c"), 0);
    }

    // -- stats --------------------------------------------------------------

    #[test]
    fn test_vec_stats_initial() {
        let exp = VectorQueryExpander::new(default_vec_config());
        let s = exp.stats();
        assert_eq!(s.total_terms, 0);
        assert_eq!(s.total_synonyms, 0);
        assert_eq!(s.expansions_performed, 0);
    }

    #[test]
    fn test_vec_stats_after_operations() {
        let mut exp = VectorQueryExpander::new(default_vec_config());
        exp.add_synonym("a", "b", vec![1.0]);
        exp.add_synonym("a", "c", vec![2.0]);
        exp.add_synonym("x", "y", vec![3.0]);
        let s = exp.stats();
        assert_eq!(s.total_terms, 2);
        assert_eq!(s.total_synonyms, 3);
    }

    #[test]
    fn test_vec_stats_expansions_performed() {
        let mut exp = VectorQueryExpander::new(default_vec_config());
        let qv = vec_of(&[1.0]);
        exp.expand_query("a", &qv);
        exp.expand_query("b", &qv);
        exp.expand_query("c", &qv);
        assert_eq!(exp.stats().expansions_performed, 3);
    }

    // -- multiple terms with overlapping synonyms ---------------------------

    #[test]
    fn test_vec_overlapping_synonyms_different_terms() {
        let cfg = VectorExpanderConfig {
            similarity_threshold: 0.0,
            ..default_vec_config()
        };
        let mut exp = VectorQueryExpander::new(cfg);
        // "shared_syn" is a synonym for both "a" and "b"
        exp.add_synonym("a", "shared_syn", vec_of(&[1.0, 0.0]));
        exp.add_synonym("b", "shared_syn", vec_of(&[0.0, 1.0]));

        let r_a = exp.expand_query("a", &vec_of(&[1.0, 0.0]));
        let r_b = exp.expand_query("b", &vec_of(&[0.0, 1.0]));
        assert_eq!(r_a.expansions.len(), 1);
        assert_eq!(r_b.expansions.len(), 1);
        assert_eq!(r_a.expansions[0].term, "shared_syn");
        assert_eq!(r_b.expansions[0].term, "shared_syn");
    }

    // -- combine_vectors standalone -----------------------------------------

    #[test]
    fn test_vec_combine_vectors_no_expansions() {
        let orig = vec_of(&[3.0, 4.0]);
        let combined = VectorQueryExpander::combine_vectors(&orig, &[]);
        let norm: f64 = combined.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        // Should point in same direction as original
        assert!(combined[0] > 0.0);
        assert!(combined[1] > 0.0);
    }

    #[test]
    fn test_vec_combine_vectors_weighted() {
        let orig = vec_of(&[1.0, 0.0]);
        let exp_vec = vec_of(&[0.0, 1.0]);
        let combined = VectorQueryExpander::combine_vectors(&orig, &[(exp_vec, 1.0)]);
        // With equal weight: (1+0)/2, (0+1)/2 = (0.5, 0.5), normalised
        let expected = 1.0 / 2.0_f64.sqrt();
        assert!(
            (combined[0] - expected).abs() < 1e-9,
            "expected {expected}, got {}",
            combined[0]
        );
        assert!(
            (combined[1] - expected).abs() < 1e-9,
            "expected {expected}, got {}",
            combined[1]
        );
    }

    // -- synonym_count for unknown term -------------------------------------

    #[test]
    fn test_vec_synonym_count_unknown_term() {
        let exp = VectorQueryExpander::new(default_vec_config());
        assert_eq!(exp.synonym_count("nonexistent"), 0);
    }
}
