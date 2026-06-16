//! Entity Disambiguation and Resolution
//!
//! Links entity mentions to canonical entities via exact match, alias lookup,
//! fuzzy string similarity, and embedding cosine similarity — in that priority order.

use std::collections::HashMap;

/// Broad semantic category for a canonical entity.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum EntityType {
    Person,
    Organization,
    Location,
    Product,
    Concept,
    Other(String),
}

/// A canonical entity stored in the resolver's registry.
#[derive(Debug, Clone)]
pub struct CanonicalEntity {
    /// Stable unique identifier.
    pub entity_id: String,
    /// The primary, canonical surface form.
    pub canonical_name: String,
    /// Semantic category.
    pub entity_type: EntityType,
    /// Alternative surface forms (aliases, abbreviations, etc.).
    pub aliases: Vec<String>,
    /// Optional prototype embedding vector (f64 to match workspace conventions).
    pub embedding: Option<Vec<f64>>,
    /// Confidence in the entity's representation quality; [0.0, 1.0].
    pub confidence: f64,
}

/// A mention of an entity as it appears in source text.
#[derive(Debug, Clone)]
pub struct EntityMention {
    /// The raw mention text (surface form).
    pub text: String,
    /// Byte offset of the first character of the mention in the source.
    pub start: usize,
    /// Byte offset one past the last character of the mention.
    pub end: usize,
    /// Surrounding text providing additional disambiguation signal.
    pub context: String,
}

/// How a mention was resolved to a canonical entity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResolutionMethod {
    /// Normalized mention text equals normalized canonical name.
    ExactMatch,
    /// Normalized mention text equals a normalized alias.
    AliasMatch,
    /// Edit-distance similarity meets the configured fuzzy threshold.
    FuzzyMatch,
    /// Embedding cosine similarity meets the configured threshold.
    EmbeddingMatch,
    /// No entity could be linked.
    Unresolved,
}

/// The result of resolving a single [`EntityMention`].
#[derive(Debug, Clone)]
pub struct ResolutionResult {
    /// The mention that was resolved.
    pub mention: EntityMention,
    /// Resolved entity ID, if any.
    pub entity_id: Option<String>,
    /// Resolved canonical name, if any.
    pub canonical_name: Option<String>,
    /// Confidence score in [0.0, 1.0].
    pub confidence: f64,
    /// Method used to resolve (or fail to resolve) this mention.
    pub method: ResolutionMethod,
}

/// Tunable parameters for [`EntityResolver`].
#[derive(Debug, Clone)]
pub struct ResolverConfig {
    /// Minimum string-similarity score (0.0–1.0) for a fuzzy match to be accepted.
    pub fuzzy_threshold: f64,
    /// Minimum cosine similarity (0.0–1.0) for an embedding match to be accepted.
    pub embedding_threshold: f64,
    /// Maximum number of candidate entities examined per mention.
    pub max_candidates: usize,
    /// When `false` (default), all comparisons are case-insensitive.
    pub case_sensitive: bool,
}

impl Default for ResolverConfig {
    fn default() -> Self {
        Self {
            fuzzy_threshold: 0.8,
            embedding_threshold: 0.85,
            max_candidates: 10,
            case_sensitive: false,
        }
    }
}

/// Running counters updated by [`EntityResolver`] during resolution.
#[derive(Debug, Clone, Default)]
pub struct ResolverStats {
    pub total_resolved: u64,
    pub exact_matches: u64,
    pub alias_matches: u64,
    pub fuzzy_matches: u64,
    pub embedding_matches: u64,
    pub unresolved: u64,
}

/// Disambiguates and resolves entity mentions to canonical entities.
///
/// Resolution priority (highest to lowest):
/// 1. Exact match on normalized canonical name.
/// 2. Alias match on any normalized alias.
/// 3. Fuzzy (edit-distance) match if similarity ≥ `fuzzy_threshold`.
/// 4. Embedding cosine similarity match if score ≥ `embedding_threshold`.
/// 5. Unresolved.
pub struct EntityResolver {
    config: ResolverConfig,
    /// Primary store: entity_id → entity.
    entities: HashMap<String, CanonicalEntity>,
    /// Inverted index: normalized surface form / alias → entity_id.
    alias_index: HashMap<String, String>,
    stats: ResolverStats,
}

impl EntityResolver {
    /// Create a new resolver with the given configuration.
    pub fn new(config: ResolverConfig) -> Self {
        Self {
            config,
            entities: HashMap::new(),
            alias_index: HashMap::new(),
            stats: ResolverStats::default(),
        }
    }

    /// Register a canonical entity.  Returns `true` if the entity was inserted,
    /// `false` if an entity with the same `entity_id` already exists (no update).
    pub fn register_entity(&mut self, entity: CanonicalEntity) -> bool {
        if self.entities.contains_key(&entity.entity_id) {
            return false;
        }

        // Index canonical name.
        let norm_canonical = Self::normalize(&entity.canonical_name);
        self.alias_index
            .entry(norm_canonical)
            .or_insert_with(|| entity.entity_id.clone());

        // Index each alias.
        for alias in &entity.aliases {
            let norm_alias = Self::normalize(alias);
            self.alias_index
                .entry(norm_alias)
                .or_insert_with(|| entity.entity_id.clone());
        }

        self.entities.insert(entity.entity_id.clone(), entity);
        true
    }

    /// Resolve a single mention to a canonical entity.
    pub fn resolve(&mut self, mention: EntityMention) -> ResolutionResult {
        let norm_mention = if self.config.case_sensitive {
            mention.text.trim().to_string()
        } else {
            Self::normalize(&mention.text)
        };

        // 1. Exact / alias match via inverted index (O(1)).
        if let Some(entity_id) = self.alias_index.get(&norm_mention) {
            if let Some(entity) = self.entities.get(entity_id) {
                let norm_canonical = Self::normalize(&entity.canonical_name);
                let method = if norm_mention == norm_canonical {
                    self.stats.exact_matches += 1;
                    ResolutionMethod::ExactMatch
                } else {
                    self.stats.alias_matches += 1;
                    ResolutionMethod::AliasMatch
                };
                self.stats.total_resolved += 1;
                return ResolutionResult {
                    mention,
                    entity_id: Some(entity_id.clone()),
                    canonical_name: Some(entity.canonical_name.clone()),
                    confidence: entity.confidence,
                    method,
                };
            }
        }

        // 2. Fuzzy scan over top-N candidates.
        // Collect owned data up-front so we don't hold borrows into `self`
        // when we later mutate `self.stats`.
        let max_candidates = self.config.max_candidates;
        let fuzzy_threshold = self.config.fuzzy_threshold;

        // Each entry: (entity_id, canonical_name, best_sim, entity_confidence)
        let candidate_data: Vec<(String, String, f64, f64)> = {
            let candidates = self.find_candidates(&norm_mention, max_candidates);
            candidates
                .iter()
                .map(|e| {
                    let norm_cand = Self::normalize(&e.canonical_name);
                    let sim = Self::string_similarity(&norm_mention, &norm_cand);
                    (
                        e.entity_id.clone(),
                        e.canonical_name.clone(),
                        sim,
                        e.confidence,
                    )
                })
                .collect()
        };

        // --- Fuzzy pass ---
        let best_fuzzy = candidate_data
            .iter()
            .filter(|(_, _, sim, _)| *sim >= fuzzy_threshold)
            .max_by(|(_, _, sa, _), (_, _, sb, _)| {
                sa.partial_cmp(sb).unwrap_or(std::cmp::Ordering::Equal)
            });

        if let Some((entity_id, canonical_name, sim, conf)) = best_fuzzy {
            self.stats.fuzzy_matches += 1;
            self.stats.total_resolved += 1;
            return ResolutionResult {
                mention,
                entity_id: Some(entity_id.clone()),
                canonical_name: Some(canonical_name.clone()),
                confidence: sim * conf,
                method: ResolutionMethod::FuzzyMatch,
            };
        }

        // --- Embedding pass (no mention embedding available in `resolve`) ---
        // Callers with a query embedding should use `resolve_with_embedding`.
        // Here we simply fall through to Unresolved.

        // Unresolved.
        self.stats.unresolved += 1;
        ResolutionResult {
            mention,
            entity_id: None,
            canonical_name: None,
            confidence: 0.0,
            method: ResolutionMethod::Unresolved,
        }
    }

    /// Resolve a mention when the caller also supplies a query embedding.
    ///
    /// Falls through the same priority chain as `resolve` but adds an
    /// embedding-cosine step before returning Unresolved.
    pub fn resolve_with_embedding(
        &mut self,
        mention: EntityMention,
        query_embedding: &[f64],
    ) -> ResolutionResult {
        let norm_mention = if self.config.case_sensitive {
            mention.text.trim().to_string()
        } else {
            Self::normalize(&mention.text)
        };

        // 1. Exact / alias via index.
        if let Some(entity_id) = self.alias_index.get(&norm_mention).cloned() {
            if let Some(entity) = self.entities.get(&entity_id) {
                let norm_canonical = Self::normalize(&entity.canonical_name);
                let method = if norm_mention == norm_canonical {
                    self.stats.exact_matches += 1;
                    ResolutionMethod::ExactMatch
                } else {
                    self.stats.alias_matches += 1;
                    ResolutionMethod::AliasMatch
                };
                self.stats.total_resolved += 1;
                return ResolutionResult {
                    mention,
                    entity_id: Some(entity_id),
                    canonical_name: Some(entity.canonical_name.clone()),
                    confidence: entity.confidence,
                    method,
                };
            }
        }

        let candidates = self.find_candidates(&norm_mention, self.config.max_candidates);

        // 2. Fuzzy.
        let fuzzy_threshold = self.config.fuzzy_threshold;
        let mut best_fuzzy: Option<(String, String, f64, f64)> = None; // (id, name, sim, conf)
        for candidate in &candidates {
            let norm_cand = Self::normalize(&candidate.canonical_name);
            let sim = Self::string_similarity(&norm_mention, &norm_cand);
            if sim >= fuzzy_threshold {
                let better = best_fuzzy
                    .as_ref()
                    .is_none_or(|(_, _, prev, _)| sim > *prev);
                if better {
                    best_fuzzy = Some((
                        candidate.entity_id.clone(),
                        candidate.canonical_name.clone(),
                        sim,
                        candidate.confidence,
                    ));
                }
            }
        }
        if let Some((entity_id, canonical_name, sim, conf)) = best_fuzzy {
            self.stats.fuzzy_matches += 1;
            self.stats.total_resolved += 1;
            return ResolutionResult {
                mention,
                entity_id: Some(entity_id),
                canonical_name: Some(canonical_name),
                confidence: sim * conf,
                method: ResolutionMethod::FuzzyMatch,
            };
        }

        // 3. Embedding cosine similarity.
        let embedding_threshold = self.config.embedding_threshold;
        let mut best_emb: Option<(String, String, f64, f64)> = None;
        for candidate in &candidates {
            if let Some(emb) = &candidate.embedding {
                let sim = Self::cosine_similarity(query_embedding, emb);
                if sim >= embedding_threshold {
                    let better = best_emb.as_ref().is_none_or(|(_, _, prev, _)| sim > *prev);
                    if better {
                        best_emb = Some((
                            candidate.entity_id.clone(),
                            candidate.canonical_name.clone(),
                            sim,
                            candidate.confidence,
                        ));
                    }
                }
            }
        }
        if let Some((entity_id, canonical_name, sim, conf)) = best_emb {
            self.stats.embedding_matches += 1;
            self.stats.total_resolved += 1;
            return ResolutionResult {
                mention,
                entity_id: Some(entity_id),
                canonical_name: Some(canonical_name),
                confidence: sim * conf,
                method: ResolutionMethod::EmbeddingMatch,
            };
        }

        // Unresolved.
        self.stats.unresolved += 1;
        ResolutionResult {
            mention,
            entity_id: None,
            canonical_name: None,
            confidence: 0.0,
            method: ResolutionMethod::Unresolved,
        }
    }

    /// Resolve a batch of mentions, returning one [`ResolutionResult`] per mention.
    pub fn resolve_batch(&mut self, mentions: Vec<EntityMention>) -> Vec<ResolutionResult> {
        mentions.into_iter().map(|m| self.resolve(m)).collect()
    }

    /// Normalize a string: trim, lowercase, collapse interior whitespace runs.
    pub fn normalize(text: &str) -> String {
        text.trim()
            .to_lowercase()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Classic Levenshtein edit distance between two strings.
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

        // Use two rows to keep memory O(n).
        let mut prev: Vec<usize> = (0..=n).collect();
        let mut curr = vec![0usize; n + 1];

        for i in 1..=m {
            curr[0] = i;
            for j in 1..=n {
                let cost = if a_chars[i - 1] == b_chars[j - 1] {
                    0
                } else {
                    1
                };
                curr[j] = (prev[j] + 1).min(curr[j - 1] + 1).min(prev[j - 1] + cost);
            }
            std::mem::swap(&mut prev, &mut curr);
        }

        prev[n]
    }

    /// String similarity in [0.0, 1.0]: `1.0 - edit_distance / max_len`.
    /// Returns 1.0 if both strings are empty.
    pub fn string_similarity(a: &str, b: &str) -> f64 {
        let max_len = a.chars().count().max(b.chars().count());
        if max_len == 0 {
            return 1.0;
        }
        let dist = Self::edit_distance(a, b);
        1.0 - (dist as f64 / max_len as f64)
    }

    /// Cosine similarity between two vectors; returns 0.0 for zero-norm inputs.
    pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
        if a.is_empty() || b.is_empty() || a.len() != b.len() {
            return 0.0;
        }
        let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }
        (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
    }

    /// Return up to `n` candidate entities ranked by string similarity to `mention`.
    ///
    /// This is an O(|entities|) scan.  For very large registries a dedicated
    /// inverted-index or BK-tree would be more efficient, but correctness is
    /// the priority here.
    pub fn find_candidates<'a>(&'a self, mention: &str, n: usize) -> Vec<&'a CanonicalEntity> {
        if n == 0 {
            return Vec::new();
        }

        let norm_mention = Self::normalize(mention);
        let mut scored: Vec<(f64, &CanonicalEntity)> = self
            .entities
            .values()
            .map(|e| {
                let norm_name = Self::normalize(&e.canonical_name);
                let name_sim = Self::string_similarity(&norm_mention, &norm_name);
                // Also score against each alias, take the best.
                let alias_sim = e
                    .aliases
                    .iter()
                    .map(|a| Self::string_similarity(&norm_mention, &Self::normalize(a)))
                    .fold(0.0_f64, f64::max);
                let best_sim = name_sim.max(alias_sim);
                (best_sim, e)
            })
            .collect();

        // Sort descending by similarity, then alphabetically for determinism.
        scored.sort_by(|(sa, ea), (sb, eb)| {
            sb.partial_cmp(sa)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| ea.entity_id.cmp(&eb.entity_id))
        });

        scored.into_iter().take(n).map(|(_, e)| e).collect()
    }

    /// Number of registered canonical entities.
    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }

    /// Look up a canonical entity by its stable ID.
    pub fn get_entity(&self, entity_id: &str) -> Option<&CanonicalEntity> {
        self.entities.get(entity_id)
    }

    /// Current resolution statistics.
    pub fn stats(&self) -> &ResolverStats {
        &self.stats
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn make_entity(
        id: &str,
        name: &str,
        ty: EntityType,
        aliases: Vec<&str>,
        embedding: Option<Vec<f64>>,
    ) -> CanonicalEntity {
        CanonicalEntity {
            entity_id: id.to_string(),
            canonical_name: name.to_string(),
            entity_type: ty,
            aliases: aliases.into_iter().map(|s| s.to_string()).collect(),
            embedding,
            confidence: 1.0,
        }
    }

    fn mention(text: &str) -> EntityMention {
        EntityMention {
            text: text.to_string(),
            start: 0,
            end: text.len(),
            context: String::new(),
        }
    }

    fn default_resolver() -> EntityResolver {
        EntityResolver::new(ResolverConfig::default())
    }

    // ── edit_distance ─────────────────────────────────────────────────────────

    #[test]
    fn test_edit_distance_identical() {
        assert_eq!(EntityResolver::edit_distance("hello", "hello"), 0);
    }

    #[test]
    fn test_edit_distance_empty_left() {
        assert_eq!(EntityResolver::edit_distance("", "abc"), 3);
    }

    #[test]
    fn test_edit_distance_empty_right() {
        assert_eq!(EntityResolver::edit_distance("abc", ""), 3);
    }

    #[test]
    fn test_edit_distance_both_empty() {
        assert_eq!(EntityResolver::edit_distance("", ""), 0);
    }

    #[test]
    fn test_edit_distance_kitten_sitting() {
        // Classic example: kitten → sitting = 3
        assert_eq!(EntityResolver::edit_distance("kitten", "sitting"), 3);
    }

    #[test]
    fn test_edit_distance_one_insertion() {
        assert_eq!(EntityResolver::edit_distance("cat", "cats"), 1);
    }

    // ── string_similarity ────────────────────────────────────────────────────

    #[test]
    fn test_string_similarity_identical() {
        let s = EntityResolver::string_similarity("apple", "apple");
        assert!((s - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_string_similarity_both_empty() {
        let s = EntityResolver::string_similarity("", "");
        assert!((s - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_string_similarity_completely_different() {
        // "ab" vs "cd" → edit distance 2, max_len 2 → sim 0.0
        let s = EntityResolver::string_similarity("ab", "cd");
        assert!((s - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_string_similarity_partial() {
        let s = EntityResolver::string_similarity("kitten", "sitting");
        // 1 - 3/7 ≈ 0.571
        assert!(s > 0.5 && s < 0.7);
    }

    // ── cosine_similarity ────────────────────────────────────────────────────

    #[test]
    fn test_cosine_similarity_identical() {
        let v = vec![1.0, 0.0, 0.0];
        let s = EntityResolver::cosine_similarity(&v, &v);
        assert!((s - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let s = EntityResolver::cosine_similarity(&a, &b);
        assert!(s.abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let a = vec![0.0, 0.0];
        let b = vec![1.0, 1.0];
        let s = EntityResolver::cosine_similarity(&a, &b);
        assert!(s.abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_mismatched_len() {
        let a = vec![1.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let s = EntityResolver::cosine_similarity(&a, &b);
        assert!(s.abs() < 1e-10);
    }

    // ── normalize ─────────────────────────────────────────────────────────────

    #[test]
    fn test_normalize_trims_and_lowercases() {
        assert_eq!(EntityResolver::normalize("  Hello World  "), "hello world");
    }

    #[test]
    fn test_normalize_collapses_whitespace() {
        assert_eq!(EntityResolver::normalize("foo   bar\tbaz"), "foo bar baz");
    }

    // ── register_entity ───────────────────────────────────────────────────────

    #[test]
    fn test_register_entity_success() {
        let mut r = default_resolver();
        let e = make_entity("e1", "Apple", EntityType::Organization, vec!["AAPL"], None);
        assert!(r.register_entity(e));
        assert_eq!(r.entity_count(), 1);
    }

    #[test]
    fn test_register_entity_duplicate_returns_false() {
        let mut r = default_resolver();
        let e1 = make_entity("e1", "Apple", EntityType::Organization, vec![], None);
        let e2 = make_entity("e1", "Apple Inc.", EntityType::Organization, vec![], None);
        assert!(r.register_entity(e1));
        assert!(!r.register_entity(e2));
        // Original should still be there.
        assert_eq!(
            r.get_entity("e1").map(|e| e.canonical_name.as_str()),
            Some("Apple")
        );
    }

    // ── resolve — exact match ─────────────────────────────────────────────────

    #[test]
    fn test_resolve_exact_match() {
        let mut r = default_resolver();
        r.register_entity(make_entity(
            "e1",
            "Apple",
            EntityType::Organization,
            vec![],
            None,
        ));
        let res = r.resolve(mention("Apple"));
        assert_eq!(res.method, ResolutionMethod::ExactMatch);
        assert_eq!(res.entity_id.as_deref(), Some("e1"));
    }

    #[test]
    fn test_resolve_exact_match_case_insensitive() {
        let mut r = default_resolver();
        r.register_entity(make_entity(
            "e1",
            "Apple",
            EntityType::Organization,
            vec![],
            None,
        ));
        let res = r.resolve(mention("APPLE"));
        assert_eq!(res.method, ResolutionMethod::ExactMatch);
        assert_eq!(res.entity_id.as_deref(), Some("e1"));
    }

    // ── resolve — alias match ─────────────────────────────────────────────────

    #[test]
    fn test_resolve_alias_match() {
        let mut r = default_resolver();
        r.register_entity(make_entity(
            "e1",
            "Apple Inc.",
            EntityType::Organization,
            vec!["Apple", "AAPL"],
            None,
        ));
        let res = r.resolve(mention("aapl"));
        assert_eq!(res.method, ResolutionMethod::AliasMatch);
        assert_eq!(res.entity_id.as_deref(), Some("e1"));
    }

    // ── resolve — fuzzy match ─────────────────────────────────────────────────

    #[test]
    fn test_resolve_fuzzy_match_above_threshold() {
        // "Aple" vs "Apple": edit_distance = 1, max_len = 5, sim = 0.8 → meets default 0.8
        let mut r = default_resolver();
        r.register_entity(make_entity(
            "e1",
            "Apple",
            EntityType::Organization,
            vec![],
            None,
        ));
        let res = r.resolve(mention("Aple"));
        assert_eq!(res.method, ResolutionMethod::FuzzyMatch);
        assert_eq!(res.entity_id.as_deref(), Some("e1"));
    }

    #[test]
    fn test_resolve_fuzzy_below_threshold_is_unresolved() {
        // Very dissimilar string — should not meet threshold
        let mut r = default_resolver();
        r.register_entity(make_entity(
            "e1",
            "Apple",
            EntityType::Organization,
            vec![],
            None,
        ));
        let res = r.resolve(mention("XYZ"));
        assert_eq!(res.method, ResolutionMethod::Unresolved);
        assert!(res.entity_id.is_none());
    }

    // ── resolve — embedding match ─────────────────────────────────────────────

    #[test]
    fn test_resolve_embedding_match() {
        let emb = vec![1.0, 0.0, 0.0];
        let query = vec![0.99, 0.14, 0.0]; // cosine ~ 0.99

        let mut r = EntityResolver::new(ResolverConfig {
            fuzzy_threshold: 0.99, // force fuzzy to fail
            embedding_threshold: 0.9,
            max_candidates: 10,
            case_sensitive: false,
        });
        r.register_entity(make_entity(
            "e1",
            "TechCorp",
            EntityType::Organization,
            vec![],
            Some(emb),
        ));
        let res = r.resolve_with_embedding(mention("TechCorp-X"), &query);
        assert_eq!(res.method, ResolutionMethod::EmbeddingMatch);
        assert_eq!(res.entity_id.as_deref(), Some("e1"));
    }

    // ── resolve — multi-method fallback ──────────────────────────────────────

    #[test]
    fn test_resolve_falls_back_through_methods() {
        // Register entity; query with something close but not exact/alias.
        let mut r = EntityResolver::new(ResolverConfig {
            fuzzy_threshold: 0.6,
            embedding_threshold: 0.9,
            max_candidates: 10,
            case_sensitive: false,
        });
        r.register_entity(make_entity(
            "e1",
            "Microsoft",
            EntityType::Organization,
            vec![],
            None,
        ));
        // "Micr0soft" — one substitution; similarity = 1 - 1/9 ≈ 0.89 ≥ 0.6
        let res = r.resolve(mention("Micr0soft"));
        assert_eq!(res.method, ResolutionMethod::FuzzyMatch);
    }

    // ── batch resolution ─────────────────────────────────────────────────────

    #[test]
    fn test_resolve_batch() {
        let mut r = default_resolver();
        r.register_entity(make_entity("e1", "Alice", EntityType::Person, vec![], None));
        r.register_entity(make_entity("e2", "Bob", EntityType::Person, vec![], None));
        let results = r.resolve_batch(vec![mention("Alice"), mention("Bob"), mention("Unknown")]);
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].entity_id.as_deref(), Some("e1"));
        assert_eq!(results[1].entity_id.as_deref(), Some("e2"));
        assert!(results[2].entity_id.is_none());
    }

    // ── case sensitivity ─────────────────────────────────────────────────────

    #[test]
    fn test_resolve_case_sensitive_no_match() {
        let mut r = EntityResolver::new(ResolverConfig {
            case_sensitive: true,
            fuzzy_threshold: 1.1, // impossible threshold — disable fuzzy
            ..ResolverConfig::default()
        });
        r.register_entity(make_entity(
            "e1",
            "Apple",
            EntityType::Organization,
            vec![],
            None,
        ));
        // The alias index is keyed by normalize("Apple") = "apple".
        // When case_sensitive=true the lookup key is the raw mention text.
        // Mention "APPLE" (all caps) → raw key "APPLE" ≠ alias key "apple" → no match.
        let res = r.resolve(mention("APPLE"));
        assert_eq!(res.entity_id.as_deref(), None);
    }

    // ── stats tracking ────────────────────────────────────────────────────────

    #[test]
    fn test_stats_exact_match_increments() {
        let mut r = default_resolver();
        r.register_entity(make_entity(
            "e1",
            "Google",
            EntityType::Organization,
            vec![],
            None,
        ));
        r.resolve(mention("Google"));
        let s = r.stats();
        assert_eq!(s.exact_matches, 1);
        assert_eq!(s.total_resolved, 1);
        assert_eq!(s.unresolved, 0);
    }

    #[test]
    fn test_stats_alias_match_increments() {
        let mut r = default_resolver();
        r.register_entity(make_entity(
            "e1",
            "Alphabet",
            EntityType::Organization,
            vec!["Google"],
            None,
        ));
        r.resolve(mention("Google"));
        let s = r.stats();
        assert_eq!(s.alias_matches, 1);
        assert_eq!(s.total_resolved, 1);
    }

    #[test]
    fn test_stats_unresolved_increments() {
        let mut r = default_resolver();
        r.register_entity(make_entity(
            "e1",
            "Google",
            EntityType::Organization,
            vec![],
            None,
        ));
        r.resolve(mention("zzzzzzz"));
        let s = r.stats();
        assert_eq!(s.unresolved, 1);
        assert_eq!(s.total_resolved, 0);
    }

    #[test]
    fn test_stats_fuzzy_increments() {
        let mut r = EntityResolver::new(ResolverConfig {
            fuzzy_threshold: 0.6,
            ..ResolverConfig::default()
        });
        r.register_entity(make_entity(
            "e1",
            "Google",
            EntityType::Organization,
            vec![],
            None,
        ));
        // "Gogle" — 1 deletion from "google", sim = 5/6 ≈ 0.83 ≥ 0.6
        r.resolve(mention("Gogle"));
        let s = r.stats();
        assert_eq!(s.fuzzy_matches, 1);
    }

    // ── empty mention ─────────────────────────────────────────────────────────

    #[test]
    fn test_resolve_empty_mention() {
        let mut r = default_resolver();
        r.register_entity(make_entity(
            "e1",
            "Apple",
            EntityType::Organization,
            vec![],
            None,
        ));
        let res = r.resolve(mention(""));
        // Empty string has similarity 0 with "apple" (max_len = 5, dist = 5, sim = 0).
        assert_eq!(res.method, ResolutionMethod::Unresolved);
    }

    // ── get_entity / entity_count ─────────────────────────────────────────────

    #[test]
    fn test_get_entity_present() {
        let mut r = default_resolver();
        r.register_entity(make_entity(
            "e1",
            "Apple",
            EntityType::Organization,
            vec![],
            None,
        ));
        let e = r.get_entity("e1");
        assert!(e.is_some());
        assert_eq!(e.map(|x| x.canonical_name.as_str()), Some("Apple"));
    }

    #[test]
    fn test_get_entity_absent() {
        let r = default_resolver();
        assert!(r.get_entity("nonexistent").is_none());
    }

    #[test]
    fn test_entity_count_empty() {
        let r = default_resolver();
        assert_eq!(r.entity_count(), 0);
    }

    #[test]
    fn test_entity_count_after_registration() {
        let mut r = default_resolver();
        r.register_entity(make_entity("e1", "A", EntityType::Concept, vec![], None));
        r.register_entity(make_entity("e2", "B", EntityType::Concept, vec![], None));
        assert_eq!(r.entity_count(), 2);
    }

    // ── EntityType ────────────────────────────────────────────────────────────

    #[test]
    fn test_entity_type_other_equality() {
        let t1 = EntityType::Other("custom".to_string());
        let t2 = EntityType::Other("custom".to_string());
        let t3 = EntityType::Other("other".to_string());
        assert_eq!(t1, t2);
        assert_ne!(t1, t3);
    }

    // ── find_candidates ───────────────────────────────────────────────────────

    #[test]
    fn test_find_candidates_limits_results() {
        let mut r = default_resolver();
        for i in 0..20_u32 {
            r.register_entity(make_entity(
                &format!("e{i}"),
                &format!("entity{i}"),
                EntityType::Concept,
                vec![],
                None,
            ));
        }
        let candidates = r.find_candidates("entity", 5);
        assert_eq!(candidates.len(), 5);
    }

    #[test]
    fn test_find_candidates_empty_registry() {
        let r = default_resolver();
        let candidates = r.find_candidates("anything", 10);
        assert!(candidates.is_empty());
    }

    // ── ResolverStats default ─────────────────────────────────────────────────

    #[test]
    fn test_resolver_stats_default_zeroes() {
        let s = ResolverStats::default();
        assert_eq!(s.total_resolved, 0);
        assert_eq!(s.exact_matches, 0);
        assert_eq!(s.alias_matches, 0);
        assert_eq!(s.fuzzy_matches, 0);
        assert_eq!(s.embedding_matches, 0);
        assert_eq!(s.unresolved, 0);
    }
}
