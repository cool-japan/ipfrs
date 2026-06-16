//! Semantic Entity Linker
//!
//! Links named entity mentions in queries to their canonical entities in a knowledge base,
//! using embedding similarity and surface form matching.

use std::collections::HashMap;

/// The kind of match that linked a mention to an entity.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum MentionKind {
    /// Surface form exactly matches entity canonical name.
    ExactMatch,
    /// Normalized surface form matches entity name.
    FuzzyMatch,
    /// Linked via embedding cosine similarity.
    EmbeddingMatch,
    /// Surface form is a known alias for the entity.
    Alias,
}

/// A canonical entity stored in the knowledge base.
#[derive(Clone, Debug)]
pub struct KbEntity {
    /// Unique identifier for this entity.
    pub entity_id: u64,
    /// The canonical (primary) name of the entity.
    pub canonical_name: String,
    /// Alternative surface forms that refer to this entity.
    pub aliases: Vec<String>,
    /// Prototype embedding vector for this entity.
    pub embedding: Vec<f32>,
    /// Semantic type, e.g. "person", "org", "location".
    pub entity_type: String,
}

/// A resolved link from a mention text to a knowledge-base entity.
#[derive(Clone, Debug)]
pub struct LinkedMention {
    /// The original mention text as it appeared in the query.
    pub mention_text: String,
    /// The entity that this mention was resolved to.
    pub entity_id: u64,
    /// The kind of evidence used to resolve this mention.
    pub kind: MentionKind,
    /// Confidence score in [0.0, 1.0].
    pub confidence: f32,
}

/// Configuration for the `SemanticEntityLinker`.
#[derive(Clone, Debug)]
pub struct LinkerConfig {
    /// Confidence assigned to an exact-name match.
    pub exact_match_confidence: f32,
    /// Confidence assigned when a mention matches a known alias.
    pub alias_match_confidence: f32,
    /// Minimum cosine similarity required to accept an embedding match.
    pub embedding_threshold: f32,
    /// `confidence = cosine_similarity * embedding_match_confidence_scale`.
    pub embedding_match_confidence_scale: f32,
}

impl Default for LinkerConfig {
    fn default() -> Self {
        Self {
            exact_match_confidence: 1.0,
            alias_match_confidence: 0.95,
            embedding_threshold: 0.80,
            embedding_match_confidence_scale: 0.9,
        }
    }
}

/// Aggregated statistics for a `SemanticEntityLinker` instance.
#[derive(Clone, Debug, Default)]
pub struct LinkerStats {
    /// Number of entities currently registered.
    pub total_entities: usize,
    /// Total number of successful links produced.
    pub total_links: u64,
    /// Links resolved via exact name match.
    pub exact_match_count: u64,
    /// Links resolved via alias match.
    pub alias_match_count: u64,
    /// Links resolved via embedding similarity.
    pub embedding_match_count: u64,
    /// Mention texts that could not be resolved to any entity.
    pub unlinked_count: u64,
}

/// Compute the cosine similarity between two equal-length float vectors.
///
/// Returns `0.0` if either vector has zero magnitude.
pub fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}

/// A linker that resolves mention texts to canonical knowledge-base entities.
///
/// Resolution is attempted in priority order:
/// 1. Exact name match (lowercase comparison against canonical names)
/// 2. Alias match (lowercase comparison against known aliases)
/// 3. Embedding similarity (cosine similarity above configured threshold)
pub struct SemanticEntityLinker {
    /// All registered entities, keyed by `entity_id`.
    pub entities: HashMap<u64, KbEntity>,
    /// Lowercase canonical name → entity_id.
    pub name_index: HashMap<String, u64>,
    /// Lowercase alias → entity_id.
    pub alias_index: HashMap<String, u64>,
    /// Configuration parameters.
    pub config: LinkerConfig,
    /// Running statistics.
    pub stats: LinkerStats,
}

impl SemanticEntityLinker {
    /// Create a new, empty linker with the given configuration.
    pub fn new(config: LinkerConfig) -> Self {
        Self {
            entities: HashMap::new(),
            name_index: HashMap::new(),
            alias_index: HashMap::new(),
            config,
            stats: LinkerStats::default(),
        }
    }

    /// Register an entity in the knowledge base.
    ///
    /// Indexes the entity's canonical name and all aliases (lowercased).
    /// If an entity with the same `entity_id` was already registered, it is replaced.
    pub fn register(&mut self, entity: KbEntity) {
        let entity_id = entity.entity_id;

        // Index canonical name (lowercased).
        self.name_index
            .insert(entity.canonical_name.to_lowercase(), entity_id);

        // Index every alias (lowercased).
        for alias in &entity.aliases {
            self.alias_index.insert(alias.to_lowercase(), entity_id);
        }

        self.entities.insert(entity_id, entity);
        self.stats.total_entities = self.entities.len();
    }

    /// Attempt to link a single mention text to an entity.
    ///
    /// Resolution priority:
    /// 1. **ExactMatch** — lowercase mention in `name_index`
    /// 2. **Alias** — lowercase mention in `alias_index`
    /// 3. **EmbeddingMatch** — highest cosine similarity ≥ `embedding_threshold`
    ///
    /// Returns `None` and increments `unlinked_count` if no match is found.
    pub fn link(
        &mut self,
        mention_text: &str,
        mention_embedding: Option<&[f32]>,
    ) -> Option<LinkedMention> {
        let lower = mention_text.to_lowercase();

        // 1. Exact name match.
        if let Some(&entity_id) = self.name_index.get(&lower) {
            let confidence = self.config.exact_match_confidence;
            self.stats.exact_match_count += 1;
            self.stats.total_links += 1;
            return Some(LinkedMention {
                mention_text: mention_text.to_owned(),
                entity_id,
                kind: MentionKind::ExactMatch,
                confidence,
            });
        }

        // 2. Alias match.
        if let Some(&entity_id) = self.alias_index.get(&lower) {
            let confidence = self.config.alias_match_confidence;
            self.stats.alias_match_count += 1;
            self.stats.total_links += 1;
            return Some(LinkedMention {
                mention_text: mention_text.to_owned(),
                entity_id,
                kind: MentionKind::Alias,
                confidence,
            });
        }

        // 3. Embedding similarity match.
        if let Some(query_emb) = mention_embedding {
            let threshold = self.config.embedding_threshold;
            let scale = self.config.embedding_match_confidence_scale;

            let best = self
                .entities
                .values()
                .filter(|e| !e.embedding.is_empty())
                .map(|e| {
                    let sim = cosine_sim(query_emb, &e.embedding);
                    (e.entity_id, sim)
                })
                .filter(|&(_, sim)| sim >= threshold)
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            if let Some((entity_id, sim)) = best {
                let confidence = sim * scale;
                self.stats.embedding_match_count += 1;
                self.stats.total_links += 1;
                return Some(LinkedMention {
                    mention_text: mention_text.to_owned(),
                    entity_id,
                    kind: MentionKind::EmbeddingMatch,
                    confidence,
                });
            }
        }

        // No match found.
        self.stats.unlinked_count += 1;
        None
    }

    /// Link a batch of mentions, each paired with an optional embedding.
    ///
    /// Returns one `Option<LinkedMention>` per input mention, in the same order.
    pub fn link_batch(
        &mut self,
        mentions: Vec<(String, Option<Vec<f32>>)>,
    ) -> Vec<Option<LinkedMention>> {
        mentions
            .into_iter()
            .map(|(text, emb)| self.link(&text, emb.as_deref()))
            .collect()
    }

    /// Look up a registered entity by its `entity_id`.
    pub fn entity(&self, entity_id: u64) -> Option<&KbEntity> {
        self.entities.get(&entity_id)
    }

    /// Return a reference to the current linker statistics.
    pub fn stats(&self) -> &LinkerStats {
        &self.stats
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ──────────────────────────────────────────────────────────────

    fn make_entity(id: u64, name: &str, aliases: &[&str], emb: Vec<f32>, etype: &str) -> KbEntity {
        KbEntity {
            entity_id: id,
            canonical_name: name.to_owned(),
            aliases: aliases.iter().map(|s| s.to_string()).collect(),
            embedding: emb,
            entity_type: etype.to_owned(),
        }
    }

    fn default_linker() -> SemanticEntityLinker {
        SemanticEntityLinker::new(LinkerConfig::default())
    }

    // Normalised 3-D unit vector: [1, 0, 0]
    fn emb_a() -> Vec<f32> {
        vec![1.0, 0.0, 0.0]
    }
    // Normalised 3-D vector close to emb_a: cosine ≈ 0.9487
    fn emb_a_close() -> Vec<f32> {
        let raw = vec![2.0_f32, 1.0, 0.0];
        let norm = raw.iter().map(|x| x * x).sum::<f32>().sqrt();
        raw.into_iter().map(|x| x / norm).collect()
    }
    // Orthogonal to emb_a: cosine = 0.0
    fn emb_b() -> Vec<f32> {
        vec![0.0, 1.0, 0.0]
    }

    // ── basic construction ────────────────────────────────────────────────────

    #[test]
    fn test_new_starts_empty() {
        let linker = default_linker();
        assert!(linker.entities.is_empty());
        assert!(linker.name_index.is_empty());
        assert!(linker.alias_index.is_empty());
        assert_eq!(linker.stats().total_entities, 0);
        assert_eq!(linker.stats().total_links, 0);
    }

    // ── register ─────────────────────────────────────────────────────────────

    #[test]
    fn test_register_stores_entity() {
        let mut linker = default_linker();
        linker.register(make_entity(1, "Alice", &[], emb_a(), "person"));
        assert_eq!(linker.entities.len(), 1);
        assert!(linker.entity(1).is_some());
    }

    #[test]
    fn test_register_indexes_canonical_name() {
        let mut linker = default_linker();
        linker.register(make_entity(1, "Alice", &[], emb_a(), "person"));
        assert!(linker.name_index.contains_key("alice"));
    }

    #[test]
    fn test_register_indexes_aliases() {
        let mut linker = default_linker();
        linker.register(make_entity(1, "Alice", &["Ali", "Al"], emb_a(), "person"));
        assert!(linker.alias_index.contains_key("ali"));
        assert!(linker.alias_index.contains_key("al"));
    }

    #[test]
    fn test_register_updates_total_entities_stat() {
        let mut linker = default_linker();
        linker.register(make_entity(1, "Alice", &[], emb_a(), "person"));
        linker.register(make_entity(2, "Bob", &[], emb_b(), "person"));
        assert_eq!(linker.stats().total_entities, 2);
    }

    // ── link: exact match ─────────────────────────────────────────────────────

    #[test]
    fn test_link_exact_match_same_case() {
        let mut linker = default_linker();
        linker.register(make_entity(1, "Alice", &[], emb_a(), "person"));
        let result = linker.link("Alice", None);
        assert!(result.is_some());
        let m = result.expect("test: exact match same case should link");
        assert_eq!(m.entity_id, 1);
        assert_eq!(m.kind, MentionKind::ExactMatch);
    }

    #[test]
    fn test_link_exact_match_case_insensitive() {
        let mut linker = default_linker();
        linker.register(make_entity(1, "Alice", &[], emb_a(), "person"));
        let result = linker.link("ALICE", None);
        assert!(result.is_some());
        assert_eq!(
            result
                .expect("test: exact match case insensitive should link")
                .kind,
            MentionKind::ExactMatch
        );
    }

    #[test]
    fn test_link_exact_match_confidence() {
        let mut linker = default_linker();
        linker.register(make_entity(1, "Alice", &[], emb_a(), "person"));
        let m = linker
            .link("Alice", None)
            .expect("test: exact match confidence should link");
        assert!((m.confidence - 1.0).abs() < 1e-6);
    }

    // ── link: alias match ─────────────────────────────────────────────────────

    #[test]
    fn test_link_alias_match() {
        let mut linker = default_linker();
        linker.register(make_entity(1, "Alice", &["Ali"], emb_a(), "person"));
        let result = linker.link("Ali", None);
        assert!(result.is_some());
        assert_eq!(
            result.expect("test: alias match should link").kind,
            MentionKind::Alias
        );
    }

    #[test]
    fn test_link_alias_case_insensitive() {
        let mut linker = default_linker();
        linker.register(make_entity(1, "Alice", &["Ali"], emb_a(), "person"));
        let result = linker.link("ALI", None);
        assert!(result.is_some());
        assert_eq!(
            result
                .expect("test: alias case insensitive should link")
                .kind,
            MentionKind::Alias
        );
    }

    #[test]
    fn test_link_alias_confidence() {
        let mut linker = default_linker();
        linker.register(make_entity(1, "Alice", &["Ali"], emb_a(), "person"));
        let m = linker
            .link("Ali", None)
            .expect("test: alias confidence should link");
        assert!((m.confidence - 0.95).abs() < 1e-6);
    }

    // ── link: embedding match ─────────────────────────────────────────────────

    #[test]
    fn test_link_embedding_match_above_threshold() {
        let mut linker = default_linker();
        linker.register(make_entity(1, "Alice", &[], emb_a(), "person"));
        // emb_a_close has cosine ≈ 0.894 with emb_a — above default threshold 0.80
        let result = linker.link("unknown", Some(&emb_a_close()));
        assert!(result.is_some());
        assert_eq!(
            result.expect("test: embedding match above threshold").kind,
            MentionKind::EmbeddingMatch
        );
    }

    #[test]
    fn test_link_embedding_match_below_threshold_returns_none() {
        let mut linker = default_linker();
        linker.register(make_entity(1, "Alice", &[], emb_a(), "person"));
        // emb_b is orthogonal to emb_a; cosine = 0.0 < threshold
        let result = linker.link("unknown", Some(&emb_b()));
        assert!(result.is_none());
    }

    #[test]
    fn test_link_embedding_confidence_equals_sim_times_scale() {
        let mut linker = default_linker();
        linker.register(make_entity(1, "Alice", &[], emb_a(), "person"));
        let query = emb_a_close();
        let expected_sim = cosine_sim(&query, &emb_a());
        let m = linker
            .link("unknown", Some(&query))
            .expect("test: embedding confidence link");
        let expected_conf = expected_sim * linker.config.embedding_match_confidence_scale;
        assert!((m.confidence - expected_conf).abs() < 1e-5);
    }

    // ── link: priority ────────────────────────────────────────────────────────

    #[test]
    fn test_link_priority_exact_before_alias() {
        let mut linker = default_linker();
        // Entity 1 has canonical name "Alice", entity 2 has alias "alice"
        linker.register(make_entity(1, "Alice", &[], emb_a(), "person"));
        linker.register(make_entity(2, "Bob", &["alice"], emb_b(), "person"));
        let m = linker
            .link("alice", None)
            .expect("test: exact beats alias priority");
        // ExactMatch should win over Alias
        assert_eq!(m.kind, MentionKind::ExactMatch);
        assert_eq!(m.entity_id, 1);
    }

    #[test]
    fn test_link_priority_alias_before_embedding() {
        let mut linker = default_linker();
        linker.register(make_entity(1, "Alice", &["mystery"], emb_b(), "person"));
        // "mystery" is an alias → must win over embedding match
        let result = linker.link("mystery", Some(&emb_a()));
        let m = result.expect("test: alias beats embedding priority");
        assert_eq!(m.kind, MentionKind::Alias);
        assert_eq!(m.entity_id, 1);
    }

    // ── link: no match ────────────────────────────────────────────────────────

    #[test]
    fn test_link_none_when_no_match() {
        let mut linker = default_linker();
        linker.register(make_entity(1, "Alice", &[], emb_a(), "person"));
        let result = linker.link("Completely Unknown Entity", None);
        assert!(result.is_none());
    }

    // ── stats counters ────────────────────────────────────────────────────────

    #[test]
    fn test_exact_match_count_increments() {
        let mut linker = default_linker();
        linker.register(make_entity(1, "Alice", &[], emb_a(), "person"));
        linker.link("Alice", None);
        linker.link("Alice", None);
        assert_eq!(linker.stats().exact_match_count, 2);
    }

    #[test]
    fn test_alias_match_count_increments() {
        let mut linker = default_linker();
        linker.register(make_entity(1, "Alice", &["Ali"], emb_a(), "person"));
        linker.link("Ali", None);
        assert_eq!(linker.stats().alias_match_count, 1);
    }

    #[test]
    fn test_embedding_match_count_increments() {
        let mut linker = default_linker();
        linker.register(make_entity(1, "Alice", &[], emb_a(), "person"));
        linker.link("unknown1", Some(&emb_a_close()));
        linker.link("unknown2", Some(&emb_a_close()));
        assert_eq!(linker.stats().embedding_match_count, 2);
    }

    #[test]
    fn test_unlinked_count_increments() {
        let mut linker = default_linker();
        linker.link("nobody", None);
        linker.link("nobody2", None);
        assert_eq!(linker.stats().unlinked_count, 2);
    }

    #[test]
    fn test_total_links_accumulates() {
        let mut linker = default_linker();
        linker.register(make_entity(1, "Alice", &["Ali"], emb_a(), "person"));
        linker.link("Alice", None); // exact
        linker.link("Ali", None); // alias
        linker.link("unknown", Some(&emb_a_close())); // embedding
        assert_eq!(linker.stats().total_links, 3);
    }

    #[test]
    fn test_linked_mention_kind_set_correctly() {
        let mut linker = default_linker();
        linker.register(make_entity(1, "Alice", &["Ali"], emb_a(), "person"));
        assert_eq!(
            linker
                .link("Alice", None)
                .expect("test: mention kind ExactMatch")
                .kind,
            MentionKind::ExactMatch
        );
        assert_eq!(
            linker
                .link("Ali", None)
                .expect("test: mention kind Alias")
                .kind,
            MentionKind::Alias
        );
    }

    // ── link_batch ────────────────────────────────────────────────────────────

    #[test]
    fn test_link_batch_processes_multiple_mentions() {
        let mut linker = default_linker();
        linker.register(make_entity(1, "Alice", &["Ali"], emb_a(), "person"));
        linker.register(make_entity(2, "Bob", &[], emb_b(), "person"));

        let mentions: Vec<(String, Option<Vec<f32>>)> = vec![
            ("Alice".to_owned(), None),
            ("Ali".to_owned(), None),
            ("nobody".to_owned(), None),
        ];
        let results = linker.link_batch(mentions);
        assert_eq!(results.len(), 3);
        assert!(results[0].is_some());
        assert!(results[1].is_some());
        assert!(results[2].is_none());
    }

    // ── entity() lookup ───────────────────────────────────────────────────────

    #[test]
    fn test_entity_some() {
        let mut linker = default_linker();
        linker.register(make_entity(42, "Alice", &[], emb_a(), "person"));
        assert!(linker.entity(42).is_some());
        assert_eq!(
            linker
                .entity(42)
                .expect("test: entity 42 should exist")
                .canonical_name,
            "Alice"
        );
    }

    #[test]
    fn test_entity_none() {
        let linker = default_linker();
        assert!(linker.entity(999).is_none());
    }

    // ── cosine_sim ────────────────────────────────────────────────────────────

    #[test]
    fn test_cosine_sim_zero_vector_returns_zero() {
        let zero = vec![0.0_f32, 0.0, 0.0];
        let a = vec![1.0_f32, 0.0, 0.0];
        assert_eq!(cosine_sim(&zero, &a), 0.0);
        assert_eq!(cosine_sim(&a, &zero), 0.0);
        assert_eq!(cosine_sim(&zero, &zero), 0.0);
    }

    #[test]
    fn test_cosine_sim_identical_vectors() {
        let a = vec![1.0_f32, 2.0, 3.0];
        let sim = cosine_sim(&a, &a);
        assert!(
            (sim - 1.0).abs() < 1e-6,
            "identical vectors → sim = 1.0, got {sim}"
        );
    }

    #[test]
    fn test_cosine_sim_orthogonal_vectors() {
        let a = vec![1.0_f32, 0.0, 0.0];
        let b = vec![0.0_f32, 1.0, 0.0];
        assert!((cosine_sim(&a, &b)).abs() < 1e-6);
    }
}
