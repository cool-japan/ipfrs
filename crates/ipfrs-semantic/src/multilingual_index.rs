// multilingual_index.rs — SemanticMultilingualIndex
//
// Indexes document embeddings organised by language, enabling cross-lingual
// semantic search via language-specific and language-agnostic retrieval.

use std::collections::HashMap;

// ── Language ─────────────────────────────────────────────────────────────────

/// ISO-639-1-aware language discriminant.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Language {
    English,
    Japanese,
    German,
    French,
    Spanish,
    /// Any other language identified by its ISO 639-1 code.
    Other {
        code: String,
    },
}

/// Return the canonical two-letter label used as map keys throughout this
/// module.  Kept private; callers should rely on [`Language`] values instead.
fn lang_label(lang: &Language) -> String {
    match lang {
        Language::English => "en".to_owned(),
        Language::Japanese => "ja".to_owned(),
        Language::German => "de".to_owned(),
        Language::French => "fr".to_owned(),
        Language::Spanish => "es".to_owned(),
        Language::Other { code } => code.clone(),
    }
}

// ── Cosine similarity ─────────────────────────────────────────────────────────

/// Cosine similarity in [−1, 1].  Returns 0.0 for empty slices, dimension
/// mismatches, or zero-norm vectors.
fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || a.len() != b.len() {
        return 0.0;
    }
    let mut dot = 0.0_f32;
    let mut mag_a = 0.0_f32;
    let mut mag_b = 0.0_f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        mag_a += x * x;
        mag_b += y * y;
    }
    if mag_a == 0.0 || mag_b == 0.0 {
        return 0.0;
    }
    dot / (mag_a.sqrt() * mag_b.sqrt())
}

// ── MultilingualDoc ───────────────────────────────────────────────────────────

/// A single document stored in the index.
#[derive(Clone, Debug)]
pub struct MultilingualDoc {
    pub doc_id: u64,
    pub language: Language,
    pub embedding: Vec<f32>,
    pub metadata: HashMap<String, String>,
}

// ── CrossLingualQuery ─────────────────────────────────────────────────────────

/// A query that can target a subset of languages (or all, when the vec is
/// empty).
#[derive(Clone, Debug)]
pub struct CrossLingualQuery {
    pub query_embedding: Vec<f32>,
    /// Languages to restrict retrieval to.  Empty → search all languages.
    pub target_languages: Vec<Language>,
    pub top_k: usize,
}

// ── MultilingualResult ────────────────────────────────────────────────────────

/// A single result returned by [`SemanticMultilingualIndex::search`].
#[derive(Clone, Debug)]
pub struct MultilingualResult {
    pub doc_id: u64,
    pub language: Language,
    pub similarity: f32,
    /// 0-indexed rank in the result list (0 = most similar).
    pub rank: usize,
}

// ── MultilingualIndexStats ────────────────────────────────────────────────────

/// Aggregate statistics exposed by the index.
#[derive(Clone, Debug, Default)]
pub struct MultilingualIndexStats {
    pub total_docs: usize,
    /// Keyed by language label string (e.g. `"en"`, `"ja"`).
    pub by_language: HashMap<String, usize>,
    pub total_searches: u64,
}

// ── SemanticMultilingualIndex ─────────────────────────────────────────────────

/// Embedding index organised by language that supports both monolingual and
/// cross-lingual cosine-similarity search.
#[derive(Debug)]
pub struct SemanticMultilingualIndex {
    /// Primary store: doc_id → document.
    docs: HashMap<u64, MultilingualDoc>,
    /// Inverted index: lang_label → sorted list of doc_ids.
    language_index: HashMap<String, Vec<u64>>,
    stats: MultilingualIndexStats,
}

impl SemanticMultilingualIndex {
    /// Create an empty index.
    pub fn new() -> Self {
        Self {
            docs: HashMap::new(),
            language_index: HashMap::new(),
            stats: MultilingualIndexStats::default(),
        }
    }

    /// Insert a document.  If a document with the same `doc_id` already
    /// exists it is replaced (old language-index entry is cleaned up first).
    pub fn add_doc(&mut self, doc: MultilingualDoc) {
        // If replacing, remove the stale language-index entry.
        if let Some(old) = self.docs.get(&doc.doc_id) {
            let old_label = lang_label(&old.language);
            if let Some(ids) = self.language_index.get_mut(&old_label) {
                ids.retain(|&id| id != doc.doc_id);
            }
            // Update by_language counter for old language.
            let cnt = self.stats.by_language.entry(old_label).or_insert(0);
            *cnt = cnt.saturating_sub(1);
            self.stats.total_docs = self.stats.total_docs.saturating_sub(1);
        }

        let label = lang_label(&doc.language);

        // Update language index.
        let ids = self.language_index.entry(label.clone()).or_default();
        ids.push(doc.doc_id);
        ids.sort_unstable();

        // Update stats.
        *self.stats.by_language.entry(label).or_insert(0) += 1;
        self.stats.total_docs += 1;

        self.docs.insert(doc.doc_id, doc);
    }

    /// Remove a document by `doc_id`.  Returns `true` if the document existed.
    pub fn remove_doc(&mut self, doc_id: u64) -> bool {
        let Some(doc) = self.docs.remove(&doc_id) else {
            return false;
        };
        let label = lang_label(&doc.language);

        // Remove from language index.
        if let Some(ids) = self.language_index.get_mut(&label) {
            ids.retain(|&id| id != doc_id);
        }

        // Update stats.
        let cnt = self.stats.by_language.entry(label).or_insert(0);
        *cnt = cnt.saturating_sub(1);
        self.stats.total_docs = self.stats.total_docs.saturating_sub(1);

        true
    }

    /// Cross-lingual cosine-similarity search.
    ///
    /// * `target_languages` empty → search all documents.
    /// * Results are sorted by similarity descending; ties broken by `doc_id`
    ///   ascending.
    /// * At most `top_k` results are returned, each with a 0-indexed `rank`.
    pub fn search(&mut self, query: &CrossLingualQuery) -> Vec<MultilingualResult> {
        self.stats.total_searches += 1;

        // Collect candidate doc_ids.
        let candidate_ids: Vec<u64> = if query.target_languages.is_empty() {
            self.docs.keys().copied().collect()
        } else {
            let mut ids = Vec::new();
            for lang in &query.target_languages {
                let label = lang_label(lang);
                if let Some(lang_ids) = self.language_index.get(&label) {
                    ids.extend_from_slice(lang_ids);
                }
            }
            ids.sort_unstable();
            ids.dedup();
            ids
        };

        // Score candidates.
        let mut scored: Vec<(u64, f32)> = candidate_ids
            .into_iter()
            .filter_map(|id| {
                let doc = self.docs.get(&id)?;
                let sim = cosine_sim(&query.query_embedding, &doc.embedding);
                Some((id, sim))
            })
            .collect();

        // Sort: similarity descending, then doc_id ascending for ties.
        scored.sort_unstable_by(|(id_a, sim_a), (id_b, sim_b)| {
            sim_b
                .partial_cmp(sim_a)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| id_a.cmp(id_b))
        });

        // Truncate to top_k and assign ranks.
        scored.truncate(query.top_k);

        scored
            .into_iter()
            .enumerate()
            .filter_map(|(rank, (doc_id, similarity))| {
                let doc = self.docs.get(&doc_id)?;
                Some(MultilingualResult {
                    doc_id,
                    language: doc.language.clone(),
                    similarity,
                    rank,
                })
            })
            .collect()
    }

    /// Return all `doc_id`s for the given language, sorted ascending.
    pub fn docs_for_language(&self, language: &Language) -> Vec<u64> {
        let label = lang_label(language);
        match self.language_index.get(&label) {
            Some(ids) => {
                let mut sorted = ids.clone();
                sorted.sort_unstable();
                sorted
            }
            None => Vec::new(),
        }
    }

    /// Borrow the current statistics snapshot.
    pub fn stats(&self) -> &MultilingualIndexStats {
        &self.stats
    }
}

impl Default for SemanticMultilingualIndex {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_doc(doc_id: u64, language: Language, embedding: Vec<f32>) -> MultilingualDoc {
        MultilingualDoc {
            doc_id,
            language,
            embedding,
            metadata: HashMap::new(),
        }
    }

    fn make_doc_meta(
        doc_id: u64,
        language: Language,
        embedding: Vec<f32>,
        meta: &[(&str, &str)],
    ) -> MultilingualDoc {
        let metadata = meta
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        MultilingualDoc {
            doc_id,
            language,
            embedding,
            metadata,
        }
    }

    // ── add_doc ───────────────────────────────────────────────────────────────

    #[test]
    fn test_add_doc_inserts_into_docs_map() {
        let mut idx = SemanticMultilingualIndex::new();
        idx.add_doc(make_doc(1, Language::English, vec![1.0, 0.0]));
        assert!(idx.docs.contains_key(&1));
    }

    #[test]
    fn test_add_doc_inserts_into_language_index() {
        let mut idx = SemanticMultilingualIndex::new();
        idx.add_doc(make_doc(42, Language::Japanese, vec![0.0, 1.0]));
        let ids = idx.docs_for_language(&Language::Japanese);
        assert_eq!(ids, vec![42]);
    }

    #[test]
    fn test_add_doc_updates_total_docs_stat() {
        let mut idx = SemanticMultilingualIndex::new();
        idx.add_doc(make_doc(1, Language::German, vec![1.0]));
        idx.add_doc(make_doc(2, Language::German, vec![0.0]));
        assert_eq!(idx.stats().total_docs, 2);
    }

    #[test]
    fn test_add_doc_updates_by_language_stat() {
        let mut idx = SemanticMultilingualIndex::new();
        idx.add_doc(make_doc(1, Language::French, vec![1.0]));
        idx.add_doc(make_doc(2, Language::French, vec![0.0]));
        assert_eq!(idx.stats().by_language.get("fr").copied().unwrap_or(0), 2);
    }

    #[test]
    fn test_add_doc_replace_updates_language_index() {
        let mut idx = SemanticMultilingualIndex::new();
        idx.add_doc(make_doc(1, Language::English, vec![1.0, 0.0]));
        // Replace doc 1 with a different language.
        idx.add_doc(make_doc(1, Language::Spanish, vec![0.0, 1.0]));
        // Must not appear under English any more.
        assert!(!idx.docs_for_language(&Language::English).contains(&1));
        // Must appear under Spanish.
        assert!(idx.docs_for_language(&Language::Spanish).contains(&1));
        // Total docs must still be 1.
        assert_eq!(idx.stats().total_docs, 1);
    }

    // ── remove_doc ────────────────────────────────────────────────────────────

    #[test]
    fn test_remove_doc_returns_true_when_exists() {
        let mut idx = SemanticMultilingualIndex::new();
        idx.add_doc(make_doc(10, Language::English, vec![1.0]));
        assert!(idx.remove_doc(10));
    }

    #[test]
    fn test_remove_doc_returns_false_when_missing() {
        let mut idx = SemanticMultilingualIndex::new();
        assert!(!idx.remove_doc(999));
    }

    #[test]
    fn test_remove_doc_clears_docs_map_entry() {
        let mut idx = SemanticMultilingualIndex::new();
        idx.add_doc(make_doc(5, Language::English, vec![1.0]));
        idx.remove_doc(5);
        assert!(!idx.docs.contains_key(&5));
    }

    #[test]
    fn test_remove_doc_clears_language_index_entry() {
        let mut idx = SemanticMultilingualIndex::new();
        idx.add_doc(make_doc(7, Language::Japanese, vec![1.0]));
        idx.remove_doc(7);
        assert!(idx.docs_for_language(&Language::Japanese).is_empty());
    }

    #[test]
    fn test_remove_doc_updates_stats() {
        let mut idx = SemanticMultilingualIndex::new();
        idx.add_doc(make_doc(1, Language::German, vec![1.0]));
        idx.add_doc(make_doc(2, Language::German, vec![0.5]));
        idx.remove_doc(1);
        assert_eq!(idx.stats().total_docs, 1);
        assert_eq!(idx.stats().by_language.get("de").copied().unwrap_or(0), 1);
    }

    // ── search — all languages ────────────────────────────────────────────────

    #[test]
    fn test_search_all_languages_when_target_empty() {
        let mut idx = SemanticMultilingualIndex::new();
        idx.add_doc(make_doc(1, Language::English, vec![1.0, 0.0]));
        idx.add_doc(make_doc(2, Language::Japanese, vec![0.0, 1.0]));
        let q = CrossLingualQuery {
            query_embedding: vec![1.0, 0.0],
            target_languages: vec![],
            top_k: 10,
        };
        let results = idx.search(&q);
        let ids: Vec<u64> = results.iter().map(|r| r.doc_id).collect();
        assert!(ids.contains(&1));
        assert!(ids.contains(&2));
    }

    #[test]
    fn test_search_empty_index_returns_empty() {
        let mut idx = SemanticMultilingualIndex::new();
        let q = CrossLingualQuery {
            query_embedding: vec![1.0, 0.0],
            target_languages: vec![],
            top_k: 5,
        };
        assert!(idx.search(&q).is_empty());
    }

    // ── search — language filter ──────────────────────────────────────────────

    #[test]
    fn test_search_filtered_by_target_languages() {
        let mut idx = SemanticMultilingualIndex::new();
        idx.add_doc(make_doc(1, Language::English, vec![1.0, 0.0]));
        idx.add_doc(make_doc(2, Language::French, vec![1.0, 0.0]));
        let q = CrossLingualQuery {
            query_embedding: vec![1.0, 0.0],
            target_languages: vec![Language::English],
            top_k: 10,
        };
        let results = idx.search(&q);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].doc_id, 1);
    }

    #[test]
    fn test_search_multiple_target_languages() {
        let mut idx = SemanticMultilingualIndex::new();
        idx.add_doc(make_doc(1, Language::English, vec![1.0, 0.0]));
        idx.add_doc(make_doc(2, Language::German, vec![1.0, 0.0]));
        idx.add_doc(make_doc(3, Language::Spanish, vec![1.0, 0.0]));
        let q = CrossLingualQuery {
            query_embedding: vec![1.0, 0.0],
            target_languages: vec![Language::English, Language::German],
            top_k: 10,
        };
        let results = idx.search(&q);
        let ids: Vec<u64> = results.iter().map(|r| r.doc_id).collect();
        assert!(ids.contains(&1));
        assert!(ids.contains(&2));
        assert!(!ids.contains(&3));
    }

    // ── search — top_k truncation ─────────────────────────────────────────────

    #[test]
    fn test_search_top_k_truncation() {
        let mut idx = SemanticMultilingualIndex::new();
        for id in 0..10_u64 {
            idx.add_doc(make_doc(id, Language::English, vec![1.0, 0.0]));
        }
        let q = CrossLingualQuery {
            query_embedding: vec![1.0, 0.0],
            target_languages: vec![],
            top_k: 3,
        };
        let results = idx.search(&q);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_search_top_k_zero_returns_empty() {
        let mut idx = SemanticMultilingualIndex::new();
        idx.add_doc(make_doc(1, Language::English, vec![1.0]));
        let q = CrossLingualQuery {
            query_embedding: vec![1.0],
            target_languages: vec![],
            top_k: 0,
        };
        assert!(idx.search(&q).is_empty());
    }

    // ── search — rank assignment ──────────────────────────────────────────────

    #[test]
    fn test_search_rank_assigned_correctly() {
        let mut idx = SemanticMultilingualIndex::new();
        // doc 1 is more similar to query [1,0] than doc 2
        idx.add_doc(make_doc(1, Language::English, vec![1.0, 0.0]));
        idx.add_doc(make_doc(2, Language::English, vec![0.0, 1.0]));
        let q = CrossLingualQuery {
            query_embedding: vec![1.0, 0.0],
            target_languages: vec![],
            top_k: 5,
        };
        let results = idx.search(&q);
        assert_eq!(results.len(), 2);
        // Most similar doc should have rank 0.
        assert_eq!(results[0].rank, 0);
        assert_eq!(results[1].rank, 1);
        // Most similar to [1,0] is doc 1.
        assert_eq!(results[0].doc_id, 1);
    }

    // ── search — cosine similarity ordering ──────────────────────────────────

    #[test]
    fn test_search_cosine_similarity_ordering() {
        let mut idx = SemanticMultilingualIndex::new();
        // Similarities to [1, 0, 0]:
        //  doc 1: [1, 0, 0] → 1.0
        //  doc 2: [0.7, 0.7, 0] → cos ≈ 0.707
        //  doc 3: [0, 1, 0] → 0.0
        idx.add_doc(make_doc(1, Language::English, vec![1.0, 0.0, 0.0]));
        idx.add_doc(make_doc(2, Language::English, vec![0.707, 0.707, 0.0]));
        idx.add_doc(make_doc(3, Language::English, vec![0.0, 1.0, 0.0]));
        let q = CrossLingualQuery {
            query_embedding: vec![1.0, 0.0, 0.0],
            target_languages: vec![],
            top_k: 3,
        };
        let results = idx.search(&q);
        assert_eq!(results[0].doc_id, 1);
        assert_eq!(results[1].doc_id, 2);
        assert_eq!(results[2].doc_id, 3);
        assert!(results[0].similarity > results[1].similarity);
        assert!(results[1].similarity > results[2].similarity);
    }

    // ── search — doc_id tie-breaking ─────────────────────────────────────────

    #[test]
    fn test_search_doc_id_tie_breaking() {
        let mut idx = SemanticMultilingualIndex::new();
        // All docs have equal cosine similarity to [1,0].
        idx.add_doc(make_doc(30, Language::English, vec![1.0, 0.0]));
        idx.add_doc(make_doc(10, Language::English, vec![1.0, 0.0]));
        idx.add_doc(make_doc(20, Language::English, vec![1.0, 0.0]));
        let q = CrossLingualQuery {
            query_embedding: vec![1.0, 0.0],
            target_languages: vec![],
            top_k: 3,
        };
        let results = idx.search(&q);
        let ids: Vec<u64> = results.iter().map(|r| r.doc_id).collect();
        // Should be sorted ascending by doc_id.
        assert_eq!(ids, vec![10, 20, 30]);
    }

    // ── docs_for_language ─────────────────────────────────────────────────────

    #[test]
    fn test_docs_for_language_sorted_ascending() {
        let mut idx = SemanticMultilingualIndex::new();
        idx.add_doc(make_doc(50, Language::Spanish, vec![1.0]));
        idx.add_doc(make_doc(10, Language::Spanish, vec![0.5]));
        idx.add_doc(make_doc(30, Language::Spanish, vec![0.8]));
        let ids = idx.docs_for_language(&Language::Spanish);
        assert_eq!(ids, vec![10, 30, 50]);
    }

    #[test]
    fn test_docs_for_language_empty_when_none() {
        let idx = SemanticMultilingualIndex::new();
        assert!(idx.docs_for_language(&Language::French).is_empty());
    }

    // ── by_language stats ─────────────────────────────────────────────────────

    #[test]
    fn test_by_language_stats_updated_on_add() {
        let mut idx = SemanticMultilingualIndex::new();
        idx.add_doc(make_doc(1, Language::English, vec![1.0]));
        idx.add_doc(make_doc(2, Language::English, vec![0.0]));
        idx.add_doc(make_doc(3, Language::German, vec![1.0]));
        let stats = idx.stats();
        assert_eq!(stats.by_language.get("en").copied().unwrap_or(0), 2);
        assert_eq!(stats.by_language.get("de").copied().unwrap_or(0), 1);
    }

    #[test]
    fn test_by_language_stats_updated_on_remove() {
        let mut idx = SemanticMultilingualIndex::new();
        idx.add_doc(make_doc(1, Language::English, vec![1.0]));
        idx.add_doc(make_doc(2, Language::English, vec![0.0]));
        idx.remove_doc(1);
        assert_eq!(idx.stats().by_language.get("en").copied().unwrap_or(0), 1);
    }

    // ── Other language code handling ──────────────────────────────────────────

    #[test]
    fn test_other_language_code_handling() {
        let lang = Language::Other {
            code: "zh".to_owned(),
        };
        let mut idx = SemanticMultilingualIndex::new();
        idx.add_doc(make_doc(1, lang.clone(), vec![1.0, 0.0]));
        let ids = idx.docs_for_language(&lang);
        assert_eq!(ids, vec![1]);
        assert_eq!(idx.stats().by_language.get("zh").copied().unwrap_or(0), 1);
    }

    #[test]
    fn test_other_language_code_in_search_filter() {
        let lang = Language::Other {
            code: "ko".to_owned(),
        };
        let mut idx = SemanticMultilingualIndex::new();
        idx.add_doc(make_doc(1, lang.clone(), vec![1.0, 0.0]));
        idx.add_doc(make_doc(2, Language::English, vec![1.0, 0.0]));
        let q = CrossLingualQuery {
            query_embedding: vec![1.0, 0.0],
            target_languages: vec![lang],
            top_k: 10,
        };
        let results = idx.search(&q);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].doc_id, 1);
    }

    // ── stats.total_searches ──────────────────────────────────────────────────

    #[test]
    fn test_total_searches_increments() {
        let mut idx = SemanticMultilingualIndex::new();
        let q = CrossLingualQuery {
            query_embedding: vec![1.0],
            target_languages: vec![],
            top_k: 5,
        };
        idx.search(&q);
        idx.search(&q);
        assert_eq!(idx.stats().total_searches, 2);
    }

    // ── metadata round-trip ───────────────────────────────────────────────────

    #[test]
    fn test_metadata_preserved() {
        let mut idx = SemanticMultilingualIndex::new();
        let doc = make_doc_meta(
            99,
            Language::English,
            vec![1.0],
            &[("author", "alice"), ("year", "2024")],
        );
        idx.add_doc(doc);
        let stored = idx.docs.get(&99).expect("doc must exist");
        assert_eq!(
            stored.metadata.get("author").map(String::as_str),
            Some("alice")
        );
        assert_eq!(
            stored.metadata.get("year").map(String::as_str),
            Some("2024")
        );
    }

    // ── language field on result ──────────────────────────────────────────────

    #[test]
    fn test_result_language_matches_doc() {
        let mut idx = SemanticMultilingualIndex::new();
        idx.add_doc(make_doc(1, Language::Japanese, vec![1.0, 0.0]));
        let q = CrossLingualQuery {
            query_embedding: vec![1.0, 0.0],
            target_languages: vec![],
            top_k: 5,
        };
        let results = idx.search(&q);
        assert_eq!(results[0].language, Language::Japanese);
    }

    // ── zero-norm vector returns 0.0 similarity ───────────────────────────────

    #[test]
    fn test_zero_norm_embedding_gives_zero_similarity() {
        let mut idx = SemanticMultilingualIndex::new();
        idx.add_doc(make_doc(1, Language::English, vec![0.0, 0.0]));
        let q = CrossLingualQuery {
            query_embedding: vec![1.0, 0.0],
            target_languages: vec![],
            top_k: 5,
        };
        let results = idx.search(&q);
        assert_eq!(results[0].similarity, 0.0);
    }

    // ── dimension mismatch returns 0.0 ────────────────────────────────────────

    #[test]
    fn test_dimension_mismatch_gives_zero_similarity() {
        let mut idx = SemanticMultilingualIndex::new();
        // 3-D doc, 2-D query.
        idx.add_doc(make_doc(1, Language::English, vec![1.0, 0.0, 0.0]));
        let q = CrossLingualQuery {
            query_embedding: vec![1.0, 0.0],
            target_languages: vec![],
            top_k: 5,
        };
        let results = idx.search(&q);
        assert_eq!(results[0].similarity, 0.0);
    }
}
