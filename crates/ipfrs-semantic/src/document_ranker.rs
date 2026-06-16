//! Multi-factor document ranking combining BM25 lexical scoring with semantic similarity.
//!
//! This module provides a [`DocumentRanker`] that fuses traditional BM25 term-based scoring
//! with dense-vector cosine similarity to produce a single combined relevance score for
//! document retrieval.
//!
//! ## Algorithm Overview
//!
//! For each document `d` and query `q`:
//!
//! ```text
//! combined(d, q) = lexical_weight * BM25(d, q) + semantic_weight * cosine(emb_d, emb_q)
//! ```
//!
//! BM25 per-term contribution:
//!
//! ```text
//! idf(t) * tf(t,d)*(k1+1) / (tf(t,d) + k1*(1 - b + b*|d|/avgdl))
//! ```
//!
//! IDF formula (Robertson–Sparck Jones with smoothing):
//!
//! ```text
//! idf(t) = ln((N - df + 0.5) / (df + 0.5) + 1)
//! ```

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the [`DocumentRanker`].
#[derive(Debug, Clone)]
pub struct RankingConfig {
    /// BM25 term-frequency saturation constant (default 1.5).
    pub bm25_k1: f64,
    /// BM25 length-normalisation constant (default 0.75).
    pub bm25_b: f64,
    /// Weight applied to the semantic (cosine) score in [0, 1].
    pub semantic_weight: f64,
    /// Weight applied to the BM25 lexical score in [0, 1].
    pub lexical_weight: f64,
    /// Maximum number of results to return.
    pub max_results: usize,
    /// Minimum combined score threshold; documents below this are dropped.
    pub min_score: f64,
}

impl Default for RankingConfig {
    fn default() -> Self {
        Self {
            bm25_k1: 1.5,
            bm25_b: 0.75,
            semantic_weight: 0.5,
            lexical_weight: 0.5,
            max_results: 10,
            min_score: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// DocumentIndex
// ---------------------------------------------------------------------------

/// A document representation stored inside the ranker index.
#[derive(Debug, Clone)]
pub struct DocumentIndex {
    /// Unique document identifier.
    pub doc_id: String,
    /// Pre-computed term frequency map: term → raw count (normalised to `f64`).
    pub term_frequencies: HashMap<String, f64>,
    /// Total number of tokens in the document.
    pub doc_length: usize,
    /// Optional dense embedding used for semantic scoring.
    pub embedding: Option<Vec<f64>>,
}

impl DocumentIndex {
    /// Constructs a [`DocumentIndex`] from a plain token list.
    ///
    /// Term frequencies are computed as raw counts; the caller may pass a
    /// pre-embedded vector if semantic ranking is desired.
    pub fn from_tokens(
        doc_id: impl Into<String>,
        tokens: &[&str],
        embedding: Option<Vec<f64>>,
    ) -> Self {
        let mut term_frequencies: HashMap<String, f64> = HashMap::new();
        for &tok in tokens {
            let entry = term_frequencies.entry(tok.to_lowercase()).or_insert(0.0);
            *entry += 1.0;
        }
        let doc_length = tokens.len();
        Self {
            doc_id: doc_id.into(),
            term_frequencies,
            doc_length,
            embedding,
        }
    }
}

// ---------------------------------------------------------------------------
// RankedDocument
// ---------------------------------------------------------------------------

/// A scored document returned by [`DocumentRanker::rank`].
#[derive(Debug, Clone)]
pub struct RankedDocument {
    /// Document identifier.
    pub doc_id: String,
    /// Raw BM25 lexical score (un-weighted).
    pub bm25_score: f64,
    /// Raw cosine semantic score in \[0, 1\] (un-weighted), or 0.0 if unavailable.
    pub semantic_score: f64,
    /// Weighted combined score: `lexical_weight*bm25 + semantic_weight*cosine`.
    pub combined_score: f64,
    /// 1-based rank position in the result list.
    pub rank: usize,
}

// ---------------------------------------------------------------------------
// RankerStats
// ---------------------------------------------------------------------------

/// Aggregate statistics collected by [`DocumentRanker`] across all queries.
#[derive(Debug, Clone, Default)]
pub struct RankerStats {
    /// Total number of `rank()` calls executed.
    pub total_queries: u64,
    /// Total number of documents that appeared in at least one result set.
    pub documents_ranked: u64,
    /// Rolling average of the result-set size across all queries.
    pub avg_results_per_query: f64,
}

// ---------------------------------------------------------------------------
// DocumentRanker
// ---------------------------------------------------------------------------

/// Multi-factor document ranker combining BM25 lexical scoring with semantic similarity.
///
/// # Usage
///
/// ```rust
/// use ipfrs_semantic::document_ranker::{DocumentRanker, RankingConfig, DocumentIndex};
///
/// let config = RankingConfig::default();
/// let mut ranker = DocumentRanker::new(config);
///
/// let doc = DocumentIndex::from_tokens("doc1", &["hello", "world"], None);
/// ranker.index_document(doc);
///
/// let results = ranker.rank(&["hello".to_string()], None);
/// assert!(!results.is_empty());
/// ```
pub struct DocumentRanker {
    config: RankingConfig,
    documents: HashMap<String, DocumentIndex>,
    avg_doc_length: f64,
    idf_cache: HashMap<String, f64>,
    stats: RankerStats,
}

impl DocumentRanker {
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    /// Creates a new [`DocumentRanker`] with the given configuration.
    pub fn new(config: RankingConfig) -> Self {
        Self {
            config,
            documents: HashMap::new(),
            avg_doc_length: 0.0,
            idf_cache: HashMap::new(),
            stats: RankerStats::default(),
        }
    }

    // -----------------------------------------------------------------------
    // Index management
    // -----------------------------------------------------------------------

    /// Indexes (or re-indexes) a document.
    ///
    /// Inserting a document with the same `doc_id` as an existing one will
    /// overwrite the previous entry.  After insertion the average document
    /// length and IDF cache are refreshed for all terms present in the new
    /// document.
    pub fn index_document(&mut self, doc: DocumentIndex) {
        let terms: Vec<String> = doc.term_frequencies.keys().cloned().collect();
        self.documents.insert(doc.doc_id.clone(), doc);
        self.update_avg_length();
        self.update_idf_cache(&terms);
    }

    // -----------------------------------------------------------------------
    // Core ranking
    // -----------------------------------------------------------------------

    /// Ranks all indexed documents against the given query terms and optional
    /// query embedding.
    ///
    /// Results are filtered by [`RankingConfig::min_score`], sorted by
    /// descending combined score, and truncated to at most
    /// [`RankingConfig::max_results`] entries.  Each returned [`RankedDocument`]
    /// carries a 1-based `rank` field.
    pub fn rank(
        &mut self,
        query_terms: &[String],
        query_embedding: Option<&[f64]>,
    ) -> Vec<RankedDocument> {
        // Ensure IDF cache is populated for all query terms.
        self.update_idf_cache(query_terms);

        let mut scored: Vec<RankedDocument> = self
            .documents
            .values()
            .map(|doc| {
                let bm25 = self.bm25_score(doc, query_terms);
                let sem = match (query_embedding, doc.embedding.as_deref()) {
                    (Some(qe), Some(de)) => Self::cosine_similarity(qe, de),
                    _ => 0.0,
                };
                let combined =
                    self.config.lexical_weight * bm25 + self.config.semantic_weight * sem;
                RankedDocument {
                    doc_id: doc.doc_id.clone(),
                    bm25_score: bm25,
                    semantic_score: sem,
                    combined_score: combined,
                    rank: 0, // filled in below
                }
            })
            .filter(|rd| rd.combined_score >= self.config.min_score)
            .collect();

        // Sort descending by combined score; break ties alphabetically by doc_id.
        scored.sort_unstable_by(|a, b| {
            b.combined_score
                .partial_cmp(&a.combined_score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.doc_id.cmp(&b.doc_id))
        });

        scored.truncate(self.config.max_results);

        // Assign 1-based ranks.
        for (i, rd) in scored.iter_mut().enumerate() {
            rd.rank = i + 1;
        }

        // Update stats.
        let result_count = scored.len() as u64;
        self.stats.total_queries += 1;
        self.stats.documents_ranked += result_count;
        let n = self.stats.total_queries as f64;
        self.stats.avg_results_per_query =
            (self.stats.avg_results_per_query * (n - 1.0) + result_count as f64) / n;

        scored
    }

    // -----------------------------------------------------------------------
    // BM25
    // -----------------------------------------------------------------------

    /// Computes the BM25 score for a single document given the query terms.
    ///
    /// Uses the Robertson–Sparck Jones IDF with BM25+ numerator adjustment.
    pub fn bm25_score(&self, doc: &DocumentIndex, query_terms: &[String]) -> f64 {
        let k1 = self.config.bm25_k1;
        let b = self.config.bm25_b;
        let avgdl = self.avg_doc_length.max(1.0);
        let dl = doc.doc_length as f64;

        query_terms.iter().fold(0.0_f64, |acc, term| {
            let tf = doc
                .term_frequencies
                .get(term.as_str())
                .copied()
                .unwrap_or(0.0);
            if tf == 0.0 {
                return acc;
            }
            let idf = self
                .idf_cache
                .get(term.as_str())
                .copied()
                .unwrap_or_else(|| self.compute_idf(term));
            let numerator = tf * (k1 + 1.0);
            let denominator = tf + k1 * (1.0 - b + b * dl / avgdl);
            acc + idf * numerator / denominator
        })
    }

    /// Computes the IDF of a term using Robertson–Sparck Jones smoothed formula:
    ///
    /// ```text
    /// ln((N - df + 0.5) / (df + 0.5) + 1)
    /// ```
    ///
    /// where `N` is the total number of indexed documents and `df` is the
    /// document frequency of `term`.
    pub fn compute_idf(&self, term: &str) -> f64 {
        let n = self.documents.len() as f64;
        let df = self
            .documents
            .values()
            .filter(|doc| doc.term_frequencies.contains_key(term))
            .count() as f64;
        ((n - df + 0.5) / (df + 0.5) + 1.0).ln()
    }

    // -----------------------------------------------------------------------
    // Semantic similarity
    // -----------------------------------------------------------------------

    /// Computes the cosine similarity between two embedding vectors.
    ///
    /// Returns 0.0 when either vector is zero-length or the lengths differ.
    pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
        if a.len() != b.len() || a.is_empty() {
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

    // -----------------------------------------------------------------------
    // Index maintenance helpers
    // -----------------------------------------------------------------------

    /// Recomputes the average document length across all indexed documents.
    ///
    /// Called automatically after each [`index_document`](Self::index_document).
    pub fn update_avg_length(&mut self) {
        if self.documents.is_empty() {
            self.avg_doc_length = 0.0;
            return;
        }
        let total: usize = self.documents.values().map(|d| d.doc_length).sum();
        self.avg_doc_length = total as f64 / self.documents.len() as f64;
    }

    /// Refreshes the IDF cache for the given term list.
    ///
    /// Existing cache entries for terms *not* in `terms` are preserved.
    pub fn update_idf_cache(&mut self, terms: &[String]) {
        for term in terms {
            let idf = self.compute_idf(term);
            self.idf_cache.insert(term.clone(), idf);
        }
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /// Returns the number of documents currently in the index.
    pub fn document_count(&self) -> usize {
        self.documents.len()
    }

    /// Returns a reference to the accumulated query statistics.
    pub fn stats(&self) -> &RankerStats {
        &self.stats
    }

    /// Returns the current average document length used by BM25.
    pub fn avg_doc_length(&self) -> f64 {
        self.avg_doc_length
    }

    /// Returns a reference to a specific indexed document, if present.
    pub fn get_document(&self, doc_id: &str) -> Option<&DocumentIndex> {
        self.documents.get(doc_id)
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

    fn make_ranker() -> DocumentRanker {
        DocumentRanker::new(RankingConfig::default())
    }

    fn simple_doc(id: &str, tokens: &[&str]) -> DocumentIndex {
        DocumentIndex::from_tokens(id, tokens, None)
    }

    fn embed_doc(id: &str, tokens: &[&str], emb: Vec<f64>) -> DocumentIndex {
        DocumentIndex::from_tokens(id, tokens, Some(emb))
    }

    // -----------------------------------------------------------------------
    // 1. Index and rank single doc
    // -----------------------------------------------------------------------
    #[test]
    fn test_single_doc_indexed_and_ranked() {
        let mut ranker = make_ranker();
        ranker.index_document(simple_doc("d1", &["hello", "world"]));
        let results = ranker.rank(&["hello".to_string()], None);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].doc_id, "d1");
        assert_eq!(results[0].rank, 1);
    }

    // -----------------------------------------------------------------------
    // 2. BM25 term saturation: doubling TF should not double score
    // -----------------------------------------------------------------------
    #[test]
    fn test_bm25_term_saturation() {
        let config = RankingConfig {
            lexical_weight: 1.0,
            semantic_weight: 0.0,
            ..RankingConfig::default()
        };
        let mut ranker = DocumentRanker::new(config);
        // Doc A: "rust" appears once; Doc B: "rust" appears many times.
        ranker.index_document(simple_doc("sparse", &["rust"]));
        ranker.index_document(simple_doc(
            "dense",
            &[
                "rust", "rust", "rust", "rust", "rust", "rust", "rust", "rust", "rust", "rust",
            ],
        ));
        let results = ranker.rank(&["rust".to_string()], None);
        assert_eq!(results.len(), 2);
        let sparse_score = results
            .iter()
            .find(|r| r.doc_id == "sparse")
            .map(|r| r.bm25_score)
            .unwrap_or(0.0);
        let dense_score = results
            .iter()
            .find(|r| r.doc_id == "dense")
            .map(|r| r.bm25_score)
            .unwrap_or(0.0);
        // Dense should score higher but not proportionally more (saturation).
        assert!(
            dense_score > sparse_score,
            "dense={dense_score}, sparse={sparse_score}"
        );
        assert!(
            dense_score < sparse_score * 10.0,
            "no saturation? dense={dense_score}, sparse={sparse_score}"
        );
    }

    // -----------------------------------------------------------------------
    // 3. Length normalisation: shorter docs score higher for same TF
    // -----------------------------------------------------------------------
    #[test]
    fn test_bm25_length_normalisation() {
        let config = RankingConfig {
            lexical_weight: 1.0,
            semantic_weight: 0.0,
            ..RankingConfig::default()
        };
        let mut ranker = DocumentRanker::new(config);
        // Short doc: "rust" in a 2-token document.
        // Long doc: "rust" buried among many other tokens.
        let long_tokens: Vec<&str> = std::iter::once("rust")
            .chain(std::iter::repeat_n("filler", 49))
            .collect();
        ranker.index_document(simple_doc("short", &["rust", "code"]));
        ranker.index_document(simple_doc("long", &long_tokens));
        let results = ranker.rank(&["rust".to_string()], None);
        let short_score = results
            .iter()
            .find(|r| r.doc_id == "short")
            .map(|r| r.bm25_score)
            .unwrap_or(0.0);
        let long_score = results
            .iter()
            .find(|r| r.doc_id == "long")
            .map(|r| r.bm25_score)
            .unwrap_or(0.0);
        assert!(
            short_score > long_score,
            "short={short_score}, long={long_score}"
        );
    }

    // -----------------------------------------------------------------------
    // 4. IDF computation — rare term gets higher IDF
    // -----------------------------------------------------------------------
    #[test]
    fn test_idf_rare_term_higher() {
        let mut ranker = make_ranker();
        // "common" appears in all 3 docs; "rare" only in 1.
        ranker.index_document(simple_doc("d1", &["common", "rare"]));
        ranker.index_document(simple_doc("d2", &["common"]));
        ranker.index_document(simple_doc("d3", &["common"]));
        let idf_common = ranker.compute_idf("common");
        let idf_rare = ranker.compute_idf("rare");
        assert!(
            idf_rare > idf_common,
            "rare={idf_rare}, common={idf_common}"
        );
    }

    // -----------------------------------------------------------------------
    // 5. IDF is positive for all cases
    // -----------------------------------------------------------------------
    #[test]
    fn test_idf_always_positive() {
        let mut ranker = make_ranker();
        ranker.index_document(simple_doc("d1", &["alpha", "beta"]));
        ranker.index_document(simple_doc("d2", &["alpha", "gamma"]));
        for term in &["alpha", "beta", "gamma", "unseen"] {
            let idf = ranker.compute_idf(term);
            assert!(idf >= 0.0, "negative IDF for '{term}': {idf}");
        }
    }

    // -----------------------------------------------------------------------
    // 6. Semantic ranking — embedding alone selects correct doc
    // -----------------------------------------------------------------------
    #[test]
    fn test_semantic_ranking_selects_closest() {
        let config = RankingConfig {
            lexical_weight: 0.0,
            semantic_weight: 1.0,
            ..RankingConfig::default()
        };
        let mut ranker = DocumentRanker::new(config);
        ranker.index_document(embed_doc("near", &[], vec![1.0, 0.0, 0.0]));
        ranker.index_document(embed_doc("far", &[], vec![0.0, 1.0, 0.0]));
        let query_emb = vec![1.0, 0.0, 0.0];
        let results = ranker.rank(&[], Some(&query_emb));
        assert!(!results.is_empty());
        assert_eq!(results[0].doc_id, "near");
    }

    // -----------------------------------------------------------------------
    // 7. Combined score weighting (50/50 split)
    // -----------------------------------------------------------------------
    #[test]
    fn test_combined_score_weighting() {
        let config = RankingConfig {
            lexical_weight: 0.5,
            semantic_weight: 0.5,
            ..RankingConfig::default()
        };
        let mut ranker = DocumentRanker::new(config);
        // doc_a: perfect semantic match, no lexical match.
        ranker.index_document(embed_doc("doc_a", &["foo"], vec![1.0, 0.0]));
        // doc_b: exact lexical match, poor semantic match.
        ranker.index_document(embed_doc("doc_b", &["rust"], vec![0.0, 1.0]));
        let query_emb = vec![1.0, 0.0];
        let results = ranker.rank(&["rust".to_string()], Some(&query_emb));
        let a = results
            .iter()
            .find(|r| r.doc_id == "doc_a")
            .expect("doc_a missing");
        let b = results
            .iter()
            .find(|r| r.doc_id == "doc_b")
            .expect("doc_b missing");
        // doc_a should have higher semantic contribution.
        assert!(a.semantic_score > b.semantic_score);
        // doc_b should have higher BM25 contribution.
        assert!(b.bm25_score > a.bm25_score);
    }

    // -----------------------------------------------------------------------
    // 8. max_results limits output
    // -----------------------------------------------------------------------
    #[test]
    fn test_max_results_limits_output() {
        let config = RankingConfig {
            max_results: 3,
            ..RankingConfig::default()
        };
        let mut ranker = DocumentRanker::new(config);
        for i in 0..10_usize {
            ranker.index_document(simple_doc(&format!("d{i}"), &["rust"]));
        }
        let results = ranker.rank(&["rust".to_string()], None);
        assert_eq!(results.len(), 3);
    }

    // -----------------------------------------------------------------------
    // 9. min_score filter removes low-scoring documents
    // -----------------------------------------------------------------------
    #[test]
    fn test_min_score_filter() {
        let config = RankingConfig {
            min_score: 999.0, // impossibly high threshold
            ..RankingConfig::default()
        };
        let mut ranker = DocumentRanker::new(config);
        ranker.index_document(simple_doc("d1", &["hello"]));
        let results = ranker.rank(&["hello".to_string()], None);
        assert!(results.is_empty());
    }

    // -----------------------------------------------------------------------
    // 10. Multi-doc ranking order is deterministic and correct
    // -----------------------------------------------------------------------
    #[test]
    fn test_multi_doc_ranking_order() {
        let config = RankingConfig {
            lexical_weight: 1.0,
            semantic_weight: 0.0,
            ..RankingConfig::default()
        };
        let mut ranker = DocumentRanker::new(config);
        ranker.index_document(simple_doc("d1", &["rust"]));
        ranker.index_document(simple_doc("d2", &["rust", "rust"]));
        ranker.index_document(simple_doc("d3", &["python"]));
        let results = ranker.rank(&["rust".to_string()], None);
        // d3 should rank last (no rust), d2 should rank above d1 (higher tf).
        assert!(results
            .iter()
            .position(|r| r.doc_id == "d3")
            .map(|p| p > results.iter().position(|r| r.doc_id == "d1").unwrap_or(0))
            .unwrap_or(true));
        // Scores descend.
        for w in results.windows(2) {
            assert!(w[0].combined_score >= w[1].combined_score);
        }
    }

    // -----------------------------------------------------------------------
    // 11. Empty query returns all docs with zero BM25 score
    // -----------------------------------------------------------------------
    #[test]
    fn test_empty_query_returns_zero_bm25() {
        let mut ranker = make_ranker();
        ranker.index_document(simple_doc("d1", &["rust"]));
        ranker.index_document(simple_doc("d2", &["python"]));
        let results = ranker.rank(&[], None);
        // With no query terms BM25=0 and no embedding, combined_score=0.
        // Both docs pass min_score=0.0 (0 >= 0).
        assert_eq!(results.len(), 2);
        for r in &results {
            assert_eq!(r.bm25_score, 0.0);
            assert_eq!(r.semantic_score, 0.0);
        }
    }

    // -----------------------------------------------------------------------
    // 12. Missing embedding is handled gracefully (no panic)
    // -----------------------------------------------------------------------
    #[test]
    fn test_missing_embedding_graceful() {
        let config = RankingConfig {
            semantic_weight: 1.0,
            lexical_weight: 0.0,
            ..RankingConfig::default()
        };
        let mut ranker = DocumentRanker::new(config);
        // Doc without embedding.
        ranker.index_document(simple_doc("no_emb", &["hello"]));
        let query_emb = vec![1.0, 0.0];
        // Should not panic; semantic_score should be 0.
        let results = ranker.rank(&[], Some(&query_emb));
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].semantic_score, 0.0);
    }

    // -----------------------------------------------------------------------
    // 13. Query embedding missing for doc that has one
    // -----------------------------------------------------------------------
    #[test]
    fn test_no_query_embedding_graceful() {
        let mut ranker = make_ranker();
        ranker.index_document(embed_doc("d1", &["hello"], vec![1.0, 0.0]));
        let results = ranker.rank(&["hello".to_string()], None);
        assert!(!results.is_empty());
        assert_eq!(results[0].semantic_score, 0.0);
    }

    // -----------------------------------------------------------------------
    // 14. Stats tracking — total_queries increments
    // -----------------------------------------------------------------------
    #[test]
    fn test_stats_total_queries() {
        let mut ranker = make_ranker();
        ranker.index_document(simple_doc("d1", &["hello"]));
        ranker.rank(&["hello".to_string()], None);
        ranker.rank(&["world".to_string()], None);
        assert_eq!(ranker.stats().total_queries, 2);
    }

    // -----------------------------------------------------------------------
    // 15. Stats tracking — documents_ranked accumulates
    // -----------------------------------------------------------------------
    #[test]
    fn test_stats_documents_ranked() {
        let config = RankingConfig {
            max_results: 100,
            min_score: 0.0,
            ..RankingConfig::default()
        };
        let mut ranker = DocumentRanker::new(config);
        for i in 0..5_usize {
            ranker.index_document(simple_doc(&format!("d{i}"), &["rust"]));
        }
        ranker.rank(&["rust".to_string()], None);
        // All 5 docs match with non-zero score (actually 0.0 == min_score so still pass).
        assert_eq!(ranker.stats().documents_ranked, 5);
    }

    // -----------------------------------------------------------------------
    // 16. Stats tracking — avg_results_per_query
    // -----------------------------------------------------------------------
    #[test]
    fn test_stats_avg_results() {
        let config = RankingConfig {
            max_results: 100,
            min_score: 0.0,
            ..RankingConfig::default()
        };
        let mut ranker = DocumentRanker::new(config);
        ranker.index_document(simple_doc("d1", &["rust"]));
        ranker.index_document(simple_doc("d2", &["python"]));
        ranker.rank(&["rust".to_string()], None); // 2 docs pass (score >= 0)
        ranker.rank(&["python".to_string()], None); // 2 docs again
        let avg = ranker.stats().avg_results_per_query;
        assert!((avg - 2.0).abs() < 1e-9, "expected 2.0 got {avg}");
    }

    // -----------------------------------------------------------------------
    // 17. avg_doc_length update
    // -----------------------------------------------------------------------
    #[test]
    fn test_avg_doc_length_update() {
        let mut ranker = make_ranker();
        assert_eq!(ranker.avg_doc_length(), 0.0);
        ranker.index_document(simple_doc("d1", &["a", "b"])); // length=2
        ranker.index_document(simple_doc("d2", &["x", "y", "z"])); // length=3
        let expected = (2.0 + 3.0) / 2.0;
        assert!((ranker.avg_doc_length() - expected).abs() < 1e-9);
    }

    // -----------------------------------------------------------------------
    // 18. document_count
    // -----------------------------------------------------------------------
    #[test]
    fn test_document_count() {
        let mut ranker = make_ranker();
        assert_eq!(ranker.document_count(), 0);
        ranker.index_document(simple_doc("d1", &["a"]));
        ranker.index_document(simple_doc("d2", &["b"]));
        assert_eq!(ranker.document_count(), 2);
    }

    // -----------------------------------------------------------------------
    // 19. Re-indexing the same doc_id overwrites
    // -----------------------------------------------------------------------
    #[test]
    fn test_reindex_overwrites() {
        let mut ranker = make_ranker();
        ranker.index_document(simple_doc("d1", &["rust"]));
        ranker.index_document(simple_doc("d1", &["python"])); // overwrite
        assert_eq!(ranker.document_count(), 1);
        let doc = ranker.get_document("d1").expect("d1 should exist");
        assert!(doc.term_frequencies.contains_key("python"));
        assert!(!doc.term_frequencies.contains_key("rust"));
    }

    // -----------------------------------------------------------------------
    // 20. cosine_similarity — identical vectors give 1.0
    // -----------------------------------------------------------------------
    #[test]
    fn test_cosine_identical() {
        let v = vec![0.3, 0.4, 0.5];
        let sim = DocumentRanker::cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 1e-9);
    }

    // -----------------------------------------------------------------------
    // 21. cosine_similarity — orthogonal vectors give 0.0
    // -----------------------------------------------------------------------
    #[test]
    fn test_cosine_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert_eq!(DocumentRanker::cosine_similarity(&a, &b), 0.0);
    }

    // -----------------------------------------------------------------------
    // 22. cosine_similarity — zero vector gives 0.0 (no NaN)
    // -----------------------------------------------------------------------
    #[test]
    fn test_cosine_zero_vector() {
        let a = vec![0.0, 0.0];
        let b = vec![1.0, 0.0];
        assert_eq!(DocumentRanker::cosine_similarity(&a, &b), 0.0);
    }

    // -----------------------------------------------------------------------
    // 23. cosine_similarity — mismatched lengths give 0.0 (no panic)
    // -----------------------------------------------------------------------
    #[test]
    fn test_cosine_length_mismatch() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0];
        assert_eq!(DocumentRanker::cosine_similarity(&a, &b), 0.0);
    }

    // -----------------------------------------------------------------------
    // 24. cosine_similarity — empty vectors give 0.0
    // -----------------------------------------------------------------------
    #[test]
    fn test_cosine_empty() {
        assert_eq!(DocumentRanker::cosine_similarity(&[], &[]), 0.0);
    }

    // -----------------------------------------------------------------------
    // 25. IDF cache is populated after update_idf_cache
    // -----------------------------------------------------------------------
    #[test]
    fn test_idf_cache_populated() {
        let mut ranker = make_ranker();
        ranker.index_document(simple_doc("d1", &["alpha"]));
        let terms = vec!["alpha".to_string(), "beta".to_string()];
        ranker.update_idf_cache(&terms);
        // Cache should have entries for both (beta may have idf even if df=0).
        assert!(ranker.idf_cache.contains_key("alpha"));
        assert!(ranker.idf_cache.contains_key("beta"));
    }

    // -----------------------------------------------------------------------
    // 26. rank result set rank values are 1..=N
    // -----------------------------------------------------------------------
    #[test]
    fn test_rank_values_sequential() {
        let mut ranker = make_ranker();
        for i in 0..5_usize {
            ranker.index_document(simple_doc(&format!("d{i}"), &["rust"]));
        }
        let results = ranker.rank(&["rust".to_string()], None);
        for (i, r) in results.iter().enumerate() {
            assert_eq!(r.rank, i + 1);
        }
    }

    // -----------------------------------------------------------------------
    // 27. RankingConfig default values
    // -----------------------------------------------------------------------
    #[test]
    fn test_ranking_config_defaults() {
        let cfg = RankingConfig::default();
        assert_eq!(cfg.bm25_k1, 1.5);
        assert_eq!(cfg.bm25_b, 0.75);
        assert_eq!(cfg.semantic_weight, 0.5);
        assert_eq!(cfg.lexical_weight, 0.5);
        assert_eq!(cfg.max_results, 10);
        assert_eq!(cfg.min_score, 0.0);
    }

    // -----------------------------------------------------------------------
    // 28. DocumentIndex from_tokens term normalisation (lowercase)
    // -----------------------------------------------------------------------
    #[test]
    fn test_from_tokens_lowercase() {
        let doc = DocumentIndex::from_tokens("d1", &["Rust", "RUST", "rust"], None);
        assert_eq!(
            doc.term_frequencies.get("rust").copied().unwrap_or(0.0),
            3.0
        );
        assert!(!doc.term_frequencies.contains_key("Rust"));
    }

    // -----------------------------------------------------------------------
    // 29. BM25 — term not in doc contributes 0
    // -----------------------------------------------------------------------
    #[test]
    fn test_bm25_missing_term_zero() {
        let mut ranker = make_ranker();
        ranker.index_document(simple_doc("d1", &["hello"]));
        let doc = ranker.get_document("d1").expect("d1 missing").clone();
        let score = ranker.bm25_score(&doc, &["nonexistent".to_string()]);
        assert_eq!(score, 0.0);
    }

    // -----------------------------------------------------------------------
    // 30. Semantic-only mode with zero lexical weight
    // -----------------------------------------------------------------------
    #[test]
    fn test_semantic_only_mode() {
        let config = RankingConfig {
            lexical_weight: 0.0,
            semantic_weight: 1.0,
            ..RankingConfig::default()
        };
        let mut ranker = DocumentRanker::new(config);
        ranker.index_document(embed_doc("close", &[], vec![0.9, 0.1]));
        ranker.index_document(embed_doc("distant", &[], vec![0.1, 0.9]));
        let qe = vec![1.0, 0.0];
        let results = ranker.rank(&[], Some(&qe));
        assert!(!results.is_empty());
        assert_eq!(results[0].doc_id, "close");
    }
}
