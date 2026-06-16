//! # Semantic Search Pipeline
//!
//! An end-to-end semantic search pipeline combining:
//! - Vector similarity search (cosine similarity)
//! - BM25 keyword matching
//! - Result fusion (RRF, LinearCombination, CombSUM)
//! - Re-ranking and metadata filtering
//!
//! ## Example
//!
//! ```rust
//! use ipfrs_semantic::search_pipeline::{
//!     SemanticSearchPipeline, SpPipelineConfig, SpSearchQuery,
//!     SearchDocument, FusionMethod,
//! };
//! use std::collections::HashMap;
//!
//! let config = SpPipelineConfig::default();
//! let mut pipeline = SemanticSearchPipeline::new(config);
//!
//! let doc = SearchDocument {
//!     id: "doc1".to_string(),
//!     content: "rust programming language systems".to_string(),
//!     embedding: vec![0.1, 0.2, 0.3, 0.4],
//!     metadata: HashMap::new(),
//! };
//! pipeline.add_document(doc);
//!
//! let query = SpSearchQuery {
//!     text: "rust programming".to_string(),
//!     embedding: Some(vec![0.1, 0.2, 0.3, 0.4]),
//!     filters: HashMap::new(),
//!     top_k: 10,
//!     min_score: 0.0,
//! };
//!
//! let result = pipeline.search(&query);
//! assert!(!result.hits.is_empty());
//! ```

use std::collections::HashMap;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// A document stored in the pipeline corpus.
#[derive(Debug, Clone)]
pub struct SearchDocument {
    pub id: String,
    pub content: String,
    pub embedding: Vec<f64>,
    pub metadata: HashMap<String, String>,
}

/// A search query issued against the pipeline.
///
/// Named `SpSearchQuery` to avoid collision with `multimodal_search::SearchQuery`.
#[derive(Debug, Clone)]
pub struct SpSearchQuery {
    pub text: String,
    pub embedding: Option<Vec<f64>>,
    pub filters: HashMap<String, String>,
    pub top_k: usize,
    pub min_score: f64,
}

/// A single result hit returned by the pipeline.
#[derive(Debug, Clone)]
pub struct SearchHit {
    pub doc_id: String,
    pub score: f64,
    pub vector_score: f64,
    pub bm25_score: f64,
    pub rank: usize,
}

/// The complete result of a pipeline search.
#[derive(Debug, Clone)]
pub struct SearchPipelineResult {
    pub query_text: String,
    pub hits: Vec<SearchHit>,
    pub total_candidates: usize,
    pub search_time_ms: u64,
}

/// Method used to fuse vector and BM25 ranked lists.
#[derive(Debug, Clone)]
pub enum FusionMethod {
    /// Reciprocal Rank Fusion: score = Σ 1 / (k + rank).
    /// `k` defaults to 60.0 and controls rank sensitivity.
    ReciprocalRankFusion { k: f64 },
    /// Weighted sum of min-max-normalised scores.
    LinearCombination {
        vector_weight: f64,
        bm25_weight: f64,
    },
    /// Sum of normalised scores with equal weights (0.5 / 0.5).
    CombSUM,
}

impl Default for FusionMethod {
    fn default() -> Self {
        FusionMethod::ReciprocalRankFusion { k: 60.0 }
    }
}

/// Configuration for the [`SemanticSearchPipeline`].
///
/// Named `SpPipelineConfig` to avoid collision with `query_pipeline::PipelineConfig`.
#[derive(Debug, Clone)]
pub struct SpPipelineConfig {
    pub fusion_method: FusionMethod,
    /// Number of top-scoring candidates returned by vector search.
    pub vector_candidates: usize,
    /// Number of top-scoring candidates returned by BM25.
    pub bm25_candidates: usize,
    /// Re-rank this many fused candidates before applying `min_score` / `top_k`.
    pub rerank_top_n: usize,
}

impl Default for SpPipelineConfig {
    fn default() -> Self {
        SpPipelineConfig {
            fusion_method: FusionMethod::ReciprocalRankFusion { k: 60.0 },
            vector_candidates: 100,
            bm25_candidates: 100,
            rerank_top_n: 20,
        }
    }
}

/// Runtime statistics for the pipeline.
///
/// Named `SpPipelineStats` to avoid collision with `embedding_pipeline::PipelineStats`.
#[derive(Debug, Clone)]
pub struct SpPipelineStats {
    pub doc_count: usize,
    pub vocabulary_size: usize,
    pub avg_doc_length: f64,
    pub total_searches: u64,
    pub avg_hits_per_search: f64,
}

// ---------------------------------------------------------------------------
// Internal BM25 per-document data
// ---------------------------------------------------------------------------

/// BM25 index data for a single document: token list and term-frequency map.
#[derive(Debug, Clone)]
struct DocBm25 {
    tokens: Vec<String>,
    tf: HashMap<String, f64>,
}

impl DocBm25 {
    fn from_content(content: &str) -> Self {
        let tokens: Vec<String> = tokenize(content);
        let mut tf: HashMap<String, f64> = HashMap::new();
        for t in &tokens {
            *tf.entry(t.clone()).or_insert(0.0) += 1.0;
        }
        DocBm25 { tokens, tf }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Lowercase whitespace tokeniser.
fn tokenize(text: &str) -> Vec<String> {
    text.split_whitespace().map(|w| w.to_lowercase()).collect()
}

/// Min-max normalise a scored list so that the maximum value becomes 1.0.
/// Returns all zeros if the maximum is zero.
fn normalise(scores: &[(String, f64)]) -> Vec<(String, f64)> {
    let max = scores.iter().map(|(_, s)| *s).fold(0.0_f64, f64::max);
    if max == 0.0 {
        return scores.iter().map(|(id, _)| (id.clone(), 0.0)).collect();
    }
    scores.iter().map(|(id, s)| (id.clone(), s / max)).collect()
}

// ---------------------------------------------------------------------------
// SemanticSearchPipeline
// ---------------------------------------------------------------------------

/// End-to-end semantic search pipeline.
///
/// Combines cosine-similarity vector search with BM25 keyword retrieval and
/// fuses the two ranked lists via a configurable [`FusionMethod`].  Results
/// are optionally filtered by document metadata and a minimum score threshold.
#[derive(Debug)]
pub struct SemanticSearchPipeline {
    pub config: SpPipelineConfig,
    /// Document corpus indexed by doc id.
    pub documents: HashMap<String, SearchDocument>,
    /// Pre-computed BM25 data (tokens, TF) indexed by doc id.
    bm25_data: HashMap<String, DocBm25>,
    /// IDF table: term → idf value.
    pub idf: HashMap<String, f64>,
    /// Document-frequency table: term → number of documents containing it.
    df: HashMap<String, usize>,
    /// Total number of documents (kept in sync with `documents.len()`).
    pub total_docs: usize,
    /// Cumulative search counter.
    total_searches: u64,
    /// Cumulative hit count (used for `avg_hits_per_search`).
    total_hits: u64,
}

impl SemanticSearchPipeline {
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    /// Create a new pipeline with the given configuration.
    pub fn new(config: SpPipelineConfig) -> Self {
        SemanticSearchPipeline {
            config,
            documents: HashMap::new(),
            bm25_data: HashMap::new(),
            idf: HashMap::new(),
            df: HashMap::new(),
            total_docs: 0,
            total_searches: 0,
            total_hits: 0,
        }
    }

    // -----------------------------------------------------------------------
    // Corpus management
    // -----------------------------------------------------------------------

    /// Add a document to the corpus and update IDF for all terms in its content.
    pub fn add_document(&mut self, doc: SearchDocument) {
        let bm25 = DocBm25::from_content(&doc.content);

        // Update document-frequency counts for every unique term in the doc.
        for term in bm25.tf.keys() {
            *self.df.entry(term.clone()).or_insert(0) += 1;
        }

        let doc_id = doc.id.clone();
        self.documents.insert(doc_id.clone(), doc);
        self.bm25_data.insert(doc_id, bm25);
        self.total_docs = self.documents.len();
        self.recompute_idf();
    }

    /// Remove a document from the corpus by ID, updating the BM25 index and IDF.
    /// Returns `true` if the document was present and was removed.
    pub fn remove_document(&mut self, doc_id: &str) -> bool {
        if let Some(bm25) = self.bm25_data.remove(doc_id) {
            self.documents.remove(doc_id);
            // Decrease DF for each term present in the removed document.
            for term in bm25.tf.keys() {
                if let Some(count) = self.df.get_mut(term.as_str()) {
                    if *count <= 1 {
                        self.df.remove(term.as_str());
                    } else {
                        *count -= 1;
                    }
                }
            }
            self.total_docs = self.documents.len();
            self.recompute_idf();
            true
        } else {
            false
        }
    }

    /// Number of documents currently indexed.
    pub fn doc_count(&self) -> usize {
        self.documents.len()
    }

    /// Return runtime statistics for the pipeline.
    pub fn stats(&self) -> SpPipelineStats {
        let avg_doc_length = if self.bm25_data.is_empty() {
            0.0
        } else {
            let total_tokens: usize = self.bm25_data.values().map(|b| b.tokens.len()).sum();
            total_tokens as f64 / self.bm25_data.len() as f64
        };

        let avg_hits_per_search = if self.total_searches == 0 {
            0.0
        } else {
            self.total_hits as f64 / self.total_searches as f64
        };

        SpPipelineStats {
            doc_count: self.total_docs,
            vocabulary_size: self.idf.len(),
            avg_doc_length,
            total_searches: self.total_searches,
            avg_hits_per_search,
        }
    }

    // -----------------------------------------------------------------------
    // Full pipeline search
    // -----------------------------------------------------------------------

    /// Run the full pipeline for a query and return ranked, filtered results.
    ///
    /// Steps:
    /// 1. Vector search (if `query.embedding` is `Some`)
    /// 2. BM25 keyword search
    /// 3. Fuse the two ranked lists
    /// 4. Take top `rerank_top_n`, filter by `min_score` and metadata
    /// 5. Limit to `top_k`, assign ranks 1..n, return
    pub fn search(&mut self, query: &SpSearchQuery) -> SearchPipelineResult {
        let start = Instant::now();

        // 1. Vector search (optional).
        let vector_results: Vec<(String, f64)> = query
            .embedding
            .as_deref()
            .map(|emb| self.vector_search(emb, self.config.vector_candidates))
            .unwrap_or_default();

        // 2. BM25 keyword search.
        let bm25_results = self.bm25_search(&query.text, self.config.bm25_candidates);

        // 3. Count distinct candidates across both lists.
        let mut candidate_ids: std::collections::HashSet<&str> = std::collections::HashSet::new();
        for (id, _) in &vector_results {
            candidate_ids.insert(id.as_str());
        }
        for (id, _) in &bm25_results {
            candidate_ids.insert(id.as_str());
        }
        let total_candidates = candidate_ids.len();

        // Build quick lookup maps for individual scores (populate SearchHit fields).
        let vector_map: HashMap<&str, f64> = vector_results
            .iter()
            .map(|(id, s)| (id.as_str(), *s))
            .collect();
        let bm25_map: HashMap<&str, f64> = bm25_results
            .iter()
            .map(|(id, s)| (id.as_str(), *s))
            .collect();

        // 4. Fuse the two ranked lists.
        let mut fused = self.fuse(&vector_results, &bm25_results);
        fused.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let rerank_top_n = self.config.rerank_top_n;
        let min_score = query.min_score;
        let top_k = query.top_k;

        // 5. Apply rerank_top_n, min_score filter, metadata filter, top_k cap.
        let mut hits: Vec<SearchHit> = fused
            .into_iter()
            .take(rerank_top_n)
            .filter(|(_, score)| *score >= min_score)
            .filter(|(id, _)| self.matches_filters(id, &query.filters))
            .take(top_k)
            .map(|(id, score)| {
                let vs = vector_map.get(id.as_str()).copied().unwrap_or(0.0);
                let bs = bm25_map.get(id.as_str()).copied().unwrap_or(0.0);
                SearchHit {
                    doc_id: id,
                    score,
                    vector_score: vs,
                    bm25_score: bs,
                    rank: 0, // assigned below
                }
            })
            .collect();

        // 6. Final sort and rank assignment (1-based).
        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for (rank, hit) in hits.iter_mut().enumerate() {
            hit.rank = rank + 1;
        }

        // Update internal counters.
        self.total_searches += 1;
        self.total_hits += hits.len() as u64;

        let search_time_ms = start.elapsed().as_millis() as u64;

        SearchPipelineResult {
            query_text: query.text.clone(),
            hits,
            total_candidates,
            search_time_ms,
        }
    }

    // -----------------------------------------------------------------------
    // Vector search
    // -----------------------------------------------------------------------

    /// Compute cosine similarity against every document and return the top-k
    /// `(doc_id, similarity)` pairs, sorted descending.
    pub fn vector_search(&self, embedding: &[f64], top_k: usize) -> Vec<(String, f64)> {
        let mut scores: Vec<(String, f64)> = self
            .documents
            .iter()
            .map(|(id, doc)| {
                let sim = Self::cosine_similarity(embedding, &doc.embedding);
                (id.clone(), sim)
            })
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(top_k);
        scores
    }

    // -----------------------------------------------------------------------
    // BM25 search
    // -----------------------------------------------------------------------

    /// Compute BM25 scores for `query_text` against every document.
    ///
    /// Parameters: k₁ = 1.5, b = 0.75 (standard Okapi BM25 defaults).
    /// Returns the top-k `(doc_id, bm25_score)` pairs, sorted descending.
    pub fn bm25_search(&self, query_text: &str, top_k: usize) -> Vec<(String, f64)> {
        let query_tokens = tokenize(query_text);
        if query_tokens.is_empty() || self.total_docs == 0 {
            return Vec::new();
        }

        let avgdl = self.average_doc_length();
        let k1 = 1.5_f64;
        let b = 0.75_f64;

        let mut scores: Vec<(String, f64)> = self
            .bm25_data
            .iter()
            .filter_map(|(doc_id, bm25)| {
                let dl = bm25.tokens.len() as f64;
                let score: f64 = query_tokens
                    .iter()
                    .map(|term| {
                        let idf = self.idf.get(term.as_str()).copied().unwrap_or(0.0);
                        if idf <= 0.0 {
                            return 0.0;
                        }
                        let tf = bm25.tf.get(term.as_str()).copied().unwrap_or(0.0);
                        let denom = tf + k1 * (1.0 - b + b * dl / avgdl.max(1.0));
                        idf * tf * (k1 + 1.0) / denom
                    })
                    .sum();
                if score > 0.0 {
                    Some((doc_id.clone(), score))
                } else {
                    None
                }
            })
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(top_k);
        scores
    }

    // -----------------------------------------------------------------------
    // Fusion
    // -----------------------------------------------------------------------

    /// Fuse two ranked lists according to the configured [`FusionMethod`].
    ///
    /// Each input list is assumed to be sorted descending by score (rank 1 = best).
    /// Returns an unsorted `(doc_id, fused_score)` map as a `Vec`.
    pub fn fuse(
        &self,
        vector_results: &[(String, f64)],
        bm25_results: &[(String, f64)],
    ) -> Vec<(String, f64)> {
        match &self.config.fusion_method {
            FusionMethod::ReciprocalRankFusion { k } => {
                self.rrf_fuse(vector_results, bm25_results, *k)
            }
            FusionMethod::LinearCombination {
                vector_weight,
                bm25_weight,
            } => self.linear_fuse(vector_results, bm25_results, *vector_weight, *bm25_weight),
            FusionMethod::CombSUM => self.linear_fuse(vector_results, bm25_results, 0.5, 0.5),
        }
    }

    // -----------------------------------------------------------------------
    // Cosine similarity (public static)
    // -----------------------------------------------------------------------

    /// Cosine similarity between two equal-length vectors.
    ///
    /// Returns 0.0 for empty inputs, mismatched lengths, or zero-norm vectors.
    pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
        if a.is_empty() || a.len() != b.len() {
            return 0.0;
        }
        let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }
        dot / (norm_a * norm_b)
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Recompute the full IDF table from the current document-frequency table.
    fn recompute_idf(&mut self) {
        let n = self.total_docs as f64;
        self.idf = self
            .df
            .iter()
            .map(|(term, &df_count)| {
                let df_f = df_count as f64;
                // Okapi BM25 IDF with smoothing.
                let idf = ((n - df_f + 0.5) / (df_f + 0.5) + 1.0).ln();
                (term.clone(), idf)
            })
            .collect();
    }

    /// Average document length in tokens across the current corpus.
    /// Returns 1.0 for an empty corpus to avoid division by zero.
    fn average_doc_length(&self) -> f64 {
        if self.bm25_data.is_empty() {
            return 1.0;
        }
        let total: usize = self.bm25_data.values().map(|b| b.tokens.len()).sum();
        total as f64 / self.bm25_data.len() as f64
    }

    /// True iff the document's metadata contains every key-value pair in `filters`.
    fn matches_filters(&self, doc_id: &str, filters: &HashMap<String, String>) -> bool {
        if filters.is_empty() {
            return true;
        }
        self.documents
            .get(doc_id)
            .map(|doc| {
                filters.iter().all(|(k, v)| {
                    doc.metadata
                        .get(k.as_str())
                        .map(|mv| mv == v)
                        .unwrap_or(false)
                })
            })
            .unwrap_or(false)
    }

    /// Reciprocal Rank Fusion.
    ///
    /// For each document: fused_score = Σ_list 1 / (k + rank_in_list)
    /// where rank is 1-based (best = 1).
    fn rrf_fuse(
        &self,
        vector_results: &[(String, f64)],
        bm25_results: &[(String, f64)],
        k: f64,
    ) -> Vec<(String, f64)> {
        let mut scores: HashMap<String, f64> = HashMap::new();

        for (rank, (doc_id, _)) in vector_results.iter().enumerate() {
            // rank is 0-based; RRF uses 1-based, so add 1.
            *scores.entry(doc_id.clone()).or_insert(0.0) += 1.0 / (k + rank as f64 + 1.0);
        }
        for (rank, (doc_id, _)) in bm25_results.iter().enumerate() {
            *scores.entry(doc_id.clone()).or_insert(0.0) += 1.0 / (k + rank as f64 + 1.0);
        }

        scores.into_iter().collect()
    }

    /// Linear combination fusion with per-list normalised scores.
    fn linear_fuse(
        &self,
        vector_results: &[(String, f64)],
        bm25_results: &[(String, f64)],
        vector_weight: f64,
        bm25_weight: f64,
    ) -> Vec<(String, f64)> {
        let norm_vec = normalise(vector_results);
        let norm_bm25 = normalise(bm25_results);

        let mut scores: HashMap<String, f64> = HashMap::new();

        for (id, s) in &norm_vec {
            *scores.entry(id.clone()).or_insert(0.0) += vector_weight * s;
        }
        for (id, s) in &norm_bm25 {
            *scores.entry(id.clone()).or_insert(0.0) += bm25_weight * s;
        }

        scores.into_iter().collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::search_pipeline::{
        FusionMethod, SearchDocument, SemanticSearchPipeline, SpPipelineConfig, SpSearchQuery,
    };

    // -----------------------------------------------------------------------
    // Test helpers
    // -----------------------------------------------------------------------

    fn make_doc(id: &str, content: &str, embedding: Vec<f64>) -> SearchDocument {
        SearchDocument {
            id: id.to_string(),
            content: content.to_string(),
            embedding,
            metadata: HashMap::new(),
        }
    }

    fn make_doc_meta(
        id: &str,
        content: &str,
        embedding: Vec<f64>,
        metadata: HashMap<String, String>,
    ) -> SearchDocument {
        SearchDocument {
            id: id.to_string(),
            content: content.to_string(),
            embedding,
            metadata,
        }
    }

    fn default_pipeline() -> SemanticSearchPipeline {
        SemanticSearchPipeline::new(SpPipelineConfig::default())
    }

    fn pipeline_with_docs() -> SemanticSearchPipeline {
        let mut p = default_pipeline();
        p.add_document(make_doc(
            "d1",
            "rust programming language",
            vec![1.0, 0.0, 0.0],
        ));
        p.add_document(make_doc(
            "d2",
            "python data science machine learning",
            vec![0.0, 1.0, 0.0],
        ));
        p.add_document(make_doc(
            "d3",
            "rust systems programming performance",
            vec![0.9, 0.1, 0.0],
        ));
        p
    }

    fn simple_query(text: &str, embedding: Option<Vec<f64>>) -> SpSearchQuery {
        SpSearchQuery {
            text: text.to_string(),
            embedding,
            filters: HashMap::new(),
            top_k: 10,
            min_score: 0.0,
        }
    }

    // -----------------------------------------------------------------------
    // 1. Pipeline construction
    // -----------------------------------------------------------------------

    #[test]
    fn test_new_empty_pipeline() {
        let p = default_pipeline();
        assert_eq!(p.doc_count(), 0);
        assert_eq!(p.total_docs, 0);
    }

    #[test]
    fn test_config_default_vector_candidates() {
        let cfg = SpPipelineConfig::default();
        assert_eq!(cfg.vector_candidates, 100);
    }

    #[test]
    fn test_config_default_bm25_candidates() {
        let cfg = SpPipelineConfig::default();
        assert_eq!(cfg.bm25_candidates, 100);
    }

    #[test]
    fn test_config_default_rerank_top_n() {
        let cfg = SpPipelineConfig::default();
        assert_eq!(cfg.rerank_top_n, 20);
    }

    // -----------------------------------------------------------------------
    // 2. add_document / remove_document
    // -----------------------------------------------------------------------

    #[test]
    fn test_add_single_document() {
        let mut p = default_pipeline();
        p.add_document(make_doc("x", "hello world", vec![1.0, 0.0]));
        assert_eq!(p.doc_count(), 1);
        assert_eq!(p.total_docs, 1);
    }

    #[test]
    fn test_add_multiple_documents() {
        let p = pipeline_with_docs();
        assert_eq!(p.doc_count(), 3);
        assert_eq!(p.total_docs, 3);
    }

    #[test]
    fn test_remove_existing_document() {
        let mut p = pipeline_with_docs();
        let removed = p.remove_document("d1");
        assert!(removed);
        assert_eq!(p.doc_count(), 2);
    }

    #[test]
    fn test_remove_missing_document() {
        let mut p = pipeline_with_docs();
        let removed = p.remove_document("nonexistent");
        assert!(!removed);
        assert_eq!(p.doc_count(), 3);
    }

    #[test]
    fn test_remove_then_readd() {
        let mut p = pipeline_with_docs();
        p.remove_document("d1");
        p.add_document(make_doc(
            "d1",
            "rust programming language",
            vec![1.0, 0.0, 0.0],
        ));
        assert_eq!(p.doc_count(), 3);
    }

    #[test]
    fn test_doc_count_matches_total_docs() {
        let p = pipeline_with_docs();
        assert_eq!(p.doc_count(), p.total_docs);
    }

    // -----------------------------------------------------------------------
    // 3. IDF / vocabulary
    // -----------------------------------------------------------------------

    #[test]
    fn test_idf_populated_after_add() {
        let p = pipeline_with_docs();
        assert!(!p.idf.is_empty());
    }

    #[test]
    fn test_idf_rust_is_nonnegative() {
        let p = pipeline_with_docs();
        let idf_rust = p.idf.get("rust").copied().unwrap_or(0.0);
        assert!(idf_rust >= 0.0);
    }

    #[test]
    fn test_idf_decreases_as_df_increases() {
        let mut p = default_pipeline();
        p.add_document(make_doc("a", "rust programming", vec![1.0]));
        let idf_before = p.idf.get("rust").copied().unwrap_or(0.0);
        p.add_document(make_doc("b", "rust is great", vec![0.5]));
        let idf_after = p.idf.get("rust").copied().unwrap_or(0.0);
        // More docs containing "rust" → lower IDF.
        assert!(idf_after <= idf_before);
    }

    #[test]
    fn test_vocabulary_grows_on_new_terms() {
        let mut p = default_pipeline();
        p.add_document(make_doc("a", "hello", vec![1.0]));
        let v1 = p.idf.len();
        p.add_document(make_doc("b", "world unique_term_xyz", vec![0.5]));
        let v2 = p.idf.len();
        assert!(v2 > v1);
    }

    #[test]
    fn test_idf_term_removed_when_only_doc_deleted() {
        let mut p = default_pipeline();
        p.add_document(make_doc("only", "unique_xyz_term_qwerty", vec![1.0]));
        assert!(p.idf.contains_key("unique_xyz_term_qwerty"));
        p.remove_document("only");
        assert!(!p.idf.contains_key("unique_xyz_term_qwerty"));
    }

    // -----------------------------------------------------------------------
    // 4. cosine_similarity
    // -----------------------------------------------------------------------

    #[test]
    fn test_cosine_identical_vectors() {
        let v = vec![1.0, 2.0, 3.0];
        assert!((SemanticSearchPipeline::cosine_similarity(&v, &v) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_cosine_orthogonal_vectors() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!(SemanticSearchPipeline::cosine_similarity(&a, &b).abs() < 1e-9);
    }

    #[test]
    fn test_cosine_opposite_vectors() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        assert!((SemanticSearchPipeline::cosine_similarity(&a, &b) + 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_cosine_zero_vector_returns_zero() {
        let a = vec![0.0, 0.0];
        let b = vec![1.0, 2.0];
        assert_eq!(SemanticSearchPipeline::cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn test_cosine_empty_vectors() {
        assert_eq!(SemanticSearchPipeline::cosine_similarity(&[], &[]), 0.0);
    }

    #[test]
    fn test_cosine_mismatched_lengths() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0];
        assert_eq!(SemanticSearchPipeline::cosine_similarity(&a, &b), 0.0);
    }

    // -----------------------------------------------------------------------
    // 5. vector_search
    // -----------------------------------------------------------------------

    #[test]
    fn test_vector_search_returns_top_k() {
        let p = pipeline_with_docs();
        let results = p.vector_search(&[1.0, 0.0, 0.0], 2);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_vector_search_sorted_descending() {
        let p = pipeline_with_docs();
        let results = p.vector_search(&[1.0, 0.0, 0.0], 3);
        for w in results.windows(2) {
            assert!(w[0].1 >= w[1].1);
        }
    }

    #[test]
    fn test_vector_search_correct_top_result() {
        let p = pipeline_with_docs();
        // d1 = [1,0,0] — perfect cosine match with [1,0,0].
        let results = p.vector_search(&[1.0, 0.0, 0.0], 1);
        assert_eq!(results[0].0, "d1");
        assert!((results[0].1 - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_vector_search_empty_corpus() {
        let p = default_pipeline();
        let results = p.vector_search(&[1.0, 0.0], 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_vector_search_top_k_larger_than_corpus() {
        let p = pipeline_with_docs();
        let results = p.vector_search(&[1.0, 0.0, 0.0], 1000);
        assert!(results.len() <= p.doc_count());
    }

    // -----------------------------------------------------------------------
    // 6. bm25_search
    // -----------------------------------------------------------------------

    #[test]
    fn test_bm25_search_returns_rust_docs() {
        let p = pipeline_with_docs();
        let results = p.bm25_search("rust", 5);
        let ids: Vec<&str> = results.iter().map(|(id, _)| id.as_str()).collect();
        assert!(ids.contains(&"d1") || ids.contains(&"d3"));
    }

    #[test]
    fn test_bm25_empty_query_returns_empty() {
        let p = pipeline_with_docs();
        let results = p.bm25_search("", 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_bm25_unknown_term_returns_empty() {
        let p = pipeline_with_docs();
        let results = p.bm25_search("zzz_nonexistent_term", 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_bm25_scores_sorted_descending() {
        let p = pipeline_with_docs();
        let results = p.bm25_search("rust programming", 5);
        for w in results.windows(2) {
            assert!(w[0].1 >= w[1].1);
        }
    }

    #[test]
    fn test_bm25_top_k_respected() {
        let p = pipeline_with_docs();
        let results = p.bm25_search("rust", 1);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_bm25_higher_tf_scores_higher() {
        let mut p = default_pipeline();
        p.add_document(make_doc("doc_high", "rust rust language", vec![1.0]));
        p.add_document(make_doc("doc_low", "rust language", vec![0.5]));
        let results = p.bm25_search("rust", 2);
        assert_eq!(results[0].0, "doc_high");
    }

    #[test]
    fn test_bm25_score_positive_for_matching_terms() {
        let p = pipeline_with_docs();
        let results = p.bm25_search("python", 5);
        let d2_score = results
            .iter()
            .find(|(id, _)| id == "d2")
            .map(|(_, s)| *s)
            .unwrap_or(0.0);
        assert!(d2_score > 0.0);
    }

    // -----------------------------------------------------------------------
    // 7. Fusion
    // -----------------------------------------------------------------------

    #[test]
    fn test_rrf_fusion_contains_all_ids() {
        let p = default_pipeline();
        let vec_res = vec![("a".to_string(), 0.9), ("b".to_string(), 0.7)];
        let bm25_res = vec![("b".to_string(), 5.0), ("c".to_string(), 3.0)];
        let fused = p.fuse(&vec_res, &bm25_res);
        let ids: Vec<&str> = fused.iter().map(|(id, _)| id.as_str()).collect();
        assert!(ids.contains(&"a"));
        assert!(ids.contains(&"b"));
        assert!(ids.contains(&"c"));
    }

    #[test]
    fn test_rrf_shared_doc_scores_higher_than_unique() {
        let p = default_pipeline();
        let vec_res = vec![("shared".to_string(), 0.9), ("only_vec".to_string(), 0.8)];
        let bm25_res = vec![("shared".to_string(), 8.0), ("only_bm25".to_string(), 6.0)];
        let fused = p.fuse(&vec_res, &bm25_res);
        let shared_score = fused
            .iter()
            .find(|(id, _)| id == "shared")
            .map(|(_, s)| *s)
            .unwrap_or(0.0);
        let only_vec_score = fused
            .iter()
            .find(|(id, _)| id == "only_vec")
            .map(|(_, s)| *s)
            .unwrap_or(0.0);
        assert!(shared_score > only_vec_score);
    }

    #[test]
    fn test_linear_fusion_zero_bm25_weight() {
        let config = SpPipelineConfig {
            fusion_method: FusionMethod::LinearCombination {
                vector_weight: 1.0,
                bm25_weight: 0.0,
            },
            ..Default::default()
        };
        let p = SemanticSearchPipeline::new(config);
        let vec_res = vec![("a".to_string(), 1.0)];
        let bm25_res = vec![("b".to_string(), 10.0)];
        let fused = p.fuse(&vec_res, &bm25_res);
        let b_score = fused
            .iter()
            .find(|(id, _)| id == "b")
            .map(|(_, s)| *s)
            .unwrap_or(0.0);
        assert!((b_score).abs() < 1e-9);
    }

    #[test]
    fn test_combsum_equal_weights_sums_to_one() {
        let config = SpPipelineConfig {
            fusion_method: FusionMethod::CombSUM,
            ..Default::default()
        };
        let p = SemanticSearchPipeline::new(config);
        let vec_res = vec![("a".to_string(), 1.0)];
        let bm25_res = vec![("a".to_string(), 2.0)];
        let fused = p.fuse(&vec_res, &bm25_res);
        let a_score = fused
            .iter()
            .find(|(id, _)| id == "a")
            .map(|(_, s)| *s)
            .unwrap_or(0.0);
        // After normalisation both lists yield 1.0; CombSUM = 0.5*1 + 0.5*1 = 1.0.
        assert!((a_score - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_fusion_empty_inputs_returns_empty() {
        let p = default_pipeline();
        let fused = p.fuse(&[], &[]);
        assert!(fused.is_empty());
    }

    #[test]
    fn test_fusion_single_list_passthrough() {
        let p = default_pipeline();
        let vec_res = vec![("a".to_string(), 1.0), ("b".to_string(), 0.5)];
        let fused = p.fuse(&vec_res, &[]);
        assert_eq!(fused.len(), 2);
    }

    #[test]
    fn test_rrf_k_lower_gives_higher_score() {
        let p_low = SemanticSearchPipeline::new(SpPipelineConfig {
            fusion_method: FusionMethod::ReciprocalRankFusion { k: 1.0 },
            ..Default::default()
        });
        let p_high = SemanticSearchPipeline::new(SpPipelineConfig {
            fusion_method: FusionMethod::ReciprocalRankFusion { k: 1000.0 },
            ..Default::default()
        });
        let vec_res = vec![("a".to_string(), 1.0)];
        let bm25_res = vec![("a".to_string(), 1.0)];
        let score_low = p_low
            .fuse(&vec_res, &bm25_res)
            .first()
            .map(|(_, s)| *s)
            .unwrap_or(0.0);
        let score_high = p_high
            .fuse(&vec_res, &bm25_res)
            .first()
            .map(|(_, s)| *s)
            .unwrap_or(0.0);
        assert!(score_low > score_high);
    }

    // -----------------------------------------------------------------------
    // 8. Full search pipeline
    // -----------------------------------------------------------------------

    #[test]
    fn test_search_returns_hits() {
        let mut p = pipeline_with_docs();
        let q = simple_query("rust", Some(vec![1.0, 0.0, 0.0]));
        let result = p.search(&q);
        assert!(!result.hits.is_empty());
    }

    #[test]
    fn test_search_hits_sorted_by_score_desc() {
        let mut p = pipeline_with_docs();
        let q = simple_query("rust programming", Some(vec![1.0, 0.0, 0.0]));
        let result = p.search(&q);
        for w in result.hits.windows(2) {
            assert!(w[0].score >= w[1].score);
        }
    }

    #[test]
    fn test_search_rank_starts_at_one() {
        let mut p = pipeline_with_docs();
        let q = simple_query("rust", None);
        let result = p.search(&q);
        if !result.hits.is_empty() {
            assert_eq!(result.hits[0].rank, 1);
        }
    }

    #[test]
    fn test_search_ranks_are_sequential() {
        let mut p = pipeline_with_docs();
        let q = simple_query("rust programming", Some(vec![1.0, 0.0, 0.0]));
        let result = p.search(&q);
        for (i, hit) in result.hits.iter().enumerate() {
            assert_eq!(hit.rank, i + 1);
        }
    }

    #[test]
    fn test_search_respects_top_k() {
        let mut p = pipeline_with_docs();
        let q = SpSearchQuery {
            text: "rust programming language systems".to_string(),
            embedding: Some(vec![1.0, 0.0, 0.0]),
            filters: HashMap::new(),
            top_k: 1,
            min_score: 0.0,
        };
        let result = p.search(&q);
        assert!(result.hits.len() <= 1);
    }

    #[test]
    fn test_search_min_score_filters_all() {
        let mut p = pipeline_with_docs();
        let q = SpSearchQuery {
            text: "rust".to_string(),
            embedding: Some(vec![1.0, 0.0, 0.0]),
            filters: HashMap::new(),
            top_k: 10,
            min_score: 9999.0,
        };
        let result = p.search(&q);
        assert!(result.hits.is_empty());
    }

    #[test]
    fn test_search_total_candidates_positive() {
        let mut p = pipeline_with_docs();
        let q = simple_query("rust", Some(vec![1.0, 0.0, 0.0]));
        let result = p.search(&q);
        assert!(result.total_candidates > 0);
    }

    #[test]
    fn test_search_query_text_in_result() {
        let mut p = pipeline_with_docs();
        let q = simple_query("hello world", None);
        let result = p.search(&q);
        assert_eq!(result.query_text, "hello world");
    }

    #[test]
    fn test_search_vector_only_finds_similar_doc() {
        let mut p = pipeline_with_docs();
        let q = SpSearchQuery {
            text: String::new(),
            embedding: Some(vec![0.0, 1.0, 0.0]),
            filters: HashMap::new(),
            top_k: 5,
            min_score: 0.0,
        };
        let result = p.search(&q);
        // d2 has embedding [0,1,0] — should be the top hit.
        assert!(!result.hits.is_empty());
        assert_eq!(result.hits[0].doc_id, "d2");
    }

    #[test]
    fn test_search_on_empty_corpus() {
        let mut p = default_pipeline();
        let q = simple_query("rust", Some(vec![1.0, 0.0]));
        let result = p.search(&q);
        assert!(result.hits.is_empty());
        assert_eq!(result.total_candidates, 0);
    }

    #[test]
    fn test_hit_fields_all_populated() {
        let mut p = pipeline_with_docs();
        let q = simple_query("rust", Some(vec![1.0, 0.0, 0.0]));
        let result = p.search(&q);
        for hit in &result.hits {
            assert!(!hit.doc_id.is_empty());
            assert!(hit.score >= 0.0);
            assert!(hit.rank >= 1);
        }
    }

    // -----------------------------------------------------------------------
    // 9. Metadata filtering
    // -----------------------------------------------------------------------

    #[test]
    fn test_metadata_filter_exact_match() {
        let mut meta = HashMap::new();
        meta.insert("lang".to_string(), "rust".to_string());
        let mut p = default_pipeline();
        p.add_document(make_doc_meta(
            "d1",
            "systems programming",
            vec![1.0, 0.0],
            meta,
        ));
        p.add_document(make_doc("d2", "systems programming", vec![1.0, 0.0]));

        let mut filters = HashMap::new();
        filters.insert("lang".to_string(), "rust".to_string());
        let q = SpSearchQuery {
            text: "systems programming".to_string(),
            embedding: Some(vec![1.0, 0.0]),
            filters,
            top_k: 10,
            min_score: 0.0,
        };
        let result = p.search(&q);
        assert!(result.hits.iter().all(|h| h.doc_id == "d1"));
    }

    #[test]
    fn test_metadata_filter_no_match_returns_empty() {
        let mut p = pipeline_with_docs();
        let mut filters = HashMap::new();
        filters.insert("nonexistent".to_string(), "value".to_string());
        let q = SpSearchQuery {
            text: "rust".to_string(),
            embedding: Some(vec![1.0, 0.0, 0.0]),
            filters,
            top_k: 10,
            min_score: 0.0,
        };
        let result = p.search(&q);
        assert!(result.hits.is_empty());
    }

    #[test]
    fn test_metadata_empty_filter_passes_all() {
        let mut p = pipeline_with_docs();
        let q = simple_query("rust", Some(vec![1.0, 0.0, 0.0]));
        let result = p.search(&q);
        assert!(!result.hits.is_empty());
    }

    // -----------------------------------------------------------------------
    // 10. Stats
    // -----------------------------------------------------------------------

    #[test]
    fn test_stats_initial_state() {
        let p = default_pipeline();
        let s = p.stats();
        assert_eq!(s.doc_count, 0);
        assert_eq!(s.total_searches, 0);
        assert_eq!(s.vocabulary_size, 0);
        assert_eq!(s.avg_doc_length, 0.0);
        assert_eq!(s.avg_hits_per_search, 0.0);
    }

    #[test]
    fn test_stats_after_docs_added() {
        let p = pipeline_with_docs();
        let s = p.stats();
        assert_eq!(s.doc_count, 3);
        assert!(s.vocabulary_size > 0);
        assert!(s.avg_doc_length > 0.0);
    }

    #[test]
    fn test_stats_total_searches_increments() {
        let mut p = pipeline_with_docs();
        let q = simple_query("rust", None);
        p.search(&q);
        p.search(&q);
        assert_eq!(p.stats().total_searches, 2);
    }

    #[test]
    fn test_stats_avg_hits_computed_correctly() {
        let mut p = pipeline_with_docs();
        let q = simple_query("rust", None);
        let r = p.search(&q);
        let hits = r.hits.len() as f64;
        let s = p.stats();
        assert!((s.avg_hits_per_search - hits).abs() < 1e-9);
    }
}
