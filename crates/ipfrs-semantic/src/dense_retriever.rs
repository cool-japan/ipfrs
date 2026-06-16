//! Dense Retriever — hybrid dense vector + BM25 sparse retrieval system.
//!
//! Combines exact cosine-similarity nearest-neighbour search over raw f64 embeddings
//! with a BM25 inverted index for lexical scoring.  Results from both paths are
//! independently min-max normalised and then fused via a configurable `hybrid_alpha`
//! parameter before being returned as ranked [`RetrievalResult`] items.
//!
//! # Example
//!
//! ```rust
//! use ipfrs_semantic::dense_retriever::{
//!     DenseRetriever, Document, RetrievalQuery, RetrieverConfig,
//! };
//! use std::collections::HashMap;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let config = RetrieverConfig::default();
//! let mut retriever = DenseRetriever::new(config);
//!
//! let doc = Document {
//!     id: "doc1".to_string(),
//!     content: "Rust is a systems programming language".to_string(),
//!     embedding: vec![0.1, 0.2, 0.3, 0.4],
//!     metadata: HashMap::new(),
//! };
//! retriever.add_document(doc)?;
//!
//! let query = RetrievalQuery {
//!     text: "systems programming".to_string(),
//!     embedding: vec![0.1, 0.2, 0.3, 0.4],
//!     top_k: 5,
//!     hybrid_alpha: 0.7,
//! };
//! let results = retriever.hybrid_search(&mut query.clone());
//! println!("hits: {}", results.len());
//! # Ok(())
//! # }
//! ```

use std::collections::HashMap;

use thiserror::Error;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors produced by [`DenseRetriever`].
#[derive(Debug, Error, Clone, PartialEq)]
pub enum RetrieverError {
    /// The corpus has reached `config.max_documents`.
    #[error("maximum document capacity ({0}) reached")]
    MaxDocumentsReached(usize),

    /// The supplied embedding has the wrong dimensionality.
    #[error("embedding dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch {
        /// Expected dimension as configured.
        expected: usize,
        /// Actual dimension of the supplied embedding.
        got: usize,
    },

    /// No document with the given id exists.
    #[error("document not found: {0}")]
    DocumentNotFound(String),
}

// ---------------------------------------------------------------------------
// Public domain types
// ---------------------------------------------------------------------------

/// A document stored in the retriever.
#[derive(Debug, Clone)]
pub struct Document {
    /// Unique identifier.
    pub id: String,
    /// Raw text content used for BM25 indexing.
    pub content: String,
    /// Dense embedding vector (length must equal `RetrieverConfig::embedding_dim`).
    pub embedding: Vec<f64>,
    /// Arbitrary key-value metadata.
    pub metadata: HashMap<String, String>,
}

/// Query submitted to [`DenseRetriever::hybrid_search`].
#[derive(Debug, Clone)]
pub struct RetrievalQuery {
    /// Free-text query for BM25 sparse path.
    pub text: String,
    /// Dense embedding for cosine-similarity search.
    pub embedding: Vec<f64>,
    /// Number of top results to return.
    pub top_k: usize,
    /// Interpolation weight: 1.0 = pure dense, 0.0 = pure sparse.
    pub hybrid_alpha: f64,
}

/// A single result returned by [`DenseRetriever::hybrid_search`].
#[derive(Debug, Clone)]
pub struct RetrievalResult {
    /// Document identifier.
    pub doc_id: String,
    /// Normalised dense (cosine) score ∈ [0, 1].
    pub dense_score: f64,
    /// Normalised sparse (BM25) score ∈ [0, 1].
    pub sparse_score: f64,
    /// `alpha * dense_score + (1 − alpha) * sparse_score`.
    pub hybrid_score: f64,
    /// 1-based rank position in the returned list.
    pub rank: usize,
}

/// Runtime statistics snapshot.
#[derive(Debug, Clone)]
pub struct RetrieverStats {
    /// Current number of indexed documents.
    pub document_count: usize,
    /// Total number of hybrid queries executed so far.
    pub total_queries: u64,
    /// Mean document token length.
    pub avg_doc_length: f64,
    /// Number of unique terms in the vocabulary.
    pub vocabulary_size: usize,
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for [`DenseRetriever`].
#[derive(Debug, Clone)]
pub struct RetrieverConfig {
    /// Expected dimensionality of all document / query embeddings.
    pub embedding_dim: usize,
    /// Upper bound on indexed documents.
    pub max_documents: usize,
    /// BM25 term-frequency saturation parameter k₁.
    pub bm25_k1: f64,
    /// BM25 document-length normalisation parameter b.
    pub bm25_b: f64,
}

impl Default for RetrieverConfig {
    fn default() -> Self {
        Self {
            embedding_dim: 128,
            max_documents: 100_000,
            bm25_k1: 1.2,
            bm25_b: 0.75,
        }
    }
}

// ---------------------------------------------------------------------------
// BM25 inverted index
// ---------------------------------------------------------------------------

/// Inverted BM25 index maintained alongside the document store.
#[derive(Debug, Clone, Default)]
pub struct BM25Index {
    /// Token counts per document (parallel to the document vector).
    pub doc_lengths: Vec<usize>,
    /// Posting lists: term → [(doc_index, raw_tf)].
    pub term_freq: HashMap<String, Vec<(usize, f64)>>,
    /// Per-term document frequency (number of documents the term appears in).
    pub doc_freq: HashMap<String, usize>,
    /// Mean document length (re-computed after every mutation).
    pub avg_doc_length: f64,
}

impl BM25Index {
    /// Create an empty index.
    pub fn new() -> Self {
        Self::default()
    }

    /// Tokenise `text` into lower-cased alphabetic tokens.
    pub fn tokenize(text: &str) -> Vec<String> {
        text.split(|c: char| !c.is_alphanumeric())
            .filter(|s| !s.is_empty())
            .map(|s| s.to_lowercase())
            .collect()
    }

    /// Recompute `avg_doc_length` from the current `doc_lengths` slice.
    fn recompute_avg(&mut self) {
        if self.doc_lengths.is_empty() {
            self.avg_doc_length = 0.0;
        } else {
            let total: usize = self.doc_lengths.iter().sum();
            self.avg_doc_length = total as f64 / self.doc_lengths.len() as f64;
        }
    }
}

// ---------------------------------------------------------------------------
// Main retriever
// ---------------------------------------------------------------------------

/// Dense + sparse hybrid retriever.
///
/// See the [module documentation](self) for a full usage example.
pub struct DenseRetriever {
    /// Configuration (immutable after creation).
    pub config: RetrieverConfig,
    /// Ordered collection of indexed documents.
    pub documents: Vec<Document>,
    /// BM25 index mirroring `documents`.
    pub bm25: BM25Index,
    /// Monotonically increasing query counter.
    pub total_queries: u64,
}

impl DenseRetriever {
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    /// Create a new retriever with the supplied `config`.
    pub fn new(config: RetrieverConfig) -> Self {
        Self {
            config,
            documents: Vec::new(),
            bm25: BM25Index::new(),
            total_queries: 0,
        }
    }

    // -----------------------------------------------------------------------
    // Mutation helpers
    // -----------------------------------------------------------------------

    /// Add a document to the index.
    ///
    /// # Errors
    ///
    /// - [`RetrieverError::MaxDocumentsReached`] if the corpus is full.
    /// - [`RetrieverError::DimensionMismatch`] if the embedding length differs from
    ///   `config.embedding_dim`.
    pub fn add_document(&mut self, doc: Document) -> Result<(), RetrieverError> {
        if self.documents.len() >= self.config.max_documents {
            return Err(RetrieverError::MaxDocumentsReached(
                self.config.max_documents,
            ));
        }
        if doc.embedding.len() != self.config.embedding_dim {
            return Err(RetrieverError::DimensionMismatch {
                expected: self.config.embedding_dim,
                got: doc.embedding.len(),
            });
        }

        let doc_idx = self.documents.len();
        let tokens = BM25Index::tokenize(&doc.content);
        let doc_len = tokens.len();

        // Count per-document term frequencies.
        let mut local_tf: HashMap<String, f64> = HashMap::new();
        for token in &tokens {
            *local_tf.entry(token.clone()).or_insert(0.0) += 1.0;
        }

        // Update posting lists and doc-freq.
        for (term, tf) in local_tf {
            let posts = self.bm25.term_freq.entry(term.clone()).or_default();
            posts.push((doc_idx, tf));
            *self.bm25.doc_freq.entry(term).or_insert(0) += 1;
        }

        self.bm25.doc_lengths.push(doc_len);
        self.documents.push(doc);
        self.bm25.recompute_avg();
        Ok(())
    }

    /// Remove the document with the given `id`.
    ///
    /// Returns `true` if the document was found and removed, `false` otherwise.
    /// The BM25 index is fully rebuilt after a successful removal.
    pub fn remove_document(&mut self, id: &str) -> bool {
        let pos = self.documents.iter().position(|d| d.id == id);
        match pos {
            None => false,
            Some(idx) => {
                self.documents.swap_remove(idx);
                self.rebuild_bm25();
                true
            }
        }
    }

    /// Rebuild the BM25 index from scratch using the current document set.
    ///
    /// Called automatically by [`remove_document`](Self::remove_document).
    pub fn rebuild_bm25(&mut self) {
        self.bm25 = BM25Index::new();

        for (doc_idx, doc) in self.documents.iter().enumerate() {
            let tokens = BM25Index::tokenize(&doc.content);
            let doc_len = tokens.len();

            let mut local_tf: HashMap<String, f64> = HashMap::new();
            for token in &tokens {
                *local_tf.entry(token.clone()).or_insert(0.0) += 1.0;
            }

            for (term, tf) in local_tf {
                let posts = self.bm25.term_freq.entry(term.clone()).or_default();
                posts.push((doc_idx, tf));
                *self.bm25.doc_freq.entry(term).or_insert(0) += 1;
            }

            self.bm25.doc_lengths.push(doc_len);
        }

        self.bm25.recompute_avg();
    }

    // -----------------------------------------------------------------------
    // Search primitives
    // -----------------------------------------------------------------------

    /// Return the top-`k` documents by cosine similarity to `query_embedding`.
    ///
    /// Result elements are `(doc_index, cosine_similarity)` sorted descending.
    pub fn dense_search(&self, query_embedding: &[f64], k: usize) -> Vec<(usize, f64)> {
        if self.documents.is_empty() || k == 0 {
            return Vec::new();
        }

        let q_norm = l2_norm(query_embedding);

        let mut scores: Vec<(usize, f64)> = self
            .documents
            .iter()
            .enumerate()
            .map(|(idx, doc)| {
                let sim = cosine_sim_normed(query_embedding, &doc.embedding, q_norm);
                (idx, sim)
            })
            .collect();

        // Partial sort — O(n log k) via full sort for simplicity; corpus is bounded.
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(k);
        scores
    }

    /// Compute the BM25 score for document `doc_idx` given a list of query tokens.
    pub fn bm25_score(&self, doc_idx: usize, query_terms: &[String]) -> f64 {
        let n = self.documents.len() as f64;
        let dl = self.bm25.doc_lengths.get(doc_idx).copied().unwrap_or(0) as f64;
        let avg_dl = self.bm25.avg_doc_length.max(1e-9);
        let k1 = self.config.bm25_k1;
        let b = self.config.bm25_b;

        let mut score = 0.0_f64;

        for term in query_terms {
            let df = self.bm25.doc_freq.get(term).copied().unwrap_or(0) as f64;
            if df == 0.0 {
                continue;
            }
            // Retrieve term frequency for this document.
            let tf = self
                .bm25
                .term_freq
                .get(term)
                .and_then(|posts| {
                    posts
                        .iter()
                        .find(|(idx, _)| *idx == doc_idx)
                        .map(|(_, tf)| *tf)
                })
                .unwrap_or(0.0);

            if tf == 0.0 {
                continue;
            }

            // Robertson-Spärck Jones IDF with additive smoothing.
            let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();
            let tf_norm = tf * (k1 + 1.0) / (tf + k1 * (1.0 - b + b * dl / avg_dl));
            score += idf * tf_norm;
        }

        score
    }

    /// Return the top-`k` documents by BM25 score for `query_text`.
    ///
    /// Result elements are `(doc_index, bm25_score)` sorted descending.
    pub fn sparse_search(&self, query_text: &str, k: usize) -> Vec<(usize, f64)> {
        if self.documents.is_empty() || k == 0 {
            return Vec::new();
        }

        let terms = BM25Index::tokenize(query_text);
        if terms.is_empty() {
            return Vec::new();
        }

        let mut scores: Vec<(usize, f64)> = (0..self.documents.len())
            .map(|idx| (idx, self.bm25_score(idx, &terms)))
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(k);
        scores
    }

    // -----------------------------------------------------------------------
    // Hybrid search
    // -----------------------------------------------------------------------

    /// Execute a hybrid search combining dense and sparse retrieval.
    ///
    /// Both score lists are independently min-max normalised to [0, 1] before
    /// fusion so that the magnitude of each raw signal does not dominate.
    /// The final score is:
    ///
    /// ```text
    /// hybrid_score = alpha * dense_norm + (1 - alpha) * sparse_norm
    /// ```
    ///
    /// Documents that appear only in one list receive a score of 0.0 for the
    /// other component.
    pub fn hybrid_search(&mut self, query: &RetrievalQuery) -> Vec<RetrievalResult> {
        self.total_queries += 1;

        let alpha = query.hybrid_alpha.clamp(0.0, 1.0);
        let k = query.top_k.max(1);

        // Run both retrieval paths with generous candidate sets.
        let candidate_k = (k * 4).max(k + 10).min(self.documents.len().max(1));

        let dense_raw = self.dense_search(&query.embedding, candidate_k);
        let sparse_raw = self.sparse_search(&query.text, candidate_k);

        // Normalise each list independently.
        let dense_norm = min_max_normalise(&dense_raw);
        let sparse_norm = min_max_normalise(&sparse_raw);

        // Merge by document index.
        let mut merged: HashMap<usize, (f64, f64)> = HashMap::new();

        for (doc_idx, score) in &dense_norm {
            merged.entry(*doc_idx).or_insert((0.0, 0.0)).0 = *score;
        }
        for (doc_idx, score) in &sparse_norm {
            merged.entry(*doc_idx).or_insert((0.0, 0.0)).1 = *score;
        }

        // Compute hybrid scores and sort.
        let mut fused: Vec<(usize, f64, f64, f64)> = merged
            .into_iter()
            .map(|(idx, (d, s))| {
                let h = alpha * d + (1.0 - alpha) * s;
                (idx, d, s, h)
            })
            .collect();

        fused.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal));
        fused.truncate(k);

        fused
            .into_iter()
            .enumerate()
            .filter_map(|(rank_idx, (doc_idx, d, s, h))| {
                let doc_id = self.documents.get(doc_idx)?.id.clone();
                Some(RetrievalResult {
                    doc_id,
                    dense_score: d,
                    sparse_score: s,
                    hybrid_score: h,
                    rank: rank_idx + 1,
                })
            })
            .collect()
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /// Look up a document by its string identifier.
    pub fn get_document(&self, id: &str) -> Option<&Document> {
        self.documents.iter().find(|d| d.id == id)
    }

    /// Return the current number of indexed documents.
    pub fn document_count(&self) -> usize {
        self.documents.len()
    }

    /// Return a statistics snapshot.
    pub fn retriever_stats(&self) -> RetrieverStats {
        RetrieverStats {
            document_count: self.documents.len(),
            total_queries: self.total_queries,
            avg_doc_length: self.bm25.avg_doc_length,
            vocabulary_size: self.bm25.doc_freq.len(),
        }
    }
}

// ---------------------------------------------------------------------------
// Internal utilities
// ---------------------------------------------------------------------------

/// Compute the L2 norm of a slice.
fn l2_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Cosine similarity when the query L2-norm is pre-computed.
///
/// Returns 0.0 if either vector is the zero vector.
fn cosine_sim_normed(query: &[f64], doc: &[f64], q_norm: f64) -> f64 {
    if q_norm < 1e-12 {
        return 0.0;
    }
    let d_norm = l2_norm(doc);
    if d_norm < 1e-12 {
        return 0.0;
    }
    let dot: f64 = query.iter().zip(doc.iter()).map(|(q, d)| q * d).sum();
    dot / (q_norm * d_norm)
}

/// Min-max normalise a score list to [0, 1].
///
/// If all scores are identical the result is a list of 1.0 values (to avoid
/// division-by-zero and to preserve all candidates as equally relevant).
fn min_max_normalise(scores: &[(usize, f64)]) -> Vec<(usize, f64)> {
    if scores.is_empty() {
        return Vec::new();
    }
    let min = scores.iter().map(|(_, s)| *s).fold(f64::INFINITY, f64::min);
    let max = scores
        .iter()
        .map(|(_, s)| *s)
        .fold(f64::NEG_INFINITY, f64::max);

    let range = max - min;
    scores
        .iter()
        .map(|(idx, s)| {
            let norm = if range < 1e-12 {
                1.0
            } else {
                (s - min) / range
            };
            (*idx, norm)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::dense_retriever::{
        min_max_normalise, BM25Index, DenseRetriever, Document, RetrievalQuery, RetrieverConfig,
        RetrieverError,
    };

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    fn make_config(dim: usize) -> RetrieverConfig {
        RetrieverConfig {
            embedding_dim: dim,
            max_documents: 100,
            bm25_k1: 1.2,
            bm25_b: 0.75,
        }
    }

    fn make_doc(id: &str, content: &str, emb: Vec<f64>) -> Document {
        Document {
            id: id.to_string(),
            content: content.to_string(),
            embedding: emb,
            metadata: HashMap::new(),
        }
    }

    fn unit_vec(dim: usize, fill: f64) -> Vec<f64> {
        vec![fill; dim]
    }

    // ------------------------------------------------------------------
    // 1. Construction
    // ------------------------------------------------------------------

    #[test]
    fn test_new_retriever_is_empty() {
        let r = DenseRetriever::new(make_config(4));
        assert_eq!(r.document_count(), 0);
        assert_eq!(r.total_queries, 0);
    }

    #[test]
    fn test_default_config_values() {
        let cfg = RetrieverConfig::default();
        assert_eq!(cfg.embedding_dim, 128);
        assert_eq!(cfg.max_documents, 100_000);
        assert!((cfg.bm25_k1 - 1.2).abs() < 1e-9);
        assert!((cfg.bm25_b - 0.75).abs() < 1e-9);
    }

    // ------------------------------------------------------------------
    // 2. add_document
    // ------------------------------------------------------------------

    #[test]
    fn test_add_single_document() {
        let mut r = DenseRetriever::new(make_config(4));
        let doc = make_doc("d1", "hello world", vec![0.1, 0.2, 0.3, 0.4]);
        assert!(r.add_document(doc).is_ok());
        assert_eq!(r.document_count(), 1);
    }

    #[test]
    fn test_add_document_dimension_mismatch() {
        let mut r = DenseRetriever::new(make_config(4));
        let doc = make_doc("d1", "hello", vec![0.1, 0.2]); // wrong dim
        let err = r
            .add_document(doc)
            .expect_err("test: add_document with wrong dimension should return error");
        assert!(matches!(
            err,
            RetrieverError::DimensionMismatch {
                expected: 4,
                got: 2
            }
        ));
    }

    #[test]
    fn test_add_document_capacity_limit() {
        let mut cfg = make_config(2);
        cfg.max_documents = 2;
        let mut r = DenseRetriever::new(cfg);
        r.add_document(make_doc("d1", "a", vec![1.0, 0.0]))
            .expect("test: add_document should succeed");
        r.add_document(make_doc("d2", "b", vec![0.0, 1.0]))
            .expect("test: add_document should succeed");
        let err = r
            .add_document(make_doc("d3", "c", vec![0.5, 0.5]))
            .expect_err("test: add_document beyond capacity should return error");
        assert!(matches!(err, RetrieverError::MaxDocumentsReached(2)));
    }

    #[test]
    fn test_bm25_index_updated_on_add() {
        let mut r = DenseRetriever::new(make_config(2));
        r.add_document(make_doc("d1", "rust is great", vec![1.0, 0.0]))
            .expect("test: add_document should succeed");
        assert!(r.bm25.doc_freq.contains_key("rust"));
        assert!(r.bm25.doc_freq.contains_key("is"));
        assert!(r.bm25.doc_freq.contains_key("great"));
    }

    #[test]
    fn test_avg_doc_length_updated() {
        let mut r = DenseRetriever::new(make_config(2));
        r.add_document(make_doc("d1", "one two three", vec![1.0, 0.0]))
            .expect("test: add_document should succeed");
        r.add_document(make_doc("d2", "four five", vec![0.0, 1.0]))
            .expect("test: add_document should succeed");
        // avg_doc_length = (3 + 2) / 2 = 2.5
        assert!((r.bm25.avg_doc_length - 2.5).abs() < 1e-9);
    }

    // ------------------------------------------------------------------
    // 3. remove_document
    // ------------------------------------------------------------------

    #[test]
    fn test_remove_existing_document() {
        let mut r = DenseRetriever::new(make_config(2));
        r.add_document(make_doc("d1", "hello world", vec![1.0, 0.0]))
            .expect("test: add_document should succeed");
        let removed = r.remove_document("d1");
        assert!(removed);
        assert_eq!(r.document_count(), 0);
    }

    #[test]
    fn test_remove_nonexistent_returns_false() {
        let mut r = DenseRetriever::new(make_config(2));
        r.add_document(make_doc("d1", "hello", vec![1.0, 0.0]))
            .expect("test: add_document should succeed");
        assert!(!r.remove_document("does_not_exist"));
    }

    #[test]
    fn test_bm25_rebuilt_after_remove() {
        let mut r = DenseRetriever::new(make_config(2));
        r.add_document(make_doc("d1", "alpha beta", vec![1.0, 0.0]))
            .expect("test: add_document should succeed");
        r.add_document(make_doc("d2", "alpha gamma", vec![0.0, 1.0]))
            .expect("test: add_document should succeed");
        r.remove_document("d1");
        // "beta" should be gone; "alpha" still present in d2
        assert!(!r.bm25.doc_freq.contains_key("beta"));
        assert!(r.bm25.doc_freq.contains_key("alpha"));
    }

    // ------------------------------------------------------------------
    // 4. BM25Index tokenizer
    // ------------------------------------------------------------------

    #[test]
    fn test_tokenizer_splits_on_whitespace() {
        let tokens = BM25Index::tokenize("hello world foo");
        assert_eq!(tokens, vec!["hello", "world", "foo"]);
    }

    #[test]
    fn test_tokenizer_splits_on_punctuation() {
        let tokens = BM25Index::tokenize("hello, world! foo.");
        assert_eq!(tokens, vec!["hello", "world", "foo"]);
    }

    #[test]
    fn test_tokenizer_lowercases() {
        let tokens = BM25Index::tokenize("Hello WORLD");
        assert_eq!(tokens, vec!["hello", "world"]);
    }

    #[test]
    fn test_tokenizer_empty_string() {
        let tokens = BM25Index::tokenize("");
        assert!(tokens.is_empty());
    }

    #[test]
    fn test_tokenizer_only_punctuation() {
        let tokens = BM25Index::tokenize("!!! ,,, ...");
        assert!(tokens.is_empty());
    }

    // ------------------------------------------------------------------
    // 5. dense_search
    // ------------------------------------------------------------------

    #[test]
    fn test_dense_search_empty_index() {
        let r = DenseRetriever::new(make_config(4));
        let res = r.dense_search(&[1.0, 0.0, 0.0, 0.0], 5);
        assert!(res.is_empty());
    }

    #[test]
    fn test_dense_search_returns_at_most_k() {
        let mut r = DenseRetriever::new(make_config(2));
        for i in 0..10u32 {
            r.add_document(make_doc(
                &i.to_string(),
                "doc",
                vec![i as f64, (10 - i) as f64],
            ))
            .expect("test: add_document should succeed");
        }
        let res = r.dense_search(&[1.0, 0.0], 3);
        assert_eq!(res.len(), 3);
    }

    #[test]
    fn test_dense_search_highest_similarity_first() {
        let mut r = DenseRetriever::new(make_config(2));
        r.add_document(make_doc("d1", "a", vec![1.0, 0.0]))
            .expect("test: add_document should succeed");
        r.add_document(make_doc("d2", "b", vec![0.0, 1.0]))
            .expect("test: add_document should succeed");

        // Query aligned with d1 → d1 should rank first.
        let res = r.dense_search(&[1.0, 0.0], 2);
        assert_eq!(res[0].0, 0); // index 0 = d1
    }

    #[test]
    fn test_dense_search_zero_query_vector() {
        let mut r = DenseRetriever::new(make_config(2));
        r.add_document(make_doc("d1", "x", vec![1.0, 0.0]))
            .expect("test: add_document should succeed");
        let res = r.dense_search(&[0.0, 0.0], 1);
        // Zero vector → similarity = 0 for all docs; still returns entry
        assert_eq!(res.len(), 1);
        assert!((res[0].1).abs() < 1e-9);
    }

    // ------------------------------------------------------------------
    // 6. bm25_score
    // ------------------------------------------------------------------

    #[test]
    fn test_bm25_score_zero_for_missing_term() {
        let mut r = DenseRetriever::new(make_config(2));
        r.add_document(make_doc("d1", "apple banana", vec![1.0, 0.0]))
            .expect("test: add_document should succeed");
        let score = r.bm25_score(0, &["pear".to_string()]);
        assert!((score).abs() < 1e-9);
    }

    #[test]
    fn test_bm25_score_positive_for_matching_term() {
        let mut r = DenseRetriever::new(make_config(2));
        r.add_document(make_doc("d1", "rust programming language", vec![1.0, 0.0]))
            .expect("test: add_document should succeed");
        let score = r.bm25_score(0, &["rust".to_string()]);
        assert!(score > 0.0);
    }

    #[test]
    fn test_bm25_score_increases_with_tf() {
        let mut r = DenseRetriever::new(make_config(2));
        r.add_document(make_doc(
            "d1",
            "rust rust rust other words here",
            vec![1.0, 0.0],
        ))
        .expect("test: add_document should succeed");
        r.add_document(make_doc("d2", "rust other words here", vec![0.0, 1.0]))
            .expect("test: add_document should succeed");
        let s1 = r.bm25_score(0, &["rust".to_string()]);
        let s2 = r.bm25_score(1, &["rust".to_string()]);
        assert!(s1 > s2, "s1={s1} s2={s2}");
    }

    // ------------------------------------------------------------------
    // 7. sparse_search
    // ------------------------------------------------------------------

    #[test]
    fn test_sparse_search_empty_query() {
        let mut r = DenseRetriever::new(make_config(2));
        r.add_document(make_doc("d1", "hello world", vec![1.0, 0.0]))
            .expect("test: add_document should succeed");
        let res = r.sparse_search("", 5);
        assert!(res.is_empty());
    }

    #[test]
    fn test_sparse_search_returns_sorted_desc() {
        let mut r = DenseRetriever::new(make_config(2));
        r.add_document(make_doc("d1", "rust", vec![1.0, 0.0]))
            .expect("test: add_document should succeed");
        r.add_document(make_doc("d2", "rust rust programming", vec![0.0, 1.0]))
            .expect("test: add_document should succeed");
        let res = r.sparse_search("rust", 2);
        assert!(!res.is_empty());
        assert!(res[0].1 >= res[1].1);
    }

    #[test]
    fn test_sparse_search_no_match_returns_all_zero() {
        let mut r = DenseRetriever::new(make_config(2));
        r.add_document(make_doc("d1", "apple banana", vec![1.0, 0.0]))
            .expect("test: add_document should succeed");
        let res = r.sparse_search("zephyr", 1);
        // Document appears with score 0; the list is still non-empty but trivially zeroed.
        if !res.is_empty() {
            assert!((res[0].1).abs() < 1e-9);
        }
    }

    // ------------------------------------------------------------------
    // 8. hybrid_search
    // ------------------------------------------------------------------

    fn make_query(text: &str, emb: Vec<f64>, k: usize, alpha: f64) -> RetrievalQuery {
        RetrievalQuery {
            text: text.to_string(),
            embedding: emb,
            top_k: k,
            hybrid_alpha: alpha,
        }
    }

    #[test]
    fn test_hybrid_search_increments_query_count() {
        let mut r = DenseRetriever::new(make_config(2));
        r.add_document(make_doc("d1", "hello", vec![1.0, 0.0]))
            .expect("test: add_document should succeed");
        let q = make_query("hello", vec![1.0, 0.0], 1, 0.5);
        r.hybrid_search(&q);
        r.hybrid_search(&q);
        assert_eq!(r.total_queries, 2);
    }

    #[test]
    fn test_hybrid_search_returns_at_most_top_k() {
        let mut r = DenseRetriever::new(make_config(2));
        for i in 0..10u32 {
            r.add_document(make_doc(
                &i.to_string(),
                "hello world",
                vec![i as f64 + 1.0, 1.0],
            ))
            .expect("test: add_document should succeed");
        }
        let q = make_query("hello world", vec![1.0, 0.5], 3, 0.5);
        let res = r.hybrid_search(&q);
        assert!(res.len() <= 3);
    }

    #[test]
    fn test_hybrid_search_ranks_start_at_one() {
        let mut r = DenseRetriever::new(make_config(2));
        r.add_document(make_doc("d1", "hello", vec![1.0, 0.0]))
            .expect("test: add_document should succeed");
        r.add_document(make_doc("d2", "world", vec![0.0, 1.0]))
            .expect("test: add_document should succeed");
        let q = make_query("hello world", vec![0.8, 0.2], 2, 0.5);
        let res = r.hybrid_search(&q);
        assert_eq!(res[0].rank, 1);
        if res.len() > 1 {
            assert_eq!(res[1].rank, 2);
        }
    }

    #[test]
    fn test_hybrid_search_pure_dense_alpha_one() {
        let mut r = DenseRetriever::new(make_config(2));
        r.add_document(make_doc("d1", "irrelevant text abc", vec![1.0, 0.0]))
            .expect("test: add_document should succeed");
        r.add_document(make_doc("d2", "irrelevant text xyz", vec![0.0, 1.0]))
            .expect("test: add_document should succeed");

        // Query embedding aligned with d1; alpha=1.0 → pure dense
        let q = make_query("unrelated", vec![1.0, 0.0], 2, 1.0);
        let res = r.hybrid_search(&q);
        // With alpha=1.0 sparse_score is irrelevant; top result must be d1.
        assert_eq!(res[0].doc_id, "d1");
        // And the sparse component of the hybrid score contributes 0.
        // hybrid_score = 1.0 * dense_score + 0.0 * sparse_score
        assert!((res[0].hybrid_score - res[0].dense_score).abs() < 1e-9);
    }

    #[test]
    fn test_hybrid_search_pure_sparse_alpha_zero() {
        let mut r = DenseRetriever::new(make_config(2));
        r.add_document(make_doc("d1", "rust programming systems", vec![0.5, 0.5]))
            .expect("test: add_document should succeed");
        r.add_document(make_doc("d2", "python scripting", vec![0.5, 0.5]))
            .expect("test: add_document should succeed");

        // Both embeddings are identical so dense is a tie; sparse breaks the tie.
        let q = make_query("rust", vec![0.5, 0.5], 2, 0.0);
        let res = r.hybrid_search(&q);
        assert_eq!(res[0].doc_id, "d1", "BM25 should prefer d1 for 'rust'");
    }

    #[test]
    fn test_hybrid_score_formula() {
        // hybrid_score = alpha * dense + (1 - alpha) * sparse
        let alpha = 0.6_f64;
        let dense = 0.8_f64;
        let sparse = 0.5_f64;
        let expected = alpha * dense + (1.0 - alpha) * sparse;
        let computed = alpha * dense + (1.0 - alpha) * sparse;
        assert!((expected - computed).abs() < 1e-12);
    }

    #[test]
    fn test_hybrid_search_empty_index() {
        let mut r = DenseRetriever::new(make_config(2));
        let q = make_query("hello", vec![1.0, 0.0], 5, 0.5);
        let res = r.hybrid_search(&q);
        assert!(res.is_empty());
    }

    #[test]
    fn test_hybrid_search_alpha_clamp_above_one() {
        let mut r = DenseRetriever::new(make_config(2));
        r.add_document(make_doc("d1", "foo", vec![1.0, 0.0]))
            .expect("test: add_document should succeed");
        let q = make_query("foo", vec![1.0, 0.0], 1, 2.5); // alpha > 1.0
        let res = r.hybrid_search(&q);
        assert_eq!(res.len(), 1);
        // Clamped to 1.0 → hybrid_score == dense_score
        assert!((res[0].hybrid_score - res[0].dense_score).abs() < 1e-9);
    }

    #[test]
    fn test_hybrid_search_alpha_clamp_below_zero() {
        let mut r = DenseRetriever::new(make_config(2));
        r.add_document(make_doc("d1", "foo", vec![1.0, 0.0]))
            .expect("test: add_document should succeed");
        let q = make_query("foo", vec![1.0, 0.0], 1, -0.5); // alpha < 0.0
        let res = r.hybrid_search(&q);
        assert_eq!(res.len(), 1);
        // Clamped to 0.0 → hybrid_score == sparse_score
        assert!((res[0].hybrid_score - res[0].sparse_score).abs() < 1e-9);
    }

    // ------------------------------------------------------------------
    // 9. get_document / document_count
    // ------------------------------------------------------------------

    #[test]
    fn test_get_document_found() {
        let mut r = DenseRetriever::new(make_config(2));
        r.add_document(make_doc("d1", "hello", vec![1.0, 0.0]))
            .expect("test: add_document should succeed");
        let doc = r.get_document("d1");
        assert!(doc.is_some());
        assert_eq!(
            doc.expect("test: get_document should return Some after insert")
                .id,
            "d1"
        );
    }

    #[test]
    fn test_get_document_not_found() {
        let r = DenseRetriever::new(make_config(2));
        assert!(r.get_document("missing").is_none());
    }

    #[test]
    fn test_document_count_after_operations() {
        let mut r = DenseRetriever::new(make_config(2));
        assert_eq!(r.document_count(), 0);
        r.add_document(make_doc("d1", "a", vec![1.0, 0.0]))
            .expect("test: add_document should succeed");
        assert_eq!(r.document_count(), 1);
        r.add_document(make_doc("d2", "b", vec![0.0, 1.0]))
            .expect("test: add_document should succeed");
        assert_eq!(r.document_count(), 2);
        r.remove_document("d1");
        assert_eq!(r.document_count(), 1);
    }

    // ------------------------------------------------------------------
    // 10. retriever_stats
    // ------------------------------------------------------------------

    #[test]
    fn test_stats_after_queries() {
        let mut r = DenseRetriever::new(make_config(2));
        r.add_document(make_doc("d1", "word1 word2", vec![1.0, 0.0]))
            .expect("test: add_document should succeed");
        let q = make_query("word1", vec![1.0, 0.0], 1, 0.5);
        r.hybrid_search(&q);
        let stats = r.retriever_stats();
        assert_eq!(stats.document_count, 1);
        assert_eq!(stats.total_queries, 1);
        assert!(stats.vocabulary_size >= 2);
    }

    #[test]
    fn test_stats_vocabulary_size() {
        let mut r = DenseRetriever::new(make_config(2));
        r.add_document(make_doc("d1", "apple banana cherry", vec![1.0, 0.0]))
            .expect("test: add_document should succeed");
        r.add_document(make_doc("d2", "cherry date elderberry", vec![0.0, 1.0]))
            .expect("test: add_document should succeed");
        let stats = r.retriever_stats();
        // unique tokens: apple, banana, cherry, date, elderberry = 5
        assert_eq!(stats.vocabulary_size, 5);
    }

    // ------------------------------------------------------------------
    // 11. min_max_normalise helper
    // ------------------------------------------------------------------

    #[test]
    fn test_min_max_normalise_empty() {
        let out = min_max_normalise(&[]);
        assert!(out.is_empty());
    }

    #[test]
    fn test_min_max_normalise_single_element() {
        let out = min_max_normalise(&[(0, 5.0)]);
        // Single element → range == 0 → normalised to 1.0
        assert!((out[0].1 - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_min_max_normalise_range() {
        let scores = vec![(0, 0.0), (1, 5.0), (2, 10.0)];
        let out = min_max_normalise(&scores);
        assert!((out[0].1 - 0.0).abs() < 1e-9);
        assert!((out[1].1 - 0.5).abs() < 1e-9);
        assert!((out[2].1 - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_min_max_normalise_all_equal() {
        let scores = vec![(0, 3.0), (1, 3.0), (2, 3.0)];
        let out = min_max_normalise(&scores);
        for (_, s) in &out {
            assert!((s - 1.0).abs() < 1e-9);
        }
    }

    // ------------------------------------------------------------------
    // 12. rebuild_bm25 / BM25 consistency
    // ------------------------------------------------------------------

    #[test]
    fn test_rebuild_bm25_idempotent() {
        let mut r = DenseRetriever::new(make_config(2));
        r.add_document(make_doc("d1", "foo bar", vec![1.0, 0.0]))
            .expect("test: add_document should succeed");
        r.add_document(make_doc("d2", "bar baz", vec![0.0, 1.0]))
            .expect("test: add_document should succeed");
        let avg_before = r.bm25.avg_doc_length;
        let vocab_before = r.bm25.doc_freq.len();
        r.rebuild_bm25();
        let avg_after = r.bm25.avg_doc_length;
        let vocab_after = r.bm25.doc_freq.len();
        assert!((avg_before - avg_after).abs() < 1e-9);
        assert_eq!(vocab_before, vocab_after);
    }

    #[test]
    fn test_bm25_doc_freq_counts_documents_not_occurrences() {
        let mut r = DenseRetriever::new(make_config(2));
        // "rust" appears 3 times in d1 but doc_freq should be 1 for d1 alone.
        r.add_document(make_doc("d1", "rust rust rust", vec![1.0, 0.0]))
            .expect("test: add_document should succeed");
        assert_eq!(*r.bm25.doc_freq.get("rust").unwrap_or(&0), 1);
        r.add_document(make_doc("d2", "rust code", vec![0.0, 1.0]))
            .expect("test: add_document should succeed");
        assert_eq!(*r.bm25.doc_freq.get("rust").unwrap_or(&0), 2);
    }

    // ------------------------------------------------------------------
    // 13. Error type
    // ------------------------------------------------------------------

    #[test]
    fn test_error_display_max_documents_reached() {
        let err = RetrieverError::MaxDocumentsReached(50);
        let s = err.to_string();
        assert!(s.contains("50"));
    }

    #[test]
    fn test_error_display_dimension_mismatch() {
        let err = RetrieverError::DimensionMismatch {
            expected: 128,
            got: 64,
        };
        let s = err.to_string();
        assert!(s.contains("128") && s.contains("64"));
    }

    #[test]
    fn test_error_display_not_found() {
        let err = RetrieverError::DocumentNotFound("abc".to_string());
        assert!(err.to_string().contains("abc"));
    }

    // ------------------------------------------------------------------
    // 14. Large-scale smoke test
    // ------------------------------------------------------------------

    #[test]
    fn test_large_corpus_hybrid_search() {
        use std::collections::HashMap;
        let dim = 8_usize;
        let mut r = DenseRetriever::new(RetrieverConfig {
            embedding_dim: dim,
            max_documents: 500,
            bm25_k1: 1.2,
            bm25_b: 0.75,
        });

        let words = ["alpha", "beta", "gamma", "delta", "epsilon"];
        for i in 0..200u32 {
            let word = words[(i as usize) % words.len()];
            let emb: Vec<f64> = (0..dim).map(|j| (i as f64 + j as f64) / 200.0).collect();
            r.add_document(Document {
                id: format!("d{i}"),
                content: format!("{word} document number {i}"),
                embedding: emb,
                metadata: HashMap::new(),
            })
            .expect("test: add_document in large corpus test should succeed");
        }

        let q_emb: Vec<f64> = (0..dim).map(|j| j as f64 / 8.0).collect();
        let q = RetrievalQuery {
            text: "alpha document".to_string(),
            embedding: q_emb,
            top_k: 10,
            hybrid_alpha: 0.5,
        };
        let res = r.hybrid_search(&q);
        assert!(!res.is_empty());
        assert!(res.len() <= 10);

        // Ranks must be strictly ascending and start at 1.
        for (i, hit) in res.iter().enumerate() {
            assert_eq!(hit.rank, i + 1);
        }

        // Scores must be in descending order.
        for w in res.windows(2) {
            assert!(w[0].hybrid_score >= w[1].hybrid_score);
        }
    }

    #[test]
    fn test_unit_embeddings_give_cosine_one() {
        let mut r = DenseRetriever::new(make_config(3));
        r.add_document(make_doc("d1", "x", unit_vec(3, 1.0)))
            .expect("test: add_document should succeed");
        // Cosine of two identical vectors = 1.0
        let res = r.dense_search(&unit_vec(3, 1.0), 1);
        assert!((res[0].1 - 1.0).abs() < 1e-6);
    }
}
