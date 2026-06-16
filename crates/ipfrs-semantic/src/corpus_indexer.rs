//! Inverted-index corpus indexer with BM25 scoring and faceted filtering.
//!
//! [`CorpusIndexer`] builds an in-memory inverted index over a set of
//! [`IndexedDocument`]s, scores queries with BM25 (k1 = 1.5, b = 0.75), and
//! supports arbitrary key-value facet filters.

use std::collections::{HashMap, HashSet};

// ── Error ─────────────────────────────────────────────────────────────────────

/// Errors returned by [`CorpusIndexer`].
#[derive(Debug, Clone, PartialEq)]
pub enum IndexError {
    /// A document with this id was already present in the index.
    DocumentAlreadyExists(String),
    /// No document with this id was found.
    DocumentNotFound(String),
}

impl std::fmt::Display for IndexError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DocumentAlreadyExists(id) => {
                write!(f, "document already exists: {id}")
            }
            Self::DocumentNotFound(id) => {
                write!(f, "document not found: {id}")
            }
        }
    }
}

impl std::error::Error for IndexError {}

// ── Core data types ───────────────────────────────────────────────────────────

/// A document stored in the corpus index.
#[derive(Debug, Clone)]
pub struct IndexedDocument {
    /// Unique identifier for this document.
    pub doc_id: String,
    /// Full text content to be indexed.
    pub content: String,
    /// Arbitrary metadata key-value pairs (author, date, category, tags, …).
    pub fields: HashMap<String, String>,
    /// Optional vector embedding for hybrid search.
    pub embedding: Option<Vec<f64>>,
    /// Unix timestamp (seconds) when the document was indexed.
    pub indexed_at: u64,
}

impl IndexedDocument {
    /// Create a new [`IndexedDocument`] with the given id and content.
    pub fn new(doc_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            doc_id: doc_id.into(),
            content: content.into(),
            fields: HashMap::new(),
            embedding: None,
            indexed_at: 0,
        }
    }

    /// Set a metadata field, returning `self` for chaining.
    pub fn with_field(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.fields.insert(key.into(), value.into());
        self
    }

    /// Attach an embedding vector, returning `self` for chaining.
    pub fn with_embedding(mut self, emb: Vec<f64>) -> Self {
        self.embedding = Some(emb);
        self
    }

    /// Set `indexed_at`, returning `self` for chaining.
    pub fn with_indexed_at(mut self, ts: u64) -> Self {
        self.indexed_at = ts;
        self
    }
}

/// An entry in the postings list for one (term, document) pair.
#[derive(Debug, Clone)]
pub struct PostingEntry {
    /// Document identifier.
    pub doc_id: String,
    /// Number of times the term appears in the document.
    pub term_freq: u32,
    /// Word-index positions of each term occurrence (0-based).
    pub positions: Vec<u32>,
}

/// The core inverted index data structure.
#[derive(Debug, Clone, Default)]
pub struct InvertedIndex {
    /// Maps each term to the list of documents that contain it.
    pub term_to_postings: HashMap<String, Vec<PostingEntry>>,
    /// Maps each term to the number of documents that contain it.
    pub doc_freq: HashMap<String, u32>,
    /// Total number of indexed documents.
    pub total_docs: u32,
    /// Average document length (in tokens, excluding stopwords).
    pub avg_doc_length: f64,
}

/// A ranked result returned by [`CorpusIndexer::search`].
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// Document identifier.
    pub doc_id: String,
    /// BM25 score.
    pub score: f64,
    /// Query terms that matched this document.
    pub matched_terms: Vec<String>,
    /// Optional context snippet around the first matching term.
    pub snippet: Option<String>,
}

/// A key-value filter applied to document metadata fields.
#[derive(Debug, Clone)]
pub struct FacetFilter {
    /// Metadata field name.
    pub field: String,
    /// Required exact value.
    pub value: String,
}

impl FacetFilter {
    /// Construct a new facet filter.
    pub fn new(field: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            field: field.into(),
            value: value.into(),
        }
    }
}

/// A search query submitted to [`CorpusIndexer::search`].
#[derive(Debug, Clone)]
pub struct IndexQuery {
    /// Terms to search for (will be tokenized/normalised internally).
    pub terms: Vec<String>,
    /// Facet filters that must all be satisfied.
    pub facets: Vec<FacetFilter>,
    /// Maximum number of results to return.
    pub top_k: usize,
    /// Minimum BM25 score threshold.
    pub min_score: f64,
    /// If `true`, only documents that contain *all* terms are returned (AND).
    /// If `false`, documents containing *any* term are returned (OR).
    pub require_all_terms: bool,
}

impl IndexQuery {
    /// Construct a simple keyword query.
    pub fn new(terms: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            terms: terms.into_iter().map(|t| t.into()).collect(),
            facets: Vec::new(),
            top_k: 10,
            min_score: 0.0,
            require_all_terms: false,
        }
    }
}

/// Aggregate index statistics.
#[derive(Debug, Clone)]
pub struct IndexStats {
    /// Number of documents currently in the index.
    pub doc_count: usize,
    /// Number of unique terms in the vocabulary.
    pub vocabulary_size: usize,
    /// Average document length (tokens after stopword removal).
    pub avg_doc_length: f64,
    /// Total number of (term, document) posting entries.
    pub total_postings: usize,
}

// ── Tokenisation helpers ──────────────────────────────────────────────────────

/// Tokenise `text`: lowercase, split on non-alphanumeric, remove stopwords, drop empty tokens.
fn tokenize(text: &str, stopwords: &HashSet<String>) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter_map(|tok| {
            let lower = tok.to_lowercase();
            if lower.is_empty() || stopwords.contains(&lower) {
                None
            } else {
                Some(lower)
            }
        })
        .collect()
}

/// Return the default English stopword set.
fn default_stopwords() -> HashSet<String> {
    [
        "the", "a", "an", "is", "it", "in", "on", "at", "to", "of", "and", "or", "but", "for",
        "with", "this", "that", "are", "was", "were", "be", "been", "have", "has", "had", "do",
        "does", "did", "will", "would", "could", "should",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

// ── BM25 ──────────────────────────────────────────────────────────────────────

const BM25_K1: f64 = 1.5;
const BM25_B: f64 = 0.75;

/// Compute the IDF component of BM25.
///
/// `idf(t) = ln((N - df + 0.5) / (df + 0.5) + 1)`
#[inline]
fn bm25_idf(n: u32, df: u32) -> f64 {
    let n_f = n as f64;
    let df_f = df as f64;
    ((n_f - df_f + 0.5) / (df_f + 0.5) + 1.0).ln()
}

/// Compute the TF saturation component of BM25.
///
/// `tf_sat(t, d) = tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avg_dl))`
#[inline]
fn bm25_tf(tf: u32, doc_len: u32, avg_dl: f64) -> f64 {
    let tf_f = tf as f64;
    let dl_f = doc_len as f64;
    tf_f * (BM25_K1 + 1.0) / (tf_f + BM25_K1 * (1.0 - BM25_B + BM25_B * dl_f / avg_dl.max(1.0)))
}

// ── CorpusIndexer ─────────────────────────────────────────────────────────────

/// In-memory inverted index with BM25 scoring and faceted filtering.
///
/// # Example
///
/// ```rust
/// use ipfrs_semantic::corpus_indexer::{CorpusIndexer, IndexedDocument, IndexQuery};
///
/// let mut indexer = CorpusIndexer::new();
///
/// let doc = IndexedDocument::new("doc1", "Rust is a systems programming language");
/// indexer.add_document(doc).unwrap();
///
/// let query = IndexQuery::new(["rust", "programming"]);
/// let results = indexer.search(&query);
/// assert!(!results.is_empty());
/// assert_eq!(results[0].doc_id, "doc1");
/// ```
pub struct CorpusIndexer {
    /// Core inverted index.
    pub index: InvertedIndex,
    /// Original documents keyed by doc_id.
    pub documents: HashMap<String, IndexedDocument>,
    /// Set of words that are ignored during tokenisation.
    pub stopwords: HashSet<String>,
    /// Per-document token length cache (tokens after stopword removal).
    doc_lengths: HashMap<String, u32>,
}

impl Default for CorpusIndexer {
    fn default() -> Self {
        Self::new()
    }
}

impl CorpusIndexer {
    // ── Construction ──────────────────────────────────────────────────────

    /// Create a new [`CorpusIndexer`] initialised with default English stopwords.
    pub fn new() -> Self {
        Self {
            index: InvertedIndex::default(),
            documents: HashMap::new(),
            stopwords: default_stopwords(),
            doc_lengths: HashMap::new(),
        }
    }

    // ── Mutation ──────────────────────────────────────────────────────────

    /// Index a new document.
    ///
    /// # Errors
    ///
    /// Returns [`IndexError::DocumentAlreadyExists`] if a document with the
    /// same `doc_id` has already been indexed.
    pub fn add_document(&mut self, doc: IndexedDocument) -> Result<(), IndexError> {
        if self.documents.contains_key(&doc.doc_id) {
            return Err(IndexError::DocumentAlreadyExists(doc.doc_id.clone()));
        }
        self.index_document(&doc);
        self.documents.insert(doc.doc_id.clone(), doc);
        self.refresh_avg_doc_length();
        Ok(())
    }

    /// Remove a document from the index.
    ///
    /// # Errors
    ///
    /// Returns [`IndexError::DocumentNotFound`] if no document with that id exists.
    pub fn remove_document(&mut self, doc_id: &str) -> Result<(), IndexError> {
        if !self.documents.contains_key(doc_id) {
            return Err(IndexError::DocumentNotFound(doc_id.to_string()));
        }
        self.unindex_document(doc_id);
        self.documents.remove(doc_id);
        self.doc_lengths.remove(doc_id);
        self.refresh_avg_doc_length();
        Ok(())
    }

    /// Replace an existing document with an updated version.
    ///
    /// Internally performs `remove_document` followed by `add_document`.
    ///
    /// # Errors
    ///
    /// Returns [`IndexError::DocumentNotFound`] if the document did not exist.
    pub fn update_document(&mut self, doc: IndexedDocument) -> Result<(), IndexError> {
        let id = doc.doc_id.clone();
        self.remove_document(&id)?;
        // After removal the id no longer exists, so add_document will succeed.
        self.add_document(doc)
            .map_err(|_| IndexError::DocumentNotFound(id))
    }

    // ── Search ────────────────────────────────────────────────────────────

    /// Execute a [`IndexQuery`] against the index.
    ///
    /// Steps:
    /// 1. Normalise query terms (lowercase + stopword removal).
    /// 2. Find candidate documents using the inverted index.
    /// 3. Apply AND / OR logic.
    /// 4. Apply facet filters.
    /// 5. Compute BM25 scores.
    /// 6. Apply `min_score` threshold.
    /// 7. Attach snippets.
    /// 8. Sort by score descending and return `top_k` results.
    pub fn search(&self, query: &IndexQuery) -> Vec<SearchResult> {
        if self.index.total_docs == 0 || query.terms.is_empty() {
            return Vec::new();
        }

        // Normalise query terms — remove stopwords and lowercase.
        let norm_terms: Vec<String> = query
            .terms
            .iter()
            .filter_map(|t| {
                let lower = t.to_lowercase();
                if lower.is_empty() || self.stopwords.contains(&lower) {
                    None
                } else {
                    Some(lower)
                }
            })
            .collect();

        if norm_terms.is_empty() {
            return Vec::new();
        }

        let avg_dl = self.index.avg_doc_length;
        let n = self.index.total_docs;

        // For each (doc_id) accumulate BM25 score and track matched terms.
        let mut scores: HashMap<&str, (f64, Vec<String>)> = HashMap::new();

        for term in &norm_terms {
            let df = self.index.doc_freq.get(term).copied().unwrap_or(0);
            if df == 0 {
                continue;
            }
            let idf = bm25_idf(n, df);

            if let Some(postings) = self.index.term_to_postings.get(term) {
                for entry in postings {
                    let dl = self
                        .doc_lengths
                        .get(entry.doc_id.as_str())
                        .copied()
                        .unwrap_or(1);
                    let tf_sat = bm25_tf(entry.term_freq, dl, avg_dl);
                    let contribution = idf * tf_sat;

                    let slot = scores
                        .entry(entry.doc_id.as_str())
                        .or_insert((0.0, Vec::new()));
                    slot.0 += contribution;
                    if !slot.1.contains(term) {
                        slot.1.push(term.clone());
                    }
                }
            }
        }

        // AND filter: require all normalised terms to match.
        if query.require_all_terms {
            scores.retain(|_, (_, matched)| norm_terms.iter().all(|t| matched.contains(t)));
        }

        // Facet filtering.
        if !query.facets.is_empty() {
            scores.retain(|doc_id, _| {
                if let Some(doc) = self.documents.get(*doc_id) {
                    query.facets.iter().all(|f| {
                        doc.fields
                            .get(&f.field)
                            .map(|v| v == &f.value)
                            .unwrap_or(false)
                    })
                } else {
                    false
                }
            });
        }

        // Apply min_score, build results, attach snippets.
        let mut results: Vec<SearchResult> = scores
            .into_iter()
            .filter(|(_, (score, _))| *score >= query.min_score)
            .map(|(doc_id, (score, matched_terms))| {
                let snippet = self.snippet(doc_id, &matched_terms, 8);
                SearchResult {
                    doc_id: doc_id.to_string(),
                    score,
                    matched_terms,
                    snippet,
                }
            })
            .collect();

        // Sort by score descending, then doc_id ascending for determinism.
        results.sort_unstable_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.doc_id.cmp(&b.doc_id))
        });

        results.truncate(query.top_k);
        results
    }

    // ── Inspection ────────────────────────────────────────────────────────

    /// Return a context snippet from `doc_id` centered on the first occurrence
    /// of any of `terms`.
    ///
    /// `window` is the number of words on each side of the matched word.
    /// Returns `None` if the document is not found or no term matches.
    pub fn snippet(&self, doc_id: &str, terms: &[String], window: usize) -> Option<String> {
        let doc = self.documents.get(doc_id)?;
        let tokens: Vec<&str> = doc.content.split_whitespace().collect();

        // Find the first position where any normalised query term appears.
        let lower_terms: Vec<String> = terms.iter().map(|t| t.to_lowercase()).collect();

        let pos = tokens.iter().position(|&w| {
            let lw = w.to_lowercase();
            // Strip punctuation from word edges for matching purposes.
            let stripped: String = lw.chars().filter(|c| c.is_alphanumeric()).collect();
            lower_terms.iter().any(|t| t == &stripped || t == &lw)
        })?;

        let start = pos.saturating_sub(window);
        let end = (pos + window + 1).min(tokens.len());
        Some(tokens[start..end].join(" "))
    }

    /// Term frequency of `term` inside `doc_id` (0 if not found).
    pub fn term_frequency(&self, term: &str, doc_id: &str) -> u32 {
        let lower = term.to_lowercase();
        self.index
            .term_to_postings
            .get(&lower)
            .and_then(|postings| postings.iter().find(|e| e.doc_id == doc_id))
            .map(|e| e.term_freq)
            .unwrap_or(0)
    }

    /// Number of documents that contain `term`.
    pub fn document_frequency(&self, term: &str) -> u32 {
        let lower = term.to_lowercase();
        self.index.doc_freq.get(&lower).copied().unwrap_or(0)
    }

    /// Number of documents currently indexed.
    pub fn doc_count(&self) -> usize {
        self.documents.len()
    }

    /// Number of distinct terms in the vocabulary.
    pub fn vocabulary_size(&self) -> usize {
        self.index.term_to_postings.len()
    }

    /// Top `n` terms by document frequency (descending).
    ///
    /// Ties are broken alphabetically.
    pub fn top_terms(&self, n: usize) -> Vec<(String, u32)> {
        let mut pairs: Vec<(String, u32)> = self
            .index
            .doc_freq
            .iter()
            .map(|(k, &v)| (k.clone(), v))
            .collect();
        pairs.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        pairs.truncate(n);
        pairs
    }

    /// Return the doc_ids of all documents that have `fields[field] == value`.
    pub fn documents_with_field<'a>(&'a self, field: &str, value: &str) -> Vec<&'a str> {
        self.documents
            .values()
            .filter(|doc| {
                doc.fields
                    .get(field)
                    .map(|v| v.as_str() == value)
                    .unwrap_or(false)
            })
            .map(|doc| doc.doc_id.as_str())
            .collect()
    }

    /// Aggregate statistics about the current state of the index.
    pub fn stats(&self) -> IndexStats {
        let total_postings: usize = self.index.term_to_postings.values().map(|v| v.len()).sum();
        IndexStats {
            doc_count: self.doc_count(),
            vocabulary_size: self.vocabulary_size(),
            avg_doc_length: self.index.avg_doc_length,
            total_postings,
        }
    }

    // ── Private helpers ───────────────────────────────────────────────────

    /// Add `doc` to the inverted index and length cache.
    fn index_document(&mut self, doc: &IndexedDocument) {
        let tokens = tokenize(&doc.content, &self.stopwords);
        let doc_len = tokens.len() as u32;
        self.doc_lengths.insert(doc.doc_id.clone(), doc_len);
        self.index.total_docs += 1;

        // Build term → positions map for this document.
        let mut term_positions: HashMap<String, Vec<u32>> = HashMap::new();
        for (pos, token) in tokens.iter().enumerate() {
            term_positions
                .entry(token.clone())
                .or_default()
                .push(pos as u32);
        }

        for (term, positions) in term_positions {
            let tf = positions.len() as u32;
            // Update postings list.
            self.index
                .term_to_postings
                .entry(term.clone())
                .or_default()
                .push(PostingEntry {
                    doc_id: doc.doc_id.clone(),
                    term_freq: tf,
                    positions,
                });
            // Update document frequency.
            *self.index.doc_freq.entry(term).or_insert(0) += 1;
        }
    }

    /// Remove `doc_id` from all postings lists, cleaning up empty entries.
    fn unindex_document(&mut self, doc_id: &str) {
        // Collect affected terms first to avoid borrowing issues.
        let affected_terms: Vec<String> = self
            .index
            .term_to_postings
            .iter()
            .filter(|(_, postings)| postings.iter().any(|e| e.doc_id == doc_id))
            .map(|(term, _)| term.clone())
            .collect();

        for term in affected_terms {
            if let Some(postings) = self.index.term_to_postings.get_mut(&term) {
                postings.retain(|e| e.doc_id != doc_id);
                if postings.is_empty() {
                    self.index.term_to_postings.remove(&term);
                    self.index.doc_freq.remove(&term);
                } else {
                    // Decrement df.
                    if let Some(df) = self.index.doc_freq.get_mut(&term) {
                        *df = df.saturating_sub(1);
                    }
                }
            }
        }
        self.index.total_docs = self.index.total_docs.saturating_sub(1);
    }

    /// Recalculate and store `avg_doc_length` from `doc_lengths`.
    fn refresh_avg_doc_length(&mut self) {
        if self.doc_lengths.is_empty() {
            self.index.avg_doc_length = 0.0;
        } else {
            let total: u64 = self.doc_lengths.values().map(|&v| v as u64).sum();
            self.index.avg_doc_length = total as f64 / self.doc_lengths.len() as f64;
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::{
        bm25_idf, bm25_tf, default_stopwords, tokenize, CorpusIndexer, FacetFilter, IndexError,
        IndexQuery, IndexedDocument, PostingEntry,
    };

    // ── Helper ──────────────────────────────────────────────────────────────

    fn make_doc(id: &str, content: &str) -> IndexedDocument {
        IndexedDocument::new(id, content)
    }

    // ── Tokenisation ────────────────────────────────────────────────────────

    #[test]
    fn tokenize_basic() {
        let sw = default_stopwords();
        let tokens = tokenize("Hello, world! This is a test.", &sw);
        // "this", "is", "a" are stopwords
        assert!(tokens.contains(&"hello".to_string()));
        assert!(tokens.contains(&"world".to_string()));
        assert!(tokens.contains(&"test".to_string()));
        assert!(!tokens.contains(&"this".to_string()));
        assert!(!tokens.contains(&"is".to_string()));
    }

    #[test]
    fn tokenize_lowercase() {
        let sw = default_stopwords();
        let tokens = tokenize("RUST Programming", &sw);
        assert!(tokens.contains(&"rust".to_string()));
        assert!(tokens.contains(&"programming".to_string()));
    }

    #[test]
    fn tokenize_empty_string() {
        let sw = default_stopwords();
        let tokens = tokenize("", &sw);
        assert!(tokens.is_empty());
    }

    #[test]
    fn tokenize_only_stopwords() {
        let sw = default_stopwords();
        let tokens = tokenize("the and or but", &sw);
        assert!(tokens.is_empty());
    }

    #[test]
    fn tokenize_numbers_are_kept() {
        let sw = default_stopwords();
        let tokens = tokenize("version 2024 release", &sw);
        assert!(tokens.contains(&"2024".to_string()));
    }

    // ── BM25 maths ──────────────────────────────────────────────────────────

    #[test]
    fn bm25_idf_positive_for_rare_term() {
        // A term that appears in 1 out of 100 docs should have positive IDF.
        let idf = bm25_idf(100, 1);
        assert!(idf > 0.0);
    }

    #[test]
    fn bm25_idf_decreases_with_df() {
        let idf_rare = bm25_idf(100, 1);
        let idf_common = bm25_idf(100, 50);
        assert!(idf_rare > idf_common);
    }

    #[test]
    fn bm25_tf_increases_with_raw_tf() {
        let tf1 = bm25_tf(1, 100, 100.0);
        let tf5 = bm25_tf(5, 100, 100.0);
        assert!(tf5 > tf1);
    }

    #[test]
    fn bm25_tf_saturates() {
        // Large TF should not grow unboundedly.
        let tf_large = bm25_tf(1_000_000, 100, 100.0);
        assert!(tf_large < 10.0, "BM25 TF saturation failed: {tf_large}");
    }

    // ── add_document ────────────────────────────────────────────────────────

    #[test]
    fn add_document_increases_doc_count() {
        let mut idx = CorpusIndexer::new();
        idx.add_document(make_doc("d1", "rust is great"))
            .expect("test: add_document should succeed");
        assert_eq!(idx.doc_count(), 1);
    }

    #[test]
    fn add_document_duplicate_returns_error() {
        let mut idx = CorpusIndexer::new();
        idx.add_document(make_doc("d1", "content"))
            .expect("test: add_document should succeed");
        let err = idx
            .add_document(make_doc("d1", "other"))
            .expect_err("test: duplicate add_document should return error");
        assert_eq!(err, IndexError::DocumentAlreadyExists("d1".to_string()));
    }

    #[test]
    fn add_document_builds_postings() {
        let mut idx = CorpusIndexer::new();
        idx.add_document(make_doc("d1", "rust programming language"))
            .expect("test: add_document should succeed");
        assert!(idx.index.term_to_postings.contains_key("rust"));
        assert!(idx.index.term_to_postings.contains_key("programming"));
        assert!(idx.index.term_to_postings.contains_key("language"));
    }

    #[test]
    fn add_document_updates_avg_doc_length() {
        let mut idx = CorpusIndexer::new();
        idx.add_document(make_doc("d1", "rust programming"))
            .expect("test: add_document should succeed");
        assert!(idx.index.avg_doc_length > 0.0);
    }

    #[test]
    fn add_multiple_documents() {
        let mut idx = CorpusIndexer::new();
        for i in 0..5 {
            idx.add_document(make_doc(&format!("d{i}"), &format!("document {i} content")))
                .expect("test: add_document should succeed");
        }
        assert_eq!(idx.doc_count(), 5);
    }

    // ── remove_document ─────────────────────────────────────────────────────

    #[test]
    fn remove_document_decreases_count() {
        let mut idx = CorpusIndexer::new();
        idx.add_document(make_doc("d1", "rust programming"))
            .expect("test: add_document should succeed");
        idx.remove_document("d1")
            .expect("test: remove_document should succeed");
        assert_eq!(idx.doc_count(), 0);
    }

    #[test]
    fn remove_document_not_found_returns_error() {
        let mut idx = CorpusIndexer::new();
        let err = idx
            .remove_document("missing")
            .expect_err("test: remove_document should return error for missing id");
        assert_eq!(err, IndexError::DocumentNotFound("missing".to_string()));
    }

    #[test]
    fn remove_document_cleans_postings() {
        let mut idx = CorpusIndexer::new();
        idx.add_document(make_doc("d1", "rust programming"))
            .expect("test: add_document should succeed");
        idx.remove_document("d1")
            .expect("test: remove_document should succeed");
        // Posting lists should be gone.
        assert!(!idx.index.term_to_postings.contains_key("rust"));
    }

    #[test]
    fn remove_one_of_two_docs_keeps_shared_term() {
        let mut idx = CorpusIndexer::new();
        idx.add_document(make_doc("d1", "rust programming language"))
            .expect("test: add_document should succeed");
        idx.add_document(make_doc("d2", "rust systems"))
            .expect("test: add_document should succeed");
        idx.remove_document("d1")
            .expect("test: remove_document should succeed");
        // "rust" is still present via d2.
        assert!(idx.index.term_to_postings.contains_key("rust"));
        assert_eq!(idx.document_frequency("rust"), 1);
    }

    // ── update_document ─────────────────────────────────────────────────────

    #[test]
    fn update_document_changes_content() {
        let mut idx = CorpusIndexer::new();
        idx.add_document(make_doc("d1", "old content words"))
            .expect("test: add_document should succeed");
        let updated = make_doc("d1", "brand new text");
        idx.update_document(updated)
            .expect("test: update_document should succeed");
        assert_eq!(idx.doc_count(), 1);
        assert!(idx.index.term_to_postings.contains_key("brand"));
        assert!(!idx.index.term_to_postings.contains_key("old"));
    }

    #[test]
    fn update_document_not_found_returns_error() {
        let mut idx = CorpusIndexer::new();
        let err = idx
            .update_document(make_doc("ghost", "content"))
            .expect_err("test: update_document should return error for nonexistent document");
        assert_eq!(err, IndexError::DocumentNotFound("ghost".to_string()));
    }

    // ── term_frequency / document_frequency ─────────────────────────────────

    #[test]
    fn term_frequency_counts_correctly() {
        let mut idx = CorpusIndexer::new();
        idx.add_document(make_doc("d1", "rust rust rust programming"))
            .expect("test: add_document should succeed");
        assert_eq!(idx.term_frequency("rust", "d1"), 3);
    }

    #[test]
    fn term_frequency_missing_term() {
        let mut idx = CorpusIndexer::new();
        idx.add_document(make_doc("d1", "hello world"))
            .expect("test: add_document should succeed");
        assert_eq!(idx.term_frequency("rust", "d1"), 0);
    }

    #[test]
    fn document_frequency_single() {
        let mut idx = CorpusIndexer::new();
        idx.add_document(make_doc("d1", "rust programming"))
            .expect("test: add_document should succeed");
        assert_eq!(idx.document_frequency("rust"), 1);
    }

    #[test]
    fn document_frequency_multiple() {
        let mut idx = CorpusIndexer::new();
        idx.add_document(make_doc("d1", "rust programming"))
            .expect("test: add_document should succeed");
        idx.add_document(make_doc("d2", "rust systems"))
            .expect("test: add_document should succeed");
        assert_eq!(idx.document_frequency("rust"), 2);
    }

    #[test]
    fn document_frequency_zero_for_absent() {
        let idx = CorpusIndexer::new();
        assert_eq!(idx.document_frequency("nonexistent"), 0);
    }

    // ── vocabulary_size / doc_count ─────────────────────────────────────────

    #[test]
    fn vocabulary_size_grows_with_new_terms() {
        let mut idx = CorpusIndexer::new();
        idx.add_document(make_doc("d1", "alpha beta gamma"))
            .expect("test: add_document should succeed");
        assert_eq!(idx.vocabulary_size(), 3);
    }

    #[test]
    fn doc_count_empty() {
        let idx = CorpusIndexer::new();
        assert_eq!(idx.doc_count(), 0);
    }

    // ── top_terms ───────────────────────────────────────────────────────────

    #[test]
    fn top_terms_returns_by_df_descending() {
        let mut idx = CorpusIndexer::new();
        // "rust" appears in d1, d2, d3 → df = 3
        // "programming" appears only in d1 → df = 1
        idx.add_document(make_doc("d1", "rust programming"))
            .expect("test: add_document should succeed");
        idx.add_document(make_doc("d2", "rust systems"))
            .expect("test: add_document should succeed");
        idx.add_document(make_doc("d3", "rust language"))
            .expect("test: add_document should succeed");
        let top = idx.top_terms(1);
        assert_eq!(top[0].0, "rust");
        assert_eq!(top[0].1, 3);
    }

    #[test]
    fn top_terms_bounded_by_n() {
        let mut idx = CorpusIndexer::new();
        idx.add_document(make_doc("d1", "alpha beta gamma delta epsilon"))
            .expect("test: add_document should succeed");
        let top = idx.top_terms(3);
        assert_eq!(top.len(), 3);
    }

    // ── documents_with_field ────────────────────────────────────────────────

    #[test]
    fn documents_with_field_matches_exactly() {
        let mut idx = CorpusIndexer::new();
        let doc = make_doc("d1", "content").with_field("author", "alice");
        idx.add_document(doc)
            .expect("test: add_document should succeed");
        let results = idx.documents_with_field("author", "alice");
        assert_eq!(results, vec!["d1"]);
    }

    #[test]
    fn documents_with_field_no_match() {
        let mut idx = CorpusIndexer::new();
        let doc = make_doc("d1", "content").with_field("author", "alice");
        idx.add_document(doc)
            .expect("test: add_document should succeed");
        let results = idx.documents_with_field("author", "bob");
        assert!(results.is_empty());
    }

    #[test]
    fn documents_with_field_multiple_matches() {
        let mut idx = CorpusIndexer::new();
        for (id, author) in [("d1", "alice"), ("d2", "alice"), ("d3", "bob")] {
            idx.add_document(make_doc(id, "text").with_field("author", author))
                .expect("test: add_document should succeed");
        }
        let mut results = idx.documents_with_field("author", "alice");
        results.sort_unstable();
        assert_eq!(results, vec!["d1", "d2"]);
    }

    // ── snippet ─────────────────────────────────────────────────────────────

    #[test]
    fn snippet_returns_context_around_term() {
        let mut idx = CorpusIndexer::new();
        idx.add_document(make_doc(
            "d1",
            "the quick brown fox jumps over the lazy dog",
        ))
        .expect("test: add_document should succeed");
        let terms = vec!["fox".to_string()];
        let snip = idx
            .snippet("d1", &terms, 2)
            .expect("test: snippet should return Some for present term");
        assert!(snip.contains("fox"), "snippet={snip}");
    }

    #[test]
    fn snippet_returns_none_for_missing_doc() {
        let idx = CorpusIndexer::new();
        let result = idx.snippet("missing", &["term".to_string()], 3);
        assert!(result.is_none());
    }

    #[test]
    fn snippet_returns_none_for_absent_term() {
        let mut idx = CorpusIndexer::new();
        idx.add_document(make_doc("d1", "hello world"))
            .expect("test: add_document should succeed");
        let result = idx.snippet("d1", &["zzz".to_string()], 3);
        assert!(result.is_none());
    }

    #[test]
    fn snippet_window_clamps_at_boundaries() {
        let mut idx = CorpusIndexer::new();
        idx.add_document(make_doc("d1", "rust is fast"))
            .expect("test: add_document should succeed");
        // "rust" is the first word — window should not panic.
        let snip = idx
            .snippet("d1", &["rust".to_string()], 5)
            .expect("test: snippet should return Some when term is at doc boundary");
        assert!(snip.contains("rust"));
    }

    // ── search ──────────────────────────────────────────────────────────────

    #[test]
    fn search_returns_matching_doc() {
        let mut idx = CorpusIndexer::new();
        idx.add_document(make_doc("d1", "rust systems programming language"))
            .expect("test: add_document should succeed");
        idx.add_document(make_doc("d2", "python machine learning library"))
            .expect("test: add_document should succeed");

        let q = IndexQuery::new(["rust"]);
        let results = idx.search(&q);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].doc_id, "d1");
    }

    #[test]
    fn search_empty_index_returns_empty() {
        let idx = CorpusIndexer::new();
        let q = IndexQuery::new(["rust"]);
        assert!(idx.search(&q).is_empty());
    }

    #[test]
    fn search_no_match_returns_empty() {
        let mut idx = CorpusIndexer::new();
        idx.add_document(make_doc("d1", "hello world"))
            .expect("test: add_document should succeed");
        let q = IndexQuery::new(["rust"]);
        assert!(idx.search(&q).is_empty());
    }

    #[test]
    fn search_or_mode_returns_partial_matches() {
        let mut idx = CorpusIndexer::new();
        idx.add_document(make_doc("d1", "rust language"))
            .expect("test: add_document should succeed");
        idx.add_document(make_doc("d2", "python language"))
            .expect("test: add_document should succeed");
        idx.add_document(make_doc("d3", "java language"))
            .expect("test: add_document should succeed");

        let mut q = IndexQuery::new(["rust", "python"]);
        q.require_all_terms = false;
        let results = idx.search(&q);
        let ids: Vec<&str> = results.iter().map(|r| r.doc_id.as_str()).collect();
        assert!(ids.contains(&"d1"));
        assert!(ids.contains(&"d2"));
        assert!(!ids.contains(&"d3"));
    }

    #[test]
    fn search_and_mode_requires_all_terms() {
        let mut idx = CorpusIndexer::new();
        idx.add_document(make_doc("d1", "rust systems fast"))
            .expect("test: add_document should succeed");
        idx.add_document(make_doc("d2", "rust language"))
            .expect("test: add_document should succeed");

        let mut q = IndexQuery::new(["rust", "systems"]);
        q.require_all_terms = true;
        let results = idx.search(&q);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].doc_id, "d1");
    }

    #[test]
    fn search_top_k_limits_results() {
        let mut idx = CorpusIndexer::new();
        for i in 0..10 {
            idx.add_document(make_doc(
                &format!("d{i}"),
                &format!("rust document number {i}"),
            ))
            .expect("test: add_document should succeed");
        }
        let mut q = IndexQuery::new(["rust"]);
        q.top_k = 3;
        let results = idx.search(&q);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn search_sorted_by_score_desc() {
        let mut idx = CorpusIndexer::new();
        // d1: "rust" appears 3 times → higher TF.
        idx.add_document(make_doc("d1", "rust rust rust systems"))
            .expect("test: add_document should succeed");
        // d2: "rust" appears 1 time.
        idx.add_document(make_doc("d2", "rust language"))
            .expect("test: add_document should succeed");

        let q = IndexQuery::new(["rust"]);
        let results = idx.search(&q);
        assert!(results[0].score >= results[1].score);
    }

    #[test]
    fn search_min_score_filters_low_scores() {
        let mut idx = CorpusIndexer::new();
        idx.add_document(make_doc("d1", "rust rust rust"))
            .expect("test: add_document should succeed");
        idx.add_document(make_doc("d2", "rust language"))
            .expect("test: add_document should succeed");

        let mut q = IndexQuery::new(["rust"]);
        q.min_score = 999.0; // unreachably high
        let results = idx.search(&q);
        assert!(results.is_empty());
    }

    #[test]
    fn search_facet_filter_applied() {
        let mut idx = CorpusIndexer::new();
        idx.add_document(make_doc("d1", "rust systems").with_field("lang", "rust"))
            .expect("test: add_document should succeed");
        idx.add_document(make_doc("d2", "python ml").with_field("lang", "python"))
            .expect("test: add_document should succeed");

        let mut q = IndexQuery::new(["rust", "python", "systems", "ml"]);
        q.facets.push(FacetFilter::new("lang", "rust"));
        let results = idx.search(&q);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].doc_id, "d1");
    }

    #[test]
    fn search_stopword_only_query_returns_empty() {
        let mut idx = CorpusIndexer::new();
        idx.add_document(make_doc("d1", "rust programming"))
            .expect("test: add_document should succeed");
        let q = IndexQuery::new(["the", "and", "or"]);
        assert!(idx.search(&q).is_empty());
    }

    // ── stats ────────────────────────────────────────────────────────────────

    #[test]
    fn stats_reflect_current_state() {
        let mut idx = CorpusIndexer::new();
        idx.add_document(make_doc("d1", "alpha beta gamma"))
            .expect("test: add_document should succeed");
        let s = idx.stats();
        assert_eq!(s.doc_count, 1);
        assert_eq!(s.vocabulary_size, 3);
        assert!(s.avg_doc_length > 0.0);
        assert!(s.total_postings > 0);
    }

    #[test]
    fn stats_after_removal_decrements_correctly() {
        let mut idx = CorpusIndexer::new();
        idx.add_document(make_doc("d1", "alpha beta"))
            .expect("test: add_document should succeed");
        idx.add_document(make_doc("d2", "gamma delta"))
            .expect("test: add_document should succeed");
        idx.remove_document("d1")
            .expect("test: remove_document should succeed");
        let s = idx.stats();
        assert_eq!(s.doc_count, 1);
    }

    // ── PostingEntry construction ───────────────────────────────────────────

    #[test]
    fn posting_entry_stores_positions() {
        let entry = PostingEntry {
            doc_id: "d1".to_string(),
            term_freq: 2,
            positions: vec![0, 5],
        };
        assert_eq!(entry.term_freq, 2);
        assert_eq!(entry.positions, vec![0, 5]);
    }

    // ── IndexedDocument builder ─────────────────────────────────────────────

    #[test]
    fn indexed_document_builder_chain() {
        let doc = IndexedDocument::new("id", "content")
            .with_field("author", "alice")
            .with_embedding(vec![0.1, 0.2])
            .with_indexed_at(12345);
        assert_eq!(doc.fields["author"], "alice");
        assert_eq!(
            doc.embedding
                .as_ref()
                .expect("test: embedding should be Some after with_embedding call")[0],
            0.1
        );
        assert_eq!(doc.indexed_at, 12345);
    }

    // ── Edge cases ───────────────────────────────────────────────────────────

    #[test]
    fn search_query_terms_are_normalised() {
        let mut idx = CorpusIndexer::new();
        idx.add_document(make_doc("d1", "rust programming language"))
            .expect("test: add_document should succeed");
        // Query in uppercase should still match.
        let q = IndexQuery::new(["RUST"]);
        let results = idx.search(&q);
        assert!(!results.is_empty());
    }

    #[test]
    fn remove_then_readd_same_id_succeeds() {
        let mut idx = CorpusIndexer::new();
        idx.add_document(make_doc("d1", "first content"))
            .expect("test: add_document should succeed");
        idx.remove_document("d1")
            .expect("test: remove_document should succeed");
        idx.add_document(make_doc("d1", "second content"))
            .expect("test: add_document should succeed");
        assert_eq!(idx.doc_count(), 1);
        assert!(idx.index.term_to_postings.contains_key("second"));
        assert!(!idx.index.term_to_postings.contains_key("first"));
    }

    #[test]
    fn search_with_empty_terms_returns_empty() {
        let mut idx = CorpusIndexer::new();
        idx.add_document(make_doc("d1", "content"))
            .expect("test: add_document should succeed");
        let q = IndexQuery::new(Vec::<String>::new());
        assert!(idx.search(&q).is_empty());
    }

    #[test]
    fn avg_doc_length_updates_on_add_and_remove() {
        let mut idx = CorpusIndexer::new();
        idx.add_document(make_doc("d1", "alpha beta gamma delta"))
            .expect("test: add_document should succeed");
        let len1 = idx.index.avg_doc_length;
        idx.add_document(make_doc("d2", "short"))
            .expect("test: add_document should succeed");
        let len2 = idx.index.avg_doc_length;
        assert_ne!(len1, len2);
        idx.remove_document("d2")
            .expect("test: remove_document should succeed");
        let len3 = idx.index.avg_doc_length;
        assert!((len1 - len3).abs() < 1e-9);
    }

    #[test]
    fn search_result_has_matched_terms() {
        let mut idx = CorpusIndexer::new();
        idx.add_document(make_doc("d1", "rust systems programming"))
            .expect("test: add_document should succeed");
        let q = IndexQuery::new(["rust", "systems"]);
        let results = idx.search(&q);
        let matched = &results[0].matched_terms;
        assert!(matched.contains(&"rust".to_string()));
        assert!(matched.contains(&"systems".to_string()));
    }
}
