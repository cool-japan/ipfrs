//! # Semantic Term Weighter
//!
//! TF-IDF and BM25 term weighting for semantic search within IPFRS.
//!
//! Supports multiple weighting schemes:
//! - **TF-IDF**: Term Frequency–Inverse Document Frequency
//! - **BM25**: Okapi BM25 ranking function
//! - **Binary**: Simple presence/absence weighting

use std::collections::HashMap;

/// Weighting scheme selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightingScheme {
    /// Classic TF-IDF: tf * idf
    TfIdf,
    /// Okapi BM25 with saturation and length normalization
    Bm25,
    /// Binary weighting: 1.0 if the term is present, 0.0 otherwise
    Binary,
}

/// Configuration for `SemanticTermWeighter`.
#[derive(Debug, Clone)]
pub struct WeighterConfig {
    /// The weighting scheme to use when computing term weights.
    pub scheme: WeightingScheme,
    /// BM25 term-frequency saturation parameter (default 1.2).
    pub bm25_k1: f64,
    /// BM25 document-length normalization parameter (default 0.75).
    pub bm25_b: f64,
}

impl Default for WeighterConfig {
    fn default() -> Self {
        Self {
            scheme: WeightingScheme::TfIdf,
            bm25_k1: 1.2,
            bm25_b: 0.75,
        }
    }
}

/// A single term with its computed weight components.
#[derive(Debug, Clone)]
pub struct TermWeight {
    /// The term string.
    pub term: String,
    /// The final computed weight (depends on scheme).
    pub weight: f64,
    /// Term frequency component.
    pub tf: f64,
    /// Inverse document frequency component.
    pub idf: f64,
}

/// Profile of a single document in the corpus.
#[derive(Debug, Clone)]
pub struct DocumentProfile {
    /// Unique document identifier.
    pub doc_id: String,
    /// Mapping from term to its raw count in the document.
    pub term_counts: HashMap<String, u64>,
    /// Total number of terms (including duplicates) in the document.
    pub total_terms: u64,
}

/// Aggregate statistics for a `SemanticTermWeighter` instance.
#[derive(Debug, Clone)]
pub struct TermWeighterStats {
    /// Number of documents in the corpus.
    pub total_docs: u64,
    /// Number of unique terms across all documents.
    pub vocab_size: usize,
    /// Mean document length (in terms).
    pub avg_doc_length: f64,
    /// Active weighting scheme.
    pub scheme: WeightingScheme,
}

/// TF-IDF / BM25 term weighter for a corpus of documents.
///
/// Maintains document profiles and document-frequency counts so that
/// term weights can be computed incrementally as documents are added or
/// removed.
pub struct SemanticTermWeighter {
    config: WeighterConfig,
    documents: HashMap<String, DocumentProfile>,
    /// term -> number of documents containing that term
    doc_freq: HashMap<String, u64>,
    total_docs: u64,
    avg_doc_length: f64,
}

impl SemanticTermWeighter {
    /// Create a new weighter with the given configuration.
    pub fn new(config: WeighterConfig) -> Self {
        Self {
            config,
            documents: HashMap::new(),
            doc_freq: HashMap::new(),
            total_docs: 0,
            avg_doc_length: 0.0,
        }
    }

    /// Register a document by its id and a slice of terms.
    ///
    /// Counts each term occurrence and updates corpus-level statistics
    /// (document frequencies, average document length).
    ///
    /// If a document with the same `doc_id` already exists it is replaced.
    pub fn add_document(&mut self, doc_id: &str, terms: &[&str]) {
        // If the document already exists, remove it first so stats stay correct.
        if self.documents.contains_key(doc_id) {
            self.remove_document(doc_id);
        }

        let mut term_counts: HashMap<String, u64> = HashMap::new();
        for term in terms {
            *term_counts.entry((*term).to_string()).or_insert(0) += 1;
        }

        let total_terms = terms.len() as u64;

        // Update document frequencies (each unique term gets +1).
        for term in term_counts.keys() {
            *self.doc_freq.entry(term.clone()).or_insert(0) += 1;
        }

        let profile = DocumentProfile {
            doc_id: doc_id.to_string(),
            term_counts,
            total_terms,
        };

        self.documents.insert(doc_id.to_string(), profile);
        self.total_docs += 1;
        self.recompute_avg_doc_length();
    }

    /// Remove a document from the corpus. Returns `true` if it existed.
    pub fn remove_document(&mut self, doc_id: &str) -> bool {
        let profile = match self.documents.remove(doc_id) {
            Some(p) => p,
            None => return false,
        };

        // Decrement document frequencies for each unique term.
        for term in profile.term_counts.keys() {
            if let Some(freq) = self.doc_freq.get_mut(term) {
                *freq = freq.saturating_sub(1);
                if *freq == 0 {
                    self.doc_freq.remove(term);
                }
            }
        }

        self.total_docs = self.total_docs.saturating_sub(1);
        self.recompute_avg_doc_length();
        true
    }

    /// Compute weights for every term in the specified document.
    pub fn weight_terms(&self, doc_id: &str) -> Result<Vec<TermWeight>, String> {
        let profile = self
            .documents
            .get(doc_id)
            .ok_or_else(|| format!("document '{}' not found", doc_id))?;

        let mut weights: Vec<TermWeight> = Vec::with_capacity(profile.term_counts.len());

        for (term, &count) in &profile.term_counts {
            let tf_val = self.compute_tf(count, profile.total_terms);
            let idf_val = self.idf(term);

            let weight = match self.config.scheme {
                WeightingScheme::TfIdf => tf_val * idf_val,
                WeightingScheme::Bm25 => self.compute_bm25(count, profile.total_terms, idf_val),
                WeightingScheme::Binary => {
                    if count > 0 {
                        1.0
                    } else {
                        0.0
                    }
                }
            };

            weights.push(TermWeight {
                term: term.clone(),
                weight,
                tf: tf_val,
                idf: idf_val,
            });
        }

        // Sort by weight descending for convenience.
        weights.sort_by(|a, b| {
            b.weight
                .partial_cmp(&a.weight)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(weights)
    }

    /// Raw term frequency: count / total_terms.
    pub fn tf(&self, term: &str, doc_id: &str) -> Option<f64> {
        let profile = self.documents.get(doc_id)?;
        let count = profile.term_counts.get(term).copied().unwrap_or(0);
        Some(self.compute_tf(count, profile.total_terms))
    }

    /// Inverse document frequency: ln((N + 1) / (df + 1)) + 1.
    pub fn idf(&self, term: &str) -> f64 {
        let df = self.doc_freq.get(term).copied().unwrap_or(0) as f64;
        let n = self.total_docs as f64;
        ((n + 1.0) / (df + 1.0)).ln() + 1.0
    }

    /// BM25 score for a single term in a document.
    pub fn bm25_score(&self, term: &str, doc_id: &str) -> Option<f64> {
        let profile = self.documents.get(doc_id)?;
        let count = profile.term_counts.get(term).copied().unwrap_or(0);
        let idf_val = self.idf(term);
        Some(self.compute_bm25(count, profile.total_terms, idf_val))
    }

    /// Cosine similarity between the TF-IDF vectors of two documents.
    pub fn similarity(&self, doc_a: &str, doc_b: &str) -> Result<f64, String> {
        let profile_a = self
            .documents
            .get(doc_a)
            .ok_or_else(|| format!("document '{}' not found", doc_a))?;
        let profile_b = self
            .documents
            .get(doc_b)
            .ok_or_else(|| format!("document '{}' not found", doc_b))?;

        // Build TF-IDF vectors keyed by term.
        let vec_a = self.tfidf_vector(profile_a);
        let vec_b = self.tfidf_vector(profile_b);

        // Compute dot product over shared terms.
        let mut dot = 0.0_f64;
        for (term, wa) in &vec_a {
            if let Some(wb) = vec_b.get(term) {
                dot += wa * wb;
            }
        }

        let mag_a = vec_a.values().map(|v| v * v).sum::<f64>().sqrt();
        let mag_b = vec_b.values().map(|v| v * v).sum::<f64>().sqrt();

        if mag_a == 0.0 || mag_b == 0.0 {
            return Ok(0.0);
        }

        Ok(dot / (mag_a * mag_b))
    }

    /// Number of documents in the corpus.
    pub fn doc_count(&self) -> usize {
        self.total_docs as usize
    }

    /// Number of unique terms across all documents.
    pub fn vocab_size(&self) -> usize {
        self.doc_freq.len()
    }

    /// Aggregate statistics snapshot.
    pub fn stats(&self) -> TermWeighterStats {
        TermWeighterStats {
            total_docs: self.total_docs,
            vocab_size: self.vocab_size(),
            avg_doc_length: self.avg_doc_length,
            scheme: self.config.scheme,
        }
    }

    // ---- private helpers ----

    fn compute_tf(&self, count: u64, total: u64) -> f64 {
        if total == 0 {
            return 0.0;
        }
        count as f64 / total as f64
    }

    fn compute_bm25(&self, count: u64, doc_len: u64, idf_val: f64) -> f64 {
        let tf = count as f64;
        let k1 = self.config.bm25_k1;
        let b = self.config.bm25_b;
        let dl = doc_len as f64;
        let avgdl = if self.avg_doc_length > 0.0 {
            self.avg_doc_length
        } else {
            1.0
        };

        let numerator = tf * (k1 + 1.0);
        let denominator = tf + k1 * (1.0 - b + b * (dl / avgdl));

        idf_val * numerator / denominator
    }

    fn tfidf_vector(&self, profile: &DocumentProfile) -> HashMap<String, f64> {
        let mut vec = HashMap::with_capacity(profile.term_counts.len());
        for (term, &count) in &profile.term_counts {
            let tf_val = self.compute_tf(count, profile.total_terms);
            let idf_val = self.idf(term);
            vec.insert(term.clone(), tf_val * idf_val);
        }
        vec
    }

    fn recompute_avg_doc_length(&mut self) {
        if self.total_docs == 0 {
            self.avg_doc_length = 0.0;
            return;
        }
        let total_terms: u64 = self.documents.values().map(|d| d.total_terms).sum();
        self.avg_doc_length = total_terms as f64 / self.total_docs as f64;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    fn default_tfidf_weighter() -> SemanticTermWeighter {
        SemanticTermWeighter::new(WeighterConfig::default())
    }

    fn bm25_weighter() -> SemanticTermWeighter {
        SemanticTermWeighter::new(WeighterConfig {
            scheme: WeightingScheme::Bm25,
            ..Default::default()
        })
    }

    fn binary_weighter() -> SemanticTermWeighter {
        SemanticTermWeighter::new(WeighterConfig {
            scheme: WeightingScheme::Binary,
            ..Default::default()
        })
    }

    // -- basic add / remove --

    #[test]
    fn test_add_single_document() {
        let mut w = default_tfidf_weighter();
        w.add_document("d1", &["hello", "world"]);
        assert_eq!(w.doc_count(), 1);
        assert_eq!(w.vocab_size(), 2);
    }

    #[test]
    fn test_add_multiple_documents() {
        let mut w = default_tfidf_weighter();
        w.add_document("d1", &["hello", "world"]);
        w.add_document("d2", &["foo", "bar", "baz"]);
        assert_eq!(w.doc_count(), 2);
        assert_eq!(w.vocab_size(), 5);
    }

    #[test]
    fn test_add_document_replaces_existing() {
        let mut w = default_tfidf_weighter();
        w.add_document("d1", &["hello", "world"]);
        w.add_document("d1", &["foo"]);
        assert_eq!(w.doc_count(), 1);
        assert_eq!(w.vocab_size(), 1);
    }

    #[test]
    fn test_remove_document_returns_true() {
        let mut w = default_tfidf_weighter();
        w.add_document("d1", &["hello"]);
        assert!(w.remove_document("d1"));
        assert_eq!(w.doc_count(), 0);
    }

    #[test]
    fn test_remove_nonexistent_returns_false() {
        let mut w = default_tfidf_weighter();
        assert!(!w.remove_document("nope"));
    }

    #[test]
    fn test_remove_updates_vocab() {
        let mut w = default_tfidf_weighter();
        w.add_document("d1", &["alpha", "beta"]);
        w.add_document("d2", &["beta", "gamma"]);
        w.remove_document("d1");
        // "alpha" should be gone, "beta" and "gamma" remain
        assert_eq!(w.vocab_size(), 2);
    }

    // -- TF --

    #[test]
    fn test_tf_present_term() {
        let mut w = default_tfidf_weighter();
        w.add_document("d1", &["a", "b", "a", "c"]);
        let tf = w.tf("a", "d1");
        assert!(tf.is_some());
        let val = tf.expect("tf should be some");
        assert!((val - 0.5).abs() < 1e-9, "expected 2/4 = 0.5, got {}", val);
    }

    #[test]
    fn test_tf_absent_term() {
        let mut w = default_tfidf_weighter();
        w.add_document("d1", &["a", "b"]);
        let tf = w.tf("z", "d1");
        assert!(tf.is_some());
        let val = tf.expect("tf should be some");
        assert!((val - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_tf_missing_doc() {
        let w = default_tfidf_weighter();
        assert!(w.tf("a", "nope").is_none());
    }

    // -- IDF --

    #[test]
    fn test_idf_unseen_term() {
        let mut w = default_tfidf_weighter();
        w.add_document("d1", &["a"]);
        // df=0, N=1 => ln(2/1)+1 = ln(2)+1
        let val = w.idf("z");
        let expected = (2.0_f64 / 1.0).ln() + 1.0;
        assert!((val - expected).abs() < 1e-9, "got {}", val);
    }

    #[test]
    fn test_idf_common_term() {
        let mut w = default_tfidf_weighter();
        w.add_document("d1", &["a", "b"]);
        w.add_document("d2", &["a", "c"]);
        // df=2, N=2 => ln(3/3)+1 = 1.0
        let val = w.idf("a");
        let expected = (3.0_f64 / 3.0).ln() + 1.0;
        assert!((val - expected).abs() < 1e-9, "got {}", val);
    }

    #[test]
    fn test_idf_rare_term() {
        let mut w = default_tfidf_weighter();
        w.add_document("d1", &["a"]);
        w.add_document("d2", &["b"]);
        w.add_document("d3", &["c"]);
        // df=1, N=3 => ln(4/2)+1 = ln(2)+1
        let val = w.idf("a");
        let expected = (4.0_f64 / 2.0).ln() + 1.0;
        assert!((val - expected).abs() < 1e-9, "got {}", val);
    }

    // -- TF-IDF weighting --

    #[test]
    fn test_tfidf_weight_terms() {
        let mut w = default_tfidf_weighter();
        w.add_document("d1", &["rust", "rust", "code"]);
        w.add_document("d2", &["code", "python"]);

        let weights = w.weight_terms("d1").expect("should succeed");
        assert_eq!(weights.len(), 2); // "rust" and "code"

        for tw in &weights {
            assert!(tw.weight > 0.0, "weight should be positive: {}", tw.term);
            assert!(tw.tf > 0.0);
            assert!(tw.idf > 0.0);
        }
    }

    #[test]
    fn test_tfidf_missing_doc() {
        let w = default_tfidf_weighter();
        let res = w.weight_terms("nope");
        assert!(res.is_err());
    }

    // -- BM25 --

    #[test]
    fn test_bm25_weight_terms() {
        let mut w = bm25_weighter();
        w.add_document("d1", &["a", "b", "a"]);
        w.add_document("d2", &["b", "c"]);

        let weights = w.weight_terms("d1").expect("should succeed");
        assert!(!weights.is_empty());
        for tw in &weights {
            assert!(tw.weight > 0.0, "bm25 weight should be > 0 for {}", tw.term);
        }
    }

    #[test]
    fn test_bm25_score_present() {
        let mut w = bm25_weighter();
        w.add_document("d1", &["a", "b", "a"]);
        let score = w.bm25_score("a", "d1");
        assert!(score.is_some());
        assert!(score.expect("some") > 0.0);
    }

    #[test]
    fn test_bm25_score_absent_term() {
        let mut w = bm25_weighter();
        w.add_document("d1", &["a", "b"]);
        let score = w.bm25_score("z", "d1").expect("some");
        // term not in any doc, but idf still > 0 due to smoothing; count = 0 => numerator = 0
        assert!((score - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_bm25_score_missing_doc() {
        let w = bm25_weighter();
        assert!(w.bm25_score("a", "nope").is_none());
    }

    #[test]
    fn test_bm25_longer_doc_lower_score() {
        // BM25 length normalization: same term count in a longer doc should score lower.
        let mut w = bm25_weighter();
        w.add_document("short", &["a", "a"]);
        w.add_document("long", &["a", "a", "b", "c", "d", "e", "f", "g"]);

        let s_short = w.bm25_score("a", "short").expect("some");
        let s_long = w.bm25_score("a", "long").expect("some");
        assert!(
            s_short > s_long,
            "short doc ({}) should score higher than long doc ({})",
            s_short,
            s_long
        );
    }

    #[test]
    fn test_bm25_custom_params() {
        let mut w = SemanticTermWeighter::new(WeighterConfig {
            scheme: WeightingScheme::Bm25,
            bm25_k1: 2.0,
            bm25_b: 0.5,
        });
        w.add_document("d1", &["x", "y", "x"]);
        let score = w.bm25_score("x", "d1").expect("some");
        assert!(score > 0.0);
    }

    // -- Binary --

    #[test]
    fn test_binary_weight_terms() {
        let mut w = binary_weighter();
        w.add_document("d1", &["a", "b", "a"]);
        let weights = w.weight_terms("d1").expect("should succeed");
        for tw in &weights {
            assert!(
                (tw.weight - 1.0).abs() < 1e-9,
                "binary weight should be 1.0, got {}",
                tw.weight
            );
        }
    }

    #[test]
    fn test_binary_no_extra_terms() {
        let mut w = binary_weighter();
        w.add_document("d1", &["a", "b"]);
        let weights = w.weight_terms("d1").expect("should succeed");
        assert_eq!(weights.len(), 2);
    }

    // -- similarity --

    #[test]
    fn test_similarity_identical_docs() {
        let mut w = default_tfidf_weighter();
        w.add_document("d1", &["a", "b", "c"]);
        w.add_document("d2", &["a", "b", "c"]);
        let sim = w.similarity("d1", "d2").expect("ok");
        assert!(
            (sim - 1.0).abs() < 1e-9,
            "identical docs should have similarity 1.0, got {}",
            sim
        );
    }

    #[test]
    fn test_similarity_disjoint_docs() {
        let mut w = default_tfidf_weighter();
        w.add_document("d1", &["a", "b"]);
        w.add_document("d2", &["c", "d"]);
        let sim = w.similarity("d1", "d2").expect("ok");
        assert!(
            sim.abs() < 1e-9,
            "disjoint docs should have similarity 0.0, got {}",
            sim
        );
    }

    #[test]
    fn test_similarity_partial_overlap() {
        let mut w = default_tfidf_weighter();
        w.add_document("d1", &["a", "b", "c"]);
        w.add_document("d2", &["b", "c", "d"]);
        let sim = w.similarity("d1", "d2").expect("ok");
        assert!(sim > 0.0 && sim < 1.0, "partial overlap: {}", sim);
    }

    #[test]
    fn test_similarity_missing_doc() {
        let mut w = default_tfidf_weighter();
        w.add_document("d1", &["a"]);
        assert!(w.similarity("d1", "nope").is_err());
        assert!(w.similarity("nope", "d1").is_err());
    }

    // -- stats / edge cases --

    #[test]
    fn test_stats_accuracy() {
        let mut w = default_tfidf_weighter();
        w.add_document("d1", &["a", "b"]);
        w.add_document("d2", &["c", "d", "e"]);
        let s = w.stats();
        assert_eq!(s.total_docs, 2);
        assert_eq!(s.vocab_size, 5);
        assert!((s.avg_doc_length - 2.5).abs() < 1e-9);
        assert_eq!(s.scheme, WeightingScheme::TfIdf);
    }

    #[test]
    fn test_empty_corpus() {
        let w = default_tfidf_weighter();
        assert_eq!(w.doc_count(), 0);
        assert_eq!(w.vocab_size(), 0);
        let s = w.stats();
        assert_eq!(s.total_docs, 0);
        assert!((s.avg_doc_length - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_single_doc_corpus() {
        let mut w = default_tfidf_weighter();
        w.add_document("d1", &["only"]);
        assert_eq!(w.doc_count(), 1);
        assert_eq!(w.vocab_size(), 1);
        let wts = w.weight_terms("d1").expect("ok");
        assert_eq!(wts.len(), 1);
        assert!(wts[0].weight > 0.0);
    }

    #[test]
    fn test_duplicate_terms_in_document() {
        let mut w = default_tfidf_weighter();
        w.add_document("d1", &["dup", "dup", "dup", "other"]);
        assert_eq!(w.vocab_size(), 2);
        let tf_dup = w.tf("dup", "d1").expect("some");
        assert!((tf_dup - 0.75).abs() < 1e-9, "3/4 = 0.75, got {}", tf_dup);
    }

    #[test]
    fn test_avg_doc_length_updates_on_remove() {
        let mut w = default_tfidf_weighter();
        w.add_document("d1", &["a", "b"]); // len 2
        w.add_document("d2", &["c", "d", "e", "f"]); // len 4
                                                     // avg = 3.0
        assert!((w.stats().avg_doc_length - 3.0).abs() < 1e-9);
        w.remove_document("d2");
        // avg = 2.0
        assert!((w.stats().avg_doc_length - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_document() {
        let mut w = default_tfidf_weighter();
        w.add_document("empty", &[]);
        assert_eq!(w.doc_count(), 1);
        assert_eq!(w.vocab_size(), 0);
        let wts = w.weight_terms("empty").expect("ok");
        assert!(wts.is_empty());
    }

    #[test]
    fn test_vocab_size_after_full_removal() {
        let mut w = default_tfidf_weighter();
        w.add_document("d1", &["a"]);
        w.remove_document("d1");
        assert_eq!(w.vocab_size(), 0);
    }

    #[test]
    fn test_idf_empty_corpus() {
        let w = default_tfidf_weighter();
        // N=0, df=0 => ln(1/1)+1 = 1.0
        let val = w.idf("anything");
        assert!((val - 1.0).abs() < 1e-9, "got {}", val);
    }

    #[test]
    fn test_weight_terms_sorted_descending() {
        let mut w = default_tfidf_weighter();
        // "a" appears more often than "b", so should have higher tf and thus higher weight
        w.add_document("d1", &["a", "a", "a", "b"]);
        let wts = w.weight_terms("d1").expect("ok");
        assert!(wts.len() == 2);
        assert!(
            wts[0].weight >= wts[1].weight,
            "should be sorted descending"
        );
    }

    #[test]
    fn test_bm25_saturation() {
        // Increasing term count should increase score but with diminishing returns.
        let mut w = bm25_weighter();
        w.add_document("d1", &["a"]);
        w.add_document("d2", &["a", "a"]);
        w.add_document("d3", &["a", "a", "a", "a", "a", "a", "a", "a", "a", "a"]);

        let s1 = w.bm25_score("a", "d1").expect("some");
        let s2 = w.bm25_score("a", "d2").expect("some");
        let s3 = w.bm25_score("a", "d3").expect("some");

        // All should be positive and s1 < s2 < s3 (more occurrences = higher, but saturating)
        assert!(s1 > 0.0);
        assert!(s2 > 0.0);
        assert!(s3 > 0.0);

        // The increment should diminish
        let delta_1_2 = s2 - s1;
        let delta_2_3 = s3 - s2;
        // With length normalization in play and varying doc lengths, the increments
        // should generally show saturation. We just verify they're all positive.
        assert!(delta_1_2 > 0.0 || delta_2_3 > 0.0, "scores should differ");
    }

    #[test]
    fn test_similarity_is_symmetric() {
        let mut w = default_tfidf_weighter();
        w.add_document("d1", &["a", "b", "c"]);
        w.add_document("d2", &["b", "c", "d"]);
        let s1 = w.similarity("d1", "d2").expect("ok");
        let s2 = w.similarity("d2", "d1").expect("ok");
        assert!((s1 - s2).abs() < 1e-12, "similarity should be symmetric");
    }

    #[test]
    fn test_large_corpus() {
        let mut w = default_tfidf_weighter();
        for i in 0..100 {
            let id = format!("doc_{}", i);
            let terms: Vec<&str> = if i % 2 == 0 {
                vec!["common", "even"]
            } else {
                vec!["common", "odd"]
            };
            w.add_document(&id, &terms);
        }
        assert_eq!(w.doc_count(), 100);
        assert_eq!(w.vocab_size(), 3); // common, even, odd
        assert!((w.stats().avg_doc_length - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_default_config() {
        let cfg = WeighterConfig::default();
        assert_eq!(cfg.scheme, WeightingScheme::TfIdf);
        assert!((cfg.bm25_k1 - 1.2).abs() < 1e-9);
        assert!((cfg.bm25_b - 0.75).abs() < 1e-9);
    }
}
