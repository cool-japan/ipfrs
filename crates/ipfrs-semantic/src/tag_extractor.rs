//! Semantic Tag Extractor
//!
//! Extracts semantic tags from embedding vectors using similarity-based tag assignment
//! and TF-IDF-like scoring for production-quality tag management.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Tag
// ---------------------------------------------------------------------------

/// A registered semantic tag with its representative embedding and usage statistics.
#[derive(Debug, Clone)]
pub struct Tag {
    /// Human-readable name of this tag.
    pub name: String,
    /// Representative embedding vector for this tag.
    pub embedding: Vec<f32>,
    /// Number of documents that have been assigned this tag.
    pub doc_frequency: u64,
}

impl Tag {
    /// Create a new [`Tag`] with zero doc-frequency.
    pub fn new(name: String, embedding: Vec<f32>) -> Self {
        Self {
            name,
            embedding,
            doc_frequency: 0,
        }
    }

    /// Cosine similarity between this tag's embedding and `vec`.
    ///
    /// Returns `0.0` if either vector has a zero norm.
    pub fn similarity(&self, vec: &[f32]) -> f32 {
        cosine_similarity(&self.embedding, vec)
    }
}

// ---------------------------------------------------------------------------
// TagAssignment
// ---------------------------------------------------------------------------

/// The result of assigning a tag to a specific document.
#[derive(Debug, Clone)]
pub struct TagAssignment {
    /// Identifier of the document that received this tag.
    pub document_id: u64,
    /// Name of the assigned tag.
    pub tag_name: String,
    /// Combined score: `similarity * idf_weight` (or just `similarity` when IDF is disabled).
    pub score: f32,
}

impl TagAssignment {
    /// Smoothed IDF weight.
    ///
    /// Formula: `ln((total_docs + 1) / (doc_freq + 1)) + 1`
    pub fn idf_weight(doc_freq: u64, total_docs: u64) -> f32 {
        let numerator = total_docs as f32 + 1.0;
        let denominator = doc_freq as f32 + 1.0;
        (numerator / denominator).ln() + 1.0
    }
}

// ---------------------------------------------------------------------------
// ExtractionConfig
// ---------------------------------------------------------------------------

/// Configuration knobs for [`SemanticTagExtractor`].
#[derive(Debug, Clone)]
pub struct ExtractionConfig {
    /// Tags with cosine similarity below this threshold are discarded (default `0.5`).
    pub min_similarity: f32,
    /// Maximum number of tags assigned per document (default `10`).
    pub max_tags_per_doc: usize,
    /// When `true` (default), scores are weighted by the smoothed IDF.
    pub use_idf_weighting: bool,
}

impl Default for ExtractionConfig {
    fn default() -> Self {
        Self {
            min_similarity: 0.5,
            max_tags_per_doc: 10,
            use_idf_weighting: true,
        }
    }
}

// ---------------------------------------------------------------------------
// ExtractorStats
// ---------------------------------------------------------------------------

/// Running statistics collected by [`SemanticTagExtractor`].
#[derive(Debug, Clone, Default)]
pub struct ExtractorStats {
    /// Total number of documents processed.
    pub total_documents: u64,
    /// Total number of tag assignments made across all documents.
    pub total_tags_assigned: u64,
}

impl ExtractorStats {
    /// Average number of tags assigned per document.
    ///
    /// Returns `0.0` when no documents have been processed.
    pub fn avg_tags_per_doc(&self) -> f64 {
        if self.total_documents == 0 {
            0.0
        } else {
            self.total_tags_assigned as f64 / self.total_documents as f64
        }
    }
}

// ---------------------------------------------------------------------------
// SemanticTagExtractor
// ---------------------------------------------------------------------------

/// Assigns semantic tags to documents by comparing their embedding vectors against
/// a registry of tag embeddings, optionally weighted by an IDF-like score.
pub struct SemanticTagExtractor {
    /// Registered tags, keyed by tag name.
    pub tags: HashMap<String, Tag>,
    /// Extraction configuration.
    pub config: ExtractionConfig,
    /// Running statistics.
    pub stats: ExtractorStats,
    /// Total number of documents seen; used for IDF computation.
    pub total_docs: u64,
}

impl SemanticTagExtractor {
    /// Create a new extractor with the given configuration.
    pub fn new(config: ExtractionConfig) -> Self {
        Self {
            tags: HashMap::new(),
            config,
            stats: ExtractorStats::default(),
            total_docs: 0,
        }
    }

    /// Register a tag with its representative embedding.
    ///
    /// If a tag with the same name already exists it is replaced (doc_frequency reset to 0).
    pub fn register_tag(&mut self, name: String, embedding: Vec<f32>) {
        self.tags.insert(name.clone(), Tag::new(name, embedding));
    }

    /// Extract and score tags for a document given its embedding vector.
    ///
    /// The method:
    /// 1. Computes cosine similarity between `doc_embedding` and every registered tag.
    /// 2. Discards tags below `config.min_similarity`.
    /// 3. Optionally multiplies similarity by a smoothed IDF weight.
    /// 4. Sorts descending by score and truncates to `config.max_tags_per_doc`.
    /// 5. Increments `doc_frequency` for selected tags and updates statistics.
    pub fn extract_tags(&mut self, document_id: u64, doc_embedding: &[f32]) -> Vec<TagAssignment> {
        // Increment total_docs first so that IDF reflects the document being processed.
        self.total_docs += 1;
        self.stats.total_documents += 1;

        // Collect candidates that meet the similarity threshold.
        let mut candidates: Vec<TagAssignment> = self
            .tags
            .values()
            .filter_map(|tag| {
                let sim = tag.similarity(doc_embedding);
                if sim < self.config.min_similarity {
                    return None;
                }
                let score = if self.config.use_idf_weighting {
                    sim * TagAssignment::idf_weight(tag.doc_frequency, self.total_docs)
                } else {
                    sim
                };
                Some(TagAssignment {
                    document_id,
                    tag_name: tag.name.clone(),
                    score,
                })
            })
            .collect();

        // Sort descending by score (higher is better).
        candidates.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Enforce the per-document cap.
        candidates.truncate(self.config.max_tags_per_doc);

        // Update per-tag doc frequencies and global stats.
        for assignment in &candidates {
            if let Some(tag) = self.tags.get_mut(&assignment.tag_name) {
                tag.doc_frequency += 1;
            }
        }
        self.stats.total_tags_assigned += candidates.len() as u64;

        candidates
    }

    /// Return the top-k tags sorted by doc_frequency (descending).
    pub fn top_tags(&self, k: usize) -> Vec<&Tag> {
        let mut sorted: Vec<&Tag> = self.tags.values().collect();
        sorted.sort_by_key(|b| std::cmp::Reverse(b.doc_frequency));
        sorted.truncate(k);
        sorted
    }

    /// Convenience wrapper that returns only tag names from `extract_tags`.
    pub fn tags_for_document(&mut self, document_id: u64, doc_embedding: &[f32]) -> Vec<String> {
        self.extract_tags(document_id, doc_embedding)
            .into_iter()
            .map(|a| a.tag_name)
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Compute cosine similarity between two slices.
///
/// Returns `0.0` if either vector has a zero norm or the slices have mismatched lengths.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: build a simple unit-vector tag embedding pointing in a given direction.
    fn unit_vec(dim: usize, hot_index: usize) -> Vec<f32> {
        let mut v = vec![0.0_f32; dim];
        v[hot_index] = 1.0;
        v
    }

    // Helper: build a normalised diagonal embedding (all components equal).
    fn diagonal_vec(dim: usize) -> Vec<f32> {
        let val = 1.0_f32 / (dim as f32).sqrt();
        vec![val; dim]
    }

    fn default_extractor() -> SemanticTagExtractor {
        SemanticTagExtractor::new(ExtractionConfig::default())
    }

    // ------------------------------------------------------------------
    // Tag::similarity
    // ------------------------------------------------------------------

    #[test]
    fn test_tag_similarity_identical() {
        let tag = Tag::new("rust".into(), vec![1.0, 0.0, 0.0]);
        let result = tag.similarity(&[1.0, 0.0, 0.0]);
        assert!(
            (result - 1.0).abs() < 1e-6,
            "identical vectors should yield similarity 1.0"
        );
    }

    #[test]
    fn test_tag_similarity_orthogonal() {
        let tag = Tag::new("rust".into(), vec![1.0, 0.0, 0.0]);
        let result = tag.similarity(&[0.0, 1.0, 0.0]);
        assert!(
            (result - 0.0).abs() < 1e-6,
            "orthogonal vectors should yield similarity 0.0"
        );
    }

    #[test]
    fn test_tag_similarity_zero_embedding_returns_zero() {
        let tag = Tag::new("empty".into(), vec![0.0, 0.0, 0.0]);
        let result = tag.similarity(&[1.0, 0.0, 0.0]);
        assert_eq!(result, 0.0, "zero-norm tag embedding should yield 0.0");
    }

    #[test]
    fn test_tag_similarity_zero_query_returns_zero() {
        let tag = Tag::new("rust".into(), vec![1.0, 0.0, 0.0]);
        let result = tag.similarity(&[0.0, 0.0, 0.0]);
        assert_eq!(result, 0.0, "zero-norm query should yield 0.0");
    }

    // ------------------------------------------------------------------
    // TagAssignment::idf_weight
    // ------------------------------------------------------------------

    #[test]
    fn test_idf_weight_formula_new_tag() {
        // doc_freq=0, total_docs=100 → ln(101/1) + 1 ≈ 5.615
        let w = TagAssignment::idf_weight(0, 100);
        let expected = (101.0_f32 / 1.0_f32).ln() + 1.0;
        assert!(
            (w - expected).abs() < 1e-5,
            "idf_weight mismatch for new tag"
        );
    }

    #[test]
    fn test_idf_weight_formula_high_freq() {
        // doc_freq=50, total_docs=100 → ln(101/51) + 1 ≈ 1.683
        let w = TagAssignment::idf_weight(50, 100);
        let expected = (101.0_f32 / 51.0_f32).ln() + 1.0;
        assert!(
            (w - expected).abs() < 1e-5,
            "idf_weight mismatch for high-freq tag"
        );
    }

    #[test]
    fn test_idf_weight_decreases_with_doc_frequency() {
        let total = 1000_u64;
        let w_rare = TagAssignment::idf_weight(1, total);
        let w_common = TagAssignment::idf_weight(500, total);
        assert!(w_rare > w_common, "rare tag should have higher IDF weight");
    }

    // ------------------------------------------------------------------
    // register_tag
    // ------------------------------------------------------------------

    #[test]
    fn test_register_tag_adds_entry() {
        let mut extractor = default_extractor();
        extractor.register_tag("science".into(), vec![0.1, 0.2, 0.3]);
        assert!(extractor.tags.contains_key("science"));
    }

    #[test]
    fn test_register_tag_initial_doc_frequency_is_zero() {
        let mut extractor = default_extractor();
        extractor.register_tag("tech".into(), vec![1.0, 0.0]);
        assert_eq!(extractor.tags["tech"].doc_frequency, 0);
    }

    #[test]
    fn test_register_tag_overwrites_existing() {
        let mut extractor = default_extractor();
        extractor.register_tag("music".into(), vec![1.0, 0.0]);
        // Manually bump doc_frequency to simulate prior usage.
        extractor
            .tags
            .get_mut("music")
            .expect("tag must exist")
            .doc_frequency = 42;
        // Re-register should reset.
        extractor.register_tag("music".into(), vec![0.0, 1.0]);
        assert_eq!(
            extractor.tags["music"].doc_frequency, 0,
            "re-registration should reset doc_frequency"
        );
        assert_eq!(extractor.tags["music"].embedding, vec![0.0, 1.0]);
    }

    // ------------------------------------------------------------------
    // extract_tags — basic
    // ------------------------------------------------------------------

    #[test]
    fn test_extract_tags_above_threshold_returned() {
        let mut extractor = default_extractor(); // min_similarity = 0.5
        extractor.register_tag("rust".into(), unit_vec(4, 0));
        extractor.register_tag("python".into(), unit_vec(4, 1));

        // Query almost identical to "rust" tag.
        let doc_emb = unit_vec(4, 0);
        let assignments = extractor.extract_tags(1, &doc_emb);

        let names: Vec<&str> = assignments.iter().map(|a| a.tag_name.as_str()).collect();
        assert!(names.contains(&"rust"), "rust should be assigned");
    }

    #[test]
    fn test_extract_tags_below_threshold_not_returned() {
        let mut extractor = default_extractor(); // min_similarity = 0.5
        extractor.register_tag("unrelated".into(), unit_vec(4, 3));

        // Query orthogonal to the tag.
        let doc_emb = unit_vec(4, 0);
        let assignments = extractor.extract_tags(1, &doc_emb);
        assert!(
            assignments.is_empty(),
            "tag below threshold must not be returned"
        );
    }

    #[test]
    fn test_extract_tags_max_tags_per_doc_enforced() {
        let config = ExtractionConfig {
            min_similarity: 0.0,
            max_tags_per_doc: 3,
            use_idf_weighting: false,
        };
        let mut extractor = SemanticTagExtractor::new(config);

        // Register 6 tags, all identical to the query so all similarity = 1.0.
        for i in 0..6_usize {
            extractor.register_tag(format!("tag_{i}"), diagonal_vec(4));
        }

        let doc_emb = diagonal_vec(4);
        let assignments = extractor.extract_tags(1, &doc_emb);
        assert_eq!(
            assignments.len(),
            3,
            "at most max_tags_per_doc tags should be returned"
        );
    }

    #[test]
    fn test_extract_tags_sorted_by_score_descending() {
        let config = ExtractionConfig {
            min_similarity: 0.0,
            max_tags_per_doc: 10,
            use_idf_weighting: false,
        };
        let mut extractor = SemanticTagExtractor::new(config);

        // "exact" tag is identical to the query; "partial" is at 45°.
        extractor.register_tag("exact".into(), unit_vec(2, 0));
        let partial_emb = vec![1.0_f32 / 2.0_f32.sqrt(), 1.0_f32 / 2.0_f32.sqrt()];
        extractor.register_tag("partial".into(), partial_emb);

        let doc_emb = unit_vec(2, 0);
        let assignments = extractor.extract_tags(1, &doc_emb);

        assert!(
            !assignments.is_empty(),
            "should have at least one assignment"
        );
        for window in assignments.windows(2) {
            assert!(
                window[0].score >= window[1].score,
                "assignments must be sorted descending by score"
            );
        }
    }

    // ------------------------------------------------------------------
    // IDF weighting
    // ------------------------------------------------------------------

    #[test]
    fn test_idf_weighting_reduces_high_freq_tag_score() {
        let config = ExtractionConfig {
            min_similarity: 0.0,
            max_tags_per_doc: 10,
            use_idf_weighting: true,
        };
        let mut extractor = SemanticTagExtractor::new(config);

        // Both tags have the same embedding as the query.
        extractor.register_tag("rare".into(), diagonal_vec(4));
        extractor.register_tag("common".into(), diagonal_vec(4));

        // Artificially inflate "common" tag's doc_frequency.
        extractor
            .tags
            .get_mut("common")
            .expect("tag exists")
            .doc_frequency = 999;

        let doc_emb = diagonal_vec(4);
        let assignments = extractor.extract_tags(1, &doc_emb);

        let rare_score = assignments
            .iter()
            .find(|a| a.tag_name == "rare")
            .map(|a| a.score)
            .expect("rare must be assigned");
        let common_score = assignments
            .iter()
            .find(|a| a.tag_name == "common")
            .map(|a| a.score)
            .expect("common must be assigned");

        assert!(
            rare_score > common_score,
            "rare tag should score higher than common tag (IDF weighting)"
        );
    }

    #[test]
    fn test_idf_weighting_disabled_uses_raw_similarity() {
        let config = ExtractionConfig {
            min_similarity: 0.0,
            max_tags_per_doc: 10,
            use_idf_weighting: false,
        };
        let mut extractor = SemanticTagExtractor::new(config);
        extractor.register_tag("tag_a".into(), unit_vec(3, 0));

        let doc_emb = unit_vec(3, 0);
        let assignments = extractor.extract_tags(1, &doc_emb);
        let score = assignments
            .first()
            .map(|a| a.score)
            .expect("should have assignment");

        // Without IDF, score == cosine similarity ≈ 1.0.
        assert!(
            (score - 1.0_f32).abs() < 1e-5,
            "score without IDF should equal raw cosine similarity"
        );
    }

    // ------------------------------------------------------------------
    // doc_frequency increments
    // ------------------------------------------------------------------

    #[test]
    fn test_doc_frequency_increments_on_assignment() {
        let mut extractor = default_extractor();
        extractor.register_tag("rust".into(), unit_vec(3, 0));

        let doc_emb = unit_vec(3, 0);
        extractor.extract_tags(1, &doc_emb);
        extractor.extract_tags(2, &doc_emb);

        assert_eq!(
            extractor.tags["rust"].doc_frequency, 2,
            "doc_frequency should increment for each assignment"
        );
    }

    #[test]
    fn test_doc_frequency_not_incremented_below_threshold() {
        let mut extractor = default_extractor();
        extractor.register_tag("unrelated".into(), unit_vec(3, 2));

        let doc_emb = unit_vec(3, 0); // orthogonal → not assigned
        extractor.extract_tags(1, &doc_emb);

        assert_eq!(
            extractor.tags["unrelated"].doc_frequency, 0,
            "doc_frequency must not increment when tag is below threshold"
        );
    }

    // ------------------------------------------------------------------
    // top_tags
    // ------------------------------------------------------------------

    #[test]
    fn test_top_tags_sorted_by_doc_frequency_descending() {
        let mut extractor = default_extractor();
        extractor.register_tag("a".into(), diagonal_vec(2));
        extractor.register_tag("b".into(), diagonal_vec(2));
        extractor.register_tag("c".into(), diagonal_vec(2));

        // Manually set doc_frequency to create a known ordering.
        extractor.tags.get_mut("a").expect("a exists").doc_frequency = 5;
        extractor.tags.get_mut("b").expect("b exists").doc_frequency = 20;
        extractor.tags.get_mut("c").expect("c exists").doc_frequency = 10;

        let top = extractor.top_tags(2);
        assert_eq!(top.len(), 2);
        assert_eq!(
            top[0].name, "b",
            "highest doc_frequency tag should be first"
        );
        assert_eq!(top[1].name, "c", "second highest should be second");
    }

    #[test]
    fn test_top_tags_k_larger_than_registry_returns_all() {
        let mut extractor = default_extractor();
        extractor.register_tag("x".into(), unit_vec(2, 0));
        extractor.register_tag("y".into(), unit_vec(2, 1));

        let top = extractor.top_tags(100);
        assert_eq!(
            top.len(),
            2,
            "should return all tags when k exceeds registry size"
        );
    }

    // ------------------------------------------------------------------
    // stats
    // ------------------------------------------------------------------

    #[test]
    fn test_stats_total_documents_increments() {
        let mut extractor = default_extractor();
        extractor.register_tag("t".into(), diagonal_vec(3));
        let doc_emb = diagonal_vec(3);

        extractor.extract_tags(1, &doc_emb);
        extractor.extract_tags(2, &doc_emb);
        extractor.extract_tags(3, &doc_emb);

        assert_eq!(extractor.stats.total_documents, 3);
    }

    #[test]
    fn test_stats_avg_tags_per_doc_correct() {
        let config = ExtractionConfig {
            min_similarity: 0.0,
            max_tags_per_doc: 10,
            use_idf_weighting: false,
        };
        let mut extractor = SemanticTagExtractor::new(config);
        extractor.register_tag("a".into(), unit_vec(3, 0));
        extractor.register_tag("b".into(), unit_vec(3, 0));

        // Both tags identical to query → 2 assignments per doc.
        let doc_emb = unit_vec(3, 0);
        extractor.extract_tags(1, &doc_emb);
        extractor.extract_tags(2, &doc_emb);

        let avg = extractor.stats.avg_tags_per_doc();
        assert!(
            (avg - 2.0).abs() < 1e-9,
            "avg_tags_per_doc should be 2.0, got {avg}"
        );
    }

    #[test]
    fn test_stats_avg_tags_per_doc_zero_when_no_docs() {
        let extractor = default_extractor();
        assert_eq!(extractor.stats.avg_tags_per_doc(), 0.0);
    }

    // ------------------------------------------------------------------
    // Edge cases
    // ------------------------------------------------------------------

    #[test]
    fn test_empty_doc_embedding_returns_no_tags() {
        let mut extractor = default_extractor();
        extractor.register_tag("rust".into(), vec![1.0, 0.0]);

        // Empty slice → cosine_similarity returns 0.0 < min_similarity.
        let assignments = extractor.extract_tags(1, &[]);
        assert!(
            assignments.is_empty(),
            "empty embedding should produce no assignments"
        );
    }

    #[test]
    fn test_no_tags_registered_returns_empty() {
        let mut extractor = default_extractor();
        let doc_emb = diagonal_vec(4);
        let assignments = extractor.extract_tags(1, &doc_emb);
        assert!(
            assignments.is_empty(),
            "no registered tags → no assignments"
        );
    }

    #[test]
    fn test_tags_for_document_returns_names() {
        let config = ExtractionConfig {
            min_similarity: 0.0,
            max_tags_per_doc: 10,
            use_idf_weighting: false,
        };
        let mut extractor = SemanticTagExtractor::new(config);
        extractor.register_tag("alpha".into(), unit_vec(3, 0));
        extractor.register_tag("beta".into(), unit_vec(3, 0));

        let doc_emb = unit_vec(3, 0);
        let names = extractor.tags_for_document(1, &doc_emb);
        assert!(names.contains(&"alpha".to_string()));
        assert!(names.contains(&"beta".to_string()));
    }
}
