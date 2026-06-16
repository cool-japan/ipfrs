//! Extractive summarization based on sentence embedding similarity.
//!
//! [`SemanticSummaryExtractor`] selects representative sentences from a
//! collection by scoring them against an optional query embedding (cosine
//! similarity) or, when no query is provided, by centrality (average cosine
//! similarity to every other sentence).  A greedy selection loop with a
//! configurable diversity penalty prevents redundant picks.

// ── Configuration ────────────────────────────────────────────────────────────

/// Controls the behaviour of [`SemanticSummaryExtractor::extract`].
#[derive(Debug, Clone)]
pub struct ExtractorSummaryConfig {
    /// Maximum number of sentences to include in the summary (default 5).
    pub max_sentences: usize,
    /// Minimum score required for a sentence to be included (default 0.3).
    pub similarity_threshold: f64,
    /// How aggressively to penalise similarity to already-selected sentences
    /// (default 0.5).  Higher values encourage more diverse summaries.
    pub diversity_penalty: f64,
}

impl Default for ExtractorSummaryConfig {
    fn default() -> Self {
        Self {
            max_sentences: 5,
            similarity_threshold: 0.3,
            diversity_penalty: 0.5,
        }
    }
}

// ── Output types ─────────────────────────────────────────────────────────────

/// A sentence together with its embedding, score, and selection flag.
#[derive(Debug, Clone)]
pub struct ExtractorScoredSentence {
    /// Zero-based index of this sentence in the input slice.
    pub index: usize,
    /// Raw text of the sentence.
    pub text: String,
    /// Dense embedding vector.
    pub embedding: Vec<f64>,
    /// Computed score (possibly penalised by diversity).
    pub score: f64,
    /// Whether this sentence was selected for the summary.
    pub selected: bool,
}

/// Result of an extraction run.
#[derive(Debug, Clone)]
pub struct ExtractionResult {
    /// Indices of the selected sentences, in selection order.
    pub selected_indices: Vec<usize>,
    /// All sentences with their final scores and selection flags.
    pub sentences: Vec<ExtractorScoredSentence>,
    /// Coverage: average of the maximum similarity from each *unselected*
    /// sentence to its nearest selected sentence.
    pub coverage_score: f64,
}

/// Simple stats counter.
#[derive(Debug, Clone)]
pub struct SummaryExtractorStats {
    /// Total number of extraction calls completed so far.
    pub extractions_performed: u64,
}

// ── Extractor ────────────────────────────────────────────────────────────────

/// Extractive summariser driven by sentence embeddings.
pub struct SemanticSummaryExtractor {
    config: ExtractorSummaryConfig,
    extractions_performed: u64,
}

impl SemanticSummaryExtractor {
    /// Create a new extractor with the given configuration.
    pub fn new(config: ExtractorSummaryConfig) -> Self {
        Self {
            config,
            extractions_performed: 0,
        }
    }

    /// Extract the most representative sentences.
    ///
    /// `sentences` is a slice of `(text, embedding)` pairs.  If
    /// `query_embedding` is provided the initial scores are cosine similarities
    /// to the query; otherwise centrality scores are used.
    pub fn extract(
        &mut self,
        sentences: &[(String, Vec<f64>)],
        query_embedding: Option<&[f64]>,
    ) -> Result<ExtractionResult, String> {
        if sentences.is_empty() {
            return Err("input sentences must not be empty".to_string());
        }

        let embeddings: Vec<&Vec<f64>> = sentences.iter().map(|(_, e)| e).collect();

        // ── 1. Initial scores ────────────────────────────────────────────
        let base_scores: Vec<f64> = if let Some(query) = query_embedding {
            embeddings
                .iter()
                .map(|e| Self::cosine_similarity(e, query))
                .collect()
        } else {
            let embs: Vec<Vec<f64>> = embeddings.iter().map(|e| (*e).clone()).collect();
            Self::centrality_scores(&embs)
        };

        // Working copy of scores that will be penalised during selection.
        let mut scores = base_scores.clone();
        let mut selected_flags = vec![false; sentences.len()];
        let mut selected_indices: Vec<usize> = Vec::new();

        // ── 2. Greedy selection ──────────────────────────────────────────
        for _ in 0..self.config.max_sentences {
            // Find the best unselected sentence.
            let best = scores
                .iter()
                .enumerate()
                .filter(|(i, _)| !selected_flags[*i])
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let (best_idx, &best_score) = match best {
                Some(pair) => pair,
                None => break, // all selected already
            };

            if best_score < self.config.similarity_threshold {
                break;
            }

            selected_flags[best_idx] = true;
            selected_indices.push(best_idx);

            // Penalise remaining candidates by their similarity to the just-selected sentence.
            for (j, s) in scores.iter_mut().enumerate() {
                if !selected_flags[j] {
                    let sim = Self::cosine_similarity(embeddings[j], embeddings[best_idx]);
                    *s -= self.config.diversity_penalty * sim;
                }
            }
        }

        // ── 3. Build result ──────────────────────────────────────────────
        let scored: Vec<ExtractorScoredSentence> = sentences
            .iter()
            .enumerate()
            .map(|(i, (text, emb))| ExtractorScoredSentence {
                index: i,
                text: text.clone(),
                embedding: emb.clone(),
                score: scores[i],
                selected: selected_flags[i],
            })
            .collect();

        let selected_embs: Vec<Vec<f64>> = selected_indices
            .iter()
            .map(|&i| sentences[i].1.clone())
            .collect();
        let all_embs: Vec<Vec<f64>> = sentences.iter().map(|(_, e)| e.clone()).collect();
        let coverage_score = Self::coverage(&selected_embs, &all_embs);

        self.extractions_performed += 1;

        Ok(ExtractionResult {
            selected_indices,
            sentences: scored,
            coverage_score,
        })
    }

    // ── Utility functions ────────────────────────────────────────────────────

    /// Cosine similarity between two vectors.
    ///
    /// Returns 0.0 when either vector has zero magnitude.
    pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
        let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let mag_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let mag_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
        if mag_a == 0.0 || mag_b == 0.0 {
            return 0.0;
        }
        dot / (mag_a * mag_b)
    }

    /// Average cosine similarity of each embedding to all others (centrality).
    pub fn centrality_scores(embeddings: &[Vec<f64>]) -> Vec<f64> {
        let n = embeddings.len();
        if n <= 1 {
            return vec![0.0; n];
        }
        let mut scores = vec![0.0_f64; n];
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    scores[i] += Self::cosine_similarity(&embeddings[i], &embeddings[j]);
                }
            }
            scores[i] /= (n - 1) as f64;
        }
        scores
    }

    /// Coverage: for each sentence in `all`, compute the maximum cosine similarity
    /// to any sentence in `selected`, then return the average of those maxima.
    ///
    /// Sentences that are themselves in `selected` are included in the average
    /// (their max similarity to a selected sentence is trivially 1.0).
    pub fn coverage(selected: &[Vec<f64>], all: &[Vec<f64>]) -> f64 {
        if selected.is_empty() || all.is_empty() {
            return 0.0;
        }
        let total: f64 = all
            .iter()
            .map(|a| {
                selected
                    .iter()
                    .map(|s| Self::cosine_similarity(a, s))
                    .fold(f64::NEG_INFINITY, f64::max)
            })
            .sum();
        total / all.len() as f64
    }

    /// Return cumulative statistics.
    pub fn stats(&self) -> SummaryExtractorStats {
        SummaryExtractorStats {
            extractions_performed: self.extractions_performed,
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_extractor() -> SemanticSummaryExtractor {
        SemanticSummaryExtractor::new(ExtractorSummaryConfig::default())
    }

    fn make_sentences(vecs: &[Vec<f64>]) -> Vec<(String, Vec<f64>)> {
        vecs.iter()
            .enumerate()
            .map(|(i, v)| (format!("sentence {i}"), v.clone()))
            .collect()
    }

    // ── cosine_similarity ────────────────────────────────────────────────

    #[test]
    fn cosine_parallel_vectors() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![2.0, 0.0, 0.0];
        let sim = SemanticSummaryExtractor::cosine_similarity(&a, &b);
        assert!(
            (sim - 1.0).abs() < 1e-9,
            "parallel vectors should have similarity 1.0"
        );
    }

    #[test]
    fn cosine_orthogonal_vectors() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = SemanticSummaryExtractor::cosine_similarity(&a, &b);
        assert!(
            sim.abs() < 1e-9,
            "orthogonal vectors should have similarity 0.0"
        );
    }

    #[test]
    fn cosine_antiparallel_vectors() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        let sim = SemanticSummaryExtractor::cosine_similarity(&a, &b);
        assert!(
            (sim + 1.0).abs() < 1e-9,
            "antiparallel vectors should have similarity -1.0"
        );
    }

    #[test]
    fn cosine_zero_vector() {
        let a = vec![0.0, 0.0];
        let b = vec![1.0, 2.0];
        assert_eq!(SemanticSummaryExtractor::cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn cosine_identical_vectors() {
        let a = vec![0.3, 0.4, 0.5];
        let sim = SemanticSummaryExtractor::cosine_similarity(&a, &a);
        assert!((sim - 1.0).abs() < 1e-9);
    }

    // ── centrality_scores ────────────────────────────────────────────────

    #[test]
    fn centrality_single_embedding() {
        let embs = vec![vec![1.0, 0.0]];
        let scores = SemanticSummaryExtractor::centrality_scores(&embs);
        assert_eq!(scores, vec![0.0]);
    }

    #[test]
    fn centrality_two_identical() {
        let embs = vec![vec![1.0, 0.0], vec![1.0, 0.0]];
        let scores = SemanticSummaryExtractor::centrality_scores(&embs);
        assert!((scores[0] - 1.0).abs() < 1e-9);
        assert!((scores[1] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn centrality_orthogonal_pair() {
        let embs = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let scores = SemanticSummaryExtractor::centrality_scores(&embs);
        assert!(scores[0].abs() < 1e-9);
        assert!(scores[1].abs() < 1e-9);
    }

    #[test]
    fn centrality_three_embeddings() {
        let embs = vec![vec![1.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]];
        let scores = SemanticSummaryExtractor::centrality_scores(&embs);
        // emb0 and emb1 are identical → sim=1; emb0 and emb2 → sim=0 → avg = 0.5
        assert!((scores[0] - 0.5).abs() < 1e-9);
        assert!((scores[1] - 0.5).abs() < 1e-9);
        assert!((scores[2] - 0.0).abs() < 1e-9);
    }

    // ── extract with query ───────────────────────────────────────────────

    #[test]
    fn extract_with_query_selects_most_similar() {
        let mut ext = default_extractor();
        let sents = make_sentences(&[vec![1.0, 0.0], vec![0.0, 1.0], vec![0.7, 0.7]]);
        let query = vec![1.0, 0.0];
        let res = ext.extract(&sents, Some(&query)).expect("should succeed");
        // sentence 0 has cosine 1.0 to query → selected first
        assert_eq!(res.selected_indices[0], 0);
    }

    #[test]
    fn extract_with_query_respects_max_sentences() {
        let mut ext = SemanticSummaryExtractor::new(ExtractorSummaryConfig {
            max_sentences: 2,
            similarity_threshold: 0.0,
            diversity_penalty: 0.0,
        });
        let sents = make_sentences(&[
            vec![1.0, 0.0],
            vec![0.9, 0.1],
            vec![0.8, 0.2],
            vec![0.7, 0.3],
        ]);
        let query = vec![1.0, 0.0];
        let res = ext.extract(&sents, Some(&query)).expect("should succeed");
        assert_eq!(res.selected_indices.len(), 2);
    }

    #[test]
    fn extract_with_query_all_selected() {
        let mut ext = SemanticSummaryExtractor::new(ExtractorSummaryConfig {
            max_sentences: 10,
            similarity_threshold: 0.0,
            diversity_penalty: 0.0,
        });
        let sents = make_sentences(&[vec![1.0, 0.0], vec![0.0, 1.0]]);
        let query = vec![1.0, 1.0];
        let res = ext.extract(&sents, Some(&query)).expect("should succeed");
        assert_eq!(res.selected_indices.len(), 2);
    }

    // ── extract without query (centrality) ───────────────────────────────

    #[test]
    fn extract_without_query_uses_centrality() {
        let mut ext = SemanticSummaryExtractor::new(ExtractorSummaryConfig {
            max_sentences: 1,
            similarity_threshold: 0.0,
            diversity_penalty: 0.0,
        });
        // Three sentences: 0 and 1 are similar, 2 is orthogonal.
        // Centrality of 0 ≈ centrality of 1 > centrality of 2.
        let sents = make_sentences(&[vec![1.0, 0.1], vec![1.0, 0.0], vec![0.0, 1.0]]);
        let res = ext.extract(&sents, None).expect("should succeed");
        assert!(
            res.selected_indices[0] == 0 || res.selected_indices[0] == 1,
            "should select one of the two similar sentences"
        );
    }

    #[test]
    fn extract_centrality_with_diversity() {
        let mut ext = SemanticSummaryExtractor::new(ExtractorSummaryConfig {
            max_sentences: 2,
            similarity_threshold: -10.0,
            diversity_penalty: 0.5,
        });
        let sents = make_sentences(&[
            vec![1.0, 0.0],
            vec![1.0, 0.01], // nearly identical to 0
            vec![0.0, 1.0],  // orthogonal
        ]);
        let res = ext.extract(&sents, None).expect("should succeed");
        // After selecting one of {0,1}, the diversity penalty should push the
        // other one down, making 2 more likely as the second pick.
        assert_eq!(res.selected_indices.len(), 2);
        // Both selected indices should be distinct
        assert_ne!(res.selected_indices[0], res.selected_indices[1]);
    }

    // ── diversity penalty ────────────────────────────────────────────────

    #[test]
    fn diversity_penalty_reduces_redundancy() {
        // Without penalty: two nearly identical sentences score highest.
        // With penalty: second pick should differ.
        let sents = make_sentences(&[vec![1.0, 0.0], vec![0.99, 0.01], vec![0.0, 1.0]]);
        let query = vec![1.0, 0.0];

        let mut no_penalty = SemanticSummaryExtractor::new(ExtractorSummaryConfig {
            max_sentences: 2,
            similarity_threshold: 0.0,
            diversity_penalty: 0.0,
        });
        let r1 = no_penalty.extract(&sents, Some(&query)).expect("ok");
        // Without penalty sentence 1 is second pick (highest remaining score).
        assert_eq!(r1.selected_indices, vec![0, 1]);

        let mut with_penalty = SemanticSummaryExtractor::new(ExtractorSummaryConfig {
            max_sentences: 2,
            similarity_threshold: 0.0,
            diversity_penalty: 2.0, // strong penalty
        });
        let r2 = with_penalty.extract(&sents, Some(&query)).expect("ok");
        // With a large penalty sentence 2 should be preferred over 1.
        assert_eq!(r2.selected_indices[0], 0);
        assert_eq!(r2.selected_indices[1], 2);
    }

    // ── max_sentences cap ────────────────────────────────────────────────

    #[test]
    fn max_sentences_caps_output() {
        let mut ext = SemanticSummaryExtractor::new(ExtractorSummaryConfig {
            max_sentences: 1,
            similarity_threshold: 0.0,
            diversity_penalty: 0.0,
        });
        let sents = make_sentences(&[vec![1.0, 0.0], vec![0.0, 1.0], vec![0.5, 0.5]]);
        let res = ext.extract(&sents, Some(&[1.0, 0.0])).expect("ok");
        assert_eq!(res.selected_indices.len(), 1);
    }

    // ── empty input ──────────────────────────────────────────────────────

    #[test]
    fn empty_input_returns_error() {
        let mut ext = default_extractor();
        let res = ext.extract(&[], None);
        assert!(res.is_err());
    }

    // ── single sentence ──────────────────────────────────────────────────

    #[test]
    fn single_sentence_selected() {
        let mut ext = SemanticSummaryExtractor::new(ExtractorSummaryConfig {
            max_sentences: 5,
            similarity_threshold: 0.0,
            diversity_penalty: 0.5,
        });
        let sents = make_sentences(&[vec![1.0, 0.0]]);
        // centrality of a single sentence is 0.0 which is >= threshold 0.0
        let res = ext.extract(&sents, None).expect("ok");
        assert_eq!(res.selected_indices.len(), 1);
        assert_eq!(res.selected_indices[0], 0);
    }

    #[test]
    fn single_sentence_with_query() {
        let mut ext = default_extractor();
        let sents = make_sentences(&[vec![1.0, 0.0]]);
        let res = ext.extract(&sents, Some(&[1.0, 0.0])).expect("ok");
        assert_eq!(res.selected_indices.len(), 1);
    }

    // ── threshold filtering ──────────────────────────────────────────────

    #[test]
    fn all_below_threshold() {
        let mut ext = SemanticSummaryExtractor::new(ExtractorSummaryConfig {
            max_sentences: 5,
            similarity_threshold: 0.99,
            diversity_penalty: 0.0,
        });
        // query is [1,0], sentence is [0,1] → sim = 0.0 < 0.99
        let sents = make_sentences(&[vec![0.0, 1.0]]);
        let res = ext.extract(&sents, Some(&[1.0, 0.0])).expect("ok");
        assert!(res.selected_indices.is_empty());
    }

    #[test]
    fn threshold_filters_partial() {
        let mut ext = SemanticSummaryExtractor::new(ExtractorSummaryConfig {
            max_sentences: 10,
            similarity_threshold: 0.9,
            diversity_penalty: 0.0,
        });
        let sents = make_sentences(&[
            vec![1.0, 0.0], // sim to query = 1.0
            vec![0.0, 1.0], // sim to query = 0.0
        ]);
        let res = ext.extract(&sents, Some(&[1.0, 0.0])).expect("ok");
        assert_eq!(res.selected_indices, vec![0]);
    }

    // ── coverage ─────────────────────────────────────────────────────────

    #[test]
    fn coverage_perfect_when_all_selected() {
        let all = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let cov = SemanticSummaryExtractor::coverage(&all, &all);
        assert!(
            (cov - 1.0).abs() < 1e-9,
            "coverage should be 1.0 when all selected"
        );
    }

    #[test]
    fn coverage_zero_when_none_selected() {
        let all = vec![vec![1.0, 0.0]];
        let cov = SemanticSummaryExtractor::coverage(&[], &all);
        assert_eq!(cov, 0.0);
    }

    #[test]
    fn coverage_partial() {
        let selected = vec![vec![1.0, 0.0]];
        let all = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let cov = SemanticSummaryExtractor::coverage(&selected, &all);
        // sentence 0 → max sim to selected = 1.0
        // sentence 1 → max sim to selected = 0.0
        // avg = 0.5
        assert!((cov - 0.5).abs() < 1e-9);
    }

    #[test]
    fn coverage_with_similar_sentences() {
        let selected = vec![vec![1.0, 0.0]];
        let all = vec![vec![1.0, 0.0], vec![0.9, 0.1]];
        let cov = SemanticSummaryExtractor::coverage(&selected, &all);
        // sentence 0 → 1.0
        // sentence 1 → cos([1,0],[0.9,0.1]) ≈ 0.994
        // avg ≈ 0.997
        assert!(cov > 0.99);
    }

    // ── stats tracking ───────────────────────────────────────────────────

    #[test]
    fn stats_tracks_extractions() {
        let mut ext = default_extractor();
        assert_eq!(ext.stats().extractions_performed, 0);

        let sents = make_sentences(&[vec![1.0, 0.0]]);
        let _ = ext.extract(&sents, Some(&[1.0, 0.0]));
        assert_eq!(ext.stats().extractions_performed, 1);

        let _ = ext.extract(&sents, None);
        assert_eq!(ext.stats().extractions_performed, 2);
    }

    #[test]
    fn stats_not_incremented_on_error() {
        let mut ext = default_extractor();
        let _ = ext.extract(&[], None); // error
        assert_eq!(ext.stats().extractions_performed, 0);
    }

    // ── deterministic output ─────────────────────────────────────────────

    #[test]
    fn deterministic_results() {
        let sents = make_sentences(&[vec![1.0, 0.0], vec![0.5, 0.5], vec![0.0, 1.0]]);
        let query = vec![0.6, 0.4];

        let mut ext1 = default_extractor();
        let mut ext2 = default_extractor();

        let r1 = ext1.extract(&sents, Some(&query)).expect("ok");
        let r2 = ext2.extract(&sents, Some(&query)).expect("ok");

        assert_eq!(r1.selected_indices, r2.selected_indices);
        assert!((r1.coverage_score - r2.coverage_score).abs() < 1e-12);
    }

    // ── selected flag ────────────────────────────────────────────────────

    #[test]
    fn selected_flags_match_indices() {
        let mut ext = SemanticSummaryExtractor::new(ExtractorSummaryConfig {
            max_sentences: 2,
            similarity_threshold: 0.0,
            diversity_penalty: 0.0,
        });
        let sents = make_sentences(&[vec![1.0, 0.0], vec![0.5, 0.5], vec![0.0, 1.0]]);
        let res = ext.extract(&sents, Some(&[1.0, 0.0])).expect("ok");

        for sent in &res.sentences {
            if res.selected_indices.contains(&sent.index) {
                assert!(sent.selected);
            } else {
                assert!(!sent.selected);
            }
        }
    }

    // ── default config ───────────────────────────────────────────────────

    #[test]
    fn default_config_values() {
        let cfg = ExtractorSummaryConfig::default();
        assert_eq!(cfg.max_sentences, 5);
        assert!((cfg.similarity_threshold - 0.3).abs() < 1e-9);
        assert!((cfg.diversity_penalty - 0.5).abs() < 1e-9);
    }

    // ── high-dimensional embeddings ──────────────────────────────────────

    #[test]
    fn high_dimensional_embeddings() {
        let dim = 128;
        let mut ext = default_extractor();
        let mut v1 = vec![0.0; dim];
        v1[0] = 1.0;
        let mut v2 = vec![0.0; dim];
        v2[1] = 1.0;
        let mut v3 = vec![0.0; dim];
        v3[0] = 0.7;
        v3[1] = 0.7;

        let sents = make_sentences(&[v1.clone(), v2, v3]);
        let res = ext.extract(&sents, Some(&v1)).expect("ok");
        assert!(!res.selected_indices.is_empty());
    }

    // ── coverage integrated into extract result ──────────────────────────

    #[test]
    fn extract_result_coverage_is_consistent() {
        let mut ext = SemanticSummaryExtractor::new(ExtractorSummaryConfig {
            max_sentences: 2,
            similarity_threshold: 0.0,
            diversity_penalty: 0.0,
        });
        let sents = make_sentences(&[vec![1.0, 0.0], vec![0.0, 1.0], vec![0.5, 0.5]]);
        let res = ext.extract(&sents, Some(&[1.0, 0.0])).expect("ok");

        // Recompute coverage manually
        let selected_embs: Vec<Vec<f64>> = res
            .selected_indices
            .iter()
            .map(|&i| sents[i].1.clone())
            .collect();
        let all_embs: Vec<Vec<f64>> = sents.iter().map(|(_, e)| e.clone()).collect();
        let expected_cov = SemanticSummaryExtractor::coverage(&selected_embs, &all_embs);

        assert!((res.coverage_score - expected_cov).abs() < 1e-12);
    }

    // ── edge: all sentences identical ────────────────────────────────────

    #[test]
    fn all_identical_sentences() {
        let mut ext = SemanticSummaryExtractor::new(ExtractorSummaryConfig {
            max_sentences: 3,
            similarity_threshold: 0.0,
            diversity_penalty: 0.5,
        });
        let sents = make_sentences(&[vec![1.0, 0.0], vec![1.0, 0.0], vec![1.0, 0.0]]);
        let res = ext.extract(&sents, Some(&[1.0, 0.0])).expect("ok");
        // Should still select at least one
        assert!(!res.selected_indices.is_empty());
    }

    // ── negative scores after heavy penalty ──────────────────────────────

    #[test]
    fn negative_scores_below_threshold_not_selected() {
        let mut ext = SemanticSummaryExtractor::new(ExtractorSummaryConfig {
            max_sentences: 5,
            similarity_threshold: 0.3,
            diversity_penalty: 5.0, // extremely high
        });
        let sents = make_sentences(&[vec![1.0, 0.0], vec![0.9, 0.1], vec![0.8, 0.2]]);
        let res = ext.extract(&sents, Some(&[1.0, 0.0])).expect("ok");
        // First is selected (score 1.0 > 0.3), but after heavy penalty
        // others may drop below threshold
        assert!(res.selected_indices.contains(&0));
        // The other two should have been penalised below 0.3
        assert!(res.selected_indices.len() <= 2);
    }
}
