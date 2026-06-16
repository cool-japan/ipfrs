//! Maximal Marginal Relevance (MMR) diversification for semantic search results.
//!
//! This module implements the MMR algorithm which selects top-k results that
//! balance relevance to the query with diversity among the selected items.
//! The trade-off is controlled by a `lambda` parameter: 1.0 = pure relevance,
//! 0.0 = pure diversity.

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// A single candidate for diversified selection.
#[derive(Clone, Debug)]
pub struct DiversificationCandidate {
    /// Opaque document identifier.
    pub doc_id: u64,
    /// Dense embedding vector for this document.
    pub embedding: Vec<f32>,
    /// Similarity score to the query (higher = more relevant).
    pub relevance_score: f32,
}

/// Configuration for the MMR diversification algorithm.
#[derive(Clone, Debug)]
pub struct DiversifierConfig {
    /// Trade-off between relevance and diversity.
    /// - `1.0` = pure relevance (select by relevance_score only)
    /// - `0.0` = pure diversity (select most different items)
    /// - `0.5` = balanced (default)
    pub lambda: f32,
}

impl Default for DiversifierConfig {
    fn default() -> Self {
        Self { lambda: 0.5 }
    }
}

/// A single result from the diversified selection process.
#[derive(Clone, Debug)]
pub struct DiversifiedResult {
    /// Opaque document identifier.
    pub doc_id: u64,
    /// Similarity score to the query (inherited from the candidate).
    pub relevance_score: f32,
    /// MMR score used for selection in this round.
    pub mmr_score: f32,
    /// 0-indexed order in which this document was selected.
    pub selection_rank: usize,
}

/// Aggregate statistics across all `select` calls made on a
/// [`SemanticDiversifier`] instance.
#[derive(Clone, Debug, Default)]
pub struct DiversifierStats {
    /// Number of times [`SemanticDiversifier::select`] was called (runs that
    /// actually produced at least one result count, as do empty-result calls).
    pub total_runs: u64,
    /// Total number of candidates processed across all runs (sum of the input
    /// `candidates` lengths).
    pub total_candidates_processed: u64,
    /// Total number of results selected across all runs.
    pub total_selected: u64,
    /// Running mean of the `lambda` value used across all runs.
    pub avg_lambda: f64,
}

// ---------------------------------------------------------------------------
// Mathematical helpers
// ---------------------------------------------------------------------------

/// Cosine similarity between two vectors.
///
/// Returns `0.0` when:
/// - Either slice is empty.
/// - The slices have different lengths.
/// - Either vector has zero magnitude.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || b.is_empty() || a.len() != b.len() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

// ---------------------------------------------------------------------------
// Core engine
// ---------------------------------------------------------------------------

/// Diversifies a ranked list of candidates using Maximal Marginal Relevance.
///
/// Each call to [`select`](SemanticDiversifier::select) greedily picks items
/// that simultaneously maximise relevance to the query and minimise similarity
/// to previously selected items.
pub struct SemanticDiversifier {
    /// MMR configuration (lambda, etc.).
    pub config: DiversifierConfig,
    /// Aggregate statistics accumulated across all `select` calls.
    stats: DiversifierStats,
}

impl SemanticDiversifier {
    /// Create a new diversifier with the given configuration.
    pub fn new(config: DiversifierConfig) -> Self {
        Self {
            config,
            stats: DiversifierStats::default(),
        }
    }

    /// Select up to `k` diversified results from `candidates` using MMR.
    ///
    /// The algorithm greedily builds a selected set `S`, starting from empty.
    /// Each iteration scores every remaining candidate with:
    ///
    /// ```text
    /// mmr = lambda * relevance_score - (1 - lambda) * max_sim_to_selected
    /// ```
    ///
    /// Ties are broken by `doc_id` ascending.
    ///
    /// Returns an empty `Vec` when `candidates` is empty or `k == 0`.
    pub fn select(
        &mut self,
        _query: &[f32],
        candidates: Vec<DiversificationCandidate>,
        k: usize,
    ) -> Vec<DiversifiedResult> {
        // Record the run regardless of whether results are produced.
        let n = candidates.len();
        self.update_stats_pre(n as u64);

        if candidates.is_empty() || k == 0 {
            return Vec::new();
        }

        let target = k.min(n);
        let lambda = self.config.lambda;

        // Track which candidates are still available using a boolean mask.
        let mut available: Vec<bool> = vec![true; n];
        // Selected results in order of selection.
        let mut selected: Vec<DiversifiedResult> = Vec::with_capacity(target);
        // Embeddings of the already-selected documents (for similarity lookups).
        let mut selected_embeddings: Vec<&[f32]> = Vec::with_capacity(target);

        for rank in 0..target {
            // --- Score every remaining candidate ---
            let mut best_idx: Option<usize> = None;
            let mut best_mmr = f32::NEG_INFINITY;
            let mut best_doc_id: u64 = u64::MAX;

            for (idx, candidate) in candidates.iter().enumerate() {
                if !available[idx] {
                    continue;
                }

                // Maximum similarity to any already-selected document.
                let max_sim_to_selected: f32 = if selected_embeddings.is_empty() {
                    0.0
                } else {
                    selected_embeddings
                        .iter()
                        .map(|s| cosine_similarity(&candidate.embedding, s))
                        .fold(f32::NEG_INFINITY, f32::max)
                };

                let mmr = lambda * candidate.relevance_score - (1.0 - lambda) * max_sim_to_selected;

                // Tie-breaking: prefer lower doc_id.
                let is_better =
                    mmr > best_mmr || (mmr == best_mmr && candidate.doc_id < best_doc_id);

                if is_better {
                    best_mmr = mmr;
                    best_idx = Some(idx);
                    best_doc_id = candidate.doc_id;
                }
            }

            // `best_idx` is always `Some` here because there is at least one
            // available candidate (we loop `target` times which is ≤ available count).
            if let Some(idx) = best_idx {
                available[idx] = false;
                selected_embeddings.push(&candidates[idx].embedding);
                selected.push(DiversifiedResult {
                    doc_id: candidates[idx].doc_id,
                    relevance_score: candidates[idx].relevance_score,
                    mmr_score: best_mmr,
                    selection_rank: rank,
                });
            }
        }

        let num_selected = selected.len() as u64;
        self.update_stats_post(num_selected, lambda);

        selected
    }

    /// Return a reference to the accumulated statistics.
    pub fn stats(&self) -> &DiversifierStats {
        &self.stats
    }

    /// Override the lambda trade-off parameter, clamped to `[0.0, 1.0]`.
    pub fn set_lambda(&mut self, lambda: f32) {
        self.config.lambda = lambda.clamp(0.0, 1.0);
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Record the start of a new run (candidates count).
    fn update_stats_pre(&mut self, candidate_count: u64) {
        self.stats.total_runs += 1;
        self.stats.total_candidates_processed += candidate_count;
    }

    /// Record the end of a run (selected count and lambda contribution).
    fn update_stats_post(&mut self, num_selected: u64, lambda: f32) {
        self.stats.total_selected += num_selected;
        // Maintain running mean of lambda.
        let n = self.stats.total_runs as f64;
        self.stats.avg_lambda = self.stats.avg_lambda + (lambda as f64 - self.stats.avg_lambda) / n;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_candidate(
        doc_id: u64,
        embedding: Vec<f32>,
        relevance_score: f32,
    ) -> DiversificationCandidate {
        DiversificationCandidate {
            doc_id,
            embedding,
            relevance_score,
        }
    }

    fn default_diversifier() -> SemanticDiversifier {
        SemanticDiversifier::new(DiversifierConfig::default())
    }

    // 1. select returns empty for empty candidates
    #[test]
    fn test_select_empty_candidates() {
        let mut d = default_diversifier();
        let query = vec![1.0f32, 0.0];
        let result = d.select(&query, vec![], 5);
        assert!(result.is_empty());
    }

    // 2. select returns empty for k == 0
    #[test]
    fn test_select_k_zero() {
        let mut d = default_diversifier();
        let query = vec![1.0f32, 0.0];
        let candidates = vec![
            make_candidate(1, vec![1.0, 0.0], 0.9),
            make_candidate(2, vec![0.0, 1.0], 0.8),
        ];
        let result = d.select(&query, candidates, 0);
        assert!(result.is_empty());
    }

    // 3. lambda=1.0 (pure relevance): selects by relevance_score order
    #[test]
    fn test_lambda_one_pure_relevance() {
        let mut d = SemanticDiversifier::new(DiversifierConfig { lambda: 1.0 });
        let query = vec![1.0f32, 0.0];
        let candidates = vec![
            make_candidate(1, vec![1.0, 0.0], 0.5),
            make_candidate(2, vec![1.0, 0.0], 0.9),
            make_candidate(3, vec![1.0, 0.0], 0.7),
        ];
        let result = d.select(&query, candidates, 3);
        assert_eq!(result.len(), 3);
        // Should be ordered by relevance: 0.9, 0.7, 0.5
        assert_eq!(result[0].doc_id, 2); // relevance 0.9
        assert_eq!(result[1].doc_id, 3); // relevance 0.7
        assert_eq!(result[2].doc_id, 1); // relevance 0.5
    }

    // 4. lambda=0.0 (pure diversity): selects most different items
    #[test]
    fn test_lambda_zero_pure_diversity() {
        let mut d = SemanticDiversifier::new(DiversifierConfig { lambda: 0.0 });
        let query = vec![1.0f32, 0.0, 0.0];
        // Three orthogonal unit vectors — max diversity selects all with equal mmr=0
        // but first selection (S empty) always picks the first by doc_id tie-break
        let candidates = vec![
            make_candidate(10, vec![1.0, 0.0, 0.0], 0.9),
            make_candidate(20, vec![0.0, 1.0, 0.0], 0.1),
            make_candidate(30, vec![0.0, 0.0, 1.0], 0.5),
        ];
        let result = d.select(&query, candidates, 3);
        assert_eq!(result.len(), 3);
        // All selected; just verify all doc_ids are present
        let doc_ids: Vec<u64> = result.iter().map(|r| r.doc_id).collect();
        assert!(doc_ids.contains(&10));
        assert!(doc_ids.contains(&20));
        assert!(doc_ids.contains(&30));
    }

    // 5. selection_rank assigned correctly (0, 1, 2, ...)
    #[test]
    fn test_selection_rank_assigned_correctly() {
        let mut d = default_diversifier();
        let query = vec![1.0f32, 0.0];
        let candidates = vec![
            make_candidate(1, vec![1.0, 0.0], 0.9),
            make_candidate(2, vec![0.0, 1.0], 0.8),
            make_candidate(3, vec![0.5, 0.5], 0.7),
        ];
        let result = d.select(&query, candidates, 3);
        for (i, r) in result.iter().enumerate() {
            assert_eq!(r.selection_rank, i, "rank mismatch at position {i}");
        }
    }

    // 6. k larger than candidates: returns all candidates
    #[test]
    fn test_k_larger_than_candidates() {
        let mut d = default_diversifier();
        let query = vec![1.0f32, 0.0];
        let candidates = vec![
            make_candidate(1, vec![1.0, 0.0], 0.9),
            make_candidate(2, vec![0.0, 1.0], 0.5),
        ];
        let result = d.select(&query, candidates, 100);
        assert_eq!(result.len(), 2);
    }

    // 7a. cosine_similarity: orthogonal vectors → 0.0
    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0f32, 0.0];
        let b = vec![0.0f32, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-6, "Expected ~0.0, got {sim}");
    }

    // 7b. cosine_similarity: identical vectors → 1.0
    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![0.3f32, 0.4, 0.5];
        let sim = cosine_similarity(&a, &a);
        assert!((sim - 1.0).abs() < 1e-6, "Expected ~1.0, got {sim}");
    }

    // 7c. cosine_similarity: zero vector → 0.0
    #[test]
    fn test_cosine_similarity_zero_vector() {
        let a = vec![0.0f32, 0.0, 0.0];
        let b = vec![1.0f32, 2.0, 3.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-6, "Expected 0.0, got {sim}");
    }

    // 7d. cosine_similarity: dimension mismatch → 0.0
    #[test]
    fn test_cosine_similarity_dimension_mismatch() {
        let a = vec![1.0f32, 0.0];
        let b = vec![1.0f32, 0.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-6, "Expected 0.0, got {sim}");
    }

    // 7e. cosine_similarity: empty slices → 0.0
    #[test]
    fn test_cosine_similarity_empty() {
        let sim = cosine_similarity(&[], &[]);
        assert!(sim.abs() < 1e-6, "Expected 0.0, got {sim}");
    }

    // 8. stats accumulate correctly across multiple runs
    #[test]
    fn test_stats_accumulate() {
        let mut d = default_diversifier();
        let query = vec![1.0f32, 0.0];
        let c1 = vec![
            make_candidate(1, vec![1.0, 0.0], 0.9),
            make_candidate(2, vec![0.0, 1.0], 0.5),
        ];
        let c2 = vec![make_candidate(3, vec![0.5, 0.5], 0.7)];
        d.select(&query, c1, 2);
        d.select(&query, c2, 1);
        let stats = d.stats();
        assert_eq!(stats.total_runs, 2);
        assert_eq!(stats.total_candidates_processed, 3);
        assert_eq!(stats.total_selected, 3);
    }

    // 9. set_lambda clamps to [0.0, 1.0] — above 1.0
    #[test]
    fn test_set_lambda_clamps_above_one() {
        let mut d = default_diversifier();
        d.set_lambda(2.5);
        assert!((d.config.lambda - 1.0).abs() < 1e-6);
    }

    // 10. set_lambda clamps to [0.0, 1.0] — below 0.0
    #[test]
    fn test_set_lambda_clamps_below_zero() {
        let mut d = default_diversifier();
        d.set_lambda(-0.3);
        assert!(d.config.lambda.abs() < 1e-6);
    }

    // 11. set_lambda within [0.0, 1.0] is stored unchanged
    #[test]
    fn test_set_lambda_valid_value() {
        let mut d = default_diversifier();
        d.set_lambda(0.7);
        assert!((d.config.lambda - 0.7).abs() < 1e-6);
    }

    // 12. Tie-breaking by doc_id ascending
    #[test]
    fn test_tie_breaking_by_doc_id_ascending() {
        // lambda=1.0 and identical relevance scores → tie → lower doc_id wins
        let mut d = SemanticDiversifier::new(DiversifierConfig { lambda: 1.0 });
        let query = vec![1.0f32, 0.0];
        let candidates = vec![
            make_candidate(30, vec![1.0, 0.0], 0.8),
            make_candidate(10, vec![1.0, 0.0], 0.8),
            make_candidate(20, vec![1.0, 0.0], 0.8),
        ];
        let result = d.select(&query, candidates, 1);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].doc_id, 10, "lowest doc_id should win tie");
    }

    // 13. Mixed lambda=0.5 produces balanced selection
    #[test]
    fn test_mixed_lambda_balanced_selection() {
        // Two candidates: one very relevant but redundant with the first selection,
        // one less relevant but diverse. With lambda=0.5 and the diverse candidate
        // having zero similarity to the first selected, it should be preferred over
        // a highly similar but more relevant candidate.
        let mut d = SemanticDiversifier::new(DiversifierConfig { lambda: 0.5 });
        let query = vec![1.0f32, 0.0, 0.0];

        let candidates = vec![
            // First selection will be this one (highest relevance_score).
            make_candidate(1, vec![1.0, 0.0, 0.0], 1.0),
            // Same direction as doc 1 — high similarity penalty when selected 2nd.
            make_candidate(2, vec![1.0, 0.0, 0.0], 0.9),
            // Orthogonal to doc 1 — no similarity penalty.
            make_candidate(3, vec![0.0, 1.0, 0.0], 0.5),
        ];

        let result = d.select(&query, candidates, 2);
        assert_eq!(result.len(), 2);
        assert_eq!(
            result[0].doc_id, 1,
            "first selection should be doc 1 (highest relevance)"
        );

        // For second selection:
        // doc 2: mmr = 0.5*0.9 - 0.5*1.0 = 0.45 - 0.5 = -0.05
        // doc 3: mmr = 0.5*0.5 - 0.5*0.0 = 0.25 - 0.0 = 0.25
        // doc 3 wins despite lower relevance due to diversity bonus.
        assert_eq!(
            result[1].doc_id, 3,
            "second selection should be doc 3 (diversity)"
        );
    }

    // 14. DiversifierConfig default lambda is 0.5
    #[test]
    fn test_config_default_lambda() {
        let config = DiversifierConfig::default();
        assert!((config.lambda - 0.5).abs() < 1e-6);
    }

    // 15. stats starts at zero
    #[test]
    fn test_stats_initial_zero() {
        let d = default_diversifier();
        let s = d.stats();
        assert_eq!(s.total_runs, 0);
        assert_eq!(s.total_candidates_processed, 0);
        assert_eq!(s.total_selected, 0);
        assert!(s.avg_lambda.abs() < 1e-9);
    }

    // 16. Empty candidates still increments total_runs
    #[test]
    fn test_empty_candidates_increments_runs() {
        let mut d = default_diversifier();
        d.select(&[1.0f32], vec![], 5);
        assert_eq!(d.stats().total_runs, 1);
        assert_eq!(d.stats().total_candidates_processed, 0);
        assert_eq!(d.stats().total_selected, 0);
    }

    // 17. k=0 still increments total_runs (because update_stats_pre is called first)
    #[test]
    fn test_k_zero_increments_runs() {
        let mut d = default_diversifier();
        let candidates = vec![make_candidate(1, vec![1.0f32], 0.9)];
        d.select(&[1.0f32], candidates, 0);
        assert_eq!(d.stats().total_runs, 1);
        assert_eq!(d.stats().total_candidates_processed, 1);
        assert_eq!(d.stats().total_selected, 0);
    }

    // 18. avg_lambda computed correctly over two runs with different lambdas
    #[test]
    fn test_avg_lambda_two_runs() {
        let mut d = SemanticDiversifier::new(DiversifierConfig { lambda: 0.2 });
        let query = vec![1.0f32, 0.0];
        // Run 1: lambda = 0.2
        d.select(&query, vec![make_candidate(1, vec![1.0, 0.0], 0.9)], 1);
        // Run 2: lambda = 0.8
        d.set_lambda(0.8);
        d.select(&query, vec![make_candidate(2, vec![0.0, 1.0], 0.7)], 1);
        // mean of 0.2 and 0.8 = 0.5
        let avg = d.stats().avg_lambda;
        assert!((avg - 0.5).abs() < 1e-6, "avg_lambda={avg}");
    }

    // 19. Single candidate is always selected when k >= 1
    #[test]
    fn test_single_candidate_selected() {
        let mut d = default_diversifier();
        let query = vec![0.5f32, 0.5];
        let result = d.select(&query, vec![make_candidate(42, vec![0.5, 0.5], 0.8)], 1);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].doc_id, 42);
        assert_eq!(result[0].selection_rank, 0);
    }

    // 20. cosine_similarity: parallel (same direction) vectors → 1.0
    #[test]
    fn test_cosine_similarity_parallel() {
        let a = vec![2.0f32, 0.0];
        let b = vec![5.0f32, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 1e-6, "Expected ~1.0, got {sim}");
    }

    // 21. cosine_similarity: anti-parallel vectors → -1.0
    #[test]
    fn test_cosine_similarity_anti_parallel() {
        let a = vec![1.0f32, 0.0];
        let b = vec![-1.0f32, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim + 1.0).abs() < 1e-6, "Expected ~-1.0, got {sim}");
    }

    // 22. Selection order matches selection_rank field
    #[test]
    fn test_selection_order_matches_rank() {
        let mut d = SemanticDiversifier::new(DiversifierConfig { lambda: 1.0 });
        let query = vec![1.0f32, 0.0];
        let candidates = vec![
            make_candidate(1, vec![1.0, 0.0], 0.3),
            make_candidate(2, vec![1.0, 0.0], 0.8),
            make_candidate(3, vec![1.0, 0.0], 0.6),
            make_candidate(4, vec![1.0, 0.0], 0.1),
        ];
        let result = d.select(&query, candidates, 4);
        assert_eq!(result.len(), 4);
        for (pos, r) in result.iter().enumerate() {
            assert_eq!(r.selection_rank, pos);
        }
        // Order should be doc 2, 3, 1, 4 (decreasing relevance)
        assert_eq!(result[0].doc_id, 2);
        assert_eq!(result[1].doc_id, 3);
        assert_eq!(result[2].doc_id, 1);
        assert_eq!(result[3].doc_id, 4);
    }

    // 23. Total selected never exceeds total candidates
    #[test]
    fn test_total_selected_never_exceeds_candidates() {
        let mut d = default_diversifier();
        let query = vec![1.0f32, 0.0];
        let c = vec![
            make_candidate(1, vec![1.0, 0.0], 0.9),
            make_candidate(2, vec![0.0, 1.0], 0.8),
        ];
        let result = d.select(&query, c, 10);
        assert!(result.len() <= 2);
        assert_eq!(result.len(), 2);
    }

    // 24. MMR score field is populated on DiversifiedResult
    #[test]
    fn test_mmr_score_populated() {
        let mut d = SemanticDiversifier::new(DiversifierConfig { lambda: 1.0 });
        let query = vec![1.0f32, 0.0];
        let candidates = vec![make_candidate(1, vec![1.0, 0.0], 0.8)];
        let result = d.select(&query, candidates, 1);
        assert_eq!(result.len(), 1);
        // With empty S and lambda=1.0: mmr = 1.0 * 0.8 - 0.0 * 0.0 = 0.8
        assert!(
            (result[0].mmr_score - 0.8).abs() < 1e-6,
            "mmr_score={}",
            result[0].mmr_score
        );
    }

    // 25. DiversifiedResult.relevance_score matches candidate.relevance_score
    #[test]
    fn test_relevance_score_preserved() {
        let mut d = default_diversifier();
        let query = vec![1.0f32, 0.0];
        let candidates = vec![
            make_candidate(1, vec![1.0, 0.0], 0.777),
            make_candidate(2, vec![0.0, 1.0], 0.333),
        ];
        let result = d.select(&query, candidates, 2);
        let by_doc: std::collections::HashMap<u64, f32> = result
            .iter()
            .map(|r| (r.doc_id, r.relevance_score))
            .collect();
        assert!((by_doc[&1] - 0.777).abs() < 1e-5);
        assert!((by_doc[&2] - 0.333).abs() < 1e-5);
    }

    // 26. stats() returns a reference (not consuming self)
    #[test]
    fn test_stats_returns_reference() {
        let d = default_diversifier();
        let _s1 = d.stats();
        let _s2 = d.stats(); // can call multiple times
    }

    // 27. Pure diversity with two identical embeddings: tie broken by doc_id
    #[test]
    fn test_pure_diversity_identical_embeddings_tie_break() {
        let mut d = SemanticDiversifier::new(DiversifierConfig { lambda: 0.0 });
        let query = vec![1.0f32, 0.0];
        // Both have identical embeddings and same relevance → first round tie at mmr=0
        let candidates = vec![
            make_candidate(5, vec![1.0, 0.0], 0.5),
            make_candidate(3, vec![1.0, 0.0], 0.5),
        ];
        let result = d.select(&query, candidates, 1);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].doc_id, 3, "lower doc_id should win tie");
    }

    // 28. total_candidates_processed accumulates correctly across empty and non-empty runs
    #[test]
    fn test_total_candidates_processed_cumulative() {
        let mut d = default_diversifier();
        let q = vec![1.0f32];
        d.select(&q, vec![], 5); // 0 candidates
        d.select(&q, vec![make_candidate(1, vec![1.0], 0.9)], 3); // 1 candidate
        d.select(
            &q,
            vec![
                make_candidate(2, vec![0.5], 0.8),
                make_candidate(3, vec![0.1], 0.7),
            ],
            2,
        ); // 2 candidates
        assert_eq!(d.stats().total_candidates_processed, 3);
        assert_eq!(d.stats().total_runs, 3);
    }
}
