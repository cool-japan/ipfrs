//! Multi-Modal Search Coordinator — cross-modality result fusion and deduplication.
//!
//! Coordinates search across text, image, audio, code, and structured modalities,
//! fusing per-modality result lists into a unified ranked output using configurable
//! fusion strategies (ScoreSum, ScoreMax, WeightedSum, RankFusion).

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Modality
// ---------------------------------------------------------------------------

/// Supported search modalities.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Modality {
    /// Natural-language text embeddings.
    Text,
    /// Image embeddings (e.g., CLIP, ResNet).
    Image,
    /// Audio embeddings (e.g., Wav2Vec, CLAP).
    Audio,
    /// Source-code embeddings (e.g., CodeBERT).
    Code,
    /// Structured / tabular data embeddings.
    Structured,
}

// ---------------------------------------------------------------------------
// ModalityResult
// ---------------------------------------------------------------------------

/// A single result returned from one modality's search.
#[derive(Clone, Debug)]
pub struct ModalityResult {
    /// Unique identifier for this result (shared across modalities for the same item).
    pub result_id: u64,
    /// Which modality produced this result.
    pub modality: Modality,
    /// Relevance score from the modality's ranker (higher = more relevant).
    pub score: f64,
    /// Content identifier (CID) of the underlying resource.
    pub cid: String,
}

// ---------------------------------------------------------------------------
// FusionStrategy
// ---------------------------------------------------------------------------

/// How scores from different modalities are combined into a single fused score.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FusionStrategy {
    /// Sum all per-modality scores for the same `result_id`.
    ScoreSum,
    /// Take the maximum per-modality score for the same `result_id`.
    ScoreMax,
    /// Weighted sum: each modality's score is multiplied by its weight before summing.
    WeightedSum {
        /// Weight applied to the Text modality score.
        text_w: f64,
        /// Weight applied to the Image modality score.
        image_w: f64,
        /// Weight applied to the Audio modality score.
        audio_w: f64,
        /// Weight applied to the Code modality score.
        code_w: f64,
        /// Weight applied to the Structured modality score.
        struct_w: f64,
    },
    /// Reciprocal rank fusion: fused_score = Σ 1 / (rank + 60) across modalities.
    ///
    /// Rank is 0-based within each modality's result list (best rank = 0).
    RankFusion,
}

// ---------------------------------------------------------------------------
// SearchQuery
// ---------------------------------------------------------------------------

/// Describes a multi-modal search request.
#[derive(Clone, Debug)]
pub struct SearchQuery {
    /// Caller-assigned query identifier.
    pub query_id: u64,
    /// Which modalities to search.
    pub modalities: Vec<Modality>,
    /// Number of top results to return after fusion.
    pub k: usize,
}

// ---------------------------------------------------------------------------
// FusedResult
// ---------------------------------------------------------------------------

/// A result after cross-modal fusion and deduplication.
#[derive(Clone, Debug)]
pub struct FusedResult {
    /// Unique identifier (matches `ModalityResult::result_id`).
    pub result_id: u64,
    /// Combined score after applying the fusion strategy.
    pub fused_score: f64,
    /// Modalities that contributed to this result.
    pub contributing_modalities: Vec<Modality>,
    /// Content identifier inherited from the contributing results.
    pub cid: String,
}

// ---------------------------------------------------------------------------
// CoordinatorStats
// ---------------------------------------------------------------------------

/// Accumulated statistics for a `MultiModalSearchCoordinator`.
#[derive(Clone, Debug, Default)]
pub struct CoordinatorStats {
    /// Total number of `fuse_results` calls processed.
    pub total_queries: u64,
    /// How many times each modality appeared in a query's `modalities` list.
    pub modality_counts: HashMap<Modality, u64>,
}

impl CoordinatorStats {
    /// Returns the modality that has been searched the most, or `None` if no
    /// queries have been processed yet.
    pub fn most_used_modality(&self) -> Option<Modality> {
        self.modality_counts
            .iter()
            .max_by_key(|(_m, &count)| count)
            .map(|(&m, _)| m)
    }
}

// ---------------------------------------------------------------------------
// MultiModalSearchCoordinator
// ---------------------------------------------------------------------------

/// Coordinates multi-modal search by fusing per-modality result lists.
///
/// # Example
/// ```
/// use ipfrs_semantic::multimodal_search::{
///     MultiModalSearchCoordinator, Modality, ModalityResult, SearchQuery, FusionStrategy,
/// };
/// use std::collections::HashMap;
///
/// let mut coordinator = MultiModalSearchCoordinator::new();
///
/// let query = SearchQuery { query_id: 1, modalities: vec![Modality::Text], k: 5 };
/// let mut per_modality: HashMap<Modality, Vec<ModalityResult>> = HashMap::new();
/// per_modality.insert(Modality::Text, vec![
///     ModalityResult { result_id: 42, modality: Modality::Text, score: 0.9, cid: "bafy…".into() },
/// ]);
///
/// let fused = coordinator.fuse_results(&query, per_modality, FusionStrategy::ScoreSum);
/// assert_eq!(fused[0].result_id, 42);
/// ```
pub struct MultiModalSearchCoordinator {
    /// Running statistics.
    pub stats: CoordinatorStats,
}

impl MultiModalSearchCoordinator {
    /// Create a new coordinator with zeroed statistics.
    pub fn new() -> Self {
        Self {
            stats: CoordinatorStats::default(),
        }
    }

    /// Fuse per-modality result lists according to `strategy`.
    ///
    /// Steps:
    /// 1. Update stats.
    /// 2. Build a per-`result_id` accumulator.
    /// 3. Apply the fusion strategy.
    /// 4. Deduplicate (merge `contributing_modalities`), sort descending by
    ///    `fused_score`, return the top-`query.k` results.
    pub fn fuse_results(
        &mut self,
        query: &SearchQuery,
        per_modality_results: HashMap<Modality, Vec<ModalityResult>>,
        strategy: FusionStrategy,
    ) -> Vec<FusedResult> {
        // --- 1. Update stats ---
        self.stats.total_queries += 1;
        for modality in &query.modalities {
            *self.stats.modality_counts.entry(*modality).or_insert(0) += 1;
        }

        // --- 2. Accumulate raw results per result_id ---
        // Map: result_id -> list of (modality, score, rank_within_modality, cid)
        let mut accumulator: HashMap<u64, Vec<(Modality, f64, usize, String)>> = HashMap::new();

        for (modality, results) in &per_modality_results {
            for (rank, r) in results.iter().enumerate() {
                accumulator.entry(r.result_id).or_default().push((
                    *modality,
                    r.score,
                    rank,
                    r.cid.clone(),
                ));
            }
        }

        // --- 3. Apply fusion strategy ---
        let mut fused: Vec<FusedResult> = accumulator
            .into_iter()
            .map(|(result_id, contributions)| {
                let fused_score = match strategy {
                    FusionStrategy::ScoreSum => {
                        contributions.iter().map(|(_, score, _, _)| *score).sum()
                    }
                    FusionStrategy::ScoreMax => contributions
                        .iter()
                        .map(|(_, score, _, _)| *score)
                        .fold(f64::NEG_INFINITY, f64::max),
                    FusionStrategy::WeightedSum {
                        text_w,
                        image_w,
                        audio_w,
                        code_w,
                        struct_w,
                    } => contributions
                        .iter()
                        .map(|(modality, score, _, _)| {
                            let weight = match modality {
                                Modality::Text => text_w,
                                Modality::Image => image_w,
                                Modality::Audio => audio_w,
                                Modality::Code => code_w,
                                Modality::Structured => struct_w,
                            };
                            weight * score
                        })
                        .sum(),
                    FusionStrategy::RankFusion => {
                        // score = Σ 1 / (rank + 60) — rank is 0-based
                        contributions
                            .iter()
                            .map(|(_, _, rank, _)| 1.0 / (*rank as f64 + 60.0))
                            .sum()
                    }
                };

                // Collect unique contributing modalities (preserve insertion order)
                let mut seen_modalities: Vec<Modality> = Vec::new();
                for (modality, _, _, _) in &contributions {
                    if !seen_modalities.contains(modality) {
                        seen_modalities.push(*modality);
                    }
                }

                // Use the CID from the first contribution (all contributions for the
                // same result_id should share the same CID).
                let cid = contributions[0].3.clone();

                FusedResult {
                    result_id,
                    fused_score,
                    contributing_modalities: seen_modalities,
                    cid,
                }
            })
            .collect();

        // --- 4. Sort descending by fused_score, truncate to top-k ---
        fused.sort_by(|a, b| {
            b.fused_score
                .partial_cmp(&a.fused_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        fused.truncate(query.k);

        fused
    }

    /// Return a reference to the accumulated coordinator statistics.
    pub fn stats(&self) -> &CoordinatorStats {
        &self.stats
    }
}

impl Default for MultiModalSearchCoordinator {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: build a single ModalityResult.
    fn mr(result_id: u64, modality: Modality, score: f64, cid: &str) -> ModalityResult {
        ModalityResult {
            result_id,
            modality,
            score,
            cid: cid.to_string(),
        }
    }

    // Helper: build a SearchQuery.
    fn query(modalities: Vec<Modality>, k: usize) -> SearchQuery {
        SearchQuery {
            query_id: 1,
            modalities,
            k,
        }
    }

    // -----------------------------------------------------------------------
    // 1. ScoreSum fuses two modalities
    // -----------------------------------------------------------------------
    #[test]
    fn test_score_sum_fuses_two_modalities() {
        let mut c = MultiModalSearchCoordinator::new();
        let q = query(vec![Modality::Text, Modality::Image], 10);

        let mut pm: HashMap<Modality, Vec<ModalityResult>> = HashMap::new();
        pm.insert(Modality::Text, vec![mr(1, Modality::Text, 0.8, "cid1")]);
        pm.insert(Modality::Image, vec![mr(1, Modality::Image, 0.5, "cid1")]);

        let results = c.fuse_results(&q, pm, FusionStrategy::ScoreSum);

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].result_id, 1);
        let expected = 0.8 + 0.5;
        assert!((results[0].fused_score - expected).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // 2. ScoreMax picks max across modalities
    // -----------------------------------------------------------------------
    #[test]
    fn test_score_max_picks_max() {
        let mut c = MultiModalSearchCoordinator::new();
        let q = query(vec![Modality::Text, Modality::Audio], 10);

        let mut pm: HashMap<Modality, Vec<ModalityResult>> = HashMap::new();
        pm.insert(Modality::Text, vec![mr(7, Modality::Text, 0.3, "cid7")]);
        pm.insert(Modality::Audio, vec![mr(7, Modality::Audio, 0.9, "cid7")]);

        let results = c.fuse_results(&q, pm, FusionStrategy::ScoreMax);

        assert_eq!(results.len(), 1);
        assert!((results[0].fused_score - 0.9).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // 3. ScoreMax: single modality returns that score unchanged
    // -----------------------------------------------------------------------
    #[test]
    fn test_score_max_single_modality() {
        let mut c = MultiModalSearchCoordinator::new();
        let q = query(vec![Modality::Code], 5);

        let mut pm: HashMap<Modality, Vec<ModalityResult>> = HashMap::new();
        pm.insert(
            Modality::Code,
            vec![
                mr(10, Modality::Code, 0.6, "cidA"),
                mr(20, Modality::Code, 0.2, "cidB"),
            ],
        );

        let results = c.fuse_results(&q, pm, FusionStrategy::ScoreMax);
        assert_eq!(results[0].result_id, 10);
        assert!((results[0].fused_score - 0.6).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // 4. WeightedSum applies per-modality weights
    // -----------------------------------------------------------------------
    #[test]
    fn test_weighted_sum_applies_weights() {
        let mut c = MultiModalSearchCoordinator::new();
        let q = query(vec![Modality::Text, Modality::Image], 10);

        let strategy = FusionStrategy::WeightedSum {
            text_w: 2.0,
            image_w: 0.5,
            audio_w: 1.0,
            code_w: 1.0,
            struct_w: 1.0,
        };

        let mut pm: HashMap<Modality, Vec<ModalityResult>> = HashMap::new();
        pm.insert(Modality::Text, vec![mr(1, Modality::Text, 1.0, "cid1")]);
        pm.insert(Modality::Image, vec![mr(1, Modality::Image, 1.0, "cid1")]);

        let results = c.fuse_results(&q, pm, strategy);

        assert_eq!(results.len(), 1);
        // expected = 2.0*1.0 + 0.5*1.0 = 2.5
        assert!((results[0].fused_score - 2.5).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // 5. WeightedSum: different weights produce correct ordering
    // -----------------------------------------------------------------------
    #[test]
    fn test_weighted_sum_ordering() {
        let mut c = MultiModalSearchCoordinator::new();
        let q = query(vec![Modality::Text, Modality::Audio], 10);

        let strategy = FusionStrategy::WeightedSum {
            text_w: 1.0,
            image_w: 1.0,
            audio_w: 10.0,
            code_w: 1.0,
            struct_w: 1.0,
        };

        let mut pm: HashMap<Modality, Vec<ModalityResult>> = HashMap::new();
        // result 1: high text score, zero audio
        // result 2: low text score, high audio score -> should win after weighting
        pm.insert(
            Modality::Text,
            vec![
                mr(1, Modality::Text, 0.9, "cid1"),
                mr(2, Modality::Text, 0.1, "cid2"),
            ],
        );
        pm.insert(
            Modality::Audio,
            vec![
                mr(2, Modality::Audio, 0.9, "cid2"),
                mr(1, Modality::Audio, 0.0, "cid1"),
            ],
        );

        let results = c.fuse_results(&q, pm, strategy);

        // result 2: 1.0*0.1 + 10.0*0.9 = 9.1
        // result 1: 1.0*0.9 + 10.0*0.0 = 0.9
        assert_eq!(results[0].result_id, 2);
    }

    // -----------------------------------------------------------------------
    // 6. RankFusion: reciprocal rank computation
    // -----------------------------------------------------------------------
    #[test]
    fn test_rank_fusion_reciprocal_rank() {
        let mut c = MultiModalSearchCoordinator::new();
        let q = query(vec![Modality::Text], 10);

        // Two text results at ranks 0 and 1.
        let mut pm: HashMap<Modality, Vec<ModalityResult>> = HashMap::new();
        pm.insert(
            Modality::Text,
            vec![
                mr(1, Modality::Text, 0.9, "cid1"), // rank 0 -> 1/60
                mr(2, Modality::Text, 0.4, "cid2"), // rank 1 -> 1/61
            ],
        );

        let results = c.fuse_results(&q, pm, FusionStrategy::RankFusion);

        // result_id 1 has rank 0 -> score 1/60 ≈ 0.01667
        // result_id 2 has rank 1 -> score 1/61 ≈ 0.01639
        assert_eq!(results[0].result_id, 1);
        assert!((results[0].fused_score - 1.0 / 60.0).abs() < 1e-10);
        assert!((results[1].fused_score - 1.0 / 61.0).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // 7. RankFusion across two modalities accumulates ranks
    // -----------------------------------------------------------------------
    #[test]
    fn test_rank_fusion_two_modalities() {
        let mut c = MultiModalSearchCoordinator::new();
        let q = query(vec![Modality::Text, Modality::Image], 10);

        let mut pm: HashMap<Modality, Vec<ModalityResult>> = HashMap::new();
        // result_id 42 is rank 0 in text and rank 0 in image
        pm.insert(Modality::Text, vec![mr(42, Modality::Text, 0.5, "cidX")]);
        pm.insert(Modality::Image, vec![mr(42, Modality::Image, 0.5, "cidX")]);

        let results = c.fuse_results(&q, pm, FusionStrategy::RankFusion);

        assert_eq!(results.len(), 1);
        // score = 1/60 + 1/60 = 2/60
        let expected = 2.0 / 60.0;
        assert!((results[0].fused_score - expected).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // 8. Deduplication: same result_id from two modalities -> one FusedResult
    // -----------------------------------------------------------------------
    #[test]
    fn test_deduplication_merges_same_result_id() {
        let mut c = MultiModalSearchCoordinator::new();
        let q = query(vec![Modality::Text, Modality::Image], 10);

        let mut pm: HashMap<Modality, Vec<ModalityResult>> = HashMap::new();
        pm.insert(Modality::Text, vec![mr(99, Modality::Text, 0.7, "cidZ")]);
        pm.insert(Modality::Image, vec![mr(99, Modality::Image, 0.3, "cidZ")]);

        let results = c.fuse_results(&q, pm, FusionStrategy::ScoreSum);

        assert_eq!(results.len(), 1, "same result_id must be deduplicated");
        assert_eq!(results[0].result_id, 99);
    }

    // -----------------------------------------------------------------------
    // 9. Top-k truncation
    // -----------------------------------------------------------------------
    #[test]
    fn test_top_k_truncation() {
        let mut c = MultiModalSearchCoordinator::new();
        let q = query(vec![Modality::Text], 2);

        let mut pm: HashMap<Modality, Vec<ModalityResult>> = HashMap::new();
        pm.insert(
            Modality::Text,
            vec![
                mr(1, Modality::Text, 0.9, "c1"),
                mr(2, Modality::Text, 0.8, "c2"),
                mr(3, Modality::Text, 0.7, "c3"),
                mr(4, Modality::Text, 0.6, "c4"),
            ],
        );

        let results = c.fuse_results(&q, pm, FusionStrategy::ScoreSum);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].result_id, 1);
        assert_eq!(results[1].result_id, 2);
    }

    // -----------------------------------------------------------------------
    // 10. contributing_modalities contains all contributing modalities
    // -----------------------------------------------------------------------
    #[test]
    fn test_contributing_modalities_list() {
        let mut c = MultiModalSearchCoordinator::new();
        let q = query(vec![Modality::Text, Modality::Image, Modality::Audio], 10);

        let mut pm: HashMap<Modality, Vec<ModalityResult>> = HashMap::new();
        pm.insert(Modality::Text, vec![mr(5, Modality::Text, 0.5, "c5")]);
        pm.insert(Modality::Image, vec![mr(5, Modality::Image, 0.4, "c5")]);
        pm.insert(Modality::Audio, vec![mr(5, Modality::Audio, 0.3, "c5")]);

        let results = c.fuse_results(&q, pm, FusionStrategy::ScoreSum);

        assert_eq!(results.len(), 1);
        let contribs = &results[0].contributing_modalities;
        assert_eq!(contribs.len(), 3);
        assert!(contribs.contains(&Modality::Text));
        assert!(contribs.contains(&Modality::Image));
        assert!(contribs.contains(&Modality::Audio));
    }

    // -----------------------------------------------------------------------
    // 11. contributing_modalities: only one modality when result appears once
    // -----------------------------------------------------------------------
    #[test]
    fn test_contributing_modalities_single() {
        let mut c = MultiModalSearchCoordinator::new();
        let q = query(vec![Modality::Code], 10);

        let mut pm: HashMap<Modality, Vec<ModalityResult>> = HashMap::new();
        pm.insert(Modality::Code, vec![mr(3, Modality::Code, 0.7, "c3")]);

        let results = c.fuse_results(&q, pm, FusionStrategy::ScoreSum);

        assert_eq!(results[0].contributing_modalities.len(), 1);
        assert_eq!(results[0].contributing_modalities[0], Modality::Code);
    }

    // -----------------------------------------------------------------------
    // 12. Empty per_modality_results returns empty Vec
    // -----------------------------------------------------------------------
    #[test]
    fn test_empty_per_modality_returns_empty() {
        let mut c = MultiModalSearchCoordinator::new();
        let q = query(vec![Modality::Text], 10);

        let pm: HashMap<Modality, Vec<ModalityResult>> = HashMap::new();
        let results = c.fuse_results(&q, pm, FusionStrategy::ScoreSum);

        assert!(results.is_empty());
    }

    // -----------------------------------------------------------------------
    // 13. Single modality passthrough (ScoreSum = original scores, correct order)
    // -----------------------------------------------------------------------
    #[test]
    fn test_single_modality_passthrough() {
        let mut c = MultiModalSearchCoordinator::new();
        let q = query(vec![Modality::Image], 10);

        let mut pm: HashMap<Modality, Vec<ModalityResult>> = HashMap::new();
        pm.insert(
            Modality::Image,
            vec![
                mr(10, Modality::Image, 0.2, "a"),
                mr(20, Modality::Image, 0.8, "b"),
                mr(30, Modality::Image, 0.5, "c"),
            ],
        );

        let results = c.fuse_results(&q, pm, FusionStrategy::ScoreSum);

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].result_id, 20);
        assert_eq!(results[1].result_id, 30);
        assert_eq!(results[2].result_id, 10);
    }

    // -----------------------------------------------------------------------
    // 14. stats.total_queries increments on each call
    // -----------------------------------------------------------------------
    #[test]
    fn test_stats_total_queries_increments() {
        let mut c = MultiModalSearchCoordinator::new();
        assert_eq!(c.stats().total_queries, 0);

        let q = query(vec![Modality::Text], 5);
        c.fuse_results(&q, HashMap::new(), FusionStrategy::ScoreSum);
        assert_eq!(c.stats().total_queries, 1);
        c.fuse_results(&q, HashMap::new(), FusionStrategy::ScoreSum);
        assert_eq!(c.stats().total_queries, 2);
    }

    // -----------------------------------------------------------------------
    // 15. stats.modality_counts tracks queried modalities
    // -----------------------------------------------------------------------
    #[test]
    fn test_stats_modality_counts() {
        let mut c = MultiModalSearchCoordinator::new();

        let q1 = query(vec![Modality::Text, Modality::Image], 5);
        c.fuse_results(&q1, HashMap::new(), FusionStrategy::ScoreSum);

        let q2 = query(vec![Modality::Text], 5);
        c.fuse_results(&q2, HashMap::new(), FusionStrategy::ScoreSum);

        let counts = &c.stats().modality_counts;
        assert_eq!(*counts.get(&Modality::Text).unwrap_or(&0), 2);
        assert_eq!(*counts.get(&Modality::Image).unwrap_or(&0), 1);
        assert_eq!(*counts.get(&Modality::Audio).unwrap_or(&0), 0);
    }

    // -----------------------------------------------------------------------
    // 16. most_used_modality returns correct modality
    // -----------------------------------------------------------------------
    #[test]
    fn test_most_used_modality() {
        let mut c = MultiModalSearchCoordinator::new();

        // Audio appears 3 times, Text twice, Code once.
        for _ in 0..3 {
            let q = query(vec![Modality::Audio], 1);
            c.fuse_results(&q, HashMap::new(), FusionStrategy::ScoreSum);
        }
        for _ in 0..2 {
            let q = query(vec![Modality::Text], 1);
            c.fuse_results(&q, HashMap::new(), FusionStrategy::ScoreSum);
        }
        let q = query(vec![Modality::Code], 1);
        c.fuse_results(&q, HashMap::new(), FusionStrategy::ScoreSum);

        assert_eq!(c.stats().most_used_modality(), Some(Modality::Audio));
    }

    // -----------------------------------------------------------------------
    // 17. most_used_modality returns None when no queries processed
    // -----------------------------------------------------------------------
    #[test]
    fn test_most_used_modality_none_when_empty() {
        let c = MultiModalSearchCoordinator::new();
        assert_eq!(c.stats().most_used_modality(), None);
    }

    // -----------------------------------------------------------------------
    // 18. CID is preserved in fused results
    // -----------------------------------------------------------------------
    #[test]
    fn test_cid_preserved_in_fused_result() {
        let mut c = MultiModalSearchCoordinator::new();
        let q = query(vec![Modality::Structured], 5);

        let mut pm: HashMap<Modality, Vec<ModalityResult>> = HashMap::new();
        pm.insert(
            Modality::Structured,
            vec![mr(11, Modality::Structured, 0.5, "bafybeiabc123")],
        );

        let results = c.fuse_results(&q, pm, FusionStrategy::ScoreSum);
        assert_eq!(results[0].cid, "bafybeiabc123");
    }

    // -----------------------------------------------------------------------
    // 19. Sorting is stable for descending fused_score order
    // -----------------------------------------------------------------------
    #[test]
    fn test_results_sorted_descending() {
        let mut c = MultiModalSearchCoordinator::new();
        let q = query(vec![Modality::Text], 10);

        let scores = [0.3, 0.9, 0.1, 0.7, 0.5];
        let results_in: Vec<ModalityResult> = scores
            .iter()
            .enumerate()
            .map(|(i, &s)| mr(i as u64, Modality::Text, s, "c"))
            .collect();

        let mut pm: HashMap<Modality, Vec<ModalityResult>> = HashMap::new();
        pm.insert(Modality::Text, results_in);

        let results = c.fuse_results(&q, pm, FusionStrategy::ScoreSum);

        assert_eq!(results.len(), 5);
        for i in 1..results.len() {
            assert!(
                results[i - 1].fused_score >= results[i].fused_score,
                "results not sorted descending at index {}",
                i
            );
        }
    }
}
