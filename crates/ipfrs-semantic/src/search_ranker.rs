//! Vector Search Re-Ranker
//!
//! Re-ranks raw k-NN candidate results using a multi-signal scoring model:
//! vector similarity, recency, tag overlap, and peer reliability.

/// A single ranking signal with its weight and parameters.
#[derive(Debug, Clone)]
pub enum RankingSignal {
    /// Cosine similarity score in \[0,1\].
    VectorSimilarity { weight: f64 },
    /// Exponential recency decay: exp(-age_secs / decay_secs).
    RecencyBoost { weight: f64, decay_secs: u64 },
    /// Fraction of query_tags present in the result's tags.
    TagOverlap {
        weight: f64,
        query_tags: Vec<String>,
    },
    /// Peer reliability score in \[0,1\].
    PeerReliability { weight: f64 },
}

impl RankingSignal {
    /// Returns the weight of the signal.
    pub fn weight(&self) -> f64 {
        match self {
            RankingSignal::VectorSimilarity { weight } => *weight,
            RankingSignal::RecencyBoost { weight, .. } => *weight,
            RankingSignal::TagOverlap { weight, .. } => *weight,
            RankingSignal::PeerReliability { weight } => *weight,
        }
    }

    /// Returns the canonical name of the signal for explainability output.
    pub fn name(&self) -> &'static str {
        match self {
            RankingSignal::VectorSimilarity { .. } => "similarity",
            RankingSignal::RecencyBoost { .. } => "recency",
            RankingSignal::TagOverlap { .. } => "tag_overlap",
            RankingSignal::PeerReliability { .. } => "peer_reliability",
        }
    }
}

/// Raw k-NN candidate returned by the vector index.
#[derive(Debug, Clone)]
pub struct RawCandidate {
    /// Numeric identifier for the candidate.
    pub id: u64,
    /// Content identifier (CID) string.
    pub cid: String,
    /// Raw cosine similarity score in \[0,1\].
    pub similarity_score: f32,
    /// Unix timestamp (seconds) at which the content was created.
    pub created_at_secs: u64,
    /// Tags associated with the content.
    pub tags: Vec<String>,
    /// Peer reliability score in [0.0, 1.0].
    pub peer_reliability: f64,
    /// Arbitrary metadata string.
    pub metadata: String,
}

/// A candidate after re-ranking, carrying per-signal scores for explainability.
#[derive(Debug, Clone)]
pub struct RankedResult {
    /// The original candidate.
    pub candidate: RawCandidate,
    /// Weighted final score in \[0,1\].
    pub final_score: f64,
    /// Per-signal (name, weighted_score) pairs.
    pub signal_scores: Vec<(String, f64)>,
}

/// Configuration for the `VectorSearchRanker`.
#[derive(Debug, Clone)]
pub struct RankerConfig {
    /// Ordered list of ranking signals to apply.
    pub signals: Vec<RankingSignal>,
    /// Current Unix timestamp in seconds — injected for testability.
    pub now_secs: u64,
}

impl RankerConfig {
    /// Sum of all signal weights.
    pub fn total_weight(&self) -> f64 {
        self.signals.iter().map(|s| s.weight()).sum()
    }
}

/// Re-ranks raw k-NN candidates using a configurable multi-signal scoring model.
pub struct VectorSearchRanker {
    /// Configuration including signals and the reference timestamp.
    pub config: RankerConfig,
}

impl VectorSearchRanker {
    /// Creates a new `VectorSearchRanker` with the given configuration.
    pub fn new(config: RankerConfig) -> Self {
        Self { config }
    }

    /// Scores a single candidate against all configured signals.
    ///
    /// Returns a `RankedResult` with the computed `final_score` and per-signal
    /// breakdown for explainability.
    pub fn score_candidate(&self, candidate: &RawCandidate) -> RankedResult {
        let total_weight = self.config.total_weight();
        let mut weighted_sum = 0.0_f64;
        let mut signal_scores: Vec<(String, f64)> = Vec::with_capacity(self.config.signals.len());

        for signal in &self.config.signals {
            let (raw_score, weight) = match signal {
                RankingSignal::VectorSimilarity { weight } => {
                    let raw = candidate.similarity_score as f64;
                    (raw, *weight)
                }
                RankingSignal::RecencyBoost { weight, decay_secs } => {
                    let age = self
                        .config
                        .now_secs
                        .saturating_sub(candidate.created_at_secs);
                    let decay = if *decay_secs == 0 {
                        // Avoid divide-by-zero: treat zero decay_secs as no decay (score = 1)
                        1.0_f64
                    } else {
                        (-(age as f64) / (*decay_secs as f64)).exp()
                    };
                    (decay, *weight)
                }
                RankingSignal::TagOverlap { weight, query_tags } => {
                    let raw = if query_tags.is_empty() {
                        0.0_f64
                    } else {
                        let matches = query_tags
                            .iter()
                            .filter(|qt| candidate.tags.contains(qt))
                            .count();
                        matches as f64 / query_tags.len() as f64
                    };
                    (raw, *weight)
                }
                RankingSignal::PeerReliability { weight } => (candidate.peer_reliability, *weight),
            };

            let weighted = weight * raw_score;
            weighted_sum += weighted;
            signal_scores.push((signal.name().to_owned(), weighted));
        }

        let final_score = if total_weight == 0.0 {
            0.0
        } else {
            weighted_sum / total_weight
        };

        RankedResult {
            candidate: candidate.clone(),
            final_score,
            signal_scores,
        }
    }

    /// Scores all candidates and returns them sorted by `final_score` descending.
    pub fn rank(&self, candidates: &[RawCandidate]) -> Vec<RankedResult> {
        let mut results: Vec<RankedResult> =
            candidates.iter().map(|c| self.score_candidate(c)).collect();
        results.sort_by(|a, b| {
            b.final_score
                .partial_cmp(&a.final_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results
    }

    /// Returns at most `k` top-ranked candidates.
    pub fn rank_top_k(&self, candidates: &[RawCandidate], k: usize) -> Vec<RankedResult> {
        let mut ranked = self.rank(candidates);
        ranked.truncate(k);
        ranked
    }

    /// Returns a human-readable explanation of a `RankedResult`.
    ///
    /// Format: `"id=X score=Y.YY [signal=A, ...]"`
    pub fn explain(&self, result: &RankedResult) -> String {
        let signals_str: Vec<String> = result
            .signal_scores
            .iter()
            .map(|(name, score)| format!("{}={:.4}", name, score))
            .collect();
        format!(
            "id={} score={:.4} [{}]",
            result.candidate.id,
            result.final_score,
            signals_str.join(", ")
        )
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_candidate(
        id: u64,
        similarity: f32,
        created_at: u64,
        tags: Vec<&str>,
        peer_reliability: f64,
    ) -> RawCandidate {
        RawCandidate {
            id,
            cid: format!("cid-{}", id),
            similarity_score: similarity,
            created_at_secs: created_at,
            tags: tags.into_iter().map(str::to_owned).collect(),
            peer_reliability,
            metadata: String::new(),
        }
    }

    // ── 1. new() stores config ───────────────────────────────────────────────
    #[test]
    fn test_new_stores_config() {
        let config = RankerConfig {
            signals: vec![RankingSignal::VectorSimilarity { weight: 1.0 }],
            now_secs: 1000,
        };
        let ranker = VectorSearchRanker::new(config.clone());
        assert_eq!(ranker.config.now_secs, 1000);
        assert_eq!(ranker.config.signals.len(), 1);
    }

    // ── 2. similarity only → final_score = similarity_score ─────────────────
    #[test]
    fn test_score_candidate_similarity_only() {
        let config = RankerConfig {
            signals: vec![RankingSignal::VectorSimilarity { weight: 1.0 }],
            now_secs: 0,
        };
        let ranker = VectorSearchRanker::new(config);
        let candidate = make_candidate(1, 0.75, 0, vec![], 0.0);
        let result = ranker.score_candidate(&candidate);
        let diff = (result.final_score - 0.75).abs();
        assert!(diff < 1e-9, "expected 0.75, got {}", result.final_score);
    }

    // ── 3. recency at age=0 → exp(0) = 1.0 ──────────────────────────────────
    #[test]
    fn test_score_candidate_recency_age_zero() {
        let now = 5000_u64;
        let config = RankerConfig {
            signals: vec![RankingSignal::RecencyBoost {
                weight: 1.0,
                decay_secs: 3600,
            }],
            now_secs: now,
        };
        let ranker = VectorSearchRanker::new(config);
        // created_at == now → age = 0 → exp(0) = 1.0
        let candidate = make_candidate(1, 0.5, now, vec![], 0.0);
        let result = ranker.score_candidate(&candidate);
        let diff = (result.final_score - 1.0).abs();
        assert!(diff < 1e-9, "expected 1.0, got {}", result.final_score);
    }

    // ── 4. recency with age > 0 → decayed correctly ──────────────────────────
    #[test]
    fn test_score_candidate_recency_decayed() {
        let decay_secs = 3600_u64;
        let now = 7200_u64;
        let created_at = 0_u64;
        // age = 7200, raw = exp(-7200/3600) = exp(-2)
        let expected_raw = (-2.0_f64).exp();
        let config = RankerConfig {
            signals: vec![RankingSignal::RecencyBoost {
                weight: 1.0,
                decay_secs,
            }],
            now_secs: now,
        };
        let ranker = VectorSearchRanker::new(config);
        let candidate = make_candidate(1, 0.5, created_at, vec![], 0.0);
        let result = ranker.score_candidate(&candidate);
        let diff = (result.final_score - expected_raw).abs();
        assert!(
            diff < 1e-9,
            "expected {}, got {}",
            expected_raw,
            result.final_score
        );
    }

    // ── 5. tag_overlap no query_tags → 0.0 ───────────────────────────────────
    #[test]
    fn test_score_candidate_tag_overlap_empty_query() {
        let config = RankerConfig {
            signals: vec![RankingSignal::TagOverlap {
                weight: 1.0,
                query_tags: vec![],
            }],
            now_secs: 0,
        };
        let ranker = VectorSearchRanker::new(config);
        let candidate = make_candidate(1, 0.5, 0, vec!["rust", "ipfs"], 0.0);
        let result = ranker.score_candidate(&candidate);
        assert_eq!(result.final_score, 0.0);
    }

    // ── 6. tag_overlap all match → 1.0 ───────────────────────────────────────
    #[test]
    fn test_score_candidate_tag_overlap_full_match() {
        let config = RankerConfig {
            signals: vec![RankingSignal::TagOverlap {
                weight: 1.0,
                query_tags: vec!["rust".to_owned(), "ipfs".to_owned()],
            }],
            now_secs: 0,
        };
        let ranker = VectorSearchRanker::new(config);
        let candidate = make_candidate(1, 0.5, 0, vec!["rust", "ipfs"], 0.0);
        let result = ranker.score_candidate(&candidate);
        let diff = (result.final_score - 1.0).abs();
        assert!(diff < 1e-9, "expected 1.0, got {}", result.final_score);
    }

    // ── 7. tag_overlap partial match → correct fraction ───────────────────────
    #[test]
    fn test_score_candidate_tag_overlap_partial_match() {
        // query_tags = ["rust", "ipfs", "p2p"], candidate has ["rust", "p2p"] → 2/3
        let config = RankerConfig {
            signals: vec![RankingSignal::TagOverlap {
                weight: 1.0,
                query_tags: vec!["rust".to_owned(), "ipfs".to_owned(), "p2p".to_owned()],
            }],
            now_secs: 0,
        };
        let ranker = VectorSearchRanker::new(config);
        let candidate = make_candidate(1, 0.5, 0, vec!["rust", "p2p"], 0.0);
        let result = ranker.score_candidate(&candidate);
        let expected = 2.0 / 3.0;
        let diff = (result.final_score - expected).abs();
        assert!(
            diff < 1e-9,
            "expected {}, got {}",
            expected,
            result.final_score
        );
    }

    // ── 8. peer_reliability signal ────────────────────────────────────────────
    #[test]
    fn test_score_candidate_peer_reliability() {
        let config = RankerConfig {
            signals: vec![RankingSignal::PeerReliability { weight: 1.0 }],
            now_secs: 0,
        };
        let ranker = VectorSearchRanker::new(config);
        let candidate = make_candidate(1, 0.0, 0, vec![], 0.85);
        let result = ranker.score_candidate(&candidate);
        let diff = (result.final_score - 0.85).abs();
        assert!(diff < 1e-9, "expected 0.85, got {}", result.final_score);
    }

    // ── 9. multi-signal weighted average ──────────────────────────────────────
    #[test]
    fn test_score_candidate_multi_signal_weighted_average() {
        // similarity w=2 score=0.8 → weighted=1.6
        // peer_reliability w=1 score=0.5 → weighted=0.5
        // total_weight=3, final = (1.6+0.5)/3 = 0.7
        let config = RankerConfig {
            signals: vec![
                RankingSignal::VectorSimilarity { weight: 2.0 },
                RankingSignal::PeerReliability { weight: 1.0 },
            ],
            now_secs: 0,
        };
        let ranker = VectorSearchRanker::new(config);
        let candidate = make_candidate(1, 0.8, 0, vec![], 0.5);
        let result = ranker.score_candidate(&candidate);
        // similarity_score is stored as f32 (0.8f32), so casting to f64 introduces
        // a small representation error (~1e-8). Use f32 cast for the expected value.
        let sim_f64 = 0.8_f32 as f64;
        let expected = (2.0 * sim_f64 + 1.0 * 0.5) / 3.0;
        let diff = (result.final_score - expected).abs();
        assert!(
            diff < 1e-9,
            "expected {}, got {}",
            expected,
            result.final_score
        );
    }

    // ── 10. signal_scores length matches signals count ─────────────────────────
    #[test]
    fn test_score_candidate_signal_scores_length() {
        let config = RankerConfig {
            signals: vec![
                RankingSignal::VectorSimilarity { weight: 1.0 },
                RankingSignal::RecencyBoost {
                    weight: 1.0,
                    decay_secs: 3600,
                },
                RankingSignal::TagOverlap {
                    weight: 1.0,
                    query_tags: vec!["a".to_owned()],
                },
                RankingSignal::PeerReliability { weight: 1.0 },
            ],
            now_secs: 1000,
        };
        let ranker = VectorSearchRanker::new(config);
        let candidate = make_candidate(1, 0.5, 1000, vec!["a"], 0.9);
        let result = ranker.score_candidate(&candidate);
        assert_eq!(result.signal_scores.len(), 4);
    }

    // ── 11. rank() sorts descending by final_score ────────────────────────────
    #[test]
    fn test_rank_sorts_descending() {
        let config = RankerConfig {
            signals: vec![RankingSignal::VectorSimilarity { weight: 1.0 }],
            now_secs: 0,
        };
        let ranker = VectorSearchRanker::new(config);
        let candidates = vec![
            make_candidate(1, 0.3, 0, vec![], 0.0),
            make_candidate(2, 0.9, 0, vec![], 0.0),
            make_candidate(3, 0.6, 0, vec![], 0.0),
        ];
        let ranked = ranker.rank(&candidates);
        assert_eq!(ranked[0].candidate.id, 2);
        assert_eq!(ranked[1].candidate.id, 3);
        assert_eq!(ranked[2].candidate.id, 1);
    }

    // ── 12. rank() empty candidates returns empty ─────────────────────────────
    #[test]
    fn test_rank_empty_candidates() {
        let config = RankerConfig {
            signals: vec![RankingSignal::VectorSimilarity { weight: 1.0 }],
            now_secs: 0,
        };
        let ranker = VectorSearchRanker::new(config);
        let ranked = ranker.rank(&[]);
        assert!(ranked.is_empty());
    }

    // ── 13. rank_top_k() truncates correctly ──────────────────────────────────
    #[test]
    fn test_rank_top_k_truncates() {
        let config = RankerConfig {
            signals: vec![RankingSignal::VectorSimilarity { weight: 1.0 }],
            now_secs: 0,
        };
        let ranker = VectorSearchRanker::new(config);
        let candidates = vec![
            make_candidate(1, 0.3, 0, vec![], 0.0),
            make_candidate(2, 0.9, 0, vec![], 0.0),
            make_candidate(3, 0.6, 0, vec![], 0.0),
        ];
        let top2 = ranker.rank_top_k(&candidates, 2);
        assert_eq!(top2.len(), 2);
        assert_eq!(top2[0].candidate.id, 2);
        assert_eq!(top2[1].candidate.id, 3);
    }

    // ── 14. rank_top_k() k > len returns all ─────────────────────────────────
    #[test]
    fn test_rank_top_k_k_exceeds_len() {
        let config = RankerConfig {
            signals: vec![RankingSignal::VectorSimilarity { weight: 1.0 }],
            now_secs: 0,
        };
        let ranker = VectorSearchRanker::new(config);
        let candidates = vec![
            make_candidate(1, 0.5, 0, vec![], 0.0),
            make_candidate(2, 0.8, 0, vec![], 0.0),
        ];
        let top10 = ranker.rank_top_k(&candidates, 10);
        assert_eq!(top10.len(), 2);
    }

    // ── 15. explain() returns non-empty string containing the id ──────────────
    #[test]
    fn test_explain_contains_id() {
        let config = RankerConfig {
            signals: vec![RankingSignal::VectorSimilarity { weight: 1.0 }],
            now_secs: 0,
        };
        let ranker = VectorSearchRanker::new(config);
        let candidate = make_candidate(42, 0.7, 0, vec![], 0.0);
        let result = ranker.score_candidate(&candidate);
        let explanation = ranker.explain(&result);
        assert!(!explanation.is_empty());
        assert!(
            explanation.contains("id=42"),
            "explanation should contain 'id=42', got: {}",
            explanation
        );
    }

    // ── 16. total_weight() sum ────────────────────────────────────────────────
    #[test]
    fn test_total_weight_sum() {
        let config = RankerConfig {
            signals: vec![
                RankingSignal::VectorSimilarity { weight: 2.0 },
                RankingSignal::PeerReliability { weight: 3.0 },
                RankingSignal::RecencyBoost {
                    weight: 1.5,
                    decay_secs: 60,
                },
            ],
            now_secs: 0,
        };
        let diff = (config.total_weight() - 6.5).abs();
        assert!(diff < 1e-9, "expected 6.5, got {}", config.total_weight());
    }

    // ── 17. RankerConfig with zero total_weight → final_score = 0.0 ──────────
    #[test]
    fn test_zero_total_weight_gives_zero_final_score() {
        let config = RankerConfig {
            signals: vec![RankingSignal::VectorSimilarity { weight: 0.0 }],
            now_secs: 0,
        };
        let ranker = VectorSearchRanker::new(config);
        let candidate = make_candidate(1, 1.0, 0, vec![], 1.0);
        let result = ranker.score_candidate(&candidate);
        assert_eq!(result.final_score, 0.0);
    }
}

// ════════════════════════════════════════════════════════════════════════════
// SemanticSearchRanker — multi-signal re-ranking with stats accumulation
// ════════════════════════════════════════════════════════════════════════════

use std::collections::HashMap;

/// Individual ranking signal identifiers used as keys in the per-result score map.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RankSignal {
    /// Cosine similarity score in [0, 1].
    Similarity,
    /// Freshness signal: 0.5^(age / half_life), yielding [0, 1].
    Recency,
    /// Access-frequency signal normalised to [0, 1].
    Popularity,
    /// User-defined multiplicative boost (stored as the raw multiplier).
    UserBoost,
}

/// A single candidate produced by an upstream vector-search query.
#[derive(Debug, Clone)]
pub struct SearchCandidate {
    /// Numeric identifier for the vector-index entry.
    pub id: u64,
    /// Content-addressed identifier string.
    pub cid: String,
    /// Cosine similarity score from the vector index; must be in [0.0, 1.0].
    pub similarity: f32,
    /// Unix timestamp (seconds) at which this content was created.
    pub created_at_secs: u64,
    /// Number of times this content has been accessed.
    pub access_count: u64,
    /// Multiplicative user-defined relevance boost; default should be 1.0.
    pub user_boost: f32,
}

/// Configuration knobs for `SemanticSearchRanker`.
#[derive(Debug, Clone)]
pub struct SemanticRankerConfig {
    /// Weight applied to the similarity signal (default 0.6).
    pub similarity_weight: f32,
    /// Weight applied to the recency signal (default 0.2).
    pub recency_weight: f32,
    /// Weight applied to the popularity signal (default 0.2).
    pub popularity_weight: f32,
    /// Half-life in seconds for the recency decay (default 86 400 = 1 day).
    pub recency_half_life_secs: u64,
    /// Access-count ceiling used to normalise popularity (default 10 000).
    pub max_access_count: u64,
}

impl Default for SemanticRankerConfig {
    fn default() -> Self {
        Self {
            similarity_weight: 0.6,
            recency_weight: 0.2,
            popularity_weight: 0.2,
            recency_half_life_secs: 86_400,
            max_access_count: 10_000,
        }
    }
}

/// A candidate after multi-signal re-ranking, ready for presentation.
#[derive(Debug, Clone)]
pub struct SemanticRankedResult {
    /// The original candidate, cloned for ownership.
    pub candidate: SearchCandidate,
    /// Per-signal normalised scores (before applying `user_boost`).
    pub signal_scores: HashMap<RankSignal, f32>,
    /// Weighted combination multiplied by `user_boost`.
    pub final_score: f32,
    /// 1-based position in the sorted result list.
    pub rank: usize,
}

/// Aggregate statistics emitted by `SemanticSearchRanker`.
#[derive(Debug, Clone)]
pub struct RankerStats {
    /// Total number of candidates ranked across all `rank()` calls.
    pub total_ranked: u64,
    /// Mean `final_score` across all ranked candidates; 0.0 when nothing has
    /// been ranked yet.
    pub avg_final_score: f64,
    /// Mean number of candidates processed per `rank()` call; 0.0 when no
    /// call has been made.
    pub avg_candidates_per_call: f64,
}

/// Stateful re-ranker that combines similarity, recency, popularity, and a
/// user-defined boost into a single `final_score`, then accumulates
/// performance statistics across calls.
pub struct SemanticSearchRanker {
    /// Weighting and decay configuration.
    pub config: SemanticRankerConfig,
    /// Running total of candidates ranked (for stats).
    pub total_ranked: u64,
    /// Alias of `total_ranked` — kept separately so `avg_candidates_per_call`
    /// can be derived without an extra counter.
    pub total_candidates: u64,
    /// Cumulative sum of all `final_score` values (for average computation).
    pub total_score_sum: f64,
    /// Number of times `rank()` has been called (denominator for per-call avg).
    call_count: u64,
}

impl SemanticSearchRanker {
    /// Constructs a new ranker with the supplied configuration and zeroed stats.
    pub fn new(config: SemanticRankerConfig) -> Self {
        Self {
            config,
            total_ranked: 0,
            total_candidates: 0,
            total_score_sum: 0.0,
            call_count: 0,
        }
    }

    /// Re-ranks `candidates` using the configured signals and `now_secs` as
    /// the reference timestamp for recency computation.
    ///
    /// Returns the results sorted by `final_score` descending with 1-based
    /// `rank` fields assigned.  An empty input yields an empty output.
    pub fn rank(
        &mut self,
        candidates: Vec<SearchCandidate>,
        now_secs: u64,
    ) -> Vec<SemanticRankedResult> {
        let n = candidates.len();
        self.call_count = self.call_count.saturating_add(1);

        if n == 0 {
            return Vec::new();
        }

        let half_life = self.config.recency_half_life_secs;
        let max_ac = self.config.max_access_count;
        let sim_w = self.config.similarity_weight;
        let rec_w = self.config.recency_weight;
        let pop_w = self.config.popularity_weight;

        let mut results: Vec<SemanticRankedResult> = candidates
            .into_iter()
            .map(|c| {
                // ── recency ─────────────────────────────────────────────────
                let age_secs = now_secs.saturating_sub(c.created_at_secs);
                let recency_score = if half_life == 0 {
                    // Degenerate config: treat as fully fresh.
                    1.0_f32
                } else {
                    0.5_f32.powf(age_secs as f32 / half_life as f32)
                };

                // ── popularity ───────────────────────────────────────────────
                let popularity_score = if max_ac == 0 {
                    // Degenerate config: no normalisation possible.
                    0.0_f32
                } else {
                    c.access_count.min(max_ac) as f32 / max_ac as f32
                };

                // ── weighted combination ─────────────────────────────────────
                let weighted =
                    sim_w * c.similarity + rec_w * recency_score + pop_w * popularity_score;
                let final_score = weighted * c.user_boost;

                // ── signal map ───────────────────────────────────────────────
                let mut signal_scores: HashMap<RankSignal, f32> = HashMap::with_capacity(4);
                signal_scores.insert(RankSignal::Similarity, c.similarity);
                signal_scores.insert(RankSignal::Recency, recency_score);
                signal_scores.insert(RankSignal::Popularity, popularity_score);
                signal_scores.insert(RankSignal::UserBoost, c.user_boost);

                SemanticRankedResult {
                    candidate: c,
                    signal_scores,
                    final_score,
                    rank: 0, // filled in after sorting
                }
            })
            .collect();

        // ── sort descending by final_score ───────────────────────────────────
        results.sort_by(|a, b| {
            b.final_score
                .partial_cmp(&a.final_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // ── assign 1-based ranks ─────────────────────────────────────────────
        for (i, result) in results.iter_mut().enumerate() {
            result.rank = i + 1;
        }

        // ── accumulate stats ─────────────────────────────────────────────────
        let score_sum: f64 = results.iter().map(|r| r.final_score as f64).sum();
        self.total_ranked = self.total_ranked.saturating_add(n as u64);
        self.total_candidates = self.total_candidates.saturating_add(n as u64);
        self.total_score_sum += score_sum;

        results
    }

    /// Returns a snapshot of the accumulated statistics.
    pub fn stats(&self) -> RankerStats {
        let avg_final_score = if self.total_ranked == 0 {
            0.0
        } else {
            self.total_score_sum / self.total_ranked as f64
        };

        let avg_candidates_per_call = if self.call_count == 0 {
            0.0
        } else {
            self.total_candidates as f64 / self.call_count as f64
        };

        RankerStats {
            total_ranked: self.total_ranked,
            avg_final_score,
            avg_candidates_per_call,
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests for SemanticSearchRanker
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod semantic_ranker_tests {
    use super::*;

    // Helper: build a SearchCandidate with sensible defaults.
    fn make_sc(
        id: u64,
        similarity: f32,
        created_at_secs: u64,
        access_count: u64,
        user_boost: f32,
    ) -> SearchCandidate {
        SearchCandidate {
            id,
            cid: format!("cid-{}", id),
            similarity,
            created_at_secs,
            access_count,
            user_boost,
        }
    }

    fn default_ranker() -> SemanticSearchRanker {
        SemanticSearchRanker::new(SemanticRankerConfig::default())
    }

    // ── 1. new() starts with zero stats ─────────────────────────────────────
    #[test]
    fn test_new_zero_stats() {
        let ranker = default_ranker();
        assert_eq!(ranker.total_ranked, 0);
        assert_eq!(ranker.total_candidates, 0);
        assert_eq!(ranker.total_score_sum, 0.0);
        let stats = ranker.stats();
        assert_eq!(stats.total_ranked, 0);
        assert_eq!(stats.avg_final_score, 0.0);
        assert_eq!(stats.avg_candidates_per_call, 0.0);
    }

    // ── 2. rank empty returns empty ──────────────────────────────────────────
    #[test]
    fn test_rank_empty_returns_empty() {
        let mut ranker = default_ranker();
        let results = ranker.rank(vec![], 1_000_000);
        assert!(results.is_empty());
    }

    // ── 3. rank single candidate assigns rank 1 ──────────────────────────────
    #[test]
    fn test_rank_single_assigns_rank_one() {
        let mut ranker = default_ranker();
        let c = make_sc(7, 0.8, 0, 0, 1.0);
        let results = ranker.rank(vec![c], 1_000);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].rank, 1);
    }

    // ── 4. similarity signal stored correctly ────────────────────────────────
    #[test]
    fn test_similarity_signal_stored() {
        let mut ranker = default_ranker();
        let c = make_sc(1, 0.75, 0, 0, 1.0);
        let results = ranker.rank(vec![c], 86_400);
        let sim = results[0].signal_scores[&RankSignal::Similarity];
        let diff = (sim - 0.75).abs();
        assert!(diff < 1e-6, "expected 0.75, got {}", sim);
    }

    // ── 5. recency signal: age=0 → 1.0 ──────────────────────────────────────
    #[test]
    fn test_recency_age_zero_is_one() {
        let now = 500_000_u64;
        let mut ranker = default_ranker();
        let c = make_sc(1, 0.0, now, 0, 1.0); // created_at == now → age 0
        let results = ranker.rank(vec![c], now);
        let rec = results[0].signal_scores[&RankSignal::Recency];
        let diff = (rec - 1.0).abs();
        assert!(diff < 1e-6, "expected 1.0, got {}", rec);
    }

    // ── 6. recency signal: age=half_life → ~0.5 ─────────────────────────────
    #[test]
    fn test_recency_age_equals_half_life_gives_half() {
        let half_life = 86_400_u64;
        let now = half_life * 2;
        let created_at = half_life; // age = half_life
        let mut ranker = default_ranker();
        let c = make_sc(1, 0.0, created_at, 0, 1.0);
        let results = ranker.rank(vec![c], now);
        let rec = results[0].signal_scores[&RankSignal::Recency];
        let diff = (rec - 0.5).abs();
        assert!(diff < 1e-6, "expected ~0.5, got {}", rec);
    }

    // ── 7. recency signal: age >> half_life → approaches 0 ──────────────────
    #[test]
    fn test_recency_large_age_approaches_zero() {
        let half_life = 86_400_u64;
        let now = half_life * 100; // age = 100 * half_life → 0.5^100 ≈ 0
        let mut ranker = default_ranker();
        let c = make_sc(1, 0.0, 0, 0, 1.0);
        let results = ranker.rank(vec![c], now);
        let rec = results[0].signal_scores[&RankSignal::Recency];
        assert!(rec < 1e-6, "expected near 0, got {}", rec);
    }

    // ── 8. popularity signal: access_count=0 → 0.0 ──────────────────────────
    #[test]
    fn test_popularity_zero_access_count() {
        let mut ranker = default_ranker();
        let c = make_sc(1, 0.0, 0, 0, 1.0);
        let results = ranker.rank(vec![c], 0);
        let pop = results[0].signal_scores[&RankSignal::Popularity];
        assert_eq!(pop, 0.0);
    }

    // ── 9. popularity signal: access_count=max → 1.0 ────────────────────────
    #[test]
    fn test_popularity_max_access_count() {
        let max_ac = 10_000_u64;
        let mut ranker = default_ranker();
        let c = make_sc(1, 0.0, 0, max_ac, 1.0);
        let results = ranker.rank(vec![c], 0);
        let pop = results[0].signal_scores[&RankSignal::Popularity];
        let diff = (pop - 1.0).abs();
        assert!(diff < 1e-6, "expected 1.0, got {}", pop);
    }

    // ── 10. popularity capped at max_access_count ────────────────────────────
    #[test]
    fn test_popularity_capped_at_max() {
        let max_ac = 10_000_u64;
        let mut ranker = default_ranker();
        // access_count > max → still clamped to 1.0
        let c = make_sc(1, 0.0, 0, max_ac * 5, 1.0);
        let results = ranker.rank(vec![c], 0);
        let pop = results[0].signal_scores[&RankSignal::Popularity];
        let diff = (pop - 1.0).abs();
        assert!(diff < 1e-6, "expected 1.0 after capping, got {}", pop);
    }

    // ── 11. user_boost multiplies final_score ────────────────────────────────
    #[test]
    fn test_user_boost_multiplies_final_score() {
        let config = SemanticRankerConfig {
            similarity_weight: 1.0,
            recency_weight: 0.0,
            popularity_weight: 0.0,
            recency_half_life_secs: 86_400,
            max_access_count: 10_000,
        };
        let mut ranker = SemanticSearchRanker::new(config);
        let boost = 2.5_f32;
        let sim = 0.8_f32;
        let c = make_sc(1, sim, 0, 0, boost);
        let results = ranker.rank(vec![c], 0);
        let expected = sim * boost;
        let diff = (results[0].final_score - expected).abs();
        assert!(
            diff < 1e-5,
            "expected {}, got {}",
            expected,
            results[0].final_score
        );
    }

    // ── 12. user_boost=1.0 has no effect ────────────────────────────────────
    #[test]
    fn test_user_boost_one_no_effect() {
        let config = SemanticRankerConfig {
            similarity_weight: 1.0,
            recency_weight: 0.0,
            popularity_weight: 0.0,
            recency_half_life_secs: 86_400,
            max_access_count: 10_000,
        };
        let mut ranker = SemanticSearchRanker::new(config);
        let sim = 0.6_f32;
        let c = make_sc(1, sim, 0, 0, 1.0);
        let results = ranker.rank(vec![c], 0);
        let diff = (results[0].final_score - sim).abs();
        assert!(
            diff < 1e-6,
            "expected {}, got {}",
            sim,
            results[0].final_score
        );
    }

    // ── 13. higher similarity ranks first ───────────────────────────────────
    #[test]
    fn test_higher_similarity_ranks_first() {
        let config = SemanticRankerConfig {
            similarity_weight: 1.0,
            recency_weight: 0.0,
            popularity_weight: 0.0,
            recency_half_life_secs: 86_400,
            max_access_count: 10_000,
        };
        let mut ranker = SemanticSearchRanker::new(config);
        let c1 = make_sc(1, 0.3, 0, 0, 1.0);
        let c2 = make_sc(2, 0.9, 0, 0, 1.0);
        let c3 = make_sc(3, 0.6, 0, 0, 1.0);
        let results = ranker.rank(vec![c1, c2, c3], 0);
        assert_eq!(results[0].candidate.id, 2);
        assert_eq!(results[1].candidate.id, 3);
        assert_eq!(results[2].candidate.id, 1);
    }

    // ── 14. weights sum to correct combined score ────────────────────────────
    #[test]
    fn test_weights_produce_correct_combined_score() {
        // Use age=0 so recency=1.0, access_count=max_ac so popularity=1.0, user_boost=1.0
        let config = SemanticRankerConfig {
            similarity_weight: 0.6,
            recency_weight: 0.2,
            popularity_weight: 0.2,
            recency_half_life_secs: 86_400,
            max_access_count: 10_000,
        };
        let now = 0_u64;
        let mut ranker = SemanticSearchRanker::new(config);
        let sim = 0.5_f32;
        let c = make_sc(1, sim, now, 10_000, 1.0);
        let results = ranker.rank(vec![c], now);
        // expected = 0.6*0.5 + 0.2*1.0 + 0.2*1.0 = 0.3 + 0.2 + 0.2 = 0.7
        let expected = 0.7_f32;
        let diff = (results[0].final_score - expected).abs();
        assert!(
            diff < 1e-5,
            "expected {}, got {}",
            expected,
            results[0].final_score
        );
    }

    // ── 15. rank field is 1-based ────────────────────────────────────────────
    #[test]
    fn test_rank_field_is_one_based() {
        let mut ranker = default_ranker();
        let c = make_sc(1, 0.5, 0, 0, 1.0);
        let results = ranker.rank(vec![c], 0);
        assert_eq!(results[0].rank, 1);
    }

    // ── 16. rank field is sequential ────────────────────────────────────────
    #[test]
    fn test_rank_field_sequential() {
        let mut ranker = default_ranker();
        let candidates = vec![
            make_sc(1, 0.9, 0, 0, 1.0),
            make_sc(2, 0.7, 0, 0, 1.0),
            make_sc(3, 0.5, 0, 0, 1.0),
        ];
        let results = ranker.rank(candidates, 0);
        for (i, r) in results.iter().enumerate() {
            assert_eq!(r.rank, i + 1, "expected rank {} at position {}", i + 1, i);
        }
    }

    // ── 17. stats total_ranked accumulates across calls ──────────────────────
    #[test]
    fn test_stats_total_ranked_accumulates() {
        let mut ranker = default_ranker();
        ranker.rank(
            vec![make_sc(1, 0.5, 0, 0, 1.0), make_sc(2, 0.6, 0, 0, 1.0)],
            0,
        );
        ranker.rank(vec![make_sc(3, 0.7, 0, 0, 1.0)], 0);
        let stats = ranker.stats();
        assert_eq!(stats.total_ranked, 3);
    }

    // ── 18. stats avg_final_score computed ──────────────────────────────────
    #[test]
    fn test_stats_avg_final_score_computed() {
        // similarity_weight=1, others=0, user_boost=1 → final_score == similarity
        let config = SemanticRankerConfig {
            similarity_weight: 1.0,
            recency_weight: 0.0,
            popularity_weight: 0.0,
            recency_half_life_secs: 86_400,
            max_access_count: 10_000,
        };
        let mut ranker = SemanticSearchRanker::new(config);
        ranker.rank(
            vec![make_sc(1, 0.4, 0, 0, 1.0), make_sc(2, 0.8, 0, 0, 1.0)],
            0,
        );
        let stats = ranker.stats();
        // avg = (0.4 + 0.8) / 2 = 0.6
        let diff = (stats.avg_final_score - 0.6).abs();
        assert!(
            diff < 1e-5,
            "expected avg ~0.6, got {}",
            stats.avg_final_score
        );
    }

    // ── 19. stats avg_candidates_per_call ───────────────────────────────────
    #[test]
    fn test_stats_avg_candidates_per_call() {
        let mut ranker = default_ranker();
        // call 1: 3 candidates; call 2: 1 candidate → avg = (3+1)/2 = 2.0
        ranker.rank(
            vec![
                make_sc(1, 0.5, 0, 0, 1.0),
                make_sc(2, 0.6, 0, 0, 1.0),
                make_sc(3, 0.7, 0, 0, 1.0),
            ],
            0,
        );
        ranker.rank(vec![make_sc(4, 0.4, 0, 0, 1.0)], 0);
        let stats = ranker.stats();
        let diff = (stats.avg_candidates_per_call - 2.0).abs();
        assert!(
            diff < 1e-9,
            "expected 2.0, got {}",
            stats.avg_candidates_per_call
        );
    }

    // ── 20. multiple calls accumulate stats ──────────────────────────────────
    #[test]
    fn test_multiple_calls_accumulate_stats() {
        let config = SemanticRankerConfig {
            similarity_weight: 1.0,
            recency_weight: 0.0,
            popularity_weight: 0.0,
            recency_half_life_secs: 86_400,
            max_access_count: 10_000,
        };
        let mut ranker = SemanticSearchRanker::new(config);
        for _ in 0..5 {
            ranker.rank(vec![make_sc(1, 1.0, 0, 0, 1.0)], 0);
        }
        let stats = ranker.stats();
        assert_eq!(stats.total_ranked, 5);
        let diff = (stats.avg_final_score - 1.0).abs();
        assert!(
            diff < 1e-5,
            "expected avg 1.0, got {}",
            stats.avg_final_score
        );
    }

    // ── 21. signal_scores HashMap populated with all four keys ───────────────
    #[test]
    fn test_signal_scores_hashmap_populated() {
        let mut ranker = default_ranker();
        let c = make_sc(1, 0.5, 0, 500, 2.0);
        let results = ranker.rank(vec![c], 0);
        let scores = &results[0].signal_scores;
        assert!(
            scores.contains_key(&RankSignal::Similarity),
            "missing Similarity"
        );
        assert!(scores.contains_key(&RankSignal::Recency), "missing Recency");
        assert!(
            scores.contains_key(&RankSignal::Popularity),
            "missing Popularity"
        );
        assert!(
            scores.contains_key(&RankSignal::UserBoost),
            "missing UserBoost"
        );
        assert_eq!(scores.len(), 4);
    }

    // ── 22. empty rank() call does not corrupt stats ─────────────────────────
    #[test]
    fn test_empty_rank_does_not_corrupt_stats() {
        let mut ranker = default_ranker();
        ranker.rank(vec![make_sc(1, 0.9, 0, 0, 1.0)], 0);
        ranker.rank(vec![], 0); // empty — should not crash or corrupt
        let stats = ranker.stats();
        assert_eq!(stats.total_ranked, 1);
    }

    // ── 23. final_score clamping edge case: zero weights → zero weighted, user_boost applied ──
    #[test]
    fn test_zero_weights_and_user_boost() {
        let config = SemanticRankerConfig {
            similarity_weight: 0.0,
            recency_weight: 0.0,
            popularity_weight: 0.0,
            recency_half_life_secs: 86_400,
            max_access_count: 10_000,
        };
        let mut ranker = SemanticSearchRanker::new(config);
        let c = make_sc(1, 0.9, 0, 9999, 3.0);
        let results = ranker.rank(vec![c], 0);
        // weighted = 0*0.9 + 0*... = 0; final_score = 0 * 3.0 = 0
        assert_eq!(results[0].final_score, 0.0);
    }
}
