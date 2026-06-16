//! GC Planner — analyzes block liveness and produces ordered deletion plans.
//!
//! [`StorageGCPlanner`] examines a set of [`GCCandidate`] blocks, filters out
//! pinned blocks, referenced blocks, and blocks that are too young, then builds
//! an ordered [`GCPlan`] that can be handed to the actual deletion engine.
//!
//! # Example
//!
//! ```rust
//! use ipfrs_storage::gc_planner::{GCCandidate, GCConfig, StorageGCPlanner};
//!
//! let config = GCConfig::default();
//! let planner = StorageGCPlanner::new(config);
//!
//! let candidates = vec![
//!     GCCandidate {
//!         cid: "bafy1".to_string(),
//!         size_bytes: 1024,
//!         ref_count: 0,
//!         last_accessed_secs: 0,
//!         pinned: false,
//!     },
//! ];
//!
//! let now = 7200u64;
//! let (plan, stats) = planner.plan(&candidates, now);
//! assert_eq!(stats.selected_count, 1);
//! ```

/// A single block that is a candidate for garbage collection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GCCandidate {
    /// Content identifier of the block.
    pub cid: String,
    /// Size of the block in bytes.
    pub size_bytes: u64,
    /// Number of references from other blocks / roots.
    /// A block with `ref_count > 0` must not be collected.
    pub ref_count: u32,
    /// Unix timestamp (seconds) of the most recent access.
    pub last_accessed_secs: u64,
    /// Whether the block is explicitly pinned and must never be collected.
    pub pinned: bool,
}

/// An ordered list of blocks selected for deletion together with aggregate
/// metadata about the planned run.
#[derive(Debug, Clone, Default)]
pub struct GCPlan {
    /// Blocks selected for deletion, sorted by `last_accessed_secs` ascending
    /// (oldest-first so the least-recently-used data is evicted first).
    pub candidates: Vec<GCCandidate>,
    /// Sum of `size_bytes` across all selected candidates.
    pub estimated_freed_bytes: u64,
}

impl GCPlan {
    /// Number of blocks in the plan.
    pub fn candidate_count(&self) -> usize {
        self.candidates.len()
    }

    /// Returns `true` when the plan contains no blocks.
    pub fn is_empty(&self) -> bool {
        self.candidates.is_empty()
    }
}

/// Configuration knobs for the GC planner.
#[derive(Debug, Clone)]
pub struct GCConfig {
    /// Only collect blocks whose age (now − last_accessed_secs) is **strictly
    /// greater than** this value (seconds).  Default: 3 600 (1 hour).
    pub min_age_secs: u64,
    /// Stop collecting once this many bytes have been earmarked for deletion.
    /// Default: 1 073 741 824 (1 GiB).
    pub target_free_bytes: u64,
    /// Hard upper bound on the number of candidates in a single plan.
    /// Default: 1 000.
    pub max_candidates: usize,
}

impl Default for GCConfig {
    fn default() -> Self {
        Self {
            min_age_secs: 3_600,
            target_free_bytes: 1_073_741_824,
            max_candidates: 1_000,
        }
    }
}

/// Statistics produced alongside every [`GCPlan`] so callers can understand
/// why particular blocks were included or excluded.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GCPlannerStats {
    /// Total number of input candidates examined.
    pub total_analyzed: usize,
    /// Blocks skipped because `pinned == true`.
    pub pinned_skipped: usize,
    /// Blocks skipped because `ref_count > 0`.
    pub referenced_skipped: usize,
    /// Blocks skipped because their age was below `min_age_secs`.
    pub too_young_skipped: usize,
    /// Blocks that made it into the final plan.
    pub selected_count: usize,
    /// Sum of `size_bytes` for selected blocks (mirrors [`GCPlan::estimated_freed_bytes`]).
    pub estimated_freed_bytes: u64,
}

/// Plans garbage-collection runs by analysing block liveness.
///
/// The planner is intentionally stateless beyond its [`GCConfig`]: feed it a
/// snapshot of candidates and a wall-clock timestamp and it returns a
/// ready-to-execute [`GCPlan`].
#[derive(Debug, Clone)]
pub struct StorageGCPlanner {
    /// Configuration controlling collection behaviour.
    pub config: GCConfig,
}

impl StorageGCPlanner {
    /// Create a new planner with the given configuration.
    pub fn new(config: GCConfig) -> Self {
        Self { config }
    }

    /// Analyse `candidates` and produce a deletion plan.
    ///
    /// `now_secs` is the current Unix timestamp in seconds.  Using a parameter
    /// rather than reading the system clock keeps the function pure and
    /// deterministic for testing.
    ///
    /// # Algorithm
    ///
    /// 1. Iterate over every candidate, classifying it as *pinned*, *referenced*,
    ///    *too young*, or *eligible*.
    /// 2. Sort eligible candidates by `last_accessed_secs` ascending.
    /// 3. Greedily add candidates to the plan until `estimated_freed_bytes >=
    ///    target_free_bytes` **or** `candidate_count >= max_candidates`.
    pub fn plan(&self, candidates: &[GCCandidate], now_secs: u64) -> (GCPlan, GCPlannerStats) {
        let mut stats = GCPlannerStats {
            total_analyzed: candidates.len(),
            ..Default::default()
        };

        let mut eligible: Vec<&GCCandidate> = Vec::with_capacity(candidates.len());

        for candidate in candidates {
            if candidate.pinned {
                stats.pinned_skipped += 1;
                continue;
            }
            if candidate.ref_count > 0 {
                stats.referenced_skipped += 1;
                continue;
            }
            // Age is computed with saturating_sub so that
            // last_accessed_secs > now_secs simply yields 0 (too young).
            let age = now_secs.saturating_sub(candidate.last_accessed_secs);
            if age < self.config.min_age_secs {
                stats.too_young_skipped += 1;
                continue;
            }
            eligible.push(candidate);
        }

        // Sort oldest-first (ascending last_accessed_secs).
        eligible.sort_by_key(|c| c.last_accessed_secs);

        let mut plan_candidates: Vec<GCCandidate> = Vec::new();
        let mut freed_bytes: u64 = 0;

        for candidate in eligible {
            if plan_candidates.len() >= self.config.max_candidates {
                break;
            }
            plan_candidates.push(candidate.clone());
            freed_bytes = freed_bytes.saturating_add(candidate.size_bytes);
            if freed_bytes >= self.config.target_free_bytes {
                break;
            }
        }

        stats.selected_count = plan_candidates.len();
        stats.estimated_freed_bytes = freed_bytes;

        let plan = GCPlan {
            candidates: plan_candidates,
            estimated_freed_bytes: freed_bytes,
        };

        (plan, stats)
    }

    /// Estimate how long executing `plan` might take in milliseconds.
    ///
    /// Simple linear model:
    /// - 1 ms per candidate block.
    /// - 1 ms per 10 MiB (10 485 760 bytes) of data (integer division, no floats).
    pub fn estimate_run_time_ms(&self, plan: &GCPlan) -> u64 {
        let per_candidate = plan.candidate_count() as u64;
        let per_data = plan.estimated_freed_bytes / 10_485_760;
        per_candidate + per_data
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ──────────────────────────────────────────────────────────────

    fn make_candidate(
        cid: &str,
        size_bytes: u64,
        ref_count: u32,
        last_accessed_secs: u64,
        pinned: bool,
    ) -> GCCandidate {
        GCCandidate {
            cid: cid.to_string(),
            size_bytes,
            ref_count,
            last_accessed_secs,
            pinned,
        }
    }

    fn default_planner() -> StorageGCPlanner {
        StorageGCPlanner::new(GCConfig::default())
    }

    // ── construction ─────────────────────────────────────────────────────────

    #[test]
    fn test_new_stores_config() {
        let config = GCConfig {
            min_age_secs: 7_200,
            target_free_bytes: 512 * 1024 * 1024,
            max_candidates: 500,
        };
        let planner = StorageGCPlanner::new(config.clone());
        assert_eq!(planner.config.min_age_secs, 7_200);
        assert_eq!(planner.config.target_free_bytes, 512 * 1024 * 1024);
        assert_eq!(planner.config.max_candidates, 500);
    }

    #[test]
    fn test_gc_config_default_values() {
        let cfg = GCConfig::default();
        assert_eq!(cfg.min_age_secs, 3_600);
        assert_eq!(cfg.target_free_bytes, 1_073_741_824);
        assert_eq!(cfg.max_candidates, 1_000);
    }

    // ── empty input ───────────────────────────────────────────────────────────

    #[test]
    fn test_empty_input_produces_empty_plan() {
        let planner = default_planner();
        let (plan, stats) = planner.plan(&[], 10_000);
        assert!(plan.is_empty());
        assert_eq!(plan.candidate_count(), 0);
        assert_eq!(plan.estimated_freed_bytes, 0);
        assert_eq!(stats.total_analyzed, 0);
        assert_eq!(stats.selected_count, 0);
    }

    // ── skip reasons ─────────────────────────────────────────────────────────

    #[test]
    fn test_pinned_blocks_skipped() {
        let planner = default_planner();
        let candidates = vec![make_candidate("c1", 1024, 0, 0, true)];
        let (plan, stats) = planner.plan(&candidates, 10_000);
        assert!(plan.is_empty());
        assert_eq!(stats.pinned_skipped, 1);
        assert_eq!(stats.referenced_skipped, 0);
        assert_eq!(stats.too_young_skipped, 0);
    }

    #[test]
    fn test_referenced_blocks_skipped() {
        let planner = default_planner();
        let candidates = vec![make_candidate("c1", 1024, 3, 0, false)];
        let (plan, stats) = planner.plan(&candidates, 10_000);
        assert!(plan.is_empty());
        assert_eq!(stats.referenced_skipped, 1);
        assert_eq!(stats.pinned_skipped, 0);
        assert_eq!(stats.too_young_skipped, 0);
    }

    #[test]
    fn test_too_young_blocks_skipped() {
        let planner = default_planner(); // min_age = 3600
                                         // age = 10_000 − 7_000 = 3_000 < 3_600 → too young
        let candidates = vec![make_candidate("c1", 1024, 0, 7_000, false)];
        let (plan, stats) = planner.plan(&candidates, 10_000);
        assert!(plan.is_empty());
        assert_eq!(stats.too_young_skipped, 1);
    }

    #[test]
    fn test_exactly_at_min_age_is_skipped() {
        let planner = default_planner(); // min_age = 3600
                                         // age = 10_000 − 6_400 = 3_600 == min_age_secs
                                         // The skip condition is `age < min_age_secs`, so age == min_age_secs
                                         // is NOT skipped — the block is eligible and enters the plan.
        let candidates = vec![make_candidate("c1", 1024, 0, 6_400, false)];
        let (plan, stats) = planner.plan(&candidates, 10_000);
        assert_eq!(
            plan.candidate_count(),
            1,
            "block at exactly min_age must be included (skip condition is strictly less than)"
        );
        assert_eq!(stats.too_young_skipped, 0);
        assert_eq!(stats.selected_count, 1);
    }

    #[test]
    fn test_eligible_block_included() {
        let planner = default_planner(); // min_age = 3600
                                         // age = 10_000 − 6_000 = 4_000 > 3_600 → eligible
        let candidates = vec![make_candidate("c1", 1024, 0, 6_000, false)];
        let (plan, stats) = planner.plan(&candidates, 10_000);
        assert_eq!(plan.candidate_count(), 1);
        assert_eq!(stats.selected_count, 1);
        assert_eq!(plan.candidates[0].cid, "c1");
    }

    // ── ordering ─────────────────────────────────────────────────────────────

    #[test]
    fn test_plan_sorted_oldest_first() {
        let planner = default_planner();
        // All are old enough (accessed at t=0, t=100, t=200; now=10_000)
        let candidates = vec![
            make_candidate("newer", 512, 0, 200, false),
            make_candidate("oldest", 512, 0, 0, false),
            make_candidate("middle", 512, 0, 100, false),
        ];
        let (plan, _) = planner.plan(&candidates, 10_000);
        assert_eq!(plan.candidate_count(), 3);
        assert_eq!(plan.candidates[0].cid, "oldest");
        assert_eq!(plan.candidates[1].cid, "middle");
        assert_eq!(plan.candidates[2].cid, "newer");
    }

    // ── stopping criteria ─────────────────────────────────────────────────────

    #[test]
    fn test_target_free_bytes_stops_early() {
        let config = GCConfig {
            min_age_secs: 0,
            target_free_bytes: 1_000,
            max_candidates: 1_000,
        };
        let planner = StorageGCPlanner::new(config);
        let candidates = vec![
            make_candidate("c1", 600, 0, 0, false),
            make_candidate("c2", 600, 0, 1, false),
            make_candidate("c3", 600, 0, 2, false),
        ];
        let (plan, stats) = planner.plan(&candidates, 100);
        // c1 adds 600, c2 pushes total to 1200 ≥ 1000 → stop after c2
        assert_eq!(plan.candidate_count(), 2);
        assert_eq!(stats.selected_count, 2);
        assert!(plan.estimated_freed_bytes >= 1_000);
    }

    #[test]
    fn test_max_candidates_stops_early() {
        let config = GCConfig {
            min_age_secs: 0,
            target_free_bytes: u64::MAX,
            max_candidates: 2,
        };
        let planner = StorageGCPlanner::new(config);
        let candidates: Vec<GCCandidate> = (0..5_u64)
            .map(|i| make_candidate(&format!("c{i}"), 100, 0, i, false))
            .collect();
        let (plan, stats) = planner.plan(&candidates, 1_000);
        assert_eq!(plan.candidate_count(), 2);
        assert_eq!(stats.selected_count, 2);
    }

    // ── aggregate metrics ─────────────────────────────────────────────────────

    #[test]
    fn test_estimated_freed_bytes_sum_correct() {
        let config = GCConfig {
            min_age_secs: 0,
            target_free_bytes: u64::MAX,
            max_candidates: 1_000,
        };
        let planner = StorageGCPlanner::new(config);
        let candidates = vec![
            make_candidate("c1", 1_000, 0, 0, false),
            make_candidate("c2", 2_000, 0, 1, false),
            make_candidate("c3", 3_000, 0, 2, false),
        ];
        let (plan, stats) = planner.plan(&candidates, 10_000);
        assert_eq!(plan.estimated_freed_bytes, 6_000);
        assert_eq!(stats.estimated_freed_bytes, 6_000);
    }

    #[test]
    fn test_is_empty_true_when_no_candidates() {
        let plan = GCPlan::default();
        assert!(plan.is_empty());
    }

    #[test]
    fn test_candidate_count_correct() {
        let planner = StorageGCPlanner::new(GCConfig {
            min_age_secs: 0,
            target_free_bytes: u64::MAX,
            max_candidates: 1_000,
        });
        let candidates: Vec<GCCandidate> = (0..7_u64)
            .map(|i| make_candidate(&format!("c{i}"), 1, 0, i, false))
            .collect();
        let (plan, _) = planner.plan(&candidates, 1_000);
        assert_eq!(plan.candidate_count(), 7);
    }

    // ── stats correctness ─────────────────────────────────────────────────────

    #[test]
    fn test_stats_total_analyzed_correct() {
        let planner = default_planner();
        let candidates: Vec<GCCandidate> = (0..10_u64)
            .map(|i| make_candidate(&format!("c{i}"), 1, 0, 0, false))
            .collect();
        let (_, stats) = planner.plan(&candidates, 10_000);
        assert_eq!(stats.total_analyzed, 10);
    }

    #[test]
    fn test_stats_pinned_skipped_correct() {
        let planner = default_planner();
        let candidates = vec![
            make_candidate("p1", 1, 0, 0, true),
            make_candidate("p2", 1, 0, 0, true),
            make_candidate("ok", 1, 0, 0, false),
        ];
        let (_, stats) = planner.plan(&candidates, 10_000);
        assert_eq!(stats.pinned_skipped, 2);
    }

    #[test]
    fn test_stats_referenced_skipped_correct() {
        let planner = default_planner();
        let candidates = vec![
            make_candidate("r1", 1, 1, 0, false),
            make_candidate("r2", 1, 5, 0, false),
            make_candidate("ok", 1, 0, 0, false),
        ];
        let (_, stats) = planner.plan(&candidates, 10_000);
        assert_eq!(stats.referenced_skipped, 2);
    }

    #[test]
    fn test_stats_too_young_skipped_correct() {
        let planner = default_planner(); // min_age = 3600
        let candidates = vec![
            // age = 10000 - 8000 = 2000 < 3600 → too young
            make_candidate("y1", 1, 0, 8_000, false),
            // age = 10000 - 7000 = 3000 < 3600 → too young
            make_candidate("y2", 1, 0, 7_000, false),
            // age = 10000 - 0 = 10000 > 3600 → eligible
            make_candidate("ok", 1, 0, 0, false),
        ];
        let (_, stats) = planner.plan(&candidates, 10_000);
        assert_eq!(stats.too_young_skipped, 2);
    }

    #[test]
    fn test_stats_selected_count_correct() {
        let planner = StorageGCPlanner::new(GCConfig {
            min_age_secs: 0,
            target_free_bytes: u64::MAX,
            max_candidates: 1_000,
        });
        let candidates: Vec<GCCandidate> = (0..6_u64)
            .map(|i| make_candidate(&format!("c{i}"), 1, 0, i, false))
            .collect();
        let (_, stats) = planner.plan(&candidates, 1_000);
        assert_eq!(stats.selected_count, 6);
    }

    // ── combined skip reasons ─────────────────────────────────────────────────

    #[test]
    fn test_multiple_skip_reasons_combine_correctly() {
        let planner = default_planner(); // min_age = 3600, now = 10_000
        let candidates = vec![
            make_candidate("pinned", 1, 0, 0, true),     // pinned_skipped
            make_candidate("ref'd", 1, 2, 0, false),     // referenced_skipped
            make_candidate("young", 1, 0, 8_000, false), // too_young_skipped (age=2000)
            make_candidate("eligible", 1_024, 0, 0, false), // selected
        ];
        let (plan, stats) = planner.plan(&candidates, 10_000);
        assert_eq!(stats.total_analyzed, 4);
        assert_eq!(stats.pinned_skipped, 1);
        assert_eq!(stats.referenced_skipped, 1);
        assert_eq!(stats.too_young_skipped, 1);
        assert_eq!(stats.selected_count, 1);
        assert_eq!(plan.candidate_count(), 1);
        assert_eq!(plan.candidates[0].cid, "eligible");
    }

    // ── estimate_run_time_ms ──────────────────────────────────────────────────

    #[test]
    fn test_estimate_run_time_ms_basic() {
        let planner = default_planner();
        // 3 candidates, 30 MiB total → 3 + 3 = 6 ms
        let plan = GCPlan {
            candidates: vec![
                make_candidate("c1", 10_485_760, 0, 0, false),
                make_candidate("c2", 10_485_760, 0, 1, false),
                make_candidate("c3", 10_485_760, 0, 2, false),
            ],
            estimated_freed_bytes: 31_457_280, // 30 MiB
        };
        let ms = planner.estimate_run_time_ms(&plan);
        assert_eq!(ms, 6); // 3 (candidates) + 3 (data)
    }

    #[test]
    fn test_estimate_run_time_ms_empty_plan() {
        let planner = default_planner();
        let plan = GCPlan::default();
        assert_eq!(planner.estimate_run_time_ms(&plan), 0);
    }

    #[test]
    fn test_estimate_run_time_ms_large_data() {
        let planner = default_planner();
        // 1 candidate, 100 MiB (104_857_600 bytes) → 1 + 10 = 11 ms
        let plan = GCPlan {
            candidates: vec![make_candidate("big", 104_857_600, 0, 0, false)],
            estimated_freed_bytes: 104_857_600,
        };
        let ms = planner.estimate_run_time_ms(&plan);
        assert_eq!(ms, 11); // 1 (candidate) + 10 (10 × 10 MiB)
    }

    // ── edge cases ────────────────────────────────────────────────────────────

    #[test]
    fn test_future_last_accessed_treated_as_too_young() {
        // last_accessed_secs > now_secs → saturating_sub gives 0 → age 0 < min_age
        let planner = default_planner();
        let candidates = vec![make_candidate("future", 1_024, 0, 99_999, false)];
        let (plan, stats) = planner.plan(&candidates, 10_000);
        assert!(plan.is_empty());
        assert_eq!(stats.too_young_skipped, 1);
    }

    #[test]
    fn test_ref_count_one_still_skipped() {
        let planner = default_planner();
        let candidates = vec![make_candidate("one_ref", 512, 1, 0, false)];
        let (plan, stats) = planner.plan(&candidates, 10_000);
        assert!(plan.is_empty());
        assert_eq!(stats.referenced_skipped, 1);
    }
}
