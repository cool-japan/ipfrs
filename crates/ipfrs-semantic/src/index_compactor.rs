//! # Index Compactor
//!
//! HNSW indexes accumulate deleted vectors over time that waste memory and degrade query
//! performance. This module identifies fragmented indexes and coordinates rebuilds based
//! on configurable compaction policies.
//!
//! ## Overview
//!
//! - [`CompactionPolicy`] — configures when compaction should be triggered
//! - [`IndexFragmentStats`] — captures fragmentation metrics for a single index
//! - [`IndexCompactor`] — analyzes indexes and produces [`CompactionPlan`]s
//! - [`CompactionPlan`] — describes the planned compaction and its expected impact
//! - [`CompactorStats`] — atomic counters tracking compactor activity

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// IndexFragmentStats
// ---------------------------------------------------------------------------

/// Statistics describing the fragmentation state of a single HNSW index.
#[derive(Debug, Clone, PartialEq)]
pub struct IndexFragmentStats {
    /// Total number of vector slots (live + deleted).
    pub total_vectors: usize,
    /// Number of slots marked as deleted but not yet reclaimed.
    pub deleted_vectors: usize,
    /// Estimated on-disk / in-memory footprint in bytes.
    pub index_bytes: u64,
}

impl IndexFragmentStats {
    /// Creates a new `IndexFragmentStats`.
    pub fn new(total_vectors: usize, deleted_vectors: usize, index_bytes: u64) -> Self {
        Self {
            total_vectors,
            deleted_vectors,
            index_bytes,
        }
    }

    /// Ratio of deleted vectors to total vectors.
    ///
    /// Returns `0.0` when `total_vectors == 0` to avoid division-by-zero.
    pub fn deleted_ratio(&self) -> f64 {
        if self.total_vectors == 0 {
            return 0.0;
        }
        self.deleted_vectors as f64 / self.total_vectors as f64
    }

    /// Number of live (non-deleted) vectors.
    pub fn live_vectors(&self) -> usize {
        self.total_vectors.saturating_sub(self.deleted_vectors)
    }
}

// ---------------------------------------------------------------------------
// CompactionReason
// ---------------------------------------------------------------------------

/// The primary reason a compaction was recommended.
#[derive(Debug, Clone, PartialEq)]
pub enum CompactionReason {
    /// The fraction of deleted vectors exceeded the policy threshold.
    HighDeletedRatio {
        /// Observed deleted-to-total ratio.
        ratio: f64,
    },
    /// The index size exceeded the policy's byte limit.
    SizeExceeded {
        /// Current measured size in bytes.
        current_bytes: u64,
        /// Configured size limit in bytes.
        limit_bytes: u64,
    },
    /// Compaction was requested on a fixed schedule regardless of metrics.
    Scheduled,
}

// ---------------------------------------------------------------------------
// CompactionPriority
// ---------------------------------------------------------------------------

/// How urgently a compaction should be scheduled.
///
/// Variants are ordered from lowest to highest priority so that `Critical > High > Normal > Low`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum CompactionPriority {
    /// Index health is acceptable; schedule compaction during a maintenance window.
    Low = 0,
    /// Standard compaction during off-peak hours.
    Normal = 1,
    /// Elevated fragmentation; schedule soon.
    High = 2,
    /// Severe fragmentation; compact immediately.
    Critical = 3,
}

impl PartialOrd for CompactionPriority {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for CompactionPriority {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (*self as u8).cmp(&(*other as u8))
    }
}

// ---------------------------------------------------------------------------
// CompactionPlan
// ---------------------------------------------------------------------------

/// A concrete recommendation to compact an index, including estimated impact.
#[derive(Debug, Clone, PartialEq)]
pub struct CompactionPlan {
    /// Why the compaction was triggered.
    pub reason: CompactionReason,
    /// Estimated number of live vectors after the rebuild.
    pub estimated_vectors_after: usize,
    /// Approximate bytes that will be freed by the compaction.
    pub estimated_bytes_saved: u64,
    /// How urgently the compaction should be performed.
    pub priority: CompactionPriority,
}

// ---------------------------------------------------------------------------
// CompactionPolicy
// ---------------------------------------------------------------------------

/// Configures the triggers that cause [`IndexCompactor`] to recommend a compaction.
#[derive(Debug, Clone)]
pub struct CompactionPolicy {
    /// Recommend compaction when `deleted / total > max_deleted_ratio`.
    ///
    /// Default: `0.1` (10 %).
    pub max_deleted_ratio: f64,
    /// Recommend compaction when the index exceeds this many bytes.
    ///
    /// Default: 512 MiB (`512 * 1024 * 1024`).
    pub max_index_bytes: u64,
    /// Skip compaction analysis for indexes with fewer than this many total vectors.
    ///
    /// Default: `1000`.
    pub min_vectors_for_compaction: usize,
}

impl Default for CompactionPolicy {
    fn default() -> Self {
        Self {
            max_deleted_ratio: 0.1,
            max_index_bytes: 512 * 1024 * 1024,
            min_vectors_for_compaction: 1000,
        }
    }
}

impl CompactionPolicy {
    /// Returns `true` when `stats` indicates that compaction is needed according to this policy.
    ///
    /// Compaction is *not* triggered when `total_vectors < min_vectors_for_compaction`, even if
    /// other thresholds are exceeded (tiny indexes are cheap to traverse as-is).
    pub fn should_compact(&self, stats: &IndexFragmentStats) -> bool {
        if stats.total_vectors < self.min_vectors_for_compaction {
            return false;
        }
        if stats.deleted_ratio() > self.max_deleted_ratio {
            return true;
        }
        if stats.index_bytes > self.max_index_bytes {
            return true;
        }
        false
    }
}

// ---------------------------------------------------------------------------
// CompactorStats (atomic counters)
// ---------------------------------------------------------------------------

/// Internal mutable state of [`CompactorStats`] — not exposed directly.
struct CompactorStatsInner {
    total_analyzed: AtomicU64,
    total_compaction_needed: AtomicU64,
    total_skipped: AtomicU64,
}

impl CompactorStatsInner {
    fn new() -> Self {
        Self {
            total_analyzed: AtomicU64::new(0),
            total_compaction_needed: AtomicU64::new(0),
            total_skipped: AtomicU64::new(0),
        }
    }
}

/// A snapshot of [`CompactorStats`] counters taken at a point in time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompactorStatsSnapshot {
    /// Total number of [`IndexFragmentStats`] passed to [`IndexCompactor::analyze`].
    pub total_analyzed: u64,
    /// Analyses that resulted in a [`CompactionPlan`] being returned.
    pub total_compaction_needed: u64,
    /// Analyses where no compaction was necessary.
    pub total_skipped: u64,
}

/// Thread-safe counters tracking compactor activity.
///
/// Internally uses [`AtomicU64`] values so no locking is required.
#[derive(Clone)]
pub struct CompactorStats {
    inner: Arc<CompactorStatsInner>,
}

impl CompactorStats {
    fn new() -> Self {
        Self {
            inner: Arc::new(CompactorStatsInner::new()),
        }
    }

    fn record_analyzed(&self) {
        self.inner.total_analyzed.fetch_add(1, Ordering::Relaxed);
    }

    fn record_compaction_needed(&self) {
        self.inner
            .total_compaction_needed
            .fetch_add(1, Ordering::Relaxed);
    }

    fn record_skipped(&self) {
        self.inner.total_skipped.fetch_add(1, Ordering::Relaxed);
    }

    /// Returns a consistent point-in-time snapshot of all counters.
    pub fn snapshot(&self) -> CompactorStatsSnapshot {
        CompactorStatsSnapshot {
            total_analyzed: self.inner.total_analyzed.load(Ordering::Relaxed),
            total_compaction_needed: self.inner.total_compaction_needed.load(Ordering::Relaxed),
            total_skipped: self.inner.total_skipped.load(Ordering::Relaxed),
        }
    }
}

impl std::fmt::Debug for CompactorStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let snap = self.snapshot();
        f.debug_struct("CompactorStats")
            .field("total_analyzed", &snap.total_analyzed)
            .field("total_compaction_needed", &snap.total_compaction_needed)
            .field("total_skipped", &snap.total_skipped)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// IndexCompactor
// ---------------------------------------------------------------------------

/// Analyses HNSW index fragmentation and recommends compaction plans.
///
/// `IndexCompactor` is the central entry point for compaction decisions.  Given
/// an [`IndexFragmentStats`] it applies the configured [`CompactionPolicy`] and
/// returns an [`Option<CompactionPlan>`]: `Some` when compaction is warranted,
/// `None` when the index is healthy enough to leave alone.
///
/// All activity is recorded in [`CompactorStats`] so operators can observe how
/// frequently indexes require maintenance.
#[derive(Debug, Clone)]
pub struct IndexCompactor {
    /// Configurable thresholds that drive compaction decisions.
    pub policy: CompactionPolicy,
    /// Atomic activity counters.
    pub stats: CompactorStats,
}

impl IndexCompactor {
    /// Creates a new `IndexCompactor` with the supplied policy.
    pub fn new(policy: CompactionPolicy) -> Self {
        Self {
            policy,
            stats: CompactorStats::new(),
        }
    }

    /// Analyses `fragment_stats` and returns a [`CompactionPlan`] when compaction is needed.
    ///
    /// Returns `None` when the index is below all policy thresholds.
    pub fn analyze(&self, fragment_stats: &IndexFragmentStats) -> Option<CompactionPlan> {
        self.stats.record_analyzed();

        if !self.policy.should_compact(fragment_stats) {
            self.stats.record_skipped();
            return None;
        }

        self.stats.record_compaction_needed();

        let reason = self.primary_reason(fragment_stats);
        let priority = self.estimate_priority(fragment_stats);
        let estimated_bytes_saved = self.estimate_bytes_saved(fragment_stats);
        let estimated_vectors_after = fragment_stats.live_vectors();

        Some(CompactionPlan {
            reason,
            estimated_vectors_after,
            estimated_bytes_saved,
            priority,
        })
    }

    /// Determines the primary compaction reason for the given stats.
    ///
    /// Deleted-ratio threshold takes precedence over size threshold when both are exceeded.
    fn primary_reason(&self, stats: &IndexFragmentStats) -> CompactionReason {
        let ratio = stats.deleted_ratio();
        if ratio > self.policy.max_deleted_ratio {
            return CompactionReason::HighDeletedRatio { ratio };
        }
        if stats.index_bytes > self.policy.max_index_bytes {
            return CompactionReason::SizeExceeded {
                current_bytes: stats.index_bytes,
                limit_bytes: self.policy.max_index_bytes,
            };
        }
        // Fallback — only reached when `should_compact` returned `true` via a scheduled path
        // that bypasses both checks.  Defined here for exhaustiveness.
        CompactionReason::Scheduled
    }

    /// Estimates the urgency of the required compaction.
    ///
    /// | Condition | Priority |
    /// |-----------|----------|
    /// | `deleted_ratio > 0.5` **or** `index_bytes > 2 × limit` | `Critical` |
    /// | `deleted_ratio > 0.25` | `High` |
    /// | compaction needed | `Normal` |
    /// | compaction *not* needed | `Low` |
    pub fn estimate_priority(&self, stats: &IndexFragmentStats) -> CompactionPriority {
        let ratio = stats.deleted_ratio();
        let double_limit = self.policy.max_index_bytes.saturating_mul(2);

        if ratio > 0.5 || stats.index_bytes > double_limit {
            return CompactionPriority::Critical;
        }
        if ratio > 0.25 {
            return CompactionPriority::High;
        }
        if self.policy.should_compact(stats) {
            return CompactionPriority::Normal;
        }
        CompactionPriority::Low
    }

    /// Estimates how many bytes will be reclaimed by a compaction.
    ///
    /// Uses the approximation `deleted_ratio × index_bytes`.
    pub fn estimate_bytes_saved(&self, stats: &IndexFragmentStats) -> u64 {
        let ratio = stats.deleted_ratio();
        (ratio * stats.index_bytes as f64) as u64
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    fn default_compactor() -> IndexCompactor {
        IndexCompactor::new(CompactionPolicy::default())
    }

    /// Builds stats with `total = 10_000` vectors unless overridden.
    fn stats(total: usize, deleted: usize, bytes: u64) -> IndexFragmentStats {
        IndexFragmentStats::new(total, deleted, bytes)
    }

    const MB: u64 = 1024 * 1024;

    // ------------------------------------------------------------------
    // 1. should_compact — high deleted ratio trigger
    // ------------------------------------------------------------------

    #[test]
    fn test_should_compact_high_deleted_ratio() {
        let policy = CompactionPolicy::default();
        // 15 % deleted exceeds the 10 % default threshold
        let s = stats(10_000, 1_500, 100 * MB);
        assert!(policy.should_compact(&s));
    }

    // ------------------------------------------------------------------
    // 2. should_compact — size exceeded trigger
    // ------------------------------------------------------------------

    #[test]
    fn test_should_compact_size_exceeded() {
        let policy = CompactionPolicy::default();
        // 0 deleted vectors, but index is over the 512 MiB limit
        let s = stats(10_000, 0, 600 * MB);
        assert!(policy.should_compact(&s));
    }

    // ------------------------------------------------------------------
    // 3. should_compact — scheduled (custom policy overriding thresholds)
    //    We test both triggers simultaneously to verify that either alone suffices.
    // ------------------------------------------------------------------

    #[test]
    fn test_should_compact_both_triggers() {
        let policy = CompactionPolicy::default();
        let s = stats(10_000, 2_000, 700 * MB);
        assert!(policy.should_compact(&s));
    }

    // ------------------------------------------------------------------
    // 4. should_compact returns false when all metrics are below thresholds
    // ------------------------------------------------------------------

    #[test]
    fn test_should_compact_returns_false_when_healthy() {
        let policy = CompactionPolicy::default();
        // 5 % deleted (below 10 %), 100 MiB (below 512 MiB)
        let s = stats(10_000, 500, 100 * MB);
        assert!(!policy.should_compact(&s));
    }

    // ------------------------------------------------------------------
    // 5. min_vectors guard — tiny index must not be compacted even if ratio is high
    // ------------------------------------------------------------------

    #[test]
    fn test_should_compact_min_vectors_guard() {
        let policy = CompactionPolicy::default();
        // Only 500 vectors (below the 1000 minimum), 50 % deleted
        let s = stats(500, 250, 10 * MB);
        assert!(!policy.should_compact(&s));
    }

    // ------------------------------------------------------------------
    // 6. min_vectors guard — exactly at the boundary (1000) should compact
    // ------------------------------------------------------------------

    #[test]
    fn test_should_compact_min_vectors_boundary() {
        let policy = CompactionPolicy::default();
        // Exactly 1000 vectors, 20 % deleted (above 10 % threshold)
        let s = stats(1_000, 200, 50 * MB);
        assert!(policy.should_compact(&s));
    }

    // ------------------------------------------------------------------
    // 7. deleted_ratio calculation
    // ------------------------------------------------------------------

    #[test]
    fn test_deleted_ratio_calculation() {
        let s = stats(10_000, 2_500, 100 * MB);
        let ratio = s.deleted_ratio();
        assert!(
            (ratio - 0.25).abs() < f64::EPSILON,
            "expected 0.25, got {ratio}"
        );
    }

    // ------------------------------------------------------------------
    // 8. deleted_ratio returns 0.0 for empty index
    // ------------------------------------------------------------------

    #[test]
    fn test_deleted_ratio_empty_index() {
        let s = stats(0, 0, 0);
        assert_eq!(s.deleted_ratio(), 0.0);
    }

    // ------------------------------------------------------------------
    // 9. live_vectors helper
    // ------------------------------------------------------------------

    #[test]
    fn test_live_vectors() {
        let s = stats(10_000, 3_000, 100 * MB);
        assert_eq!(s.live_vectors(), 7_000);
    }

    // ------------------------------------------------------------------
    // 10. analyze returns Some when compaction is needed
    // ------------------------------------------------------------------

    #[test]
    fn test_analyze_returns_some_when_needed() {
        let compactor = default_compactor();
        let s = stats(10_000, 2_000, 100 * MB); // 20 % deleted → above 10 % threshold
        assert!(compactor.analyze(&s).is_some());
    }

    // ------------------------------------------------------------------
    // 11. analyze returns None when index is clean
    // ------------------------------------------------------------------

    #[test]
    fn test_analyze_returns_none_when_clean() {
        let compactor = default_compactor();
        let s = stats(10_000, 500, 100 * MB); // 5 % deleted, 100 MiB — healthy
        assert!(compactor.analyze(&s).is_none());
    }

    // ------------------------------------------------------------------
    // 12. estimate_priority — Low when below all thresholds
    // ------------------------------------------------------------------

    #[test]
    fn test_estimate_priority_low() {
        let compactor = default_compactor();
        let s = stats(10_000, 100, 50 * MB); // 1 % deleted, 50 MiB
        assert_eq!(compactor.estimate_priority(&s), CompactionPriority::Low);
    }

    // ------------------------------------------------------------------
    // 13. estimate_priority — Normal when slightly over deleted threshold
    // ------------------------------------------------------------------

    #[test]
    fn test_estimate_priority_normal() {
        let compactor = default_compactor();
        // 15 % deleted: above 10 % (normal) but below 25 % (high)
        let s = stats(10_000, 1_500, 100 * MB);
        assert_eq!(compactor.estimate_priority(&s), CompactionPriority::Normal);
    }

    // ------------------------------------------------------------------
    // 14. estimate_priority — High when deleted_ratio > 0.25
    // ------------------------------------------------------------------

    #[test]
    fn test_estimate_priority_high() {
        let compactor = default_compactor();
        // 30 % deleted
        let s = stats(10_000, 3_000, 100 * MB);
        assert_eq!(compactor.estimate_priority(&s), CompactionPriority::High);
    }

    // ------------------------------------------------------------------
    // 15. estimate_priority — Critical when deleted_ratio > 0.5
    // ------------------------------------------------------------------

    #[test]
    fn test_estimate_priority_critical_high_ratio() {
        let compactor = default_compactor();
        // 60 % deleted
        let s = stats(10_000, 6_000, 100 * MB);
        assert_eq!(
            compactor.estimate_priority(&s),
            CompactionPriority::Critical
        );
    }

    // ------------------------------------------------------------------
    // 16. estimate_priority — Critical when bytes > 2× limit
    // ------------------------------------------------------------------

    #[test]
    fn test_estimate_priority_critical_size() {
        let compactor = default_compactor();
        // 0 deleted, but > 1024 MiB (2 × 512 MiB default limit)
        let s = stats(10_000, 0, 1_100 * MB);
        assert_eq!(
            compactor.estimate_priority(&s),
            CompactionPriority::Critical
        );
    }

    // ------------------------------------------------------------------
    // 17. CompactionPriority ordering (Critical > High > Normal > Low)
    // ------------------------------------------------------------------

    #[test]
    fn test_compaction_priority_ordering() {
        assert!(CompactionPriority::Critical > CompactionPriority::High);
        assert!(CompactionPriority::High > CompactionPriority::Normal);
        assert!(CompactionPriority::Normal > CompactionPriority::Low);
        assert!(CompactionPriority::Critical > CompactionPriority::Low);
    }

    // ------------------------------------------------------------------
    // 18. Stats increment on each call to analyze
    // ------------------------------------------------------------------

    #[test]
    fn test_stats_increment_on_analyze() {
        let compactor = default_compactor();

        // First call — below thresholds → skipped
        let clean = stats(10_000, 500, 100 * MB);
        compactor.analyze(&clean);

        // Second call — above threshold → compaction needed
        let dirty = stats(10_000, 2_000, 100 * MB);
        compactor.analyze(&dirty);

        let snap = compactor.stats.snapshot();
        assert_eq!(snap.total_analyzed, 2);
        assert_eq!(snap.total_compaction_needed, 1);
        assert_eq!(snap.total_skipped, 1);
    }

    // ------------------------------------------------------------------
    // 19. estimate_bytes_saved formula: deleted_ratio × index_bytes
    // ------------------------------------------------------------------

    #[test]
    fn test_estimate_bytes_saved_formula() {
        let compactor = default_compactor();
        // 25 % deleted, 400 MiB index → expect ~100 MiB saved
        let s = stats(10_000, 2_500, 400 * MB);
        let saved = compactor.estimate_bytes_saved(&s);
        let expected = (0.25_f64 * (400 * MB) as f64) as u64;
        assert_eq!(saved, expected);
    }

    // ------------------------------------------------------------------
    // 20. CompactionPlan fields are correct
    // ------------------------------------------------------------------

    #[test]
    fn test_compaction_plan_fields() {
        let compactor = default_compactor();
        let s = stats(10_000, 2_000, 200 * MB); // 20 % deleted

        let plan = compactor.analyze(&s).expect("should produce a plan");
        assert_eq!(plan.estimated_vectors_after, 8_000);
        assert!(plan.estimated_bytes_saved > 0);
        assert_eq!(plan.priority, CompactionPriority::Normal);
        match plan.reason {
            CompactionReason::HighDeletedRatio { ratio } => {
                assert!((ratio - 0.2).abs() < f64::EPSILON);
            }
            other => panic!("unexpected reason: {other:?}"),
        }
    }

    // ------------------------------------------------------------------
    // 21. Custom policy — lower thresholds
    // ------------------------------------------------------------------

    #[test]
    fn test_custom_policy_lower_thresholds() {
        let policy = CompactionPolicy {
            max_deleted_ratio: 0.05,
            max_index_bytes: 50 * MB,
            min_vectors_for_compaction: 100,
        };
        let compactor = IndexCompactor::new(policy);

        // 8 % deleted (> 5 % custom threshold) — should trigger
        let s = stats(1_000, 80, 10 * MB);
        assert!(compactor.analyze(&s).is_some());
    }

    // ------------------------------------------------------------------
    // 22. analyze on tiny index below min_vectors returns None
    // ------------------------------------------------------------------

    #[test]
    fn test_analyze_tiny_index_skipped() {
        let compactor = default_compactor();
        // 900 vectors with 50 % deleted — but below the 1000-vector minimum
        let s = stats(900, 450, 50 * MB);
        assert!(compactor.analyze(&s).is_none());
    }

    // ------------------------------------------------------------------
    // 23. Reason is SizeExceeded when only size is breached
    // ------------------------------------------------------------------

    #[test]
    fn test_reason_size_exceeded() {
        let compactor = default_compactor();
        // 0 % deleted, 600 MiB (> 512 MiB limit)
        let s = stats(10_000, 0, 600 * MB);
        let plan = compactor.analyze(&s).expect("should produce a plan");
        match plan.reason {
            CompactionReason::SizeExceeded {
                current_bytes,
                limit_bytes,
            } => {
                assert_eq!(current_bytes, 600 * MB);
                assert_eq!(limit_bytes, 512 * MB);
            }
            other => panic!("unexpected reason: {other:?}"),
        }
    }

    // ------------------------------------------------------------------
    // 24. Multiple analyses accumulate stats correctly
    // ------------------------------------------------------------------

    #[test]
    fn test_stats_multiple_analyses() {
        let compactor = default_compactor();

        let clean = stats(10_000, 100, 50 * MB);
        let dirty = stats(10_000, 5_000, 200 * MB);

        for _ in 0..5 {
            compactor.analyze(&clean);
        }
        for _ in 0..3 {
            compactor.analyze(&dirty);
        }

        let snap = compactor.stats.snapshot();
        assert_eq!(snap.total_analyzed, 8);
        assert_eq!(snap.total_skipped, 5);
        assert_eq!(snap.total_compaction_needed, 3);
    }
}
