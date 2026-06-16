//! Compaction advisor for data-driven storage compaction recommendations.
//!
//! Analyzes storage state and provides recommendations for when and how to
//! compact, merge, or drop data, based on configurable thresholds.

use serde::{Deserialize, Serialize};

/// A recommended compaction or maintenance action.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompactionAction {
    /// Flush the WAL and run sled compaction.
    CompactSled,
    /// Merge the listed SSTable files together.
    MergeSSTables { table_ids: Vec<String> },
    /// Move cold data out of the named tier to free up space.
    EvictTier {
        tier_name: String,
        bytes_to_free: u64,
    },
    /// Garbage-collect unreferenced (orphan) blocks.
    DropOrphanBlocks { cid_count: usize },
    /// No action is needed at this time.
    NoActionNeeded,
}

/// A snapshot of current storage system metrics used to drive advisory logic.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StorageMetrics {
    /// Bytes currently used in the sled database.
    pub sled_bytes_used: u64,
    /// Bytes currently free in the sled database.
    pub sled_bytes_free: u64,
    /// Number of WAL operations that have not yet been flushed.
    pub wal_pending_ops: u64,
    /// Number of blocks with no live references (orphans).
    pub orphan_block_count: u64,
    /// Total bytes occupied by orphan blocks.
    pub orphan_block_bytes: u64,
    /// Bytes stored in the hot tier.
    pub hot_tier_bytes: u64,
    /// Bytes stored in the warm tier.
    pub warm_tier_bytes: u64,
    /// Bytes stored in the cold tier.
    pub cold_tier_bytes: u64,
    /// Number of SSTable files currently present.
    pub sstable_count: u64,
    /// Fragmentation ratio in the range 0.0–1.0 (higher = more fragmented).
    pub fragmentation_ratio: f64,
}

impl StorageMetrics {
    /// Returns the total bytes across all tiers plus sled.
    pub fn total_bytes(&self) -> u64 {
        self.sled_bytes_used
            .saturating_add(self.hot_tier_bytes)
            .saturating_add(self.warm_tier_bytes)
            .saturating_add(self.cold_tier_bytes)
    }

    /// Returns the usage ratio (used / (used + free)).
    ///
    /// Returns `0.0` when both used and free are zero.
    pub fn usage_ratio(&self) -> f64 {
        let used = self.sled_bytes_used;
        let free = self.sled_bytes_free;
        if used == 0 && free == 0 {
            return 0.0;
        }
        used as f64 / (used as f64 + free as f64)
    }
}

/// Configurable thresholds that drive the advisory logic.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdvisoryThresholds {
    /// Flush the WAL when pending operations exceed this value (default: 1 000).
    pub wal_flush_threshold: u64,
    /// Run orphan GC when orphan block count exceeds this value (default: 100).
    pub orphan_gc_threshold: u64,
    /// Compact when fragmentation ratio exceeds this value (default: 0.3).
    pub fragmentation_compact_threshold: f64,
    /// Merge SSTables when the file count exceeds this value (default: 10).
    pub sstable_merge_threshold: u64,
    /// Evict cold data when the usage ratio exceeds this value (default: 0.8).
    pub cold_tier_evict_ratio: f64,
}

impl Default for AdvisoryThresholds {
    fn default() -> Self {
        Self {
            wal_flush_threshold: 1_000,
            orphan_gc_threshold: 100,
            fragmentation_compact_threshold: 0.3,
            sstable_merge_threshold: 10,
            cold_tier_evict_ratio: 0.8,
        }
    }
}

/// The result of an advisory analysis: a set of recommended actions with
/// contextual metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompactionAdvice {
    /// Ordered list of recommended actions.
    pub actions: Vec<CompactionAction>,
    /// Urgency level: 0 = none, 1 = low, 2 = medium, 3 = high.
    pub urgency: u8,
    /// Human-readable summary of why these actions were chosen.
    pub reason: String,
    /// Estimated number of bytes that will be freed if all actions are applied.
    pub estimated_bytes_freed: u64,
}

impl CompactionAdvice {
    /// Returns `true` when urgency is medium or high (>= 2).
    pub fn is_urgent(&self) -> bool {
        self.urgency >= 2
    }
}

/// Analyses a [`StorageMetrics`] snapshot against a set of [`AdvisoryThresholds`]
/// and produces a [`CompactionAdvice`] describing what maintenance work should
/// be performed.
#[derive(Debug, Clone)]
pub struct CompactionAdvisor {
    /// The threshold configuration used for all advisory decisions.
    pub thresholds: AdvisoryThresholds,
}

impl CompactionAdvisor {
    /// Creates a new advisor with the provided thresholds.
    pub fn new(thresholds: AdvisoryThresholds) -> Self {
        Self { thresholds }
    }

    /// Analyses `metrics` and returns a [`CompactionAdvice`] describing any
    /// recommended actions.
    pub fn advise(&self, metrics: &StorageMetrics) -> CompactionAdvice {
        let mut actions: Vec<CompactionAction> = Vec::new();
        let mut reason_parts: Vec<String> = Vec::new();
        let mut compact_sled_added = false;
        let mut evict_bytes: u64 = 0;

        // 1. WAL threshold check.
        if metrics.wal_pending_ops > self.thresholds.wal_flush_threshold {
            actions.push(CompactionAction::CompactSled);
            compact_sled_added = true;
            reason_parts.push(format!(
                "WAL has {} pending ops (threshold {})",
                metrics.wal_pending_ops, self.thresholds.wal_flush_threshold
            ));
        }

        // 2. Orphan GC threshold check.
        if metrics.orphan_block_count > self.thresholds.orphan_gc_threshold {
            actions.push(CompactionAction::DropOrphanBlocks {
                cid_count: metrics.orphan_block_count as usize,
            });
            reason_parts.push(format!(
                "orphan block count {} exceeds threshold {}",
                metrics.orphan_block_count, self.thresholds.orphan_gc_threshold
            ));
        }

        // 3. Fragmentation threshold check (add CompactSled only if not already present).
        if metrics.fragmentation_ratio > self.thresholds.fragmentation_compact_threshold
            && !compact_sled_added
        {
            actions.push(CompactionAction::CompactSled);
            // compact_sled_added = true; // not read again, but kept for correctness
            reason_parts.push(format!(
                "fragmentation ratio {:.3} exceeds threshold {:.3}",
                metrics.fragmentation_ratio, self.thresholds.fragmentation_compact_threshold
            ));
        }

        // 4. SSTable merge threshold check.
        if metrics.sstable_count > self.thresholds.sstable_merge_threshold {
            let table_ids: Vec<String> = (0..metrics.sstable_count as usize)
                .map(|i| format!("table_{}", i))
                .collect();
            actions.push(CompactionAction::MergeSSTables { table_ids });
            reason_parts.push(format!(
                "SSTable count {} exceeds merge threshold {}",
                metrics.sstable_count, self.thresholds.sstable_merge_threshold
            ));
        }

        // 5. Cold-tier eviction threshold check.
        if metrics.usage_ratio() > self.thresholds.cold_tier_evict_ratio {
            let bytes_to_free = metrics.cold_tier_bytes / 4;
            evict_bytes = bytes_to_free;
            actions.push(CompactionAction::EvictTier {
                tier_name: "cold".to_string(),
                bytes_to_free,
            });
            reason_parts.push(format!(
                "usage ratio {:.3} exceeds cold eviction threshold {:.3}",
                metrics.usage_ratio(),
                self.thresholds.cold_tier_evict_ratio
            ));
        }

        // 6. If nothing triggered, signal no-op.
        if actions.is_empty() {
            actions.push(CompactionAction::NoActionNeeded);
        }

        // Compute urgency: exclude NoActionNeeded from the count.
        let real_action_count = actions
            .iter()
            .filter(|a| !matches!(a, CompactionAction::NoActionNeeded))
            .count();

        let urgency = match real_action_count {
            0 => 0,
            1 => 1,
            2 => 2,
            _ => 3,
        };

        // Estimated bytes freed = orphan bytes + evict bytes.
        let estimated_bytes_freed = metrics.orphan_block_bytes.saturating_add(evict_bytes);

        let reason = if reason_parts.is_empty() {
            "No issues detected; storage is healthy.".to_string()
        } else {
            reason_parts.join("; ")
        };

        CompactionAdvice {
            actions,
            urgency,
            reason,
            estimated_bytes_freed,
        }
    }

    /// Returns one explanatory string per action describing what will be done
    /// and why.
    pub fn explain(&self, advice: &CompactionAdvice) -> Vec<String> {
        advice
            .actions
            .iter()
            .map(|action| match action {
                CompactionAction::CompactSled => {
                    "CompactSled: flush the WAL and run sled compaction \
                     to reclaim fragmented space and reduce pending writes."
                        .to_string()
                }
                CompactionAction::MergeSSTables { table_ids } => {
                    format!(
                        "MergeSSTables: merge {} SSTable files ({}) \
                         to reduce read amplification and reclaim space.",
                        table_ids.len(),
                        table_ids.join(", ")
                    )
                }
                CompactionAction::EvictTier {
                    tier_name,
                    bytes_to_free,
                } => {
                    format!(
                        "EvictTier({}): evict approximately {} bytes of cold data \
                         because storage usage exceeds the configured ratio.",
                        tier_name, bytes_to_free
                    )
                }
                CompactionAction::DropOrphanBlocks { cid_count } => {
                    format!(
                        "DropOrphanBlocks: garbage-collect {} unreferenced blocks \
                         to reclaim storage occupied by data with no live references.",
                        cid_count
                    )
                }
                CompactionAction::NoActionNeeded => {
                    "NoActionNeeded: all storage metrics are within healthy thresholds.".to_string()
                }
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper that returns a healthy metrics snapshot (nothing over threshold).
    fn healthy_metrics() -> StorageMetrics {
        StorageMetrics {
            sled_bytes_used: 100,
            sled_bytes_free: 900,
            wal_pending_ops: 10,
            orphan_block_count: 5,
            orphan_block_bytes: 1_024,
            hot_tier_bytes: 200,
            warm_tier_bytes: 100,
            cold_tier_bytes: 400,
            sstable_count: 3,
            fragmentation_ratio: 0.05,
        }
    }

    fn default_advisor() -> CompactionAdvisor {
        CompactionAdvisor::new(AdvisoryThresholds::default())
    }

    // ------------------------------------------------------------------
    // Construction
    // ------------------------------------------------------------------

    #[test]
    fn test_new_with_custom_thresholds() {
        let thresholds = AdvisoryThresholds {
            wal_flush_threshold: 500,
            orphan_gc_threshold: 50,
            fragmentation_compact_threshold: 0.2,
            sstable_merge_threshold: 5,
            cold_tier_evict_ratio: 0.7,
        };
        let advisor = CompactionAdvisor::new(thresholds.clone());
        assert_eq!(advisor.thresholds, thresholds);
    }

    // ------------------------------------------------------------------
    // advise() – no issues
    // ------------------------------------------------------------------

    #[test]
    fn test_advise_no_issues_returns_no_action_needed() {
        let advisor = default_advisor();
        let advice = advisor.advise(&healthy_metrics());
        assert_eq!(advice.actions, vec![CompactionAction::NoActionNeeded]);
        assert_eq!(advice.urgency, 0);
    }

    // ------------------------------------------------------------------
    // advise() – individual threshold breaches
    // ------------------------------------------------------------------

    #[test]
    fn test_advise_wal_over_threshold_adds_compact_sled() {
        let advisor = default_advisor();
        let mut metrics = healthy_metrics();
        metrics.wal_pending_ops = 1_001;
        let advice = advisor.advise(&metrics);
        assert!(
            advice.actions.contains(&CompactionAction::CompactSled),
            "Expected CompactSled action"
        );
        assert!(!advice.actions.contains(&CompactionAction::NoActionNeeded));
    }

    #[test]
    fn test_advise_orphans_over_threshold_adds_drop_orphan_blocks() {
        let advisor = default_advisor();
        let mut metrics = healthy_metrics();
        metrics.orphan_block_count = 101;
        let advice = advisor.advise(&metrics);
        let has_orphan = advice.actions.iter().any(
            |a| matches!(a, CompactionAction::DropOrphanBlocks { cid_count } if *cid_count == 101),
        );
        assert!(has_orphan, "Expected DropOrphanBlocks with cid_count=101");
    }

    #[test]
    fn test_advise_fragmentation_over_threshold_adds_compact_sled() {
        let advisor = default_advisor();
        let mut metrics = healthy_metrics();
        metrics.fragmentation_ratio = 0.5;
        let advice = advisor.advise(&metrics);
        assert!(
            advice.actions.contains(&CompactionAction::CompactSled),
            "Expected CompactSled due to fragmentation"
        );
    }

    #[test]
    fn test_advise_sstables_over_threshold_adds_merge_sstables_with_correct_ids() {
        let advisor = default_advisor();
        let mut metrics = healthy_metrics();
        metrics.sstable_count = 12;
        let advice = advisor.advise(&metrics);
        let merge_action = advice.actions.iter().find_map(|a| {
            if let CompactionAction::MergeSSTables { table_ids } = a {
                Some(table_ids.clone())
            } else {
                None
            }
        });
        let table_ids = merge_action.expect("Expected MergeSSTables action");
        assert_eq!(table_ids.len(), 12);
        assert_eq!(table_ids[0], "table_0");
        assert_eq!(table_ids[11], "table_11");
    }

    #[test]
    fn test_advise_cold_tier_eviction_triggered() {
        let advisor = default_advisor();
        let mut metrics = healthy_metrics();
        // usage = 900 / (900 + 1) ≈ 0.999 → well above 0.8
        metrics.sled_bytes_used = 900;
        metrics.sled_bytes_free = 1;
        metrics.cold_tier_bytes = 800;
        let advice = advisor.advise(&metrics);
        let evict_action = advice.actions.iter().find_map(|a| {
            if let CompactionAction::EvictTier {
                tier_name,
                bytes_to_free,
            } = a
            {
                Some((tier_name.clone(), *bytes_to_free))
            } else {
                None
            }
        });
        let (tier, freed) = evict_action.expect("Expected EvictTier action");
        assert_eq!(tier, "cold");
        assert_eq!(freed, 200); // 800 / 4
    }

    // ------------------------------------------------------------------
    // advise() – multiple conditions
    // ------------------------------------------------------------------

    #[test]
    fn test_advise_multiple_conditions_all_actions_present() {
        let advisor = default_advisor();
        let mut metrics = healthy_metrics();
        metrics.wal_pending_ops = 2_000; // triggers CompactSled
        metrics.orphan_block_count = 200; // triggers DropOrphanBlocks
        metrics.sstable_count = 15; // triggers MergeSSTables
        let advice = advisor.advise(&metrics);
        assert!(advice.actions.contains(&CompactionAction::CompactSled));
        assert!(advice
            .actions
            .iter()
            .any(|a| matches!(a, CompactionAction::DropOrphanBlocks { .. })));
        assert!(advice
            .actions
            .iter()
            .any(|a| matches!(a, CompactionAction::MergeSSTables { .. })));
        assert!(!advice.actions.contains(&CompactionAction::NoActionNeeded));
    }

    // ------------------------------------------------------------------
    // Urgency levels
    // ------------------------------------------------------------------

    #[test]
    fn test_urgency_zero_for_no_actions() {
        let advisor = default_advisor();
        let advice = advisor.advise(&healthy_metrics());
        assert_eq!(advice.urgency, 0);
    }

    #[test]
    fn test_urgency_one_for_single_action() {
        let advisor = default_advisor();
        let mut metrics = healthy_metrics();
        metrics.wal_pending_ops = 5_000; // exactly one action
        let advice = advisor.advise(&metrics);
        assert_eq!(advice.urgency, 1);
    }

    #[test]
    fn test_urgency_two_for_two_actions() {
        let advisor = default_advisor();
        let mut metrics = healthy_metrics();
        metrics.wal_pending_ops = 5_000; // CompactSled
        metrics.orphan_block_count = 200; // DropOrphanBlocks
        let advice = advisor.advise(&metrics);
        assert_eq!(advice.urgency, 2);
    }

    #[test]
    fn test_urgency_three_for_three_or_more_actions() {
        let advisor = default_advisor();
        let mut metrics = healthy_metrics();
        metrics.wal_pending_ops = 5_000; // CompactSled
        metrics.orphan_block_count = 200; // DropOrphanBlocks
        metrics.sstable_count = 20; // MergeSSTables
        let advice = advisor.advise(&metrics);
        assert_eq!(advice.urgency, 3);
    }

    // ------------------------------------------------------------------
    // is_urgent()
    // ------------------------------------------------------------------

    #[test]
    fn test_is_urgent_true_for_urgency_two() {
        let advice = CompactionAdvice {
            actions: vec![CompactionAction::CompactSled],
            urgency: 2,
            reason: "test".to_string(),
            estimated_bytes_freed: 0,
        };
        assert!(advice.is_urgent());
    }

    #[test]
    fn test_is_urgent_false_for_urgency_one() {
        let advice = CompactionAdvice {
            actions: vec![CompactionAction::CompactSled],
            urgency: 1,
            reason: "test".to_string(),
            estimated_bytes_freed: 0,
        };
        assert!(!advice.is_urgent());
    }

    #[test]
    fn test_is_urgent_true_for_urgency_three() {
        let advice = CompactionAdvice {
            actions: vec![CompactionAction::CompactSled],
            urgency: 3,
            reason: "test".to_string(),
            estimated_bytes_freed: 0,
        };
        assert!(advice.is_urgent());
    }

    // ------------------------------------------------------------------
    // estimated_bytes_freed
    // ------------------------------------------------------------------

    #[test]
    fn test_estimated_bytes_freed_calculation() {
        let advisor = default_advisor();
        let mut metrics = healthy_metrics();
        metrics.orphan_block_count = 200;
        metrics.orphan_block_bytes = 10_000;
        // Also trigger eviction: usage > 0.8
        metrics.sled_bytes_used = 900;
        metrics.sled_bytes_free = 1;
        metrics.cold_tier_bytes = 400; // evict 100
        let advice = advisor.advise(&metrics);
        // orphan_block_bytes (10_000) + evict bytes (400/4 = 100) = 10_100
        assert_eq!(advice.estimated_bytes_freed, 10_100);
    }

    // ------------------------------------------------------------------
    // explain()
    // ------------------------------------------------------------------

    #[test]
    fn test_explain_returns_non_empty_strings_for_each_action() {
        let advisor = default_advisor();
        let mut metrics = healthy_metrics();
        metrics.wal_pending_ops = 5_000;
        metrics.orphan_block_count = 200;
        let advice = advisor.advise(&metrics);
        let explanations = advisor.explain(&advice);
        assert_eq!(explanations.len(), advice.actions.len());
        for explanation in &explanations {
            assert!(
                !explanation.is_empty(),
                "Expected non-empty explanation string"
            );
        }
    }

    #[test]
    fn test_explain_no_action_needed_has_description() {
        let advisor = default_advisor();
        let advice = advisor.advise(&healthy_metrics());
        let explanations = advisor.explain(&advice);
        assert_eq!(explanations.len(), 1);
        assert!(explanations[0].contains("NoActionNeeded"));
    }

    // ------------------------------------------------------------------
    // StorageMetrics helpers
    // ------------------------------------------------------------------

    #[test]
    fn test_usage_ratio_zero_when_both_used_and_free_are_zero() {
        let metrics = StorageMetrics {
            sled_bytes_used: 0,
            sled_bytes_free: 0,
            wal_pending_ops: 0,
            orphan_block_count: 0,
            orphan_block_bytes: 0,
            hot_tier_bytes: 0,
            warm_tier_bytes: 0,
            cold_tier_bytes: 0,
            sstable_count: 0,
            fragmentation_ratio: 0.0,
        };
        assert_eq!(metrics.usage_ratio(), 0.0);
    }

    #[test]
    fn test_total_bytes_sums_all_tiers() {
        let metrics = StorageMetrics {
            sled_bytes_used: 100,
            sled_bytes_free: 0,
            wal_pending_ops: 0,
            orphan_block_count: 0,
            orphan_block_bytes: 0,
            hot_tier_bytes: 200,
            warm_tier_bytes: 300,
            cold_tier_bytes: 400,
            sstable_count: 0,
            fragmentation_ratio: 0.0,
        };
        assert_eq!(metrics.total_bytes(), 1_000);
    }

    // ------------------------------------------------------------------
    // Fragmentation does not duplicate CompactSled when WAL also triggers it
    // ------------------------------------------------------------------

    #[test]
    fn test_fragmentation_does_not_duplicate_compact_sled_when_wal_also_triggers() {
        let advisor = default_advisor();
        let mut metrics = healthy_metrics();
        metrics.wal_pending_ops = 5_000; // triggers CompactSled
        metrics.fragmentation_ratio = 0.9; // would also trigger CompactSled
        let advice = advisor.advise(&metrics);
        let compact_count = advice
            .actions
            .iter()
            .filter(|a| matches!(a, CompactionAction::CompactSled))
            .count();
        assert_eq!(compact_count, 1, "CompactSled should appear exactly once");
    }
}
