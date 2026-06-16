//! Storage Block Compactor
//!
//! Merges small fragmented blocks into larger compacted segments for storage efficiency,
//! tracking fragmentation metrics and producing compaction plans.

use std::collections::HashMap;

/// A small, fragmented block eligible for compaction.
#[derive(Debug, Clone)]
pub struct BlockFragment {
    /// Unique block identifier.
    pub block_id: u64,
    /// Content identifier (CID) for this block.
    pub cid: String,
    /// Size of the block in bytes.
    pub size_bytes: u64,
    /// Logical tick when the block was created.
    pub created_at_tick: u64,
    /// Logical tick when the block was last accessed.
    pub last_accessed_tick: u64,
}

/// A compacted segment consisting of multiple merged blocks.
#[derive(Debug, Clone)]
pub struct CompactionSegment {
    /// Unique segment identifier.
    pub segment_id: u64,
    /// Constituent block IDs, sorted ascending.
    pub block_ids: Vec<u64>,
    /// Total bytes of all constituent blocks.
    pub total_bytes: u64,
    /// Ideal target size for this segment in bytes.
    pub target_size_bytes: u64,
}

impl CompactionSegment {
    /// Returns the ratio of total_bytes to target_size_bytes.
    /// Returns 0.0 if target_size_bytes is 0.
    pub fn fill_ratio(&self) -> f64 {
        if self.target_size_bytes == 0 {
            0.0
        } else {
            self.total_bytes as f64 / self.target_size_bytes as f64
        }
    }

    /// Returns true if total_bytes >= target_size_bytes.
    pub fn is_full(&self) -> bool {
        self.total_bytes >= self.target_size_bytes
    }
}

/// A plan describing which blocks to merge into which segments.
#[derive(Debug, Clone)]
pub struct CompactionPlan {
    /// Segments to create.
    pub segments: Vec<CompactionSegment>,
    /// Total number of blocks assigned to segments.
    pub blocks_compacted: usize,
    /// Total bytes of all blocks assigned to segments.
    pub bytes_compacted: u64,
    /// Estimated overhead saved (64 bytes per block merged).
    pub estimated_savings_bytes: u64,
}

impl CompactionPlan {
    /// Returns the number of segments in this plan.
    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }
}

/// Configuration for the `StorageBlockCompactor`.
#[derive(Debug, Clone)]
pub struct CompactorConfig {
    /// Ideal segment size in bytes (default: 4 MB).
    pub target_segment_bytes: u64,
    /// Only compact blocks strictly smaller than this threshold (default: 64 KB).
    pub min_block_size_to_compact: u64,
    /// Maximum number of blocks per compacted segment (default: 128).
    pub max_blocks_per_segment: usize,
}

impl Default for CompactorConfig {
    fn default() -> Self {
        Self {
            target_segment_bytes: 4_194_304,   // 4 MB
            min_block_size_to_compact: 65_536, // 64 KB
            max_blocks_per_segment: 128,
        }
    }
}

/// Accumulated statistics for the `StorageBlockCompactor`.
#[derive(Debug, Clone, Default)]
pub struct CompactorStats {
    /// Number of compaction plans generated so far.
    pub total_plans_generated: u64,
    /// Total blocks that have been included in compaction plans.
    pub total_blocks_compacted: u64,
    /// Total bytes included in compaction plans.
    pub total_bytes_compacted: u64,
    /// Total segments created across all plans.
    pub total_segments_created: u64,
}

/// Merges small fragmented blocks into larger compacted segments.
pub struct StorageBlockCompactor {
    /// Registered fragments keyed by block_id.
    pub fragments: HashMap<u64, BlockFragment>,
    /// Compactor configuration.
    pub config: CompactorConfig,
    /// Accumulated statistics.
    pub stats: CompactorStats,
    /// Counter for generating unique segment IDs.
    pub next_segment_id: u64,
}

impl StorageBlockCompactor {
    /// Creates a new `StorageBlockCompactor` with the given configuration.
    pub fn new(config: CompactorConfig) -> Self {
        Self {
            fragments: HashMap::new(),
            config,
            stats: CompactorStats::default(),
            next_segment_id: 0,
        }
    }

    /// Registers a block as a compaction candidate.
    ///
    /// Only blocks with `size_bytes < min_block_size_to_compact` are registered;
    /// larger blocks are silently skipped.
    pub fn register_block(&mut self, block_id: u64, cid: String, size_bytes: u64, tick: u64) {
        if size_bytes >= self.config.min_block_size_to_compact {
            return;
        }
        let fragment = BlockFragment {
            block_id,
            cid,
            size_bytes,
            created_at_tick: tick,
            last_accessed_tick: tick,
        };
        self.fragments.insert(block_id, fragment);
    }

    /// Updates the last_accessed_tick for a registered block.
    ///
    /// Returns `true` if the block was found and updated, `false` otherwise.
    pub fn touch(&mut self, block_id: u64, tick: u64) -> bool {
        match self.fragments.get_mut(&block_id) {
            Some(fragment) => {
                fragment.last_accessed_tick = tick;
                true
            }
            None => false,
        }
    }

    /// Produces a compaction plan by greedily grouping fragments into segments.
    ///
    /// Fragments are sorted by size (ascending), then by block_id (ascending) for stable ordering.
    /// Segments with fewer than 2 blocks are excluded from the plan.
    pub fn plan_compaction(&mut self) -> CompactionPlan {
        // Collect and sort fragments: by size ascending, then block_id ascending for ties.
        let mut sorted: Vec<BlockFragment> = self.fragments.values().cloned().collect();
        sorted.sort_by(|a, b| {
            a.size_bytes
                .cmp(&b.size_bytes)
                .then_with(|| a.block_id.cmp(&b.block_id))
        });

        let target = self.config.target_segment_bytes;
        let max_per_seg = self.config.max_blocks_per_segment;

        let mut segments: Vec<CompactionSegment> = Vec::new();

        let mut idx = 0;
        while idx < sorted.len() {
            let mut seg_block_ids: Vec<u64> = Vec::new();
            let mut seg_bytes: u64 = 0;

            // Fill one segment.
            while idx < sorted.len() && seg_bytes < target && seg_block_ids.len() < max_per_seg {
                let frag = &sorted[idx];
                seg_block_ids.push(frag.block_id);
                seg_bytes += frag.size_bytes;
                idx += 1;
            }

            // Only keep segments with at least 2 blocks.
            if seg_block_ids.len() < 2 {
                continue;
            }

            seg_block_ids.sort_unstable();

            let seg = CompactionSegment {
                segment_id: self.next_segment_id,
                block_ids: seg_block_ids,
                total_bytes: seg_bytes,
                target_size_bytes: target,
            };
            self.next_segment_id += 1;
            segments.push(seg);
        }

        let blocks_compacted: usize = segments.iter().map(|s| s.block_ids.len()).sum();
        let bytes_compacted: u64 = segments.iter().map(|s| s.total_bytes).sum();
        let estimated_savings_bytes = 64 * blocks_compacted as u64;

        // Update accumulated statistics.
        self.stats.total_plans_generated += 1;
        self.stats.total_blocks_compacted += blocks_compacted as u64;
        self.stats.total_bytes_compacted += bytes_compacted;
        self.stats.total_segments_created += segments.len() as u64;

        CompactionPlan {
            segments,
            blocks_compacted,
            bytes_compacted,
            estimated_savings_bytes,
        }
    }

    /// Removes a registered block fragment.
    ///
    /// Returns `true` if the block was present and removed, `false` otherwise.
    pub fn remove_block(&mut self, block_id: u64) -> bool {
        self.fragments.remove(&block_id).is_some()
    }

    /// Returns the fragmentation ratio: blocks smaller than `target_segment_bytes / 2`
    /// divided by total registered blocks.
    ///
    /// Returns `0.0` if no fragments are registered.
    pub fn fragmentation_ratio(&self) -> f64 {
        let total = self.fragments.len();
        if total == 0 {
            return 0.0;
        }
        let threshold = self.config.target_segment_bytes / 2;
        let small_count = self
            .fragments
            .values()
            .filter(|f| f.size_bytes < threshold)
            .count();
        small_count as f64 / total as f64
    }

    /// Returns a reference to the accumulated statistics.
    pub fn stats(&self) -> &CompactorStats {
        &self.stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_compactor() -> StorageBlockCompactor {
        StorageBlockCompactor::new(CompactorConfig::default())
    }

    // ── register_block ───────────────────────────────────────────────────────

    #[test]
    fn test_register_small_block_accepted() {
        let mut c = default_compactor();
        c.register_block(1, "cid1".into(), 1024, 10);
        assert_eq!(c.fragments.len(), 1);
    }

    #[test]
    fn test_register_block_at_boundary_excluded() {
        let mut c = default_compactor();
        // size == min_block_size_to_compact → excluded
        c.register_block(1, "cid1".into(), 65_536, 10);
        assert!(c.fragments.is_empty());
    }

    #[test]
    fn test_register_large_block_not_registered() {
        let mut c = default_compactor();
        c.register_block(99, "cid99".into(), 1_000_000, 5);
        assert!(c.fragments.is_empty());
    }

    #[test]
    fn test_register_block_stores_fields() {
        let mut c = default_compactor();
        c.register_block(42, "bafycid".into(), 512, 100);
        let frag = c.fragments.get(&42).expect("fragment should exist");
        assert_eq!(frag.block_id, 42);
        assert_eq!(frag.cid, "bafycid");
        assert_eq!(frag.size_bytes, 512);
        assert_eq!(frag.created_at_tick, 100);
        assert_eq!(frag.last_accessed_tick, 100);
    }

    #[test]
    fn test_register_multiple_small_blocks() {
        let mut c = default_compactor();
        for i in 0..10_u64 {
            c.register_block(i, format!("cid{i}"), 1024 * (i + 1), i);
        }
        assert_eq!(c.fragments.len(), 10);
    }

    // ── touch ────────────────────────────────────────────────────────────────

    #[test]
    fn test_touch_updates_last_accessed_tick() {
        let mut c = default_compactor();
        c.register_block(1, "cid1".into(), 512, 5);
        let updated = c.touch(1, 99);
        assert!(updated);
        let frag = c.fragments.get(&1).expect("fragment should exist");
        assert_eq!(frag.last_accessed_tick, 99);
        assert_eq!(frag.created_at_tick, 5); // created_at unchanged
    }

    #[test]
    fn test_touch_returns_false_for_missing_block() {
        let mut c = default_compactor();
        assert!(!c.touch(999, 10));
    }

    #[test]
    fn test_touch_does_not_change_created_at() {
        let mut c = default_compactor();
        c.register_block(7, "cid7".into(), 256, 1);
        c.touch(7, 500);
        let frag = c.fragments.get(&7).unwrap();
        assert_eq!(frag.created_at_tick, 1);
    }

    // ── remove_block ─────────────────────────────────────────────────────────

    #[test]
    fn test_remove_existing_block_returns_true() {
        let mut c = default_compactor();
        c.register_block(5, "cid5".into(), 1000, 0);
        assert!(c.remove_block(5));
        assert!(c.fragments.is_empty());
    }

    #[test]
    fn test_remove_missing_block_returns_false() {
        let mut c = default_compactor();
        assert!(!c.remove_block(404));
    }

    // ── CompactionSegment helpers ─────────────────────────────────────────────

    #[test]
    fn test_fill_ratio_normal() {
        let seg = CompactionSegment {
            segment_id: 0,
            block_ids: vec![1, 2],
            total_bytes: 2_097_152,
            target_size_bytes: 4_194_304,
        };
        let ratio = seg.fill_ratio();
        assert!((ratio - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_fill_ratio_zero_target() {
        let seg = CompactionSegment {
            segment_id: 0,
            block_ids: vec![1],
            total_bytes: 1024,
            target_size_bytes: 0,
        };
        assert_eq!(seg.fill_ratio(), 0.0);
    }

    #[test]
    fn test_is_full_true() {
        let seg = CompactionSegment {
            segment_id: 0,
            block_ids: vec![1, 2],
            total_bytes: 4_194_304,
            target_size_bytes: 4_194_304,
        };
        assert!(seg.is_full());
    }

    #[test]
    fn test_is_full_false() {
        let seg = CompactionSegment {
            segment_id: 0,
            block_ids: vec![1, 2],
            total_bytes: 1024,
            target_size_bytes: 4_194_304,
        };
        assert!(!seg.is_full());
    }

    // ── plan_compaction ───────────────────────────────────────────────────────

    #[test]
    fn test_plan_groups_small_blocks_into_segment() {
        let mut c = default_compactor();
        for i in 0..5_u64 {
            c.register_block(i, format!("cid{i}"), 8_192, i);
        }
        let plan = c.plan_compaction();
        assert!(!plan.segments.is_empty());
        assert_eq!(plan.blocks_compacted, 5);
    }

    #[test]
    fn test_plan_excludes_singleton_segment() {
        // Only one block registered — can't form a segment of >= 2.
        let mut c = default_compactor();
        c.register_block(1, "cid1".into(), 1024, 0);
        let plan = c.plan_compaction();
        assert_eq!(plan.segment_count(), 0);
        assert_eq!(plan.blocks_compacted, 0);
        assert_eq!(plan.bytes_compacted, 0);
        assert_eq!(plan.estimated_savings_bytes, 0);
    }

    #[test]
    fn test_plan_blocks_compacted_total() {
        let mut c = default_compactor();
        for i in 0..10_u64 {
            c.register_block(i, format!("cid{i}"), 4096, i);
        }
        let plan = c.plan_compaction();
        assert_eq!(plan.blocks_compacted, 10);
    }

    #[test]
    fn test_plan_bytes_compacted_total() {
        let mut c = default_compactor();
        for i in 0..4_u64 {
            c.register_block(i, format!("cid{i}"), 1000, 0);
        }
        let plan = c.plan_compaction();
        assert_eq!(plan.bytes_compacted, 4000);
    }

    #[test]
    fn test_estimated_savings_bytes() {
        let mut c = default_compactor();
        for i in 0..6_u64 {
            c.register_block(i, format!("cid{i}"), 512, 0);
        }
        let plan = c.plan_compaction();
        assert_eq!(
            plan.estimated_savings_bytes,
            64 * plan.blocks_compacted as u64
        );
    }

    #[test]
    fn test_segment_count() {
        let mut c = default_compactor();
        // Fill more than one segment worth of blocks (target 4 MB, each block 1 MB-ish).
        // Use 65_535 bytes per block (just under the 64 KB threshold).
        // 4_194_304 / 65_535 ≈ 64 blocks to fill one segment.
        for i in 0..130_u64 {
            c.register_block(i, format!("cid{i}"), 65_535, i);
        }
        let plan = c.plan_compaction();
        assert!(plan.segment_count() >= 2);
    }

    #[test]
    fn test_max_blocks_per_segment_cap() {
        let config = CompactorConfig {
            max_blocks_per_segment: 4,
            target_segment_bytes: 4_194_304,
            ..CompactorConfig::default()
        };
        let mut c = StorageBlockCompactor::new(config);
        for i in 0..10_u64 {
            c.register_block(i, format!("cid{i}"), 256, 0);
        }
        let plan = c.plan_compaction();
        for seg in &plan.segments {
            assert!(seg.block_ids.len() <= 4);
        }
    }

    #[test]
    fn test_segment_block_ids_sorted_ascending() {
        let mut c = default_compactor();
        // Insert in reverse order to ensure sorting happens.
        for i in (0_u64..5).rev() {
            c.register_block(i, format!("cid{i}"), 1024, 0);
        }
        let plan = c.plan_compaction();
        for seg in &plan.segments {
            let sorted = {
                let mut ids = seg.block_ids.clone();
                ids.sort_unstable();
                ids
            };
            assert_eq!(seg.block_ids, sorted);
        }
    }

    #[test]
    fn test_plan_with_no_fragments_returns_empty_plan() {
        let mut c = default_compactor();
        let plan = c.plan_compaction();
        assert_eq!(plan.segment_count(), 0);
        assert_eq!(plan.blocks_compacted, 0);
        assert_eq!(plan.bytes_compacted, 0);
        assert_eq!(plan.estimated_savings_bytes, 0);
    }

    // ── fragmentation_ratio ───────────────────────────────────────────────────

    #[test]
    fn test_fragmentation_ratio_no_fragments() {
        let c = default_compactor();
        assert_eq!(c.fragmentation_ratio(), 0.0);
    }

    #[test]
    fn test_fragmentation_ratio_all_small() {
        let mut c = default_compactor();
        // target/2 = 2_097_152; register blocks well below that.
        for i in 0..5_u64 {
            c.register_block(i, format!("cid{i}"), 1024, 0);
        }
        let ratio = c.fragmentation_ratio();
        assert!((ratio - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_fragmentation_ratio_mixed() {
        let config = CompactorConfig {
            target_segment_bytes: 4_000,
            min_block_size_to_compact: 10_000,
            max_blocks_per_segment: 128,
        };
        let mut c = StorageBlockCompactor::new(config);
        // threshold = 4000 / 2 = 2000
        // 3 blocks below 2000, 2 blocks at or above 2000 but below 10000
        c.register_block(1, "a".into(), 500, 0); // < 2000
        c.register_block(2, "b".into(), 1000, 0); // < 2000
        c.register_block(3, "c".into(), 1500, 0); // < 2000
        c.register_block(4, "d".into(), 2000, 0); // >= 2000
        c.register_block(5, "e".into(), 3000, 0); // >= 2000
        let ratio = c.fragmentation_ratio();
        assert!((ratio - 3.0 / 5.0).abs() < 1e-9);
    }

    // ── stats accumulation ────────────────────────────────────────────────────

    #[test]
    fn test_stats_accumulate_across_plans() {
        let mut c = default_compactor();
        for i in 0..4_u64 {
            c.register_block(i, format!("cid{i}"), 1024, 0);
        }
        c.plan_compaction();
        c.plan_compaction(); // second plan on same fragments

        let s = c.stats();
        assert_eq!(s.total_plans_generated, 2);
        // Each plan covers 4 blocks (fragments are not removed by planning).
        assert_eq!(s.total_blocks_compacted, 8);
    }

    #[test]
    fn test_stats_total_segments_created() {
        let mut c = default_compactor();
        for i in 0..4_u64 {
            c.register_block(i, format!("cid{i}"), 512, 0);
        }
        let plan = c.plan_compaction();
        let seg_count = plan.segment_count() as u64;
        assert_eq!(c.stats().total_segments_created, seg_count);
    }

    #[test]
    fn test_stats_bytes_compacted_accumulate() {
        let mut c = default_compactor();
        for i in 0..4_u64 {
            c.register_block(i, format!("cid{i}"), 1000, 0);
        }
        c.plan_compaction(); // 4000 bytes
        c.plan_compaction(); // 4000 bytes again

        assert_eq!(c.stats().total_bytes_compacted, 8000);
    }

    #[test]
    fn test_stats_initial_zero() {
        let c = default_compactor();
        let s = c.stats();
        assert_eq!(s.total_plans_generated, 0);
        assert_eq!(s.total_blocks_compacted, 0);
        assert_eq!(s.total_bytes_compacted, 0);
        assert_eq!(s.total_segments_created, 0);
    }
}
