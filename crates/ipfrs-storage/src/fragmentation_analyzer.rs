//! Storage Fragmentation Analyzer
//!
//! Analyzes storage layout to detect fragmentation, recommend defragmentation
//! targets, and estimate compaction savings.

/// Represents a contiguous region of storage — either occupied by a block or free.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StorageExtent {
    /// Byte offset at which this extent begins.
    pub start_offset: u64,
    /// Length of this extent in bytes.
    pub length_bytes: u64,
    /// CID of the block stored here, or `None` if this is free space.
    pub cid: Option<String>,
}

impl StorageExtent {
    /// Returns `true` when this extent is unoccupied (free space).
    #[inline]
    pub fn is_free(&self) -> bool {
        self.cid.is_none()
    }

    /// Returns the exclusive end offset of this extent.
    #[inline]
    pub fn end_offset(&self) -> u64 {
        self.start_offset + self.length_bytes
    }
}

/// Summary statistics produced by [`StorageFragmentationAnalyzer::analyze`].
#[derive(Debug, Clone)]
pub struct FragmentationReport {
    /// Total addressable bytes (used + free).
    pub total_bytes: u64,
    /// Bytes occupied by live blocks.
    pub used_bytes: u64,
    /// Bytes that are currently unoccupied.
    pub free_bytes: u64,
    /// Number of distinct free extents.
    pub free_extent_count: usize,
    /// Size (in bytes) of the single largest free extent.
    pub largest_free_extent: u64,
    /// Fragmentation score in [0.0, 1.0].
    ///
    /// `0.0` means all free space is in one contiguous run (perfect).
    /// `1.0` means every byte of free space is isolated (totally fragmented).
    ///
    /// Formula: `1.0 - (largest_free_extent / free_bytes)` when `free_bytes > 0`, else `0.0`.
    pub fragmentation_score: f64,
}

impl FragmentationReport {
    /// Fraction of total storage that is in use: `used_bytes / total_bytes.max(1)`.
    pub fn utilization(&self) -> f64 {
        self.used_bytes as f64 / self.total_bytes.max(1) as f64
    }
}

/// A single block that would move during a compaction pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompactionCandidate {
    /// CID of the block to move.
    pub cid: String,
    /// Where the block lives today.
    pub current_offset: u64,
    /// Where the block would land after left-packing.
    pub target_offset: u64,
    /// Size of the block in bytes.
    pub size_bytes: u64,
}

impl CompactionCandidate {
    /// Bytes of gap eliminated by moving this block.
    ///
    /// Equivalent to `current_offset - target_offset`; saturating to avoid
    /// underflow if the caller constructs a candidate where target ≥ current.
    pub fn bytes_saved(&self) -> u64 {
        self.current_offset.saturating_sub(self.target_offset)
    }
}

/// Analyzes storage layout to detect fragmentation, recommend defragmentation
/// targets, and estimate compaction savings.
///
/// Extents are maintained in ascending `start_offset` order at all times.
pub struct StorageFragmentationAnalyzer {
    /// Storage extents, always sorted by `start_offset`.
    pub extents: Vec<StorageExtent>,
}

impl StorageFragmentationAnalyzer {
    /// Create a new, empty analyzer.
    pub fn new() -> Self {
        Self {
            extents: Vec::new(),
        }
    }

    /// Add an extent, maintaining sorted order by `start_offset`.
    pub fn add_extent(&mut self, extent: StorageExtent) {
        // Binary search for the insertion position to keep the vec sorted.
        let pos = self
            .extents
            .partition_point(|e| e.start_offset < extent.start_offset);
        self.extents.insert(pos, extent);
    }

    /// Mark the extent whose CID matches `cid` as free space.
    ///
    /// If no extent matches, this is a no-op.
    pub fn free_extent(&mut self, cid: &str) {
        for extent in &mut self.extents {
            if extent.cid.as_deref() == Some(cid) {
                extent.cid = None;
            }
        }
    }

    /// Walk all extents and compute fragmentation statistics.
    pub fn analyze(&self) -> FragmentationReport {
        let mut total_bytes: u64 = 0;
        let mut used_bytes: u64 = 0;
        let mut free_bytes: u64 = 0;
        let mut free_extent_count: usize = 0;
        let mut largest_free_extent: u64 = 0;

        for extent in &self.extents {
            total_bytes += extent.length_bytes;
            if extent.is_free() {
                free_bytes += extent.length_bytes;
                free_extent_count += 1;
                if extent.length_bytes > largest_free_extent {
                    largest_free_extent = extent.length_bytes;
                }
            } else {
                used_bytes += extent.length_bytes;
            }
        }

        let fragmentation_score = if free_bytes > 0 {
            1.0 - (largest_free_extent as f64 / free_bytes as f64)
        } else {
            0.0
        };

        FragmentationReport {
            total_bytes,
            used_bytes,
            free_bytes,
            free_extent_count,
            largest_free_extent,
            fragmentation_score,
        }
    }

    /// Compute which live blocks would need to move to achieve a fully
    /// left-packed layout, ordered by descending `bytes_saved`.
    ///
    /// Only blocks whose `target_offset` is strictly less than
    /// `current_offset` are included.
    pub fn compaction_plan(&self) -> Vec<CompactionCandidate> {
        let mut candidates: Vec<CompactionCandidate> = Vec::new();
        let mut write_cursor: u64 = 0;

        for extent in &self.extents {
            if !extent.is_free() {
                let target_offset = write_cursor;
                let current_offset = extent.start_offset;

                if target_offset < current_offset {
                    candidates.push(CompactionCandidate {
                        cid: extent.cid.clone().unwrap_or_default(),
                        current_offset,
                        target_offset,
                        size_bytes: extent.length_bytes,
                    });
                }

                write_cursor += extent.length_bytes;
            }
        }

        // Sort by bytes_saved descending (most valuable moves first).
        candidates.sort_by_key(|c| std::cmp::Reverse(c.bytes_saved()));
        candidates
    }

    /// Merge adjacent free extents into a single larger free extent.
    ///
    /// After this call the extent list is still sorted by `start_offset`.
    pub fn merge_free_extents(&mut self) {
        if self.extents.is_empty() {
            return;
        }

        let mut merged: Vec<StorageExtent> = Vec::with_capacity(self.extents.len());

        for extent in self.extents.drain(..) {
            if let Some(last) = merged.last_mut() {
                // Merge only when both extents are free and contiguous.
                if last.is_free() && extent.is_free() && last.end_offset() == extent.start_offset {
                    last.length_bytes += extent.length_bytes;
                    continue;
                }
            }
            merged.push(extent);
        }

        self.extents = merged;
    }

    /// Return the total number of extents (used and free).
    pub fn total_extents(&self) -> usize {
        self.extents.len()
    }
}

impl Default for StorageFragmentationAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn used(start: u64, len: u64, cid: &str) -> StorageExtent {
        StorageExtent {
            start_offset: start,
            length_bytes: len,
            cid: Some(cid.to_string()),
        }
    }

    fn free(start: u64, len: u64) -> StorageExtent {
        StorageExtent {
            start_offset: start,
            length_bytes: len,
            cid: None,
        }
    }

    // ── StorageExtent helpers ─────────────────────────────────────────────────

    #[test]
    fn test_extent_is_free_true_for_none_cid() {
        let e = free(0, 512);
        assert!(e.is_free());
    }

    #[test]
    fn test_extent_is_free_false_for_some_cid() {
        let e = used(0, 512, "QmA");
        assert!(!e.is_free());
    }

    #[test]
    fn test_extent_end_offset() {
        let e = used(100, 400, "QmA");
        assert_eq!(e.end_offset(), 500);
    }

    // ── add_extent maintains sorted order ─────────────────────────────────────

    #[test]
    fn test_add_extent_maintains_sorted_order() {
        let mut analyzer = StorageFragmentationAnalyzer::new();
        // Insert out of order
        analyzer.add_extent(used(1000, 256, "QmC"));
        analyzer.add_extent(used(0, 512, "QmA"));
        analyzer.add_extent(used(512, 488, "QmB"));

        let offsets: Vec<u64> = analyzer.extents.iter().map(|e| e.start_offset).collect();
        assert_eq!(offsets, vec![0, 512, 1000]);
    }

    #[test]
    fn test_add_extent_single_element() {
        let mut analyzer = StorageFragmentationAnalyzer::new();
        analyzer.add_extent(used(42, 100, "QmX"));
        assert_eq!(analyzer.total_extents(), 1);
        assert_eq!(analyzer.extents[0].start_offset, 42);
    }

    // ── free_extent marks as free ─────────────────────────────────────────────

    #[test]
    fn test_free_extent_marks_as_free() {
        let mut analyzer = StorageFragmentationAnalyzer::new();
        analyzer.add_extent(used(0, 512, "QmA"));
        analyzer.add_extent(used(512, 512, "QmB"));

        analyzer.free_extent("QmA");

        assert!(analyzer.extents[0].is_free());
        assert!(!analyzer.extents[1].is_free());
    }

    #[test]
    fn test_free_extent_noop_for_unknown_cid() {
        let mut analyzer = StorageFragmentationAnalyzer::new();
        analyzer.add_extent(used(0, 512, "QmA"));
        analyzer.free_extent("QmZZZ");
        // Should remain used
        assert!(!analyzer.extents[0].is_free());
    }

    // ── analyze: fragmentation_score = 0 when contiguous ─────────────────────

    #[test]
    fn test_analyze_score_zero_when_no_free_space() {
        let mut analyzer = StorageFragmentationAnalyzer::new();
        analyzer.add_extent(used(0, 1024, "QmA"));
        analyzer.add_extent(used(1024, 1024, "QmB"));

        let report = analyzer.analyze();
        assert_eq!(report.fragmentation_score, 0.0);
        assert_eq!(report.free_extent_count, 0);
    }

    #[test]
    fn test_analyze_score_zero_when_free_space_is_contiguous() {
        let mut analyzer = StorageFragmentationAnalyzer::new();
        // One used block followed by one big free block — all free space in one run.
        analyzer.add_extent(used(0, 512, "QmA"));
        analyzer.add_extent(free(512, 1024));

        let report = analyzer.analyze();
        assert_eq!(report.fragmentation_score, 0.0);
        assert_eq!(report.free_extent_count, 1);
        assert_eq!(report.largest_free_extent, 1024);
    }

    // ── analyze: score > 0 when fragmented ────────────────────────────────────

    #[test]
    fn test_analyze_score_positive_when_fragmented() {
        let mut analyzer = StorageFragmentationAnalyzer::new();
        // Alternating used/free — highly fragmented.
        analyzer.add_extent(used(0, 100, "QmA"));
        analyzer.add_extent(free(100, 100));
        analyzer.add_extent(used(200, 100, "QmB"));
        analyzer.add_extent(free(300, 100));
        analyzer.add_extent(used(400, 100, "QmC"));
        analyzer.add_extent(free(500, 100));

        let report = analyzer.analyze();
        // Three free extents of 100 bytes each; largest == 100 == 1/3 of free_bytes.
        // score = 1 - (100/300) = 1 - 0.333... ≈ 0.666...
        assert!(report.fragmentation_score > 0.0);
        assert!(report.fragmentation_score < 1.0);
        assert_eq!(report.free_extent_count, 3);
    }

    // ── analyze: largest_free_extent ─────────────────────────────────────────

    #[test]
    fn test_analyze_largest_free_extent() {
        let mut analyzer = StorageFragmentationAnalyzer::new();
        analyzer.add_extent(free(0, 50));
        analyzer.add_extent(used(50, 100, "QmA"));
        analyzer.add_extent(free(150, 200));
        analyzer.add_extent(used(350, 100, "QmB"));
        analyzer.add_extent(free(450, 30));

        let report = analyzer.analyze();
        assert_eq!(report.largest_free_extent, 200);
        assert_eq!(report.free_bytes, 280); // 50 + 200 + 30
        assert_eq!(report.free_extent_count, 3);
    }

    // ── analyze: utilization ──────────────────────────────────────────────────

    #[test]
    fn test_analyze_utilization() {
        let mut analyzer = StorageFragmentationAnalyzer::new();
        analyzer.add_extent(used(0, 750, "QmA"));
        analyzer.add_extent(free(750, 250));

        let report = analyzer.analyze();
        assert!((report.utilization() - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_analyze_utilization_full() {
        let mut analyzer = StorageFragmentationAnalyzer::new();
        analyzer.add_extent(used(0, 1024, "QmA"));

        let report = analyzer.analyze();
        assert!((report.utilization() - 1.0).abs() < 1e-10);
    }

    // ── compaction_plan ordering ──────────────────────────────────────────────

    #[test]
    fn test_compaction_plan_ordered_by_bytes_saved_desc() {
        let mut analyzer = StorageFragmentationAnalyzer::new();
        // Layout: [A:100][free:500][B:100][free:100][C:100]
        //   A stays at 0 → no movement
        //   B target = 100, current = 600 → saved = 500
        //   C target = 200, current = 800 → saved = 600
        analyzer.add_extent(used(0, 100, "QmA"));
        analyzer.add_extent(free(100, 500));
        analyzer.add_extent(used(600, 100, "QmB"));
        analyzer.add_extent(free(700, 100));
        analyzer.add_extent(used(800, 100, "QmC"));

        let plan = analyzer.compaction_plan();
        // QmC saves 600, QmB saves 500
        assert_eq!(plan.len(), 2);
        assert_eq!(plan[0].cid, "QmC");
        assert_eq!(plan[0].bytes_saved(), 600);
        assert_eq!(plan[1].cid, "QmB");
        assert_eq!(plan[1].bytes_saved(), 500);
    }

    // ── compaction_plan: bytes_saved ─────────────────────────────────────────

    #[test]
    fn test_compaction_candidate_bytes_saved() {
        let c = CompactionCandidate {
            cid: "QmA".to_string(),
            current_offset: 1000,
            target_offset: 300,
            size_bytes: 200,
        };
        assert_eq!(c.bytes_saved(), 700);
    }

    #[test]
    fn test_compaction_candidate_bytes_saved_saturating() {
        // target >= current should not underflow
        let c = CompactionCandidate {
            cid: "QmA".to_string(),
            current_offset: 100,
            target_offset: 200,
            size_bytes: 50,
        };
        assert_eq!(c.bytes_saved(), 0);
    }

    // ── merge_free_extents ────────────────────────────────────────────────────

    #[test]
    fn test_merge_free_extents_combines_adjacent() {
        let mut analyzer = StorageFragmentationAnalyzer::new();
        // Two adjacent free extents
        analyzer.add_extent(free(0, 100));
        analyzer.add_extent(free(100, 200));
        analyzer.add_extent(used(300, 50, "QmA"));

        analyzer.merge_free_extents();

        assert_eq!(analyzer.total_extents(), 2);
        let first = &analyzer.extents[0];
        assert!(first.is_free());
        assert_eq!(first.length_bytes, 300); // merged
        assert_eq!(first.start_offset, 0);
    }

    #[test]
    fn test_merge_free_extents_does_not_merge_separated() {
        let mut analyzer = StorageFragmentationAnalyzer::new();
        analyzer.add_extent(free(0, 100));
        analyzer.add_extent(used(100, 50, "QmA"));
        analyzer.add_extent(free(150, 100));

        analyzer.merge_free_extents();

        // Free extents are separated by a used block — should stay as two.
        assert_eq!(analyzer.total_extents(), 3);
    }

    #[test]
    fn test_merge_free_extents_multiple_runs() {
        let mut analyzer = StorageFragmentationAnalyzer::new();
        // free|free|used|free|free|free
        analyzer.add_extent(free(0, 50));
        analyzer.add_extent(free(50, 50));
        analyzer.add_extent(used(100, 100, "QmA"));
        analyzer.add_extent(free(200, 60));
        analyzer.add_extent(free(260, 40));
        analyzer.add_extent(free(300, 100));

        analyzer.merge_free_extents();

        // Two merged free extents + one used = 3 extents
        assert_eq!(analyzer.total_extents(), 3);
        assert_eq!(analyzer.extents[0].length_bytes, 100); // 50+50
        assert_eq!(analyzer.extents[2].length_bytes, 200); // 60+40+100
    }

    // ── total_extents ─────────────────────────────────────────────────────────

    #[test]
    fn test_total_extents_empty() {
        let analyzer = StorageFragmentationAnalyzer::new();
        assert_eq!(analyzer.total_extents(), 0);
    }

    #[test]
    fn test_total_extents_counts_all() {
        let mut analyzer = StorageFragmentationAnalyzer::new();
        analyzer.add_extent(used(0, 100, "QmA"));
        analyzer.add_extent(free(100, 50));
        analyzer.add_extent(used(150, 200, "QmB"));
        assert_eq!(analyzer.total_extents(), 3);
    }

    // ── free_bytes count ──────────────────────────────────────────────────────

    #[test]
    fn test_analyze_free_bytes_count() {
        let mut analyzer = StorageFragmentationAnalyzer::new();
        analyzer.add_extent(used(0, 400, "QmA"));
        analyzer.add_extent(free(400, 100));
        analyzer.add_extent(used(500, 400, "QmB"));
        analyzer.add_extent(free(900, 100));

        let report = analyzer.analyze();
        assert_eq!(report.free_bytes, 200);
        assert_eq!(report.used_bytes, 800);
        assert_eq!(report.total_bytes, 1000);
    }
}
