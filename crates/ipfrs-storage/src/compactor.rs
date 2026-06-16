//! Storage Compactor
//!
//! Background compaction manager with fragmentation detection. Tracks storage
//! regions (offset, size, used/free), analyses fragmentation ratios, and merges
//! adjacent free regions to reclaim wasted space.

/// The current lifecycle state of the compactor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompactionState {
    /// No compaction activity.
    Idle,
    /// Fragmentation analysis in progress.
    Analyzing,
    /// Active compaction running.
    Compacting,
    /// Last compaction finished successfully.
    Completed,
    /// Last compaction failed.
    Failed,
}

/// Configuration knobs for [`StorageCompactor`].
#[derive(Debug, Clone)]
pub struct CompactorConfig {
    /// Fragmentation ratio above which compaction is warranted (0.0–1.0).
    /// Default: 0.3 (30%).
    pub fragmentation_threshold: f64,
    /// Maximum bytes the compactor will process in a single run.
    /// Default: 100 MB.
    pub max_budget_bytes: u64,
    /// Minimum number of ticks between compaction runs.
    /// Default: 100.
    pub min_interval_ticks: u64,
}

impl Default for CompactorConfig {
    fn default() -> Self {
        Self {
            fragmentation_threshold: 0.3,
            max_budget_bytes: 100 * 1024 * 1024, // 100 MB
            min_interval_ticks: 100,
        }
    }
}

/// Result of a fragmentation analysis pass.
#[derive(Debug, Clone)]
pub struct FragmentationReport {
    /// Total bytes spanned by all regions.
    pub total_bytes: u64,
    /// Bytes occupied by used regions.
    pub used_bytes: u64,
    /// Ratio of free space to total space: `(total - used) / total`.
    pub fragmentation_ratio: f64,
    /// Number of distinct free (unused) regions.
    pub fragmented_regions: usize,
}

/// Result of a single compaction run.
#[derive(Debug, Clone)]
pub struct CompactionResult {
    /// Number of region-merge operations performed.
    pub regions_merged: usize,
    /// Total bytes freed by merging adjacent free regions.
    pub bytes_reclaimed: u64,
    /// Ticks elapsed during this compaction (always 1 for synchronous runs).
    pub duration_ticks: u64,
}

/// Cumulative statistics for the compactor.
#[derive(Debug, Clone)]
pub struct CompactorStats {
    /// Current lifecycle state.
    pub state: CompactionState,
    /// Number of compaction runs completed.
    pub runs_completed: u64,
    /// Lifetime bytes reclaimed.
    pub bytes_reclaimed_total: u64,
    /// Fragmentation ratio from the most recent analysis.
    pub current_fragmentation: f64,
}

/// Background compaction manager.
///
/// Maintains a set of storage regions described by `(offset, size, is_used)`
/// tuples and can analyse/compact them on demand.
pub struct StorageCompactor {
    config: CompactorConfig,
    state: CompactionState,
    last_run_tick: u64,
    current_tick: u64,
    runs_completed: u64,
    bytes_reclaimed_total: u64,
    /// Storage regions: (offset, size, is_used).
    regions: Vec<(u64, u64, bool)>,
    /// Cached fragmentation ratio from the most recent `analyze()`.
    last_fragmentation: f64,
}

impl StorageCompactor {
    /// Create a new compactor with the given configuration.
    pub fn new(config: CompactorConfig) -> Self {
        Self {
            config,
            state: CompactionState::Idle,
            last_run_tick: 0,
            current_tick: 0,
            runs_completed: 0,
            bytes_reclaimed_total: 0,
            regions: Vec::new(),
            last_fragmentation: 0.0,
        }
    }

    /// Register a storage region.
    pub fn add_region(&mut self, offset: u64, size: u64, is_used: bool) {
        self.regions.push((offset, size, is_used));
    }

    /// Remove the region at the given offset.
    /// Returns `true` if a region was found and removed.
    pub fn remove_region(&mut self, offset: u64) -> bool {
        if let Some(pos) = self.regions.iter().position(|r| r.0 == offset) {
            self.regions.remove(pos);
            true
        } else {
            false
        }
    }

    /// Analyse the current region set and return a fragmentation report.
    ///
    /// Sets the compactor state to [`CompactionState::Analyzing`] during the
    /// call and back to [`CompactionState::Idle`] afterwards (unless the
    /// compactor was already in a terminal state).
    pub fn analyze(&mut self) -> FragmentationReport {
        self.state = CompactionState::Analyzing;

        let total_bytes: u64 = self.regions.iter().map(|r| r.1).sum();
        let used_bytes: u64 = self.regions.iter().filter(|r| r.2).map(|r| r.1).sum();
        let fragmented_regions = self.regions.iter().filter(|r| !r.2).count();

        let fragmentation_ratio = if total_bytes == 0 {
            0.0
        } else {
            (total_bytes - used_bytes) as f64 / total_bytes as f64
        };

        self.last_fragmentation = fragmentation_ratio;

        // Return to idle after analysis
        self.state = CompactionState::Idle;

        FragmentationReport {
            total_bytes,
            used_bytes,
            fragmentation_ratio,
            fragmented_regions,
        }
    }

    /// Returns `true` if compaction should be triggered based on the current
    /// fragmentation ratio and the configured interval.
    pub fn should_compact(&self) -> bool {
        self.last_fragmentation > self.config.fragmentation_threshold
            && (self.current_tick.saturating_sub(self.last_run_tick)
                >= self.config.min_interval_ticks)
    }

    /// Run compaction: sort regions by offset, merge adjacent free regions.
    ///
    /// Returns an error if the compactor is already in the `Compacting` state.
    pub fn compact(&mut self) -> Result<CompactionResult, String> {
        if self.state == CompactionState::Compacting {
            return Err("compaction already in progress".to_string());
        }

        self.state = CompactionState::Compacting;

        // Sort by offset
        self.regions.sort_by_key(|r| r.0);

        let mut merged: Vec<(u64, u64, bool)> = Vec::with_capacity(self.regions.len());
        let mut regions_merged: usize = 0;
        let mut bytes_reclaimed: u64 = 0;
        let mut budget_remaining = self.config.max_budget_bytes;

        for &region in &self.regions {
            if let Some(last) = merged.last_mut() {
                // Two adjacent free regions can be merged if: both free AND
                // the second starts right where the first ends.
                let adjacent = last.0 + last.1 == region.0;
                let both_free = !last.2 && !region.2;

                if adjacent && both_free && budget_remaining >= region.1 {
                    // Merge: grow the last region, count the merge.
                    bytes_reclaimed += region.1;
                    budget_remaining = budget_remaining.saturating_sub(region.1);
                    last.1 += region.1;
                    regions_merged += 1;
                    continue;
                }
            }
            merged.push(region);
        }

        self.regions = merged;
        self.bytes_reclaimed_total += bytes_reclaimed;
        self.runs_completed += 1;
        self.last_run_tick = self.current_tick;
        self.state = CompactionState::Completed;

        Ok(CompactionResult {
            regions_merged,
            bytes_reclaimed,
            duration_ticks: 1,
        })
    }

    /// Advance the logical clock by one tick.
    pub fn tick(&mut self) {
        self.current_tick += 1;
    }

    /// Total bytes reclaimed across all compaction runs.
    pub fn reclaimed_bytes(&self) -> u64 {
        self.bytes_reclaimed_total
    }

    /// Return a snapshot of the compactor's cumulative statistics.
    pub fn stats(&self) -> CompactorStats {
        CompactorStats {
            state: self.state,
            runs_completed: self.runs_completed,
            bytes_reclaimed_total: self.bytes_reclaimed_total,
            current_fragmentation: self.last_fragmentation,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_compactor() -> StorageCompactor {
        StorageCompactor::new(CompactorConfig::default())
    }

    // --- Empty compactor ---

    #[test]
    fn empty_compactor_zero_fragmentation() {
        let mut c = default_compactor();
        let report = c.analyze();
        assert_eq!(report.total_bytes, 0);
        assert_eq!(report.used_bytes, 0);
        assert!((report.fragmentation_ratio - 0.0).abs() < f64::EPSILON);
        assert_eq!(report.fragmented_regions, 0);
    }

    #[test]
    fn empty_compactor_state_idle() {
        let c = default_compactor();
        assert_eq!(c.state, CompactionState::Idle);
    }

    #[test]
    fn empty_compactor_stats() {
        let c = default_compactor();
        let s = c.stats();
        assert_eq!(s.runs_completed, 0);
        assert_eq!(s.bytes_reclaimed_total, 0);
        assert!((s.current_fragmentation - 0.0).abs() < f64::EPSILON);
    }

    // --- Add regions / analyse ---

    #[test]
    fn add_regions_and_analyze_all_used() {
        let mut c = default_compactor();
        c.add_region(0, 1000, true);
        c.add_region(1000, 2000, true);
        let report = c.analyze();
        assert_eq!(report.total_bytes, 3000);
        assert_eq!(report.used_bytes, 3000);
        assert!((report.fragmentation_ratio - 0.0).abs() < f64::EPSILON);
        assert_eq!(report.fragmented_regions, 0);
    }

    #[test]
    fn add_regions_and_analyze_half_free() {
        let mut c = default_compactor();
        c.add_region(0, 500, true);
        c.add_region(500, 500, false);
        let report = c.analyze();
        assert_eq!(report.total_bytes, 1000);
        assert_eq!(report.used_bytes, 500);
        assert!((report.fragmentation_ratio - 0.5).abs() < f64::EPSILON);
        assert_eq!(report.fragmented_regions, 1);
    }

    #[test]
    fn analyze_multiple_free_regions() {
        let mut c = default_compactor();
        c.add_region(0, 100, false);
        c.add_region(100, 200, true);
        c.add_region(300, 100, false);
        c.add_region(400, 100, false);
        let report = c.analyze();
        assert_eq!(report.total_bytes, 500);
        assert_eq!(report.used_bytes, 200);
        assert_eq!(report.fragmented_regions, 3);
        let expected = 300.0 / 500.0;
        assert!((report.fragmentation_ratio - expected).abs() < f64::EPSILON);
    }

    // --- should_compact ---

    #[test]
    fn should_compact_below_threshold() {
        let mut c = StorageCompactor::new(CompactorConfig {
            fragmentation_threshold: 0.5,
            min_interval_ticks: 0,
            ..CompactorConfig::default()
        });
        c.add_region(0, 800, true);
        c.add_region(800, 200, false);
        c.analyze(); // ratio = 0.2 < 0.5
        assert!(!c.should_compact());
    }

    #[test]
    fn should_compact_above_threshold() {
        let mut c = StorageCompactor::new(CompactorConfig {
            fragmentation_threshold: 0.1,
            min_interval_ticks: 0,
            ..CompactorConfig::default()
        });
        c.add_region(0, 200, true);
        c.add_region(200, 800, false);
        c.analyze(); // ratio = 0.8 > 0.1
        assert!(c.should_compact());
    }

    #[test]
    fn should_compact_respects_interval() {
        let mut c = StorageCompactor::new(CompactorConfig {
            fragmentation_threshold: 0.1,
            min_interval_ticks: 50,
            ..CompactorConfig::default()
        });
        c.add_region(0, 100, true);
        c.add_region(100, 900, false);
        c.analyze();
        // Haven't advanced enough ticks since last_run_tick=0 and current_tick=0
        // Actually current_tick=0, last_run_tick=0, diff=0 < 50
        assert!(!c.should_compact());

        for _ in 0..50 {
            c.tick();
        }
        assert!(c.should_compact());
    }

    #[test]
    fn should_compact_not_after_recent_run() {
        let mut c = StorageCompactor::new(CompactorConfig {
            fragmentation_threshold: 0.1,
            min_interval_ticks: 10,
            ..CompactorConfig::default()
        });
        c.add_region(0, 100, false);
        c.add_region(100, 100, false);
        for _ in 0..20 {
            c.tick();
        }
        c.analyze();
        assert!(c.should_compact());
        let _ = c.compact();
        // Re-analyse with new regions
        c.add_region(200, 100, false);
        c.analyze();
        // Only 0 ticks since last run
        assert!(!c.should_compact());
    }

    // --- compact ---

    #[test]
    fn compact_merges_adjacent_free_regions() {
        let mut c = default_compactor();
        c.add_region(0, 100, false);
        c.add_region(100, 200, false);
        c.add_region(300, 300, false);
        let result = c.compact().expect("compaction should succeed");
        assert_eq!(result.regions_merged, 2);
        assert_eq!(result.bytes_reclaimed, 500); // 200 + 300
                                                 // All three merged into one region starting at 0 with size 600
        assert_eq!(c.regions.len(), 1);
        assert_eq!(c.regions[0], (0, 600, false));
    }

    #[test]
    fn compact_does_not_merge_used_regions() {
        let mut c = default_compactor();
        c.add_region(0, 100, true);
        c.add_region(100, 200, true);
        let result = c.compact().expect("compaction should succeed");
        assert_eq!(result.regions_merged, 0);
        assert_eq!(result.bytes_reclaimed, 0);
        assert_eq!(c.regions.len(), 2);
    }

    #[test]
    fn compact_non_adjacent_free_regions_stay_separate() {
        let mut c = default_compactor();
        c.add_region(0, 100, false);
        c.add_region(200, 100, false); // gap at offset 100..200
        let result = c.compact().expect("compaction should succeed");
        assert_eq!(result.regions_merged, 0);
        assert_eq!(result.bytes_reclaimed, 0);
        assert_eq!(c.regions.len(), 2);
    }

    #[test]
    fn compact_mixed_regions() {
        let mut c = default_compactor();
        // free, free, used, free, free
        c.add_region(0, 100, false);
        c.add_region(100, 100, false);
        c.add_region(200, 100, true);
        c.add_region(300, 100, false);
        c.add_region(400, 100, false);
        let result = c.compact().expect("compaction should succeed");
        // First two free merge, last two free merge
        assert_eq!(result.regions_merged, 2);
        assert_eq!(result.bytes_reclaimed, 200); // 100 + 100
        assert_eq!(c.regions.len(), 3);
    }

    #[test]
    fn compact_free_used_alternating() {
        let mut c = default_compactor();
        c.add_region(0, 50, false);
        c.add_region(50, 50, true);
        c.add_region(100, 50, false);
        c.add_region(150, 50, true);
        c.add_region(200, 50, false);
        let result = c.compact().expect("compaction should succeed");
        assert_eq!(result.regions_merged, 0);
        assert_eq!(c.regions.len(), 5);
    }

    // --- State transitions ---

    #[test]
    fn state_idle_to_analyzing_to_idle() {
        let mut c = default_compactor();
        assert_eq!(c.state, CompactionState::Idle);
        // analyze sets Analyzing then returns to Idle
        c.add_region(0, 100, true);
        let _ = c.analyze();
        assert_eq!(c.state, CompactionState::Idle);
    }

    #[test]
    fn state_compacting_to_completed() {
        let mut c = default_compactor();
        c.add_region(0, 100, false);
        let _ = c.compact();
        assert_eq!(c.state, CompactionState::Completed);
    }

    #[test]
    fn cannot_compact_while_compacting() {
        // We need to test this by manipulating state directly since compact()
        // always finishes synchronously. We'll verify the guard works.
        let mut c = default_compactor();
        c.state = CompactionState::Compacting;
        let result = c.compact();
        assert!(result.is_err());
        assert_eq!(
            result.expect_err("should be error"),
            "compaction already in progress"
        );
    }

    // --- remove_region ---

    #[test]
    fn remove_region_existing() {
        let mut c = default_compactor();
        c.add_region(0, 100, true);
        c.add_region(100, 200, false);
        assert!(c.remove_region(0));
        assert_eq!(c.regions.len(), 1);
        assert_eq!(c.regions[0].0, 100);
    }

    #[test]
    fn remove_region_nonexistent() {
        let mut c = default_compactor();
        c.add_region(0, 100, true);
        assert!(!c.remove_region(999));
        assert_eq!(c.regions.len(), 1);
    }

    // --- Bytes reclaimed tracking ---

    #[test]
    fn reclaimed_bytes_zero_initially() {
        let c = default_compactor();
        assert_eq!(c.reclaimed_bytes(), 0);
    }

    #[test]
    fn reclaimed_bytes_after_compact() {
        let mut c = default_compactor();
        c.add_region(0, 100, false);
        c.add_region(100, 200, false);
        let _ = c.compact();
        assert_eq!(c.reclaimed_bytes(), 200);
    }

    // --- Multiple compaction runs ---

    #[test]
    fn multiple_runs_accumulate_stats() {
        let mut c = default_compactor();
        // First run: two adjacent free
        c.add_region(0, 100, false);
        c.add_region(100, 100, false);
        let r1 = c.compact().expect("run 1");
        assert_eq!(r1.bytes_reclaimed, 100);
        assert_eq!(c.runs_completed, 1);

        // Add more free regions — these are adjacent to the merged (0,200) region
        c.add_region(200, 50, false);
        c.add_region(250, 50, false);
        let r2 = c.compact().expect("run 2");
        // All three free regions merge: (0,200)+(200,50)+(250,50) → reclaims 100
        assert_eq!(r2.bytes_reclaimed, 100);
        assert_eq!(c.runs_completed, 2);
        assert_eq!(c.reclaimed_bytes(), 200);
    }

    #[test]
    fn stats_reflect_multiple_runs() {
        let mut c = default_compactor();
        c.add_region(0, 100, false);
        c.add_region(100, 100, false);
        let _ = c.compact(); // merges → (0,200), reclaimed=100
                             // Add non-adjacent free regions (with a used gap)
        c.add_region(500, 100, false);
        c.add_region(600, 100, false);
        let _ = c.compact(); // merges (500,100)+(600,100), reclaimed=100

        let s = c.stats();
        assert_eq!(s.runs_completed, 2);
        assert_eq!(s.bytes_reclaimed_total, 200);
        assert_eq!(s.state, CompactionState::Completed);
    }

    // --- Tick ---

    #[test]
    fn tick_increments() {
        let mut c = default_compactor();
        assert_eq!(c.current_tick, 0);
        c.tick();
        c.tick();
        c.tick();
        assert_eq!(c.current_tick, 3);
    }

    // --- Budget limit ---

    #[test]
    fn compact_respects_budget() {
        let mut c = StorageCompactor::new(CompactorConfig {
            max_budget_bytes: 150,
            ..CompactorConfig::default()
        });
        // Three adjacent free regions of 100 each
        c.add_region(0, 100, false);
        c.add_region(100, 100, false);
        c.add_region(200, 100, false);
        let result = c.compact().expect("compaction should succeed");
        // Budget = 150, first merge costs 100, second costs 100 => second exceeds 50 remaining
        assert_eq!(result.regions_merged, 1);
        assert_eq!(result.bytes_reclaimed, 100);
    }

    // --- Edge cases ---

    #[test]
    fn compact_single_free_region() {
        let mut c = default_compactor();
        c.add_region(0, 500, false);
        let result = c.compact().expect("compaction should succeed");
        assert_eq!(result.regions_merged, 0);
        assert_eq!(result.bytes_reclaimed, 0);
        assert_eq!(c.regions.len(), 1);
    }

    #[test]
    fn compact_single_used_region() {
        let mut c = default_compactor();
        c.add_region(0, 500, true);
        let result = c.compact().expect("compaction should succeed");
        assert_eq!(result.regions_merged, 0);
        assert_eq!(c.regions.len(), 1);
    }

    #[test]
    fn analyze_only_free_regions() {
        let mut c = default_compactor();
        c.add_region(0, 1000, false);
        let report = c.analyze();
        assert_eq!(report.fragmentation_ratio, 1.0);
        assert_eq!(report.fragmented_regions, 1);
    }

    #[test]
    fn compact_many_adjacent_free() {
        let mut c = default_compactor();
        for i in 0..10 {
            c.add_region(i * 100, 100, false);
        }
        let result = c.compact().expect("compaction should succeed");
        assert_eq!(result.regions_merged, 9);
        assert_eq!(result.bytes_reclaimed, 900);
        assert_eq!(c.regions.len(), 1);
        assert_eq!(c.regions[0], (0, 1000, false));
    }

    #[test]
    fn compact_unsorted_regions() {
        let mut c = default_compactor();
        // Add in reverse order — compact should sort first
        c.add_region(200, 100, false);
        c.add_region(0, 100, false);
        c.add_region(100, 100, false);
        let result = c.compact().expect("compaction should succeed");
        assert_eq!(result.regions_merged, 2);
        assert_eq!(c.regions.len(), 1);
        assert_eq!(c.regions[0], (0, 300, false));
    }

    #[test]
    fn default_config_values() {
        let cfg = CompactorConfig::default();
        assert!((cfg.fragmentation_threshold - 0.3).abs() < f64::EPSILON);
        assert_eq!(cfg.max_budget_bytes, 100 * 1024 * 1024);
        assert_eq!(cfg.min_interval_ticks, 100);
    }

    #[test]
    fn compaction_result_fields() {
        let mut c = default_compactor();
        c.add_region(0, 64, false);
        c.add_region(64, 64, false);
        let result = c.compact().expect("compaction should succeed");
        assert_eq!(result.duration_ticks, 1);
        assert_eq!(result.regions_merged, 1);
        assert_eq!(result.bytes_reclaimed, 64);
    }

    #[test]
    fn state_transitions_full_cycle() {
        let mut c = default_compactor();
        assert_eq!(c.state, CompactionState::Idle);
        c.add_region(0, 100, false);
        c.add_region(100, 100, false);
        let _ = c.analyze();
        assert_eq!(c.state, CompactionState::Idle);
        let _ = c.compact();
        assert_eq!(c.state, CompactionState::Completed);
    }

    #[test]
    fn remove_region_then_analyze() {
        let mut c = default_compactor();
        c.add_region(0, 100, true);
        c.add_region(100, 100, false);
        c.remove_region(100);
        let report = c.analyze();
        assert_eq!(report.total_bytes, 100);
        assert_eq!(report.used_bytes, 100);
        assert_eq!(report.fragmented_regions, 0);
    }

    #[test]
    fn compact_free_between_used() {
        let mut c = default_compactor();
        c.add_region(0, 100, true);
        c.add_region(100, 100, false);
        c.add_region(200, 100, true);
        let result = c.compact().expect("compaction should succeed");
        assert_eq!(result.regions_merged, 0);
        assert_eq!(c.regions.len(), 3);
    }
}
