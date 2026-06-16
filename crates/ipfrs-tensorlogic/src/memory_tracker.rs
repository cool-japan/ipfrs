//! Inference Memory Tracker
//!
//! Tracks memory allocations made during inference sessions, detecting leaks,
//! peaks, and per-rule memory usage.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// AllocEvent
// ---------------------------------------------------------------------------

/// An event that can be recorded by [`InferenceMemoryTracker`].
#[derive(Clone, Debug, PartialEq)]
pub enum AllocEvent {
    /// A memory region was allocated.
    Alloc {
        /// Unique identifier for the region.
        region_id: u64,
        /// Size of the allocation in bytes.
        size_bytes: u64,
        /// Optional rule that triggered this allocation.
        rule_id: Option<u64>,
    },
    /// A previously allocated region was freed.
    Free {
        /// Identifier of the region to free.
        region_id: u64,
    },
    /// A named checkpoint in the event log.
    Checkpoint {
        /// Human-readable label for the checkpoint.
        label: String,
    },
}

// ---------------------------------------------------------------------------
// RegionInfo
// ---------------------------------------------------------------------------

/// Metadata about a single allocated memory region.
#[derive(Clone, Debug, PartialEq)]
pub struct RegionInfo {
    /// Unique identifier for this region.
    pub region_id: u64,
    /// Size of the region in bytes.
    pub size_bytes: u64,
    /// Rule that caused this allocation, if any.
    pub rule_id: Option<u64>,
    /// Index into the event log at which this region was allocated.
    pub allocated_at_event: usize,
    /// Whether this region has been freed.
    pub freed: bool,
}

// ---------------------------------------------------------------------------
// MemorySnapshot
// ---------------------------------------------------------------------------

/// A point-in-time view of tracked memory usage.
#[derive(Clone, Debug, PartialEq)]
pub struct MemorySnapshot {
    /// Total bytes in currently-live (non-freed) regions.
    pub live_bytes: u64,
    /// Number of currently-live regions.
    pub live_regions: usize,
    /// Highest `live_bytes` value observed during the session.
    pub peak_bytes: u64,
    /// Bytes in regions that were never freed (live at end of session).
    pub leaked_bytes: u64,
}

impl MemorySnapshot {
    /// Returns the fraction of `capacity_bytes` that is currently live.
    ///
    /// Returns `0.0` when `capacity_bytes` is zero to avoid division by zero.
    pub fn utilization(&self, capacity_bytes: u64) -> f64 {
        if capacity_bytes == 0 {
            return 0.0;
        }
        self.live_bytes as f64 / capacity_bytes as f64
    }
}

// ---------------------------------------------------------------------------
// TrackerStats
// ---------------------------------------------------------------------------

/// Aggregate counters for an [`InferenceMemoryTracker`] session.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TrackerStats {
    /// Total number of `Alloc` events recorded.
    pub total_allocs: u64,
    /// Total number of `Free` events recorded (including ignored double-frees).
    pub total_frees: u64,
    /// Total number of `Checkpoint` events recorded.
    pub total_checkpoints: u64,
}

impl TrackerStats {
    /// Returns the number of allocations that were never matched by a free.
    ///
    /// This is computed as `total_allocs - total_frees`, clamped to zero so
    /// that spurious double-frees do not produce a negative result.
    pub fn leak_count(&self) -> u64 {
        self.total_allocs.saturating_sub(self.total_frees)
    }
}

// ---------------------------------------------------------------------------
// InferenceMemoryTracker
// ---------------------------------------------------------------------------

/// Tracks memory allocations made during a single inference session.
///
/// Record [`AllocEvent`]s via [`InferenceMemoryTracker::record`], then query
/// the current state via [`InferenceMemoryTracker::snapshot`],
/// [`InferenceMemoryTracker::leaked_regions`], and
/// [`InferenceMemoryTracker::memory_by_rule`].
#[derive(Debug)]
pub struct InferenceMemoryTracker {
    /// Full, ordered event log.
    pub events: Vec<AllocEvent>,
    /// Map from `region_id` to its [`RegionInfo`].
    pub regions: HashMap<u64, RegionInfo>,
    /// Current live byte total.
    pub live_bytes: u64,
    /// Highest live_bytes value seen so far.
    pub peak_bytes: u64,
    /// Aggregate counters.
    pub stats: TrackerStats,
}

impl InferenceMemoryTracker {
    /// Creates a new, empty tracker.
    pub fn new() -> Self {
        Self {
            events: Vec::new(),
            regions: HashMap::new(),
            live_bytes: 0,
            peak_bytes: 0,
            stats: TrackerStats::default(),
        }
    }

    /// Records an event and updates internal state accordingly.
    ///
    /// - [`AllocEvent::Alloc`]: inserts a new [`RegionInfo`], increases
    ///   `live_bytes`, and updates `peak_bytes`.
    /// - [`AllocEvent::Free`]: marks the region freed and decreases
    ///   `live_bytes`. Silently ignored when the region is not found or
    ///   already freed (double-free).
    /// - [`AllocEvent::Checkpoint`]: just appended to the event log.
    pub fn record(&mut self, event: AllocEvent) {
        match &event {
            AllocEvent::Alloc {
                region_id,
                size_bytes,
                rule_id,
            } => {
                let event_idx = self.events.len();
                self.regions.insert(
                    *region_id,
                    RegionInfo {
                        region_id: *region_id,
                        size_bytes: *size_bytes,
                        rule_id: *rule_id,
                        allocated_at_event: event_idx,
                        freed: false,
                    },
                );
                self.live_bytes = self.live_bytes.saturating_add(*size_bytes);
                if self.live_bytes > self.peak_bytes {
                    self.peak_bytes = self.live_bytes;
                }
                self.stats.total_allocs += 1;
            }
            AllocEvent::Free { region_id } => {
                self.stats.total_frees += 1;
                if let Some(region) = self.regions.get_mut(region_id) {
                    if !region.freed {
                        region.freed = true;
                        self.live_bytes = self.live_bytes.saturating_sub(region.size_bytes);
                    }
                    // else: already freed — double-free, ignore silently
                }
                // region not found — ignore silently
            }
            AllocEvent::Checkpoint { .. } => {
                self.stats.total_checkpoints += 1;
            }
        }
        self.events.push(event);
    }

    /// Returns a snapshot of the current memory state.
    ///
    /// `leaked_bytes` is the sum of `size_bytes` for every region that has
    /// never been freed.
    pub fn snapshot(&self) -> MemorySnapshot {
        let live_regions = self.regions.values().filter(|r| !r.freed).count();
        let leaked_bytes: u64 = self
            .regions
            .values()
            .filter(|r| !r.freed)
            .map(|r| r.size_bytes)
            .sum();

        MemorySnapshot {
            live_bytes: self.live_bytes,
            live_regions,
            peak_bytes: self.peak_bytes,
            leaked_bytes,
        }
    }

    /// Returns the total bytes currently held by live regions associated with
    /// the given `rule_id`.
    pub fn memory_by_rule(&self, rule_id: u64) -> u64 {
        self.regions
            .values()
            .filter(|r| !r.freed && r.rule_id == Some(rule_id))
            .map(|r| r.size_bytes)
            .sum()
    }

    /// Returns references to all regions that have not been freed.
    pub fn leaked_regions(&self) -> Vec<&RegionInfo> {
        let mut leaked: Vec<&RegionInfo> = self.regions.values().filter(|r| !r.freed).collect();
        // Sort by region_id for deterministic output.
        leaked.sort_by_key(|r| r.region_id);
        leaked
    }

    /// Returns a reference to the aggregate counters.
    pub fn stats(&self) -> &TrackerStats {
        &self.stats
    }

    /// Resets all state so the tracker can be reused for a new session.
    pub fn reset(&mut self) {
        self.events.clear();
        self.regions.clear();
        self.live_bytes = 0;
        self.peak_bytes = 0;
        self.stats = TrackerStats::default();
    }
}

impl Default for InferenceMemoryTracker {
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

    fn alloc(region_id: u64, size_bytes: u64, rule_id: Option<u64>) -> AllocEvent {
        AllocEvent::Alloc {
            region_id,
            size_bytes,
            rule_id,
        }
    }

    fn free(region_id: u64) -> AllocEvent {
        AllocEvent::Free { region_id }
    }

    fn checkpoint(label: &str) -> AllocEvent {
        AllocEvent::Checkpoint {
            label: label.to_string(),
        }
    }

    // -----------------------------------------------------------------------
    // 1. Alloc increases live_bytes
    // -----------------------------------------------------------------------
    #[test]
    fn test_alloc_increases_live_bytes() {
        let mut tracker = InferenceMemoryTracker::new();
        tracker.record(alloc(1, 1024, None));
        assert_eq!(tracker.live_bytes, 1024);
        tracker.record(alloc(2, 512, None));
        assert_eq!(tracker.live_bytes, 1536);
    }

    // -----------------------------------------------------------------------
    // 2. Free decreases live_bytes
    // -----------------------------------------------------------------------
    #[test]
    fn test_free_decreases_live_bytes() {
        let mut tracker = InferenceMemoryTracker::new();
        tracker.record(alloc(1, 1024, None));
        tracker.record(free(1));
        assert_eq!(tracker.live_bytes, 0);
    }

    // -----------------------------------------------------------------------
    // 3. peak_bytes tracks highest value
    // -----------------------------------------------------------------------
    #[test]
    fn test_peak_bytes_tracked() {
        let mut tracker = InferenceMemoryTracker::new();
        tracker.record(alloc(1, 2048, None));
        tracker.record(alloc(2, 1024, None));
        tracker.record(free(1));
        tracker.record(free(2));
        // Peak was 3072 (2048 + 1024); current live is 0.
        assert_eq!(tracker.peak_bytes, 3072);
        assert_eq!(tracker.live_bytes, 0);
    }

    // -----------------------------------------------------------------------
    // 4. Double-free is ignored (live_bytes does not go negative)
    // -----------------------------------------------------------------------
    #[test]
    fn test_double_free_ignored() {
        let mut tracker = InferenceMemoryTracker::new();
        tracker.record(alloc(1, 500, None));
        tracker.record(free(1));
        tracker.record(free(1)); // second free — should be a no-op
        assert_eq!(tracker.live_bytes, 0);
    }

    // -----------------------------------------------------------------------
    // 5. Free of unknown region is ignored
    // -----------------------------------------------------------------------
    #[test]
    fn test_free_unknown_region_ignored() {
        let mut tracker = InferenceMemoryTracker::new();
        tracker.record(free(99)); // region 99 was never allocated
        assert_eq!(tracker.live_bytes, 0);
        assert_eq!(tracker.stats().total_frees, 1); // event still counted
    }

    // -----------------------------------------------------------------------
    // 6. leaked_regions after session (some regions never freed)
    // -----------------------------------------------------------------------
    #[test]
    fn test_leaked_regions_after_session() {
        let mut tracker = InferenceMemoryTracker::new();
        tracker.record(alloc(1, 100, None));
        tracker.record(alloc(2, 200, None));
        tracker.record(free(1));
        // Region 2 is never freed.
        let leaked = tracker.leaked_regions();
        assert_eq!(leaked.len(), 1);
        assert_eq!(leaked[0].region_id, 2);
        assert_eq!(leaked[0].size_bytes, 200);
    }

    // -----------------------------------------------------------------------
    // 7. leaked_regions returns empty when all regions freed
    // -----------------------------------------------------------------------
    #[test]
    fn test_leaked_regions_empty_when_all_freed() {
        let mut tracker = InferenceMemoryTracker::new();
        tracker.record(alloc(1, 100, None));
        tracker.record(free(1));
        assert!(tracker.leaked_regions().is_empty());
    }

    // -----------------------------------------------------------------------
    // 8. memory_by_rule sums correctly
    // -----------------------------------------------------------------------
    #[test]
    fn test_memory_by_rule_sums_correctly() {
        let mut tracker = InferenceMemoryTracker::new();
        tracker.record(alloc(1, 400, Some(10)));
        tracker.record(alloc(2, 600, Some(10)));
        tracker.record(alloc(3, 200, Some(20)));
        assert_eq!(tracker.memory_by_rule(10), 1000);
        assert_eq!(tracker.memory_by_rule(20), 200);
        assert_eq!(tracker.memory_by_rule(99), 0);
    }

    // -----------------------------------------------------------------------
    // 9. memory_by_rule excludes freed regions
    // -----------------------------------------------------------------------
    #[test]
    fn test_memory_by_rule_excludes_freed() {
        let mut tracker = InferenceMemoryTracker::new();
        tracker.record(alloc(1, 400, Some(10)));
        tracker.record(alloc(2, 600, Some(10)));
        tracker.record(free(2));
        assert_eq!(tracker.memory_by_rule(10), 400);
    }

    // -----------------------------------------------------------------------
    // 10. Checkpoint is counted in stats
    // -----------------------------------------------------------------------
    #[test]
    fn test_checkpoint_counted() {
        let mut tracker = InferenceMemoryTracker::new();
        tracker.record(checkpoint("start"));
        tracker.record(checkpoint("middle"));
        assert_eq!(tracker.stats().total_checkpoints, 2);
    }

    // -----------------------------------------------------------------------
    // 11. snapshot.leaked_bytes reflects unfree'd regions
    // -----------------------------------------------------------------------
    #[test]
    fn test_snapshot_leaked_bytes() {
        let mut tracker = InferenceMemoryTracker::new();
        tracker.record(alloc(1, 100, None));
        tracker.record(alloc(2, 300, None));
        tracker.record(free(1));
        let snap = tracker.snapshot();
        assert_eq!(snap.leaked_bytes, 300);
    }

    // -----------------------------------------------------------------------
    // 12. snapshot.leaked_bytes is zero when nothing leaks
    // -----------------------------------------------------------------------
    #[test]
    fn test_snapshot_no_leaks() {
        let mut tracker = InferenceMemoryTracker::new();
        tracker.record(alloc(1, 100, None));
        tracker.record(free(1));
        let snap = tracker.snapshot();
        assert_eq!(snap.leaked_bytes, 0);
    }

    // -----------------------------------------------------------------------
    // 13. utilization calculation
    // -----------------------------------------------------------------------
    #[test]
    fn test_utilization() {
        let snap = MemorySnapshot {
            live_bytes: 512,
            live_regions: 1,
            peak_bytes: 1024,
            leaked_bytes: 512,
        };
        let util = snap.utilization(1024);
        assert!((util - 0.5).abs() < f64::EPSILON);
    }

    // -----------------------------------------------------------------------
    // 14. utilization with zero capacity returns 0.0
    // -----------------------------------------------------------------------
    #[test]
    fn test_utilization_zero_capacity() {
        let snap = MemorySnapshot {
            live_bytes: 512,
            live_regions: 1,
            peak_bytes: 512,
            leaked_bytes: 512,
        };
        assert_eq!(snap.utilization(0), 0.0);
    }

    // -----------------------------------------------------------------------
    // 15. reset clears all state
    // -----------------------------------------------------------------------
    #[test]
    fn test_reset_clears_state() {
        let mut tracker = InferenceMemoryTracker::new();
        tracker.record(alloc(1, 1024, Some(5)));
        tracker.record(checkpoint("cp"));
        tracker.reset();
        assert_eq!(tracker.live_bytes, 0);
        assert_eq!(tracker.peak_bytes, 0);
        assert!(tracker.events.is_empty());
        assert!(tracker.regions.is_empty());
        assert_eq!(tracker.stats().total_allocs, 0);
        assert_eq!(tracker.stats().total_frees, 0);
        assert_eq!(tracker.stats().total_checkpoints, 0);
    }

    // -----------------------------------------------------------------------
    // 16. stats.leak_count equals allocs minus frees
    // -----------------------------------------------------------------------
    #[test]
    fn test_stats_leak_count() {
        let mut tracker = InferenceMemoryTracker::new();
        tracker.record(alloc(1, 100, None));
        tracker.record(alloc(2, 200, None));
        tracker.record(alloc(3, 300, None));
        tracker.record(free(1));
        // 3 allocs, 1 free → leak_count = 2
        assert_eq!(tracker.stats().leak_count(), 2);
    }

    // -----------------------------------------------------------------------
    // 17. stats.leak_count does not underflow on more frees than allocs
    // -----------------------------------------------------------------------
    #[test]
    fn test_stats_leak_count_no_underflow() {
        let mut tracker = InferenceMemoryTracker::new();
        tracker.record(alloc(1, 100, None));
        tracker.record(free(1));
        tracker.record(free(1)); // double-free still counted in total_frees
                                 // total_allocs=1, total_frees=2 → saturating_sub → 0
        assert_eq!(tracker.stats().leak_count(), 0);
    }

    // -----------------------------------------------------------------------
    // 18. Full session: alloc → work → free → snapshot integrity
    // -----------------------------------------------------------------------
    #[test]
    fn test_full_session_snapshot_integrity() {
        let mut tracker = InferenceMemoryTracker::new();
        tracker.record(checkpoint("session_start"));
        tracker.record(alloc(10, 1000, Some(1)));
        tracker.record(alloc(11, 2000, Some(1)));
        tracker.record(alloc(12, 500, Some(2)));
        tracker.record(checkpoint("after_allocs"));
        tracker.record(free(10));
        tracker.record(free(12));
        // Region 11 is never freed.
        let snap = tracker.snapshot();
        assert_eq!(snap.live_bytes, 2000);
        assert_eq!(snap.live_regions, 1);
        assert_eq!(snap.peak_bytes, 3500);
        assert_eq!(snap.leaked_bytes, 2000);
        assert_eq!(tracker.stats().total_allocs, 3);
        assert_eq!(tracker.stats().total_frees, 2);
        assert_eq!(tracker.stats().total_checkpoints, 2);
        assert_eq!(tracker.stats().leak_count(), 1);
    }

    // -----------------------------------------------------------------------
    // 19. event log is complete and ordered
    // -----------------------------------------------------------------------
    #[test]
    fn test_event_log_ordered() {
        let mut tracker = InferenceMemoryTracker::new();
        tracker.record(checkpoint("a"));
        tracker.record(alloc(1, 100, None));
        tracker.record(free(1));
        assert_eq!(tracker.events.len(), 3);
        assert_eq!(tracker.events[0], checkpoint("a"));
        assert_eq!(tracker.events[1], alloc(1, 100, None));
        assert_eq!(tracker.events[2], free(1));
    }

    // -----------------------------------------------------------------------
    // 20. allocated_at_event index is set correctly
    // -----------------------------------------------------------------------
    #[test]
    fn test_allocated_at_event_index() {
        let mut tracker = InferenceMemoryTracker::new();
        tracker.record(checkpoint("init")); // event 0
        tracker.record(alloc(1, 100, None)); // event 1
        tracker.record(alloc(2, 200, None)); // event 2
        assert_eq!(tracker.regions[&1].allocated_at_event, 1);
        assert_eq!(tracker.regions[&2].allocated_at_event, 2);
    }
}
