//! Storage Heatmap Tracker
//!
//! Tracks per-block access heat scores over time using exponential decay,
//! producing a ranked heatmap for prefetch and cache prioritization.

use std::collections::HashMap;

/// Heat classification bucket for a block.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum HeatBucket {
    /// score < 1.0
    Cold = 0,
    /// 1.0 <= score < 10.0
    Warm = 1,
    /// 10.0 <= score < 100.0
    Hot = 2,
    /// score >= 100.0
    Scorching = 3,
}

impl HeatBucket {
    /// Classify a heat score into a bucket.
    pub fn from_score(score: f64) -> HeatBucket {
        if score >= 100.0 {
            HeatBucket::Scorching
        } else if score >= 10.0 {
            HeatBucket::Hot
        } else if score >= 1.0 {
            HeatBucket::Warm
        } else {
            HeatBucket::Cold
        }
    }
}

/// A single block's heat tracking record.
#[derive(Clone, Debug)]
pub struct HeatEntry {
    /// Content identifier for the block.
    pub cid: String,
    /// Accumulated heat score (decays over time).
    pub score: f64,
    /// Total number of times this block was accessed.
    pub access_count: u64,
    /// The tick at which this block was last accessed.
    pub last_accessed_tick: u64,
}

impl HeatEntry {
    /// Return the heat bucket classification for this entry's current score.
    pub fn bucket(&self) -> HeatBucket {
        HeatBucket::from_score(self.score)
    }

    /// Apply exponential decay for `ticks_elapsed` ticks.
    ///
    /// `score *= decay_factor.powf(ticks_elapsed as f64)`
    pub fn decay(&mut self, ticks_elapsed: u64, decay_factor: f64) {
        self.score *= decay_factor.powf(ticks_elapsed as f64);
    }
}

/// Aggregate statistics over all tracked entries.
#[derive(Clone, Debug, PartialEq)]
pub struct HeatmapStats {
    /// Total number of tracked entries.
    pub total_entries: usize,
    /// Number of entries in the Scorching bucket.
    pub scorching: usize,
    /// Number of entries in the Hot bucket.
    pub hot: usize,
    /// Number of entries in the Warm bucket.
    pub warm: usize,
    /// Number of entries in the Cold bucket.
    pub cold: usize,
    /// Sum of access_count across all entries.
    pub total_accesses: u64,
}

impl HeatmapStats {
    /// Fraction of total entries represented by the top-k.
    ///
    /// `top_k / total_entries.max(1) as f64`
    pub fn coverage_ratio(&self, top_k: usize) -> f64 {
        top_k as f64 / self.total_entries.max(1) as f64
    }
}

/// Tracks per-block access heat scores over time using exponential decay.
pub struct StorageHeatmapTracker {
    /// All tracked entries, keyed by CID.
    pub entries: HashMap<String, HeatEntry>,
    /// Per-tick decay factor (e.g. 0.9 means score *= 0.9 each tick).
    pub decay_factor: f64,
    /// Current logical time tick.
    pub current_tick: u64,
}

impl StorageHeatmapTracker {
    /// Create a new tracker with the given per-tick decay factor.
    pub fn new(decay_factor: f64) -> Self {
        Self {
            entries: HashMap::new(),
            decay_factor,
            current_tick: 0,
        }
    }

    /// Record an access to `cid`, adding `heat_gain` to its score.
    ///
    /// If the entry already exists, lazy decay is applied for the ticks that
    /// elapsed since the last access before adding the gain. If the entry is
    /// new it is created with `score = heat_gain`.
    pub fn record_access(&mut self, cid: &str, heat_gain: f64) {
        let current_tick = self.current_tick;
        let decay_factor = self.decay_factor;

        if let Some(entry) = self.entries.get_mut(cid) {
            let ticks_elapsed = current_tick.saturating_sub(entry.last_accessed_tick);
            entry.decay(ticks_elapsed, decay_factor);
            entry.score += heat_gain;
            entry.access_count += 1;
            entry.last_accessed_tick = current_tick;
        } else {
            self.entries.insert(
                cid.to_string(),
                HeatEntry {
                    cid: cid.to_string(),
                    score: heat_gain,
                    access_count: 1,
                    last_accessed_tick: current_tick,
                },
            );
        }
    }

    /// Advance the logical clock by one tick.
    ///
    /// Decay is applied lazily on the next access; use `decay_all` to force
    /// an eager decay pass across all entries.
    pub fn advance_tick(&mut self) {
        self.current_tick += 1;
    }

    /// Return the top-`k` entries by score, in descending order.
    pub fn top_k(&self, k: usize) -> Vec<&HeatEntry> {
        let mut entries: Vec<&HeatEntry> = self.entries.values().collect();
        entries.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        entries.truncate(k);
        entries
    }

    /// Remove all entries whose score is strictly less than `min_score`.
    ///
    /// Returns the number of entries removed.
    pub fn evict_cold(&mut self, min_score: f64) -> usize {
        let before = self.entries.len();
        self.entries.retain(|_, entry| entry.score >= min_score);
        before - self.entries.len()
    }

    /// Compute aggregate statistics over all current entries.
    pub fn stats(&self) -> HeatmapStats {
        let mut scorching = 0usize;
        let mut hot = 0usize;
        let mut warm = 0usize;
        let mut cold = 0usize;
        let mut total_accesses = 0u64;

        for entry in self.entries.values() {
            total_accesses += entry.access_count;
            match entry.bucket() {
                HeatBucket::Scorching => scorching += 1,
                HeatBucket::Hot => hot += 1,
                HeatBucket::Warm => warm += 1,
                HeatBucket::Cold => cold += 1,
            }
        }

        HeatmapStats {
            total_entries: self.entries.len(),
            scorching,
            hot,
            warm,
            cold,
            total_accesses,
        }
    }

    /// Eagerly apply decay to every entry relative to `current_tick`.
    ///
    /// After this call, every entry's `last_accessed_tick` is updated to
    /// `current_tick` so that a second call in the same tick is a no-op.
    pub fn decay_all(&mut self) {
        let current_tick = self.current_tick;
        let decay_factor = self.decay_factor;

        for entry in self.entries.values_mut() {
            let ticks_elapsed = current_tick.saturating_sub(entry.last_accessed_tick);
            entry.decay(ticks_elapsed, decay_factor);
            entry.last_accessed_tick = current_tick;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── HeatBucket::from_score ────────────────────────────────────────────────

    #[test]
    fn test_heat_bucket_cold_below_one() {
        assert_eq!(HeatBucket::from_score(0.0), HeatBucket::Cold);
        assert_eq!(HeatBucket::from_score(0.99), HeatBucket::Cold);
    }

    #[test]
    fn test_heat_bucket_warm_one_to_ten() {
        assert_eq!(HeatBucket::from_score(1.0), HeatBucket::Warm);
        assert_eq!(HeatBucket::from_score(5.0), HeatBucket::Warm);
        assert_eq!(HeatBucket::from_score(9.999), HeatBucket::Warm);
    }

    #[test]
    fn test_heat_bucket_hot_ten_to_hundred() {
        assert_eq!(HeatBucket::from_score(10.0), HeatBucket::Hot);
        assert_eq!(HeatBucket::from_score(50.0), HeatBucket::Hot);
        assert_eq!(HeatBucket::from_score(99.99), HeatBucket::Hot);
    }

    #[test]
    fn test_heat_bucket_scorching_at_and_above_hundred() {
        assert_eq!(HeatBucket::from_score(100.0), HeatBucket::Scorching);
        assert_eq!(HeatBucket::from_score(1_000.0), HeatBucket::Scorching);
    }

    // ── HeatEntry ────────────────────────────────────────────────────────────

    #[test]
    fn test_heat_entry_bucket_delegates_to_from_score() {
        let entry = HeatEntry {
            cid: "bafybei1".to_string(),
            score: 15.0,
            access_count: 3,
            last_accessed_tick: 0,
        };
        assert_eq!(entry.bucket(), HeatBucket::Hot);
    }

    #[test]
    fn test_heat_entry_decay_reduces_score() {
        let mut entry = HeatEntry {
            cid: "bafybei2".to_string(),
            score: 100.0,
            access_count: 1,
            last_accessed_tick: 0,
        };
        entry.decay(1, 0.9);
        // 100.0 * 0.9^1 = 90.0
        assert!((entry.score - 90.0).abs() < 1e-9);
    }

    #[test]
    fn test_heat_entry_decay_multiple_ticks() {
        let mut entry = HeatEntry {
            cid: "bafybei3".to_string(),
            score: 100.0,
            access_count: 1,
            last_accessed_tick: 0,
        };
        entry.decay(10, 0.9);
        let expected = 100.0_f64 * 0.9_f64.powf(10.0);
        assert!((entry.score - expected).abs() < 1e-9);
    }

    // ── StorageHeatmapTracker::record_access ─────────────────────────────────

    #[test]
    fn test_record_access_creates_entry() {
        let mut tracker = StorageHeatmapTracker::new(0.9);
        tracker.record_access("cid-a", 5.0);
        let entry = tracker.entries.get("cid-a").expect("entry must exist");
        assert!((entry.score - 5.0).abs() < 1e-9);
        assert_eq!(entry.access_count, 1);
        assert_eq!(entry.last_accessed_tick, 0);
    }

    #[test]
    fn test_record_access_heat_accumulates_same_tick() {
        let mut tracker = StorageHeatmapTracker::new(0.9);
        tracker.record_access("cid-b", 3.0);
        tracker.record_access("cid-b", 7.0);
        let entry = tracker.entries.get("cid-b").expect("entry must exist");
        // second access: score was 3.0, decayed 0 ticks → still 3.0, then +7.0 = 10.0
        assert!((entry.score - 10.0).abs() < 1e-9);
        assert_eq!(entry.access_count, 2);
    }

    #[test]
    fn test_record_access_lazy_decay_applied_on_next_access() {
        let mut tracker = StorageHeatmapTracker::new(0.9);
        tracker.record_access("cid-c", 100.0);
        // advance 2 ticks without touching the entry
        tracker.advance_tick();
        tracker.advance_tick();
        tracker.record_access("cid-c", 0.0);
        let entry = tracker.entries.get("cid-c").expect("entry must exist");
        let expected = 100.0_f64 * 0.9_f64.powf(2.0);
        assert!((entry.score - expected).abs() < 1e-9);
        assert_eq!(entry.access_count, 2);
        assert_eq!(entry.last_accessed_tick, 2);
    }

    // ── StorageHeatmapTracker::advance_tick ──────────────────────────────────

    #[test]
    fn test_advance_tick_increments_current_tick() {
        let mut tracker = StorageHeatmapTracker::new(0.9);
        assert_eq!(tracker.current_tick, 0);
        tracker.advance_tick();
        assert_eq!(tracker.current_tick, 1);
        tracker.advance_tick();
        assert_eq!(tracker.current_tick, 2);
    }

    // ── StorageHeatmapTracker::top_k ─────────────────────────────────────────

    #[test]
    fn test_top_k_ordered_descending() {
        let mut tracker = StorageHeatmapTracker::new(0.9);
        tracker.record_access("low", 1.0);
        tracker.record_access("mid", 50.0);
        tracker.record_access("high", 200.0);
        let top = tracker.top_k(3);
        assert_eq!(top.len(), 3);
        assert_eq!(top[0].cid, "high");
        assert_eq!(top[1].cid, "mid");
        assert_eq!(top[2].cid, "low");
    }

    #[test]
    fn test_top_k_fewer_than_k_entries() {
        let mut tracker = StorageHeatmapTracker::new(0.9);
        tracker.record_access("only", 10.0);
        let top = tracker.top_k(5);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_top_k_returns_exactly_k() {
        let mut tracker = StorageHeatmapTracker::new(0.9);
        for i in 0..10u64 {
            tracker.record_access(&format!("cid-{i}"), i as f64);
        }
        let top = tracker.top_k(3);
        assert_eq!(top.len(), 3);
    }

    // ── StorageHeatmapTracker::evict_cold ────────────────────────────────────

    #[test]
    fn test_evict_cold_removes_low_score_entries() {
        let mut tracker = StorageHeatmapTracker::new(0.9);
        tracker.record_access("keep", 10.0);
        tracker.record_access("remove", 0.5);
        let removed = tracker.evict_cold(1.0);
        assert_eq!(removed, 1);
        assert!(tracker.entries.contains_key("keep"));
        assert!(!tracker.entries.contains_key("remove"));
    }

    #[test]
    fn test_evict_cold_keeps_exactly_at_threshold() {
        let mut tracker = StorageHeatmapTracker::new(0.9);
        tracker.record_access("exact", 1.0);
        let removed = tracker.evict_cold(1.0);
        assert_eq!(removed, 0);
        assert!(tracker.entries.contains_key("exact"));
    }

    #[test]
    fn test_new_entry_after_eviction() {
        let mut tracker = StorageHeatmapTracker::new(0.9);
        tracker.record_access("cid-x", 0.1);
        tracker.evict_cold(1.0);
        assert!(!tracker.entries.contains_key("cid-x"));
        // re-add after eviction
        tracker.record_access("cid-x", 5.0);
        let entry = tracker
            .entries
            .get("cid-x")
            .expect("entry must exist after re-add");
        assert!((entry.score - 5.0).abs() < 1e-9);
        assert_eq!(entry.access_count, 1);
    }

    // ── StorageHeatmapTracker::stats ─────────────────────────────────────────

    #[test]
    fn test_stats_counts_per_bucket() {
        let mut tracker = StorageHeatmapTracker::new(0.9);
        tracker.record_access("scorching", 200.0);
        tracker.record_access("hot", 50.0);
        tracker.record_access("warm", 5.0);
        tracker.record_access("cold", 0.1);
        let stats = tracker.stats();
        assert_eq!(stats.total_entries, 4);
        assert_eq!(stats.scorching, 1);
        assert_eq!(stats.hot, 1);
        assert_eq!(stats.warm, 1);
        assert_eq!(stats.cold, 1);
        assert_eq!(stats.total_accesses, 4);
    }

    // ── HeatmapStats::coverage_ratio ─────────────────────────────────────────

    #[test]
    fn test_coverage_ratio_basic() {
        let stats = HeatmapStats {
            total_entries: 10,
            scorching: 1,
            hot: 2,
            warm: 3,
            cold: 4,
            total_accesses: 20,
        };
        let ratio = stats.coverage_ratio(3);
        assert!((ratio - 0.3).abs() < 1e-9);
    }

    #[test]
    fn test_coverage_ratio_zero_entries_does_not_divide_by_zero() {
        let stats = HeatmapStats {
            total_entries: 0,
            scorching: 0,
            hot: 0,
            warm: 0,
            cold: 0,
            total_accesses: 0,
        };
        // total_entries.max(1) == 1, so top_k / 1
        let ratio = stats.coverage_ratio(5);
        assert!((ratio - 5.0).abs() < 1e-9);
    }

    // ── StorageHeatmapTracker::decay_all ─────────────────────────────────────

    #[test]
    fn test_decay_all_applies_decay_to_all_entries() {
        let mut tracker = StorageHeatmapTracker::new(0.9);
        tracker.record_access("a", 100.0);
        tracker.record_access("b", 50.0);
        tracker.advance_tick();
        tracker.advance_tick();
        tracker.decay_all();
        let a = tracker.entries.get("a").expect("a must exist");
        let b = tracker.entries.get("b").expect("b must exist");
        let expected_a = 100.0_f64 * 0.9_f64.powf(2.0);
        let expected_b = 50.0_f64 * 0.9_f64.powf(2.0);
        assert!((a.score - expected_a).abs() < 1e-9);
        assert!((b.score - expected_b).abs() < 1e-9);
        // last_accessed_tick must be updated so a second decay_all is a no-op
        assert_eq!(a.last_accessed_tick, 2);
        assert_eq!(b.last_accessed_tick, 2);
    }

    #[test]
    fn test_decay_all_second_call_same_tick_is_noop() {
        let mut tracker = StorageHeatmapTracker::new(0.9);
        tracker.record_access("a", 100.0);
        tracker.advance_tick();
        tracker.decay_all();
        let score_after_first = tracker.entries["a"].score;
        tracker.decay_all();
        let score_after_second = tracker.entries["a"].score;
        assert!((score_after_first - score_after_second).abs() < 1e-9);
    }

    // ── HeatBucket ordering ───────────────────────────────────────────────────

    #[test]
    fn test_heat_bucket_ordering() {
        assert!(HeatBucket::Cold < HeatBucket::Warm);
        assert!(HeatBucket::Warm < HeatBucket::Hot);
        assert!(HeatBucket::Hot < HeatBucket::Scorching);
    }
}
