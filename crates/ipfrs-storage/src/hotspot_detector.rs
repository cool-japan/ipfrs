//! Storage hotspot detector using sliding window access counting and exponential decay scoring.
//!
//! Identifies frequently accessed blocks for prefetching or tier promotion by
//! maintaining per-block access records and decaying scores over logical time ticks.

use std::collections::HashMap;

// ──────────────────────────────────────────────────────────────────────────────
// Public types
// ──────────────────────────────────────────────────────────────────────────────

/// The type of a storage block access operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AccessType {
    /// A standard read from the block.
    Read,
    /// A write to (or creation of) the block.
    Write,
    /// A speculative prefetch of the block.
    Prefetch,
}

/// A single access event for a specific block at a given logical tick.
#[derive(Clone, Debug)]
pub struct AccessEvent {
    /// Numeric identifier of the block being accessed.
    pub block_id: u64,
    /// Logical clock tick at which the access occurred.
    pub tick: u64,
    /// The kind of access that occurred.
    pub access_type: AccessType,
}

/// Computed hotness score for a single block.
#[derive(Clone, Debug)]
pub struct HotspotScore {
    /// Numeric identifier of the block.
    pub block_id: u64,
    /// Decay-adjusted hotness score; higher values indicate hotter blocks.
    pub score: f64,
    /// Number of accesses that fall within the current sliding window.
    pub recent_accesses: u32,
    /// Whether the block's score meets or exceeds the configured hotspot threshold.
    pub is_hotspot: bool,
}

/// Configuration for a [`StorageHotspotDetector`].
#[derive(Clone, Debug)]
pub struct DetectorConfig {
    /// Length of the sliding window in logical ticks.
    pub window_ticks: u64,
    /// Minimum score required to classify a block as a hotspot.
    pub hotspot_threshold: f64,
    /// Multiplicative decay applied per elapsed tick (must be in `(0, 1]`).
    pub decay_factor: f64,
    /// Score contribution added for each [`AccessType::Read`] event.
    pub read_weight: f64,
    /// Score contribution added for each [`AccessType::Write`] event.
    pub write_weight: f64,
    /// Score contribution added for each [`AccessType::Prefetch`] event.
    pub prefetch_weight: f64,
}

impl Default for DetectorConfig {
    fn default() -> Self {
        Self {
            window_ticks: 100,
            hotspot_threshold: 5.0,
            decay_factor: 0.95,
            read_weight: 1.0,
            write_weight: 1.5,
            prefetch_weight: 0.5,
        }
    }
}

/// Aggregate statistics reported by a [`StorageHotspotDetector`].
#[derive(Clone, Debug)]
pub struct HotspotDetectorStats {
    /// Total number of distinct blocks currently being tracked.
    pub total_blocks_tracked: usize,
    /// Number of those blocks currently classified as hotspots.
    pub hotspot_count: usize,
    /// Cumulative count of access events recorded since detector creation.
    pub total_events_recorded: u64,
}

/// Internal per-block access record (public to allow external inspection).
#[derive(Clone, Debug)]
pub struct BlockAccessRecord {
    /// Numeric identifier of the tracked block.
    pub block_id: u64,
    /// Current accumulated (possibly already decayed) score.
    pub score: f64,
    /// All access events still within the sliding window.
    pub events: Vec<AccessEvent>,
    /// The tick at which decay was last applied.
    pub last_decay_tick: u64,
}

// ──────────────────────────────────────────────────────────────────────────────
// Detector
// ──────────────────────────────────────────────────────────────────────────────

/// Detects storage hotspots using sliding-window access counting and exponential decay scoring.
///
/// # Algorithm
///
/// Each block maintains a floating-point score that grows whenever an access is
/// recorded (weighted by [`AccessType`]) and shrinks exponentially as logical
/// time elapses (`score *= decay_factor^ticks_elapsed`).  Events older than
/// `window_ticks` are purged to bound memory usage.  A block is classified as a
/// hotspot when its decay-adjusted score equals or exceeds `hotspot_threshold`.
pub struct StorageHotspotDetector {
    /// Per-block access records, keyed by block identifier.
    pub records: HashMap<u64, BlockAccessRecord>,
    /// Detector configuration.
    pub config: DetectorConfig,
    /// Total number of access events recorded since creation.
    pub total_events: u64,
}

impl StorageHotspotDetector {
    /// Create a new detector with the supplied configuration.
    pub fn new(config: DetectorConfig) -> Self {
        Self {
            records: HashMap::new(),
            config,
            total_events: 0,
        }
    }

    /// Record a block access event.
    ///
    /// The record for `event.block_id` is created on first access.  Decay is
    /// applied for every tick that has elapsed since the last recorded event,
    /// events older than the window are evicted, and the event weight is added
    /// to the score.
    pub fn record_access(&mut self, event: AccessEvent) {
        let block_id = event.block_id;
        let tick = event.tick;

        // Obtain or initialise the block record.
        let record = self
            .records
            .entry(block_id)
            .or_insert_with(|| BlockAccessRecord {
                block_id,
                score: 0.0,
                events: Vec::new(),
                last_decay_tick: tick,
            });

        // Apply exponential decay for elapsed ticks.
        if tick > record.last_decay_tick {
            let ticks_elapsed = tick - record.last_decay_tick;
            // Use manual loop for integer exponentiation to avoid precision pitfalls.
            let decay = pow_f64(self.config.decay_factor, ticks_elapsed);
            record.score *= decay;
            record.last_decay_tick = tick;
        }

        // Evict events that have fallen outside the sliding window.
        let window_start = tick.saturating_sub(self.config.window_ticks);
        record.events.retain(|e| e.tick >= window_start);

        // Accumulate score for this event.
        let weight = match event.access_type {
            AccessType::Read => self.config.read_weight,
            AccessType::Write => self.config.write_weight,
            AccessType::Prefetch => self.config.prefetch_weight,
        };
        record.score += weight;

        // Store the event.
        record.events.push(event);

        self.total_events += 1;
    }

    /// Return the decay-adjusted [`HotspotScore`] for `block_id` as of `current_tick`.
    ///
    /// Returns `None` if the block has never been tracked.
    pub fn hotspot_score(&self, block_id: u64, current_tick: u64) -> Option<HotspotScore> {
        let record = self.records.get(&block_id)?;

        let decayed_score = if current_tick > record.last_decay_tick {
            let ticks_elapsed = current_tick - record.last_decay_tick;
            record.score * pow_f64(self.config.decay_factor, ticks_elapsed)
        } else {
            record.score
        };

        let window_start = current_tick.saturating_sub(self.config.window_ticks);
        let recent_accesses = record
            .events
            .iter()
            .filter(|e| e.tick >= window_start)
            .count() as u32;

        let is_hotspot = decayed_score >= self.config.hotspot_threshold;

        Some(HotspotScore {
            block_id,
            score: decayed_score,
            recent_accesses,
            is_hotspot,
        })
    }

    /// Return all blocks currently classified as hotspots, sorted by score descending.
    pub fn hotspots(&self, current_tick: u64) -> Vec<HotspotScore> {
        let mut scores: Vec<HotspotScore> = self
            .records
            .keys()
            .filter_map(|&id| self.hotspot_score(id, current_tick))
            .filter(|s| s.is_hotspot)
            .collect();

        scores.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scores
    }

    /// Return the top `n` blocks ordered by decay-adjusted score descending.
    ///
    /// Non-hotspot blocks are included; the caller must check `is_hotspot` if
    /// only confirmed hotspots are required.
    pub fn top_n(&self, n: usize, current_tick: u64) -> Vec<HotspotScore> {
        let mut scores: Vec<HotspotScore> = self
            .records
            .keys()
            .filter_map(|&id| self.hotspot_score(id, current_tick))
            .collect();

        scores.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scores.truncate(n);
        scores
    }

    /// Remove all block records whose decay-adjusted score has fallen below `0.001`.
    ///
    /// Returns the number of records that were removed.
    pub fn evict_cold(&mut self, current_tick: u64) -> usize {
        const COLD_THRESHOLD: f64 = 0.001;

        let cold_ids: Vec<u64> = self
            .records
            .iter()
            .filter_map(|(&id, record)| {
                let decayed = if current_tick > record.last_decay_tick {
                    let elapsed = current_tick - record.last_decay_tick;
                    record.score * pow_f64(self.config.decay_factor, elapsed)
                } else {
                    record.score
                };
                if decayed < COLD_THRESHOLD {
                    Some(id)
                } else {
                    None
                }
            })
            .collect();

        let removed = cold_ids.len();
        for id in cold_ids {
            self.records.remove(&id);
        }
        removed
    }

    /// Return aggregate statistics for the current state of the detector.
    pub fn stats(&self, current_tick: u64) -> HotspotDetectorStats {
        let total_blocks_tracked = self.records.len();
        let hotspot_count = self
            .records
            .keys()
            .filter_map(|&id| self.hotspot_score(id, current_tick))
            .filter(|s| s.is_hotspot)
            .count();

        HotspotDetectorStats {
            total_blocks_tracked,
            hotspot_count,
            total_events_recorded: self.total_events,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Helper
// ──────────────────────────────────────────────────────────────────────────────

/// Raise `base` to an integer power using repeated multiplication.
///
/// Preferred over `f64::powi` for clarity and to avoid any sign-convention
/// surprises with very large exponents.
#[inline]
fn pow_f64(base: f64, exp: u64) -> f64 {
    // Fast path for common cases.
    match exp {
        0 => 1.0,
        1 => base,
        _ => {
            let mut result = 1.0_f64;
            let mut b = base;
            let mut e = exp;
            // Binary exponentiation.
            while e > 0 {
                if e & 1 == 1 {
                    result *= b;
                }
                b *= b;
                e >>= 1;
            }
            result
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_detector() -> StorageHotspotDetector {
        StorageHotspotDetector::new(DetectorConfig::default())
    }

    fn make_event(block_id: u64, tick: u64, access_type: AccessType) -> AccessEvent {
        AccessEvent {
            block_id,
            tick,
            access_type,
        }
    }

    // ── 1. record_access creates a record ─────────────────────────────────────

    #[test]
    fn test_record_creates_entry() {
        let mut det = default_detector();
        assert!(det.records.is_empty());
        det.record_access(make_event(42, 0, AccessType::Read));
        assert_eq!(det.records.len(), 1);
        assert!(det.records.contains_key(&42));
    }

    // ── 2. Score increases with accesses ──────────────────────────────────────

    #[test]
    fn test_score_increases_with_reads() {
        let mut det = default_detector();
        det.record_access(make_event(1, 0, AccessType::Read));
        let s1 = det.hotspot_score(1, 0).unwrap().score;
        det.record_access(make_event(1, 0, AccessType::Read));
        let s2 = det.hotspot_score(1, 0).unwrap().score;
        assert!(s2 > s1, "score should grow: {s2} > {s1}");
    }

    #[test]
    fn test_write_weight_higher_than_read() {
        let mut det = default_detector();
        det.record_access(make_event(1, 0, AccessType::Read));
        let read_score = det.hotspot_score(1, 0).unwrap().score;

        let mut det2 = default_detector();
        det2.record_access(make_event(2, 0, AccessType::Write));
        let write_score = det2.hotspot_score(2, 0).unwrap().score;

        assert!(
            write_score > read_score,
            "write ({write_score}) > read ({read_score})"
        );
    }

    #[test]
    fn test_prefetch_weight_lower_than_read() {
        let mut det = default_detector();
        det.record_access(make_event(1, 0, AccessType::Read));
        let read_score = det.hotspot_score(1, 0).unwrap().score;

        let mut det2 = default_detector();
        det2.record_access(make_event(2, 0, AccessType::Prefetch));
        let prefetch_score = det2.hotspot_score(2, 0).unwrap().score;

        assert!(
            prefetch_score < read_score,
            "prefetch ({prefetch_score}) < read ({read_score})"
        );
    }

    // ── 3. Explicit weight values ──────────────────────────────────────────────

    #[test]
    fn test_read_weight_exact() {
        let mut det = default_detector();
        det.record_access(make_event(5, 10, AccessType::Read));
        let score = det.hotspot_score(5, 10).unwrap().score;
        assert!(
            (score - 1.0).abs() < 1e-10,
            "read weight should be 1.0, got {score}"
        );
    }

    #[test]
    fn test_write_weight_exact() {
        let mut det = default_detector();
        det.record_access(make_event(5, 10, AccessType::Write));
        let score = det.hotspot_score(5, 10).unwrap().score;
        assert!(
            (score - 1.5).abs() < 1e-10,
            "write weight should be 1.5, got {score}"
        );
    }

    #[test]
    fn test_prefetch_weight_exact() {
        let mut det = default_detector();
        det.record_access(make_event(5, 10, AccessType::Prefetch));
        let score = det.hotspot_score(5, 10).unwrap().score;
        assert!(
            (score - 0.5).abs() < 1e-10,
            "prefetch weight should be 0.5, got {score}"
        );
    }

    // ── 4. Decay applied based on ticks elapsed ────────────────────────────────

    #[test]
    fn test_decay_reduces_score_over_time() {
        let mut det = default_detector();
        det.record_access(make_event(1, 0, AccessType::Read));
        let score_at_0 = det.hotspot_score(1, 0).unwrap().score;
        let score_at_10 = det.hotspot_score(1, 10).unwrap().score;
        assert!(
            score_at_10 < score_at_0,
            "score should decay: {score_at_10} < {score_at_0}"
        );
    }

    #[test]
    fn test_decay_factor_applied_correctly() {
        let config = DetectorConfig {
            decay_factor: 0.5,
            ..Default::default()
        };
        let mut det = StorageHotspotDetector::new(config);
        det.record_access(make_event(1, 0, AccessType::Read)); // score = 1.0
                                                               // After 3 ticks: 1.0 * 0.5^3 = 0.125
        let score = det.hotspot_score(1, 3).unwrap().score;
        assert!((score - 0.125).abs() < 1e-10, "expected 0.125, got {score}");
    }

    #[test]
    fn test_decay_applied_on_new_event() {
        let config = DetectorConfig {
            decay_factor: 0.5,
            ..Default::default()
        };
        let mut det = StorageHotspotDetector::new(config);
        det.record_access(make_event(1, 0, AccessType::Read)); // score = 1.0
                                                               // Record again at tick 2: decay first (1.0 * 0.25 = 0.25), then add 1.0 → 1.25
        det.record_access(make_event(1, 2, AccessType::Read));
        let score = det.hotspot_score(1, 2).unwrap().score;
        assert!((score - 1.25).abs() < 1e-10, "expected 1.25, got {score}");
    }

    #[test]
    fn test_no_decay_when_tick_unchanged() {
        let mut det = default_detector();
        det.record_access(make_event(7, 5, AccessType::Read));
        det.record_access(make_event(7, 5, AccessType::Read));
        let score = det.hotspot_score(7, 5).unwrap().score;
        // Two reads at same tick: 1.0 + 1.0 = 2.0
        assert!((score - 2.0).abs() < 1e-10, "expected 2.0, got {score}");
    }

    // ── 5. Window eviction removes old events ─────────────────────────────────

    #[test]
    fn test_window_eviction_removes_old_events() {
        let config = DetectorConfig {
            window_ticks: 10,
            ..Default::default()
        };
        let mut det = StorageHotspotDetector::new(config);
        det.record_access(make_event(1, 0, AccessType::Read));
        det.record_access(make_event(1, 5, AccessType::Read));
        // Trigger eviction by recording at tick 15 (window=[5,15], tick 0 falls out)
        det.record_access(make_event(1, 15, AccessType::Read));
        let record = &det.records[&1];
        // Only events at tick >= 15-10=5 remain, i.e. ticks 5 and 15.
        assert_eq!(record.events.len(), 2, "events at tick 0 should be evicted");
    }

    #[test]
    fn test_recent_accesses_counts_window_only() {
        let config = DetectorConfig {
            window_ticks: 10,
            ..Default::default()
        };
        let mut det = StorageHotspotDetector::new(config);
        det.record_access(make_event(1, 0, AccessType::Read));
        det.record_access(make_event(1, 1, AccessType::Read));
        det.record_access(make_event(1, 5, AccessType::Read));
        // At tick 15: window=[5,15], so only the event at tick 5 is recent.
        let hs = det.hotspot_score(1, 15).unwrap();
        assert_eq!(hs.recent_accesses, 1);
    }

    // ── 6. hotspot_score returns None for untracked ────────────────────────────

    #[test]
    fn test_hotspot_score_none_for_unknown() {
        let det = default_detector();
        assert!(det.hotspot_score(999, 0).is_none());
    }

    // ── 7. is_hotspot based on threshold ──────────────────────────────────────

    #[test]
    fn test_is_hotspot_false_below_threshold() {
        let mut det = default_detector(); // threshold = 5.0
        det.record_access(make_event(1, 0, AccessType::Read)); // score = 1.0
        let hs = det.hotspot_score(1, 0).unwrap();
        assert!(!hs.is_hotspot);
    }

    #[test]
    fn test_is_hotspot_true_above_threshold() {
        let mut det = default_detector(); // threshold = 5.0
        for _ in 0..6 {
            det.record_access(make_event(1, 0, AccessType::Read)); // +6.0
        }
        let hs = det.hotspot_score(1, 0).unwrap();
        assert!(hs.is_hotspot, "score {} should be >= 5.0", hs.score);
    }

    #[test]
    fn test_is_hotspot_exactly_at_threshold() {
        let config = DetectorConfig {
            hotspot_threshold: 3.0,
            read_weight: 3.0,
            ..Default::default()
        };
        let mut det = StorageHotspotDetector::new(config);
        det.record_access(make_event(1, 0, AccessType::Read)); // score = 3.0
        let hs = det.hotspot_score(1, 0).unwrap();
        assert!(hs.is_hotspot, "score == threshold should be hotspot");
    }

    // ── 8. hotspots returns sorted list ───────────────────────────────────────

    #[test]
    fn test_hotspots_sorted_descending() {
        let mut det = default_detector(); // threshold = 5.0
                                          // block 10: 8 reads → 8.0
        for _ in 0..8 {
            det.record_access(make_event(10, 0, AccessType::Read));
        }
        // block 20: 6 reads → 6.0
        for _ in 0..6 {
            det.record_access(make_event(20, 0, AccessType::Read));
        }
        // block 30: 3 reads → 3.0 (not a hotspot)
        for _ in 0..3 {
            det.record_access(make_event(30, 0, AccessType::Read));
        }
        let hs = det.hotspots(0);
        assert_eq!(hs.len(), 2, "only blocks 10 and 20 should be hotspots");
        assert_eq!(hs[0].block_id, 10);
        assert_eq!(hs[1].block_id, 20);
        assert!(hs[0].score >= hs[1].score);
    }

    #[test]
    fn test_hotspots_empty_when_none_qualify() {
        let mut det = default_detector();
        det.record_access(make_event(1, 0, AccessType::Read));
        let hs = det.hotspots(0);
        assert!(hs.is_empty());
    }

    // ── 9. top_n returns top n blocks ─────────────────────────────────────────

    #[test]
    fn test_top_n_returns_correct_count() {
        let mut det = default_detector();
        for id in 0..10_u64 {
            for _ in 0..((id + 1) as usize) {
                det.record_access(make_event(id, 0, AccessType::Read));
            }
        }
        let top = det.top_n(3, 0);
        assert_eq!(top.len(), 3);
    }

    #[test]
    fn test_top_n_sorted_descending() {
        let mut det = default_detector();
        for id in 0..5_u64 {
            for _ in 0..((id + 1) as usize) {
                det.record_access(make_event(id, 0, AccessType::Read));
            }
        }
        let top = det.top_n(5, 0);
        for w in top.windows(2) {
            assert!(w[0].score >= w[1].score);
        }
    }

    #[test]
    fn test_top_n_fewer_than_n_blocks() {
        let mut det = default_detector();
        det.record_access(make_event(1, 0, AccessType::Read));
        det.record_access(make_event(2, 0, AccessType::Read));
        let top = det.top_n(10, 0);
        assert_eq!(top.len(), 2);
    }

    #[test]
    fn test_top_n_zero_returns_empty() {
        let mut det = default_detector();
        det.record_access(make_event(1, 0, AccessType::Read));
        let top = det.top_n(0, 0);
        assert!(top.is_empty());
    }

    // ── 10. evict_cold removes cold blocks ───────────────────────────────────

    #[test]
    fn test_evict_cold_removes_stale_blocks() {
        let config = DetectorConfig {
            decay_factor: 0.5,
            ..Default::default()
        };
        let mut det = StorageHotspotDetector::new(config);
        det.record_access(make_event(1, 0, AccessType::Read)); // score = 1.0
                                                               // After 20 ticks at 0.5 decay: 1.0 * 0.5^20 ≈ 9.5e-7 < 0.001
        let removed = det.evict_cold(20);
        assert_eq!(removed, 1);
        assert!(det.records.is_empty());
    }

    #[test]
    fn test_evict_cold_keeps_warm_blocks() {
        let mut det = default_detector();
        for _ in 0..10 {
            det.record_access(make_event(1, 0, AccessType::Read)); // score = 10.0
        }
        let removed = det.evict_cold(0);
        assert_eq!(removed, 0);
        assert_eq!(det.records.len(), 1);
    }

    #[test]
    fn test_evict_cold_returns_count() {
        let config = DetectorConfig {
            decay_factor: 0.1,
            ..Default::default()
        };
        let mut det = StorageHotspotDetector::new(config);
        det.record_access(make_event(1, 0, AccessType::Read));
        det.record_access(make_event(2, 0, AccessType::Read));
        det.record_access(make_event(3, 0, AccessType::Read));
        // 0.1^10 < 0.001 for all three
        let removed = det.evict_cold(10);
        assert_eq!(removed, 3);
    }

    // ── 11. stats ─────────────────────────────────────────────────────────────

    #[test]
    fn test_stats_total_events() {
        let mut det = default_detector();
        det.record_access(make_event(1, 0, AccessType::Read));
        det.record_access(make_event(2, 0, AccessType::Write));
        det.record_access(make_event(1, 1, AccessType::Read));
        let s = det.stats(1);
        assert_eq!(s.total_events_recorded, 3);
    }

    #[test]
    fn test_stats_blocks_tracked() {
        let mut det = default_detector();
        det.record_access(make_event(1, 0, AccessType::Read));
        det.record_access(make_event(2, 0, AccessType::Read));
        det.record_access(make_event(1, 0, AccessType::Read)); // duplicate block
        let s = det.stats(0);
        assert_eq!(s.total_blocks_tracked, 2);
    }

    #[test]
    fn test_stats_hotspot_count() {
        let mut det = default_detector(); // threshold = 5.0
        for _ in 0..6 {
            det.record_access(make_event(1, 0, AccessType::Read));
        }
        det.record_access(make_event(2, 0, AccessType::Read)); // score = 1.0, cold
        let s = det.stats(0);
        assert_eq!(s.hotspot_count, 1);
    }

    // ── 12. Custom weights ────────────────────────────────────────────────────

    #[test]
    fn test_custom_weights_affect_score() {
        let config = DetectorConfig {
            read_weight: 2.0,
            write_weight: 3.0,
            prefetch_weight: 0.1,
            ..Default::default()
        };
        let mut det = StorageHotspotDetector::new(config);
        det.record_access(make_event(1, 0, AccessType::Read));
        det.record_access(make_event(2, 0, AccessType::Write));
        det.record_access(make_event(3, 0, AccessType::Prefetch));

        let r = det.hotspot_score(1, 0).unwrap().score;
        let w = det.hotspot_score(2, 0).unwrap().score;
        let p = det.hotspot_score(3, 0).unwrap().score;

        assert!((r - 2.0).abs() < 1e-10, "read: {r}");
        assert!((w - 3.0).abs() < 1e-10, "write: {w}");
        assert!((p - 0.1).abs() < 1e-10, "prefetch: {p}");
    }

    // ── 13. pow_f64 helper ────────────────────────────────────────────────────

    #[test]
    fn test_pow_f64_zero_exponent() {
        assert!((super::pow_f64(0.95, 0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_pow_f64_one_exponent() {
        assert!((super::pow_f64(0.95, 1) - 0.95).abs() < 1e-12);
    }

    #[test]
    fn test_pow_f64_large_exponent() {
        let result = super::pow_f64(0.5, 10);
        assert!((result - (0.5_f64).powi(10)).abs() < 1e-12);
    }
}
