//! Storage access predictor for proactive prefetching and cache warming.
//!
//! Analyzes historical block access sequences to predict future access patterns,
//! enabling intelligent prefetch scheduling and cache management decisions.

use std::collections::HashMap;

// ──────────────────────────────────────────────────────────────────────────────
// Public types
// ──────────────────────────────────────────────────────────────────────────────

/// Detected access pattern for a content-addressed block.
#[derive(Clone, Debug, PartialEq)]
pub enum AccessPattern {
    /// Accesses follow a predictable, monotonically increasing sequence.
    Sequential,
    /// Block is accessed every `interval_ticks` ticks.
    Repeated { interval_ticks: u64 },
    /// Multiple accesses occur in short bursts (inter-access gap ≤ 5 ticks).
    Bursty { burst_size: usize },
    /// No detectable temporal pattern.
    Random,
    /// Access frequency is decreasing (intervals are strictly growing).
    Cooling,
}

/// A single block access event recorded by the predictor.
#[derive(Clone, Debug)]
pub struct AccessEvent {
    /// Content identifier of the block.
    pub cid: String,
    /// Logical clock tick at which the access occurred.
    pub tick: u64,
    /// Size of the block in bytes.
    pub size_bytes: u64,
}

/// Result of a pattern prediction for a single CID.
#[derive(Clone, Debug)]
pub struct PredictionResult {
    /// Content identifier of the block.
    pub cid: String,
    /// Detected access pattern.
    pub pattern: AccessPattern,
    /// Predicted tick of the next access; `None` for `Random` and `Cooling`.
    pub next_access_tick: Option<u64>,
    /// Confidence score in the range `[0.0, 1.0]`.
    pub confidence: f32,
}

/// Aggregate statistics emitted by the predictor.
#[derive(Clone, Debug, Default)]
pub struct PredictorStats {
    /// Total number of recorded access events across all CIDs.
    pub total_events: u64,
    /// Number of distinct CIDs currently tracked.
    pub unique_cids: usize,
    /// Number of CIDs whose last prediction was `Sequential`.
    pub sequential_count: usize,
    /// Number of CIDs whose last prediction was `Repeated`.
    pub repeated_count: usize,
    /// Number of CIDs whose last prediction was `Random`.
    pub random_count: usize,
}

// ──────────────────────────────────────────────────────────────────────────────
// StorageAccessPredictor
// ──────────────────────────────────────────────────────────────────────────────

/// Maximum number of access events retained per CID.
const MAX_HISTORY: usize = 20;

/// Predicts future block access patterns from historical access sequences.
///
/// Events are stored in a bounded per-CID ring (capped at `MAX_HISTORY`).
/// The predictor derives an [`AccessPattern`] from the inter-access intervals
/// and computes a confidence score for each prediction.
pub struct StorageAccessPredictor {
    /// Per-CID ordered list of access events (oldest first).
    pub history: HashMap<String, Vec<AccessEvent>>,
}

impl StorageAccessPredictor {
    /// Create a new, empty predictor.
    pub fn new() -> Self {
        Self {
            history: HashMap::new(),
        }
    }

    /// Record a new access event.
    ///
    /// Events are appended to the per-CID history.  When the history exceeds
    /// `MAX_HISTORY` entries the oldest entry (index 0) is evicted.
    pub fn record(&mut self, event: AccessEvent) {
        let entries = self.history.entry(event.cid.clone()).or_default();
        entries.push(event);
        if entries.len() > MAX_HISTORY {
            entries.remove(0);
        }
    }

    /// Predict the access pattern for a given CID.
    ///
    /// Returns a [`PredictionResult`] with `pattern = Random` and
    /// `confidence = 0.0` when no history is available, or `confidence = 0.1`
    /// when only a single event has been recorded.
    pub fn predict(&self, cid: &str) -> PredictionResult {
        let no_history = || PredictionResult {
            cid: cid.to_owned(),
            pattern: AccessPattern::Random,
            next_access_tick: None,
            confidence: 0.0,
        };

        let entries = match self.history.get(cid) {
            Some(v) if !v.is_empty() => v,
            _ => return no_history(),
        };

        if entries.len() == 1 {
            return PredictionResult {
                cid: cid.to_owned(),
                pattern: AccessPattern::Random,
                next_access_tick: None,
                confidence: 0.1,
            };
        }

        // Compute successive intervals between recorded ticks.
        let intervals: Vec<u64> = entries
            .windows(2)
            .map(|w| w[1].tick.saturating_sub(w[0].tick))
            .collect();

        let last_tick = entries.last().map(|e| e.tick).unwrap_or(0);

        // ── 1. All intervals are identical → Repeated ────────────────────────
        if intervals.iter().all(|&i| i == intervals[0]) {
            let interval = intervals[0];
            return PredictionResult {
                cid: cid.to_owned(),
                pattern: AccessPattern::Repeated {
                    interval_ticks: interval,
                },
                next_access_tick: Some(last_tick + interval),
                confidence: 0.9,
            };
        }

        // ── 2. Intervals are strictly decreasing → Cooling ───────────────────
        let strictly_decreasing = intervals.windows(2).all(|w| w[0] > w[1]);
        if strictly_decreasing {
            return PredictionResult {
                cid: cid.to_owned(),
                pattern: AccessPattern::Cooling,
                next_access_tick: None,
                confidence: 0.7,
            };
        }

        // ── 3. All intervals ≤ 5 → Bursty ───────────────────────────────────
        if intervals.iter().all(|&i| i <= 5) {
            return PredictionResult {
                cid: cid.to_owned(),
                pattern: AccessPattern::Bursty {
                    burst_size: entries.len(),
                },
                next_access_tick: None,
                confidence: 0.6,
            };
        }

        // ── 4. Non-decreasing intervals that vary by ≤ 10% → Sequential ─────
        let non_decreasing = intervals.windows(2).all(|w| w[1] >= w[0]);
        if non_decreasing {
            let avg = intervals.iter().sum::<u64>() as f64 / intervals.len() as f64;
            let max_interval = *intervals.iter().max().unwrap_or(&0);
            let min_interval = *intervals.iter().min().unwrap_or(&0);
            // Variation = (max - min) / avg
            let variation = if avg > 0.0 {
                (max_interval - min_interval) as f64 / avg
            } else {
                0.0
            };
            if variation <= 0.10 {
                let avg_interval = avg.round() as u64;
                return PredictionResult {
                    cid: cid.to_owned(),
                    pattern: AccessPattern::Sequential,
                    next_access_tick: Some(last_tick + avg_interval),
                    confidence: 0.75,
                };
            }
        }

        // ── 5. Fallback → Random ─────────────────────────────────────────────
        PredictionResult {
            cid: cid.to_owned(),
            pattern: AccessPattern::Random,
            next_access_tick: None,
            confidence: 0.2,
        }
    }

    /// Return predictions for all tracked CIDs whose pattern is `Sequential`
    /// or `Repeated`, sorted by descending confidence.
    pub fn top_predicted(&self) -> Vec<PredictionResult> {
        let mut results: Vec<PredictionResult> = self
            .history
            .keys()
            .map(|cid| self.predict(cid))
            .filter(|r| {
                matches!(
                    r.pattern,
                    AccessPattern::Sequential | AccessPattern::Repeated { .. }
                )
            })
            .collect();

        results.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results
    }

    /// Compute aggregate statistics for the current state of the predictor.
    pub fn stats(&self) -> PredictorStats {
        let total_events: u64 = self.history.values().map(|v| v.len() as u64).sum();
        let unique_cids = self.history.len();

        let mut sequential_count = 0usize;
        let mut repeated_count = 0usize;
        let mut random_count = 0usize;

        for cid in self.history.keys() {
            match self.predict(cid).pattern {
                AccessPattern::Sequential => sequential_count += 1,
                AccessPattern::Repeated { .. } => repeated_count += 1,
                AccessPattern::Random => random_count += 1,
                _ => {}
            }
        }

        PredictorStats {
            total_events,
            unique_cids,
            sequential_count,
            repeated_count,
            random_count,
        }
    }
}

impl Default for StorageAccessPredictor {
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

    fn make_event(cid: &str, tick: u64) -> AccessEvent {
        AccessEvent {
            cid: cid.to_owned(),
            tick,
            size_bytes: 512,
        }
    }

    // ── 1. new() starts empty ─────────────────────────────────────────────────
    #[test]
    fn test_new_starts_empty() {
        let predictor = StorageAccessPredictor::new();
        assert!(predictor.history.is_empty());
    }

    // ── 2. record stores event ────────────────────────────────────────────────
    #[test]
    fn test_record_stores_event() {
        let mut p = StorageAccessPredictor::new();
        p.record(make_event("cid-a", 1));
        assert_eq!(p.history["cid-a"].len(), 1);
        assert_eq!(p.history["cid-a"][0].tick, 1);
    }

    // ── 3. record caps at 20 events (oldest evicted) ──────────────────────────
    #[test]
    fn test_record_caps_at_20() {
        let mut p = StorageAccessPredictor::new();
        for i in 0..25u64 {
            p.record(make_event("cid-a", i));
        }
        let entries = &p.history["cid-a"];
        assert_eq!(entries.len(), 20);
        // Oldest kept should be tick 5 (ticks 0-4 were evicted)
        assert_eq!(entries[0].tick, 5);
    }

    // ── 4. predict unknown cid → Random confidence=0.0 ───────────────────────
    #[test]
    fn test_predict_unknown_cid() {
        let p = StorageAccessPredictor::new();
        let result = p.predict("unknown");
        assert_eq!(result.pattern, AccessPattern::Random);
        assert!((result.confidence - 0.0).abs() < f32::EPSILON);
        assert!(result.next_access_tick.is_none());
    }

    // ── 5. predict single event → Random confidence=0.1 ──────────────────────
    #[test]
    fn test_predict_single_event() {
        let mut p = StorageAccessPredictor::new();
        p.record(make_event("cid-a", 10));
        let result = p.predict("cid-a");
        assert_eq!(result.pattern, AccessPattern::Random);
        assert!((result.confidence - 0.1).abs() < f32::EPSILON);
    }

    // ── 6. predict equal intervals → Repeated ────────────────────────────────
    #[test]
    fn test_predict_equal_intervals_repeated() {
        let mut p = StorageAccessPredictor::new();
        for i in 0u64..5 {
            p.record(make_event("cid-r", i * 10));
        }
        let result = p.predict("cid-r");
        assert_eq!(
            result.pattern,
            AccessPattern::Repeated { interval_ticks: 10 }
        );
    }

    // ── 7. predict Repeated: next_access = last_tick + interval ──────────────
    #[test]
    fn test_predict_repeated_next_access() {
        let mut p = StorageAccessPredictor::new();
        for i in 0u64..4 {
            p.record(make_event("cid-r", i * 5));
        }
        let result = p.predict("cid-r");
        // last_tick = 15, interval = 5 → next = 20
        assert_eq!(result.next_access_tick, Some(20));
    }

    // ── 8. predict Repeated: confidence = 0.9 ────────────────────────────────
    #[test]
    fn test_predict_repeated_confidence() {
        let mut p = StorageAccessPredictor::new();
        for i in 0u64..3 {
            p.record(make_event("cid-r", i * 7));
        }
        let result = p.predict("cid-r");
        assert!((result.confidence - 0.9).abs() < f32::EPSILON);
    }

    // ── 9. predict decreasing intervals → Cooling ────────────────────────────
    #[test]
    fn test_predict_decreasing_intervals_cooling() {
        let mut p = StorageAccessPredictor::new();
        // ticks: 0, 100, 190, 270, 340  → intervals: 100, 90, 80, 70 (strictly decreasing)
        let ticks = [0u64, 100, 190, 270, 340];
        for &t in &ticks {
            p.record(make_event("cid-c", t));
        }
        let result = p.predict("cid-c");
        assert_eq!(result.pattern, AccessPattern::Cooling);
    }

    // ── 10. predict Cooling: next_access is None ──────────────────────────────
    #[test]
    fn test_predict_cooling_next_access_none() {
        let mut p = StorageAccessPredictor::new();
        let ticks = [0u64, 100, 190, 270, 340];
        for &t in &ticks {
            p.record(make_event("cid-c", t));
        }
        let result = p.predict("cid-c");
        assert!(result.next_access_tick.is_none());
    }

    // ── 11. predict small intervals (≤5) → Bursty ────────────────────────────
    #[test]
    fn test_predict_small_intervals_bursty() {
        let mut p = StorageAccessPredictor::new();
        // ticks: 0, 2, 4, 6 → all intervals = 2 ≤ 5
        // But equal intervals → Repeated takes priority; use mixed ≤5 values
        for &t in &[0u64, 1, 3, 5, 7] {
            p.record(make_event("cid-b", t));
        }
        let result = p.predict("cid-b");
        assert_eq!(result.pattern, AccessPattern::Bursty { burst_size: 5 });
    }

    // ── 12. predict Bursty: burst_size = history len ──────────────────────────
    #[test]
    fn test_predict_bursty_burst_size() {
        let mut p = StorageAccessPredictor::new();
        for &t in &[0u64, 1, 3, 5, 7, 9] {
            p.record(make_event("cid-b", t));
        }
        let result = p.predict("cid-b");
        if let AccessPattern::Bursty { burst_size } = result.pattern {
            assert_eq!(burst_size, 6);
        } else {
            panic!("Expected Bursty, got {:?}", result.pattern);
        }
    }

    // ── 13. predict varying intervals → Random confidence=0.2 ────────────────
    #[test]
    fn test_predict_random_confidence() {
        let mut p = StorageAccessPredictor::new();
        // Irregular intervals that are not monotone and not all ≤5
        for &t in &[0u64, 10, 15, 100, 102, 200] {
            p.record(make_event("cid-x", t));
        }
        let result = p.predict("cid-x");
        assert_eq!(result.pattern, AccessPattern::Random);
        assert!((result.confidence - 0.2).abs() < f32::EPSILON);
    }

    // ── 14. top_predicted returns only Sequential/Repeated ───────────────────
    #[test]
    fn test_top_predicted_only_sequential_repeated() {
        let mut p = StorageAccessPredictor::new();

        // Repeated pattern
        for i in 0u64..4 {
            p.record(make_event("cid-rep", i * 10));
        }
        // Random pattern
        for &t in &[0u64, 10, 15, 100, 102, 200] {
            p.record(make_event("cid-rand", t));
        }

        let top = p.top_predicted();
        for r in &top {
            assert!(
                matches!(
                    r.pattern,
                    AccessPattern::Sequential | AccessPattern::Repeated { .. }
                ),
                "Unexpected pattern {:?} for {}",
                r.pattern,
                r.cid
            );
        }
        assert!(!top.is_empty());
    }

    // ── 15. top_predicted sorted by confidence desc ───────────────────────────
    #[test]
    fn test_top_predicted_sorted_by_confidence_desc() {
        let mut p = StorageAccessPredictor::new();

        // Repeated (confidence 0.9)
        for i in 0u64..4 {
            p.record(make_event("cid-rep", i * 10));
        }
        // Sequential (confidence 0.75) — non-decreasing, ≤10% variation
        // intervals: 10, 10, 11, 11 — avg=10.5 var=(11-10)/10.5≈0.095 ≤0.10
        for &t in &[0u64, 10, 20, 31, 42] {
            p.record(make_event("cid-seq", t));
        }

        let top = p.top_predicted();
        for pair in top.windows(2) {
            assert!(pair[0].confidence >= pair[1].confidence);
        }
    }

    // ── 16. stats total_events increments ─────────────────────────────────────
    #[test]
    fn test_stats_total_events() {
        let mut p = StorageAccessPredictor::new();
        assert_eq!(p.stats().total_events, 0);
        p.record(make_event("cid-a", 1));
        assert_eq!(p.stats().total_events, 1);
        p.record(make_event("cid-a", 2));
        assert_eq!(p.stats().total_events, 2);
        p.record(make_event("cid-b", 5));
        assert_eq!(p.stats().total_events, 3);
    }

    // ── 17. stats unique_cids correct ─────────────────────────────────────────
    #[test]
    fn test_stats_unique_cids() {
        let mut p = StorageAccessPredictor::new();
        assert_eq!(p.stats().unique_cids, 0);
        p.record(make_event("a", 1));
        assert_eq!(p.stats().unique_cids, 1);
        p.record(make_event("b", 2));
        assert_eq!(p.stats().unique_cids, 2);
        p.record(make_event("a", 3));
        assert_eq!(p.stats().unique_cids, 2); // still 2 unique
    }

    // ── 18. multiple CIDs tracked independently ───────────────────────────────
    #[test]
    fn test_multiple_cids_independent() {
        let mut p = StorageAccessPredictor::new();
        for i in 0u64..3 {
            p.record(make_event("alpha", i * 10));
            p.record(make_event("beta", i * 7));
        }
        let ra = p.predict("alpha");
        let rb = p.predict("beta");
        assert_eq!(ra.pattern, AccessPattern::Repeated { interval_ticks: 10 });
        assert_eq!(rb.pattern, AccessPattern::Repeated { interval_ticks: 7 });
    }

    // ── 19. stats sequential_count / repeated_count / random_count ────────────
    #[test]
    fn test_stats_pattern_counts() {
        let mut p = StorageAccessPredictor::new();

        // Repeated
        for i in 0u64..4 {
            p.record(make_event("rep", i * 5));
        }
        // Random
        for &t in &[0u64, 10, 15, 100, 102, 200] {
            p.record(make_event("rnd", t));
        }

        let s = p.stats();
        assert_eq!(s.repeated_count, 1);
        assert_eq!(s.random_count, 1);
    }

    // ── 20. history eviction preserves order ──────────────────────────────────
    #[test]
    fn test_history_eviction_preserves_order() {
        let mut p = StorageAccessPredictor::new();
        for i in 0..25u64 {
            p.record(make_event("cid-ord", i));
        }
        let entries = &p.history["cid-ord"];
        // After evicting 5 (ticks 0..4), remaining should be 5..24 in order.
        assert_eq!(entries.len(), 20);
        for (idx, entry) in entries.iter().enumerate() {
            assert_eq!(entry.tick, (idx as u64) + 5);
        }
    }

    // ── 21. predict Bursty: confidence = 0.6 ─────────────────────────────────
    #[test]
    fn test_predict_bursty_confidence() {
        let mut p = StorageAccessPredictor::new();
        for &t in &[0u64, 1, 3, 5, 7] {
            p.record(make_event("cid-b2", t));
        }
        let result = p.predict("cid-b2");
        assert!((result.confidence - 0.6).abs() < f32::EPSILON);
    }

    // ── 22. predict Cooling: confidence = 0.7 ────────────────────────────────
    #[test]
    fn test_predict_cooling_confidence() {
        let mut p = StorageAccessPredictor::new();
        let ticks = [0u64, 100, 190, 270, 340];
        for &t in &ticks {
            p.record(make_event("cid-cool2", t));
        }
        let result = p.predict("cid-cool2");
        assert!((result.confidence - 0.7).abs() < f32::EPSILON);
    }

    // ── 23. top_predicted is empty when no predictable patterns ───────────────
    #[test]
    fn test_top_predicted_empty_when_no_patterns() {
        let mut p = StorageAccessPredictor::new();
        // Only random-pattern CIDs
        for &t in &[0u64, 10, 15, 100, 102, 200] {
            p.record(make_event("rnd1", t));
        }
        for &t in &[0u64, 50, 55, 300, 305, 700] {
            p.record(make_event("rnd2", t));
        }
        assert!(p.top_predicted().is_empty());
    }

    // ── 24. default() produces same state as new() ────────────────────────────
    #[test]
    fn test_default_is_empty() {
        let p = StorageAccessPredictor::default();
        assert!(p.history.is_empty());
        assert_eq!(p.stats().total_events, 0);
    }
}
