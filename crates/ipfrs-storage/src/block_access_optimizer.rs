//! Block Access Optimizer — learns co-access patterns, predicts future accesses,
//! and generates prefetch recommendations to minimize storage latency.

use std::cmp::Reverse;
use std::collections::{HashMap, VecDeque};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// A single block access event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AccessEvent {
    /// Content identifier of the accessed block.
    pub cid: String,
    /// Wall-clock timestamp in milliseconds since epoch.
    pub timestamp_ms: u64,
    /// Observed latency for this access in milliseconds.
    pub latency_ms: u32,
    /// Whether the access was served from cache.
    pub cache_hit: bool,
}

/// Tracks how often two blocks are accessed together (within the sliding window).
#[derive(Debug, Clone)]
pub struct CoAccessPair {
    pub cid_a: String,
    pub cid_b: String,
    /// How many times these two CIDs have been co-accessed.
    pub co_access_count: u64,
    /// Average time interval between co-accesses in milliseconds.
    pub avg_interval_ms: u64,
}

/// A prefetch recommendation generated when `trigger_cid` is accessed.
#[derive(Debug, Clone)]
pub struct PrefetchRecommendation {
    /// The CID whose access triggered this recommendation.
    pub trigger_cid: String,
    /// Ordered list of CIDs to prefetch (highest confidence first).
    pub prefetch_cids: Vec<String>,
    /// Aggregate confidence score in [0, 1].
    pub confidence: f64,
    /// Estimated latency savings in milliseconds if the prefetch is served from cache.
    pub estimated_benefit_ms: u32,
}

/// Summarises the access history for a single block.
#[derive(Debug, Clone)]
pub struct AccessPattern {
    pub cid: String,
    /// Number of times this block has been accessed.
    pub total_accesses: u64,
    /// EWMA-smoothed average inter-access interval in milliseconds.
    pub avg_interval_ms: u64,
    /// Timestamp of the most recent access in milliseconds.
    pub last_access_ms: u64,
    /// Predicted timestamp of the next access: `last_access_ms + avg_interval_ms`.
    pub predicted_next_ms: u64,
}

/// Configuration knobs for `BlockAccessOptimizer`.
#[derive(Debug, Clone)]
pub struct OptimizerConfig {
    /// Maximum number of events kept in the sliding window.
    pub window_size: usize,
    /// Minimum co-access count before a pair is considered for prefetch.
    pub min_co_access_count: u64,
    /// Maximum number of CIDs included in a single prefetch recommendation.
    pub max_prefetch_candidates: usize,
    /// Minimum confidence score for a CID to be included in a recommendation.
    pub confidence_threshold: f64,
    /// Multiplicative decay applied to co-access counts on each `apply_decay` call.
    pub pattern_decay_factor: f64,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            window_size: 1000,
            min_co_access_count: 3,
            max_prefetch_candidates: 5,
            confidence_threshold: 0.6,
            pattern_decay_factor: 0.95,
        }
    }
}

/// Snapshot of optimizer-level statistics.
#[derive(Debug, Clone)]
pub struct OptimizerStats {
    pub total_events: u64,
    pub unique_blocks: usize,
    pub co_access_pairs: usize,
    pub cache_hit_rate: f64,
    pub window_size: usize,
}

// ---------------------------------------------------------------------------
// BlockAccessOptimizer
// ---------------------------------------------------------------------------

/// Access-pattern optimizer that learns block co-access patterns, predicts
/// future accesses, and generates prefetch recommendations.
pub struct BlockAccessOptimizer {
    /// Configuration parameters.
    pub config: OptimizerConfig,
    /// Sliding window of recent access events (bounded by `config.window_size`).
    pub recent_window: VecDeque<AccessEvent>,
    /// Per-block access statistics.
    pub patterns: HashMap<String, AccessPattern>,
    /// Co-access statistics: `co_access[cid_a][cid_b]`.
    pub co_access: HashMap<String, HashMap<String, CoAccessPair>>,
    /// Total number of events ever recorded.
    pub total_events: u64,
    /// Number of events that were cache hits.
    pub cache_hit_count: u64,
}

impl BlockAccessOptimizer {
    /// Create a new optimizer with the given configuration.
    pub fn new(config: OptimizerConfig) -> Self {
        Self {
            config,
            recent_window: VecDeque::new(),
            patterns: HashMap::new(),
            co_access: HashMap::new(),
            total_events: 0,
            cache_hit_count: 0,
        }
    }

    /// Create a new optimizer with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(OptimizerConfig::default())
    }

    // -----------------------------------------------------------------------
    // Core recording logic
    // -----------------------------------------------------------------------

    /// Record a block access event.
    ///
    /// This method:
    /// 1. Updates the sliding window (evicting the oldest event if necessary).
    /// 2. Updates the per-block `AccessPattern` using EWMA smoothing (α = 0.3).
    /// 3. Records co-access pairs with the last 5 events in the window.
    pub fn record_access(&mut self, event: AccessEvent) {
        // --- counters ---
        self.total_events += 1;
        if event.cache_hit {
            self.cache_hit_count += 1;
        }

        let cid = event.cid.clone();
        let ts = event.timestamp_ms;

        // --- update per-block pattern ---
        match self.patterns.get_mut(&cid) {
            Some(pat) => {
                // Compute inter-access interval only when we have a previous access.
                let interval = ts.saturating_sub(pat.last_access_ms);
                // EWMA with α = 0.3
                const ALPHA: f64 = 0.3;
                let new_avg = if pat.total_accesses == 1 {
                    // First update: bootstrap with the observed interval.
                    interval as f64
                } else {
                    ALPHA * interval as f64 + (1.0 - ALPHA) * pat.avg_interval_ms as f64
                };
                pat.avg_interval_ms = new_avg.round() as u64;
                pat.last_access_ms = ts;
                pat.predicted_next_ms = ts + pat.avg_interval_ms;
                pat.total_accesses += 1;
            }
            None => {
                self.patterns.insert(
                    cid.clone(),
                    AccessPattern {
                        cid: cid.clone(),
                        total_accesses: 1,
                        avg_interval_ms: 0,
                        last_access_ms: ts,
                        predicted_next_ms: ts,
                    },
                );
            }
        }

        // --- record co-access pairs with last 5 window events ---
        // Collect previous (cid, timestamp) before any mutable borrows to avoid
        // conflicting borrows on `self`.
        let look_back = self.recent_window.len().min(5);
        let start = self.recent_window.len().saturating_sub(look_back);
        let prev_events: Vec<(String, u64)> = self
            .recent_window
            .iter()
            .skip(start)
            .filter(|e| e.cid != cid)
            .map(|e| (e.cid.clone(), e.timestamp_ms))
            .collect();

        for (prev_cid, prev_ts) in prev_events {
            let interval = ts.saturating_sub(prev_ts);
            self.update_co_access_pair(cid.clone(), prev_cid.clone(), interval);
            self.update_co_access_pair(prev_cid, cid.clone(), interval);
        }

        // --- maintain sliding window ---
        if self.recent_window.len() >= self.config.window_size {
            self.recent_window.pop_front();
        }
        self.recent_window.push_back(event);
    }

    /// Update or insert a `CoAccessPair` entry.
    fn update_co_access_pair(&mut self, cid_a: String, cid_b: String, interval_ms: u64) {
        let inner = self.co_access.entry(cid_a.clone()).or_default();
        match inner.get_mut(&cid_b) {
            Some(pair) => {
                pair.co_access_count += 1;
                // EWMA with α = 0.3 for interval smoothing.
                const ALPHA: f64 = 0.3;
                let new_avg =
                    ALPHA * interval_ms as f64 + (1.0 - ALPHA) * pair.avg_interval_ms as f64;
                pair.avg_interval_ms = new_avg.round() as u64;
            }
            None => {
                inner.insert(
                    cid_b.clone(),
                    CoAccessPair {
                        cid_a,
                        cid_b,
                        co_access_count: 1,
                        avg_interval_ms: interval_ms,
                    },
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // Recommendations & predictions
    // -----------------------------------------------------------------------

    /// Generate a prefetch recommendation for a given trigger CID.
    ///
    /// Returns a `PrefetchRecommendation` even when no candidates meet the
    /// threshold (in that case `prefetch_cids` will be empty and confidence = 0).
    pub fn recommend_prefetch(&self, trigger_cid: &str) -> PrefetchRecommendation {
        let candidates = match self.co_access.get(trigger_cid) {
            Some(inner) => inner,
            None => {
                return PrefetchRecommendation {
                    trigger_cid: trigger_cid.to_owned(),
                    prefetch_cids: vec![],
                    confidence: 0.0,
                    estimated_benefit_ms: 0,
                };
            }
        };

        // Filter by minimum co-access count.
        let mut pairs: Vec<&CoAccessPair> = candidates
            .values()
            .filter(|p| p.co_access_count >= self.config.min_co_access_count)
            .collect();

        // Sort by co-access count descending.
        pairs.sort_by_key(|b| Reverse(b.co_access_count));

        // Take the top candidates and compute per-candidate confidence.
        let mut prefetch_cids: Vec<String> = Vec::new();
        let mut total_confidence = 0.0_f64;
        let mut count = 0usize;

        for pair in pairs.into_iter().take(self.config.max_prefetch_candidates) {
            let n = pair.co_access_count as f64;
            let confidence = (n / (n + 1.0)).min(1.0);
            if confidence >= self.config.confidence_threshold {
                prefetch_cids.push(pair.cid_b.clone());
                total_confidence += confidence;
                count += 1;
            }
        }

        let aggregate_confidence = if count > 0 {
            (total_confidence / count as f64).min(1.0)
        } else {
            0.0
        };

        // Estimated benefit: 50 ms placeholder per candidate.
        let estimated_benefit_ms = (count as u32) * 50;

        PrefetchRecommendation {
            trigger_cid: trigger_cid.to_owned(),
            prefetch_cids,
            confidence: aggregate_confidence,
            estimated_benefit_ms,
        }
    }

    /// Predict the next access timestamp (ms) for the given CID.
    pub fn predict_next_access(&self, cid: &str) -> Option<u64> {
        self.patterns.get(cid).map(|p| p.predicted_next_ms)
    }

    // -----------------------------------------------------------------------
    // Analytics helpers
    // -----------------------------------------------------------------------

    /// Return the top-`k` hottest blocks by total access count.
    pub fn hot_blocks(&self, k: usize) -> Vec<&AccessPattern> {
        let mut patterns: Vec<&AccessPattern> = self.patterns.values().collect();
        patterns.sort_by_key(|b| Reverse(b.total_accesses));
        patterns.into_iter().take(k).collect()
    }

    /// Return the top-`k` co-access pairs by co-access count across all CIDs.
    pub fn top_co_access_pairs(&self, k: usize) -> Vec<&CoAccessPair> {
        let mut all_pairs: Vec<&CoAccessPair> = self
            .co_access
            .values()
            .flat_map(|inner| inner.values())
            .collect();
        all_pairs.sort_by_key(|b| Reverse(b.co_access_count));
        all_pairs.into_iter().take(k).collect()
    }

    // -----------------------------------------------------------------------
    // Maintenance
    // -----------------------------------------------------------------------

    /// Apply exponential decay to all co-access counts.
    ///
    /// Counts are multiplied by `pattern_decay_factor` and rounded.
    /// Pairs whose count drops below 1 are removed.
    pub fn apply_decay(&mut self) {
        let decay = self.config.pattern_decay_factor;
        for inner in self.co_access.values_mut() {
            inner.retain(|_, pair| {
                let decayed = (pair.co_access_count as f64 * decay).round() as u64;
                pair.co_access_count = decayed;
                decayed >= 1
            });
        }
        // Remove top-level entries that have become empty.
        self.co_access.retain(|_, inner| !inner.is_empty());
    }

    // -----------------------------------------------------------------------
    // Statistics
    // -----------------------------------------------------------------------

    /// Overall cache-hit rate across all recorded events.
    pub fn cache_hit_rate(&self) -> f64 {
        if self.total_events == 0 {
            0.0
        } else {
            self.cache_hit_count as f64 / self.total_events as f64
        }
    }

    /// Return a snapshot of high-level optimizer statistics.
    pub fn optimizer_stats(&self) -> OptimizerStats {
        let co_access_pairs = self
            .co_access
            .values()
            .map(|inner| inner.len())
            .sum::<usize>();

        OptimizerStats {
            total_events: self.total_events,
            unique_blocks: self.patterns.len(),
            co_access_pairs,
            cache_hit_rate: self.cache_hit_rate(),
            window_size: self.recent_window.len(),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use crate::block_access_optimizer::{AccessEvent, BlockAccessOptimizer, OptimizerConfig};

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn event(cid: &str, ts: u64, latency: u32, hit: bool) -> AccessEvent {
        AccessEvent {
            cid: cid.to_owned(),
            timestamp_ms: ts,
            latency_ms: latency,
            cache_hit: hit,
        }
    }

    fn default_optimizer() -> BlockAccessOptimizer {
        BlockAccessOptimizer::with_defaults()
    }

    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    #[test]
    fn test_new_empty() {
        let opt = default_optimizer();
        assert_eq!(opt.total_events, 0);
        assert_eq!(opt.cache_hit_count, 0);
        assert!(opt.patterns.is_empty());
        assert!(opt.co_access.is_empty());
        assert!(opt.recent_window.is_empty());
    }

    #[test]
    fn test_new_with_config() {
        let cfg = OptimizerConfig {
            window_size: 50,
            min_co_access_count: 2,
            max_prefetch_candidates: 3,
            confidence_threshold: 0.5,
            pattern_decay_factor: 0.9,
        };
        let opt = BlockAccessOptimizer::new(cfg.clone());
        assert_eq!(opt.config.window_size, 50);
        assert_eq!(opt.config.min_co_access_count, 2);
        assert_eq!(opt.config.max_prefetch_candidates, 3);
        assert!((opt.config.confidence_threshold - 0.5).abs() < 1e-9);
        assert!((opt.config.pattern_decay_factor - 0.9).abs() < 1e-9);
    }

    // -----------------------------------------------------------------------
    // record_access — basic
    // -----------------------------------------------------------------------

    #[test]
    fn test_record_increments_total_events() {
        let mut opt = default_optimizer();
        opt.record_access(event("cid1", 1000, 10, false));
        assert_eq!(opt.total_events, 1);
        opt.record_access(event("cid2", 2000, 5, true));
        assert_eq!(opt.total_events, 2);
    }

    #[test]
    fn test_record_increments_cache_hit_count() {
        let mut opt = default_optimizer();
        opt.record_access(event("cid1", 1000, 10, true));
        opt.record_access(event("cid2", 2000, 5, false));
        opt.record_access(event("cid3", 3000, 8, true));
        assert_eq!(opt.cache_hit_count, 2);
    }

    #[test]
    fn test_record_creates_pattern() {
        let mut opt = default_optimizer();
        opt.record_access(event("cid1", 1000, 10, false));
        assert!(opt.patterns.contains_key("cid1"));
        let pat = opt.patterns.get("cid1").expect("pattern should exist");
        assert_eq!(pat.cid, "cid1");
        assert_eq!(pat.total_accesses, 1);
        assert_eq!(pat.last_access_ms, 1000);
    }

    #[test]
    fn test_record_increments_total_accesses() {
        let mut opt = default_optimizer();
        opt.record_access(event("cid1", 1000, 10, false));
        opt.record_access(event("cid1", 2000, 5, false));
        let pat = opt.patterns.get("cid1").expect("pattern should exist");
        assert_eq!(pat.total_accesses, 2);
    }

    #[test]
    fn test_record_updates_last_access() {
        let mut opt = default_optimizer();
        opt.record_access(event("cid1", 1000, 10, false));
        opt.record_access(event("cid1", 3000, 5, false));
        let pat = opt.patterns.get("cid1").expect("pattern should exist");
        assert_eq!(pat.last_access_ms, 3000);
    }

    #[test]
    fn test_predicted_next_equals_last_plus_avg() {
        let mut opt = default_optimizer();
        opt.record_access(event("cid1", 1000, 10, false));
        opt.record_access(event("cid1", 3000, 5, false));
        let pat = opt.patterns.get("cid1").expect("pattern should exist");
        assert_eq!(
            pat.predicted_next_ms,
            pat.last_access_ms + pat.avg_interval_ms
        );
    }

    // -----------------------------------------------------------------------
    // EWMA
    // -----------------------------------------------------------------------

    #[test]
    fn test_avg_interval_bootstrap_on_second_access() {
        let mut opt = default_optimizer();
        opt.record_access(event("cid1", 0, 10, false));
        opt.record_access(event("cid1", 1000, 5, false));
        // After first update EWMA bootstraps to the observed interval (1000 ms).
        let pat = opt.patterns.get("cid1").expect("pattern should exist");
        assert_eq!(pat.avg_interval_ms, 1000);
    }

    #[test]
    fn test_avg_interval_ewma_third_access() {
        let mut opt = default_optimizer();
        opt.record_access(event("cid1", 0, 10, false));
        opt.record_access(event("cid1", 1000, 5, false)); // avg = 1000
        opt.record_access(event("cid1", 3000, 5, false)); // interval = 2000, alpha = 0.3
                                                          // new_avg = 0.3 * 2000 + 0.7 * 1000 = 600 + 700 = 1300
        let pat = opt.patterns.get("cid1").expect("pattern should exist");
        assert_eq!(pat.avg_interval_ms, 1300);
    }

    // -----------------------------------------------------------------------
    // Sliding window
    // -----------------------------------------------------------------------

    #[test]
    fn test_window_respects_size() {
        let cfg = OptimizerConfig {
            window_size: 5,
            ..OptimizerConfig::default()
        };
        let mut opt = BlockAccessOptimizer::new(cfg);
        for i in 0..10u64 {
            opt.record_access(event(&format!("c{}", i), i * 100, 10, false));
        }
        assert_eq!(opt.recent_window.len(), 5);
    }

    #[test]
    fn test_window_evicts_oldest() {
        let cfg = OptimizerConfig {
            window_size: 3,
            ..OptimizerConfig::default()
        };
        let mut opt = BlockAccessOptimizer::new(cfg);
        opt.record_access(event("old", 100, 10, false));
        opt.record_access(event("mid", 200, 10, false));
        opt.record_access(event("new1", 300, 10, false));
        opt.record_access(event("new2", 400, 10, false));
        // "old" should be evicted
        assert!(opt.recent_window.iter().all(|e| e.cid != "old"));
    }

    // -----------------------------------------------------------------------
    // Co-access tracking
    // -----------------------------------------------------------------------

    #[test]
    fn test_co_access_created_for_pair() {
        let mut opt = default_optimizer();
        opt.record_access(event("A", 100, 5, false));
        opt.record_access(event("B", 200, 5, false));
        // B was recorded after A; A should be in B's co-access and vice versa.
        assert!(opt.co_access.get("B").is_some_and(|m| m.contains_key("A")));
        assert!(opt.co_access.get("A").is_some_and(|m| m.contains_key("B")));
    }

    #[test]
    fn test_co_access_count_increments() {
        let mut opt = default_optimizer();
        opt.record_access(event("A", 100, 5, false));
        opt.record_access(event("B", 200, 5, false));
        opt.record_access(event("A", 300, 5, false));
        opt.record_access(event("B", 400, 5, false));
        let count = opt
            .co_access
            .get("B")
            .and_then(|m| m.get("A"))
            .map(|p| p.co_access_count)
            .unwrap_or(0);
        assert!(count >= 2, "expected at least 2 co-accesses, got {count}");
    }

    #[test]
    fn test_co_access_same_cid_skipped() {
        let mut opt = default_optimizer();
        opt.record_access(event("A", 100, 5, false));
        opt.record_access(event("A", 200, 5, false));
        // Self-pair should not be recorded.
        let has_self = opt.co_access.get("A").is_some_and(|m| m.contains_key("A"));
        assert!(!has_self);
    }

    #[test]
    fn test_co_access_look_back_limit() {
        let mut opt = default_optimizer();
        // Record 8 distinct CIDs.
        for i in 0..8u64 {
            opt.record_access(event(&format!("c{}", i), i * 100, 5, false));
        }
        // The last event (c7) should have co-accesses with at most 5 predecessors (c2..c6).
        let inner = opt.co_access.get("c7").cloned().unwrap_or_default();
        // c0 and c1 should NOT be in c7's co-access (outside look-back window).
        assert!(!inner.contains_key("c0"), "c0 should be outside look-back");
        assert!(!inner.contains_key("c1"), "c1 should be outside look-back");
    }

    // -----------------------------------------------------------------------
    // predict_next_access
    // -----------------------------------------------------------------------

    #[test]
    fn test_predict_next_returns_none_for_unknown() {
        let opt = default_optimizer();
        assert!(opt.predict_next_access("nonexistent").is_none());
    }

    #[test]
    fn test_predict_next_returns_some_after_access() {
        let mut opt = default_optimizer();
        opt.record_access(event("A", 1000, 5, false));
        assert!(opt.predict_next_access("A").is_some());
    }

    #[test]
    fn test_predict_next_matches_pattern() {
        let mut opt = default_optimizer();
        opt.record_access(event("A", 0, 5, false));
        opt.record_access(event("A", 2000, 5, false));
        let predicted = opt.predict_next_access("A").unwrap_or(0);
        let pat = opt.patterns.get("A").expect("pattern");
        assert_eq!(predicted, pat.predicted_next_ms);
    }

    // -----------------------------------------------------------------------
    // hot_blocks
    // -----------------------------------------------------------------------

    #[test]
    fn test_hot_blocks_ordering() {
        let mut opt = default_optimizer();
        // "hot" accessed 5 times, "cold" accessed once.
        for i in 0..5u64 {
            opt.record_access(event("hot", i * 100, 5, false));
        }
        opt.record_access(event("cold", 600, 5, false));
        let top = opt.hot_blocks(1);
        assert_eq!(top.len(), 1);
        assert_eq!(top[0].cid, "hot");
    }

    #[test]
    fn test_hot_blocks_k_limit() {
        let mut opt = default_optimizer();
        for i in 0..10u64 {
            opt.record_access(event(&format!("c{}", i), i * 100, 5, false));
        }
        let top = opt.hot_blocks(3);
        assert_eq!(top.len(), 3);
    }

    #[test]
    fn test_hot_blocks_empty() {
        let opt = default_optimizer();
        assert!(opt.hot_blocks(5).is_empty());
    }

    // -----------------------------------------------------------------------
    // top_co_access_pairs
    // -----------------------------------------------------------------------

    #[test]
    fn test_top_co_access_pairs_ordering() {
        let mut opt = default_optimizer();
        // Generate many co-accesses between A and B.
        for i in 0..10u64 {
            opt.record_access(event("A", i * 100, 5, false));
            opt.record_access(event("B", i * 100 + 50, 5, false));
        }
        // One access between C and D.
        opt.record_access(event("C", 5000, 5, false));
        opt.record_access(event("D", 5050, 5, false));
        let top = opt.top_co_access_pairs(1);
        assert_eq!(top.len(), 1);
        assert!(
            (top[0].cid_a == "A" && top[0].cid_b == "B")
                || (top[0].cid_a == "B" && top[0].cid_b == "A"),
            "Expected A-B pair at top, got {}-{}",
            top[0].cid_a,
            top[0].cid_b,
        );
    }

    #[test]
    fn test_top_co_access_pairs_empty() {
        let opt = default_optimizer();
        assert!(opt.top_co_access_pairs(5).is_empty());
    }

    // -----------------------------------------------------------------------
    // recommend_prefetch
    // -----------------------------------------------------------------------

    #[test]
    fn test_recommend_prefetch_empty_for_unknown_cid() {
        let opt = default_optimizer();
        let rec = opt.recommend_prefetch("nope");
        assert!(rec.prefetch_cids.is_empty());
        assert_eq!(rec.trigger_cid, "nope");
        assert!((rec.confidence - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_recommend_prefetch_filters_below_threshold() {
        let cfg = OptimizerConfig {
            min_co_access_count: 10, // very high
            ..OptimizerConfig::default()
        };
        let mut opt = BlockAccessOptimizer::new(cfg);
        opt.record_access(event("A", 100, 5, false));
        opt.record_access(event("B", 200, 5, false));
        let rec = opt.recommend_prefetch("A");
        // Only 1 co-access, below min_co_access_count=10 → empty.
        assert!(rec.prefetch_cids.is_empty());
    }

    #[test]
    fn test_recommend_prefetch_returns_candidates() {
        let cfg = OptimizerConfig {
            min_co_access_count: 2,
            confidence_threshold: 0.5,
            ..OptimizerConfig::default()
        };
        let mut opt = BlockAccessOptimizer::new(cfg);
        // Generate enough co-accesses to exceed threshold.
        for i in 0..10u64 {
            opt.record_access(event("trigger", i * 200, 5, false));
            opt.record_access(event("follower", i * 200 + 100, 5, false));
        }
        let rec = opt.recommend_prefetch("trigger");
        assert!(!rec.prefetch_cids.is_empty());
        assert!(rec.confidence >= 0.5);
    }

    #[test]
    fn test_recommend_prefetch_respects_max_candidates() {
        let cfg = OptimizerConfig {
            min_co_access_count: 1,
            max_prefetch_candidates: 2,
            confidence_threshold: 0.0,
            ..OptimizerConfig::default()
        };
        let mut opt = BlockAccessOptimizer::new(cfg);
        opt.record_access(event("trigger", 100, 5, false));
        for i in 1..=6u64 {
            opt.record_access(event(&format!("follow{}", i), 100 + i * 10, 5, false));
        }
        // Prime the window.
        opt.record_access(event("trigger", 200, 5, false));
        let rec = opt.recommend_prefetch("trigger");
        assert!(rec.prefetch_cids.len() <= 2);
    }

    #[test]
    fn test_recommend_prefetch_trigger_cid_in_result() {
        let opt = default_optimizer();
        let rec = opt.recommend_prefetch("xyz");
        assert_eq!(rec.trigger_cid, "xyz");
    }

    #[test]
    fn test_recommend_prefetch_estimated_benefit_nonzero() {
        let cfg = OptimizerConfig {
            min_co_access_count: 2,
            confidence_threshold: 0.5,
            ..OptimizerConfig::default()
        };
        let mut opt = BlockAccessOptimizer::new(cfg);
        for i in 0..10u64 {
            opt.record_access(event("A", i * 200, 5, false));
            opt.record_access(event("B", i * 200 + 100, 5, false));
        }
        let rec = opt.recommend_prefetch("A");
        if !rec.prefetch_cids.is_empty() {
            assert!(rec.estimated_benefit_ms > 0);
        }
    }

    // -----------------------------------------------------------------------
    // cache_hit_rate
    // -----------------------------------------------------------------------

    #[test]
    fn test_cache_hit_rate_zero_events() {
        let opt = default_optimizer();
        assert!((opt.cache_hit_rate() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_cache_hit_rate_all_hits() {
        let mut opt = default_optimizer();
        opt.record_access(event("A", 100, 5, true));
        opt.record_access(event("B", 200, 5, true));
        assert!((opt.cache_hit_rate() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_cache_hit_rate_no_hits() {
        let mut opt = default_optimizer();
        opt.record_access(event("A", 100, 5, false));
        opt.record_access(event("B", 200, 5, false));
        assert!((opt.cache_hit_rate() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_cache_hit_rate_partial() {
        let mut opt = default_optimizer();
        opt.record_access(event("A", 100, 5, true));
        opt.record_access(event("B", 200, 5, false));
        let rate = opt.cache_hit_rate();
        assert!((rate - 0.5).abs() < 1e-9);
    }

    // -----------------------------------------------------------------------
    // apply_decay
    // -----------------------------------------------------------------------

    #[test]
    fn test_apply_decay_reduces_counts() {
        let cfg = OptimizerConfig {
            pattern_decay_factor: 0.5,
            ..OptimizerConfig::default()
        };
        let mut opt = BlockAccessOptimizer::new(cfg);
        // Build up a co-access count of 10.
        for i in 0..10u64 {
            opt.record_access(event("A", i * 100, 5, false));
            opt.record_access(event("B", i * 100 + 50, 5, false));
        }
        let before = opt
            .co_access
            .get("A")
            .and_then(|m| m.get("B"))
            .map(|p| p.co_access_count)
            .unwrap_or(0);
        opt.apply_decay();
        let after = opt
            .co_access
            .get("A")
            .and_then(|m| m.get("B"))
            .map(|p| p.co_access_count)
            .unwrap_or(0);
        assert!(after < before, "after={after} should be < before={before}");
    }

    #[test]
    fn test_apply_decay_removes_low_count_pairs() {
        let cfg = OptimizerConfig {
            pattern_decay_factor: 0.01, // extreme decay
            ..OptimizerConfig::default()
        };
        let mut opt = BlockAccessOptimizer::new(cfg);
        opt.record_access(event("A", 100, 5, false));
        opt.record_access(event("B", 200, 5, false));
        opt.apply_decay();
        // count was 1, after *0.01 rounds to 0 → removed.
        let exists = opt.co_access.get("A").is_some_and(|m| m.contains_key("B"));
        assert!(!exists, "pair should have been removed after extreme decay");
    }

    #[test]
    fn test_apply_decay_cleans_empty_outer_map() {
        let cfg = OptimizerConfig {
            pattern_decay_factor: 0.01,
            ..OptimizerConfig::default()
        };
        let mut opt = BlockAccessOptimizer::new(cfg);
        opt.record_access(event("A", 100, 5, false));
        opt.record_access(event("B", 200, 5, false));
        opt.apply_decay();
        // Outer map for "A" should be removed when it becomes empty.
        let outer_empty = !opt.co_access.contains_key("A")
            || opt.co_access.get("A").is_some_and(|m| m.is_empty());
        assert!(outer_empty);
    }

    // -----------------------------------------------------------------------
    // optimizer_stats
    // -----------------------------------------------------------------------

    #[test]
    fn test_optimizer_stats_initial() {
        let opt = default_optimizer();
        let stats = opt.optimizer_stats();
        assert_eq!(stats.total_events, 0);
        assert_eq!(stats.unique_blocks, 0);
        assert_eq!(stats.co_access_pairs, 0);
        assert_eq!(stats.window_size, 0);
    }

    #[test]
    fn test_optimizer_stats_after_events() {
        let mut opt = default_optimizer();
        opt.record_access(event("A", 100, 5, true));
        opt.record_access(event("B", 200, 5, false));
        let stats = opt.optimizer_stats();
        assert_eq!(stats.total_events, 2);
        assert_eq!(stats.unique_blocks, 2);
        assert_eq!(stats.window_size, 2);
        assert!((stats.cache_hit_rate - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_optimizer_stats_co_access_count() {
        let mut opt = default_optimizer();
        opt.record_access(event("A", 100, 5, false));
        opt.record_access(event("B", 200, 5, false));
        let stats = opt.optimizer_stats();
        // A<->B creates 2 directional pairs.
        assert_eq!(stats.co_access_pairs, 2);
    }

    // -----------------------------------------------------------------------
    // Edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_single_event_no_co_access() {
        let mut opt = default_optimizer();
        opt.record_access(event("only", 100, 5, false));
        assert!(opt.co_access.is_empty());
    }

    #[test]
    fn test_same_cid_repeated_no_co_access_with_itself() {
        let mut opt = default_optimizer();
        for i in 0..5u64 {
            opt.record_access(event("X", i * 100, 5, false));
        }
        let has_self_pair = opt.co_access.get("X").is_some_and(|m| m.contains_key("X"));
        assert!(!has_self_pair);
    }

    #[test]
    fn test_hot_blocks_k_larger_than_available() {
        let mut opt = default_optimizer();
        opt.record_access(event("only", 100, 5, false));
        let top = opt.hot_blocks(100);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_top_co_access_pairs_k_larger_than_available() {
        let mut opt = default_optimizer();
        opt.record_access(event("A", 100, 5, false));
        opt.record_access(event("B", 200, 5, false));
        let pairs = opt.top_co_access_pairs(100);
        assert_eq!(pairs.len(), 2); // A->B and B->A
    }

    #[test]
    fn test_decay_on_empty_optimizer() {
        let mut opt = default_optimizer();
        // Should not panic.
        opt.apply_decay();
        assert!(opt.co_access.is_empty());
    }

    #[test]
    fn test_multiple_decays_converge_to_zero() {
        // Use an extreme decay factor (0.01) so that any positive count rounds
        // to 0 after a single application: round(1 * 0.01) = 0 < 1 → removed.
        let cfg = OptimizerConfig {
            pattern_decay_factor: 0.01,
            ..OptimizerConfig::default()
        };
        let mut opt = BlockAccessOptimizer::new(cfg);
        // A single A->B co-access creates directional pairs with count=1.
        opt.record_access(event("A", 100, 5, false));
        opt.record_access(event("B", 200, 5, false));
        // A single decay should prune everything because round(1 * 0.01) = 0.
        opt.apply_decay();
        assert!(
            opt.co_access.is_empty(),
            "all pairs should be pruned after one aggressive decay"
        );
    }
}
