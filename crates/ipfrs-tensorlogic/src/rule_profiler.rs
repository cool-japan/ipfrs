//! Rule Execution Profiler — invocation counts, latencies, hit rates, and hotspot detection.
//!
//! This module provides [`RuleExecutionProfiler`], which tracks per-rule performance
//! metrics across a running inference engine: invocation counts, success/failure rates,
//! cumulative and per-call latencies, and automatic hotspot detection.

use std::collections::HashMap;

// ─── RuleProfile ─────────────────────────────────────────────────────────────

/// Per-rule execution statistics.
#[derive(Debug, Clone)]
pub struct RuleProfile {
    /// Stable numeric identifier for the rule.
    pub rule_id: u64,
    /// Human-readable rule name.
    pub rule_name: String,
    /// Total number of times the rule was attempted.
    pub invocations: u64,
    /// Number of times the rule fired and produced at least one binding.
    pub successes: u64,
    /// Number of times the rule was tried but did not match.
    pub failures: u64,
    /// Cumulative wall-clock time spent in this rule (microseconds).
    pub total_time_us: u64,
    /// Minimum single-invocation time observed (microseconds).
    pub min_time_us: u64,
    /// Maximum single-invocation time observed (microseconds).
    pub max_time_us: u64,
}

impl RuleProfile {
    /// Fraction of invocations that resulted in a successful match.
    ///
    /// Returns `0.0` when `invocations == 0` (avoids division by zero).
    #[inline]
    pub fn success_rate(&self) -> f64 {
        self.successes as f64 / self.invocations.max(1) as f64
    }

    /// Average time per invocation in microseconds.
    ///
    /// Returns `0.0` when `invocations == 0`.
    #[inline]
    pub fn avg_time_us(&self) -> f64 {
        self.total_time_us as f64 / self.invocations.max(1) as f64
    }

    /// Returns `true` when the average invocation time exceeds `threshold_us`.
    #[inline]
    pub fn is_hotspot(&self, threshold_us: u64) -> bool {
        self.avg_time_us() > threshold_us as f64
    }
}

// ─── ProfilerStats ───────────────────────────────────────────────────────────

/// Aggregate statistics across all tracked rules.
#[derive(Debug, Clone)]
pub struct ProfilerStats {
    /// Number of distinct rules being tracked.
    pub total_rules: usize,
    /// Sum of invocations across all rules.
    pub total_invocations: u64,
    /// Sum of wall-clock time across all rules (microseconds).
    pub total_time_us: u64,
    /// Number of rules currently classified as hotspots.
    pub hotspot_count: usize,
}

impl ProfilerStats {
    /// Mean success rate across the supplied rule profiles.
    ///
    /// Returns `0.0` when `profiles` is empty.
    pub fn avg_success_rate(&self, profiles: &[RuleProfile]) -> f64 {
        if profiles.is_empty() {
            return 0.0;
        }
        let sum: f64 = profiles.iter().map(|p| p.success_rate()).sum();
        sum / profiles.len() as f64
    }
}

// ─── RuleExecutionProfiler ───────────────────────────────────────────────────

/// Profiles individual rule execution performance.
///
/// Tracks invocation counts, latencies, hit rates, and hotspot detection for
/// every rule that passes through [`record_invocation`](RuleExecutionProfiler::record_invocation).
///
/// # Example
///
/// ```
/// use ipfrs_tensorlogic::rule_profiler::RuleExecutionProfiler;
///
/// let mut profiler = RuleExecutionProfiler::new(1_000);
/// profiler.record_invocation(1, "ancestor", 500, true);
/// profiler.record_invocation(1, "ancestor", 1_500, false);
///
/// let hotspots = profiler.hotspots();
/// // avg = (500 + 1500) / 2 = 1000, threshold = 1000 → NOT > threshold
/// assert!(hotspots.is_empty());
/// ```
pub struct RuleExecutionProfiler {
    /// Per-rule profiles indexed by `rule_id`.
    pub profiles: HashMap<u64, RuleProfile>,
    /// Average time above which a rule is considered a hotspot (microseconds).
    pub hotspot_threshold_us: u64,
}

impl RuleExecutionProfiler {
    /// Creates a new profiler with the specified hotspot threshold.
    ///
    /// A rule is a hotspot when its `avg_time_us()` **strictly exceeds**
    /// `hotspot_threshold_us`.
    pub fn new(hotspot_threshold_us: u64) -> Self {
        Self {
            profiles: HashMap::new(),
            hotspot_threshold_us,
        }
    }

    /// Records a single rule invocation.
    ///
    /// # Parameters
    ///
    /// - `rule_id`   — stable numeric identifier for the rule
    /// - `rule_name` — human-readable label (used only on first insertion)
    /// - `time_us`   — wall-clock duration of this invocation in microseconds
    /// - `success`   — `true` if the rule produced at least one binding
    pub fn record_invocation(
        &mut self,
        rule_id: u64,
        rule_name: &str,
        time_us: u64,
        success: bool,
    ) {
        let profile = self.profiles.entry(rule_id).or_insert_with(|| RuleProfile {
            rule_id,
            rule_name: rule_name.to_owned(),
            invocations: 0,
            successes: 0,
            failures: 0,
            total_time_us: 0,
            // Sentinel: will be lowered to the first observed value on the first update.
            min_time_us: u64::MAX,
            max_time_us: 0,
        });

        profile.invocations += 1;
        if success {
            profile.successes += 1;
        } else {
            profile.failures += 1;
        }
        profile.total_time_us += time_us;
        profile.min_time_us = profile.min_time_us.min(time_us);
        profile.max_time_us = profile.max_time_us.max(time_us);
    }

    /// Returns all rules whose average invocation time exceeds the hotspot
    /// threshold, sorted by `avg_time_us` in **descending** order.
    pub fn hotspots(&self) -> Vec<&RuleProfile> {
        let threshold = self.hotspot_threshold_us;
        let mut result: Vec<&RuleProfile> = self
            .profiles
            .values()
            .filter(|p| p.is_hotspot(threshold))
            .collect();
        result.sort_by(|a, b| {
            b.avg_time_us()
                .partial_cmp(&a.avg_time_us())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        result
    }

    /// Returns up to `n` rules ordered by **total invocation count** (highest first).
    pub fn top_rules_by_invocations(&self, n: usize) -> Vec<&RuleProfile> {
        let mut result: Vec<&RuleProfile> = self.profiles.values().collect();
        result.sort_by_key(|b| std::cmp::Reverse(b.invocations));
        result.truncate(n);
        result
    }

    /// Returns up to `n` rules ordered by **average time per invocation** (highest first).
    pub fn slowest_rules(&self, n: usize) -> Vec<&RuleProfile> {
        let mut result: Vec<&RuleProfile> = self.profiles.values().collect();
        result.sort_by(|a, b| {
            b.avg_time_us()
                .partial_cmp(&a.avg_time_us())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        result.truncate(n);
        result
    }

    /// Removes the profile for `rule_id`.
    ///
    /// Returns `true` if the rule was present and has been removed, `false`
    /// if no profile existed for that id.
    pub fn reset_rule(&mut self, rule_id: u64) -> bool {
        self.profiles.remove(&rule_id).is_some()
    }

    /// Returns aggregate statistics for all currently tracked rules.
    pub fn stats(&self) -> ProfilerStats {
        let threshold = self.hotspot_threshold_us;
        let total_invocations = self.profiles.values().map(|p| p.invocations).sum();
        let total_time_us = self.profiles.values().map(|p| p.total_time_us).sum();
        let hotspot_count = self
            .profiles
            .values()
            .filter(|p| p.is_hotspot(threshold))
            .count();

        ProfilerStats {
            total_rules: self.profiles.len(),
            total_invocations,
            total_time_us,
            hotspot_count,
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ──────────────────────────────────────────────────────────────

    fn make_profiler() -> RuleExecutionProfiler {
        RuleExecutionProfiler::new(1_000)
    }

    // ── test 1: record_invocation creates a profile ───────────────────────────
    #[test]
    fn test_record_creates_profile() {
        let mut p = make_profiler();
        p.record_invocation(42, "ancestor", 200, true);
        assert!(p.profiles.contains_key(&42));
        let prof = &p.profiles[&42];
        assert_eq!(prof.rule_id, 42);
        assert_eq!(prof.rule_name, "ancestor");
    }

    // ── test 2: invocation count increments correctly ─────────────────────────
    #[test]
    fn test_invocation_count() {
        let mut p = make_profiler();
        p.record_invocation(1, "rule_a", 100, true);
        p.record_invocation(1, "rule_a", 200, false);
        p.record_invocation(1, "rule_a", 300, true);
        assert_eq!(p.profiles[&1].invocations, 3);
    }

    // ── test 3: success tracking ──────────────────────────────────────────────
    #[test]
    fn test_success_tracking() {
        let mut p = make_profiler();
        p.record_invocation(2, "rule_b", 100, true);
        p.record_invocation(2, "rule_b", 100, true);
        p.record_invocation(2, "rule_b", 100, false);
        assert_eq!(p.profiles[&2].successes, 2);
    }

    // ── test 4: failure tracking ──────────────────────────────────────────────
    #[test]
    fn test_failure_tracking() {
        let mut p = make_profiler();
        p.record_invocation(3, "rule_c", 50, false);
        p.record_invocation(3, "rule_c", 50, false);
        assert_eq!(p.profiles[&3].failures, 2);
    }

    // ── test 5: min time update ───────────────────────────────────────────────
    #[test]
    fn test_min_time_update() {
        let mut p = make_profiler();
        p.record_invocation(4, "rule_d", 500, true);
        p.record_invocation(4, "rule_d", 100, true);
        p.record_invocation(4, "rule_d", 300, false);
        assert_eq!(p.profiles[&4].min_time_us, 100);
    }

    // ── test 6: max time update ───────────────────────────────────────────────
    #[test]
    fn test_max_time_update() {
        let mut p = make_profiler();
        p.record_invocation(5, "rule_e", 200, true);
        p.record_invocation(5, "rule_e", 800, false);
        p.record_invocation(5, "rule_e", 50, true);
        assert_eq!(p.profiles[&5].max_time_us, 800);
    }

    // ── test 7: min_time_us initialized to u64::MAX then lowered ─────────────
    #[test]
    fn test_min_time_initialized_correctly() {
        let mut p = make_profiler();
        // First call — min must equal exactly the first time_us.
        p.record_invocation(6, "rule_f", 777, true);
        assert_eq!(p.profiles[&6].min_time_us, 777);
    }

    // ── test 8: avg_time_us correctness ──────────────────────────────────────
    #[test]
    fn test_avg_time_us() {
        let mut p = make_profiler();
        p.record_invocation(7, "rule_g", 200, true);
        p.record_invocation(7, "rule_g", 400, false);
        // avg = 600 / 2 = 300.0
        let avg = p.profiles[&7].avg_time_us();
        assert!((avg - 300.0).abs() < f64::EPSILON);
    }

    // ── test 9: avg_time_us with zero invocations ─────────────────────────────
    #[test]
    fn test_avg_time_us_zero_invocations() {
        let prof = RuleProfile {
            rule_id: 99,
            rule_name: "ghost".to_owned(),
            invocations: 0,
            successes: 0,
            failures: 0,
            total_time_us: 0,
            min_time_us: u64::MAX,
            max_time_us: 0,
        };
        assert_eq!(prof.avg_time_us(), 0.0);
    }

    // ── test 10: success_rate correctness ─────────────────────────────────────
    #[test]
    fn test_success_rate() {
        let mut p = make_profiler();
        p.record_invocation(8, "rule_h", 100, true);
        p.record_invocation(8, "rule_h", 100, false);
        p.record_invocation(8, "rule_h", 100, false);
        p.record_invocation(8, "rule_h", 100, true);
        // 2 successes / 4 = 0.5
        let rate = p.profiles[&8].success_rate();
        assert!((rate - 0.5).abs() < f64::EPSILON);
    }

    // ── test 11: is_hotspot ───────────────────────────────────────────────────
    #[test]
    fn test_is_hotspot() {
        let mut p = make_profiler(); // threshold = 1000
                                     // avg = 2000 → hotspot
        p.record_invocation(9, "slow_rule", 2_000, true);
        assert!(p.profiles[&9].is_hotspot(1_000));
        // avg = 500 → not a hotspot
        let mut p2 = make_profiler();
        p2.record_invocation(10, "fast_rule", 500, true);
        assert!(!p2.profiles[&10].is_hotspot(1_000));
    }

    // ── test 12: hotspots returns correct rules sorted desc ───────────────────
    #[test]
    fn test_hotspots_sorted_desc() {
        let mut p = RuleExecutionProfiler::new(500);
        // avg = 1500 (hotspot)
        p.record_invocation(1, "hot_a", 1_500, true);
        // avg = 2500 (hotspot, slower)
        p.record_invocation(2, "hot_b", 2_500, true);
        // avg = 300 (not hotspot)
        p.record_invocation(3, "cold", 300, false);

        let hs = p.hotspots();
        assert_eq!(hs.len(), 2);
        // Sorted descending by avg_time_us: hot_b (2500) first
        assert_eq!(hs[0].rule_id, 2);
        assert_eq!(hs[1].rule_id, 1);
    }

    // ── test 13: top_rules_by_invocations ────────────────────────────────────
    #[test]
    fn test_top_rules_by_invocations() {
        let mut p = make_profiler();
        for _ in 0..5 {
            p.record_invocation(1, "frequent", 10, true);
        }
        for _ in 0..2 {
            p.record_invocation(2, "infrequent", 10, true);
        }
        for _ in 0..8 {
            p.record_invocation(3, "most_frequent", 10, true);
        }

        let top = p.top_rules_by_invocations(2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].rule_id, 3); // 8 invocations
        assert_eq!(top[1].rule_id, 1); // 5 invocations
    }

    // ── test 14: slowest_rules ────────────────────────────────────────────────
    #[test]
    fn test_slowest_rules() {
        let mut p = make_profiler();
        p.record_invocation(1, "fast", 100, true);
        p.record_invocation(2, "medium", 500, true);
        p.record_invocation(3, "slow", 3_000, true);

        let slowest = p.slowest_rules(2);
        assert_eq!(slowest.len(), 2);
        assert_eq!(slowest[0].rule_id, 3);
        assert_eq!(slowest[1].rule_id, 2);
    }

    // ── test 15: reset_rule returns true when rule existed ────────────────────
    #[test]
    fn test_reset_rule_existing() {
        let mut p = make_profiler();
        p.record_invocation(11, "rule_to_reset", 200, true);
        let removed = p.reset_rule(11);
        assert!(removed);
        assert!(!p.profiles.contains_key(&11));
    }

    // ── test 16: reset_rule returns false when rule absent ────────────────────
    #[test]
    fn test_reset_rule_absent() {
        let mut p = make_profiler();
        assert!(!p.reset_rule(999));
    }

    // ── test 17: stats.total_invocations ─────────────────────────────────────
    #[test]
    fn test_stats_total_invocations() {
        let mut p = make_profiler();
        p.record_invocation(1, "r1", 100, true);
        p.record_invocation(1, "r1", 200, false);
        p.record_invocation(2, "r2", 300, true);
        let stats = p.stats();
        assert_eq!(stats.total_invocations, 3);
        assert_eq!(stats.total_rules, 2);
    }

    // ── test 18: stats.total_time_us ─────────────────────────────────────────
    #[test]
    fn test_stats_total_time_us() {
        let mut p = make_profiler();
        p.record_invocation(1, "r1", 400, true);
        p.record_invocation(2, "r2", 600, false);
        let stats = p.stats();
        assert_eq!(stats.total_time_us, 1_000);
    }

    // ── test 19: stats.hotspot_count ─────────────────────────────────────────
    #[test]
    fn test_stats_hotspot_count() {
        let mut p = RuleExecutionProfiler::new(1_000);
        p.record_invocation(1, "slow", 2_000, true); // hotspot
        p.record_invocation(2, "fast", 500, true); // not hotspot
        let stats = p.stats();
        assert_eq!(stats.hotspot_count, 1);
    }

    // ── test 20: avg_success_rate ─────────────────────────────────────────────
    #[test]
    fn test_avg_success_rate() {
        let mut p = make_profiler();
        // rule 1: 1/2 = 0.5
        p.record_invocation(1, "r1", 100, true);
        p.record_invocation(1, "r1", 100, false);
        // rule 2: 2/2 = 1.0
        p.record_invocation(2, "r2", 100, true);
        p.record_invocation(2, "r2", 100, true);

        let profiles: Vec<RuleProfile> = p.profiles.values().cloned().collect();
        let stats = p.stats();
        // avg = (0.5 + 1.0) / 2 = 0.75
        let avg_rate = stats.avg_success_rate(&profiles);
        assert!((avg_rate - 0.75).abs() < 1e-10);
    }

    // ── test 21: avg_success_rate with empty slice ────────────────────────────
    #[test]
    fn test_avg_success_rate_empty() {
        let p = make_profiler();
        let stats = p.stats();
        assert_eq!(stats.avg_success_rate(&[]), 0.0);
    }

    // ── test 22: multiple rules isolated from each other ─────────────────────
    #[test]
    fn test_multiple_rules_isolated() {
        let mut p = make_profiler();
        p.record_invocation(10, "rule_x", 1_000, true);
        p.record_invocation(20, "rule_y", 2_000, false);

        assert_eq!(p.profiles[&10].invocations, 1);
        assert_eq!(p.profiles[&20].invocations, 1);
        assert_eq!(p.profiles[&10].failures, 0);
        assert_eq!(p.profiles[&20].successes, 0);
    }

    // ── test 23: total_time_us accumulates across invocations ─────────────────
    #[test]
    fn test_total_time_us_accumulation() {
        let mut p = make_profiler();
        p.record_invocation(5, "acc_rule", 100, true);
        p.record_invocation(5, "acc_rule", 200, false);
        p.record_invocation(5, "acc_rule", 700, true);
        assert_eq!(p.profiles[&5].total_time_us, 1_000);
    }
}
