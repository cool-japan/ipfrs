//! StoragePrefetchEngine — Intelligent prefetch engine that learns access patterns
//! and pre-warms the cache by predicting future access.
//!
//! ## Overview
//!
//! The prefetch engine observes storage access events, maintains a sliding window of
//! recent accesses, and identifies co-access pairs (CIDs frequently accessed together
//! within a configurable time window).  When a CID is accessed the engine scores all
//! known co-access partners using a recency-weighted count and returns up to
//! `max_prefetch_hints` sorted hints to the caller.
//!
//! In addition to co-access correlation the engine detects per-CID access patterns
//! (`Sequential`, `Repeated`, `Random`, `Strided`, `Unknown`) to help callers adapt
//! their prefetch depth.

use std::cmp::Reverse;
use std::collections::{HashMap, VecDeque};

// ── Configuration ────────────────────────────────────────────────────────────

/// Configuration for [`StoragePrefetchEngine`].
#[derive(Debug, Clone)]
pub struct PeConfig {
    /// Time window (milliseconds) within which two accesses are considered co-located.
    pub coaccess_window_ms: u64,
    /// Maximum number of co-access pairs stored in memory.
    pub max_pair_history: usize,
    /// Minimum times a pair must have co-occurred before it is emitted as a hint.
    pub min_coaccess_count: u32,
    /// Maximum number of prefetch hints returned per `record_access` call.
    pub max_prefetch_hints: usize,
    /// Number of recent per-CID accesses used for pattern detection.
    pub pattern_window: usize,
}

impl Default for PeConfig {
    fn default() -> Self {
        Self {
            coaccess_window_ms: 5_000,
            max_pair_history: 10_000,
            min_coaccess_count: 2,
            max_prefetch_hints: 20,
            pattern_window: 20,
        }
    }
}

// ── Access primitives ─────────────────────────────────────────────────────────

/// The kind of storage operation that was performed.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PeAccessType {
    /// Block was read.
    Read,
    /// Block was written.
    Write,
    /// Block was deleted.
    Delete,
}

/// A single storage access event recorded by the prefetch engine.
#[derive(Debug, Clone)]
pub struct PeAccessEvent {
    /// The CID of the accessed block.
    pub cid: String,
    /// Unix timestamp in milliseconds at which the access occurred.
    pub timestamp: u64,
    /// The type of access.
    pub access_type: PeAccessType,
}

// ── Co-access tracking ────────────────────────────────────────────────────────

/// A pair of CIDs that have been observed to be accessed within the same time window.
#[derive(Debug, Clone)]
pub struct CoAccessPair {
    /// The lexicographically smaller CID.
    pub cid_a: String,
    /// The lexicographically larger CID.
    pub cid_b: String,
    /// Number of times this pair has been co-accessed.
    pub coaccess_count: u32,
    /// Unix timestamp (ms) of the most recent co-access.
    pub last_seen: u64,
}

// ── Prefetch hints ────────────────────────────────────────────────────────────

/// A predicted-access hint produced by the engine.
#[derive(Debug, Clone)]
pub struct PePrefetchHint {
    /// The CID predicted to be accessed next.
    pub cid: String,
    /// Normalised priority in [0, 1]; higher means more likely to be needed.
    pub priority: f64,
    /// Human-readable justification for this hint.
    pub reason: String,
}

// ── Pattern detection ─────────────────────────────────────────────────────────

/// The access pattern detected for a single CID.
#[derive(Debug, Clone, PartialEq)]
pub enum PeAccessPattern {
    /// Accesses are evenly spaced in time (inter-access std-dev < 20 % of mean).
    Sequential,
    /// Accesses appear fully random (no other pattern detected).
    Random,
    /// Accesses have a consistent stride `stride` in their ordering within the window.
    Strided { stride: usize },
    /// The CID has been accessed ≥ 3 times in the recent pattern window.
    Repeated,
    /// Not enough data to determine a pattern.
    Unknown,
}

// ── Statistics ────────────────────────────────────────────────────────────────

/// Aggregate statistics produced by [`StoragePrefetchEngine::stats`].
#[derive(Debug, Clone)]
pub struct PePrefetchStats {
    /// Total number of access events recorded since creation.
    pub total_events: u64,
    /// Number of unique co-access pairs currently tracked.
    pub total_pairs: usize,
    /// Mean co-access count across all tracked pairs.
    pub avg_coaccess_count: f64,
    /// Textual description of the dominant pattern observed globally.
    pub top_pattern: String,
    /// Total number of prefetch hints generated since creation.
    pub hints_generated: u64,
}

// ── Main engine ───────────────────────────────────────────────────────────────

/// Intelligent prefetch engine: records access events, identifies co-access
/// pairs, detects per-CID access patterns, and generates prioritised prefetch
/// hints.
pub struct StoragePrefetchEngine {
    /// Engine configuration (immutable after construction).
    pub config: PeConfig,
    /// Ordered history of recent access events (front = oldest).
    pub access_history: VecDeque<PeAccessEvent>,
    /// Co-access pairs keyed by `"<cid_a>:<cid_b>"` (cid_a < cid_b).
    pub coaccesses: HashMap<String, CoAccessPair>,
    /// CIDs seen in recent accesses (last `config.pattern_window` entries).
    pub recent_cids: VecDeque<String>,
    /// Per-CID list of recent access timestamps (last `pattern_window` entries).
    pub pattern_state: HashMap<String, Vec<u64>>,

    // Internal counters.
    total_events: u64,
    hints_generated: u64,
}

impl StoragePrefetchEngine {
    // ── Construction ─────────────────────────────────────────────────────────

    /// Create a new engine with the given configuration.
    pub fn new(config: PeConfig) -> Self {
        Self {
            config,
            access_history: VecDeque::new(),
            coaccesses: HashMap::new(),
            recent_cids: VecDeque::new(),
            pattern_state: HashMap::new(),
            total_events: 0,
            hints_generated: 0,
        }
    }

    // ── Event recording ───────────────────────────────────────────────────────

    /// Record an access event.
    ///
    /// Side effects:
    /// 1. Appends `event` to `access_history`.
    /// 2. Updates co-access pairs for every event within `coaccess_window_ms`.
    /// 3. Updates `pattern_state` for the accessed CID.
    /// 4. Returns prefetch hints for the accessed CID.
    pub fn record_access(&mut self, event: PeAccessEvent) -> Vec<PePrefetchHint> {
        let now = event.timestamp;
        let cid = event.cid.clone();

        // -- Update pattern state -------------------------------------------------
        {
            let ts_vec = self.pattern_state.entry(cid.clone()).or_default();
            ts_vec.push(now);
            let window = self.config.pattern_window;
            if ts_vec.len() > window {
                let drain_count = ts_vec.len() - window;
                ts_vec.drain(..drain_count);
            }
        }

        // -- Append to history ---------------------------------------------------
        self.access_history.push_back(event);
        self.total_events += 1;

        // -- Update co-access pairs for events within the window -----------------
        self.update_coaccesses(now, &cid);

        // -- Trim history to avoid unbounded growth ------------------------------
        // We keep events that are still within the coaccess window plus a safety
        // margin (×2) so that the window look-back always has enough history.
        let cutoff = now.saturating_sub(self.config.coaccess_window_ms * 2);
        while self
            .access_history
            .front()
            .map(|e| e.timestamp < cutoff)
            .unwrap_or(false)
        {
            self.access_history.pop_front();
        }

        // -- Update recent_cids deque --------------------------------------------
        self.recent_cids.push_back(cid.clone());
        let pw = self.config.pattern_window;
        while self.recent_cids.len() > pw {
            self.recent_cids.pop_front();
        }

        // -- Enforce max_pair_history --------------------------------------------
        if self.coaccesses.len() > self.config.max_pair_history {
            self.evict_lowest_pairs();
        }

        // -- Generate hints for the just-accessed CID ----------------------------
        let hints = self.generate_hints(&cid, now);
        self.hints_generated += hints.len() as u64;
        hints
    }

    /// Generate prefetch hints for `cid` at the given `now` timestamp.
    ///
    /// Returns up to `config.max_prefetch_hints` hints sorted by descending
    /// `coaccess_count * recency_weight`.
    pub fn generate_hints(&self, cid: &str, now: u64) -> Vec<PePrefetchHint> {
        let max = self.config.max_prefetch_hints;
        let min_count = self.config.min_coaccess_count;

        // Collect (partner_cid, score, pair_ref) for all pairs involving `cid`.
        let mut scored: Vec<(String, f64, u32)> = self
            .coaccesses
            .values()
            .filter(|p| p.coaccess_count >= min_count && (p.cid_a == cid || p.cid_b == cid))
            .map(|p| {
                let partner = if p.cid_a == cid {
                    p.cid_b.clone()
                } else {
                    p.cid_a.clone()
                };
                let age_ms = now.saturating_sub(p.last_seen) as f64;
                let recency_weight = (-age_ms / 60_000.0_f64).exp();
                let raw_score = p.coaccess_count as f64 * recency_weight;
                (partner, raw_score, p.coaccess_count)
            })
            .collect();

        // Sort descending by score.
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(max);

        scored
            .into_iter()
            .map(|(partner, raw_score, count)| {
                let priority = (raw_score / 10.0_f64).min(1.0_f64);
                PePrefetchHint {
                    cid: partner,
                    priority,
                    reason: format!("co-accessed {} time(s) with {}", count, cid),
                }
            })
            .collect()
    }

    // ── Pattern detection ─────────────────────────────────────────────────────

    /// Detect the access pattern for a given CID using the last `pattern_window`
    /// recorded timestamps.
    ///
    /// Decision logic (in order):
    /// 1. `Repeated`   — ≥ 3 timestamps in the pattern state.
    /// 2. `Sequential` — inter-access std-dev < 20 % of the mean interval.
    /// 3. `Unknown`    — otherwise.
    pub fn detect_pattern(&self, cid: &str) -> PeAccessPattern {
        let ts = match self.pattern_state.get(cid) {
            Some(v) if !v.is_empty() => v,
            _ => return PeAccessPattern::Unknown,
        };

        if ts.len() >= 3 {
            // First check: Repeated
            // Then try Sequential if we have ≥ 2 intervals.
            if ts.len() >= 2 {
                let intervals: Vec<f64> = ts
                    .windows(2)
                    .map(|w| (w[1].saturating_sub(w[0])) as f64)
                    .collect();

                if !intervals.is_empty() {
                    let mean = intervals.iter().sum::<f64>() / intervals.len() as f64;
                    if mean > 0.0 {
                        let variance = intervals.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                            / intervals.len() as f64;
                        let std_dev = variance.sqrt();
                        if std_dev < 0.2 * mean {
                            return PeAccessPattern::Sequential;
                        }
                    }
                }
            }
            return PeAccessPattern::Repeated;
        }

        PeAccessPattern::Unknown
    }

    // ── Co-access helpers ─────────────────────────────────────────────────────

    /// Return the top `n` co-access pairs involving `cid`, sorted by
    /// `coaccess_count` descending.
    pub fn top_coaccessed<'a>(&'a self, cid: &str, n: usize) -> Vec<&'a CoAccessPair> {
        let mut pairs: Vec<&CoAccessPair> = self
            .coaccesses
            .values()
            .filter(|p| p.cid_a == cid || p.cid_b == cid)
            .collect();

        pairs.sort_by_key(|p| Reverse(p.coaccess_count));
        pairs.truncate(n);
        pairs
    }

    /// Remove co-access pairs not seen within the last `max_age_ms` milliseconds.
    ///
    /// Returns the number of pairs removed.
    pub fn evict_stale_pairs(&mut self, max_age_ms: u64, now: u64) -> usize {
        let cutoff = now.saturating_sub(max_age_ms);
        let before = self.coaccesses.len();
        self.coaccesses.retain(|_, p| p.last_seen >= cutoff);
        before - self.coaccesses.len()
    }

    // ── Simple accessors ──────────────────────────────────────────────────────

    /// Return the total number of co-access pairs currently tracked.
    pub fn total_pairs(&self) -> usize {
        self.coaccesses.len()
    }

    /// Return the number of events in the access history buffer.
    pub fn history_len(&self) -> usize {
        self.access_history.len()
    }

    /// Return the top `n` most-accessed CIDs by total appearance in
    /// `access_history`, as `(cid, count)` pairs sorted by count descending.
    pub fn most_accessed_cids(&self, n: usize) -> Vec<(String, usize)> {
        let mut counts: HashMap<&str, usize> = HashMap::new();
        for ev in &self.access_history {
            *counts.entry(ev.cid.as_str()).or_insert(0) += 1;
        }
        let mut vec: Vec<(String, usize)> =
            counts.into_iter().map(|(k, v)| (k.to_owned(), v)).collect();
        vec.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        vec.truncate(n);
        vec
    }

    /// Return aggregate statistics.
    pub fn stats(&self) -> PePrefetchStats {
        let total_pairs = self.coaccesses.len();
        let avg_coaccess_count = if total_pairs == 0 {
            0.0
        } else {
            self.coaccesses
                .values()
                .map(|p| p.coaccess_count as f64)
                .sum::<f64>()
                / total_pairs as f64
        };

        // Determine dominant pattern by sampling the pattern_state.
        let top_pattern = self.dominant_pattern();

        PePrefetchStats {
            total_events: self.total_events,
            total_pairs,
            avg_coaccess_count,
            top_pattern,
            hints_generated: self.hints_generated,
        }
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /// Update co-access pairs: for every event in `access_history` whose
    /// timestamp is within `coaccess_window_ms` of `now`, register a co-access
    /// with `new_cid`.
    fn update_coaccesses(&mut self, now: u64, new_cid: &str) {
        let window = self.config.coaccess_window_ms;
        let cutoff = now.saturating_sub(window);

        // Collect partner CIDs within the window (excluding the new event itself
        // which we just pushed — use `len-1` to skip the last element).
        let history_len = self.access_history.len();
        let slice_len = if history_len > 0 { history_len - 1 } else { 0 };

        let partners: Vec<String> = self
            .access_history
            .iter()
            .take(slice_len)
            .filter(|e| e.timestamp >= cutoff && e.cid != new_cid)
            .map(|e| e.cid.clone())
            .collect();

        for partner in partners {
            let key = coacccess_key(&partner, new_cid);
            let (cid_a, cid_b) = ordered_pair(&partner, new_cid);
            let entry = self.coaccesses.entry(key).or_insert_with(|| CoAccessPair {
                cid_a: cid_a.to_owned(),
                cid_b: cid_b.to_owned(),
                coaccess_count: 0,
                last_seen: 0,
            });
            entry.coaccess_count = entry.coaccess_count.saturating_add(1);
            if now > entry.last_seen {
                entry.last_seen = now;
            }
        }
    }

    /// Evict the lowest-count pairs when the map exceeds `max_pair_history`.
    fn evict_lowest_pairs(&mut self) {
        let max = self.config.max_pair_history;
        if self.coaccesses.len() <= max {
            return;
        }
        // Collect keys sorted by count ascending; drop the bottom half excess.
        let excess = self.coaccesses.len() - max;
        let mut keyed: Vec<(String, u32)> = self
            .coaccesses
            .iter()
            .map(|(k, v)| (k.clone(), v.coaccess_count))
            .collect();
        keyed.sort_by_key(|(_, c)| *c);
        for (key, _) in keyed.into_iter().take(excess) {
            self.coaccesses.remove(&key);
        }
    }

    /// Determine the dominant pattern across all tracked CIDs.
    fn dominant_pattern(&self) -> String {
        let mut repeated = 0usize;
        let mut sequential = 0usize;
        let mut unknown = 0usize;

        for cid in self.pattern_state.keys() {
            match self.detect_pattern(cid) {
                PeAccessPattern::Repeated => repeated += 1,
                PeAccessPattern::Sequential => sequential += 1,
                PeAccessPattern::Unknown
                | PeAccessPattern::Random
                | PeAccessPattern::Strided { .. } => unknown += 1,
            }
        }

        if repeated >= sequential && repeated >= unknown {
            "Repeated".to_owned()
        } else if sequential >= unknown {
            "Sequential".to_owned()
        } else {
            "Unknown".to_owned()
        }
    }
}

// ── Free functions ────────────────────────────────────────────────────────────

/// Build the canonical map key for a co-access pair.
///
/// The key is `"<smaller>:<larger>"` so that each unordered pair has exactly
/// one key regardless of which CID triggered the update.
fn coacccess_key(a: &str, b: &str) -> String {
    if a <= b {
        format!("{}:{}", a, b)
    } else {
        format!("{}:{}", b, a)
    }
}

/// Return `(cid_a, cid_b)` in lexicographic order.
fn ordered_pair<'a>(a: &'a str, b: &'a str) -> (&'a str, &'a str) {
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::{
        CoAccessPair, PeAccessEvent, PeAccessPattern, PeAccessType, PeConfig, PePrefetchHint,
        PePrefetchStats, StoragePrefetchEngine,
    };

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn default_engine() -> StoragePrefetchEngine {
        StoragePrefetchEngine::new(PeConfig::default())
    }

    fn make_event(cid: &str, ts: u64) -> PeAccessEvent {
        PeAccessEvent {
            cid: cid.to_owned(),
            timestamp: ts,
            access_type: PeAccessType::Read,
        }
    }

    fn make_write_event(cid: &str, ts: u64) -> PeAccessEvent {
        PeAccessEvent {
            cid: cid.to_owned(),
            timestamp: ts,
            access_type: PeAccessType::Write,
        }
    }

    fn make_delete_event(cid: &str, ts: u64) -> PeAccessEvent {
        PeAccessEvent {
            cid: cid.to_owned(),
            timestamp: ts,
            access_type: PeAccessType::Delete,
        }
    }

    // ── Construction tests ────────────────────────────────────────────────────

    #[test]
    fn test_new_default_config() {
        let engine = default_engine();
        assert_eq!(engine.config.coaccess_window_ms, 5_000);
        assert_eq!(engine.config.max_pair_history, 10_000);
        assert_eq!(engine.config.min_coaccess_count, 2);
        assert_eq!(engine.config.max_prefetch_hints, 20);
        assert_eq!(engine.config.pattern_window, 20);
    }

    #[test]
    fn test_new_custom_config() {
        let cfg = PeConfig {
            coaccess_window_ms: 1_000,
            max_pair_history: 100,
            min_coaccess_count: 3,
            max_prefetch_hints: 5,
            pattern_window: 10,
        };
        let engine = StoragePrefetchEngine::new(cfg.clone());
        assert_eq!(engine.config.coaccess_window_ms, 1_000);
        assert_eq!(engine.config.max_pair_history, 100);
        assert_eq!(engine.config.min_coaccess_count, 3);
        assert_eq!(engine.config.max_prefetch_hints, 5);
        assert_eq!(engine.config.pattern_window, 10);
    }

    #[test]
    fn test_initial_state() {
        let engine = default_engine();
        assert_eq!(engine.total_pairs(), 0);
        assert_eq!(engine.history_len(), 0);
        assert!(engine.most_accessed_cids(10).is_empty());
    }

    // ── record_access / history tests ─────────────────────────────────────────

    #[test]
    fn test_record_single_event() {
        let mut engine = default_engine();
        engine.record_access(make_event("Qm1", 1_000));
        assert_eq!(engine.history_len(), 1);
    }

    #[test]
    fn test_history_len_grows() {
        let mut engine = default_engine();
        for i in 0u64..5 {
            engine.record_access(make_event(&format!("Qm{}", i), i * 100));
        }
        assert_eq!(engine.history_len(), 5);
    }

    #[test]
    fn test_record_returns_empty_hints_when_no_pairs() {
        let mut engine = default_engine();
        let hints = engine.record_access(make_event("QmA", 0));
        assert!(hints.is_empty(), "no pairs yet → no hints");
    }

    #[test]
    fn test_write_event_recorded() {
        let mut engine = default_engine();
        engine.record_access(make_write_event("QmW", 500));
        assert_eq!(engine.history_len(), 1);
    }

    #[test]
    fn test_delete_event_recorded() {
        let mut engine = default_engine();
        engine.record_access(make_delete_event("QmD", 600));
        assert_eq!(engine.history_len(), 1);
    }

    // ── Co-access pair tests ──────────────────────────────────────────────────

    #[test]
    fn test_coaccess_pair_formed_within_window() {
        let mut engine = default_engine();
        engine.record_access(make_event("QmA", 1_000));
        engine.record_access(make_event("QmB", 1_500)); // 500 ms later, within 5s window
        assert_eq!(engine.total_pairs(), 1);
    }

    #[test]
    fn test_coaccess_pair_not_formed_outside_window() {
        let mut engine = default_engine();
        engine.record_access(make_event("QmA", 1_000));
        engine.record_access(make_event("QmB", 10_000)); // 9 seconds later
        assert_eq!(engine.total_pairs(), 0);
    }

    #[test]
    fn test_coaccess_count_increments() {
        let mut engine = default_engine();
        // Access A and B twice within the window.
        engine.record_access(make_event("QmA", 1_000));
        engine.record_access(make_event("QmB", 1_200));
        engine.record_access(make_event("QmA", 2_000));
        engine.record_access(make_event("QmB", 2_200));
        let pairs = engine.top_coaccessed("QmA", 5);
        assert!(!pairs.is_empty());
        let pair = pairs[0];
        assert!(pair.coaccess_count >= 2, "count={}", pair.coaccess_count);
    }

    #[test]
    fn test_coaccess_key_is_sorted() {
        // Regardless of access order, the key should always be lex-sorted.
        let mut engine = default_engine();
        engine.record_access(make_event("QmZ", 1_000));
        engine.record_access(make_event("QmA", 1_100));
        assert_eq!(engine.total_pairs(), 1);
        let pairs: Vec<&CoAccessPair> = engine.coaccesses.values().collect();
        assert_eq!(pairs[0].cid_a, "QmA");
        assert_eq!(pairs[0].cid_b, "QmZ");
    }

    #[test]
    fn test_top_coaccessed_returns_correct_cids() {
        let mut engine = default_engine();
        // Build 3 pairs for QmA.
        for ts_offset in 0u64..3 {
            engine.record_access(make_event("QmA", 1_000 + ts_offset * 500));
            engine.record_access(make_event(
                &format!("QmX{}", ts_offset),
                1_100 + ts_offset * 500,
            ));
        }
        let top = engine.top_coaccessed("QmA", 10);
        // All three partners should appear.
        assert_eq!(top.len(), 3);
    }

    #[test]
    fn test_top_coaccessed_sorted_desc() {
        let mut engine = default_engine();
        // Create a high-count pair A-B and a low-count pair A-C.
        for i in 0u64..4 {
            engine.record_access(make_event("QmA", 1_000 + i * 300));
            engine.record_access(make_event("QmB", 1_050 + i * 300));
        }
        engine.record_access(make_event("QmA", 10_000));
        engine.record_access(make_event("QmC", 10_050));
        let top = engine.top_coaccessed("QmA", 10);
        assert!(!top.is_empty());
        // First element should have highest count.
        for w in top.windows(2) {
            assert!(w[0].coaccess_count >= w[1].coaccess_count);
        }
    }

    // ── generate_hints tests ──────────────────────────────────────────────────

    #[test]
    fn test_generate_hints_respects_min_count() {
        let cfg = PeConfig {
            min_coaccess_count: 3,
            ..Default::default()
        };
        let mut engine = StoragePrefetchEngine::new(cfg);
        // Create a pair accessed only once.
        engine.record_access(make_event("QmA", 1_000));
        engine.record_access(make_event("QmB", 1_100));
        let hints = engine.generate_hints("QmA", 2_000);
        assert!(hints.is_empty(), "count=1 < min_coaccess_count=3");
    }

    #[test]
    fn test_generate_hints_returns_after_sufficient_coaccesses() {
        let cfg = PeConfig {
            min_coaccess_count: 2,
            ..Default::default()
        };
        let mut engine = StoragePrefetchEngine::new(cfg);
        for i in 0u64..2 {
            engine.record_access(make_event("QmA", 1_000 + i * 200));
            engine.record_access(make_event("QmB", 1_050 + i * 200));
        }
        let hints = engine.generate_hints("QmA", 2_000);
        assert!(!hints.is_empty());
        assert_eq!(hints[0].cid, "QmB");
    }

    #[test]
    fn test_generate_hints_priority_capped_at_one() {
        let mut engine = default_engine();
        // Many co-accesses to push raw score above 10.
        for i in 0u64..20 {
            engine.record_access(make_event("QmA", 1_000 + i * 10));
            engine.record_access(make_event("QmB", 1_005 + i * 10));
        }
        let hints = engine.generate_hints("QmA", 1_500);
        for h in &hints {
            assert!(h.priority <= 1.0, "priority={} > 1.0", h.priority);
        }
    }

    #[test]
    fn test_generate_hints_max_hints_limit() {
        let cfg = PeConfig {
            max_prefetch_hints: 3,
            min_coaccess_count: 1,
            ..Default::default()
        };
        let mut engine = StoragePrefetchEngine::new(cfg);
        for i in 0u64..10 {
            engine.record_access(make_event("QmA", 1_000 + i * 5));
            engine.record_access(make_event(&format!("QmP{}", i), 1_002 + i * 5));
        }
        let hints = engine.generate_hints("QmA", 2_000);
        assert!(hints.len() <= 3, "len={}", hints.len());
    }

    #[test]
    fn test_hint_reason_mentions_cid() {
        let mut engine = default_engine();
        for i in 0u64..2 {
            engine.record_access(make_event("QmA", 1_000 + i * 100));
            engine.record_access(make_event("QmB", 1_050 + i * 100));
        }
        let hints = engine.generate_hints("QmA", 2_000);
        assert!(!hints.is_empty());
        assert!(
            hints[0].reason.contains("QmA"),
            "reason missing trigger CID"
        );
    }

    // ── detect_pattern tests ──────────────────────────────────────────────────

    #[test]
    fn test_detect_pattern_unknown_for_new_cid() {
        let engine = default_engine();
        assert_eq!(engine.detect_pattern("QmNone"), PeAccessPattern::Unknown);
    }

    #[test]
    fn test_detect_pattern_unknown_for_two_events() {
        let mut engine = default_engine();
        engine.record_access(make_event("QmA", 1_000));
        engine.record_access(make_event("QmA", 2_000));
        // Only 2 timestamps — not enough for Repeated (need ≥ 3).
        assert_eq!(engine.detect_pattern("QmA"), PeAccessPattern::Unknown);
    }

    #[test]
    fn test_detect_pattern_repeated_three_accesses() {
        let mut engine = default_engine();
        for ts in [1_000u64, 2_000, 3_000] {
            engine.record_access(make_event("QmR", ts));
        }
        let pattern = engine.detect_pattern("QmR");
        // Evenly spaced → Sequential (std_dev = 0 < 0.2 * 1000).
        assert!(
            matches!(
                pattern,
                PeAccessPattern::Sequential | PeAccessPattern::Repeated
            ),
            "pattern={:?}",
            pattern
        );
    }

    #[test]
    fn test_detect_pattern_sequential_evenly_spaced() {
        let mut engine = default_engine();
        // Access every 1000 ms — perfectly sequential.
        for i in 0u64..5 {
            engine.record_access(make_event("QmS", i * 1_000));
        }
        let pattern = engine.detect_pattern("QmS");
        assert_eq!(
            pattern,
            PeAccessPattern::Sequential,
            "pattern={:?}",
            pattern
        );
    }

    #[test]
    fn test_detect_pattern_repeated_uneven_intervals() {
        let mut engine = default_engine();
        // Timestamps with high variance — should be Repeated (≥ 3 seen) but not Sequential.
        for ts in [100u64, 200, 1_900] {
            engine.record_access(make_event("QmU", ts));
        }
        let pattern = engine.detect_pattern("QmU");
        // std_dev of [100, 1700] = ~800, mean = 900 → std_dev/mean ≈ 0.89 > 0.2.
        assert!(
            matches!(
                pattern,
                PeAccessPattern::Repeated | PeAccessPattern::Unknown
            ),
            "pattern={:?}",
            pattern
        );
    }

    // ── evict_stale_pairs tests ───────────────────────────────────────────────

    #[test]
    fn test_evict_stale_pairs_removes_old() {
        let mut engine = default_engine();
        engine.record_access(make_event("QmA", 1_000));
        engine.record_access(make_event("QmB", 1_100));
        assert_eq!(engine.total_pairs(), 1);

        let removed = engine.evict_stale_pairs(500, 2_000);
        assert_eq!(
            removed, 1,
            "pair at t=1100 is older than 500ms ago from t=2000"
        );
        assert_eq!(engine.total_pairs(), 0);
    }

    #[test]
    fn test_evict_stale_pairs_keeps_fresh() {
        let mut engine = default_engine();
        engine.record_access(make_event("QmA", 1_000));
        engine.record_access(make_event("QmB", 1_100));

        let removed = engine.evict_stale_pairs(5_000, 1_500);
        // last_seen = 1100, cutoff = 1500 - 5000 = 0 (saturating) → pair survives.
        assert_eq!(removed, 0);
        assert_eq!(engine.total_pairs(), 1);
    }

    #[test]
    fn test_evict_stale_returns_count() {
        let mut engine = default_engine();
        for i in 0u64..3 {
            engine.record_access(make_event("QmA", i * 100));
            engine.record_access(make_event(&format!("QmX{}", i), i * 100 + 10));
        }
        let pairs_before = engine.total_pairs();
        let removed = engine.evict_stale_pairs(0, 10_000);
        assert_eq!(removed, pairs_before);
        assert_eq!(engine.total_pairs(), 0);
    }

    // ── most_accessed_cids tests ──────────────────────────────────────────────

    #[test]
    fn test_most_accessed_cids_single() {
        let mut engine = default_engine();
        engine.record_access(make_event("QmA", 1_000));
        let top = engine.most_accessed_cids(5);
        assert_eq!(top.len(), 1);
        assert_eq!(top[0].0, "QmA");
        assert_eq!(top[0].1, 1);
    }

    #[test]
    fn test_most_accessed_cids_counts() {
        let mut engine = default_engine();
        for ts in [1_000u64, 2_000, 3_000] {
            engine.record_access(make_event("QmHot", ts));
        }
        engine.record_access(make_event("QmCool", 4_000));
        let top = engine.most_accessed_cids(2);
        assert_eq!(top[0].0, "QmHot");
        assert_eq!(top[0].1, 3);
    }

    #[test]
    fn test_most_accessed_cids_n_limit() {
        let mut engine = default_engine();
        for i in 0u64..10 {
            engine.record_access(make_event(&format!("Qm{}", i), i * 100));
        }
        let top = engine.most_accessed_cids(3);
        assert_eq!(top.len(), 3);
    }

    #[test]
    fn test_most_accessed_cids_sorted_desc() {
        let mut engine = default_engine();
        engine.record_access(make_event("QmA", 100));
        for ts in [200u64, 300, 400] {
            engine.record_access(make_event("QmB", ts));
        }
        for ts in [500u64, 600] {
            engine.record_access(make_event("QmC", ts));
        }
        let top = engine.most_accessed_cids(10);
        for w in top.windows(2) {
            assert!(w[0].1 >= w[1].1);
        }
    }

    // ── stats tests ───────────────────────────────────────────────────────────

    #[test]
    fn test_stats_initial() {
        let engine = default_engine();
        let s = engine.stats();
        assert_eq!(s.total_events, 0);
        assert_eq!(s.total_pairs, 0);
        assert_eq!(s.avg_coaccess_count, 0.0);
        assert_eq!(s.hints_generated, 0);
    }

    #[test]
    fn test_stats_total_events() {
        let mut engine = default_engine();
        for i in 0u64..7 {
            engine.record_access(make_event(&format!("Qm{}", i), i * 1_000));
        }
        assert_eq!(engine.stats().total_events, 7);
    }

    #[test]
    fn test_stats_hints_generated_tracked() {
        let cfg = PeConfig {
            min_coaccess_count: 1,
            ..Default::default()
        };
        let mut engine = StoragePrefetchEngine::new(cfg);
        for i in 0u64..2 {
            engine.record_access(make_event("QmA", 1_000 + i * 100));
            engine.record_access(make_event("QmB", 1_050 + i * 100));
        }
        // After 4 events with one pair, hints should have been generated.
        let s = engine.stats();
        assert!(
            s.hints_generated > 0,
            "hints_generated={}",
            s.hints_generated
        );
    }

    #[test]
    fn test_stats_avg_coaccess_count() {
        let mut engine = default_engine();
        for i in 0u64..3 {
            engine.record_access(make_event("QmA", 1_000 + i * 100));
            engine.record_access(make_event("QmB", 1_050 + i * 100));
        }
        let s = engine.stats();
        assert!(s.avg_coaccess_count > 0.0);
    }

    #[test]
    fn test_stats_top_pattern_string() {
        let engine = default_engine();
        let s = engine.stats();
        // Unknown / no data → must return a non-empty string.
        assert!(!s.top_pattern.is_empty());
    }

    // ── recency weight tests ──────────────────────────────────────────────────

    #[test]
    fn test_hints_sorted_by_recency() {
        // Two partners for QmA: QmB accessed recently, QmC accessed long ago.
        let cfg = PeConfig {
            min_coaccess_count: 1,
            ..Default::default()
        };
        let mut engine = StoragePrefetchEngine::new(cfg);

        // QmA + QmC co-accessed at t=0
        engine.record_access(make_event("QmA", 0));
        engine.record_access(make_event("QmC", 50));

        // QmA + QmB co-accessed at t=100_000 (recent)
        engine.record_access(make_event("QmA", 100_000));
        engine.record_access(make_event("QmB", 100_050));

        let hints = engine.generate_hints("QmA", 100_100);
        assert!(!hints.is_empty());
        // QmB should rank higher than QmC (more recent).
        if hints.len() >= 2 {
            let pos_b = hints
                .iter()
                .position(|h| h.cid == "QmB")
                .unwrap_or(usize::MAX);
            let pos_c = hints
                .iter()
                .position(|h| h.cid == "QmC")
                .unwrap_or(usize::MAX);
            assert!(
                pos_b < pos_c,
                "QmB should precede QmC; hints={:?}",
                hints.iter().map(|h| &h.cid).collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn test_priority_non_negative() {
        let cfg = PeConfig {
            min_coaccess_count: 1,
            ..Default::default()
        };
        let mut engine = StoragePrefetchEngine::new(cfg);
        for i in 0u64..2 {
            engine.record_access(make_event("QmA", i * 200));
            engine.record_access(make_event("QmB", i * 200 + 10));
        }
        let hints = engine.generate_hints("QmA", 500_000);
        for h in hints {
            assert!(h.priority >= 0.0);
        }
    }

    // ── total_pairs / history_len tests ──────────────────────────────────────

    #[test]
    fn test_total_pairs_after_eviction() {
        let mut engine = default_engine();
        engine.record_access(make_event("QmA", 1_000));
        engine.record_access(make_event("QmB", 1_100));
        assert_eq!(engine.total_pairs(), 1);
        engine.evict_stale_pairs(0, 100_000);
        assert_eq!(engine.total_pairs(), 0);
    }

    #[test]
    fn test_access_type_variants_compile() {
        let _ = PeAccessType::Read;
        let _ = PeAccessType::Write;
        let _ = PeAccessType::Delete;
    }

    #[test]
    fn test_pe_prefetch_hint_fields() {
        let hint = PePrefetchHint {
            cid: "QmX".to_owned(),
            priority: 0.75,
            reason: "test".to_owned(),
        };
        assert_eq!(hint.cid, "QmX");
        assert!((hint.priority - 0.75).abs() < 1e-9);
    }

    #[test]
    fn test_pe_prefetch_stats_fields() {
        let s = PePrefetchStats {
            total_events: 5,
            total_pairs: 2,
            avg_coaccess_count: 1.5,
            top_pattern: "Repeated".to_owned(),
            hints_generated: 10,
        };
        assert_eq!(s.total_events, 5);
        assert_eq!(s.total_pairs, 2);
    }

    // ── edge-case tests ───────────────────────────────────────────────────────

    #[test]
    fn test_same_cid_does_not_self_pair() {
        let mut engine = default_engine();
        engine.record_access(make_event("QmSelf", 1_000));
        engine.record_access(make_event("QmSelf", 1_100));
        // Same CID appearing twice should NOT create a self-pair.
        assert_eq!(engine.total_pairs(), 0);
    }

    #[test]
    fn test_multiple_pairs_single_anchor() {
        let mut engine = default_engine();
        // Access QmA, then 5 different CIDs in quick succession.
        engine.record_access(make_event("QmA", 1_000));
        for i in 1u64..=5 {
            engine.record_access(make_event(&format!("QmX{}", i), 1_000 + i * 50));
        }
        // All 5 CIDs should be co-accessed with QmA (and also with each other).
        // The engine registers pairs for every CID within the window, so the total
        // will be at least 5 (the QmA-QmXn pairs) plus the cross-pairs among QmX*.
        assert!(
            engine.total_pairs() >= 5,
            "expected >= 5 pairs, got {}",
            engine.total_pairs()
        );
        // Verify the 5 QmA-specific pairs are present.
        let qm_a_pairs = engine.top_coaccessed("QmA", 10);
        assert_eq!(
            qm_a_pairs.len(),
            5,
            "QmA should have 5 direct co-access partners"
        );
    }

    #[test]
    fn test_pattern_window_limits_state() {
        let cfg = PeConfig {
            pattern_window: 3,
            ..Default::default()
        };
        let mut engine = StoragePrefetchEngine::new(cfg);
        for i in 0u64..10 {
            engine.record_access(make_event("QmPW", i * 100));
        }
        let ts = engine
            .pattern_state
            .get("QmPW")
            .cloned()
            .unwrap_or_default();
        assert!(ts.len() <= 3, "len={}", ts.len());
    }

    #[test]
    fn test_generate_hints_no_match() {
        let engine = default_engine();
        let hints = engine.generate_hints("QmNone", 0);
        assert!(hints.is_empty());
    }
}
