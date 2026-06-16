//! Access frequency and recency tracking for storage tiering and eviction decisions.
//!
//! [`AccessTracker`] maintains per-CID access records and computes a composite
//! score that balances how often a block is accessed (frequency) against how
//! recently it was last accessed (recency).  The score drives hot/cold
//! classification and LRU-style eviction.
//!
//! # Score formula
//!
//! ```text
//! score = access_count / (1 + time_since_last_access_secs * decay_factor)
//! ```
//!
//! Blocks with a score above `hot_threshold` are classified as *hot*; blocks
//! below `cold_threshold` are classified as *cold*.  When the tracker is at
//! capacity, [`AccessTracker::enforce_capacity`] evicts the coldest blocks
//! first.
//!
//! # Example
//!
//! ```rust
//! use ipfrs_storage::access_tracker::{AccessTracker, TrackerConfig};
//!
//! let config = TrackerConfig::default();
//! let mut tracker = AccessTracker::new(config);
//!
//! tracker.record_access("bafybeig", 1024, 0);
//! tracker.record_access("bafybeig", 1024, 10);
//!
//! assert!(tracker.is_hot("bafybeig", 10));
//! ```

use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration parameters that govern tracker behaviour.
#[derive(Debug, Clone)]
pub struct TrackerConfig {
    /// Maximum number of records held at any time.  When exceeded,
    /// [`AccessTracker::enforce_capacity`] evicts the coldest entries.
    pub max_records: usize,
    /// Multiplicative weight applied to the recency component of the score.
    /// Higher values reward recently-accessed blocks more strongly.
    pub recency_weight: f64,
    /// Time-based decay applied per second of idle time.  Larger values make
    /// scores fall faster as blocks sit unused.
    pub decay_factor: f64,
    /// Score threshold above which a block is classified as *hot*.
    pub hot_threshold: f64,
    /// Score threshold below which a block is classified as *cold*.
    pub cold_threshold: f64,
}

impl Default for TrackerConfig {
    fn default() -> Self {
        Self {
            max_records: 10_000,
            recency_weight: 1.0,
            decay_factor: 0.001,
            hot_threshold: 5.0,
            cold_threshold: 0.5,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AccessRecord
// ─────────────────────────────────────────────────────────────────────────────

/// A single CID's access history and computed temperature score.
#[derive(Debug, Clone)]
pub struct AccessRecord {
    /// The Content Identifier this record belongs to.
    pub cid: String,
    /// Total number of times this CID has been accessed.
    pub access_count: u64,
    /// Logical timestamp (seconds) of the most recent access.
    pub last_access: u64,
    /// Logical timestamp (seconds) of the very first access.
    pub first_access: u64,
    /// Size in bytes reported at the most recent access.
    pub size_bytes: u64,
    /// Composite temperature score: `frequency * recency_weight`.
    /// Recomputed by [`AccessTracker::update_score`].
    pub access_score: f64,
}

// ─────────────────────────────────────────────────────────────────────────────
// TrackerStats
// ─────────────────────────────────────────────────────────────────────────────

/// Aggregate statistics maintained by the tracker.
#[derive(Debug, Clone, Default)]
pub struct TrackerStats {
    /// Cumulative number of calls to [`AccessTracker::record_access`].
    pub total_accesses: u64,
    /// Number of distinct CIDs currently tracked.
    pub unique_cids: u64,
    /// Number of CIDs currently classified as hot.
    pub hot_blocks: u64,
    /// Number of CIDs currently classified as cold.
    pub cold_blocks: u64,
    /// Number of records removed by eviction (capacity or explicit).
    pub evictions: u64,
}

// ─────────────────────────────────────────────────────────────────────────────
// AccessTracker
// ─────────────────────────────────────────────────────────────────────────────

/// Tracks access frequency and recency for storage tiering and eviction.
pub struct AccessTracker {
    config: TrackerConfig,
    records: HashMap<String, AccessRecord>,
    stats: TrackerStats,
}

impl AccessTracker {
    // ── Construction ────────────────────────────────────────────────────────

    /// Create a new tracker with the provided configuration.
    pub fn new(config: TrackerConfig) -> Self {
        Self {
            records: HashMap::new(),
            stats: TrackerStats::default(),
            config,
        }
    }

    // ── Core recording ──────────────────────────────────────────────────────

    /// Record one access to `cid` at logical time `now` (seconds).
    ///
    /// If this is the first access for `cid` a new [`AccessRecord`] is
    /// created.  Otherwise the existing record is updated in-place and its
    /// score is recomputed.  After recording, capacity is enforced.
    pub fn record_access(&mut self, cid: &str, size: u64, now: u64) {
        self.stats.total_accesses += 1;

        if let Some(record) = self.records.get_mut(cid) {
            record.access_count += 1;
            record.last_access = now;
            if size > 0 {
                record.size_bytes = size;
            }
            let score = Self::compute_score_inner(&self.config, record, now);
            record.access_score = score;
        } else {
            let score_initial = self.config.recency_weight; // count=1 → 1/(1+0)
            let record = AccessRecord {
                cid: cid.to_owned(),
                access_count: 1,
                last_access: now,
                first_access: now,
                size_bytes: size,
                access_score: score_initial,
            };
            self.records.insert(cid.to_owned(), record);
            self.stats.unique_cids += 1;
        }

        // Keep within capacity — but do not count forced evictions twice.
        self.enforce_capacity(now);
        self.refresh_stats(now);
    }

    // ── Score computation ────────────────────────────────────────────────────

    /// Compute the temperature score for `record` at logical time `now`.
    ///
    /// Formula:
    /// ```text
    /// score = access_count / (1 + time_since_last_access * decay_factor)
    /// ```
    ///
    /// `recency_weight` from the config scales the final result so that
    /// operators can tune how much recency matters relative to raw frequency.
    pub fn compute_score(&self, record: &AccessRecord, now: u64) -> f64 {
        Self::compute_score_inner(&self.config, record, now)
    }

    /// Internal stateless version that accepts the config by reference.
    fn compute_score_inner(config: &TrackerConfig, record: &AccessRecord, now: u64) -> f64 {
        let time_since = now.saturating_sub(record.last_access) as f64;
        let denominator = 1.0 + time_since * config.decay_factor;
        (record.access_count as f64 / denominator) * config.recency_weight
    }

    /// Recompute and persist the score for the CID identified by `cid`.
    ///
    /// This is useful when time has advanced without any new accesses.
    pub fn update_score(&mut self, cid: &str, now: u64) {
        if let Some(record) = self.records.get_mut(cid) {
            let score = Self::compute_score_inner(&self.config, record, now);
            record.access_score = score;
        }
    }

    // ── Hot / cold classification ────────────────────────────────────────────

    /// Returns `true` when the CID's current score exceeds `hot_threshold`.
    pub fn is_hot(&self, cid: &str, now: u64) -> bool {
        self.records
            .get(cid)
            .map(|r| Self::compute_score_inner(&self.config, r, now) >= self.config.hot_threshold)
            .unwrap_or(false)
    }

    /// Returns `true` when the CID's current score is below `cold_threshold`.
    pub fn is_cold(&self, cid: &str, now: u64) -> bool {
        self.records
            .get(cid)
            .map(|r| Self::compute_score_inner(&self.config, r, now) < self.config.cold_threshold)
            .unwrap_or(true)
    }

    /// All records currently scoring at or above `hot_threshold`.
    pub fn hot_blocks(&self, now: u64) -> Vec<&AccessRecord> {
        self.records
            .values()
            .filter(|r| {
                Self::compute_score_inner(&self.config, r, now) >= self.config.hot_threshold
            })
            .collect()
    }

    /// All records currently scoring below `cold_threshold`.
    pub fn cold_blocks(&self, now: u64) -> Vec<&AccessRecord> {
        self.records
            .values()
            .filter(|r| {
                Self::compute_score_inner(&self.config, r, now) < self.config.cold_threshold
            })
            .collect()
    }

    // ── Eviction ─────────────────────────────────────────────────────────────

    /// Remove and return the single record with the lowest temperature score.
    ///
    /// Returns `None` when the tracker is empty.
    pub fn evict_coldest(&mut self, now: u64) -> Option<AccessRecord> {
        let coldest_key = self
            .records
            .iter()
            .min_by(|a, b| {
                let sa = Self::compute_score_inner(&self.config, a.1, now);
                let sb = Self::compute_score_inner(&self.config, b.1, now);
                sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(k, _)| k.clone());

        if let Some(key) = coldest_key {
            let record = self.records.remove(&key)?;
            self.stats.evictions += 1;
            self.stats.unique_cids = self.stats.unique_cids.saturating_sub(1);
            self.refresh_stats(now);
            Some(record)
        } else {
            None
        }
    }

    /// Evict records until `records.len() <= max_records`.
    ///
    /// Returns the number of records evicted in this call.
    pub fn enforce_capacity(&mut self, now: u64) -> usize {
        let mut evicted = 0usize;
        while self.records.len() > self.config.max_records {
            // Identify the key with the lowest score without borrowing self.
            let coldest_key = self
                .records
                .iter()
                .min_by(|a, b| {
                    let sa = Self::compute_score_inner(&self.config, a.1, now);
                    let sb = Self::compute_score_inner(&self.config, b.1, now);
                    sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(k, _)| k.clone());

            if let Some(key) = coldest_key {
                if self.records.remove(&key).is_some() {
                    self.stats.evictions += 1;
                    self.stats.unique_cids = self.stats.unique_cids.saturating_sub(1);
                    evicted += 1;
                }
            } else {
                break;
            }
        }
        if evicted > 0 {
            self.refresh_stats(now);
        }
        evicted
    }

    // ── Queries ───────────────────────────────────────────────────────────────

    /// Look up the record for `cid`, if present.
    pub fn get_record(&self, cid: &str) -> Option<&AccessRecord> {
        self.records.get(cid)
    }

    /// Number of records currently held.
    pub fn record_count(&self) -> usize {
        self.records.len()
    }

    /// Reference to the current aggregate statistics.
    pub fn stats(&self) -> &TrackerStats {
        &self.stats
    }

    /// The `n` most-accessed records, sorted by `access_count` descending.
    ///
    /// If `n` exceeds the number of records, all records are returned.
    pub fn top_accessed(&self, n: usize) -> Vec<&AccessRecord> {
        let mut records: Vec<&AccessRecord> = self.records.values().collect();
        records.sort_by_key(|b| std::cmp::Reverse(b.access_count));
        records.truncate(n);
        records
    }

    // ── Internal helpers ─────────────────────────────────────────────────────

    /// Recompute the derived counters (hot_blocks, cold_blocks) in `stats`.
    fn refresh_stats(&mut self, now: u64) {
        let hot_threshold = self.config.hot_threshold;
        let cold_threshold = self.config.cold_threshold;
        let config = &self.config;

        let mut hot = 0u64;
        let mut cold = 0u64;
        for record in self.records.values() {
            let score = Self::compute_score_inner(config, record, now);
            if score >= hot_threshold {
                hot += 1;
            } else if score < cold_threshold {
                cold += 1;
            }
        }
        self.stats.hot_blocks = hot;
        self.stats.cold_blocks = cold;
        self.stats.unique_cids = self.records.len() as u64;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ────────────────────────────────────────────────────────────

    fn default_config() -> TrackerConfig {
        TrackerConfig {
            max_records: 5,
            recency_weight: 1.0,
            decay_factor: 0.1,
            hot_threshold: 3.0,
            cold_threshold: 0.5,
        }
    }

    fn tracker() -> AccessTracker {
        AccessTracker::new(default_config())
    }

    // ── 1. Record access creates a new entry ───────────────────────────────

    #[test]
    fn test_record_access_creates_entry() {
        let mut t = tracker();
        t.record_access("cid1", 512, 0);
        assert!(t.get_record("cid1").is_some());
    }

    // ── 2. Missing CID returns None ────────────────────────────────────────

    #[test]
    fn test_get_record_missing() {
        let t = tracker();
        assert!(t.get_record("nonexistent").is_none());
    }

    // ── 3. Multiple accesses accumulate ───────────────────────────────────

    #[test]
    fn test_multiple_accesses_accumulate() {
        let mut t = tracker();
        t.record_access("cid1", 512, 0);
        t.record_access("cid1", 512, 1);
        t.record_access("cid1", 512, 2);
        let rec = t.get_record("cid1").expect("record must exist");
        assert_eq!(rec.access_count, 3);
    }

    // ── 4. first_access is preserved ──────────────────────────────────────

    #[test]
    fn test_first_access_preserved() {
        let mut t = tracker();
        t.record_access("cid1", 512, 100);
        t.record_access("cid1", 512, 200);
        let rec = t.get_record("cid1").expect("must exist");
        assert_eq!(rec.first_access, 100);
        assert_eq!(rec.last_access, 200);
    }

    // ── 5. Score computation at t=0 ───────────────────────────────────────

    #[test]
    fn test_compute_score_at_access_time() {
        let t = tracker();
        let rec = AccessRecord {
            cid: "c".to_owned(),
            access_count: 10,
            last_access: 50,
            first_access: 0,
            size_bytes: 1024,
            access_score: 0.0,
        };
        // score = 10 / (1 + 0 * 0.1) * 1.0 = 10.0
        let score = t.compute_score(&rec, 50);
        assert!((score - 10.0).abs() < 1e-9);
    }

    // ── 6. Score decays with elapsed time ─────────────────────────────────

    #[test]
    fn test_score_decays_with_time() {
        let t = tracker();
        let rec = AccessRecord {
            cid: "c".to_owned(),
            access_count: 10,
            last_access: 0,
            first_access: 0,
            size_bytes: 1024,
            access_score: 0.0,
        };
        let score_now = t.compute_score(&rec, 0);
        let score_later = t.compute_score(&rec, 100);
        assert!(score_later < score_now, "score must decay over time");
    }

    // ── 7. is_hot — block above threshold ────────────────────────────────

    #[test]
    fn test_is_hot_true() {
        let mut t = tracker();
        // Record 4 times in quick succession → score ≈ 4.0 > 3.0
        for i in 0u64..4 {
            t.record_access("hot_cid", 512, i);
        }
        assert!(t.is_hot("hot_cid", 3));
    }

    // ── 8. is_hot — missing CID is not hot ───────────────────────────────

    #[test]
    fn test_is_hot_missing() {
        let t = tracker();
        assert!(!t.is_hot("nope", 0));
    }

    // ── 9. is_cold — fresh single-access block with age ──────────────────

    #[test]
    fn test_is_cold_with_old_block() {
        let mut t = tracker();
        t.record_access("cold_cid", 512, 0);
        // After 10 000 seconds the score = 1 / (1 + 1000) ≈ 0.001 < 0.5
        assert!(t.is_cold("cold_cid", 10_000));
    }

    // ── 10. is_cold — missing CID defaults to cold ───────────────────────

    #[test]
    fn test_is_cold_missing() {
        let t = tracker();
        assert!(t.is_cold("nope", 0));
    }

    // ── 11. hot_blocks returns only hot records ───────────────────────────

    #[test]
    fn test_hot_blocks_list() {
        let mut t = tracker();
        for i in 0u64..4 {
            t.record_access("hot", 512, i);
        }
        t.record_access("warm", 512, 0);
        let hot = t.hot_blocks(3);
        assert!(hot.iter().any(|r| r.cid == "hot"));
        assert!(!hot.iter().any(|r| r.cid == "warm"));
    }

    // ── 12. cold_blocks returns only cold records ─────────────────────────

    #[test]
    fn test_cold_blocks_list() {
        let mut t = tracker();
        t.record_access("cold", 512, 0);
        let cold = t.cold_blocks(10_000);
        assert!(cold.iter().any(|r| r.cid == "cold"));
    }

    // ── 13. evict_coldest removes the lowest-score entry ─────────────────

    #[test]
    fn test_evict_coldest_removes_lowest() {
        let mut t = tracker();
        t.record_access("a", 512, 0); // score ≈ 1.0
        for i in 0u64..4 {
            t.record_access("b", 512, i); // score ≈ 4.0
        }
        let evicted = t.evict_coldest(3).expect("must evict something");
        assert_eq!(evicted.cid, "a");
        assert!(t.get_record("a").is_none());
        assert!(t.get_record("b").is_some());
    }

    // ── 14. evict_coldest on empty tracker returns None ───────────────────

    #[test]
    fn test_evict_coldest_empty() {
        let mut t = tracker();
        assert!(t.evict_coldest(0).is_none());
    }

    // ── 15. enforce_capacity trims to max_records ─────────────────────────

    #[test]
    fn test_enforce_capacity() {
        let mut t = tracker(); // max_records = 5
        for i in 0u64..8 {
            // Insert 8 distinct CIDs, enforce runs inside record_access
            // so the last insertion triggers enforcement. We only have 5 slots.
            t.records.insert(
                format!("cid{i}"),
                AccessRecord {
                    cid: format!("cid{i}"),
                    access_count: i + 1,
                    last_access: i,
                    first_access: 0,
                    size_bytes: 512,
                    access_score: (i + 1) as f64,
                },
            );
        }
        t.stats.unique_cids = t.records.len() as u64;
        let evicted = t.enforce_capacity(100);
        assert_eq!(evicted, 3);
        assert_eq!(t.record_count(), 5);
    }

    // ── 16. enforce_capacity returns 0 when within limits ────────────────

    #[test]
    fn test_enforce_capacity_no_op() {
        let mut t = tracker();
        t.record_access("only_one", 512, 0);
        let evicted = t.enforce_capacity(0);
        assert_eq!(evicted, 0);
    }

    // ── 17. top_accessed ordering by access_count desc ───────────────────

    #[test]
    fn test_top_accessed_ordering() {
        let mut t = tracker();
        t.record_access("low", 512, 0);
        for _ in 0..3 {
            t.record_access("mid", 512, 0);
        }
        for _ in 0..5 {
            t.record_access("high", 512, 0);
        }
        let top = t.top_accessed(3);
        assert_eq!(top[0].cid, "high");
        assert_eq!(top[1].cid, "mid");
        assert_eq!(top[2].cid, "low");
    }

    // ── 18. top_accessed(n) where n > records returns all ─────────────────

    #[test]
    fn test_top_accessed_more_than_records() {
        let mut t = tracker();
        t.record_access("a", 512, 0);
        t.record_access("b", 512, 0);
        let top = t.top_accessed(100);
        assert_eq!(top.len(), 2);
    }

    // ── 19. top_accessed(0) returns empty slice ────────────────────────────

    #[test]
    fn test_top_accessed_zero() {
        let mut t = tracker();
        t.record_access("a", 512, 0);
        assert!(t.top_accessed(0).is_empty());
    }

    // ── 20. stats.total_accesses increments correctly ────────────────────

    #[test]
    fn test_stats_total_accesses() {
        let mut t = tracker();
        t.record_access("x", 512, 0);
        t.record_access("x", 512, 1);
        t.record_access("y", 512, 2);
        assert_eq!(t.stats().total_accesses, 3);
    }

    // ── 21. stats.unique_cids is accurate ────────────────────────────────

    #[test]
    fn test_stats_unique_cids() {
        let mut t = tracker();
        t.record_access("a", 512, 0);
        t.record_access("b", 512, 0);
        t.record_access("a", 512, 1); // duplicate
        assert_eq!(t.stats().unique_cids, 2);
    }

    // ── 22. stats.evictions increments on evict_coldest ──────────────────

    #[test]
    fn test_stats_evictions_on_evict_coldest() {
        let mut t = tracker();
        t.record_access("a", 512, 0);
        t.evict_coldest(0);
        assert_eq!(t.stats().evictions, 1);
    }

    // ── 23. record_count returns correct value ────────────────────────────

    #[test]
    fn test_record_count() {
        let mut t = tracker();
        assert_eq!(t.record_count(), 0);
        t.record_access("a", 512, 0);
        t.record_access("b", 512, 0);
        assert_eq!(t.record_count(), 2);
    }

    // ── 24. update_score refreshes stored score ───────────────────────────

    #[test]
    fn test_update_score() {
        let mut t = tracker();
        t.record_access("a", 512, 0);
        let initial_score = t.get_record("a").map(|r| r.access_score).unwrap_or(0.0);
        t.update_score("a", 1000); // much later → score should be smaller
        let later_score = t.get_record("a").map(|r| r.access_score).unwrap_or(0.0);
        assert!(later_score < initial_score);
    }

    // ── 25. hot_blocks and cold_blocks are mutually exclusive ─────────────
    //
    // We use a single logical `now` so that the same scoring function is
    // applied to both lists.  A block cannot simultaneously be >= hot_threshold
    // and < cold_threshold (assuming hot_threshold > cold_threshold).

    #[test]
    fn test_hot_and_cold_disjoint() {
        let mut t = tracker();
        // Record "hot" 4 times at t=0..3; its score at t=3 ≈ 4/(1+0) = 4.0 (> 3.0).
        for i in 0u64..4 {
            t.record_access("hot", 512, i);
        }
        // Record "cold" once at t=0; its score at t=3 = 1/(1+0.3) ≈ 0.77 (warm, not cold).
        // To ensure it is cold we pick a time far in the future: score ≈ 1/(1+1000*0.1)=0.01.
        t.record_access("cold", 512, 0);

        // Use the same `now` for both queries so scoring is consistent.
        let query_now = 3u64;
        let hot_ids: std::collections::HashSet<String> = t
            .hot_blocks(query_now)
            .iter()
            .map(|r| r.cid.clone())
            .collect();
        let cold_ids: std::collections::HashSet<String> = t
            .cold_blocks(query_now)
            .iter()
            .map(|r| r.cid.clone())
            .collect();

        // Intersection must be empty — by definition hot_threshold > cold_threshold.
        let overlap: Vec<_> = hot_ids.intersection(&cold_ids).collect();
        assert!(
            overlap.is_empty(),
            "hot and cold sets must be disjoint at the same timestamp"
        );
    }

    // ── 26. stats hot_blocks count matches hot_blocks() length ────────────

    #[test]
    fn test_stats_hot_blocks_count() {
        let mut t = tracker();
        for i in 0u64..4 {
            t.record_access("h", 512, i);
        }
        t.record_access("c", 512, 0);
        // Force stats refresh
        t.refresh_stats(3);
        assert_eq!(t.stats().hot_blocks, t.hot_blocks(3).len() as u64);
    }

    // ── 27. size_bytes is updated on subsequent accesses ──────────────────

    #[test]
    fn test_size_bytes_updated() {
        let mut t = tracker();
        t.record_access("a", 512, 0);
        t.record_access("a", 2048, 1);
        let rec = t.get_record("a").expect("must exist");
        assert_eq!(rec.size_bytes, 2048);
    }
}
