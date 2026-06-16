//! StorageTierManager — policy-driven hot/warm/cold/archive tier classification.
//!
//! Tracks per-block access rates and reclassifies blocks across four storage
//! tiers: Hot, Warm, Cold, and Archive. Provides eviction candidates sorted by
//! least-recently-used ordering and exposes atomic statistics for observability.

use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// StorageTier
// ---------------------------------------------------------------------------

/// Four-level storage tier hierarchy, ordered from fastest/most-expensive to
/// slowest/cheapest.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StorageTier {
    /// Hot tier: NVMe / memory cache — target for blocks accessed >1 /s.
    Hot,
    /// Warm tier: fast SSD — blocks accessed >0.1 /s.
    Warm,
    /// Cold tier: HDD / object store — blocks accessed >0.01 /s.
    Cold,
    /// Archive tier: tape / deep cold storage — everything else.
    Archive,
}

impl StorageTier {
    /// Representative one-way latency estimate in milliseconds for each tier.
    pub fn tier_latency_ms_estimate(&self) -> u64 {
        match self {
            StorageTier::Hot => 1,
            StorageTier::Warm => 10,
            StorageTier::Cold => 100,
            StorageTier::Archive => 10_000,
        }
    }

    /// Human-readable tier name (used as key in `tier_summary`).
    pub fn name(&self) -> &'static str {
        match self {
            StorageTier::Hot => "hot",
            StorageTier::Warm => "warm",
            StorageTier::Cold => "cold",
            StorageTier::Archive => "archive",
        }
    }
}

// ---------------------------------------------------------------------------
// TierPolicy
// ---------------------------------------------------------------------------

/// Access-rate thresholds that govern classification into tiers.
///
/// A block is placed in the *highest* tier whose threshold it meets:
/// - `access_rate >= hot_threshold`    → `Hot`
/// - `access_rate >= warm_threshold`   → `Warm`
/// - `access_rate >= cold_threshold`   → `Cold`
/// - otherwise                         → `Archive`
#[derive(Debug, Clone)]
pub struct TierPolicy {
    /// Minimum accesses/sec to remain in Hot tier (default: 1.0).
    pub hot_threshold_access_rate: f64,
    /// Minimum accesses/sec to remain in Warm tier (default: 0.1).
    pub warm_threshold_access_rate: f64,
    /// Minimum accesses/sec to remain in Cold tier (default: 0.01).
    pub cold_threshold_access_rate: f64,
}

impl Default for TierPolicy {
    fn default() -> Self {
        Self {
            hot_threshold_access_rate: 1.0,
            warm_threshold_access_rate: 0.1,
            cold_threshold_access_rate: 0.01,
        }
    }
}

impl TierPolicy {
    /// Classify a block given its current `access_rate` (accesses per second).
    pub fn classify(&self, access_rate: f64) -> StorageTier {
        if access_rate >= self.hot_threshold_access_rate {
            StorageTier::Hot
        } else if access_rate >= self.warm_threshold_access_rate {
            StorageTier::Warm
        } else if access_rate >= self.cold_threshold_access_rate {
            StorageTier::Cold
        } else {
            StorageTier::Archive
        }
    }
}

// ---------------------------------------------------------------------------
// BlockTierRecord
// ---------------------------------------------------------------------------

/// Per-block tracking record held inside `StorageTierManager`.
#[derive(Debug, Clone)]
pub struct BlockTierRecord {
    /// Content identifier string for the block.
    pub cid: String,
    /// Current tier assignment.
    pub current_tier: StorageTier,
    /// Cumulative access count since the record was created.
    pub access_count: u64,
    /// Monotonic timestamp of the most-recent access.
    pub last_accessed: Instant,
    /// Block size in bytes (supplied by the caller on first access).
    pub size_bytes: u64,
    /// Monotonic timestamp of record creation.
    pub created_at: Instant,
}

impl BlockTierRecord {
    /// Instantaneous access rate over the supplied measurement `window`.
    ///
    /// Returns `access_count / window.as_secs_f64()`, or `0.0` if the window
    /// has zero length (avoids division by zero).
    pub fn access_rate_per_sec(&self, window: Duration) -> f64 {
        let secs = window.as_secs_f64();
        if secs <= 0.0 {
            return 0.0;
        }
        self.access_count as f64 / secs
    }
}

// ---------------------------------------------------------------------------
// TierTransition
// ---------------------------------------------------------------------------

/// Describes a single tier change produced by `reclassify_all`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TierTransition {
    /// CID that changed tier.
    pub cid: String,
    /// Previous tier.
    pub from: StorageTier,
    /// New tier.
    pub to: StorageTier,
}

// ---------------------------------------------------------------------------
// TierStats
// ---------------------------------------------------------------------------

/// Atomic counters for observability.
#[derive(Debug, Default)]
pub struct TierStats {
    /// Total number of `record_access` calls processed.
    pub total_accesses_recorded: AtomicU64,
    /// Total number of `reclassify_all` calls made.
    pub total_reclassifications: AtomicU64,
    /// Total number of individual tier transitions detected across all calls.
    pub total_transitions: AtomicU64,
}

/// Point-in-time snapshot of `TierStats` (non-atomic values for easy inspection).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TierStatsSnapshot {
    pub total_accesses_recorded: u64,
    pub total_reclassifications: u64,
    pub total_transitions: u64,
}

impl TierStats {
    /// Capture a consistent snapshot of the counters.
    pub fn snapshot(&self) -> TierStatsSnapshot {
        TierStatsSnapshot {
            total_accesses_recorded: self.total_accesses_recorded.load(Ordering::Relaxed),
            total_reclassifications: self.total_reclassifications.load(Ordering::Relaxed),
            total_transitions: self.total_transitions.load(Ordering::Relaxed),
        }
    }
}

// ---------------------------------------------------------------------------
// StorageTierManager
// ---------------------------------------------------------------------------

/// Central manager for block tier tracking and reclassification.
///
/// # Concurrency
///
/// All block records are protected by a single `RwLock<HashMap<…>>`. Reads
/// (classification queries, summaries) hold a shared lock; writes
/// (access recording, reclassification) hold an exclusive lock.
pub struct StorageTierManager {
    /// Block records keyed by CID string.
    pub records: RwLock<HashMap<String, BlockTierRecord>>,
    /// Policy that governs tier classification.
    pub policy: TierPolicy,
    /// Observable statistics.
    pub stats: TierStats,
}

impl StorageTierManager {
    /// Create a new manager with the supplied policy.
    pub fn new(policy: TierPolicy) -> Self {
        Self {
            records: RwLock::new(HashMap::new()),
            policy,
            stats: TierStats::default(),
        }
    }

    /// Create a new manager with the default policy.
    pub fn with_default_policy() -> Self {
        Self::new(TierPolicy::default())
    }

    // -----------------------------------------------------------------------
    // Mutation
    // -----------------------------------------------------------------------

    /// Record one access to the block identified by `cid`.
    ///
    /// Creates the record on first access; subsequent calls increment
    /// `access_count` and update `last_accessed`.  The `size_bytes` parameter
    /// is used only when creating a new record — it is ignored for existing
    /// records (callers do not need to supply it after the first call).
    pub fn record_access(&self, cid: &str, size_bytes: u64) {
        let now = Instant::now();
        let mut guard = self.records.write();
        match guard.get_mut(cid) {
            Some(rec) => {
                rec.access_count += 1;
                rec.last_accessed = now;
            }
            None => {
                guard.insert(
                    cid.to_owned(),
                    BlockTierRecord {
                        cid: cid.to_owned(),
                        current_tier: StorageTier::Archive,
                        access_count: 1,
                        last_accessed: now,
                        size_bytes,
                        created_at: now,
                    },
                );
            }
        }
        self.stats
            .total_accesses_recorded
            .fetch_add(1, Ordering::Relaxed);
    }

    // -----------------------------------------------------------------------
    // Classification
    // -----------------------------------------------------------------------

    /// Classify the block identified by `cid` using its current access rate
    /// computed over `window`.
    ///
    /// Returns `StorageTier::Archive` for unknown CIDs.
    pub fn classify(&self, cid: &str, window: Duration) -> StorageTier {
        let guard = self.records.read();
        match guard.get(cid) {
            Some(rec) => self.policy.classify(rec.access_rate_per_sec(window)),
            None => StorageTier::Archive,
        }
    }

    /// Reclassify every known block using `window` and return a list of
    /// [`TierTransition`]s for blocks whose tier changed.
    ///
    /// Also updates `current_tier` on the record so that subsequent calls to
    /// `blocks_in_tier` reflect the new assignment.
    pub fn reclassify_all(&self, window: Duration) -> Vec<TierTransition> {
        let mut transitions = Vec::new();
        {
            let mut guard = self.records.write();
            for rec in guard.values_mut() {
                let new_tier = self.policy.classify(rec.access_rate_per_sec(window));
                if new_tier != rec.current_tier {
                    transitions.push(TierTransition {
                        cid: rec.cid.clone(),
                        from: rec.current_tier,
                        to: new_tier,
                    });
                    rec.current_tier = new_tier;
                }
            }
        }
        self.stats
            .total_reclassifications
            .fetch_add(1, Ordering::Relaxed);
        self.stats
            .total_transitions
            .fetch_add(transitions.len() as u64, Ordering::Relaxed);
        transitions
    }

    // -----------------------------------------------------------------------
    // Queries
    // -----------------------------------------------------------------------

    /// Return all CIDs whose `current_tier` matches `tier`.
    pub fn blocks_in_tier(&self, tier: StorageTier) -> Vec<String> {
        let guard = self.records.read();
        guard
            .values()
            .filter(|rec| rec.current_tier == tier)
            .map(|rec| rec.cid.clone())
            .collect()
    }

    /// Return a map from tier name (`"hot"`, `"warm"`, `"cold"`, `"archive"`)
    /// to the number of blocks currently assigned to that tier.
    pub fn tier_summary(&self) -> HashMap<String, usize> {
        let guard = self.records.read();
        let mut summary: HashMap<String, usize> = HashMap::new();
        for tier in &[
            StorageTier::Hot,
            StorageTier::Warm,
            StorageTier::Cold,
            StorageTier::Archive,
        ] {
            summary.insert(tier.name().to_owned(), 0);
        }
        for rec in guard.values() {
            *summary
                .entry(rec.current_tier.name().to_owned())
                .or_insert(0) += 1;
        }
        summary
    }

    /// Return up to `max_count` CIDs from `tier` sorted from least-recently
    /// accessed (oldest `last_accessed`) to most-recently accessed (best
    /// eviction candidates first).
    pub fn eviction_candidates(&self, tier: StorageTier, max_count: usize) -> Vec<String> {
        let guard = self.records.read();
        let mut candidates: Vec<&BlockTierRecord> = guard
            .values()
            .filter(|rec| rec.current_tier == tier)
            .collect();
        // Sort ascending by last_accessed so the oldest (LRU) come first.
        candidates.sort_by_key(|rec| rec.last_accessed);
        candidates
            .into_iter()
            .take(max_count)
            .map(|rec| rec.cid.clone())
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    // ---- TierPolicy::classify --------------------------------------------------

    #[test]
    fn test_classify_hot() {
        let policy = TierPolicy::default();
        assert_eq!(policy.classify(5.0), StorageTier::Hot);
        assert_eq!(policy.classify(1.0), StorageTier::Hot);
    }

    #[test]
    fn test_classify_warm() {
        let policy = TierPolicy::default();
        assert_eq!(policy.classify(0.5), StorageTier::Warm);
        assert_eq!(policy.classify(0.1), StorageTier::Warm);
    }

    #[test]
    fn test_classify_cold() {
        let policy = TierPolicy::default();
        assert_eq!(policy.classify(0.05), StorageTier::Cold);
        assert_eq!(policy.classify(0.01), StorageTier::Cold);
    }

    #[test]
    fn test_classify_archive() {
        let policy = TierPolicy::default();
        assert_eq!(policy.classify(0.005), StorageTier::Archive);
        assert_eq!(policy.classify(0.0), StorageTier::Archive);
    }

    // ---- Tier latency estimates ------------------------------------------------

    #[test]
    fn test_tier_latency_estimates() {
        assert_eq!(StorageTier::Hot.tier_latency_ms_estimate(), 1);
        assert_eq!(StorageTier::Warm.tier_latency_ms_estimate(), 10);
        assert_eq!(StorageTier::Cold.tier_latency_ms_estimate(), 100);
        assert_eq!(StorageTier::Archive.tier_latency_ms_estimate(), 10_000);
    }

    // ---- record_access creates record -----------------------------------------

    #[test]
    fn test_record_access_creates_record() {
        let mgr = StorageTierManager::with_default_policy();
        mgr.record_access("cid-alpha", 1024);

        let guard = mgr.records.read();
        let rec = guard.get("cid-alpha").expect("record should exist");
        assert_eq!(rec.cid, "cid-alpha");
        assert_eq!(rec.access_count, 1);
        assert_eq!(rec.size_bytes, 1024);
    }

    #[test]
    fn test_record_access_increments_existing() {
        let mgr = StorageTierManager::with_default_policy();
        mgr.record_access("cid-beta", 512);
        mgr.record_access("cid-beta", 512);
        mgr.record_access("cid-beta", 512);

        let guard = mgr.records.read();
        let rec = guard.get("cid-beta").expect("record should exist");
        assert_eq!(rec.access_count, 3);
    }

    // ---- access_rate_per_sec formula ------------------------------------------

    #[test]
    fn test_access_rate_per_sec_formula() {
        let rec = BlockTierRecord {
            cid: "test".to_owned(),
            current_tier: StorageTier::Archive,
            access_count: 100,
            last_accessed: Instant::now(),
            size_bytes: 256,
            created_at: Instant::now(),
        };
        let rate = rec.access_rate_per_sec(Duration::from_secs(10));
        // 100 accesses / 10 s = 10.0
        assert!((rate - 10.0).abs() < 1e-9, "expected 10.0, got {rate}");
    }

    #[test]
    fn test_access_rate_zero_window() {
        let rec = BlockTierRecord {
            cid: "test".to_owned(),
            current_tier: StorageTier::Archive,
            access_count: 50,
            last_accessed: Instant::now(),
            size_bytes: 128,
            created_at: Instant::now(),
        };
        // Zero-length window must not panic and must return 0.0.
        let rate = rec.access_rate_per_sec(Duration::from_secs(0));
        assert_eq!(rate, 0.0);
    }

    // ---- classify() -----------------------------------------------------------

    #[test]
    fn test_classify_returns_correct_tier_after_accesses() {
        let mgr = StorageTierManager::with_default_policy();
        // Record 50 accesses — over a 10-second window that is 5.0 /s → Hot.
        for _ in 0..50 {
            mgr.record_access("hot-cid", 128);
        }
        let tier = mgr.classify("hot-cid", Duration::from_secs(10));
        assert_eq!(tier, StorageTier::Hot);
    }

    #[test]
    fn test_classify_unknown_cid_returns_archive() {
        let mgr = StorageTierManager::with_default_policy();
        let tier = mgr.classify("no-such-cid", Duration::from_secs(60));
        assert_eq!(tier, StorageTier::Archive);
    }

    // ---- reclassify_all() -----------------------------------------------------

    #[test]
    fn test_reclassify_all_detects_transitions() {
        let mgr = StorageTierManager::with_default_policy();

        // Insert a record manually in Archive, then fire enough accesses to
        // push it to Hot over a short window.
        mgr.record_access("migrate-cid", 64);
        // Override tier to Archive so there is definitely a transition.
        {
            let mut guard = mgr.records.write();
            if let Some(rec) = guard.get_mut("migrate-cid") {
                rec.current_tier = StorageTier::Archive;
                rec.access_count = 200;
            }
        }

        // 200 accesses over 10 s = 20 /s → Hot.
        let transitions = mgr.reclassify_all(Duration::from_secs(10));
        assert_eq!(transitions.len(), 1);
        assert_eq!(transitions[0].cid, "migrate-cid");
        assert_eq!(transitions[0].from, StorageTier::Archive);
        assert_eq!(transitions[0].to, StorageTier::Hot);
    }

    #[test]
    fn test_reclassify_all_no_transitions_when_unchanged() {
        let mgr = StorageTierManager::with_default_policy();
        // 1 access over 60 s = ~0.0167 /s → Cold.
        mgr.record_access("stable-cid", 256);
        // First reclassify: Archive → Cold (1 transition).
        let t1 = mgr.reclassify_all(Duration::from_secs(60));
        assert_eq!(t1.len(), 1);
        // Second reclassify with same window: no change.
        let t2 = mgr.reclassify_all(Duration::from_secs(60));
        assert_eq!(t2.len(), 0);
    }

    // ---- blocks_in_tier() -----------------------------------------------------

    #[test]
    fn test_blocks_in_tier_returns_correct_cids() {
        let mgr = StorageTierManager::with_default_policy();
        // Manually plant two Hot records and one Warm record.
        {
            let mut guard = mgr.records.write();
            for name in &["hot-1", "hot-2"] {
                guard.insert(
                    name.to_string(),
                    BlockTierRecord {
                        cid: name.to_string(),
                        current_tier: StorageTier::Hot,
                        access_count: 100,
                        last_accessed: Instant::now(),
                        size_bytes: 64,
                        created_at: Instant::now(),
                    },
                );
            }
            guard.insert(
                "warm-1".to_owned(),
                BlockTierRecord {
                    cid: "warm-1".to_owned(),
                    current_tier: StorageTier::Warm,
                    access_count: 5,
                    last_accessed: Instant::now(),
                    size_bytes: 64,
                    created_at: Instant::now(),
                },
            );
        }
        let mut hot_blocks = mgr.blocks_in_tier(StorageTier::Hot);
        hot_blocks.sort();
        assert_eq!(hot_blocks, vec!["hot-1", "hot-2"]);

        let warm_blocks = mgr.blocks_in_tier(StorageTier::Warm);
        assert_eq!(warm_blocks, vec!["warm-1"]);

        let cold_blocks = mgr.blocks_in_tier(StorageTier::Cold);
        assert!(cold_blocks.is_empty());
    }

    // ---- eviction_candidates() ------------------------------------------------

    #[test]
    fn test_eviction_candidates_sorted_by_lru() {
        let mgr = StorageTierManager::with_default_policy();

        // Insert three Cold records with increasing last_accessed timestamps.
        // We sleep briefly between insertions so Instant::now() differs.
        let cids = ["cold-a", "cold-b", "cold-c"];
        for cid in &cids {
            mgr.record_access(cid, 128);
            // A tiny sleep ensures distinct Instant values across platforms.
            thread::sleep(Duration::from_millis(2));
        }
        // Force all to Cold tier.
        {
            let mut guard = mgr.records.write();
            for cid in &cids {
                if let Some(rec) = guard.get_mut(*cid) {
                    rec.current_tier = StorageTier::Cold;
                }
            }
        }

        // Oldest access → first candidate.
        let candidates = mgr.eviction_candidates(StorageTier::Cold, 2);
        assert_eq!(candidates.len(), 2);
        // "cold-a" was accessed first, so it should be the top eviction candidate.
        assert_eq!(candidates[0], "cold-a");
        assert_eq!(candidates[1], "cold-b");
    }

    // ---- tier_summary() -------------------------------------------------------

    #[test]
    fn test_tier_summary_counts_correct() {
        let mgr = StorageTierManager::with_default_policy();
        {
            let mut guard = mgr.records.write();
            let tiers = [
                ("s-hot-1", StorageTier::Hot),
                ("s-hot-2", StorageTier::Hot),
                ("s-warm-1", StorageTier::Warm),
                ("s-cold-1", StorageTier::Cold),
            ];
            for (name, tier) in &tiers {
                guard.insert(
                    name.to_string(),
                    BlockTierRecord {
                        cid: name.to_string(),
                        current_tier: *tier,
                        access_count: 1,
                        last_accessed: Instant::now(),
                        size_bytes: 64,
                        created_at: Instant::now(),
                    },
                );
            }
        }
        let summary = mgr.tier_summary();
        assert_eq!(*summary.get("hot").unwrap_or(&0), 2);
        assert_eq!(*summary.get("warm").unwrap_or(&0), 1);
        assert_eq!(*summary.get("cold").unwrap_or(&0), 1);
        assert_eq!(*summary.get("archive").unwrap_or(&0), 0);
    }

    // ---- Stats accumulation ---------------------------------------------------

    #[test]
    fn test_stats_accumulation() {
        let mgr = StorageTierManager::with_default_policy();

        mgr.record_access("stat-cid-1", 64);
        mgr.record_access("stat-cid-2", 64);
        mgr.record_access("stat-cid-1", 64); // increment

        let snap_before = mgr.stats.snapshot();
        assert_eq!(snap_before.total_accesses_recorded, 3);

        // Two CIDs default to Archive; reclassify over a short window will
        // produce 0 or more transitions depending on rate.
        mgr.reclassify_all(Duration::from_secs(1));
        let snap_after = mgr.stats.snapshot();
        assert_eq!(snap_after.total_reclassifications, 1);
        // total_transitions may be 0 or more — just verify it is consistent.
        assert!(snap_after.total_transitions >= snap_before.total_transitions);
    }
}
