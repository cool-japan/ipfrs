//! Block tier migration manager.
//!
//! Manages migration of blocks between storage tiers (Hot/Warm/Cold/Archive)
//! based on access patterns and configurable policies.
//!
//! # Example
//!
//! ```rust
//! use ipfrs_storage::tier_migrator::{BlockTierMigrator, MigrationPolicy, StorageTier};
//!
//! let policy = MigrationPolicy::default();
//! let mut migrator = BlockTierMigrator::new(policy);
//!
//! let now = 1_000_000u64;
//! migrator.register("cid1".to_string(), StorageTier::Hot, 8192, now);
//!
//! // After idle time exceeds hot_to_warm_idle_secs, plan migration
//! let tasks = migrator.plan_migrations(now + 90_000);
//! for task in &tasks {
//!     migrator.apply_migration(task);
//! }
//! ```

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// StorageTier
// ---------------------------------------------------------------------------

/// Storage tier classification with ordering from hottest (0) to coldest (3).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum StorageTier {
    /// Hot tier — low latency, highest cost per GB.
    Hot = 0,
    /// Warm tier — moderate latency and cost.
    Warm = 1,
    /// Cold tier — higher latency, low cost.
    Cold = 2,
    /// Archive tier — very high latency, cheapest storage.
    Archive = 3,
}

impl StorageTier {
    /// Cost in USD per GB per month.
    pub fn cost_per_gb(&self) -> f64 {
        match self {
            StorageTier::Hot => 0.10,
            StorageTier::Warm => 0.04,
            StorageTier::Cold => 0.01,
            StorageTier::Archive => 0.002,
        }
    }

    /// Typical read access latency in milliseconds.
    pub fn access_latency_ms(&self) -> u64 {
        match self {
            StorageTier::Hot => 1,
            StorageTier::Warm => 10,
            StorageTier::Cold => 100,
            StorageTier::Archive => 1000,
        }
    }
}

// ---------------------------------------------------------------------------
// MigrationPolicy
// ---------------------------------------------------------------------------

/// Policy parameters that govern when blocks are migrated between tiers.
#[derive(Clone, Debug)]
pub struct MigrationPolicy {
    /// Seconds of idle time before a Hot block is demoted to Warm.
    pub hot_to_warm_idle_secs: u64,
    /// Seconds of idle time before a Warm block is demoted to Cold.
    pub warm_to_cold_idle_secs: u64,
    /// Seconds of idle time before a Cold block is demoted to Archive.
    pub cold_to_archive_idle_secs: u64,
    /// Blocks strictly smaller than this byte threshold are not moved to Cold or Archive.
    pub min_size_for_cold: u64,
}

impl Default for MigrationPolicy {
    fn default() -> Self {
        Self {
            hot_to_warm_idle_secs: 86_400,        // 1 day
            warm_to_cold_idle_secs: 604_800,      // 7 days
            cold_to_archive_idle_secs: 2_592_000, // 30 days
            min_size_for_cold: 4096,
        }
    }
}

// ---------------------------------------------------------------------------
// BlockTierRecord
// ---------------------------------------------------------------------------

/// Metadata record for a single block tracked by the migrator.
#[derive(Clone, Debug)]
pub struct BlockTierRecord {
    /// Content identifier of the block.
    pub cid: String,
    /// Current storage tier.
    pub tier: StorageTier,
    /// Size of the block in bytes.
    pub size_bytes: u64,
    /// Unix timestamp (seconds) of the most recent access.
    pub last_accessed_secs: u64,
    /// Total number of accesses recorded.
    pub access_count: u64,
}

impl BlockTierRecord {
    /// Seconds elapsed since the last access, saturating at zero.
    pub fn idle_secs(&self, now: u64) -> u64 {
        now.saturating_sub(self.last_accessed_secs)
    }

    /// Estimated monthly storage cost (USD) for this block.
    pub fn monthly_cost(&self) -> f64 {
        let gib = self.size_bytes as f64 / (1024_u64.pow(3) as f64);
        gib * self.tier.cost_per_gb()
    }
}

// ---------------------------------------------------------------------------
// MigrationTask
// ---------------------------------------------------------------------------

/// A planned migration action for a single block.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MigrationTask {
    /// Content identifier of the block to migrate.
    pub cid: String,
    /// Source tier.
    pub from_tier: StorageTier,
    /// Destination tier.
    pub to_tier: StorageTier,
    /// Block size in bytes (informational).
    pub size_bytes: u64,
}

// ---------------------------------------------------------------------------
// MigratorStats
// ---------------------------------------------------------------------------

/// Aggregate statistics over all tracked blocks.
#[derive(Clone, Debug, Default)]
pub struct MigratorStats {
    /// Total number of tracked blocks.
    pub total_blocks: usize,
    /// Number of blocks per tier, indexed by `StorageTier as usize`.
    pub blocks_per_tier: [usize; 4],
    /// Total bytes stored per tier, indexed by `StorageTier as usize`.
    pub total_bytes_per_tier: [u64; 4],
}

impl MigratorStats {
    /// Sum of monthly storage costs across all provided records.
    pub fn total_monthly_cost(&self, records: &[BlockTierRecord]) -> f64 {
        records.iter().map(|r| r.monthly_cost()).sum()
    }
}

// ---------------------------------------------------------------------------
// BlockTierMigrator
// ---------------------------------------------------------------------------

/// Manages migration of blocks between storage tiers based on access patterns
/// and a configurable [`MigrationPolicy`].
pub struct BlockTierMigrator {
    /// Tracked block records keyed by CID.
    pub records: HashMap<String, BlockTierRecord>,
    /// Active migration policy.
    pub policy: MigrationPolicy,
}

impl BlockTierMigrator {
    /// Create a new migrator with the given policy.
    pub fn new(policy: MigrationPolicy) -> Self {
        Self {
            records: HashMap::new(),
            policy,
        }
    }

    /// Register a new block in the given tier.
    ///
    /// If a block with the same CID already exists it is overwritten.
    pub fn register(&mut self, cid: String, tier: StorageTier, size_bytes: u64, now_secs: u64) {
        self.records.insert(
            cid.clone(),
            BlockTierRecord {
                cid,
                tier,
                size_bytes,
                last_accessed_secs: now_secs,
                access_count: 0,
            },
        );
    }

    /// Record an access for a block.
    ///
    /// Updates `last_accessed_secs`, increments `access_count`, and promotes
    /// the block back to [`StorageTier::Hot`] if it is currently in a colder tier.
    pub fn record_access(&mut self, cid: &str, now_secs: u64) {
        if let Some(record) = self.records.get_mut(cid) {
            record.last_accessed_secs = now_secs;
            record.access_count = record.access_count.saturating_add(1);
            if record.tier != StorageTier::Hot {
                record.tier = StorageTier::Hot;
            }
        }
    }

    /// Compute a set of migration tasks based on current idle times and the
    /// active policy.
    ///
    /// Only tasks where the target tier differs from the current tier are emitted.
    /// Small blocks (below `policy.min_size_for_cold`) are never moved to Cold or
    /// Archive.
    pub fn plan_migrations(&self, now_secs: u64) -> Vec<MigrationTask> {
        let mut tasks = Vec::new();

        for record in self.records.values() {
            let target = self.target_tier(record, now_secs);
            if target != record.tier {
                tasks.push(MigrationTask {
                    cid: record.cid.clone(),
                    from_tier: record.tier,
                    to_tier: target,
                    size_bytes: record.size_bytes,
                });
            }
        }

        tasks
    }

    /// Apply a previously planned migration task, updating the stored tier.
    ///
    /// If the block referenced by the task is not found, this is a no-op.
    pub fn apply_migration(&mut self, task: &MigrationTask) {
        if let Some(record) = self.records.get_mut(&task.cid) {
            record.tier = task.to_tier;
        }
    }

    /// Compute aggregate statistics over all tracked blocks.
    pub fn stats(&self) -> MigratorStats {
        let mut blocks_per_tier = [0usize; 4];
        let mut total_bytes_per_tier = [0u64; 4];

        for record in self.records.values() {
            let idx = record.tier as usize;
            blocks_per_tier[idx] += 1;
            total_bytes_per_tier[idx] = total_bytes_per_tier[idx].saturating_add(record.size_bytes);
        }

        MigratorStats {
            total_blocks: self.records.len(),
            blocks_per_tier,
            total_bytes_per_tier,
        }
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Determine the ideal target tier for a record given the current time.
    fn target_tier(&self, record: &BlockTierRecord, now_secs: u64) -> StorageTier {
        let idle = record.idle_secs(now_secs);
        let small = record.size_bytes < self.policy.min_size_for_cold;

        match record.tier {
            StorageTier::Hot => {
                if idle >= self.policy.hot_to_warm_idle_secs {
                    StorageTier::Warm
                } else {
                    StorageTier::Hot
                }
            }
            StorageTier::Warm => {
                if idle >= self.policy.warm_to_cold_idle_secs && !small {
                    StorageTier::Cold
                } else {
                    StorageTier::Warm
                }
            }
            StorageTier::Cold => {
                if idle >= self.policy.cold_to_archive_idle_secs && !small {
                    StorageTier::Archive
                } else {
                    StorageTier::Cold
                }
            }
            StorageTier::Archive => StorageTier::Archive,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_migrator() -> BlockTierMigrator {
        BlockTierMigrator::new(MigrationPolicy::default())
    }

    // 1. Register creates a record
    #[test]
    fn test_register_creates_record() {
        let mut m = default_migrator();
        m.register("cid1".to_string(), StorageTier::Hot, 1024, 1000);
        assert!(m.records.contains_key("cid1"));
    }

    // 2. Register sets correct tier
    #[test]
    fn test_register_sets_tier() {
        let mut m = default_migrator();
        m.register("cid2".to_string(), StorageTier::Warm, 2048, 2000);
        assert_eq!(m.records["cid2"].tier, StorageTier::Warm);
    }

    // 3. Register initialises access_count to 0
    #[test]
    fn test_register_access_count_zero() {
        let mut m = default_migrator();
        m.register("cid3".to_string(), StorageTier::Cold, 512, 3000);
        assert_eq!(m.records["cid3"].access_count, 0);
    }

    // 4. record_access promotes cold block to Hot
    #[test]
    fn test_record_access_promotes_cold_to_hot() {
        let mut m = default_migrator();
        m.register("cid4".to_string(), StorageTier::Cold, 8192, 1000);
        m.record_access("cid4", 2000);
        assert_eq!(m.records["cid4"].tier, StorageTier::Hot);
    }

    // 5. record_access promotes warm block to Hot
    #[test]
    fn test_record_access_promotes_warm_to_hot() {
        let mut m = default_migrator();
        m.register("cid5".to_string(), StorageTier::Warm, 8192, 1000);
        m.record_access("cid5", 2000);
        assert_eq!(m.records["cid5"].tier, StorageTier::Hot);
    }

    // 6. record_access increments access_count
    #[test]
    fn test_record_access_increments_count() {
        let mut m = default_migrator();
        m.register("cid6".to_string(), StorageTier::Hot, 1024, 1000);
        m.record_access("cid6", 1001);
        m.record_access("cid6", 1002);
        assert_eq!(m.records["cid6"].access_count, 2);
    }

    // 7. record_access updates last_accessed_secs
    #[test]
    fn test_record_access_updates_timestamp() {
        let mut m = default_migrator();
        m.register("cid7".to_string(), StorageTier::Hot, 1024, 1000);
        m.record_access("cid7", 9999);
        assert_eq!(m.records["cid7"].last_accessed_secs, 9999);
    }

    // 8. record_access on unknown CID is a no-op (no panic)
    #[test]
    fn test_record_access_unknown_cid_noop() {
        let mut m = default_migrator();
        m.record_access("nonexistent", 5000); // must not panic
    }

    // 9. plan_migrations: Hot → Warm after idle exceeds hot_to_warm threshold
    #[test]
    fn test_plan_migrations_hot_to_warm() {
        let mut m = default_migrator();
        let start = 0u64;
        m.register("cid9".to_string(), StorageTier::Hot, 8192, start);
        // idle = 86_401 > 86_400
        let tasks = m.plan_migrations(start + 86_401);
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].from_tier, StorageTier::Hot);
        assert_eq!(tasks[0].to_tier, StorageTier::Warm);
    }

    // 10. plan_migrations: no task when idle < hot_to_warm threshold
    #[test]
    fn test_plan_migrations_hot_not_yet_idle() {
        let mut m = default_migrator();
        let start = 0u64;
        m.register("cid10".to_string(), StorageTier::Hot, 8192, start);
        let tasks = m.plan_migrations(start + 86_399);
        assert!(tasks.is_empty());
    }

    // 11. plan_migrations: Warm → Cold for large block after threshold
    #[test]
    fn test_plan_migrations_warm_to_cold_large_block() {
        let mut m = default_migrator();
        let start = 0u64;
        m.register("cid11".to_string(), StorageTier::Warm, 8192, start);
        let tasks = m.plan_migrations(start + 604_801);
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].from_tier, StorageTier::Warm);
        assert_eq!(tasks[0].to_tier, StorageTier::Cold);
    }

    // 12. plan_migrations: small block stays Warm even after warm_to_cold threshold
    #[test]
    fn test_plan_migrations_small_block_stays_warm() {
        let mut m = default_migrator();
        let start = 0u64;
        // size_bytes (1024) < min_size_for_cold (4096)
        m.register("cid12".to_string(), StorageTier::Warm, 1024, start);
        let tasks = m.plan_migrations(start + 604_801);
        assert!(tasks.is_empty());
    }

    // 13. plan_migrations: Cold → Archive after threshold
    #[test]
    fn test_plan_migrations_cold_to_archive() {
        let mut m = default_migrator();
        let start = 0u64;
        m.register("cid13".to_string(), StorageTier::Cold, 8192, start);
        let tasks = m.plan_migrations(start + 2_592_001);
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].from_tier, StorageTier::Cold);
        assert_eq!(tasks[0].to_tier, StorageTier::Archive);
    }

    // 14. plan_migrations: Archive blocks never migrate
    #[test]
    fn test_plan_migrations_archive_stays() {
        let mut m = default_migrator();
        m.register("cid14".to_string(), StorageTier::Archive, 8192, 0);
        let tasks = m.plan_migrations(u64::MAX);
        assert!(tasks.is_empty());
    }

    // 15. apply_migration updates the block's tier
    #[test]
    fn test_apply_migration_updates_tier() {
        let mut m = default_migrator();
        m.register("cid15".to_string(), StorageTier::Hot, 8192, 0);
        let task = MigrationTask {
            cid: "cid15".to_string(),
            from_tier: StorageTier::Hot,
            to_tier: StorageTier::Warm,
            size_bytes: 8192,
        };
        m.apply_migration(&task);
        assert_eq!(m.records["cid15"].tier, StorageTier::Warm);
    }

    // 16. apply_migration on unknown CID is a no-op
    #[test]
    fn test_apply_migration_unknown_cid_noop() {
        let mut m = default_migrator();
        let task = MigrationTask {
            cid: "ghost".to_string(),
            from_tier: StorageTier::Hot,
            to_tier: StorageTier::Warm,
            size_bytes: 512,
        };
        m.apply_migration(&task); // must not panic
    }

    // 17. stats: counts blocks per tier correctly
    #[test]
    fn test_stats_counts_per_tier() {
        let mut m = default_migrator();
        m.register("h1".to_string(), StorageTier::Hot, 1024, 0);
        m.register("h2".to_string(), StorageTier::Hot, 1024, 0);
        m.register("w1".to_string(), StorageTier::Warm, 2048, 0);
        m.register("c1".to_string(), StorageTier::Cold, 4096, 0);
        m.register("a1".to_string(), StorageTier::Archive, 8192, 0);

        let s = m.stats();
        assert_eq!(s.total_blocks, 5);
        assert_eq!(s.blocks_per_tier[StorageTier::Hot as usize], 2);
        assert_eq!(s.blocks_per_tier[StorageTier::Warm as usize], 1);
        assert_eq!(s.blocks_per_tier[StorageTier::Cold as usize], 1);
        assert_eq!(s.blocks_per_tier[StorageTier::Archive as usize], 1);
    }

    // 18. stats: total bytes per tier
    #[test]
    fn test_stats_bytes_per_tier() {
        let mut m = default_migrator();
        m.register("h1".to_string(), StorageTier::Hot, 500, 0);
        m.register("h2".to_string(), StorageTier::Hot, 300, 0);
        m.register("w1".to_string(), StorageTier::Warm, 1000, 0);

        let s = m.stats();
        assert_eq!(s.total_bytes_per_tier[StorageTier::Hot as usize], 800);
        assert_eq!(s.total_bytes_per_tier[StorageTier::Warm as usize], 1000);
        assert_eq!(s.total_bytes_per_tier[StorageTier::Cold as usize], 0);
    }

    // 19. idle_secs saturates at zero when now < last_accessed
    #[test]
    fn test_idle_secs_saturates() {
        let record = BlockTierRecord {
            cid: "x".to_string(),
            tier: StorageTier::Hot,
            size_bytes: 1024,
            last_accessed_secs: 9999,
            access_count: 0,
        };
        assert_eq!(record.idle_secs(1000), 0);
    }

    // 20. monthly_cost calculation
    #[test]
    fn test_monthly_cost_hot() {
        let record = BlockTierRecord {
            cid: "y".to_string(),
            tier: StorageTier::Hot,
            size_bytes: 1024 * 1024 * 1024, // 1 GiB
            last_accessed_secs: 0,
            access_count: 0,
        };
        let cost = record.monthly_cost();
        // 1 GiB * $0.10/GiB = $0.10
        assert!((cost - 0.10).abs() < 1e-9);
    }

    // 21. MigratorStats::total_monthly_cost sums correctly
    #[test]
    fn test_total_monthly_cost() {
        let records = vec![
            BlockTierRecord {
                cid: "a".to_string(),
                tier: StorageTier::Hot,
                size_bytes: 1024 * 1024 * 1024,
                last_accessed_secs: 0,
                access_count: 0,
            },
            BlockTierRecord {
                cid: "b".to_string(),
                tier: StorageTier::Archive,
                size_bytes: 1024 * 1024 * 1024,
                last_accessed_secs: 0,
                access_count: 0,
            },
        ];
        let stats = MigratorStats::default();
        let total = stats.total_monthly_cost(&records);
        // 0.10 + 0.002 = 0.102
        assert!((total - 0.102).abs() < 1e-9);
    }

    // 22. StorageTier ordering
    #[test]
    fn test_tier_ordering() {
        assert!(StorageTier::Hot < StorageTier::Warm);
        assert!(StorageTier::Warm < StorageTier::Cold);
        assert!(StorageTier::Cold < StorageTier::Archive);
    }

    // 23. cost_per_gb values
    #[test]
    fn test_cost_per_gb() {
        assert!((StorageTier::Hot.cost_per_gb() - 0.10).abs() < 1e-12);
        assert!((StorageTier::Warm.cost_per_gb() - 0.04).abs() < 1e-12);
        assert!((StorageTier::Cold.cost_per_gb() - 0.01).abs() < 1e-12);
        assert!((StorageTier::Archive.cost_per_gb() - 0.002).abs() < 1e-12);
    }

    // 24. access_latency_ms values
    #[test]
    fn test_access_latency_ms() {
        assert_eq!(StorageTier::Hot.access_latency_ms(), 1);
        assert_eq!(StorageTier::Warm.access_latency_ms(), 10);
        assert_eq!(StorageTier::Cold.access_latency_ms(), 100);
        assert_eq!(StorageTier::Archive.access_latency_ms(), 1000);
    }

    // 25. plan_migrations + apply_migration round-trip for full demotion chain
    #[test]
    fn test_full_demotion_chain() {
        let mut m = default_migrator();
        let start = 0u64;
        m.register("chain".to_string(), StorageTier::Hot, 8192, start);

        // Hot → Warm
        let now1 = start + 86_401;
        let tasks = m.plan_migrations(now1);
        assert_eq!(tasks.len(), 1);
        m.apply_migration(&tasks[0]);
        assert_eq!(m.records["chain"].tier, StorageTier::Warm);

        // Warm → Cold (last_accessed still = start, idle from start)
        let now2 = start + 604_801;
        let tasks = m.plan_migrations(now2);
        assert_eq!(tasks.len(), 1);
        m.apply_migration(&tasks[0]);
        assert_eq!(m.records["chain"].tier, StorageTier::Cold);

        // Cold → Archive
        let now3 = start + 2_592_001;
        let tasks = m.plan_migrations(now3);
        assert_eq!(tasks.len(), 1);
        m.apply_migration(&tasks[0]);
        assert_eq!(m.records["chain"].tier, StorageTier::Archive);

        // Archive stays
        let tasks = m.plan_migrations(u64::MAX);
        assert!(tasks.is_empty());
    }
}
