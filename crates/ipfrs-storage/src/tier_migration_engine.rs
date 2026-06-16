//! StorageTierMigrator — intelligent block migration engine for storage tiers.
//!
//! Moves blocks between Hot, Warm, Cold, and Archive tiers based on access
//! frequency, age, and configurable size policies. Supports dry-run mode,
//! batched execution, and detailed migration logs.
//!
//! # Example
//!
//! ```rust
//! use ipfrs_storage::tier_migration_engine::{
//!     BlockMeta, MigratorConfig, StorageTierMigrator, TierPolicy,
//!     StorageTier as TmStorageTier,
//! };
//!
//! let policy = TierPolicy {
//!     tier: TmStorageTier::Hot,
//!     max_age_ms: Some(3_600_000),   // 1 hour
//!     min_access_count: Some(5),
//!     max_block_size_bytes: None,
//!     min_block_size_bytes: None,
//! };
//! let config = MigratorConfig {
//!     policies: vec![policy],
//!     dry_run: false,
//!     batch_size: 100,
//!     max_migrations_per_run: 1000,
//! };
//! let mut migrator = StorageTierMigrator::new(config);
//!
//! let now_ms = 1_000_000_u64;
//! let meta = BlockMeta {
//!     cid: "QmTest1".to_string(),
//!     size_bytes: 4096,
//!     access_count: 2,
//!     last_accessed_ms: now_ms - 10_000_000,
//!     created_ms: now_ms - 20_000_000,
//!     current_tier: TmStorageTier::Hot,
//! };
//! migrator.register_block(meta);
//! let result = migrator.run_migration_cycle(now_ms + 5_000_000);
//! assert!(result.actions_planned >= 1);
//! ```

use std::collections::{HashMap, VecDeque};

// ---------------------------------------------------------------------------
// StorageTier
// ---------------------------------------------------------------------------

/// Storage tier classification — Hot is fastest/most expensive; Archive is
/// slowest/cheapest. `priority()` returns a descending-urgency value (Hot=3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StorageTier {
    /// Hot tier: NVMe / memory — lowest latency, highest cost.
    Hot,
    /// Warm tier: fast SSD — moderate latency and cost.
    Warm,
    /// Cold tier: HDD / object store — high latency, low cost.
    Cold,
    /// Archive tier: tape / deep cold — very high latency, cheapest.
    Archive,
}

impl StorageTier {
    /// Descending priority: Hot=3 … Archive=0. Used to sort migration actions
    /// (hottest blocks with pending demotions are processed first).
    pub fn priority(&self) -> u8 {
        match self {
            StorageTier::Hot => 3,
            StorageTier::Warm => 2,
            StorageTier::Cold => 1,
            StorageTier::Archive => 0,
        }
    }

    /// Returns the next colder tier, or `None` if already at Archive.
    pub fn next_colder(&self) -> Option<StorageTier> {
        match self {
            StorageTier::Hot => Some(StorageTier::Warm),
            StorageTier::Warm => Some(StorageTier::Cold),
            StorageTier::Cold => Some(StorageTier::Archive),
            StorageTier::Archive => None,
        }
    }

    /// Human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            StorageTier::Hot => "hot",
            StorageTier::Warm => "warm",
            StorageTier::Cold => "cold",
            StorageTier::Archive => "archive",
        }
    }
}

impl std::fmt::Display for StorageTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

// ---------------------------------------------------------------------------
// TierPolicy
// ---------------------------------------------------------------------------

/// Policy that governs when blocks should be demoted *out* of a given tier.
///
/// All specified conditions are evaluated independently: the first one that
/// triggers causes a demotion. `None` means "no limit / don't check".
#[derive(Debug, Clone)]
pub struct TierPolicy {
    /// The tier this policy applies to.
    pub tier: StorageTier,
    /// Maximum idle time in milliseconds. Blocks whose
    /// `last_accessed_ms + max_age_ms < now_ms` are considered stale.
    pub max_age_ms: Option<u64>,
    /// Minimum access count required to stay in this tier. Blocks with fewer
    /// accesses than this threshold are under-accessed.
    pub min_access_count: Option<u64>,
    /// Blocks larger than this byte count should not reside in this tier.
    pub max_block_size_bytes: Option<u64>,
    /// Blocks smaller than this byte count should not reside in this tier.
    pub min_block_size_bytes: Option<u64>,
}

// ---------------------------------------------------------------------------
// BlockMeta
// ---------------------------------------------------------------------------

/// Runtime metadata for a single content-addressed block tracked by the migrator.
#[derive(Debug, Clone)]
pub struct BlockMeta {
    /// Content identifier (CID) of the block — primary key.
    pub cid: String,
    /// Block payload size in bytes.
    pub size_bytes: u64,
    /// Cumulative access count since the block was registered.
    pub access_count: u64,
    /// Unix timestamp in milliseconds of the most-recent access.
    pub last_accessed_ms: u64,
    /// Unix timestamp in milliseconds when the block was first created/registered.
    pub created_ms: u64,
    /// Current storage tier for the block.
    pub current_tier: StorageTier,
}

// ---------------------------------------------------------------------------
// MigrationReason
// ---------------------------------------------------------------------------

/// Reason code explaining why a migration was triggered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MigrationReason {
    /// Block has not been accessed within the policy's `max_age_ms` window.
    TooOld,
    /// Block's cumulative access count is below `min_access_count`.
    UnderAccessed,
    /// Block is too large for the current tier (`max_block_size_bytes` exceeded).
    Oversized,
    /// Block is too small for the current tier (`min_block_size_bytes` not met).
    Undersized,
    /// An explicit policy change was applied — re-evaluation required.
    PolicyChange,
    /// Migration was requested manually (e.g. operator command).
    Manual,
}

impl std::fmt::Display for MigrationReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            MigrationReason::TooOld => "too_old",
            MigrationReason::UnderAccessed => "under_accessed",
            MigrationReason::Oversized => "oversized",
            MigrationReason::Undersized => "undersized",
            MigrationReason::PolicyChange => "policy_change",
            MigrationReason::Manual => "manual",
        };
        f.write_str(s)
    }
}

// ---------------------------------------------------------------------------
// MigrationAction
// ---------------------------------------------------------------------------

/// A planned or executed migration for a single block.
#[derive(Debug, Clone)]
pub struct MigrationAction {
    /// CID of the block to migrate.
    pub cid: String,
    /// Source tier (current location before migration).
    pub from_tier: StorageTier,
    /// Destination tier (target location after migration).
    pub to_tier: StorageTier,
    /// Reason the migration was triggered.
    pub reason: MigrationReason,
    /// Unix timestamp in milliseconds when this action was scheduled/created.
    pub scheduled_at: u64,
}

// ---------------------------------------------------------------------------
// MigrationResult
// ---------------------------------------------------------------------------

/// Summary produced by [`StorageTierMigrator::execute_migrations`] or
/// [`StorageTierMigrator::run_migration_cycle`].
#[derive(Debug, Clone, Default)]
pub struct MigrationResult {
    /// Total migration actions that were planned in this run.
    pub actions_planned: usize,
    /// Actions that were actually executed (0 in dry-run mode).
    pub actions_executed: usize,
    /// Total bytes migrated across all executed actions.
    pub bytes_migrated: u64,
    /// CIDs for which migration failed (e.g. block not found in registry).
    pub failed_cids: Vec<String>,
}

// ---------------------------------------------------------------------------
// MigratorConfig
// ---------------------------------------------------------------------------

/// Configuration for [`StorageTierMigrator`].
#[derive(Debug, Clone)]
pub struct MigratorConfig {
    /// Ordered list of tier policies evaluated during each migration cycle.
    /// Policies are sorted internally by `tier.priority()` descending before
    /// evaluation, so the order supplied here does not matter.
    pub policies: Vec<TierPolicy>,
    /// When `true`, migrations are planned but blocks are *not* moved.
    /// Counters still reflect what *would* have happened.
    pub dry_run: bool,
    /// Number of migration actions to process per internal batch.
    pub batch_size: usize,
    /// Hard cap on total migrations executed in a single run.
    pub max_migrations_per_run: usize,
}

impl Default for MigratorConfig {
    fn default() -> Self {
        Self {
            policies: Vec::new(),
            dry_run: false,
            batch_size: 100,
            max_migrations_per_run: 1000,
        }
    }
}

// ---------------------------------------------------------------------------
// MigratorStats
// ---------------------------------------------------------------------------

/// Snapshot of migrator statistics for observability.
#[derive(Debug, Clone)]
pub struct MigratorStats {
    /// Total number of blocks currently tracked.
    pub total_blocks: usize,
    /// Blocks currently in Hot tier.
    pub hot_blocks: usize,
    /// Blocks currently in Warm tier.
    pub warm_blocks: usize,
    /// Blocks currently in Cold tier.
    pub cold_blocks: usize,
    /// Blocks currently in Archive tier.
    pub archive_blocks: usize,
    /// All-time total migration executions.
    pub total_migrations: u64,
    /// All-time total bytes migrated.
    pub total_bytes_migrated: u64,
}

// ---------------------------------------------------------------------------
// StorageTierMigrator
// ---------------------------------------------------------------------------

/// Production-grade intelligent block migration engine.
///
/// Tracks blocks across four storage tiers (Hot, Warm, Cold, Archive) and
/// demotes them according to configurable [`TierPolicy`] rules based on age,
/// access frequency, and size constraints.
///
/// # Lifecycle
///
/// 1. Configure via [`MigratorConfig`] and create with [`StorageTierMigrator::new`].
/// 2. Register blocks with [`register_block`](StorageTierMigrator::register_block).
/// 3. Record access events with [`record_access`](StorageTierMigrator::record_access).
/// 4. Trigger a migration cycle with [`run_migration_cycle`](StorageTierMigrator::run_migration_cycle).
/// 5. Inspect results via [`migration_log`](StorageTierMigrator::migration_log) or
///    [`migrator_stats`](StorageTierMigrator::migrator_stats).
pub struct StorageTierMigrator {
    /// Migrator configuration (policies, dry-run flag, batch/max sizes).
    pub config: MigratorConfig,
    /// Map from CID → block metadata. Central registry.
    pub blocks: HashMap<String, BlockMeta>,
    /// Bounded ring-buffer of recently executed (or dry-run planned) migrations.
    pub migration_log: VecDeque<MigrationAction>,
    /// All-time migration execution count.
    pub total_migrations: u64,
    /// All-time total bytes migrated.
    pub total_bytes_migrated: u64,
}

/// Maximum number of log entries kept in the in-memory ring buffer.
const LOG_CAPACITY: usize = 10_000;

impl StorageTierMigrator {
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    /// Create a new migrator from the given configuration.
    pub fn new(config: MigratorConfig) -> Self {
        Self {
            config,
            blocks: HashMap::new(),
            migration_log: VecDeque::with_capacity(LOG_CAPACITY),
            total_migrations: 0,
            total_bytes_migrated: 0,
        }
    }

    // -----------------------------------------------------------------------
    // Block registry
    // -----------------------------------------------------------------------

    /// Register a new block or overwrite an existing entry with updated metadata.
    pub fn register_block(&mut self, meta: BlockMeta) {
        self.blocks.insert(meta.cid.clone(), meta);
    }

    /// Remove a block from the registry. Returns `true` if the block existed.
    pub fn unregister_block(&mut self, cid: &str) -> bool {
        self.blocks.remove(cid).is_some()
    }

    /// Record an access event for a block, incrementing its counter and
    /// updating `last_accessed_ms`. Returns `true` if the block was found.
    pub fn record_access(&mut self, cid: &str, now_ms: u64) -> bool {
        if let Some(meta) = self.blocks.get_mut(cid) {
            meta.access_count = meta.access_count.saturating_add(1);
            meta.last_accessed_ms = now_ms;
            true
        } else {
            false
        }
    }

    // -----------------------------------------------------------------------
    // Evaluation
    // -----------------------------------------------------------------------

    /// Evaluate whether a single block needs to be migrated, given the current
    /// timestamp `now_ms`.
    ///
    /// Policies are checked in descending tier priority order. If a block's
    /// current tier matches a policy and any demotion condition is satisfied,
    /// a [`MigrationAction`] targeting the next-colder tier is returned.
    /// Returns `None` if no migration is required.
    pub fn evaluate_block(&self, meta: &BlockMeta, now_ms: u64) -> Option<MigrationAction> {
        // Collect policies for the block's current tier, sorted by priority desc.
        let mut applicable: Vec<&TierPolicy> = self
            .config
            .policies
            .iter()
            .filter(|p| p.tier == meta.current_tier)
            .collect();

        // Sort descending by priority (Hot=3 first) for deterministic ordering
        // when multiple policies match the same tier (unlikely but supported).
        applicable.sort_by_key(|b| std::cmp::Reverse(b.tier.priority()));

        for policy in applicable {
            if let Some(reason) = Self::check_demotion(policy, meta, now_ms) {
                let to_tier = meta.current_tier.next_colder()?;
                return Some(MigrationAction {
                    cid: meta.cid.clone(),
                    from_tier: meta.current_tier,
                    to_tier,
                    reason,
                    scheduled_at: now_ms,
                });
            }
        }
        None
    }

    /// Internal: check which demotion condition fires first for a policy.
    fn check_demotion(
        policy: &TierPolicy,
        meta: &BlockMeta,
        now_ms: u64,
    ) -> Option<MigrationReason> {
        // Age check: block hasn't been accessed within `max_age_ms`.
        if let Some(max_age) = policy.max_age_ms {
            if meta.last_accessed_ms.saturating_add(max_age) < now_ms {
                return Some(MigrationReason::TooOld);
            }
        }

        // Access-count check: block hasn't been accessed enough.
        if let Some(min_access) = policy.min_access_count {
            if meta.access_count < min_access {
                return Some(MigrationReason::UnderAccessed);
            }
        }

        // Size upper-bound: block is too large for this tier.
        if let Some(max_size) = policy.max_block_size_bytes {
            if meta.size_bytes > max_size {
                return Some(MigrationReason::Oversized);
            }
        }

        // Size lower-bound: block is too small for this tier.
        if let Some(min_size) = policy.min_block_size_bytes {
            if meta.size_bytes < min_size {
                return Some(MigrationReason::Undersized);
            }
        }

        None
    }

    // -----------------------------------------------------------------------
    // Planning
    // -----------------------------------------------------------------------

    /// Evaluate every tracked block and return the list of [`MigrationAction`]s
    /// that should be executed.
    ///
    /// Actions are sorted by `from_tier.priority()` descending (hottest blocks
    /// first) and capped at [`MigratorConfig::max_migrations_per_run`].
    pub fn plan_migrations(&self, now_ms: u64) -> Vec<MigrationAction> {
        let mut actions: Vec<MigrationAction> = self
            .blocks
            .values()
            .filter_map(|meta| self.evaluate_block(meta, now_ms))
            .collect();

        // Sort by from_tier priority descending.
        actions.sort_by_key(|b| std::cmp::Reverse(b.from_tier.priority()));

        // Cap at max_migrations_per_run.
        actions.truncate(self.config.max_migrations_per_run);
        actions
    }

    // -----------------------------------------------------------------------
    // Execution
    // -----------------------------------------------------------------------

    /// Execute a pre-planned list of migration actions.
    ///
    /// In dry-run mode: populates `actions_planned` but leaves block metadata
    /// untouched. In live mode: blocks are processed in batches of
    /// [`MigratorConfig::batch_size`]. CIDs that no longer exist in the registry
    /// are recorded in `failed_cids`.
    pub fn execute_migrations(
        &mut self,
        actions: Vec<MigrationAction>,
        now_ms: u64,
    ) -> MigrationResult {
        let actions_planned = actions.len();
        let mut result = MigrationResult {
            actions_planned,
            actions_executed: 0,
            bytes_migrated: 0,
            failed_cids: Vec::new(),
        };

        if self.config.dry_run {
            // In dry-run mode append to the log but do NOT mutate block state.
            for action in actions {
                self.push_log(action);
            }
            return result;
        }

        let batch_size = self.config.batch_size.max(1);
        for chunk in actions.chunks(batch_size) {
            for action in chunk {
                match self.blocks.get_mut(&action.cid) {
                    Some(meta) => {
                        let size = meta.size_bytes;
                        meta.current_tier = action.to_tier;
                        result.actions_executed += 1;
                        result.bytes_migrated = result.bytes_migrated.saturating_add(size);
                        self.total_migrations = self.total_migrations.saturating_add(1);
                        self.total_bytes_migrated = self.total_bytes_migrated.saturating_add(size);
                        // Clone action for log before borrowing issues arise.
                        let log_entry = MigrationAction {
                            cid: action.cid.clone(),
                            from_tier: action.from_tier,
                            to_tier: action.to_tier,
                            reason: action.reason,
                            scheduled_at: now_ms,
                        };
                        self.push_log(log_entry);
                    }
                    None => {
                        result.failed_cids.push(action.cid.clone());
                    }
                }
            }
        }

        result
    }

    /// Convenience method: plan migrations for `now_ms`, then execute them in
    /// one atomic call. Returns a combined [`MigrationResult`].
    pub fn run_migration_cycle(&mut self, now_ms: u64) -> MigrationResult {
        let actions = self.plan_migrations(now_ms);
        self.execute_migrations(actions, now_ms)
    }

    // -----------------------------------------------------------------------
    // Queries
    // -----------------------------------------------------------------------

    /// Returns all blocks currently residing in `tier`, sorted by
    /// `access_count` descending (most-accessed first).
    pub fn blocks_in_tier(&self, tier: &StorageTier) -> Vec<&BlockMeta> {
        let mut result: Vec<&BlockMeta> = self
            .blocks
            .values()
            .filter(|m| &m.current_tier == tier)
            .collect();
        result.sort_by_key(|b| std::cmp::Reverse(b.access_count));
        result
    }

    /// Immutable reference to the internal migration log ring-buffer.
    pub fn migration_log(&self) -> &VecDeque<MigrationAction> {
        &self.migration_log
    }

    /// Snapshot of current migrator statistics.
    pub fn migrator_stats(&self) -> MigratorStats {
        let mut hot = 0usize;
        let mut warm = 0usize;
        let mut cold = 0usize;
        let mut archive = 0usize;

        for meta in self.blocks.values() {
            match meta.current_tier {
                StorageTier::Hot => hot += 1,
                StorageTier::Warm => warm += 1,
                StorageTier::Cold => cold += 1,
                StorageTier::Archive => archive += 1,
            }
        }

        MigratorStats {
            total_blocks: self.blocks.len(),
            hot_blocks: hot,
            warm_blocks: warm,
            cold_blocks: cold,
            archive_blocks: archive,
            total_migrations: self.total_migrations,
            total_bytes_migrated: self.total_bytes_migrated,
        }
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Push a log entry, evicting the oldest if the buffer is full.
    fn push_log(&mut self, action: MigrationAction) {
        if self.migration_log.len() >= LOG_CAPACITY {
            self.migration_log.pop_front();
        }
        self.migration_log.push_back(action);
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::{
        BlockMeta, MigrationReason, MigratorConfig, StorageTier, StorageTierMigrator, TierPolicy,
    };

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn make_policy(
        tier: StorageTier,
        max_age_ms: Option<u64>,
        min_access: Option<u64>,
        max_size: Option<u64>,
        min_size: Option<u64>,
    ) -> TierPolicy {
        TierPolicy {
            tier,
            max_age_ms,
            min_access_count: min_access,
            max_block_size_bytes: max_size,
            min_block_size_bytes: min_size,
        }
    }

    fn make_block(
        cid: &str,
        tier: StorageTier,
        size: u64,
        access_count: u64,
        last_accessed_ms: u64,
        created_ms: u64,
    ) -> BlockMeta {
        BlockMeta {
            cid: cid.to_string(),
            size_bytes: size,
            access_count,
            last_accessed_ms,
            created_ms,
            current_tier: tier,
        }
    }

    fn simple_migrator() -> StorageTierMigrator {
        let policy = make_policy(
            StorageTier::Hot,
            Some(3_600_000), // 1 hour
            Some(5),
            None,
            None,
        );
        StorageTierMigrator::new(MigratorConfig {
            policies: vec![policy],
            ..MigratorConfig::default()
        })
    }

    const NOW: u64 = 1_000_000_000;

    // -----------------------------------------------------------------------
    // StorageTier tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_tier_priority_ordering() {
        assert!(StorageTier::Hot.priority() > StorageTier::Warm.priority());
        assert!(StorageTier::Warm.priority() > StorageTier::Cold.priority());
        assert!(StorageTier::Cold.priority() > StorageTier::Archive.priority());
    }

    #[test]
    fn test_tier_priority_values() {
        assert_eq!(StorageTier::Hot.priority(), 3);
        assert_eq!(StorageTier::Warm.priority(), 2);
        assert_eq!(StorageTier::Cold.priority(), 1);
        assert_eq!(StorageTier::Archive.priority(), 0);
    }

    #[test]
    fn test_tier_next_colder() {
        assert_eq!(StorageTier::Hot.next_colder(), Some(StorageTier::Warm));
        assert_eq!(StorageTier::Warm.next_colder(), Some(StorageTier::Cold));
        assert_eq!(StorageTier::Cold.next_colder(), Some(StorageTier::Archive));
        assert_eq!(StorageTier::Archive.next_colder(), None);
    }

    #[test]
    fn test_tier_display() {
        assert_eq!(StorageTier::Hot.to_string(), "hot");
        assert_eq!(StorageTier::Warm.to_string(), "warm");
        assert_eq!(StorageTier::Cold.to_string(), "cold");
        assert_eq!(StorageTier::Archive.to_string(), "archive");
    }

    #[test]
    fn test_tier_equality() {
        assert_eq!(StorageTier::Hot, StorageTier::Hot);
        assert_ne!(StorageTier::Hot, StorageTier::Cold);
    }

    // -----------------------------------------------------------------------
    // Block registry tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_register_block() {
        let mut m = simple_migrator();
        let block = make_block("cid1", StorageTier::Hot, 4096, 10, NOW, NOW - 1000);
        m.register_block(block);
        assert_eq!(m.blocks.len(), 1);
        assert!(m.blocks.contains_key("cid1"));
    }

    #[test]
    fn test_register_block_overwrite() {
        let mut m = simple_migrator();
        m.register_block(make_block("cid1", StorageTier::Hot, 4096, 10, NOW, NOW));
        m.register_block(make_block("cid1", StorageTier::Warm, 8192, 20, NOW, NOW));
        assert_eq!(m.blocks.len(), 1);
        let b = m.blocks.get("cid1").expect("block must exist");
        assert_eq!(b.current_tier, StorageTier::Warm);
        assert_eq!(b.size_bytes, 8192);
    }

    #[test]
    fn test_unregister_existing_block() {
        let mut m = simple_migrator();
        m.register_block(make_block("cid1", StorageTier::Hot, 1024, 5, NOW, NOW));
        assert!(m.unregister_block("cid1"));
        assert!(m.blocks.is_empty());
    }

    #[test]
    fn test_unregister_missing_block() {
        let mut m = simple_migrator();
        assert!(!m.unregister_block("nonexistent"));
    }

    // -----------------------------------------------------------------------
    // Access recording tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_record_access_increments_count() {
        let mut m = simple_migrator();
        m.register_block(make_block("cid1", StorageTier::Hot, 512, 0, NOW, NOW));
        assert!(m.record_access("cid1", NOW + 100));
        let b = m.blocks.get("cid1").expect("block must exist");
        assert_eq!(b.access_count, 1);
        assert_eq!(b.last_accessed_ms, NOW + 100);
    }

    #[test]
    fn test_record_access_multiple_times() {
        let mut m = simple_migrator();
        m.register_block(make_block("cid1", StorageTier::Hot, 512, 0, NOW, NOW));
        for i in 1..=5 {
            m.record_access("cid1", NOW + i * 100);
        }
        let b = m.blocks.get("cid1").expect("block must exist");
        assert_eq!(b.access_count, 5);
    }

    #[test]
    fn test_record_access_missing_returns_false() {
        let mut m = simple_migrator();
        assert!(!m.record_access("ghost", NOW));
    }

    // -----------------------------------------------------------------------
    // evaluate_block tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_evaluate_block_no_policy_no_action() {
        let m = StorageTierMigrator::new(MigratorConfig::default());
        let block = make_block("c1", StorageTier::Hot, 512, 0, NOW, NOW);
        assert!(m.evaluate_block(&block, NOW + 1_000_000).is_none());
    }

    #[test]
    fn test_evaluate_block_too_old_triggers() {
        let policy = make_policy(StorageTier::Hot, Some(1_000), None, None, None);
        let m = StorageTierMigrator::new(MigratorConfig {
            policies: vec![policy],
            ..MigratorConfig::default()
        });
        // last_accessed_ms + max_age (1000) < now → stale
        let block = make_block("c1", StorageTier::Hot, 512, 10, NOW - 2_000, NOW - 5_000);
        let action = m.evaluate_block(&block, NOW).expect("should trigger");
        assert_eq!(action.reason, MigrationReason::TooOld);
        assert_eq!(action.from_tier, StorageTier::Hot);
        assert_eq!(action.to_tier, StorageTier::Warm);
    }

    #[test]
    fn test_evaluate_block_under_accessed_triggers() {
        let policy = make_policy(StorageTier::Hot, None, Some(10), None, None);
        let m = StorageTierMigrator::new(MigratorConfig {
            policies: vec![policy],
            ..MigratorConfig::default()
        });
        let block = make_block("c1", StorageTier::Hot, 512, 3, NOW, NOW);
        let action = m.evaluate_block(&block, NOW).expect("should trigger");
        assert_eq!(action.reason, MigrationReason::UnderAccessed);
    }

    #[test]
    fn test_evaluate_block_oversized_triggers() {
        let policy = make_policy(StorageTier::Hot, None, None, Some(1024), None);
        let m = StorageTierMigrator::new(MigratorConfig {
            policies: vec![policy],
            ..MigratorConfig::default()
        });
        let block = make_block("c1", StorageTier::Hot, 2048, 100, NOW, NOW);
        let action = m.evaluate_block(&block, NOW).expect("should trigger");
        assert_eq!(action.reason, MigrationReason::Oversized);
    }

    #[test]
    fn test_evaluate_block_undersized_triggers() {
        let policy = make_policy(StorageTier::Warm, None, None, None, Some(4096));
        let m = StorageTierMigrator::new(MigratorConfig {
            policies: vec![policy],
            ..MigratorConfig::default()
        });
        let block = make_block("c1", StorageTier::Warm, 512, 100, NOW, NOW);
        let action = m.evaluate_block(&block, NOW).expect("should trigger");
        assert_eq!(action.reason, MigrationReason::Undersized);
        assert_eq!(action.to_tier, StorageTier::Cold);
    }

    #[test]
    fn test_evaluate_block_no_trigger_when_within_policy() {
        let policy = make_policy(StorageTier::Hot, Some(3_600_000), Some(5), None, None);
        let m = StorageTierMigrator::new(MigratorConfig {
            policies: vec![policy],
            ..MigratorConfig::default()
        });
        // Accessed recently and often enough → no demotion.
        let block = make_block("c1", StorageTier::Hot, 512, 10, NOW - 100, NOW - 200);
        assert!(m.evaluate_block(&block, NOW).is_none());
    }

    #[test]
    fn test_evaluate_block_wrong_tier_policy_skipped() {
        // Policy is for Cold tier; block is Hot → no match → no migration.
        let policy = make_policy(StorageTier::Cold, Some(1_000), None, None, None);
        let m = StorageTierMigrator::new(MigratorConfig {
            policies: vec![policy],
            ..MigratorConfig::default()
        });
        let block = make_block("c1", StorageTier::Hot, 512, 0, NOW - 5_000, NOW - 5_000);
        assert!(m.evaluate_block(&block, NOW).is_none());
    }

    #[test]
    fn test_evaluate_archive_block_cannot_demote() {
        // Archive has no next-colder tier, so even if a policy fires, no action
        // is created (next_colder() returns None → evaluate_block returns None).
        let policy = make_policy(StorageTier::Archive, Some(1_000), None, None, None);
        let m = StorageTierMigrator::new(MigratorConfig {
            policies: vec![policy],
            ..MigratorConfig::default()
        });
        let block = make_block("c1", StorageTier::Archive, 512, 0, NOW - 5_000, NOW - 5_000);
        assert!(m.evaluate_block(&block, NOW).is_none());
    }

    // -----------------------------------------------------------------------
    // plan_migrations tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_plan_migrations_empty() {
        let m = simple_migrator();
        assert!(m.plan_migrations(NOW).is_empty());
    }

    #[test]
    fn test_plan_migrations_detects_stale_blocks() {
        let mut m = simple_migrator();
        // last_accessed = NOW - 7200000 ms (2h ago), max_age = 1h → stale
        m.register_block(make_block(
            "c1",
            StorageTier::Hot,
            512,
            10,
            NOW - 7_200_000,
            NOW - 7_200_000,
        ));
        let actions = m.plan_migrations(NOW);
        assert_eq!(actions.len(), 1);
        assert_eq!(actions[0].cid, "c1");
    }

    #[test]
    fn test_plan_migrations_sorted_by_priority() {
        let p_hot = make_policy(StorageTier::Hot, Some(1_000), None, None, None);
        let p_warm = make_policy(StorageTier::Warm, Some(1_000), None, None, None);
        let mut m = StorageTierMigrator::new(MigratorConfig {
            policies: vec![p_hot, p_warm],
            ..MigratorConfig::default()
        });
        m.register_block(make_block(
            "warm1",
            StorageTier::Warm,
            512,
            0,
            NOW - 5_000,
            NOW - 5_000,
        ));
        m.register_block(make_block(
            "hot1",
            StorageTier::Hot,
            512,
            0,
            NOW - 5_000,
            NOW - 5_000,
        ));
        let actions = m.plan_migrations(NOW);
        assert_eq!(actions.len(), 2);
        // Hot actions come first (priority=3 > priority=2)
        assert_eq!(actions[0].from_tier, StorageTier::Hot);
        assert_eq!(actions[1].from_tier, StorageTier::Warm);
    }

    #[test]
    fn test_plan_migrations_respects_max_cap() {
        let policy = make_policy(StorageTier::Hot, Some(1_000), None, None, None);
        let mut m = StorageTierMigrator::new(MigratorConfig {
            policies: vec![policy],
            max_migrations_per_run: 3,
            ..MigratorConfig::default()
        });
        for i in 0..10 {
            m.register_block(make_block(
                &format!("c{i}"),
                StorageTier::Hot,
                512,
                0,
                NOW - 5_000,
                NOW - 5_000,
            ));
        }
        let actions = m.plan_migrations(NOW);
        assert_eq!(actions.len(), 3);
    }

    // -----------------------------------------------------------------------
    // execute_migrations tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_execute_migrations_updates_tier() {
        let mut m = simple_migrator();
        m.register_block(make_block(
            "c1",
            StorageTier::Hot,
            1024,
            0,
            NOW - 7_200_000,
            NOW,
        ));
        let actions = m.plan_migrations(NOW);
        let result = m.execute_migrations(actions, NOW);
        assert_eq!(result.actions_executed, 1);
        assert_eq!(result.bytes_migrated, 1024);
        let b = m.blocks.get("c1").expect("block must exist");
        assert_eq!(b.current_tier, StorageTier::Warm);
    }

    #[test]
    fn test_execute_migrations_dry_run_does_not_mutate() {
        let policy = make_policy(StorageTier::Hot, Some(1_000), None, None, None);
        let mut m = StorageTierMigrator::new(MigratorConfig {
            policies: vec![policy],
            dry_run: true,
            ..MigratorConfig::default()
        });
        m.register_block(make_block(
            "c1",
            StorageTier::Hot,
            512,
            0,
            NOW - 5_000,
            NOW - 5_000,
        ));
        let actions = m.plan_migrations(NOW);
        let result = m.execute_migrations(actions, NOW);
        // In dry-run, planned count recorded but tier unchanged.
        assert!(result.actions_planned >= 1);
        assert_eq!(result.actions_executed, 0);
        let b = m.blocks.get("c1").expect("block must exist");
        assert_eq!(b.current_tier, StorageTier::Hot);
    }

    #[test]
    fn test_execute_migrations_missing_block_recorded_as_failed() {
        use super::MigrationAction;
        let mut m = simple_migrator();
        let action = MigrationAction {
            cid: "ghost".to_string(),
            from_tier: StorageTier::Hot,
            to_tier: StorageTier::Warm,
            reason: MigrationReason::TooOld,
            scheduled_at: NOW,
        };
        let result = m.execute_migrations(vec![action], NOW);
        assert_eq!(result.failed_cids, vec!["ghost".to_string()]);
        assert_eq!(result.actions_executed, 0);
    }

    #[test]
    fn test_execute_migrations_batching() {
        let policy = make_policy(StorageTier::Hot, Some(1_000), None, None, None);
        let mut m = StorageTierMigrator::new(MigratorConfig {
            policies: vec![policy],
            batch_size: 2,
            max_migrations_per_run: 1000,
            ..MigratorConfig::default()
        });
        for i in 0..6 {
            m.register_block(make_block(
                &format!("c{i}"),
                StorageTier::Hot,
                256,
                0,
                NOW - 5_000,
                NOW - 5_000,
            ));
        }
        let actions = m.plan_migrations(NOW);
        let result = m.execute_migrations(actions, NOW);
        assert_eq!(result.actions_executed, 6);
    }

    // -----------------------------------------------------------------------
    // run_migration_cycle tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_run_migration_cycle_full_flow() {
        let mut m = simple_migrator();
        m.register_block(make_block(
            "c1",
            StorageTier::Hot,
            2048,
            2,
            NOW - 7_200_000,
            NOW,
        ));
        let result = m.run_migration_cycle(NOW);
        assert!(result.actions_planned >= 1);
        assert!(result.actions_executed >= 1);
    }

    #[test]
    fn test_run_migration_cycle_updates_counters() {
        let mut m = simple_migrator();
        m.register_block(make_block(
            "c1",
            StorageTier::Hot,
            2048,
            2,
            NOW - 7_200_000,
            NOW,
        ));
        m.run_migration_cycle(NOW);
        assert!(m.total_migrations >= 1);
        assert!(m.total_bytes_migrated >= 2048);
    }

    #[test]
    fn test_run_migration_cycle_empty_produces_zero_result() {
        let mut m = simple_migrator();
        let result = m.run_migration_cycle(NOW);
        assert_eq!(result.actions_planned, 0);
        assert_eq!(result.actions_executed, 0);
        assert_eq!(result.bytes_migrated, 0);
    }

    // -----------------------------------------------------------------------
    // blocks_in_tier tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_blocks_in_tier_empty() {
        let m = simple_migrator();
        assert!(m.blocks_in_tier(&StorageTier::Hot).is_empty());
    }

    #[test]
    fn test_blocks_in_tier_correct_tier() {
        let mut m = simple_migrator();
        m.register_block(make_block("hot1", StorageTier::Hot, 512, 5, NOW, NOW));
        m.register_block(make_block("warm1", StorageTier::Warm, 512, 5, NOW, NOW));
        let hot = m.blocks_in_tier(&StorageTier::Hot);
        assert_eq!(hot.len(), 1);
        assert_eq!(hot[0].cid, "hot1");
    }

    #[test]
    fn test_blocks_in_tier_sorted_by_access_count_desc() {
        let mut m = simple_migrator();
        m.register_block(make_block("c1", StorageTier::Warm, 512, 1, NOW, NOW));
        m.register_block(make_block("c2", StorageTier::Warm, 512, 50, NOW, NOW));
        m.register_block(make_block("c3", StorageTier::Warm, 512, 10, NOW, NOW));
        let warm = m.blocks_in_tier(&StorageTier::Warm);
        assert_eq!(warm.len(), 3);
        assert_eq!(warm[0].access_count, 50);
        assert_eq!(warm[1].access_count, 10);
        assert_eq!(warm[2].access_count, 1);
    }

    // -----------------------------------------------------------------------
    // migration_log tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_migration_log_appended_on_execute() {
        let mut m = simple_migrator();
        m.register_block(make_block(
            "c1",
            StorageTier::Hot,
            512,
            0,
            NOW - 7_200_000,
            NOW,
        ));
        m.run_migration_cycle(NOW);
        assert!(!m.migration_log().is_empty());
        assert_eq!(m.migration_log()[0].cid, "c1");
    }

    #[test]
    fn test_migration_log_bounded_at_capacity() {
        use super::LOG_CAPACITY;
        let policy = make_policy(StorageTier::Hot, Some(1), None, None, None);
        let mut m = StorageTierMigrator::new(MigratorConfig {
            policies: vec![policy],
            max_migrations_per_run: LOG_CAPACITY + 100,
            ..MigratorConfig::default()
        });
        // Register LOG_CAPACITY + 100 blocks, each stale.
        for i in 0..(LOG_CAPACITY + 100) {
            m.register_block(make_block(
                &format!("c{i}"),
                StorageTier::Hot,
                64,
                0,
                NOW - 5_000,
                NOW - 5_000,
            ));
        }
        m.run_migration_cycle(NOW);
        assert!(m.migration_log().len() <= LOG_CAPACITY);
    }

    #[test]
    fn test_migration_log_dry_run_appends_but_no_exec() {
        let policy = make_policy(StorageTier::Hot, Some(1_000), None, None, None);
        let mut m = StorageTierMigrator::new(MigratorConfig {
            policies: vec![policy],
            dry_run: true,
            ..MigratorConfig::default()
        });
        m.register_block(make_block("c1", StorageTier::Hot, 512, 0, NOW - 5_000, NOW));
        let actions = m.plan_migrations(NOW);
        m.execute_migrations(actions, NOW);
        // In dry-run, log receives entries but block is not mutated.
        assert!(!m.migration_log().is_empty());
        let b = m.blocks.get("c1").expect("block must exist");
        assert_eq!(b.current_tier, StorageTier::Hot);
    }

    // -----------------------------------------------------------------------
    // migrator_stats tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_migrator_stats_initial() {
        let m = simple_migrator();
        let s = m.migrator_stats();
        assert_eq!(s.total_blocks, 0);
        assert_eq!(s.total_migrations, 0);
        assert_eq!(s.total_bytes_migrated, 0);
    }

    #[test]
    fn test_migrator_stats_tier_counts() {
        let mut m = simple_migrator();
        m.register_block(make_block("h1", StorageTier::Hot, 256, 0, NOW, NOW));
        m.register_block(make_block("h2", StorageTier::Hot, 256, 0, NOW, NOW));
        m.register_block(make_block("w1", StorageTier::Warm, 256, 0, NOW, NOW));
        m.register_block(make_block("c1", StorageTier::Cold, 256, 0, NOW, NOW));
        m.register_block(make_block("a1", StorageTier::Archive, 256, 0, NOW, NOW));
        let s = m.migrator_stats();
        assert_eq!(s.total_blocks, 5);
        assert_eq!(s.hot_blocks, 2);
        assert_eq!(s.warm_blocks, 1);
        assert_eq!(s.cold_blocks, 1);
        assert_eq!(s.archive_blocks, 1);
    }

    #[test]
    fn test_migrator_stats_after_migration() {
        let mut m = simple_migrator();
        m.register_block(make_block(
            "c1",
            StorageTier::Hot,
            1024,
            0,
            NOW - 7_200_000,
            NOW,
        ));
        m.run_migration_cycle(NOW);
        let s = m.migrator_stats();
        assert!(s.total_migrations >= 1);
        assert!(s.total_bytes_migrated >= 1024);
        assert_eq!(s.hot_blocks, 0);
        assert_eq!(s.warm_blocks, 1);
    }

    // -----------------------------------------------------------------------
    // Multi-policy and multi-tier cascade tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_multi_tier_cascade_two_cycles() {
        let p_hot = make_policy(StorageTier::Hot, Some(1_000), None, None, None);
        let p_warm = make_policy(StorageTier::Warm, Some(1_000), None, None, None);
        let mut m = StorageTierMigrator::new(MigratorConfig {
            policies: vec![p_hot, p_warm],
            ..MigratorConfig::default()
        });
        m.register_block(make_block(
            "c1",
            StorageTier::Hot,
            512,
            0,
            NOW - 5_000,
            NOW - 5_000,
        ));
        // First cycle: Hot → Warm
        m.run_migration_cycle(NOW);
        assert_eq!(m.blocks["c1"].current_tier, StorageTier::Warm);
        // Second cycle: Warm → Cold (last_accessed_ms is still old)
        m.run_migration_cycle(NOW + 10_000);
        assert_eq!(m.blocks["c1"].current_tier, StorageTier::Cold);
    }

    #[test]
    fn test_age_check_boundary_exact() {
        // Exactly at the boundary: last_accessed + max_age == now → NOT stale.
        let policy = make_policy(StorageTier::Hot, Some(1_000), None, None, None);
        let m = StorageTierMigrator::new(MigratorConfig {
            policies: vec![policy],
            ..MigratorConfig::default()
        });
        let block = make_block("c1", StorageTier::Hot, 512, 0, NOW - 1_000, NOW - 1_000);
        // last_accessed_ms (NOW-1000) + max_age (1000) = NOW, which is NOT < NOW
        assert!(m.evaluate_block(&block, NOW).is_none());
    }

    #[test]
    fn test_age_check_boundary_one_over() {
        // One ms past the boundary → stale.
        let policy = make_policy(StorageTier::Hot, Some(1_000), None, None, None);
        let m = StorageTierMigrator::new(MigratorConfig {
            policies: vec![policy],
            ..MigratorConfig::default()
        });
        let block = make_block("c1", StorageTier::Hot, 512, 0, NOW - 1_001, NOW - 1_001);
        assert!(m.evaluate_block(&block, NOW).is_some());
    }

    #[test]
    fn test_reason_display() {
        assert_eq!(MigrationReason::TooOld.to_string(), "too_old");
        assert_eq!(MigrationReason::UnderAccessed.to_string(), "under_accessed");
        assert_eq!(MigrationReason::Oversized.to_string(), "oversized");
        assert_eq!(MigrationReason::Undersized.to_string(), "undersized");
        assert_eq!(MigrationReason::PolicyChange.to_string(), "policy_change");
        assert_eq!(MigrationReason::Manual.to_string(), "manual");
    }

    #[test]
    fn test_multiple_blocks_partial_migration() {
        let policy = make_policy(StorageTier::Hot, Some(1_000), None, None, None);
        let mut m = StorageTierMigrator::new(MigratorConfig {
            policies: vec![policy],
            ..MigratorConfig::default()
        });
        // One stale, one fresh.
        m.register_block(make_block(
            "stale",
            StorageTier::Hot,
            512,
            0,
            NOW - 5_000,
            NOW,
        ));
        m.register_block(make_block(
            "fresh",
            StorageTier::Hot,
            512,
            0,
            NOW - 100,
            NOW,
        ));
        let result = m.run_migration_cycle(NOW);
        assert_eq!(result.actions_planned, 1);
        assert_eq!(result.actions_executed, 1);
        assert_eq!(m.blocks["stale"].current_tier, StorageTier::Warm);
        assert_eq!(m.blocks["fresh"].current_tier, StorageTier::Hot);
    }

    #[test]
    fn test_default_config_values() {
        let c = MigratorConfig::default();
        assert!(!c.dry_run);
        assert_eq!(c.batch_size, 100);
        assert_eq!(c.max_migrations_per_run, 1000);
        assert!(c.policies.is_empty());
    }

    #[test]
    fn test_bytes_migrated_accumulates_across_cycles() {
        let policy = make_policy(StorageTier::Hot, Some(1_000), None, None, None);
        let mut m = StorageTierMigrator::new(MigratorConfig {
            policies: vec![policy.clone()],
            ..MigratorConfig::default()
        });
        m.register_block(make_block(
            "c1",
            StorageTier::Hot,
            1024,
            0,
            NOW - 5_000,
            NOW,
        ));
        m.run_migration_cycle(NOW);
        assert_eq!(m.total_bytes_migrated, 1024);
    }

    #[test]
    fn test_zero_batch_size_handled_gracefully() {
        // batch_size=0 is clamped to 1 in execute_migrations.
        let policy = make_policy(StorageTier::Hot, Some(1_000), None, None, None);
        let mut m = StorageTierMigrator::new(MigratorConfig {
            policies: vec![policy],
            batch_size: 0,
            ..MigratorConfig::default()
        });
        m.register_block(make_block("c1", StorageTier::Hot, 512, 0, NOW - 5_000, NOW));
        let result = m.run_migration_cycle(NOW);
        assert_eq!(result.actions_executed, 1);
    }
}
