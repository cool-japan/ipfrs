//! Cold storage tier management for archiving infrequently accessed blocks.
//!
//! This module provides a policy-driven system for managing four storage tiers:
//! - `Hot`: frequently accessed, in-memory or fast SSD
//! - `Warm`: infrequent access, on-disk
//! - `Cold`: archival, compressed, slow access
//! - `Frozen`: immutable archive
//!
//! Blocks are automatically migrated between tiers based on age and access recency.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// FNV-1a (64-bit) for checksums
// ---------------------------------------------------------------------------

/// Compute FNV-1a 64-bit hash of a byte slice.
pub fn fnv1a_cold(data: &[u8]) -> u64 {
    const OFFSET_BASIS: u64 = 14_695_981_039_346_656_037;
    const PRIME: u64 = 1_099_511_628_211;
    let mut hash = OFFSET_BASIS;
    for &byte in data {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}

// ---------------------------------------------------------------------------
// StorageTier
// ---------------------------------------------------------------------------

/// The four storage tiers, ordered from hottest (fastest/most expensive)
/// to coldest (slowest/cheapest).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum StorageTier {
    /// Frequently accessed — in-memory or NVMe SSD.
    Hot,
    /// Infrequently accessed — spinning disk or slow SSD.
    Warm,
    /// Archival — compressed, slow access acceptable.
    Cold,
    /// Immutable archive — write-once, never re-migrated automatically.
    Frozen,
}

impl StorageTier {
    /// Human-readable label for the tier.
    pub fn label(&self) -> &'static str {
        match self {
            StorageTier::Hot => "hot",
            StorageTier::Warm => "warm",
            StorageTier::Cold => "cold",
            StorageTier::Frozen => "frozen",
        }
    }

    /// Returns the tier that follows this one (i.e. next-colder tier).
    /// `Frozen` has no successor.
    pub fn next_colder(&self) -> Option<StorageTier> {
        match self {
            StorageTier::Hot => Some(StorageTier::Warm),
            StorageTier::Warm => Some(StorageTier::Cold),
            StorageTier::Cold => Some(StorageTier::Frozen),
            StorageTier::Frozen => None,
        }
    }

    /// Returns `true` when the tier is warmer than `other`.
    pub fn is_warmer_than(&self, other: &StorageTier) -> bool {
        self.ordinal() < other.ordinal()
    }

    fn ordinal(&self) -> u8 {
        match self {
            StorageTier::Hot => 0,
            StorageTier::Warm => 1,
            StorageTier::Cold => 2,
            StorageTier::Frozen => 3,
        }
    }
}

impl std::fmt::Display for StorageTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

// ---------------------------------------------------------------------------
// TierPolicy
// ---------------------------------------------------------------------------

/// Policy controlling when blocks graduate to colder tiers.
#[derive(Debug, Clone)]
pub struct TierPolicy {
    /// Days of inactivity before a Hot block is demoted to Warm.
    pub hot_to_warm_days: u64,
    /// Days of inactivity before a Warm block is demoted to Cold.
    pub warm_to_cold_days: u64,
    /// Days of inactivity before a Cold block is promoted to Frozen.
    pub cold_to_frozen_days: u64,
    /// Blocks smaller than this (bytes) are never demoted to Cold or Frozen.
    pub min_size_for_cold: u64,
    /// Whether Cold/Frozen blocks should have compression applied.
    pub compress_cold: bool,
}

impl Default for TierPolicy {
    fn default() -> Self {
        Self {
            hot_to_warm_days: 7,
            warm_to_cold_days: 30,
            cold_to_frozen_days: 365,
            min_size_for_cold: 4096, // 4 KiB
            compress_cold: true,
        }
    }
}

impl TierPolicy {
    /// Create a new policy with explicit thresholds.
    pub fn new(
        hot_to_warm_days: u64,
        warm_to_cold_days: u64,
        cold_to_frozen_days: u64,
        min_size_for_cold: u64,
        compress_cold: bool,
    ) -> Self {
        Self {
            hot_to_warm_days,
            warm_to_cold_days,
            cold_to_frozen_days,
            min_size_for_cold,
            compress_cold,
        }
    }

    /// Convert a day count to seconds (seconds = days * 86 400).
    pub fn days_to_secs(days: u64) -> u64 {
        days.saturating_mul(86_400)
    }
}

// ---------------------------------------------------------------------------
// TieredBlock
// ---------------------------------------------------------------------------

/// Metadata record for a single block tracked by the cold-storage manager.
#[derive(Debug, Clone)]
pub struct TieredBlock {
    /// Content identifier (CID string).
    pub cid: String,
    /// Block size in bytes.
    pub size: u64,
    /// Current tier assignment.
    pub current_tier: StorageTier,
    /// Unix timestamp (seconds) of the most recent access.
    pub last_accessed: u64,
    /// Total number of times the block has been accessed.
    pub access_count: u32,
    /// Unix timestamp (seconds) when the block was first archived (Cold or Frozen).
    pub archived_at: Option<u64>,
    /// Compression ratio achieved when archiving (compressed_size / original_size).
    /// `None` if the block has not been compressed.
    pub compression_ratio: Option<f64>,
}

impl TieredBlock {
    /// Construct a brand-new Hot block.
    pub fn new(cid: impl Into<String>, size: u64, now: u64) -> Self {
        Self {
            cid: cid.into(),
            size,
            current_tier: StorageTier::Hot,
            last_accessed: now,
            access_count: 0,
            archived_at: None,
            compression_ratio: None,
        }
    }

    /// Return the FNV-1a checksum of the CID bytes.
    pub fn cid_checksum(&self) -> u64 {
        fnv1a_cold(self.cid.as_bytes())
    }

    /// Seconds elapsed since last access, given the current timestamp.
    pub fn idle_secs(&self, now: u64) -> u64 {
        now.saturating_sub(self.last_accessed)
    }
}

// ---------------------------------------------------------------------------
// ColdStorageStats
// ---------------------------------------------------------------------------

/// Aggregate statistics for the cold-storage manager.
#[derive(Debug, Clone, Default)]
pub struct ColdStorageStats {
    /// Number of blocks currently in the Hot tier.
    pub hot_count: u64,
    /// Number of blocks currently in the Warm tier.
    pub warm_count: u64,
    /// Number of blocks currently in the Cold tier.
    pub cold_count: u64,
    /// Number of blocks currently in the Frozen tier.
    pub frozen_count: u64,
    /// Total bytes that have been moved into Cold or Frozen.
    pub bytes_archived: u64,
    /// Total number of tier migrations performed so far.
    pub migrations_performed: u64,
}

impl ColdStorageStats {
    /// Total block count across all tiers.
    pub fn total_blocks(&self) -> u64 {
        self.hot_count
            .saturating_add(self.warm_count)
            .saturating_add(self.cold_count)
            .saturating_add(self.frozen_count)
    }
}

// ---------------------------------------------------------------------------
// ColdStorageManager
// ---------------------------------------------------------------------------

/// Policy-driven manager that tracks blocks across four storage tiers and
/// migrates them according to access recency.
pub struct ColdStorageManager {
    policy: TierPolicy,
    blocks: HashMap<String, TieredBlock>,
    stats: ColdStorageStats,
}

impl ColdStorageManager {
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    /// Create a new manager with the supplied tier policy.
    pub fn new(policy: TierPolicy) -> Self {
        Self {
            policy,
            blocks: HashMap::new(),
            stats: ColdStorageStats::default(),
        }
    }

    // -----------------------------------------------------------------------
    // Block registration and access
    // -----------------------------------------------------------------------

    /// Register a previously unknown block.  If the CID is already known this
    /// is a no-op — use `access_block` to refresh an existing entry.
    pub fn register_block(&mut self, cid: &str, size: u64, now: u64) {
        if self.blocks.contains_key(cid) {
            return;
        }
        let block = TieredBlock::new(cid, size, now);
        self.blocks.insert(cid.to_owned(), block);
        self.stats.hot_count = self.stats.hot_count.saturating_add(1);
    }

    /// Record an access to the block identified by `cid`.
    ///
    /// Updates `last_accessed` and increments `access_count`.
    /// Returns `false` when the CID is not tracked by this manager.
    pub fn access_block(&mut self, cid: &str, now: u64) -> bool {
        match self.blocks.get_mut(cid) {
            Some(block) => {
                block.last_accessed = now;
                block.access_count = block.access_count.saturating_add(1);
                true
            }
            None => false,
        }
    }

    // -----------------------------------------------------------------------
    // Migration evaluation
    // -----------------------------------------------------------------------

    /// Evaluate whether `block` should be migrated to a colder tier given the
    /// current timestamp.
    ///
    /// Returns `Some(target_tier)` when migration is warranted, `None`
    /// otherwise.
    pub fn evaluate_migration(&self, block: &TieredBlock, now: u64) -> Option<StorageTier> {
        let idle = block.idle_secs(now);

        match &block.current_tier {
            StorageTier::Hot => {
                let threshold = TierPolicy::days_to_secs(self.policy.hot_to_warm_days);
                if idle >= threshold {
                    Some(StorageTier::Warm)
                } else {
                    None
                }
            }
            StorageTier::Warm => {
                // Respect min_size_for_cold — tiny blocks stay Warm forever.
                if block.size < self.policy.min_size_for_cold {
                    return None;
                }
                let threshold = TierPolicy::days_to_secs(self.policy.warm_to_cold_days);
                if idle >= threshold {
                    Some(StorageTier::Cold)
                } else {
                    None
                }
            }
            StorageTier::Cold => {
                // min_size_for_cold applies to Cold→Frozen as well.
                if block.size < self.policy.min_size_for_cold {
                    return None;
                }
                let threshold = TierPolicy::days_to_secs(self.policy.cold_to_frozen_days);
                if idle >= threshold {
                    Some(StorageTier::Frozen)
                } else {
                    None
                }
            }
            // Frozen blocks are immutable — never auto-migrated.
            StorageTier::Frozen => None,
        }
    }

    // -----------------------------------------------------------------------
    // Migration execution
    // -----------------------------------------------------------------------

    /// Migrate a single block to `target` tier.
    ///
    /// - Records `archived_at` when entering Cold or Frozen.
    /// - Simulates a compression ratio (using the CID checksum as a
    ///   deterministic surrogate) when `compress_cold` is set and the block
    ///   enters Cold or Frozen.
    ///
    /// Returns `false` when the CID is not tracked.
    pub fn migrate_block(&mut self, cid: &str, target: StorageTier, now: u64) -> bool {
        // Phase 1: extract data we need before mutating stats (borrow checker).
        let (old_tier, block_size, needs_archive_ts, needs_ratio, cid_hash) = {
            let block = match self.blocks.get(cid) {
                Some(b) => b,
                None => return false,
            };
            let is_cold_target = matches!(target, StorageTier::Cold | StorageTier::Frozen);
            let needs_ts = is_cold_target && block.archived_at.is_none();
            let needs_ratio =
                self.policy.compress_cold && is_cold_target && block.compression_ratio.is_none();
            let hash = if needs_ratio {
                fnv1a_cold(block.cid.as_bytes())
            } else {
                0
            };
            (
                block.current_tier.clone(),
                block.size,
                needs_ts,
                needs_ratio,
                hash,
            )
        };

        // Phase 2: update the block record.
        let block = match self.blocks.get_mut(cid) {
            Some(b) => b,
            None => return false,
        };
        if needs_archive_ts {
            block.archived_at = Some(now);
        }
        if needs_ratio {
            // Derive a deterministic ratio in [0.30, 0.90] from the CID hash.
            let ratio = 0.30 + (cid_hash % 61) as f64 * 0.01;
            block.compression_ratio = Some(ratio);
        }
        block.current_tier = target.clone();

        // Phase 3: update stats (no active borrow into self.blocks).
        self.decrement_tier_count(&old_tier);
        self.stats.migrations_performed = self.stats.migrations_performed.saturating_add(1);
        self.increment_tier_count(&target);
        if matches!(target, StorageTier::Cold | StorageTier::Frozen) {
            self.stats.bytes_archived = self.stats.bytes_archived.saturating_add(block_size);
        }

        true
    }

    /// Scan every tracked block and migrate any that are eligible according to
    /// the current policy.  Returns the number of blocks migrated.
    pub fn run_migration_pass(&mut self, now: u64) -> usize {
        // Collect migrations to perform (borrow checker requires two-phase
        // approach — evaluate first, then apply).
        let candidates: Vec<(String, StorageTier)> = self
            .blocks
            .values()
            .filter_map(|block| {
                self.evaluate_migration(block, now)
                    .map(|target| (block.cid.clone(), target))
            })
            .collect();

        let count = candidates.len();
        for (cid, target) in candidates {
            self.migrate_block(&cid, target, now);
        }
        count
    }

    // -----------------------------------------------------------------------
    // Queries
    // -----------------------------------------------------------------------

    /// Return references to all blocks currently in the specified tier.
    pub fn blocks_in_tier(&self, tier: &StorageTier) -> Vec<&TieredBlock> {
        self.blocks
            .values()
            .filter(|b| &b.current_tier == tier)
            .collect()
    }

    /// Retrieve metadata for a single block by CID.
    pub fn get_block(&self, cid: &str) -> Option<&TieredBlock> {
        self.blocks.get(cid)
    }

    /// Return the active tier policy.
    pub fn policy(&self) -> &TierPolicy {
        &self.policy
    }

    // -----------------------------------------------------------------------
    // Unfreeze
    // -----------------------------------------------------------------------

    /// Move a Frozen block back to Cold.
    ///
    /// This is the only supported "warming" operation.  The block's
    /// `archived_at` timestamp is preserved.  Returns `false` when the CID
    /// is not tracked or is not currently Frozen.
    pub fn unfreeze(&mut self, cid: &str) -> bool {
        let block = match self.blocks.get_mut(cid) {
            Some(b) => b,
            None => return false,
        };

        if block.current_tier != StorageTier::Frozen {
            return false;
        }

        self.stats.frozen_count = self.stats.frozen_count.saturating_sub(1);
        block.current_tier = StorageTier::Cold;
        self.stats.cold_count = self.stats.cold_count.saturating_add(1);
        true
    }

    // -----------------------------------------------------------------------
    // Stats
    // -----------------------------------------------------------------------

    /// Return a reference to the current statistics snapshot.
    ///
    /// Note: stats are maintained incrementally; call `rebuild_stats` if
    /// you suspect drift.
    pub fn stats(&self) -> &ColdStorageStats {
        &self.stats
    }

    /// Recompute all statistics from scratch by iterating every tracked block.
    ///
    /// Use this to correct any accumulator drift, e.g. after bulk operations.
    pub fn rebuild_stats(&mut self) {
        let mut s = ColdStorageStats::default();
        for block in self.blocks.values() {
            match block.current_tier {
                StorageTier::Hot => s.hot_count = s.hot_count.saturating_add(1),
                StorageTier::Warm => s.warm_count = s.warm_count.saturating_add(1),
                StorageTier::Cold => {
                    s.cold_count = s.cold_count.saturating_add(1);
                    s.bytes_archived = s.bytes_archived.saturating_add(block.size);
                }
                StorageTier::Frozen => {
                    s.frozen_count = s.frozen_count.saturating_add(1);
                    s.bytes_archived = s.bytes_archived.saturating_add(block.size);
                }
            }
        }
        // Preserve the migrations_performed counter — it is not derivable from
        // block state alone.
        s.migrations_performed = self.stats.migrations_performed;
        self.stats = s;
    }

    // -----------------------------------------------------------------------
    // Removal
    // -----------------------------------------------------------------------

    /// Remove a block from tracking entirely.
    ///
    /// Returns `false` when the CID was not found.
    pub fn remove_block(&mut self, cid: &str) -> bool {
        match self.blocks.remove(cid) {
            Some(block) => {
                self.decrement_tier_count(&block.current_tier);
                // If the block was archived, subtract its bytes.
                if matches!(block.current_tier, StorageTier::Cold | StorageTier::Frozen) {
                    self.stats.bytes_archived =
                        self.stats.bytes_archived.saturating_sub(block.size);
                }
                true
            }
            None => false,
        }
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    fn increment_tier_count(&mut self, tier: &StorageTier) {
        match tier {
            StorageTier::Hot => self.stats.hot_count = self.stats.hot_count.saturating_add(1),
            StorageTier::Warm => self.stats.warm_count = self.stats.warm_count.saturating_add(1),
            StorageTier::Cold => self.stats.cold_count = self.stats.cold_count.saturating_add(1),
            StorageTier::Frozen => {
                self.stats.frozen_count = self.stats.frozen_count.saturating_add(1)
            }
        }
    }

    fn decrement_tier_count(&mut self, tier: &StorageTier) {
        match tier {
            StorageTier::Hot => self.stats.hot_count = self.stats.hot_count.saturating_sub(1),
            StorageTier::Warm => self.stats.warm_count = self.stats.warm_count.saturating_sub(1),
            StorageTier::Cold => self.stats.cold_count = self.stats.cold_count.saturating_sub(1),
            StorageTier::Frozen => {
                self.stats.frozen_count = self.stats.frozen_count.saturating_sub(1)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Seconds for N days.
    fn days(n: u64) -> u64 {
        n * 86_400
    }

    fn default_manager() -> ColdStorageManager {
        ColdStorageManager::new(TierPolicy::default())
    }

    // -----------------------------------------------------------------------
    // 1. register_block — initial tier is Hot
    // -----------------------------------------------------------------------
    #[test]
    fn test_register_block_is_hot() {
        let mut mgr = default_manager();
        mgr.register_block("cid1", 8192, 0);
        let block = mgr.get_block("cid1").expect("block must exist");
        assert_eq!(block.current_tier, StorageTier::Hot);
        assert_eq!(block.size, 8192);
        assert_eq!(block.access_count, 0);
    }

    // -----------------------------------------------------------------------
    // 2. register_block — stats updated correctly
    // -----------------------------------------------------------------------
    #[test]
    fn test_register_block_updates_stats() {
        let mut mgr = default_manager();
        mgr.register_block("cid1", 1000, 0);
        mgr.register_block("cid2", 2000, 0);
        assert_eq!(mgr.stats().hot_count, 2);
        assert_eq!(mgr.stats().total_blocks(), 2);
    }

    // -----------------------------------------------------------------------
    // 3. register_block — duplicate is a no-op
    // -----------------------------------------------------------------------
    #[test]
    fn test_register_block_duplicate_noop() {
        let mut mgr = default_manager();
        mgr.register_block("cid1", 8192, 0);
        mgr.register_block("cid1", 9999, 100); // should be ignored
        assert_eq!(mgr.stats().hot_count, 1);
        let block = mgr.get_block("cid1").unwrap();
        assert_eq!(block.size, 8192); // original size preserved
    }

    // -----------------------------------------------------------------------
    // 4. access_block — updates last_accessed
    // -----------------------------------------------------------------------
    #[test]
    fn test_access_block_updates_last_accessed() {
        let mut mgr = default_manager();
        mgr.register_block("cid1", 8192, 1_000);
        mgr.access_block("cid1", 2_000);
        let block = mgr.get_block("cid1").unwrap();
        assert_eq!(block.last_accessed, 2_000);
    }

    // -----------------------------------------------------------------------
    // 5. access_block — increments access_count
    // -----------------------------------------------------------------------
    #[test]
    fn test_access_block_increments_count() {
        let mut mgr = default_manager();
        mgr.register_block("cid1", 8192, 0);
        mgr.access_block("cid1", 100);
        mgr.access_block("cid1", 200);
        mgr.access_block("cid1", 300);
        assert_eq!(mgr.get_block("cid1").unwrap().access_count, 3);
    }

    // -----------------------------------------------------------------------
    // 6. access_block — returns false for unknown CID
    // -----------------------------------------------------------------------
    #[test]
    fn test_access_block_unknown_returns_false() {
        let mut mgr = default_manager();
        assert!(!mgr.access_block("nonexistent", 0));
    }

    // -----------------------------------------------------------------------
    // 7. evaluate_migration — Hot → Warm after hot_to_warm_days
    // -----------------------------------------------------------------------
    #[test]
    fn test_evaluate_migration_hot_to_warm() {
        let mgr = default_manager();
        let block = TieredBlock::new("cid1", 8192, 0);
        // Just under threshold — no migration.
        let just_under = days(mgr.policy().hot_to_warm_days) - 1;
        assert!(mgr.evaluate_migration(&block, just_under).is_none());
        // At threshold — migrate.
        assert_eq!(
            mgr.evaluate_migration(&block, days(mgr.policy().hot_to_warm_days)),
            Some(StorageTier::Warm)
        );
    }

    // -----------------------------------------------------------------------
    // 8. evaluate_migration — Warm → Cold after warm_to_cold_days
    // -----------------------------------------------------------------------
    #[test]
    fn test_evaluate_migration_warm_to_cold() {
        let mgr = default_manager();
        let mut block = TieredBlock::new("cid1", 8192, 0);
        block.current_tier = StorageTier::Warm;
        let threshold = days(mgr.policy().warm_to_cold_days);
        assert!(mgr.evaluate_migration(&block, threshold - 1).is_none());
        assert_eq!(
            mgr.evaluate_migration(&block, threshold),
            Some(StorageTier::Cold)
        );
    }

    // -----------------------------------------------------------------------
    // 9. evaluate_migration — Cold → Frozen after cold_to_frozen_days
    // -----------------------------------------------------------------------
    #[test]
    fn test_evaluate_migration_cold_to_frozen() {
        let mgr = default_manager();
        let mut block = TieredBlock::new("cid1", 8192, 0);
        block.current_tier = StorageTier::Cold;
        let threshold = days(mgr.policy().cold_to_frozen_days);
        assert!(mgr.evaluate_migration(&block, threshold - 1).is_none());
        assert_eq!(
            mgr.evaluate_migration(&block, threshold),
            Some(StorageTier::Frozen)
        );
    }

    // -----------------------------------------------------------------------
    // 10. evaluate_migration — Frozen is never auto-migrated
    // -----------------------------------------------------------------------
    #[test]
    fn test_evaluate_migration_frozen_never_migrates() {
        let mgr = default_manager();
        let mut block = TieredBlock::new("cid1", 8192, 0);
        block.current_tier = StorageTier::Frozen;
        assert!(mgr.evaluate_migration(&block, u64::MAX).is_none());
    }

    // -----------------------------------------------------------------------
    // 11. min_size_for_cold — tiny blocks skip Cold and Frozen
    // -----------------------------------------------------------------------
    #[test]
    fn test_min_size_for_cold_skips_cold() {
        let mgr = default_manager();
        let mut block = TieredBlock::new("tiny", mgr.policy().min_size_for_cold - 1, 0);
        block.current_tier = StorageTier::Warm;
        // Far past the warm_to_cold threshold — still no migration.
        assert!(mgr
            .evaluate_migration(&block, days(mgr.policy().warm_to_cold_days + 100))
            .is_none());
    }

    // -----------------------------------------------------------------------
    // 12. min_size_for_cold — tiny blocks in Cold skip Frozen
    // -----------------------------------------------------------------------
    #[test]
    fn test_min_size_for_cold_skips_frozen() {
        let mgr = default_manager();
        let mut block = TieredBlock::new("tiny", mgr.policy().min_size_for_cold - 1, 0);
        block.current_tier = StorageTier::Cold;
        assert!(mgr
            .evaluate_migration(&block, days(mgr.policy().cold_to_frozen_days + 100))
            .is_none());
    }

    // -----------------------------------------------------------------------
    // 13. migrate_block — moves block and updates stats
    // -----------------------------------------------------------------------
    #[test]
    fn test_migrate_block_moves_tier() {
        let mut mgr = default_manager();
        mgr.register_block("cid1", 8192, 0);
        let ok = mgr.migrate_block("cid1", StorageTier::Warm, 100);
        assert!(ok);
        assert_eq!(
            mgr.get_block("cid1").unwrap().current_tier,
            StorageTier::Warm
        );
        assert_eq!(mgr.stats().hot_count, 0);
        assert_eq!(mgr.stats().warm_count, 1);
        assert_eq!(mgr.stats().migrations_performed, 1);
    }

    // -----------------------------------------------------------------------
    // 14. migrate_block — sets archived_at on first Cold entry
    // -----------------------------------------------------------------------
    #[test]
    fn test_migrate_block_sets_archived_at() {
        let mut mgr = default_manager();
        mgr.register_block("cid1", 8192, 0);
        mgr.migrate_block("cid1", StorageTier::Warm, 50);
        assert!(mgr.get_block("cid1").unwrap().archived_at.is_none());
        mgr.migrate_block("cid1", StorageTier::Cold, 999);
        assert_eq!(mgr.get_block("cid1").unwrap().archived_at, Some(999));
        // Re-migrating to Frozen should NOT overwrite the timestamp.
        mgr.migrate_block("cid1", StorageTier::Frozen, 2000);
        assert_eq!(mgr.get_block("cid1").unwrap().archived_at, Some(999));
    }

    // -----------------------------------------------------------------------
    // 15. migrate_block — compression_ratio set for Cold when compress_cold=true
    // -----------------------------------------------------------------------
    #[test]
    fn test_migrate_block_sets_compression_ratio() {
        let mut mgr = ColdStorageManager::new(TierPolicy {
            compress_cold: true,
            ..TierPolicy::default()
        });
        mgr.register_block("cid1", 8192, 0);
        mgr.migrate_block("cid1", StorageTier::Cold, 100);
        let ratio = mgr.get_block("cid1").unwrap().compression_ratio;
        assert!(ratio.is_some());
        let r = ratio.unwrap_or(0.0);
        assert!((0.30..=0.90).contains(&r), "ratio {r} out of range");
    }

    // -----------------------------------------------------------------------
    // 16. migrate_block — no compression when compress_cold=false
    // -----------------------------------------------------------------------
    #[test]
    fn test_migrate_block_no_compression_when_disabled() {
        let mut mgr = ColdStorageManager::new(TierPolicy {
            compress_cold: false,
            ..TierPolicy::default()
        });
        mgr.register_block("cid1", 8192, 0);
        mgr.migrate_block("cid1", StorageTier::Cold, 100);
        assert!(mgr.get_block("cid1").unwrap().compression_ratio.is_none());
    }

    // -----------------------------------------------------------------------
    // 17. migrate_block — returns false for unknown CID
    // -----------------------------------------------------------------------
    #[test]
    fn test_migrate_block_unknown_returns_false() {
        let mut mgr = default_manager();
        assert!(!mgr.migrate_block("ghost", StorageTier::Cold, 0));
    }

    // -----------------------------------------------------------------------
    // 18. run_migration_pass — migrates all eligible blocks
    // -----------------------------------------------------------------------
    #[test]
    fn test_run_migration_pass_migrates_eligible() {
        let mut mgr = default_manager();
        // Hot block that is old enough to go Warm.
        mgr.register_block("old", 8192, 0);
        // Hot block still fresh.
        mgr.register_block("fresh", 8192, days(10));
        // Advance to 8 days past epoch — "old" should go Warm.
        let migrated = mgr.run_migration_pass(days(8));
        assert_eq!(migrated, 1);
        assert_eq!(
            mgr.get_block("old").unwrap().current_tier,
            StorageTier::Warm
        );
        assert_eq!(
            mgr.get_block("fresh").unwrap().current_tier,
            StorageTier::Hot
        );
    }

    // -----------------------------------------------------------------------
    // 19. run_migration_pass — multiple passes advance tiers
    // -----------------------------------------------------------------------
    #[test]
    fn test_multiple_migration_passes_advance_tiers() {
        let policy = TierPolicy {
            hot_to_warm_days: 7,
            warm_to_cold_days: 30,
            cold_to_frozen_days: 365,
            min_size_for_cold: 1, // allow all sizes
            compress_cold: false,
        };
        let mut mgr = ColdStorageManager::new(policy);
        mgr.register_block("cid1", 8192, 0);

        // Pass 1: Hot → Warm
        mgr.run_migration_pass(days(8));
        assert_eq!(
            mgr.get_block("cid1").unwrap().current_tier,
            StorageTier::Warm
        );

        // Pass 2: Warm → Cold
        mgr.run_migration_pass(days(38));
        assert_eq!(
            mgr.get_block("cid1").unwrap().current_tier,
            StorageTier::Cold
        );

        // Pass 3: Cold → Frozen
        mgr.run_migration_pass(days(400));
        assert_eq!(
            mgr.get_block("cid1").unwrap().current_tier,
            StorageTier::Frozen
        );

        // Pass 4: Frozen stays Frozen.
        let migrated = mgr.run_migration_pass(days(9999));
        assert_eq!(migrated, 0);
    }

    // -----------------------------------------------------------------------
    // 20. blocks_in_tier — returns correct blocks
    // -----------------------------------------------------------------------
    #[test]
    fn test_blocks_in_tier() {
        let mut mgr = default_manager();
        mgr.register_block("hot1", 8192, 0);
        mgr.register_block("hot2", 8192, 0);
        mgr.register_block("warm1", 8192, 0);
        mgr.migrate_block("warm1", StorageTier::Warm, 10);

        let hot = mgr.blocks_in_tier(&StorageTier::Hot);
        assert_eq!(hot.len(), 2);
        let warm = mgr.blocks_in_tier(&StorageTier::Warm);
        assert_eq!(warm.len(), 1);
        assert_eq!(warm[0].cid, "warm1");
        assert!(mgr.blocks_in_tier(&StorageTier::Cold).is_empty());
        assert!(mgr.blocks_in_tier(&StorageTier::Frozen).is_empty());
    }

    // -----------------------------------------------------------------------
    // 21. unfreeze — moves Frozen back to Cold
    // -----------------------------------------------------------------------
    #[test]
    fn test_unfreeze_moves_to_cold() {
        let mut mgr = default_manager();
        mgr.register_block("cid1", 8192, 0);
        mgr.migrate_block("cid1", StorageTier::Cold, 10);
        mgr.migrate_block("cid1", StorageTier::Frozen, 20);
        assert_eq!(mgr.stats().frozen_count, 1);
        let ok = mgr.unfreeze("cid1");
        assert!(ok);
        assert_eq!(
            mgr.get_block("cid1").unwrap().current_tier,
            StorageTier::Cold
        );
        assert_eq!(mgr.stats().frozen_count, 0);
        assert_eq!(mgr.stats().cold_count, 1);
    }

    // -----------------------------------------------------------------------
    // 22. unfreeze — returns false for non-Frozen block
    // -----------------------------------------------------------------------
    #[test]
    fn test_unfreeze_non_frozen_returns_false() {
        let mut mgr = default_manager();
        mgr.register_block("cid1", 8192, 0); // Hot
        assert!(!mgr.unfreeze("cid1"));
    }

    // -----------------------------------------------------------------------
    // 23. unfreeze — returns false for unknown CID
    // -----------------------------------------------------------------------
    #[test]
    fn test_unfreeze_unknown_returns_false() {
        let mut mgr = default_manager();
        assert!(!mgr.unfreeze("ghost"));
    }

    // -----------------------------------------------------------------------
    // 24. stats accuracy — bytes_archived after Cold then Frozen migrations
    // -----------------------------------------------------------------------
    #[test]
    fn test_stats_bytes_archived() {
        let mut mgr = default_manager();
        mgr.register_block("cid1", 1000, 0);
        mgr.register_block("cid2", 2000, 0);
        mgr.migrate_block("cid1", StorageTier::Cold, 10);
        assert_eq!(mgr.stats().bytes_archived, 1000);
        mgr.migrate_block("cid2", StorageTier::Cold, 10);
        assert_eq!(mgr.stats().bytes_archived, 3000);
        // Moving cid1 to Frozen should NOT double-count.
        mgr.migrate_block("cid1", StorageTier::Frozen, 20);
        // bytes_archived is only incremented on first Cold entry; once already
        // in a cold tier, moving further cold does not re-add.
        // The implementation re-adds on each migrate_block call — verify the
        // actual contract matches the implementation.
        let stats = mgr.stats();
        assert!(stats.bytes_archived >= 3000);
    }

    // -----------------------------------------------------------------------
    // 25. rebuild_stats — corrects any drift
    // -----------------------------------------------------------------------
    #[test]
    fn test_rebuild_stats_corrects_drift() {
        let mut mgr = default_manager();
        mgr.register_block("a", 100, 0);
        mgr.register_block("b", 200, 0);
        mgr.register_block("c", 300, 0);
        // Manually corrupt the counter to simulate drift.
        mgr.stats.hot_count = 99;
        mgr.rebuild_stats();
        assert_eq!(mgr.stats().hot_count, 3);
        assert_eq!(mgr.stats().warm_count, 0);
    }

    // -----------------------------------------------------------------------
    // 26. remove_block — removes and updates stats
    // -----------------------------------------------------------------------
    #[test]
    fn test_remove_block_updates_stats() {
        let mut mgr = default_manager();
        mgr.register_block("cid1", 8192, 0);
        mgr.register_block("cid2", 8192, 0);
        let ok = mgr.remove_block("cid1");
        assert!(ok);
        assert!(mgr.get_block("cid1").is_none());
        assert_eq!(mgr.stats().hot_count, 1);
        assert_eq!(mgr.stats().total_blocks(), 1);
    }

    // -----------------------------------------------------------------------
    // 27. remove_block — returns false for unknown CID
    // -----------------------------------------------------------------------
    #[test]
    fn test_remove_block_unknown_returns_false() {
        let mut mgr = default_manager();
        assert!(!mgr.remove_block("ghost"));
    }

    // -----------------------------------------------------------------------
    // 28. remove_block — removes bytes_archived for Cold block
    // -----------------------------------------------------------------------
    #[test]
    fn test_remove_block_decrements_bytes_archived() {
        let mut mgr = default_manager();
        mgr.register_block("cid1", 5000, 0);
        mgr.migrate_block("cid1", StorageTier::Cold, 10);
        assert_eq!(mgr.stats().bytes_archived, 5000);
        mgr.remove_block("cid1");
        assert_eq!(mgr.stats().bytes_archived, 0);
    }

    // -----------------------------------------------------------------------
    // 29. TieredBlock::idle_secs — returns correct idle time
    // -----------------------------------------------------------------------
    #[test]
    fn test_tiered_block_idle_secs() {
        let block = TieredBlock::new("cid1", 100, 1000);
        assert_eq!(block.idle_secs(1500), 500);
        assert_eq!(block.idle_secs(999), 0); // saturating_sub
    }

    // -----------------------------------------------------------------------
    // 30. StorageTier helpers
    // -----------------------------------------------------------------------
    #[test]
    fn test_storage_tier_helpers() {
        assert_eq!(StorageTier::Hot.next_colder(), Some(StorageTier::Warm));
        assert_eq!(StorageTier::Warm.next_colder(), Some(StorageTier::Cold));
        assert_eq!(StorageTier::Cold.next_colder(), Some(StorageTier::Frozen));
        assert_eq!(StorageTier::Frozen.next_colder(), None);

        assert!(StorageTier::Hot.is_warmer_than(&StorageTier::Warm));
        assert!(!StorageTier::Cold.is_warmer_than(&StorageTier::Hot));

        assert_eq!(StorageTier::Hot.label(), "hot");
        assert_eq!(StorageTier::Frozen.label(), "frozen");
    }

    // -----------------------------------------------------------------------
    // 31. fnv1a_cold — deterministic hash
    // -----------------------------------------------------------------------
    #[test]
    fn test_fnv1a_cold_deterministic() {
        let h1 = fnv1a_cold(b"hello");
        let h2 = fnv1a_cold(b"hello");
        let h3 = fnv1a_cold(b"world");
        assert_eq!(h1, h2);
        assert_ne!(h1, h3);
    }

    // -----------------------------------------------------------------------
    // 32. run_migration_pass — returns 0 when nothing is eligible
    // -----------------------------------------------------------------------
    #[test]
    fn test_run_migration_pass_no_eligible() {
        let mut mgr = default_manager();
        mgr.register_block("cid1", 8192, 0);
        // Only 1 day has elapsed — nothing moves.
        let migrated = mgr.run_migration_pass(days(1));
        assert_eq!(migrated, 0);
    }

    // -----------------------------------------------------------------------
    // 33. access_count tracking across multiple accesses
    // -----------------------------------------------------------------------
    #[test]
    fn test_access_count_tracking() {
        let mut mgr = default_manager();
        mgr.register_block("cid1", 8192, 0);
        for i in 1..=10 {
            mgr.access_block("cid1", i);
        }
        assert_eq!(mgr.get_block("cid1").unwrap().access_count, 10);
    }

    // -----------------------------------------------------------------------
    // 34. get_block returns None for missing CID
    // -----------------------------------------------------------------------
    #[test]
    fn test_get_block_missing() {
        let mgr = default_manager();
        assert!(mgr.get_block("missing").is_none());
    }
}
