//! Sharded block storage system for distributed content-addressed blocks.
//!
//! Distributes blocks across logical shards using CID-prefix routing (FNV-1a),
//! tracks per-shard statistics, and supports rebalancing and LRU eviction.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// FNV-1a (64-bit) — used for shard routing
// ---------------------------------------------------------------------------

#[inline]
fn fnv1a_64(data: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &byte in data {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

// ---------------------------------------------------------------------------
// ShardKey
// ---------------------------------------------------------------------------

/// Identifies which shard a block belongs to (0..num_shards).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ShardKey(pub u8);

impl ShardKey {
    /// Returns the inner shard index.
    #[inline]
    pub fn index(self) -> u8 {
        self.0
    }

    /// Returns the inner shard index as `usize`.
    #[inline]
    pub fn as_usize(self) -> usize {
        self.0 as usize
    }
}

// ---------------------------------------------------------------------------
// BlockRecord
// ---------------------------------------------------------------------------

/// A single content-addressed block stored in a shard.
#[derive(Clone, Debug)]
pub struct BlockRecord {
    /// The content identifier (CID) of this block.
    pub cid: String,
    /// Raw block data.
    pub data: Vec<u8>,
    /// The shard this block belongs to.
    pub shard: ShardKey,
    /// Unix timestamp (seconds) when the block was first inserted.
    pub inserted_at: u64,
    /// Number of times this block has been read via `get`.
    pub access_count: u64,
}

// ---------------------------------------------------------------------------
// ShardingConfig
// ---------------------------------------------------------------------------

/// Configuration for `BlockStoreSharding`.
#[derive(Clone, Debug)]
pub struct ShardingConfig {
    /// Number of logical shards (must be ≥ 1).
    pub num_shards: u8,
    /// Maximum number of blocks allowed per shard.
    pub max_blocks_per_shard: usize,
    /// Rebalance if any shard exceeds `avg * (1 + rebalance_threshold)`.
    pub rebalance_threshold: f64,
}

impl Default for ShardingConfig {
    fn default() -> Self {
        Self {
            num_shards: 16,
            max_blocks_per_shard: 65536,
            rebalance_threshold: 0.25,
        }
    }
}

// ---------------------------------------------------------------------------
// ShardMetrics
// ---------------------------------------------------------------------------

/// Per-shard statistics.
#[derive(Clone, Debug, Default)]
pub struct ShardMetrics {
    /// Which shard these metrics describe.
    pub shard_id: u8,
    /// Current block count in this shard.
    pub block_count: usize,
    /// Total bytes stored in this shard.
    pub total_bytes: u64,
    /// Number of successful `get` calls that found a block.
    pub hit_count: u64,
    /// Number of `get` calls that found no block.
    pub miss_count: u64,
}

// ---------------------------------------------------------------------------
// BlockStoreGlobalStats
// ---------------------------------------------------------------------------

/// Aggregate statistics for the whole `BlockStoreSharding`.
#[derive(Clone, Debug)]
pub struct BlockStoreGlobalStats {
    /// Total block count across all shards.
    pub total_blocks: usize,
    /// Total bytes stored across all shards.
    pub total_bytes: u64,
    /// Cumulative insertions (including replacements).
    pub total_insertions: u64,
    /// Cumulative `get` calls.
    pub total_lookups: u64,
    /// Number of configured shards.
    pub num_shards: u8,
    /// Whether any shard currently exceeds the rebalance threshold.
    pub needs_rebalance: bool,
}

// ---------------------------------------------------------------------------
// ShardingError
// ---------------------------------------------------------------------------

/// Errors produced by `BlockStoreSharding` operations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ShardingError {
    /// The target shard has reached `max_blocks_per_shard`.
    ShardFull {
        /// The shard that is full.
        shard_id: u8,
    },
    /// A CID was expected to exist but could not be found.
    CidNotFound(String),
    /// The given shard id is out of range.
    InvalidShardId(u8),
}

impl std::fmt::Display for ShardingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ShardFull { shard_id } => {
                write!(f, "shard {} is full", shard_id)
            }
            Self::CidNotFound(cid) => write!(f, "CID not found: {}", cid),
            Self::InvalidShardId(id) => write!(f, "invalid shard id: {}", id),
        }
    }
}

impl std::error::Error for ShardingError {}

// ---------------------------------------------------------------------------
// BlockStoreSharding
// ---------------------------------------------------------------------------

/// A sharded, content-addressed block store.
///
/// Distributes blocks across `num_shards` logical shards using FNV-1a hashing
/// of the CID. Provides per-shard metrics, rebalancing, and LRU eviction.
pub struct BlockStoreSharding {
    /// Runtime configuration.
    pub config: ShardingConfig,
    /// Per-shard block maps: `shard_id → (cid → BlockRecord)`.
    shards: Vec<HashMap<String, BlockRecord>>,
    /// Per-shard statistics.
    shard_metrics: Vec<ShardMetrics>,
    /// Total insertions since creation (new + replaced).
    pub total_insertions: u64,
    /// Total `get` calls since creation.
    pub total_lookups: u64,
    /// Total bytes currently stored.
    pub total_bytes: u64,
}

impl BlockStoreSharding {
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    /// Creates a new `BlockStoreSharding` with the given configuration.
    ///
    /// Panics if `config.num_shards == 0`.
    pub fn new(config: ShardingConfig) -> Self {
        assert!(config.num_shards > 0, "num_shards must be > 0");
        let n = config.num_shards as usize;
        let mut shard_metrics: Vec<ShardMetrics> = Vec::with_capacity(n);
        for i in 0..n {
            shard_metrics.push(ShardMetrics {
                shard_id: i as u8,
                ..ShardMetrics::default()
            });
        }
        Self {
            shards: (0..n).map(|_| HashMap::new()).collect(),
            shard_metrics,
            config,
            total_insertions: 0,
            total_lookups: 0,
            total_bytes: 0,
        }
    }

    // -----------------------------------------------------------------------
    // Routing
    // -----------------------------------------------------------------------

    /// Deterministically maps a CID string to a shard using FNV-1a % num_shards.
    pub fn shard_for_cid(&self, cid: &str) -> ShardKey {
        let hash = fnv1a_64(cid.as_bytes());
        let idx = (hash % self.config.num_shards as u64) as u8;
        ShardKey(idx)
    }

    // -----------------------------------------------------------------------
    // Write
    // -----------------------------------------------------------------------

    /// Inserts or replaces a block.
    ///
    /// Returns `Ok(true)` if the block was newly inserted, `Ok(false)` if an
    /// existing block for the same CID was replaced.
    /// Returns `Err(ShardingError::ShardFull)` if the shard is at capacity and
    /// there is no existing entry for the CID to replace.
    pub fn put(&mut self, cid: String, data: Vec<u8>, now: u64) -> Result<bool, ShardingError> {
        let key = self.shard_for_cid(&cid);
        let idx = key.as_usize();

        let shard = &mut self.shards[idx];
        let metrics = &mut self.shard_metrics[idx];

        if let Some(existing) = shard.get_mut(&cid) {
            // Replace existing block
            self.total_bytes = self
                .total_bytes
                .saturating_sub(existing.data.len() as u64)
                .saturating_add(data.len() as u64);
            metrics.total_bytes = metrics
                .total_bytes
                .saturating_sub(existing.data.len() as u64)
                .saturating_add(data.len() as u64);
            existing.data = data;
            existing.inserted_at = now;
            self.total_insertions = self.total_insertions.saturating_add(1);
            return Ok(false);
        }

        // New block — check capacity
        if shard.len() >= self.config.max_blocks_per_shard {
            return Err(ShardingError::ShardFull { shard_id: key.0 });
        }

        let byte_len = data.len() as u64;
        let record = BlockRecord {
            cid: cid.clone(),
            data,
            shard: key,
            inserted_at: now,
            access_count: 0,
        };
        shard.insert(cid, record);
        metrics.block_count += 1;
        metrics.total_bytes = metrics.total_bytes.saturating_add(byte_len);
        self.total_bytes = self.total_bytes.saturating_add(byte_len);
        self.total_insertions = self.total_insertions.saturating_add(1);
        Ok(true)
    }

    // -----------------------------------------------------------------------
    // Read
    // -----------------------------------------------------------------------

    /// Looks up a block by CID.
    ///
    /// Updates the shard's `hit_count` / `miss_count` and, on hit, increments
    /// the record's `access_count`. Returns `None` if not found.
    pub fn get(&mut self, cid: &str) -> Option<&BlockRecord> {
        let key = self.shard_for_cid(cid);
        let idx = key.as_usize();
        self.total_lookups = self.total_lookups.saturating_add(1);

        if self.shards[idx].contains_key(cid) {
            self.shard_metrics[idx].hit_count = self.shard_metrics[idx].hit_count.saturating_add(1);
            let record = self.shards[idx].get_mut(cid)?;
            record.access_count = record.access_count.saturating_add(1);
            Some(record)
        } else {
            self.shard_metrics[idx].miss_count =
                self.shard_metrics[idx].miss_count.saturating_add(1);
            None
        }
    }

    /// Returns a mutable reference to a block record (no metrics update).
    pub fn get_mut(&mut self, cid: &str) -> Option<&mut BlockRecord> {
        let key = self.shard_for_cid(cid);
        let idx = key.as_usize();
        self.shards[idx].get_mut(cid)
    }

    /// Returns `true` if the CID is present in the store.
    pub fn contains(&self, cid: &str) -> bool {
        let key = self.shard_for_cid(cid);
        self.shards[key.as_usize()].contains_key(cid)
    }

    // -----------------------------------------------------------------------
    // Delete
    // -----------------------------------------------------------------------

    /// Removes a block by CID.
    ///
    /// Returns `true` if the block existed and was removed, `false` otherwise.
    pub fn remove(&mut self, cid: &str) -> bool {
        let key = self.shard_for_cid(cid);
        let idx = key.as_usize();
        if let Some(record) = self.shards[idx].remove(cid) {
            let byte_len = record.data.len() as u64;
            self.shard_metrics[idx].block_count =
                self.shard_metrics[idx].block_count.saturating_sub(1);
            self.shard_metrics[idx].total_bytes =
                self.shard_metrics[idx].total_bytes.saturating_sub(byte_len);
            self.total_bytes = self.total_bytes.saturating_sub(byte_len);
            true
        } else {
            false
        }
    }

    // -----------------------------------------------------------------------
    // Metrics
    // -----------------------------------------------------------------------

    /// Returns metrics for a specific shard, or `None` if the shard id is
    /// out of range.
    pub fn shard_metrics(&self, shard: ShardKey) -> Option<&ShardMetrics> {
        self.shard_metrics.get(shard.as_usize())
    }

    /// Returns a slice of per-shard metrics.
    pub fn all_shard_metrics(&self) -> &[ShardMetrics] {
        &self.shard_metrics
    }

    // -----------------------------------------------------------------------
    // Rebalancing
    // -----------------------------------------------------------------------

    /// Returns `true` when at least one shard exceeds `avg * (1 + threshold)`.
    ///
    /// Returns `false` when no blocks are stored.
    pub fn needs_rebalance(&self) -> bool {
        let total = self.total_block_count();
        if total == 0 {
            return false;
        }
        let avg = total as f64 / self.config.num_shards as f64;
        let ceiling = avg * (1.0 + self.config.rebalance_threshold);
        self.shards.iter().any(|s| s.len() as f64 > ceiling)
    }

    /// Rebalances the store by moving blocks from over-loaded shards to
    /// under-loaded shards.
    ///
    /// For each shard that exceeds the average by more than the threshold,
    /// the function removes the surplus blocks (those with the lowest
    /// `access_count`) and re-inserts them using their natural `shard_for_cid`
    /// routing. This is a no-op if the shard they naturally belong to has not
    /// changed (e.g. due to the same num_shards); however the inserted_at
    /// timestamp is refreshed to `now`.
    ///
    /// Returns the total number of blocks moved.
    pub fn rebalance(&mut self, now: u64) -> usize {
        let total = self.total_block_count();
        if total == 0 {
            return 0;
        }
        let avg = total as f64 / self.config.num_shards as f64;
        let ceiling = avg * (1.0 + self.config.rebalance_threshold);

        // Collect which shards are overloaded and by how much
        let overloaded: Vec<(usize, usize)> = self
            .shards
            .iter()
            .enumerate()
            .filter_map(|(i, s)| {
                let len = s.len() as f64;
                if len > ceiling {
                    let overflow = (len - avg.ceil()) as usize;
                    Some((i, overflow))
                } else {
                    None
                }
            })
            .collect();

        if overloaded.is_empty() {
            return 0;
        }

        let mut moved_total = 0usize;

        for (shard_idx, overflow_count) in overloaded {
            // Sort the CIDs in this shard by access_count ascending (evict coldest first)
            let mut cids_sorted: Vec<(String, u64)> = self.shards[shard_idx]
                .values()
                .map(|r| (r.cid.clone(), r.access_count))
                .collect();
            cids_sorted.sort_by_key(|(_, ac)| *ac);
            cids_sorted.truncate(overflow_count);

            // Extract these records
            let mut to_move: Vec<BlockRecord> = Vec::with_capacity(cids_sorted.len());
            for (cid, _) in &cids_sorted {
                if let Some(mut record) = self.shards[shard_idx].remove(cid.as_str()) {
                    let byte_len = record.data.len() as u64;
                    self.shard_metrics[shard_idx].block_count =
                        self.shard_metrics[shard_idx].block_count.saturating_sub(1);
                    self.shard_metrics[shard_idx].total_bytes = self.shard_metrics[shard_idx]
                        .total_bytes
                        .saturating_sub(byte_len);
                    self.total_bytes = self.total_bytes.saturating_sub(byte_len);
                    record.inserted_at = now;
                    to_move.push(record);
                }
            }

            // Re-insert with natural routing
            for mut record in to_move {
                let natural_key = self.shard_for_cid(&record.cid);
                let nat_idx = natural_key.as_usize();
                record.shard = natural_key;
                let byte_len = record.data.len() as u64;
                self.shards[nat_idx].insert(record.cid.clone(), record);
                self.shard_metrics[nat_idx].block_count += 1;
                self.shard_metrics[nat_idx].total_bytes = self.shard_metrics[nat_idx]
                    .total_bytes
                    .saturating_add(byte_len);
                self.total_bytes = self.total_bytes.saturating_add(byte_len);
                moved_total += 1;
            }
        }

        moved_total
    }

    // -----------------------------------------------------------------------
    // Eviction
    // -----------------------------------------------------------------------

    /// Evicts up to `count` blocks from the given shard, preferring blocks
    /// with the lowest `access_count` (LRU approximation).
    ///
    /// Returns the number of blocks actually evicted (may be less than `count`
    /// if the shard has fewer blocks).
    pub fn evict_lru(&mut self, shard: ShardKey, count: usize) -> usize {
        let idx = shard.as_usize();
        if idx >= self.shards.len() || count == 0 {
            return 0;
        }

        let mut cids_sorted: Vec<(String, u64)> = self.shards[idx]
            .values()
            .map(|r| (r.cid.clone(), r.access_count))
            .collect();
        cids_sorted.sort_by_key(|(_, ac)| *ac);
        let to_evict = count.min(cids_sorted.len());

        let mut evicted = 0usize;
        for (cid, _) in cids_sorted.into_iter().take(to_evict) {
            if let Some(record) = self.shards[idx].remove(&cid) {
                let byte_len = record.data.len() as u64;
                self.shard_metrics[idx].block_count =
                    self.shard_metrics[idx].block_count.saturating_sub(1);
                self.shard_metrics[idx].total_bytes =
                    self.shard_metrics[idx].total_bytes.saturating_sub(byte_len);
                self.total_bytes = self.total_bytes.saturating_sub(byte_len);
                evicted += 1;
            }
        }
        evicted
    }

    // -----------------------------------------------------------------------
    // Introspection
    // -----------------------------------------------------------------------

    /// Returns all CIDs stored in the given shard.
    pub fn cids_in_shard(&self, shard: ShardKey) -> Vec<&str> {
        let idx = shard.as_usize();
        if idx >= self.shards.len() {
            return Vec::new();
        }
        self.shards[idx].keys().map(|k| k.as_str()).collect()
    }

    /// Returns the total number of blocks across all shards.
    pub fn total_block_count(&self) -> usize {
        self.shards.iter().map(|s| s.len()).sum()
    }

    /// Returns the total bytes stored across all shards.
    pub fn total_bytes(&self) -> u64 {
        self.total_bytes
    }

    /// Returns a snapshot of global statistics.
    pub fn global_stats(&self) -> BlockStoreGlobalStats {
        BlockStoreGlobalStats {
            total_blocks: self.total_block_count(),
            total_bytes: self.total_bytes,
            total_insertions: self.total_insertions,
            total_lookups: self.total_lookups,
            num_shards: self.config.num_shards,
            needs_rebalance: self.needs_rebalance(),
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::{
        fnv1a_64, BlockRecord, BlockStoreGlobalStats, BlockStoreSharding, ShardKey, ShardMetrics,
        ShardingConfig, ShardingError,
    };

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn default_store() -> BlockStoreSharding {
        BlockStoreSharding::new(ShardingConfig::default())
    }

    fn small_store(num_shards: u8) -> BlockStoreSharding {
        BlockStoreSharding::new(ShardingConfig {
            num_shards,
            max_blocks_per_shard: 100,
            rebalance_threshold: 0.25,
        })
    }

    fn make_cid(n: usize) -> String {
        format!("bafy2bzaced{:040}", n)
    }

    #[allow(dead_code)]
    fn insert(store: &mut BlockStoreSharding, cid: &str, data: &[u8]) -> bool {
        store
            .put(cid.to_string(), data.to_vec(), 1000)
            .expect("insert failed")
    }

    // -----------------------------------------------------------------------
    // 1. new() initialises correct number of shards
    // -----------------------------------------------------------------------
    #[test]
    fn test_new_default_shards() {
        let store = default_store();
        assert_eq!(store.config.num_shards, 16);
        assert_eq!(store.shards.len(), 16);
        assert_eq!(store.shard_metrics.len(), 16);
    }

    // 2. new() with custom num_shards
    #[test]
    fn test_new_custom_shards() {
        let store = small_store(4);
        assert_eq!(store.config.num_shards, 4);
        assert_eq!(store.shards.len(), 4);
    }

    // 3. shard_for_cid returns index in range
    #[test]
    fn test_shard_for_cid_in_range() {
        let store = default_store();
        for i in 0..100usize {
            let cid = make_cid(i);
            let key = store.shard_for_cid(&cid);
            assert!(key.0 < store.config.num_shards);
        }
    }

    // 4. shard_for_cid is deterministic
    #[test]
    fn test_shard_for_cid_deterministic() {
        let store = default_store();
        let cid = make_cid(42);
        assert_eq!(store.shard_for_cid(&cid), store.shard_for_cid(&cid));
    }

    // 5. shard_for_cid matches fnv1a_64 formula
    #[test]
    fn test_shard_for_cid_formula() {
        let store = small_store(8);
        let cid = "QmTestCid12345";
        let expected_idx = (fnv1a_64(cid.as_bytes()) % 8) as u8;
        assert_eq!(store.shard_for_cid(cid).0, expected_idx);
    }

    // 6. put() returns true for new block
    #[test]
    fn test_put_new_returns_true() {
        let mut store = default_store();
        let result = store.put("cid1".to_string(), vec![1, 2, 3], 100);
        assert_eq!(result, Ok(true));
    }

    // 7. put() returns false for replacement
    #[test]
    fn test_put_replace_returns_false() {
        let mut store = default_store();
        store
            .put("cid1".to_string(), vec![1, 2, 3], 100)
            .expect("first insert");
        let result = store.put("cid1".to_string(), vec![4, 5, 6], 200);
        assert_eq!(result, Ok(false));
    }

    // 8. put() respects max_blocks_per_shard
    #[test]
    fn test_put_shard_full() {
        let mut store = BlockStoreSharding::new(ShardingConfig {
            num_shards: 1,
            max_blocks_per_shard: 2,
            rebalance_threshold: 0.25,
        });
        store.put("a".to_string(), vec![1], 0).expect("a");
        store.put("b".to_string(), vec![2], 0).expect("b");
        let result = store.put("c".to_string(), vec![3], 0);
        assert!(matches!(
            result,
            Err(ShardingError::ShardFull { shard_id: 0 })
        ));
    }

    // 9. put() replacement does not consume capacity
    #[test]
    fn test_put_replace_no_capacity_loss() {
        let mut store = BlockStoreSharding::new(ShardingConfig {
            num_shards: 1,
            max_blocks_per_shard: 1,
            rebalance_threshold: 0.25,
        });
        store.put("a".to_string(), vec![1], 0).expect("first");
        // Replacing should succeed even though capacity is 1
        let result = store.put("a".to_string(), vec![99], 1);
        assert_eq!(result, Ok(false));
    }

    // 10. total_insertions increments on each put
    #[test]
    fn test_total_insertions() {
        let mut store = default_store();
        store.put("c1".to_string(), vec![1], 0).expect("c1");
        store.put("c2".to_string(), vec![2], 0).expect("c2");
        store.put("c1".to_string(), vec![3], 1).expect("replace c1");
        assert_eq!(store.total_insertions, 3);
    }

    // 11. get() returns correct record
    #[test]
    fn test_get_returns_record() {
        let mut store = default_store();
        store
            .put("cid_x".to_string(), vec![10, 20], 500)
            .expect("put");
        let record = store.get("cid_x").expect("should exist");
        assert_eq!(record.cid, "cid_x");
        assert_eq!(record.data, vec![10, 20]);
        assert_eq!(record.inserted_at, 500);
    }

    // 12. get() increments access_count
    #[test]
    fn test_get_increments_access_count() {
        let mut store = default_store();
        store.put("cid_ac".to_string(), vec![1], 0).expect("put");
        store.get("cid_ac");
        store.get("cid_ac");
        let record = store.get("cid_ac").expect("record");
        assert_eq!(record.access_count, 3);
    }

    // 13. get() updates hit_count
    #[test]
    fn test_get_updates_hit_count() {
        let mut store = default_store();
        store.put("cid_h".to_string(), vec![1], 0).expect("put");
        let key = store.shard_for_cid("cid_h");
        store.get("cid_h");
        store.get("cid_h");
        assert_eq!(store.shard_metrics(key).expect("metrics").hit_count, 2);
    }

    // 14. get() on missing CID returns None and updates miss_count
    #[test]
    fn test_get_miss_updates_miss_count() {
        let mut store = default_store();
        let key = store.shard_for_cid("nonexistent");
        store.get("nonexistent");
        assert_eq!(store.shard_metrics(key).expect("metrics").miss_count, 1);
        assert!(store.get("nonexistent").is_none());
    }

    // 15. get() increments total_lookups
    #[test]
    fn test_get_increments_total_lookups() {
        let mut store = default_store();
        store.put("lk".to_string(), vec![1], 0).expect("put");
        store.get("lk");
        store.get("missing");
        assert_eq!(store.total_lookups, 2);
    }

    // 16. get_mut() returns mutable reference
    #[test]
    fn test_get_mut_returns_mutable() {
        let mut store = default_store();
        store.put("m".to_string(), vec![5], 0).expect("put");
        {
            let record = store.get_mut("m").expect("should exist");
            record.access_count = 99;
        }
        let record = store.get_mut("m").expect("still exists");
        assert_eq!(record.access_count, 99);
    }

    // 17. contains() returns true for inserted block
    #[test]
    fn test_contains_true() {
        let mut store = default_store();
        store.put("x".to_string(), vec![1], 0).expect("put");
        assert!(store.contains("x"));
    }

    // 18. contains() returns false for missing block
    #[test]
    fn test_contains_false() {
        let store = default_store();
        assert!(!store.contains("zzz"));
    }

    // 19. remove() returns true and decrements block_count
    #[test]
    fn test_remove_existing() {
        let mut store = default_store();
        store.put("r".to_string(), vec![1, 2, 3], 0).expect("put");
        assert!(store.remove("r"));
        assert!(!store.contains("r"));
    }

    // 20. remove() updates total_bytes
    #[test]
    fn test_remove_updates_total_bytes() {
        let mut store = default_store();
        store
            .put("rb".to_string(), vec![1, 2, 3, 4], 0)
            .expect("put");
        let before = store.total_bytes();
        store.remove("rb");
        assert_eq!(store.total_bytes(), before - 4);
    }

    // 21. remove() on missing CID returns false
    #[test]
    fn test_remove_missing_returns_false() {
        let mut store = default_store();
        assert!(!store.remove("no_such"));
    }

    // 22. total_block_count() sums across shards
    #[test]
    fn test_total_block_count() {
        let mut store = default_store();
        for i in 0..20usize {
            store.put(make_cid(i), vec![i as u8], 0).expect("put");
        }
        assert_eq!(store.total_block_count(), 20);
    }

    // 23. total_bytes() reflects put and remove
    #[test]
    fn test_total_bytes_consistency() {
        let mut store = default_store();
        store
            .put("a".to_string(), vec![0u8; 100], 0)
            .expect("put a");
        store
            .put("b".to_string(), vec![0u8; 200], 0)
            .expect("put b");
        assert_eq!(store.total_bytes(), 300);
        store.remove("a");
        assert_eq!(store.total_bytes(), 200);
    }

    // 24. shard_metrics() returns correct shard_id
    #[test]
    fn test_shard_metrics_correct_id() {
        let store = small_store(4);
        for i in 0..4u8 {
            let m = store.shard_metrics(ShardKey(i)).expect("metrics");
            assert_eq!(m.shard_id, i);
        }
    }

    // 25. shard_metrics() returns None for out-of-range shard
    #[test]
    fn test_shard_metrics_out_of_range() {
        let store = small_store(4);
        assert!(store.shard_metrics(ShardKey(4)).is_none());
    }

    // 26. all_shard_metrics() length matches num_shards
    #[test]
    fn test_all_shard_metrics_length() {
        let store = small_store(8);
        assert_eq!(store.all_shard_metrics().len(), 8);
    }

    // 27. cids_in_shard() returns correct CIDs
    #[test]
    fn test_cids_in_shard() {
        let mut store = small_store(1);
        store.put("p".to_string(), vec![1], 0).expect("put");
        store.put("q".to_string(), vec![2], 0).expect("put");
        let mut cids = store.cids_in_shard(ShardKey(0));
        cids.sort_unstable();
        assert!(cids.contains(&"p"));
        assert!(cids.contains(&"q"));
    }

    // 28. cids_in_shard() for out-of-range returns empty vec
    #[test]
    fn test_cids_in_shard_out_of_range() {
        let store = small_store(2);
        assert!(store.cids_in_shard(ShardKey(99)).is_empty());
    }

    // 29. needs_rebalance() false when empty
    #[test]
    fn test_needs_rebalance_empty() {
        let store = default_store();
        assert!(!store.needs_rebalance());
    }

    // 30. needs_rebalance() false when evenly distributed
    #[test]
    fn test_needs_rebalance_even() {
        // With 1 shard and threshold 0.25, a single block avg==1 ceiling==1.25
        // so 1 block should be fine
        let mut store = BlockStoreSharding::new(ShardingConfig {
            num_shards: 1,
            max_blocks_per_shard: 10000,
            rebalance_threshold: 0.25,
        });
        for i in 0..5usize {
            store.put(make_cid(i), vec![1], 0).expect("put");
        }
        // With 1 shard only: 5 blocks, avg=5, ceiling=6.25; 5 <= 6.25 → no rebalance
        assert!(!store.needs_rebalance());
    }

    // 31. needs_rebalance() true when highly imbalanced
    #[test]
    fn test_needs_rebalance_imbalanced() {
        // 2 shards, force all into shard 0 by using shard-1 store and injecting
        // directly. We need to craft CIDs that all hash to shard 0.
        let mut store = BlockStoreSharding::new(ShardingConfig {
            num_shards: 2,
            max_blocks_per_shard: 10000,
            rebalance_threshold: 0.10,
        });
        // Put many blocks – not all will go to shard 0 but we can check
        // that after enough insertions, if one shard has >> avg, it triggers.
        // To guarantee, directly insert into shards via normal API using CIDs
        // that we pre-compute to land on shard 0.
        let mut shard0_count = 0usize;
        let mut i = 0usize;
        while shard0_count < 20 {
            let cid = format!("rebal_cid_{}", i);
            let key = store.shard_for_cid(&cid);
            if key.0 == 0 {
                store.put(cid, vec![1], 0).expect("put");
                shard0_count += 1;
            }
            i += 1;
        }
        // shard0 has 20 blocks, shard1 has 0 → avg=10, ceiling=11 → imbalanced
        assert!(store.needs_rebalance());
    }

    // 32. rebalance() returns 0 when empty
    #[test]
    fn test_rebalance_empty() {
        let mut store = default_store();
        assert_eq!(store.rebalance(0), 0);
    }

    // 33. rebalance() reduces imbalance
    #[test]
    fn test_rebalance_reduces_imbalance() {
        let mut store = BlockStoreSharding::new(ShardingConfig {
            num_shards: 2,
            max_blocks_per_shard: 10000,
            rebalance_threshold: 0.10,
        });
        let mut shard0_count = 0usize;
        let mut i = 0usize;
        while shard0_count < 20 {
            let cid = format!("rb_cid_{}", i);
            let key = store.shard_for_cid(&cid);
            if key.0 == 0 {
                store.put(cid, vec![1], 0).expect("put");
                shard0_count += 1;
            }
            i += 1;
        }
        let total_before = store.total_block_count();
        let _moved = store.rebalance(1000);
        let total_after = store.total_block_count();
        // Total blocks preserved
        assert_eq!(total_before, total_after);
    }

    // 34. evict_lru() removes correct count
    #[test]
    fn test_evict_lru_count() {
        let mut store = small_store(1);
        for i in 0..10usize {
            store.put(make_cid(i), vec![i as u8], 0).expect("put");
        }
        let evicted = store.evict_lru(ShardKey(0), 3);
        assert_eq!(evicted, 3);
        assert_eq!(store.total_block_count(), 7);
    }

    // 35. evict_lru() prefers lowest access_count
    #[test]
    fn test_evict_lru_lowest_access() {
        let mut store = small_store(1);
        store.put("hot".to_string(), vec![1], 0).expect("put");
        store.put("cold".to_string(), vec![2], 0).expect("put");
        // Access "hot" 10 times
        for _ in 0..10 {
            store.get("hot");
        }
        // Evict 1 — should remove "cold" (access_count == 0)
        store.evict_lru(ShardKey(0), 1);
        assert!(store.contains("hot"));
        assert!(!store.contains("cold"));
    }

    // 36. evict_lru() with count larger than shard size
    #[test]
    fn test_evict_lru_clamps_to_shard_size() {
        let mut store = small_store(1);
        for i in 0..5usize {
            store.put(make_cid(i), vec![1], 0).expect("put");
        }
        let evicted = store.evict_lru(ShardKey(0), 100);
        assert_eq!(evicted, 5);
        assert_eq!(store.total_block_count(), 0);
    }

    // 37. evict_lru() on out-of-range shard returns 0
    #[test]
    fn test_evict_lru_out_of_range() {
        let mut store = small_store(2);
        assert_eq!(store.evict_lru(ShardKey(99), 5), 0);
    }

    // 38. global_stats() reflects current state
    #[test]
    fn test_global_stats_basic() {
        let mut store = default_store();
        store.put("g1".to_string(), vec![1, 2], 0).expect("put");
        store.put("g2".to_string(), vec![3, 4, 5], 0).expect("put");
        let stats = store.global_stats();
        assert_eq!(stats.total_blocks, 2);
        assert_eq!(stats.total_bytes, 5);
        assert_eq!(stats.total_insertions, 2);
        assert_eq!(stats.num_shards, 16);
    }

    // 39. global_stats().needs_rebalance matches needs_rebalance()
    #[test]
    fn test_global_stats_needs_rebalance_flag() {
        let store = default_store();
        let stats = store.global_stats();
        assert_eq!(stats.needs_rebalance, store.needs_rebalance());
    }

    // 40. replace block updates total_bytes correctly
    #[test]
    fn test_replace_updates_bytes() {
        let mut store = default_store();
        store
            .put("replace_me".to_string(), vec![0u8; 10], 0)
            .expect("first");
        store
            .put("replace_me".to_string(), vec![0u8; 25], 1)
            .expect("replace");
        assert_eq!(store.total_bytes(), 25);
    }

    // 41. ShardMetrics.block_count tracks correctly
    #[test]
    fn test_shard_metrics_block_count() {
        let mut store = small_store(1);
        store.put("a".to_string(), vec![1], 0).expect("put");
        store.put("b".to_string(), vec![2], 0).expect("put");
        let m = store.shard_metrics(ShardKey(0)).expect("metrics");
        assert_eq!(m.block_count, 2);
        store.remove("a");
        let m2 = store.shard_metrics(ShardKey(0)).expect("metrics");
        assert_eq!(m2.block_count, 1);
    }

    // 42. ShardMetrics.total_bytes tracks correctly
    #[test]
    fn test_shard_metrics_total_bytes() {
        let mut store = small_store(1);
        store.put("x".to_string(), vec![0u8; 50], 0).expect("put");
        let m = store.shard_metrics(ShardKey(0)).expect("metrics");
        assert_eq!(m.total_bytes, 50);
    }

    // 43. ShardKey display/access
    #[test]
    fn test_shard_key_accessors() {
        let k = ShardKey(7);
        assert_eq!(k.index(), 7);
        assert_eq!(k.as_usize(), 7);
    }

    // 44. ShardingError::ShardFull carries correct shard_id
    #[test]
    fn test_sharding_error_shard_full_id() {
        let mut store = BlockStoreSharding::new(ShardingConfig {
            num_shards: 1,
            max_blocks_per_shard: 0,
            rebalance_threshold: 0.25,
        });
        let result = store.put("any".to_string(), vec![1], 0);
        assert_eq!(result, Err(ShardingError::ShardFull { shard_id: 0 }));
    }

    // 45. ShardingError Display
    #[test]
    fn test_sharding_error_display() {
        let e1 = ShardingError::ShardFull { shard_id: 3 };
        let e2 = ShardingError::CidNotFound("abc".to_string());
        let e3 = ShardingError::InvalidShardId(99);
        assert!(e1.to_string().contains("3"));
        assert!(e2.to_string().contains("abc"));
        assert!(e3.to_string().contains("99"));
    }

    // 46. BlockRecord fields are public and readable
    #[test]
    fn test_block_record_fields() {
        let r = BlockRecord {
            cid: "test_cid".to_string(),
            data: vec![1, 2, 3],
            shard: ShardKey(2),
            inserted_at: 12345,
            access_count: 7,
        };
        assert_eq!(r.cid, "test_cid");
        assert_eq!(r.shard.0, 2);
        assert_eq!(r.inserted_at, 12345);
        assert_eq!(r.access_count, 7);
    }

    // 47. BlockStoreGlobalStats is clonable and fields accessible
    #[test]
    fn test_global_stats_clone() {
        let s = BlockStoreGlobalStats {
            total_blocks: 10,
            total_bytes: 500,
            total_insertions: 12,
            total_lookups: 8,
            num_shards: 4,
            needs_rebalance: false,
        };
        let s2 = s.clone();
        assert_eq!(s2.total_blocks, 10);
    }

    // 48. Large number of blocks distributed across shards
    #[test]
    fn test_large_insertion_distribution() {
        let mut store = default_store();
        for i in 0..1000usize {
            store.put(make_cid(i), vec![1], 0).expect("put");
        }
        assert_eq!(store.total_block_count(), 1000);
        // Each shard should have some blocks (with 16 shards and 1000 inserts)
        let non_empty = store
            .all_shard_metrics()
            .iter()
            .filter(|m| m.block_count > 0)
            .count();
        assert!(
            non_empty > 1,
            "blocks should be distributed across multiple shards"
        );
    }

    // 49. Shard metrics total_bytes sums equal global total_bytes
    #[test]
    fn test_shard_metrics_bytes_sum_equals_global() {
        let mut store = default_store();
        for i in 0..50usize {
            store.put(make_cid(i), vec![0u8; i + 1], 0).expect("put");
        }
        let metrics_sum: u64 = store
            .all_shard_metrics()
            .iter()
            .map(|m| m.total_bytes)
            .sum();
        assert_eq!(metrics_sum, store.total_bytes());
    }

    // 50. Shard metrics block_count sums equal global block count
    #[test]
    fn test_shard_metrics_block_count_sum_equals_total() {
        let mut store = default_store();
        for i in 0..50usize {
            store.put(make_cid(i), vec![1], 0).expect("put");
        }
        let metrics_sum: usize = store
            .all_shard_metrics()
            .iter()
            .map(|m| m.block_count)
            .sum();
        assert_eq!(metrics_sum, store.total_block_count());
    }

    // 51. ShardMetrics implements Clone
    #[test]
    fn test_shard_metrics_clone() {
        let m = ShardMetrics {
            shard_id: 3,
            block_count: 10,
            total_bytes: 500,
            hit_count: 7,
            miss_count: 2,
        };
        let m2 = m.clone();
        assert_eq!(m2.shard_id, 3);
        assert_eq!(m2.hit_count, 7);
    }

    // 52. get_mut does not update hit/miss counts
    #[test]
    fn test_get_mut_no_metrics_update() {
        let mut store = small_store(1);
        store.put("z".to_string(), vec![1], 0).expect("put");
        store.get_mut("z");
        let m = store.shard_metrics(ShardKey(0)).expect("metrics");
        assert_eq!(m.hit_count, 0);
        assert_eq!(m.miss_count, 0);
    }

    // 53. rebalance() preserves total_bytes
    #[test]
    fn test_rebalance_preserves_bytes() {
        let mut store = BlockStoreSharding::new(ShardingConfig {
            num_shards: 2,
            max_blocks_per_shard: 10000,
            rebalance_threshold: 0.10,
        });
        let mut shard0_count = 0usize;
        let mut i = 0usize;
        while shard0_count < 10 {
            let cid = format!("rb_bytes_{}", i);
            let key = store.shard_for_cid(&cid);
            if key.0 == 0 {
                store.put(cid, vec![0u8; 8], 0).expect("put");
                shard0_count += 1;
            }
            i += 1;
        }
        let bytes_before = store.total_bytes();
        store.rebalance(1000);
        assert_eq!(store.total_bytes(), bytes_before);
    }
}
