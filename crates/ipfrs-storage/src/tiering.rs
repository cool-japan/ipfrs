//! Hot/cold storage tiering with access tracking.
//!
//! Manages automatic migration of blocks between fast (hot) and
//! slow (cold) storage tiers based on access patterns.
//!
//! # Access Tracking
//!
//! Uses a combination of access frequency and recency to determine
//! block temperature. Blocks with high access frequency stay hot,
//! while rarely accessed blocks become cold over time.
//!
//! # Example
//!
//! ```rust,ignore
//! use ipfrs_storage::tiering::{AccessTracker, TierConfig};
//!
//! let tracker = AccessTracker::new(TierConfig::default());
//! tracker.record_access(&cid);
//!
//! if tracker.is_hot(&cid) {
//!     // Block is frequently accessed
//! }
//! ```

use dashmap::DashMap;
use ipfrs_core::{Cid, Error, Result};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Storage tier classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Tier {
    /// Hot tier - frequently accessed, kept in fast storage
    Hot,
    /// Warm tier - occasionally accessed
    Warm,
    /// Cold tier - rarely accessed, can be moved to slow storage
    Cold,
    /// Archive tier - very rarely accessed, cheapest storage
    Archive,
}

impl Tier {
    /// Get the next colder tier
    pub fn colder(self) -> Option<Tier> {
        match self {
            Tier::Hot => Some(Tier::Warm),
            Tier::Warm => Some(Tier::Cold),
            Tier::Cold => Some(Tier::Archive),
            Tier::Archive => None,
        }
    }

    /// Get the next hotter tier
    pub fn hotter(self) -> Option<Tier> {
        match self {
            Tier::Archive => Some(Tier::Cold),
            Tier::Cold => Some(Tier::Warm),
            Tier::Warm => Some(Tier::Hot),
            Tier::Hot => None,
        }
    }
}

/// Configuration for tiering behavior
#[derive(Debug, Clone)]
pub struct TierConfig {
    /// Threshold for hot tier (accesses per hour)
    pub hot_threshold: f64,
    /// Threshold for warm tier (accesses per hour)
    pub warm_threshold: f64,
    /// Threshold for cold tier (accesses per hour)
    pub cold_threshold: f64,
    /// Time window for calculating access rate (in seconds)
    pub time_window_secs: u64,
    /// Decay factor for old accesses (0.0 - 1.0)
    pub decay_factor: f64,
    /// How often to run cleanup/decay (in seconds)
    pub cleanup_interval_secs: u64,
}

impl Default for TierConfig {
    fn default() -> Self {
        Self {
            hot_threshold: 10.0,        // 10+ accesses/hour = hot
            warm_threshold: 1.0,        // 1-10 accesses/hour = warm
            cold_threshold: 0.1,        // 0.1-1 accesses/hour = cold
            time_window_secs: 3600,     // 1 hour window
            decay_factor: 0.9,          // 10% decay per period
            cleanup_interval_secs: 300, // Cleanup every 5 minutes
        }
    }
}

/// Access statistics for a single block
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessStats {
    /// Total access count
    pub total_accesses: u64,
    /// Weighted access count (with time decay)
    pub weighted_accesses: f64,
    /// Last access timestamp (Unix timestamp)
    pub last_access: u64,
    /// First access timestamp (Unix timestamp)
    pub first_access: u64,
    /// Current tier
    pub tier: Tier,
}

impl AccessStats {
    fn new() -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Self {
            total_accesses: 1,
            weighted_accesses: 1.0,
            last_access: now,
            first_access: now,
            tier: Tier::Hot, // New blocks start hot
        }
    }

    fn record_access(&mut self) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        self.total_accesses += 1;
        self.weighted_accesses += 1.0;
        self.last_access = now;
    }

    /// Calculate access rate (accesses per hour)
    fn access_rate(&self, time_window_secs: u64) -> f64 {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let elapsed = now.saturating_sub(self.first_access).max(1);
        let window = elapsed.min(time_window_secs) as f64;

        // Accesses per hour
        self.weighted_accesses * 3600.0 / window
    }

    /// Apply time decay to weighted accesses
    fn apply_decay(&mut self, decay_factor: f64) {
        self.weighted_accesses *= decay_factor;
    }
}

/// Access tracker for monitoring block access patterns
pub struct AccessTracker {
    /// Access statistics per CID
    stats: DashMap<Vec<u8>, AccessStats>,
    /// Configuration
    config: TierConfig,
    /// Last cleanup time
    last_cleanup: RwLock<Instant>,
    /// Global statistics
    global_stats: GlobalAccessStats,
}

/// Global access statistics
#[derive(Default)]
struct GlobalAccessStats {
    total_accesses: AtomicU64,
    hot_blocks: AtomicU64,
    warm_blocks: AtomicU64,
    cold_blocks: AtomicU64,
    archive_blocks: AtomicU64,
}

impl AccessTracker {
    /// Create a new access tracker
    pub fn new(config: TierConfig) -> Self {
        Self {
            stats: DashMap::new(),
            config,
            last_cleanup: RwLock::new(Instant::now()),
            global_stats: GlobalAccessStats::default(),
        }
    }

    /// Record an access to a block
    pub fn record_access(&self, cid: &Cid) {
        let key = cid.to_bytes();
        self.global_stats
            .total_accesses
            .fetch_add(1, Ordering::Relaxed);

        self.stats
            .entry(key)
            .and_modify(|stats| {
                let old_tier = stats.tier;
                stats.record_access();
                let new_tier = self.classify_tier(stats);
                if old_tier != new_tier {
                    self.update_tier_counts(old_tier, new_tier);
                    stats.tier = new_tier;
                }
            })
            .or_insert_with(|| {
                self.global_stats.hot_blocks.fetch_add(1, Ordering::Relaxed);
                AccessStats::new()
            });

        // Periodic cleanup
        self.maybe_cleanup();
    }

    /// Get the current tier for a block
    pub fn get_tier(&self, cid: &Cid) -> Option<Tier> {
        self.stats.get(&cid.to_bytes()).map(|s| s.tier)
    }

    /// Check if a block is in the hot tier
    pub fn is_hot(&self, cid: &Cid) -> bool {
        self.get_tier(cid) == Some(Tier::Hot)
    }

    /// Check if a block is cold (cold or archive tier)
    pub fn is_cold(&self, cid: &Cid) -> bool {
        matches!(self.get_tier(cid), Some(Tier::Cold) | Some(Tier::Archive))
    }

    /// Get access statistics for a block
    pub fn get_stats(&self, cid: &Cid) -> Option<AccessStats> {
        self.stats.get(&cid.to_bytes()).map(|s| s.clone())
    }

    /// List all blocks in a specific tier
    pub fn list_by_tier(&self, tier: Tier) -> Result<Vec<Cid>> {
        let mut result = Vec::new();
        for entry in self.stats.iter() {
            if entry.value().tier == tier {
                let cid = Cid::try_from(entry.key().clone())
                    .map_err(|e| Error::Cid(format!("Invalid CID: {e}")))?;
                result.push(cid);
            }
        }
        Ok(result)
    }

    /// Get candidates for migration to a colder tier
    pub fn get_cold_candidates(&self, max_count: usize) -> Result<Vec<(Cid, Tier)>> {
        let mut candidates: Vec<_> = self
            .stats
            .iter()
            .filter_map(|entry| {
                let stats = entry.value();
                if let Some(colder_tier) = stats.tier.colder() {
                    let rate = stats.access_rate(self.config.time_window_secs);
                    let threshold = self.tier_threshold(colder_tier);
                    if rate < threshold {
                        let cid = Cid::try_from(entry.key().clone()).ok()?;
                        return Some((cid, colder_tier, rate));
                    }
                }
                None
            })
            .collect();

        // Sort by access rate (lowest first)
        candidates.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

        Ok(candidates
            .into_iter()
            .take(max_count)
            .map(|(cid, tier, _)| (cid, tier))
            .collect())
    }

    /// Manually set the tier for a block
    pub fn set_tier(&self, cid: &Cid, tier: Tier) {
        let key = cid.to_bytes();
        if let Some(mut entry) = self.stats.get_mut(&key) {
            let old_tier = entry.tier;
            if old_tier != tier {
                self.update_tier_counts(old_tier, tier);
                entry.tier = tier;
            }
        }
    }

    /// Get global statistics
    pub fn global_stats(&self) -> TierStatsSnapshot {
        TierStatsSnapshot {
            total_accesses: self.global_stats.total_accesses.load(Ordering::Relaxed),
            tracked_blocks: self.stats.len() as u64,
            hot_blocks: self.global_stats.hot_blocks.load(Ordering::Relaxed),
            warm_blocks: self.global_stats.warm_blocks.load(Ordering::Relaxed),
            cold_blocks: self.global_stats.cold_blocks.load(Ordering::Relaxed),
            archive_blocks: self.global_stats.archive_blocks.load(Ordering::Relaxed),
        }
    }

    /// Force a cleanup/decay pass
    pub fn run_cleanup(&self) {
        for mut entry in self.stats.iter_mut() {
            let stats = entry.value_mut();
            let old_tier = stats.tier;

            // Apply decay
            stats.apply_decay(self.config.decay_factor);

            // Reclassify tier
            let new_tier = self.classify_tier(stats);
            if old_tier != new_tier {
                self.update_tier_counts(old_tier, new_tier);
                stats.tier = new_tier;
            }
        }

        *self.last_cleanup.write() = Instant::now();
    }

    /// Classify a block into a tier based on its access rate
    fn classify_tier(&self, stats: &AccessStats) -> Tier {
        let rate = stats.access_rate(self.config.time_window_secs);

        if rate >= self.config.hot_threshold {
            Tier::Hot
        } else if rate >= self.config.warm_threshold {
            Tier::Warm
        } else if rate >= self.config.cold_threshold {
            Tier::Cold
        } else {
            Tier::Archive
        }
    }

    /// Get the threshold for a tier
    fn tier_threshold(&self, tier: Tier) -> f64 {
        match tier {
            Tier::Hot => self.config.hot_threshold,
            Tier::Warm => self.config.warm_threshold,
            Tier::Cold => self.config.cold_threshold,
            Tier::Archive => 0.0,
        }
    }

    /// Update tier counts when a block changes tiers
    fn update_tier_counts(&self, old_tier: Tier, new_tier: Tier) {
        match old_tier {
            Tier::Hot => self.global_stats.hot_blocks.fetch_sub(1, Ordering::Relaxed),
            Tier::Warm => self
                .global_stats
                .warm_blocks
                .fetch_sub(1, Ordering::Relaxed),
            Tier::Cold => self
                .global_stats
                .cold_blocks
                .fetch_sub(1, Ordering::Relaxed),
            Tier::Archive => self
                .global_stats
                .archive_blocks
                .fetch_sub(1, Ordering::Relaxed),
        };
        match new_tier {
            Tier::Hot => self.global_stats.hot_blocks.fetch_add(1, Ordering::Relaxed),
            Tier::Warm => self
                .global_stats
                .warm_blocks
                .fetch_add(1, Ordering::Relaxed),
            Tier::Cold => self
                .global_stats
                .cold_blocks
                .fetch_add(1, Ordering::Relaxed),
            Tier::Archive => self
                .global_stats
                .archive_blocks
                .fetch_add(1, Ordering::Relaxed),
        };
    }

    /// Check if cleanup is needed and run it
    fn maybe_cleanup(&self) {
        let should_cleanup = {
            let last = self.last_cleanup.read();
            last.elapsed() > Duration::from_secs(self.config.cleanup_interval_secs)
        };

        if should_cleanup {
            self.run_cleanup();
        }
    }

    /// Remove tracking for a block
    pub fn remove(&self, cid: &Cid) {
        if let Some((_, stats)) = self.stats.remove(&cid.to_bytes()) {
            match stats.tier {
                Tier::Hot => self.global_stats.hot_blocks.fetch_sub(1, Ordering::Relaxed),
                Tier::Warm => self
                    .global_stats
                    .warm_blocks
                    .fetch_sub(1, Ordering::Relaxed),
                Tier::Cold => self
                    .global_stats
                    .cold_blocks
                    .fetch_sub(1, Ordering::Relaxed),
                Tier::Archive => self
                    .global_stats
                    .archive_blocks
                    .fetch_sub(1, Ordering::Relaxed),
            };
        }
    }

    /// Clear all tracking data
    pub fn clear(&self) {
        self.stats.clear();
        self.global_stats.total_accesses.store(0, Ordering::Relaxed);
        self.global_stats.hot_blocks.store(0, Ordering::Relaxed);
        self.global_stats.warm_blocks.store(0, Ordering::Relaxed);
        self.global_stats.cold_blocks.store(0, Ordering::Relaxed);
        self.global_stats.archive_blocks.store(0, Ordering::Relaxed);
    }
}

/// Snapshot of tier statistics
#[derive(Debug, Clone)]
pub struct TierStatsSnapshot {
    /// Total accesses recorded
    pub total_accesses: u64,
    /// Number of blocks being tracked
    pub tracked_blocks: u64,
    /// Number of blocks in hot tier
    pub hot_blocks: u64,
    /// Number of blocks in warm tier
    pub warm_blocks: u64,
    /// Number of blocks in cold tier
    pub cold_blocks: u64,
    /// Number of blocks in archive tier
    pub archive_blocks: u64,
}

/// Tiered block store that tracks access patterns and supports migration
use crate::traits::BlockStore;
use async_trait::async_trait;
use ipfrs_core::Block;

pub struct TieredStore<H: BlockStore, C: BlockStore> {
    /// Hot storage (fast, expensive)
    hot_store: H,
    /// Cold storage (slow, cheap)
    cold_store: C,
    /// Access tracker
    tracker: AccessTracker,
    /// Configuration
    config: TierConfig,
}

impl<H: BlockStore, C: BlockStore> TieredStore<H, C> {
    /// Create a new tiered store
    pub fn new(hot_store: H, cold_store: C, config: TierConfig) -> Self {
        Self {
            hot_store,
            cold_store,
            tracker: AccessTracker::new(config.clone()),
            config,
        }
    }

    /// Get the access tracker
    pub fn tracker(&self) -> &AccessTracker {
        &self.tracker
    }

    /// Get the tier configuration
    pub fn config(&self) -> &TierConfig {
        &self.config
    }

    /// Migrate cold blocks from hot to cold storage
    pub async fn migrate_cold_blocks(&self, max_count: usize) -> Result<usize> {
        let candidates = self.tracker.get_cold_candidates(max_count)?;
        let mut migrated = 0;

        for (cid, _new_tier) in candidates {
            // Get from hot storage
            if let Some(block) = self.hot_store.get(&cid).await? {
                // Store in cold storage
                self.cold_store.put(&block).await?;
                // Remove from hot storage
                self.hot_store.delete(&cid).await?;
                migrated += 1;
            }
        }

        Ok(migrated)
    }

    /// Promote a block from cold to hot storage
    pub async fn promote_block(&self, cid: &Cid) -> Result<bool> {
        if let Some(block) = self.cold_store.get(cid).await? {
            self.hot_store.put(&block).await?;
            self.cold_store.delete(cid).await?;
            self.tracker.set_tier(cid, Tier::Hot);
            Ok(true)
        } else {
            Ok(false)
        }
    }
}

#[async_trait]
impl<H: BlockStore, C: BlockStore> BlockStore for TieredStore<H, C> {
    async fn put(&self, block: &Block) -> Result<()> {
        // New blocks go to hot storage
        self.tracker.record_access(block.cid());
        self.hot_store.put(block).await
    }

    async fn get(&self, cid: &Cid) -> Result<Option<Block>> {
        self.tracker.record_access(cid);

        // Try hot storage first
        if let Some(block) = self.hot_store.get(cid).await? {
            return Ok(Some(block));
        }

        // Fall back to cold storage
        if let Some(block) = self.cold_store.get(cid).await? {
            // Optionally promote to hot storage on access
            if self.tracker.is_hot(cid) {
                // Block is now hot, migrate it
                self.hot_store.put(&block).await?;
                self.cold_store.delete(cid).await?;
            }
            return Ok(Some(block));
        }

        Ok(None)
    }

    async fn has(&self, cid: &Cid) -> Result<bool> {
        if self.hot_store.has(cid).await? {
            return Ok(true);
        }
        self.cold_store.has(cid).await
    }

    async fn delete(&self, cid: &Cid) -> Result<()> {
        self.tracker.remove(cid);
        // Delete from both stores
        let _ = self.hot_store.delete(cid).await;
        let _ = self.cold_store.delete(cid).await;
        Ok(())
    }

    fn list_cids(&self) -> Result<Vec<Cid>> {
        // Combine CIDs from both stores
        let mut cids = self.hot_store.list_cids()?;
        let cold_cids = self.cold_store.list_cids()?;
        cids.extend(cold_cids);
        // Remove duplicates
        cids.sort_by_key(|a| a.to_bytes());
        cids.dedup_by(|a, b| a.to_bytes() == b.to_bytes());
        Ok(cids)
    }

    fn len(&self) -> usize {
        self.hot_store.len() + self.cold_store.len()
    }

    fn is_empty(&self) -> bool {
        self.hot_store.is_empty() && self.cold_store.is_empty()
    }

    async fn flush(&self) -> Result<()> {
        self.hot_store.flush().await?;
        self.cold_store.flush().await
    }

    async fn close(&self) -> Result<()> {
        self.hot_store.close().await?;
        self.cold_store.close().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;
    use ipfrs_core::Block;

    fn make_test_cid(data: &[u8]) -> Cid {
        let block = Block::new(Bytes::copy_from_slice(data)).unwrap();
        *block.cid()
    }

    #[test]
    fn test_tier_classification() {
        let config = TierConfig::default();
        let tracker = AccessTracker::new(config);
        let cid = make_test_cid(b"test");

        // First access - should be hot
        tracker.record_access(&cid);
        assert!(tracker.is_hot(&cid));
    }

    #[test]
    fn test_access_stats() {
        let config = TierConfig::default();
        let tracker = AccessTracker::new(config);
        let cid = make_test_cid(b"test");

        for _ in 0..10 {
            tracker.record_access(&cid);
        }

        let stats = tracker.get_stats(&cid).unwrap();
        assert_eq!(stats.total_accesses, 10);
    }

    #[test]
    fn test_tier_stats() {
        let config = TierConfig::default();
        let tracker = AccessTracker::new(config);

        for i in 0..5 {
            let cid = make_test_cid(&[i]);
            tracker.record_access(&cid);
        }

        let stats = tracker.global_stats();
        assert_eq!(stats.tracked_blocks, 5);
        assert_eq!(stats.hot_blocks, 5);
    }

    #[test]
    fn test_tier_transitions() {
        assert_eq!(Tier::Hot.colder(), Some(Tier::Warm));
        assert_eq!(Tier::Warm.colder(), Some(Tier::Cold));
        assert_eq!(Tier::Cold.colder(), Some(Tier::Archive));
        assert_eq!(Tier::Archive.colder(), None);

        assert_eq!(Tier::Archive.hotter(), Some(Tier::Cold));
        assert_eq!(Tier::Hot.hotter(), None);
    }
}
