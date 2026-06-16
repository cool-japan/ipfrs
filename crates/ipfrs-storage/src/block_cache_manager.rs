//! Multi-tier, policy-driven block cache manager for IPFS-style content-addressed storage.
//!
//! `BlockCacheManager` provides a three-tier caching system (Hot, Warm, Cold) with
//! configurable eviction policies (LRU, LFU, TwoQ, ARC), promotion/demotion logic
//! based on access-count thresholds, and pin-protection for blocks that must survive eviction.

use std::collections::HashMap;

// ────────────────────────────────────────────────────────────────────────────
// CacheTier
// ────────────────────────────────────────────────────────────────────────────

/// Which tier a cached block lives in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CacheTier {
    /// Frequently-accessed blocks; highest retention priority.
    Hot,
    /// Moderately-accessed blocks.
    Warm,
    /// Least-recently-accessed or newly-inserted blocks.
    Cold,
}

impl CacheTier {
    /// Numeric priority: higher means more valuable to keep.
    #[must_use]
    pub fn priority(&self) -> u8 {
        match self {
            CacheTier::Hot => 3,
            CacheTier::Warm => 2,
            CacheTier::Cold => 1,
        }
    }

    /// The tier one step lower (Cold stays Cold).
    fn demoted(self) -> Self {
        match self {
            CacheTier::Hot => CacheTier::Warm,
            CacheTier::Warm => CacheTier::Cold,
            CacheTier::Cold => CacheTier::Cold,
        }
    }

    /// The tier one step higher (Hot stays Hot).
    #[allow(dead_code)]
    fn promoted(self) -> Self {
        match self {
            CacheTier::Cold => CacheTier::Warm,
            CacheTier::Warm => CacheTier::Hot,
            CacheTier::Hot => CacheTier::Hot,
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// EvictionPolicy
// ────────────────────────────────────────────────────────────────────────────

/// Strategy used to select victims during eviction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvictionPolicy {
    /// Least-Recently-Used: evict the block with the smallest `last_accessed`.
    LRU,
    /// Least-Frequently-Used: evict the block with the smallest `access_count`
    /// (tie-broken by `last_accessed`).
    LFU,
    /// Two-Queue approximation (currently implemented as LRU).
    TwoQ,
    /// Adaptive Replacement Cache approximation (currently implemented as LRU).
    ARC,
}

// ────────────────────────────────────────────────────────────────────────────
// BcmCacheConfig
// ────────────────────────────────────────────────────────────────────────────

/// Configuration for [`BlockCacheManager`].
///
/// Note: named `BcmCacheConfig` to avoid collision with the `CacheConfig` in
/// `ipfrs-storage::cache`.
#[derive(Debug, Clone)]
pub struct BcmCacheConfig {
    /// Maximum total bytes that may reside in the Hot tier.
    pub max_hot_bytes: u64,
    /// Maximum total bytes that may reside in the Warm tier.
    pub max_warm_bytes: u64,
    /// Maximum total bytes that may reside in the Cold tier.
    pub max_cold_bytes: u64,
    /// Minimum `access_count` required to promote a block to Hot.
    pub hot_threshold: u64,
    /// Minimum `access_count` required to promote a block to Warm (must be < `hot_threshold`).
    pub warm_threshold: u64,
    /// Eviction algorithm to use within each tier.
    pub eviction_policy: EvictionPolicy,
}

impl Default for BcmCacheConfig {
    fn default() -> Self {
        Self {
            max_hot_bytes: 64 * 1024 * 1024,   // 64 MiB
            max_warm_bytes: 128 * 1024 * 1024, // 128 MiB
            max_cold_bytes: 256 * 1024 * 1024, // 256 MiB
            hot_threshold: 10,
            warm_threshold: 3,
            eviction_policy: EvictionPolicy::LRU,
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// BcmCachedBlock
// ────────────────────────────────────────────────────────────────────────────

/// A single entry in the block cache.
///
/// Named `BcmCachedBlock` to avoid collision with any existing `CachedBlock`.
#[derive(Debug, Clone)]
pub struct BcmCachedBlock {
    /// Content identifier (CID string) for this block.
    pub cid: String,
    /// Raw block data.
    pub data: Vec<u8>,
    /// Which tier this block currently lives in.
    pub tier: CacheTier,
    /// Unix-timestamp (or logical clock) when this block was first inserted.
    pub inserted_at: u64,
    /// Unix-timestamp (or logical clock) of the most recent access.
    pub last_accessed: u64,
    /// Number of times this block has been read since insertion.
    pub access_count: u64,
    /// If `true`, eviction is forbidden for this block.
    pub pinned: bool,
}

impl BcmCachedBlock {
    fn new(cid: String, data: Vec<u8>, now: u64) -> Self {
        Self {
            cid,
            data,
            tier: CacheTier::Cold,
            inserted_at: now,
            last_accessed: now,
            access_count: 0,
            pinned: false,
        }
    }

    /// Byte size of the block data.
    #[inline]
    pub fn byte_size(&self) -> u64 {
        self.data.len() as u64
    }
}

// ────────────────────────────────────────────────────────────────────────────
// BcmCacheStats
// ────────────────────────────────────────────────────────────────────────────

/// Snapshot of [`BlockCacheManager`] operational statistics.
///
/// Named `BcmCacheStats` to avoid collision with the existing `CacheStats`.
#[derive(Debug, Clone, Default)]
pub struct BcmCacheStats {
    pub hot_count: usize,
    pub warm_count: usize,
    pub cold_count: usize,
    pub hot_bytes: u64,
    pub warm_bytes: u64,
    pub cold_bytes: u64,
    pub pinned_count: usize,
    pub evictions: u64,
    pub promotions: u64,
    pub demotions: u64,
}

// ────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ────────────────────────────────────────────────────────────────────────────

/// Choose a victim CID from `tier_map` according to `policy`.
/// Returns `None` if the tier is empty or every entry is pinned.
fn choose_victim(
    tier_map: &HashMap<String, BcmCachedBlock>,
    policy: EvictionPolicy,
) -> Option<String> {
    let candidates: Vec<&BcmCachedBlock> = tier_map.values().filter(|b| !b.pinned).collect();
    if candidates.is_empty() {
        return None;
    }

    let victim = match policy {
        EvictionPolicy::LFU => candidates.iter().min_by(|a, b| {
            a.access_count
                .cmp(&b.access_count)
                .then_with(|| a.last_accessed.cmp(&b.last_accessed))
        }),
        // LRU / TwoQ / ARC all use last_accessed
        EvictionPolicy::LRU | EvictionPolicy::TwoQ | EvictionPolicy::ARC => {
            candidates.iter().min_by_key(|b| b.last_accessed)
        }
    };

    victim.map(|b| b.cid.clone())
}

/// Sum the byte sizes of all entries in a tier map.
fn tier_bytes(tier_map: &HashMap<String, BcmCachedBlock>) -> u64 {
    tier_map.values().map(|b| b.byte_size()).sum()
}

// ────────────────────────────────────────────────────────────────────────────
// BlockCacheManager
// ────────────────────────────────────────────────────────────────────────────

/// A multi-tier, policy-driven block cache.
///
/// Blocks are stored across three tiers (`Cold` → `Warm` → `Hot`) and migrate
/// upward based on access frequency thresholds and downward via explicit
/// `demote` calls or administrator policy.
#[derive(Debug)]
pub struct BlockCacheManager {
    /// Cache configuration.
    pub config: BcmCacheConfig,

    // Tier storage — each keyed by CID string.
    pub hot: HashMap<String, BcmCachedBlock>,
    pub warm: HashMap<String, BcmCachedBlock>,
    pub cold: HashMap<String, BcmCachedBlock>,

    // Counters
    evictions: u64,
    promotions: u64,
    demotions: u64,
}

impl BlockCacheManager {
    // ── Construction ─────────────────────────────────────────────────────────

    /// Create a new [`BlockCacheManager`] with the given configuration.
    #[must_use]
    pub fn new(config: BcmCacheConfig) -> Self {
        Self {
            config,
            hot: HashMap::new(),
            warm: HashMap::new(),
            cold: HashMap::new(),
            evictions: 0,
            promotions: 0,
            demotions: 0,
        }
    }

    // ── Insertion ────────────────────────────────────────────────────────────

    /// Insert a block into the Cold tier.
    ///
    /// Returns `false` if the CID is already present in any tier.
    /// If the Cold tier is full, attempts eviction before inserting.
    pub fn insert(&mut self, cid: String, data: Vec<u8>, now: u64) -> bool {
        if self.contains(&cid) {
            return false;
        }

        let needed = data.len() as u64;

        // Make room in Cold tier if necessary.
        let cold_used = tier_bytes(&self.cold);
        if cold_used + needed > self.config.max_cold_bytes {
            let to_free = (cold_used + needed).saturating_sub(self.config.max_cold_bytes);
            self.evict_to_fit(CacheTier::Cold, to_free, now);
        }

        let block = BcmCachedBlock::new(cid.clone(), data, now);
        self.cold.insert(cid, block);
        true
    }

    // ── Retrieval ────────────────────────────────────────────────────────────

    /// Return a reference to the block data, or `None` if not present.
    ///
    /// Updates `access_count` and `last_accessed`, then checks whether the
    /// block should be promoted to a higher tier.
    pub fn get(&mut self, cid: &str, now: u64) -> Option<&[u8]> {
        // Update metadata in whichever tier holds the block.
        let found_tier = if let Some(block) = self.hot.get_mut(cid) {
            block.access_count = block.access_count.saturating_add(1);
            block.last_accessed = now;
            Some(CacheTier::Hot)
        } else if let Some(block) = self.warm.get_mut(cid) {
            block.access_count = block.access_count.saturating_add(1);
            block.last_accessed = now;
            Some(CacheTier::Warm)
        } else if let Some(block) = self.cold.get_mut(cid) {
            block.access_count = block.access_count.saturating_add(1);
            block.last_accessed = now;
            Some(CacheTier::Cold)
        } else {
            None
        };

        found_tier?;

        // Attempt promotion (takes a clone of CID so we don't borrow twice).
        let cid_owned = cid.to_owned();
        self.promote(&cid_owned, now);

        // Return reference from whichever tier now holds it.
        if let Some(block) = self.hot.get(cid) {
            return Some(&block.data);
        }
        if let Some(block) = self.warm.get(cid) {
            return Some(&block.data);
        }
        if let Some(block) = self.cold.get(cid) {
            return Some(&block.data);
        }
        None
    }

    // ── Pinning ───────────────────────────────────────────────────────────────

    /// Mark a block as pinned so it cannot be evicted.  Returns `false` if not found.
    pub fn pin(&mut self, cid: &str) -> bool {
        if let Some(b) = self.hot.get_mut(cid) {
            b.pinned = true;
            return true;
        }
        if let Some(b) = self.warm.get_mut(cid) {
            b.pinned = true;
            return true;
        }
        if let Some(b) = self.cold.get_mut(cid) {
            b.pinned = true;
            return true;
        }
        false
    }

    /// Remove the pin from a block, allowing it to be evicted again.  Returns `false` if not found.
    pub fn unpin(&mut self, cid: &str) -> bool {
        if let Some(b) = self.hot.get_mut(cid) {
            b.pinned = false;
            return true;
        }
        if let Some(b) = self.warm.get_mut(cid) {
            b.pinned = false;
            return true;
        }
        if let Some(b) = self.cold.get_mut(cid) {
            b.pinned = false;
            return true;
        }
        false
    }

    // ── Eviction ─────────────────────────────────────────────────────────────

    /// Evict unpinned blocks from `tier` until at least `needed_bytes` have been freed.
    ///
    /// Returns the total number of bytes actually freed.  Pinned blocks are
    /// never evicted; if only pinned blocks remain, eviction stops early.
    pub fn evict_to_fit(&mut self, tier: CacheTier, needed_bytes: u64, now: u64) -> u64 {
        // `now` is kept for potential future use (e.g., time-aware eviction).
        let _ = now;
        let mut freed: u64 = 0;
        let policy = self.config.eviction_policy;

        let tier_map = match tier {
            CacheTier::Hot => &mut self.hot,
            CacheTier::Warm => &mut self.warm,
            CacheTier::Cold => &mut self.cold,
        };

        while freed < needed_bytes {
            match choose_victim(tier_map, policy) {
                None => break,
                Some(victim_cid) => {
                    if let Some(block) = tier_map.remove(&victim_cid) {
                        freed += block.byte_size();
                        self.evictions += 1;
                    }
                }
            }
        }
        freed
    }

    // ── Promotion / Demotion ──────────────────────────────────────────────────

    /// Check whether the block identified by `cid` should be promoted to a
    /// higher tier based on its current `access_count`, and move it if so.
    ///
    /// Promotion chain:
    /// * Cold  → Warm  if `access_count >= warm_threshold`
    /// * Warm  → Hot   if `access_count >= hot_threshold`
    pub fn promote(&mut self, cid: &str, now: u64) {
        // Determine current tier and access_count without a mutable borrow.
        let (current_tier, access_count) = {
            if let Some(b) = self.hot.get(cid) {
                (CacheTier::Hot, b.access_count)
            } else if let Some(b) = self.warm.get(cid) {
                (CacheTier::Warm, b.access_count)
            } else if let Some(b) = self.cold.get(cid) {
                (CacheTier::Cold, b.access_count)
            } else {
                return; // block not found
            }
        };

        let target_tier = {
            if access_count >= self.config.hot_threshold {
                CacheTier::Hot
            } else if access_count >= self.config.warm_threshold {
                CacheTier::Warm
            } else {
                // Below warm threshold — stays where it is.
                return;
            }
        };

        if target_tier == current_tier || (target_tier as u8) <= (current_tier as u8) {
            // Already at target or higher.
            if current_tier == target_tier {
                return;
            }
            // If target is *lower* than current, don't demote during promote.
            if target_tier.priority() <= current_tier.priority() {
                return;
            }
        }

        // Move from source to destination tier.
        self.move_block(cid, current_tier, target_tier, now, true);
    }

    /// Move a block one tier down.  Has no effect if the block is in Cold tier
    /// or not found.
    pub fn demote(&mut self, cid: &str, now: u64) {
        let current_tier = {
            if self.hot.contains_key(cid) {
                CacheTier::Hot
            } else if self.warm.contains_key(cid) {
                CacheTier::Warm
            } else {
                // Cold or missing — nothing to demote.
                return;
            }
        };

        let target_tier = current_tier.demoted();
        if target_tier == current_tier {
            return;
        }

        self.move_block(cid, current_tier, target_tier, now, false);
    }

    // ── Internal move helper ──────────────────────────────────────────────────

    /// Move a block between tiers, evicting in the destination tier if needed.
    fn move_block(
        &mut self,
        cid: &str,
        from: CacheTier,
        to: CacheTier,
        now: u64,
        is_promotion: bool,
    ) {
        // Extract block from source tier.
        let mut block = match from {
            CacheTier::Hot => self.hot.remove(cid),
            CacheTier::Warm => self.warm.remove(cid),
            CacheTier::Cold => self.cold.remove(cid),
        };

        let block = match block.as_mut() {
            Some(b) => b,
            None => return,
        };

        block.tier = to;
        block.last_accessed = now;

        // Make room in destination tier if needed.
        let block_size = block.byte_size();
        let max_dest = match to {
            CacheTier::Hot => self.config.max_hot_bytes,
            CacheTier::Warm => self.config.max_warm_bytes,
            CacheTier::Cold => self.config.max_cold_bytes,
        };
        let dest_used = match to {
            CacheTier::Hot => tier_bytes(&self.hot),
            CacheTier::Warm => tier_bytes(&self.warm),
            CacheTier::Cold => tier_bytes(&self.cold),
        };
        if dest_used + block_size > max_dest {
            let to_free = (dest_used + block_size).saturating_sub(max_dest);
            self.evict_to_fit(to, to_free, now);
        }

        let block = block.clone();
        match to {
            CacheTier::Hot => {
                self.hot.insert(cid.to_owned(), block);
            }
            CacheTier::Warm => {
                self.warm.insert(cid.to_owned(), block);
            }
            CacheTier::Cold => {
                self.cold.insert(cid.to_owned(), block);
            }
        }

        if is_promotion {
            self.promotions += 1;
        } else {
            self.demotions += 1;
        }
    }

    // ── Byte-size queries ─────────────────────────────────────────────────────

    /// Total bytes used by the Hot tier.
    #[must_use]
    pub fn total_hot_bytes(&self) -> u64 {
        tier_bytes(&self.hot)
    }

    /// Total bytes used by the Warm tier.
    #[must_use]
    pub fn total_warm_bytes(&self) -> u64 {
        tier_bytes(&self.warm)
    }

    /// Total bytes used by the Cold tier.
    #[must_use]
    pub fn total_cold_bytes(&self) -> u64 {
        tier_bytes(&self.cold)
    }

    /// Total bytes across all tiers.
    #[must_use]
    pub fn total_bytes(&self) -> u64 {
        self.total_hot_bytes() + self.total_warm_bytes() + self.total_cold_bytes()
    }

    // ── Counting ─────────────────────────────────────────────────────────────

    /// Total number of blocks across all tiers.
    #[must_use]
    pub fn len(&self) -> usize {
        self.hot.len() + self.warm.len() + self.cold.len()
    }

    /// `true` if no blocks are currently cached.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    // ── Removal ──────────────────────────────────────────────────────────────

    /// Remove a block from the cache, regardless of tier.
    ///
    /// Returns `false` if the block is pinned or not found.
    pub fn remove(&mut self, cid: &str) -> bool {
        // Check pinned state before removal.
        let is_pinned = self.hot.get(cid).is_some_and(|b| b.pinned)
            || self.warm.get(cid).is_some_and(|b| b.pinned)
            || self.cold.get(cid).is_some_and(|b| b.pinned);

        if is_pinned {
            return false;
        }

        self.hot.remove(cid).is_some()
            || self.warm.remove(cid).is_some()
            || self.cold.remove(cid).is_some()
    }

    // ── Lookup ───────────────────────────────────────────────────────────────

    /// Return `true` if the CID is cached in any tier.
    #[must_use]
    pub fn contains(&self, cid: &str) -> bool {
        self.hot.contains_key(cid) || self.warm.contains_key(cid) || self.cold.contains_key(cid)
    }

    /// Return which tier the block lives in, or `None` if not cached.
    #[must_use]
    pub fn tier_of(&self, cid: &str) -> Option<CacheTier> {
        if self.hot.contains_key(cid) {
            Some(CacheTier::Hot)
        } else if self.warm.contains_key(cid) {
            Some(CacheTier::Warm)
        } else if self.cold.contains_key(cid) {
            Some(CacheTier::Cold)
        } else {
            None
        }
    }

    // ── Statistics ────────────────────────────────────────────────────────────

    /// Snapshot of current cache statistics.
    #[must_use]
    pub fn stats(&self) -> BcmCacheStats {
        let pinned_count = self.hot.values().filter(|b| b.pinned).count()
            + self.warm.values().filter(|b| b.pinned).count()
            + self.cold.values().filter(|b| b.pinned).count();

        BcmCacheStats {
            hot_count: self.hot.len(),
            warm_count: self.warm.len(),
            cold_count: self.cold.len(),
            hot_bytes: self.total_hot_bytes(),
            warm_bytes: self.total_warm_bytes(),
            cold_bytes: self.total_cold_bytes(),
            pinned_count,
            evictions: self.evictions,
            promotions: self.promotions,
            demotions: self.demotions,
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// CacheTier as u8 helper (for priority comparison)
// ────────────────────────────────────────────────────────────────────────────

impl From<CacheTier> for u8 {
    fn from(tier: CacheTier) -> u8 {
        tier.priority()
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::{
        choose_victim, tier_bytes, BcmCacheConfig, BcmCacheStats, BcmCachedBlock,
        BlockCacheManager, CacheTier, EvictionPolicy,
    };
    use std::collections::HashMap;

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn default_config() -> BcmCacheConfig {
        BcmCacheConfig {
            max_hot_bytes: 1024,
            max_warm_bytes: 2048,
            max_cold_bytes: 4096,
            hot_threshold: 5,
            warm_threshold: 2,
            eviction_policy: EvictionPolicy::LRU,
        }
    }

    fn make_manager() -> BlockCacheManager {
        BlockCacheManager::new(default_config())
    }

    fn make_block(cid: &str, size: usize, now: u64) -> BcmCachedBlock {
        BcmCachedBlock::new(cid.to_owned(), vec![0u8; size], now)
    }

    // ── CacheTier ────────────────────────────────────────────────────────────

    #[test]
    fn test_tier_priority() {
        assert_eq!(CacheTier::Hot.priority(), 3);
        assert_eq!(CacheTier::Warm.priority(), 2);
        assert_eq!(CacheTier::Cold.priority(), 1);
    }

    #[test]
    fn test_tier_promoted() {
        assert_eq!(CacheTier::Cold.promoted(), CacheTier::Warm);
        assert_eq!(CacheTier::Warm.promoted(), CacheTier::Hot);
        assert_eq!(CacheTier::Hot.promoted(), CacheTier::Hot);
    }

    #[test]
    fn test_tier_demoted() {
        assert_eq!(CacheTier::Hot.demoted(), CacheTier::Warm);
        assert_eq!(CacheTier::Warm.demoted(), CacheTier::Cold);
        assert_eq!(CacheTier::Cold.demoted(), CacheTier::Cold);
    }

    #[test]
    fn test_tier_from_u8_priority() {
        let hot: u8 = CacheTier::Hot.into();
        let warm: u8 = CacheTier::Warm.into();
        let cold: u8 = CacheTier::Cold.into();
        assert!(hot > warm && warm > cold);
    }

    // ── BcmCachedBlock ────────────────────────────────────────────────────────

    #[test]
    fn test_cached_block_initial_state() {
        let block = make_block("cid1", 128, 1000);
        assert_eq!(block.cid, "cid1");
        assert_eq!(block.data.len(), 128);
        assert_eq!(block.tier, CacheTier::Cold);
        assert_eq!(block.inserted_at, 1000);
        assert_eq!(block.last_accessed, 1000);
        assert_eq!(block.access_count, 0);
        assert!(!block.pinned);
    }

    #[test]
    fn test_cached_block_byte_size() {
        let block = make_block("cid2", 64, 0);
        assert_eq!(block.byte_size(), 64);
    }

    // ── tier_bytes helper ────────────────────────────────────────────────────

    #[test]
    fn test_tier_bytes_empty() {
        let map: HashMap<String, BcmCachedBlock> = HashMap::new();
        assert_eq!(tier_bytes(&map), 0);
    }

    #[test]
    fn test_tier_bytes_sum() {
        let mut map = HashMap::new();
        map.insert("a".to_owned(), make_block("a", 100, 0));
        map.insert("b".to_owned(), make_block("b", 200, 0));
        assert_eq!(tier_bytes(&map), 300);
    }

    // ── choose_victim ────────────────────────────────────────────────────────

    #[test]
    fn test_choose_victim_empty() {
        let map: HashMap<String, BcmCachedBlock> = HashMap::new();
        assert!(choose_victim(&map, EvictionPolicy::LRU).is_none());
    }

    #[test]
    fn test_choose_victim_all_pinned() {
        let mut map = HashMap::new();
        let mut b = make_block("cid1", 10, 0);
        b.pinned = true;
        map.insert("cid1".to_owned(), b);
        assert!(choose_victim(&map, EvictionPolicy::LRU).is_none());
    }

    #[test]
    fn test_choose_victim_lru_selects_oldest() {
        let mut map = HashMap::new();
        let mut b1 = make_block("cid_old", 10, 0);
        b1.last_accessed = 1;
        let mut b2 = make_block("cid_new", 10, 0);
        b2.last_accessed = 100;
        map.insert("cid_old".to_owned(), b1);
        map.insert("cid_new".to_owned(), b2);
        let victim = choose_victim(&map, EvictionPolicy::LRU).unwrap();
        assert_eq!(victim, "cid_old");
    }

    #[test]
    fn test_choose_victim_lfu_selects_least_frequent() {
        let mut map = HashMap::new();
        let mut b1 = make_block("cid_rare", 10, 0);
        b1.access_count = 1;
        b1.last_accessed = 50;
        let mut b2 = make_block("cid_freq", 10, 0);
        b2.access_count = 99;
        b2.last_accessed = 10;
        map.insert("cid_rare".to_owned(), b1);
        map.insert("cid_freq".to_owned(), b2);
        let victim = choose_victim(&map, EvictionPolicy::LFU).unwrap();
        assert_eq!(victim, "cid_rare");
    }

    #[test]
    fn test_choose_victim_lfu_tiebreak_by_last_accessed() {
        let mut map = HashMap::new();
        let mut b1 = make_block("cid_a", 10, 0);
        b1.access_count = 3;
        b1.last_accessed = 5;
        let mut b2 = make_block("cid_b", 10, 0);
        b2.access_count = 3;
        b2.last_accessed = 10;
        map.insert("cid_a".to_owned(), b1);
        map.insert("cid_b".to_owned(), b2);
        let victim = choose_victim(&map, EvictionPolicy::LFU).unwrap();
        assert_eq!(victim, "cid_a");
    }

    // ── BlockCacheManager ─────────────────────────────────────────────────────

    #[test]
    fn test_new_empty() {
        let mgr = make_manager();
        assert!(mgr.is_empty());
        assert_eq!(mgr.len(), 0);
    }

    #[test]
    fn test_insert_basic() {
        let mut mgr = make_manager();
        assert!(mgr.insert("cid1".to_owned(), vec![1u8; 64], 100));
        assert!(!mgr.is_empty());
        assert_eq!(mgr.len(), 1);
        assert_eq!(mgr.tier_of("cid1"), Some(CacheTier::Cold));
    }

    #[test]
    fn test_insert_duplicate_returns_false() {
        let mut mgr = make_manager();
        assert!(mgr.insert("cid1".to_owned(), vec![0u8; 64], 100));
        assert!(!mgr.insert("cid1".to_owned(), vec![1u8; 64], 101));
    }

    #[test]
    fn test_insert_multiple() {
        let mut mgr = make_manager();
        for i in 0..5u64 {
            assert!(mgr.insert(format!("cid{i}"), vec![0u8; 50], i));
        }
        assert_eq!(mgr.len(), 5);
    }

    #[test]
    fn test_contains() {
        let mut mgr = make_manager();
        mgr.insert("cid1".to_owned(), vec![0u8; 10], 0);
        assert!(mgr.contains("cid1"));
        assert!(!mgr.contains("cid_missing"));
    }

    #[test]
    fn test_get_returns_data() {
        let mut mgr = make_manager();
        let data = vec![42u8; 64];
        mgr.insert("cid1".to_owned(), data.clone(), 0);
        let retrieved = mgr.get("cid1", 1);
        assert_eq!(retrieved, Some(data.as_slice()));
    }

    #[test]
    fn test_get_missing() {
        let mut mgr = make_manager();
        assert!(mgr.get("cid_missing", 0).is_none());
    }

    #[test]
    fn test_get_updates_access_count() {
        let mut mgr = make_manager();
        mgr.insert("cid1".to_owned(), vec![0u8; 10], 0);
        mgr.get("cid1", 1);
        // After 1 get the cold block should still be cold (below warm_threshold=2).
        // access_count should be 1.
        let block = mgr.cold.get("cid1").unwrap();
        assert_eq!(block.access_count, 1);
    }

    #[test]
    fn test_get_updates_last_accessed() {
        let mut mgr = make_manager();
        mgr.insert("cid1".to_owned(), vec![0u8; 10], 100);
        mgr.get("cid1", 200);
        // Block may have promoted — check wherever it is.
        let ts = mgr
            .cold
            .get("cid1")
            .or_else(|| mgr.warm.get("cid1"))
            .or_else(|| mgr.hot.get("cid1"))
            .map(|b| b.last_accessed)
            .unwrap_or(0);
        assert_eq!(ts, 200);
    }

    // ── Promotion ────────────────────────────────────────────────────────────

    #[test]
    fn test_promote_cold_to_warm() {
        let mut mgr = make_manager();
        mgr.insert("cid1".to_owned(), vec![0u8; 10], 0);
        // access warm_threshold = 2 times to trigger warm promotion.
        for i in 1..=2u64 {
            mgr.get("cid1", i);
        }
        assert_eq!(mgr.tier_of("cid1"), Some(CacheTier::Warm));
        assert_eq!(mgr.stats().promotions, 1);
    }

    #[test]
    fn test_promote_warm_to_hot() {
        let mut mgr = make_manager();
        mgr.insert("cid1".to_owned(), vec![0u8; 10], 0);
        // hot_threshold = 5
        for i in 1..=5u64 {
            mgr.get("cid1", i);
        }
        assert_eq!(mgr.tier_of("cid1"), Some(CacheTier::Hot));
        let s = mgr.stats();
        // At least two promotions: cold→warm then warm→hot (may be more due to re-checks).
        assert!(s.promotions >= 2);
    }

    #[test]
    fn test_no_promotion_below_warm_threshold() {
        let mut mgr = make_manager();
        mgr.insert("cid1".to_owned(), vec![0u8; 10], 0);
        // Access just once — below warm_threshold=2.
        mgr.get("cid1", 1);
        assert_eq!(mgr.tier_of("cid1"), Some(CacheTier::Cold));
        assert_eq!(mgr.stats().promotions, 0);
    }

    // ── Demotion ─────────────────────────────────────────────────────────────

    #[test]
    fn test_demote_hot_to_warm() {
        let mut mgr = make_manager();
        mgr.insert("cid1".to_owned(), vec![0u8; 10], 0);
        // promote to Hot
        for i in 1..=5u64 {
            mgr.get("cid1", i);
        }
        assert_eq!(mgr.tier_of("cid1"), Some(CacheTier::Hot));
        mgr.demote("cid1", 10);
        assert_eq!(mgr.tier_of("cid1"), Some(CacheTier::Warm));
        assert_eq!(mgr.stats().demotions, 1);
    }

    #[test]
    fn test_demote_warm_to_cold() {
        let mut mgr = make_manager();
        mgr.insert("cid1".to_owned(), vec![0u8; 10], 0);
        for i in 1..=2u64 {
            mgr.get("cid1", i);
        }
        assert_eq!(mgr.tier_of("cid1"), Some(CacheTier::Warm));
        mgr.demote("cid1", 10);
        assert_eq!(mgr.tier_of("cid1"), Some(CacheTier::Cold));
    }

    #[test]
    fn test_demote_cold_is_noop() {
        let mut mgr = make_manager();
        mgr.insert("cid1".to_owned(), vec![0u8; 10], 0);
        mgr.demote("cid1", 1);
        assert_eq!(mgr.tier_of("cid1"), Some(CacheTier::Cold));
        assert_eq!(mgr.stats().demotions, 0);
    }

    #[test]
    fn test_demote_missing_is_noop() {
        let mut mgr = make_manager();
        mgr.demote("cid_missing", 0);
        assert_eq!(mgr.stats().demotions, 0);
    }

    // ── Pinning ───────────────────────────────────────────────────────────────

    #[test]
    fn test_pin_and_unpin() {
        let mut mgr = make_manager();
        mgr.insert("cid1".to_owned(), vec![0u8; 10], 0);
        assert!(mgr.pin("cid1"));
        assert_eq!(mgr.stats().pinned_count, 1);
        assert!(mgr.unpin("cid1"));
        assert_eq!(mgr.stats().pinned_count, 0);
    }

    #[test]
    fn test_pin_missing() {
        let mut mgr = make_manager();
        assert!(!mgr.pin("cid_missing"));
    }

    #[test]
    fn test_unpin_missing() {
        let mut mgr = make_manager();
        assert!(!mgr.unpin("cid_missing"));
    }

    #[test]
    fn test_pinned_block_not_evicted() {
        let mut mgr = make_manager();
        mgr.insert("cid1".to_owned(), vec![0u8; 10], 0);
        mgr.pin("cid1");
        // Force eviction of more bytes than the block uses.
        let freed = mgr.evict_to_fit(CacheTier::Cold, 10, 1);
        assert_eq!(freed, 0);
        assert!(mgr.contains("cid1"));
    }

    #[test]
    fn test_remove_unpinned() {
        let mut mgr = make_manager();
        mgr.insert("cid1".to_owned(), vec![0u8; 10], 0);
        assert!(mgr.remove("cid1"));
        assert!(!mgr.contains("cid1"));
    }

    #[test]
    fn test_remove_pinned_fails() {
        let mut mgr = make_manager();
        mgr.insert("cid1".to_owned(), vec![0u8; 10], 0);
        mgr.pin("cid1");
        assert!(!mgr.remove("cid1"));
        assert!(mgr.contains("cid1"));
    }

    #[test]
    fn test_remove_missing() {
        let mut mgr = make_manager();
        assert!(!mgr.remove("cid_missing"));
    }

    // ── Byte accounting ───────────────────────────────────────────────────────

    #[test]
    fn test_total_bytes() {
        let mut mgr = make_manager();
        mgr.insert("cid1".to_owned(), vec![0u8; 100], 0);
        assert_eq!(mgr.total_bytes(), 100);
        assert_eq!(mgr.total_cold_bytes(), 100);
        assert_eq!(mgr.total_hot_bytes(), 0);
        assert_eq!(mgr.total_warm_bytes(), 0);
    }

    #[test]
    fn test_bytes_after_promotion() {
        let mut mgr = make_manager();
        mgr.insert("cid1".to_owned(), vec![0u8; 100], 0);
        for i in 1..=2u64 {
            mgr.get("cid1", i);
        }
        assert_eq!(mgr.tier_of("cid1"), Some(CacheTier::Warm));
        assert_eq!(mgr.total_warm_bytes(), 100);
        assert_eq!(mgr.total_cold_bytes(), 0);
        assert_eq!(mgr.total_bytes(), 100);
    }

    // ── Eviction ─────────────────────────────────────────────────────────────

    #[test]
    fn test_evict_to_fit_frees_bytes() {
        let mut mgr = make_manager();
        mgr.insert("cid1".to_owned(), vec![0u8; 100], 0);
        mgr.insert("cid2".to_owned(), vec![0u8; 100], 1);
        let freed = mgr.evict_to_fit(CacheTier::Cold, 100, 2);
        assert!(freed >= 100);
    }

    #[test]
    fn test_evict_empty_tier_returns_zero() {
        let mut mgr = make_manager();
        let freed = mgr.evict_to_fit(CacheTier::Hot, 100, 0);
        assert_eq!(freed, 0);
    }

    #[test]
    fn test_auto_eviction_on_cold_overflow() {
        // Cold capacity = 4096 bytes; insert blocks totaling > 4096.
        let mut mgr = make_manager();
        // Insert 50 blocks of 100 bytes = 5000 bytes > 4096.
        for i in 0..50u64 {
            mgr.insert(format!("cid{i}"), vec![0u8; 100], i);
        }
        assert!(
            mgr.total_cold_bytes() <= 4096,
            "Cold tier exceeded capacity"
        );
    }

    #[test]
    fn test_evictions_counter() {
        let mut mgr = make_manager();
        for i in 0..50u64 {
            mgr.insert(format!("cid{i}"), vec![0u8; 100], i);
        }
        assert!(mgr.stats().evictions > 0);
    }

    // ── Stats ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_stats_initial() {
        let mgr = make_manager();
        let s = mgr.stats();
        assert_eq!(s.hot_count, 0);
        assert_eq!(s.warm_count, 0);
        assert_eq!(s.cold_count, 0);
        assert_eq!(s.evictions, 0);
        assert_eq!(s.promotions, 0);
        assert_eq!(s.demotions, 0);
    }

    #[test]
    fn test_stats_counts() {
        let mut mgr = make_manager();
        mgr.insert("cid1".to_owned(), vec![0u8; 10], 0);
        mgr.insert("cid2".to_owned(), vec![0u8; 10], 1);
        let s = mgr.stats();
        assert_eq!(s.cold_count, 2);
    }

    #[test]
    fn test_stats_default() {
        let s = BcmCacheStats::default();
        assert_eq!(s.evictions, 0);
        assert_eq!(s.hot_count, 0);
    }

    // ── tier_of ───────────────────────────────────────────────────────────────

    #[test]
    fn test_tier_of_missing() {
        let mgr = make_manager();
        assert!(mgr.tier_of("missing").is_none());
    }

    #[test]
    fn test_tier_of_cold() {
        let mut mgr = make_manager();
        mgr.insert("cid1".to_owned(), vec![0u8; 10], 0);
        assert_eq!(mgr.tier_of("cid1"), Some(CacheTier::Cold));
    }

    // ── LFU policy ────────────────────────────────────────────────────────────

    #[test]
    fn test_lfu_evicts_least_frequent() {
        let config = BcmCacheConfig {
            max_cold_bytes: 200,
            eviction_policy: EvictionPolicy::LFU,
            ..default_config()
        };
        let mut mgr = BlockCacheManager::new(config);

        mgr.insert("cid_rare".to_owned(), vec![0u8; 100], 0);
        // access cid_rare only once
        mgr.get("cid_rare", 1);

        mgr.insert("cid_freq".to_owned(), vec![0u8; 100], 2);
        // access cid_freq many times — keep it below warm_threshold to avoid promotion
        // (warm_threshold = 2, so access 1 time to stay in cold but have higher count than cid_rare's 1)
        // We need cid_rare to remain cold too. Reset access_count tricks won't work.
        // Instead: cid_rare was accessed at t=1 (count=1), cid_freq at t=2 (count=0 still).
        // So LFU should evict cid_freq (count=0).

        // Now force insert a third block to trigger eviction.
        mgr.insert("cid_new".to_owned(), vec![0u8; 100], 3);

        // The block with lower access_count (cid_freq, count=0) should be evicted.
        assert!(
            !mgr.contains("cid_freq"),
            "cid_freq should have been evicted by LFU"
        );
        assert!(mgr.contains("cid_rare"), "cid_rare should still be present");
    }

    // ── TwoQ / ARC fall back to LRU ──────────────────────────────────────────

    #[test]
    fn test_twoq_behaves_like_lru() {
        let config = BcmCacheConfig {
            max_cold_bytes: 200,
            eviction_policy: EvictionPolicy::TwoQ,
            ..default_config()
        };
        let mut mgr = BlockCacheManager::new(config);
        mgr.insert("cid_old".to_owned(), vec![0u8; 100], 0);
        mgr.insert("cid_new".to_owned(), vec![0u8; 100], 10);
        // force eviction
        mgr.insert("cid_extra".to_owned(), vec![0u8; 100], 20);
        // cid_old (last_accessed=0) should be evicted over cid_new (last_accessed=10).
        assert!(!mgr.contains("cid_old"));
    }

    #[test]
    fn test_arc_behaves_like_lru() {
        let config = BcmCacheConfig {
            max_cold_bytes: 200,
            eviction_policy: EvictionPolicy::ARC,
            ..default_config()
        };
        let mut mgr = BlockCacheManager::new(config);
        mgr.insert("cid_old".to_owned(), vec![0u8; 100], 0);
        mgr.insert("cid_new".to_owned(), vec![0u8; 100], 10);
        mgr.insert("cid_extra".to_owned(), vec![0u8; 100], 20);
        assert!(!mgr.contains("cid_old"));
    }

    // ── Promote explicitly called ─────────────────────────────────────────────

    #[test]
    fn test_explicit_promote_missing_is_noop() {
        let mut mgr = make_manager();
        mgr.promote("cid_missing", 0); // should not panic
    }

    #[test]
    fn test_explicit_demote_on_hot_block() {
        let mut mgr = make_manager();
        mgr.insert("cid1".to_owned(), vec![0u8; 10], 0);
        for i in 1..=5u64 {
            mgr.get("cid1", i);
        }
        assert_eq!(mgr.tier_of("cid1"), Some(CacheTier::Hot));

        mgr.demote("cid1", 10);
        assert_eq!(mgr.tier_of("cid1"), Some(CacheTier::Warm));
        mgr.demote("cid1", 11);
        assert_eq!(mgr.tier_of("cid1"), Some(CacheTier::Cold));
        // Cold → demote → still Cold
        mgr.demote("cid1", 12);
        assert_eq!(mgr.tier_of("cid1"), Some(CacheTier::Cold));
    }

    // ── Default config ────────────────────────────────────────────────────────

    #[test]
    fn test_default_config_values() {
        let cfg = BcmCacheConfig::default();
        assert_eq!(cfg.max_hot_bytes, 64 * 1024 * 1024);
        assert_eq!(cfg.max_warm_bytes, 128 * 1024 * 1024);
        assert_eq!(cfg.max_cold_bytes, 256 * 1024 * 1024);
        assert_eq!(cfg.hot_threshold, 10);
        assert_eq!(cfg.warm_threshold, 3);
    }

    // ── Large insert / eviction stress ───────────────────────────────────────

    #[test]
    fn test_large_insert_respects_cold_cap() {
        let mut mgr = make_manager(); // cold cap = 4096
        for i in 0..100u64 {
            mgr.insert(format!("c{i}"), vec![9u8; 100], i);
        }
        assert!(
            mgr.total_cold_bytes() <= 4096 + 100,
            "cold bytes {} exceed cap",
            mgr.total_cold_bytes()
        );
    }

    #[test]
    fn test_total_bytes_consistent() {
        let mut mgr = make_manager();
        for i in 0..10u64 {
            mgr.insert(format!("c{i}"), vec![0u8; 50], i);
        }
        let s = mgr.stats();
        assert_eq!(s.hot_bytes + s.warm_bytes + s.cold_bytes, mgr.total_bytes());
    }
}
