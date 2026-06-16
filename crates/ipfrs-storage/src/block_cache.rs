//! In-memory LRU/LFU block cache with TTL-based and size-based eviction.
//!
//! # Overview
//!
//! [`BlockCache`] is a standalone, synchronous in-memory cache for raw block
//! data (identified by CID strings).  It supports three eviction policies:
//!
//! * [`EvictionPolicy::LRU`] — evict the entry that was accessed least recently.
//! * [`EvictionPolicy::LFU`] — evict the entry that was accessed least frequently
//!   (tie-broken by last-access time so cold entries leave first).
//! * [`EvictionPolicy::TTL`] — evict expired entries first; if none are expired,
//!   fall back to LRU.
//!
//! Capacity is bounded by **both** a maximum entry count and a maximum byte
//! footprint.  Expired entries are lazily removed on access and can also be
//! swept in bulk via [`BlockCache::evict_expired`].

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// CacheEntry
// ---------------------------------------------------------------------------

/// A single cached block together with its bookkeeping metadata.
#[derive(Debug, Clone)]
pub struct CacheEntry {
    /// Content identifier of the block.
    pub cid: String,
    /// Raw block bytes.
    pub data: Vec<u8>,
    /// Unix timestamp (seconds) at which this entry was first inserted.
    pub inserted_at_secs: u64,
    /// Unix timestamp (seconds) of the most recent cache hit; updated on every
    /// successful [`BlockCache::get`] call.
    pub last_accessed_secs: u64,
    /// Number of times this entry has been returned by [`BlockCache::get`].
    pub access_count: u64,
}

impl CacheEntry {
    /// Returns `true` when the entry has lived longer than `ttl_secs` seconds
    /// relative to `now_secs`.
    #[inline]
    pub fn is_expired(&self, ttl_secs: u64, now_secs: u64) -> bool {
        now_secs.saturating_sub(self.inserted_at_secs) >= ttl_secs
    }

    /// Returns the size of the cached data in bytes.
    #[inline]
    pub fn size_bytes(&self) -> usize {
        self.data.len()
    }
}

// ---------------------------------------------------------------------------
// EvictionPolicy
// ---------------------------------------------------------------------------

/// Determines which entry is chosen for eviction when the cache is full.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvictionPolicy {
    /// Evict the entry that was accessed least recently (minimum
    /// `last_accessed_secs`).
    LRU,
    /// Evict the entry that has been accessed fewest times (`access_count`),
    /// with `last_accessed_secs` as a tie-breaker (older access loses).
    LFU,
    /// Evict expired entries first.  If no entries are expired, fall back to
    /// LRU eviction.
    TTL,
}

// ---------------------------------------------------------------------------
// CacheConfig
// ---------------------------------------------------------------------------

/// Configuration for a [`BlockCache`].
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Maximum number of entries the cache may hold simultaneously.
    ///
    /// Default: 1 000.
    pub max_entries: usize,
    /// Maximum total byte footprint of all cached blocks.
    ///
    /// Default: 64 MiB.
    pub max_bytes: usize,
    /// Time-to-live for each entry in seconds.  Entries older than this
    /// (relative to their `inserted_at_secs`) are considered expired.
    ///
    /// Default: 300 s (5 minutes).
    pub ttl_secs: u64,
    /// Eviction policy applied when the cache is at capacity.
    ///
    /// Default: [`EvictionPolicy::LRU`].
    pub policy: EvictionPolicy,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 1_000,
            max_bytes: 64 * 1024 * 1024,
            ttl_secs: 300,
            policy: EvictionPolicy::LRU,
        }
    }
}

// ---------------------------------------------------------------------------
// CacheStats
// ---------------------------------------------------------------------------

/// Cumulative statistics for a [`BlockCache`].
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Total number of successful cache lookups.
    pub hits: u64,
    /// Total number of failed cache lookups (including expired-entry misses).
    pub misses: u64,
    /// Total number of entries evicted due to capacity pressure.
    pub evictions: u64,
    /// Subset of `misses` caused by an expired entry being removed.
    pub expired_evictions: u64,
    /// Current aggregate size (bytes) of all live entries.
    pub total_bytes: u64,
    /// Current number of live entries.
    pub entry_count: u64,
}

impl CacheStats {
    /// Returns the fraction of lookups that were satisfied from the cache.
    ///
    /// Returns `0.0` when neither hits nor misses have been recorded yet.
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

// ---------------------------------------------------------------------------
// BlockCache
// ---------------------------------------------------------------------------

/// An in-memory cache for raw block data with LRU/LFU/TTL eviction and
/// size-based capacity limits.
#[derive(Debug)]
pub struct BlockCache {
    entries: HashMap<String, CacheEntry>,
    config: CacheConfig,
    stats: CacheStats,
}

impl BlockCache {
    /// Creates a new `BlockCache` with the given [`CacheConfig`].
    pub fn new(config: CacheConfig) -> Self {
        Self {
            entries: HashMap::new(),
            config,
            stats: CacheStats::default(),
        }
    }

    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------

    /// Looks up a block by CID.
    ///
    /// * **Hit (fresh)**: updates `last_accessed_secs` and `access_count`,
    ///   increments `stats.hits`, and returns a slice of the data.
    /// * **Hit (expired)**: removes the entry, increments `stats.misses` *and*
    ///   `stats.expired_evictions`, and returns `None`.
    /// * **Miss**: increments `stats.misses` and returns `None`.
    pub fn get(&mut self, cid: &str, now_secs: u64) -> Option<&[u8]> {
        // Check expiry without a borrow on `self.entries` first so we can
        // mutate the map afterwards.
        let expired = self
            .entries
            .get(cid)
            .map(|e| e.is_expired(self.config.ttl_secs, now_secs))
            .unwrap_or(false);

        if expired {
            // Remove the entry and account for freed bytes.
            if let Some(entry) = self.entries.remove(cid) {
                let freed = entry.size_bytes() as u64;
                self.stats.total_bytes = self.stats.total_bytes.saturating_sub(freed);
                self.stats.entry_count = self.stats.entry_count.saturating_sub(1);
            }
            self.stats.misses += 1;
            self.stats.expired_evictions += 1;
            return None;
        }

        if let Some(entry) = self.entries.get_mut(cid) {
            entry.last_accessed_secs = now_secs;
            entry.access_count += 1;
            self.stats.hits += 1;
            // SAFETY: the borrow of `entry` ends here; we re-borrow immutably
            // for the return value.
            return Some(&self.entries[cid].data);
        }

        self.stats.misses += 1;
        None
    }

    /// Inserts a new block into the cache.
    ///
    /// If the cache is at or above capacity (entry count **or** byte budget),
    /// entries are evicted according to [`CacheConfig::policy`] until the new
    /// block fits.  If the block itself exceeds `max_bytes` on its own, all
    /// existing entries are evicted and the oversized block is still inserted
    /// (single-block cache).
    pub fn insert(&mut self, cid: String, data: Vec<u8>, now_secs: u64) {
        let new_size = data.len() as u64;

        // If this CID is already cached, remove the old entry first so we
        // don't double-count its bytes.
        if let Some(old) = self.entries.remove(&cid) {
            let freed = old.size_bytes() as u64;
            self.stats.total_bytes = self.stats.total_bytes.saturating_sub(freed);
            self.stats.entry_count = self.stats.entry_count.saturating_sub(1);
        }

        // Evict until both constraints are satisfied.
        while !self.entries.is_empty()
            && (self.entries.len() >= self.config.max_entries
                || self.stats.total_bytes + new_size > self.config.max_bytes as u64)
        {
            if !self.evict_one(now_secs) {
                // Nothing left to evict.
                break;
            }
        }

        let entry = CacheEntry {
            cid: cid.clone(),
            data,
            inserted_at_secs: now_secs,
            last_accessed_secs: now_secs,
            access_count: 0,
        };

        self.stats.total_bytes += new_size;
        self.stats.entry_count += 1;
        self.entries.insert(cid, entry);
    }

    /// Evicts a single entry according to the configured policy.
    ///
    /// Returns `true` if an entry was removed, `false` if the cache was already
    /// empty.
    pub fn evict_one(&mut self, now_secs: u64) -> bool {
        if self.entries.is_empty() {
            return false;
        }

        let victim_cid = match self.config.policy {
            EvictionPolicy::LRU => self.find_lru_victim(),
            EvictionPolicy::LFU => self.find_lfu_victim(),
            EvictionPolicy::TTL => {
                // Prefer an expired entry; fall back to LRU.
                self.find_expired_victim(now_secs)
                    .or_else(|| self.find_lru_victim())
            }
        };

        if let Some(cid) = victim_cid {
            if let Some(entry) = self.entries.remove(&cid) {
                let freed = entry.size_bytes() as u64;
                self.stats.total_bytes = self.stats.total_bytes.saturating_sub(freed);
                self.stats.entry_count = self.stats.entry_count.saturating_sub(1);
                self.stats.evictions += 1;
                return true;
            }
        }

        false
    }

    /// Removes all expired entries from the cache in one pass.
    ///
    /// Returns the number of entries that were removed.
    pub fn evict_expired(&mut self, now_secs: u64) -> usize {
        let ttl = self.config.ttl_secs;
        let expired_cids: Vec<String> = self
            .entries
            .iter()
            .filter(|(_, e)| e.is_expired(ttl, now_secs))
            .map(|(cid, _)| cid.clone())
            .collect();

        let count = expired_cids.len();
        for cid in &expired_cids {
            if let Some(entry) = self.entries.remove(cid) {
                let freed = entry.size_bytes() as u64;
                self.stats.total_bytes = self.stats.total_bytes.saturating_sub(freed);
                self.stats.entry_count = self.stats.entry_count.saturating_sub(1);
                self.stats.evictions += 1;
                self.stats.expired_evictions += 1;
            }
        }

        count
    }

    /// Explicitly removes a cached entry by CID.
    ///
    /// Returns `true` if the entry existed and was removed.
    pub fn remove(&mut self, cid: &str) -> bool {
        if let Some(entry) = self.entries.remove(cid) {
            let freed = entry.size_bytes() as u64;
            self.stats.total_bytes = self.stats.total_bytes.saturating_sub(freed);
            self.stats.entry_count = self.stats.entry_count.saturating_sub(1);
            true
        } else {
            false
        }
    }

    /// Returns `true` if a non-expired entry for `cid` is currently in the
    /// cache.  Unlike [`get`](Self::get) this method does **not** update any
    /// access metadata and does **not** remove expired entries.
    pub fn contains(&self, cid: &str, now_secs: u64) -> bool {
        self.entries
            .get(cid)
            .map(|e| !e.is_expired(self.config.ttl_secs, now_secs))
            .unwrap_or(false)
    }

    /// Returns a reference to the cumulative [`CacheStats`].
    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }

    /// Removes all entries and resets all statistics.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.stats = CacheStats::default();
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Finds the CID of the entry with the smallest `last_accessed_secs`.
    fn find_lru_victim(&self) -> Option<String> {
        self.entries
            .iter()
            .min_by_key(|(_, e)| e.last_accessed_secs)
            .map(|(cid, _)| cid.clone())
    }

    /// Finds the CID of the entry with the smallest `access_count`, breaking
    /// ties by preferring the entry with the oldest `last_accessed_secs`.
    fn find_lfu_victim(&self) -> Option<String> {
        self.entries
            .iter()
            .min_by(|(_, a), (_, b)| {
                a.access_count
                    .cmp(&b.access_count)
                    .then_with(|| a.last_accessed_secs.cmp(&b.last_accessed_secs))
            })
            .map(|(cid, _)| cid.clone())
    }

    /// Finds any expired entry.  The specific choice is deterministic (minimum
    /// `inserted_at_secs`) so tests can reason about which entry is chosen.
    fn find_expired_victim(&self, now_secs: u64) -> Option<String> {
        let ttl = self.config.ttl_secs;
        self.entries
            .iter()
            .filter(|(_, e)| e.is_expired(ttl, now_secs))
            .min_by_key(|(_, e)| e.inserted_at_secs)
            .map(|(cid, _)| cid.clone())
    }
}

// ---------------------------------------------------------------------------
// Convenience constructor: current-time helpers
// ---------------------------------------------------------------------------

/// Returns the current Unix time in seconds, falling back to 0 on error.
///
/// Provided as a free function so callers that already have a timestamp do not
/// need to touch the system clock.
pub fn current_unix_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: build a cache with a given capacity and TTL.
    fn make_cache(
        max_entries: usize,
        max_bytes: usize,
        ttl_secs: u64,
        policy: EvictionPolicy,
    ) -> BlockCache {
        BlockCache::new(CacheConfig {
            max_entries,
            max_bytes,
            ttl_secs,
            policy,
        })
    }

    // -----------------------------------------------------------------------
    // 1. new() with default config
    // -----------------------------------------------------------------------
    #[test]
    fn test_new_default_config() {
        let cache = BlockCache::new(CacheConfig::default());
        assert_eq!(cache.stats().entry_count, 0);
        assert_eq!(cache.stats().total_bytes, 0);
        assert_eq!(cache.stats().hits, 0);
        assert_eq!(cache.stats().misses, 0);
    }

    // -----------------------------------------------------------------------
    // 2. insert and get — cache hit
    // -----------------------------------------------------------------------
    #[test]
    fn test_insert_and_get_hit() {
        let mut cache = make_cache(100, 1024 * 1024, 300, EvictionPolicy::LRU);
        cache.insert("cid1".to_string(), vec![1u8, 2, 3], 1000);
        let result = cache.get("cid1", 1001);
        assert_eq!(result, Some([1u8, 2, 3].as_ref()));
        assert_eq!(cache.stats().hits, 1);
    }

    // -----------------------------------------------------------------------
    // 3. get miss increments misses
    // -----------------------------------------------------------------------
    #[test]
    fn test_get_miss_increments_misses() {
        let mut cache = make_cache(100, 1024 * 1024, 300, EvictionPolicy::LRU);
        let result = cache.get("nonexistent", 1000);
        assert!(result.is_none());
        assert_eq!(cache.stats().misses, 1);
        assert_eq!(cache.stats().hits, 0);
    }

    // -----------------------------------------------------------------------
    // 4. get on expired entry removes it and returns None
    // -----------------------------------------------------------------------
    #[test]
    fn test_get_expired_removed_returns_none() {
        let mut cache = make_cache(100, 1024 * 1024, 10, EvictionPolicy::LRU);
        cache.insert("cid_exp".to_string(), vec![9u8; 8], 1000);

        // now_secs is 20 seconds after insertion — TTL is 10 s, so expired.
        let result = cache.get("cid_exp", 1020);
        assert!(result.is_none());
        assert_eq!(cache.stats().misses, 1);
        assert_eq!(cache.stats().expired_evictions, 1);
        assert_eq!(cache.stats().entry_count, 0);
        assert_eq!(cache.stats().total_bytes, 0);
    }

    // -----------------------------------------------------------------------
    // 5. insert at entry capacity evicts LRU
    // -----------------------------------------------------------------------
    #[test]
    fn test_insert_at_entry_capacity_evicts_lru() {
        // capacity = 2 entries
        let mut cache = make_cache(2, 1024 * 1024, 3600, EvictionPolicy::LRU);

        // Insert two entries at different times so we know the LRU victim.
        cache.insert("oldest".to_string(), vec![1u8], 100);
        cache.insert("newer".to_string(), vec![2u8], 200);

        // Simulate a hit on "oldest" to make "newer" the least-recently-used
        // entry, then verify the right one is evicted.
        // Actually: after insertion both entries have last_accessed == inserted.
        // "oldest" has last_accessed=100, "newer" has last_accessed=200.
        // So LRU victim is "oldest".

        // Inserting a third entry should evict "oldest".
        cache.insert("third".to_string(), vec![3u8], 300);
        assert_eq!(cache.stats().evictions, 1);
        assert!(cache.get("oldest", 300).is_none()); // misses
        assert!(cache.get("newer", 300).is_some());
        assert!(cache.get("third", 300).is_some());
    }

    // -----------------------------------------------------------------------
    // 6. insert at byte capacity evicts until it fits
    // -----------------------------------------------------------------------
    #[test]
    fn test_insert_at_byte_capacity_evicts_until_fits() {
        // max_bytes = 20; each block is 8 bytes.
        // After 2 entries (16 bytes), adding another 8-byte block (24 > 20)
        // requires eviction.
        let mut cache = make_cache(1000, 20, 3600, EvictionPolicy::LRU);
        cache.insert("a".to_string(), vec![0u8; 8], 100);
        cache.insert("b".to_string(), vec![0u8; 8], 200);
        assert_eq!(cache.stats().total_bytes, 16);

        // Adding 8 more bytes (total 24 > 20) should evict "a" (LRU, last_accessed=100).
        cache.insert("c".to_string(), vec![0u8; 8], 300);
        assert_eq!(cache.stats().total_bytes, 16); // b(8) + c(8)
        assert_eq!(cache.stats().evictions, 1);
        assert!(cache.get("a", 300).is_none());
        assert!(cache.get("b", 300).is_some());
        assert!(cache.get("c", 300).is_some());
    }

    // -----------------------------------------------------------------------
    // 7. LFU eviction picks lowest access_count
    // -----------------------------------------------------------------------
    #[test]
    fn test_lfu_eviction_picks_lowest_access_count() {
        let mut cache = make_cache(2, 1024 * 1024, 3600, EvictionPolicy::LFU);
        cache.insert("popular".to_string(), vec![1u8], 100);
        cache.insert("cold".to_string(), vec![2u8], 100);

        // Access "popular" twice.
        cache.get("popular", 101);
        cache.get("popular", 102);
        // "cold" has access_count=0, "popular" has access_count=2.

        // Inserting a third entry should evict "cold".
        cache.insert("new".to_string(), vec![3u8], 200);
        assert_eq!(cache.stats().evictions, 1);
        assert!(cache.get("cold", 200).is_none()); // evicted
        assert!(cache.get("popular", 200).is_some());
        assert!(cache.get("new", 200).is_some());
    }

    // -----------------------------------------------------------------------
    // 8. TTL policy evicts expired first, then falls back to LRU
    // -----------------------------------------------------------------------
    #[test]
    fn test_ttl_policy_expired_first_then_lru() {
        // TTL = 50 s; capacity = 2.
        let mut cache = make_cache(2, 1024 * 1024, 50, EvictionPolicy::TTL);

        // Insert "stale" at t=0 and "fresh" at t=0.
        cache.insert("stale".to_string(), vec![1u8], 0);
        cache.insert("fresh".to_string(), vec![2u8], 0);

        // At t=100, "stale" is expired (inserted_at=0, ttl=50, 100 >= 50).
        // "fresh" is also expired. Both are expired.
        // The TTL policy picks the one with minimum inserted_at_secs first.
        // Both have inserted_at=0, so the sort is stable on cid string order.
        // Regardless: eviction should remove an expired one.
        cache.insert("new".to_string(), vec![3u8], 100);
        assert_eq!(cache.stats().evictions, 1);
        // At least one entry from the original two should be gone.
        let stale_present = cache.get("stale", 100).is_some();
        let fresh_present = cache.get("fresh", 100).is_some();
        // Both are actually expired at t=100 so whichever was evicted is None,
        // the other is also a miss due to TTL-on-get.  Either way "new" must be there.
        assert!(!stale_present && !fresh_present);
        assert!(cache.get("new", 100).is_some());
    }

    // -----------------------------------------------------------------------
    // 9. access_count incremented on each hit
    // -----------------------------------------------------------------------
    #[test]
    fn test_access_count_incremented_on_hit() {
        let mut cache = make_cache(100, 1024 * 1024, 3600, EvictionPolicy::LRU);
        cache.insert("x".to_string(), vec![42u8], 1000);
        cache.get("x", 1001);
        cache.get("x", 1002);
        cache.get("x", 1003);

        let entry = cache.entries.get("x").expect("entry must exist");
        assert_eq!(entry.access_count, 3);
    }

    // -----------------------------------------------------------------------
    // 10. last_accessed_secs updated on hit
    // -----------------------------------------------------------------------
    #[test]
    fn test_last_accessed_updated_on_hit() {
        let mut cache = make_cache(100, 1024 * 1024, 3600, EvictionPolicy::LRU);
        cache.insert("y".to_string(), vec![7u8], 1000);
        cache.get("y", 2000);

        let entry = cache.entries.get("y").expect("entry must exist");
        assert_eq!(entry.last_accessed_secs, 2000);
    }

    // -----------------------------------------------------------------------
    // 11. evict_expired removes multiple entries
    // -----------------------------------------------------------------------
    #[test]
    fn test_evict_expired_removes_multiple() {
        let mut cache = make_cache(100, 1024 * 1024, 30, EvictionPolicy::LRU);
        cache.insert("e1".to_string(), vec![1u8], 0);
        cache.insert("e2".to_string(), vec![2u8], 0);
        cache.insert("fresh".to_string(), vec![3u8], 50);

        // At t=40, e1 and e2 are expired (age 40 >= ttl 30); "fresh" is not.
        let removed = cache.evict_expired(40);
        assert_eq!(removed, 2);
        assert_eq!(cache.stats().entry_count, 1);
        // "fresh" inserted at t=50 is not yet expired at t=40 (age is negative
        // conceptually — it was inserted in the future relative to now_secs).
        // Actually: age = now_secs(40) - inserted_at(50) = saturating_sub -> 0.
        // 0 < 30, so not expired. Correct.
    }

    // -----------------------------------------------------------------------
    // 12. evict_expired returns correct count
    // -----------------------------------------------------------------------
    #[test]
    fn test_evict_expired_returns_correct_count() {
        let mut cache = make_cache(100, 1024 * 1024, 10, EvictionPolicy::LRU);
        for i in 0u64..5 {
            cache.insert(format!("cid_{i}"), vec![i as u8], i * 2);
        }
        // TTL = 10 s; at t=12:
        //   cid_0: age=12 >= 10 → expired
        //   cid_1: age=10 >= 10 → expired
        //   cid_2: age=8  < 10  → fresh
        //   cid_3: age=6  < 10  → fresh
        //   cid_4: age=4  < 10  → fresh
        let count = cache.evict_expired(12);
        assert_eq!(count, 2);
    }

    // -----------------------------------------------------------------------
    // 13. remove explicit returns true
    // -----------------------------------------------------------------------
    #[test]
    fn test_remove_explicit_returns_true() {
        let mut cache = make_cache(100, 1024 * 1024, 3600, EvictionPolicy::LRU);
        cache.insert("r1".to_string(), vec![0u8; 16], 1000);
        let removed = cache.remove("r1");
        assert!(removed);
        assert_eq!(cache.stats().entry_count, 0);
        assert_eq!(cache.stats().total_bytes, 0);
    }

    // -----------------------------------------------------------------------
    // 14. remove non-existent returns false
    // -----------------------------------------------------------------------
    #[test]
    fn test_remove_nonexistent_returns_false() {
        let mut cache = make_cache(100, 1024 * 1024, 3600, EvictionPolicy::LRU);
        let removed = cache.remove("ghost");
        assert!(!removed);
    }

    // -----------------------------------------------------------------------
    // 15. contains() true for fresh entry
    // -----------------------------------------------------------------------
    #[test]
    fn test_contains_true_for_fresh_entry() {
        let mut cache = make_cache(100, 1024 * 1024, 300, EvictionPolicy::LRU);
        cache.insert("c1".to_string(), vec![1u8], 1000);
        assert!(cache.contains("c1", 1100));
    }

    // -----------------------------------------------------------------------
    // 16. contains() false for expired entry
    // -----------------------------------------------------------------------
    #[test]
    fn test_contains_false_for_expired() {
        let mut cache = make_cache(100, 1024 * 1024, 10, EvictionPolicy::LRU);
        cache.insert("c2".to_string(), vec![1u8], 1000);
        // age = 1020 - 1000 = 20 >= ttl 10 → expired
        assert!(!cache.contains("c2", 1020));
    }

    // -----------------------------------------------------------------------
    // 17. hit_rate() calculation
    // -----------------------------------------------------------------------
    #[test]
    fn test_hit_rate_calculation() {
        let mut cache = make_cache(100, 1024 * 1024, 3600, EvictionPolicy::LRU);
        // Initially 0/0 → 0.0
        assert_eq!(cache.stats().hit_rate(), 0.0);

        cache.insert("hr".to_string(), vec![1u8], 1000);
        cache.get("hr", 1001); // hit
        cache.get("hr", 1002); // hit
        cache.get("nope", 1003); // miss

        // 2 hits, 1 miss → 2/3
        let rate = cache.stats().hit_rate();
        assert!((rate - 2.0 / 3.0).abs() < 1e-9, "expected 2/3, got {rate}");
    }

    // -----------------------------------------------------------------------
    // 18. clear() resets everything including stats
    // -----------------------------------------------------------------------
    #[test]
    fn test_clear_resets_everything() {
        let mut cache = make_cache(100, 1024 * 1024, 3600, EvictionPolicy::LRU);
        cache.insert("z1".to_string(), vec![0u8; 32], 1000);
        cache.insert("z2".to_string(), vec![0u8; 64], 1000);
        cache.get("z1", 1001);

        cache.clear();

        assert_eq!(cache.stats().hits, 0);
        assert_eq!(cache.stats().misses, 0);
        assert_eq!(cache.stats().evictions, 0);
        assert_eq!(cache.stats().expired_evictions, 0);
        assert_eq!(cache.stats().total_bytes, 0);
        assert_eq!(cache.stats().entry_count, 0);
        assert!(cache.get("z1", 1002).is_none());
        assert!(cache.get("z2", 1002).is_none());
    }

    // -----------------------------------------------------------------------
    // 19. CacheEntry::is_expired boundary conditions
    // -----------------------------------------------------------------------
    #[test]
    fn test_cache_entry_is_expired_boundary() {
        let entry = CacheEntry {
            cid: "x".to_string(),
            data: vec![],
            inserted_at_secs: 100,
            last_accessed_secs: 100,
            access_count: 0,
        };
        // age == ttl → expired (>=)
        assert!(entry.is_expired(50, 150));
        // age < ttl → not expired
        assert!(!entry.is_expired(50, 149));
    }

    // -----------------------------------------------------------------------
    // 20. Reinsertion of existing CID updates data without double-counting bytes
    // -----------------------------------------------------------------------
    #[test]
    fn test_reinsertion_updates_data() {
        let mut cache = make_cache(100, 1024 * 1024, 3600, EvictionPolicy::LRU);
        cache.insert("dup".to_string(), vec![0u8; 8], 1000);
        assert_eq!(cache.stats().total_bytes, 8);

        cache.insert("dup".to_string(), vec![0u8; 16], 1001);
        // Only the new 16 bytes should be accounted for.
        assert_eq!(cache.stats().total_bytes, 16);
        assert_eq!(cache.stats().entry_count, 1);
    }
}
