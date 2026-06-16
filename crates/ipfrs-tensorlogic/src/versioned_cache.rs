//! Versioned Inference Cache with Atomic KB Invalidation
//!
//! Extends the basic `InferenceCache` with explicit version tracking per knowledge base.
//! When a knowledge base is updated, all cached inferences for that KB are atomically
//! invalidated in O(n) time, where n is the number of cached entries for that KB.
//!
//! ## Design
//!
//! - [`CacheKey`] identifies an inference result by goal hash, KB identity, and KB version.
//! - [`CacheEntry`] stores variable bindings, a TTL, and a hit counter.
//! - [`VersionedInferenceCache`] is a concurrent, version-aware cache backed by `RwLock<HashMap>`.
//! - [`CacheStats`] tracks inserts/hits/misses/invalidations/evictions via `AtomicU64`.
//!
//! ## Thread Safety
//!
//! All public methods are `&self` (shared reference). Internal mutation is protected by
//! `RwLock`. [`CacheStats`] uses `AtomicU64` with `Relaxed` ordering for performance —
//! exact ordering guarantees are not required for statistics counters.

use std::collections::HashMap;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    RwLock,
};
use std::time::{Duration, Instant};

use thiserror::Error;

// ---------------------------------------------------------------------------
// FNV-1a
// ---------------------------------------------------------------------------

/// Compute FNV-1a hash of an arbitrary byte slice.
fn fnv1a(data: &[u8]) -> u64 {
    const FNV_OFFSET: u64 = 14_695_981_039_346_656_037;
    const FNV_PRIME: u64 = 1_099_511_628_211;
    let mut hash = FNV_OFFSET;
    for &byte in data {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

// ---------------------------------------------------------------------------
// CacheError
// ---------------------------------------------------------------------------

/// Errors that can occur when interacting with [`VersionedInferenceCache`].
#[derive(Debug, Error)]
pub enum CacheError {
    /// The cache is at capacity; the entry was not inserted.
    #[error("cache capacity exceeded: current={current}, max={max}")]
    CapacityExceeded {
        /// Number of entries currently in the cache.
        current: usize,
        /// Maximum number of entries allowed.
        max: usize,
    },

    /// The supplied key's `kb_version` is older than the current version for that KB.
    #[error(
        "stale version for KB '{kb_id}': cache_version={cache_version}, current_version={current_version}"
    )]
    StaleVersion {
        /// Knowledge base identifier.
        kb_id: String,
        /// Version stored in the cache key.
        cache_version: u64,
        /// Version currently tracked for this KB.
        current_version: u64,
    },
}

// ---------------------------------------------------------------------------
// CacheKey
// ---------------------------------------------------------------------------

/// Composite key for a single cached inference result.
///
/// Two keys are equal iff `goal_hash`, `kb_id`, **and** `kb_version` all match.
/// This means the same goal cached against version 1 and version 2 of the same
/// KB are stored as completely independent entries.
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct CacheKey {
    /// FNV-1a hash of the goal string.
    pub goal_hash: u64,
    /// Identifier of the knowledge base that was queried.
    pub kb_id: String,
    /// Version of the knowledge base at the time of inference.
    pub kb_version: u64,
}

impl CacheKey {
    /// Construct a new [`CacheKey`], hashing `goal` with FNV-1a.
    pub fn new(goal: &str, kb_id: &str, kb_version: u64) -> Self {
        Self {
            goal_hash: fnv1a(goal.as_bytes()),
            kb_id: kb_id.to_owned(),
            kb_version,
        }
    }
}

// ---------------------------------------------------------------------------
// CacheEntry
// ---------------------------------------------------------------------------

/// Default TTL for a cached inference result: 5 minutes.
const DEFAULT_TTL: Duration = Duration::from_secs(5 * 60);

/// A single cached inference result.
#[derive(Debug, Clone)]
pub struct CacheEntry {
    /// Variable bindings produced by the inference engine.
    pub bindings: Vec<HashMap<String, String>>,
    /// `true` if the result is definitive (no further answers possible).
    pub is_final: bool,
    /// Wall-clock instant at which this entry was inserted.
    pub inserted_at: Instant,
    /// Maximum time this entry may be retained.
    pub ttl: Duration,
    /// Number of times this entry has been returned via [`VersionedInferenceCache::get`].
    pub hit_count: u64,
}

impl CacheEntry {
    /// Create a new [`CacheEntry`] with the default TTL (5 minutes).
    pub fn new(bindings: Vec<HashMap<String, String>>, is_final: bool) -> Self {
        Self {
            bindings,
            is_final,
            inserted_at: Instant::now(),
            ttl: DEFAULT_TTL,
            hit_count: 0,
        }
    }

    /// Create a new [`CacheEntry`] with a custom TTL.
    pub fn with_ttl(bindings: Vec<HashMap<String, String>>, is_final: bool, ttl: Duration) -> Self {
        Self {
            bindings,
            is_final,
            inserted_at: Instant::now(),
            ttl,
            hit_count: 0,
        }
    }

    /// Returns `true` if this entry's TTL has elapsed relative to `now`.
    pub fn is_expired(&self, now: Instant) -> bool {
        now.duration_since(self.inserted_at) >= self.ttl
    }
}

// ---------------------------------------------------------------------------
// CacheStats
// ---------------------------------------------------------------------------

/// Atomic statistics counters for [`VersionedInferenceCache`].
#[derive(Debug, Default)]
pub struct CacheStats {
    /// Total number of successful `insert` calls.
    pub total_inserts: AtomicU64,
    /// Total number of cache hits (entry found and not expired).
    pub total_hits: AtomicU64,
    /// Total number of cache misses (entry not found or expired).
    pub total_misses: AtomicU64,
    /// Total number of entries removed via `invalidate_kb` or `bump_kb_version`.
    pub total_invalidations: AtomicU64,
    /// Total number of entries removed via `evict_expired`.
    pub total_evictions: AtomicU64,
}

/// A point-in-time snapshot of [`CacheStats`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheStatsSnapshot {
    /// Total successful inserts since the cache was created.
    pub total_inserts: u64,
    /// Total hits since the cache was created.
    pub total_hits: u64,
    /// Total misses since the cache was created.
    pub total_misses: u64,
    /// Total invalidations since the cache was created.
    pub total_invalidations: u64,
    /// Total evictions since the cache was created.
    pub total_evictions: u64,
}

impl CacheStats {
    fn new() -> Self {
        Self::default()
    }

    /// Take a consistent snapshot.  Note: individual counters are read with
    /// `Relaxed`; the values are informational and do not require fencing.
    pub fn snapshot(&self) -> CacheStatsSnapshot {
        CacheStatsSnapshot {
            total_inserts: self.total_inserts.load(Ordering::Relaxed),
            total_hits: self.total_hits.load(Ordering::Relaxed),
            total_misses: self.total_misses.load(Ordering::Relaxed),
            total_invalidations: self.total_invalidations.load(Ordering::Relaxed),
            total_evictions: self.total_evictions.load(Ordering::Relaxed),
        }
    }
}

// ---------------------------------------------------------------------------
// VersionedInferenceCache
// ---------------------------------------------------------------------------

/// Default maximum number of entries in a [`VersionedInferenceCache`].
const DEFAULT_MAX_ENTRIES: usize = 10_000;

/// Concurrent, version-aware memoization cache for inference results.
///
/// Entries are keyed by `(goal_hash, kb_id, kb_version)`.  Calling
/// [`bump_kb_version`](Self::bump_kb_version) atomically increments the version
/// for a given KB **and** removes every cached entry that was derived from that
/// KB, regardless of their recorded version.
///
/// # Concurrency
///
/// Read operations (`get`, `current_kb_version`, `entry_count`, `stats`) acquire
/// a read lock; write operations acquire an exclusive write lock.
pub struct VersionedInferenceCache {
    entries: RwLock<HashMap<CacheKey, CacheEntry>>,
    kb_versions: RwLock<HashMap<String, u64>>,
    max_entries: usize,
    stats: CacheStats,
}

impl VersionedInferenceCache {
    /// Create a new cache with the given capacity limit.
    ///
    /// If `max_entries` is 0 the default (`10_000`) is used.
    pub fn new(max_entries: usize) -> Self {
        let max_entries = if max_entries == 0 {
            DEFAULT_MAX_ENTRIES
        } else {
            max_entries
        };
        Self {
            entries: RwLock::new(HashMap::new()),
            kb_versions: RwLock::new(HashMap::new()),
            max_entries,
            stats: CacheStats::new(),
        }
    }

    // ------------------------------------------------------------------
    // Insert
    // ------------------------------------------------------------------

    /// Insert a new entry.
    ///
    /// Returns [`CacheError::CapacityExceeded`] when the cache is full.
    /// Returns [`CacheError::StaleVersion`] when `key.kb_version` is lower
    /// than the currently tracked version for `key.kb_id`.
    ///
    /// If an entry with the same key already exists it is overwritten (the
    /// capacity check is skipped for replacements).
    pub fn insert(&self, key: CacheKey, entry: CacheEntry) -> Result<(), CacheError> {
        // Check for stale version before acquiring the write lock.
        let current_ver = self.current_kb_version(&key.kb_id);
        if current_ver > 0 && key.kb_version < current_ver {
            return Err(CacheError::StaleVersion {
                kb_id: key.kb_id.clone(),
                cache_version: key.kb_version,
                current_version: current_ver,
            });
        }

        let mut map = self
            .entries
            .write()
            .expect("versioned_cache: entries write lock poisoned");

        use std::collections::hash_map::Entry;

        // Snapshot length before the entry borrow.
        let current_len = map.len();

        match map.entry(key) {
            Entry::Occupied(mut occ) => {
                // Replacement — update in place; no capacity check needed.
                *occ.get_mut() = entry;
                self.stats.total_inserts.fetch_add(1, Ordering::Relaxed);
                Ok(())
            }
            Entry::Vacant(vac) => {
                // Capacity check before a new insertion.
                if current_len >= self.max_entries {
                    return Err(CacheError::CapacityExceeded {
                        current: current_len,
                        max: self.max_entries,
                    });
                }
                vac.insert(entry);
                self.stats.total_inserts.fetch_add(1, Ordering::Relaxed);
                Ok(())
            }
        }
    }

    // ------------------------------------------------------------------
    // Get
    // ------------------------------------------------------------------

    /// Look up a cached entry by key.
    ///
    /// Returns `None` if the entry is not found **or** if it has expired.
    /// On a hit the entry's `hit_count` is incremented and a clone is returned.
    pub fn get(&self, key: &CacheKey) -> Option<CacheEntry> {
        let now = Instant::now();

        let map = self
            .entries
            .read()
            .expect("versioned_cache: entries read lock poisoned");

        match map.get(key) {
            None => {
                self.stats.total_misses.fetch_add(1, Ordering::Relaxed);
                None
            }
            Some(entry) if entry.is_expired(now) => {
                self.stats.total_misses.fetch_add(1, Ordering::Relaxed);
                None
            }
            Some(entry) => {
                self.stats.total_hits.fetch_add(1, Ordering::Relaxed);
                let mut clone = entry.clone();
                clone.hit_count += 1;
                // We return the clone with the updated hit_count; we also need
                // to persist the increment in the map.  Drop the read lock first.
                drop(map);
                // Re-acquire write lock to bump the stored hit_count.
                if let Ok(mut wmap) = self.entries.write() {
                    if let Some(stored) = wmap.get_mut(key) {
                        stored.hit_count += 1;
                    }
                }
                Some(clone)
            }
        }
    }

    // ------------------------------------------------------------------
    // KB version management
    // ------------------------------------------------------------------

    /// Return the current version for the given KB, or `0` if unknown.
    pub fn current_kb_version(&self, kb_id: &str) -> u64 {
        let map = self
            .kb_versions
            .read()
            .expect("versioned_cache: kb_versions read lock poisoned");
        map.get(kb_id).copied().unwrap_or(0)
    }

    /// Increment the version counter for `kb_id` and atomically invalidate
    /// all cached entries that were derived from that KB.
    ///
    /// Returns the **new** version number.
    pub fn bump_kb_version(&self, kb_id: &str) -> u64 {
        // Bump the version counter.
        let new_version = {
            let mut ver_map = self
                .kb_versions
                .write()
                .expect("versioned_cache: kb_versions write lock poisoned");
            let v = ver_map.entry(kb_id.to_owned()).or_insert(0);
            *v += 1;
            *v
        };

        // Invalidate all entries for this KB (any version).
        self.invalidate_kb(kb_id);

        new_version
    }

    /// Remove **all** cached entries whose `kb_id` matches, regardless of
    /// their recorded version.
    ///
    /// This is called internally by [`bump_kb_version`](Self::bump_kb_version)
    /// but can also be called directly (e.g., when a KB is deleted).
    pub fn invalidate_kb(&self, kb_id: &str) {
        let mut map = self
            .entries
            .write()
            .expect("versioned_cache: entries write lock poisoned");

        let before = map.len();
        map.retain(|k, _| k.kb_id != kb_id);
        let removed = before - map.len();

        self.stats
            .total_invalidations
            .fetch_add(removed as u64, Ordering::Relaxed);
    }

    // ------------------------------------------------------------------
    // Eviction
    // ------------------------------------------------------------------

    /// Remove all expired entries from the cache.
    ///
    /// Returns the number of entries removed.
    pub fn evict_expired(&self) -> usize {
        let now = Instant::now();
        let mut map = self
            .entries
            .write()
            .expect("versioned_cache: entries write lock poisoned");

        let before = map.len();
        map.retain(|_, entry| !entry.is_expired(now));
        let removed = before - map.len();

        self.stats
            .total_evictions
            .fetch_add(removed as u64, Ordering::Relaxed);
        removed
    }

    // ------------------------------------------------------------------
    // Introspection
    // ------------------------------------------------------------------

    /// Current number of entries in the cache (including possibly-expired ones).
    pub fn entry_count(&self) -> usize {
        self.entries
            .read()
            .expect("versioned_cache: entries read lock poisoned")
            .len()
    }

    /// Snapshot of the current statistics counters.
    pub fn stats(&self) -> CacheStatsSnapshot {
        self.stats.snapshot()
    }
}

impl Default for VersionedInferenceCache {
    fn default() -> Self {
        Self::new(DEFAULT_MAX_ENTRIES)
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

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    fn make_key(goal: &str, kb_id: &str, kb_version: u64) -> CacheKey {
        CacheKey::new(goal, kb_id, kb_version)
    }

    fn make_entry(bindings: Vec<(&str, &str)>, is_final: bool) -> CacheEntry {
        let map: HashMap<String, String> = bindings
            .into_iter()
            .map(|(k, v)| (k.to_owned(), v.to_owned()))
            .collect();
        CacheEntry::new(vec![map], is_final)
    }

    fn empty_entry() -> CacheEntry {
        CacheEntry::new(vec![], true)
    }

    fn expired_entry() -> CacheEntry {
        CacheEntry::with_ttl(vec![], true, Duration::from_nanos(1))
    }

    // ------------------------------------------------------------------
    // 1. Insert and get returns entry
    // ------------------------------------------------------------------

    #[test]
    fn test_insert_and_get_returns_entry() {
        let cache = VersionedInferenceCache::new(100);
        let key = make_key("parent(X, bob)", "kb1", 1);
        let entry = make_entry(vec![("X", "alice")], true);

        cache.insert(key.clone(), entry).expect("insert failed");
        let result = cache.get(&key).expect("expected Some");

        assert_eq!(result.bindings.len(), 1);
        assert_eq!(
            result.bindings[0].get("X").map(String::as_str),
            Some("alice")
        );
        assert!(result.is_final);
    }

    // ------------------------------------------------------------------
    // 2. Get expired entry returns None
    // ------------------------------------------------------------------

    #[test]
    fn test_get_expired_entry_returns_none() {
        let cache = VersionedInferenceCache::new(100);
        let key = make_key("ancestor(X, eve)", "kb1", 1);
        let entry = expired_entry();

        cache.insert(key.clone(), entry).expect("insert failed");
        // Brief sleep to ensure the 1ns TTL has elapsed.
        thread::sleep(Duration::from_millis(1));

        let result = cache.get(&key);
        assert!(result.is_none(), "expected None for expired entry");
    }

    // ------------------------------------------------------------------
    // 3. invalidate_kb removes all entries for that KB
    // ------------------------------------------------------------------

    #[test]
    fn test_invalidate_kb_removes_all_entries_for_kb() {
        let cache = VersionedInferenceCache::new(100);

        cache
            .insert(make_key("g1", "kb_a", 1), empty_entry())
            .expect("test: should succeed");
        cache
            .insert(make_key("g2", "kb_a", 1), empty_entry())
            .expect("test: should succeed");
        cache
            .insert(make_key("g3", "kb_b", 1), empty_entry())
            .expect("test: should succeed");

        assert_eq!(cache.entry_count(), 3);

        cache.invalidate_kb("kb_a");

        assert_eq!(cache.entry_count(), 1);
        assert!(cache.get(&make_key("g1", "kb_a", 1)).is_none());
        assert!(cache.get(&make_key("g2", "kb_a", 1)).is_none());
        assert!(cache.get(&make_key("g3", "kb_b", 1)).is_some());
    }

    // ------------------------------------------------------------------
    // 4. bump_kb_version increments version and invalidates
    // ------------------------------------------------------------------

    #[test]
    fn test_bump_kb_version_increments_and_invalidates() {
        let cache = VersionedInferenceCache::new(100);

        let key_v1 = make_key("fact(x)", "kb_main", 1);
        cache
            .insert(key_v1.clone(), empty_entry())
            .expect("test: should succeed");
        assert_eq!(cache.entry_count(), 1);

        let new_ver = cache.bump_kb_version("kb_main");
        assert_eq!(new_ver, 1);

        // All kb_main entries gone.
        assert_eq!(cache.entry_count(), 0);
        assert!(cache.get(&key_v1).is_none());
    }

    // ------------------------------------------------------------------
    // 5. bump_kb_version starts at 1 from 0
    // ------------------------------------------------------------------

    #[test]
    fn test_bump_kb_version_from_zero() {
        let cache = VersionedInferenceCache::new(100);

        assert_eq!(cache.current_kb_version("new_kb"), 0);
        let v1 = cache.bump_kb_version("new_kb");
        assert_eq!(v1, 1);
        let v2 = cache.bump_kb_version("new_kb");
        assert_eq!(v2, 2);
    }

    // ------------------------------------------------------------------
    // 6. current_kb_version returns 0 for unknown KB
    // ------------------------------------------------------------------

    #[test]
    fn test_current_kb_version_returns_zero_for_unknown() {
        let cache = VersionedInferenceCache::new(100);
        assert_eq!(cache.current_kb_version("totally_unknown_kb"), 0);
    }

    // ------------------------------------------------------------------
    // 7. Capacity limit enforcement
    // ------------------------------------------------------------------

    #[test]
    fn test_capacity_limit_enforced() {
        let cache = VersionedInferenceCache::new(3);

        cache
            .insert(make_key("g1", "kb", 1), empty_entry())
            .expect("test: should succeed");
        cache
            .insert(make_key("g2", "kb", 1), empty_entry())
            .expect("test: should succeed");
        cache
            .insert(make_key("g3", "kb", 1), empty_entry())
            .expect("test: should succeed");

        let err = cache
            .insert(make_key("g4", "kb", 1), empty_entry())
            .unwrap_err();

        match err {
            CacheError::CapacityExceeded { current, max } => {
                assert_eq!(current, 3);
                assert_eq!(max, 3);
            }
            other => panic!("unexpected error: {:?}", other),
        }
    }

    // ------------------------------------------------------------------
    // 8. evict_expired removes stale, keeps fresh
    // ------------------------------------------------------------------

    #[test]
    fn test_evict_expired_removes_stale_keeps_fresh() {
        let cache = VersionedInferenceCache::new(100);

        // Insert 2 entries with instant expiry and 2 fresh ones.
        cache
            .insert(make_key("stale1", "kb", 1), expired_entry())
            .expect("test: should succeed");
        cache
            .insert(make_key("stale2", "kb", 1), expired_entry())
            .expect("test: should succeed");
        cache
            .insert(make_key("fresh1", "kb", 1), empty_entry())
            .expect("test: should succeed");
        cache
            .insert(make_key("fresh2", "kb", 1), empty_entry())
            .expect("test: should succeed");

        thread::sleep(Duration::from_millis(1));
        let removed = cache.evict_expired();

        assert_eq!(removed, 2);
        assert_eq!(cache.entry_count(), 2);
    }

    // ------------------------------------------------------------------
    // 9. Hit count increments on repeated get
    // ------------------------------------------------------------------

    #[test]
    fn test_hit_count_increments_on_repeated_get() {
        let cache = VersionedInferenceCache::new(100);
        let key = make_key("rule(Y)", "kb", 1);
        cache
            .insert(key.clone(), empty_entry())
            .expect("test: should succeed");

        let r1 = cache.get(&key).expect("test: should succeed");
        assert_eq!(r1.hit_count, 1);

        let r2 = cache.get(&key).expect("test: should succeed");
        assert_eq!(r2.hit_count, 2);

        let r3 = cache.get(&key).expect("test: should succeed");
        assert_eq!(r3.hit_count, 3);
    }

    // ------------------------------------------------------------------
    // 10. Stats: inserts counted
    // ------------------------------------------------------------------

    #[test]
    fn test_stats_inserts_counted() {
        let cache = VersionedInferenceCache::new(100);
        cache
            .insert(make_key("g1", "kb", 1), empty_entry())
            .expect("test: should succeed");
        cache
            .insert(make_key("g2", "kb", 1), empty_entry())
            .expect("test: should succeed");

        assert_eq!(cache.stats().total_inserts, 2);
    }

    // ------------------------------------------------------------------
    // 11. Stats: hits and misses counted
    // ------------------------------------------------------------------

    #[test]
    fn test_stats_hits_and_misses() {
        let cache = VersionedInferenceCache::new(100);
        let key = make_key("p(a)", "kb", 1);
        cache
            .insert(key.clone(), empty_entry())
            .expect("test: should succeed");

        cache.get(&key); // hit
        cache.get(&make_key("missing", "kb", 1)); // miss

        let snap = cache.stats();
        assert_eq!(snap.total_hits, 1);
        assert_eq!(snap.total_misses, 1);
    }

    // ------------------------------------------------------------------
    // 12. Stats: invalidations counted
    // ------------------------------------------------------------------

    #[test]
    fn test_stats_invalidations_counted() {
        let cache = VersionedInferenceCache::new(100);
        cache
            .insert(make_key("g1", "kb", 1), empty_entry())
            .expect("test: should succeed");
        cache
            .insert(make_key("g2", "kb", 1), empty_entry())
            .expect("test: should succeed");

        cache.invalidate_kb("kb");

        assert_eq!(cache.stats().total_invalidations, 2);
    }

    // ------------------------------------------------------------------
    // 13. Stats: evictions counted
    // ------------------------------------------------------------------

    #[test]
    fn test_stats_evictions_counted() {
        let cache = VersionedInferenceCache::new(100);
        cache
            .insert(make_key("old1", "kb", 1), expired_entry())
            .expect("test: should succeed");
        cache
            .insert(make_key("old2", "kb", 1), expired_entry())
            .expect("test: should succeed");
        cache
            .insert(make_key("old3", "kb", 1), expired_entry())
            .expect("test: should succeed");

        thread::sleep(Duration::from_millis(1));
        let removed = cache.evict_expired();

        assert_eq!(removed, 3);
        assert_eq!(cache.stats().total_evictions, 3);
    }

    // ------------------------------------------------------------------
    // 14. Same goal + kb_id, different kb_version => distinct keys
    // ------------------------------------------------------------------

    #[test]
    fn test_different_kb_versions_are_distinct_keys() {
        let cache = VersionedInferenceCache::new(100);

        let key_v1 = make_key("same_goal", "kb", 1);
        let key_v2 = make_key("same_goal", "kb", 2);

        assert_ne!(key_v1, key_v2, "keys with different versions must differ");

        let entry_v1 = make_entry(vec![("X", "alice")], true);
        let entry_v2 = make_entry(vec![("X", "bob")], true);

        // Insert both independently (bypass version-check by not bumping).
        let mut entries_map = cache.entries.write().expect("entries write lock");
        entries_map.insert(key_v1.clone(), entry_v1);
        entries_map.insert(key_v2.clone(), entry_v2);
        drop(entries_map);

        assert_eq!(cache.entry_count(), 2);

        let r1 = cache.get(&key_v1).expect("test: should succeed");
        let r2 = cache.get(&key_v2).expect("test: should succeed");

        assert_eq!(r1.bindings[0].get("X").map(String::as_str), Some("alice"));
        assert_eq!(r2.bindings[0].get("X").map(String::as_str), Some("bob"));
    }

    // ------------------------------------------------------------------
    // 15. entry_count decreases after eviction
    // ------------------------------------------------------------------

    #[test]
    fn test_entry_count_decreases_after_eviction() {
        let cache = VersionedInferenceCache::new(100);

        for i in 0u64..5 {
            let entry = CacheEntry::with_ttl(vec![], true, Duration::from_nanos(1));
            let key = CacheKey {
                goal_hash: i,
                kb_id: "kb".to_owned(),
                kb_version: 1,
            };
            cache.insert(key, entry).expect("test: should succeed");
        }

        assert_eq!(cache.entry_count(), 5);

        thread::sleep(Duration::from_millis(1));
        let evicted = cache.evict_expired();

        assert_eq!(evicted, 5);
        assert_eq!(cache.entry_count(), 0);
    }

    // ------------------------------------------------------------------
    // 16. StaleVersion error when inserting with outdated kb_version
    // ------------------------------------------------------------------

    #[test]
    fn test_stale_version_error_on_insert() {
        let cache = VersionedInferenceCache::new(100);

        // Bump to version 3.
        cache.bump_kb_version("kb");
        cache.bump_kb_version("kb");
        cache.bump_kb_version("kb");

        assert_eq!(cache.current_kb_version("kb"), 3);

        // Attempt to insert with an old version.
        let stale_key = make_key("goal", "kb", 1);
        let err = cache.insert(stale_key, empty_entry()).unwrap_err();

        match err {
            CacheError::StaleVersion {
                kb_id,
                cache_version,
                current_version,
            } => {
                assert_eq!(kb_id, "kb");
                assert_eq!(cache_version, 1);
                assert_eq!(current_version, 3);
            }
            other => panic!("unexpected error: {:?}", other),
        }
    }

    // ------------------------------------------------------------------
    // 17. bump_kb_version invalidates entries across multiple versions
    // ------------------------------------------------------------------

    #[test]
    fn test_bump_invalidates_across_multiple_versions() {
        let cache = VersionedInferenceCache::new(100);

        // Manually insert entries with different versions (bypass stale check).
        {
            let mut map = cache.entries.write().expect("write lock");
            for v in 1u64..=4 {
                let k = CacheKey {
                    goal_hash: v,
                    kb_id: "kb_x".to_owned(),
                    kb_version: v,
                };
                map.insert(k, CacheEntry::new(vec![], true));
            }
            // One entry from a different KB — must survive.
            let other = CacheKey {
                goal_hash: 99,
                kb_id: "other_kb".to_owned(),
                kb_version: 1,
            };
            map.insert(other, CacheEntry::new(vec![], true));
        }

        assert_eq!(cache.entry_count(), 5);

        cache.invalidate_kb("kb_x");

        assert_eq!(cache.entry_count(), 1);
        assert_eq!(cache.stats().total_invalidations, 4);
    }

    // ------------------------------------------------------------------
    // 18. Default cache has max_entries = 10_000
    // ------------------------------------------------------------------

    #[test]
    fn test_default_max_entries() {
        let cache = VersionedInferenceCache::default();
        // Insert one entry — should succeed even in default config.
        cache
            .insert(make_key("test_goal", "default_kb", 0), empty_entry())
            .expect("test: should succeed");
        assert_eq!(cache.entry_count(), 1);
    }
}
