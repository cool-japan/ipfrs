//! Semantic query cache with TTL expiry, LRU eviction, and hit/miss statistics.
//!
//! Caches semantic search results keyed by query embedding fingerprint (FNV-1a).
//! Entries expire after a configurable number of ticks and are evicted in LRU
//! order when the cache reaches its capacity limit.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// FNV-1a fingerprint
// ---------------------------------------------------------------------------

/// Compute an FNV-1a 64-bit fingerprint over the raw bytes of a `&[f32]` slice.
///
/// Each `f32` is decomposed into its 4 little-endian bytes before hashing, so
/// the fingerprint is independent of the host's native endianness.
fn embedding_fingerprint(embedding: &[f32]) -> u64 {
    const OFFSET_BASIS: u64 = 14_695_981_039_346_656_037;
    const PRIME: u64 = 1_099_511_628_211;

    let mut hash = OFFSET_BASIS;
    for &v in embedding {
        for byte in v.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(PRIME);
        }
    }
    hash
}

// ---------------------------------------------------------------------------
// CachedQueryResult
// ---------------------------------------------------------------------------

/// A single cached search result entry, keyed by query embedding fingerprint.
#[derive(Debug, Clone)]
pub struct CachedQueryResult {
    /// FNV-1a fingerprint of the query embedding bytes (little-endian f32).
    pub query_fingerprint: u64,
    /// Ordered list of document IDs returned by the search.
    pub result_doc_ids: Vec<u64>,
    /// Tick at which this entry was inserted or last refreshed.
    pub cached_at_tick: u64,
    /// Number of ticks this entry remains valid.
    pub ttl_ticks: u64,
    /// Number of times this entry has been returned as a cache hit.
    pub hit_count: u64,
}

impl CachedQueryResult {
    /// Returns `true` when `current_tick` has advanced past the entry's
    /// expiry boundary (`cached_at_tick + ttl_ticks`).
    #[inline]
    pub fn is_expired(&self, current_tick: u64) -> bool {
        current_tick > self.cached_at_tick.saturating_add(self.ttl_ticks)
    }
}

// ---------------------------------------------------------------------------
// QueryCacheConfig
// ---------------------------------------------------------------------------

/// Configuration for a [`SemanticQueryCache`] instance.
#[derive(Debug, Clone)]
pub struct QueryCacheConfig {
    /// Maximum number of entries that may be held in the cache simultaneously.
    /// When this limit is exceeded the least-recently-used entry is evicted.
    pub max_entries: usize,
    /// Default TTL (in ticks) applied to new entries when no override is given.
    pub default_ttl_ticks: u64,
}

impl Default for QueryCacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 512,
            default_ttl_ticks: 300,
        }
    }
}

// ---------------------------------------------------------------------------
// QueryCacheStats  (named QueryCacheStats to avoid collision with router::CacheStats)
// ---------------------------------------------------------------------------

/// Accumulated statistics for a [`SemanticQueryCache`] instance.
///
/// Exported as `QueryCacheStats` to avoid a name collision with the
/// `CacheStats` type that already exists in `router`.
#[derive(Debug, Clone, Default)]
pub struct QueryCacheStats {
    /// Number of entries currently stored in the cache.
    pub total_entries: usize,
    /// Total number of successful cache lookups.
    pub hits: u64,
    /// Total number of unsuccessful cache lookups (including expired entries).
    pub misses: u64,
    /// Total number of LRU evictions (capacity overflow).
    pub evictions: u64,
    /// Total number of TTL expirations detected during lookups or explicit
    /// calls to [`SemanticQueryCache::evict_expired`].
    pub expirations: u64,
}

impl QueryCacheStats {
    /// Returns the fraction of lookups that resulted in a cache hit.
    ///
    /// Returns `0.0` when no lookups have been performed yet.
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
// SemanticQueryCache
// ---------------------------------------------------------------------------

/// An in-memory cache for semantic search results with:
///
/// * **FNV-1a fingerprinting** — O(d) key computation over the query embedding.
/// * **TTL expiry** — stale entries are rejected on lookup and purged by
///   [`Self::evict_expired`].
/// * **LRU eviction** — when the cache is full the least-recently-used entry
///   (front of `access_order`) is discarded.
/// * **Hit/miss statistics** — exposed through [`Self::stats`].
pub struct SemanticQueryCache {
    /// Primary storage keyed by embedding fingerprint.
    entries: HashMap<u64, CachedQueryResult>,
    /// Ordered list of fingerprints from least- to most-recently used.
    ///
    /// The front of the vector is the LRU candidate; the back is the MRU entry.
    access_order: Vec<u64>,
    /// Immutable configuration provided at construction time.
    config: QueryCacheConfig,
    /// Running statistics updated on every operation.
    stats: QueryCacheStats,
}

impl SemanticQueryCache {
    /// Create a new, empty cache with the given [`QueryCacheConfig`].
    pub fn new(config: QueryCacheConfig) -> Self {
        Self {
            entries: HashMap::new(),
            access_order: Vec::new(),
            config,
            stats: QueryCacheStats::default(),
        }
    }

    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------

    /// Look up cached results for `embedding` at the given `current_tick`.
    ///
    /// # Return value
    ///
    /// * `Some(doc_ids)` — cache hit; `doc_ids` is a clone of the stored
    ///   result vector.
    /// * `None` — cache miss (unknown key or expired entry).
    pub fn get(&mut self, embedding: &[f32], current_tick: u64) -> Option<Vec<u64>> {
        let fp = embedding_fingerprint(embedding);

        match self.entries.get(&fp) {
            None => {
                self.stats.misses += 1;
                None
            }
            Some(entry) if entry.is_expired(current_tick) => {
                // Remove the stale entry.
                self.entries.remove(&fp);
                Self::remove_from_order(&mut self.access_order, fp);
                self.stats.expirations += 1;
                self.stats.misses += 1;
                self.stats.total_entries = self.entries.len();
                None
            }
            Some(_) => {
                // Valid hit — update hit_count and LRU order.
                let entry = self
                    .entries
                    .get_mut(&fp)
                    .expect("key confirmed present above");
                entry.hit_count += 1;
                let result = entry.result_doc_ids.clone();
                Self::move_to_back(&mut self.access_order, fp);
                self.stats.hits += 1;
                Some(result)
            }
        }
    }

    /// Store `result_doc_ids` for `embedding` in the cache.
    ///
    /// If an entry already exists for this embedding it is refreshed in-place
    /// (TTL reset, results replaced, hit_count zeroed).  Otherwise a new entry
    /// is created, potentially evicting the LRU entry when the cache is full.
    ///
    /// Pass `ttl_override` to use a TTL other than
    /// [`QueryCacheConfig::default_ttl_ticks`] for this specific entry.
    pub fn put(
        &mut self,
        embedding: &[f32],
        result_doc_ids: Vec<u64>,
        current_tick: u64,
        ttl_override: Option<u64>,
    ) {
        let fp = embedding_fingerprint(embedding);
        let ttl = ttl_override.unwrap_or(self.config.default_ttl_ticks);

        if let Some(entry) = self.entries.get_mut(&fp) {
            // Refresh existing entry.
            entry.result_doc_ids = result_doc_ids;
            entry.cached_at_tick = current_tick;
            entry.ttl_ticks = ttl;
            entry.hit_count = 0;
            Self::move_to_back(&mut self.access_order, fp);
        } else {
            // Evict LRU if at capacity.
            if self.entries.len() >= self.config.max_entries {
                if let Some(lru_fp) = Self::pop_front(&mut self.access_order) {
                    self.entries.remove(&lru_fp);
                    self.stats.evictions += 1;
                }
            }

            // Insert the new entry.
            self.entries.insert(
                fp,
                CachedQueryResult {
                    query_fingerprint: fp,
                    result_doc_ids,
                    cached_at_tick: current_tick,
                    ttl_ticks: ttl,
                    hit_count: 0,
                },
            );
            self.access_order.push(fp);
        }

        self.stats.total_entries = self.entries.len();
    }

    /// Remove the cache entry for `embedding`.
    ///
    /// Returns `true` if an entry existed and was removed, `false` otherwise.
    pub fn invalidate(&mut self, embedding: &[f32]) -> bool {
        let fp = embedding_fingerprint(embedding);
        let existed = self.entries.remove(&fp).is_some();
        if existed {
            Self::remove_from_order(&mut self.access_order, fp);
            self.stats.total_entries = self.entries.len();
        }
        existed
    }

    /// Remove all entries whose TTL has elapsed at `current_tick`.
    ///
    /// Returns the number of entries removed.  Each removed entry increments
    /// [`QueryCacheStats::expirations`].
    pub fn evict_expired(&mut self, current_tick: u64) -> usize {
        let expired: Vec<u64> = self
            .entries
            .iter()
            .filter(|(_, e)| e.is_expired(current_tick))
            .map(|(k, _)| *k)
            .collect();

        let count = expired.len();
        for fp in expired {
            self.entries.remove(&fp);
            Self::remove_from_order(&mut self.access_order, fp);
        }

        self.stats.expirations += count as u64;
        self.stats.total_entries = self.entries.len();
        count
    }

    /// Return a shared reference to the accumulated cache statistics.
    pub fn stats(&self) -> &QueryCacheStats {
        &self.stats
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Move `fp` to the back of `order` (most-recently used position).
    ///
    /// If `fp` is not already present it is appended without removing anything.
    fn move_to_back(order: &mut Vec<u64>, fp: u64) {
        if let Some(pos) = order.iter().position(|&x| x == fp) {
            order.remove(pos);
        }
        order.push(fp);
    }

    /// Remove the first occurrence of `fp` from `order`.
    fn remove_from_order(order: &mut Vec<u64>, fp: u64) {
        if let Some(pos) = order.iter().position(|&x| x == fp) {
            order.remove(pos);
        }
    }

    /// Remove and return the front element of `order` (least-recently used).
    fn pop_front(order: &mut Vec<u64>) -> Option<u64> {
        if order.is_empty() {
            None
        } else {
            Some(order.remove(0))
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn default_cache() -> SemanticQueryCache {
        SemanticQueryCache::new(QueryCacheConfig::default())
    }

    fn cache_with_max(max_entries: usize) -> SemanticQueryCache {
        SemanticQueryCache::new(QueryCacheConfig {
            max_entries,
            ..QueryCacheConfig::default()
        })
    }

    fn cache_with_ttl(ttl: u64) -> SemanticQueryCache {
        SemanticQueryCache::new(QueryCacheConfig {
            default_ttl_ticks: ttl,
            ..QueryCacheConfig::default()
        })
    }

    fn vec_a() -> Vec<f32> {
        vec![1.0_f32, 0.0, 0.0]
    }

    fn vec_b() -> Vec<f32> {
        vec![0.0_f32, 1.0, 0.0]
    }

    fn vec_c() -> Vec<f32> {
        vec![0.0_f32, 0.0, 1.0]
    }

    // -----------------------------------------------------------------------
    // 1. get returns None for unknown embedding
    // -----------------------------------------------------------------------
    #[test]
    fn test_get_returns_none_for_unknown_embedding() {
        let mut cache = default_cache();
        assert!(cache.get(&vec_a(), 0).is_none());
    }

    // -----------------------------------------------------------------------
    // 2. get increments misses for unknown embedding
    // -----------------------------------------------------------------------
    #[test]
    fn test_get_increments_misses_for_unknown_embedding() {
        let mut cache = default_cache();
        cache.get(&vec_a(), 0);
        cache.get(&vec_b(), 0);
        assert_eq!(cache.stats().misses, 2);
        assert_eq!(cache.stats().hits, 0);
    }

    // -----------------------------------------------------------------------
    // 3. get returns None for expired entry
    // -----------------------------------------------------------------------
    #[test]
    fn test_get_returns_none_for_expired_entry() {
        let mut cache = cache_with_ttl(10);
        cache.put(&vec_a(), vec![1, 2, 3], 0, None);
        // Expiry boundary: 0 + 10 = 10; tick 11 > 10 → expired
        assert!(cache.get(&vec_a(), 11).is_none());
    }

    // -----------------------------------------------------------------------
    // 4. expired entry is removed from storage
    // -----------------------------------------------------------------------
    #[test]
    fn test_expired_entry_is_removed_from_storage() {
        let mut cache = cache_with_ttl(5);
        cache.put(&vec_a(), vec![42], 0, None);
        cache.get(&vec_a(), 10);
        assert_eq!(cache.entries.len(), 0);
        assert_eq!(cache.access_order.len(), 0);
    }

    // -----------------------------------------------------------------------
    // 5. expired lookup increments expirations and misses
    // -----------------------------------------------------------------------
    #[test]
    fn test_expired_lookup_increments_expirations_and_misses() {
        let mut cache = cache_with_ttl(5);
        cache.put(&vec_a(), vec![1], 0, None);
        cache.get(&vec_a(), 100);
        assert_eq!(cache.stats().expirations, 1);
        assert_eq!(cache.stats().misses, 1);
        assert_eq!(cache.stats().hits, 0);
    }

    // -----------------------------------------------------------------------
    // 6. get returns cached results on valid hit
    // -----------------------------------------------------------------------
    #[test]
    fn test_get_returns_cached_results_on_valid_hit() {
        let mut cache = default_cache();
        let ids = vec![10_u64, 20, 30];
        cache.put(&vec_a(), ids.clone(), 0, None);
        let result = cache.get(&vec_a(), 0);
        assert_eq!(result, Some(ids));
    }

    // -----------------------------------------------------------------------
    // 7. get increments hit_count on the entry
    // -----------------------------------------------------------------------
    #[test]
    fn test_get_increments_hit_count_on_entry() {
        let mut cache = default_cache();
        cache.put(&vec_a(), vec![1], 0, None);
        cache.get(&vec_a(), 0);
        cache.get(&vec_a(), 0);
        let fp = embedding_fingerprint(&vec_a());
        let hit_count = cache.entries[&fp].hit_count;
        assert_eq!(hit_count, 2);
    }

    // -----------------------------------------------------------------------
    // 8. get increments hits stat on valid hit
    // -----------------------------------------------------------------------
    #[test]
    fn test_get_increments_hits_stat() {
        let mut cache = default_cache();
        cache.put(&vec_a(), vec![1], 0, None);
        cache.get(&vec_a(), 0);
        assert_eq!(cache.stats().hits, 1);
    }

    // -----------------------------------------------------------------------
    // 9. put inserts a new entry
    // -----------------------------------------------------------------------
    #[test]
    fn test_put_inserts_new_entry() {
        let mut cache = default_cache();
        cache.put(&vec_a(), vec![5, 6], 0, None);
        assert_eq!(cache.entries.len(), 1);
        assert_eq!(cache.stats().total_entries, 1);
    }

    // -----------------------------------------------------------------------
    // 10. put updates existing entry (results replaced, hit_count reset)
    // -----------------------------------------------------------------------
    #[test]
    fn test_put_updates_existing_entry() {
        let mut cache = default_cache();
        cache.put(&vec_a(), vec![1, 2], 0, None);
        // Simulate a hit to raise hit_count.
        cache.get(&vec_a(), 0);
        // Now overwrite.
        cache.put(&vec_a(), vec![99], 100, None);
        let result = cache.get(&vec_a(), 100);
        assert_eq!(result, Some(vec![99_u64]));
        // hit_count should have been reset to 0 then incremented once by the get above.
        let fp = embedding_fingerprint(&vec_a());
        assert_eq!(cache.entries[&fp].hit_count, 1);
        // Still only one logical entry.
        assert_eq!(cache.entries.len(), 1);
    }

    // -----------------------------------------------------------------------
    // 11. LRU eviction when max_entries exceeded
    // -----------------------------------------------------------------------
    #[test]
    fn test_lru_eviction_when_max_entries_exceeded() {
        let mut cache = cache_with_max(2);

        // Insert A then B (A is LRU).
        cache.put(&vec_a(), vec![1], 0, None);
        cache.put(&vec_b(), vec![2], 0, None);

        // Access A to make it MRU; B becomes LRU.
        cache.get(&vec_a(), 0);

        // Inserting C should evict B (the LRU).
        cache.put(&vec_c(), vec![3], 0, None);

        assert_eq!(cache.entries.len(), 2);
        assert!(cache.entries.contains_key(&embedding_fingerprint(&vec_a())));
        assert!(!cache.entries.contains_key(&embedding_fingerprint(&vec_b())));
        assert!(cache.entries.contains_key(&embedding_fingerprint(&vec_c())));
        assert_eq!(cache.stats().evictions, 1);
    }

    // -----------------------------------------------------------------------
    // 12. LRU eviction increments evictions stat
    // -----------------------------------------------------------------------
    #[test]
    fn test_lru_eviction_increments_evictions_stat() {
        let mut cache = cache_with_max(1);
        cache.put(&vec_a(), vec![1], 0, None);
        cache.put(&vec_b(), vec![2], 0, None);
        assert_eq!(cache.stats().evictions, 1);
    }

    // -----------------------------------------------------------------------
    // 13. evict_expired removes expired entries
    // -----------------------------------------------------------------------
    #[test]
    fn test_evict_expired_removes_expired_entries() {
        let mut cache = cache_with_ttl(10);
        cache.put(&vec_a(), vec![1], 0, None); // expires at tick 11
        cache.put(&vec_b(), vec![2], 100, None); // expires at tick 111
        let removed = cache.evict_expired(20);
        assert_eq!(removed, 1);
        assert!(!cache.entries.contains_key(&embedding_fingerprint(&vec_a())));
        assert!(cache.entries.contains_key(&embedding_fingerprint(&vec_b())));
    }

    // -----------------------------------------------------------------------
    // 14. evict_expired updates total_entries
    // -----------------------------------------------------------------------
    #[test]
    fn test_evict_expired_updates_total_entries() {
        let mut cache = cache_with_ttl(5);
        cache.put(&vec_a(), vec![1], 0, None);
        cache.put(&vec_b(), vec![2], 0, None);
        cache.evict_expired(10);
        assert_eq!(cache.stats().total_entries, 0);
    }

    // -----------------------------------------------------------------------
    // 15. evict_expired increments expirations stat
    // -----------------------------------------------------------------------
    #[test]
    fn test_evict_expired_increments_expirations_stat() {
        let mut cache = cache_with_ttl(5);
        cache.put(&vec_a(), vec![1], 0, None);
        cache.put(&vec_b(), vec![2], 0, None);
        cache.evict_expired(10);
        assert_eq!(cache.stats().expirations, 2);
    }

    // -----------------------------------------------------------------------
    // 16. evict_expired returns zero when nothing expired
    // -----------------------------------------------------------------------
    #[test]
    fn test_evict_expired_returns_zero_when_nothing_expired() {
        let mut cache = cache_with_ttl(100);
        cache.put(&vec_a(), vec![1], 0, None);
        let removed = cache.evict_expired(5);
        assert_eq!(removed, 0);
    }

    // -----------------------------------------------------------------------
    // 17. invalidate removes specific entry
    // -----------------------------------------------------------------------
    #[test]
    fn test_invalidate_removes_specific_entry() {
        let mut cache = default_cache();
        cache.put(&vec_a(), vec![1], 0, None);
        cache.put(&vec_b(), vec![2], 0, None);
        let existed = cache.invalidate(&vec_a());
        assert!(existed);
        assert!(!cache.entries.contains_key(&embedding_fingerprint(&vec_a())));
        assert!(cache.entries.contains_key(&embedding_fingerprint(&vec_b())));
    }

    // -----------------------------------------------------------------------
    // 18. invalidate returns false for unknown entry
    // -----------------------------------------------------------------------
    #[test]
    fn test_invalidate_returns_false_for_unknown_entry() {
        let mut cache = default_cache();
        assert!(!cache.invalidate(&vec_a()));
    }

    // -----------------------------------------------------------------------
    // 19. invalidate updates access_order
    // -----------------------------------------------------------------------
    #[test]
    fn test_invalidate_updates_access_order() {
        let mut cache = default_cache();
        cache.put(&vec_a(), vec![1], 0, None);
        cache.invalidate(&vec_a());
        assert!(cache.access_order.is_empty());
    }

    // -----------------------------------------------------------------------
    // 20. hit_rate computation — all hits
    // -----------------------------------------------------------------------
    #[test]
    fn test_hit_rate_all_hits() {
        let mut cache = default_cache();
        cache.put(&vec_a(), vec![1], 0, None);
        cache.get(&vec_a(), 0);
        cache.get(&vec_a(), 0);
        assert!((cache.stats().hit_rate() - 1.0).abs() < f64::EPSILON);
    }

    // -----------------------------------------------------------------------
    // 21. hit_rate computation — all misses
    // -----------------------------------------------------------------------
    #[test]
    fn test_hit_rate_all_misses() {
        let mut cache = default_cache();
        cache.get(&vec_a(), 0);
        cache.get(&vec_b(), 0);
        assert!((cache.stats().hit_rate() - 0.0).abs() < f64::EPSILON);
    }

    // -----------------------------------------------------------------------
    // 22. hit_rate computation — no queries
    // -----------------------------------------------------------------------
    #[test]
    fn test_hit_rate_no_queries_returns_zero() {
        let cache = default_cache();
        assert_eq!(cache.stats().hit_rate(), 0.0);
    }

    // -----------------------------------------------------------------------
    // 23. hit_rate computation — mixed hits and misses
    // -----------------------------------------------------------------------
    #[test]
    fn test_hit_rate_mixed() {
        let mut cache = default_cache();
        cache.put(&vec_a(), vec![1], 0, None);
        cache.get(&vec_a(), 0); // hit
        cache.get(&vec_a(), 0); // hit
        cache.get(&vec_b(), 0); // miss
                                // 2 hits / 3 total = 0.666…
        let rate = cache.stats().hit_rate();
        assert!((rate - 2.0 / 3.0).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // 24. ttl_override in put is respected
    // -----------------------------------------------------------------------
    #[test]
    fn test_ttl_override_in_put_is_respected() {
        // Default TTL is 300 ticks; override to 5.
        let mut cache = default_cache();
        cache.put(&vec_a(), vec![7], 0, Some(5));
        // Should be present at tick 5 (boundary: 0 + 5 = 5, current 5 is NOT > 5).
        assert!(cache.get(&vec_a(), 5).is_some());
        // Should be expired at tick 6.
        cache.put(&vec_a(), vec![7], 0, Some(5)); // re-insert after expiry removed it
        assert!(cache.get(&vec_a(), 6).is_none());
    }

    // -----------------------------------------------------------------------
    // 25. expirations stat accumulated correctly
    // -----------------------------------------------------------------------
    #[test]
    fn test_expirations_stat_accumulated_correctly() {
        let mut cache = cache_with_ttl(1);
        cache.put(&vec_a(), vec![1], 0, None);
        cache.put(&vec_b(), vec![2], 0, None);
        cache.put(&vec_c(), vec![3], 0, None);
        // Trigger expiry via get.
        cache.get(&vec_a(), 5);
        cache.get(&vec_b(), 5);
        // Trigger expiry via evict_expired for C.
        cache.evict_expired(5);
        assert_eq!(cache.stats().expirations, 3);
    }

    // -----------------------------------------------------------------------
    // 26. evictions stat accumulated correctly across multiple evictions
    // -----------------------------------------------------------------------
    #[test]
    fn test_evictions_stat_accumulated_correctly() {
        let mut cache = cache_with_max(1);
        cache.put(&vec_a(), vec![1], 0, None);
        cache.put(&vec_b(), vec![2], 0, None); // evicts A
        cache.put(&vec_c(), vec![3], 0, None); // evicts B
        assert_eq!(cache.stats().evictions, 2);
    }

    // -----------------------------------------------------------------------
    // 27. is_expired boundary conditions
    // -----------------------------------------------------------------------
    #[test]
    fn test_is_expired_boundary_conditions() {
        let entry = CachedQueryResult {
            query_fingerprint: 0,
            result_doc_ids: vec![],
            cached_at_tick: 100,
            ttl_ticks: 50,
            hit_count: 0,
        };
        // Exactly at expiry boundary — NOT expired (current_tick must be GREATER THAN).
        assert!(!entry.is_expired(150));
        // One tick beyond boundary — expired.
        assert!(entry.is_expired(151));
        // Well before expiry.
        assert!(!entry.is_expired(100));
    }

    // -----------------------------------------------------------------------
    // 28. QueryCacheConfig::default() values
    // -----------------------------------------------------------------------
    #[test]
    fn test_query_cache_config_default_values() {
        let cfg = QueryCacheConfig::default();
        assert_eq!(cfg.max_entries, 512);
        assert_eq!(cfg.default_ttl_ticks, 300);
    }

    // -----------------------------------------------------------------------
    // 29. FNV-1a fingerprint is deterministic
    // -----------------------------------------------------------------------
    #[test]
    fn test_embedding_fingerprint_is_deterministic() {
        let v = vec![1.0_f32, 2.0, 3.0, 4.0];
        assert_eq!(embedding_fingerprint(&v), embedding_fingerprint(&v));
    }

    // -----------------------------------------------------------------------
    // 30. FNV-1a fingerprint differs for distinct embeddings
    // -----------------------------------------------------------------------
    #[test]
    fn test_embedding_fingerprint_differs_for_distinct_embeddings() {
        let v1 = vec![1.0_f32, 2.0, 3.0];
        let v2 = vec![3.0_f32, 2.0, 1.0];
        assert_ne!(embedding_fingerprint(&v1), embedding_fingerprint(&v2));
    }

    // -----------------------------------------------------------------------
    // 31. put then multiple invalidations — second returns false
    // -----------------------------------------------------------------------
    #[test]
    fn test_double_invalidate_second_returns_false() {
        let mut cache = default_cache();
        cache.put(&vec_a(), vec![1], 0, None);
        assert!(cache.invalidate(&vec_a()));
        assert!(!cache.invalidate(&vec_a()));
    }

    // -----------------------------------------------------------------------
    // 32. LRU order maintained across multiple accesses
    // -----------------------------------------------------------------------
    #[test]
    fn test_lru_order_maintained_across_accesses() {
        let mut cache = cache_with_max(3);

        // Insert A, B, C in order.
        cache.put(&vec_a(), vec![1], 0, None);
        cache.put(&vec_b(), vec![2], 0, None);
        cache.put(&vec_c(), vec![3], 0, None);

        // Access A — order is now B, C, A (B is LRU).
        cache.get(&vec_a(), 0);

        // Access B — order is now C, A, B (C is LRU).
        cache.get(&vec_b(), 0);

        // Insert a fourth element — C should be evicted.
        let vec_d = vec![0.5_f32, 0.5, 0.0];
        cache.put(&vec_d, vec![4], 0, None);

        assert!(!cache.entries.contains_key(&embedding_fingerprint(&vec_c())));
        assert!(cache.entries.contains_key(&embedding_fingerprint(&vec_a())));
        assert!(cache.entries.contains_key(&embedding_fingerprint(&vec_b())));
        assert!(cache.entries.contains_key(&embedding_fingerprint(&vec_d)));
    }
}
