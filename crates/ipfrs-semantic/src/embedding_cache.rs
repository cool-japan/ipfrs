//! # Semantic Embedding Cache
//!
//! Cache for computed embeddings to avoid re-computation. Provides
//! TTL-based expiry, LRU batch eviction, prefix invalidation, and
//! hit/miss statistics.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

/// A single cached embedding entry.
#[derive(Debug, Clone)]
pub struct CachedEmbedding {
    /// The cache key that identifies this embedding.
    pub key: String,
    /// The embedding vector.
    pub embedding: Vec<f64>,
    /// Tick at which this entry was stored.
    pub computed_tick: u64,
    /// Number of times the entry has been accessed via `get`.
    pub access_count: u64,
    /// Time-to-live measured in ticks.
    pub ttl_ticks: u64,
    /// Monotonic insertion sequence for deterministic eviction tie-breaking.
    pub insertion_seq: u64,
}

/// Configuration for [`SemanticEmbeddingCache`].
#[derive(Debug, Clone)]
pub struct EmbeddingCacheConfig {
    /// Maximum number of entries before eviction triggers (default 10 000).
    pub max_entries: usize,
    /// Default TTL in ticks for entries that do not specify one (default 500).
    pub default_ttl_ticks: u64,
    /// Number of entries to evict in one batch when the cache is full (default 100).
    pub eviction_batch_size: usize,
}

impl Default for EmbeddingCacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 10_000,
            default_ttl_ticks: 500,
            eviction_batch_size: 100,
        }
    }
}

/// Aggregate statistics for the cache.
#[derive(Debug, Clone)]
pub struct EmbeddingCacheStats {
    /// Current number of entries.
    pub entry_count: usize,
    /// Total number of cache hits.
    pub hits: u64,
    /// Total number of cache misses.
    pub misses: u64,
    /// Total number of evicted entries.
    pub evictions: u64,
    /// Hit rate as a fraction in `[0.0, 1.0]` (0.0 when no lookups).
    pub hit_rate: f64,
    /// Estimated memory footprint in bytes.
    pub memory_bytes_estimate: usize,
}

// ---------------------------------------------------------------------------
// Cache implementation
// ---------------------------------------------------------------------------

/// A tick-based embedding cache with LRU batch eviction and TTL expiry.
pub struct SemanticEmbeddingCache {
    config: EmbeddingCacheConfig,
    entries: HashMap<String, CachedEmbedding>,
    current_tick: u64,
    next_insertion_seq: u64,
    hits: u64,
    misses: u64,
    evictions: u64,
}

impl SemanticEmbeddingCache {
    /// Create a new cache with the given configuration.
    pub fn new(config: EmbeddingCacheConfig) -> Self {
        Self {
            entries: HashMap::with_capacity(config.max_entries),
            config,
            current_tick: 0,
            next_insertion_seq: 0,
            hits: 0,
            misses: 0,
            evictions: 0,
        }
    }

    /// Insert an embedding into the cache.
    ///
    /// If `ttl` is `None` the default TTL from the config is used.
    /// When the cache exceeds `max_entries` after insertion the least-recently
    /// used entries (by `access_count`, then oldest `computed_tick`) are evicted
    /// in a batch.
    pub fn put(&mut self, key: &str, embedding: Vec<f64>, ttl: Option<u64>) {
        let ttl_ticks = ttl.unwrap_or(self.config.default_ttl_ticks);
        let insertion_seq = self.next_insertion_seq;
        self.next_insertion_seq += 1;
        let entry = CachedEmbedding {
            key: key.to_string(),
            embedding,
            computed_tick: self.current_tick,
            access_count: 0,
            ttl_ticks,
            insertion_seq,
        };
        self.entries.insert(key.to_string(), entry);

        if self.entries.len() > self.config.max_entries {
            self.evict_batch();
        }
    }

    /// Retrieve the embedding for `key` if it exists and has not expired.
    ///
    /// Increments the entry's `access_count` and records a hit. Returns `None`
    /// (and records a miss) when the key is absent or the entry has expired.
    pub fn get(&mut self, key: &str) -> Option<&[f64]> {
        // Two-phase: check expiry first, then borrow mutably.
        let expired = self
            .entries
            .get(key)
            .map(|e| self.current_tick >= e.computed_tick + e.ttl_ticks)
            .unwrap_or(true);

        if expired {
            // Remove if it exists but is expired.
            self.entries.remove(key);
            self.misses += 1;
            return None;
        }

        self.hits += 1;
        // SAFETY: we know the key exists and is not expired.
        let entry = self
            .entries
            .get_mut(key)
            .expect("entry must exist after non-expired check");
        entry.access_count += 1;
        Some(&entry.embedding)
    }

    /// Check whether `key` is present **and** not expired without updating
    /// the entry's `access_count`.
    pub fn contains(&self, key: &str) -> bool {
        self.entries
            .get(key)
            .is_some_and(|e| self.current_tick < e.computed_tick + e.ttl_ticks)
    }

    /// Remove a specific entry. Returns `true` if the entry existed.
    pub fn invalidate(&mut self, key: &str) -> bool {
        self.entries.remove(key).is_some()
    }

    /// Remove all entries whose key starts with `prefix`. Returns the number
    /// of entries removed.
    pub fn invalidate_prefix(&mut self, prefix: &str) -> usize {
        let keys_to_remove: Vec<String> = self
            .entries
            .keys()
            .filter(|k| k.starts_with(prefix))
            .cloned()
            .collect();
        let count = keys_to_remove.len();
        for k in keys_to_remove {
            self.entries.remove(&k);
        }
        count
    }

    /// Advance the internal tick counter by one and remove all expired entries.
    pub fn tick_cleanup(&mut self) {
        self.current_tick += 1;
        self.entries
            .retain(|_, e| self.current_tick < e.computed_tick + e.ttl_ticks);
    }

    /// Return the current number of entries in the cache.
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// Return the hit rate as a value in `[0.0, 1.0]`.
    ///
    /// Returns `0.0` when no lookups have been performed.
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }

    /// Estimate the memory footprint of all cached entries in bytes.
    ///
    /// Each entry contributes `key.len() + embedding.len() * 8` bytes (the
    /// heap portion of the key string plus the heap portion of the `Vec<f64>`).
    pub fn memory_estimate(&self) -> usize {
        self.entries
            .values()
            .map(|e| e.key.len() + e.embedding.len() * 8)
            .sum()
    }

    /// Return a snapshot of the cache statistics.
    pub fn stats(&self) -> EmbeddingCacheStats {
        EmbeddingCacheStats {
            entry_count: self.entry_count(),
            hits: self.hits,
            misses: self.misses,
            evictions: self.evictions,
            hit_rate: self.hit_rate(),
            memory_bytes_estimate: self.memory_estimate(),
        }
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Evict the `eviction_batch_size` least-recently-used entries.
    ///
    /// "Least recently used" is defined as lowest `access_count`; ties are
    /// broken by oldest `computed_tick`.
    fn evict_batch(&mut self) {
        let batch = self.config.eviction_batch_size;
        if batch == 0 || self.entries.is_empty() {
            return;
        }

        // Collect (key, access_count, computed_tick, insertion_seq) for sorting.
        let mut candidates: Vec<(String, u64, u64, u64)> = self
            .entries
            .iter()
            .map(|(k, e)| (k.clone(), e.access_count, e.computed_tick, e.insertion_seq))
            .collect();

        // Sort ascending by access_count, then computed_tick, then insertion_seq (oldest first).
        candidates.sort_by(|a, b| {
            a.1.cmp(&b.1)
                .then_with(|| a.2.cmp(&b.2))
                .then_with(|| a.3.cmp(&b.3))
        });

        let to_remove = batch.min(candidates.len());
        for (key, _, _, _) in candidates.into_iter().take(to_remove) {
            self.entries.remove(&key);
            self.evictions += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_cache() -> SemanticEmbeddingCache {
        SemanticEmbeddingCache::new(EmbeddingCacheConfig::default())
    }

    fn small_cache(max: usize, ttl: u64, batch: usize) -> SemanticEmbeddingCache {
        SemanticEmbeddingCache::new(EmbeddingCacheConfig {
            max_entries: max,
            default_ttl_ticks: ttl,
            eviction_batch_size: batch,
        })
    }

    // 1 ─ put / get roundtrip
    #[test]
    fn test_put_get_roundtrip() {
        let mut c = default_cache();
        c.put("a", vec![1.0, 2.0, 3.0], None);
        let v = c.get("a");
        assert!(v.is_some());
        assert_eq!(v.map(|s| s.to_vec()), Some(vec![1.0, 2.0, 3.0]));
    }

    // 2 ─ get missing key
    #[test]
    fn test_get_missing() {
        let mut c = default_cache();
        assert!(c.get("no_such_key").is_none());
    }

    // 3 ─ TTL expiry
    #[test]
    fn test_ttl_expiry() {
        let mut c = small_cache(100, 2, 10);
        c.put("x", vec![1.0], None); // ttl = 2
        assert!(c.get("x").is_some()); // tick 0, computed 0, expires at 2
        c.tick_cleanup(); // tick → 1
        assert!(c.get("x").is_some());
        c.tick_cleanup(); // tick → 2, entry expires (2 >= 0+2)
        assert!(c.get("x").is_none());
    }

    // 4 ─ custom TTL
    #[test]
    fn test_custom_ttl() {
        let mut c = small_cache(100, 100, 10);
        c.put("short", vec![1.0], Some(1));
        c.tick_cleanup(); // tick → 1, entry expires (1 >= 0+1)
        assert!(c.get("short").is_none());
    }

    // 5 ─ LRU eviction triggers
    #[test]
    fn test_lru_eviction_triggers() {
        let mut c = small_cache(5, 1000, 2);
        for i in 0..5 {
            c.put(&format!("k{i}"), vec![i as f64], None);
        }
        assert_eq!(c.entry_count(), 5);
        // Access k3 and k4 to raise their access_count.
        let _ = c.get("k3");
        let _ = c.get("k4");
        // Insert one more → triggers eviction of 2 LRU entries.
        c.put("k5", vec![5.0], None);
        assert!(c.entry_count() <= 5);
        // k3 and k4 should survive (higher access_count).
        assert!(c.contains("k3"));
        assert!(c.contains("k4"));
    }

    // 6 ─ eviction counter
    #[test]
    fn test_eviction_counter() {
        let mut c = small_cache(3, 1000, 2);
        for i in 0..4 {
            c.put(&format!("e{i}"), vec![0.0], None);
        }
        assert_eq!(c.evictions, 2);
    }

    // 7 ─ hit counting
    #[test]
    fn test_hit_counting() {
        let mut c = default_cache();
        c.put("h", vec![1.0], None);
        let _ = c.get("h");
        let _ = c.get("h");
        assert_eq!(c.hits, 2);
    }

    // 8 ─ miss counting
    #[test]
    fn test_miss_counting() {
        let mut c = default_cache();
        let _ = c.get("nope");
        let _ = c.get("nope2");
        assert_eq!(c.misses, 2);
    }

    // 9 ─ hit rate
    #[test]
    fn test_hit_rate() {
        let mut c = default_cache();
        c.put("r", vec![1.0], None);
        let _ = c.get("r"); // hit
        let _ = c.get("miss"); // miss
        let rate = c.hit_rate();
        assert!((rate - 0.5).abs() < 1e-10);
    }

    // 10 ─ hit rate empty
    #[test]
    fn test_hit_rate_empty() {
        let c = default_cache();
        assert!((c.hit_rate() - 0.0).abs() < 1e-10);
    }

    // 11 ─ invalidate existing
    #[test]
    fn test_invalidate_existing() {
        let mut c = default_cache();
        c.put("del", vec![1.0], None);
        assert!(c.invalidate("del"));
        assert!(!c.contains("del"));
    }

    // 12 ─ invalidate missing
    #[test]
    fn test_invalidate_missing() {
        let mut c = default_cache();
        assert!(!c.invalidate("ghost"));
    }

    // 13 ─ invalidate prefix
    #[test]
    fn test_invalidate_prefix() {
        let mut c = default_cache();
        c.put("img:a", vec![1.0], None);
        c.put("img:b", vec![2.0], None);
        c.put("txt:c", vec![3.0], None);
        let removed = c.invalidate_prefix("img:");
        assert_eq!(removed, 2);
        assert_eq!(c.entry_count(), 1);
        assert!(c.contains("txt:c"));
    }

    // 14 ─ invalidate prefix none match
    #[test]
    fn test_invalidate_prefix_none() {
        let mut c = default_cache();
        c.put("abc", vec![1.0], None);
        assert_eq!(c.invalidate_prefix("xyz"), 0);
    }

    // 15 ─ contains does not update access count
    #[test]
    fn test_contains_no_access_update() {
        let mut c = default_cache();
        c.put("c", vec![1.0], None);
        assert!(c.contains("c"));
        assert!(c.contains("c"));
        let entry = c.entries.get("c");
        assert!(entry.is_some());
        assert_eq!(entry.map(|e| e.access_count), Some(0));
    }

    // 16 ─ contains expired returns false
    #[test]
    fn test_contains_expired() {
        let mut c = small_cache(100, 1, 10);
        c.put("e", vec![1.0], None);
        c.tick_cleanup(); // tick → 1, entry expires
        assert!(!c.contains("e"));
    }

    // 17 ─ tick_cleanup removes expired
    #[test]
    fn test_tick_cleanup() {
        let mut c = small_cache(100, 2, 10);
        c.put("a", vec![1.0], None);
        c.put("b", vec![2.0], Some(5));
        c.tick_cleanup(); // tick 1
        c.tick_cleanup(); // tick 2 → a expires
        assert_eq!(c.entry_count(), 1);
        assert!(c.contains("b"));
    }

    // 18 ─ memory estimate
    #[test]
    fn test_memory_estimate() {
        let mut c = default_cache();
        // key "ab" (2 bytes) + 3 f64s (24 bytes) = 26
        c.put("ab", vec![1.0, 2.0, 3.0], None);
        assert_eq!(c.memory_estimate(), 26);
    }

    // 19 ─ memory estimate multiple entries
    #[test]
    fn test_memory_estimate_multiple() {
        let mut c = default_cache();
        c.put("a", vec![1.0], None); // 1 + 8 = 9
        c.put("bb", vec![1.0, 2.0], None); // 2 + 16 = 18
        assert_eq!(c.memory_estimate(), 27);
    }

    // 20 ─ stats accuracy
    #[test]
    fn test_stats_accuracy() {
        let mut c = small_cache(5, 1000, 2);
        c.put("s1", vec![1.0], None);
        c.put("s2", vec![2.0], None);
        let _ = c.get("s1"); // hit
        let _ = c.get("nope"); // miss
        let s = c.stats();
        assert_eq!(s.entry_count, 2);
        assert_eq!(s.hits, 1);
        assert_eq!(s.misses, 1);
        assert!((s.hit_rate - 0.5).abs() < 1e-10);
        assert!(s.memory_bytes_estimate > 0);
    }

    // 21 ─ empty cache
    #[test]
    fn test_empty_cache() {
        let c = default_cache();
        assert_eq!(c.entry_count(), 0);
        assert_eq!(c.memory_estimate(), 0);
        assert_eq!(c.hits, 0);
        assert_eq!(c.misses, 0);
    }

    // 22 ─ batch eviction removes correct count
    #[test]
    fn test_batch_eviction_size() {
        let mut c = small_cache(10, 1000, 5);
        for i in 0..11 {
            c.put(&format!("b{i}"), vec![0.0], None);
        }
        // 11 inserted, max 10 → evict 5 → 6 remain
        assert_eq!(c.entry_count(), 6);
        assert_eq!(c.evictions, 5);
    }

    // 23 ─ overwrite existing key
    #[test]
    fn test_overwrite_key() {
        let mut c = default_cache();
        c.put("ow", vec![1.0], None);
        let _ = c.get("ow"); // access_count → 1
        c.put("ow", vec![9.0], None);
        let v = c.get("ow");
        assert_eq!(v.map(|s| s.to_vec()), Some(vec![9.0]));
        // access_count reset to 0 then incremented by get → 1
        assert_eq!(c.entries.get("ow").map(|e| e.access_count), Some(1));
    }

    // 24 ─ default config values
    #[test]
    fn test_default_config() {
        let cfg = EmbeddingCacheConfig::default();
        assert_eq!(cfg.max_entries, 10_000);
        assert_eq!(cfg.default_ttl_ticks, 500);
        assert_eq!(cfg.eviction_batch_size, 100);
    }

    // 25 ─ large embedding
    #[test]
    fn test_large_embedding() {
        let mut c = default_cache();
        let big = vec![0.42; 768];
        c.put("big", big.clone(), None);
        let v = c.get("big");
        assert_eq!(v.map(|s| s.len()), Some(768));
        assert_eq!(v.map(|s| s[0]), Some(0.42));
    }

    // 26 ─ multiple tick cleanups
    #[test]
    fn test_multiple_tick_cleanups() {
        let mut c = small_cache(100, 3, 10);
        c.put("t1", vec![1.0], Some(1)); // expires tick 1
        c.put("t2", vec![2.0], Some(2)); // expires tick 2
        c.put("t3", vec![3.0], Some(3)); // expires tick 3
        c.tick_cleanup(); // tick 1
        assert_eq!(c.entry_count(), 2);
        c.tick_cleanup(); // tick 2
        assert_eq!(c.entry_count(), 1);
        c.tick_cleanup(); // tick 3
        assert_eq!(c.entry_count(), 0);
    }

    // 27 ─ eviction prefers lowest access_count
    #[test]
    fn test_eviction_prefers_lowest_access() {
        let mut c = small_cache(4, 1000, 1);
        c.put("lo1", vec![0.0], None);
        c.put("lo2", vec![0.0], None);
        c.put("hi1", vec![0.0], None);
        c.put("hi2", vec![0.0], None);

        // Bump access counts for hi1, hi2, lo2.
        for _ in 0..5 {
            let _ = c.get("hi1");
            let _ = c.get("hi2");
            let _ = c.get("lo2");
        }

        // Trigger eviction of 1 entry (lo1 has lowest access_count = 0).
        c.put("new", vec![0.0], None);
        // lo1 should have been evicted (access_count 0, oldest).
        assert!(!c.contains("lo1"));
        assert!(c.contains("hi1"));
        assert!(c.contains("hi2"));
        assert!(c.contains("lo2"));
    }

    // 28 ─ stats evictions field
    #[test]
    fn test_stats_evictions() {
        let mut c = small_cache(3, 1000, 1);
        c.put("a", vec![0.0], None);
        c.put("b", vec![0.0], None);
        c.put("c", vec![0.0], None);
        c.put("d", vec![0.0], None); // triggers 1 eviction
        let s = c.stats();
        assert_eq!(s.evictions, 1);
    }

    // 29 ─ invalidate prefix with empty prefix removes all
    #[test]
    fn test_invalidate_prefix_empty() {
        let mut c = default_cache();
        c.put("a", vec![1.0], None);
        c.put("b", vec![2.0], None);
        let removed = c.invalidate_prefix("");
        assert_eq!(removed, 2);
        assert_eq!(c.entry_count(), 0);
    }

    // 30 ─ get expired entry records miss
    #[test]
    fn test_get_expired_records_miss() {
        let mut c = small_cache(100, 1, 10);
        c.put("exp", vec![1.0], None);
        c.tick_cleanup(); // tick 1 → expired
        let _ = c.get("exp");
        assert_eq!(c.misses, 1);
        assert_eq!(c.hits, 0);
    }
}
