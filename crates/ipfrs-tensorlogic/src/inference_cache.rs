//! Inference Cache with Invalidation
//!
//! Memoization cache for inference results, keyed by
//! `(goal_term_hash, kb_version) → bindings`.
//! Supports LRU eviction and monotonic-version-based invalidation.

use std::collections::HashMap;
use std::time::Instant;

/// FNV-1a hash of a byte slice
fn fnv1a(data: &[u8]) -> u64 {
    const FNV_OFFSET: u64 = 14_695_981_039_346_656_037;
    const FNV_PRIME: u64 = 1_099_511_628_211;
    let mut hash = FNV_OFFSET;
    for &byte in data {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Compute the FNV-1a hash for a serialised goal term.
pub fn hash_goal(goal_repr: &str) -> u64 {
    fnv1a(goal_repr.as_bytes())
}

/// Cache key: combination of goal hash and KB version.
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct InferenceCacheKey {
    /// FNV-1a of goal term serialization
    pub goal_hash: u64,
    /// Monotonic KB version counter
    pub kb_version: u64,
}

impl InferenceCacheKey {
    /// Create a new cache key.
    pub fn new(goal_hash: u64, kb_version: u64) -> Self {
        Self {
            goal_hash,
            kb_version,
        }
    }
}

/// One cached inference result.
pub struct CachedResult {
    /// Variable bindings found during inference
    pub bindings: Vec<HashMap<String, String>>,
    /// Wall-clock instant at which this result was computed
    pub computed_at: Instant,
    /// Maximum proof depth reached
    pub proof_depth: usize,
}

/// Summary statistics for the cache.
#[derive(Debug, Clone)]
pub struct CacheStats {
    /// Total cache hits
    pub hits: u64,
    /// Total cache misses
    pub misses: u64,
    /// Total evictions due to capacity
    pub evictions: u64,
    /// Current number of entries
    pub entries: usize,
    /// Hit rate in [0.0, 1.0]
    pub hit_rate: f64,
}

/// LRU-ordered key list for eviction purposes.
/// The front is the most recently used; the back is the least recently used.
struct LruOrder {
    order: Vec<InferenceCacheKey>,
}

impl LruOrder {
    fn new() -> Self {
        Self { order: Vec::new() }
    }

    /// Touch a key (move to front). If not present, push to front.
    fn touch(&mut self, key: &InferenceCacheKey) {
        if let Some(pos) = self.order.iter().position(|k| k == key) {
            self.order.remove(pos);
        }
        self.order.insert(0, key.clone());
    }

    /// Remove a key entirely.
    fn remove(&mut self, key: &InferenceCacheKey) {
        self.order.retain(|k| k != key);
    }

    /// Pop the least recently used key.
    fn pop_lru(&mut self) -> Option<InferenceCacheKey> {
        self.order.pop()
    }
}

/// Memoization cache for inference results.
/// Keyed by `(goal_term_hash, kb_version) → bindings`.
pub struct InferenceCache {
    entries: HashMap<InferenceCacheKey, CachedResult>,
    lru: LruOrder,
    max_entries: usize,
    hit_count: u64,
    miss_count: u64,
    eviction_count: u64,
}

impl InferenceCache {
    /// Create a new cache with the given capacity (default 1024).
    pub fn new(max_entries: usize) -> Self {
        let max_entries = if max_entries == 0 { 1024 } else { max_entries };
        Self {
            entries: HashMap::new(),
            lru: LruOrder::new(),
            max_entries,
            hit_count: 0,
            miss_count: 0,
            eviction_count: 0,
        }
    }

    /// Look up a cached result. Records hit/miss statistics.
    pub fn get(&mut self, key: &InferenceCacheKey) -> Option<&CachedResult> {
        if self.entries.contains_key(key) {
            self.hit_count += 1;
            self.lru.touch(key);
            self.entries.get(key)
        } else {
            self.miss_count += 1;
            None
        }
    }

    /// Insert a result. Evicts the least recently used entry when at capacity.
    pub fn insert(&mut self, key: InferenceCacheKey, result: CachedResult) {
        if self.entries.contains_key(&key) {
            // Update in place
            self.entries.insert(key.clone(), result);
            self.lru.touch(&key);
            return;
        }

        // Evict if at capacity
        while self.entries.len() >= self.max_entries {
            if let Some(lru_key) = self.lru.pop_lru() {
                self.entries.remove(&lru_key);
                self.eviction_count += 1;
            } else {
                break;
            }
        }

        self.lru.touch(&key);
        self.entries.insert(key, result);
    }

    /// Invalidate all entries that were computed for a specific KB version.
    pub fn invalidate_for_kb_version(&mut self, old_version: u64) {
        let keys_to_remove: Vec<InferenceCacheKey> = self
            .entries
            .keys()
            .filter(|k| k.kb_version == old_version)
            .cloned()
            .collect();

        for key in &keys_to_remove {
            self.entries.remove(key);
            self.lru.remove(key);
        }
    }

    /// Compute the hit rate as a fraction in [0.0, 1.0].
    pub fn hit_rate(&self) -> f64 {
        let total = self.hit_count + self.miss_count;
        if total == 0 {
            0.0
        } else {
            self.hit_count as f64 / total as f64
        }
    }

    /// Return a snapshot of cache statistics.
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            hits: self.hit_count,
            misses: self.miss_count,
            evictions: self.eviction_count,
            entries: self.entries.len(),
            hit_rate: self.hit_rate(),
        }
    }

    /// Number of currently stored entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl Default for InferenceCache {
    fn default() -> Self {
        Self::new(1024)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_result(bindings: Vec<HashMap<String, String>>) -> CachedResult {
        CachedResult {
            bindings,
            computed_at: Instant::now(),
            proof_depth: 1,
        }
    }

    #[test]
    fn test_cache_hit() {
        let mut cache = InferenceCache::new(1024);
        let key = InferenceCacheKey::new(42, 1);
        let mut binding = HashMap::new();
        binding.insert("X".to_string(), "alice".to_string());
        cache.insert(key.clone(), make_result(vec![binding.clone()]));

        let result = cache.get(&key);
        assert!(result.is_some());
        let r = result.expect("test: should succeed");
        assert_eq!(r.bindings.len(), 1);
        assert_eq!(r.bindings[0].get("X").map(|s| s.as_str()), Some("alice"));
    }

    #[test]
    fn test_cache_miss() {
        let mut cache = InferenceCache::new(1024);
        let key = InferenceCacheKey::new(99, 1);

        let result = cache.get(&key);
        assert!(result.is_none());
        assert_eq!(cache.stats().misses, 1);
        assert_eq!(cache.stats().hits, 0);
    }

    #[test]
    fn test_cache_invalidation() {
        let mut cache = InferenceCache::new(1024);

        // Insert entries for version 1
        let key1 = InferenceCacheKey::new(1, 1);
        let key2 = InferenceCacheKey::new(2, 1);
        // Also insert an entry for version 2 that should survive
        let key3 = InferenceCacheKey::new(3, 2);

        cache.insert(key1.clone(), make_result(vec![]));
        cache.insert(key2.clone(), make_result(vec![]));
        cache.insert(key3.clone(), make_result(vec![]));

        assert_eq!(cache.len(), 3);

        // Invalidate version 1
        cache.invalidate_for_kb_version(1);

        assert_eq!(cache.len(), 1);
        assert!(cache.get(&key1).is_none());
        assert!(cache.get(&key2).is_none());
        // Version 2 entry remains, but get() will record a hit now
        let stats_before = cache.stats();
        let _ = cache.get(&key3);
        let stats_after = cache.stats();
        assert_eq!(stats_after.hits, stats_before.hits + 1);
    }

    #[test]
    fn test_cache_hit_rate() {
        let mut cache = InferenceCache::new(1024);
        let key = InferenceCacheKey::new(7, 1);
        cache.insert(key.clone(), make_result(vec![]));

        // 3 hits
        cache.get(&key);
        cache.get(&key);
        cache.get(&key);

        // 1 miss
        let missing = InferenceCacheKey::new(999, 1);
        cache.get(&missing);

        let rate = cache.hit_rate();
        // 3 hits out of 4 total = 0.75
        assert!((rate - 0.75).abs() < 1e-9, "expected 0.75 but got {}", rate);
    }

    #[test]
    fn test_kb_version_bumps_on_rule_add() {
        use crate::ir::{Predicate, Rule, Term};
        use crate::storage::TensorLogicStore;
        use ipfrs_storage::{BlockStoreConfig, SledBlockStore};
        use std::sync::Arc;

        let path = std::env::temp_dir().join("ipfrs-test-kb-version-bump");
        let _ = std::fs::remove_dir_all(&path);
        let config = BlockStoreConfig {
            path,
            cache_size: 32 * 1024 * 1024,
        };
        let store = Arc::new(SledBlockStore::new(config).expect("sled store"));
        let tl_store = TensorLogicStore::new(store).expect("tensorlogic store");

        let version_before = tl_store.kb_version();

        let rule = Rule::new(
            Predicate::new(
                "ancestor".to_string(),
                vec![Term::Var("X".to_string()), Term::Var("Y".to_string())],
            ),
            vec![Predicate::new(
                "parent".to_string(),
                vec![Term::Var("X".to_string()), Term::Var("Y".to_string())],
            )],
        );

        tl_store.add_rule(rule).expect("add_rule");

        let version_after = tl_store.kb_version();
        assert!(
            version_after > version_before,
            "kb_version should increment after add_rule: before={}, after={}",
            version_before,
            version_after,
        );
    }
}
