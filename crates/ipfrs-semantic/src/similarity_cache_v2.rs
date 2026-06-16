//! Pairwise cosine-similarity cache with LFU eviction and tick-based TTL.
//!
//! Unlike `similarity_cache`, which is optimised for k-NN query result caching,
//! `similarity_cache_v2` focuses on caching individual *pairwise* similarity
//! scores so that repeated lookups of the same `(id_a, id_b)` tuple avoid
//! redundant floating-point computation.
//!
//! # Design
//!
//! * [`PairKey`] canonicalises the pair so that `(a, b)` and `(b, a)` map to
//!   the same entry.
//! * [`SemanticSimilarityCache`] stores up to `max_entries` pairs. When full,
//!   the entry with the lowest `access_count` (LFU) is evicted.
//! * TTL is expressed in abstract *ticks* (driven by the caller via
//!   [`SemanticSimilarityCache::advance_tick`]), keeping the cache
//!   deterministic and test-friendly.
//!
//! # Example
//! ```rust
//! use ipfrs_semantic::similarity_cache_v2::{PairCacheConfig, SemanticSimilarityCache};
//!
//! let config = PairCacheConfig { max_entries: 100, ttl_ticks: 50 };
//! let mut cache = SemanticSimilarityCache::new(config);
//!
//! let va = vec![1.0_f32, 0.0, 0.0];
//! let vb = vec![0.0_f32, 1.0, 0.0];
//! let sim = cache.compute_and_cache(1, &va, 2, &vb);
//! assert!((sim - 0.0).abs() < 1e-6);
//! ```

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// PairKey
// ---------------------------------------------------------------------------

/// Canonical key for a similarity pair.
///
/// The constructor ensures `id_a == min(a, b)` and `id_b == max(a, b)`, so
/// `PairKey::new(x, y) == PairKey::new(y, x)` for all `x`, `y`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PairKey {
    /// The smaller of the two embedding IDs.
    pub id_a: u64,
    /// The larger of the two embedding IDs.
    pub id_b: u64,
}

impl PairKey {
    /// Construct a canonical [`PairKey`], always placing the smaller ID first.
    #[inline]
    pub fn new(a: u64, b: u64) -> Self {
        Self {
            id_a: a.min(b),
            id_b: a.max(b),
        }
    }
}

// ---------------------------------------------------------------------------
// SimilarityEntry
// ---------------------------------------------------------------------------

/// A single cached pairwise similarity record.
#[derive(Debug, Clone)]
pub struct SimilarityEntry {
    /// The canonical key this entry belongs to.
    pub key: PairKey,
    /// The cosine similarity score, in `[-1.0, 1.0]`.
    pub similarity: f32,
    /// The tick at which this entry was last written.
    pub computed_at_tick: u64,
    /// Number of successful cache reads (hits) for this entry.
    pub access_count: u64,
}

impl SimilarityEntry {
    /// Returns `true` when the entry is older than `ttl_ticks` ticks relative
    /// to `current_tick`.
    #[inline]
    pub fn is_stale(&self, ttl_ticks: u64, current_tick: u64) -> bool {
        current_tick.saturating_sub(self.computed_at_tick) > ttl_ticks
    }
}

// ---------------------------------------------------------------------------
// PairCacheConfig
// ---------------------------------------------------------------------------

/// Configuration for [`SemanticSimilarityCache`].
#[derive(Debug, Clone)]
pub struct PairCacheConfig {
    /// Maximum number of pairwise entries stored simultaneously.
    pub max_entries: usize,
    /// Number of ticks after which an entry is considered stale.
    pub ttl_ticks: u64,
}

impl Default for PairCacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 10_000,
            ttl_ticks: 1_000,
        }
    }
}

// ---------------------------------------------------------------------------
// PairCacheStats
// ---------------------------------------------------------------------------

/// Aggregate statistics for a [`SemanticSimilarityCache`].
#[derive(Debug, Clone, Default)]
pub struct PairCacheStats {
    /// Total successful cache reads.
    pub hits: u64,
    /// Total cache misses (key absent or stale).
    pub misses: u64,
    /// Total LFU evictions performed.
    pub evictions: u64,
}

impl PairCacheStats {
    /// Fraction of lookups that were cache hits.
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
// SemanticSimilarityCache
// ---------------------------------------------------------------------------

/// Pairwise cosine-similarity cache with LFU eviction and TTL staleness.
///
/// All public mutating methods are `&mut self`; no internal locking is
/// performed — the caller is responsible for synchronisation when used from
/// multiple threads.
pub struct SemanticSimilarityCache {
    entries: HashMap<PairKey, SimilarityEntry>,
    config: PairCacheConfig,
    stats: PairCacheStats,
    current_tick: u64,
}

impl SemanticSimilarityCache {
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    /// Create a new cache with the given [`PairCacheConfig`].
    pub fn new(config: PairCacheConfig) -> Self {
        Self {
            entries: HashMap::new(),
            config,
            stats: PairCacheStats::default(),
            current_tick: 0,
        }
    }

    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------

    /// Look up the cached similarity for the pair `(a, b)`.
    ///
    /// Returns `None` when:
    /// * the pair is not in the cache, or
    /// * the cached entry is stale (older than `ttl_ticks`).
    ///
    /// On a hit the entry's `access_count` is incremented.
    pub fn get(&mut self, a: u64, b: u64) -> Option<f32> {
        let key = PairKey::new(a, b);
        let ttl = self.config.ttl_ticks;
        let tick = self.current_tick;

        match self.entries.get_mut(&key) {
            Some(entry) if !entry.is_stale(ttl, tick) => {
                entry.access_count += 1;
                self.stats.hits += 1;
                Some(entry.similarity)
            }
            _ => {
                self.stats.misses += 1;
                None
            }
        }
    }

    /// Insert (or update) the similarity score for the pair `(a, b)`.
    ///
    /// When the cache is at capacity the entry with the lowest
    /// `access_count` is evicted first (LFU).
    pub fn insert(&mut self, a: u64, b: u64, similarity: f32) {
        let key = PairKey::new(a, b);

        // If already present, just overwrite without triggering eviction.
        if self.entries.contains_key(&key) {
            if let Some(entry) = self.entries.get_mut(&key) {
                entry.similarity = similarity;
                entry.computed_at_tick = self.current_tick;
            }
            return;
        }

        // Evict if we are at capacity.
        if self.entries.len() >= self.config.max_entries {
            self.evict_lfu();
        }

        self.entries.insert(
            key,
            SimilarityEntry {
                key,
                similarity,
                computed_at_tick: self.current_tick,
                access_count: 0,
            },
        );
    }

    /// Compute the cosine similarity between `vec_a` and `vec_b`, cache the
    /// result under `(a, b)`, and return the score.
    ///
    /// If `a == b` the result is `1.0` without evaluating the vectors.
    pub fn compute_and_cache(&mut self, a: u64, vec_a: &[f32], b: u64, vec_b: &[f32]) -> f32 {
        let similarity = if a == b {
            1.0_f32
        } else {
            Self::cosine_similarity(vec_a, vec_b)
        };
        self.insert(a, b, similarity);
        similarity
    }

    /// Remove all stale entries from the cache and return the number removed.
    pub fn evict_stale(&mut self) -> usize {
        let ttl = self.config.ttl_ticks;
        let tick = self.current_tick;
        let before = self.entries.len();
        self.entries.retain(|_, entry| !entry.is_stale(ttl, tick));
        let removed = before - self.entries.len();
        self.stats.evictions += removed as u64;
        removed
    }

    /// Advance the internal logical clock by one tick.
    pub fn advance_tick(&mut self) {
        self.current_tick = self.current_tick.saturating_add(1);
    }

    /// Return a reference to the current aggregate statistics.
    pub fn stats(&self) -> &PairCacheStats {
        &self.stats
    }

    /// Return the current tick value.
    pub fn current_tick(&self) -> u64 {
        self.current_tick
    }

    /// Return the number of entries currently stored in the cache.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Return `true` when the cache contains no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Evict the entry with the lowest `access_count` (LFU policy).
    ///
    /// When multiple entries share the same minimum count the one that was
    /// computed earliest is evicted (i.e., smallest `computed_at_tick`).
    fn evict_lfu(&mut self) {
        let victim = self
            .entries
            .iter()
            .min_by(|(_ka, ea), (_kb, eb)| {
                ea.access_count
                    .cmp(&eb.access_count)
                    .then(ea.computed_at_tick.cmp(&eb.computed_at_tick))
            })
            .map(|(k, _)| *k);

        if let Some(key) = victim {
            self.entries.remove(&key);
            self.stats.evictions += 1;
        }
    }

    /// Compute cosine similarity between two float slices.
    ///
    /// Returns `0.0` when either vector has zero norm or the slices differ in
    /// length.
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }

        let mut dot = 0.0_f64;
        let mut norm_a = 0.0_f64;
        let mut norm_b = 0.0_f64;

        for (&ai, &bi) in a.iter().zip(b.iter()) {
            let af = ai as f64;
            let bf = bi as f64;
            dot += af * bf;
            norm_a += af * af;
            norm_b += bf * bf;
        }

        let denom = norm_a.sqrt() * norm_b.sqrt();
        if denom < f64::EPSILON {
            0.0
        } else {
            (dot / denom).clamp(-1.0, 1.0) as f32
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_cache() -> SemanticSimilarityCache {
        SemanticSimilarityCache::new(PairCacheConfig::default())
    }

    fn small_cache(max: usize) -> SemanticSimilarityCache {
        SemanticSimilarityCache::new(PairCacheConfig {
            max_entries: max,
            ttl_ticks: 100,
        })
    }

    // -----------------------------------------------------------------------
    // 1. PairKey canonical ordering
    // -----------------------------------------------------------------------

    #[test]
    fn pair_key_canonical_ordering_small_first() {
        let k1 = PairKey::new(3, 7);
        assert_eq!(k1.id_a, 3);
        assert_eq!(k1.id_b, 7);
    }

    #[test]
    fn pair_key_canonical_ordering_large_first() {
        let k1 = PairKey::new(99, 1);
        assert_eq!(k1.id_a, 1);
        assert_eq!(k1.id_b, 99);
    }

    #[test]
    fn pair_key_symmetry() {
        assert_eq!(PairKey::new(5, 10), PairKey::new(10, 5));
    }

    #[test]
    fn pair_key_equal_ids() {
        let k = PairKey::new(42, 42);
        assert_eq!(k.id_a, 42);
        assert_eq!(k.id_b, 42);
    }

    // -----------------------------------------------------------------------
    // 2. get — cache miss
    // -----------------------------------------------------------------------

    #[test]
    fn get_miss_on_empty_cache() {
        let mut cache = default_cache();
        assert!(cache.get(1, 2).is_none());
        assert_eq!(cache.stats().misses, 1);
        assert_eq!(cache.stats().hits, 0);
    }

    #[test]
    fn get_miss_increments_miss_counter() {
        let mut cache = default_cache();
        cache.get(10, 20);
        cache.get(10, 20);
        assert_eq!(cache.stats().misses, 2);
    }

    // -----------------------------------------------------------------------
    // 3. insert + get hit
    // -----------------------------------------------------------------------

    #[test]
    fn insert_and_get_hit() {
        let mut cache = default_cache();
        cache.insert(1, 2, 0.75);
        let result = cache.get(1, 2);
        assert!(result.is_some());
        assert!(
            (result.expect("test: get after insert should return cached similarity") - 0.75).abs()
                < 1e-6
        );
    }

    #[test]
    fn insert_symmetric_get() {
        let mut cache = default_cache();
        cache.insert(7, 3, 0.5);
        // Reverse order should hit the same entry.
        let result = cache.get(3, 7);
        assert!(result.is_some());
        assert!(
            (result.expect("test: symmetric get should return cached similarity") - 0.5).abs()
                < 1e-6
        );
    }

    #[test]
    fn get_hit_increments_hit_counter() {
        let mut cache = default_cache();
        cache.insert(1, 2, 0.9);
        cache.get(1, 2);
        cache.get(2, 1);
        assert_eq!(cache.stats().hits, 2);
    }

    // -----------------------------------------------------------------------
    // 4. Stale entry miss
    // -----------------------------------------------------------------------

    #[test]
    fn stale_entry_returns_none() {
        let mut cache = SemanticSimilarityCache::new(PairCacheConfig {
            max_entries: 100,
            ttl_ticks: 2,
        });
        cache.insert(1, 2, 0.8);
        // Advance past TTL.
        cache.advance_tick();
        cache.advance_tick();
        cache.advance_tick(); // tick 3 > ttl 2
        assert!(cache.get(1, 2).is_none());
        assert_eq!(cache.stats().misses, 1);
    }

    #[test]
    fn entry_not_stale_within_ttl() {
        let mut cache = SemanticSimilarityCache::new(PairCacheConfig {
            max_entries: 100,
            ttl_ticks: 5,
        });
        cache.insert(1, 2, 0.6);
        cache.advance_tick();
        cache.advance_tick(); // tick 2, ttl 5 — still fresh
        assert!(cache.get(1, 2).is_some());
    }

    // -----------------------------------------------------------------------
    // 5. LFU eviction
    // -----------------------------------------------------------------------

    #[test]
    fn lfu_eviction_removes_lowest_access_count() {
        let mut cache = small_cache(2);

        cache.insert(1, 2, 0.1);
        cache.insert(3, 4, 0.2);

        // Access (1,2) twice so its access_count > (3,4).
        cache.get(1, 2);
        cache.get(1, 2);

        // Inserting a third entry must evict (3,4) because access_count == 0.
        cache.insert(5, 6, 0.3);

        assert!(cache.get(1, 2).is_some(), "(1,2) should survive");
        assert!(cache.get(3, 4).is_none(), "(3,4) should be evicted");
        assert!(cache.get(5, 6).is_some(), "(5,6) should be present");
        assert_eq!(cache.stats().evictions, 1);
    }

    #[test]
    fn lfu_eviction_on_full_cache() {
        let mut cache = small_cache(1);
        cache.insert(1, 2, 0.5);
        // Warm up access count.
        cache.get(1, 2);
        cache.get(1, 2);

        // Insert a new pair — must evict (1,2) since the cache is full.
        cache.insert(3, 4, 0.9);
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.stats().evictions, 1);
        assert!(cache.get(3, 4).is_some());
    }

    // -----------------------------------------------------------------------
    // 6. compute_and_cache
    // -----------------------------------------------------------------------

    #[test]
    fn compute_and_cache_returns_correct_value() {
        let mut cache = default_cache();
        let va = vec![1.0_f32, 0.0, 0.0];
        let vb = vec![1.0_f32, 0.0, 0.0];
        let sim = cache.compute_and_cache(10, &va, 20, &vb);
        assert!((sim - 1.0).abs() < 1e-5, "parallel vectors => similarity 1");
    }

    #[test]
    fn compute_and_cache_orthogonal_vectors() {
        let mut cache = default_cache();
        let va = vec![1.0_f32, 0.0];
        let vb = vec![0.0_f32, 1.0];
        let sim = cache.compute_and_cache(1, &va, 2, &vb);
        assert!((sim - 0.0).abs() < 1e-5, "orthogonal => similarity 0");
    }

    #[test]
    fn compute_and_cache_stores_in_cache() {
        let mut cache = default_cache();
        let va = vec![1.0_f32, 1.0];
        let vb = vec![1.0_f32, 1.0];
        cache.compute_and_cache(5, &va, 6, &vb);
        let result = cache.get(5, 6);
        assert!(result.is_some());
    }

    // -----------------------------------------------------------------------
    // 7. Same-id pair → similarity 1.0
    // -----------------------------------------------------------------------

    #[test]
    fn same_id_pair_returns_one() {
        let mut cache = default_cache();
        let v = vec![3.0_f32, 4.0, 0.0];
        let sim = cache.compute_and_cache(7, &v, 7, &v);
        assert!((sim - 1.0).abs() < 1e-6, "same-id pair must return 1.0");
    }

    // -----------------------------------------------------------------------
    // 8. evict_stale
    // -----------------------------------------------------------------------

    #[test]
    fn evict_stale_removes_expired_entries() {
        let mut cache = SemanticSimilarityCache::new(PairCacheConfig {
            max_entries: 100,
            ttl_ticks: 1,
        });
        cache.insert(1, 2, 0.1);
        cache.insert(3, 4, 0.2);
        // Advance past TTL.
        cache.advance_tick();
        cache.advance_tick(); // tick == 2 > ttl 1

        // Fresh entry inserted at tick 2.
        cache.insert(5, 6, 0.3);

        let removed = cache.evict_stale();
        assert_eq!(removed, 2, "two stale entries should be removed");
        assert_eq!(cache.len(), 1, "only the fresh entry remains");
    }

    #[test]
    fn evict_stale_noop_when_all_fresh() {
        let mut cache = default_cache();
        cache.insert(1, 2, 0.5);
        let removed = cache.evict_stale();
        assert_eq!(removed, 0);
    }

    // -----------------------------------------------------------------------
    // 9. advance_tick
    // -----------------------------------------------------------------------

    #[test]
    fn advance_tick_increments() {
        let mut cache = default_cache();
        assert_eq!(cache.current_tick(), 0);
        cache.advance_tick();
        assert_eq!(cache.current_tick(), 1);
        cache.advance_tick();
        assert_eq!(cache.current_tick(), 2);
    }

    // -----------------------------------------------------------------------
    // 10. hit_rate
    // -----------------------------------------------------------------------

    #[test]
    fn hit_rate_zero_when_no_lookups() {
        let cache = default_cache();
        assert!((cache.stats().hit_rate() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn hit_rate_one_hundred_percent() {
        let mut cache = default_cache();
        cache.insert(1, 2, 0.5);
        cache.get(1, 2);
        assert!((cache.stats().hit_rate() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn hit_rate_fifty_percent() {
        let mut cache = default_cache();
        cache.insert(1, 2, 0.5);
        cache.get(1, 2); // hit
        cache.get(3, 4); // miss
        let rate = cache.stats().hit_rate();
        assert!((rate - 0.5).abs() < f64::EPSILON);
    }

    // -----------------------------------------------------------------------
    // 11. access_count increments on hit
    // -----------------------------------------------------------------------

    #[test]
    fn access_count_increments_on_each_hit() {
        let mut cache = default_cache();
        cache.insert(1, 2, 0.9);
        for _ in 0..5 {
            cache.get(1, 2);
        }
        // Inspect the entry directly through the map.
        let key = PairKey::new(1, 2);
        let entry = cache.entries.get(&key).expect("entry must exist");
        assert_eq!(entry.access_count, 5);
    }

    // -----------------------------------------------------------------------
    // 12. insert upsert does not evict when key already exists
    // -----------------------------------------------------------------------

    #[test]
    fn insert_upsert_no_spurious_eviction() {
        let mut cache = small_cache(1);
        cache.insert(1, 2, 0.3);
        // Re-insert same key — should overwrite, not evict.
        cache.insert(2, 1, 0.7);
        assert_eq!(cache.stats().evictions, 0);
        let result = cache.get(1, 2);
        assert!(result.is_some());
        assert!(
            (result.expect("test: get after upsert should return updated similarity") - 0.7).abs()
                < 1e-6
        );
    }

    // -----------------------------------------------------------------------
    // 13. Zero-norm vector handling
    // -----------------------------------------------------------------------

    #[test]
    fn zero_norm_vector_returns_zero_similarity() {
        let mut cache = default_cache();
        let zero = vec![0.0_f32, 0.0, 0.0];
        let v = vec![1.0_f32, 2.0, 3.0];
        let sim = cache.compute_and_cache(1, &zero, 2, &v);
        assert!((sim - 0.0).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // 14. is_empty / len helpers
    // -----------------------------------------------------------------------

    #[test]
    fn is_empty_and_len() {
        let mut cache = default_cache();
        assert!(cache.is_empty());
        cache.insert(1, 2, 0.5);
        assert!(!cache.is_empty());
        assert_eq!(cache.len(), 1);
    }

    // -----------------------------------------------------------------------
    // 15. PairCacheConfig default values
    // -----------------------------------------------------------------------

    #[test]
    fn pair_cache_config_defaults() {
        let cfg = PairCacheConfig::default();
        assert_eq!(cfg.max_entries, 10_000);
        assert_eq!(cfg.ttl_ticks, 1_000);
    }

    // -----------------------------------------------------------------------
    // 16. SimilarityEntry::is_stale boundary conditions
    // -----------------------------------------------------------------------

    #[test]
    fn is_stale_exactly_at_ttl_boundary() {
        let entry = SimilarityEntry {
            key: PairKey::new(0, 1),
            similarity: 0.5,
            computed_at_tick: 0,
            access_count: 0,
        };
        // Age == ttl_ticks: NOT stale (> required, not >=).
        assert!(!entry.is_stale(5, 5));
        // Age > ttl_ticks: stale.
        assert!(entry.is_stale(5, 6));
    }

    // -----------------------------------------------------------------------
    // 17. cosine_similarity: mismatched lengths
    // -----------------------------------------------------------------------

    #[test]
    fn cosine_similarity_mismatched_lengths() {
        // Access private fn through compute_and_cache with mismatched vectors.
        let mut cache = default_cache();
        let va = vec![1.0_f32, 0.0];
        let vb = vec![1.0_f32, 0.0, 0.0];
        let sim = cache.compute_and_cache(1, &va, 2, &vb);
        assert!((sim - 0.0).abs() < 1e-6, "mismatched lengths => 0.0");
    }
}
