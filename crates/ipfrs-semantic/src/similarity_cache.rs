//! Two-level cache for cosine similarity scores between embedding pairs.
//!
//! Avoids redundant computation during k-NN searches by caching previously
//! computed cosine similarity scores with LFU eviction and TTL-based staleness.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// SimilarityKey
// ---------------------------------------------------------------------------

/// Canonical key for a similarity pair. Always stores the smaller ID first so
/// that `new(a, b) == new(b, a)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SimilarityKey {
    /// Smaller of the two embedding IDs.
    pub a_id: u64,
    /// Larger of the two embedding IDs.
    pub b_id: u64,
}

impl SimilarityKey {
    /// Construct a canonical key ensuring `a_id <= b_id`.
    #[inline]
    pub fn new(x: u64, y: u64) -> Self {
        Self {
            a_id: x.min(y),
            b_id: x.max(y),
        }
    }
}

// ---------------------------------------------------------------------------
// SimilarityEntry
// ---------------------------------------------------------------------------

/// A cached similarity score along with bookkeeping metadata.
#[derive(Debug, Clone)]
pub struct SimilarityEntry {
    /// The cosine similarity score in `[-1.0, 1.0]`.
    pub score: f32,
    /// Unix timestamp (seconds) at which this entry was computed.
    pub computed_at: u64,
    /// Number of times this entry has been returned from the cache.
    pub hit_count: u32,
}

impl SimilarityEntry {
    /// Returns `true` if the entry is older than `ttl_secs` relative to
    /// `now_secs`.
    #[inline]
    pub fn is_stale(&self, ttl_secs: u64, now_secs: u64) -> bool {
        now_secs.saturating_sub(self.computed_at) > ttl_secs
    }
}

// ---------------------------------------------------------------------------
// CacheStats
// ---------------------------------------------------------------------------

/// Aggregate statistics for an [`EmbeddingSimilarityCache`].
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Total number of successful cache lookups.
    pub hits: u64,
    /// Total number of failed cache lookups (including stale evictions).
    pub misses: u64,
    /// Total number of entries evicted (both LFU capacity evictions and stale
    /// sweeps).
    pub evictions: u64,
    /// Current number of entries stored in the cache.
    pub current_size: usize,
}

impl CacheStats {
    /// Fraction of lookups that were satisfied by the cache.
    ///
    /// Returns `0.0` when no lookups have been performed.
    #[inline]
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
// EmbeddingSimilarityCache
// ---------------------------------------------------------------------------

/// Two-level cache for cosine similarity scores between embedding ID pairs.
///
/// Entries are evicted using a **Least-Frequently-Used (LFU)** policy when the
/// cache reaches `max_capacity`, and are considered stale after `ttl_secs`
/// seconds.
pub struct EmbeddingSimilarityCache {
    cache: HashMap<SimilarityKey, SimilarityEntry>,
    max_capacity: usize,
    ttl_secs: u64,
    stats: CacheStats,
}

impl EmbeddingSimilarityCache {
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    /// Create a new cache with the given capacity and TTL.
    pub fn new(max_capacity: usize, ttl_secs: u64) -> Self {
        Self {
            cache: HashMap::new(),
            max_capacity,
            ttl_secs,
            stats: CacheStats::default(),
        }
    }

    // -----------------------------------------------------------------------
    // Core cache operations
    // -----------------------------------------------------------------------

    /// Look up the similarity score for the pair `(a, b)`.
    ///
    /// - Returns `Some(score)` and increments `hit_count` + `stats.hits` on a
    ///   fresh hit.
    /// - Removes the entry and increments `stats.misses` when stale.
    /// - Returns `None` and increments `stats.misses` when absent.
    pub fn get(&mut self, a: u64, b: u64) -> Option<f32> {
        let key = SimilarityKey::new(a, b);
        let now = current_unix_secs();

        match self.cache.get_mut(&key) {
            Some(entry) if !entry.is_stale(self.ttl_secs, now) => {
                entry.hit_count = entry.hit_count.saturating_add(1);
                let score = entry.score;
                self.stats.hits += 1;
                self.stats.current_size = self.cache.len();
                Some(score)
            }
            Some(_stale) => {
                // Entry exists but is stale — remove it.
                self.cache.remove(&key);
                self.stats.misses += 1;
                self.stats.evictions += 1;
                self.stats.current_size = self.cache.len();
                None
            }
            None => {
                self.stats.misses += 1;
                None
            }
        }
    }

    /// Insert a similarity score for the pair `(a, b)`.
    ///
    /// When the cache is at capacity the entry with the lowest `hit_count` is
    /// evicted first (ties broken arbitrarily).
    pub fn insert(&mut self, a: u64, b: u64, score: f32) {
        let key = SimilarityKey::new(a, b);

        // If the key already exists, overwrite in-place without eviction.
        if self.cache.contains_key(&key) {
            let entry = self.cache.get_mut(&key).expect("key confirmed present");
            entry.score = score;
            entry.computed_at = current_unix_secs();
            // Reset hit_count for the refreshed entry.
            entry.hit_count = 0;
            self.stats.current_size = self.cache.len();
            return;
        }

        // Evict LFU entry when at capacity.
        if self.max_capacity > 0 && self.cache.len() >= self.max_capacity {
            if let Some(lfu_key) = self
                .cache
                .iter()
                .min_by_key(|(_, e)| e.hit_count)
                .map(|(k, _)| *k)
            {
                self.cache.remove(&lfu_key);
                self.stats.evictions += 1;
            }
        }

        self.cache.insert(
            key,
            SimilarityEntry {
                score,
                computed_at: current_unix_secs(),
                hit_count: 0,
            },
        );
        self.stats.current_size = self.cache.len();
    }

    // -----------------------------------------------------------------------
    // Similarity computation
    // -----------------------------------------------------------------------

    /// Compute the cosine similarity between two dense vectors.
    ///
    /// Returns `0.0` if either vector has a zero norm.  Vectors of different
    /// lengths are treated as though padded with zeros — only the shorter
    /// length is used for the dot product, and each norm is computed over its
    /// own full slice.
    pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot / (norm_a * norm_b)
        }
    }

    /// Return the cached similarity for `(a_id, b_id)`, computing and caching
    /// it on a miss.
    pub fn compute_and_cache(&mut self, a_id: u64, a_vec: &[f32], b_id: u64, b_vec: &[f32]) -> f32 {
        if let Some(score) = self.get(a_id, b_id) {
            return score;
        }
        let score = Self::cosine_similarity(a_vec, b_vec);
        self.insert(a_id, b_id, score);
        score
    }

    // -----------------------------------------------------------------------
    // Maintenance
    // -----------------------------------------------------------------------

    /// Remove all entries whose TTL has expired.
    ///
    /// Returns the number of entries removed and updates `stats.evictions`.
    pub fn evict_stale(&mut self) -> usize {
        let now = current_unix_secs();
        let ttl = self.ttl_secs;
        let before = self.cache.len();
        self.cache.retain(|_, e| !e.is_stale(ttl, now));
        let removed = before - self.cache.len();
        self.stats.evictions += removed as u64;
        self.stats.current_size = self.cache.len();
        removed
    }

    /// Return a reference to the current cache statistics.
    #[inline]
    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }

    /// Clear all entries and reset all statistics.
    pub fn clear(&mut self) {
        self.cache.clear();
        self.stats = CacheStats::default();
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Returns the current Unix timestamp in seconds, or 0 on error.
#[inline]
fn current_unix_secs() -> u64 {
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

    // -- SimilarityKey -------------------------------------------------------

    #[test]
    fn test_similarity_key_canonical_ordering() {
        let k1 = SimilarityKey::new(5, 3);
        assert_eq!(k1.a_id, 3);
        assert_eq!(k1.b_id, 5);
    }

    #[test]
    fn test_similarity_key_already_ordered() {
        let k = SimilarityKey::new(1, 9);
        assert_eq!(k.a_id, 1);
        assert_eq!(k.b_id, 9);
    }

    #[test]
    fn test_similarity_key_symmetric() {
        assert_eq!(SimilarityKey::new(1, 2), SimilarityKey::new(2, 1));
    }

    #[test]
    fn test_similarity_key_same_ids() {
        let k = SimilarityKey::new(7, 7);
        assert_eq!(k.a_id, 7);
        assert_eq!(k.b_id, 7);
    }

    // -- SimilarityEntry -----------------------------------------------------

    #[test]
    fn test_similarity_entry_not_stale() {
        let now = current_unix_secs();
        let entry = SimilarityEntry {
            score: 0.9,
            computed_at: now,
            hit_count: 0,
        };
        assert!(!entry.is_stale(60, now));
    }

    #[test]
    fn test_similarity_entry_stale() {
        let entry = SimilarityEntry {
            score: 0.9,
            computed_at: 100,
            hit_count: 0,
        };
        // now = 200, ttl = 50 → age = 100 > 50 → stale
        assert!(entry.is_stale(50, 200));
    }

    #[test]
    fn test_similarity_entry_exactly_at_ttl_boundary() {
        // age == ttl should NOT be stale (strict >)
        let entry = SimilarityEntry {
            score: 0.5,
            computed_at: 100,
            hit_count: 0,
        };
        assert!(!entry.is_stale(100, 200)); // age == ttl → not stale
        assert!(entry.is_stale(99, 200)); // age > ttl → stale
    }

    // -- CacheStats ----------------------------------------------------------

    #[test]
    fn test_cache_stats_hit_rate_zero_when_empty() {
        let stats = CacheStats::default();
        assert_eq!(stats.hit_rate(), 0.0);
    }

    #[test]
    fn test_cache_stats_hit_rate() {
        let stats = CacheStats {
            hits: 3,
            misses: 1,
            evictions: 0,
            current_size: 0,
        };
        assert!((stats.hit_rate() - 0.75).abs() < f64::EPSILON);
    }

    // -- get() ---------------------------------------------------------------

    #[test]
    fn test_get_miss_increments_misses() {
        let mut cache = EmbeddingSimilarityCache::new(100, 3600);
        assert!(cache.get(1, 2).is_none());
        assert_eq!(cache.stats().misses, 1);
        assert_eq!(cache.stats().hits, 0);
    }

    #[test]
    fn test_get_hit_increments_hits_and_hit_count() {
        let mut cache = EmbeddingSimilarityCache::new(100, 3600);
        cache.insert(1, 2, 0.8);
        let score = cache.get(1, 2);
        assert!(score.is_some());
        assert!(
            (score.expect("test: similarity score should be present after insert") - 0.8).abs()
                < f32::EPSILON
        );
        assert_eq!(cache.stats().hits, 1);
        assert_eq!(cache.stats().misses, 0);

        // Verify hit_count was incremented
        let key = SimilarityKey::new(1, 2);
        assert_eq!(cache.cache[&key].hit_count, 1);
    }

    #[test]
    fn test_get_stale_entry_removed_and_misses_incremented() {
        let mut cache = EmbeddingSimilarityCache::new(100, 0);
        // Insert with ttl=0; any positive age makes it stale.
        // We manually plant a stale entry.
        let key = SimilarityKey::new(10, 20);
        cache.cache.insert(
            key,
            SimilarityEntry {
                score: 0.5,
                computed_at: 1, // epoch start → definitely stale
                hit_count: 0,
            },
        );
        let result = cache.get(10, 20);
        assert!(result.is_none());
        assert_eq!(cache.stats().misses, 1);
        assert!(!cache.cache.contains_key(&key));
    }

    // -- insert() ------------------------------------------------------------

    #[test]
    fn test_insert_under_capacity() {
        let mut cache = EmbeddingSimilarityCache::new(10, 3600);
        cache.insert(1, 2, 0.7);
        assert_eq!(cache.stats().current_size, 1);
    }

    #[test]
    fn test_insert_at_capacity_evicts_lowest_hit_count() {
        let mut cache = EmbeddingSimilarityCache::new(3, 3600);
        cache.insert(1, 2, 0.1);
        cache.insert(3, 4, 0.2);
        cache.insert(5, 6, 0.3);

        // Bump hit_count on (1,2) and (3,4) so (5,6) has the lowest.
        cache.get(1, 2);
        cache.get(1, 2);
        cache.get(3, 4);

        // Now insert a 4th entry — should evict (5,6) with hit_count=0.
        cache.insert(7, 8, 0.9);
        assert_eq!(cache.stats().current_size, 3);
        assert!(!cache.cache.contains_key(&SimilarityKey::new(5, 6)));
        assert!(cache.cache.contains_key(&SimilarityKey::new(1, 2)));
        assert!(cache.cache.contains_key(&SimilarityKey::new(3, 4)));
        assert!(cache.cache.contains_key(&SimilarityKey::new(7, 8)));
    }

    // -- cosine_similarity() -------------------------------------------------

    #[test]
    fn test_cosine_similarity_identical_vectors() {
        let v = vec![1.0_f32, 2.0, 3.0];
        let sim = EmbeddingSimilarityCache::cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal_vectors() {
        let a = vec![1.0_f32, 0.0];
        let b = vec![0.0_f32, 1.0];
        let sim = EmbeddingSimilarityCache::cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let a = vec![0.0_f32, 0.0, 0.0];
        let b = vec![1.0_f32, 2.0, 3.0];
        let sim = EmbeddingSimilarityCache::cosine_similarity(&a, &b);
        assert_eq!(sim, 0.0);
    }

    #[test]
    fn test_cosine_similarity_positive() {
        let a = vec![1.0_f32, 1.0];
        let b = vec![1.0_f32, 0.5];
        let sim = EmbeddingSimilarityCache::cosine_similarity(&a, &b);
        assert!(sim > 0.0 && sim <= 1.0);
    }

    #[test]
    fn test_cosine_similarity_negative() {
        let a = vec![1.0_f32, 0.0];
        let b = vec![-1.0_f32, 0.0];
        let sim = EmbeddingSimilarityCache::cosine_similarity(&a, &b);
        assert!((sim - (-1.0)).abs() < 1e-6);
    }

    // -- compute_and_cache() -------------------------------------------------

    #[test]
    fn test_compute_and_cache_on_miss_computes_and_inserts() {
        let mut cache = EmbeddingSimilarityCache::new(100, 3600);
        let a = vec![1.0_f32, 0.0];
        let b = vec![0.0_f32, 1.0];
        let score = cache.compute_and_cache(1, &a, 2, &b);
        assert!(score.abs() < 1e-6); // orthogonal
        assert_eq!(cache.stats().current_size, 1);
        assert_eq!(cache.stats().misses, 1);
    }

    #[test]
    fn test_compute_and_cache_on_hit_returns_cached() {
        let mut cache = EmbeddingSimilarityCache::new(100, 3600);
        let a = vec![1.0_f32, 0.0];
        let b = vec![1.0_f32, 0.0];
        cache.compute_and_cache(1, &a, 2, &b); // miss → compute
        let score = cache.compute_and_cache(1, &a, 2, &b); // hit
        assert!((score - 1.0).abs() < 1e-6);
        assert_eq!(cache.stats().hits, 1);
        assert_eq!(cache.stats().misses, 1);
    }

    // -- evict_stale() -------------------------------------------------------

    #[test]
    fn test_evict_stale_removes_expired_entries() {
        let mut cache = EmbeddingSimilarityCache::new(100, 3600);
        // Plant two entries with ancient timestamps.
        for (a, b) in [(1u64, 2u64), (3, 4)] {
            cache.cache.insert(
                SimilarityKey::new(a, b),
                SimilarityEntry {
                    score: 0.5,
                    computed_at: 1,
                    hit_count: 0,
                },
            );
        }
        // Insert one fresh entry.
        cache.insert(5, 6, 0.9);

        let removed = cache.evict_stale();
        assert_eq!(removed, 2);
        assert_eq!(cache.stats().current_size, 1);
    }

    #[test]
    fn test_evict_stale_updates_eviction_count() {
        let mut cache = EmbeddingSimilarityCache::new(100, 3600);
        cache.cache.insert(
            SimilarityKey::new(99, 100),
            SimilarityEntry {
                score: 0.3,
                computed_at: 1,
                hit_count: 0,
            },
        );
        cache.evict_stale();
        assert_eq!(cache.stats().evictions, 1);
    }

    // -- clear() -------------------------------------------------------------

    #[test]
    fn test_clear_resets_cache_and_stats() {
        let mut cache = EmbeddingSimilarityCache::new(100, 3600);
        cache.insert(1, 2, 0.5);
        cache.get(1, 2);
        cache.clear();

        assert_eq!(cache.stats().hits, 0);
        assert_eq!(cache.stats().misses, 0);
        assert_eq!(cache.stats().evictions, 0);
        assert_eq!(cache.stats().current_size, 0);
        assert!(cache.cache.is_empty());
    }
}
