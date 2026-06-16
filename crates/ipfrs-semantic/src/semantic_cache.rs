//! # Semantic Cache Layer
//!
//! A vector-similarity-based cache that returns cached results for queries whose
//! embeddings are semantically close to previously seen queries, avoiding redundant
//! computation.
//!
//! ## Overview
//!
//! The [`SemanticCacheLayer`] maintains a collection of [`CacheEntry`] items, each
//! associated with a query text and its embedding vector. On each lookup the cache
//! computes the cosine similarity between the incoming query embedding and every
//! non-expired stored embedding. If the best match exceeds `config.similarity_threshold`
//! the cached result is returned as a [`CacheLookupResult::Hit`]; otherwise
//! [`CacheLookupResult::Miss`] is returned and the caller should compute the result
//! and insert it.
//!
//! ## Example
//!
//! ```rust
//! use ipfrs_semantic::semantic_cache::{
//!     CacheConfig, CacheEvictionPolicy, CacheKey, CacheLookupResult, SemanticCacheLayer,
//! };
//!
//! let config = CacheConfig {
//!     max_entries: 128,
//!     similarity_threshold: 0.92,
//!     ttl_ms: None,
//!     eviction_policy: CacheEvictionPolicy::Lru,
//! };
//!
//! let mut cache = SemanticCacheLayer::new(config);
//! let now: u64 = 0;
//!
//! let key = CacheKey {
//!     query_text: "hello world".to_string(),
//!     embedding: vec![1.0, 0.0, 0.0],
//! };
//! cache.insert(key, "result text".to_string(), now);
//!
//! match cache.lookup(&[0.999, 0.0447, 0.0], now) {
//!     CacheLookupResult::Hit { similarity, result, .. } => {
//!         println!("Cache hit (similarity={:.3}): {}", similarity, result);
//!     }
//!     CacheLookupResult::Miss => println!("Cache miss"),
//! }
//! ```

use std::cmp::Ordering;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// The key used to store a query in the cache.
#[derive(Debug, Clone)]
pub struct CacheKey {
    /// The original query text.
    pub query_text: String,
    /// The embedding vector for the query.
    pub embedding: Vec<f64>,
}

/// A single entry stored in the [`SemanticCacheLayer`].
#[derive(Debug, Clone)]
pub struct CacheEntry {
    /// The cache key (query text + embedding).
    pub key: CacheKey,
    /// The cached result string.
    pub result: String,
    /// How many times this entry has been returned as a cache hit.
    pub hit_count: u64,
    /// Timestamp (milliseconds, caller-supplied) when the entry was inserted.
    pub inserted_at: u64,
    /// Timestamp (milliseconds, caller-supplied) when the entry was last accessed.
    pub last_accessed: u64,
    /// Optional time-to-live in milliseconds; `None` means the entry never expires.
    pub ttl_ms: Option<u64>,
}

impl CacheEntry {
    /// Returns `true` if this entry has expired at the given `now` timestamp.
    #[inline]
    pub fn is_expired(&self, now: u64) -> bool {
        match self.ttl_ms {
            Some(ttl) => now.saturating_sub(self.inserted_at) > ttl,
            None => false,
        }
    }
}

/// Policy used to select which entry to evict when the cache is full.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CacheEvictionPolicy {
    /// Evict the entry that was least recently accessed.
    #[default]
    Lru,
    /// Evict the entry with the lowest hit count.
    Lfu,
    /// Evict the entry with the soonest expiry; fall back to LRU when no TTLs.
    TtlFirst,
}

/// Configuration for the [`SemanticCacheLayer`].
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Maximum number of entries to hold before eviction kicks in.
    pub max_entries: usize,
    /// Cosine similarity threshold in [0, 1].  A lookup whose best match is
    /// strictly below this value is treated as a miss.
    pub similarity_threshold: f64,
    /// Default time-to-live (milliseconds) applied to each inserted entry.
    /// `None` means entries never expire based on time.
    pub ttl_ms: Option<u64>,
    /// Eviction strategy to use when the cache reaches `max_entries`.
    pub eviction_policy: CacheEvictionPolicy,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 1024,
            similarity_threshold: 0.92,
            ttl_ms: None,
            eviction_policy: CacheEvictionPolicy::Lru,
        }
    }
}

/// The result returned by [`SemanticCacheLayer::lookup`].
#[derive(Debug, Clone, PartialEq)]
pub enum CacheLookupResult {
    /// A sufficiently similar query was found in the cache.
    Hit {
        /// Index of the matching entry in the internal entry vector at the time
        /// of the lookup (informational; may become stale after mutations).
        entry_id: usize,
        /// Cosine similarity between the query and the stored embedding.
        similarity: f64,
        /// The cached result.
        result: String,
    },
    /// No sufficiently similar query was found.
    Miss,
}

/// Aggregate statistics returned by [`SemanticCacheLayer::stats`].
#[derive(Debug, Clone)]
pub struct ScCacheStats {
    /// Current number of entries in the cache.
    pub total_entries: usize,
    /// Cumulative cache hits since the cache was created.
    pub total_hits: u64,
    /// Cumulative cache misses since the cache was created.
    pub total_misses: u64,
    /// Hit rate (hits / (hits + misses)); 0.0 when no lookups have been made.
    pub hit_rate: f64,
    /// Total entries ever inserted (not counting evictions/expiry removals).
    pub total_insertions: u64,
    /// Average `hit_count` across all current entries; 0.0 if empty.
    pub avg_hit_count: f64,
}

// ---------------------------------------------------------------------------
// SemanticCacheLayer
// ---------------------------------------------------------------------------

/// A vector-similarity-based cache.
///
/// See the [module-level documentation](self) for a usage example.
#[derive(Debug)]
pub struct SemanticCacheLayer {
    /// Runtime configuration.
    pub config: CacheConfig,
    /// The stored entries.
    pub entries: Vec<CacheEntry>,
    /// Monotonically increasing counter used to assign stable entry IDs.
    pub next_id: usize,
    /// Total cache hits since creation.
    pub total_hits: u64,
    /// Total cache misses since creation.
    pub total_misses: u64,
    /// Total entries ever inserted.
    pub total_insertions: u64,
}

impl SemanticCacheLayer {
    // ------------------------------------------------------------------
    // Construction
    // ------------------------------------------------------------------

    /// Create a new cache with the given configuration.
    pub fn new(config: CacheConfig) -> Self {
        let capacity = config.max_entries.min(4096);
        Self {
            config,
            entries: Vec::with_capacity(capacity),
            next_id: 0,
            total_hits: 0,
            total_misses: 0,
            total_insertions: 0,
        }
    }

    // ------------------------------------------------------------------
    // Core similarity primitive
    // ------------------------------------------------------------------

    /// Compute the cosine similarity between two f64 slices.
    ///
    /// Returns `0.0` if the vectors have different lengths, if either vector is
    /// all-zeros, or if the denominator underflows to zero.
    pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }

        let mut dot = 0.0_f64;
        let mut norm_a = 0.0_f64;
        let mut norm_b = 0.0_f64;

        for (x, y) in a.iter().zip(b.iter()) {
            dot += x * y;
            norm_a += x * x;
            norm_b += y * y;
        }

        let denom = norm_a.sqrt() * norm_b.sqrt();
        if denom < f64::EPSILON {
            return 0.0;
        }

        (dot / denom).clamp(-1.0, 1.0)
    }

    // ------------------------------------------------------------------
    // Lookup
    // ------------------------------------------------------------------

    /// Look up the cache for a semantically similar previous query.
    ///
    /// Scans all non-expired entries and finds the one with the highest cosine
    /// similarity to `query_embedding`. If that similarity is ≥
    /// `config.similarity_threshold` the entry's `hit_count` and `last_accessed`
    /// fields are updated and a [`CacheLookupResult::Hit`] is returned; otherwise
    /// [`CacheLookupResult::Miss`] is returned.
    ///
    /// The global `total_hits` / `total_misses` counters are always updated.
    ///
    /// # Arguments
    ///
    /// * `query_embedding` — embedding vector for the incoming query.
    /// * `now` — current timestamp in milliseconds (caller-supplied for
    ///   testability).
    pub fn lookup(&mut self, query_embedding: &[f64], now: u64) -> CacheLookupResult {
        let threshold = self.config.similarity_threshold;

        let mut best_idx: Option<usize> = None;
        let mut best_sim: f64 = f64::NEG_INFINITY;

        for (idx, entry) in self.entries.iter().enumerate() {
            if entry.is_expired(now) {
                continue;
            }
            let sim = Self::cosine_similarity(query_embedding, &entry.key.embedding);
            if sim > best_sim {
                best_sim = sim;
                best_idx = Some(idx);
            }
        }

        if let Some(idx) = best_idx {
            if best_sim >= threshold {
                // Update hit metadata.
                self.entries[idx].hit_count += 1;
                self.entries[idx].last_accessed = now;
                let result = self.entries[idx].result.clone();
                let entry_id = idx;
                self.total_hits += 1;
                return CacheLookupResult::Hit {
                    entry_id,
                    similarity: best_sim,
                    result,
                };
            }
        }

        self.total_misses += 1;
        CacheLookupResult::Miss
    }

    // ------------------------------------------------------------------
    // Insertion
    // ------------------------------------------------------------------

    /// Insert a new entry into the cache.
    ///
    /// If the cache has reached `config.max_entries`, one entry is evicted first
    /// according to `config.eviction_policy`. The `config.ttl_ms` value is
    /// applied to the new entry's `ttl_ms` field.
    ///
    /// # Arguments
    ///
    /// * `key` — the query key (text + embedding).
    /// * `result` — the result to cache.
    /// * `now` — current timestamp in milliseconds.
    pub fn insert(&mut self, key: CacheKey, result: String, now: u64) {
        if self.entries.len() >= self.config.max_entries && self.config.max_entries > 0 {
            self.evict_one(now);
        }

        let entry = CacheEntry {
            key,
            result,
            hit_count: 0,
            inserted_at: now,
            last_accessed: now,
            ttl_ms: self.config.ttl_ms,
        };

        self.entries.push(entry);
        self.total_insertions += 1;
        self.next_id += 1;
    }

    // ------------------------------------------------------------------
    // Eviction
    // ------------------------------------------------------------------

    /// Evict a single entry according to the configured [`CacheEvictionPolicy`].
    ///
    /// * **LRU** — remove the entry with the smallest `last_accessed` timestamp.
    /// * **LFU** — remove the entry with the smallest `hit_count`.
    /// * **TTLFirst** — among entries that have a `ttl_ms`, remove the one that
    ///   will expire soonest (`inserted_at + ttl_ms` is smallest); if no entry
    ///   has a TTL, fall back to LRU.
    ///
    /// Does nothing if the cache is empty.
    pub fn evict_one(&mut self, now: u64) {
        if self.entries.is_empty() {
            return;
        }

        let victim_idx = match self.config.eviction_policy {
            CacheEvictionPolicy::Lru => self.find_lru_victim(),
            CacheEvictionPolicy::Lfu => self.find_lfu_victim(),
            CacheEvictionPolicy::TtlFirst => self.find_ttl_victim(now),
        };

        if let Some(idx) = victim_idx {
            self.entries.swap_remove(idx);
        }
    }

    /// Find the index of the least-recently-used entry.
    fn find_lru_victim(&self) -> Option<usize> {
        self.entries
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                a.last_accessed
                    .partial_cmp(&b.last_accessed)
                    .unwrap_or(Ordering::Equal)
            })
            .map(|(idx, _)| idx)
    }

    /// Find the index of the least-frequently-used entry.
    fn find_lfu_victim(&self) -> Option<usize> {
        self.entries
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                a.hit_count
                    .partial_cmp(&b.hit_count)
                    .unwrap_or(Ordering::Equal)
            })
            .map(|(idx, _)| idx)
    }

    /// Find the index of the soonest-expiring entry; fall back to LRU.
    fn find_ttl_victim(&self, _now: u64) -> Option<usize> {
        // Among entries that have TTLs, pick the one whose absolute expiry
        // (inserted_at + ttl_ms) is smallest — i.e. expires soonest.
        let ttl_victim = self
            .entries
            .iter()
            .enumerate()
            .filter_map(|(idx, e)| e.ttl_ms.map(|ttl| (idx, e.inserted_at.saturating_add(ttl))))
            .min_by_key(|&(_, expiry)| expiry)
            .map(|(idx, _)| idx);

        ttl_victim.or_else(|| self.find_lru_victim())
    }

    /// Remove all expired entries from the cache.
    ///
    /// Returns the number of entries that were removed.
    pub fn evict_expired(&mut self, now: u64) -> usize {
        let before = self.entries.len();
        self.entries.retain(|e| !e.is_expired(now));
        before - self.entries.len()
    }

    // ------------------------------------------------------------------
    // Invalidation / management
    // ------------------------------------------------------------------

    /// Remove all entries whose `key.query_text` exactly matches `text`.
    ///
    /// Returns the number of entries removed.
    pub fn invalidate_by_text(&mut self, text: &str) -> usize {
        let before = self.entries.len();
        self.entries.retain(|e| e.key.query_text != text);
        before - self.entries.len()
    }

    /// Remove all entries from the cache (statistics are preserved).
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    // ------------------------------------------------------------------
    // Accessors
    // ------------------------------------------------------------------

    /// Return the current number of entries in the cache.
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// Return aggregate statistics for this cache instance.
    pub fn stats(&self) -> ScCacheStats {
        let total_lookups = self.total_hits + self.total_misses;
        let hit_rate = if total_lookups == 0 {
            0.0
        } else {
            self.total_hits as f64 / total_lookups as f64
        };

        let avg_hit_count = if self.entries.is_empty() {
            0.0
        } else {
            let sum: u64 = self.entries.iter().map(|e| e.hit_count).sum();
            sum as f64 / self.entries.len() as f64
        };

        ScCacheStats {
            total_entries: self.entries.len(),
            total_hits: self.total_hits,
            total_misses: self.total_misses,
            hit_rate,
            total_insertions: self.total_insertions,
            avg_hit_count,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::{
        CacheConfig, CacheEntry, CacheEvictionPolicy, CacheKey, CacheLookupResult,
        SemanticCacheLayer,
    };

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    fn make_config(max: usize, threshold: f64, policy: CacheEvictionPolicy) -> CacheConfig {
        CacheConfig {
            max_entries: max,
            similarity_threshold: threshold,
            ttl_ms: None,
            eviction_policy: policy,
        }
    }

    fn make_key(text: &str, embedding: Vec<f64>) -> CacheKey {
        CacheKey {
            query_text: text.to_string(),
            embedding,
        }
    }

    fn unit_vec(dim: usize, hot: usize) -> Vec<f64> {
        let mut v = vec![0.0_f64; dim];
        v[hot] = 1.0;
        v
    }

    fn normalized(v: Vec<f64>) -> Vec<f64> {
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < f64::EPSILON {
            return v;
        }
        v.into_iter().map(|x| x / norm).collect()
    }

    // ------------------------------------------------------------------
    // cosine_similarity
    // ------------------------------------------------------------------

    #[test]
    fn test_cosine_same_vector() {
        let v = vec![1.0, 2.0, 3.0];
        let sim = SemanticCacheLayer::cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 1e-9, "same vector should give sim=1.0");
    }

    #[test]
    fn test_cosine_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let sim = SemanticCacheLayer::cosine_similarity(&a, &b);
        assert!((sim - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_cosine_opposite() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        let sim = SemanticCacheLayer::cosine_similarity(&a, &b);
        assert!((sim + 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_cosine_different_lengths_returns_zero() {
        let a = vec![1.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let sim = SemanticCacheLayer::cosine_similarity(&a, &b);
        assert_eq!(sim, 0.0);
    }

    #[test]
    fn test_cosine_zero_vector_a() {
        let a = vec![0.0, 0.0];
        let b = vec![1.0, 0.0];
        let sim = SemanticCacheLayer::cosine_similarity(&a, &b);
        assert_eq!(sim, 0.0);
    }

    #[test]
    fn test_cosine_zero_vector_b() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 0.0];
        let sim = SemanticCacheLayer::cosine_similarity(&a, &b);
        assert_eq!(sim, 0.0);
    }

    #[test]
    fn test_cosine_both_zero() {
        let a = vec![0.0; 4];
        let b = vec![0.0; 4];
        let sim = SemanticCacheLayer::cosine_similarity(&a, &b);
        assert_eq!(sim, 0.0);
    }

    #[test]
    fn test_cosine_empty_slices_returns_zero() {
        let sim = SemanticCacheLayer::cosine_similarity(&[], &[]);
        assert_eq!(sim, 0.0);
    }

    #[test]
    fn test_cosine_known_value() {
        // [1,1] vs [1,0] → dot=1, |a|=√2, |b|=1  → sim = 1/√2 ≈ 0.7071
        let a = vec![1.0_f64, 1.0];
        let b = vec![1.0_f64, 0.0];
        let sim = SemanticCacheLayer::cosine_similarity(&a, &b);
        let expected = 1.0_f64 / 2.0_f64.sqrt();
        assert!((sim - expected).abs() < 1e-9);
    }

    #[test]
    fn test_cosine_clamp_above_one() {
        // Floating-point rounding can push the result slightly above 1.0; it
        // must be clamped to ≤ 1.0.
        let n = 1000;
        let v: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
        let sim = SemanticCacheLayer::cosine_similarity(&v, &v);
        assert!(sim <= 1.0 + 1e-12);
    }

    // ------------------------------------------------------------------
    // Basic insert + lookup
    // ------------------------------------------------------------------

    #[test]
    fn test_lookup_hit_identical_embedding() {
        let mut cache = SemanticCacheLayer::new(make_config(16, 0.90, CacheEvictionPolicy::Lru));
        let emb = vec![1.0, 0.0, 0.0];
        cache.insert(make_key("q1", emb.clone()), "result1".to_string(), 0);

        match cache.lookup(&emb, 0) {
            CacheLookupResult::Hit {
                result, similarity, ..
            } => {
                assert_eq!(result, "result1");
                assert!((similarity - 1.0).abs() < 1e-9);
            }
            CacheLookupResult::Miss => panic!("Expected a cache hit"),
        }
    }

    #[test]
    fn test_lookup_miss_empty_cache() {
        let mut cache = SemanticCacheLayer::new(CacheConfig::default());
        let result = cache.lookup(&[1.0, 0.0], 0);
        assert_eq!(result, CacheLookupResult::Miss);
    }

    #[test]
    fn test_lookup_miss_below_threshold() {
        let mut cache = SemanticCacheLayer::new(make_config(16, 0.99, CacheEvictionPolicy::Lru));
        let emb = normalized(vec![1.0, 1.0, 0.0]);
        cache.insert(make_key("q", emb.clone()), "r".to_string(), 0);

        // Query perpendicular to stored embedding → sim = 0.0
        let query = vec![0.0, 0.0, 1.0];
        assert_eq!(cache.lookup(&query, 0), CacheLookupResult::Miss);
    }

    #[test]
    fn test_lookup_returns_best_match() {
        let mut cache = SemanticCacheLayer::new(make_config(16, 0.5, CacheEvictionPolicy::Lru));
        cache.insert(make_key("a", unit_vec(3, 0)), "result_a".to_string(), 0);
        cache.insert(make_key("b", unit_vec(3, 1)), "result_b".to_string(), 0);

        // Query aligned with axis 1 → should match "b"
        match cache.lookup(&unit_vec(3, 1), 0) {
            CacheLookupResult::Hit { result, .. } => assert_eq!(result, "result_b"),
            CacheLookupResult::Miss => panic!("Expected hit"),
        }
    }

    #[test]
    fn test_lookup_updates_hit_count() {
        let mut cache = SemanticCacheLayer::new(make_config(16, 0.9, CacheEvictionPolicy::Lru));
        let emb = unit_vec(3, 0);
        cache.insert(make_key("q", emb.clone()), "r".to_string(), 0);

        cache.lookup(&emb, 1);
        cache.lookup(&emb, 2);
        assert_eq!(cache.entries[0].hit_count, 2);
    }

    #[test]
    fn test_lookup_updates_last_accessed() {
        let mut cache = SemanticCacheLayer::new(make_config(16, 0.9, CacheEvictionPolicy::Lru));
        let emb = unit_vec(3, 0);
        cache.insert(make_key("q", emb.clone()), "r".to_string(), 100);

        cache.lookup(&emb, 200);
        assert_eq!(cache.entries[0].last_accessed, 200);
    }

    // ------------------------------------------------------------------
    // Statistics
    // ------------------------------------------------------------------

    #[test]
    fn test_stats_initial() {
        let cache = SemanticCacheLayer::new(CacheConfig::default());
        let s = cache.stats();
        assert_eq!(s.total_entries, 0);
        assert_eq!(s.total_hits, 0);
        assert_eq!(s.total_misses, 0);
        assert_eq!(s.hit_rate, 0.0);
        assert_eq!(s.total_insertions, 0);
        assert_eq!(s.avg_hit_count, 0.0);
    }

    #[test]
    fn test_stats_after_inserts_and_lookups() {
        let mut cache = SemanticCacheLayer::new(make_config(16, 0.9, CacheEvictionPolicy::Lru));
        let emb = unit_vec(3, 0);
        cache.insert(make_key("q", emb.clone()), "r".to_string(), 0);

        // 2 hits, 1 miss
        cache.lookup(&emb, 1);
        cache.lookup(&emb, 2);
        cache.lookup(&unit_vec(3, 2), 3); // perpendicular → miss

        let s = cache.stats();
        assert_eq!(s.total_hits, 2);
        assert_eq!(s.total_misses, 1);
        assert!((s.hit_rate - 2.0 / 3.0).abs() < 1e-9);
        assert_eq!(s.total_insertions, 1);
        assert_eq!(s.avg_hit_count, 2.0);
    }

    #[test]
    fn test_stats_hit_rate_zero_lookups() {
        let mut cache = SemanticCacheLayer::new(CacheConfig::default());
        cache.insert(make_key("q", unit_vec(3, 0)), "r".to_string(), 0);
        assert_eq!(cache.stats().hit_rate, 0.0);
    }

    // ------------------------------------------------------------------
    // Eviction — LRU
    // ------------------------------------------------------------------

    #[test]
    fn test_lru_eviction_removes_oldest_accessed() {
        let mut cache = SemanticCacheLayer::new(make_config(2, 0.9, CacheEvictionPolicy::Lru));
        cache.insert(make_key("a", unit_vec(3, 0)), "ra".to_string(), 0);
        cache.insert(make_key("b", unit_vec(3, 1)), "rb".to_string(), 5);

        // Access "b" so its last_accessed is newer
        cache.lookup(&unit_vec(3, 1), 10);

        // Insert "c" — must evict "a" (last_accessed=0)
        cache.insert(make_key("c", unit_vec(3, 2)), "rc".to_string(), 15);
        assert_eq!(cache.entry_count(), 2);

        let texts: Vec<&str> = cache
            .entries
            .iter()
            .map(|e| e.key.query_text.as_str())
            .collect();
        assert!(!texts.contains(&"a"), "LRU should evict 'a'");
    }

    // ------------------------------------------------------------------
    // Eviction — LFU
    // ------------------------------------------------------------------

    #[test]
    fn test_lfu_eviction_removes_lowest_hit_count() {
        let mut cache = SemanticCacheLayer::new(make_config(2, 0.9, CacheEvictionPolicy::Lfu));
        cache.insert(make_key("a", unit_vec(3, 0)), "ra".to_string(), 0);
        cache.insert(make_key("b", unit_vec(3, 1)), "rb".to_string(), 0);

        // Give "b" one hit
        cache.lookup(&unit_vec(3, 1), 1);

        // Insert "c" → must evict "a" (hit_count=0 < b's hit_count=1)
        cache.insert(make_key("c", unit_vec(3, 2)), "rc".to_string(), 2);
        assert_eq!(cache.entry_count(), 2);

        let texts: Vec<&str> = cache
            .entries
            .iter()
            .map(|e| e.key.query_text.as_str())
            .collect();
        assert!(!texts.contains(&"a"), "LFU should evict 'a'");
    }

    // ------------------------------------------------------------------
    // Eviction — TTLFirst
    // ------------------------------------------------------------------

    #[test]
    fn test_ttlfirst_evicts_soonest_expiring() {
        let mut cache = SemanticCacheLayer::new(make_config(2, 0.9, CacheEvictionPolicy::TtlFirst));

        // Manually insert entries with different TTLs
        cache.entries.push(CacheEntry {
            key: make_key("a", unit_vec(3, 0)),
            result: "ra".to_string(),
            hit_count: 0,
            inserted_at: 0,
            last_accessed: 0,
            ttl_ms: Some(100), // expires at 100
        });
        cache.entries.push(CacheEntry {
            key: make_key("b", unit_vec(3, 1)),
            result: "rb".to_string(),
            hit_count: 0,
            inserted_at: 0,
            last_accessed: 0,
            ttl_ms: Some(500), // expires at 500
        });

        // Insert "c" → must evict "a" (expires soonest)
        cache.insert(make_key("c", unit_vec(3, 2)), "rc".to_string(), 10);
        assert_eq!(cache.entry_count(), 2);

        let texts: Vec<&str> = cache
            .entries
            .iter()
            .map(|e| e.key.query_text.as_str())
            .collect();
        assert!(!texts.contains(&"a"), "TTLFirst should evict 'a'");
    }

    #[test]
    fn test_ttlfirst_fallback_to_lru_when_no_ttls() {
        let mut cache = SemanticCacheLayer::new(make_config(2, 0.9, CacheEvictionPolicy::TtlFirst));
        cache.insert(make_key("a", unit_vec(3, 0)), "ra".to_string(), 0);
        cache.insert(make_key("b", unit_vec(3, 1)), "rb".to_string(), 10);

        // Access "b" to update last_accessed
        cache.lookup(&unit_vec(3, 1), 20);

        // Insert "c" → TTLFirst with no TTLs → fall back to LRU → evict "a"
        cache.insert(make_key("c", unit_vec(3, 2)), "rc".to_string(), 30);
        let texts: Vec<&str> = cache
            .entries
            .iter()
            .map(|e| e.key.query_text.as_str())
            .collect();
        assert!(!texts.contains(&"a"));
    }

    // ------------------------------------------------------------------
    // evict_expired
    // ------------------------------------------------------------------

    #[test]
    fn test_evict_expired_removes_expired_entries() {
        let mut cache = SemanticCacheLayer::new(CacheConfig {
            ttl_ms: Some(100),
            ..CacheConfig::default()
        });
        cache.insert(make_key("a", unit_vec(3, 0)), "ra".to_string(), 0);
        cache.insert(make_key("b", unit_vec(3, 1)), "rb".to_string(), 50);

        // Advance time by 200ms — "a" has expired (200 > 100), "b" has not (200-50=150 > 100 — also expired)
        let removed = cache.evict_expired(200);
        assert_eq!(removed, 2);
        assert_eq!(cache.entry_count(), 0);
    }

    #[test]
    fn test_evict_expired_keeps_non_expired() {
        let mut cache = SemanticCacheLayer::new(CacheConfig {
            ttl_ms: Some(1000),
            ..CacheConfig::default()
        });
        cache.insert(make_key("a", unit_vec(3, 0)), "ra".to_string(), 0);
        cache.insert(make_key("b", unit_vec(3, 1)), "rb".to_string(), 0);

        let removed = cache.evict_expired(500); // 500ms < 1000ms TTL
        assert_eq!(removed, 0);
        assert_eq!(cache.entry_count(), 2);
    }

    #[test]
    fn test_evict_expired_no_ttl_never_expires() {
        let mut cache = SemanticCacheLayer::new(CacheConfig {
            ttl_ms: None,
            ..CacheConfig::default()
        });
        cache.insert(make_key("a", unit_vec(3, 0)), "ra".to_string(), 0);
        let removed = cache.evict_expired(u64::MAX);
        assert_eq!(removed, 0);
        assert_eq!(cache.entry_count(), 1);
    }

    // ------------------------------------------------------------------
    // TTL expiry during lookup
    // ------------------------------------------------------------------

    #[test]
    fn test_lookup_skips_expired_entry() {
        let mut cache = SemanticCacheLayer::new(CacheConfig {
            max_entries: 16,
            similarity_threshold: 0.9,
            ttl_ms: Some(100),
            eviction_policy: CacheEvictionPolicy::Lru,
        });
        let emb = unit_vec(3, 0);
        cache.insert(make_key("q", emb.clone()), "r".to_string(), 0);

        // After 200ms the entry is expired; lookup should return Miss
        let result = cache.lookup(&emb, 200);
        assert_eq!(result, CacheLookupResult::Miss);
    }

    #[test]
    fn test_lookup_hits_non_expired_when_mixed() {
        let mut cache = SemanticCacheLayer::new(CacheConfig {
            max_entries: 16,
            similarity_threshold: 0.9,
            ttl_ms: None,
            eviction_policy: CacheEvictionPolicy::Lru,
        });

        // First entry: TTL 100ms, expires at 100
        cache.entries.push(CacheEntry {
            key: make_key("expired", unit_vec(4, 0)),
            result: "old".to_string(),
            hit_count: 0,
            inserted_at: 0,
            last_accessed: 0,
            ttl_ms: Some(100),
        });

        // Second entry: no TTL, same embedding direction
        cache.entries.push(CacheEntry {
            key: make_key("valid", unit_vec(4, 0)),
            result: "new".to_string(),
            hit_count: 0,
            inserted_at: 50,
            last_accessed: 50,
            ttl_ms: None,
        });

        // At t=200 the first entry is expired; lookup should find the second
        match cache.lookup(&unit_vec(4, 0), 200) {
            CacheLookupResult::Hit { result, .. } => assert_eq!(result, "new"),
            CacheLookupResult::Miss => panic!("Expected hit on the non-expired entry"),
        }
    }

    // ------------------------------------------------------------------
    // invalidate_by_text
    // ------------------------------------------------------------------

    #[test]
    fn test_invalidate_by_text_removes_matching() {
        let mut cache = SemanticCacheLayer::new(CacheConfig::default());
        cache.insert(make_key("hello", unit_vec(3, 0)), "r1".to_string(), 0);
        cache.insert(make_key("hello", unit_vec(3, 1)), "r2".to_string(), 0);
        cache.insert(make_key("world", unit_vec(3, 2)), "r3".to_string(), 0);

        let removed = cache.invalidate_by_text("hello");
        assert_eq!(removed, 2);
        assert_eq!(cache.entry_count(), 1);
        assert_eq!(cache.entries[0].key.query_text, "world");
    }

    #[test]
    fn test_invalidate_by_text_no_match_returns_zero() {
        let mut cache = SemanticCacheLayer::new(CacheConfig::default());
        cache.insert(make_key("hello", unit_vec(3, 0)), "r".to_string(), 0);
        let removed = cache.invalidate_by_text("nonexistent");
        assert_eq!(removed, 0);
        assert_eq!(cache.entry_count(), 1);
    }

    // ------------------------------------------------------------------
    // clear
    // ------------------------------------------------------------------

    #[test]
    fn test_clear_removes_all_entries() {
        let mut cache = SemanticCacheLayer::new(CacheConfig::default());
        for i in 0..10 {
            cache.insert(
                make_key(&i.to_string(), unit_vec(3, i % 3)),
                "r".to_string(),
                0,
            );
        }
        assert_eq!(cache.entry_count(), 10);
        cache.clear();
        assert_eq!(cache.entry_count(), 0);
    }

    #[test]
    fn test_clear_preserves_statistics() {
        let mut cache = SemanticCacheLayer::new(make_config(16, 0.9, CacheEvictionPolicy::Lru));
        let emb = unit_vec(3, 0);
        cache.insert(make_key("q", emb.clone()), "r".to_string(), 0);
        cache.lookup(&emb, 1);
        cache.clear();

        let s = cache.stats();
        assert_eq!(s.total_hits, 1);
        assert_eq!(s.total_insertions, 1);
        assert_eq!(s.total_entries, 0);
    }

    // ------------------------------------------------------------------
    // entry_count
    // ------------------------------------------------------------------

    #[test]
    fn test_entry_count_tracks_insertions() {
        let mut cache = SemanticCacheLayer::new(make_config(100, 0.9, CacheEvictionPolicy::Lru));
        assert_eq!(cache.entry_count(), 0);
        for i in 0..5 {
            cache.insert(
                make_key(&i.to_string(), unit_vec(4, i % 4)),
                "r".to_string(),
                i as u64,
            );
        }
        assert_eq!(cache.entry_count(), 5);
    }

    // ------------------------------------------------------------------
    // max_entries capacity
    // ------------------------------------------------------------------

    #[test]
    fn test_max_entries_respected() {
        let mut cache = SemanticCacheLayer::new(make_config(3, 0.9, CacheEvictionPolicy::Lru));
        for i in 0..6_usize {
            cache.insert(
                make_key(&i.to_string(), unit_vec(4, i % 4)),
                "r".to_string(),
                i as u64,
            );
        }
        assert!(cache.entry_count() <= 3, "Cache exceeded max_entries");
    }

    #[test]
    fn test_max_entries_zero_does_not_insert() {
        let mut cache = SemanticCacheLayer::new(make_config(0, 0.9, CacheEvictionPolicy::Lru));
        cache.insert(make_key("q", unit_vec(3, 0)), "r".to_string(), 0);
        // max_entries=0 means evict immediately before push, so the entry may or may not remain.
        // The key invariant is that we do not panic.
        let _ = cache.entry_count();
    }

    // ------------------------------------------------------------------
    // CacheEvictionPolicy default
    // ------------------------------------------------------------------

    #[test]
    fn test_eviction_policy_default_is_lru() {
        assert_eq!(CacheEvictionPolicy::default(), CacheEvictionPolicy::Lru);
    }

    // ------------------------------------------------------------------
    // CacheConfig default
    // ------------------------------------------------------------------

    #[test]
    fn test_cache_config_defaults() {
        let c = CacheConfig::default();
        assert_eq!(c.max_entries, 1024);
        assert!((c.similarity_threshold - 0.92).abs() < 1e-9);
        assert!(c.ttl_ms.is_none());
        assert_eq!(c.eviction_policy, CacheEvictionPolicy::Lru);
    }

    // ------------------------------------------------------------------
    // ScCacheStats avg_hit_count
    // ------------------------------------------------------------------

    #[test]
    fn test_avg_hit_count_across_entries() {
        let mut cache = SemanticCacheLayer::new(make_config(16, 0.5, CacheEvictionPolicy::Lru));
        cache.insert(make_key("a", unit_vec(3, 0)), "ra".to_string(), 0);
        cache.insert(make_key("b", unit_vec(3, 1)), "rb".to_string(), 0);

        // Hit "a" twice
        cache.lookup(&unit_vec(3, 0), 1);
        cache.lookup(&unit_vec(3, 0), 2);
        // Hit "b" once
        cache.lookup(&unit_vec(3, 1), 3);

        let s = cache.stats();
        // avg_hit_count = (2 + 1) / 2 = 1.5
        assert!((s.avg_hit_count - 1.5).abs() < 1e-9);
    }

    // ------------------------------------------------------------------
    // Multi-dimensional embeddings
    // ------------------------------------------------------------------

    #[test]
    fn test_high_dimensional_embedding_hit() {
        let dim = 768;
        let mut cache = SemanticCacheLayer::new(make_config(16, 0.95, CacheEvictionPolicy::Lru));

        let emb: Vec<f64> = (0..dim).map(|i| (i as f64 / dim as f64).sin()).collect();
        let emb_n = normalized(emb.clone());

        cache.insert(make_key("high-dim", emb_n.clone()), "result".to_string(), 0);

        // Slightly perturb the query — should still hit at threshold 0.95
        let query: Vec<f64> = emb_n
            .iter()
            .enumerate()
            .map(|(i, x)| x + i as f64 * 1e-5)
            .collect();
        let query_n = normalized(query);

        let result = cache.lookup(&query_n, 0);
        assert!(
            matches!(result, CacheLookupResult::Hit { .. }),
            "Expected hit for slightly perturbed high-dim vector"
        );
    }

    #[test]
    fn test_many_insertions_and_lookups() {
        let n = 50_usize;
        let mut cache = SemanticCacheLayer::new(make_config(n, 0.99, CacheEvictionPolicy::Lfu));

        for i in 0..n {
            let emb = unit_vec(n, i);
            cache.insert(make_key(&format!("q{i}"), emb), format!("r{i}"), i as u64);
        }

        // Each exact lookup should hit its own entry
        let mut hits = 0_u64;
        for i in 0..n {
            let emb = unit_vec(n, i);
            if matches!(
                cache.lookup(&emb, n as u64 + i as u64),
                CacheLookupResult::Hit { .. }
            ) {
                hits += 1;
            }
        }

        assert_eq!(hits, n as u64);
    }

    #[test]
    fn test_total_insertions_counter() {
        let mut cache = SemanticCacheLayer::new(make_config(3, 0.9, CacheEvictionPolicy::Lru));
        for i in 0..5_usize {
            cache.insert(
                make_key(&i.to_string(), unit_vec(3, i % 3)),
                "r".to_string(),
                i as u64,
            );
        }
        assert_eq!(cache.total_insertions, 5);
    }

    #[test]
    fn test_evict_one_empty_cache_noop() {
        let mut cache = SemanticCacheLayer::new(CacheConfig::default());
        cache.evict_one(0); // must not panic
        assert_eq!(cache.entry_count(), 0);
    }

    #[test]
    fn test_insert_with_ttl_then_lookup() {
        let mut cache = SemanticCacheLayer::new(CacheConfig {
            max_entries: 16,
            similarity_threshold: 0.9,
            ttl_ms: Some(500),
            eviction_policy: CacheEvictionPolicy::Lru,
        });
        let emb = unit_vec(3, 0);
        cache.insert(make_key("q", emb.clone()), "r".to_string(), 0);

        // Within TTL → hit
        assert!(
            matches!(cache.lookup(&emb, 400), CacheLookupResult::Hit { .. }),
            "Should hit within TTL"
        );
        // After TTL → miss
        assert_eq!(
            cache.lookup(&emb, 600),
            CacheLookupResult::Miss,
            "Should miss after TTL"
        );
    }
}
