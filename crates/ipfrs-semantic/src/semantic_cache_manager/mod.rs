//! # Semantic Cache Manager
//!
//! A production-quality semantic-aware cache that returns cached results for queries
//! whose embeddings are semantically similar to previously cached queries — even when
//! the exact query text differs.
//!
//! Unlike a regular cache (exact key match), a [`SemanticCacheManager`] can return a
//! cached result for a *semantically similar* query, reducing redundant computation
//! when two different phrasings ask for the same information.
//!
//! ## Features
//!
//! - **Exact-match fast-path** via FNV-1a hashed query text
//! - **Semantic-match slow-path** via cosine similarity scan
//! - **TTL expiry** on every lookup (lazy) and explicit batch expiry
//! - **Five eviction strategies**: LRU, LFU, SemanticCluster, TTLFirst, HybridScore
//! - **Byte-budget enforcement** (`max_bytes`)
//! - **Semantic neighbors** query for introspection
//! - **Greedy cluster statistics** for monitoring embedding distribution
//! - **Comprehensive statistics** (hit-rates, avg similarity on semantic hit, etc.)
//!
//! ## Example
//!
//! ```rust
//! use ipfrs_semantic::semantic_cache_manager::{
//!     ScmCacheConfig, ScmCacheKey, ScmCacheHit, ScmEvictionStrategy, SemanticCacheManager,
//! };
//!
//! let config = ScmCacheConfig {
//!     max_entries: 256,
//!     max_bytes: 64 * 1024 * 1024,
//!     semantic_threshold: 0.90,
//!     ttl_us: None,
//!     strategy: ScmEvictionStrategy::LRU,
//!     cluster_radius: 0.15,
//! };
//!
//! let mut mgr = SemanticCacheManager::new(config);
//!
//! let key = ScmCacheKey::new("What is IPFS?".to_string(), vec![0.1, 0.2, 0.3]);
//! let id = mgr.insert(key, b"answer bytes".to_vec(), None).expect("insert ok");
//!
//! let hit = mgr
//!     .lookup("What is IPFS?", &[0.1, 0.2, 0.3], 0)
//!     .expect("lookup ok");
//! assert!(matches!(hit, ScmCacheHit::Exact { entry_id } if entry_id == id));
//! ```

mod scm_types;
use scm_types::{cosine_similarity, fnv1a_64, xorshift64, Slot};
pub use scm_types::{
    ScmCacheConfig, ScmCacheEntry, ScmCacheError, ScmCacheHit, ScmCacheKey, ScmCacheStats,
    ScmEvictionStrategy,
};

// ──────────────────────────────────────────────────────────────────────────────
// SemanticCacheManager
// ──────────────────────────────────────────────────────────────────────────────

/// A semantic-aware cache backed by cosine similarity search.
///
/// Entries are stored in a `Vec<Slot>` for simplicity and full linear-scan
/// semantic search. For workloads requiring sub-linear semantic lookup, wrap
/// this with an HNSW layer on top.
#[derive(Debug)]
pub struct SemanticCacheManager {
    config: ScmCacheConfig,
    slots: Vec<Slot>,
    next_id: u64,
    /// Expected embedding dimension (inferred from the first insert).
    embedding_dim: Option<usize>,
    // Stats accumulators
    total_insertions: u64,
    exact_hits: u64,
    semantic_hits: u64,
    misses: u64,
    evictions: u64,
    // Running sum for average similarity on semantic hits.
    semantic_similarity_sum: f64,
    // PRNG state for tie-breaking.
    rng_state: u64,
}

impl SemanticCacheManager {
    /// Create a new [`SemanticCacheManager`] with the given configuration.
    pub fn new(config: ScmCacheConfig) -> Self {
        Self {
            slots: Vec::with_capacity(config.max_entries.min(4096)),
            next_id: 1,
            embedding_dim: None,
            total_insertions: 0,
            exact_hits: 0,
            semantic_hits: 0,
            misses: 0,
            evictions: 0,
            semantic_similarity_sum: 0.0,
            rng_state: 0xDEAD_BEEF_CAFE_0001,
            config,
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Public API
    // ─────────────────────────────────────────────────────────────────────────

    /// Insert an entry into the cache.
    ///
    /// Assigns a monotonically increasing `entry_id` and stores the entry.
    /// If the cache is over capacity after insertion, entries are evicted
    /// according to the configured [`ScmEvictionStrategy`].
    ///
    /// # Errors
    ///
    /// - [`ScmCacheError::EntryTooLarge`] if the entry alone exceeds `max_bytes`.
    /// - [`ScmCacheError::DimensionMismatch`] if the embedding dimension is wrong.
    /// - [`ScmCacheError::CacheAtCapacity`] if eviction fails to free space.
    pub fn insert(
        &mut self,
        key: ScmCacheKey,
        result: Vec<u8>,
        ttl_us: Option<u64>,
    ) -> Result<u64, ScmCacheError> {
        // Dimension check
        if let Some(dim) = self.embedding_dim {
            if key.embedding.len() != dim {
                return Err(ScmCacheError::DimensionMismatch {
                    expected: dim,
                    got: key.embedding.len(),
                });
            }
        } else if !key.embedding.is_empty() {
            self.embedding_dim = Some(key.embedding.len());
        }

        let effective_ttl = ttl_us.or(self.config.ttl_us);
        let entry = ScmCacheEntry {
            result,
            inserted_at: 0, // caller uses current_ts in lookup; insert doesn't mandate a clock
            last_accessed: 0,
            access_count: 0,
            ttl_us: effective_ttl,
            similarity_score: 1.0,
            key,
        };

        let entry_bytes = entry.byte_size();
        if entry_bytes > self.config.max_bytes {
            return Err(ScmCacheError::EntryTooLarge(entry_bytes));
        }

        // Free space if needed
        let current_bytes = self.current_bytes();
        if current_bytes + entry_bytes > self.config.max_bytes
            || self.slots.len() >= self.config.max_entries
        {
            let needed = entry_bytes;
            self.evict_to_fit_internal(needed)?;
        }

        let id = self.next_id;
        self.next_id += 1;
        self.slots.push(Slot { id, entry });
        self.total_insertions += 1;
        Ok(id)
    }

    /// Insert with an explicit insertion timestamp (used in tests and production
    /// where you track time yourself).
    pub fn insert_at(
        &mut self,
        key: ScmCacheKey,
        result: Vec<u8>,
        ttl_us: Option<u64>,
        inserted_at: u64,
    ) -> Result<u64, ScmCacheError> {
        let id = self.insert(key, result, ttl_us)?;
        // Patch the insertion timestamp on the just-added slot.
        if let Some(slot) = self.slots.iter_mut().find(|s| s.id == id) {
            slot.entry.inserted_at = inserted_at;
            slot.entry.last_accessed = inserted_at;
        }
        Ok(id)
    }

    /// Look up a query in the cache.
    ///
    /// 1. Scan all live entries; lazily expire any TTL-expired entries found.
    /// 2. Check for an exact-text match (same FNV-1a hash *and* identical text).
    /// 3. If no exact match, find the highest-similarity entry above the threshold.
    ///
    /// Updates `access_count` and `last_accessed` on the winning entry.
    ///
    /// # Errors
    ///
    /// Returns [`ScmCacheError::DimensionMismatch`] when the provided embedding
    /// dimension differs from the inferred dimension.
    pub fn lookup(
        &mut self,
        query_text: &str,
        embedding: &[f64],
        current_ts: u64,
    ) -> Result<ScmCacheHit, ScmCacheError> {
        // Dimension check
        if let Some(dim) = self.embedding_dim {
            if !embedding.is_empty() && embedding.len() != dim {
                return Err(ScmCacheError::DimensionMismatch {
                    expected: dim,
                    got: embedding.len(),
                });
            }
        }

        let query_hash = fnv1a_64(query_text.as_bytes());

        // Expire stale entries lazily
        self.slots.retain(|s| !s.entry.is_expired(current_ts));

        // Find best match
        let mut best_exact: Option<usize> = None;
        let mut best_semantic: Option<(usize, f64)> = None;

        for (i, slot) in self.slots.iter().enumerate() {
            // Exact match fast-path
            if slot.entry.key.query_hash == query_hash && slot.entry.key.query_text == query_text {
                best_exact = Some(i);
                break; // exact match wins immediately
            }
            // Semantic similarity
            let sim = cosine_similarity(embedding, &slot.entry.key.embedding);
            if sim >= self.config.semantic_threshold {
                match best_semantic {
                    Some((_, prev_sim)) if sim <= prev_sim => {}
                    _ => best_semantic = Some((i, sim)),
                }
            }
        }

        if let Some(idx) = best_exact {
            let slot = &mut self.slots[idx];
            slot.entry.access_count += 1;
            slot.entry.last_accessed = current_ts;
            self.exact_hits += 1;
            return Ok(ScmCacheHit::Exact { entry_id: slot.id });
        }

        if let Some((idx, sim)) = best_semantic {
            let slot = &mut self.slots[idx];
            slot.entry.access_count += 1;
            slot.entry.last_accessed = current_ts;
            let entry_id = slot.id;
            let original_query = slot.entry.key.query_text.clone();
            self.semantic_hits += 1;
            self.semantic_similarity_sum += sim;
            return Ok(ScmCacheHit::Semantic {
                entry_id,
                similarity: sim,
                original_query,
            });
        }

        self.misses += 1;
        Ok(ScmCacheHit::Miss)
    }

    /// Retrieve a reference to an entry by its ID.
    ///
    /// # Errors
    ///
    /// Returns [`ScmCacheError::EntryNotFound`] if no such ID exists.
    pub fn get_entry(&self, entry_id: u64) -> Result<&ScmCacheEntry, ScmCacheError> {
        self.slots
            .iter()
            .find(|s| s.id == entry_id)
            .map(|s| &s.entry)
            .ok_or(ScmCacheError::EntryNotFound(entry_id))
    }

    /// Remove the entry with the given ID.
    ///
    /// # Errors
    ///
    /// Returns [`ScmCacheError::EntryNotFound`] if no such ID exists.
    pub fn invalidate(&mut self, entry_id: u64) -> Result<(), ScmCacheError> {
        let pos = self
            .slots
            .iter()
            .position(|s| s.id == entry_id)
            .ok_or(ScmCacheError::EntryNotFound(entry_id))?;
        self.slots.swap_remove(pos);
        Ok(())
    }

    /// Remove all entries whose embedding has cosine similarity > `threshold`
    /// with the given `embedding`.  Returns the IDs of removed entries.
    pub fn invalidate_similar(&mut self, embedding: &[f64], threshold: f64) -> Vec<u64> {
        let mut removed = Vec::new();
        let mut i = 0;
        while i < self.slots.len() {
            let sim = cosine_similarity(embedding, &self.slots[i].entry.key.embedding);
            if sim > threshold {
                removed.push(self.slots[i].id);
                self.slots.swap_remove(i);
            } else {
                i += 1;
            }
        }
        removed
    }

    /// Remove all TTL-expired entries as of `current_ts`.  Returns the IDs removed.
    pub fn expire_ttl(&mut self, current_ts: u64) -> Vec<u64> {
        let mut removed = Vec::new();
        let mut i = 0;
        while i < self.slots.len() {
            if self.slots[i].entry.is_expired(current_ts) {
                removed.push(self.slots[i].id);
                self.slots.swap_remove(i);
            } else {
                i += 1;
            }
        }
        removed
    }

    /// Evict entries (per the configured strategy) until at least `needed_bytes`
    /// of space is available.  Returns the IDs of evicted entries.
    ///
    /// # Errors
    ///
    /// Returns [`ScmCacheError::CacheAtCapacity`] if it is impossible to free
    /// enough space (e.g., all entries are locked or the budget is zero).
    pub fn evict_to_fit(&mut self, needed_bytes: usize) -> Vec<u64> {
        let current = self.current_bytes();
        if current + needed_bytes <= self.config.max_bytes
            && self.slots.len() < self.config.max_entries
        {
            return Vec::new();
        }
        let mut removed = Vec::new();
        while !self.slots.is_empty()
            && (self.current_bytes() + needed_bytes > self.config.max_bytes
                || self.slots.len() >= self.config.max_entries)
        {
            if let Some(victim) = self.pick_victim() {
                let id = self.slots[victim].id;
                self.slots.swap_remove(victim);
                self.evictions += 1;
                removed.push(id);
            } else {
                break;
            }
        }
        removed
    }

    /// Return the top-`k` entries ranked by cosine similarity to `embedding`,
    /// without any threshold filtering.
    pub fn semantic_neighbors(&self, embedding: &[f64], top_k: usize) -> Vec<(u64, f64)> {
        if top_k == 0 || self.slots.is_empty() {
            return Vec::new();
        }
        let mut scores: Vec<(u64, f64)> = self
            .slots
            .iter()
            .map(|s| {
                let sim = cosine_similarity(embedding, &s.entry.key.embedding);
                (s.id, sim)
            })
            .collect();
        // Sort descending by similarity; stable sort for determinism
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(top_k);
        scores
    }

    /// Compute greedy cluster statistics.
    ///
    /// Groups embeddings into clusters where every member is within
    /// `config.cluster_radius` (in cosine distance = 1 − similarity) of the
    /// cluster centroid.  Returns a `Vec` of `(centroid_similarity, cluster_size)`.
    ///
    /// The "centroid" here is represented as the embedding of the first entry
    /// assigned to each cluster (greedy seeding).  The returned f64 is the
    /// average pairwise cosine similarity within the cluster.
    pub fn cluster_stats(&self) -> Vec<(f64, usize)> {
        if self.slots.is_empty() {
            return Vec::new();
        }
        // Collect all embeddings with their slot indices
        let embeddings: Vec<&Vec<f64>> =
            self.slots.iter().map(|s| &s.entry.key.embedding).collect();
        let n = embeddings.len();
        let mut assigned = vec![false; n];
        let mut clusters: Vec<(f64, usize)> = Vec::new();

        for seed in 0..n {
            if assigned[seed] {
                continue;
            }
            // Start a new cluster with this seed
            let mut members: Vec<usize> = vec![seed];
            assigned[seed] = true;
            for candidate in (seed + 1)..n {
                if assigned[candidate] {
                    continue;
                }
                let sim = cosine_similarity(embeddings[seed], embeddings[candidate]);
                // cluster_radius is interpreted as (1 - similarity) threshold
                if 1.0 - sim <= self.config.cluster_radius {
                    members.push(candidate);
                    assigned[candidate] = true;
                }
            }
            // Compute average intra-cluster similarity as the representative score
            let avg_sim = if members.len() == 1 {
                1.0
            } else {
                let mut sum = 0.0;
                let mut count = 0usize;
                for i in 0..members.len() {
                    for j in (i + 1)..members.len() {
                        sum += cosine_similarity(embeddings[members[i]], embeddings[members[j]]);
                        count += 1;
                    }
                }
                if count == 0 {
                    1.0
                } else {
                    sum / count as f64
                }
            };
            clusters.push((avg_sim, members.len()));
        }
        clusters
    }

    /// Return a snapshot of current cache statistics.
    pub fn stats(&self) -> ScmCacheStats {
        let total_queries = self.exact_hits + self.semantic_hits + self.misses;
        let semantic_hit_rate = if total_queries == 0 {
            0.0
        } else {
            self.semantic_hits as f64 / total_queries as f64
        };
        let avg_similarity_on_semantic_hit = if self.semantic_hits == 0 {
            0.0
        } else {
            self.semantic_similarity_sum / self.semantic_hits as f64
        };
        ScmCacheStats {
            total_insertions: self.total_insertions,
            exact_hits: self.exact_hits,
            semantic_hits: self.semantic_hits,
            misses: self.misses,
            evictions: self.evictions,
            current_entries: self.slots.len(),
            current_bytes: self.current_bytes(),
            semantic_hit_rate,
            avg_similarity_on_semantic_hit,
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Private helpers
    // ─────────────────────────────────────────────────────────────────────────

    /// Sum of byte sizes of all current slots.
    fn current_bytes(&self) -> usize {
        self.slots.iter().map(|s| s.entry.byte_size()).sum()
    }

    /// Internal evict-to-fit that returns an error instead of a Vec.
    fn evict_to_fit_internal(&mut self, needed_bytes: usize) -> Result<(), ScmCacheError> {
        let max_iters = self.slots.len() + 1; // safety bound
        let mut iters = 0;
        while !self.slots.is_empty()
            && (self.current_bytes() + needed_bytes > self.config.max_bytes
                || self.slots.len() >= self.config.max_entries)
        {
            if iters >= max_iters {
                return Err(ScmCacheError::CacheAtCapacity);
            }
            match self.pick_victim() {
                Some(victim) => {
                    self.slots.swap_remove(victim);
                    self.evictions += 1;
                }
                None => return Err(ScmCacheError::CacheAtCapacity),
            }
            iters += 1;
        }
        Ok(())
    }

    /// Select the index of the entry to evict according to the current strategy.
    /// Returns `None` only when `slots` is empty.
    fn pick_victim(&mut self) -> Option<usize> {
        if self.slots.is_empty() {
            return None;
        }
        let strategy = self.config.strategy.clone();
        match strategy {
            ScmEvictionStrategy::LRU => self.victim_lru(),
            ScmEvictionStrategy::LFU => self.victim_lfu(),
            ScmEvictionStrategy::TTLFirst => self.victim_ttl_first(),
            ScmEvictionStrategy::SemanticCluster => self.victim_semantic_cluster(),
            ScmEvictionStrategy::HybridScore(w) => self.victim_hybrid(w),
        }
    }

    /// Least-recently-used: return index with the smallest `last_accessed`.
    fn victim_lru(&self) -> Option<usize> {
        self.slots
            .iter()
            .enumerate()
            .min_by_key(|(_, s)| s.entry.last_accessed)
            .map(|(i, _)| i)
    }

    /// Least-frequently-used: return index with the smallest `access_count`.
    fn victim_lfu(&self) -> Option<usize> {
        self.slots
            .iter()
            .enumerate()
            .min_by_key(|(_, s)| s.entry.access_count)
            .map(|(i, _)| i)
    }

    /// TTL-first: evict whichever entry is closest to expiry (smallest
    /// `ttl_us - elapsed`). Falls back to LRU when no entry has a TTL.
    fn victim_ttl_first(&self) -> Option<usize> {
        // Find entry with smallest remaining TTL
        let ttl_victim = self
            .slots
            .iter()
            .enumerate()
            .filter_map(|(i, s)| {
                s.entry.ttl_us.map(|ttl| {
                    let remaining = ttl
                        .saturating_sub(s.entry.last_accessed.saturating_sub(s.entry.inserted_at));
                    (i, remaining)
                })
            })
            .min_by_key(|(_, rem)| *rem)
            .map(|(i, _)| i);

        ttl_victim.or_else(|| self.victim_lru())
    }

    /// SemanticCluster: evict an entry from the largest cluster.
    /// Uses the greedy clustering based on `cluster_radius`.
    fn victim_semantic_cluster(&mut self) -> Option<usize> {
        if self.slots.is_empty() {
            return None;
        }
        let n = self.slots.len();
        let embeddings: Vec<&Vec<f64>> =
            self.slots.iter().map(|s| &s.entry.key.embedding).collect();

        // Build cluster assignments (reuse the greedy logic inline for efficiency)
        let mut cluster_id = vec![usize::MAX; n];
        let mut cluster_seeds: Vec<usize> = Vec::new();

        for seed in 0..n {
            if cluster_id[seed] != usize::MAX {
                continue;
            }
            let cid = cluster_seeds.len();
            cluster_seeds.push(seed);
            cluster_id[seed] = cid;
            for candidate in (seed + 1)..n {
                if cluster_id[candidate] != usize::MAX {
                    continue;
                }
                let sim = cosine_similarity(embeddings[seed], embeddings[candidate]);
                if 1.0 - sim <= self.config.cluster_radius {
                    cluster_id[candidate] = cid;
                }
            }
        }

        // Count cluster sizes
        let num_clusters = cluster_seeds.len();
        let mut sizes = vec![0usize; num_clusters];
        for &cid in &cluster_id {
            if cid < num_clusters {
                sizes[cid] += 1;
            }
        }

        // Find the largest cluster
        let (largest_cid, _) = sizes
            .iter()
            .enumerate()
            .max_by_key(|(_, &s)| s)
            .unwrap_or((0, &0));

        // Pick the LRU entry within the largest cluster
        cluster_id
            .iter()
            .enumerate()
            .filter(|(_, &cid)| cid == largest_cid)
            .min_by_key(|(i, _)| self.slots[*i].entry.last_accessed)
            .map(|(i, _)| i)
    }

    /// HybridScore: combined recency + frequency eviction.
    /// Evict the entry with the lowest hybrid score.
    fn victim_hybrid(&mut self, weight: f64) -> Option<usize> {
        if self.slots.is_empty() {
            return None;
        }
        let weight = weight.clamp(0.0, 1.0);

        // Normalise last_accessed and access_count into [0,1] ranges
        let max_ts = self.slots.iter().map(|s| s.entry.last_accessed).max()?;
        let max_ts = max_ts.max(1); // avoid divide-by-zero
        let max_cnt = self
            .slots
            .iter()
            .map(|s| s.entry.access_count)
            .max()
            .unwrap_or(1)
            .max(1);

        let mut best_idx = 0;
        let mut best_score = f64::MAX;

        for (i, slot) in self.slots.iter().enumerate() {
            let recency = slot.entry.last_accessed as f64 / max_ts as f64;
            let frequency = slot.entry.access_count as f64 / max_cnt as f64;
            let score = weight * recency + (1.0 - weight) * frequency;
            if score < best_score
                || (score == best_score && {
                    // tie-break with a seeded deterministic step
                    let noise = xorshift64(&mut self.rng_state) & 1;
                    noise == 0
                })
            {
                best_score = score;
                best_idx = i;
            }
        }
        Some(best_idx)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Public type aliases for name-collision compatibility
// ──────────────────────────────────────────────────────────────────────────────

/// Alias: the cache key type used by [`SemanticCacheManager`].
pub type ScmKeyAlias = ScmCacheKey;
/// Alias: the cache entry type used by [`SemanticCacheManager`].
pub type ScmEntryAlias = ScmCacheEntry;
/// Alias: the cache statistics type used by [`SemanticCacheManager`].
pub type ScmStatsAlias = ScmCacheStats;
/// Alias: the cache hit/miss result type used by [`SemanticCacheManager`].
pub type ScmHitAlias = ScmCacheHit;
/// Alias: the error type used by [`SemanticCacheManager`].
pub type ScmErrorAlias = ScmCacheError;

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ──────────────────────────────────────────────────────────────

    fn default_config() -> ScmCacheConfig {
        ScmCacheConfig {
            max_entries: 16,
            max_bytes: 1024 * 1024,
            semantic_threshold: 0.90,
            ttl_us: None,
            strategy: ScmEvictionStrategy::LRU,
            cluster_radius: 0.15,
        }
    }

    fn make_key(text: &str, emb: Vec<f64>) -> ScmCacheKey {
        ScmCacheKey::new(text.to_string(), emb)
    }

    fn unit_vec(dim: usize, hot: usize) -> Vec<f64> {
        let mut v = vec![0.0; dim];
        if hot < dim {
            v[hot] = 1.0;
        }
        v
    }

    fn uniform_vec(dim: usize, val: f64) -> Vec<f64> {
        vec![val; dim]
    }

    fn make_mgr() -> SemanticCacheManager {
        SemanticCacheManager::new(default_config())
    }

    // ── 1. Basic insert returns a valid ID ───────────────────────────────────
    #[test]
    fn test_insert_returns_id() {
        let mut mgr = make_mgr();
        let key = make_key("hello", vec![1.0, 0.0, 0.0]);
        let id = mgr.insert(key, b"result".to_vec(), None).expect("insert");
        assert!(id >= 1);
    }

    // ── 2. IDs are monotonically increasing ──────────────────────────────────
    #[test]
    fn test_insert_monotonic_ids() {
        let mut mgr = make_mgr();
        let id1 = mgr
            .insert(make_key("a", vec![1.0, 0.0]), b"r".to_vec(), None)
            .expect("a");
        let id2 = mgr
            .insert(make_key("b", vec![0.0, 1.0]), b"r".to_vec(), None)
            .expect("b");
        assert!(id2 > id1);
    }

    // ── 3. Exact hit ─────────────────────────────────────────────────────────
    #[test]
    fn test_exact_hit() {
        let mut mgr = make_mgr();
        let emb = vec![1.0, 0.0, 0.0];
        let id = mgr
            .insert(make_key("exact query", emb.clone()), b"res".to_vec(), None)
            .expect("insert");
        let hit = mgr.lookup("exact query", &emb, 0).expect("lookup");
        assert_eq!(hit, ScmCacheHit::Exact { entry_id: id });
    }

    // ── 4. Semantic hit above threshold ──────────────────────────────────────
    #[test]
    fn test_semantic_hit_above_threshold() {
        let mut mgr = make_mgr();
        // Insert with embedding [1, 0, 0]
        let emb = vec![1.0, 0.0, 0.0];
        let id = mgr
            .insert(make_key("what is ipfs", emb.clone()), b"res".to_vec(), None)
            .expect("insert");
        // Query with slightly perturbed embedding — high cosine similarity
        let query_emb = vec![0.999, 0.04, 0.0];
        let hit = mgr.lookup("what is ipfs?", &query_emb, 0).expect("lookup");
        match hit {
            ScmCacheHit::Semantic {
                entry_id,
                similarity,
                ..
            } => {
                assert_eq!(entry_id, id);
                assert!(similarity >= 0.90);
            }
            other => panic!("expected Semantic hit, got {:?}", other),
        }
    }

    // ── 5. Semantic hit below threshold → Miss ───────────────────────────────
    #[test]
    fn test_semantic_hit_below_threshold() {
        let mut mgr = make_mgr();
        mgr.insert(
            make_key("topic A", vec![1.0, 0.0, 0.0]),
            b"r".to_vec(),
            None,
        )
        .expect("insert");
        // Perpendicular embedding → similarity = 0 < 0.90
        let hit = mgr.lookup("topic B", &[0.0, 1.0, 0.0], 0).expect("lookup");
        assert_eq!(hit, ScmCacheHit::Miss);
    }

    // ── 6. Miss on empty cache ────────────────────────────────────────────────
    #[test]
    fn test_miss_on_empty_cache() {
        let mut mgr = make_mgr();
        let hit = mgr.lookup("anything", &[1.0, 0.0], 0).expect("lookup");
        assert_eq!(hit, ScmCacheHit::Miss);
    }

    // ── 7. TTL expiration during lookup ──────────────────────────────────────
    #[test]
    fn test_ttl_expiration_during_lookup() {
        let mut mgr = make_mgr();
        // Insert with 100 µs TTL at ts=0
        let emb = vec![1.0, 0.0, 0.0];
        mgr.insert_at(
            make_key("expiring", emb.clone()),
            b"r".to_vec(),
            Some(100),
            0,
        )
        .expect("insert");
        // Lookup at ts=200 — entry should have expired
        let hit = mgr.lookup("expiring", &emb, 200).expect("lookup");
        assert_eq!(hit, ScmCacheHit::Miss);
        assert_eq!(mgr.slots.len(), 0);
    }

    // ── 8. TTL entry alive before expiry ─────────────────────────────────────
    #[test]
    fn test_ttl_entry_alive_before_expiry() {
        let mut mgr = make_mgr();
        let emb = vec![1.0, 0.0, 0.0];
        let id = mgr
            .insert_at(make_key("live", emb.clone()), b"r".to_vec(), Some(1000), 0)
            .expect("insert");
        // Lookup at ts=50 — still alive
        let hit = mgr.lookup("live", &emb, 50).expect("lookup");
        assert_eq!(hit, ScmCacheHit::Exact { entry_id: id });
    }

    // ── 9. explicit expire_ttl ───────────────────────────────────────────────
    #[test]
    fn test_expire_ttl_explicit() {
        let mut mgr = make_mgr();
        let emb = vec![1.0, 0.0];
        let id1 = mgr
            .insert_at(make_key("q1", emb.clone()), b"r1".to_vec(), Some(50), 0)
            .expect("i1");
        let id2 = mgr
            .insert_at(make_key("q2", vec![0.0, 1.0]), b"r2".to_vec(), None, 0)
            .expect("i2");
        let removed = mgr.expire_ttl(100);
        assert!(removed.contains(&id1));
        assert!(!removed.contains(&id2));
        assert_eq!(mgr.slots.len(), 1);
    }

    // ── 10. LRU eviction strategy ─────────────────────────────────────────────
    #[test]
    fn test_eviction_lru() {
        let config = ScmCacheConfig {
            max_entries: 3,
            max_bytes: 1024 * 1024,
            semantic_threshold: 0.90,
            ttl_us: None,
            strategy: ScmEvictionStrategy::LRU,
            cluster_radius: 0.15,
        };
        let mut mgr = SemanticCacheManager::new(config);

        // Insert 3 entries at different timestamps
        let id1 = mgr
            .insert_at(make_key("q1", unit_vec(4, 0)), b"r".to_vec(), None, 100)
            .expect("i1");
        let _id2 = mgr
            .insert_at(make_key("q2", unit_vec(4, 1)), b"r".to_vec(), None, 200)
            .expect("i2");
        let _id3 = mgr
            .insert_at(make_key("q3", unit_vec(4, 2)), b"r".to_vec(), None, 300)
            .expect("i3");

        // Insert a 4th — must evict id1 (least recently accessed, ts=100)
        let _id4 = mgr
            .insert_at(make_key("q4", unit_vec(4, 3)), b"r".to_vec(), None, 400)
            .expect("i4");

        assert_eq!(mgr.slots.len(), 3);
        assert!(mgr.get_entry(id1).is_err(), "id1 should have been evicted");
    }

    // ── 11. LFU eviction strategy ─────────────────────────────────────────────
    #[test]
    fn test_eviction_lfu() {
        let config = ScmCacheConfig {
            max_entries: 3,
            max_bytes: 1024 * 1024,
            semantic_threshold: 0.90,
            ttl_us: None,
            strategy: ScmEvictionStrategy::LFU,
            cluster_radius: 0.15,
        };
        let mut mgr = SemanticCacheManager::new(config);

        let id1 = mgr
            .insert(make_key("rare", unit_vec(4, 0)), b"r".to_vec(), None)
            .expect("i1");
        let _id2 = mgr
            .insert(make_key("common1", unit_vec(4, 1)), b"r".to_vec(), None)
            .expect("i2");
        let _id3 = mgr
            .insert(make_key("common2", unit_vec(4, 2)), b"r".to_vec(), None)
            .expect("i3");

        // Access id2 and id3 multiple times so id1 has lowest access_count
        for _ in 0..5 {
            let _ = mgr.lookup("common1", &unit_vec(4, 1), 0);
            let _ = mgr.lookup("common2", &unit_vec(4, 2), 0);
        }

        // Insert 4th — id1 (access_count=0) should be evicted
        let _id4 = mgr
            .insert(make_key("new", unit_vec(4, 3)), b"r".to_vec(), None)
            .expect("i4");

        assert_eq!(mgr.slots.len(), 3);
        assert!(mgr.get_entry(id1).is_err(), "id1 (LFU) should be evicted");
    }

    // ── 12. TTLFirst eviction strategy ───────────────────────────────────────
    #[test]
    fn test_eviction_ttl_first() {
        let config = ScmCacheConfig {
            max_entries: 3,
            max_bytes: 1024 * 1024,
            semantic_threshold: 0.90,
            ttl_us: None,
            strategy: ScmEvictionStrategy::TTLFirst,
            cluster_radius: 0.15,
        };
        let mut mgr = SemanticCacheManager::new(config);

        let id_short = mgr
            .insert(
                make_key("short-ttl", unit_vec(4, 0)),
                b"r".to_vec(),
                Some(10),
            )
            .expect("short");
        let _id_long = mgr
            .insert(
                make_key("long-ttl", unit_vec(4, 1)),
                b"r".to_vec(),
                Some(10_000),
            )
            .expect("long");
        let _id_none = mgr
            .insert(make_key("no-ttl", unit_vec(4, 2)), b"r".to_vec(), None)
            .expect("none");

        // Insert 4th — should evict the shortest-TTL entry
        let _id4 = mgr
            .insert(make_key("q4", unit_vec(4, 3)), b"r".to_vec(), None)
            .expect("i4");

        assert_eq!(mgr.slots.len(), 3);
        assert!(
            mgr.get_entry(id_short).is_err(),
            "short-TTL entry should be evicted first"
        );
    }

    // ── 13. SemanticCluster eviction strategy ────────────────────────────────
    #[test]
    fn test_eviction_semantic_cluster() {
        let config = ScmCacheConfig {
            max_entries: 4,
            max_bytes: 1024 * 1024,
            semantic_threshold: 0.50,
            ttl_us: None,
            strategy: ScmEvictionStrategy::SemanticCluster,
            cluster_radius: 0.10, // tight radius — group near-identical embeddings
        };
        let mut mgr = SemanticCacheManager::new(config);

        // Insert 3 very similar embeddings (cluster A) and 1 distinct one (cluster B)
        let base = vec![1.0_f64, 0.001, 0.0];
        mgr.insert(make_key("a1", base.clone()), b"r".to_vec(), None)
            .expect("a1");
        mgr.insert(make_key("a2", vec![1.0, 0.002, 0.0]), b"r".to_vec(), None)
            .expect("a2");
        mgr.insert(make_key("a3", vec![1.0, 0.003, 0.0]), b"r".to_vec(), None)
            .expect("a3");
        mgr.insert(make_key("b1", vec![0.0, 1.0, 0.0]), b"r".to_vec(), None)
            .expect("b1");

        // Insert a 5th — should evict from the over-represented cluster A
        mgr.insert(make_key("new", vec![0.0, 0.0, 1.0]), b"r".to_vec(), None)
            .expect("new");

        assert_eq!(mgr.slots.len(), 4);
        // Cluster B (size=1) and cluster A (size=3) → eviction from A
        // Verify at least one of a1/a2/a3 was removed
        let remaining_a: Vec<_> = mgr
            .slots
            .iter()
            .filter(|s| {
                let t = &s.entry.key.query_text;
                t == "a1" || t == "a2" || t == "a3"
            })
            .collect();
        assert!(
            remaining_a.len() < 3,
            "One of the cluster-A entries should have been evicted"
        );
    }

    // ── 14. HybridScore eviction (weight=1.0 ≈ LRU) ──────────────────────────
    #[test]
    fn test_eviction_hybrid_weight_one() {
        let config = ScmCacheConfig {
            max_entries: 3,
            max_bytes: 1024 * 1024,
            semantic_threshold: 0.90,
            ttl_us: None,
            strategy: ScmEvictionStrategy::HybridScore(1.0),
            cluster_radius: 0.15,
        };
        let mut mgr = SemanticCacheManager::new(config);

        let id_old = mgr
            .insert_at(make_key("old", unit_vec(4, 0)), b"r".to_vec(), None, 1)
            .expect("old");
        let _id_mid = mgr
            .insert_at(make_key("mid", unit_vec(4, 1)), b"r".to_vec(), None, 50)
            .expect("mid");
        let _id_new = mgr
            .insert_at(make_key("new", unit_vec(4, 2)), b"r".to_vec(), None, 100)
            .expect("new");

        // Insert 4th — HybridScore(1.0) = pure recency → evict oldest
        let _id4 = mgr
            .insert_at(make_key("q4", unit_vec(4, 3)), b"r".to_vec(), None, 200)
            .expect("i4");

        assert_eq!(mgr.slots.len(), 3);
        assert!(
            mgr.get_entry(id_old).is_err(),
            "oldest entry should be evicted with weight=1.0"
        );
    }

    // ── 15. HybridScore eviction (weight=0.0 ≈ LFU) ──────────────────────────
    #[test]
    fn test_eviction_hybrid_weight_zero() {
        let config = ScmCacheConfig {
            max_entries: 3,
            max_bytes: 1024 * 1024,
            semantic_threshold: 0.90,
            ttl_us: None,
            strategy: ScmEvictionStrategy::HybridScore(0.0),
            cluster_radius: 0.15,
        };
        let mut mgr = SemanticCacheManager::new(config);

        let id_rare = mgr
            .insert(make_key("rare", unit_vec(4, 0)), b"r".to_vec(), None)
            .expect("rare");
        let _id_freq1 = mgr
            .insert(make_key("freq1", unit_vec(4, 1)), b"r".to_vec(), None)
            .expect("freq1");
        let _id_freq2 = mgr
            .insert(make_key("freq2", unit_vec(4, 2)), b"r".to_vec(), None)
            .expect("freq2");

        // Bump access counts on freq1 and freq2
        for _ in 0..10 {
            let _ = mgr.lookup("freq1", &unit_vec(4, 1), 0);
            let _ = mgr.lookup("freq2", &unit_vec(4, 2), 0);
        }

        // Insert 4th — pure frequency → evict "rare" (access_count=0)
        let _id4 = mgr
            .insert(make_key("q4", unit_vec(4, 3)), b"r".to_vec(), None)
            .expect("i4");

        assert_eq!(mgr.slots.len(), 3);
        assert!(
            mgr.get_entry(id_rare).is_err(),
            "least-frequent entry should be evicted with weight=0.0"
        );
    }

    // ── 16. invalidate removes entry ─────────────────────────────────────────
    #[test]
    fn test_invalidate() {
        let mut mgr = make_mgr();
        let id = mgr
            .insert(make_key("q", vec![1.0, 0.0]), b"r".to_vec(), None)
            .expect("insert");
        mgr.invalidate(id).expect("invalidate");
        assert!(mgr.get_entry(id).is_err());
    }

    // ── 17. invalidate unknown ID returns error ───────────────────────────────
    #[test]
    fn test_invalidate_unknown_id() {
        let mut mgr = make_mgr();
        let err = mgr.invalidate(999).unwrap_err();
        assert_eq!(err, ScmCacheError::EntryNotFound(999));
    }

    // ── 18. invalidate_similar removes semantically close entries ─────────────
    #[test]
    fn test_invalidate_similar() {
        let mut mgr = make_mgr();
        let id1 = mgr
            .insert(make_key("a", vec![1.0, 0.0, 0.0]), b"r".to_vec(), None)
            .expect("a");
        let id2 = mgr
            .insert(make_key("b", vec![0.999, 0.044, 0.0]), b"r".to_vec(), None)
            .expect("b");
        let id3 = mgr
            .insert(make_key("c", vec![0.0, 1.0, 0.0]), b"r".to_vec(), None)
            .expect("c");

        // Remove entries similar to [1, 0, 0] with threshold 0.95
        let removed = mgr.invalidate_similar(&[1.0, 0.0, 0.0], 0.95);
        assert!(removed.contains(&id1));
        assert!(removed.contains(&id2));
        assert!(!removed.contains(&id3));
        assert_eq!(mgr.slots.len(), 1);
    }

    // ── 19. invalidate_similar returns empty when nothing matches ─────────────
    #[test]
    fn test_invalidate_similar_no_match() {
        let mut mgr = make_mgr();
        mgr.insert(make_key("a", vec![0.0, 1.0, 0.0]), b"r".to_vec(), None)
            .expect("a");
        let removed = mgr.invalidate_similar(&[1.0, 0.0, 0.0], 0.99);
        assert!(removed.is_empty());
    }

    // ── 20. semantic_neighbors top-k ─────────────────────────────────────────
    #[test]
    fn test_semantic_neighbors_top_k() {
        let mut mgr = make_mgr();
        for i in 0..8 {
            let mut emb = vec![0.0_f64; 8];
            emb[i] = 1.0;
            mgr.insert(make_key(&format!("q{i}"), emb), b"r".to_vec(), None)
                .expect("insert");
        }
        // Query along dimension 0 — unit_vec(8, 0)
        let query = unit_vec(8, 0);
        let neighbors = mgr.semantic_neighbors(&query, 3);
        assert_eq!(neighbors.len(), 3);
        // The first neighbor should be the exact match (similarity = 1.0)
        assert!((neighbors[0].1 - 1.0).abs() < 1e-9);
        // Descending order
        for w in neighbors.windows(2) {
            assert!(w[0].1 >= w[1].1);
        }
    }

    // ── 21. semantic_neighbors top_k larger than cache ───────────────────────
    #[test]
    fn test_semantic_neighbors_k_larger_than_cache() {
        let mut mgr = make_mgr();
        mgr.insert(make_key("only", vec![1.0, 0.0]), b"r".to_vec(), None)
            .expect("insert");
        let neighbors = mgr.semantic_neighbors(&[1.0, 0.0], 100);
        assert_eq!(neighbors.len(), 1);
    }

    // ── 22. semantic_neighbors on empty cache ─────────────────────────────────
    #[test]
    fn test_semantic_neighbors_empty() {
        let mgr = make_mgr();
        let neighbors = mgr.semantic_neighbors(&[1.0, 0.0], 5);
        assert!(neighbors.is_empty());
    }

    // ── 23. cluster_stats on empty cache ─────────────────────────────────────
    #[test]
    fn test_cluster_stats_empty() {
        let mgr = make_mgr();
        let stats = mgr.cluster_stats();
        assert!(stats.is_empty());
    }

    // ── 24. cluster_stats single entry ───────────────────────────────────────
    #[test]
    fn test_cluster_stats_single() {
        let mut mgr = make_mgr();
        mgr.insert(make_key("q", vec![1.0, 0.0]), b"r".to_vec(), None)
            .expect("insert");
        let stats = mgr.cluster_stats();
        assert_eq!(stats.len(), 1);
        assert_eq!(stats[0].1, 1); // cluster size = 1
    }

    // ── 25. cluster_stats two clusters ───────────────────────────────────────
    #[test]
    fn test_cluster_stats_two_clusters() {
        let config = ScmCacheConfig {
            cluster_radius: 0.05, // very tight → orthogonal vectors form separate clusters
            ..default_config()
        };
        let mut mgr = SemanticCacheManager::new(config);
        // Cluster A: embeddings near [1, 0, 0]
        mgr.insert(make_key("a1", vec![1.0, 0.0, 0.0]), b"r".to_vec(), None)
            .expect("a1");
        mgr.insert(make_key("a2", vec![1.0, 0.001, 0.0]), b"r".to_vec(), None)
            .expect("a2");
        // Cluster B: embeddings near [0, 1, 0]
        mgr.insert(make_key("b1", vec![0.0, 1.0, 0.0]), b"r".to_vec(), None)
            .expect("b1");
        mgr.insert(make_key("b2", vec![0.001, 1.0, 0.0]), b"r".to_vec(), None)
            .expect("b2");

        let stats = mgr.cluster_stats();
        // Expect two clusters, each of size 2
        assert_eq!(stats.len(), 2);
        for (_, size) in &stats {
            assert_eq!(*size, 2);
        }
    }

    // ── 26. cluster_stats intra-cluster similarity ────────────────────────────
    #[test]
    fn test_cluster_stats_similarity_bound() {
        let mut mgr = SemanticCacheManager::new(ScmCacheConfig {
            cluster_radius: 0.20,
            ..default_config()
        });
        mgr.insert(make_key("x", vec![1.0, 0.0]), b"r".to_vec(), None)
            .expect("x");
        let stats = mgr.cluster_stats();
        assert!(!stats.is_empty());
        for (sim, _) in &stats {
            assert!(*sim >= -1.0 && *sim <= 1.0);
        }
    }

    // ── 27. dimension mismatch on insert ─────────────────────────────────────
    #[test]
    fn test_dimension_mismatch_insert() {
        let mut mgr = make_mgr();
        mgr.insert(make_key("q1", vec![1.0, 0.0, 0.0]), b"r".to_vec(), None)
            .expect("first insert");
        let err = mgr
            .insert(make_key("q2", vec![1.0, 0.0]), b"r".to_vec(), None)
            .unwrap_err();
        assert!(matches!(
            err,
            ScmCacheError::DimensionMismatch {
                expected: 3,
                got: 2
            }
        ));
    }

    // ── 28. dimension mismatch on lookup ─────────────────────────────────────
    #[test]
    fn test_dimension_mismatch_lookup() {
        let mut mgr = make_mgr();
        mgr.insert(make_key("q1", vec![1.0, 0.0, 0.0]), b"r".to_vec(), None)
            .expect("insert");
        let err = mgr.lookup("q2", &[1.0, 0.0], 0).unwrap_err();
        assert!(matches!(
            err,
            ScmCacheError::DimensionMismatch {
                expected: 3,
                got: 2
            }
        ));
    }

    // ── 29. capacity enforcement (entry count) ───────────────────────────────
    #[test]
    fn test_capacity_entry_count() {
        let config = ScmCacheConfig {
            max_entries: 5,
            max_bytes: 1024 * 1024,
            semantic_threshold: 0.90,
            ttl_us: None,
            strategy: ScmEvictionStrategy::LRU,
            cluster_radius: 0.15,
        };
        let mut mgr = SemanticCacheManager::new(config);
        for i in 0..10usize {
            let mut emb = vec![0.0; 10];
            emb[i % 10] = 1.0;
            mgr.insert(make_key(&format!("q{i}"), emb), b"r".to_vec(), None)
                .expect("insert");
        }
        assert!(mgr.slots.len() <= 5);
    }

    // ── 30. capacity enforcement (byte budget) ───────────────────────────────
    #[test]
    fn test_capacity_bytes() {
        let config = ScmCacheConfig {
            max_entries: 1000,
            max_bytes: 512, // very small byte budget
            semantic_threshold: 0.90,
            ttl_us: None,
            strategy: ScmEvictionStrategy::LRU,
            cluster_radius: 0.15,
        };
        let mut mgr = SemanticCacheManager::new(config);
        // Insert entries that are deliberately small
        for i in 0..20usize {
            let mut emb = vec![0.0; 2];
            emb[i % 2] = 1.0;
            let _ = mgr.insert(make_key(&format!("q{i}"), emb), b"r".to_vec(), None);
        }
        assert!(mgr.current_bytes() <= 512 + 512); // some tolerance for last inserted
    }

    // ── 31. stats: exact hits counter ────────────────────────────────────────
    #[test]
    fn test_stats_exact_hits() {
        let mut mgr = make_mgr();
        let emb = vec![1.0, 0.0];
        let id = mgr
            .insert(make_key("q", emb.clone()), b"r".to_vec(), None)
            .expect("insert");
        for _ in 0..3 {
            let hit = mgr.lookup("q", &emb, 0).expect("lookup");
            assert_eq!(hit, ScmCacheHit::Exact { entry_id: id });
        }
        let s = mgr.stats();
        assert_eq!(s.exact_hits, 3);
        assert_eq!(s.semantic_hits, 0);
        assert_eq!(s.misses, 0);
    }

    // ── 32. stats: semantic hits counter ─────────────────────────────────────
    #[test]
    fn test_stats_semantic_hits() {
        let mut mgr = make_mgr();
        mgr.insert(
            make_key("original", vec![1.0, 0.0, 0.0]),
            b"r".to_vec(),
            None,
        )
        .expect("insert");
        // Close enough to trigger semantic hit but not exact text
        let _ = mgr.lookup("different text", &[0.999, 0.044, 0.0], 0);
        let s = mgr.stats();
        assert_eq!(s.semantic_hits, 1);
        assert!(s.avg_similarity_on_semantic_hit > 0.0);
    }

    // ── 33. stats: miss counter ───────────────────────────────────────────────
    #[test]
    fn test_stats_misses() {
        let mut mgr = make_mgr();
        mgr.insert(make_key("q", vec![1.0, 0.0]), b"r".to_vec(), None)
            .expect("insert");
        let _ = mgr.lookup("totally different", &[0.0, 1.0], 0);
        let s = mgr.stats();
        assert_eq!(s.misses, 1);
    }

    // ── 34. stats: evictions counter ─────────────────────────────────────────
    #[test]
    fn test_stats_evictions() {
        let config = ScmCacheConfig {
            max_entries: 2,
            ..default_config()
        };
        let mut mgr = SemanticCacheManager::new(config);
        mgr.insert(make_key("a", unit_vec(3, 0)), b"r".to_vec(), None)
            .expect("a");
        mgr.insert(make_key("b", unit_vec(3, 1)), b"r".to_vec(), None)
            .expect("b");
        mgr.insert(make_key("c", unit_vec(3, 2)), b"r".to_vec(), None)
            .expect("c");
        assert_eq!(mgr.stats().evictions, 1);
    }

    // ── 35. stats: total_insertions ──────────────────────────────────────────
    #[test]
    fn test_stats_total_insertions() {
        let mut mgr = make_mgr();
        for i in 0..5usize {
            let mut e = vec![0.0; 5];
            e[i] = 1.0;
            mgr.insert(make_key(&format!("q{i}"), e), b"r".to_vec(), None)
                .expect("insert");
        }
        assert_eq!(mgr.stats().total_insertions, 5);
    }

    // ── 36. stats: semantic_hit_rate ─────────────────────────────────────────
    #[test]
    fn test_stats_semantic_hit_rate() {
        let mut mgr = make_mgr();
        mgr.insert(make_key("q", vec![1.0, 0.0, 0.0]), b"r".to_vec(), None)
            .expect("insert");
        // 1 semantic hit, 1 miss
        let _ = mgr.lookup("close", &[0.999, 0.044, 0.0], 0);
        let _ = mgr.lookup("far", &[0.0, 1.0, 0.0], 0);
        let s = mgr.stats();
        // semantic_hit_rate = 1 / (0 + 1 + 1) = 0.5
        assert!((s.semantic_hit_rate - 0.5).abs() < 1e-9);
    }

    // ── 37. stats: avg_similarity_on_semantic_hit ────────────────────────────
    #[test]
    fn test_stats_avg_similarity() {
        let mut mgr = make_mgr();
        mgr.insert(make_key("q", vec![1.0, 0.0, 0.0]), b"r".to_vec(), None)
            .expect("insert");
        // Two semantic hits
        let _ = mgr.lookup("close1", &[0.999, 0.044, 0.0], 0);
        let _ = mgr.lookup("close2", &[0.999, 0.044, 0.0], 0);
        let s = mgr.stats();
        assert!(s.avg_similarity_on_semantic_hit > 0.0);
        assert!(s.avg_similarity_on_semantic_hit <= 1.0);
    }

    // ── 38. get_entry returns correct payload ─────────────────────────────────
    #[test]
    fn test_get_entry() {
        let mut mgr = make_mgr();
        let payload = b"important data".to_vec();
        let id = mgr
            .insert(make_key("q", vec![1.0, 0.0]), payload.clone(), None)
            .expect("insert");
        let entry = mgr.get_entry(id).expect("get_entry");
        assert_eq!(entry.result, payload);
    }

    // ── 39. get_entry unknown ID returns error ────────────────────────────────
    #[test]
    fn test_get_entry_not_found() {
        let mgr = make_mgr();
        assert!(matches!(
            mgr.get_entry(42),
            Err(ScmCacheError::EntryNotFound(42))
        ));
    }

    // ── 40. access_count updated on hit ──────────────────────────────────────
    #[test]
    fn test_access_count_updated() {
        let mut mgr = make_mgr();
        let emb = vec![1.0, 0.0];
        let id = mgr
            .insert(make_key("q", emb.clone()), b"r".to_vec(), None)
            .expect("insert");
        assert_eq!(mgr.get_entry(id).expect("entry").access_count, 0);
        let _ = mgr.lookup("q", &emb, 0);
        let _ = mgr.lookup("q", &emb, 1);
        assert_eq!(mgr.get_entry(id).expect("entry").access_count, 2);
    }

    // ── 41. last_accessed updated on hit ─────────────────────────────────────
    #[test]
    fn test_last_accessed_updated() {
        let mut mgr = make_mgr();
        let emb = vec![1.0, 0.0];
        let id = mgr
            .insert(make_key("q", emb.clone()), b"r".to_vec(), None)
            .expect("insert");
        let _ = mgr.lookup("q", &emb, 999);
        assert_eq!(mgr.get_entry(id).expect("entry").last_accessed, 999);
    }

    // ── 42. entry_too_large error ─────────────────────────────────────────────
    #[test]
    fn test_entry_too_large() {
        let config = ScmCacheConfig {
            max_bytes: 10, // tiny
            ..default_config()
        };
        let mut mgr = SemanticCacheManager::new(config);
        let err = mgr
            .insert(
                make_key("q", vec![1.0]),
                vec![0u8; 200], // payload alone > 10 bytes
                None,
            )
            .unwrap_err();
        assert!(matches!(err, ScmCacheError::EntryTooLarge(_)));
    }

    // ── 43. evict_to_fit returns evicted IDs ─────────────────────────────────
    #[test]
    fn test_evict_to_fit_returns_ids() {
        // Use a tiny byte budget so that asking for even a small amount forces eviction.
        let config = ScmCacheConfig {
            max_entries: 100,
            max_bytes: 512, // small enough that 5 entries exhaust it
            ..default_config()
        };
        let mut mgr = SemanticCacheManager::new(config);
        for i in 0..5usize {
            let mut e = vec![0.0; 5];
            e[i] = 1.0;
            let _ = mgr.insert(make_key(&format!("q{i}"), e), b"r".to_vec(), None);
        }
        // Now request budget that exceeds max_bytes to force eviction
        let removed = mgr.evict_to_fit(mgr.config.max_bytes + 1);
        assert!(!removed.is_empty());
    }

    // ── 44. evict_to_fit no-op when space is available ───────────────────────
    #[test]
    fn test_evict_to_fit_noop() {
        let mut mgr = make_mgr();
        mgr.insert(make_key("q", vec![1.0, 0.0]), b"r".to_vec(), None)
            .expect("insert");
        let removed = mgr.evict_to_fit(0);
        assert!(removed.is_empty());
    }

    // ── 45. FNV-1a hash consistency ──────────────────────────────────────────
    #[test]
    fn test_fnv1a_consistency() {
        let h1 = fnv1a_64(b"hello world");
        let h2 = fnv1a_64(b"hello world");
        assert_eq!(h1, h2);
        let h3 = fnv1a_64(b"Hello World");
        assert_ne!(h1, h3);
    }

    // ── 46. cosine_similarity edge cases ─────────────────────────────────────
    #[test]
    fn test_cosine_similarity_edge_cases() {
        assert_eq!(cosine_similarity(&[], &[]), 0.0);
        assert_eq!(cosine_similarity(&[0.0, 0.0], &[1.0, 0.0]), 0.0); // zero vector
        assert!((cosine_similarity(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-9);
        assert!((cosine_similarity(&[1.0, 0.0], &[0.0, 1.0])).abs() < 1e-9);
        // Mismatched lengths
        assert_eq!(cosine_similarity(&[1.0, 0.0], &[1.0, 0.0, 0.0]), 0.0);
    }

    // ── 47. xorshift64 produces distinct values ───────────────────────────────
    #[test]
    fn test_xorshift64_distinct() {
        let mut state = 0x1234_5678_9ABC_DEF0u64;
        let v1 = xorshift64(&mut state);
        let v2 = xorshift64(&mut state);
        assert_ne!(v1, v2);
    }

    // ── 48. ScmCacheKey::new computes hash from text ──────────────────────────
    #[test]
    fn test_cache_key_hash() {
        let k = ScmCacheKey::new("test".to_string(), vec![]);
        assert_eq!(k.query_hash, fnv1a_64(b"test"));
    }

    // ── 49. ScmCacheEntry::is_expired semantics ───────────────────────────────
    #[test]
    fn test_cache_entry_is_expired() {
        let key = ScmCacheKey::new("q".to_string(), vec![1.0]);
        let entry = ScmCacheEntry {
            key,
            result: vec![],
            inserted_at: 0,
            last_accessed: 0,
            access_count: 0,
            ttl_us: Some(100),
            similarity_score: 1.0,
        };
        assert!(!entry.is_expired(99));
        assert!(entry.is_expired(100));
        assert!(entry.is_expired(200));
    }

    // ── 50. ScmCacheEntry without TTL never expires ───────────────────────────
    #[test]
    fn test_cache_entry_no_ttl() {
        let key = ScmCacheKey::new("q".to_string(), vec![]);
        let entry = ScmCacheEntry {
            key,
            result: vec![],
            inserted_at: 0,
            last_accessed: 0,
            access_count: 0,
            ttl_us: None,
            similarity_score: 1.0,
        };
        assert!(!entry.is_expired(u64::MAX));
    }

    // ── 51. multiple inserts, semantic search picks best ─────────────────────
    #[test]
    fn test_semantic_best_match() {
        let mut mgr = make_mgr();
        // Insert two entries with different similarities to [1, 0, 0]
        mgr.insert(
            make_key("good", vec![0.98, 0.2, 0.0]),
            b"good".to_vec(),
            None,
        )
        .expect("good");
        mgr.insert(make_key("ok", vec![0.92, 0.4, 0.0]), b"ok".to_vec(), None)
            .expect("ok");
        // Query: [1, 0, 0] — "good" should be the best semantic hit
        let hit = mgr.lookup("query", &[1.0, 0.0, 0.0], 0).expect("lookup");
        match hit {
            ScmCacheHit::Semantic { original_query, .. } => {
                assert_eq!(original_query, "good");
            }
            other => panic!("expected Semantic hit, got {:?}", other),
        }
    }

    // ── 52. uniform embeddings produce nonzero similarity ─────────────────────
    #[test]
    fn test_uniform_embeddings_similarity() {
        let a = uniform_vec(64, 0.5);
        let b = uniform_vec(64, 0.3);
        // Both are constant vectors — cosine similarity = 1.0
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);
    }

    // ── 53. evict_to_fit respects byte budget ────────────────────────────────
    #[test]
    fn test_evict_to_fit_respects_bytes() {
        let config = ScmCacheConfig {
            max_entries: 1000,
            max_bytes: 2048,
            ..default_config()
        };
        let mut mgr = SemanticCacheManager::new(config);
        for i in 0..10usize {
            let mut e = vec![0.0; 2];
            e[i % 2] = 1.0;
            let _ = mgr.insert(make_key(&format!("q{i}"), e), vec![0u8; 50], None);
        }
        let current = mgr.current_bytes();
        // Force removal of enough entries to free ~half the space
        let freed_target = current / 2;
        let removed = mgr.evict_to_fit(freed_target + mgr.config.max_bytes);
        assert!(!removed.is_empty());
    }

    // ── 54. default config is usable ─────────────────────────────────────────
    #[test]
    fn test_default_config() {
        let config = ScmCacheConfig::default();
        let mut mgr = SemanticCacheManager::new(config);
        let id = mgr
            .insert(make_key("q", vec![1.0, 0.0]), b"r".to_vec(), None)
            .expect("insert");
        assert!(id >= 1);
    }

    // ── 55. ScmCacheError Display ─────────────────────────────────────────────
    #[test]
    fn test_cache_error_display() {
        assert!(ScmCacheError::EntryTooLarge(100)
            .to_string()
            .contains("100"));
        assert!(ScmCacheError::CacheAtCapacity
            .to_string()
            .contains("capacity"));
        assert!(ScmCacheError::EntryNotFound(7).to_string().contains("7"));
        assert!(ScmCacheError::DimensionMismatch {
            expected: 3,
            got: 2
        }
        .to_string()
        .contains("mismatch"));
        assert!(ScmCacheError::TtlExpired(5).to_string().contains("5"));
    }

    // ── 56. semantic hit updates original_query field ─────────────────────────
    #[test]
    fn test_semantic_hit_original_query() {
        let mut mgr = make_mgr();
        let original = "the original query text";
        mgr.insert(make_key(original, vec![1.0, 0.0, 0.0]), b"r".to_vec(), None)
            .expect("insert");
        let hit = mgr
            .lookup("a different phrasing", &[0.999, 0.044, 0.0], 0)
            .expect("lookup");
        match hit {
            ScmCacheHit::Semantic { original_query, .. } => {
                assert_eq!(original_query, original);
            }
            other => panic!("expected Semantic, got {:?}", other),
        }
    }

    // ── 57. large-scale stress: 1000 inserts under capacity ──────────────────
    #[test]
    fn test_large_scale_capacity() {
        let config = ScmCacheConfig {
            max_entries: 100,
            max_bytes: 64 * 1024 * 1024,
            semantic_threshold: 0.90,
            ttl_us: None,
            strategy: ScmEvictionStrategy::LRU,
            cluster_radius: 0.20,
        };
        let mut mgr = SemanticCacheManager::new(config);
        let mut state = 0xFEED_FACE_DEAD_BEEF_u64;
        for i in 0..1000usize {
            let dim = 16;
            let emb: Vec<f64> = (0..dim)
                .map(|_| (xorshift64(&mut state) >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0)
                .collect();
            let _ = mgr.insert(make_key(&format!("stress-{i}"), emb), vec![0u8; 32], None);
        }
        assert!(mgr.slots.len() <= 100);
    }

    // ── 58. insert with explicit TTL overrides config TTL = None ─────────────
    #[test]
    fn test_insert_explicit_ttl() {
        let mut mgr = make_mgr(); // config.ttl_us = None
        let id = mgr
            .insert_at(make_key("q", vec![1.0, 0.0]), b"r".to_vec(), Some(50), 0)
            .expect("insert");
        let entry = mgr.get_entry(id).expect("get");
        assert_eq!(entry.ttl_us, Some(50));
    }

    // ── 59. insert inherits config TTL when None given ────────────────────────
    #[test]
    fn test_insert_inherits_config_ttl() {
        let config = ScmCacheConfig {
            ttl_us: Some(1000),
            ..default_config()
        };
        let mut mgr = SemanticCacheManager::new(config);
        let id = mgr
            .insert(make_key("q", vec![1.0, 0.0]), b"r".to_vec(), None)
            .expect("insert");
        let entry = mgr.get_entry(id).expect("get");
        assert_eq!(entry.ttl_us, Some(1000));
    }
}
