//! Proof Caching Layer
//!
//! Caches proof results keyed by `(goal_hash, kb_version)` to avoid redundant
//! inference. Supports LFU-style eviction (lowest access_count first) and
//! TTL-based expiry.
//!
//! # Design
//!
//! - Linear scan over a `Vec<CachedProof>` (suitable for small caches ≤ 256).
//! - LFU eviction: when at capacity, the entry with the lowest `access_count`
//!   is removed to make room for a new entry.
//! - TTL: entries are considered stale if `now_secs - cached_at_secs >= ttl_secs`.
//! - Optional invalidation on KB version change via `invalidate_on_kb_change`.

// ──────────────────────────────────────────────────────────────────────────────
// FNV-1a hash helper
// ──────────────────────────────────────────────────────────────────────────────

const FNV_OFFSET_BASIS: u64 = 14_695_981_039_346_656_037;
const FNV_PRIME: u64 = 1_099_511_628_211;

/// Compute the FNV-1a hash of a string.
///
/// This is exposed as `pub` so that callers can build [`ProofCacheKey`] values
/// without importing a separate hashing crate.
///
/// # Examples
///
/// ```
/// use ipfrs_tensorlogic::proof_cache::fnv1a_hash;
/// let h = fnv1a_hash("parent(alice, bob)");
/// assert_ne!(h, 0);
/// // Deterministic across calls
/// assert_eq!(h, fnv1a_hash("parent(alice, bob)"));
/// ```
pub fn fnv1a_hash(s: &str) -> u64 {
    let mut hash = FNV_OFFSET_BASIS;
    for byte in s.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

// ──────────────────────────────────────────────────────────────────────────────
// ProofCacheKey
// ──────────────────────────────────────────────────────────────────────────────

/// Composite key for a cached proof: goal identity × KB version.
///
/// Using the FNV-1a hash of the goal string avoids storing arbitrarily long
/// goal representations in the cache.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct ProofCacheKey {
    /// FNV-1a hash of the serialised goal string.
    pub goal_hash: u64,
    /// Monotonic version counter of the knowledge base at proof time.
    pub kb_version: u64,
}

impl ProofCacheKey {
    /// Construct a key from already-computed parts.
    pub fn new(goal_hash: u64, kb_version: u64) -> Self {
        Self {
            goal_hash,
            kb_version,
        }
    }

    /// Convenience constructor: hash the goal string on your behalf.
    pub fn from_goal(goal: &str, kb_version: u64) -> Self {
        Self {
            goal_hash: fnv1a_hash(goal),
            kb_version,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// CachedProof
// ──────────────────────────────────────────────────────────────────────────────

/// A single cached proof result together with its bookkeeping metadata.
#[derive(Debug, Clone)]
pub struct CachedProof {
    /// The key under which this proof is stored.
    pub key: ProofCacheKey,
    /// Whether the proof succeeded.
    pub proved: bool,
    /// Variable→value binding pairs produced by the proof.
    pub bindings: Vec<(String, String)>,
    /// Maximum inference depth reached during the proof.
    pub proof_depth: usize,
    /// Unix timestamp (seconds) at which the proof was cached.
    pub cached_at_secs: u64,
    /// Unix timestamp (seconds) of the most recent access.
    pub last_accessed_secs: u64,
    /// Total number of times this entry has been accessed via [`ProofCachingLayer::lookup`].
    pub access_count: u64,
}

impl CachedProof {
    /// Returns `true` if the entry has lived past its TTL.
    ///
    /// An entry is stale when `now_secs - cached_at_secs >= ttl_secs`.
    pub fn is_stale(&self, ttl_secs: u64, now_secs: u64) -> bool {
        now_secs.saturating_sub(self.cached_at_secs) >= ttl_secs
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// ProofCacheConfig
// ──────────────────────────────────────────────────────────────────────────────

/// Configuration for [`ProofCachingLayer`].
#[derive(Debug, Clone)]
pub struct ProofCacheConfig {
    /// Maximum number of entries held in the cache at any time.
    ///
    /// Defaults to 256.
    pub max_entries: usize,
    /// Time-to-live in seconds for cached entries.
    ///
    /// Defaults to 300 (5 minutes).
    pub ttl_secs: u64,
    /// When `true`, all entries whose `key.kb_version` matches the supplied
    /// `old_version` are removed on [`ProofCachingLayer::invalidate_kb_version`].
    ///
    /// Defaults to `true`.
    pub invalidate_on_kb_change: bool,
}

impl Default for ProofCacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 256,
            ttl_secs: 300,
            invalidate_on_kb_change: true,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// ProofCacheStats
// ──────────────────────────────────────────────────────────────────────────────

/// Cumulative statistics for a [`ProofCachingLayer`].
#[derive(Debug, Clone, Default)]
pub struct ProofCacheStats {
    /// Total successful lookups (stale entries do **not** count as hits).
    pub hits: u64,
    /// Total failed lookups (no entry or stale entry).
    pub misses: u64,
    /// Total entries removed by LFU eviction to make room for new entries.
    pub evictions: u64,
    /// Total entries removed by explicit KB-version invalidation.
    pub invalidations: u64,
}

impl ProofCacheStats {
    /// Fraction of lookups that resulted in a cache hit, in `[0.0, 1.0]`.
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
}

// ──────────────────────────────────────────────────────────────────────────────
// ProofCachingLayer
// ──────────────────────────────────────────────────────────────────────────────

/// An LFU-evicting, TTL-expiring cache for proof results.
///
/// The internal store is a plain `Vec` intentionally: the default capacity is
/// 256 entries so linear scan overhead is negligible, and it avoids the
/// overhead of a hash-map for this small working set.
#[derive(Debug)]
pub struct ProofCachingLayer {
    /// All currently cached proofs.
    entries: Vec<CachedProof>,
    /// Cache behaviour configuration.
    config: ProofCacheConfig,
    /// Cumulative statistics.
    stats: ProofCacheStats,
}

impl ProofCachingLayer {
    /// Create a new caching layer with the supplied configuration.
    pub fn new(config: ProofCacheConfig) -> Self {
        Self {
            entries: Vec::new(),
            config,
            stats: ProofCacheStats::default(),
        }
    }

    /// Look up a proof by key.
    ///
    /// - Stale entries (TTL exceeded) are skipped and treated as misses.
    /// - On a hit, `access_count` and `last_accessed_secs` of the entry are
    ///   updated in place, and `stats.hits` is incremented.
    /// - On a miss, `stats.misses` is incremented.
    ///
    /// Returns a shared reference into `self.entries` on success.
    pub fn lookup(&mut self, key: &ProofCacheKey, now_secs: u64) -> Option<&CachedProof> {
        let ttl = self.config.ttl_secs;

        // Find a non-stale entry that matches the key.
        let pos = self
            .entries
            .iter()
            .position(|e| &e.key == key && !e.is_stale(ttl, now_secs));

        match pos {
            Some(idx) => {
                // Update bookkeeping fields.
                self.entries[idx].access_count += 1;
                self.entries[idx].last_accessed_secs = now_secs;
                self.stats.hits += 1;
                Some(&self.entries[idx])
            }
            None => {
                self.stats.misses += 1;
                None
            }
        }
    }

    /// Insert a proof into the cache.
    ///
    /// - If an entry with the same key already exists it is replaced.
    /// - If the cache is at capacity, the entry with the **lowest**
    ///   `access_count` is evicted first (LFU policy), and
    ///   `stats.evictions` is incremented.
    pub fn insert(&mut self, proof: CachedProof) {
        // Replace existing entry with the same key.
        if let Some(idx) = self.entries.iter().position(|e| e.key == proof.key) {
            self.entries[idx] = proof;
            return;
        }

        // Evict the least-frequently-used entry when at capacity.
        if self.entries.len() >= self.config.max_entries {
            if let Some(lfu_idx) = self
                .entries
                .iter()
                .enumerate()
                .min_by_key(|(_, e)| e.access_count)
                .map(|(i, _)| i)
            {
                self.entries.swap_remove(lfu_idx);
                self.stats.evictions += 1;
            }
        }

        self.entries.push(proof);
    }

    /// Remove all entries whose `key.kb_version == old_version` (when
    /// `invalidate_on_kb_change` is `true`).
    ///
    /// `stats.invalidations` is incremented by the number of entries removed.
    pub fn invalidate_kb_version(&mut self, old_version: u64) {
        if !self.config.invalidate_on_kb_change {
            return;
        }

        let before = self.entries.len();
        self.entries.retain(|e| e.key.kb_version != old_version);
        let removed = before - self.entries.len();
        self.stats.invalidations += removed as u64;
    }

    /// Remove all stale entries and return the number removed.
    pub fn evict_stale(&mut self, now_secs: u64) -> usize {
        let ttl = self.config.ttl_secs;
        let before = self.entries.len();
        self.entries.retain(|e| !e.is_stale(ttl, now_secs));
        before - self.entries.len()
    }

    /// Return a shared reference to the current statistics.
    pub fn stats(&self) -> &ProofCacheStats {
        &self.stats
    }

    /// Number of entries currently in the cache.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// `true` if the cache holds no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl Default for ProofCachingLayer {
    fn default() -> Self {
        Self::new(ProofCacheConfig::default())
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── tiny clock shim ──────────────────────────────────────────────────────
    const T0: u64 = 1_000_000; // arbitrary epoch offset

    fn default_layer() -> ProofCachingLayer {
        ProofCachingLayer::new(ProofCacheConfig::default())
    }

    /// Build a minimal [`CachedProof`] for tests.
    fn make_proof(
        key: ProofCacheKey,
        proved: bool,
        bindings: Vec<(String, String)>,
        now_secs: u64,
    ) -> CachedProof {
        CachedProof {
            key,
            proved,
            bindings,
            proof_depth: 1,
            cached_at_secs: now_secs,
            last_accessed_secs: now_secs,
            access_count: 0,
        }
    }

    fn proof_at(goal: &str, kb_version: u64, proved: bool, now: u64) -> CachedProof {
        let key = ProofCacheKey::from_goal(goal, kb_version);
        make_proof(key, proved, vec![], now)
    }

    // ── 1. lookup miss ───────────────────────────────────────────────────────
    #[test]
    fn test_lookup_miss_returns_none() {
        let mut layer = default_layer();
        let key = ProofCacheKey::from_goal("missing_goal", 1);
        let result = layer.lookup(&key, T0);
        assert!(result.is_none());
        assert_eq!(layer.stats().misses, 1);
        assert_eq!(layer.stats().hits, 0);
    }

    // ── 2. insert then lookup hit ────────────────────────────────────────────
    #[test]
    fn test_insert_then_lookup_hit() {
        let mut layer = default_layer();
        let proof = proof_at("parent(a, b)", 1, true, T0);
        let key = proof.key;
        layer.insert(proof);

        let result = layer.lookup(&key, T0);
        assert!(result.is_some());
        assert_eq!(layer.stats().hits, 1);
        assert_eq!(layer.stats().misses, 0);
    }

    // ── 3. stale entry is skipped (treated as miss) ──────────────────────────
    #[test]
    fn test_stale_entry_skipped() {
        let config = ProofCacheConfig {
            ttl_secs: 60,
            ..Default::default()
        };
        let mut layer = ProofCachingLayer::new(config);
        let proof = proof_at("ancestor(a, c)", 1, true, T0);
        let key = proof.key;
        layer.insert(proof);

        // Advance time past TTL
        let future = T0 + 61;
        let result = layer.lookup(&key, future);
        assert!(result.is_none(), "stale entry should not be returned");
        assert_eq!(layer.stats().misses, 1);
        assert_eq!(layer.stats().hits, 0);
    }

    // ── 4. LFU eviction ──────────────────────────────────────────────────────
    #[test]
    fn test_lfu_eviction() {
        let config = ProofCacheConfig {
            max_entries: 2,
            ..Default::default()
        };
        let mut layer = ProofCachingLayer::new(config);

        let proof_a = proof_at("goal_a", 1, true, T0);
        let proof_b = proof_at("goal_b", 1, true, T0);
        let key_a = proof_a.key;
        let key_b = proof_b.key;

        layer.insert(proof_a);
        layer.insert(proof_b);

        // Access goal_a once so it has access_count = 1; goal_b stays at 0.
        let _ = layer.lookup(&key_a, T0);

        // Inserting a third entry should evict goal_b (access_count = 0).
        let proof_c = proof_at("goal_c", 1, true, T0);
        let key_c = proof_c.key;
        layer.insert(proof_c);

        assert_eq!(layer.stats().evictions, 1);
        assert_eq!(layer.len(), 2);

        // goal_b should be gone, goal_a and goal_c should remain.
        assert!(layer.lookup(&key_b, T0).is_none());
        assert!(layer.lookup(&key_a, T0).is_some());
        assert!(layer.lookup(&key_c, T0).is_some());
    }

    // ── 5. replace same key ──────────────────────────────────────────────────
    #[test]
    fn test_replace_same_key() {
        let mut layer = default_layer();
        let key = ProofCacheKey::from_goal("goal_replace", 1);

        let proof_v1 = CachedProof {
            key,
            proved: false,
            bindings: vec![],
            proof_depth: 1,
            cached_at_secs: T0,
            last_accessed_secs: T0,
            access_count: 0,
        };
        layer.insert(proof_v1);

        let proof_v2 = CachedProof {
            key,
            proved: true,
            bindings: vec![("X".to_string(), "alice".to_string())],
            proof_depth: 3,
            cached_at_secs: T0 + 1,
            last_accessed_secs: T0 + 1,
            access_count: 0,
        };
        layer.insert(proof_v2);

        // Should still have only one entry (replaced, not duplicated).
        assert_eq!(layer.len(), 1);
        assert_eq!(layer.stats().evictions, 0);

        let result = layer.lookup(&key, T0 + 1).expect("entry should exist");
        assert!(result.proved);
        assert_eq!(result.bindings.len(), 1);
    }

    // ── 6. invalidate_kb_version removes correct entries ────────────────────
    #[test]
    fn test_invalidate_kb_version_removes_correct_entries() {
        let mut layer = default_layer();

        layer.insert(proof_at("g1", 1, true, T0));
        layer.insert(proof_at("g2", 1, true, T0));
        layer.insert(proof_at("g3", 2, true, T0)); // different version

        layer.invalidate_kb_version(1);

        assert_eq!(layer.len(), 1, "only version-2 entry should remain");
        assert_eq!(layer.stats().invalidations, 2);

        // The version-2 entry must still be accessible.
        let key3 = ProofCacheKey::from_goal("g3", 2);
        assert!(layer.lookup(&key3, T0).is_some());
    }

    // ── 7. evict_stale count ─────────────────────────────────────────────────
    #[test]
    fn test_evict_stale_count() {
        let config = ProofCacheConfig {
            ttl_secs: 10,
            ..Default::default()
        };
        let mut layer = ProofCachingLayer::new(config);

        layer.insert(proof_at("gs1", 1, true, T0));
        layer.insert(proof_at("gs2", 1, true, T0));
        layer.insert(proof_at("gs3", 1, true, T0 + 5)); // not yet stale at T0+11

        let removed = layer.evict_stale(T0 + 11);
        assert_eq!(removed, 2, "two entries should be stale and removed");
        assert_eq!(layer.len(), 1);
    }

    // ── 8. hit_rate computation ──────────────────────────────────────────────
    #[test]
    fn test_hit_rate() {
        let mut layer = default_layer();
        let proof = proof_at("hr_goal", 1, true, T0);
        let key = proof.key;
        layer.insert(proof);

        // 3 hits
        layer.lookup(&key, T0);
        layer.lookup(&key, T0);
        layer.lookup(&key, T0);

        // 1 miss
        let absent = ProofCacheKey::from_goal("absent", 1);
        layer.lookup(&absent, T0);

        let rate = layer.stats().hit_rate();
        // 3/(3+1) = 0.75
        assert!((rate - 0.75).abs() < 1e-9, "expected 0.75, got {}", rate);
    }

    // ── 9. access_count increments on each lookup hit ───────────────────────
    #[test]
    fn test_access_count_increments() {
        let mut layer = default_layer();
        let proof = proof_at("access_goal", 1, true, T0);
        let key = proof.key;
        layer.insert(proof);

        layer.lookup(&key, T0);
        layer.lookup(&key, T0);
        layer.lookup(&key, T0);

        // Inspect the entry directly via a final lookup.
        let entry = layer.lookup(&key, T0).expect("entry present");
        assert_eq!(entry.access_count, 4);
    }

    // ── 10. fnv1a_hash is deterministic ─────────────────────────────────────
    #[test]
    fn test_fnv1a_hash_deterministic() {
        let s = "parent(alice, bob)";
        let h1 = fnv1a_hash(s);
        let h2 = fnv1a_hash(s);
        assert_eq!(h1, h2, "hash must be deterministic");
        assert_ne!(h1, 0, "hash should not be zero for non-empty input");
    }

    // ── 11. fnv1a_hash different strings differ ──────────────────────────────
    #[test]
    fn test_fnv1a_hash_different_strings_differ() {
        let h1 = fnv1a_hash("ancestor(a, b)");
        let h2 = fnv1a_hash("ancestor(b, a)");
        assert_ne!(h1, h2, "hash of different strings should differ");
    }

    // ── 12. invalidate_on_kb_change=false skips invalidation ────────────────
    #[test]
    fn test_invalidate_on_kb_change_false_skips() {
        let config = ProofCacheConfig {
            invalidate_on_kb_change: false,
            ..Default::default()
        };
        let mut layer = ProofCachingLayer::new(config);

        layer.insert(proof_at("g1", 1, true, T0));
        layer.insert(proof_at("g2", 1, true, T0));

        layer.invalidate_kb_version(1);

        // Nothing should have been removed.
        assert_eq!(layer.len(), 2);
        assert_eq!(layer.stats().invalidations, 0);
    }

    // ── 13. empty cache has zero hit_rate ────────────────────────────────────
    #[test]
    fn test_empty_cache_hit_rate_is_zero() {
        let layer = default_layer();
        assert_eq!(layer.stats().hit_rate(), 0.0);
    }

    // ── 14. is_stale boundary conditions ────────────────────────────────────
    #[test]
    fn test_is_stale_boundary() {
        let proof = CachedProof {
            key: ProofCacheKey::new(1, 1),
            proved: true,
            bindings: vec![],
            proof_depth: 0,
            cached_at_secs: 100,
            last_accessed_secs: 100,
            access_count: 0,
        };
        // Exactly at TTL boundary: stale
        assert!(proof.is_stale(50, 150));
        // One second before: not stale
        assert!(!proof.is_stale(50, 149));
        // Far past TTL: stale
        assert!(proof.is_stale(50, 9999));
    }

    // ── 15. evict_stale on empty cache returns 0 ────────────────────────────
    #[test]
    fn test_evict_stale_empty_cache() {
        let mut layer = default_layer();
        let removed = layer.evict_stale(T0 + 10_000);
        assert_eq!(removed, 0);
    }

    // ── 16. fresh entries survive evict_stale ───────────────────────────────
    #[test]
    fn test_evict_stale_fresh_entries_survive() {
        let config = ProofCacheConfig {
            ttl_secs: 300,
            ..Default::default()
        };
        let mut layer = ProofCachingLayer::new(config);

        layer.insert(proof_at("fresh1", 1, true, T0));
        layer.insert(proof_at("fresh2", 1, true, T0));

        // Only 10 seconds have passed — both entries are still fresh.
        let removed = layer.evict_stale(T0 + 10);
        assert_eq!(removed, 0);
        assert_eq!(layer.len(), 2);
    }

    // ── 17. last_accessed_secs updated on hit ───────────────────────────────
    #[test]
    fn test_last_accessed_secs_updated_on_hit() {
        let mut layer = default_layer();
        let proof = proof_at("la_goal", 1, true, T0);
        let key = proof.key;
        layer.insert(proof);

        let later = T0 + 42;
        {
            let entry = layer.lookup(&key, later).expect("entry present");
            assert_eq!(entry.last_accessed_secs, later);
        }
    }

    // ── 18. insert does not evict when below capacity ────────────────────────
    #[test]
    fn test_no_eviction_below_capacity() {
        let config = ProofCacheConfig {
            max_entries: 10,
            ..Default::default()
        };
        let mut layer = ProofCachingLayer::new(config);

        for i in 0..10u64 {
            layer.insert(proof_at(&format!("goal_{i}"), 1, true, T0));
        }
        assert_eq!(layer.stats().evictions, 0);
        assert_eq!(layer.len(), 10);
    }

    // ── 19. bindings preserved through insert/lookup cycle ──────────────────
    #[test]
    fn test_bindings_preserved() {
        let mut layer = default_layer();
        let key = ProofCacheKey::from_goal("bound_goal", 1);
        let proof = CachedProof {
            key,
            proved: true,
            bindings: vec![
                ("X".to_string(), "alice".to_string()),
                ("Y".to_string(), "bob".to_string()),
            ],
            proof_depth: 2,
            cached_at_secs: T0,
            last_accessed_secs: T0,
            access_count: 0,
        };
        layer.insert(proof);

        let entry = layer.lookup(&key, T0).expect("entry present");
        assert_eq!(entry.bindings.len(), 2);
        assert_eq!(entry.bindings[0], ("X".to_string(), "alice".to_string()));
        assert_eq!(entry.bindings[1], ("Y".to_string(), "bob".to_string()));
    }
}
