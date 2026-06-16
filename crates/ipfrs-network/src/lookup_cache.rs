//! DHT lookup result caching and parallel alpha query execution.
//!
//! This module provides:
//! - `LookupCache`: TTL-based CID → providers cache for DHT lookups
//! - `ParallelLookupExecutor`: Parallel DHT provider lookups with caching

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use futures::future::join_all;
use parking_lot::RwLock;

/// A cached result for a CID provider lookup.
#[derive(Debug, Clone)]
pub struct CachedProviders {
    /// The CID string key for this entry.
    pub cid_str: String,
    /// List of peer ID strings that provide this CID.
    pub providers: Vec<String>,
    /// Millisecond timestamp at which this entry was inserted.
    pub cached_at_ms: u64,
    /// Time-to-live for this entry in milliseconds.
    pub ttl_ms: u64,
    /// Number of times this entry has been served from cache.
    pub hit_count: u64,
}

impl CachedProviders {
    /// Create a new `CachedProviders` entry.
    pub fn new(
        cid_str: impl Into<String>,
        providers: Vec<String>,
        now_ms: u64,
        ttl_ms: u64,
    ) -> Self {
        Self {
            cid_str: cid_str.into(),
            providers,
            cached_at_ms: now_ms,
            ttl_ms,
            hit_count: 0,
        }
    }

    /// Returns `true` when the entry has expired at `now_ms`.
    #[inline]
    pub fn is_expired(&self, now_ms: u64) -> bool {
        now_ms > self.cached_at_ms + self.ttl_ms
    }

    /// Returns the age of this entry in milliseconds, saturating at zero.
    #[inline]
    pub fn age_ms(&self, now_ms: u64) -> u64 {
        now_ms.saturating_sub(self.cached_at_ms)
    }
}

/// Configuration for the [`LookupCache`].
#[derive(Debug, Clone)]
pub struct LookupCacheConfig {
    /// TTL for positive results (providers found). Default: 300 000 ms (5 minutes).
    pub positive_ttl_ms: u64,
    /// TTL for negative results (no providers). Default: 30 000 ms (30 seconds).
    pub negative_ttl_ms: u64,
    /// Maximum number of entries before eviction is triggered. Default: 10 000.
    pub max_entries: usize,
}

impl Default for LookupCacheConfig {
    fn default() -> Self {
        Self {
            positive_ttl_ms: 300_000,
            negative_ttl_ms: 30_000,
            max_entries: 10_000,
        }
    }
}

/// TTL-based CID → providers cache for DHT lookups.
pub struct LookupCache {
    config: LookupCacheConfig,
    entries: RwLock<HashMap<String, CachedProviders>>,
    hits: AtomicU64,
    misses: AtomicU64,
    evictions: AtomicU64,
}

/// Snapshot statistics for a [`LookupCache`].
#[derive(Debug, Clone)]
pub struct LookupCacheStats {
    /// Current number of entries (including expired ones not yet evicted).
    pub total_entries: usize,
    /// Total cache hits served.
    pub hits: u64,
    /// Total cache misses.
    pub misses: u64,
    /// Total entries removed by eviction.
    pub evictions: u64,
    /// `hits / (hits + misses)`, or `0.0` when no lookups have been made.
    pub hit_rate: f64,
}

impl LookupCache {
    /// Create a new `LookupCache` wrapped in an `Arc`.
    pub fn new(config: LookupCacheConfig) -> Arc<Self> {
        Arc::new(Self {
            config,
            entries: RwLock::new(HashMap::new()),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
        })
    }

    /// Look up providers for a CID.
    ///
    /// Returns `None` on a cache miss or when the entry has expired.
    /// On a hit the `hit_count` for the entry is incremented and the
    /// provider list is returned.
    pub fn get(&self, cid_str: &str, now_ms: u64) -> Option<Vec<String>> {
        // First try with a read lock to avoid write contention.
        {
            let entries = self.entries.read();
            if let Some(entry) = entries.get(cid_str) {
                if entry.is_expired(now_ms) {
                    // Expired — treat as miss; removal happens lazily.
                    self.misses.fetch_add(1, Ordering::Relaxed);
                    return None;
                }
                // Hot path: record the providers before upgrading the lock.
                let providers = entry.providers.clone();
                drop(entries);

                // Upgrade to write lock to bump hit_count.
                {
                    let mut entries = self.entries.write();
                    if let Some(entry) = entries.get_mut(cid_str) {
                        entry.hit_count = entry.hit_count.saturating_add(1);
                    }
                }
                self.hits.fetch_add(1, Ordering::Relaxed);
                return Some(providers);
            }
        }

        self.misses.fetch_add(1, Ordering::Relaxed);
        None
    }

    /// Store a positive result (providers found) for `cid_str`.
    ///
    /// If the cache is at capacity the new entry still replaces any existing
    /// entry for the same CID; overall capacity enforcement is left to
    /// [`evict_expired`](Self::evict_expired).
    pub fn put(&self, cid_str: impl Into<String>, providers: Vec<String>, now_ms: u64) {
        let key = cid_str.into();
        let ttl_ms = self.config.positive_ttl_ms;
        let entry = CachedProviders::new(key.clone(), providers, now_ms, ttl_ms);
        let mut entries = self.entries.write();

        // Enforce max_entries by evicting expired items first when at capacity.
        if !entries.contains_key(&key) && entries.len() >= self.config.max_entries {
            let expired_keys: Vec<String> = entries
                .iter()
                .filter(|(_, v)| v.is_expired(now_ms))
                .map(|(k, _)| k.clone())
                .collect();
            let removed = expired_keys.len();
            for k in expired_keys {
                entries.remove(&k);
            }
            self.evictions.fetch_add(removed as u64, Ordering::Relaxed);
        }

        entries.insert(key, entry);
    }

    /// Store a negative result (no providers found) for `cid_str`.
    ///
    /// Negative entries use [`LookupCacheConfig::negative_ttl_ms`] as TTL and
    /// an empty provider list.
    pub fn put_negative(&self, cid_str: impl Into<String>, now_ms: u64) {
        let key = cid_str.into();
        let ttl_ms = self.config.negative_ttl_ms;
        let entry = CachedProviders::new(key.clone(), Vec::new(), now_ms, ttl_ms);
        let mut entries = self.entries.write();
        entries.insert(key, entry);
    }

    /// Invalidate the entry for `cid_str`.
    ///
    /// Returns `true` if an entry existed and was removed.
    pub fn invalidate(&self, cid_str: &str) -> bool {
        let mut entries = self.entries.write();
        entries.remove(cid_str).is_some()
    }

    /// Remove all expired entries.
    ///
    /// Returns the number of entries removed.
    pub fn evict_expired(&self, now_ms: u64) -> usize {
        let mut entries = self.entries.write();
        let expired_keys: Vec<String> = entries
            .iter()
            .filter(|(_, v)| v.is_expired(now_ms))
            .map(|(k, _)| k.clone())
            .collect();
        let count = expired_keys.len();
        for k in expired_keys {
            entries.remove(&k);
        }
        self.evictions.fetch_add(count as u64, Ordering::Relaxed);
        count
    }

    /// Return a snapshot of current cache statistics.
    pub fn stats(&self) -> LookupCacheStats {
        let total_entries = self.entries.read().len();
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        let evictions = self.evictions.load(Ordering::Relaxed);
        let total = hits + misses;
        let hit_rate = if total == 0 {
            0.0
        } else {
            hits as f64 / total as f64
        };
        LookupCacheStats {
            total_entries,
            hits,
            misses,
            evictions,
            hit_rate,
        }
    }

    /// Total number of cached entries (including expired ones not yet evicted).
    pub fn len(&self) -> usize {
        self.entries.read().len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.read().is_empty()
    }
}

// ── ParallelLookupExecutor ────────────────────────────────────────────────────

/// Configuration for [`ParallelLookupExecutor`].
#[derive(Debug, Clone)]
pub struct ParallelLookupConfig {
    /// Number of concurrent DHT queries (alpha). Default: 3 (Kademlia default).
    pub alpha: usize,
    /// Maximum total lookups per request. Default: 20.
    pub max_lookups: usize,
    /// Per-lookup timeout in milliseconds. Default: 5 000.
    pub timeout_ms: u64,
}

impl Default for ParallelLookupConfig {
    fn default() -> Self {
        Self {
            alpha: 3,
            max_lookups: 20,
            timeout_ms: 5_000,
        }
    }
}

/// Result of a parallel provider lookup.
#[derive(Debug, Clone)]
pub struct ParallelLookupResult {
    /// The CID that was looked up.
    pub cid_str: String,
    /// Deduplicated list of peer ID strings that provide the CID.
    pub providers: Vec<String>,
    /// `true` when the result was served from the local cache.
    pub from_cache: bool,
    /// Number of DHT queries actually issued (0 on a cache hit).
    pub lookup_count: usize,
    /// Wall-clock time in milliseconds from call start to return.
    pub elapsed_ms: u64,
}

/// Executes parallel DHT provider lookups with integrated caching.
pub struct ParallelLookupExecutor {
    cache: Arc<LookupCache>,
    config: ParallelLookupConfig,
    total_lookups: AtomicU64,
    cache_saves: AtomicU64,
}

impl ParallelLookupExecutor {
    /// Create a new `ParallelLookupExecutor` wrapped in an `Arc`.
    pub fn new(cache: Arc<LookupCache>, config: ParallelLookupConfig) -> Arc<Self> {
        Arc::new(Self {
            cache,
            config,
            total_lookups: AtomicU64::new(0),
            cache_saves: AtomicU64::new(0),
        })
    }

    /// Look up providers for `cid_str`, using the cache first.
    ///
    /// # Cache hit
    /// Returns immediately with `from_cache = true` and `lookup_count = 0`.
    ///
    /// # Cache miss
    /// Runs `alpha` parallel calls to `query_fn(cid_str)`, merges and
    /// deduplicates the results, stores them in the cache, and returns with
    /// `from_cache = false` and `lookup_count = alpha`.
    pub async fn lookup<F, Fut>(
        &self,
        cid_str: &str,
        now_ms: u64,
        query_fn: F,
    ) -> ParallelLookupResult
    where
        F: Fn(String) -> Fut + Send + Sync,
        Fut: std::future::Future<Output = Vec<String>> + Send,
    {
        self.total_lookups.fetch_add(1, Ordering::Relaxed);
        let t0 = now_ms; // caller-provided; real wall clock would be std::time::Instant::now()

        // 1. Check cache.
        if let Some(providers) = self.cache.get(cid_str, now_ms) {
            self.cache_saves.fetch_add(1, Ordering::Relaxed);
            return ParallelLookupResult {
                cid_str: cid_str.to_string(),
                providers,
                from_cache: true,
                lookup_count: 0,
                elapsed_ms: 0,
            };
        }

        // 2. Run `alpha` parallel DHT queries.
        let alpha = self.config.alpha;
        let futures: Vec<_> = (0..alpha).map(|_| query_fn(cid_str.to_string())).collect();
        let results: Vec<Vec<String>> = join_all(futures).await;

        // 3. Merge and deduplicate.
        let mut seen = std::collections::HashSet::new();
        let mut providers: Vec<String> = Vec::new();
        for batch in results {
            for peer in batch {
                if seen.insert(peer.clone()) {
                    providers.push(peer);
                }
            }
        }

        // 4. Store in cache.
        if providers.is_empty() {
            self.cache.put_negative(cid_str, now_ms);
        } else {
            self.cache.put(cid_str, providers.clone(), now_ms);
        }

        // 5. Return result.
        // elapsed_ms is 0 here because we use caller-supplied `now_ms`;
        // a real implementation would compute Instant::now() - t0.
        let elapsed_ms = now_ms.saturating_sub(t0);
        ParallelLookupResult {
            cid_str: cid_str.to_string(),
            providers,
            from_cache: false,
            lookup_count: alpha,
            elapsed_ms,
        }
    }

    /// Total number of cache hits served (lookups avoided).
    pub fn cache_saves(&self) -> u64 {
        self.cache_saves.load(Ordering::Relaxed)
    }

    /// Total number of lookup calls (cache hit or miss).
    pub fn total_lookups(&self) -> u64 {
        self.total_lookups.load(Ordering::Relaxed)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn now() -> u64 {
        // Use a fixed base time so tests are deterministic.
        1_000_000_u64
    }

    fn cache() -> Arc<LookupCache> {
        LookupCache::new(LookupCacheConfig::default())
    }

    // ── LookupCache unit tests ─────────────────────────────────────────────

    #[test]
    fn test_cache_miss_on_empty() {
        let c = cache();
        assert!(c.get("QmEmpty", now()).is_none());
        let stats = c.stats();
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.hits, 0);
    }

    #[test]
    fn test_cache_put_and_get() {
        let c = cache();
        let providers = vec!["peer1".to_string(), "peer2".to_string()];
        c.put("QmFoo", providers.clone(), now());
        let result = c.get("QmFoo", now()).expect("should be a cache hit");
        assert_eq!(result, providers);
        assert_eq!(c.stats().hits, 1);
    }

    #[test]
    fn test_cache_expired_entry() {
        let c = cache();
        // Use a custom config with ttl_ms = 0 for the entry via direct construction.
        let entry_time = now();
        let providers = vec!["peer_x".to_string()];
        {
            let mut entries = c.entries.write();
            entries.insert(
                "QmExpired".to_string(),
                CachedProviders::new("QmExpired", providers, entry_time, 0),
            );
        }
        // Query at exactly entry_time + 1 so the entry is expired.
        let result = c.get("QmExpired", entry_time + 1);
        assert!(result.is_none(), "expired entry should be a miss");
    }

    #[test]
    fn test_cache_negative_result() {
        let c = cache();
        c.put_negative("QmNeg", now());
        let result = c
            .get("QmNeg", now())
            .expect("negative entry should be present");
        assert!(result.is_empty(), "negative entry should have no providers");
    }

    #[test]
    fn test_cache_invalidate() {
        let c = cache();
        c.put("QmInv", vec!["peerA".to_string()], now());
        assert!(c.get("QmInv", now()).is_some());
        let removed = c.invalidate("QmInv");
        assert!(removed);
        assert!(c.get("QmInv", now()).is_none());
        // Invalidating a non-existent key returns false.
        assert!(!c.invalidate("QmDoesNotExist"));
    }

    #[test]
    fn test_cache_evict_expired() {
        let c = cache();
        let t = now();
        // Insert one normal and one immediately-expired entry.
        c.put("QmGood", vec!["peerG".to_string()], t);
        {
            let mut entries = c.entries.write();
            entries.insert(
                "QmBad".to_string(),
                CachedProviders::new("QmBad", vec!["peerB".to_string()], t, 0),
            );
        }
        assert_eq!(c.len(), 2);
        let removed = c.evict_expired(t + 1);
        assert_eq!(removed, 1);
        assert_eq!(c.len(), 1);
        assert!(c.get("QmGood", t + 1).is_some());
        assert_eq!(c.stats().evictions, 1);
    }

    #[test]
    fn test_cache_stats_hit_rate() {
        let c = cache();
        let t = now();
        c.put("QmRate", vec!["peerR".to_string()], t);
        // One hit, one miss.
        c.get("QmRate", t);
        c.get("QmMissing", t);
        let stats = c.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        let expected = 0.5_f64;
        let diff = (stats.hit_rate - expected).abs();
        assert!(
            diff < 1e-9,
            "hit_rate should be 0.5, got {}",
            stats.hit_rate
        );
    }

    #[test]
    fn test_cache_put_overwrites() {
        let c = cache();
        let t = now();
        c.put("QmOver", vec!["old".to_string()], t);
        c.put("QmOver", vec!["new1".to_string(), "new2".to_string()], t);
        let result = c.get("QmOver", t).expect("should hit");
        assert_eq!(result, vec!["new1".to_string(), "new2".to_string()]);
    }

    // ── ParallelLookupExecutor tests ───────────────────────────────────────

    #[tokio::test]
    async fn test_parallel_lookup_cache_hit() {
        let c = LookupCache::new(LookupCacheConfig::default());
        let t = now();
        c.put("QmHit", vec!["peerCached".to_string()], t);

        let exec = ParallelLookupExecutor::new(Arc::clone(&c), ParallelLookupConfig::default());

        let result = exec
            .lookup("QmHit", t, |_cid| async {
                vec!["shouldNotBeCalled".to_string()]
            })
            .await;

        assert!(result.from_cache);
        assert_eq!(result.lookup_count, 0);
        assert_eq!(result.providers, vec!["peerCached".to_string()]);
        assert_eq!(exec.cache_saves(), 1);
    }

    #[tokio::test]
    async fn test_parallel_lookup_cache_miss() {
        let c = LookupCache::new(LookupCacheConfig::default());
        let t = now();

        let config = ParallelLookupConfig {
            alpha: 3,
            ..Default::default()
        };
        let exec = ParallelLookupExecutor::new(Arc::clone(&c), config);

        // Each alpha query returns a distinct provider.
        let call_count = Arc::new(AtomicU64::new(0));
        let call_count_clone = Arc::clone(&call_count);
        let result = exec
            .lookup("QmMiss", t, move |_cid| {
                let idx = call_count_clone.fetch_add(1, Ordering::Relaxed);
                async move { vec![format!("peer_{}", idx)] }
            })
            .await;

        assert!(!result.from_cache);
        assert_eq!(result.lookup_count, 3);
        // All three distinct providers should be present.
        assert_eq!(result.providers.len(), 3);
        // Result should now be cached.
        let cached = c.get("QmMiss", t).expect("should be cached after miss");
        assert_eq!(cached.len(), 3);
    }

    #[tokio::test]
    async fn test_parallel_lookup_dedup() {
        let c = LookupCache::new(LookupCacheConfig::default());
        let t = now();

        let config = ParallelLookupConfig {
            alpha: 3,
            ..Default::default()
        };
        let exec = ParallelLookupExecutor::new(Arc::clone(&c), config);

        // All alpha queries return the same provider → should be deduplicated.
        let result = exec
            .lookup("QmDedup", t, |_cid| async { vec!["samePeer".to_string()] })
            .await;

        assert_eq!(result.providers.len(), 1);
        assert_eq!(result.providers[0], "samePeer");
    }

    #[tokio::test]
    async fn test_cache_saves_counter() {
        let c = LookupCache::new(LookupCacheConfig::default());
        let t = now();
        c.put("QmSave", vec!["p1".to_string()], t);

        let exec = ParallelLookupExecutor::new(Arc::clone(&c), ParallelLookupConfig::default());

        // Three cache-hit lookups.
        for _ in 0..3 {
            exec.lookup("QmSave", t, |_| async { vec![] }).await;
        }

        assert_eq!(exec.cache_saves(), 3);
        assert_eq!(exec.total_lookups(), 3);
    }

    #[test]
    fn test_default_config() {
        let cache_cfg = LookupCacheConfig::default();
        assert_eq!(cache_cfg.positive_ttl_ms, 300_000);
        assert_eq!(cache_cfg.negative_ttl_ms, 30_000);
        assert_eq!(cache_cfg.max_entries, 10_000);

        let lookup_cfg = ParallelLookupConfig::default();
        assert_eq!(lookup_cfg.alpha, 3);
        assert_eq!(lookup_cfg.max_lookups, 20);
        assert_eq!(lookup_cfg.timeout_ms, 5_000);
    }
}
