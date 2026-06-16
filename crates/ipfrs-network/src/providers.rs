//! Provider record cache with TTL
//!
//! This module implements a cache for DHT provider records to reduce
//! network load by avoiding redundant GET_PROVIDERS queries.

use cid::Cid;
use libp2p::PeerId;
use parking_lot::RwLock;
use serde::Serialize;
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};
use tracing::{debug, info};

/// Default TTL for provider records (1 hour)
const DEFAULT_TTL: Duration = Duration::from_secs(3600);

/// Default maximum cache entries
const DEFAULT_MAX_ENTRIES: usize = 10000;

/// Configuration for the provider cache
#[derive(Debug, Clone)]
pub struct ProviderCacheConfig {
    /// Time-to-live for cached provider records
    pub ttl: Duration,
    /// Maximum number of entries in the cache
    pub max_entries: usize,
    /// Minimum providers per CID before refresh
    pub min_providers: usize,
}

impl Default for ProviderCacheConfig {
    fn default() -> Self {
        Self {
            ttl: DEFAULT_TTL,
            max_entries: DEFAULT_MAX_ENTRIES,
            min_providers: 1,
        }
    }
}

/// A cached provider record
#[derive(Debug, Clone)]
struct CachedProviders {
    /// Provider peer IDs
    providers: HashSet<PeerId>,
    /// Time when the record was cached
    cached_at: Instant,
    /// Last access time (for LRU eviction)
    last_accessed: Instant,
    /// Number of times this record was accessed
    access_count: u64,
}

impl CachedProviders {
    fn new(providers: HashSet<PeerId>) -> Self {
        let now = Instant::now();
        Self {
            providers,
            cached_at: now,
            last_accessed: now,
            access_count: 0,
        }
    }

    fn is_expired(&self, ttl: Duration) -> bool {
        self.cached_at.elapsed() > ttl
    }

    fn touch(&mut self) {
        self.last_accessed = Instant::now();
        self.access_count += 1;
    }
}

/// Provider record cache
pub struct ProviderCache {
    /// Configuration
    config: ProviderCacheConfig,
    /// Cached provider records
    cache: RwLock<HashMap<Cid, CachedProviders>>,
    /// Cache statistics
    stats: RwLock<CacheStats>,
}

/// Internal mutable statistics
#[derive(Default)]
struct CacheStats {
    hits: u64,
    misses: u64,
    evictions: u64,
    expirations: u64,
}

impl ProviderCache {
    /// Create a new provider cache with default configuration
    pub fn new() -> Self {
        Self::with_config(ProviderCacheConfig::default())
    }

    /// Create a new provider cache with custom configuration
    pub fn with_config(config: ProviderCacheConfig) -> Self {
        Self {
            config,
            cache: RwLock::new(HashMap::new()),
            stats: RwLock::new(CacheStats::default()),
        }
    }

    /// Get providers for a CID from cache
    ///
    /// Returns None if not cached or expired
    pub fn get(&self, cid: &Cid) -> Option<Vec<PeerId>> {
        let mut cache = self.cache.write();
        let mut stats = self.stats.write();

        if let Some(entry) = cache.get_mut(cid) {
            if entry.is_expired(self.config.ttl) {
                // Expired entry
                cache.remove(cid);
                stats.expirations += 1;
                stats.misses += 1;
                debug!("Provider cache expired for {}", cid);
                return None;
            }

            // Cache hit
            entry.touch();
            stats.hits += 1;
            debug!(
                "Provider cache hit for {} ({} providers)",
                cid,
                entry.providers.len()
            );
            return Some(entry.providers.iter().cloned().collect());
        }

        // Cache miss
        stats.misses += 1;
        None
    }

    /// Check if we have valid (non-expired) providers cached
    pub fn has_providers(&self, cid: &Cid) -> bool {
        let cache = self.cache.read();
        if let Some(entry) = cache.get(cid) {
            !entry.is_expired(self.config.ttl) && !entry.providers.is_empty()
        } else {
            false
        }
    }

    /// Check if providers need refresh (expired or below minimum)
    pub fn needs_refresh(&self, cid: &Cid) -> bool {
        let cache = self.cache.read();
        if let Some(entry) = cache.get(cid) {
            entry.is_expired(self.config.ttl) || entry.providers.len() < self.config.min_providers
        } else {
            true
        }
    }

    /// Add or update providers for a CID
    pub fn put(&self, cid: Cid, providers: Vec<PeerId>) {
        let provider_set: HashSet<PeerId> = providers.into_iter().collect();

        if provider_set.is_empty() {
            debug!("Not caching empty provider list for {}", cid);
            return;
        }

        let mut cache = self.cache.write();

        // Check if we need to evict
        if cache.len() >= self.config.max_entries {
            self.evict_lru(&mut cache);
        }

        let count = provider_set.len();
        cache.insert(cid, CachedProviders::new(provider_set));
        info!("Cached {} providers for {}", count, cid);
    }

    /// Add a single provider to an existing cache entry
    pub fn add_provider(&self, cid: &Cid, provider: PeerId) {
        let mut cache = self.cache.write();

        if let Some(entry) = cache.get_mut(cid) {
            if !entry.is_expired(self.config.ttl) {
                entry.providers.insert(provider);
                entry.touch();
                debug!("Added provider {} to cache for {}", provider, cid);
            }
        }
    }

    /// Remove a provider from cache (e.g., when it disconnects)
    pub fn remove_provider(&self, cid: &Cid, provider: &PeerId) {
        let mut cache = self.cache.write();

        if let Some(entry) = cache.get_mut(cid) {
            entry.providers.remove(provider);
            debug!("Removed provider {} from cache for {}", provider, cid);
        }
    }

    /// Remove all entries for a CID
    pub fn invalidate(&self, cid: &Cid) {
        let mut cache = self.cache.write();
        cache.remove(cid);
        debug!("Invalidated cache for {}", cid);
    }

    /// Remove all expired entries
    pub fn cleanup_expired(&self) {
        let mut cache = self.cache.write();
        let mut stats = self.stats.write();
        let ttl = self.config.ttl;

        let before = cache.len();
        cache.retain(|_, entry| !entry.is_expired(ttl));
        let removed = before - cache.len();

        if removed > 0 {
            stats.expirations += removed as u64;
            info!("Cleaned up {} expired provider cache entries", removed);
        }
    }

    /// Clear the entire cache
    pub fn clear(&self) {
        let mut cache = self.cache.write();
        cache.clear();
        info!("Provider cache cleared");
    }

    /// Get cache statistics
    pub fn stats(&self) -> ProviderCacheStats {
        let cache = self.cache.read();
        let stats = self.stats.read();

        let total_providers: usize = cache.values().map(|e| e.providers.len()).sum();
        let hit_rate = if stats.hits + stats.misses > 0 {
            stats.hits as f64 / (stats.hits + stats.misses) as f64
        } else {
            0.0
        };

        ProviderCacheStats {
            entries: cache.len(),
            max_entries: self.config.max_entries,
            total_providers,
            hits: stats.hits,
            misses: stats.misses,
            hit_rate,
            evictions: stats.evictions,
            expirations: stats.expirations,
        }
    }

    /// Get the number of cached CIDs
    pub fn len(&self) -> usize {
        self.cache.read().len()
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.cache.read().is_empty()
    }

    /// Evict least recently used entries
    fn evict_lru(&self, cache: &mut HashMap<Cid, CachedProviders>) {
        // Evict 10% of entries or at least 1
        let to_evict = (self.config.max_entries / 10).max(1);

        let mut entries: Vec<_> = cache
            .iter()
            .map(|(cid, entry)| (*cid, entry.last_accessed))
            .collect();

        // Sort by last accessed (oldest first)
        entries.sort_by_key(|a| a.1);

        let mut stats = self.stats.write();
        for (cid, _) in entries.into_iter().take(to_evict) {
            cache.remove(&cid);
            stats.evictions += 1;
        }

        debug!("Evicted {} LRU cache entries", to_evict);
    }
}

impl Default for ProviderCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Provider cache statistics
#[derive(Debug, Clone, Serialize)]
pub struct ProviderCacheStats {
    /// Number of cached CIDs
    pub entries: usize,
    /// Maximum entries allowed
    pub max_entries: usize,
    /// Total providers across all entries
    pub total_providers: usize,
    /// Cache hits
    pub hits: u64,
    /// Cache misses
    pub misses: u64,
    /// Hit rate (0.0 - 1.0)
    pub hit_rate: f64,
    /// Number of entries evicted due to capacity
    pub evictions: u64,
    /// Number of entries expired
    pub expirations: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use multihash_codetable::{Code, MultihashDigest};

    fn make_cid(data: &[u8]) -> Cid {
        let hash = Code::Sha2_256.digest(data);
        Cid::new_v1(0x55, hash)
    }

    fn random_peer_id() -> PeerId {
        PeerId::random()
    }

    #[test]
    fn test_provider_cache_basic() {
        let cache = ProviderCache::new();
        let cid = make_cid(b"test data");
        let peer1 = random_peer_id();
        let peer2 = random_peer_id();

        // Initially empty
        assert!(cache.get(&cid).is_none());
        assert!(cache.needs_refresh(&cid));

        // Add providers
        cache.put(cid, vec![peer1, peer2]);

        // Should be cached now
        let providers = cache
            .get(&cid)
            .expect("test: cache should contain providers after put");
        assert_eq!(providers.len(), 2);
        assert!(providers.contains(&peer1));
        assert!(providers.contains(&peer2));
        assert!(cache.has_providers(&cid));
        assert!(!cache.needs_refresh(&cid));
    }

    #[test]
    fn test_provider_cache_add_remove() {
        let cache = ProviderCache::new();
        let cid = make_cid(b"test");
        let peer1 = random_peer_id();
        let peer2 = random_peer_id();
        let peer3 = random_peer_id();

        cache.put(cid, vec![peer1, peer2]);

        // Add a provider
        cache.add_provider(&cid, peer3);
        let providers = cache
            .get(&cid)
            .expect("test: cache should have 3 providers after add");
        assert_eq!(providers.len(), 3);

        // Remove a provider
        cache.remove_provider(&cid, &peer1);
        let providers = cache
            .get(&cid)
            .expect("test: cache should have 2 providers after remove");
        assert_eq!(providers.len(), 2);
        assert!(!providers.contains(&peer1));
    }

    #[test]
    fn test_provider_cache_expiration() {
        let config = ProviderCacheConfig {
            ttl: Duration::from_millis(50),
            ..Default::default()
        };
        let cache = ProviderCache::with_config(config);
        let cid = make_cid(b"expiring");
        let peer = random_peer_id();

        cache.put(cid, vec![peer]);
        assert!(cache.get(&cid).is_some());

        // Wait for expiration
        std::thread::sleep(Duration::from_millis(100));

        // Should be expired now
        assert!(cache.get(&cid).is_none());
        assert!(cache.needs_refresh(&cid));

        let stats = cache.stats();
        assert!(stats.expirations > 0);
    }

    #[test]
    fn test_provider_cache_lru_eviction() {
        let config = ProviderCacheConfig {
            ttl: Duration::from_secs(3600),
            max_entries: 5,
            ..Default::default()
        };
        let cache = ProviderCache::with_config(config);
        let peer = random_peer_id();

        // Add 5 entries
        for i in 0..5 {
            let cid = make_cid(&[i as u8]);
            cache.put(cid, vec![peer]);
        }

        assert_eq!(cache.len(), 5);

        // Access some entries to update their LRU time
        let cid_2 = make_cid(&[2]);
        let cid_3 = make_cid(&[3]);
        cache.get(&cid_2);
        cache.get(&cid_3);

        // Add a new entry, triggering eviction
        let new_cid = make_cid(&[100]);
        cache.put(new_cid, vec![peer]);

        // Should have evicted LRU entries
        assert!(cache.len() <= 5);

        let stats = cache.stats();
        assert!(stats.evictions > 0);
    }

    #[test]
    fn test_provider_cache_stats() {
        let cache = ProviderCache::new();
        let cid1 = make_cid(b"one");
        let cid2 = make_cid(b"two");
        let peer = random_peer_id();

        // Miss
        cache.get(&cid1);

        // Put and hit
        cache.put(cid1, vec![peer]);
        cache.get(&cid1);

        // Another miss
        cache.get(&cid2);

        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 2);
        assert_eq!(stats.entries, 1);
    }

    #[test]
    fn test_provider_cache_invalidate() {
        let cache = ProviderCache::new();
        let cid = make_cid(b"invalidate me");
        let peer = random_peer_id();

        cache.put(cid, vec![peer]);
        assert!(cache.has_providers(&cid));

        cache.invalidate(&cid);
        assert!(!cache.has_providers(&cid));
    }

    #[test]
    fn test_provider_cache_cleanup() {
        let config = ProviderCacheConfig {
            ttl: Duration::from_millis(10),
            ..Default::default()
        };
        let cache = ProviderCache::with_config(config);
        let peer = random_peer_id();

        // Add entries
        for i in 0..5 {
            let cid = make_cid(&[i as u8]);
            cache.put(cid, vec![peer]);
        }

        assert_eq!(cache.len(), 5);

        // Wait for expiration
        std::thread::sleep(Duration::from_millis(50));

        // Cleanup
        cache.cleanup_expired();
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_provider_cache_empty_providers_not_cached() {
        let cache = ProviderCache::new();
        let cid = make_cid(b"empty");

        cache.put(cid, vec![]);
        assert!(!cache.has_providers(&cid));
        assert_eq!(cache.len(), 0);
    }
}
