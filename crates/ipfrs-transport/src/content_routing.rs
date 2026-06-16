//! Content Routing Integration
//!
//! Provides DHT-based provider discovery, content advertising, and cache-aware routing
//! for global content discovery in the IPFS network.
//!
//! # Example
//!
//! ```
//! use ipfrs_transport::content_routing::{ContentRouter, ContentRoutingConfig};
//! use multihash::Multihash;
//! use cid::Cid;
//!
//! # #[tokio::main]
//! # async fn main() {
//! use libp2p::PeerId;
//!
//! // Create a content router with default config
//! let router = ContentRouter::new();
//!
//! // Create example CID
//! let hash = Multihash::wrap(0x12, &[1u8; 32]).unwrap();
//! let cid = Cid::new_v1(0x55, hash);
//!
//! // Create a dummy peer ID for testing
//! let peer_id = PeerId::random();
//!
//! // Advertise that we provide this content
//! router.advertise_content(cid, peer_id).await;
//!
//! // Find providers for content
//! let providers = router.find_providers(&cid).await;
//! assert!(!providers.is_empty());
//!
//! // Check how many providers we have
//! let count = router.provider_count(&cid);
//! println!("Number of providers: {}", count);
//! # }
//! ```

use cid::Cid;
use dashmap::DashMap;
use libp2p::PeerId;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

/// Configuration for content routing
#[derive(Debug, Clone)]
pub struct ContentRoutingConfig {
    /// Maximum number of providers to track per CID
    pub max_providers_per_cid: usize,
    /// Provider record TTL
    pub provider_record_ttl: Duration,
    /// Cache entry TTL
    pub cache_entry_ttl: Duration,
    /// Maximum cache size (number of entries)
    pub max_cache_size: usize,
    /// Number of parallel DHT queries
    pub dht_query_parallelism: usize,
    /// DHT query timeout
    pub dht_query_timeout: Duration,
    /// Whether to enable cache-aware routing
    pub enable_cache_aware_routing: bool,
}

impl Default for ContentRoutingConfig {
    fn default() -> Self {
        Self {
            max_providers_per_cid: 20,
            provider_record_ttl: Duration::from_secs(24 * 3600), // 24 hours
            cache_entry_ttl: Duration::from_secs(3600),          // 1 hour
            max_cache_size: 10_000,
            dht_query_parallelism: 3,
            dht_query_timeout: Duration::from_secs(30),
            enable_cache_aware_routing: true,
        }
    }
}

/// Provider record in the DHT
#[derive(Debug, Clone)]
pub struct ProviderRecord {
    /// Peer ID of the provider
    pub peer_id: PeerId,
    /// Timestamp when this record was added
    pub added_at: Instant,
    /// Time-to-live for this record
    pub ttl: Duration,
    /// Provider score (for ranking)
    pub score: f64,
    /// Number of successful retrievals from this provider
    pub successful_retrievals: u64,
    /// Number of failed retrievals from this provider
    pub failed_retrievals: u64,
}

impl ProviderRecord {
    fn new(peer_id: PeerId, ttl: Duration) -> Self {
        Self {
            peer_id,
            added_at: Instant::now(),
            ttl,
            score: 1.0,
            successful_retrievals: 0,
            failed_retrievals: 0,
        }
    }

    fn is_expired(&self) -> bool {
        self.added_at.elapsed() > self.ttl
    }

    fn update_score(&mut self) {
        let total = self.successful_retrievals + self.failed_retrievals;
        if total > 0 {
            self.score = self.successful_retrievals as f64 / total as f64;
        }
    }
}

/// Cache entry for content location
#[derive(Debug, Clone)]
struct CacheEntry {
    /// CID of the content
    #[allow(dead_code)]
    cid: Cid,
    /// List of providers
    providers: Vec<PeerId>,
    /// Timestamp when cached
    cached_at: Instant,
    /// Number of cache hits
    hits: u64,
}

impl CacheEntry {
    fn is_expired(&self, ttl: Duration) -> bool {
        self.cached_at.elapsed() > ttl
    }
}

/// Content routing statistics
#[derive(Debug, Default, Clone)]
pub struct ContentRoutingStats {
    /// Total DHT queries performed
    pub dht_queries: u64,
    /// Successful DHT queries
    pub dht_queries_success: u64,
    /// Failed DHT queries
    pub dht_queries_failed: u64,
    /// Cache hits
    pub cache_hits: u64,
    /// Cache misses
    pub cache_misses: u64,
    /// Total content advertisements
    pub content_advertisements: u64,
    /// Total provider records added
    pub provider_records_added: u64,
    /// Total provider records removed (expired)
    pub provider_records_removed: u64,
}

/// Content routing manager for DHT-based provider discovery
pub struct ContentRouter {
    /// Configuration
    config: ContentRoutingConfig,
    /// Provider records: CID -> Set of providers
    providers: Arc<DashMap<Cid, Vec<ProviderRecord>>>,
    /// Provider cache for fast lookups
    cache: Arc<RwLock<HashMap<Cid, CacheEntry>>>,
    /// Statistics
    stats: Arc<RwLock<ContentRoutingStats>>,
    /// Set of CIDs we're providing
    provided_content: Arc<DashMap<Cid, Instant>>,
}

impl ContentRouter {
    /// Create a new content router with default configuration
    pub fn new() -> Self {
        Self::with_config(ContentRoutingConfig::default())
    }

    /// Create a new content router with custom configuration
    pub fn with_config(config: ContentRoutingConfig) -> Self {
        Self {
            config,
            providers: Arc::new(DashMap::new()),
            cache: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(ContentRoutingStats::default())),
            provided_content: Arc::new(DashMap::new()),
        }
    }

    /// Advertise that we have content
    ///
    /// This adds ourselves as a provider for the given CID in the DHT
    pub async fn advertise_content(&self, cid: Cid, peer_id: PeerId) {
        // Record that we're providing this content
        self.provided_content.insert(cid, Instant::now());

        // Add ourselves as a provider
        self.add_provider(cid, peer_id, self.config.provider_record_ttl)
            .await;

        // Update stats
        let mut stats = self.stats.write().await;
        stats.content_advertisements += 1;
    }

    /// Add a provider for a CID
    async fn add_provider(&self, cid: Cid, peer_id: PeerId, ttl: Duration) {
        let record = ProviderRecord::new(peer_id, ttl);

        // Update or insert provider record
        self.providers
            .entry(cid)
            .and_modify(|providers| {
                // Remove expired providers
                providers.retain(|p| !p.is_expired());

                // Update existing provider or add new one
                if let Some(existing) = providers.iter_mut().find(|p| p.peer_id == peer_id) {
                    existing.added_at = Instant::now();
                    existing.ttl = ttl;
                } else if providers.len() < self.config.max_providers_per_cid {
                    providers.push(record.clone());
                } else {
                    // Replace lowest-scored provider
                    if let Some(worst) = providers.iter_mut().min_by(|a, b| {
                        a.score
                            .partial_cmp(&b.score)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    }) {
                        *worst = record.clone();
                    }
                }
            })
            .or_insert_with(|| vec![record]);

        // Invalidate cache for this CID
        self.cache.write().await.remove(&cid);

        // Update stats
        let mut stats = self.stats.write().await;
        stats.provider_records_added += 1;
    }

    /// Find providers for a CID
    ///
    /// Returns a list of peer IDs that can provide the content, ordered by score
    pub async fn find_providers(&self, cid: &Cid) -> Vec<PeerId> {
        // Check cache first
        if self.config.enable_cache_aware_routing {
            let mut cache = self.cache.write().await;
            if let Some(entry) = cache.get_mut(cid) {
                if !entry.is_expired(self.config.cache_entry_ttl) {
                    entry.hits += 1;
                    let mut stats = self.stats.write().await;
                    stats.cache_hits += 1;
                    return entry.providers.clone();
                } else {
                    cache.remove(cid);
                }
            }

            let mut stats = self.stats.write().await;
            stats.cache_misses += 1;
        }

        // Query DHT for providers
        let providers = self.query_dht_providers(cid).await;

        // Update cache
        if self.config.enable_cache_aware_routing && !providers.is_empty() {
            let mut cache = self.cache.write().await;
            if cache.len() >= self.config.max_cache_size {
                // Evict least recently used entry
                if let Some(lru_cid) = cache
                    .iter()
                    .min_by_key(|(_, entry)| entry.hits)
                    .map(|(cid, _)| *cid)
                {
                    cache.remove(&lru_cid);
                }
            }

            cache.insert(
                *cid,
                CacheEntry {
                    cid: *cid,
                    providers: providers.clone(),
                    cached_at: Instant::now(),
                    hits: 1,
                },
            );
        }

        providers
    }

    /// Query DHT for providers (simulated)
    async fn query_dht_providers(&self, cid: &Cid) -> Vec<PeerId> {
        let mut stats = self.stats.write().await;
        stats.dht_queries += 1;

        // Get providers from local store
        if let Some(providers) = self.providers.get(cid) {
            // Remove expired providers
            let valid_providers: Vec<_> = providers.iter().filter(|p| !p.is_expired()).collect();

            if !valid_providers.is_empty() {
                stats.dht_queries_success += 1;

                // Sort by score (highest first)
                let mut sorted_providers = valid_providers.clone();
                sorted_providers.sort_by(|a, b| {
                    b.score
                        .partial_cmp(&a.score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

                return sorted_providers.iter().map(|p| p.peer_id).collect();
            }
        }

        stats.dht_queries_failed += 1;
        Vec::new()
    }

    /// Record successful retrieval from a provider
    pub async fn record_success(&self, cid: &Cid, peer_id: &PeerId) {
        if let Some(mut providers) = self.providers.get_mut(cid) {
            if let Some(provider) = providers.iter_mut().find(|p| p.peer_id == *peer_id) {
                provider.successful_retrievals += 1;
                provider.update_score();
            }
        }
    }

    /// Record failed retrieval from a provider
    pub async fn record_failure(&self, cid: &Cid, peer_id: &PeerId) {
        if let Some(mut providers) = self.providers.get_mut(cid) {
            if let Some(provider) = providers.iter_mut().find(|p| p.peer_id == *peer_id) {
                provider.failed_retrievals += 1;
                provider.update_score();
            }
        }
    }

    /// Clean up expired provider records
    pub async fn cleanup_expired(&self) {
        let mut removed_count = 0;

        // Clean up provider records
        let cids_to_remove: Vec<Cid> = self
            .providers
            .iter()
            .filter_map(|entry| {
                let cid = *entry.key();
                let mut providers = entry.value().clone();
                providers.retain(|p| !p.is_expired());

                if providers.is_empty() {
                    Some(cid)
                } else {
                    removed_count += entry.value().len() - providers.len();
                    None
                }
            })
            .collect();

        for cid in cids_to_remove {
            removed_count += self.providers.get(&cid).map(|p| p.len()).unwrap_or(0);
            self.providers.remove(&cid);
        }

        // Clean up cache
        if self.config.enable_cache_aware_routing {
            let mut cache = self.cache.write().await;
            cache.retain(|_, entry| !entry.is_expired(self.config.cache_entry_ttl));
        }

        // Update stats
        let mut stats = self.stats.write().await;
        stats.provider_records_removed += removed_count as u64;
    }

    /// Get statistics
    pub async fn stats(&self) -> ContentRoutingStats {
        self.stats.read().await.clone()
    }

    /// Get cache-aware route recommendation
    ///
    /// Returns the best peer to retrieve content from based on cache locality
    pub async fn get_cache_aware_route(
        &self,
        cid: &Cid,
        _local_cache: &HashSet<Cid>,
    ) -> Option<PeerId> {
        let providers = self.find_providers(cid).await;

        if providers.is_empty() {
            return None;
        }

        // Prefer providers that have content we also need (cache locality)
        // This is a simplified heuristic - in practice, we'd use more sophisticated metrics

        // For now, just return the highest-scored provider
        providers.first().copied()
    }

    /// Get number of providers for a CID
    pub fn provider_count(&self, cid: &Cid) -> usize {
        self.providers
            .get(cid)
            .map(|providers| providers.iter().filter(|p| !p.is_expired()).count())
            .unwrap_or(0)
    }

    /// List all content we're providing
    pub fn list_provided_content(&self) -> Vec<Cid> {
        self.provided_content
            .iter()
            .map(|entry| *entry.key())
            .collect()
    }
}

impl Default for ContentRouter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_cid(data: &[u8]) -> Cid {
        use ipfrs_core::CidBuilder;
        CidBuilder::new()
            .build(data)
            .expect("test: build CID from data")
    }

    #[tokio::test]
    async fn test_advertise_and_find_content() {
        let router = ContentRouter::new();
        let cid = create_test_cid(b"test content");
        let peer_id = PeerId::random();

        // Advertise content
        router.advertise_content(cid, peer_id).await;

        // Find providers
        let providers = router.find_providers(&cid).await;
        assert_eq!(providers.len(), 1);
        assert_eq!(providers[0], peer_id);
    }

    #[tokio::test]
    async fn test_multiple_providers() {
        let router = ContentRouter::new();
        let cid = create_test_cid(b"shared content");

        let peer1 = PeerId::random();
        let peer2 = PeerId::random();
        let peer3 = PeerId::random();

        // Multiple peers advertise the same content
        router.advertise_content(cid, peer1).await;
        router.advertise_content(cid, peer2).await;
        router.advertise_content(cid, peer3).await;

        let providers = router.find_providers(&cid).await;
        assert_eq!(providers.len(), 3);
        assert!(providers.contains(&peer1));
        assert!(providers.contains(&peer2));
        assert!(providers.contains(&peer3));
    }

    #[tokio::test]
    async fn test_provider_scoring() {
        let router = ContentRouter::new();
        let cid = create_test_cid(b"scored content");
        let peer = PeerId::random();

        router.advertise_content(cid, peer).await;

        // Record some successes
        router.record_success(&cid, &peer).await;
        router.record_success(&cid, &peer).await;
        router.record_success(&cid, &peer).await;

        // Record one failure
        router.record_failure(&cid, &peer).await;

        // Check provider score
        if let Some(providers) = router.providers.get(&cid) {
            let provider = providers
                .iter()
                .find(|p| p.peer_id == peer)
                .expect("test: find provider matching peer");
            assert_eq!(provider.successful_retrievals, 3);
            assert_eq!(provider.failed_retrievals, 1);
            assert!((provider.score - 0.75).abs() < 0.01); // 3/4 = 0.75
        };
    }

    #[tokio::test]
    async fn test_cache_functionality() {
        let config = ContentRoutingConfig {
            enable_cache_aware_routing: true,
            ..Default::default()
        };
        let router = ContentRouter::with_config(config);
        let cid = create_test_cid(b"cached content");
        let peer = PeerId::random();

        router.advertise_content(cid, peer).await;

        // First query - cache miss
        let providers1 = router.find_providers(&cid).await;
        let stats1 = router.stats().await;
        assert_eq!(stats1.cache_misses, 1);
        assert_eq!(stats1.cache_hits, 0);

        // Second query - cache hit
        let providers2 = router.find_providers(&cid).await;
        let stats2 = router.stats().await;
        assert_eq!(stats2.cache_misses, 1);
        assert_eq!(stats2.cache_hits, 1);

        assert_eq!(providers1, providers2);
    }

    #[tokio::test]
    async fn test_provider_expiration() {
        let config = ContentRoutingConfig {
            provider_record_ttl: Duration::from_millis(100),
            cache_entry_ttl: Duration::from_millis(50), // Short cache TTL
            enable_cache_aware_routing: false,          // Disable caching for this test
            ..Default::default()
        };
        let router = ContentRouter::with_config(config);
        let cid = create_test_cid(b"expiring content");
        let peer = PeerId::random();

        router.advertise_content(cid, peer).await;

        // Should find provider immediately
        let providers = router.find_providers(&cid).await;
        assert_eq!(providers.len(), 1);

        // Wait for expiration
        tokio::time::sleep(Duration::from_millis(150)).await;

        // Cleanup expired records
        router.cleanup_expired().await;

        // Should not find provider after expiration
        let providers = router.find_providers(&cid).await;
        assert_eq!(providers.len(), 0);
    }

    #[tokio::test]
    async fn test_max_providers_limit() {
        let config = ContentRoutingConfig {
            max_providers_per_cid: 3,
            ..Default::default()
        };
        let router = ContentRouter::with_config(config);
        let cid = create_test_cid(b"popular content");

        // Add 5 providers (exceeds limit of 3)
        for _ in 0..5 {
            let peer = PeerId::random();
            router.advertise_content(cid, peer).await;
        }

        let providers = router.find_providers(&cid).await;
        assert_eq!(providers.len(), 3);
    }

    #[tokio::test]
    async fn test_stats_tracking() {
        let router = ContentRouter::new();
        let cid = create_test_cid(b"stats content");
        let peer = PeerId::random();

        router.advertise_content(cid, peer).await;
        router.find_providers(&cid).await;

        let stats = router.stats().await;
        assert_eq!(stats.content_advertisements, 1);
        assert_eq!(stats.provider_records_added, 1);
        assert!(stats.dht_queries > 0);
    }

    #[tokio::test]
    async fn test_list_provided_content() {
        let router = ContentRouter::new();
        let peer = PeerId::random();

        let cid1 = create_test_cid(b"content 1");
        let cid2 = create_test_cid(b"content 2");

        router.advertise_content(cid1, peer).await;
        router.advertise_content(cid2, peer).await;

        let provided = router.list_provided_content();
        assert_eq!(provided.len(), 2);
        assert!(provided.contains(&cid1));
        assert!(provided.contains(&cid2));
    }

    #[tokio::test]
    async fn test_provider_count() {
        let router = ContentRouter::new();
        let cid = create_test_cid(b"counted content");

        assert_eq!(router.provider_count(&cid), 0);

        let peer1 = PeerId::random();
        router.advertise_content(cid, peer1).await;
        assert_eq!(router.provider_count(&cid), 1);

        let peer2 = PeerId::random();
        router.advertise_content(cid, peer2).await;
        assert_eq!(router.provider_count(&cid), 2);
    }
}
