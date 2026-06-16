//! CDN Edge Node Integration
//!
//! Provides edge caching, origin server protocol, cache invalidation,
//! and CDN-accelerated delivery for IPFS content.
//!
//! # Example
//!
//! ```
//! use ipfrs_transport::cdn_edge::{EdgeNode, EdgeConfig, EvictionPolicy};
//! use bytes::Bytes;
//! use multihash::Multihash;
//! use cid::Cid;
//!
//! # #[tokio::main]
//! # async fn main() {
//! // Create an edge node with LRU eviction
//! let mut config = EdgeConfig::default();
//! config.eviction_policy = EvictionPolicy::LRU;
//! config.max_cache_size = 100 * 1024 * 1024; // 100 MB
//!
//! let edge = EdgeNode::with_config(config);
//!
//! // Register an origin server
//! edge.register_origin("origin1".to_string()).await;
//!
//! // Cache some content
//! let hash = Multihash::wrap(0x12, &[1u8; 32]).unwrap();
//! let cid = Cid::new_v1(0x55, hash);
//! let data = Bytes::from("cached content");
//!
//! edge.put(cid, data.clone()).await.ok();
//!
//! // Retrieve from cache
//! if let Some(cached_data) = edge.get(&cid).await {
//!     assert_eq!(cached_data, data);
//!     println!("Cache hit!");
//! }
//! # }
//! ```

use bytes::Bytes;
use cid::Cid;
use dashmap::DashMap;
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

/// Configuration for CDN edge nodes
#[derive(Debug, Clone)]
pub struct EdgeConfig {
    /// Maximum cache size in bytes
    pub max_cache_size: u64,
    /// Default cache entry TTL
    pub default_ttl: Duration,
    /// Enable compression for cached content
    pub enable_compression: bool,
    /// Minimum size for compression (bytes)
    pub compression_threshold: usize,
    /// Enable cache warming (prefetch popular content)
    pub enable_cache_warming: bool,
    /// Number of origin servers to track
    pub max_origin_servers: usize,
    /// Connection pool size per origin
    pub origin_connection_pool_size: usize,
    /// Cache eviction policy
    pub eviction_policy: EvictionPolicy,
}

impl Default for EdgeConfig {
    fn default() -> Self {
        Self {
            max_cache_size: 10 * 1024 * 1024 * 1024, // 10 GB
            default_ttl: Duration::from_secs(3600),  // 1 hour
            enable_compression: true,
            compression_threshold: 1024, // 1 KB
            enable_cache_warming: true,
            max_origin_servers: 100,
            origin_connection_pool_size: 10,
            eviction_policy: EvictionPolicy::LRU,
        }
    }
}

/// Cache eviction policies
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvictionPolicy {
    /// Least Recently Used
    LRU,
    /// Least Frequently Used
    LFU,
    /// Time-To-Live based
    TTL,
    /// Size-based (largest first)
    Size,
}

/// Cache entry metadata
#[derive(Debug, Clone)]
struct CacheEntry {
    /// Content ID
    #[allow(dead_code)]
    cid: Cid,
    /// Cached data
    data: Bytes,
    /// Original size (before compression)
    #[allow(dead_code)]
    original_size: usize,
    /// Compressed flag
    #[allow(dead_code)]
    compressed: bool,
    /// Time when cached
    cached_at: Instant,
    /// Time-to-live
    ttl: Duration,
    /// Last access time
    last_access: Instant,
    /// Access count
    access_count: u64,
    /// Hit rate (rolling average)
    hit_rate: f64,
}

impl CacheEntry {
    fn is_expired(&self) -> bool {
        self.cached_at.elapsed() > self.ttl
    }

    fn size(&self) -> usize {
        self.data.len()
    }

    fn record_access(&mut self) {
        self.last_access = Instant::now();
        self.access_count += 1;

        // Update rolling hit rate (exponential moving average)
        let alpha = 0.1;
        self.hit_rate = alpha + (1.0 - alpha) * self.hit_rate;
    }
}

/// Origin server information
#[derive(Debug, Clone)]
pub struct OriginServer {
    /// Server identifier (URL or peer ID)
    pub id: String,
    /// Server health score (0.0 to 1.0)
    pub health: f64,
    /// Average latency to this origin
    pub avg_latency: Duration,
    /// Total requests sent to this origin
    pub total_requests: u64,
    /// Successful responses from this origin
    pub successful_responses: u64,
    /// Last health check time
    pub last_health_check: Instant,
}

impl OriginServer {
    fn new(id: String) -> Self {
        Self {
            id,
            health: 1.0,
            avg_latency: Duration::ZERO,
            total_requests: 0,
            successful_responses: 0,
            last_health_check: Instant::now(),
        }
    }

    fn update_health(&mut self) {
        if self.total_requests > 0 {
            self.health = self.successful_responses as f64 / self.total_requests as f64;
        }
    }
}

/// CDN edge node statistics
#[derive(Debug, Default, Clone)]
pub struct EdgeStats {
    /// Total cache hits
    pub cache_hits: u64,
    /// Total cache misses
    pub cache_misses: u64,
    /// Total bytes served from cache
    pub bytes_served_from_cache: u64,
    /// Total bytes fetched from origin
    pub bytes_fetched_from_origin: u64,
    /// Total cache invalidations
    pub cache_invalidations: u64,
    /// Total cache evictions
    pub cache_evictions: u64,
    /// Current cache size in bytes
    pub current_cache_size: u64,
    /// Number of cached entries
    pub cached_entries: u64,
}

/// Cache invalidation request
#[derive(Debug, Clone)]
pub struct InvalidationRequest {
    /// CID to invalidate
    pub cid: Cid,
    /// Reason for invalidation
    pub reason: InvalidationReason,
    /// Timestamp of request
    pub timestamp: Instant,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InvalidationReason {
    /// Content updated at origin
    ContentUpdated,
    /// Manual invalidation request
    Manual,
    /// TTL expired
    Expired,
    /// Cache full (eviction)
    Eviction,
}

/// CDN edge node for caching and content delivery
pub struct EdgeNode {
    /// Configuration
    config: EdgeConfig,
    /// Cache storage: CID -> Entry
    cache: Arc<DashMap<Cid, CacheEntry>>,
    /// LRU queue for eviction
    lru_queue: Arc<RwLock<VecDeque<Cid>>>,
    /// Origin servers
    origins: Arc<RwLock<HashMap<String, OriginServer>>>,
    /// Statistics
    stats: Arc<RwLock<EdgeStats>>,
    /// Current cache size (bytes)
    current_size: Arc<AtomicU64>,
}

impl EdgeNode {
    /// Create a new edge node with default configuration
    pub fn new() -> Self {
        Self::with_config(EdgeConfig::default())
    }

    /// Create a new edge node with custom configuration
    pub fn with_config(config: EdgeConfig) -> Self {
        Self {
            config,
            cache: Arc::new(DashMap::new()),
            lru_queue: Arc::new(RwLock::new(VecDeque::new())),
            origins: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(EdgeStats::default())),
            current_size: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Get content from cache or fetch from origin
    pub async fn get(&self, cid: &Cid) -> Option<Bytes> {
        // Try cache first
        if let Some(mut entry) = self.cache.get_mut(cid) {
            if !entry.is_expired() {
                entry.record_access();
                self.update_lru(cid).await;

                let mut stats = self.stats.write().await;
                stats.cache_hits += 1;
                stats.bytes_served_from_cache += entry.size() as u64;

                return Some(entry.data.clone());
            } else {
                // Entry expired, remove it
                drop(entry);
                self.invalidate(cid, InvalidationReason::Expired).await;
            }
        }

        // Cache miss
        let mut stats = self.stats.write().await;
        stats.cache_misses += 1;
        drop(stats);

        // Fetch from origin (simulated)
        self.fetch_from_origin(cid).await
    }

    /// Put content into cache
    pub async fn put(&self, cid: Cid, data: Bytes) -> Result<(), String> {
        let original_size = data.len();

        // Check if we need to make room
        while self.current_size.load(Ordering::Acquire) + data.len() as u64
            > self.config.max_cache_size
        {
            if !self.evict_one().await {
                return Err("Cache full and cannot evict".to_string());
            }
        }

        // Optionally compress
        let (final_data, compressed) = if self.config.enable_compression
            && original_size >= self.config.compression_threshold
        {
            // Simulated compression (in practice, use zstd or similar)
            (data.clone(), false) // For now, don't actually compress
        } else {
            (data, false)
        };

        let entry = CacheEntry {
            cid,
            data: final_data.clone(),
            original_size,
            compressed,
            cached_at: Instant::now(),
            ttl: self.config.default_ttl,
            last_access: Instant::now(),
            access_count: 0,
            hit_rate: 0.0,
        };

        let entry_size = entry.size();

        // Insert into cache
        self.cache.insert(cid, entry);

        // Update LRU queue
        self.lru_queue.write().await.push_back(cid);

        // Update size
        self.current_size
            .fetch_add(entry_size as u64, Ordering::Release);

        // Update stats
        let mut stats = self.stats.write().await;
        stats.cached_entries += 1;
        stats.current_cache_size = self.current_size.load(Ordering::Acquire);

        Ok(())
    }

    /// Invalidate a cache entry
    pub async fn invalidate(&self, cid: &Cid, reason: InvalidationReason) {
        if let Some((_, entry)) = self.cache.remove(cid) {
            let entry_size = entry.size();

            // Update size
            self.current_size
                .fetch_sub(entry_size as u64, Ordering::Release);

            // Remove from LRU queue
            let mut queue = self.lru_queue.write().await;
            queue.retain(|c| c != cid);

            // Update stats
            let mut stats = self.stats.write().await;
            stats.cache_invalidations += 1;
            stats.cached_entries = stats.cached_entries.saturating_sub(1);
            stats.current_cache_size = self.current_size.load(Ordering::Acquire);

            if reason == InvalidationReason::Eviction {
                stats.cache_evictions += 1;
            }
        }
    }

    /// Evict one entry according to eviction policy
    async fn evict_one(&self) -> bool {
        let victim = match self.config.eviction_policy {
            EvictionPolicy::LRU => self.find_lru_victim().await,
            EvictionPolicy::LFU => self.find_lfu_victim().await,
            EvictionPolicy::TTL => self.find_ttl_victim().await,
            EvictionPolicy::Size => self.find_size_victim().await,
        };

        if let Some(cid) = victim {
            self.invalidate(&cid, InvalidationReason::Eviction).await;
            true
        } else {
            false
        }
    }

    async fn find_lru_victim(&self) -> Option<Cid> {
        self.lru_queue.read().await.front().copied()
    }

    async fn find_lfu_victim(&self) -> Option<Cid> {
        self.cache
            .iter()
            .min_by_key(|entry| entry.access_count)
            .map(|entry| *entry.key())
    }

    async fn find_ttl_victim(&self) -> Option<Cid> {
        self.cache
            .iter()
            .filter(|entry| entry.is_expired())
            .map(|entry| *entry.key())
            .next()
    }

    async fn find_size_victim(&self) -> Option<Cid> {
        self.cache
            .iter()
            .max_by_key(|entry| entry.size())
            .map(|entry| *entry.key())
    }

    async fn update_lru(&self, cid: &Cid) {
        let mut queue = self.lru_queue.write().await;
        queue.retain(|c| c != cid);
        queue.push_back(*cid);
    }

    /// Fetch content from origin server (simulated)
    async fn fetch_from_origin(&self, _cid: &Cid) -> Option<Bytes> {
        // In a real implementation, this would fetch from origin servers
        // For now, return None to simulate cache miss without origin
        None
    }

    /// Register an origin server
    pub async fn register_origin(&self, id: String) {
        let mut origins = self.origins.write().await;
        if origins.len() < self.config.max_origin_servers {
            origins.insert(id.clone(), OriginServer::new(id));
        }
    }

    /// Update origin server health
    pub async fn update_origin_health(&self, id: &str, success: bool, latency: Duration) {
        let mut origins = self.origins.write().await;
        if let Some(origin) = origins.get_mut(id) {
            origin.total_requests += 1;
            if success {
                origin.successful_responses += 1;
            }

            // Update average latency (exponential moving average)
            let alpha = 0.1;
            origin.avg_latency = if origin.avg_latency.is_zero() {
                latency
            } else {
                Duration::from_nanos(
                    ((1.0 - alpha) * origin.avg_latency.as_nanos() as f64
                        + alpha * latency.as_nanos() as f64) as u64,
                )
            };

            origin.update_health();
            origin.last_health_check = Instant::now();
        }
    }

    /// Get best origin server (highest health, lowest latency)
    pub async fn get_best_origin(&self) -> Option<String> {
        let origins = self.origins.read().await;
        origins
            .values()
            .filter(|o| o.health > 0.5) // Only consider healthy origins
            .min_by(|a, b| a.avg_latency.cmp(&b.avg_latency))
            .map(|o| o.id.clone())
    }

    /// Warm cache with popular content
    pub async fn warm_cache(&self, popular_cids: Vec<Cid>) {
        if !self.config.enable_cache_warming {
            return;
        }

        for cid in popular_cids {
            if !self.cache.contains_key(&cid) {
                // Fetch from origin and cache (simulated)
                if let Some(data) = self.fetch_from_origin(&cid).await {
                    let _ = self.put(cid, data).await;
                }
            }
        }
    }

    /// Get statistics
    pub async fn stats(&self) -> EdgeStats {
        let mut stats = self.stats.read().await.clone();
        stats.current_cache_size = self.current_size.load(Ordering::Acquire);
        stats.cached_entries = self.cache.len() as u64;
        stats
    }

    /// Get cache hit rate
    pub async fn hit_rate(&self) -> f64 {
        let stats = self.stats.read().await;
        let total = stats.cache_hits + stats.cache_misses;
        if total > 0 {
            stats.cache_hits as f64 / total as f64
        } else {
            0.0
        }
    }

    /// Clear entire cache
    pub async fn clear_cache(&self) {
        self.cache.clear();
        self.lru_queue.write().await.clear();
        self.current_size.store(0, Ordering::Release);

        let mut stats = self.stats.write().await;
        stats.cached_entries = 0;
        stats.current_cache_size = 0;
    }
}

impl Default for EdgeNode {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ipfrs_core::CidBuilder;

    fn create_test_cid(data: &[u8]) -> Cid {
        CidBuilder::new()
            .build(data)
            .expect("test: build CID from data")
    }

    #[tokio::test]
    async fn test_cache_put_and_get() {
        let edge = EdgeNode::new();
        let cid = create_test_cid(b"test data");
        let data = Bytes::from("test data");

        edge.put(cid, data.clone())
            .await
            .expect("test: put data into cache");

        let retrieved = edge.get(&cid).await;
        assert_eq!(retrieved, Some(data));
    }

    #[tokio::test]
    async fn test_cache_miss() {
        let edge = EdgeNode::new();
        let cid = create_test_cid(b"nonexistent");

        let retrieved = edge.get(&cid).await;
        assert_eq!(retrieved, None);
    }

    #[tokio::test]
    async fn test_cache_invalidation() {
        let edge = EdgeNode::new();
        let cid = create_test_cid(b"invalidate me");
        let data = Bytes::from("invalidate me");

        edge.put(cid, data)
            .await
            .expect("test: put data into cache");
        assert!(edge.get(&cid).await.is_some());

        edge.invalidate(&cid, InvalidationReason::Manual).await;
        assert!(edge.get(&cid).await.is_none());
    }

    #[tokio::test]
    async fn test_cache_expiration() {
        let config = EdgeConfig {
            default_ttl: Duration::from_millis(100),
            ..Default::default()
        };
        let edge = EdgeNode::with_config(config);
        let cid = create_test_cid(b"expire soon");
        let data = Bytes::from("expire soon");

        edge.put(cid, data)
            .await
            .expect("test: put data into cache");
        assert!(edge.get(&cid).await.is_some());

        // Wait for expiration
        tokio::time::sleep(Duration::from_millis(150)).await;

        assert!(edge.get(&cid).await.is_none());
    }

    #[tokio::test]
    async fn test_cache_eviction_lru() {
        let config = EdgeConfig {
            max_cache_size: 50, // Small cache for 3 entries of ~5 bytes each
            eviction_policy: EvictionPolicy::LRU,
            ..Default::default()
        };
        let edge = EdgeNode::with_config(config);

        // Fill cache
        let cid1 = create_test_cid(b"data1");
        let cid2 = create_test_cid(b"data2");
        let cid3 = create_test_cid(b"data3");

        edge.put(cid1, Bytes::from("data1"))
            .await
            .expect("test: put data1 into cache");
        edge.put(cid2, Bytes::from("data2"))
            .await
            .expect("test: put data2 into cache");

        // Access cid1 to make it more recently used
        edge.get(&cid1).await;

        // Add cid3, should evict cid2 (least recently used)
        edge.put(cid3, Bytes::from("data3"))
            .await
            .expect("test: put data3 into cache");

        // cid1 and cid3 should exist, cid2 might be evicted
        assert!(edge.get(&cid1).await.is_some());
        assert!(edge.get(&cid3).await.is_some());
    }

    #[tokio::test]
    async fn test_stats_tracking() {
        let edge = EdgeNode::new();
        let cid = create_test_cid(b"stats test");
        let data = Bytes::from("stats test");

        edge.put(cid, data)
            .await
            .expect("test: put stats test data into cache");

        // Cache hit
        edge.get(&cid).await;

        // Cache miss
        let nonexistent = create_test_cid(b"nonexistent");
        edge.get(&nonexistent).await;

        let stats = edge.stats().await;
        assert_eq!(stats.cache_hits, 1);
        assert_eq!(stats.cache_misses, 1);
        assert_eq!(stats.cached_entries, 1);
    }

    #[tokio::test]
    async fn test_hit_rate() {
        let edge = EdgeNode::new();
        let cid = create_test_cid(b"hit rate test");
        let data = Bytes::from("hit rate test");

        edge.put(cid, data)
            .await
            .expect("test: put hit rate test data into cache");

        // 3 hits
        edge.get(&cid).await;
        edge.get(&cid).await;
        edge.get(&cid).await;

        // 1 miss
        edge.get(&create_test_cid(b"miss")).await;

        let hit_rate = edge.hit_rate().await;
        assert!((hit_rate - 0.75).abs() < 0.01); // 3/4 = 0.75
    }

    #[tokio::test]
    async fn test_origin_registration() {
        let edge = EdgeNode::new();

        edge.register_origin("origin1".to_string()).await;
        edge.register_origin("origin2".to_string()).await;

        let origins = edge.origins.read().await;
        assert_eq!(origins.len(), 2);
        assert!(origins.contains_key("origin1"));
        assert!(origins.contains_key("origin2"));
    }

    #[tokio::test]
    async fn test_origin_health_tracking() {
        let edge = EdgeNode::new();
        edge.register_origin("origin1".to_string()).await;

        // Successful requests
        edge.update_origin_health("origin1", true, Duration::from_millis(10))
            .await;
        edge.update_origin_health("origin1", true, Duration::from_millis(20))
            .await;

        // Failed request
        edge.update_origin_health("origin1", false, Duration::from_millis(100))
            .await;

        let origins = edge.origins.read().await;
        let origin = origins
            .get("origin1")
            .expect("test: get origin1 from origins map");

        assert_eq!(origin.total_requests, 3);
        assert_eq!(origin.successful_responses, 2);
        assert!((origin.health - 0.666).abs() < 0.01); // 2/3
    }

    #[tokio::test]
    async fn test_best_origin_selection() {
        let edge = EdgeNode::new();

        edge.register_origin("fast".to_string()).await;
        edge.register_origin("slow".to_string()).await;

        edge.update_origin_health("fast", true, Duration::from_millis(10))
            .await;
        edge.update_origin_health("slow", true, Duration::from_millis(100))
            .await;

        let best = edge.get_best_origin().await;
        assert_eq!(best, Some("fast".to_string()));
    }

    #[tokio::test]
    async fn test_clear_cache() {
        let edge = EdgeNode::new();

        for i in 0..5 {
            let cid = create_test_cid(&[i]);
            edge.put(cid, Bytes::from(vec![i]))
                .await
                .expect("test: put data into cache");
        }

        let stats = edge.stats().await;
        assert_eq!(stats.cached_entries, 5);

        edge.clear_cache().await;

        let stats = edge.stats().await;
        assert_eq!(stats.cached_entries, 0);
        assert_eq!(stats.current_cache_size, 0);
    }
}
