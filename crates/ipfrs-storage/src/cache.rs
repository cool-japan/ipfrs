//! In-memory block cache

use crate::traits::BlockStore;
use async_trait::async_trait;
use ipfrs_core::{Block, Cid, Result};
use lru::LruCache;
use parking_lot::Mutex;
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Cache statistics
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Number of cache hits
    pub hits: u64,
    /// Number of cache misses
    pub misses: u64,
    /// Current number of items in cache
    pub size: usize,
    /// Cache capacity
    pub capacity: usize,
}

impl CacheStats {
    /// Calculate hit rate (0.0 to 1.0)
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }

    /// Calculate miss rate (0.0 to 1.0)
    pub fn miss_rate(&self) -> f64 {
        1.0 - self.hit_rate()
    }
}

/// In-memory LRU cache for blocks
pub struct BlockCache {
    cache: Arc<Mutex<LruCache<Cid, Block>>>,
    capacity: usize,
    hits: Arc<AtomicU64>,
    misses: Arc<AtomicU64>,
}

impl BlockCache {
    /// Create a new LRU cache with the given capacity (number of blocks)
    pub fn new(capacity: usize) -> Self {
        let cap_val = capacity;
        let capacity = NonZeroUsize::new(capacity).unwrap_or(NonZeroUsize::new(1000).unwrap());
        Self {
            cache: Arc::new(Mutex::new(LruCache::new(capacity))),
            capacity: cap_val,
            hits: Arc::new(AtomicU64::new(0)),
            misses: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Get a block from cache
    #[inline]
    pub fn get(&self, cid: &Cid) -> Option<Block> {
        let result = self.cache.lock().get(cid).cloned();
        if result.is_some() {
            self.hits.fetch_add(1, Ordering::Relaxed);
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
        }
        result
    }

    /// Put a block into cache
    #[inline]
    pub fn put(&self, block: Block) {
        self.cache.lock().put(*block.cid(), block);
    }

    /// Remove a block from cache
    pub fn remove(&self, cid: &Cid) {
        self.cache.lock().pop(cid);
    }

    /// Clear the cache
    pub fn clear(&self) {
        self.cache.lock().clear();
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            size: self.cache.lock().len(),
            capacity: self.capacity,
        }
    }

    /// Get cache statistics (for backward compatibility)
    pub fn len(&self) -> usize {
        self.cache.lock().len()
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.cache.lock().is_empty()
    }
}

/// Caching wrapper around a block store
pub struct CachedBlockStore<S: BlockStore> {
    store: S,
    cache: BlockCache,
}

impl<S: BlockStore> CachedBlockStore<S> {
    /// Create a new caching block store
    pub fn new(store: S, cache_capacity: usize) -> Self {
        Self {
            store,
            cache: BlockCache::new(cache_capacity),
        }
    }

    /// Get reference to the underlying store
    pub fn store(&self) -> &S {
        &self.store
    }

    /// Get reference to the cache
    pub fn cache(&self) -> &BlockCache {
        &self.cache
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> CacheStats {
        self.cache.stats()
    }
}

#[async_trait]
impl<S: BlockStore> BlockStore for CachedBlockStore<S> {
    async fn put(&self, block: &Block) -> Result<()> {
        // Write-through: update both cache and store
        self.cache.put(block.clone());
        self.store.put(block).await
    }

    async fn get(&self, cid: &Cid) -> Result<Option<Block>> {
        // Check cache first
        if let Some(block) = self.cache.get(cid) {
            return Ok(Some(block));
        }

        // Cache miss: fetch from store
        if let Some(block) = self.store.get(cid).await? {
            self.cache.put(block.clone());
            Ok(Some(block))
        } else {
            Ok(None)
        }
    }

    async fn has(&self, cid: &Cid) -> Result<bool> {
        // Check cache first
        if self.cache.get(cid).is_some() {
            return Ok(true);
        }
        self.store.has(cid).await
    }

    async fn delete(&self, cid: &Cid) -> Result<()> {
        self.cache.remove(cid);
        self.store.delete(cid).await
    }

    fn list_cids(&self) -> Result<Vec<Cid>> {
        self.store.list_cids()
    }

    fn len(&self) -> usize {
        self.store.len()
    }

    fn is_empty(&self) -> bool {
        self.store.is_empty()
    }

    async fn flush(&self) -> Result<()> {
        self.store.flush().await
    }

    async fn close(&self) -> Result<()> {
        self.cache.clear();
        self.store.close().await
    }

    // Optimized batch operations to reduce lock contention
    async fn get_many(&self, cids: &[Cid]) -> Result<Vec<Option<Block>>> {
        let mut results = Vec::with_capacity(cids.len());
        let mut cache_misses = Vec::new();
        let mut miss_indices = Vec::new();

        // Single lock acquisition for all cache lookups
        {
            let cache = self.cache.cache.lock();
            for (i, cid) in cids.iter().enumerate() {
                if let Some(block) = cache.peek(cid) {
                    results.push(Some(block.clone()));
                } else {
                    results.push(None);
                    cache_misses.push(*cid);
                    miss_indices.push(i);
                }
            }
        }

        // Fetch cache misses from store
        if !cache_misses.is_empty() {
            let fetched = self.store.get_many(&cache_misses).await?;

            // Update cache and results in a single lock acquisition
            {
                let mut cache = self.cache.cache.lock();
                for (idx, block_opt) in miss_indices.iter().zip(fetched.iter()) {
                    if let Some(block) = block_opt {
                        cache.put(*block.cid(), block.clone());
                        results[*idx] = Some(block.clone());
                    }
                }
            }
        }

        Ok(results)
    }

    async fn put_many(&self, blocks: &[Block]) -> Result<()> {
        // Single lock acquisition for all cache updates
        {
            let mut cache = self.cache.cache.lock();
            for block in blocks {
                cache.put(*block.cid(), block.clone());
            }
        }

        // Write to underlying store
        self.store.put_many(blocks).await
    }

    async fn has_many(&self, cids: &[Cid]) -> Result<Vec<bool>> {
        let mut results = Vec::with_capacity(cids.len());
        let mut cache_misses = Vec::new();
        let mut miss_indices = Vec::new();

        // Single lock acquisition for all cache checks
        {
            let cache = self.cache.cache.lock();
            for (i, cid) in cids.iter().enumerate() {
                if cache.contains(cid) {
                    results.push(true);
                } else {
                    results.push(false);
                    cache_misses.push(*cid);
                    miss_indices.push(i);
                }
            }
        }

        // Check cache misses in store
        if !cache_misses.is_empty() {
            let store_results = self.store.has_many(&cache_misses).await?;
            for (idx, &exists) in miss_indices.iter().zip(store_results.iter()) {
                results[*idx] = exists;
            }
        }

        Ok(results)
    }

    async fn delete_many(&self, cids: &[Cid]) -> Result<()> {
        // Single lock acquisition for all cache deletions
        {
            let mut cache = self.cache.cache.lock();
            for cid in cids {
                cache.pop(cid);
            }
        }

        self.store.delete_many(cids).await
    }
}

/// Multi-level cache with hot (L1) and warm (L2) tiers
///
/// L1 is smaller and faster, L2 is larger but may have more contention
pub struct TieredBlockCache {
    /// L1 cache - hot blocks (small, fast)
    l1_cache: Arc<Mutex<LruCache<Cid, Block>>>,
    /// L2 cache - warm blocks (larger, slower)
    l2_cache: Arc<Mutex<LruCache<Cid, Block>>>,
    /// L1 capacity
    l1_capacity: usize,
    /// L2 capacity
    l2_capacity: usize,
    /// L1 hits
    l1_hits: Arc<AtomicU64>,
    /// L2 hits
    l2_hits: Arc<AtomicU64>,
    /// Total misses
    misses: Arc<AtomicU64>,
}

impl TieredBlockCache {
    /// Create a new tiered cache
    ///
    /// # Arguments
    /// * `l1_capacity` - Capacity of L1 (hot) cache in number of blocks
    /// * `l2_capacity` - Capacity of L2 (warm) cache in number of blocks
    pub fn new(l1_capacity: usize, l2_capacity: usize) -> Self {
        let l1_cap = NonZeroUsize::new(l1_capacity).unwrap_or(NonZeroUsize::new(100).unwrap());
        let l2_cap = NonZeroUsize::new(l2_capacity).unwrap_or(NonZeroUsize::new(1000).unwrap());

        Self {
            l1_cache: Arc::new(Mutex::new(LruCache::new(l1_cap))),
            l2_cache: Arc::new(Mutex::new(LruCache::new(l2_cap))),
            l1_capacity,
            l2_capacity,
            l1_hits: Arc::new(AtomicU64::new(0)),
            l2_hits: Arc::new(AtomicU64::new(0)),
            misses: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Get a block from cache (checks L1 first, then L2)
    #[inline]
    pub fn get(&self, cid: &Cid) -> Option<Block> {
        // Try L1 first
        if let Some(block) = self.l1_cache.lock().get(cid) {
            self.l1_hits.fetch_add(1, Ordering::Relaxed);
            return Some(block.clone());
        }

        // Try L2
        if let Some(block) = self.l2_cache.lock().get(cid) {
            self.l2_hits.fetch_add(1, Ordering::Relaxed);
            let block_clone = block.clone();
            // Promote to L1 on hit
            self.l1_cache.lock().put(*cid, block_clone.clone());
            return Some(block_clone);
        }

        self.misses.fetch_add(1, Ordering::Relaxed);
        None
    }

    /// Put a block into cache (goes to L1)
    #[inline]
    pub fn put(&self, block: Block) {
        let cid = *block.cid();

        // If block is being evicted from L1, move it to L2
        if let Some(evicted) = self.l1_cache.lock().push(cid, block.clone()) {
            // evicted is (Cid, Block)
            self.l2_cache.lock().put(evicted.0, evicted.1);
        }
    }

    /// Remove a block from cache
    pub fn remove(&self, cid: &Cid) {
        self.l1_cache.lock().pop(cid);
        self.l2_cache.lock().pop(cid);
    }

    /// Clear all caches
    pub fn clear(&self) {
        self.l1_cache.lock().clear();
        self.l2_cache.lock().clear();
        self.l1_hits.store(0, Ordering::Relaxed);
        self.l2_hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
    }

    /// Get cache statistics
    pub fn stats(&self) -> TieredCacheStats {
        TieredCacheStats {
            l1_size: self.l1_cache.lock().len(),
            l1_capacity: self.l1_capacity,
            l2_size: self.l2_cache.lock().len(),
            l2_capacity: self.l2_capacity,
            l1_hits: self.l1_hits.load(Ordering::Relaxed),
            l2_hits: self.l2_hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
        }
    }
}

/// Statistics for tiered cache
#[derive(Debug, Clone)]
pub struct TieredCacheStats {
    /// Current L1 cache size
    pub l1_size: usize,
    /// L1 cache capacity
    pub l1_capacity: usize,
    /// Current L2 cache size
    pub l2_size: usize,
    /// L2 cache capacity
    pub l2_capacity: usize,
    /// L1 cache hits
    pub l1_hits: u64,
    /// L2 cache hits
    pub l2_hits: u64,
    /// Total misses
    pub misses: u64,
}

impl TieredCacheStats {
    /// Calculate overall hit rate (0.0 to 1.0)
    pub fn hit_rate(&self) -> f64 {
        let total_hits = self.l1_hits + self.l2_hits;
        let total = total_hits + self.misses;
        if total == 0 {
            0.0
        } else {
            total_hits as f64 / total as f64
        }
    }

    /// Calculate L1 hit rate (0.0 to 1.0)
    pub fn l1_hit_rate(&self) -> f64 {
        let total = self.l1_hits + self.l2_hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.l1_hits as f64 / total as f64
        }
    }

    /// Calculate L2 hit rate (0.0 to 1.0)
    pub fn l2_hit_rate(&self) -> f64 {
        let total = self.l1_hits + self.l2_hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.l2_hits as f64 / total as f64
        }
    }

    /// Calculate miss rate (0.0 to 1.0)
    pub fn miss_rate(&self) -> f64 {
        1.0 - self.hit_rate()
    }
}

/// Tiered caching wrapper around a block store
pub struct TieredCachedBlockStore<S: BlockStore> {
    store: S,
    cache: TieredBlockCache,
}

impl<S: BlockStore> TieredCachedBlockStore<S> {
    /// Create a new tiered caching block store
    ///
    /// # Arguments
    /// * `store` - Underlying block store
    /// * `l1_capacity` - L1 cache capacity (number of blocks)
    /// * `l2_capacity` - L2 cache capacity (number of blocks)
    pub fn new(store: S, l1_capacity: usize, l2_capacity: usize) -> Self {
        Self {
            store,
            cache: TieredBlockCache::new(l1_capacity, l2_capacity),
        }
    }

    /// Get reference to the underlying store
    pub fn store(&self) -> &S {
        &self.store
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> TieredCacheStats {
        self.cache.stats()
    }
}

#[async_trait]
impl<S: BlockStore> BlockStore for TieredCachedBlockStore<S> {
    async fn put(&self, block: &Block) -> Result<()> {
        // Write-through: update cache and store
        self.cache.put(block.clone());
        self.store.put(block).await
    }

    async fn get(&self, cid: &Cid) -> Result<Option<Block>> {
        // Check cache first
        if let Some(block) = self.cache.get(cid) {
            return Ok(Some(block));
        }

        // Cache miss: fetch from store
        if let Some(block) = self.store.get(cid).await? {
            self.cache.put(block.clone());
            Ok(Some(block))
        } else {
            Ok(None)
        }
    }

    async fn has(&self, cid: &Cid) -> Result<bool> {
        // Check cache first
        if self.cache.get(cid).is_some() {
            return Ok(true);
        }
        self.store.has(cid).await
    }

    async fn delete(&self, cid: &Cid) -> Result<()> {
        self.cache.remove(cid);
        self.store.delete(cid).await
    }

    fn list_cids(&self) -> Result<Vec<Cid>> {
        self.store.list_cids()
    }

    fn len(&self) -> usize {
        self.store.len()
    }

    fn is_empty(&self) -> bool {
        self.store.is_empty()
    }

    async fn flush(&self) -> Result<()> {
        self.store.flush().await
    }

    async fn close(&self) -> Result<()> {
        self.cache.clear();
        self.store.close().await
    }
}
