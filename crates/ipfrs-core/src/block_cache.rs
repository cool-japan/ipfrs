//! LRU cache for blocks
//!
//! This module provides an efficient LRU (Least Recently Used) cache for blocks,
//! enabling fast repeated access to frequently used blocks without hitting storage.
//!
//! # Features
//!
//! - **Thread-safe** - Can be safely shared across threads
//! - **LRU eviction** - Automatically evicts least recently used blocks when full
//! - **Size limits** - Configurable maximum cache size (in bytes and/or block count)
//! - **Statistics tracking** - Monitor cache hits, misses, and evictions
//! - **Zero-copy** - Blocks are reference-counted, so cloning is cheap
//!
//! # Example
//!
//! ```rust
//! use ipfrs_core::{BlockCache, Block};
//! use bytes::Bytes;
//!
//! // Create a cache with 10MB limit
//! let cache = BlockCache::new(10 * 1024 * 1024, None);
//!
//! // Insert a block
//! let block = Block::new(Bytes::from_static(b"Hello, cache!")).unwrap();
//! cache.insert(block.clone());
//!
//! // Retrieve the block
//! if let Some(cached_block) = cache.get(block.cid()) {
//!     println!("Cache hit!");
//! }
//!
//! // Check statistics
//! let stats = cache.stats();
//! println!("Hits: {}, Misses: {}", stats.hits, stats.misses);
//! ```

use crate::block::Block;
use crate::cid::Cid;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// LRU cache for blocks
///
/// This cache uses a Least Recently Used eviction policy to maintain a bounded
/// set of frequently accessed blocks in memory. It's thread-safe and can be
/// shared across multiple threads.
///
/// # Example
///
/// ```rust
/// use ipfrs_core::{BlockCache, Block};
/// use bytes::Bytes;
///
/// let cache = BlockCache::new(1024 * 1024, Some(100)); // 1MB or 100 blocks max
///
/// let block = Block::new(Bytes::from_static(b"cached data")).unwrap();
/// cache.insert(block.clone());
///
/// assert!(cache.get(block.cid()).is_some());
/// ```
#[derive(Clone)]
pub struct BlockCache {
    inner: Arc<RwLock<BlockCacheInner>>,
}

struct BlockCacheInner {
    blocks: HashMap<Cid, CacheEntry>,
    lru_list: Vec<Cid>,
    max_size_bytes: u64,
    max_blocks: Option<usize>,
    current_size: u64,
    stats: CacheStats,
}

struct CacheEntry {
    block: Block,
    size: u64,
    last_access_index: usize,
}

impl BlockCache {
    /// Create a new block cache
    ///
    /// # Arguments
    ///
    /// * `max_size_bytes` - Maximum total size of cached blocks in bytes
    /// * `max_blocks` - Optional maximum number of blocks (None = unlimited)
    ///
    /// # Example
    ///
    /// ```rust
    /// use ipfrs_core::BlockCache;
    ///
    /// // Cache up to 10MB of blocks
    /// let cache = BlockCache::new(10 * 1024 * 1024, None);
    ///
    /// // Cache up to 1MB or 100 blocks, whichever limit is hit first
    /// let cache2 = BlockCache::new(1024 * 1024, Some(100));
    /// ```
    pub fn new(max_size_bytes: u64, max_blocks: Option<usize>) -> Self {
        Self {
            inner: Arc::new(RwLock::new(BlockCacheInner {
                blocks: HashMap::new(),
                lru_list: Vec::new(),
                max_size_bytes,
                max_blocks,
                current_size: 0,
                stats: CacheStats::default(),
            })),
        }
    }

    /// Insert a block into the cache
    ///
    /// If the cache is full, the least recently used block will be evicted.
    ///
    /// # Example
    ///
    /// ```rust
    /// use ipfrs_core::{BlockCache, Block};
    /// use bytes::Bytes;
    ///
    /// let cache = BlockCache::new(1024 * 1024, None);
    /// let block = Block::new(Bytes::from_static(b"data")).unwrap();
    ///
    /// cache.insert(block);
    /// ```
    pub fn insert(&self, block: Block) {
        let mut inner = self.inner.write().unwrap_or_else(|e| e.into_inner());
        let cid = *block.cid();
        let size = block.len() as u64;

        // If block already exists, update access time
        if inner.blocks.contains_key(&cid) {
            inner.update_access(&cid);
            return;
        }

        // Evict blocks if necessary
        while inner.would_exceed_limits(size) && !inner.blocks.is_empty() {
            inner.evict_lru();
        }

        // Insert the new block
        let access_index = inner.lru_list.len();
        inner.lru_list.push(cid);
        inner.blocks.insert(
            cid,
            CacheEntry {
                block,
                size,
                last_access_index: access_index,
            },
        );
        inner.current_size += size;
    }

    /// Get a block from the cache
    ///
    /// Returns `Some(block)` if found, `None` otherwise. Updates the access
    /// time for LRU tracking.
    ///
    /// # Example
    ///
    /// ```rust
    /// use ipfrs_core::{BlockCache, Block};
    /// use bytes::Bytes;
    ///
    /// let cache = BlockCache::new(1024 * 1024, None);
    /// let block = Block::new(Bytes::from_static(b"data")).unwrap();
    /// let cid = *block.cid();
    ///
    /// cache.insert(block);
    ///
    /// if let Some(cached) = cache.get(&cid) {
    ///     assert_eq!(cached.len(), 4);
    /// }
    /// ```
    pub fn get(&self, cid: &Cid) -> Option<Block> {
        let mut inner = self.inner.write().unwrap_or_else(|e| e.into_inner());

        if inner.blocks.contains_key(cid) {
            inner.stats.hits += 1;
            let block = inner
                .blocks
                .get(cid)
                .expect("just confirmed key is present via contains_key")
                .block
                .clone();
            inner.update_access(cid);
            Some(block)
        } else {
            inner.stats.misses += 1;
            None
        }
    }

    /// Check if the cache contains a block with the given CID
    ///
    /// This does not update LRU access time.
    pub fn contains(&self, cid: &Cid) -> bool {
        let inner = self.inner.read().unwrap_or_else(|e| e.into_inner());
        inner.blocks.contains_key(cid)
    }

    /// Remove a block from the cache
    pub fn remove(&self, cid: &Cid) -> Option<Block> {
        let mut inner = self.inner.write().unwrap_or_else(|e| e.into_inner());

        if let Some(entry) = inner.blocks.remove(cid) {
            inner.current_size -= entry.size;
            // Remove from LRU list
            if let Some(pos) = inner.lru_list.iter().position(|c| c == cid) {
                inner.lru_list.remove(pos);
            }
            Some(entry.block)
        } else {
            None
        }
    }

    /// Clear all blocks from the cache
    pub fn clear(&self) {
        let mut inner = self.inner.write().unwrap_or_else(|e| e.into_inner());
        inner.blocks.clear();
        inner.lru_list.clear();
        inner.current_size = 0;
    }

    /// Get the current cache statistics
    ///
    /// # Example
    ///
    /// ```rust
    /// use ipfrs_core::{BlockCache, Block};
    /// use bytes::Bytes;
    ///
    /// let cache = BlockCache::new(1024 * 1024, None);
    /// let block = Block::new(Bytes::from_static(b"data")).unwrap();
    ///
    /// cache.insert(block.clone());
    /// cache.get(block.cid()); // Hit
    /// cache.get(block.cid()); // Another hit
    ///
    /// let stats = cache.stats();
    /// assert_eq!(stats.hits, 2);
    /// assert_eq!(stats.misses, 0);
    /// ```
    pub fn stats(&self) -> CacheStats {
        let inner = self.inner.read().unwrap_or_else(|e| e.into_inner());
        inner.stats.clone()
    }

    /// Get the number of blocks currently in the cache
    pub fn len(&self) -> usize {
        let inner = self.inner.read().unwrap_or_else(|e| e.into_inner());
        inner.blocks.len()
    }

    /// Check if the cache is empty
    pub fn is_empty(&self) -> bool {
        let inner = self.inner.read().unwrap_or_else(|e| e.into_inner());
        inner.blocks.is_empty()
    }

    /// Get the current total size of cached blocks in bytes
    pub fn size(&self) -> u64 {
        let inner = self.inner.read().unwrap_or_else(|e| e.into_inner());
        inner.current_size
    }

    /// Get the maximum cache size in bytes
    pub fn max_size(&self) -> u64 {
        let inner = self.inner.read().unwrap_or_else(|e| e.into_inner());
        inner.max_size_bytes
    }

    /// Get the maximum number of blocks (if configured)
    pub fn max_blocks(&self) -> Option<usize> {
        let inner = self.inner.read().unwrap_or_else(|e| e.into_inner());
        inner.max_blocks
    }
}

impl BlockCacheInner {
    fn would_exceed_limits(&self, additional_size: u64) -> bool {
        let size_exceeded = self.current_size + additional_size > self.max_size_bytes;
        let count_exceeded = self
            .max_blocks
            .map(|max| self.blocks.len() >= max)
            .unwrap_or(false);

        size_exceeded || count_exceeded
    }

    fn evict_lru(&mut self) {
        if self.lru_list.is_empty() {
            return;
        }

        // Find the least recently used CID
        let lru_cid = self
            .blocks
            .iter()
            .min_by_key(|(_, entry)| entry.last_access_index)
            .map(|(cid, _)| *cid);

        if let Some(cid) = lru_cid {
            if let Some(entry) = self.blocks.remove(&cid) {
                self.current_size -= entry.size;
                self.stats.evictions += 1;

                // Remove from LRU list
                if let Some(pos) = self.lru_list.iter().position(|c| c == &cid) {
                    self.lru_list.remove(pos);
                }
            }
        }
    }

    fn update_access(&mut self, cid: &Cid) {
        if let Some(entry) = self.blocks.get_mut(cid) {
            entry.last_access_index = self.lru_list.len();
            self.lru_list.push(*cid);
        }
    }
}

/// Statistics for block cache operations
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Number of cache hits
    pub hits: u64,
    /// Number of cache misses
    pub misses: u64,
    /// Number of evictions (LRU removals)
    pub evictions: u64,
}

impl CacheStats {
    /// Calculate the hit rate (hits / total_requests)
    ///
    /// Returns 0.0 if no requests have been made.
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }

    /// Calculate the miss rate (misses / total_requests)
    ///
    /// Returns 0.0 if no requests have been made.
    pub fn miss_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.misses as f64 / total as f64
        }
    }

    /// Get the total number of requests (hits + misses)
    pub fn total_requests(&self) -> u64 {
        self.hits + self.misses
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;

    fn make_block(data: &[u8]) -> Block {
        Block::new(Bytes::copy_from_slice(data)).unwrap()
    }

    #[test]
    fn test_cache_basic_insert_get() {
        let cache = BlockCache::new(1024, None);
        let block = make_block(b"test data");
        let cid = *block.cid();

        cache.insert(block.clone());
        let retrieved = cache.get(&cid).unwrap();

        assert_eq!(retrieved.data(), block.data());
    }

    #[test]
    fn test_cache_miss() {
        let cache = BlockCache::new(1024, None);
        let block = make_block(b"test");
        let fake_cid = *make_block(b"other").cid();

        cache.insert(block);

        assert!(cache.get(&fake_cid).is_none());

        let stats = cache.stats();
        assert_eq!(stats.misses, 1);
    }

    #[test]
    fn test_cache_hit_tracking() {
        let cache = BlockCache::new(1024, None);
        let block = make_block(b"data");
        let cid = *block.cid();

        cache.insert(block);
        cache.get(&cid);
        cache.get(&cid);

        let stats = cache.stats();
        assert_eq!(stats.hits, 2);
    }

    #[test]
    fn test_cache_size_limit() {
        let cache = BlockCache::new(20, None); // Small cache
        let block1 = make_block(b"12345678901234567890"); // 20 bytes
        let block2 = make_block(b"extra"); // Will exceed limit

        cache.insert(block1.clone());
        cache.insert(block2.clone());

        // block1 should be evicted
        assert!(cache.get(block1.cid()).is_none());
        assert!(cache.get(block2.cid()).is_some());

        let stats = cache.stats();
        assert_eq!(stats.evictions, 1);
    }

    #[test]
    fn test_cache_count_limit() {
        let cache = BlockCache::new(1024, Some(2)); // Max 2 blocks
        let block1 = make_block(b"a");
        let block2 = make_block(b"b");
        let block3 = make_block(b"c");

        cache.insert(block1.clone());
        cache.insert(block2.clone());
        cache.insert(block3.clone());

        // block1 should be evicted (LRU)
        assert!(cache.get(block1.cid()).is_none());
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn test_cache_lru_eviction() {
        let cache = BlockCache::new(1024, Some(3));
        let block1 = make_block(b"1");
        let block2 = make_block(b"2");
        let block3 = make_block(b"3");
        let block4 = make_block(b"4");

        cache.insert(block1.clone());
        cache.insert(block2.clone());
        cache.insert(block3.clone());

        // Access block1 to make it more recently used
        cache.get(block1.cid());

        // Insert block4, should evict block2 (least recently used)
        cache.insert(block4.clone());

        assert!(cache.get(block1.cid()).is_some());
        assert!(cache.get(block2.cid()).is_none());
        assert!(cache.get(block3.cid()).is_some());
        assert!(cache.get(block4.cid()).is_some());
    }

    #[test]
    fn test_cache_contains() {
        let cache = BlockCache::new(1024, None);
        let block = make_block(b"test");

        cache.insert(block.clone());

        assert!(cache.contains(block.cid()));
        assert!(!cache.contains(make_block(b"other").cid()));
    }

    #[test]
    fn test_cache_remove() {
        let cache = BlockCache::new(1024, None);
        let block = make_block(b"test");
        let cid = *block.cid();

        cache.insert(block.clone());
        assert!(cache.contains(&cid));

        let removed = cache.remove(&cid);
        assert!(removed.is_some());
        assert!(!cache.contains(&cid));
    }

    #[test]
    fn test_cache_clear() {
        let cache = BlockCache::new(1024, None);
        cache.insert(make_block(b"1"));
        cache.insert(make_block(b"2"));
        cache.insert(make_block(b"3"));

        assert_eq!(cache.len(), 3);

        cache.clear();

        assert_eq!(cache.len(), 0);
        assert_eq!(cache.size(), 0);
    }

    #[test]
    fn test_cache_stats() {
        let cache = BlockCache::new(1024, None);
        let block = make_block(b"test");

        cache.insert(block.clone());
        cache.get(block.cid()); // hit
        cache.get(block.cid()); // hit
        cache.get(make_block(b"miss").cid()); // miss

        let stats = cache.stats();
        assert_eq!(stats.hits, 2);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.total_requests(), 3);
        assert!((stats.hit_rate() - 2.0 / 3.0).abs() < 0.001);
    }

    #[test]
    fn test_cache_size_tracking() {
        let cache = BlockCache::new(1024, None);
        let block1 = make_block(&[0u8; 100]);
        let block2 = make_block(&[0u8; 200]);

        cache.insert(block1.clone());
        assert_eq!(cache.size(), 100);

        cache.insert(block2.clone());
        assert_eq!(cache.size(), 300);

        cache.remove(block1.cid());
        assert_eq!(cache.size(), 200);
    }

    #[test]
    fn test_cache_duplicate_insert() {
        let cache = BlockCache::new(1024, None);
        let block = make_block(b"data");

        cache.insert(block.clone());
        cache.insert(block.clone()); // Duplicate

        assert_eq!(cache.len(), 1);
        assert_eq!(cache.size(), block.len() as u64);
    }

    #[test]
    fn test_cache_thread_safety() {
        use std::thread;

        let cache = BlockCache::new(10240, None);
        let cache_clone = cache.clone();

        let handle = thread::spawn(move || {
            for i in 0..100 {
                let block = make_block(&[i as u8; 10]);
                cache_clone.insert(block);
            }
        });

        for i in 100..200 {
            let block = make_block(&[i as u8; 10]);
            cache.insert(block);
        }

        handle.join().unwrap();

        // Should have blocks from both threads
        assert!(!cache.is_empty());
    }
}
