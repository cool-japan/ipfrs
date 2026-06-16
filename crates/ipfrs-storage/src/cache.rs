//! In-memory block cache — L1/L2 hot cache above a backing BlockStore
//!
//! # Overview
//!
//! Two public cache abstractions are provided:
//!
//! * [`BlockCache`] — a standalone LRU cache (pure in-memory, no backing store).
//! * [`CachedBlockStore<S>`] — a two-level cache that wraps any `BlockStore` (L2)
//!   with an in-process LRU cache (L1).  Stats are tracked with lock-free atomics.
//!
//! A tiered variant ([`TieredCachedBlockStore`]) is also available for workloads
//! that benefit from separate hot/warm tiers.

use crate::traits::BlockStore;
use async_trait::async_trait;
use ipfrs_core::{Block, Cid, Result};
use lru::LruCache;
use parking_lot::Mutex;
use serde::Serialize;
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// CacheConfig
// ---------------------------------------------------------------------------

/// Configuration for [`CachedBlockStore`].
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Maximum number of blocks kept in the L1 LRU cache.
    ///
    /// Default: 1 024.
    pub l1_capacity: NonZeroUsize,

    /// Blocks larger than this byte threshold are stored in L2 (the backing
    /// store) but **not** promoted to L1, keeping the hot-path cache lean.
    ///
    /// Default: 256 KiB.
    pub max_block_bytes: usize,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            l1_capacity: NonZeroUsize::new(1024).expect("1024 > 0"),
            max_block_bytes: 256 * 1024,
        }
    }
}

// ---------------------------------------------------------------------------
// CacheStats — lock-free atomic counters
// ---------------------------------------------------------------------------

/// Live, lock-free counters for a [`CachedBlockStore`].
///
/// Each field is an [`AtomicU64`] so that observers can read a consistent
/// snapshot without acquiring any locks.  Ordering is `Relaxed` throughout
/// because counter values are only informational.
#[derive(Debug, Default, Serialize)]
pub struct CacheStats {
    /// Number of get() calls satisfied from L1.
    pub hits: AtomicU64,
    /// Number of get() calls that missed L1 (including blocks not in L2).
    pub misses: AtomicU64,
    /// Approximate number of LRU evictions from L1.
    pub evictions: AtomicU64,
    /// Number of put() calls (all, including oversized blocks).
    pub puts: AtomicU64,
}

impl CacheStats {
    /// Instantaneous hit-rate in [0.0, 1.0].
    ///
    /// Returns `0.0` when no get() calls have been made yet.
    pub fn hit_rate(&self) -> f64 {
        let h = self.hits.load(Ordering::Relaxed);
        let m = self.misses.load(Ordering::Relaxed);
        let total = h + m;
        if total == 0 {
            0.0
        } else {
            h as f64 / total as f64
        }
    }

    /// Take a point-in-time snapshot of all counters.
    pub fn snapshot(&self) -> CacheStatsSnapshot {
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        let evictions = self.evictions.load(Ordering::Relaxed);
        let puts = self.puts.load(Ordering::Relaxed);
        let total = hits + misses;
        let hit_rate = if total == 0 {
            0.0
        } else {
            hits as f64 / total as f64
        };
        CacheStatsSnapshot {
            hits,
            misses,
            evictions,
            puts,
            hit_rate,
        }
    }
}

// ---------------------------------------------------------------------------
// CacheStatsSnapshot — a cloneable, serialisable view of CacheStats
// ---------------------------------------------------------------------------

/// A serialisable, cloneable point-in-time view of [`CacheStats`].
#[derive(Clone, Debug, Serialize)]
pub struct CacheStatsSnapshot {
    /// Hits at the time of the snapshot.
    pub hits: u64,
    /// Misses at the time of the snapshot.
    pub misses: u64,
    /// LRU evictions at the time of the snapshot.
    pub evictions: u64,
    /// puts at the time of the snapshot.
    pub puts: u64,
    /// Derived hit-rate \[0.0, 1.0\] at the time of the snapshot.
    pub hit_rate: f64,
}

// ---------------------------------------------------------------------------
// CachedBlockStore — L1 LRU in front of a backing BlockStore (L2)
// ---------------------------------------------------------------------------

/// A two-level cache that wraps any [`BlockStore`] (L2) with an in-process
/// LRU cache (L1).
///
/// # Cache behaviour
///
/// | Operation  | L1 action                     | L2 action                          |
/// |------------|-------------------------------|------------------------------------|
/// | `put`      | insert (if ≤ max_block_bytes) | always written                     |
/// | `get` hit  | return from L1                | not consulted                      |
/// | `get` miss | populate on L2 hit            | queried; result promoted to L1     |
/// | `delete`   | evict                         | deleted                            |
/// | `has`      | cheap presence check          | consulted only on L1 miss          |
pub struct CachedBlockStore<S: BlockStore> {
    /// The backing store (L2).
    inner: S,
    /// L1 LRU keyed by the canonical CID string representation.
    cache: Mutex<LruCache<String, bytes::Bytes>>,
    /// Capacity threshold for L1 admission (bytes).
    max_block_bytes: usize,
    /// Live statistics.
    stats: CacheStats,
}

impl<S: BlockStore> CachedBlockStore<S> {
    /// Construct a new `CachedBlockStore` with explicit configuration.
    pub fn new(inner: S, config: CacheConfig) -> Self {
        Self {
            inner,
            cache: Mutex::new(LruCache::new(config.l1_capacity)),
            max_block_bytes: config.max_block_bytes,
            stats: CacheStats::default(),
        }
    }

    /// Construct a new `CachedBlockStore` using [`CacheConfig::default`].
    pub fn with_default_config(inner: S) -> Self {
        Self::new(inner, CacheConfig::default())
    }

    /// Return a point-in-time snapshot of cache statistics.
    pub fn stats(&self) -> CacheStatsSnapshot {
        self.stats.snapshot()
    }

    /// Current number of entries in L1.
    pub fn cache_size(&self) -> usize {
        self.cache.lock().len()
    }

    /// Remove a single CID from L1 (does **not** affect L2).
    pub fn invalidate(&self, cid: &Cid) {
        self.cache.lock().pop(&cid.to_string());
    }

    /// Evict all entries from L1 (does **not** affect L2).
    pub fn clear_cache(&self) {
        self.cache.lock().clear();
    }

    /// Borrow the underlying backing store.
    pub fn inner(&self) -> &S {
        &self.inner
    }

    // ------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------

    /// Insert `data` into L1 for `cid`, tracking evictions.
    fn l1_insert(&self, cid: &Cid, data: bytes::Bytes) {
        let key = cid.to_string();
        let mut guard = self.cache.lock();
        // `push` returns the evicted entry (if any).
        if guard.push(key, data).is_some() {
            self.stats.evictions.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Look up `cid` in L1 without counting a hit/miss — used internally.
    fn l1_peek(&self, cid: &Cid) -> Option<bytes::Bytes> {
        self.cache.lock().get(&cid.to_string()).cloned()
    }
}

// ---------------------------------------------------------------------------
// BlockStore impl
// ---------------------------------------------------------------------------

#[async_trait]
impl<S: BlockStore> BlockStore for CachedBlockStore<S> {
    /// Write to L2 first, then promote to L1 if the block is small enough.
    async fn put(&self, block: &Block) -> Result<()> {
        self.stats.puts.fetch_add(1, Ordering::Relaxed);
        self.inner.put(block).await?;
        if block.data().len() <= self.max_block_bytes {
            self.l1_insert(block.cid(), block.data().clone());
        }
        Ok(())
    }

    /// Check L1 first; on miss fetch from L2 and populate L1.
    async fn get(&self, cid: &Cid) -> Result<Option<Block>> {
        // L1 hit
        if let Some(data) = self.l1_peek(cid) {
            self.stats.hits.fetch_add(1, Ordering::Relaxed);
            return Ok(Some(Block::from_parts(*cid, data)));
        }

        // L1 miss — consult L2
        self.stats.misses.fetch_add(1, Ordering::Relaxed);
        match self.inner.get(cid).await? {
            Some(block) => {
                // Populate L1 if the block fits
                if block.data().len() <= self.max_block_bytes {
                    self.l1_insert(cid, block.data().clone());
                }
                Ok(Some(block))
            }
            None => Ok(None),
        }
    }

    /// `has()` checks L1 (cheap) then L2.
    async fn has(&self, cid: &Cid) -> Result<bool> {
        if self.cache.lock().contains(&cid.to_string()) {
            return Ok(true);
        }
        self.inner.has(cid).await
    }

    /// Remove from L1 and L2.
    async fn delete(&self, cid: &Cid) -> Result<()> {
        self.cache.lock().pop(&cid.to_string());
        self.inner.delete(cid).await
    }

    fn list_cids(&self) -> Result<Vec<Cid>> {
        self.inner.list_cids()
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    async fn flush(&self) -> Result<()> {
        self.inner.flush().await
    }

    async fn close(&self) -> Result<()> {
        self.clear_cache();
        self.inner.close().await
    }

    // ------------------------------------------------------------------
    // Optimised batch operations — minimise lock acquisitions
    // ------------------------------------------------------------------

    async fn get_many(&self, cids: &[Cid]) -> Result<Vec<Option<Block>>> {
        let mut results: Vec<Option<Block>> = Vec::with_capacity(cids.len());
        let mut miss_cids: Vec<Cid> = Vec::new();
        let mut miss_indices: Vec<usize> = Vec::new();

        // Single L1 lock acquisition for all lookups.
        {
            let mut guard = self.cache.lock();
            for (i, cid) in cids.iter().enumerate() {
                if let Some(data) = guard.get(&cid.to_string()).cloned() {
                    self.stats.hits.fetch_add(1, Ordering::Relaxed);
                    results.push(Some(Block::from_parts(*cid, data)));
                } else {
                    self.stats.misses.fetch_add(1, Ordering::Relaxed);
                    results.push(None);
                    miss_cids.push(*cid);
                    miss_indices.push(i);
                }
            }
        }

        if !miss_cids.is_empty() {
            let fetched = self.inner.get_many(&miss_cids).await?;
            let mut guard = self.cache.lock();
            for (idx, block_opt) in miss_indices.iter().zip(fetched.iter()) {
                if let Some(block) = block_opt {
                    if block.data().len() <= self.max_block_bytes
                        && guard
                            .push(block.cid().to_string(), block.data().clone())
                            .is_some()
                    {
                        self.stats.evictions.fetch_add(1, Ordering::Relaxed);
                    }
                    results[*idx] = Some(block.clone());
                }
            }
        }

        Ok(results)
    }

    async fn put_many(&self, blocks: &[Block]) -> Result<()> {
        self.stats
            .puts
            .fetch_add(blocks.len() as u64, Ordering::Relaxed);

        // Write to L2 first.
        self.inner.put_many(blocks).await?;

        // Populate L1 in a single lock acquisition.
        let mut guard = self.cache.lock();
        for block in blocks {
            if block.data().len() <= self.max_block_bytes
                && guard
                    .push(block.cid().to_string(), block.data().clone())
                    .is_some()
            {
                self.stats.evictions.fetch_add(1, Ordering::Relaxed);
            }
        }

        Ok(())
    }

    async fn has_many(&self, cids: &[Cid]) -> Result<Vec<bool>> {
        let mut results: Vec<bool> = Vec::with_capacity(cids.len());
        let mut miss_cids: Vec<Cid> = Vec::new();
        let mut miss_indices: Vec<usize> = Vec::new();

        {
            let guard = self.cache.lock();
            for (i, cid) in cids.iter().enumerate() {
                if guard.contains(&cid.to_string()) {
                    results.push(true);
                } else {
                    results.push(false);
                    miss_cids.push(*cid);
                    miss_indices.push(i);
                }
            }
        }

        if !miss_cids.is_empty() {
            let store_results = self.inner.has_many(&miss_cids).await?;
            for (idx, &exists) in miss_indices.iter().zip(store_results.iter()) {
                results[*idx] = exists;
            }
        }

        Ok(results)
    }

    async fn delete_many(&self, cids: &[Cid]) -> Result<()> {
        {
            let mut guard = self.cache.lock();
            for cid in cids {
                guard.pop(&cid.to_string());
            }
        }
        self.inner.delete_many(cids).await
    }
}

// ---------------------------------------------------------------------------
// BlockCache — standalone in-memory LRU (no backing store)
// ---------------------------------------------------------------------------

/// Standalone in-memory LRU cache for blocks.
///
/// This is a building block; most callers should prefer [`CachedBlockStore`].
pub struct BlockCache {
    cache: Arc<Mutex<LruCache<Cid, Block>>>,
    capacity: usize,
    hits: Arc<AtomicU64>,
    misses: Arc<AtomicU64>,
}

impl BlockCache {
    /// Create a new LRU cache with the given capacity (number of blocks).
    pub fn new(capacity: usize) -> Self {
        let cap =
            NonZeroUsize::new(capacity).unwrap_or_else(|| NonZeroUsize::new(1000).expect("1000>0"));
        Self {
            cache: Arc::new(Mutex::new(LruCache::new(cap))),
            capacity,
            hits: Arc::new(AtomicU64::new(0)),
            misses: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Get a block from cache.
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

    /// Put a block into cache.
    #[inline]
    pub fn put(&self, block: Block) {
        self.cache.lock().put(*block.cid(), block);
    }

    /// Remove a block from cache.
    pub fn remove(&self, cid: &Cid) {
        self.cache.lock().pop(cid);
    }

    /// Clear the cache and reset counters.
    pub fn clear(&self) {
        self.cache.lock().clear();
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
    }

    /// Return a legacy-style statistics snapshot.
    pub fn stats(&self) -> LegacyCacheStats {
        LegacyCacheStats {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            size: self.cache.lock().len(),
            capacity: self.capacity,
        }
    }

    /// Current number of cached entries.
    pub fn len(&self) -> usize {
        self.cache.lock().len()
    }

    /// `true` if the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.cache.lock().is_empty()
    }
}

// ---------------------------------------------------------------------------
// LegacyCacheStats — simple snapshot used by BlockCache / helpers
// ---------------------------------------------------------------------------

/// Plain-data statistics snapshot returned by [`BlockCache::stats`].
#[derive(Debug, Clone, Default)]
pub struct LegacyCacheStats {
    /// Cache hits since creation (or last clear).
    pub hits: u64,
    /// Cache misses since creation (or last clear).
    pub misses: u64,
    /// Current cache size (number of entries).
    pub size: usize,
    /// Cache capacity (maximum entries).
    pub capacity: usize,
}

impl LegacyCacheStats {
    /// Hit-rate in [0.0, 1.0].
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }

    /// Miss-rate in [0.0, 1.0].
    pub fn miss_rate(&self) -> f64 {
        1.0 - self.hit_rate()
    }
}

// ---------------------------------------------------------------------------
// TieredBlockCache — hot (L1) + warm (L2) in-process tiers
// ---------------------------------------------------------------------------

/// Multi-level in-process cache with hot (L1) and warm (L2) tiers.
///
/// L1 is smaller and faster.  L2 catches blocks evicted from L1.
pub struct TieredBlockCache {
    l1_cache: Arc<Mutex<LruCache<Cid, Block>>>,
    l2_cache: Arc<Mutex<LruCache<Cid, Block>>>,
    l1_capacity: usize,
    l2_capacity: usize,
    l1_hits: Arc<AtomicU64>,
    l2_hits: Arc<AtomicU64>,
    misses: Arc<AtomicU64>,
}

impl TieredBlockCache {
    /// Create a new tiered cache.
    ///
    /// * `l1_capacity` — capacity of the hot tier (number of blocks).
    /// * `l2_capacity` — capacity of the warm tier (number of blocks).
    pub fn new(l1_capacity: usize, l2_capacity: usize) -> Self {
        let l1_cap = NonZeroUsize::new(l1_capacity)
            .unwrap_or_else(|| NonZeroUsize::new(100).expect("100>0"));
        let l2_cap = NonZeroUsize::new(l2_capacity)
            .unwrap_or_else(|| NonZeroUsize::new(1000).expect("1000>0"));

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

    /// Get a block — checks L1 then L2; promotes L2 hits to L1.
    #[inline]
    pub fn get(&self, cid: &Cid) -> Option<Block> {
        if let Some(block) = self.l1_cache.lock().get(cid) {
            self.l1_hits.fetch_add(1, Ordering::Relaxed);
            return Some(block.clone());
        }

        if let Some(block) = self.l2_cache.lock().get(cid) {
            self.l2_hits.fetch_add(1, Ordering::Relaxed);
            let block_clone = block.clone();
            self.l1_cache.lock().put(*cid, block_clone.clone());
            return Some(block_clone);
        }

        self.misses.fetch_add(1, Ordering::Relaxed);
        None
    }

    /// Put a block into L1; blocks evicted from L1 cascade into L2.
    #[inline]
    pub fn put(&self, block: Block) {
        let cid = *block.cid();
        if let Some(evicted) = self.l1_cache.lock().push(cid, block) {
            self.l2_cache.lock().put(evicted.0, evicted.1);
        }
    }

    /// Remove a block from both tiers.
    pub fn remove(&self, cid: &Cid) {
        self.l1_cache.lock().pop(cid);
        self.l2_cache.lock().pop(cid);
    }

    /// Clear both tiers and reset counters.
    pub fn clear(&self) {
        self.l1_cache.lock().clear();
        self.l2_cache.lock().clear();
        self.l1_hits.store(0, Ordering::Relaxed);
        self.l2_hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
    }

    /// Return a statistics snapshot.
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

/// Statistics snapshot for [`TieredBlockCache`].
#[derive(Debug, Clone)]
pub struct TieredCacheStats {
    /// Current L1 size.
    pub l1_size: usize,
    /// L1 capacity.
    pub l1_capacity: usize,
    /// Current L2 size.
    pub l2_size: usize,
    /// L2 capacity.
    pub l2_capacity: usize,
    /// L1 hits.
    pub l1_hits: u64,
    /// L2 hits.
    pub l2_hits: u64,
    /// Total misses (missed both tiers).
    pub misses: u64,
}

impl TieredCacheStats {
    /// Overall hit-rate (L1 + L2 hits / total accesses).
    pub fn hit_rate(&self) -> f64 {
        let total_hits = self.l1_hits + self.l2_hits;
        let total = total_hits + self.misses;
        if total == 0 {
            0.0
        } else {
            total_hits as f64 / total as f64
        }
    }

    /// L1 hit-rate.
    pub fn l1_hit_rate(&self) -> f64 {
        let total = self.l1_hits + self.l2_hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.l1_hits as f64 / total as f64
        }
    }

    /// L2 hit-rate.
    pub fn l2_hit_rate(&self) -> f64 {
        let total = self.l1_hits + self.l2_hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.l2_hits as f64 / total as f64
        }
    }

    /// Miss-rate.
    pub fn miss_rate(&self) -> f64 {
        1.0 - self.hit_rate()
    }
}

// ---------------------------------------------------------------------------
// TieredCachedBlockStore — tiered-cache wrapper around a BlockStore
// ---------------------------------------------------------------------------

/// A [`BlockStore`] wrapper that uses a [`TieredBlockCache`] (hot L1 + warm L2)
/// in front of any backing store.
pub struct TieredCachedBlockStore<S: BlockStore> {
    store: S,
    cache: TieredBlockCache,
}

impl<S: BlockStore> TieredCachedBlockStore<S> {
    /// Create a new tiered caching block store.
    pub fn new(store: S, l1_capacity: usize, l2_capacity: usize) -> Self {
        Self {
            store,
            cache: TieredBlockCache::new(l1_capacity, l2_capacity),
        }
    }

    /// Borrow the underlying store.
    pub fn store(&self) -> &S {
        &self.store
    }

    /// Return cache statistics.
    pub fn cache_stats(&self) -> TieredCacheStats {
        self.cache.stats()
    }
}

#[async_trait]
impl<S: BlockStore> BlockStore for TieredCachedBlockStore<S> {
    async fn put(&self, block: &Block) -> Result<()> {
        self.cache.put(block.clone());
        self.store.put(block).await
    }

    async fn get(&self, cid: &Cid) -> Result<Option<Block>> {
        if let Some(block) = self.cache.get(cid) {
            return Ok(Some(block));
        }
        if let Some(block) = self.store.get(cid).await? {
            self.cache.put(block.clone());
            Ok(Some(block))
        } else {
            Ok(None)
        }
    }

    async fn has(&self, cid: &Cid) -> Result<bool> {
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

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "sled-backend"))]
mod tests {
    use super::*;
    use crate::blockstore::BlockStoreConfig;
    use crate::memory::MemoryBlockStore;
    use bytes::Bytes;
    use ipfrs_core::Block;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn unique_path(tag: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "ipfrs-cache-test-{}-{}-{}",
            tag,
            std::process::id(),
            fastrand::u64(..)
        ))
    }

    fn make_block(content: &[u8]) -> Block {
        Block::new(Bytes::copy_from_slice(content)).expect("block creation")
    }

    fn make_store_with_config(config: CacheConfig) -> CachedBlockStore<MemoryBlockStore> {
        CachedBlockStore::new(MemoryBlockStore::new(), config)
    }

    fn make_store() -> CachedBlockStore<MemoryBlockStore> {
        CachedBlockStore::with_default_config(MemoryBlockStore::new())
    }

    // -----------------------------------------------------------------------
    // test_cache_hit
    // -----------------------------------------------------------------------

    /// Put a block then get it — must be a L1 hit (hit=1, miss=0).
    #[tokio::test]
    async fn test_cache_hit() {
        let store = make_store();
        let block = make_block(b"hello world");

        store.put(&block).await.expect("put");
        let result = store.get(block.cid()).await.expect("get");
        assert!(result.is_some());

        let snap = store.stats();
        assert_eq!(snap.hits, 1, "expected 1 L1 hit");
        assert_eq!(snap.misses, 0, "expected 0 misses");
    }

    // -----------------------------------------------------------------------
    // test_cache_miss_then_populate
    // -----------------------------------------------------------------------

    /// Get from an empty cache (miss), but backed by an L2 that has the block.
    /// After the miss L1 should be populated so a second get is a hit.
    #[tokio::test]
    async fn test_cache_miss_then_populate() {
        // Pre-populate L2 directly.
        let mem = MemoryBlockStore::new();
        let block = make_block(b"L2 resident block");
        mem.put(&block).await.expect("l2 put");

        // Wrap with a fresh L1 — L1 is empty.
        let store = CachedBlockStore::with_default_config(mem);

        // First get: L1 miss → L2 hit → L1 populated.
        let r1 = store.get(block.cid()).await.expect("get 1");
        assert!(r1.is_some());

        let snap1 = store.stats();
        assert_eq!(snap1.misses, 1);
        assert_eq!(snap1.hits, 0);

        // Second get: L1 hit.
        let r2 = store.get(block.cid()).await.expect("get 2");
        assert!(r2.is_some());

        let snap2 = store.stats();
        assert_eq!(snap2.hits, 1);
        assert_eq!(snap2.misses, 1);

        // L1 must now contain the block.
        assert_eq!(store.cache_size(), 1);
    }

    // -----------------------------------------------------------------------
    // test_cache_eviction
    // -----------------------------------------------------------------------

    /// Fill L1 to capacity then add one more block — the oldest entry must be
    /// evicted and the eviction counter must equal 1.
    #[tokio::test]
    async fn test_cache_eviction() {
        let cap = NonZeroUsize::new(3).expect("3>0");
        let config = CacheConfig {
            l1_capacity: cap,
            max_block_bytes: 64 * 1024,
        };
        let store = make_store_with_config(config);

        // Fill to capacity.
        for i in 0_u8..3 {
            let block = make_block(&[i; 8]);
            store.put(&block).await.expect("put");
        }
        assert_eq!(store.cache_size(), 3);
        assert_eq!(store.stats().evictions, 0);

        // One more — triggers eviction.
        let extra = make_block(b"extra block!!!");
        store.put(&extra).await.expect("put extra");

        assert_eq!(store.cache_size(), 3);
        assert_eq!(store.stats().evictions, 1, "expected exactly 1 eviction");
    }

    // -----------------------------------------------------------------------
    // test_cache_delete_removes_from_both
    // -----------------------------------------------------------------------

    /// Put a block, delete it, then get it — must return None from both L1 and L2.
    #[tokio::test]
    async fn test_cache_delete_removes_from_both() {
        let store = make_store();
        let block = make_block(b"to be deleted");

        store.put(&block).await.expect("put");
        store.delete(block.cid()).await.expect("delete");

        // L1 must be empty.
        assert_eq!(store.cache_size(), 0);

        // L2 must also be empty.
        let result = store.get(block.cid()).await.expect("get after delete");
        assert!(result.is_none());
    }

    // -----------------------------------------------------------------------
    // test_cache_large_block_skipped
    // -----------------------------------------------------------------------

    /// A block larger than `max_block_bytes` must not be promoted to L1 but
    /// must still be retrievable from L2.
    #[tokio::test]
    async fn test_cache_large_block_skipped() {
        let config = CacheConfig {
            l1_capacity: NonZeroUsize::new(1024).expect("1024>0"),
            max_block_bytes: 4, // deliberately tiny threshold
        };
        let store = make_store_with_config(config);

        let large_block = make_block(b"larger than threshold");
        store.put(&large_block).await.expect("put large");

        // L1 must be empty because block exceeds threshold.
        assert_eq!(store.cache_size(), 0, "large block must not enter L1");

        // L2 must still have the block.
        let result = store.get(large_block.cid()).await.expect("get large");
        assert!(result.is_some(), "large block must be in L2");
    }

    // -----------------------------------------------------------------------
    // test_hit_rate_calculation
    // -----------------------------------------------------------------------

    /// 3 hits + 1 miss → hit_rate = 0.75.
    #[tokio::test]
    async fn test_hit_rate_calculation() {
        let store = make_store();
        let block = make_block(b"hit rate test");

        store.put(&block).await.expect("put");

        // 3 hits.
        for _ in 0_u8..3 {
            store.get(block.cid()).await.expect("hit");
        }

        // 1 miss (non-existent CID).
        let absent = make_block(b"absent block");
        store.get(absent.cid()).await.expect("miss");

        let snap = store.stats();
        assert_eq!(snap.hits, 3);
        assert_eq!(snap.misses, 1);

        let expected = 3.0_f64 / 4.0_f64;
        let diff = (snap.hit_rate - expected).abs();
        assert!(
            diff < 1e-10,
            "hit_rate={} expected={}",
            snap.hit_rate,
            expected
        );
    }

    // -----------------------------------------------------------------------
    // test_cache_stats_snapshot
    // -----------------------------------------------------------------------

    /// Verify that `CacheStatsSnapshot` fields are accurate after a sequence
    /// of operations.
    #[tokio::test]
    async fn test_cache_stats_snapshot() {
        let config = CacheConfig {
            l1_capacity: NonZeroUsize::new(2).expect("2>0"),
            max_block_bytes: 64 * 1024,
        };
        let store = make_store_with_config(config);

        let b1 = make_block(b"block-one");
        let b2 = make_block(b"block-two");
        let b3 = make_block(b"block-three");

        // 3 puts.
        store.put(&b1).await.expect("put b1");
        store.put(&b2).await.expect("put b2");
        store.put(&b3).await.expect("put b3"); // evicts b1

        // 1 hit (b2 still in L1), 1 miss (b4 absent).
        store.get(b2.cid()).await.expect("get b2");
        let absent = make_block(b"does-not-exist");
        store.get(absent.cid()).await.expect("absent get");

        let snap = store.stats();
        assert_eq!(snap.puts, 3, "puts");
        assert_eq!(snap.hits, 1, "hits");
        assert_eq!(snap.misses, 1, "misses");
        assert!(snap.evictions >= 1, "at least one eviction expected");
        let expected_rate = 0.5_f64;
        let diff = (snap.hit_rate - expected_rate).abs();
        assert!(diff < 1e-10, "hit_rate mismatch: {}", snap.hit_rate);
    }

    // -----------------------------------------------------------------------
    // Sled-backed integration test (uses temp_dir)
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_cached_sled_put_get() {
        use crate::blockstore::SledBlockStore;

        let path = unique_path("sled");
        let _ = std::fs::remove_dir_all(&path);

        let config_sled = BlockStoreConfig {
            path: path.clone(),
            cache_size: 4 * 1024 * 1024,
        };
        let sled = SledBlockStore::new(config_sled).expect("sled open");
        let store = CachedBlockStore::with_default_config(sled);

        let block = make_block(b"sled cache integration");
        store.put(&block).await.expect("put");

        // First get: L1 hit (block inserted during put).
        let r = store.get(block.cid()).await.expect("get");
        assert!(r.is_some());
        assert_eq!(store.stats().hits, 1);

        let _ = std::fs::remove_dir_all(&path);
    }
}
