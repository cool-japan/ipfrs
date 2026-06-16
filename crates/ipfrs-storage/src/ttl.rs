//! Time-To-Live (TTL) support for automatic block expiration
//!
//! Provides automatic expiration of blocks after a specified duration.
//! Useful for:
//! - Cache invalidation
//! - Temporary data storage
//! - Preventing unbounded storage growth
//! - Compliance with data retention policies
//!
//! ## Example
//! ```no_run
//! use ipfrs_storage::{TtlBlockStore, TtlConfig, MemoryBlockStore};
//! use std::time::Duration;
//!
//! #[tokio::main]
//! async fn main() {
//!     let store = MemoryBlockStore::new();
//!     let config = TtlConfig::new(Duration::from_secs(3600)); // 1 hour TTL
//!     let ttl_store = TtlBlockStore::new(store, config);
//!
//!     // Blocks will automatically expire after 1 hour
//! }
//! ```

use crate::traits::BlockStore;
use async_trait::async_trait;
use ipfrs_core::{Block, Cid, Result as IpfsResult};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// TTL configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TtlConfig {
    /// Default TTL for blocks
    pub default_ttl: Duration,
    /// Enable automatic cleanup of expired blocks
    pub auto_cleanup: bool,
    /// Cleanup interval (how often to check for expired blocks)
    pub cleanup_interval: Duration,
    /// Maximum number of blocks to track
    pub max_tracked_blocks: usize,
}

impl TtlConfig {
    /// Create a new TTL configuration
    pub fn new(default_ttl: Duration) -> Self {
        Self {
            default_ttl,
            auto_cleanup: true,
            cleanup_interval: Duration::from_secs(60),
            max_tracked_blocks: 1_000_000,
        }
    }

    /// Create config with no automatic cleanup
    pub fn manual_cleanup(default_ttl: Duration) -> Self {
        Self {
            default_ttl,
            auto_cleanup: false,
            cleanup_interval: Duration::from_secs(60),
            max_tracked_blocks: 1_000_000,
        }
    }

    /// Set cleanup interval
    pub fn with_cleanup_interval(mut self, interval: Duration) -> Self {
        self.cleanup_interval = interval;
        self
    }

    /// Set maximum tracked blocks
    pub fn with_max_tracked_blocks(mut self, max: usize) -> Self {
        self.max_tracked_blocks = max;
        self
    }
}

impl Default for TtlConfig {
    fn default() -> Self {
        Self::new(Duration::from_secs(3600)) // 1 hour default
    }
}

/// TTL metadata for a block
#[derive(Debug, Clone)]
struct TtlMetadata {
    /// When the block was stored
    stored_at: Instant,
    /// TTL for this block
    ttl: Duration,
    /// Size of the block in bytes
    size: usize,
}

impl TtlMetadata {
    /// Check if block has expired
    fn is_expired(&self) -> bool {
        self.stored_at.elapsed() >= self.ttl
    }

    /// Time remaining before expiration
    fn time_remaining(&self) -> Option<Duration> {
        let elapsed = self.stored_at.elapsed();
        if elapsed < self.ttl {
            Some(self.ttl - elapsed)
        } else {
            None
        }
    }
}

/// TTL statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TtlStats {
    /// Total blocks tracked
    pub total_tracked: usize,
    /// Expired blocks cleaned up
    pub expired_cleaned: u64,
    /// Total bytes freed from cleanup
    pub bytes_freed: u64,
    /// Last cleanup time
    pub last_cleanup: Option<String>,
    /// Average TTL remaining
    pub avg_ttl_remaining_secs: u64,
}

/// Block store with TTL support
pub struct TtlBlockStore<S: BlockStore> {
    /// Underlying storage
    inner: S,
    /// TTL configuration
    config: TtlConfig,
    /// TTL metadata for blocks
    metadata: Arc<RwLock<HashMap<Cid, TtlMetadata>>>,
    /// Statistics
    stats: Arc<RwLock<TtlStats>>,
    /// Last cleanup time
    last_cleanup: Arc<RwLock<Instant>>,
}

impl<S: BlockStore> TtlBlockStore<S> {
    /// Create a new TTL block store
    pub fn new(inner: S, config: TtlConfig) -> Self {
        Self {
            inner,
            config,
            metadata: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(TtlStats::default())),
            last_cleanup: Arc::new(RwLock::new(Instant::now())),
        }
    }

    /// Set TTL for a specific block
    pub fn set_ttl(&self, cid: &Cid, ttl: Duration) {
        if let Some(metadata) = self.metadata.write().get_mut(cid) {
            metadata.ttl = ttl;
        }
    }

    /// Get TTL for a block
    pub fn get_ttl(&self, cid: &Cid) -> Option<Duration> {
        self.metadata
            .read()
            .get(cid)
            .and_then(|m| m.time_remaining())
    }

    /// Check if a block has expired
    pub fn is_expired(&self, cid: &Cid) -> bool {
        self.metadata
            .read()
            .get(cid)
            .map(|m| m.is_expired())
            .unwrap_or(false)
    }

    /// Get statistics
    pub fn stats(&self) -> TtlStats {
        let mut stats = self.stats.read().clone();
        stats.total_tracked = self.metadata.read().len();

        // Calculate average TTL remaining
        let metadata = self.metadata.read();
        if !metadata.is_empty() {
            let total_remaining: u64 = metadata
                .values()
                .filter_map(|m| m.time_remaining())
                .map(|d| d.as_secs())
                .sum();
            stats.avg_ttl_remaining_secs = total_remaining / metadata.len() as u64;
        }

        stats
    }

    /// Manually trigger cleanup of expired blocks
    pub async fn cleanup_expired(&self) -> IpfsResult<TtlCleanupResult> {
        let mut to_delete = Vec::new();
        let mut bytes_to_free = 0usize;

        // Find expired blocks
        {
            let metadata = self.metadata.read();
            for (cid, meta) in metadata.iter() {
                if meta.is_expired() {
                    to_delete.push(*cid);
                    bytes_to_free += meta.size;
                }
            }
        }

        // Delete expired blocks
        let mut deleted_count = 0;
        for cid in &to_delete {
            if self.inner.delete(cid).await.is_ok() {
                self.metadata.write().remove(cid);
                deleted_count += 1;
            }
        }

        // Update statistics
        {
            let mut stats = self.stats.write();
            stats.expired_cleaned += deleted_count;
            stats.bytes_freed += bytes_to_free as u64;
            stats.last_cleanup = Some(chrono::Utc::now().to_rfc3339());
        }

        *self.last_cleanup.write() = Instant::now();

        Ok(TtlCleanupResult {
            blocks_deleted: deleted_count,
            bytes_freed: bytes_to_free as u64,
        })
    }

    /// Check and perform auto-cleanup if needed
    async fn auto_cleanup_if_needed(&self) -> IpfsResult<()> {
        if !self.config.auto_cleanup {
            return Ok(());
        }

        let should_cleanup = {
            let last = *self.last_cleanup.read();
            last.elapsed() >= self.config.cleanup_interval
        };

        if should_cleanup {
            let _ = self.cleanup_expired().await;
        }

        Ok(())
    }

    /// Track a new block
    fn track_block(&self, cid: &Cid, size: usize, ttl: Option<Duration>) {
        let mut metadata = self.metadata.write();

        // Enforce max tracked blocks limit
        if metadata.len() >= self.config.max_tracked_blocks {
            // Remove oldest block (simple FIFO eviction)
            if let Some(oldest_cid) = metadata.keys().next().cloned() {
                metadata.remove(&oldest_cid);
            }
        }

        metadata.insert(
            *cid,
            TtlMetadata {
                stored_at: Instant::now(),
                ttl: ttl.unwrap_or(self.config.default_ttl),
                size,
            },
        );
    }
}

/// Result of TTL cleanup operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TtlCleanupResult {
    /// Number of blocks deleted
    pub blocks_deleted: u64,
    /// Bytes freed
    pub bytes_freed: u64,
}

#[async_trait]
impl<S: BlockStore> BlockStore for TtlBlockStore<S> {
    async fn get(&self, cid: &Cid) -> IpfsResult<Option<Block>> {
        // Check if expired
        if self.is_expired(cid) {
            // Remove expired block
            let _ = self.inner.delete(cid).await;
            self.metadata.write().remove(cid);
            return Ok(None);
        }

        // Trigger auto-cleanup if needed
        let _ = self.auto_cleanup_if_needed().await;

        self.inner.get(cid).await
    }

    async fn put(&self, block: &Block) -> IpfsResult<()> {
        let cid = *block.cid();
        let size = block.data().len();

        // Store block
        self.inner.put(block).await?;

        // Track TTL
        self.track_block(&cid, size, None);

        // Trigger auto-cleanup if needed
        let _ = self.auto_cleanup_if_needed().await;

        Ok(())
    }

    async fn has(&self, cid: &Cid) -> IpfsResult<bool> {
        // Check if expired
        if self.is_expired(cid) {
            return Ok(false);
        }

        self.inner.has(cid).await
    }

    async fn delete(&self, cid: &Cid) -> IpfsResult<()> {
        self.metadata.write().remove(cid);
        self.inner.delete(cid).await
    }

    fn list_cids(&self) -> IpfsResult<Vec<Cid>> {
        let mut cids = self.inner.list_cids()?;

        // Filter out expired blocks
        cids.retain(|cid| !self.is_expired(cid));

        Ok(cids)
    }

    fn len(&self) -> usize {
        self.list_cids().unwrap_or_default().len()
    }

    async fn flush(&self) -> IpfsResult<()> {
        self.inner.flush().await
    }

    async fn put_many(&self, blocks: &[Block]) -> IpfsResult<()> {
        // Track all blocks
        for block in blocks {
            self.track_block(block.cid(), block.data().len(), None);
        }

        self.inner.put_many(blocks).await
    }

    async fn get_many(&self, cids: &[Cid]) -> IpfsResult<Vec<Option<Block>>> {
        // Filter out expired CIDs
        let valid_cids: Vec<_> = cids
            .iter()
            .filter(|cid| !self.is_expired(cid))
            .cloned()
            .collect();

        self.inner.get_many(&valid_cids).await
    }

    async fn has_many(&self, cids: &[Cid]) -> IpfsResult<Vec<bool>> {
        let mut results = Vec::with_capacity(cids.len());

        for cid in cids {
            if self.is_expired(cid) {
                results.push(false);
            } else {
                results.push(self.inner.has(cid).await?);
            }
        }

        Ok(results)
    }

    async fn delete_many(&self, cids: &[Cid]) -> IpfsResult<()> {
        // Remove from metadata
        {
            let mut metadata = self.metadata.write();
            for cid in cids {
                metadata.remove(cid);
            }
        }

        self.inner.delete_many(cids).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::MemoryBlockStore;
    use crate::utils::create_block;
    use tokio::time::sleep;

    #[tokio::test]
    async fn test_ttl_basic() {
        let store = MemoryBlockStore::new();
        let config = TtlConfig::new(Duration::from_millis(100));
        let ttl_store = TtlBlockStore::new(store, config);

        let block = create_block(b"hello world".to_vec()).unwrap();
        let cid = *block.cid();

        // Put block
        ttl_store.put(&block).await.unwrap();

        // Should exist immediately
        assert!(ttl_store.has(&cid).await.unwrap());

        // Wait for expiration
        sleep(Duration::from_millis(150)).await;

        // Should be expired
        assert!(ttl_store.is_expired(&cid));
        assert!(!ttl_store.has(&cid).await.unwrap());
    }

    #[tokio::test]
    async fn test_ttl_custom_per_block() {
        let store = MemoryBlockStore::new();
        let config = TtlConfig::new(Duration::from_secs(3600));
        let ttl_store = TtlBlockStore::new(store, config);

        let block = create_block(b"test".to_vec()).unwrap();
        let cid = *block.cid();

        ttl_store.put(&block).await.unwrap();

        // Set custom TTL
        ttl_store.set_ttl(&cid, Duration::from_millis(50));

        sleep(Duration::from_millis(100)).await;

        assert!(ttl_store.is_expired(&cid));
    }

    #[tokio::test]
    async fn test_ttl_cleanup() {
        let store = MemoryBlockStore::new();
        let config = TtlConfig::new(Duration::from_millis(50));
        let ttl_store = TtlBlockStore::new(store, config);

        // Add some blocks
        for i in 0..5 {
            let block = create_block(vec![i; 100]).unwrap();
            ttl_store.put(&block).await.unwrap();
        }

        // Wait for expiration
        sleep(Duration::from_millis(100)).await;

        // Trigger cleanup
        let result = ttl_store.cleanup_expired().await.unwrap();

        assert_eq!(result.blocks_deleted, 5);
        assert!(result.bytes_freed > 0);

        let stats = ttl_store.stats();
        assert_eq!(stats.expired_cleaned, 5);
    }

    #[tokio::test]
    async fn test_ttl_stats() {
        let store = MemoryBlockStore::new();
        let config = TtlConfig::new(Duration::from_secs(3600));
        let ttl_store = TtlBlockStore::new(store, config);

        let block = create_block(b"data".to_vec()).unwrap();
        ttl_store.put(&block).await.unwrap();

        let stats = ttl_store.stats();
        assert_eq!(stats.total_tracked, 1);
        assert!(stats.avg_ttl_remaining_secs > 0);
    }

    #[tokio::test]
    async fn test_ttl_max_tracked_blocks() {
        let store = MemoryBlockStore::new();
        let config = TtlConfig::new(Duration::from_secs(3600)).with_max_tracked_blocks(3);
        let ttl_store = TtlBlockStore::new(store, config);

        // Add more blocks than the limit
        for i in 0..5 {
            let block = create_block(vec![i; 10]).unwrap();
            ttl_store.put(&block).await.unwrap();
        }

        let stats = ttl_store.stats();
        assert!(stats.total_tracked <= 3);
    }
}
