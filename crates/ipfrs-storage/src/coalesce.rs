//! Write coalescing for batching similar writes
//!
//! Combines multiple write operations into batches to improve performance:
//! - Time-based batching (flush after interval)
//! - Size-based batching (flush when batch size reached)
//! - Automatic flushing on shutdown
//! - Configurable batch sizes and intervals
//!
//! ## Example
//! ```no_run
//! use ipfrs_storage::{CoalescingBlockStore, CoalesceConfig, MemoryBlockStore};
//! use std::time::Duration;
//!
//! #[tokio::main]
//! async fn main() {
//!     let store = MemoryBlockStore::new();
//!     let config = CoalesceConfig::new(100, Duration::from_millis(100));
//!     let coalescing_store = CoalescingBlockStore::new(store, config);
//!
//!     // Writes are automatically batched
//! }
//! ```

use crate::traits::BlockStore;
use async_trait::async_trait;
use ipfrs_core::{Block, Cid, Result as IpfsResult};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::time::sleep;

/// Write coalescing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoalesceConfig {
    /// Maximum batch size before auto-flush
    pub max_batch_size: usize,
    /// Maximum time to wait before auto-flush
    pub max_batch_time: Duration,
    /// Enable automatic background flushing
    pub auto_flush: bool,
}

impl CoalesceConfig {
    /// Create a new coalescing configuration
    pub fn new(max_batch_size: usize, max_batch_time: Duration) -> Self {
        Self {
            max_batch_size,
            max_batch_time,
            auto_flush: true,
        }
    }

    /// Disable automatic background flushing
    pub fn without_auto_flush(mut self) -> Self {
        self.auto_flush = false;
        self
    }
}

impl Default for CoalesceConfig {
    fn default() -> Self {
        Self::new(100, Duration::from_millis(100))
    }
}

/// Pending write operation
#[derive(Debug, Clone)]
struct PendingWrite {
    block: Block,
    #[allow(dead_code)]
    added_at: Instant,
}

/// Internal state for write coalescing
#[derive(Debug)]
struct CoalescingState {
    /// Pending writes by CID
    pending: HashMap<Cid, PendingWrite>,
    /// When the oldest pending write was added
    oldest_write: Option<Instant>,
    /// Total writes coalesced
    total_writes: u64,
    /// Total flushes performed
    total_flushes: u64,
    /// Total blocks written
    total_blocks: u64,
}

/// Coalescing statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CoalesceStats {
    /// Total write operations received
    pub total_writes: u64,
    /// Total flush operations
    pub total_flushes: u64,
    /// Total blocks actually written
    pub total_blocks: u64,
    /// Current pending writes
    pub pending_writes: usize,
    /// Coalescing ratio (writes per flush)
    pub coalescing_ratio: f64,
}

/// Block store with write coalescing
pub struct CoalescingBlockStore<S: BlockStore> {
    inner: S,
    config: CoalesceConfig,
    state: Arc<Mutex<CoalescingState>>,
}

impl<S: BlockStore + Clone> CoalescingBlockStore<S> {
    /// Create a new coalescing block store
    pub fn new(inner: S, config: CoalesceConfig) -> Self
    where
        S: 'static,
    {
        let store = Self {
            inner: inner.clone(),
            config,
            state: Arc::new(Mutex::new(CoalescingState {
                pending: HashMap::new(),
                oldest_write: None,
                total_writes: 0,
                total_flushes: 0,
                total_blocks: 0,
            })),
        };

        // Start background flush task if auto-flush is enabled
        if store.config.auto_flush {
            let state = Arc::clone(&store.state);
            let config = store.config.clone();

            tokio::spawn(async move {
                loop {
                    sleep(config.max_batch_time / 2).await;

                    let should_flush = {
                        let state = state.lock();
                        if let Some(oldest) = state.oldest_write {
                            oldest.elapsed() >= config.max_batch_time
                        } else {
                            false
                        }
                    };

                    if should_flush {
                        let _ = Self::flush_pending(&inner, &state).await;
                    }
                }
            });
        }

        store
    }

    /// Get coalescing statistics
    pub fn stats(&self) -> CoalesceStats {
        let state = self.state.lock();

        CoalesceStats {
            total_writes: state.total_writes,
            total_flushes: state.total_flushes,
            total_blocks: state.total_blocks,
            pending_writes: state.pending.len(),
            coalescing_ratio: if state.total_flushes > 0 {
                state.total_writes as f64 / state.total_flushes as f64
            } else {
                0.0
            },
        }
    }

    /// Manually flush pending writes
    pub async fn flush_writes(&self) -> IpfsResult<usize> {
        Self::flush_pending(&self.inner, &self.state).await
    }

    /// Internal flush implementation
    async fn flush_pending(inner: &S, state: &Arc<Mutex<CoalescingState>>) -> IpfsResult<usize> {
        let blocks_to_write = {
            let mut state = state.lock();
            if state.pending.is_empty() {
                return Ok(0);
            }

            let blocks: Vec<_> = state.pending.values().map(|pw| pw.block.clone()).collect();

            let count = blocks.len();
            state.pending.clear();
            state.oldest_write = None;
            state.total_flushes += 1;
            state.total_blocks += count as u64;

            blocks
        };

        let count = blocks_to_write.len();

        // Write blocks
        inner.put_many(&blocks_to_write).await?;

        Ok(count)
    }
}

#[async_trait]
impl<S: BlockStore + Clone> BlockStore for CoalescingBlockStore<S> {
    async fn get(&self, cid: &Cid) -> IpfsResult<Option<Block>> {
        // Check pending writes first
        {
            let state = self.state.lock();
            if let Some(pending) = state.pending.get(cid) {
                return Ok(Some(pending.block.clone()));
            }
        }

        self.inner.get(cid).await
    }

    async fn put(&self, block: &Block) -> IpfsResult<()> {
        let should_flush = {
            let mut state = self.state.lock();
            state.total_writes += 1;

            let pending_write = PendingWrite {
                block: block.clone(),
                added_at: Instant::now(),
            };

            if state.oldest_write.is_none() {
                state.oldest_write = Some(Instant::now());
            }

            state.pending.insert(*block.cid(), pending_write);

            state.pending.len() >= self.config.max_batch_size
        };

        if should_flush {
            Self::flush_pending(&self.inner, &self.state).await?;
        }

        Ok(())
    }

    async fn has(&self, cid: &Cid) -> IpfsResult<bool> {
        // Check pending writes
        {
            let state = self.state.lock();
            if state.pending.contains_key(cid) {
                return Ok(true);
            }
        }

        self.inner.has(cid).await
    }

    async fn delete(&self, cid: &Cid) -> IpfsResult<()> {
        // Remove from pending if present
        {
            let mut state = self.state.lock();
            state.pending.remove(cid);
            if state.pending.is_empty() {
                state.oldest_write = None;
            }
        }

        self.inner.delete(cid).await
    }

    fn list_cids(&self) -> IpfsResult<Vec<Cid>> {
        let mut cids = self.inner.list_cids()?;

        // Add pending writes
        {
            let state = self.state.lock();
            cids.extend(state.pending.keys().copied());
        }

        cids.sort();
        cids.dedup();
        Ok(cids)
    }

    fn len(&self) -> usize {
        let pending_count = self.state.lock().pending.len();
        self.inner.len() + pending_count
    }

    async fn flush(&self) -> IpfsResult<()> {
        // Flush pending writes first
        Self::flush_pending(&self.inner, &self.state).await?;
        self.inner.flush().await
    }

    async fn put_many(&self, blocks: &[Block]) -> IpfsResult<()> {
        // Add to pending batch
        {
            let mut state = self.state.lock();
            let now = Instant::now();

            if state.oldest_write.is_none() {
                state.oldest_write = Some(now);
            }

            for block in blocks {
                state.total_writes += 1;
                state.pending.insert(
                    *block.cid(),
                    PendingWrite {
                        block: block.clone(),
                        added_at: now,
                    },
                );
            }
        }

        // Flush if batch is large enough
        let should_flush = {
            let state = self.state.lock();
            state.pending.len() >= self.config.max_batch_size
        };

        if should_flush {
            Self::flush_pending(&self.inner, &self.state).await?;
        }

        Ok(())
    }

    async fn get_many(&self, cids: &[Cid]) -> IpfsResult<Vec<Option<Block>>> {
        let mut results = Vec::with_capacity(cids.len());
        let mut missing_cids = Vec::new();

        // Check pending first
        {
            let state = self.state.lock();
            for cid in cids {
                if let Some(pending) = state.pending.get(cid) {
                    results.push(Some(pending.block.clone()));
                } else {
                    results.push(None);
                    missing_cids.push(*cid);
                }
            }
        }

        // Get missing from inner store
        if !missing_cids.is_empty() {
            let inner_results = self.inner.get_many(&missing_cids).await?;
            let mut inner_idx = 0;

            for result in &mut results {
                if result.is_none() {
                    *result = inner_results[inner_idx].clone();
                    inner_idx += 1;
                }
            }
        }

        Ok(results)
    }

    async fn has_many(&self, cids: &[Cid]) -> IpfsResult<Vec<bool>> {
        let mut results = Vec::with_capacity(cids.len());
        let mut missing_cids = Vec::new();

        // Check pending first
        {
            let state = self.state.lock();
            for cid in cids {
                if state.pending.contains_key(cid) {
                    results.push(true);
                } else {
                    results.push(false);
                    missing_cids.push(*cid);
                }
            }
        }

        // Check missing in inner store
        if !missing_cids.is_empty() {
            let inner_results = self.inner.has_many(&missing_cids).await?;
            let mut inner_idx = 0;

            for result in &mut results {
                if !*result {
                    *result = inner_results[inner_idx];
                    inner_idx += 1;
                }
            }
        }

        Ok(results)
    }

    async fn delete_many(&self, cids: &[Cid]) -> IpfsResult<()> {
        // Remove from pending
        {
            let mut state = self.state.lock();
            for cid in cids {
                state.pending.remove(cid);
            }
            if state.pending.is_empty() {
                state.oldest_write = None;
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

    #[tokio::test]
    async fn test_coalescing_basic() {
        let store = MemoryBlockStore::new();
        let config = CoalesceConfig::new(3, Duration::from_secs(10)).without_auto_flush();
        let coalescing = CoalescingBlockStore::new(store, config);

        // Add 2 blocks (below threshold)
        let block1 = create_block(b"data1".to_vec()).unwrap();
        let block2 = create_block(b"data2".to_vec()).unwrap();

        coalescing.put(&block1).await.unwrap();
        coalescing.put(&block2).await.unwrap();

        let stats = coalescing.stats();
        assert_eq!(stats.total_writes, 2);
        assert_eq!(stats.total_flushes, 0);
        assert_eq!(stats.pending_writes, 2);
    }

    #[tokio::test]
    async fn test_coalescing_auto_flush() {
        let store = MemoryBlockStore::new();
        let config = CoalesceConfig::new(2, Duration::from_secs(10)).without_auto_flush();
        let coalescing = CoalescingBlockStore::new(store, config);

        // Add blocks up to threshold
        let block1 = create_block(b"data1".to_vec()).unwrap();
        let block2 = create_block(b"data2".to_vec()).unwrap();

        coalescing.put(&block1).await.unwrap();
        coalescing.put(&block2).await.unwrap();

        // Should have flushed automatically
        let stats = coalescing.stats();
        assert_eq!(stats.total_writes, 2);
        assert_eq!(stats.total_flushes, 1);
        assert_eq!(stats.pending_writes, 0);
    }

    #[tokio::test]
    async fn test_coalescing_manual_flush() {
        let store = MemoryBlockStore::new();
        let config = CoalesceConfig::new(100, Duration::from_secs(10)).without_auto_flush();
        let coalescing = CoalescingBlockStore::new(store, config);

        // Add some blocks
        for i in 0..5 {
            let block = create_block(vec![i; 10]).unwrap();
            coalescing.put(&block).await.unwrap();
        }

        assert_eq!(coalescing.stats().pending_writes, 5);

        // Manual flush
        let flushed = coalescing.flush_writes().await.unwrap();
        assert_eq!(flushed, 5);
        assert_eq!(coalescing.stats().pending_writes, 0);
    }

    #[tokio::test]
    async fn test_coalescing_read_pending() {
        let store = MemoryBlockStore::new();
        let config = CoalesceConfig::new(100, Duration::from_secs(10)).without_auto_flush();
        let coalescing = CoalescingBlockStore::new(store, config);

        let block = create_block(b"test data".to_vec()).unwrap();
        let cid = *block.cid();

        // Write but don't flush
        coalescing.put(&block).await.unwrap();

        // Should be able to read from pending
        assert!(coalescing.has(&cid).await.unwrap());
        let retrieved = coalescing.get(&cid).await.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().data(), block.data());
    }

    #[tokio::test]
    async fn test_coalescing_stats() {
        let store = MemoryBlockStore::new();
        let config = CoalesceConfig::new(3, Duration::from_secs(10)).without_auto_flush();
        let coalescing = CoalescingBlockStore::new(store, config);

        // Add blocks
        for i in 0..6 {
            let block = create_block(vec![i; 10]).unwrap();
            coalescing.put(&block).await.unwrap();
        }

        let stats = coalescing.stats();
        assert_eq!(stats.total_writes, 6);
        assert_eq!(stats.total_flushes, 2); // Two auto-flushes at threshold
        assert!(stats.coalescing_ratio > 0.0);
    }
}
