//! Batch operation utilities for efficient bulk processing
//!
//! This module provides utilities for performing batch operations on block stores
//! with features like parallel processing, error handling, and progress tracking.

use crate::traits::BlockStore;
use ipfrs_core::{Block, Cid};
use std::sync::Arc;
use tokio::sync::Semaphore;

/// Batch operation configuration
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Maximum concurrent operations
    pub max_concurrency: usize,
    /// Batch size for chunking operations
    pub batch_size: usize,
    /// Whether to stop on first error or continue
    pub fail_fast: bool,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_concurrency: 10,
            batch_size: 100,
            fail_fast: false,
        }
    }
}

impl BatchConfig {
    /// Create a new batch config with custom settings
    pub fn new(max_concurrency: usize, batch_size: usize) -> Self {
        Self {
            max_concurrency,
            batch_size,
            fail_fast: false,
        }
    }

    /// Set whether to fail fast on first error
    pub fn with_fail_fast(mut self, fail_fast: bool) -> Self {
        self.fail_fast = fail_fast;
        self
    }

    /// Optimized for high throughput
    pub fn high_throughput() -> Self {
        Self {
            max_concurrency: 50,
            batch_size: 500,
            fail_fast: false,
        }
    }

    /// Optimized for low latency
    pub fn low_latency() -> Self {
        Self {
            max_concurrency: 20,
            batch_size: 50,
            fail_fast: false,
        }
    }

    /// Conservative settings for resource-constrained environments
    pub fn conservative() -> Self {
        Self {
            max_concurrency: 5,
            batch_size: 20,
            fail_fast: false,
        }
    }
}

/// Result of a batch operation
#[derive(Debug, Clone)]
pub struct BatchResult<T> {
    /// Successfully processed items
    pub successful: Vec<T>,
    /// Failed items with their errors
    pub failed: Vec<(T, String)>,
    /// Total number of items processed
    pub total: usize,
}

impl<T> BatchResult<T> {
    /// Create a new batch result
    pub fn new() -> Self {
        Self {
            successful: Vec::new(),
            failed: Vec::new(),
            total: 0,
        }
    }

    /// Check if all operations succeeded
    pub fn is_success(&self) -> bool {
        self.failed.is_empty()
    }

    /// Get success rate (0.0 to 1.0)
    pub fn success_rate(&self) -> f64 {
        if self.total == 0 {
            1.0
        } else {
            self.successful.len() as f64 / self.total as f64
        }
    }

    /// Get number of successful operations
    pub fn success_count(&self) -> usize {
        self.successful.len()
    }

    /// Get number of failed operations
    pub fn failure_count(&self) -> usize {
        self.failed.len()
    }
}

impl<T> Default for BatchResult<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Batch put blocks with concurrency control
///
/// Puts multiple blocks efficiently with configurable parallelism.
/// Returns a result indicating success/failure for each block.
pub async fn batch_put<S: BlockStore + Send + Sync + 'static>(
    store: Arc<S>,
    blocks: Vec<Block>,
    config: BatchConfig,
) -> BatchResult<Cid> {
    let mut result = BatchResult::new();
    result.total = blocks.len();

    let semaphore = Arc::new(Semaphore::new(config.max_concurrency));
    let mut handles = Vec::new();

    for chunk in blocks.chunks(config.batch_size) {
        for block in chunk {
            let permit = semaphore
                .clone()
                .acquire_owned()
                .await
                .expect("semaphore is never explicitly closed");
            let block = block.clone();
            let cid = *block.cid();
            let store = store.clone();

            let handle = tokio::spawn(async move {
                let _permit = permit; // Hold permit until task completes
                (cid, store.put(&block).await)
            });

            handles.push(handle);
        }

        // Wait for this chunk to complete
        for handle in handles.drain(..) {
            match handle.await {
                Ok((cid, Ok(_))) => result.successful.push(cid),
                Ok((cid, Err(e))) => {
                    result.failed.push((cid, e.to_string()));
                    if config.fail_fast {
                        return result;
                    }
                }
                Err(e) => {
                    // Task panicked or was cancelled
                    result
                        .failed
                        .push((Cid::default(), format!("Task error: {e}")));
                }
            }
        }
    }

    result
}

/// Batch get blocks with concurrency control
///
/// Retrieves multiple blocks efficiently with configurable parallelism.
pub async fn batch_get<S: BlockStore + Send + Sync + 'static>(
    store: Arc<S>,
    cids: Vec<Cid>,
    config: BatchConfig,
) -> BatchResult<Block> {
    let mut result = BatchResult::new();
    result.total = cids.len();

    let semaphore = Arc::new(Semaphore::new(config.max_concurrency));
    let mut handles = Vec::new();

    for chunk in cids.chunks(config.batch_size) {
        for cid in chunk {
            let permit = semaphore
                .clone()
                .acquire_owned()
                .await
                .expect("semaphore is never explicitly closed");
            let cid = *cid;
            let store = store.clone();

            let handle = tokio::spawn(async move {
                let _permit = permit;
                (cid, store.get(&cid).await)
            });

            handles.push(handle);
        }

        // Wait for this chunk to complete
        for handle in handles.drain(..) {
            match handle.await {
                Ok((_cid, Ok(Some(block)))) => result.successful.push(block),
                Ok((cid, Ok(None))) => {
                    result.failed.push((
                        Block::from_parts(cid, bytes::Bytes::new()),
                        "Block not found".to_string(),
                    ));
                }
                Ok((cid, Err(e))) => {
                    result
                        .failed
                        .push((Block::from_parts(cid, bytes::Bytes::new()), e.to_string()));
                    if config.fail_fast {
                        return result;
                    }
                }
                Err(e) => {
                    result.failed.push((
                        Block::from_parts(Cid::default(), bytes::Bytes::new()),
                        format!("Task error: {e}"),
                    ));
                }
            }
        }
    }

    result
}

/// Batch delete blocks with concurrency control
pub async fn batch_delete<S: BlockStore + Send + Sync + 'static>(
    store: Arc<S>,
    cids: Vec<Cid>,
    config: BatchConfig,
) -> BatchResult<Cid> {
    let mut result = BatchResult::new();
    result.total = cids.len();

    let semaphore = Arc::new(Semaphore::new(config.max_concurrency));
    let mut handles = Vec::new();

    for chunk in cids.chunks(config.batch_size) {
        for cid in chunk {
            let permit = semaphore
                .clone()
                .acquire_owned()
                .await
                .expect("semaphore is never explicitly closed");
            let cid = *cid;
            let store = store.clone();

            let handle = tokio::spawn(async move {
                let _permit = permit;
                (cid, store.delete(&cid).await)
            });

            handles.push(handle);
        }

        // Wait for this chunk to complete
        for handle in handles.drain(..) {
            match handle.await {
                Ok((cid, Ok(_))) => result.successful.push(cid),
                Ok((cid, Err(e))) => {
                    result.failed.push((cid, e.to_string()));
                    if config.fail_fast {
                        return result;
                    }
                }
                Err(e) => {
                    result
                        .failed
                        .push((Cid::default(), format!("Task error: {e}")));
                }
            }
        }
    }

    result
}

/// Batch check existence with concurrency control
pub async fn batch_has<S: BlockStore + Send + Sync + 'static>(
    store: Arc<S>,
    cids: Vec<Cid>,
    config: BatchConfig,
) -> BatchResult<(Cid, bool)> {
    let mut result = BatchResult::new();
    result.total = cids.len();

    let semaphore = Arc::new(Semaphore::new(config.max_concurrency));
    let mut handles = Vec::new();

    for chunk in cids.chunks(config.batch_size) {
        for cid in chunk {
            let permit = semaphore
                .clone()
                .acquire_owned()
                .await
                .expect("semaphore is never explicitly closed");
            let cid = *cid;
            let store = store.clone();

            let handle = tokio::spawn(async move {
                let _permit = permit;
                (cid, store.has(&cid).await)
            });

            handles.push(handle);
        }

        // Wait for this chunk to complete
        for handle in handles.drain(..) {
            match handle.await {
                Ok((cid, Ok(exists))) => result.successful.push((cid, exists)),
                Ok((cid, Err(e))) => {
                    result.failed.push(((cid, false), e.to_string()));
                    if config.fail_fast {
                        return result;
                    }
                }
                Err(e) => {
                    result
                        .failed
                        .push(((Cid::default(), false), format!("Task error: {e}")));
                }
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MemoryBlockStore;
    use bytes::Bytes;

    #[tokio::test]
    async fn test_batch_put() {
        let store = Arc::new(MemoryBlockStore::new());
        let mut blocks = Vec::new();

        for i in 0..10 {
            let data = format!("block {}", i);
            let block = Block::new(Bytes::from(data)).unwrap();
            blocks.push(block);
        }

        let config = BatchConfig::default();
        let result = batch_put(store.clone(), blocks.clone(), config).await;

        assert!(result.is_success());
        assert_eq!(result.success_count(), 10);
        assert_eq!(result.failure_count(), 0);
        assert_eq!(result.success_rate(), 1.0);
    }

    #[tokio::test]
    async fn test_batch_get() {
        let store = Arc::new(MemoryBlockStore::new());
        let mut blocks = Vec::new();
        let mut cids = Vec::new();

        for i in 0..5 {
            let data = format!("block {}", i);
            let block = Block::new(Bytes::from(data)).unwrap();
            cids.push(*block.cid());
            store.put(&block).await.unwrap();
            blocks.push(block);
        }

        let config = BatchConfig::default();
        let result = batch_get(store.clone(), cids, config).await;

        assert!(result.is_success());
        assert_eq!(result.success_count(), 5);
    }

    #[tokio::test]
    async fn test_batch_has() {
        let store = Arc::new(MemoryBlockStore::new());
        let mut cids = Vec::new();

        for i in 0..5 {
            let data = format!("block {}", i);
            let block = Block::new(Bytes::from(data)).unwrap();
            cids.push(*block.cid());
            store.put(&block).await.unwrap();
        }

        let config = BatchConfig::default();
        let result = batch_has(store.clone(), cids, config).await;

        assert!(result.is_success());
        assert_eq!(result.success_count(), 5);

        // All blocks should exist
        for (_, exists) in result.successful {
            assert!(exists);
        }
    }

    #[tokio::test]
    async fn test_batch_delete() {
        let store = Arc::new(MemoryBlockStore::new());
        let mut cids = Vec::new();

        for i in 0..5 {
            let data = format!("block {}", i);
            let block = Block::new(Bytes::from(data)).unwrap();
            cids.push(*block.cid());
            store.put(&block).await.unwrap();
        }

        let config = BatchConfig::default();
        let result = batch_delete(store.clone(), cids.clone(), config).await;

        assert!(result.is_success());
        assert_eq!(result.success_count(), 5);

        // Verify blocks are deleted
        for cid in cids {
            assert!(!store.has(&cid).await.unwrap());
        }
    }

    #[test]
    fn test_batch_config_presets() {
        let high_throughput = BatchConfig::high_throughput();
        assert_eq!(high_throughput.max_concurrency, 50);
        assert_eq!(high_throughput.batch_size, 500);

        let low_latency = BatchConfig::low_latency();
        assert_eq!(low_latency.max_concurrency, 20);
        assert_eq!(low_latency.batch_size, 50);

        let conservative = BatchConfig::conservative();
        assert_eq!(conservative.max_concurrency, 5);
        assert_eq!(conservative.batch_size, 20);
    }

    #[test]
    fn test_batch_result() {
        let mut result = BatchResult::<i32>::new();
        result.total = 10;
        result.successful = vec![1, 2, 3, 4, 5];
        result.failed = vec![(6, "error".to_string())];

        assert!(!result.is_success());
        assert_eq!(result.success_count(), 5);
        assert_eq!(result.failure_count(), 1);
        assert_eq!(result.success_rate(), 0.5);
    }
}
