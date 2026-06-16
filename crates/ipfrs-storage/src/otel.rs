//! OpenTelemetry integration for distributed tracing
//!
//! This module provides OpenTelemetry tracing integration for storage operations.
//! It allows tracking requests across distributed systems and analyzing performance.
//!
//! # Example
//!
//! ```rust,no_run
//! use ipfrs_storage::{OtelBlockStore, MemoryBlockStore, BlockStoreTrait};
//! use ipfrs_core::Block;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let store = MemoryBlockStore::new();
//! let traced_store = OtelBlockStore::new(store, "storage_node_1".to_string());
//!
//! // Operations are now automatically traced
//! let block = Block::new(b"hello".to_vec().into())?;
//! traced_store.put(&block).await?;
//! # Ok(())
//! # }
//! ```

use crate::traits::BlockStore;
use async_trait::async_trait;
use ipfrs_core::{Block, Cid, Result};
use std::sync::Arc;
use std::time::Instant;
use tracing::{debug, error, info_span, warn, Instrument};

/// OpenTelemetry-instrumented BlockStore wrapper
///
/// This wrapper adds distributed tracing spans to all storage operations,
/// making it easy to track performance and debug issues in distributed systems.
#[derive(Clone)]
pub struct OtelBlockStore<S> {
    inner: Arc<S>,
    service_name: String,
}

impl<S: BlockStore> OtelBlockStore<S> {
    /// Create a new OpenTelemetry-instrumented block store
    pub fn new(store: S, service_name: String) -> Self {
        Self {
            inner: Arc::new(store),
            service_name,
        }
    }

    /// Get reference to the inner store
    pub fn inner(&self) -> &S {
        &self.inner
    }

    /// Extract span attributes from a CID
    #[allow(dead_code)]
    fn cid_attributes(cid: &Cid) -> Vec<(&'static str, String)> {
        vec![
            ("cid", cid.to_string()),
            ("cid.version", format!("{:?}", cid.version())),
            ("cid.codec", format!("{}", cid.codec())),
        ]
    }
}

#[async_trait]
impl<S: BlockStore + Send + Sync + 'static> BlockStore for OtelBlockStore<S> {
    async fn put(&self, block: &Block) -> Result<()> {
        let cid = block.cid();
        let start = Instant::now();

        let span = info_span!(
            "blockstore.put",
            service.name = %self.service_name,
            cid = %cid,
            block.size = block.data().len(),
        );

        let result = self.inner.put(block).instrument(span).await;

        let duration_us = start.elapsed().as_micros();
        match &result {
            Ok(_) => {
                debug!(
                    cid = %cid,
                    duration_us = duration_us,
                    "Block put succeeded"
                );
            }
            Err(e) => {
                error!(
                    cid = %cid,
                    duration_us = duration_us,
                    error = %e,
                    "Block put failed"
                );
            }
        }

        result
    }

    async fn get(&self, cid: &Cid) -> Result<Option<Block>> {
        let start = Instant::now();

        let span = info_span!(
            "blockstore.get",
            service.name = %self.service_name,
            cid = %cid,
        );

        let result = self.inner.get(cid).instrument(span).await;

        let duration_us = start.elapsed().as_micros();
        match &result {
            Ok(Some(block)) => {
                debug!(
                    cid = %cid,
                    duration_us = duration_us,
                    block.size = block.data().len(),
                    "Block get succeeded (hit)"
                );
            }
            Ok(None) => {
                debug!(
                    cid = %cid,
                    duration_us = duration_us,
                    "Block get succeeded (miss)"
                );
            }
            Err(e) => {
                error!(
                    cid = %cid,
                    duration_us = duration_us,
                    error = %e,
                    "Block get failed"
                );
            }
        }

        result
    }

    async fn has(&self, cid: &Cid) -> Result<bool> {
        let start = Instant::now();

        let span = info_span!(
            "blockstore.has",
            service.name = %self.service_name,
            cid = %cid,
        );

        let result = self.inner.has(cid).instrument(span).await;

        let duration_us = start.elapsed().as_micros();
        match &result {
            Ok(exists) => {
                debug!(
                    cid = %cid,
                    duration_us = duration_us,
                    exists = exists,
                    "Block has check succeeded"
                );
            }
            Err(e) => {
                error!(
                    cid = %cid,
                    duration_us = duration_us,
                    error = %e,
                    "Block has check failed"
                );
            }
        }

        result
    }

    async fn delete(&self, cid: &Cid) -> Result<()> {
        let start = Instant::now();

        let span = info_span!(
            "blockstore.delete",
            service.name = %self.service_name,
            cid = %cid,
        );

        let result = self.inner.delete(cid).instrument(span).await;

        let duration_us = start.elapsed().as_micros();
        match &result {
            Ok(_) => {
                debug!(
                    cid = %cid,
                    duration_us = duration_us,
                    "Block delete succeeded"
                );
            }
            Err(e) => {
                error!(
                    cid = %cid,
                    duration_us = duration_us,
                    error = %e,
                    "Block delete failed"
                );
            }
        }

        result
    }

    async fn put_many(&self, blocks: &[Block]) -> Result<()> {
        let start = Instant::now();
        let total_size: usize = blocks.iter().map(|b| b.data().len()).sum();

        let span = info_span!(
            "blockstore.put_many",
            service.name = %self.service_name,
            blocks.count = blocks.len(),
            blocks.total_size = total_size,
        );

        let result = self.inner.put_many(blocks).instrument(span).await;

        let duration_us = start.elapsed().as_micros();
        match &result {
            Ok(_) => {
                debug!(
                    blocks.count = blocks.len(),
                    duration_us = duration_us,
                    throughput_mbps = (total_size as f64 / duration_us as f64) * 1000.0,
                    "Batch put succeeded"
                );
            }
            Err(e) => {
                error!(
                    blocks.count = blocks.len(),
                    duration_us = duration_us,
                    error = %e,
                    "Batch put failed"
                );
            }
        }

        result
    }

    async fn get_many(&self, cids: &[Cid]) -> Result<Vec<Option<Block>>> {
        let start = Instant::now();

        let span = info_span!(
            "blockstore.get_many",
            service.name = %self.service_name,
            cids.count = cids.len(),
        );

        let result = self.inner.get_many(cids).instrument(span).await;

        let duration_us = start.elapsed().as_micros();
        match &result {
            Ok(blocks) => {
                let hits = blocks.iter().filter(|b| b.is_some()).count();
                let total_size: usize = blocks
                    .iter()
                    .filter_map(|b| b.as_ref())
                    .map(|b| b.data().len())
                    .sum();

                debug!(
                    cids.count = cids.len(),
                    hits = hits,
                    misses = cids.len() - hits,
                    duration_us = duration_us,
                    total_size = total_size,
                    "Batch get succeeded"
                );
            }
            Err(e) => {
                error!(
                    cids.count = cids.len(),
                    duration_us = duration_us,
                    error = %e,
                    "Batch get failed"
                );
            }
        }

        result
    }

    async fn has_many(&self, cids: &[Cid]) -> Result<Vec<bool>> {
        let start = Instant::now();

        let span = info_span!(
            "blockstore.has_many",
            service.name = %self.service_name,
            cids.count = cids.len(),
        );

        let result = self.inner.has_many(cids).instrument(span).await;

        let duration_us = start.elapsed().as_micros();
        match &result {
            Ok(results) => {
                let exists_count = results.iter().filter(|&&b| b).count();
                debug!(
                    cids.count = cids.len(),
                    exists = exists_count,
                    not_exists = cids.len() - exists_count,
                    duration_us = duration_us,
                    "Batch has check succeeded"
                );
            }
            Err(e) => {
                error!(
                    cids.count = cids.len(),
                    duration_us = duration_us,
                    error = %e,
                    "Batch has check failed"
                );
            }
        }

        result
    }

    async fn delete_many(&self, cids: &[Cid]) -> Result<()> {
        let start = Instant::now();

        let span = info_span!(
            "blockstore.delete_many",
            service.name = %self.service_name,
            cids.count = cids.len(),
        );

        let result = self.inner.delete_many(cids).instrument(span).await;

        let duration_us = start.elapsed().as_micros();
        match &result {
            Ok(_) => {
                debug!(
                    cids.count = cids.len(),
                    duration_us = duration_us,
                    "Batch delete succeeded"
                );
            }
            Err(e) => {
                error!(
                    cids.count = cids.len(),
                    duration_us = duration_us,
                    error = %e,
                    "Batch delete failed"
                );
            }
        }

        result
    }

    async fn flush(&self) -> Result<()> {
        let start = Instant::now();

        let span = info_span!(
            "blockstore.flush",
            service.name = %self.service_name,
        );

        let result = self.inner.flush().instrument(span).await;

        let duration_us = start.elapsed().as_micros();
        match &result {
            Ok(_) => {
                debug!(duration_us = duration_us, "Flush succeeded");
            }
            Err(e) => {
                warn!(
                    duration_us = duration_us,
                    error = %e,
                    "Flush failed"
                );
            }
        }

        result
    }

    fn list_cids(&self) -> Result<Vec<Cid>> {
        let start = Instant::now();

        let _span = info_span!(
            "blockstore.list_cids",
            service.name = %self.service_name,
        );

        let result = self.inner.list_cids();

        let duration_us = start.elapsed().as_micros();
        match &result {
            Ok(cids) => {
                debug!(
                    cids.count = cids.len(),
                    duration_us = duration_us,
                    "List CIDs succeeded"
                );
            }
            Err(e) => {
                error!(
                    duration_us = duration_us,
                    error = %e,
                    "List CIDs failed"
                );
            }
        }

        result
    }

    fn len(&self) -> usize {
        self.inner.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::MemoryBlockStore;

    #[tokio::test]
    async fn test_otel_put_get() {
        let store = MemoryBlockStore::new();
        let traced = OtelBlockStore::new(store, "test_node".to_string());

        let block = Block::new(b"hello world".to_vec().into()).unwrap();
        let cid = block.cid();

        traced.put(&block).await.unwrap();
        let retrieved = traced.get(cid).await.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().data(), block.data());
    }

    #[tokio::test]
    async fn test_otel_has_delete() {
        let store = MemoryBlockStore::new();
        let traced = OtelBlockStore::new(store, "test_node".to_string());

        let block = Block::new(b"test data".to_vec().into()).unwrap();
        let cid = block.cid();

        traced.put(&block).await.unwrap();
        assert!(traced.has(cid).await.unwrap());

        traced.delete(cid).await.unwrap();
        assert!(!traced.has(cid).await.unwrap());
    }

    #[tokio::test]
    async fn test_otel_batch_operations() {
        let store = MemoryBlockStore::new();
        let traced = OtelBlockStore::new(store, "test_node".to_string());

        let blocks = vec![
            Block::new(b"block1".to_vec().into()).unwrap(),
            Block::new(b"block2".to_vec().into()).unwrap(),
            Block::new(b"block3".to_vec().into()).unwrap(),
        ];
        let cids: Vec<Cid> = blocks.iter().map(|b| *b.cid()).collect();

        traced.put_many(&blocks).await.unwrap();

        let has_results = traced.has_many(&cids).await.unwrap();
        assert_eq!(has_results.len(), 3);
        assert!(has_results.iter().all(|&b| b));

        let get_results = traced.get_many(&cids).await.unwrap();
        assert_eq!(get_results.len(), 3);
        assert!(get_results.iter().all(|b| b.is_some()));
    }

    #[tokio::test]
    async fn test_otel_inner_access() {
        let store = MemoryBlockStore::new();
        let traced = OtelBlockStore::new(store, "test_node".to_string());

        // Can access inner store
        let block = Block::new(b"direct access".to_vec().into()).unwrap();
        traced.inner().put(&block).await.unwrap();

        // Visible through traced wrapper
        assert!(traced.has(block.cid()).await.unwrap());
    }
}
