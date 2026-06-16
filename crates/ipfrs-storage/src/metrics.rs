//! Storage metrics and observability
//!
//! This module provides comprehensive metrics tracking for storage operations,
//! enabling production monitoring and performance analysis.

use crate::traits::BlockStore;
use async_trait::async_trait;
use ipfrs_core::{Block, Cid, Result};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Storage operation metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StorageMetrics {
    /// Total number of put operations
    pub put_count: u64,
    /// Total number of get operations
    pub get_count: u64,
    /// Total number of has operations
    pub has_count: u64,
    /// Total number of delete operations
    pub delete_count: u64,
    /// Total number of successful gets (cache hits + disk hits)
    pub get_hits: u64,
    /// Total number of failed gets (not found)
    pub get_misses: u64,
    /// Total bytes written
    pub bytes_written: u64,
    /// Total bytes read
    pub bytes_read: u64,
    /// Average put latency in microseconds
    pub avg_put_latency_us: u64,
    /// Average get latency in microseconds
    pub avg_get_latency_us: u64,
    /// Average has latency in microseconds
    pub avg_has_latency_us: u64,
    /// Peak put latency in microseconds
    pub peak_put_latency_us: u64,
    /// Peak get latency in microseconds
    pub peak_get_latency_us: u64,
    /// Number of errors encountered
    pub error_count: u64,
    /// Total number of batch operations (put_many, get_many, etc.)
    pub batch_op_count: u64,
    /// Total number of items in batch operations
    pub batch_items_count: u64,
    /// Average batch size (items per batch)
    pub avg_batch_size: u64,
}

impl StorageMetrics {
    /// Calculate cache hit rate (0.0 to 1.0)
    pub fn cache_hit_rate(&self) -> f64 {
        let total = self.get_hits + self.get_misses;
        if total == 0 {
            0.0
        } else {
            self.get_hits as f64 / total as f64
        }
    }

    /// Calculate average operation latency
    pub fn avg_operation_latency_us(&self) -> u64 {
        let total_ops = self.put_count + self.get_count + self.has_count;
        let total_latency = (self.put_count * self.avg_put_latency_us)
            + (self.get_count * self.avg_get_latency_us)
            + (self.has_count * self.avg_has_latency_us);
        total_latency.checked_div(total_ops).unwrap_or(0)
    }

    /// Calculate throughput in operations per second
    pub fn ops_per_second(&self, duration: Duration) -> f64 {
        let total_ops = self.put_count + self.get_count + self.has_count + self.delete_count;
        let seconds = duration.as_secs_f64();
        if seconds > 0.0 {
            total_ops as f64 / seconds
        } else {
            0.0
        }
    }

    /// Calculate batch efficiency (percentage of operations that were batched)
    pub fn batch_efficiency(&self) -> f64 {
        let total_ops = self.put_count + self.get_count + self.has_count + self.delete_count;
        if total_ops == 0 {
            0.0
        } else {
            self.batch_items_count as f64 / total_ops as f64
        }
    }

    /// Calculate write throughput in bytes per second
    pub fn write_throughput_bps(&self, duration: Duration) -> f64 {
        let seconds = duration.as_secs_f64();
        if seconds > 0.0 {
            self.bytes_written as f64 / seconds
        } else {
            0.0
        }
    }

    /// Calculate read throughput in bytes per second
    pub fn read_throughput_bps(&self, duration: Duration) -> f64 {
        let seconds = duration.as_secs_f64();
        if seconds > 0.0 {
            self.bytes_read as f64 / seconds
        } else {
            0.0
        }
    }
}

/// Internal metrics collector
struct MetricsCollector {
    put_count: AtomicU64,
    get_count: AtomicU64,
    has_count: AtomicU64,
    delete_count: AtomicU64,
    get_hits: AtomicU64,
    get_misses: AtomicU64,
    bytes_written: AtomicU64,
    bytes_read: AtomicU64,
    put_latency_sum: AtomicU64,
    get_latency_sum: AtomicU64,
    has_latency_sum: AtomicU64,
    peak_put_latency: AtomicU64,
    peak_get_latency: AtomicU64,
    error_count: AtomicU64,
    batch_op_count: AtomicU64,
    batch_items_count: AtomicU64,
    start_time: Instant,
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self {
            put_count: AtomicU64::new(0),
            get_count: AtomicU64::new(0),
            has_count: AtomicU64::new(0),
            delete_count: AtomicU64::new(0),
            get_hits: AtomicU64::new(0),
            get_misses: AtomicU64::new(0),
            bytes_written: AtomicU64::new(0),
            bytes_read: AtomicU64::new(0),
            put_latency_sum: AtomicU64::new(0),
            get_latency_sum: AtomicU64::new(0),
            has_latency_sum: AtomicU64::new(0),
            peak_put_latency: AtomicU64::new(0),
            peak_get_latency: AtomicU64::new(0),
            error_count: AtomicU64::new(0),
            batch_op_count: AtomicU64::new(0),
            batch_items_count: AtomicU64::new(0),
            start_time: Instant::now(),
        }
    }
}

impl MetricsCollector {
    fn snapshot(&self) -> StorageMetrics {
        let put_count = self.put_count.load(Ordering::Relaxed);
        let get_count = self.get_count.load(Ordering::Relaxed);
        let has_count = self.has_count.load(Ordering::Relaxed);
        let batch_op_count = self.batch_op_count.load(Ordering::Relaxed);
        let batch_items_count = self.batch_items_count.load(Ordering::Relaxed);

        StorageMetrics {
            put_count,
            get_count,
            has_count,
            delete_count: self.delete_count.load(Ordering::Relaxed),
            get_hits: self.get_hits.load(Ordering::Relaxed),
            get_misses: self.get_misses.load(Ordering::Relaxed),
            bytes_written: self.bytes_written.load(Ordering::Relaxed),
            bytes_read: self.bytes_read.load(Ordering::Relaxed),
            avg_put_latency_us: self
                .put_latency_sum
                .load(Ordering::Relaxed)
                .checked_div(put_count)
                .unwrap_or(0),
            avg_get_latency_us: self
                .get_latency_sum
                .load(Ordering::Relaxed)
                .checked_div(get_count)
                .unwrap_or(0),
            avg_has_latency_us: self
                .has_latency_sum
                .load(Ordering::Relaxed)
                .checked_div(has_count)
                .unwrap_or(0),
            peak_put_latency_us: self.peak_put_latency.load(Ordering::Relaxed),
            peak_get_latency_us: self.peak_get_latency.load(Ordering::Relaxed),
            error_count: self.error_count.load(Ordering::Relaxed),
            batch_op_count,
            batch_items_count,
            avg_batch_size: batch_items_count.checked_div(batch_op_count).unwrap_or(0),
        }
    }

    fn record_put(&self, bytes: u64, latency_us: u64) {
        self.put_count.fetch_add(1, Ordering::Relaxed);
        self.bytes_written.fetch_add(bytes, Ordering::Relaxed);
        self.put_latency_sum
            .fetch_add(latency_us, Ordering::Relaxed);

        let mut current_peak = self.peak_put_latency.load(Ordering::Relaxed);
        while latency_us > current_peak {
            match self.peak_put_latency.compare_exchange_weak(
                current_peak,
                latency_us,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(x) => current_peak = x,
            }
        }
    }

    fn record_get(&self, bytes: Option<u64>, latency_us: u64) {
        self.get_count.fetch_add(1, Ordering::Relaxed);

        if let Some(bytes) = bytes {
            self.get_hits.fetch_add(1, Ordering::Relaxed);
            self.bytes_read.fetch_add(bytes, Ordering::Relaxed);
        } else {
            self.get_misses.fetch_add(1, Ordering::Relaxed);
        }

        self.get_latency_sum
            .fetch_add(latency_us, Ordering::Relaxed);

        let mut current_peak = self.peak_get_latency.load(Ordering::Relaxed);
        while latency_us > current_peak {
            match self.peak_get_latency.compare_exchange_weak(
                current_peak,
                latency_us,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(x) => current_peak = x,
            }
        }
    }

    fn record_has(&self, latency_us: u64) {
        self.has_count.fetch_add(1, Ordering::Relaxed);
        self.has_latency_sum
            .fetch_add(latency_us, Ordering::Relaxed);
    }

    fn record_delete(&self) {
        self.delete_count.fetch_add(1, Ordering::Relaxed);
    }

    fn record_error(&self) {
        self.error_count.fetch_add(1, Ordering::Relaxed);
    }

    fn record_batch(&self, batch_size: usize) {
        self.batch_op_count.fetch_add(1, Ordering::Relaxed);
        self.batch_items_count
            .fetch_add(batch_size as u64, Ordering::Relaxed);
    }

    fn uptime(&self) -> Duration {
        self.start_time.elapsed()
    }

    fn reset(&self) {
        self.put_count.store(0, Ordering::Relaxed);
        self.get_count.store(0, Ordering::Relaxed);
        self.has_count.store(0, Ordering::Relaxed);
        self.delete_count.store(0, Ordering::Relaxed);
        self.get_hits.store(0, Ordering::Relaxed);
        self.get_misses.store(0, Ordering::Relaxed);
        self.bytes_written.store(0, Ordering::Relaxed);
        self.bytes_read.store(0, Ordering::Relaxed);
        self.put_latency_sum.store(0, Ordering::Relaxed);
        self.get_latency_sum.store(0, Ordering::Relaxed);
        self.has_latency_sum.store(0, Ordering::Relaxed);
        self.peak_put_latency.store(0, Ordering::Relaxed);
        self.peak_get_latency.store(0, Ordering::Relaxed);
        self.error_count.store(0, Ordering::Relaxed);
        self.batch_op_count.store(0, Ordering::Relaxed);
        self.batch_items_count.store(0, Ordering::Relaxed);
    }
}

/// Block store with metrics tracking
pub struct MetricsBlockStore<S: BlockStore> {
    inner: S,
    metrics: Arc<MetricsCollector>,
}

impl<S: BlockStore> MetricsBlockStore<S> {
    /// Create a new metrics-enabled block store
    pub fn new(inner: S) -> Self {
        Self {
            inner,
            metrics: Arc::new(MetricsCollector::default()),
        }
    }

    /// Get current metrics snapshot
    pub fn metrics(&self) -> StorageMetrics {
        self.metrics.snapshot()
    }

    /// Get uptime duration
    pub fn uptime(&self) -> Duration {
        self.metrics.uptime()
    }

    /// Reset all metrics counters
    ///
    /// This resets all counters to zero while keeping the store running.
    /// The start time is not reset, so uptime() will continue from the original start.
    pub fn reset_metrics(&self) {
        self.metrics.reset();
    }

    /// Get the inner store
    pub fn inner(&self) -> &S {
        &self.inner
    }

    /// Consume this store and return the inner store
    pub fn into_inner(self) -> S {
        self.inner
    }
}

#[async_trait]
impl<S: BlockStore> BlockStore for MetricsBlockStore<S> {
    async fn put(&self, block: &Block) -> Result<()> {
        let start = Instant::now();
        let result = self.inner.put(block).await;
        let latency_us = start.elapsed().as_micros() as u64;

        match &result {
            Ok(_) => {
                self.metrics
                    .record_put(block.data().len() as u64, latency_us);
            }
            Err(_) => {
                self.metrics.record_error();
            }
        }

        result
    }

    async fn put_many(&self, blocks: &[Block]) -> Result<()> {
        let start = Instant::now();
        let result = self.inner.put_many(blocks).await;
        let latency_us = start.elapsed().as_micros() as u64;

        match &result {
            Ok(_) => {
                // Record batch operation
                self.metrics.record_batch(blocks.len());
                // Record as individual puts for metrics
                let avg_latency = latency_us / blocks.len().max(1) as u64;
                for block in blocks {
                    self.metrics
                        .record_put(block.data().len() as u64, avg_latency);
                }
            }
            Err(_) => {
                self.metrics.record_error();
            }
        }

        result
    }

    async fn get(&self, cid: &Cid) -> Result<Option<Block>> {
        let start = Instant::now();
        let result = self.inner.get(cid).await;
        let latency_us = start.elapsed().as_micros() as u64;

        match &result {
            Ok(Some(block)) => {
                self.metrics
                    .record_get(Some(block.data().len() as u64), latency_us);
            }
            Ok(None) => {
                self.metrics.record_get(None, latency_us);
            }
            Err(_) => {
                self.metrics.record_error();
            }
        }

        result
    }

    async fn get_many(&self, cids: &[Cid]) -> Result<Vec<Option<Block>>> {
        let start = Instant::now();
        let result = self.inner.get_many(cids).await;
        let latency_us = start.elapsed().as_micros() as u64;

        match &result {
            Ok(blocks) => {
                // Record batch operation
                self.metrics.record_batch(blocks.len());
                let avg_latency = latency_us / blocks.len().max(1) as u64;
                for block in blocks {
                    match block {
                        Some(b) => {
                            self.metrics
                                .record_get(Some(b.data().len() as u64), avg_latency);
                        }
                        None => {
                            self.metrics.record_get(None, avg_latency);
                        }
                    }
                }
            }
            Err(_) => {
                self.metrics.record_error();
            }
        }

        result
    }

    async fn has(&self, cid: &Cid) -> Result<bool> {
        let start = Instant::now();
        let result = self.inner.has(cid).await;
        let latency_us = start.elapsed().as_micros() as u64;

        match &result {
            Ok(_) => {
                self.metrics.record_has(latency_us);
            }
            Err(_) => {
                self.metrics.record_error();
            }
        }

        result
    }

    async fn has_many(&self, cids: &[Cid]) -> Result<Vec<bool>> {
        let start = Instant::now();
        let result = self.inner.has_many(cids).await;
        let latency_us = start.elapsed().as_micros() as u64;

        match &result {
            Ok(results) => {
                // Record batch operation
                self.metrics.record_batch(results.len());
                let avg_latency = latency_us / results.len().max(1) as u64;
                for _ in results {
                    self.metrics.record_has(avg_latency);
                }
            }
            Err(_) => {
                self.metrics.record_error();
            }
        }

        result
    }

    async fn delete(&self, cid: &Cid) -> Result<()> {
        let result = self.inner.delete(cid).await;

        match &result {
            Ok(_) => {
                self.metrics.record_delete();
            }
            Err(_) => {
                self.metrics.record_error();
            }
        }

        result
    }

    async fn delete_many(&self, cids: &[Cid]) -> Result<()> {
        let result = self.inner.delete_many(cids).await;

        match &result {
            Ok(_) => {
                // Record batch operation
                self.metrics.record_batch(cids.len());
                for _ in cids {
                    self.metrics.record_delete();
                }
            }
            Err(_) => {
                self.metrics.record_error();
            }
        }

        result
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
        self.inner.close().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MemoryBlockStore;
    use bytes::Bytes;

    #[tokio::test]
    async fn test_metrics_tracking() {
        let store = MemoryBlockStore::new();
        let metrics_store = MetricsBlockStore::new(store);

        // Put a block
        let block = Block::new(Bytes::from("test data")).unwrap();
        metrics_store.put(&block).await.unwrap();

        let metrics = metrics_store.metrics();
        assert_eq!(metrics.put_count, 1);
        assert_eq!(metrics.bytes_written, 9); // "test data" is 9 bytes

        // Get the block
        let retrieved = metrics_store.get(block.cid()).await.unwrap();
        assert!(retrieved.is_some());

        let metrics = metrics_store.metrics();
        assert_eq!(metrics.get_count, 1);
        assert_eq!(metrics.get_hits, 1);
        assert_eq!(metrics.get_misses, 0);
        assert_eq!(metrics.bytes_read, 9);

        // Check cache hit rate
        assert_eq!(metrics.cache_hit_rate(), 1.0);
    }

    #[tokio::test]
    async fn test_metrics_cache_miss() {
        let store = MemoryBlockStore::new();
        let metrics_store = MetricsBlockStore::new(store);

        // Try to get non-existent block
        let fake_block = Block::new(Bytes::from("fake")).unwrap();
        let result = metrics_store.get(fake_block.cid()).await.unwrap();
        assert!(result.is_none());

        let metrics = metrics_store.metrics();
        assert_eq!(metrics.get_count, 1);
        assert_eq!(metrics.get_hits, 0);
        assert_eq!(metrics.get_misses, 1);
        assert_eq!(metrics.cache_hit_rate(), 0.0);
    }

    #[tokio::test]
    async fn test_metrics_latency_tracking() {
        let store = MemoryBlockStore::new();
        let metrics_store = MetricsBlockStore::new(store);

        // Put some blocks with small delays to ensure measurable latency
        for i in 0..5 {
            let block = Block::new(Bytes::from(format!("block {}", i))).unwrap();
            // Add small delay to ensure latency is measurable in microseconds
            tokio::time::sleep(std::time::Duration::from_micros(10)).await;
            metrics_store.put(&block).await.unwrap();
        }

        let metrics = metrics_store.metrics();
        assert_eq!(metrics.put_count, 5);
        assert!(metrics.avg_put_latency_us > 0);
        assert!(metrics.peak_put_latency_us > 0);
    }

    #[test]
    fn test_storage_metrics_calculations() {
        let metrics = StorageMetrics {
            put_count: 100,
            get_count: 200,
            has_count: 50,
            delete_count: 10,
            get_hits: 180,
            get_misses: 20,
            bytes_written: 10000,
            bytes_read: 18000,
            avg_put_latency_us: 100,
            avg_get_latency_us: 50,
            avg_has_latency_us: 10,
            peak_put_latency_us: 500,
            peak_get_latency_us: 200,
            error_count: 5,
            batch_op_count: 10,
            batch_items_count: 50,
            avg_batch_size: 5,
        };

        // Test cache hit rate
        assert_eq!(metrics.cache_hit_rate(), 0.9); // 180/200 = 0.9

        // Test average operation latency
        let avg_latency = metrics.avg_operation_latency_us();
        let expected = (100 * 100 + 200 * 50 + 50 * 10) / 350;
        assert_eq!(avg_latency, expected);

        // Test ops per second
        let duration = Duration::from_secs(10);
        let ops_per_sec = metrics.ops_per_second(duration);
        assert_eq!(ops_per_sec, 36.0); // (100 + 200 + 50 + 10) / 10 = 36
    }
}
