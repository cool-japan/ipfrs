//! Block storage implementation
//!
//! - [`BlockStoreConfig`] and [`DeduplicationStats`] are always available.
//! - [`SledBlockStore`] is only compiled when the `sled-backend` feature is enabled.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// BlockStoreConfig
// ---------------------------------------------------------------------------

/// Block store configuration
#[derive(Debug, Clone)]
pub struct BlockStoreConfig {
    /// Path to the database directory
    pub path: PathBuf,
    /// Cache size in bytes
    pub cache_size: usize,
}

impl Default for BlockStoreConfig {
    fn default() -> Self {
        Self {
            path: PathBuf::from(".ipfrs/blocks"),
            cache_size: 100 * 1024 * 1024, // 100MB
        }
    }
}

impl BlockStoreConfig {
    /// Create a configuration optimized for development
    /// - Small cache (50MB)
    /// - Stored in the system temp directory for easy cleanup
    pub fn development() -> Self {
        Self {
            path: std::env::temp_dir().join("ipfrs-dev"),
            cache_size: 50 * 1024 * 1024,
        }
    }

    /// Create a configuration optimized for production
    /// - Large cache (500MB)
    /// - Stored in standard location
    pub fn production(path: PathBuf) -> Self {
        Self {
            path,
            cache_size: 500 * 1024 * 1024,
        }
    }

    /// Create a configuration optimized for embedded devices
    /// - Minimal cache (10MB)
    /// - Configurable path
    pub fn embedded(path: PathBuf) -> Self {
        Self {
            path,
            cache_size: 10 * 1024 * 1024,
        }
    }

    /// Create a configuration optimized for testing
    /// - Minimal cache (5MB)
    /// - Temporary directory with unique name
    pub fn testing() -> Self {
        let temp_dir = std::env::temp_dir().join(format!("ipfrs-test-{}", std::process::id()));
        Self {
            path: temp_dir,
            cache_size: 5 * 1024 * 1024,
        }
    }

    /// Builder: Set the storage path
    pub fn with_path(mut self, path: PathBuf) -> Self {
        self.path = path;
        self
    }

    /// Builder: Set the cache size in MB
    pub fn with_cache_mb(mut self, cache_mb: usize) -> Self {
        self.cache_size = cache_mb * 1024 * 1024;
        self
    }

    /// Builder: Set the cache size in bytes
    pub fn with_cache_bytes(mut self, cache_bytes: usize) -> Self {
        self.cache_size = cache_bytes;
        self
    }
}

// ---------------------------------------------------------------------------
// DeduplicationStats
// ---------------------------------------------------------------------------

/// Point-in-time snapshot of deduplication statistics.
#[derive(Debug, Clone, PartialEq)]
pub struct DeduplicationStatsSnapshot {
    /// Total number of `put` calls (including deduplicated ones).
    pub total_puts: u64,
    /// Puts that were skipped because the CID already existed.
    pub deduplicated: u64,
    /// Sum of skipped block sizes in bytes.
    pub bytes_saved: u64,
    /// Fraction of puts that were deduplicated (`deduplicated / total_puts`).
    pub dedup_ratio: f64,
}

/// Lock-free, atomics-based deduplication statistics.
///
/// Wrap in `Arc<DeduplicationStats>` to share across threads without a mutex.
#[derive(Debug, Default)]
pub struct DeduplicationStats {
    /// Total put attempts.
    pub total_puts: AtomicU64,
    /// Puts skipped because the CID was already present.
    pub deduplicated: AtomicU64,
    /// Cumulative byte count of deduplicated (skipped) blocks.
    pub bytes_saved: AtomicU64,
}

impl DeduplicationStats {
    /// Create a new counter set wrapped in an `Arc`.
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Record a single put operation.
    ///
    /// * `deduplicated` — `true` when the block was skipped (CID already existed).
    /// * `bytes` — byte length of the block.
    pub fn record_put(&self, deduplicated: bool, bytes: usize) {
        self.total_puts.fetch_add(1, Ordering::Relaxed);
        if deduplicated {
            self.deduplicated.fetch_add(1, Ordering::Relaxed);
            self.bytes_saved.fetch_add(bytes as u64, Ordering::Relaxed);
        }
    }

    /// Take an instantaneous snapshot of all counters.
    pub fn snapshot(&self) -> DeduplicationStatsSnapshot {
        let total_puts = self.total_puts.load(Ordering::Relaxed);
        let deduplicated = self.deduplicated.load(Ordering::Relaxed);
        let bytes_saved = self.bytes_saved.load(Ordering::Relaxed);
        let dedup_ratio = if total_puts == 0 {
            0.0
        } else {
            deduplicated as f64 / total_puts as f64
        };
        DeduplicationStatsSnapshot {
            total_puts,
            deduplicated,
            bytes_saved,
            dedup_ratio,
        }
    }
}

// ---------------------------------------------------------------------------
// SledBlockStore — only compiled when the `sled-backend` feature is enabled
// ---------------------------------------------------------------------------

#[cfg(feature = "sled-backend")]
mod sled_store {
    use super::{BlockStoreConfig, DeduplicationStats};
    use crate::compaction::{CompactionConfig, CompactionScheduler};
    use crate::traits::BlockStore;
    use async_trait::async_trait;
    use ipfrs_core::{Block, Cid, Error, Result};
    use sled::Db;
    use std::sync::Arc;

    /// Block storage using Sled embedded database.
    ///
    /// Available only when the `sled-backend` feature is enabled (the default).
    /// Not available on wasm32 targets — use [`crate::MemoryBlockStore`] there.
    pub struct SledBlockStore {
        db: Db,
        /// Lock-free deduplication counters
        dedup_stats: Arc<DeduplicationStats>,
        /// Lock-free compaction scheduler
        compaction_scheduler: Arc<CompactionScheduler>,
    }

    impl SledBlockStore {
        /// Create a new block store with default compaction settings.
        pub fn new(config: BlockStoreConfig) -> Result<Self> {
            Self::new_with_compaction(config, CompactionConfig::default())
        }

        /// Create a new block store with a custom compaction configuration.
        pub fn new_with_compaction(
            config: BlockStoreConfig,
            compaction_config: CompactionConfig,
        ) -> Result<Self> {
            // Create parent directory if it doesn't exist
            if let Some(parent) = config.path.parent() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| Error::Storage(format!("Failed to create directory: {e}")))?;
            }

            let db = sled::Config::new()
                .path(&config.path)
                .cache_capacity(config.cache_size as u64)
                .open()
                .map_err(|e| Error::Storage(format!("Failed to open database: {e}")))?;

            Ok(Self {
                db,
                dedup_stats: DeduplicationStats::new(),
                compaction_scheduler: CompactionScheduler::new(compaction_config),
            })
        }

        // ── Deduplication ──────────────────────────────────────────────────────

        /// Write the block only when its CID is not already present.
        ///
        /// Returns `true` if the block was written, `false` if it was deduplicated
        /// (CID already existed).
        pub async fn put_if_absent(&self, block: &Block) -> Result<bool> {
            let key = block.cid().to_bytes();
            let exists = self
                .db
                .contains_key(&key)
                .map_err(|e| Error::Storage(format!("Failed to check block: {e}")))?;

            let bytes = block.data().len();
            self.dedup_stats.record_put(exists, bytes);

            if exists {
                return Ok(false);
            }

            self.db
                .insert(key, block.data().to_vec())
                .map_err(|e| Error::Storage(format!("Failed to insert block: {e}")))?;

            self.db
                .flush_async()
                .await
                .map_err(|e| Error::Storage(format!("Failed to flush: {e}")))?;

            self.compaction_scheduler.record_write(bytes);
            Ok(true)
        }

        /// Batch put with write-time deduplication.
        ///
        /// Returns `(written, deduped)`.
        pub async fn put_batch_dedup(&self, blocks: &[Block]) -> Result<(usize, usize)> {
            let mut written = 0usize;
            let mut deduped = 0usize;

            for block in blocks {
                if self.put_if_absent(block).await? {
                    written += 1;
                } else {
                    deduped += 1;
                }
            }
            Ok((written, deduped))
        }

        /// Return a shared handle to the deduplication statistics.
        pub fn dedup_stats(&self) -> Arc<DeduplicationStats> {
            Arc::clone(&self.dedup_stats)
        }

        // ── Compaction ─────────────────────────────────────────────────────────

        /// Flush the Sled WAL if the compaction scheduler determines it is time.
        ///
        /// Returns `true` when a flush was actually performed.
        pub async fn maybe_compact(&self) -> Result<bool> {
            if !self.compaction_scheduler.should_compact() {
                return Ok(false);
            }
            if !self.compaction_scheduler.mark_compaction_started() {
                // Another task won the race.
                return Ok(false);
            }
            let flush_result = self
                .db
                .flush_async()
                .await
                .map_err(|e| Error::Storage(format!("Failed to flush during compaction: {e}")));
            self.compaction_scheduler.mark_compaction_done();
            flush_result?;
            Ok(true)
        }

        /// Return a shared handle to the compaction scheduler.
        pub fn compaction_scheduler(&self) -> Arc<CompactionScheduler> {
            Arc::clone(&self.compaction_scheduler)
        }

        /// Open (or re-open) the `SledSnapshotPinRegistry` backed by this
        /// store's Sled database.
        pub fn snapshot_pin_registry(
            &self,
        ) -> ipfrs_core::Result<crate::gc::SledSnapshotPinRegistry> {
            crate::gc::SledSnapshotPinRegistry::open(&self.db)
        }
    }

    #[async_trait]
    impl BlockStore for SledBlockStore {
        /// Store a block.
        ///
        /// If the CID already exists the write is skipped (content-addressed data
        /// is immutable, so the stored bytes are guaranteed to be identical).
        async fn put(&self, block: &Block) -> Result<()> {
            let key = block.cid().to_bytes();
            let bytes = block.data().len();

            // Check for existence — skip write if already present (deduplication).
            let exists = self
                .db
                .contains_key(&key)
                .map_err(|e| Error::Storage(format!("Failed to check block: {e}")))?;

            self.dedup_stats.record_put(exists, bytes);

            if exists {
                return Ok(());
            }

            self.db
                .insert(key, block.data().to_vec())
                .map_err(|e| Error::Storage(format!("Failed to insert block: {e}")))?;

            self.db
                .flush_async()
                .await
                .map_err(|e| Error::Storage(format!("Failed to flush: {e}")))?;

            self.compaction_scheduler.record_write(bytes);
            Ok(())
        }

        /// Retrieve a block by CID
        async fn get(&self, cid: &Cid) -> Result<Option<Block>> {
            let key = cid.to_bytes();

            match self.db.get(&key) {
                Ok(Some(value)) => {
                    let data = bytes::Bytes::from(value.to_vec());
                    Ok(Some(Block::from_parts(*cid, data)))
                }
                Ok(None) => Ok(None),
                Err(e) => Err(Error::Storage(format!("Failed to get block: {e}"))),
            }
        }

        /// Check if a block exists
        async fn has(&self, cid: &Cid) -> Result<bool> {
            let key = cid.to_bytes();
            self.db
                .contains_key(&key)
                .map_err(|e| Error::Storage(format!("Failed to check block: {e}")))
        }

        /// Delete a block
        async fn delete(&self, cid: &Cid) -> Result<()> {
            let key = cid.to_bytes();
            self.db
                .remove(&key)
                .map_err(|e| Error::Storage(format!("Failed to delete block: {e}")))?;

            self.db
                .flush_async()
                .await
                .map_err(|e| Error::Storage(format!("Failed to flush: {e}")))?;

            Ok(())
        }

        /// Get the number of blocks stored
        fn len(&self) -> usize {
            self.db.len()
        }

        /// Check if the store is empty
        fn is_empty(&self) -> bool {
            self.db.is_empty()
        }

        /// Get all CIDs in the store
        fn list_cids(&self) -> Result<Vec<Cid>> {
            let mut cids = Vec::new();

            for item in self.db.iter() {
                let (key, _) = item.map_err(|e| Error::Storage(format!("Iteration error: {e}")))?;

                let cid = Cid::try_from(key.to_vec())
                    .map_err(|e| Error::Cid(format!("Failed to parse CID: {e}")))?;

                cids.push(cid);
            }

            Ok(cids)
        }

        /// Store multiple blocks atomically using Sled's batch API
        async fn put_many(&self, blocks: &[Block]) -> Result<()> {
            let mut batch = sled::Batch::default();

            for block in blocks {
                let key = block.cid().to_bytes();
                let value = block.data().to_vec();
                batch.insert(key, value);
            }

            self.db
                .apply_batch(batch)
                .map_err(|e| Error::Storage(format!("Failed to apply batch: {e}")))?;

            self.db
                .flush_async()
                .await
                .map_err(|e| Error::Storage(format!("Failed to flush: {e}")))?;

            Ok(())
        }

        /// Retrieve multiple blocks efficiently
        async fn get_many(&self, cids: &[Cid]) -> Result<Vec<Option<Block>>> {
            let mut results = Vec::with_capacity(cids.len());

            for cid in cids {
                let key = cid.to_bytes();
                match self.db.get(&key) {
                    Ok(Some(value)) => {
                        let data = bytes::Bytes::from(value.to_vec());
                        results.push(Some(Block::from_parts(*cid, data)));
                    }
                    Ok(None) => results.push(None),
                    Err(e) => return Err(Error::Storage(format!("Failed to get block: {e}"))),
                }
            }

            Ok(results)
        }

        /// Check if multiple blocks exist efficiently
        async fn has_many(&self, cids: &[Cid]) -> Result<Vec<bool>> {
            let mut results = Vec::with_capacity(cids.len());

            for cid in cids {
                let key = cid.to_bytes();
                let exists = self
                    .db
                    .contains_key(&key)
                    .map_err(|e| Error::Storage(format!("Failed to check block: {e}")))?;
                results.push(exists);
            }

            Ok(results)
        }

        /// Delete multiple blocks atomically
        async fn delete_many(&self, cids: &[Cid]) -> Result<()> {
            let mut batch = sled::Batch::default();

            for cid in cids {
                let key = cid.to_bytes();
                batch.remove(key);
            }

            self.db
                .apply_batch(batch)
                .map_err(|e| Error::Storage(format!("Failed to apply batch: {e}")))?;

            self.db
                .flush_async()
                .await
                .map_err(|e| Error::Storage(format!("Failed to flush: {e}")))?;

            Ok(())
        }

        /// Flush pending writes to disk
        async fn flush(&self) -> Result<()> {
            self.db
                .flush_async()
                .await
                .map_err(|e| Error::Storage(format!("Failed to flush: {e}")))?;
            Ok(())
        }
    }

    // ── Tests ──────────────────────────────────────────────────────────────────

    #[cfg(test)]
    mod tests {
        use super::*;
        use bytes::Bytes;
        use std::path::PathBuf;

        fn unique_test_dir(suffix: &str) -> PathBuf {
            std::env::temp_dir().join(format!(
                "ipfrs-test-blockstore-{}-{}",
                suffix,
                std::process::id()
            ))
        }

        #[tokio::test]
        async fn test_put_get_block() {
            let path = unique_test_dir("basic");
            let _ = std::fs::remove_dir_all(&path);
            let config = BlockStoreConfig {
                path: path.clone(),
                cache_size: 1024 * 1024,
            };

            let store = SledBlockStore::new(config).expect("open store");
            let data = Bytes::from("hello world");
            let block = Block::new(data.clone()).expect("create block");

            store.put(&block).await.expect("put");

            let retrieved = store.get(block.cid()).await.expect("get");
            assert!(retrieved.is_some());
            assert_eq!(retrieved.expect("block").data(), &data);

            assert!(store.has(block.cid()).await.expect("has"));

            store.delete(block.cid()).await.expect("delete");
            assert!(!store.has(block.cid()).await.expect("has after delete"));

            let _ = std::fs::remove_dir_all(&path);
        }

        #[tokio::test]
        async fn test_dedup_skip_on_existing_cid() {
            let path = unique_test_dir("dedup-skip");
            let _ = std::fs::remove_dir_all(&path);
            let store = SledBlockStore::new(BlockStoreConfig {
                path: path.clone(),
                cache_size: 1024 * 1024,
            })
            .expect("open store");

            let block = Block::new(Bytes::from("dedup-skip-data")).expect("create block");

            store.put(&block).await.expect("first put");
            assert!(store.has(block.cid()).await.expect("has after first put"));

            let snap_before = store.dedup_stats().snapshot();

            store.put(&block).await.expect("second put");

            let snap_after = store.dedup_stats().snapshot();
            assert_eq!(
                snap_after.deduplicated,
                snap_before.deduplicated + 1,
                "second put must be counted as deduplicated"
            );
            assert!(
                snap_after.bytes_saved > snap_before.bytes_saved,
                "bytes_saved must increase"
            );

            let _ = std::fs::remove_dir_all(&path);
        }

        #[tokio::test]
        async fn test_dedup_stats_ratio() {
            let path = unique_test_dir("dedup-ratio");
            let _ = std::fs::remove_dir_all(&path);
            let store = SledBlockStore::new(BlockStoreConfig {
                path: path.clone(),
                cache_size: 1024 * 1024,
            })
            .expect("open store");

            let block = Block::new(Bytes::from("ratio-block")).expect("create block");

            store.put(&block).await.expect("put 1");
            store.put(&block).await.expect("put 2");

            let snap = store.dedup_stats().snapshot();
            assert_eq!(snap.total_puts, 2);
            assert_eq!(snap.deduplicated, 1);
            let expected_ratio = 0.5_f64;
            assert!(
                (snap.dedup_ratio - expected_ratio).abs() < f64::EPSILON,
                "dedup_ratio must be 0.5, got {}",
                snap.dedup_ratio
            );

            let _ = std::fs::remove_dir_all(&path);
        }

        #[tokio::test]
        async fn test_dedup_bytes_saved() {
            let path = unique_test_dir("dedup-bytes");
            let _ = std::fs::remove_dir_all(&path);
            let store = SledBlockStore::new(BlockStoreConfig {
                path: path.clone(),
                cache_size: 1024 * 1024,
            })
            .expect("open store");

            let payload = Bytes::from(vec![42u8; 1024]);
            let block = Block::new(payload.clone()).expect("create block");

            store.put(&block).await.expect("put 1");
            assert_eq!(store.dedup_stats().snapshot().bytes_saved, 0);

            for _ in 0..3 {
                store.put(&block).await.expect("dup put");
            }

            let snap = store.dedup_stats().snapshot();
            assert_eq!(snap.deduplicated, 3);
            assert_eq!(
                snap.bytes_saved,
                3 * payload.len() as u64,
                "bytes_saved must equal 3 × block size"
            );

            let _ = std::fs::remove_dir_all(&path);
        }

        #[tokio::test]
        async fn test_dedup_put_if_absent() {
            let path = unique_test_dir("dedup-absent");
            let _ = std::fs::remove_dir_all(&path);
            let store = SledBlockStore::new(BlockStoreConfig {
                path: path.clone(),
                cache_size: 1024 * 1024,
            })
            .expect("open store");

            let block = Block::new(Bytes::from("deduplicated data")).expect("create block");

            let written = store.put_if_absent(&block).await.expect("put_if_absent 1");
            assert!(written, "first write must return true");
            assert!(store.has(block.cid()).await.expect("has"));

            let written_again = store.put_if_absent(&block).await.expect("put_if_absent 2");
            assert!(!written_again, "duplicate write must return false");

            let snap = store.dedup_stats().snapshot();
            assert_eq!(snap.deduplicated, 1);
            assert!(snap.bytes_saved > 0);

            let _ = std::fs::remove_dir_all(&path);
        }

        #[tokio::test]
        async fn test_batch_dedup_stats() {
            let path = unique_test_dir("batch-dedup");
            let _ = std::fs::remove_dir_all(&path);
            let store = SledBlockStore::new(BlockStoreConfig {
                path: path.clone(),
                cache_size: 1024 * 1024,
            })
            .expect("open store");

            let block_a = Block::new(Bytes::from("alpha block")).expect("block a");
            let block_b = Block::new(Bytes::from("beta block")).expect("block b");

            store.put(&block_a).await.expect("pre-store block_a");

            let (written, deduped) = store
                .put_batch_dedup(&[block_a.clone(), block_b.clone()])
                .await
                .expect("put_batch_dedup");

            assert_eq!(written, 1, "only block_b should be written");
            assert_eq!(deduped, 1, "block_a should be dedup'd");
            assert!(store.has(block_b.cid()).await.expect("has block_b"));

            let _ = std::fs::remove_dir_all(&path);
        }

        #[tokio::test]
        async fn test_maybe_compact_fires_when_due() {
            let path = unique_test_dir("compact");
            let _ = std::fs::remove_dir_all(&path);

            let compaction_config = CompactionConfig {
                idle_threshold: std::time::Duration::from_secs(0),
                min_interval: std::time::Duration::from_secs(0),
                max_bytes_since_compact: 1,
            };

            let store = SledBlockStore::new_with_compaction(
                BlockStoreConfig {
                    path: path.clone(),
                    cache_size: 1024 * 1024,
                },
                compaction_config,
            )
            .expect("open store");

            let block = Block::new(Bytes::from("compact-me")).expect("block");
            store.put(&block).await.expect("put");

            let compacted = store.maybe_compact().await.expect("maybe_compact");
            assert!(compacted, "should have compacted with near-zero thresholds");
            assert_eq!(store.compaction_scheduler().compaction_count(), 1);

            let _ = std::fs::remove_dir_all(&path);
        }
    }
}

#[cfg(feature = "sled-backend")]
pub use sled_store::SledBlockStore;
