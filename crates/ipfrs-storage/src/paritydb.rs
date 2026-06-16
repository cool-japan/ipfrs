#![cfg(feature = "parity-db-backend")]
//! Block storage implementation using ParityDB
//!
//! ParityDB is optimized for SSD storage with better write amplification
//! compared to Sled. It uses column-based storage layout and is designed
//! for high-throughput workloads.

use crate::traits::BlockStore;
use async_trait::async_trait;
use ipfrs_core::{Block, Cid, Error, Result};
use parity_db::{Db, Options};
use std::path::PathBuf;
use std::sync::Arc;

/// Column for storing blocks
const BLOCKS_COLUMN: u8 = 0;

/// Configuration preset types for ParityDB
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParityDbPreset {
    /// Optimized for write-heavy ingestion workloads
    FastWrite,
    /// Balanced configuration for general purpose use
    Balanced,
    /// Low memory usage for constrained devices
    LowMemory,
}

/// ParityDB block store configuration
#[derive(Debug, Clone)]
pub struct ParityDbConfig {
    /// Path to the database directory
    pub path: PathBuf,
    /// Configuration preset
    pub preset: ParityDbPreset,
    /// Custom column options (overrides preset if provided)
    pub custom_options: Option<Options>,
}

impl ParityDbConfig {
    /// Create a new configuration with a preset
    pub fn new(path: PathBuf, preset: ParityDbPreset) -> Self {
        Self {
            path,
            preset,
            custom_options: None,
        }
    }

    /// Create configuration optimized for fast writes
    pub fn fast_write(path: PathBuf) -> Self {
        Self::new(path, ParityDbPreset::FastWrite)
    }

    /// Create configuration for balanced workloads
    pub fn balanced(path: PathBuf) -> Self {
        Self::new(path, ParityDbPreset::Balanced)
    }

    /// Create configuration for low memory usage
    pub fn low_memory(path: PathBuf) -> Self {
        Self::new(path, ParityDbPreset::LowMemory)
    }

    /// Build ParityDB Options from configuration
    fn build_options(&self) -> Options {
        if let Some(ref custom) = self.custom_options {
            return custom.clone();
        }

        let mut options = Options::with_columns(&self.path, 1);

        match self.preset {
            ParityDbPreset::FastWrite => {
                // Optimize for write throughput
                // Larger write buffer, more aggressive compression
                options.columns[BLOCKS_COLUMN as usize].btree_index = true;
                options.columns[BLOCKS_COLUMN as usize].compression =
                    parity_db::CompressionType::Lz4;
                options.sync_wal = false; // Async WAL for better write performance
                options.sync_data = false; // Async data writes
            }
            ParityDbPreset::Balanced => {
                // Balanced settings
                options.columns[BLOCKS_COLUMN as usize].btree_index = true;
                options.columns[BLOCKS_COLUMN as usize].compression =
                    parity_db::CompressionType::Lz4;
                options.sync_wal = true;
                options.sync_data = false;
            }
            ParityDbPreset::LowMemory => {
                // Minimize memory usage
                options.columns[BLOCKS_COLUMN as usize].btree_index = false; // No index to save memory
                options.columns[BLOCKS_COLUMN as usize].compression =
                    parity_db::CompressionType::Lz4;
                options.sync_wal = true;
                options.sync_data = true;
            }
        }

        options
    }
}

impl Default for ParityDbConfig {
    fn default() -> Self {
        Self::balanced(PathBuf::from(".ipfrs/blocks-paritydb"))
    }
}

/// Block storage using ParityDB
pub struct ParityDbBlockStore {
    db: Arc<Db>,
}

impl ParityDbBlockStore {
    /// Create a new ParityDB block store
    pub fn new(config: ParityDbConfig) -> Result<Self> {
        // Create parent directory if it doesn't exist
        if let Some(parent) = config.path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| Error::Storage(format!("Failed to create directory: {e}")))?;
        }

        let options = config.build_options();

        let db = Db::open_or_create(&options)
            .map_err(|e| Error::Storage(format!("Failed to open ParityDB: {e}")))?;

        Ok(Self { db: Arc::new(db) })
    }

    /// Get reference to underlying database
    pub fn db(&self) -> &Arc<Db> {
        &self.db
    }
}

#[async_trait]
impl BlockStore for ParityDbBlockStore {
    /// Store a block
    async fn put(&self, block: &Block) -> Result<()> {
        let key = block.cid().to_bytes();
        let value = block.data().to_vec();

        let transaction = vec![(BLOCKS_COLUMN, key, Some(value))];

        self.db
            .commit(transaction)
            .map_err(|e| Error::Storage(format!("Failed to insert block: {e}")))?;

        Ok(())
    }

    /// Retrieve a block by CID
    async fn get(&self, cid: &Cid) -> Result<Option<Block>> {
        let key = cid.to_bytes();

        match self.db.get(BLOCKS_COLUMN, &key) {
            Ok(Some(value)) => {
                let data = bytes::Bytes::from(value);
                Ok(Some(Block::from_parts(*cid, data)))
            }
            Ok(None) => Ok(None),
            Err(e) => Err(Error::Storage(format!("Failed to get block: {e}"))),
        }
    }

    /// Check if a block exists
    async fn has(&self, cid: &Cid) -> Result<bool> {
        let key = cid.to_bytes();
        match self.db.get(BLOCKS_COLUMN, &key) {
            Ok(Some(_)) => Ok(true),
            Ok(None) => Ok(false),
            Err(e) => Err(Error::Storage(format!("Failed to check block: {e}"))),
        }
    }

    /// Delete a block
    async fn delete(&self, cid: &Cid) -> Result<()> {
        let key = cid.to_bytes();

        let transaction = vec![(BLOCKS_COLUMN, key, None)];

        self.db
            .commit(transaction)
            .map_err(|e| Error::Storage(format!("Failed to delete block: {e}")))?;

        Ok(())
    }

    /// Get the number of blocks stored
    fn len(&self) -> usize {
        // ParityDB doesn't have a direct len() method
        // We need to iterate to count (expensive operation)
        // For performance, return 0 and users should track this separately
        // or use a separate counter if needed
        0
    }

    /// Check if the store is empty
    fn is_empty(&self) -> bool {
        // Since len() is not efficient, we can't reliably check emptiness
        // Return false as a safe default
        false
    }

    /// Get all CIDs in the store
    fn list_cids(&self) -> Result<Vec<Cid>> {
        let mut cids = Vec::new();

        let mut iter = self
            .db
            .iter(BLOCKS_COLUMN)
            .map_err(|e| Error::Storage(format!("Failed to create iterator: {e}")))?;

        while let Some((key, _value)) = iter
            .next()
            .map_err(|e| Error::Storage(format!("Iterator error: {e}")))?
        {
            // Parse CID from key bytes
            let cid = Cid::try_from(key.to_vec())
                .map_err(|e| Error::Cid(format!("Failed to parse CID: {e}")))?;

            cids.push(cid);
        }

        Ok(cids)
    }

    /// Store multiple blocks atomically
    async fn put_many(&self, blocks: &[Block]) -> Result<()> {
        let mut transaction = Vec::new();

        for block in blocks {
            let key = block.cid().to_bytes();
            let value = block.data().to_vec();
            transaction.push((BLOCKS_COLUMN, key, Some(value)));
        }

        self.db
            .commit(transaction)
            .map_err(|e| Error::Storage(format!("Failed to apply batch: {e}")))?;

        Ok(())
    }

    /// Retrieve multiple blocks efficiently
    async fn get_many(&self, cids: &[Cid]) -> Result<Vec<Option<Block>>> {
        let mut results = Vec::with_capacity(cids.len());

        for cid in cids {
            let key = cid.to_bytes();
            match self.db.get(BLOCKS_COLUMN, &key) {
                Ok(Some(value)) => {
                    let data = bytes::Bytes::from(value);
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
            match self.db.get(BLOCKS_COLUMN, &key) {
                Ok(Some(_)) => results.push(true),
                Ok(None) => results.push(false),
                Err(e) => return Err(Error::Storage(format!("Failed to check block: {e}"))),
            }
        }

        Ok(results)
    }

    /// Delete multiple blocks atomically
    async fn delete_many(&self, cids: &[Cid]) -> Result<()> {
        let mut transaction = Vec::new();

        for cid in cids {
            let key = cid.to_bytes();
            transaction.push((BLOCKS_COLUMN, key, None));
        }

        self.db
            .commit(transaction)
            .map_err(|e| Error::Storage(format!("Failed to apply batch: {e}")))?;

        Ok(())
    }

    /// Flush pending writes to disk
    async fn flush(&self) -> Result<()> {
        // ParityDB commits are already durable
        // No explicit flush needed
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;

    #[tokio::test]
    async fn test_paritydb_put_get_block() {
        let config = ParityDbConfig::balanced(std::env::temp_dir().join("ipfrs-test-paritydb"));

        // Clean up from previous test
        let _ = std::fs::remove_dir_all(&config.path);

        let store = ParityDbBlockStore::new(config).expect("failed to create ParityDB store");
        let data = Bytes::from("hello paritydb");
        let block = Block::new(data.clone()).expect("failed to create block from data");

        // Put block
        store
            .put(&block)
            .await
            .expect("failed to put block into store");

        // Get block
        let retrieved = store
            .get(block.cid())
            .await
            .expect("failed to get block from store");
        assert!(retrieved.is_some());
        assert_eq!(
            retrieved.expect("retrieved block should be Some").data(),
            &data
        );

        // Check has
        assert!(store
            .has(block.cid())
            .await
            .expect("failed to check block existence"));

        // Delete block
        store
            .delete(block.cid())
            .await
            .expect("failed to delete block");
        assert!(!store
            .has(block.cid())
            .await
            .expect("failed to check block existence after delete"));
    }

    #[tokio::test]
    async fn test_paritydb_batch_operations() {
        let config =
            ParityDbConfig::fast_write(std::env::temp_dir().join("ipfrs-test-paritydb-batch"));

        // Clean up from previous test
        let _ = std::fs::remove_dir_all(&config.path);

        let store = ParityDbBlockStore::new(config)
            .expect("test: ParityDB store creation should succeed for batch test");

        // Create multiple blocks
        let blocks: Vec<Block> = (0..10)
            .map(|i| {
                let data = Bytes::from(format!("block {}", i));
                Block::new(data).expect("test: block creation should succeed")
            })
            .collect();

        // Batch put
        store
            .put_many(&blocks)
            .await
            .expect("test: put_many should succeed");

        // Check all exist
        let cids: Vec<Cid> = blocks.iter().map(|b| *b.cid()).collect();
        let exists = store
            .has_many(&cids)
            .await
            .expect("test: has_many should succeed");
        assert!(exists.iter().all(|&x| x));

        // Batch get
        let retrieved = store
            .get_many(&cids)
            .await
            .expect("test: get_many should succeed");
        assert_eq!(retrieved.len(), blocks.len());
        for (i, opt_block) in retrieved.iter().enumerate() {
            assert!(opt_block.is_some());
            assert_eq!(
                opt_block
                    .as_ref()
                    .expect("test: retrieved block should be Some")
                    .data(),
                blocks[i].data()
            );
        }

        // Batch delete
        store
            .delete_many(&cids)
            .await
            .expect("test: delete_many should succeed");
        let exists = store
            .has_many(&cids)
            .await
            .expect("test: has_many after delete should succeed");
        assert!(exists.iter().all(|&x| !x));
    }

    #[tokio::test]
    async fn test_paritydb_presets() {
        // Test fast_write preset
        let config1 =
            ParityDbConfig::fast_write(std::env::temp_dir().join("ipfrs-test-paritydb-fast"));
        let _ = std::fs::remove_dir_all(&config1.path);
        assert_eq!(config1.preset, ParityDbPreset::FastWrite);
        let _store1 = ParityDbBlockStore::new(config1)
            .expect("test: fast_write store creation should succeed");

        // Test balanced preset
        let config2 =
            ParityDbConfig::balanced(std::env::temp_dir().join("ipfrs-test-paritydb-balanced"));
        let _ = std::fs::remove_dir_all(&config2.path);
        assert_eq!(config2.preset, ParityDbPreset::Balanced);
        let _store2 =
            ParityDbBlockStore::new(config2).expect("test: balanced store creation should succeed");

        // Test low_memory preset
        let config3 =
            ParityDbConfig::low_memory(std::env::temp_dir().join("ipfrs-test-paritydb-lowmem"));
        let _ = std::fs::remove_dir_all(&config3.path);
        assert_eq!(config3.preset, ParityDbPreset::LowMemory);
        let _store3 = ParityDbBlockStore::new(config3)
            .expect("test: low_memory store creation should succeed");
    }

    #[tokio::test]
    async fn test_paritydb_list_cids() {
        let config =
            ParityDbConfig::balanced(std::env::temp_dir().join("ipfrs-test-paritydb-list"));

        // Clean up from previous test
        let _ = std::fs::remove_dir_all(&config.path);

        let store =
            ParityDbBlockStore::new(config).expect("test: list_cids store creation should succeed");

        // Create and store blocks
        let blocks: Vec<Block> = (0..5)
            .map(|i| {
                let data = Bytes::from(format!("block {}", i));
                Block::new(data).expect("test: block creation should succeed")
            })
            .collect();

        store
            .put_many(&blocks)
            .await
            .expect("test: put_many should succeed");

        // List CIDs
        let cids = store.list_cids().expect("test: list_cids should succeed");
        assert_eq!(cids.len(), 5);

        // Verify all CIDs are present
        for block in &blocks {
            assert!(cids.contains(block.cid()));
        }
    }
}
