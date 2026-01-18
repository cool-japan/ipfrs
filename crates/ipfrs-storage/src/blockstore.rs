//! Block storage implementation using Sled

use crate::traits::BlockStore;
use async_trait::async_trait;
use ipfrs_core::{Block, Cid, Error, Result};
use sled::Db;
use std::path::PathBuf;

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
    /// - Stored in /tmp for easy cleanup
    pub fn development() -> Self {
        Self {
            path: PathBuf::from("/tmp/ipfrs-dev"),
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

/// Block storage using Sled embedded database
pub struct SledBlockStore {
    db: Db,
}

impl SledBlockStore {
    /// Create a new block store
    pub fn new(config: BlockStoreConfig) -> Result<Self> {
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

        Ok(Self { db })
    }
}

#[async_trait]
impl BlockStore for SledBlockStore {
    /// Store a block
    async fn put(&self, block: &Block) -> Result<()> {
        let key = block.cid().to_bytes();
        let value = block.data().to_vec();

        self.db
            .insert(key, value)
            .map_err(|e| Error::Storage(format!("Failed to insert block: {e}")))?;

        self.db
            .flush_async()
            .await
            .map_err(|e| Error::Storage(format!("Failed to flush: {e}")))?;

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

            // Parse CID from key bytes
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

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;

    #[tokio::test]
    async fn test_put_get_block() {
        let config = BlockStoreConfig {
            path: PathBuf::from("/tmp/ipfrs-test-blockstore"),
            cache_size: 1024 * 1024,
        };

        // Clean up from previous test
        let _ = std::fs::remove_dir_all(&config.path);

        let store = SledBlockStore::new(config).unwrap();
        let data = Bytes::from("hello world");
        let block = Block::new(data.clone()).unwrap();

        // Put block
        store.put(&block).await.unwrap();

        // Get block
        let retrieved = store.get(block.cid()).await.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().data(), &data);

        // Check has
        assert!(store.has(block.cid()).await.unwrap());

        // Delete block
        store.delete(block.cid()).await.unwrap();
        assert!(!store.has(block.cid()).await.unwrap());
    }
}
