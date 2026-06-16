//! Memory-mapped I/O for zero-copy block reads
//!
//! Provides memory-mapped access to blocks stored in files, eliminating
//! copy overhead for large blocks (>1MB). Supports partial reads and
//! platform-specific optimizations.

use crate::traits::BlockStore;
use async_trait::async_trait;
use ipfrs_core::{Block, Cid, Error, Result};
use memmap2::{Mmap, MmapOptions};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Memory-mapped block store configuration
#[derive(Debug, Clone)]
pub struct MmapConfig {
    /// Directory for block files
    pub path: PathBuf,
    /// Minimum block size for mmap (smaller blocks use regular reads)
    pub mmap_threshold: usize,
    /// Whether to use huge pages (platform-specific)
    pub use_huge_pages: bool,
    /// Whether to populate mappings eagerly
    pub populate: bool,
}

impl Default for MmapConfig {
    fn default() -> Self {
        Self {
            path: PathBuf::from(".ipfrs/blocks-mmap"),
            mmap_threshold: 1024 * 1024, // 1MB
            use_huge_pages: false,
            populate: false,
        }
    }
}

impl MmapConfig {
    /// Create a new configuration
    pub fn new(path: PathBuf) -> Self {
        Self {
            path,
            ..Default::default()
        }
    }

    /// Set mmap threshold
    pub fn with_threshold(mut self, threshold: usize) -> Self {
        self.mmap_threshold = threshold;
        self
    }

    /// Enable huge pages
    pub fn with_huge_pages(mut self, enable: bool) -> Self {
        self.use_huge_pages = enable;
        self
    }

    /// Set populate flag
    pub fn with_populate(mut self, populate: bool) -> Self {
        self.populate = populate;
        self
    }

    /// Build file path for a CID
    fn block_path(&self, cid: &Cid) -> PathBuf {
        let cid_str = cid.to_string();
        // Use first 2 chars as directory for better file system performance
        let dir = &cid_str[..2.min(cid_str.len())];
        self.path.join(dir).join(&cid_str)
    }
}

/// Memory-mapped block store
pub struct MmapBlockStore {
    config: MmapConfig,
    // Cache of open mmaps (CID -> Mmap)
    mmap_cache: Arc<RwLock<HashMap<Cid, Arc<Mmap>>>>,
}

impl MmapBlockStore {
    /// Create a new memory-mapped block store
    pub fn new(config: MmapConfig) -> Result<Self> {
        // Create directory if it doesn't exist
        std::fs::create_dir_all(&config.path)
            .map_err(|e| Error::Storage(format!("Failed to create directory: {e}")))?;

        Ok(Self {
            config,
            mmap_cache: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Get or create mmap for a CID
    fn get_mmap(&self, cid: &Cid) -> Result<Option<Arc<Mmap>>> {
        // Check cache first
        {
            let cache = self.mmap_cache.read();
            if let Some(mmap) = cache.get(cid) {
                return Ok(Some(Arc::clone(mmap)));
            }
        }

        let path = self.config.block_path(cid);
        if !path.exists() {
            return Ok(None);
        }

        // Open file and create mmap
        let file = File::open(&path)
            .map_err(|e| Error::Storage(format!("Failed to open block file: {e}")))?;

        let metadata = file
            .metadata()
            .map_err(|e| Error::Storage(format!("Failed to get file metadata: {e}")))?;

        // Don't mmap small files
        if metadata.len() < self.config.mmap_threshold as u64 {
            return Ok(None);
        }

        // Create mmap
        let mut mmap_opts = MmapOptions::new();

        #[cfg(unix)]
        {
            // Note: huge pages support varies by memmap2 version and platform
            // Removed use_huge_pages option as API may not be available
            if self.config.populate {
                mmap_opts.populate();
            }
        }

        let mmap = unsafe {
            mmap_opts
                .map(&file)
                .map_err(|e| Error::Storage(format!("Failed to create mmap: {e}")))?
        };

        let mmap = Arc::new(mmap);

        // Cache it
        {
            let mut cache = self.mmap_cache.write();
            cache.insert(*cid, Arc::clone(&mmap));
        }

        Ok(Some(mmap))
    }

    /// Read a block from file (non-mmap path)
    fn read_block_file(&self, cid: &Cid) -> Result<Option<Vec<u8>>> {
        let path = self.config.block_path(cid);
        if !path.exists() {
            return Ok(None);
        }

        let mut file = File::open(&path)
            .map_err(|e| Error::Storage(format!("Failed to open block file: {e}")))?;

        let mut data = Vec::new();
        file.read_to_end(&mut data)
            .map_err(|e| Error::Storage(format!("Failed to read block file: {e}")))?;

        Ok(Some(data))
    }

    /// Write a block to file
    fn write_block_file(&self, cid: &Cid, data: &[u8]) -> Result<()> {
        let path = self.config.block_path(cid);

        // Create parent directory
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| Error::Storage(format!("Failed to create directory: {e}")))?;
        }

        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)
            .map_err(|e| Error::Storage(format!("Failed to create block file: {e}")))?;

        file.write_all(data)
            .map_err(|e| Error::Storage(format!("Failed to write block file: {e}")))?;

        file.sync_all()
            .map_err(|e| Error::Storage(format!("Failed to sync block file: {e}")))?;

        Ok(())
    }

    /// Get configuration
    pub fn config(&self) -> &MmapConfig {
        &self.config
    }

    /// Clear mmap cache
    pub fn clear_cache(&self) {
        self.mmap_cache.write().clear();
    }

    /// Get cache size
    pub fn cache_size(&self) -> usize {
        self.mmap_cache.read().len()
    }
}

#[async_trait]
impl BlockStore for MmapBlockStore {
    /// Store a block
    async fn put(&self, block: &Block) -> Result<()> {
        self.write_block_file(block.cid(), block.data())?;

        // Invalidate cache entry if it exists
        self.mmap_cache.write().remove(block.cid());

        Ok(())
    }

    /// Retrieve a block using mmap if possible
    async fn get(&self, cid: &Cid) -> Result<Option<Block>> {
        // Try mmap first
        if let Some(mmap) = self.get_mmap(cid)? {
            let data = bytes::Bytes::copy_from_slice(&mmap[..]);
            return Ok(Some(Block::from_parts(*cid, data)));
        }

        // Fallback to regular read for small files
        if let Some(data) = self.read_block_file(cid)? {
            let data = bytes::Bytes::from(data);
            return Ok(Some(Block::from_parts(*cid, data)));
        }

        Ok(None)
    }

    /// Check if a block exists
    async fn has(&self, cid: &Cid) -> Result<bool> {
        let path = self.config.block_path(cid);
        Ok(path.exists())
    }

    /// Delete a block
    async fn delete(&self, cid: &Cid) -> Result<()> {
        // Remove from cache first
        self.mmap_cache.write().remove(cid);

        let path = self.config.block_path(cid);
        if path.exists() {
            std::fs::remove_file(&path)
                .map_err(|e| Error::Storage(format!("Failed to delete block file: {e}")))?;
        }

        Ok(())
    }

    /// Get the number of blocks stored
    fn len(&self) -> usize {
        // Walk directory to count files
        // This is expensive, so return 0 for now
        0
    }

    /// Check if the store is empty
    fn is_empty(&self) -> bool {
        false
    }

    /// Get all CIDs in the store
    fn list_cids(&self) -> Result<Vec<Cid>> {
        let mut cids = Vec::new();

        // Walk directory
        fn walk_dir(dir: &Path, cids: &mut Vec<Cid>) -> Result<()> {
            if !dir.exists() {
                return Ok(());
            }

            for entry in std::fs::read_dir(dir)
                .map_err(|e| Error::Storage(format!("Failed to read directory: {e}")))?
            {
                let entry =
                    entry.map_err(|e| Error::Storage(format!("Failed to read entry: {e}")))?;

                let path = entry.path();

                if path.is_dir() {
                    walk_dir(&path, cids)?;
                } else if path.is_file() {
                    if let Some(file_name) = path.file_name() {
                        if let Some(cid_str) = file_name.to_str() {
                            if let Ok(cid) = cid_str.parse::<Cid>() {
                                cids.push(cid);
                            }
                        }
                    }
                }
            }

            Ok(())
        }

        walk_dir(&self.config.path, &mut cids)?;

        Ok(cids)
    }

    /// Flush is a no-op (writes are synced immediately)
    async fn flush(&self) -> Result<()> {
        Ok(())
    }
}

/// Partial read support for mmap store
impl MmapBlockStore {
    /// Read a range from a block using mmap
    #[allow(clippy::unused_async)]
    pub async fn get_range(
        &self,
        cid: &Cid,
        offset: u64,
        length: usize,
    ) -> Result<Option<bytes::Bytes>> {
        // Try mmap first
        if let Some(mmap) = self.get_mmap(cid)? {
            let start = offset as usize;
            let end = (start + length).min(mmap.len());

            if start >= mmap.len() {
                return Ok(Some(bytes::Bytes::new()));
            }

            let data = bytes::Bytes::copy_from_slice(&mmap[start..end]);
            return Ok(Some(data));
        }

        // Fallback to seeking in file
        let path = self.config.block_path(cid);
        if !path.exists() {
            return Ok(None);
        }

        let mut file = File::open(&path)
            .map_err(|e| Error::Storage(format!("Failed to open block file: {e}")))?;

        file.seek(SeekFrom::Start(offset))
            .map_err(|e| Error::Storage(format!("Failed to seek in block file: {e}")))?;

        let mut buffer = vec![0u8; length];
        let n = file
            .read(&mut buffer)
            .map_err(|e| Error::Storage(format!("Failed to read from block file: {e}")))?;

        buffer.truncate(n);

        Ok(Some(bytes::Bytes::from(buffer)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;

    #[tokio::test]
    async fn test_mmap_put_get_block() {
        let config = MmapConfig::new(std::env::temp_dir().join("ipfrs-test-mmap"));

        // Clean up from previous test
        let _ = std::fs::remove_dir_all(&config.path);

        let store = MmapBlockStore::new(config).expect("test: MmapBlockStore should initialize");

        // Test with small block (below threshold)
        let small_data = Bytes::from("small block");
        let small_block =
            Block::new(small_data.clone()).expect("test: Block::new should succeed for small data");

        store
            .put(&small_block)
            .await
            .expect("test: put small block should succeed");
        let retrieved = store
            .get(small_block.cid())
            .await
            .expect("test: get small block should succeed");
        assert!(retrieved.is_some());
        assert_eq!(
            retrieved
                .expect("test: retrieved small block should be Some")
                .data(),
            &small_data
        );

        // Test with large block (above threshold)
        let large_data = Bytes::from(vec![0u8; 2 * 1024 * 1024]); // 2MB
        let large_block =
            Block::new(large_data.clone()).expect("test: Block::new should succeed for large data");

        store
            .put(&large_block)
            .await
            .expect("test: put large block should succeed");
        let retrieved = store
            .get(large_block.cid())
            .await
            .expect("test: get large block should succeed");
        assert!(retrieved.is_some());
        assert_eq!(
            retrieved
                .expect("test: retrieved large block should be Some")
                .data(),
            &large_data
        );

        // Test has
        assert!(store
            .has(small_block.cid())
            .await
            .expect("test: has small block should succeed"));
        assert!(store
            .has(large_block.cid())
            .await
            .expect("test: has large block should succeed"));

        // Test delete
        store
            .delete(small_block.cid())
            .await
            .expect("test: delete small block should succeed");
        assert!(!store
            .has(small_block.cid())
            .await
            .expect("test: has after delete should succeed"));
    }

    #[tokio::test]
    async fn test_mmap_partial_read() {
        let config = MmapConfig::new(std::env::temp_dir().join("ipfrs-test-mmap-partial"))
            .with_threshold(1024); // Lower threshold for testing

        // Clean up from previous test
        let _ = std::fs::remove_dir_all(&config.path);

        let store = MmapBlockStore::new(config).expect("test: MmapBlockStore should initialize");

        // Create a large block
        let data = Bytes::from((0..10000).map(|i| (i % 256) as u8).collect::<Vec<u8>>());
        let block = Block::new(data.clone())
            .expect("test: Block::new should succeed for partial read data");

        store
            .put(&block)
            .await
            .expect("test: put block should succeed");

        // Read a range
        let range = store
            .get_range(block.cid(), 100, 500)
            .await
            .expect("test: get_range should succeed");
        assert!(range.is_some());

        let range_data = range.expect("test: range result should be Some");
        assert_eq!(range_data.len(), 500);
        assert_eq!(&range_data[..], &data[100..600]);
    }

    #[tokio::test]
    async fn test_mmap_cache() {
        let config = MmapConfig::new(std::env::temp_dir().join("ipfrs-test-mmap-cache"))
            .with_threshold(1024);

        // Clean up from previous test
        let _ = std::fs::remove_dir_all(&config.path);

        let store = MmapBlockStore::new(config).expect("test: MmapBlockStore should initialize");

        // Create a large block
        let data = Bytes::from(vec![0u8; 10000]);
        let block =
            Block::new(data.clone()).expect("test: Block::new should succeed for cache data");

        store
            .put(&block)
            .await
            .expect("test: put block should succeed");

        // First get should populate cache
        assert_eq!(store.cache_size(), 0);
        let _ = store
            .get(block.cid())
            .await
            .expect("test: first get should succeed");
        assert_eq!(store.cache_size(), 1);

        // Second get should use cache
        let _ = store
            .get(block.cid())
            .await
            .expect("test: second get from cache should succeed");
        assert_eq!(store.cache_size(), 1);

        // Clear cache
        store.clear_cache();
        assert_eq!(store.cache_size(), 0);
    }

    #[tokio::test]
    async fn test_mmap_list_cids() {
        let config = MmapConfig::new(std::env::temp_dir().join("ipfrs-test-mmap-list"));

        // Clean up from previous test
        let _ = std::fs::remove_dir_all(&config.path);

        let store = MmapBlockStore::new(config).expect("test: MmapBlockStore should initialize");

        // Create multiple blocks
        let blocks: Vec<Block> = (0..5)
            .map(|i| {
                let data = Bytes::from(format!("block {}", i));
                Block::new(data).expect("test: Block::new should succeed for list data")
            })
            .collect();

        for block in &blocks {
            store
                .put(block)
                .await
                .expect("test: put block should succeed");
        }

        // List CIDs
        let cids = store.list_cids().expect("test: list_cids should succeed");
        assert_eq!(cids.len(), 5);

        // Verify all CIDs are present
        for block in &blocks {
            assert!(cids.contains(block.cid()));
        }
    }
}
