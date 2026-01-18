//! Block compression support
//!
//! Provides transparent compression/decompression of blocks using various algorithms:
//! - Zstd: Best compression ratio, fast decompression
//! - Lz4: Very fast compression/decompression, moderate ratio
//! - Snappy: Fast compression, good for streaming
//!
//! # Example
//!
//! ```rust,ignore
//! use ipfrs_storage::{SledBlockStore, CompressionBlockStore, CompressionConfig, CompressionAlgorithm};
//!
//! let store = SledBlockStore::open("/tmp/blocks")?;
//! let config = CompressionConfig::new(CompressionAlgorithm::Zstd)
//!     .with_level(3)
//!     .with_threshold(1024); // Only compress blocks > 1KB
//! let compressed = CompressionBlockStore::new(store, config);
//! ```

use crate::traits::BlockStore;
use async_trait::async_trait;
use ipfrs_core::{Block, Cid, Error, Result};
use parking_lot::RwLock;
use std::sync::Arc;

/// Compression algorithm
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionAlgorithm {
    /// Zstd compression (best ratio, fast decompression)
    Zstd,
    /// Lz4 compression (very fast, moderate ratio)
    Lz4,
    /// Snappy compression (fast, good for streaming)
    Snappy,
}

impl Default for CompressionAlgorithm {
    fn default() -> Self {
        Self::Zstd
    }
}

/// Compression configuration
#[derive(Debug, Clone)]
pub struct CompressionConfig {
    /// Compression algorithm to use
    pub algorithm: CompressionAlgorithm,
    /// Compression level (algorithm-specific)
    /// - Zstd: 1-22 (default: 3)
    /// - Lz4: 1-12 (default: 1)
    /// - Snappy: ignored
    pub level: i32,
    /// Only compress blocks larger than this threshold (bytes)
    /// Default: 512 bytes
    pub threshold: usize,
    /// Maximum compression ratio to accept (compressed/original)
    /// If compression doesn't achieve this ratio, store uncompressed
    /// Default: 0.9 (only compress if we save at least 10%)
    pub max_ratio: f64,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            algorithm: CompressionAlgorithm::default(),
            level: 3,
            threshold: 512,
            max_ratio: 0.9,
        }
    }
}

impl CompressionConfig {
    /// Create new configuration with specified algorithm
    pub fn new(algorithm: CompressionAlgorithm) -> Self {
        Self {
            algorithm,
            ..Default::default()
        }
    }

    /// Set compression level
    pub fn with_level(mut self, level: i32) -> Self {
        self.level = level;
        self
    }

    /// Set size threshold for compression
    pub fn with_threshold(mut self, threshold: usize) -> Self {
        self.threshold = threshold;
        self
    }

    /// Set maximum compression ratio
    pub fn with_max_ratio(mut self, max_ratio: f64) -> Self {
        self.max_ratio = max_ratio;
        self
    }
}

/// Block compression statistics
#[derive(Debug, Clone, Default)]
pub struct BlockCompressionStats {
    /// Number of blocks compressed
    pub blocks_compressed: u64,
    /// Number of blocks stored uncompressed (below threshold or poor ratio)
    pub blocks_uncompressed: u64,
    /// Total bytes before compression
    pub bytes_original: u64,
    /// Total bytes after compression
    pub bytes_compressed: u64,
    /// Number of decompressions
    pub decompressions: u64,
}

impl BlockCompressionStats {
    /// Calculate overall compression ratio
    pub fn compression_ratio(&self) -> f64 {
        if self.bytes_compressed == 0 {
            return 1.0;
        }
        self.bytes_compressed as f64 / self.bytes_original as f64
    }

    /// Calculate space saved in bytes
    pub fn bytes_saved(&self) -> u64 {
        self.bytes_original.saturating_sub(self.bytes_compressed)
    }

    /// Calculate space saved as percentage
    pub fn savings_percent(&self) -> f64 {
        if self.bytes_original == 0 {
            return 0.0;
        }
        (self.bytes_saved() as f64 / self.bytes_original as f64) * 100.0
    }
}

/// Block store with transparent compression
pub struct CompressionBlockStore<S> {
    inner: S,
    config: CompressionConfig,
    stats: Arc<RwLock<BlockCompressionStats>>,
}

impl<S> CompressionBlockStore<S> {
    /// Create new compression block store
    pub fn new(inner: S, config: CompressionConfig) -> Self {
        Self {
            inner,
            config,
            stats: Arc::new(RwLock::new(BlockCompressionStats::default())),
        }
    }

    /// Get compression statistics
    pub fn stats(&self) -> BlockCompressionStats {
        self.stats.read().clone()
    }

    /// Reset statistics
    pub fn reset_stats(&self) {
        let mut stats = self.stats.write();
        *stats = BlockCompressionStats::default();
    }

    /// Compress data based on configuration
    fn compress(&self, data: &[u8]) -> Result<Vec<u8>> {
        // Check if data is below threshold
        if data.len() < self.config.threshold {
            return Ok(Self::encode_uncompressed(data));
        }

        let compressed = match self.config.algorithm {
            CompressionAlgorithm::Zstd => zstd::encode_all(data, self.config.level)
                .map_err(|e| Error::Storage(format!("Zstd compression failed: {e}")))?,
            CompressionAlgorithm::Lz4 => lz4::block::compress(data, None, true)
                .map_err(|e| Error::Storage(format!("Lz4 compression failed: {e}")))?,
            CompressionAlgorithm::Snappy => snap::raw::Encoder::new()
                .compress_vec(data)
                .map_err(|e| Error::Storage(format!("Snappy compression failed: {e}")))?,
        };

        // Check compression ratio
        let ratio = compressed.len() as f64 / data.len() as f64;
        if ratio > self.config.max_ratio {
            // Compression not effective, store uncompressed
            return Ok(Self::encode_uncompressed(data));
        }

        // Add compression marker
        Ok(Self::encode_compressed(&compressed, self.config.algorithm))
    }

    /// Decompress data
    fn decompress(&self, data: &[u8]) -> Result<Vec<u8>> {
        if data.is_empty() {
            return Ok(Vec::new());
        }

        // Check compression marker (first byte)
        match data[0] {
            0 => {
                // Uncompressed
                Ok(data[1..].to_vec())
            }
            1 => {
                // Zstd compressed
                zstd::decode_all(&data[1..])
                    .map_err(|e| Error::Storage(format!("Zstd decompression failed: {e}")))
            }
            2 => {
                // Lz4 compressed
                lz4::block::decompress(&data[1..], None)
                    .map_err(|e| Error::Storage(format!("Lz4 decompression failed: {e}")))
            }
            3 => {
                // Snappy compressed
                snap::raw::Decoder::new()
                    .decompress_vec(&data[1..])
                    .map_err(|e| Error::Storage(format!("Snappy decompression failed: {e}")))
            }
            _ => Err(Error::Storage(format!(
                "Unknown compression marker: {}",
                data[0]
            ))),
        }
    }

    /// Encode uncompressed data with marker
    fn encode_uncompressed(data: &[u8]) -> Vec<u8> {
        let mut result = Vec::with_capacity(data.len() + 1);
        result.push(0); // Uncompressed marker
        result.extend_from_slice(data);
        result
    }

    /// Encode compressed data with algorithm marker
    fn encode_compressed(data: &[u8], algorithm: CompressionAlgorithm) -> Vec<u8> {
        let mut result = Vec::with_capacity(data.len() + 1);
        let marker = match algorithm {
            CompressionAlgorithm::Zstd => 1,
            CompressionAlgorithm::Lz4 => 2,
            CompressionAlgorithm::Snappy => 3,
        };
        result.push(marker);
        result.extend_from_slice(data);
        result
    }
}

#[async_trait]
impl<S: BlockStore> BlockStore for CompressionBlockStore<S> {
    async fn put(&self, block: &Block) -> Result<()> {
        let original_size = block.data().len();
        let compressed = self.compress(block.data())?;
        let compressed_size = compressed.len();

        // Update stats (in explicit scope to avoid Send issues)
        {
            let mut stats = self.stats.write();
            stats.bytes_original += original_size as u64;
            stats.bytes_compressed += compressed_size as u64;

            if compressed[0] == 0 {
                // Stored uncompressed
                stats.blocks_uncompressed += 1;
            } else {
                // Stored compressed
                stats.blocks_compressed += 1;
            }
        }

        // Create new block with compressed data (preserving original CID)
        let compressed_block = Block::from_parts(*block.cid(), compressed.into());
        self.inner.put(&compressed_block).await
    }

    async fn put_many(&self, blocks: &[Block]) -> Result<()> {
        let mut compressed_blocks = Vec::with_capacity(blocks.len());

        // Update stats (in explicit scope to avoid Send issues)
        {
            let mut stats = self.stats.write();

            for block in blocks {
                let original_size = block.data().len();
                let compressed = self.compress(block.data())?;
                let compressed_size = compressed.len();

                stats.bytes_original += original_size as u64;
                stats.bytes_compressed += compressed_size as u64;

                if compressed[0] == 0 {
                    stats.blocks_uncompressed += 1;
                } else {
                    stats.blocks_compressed += 1;
                }

                compressed_blocks.push(Block::from_parts(*block.cid(), compressed.into()));
            }
        }

        self.inner.put_many(&compressed_blocks).await
    }

    async fn get(&self, cid: &Cid) -> Result<Option<Block>> {
        if let Some(compressed_block) = self.inner.get(cid).await? {
            let data = self.decompress(compressed_block.data())?;

            // Update stats (in explicit scope to avoid Send issues)
            {
                let mut stats = self.stats.write();
                stats.decompressions += 1;
            }

            Ok(Some(Block::from_parts(*cid, data.into())))
        } else {
            Ok(None)
        }
    }

    async fn get_many(&self, cids: &[Cid]) -> Result<Vec<Option<Block>>> {
        let compressed = self.inner.get_many(cids).await?;
        let mut results = Vec::with_capacity(compressed.len());
        let mut decompression_count = 0;

        for (i, item) in compressed.into_iter().enumerate() {
            if let Some(compressed_block) = item {
                let data = self.decompress(compressed_block.data())?;
                decompression_count += 1;
                results.push(Some(Block::from_parts(cids[i], data.into())));
            } else {
                results.push(None);
            }
        }

        // Update stats (in explicit scope to avoid Send issues)
        {
            let mut stats = self.stats.write();
            stats.decompressions += decompression_count;
        }

        Ok(results)
    }

    async fn has(&self, cid: &Cid) -> Result<bool> {
        self.inner.has(cid).await
    }

    async fn has_many(&self, cids: &[Cid]) -> Result<Vec<bool>> {
        self.inner.has_many(cids).await
    }

    async fn delete(&self, cid: &Cid) -> Result<()> {
        self.inner.delete(cid).await
    }

    async fn delete_many(&self, cids: &[Cid]) -> Result<()> {
        self.inner.delete_many(cids).await
    }

    fn list_cids(&self) -> Result<Vec<Cid>> {
        self.inner.list_cids()
    }

    fn len(&self) -> usize {
        self.inner.len()
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
    use crate::blockstore::SledBlockStore;

    #[tokio::test]
    async fn test_compression_basic() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let config = crate::BlockStoreConfig {
            path: temp_dir.path().join("blocks"),
            cache_size: 10_000_000,
        };
        let store = SledBlockStore::new(config).unwrap();
        let config = CompressionConfig::new(CompressionAlgorithm::Zstd);
        let compressed_store = CompressionBlockStore::new(store, config);

        let data = vec![42u8; 10000]; // Highly compressible
        let block = Block::new(data.clone().into()).unwrap();

        compressed_store.put(&block).await.unwrap();
        let retrieved = compressed_store.get(block.cid()).await.unwrap().unwrap();
        assert_eq!(data.as_slice(), retrieved.data().as_ref());

        let stats = compressed_store.stats();
        assert_eq!(stats.blocks_compressed, 1);
        assert!(stats.compression_ratio() < 0.1); // Should compress very well
    }

    #[tokio::test]
    async fn test_compression_threshold() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let store_config = crate::BlockStoreConfig {
            path: temp_dir.path().join("blocks"),
            cache_size: 10_000_000,
        };
        let store = SledBlockStore::new(store_config).unwrap();
        let config = CompressionConfig::new(CompressionAlgorithm::Zstd).with_threshold(1000);
        let compressed_store = CompressionBlockStore::new(store, config);

        // Small data (below threshold)
        let small_data = vec![42u8; 100];
        let block1 = Block::new(small_data.into()).unwrap();
        compressed_store.put(&block1).await.unwrap();

        // Large data (above threshold)
        let large_data = vec![42u8; 10000];
        let block2 = Block::new(large_data.into()).unwrap();
        compressed_store.put(&block2).await.unwrap();

        let stats = compressed_store.stats();
        assert_eq!(stats.blocks_uncompressed, 1); // Small block
        assert_eq!(stats.blocks_compressed, 1); // Large block
    }

    #[tokio::test]
    async fn test_compression_algorithms() {
        for algorithm in [
            CompressionAlgorithm::Zstd,
            CompressionAlgorithm::Lz4,
            CompressionAlgorithm::Snappy,
        ] {
            let temp_dir = tempfile::TempDir::new().unwrap();
            let store_config = crate::BlockStoreConfig {
                path: temp_dir.path().join("blocks"),
                cache_size: 10_000_000,
            };
            let store = SledBlockStore::new(store_config).unwrap();
            let config = CompressionConfig::new(algorithm);
            let compressed_store = CompressionBlockStore::new(store, config);

            let data = vec![42u8; 10000];
            let block = Block::new(data.clone().into()).unwrap();

            compressed_store.put(&block).await.unwrap();
            let retrieved = compressed_store.get(block.cid()).await.unwrap().unwrap();
            assert_eq!(data.as_slice(), retrieved.data().as_ref());
        }
    }

    #[tokio::test]
    async fn test_compression_batch() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let store_config = crate::BlockStoreConfig {
            path: temp_dir.path().join("blocks"),
            cache_size: 10_000_000,
        };
        let store = SledBlockStore::new(store_config).unwrap();
        let config = CompressionConfig::new(CompressionAlgorithm::Zstd);
        let compressed_store = CompressionBlockStore::new(store, config);

        let blocks: Vec<_> = (0..10)
            .map(|i| Block::new(vec![i; 5000].into()).unwrap())
            .collect();

        compressed_store.put_many(&blocks).await.unwrap();

        let cids: Vec<_> = blocks.iter().map(|b| *b.cid()).collect();
        let retrieved = compressed_store.get_many(&cids).await.unwrap();

        for (i, item) in retrieved.iter().enumerate() {
            let block = item.as_ref().unwrap();
            assert_eq!(block.data(), blocks[i].data());
        }

        let stats = compressed_store.stats();
        assert_eq!(stats.blocks_compressed, 10);
        assert_eq!(stats.decompressions, 10);
    }

    #[tokio::test]
    async fn test_incompressible_data() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let store_config = crate::BlockStoreConfig {
            path: temp_dir.path().join("blocks"),
            cache_size: 10_000_000,
        };
        let store = SledBlockStore::new(store_config).unwrap();
        let config = CompressionConfig::new(CompressionAlgorithm::Zstd).with_max_ratio(0.9);
        let compressed_store = CompressionBlockStore::new(store, config);

        // Random data (incompressible)
        use rand::Rng;
        let mut rng = rand::rng();
        let data: Vec<u8> = (0..10000).map(|_| rng.random_range(0..=255)).collect();

        let block = Block::new(data.clone().into()).unwrap();
        compressed_store.put(&block).await.unwrap();
        let retrieved = compressed_store.get(block.cid()).await.unwrap().unwrap();
        assert_eq!(data.as_slice(), retrieved.data().as_ref());

        let stats = compressed_store.stats();
        // Should store uncompressed due to poor ratio
        assert_eq!(stats.blocks_uncompressed, 1);
        assert_eq!(stats.blocks_compressed, 0);
    }
}
