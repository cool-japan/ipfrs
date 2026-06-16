//! Block deduplication using content-defined chunking.
//!
//! This module provides transparent deduplication for block storage using
//! FastCDC (Fast Content-Defined Chunking) algorithm.
//!
//! # Features
//! - FastCDC algorithm for reliable boundary detection
//! - Chunk-level deduplication with reference counting
//! - Deduplication statistics and savings tracking
//! - Transparent block reconstruction
//! - Automatic garbage collection of unreferenced chunks

use crate::traits::BlockStore;
use async_trait::async_trait;
use dashmap::DashMap;
use ipfrs_core::{Block, Cid, Error, Result};
use parking_lot::RwLock;
use std::sync::Arc;

/// Chunk configuration for content-defined chunking
#[derive(Debug, Clone)]
pub struct ChunkingConfig {
    /// Minimum chunk size (default: 256KB)
    pub min_chunk_size: usize,
    /// Target chunk size (default: 1MB)
    pub target_chunk_size: usize,
    /// Maximum chunk size (default: 4MB)
    pub max_chunk_size: usize,
    /// Rolling hash mask (determines avg chunk size)
    pub hash_mask: u32,
}

impl Default for ChunkingConfig {
    fn default() -> Self {
        Self {
            min_chunk_size: 256 * 1024,      // 256KB
            target_chunk_size: 1024 * 1024,  // 1MB
            max_chunk_size: 4 * 1024 * 1024, // 4MB
            hash_mask: 0xFFFF,               // ~64KB avg chunks
        }
    }
}

impl ChunkingConfig {
    /// Create config optimized for small blocks
    pub fn small() -> Self {
        Self {
            min_chunk_size: 64 * 1024,     // 64KB
            target_chunk_size: 256 * 1024, // 256KB
            max_chunk_size: 1024 * 1024,   // 1MB
            hash_mask: 0x3FFF,             // ~16KB avg chunks
        }
    }

    /// Create config optimized for large blocks
    pub fn large() -> Self {
        Self {
            min_chunk_size: 1024 * 1024,        // 1MB
            target_chunk_size: 4 * 1024 * 1024, // 4MB
            max_chunk_size: 16 * 1024 * 1024,   // 16MB
            hash_mask: 0x1FFFF,                 // ~128KB avg chunks
        }
    }
}

/// Chunk metadata stored in the dedup index
#[derive(Debug, Clone)]
struct ChunkMeta {
    /// CID of the chunk data
    cid: Cid,
    /// Reference count
    ref_count: usize,
    /// Chunk size in bytes
    size: usize,
}

/// Block manifest mapping a block to its chunks
#[derive(Debug, Clone)]
struct BlockManifest {
    /// Original block size
    original_size: usize,
    /// List of chunk CIDs that make up this block
    chunks: Vec<Cid>,
}

/// Deduplication statistics
#[derive(Debug, Clone, Default)]
pub struct DedupStats {
    /// Total number of blocks stored
    pub blocks_stored: usize,
    /// Total bytes before deduplication
    pub bytes_original: usize,
    /// Total bytes after deduplication (unique chunks)
    pub bytes_stored: usize,
    /// Number of unique chunks
    pub unique_chunks: usize,
    /// Number of duplicate chunks avoided
    pub duplicate_chunks_avoided: usize,
}

impl DedupStats {
    /// Calculate deduplication ratio (savings)
    pub fn dedup_ratio(&self) -> f64 {
        if self.bytes_original == 0 {
            return 0.0;
        }
        1.0 - (self.bytes_stored as f64 / self.bytes_original as f64)
    }

    /// Calculate space saved in bytes
    pub fn bytes_saved(&self) -> usize {
        self.bytes_original.saturating_sub(self.bytes_stored)
    }

    /// Calculate average chunk size
    pub fn avg_chunk_size(&self) -> usize {
        if self.unique_chunks == 0 {
            return 0;
        }
        self.bytes_stored / self.unique_chunks
    }
}

/// FastCDC gear hash table (precomputed random values for each byte)
/// Reserved for future use with gear-based hashing
#[allow(dead_code)]
const GEAR: [u64; 256] = [
    0x5c95c078, 0x22408989, 0x2d48a214, 0x12842087, 0x530f8afb, 0x2aaa3f86, 0x7f1bd89f, 0x62534467,
    0x22c4b83b, 0x3e36d3e7, 0x4c9fa05b, 0x0b20f0e3, 0x441c8a8c, 0x7cc27988, 0x5505c6c0, 0x3c9ae0da,
    0x153e46cd, 0x0d05f5b5, 0x51c9c3b5, 0x02e57b86, 0x74a8d4ba, 0x6f16cbb5, 0x2ffc27ea, 0x5fa83e0f,
    0x75ab67e2, 0x3ff15813, 0x2ec58ac7, 0x6f1f0520, 0x0c5d7dba, 0x4a9f5e76, 0x4ec58e64, 0x6a470c8e,
    0x40edf2ca, 0x1a1c0c8d, 0x4e32e5e4, 0x6c7a7fda, 0x4b3be9e4, 0x64d8e67b, 0x2ef8ad98, 0x34d9f7e5,
    0x7e7e4a36, 0x1a1c54d1, 0x5e2a9e7a, 0x3e5f0a8e, 0x0e01d1a0, 0x1f31aa27, 0x049c9e3e, 0x7c38f56e,
    0x4b8d9ef0, 0x0b9c4d05, 0x55f59f0d, 0x3e8e02ae, 0x25c46f84, 0x6e6fdc6f, 0x440ae4a7, 0x3e38a0e6,
    0x5b96c3d1, 0x72a06105, 0x52cd5e2d, 0x3d015fb3, 0x4d7c7064, 0x1c8c169c, 0x5c95e834, 0x0c4d9d42,
    0x3c9c8ea3, 0x10a5d9d6, 0x7dcb9d63, 0x3ecf9e96, 0x1f5c9e5f, 0x7e7854c5, 0x48a05ae3, 0x0c4e9419,
    0x6b5c9b6f, 0x7e1a6dc0, 0x3b8f9fe8, 0x6f6e8e3f, 0x39f48adb, 0x7b8d9e72, 0x29e18dc5, 0x7e6c3fc4,
    0x5d9c4ab8, 0x1f6e9dc2, 0x3e8f9fc3, 0x7d9c8ea6, 0x0e1f8d9c, 0x5f9d8e72, 0x3e9f8dcb, 0x7d8e9f72,
    0x2f9d8ea5, 0x6e8f9dc4, 0x3d9f8ec5, 0x7e8d9f63, 0x1f9e8dc3, 0x6d8f9ec4, 0x3e9d8fc5, 0x7d9e8f62,
    0x2e9f8dc4, 0x6f8d9ec5, 0x3d9e8fc3, 0x7e9d8f64, 0x1f8e9dc5, 0x6e9f8dc4, 0x3d8e9fc5, 0x7d9f8e63,
    0x2f8d9ec4, 0x6e8f9dc5, 0x3e9d8fc4, 0x7d8e9f65, 0x1f9d8ec5, 0x6d9f8dc4, 0x3e8d9fc5, 0x7e9f8d62,
    0x2d8e9fc4, 0x6f9d8ec5, 0x3d8f9dc4, 0x7e8d9f66, 0x1e9f8dc5, 0x6d8e9fc4, 0x3f9d8ec5, 0x7d9e8f61,
    0x2f9d8ec4, 0x6e8d9fc5, 0x3d9f8dc4, 0x7e8f9d67, 0x1f8d9ec5, 0x6e9d8fc4, 0x3d8e9fc5, 0x7f9d8e60,
    0x2e8f9dc4, 0x6f9e8dc5, 0x3d8d9fc4, 0x7e9f8d68, 0x1d9e8fc5, 0x6f8d9ec4, 0x3e9f8dc5, 0x7d8e9f5f,
    0x2f8e9dc4, 0x6d9f8ec5, 0x3e8d9fc4, 0x7f9e8d69, 0x1f9d8ec5, 0x6e8f9dc4, 0x3d9e8fc5, 0x7e8d9f5e,
    0x2d9f8ec4, 0x6f8e9dc5, 0x3d8f9fc4, 0x7e9d8e6a, 0x1e8f9dc5, 0x6d9e8fc4, 0x3f8d9ec5, 0x7d9f8e5d,
    0x2f8d9fc4, 0x6e9f8ec5, 0x3d8e9dc4, 0x7f8d9e6b, 0x1f8e9fc5, 0x6e8d9ec4, 0x3d9f8fc5, 0x7e9e8d5c,
    0x2e9d8fc4, 0x6f8e9dc5, 0x3e8f9ec4, 0x7d9e8f6c, 0x1f9e8dc5, 0x6d8f9fc4, 0x3e9d8ec5, 0x7d8f9e5b,
    0x2f9e8dc4, 0x6e8d9fc5, 0x3d9f8ec4, 0x7e8f9d6d, 0x1e9d8fc5, 0x6f8e9dc4, 0x3d8f9ec5, 0x7e9d8f5a,
    0x2d8f9ec4, 0x6e9d8fc5, 0x3f8e9dc4, 0x7d9f8e6e, 0x1f8d9fc5, 0x6e9e8dc4, 0x3d8f9fc5, 0x7f8e9d59,
    0x2e8d9fc4, 0x6f9e8dc5, 0x3d9f8ec4, 0x7e8d9f6f, 0x1d9f8ec5, 0x6f8d9dc4, 0x3e8e9fc5, 0x7d9f8e58,
    0x2f8e9fc4, 0x6d9f8dc5, 0x3e8d9ec4, 0x7f9e8d70, 0x1f8e9dc5, 0x6d8f9ec4, 0x3f9d8fc5, 0x7e8f9d57,
    0x2d9e8fc4, 0x6f8e9dc5, 0x3d8f9ec4, 0x7e9d8f71, 0x1e9f8dc5, 0x6f8d9ec4, 0x3d9e8fc5, 0x7f8d9e56,
    0x2f8d9ec4, 0x6e9f8dc5, 0x3e8d9fc4, 0x7d9e8f72, 0x1f9d8fc5, 0x6e8f9dc4, 0x3d8e9fc5, 0x7e9f8d55,
    0x2e8f9fc4, 0x6d9e8dc5, 0x3f8d9ec4, 0x7e8f9d73, 0x1d9f8fc5, 0x6f8e9dc4, 0x3e8d9fc5, 0x7d8f9e54,
    0x2f9e8dc4, 0x6e8f9fc5, 0x3d9d8ec4, 0x7f8e9d74, 0x1e8d9fc5, 0x6d9f8ec4, 0x3f8e9dc5, 0x7e9d8f53,
    0x2d8e9fc4, 0x6f9d8ec5, 0x3d8f9fc4, 0x7e9f8d75, 0x1f8d9ec5, 0x6e9d8fc4, 0x3d9f8ec5, 0x7f8e9d52,
    0x2e9f8dc4, 0x6d8e9fc5, 0x3f9d8ec4, 0x7d8f9e76, 0x1f9e8dc5, 0x6f8d9ec4, 0x3e9f8fc5, 0x7d9e8f51,
    0x2f8d9fc4, 0x6e9e8dc5, 0x3d8f9ec4, 0x7e8d9f77, 0x1e9f8dc5, 0x6d8f9fc4, 0x3f8e9dc5, 0x7e9d8e50,
];

/// Content-defined chunking engine using FastCDC algorithm
struct Chunker {
    config: ChunkingConfig,
}

impl Chunker {
    fn new(config: ChunkingConfig) -> Self {
        Self { config }
    }

    /// Split data into content-defined chunks using FastCDC
    fn chunk(&self, data: &[u8]) -> Vec<Vec<u8>> {
        if data.len() <= self.config.min_chunk_size {
            return vec![data.to_vec()];
        }

        let mut chunks = Vec::new();
        let mut start = 0;

        while start < data.len() {
            let remaining = data.len() - start;

            // If remaining is less than min, add it as final chunk
            if remaining <= self.config.min_chunk_size {
                chunks.push(data[start..].to_vec());
                break;
            }

            // Find chunk boundary using FastCDC
            let boundary = self.find_boundary(&data[start..]);
            let end = start + boundary;

            chunks.push(data[start..end].to_vec());
            start = end;
        }

        chunks
    }

    /// Find chunk boundary using FastCDC algorithm with normalized chunking
    #[allow(clippy::needless_range_loop)]
    fn find_boundary(&self, data: &[u8]) -> usize {
        let max_scan = self.config.max_chunk_size.min(data.len());
        let min_size = self.config.min_chunk_size.min(data.len());

        // FastCDC uses normalized chunking with two levels
        let nc_level = min_size + (self.config.target_chunk_size - min_size) / 4;

        let mut hash: u64 = 0;
        const PRIME: u64 = 0x01000193; // FNV prime
        let mask_s = self.config.hash_mask as u64; // Mask for smaller chunks
        let mask_l = (self.config.hash_mask >> 1) as u64; // Mask for larger chunks

        // Start from minimum chunk size (range needed for offset calculation)
        for idx in min_size..max_scan {
            let byte = data[idx];

            // Update rolling hash using FNV-like hash for better distribution
            hash = hash.wrapping_mul(PRIME) ^ (byte as u64);

            // Use different mask based on position (normalized chunking)
            let mask = if idx < nc_level { mask_s } else { mask_l };

            // Check if we hit a boundary
            if (hash & mask) == 0 {
                return idx + 1;
            }
        }

        // Return max chunk size if no boundary found
        max_scan
    }
}

/// Deduplicating block store wrapper
pub struct DedupBlockStore<S> {
    inner: S,
    config: ChunkingConfig,
    /// Chunk index: chunk_cid -> ChunkMeta
    chunk_index: Arc<DashMap<Cid, ChunkMeta>>,
    /// Block manifests: block_cid -> BlockManifest
    manifests: Arc<DashMap<Cid, BlockManifest>>,
    /// Statistics
    stats: Arc<RwLock<DedupStats>>,
}

impl<S: BlockStore> DedupBlockStore<S> {
    /// Create a new deduplicating block store
    pub fn new(inner: S, config: ChunkingConfig) -> Self {
        Self {
            inner,
            config,
            chunk_index: Arc::new(DashMap::new()),
            manifests: Arc::new(DashMap::new()),
            stats: Arc::new(RwLock::new(DedupStats::default())),
        }
    }

    /// Create with default configuration
    pub fn with_defaults(inner: S) -> Self {
        Self::new(inner, ChunkingConfig::default())
    }

    /// Get deduplication statistics
    pub fn stats(&self) -> DedupStats {
        self.stats.read().clone()
    }

    /// Get the underlying store
    pub fn into_inner(self) -> S {
        self.inner
    }

    /// Get a reference to the underlying store
    pub fn inner(&self) -> &S {
        &self.inner
    }

    /// Store a chunk and update the dedup index
    async fn store_chunk(&self, chunk_data: &[u8]) -> Result<Cid> {
        // Create chunk block to get its CID
        let chunk_block = Block::new(bytes::Bytes::copy_from_slice(chunk_data))?;
        let chunk_cid = *chunk_block.cid();

        // Check if chunk already exists
        if let Some(mut meta) = self.chunk_index.get_mut(&chunk_cid) {
            // Increment reference count
            meta.ref_count += 1;

            // Update stats for duplicate
            let mut stats = self.stats.write();
            stats.duplicate_chunks_avoided += 1;

            return Ok(meta.cid);
        }

        // New chunk - store it
        self.inner.put(&chunk_block).await?;

        // Add to index
        self.chunk_index.insert(
            chunk_cid,
            ChunkMeta {
                cid: chunk_cid,
                ref_count: 1,
                size: chunk_data.len(),
            },
        );

        // Update stats
        let mut stats = self.stats.write();
        stats.unique_chunks += 1;
        stats.bytes_stored += chunk_data.len();

        Ok(chunk_cid)
    }

    /// Retrieve chunks and reconstruct block
    async fn reconstruct_block(&self, manifest: &BlockManifest) -> Result<Block> {
        let mut data = Vec::with_capacity(manifest.original_size);

        for chunk_cid in &manifest.chunks {
            let chunk_block = self
                .inner
                .get(chunk_cid)
                .await?
                .ok_or_else(|| Error::BlockNotFound(chunk_cid.to_string()))?;
            data.extend_from_slice(chunk_block.data());
        }

        Block::new(bytes::Bytes::from(data))
    }

    /// Decrement chunk reference counts
    async fn decrement_chunk_refs(&self, chunk_cids: &[Cid]) -> Result<()> {
        let mut to_delete = Vec::new();

        for cid in chunk_cids {
            let should_delete = {
                if let Some(mut entry) = self.chunk_index.get_mut(cid) {
                    entry.ref_count = entry.ref_count.saturating_sub(1);
                    entry.ref_count == 0
                } else {
                    false
                }
            };

            if should_delete {
                to_delete.push(*cid);
            }
        }

        // Delete unreferenced chunks
        for cid in to_delete {
            if let Some((_, meta)) = self.chunk_index.remove(&cid) {
                self.inner.delete(&cid).await?;

                // Update stats
                let mut stats = self.stats.write();
                stats.unique_chunks = stats.unique_chunks.saturating_sub(1);
                stats.bytes_stored = stats.bytes_stored.saturating_sub(meta.size);
            }
        }

        Ok(())
    }
}

#[async_trait]
impl<S: BlockStore> BlockStore for DedupBlockStore<S> {
    async fn put(&self, block: &Block) -> Result<()> {
        let data = block.data();
        let original_size = data.len();
        let block_cid = *block.cid();

        // Check if block already exists
        let is_new_block = !self.manifests.contains_key(&block_cid);

        // If block exists with same CID, it's the same data - no need to re-store
        // Just update the manifest (idempotent operation)
        if !is_new_block {
            // Same CID means same data, chunks will be identical
            // Just ensure manifest exists (it already does)
            return Ok(());
        }

        // Chunk the data
        let chunker = Chunker::new(self.config.clone());
        let chunks = chunker.chunk(data);

        // Store each chunk
        let mut chunk_cids = Vec::new();
        for chunk in chunks {
            let cid = self.store_chunk(&chunk).await?;
            chunk_cids.push(cid);
        }

        // Create and store manifest
        let manifest = BlockManifest {
            original_size,
            chunks: chunk_cids,
        };

        self.manifests.insert(block_cid, manifest);

        // Update stats for new block
        let mut stats = self.stats.write();
        stats.blocks_stored += 1;
        stats.bytes_original += original_size;

        Ok(())
    }

    async fn get(&self, cid: &Cid) -> Result<Option<Block>> {
        // Get manifest
        let manifest = match self.manifests.get(cid) {
            Some(m) => m.clone(),
            None => return Ok(None),
        };

        // Reconstruct block from chunks
        let block = self.reconstruct_block(&manifest).await?;
        Ok(Some(block))
    }

    async fn has(&self, cid: &Cid) -> Result<bool> {
        Ok(self.manifests.contains_key(cid))
    }

    async fn delete(&self, cid: &Cid) -> Result<()> {
        // Get and remove manifest
        let manifest = match self.manifests.remove(cid) {
            Some((_, m)) => m,
            None => return Ok(()),
        };

        // Decrement chunk reference counts
        self.decrement_chunk_refs(&manifest.chunks).await?;

        // Update stats
        let mut stats = self.stats.write();
        stats.blocks_stored = stats.blocks_stored.saturating_sub(1);
        stats.bytes_original = stats.bytes_original.saturating_sub(manifest.original_size);

        Ok(())
    }

    fn list_cids(&self) -> Result<Vec<Cid>> {
        let cids: Vec<Cid> = self.manifests.iter().map(|entry| *entry.key()).collect();
        Ok(cids)
    }

    fn len(&self) -> usize {
        self.manifests.len()
    }

    fn is_empty(&self) -> bool {
        self.manifests.is_empty()
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
    use crate::blockstore::{BlockStoreConfig, SledBlockStore};

    #[test]
    fn test_chunking_config() {
        let config = ChunkingConfig::default();
        assert_eq!(config.min_chunk_size, 256 * 1024);
        assert_eq!(config.target_chunk_size, 1024 * 1024);

        let small = ChunkingConfig::small();
        assert!(small.min_chunk_size < config.min_chunk_size);

        let large = ChunkingConfig::large();
        assert!(large.min_chunk_size > config.min_chunk_size);
    }

    #[test]
    fn test_chunker_basic() {
        let config = ChunkingConfig {
            min_chunk_size: 16 * 1024,
            target_chunk_size: 64 * 1024,
            max_chunk_size: 128 * 1024,
            hash_mask: 0xFFF,
        };
        let chunker = Chunker::new(config.clone());

        // Data smaller than min should be single chunk
        let small_data: Vec<u8> = (0..10240).map(|i| (i % 256) as u8).collect(); // 10KB
        let chunks = chunker.chunk(&small_data);
        assert_eq!(chunks.len(), 1, "10KB data should be 1 chunk (min is 16KB)");
        assert_eq!(chunks[0].len(), 10240);

        // Identical data should produce identical chunks
        let small_data2: Vec<u8> = (0..10240).map(|i| (i % 256) as u8).collect(); // 10KB
        let chunks2 = chunker.chunk(&small_data2);
        assert_eq!(chunks2.len(), 1);
        assert_eq!(
            chunks[0], chunks2[0],
            "Identical data should produce identical chunks"
        );

        // Check that chunk CIDs would be the same
        let chunk_block1 = Block::new(bytes::Bytes::copy_from_slice(&chunks[0]))
            .expect("creating block from chunk data should succeed");
        let chunk_block2 = Block::new(bytes::Bytes::copy_from_slice(&chunks2[0]))
            .expect("creating block from identical chunk data should succeed");
        assert_eq!(
            chunk_block1.cid(),
            chunk_block2.cid(),
            "Identical chunks should have same CID"
        );
    }

    #[test]
    fn test_dedup_stats() {
        let stats = DedupStats {
            blocks_stored: 0,
            bytes_original: 1000,
            bytes_stored: 600,
            unique_chunks: 0,
            duplicate_chunks_avoided: 0,
        };

        assert_eq!(stats.dedup_ratio(), 0.4); // 40% savings
        assert_eq!(stats.bytes_saved(), 400);
    }

    #[test]
    fn test_chunker() {
        let config = ChunkingConfig::small();
        let chunker = Chunker::new(config.clone());

        // Small data should be single chunk
        let small_data = vec![0u8; 32 * 1024]; // 32KB
        let chunks = chunker.chunk(&small_data);
        assert_eq!(chunks.len(), 1);

        // Larger data with varied content should be chunked
        // FastCDC works best with non-uniform data
        let mut large_data = Vec::new();
        for i in 0..500 {
            // Create 1KB blocks of varying data
            let block: Vec<u8> = (0..1024).map(|j| ((i * 1024 + j) % 256) as u8).collect();
            large_data.extend_from_slice(&block);
        }
        let chunks = chunker.chunk(&large_data);

        // With varied data, FastCDC should find boundaries
        // The exact number depends on content, but should be > 1 for 500KB
        assert!(
            chunks.len() > 1,
            "Expected multiple chunks for 500KB of varied data"
        );

        // Verify chunks respect size constraints
        for (i, chunk) in chunks.iter().enumerate() {
            if i < chunks.len() - 1 {
                // Not the last chunk
                assert!(
                    chunk.len() >= config.min_chunk_size,
                    "Chunk {} size {} < min {}",
                    i,
                    chunk.len(),
                    config.min_chunk_size
                );
                assert!(
                    chunk.len() <= config.max_chunk_size,
                    "Chunk {} size {} > max {}",
                    i,
                    chunk.len(),
                    config.max_chunk_size
                );
            }
        }
    }

    #[tokio::test]
    async fn test_dedup_blockstore_basic() {
        let config = BlockStoreConfig {
            path: std::env::temp_dir().join("ipfrs-test-dedup-basic"),
            cache_size: 1024 * 1024,
        };

        // Clean up
        let _ = std::fs::remove_dir_all(&config.path);

        let inner = SledBlockStore::new(config).expect("creating sled block store should succeed");
        let store = DedupBlockStore::with_defaults(inner);

        // Store a block
        let data = bytes::Bytes::from(vec![1u8; 100 * 1024]); // 100KB
        let block = Block::new(data.clone()).expect("creating block from byte data should succeed");

        store
            .put(&block)
            .await
            .expect("storing block should succeed");

        // Retrieve it
        let retrieved = store
            .get(block.cid())
            .await
            .expect("get should not error")
            .expect("block should be present after put");
        assert_eq!(retrieved.data(), block.data());

        // Check stats
        let stats = store.stats();
        assert_eq!(stats.blocks_stored, 1);
        assert_eq!(stats.bytes_original, 100 * 1024);
    }

    #[tokio::test]
    async fn test_dedup_duplicate_blocks() {
        let config = BlockStoreConfig {
            path: std::env::temp_dir().join("ipfrs-test-dedup-duplicates"),
            cache_size: 1024 * 1024,
        };

        // Clean up
        let _ = std::fs::remove_dir_all(&config.path);

        let inner = SledBlockStore::new(config).expect("creating sled block store should succeed");
        // Use custom config optimized for dedup testing
        let chunk_config = ChunkingConfig {
            min_chunk_size: 32 * 1024,    // 32KB min
            target_chunk_size: 64 * 1024, // 64KB target
            max_chunk_size: 128 * 1024,   // 128KB max
            hash_mask: 0x1FFF,            // ~8KB avg for boundary detection
        };
        let store = DedupBlockStore::new(inner, chunk_config);

        // Create varied data patterns that FastCDC can chunk consistently
        // Use a repeating pattern that will create natural boundaries
        let mut chunk_data = Vec::new();
        for i in 0..40 {
            let pattern: Vec<u8> = (0..1024).map(|j| ((i * 1024 + j) % 256) as u8).collect();
            chunk_data.extend_from_slice(&pattern);
        }
        // chunk_data is now 40KB of patterned data

        let block1 = Block::new(bytes::Bytes::from(chunk_data.clone()))
            .expect("creating block1 from patterned data should succeed");

        // Create block2 with the same pattern repeated - FastCDC should find same chunks
        let mut data2 = chunk_data.clone();
        data2.extend_from_slice(&chunk_data); // 80KB total
        let block2 = Block::new(bytes::Bytes::from(data2))
            .expect("creating block2 from doubled pattern should succeed");

        // Store block1
        store
            .put(&block1)
            .await
            .expect("storing block1 should succeed");

        let stats_after_first = store.stats();
        let first_chunks = stats_after_first.unique_chunks;
        assert!(first_chunks >= 1, "Expected at least 1 chunk");

        // Store block2 - should reuse chunks from block1 where content matches
        store
            .put(&block2)
            .await
            .expect("storing block2 should succeed");

        let stats = store.stats();
        assert_eq!(stats.blocks_stored, 2);

        // With identical content patterns, block2 should reuse at least some chunks
        // The exact number depends on where FastCDC finds boundaries
        assert!(
            stats.duplicate_chunks_avoided > 0,
            "Expected some duplicate chunks to be avoided"
        );

        // Verify both blocks can be retrieved correctly
        let retrieved1 = store
            .get(block1.cid())
            .await
            .expect("get block1 should not error")
            .expect("block1 should be present after put");
        let retrieved2 = store
            .get(block2.cid())
            .await
            .expect("get block2 should not error")
            .expect("block2 should be present after put");

        assert_eq!(retrieved1.data(), block1.data());
        assert_eq!(retrieved2.data(), block2.data());
    }

    #[tokio::test]
    async fn test_dedup_delete() {
        let config = BlockStoreConfig {
            path: std::env::temp_dir().join("ipfrs-test-dedup-delete"),
            cache_size: 1024 * 1024,
        };

        // Clean up
        let _ = std::fs::remove_dir_all(&config.path);

        let inner = SledBlockStore::new(config).expect("test: SledBlockStore::new should succeed");
        let store = DedupBlockStore::with_defaults(inner);

        // Store a block
        let data = bytes::Bytes::from(vec![3u8; 200 * 1024]);
        let block = Block::new(data).expect("test: Block::new should succeed");

        store.put(&block).await.expect("test: put should succeed");

        let stats_before = store.stats();
        assert_eq!(stats_before.blocks_stored, 1);

        // Delete it
        store
            .delete(block.cid())
            .await
            .expect("test: delete should succeed");

        let stats_after = store.stats();
        assert_eq!(stats_after.blocks_stored, 0);

        // Should not be retrievable
        let retrieved = store
            .get(block.cid())
            .await
            .expect("test: get after delete should succeed");
        assert!(retrieved.is_none());
    }

    #[tokio::test]
    async fn test_dedup_reference_counting() {
        let config = BlockStoreConfig {
            path: std::env::temp_dir().join("ipfrs-test-dedup-refcount"),
            cache_size: 1024 * 1024,
        };

        // Clean up
        let _ = std::fs::remove_dir_all(&config.path);

        let inner = SledBlockStore::new(config).expect("test: SledBlockStore::new should succeed");
        let chunk_config = ChunkingConfig {
            min_chunk_size: 16 * 1024,
            target_chunk_size: 64 * 1024,
            max_chunk_size: 128 * 1024,
            hash_mask: 0xFFF,
        };
        let store = DedupBlockStore::new(inner, chunk_config);

        // Create blocks that are under min_chunk_size (will be single chunks)
        // Use patterned data to avoid issues with uniform data
        let data1: Vec<u8> = (0..10240).map(|i| (i % 256) as u8).collect(); // 10KB varied
        let data2 = data1.clone(); // Same content
        let data3: Vec<u8> = (0..10240).map(|i| ((i + 100) % 256) as u8).collect(); // 10KB different

        let block1 = Block::new(bytes::Bytes::from(data1))
            .expect("test: Block::new for data1 should succeed");
        let block2 = Block::new(bytes::Bytes::from(data2))
            .expect("test: Block::new for data2 should succeed");
        let block3 = Block::new(bytes::Bytes::from(data3))
            .expect("test: Block::new for data3 should succeed");

        // block1 and block2 have same content, so same CID
        assert_eq!(block1.cid(), block2.cid());
        // block3 is different
        assert_ne!(block1.cid(), block3.cid());

        // Store block1
        store
            .put(&block1)
            .await
            .expect("test: put block1 should succeed");
        let stats1 = store.stats();
        assert_eq!(stats1.unique_chunks, 1, "block1 should be 1 chunk");
        assert_eq!(stats1.blocks_stored, 1);

        // Store block2 (same CID as block1) - idempotent, no-op
        store
            .put(&block2)
            .await
            .expect("test: put block2 should succeed");
        let stats2 = store.stats();
        // Same CID means same data - put() is idempotent, no changes
        assert_eq!(
            stats2.unique_chunks, 1,
            "block2 is same as block1 (same CID)"
        );
        assert_eq!(stats2.blocks_stored, 1, "Still 1 block (same CID)");
        assert_eq!(
            stats2.duplicate_chunks_avoided, 0,
            "No chunking happened for duplicate CID"
        );

        // Store block3 (different) - should create new chunk
        store
            .put(&block3)
            .await
            .expect("test: put block3 should succeed");
        let stats3 = store.stats();
        assert_eq!(stats3.unique_chunks, 2, "block3 adds a new unique chunk");
        assert_eq!(stats3.blocks_stored, 2, "Now have 2 different blocks");

        // Verify retrieval
        let retrieved1 = store
            .get(block1.cid())
            .await
            .expect("test: get block1 should succeed")
            .expect("test: block1 should be present");
        assert_eq!(retrieved1.data(), block1.data());

        let retrieved3 = store
            .get(block3.cid())
            .await
            .expect("test: get block3 should succeed")
            .expect("test: block3 should be present");
        assert_eq!(retrieved3.data(), block3.data());

        // Delete block1/block2 (same CID) - should free its chunk
        store
            .delete(block1.cid())
            .await
            .expect("test: delete block1 should succeed");
        let stats_after_delete = store.stats();
        assert_eq!(
            stats_after_delete.unique_chunks, 1,
            "Only block3's chunk remains"
        );
        assert_eq!(stats_after_delete.blocks_stored, 1);

        // Delete block3 - should free remaining chunk
        store
            .delete(block3.cid())
            .await
            .expect("test: delete block3 should succeed");

        let stats_final = store.stats();
        assert_eq!(stats_final.unique_chunks, 0);
        assert_eq!(stats_final.bytes_stored, 0);
        assert_eq!(stats_final.blocks_stored, 0);
    }
}
