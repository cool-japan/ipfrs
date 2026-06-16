//! Batch processing utilities with parallel execution
//!
//! This module provides high-performance batch operations for processing
//! multiple blocks, CIDs, and hashes in parallel using Rayon.
//!
//! ## Example
//!
//! ```rust
//! use ipfrs_core::batch::BatchProcessor;
//! use bytes::Bytes;
//!
//! let data_chunks = vec![
//!     Bytes::from("chunk 1"),
//!     Bytes::from("chunk 2"),
//!     Bytes::from("chunk 3"),
//! ];
//!
//! // Create blocks in parallel
//! let processor = BatchProcessor::new();
//! let blocks = processor.create_blocks_parallel(data_chunks).unwrap();
//! assert_eq!(blocks.len(), 3);
//! ```

use crate::error::{Error, Result};
use crate::hash::global_hash_registry;
use crate::{compress, compression_ratio, decompress, Block, BlockBuilder, Cid, CidBuilder};
use crate::{CompressionAlgorithm, HashAlgorithm};
use bytes::Bytes;
use rayon::prelude::*;

/// High-performance batch processor for parallel operations
///
/// Provides parallel processing of multiple blocks, CIDs, and hash computations
/// using Rayon's thread pool.
pub struct BatchProcessor {
    hash_algorithm: HashAlgorithm,
}

impl BatchProcessor {
    /// Create a new batch processor with default settings (SHA2-256)
    pub fn new() -> Self {
        Self {
            hash_algorithm: HashAlgorithm::Sha256,
        }
    }

    /// Create a batch processor with a specific hash algorithm
    pub fn with_hash_algorithm(hash_algorithm: HashAlgorithm) -> Self {
        Self { hash_algorithm }
    }

    /// Create multiple blocks in parallel from data chunks
    ///
    /// This is significantly faster than creating blocks sequentially
    /// when processing many chunks.
    pub fn create_blocks_parallel(&self, data_chunks: Vec<Bytes>) -> Result<Vec<Block>> {
        let hash_algo = self.hash_algorithm;

        data_chunks
            .into_par_iter()
            .map(|data| BlockBuilder::new().hash_algorithm(hash_algo).build(data))
            .collect()
    }

    /// Generate CIDs in parallel for multiple data chunks
    ///
    /// Returns a vector of (data, CID) pairs.
    pub fn generate_cids_parallel(&self, data_chunks: Vec<Bytes>) -> Result<Vec<(Bytes, Cid)>> {
        let hash_algo = self.hash_algorithm;

        data_chunks
            .into_par_iter()
            .map(|data| {
                let cid = CidBuilder::new().hash_algorithm(hash_algo).build(&data)?;
                Ok((data, cid))
            })
            .collect()
    }

    /// Verify multiple blocks in parallel
    ///
    /// Returns `Ok(())` if all blocks are valid, or an error for the first invalid block.
    pub fn verify_blocks_parallel(&self, blocks: &[Block]) -> Result<()> {
        let all_valid: Result<bool> = blocks
            .par_iter()
            .try_fold(
                || true,
                |acc, block| -> Result<bool> {
                    let valid = block.verify()?;
                    Ok(acc && valid)
                },
            )
            .try_reduce(|| true, |a, b| -> Result<bool> { Ok(a && b) });

        if all_valid? {
            Ok(())
        } else {
            Err(Error::Verification(
                "One or more blocks failed verification".into(),
            ))
        }
    }

    /// Compute hashes in parallel for multiple data chunks
    ///
    /// Returns a vector of hash digests.
    pub fn compute_hashes_parallel(&self, data_chunks: &[&[u8]]) -> Result<Vec<Vec<u8>>> {
        let code = self.hash_algorithm.code();

        let registry = global_hash_registry();
        let engine = registry.get(code).ok_or_else(|| {
            Error::InvalidInput(format!(
                "Hash algorithm {} not supported",
                self.hash_algorithm.name()
            ))
        })?;

        Ok(data_chunks
            .par_iter()
            .map(|data| engine.digest(data))
            .collect())
    }

    /// Count total bytes across multiple blocks in parallel
    pub fn total_bytes_parallel(&self, blocks: &[Block]) -> usize {
        blocks.par_iter().map(|block| block.data().len()).sum()
    }

    /// Find blocks matching a predicate in parallel
    pub fn filter_blocks_parallel<F>(&self, blocks: Vec<Block>, predicate: F) -> Vec<Block>
    where
        F: Fn(&Block) -> bool + Sync + Send,
    {
        blocks
            .into_par_iter()
            .filter(|block| predicate(block))
            .collect()
    }

    /// Collect unique CIDs from blocks in parallel
    pub fn unique_cids_parallel(&self, blocks: &[Block]) -> Vec<Cid> {
        use std::collections::HashSet;
        use std::sync::Mutex;

        let seen = Mutex::new(HashSet::new());
        let unique: Vec<Cid> = blocks
            .par_iter()
            .filter_map(|block| {
                let cid = *block.cid();
                let mut seen = seen.lock().unwrap_or_else(|e| e.into_inner());
                if seen.insert(cid.to_string()) {
                    Some(cid)
                } else {
                    None
                }
            })
            .collect();

        unique
    }

    /// Compress multiple data chunks in parallel
    ///
    /// Compresses each data chunk using the specified algorithm and level.
    /// Returns a vector of compressed data, maintaining the same order as input.
    ///
    /// # Arguments
    ///
    /// * `data_chunks` - Vector of data chunks to compress
    /// * `algorithm` - Compression algorithm to use
    /// * `level` - Compression level (0-9)
    ///
    /// # Returns
    ///
    /// Vector of compressed data chunks
    ///
    /// # Example
    ///
    /// ```rust
    /// use ipfrs_core::batch::BatchProcessor;
    /// use ipfrs_core::CompressionAlgorithm;
    /// use bytes::Bytes;
    ///
    /// let processor = BatchProcessor::new();
    /// let data = vec![
    ///     Bytes::from(vec![0u8; 1000]),
    ///     Bytes::from(vec![1u8; 1000]),
    /// ];
    ///
    /// let compressed = processor.compress_data_parallel(
    ///     data,
    ///     CompressionAlgorithm::Zstd,
    ///     3
    /// ).unwrap();
    /// assert_eq!(compressed.len(), 2);
    /// ```
    pub fn compress_data_parallel(
        &self,
        data_chunks: Vec<Bytes>,
        algorithm: CompressionAlgorithm,
        level: u8,
    ) -> Result<Vec<Bytes>> {
        data_chunks
            .into_par_iter()
            .map(|data| compress(&data, algorithm, level))
            .collect()
    }

    /// Decompress multiple compressed chunks in parallel
    ///
    /// Decompresses each chunk using the specified algorithm.
    /// Returns a vector of decompressed data, maintaining the same order as input.
    ///
    /// # Arguments
    ///
    /// * `compressed_chunks` - Vector of compressed data chunks
    /// * `algorithm` - Compression algorithm that was used
    ///
    /// # Returns
    ///
    /// Vector of decompressed data chunks
    ///
    /// # Example
    ///
    /// ```rust
    /// use ipfrs_core::batch::BatchProcessor;
    /// use ipfrs_core::CompressionAlgorithm;
    /// use bytes::Bytes;
    ///
    /// let processor = BatchProcessor::new();
    /// let data = vec![Bytes::from(vec![0u8; 1000])];
    ///
    /// let compressed = processor.compress_data_parallel(
    ///     data.clone(),
    ///     CompressionAlgorithm::Lz4,
    ///     3
    /// ).unwrap();
    ///
    /// let decompressed = processor.decompress_data_parallel(
    ///     compressed,
    ///     CompressionAlgorithm::Lz4
    /// ).unwrap();
    /// assert_eq!(decompressed, data);
    /// ```
    pub fn decompress_data_parallel(
        &self,
        compressed_chunks: Vec<Bytes>,
        algorithm: CompressionAlgorithm,
    ) -> Result<Vec<Bytes>> {
        compressed_chunks
            .into_par_iter()
            .map(|data| decompress(&data, algorithm))
            .collect()
    }

    /// Analyze compression ratios for multiple data chunks in parallel
    ///
    /// Computes compression ratio estimates for each chunk.
    /// Returns a vector of ratios (compressed_size / original_size), where lower is better.
    ///
    /// # Arguments
    ///
    /// * `data_chunks` - Vector of data chunks to analyze
    /// * `algorithm` - Compression algorithm to use for estimation
    /// * `level` - Compression level (0-9)
    ///
    /// # Returns
    ///
    /// Vector of compression ratios (0.0 to 1.0, where 0.5 means 50% size reduction)
    ///
    /// # Example
    ///
    /// ```rust
    /// use ipfrs_core::batch::BatchProcessor;
    /// use ipfrs_core::CompressionAlgorithm;
    /// use bytes::Bytes;
    ///
    /// let processor = BatchProcessor::new();
    /// let data = vec![
    ///     Bytes::from(vec![0u8; 1000]), // Highly compressible
    ///     Bytes::from(vec![1u8; 1000]), // Highly compressible
    /// ];
    ///
    /// let ratios = processor.analyze_compression_ratios_parallel(
    ///     &data,
    ///     CompressionAlgorithm::Zstd,
    ///     3
    /// ).unwrap();
    /// assert_eq!(ratios.len(), 2);
    /// // Repetitive data should compress well (ratio < 0.5)
    /// assert!(ratios[0] < 0.5);
    /// ```
    pub fn analyze_compression_ratios_parallel(
        &self,
        data_chunks: &[Bytes],
        algorithm: CompressionAlgorithm,
        level: u8,
    ) -> Result<Vec<f64>> {
        data_chunks
            .par_iter()
            .map(|data| compression_ratio(data, algorithm, level))
            .collect()
    }
}

impl Default for BatchProcessor {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for batch operations
#[derive(Debug, Clone, PartialEq)]
pub struct BatchStats {
    /// Number of items processed
    pub items_processed: usize,
    /// Total bytes processed
    pub total_bytes: usize,
    /// Number of unique CIDs
    pub unique_cids: usize,
    /// Number of failed items
    pub failed_items: usize,
    /// Total bytes after compression (0 if not compressed)
    pub compressed_bytes: usize,
    /// Average compression ratio (0.0 if not compressed)
    pub avg_compression_ratio: f64,
}

impl BatchStats {
    /// Create new batch statistics
    pub fn new() -> Self {
        Self {
            items_processed: 0,
            total_bytes: 0,
            unique_cids: 0,
            failed_items: 0,
            compressed_bytes: 0,
            avg_compression_ratio: 0.0,
        }
    }

    /// Calculate deduplication ratio (0.0 = no dedup, 1.0 = all duplicates)
    pub fn dedup_ratio(&self) -> f64 {
        if self.items_processed == 0 {
            return 0.0;
        }
        1.0 - (self.unique_cids as f64 / self.items_processed as f64)
    }

    /// Calculate success rate (0.0 to 1.0)
    pub fn success_rate(&self) -> f64 {
        if self.items_processed == 0 {
            return 1.0;
        }
        let successful = self.items_processed - self.failed_items;
        successful as f64 / self.items_processed as f64
    }

    /// Calculate compression savings in bytes
    ///
    /// Returns the number of bytes saved by compression.
    /// Positive values indicate compression saved space.
    pub fn compression_savings(&self) -> i64 {
        if self.compressed_bytes == 0 {
            return 0;
        }
        self.total_bytes as i64 - self.compressed_bytes as i64
    }

    /// Calculate compression efficiency percentage (0.0 to 100.0)
    ///
    /// Returns the percentage of space saved by compression.
    /// For example, 50.0 means the compressed data is 50% smaller.
    pub fn compression_efficiency(&self) -> f64 {
        if self.total_bytes == 0 || self.compressed_bytes == 0 {
            return 0.0;
        }
        (1.0 - (self.compressed_bytes as f64 / self.total_bytes as f64)) * 100.0
    }
}

impl Default for BatchStats {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_blocks_parallel() {
        let processor = BatchProcessor::new();
        let chunks = vec![
            Bytes::from("chunk 1"),
            Bytes::from("chunk 2"),
            Bytes::from("chunk 3"),
        ];

        let blocks = processor.create_blocks_parallel(chunks).unwrap();
        assert_eq!(blocks.len(), 3);

        // Verify all blocks are valid
        for block in &blocks {
            assert!(block.verify().is_ok());
        }
    }

    #[test]
    fn test_generate_cids_parallel() {
        let processor = BatchProcessor::new();
        let chunks = vec![
            Bytes::from("data 1"),
            Bytes::from("data 2"),
            Bytes::from("data 3"),
        ];

        let results = processor.generate_cids_parallel(chunks.clone()).unwrap();
        assert_eq!(results.len(), 3);

        // Verify data matches
        for (i, (data, _cid)) in results.iter().enumerate() {
            assert_eq!(data, &chunks[i]);
        }
    }

    #[test]
    fn test_verify_blocks_parallel() {
        let processor = BatchProcessor::new();
        let chunks = vec![Bytes::from("test 1"), Bytes::from("test 2")];

        let blocks = processor.create_blocks_parallel(chunks).unwrap();
        assert!(processor.verify_blocks_parallel(&blocks).is_ok());
    }

    #[test]
    fn test_compute_hashes_parallel() {
        let processor = BatchProcessor::new();
        let data: Vec<&[u8]> = vec![b"hash1", b"hash2", b"hash3"];

        let hashes = processor.compute_hashes_parallel(&data).unwrap();
        assert_eq!(hashes.len(), 3);

        // All hashes should be 32 bytes (SHA-256)
        for hash in &hashes {
            assert_eq!(hash.len(), 32);
        }

        // Same input should produce same hash
        let hashes2 = processor.compute_hashes_parallel(&data).unwrap();
        assert_eq!(hashes, hashes2);
    }

    #[test]
    fn test_total_bytes_parallel() {
        let processor = BatchProcessor::new();
        let chunks = vec![
            Bytes::from("12345"),      // 5 bytes
            Bytes::from("1234567890"), // 10 bytes
            Bytes::from("123"),        // 3 bytes
        ];

        let blocks = processor.create_blocks_parallel(chunks).unwrap();
        let total = processor.total_bytes_parallel(&blocks);
        assert_eq!(total, 18);
    }

    #[test]
    fn test_filter_blocks_parallel() {
        let processor = BatchProcessor::new();
        let chunks = vec![
            Bytes::from("short"),
            Bytes::from("this is a longer chunk"),
            Bytes::from("tiny"),
        ];

        let blocks = processor.create_blocks_parallel(chunks).unwrap();

        // Filter blocks with data length > 10
        let filtered = processor.filter_blocks_parallel(blocks, |block| block.data().len() > 10);

        assert_eq!(filtered.len(), 1);
        assert!(filtered[0].data().len() > 10);
    }

    #[test]
    fn test_unique_cids_parallel() {
        let processor = BatchProcessor::new();
        let chunks = vec![
            Bytes::from("unique1"),
            Bytes::from("unique2"),
            Bytes::from("unique1"), // duplicate
            Bytes::from("unique3"),
        ];

        let blocks = processor.create_blocks_parallel(chunks).unwrap();
        let unique = processor.unique_cids_parallel(&blocks);

        assert_eq!(unique.len(), 3); // 3 unique CIDs
    }

    #[test]
    fn test_batch_stats() {
        let mut stats = BatchStats::new();
        assert_eq!(stats.dedup_ratio(), 0.0);
        assert_eq!(stats.success_rate(), 1.0);

        stats.items_processed = 10;
        stats.unique_cids = 7;
        stats.failed_items = 1;

        // Use approximate comparison for floating point
        assert!((stats.dedup_ratio() - 0.3).abs() < 0.0001);
        assert!((stats.success_rate() - 0.9).abs() < 0.0001);
    }

    #[test]
    fn test_with_different_hash_algorithms() {
        let processor_sha256 = BatchProcessor::with_hash_algorithm(HashAlgorithm::Sha256);
        let processor_sha3 = BatchProcessor::with_hash_algorithm(HashAlgorithm::Sha3_256);

        let data = vec![Bytes::from("test data")];

        let blocks_sha256 = processor_sha256
            .create_blocks_parallel(data.clone())
            .unwrap();
        let blocks_sha3 = processor_sha3.create_blocks_parallel(data).unwrap();

        // Different hash algorithms should produce different CIDs
        assert_ne!(blocks_sha256[0].cid(), blocks_sha3[0].cid());
    }

    #[test]
    fn test_large_batch_performance() {
        let processor = BatchProcessor::new();

        // Create 1000 small chunks
        let chunks: Vec<Bytes> = (0..1000)
            .map(|i| Bytes::from(format!("chunk {}", i)))
            .collect();

        let blocks = processor.create_blocks_parallel(chunks).unwrap();
        assert_eq!(blocks.len(), 1000);

        // Verify all in parallel
        assert!(processor.verify_blocks_parallel(&blocks).is_ok());
    }

    #[test]
    fn test_empty_batch() {
        let processor = BatchProcessor::new();
        let empty: Vec<Bytes> = vec![];

        let blocks = processor.create_blocks_parallel(empty).unwrap();
        assert_eq!(blocks.len(), 0);
    }

    #[test]
    fn test_compress_data_parallel() {
        let processor = BatchProcessor::new();
        let data = vec![
            Bytes::from(vec![0u8; 1000]),
            Bytes::from(vec![1u8; 1000]),
            Bytes::from(vec![2u8; 1000]),
        ];

        // Test Zstd compression
        let compressed = processor
            .compress_data_parallel(data.clone(), CompressionAlgorithm::Zstd, 3)
            .unwrap();
        assert_eq!(compressed.len(), 3);

        // Compressed data should be smaller than original for repetitive data
        for (i, comp) in compressed.iter().enumerate() {
            assert!(comp.len() < data[i].len());
        }
    }

    #[test]
    fn test_decompress_data_parallel() {
        let processor = BatchProcessor::new();
        let original = vec![Bytes::from(vec![0u8; 500]), Bytes::from(vec![1u8; 500])];

        // Compress then decompress
        let compressed = processor
            .compress_data_parallel(original.clone(), CompressionAlgorithm::Lz4, 3)
            .unwrap();

        let decompressed = processor
            .decompress_data_parallel(compressed, CompressionAlgorithm::Lz4)
            .unwrap();

        assert_eq!(decompressed.len(), original.len());
        for (i, decomp) in decompressed.iter().enumerate() {
            assert_eq!(decomp, &original[i]);
        }
    }

    #[test]
    fn test_analyze_compression_ratios_parallel() {
        let processor = BatchProcessor::new();
        let data = vec![
            Bytes::from(vec![0u8; 1000]), // Highly compressible
            Bytes::from(vec![1u8; 1000]), // Highly compressible
        ];

        let ratios = processor
            .analyze_compression_ratios_parallel(&data, CompressionAlgorithm::Zstd, 6)
            .unwrap();

        assert_eq!(ratios.len(), 2);

        // Ratios should be between 0.0 and 1.0
        for ratio in &ratios {
            assert!(*ratio >= 0.0 && *ratio <= 1.0);
        }

        // Repetitive data should have good compression ratio (< 0.5)
        for ratio in &ratios {
            assert!(*ratio < 0.5);
        }
    }

    #[test]
    fn test_compression_with_none_algorithm() {
        let processor = BatchProcessor::new();
        let data = vec![Bytes::from("test data"), Bytes::from("more data")];

        // None algorithm should return data unchanged
        let compressed = processor
            .compress_data_parallel(data.clone(), CompressionAlgorithm::None, 0)
            .unwrap();

        assert_eq!(compressed.len(), data.len());
        for (i, comp) in compressed.iter().enumerate() {
            assert_eq!(comp, &data[i]);
        }
    }

    #[test]
    fn test_batch_stats_compression() {
        let mut stats = BatchStats::new();
        stats.items_processed = 100;
        stats.total_bytes = 10000;
        stats.compressed_bytes = 5000;

        // Test compression savings
        assert_eq!(stats.compression_savings(), 5000);

        // Test compression efficiency (50% reduction)
        assert!((stats.compression_efficiency() - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_batch_stats_no_compression() {
        let stats = BatchStats::new();

        // With no compression data, savings should be 0
        assert_eq!(stats.compression_savings(), 0);
        assert_eq!(stats.compression_efficiency(), 0.0);
    }

    #[test]
    fn test_large_batch_compression() {
        let processor = BatchProcessor::new();

        // Create 100 chunks of compressible data
        let data: Vec<Bytes> = (0..100).map(|i| Bytes::from(vec![i as u8; 500])).collect();

        let compressed = processor
            .compress_data_parallel(data.clone(), CompressionAlgorithm::Zstd, 3)
            .unwrap();

        assert_eq!(compressed.len(), 100);

        // Verify roundtrip
        let decompressed = processor
            .decompress_data_parallel(compressed, CompressionAlgorithm::Zstd)
            .unwrap();

        assert_eq!(decompressed, data);
    }

    #[test]
    fn test_empty_compression_batch() {
        let processor = BatchProcessor::new();
        let empty: Vec<Bytes> = vec![];

        let compressed = processor
            .compress_data_parallel(empty, CompressionAlgorithm::Lz4, 3)
            .unwrap();

        assert_eq!(compressed.len(), 0);
    }
}
