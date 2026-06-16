//! Utility functions for testing, benchmarking, and batch operations

use bytes::Bytes;
use ipfrs_core::{Block, Cid, Result};
use sha2::{Digest, Sha256};
use std::collections::HashMap;

/// Compute CID from raw data for testing/benchmarking
///
/// This creates a CIDv1 with SHA-256 hash and raw codec
#[inline]
pub fn compute_cid(data: &[u8]) -> Cid {
    let hash = Sha256::digest(data);

    // Create CIDv1 with SHA-256 multihash (0x12) and raw codec (0x55)
    let mut multihash = Vec::with_capacity(34);
    multihash.push(0x12); // SHA-256
    multihash.push(32); // hash length
    multihash.extend_from_slice(&hash);

    // For testing, we'll use the multihash as the CID bytes directly
    // In production, this would use the proper CID encoding
    Cid::try_from(multihash).unwrap_or_else(|_| {
        // Fallback: create a simple CID from the hash
        let cid_bytes = format!("bafkreei{}", hex::encode(&hash[..16]));
        Cid::try_from(cid_bytes.as_bytes().to_vec())
            .expect("fallback CID from hex-encoded hash is always valid")
    })
}

/// Create a block from raw data for testing
#[inline]
pub fn create_block(data: Vec<u8>) -> Result<Block> {
    let cid = compute_cid(&data);
    Ok(Block::from_parts(cid, Bytes::from(data)))
}

/// Generate random block data for testing
pub fn generate_random_block(size: usize, seed: u64) -> Vec<u8> {
    let mut rng = fastrand::Rng::with_seed(seed);
    let mut data = vec![0u8; size];

    for chunk in data.chunks_mut(8) {
        let val = rng.u64(..);
        let bytes = val.to_le_bytes();
        let len = chunk.len().min(8);
        chunk[..len].copy_from_slice(&bytes[..len]);
    }

    data
}

/// Generate compressible data for testing compression
pub fn generate_compressible_data(size: usize) -> Vec<u8> {
    let mut data = vec![0u8; size];
    let pattern = b"IPFS is a distributed file system. ";

    for (i, byte) in data.iter_mut().enumerate() {
        *byte = pattern[i % pattern.len()];
    }

    data
}

/// Generate incompressible data for testing
pub fn generate_incompressible_data(size: usize, seed: u64) -> Vec<u8> {
    generate_random_block(size, seed)
}

/// Create multiple blocks from raw data efficiently
///
/// This is optimized for batch operations, creating all blocks in parallel
/// Returns a HashMap mapping CIDs to Blocks for easy lookup
pub fn create_blocks_batch(data_vec: Vec<Vec<u8>>) -> Result<HashMap<Cid, Block>> {
    let mut blocks = HashMap::with_capacity(data_vec.len());

    for data in data_vec {
        let block = create_block(data)?;
        blocks.insert(*block.cid(), block);
    }

    Ok(blocks)
}

/// Generate multiple random blocks with sequential seeds
///
/// Returns a Vec of (Block, seed) tuples for reproducibility
pub fn generate_random_blocks(count: usize, size: usize, start_seed: u64) -> Result<Vec<Block>> {
    let mut blocks = Vec::with_capacity(count);

    for i in 0..count {
        let seed = start_seed.wrapping_add(i as u64);
        let data = generate_random_block(size, seed);
        let block = create_block(data)?;
        blocks.push(block);
    }

    Ok(blocks)
}

/// Generate multiple compressible blocks for compression testing
///
/// Each block has a different pattern to test compression efficiency
pub fn generate_compressible_blocks(count: usize, size: usize) -> Result<Vec<Block>> {
    let patterns = [
        b"IPFS is a distributed file system. ".to_vec(),
        b"Content addressing with CIDs. ".to_vec(),
        b"Merkle DAGs for data structures. ".to_vec(),
        b"Peer-to-peer networking protocol. ".to_vec(),
        b"Immutable data storage layer. ".to_vec(),
    ];

    let mut blocks = Vec::with_capacity(count);

    for i in 0..count {
        let pattern = &patterns[i % patterns.len()];
        let mut data = vec![0u8; size];

        for (j, byte) in data.iter_mut().enumerate() {
            *byte = pattern[j % pattern.len()];
        }

        let block = create_block(data)?;
        blocks.push(block);
    }

    Ok(blocks)
}

/// Create a test dataset with mixed block sizes
///
/// Returns blocks with sizes: small (1KB), medium (64KB), large (1MB)
/// Useful for testing size-dependent optimizations
pub fn generate_mixed_size_blocks(small: usize, medium: usize, large: usize) -> Result<Vec<Block>> {
    let mut blocks = Vec::with_capacity(small + medium + large);
    let mut seed = 0u64;

    // Small blocks (1KB)
    for _ in 0..small {
        let data = generate_random_block(1024, seed);
        blocks.push(create_block(data)?);
        seed = seed.wrapping_add(1);
    }

    // Medium blocks (64KB)
    for _ in 0..medium {
        let data = generate_random_block(64 * 1024, seed);
        blocks.push(create_block(data)?);
        seed = seed.wrapping_add(1);
    }

    // Large blocks (1MB)
    for _ in 0..large {
        let data = generate_random_block(1024 * 1024, seed);
        blocks.push(create_block(data)?);
        seed = seed.wrapping_add(1);
    }

    Ok(blocks)
}

/// Create blocks with controlled deduplication characteristics
///
/// - `unique`: Number of unique blocks
/// - `duplicate_factor`: How many times each block is duplicated
pub fn generate_dedup_dataset(unique: usize, duplicate_factor: usize) -> Result<Vec<Block>> {
    let mut blocks = Vec::new();

    // Generate unique blocks
    let unique_blocks = generate_random_blocks(unique, 4096, 42)?;

    // Duplicate them
    for _ in 0..duplicate_factor {
        blocks.extend(unique_blocks.iter().cloned());
    }

    Ok(blocks)
}

/// Extract CIDs from a collection of blocks
pub fn extract_cids(blocks: &[Block]) -> Vec<Cid> {
    blocks.iter().map(|b| *b.cid()).collect()
}

/// Compute total size of a block collection
pub fn compute_total_size(blocks: &[Block]) -> usize {
    blocks.iter().map(|b| b.data().len()).sum()
}

/// Group blocks by size ranges for analysis
///
/// Returns (small, medium, large) where:
/// - small: < 16KB
/// - medium: 16KB - 256KB
/// - large: > 256KB
pub fn group_blocks_by_size(blocks: &[Block]) -> (Vec<Block>, Vec<Block>, Vec<Block>) {
    let mut small = Vec::new();
    let mut medium = Vec::new();
    let mut large = Vec::new();

    for block in blocks {
        let size = block.data().len();
        if size < 16 * 1024 {
            small.push(block.clone());
        } else if size < 256 * 1024 {
            medium.push(block.clone());
        } else {
            large.push(block.clone());
        }
    }

    (small, medium, large)
}

/// Validate block integrity by recomputing CID
///
/// Returns true if the block's CID matches the computed CID from its data
pub fn validate_block_integrity(block: &Block) -> bool {
    let computed_cid = compute_cid(block.data());
    computed_cid == *block.cid()
}

/// Batch validate block integrity for multiple blocks
///
/// Returns a vector of (CID, is_valid) tuples
pub fn validate_blocks_batch(blocks: &[Block]) -> Vec<(Cid, bool)> {
    blocks
        .iter()
        .map(|block| (*block.cid(), validate_block_integrity(block)))
        .collect()
}

/// Compute statistics for a collection of blocks
#[derive(Debug, Clone)]
pub struct BlockStatistics {
    /// Total number of blocks
    pub count: usize,
    /// Total size in bytes
    pub total_size: usize,
    /// Average block size
    pub avg_size: f64,
    /// Minimum block size
    pub min_size: usize,
    /// Maximum block size
    pub max_size: usize,
    /// Median block size (approximate)
    pub median_size: usize,
}

impl BlockStatistics {
    /// Compute statistics from a block collection
    pub fn from_blocks(blocks: &[Block]) -> Self {
        if blocks.is_empty() {
            return Self {
                count: 0,
                total_size: 0,
                avg_size: 0.0,
                min_size: 0,
                max_size: 0,
                median_size: 0,
            };
        }

        let mut sizes: Vec<usize> = blocks.iter().map(|b| b.data().len()).collect();
        sizes.sort_unstable();

        let count = blocks.len();
        let total_size: usize = sizes.iter().sum();
        let avg_size = total_size as f64 / count as f64;
        let min_size = sizes[0];
        let max_size = sizes[count - 1];
        let median_size = sizes[count / 2];

        Self {
            count,
            total_size,
            avg_size,
            min_size,
            max_size,
            median_size,
        }
    }

    /// Estimate memory overhead (CID + metadata)
    pub fn estimated_memory_overhead(&self) -> usize {
        // Rough estimate: 64 bytes per block for CID and metadata
        self.count * 64
    }

    /// Total memory footprint (data + overhead)
    pub fn total_memory_footprint(&self) -> usize {
        self.total_size + self.estimated_memory_overhead()
    }
}

/// Filter blocks by size range
pub fn filter_blocks_by_size(blocks: &[Block], min_size: usize, max_size: usize) -> Vec<Block> {
    blocks
        .iter()
        .filter(|block| {
            let size = block.data().len();
            size >= min_size && size <= max_size
        })
        .cloned()
        .collect()
}

/// Sort blocks by size (ascending)
pub fn sort_blocks_by_size_asc(blocks: &mut [Block]) {
    blocks.sort_by_key(|b| b.data().len());
}

/// Sort blocks by size (descending)
pub fn sort_blocks_by_size_desc(blocks: &mut [Block]) {
    blocks.sort_by_key(|b| std::cmp::Reverse(b.data().len()));
}

/// Find duplicate blocks (same CID)
///
/// Returns a HashMap mapping CIDs to their occurrence count
pub fn find_duplicates(blocks: &[Block]) -> HashMap<Cid, usize> {
    let mut counts = HashMap::new();
    for block in blocks {
        *counts.entry(*block.cid()).or_insert(0) += 1;
    }
    counts.retain(|_, count| *count > 1);
    counts
}

/// Deduplicate blocks by CID (keep first occurrence)
pub fn deduplicate_blocks(blocks: &[Block]) -> Vec<Block> {
    let mut seen = std::collections::HashSet::new();
    let mut result = Vec::new();

    for block in blocks {
        if seen.insert(*block.cid()) {
            result.push(block.clone());
        }
    }

    result
}

/// Estimate compression ratio for a block
///
/// Returns a value between 0.0 and 1.0, where lower values indicate better compression
pub fn estimate_compression_ratio(data: &[u8]) -> f64 {
    if data.is_empty() {
        return 1.0;
    }

    // Simple entropy-based estimation
    let mut counts = [0u64; 256];
    for &byte in data {
        counts[byte as usize] += 1;
    }

    let len = data.len() as f64;
    let mut entropy = 0.0;

    for &count in &counts {
        if count > 0 {
            let p = count as f64 / len;
            entropy -= p * p.log2();
        }
    }

    // Normalize entropy to 0-1 range (8 bits max entropy)
    (entropy / 8.0).min(1.0)
}

/// Sample a subset of blocks for testing
///
/// Returns up to `count` blocks, evenly distributed across the input
pub fn sample_blocks(blocks: &[Block], count: usize) -> Vec<Block> {
    if blocks.len() <= count {
        return blocks.to_vec();
    }

    let step = blocks.len() / count;
    blocks.iter().step_by(step).take(count).cloned().collect()
}

/// Generate blocks with specific patterns for testing
///
/// Patterns: "sequential", "random", "compressible", "sparse"
pub fn generate_pattern_blocks(count: usize, size: usize, pattern: &str) -> Result<Vec<Block>> {
    match pattern {
        "sequential" => {
            let mut blocks = Vec::new();
            for i in 0..count {
                let mut data = vec![0u8; size];
                for (j, byte) in data.iter_mut().enumerate() {
                    *byte = ((i + j) % 256) as u8;
                }
                blocks.push(create_block(data)?);
            }
            Ok(blocks)
        }
        "random" => generate_random_blocks(count, size, 42),
        "compressible" => generate_compressible_blocks(count, size),
        "sparse" => {
            let mut blocks = Vec::new();
            for _ in 0..count {
                let mut data = vec![0u8; size];
                // Set only 10% of bytes to non-zero
                let mut rng = fastrand::Rng::new();
                for _ in 0..size / 10 {
                    let idx = rng.usize(..size);
                    data[idx] = rng.u8(1..);
                }
                blocks.push(create_block(data)?);
            }
            Ok(blocks)
        }
        _ => generate_random_blocks(count, size, 42), // Default to random
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_cid() {
        let data = b"hello world";
        let cid1 = compute_cid(data);
        let cid2 = compute_cid(data);

        // Same data should produce same CID
        assert_eq!(cid1, cid2);
    }

    #[test]
    fn test_create_block() {
        let data = b"hello world".to_vec();
        let block = create_block(data.clone()).unwrap();
        assert_eq!(block.data(), &data);
    }

    #[test]
    fn test_generate_random_block() {
        let block1 = generate_random_block(100, 42);
        let block2 = generate_random_block(100, 42);
        let block3 = generate_random_block(100, 43);

        assert_eq!(block1, block2); // Same seed = same data
        assert_ne!(block1, block3); // Different seed = different data
    }

    #[test]
    fn test_generate_compressible_data() {
        let data = generate_compressible_data(1000);
        assert_eq!(data.len(), 1000);

        // Check it has repeating pattern
        let pattern = b"IPFS is a distributed file system. ";
        for i in 0..10 {
            assert_eq!(data[i], pattern[i % pattern.len()]);
        }
    }

    #[test]
    fn test_create_blocks_batch() {
        let data_vec = vec![b"block1".to_vec(), b"block2".to_vec(), b"block3".to_vec()];

        let blocks = create_blocks_batch(data_vec).unwrap();
        assert_eq!(blocks.len(), 3);
    }

    #[test]
    fn test_generate_random_blocks() {
        let blocks = generate_random_blocks(10, 1024, 42).unwrap();
        assert_eq!(blocks.len(), 10);

        // All blocks should have the specified size
        for block in &blocks {
            assert_eq!(block.data().len(), 1024);
        }

        // Blocks should be unique (different seeds)
        let cid1 = blocks[0].cid();
        let cid2 = blocks[1].cid();
        assert_ne!(cid1, cid2);
    }

    #[test]
    fn test_generate_compressible_blocks() {
        let blocks = generate_compressible_blocks(5, 1024).unwrap();
        assert_eq!(blocks.len(), 5);

        for block in &blocks {
            assert_eq!(block.data().len(), 1024);
        }
    }

    #[test]
    fn test_generate_mixed_size_blocks() {
        let blocks = generate_mixed_size_blocks(2, 3, 1).unwrap();
        assert_eq!(blocks.len(), 6); // 2 + 3 + 1

        // Verify sizes
        assert_eq!(blocks[0].data().len(), 1024); // small
        assert_eq!(blocks[2].data().len(), 64 * 1024); // medium
        assert_eq!(blocks[5].data().len(), 1024 * 1024); // large
    }

    #[test]
    fn test_generate_dedup_dataset() {
        let blocks = generate_dedup_dataset(10, 3).unwrap();
        assert_eq!(blocks.len(), 30); // 10 unique * 3 duplicates

        // First 10 should match next 10 (duplicates)
        for i in 0..10 {
            assert_eq!(blocks[i].cid(), blocks[i + 10].cid());
        }
    }

    #[test]
    fn test_extract_cids() {
        let blocks = generate_random_blocks(5, 1024, 42).unwrap();
        let cids = extract_cids(&blocks);
        assert_eq!(cids.len(), 5);
        assert_eq!(cids[0], *blocks[0].cid());
    }

    #[test]
    fn test_compute_total_size() {
        let blocks = generate_mixed_size_blocks(2, 2, 1).unwrap();
        let total = compute_total_size(&blocks);

        // 2 * 1KB + 2 * 64KB + 1 * 1MB
        let expected = 2 * 1024 + 2 * 64 * 1024 + 1024 * 1024;
        assert_eq!(total, expected);
    }

    #[test]
    fn test_group_blocks_by_size() {
        let blocks = generate_mixed_size_blocks(3, 2, 1).unwrap();
        let (small, medium, large) = group_blocks_by_size(&blocks);

        assert_eq!(small.len(), 3);
        assert_eq!(medium.len(), 2);
        assert_eq!(large.len(), 1);
    }

    #[test]
    fn test_validate_block_integrity() {
        let data = b"test data".to_vec();
        let block = create_block(data).unwrap();
        assert!(validate_block_integrity(&block));
    }

    #[test]
    fn test_validate_blocks_batch() {
        let blocks = generate_random_blocks(5, 1024, 42).unwrap();
        let results = validate_blocks_batch(&blocks);
        assert_eq!(results.len(), 5);
        for (_, is_valid) in results {
            assert!(is_valid);
        }
    }

    #[test]
    fn test_block_statistics() {
        let blocks = generate_mixed_size_blocks(2, 3, 1).unwrap();
        let stats = BlockStatistics::from_blocks(&blocks);

        assert_eq!(stats.count, 6);
        assert!(stats.avg_size > 0.0);
        assert!(stats.min_size <= stats.max_size);
        assert!(stats.total_memory_footprint() > stats.total_size);
    }

    #[test]
    fn test_filter_blocks_by_size() {
        let blocks = generate_mixed_size_blocks(5, 5, 5).unwrap();
        let filtered = filter_blocks_by_size(&blocks, 2000, 100_000);

        // Should filter out 1KB blocks and keep 64KB blocks and 1MB blocks
        for block in &filtered {
            let size = block.data().len();
            assert!((2000..=100_000).contains(&size));
        }
    }

    #[test]
    fn test_sort_blocks_by_size() {
        let mut blocks = generate_mixed_size_blocks(2, 2, 2).unwrap();

        sort_blocks_by_size_asc(&mut blocks);
        for i in 1..blocks.len() {
            assert!(blocks[i - 1].data().len() <= blocks[i].data().len());
        }

        sort_blocks_by_size_desc(&mut blocks);
        for i in 1..blocks.len() {
            assert!(blocks[i - 1].data().len() >= blocks[i].data().len());
        }
    }

    #[test]
    fn test_find_duplicates() {
        let unique_blocks = generate_random_blocks(5, 1024, 42).unwrap();
        let mut all_blocks = unique_blocks.clone();
        all_blocks.extend(unique_blocks.clone()); // Add duplicates

        let duplicates = find_duplicates(&all_blocks);
        assert_eq!(duplicates.len(), 5); // All 5 blocks appear twice

        for (_, count) in duplicates {
            assert_eq!(count, 2);
        }
    }

    #[test]
    fn test_deduplicate_blocks() {
        let unique_blocks = generate_random_blocks(5, 1024, 42).unwrap();
        let mut all_blocks = unique_blocks.clone();
        all_blocks.extend(unique_blocks.clone());

        let deduped = deduplicate_blocks(&all_blocks);
        assert_eq!(deduped.len(), 5);
    }

    #[test]
    fn test_estimate_compression_ratio() {
        // Compressible data (repeating pattern)
        let compressible = generate_compressible_data(1000);
        let compressible_ratio = estimate_compression_ratio(&compressible);

        // Random data (incompressible)
        let random = generate_random_block(1000, 42);
        let random_ratio = estimate_compression_ratio(&random);

        // Compressible data should have lower entropy (better compression potential)
        assert!(compressible_ratio < random_ratio);
    }

    #[test]
    fn test_sample_blocks() {
        let blocks = generate_random_blocks(100, 1024, 42).unwrap();

        let sample = sample_blocks(&blocks, 10);
        assert_eq!(sample.len(), 10);

        // Test sampling more than available
        let sample_all = sample_blocks(&blocks, 200);
        assert_eq!(sample_all.len(), 100);
    }

    #[test]
    fn test_generate_pattern_blocks() {
        let sequential = generate_pattern_blocks(5, 1024, "sequential").unwrap();
        assert_eq!(sequential.len(), 5);

        let random = generate_pattern_blocks(5, 1024, "random").unwrap();
        assert_eq!(random.len(), 5);

        let compressible = generate_pattern_blocks(5, 1024, "compressible").unwrap();
        assert_eq!(compressible.len(), 5);

        let sparse = generate_pattern_blocks(5, 1024, "sparse").unwrap();
        assert_eq!(sparse.len(), 5);

        // Test sparse pattern has mostly zeros
        let sparse_data = sparse[0].data();
        let zero_count = sparse_data.iter().filter(|&&b| b == 0).count();
        assert!(zero_count > sparse_data.len() * 8 / 10); // At least 80% zeros
    }
}
