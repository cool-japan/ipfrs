//! Utility functions for common IPFRS operations.
//!
//! This module provides convenience functions that simplify common tasks
//! when working with blocks, CIDs, and IPLD data.
//!
//! # Examples
//!
//! ```rust
//! use ipfrs_core::utils;
//! use bytes::Bytes;
//!
//! // Quick block creation with default settings
//! let block = utils::quick_block(b"Hello, World!").unwrap();
//! println!("CID: {}", block.cid());
//!
//! // Parse CID from string
//! let cid = utils::parse_cid_string("QmXXX...").ok();
//! ```

use crate::{Block, Cid, CidBuilder, HashAlgorithm, Ipld, Result};
use bytes::Bytes;
use std::collections::BTreeMap;

/// Creates a block from a byte slice using default settings (SHA2-256, CIDv1, raw codec).
///
/// This is a convenience function that combines data conversion and block creation.
///
/// # Examples
///
/// ```rust
/// use ipfrs_core::utils::quick_block;
///
/// let block = quick_block(b"Hello, IPFRS!").unwrap();
/// assert_eq!(block.data().as_ref(), b"Hello, IPFRS!");
/// ```
pub fn quick_block(data: &[u8]) -> Result<Block> {
    Block::new(Bytes::copy_from_slice(data))
}

/// Creates a block with a specific hash algorithm.
///
/// # Examples
///
/// ```rust
/// use ipfrs_core::{utils::block_with_hash, HashAlgorithm};
///
/// let block = block_with_hash(b"data", HashAlgorithm::Sha3_256).unwrap();
/// ```
pub fn block_with_hash(data: &[u8], algorithm: HashAlgorithm) -> Result<Block> {
    crate::BlockBuilder::new()
        .hash_algorithm(algorithm)
        .build_from_slice(data)
}

/// Parses a CID from a string with automatic multibase detection.
///
/// # Examples
///
/// ```rust
/// use ipfrs_core::utils::parse_cid_string;
///
/// // Parse CIDv0 (base58btc)
/// let cid_v0 = parse_cid_string("QmXXX...");
///
/// // Parse CIDv1 (base32)
/// let cid_v1 = parse_cid_string("bafyXXX...");
/// ```
pub fn parse_cid_string(s: &str) -> Result<Cid> {
    crate::cid::parse_cid(s)
}

/// Computes the CID of data using the specified hash algorithm.
///
/// # Examples
///
/// ```rust
/// use ipfrs_core::{utils::cid_of, HashAlgorithm};
///
/// let cid = cid_of(b"Hello, World!", HashAlgorithm::Sha256).unwrap();
/// ```
pub fn cid_of(data: &[u8], algorithm: HashAlgorithm) -> Result<Cid> {
    CidBuilder::new().hash_algorithm(algorithm).build(data)
}

/// Computes a SHA2-256 CID (most common).
///
/// # Examples
///
/// ```rust
/// use ipfrs_core::utils::sha256_cid;
///
/// let cid = sha256_cid(b"Hello, World!").unwrap();
/// ```
pub fn sha256_cid(data: &[u8]) -> Result<Cid> {
    cid_of(data, HashAlgorithm::Sha256)
}

/// Computes a SHA3-256 CID.
///
/// # Examples
///
/// ```rust
/// use ipfrs_core::utils::sha3_cid;
///
/// let cid = sha3_cid(b"Hello, World!").unwrap();
/// ```
pub fn sha3_cid(data: &[u8]) -> Result<Cid> {
    cid_of(data, HashAlgorithm::Sha3_256)
}

/// Computes a SHA2-512 CID (64-byte hash).
///
/// # Examples
///
/// ```rust
/// use ipfrs_core::utils::sha512_cid;
///
/// let cid = sha512_cid(b"Hello, World!").unwrap();
/// ```
pub fn sha512_cid(data: &[u8]) -> Result<Cid> {
    cid_of(data, HashAlgorithm::Sha512)
}

/// Computes a SHA3-512 CID (64-byte Keccak hash).
///
/// # Examples
///
/// ```rust
/// use ipfrs_core::utils::sha3_512_cid;
///
/// let cid = sha3_512_cid(b"Hello, World!").unwrap();
/// ```
pub fn sha3_512_cid(data: &[u8]) -> Result<Cid> {
    cid_of(data, HashAlgorithm::Sha3_512)
}

/// Computes a BLAKE2b-256 CID (fast, 32-byte hash).
///
/// # Examples
///
/// ```rust
/// use ipfrs_core::utils::blake2b256_cid;
///
/// let cid = blake2b256_cid(b"Hello, World!").unwrap();
/// ```
pub fn blake2b256_cid(data: &[u8]) -> Result<Cid> {
    cid_of(data, HashAlgorithm::Blake2b256)
}

/// Computes a BLAKE2b-512 CID (fast, 64-byte hash).
///
/// # Examples
///
/// ```rust
/// use ipfrs_core::utils::blake2b512_cid;
///
/// let cid = blake2b512_cid(b"Hello, World!").unwrap();
/// ```
pub fn blake2b512_cid(data: &[u8]) -> Result<Cid> {
    cid_of(data, HashAlgorithm::Blake2b512)
}

/// Computes a BLAKE2s-256 CID (fast, optimized for 8-32 bit platforms).
///
/// # Examples
///
/// ```rust
/// use ipfrs_core::utils::blake2s256_cid;
///
/// let cid = blake2s256_cid(b"Hello, World!").unwrap();
/// ```
pub fn blake2s256_cid(data: &[u8]) -> Result<Cid> {
    cid_of(data, HashAlgorithm::Blake2s256)
}

/// Computes a BLAKE3 CID (fastest, modern cryptographic design).
///
/// # Examples
///
/// ```rust
/// use ipfrs_core::utils::blake3_cid;
///
/// let cid = blake3_cid(b"Hello, World!").unwrap();
/// ```
pub fn blake3_cid(data: &[u8]) -> Result<Cid> {
    cid_of(data, HashAlgorithm::Blake3)
}

/// Checks if two blocks have the same CID (content equality).
///
/// # Examples
///
/// ```rust
/// use ipfrs_core::{utils::quick_block, utils::blocks_equal};
///
/// let block1 = quick_block(b"data").unwrap();
/// let block2 = quick_block(b"data").unwrap();
/// assert!(blocks_equal(&block1, &block2));
/// ```
pub fn blocks_equal(a: &Block, b: &Block) -> bool {
    a.cid() == b.cid()
}

/// Verifies that a block's CID matches its content.
///
/// Returns true if the block is valid, false otherwise.
///
/// # Examples
///
/// ```rust
/// use ipfrs_core::utils::{quick_block, verify_block};
///
/// let block = quick_block(b"Hello").unwrap();
/// assert!(verify_block(&block).unwrap());
/// ```
pub fn verify_block(block: &Block) -> Result<bool> {
    block.verify()
}

/// Creates an IPLD map from key-value pairs.
///
/// # Examples
///
/// ```rust
/// use ipfrs_core::{utils::ipld_map, Ipld};
///
/// let map = ipld_map(vec![
///     ("name", Ipld::String("Alice".to_string())),
///     ("age", Ipld::Integer(30)),
/// ]);
/// ```
pub fn ipld_map<K: Into<String>>(pairs: Vec<(K, Ipld)>) -> Ipld {
    let mut map = BTreeMap::new();
    for (k, v) in pairs {
        map.insert(k.into(), v);
    }
    Ipld::Map(map)
}

/// Creates an IPLD list from values.
///
/// # Examples
///
/// ```rust
/// use ipfrs_core::{utils::ipld_list, Ipld};
///
/// let list = ipld_list(vec![
///     Ipld::Integer(1),
///     Ipld::Integer(2),
///     Ipld::Integer(3),
/// ]);
/// ```
pub fn ipld_list(values: Vec<Ipld>) -> Ipld {
    Ipld::List(values)
}

/// Encodes IPLD data to DAG-CBOR bytes.
///
/// # Examples
///
/// ```rust
/// use ipfrs_core::{utils::ipld_to_cbor, Ipld};
///
/// let ipld = Ipld::String("hello".to_string());
/// let cbor = ipld_to_cbor(&ipld).unwrap();
/// ```
pub fn ipld_to_cbor(ipld: &Ipld) -> Result<Vec<u8>> {
    ipld.to_dag_cbor()
}

/// Decodes IPLD data from DAG-CBOR bytes.
///
/// # Examples
///
/// ```rust
/// use ipfrs_core::{utils::{ipld_to_cbor, ipld_from_cbor}, Ipld};
///
/// let ipld = Ipld::String("hello".to_string());
/// let cbor = ipld_to_cbor(&ipld).unwrap();
/// let decoded = ipld_from_cbor(&cbor).unwrap();
/// assert_eq!(ipld, decoded);
/// ```
pub fn ipld_from_cbor(data: &[u8]) -> Result<Ipld> {
    Ipld::from_dag_cbor(data)
}

/// Encodes IPLD data to DAG-JSON string.
///
/// # Examples
///
/// ```rust
/// use ipfrs_core::{utils::ipld_to_json, Ipld};
///
/// let ipld = Ipld::String("hello".to_string());
/// let json = ipld_to_json(&ipld).unwrap();
/// ```
pub fn ipld_to_json(ipld: &Ipld) -> Result<String> {
    ipld.to_dag_json()
}

/// Decodes IPLD data from DAG-JSON string.
///
/// # Examples
///
/// ```rust
/// use ipfrs_core::{utils::{ipld_to_json, ipld_from_json}, Ipld};
///
/// let ipld = Ipld::String("hello".to_string());
/// let json = ipld_to_json(&ipld).unwrap();
/// let decoded = ipld_from_json(&json).unwrap();
/// assert_eq!(ipld, decoded);
/// ```
pub fn ipld_from_json(data: &str) -> Result<Ipld> {
    Ipld::from_dag_json(data)
}

/// Formats a block size in human-readable format (KB, MB, GB).
///
/// # Examples
///
/// ```rust
/// use ipfrs_core::utils::format_size;
///
/// assert_eq!(format_size(1024), "1.00 KB");
/// assert_eq!(format_size(1_048_576), "1.00 MB");
/// assert_eq!(format_size(1_073_741_824), "1.00 GB");
/// ```
pub fn format_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}

/// Estimates the number of chunks needed for data of the given size.
///
/// Uses the default chunk size (256 KB).
///
/// # Examples
///
/// ```rust
/// use ipfrs_core::utils::estimate_chunks;
///
/// assert_eq!(estimate_chunks(1_000_000), 4); // ~1 MB → 4 chunks
/// ```
pub fn estimate_chunks(data_size: u64) -> usize {
    const DEFAULT_CHUNK_SIZE: u64 = 256 * 1024; // 256 KB
    data_size.div_ceil(DEFAULT_CHUNK_SIZE) as usize
}

/// Checks if data needs chunking based on the maximum block size.
///
/// # Examples
///
/// ```rust
/// use ipfrs_core::utils::needs_chunking;
///
/// assert!(!needs_chunking(100)); // Small data
/// assert!(needs_chunking(3_000_000)); // Large data (> 2 MiB)
/// ```
pub fn needs_chunking(data_size: u64) -> bool {
    data_size > crate::MAX_BLOCK_SIZE as u64
}

//
// Diagnostic and Validation Utilities
//

/// Information about a CID for diagnostic purposes.
#[derive(Debug, Clone)]
pub struct CidInfo {
    /// CID string representation
    pub cid_string: String,
    /// CID version (0 or 1)
    pub version: u8,
    /// Codec identifier
    pub codec: u64,
    /// Hash algorithm code
    pub hash_code: u64,
    /// Hash digest length in bytes
    pub hash_length: usize,
}

impl std::fmt::Display for CidInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "CID: {}\n  Version: {}\n  Codec: 0x{:x}\n  Hash: 0x{:x} ({} bytes)",
            self.cid_string, self.version, self.codec, self.hash_code, self.hash_length
        )
    }
}

/// Inspects a CID and returns detailed diagnostic information.
///
/// # Examples
///
/// ```rust
/// use ipfrs_core::utils::{sha256_cid, inspect_cid};
///
/// let cid = sha256_cid(b"Hello").unwrap();
/// let info = inspect_cid(&cid);
/// println!("{}", info);
/// ```
pub fn inspect_cid(cid: &Cid) -> CidInfo {
    CidInfo {
        cid_string: cid.to_string(),
        version: match cid.version() {
            cid::Version::V0 => 0,
            cid::Version::V1 => 1,
        },
        codec: cid.codec(),
        hash_code: cid.hash().code(),
        hash_length: cid.hash().digest().len(),
    }
}

/// Information about a block for diagnostic purposes.
#[derive(Debug, Clone)]
pub struct BlockInfo {
    /// Block's CID
    pub cid: String,
    /// Block size in bytes
    pub size: u64,
    /// Human-readable size
    pub size_formatted: String,
    /// Whether the block is valid (CID matches content)
    pub is_valid: bool,
}

impl std::fmt::Display for BlockInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Block:\n  CID: {}\n  Size: {} ({})\n  Valid: {}",
            self.cid, self.size, self.size_formatted, self.is_valid
        )
    }
}

/// Inspects a block and returns detailed diagnostic information.
///
/// # Examples
///
/// ```rust
/// use ipfrs_core::utils::{quick_block, inspect_block};
///
/// let block = quick_block(b"Hello, World!").unwrap();
/// let info = inspect_block(&block).unwrap();
/// println!("{}", info);
/// ```
pub fn inspect_block(block: &Block) -> Result<BlockInfo> {
    let is_valid = block.verify()?;
    Ok(BlockInfo {
        cid: block.cid().to_string(),
        size: block.size(),
        size_formatted: format_size(block.size()),
        is_valid,
    })
}

/// Validates that a string is a valid CID.
///
/// # Examples
///
/// ```rust
/// use ipfrs_core::utils::validate_cid_string;
///
/// assert!(validate_cid_string("QmXXX").is_ok() || validate_cid_string("QmXXX").is_err());
/// ```
pub fn validate_cid_string(s: &str) -> Result<Cid> {
    parse_cid_string(s)
}

/// Validates a collection of blocks, returning the number of valid and invalid blocks.
///
/// # Examples
///
/// ```rust
/// use ipfrs_core::utils::{quick_block, validate_blocks};
///
/// let blocks = vec![
///     quick_block(b"data1").unwrap(),
///     quick_block(b"data2").unwrap(),
/// ];
/// let (valid, invalid) = validate_blocks(&blocks).unwrap();
/// assert_eq!(valid, 2);
/// assert_eq!(invalid, 0);
/// ```
pub fn validate_blocks(blocks: &[Block]) -> Result<(usize, usize)> {
    let mut valid = 0;
    let mut invalid = 0;

    for block in blocks {
        if block.verify()? {
            valid += 1;
        } else {
            invalid += 1;
        }
    }

    Ok((valid, invalid))
}

/// Finds blocks in a collection that have invalid CIDs (mismatched content).
///
/// # Examples
///
/// ```rust
/// use ipfrs_core::utils::{quick_block, find_invalid_blocks};
///
/// let blocks = vec![
///     quick_block(b"data1").unwrap(),
///     quick_block(b"data2").unwrap(),
/// ];
/// let invalid = find_invalid_blocks(&blocks).unwrap();
/// assert_eq!(invalid.len(), 0);
/// ```
pub fn find_invalid_blocks(blocks: &[Block]) -> Result<Vec<usize>> {
    let mut invalid_indices = Vec::new();

    for (i, block) in blocks.iter().enumerate() {
        if !block.verify()? {
            invalid_indices.push(i);
        }
    }

    Ok(invalid_indices)
}

/// Measures the time taken to generate a CID for the given data.
///
/// Returns the duration in microseconds and the generated CID.
///
/// # Examples
///
/// ```rust
/// use ipfrs_core::utils::measure_cid_generation;
/// use ipfrs_core::HashAlgorithm;
///
/// let (duration_us, cid) = measure_cid_generation(b"test data", HashAlgorithm::Sha256).unwrap();
/// assert!(duration_us > 0);
/// ```
pub fn measure_cid_generation(data: &[u8], algorithm: HashAlgorithm) -> Result<(u64, Cid)> {
    let start = std::time::Instant::now();
    let cid = cid_of(data, algorithm)?;
    let duration = start.elapsed();
    Ok((duration.as_micros() as u64, cid))
}

/// Measures the time taken to create a block from the given data.
///
/// Returns the duration in microseconds and the created block.
///
/// # Examples
///
/// ```rust
/// use ipfrs_core::utils::measure_block_creation;
///
/// let (duration_us, block) = measure_block_creation(b"test data").unwrap();
/// assert!(duration_us > 0);
/// ```
pub fn measure_block_creation(data: &[u8]) -> Result<(u64, Block)> {
    let start = std::time::Instant::now();
    let block = quick_block(data)?;
    let duration = start.elapsed();
    Ok((duration.as_micros() as u64, block))
}

/// Calculates the deduplication ratio for a collection of blocks.
///
/// Returns a value between 0.0 and 1.0, where:
/// - 1.0 means all blocks are unique (no deduplication)
/// - 0.5 means 50% of blocks are unique (50% deduplication)
///
/// # Examples
///
/// ```rust
/// use ipfrs_core::utils::{quick_block, deduplication_ratio};
///
/// let blocks = vec![
///     quick_block(b"same").unwrap(),
///     quick_block(b"same").unwrap(),
///     quick_block(b"different").unwrap(),
/// ];
/// let ratio = deduplication_ratio(&blocks);
/// assert!((ratio - 0.666).abs() < 0.01); // 2 unique out of 3 = ~0.666
/// ```
pub fn deduplication_ratio(blocks: &[Block]) -> f64 {
    if blocks.is_empty() {
        return 0.0;
    }

    let unique_cids: std::collections::HashSet<_> = blocks.iter().map(|b| b.cid()).collect();
    unique_cids.len() as f64 / blocks.len() as f64
}

/// Counts the number of unique CIDs in a collection of blocks.
///
/// # Examples
///
/// ```rust
/// use ipfrs_core::utils::{quick_block, count_unique_blocks};
///
/// let blocks = vec![
///     quick_block(b"same").unwrap(),
///     quick_block(b"same").unwrap(),
///     quick_block(b"different").unwrap(),
/// ];
/// assert_eq!(count_unique_blocks(&blocks), 2);
/// ```
pub fn count_unique_blocks(blocks: &[Block]) -> usize {
    let unique_cids: std::collections::HashSet<_> = blocks.iter().map(|b| b.cid()).collect();
    unique_cids.len()
}

/// Calculates the total size of all blocks in bytes.
///
/// # Examples
///
/// ```rust
/// use ipfrs_core::utils::{quick_block, total_blocks_size};
///
/// let blocks = vec![
///     quick_block(b"data1").unwrap(),
///     quick_block(b"data2").unwrap(),
/// ];
/// assert_eq!(total_blocks_size(&blocks), 10); // "data1" + "data2" = 10 bytes
/// ```
pub fn total_blocks_size(blocks: &[Block]) -> u64 {
    blocks.iter().map(|b| b.size()).sum()
}

// ============================================================================
// Compression Utilities
// ============================================================================

/// Compress a block's data with the specified algorithm and level
///
/// Returns a new compressed `Bytes` buffer.
///
/// # Example
///
/// ```rust
/// use ipfrs_core::utils::{quick_block, compress_block_data, decompress_block_data};
/// use ipfrs_core::CompressionAlgorithm;
///
/// let data = "Hello, World! ".repeat(100); // Use compressible data
/// let block = quick_block(data.as_bytes()).unwrap();
/// let compressed = compress_block_data(block.data(), CompressionAlgorithm::Zstd, 3).unwrap();
/// let decompressed = decompress_block_data(&compressed, CompressionAlgorithm::Zstd).unwrap();
/// assert_eq!(block.data(), &decompressed);
/// ```
pub fn compress_block_data(
    data: &bytes::Bytes,
    algorithm: crate::CompressionAlgorithm,
    level: u8,
) -> crate::Result<bytes::Bytes> {
    crate::compress(data, algorithm, level)
}

/// Decompress block data that was previously compressed
///
/// # Example
///
/// ```rust
/// use ipfrs_core::utils::{compress_block_data, decompress_block_data};
/// use ipfrs_core::CompressionAlgorithm;
/// use bytes::Bytes;
///
/// let data = Bytes::from_static(b"Hello, World!");
/// let compressed = compress_block_data(&data, CompressionAlgorithm::Lz4, 3).unwrap();
/// let decompressed = decompress_block_data(&compressed, CompressionAlgorithm::Lz4).unwrap();
/// assert_eq!(data, decompressed);
/// ```
pub fn decompress_block_data(
    compressed: &bytes::Bytes,
    algorithm: crate::CompressionAlgorithm,
) -> crate::Result<bytes::Bytes> {
    crate::decompress(compressed, algorithm)
}

/// Estimate how much space would be saved by compressing the data
///
/// Returns a ratio between 0.0 and 1.0+ where lower is better.
/// A ratio of 0.5 means the compressed data is 50% of the original size.
///
/// # Example
///
/// ```rust
/// use ipfrs_core::utils::estimate_compression_savings;
/// use ipfrs_core::CompressionAlgorithm;
/// use bytes::Bytes;
///
/// let data = Bytes::from("a".repeat(1000)); // Highly compressible
/// let savings = estimate_compression_savings(&data, CompressionAlgorithm::Zstd, 5).unwrap();
/// assert!(savings < 0.2); // Should compress to less than 20% of original
/// ```
pub fn estimate_compression_savings(
    data: &bytes::Bytes,
    algorithm: crate::CompressionAlgorithm,
    level: u8,
) -> crate::Result<f64> {
    crate::compression_ratio(data, algorithm, level)
}

/// Check if data is worth compressing based on size and estimated ratio
///
/// Returns `true` if compression would likely save significant space.
/// Uses a threshold of 20% savings and minimum size of 1KB.
///
/// # Example
///
/// ```rust
/// use ipfrs_core::utils::should_compress;
/// use ipfrs_core::CompressionAlgorithm;
/// use bytes::Bytes;
///
/// let small_data = Bytes::from_static(b"Hello"); // Too small
/// assert!(!should_compress(&small_data, CompressionAlgorithm::Zstd, 3).unwrap());
///
/// let large_repetitive = Bytes::from("a".repeat(10000)); // Worth compressing
/// assert!(should_compress(&large_repetitive, CompressionAlgorithm::Zstd, 3).unwrap());
/// ```
pub fn should_compress(
    data: &bytes::Bytes,
    algorithm: crate::CompressionAlgorithm,
    level: u8,
) -> crate::Result<bool> {
    // Don't compress small data (overhead not worth it)
    if data.len() < 1024 {
        return Ok(false);
    }

    // Don't compress if algorithm is None
    if algorithm == crate::CompressionAlgorithm::None {
        return Ok(false);
    }

    // Check if we'd save at least 20%
    let ratio = crate::compression_ratio(data, algorithm, level)?;
    Ok(ratio < 0.8)
}

/// Get recommended compression algorithm based on use case
///
/// Returns `Zstd` for archival (best ratio) or `Lz4` for real-time (fastest).
///
/// # Example
///
/// ```rust
/// use ipfrs_core::utils::recommended_compression;
/// use ipfrs_core::CompressionAlgorithm;
///
/// let archival = recommended_compression(true);
/// assert_eq!(archival, CompressionAlgorithm::Zstd);
///
/// let realtime = recommended_compression(false);
/// assert_eq!(realtime, CompressionAlgorithm::Lz4);
/// ```
pub fn recommended_compression(prefer_ratio_over_speed: bool) -> crate::CompressionAlgorithm {
    if prefer_ratio_over_speed {
        crate::CompressionAlgorithm::Zstd // Best compression ratio
    } else {
        crate::CompressionAlgorithm::Lz4 // Fastest
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quick_block() {
        let block = quick_block(b"test data").unwrap();
        assert_eq!(block.data().as_ref(), b"test data");
    }

    #[test]
    fn test_block_with_hash() {
        let block1 = block_with_hash(b"data", HashAlgorithm::Sha256).unwrap();
        let block2 = block_with_hash(b"data", HashAlgorithm::Sha3_256).unwrap();
        // Different hash algorithms produce different CIDs
        assert_ne!(block1.cid(), block2.cid());
    }

    #[test]
    fn test_cid_functions() {
        let sha256 = sha256_cid(b"test").unwrap();
        let sha3 = sha3_cid(b"test").unwrap();
        assert_ne!(sha256, sha3);
    }

    #[test]
    fn test_all_hash_algorithm_cid_functions() {
        let data = b"test data for all hash algorithms";

        // Test all 8 hash algorithms
        let sha256 = sha256_cid(data).unwrap();
        let sha512 = sha512_cid(data).unwrap();
        let sha3_256 = sha3_cid(data).unwrap();
        let sha3_512 = sha3_512_cid(data).unwrap();
        let blake2b256 = blake2b256_cid(data).unwrap();
        let blake2b512 = blake2b512_cid(data).unwrap();
        let blake2s256 = blake2s256_cid(data).unwrap();
        let blake3 = blake3_cid(data).unwrap();

        // All should produce different CIDs
        let cids = [
            sha256, sha512, sha3_256, sha3_512, blake2b256, blake2b512, blake2s256, blake3,
        ];

        // Check uniqueness - each hash algorithm should produce different output
        for i in 0..cids.len() {
            for j in (i + 1)..cids.len() {
                assert_ne!(cids[i], cids[j], "CID {} and {} should be different", i, j);
            }
        }
    }

    #[test]
    fn test_hash_algorithm_determinism() {
        let data = b"determinism test";

        // Each algorithm should produce the same CID for the same data
        assert_eq!(sha256_cid(data).unwrap(), sha256_cid(data).unwrap());
        assert_eq!(sha512_cid(data).unwrap(), sha512_cid(data).unwrap());
        assert_eq!(sha3_cid(data).unwrap(), sha3_cid(data).unwrap());
        assert_eq!(sha3_512_cid(data).unwrap(), sha3_512_cid(data).unwrap());
        assert_eq!(blake2b256_cid(data).unwrap(), blake2b256_cid(data).unwrap());
        assert_eq!(blake2b512_cid(data).unwrap(), blake2b512_cid(data).unwrap());
        assert_eq!(blake2s256_cid(data).unwrap(), blake2s256_cid(data).unwrap());
        assert_eq!(blake3_cid(data).unwrap(), blake3_cid(data).unwrap());
    }

    #[test]
    fn test_hash_algorithm_names_and_sizes() {
        use crate::HashAlgorithm;

        // Test name() method
        assert_eq!(HashAlgorithm::Sha256.name(), "SHA2-256");
        assert_eq!(HashAlgorithm::Sha512.name(), "SHA2-512");
        assert_eq!(HashAlgorithm::Sha3_256.name(), "SHA3-256");
        assert_eq!(HashAlgorithm::Sha3_512.name(), "SHA3-512");
        assert_eq!(HashAlgorithm::Blake2b256.name(), "BLAKE2b-256");
        assert_eq!(HashAlgorithm::Blake2b512.name(), "BLAKE2b-512");
        assert_eq!(HashAlgorithm::Blake2s256.name(), "BLAKE2s-256");
        assert_eq!(HashAlgorithm::Blake3.name(), "BLAKE3");

        // Test hash_size() method
        assert_eq!(HashAlgorithm::Sha256.hash_size(), 32);
        assert_eq!(HashAlgorithm::Sha512.hash_size(), 64);
        assert_eq!(HashAlgorithm::Sha3_256.hash_size(), 32);
        assert_eq!(HashAlgorithm::Sha3_512.hash_size(), 64);
        assert_eq!(HashAlgorithm::Blake2b256.hash_size(), 32);
        assert_eq!(HashAlgorithm::Blake2b512.hash_size(), 64);
        assert_eq!(HashAlgorithm::Blake2s256.hash_size(), 32);
        assert_eq!(HashAlgorithm::Blake3.hash_size(), 32);
    }

    #[test]
    fn test_hash_algorithm_all() {
        use crate::HashAlgorithm;

        let all = HashAlgorithm::all();
        assert_eq!(all.len(), 8);

        // Verify all algorithms are present
        assert!(all.contains(&HashAlgorithm::Sha256));
        assert!(all.contains(&HashAlgorithm::Sha512));
        assert!(all.contains(&HashAlgorithm::Sha3_256));
        assert!(all.contains(&HashAlgorithm::Sha3_512));
        assert!(all.contains(&HashAlgorithm::Blake2b256));
        assert!(all.contains(&HashAlgorithm::Blake2b512));
        assert!(all.contains(&HashAlgorithm::Blake2s256));
        assert!(all.contains(&HashAlgorithm::Blake3));
    }

    #[test]
    fn test_blocks_equal() {
        let block1 = quick_block(b"same").unwrap();
        let block2 = quick_block(b"same").unwrap();
        let block3 = quick_block(b"different").unwrap();

        assert!(blocks_equal(&block1, &block2));
        assert!(!blocks_equal(&block1, &block3));
    }

    #[test]
    fn test_verify_block() {
        let block = quick_block(b"verify me").unwrap();
        assert!(verify_block(&block).unwrap());
    }

    #[test]
    fn test_ipld_map() {
        let map = ipld_map(vec![
            ("key1", Ipld::String("value1".to_string())),
            ("key2", Ipld::Integer(42)),
        ]);

        match map {
            Ipld::Map(m) => {
                assert_eq!(m.len(), 2);
                assert!(m.contains_key("key1"));
                assert!(m.contains_key("key2"));
            }
            _ => panic!("Expected map"),
        }
    }

    #[test]
    fn test_ipld_list() {
        let list = ipld_list(vec![Ipld::Integer(1), Ipld::Integer(2), Ipld::Integer(3)]);

        match list {
            Ipld::List(l) => assert_eq!(l.len(), 3),
            _ => panic!("Expected list"),
        }
    }

    #[test]
    fn test_ipld_cbor_roundtrip() {
        let ipld = Ipld::String("test".to_string());
        let cbor = ipld_to_cbor(&ipld).unwrap();
        let decoded = ipld_from_cbor(&cbor).unwrap();
        assert_eq!(ipld, decoded);
    }

    #[test]
    fn test_ipld_json_roundtrip() {
        let ipld = Ipld::String("test".to_string());
        let json = ipld_to_json(&ipld).unwrap();
        let decoded = ipld_from_json(&json).unwrap();
        assert_eq!(ipld, decoded);
    }

    #[test]
    fn test_format_size() {
        assert_eq!(format_size(512), "512 B");
        assert_eq!(format_size(1024), "1.00 KB");
        assert_eq!(format_size(1_048_576), "1.00 MB");
        assert_eq!(format_size(1_073_741_824), "1.00 GB");
        assert_eq!(format_size(2_147_483_648), "2.00 GB");
    }

    #[test]
    fn test_estimate_chunks() {
        assert_eq!(estimate_chunks(100), 1);
        assert_eq!(estimate_chunks(300_000), 2);
        assert_eq!(estimate_chunks(1_000_000), 4);
    }

    #[test]
    fn test_needs_chunking() {
        assert!(!needs_chunking(100));
        assert!(!needs_chunking(1_000_000));
        assert!(!needs_chunking(2_000_000)); // 2MB < MAX_BLOCK_SIZE (2MiB)
        assert!(needs_chunking(3_000_000)); // 3MB > MAX_BLOCK_SIZE
        assert!(needs_chunking(10_000_000));
    }

    // Diagnostic and validation tests

    #[test]
    fn test_inspect_cid() {
        let cid = sha256_cid(b"test").unwrap();
        let info = inspect_cid(&cid);
        assert_eq!(info.version, 1);
        assert!(!info.cid_string.is_empty());
        assert!(info.hash_length > 0);
    }

    #[test]
    fn test_inspect_block() {
        let block = quick_block(b"test data").unwrap();
        let info = inspect_block(&block).unwrap();
        assert!(info.is_valid);
        assert_eq!(info.size, 9_u64);
        assert!(!info.cid.is_empty());
        assert!(!info.size_formatted.is_empty());
    }

    #[test]
    fn test_cid_info_display() {
        let cid = sha256_cid(b"test").unwrap();
        let info = inspect_cid(&cid);
        let display = format!("{}", info);
        assert!(display.contains("CID:"));
        assert!(display.contains("Version:"));
        assert!(display.contains("Codec:"));
    }

    #[test]
    fn test_block_info_display() {
        let block = quick_block(b"test").unwrap();
        let info = inspect_block(&block).unwrap();
        let display = format!("{}", info);
        assert!(display.contains("Block:"));
        assert!(display.contains("CID:"));
        assert!(display.contains("Valid:"));
    }

    #[test]
    fn test_validate_blocks() {
        let blocks = vec![
            quick_block(b"data1").unwrap(),
            quick_block(b"data2").unwrap(),
            quick_block(b"data3").unwrap(),
        ];

        let (valid, invalid) = validate_blocks(&blocks).unwrap();
        assert_eq!(valid, 3);
        assert_eq!(invalid, 0);
    }

    #[test]
    fn test_validate_blocks_empty() {
        let blocks: Vec<Block> = vec![];
        let (valid, invalid) = validate_blocks(&blocks).unwrap();
        assert_eq!(valid, 0);
        assert_eq!(invalid, 0);
    }

    #[test]
    fn test_find_invalid_blocks() {
        let blocks = vec![
            quick_block(b"data1").unwrap(),
            quick_block(b"data2").unwrap(),
        ];

        let invalid = find_invalid_blocks(&blocks).unwrap();
        assert_eq!(invalid.len(), 0);
    }

    #[test]
    fn test_measure_cid_generation() {
        let (duration, cid) = measure_cid_generation(b"test data", HashAlgorithm::Sha256).unwrap();
        assert!(duration > 0);
        assert!(!cid.to_string().is_empty());
    }

    #[test]
    fn test_measure_block_creation() {
        let (duration, block) = measure_block_creation(b"test data").unwrap();
        assert!(duration > 0);
        assert_eq!(block.size(), 9_u64);
    }

    #[test]
    fn test_deduplication_ratio() {
        let blocks = vec![
            quick_block(b"same").unwrap(),
            quick_block(b"same").unwrap(),
            quick_block(b"different").unwrap(),
        ];

        let ratio = deduplication_ratio(&blocks);
        // 2 unique out of 3 total
        assert!((ratio - 0.666).abs() < 0.01);
    }

    #[test]
    fn test_deduplication_ratio_all_unique() {
        let blocks = vec![
            quick_block(b"data1").unwrap(),
            quick_block(b"data2").unwrap(),
            quick_block(b"data3").unwrap(),
        ];

        let ratio = deduplication_ratio(&blocks);
        assert_eq!(ratio, 1.0);
    }

    #[test]
    fn test_deduplication_ratio_all_same() {
        let blocks = vec![
            quick_block(b"same").unwrap(),
            quick_block(b"same").unwrap(),
            quick_block(b"same").unwrap(),
        ];

        let ratio = deduplication_ratio(&blocks);
        // 1 unique out of 3 total
        assert!((ratio - 0.333).abs() < 0.01);
    }

    #[test]
    fn test_deduplication_ratio_empty() {
        let blocks: Vec<Block> = vec![];
        let ratio = deduplication_ratio(&blocks);
        assert_eq!(ratio, 0.0);
    }

    #[test]
    fn test_count_unique_blocks() {
        let blocks = vec![
            quick_block(b"same").unwrap(),
            quick_block(b"same").unwrap(),
            quick_block(b"different").unwrap(),
        ];

        assert_eq!(count_unique_blocks(&blocks), 2);
    }

    #[test]
    fn test_count_unique_blocks_all_unique() {
        let blocks = vec![
            quick_block(b"a").unwrap(),
            quick_block(b"b").unwrap(),
            quick_block(b"c").unwrap(),
        ];

        assert_eq!(count_unique_blocks(&blocks), 3);
    }

    #[test]
    fn test_count_unique_blocks_empty() {
        let blocks: Vec<Block> = vec![];
        assert_eq!(count_unique_blocks(&blocks), 0);
    }

    #[test]
    fn test_total_blocks_size() {
        let blocks = vec![
            quick_block(b"data1").unwrap(), // 5 bytes
            quick_block(b"data2").unwrap(), // 5 bytes
        ];

        assert_eq!(total_blocks_size(&blocks), 10);
    }

    #[test]
    fn test_total_blocks_size_empty() {
        let blocks: Vec<Block> = vec![];
        assert_eq!(total_blocks_size(&blocks), 0);
    }

    #[test]
    fn test_compress_block_data() {
        use crate::CompressionAlgorithm;

        let data = bytes::Bytes::from("Hello, World! ".repeat(100));
        let compressed = compress_block_data(&data, CompressionAlgorithm::Zstd, 5).unwrap();

        // Should compress well with repetitive data
        assert!(compressed.len() < data.len());

        // Decompress and verify
        let decompressed = decompress_block_data(&compressed, CompressionAlgorithm::Zstd).unwrap();
        assert_eq!(data, decompressed);
    }

    #[test]
    fn test_estimate_compression_savings() {
        use crate::CompressionAlgorithm;

        let data = bytes::Bytes::from("a".repeat(1000));
        let ratio = estimate_compression_savings(&data, CompressionAlgorithm::Zstd, 5).unwrap();

        // Highly repetitive data should compress very well
        assert!(ratio < 0.1);
    }

    #[test]
    fn test_should_compress() {
        use crate::CompressionAlgorithm;

        // Small data should not be compressed
        let small = bytes::Bytes::from_static(b"Hello");
        assert!(!should_compress(&small, CompressionAlgorithm::Zstd, 3).unwrap());

        // Large repetitive data should be compressed
        let large = bytes::Bytes::from("a".repeat(10000));
        assert!(should_compress(&large, CompressionAlgorithm::Zstd, 3).unwrap());

        // None algorithm should never compress
        assert!(!should_compress(&large, CompressionAlgorithm::None, 3).unwrap());
    }

    #[test]
    fn test_recommended_compression() {
        use crate::CompressionAlgorithm;

        assert_eq!(recommended_compression(true), CompressionAlgorithm::Zstd);
        assert_eq!(recommended_compression(false), CompressionAlgorithm::Lz4);
    }
}
