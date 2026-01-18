//! Chunking and DAG (Directed Acyclic Graph) support for large file handling
//!
//! This module provides functionality for:
//! - Splitting large files into content-addressed blocks
//! - Creating Merkle DAG structures
//! - Reassembling files from chunks

use crate::block::{Block, MAX_BLOCK_SIZE};
use crate::cid::{Cid, CidBuilder, SerializableCid};
use crate::error::{Error, Result};
use crate::ipld::Ipld;
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Default chunk size (256 KiB) - a good balance between overhead and deduplication
pub const DEFAULT_CHUNK_SIZE: usize = 256 * 1024;

/// Minimum chunk size (1 KiB)
pub const MIN_CHUNK_SIZE: usize = 1024;

/// Maximum chunk size (1 MiB, leaving room for metadata in a 2 MiB block)
pub const MAX_CHUNK_SIZE: usize = 1024 * 1024;

/// Maximum number of links in a single DAG node
/// This prevents any single node from being too large
pub const MAX_LINKS_PER_NODE: usize = 174;

/// Chunking strategy for splitting data
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ChunkingStrategy {
    /// Fixed-size chunking (simple, deterministic)
    #[default]
    FixedSize,
    /// Content-defined chunking (CDC) with Rabin fingerprinting
    /// Enables better deduplication by finding chunk boundaries based on content
    ContentDefined,
}

/// Configuration for chunking operations
#[derive(Debug, Clone)]
pub struct ChunkingConfig {
    /// Size of each chunk in bytes
    pub chunk_size: usize,
    /// Chunking strategy to use
    pub strategy: ChunkingStrategy,
    /// Maximum links per DAG node (for balanced trees)
    pub max_links_per_node: usize,
}

impl Default for ChunkingConfig {
    fn default() -> Self {
        Self {
            chunk_size: DEFAULT_CHUNK_SIZE,
            strategy: ChunkingStrategy::FixedSize,
            max_links_per_node: MAX_LINKS_PER_NODE,
        }
    }
}

impl ChunkingConfig {
    /// Create a new chunking configuration builder
    pub fn builder() -> ChunkingConfigBuilder {
        ChunkingConfigBuilder::new()
    }

    /// Create a new chunking configuration with the specified chunk size
    pub fn with_chunk_size(chunk_size: usize) -> Result<Self> {
        if chunk_size < MIN_CHUNK_SIZE {
            return Err(Error::InvalidInput(format!(
                "Chunk size {} is below minimum {}",
                chunk_size, MIN_CHUNK_SIZE
            )));
        }
        if chunk_size > MAX_CHUNK_SIZE {
            return Err(Error::InvalidInput(format!(
                "Chunk size {} exceeds maximum {}",
                chunk_size, MAX_CHUNK_SIZE
            )));
        }
        Ok(Self {
            chunk_size,
            ..Default::default()
        })
    }

    /// Create a configuration for content-defined chunking
    pub fn content_defined() -> Self {
        Self {
            chunk_size: DEFAULT_CHUNK_SIZE,
            strategy: ChunkingStrategy::ContentDefined,
            max_links_per_node: MAX_LINKS_PER_NODE,
        }
    }

    /// Create a configuration for content-defined chunking with custom target size
    pub fn content_defined_with_size(target_size: usize) -> Result<Self> {
        if target_size < MIN_CHUNK_SIZE {
            return Err(Error::InvalidInput(format!(
                "Target chunk size {} is below minimum {}",
                target_size, MIN_CHUNK_SIZE
            )));
        }
        if target_size > MAX_CHUNK_SIZE {
            return Err(Error::InvalidInput(format!(
                "Target chunk size {} exceeds maximum {}",
                target_size, MAX_CHUNK_SIZE
            )));
        }
        Ok(Self {
            chunk_size: target_size,
            strategy: ChunkingStrategy::ContentDefined,
            max_links_per_node: MAX_LINKS_PER_NODE,
        })
    }
}

/// Builder for creating custom chunking configurations
///
/// # Example
///
/// ```
/// use ipfrs_core::{ChunkingConfig, ChunkingStrategy};
///
/// let config = ChunkingConfig::builder()
///     .chunk_size(1024 * 1024)
///     .strategy(ChunkingStrategy::ContentDefined)
///     .max_links_per_node(256)
///     .build()
///     .unwrap();
/// ```
#[derive(Debug, Clone, Default)]
pub struct ChunkingConfigBuilder {
    chunk_size: Option<usize>,
    strategy: Option<ChunkingStrategy>,
    max_links_per_node: Option<usize>,
}

impl ChunkingConfigBuilder {
    /// Create a new builder with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the chunk size
    pub fn chunk_size(mut self, size: usize) -> Self {
        self.chunk_size = Some(size);
        self
    }

    /// Set the chunking strategy
    pub fn strategy(mut self, strategy: ChunkingStrategy) -> Self {
        self.strategy = Some(strategy);
        self
    }

    /// Set the maximum links per DAG node
    pub fn max_links_per_node(mut self, max_links: usize) -> Self {
        self.max_links_per_node = Some(max_links);
        self
    }

    /// Build the chunking configuration
    ///
    /// # Errors
    ///
    /// Returns an error if the chunk size is outside valid bounds.
    pub fn build(self) -> Result<ChunkingConfig> {
        let chunk_size = self.chunk_size.unwrap_or(DEFAULT_CHUNK_SIZE);
        let strategy = self.strategy.unwrap_or(ChunkingStrategy::FixedSize);
        let max_links_per_node = self.max_links_per_node.unwrap_or(MAX_LINKS_PER_NODE);

        if chunk_size < MIN_CHUNK_SIZE {
            return Err(Error::InvalidInput(format!(
                "Chunk size {} is below minimum {}",
                chunk_size, MIN_CHUNK_SIZE
            )));
        }
        if chunk_size > MAX_CHUNK_SIZE {
            return Err(Error::InvalidInput(format!(
                "Chunk size {} exceeds maximum {}",
                chunk_size, MAX_CHUNK_SIZE
            )));
        }

        Ok(ChunkingConfig {
            chunk_size,
            strategy,
            max_links_per_node,
        })
    }
}

/// Rabin fingerprinting for content-defined chunking
///
/// This implementation uses a rolling hash to find chunk boundaries
/// based on content, enabling better deduplication for similar files.
struct RabinChunker {
    /// Target average chunk size
    #[allow(dead_code)]
    target_size: usize,
    /// Minimum chunk size (to avoid too-small chunks)
    min_size: usize,
    /// Maximum chunk size (to bound memory usage)
    max_size: usize,
    /// Polynomial for Rabin fingerprinting
    polynomial: u64,
    /// Window size for rolling hash
    window_size: usize,
    /// Mask for determining chunk boundaries
    /// A chunk boundary occurs when (hash & mask) == 0
    mask: u64,
}

impl RabinChunker {
    /// Create a new Rabin chunker with the given target chunk size
    fn new(target_size: usize) -> Self {
        // Use a well-known irreducible polynomial for Rabin fingerprinting
        const POLYNOMIAL: u64 = 0x3DA3358B4DC173;

        // Window size (typically 48-64 bytes)
        const WINDOW_SIZE: usize = 48;

        // Calculate mask based on target size
        // For target size N, we want average chunk size ≈ N
        // So we set mask bits such that probability of match is 1/N
        let mask_bits = (target_size as f64).log2().floor() as u32;
        let mask = (1u64 << mask_bits) - 1;

        Self {
            target_size,
            min_size: target_size / 4,
            max_size: target_size * 4,
            polynomial: POLYNOMIAL,
            window_size: WINDOW_SIZE,
            mask,
        }
    }

    /// Find chunk boundaries in the given data
    /// Returns a vector of chunk end positions (exclusive)
    fn find_boundaries(&self, data: &[u8]) -> Vec<usize> {
        if data.len() <= self.min_size {
            return vec![data.len()];
        }

        let mut boundaries = Vec::new();
        let mut hash: u64 = 0;
        let mut window = vec![0u8; self.window_size];
        let mut window_pos = 0;
        let mut last_boundary = 0;

        for (i, &byte) in data.iter().enumerate() {
            // Update rolling hash: remove old byte, add new byte
            let out_byte = window[window_pos];
            window[window_pos] = byte;
            window_pos = (window_pos + 1) % self.window_size;

            // Rabin fingerprint update
            hash = hash.rotate_left(1);
            hash ^= self.out_table(out_byte);
            hash ^= self.in_table(byte);

            let chunk_size = i - last_boundary;

            // Check if we found a boundary
            if chunk_size >= self.min_size {
                // Force a boundary at max_size
                if chunk_size >= self.max_size {
                    boundaries.push(i);
                    last_boundary = i;
                    hash = 0;
                    window.fill(0);
                    window_pos = 0;
                }
                // Check for content-defined boundary
                else if (hash & self.mask) == 0 {
                    boundaries.push(i);
                    last_boundary = i;
                }
            }
        }

        // Add final boundary if needed
        if last_boundary < data.len() {
            boundaries.push(data.len());
        }

        boundaries
    }

    /// Lookup table for removing bytes from the hash
    #[inline]
    fn out_table(&self, byte: u8) -> u64 {
        // Precomputed values based on window_size and polynomial
        // This is a simplified version; in production, use precomputed tables
        let mut val = byte as u64;
        for _ in 0..self.window_size {
            val = val.rotate_left(1) ^ self.polynomial;
        }
        val
    }

    /// Lookup table for adding bytes to the hash
    #[inline]
    fn in_table(&self, byte: u8) -> u64 {
        byte as u64
    }
}

/// Statistics for chunk deduplication
#[derive(Debug, Clone, Default)]
pub struct DeduplicationStats {
    /// Total number of chunks created
    pub total_chunks: usize,
    /// Number of unique chunks (after deduplication)
    pub unique_chunks: usize,
    /// Number of chunks reused (total_chunks - unique_chunks)
    pub reused_chunks: usize,
    /// Space savings percentage (0-100)
    pub space_savings_percent: f64,
    /// Total original data size
    pub total_data_size: u64,
    /// Actual storage size after deduplication
    pub deduplicated_size: u64,
}

/// A link to another block in the DAG
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DagLink {
    /// CID of the linked block
    pub cid: SerializableCid,
    /// Size of the linked block's data (including recursive)
    pub size: u64,
    /// Optional name for the link
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl DagLink {
    /// Create a new DAG link
    pub fn new(cid: Cid, size: u64) -> Self {
        Self {
            cid: SerializableCid(cid),
            size,
            name: None,
        }
    }

    /// Create a named DAG link
    pub fn with_name(cid: Cid, size: u64, name: impl Into<String>) -> Self {
        Self {
            cid: SerializableCid(cid),
            size,
            name: Some(name.into()),
        }
    }
}

/// A node in the Merkle DAG structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DagNode {
    /// Links to child nodes
    pub links: Vec<DagLink>,
    /// Total size of all data under this node
    pub total_size: u64,
    /// Raw data in this node (for leaf nodes)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Vec<u8>>,
}

impl DagNode {
    /// Create a leaf node with data
    pub fn leaf(data: Vec<u8>) -> Self {
        let size = data.len() as u64;
        Self {
            links: Vec::new(),
            total_size: size,
            data: Some(data),
        }
    }

    /// Create an intermediate node with links
    pub fn intermediate(links: Vec<DagLink>) -> Self {
        let total_size = links.iter().map(|l| l.size).sum();
        Self {
            links,
            total_size,
            data: None,
        }
    }

    /// Check if this is a leaf node
    #[must_use]
    pub fn is_leaf(&self) -> bool {
        self.links.is_empty()
    }

    /// Get the number of links
    pub fn link_count(&self) -> usize {
        self.links.len()
    }

    /// Convert to IPLD representation
    pub fn to_ipld(&self) -> Ipld {
        let mut map = BTreeMap::new();

        // Add links
        let links: Vec<Ipld> = self
            .links
            .iter()
            .map(|link| {
                let mut link_map = BTreeMap::new();
                link_map.insert("cid".to_string(), Ipld::Link(link.cid));
                link_map.insert("size".to_string(), Ipld::Integer(link.size as i128));
                if let Some(name) = &link.name {
                    link_map.insert("name".to_string(), Ipld::String(name.clone()));
                }
                Ipld::Map(link_map)
            })
            .collect();
        map.insert("links".to_string(), Ipld::List(links));

        // Add total size
        map.insert(
            "totalSize".to_string(),
            Ipld::Integer(self.total_size as i128),
        );

        // Add data if present
        if let Some(data) = &self.data {
            map.insert("data".to_string(), Ipld::Bytes(data.clone()));
        }

        Ipld::Map(map)
    }

    /// Encode to DAG-CBOR bytes
    pub fn to_dag_cbor(&self) -> Result<Vec<u8>> {
        self.to_ipld().to_dag_cbor()
    }
}

/// Result of chunking a file
#[derive(Debug, Clone)]
pub struct ChunkedFile {
    /// Root CID of the DAG
    pub root_cid: Cid,
    /// All blocks generated (including root)
    pub blocks: Vec<Block>,
    /// Total original file size
    pub total_size: u64,
    /// Number of leaf chunks
    pub chunk_count: usize,
    /// Deduplication statistics (if available)
    pub dedup_stats: Option<DeduplicationStats>,
}

/// Chunker for splitting large data into blocks
#[derive(Default)]
pub struct Chunker {
    config: ChunkingConfig,
}

impl Chunker {
    /// Create a new chunker with default configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a chunker with custom configuration
    pub fn with_config(config: ChunkingConfig) -> Self {
        Self { config }
    }

    /// Chunk data into a Merkle DAG structure
    ///
    /// Returns the root CID and all generated blocks
    pub fn chunk(&self, data: &[u8]) -> Result<ChunkedFile> {
        if data.is_empty() {
            return Err(Error::InvalidInput("Cannot chunk empty data".to_string()));
        }

        // For small files, create a single block
        if data.len() <= self.config.chunk_size {
            let block = Block::new(Bytes::copy_from_slice(data))?;
            let dedup_stats = DeduplicationStats {
                total_chunks: 1,
                unique_chunks: 1,
                reused_chunks: 0,
                space_savings_percent: 0.0,
                total_data_size: data.len() as u64,
                deduplicated_size: data.len() as u64,
            };
            return Ok(ChunkedFile {
                root_cid: *block.cid(),
                blocks: vec![block],
                total_size: data.len() as u64,
                chunk_count: 1,
                dedup_stats: Some(dedup_stats),
            });
        }

        // Split into chunks based on strategy
        let chunk_slices: Vec<&[u8]> = match self.config.strategy {
            ChunkingStrategy::FixedSize => {
                // Fixed-size chunking
                data.chunks(self.config.chunk_size).collect()
            }
            ChunkingStrategy::ContentDefined => {
                // Content-defined chunking with Rabin fingerprinting
                let rabin = RabinChunker::new(self.config.chunk_size);
                let boundaries = rabin.find_boundaries(data);
                let mut chunks = Vec::with_capacity(boundaries.len());
                let mut start = 0;
                for &end in &boundaries {
                    chunks.push(&data[start..end]);
                    start = end;
                }
                chunks
            }
        };

        let chunk_count = chunk_slices.len();

        // Create leaf blocks and track deduplication
        let mut leaf_blocks: Vec<Block> = Vec::with_capacity(chunk_slices.len());
        let mut leaf_links: Vec<DagLink> = Vec::with_capacity(chunk_slices.len());
        let mut seen_cids = std::collections::HashMap::new();

        for chunk in chunk_slices {
            let block = Block::new(Bytes::copy_from_slice(chunk))?;
            let cid = *block.cid();

            // Track deduplication
            seen_cids.entry(cid).or_insert(chunk.len());

            leaf_links.push(DagLink::new(cid, chunk.len() as u64));
            leaf_blocks.push(block);
        }

        // Calculate deduplication statistics
        let total_data_size = data.len() as u64;
        let deduplicated_size: u64 = seen_cids.values().map(|&size| size as u64).sum();
        let reused_chunks = chunk_count.saturating_sub(seen_cids.len());
        let space_savings_percent = if total_data_size > 0 {
            ((total_data_size - deduplicated_size) as f64 / total_data_size as f64) * 100.0
        } else {
            0.0
        };

        let dedup_stats = DeduplicationStats {
            total_chunks: chunk_count,
            unique_chunks: seen_cids.len(),
            reused_chunks,
            space_savings_percent,
            total_data_size,
            deduplicated_size,
        };

        // Build the DAG tree (bottom-up)
        let mut all_blocks = leaf_blocks;
        let mut current_links = leaf_links;

        while current_links.len() > 1 {
            let mut next_level_links = Vec::new();
            let mut next_level_blocks = Vec::new();

            for link_chunk in current_links.chunks(self.config.max_links_per_node) {
                let node = DagNode::intermediate(link_chunk.to_vec());
                let node_bytes = node.to_dag_cbor()?;

                // Ensure the node fits in a block
                if node_bytes.len() > MAX_BLOCK_SIZE {
                    return Err(Error::Internal(
                        "DAG node too large, reduce max_links_per_node".to_string(),
                    ));
                }

                let block = Block::builder()
                    .codec(crate::cid::codec::DAG_CBOR)
                    .build(Bytes::from(node_bytes))?;

                next_level_links.push(DagLink::new(*block.cid(), node.total_size));
                next_level_blocks.push(block);
            }

            all_blocks.extend(next_level_blocks);
            current_links = next_level_links;
        }

        // The last remaining link points to our root
        let root_cid = current_links[0].cid.0;

        Ok(ChunkedFile {
            root_cid,
            blocks: all_blocks,
            total_size: data.len() as u64,
            chunk_count,
            dedup_stats: Some(dedup_stats),
        })
    }

    /// Check if data would need to be chunked
    #[must_use]
    pub fn needs_chunking(&self, data_len: usize) -> bool {
        data_len > self.config.chunk_size
    }

    /// Estimate the number of chunks for a given data size
    pub fn estimate_chunk_count(&self, data_len: usize) -> usize {
        if data_len == 0 {
            return 0;
        }
        data_len.div_ceil(self.config.chunk_size)
    }
}

/// Builder for creating Merkle DAG structures incrementally
pub struct DagBuilder {
    config: ChunkingConfig,
    #[allow(dead_code)]
    cid_builder: CidBuilder,
}

impl Default for DagBuilder {
    fn default() -> Self {
        Self {
            config: ChunkingConfig::default(),
            cid_builder: CidBuilder::new(),
        }
    }
}

impl DagBuilder {
    /// Create a new DAG builder
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the chunking configuration
    pub fn with_config(mut self, config: ChunkingConfig) -> Self {
        self.config = config;
        self
    }

    /// Create a directory-like DAG node from named entries
    pub fn create_directory(&self, entries: Vec<(String, Cid, u64)>) -> Result<(DagNode, Block)> {
        let links: Vec<DagLink> = entries
            .into_iter()
            .map(|(name, cid, size)| DagLink::with_name(cid, size, name))
            .collect();

        let node = DagNode::intermediate(links);
        let node_bytes = node.to_dag_cbor()?;
        let block = Block::builder()
            .codec(crate::cid::codec::DAG_CBOR)
            .build(Bytes::from(node_bytes))?;

        Ok((node, block))
    }

    /// Create a simple file DAG from data
    pub fn create_file(&self, data: &[u8]) -> Result<ChunkedFile> {
        Chunker::with_config(self.config.clone()).chunk(data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_small_file_single_block() {
        let chunker = Chunker::new();
        let data = b"Hello, IPFS!";

        let result = chunker.chunk(data).unwrap();

        assert_eq!(result.chunk_count, 1);
        assert_eq!(result.blocks.len(), 1);
        assert_eq!(result.total_size, data.len() as u64);
    }

    #[test]
    fn test_large_file_chunking() {
        let config = ChunkingConfig::with_chunk_size(1024).unwrap();
        let chunker = Chunker::with_config(config);

        // Create 5KB of data
        let data: Vec<u8> = (0..5000).map(|i| (i % 256) as u8).collect();
        let result = chunker.chunk(&data).unwrap();

        assert_eq!(result.chunk_count, 5); // 5KB / 1KB = 5 chunks
        assert!(result.blocks.len() >= 5); // At least 5 leaf blocks + intermediate nodes
        assert_eq!(result.total_size, 5000);
    }

    #[test]
    fn test_estimate_chunk_count() {
        let config = ChunkingConfig::with_chunk_size(1024).unwrap();
        let chunker = Chunker::with_config(config);

        assert_eq!(chunker.estimate_chunk_count(0), 0);
        assert_eq!(chunker.estimate_chunk_count(512), 1);
        assert_eq!(chunker.estimate_chunk_count(1024), 1);
        assert_eq!(chunker.estimate_chunk_count(1025), 2);
        assert_eq!(chunker.estimate_chunk_count(3000), 3);
    }

    #[test]
    fn test_needs_chunking() {
        let config = ChunkingConfig::with_chunk_size(1024).unwrap();
        let chunker = Chunker::with_config(config);

        assert!(!chunker.needs_chunking(512));
        assert!(!chunker.needs_chunking(1024));
        assert!(chunker.needs_chunking(1025));
    }

    #[test]
    fn test_dag_node_to_ipld() {
        let cid = CidBuilder::new().build(b"test").unwrap();
        let link = DagLink::with_name(cid, 100, "test.txt");
        let node = DagNode::intermediate(vec![link]);

        let ipld = node.to_ipld();
        assert!(matches!(ipld, Ipld::Map(_)));

        // Should be encodable to DAG-CBOR
        let cbor = node.to_dag_cbor().unwrap();
        assert!(!cbor.is_empty());
    }

    #[test]
    fn test_directory_creation() {
        let builder = DagBuilder::new();
        let cid1 = CidBuilder::new().build(b"file1").unwrap();
        let cid2 = CidBuilder::new().build(b"file2").unwrap();

        let entries = vec![
            ("file1.txt".to_string(), cid1, 100),
            ("file2.txt".to_string(), cid2, 200),
        ];

        let (node, block) = builder.create_directory(entries).unwrap();

        assert_eq!(node.link_count(), 2);
        assert_eq!(node.total_size, 300);
        assert!(block.size() > 0);
    }

    #[test]
    fn test_content_defined_chunking() {
        let config = ChunkingConfig::content_defined();
        let chunker = Chunker::with_config(config);

        // Create 1MB of data with some patterns
        let mut data = Vec::with_capacity(1_000_000);
        for i in 0..1_000_000 {
            data.push((i % 256) as u8);
        }

        let result = chunker.chunk(&data).unwrap();

        // CDC should create chunks
        assert!(result.chunk_count > 0);
        assert!(!result.blocks.is_empty());
        assert_eq!(result.total_size, 1_000_000);

        // Dedup stats should be present
        assert!(result.dedup_stats.is_some());
        let stats = result.dedup_stats.unwrap();
        assert_eq!(stats.total_chunks, result.chunk_count);
    }

    #[test]
    fn test_cdc_deduplication() {
        let config = ChunkingConfig::content_defined_with_size(4096).unwrap();
        let chunker = Chunker::with_config(config);

        // Create data with repeated sections (should deduplicate well)
        let pattern = b"Hello, IPFS! This is a test pattern. ".repeat(100);
        let mut data = pattern.clone();
        data.extend_from_slice(&pattern); // Duplicate the pattern

        let result = chunker.chunk(&data).unwrap();

        // Check dedup stats
        let stats = result.dedup_stats.unwrap();
        assert!(stats.total_chunks > 0);

        // With repeated content, we might see some deduplication
        // (though it depends on where boundaries fall)
        assert_eq!(
            stats.total_chunks,
            stats.unique_chunks + stats.reused_chunks
        );
    }

    #[test]
    fn test_cdc_deterministic() {
        let config = ChunkingConfig::content_defined();
        let chunker = Chunker::with_config(config);

        let data: Vec<u8> = (0..100_000).map(|i| (i % 256) as u8).collect();

        // Chunk the same data twice
        let result1 = chunker.chunk(&data).unwrap();
        let result2 = chunker.chunk(&data).unwrap();

        // Should produce identical results
        assert_eq!(result1.root_cid, result2.root_cid);
        assert_eq!(result1.chunk_count, result2.chunk_count);
        assert_eq!(result1.total_size, result2.total_size);
    }

    #[test]
    fn test_cdc_vs_fixed_size() {
        let data: Vec<u8> = (0..100_000).map(|i| (i % 256) as u8).collect();

        // Fixed-size chunking
        let fixed_config = ChunkingConfig::with_chunk_size(8192).unwrap();
        let fixed_chunker = Chunker::with_config(fixed_config);
        let fixed_result = fixed_chunker.chunk(&data).unwrap();

        // Content-defined chunking
        let cdc_config = ChunkingConfig::content_defined_with_size(8192).unwrap();
        let cdc_chunker = Chunker::with_config(cdc_config);
        let cdc_result = cdc_chunker.chunk(&data).unwrap();

        // Both should chunk the data
        assert!(fixed_result.chunk_count > 0);
        assert!(cdc_result.chunk_count > 0);

        // Fixed-size chunking is predictable
        assert_eq!(fixed_result.chunk_count, 100_000 / 8192 + 1);

        // CDC chunking varies based on content
        // For uniform data, CDC should still create multiple chunks
        // but the exact count depends on the rolling hash
        assert!(cdc_result.chunk_count >= 1);
        assert!(cdc_result.chunk_count < 200); // Upper bound check

        // Both strategies should have similar total size
        assert_eq!(fixed_result.total_size, cdc_result.total_size);
    }

    #[test]
    fn test_rabin_chunker_boundaries() {
        let rabin = RabinChunker::new(8192);
        let data: Vec<u8> = (0..50_000).map(|i| (i % 256) as u8).collect();

        let boundaries = rabin.find_boundaries(&data);

        // Should have at least one boundary (the end)
        assert!(!boundaries.is_empty());

        // Last boundary should be at data length
        assert_eq!(*boundaries.last().unwrap(), data.len());

        // All boundaries should be valid positions
        for &boundary in &boundaries {
            assert!(boundary <= data.len());
        }

        // Boundaries should be monotonically increasing
        for i in 1..boundaries.len() {
            assert!(boundaries[i] > boundaries[i - 1]);
        }
    }

    #[test]
    fn test_rabin_min_max_chunk_size() {
        let rabin = RabinChunker::new(8192);
        let data: Vec<u8> = (0..500_000).map(|i| (i % 256) as u8).collect();

        let boundaries = rabin.find_boundaries(&data);

        // Check that chunks respect min/max size
        let mut start = 0;
        for &end in &boundaries {
            let chunk_size = end - start;

            // Each chunk should be within bounds (except possibly the last one)
            if end < data.len() {
                assert!(
                    chunk_size >= rabin.min_size,
                    "Chunk size {} is below min {}",
                    chunk_size,
                    rabin.min_size
                );
                assert!(
                    chunk_size <= rabin.max_size,
                    "Chunk size {} exceeds max {}",
                    chunk_size,
                    rabin.max_size
                );
            }

            start = end;
        }
    }

    #[test]
    fn test_deduplication_stats_calculation() {
        let config = ChunkingConfig::content_defined();
        let chunker = Chunker::with_config(config);

        let data: Vec<u8> = (0..50_000).map(|i| (i % 256) as u8).collect();
        let result = chunker.chunk(&data).unwrap();

        let stats = result.dedup_stats.unwrap();

        // Verify stats consistency
        assert_eq!(
            stats.total_chunks,
            stats.unique_chunks + stats.reused_chunks
        );
        assert_eq!(stats.total_data_size, 50_000);
        assert!(stats.deduplicated_size <= stats.total_data_size);
        assert!(stats.space_savings_percent >= 0.0);
        assert!(stats.space_savings_percent <= 100.0);
    }
}
