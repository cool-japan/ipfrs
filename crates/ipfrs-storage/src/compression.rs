//! Block-level compression for IPFRS storage
//!
//! Provides two complementary APIs:
//!
//! 1. **Low-level per-block functions** (`compress_block` / `decompress_block`):
//!    Compress individual byte slices with a 1-byte magic prefix. Designed for
//!    wiring directly into block-store `put`/`get` paths without wrapping the
//!    whole store. Targets 30–50 % size reduction for text/JSON/IPLD blocks.
//!
//! 2. **`CompressionBlockStore` wrapper**: Transparent decorator that wraps any
//!    `BlockStore` and compresses data on `put`, decompresses on `get`. Uses
//!    the same magic-byte framing as the low-level API so both layers are
//!    interoperable.
//!
//! ## Magic byte layout
//!
//! ```text
//! byte 0   → encoding flag
//!   0x00   → raw (uncompressed)
//!   0x01   → zstd
//!   0x02   → lz4  (bytes 1..5 hold original_size as u32 LE, then lz4 block data)
//!   0x03   → snappy
//! bytes 1.. → payload
//! ```
//!
//! Blocks shorter than `MIN_COMPRESS_SIZE` (256 bytes) are always stored raw.
//! If compression does not improve the ratio, the raw form is stored instead.
//!
//! ## Example
//!
//! ```rust,ignore
//! use ipfrs_storage::{SledBlockStore, CompressionBlockStore, CompressionConfig, CompressionAlgorithm};
//!
//! let store = SledBlockStore::open(std::env::temp_dir().join("blocks"))?;
//! let config = CompressionConfig::new(CompressionAlgorithm::Zstd)
//!     .with_level(3)
//!     .with_threshold(256);
//! let compressed = CompressionBlockStore::new(store, config);
//! ```

use crate::traits::BlockStore;
use async_trait::async_trait;
use bytes::Bytes;
use ipfrs_core::{Block, Cid, Error, Result};
use parking_lot::RwLock;
use std::sync::Arc;

// ─────────────────────────────────────────────────────────────────────────────
// Magic byte constants (public so users can detect stored format)
// ─────────────────────────────────────────────────────────────────────────────

/// Magic byte: block is stored without compression.
pub const MAGIC_RAW: u8 = 0x00;
/// Magic byte: block is stored with zstd compression.
pub const MAGIC_ZSTD: u8 = 0x01;
/// Magic byte: block is stored with LZ4 compression.
/// For LZ4, bytes 1–4 contain `original_size` as little-endian u32.
pub const MAGIC_LZ4: u8 = 0x02;
/// Magic byte: block is stored with Snappy compression.
pub const MAGIC_SNAPPY: u8 = 0x03;

/// Minimum block size (bytes) that is eligible for compression.
/// Blocks smaller than this are always stored raw.
pub const MIN_COMPRESS_SIZE: usize = 256;

// ─────────────────────────────────────────────────────────────────────────────
// Per-block compression/decompression (low-level API)
// ─────────────────────────────────────────────────────────────────────────────

/// Statistics returned by [`compress_block_with_stats`].
#[derive(Debug, Clone)]
pub struct BlockCompressStats {
    /// Original (uncompressed) size in bytes.
    pub original_size: usize,
    /// Size written to storage (includes 1-byte magic header).
    pub stored_size: usize,
    /// Whether the data was actually compressed (false → stored raw).
    pub compressed: bool,
    /// `stored_size / original_size` (lower is better). Always ≤ 1.0.
    pub ratio: f32,
}

/// Compress a block using zstd (level 3).
///
/// Returns a `Bytes` value prefixed with a 1-byte magic flag:
/// - If the input is short or compression yields no gain, the original data is
///   returned prefixed with [`MAGIC_RAW`].
/// - Otherwise the zstd-compressed payload is returned prefixed with
///   [`MAGIC_ZSTD`].
pub fn compress_block(data: &[u8]) -> Result<Bytes> {
    compress_block_with_level(data, 3)
}

/// Compress a block using zstd at the given level (1–22).
pub fn compress_block_with_level(data: &[u8], level: i32) -> Result<Bytes> {
    compress_block_with_algorithm_level(data, CompressionAlgorithm::Zstd, level)
}

/// Compress a block using the specified algorithm.
///
/// Applies the same size threshold and ratio check as `compress_block`.
pub fn compress_block_with_algorithm(
    data: &[u8],
    algo: CompressionAlgorithm,
    level: i32,
) -> Result<Bytes> {
    compress_block_with_algorithm_level(data, algo, level)
}

/// Compress with stats — returns the encoded payload **and** statistics.
pub fn compress_block_with_stats(data: &[u8]) -> Result<(Bytes, BlockCompressStats)> {
    compress_block_with_stats_inner(data, CompressionAlgorithm::Zstd, 3)
}

fn compress_block_with_stats_inner(
    data: &[u8],
    algo: CompressionAlgorithm,
    level: i32,
) -> Result<(Bytes, BlockCompressStats)> {
    let encoded = compress_block_with_algorithm_level(data, algo, level)?;
    let original_size = data.len();
    let stored_size = encoded.len();
    let compressed = encoded[0] != MAGIC_RAW;
    let ratio = if original_size == 0 {
        1.0
    } else {
        stored_size as f32 / original_size as f32
    };
    let stats = BlockCompressStats {
        original_size,
        stored_size,
        compressed,
        ratio,
    };
    Ok((encoded, stats))
}

fn compress_block_with_algorithm_level(
    data: &[u8],
    algo: CompressionAlgorithm,
    level: i32,
) -> Result<Bytes> {
    // Small blocks: always raw
    if data.len() < MIN_COMPRESS_SIZE {
        return Ok(encode_raw(data));
    }

    let compressed = run_compression(data, algo, level)?;

    // Only use compression if it actually helps
    if compressed.len() >= data.len() {
        return Ok(encode_raw(data));
    }

    Ok(Bytes::from(compressed))
}

/// Decompress a block previously encoded by [`compress_block`].
///
/// Reads the magic byte and applies the appropriate decompression.
/// Data without a recognised magic byte (i.e., old blocks written before
/// compression was introduced) is returned as-is so that transparent
/// backwards-compatibility is maintained.
pub fn decompress_block(data: &[u8]) -> Result<Bytes> {
    if data.is_empty() {
        return Ok(Bytes::new());
    }

    match data[0] {
        MAGIC_RAW => Ok(Bytes::from(data[1..].to_vec())),
        MAGIC_ZSTD => decompress_zstd(&data[1..]),
        MAGIC_LZ4 => decompress_lz4_stored(&data[1..]),
        MAGIC_SNAPPY => decompress_snappy(&data[1..]),
        _ => {
            // Backwards-compat: no magic header → stored as plain raw bytes
            // (blocks written before compression was introduced)
            Ok(Bytes::copy_from_slice(data))
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

fn encode_raw(data: &[u8]) -> Bytes {
    let mut out = Vec::with_capacity(data.len() + 1);
    out.push(MAGIC_RAW);
    out.extend_from_slice(data);
    Bytes::from(out)
}

fn run_compression(data: &[u8], algo: CompressionAlgorithm, level: i32) -> Result<Vec<u8>> {
    match algo {
        CompressionAlgorithm::Zstd => compress_zstd(data, level),
        CompressionAlgorithm::Lz4 => compress_lz4(data),
        CompressionAlgorithm::Snappy => compress_snappy(data),
    }
}

#[cfg(feature = "compression")]
fn compress_zstd(data: &[u8], level: i32) -> Result<Vec<u8>> {
    use oxiarc_zstd::compress_with_level;
    let payload = compress_with_level(data, level)
        .map_err(|e| Error::Storage(format!("oxiarc-zstd compression failed: {e}")))?;
    let mut out = Vec::with_capacity(payload.len() + 1);
    out.push(MAGIC_ZSTD);
    out.extend_from_slice(&payload);
    Ok(out)
}

#[cfg(not(feature = "compression"))]
fn compress_zstd(_data: &[u8], _level: i32) -> Result<Vec<u8>> {
    Err(Error::Storage(
        "zstd compression requires the 'compression' feature".to_string(),
    ))
}

#[cfg(feature = "compression")]
fn decompress_zstd(payload: &[u8]) -> Result<Bytes> {
    use oxiarc_zstd::decompress;
    let out = decompress(payload)
        .map_err(|e| Error::Storage(format!("oxiarc-zstd decompression failed: {e}")))?;
    Ok(Bytes::from(out))
}

#[cfg(not(feature = "compression"))]
fn decompress_zstd(_payload: &[u8]) -> Result<Bytes> {
    Err(Error::Storage(
        "zstd decompression requires the 'compression' feature".to_string(),
    ))
}

/// LZ4 stored format: [MAGIC_LZ4][original_size: 4 bytes LE][lz4_frame_data]
#[cfg(feature = "compression")]
fn compress_lz4(data: &[u8]) -> Result<Vec<u8>> {
    use oxiarc_lz4::compress;
    let payload = compress(data)
        .map_err(|e| Error::Storage(format!("oxiarc-lz4 compression failed: {e}")))?;
    let orig_len = data.len() as u32;
    let mut out = Vec::with_capacity(1 + 4 + payload.len());
    out.push(MAGIC_LZ4);
    out.extend_from_slice(&orig_len.to_le_bytes());
    out.extend_from_slice(&payload);
    Ok(out)
}

#[cfg(not(feature = "compression"))]
fn compress_lz4(_data: &[u8]) -> Result<Vec<u8>> {
    Err(Error::Storage(
        "lz4 compression requires the 'compression' feature".to_string(),
    ))
}

/// `stored` is the bytes *after* the MAGIC_LZ4 byte, i.e. [original_size_u32][lz4_data].
#[cfg(feature = "compression")]
fn decompress_lz4_stored(stored: &[u8]) -> Result<Bytes> {
    use oxiarc_lz4::decompress;
    if stored.len() < 4 {
        return Err(Error::Storage(
            "LZ4 stored block too short (missing original size)".to_string(),
        ));
    }
    let orig_size = u32::from_le_bytes([stored[0], stored[1], stored[2], stored[3]]) as usize;
    // Generous upper bound: 2× original size avoids truncation for edge cases
    let max_output = orig_size.saturating_mul(2).max(orig_size + 64);
    let out = decompress(&stored[4..], max_output)
        .map_err(|e| Error::Storage(format!("oxiarc-lz4 decompression failed: {e}")))?;
    Ok(Bytes::from(out))
}

#[cfg(not(feature = "compression"))]
fn decompress_lz4_stored(_stored: &[u8]) -> Result<Bytes> {
    Err(Error::Storage(
        "lz4 decompression requires the 'compression' feature".to_string(),
    ))
}

#[cfg(feature = "compression")]
fn compress_snappy(data: &[u8]) -> Result<Vec<u8>> {
    use oxiarc_snappy::compress;
    let payload = compress(data);
    let mut out = Vec::with_capacity(payload.len() + 1);
    out.push(MAGIC_SNAPPY);
    out.extend_from_slice(&payload);
    Ok(out)
}

#[cfg(not(feature = "compression"))]
fn compress_snappy(_data: &[u8]) -> Result<Vec<u8>> {
    Err(Error::Storage(
        "snappy compression requires the 'compression' feature".to_string(),
    ))
}

#[cfg(feature = "compression")]
fn decompress_snappy(payload: &[u8]) -> Result<Bytes> {
    use oxiarc_snappy::decompress;
    let out = decompress(payload)
        .map_err(|e| Error::Storage(format!("oxiarc-snappy decompression failed: {e}")))?;
    Ok(Bytes::from(out))
}

#[cfg(not(feature = "compression"))]
fn decompress_snappy(_payload: &[u8]) -> Result<Bytes> {
    Err(Error::Storage(
        "snappy decompression requires the 'compression' feature".to_string(),
    ))
}

// ─────────────────────────────────────────────────────────────────────────────
// Public config types
// ─────────────────────────────────────────────────────────────────────────────

/// Compression algorithm used by the storage layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CompressionAlgorithm {
    /// Zstd — best ratio, fast decompression. **Default.** (pure Rust via oxiarc-zstd)
    #[default]
    Zstd,
    /// LZ4 — very fast, moderate ratio. (pure Rust via oxiarc-lz4)
    Lz4,
    /// Snappy — fast, streaming-friendly. (pure Rust via oxiarc-snappy)
    Snappy,
}

/// Compression configuration for [`CompressionBlockStore`].
#[derive(Debug, Clone)]
pub struct CompressionConfig {
    /// Algorithm to use.
    pub algorithm: CompressionAlgorithm,
    /// Compression level (algorithm-specific).
    /// - Zstd: 1–22 (default: 3)
    /// - Lz4/Snappy: ignored
    pub level: i32,
    /// Only compress blocks larger than this threshold (bytes).
    /// Default: 256 bytes.
    pub threshold: usize,
    /// Maximum compression ratio to accept (`compressed / original`).
    /// If compression does not beat this, the block is stored raw.
    /// Default: 0.9 (store compressed only if we save ≥ 10 %).
    pub max_ratio: f64,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            algorithm: CompressionAlgorithm::default(),
            level: 3,
            threshold: MIN_COMPRESS_SIZE,
            max_ratio: 0.9,
        }
    }
}

impl CompressionConfig {
    /// Create a new configuration using the specified algorithm.
    pub fn new(algorithm: CompressionAlgorithm) -> Self {
        Self {
            algorithm,
            ..Default::default()
        }
    }

    /// Override the compression level.
    pub fn with_level(mut self, level: i32) -> Self {
        self.level = level;
        self
    }

    /// Override the minimum size threshold.
    pub fn with_threshold(mut self, threshold: usize) -> Self {
        self.threshold = threshold;
        self
    }

    /// Override the maximum acceptable compression ratio.
    pub fn with_max_ratio(mut self, max_ratio: f64) -> Self {
        self.max_ratio = max_ratio;
        self
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Aggregate statistics
// ─────────────────────────────────────────────────────────────────────────────

/// Aggregate compression statistics accumulated by [`CompressionBlockStore`].
#[derive(Debug, Clone, Default)]
pub struct BlockCompressionStats {
    /// Blocks that were stored in compressed form.
    pub blocks_compressed: u64,
    /// Blocks stored without compression (below threshold or poor ratio).
    pub blocks_uncompressed: u64,
    /// Sum of original sizes before compression.
    pub bytes_original: u64,
    /// Sum of sizes actually stored (post-compression if any).
    pub bytes_compressed: u64,
    /// Total decompression operations performed on read.
    pub decompressions: u64,
}

impl BlockCompressionStats {
    /// Ratio of stored bytes to original bytes (lower = better compression).
    pub fn compression_ratio(&self) -> f64 {
        if self.bytes_original == 0 {
            return 1.0;
        }
        self.bytes_compressed as f64 / self.bytes_original as f64
    }

    /// Bytes saved overall.
    pub fn bytes_saved(&self) -> u64 {
        self.bytes_original.saturating_sub(self.bytes_compressed)
    }

    /// Percentage of bytes saved.
    pub fn savings_percent(&self) -> f64 {
        if self.bytes_original == 0 {
            return 0.0;
        }
        (self.bytes_saved() as f64 / self.bytes_original as f64) * 100.0
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CompressionBlockStore — transparent wrapper
// ─────────────────────────────────────────────────────────────────────────────

/// A `BlockStore` decorator that transparently compresses data on write and
/// decompresses on read.
///
/// Uses the same magic-byte framing as the low-level [`compress_block`] /
/// [`decompress_block`] functions, so the two APIs are fully interoperable.
pub struct CompressionBlockStore<S> {
    inner: S,
    config: CompressionConfig,
    stats: Arc<RwLock<BlockCompressionStats>>,
}

impl<S> CompressionBlockStore<S> {
    /// Wrap an existing block store with transparent compression.
    pub fn new(inner: S, config: CompressionConfig) -> Self {
        Self {
            inner,
            config,
            stats: Arc::new(RwLock::new(BlockCompressionStats::default())),
        }
    }

    /// Snapshot of current compression statistics.
    pub fn stats(&self) -> BlockCompressionStats {
        self.stats.read().clone()
    }

    /// Reset accumulated statistics to zero.
    pub fn reset_stats(&self) {
        let mut stats = self.stats.write();
        *stats = BlockCompressionStats::default();
    }

    /// Compress `data` according to `config`.
    ///
    /// Returns encoded bytes (magic prefix + payload).
    fn compress(&self, data: &[u8]) -> Result<Vec<u8>> {
        // Below threshold → always raw
        if data.len() < self.config.threshold {
            return Ok(encode_raw(data).to_vec());
        }

        let compressed_bytes = run_compression(data, self.config.algorithm, self.config.level)?;

        // Check whether the ratio is good enough to bother
        let ratio = compressed_bytes.len() as f64 / (data.len() + 1) as f64; // +1 for the magic byte we haven't added yet in raw case
        if ratio > self.config.max_ratio {
            return Ok(encode_raw(data).to_vec());
        }

        Ok(compressed_bytes)
    }

    /// Decompress stored bytes. See [`decompress_block`] for the format.
    fn decompress(&self, data: &[u8]) -> Result<Vec<u8>> {
        decompress_block(data).map(|b| b.to_vec())
    }
}

#[async_trait]
impl<S: BlockStore + Send + Sync> BlockStore for CompressionBlockStore<S> {
    async fn put(&self, block: &Block) -> Result<()> {
        let original_size = block.data().len();
        let compressed = self.compress(block.data())?;
        let compressed_size = compressed.len();

        {
            let mut stats = self.stats.write();
            stats.bytes_original += original_size as u64;
            stats.bytes_compressed += compressed_size as u64;

            if compressed[0] == MAGIC_RAW {
                stats.blocks_uncompressed += 1;
            } else {
                stats.blocks_compressed += 1;
            }
        }

        let compressed_block = Block::from_parts(*block.cid(), compressed.into());
        self.inner.put(&compressed_block).await
    }

    async fn put_many(&self, blocks: &[Block]) -> Result<()> {
        let mut compressed_blocks = Vec::with_capacity(blocks.len());

        {
            let mut stats = self.stats.write();
            for block in blocks {
                let original_size = block.data().len();
                let compressed = self.compress(block.data())?;
                let compressed_size = compressed.len();

                stats.bytes_original += original_size as u64;
                stats.bytes_compressed += compressed_size as u64;

                if compressed[0] == MAGIC_RAW {
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
        if let Some(stored_block) = self.inner.get(cid).await? {
            let data = self.decompress(stored_block.data())?;
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
        let stored = self.inner.get_many(cids).await?;
        let mut results = Vec::with_capacity(stored.len());
        let mut decompression_count: u64 = 0;

        for (i, item) in stored.into_iter().enumerate() {
            if let Some(stored_block) = item {
                let data = self.decompress(stored_block.data())?;
                decompression_count += 1;
                results.push(Some(Block::from_parts(cids[i], data.into())));
            } else {
                results.push(None);
            }
        }

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

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(all(test, feature = "sled-backend"))]
mod tests {
    use super::*;
    use crate::blockstore::SledBlockStore;

    // ── Low-level compress_block / decompress_block ───────────────────────

    #[cfg(feature = "compression")]
    #[test]
    fn test_compress_decompress_roundtrip() {
        let data = b"Hello, IPFRS! This is a test block for compression. ".repeat(20);
        let encoded = compress_block(&data).expect("compress_block should succeed");
        let decoded = decompress_block(&encoded).expect("decompress_block should succeed");
        assert_eq!(decoded.as_ref(), data.as_slice());
    }

    #[cfg(feature = "compression")]
    #[test]
    fn test_small_blocks_not_compressed() {
        // Blocks below MIN_COMPRESS_SIZE must be stored raw (MAGIC_RAW prefix)
        let small = b"tiny";
        let encoded = compress_block(small).expect("compress_block should succeed");
        assert_eq!(encoded[0], MAGIC_RAW, "small block must use MAGIC_RAW");
        let decoded = decompress_block(&encoded).expect("decompress_block should succeed");
        assert_eq!(decoded.as_ref(), small.as_slice());
    }

    #[cfg(feature = "compression")]
    #[test]
    fn test_compression_ratio_for_json() {
        // Highly repetitive JSON — should compress well (target < 30 % of original)
        let json =
            r#"{"key":"value","items":[1,2,3,4,5],"nested":{"a":true,"b":false}}"#.repeat(50);
        let (encoded, stats) = compress_block_with_stats(json.as_bytes()).expect("should succeed");
        assert!(
            stats.compressed,
            "repetitive JSON should be compressed, ratio={:.3}",
            stats.ratio
        );
        assert!(
            stats.ratio < 0.5,
            "JSON compression ratio should be < 50 %, got {:.1} %",
            stats.ratio * 100.0
        );
        // Verify round-trip
        let decoded = decompress_block(&encoded).expect("decompress should succeed");
        assert_eq!(decoded.as_ref(), json.as_bytes());
    }

    #[cfg(feature = "compression")]
    #[test]
    fn test_incompressible_data_stored_raw() {
        // Pseudorandom (incompressible) data — encoder should fall back to raw
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut data = Vec::with_capacity(1024);
        for i in 0u64..512 {
            let mut h = DefaultHasher::new();
            i.hash(&mut h);
            let v = h.finish();
            data.extend_from_slice(&v.to_le_bytes());
        }
        assert!(data.len() >= MIN_COMPRESS_SIZE);

        let encoded = compress_block(&data).expect("compress_block should succeed");
        // Either raw or compressed — the only requirement is correct round-trip
        let decoded = decompress_block(&encoded).expect("decompress_block should succeed");
        assert_eq!(decoded.as_ref(), data.as_slice());
    }

    #[cfg(feature = "compression")]
    #[test]
    fn test_backwards_compat_no_magic() {
        // Data stored *before* compression was introduced (no magic byte) must
        // decode transparently — decompress_block must return the original bytes.
        let raw_legacy = b"legacy raw data stored without magic prefix";
        let decoded = decompress_block(raw_legacy).expect("should handle legacy data");
        assert_eq!(decoded.as_ref(), raw_legacy.as_slice());
    }

    #[cfg(feature = "compression")]
    #[test]
    fn test_lz4_roundtrip() {
        let data = b"LZ4 block compression test. ".repeat(30);
        let encoded = compress_block_with_algorithm(&data, CompressionAlgorithm::Lz4, 1)
            .expect("lz4 compress should succeed");
        let decoded = decompress_block(&encoded).expect("lz4 decompress should succeed");
        assert_eq!(decoded.as_ref(), data.as_slice());
    }

    #[cfg(feature = "compression")]
    #[test]
    fn test_snappy_roundtrip() {
        let data = b"Snappy block compression test. ".repeat(30);
        let encoded = compress_block_with_algorithm(&data, CompressionAlgorithm::Snappy, 0)
            .expect("snappy compress should succeed");
        let decoded = decompress_block(&encoded).expect("snappy decompress should succeed");
        assert_eq!(decoded.as_ref(), data.as_slice());
    }

    // ── CompressionBlockStore (wrapper) ───────────────────────────────────

    #[cfg(feature = "compression")]
    #[tokio::test]
    async fn test_compression_basic() {
        let temp_dir =
            std::env::temp_dir().join(format!("ipfrs-compr-basic-{}", std::process::id()));
        let config = crate::BlockStoreConfig {
            path: temp_dir.clone(),
            cache_size: 10_000_000,
        };
        let store = SledBlockStore::new(config).expect("open sled");
        let comp_config = CompressionConfig::new(CompressionAlgorithm::Zstd);
        let compressed_store = CompressionBlockStore::new(store, comp_config);

        let data = vec![42u8; 10_000]; // Highly compressible
        let block = Block::new(data.clone().into()).expect("create block");

        compressed_store.put(&block).await.expect("put");
        let retrieved = compressed_store
            .get(block.cid())
            .await
            .expect("get")
            .expect("should exist");
        assert_eq!(data.as_slice(), retrieved.data().as_ref());

        let stats = compressed_store.stats();
        assert_eq!(stats.blocks_compressed, 1, "block should be compressed");
        assert!(
            stats.compression_ratio() < 0.1,
            "all-same-byte block should compress extremely well"
        );

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[cfg(feature = "compression")]
    #[tokio::test]
    async fn test_compression_threshold() {
        let temp_dir =
            std::env::temp_dir().join(format!("ipfrs-compr-thresh-{}", std::process::id()));
        let store_config = crate::BlockStoreConfig {
            path: temp_dir.clone(),
            cache_size: 10_000_000,
        };
        let store = SledBlockStore::new(store_config).expect("open sled");
        let comp_config = CompressionConfig::new(CompressionAlgorithm::Zstd).with_threshold(1_000);
        let compressed_store = CompressionBlockStore::new(store, comp_config);

        // Small block (below threshold)
        let small_data = vec![42u8; 100];
        let block1 = Block::new(small_data.into()).expect("block1");
        compressed_store.put(&block1).await.expect("put block1");

        // Large block (above threshold)
        let large_data = vec![42u8; 10_000];
        let block2 = Block::new(large_data.into()).expect("block2");
        compressed_store.put(&block2).await.expect("put block2");

        let stats = compressed_store.stats();
        assert_eq!(stats.blocks_uncompressed, 1, "small block should be raw");
        assert_eq!(
            stats.blocks_compressed, 1,
            "large block should be compressed"
        );

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[cfg(feature = "compression")]
    #[tokio::test]
    async fn test_compression_is_transparent_in_store() {
        // Write via wrapper, read via wrapper → data must round-trip perfectly.
        let temp_dir =
            std::env::temp_dir().join(format!("ipfrs-compr-transp-{}", std::process::id()));
        let store_config = crate::BlockStoreConfig {
            path: temp_dir.clone(),
            cache_size: 10_000_000,
        };
        let json_data = r#"{"cid":"bafybeig","links":[],"data":"aGVsbG8gd29ybGQ="}"#.repeat(40);
        let store = SledBlockStore::new(store_config).expect("open sled");
        let comp_config = CompressionConfig::new(CompressionAlgorithm::Zstd);
        let compressed_store = CompressionBlockStore::new(store, comp_config);

        let block = Block::new(json_data.as_bytes().to_vec().into()).expect("block");
        compressed_store.put(&block).await.expect("put");

        let retrieved = compressed_store
            .get(block.cid())
            .await
            .expect("get")
            .expect("should exist");
        assert_eq!(retrieved.data().as_ref(), json_data.as_bytes());

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[cfg(feature = "compression")]
    #[tokio::test]
    async fn test_compression_algorithms() {
        for algorithm in [
            CompressionAlgorithm::Zstd,
            CompressionAlgorithm::Lz4,
            CompressionAlgorithm::Snappy,
        ] {
            let temp_dir = std::env::temp_dir().join(format!(
                "ipfrs-compr-algo-{:?}-{}",
                algorithm,
                std::process::id()
            ));
            let store_config = crate::BlockStoreConfig {
                path: temp_dir.clone(),
                cache_size: 10_000_000,
            };
            let store = SledBlockStore::new(store_config).expect("open sled");
            let comp_config = CompressionConfig::new(algorithm);
            let compressed_store = CompressionBlockStore::new(store, comp_config);

            let data = vec![42u8; 10_000];
            let block = Block::new(data.clone().into()).expect("block");

            compressed_store.put(&block).await.expect("put");
            let retrieved = compressed_store
                .get(block.cid())
                .await
                .expect("get")
                .expect("should exist");
            assert_eq!(data.as_slice(), retrieved.data().as_ref());

            let _ = std::fs::remove_dir_all(&temp_dir);
        }
    }

    #[cfg(feature = "compression")]
    #[tokio::test]
    async fn test_compression_batch() {
        let temp_dir =
            std::env::temp_dir().join(format!("ipfrs-compr-batch-{}", std::process::id()));
        let store_config = crate::BlockStoreConfig {
            path: temp_dir.clone(),
            cache_size: 10_000_000,
        };
        let store = SledBlockStore::new(store_config).expect("open sled");
        let comp_config = CompressionConfig::new(CompressionAlgorithm::Zstd);
        let compressed_store = CompressionBlockStore::new(store, comp_config);

        let blocks: Vec<_> = (0u8..10)
            .map(|i| Block::new(vec![i; 5_000].into()).expect("block"))
            .collect();

        compressed_store.put_many(&blocks).await.expect("put_many");

        let cids: Vec<_> = blocks.iter().map(|b| *b.cid()).collect();
        let retrieved = compressed_store.get_many(&cids).await.expect("get_many");

        for (i, item) in retrieved.iter().enumerate() {
            let block = item.as_ref().expect("block should exist");
            assert_eq!(block.data(), blocks[i].data());
        }

        let stats = compressed_store.stats();
        assert_eq!(stats.blocks_compressed, 10);
        assert_eq!(stats.decompressions, 10);

        let _ = std::fs::remove_dir_all(&temp_dir);
    }
}
