//! `StorageCompressionPipeline` — configurable multi-stage compression pipeline.
//!
//! Selects and applies compression algorithms based on content type, size heuristics,
//! and compression ratio targets. All algorithms are implemented in pure Rust with no
//! external compression crates.
//!
//! # Algorithms
//! - **None** — identity passthrough
//! - **Rle** — Run-Length Encoding (consecutive byte pairs)
//! - **Lz4** — LZ77-style sliding-window compressor (simplified, round-trip only)
//! - **Zstd** — dispatches to RLE (level ≤ 3) or LZ4-style (level ≥ 4)
//! - **Snappy** — synonym for the LZ4-style algorithm

use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// Error
// ─────────────────────────────────────────────────────────────────────────────

/// Errors that can arise during decompression or algorithm dispatch.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum CompressionError {
    /// The compressed stream could not be decoded.
    #[error("decompression failed: {0}")]
    DecompressionFailed(String),
    /// The requested algorithm name is not recognised.
    #[error("unknown algorithm: {0}")]
    UnknownAlgo(String),
    /// The compressed payload is structurally corrupt.
    #[error("corrupted data")]
    CorruptedData,
}

// ─────────────────────────────────────────────────────────────────────────────
// CompressionAlgo
// ─────────────────────────────────────────────────────────────────────────────

/// Compression algorithm used (or to be used) by a pipeline stage.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CompressionAlgo {
    /// No compression — data is stored as-is.
    None,
    /// LZ77-style sliding-window compressor (pure Rust, simplified).
    Lz4,
    /// Zstandard-inspired: dispatches to RLE for level ≤ 3, LZ4-style for level ≥ 4.
    Zstd {
        /// Compression level (1 = fastest/worst, 22 = slowest/best).
        level: i32,
    },
    /// Snappy-inspired: same algorithm as [`Lz4`](Self::Lz4).
    Snappy,
    /// Run-Length Encoding — pairs of (count, byte).
    Rle,
}

impl CompressionAlgo {
    /// A stable, human-readable name for this algorithm (used as map key).
    pub fn name(&self) -> String {
        match self {
            Self::None => "none".to_string(),
            Self::Lz4 => "lz4".to_string(),
            Self::Zstd { level } => format!("zstd({})", level),
            Self::Snappy => "snappy".to_string(),
            Self::Rle => "rle".to_string(),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CompressionHint
// ─────────────────────────────────────────────────────────────────────────────

/// Content-type hint used by the pipeline to bias algorithm selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionHint {
    /// Human-readable text (valid UTF-8).
    Text,
    /// Structured data (JSON / CBOR-like; first byte is `{` or `[`).
    Structured,
    /// Arbitrary binary data.
    Binary,
    /// Unknown content — detect automatically.
    Unknown,
}

// ─────────────────────────────────────────────────────────────────────────────
// CompressionResult
// ─────────────────────────────────────────────────────────────────────────────

/// Output of a single compression operation.
#[derive(Debug, Clone)]
pub struct CompressionResult {
    /// The algorithm that was applied.
    pub algo: CompressionAlgo,
    /// Size of the input data in bytes.
    pub original_size: usize,
    /// Size of the compressed output in bytes.
    pub compressed_size: usize,
    /// `compressed_size / original_size` (lower is better; 1.0 = no saving).
    pub ratio: f64,
    /// The compressed payload.
    pub data: Vec<u8>,
}

impl CompressionResult {
    fn new(algo: CompressionAlgo, original_size: usize, data: Vec<u8>) -> Self {
        let compressed_size = data.len();
        let ratio = if original_size == 0 {
            1.0
        } else {
            compressed_size as f64 / original_size as f64
        };
        Self {
            algo,
            original_size,
            compressed_size,
            ratio,
            data,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PipelineStage
// ─────────────────────────────────────────────────────────────────────────────

/// A single stage inside a [`PipelineConfig`].
///
/// The stage is attempted only when both size and ratio pre-conditions are met.
#[derive(Debug, Clone)]
pub struct PipelineStage {
    /// The compression algorithm for this stage.
    pub algo: CompressionAlgo,
    /// Minimum input size (bytes) required before this stage is tried.
    pub min_input_size: usize,
    /// Maximum acceptable ratio (`compressed / original`).  If the actual ratio
    /// exceeds this threshold the stage result is discarded.
    pub max_ratio: f64,
}

// ─────────────────────────────────────────────────────────────────────────────
// PipelineConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Runtime configuration for [`StorageCompressionPipeline`].
///
/// Re-exported as `CpPipelineConfig` from `lib.rs` to avoid collision with
/// `deduplication_pipeline::PipelineConfig`.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Ordered list of stages to try in sequence.
    pub stages: Vec<PipelineStage>,
    /// Algorithm used when no stage accepts the data.
    pub fallback_algo: CompressionAlgo,
    /// Global target ratio.  Used for statistics but does not gate individual stages.
    pub target_ratio: f64,
    /// When `true`, stages whose ratio exceeds `stage.max_ratio` are skipped.
    pub enable_ratio_check: bool,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            stages: vec![
                PipelineStage {
                    algo: CompressionAlgo::Rle,
                    min_input_size: 64,
                    max_ratio: 0.9,
                },
                PipelineStage {
                    algo: CompressionAlgo::Lz4,
                    min_input_size: 256,
                    max_ratio: 0.85,
                },
                PipelineStage {
                    algo: CompressionAlgo::Zstd { level: 3 },
                    min_input_size: 512,
                    max_ratio: 0.7,
                },
            ],
            fallback_algo: CompressionAlgo::None,
            target_ratio: 0.75,
            enable_ratio_check: true,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PipelineStats
// ─────────────────────────────────────────────────────────────────────────────

/// Aggregated statistics for a [`StorageCompressionPipeline`] instance.
#[derive(Debug, Clone)]
pub struct PipelineStats {
    /// Number of compress calls executed.
    pub total_compressed: u64,
    /// Total raw bytes fed into the pipeline.
    pub total_bytes_in: u64,
    /// Total compressed bytes emitted by the pipeline.
    pub total_bytes_out: u64,
    /// Overall average ratio (`total_bytes_out / total_bytes_in`).
    pub avg_ratio: f64,
    /// Per-algorithm usage counters (key = `CompressionAlgo::name()`).
    pub algo_usage: HashMap<String, u64>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Pure-Rust algorithm implementations
// ─────────────────────────────────────────────────────────────────────────────

// ── RLE ──────────────────────────────────────────────────────────────────────

/// Compress `data` using Run-Length Encoding.
///
/// Emits (count: u8, byte: u8) pairs.  Each run is capped at 255 bytes.
fn rle_compress(data: &[u8]) -> Vec<u8> {
    if data.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(data.len());
    let mut i = 0;
    while i < data.len() {
        let byte = data[i];
        let mut count: usize = 1;
        while i + count < data.len() && data[i + count] == byte && count < 255 {
            count += 1;
        }
        out.push(count as u8);
        out.push(byte);
        i += count;
    }
    out
}

/// Decompress an RLE stream produced by [`rle_compress`].
fn rle_decompress(data: &[u8]) -> Result<Vec<u8>, CompressionError> {
    if !data.len().is_multiple_of(2) {
        return Err(CompressionError::CorruptedData);
    }
    let mut out = Vec::with_capacity(data.len() * 2);
    let mut i = 0;
    while i + 1 < data.len() {
        let count = data[i] as usize;
        let byte = data[i + 1];
        if count == 0 {
            return Err(CompressionError::DecompressionFailed(
                "zero-length RLE run".to_string(),
            ));
        }
        for _ in 0..count {
            out.push(byte);
        }
        i += 2;
    }
    Ok(out)
}

// ── LZ4-style (LZ77 sliding window) ─────────────────────────────────────────

/// Window size for the LZ4-style compressor.
const LZ4_WINDOW: usize = 4096;
/// Minimum match length (bytes) to emit a back-reference.
const LZ4_MIN_MATCH: usize = 4;

/// FNV-1a hash of a 4-byte sequence (used by the LZ4-style hash table).
#[inline]
fn fnv1a_4(data: &[u8]) -> u32 {
    const OFFSET: u32 = 2_166_136_261;
    const PRIME: u32 = 16_777_619;
    let mut h = OFFSET;
    for &b in data.iter().take(4) {
        h ^= u32::from(b);
        h = h.wrapping_mul(PRIME);
    }
    h
}

/// Compress `data` using an LZ77-style sliding-window algorithm.
///
/// Output format: a sequence of tokens, where each token is:
/// - `offset` (u16 LE) + `length` (u8): back-reference — `length > 0` and `offset > 0`
/// - `0u16` + `0u8` + one literal byte: literal token
fn lz4_style_compress(data: &[u8]) -> Vec<u8> {
    if data.is_empty() {
        return Vec::new();
    }

    // Hash table: key = fnv1a_4(seq) % TABLE_SIZE → last position seen
    const TABLE_SIZE: usize = 4096;
    let mut table: [usize; TABLE_SIZE] = [usize::MAX; TABLE_SIZE];

    let mut out = Vec::with_capacity(data.len());
    let mut pos = 0;

    while pos < data.len() {
        // Need at least 4 bytes for a potential match
        if pos + LZ4_MIN_MATCH <= data.len() {
            let seq = &data[pos..pos + 4];
            let idx = fnv1a_4(seq) as usize % TABLE_SIZE;
            let candidate = table[idx];
            table[idx] = pos;

            if candidate != usize::MAX && candidate < pos {
                let offset = pos - candidate;
                if offset <= LZ4_WINDOW {
                    // Measure match length
                    let max_len = (data.len() - pos).min(255);
                    let mut match_len = 0;
                    while match_len < max_len
                        && data[candidate + match_len] == data[pos + match_len]
                    {
                        match_len += 1;
                    }

                    if match_len >= LZ4_MIN_MATCH {
                        // Emit back-reference token
                        let off = offset as u16;
                        out.extend_from_slice(&off.to_le_bytes());
                        out.push(match_len as u8);
                        pos += match_len;
                        continue;
                    }
                }
            }
        }

        // Emit literal token
        out.extend_from_slice(&0u16.to_le_bytes());
        out.push(0u8);
        out.push(data[pos]);
        pos += 1;
    }

    out
}

/// Decompress a stream produced by [`lz4_style_compress`].
fn lz4_style_decompress(data: &[u8]) -> Result<Vec<u8>, CompressionError> {
    let mut out: Vec<u8> = Vec::with_capacity(data.len() * 3);
    let mut i = 0;

    while i + 2 < data.len() {
        let offset = u16::from_le_bytes(
            data[i..i + 2]
                .try_into()
                .map_err(|_| CompressionError::CorruptedData)?,
        ) as usize;
        let length = data[i + 2] as usize;
        i += 3;

        if length == 0 {
            // Literal token: consume one byte
            if i >= data.len() {
                return Err(CompressionError::DecompressionFailed(
                    "truncated literal token".to_string(),
                ));
            }
            out.push(data[i]);
            i += 1;
        } else {
            // Back-reference token
            if offset == 0 {
                return Err(CompressionError::DecompressionFailed(
                    "zero offset in back-reference".to_string(),
                ));
            }
            let start = out
                .len()
                .checked_sub(offset)
                .ok_or(CompressionError::CorruptedData)?;
            for k in 0..length {
                let b = out
                    .get(start + k)
                    .copied()
                    .ok_or(CompressionError::CorruptedData)?;
                out.push(b);
            }
        }
    }

    // Any remaining bytes that do not form a complete token are silently ignored
    // (this can happen if the stream ends without a literal after a token header —
    // in practice our encoder never produces this).
    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// StorageCompressionPipeline
// ─────────────────────────────────────────────────────────────────────────────

/// Multi-stage compression pipeline with content-aware algorithm selection.
///
/// # Example
/// ```rust
/// use ipfrs_storage::compression_pipeline::{
///     StorageCompressionPipeline, PipelineConfig, CompressionHint,
/// };
///
/// let mut pipeline = StorageCompressionPipeline::new(PipelineConfig::default());
/// let data = b"aaaaaabbbbbbcccccccccdddddddddeeeeeeeeeee";
/// let result = pipeline.compress(data, CompressionHint::Binary);
/// let recovered = pipeline.decompress(&result.data, &result.algo).unwrap();
/// assert_eq!(&recovered, data);
/// ```
#[derive(Debug)]
pub struct StorageCompressionPipeline {
    /// Pipeline configuration.
    pub config: PipelineConfig,
    /// Number of successful compress calls.
    pub total_compressed: u64,
    /// Total raw bytes fed into the pipeline.
    pub total_bytes_in: u64,
    /// Total compressed bytes emitted by the pipeline.
    pub total_bytes_out: u64,
    /// Per-algorithm usage counters.
    pub algo_counters: HashMap<String, u64>,
}

impl StorageCompressionPipeline {
    /// Create a new pipeline with the supplied configuration.
    pub fn new(config: PipelineConfig) -> Self {
        Self {
            config,
            total_compressed: 0,
            total_bytes_in: 0,
            total_bytes_out: 0,
            algo_counters: HashMap::new(),
        }
    }

    // ── Core methods ─────────────────────────────────────────────────────────

    /// Compress `data` using the first pipeline stage whose conditions are met.
    ///
    /// Stages are tried in the order defined in [`PipelineConfig::stages`].  A
    /// stage is skipped if:
    /// - `data.len() < stage.min_input_size`, or
    /// - `config.enable_ratio_check` is `true` and the resulting ratio exceeds
    ///   `stage.max_ratio`.
    ///
    /// If no stage accepts the data, the `fallback_algo` is used.
    pub fn compress(&mut self, data: &[u8], _hint: CompressionHint) -> CompressionResult {
        let original_size = data.len();

        let result = self.try_stages(data).unwrap_or_else(|| {
            let compressed = self.compress_with_algo(data, &self.config.fallback_algo.clone());
            CompressionResult::new(self.config.fallback_algo.clone(), original_size, compressed)
        });

        // Update counters
        self.total_compressed += 1;
        self.total_bytes_in += original_size as u64;
        self.total_bytes_out += result.compressed_size as u64;
        *self.algo_counters.entry(result.algo.name()).or_insert(0) += 1;

        result
    }

    /// Try each stage in order; return the first acceptable result.
    fn try_stages(&self, data: &[u8]) -> Option<CompressionResult> {
        for stage in &self.config.stages {
            if data.len() < stage.min_input_size {
                continue;
            }
            let compressed = self.compress_with_algo(data, &stage.algo);
            let ratio = if data.is_empty() {
                1.0
            } else {
                compressed.len() as f64 / data.len() as f64
            };
            if !self.config.enable_ratio_check || ratio <= stage.max_ratio {
                return Some(CompressionResult::new(
                    stage.algo.clone(),
                    data.len(),
                    compressed,
                ));
            }
        }
        None
    }

    /// Decompress `data` using the specified algorithm.
    pub fn decompress(
        &self,
        data: &[u8],
        algo: &CompressionAlgo,
    ) -> Result<Vec<u8>, CompressionError> {
        match algo {
            CompressionAlgo::None => Ok(data.to_vec()),
            CompressionAlgo::Rle => rle_decompress(data),
            CompressionAlgo::Lz4 | CompressionAlgo::Snappy => lz4_style_decompress(data),
            CompressionAlgo::Zstd { level } => {
                if *level <= 3 {
                    rle_decompress(data)
                } else {
                    lz4_style_decompress(data)
                }
            }
        }
    }

    /// Compress `data` with a specific algorithm, returning the raw bytes.
    pub fn compress_with_algo(&self, data: &[u8], algo: &CompressionAlgo) -> Vec<u8> {
        match algo {
            CompressionAlgo::None => data.to_vec(),
            CompressionAlgo::Rle => rle_compress(data),
            CompressionAlgo::Lz4 | CompressionAlgo::Snappy => lz4_style_compress(data),
            CompressionAlgo::Zstd { level } => {
                if *level <= 3 {
                    rle_compress(data)
                } else {
                    lz4_style_compress(data)
                }
            }
        }
    }

    /// Detect content hint from the first few bytes of `data`.
    pub fn detect_hint(data: &[u8]) -> CompressionHint {
        if data.is_empty() {
            return CompressionHint::Unknown;
        }
        if std::str::from_utf8(data).is_ok() {
            let first = data[0];
            if first == b'{' || first == b'[' {
                return CompressionHint::Structured;
            }
            return CompressionHint::Text;
        }
        CompressionHint::Binary
    }

    /// Automatically detect the content hint and then compress.
    pub fn auto_compress(&mut self, data: &[u8]) -> CompressionResult {
        let hint = Self::detect_hint(data);
        self.compress(data, hint)
    }

    /// Return a snapshot of the current pipeline statistics.
    pub fn pipeline_stats(&self) -> PipelineStats {
        let avg_ratio = if self.total_bytes_in == 0 {
            1.0
        } else {
            self.total_bytes_out as f64 / self.total_bytes_in as f64
        };
        PipelineStats {
            total_compressed: self.total_compressed,
            total_bytes_in: self.total_bytes_in,
            total_bytes_out: self.total_bytes_out,
            avg_ratio,
            algo_usage: self.algo_counters.clone(),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::compression_pipeline::{
        lz4_style_compress, lz4_style_decompress, rle_compress, rle_decompress, CompressionAlgo,
        CompressionError, CompressionHint, CompressionResult, PipelineConfig, PipelineStage,
        PipelineStats, StorageCompressionPipeline,
    };

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn make_pipeline() -> StorageCompressionPipeline {
        StorageCompressionPipeline::new(PipelineConfig::default())
    }

    fn repeated_data(byte: u8, count: usize) -> Vec<u8> {
        vec![byte; count]
    }

    fn alternating_data(len: usize) -> Vec<u8> {
        (0..len).map(|i| (i % 256) as u8).collect()
    }

    // ── RLE unit tests ────────────────────────────────────────────────────────

    #[test]
    fn test_rle_empty_input() {
        let compressed = rle_compress(&[]);
        assert!(compressed.is_empty());
        let decompressed = rle_decompress(&compressed).unwrap();
        assert!(decompressed.is_empty());
    }

    #[test]
    fn test_rle_single_byte() {
        let data = [0xABu8];
        let c = rle_compress(&data);
        // One run of length 1 → (1, 0xAB)
        assert_eq!(c, vec![1, 0xAB]);
        assert_eq!(rle_decompress(&c).unwrap(), data);
    }

    #[test]
    fn test_rle_uniform_run() {
        let data = repeated_data(0x42, 100);
        let c = rle_compress(&data);
        // One run capped at 100 (< 255) → 2 bytes
        assert_eq!(c.len(), 2);
        assert_eq!(rle_decompress(&c).unwrap(), data);
    }

    #[test]
    fn test_rle_max_run_cap() {
        // 256 identical bytes → two runs: (255, b) + (1, b)
        let data = repeated_data(0xCC, 256);
        let c = rle_compress(&data);
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 255);
        assert_eq!(c[2], 1);
        assert_eq!(rle_decompress(&c).unwrap(), data);
    }

    #[test]
    fn test_rle_round_trip_alternating() {
        let data: Vec<u8> = (0u8..=127).collect();
        let c = rle_compress(&data);
        assert_eq!(rle_decompress(&c).unwrap(), data);
    }

    #[test]
    fn test_rle_odd_length_corrupt() {
        // Odd-length RLE stream must fail
        let result = rle_decompress(&[1, 2, 3]);
        assert!(matches!(result, Err(CompressionError::CorruptedData)));
    }

    #[test]
    fn test_rle_mixed_runs() {
        let data = b"aaabbbccc";
        let c = rle_compress(data);
        // (3,a)(3,b)(3,c) = 6 bytes
        assert_eq!(c.len(), 6);
        assert_eq!(rle_decompress(&c).unwrap().as_slice(), data.as_ref());
    }

    // ── LZ4-style unit tests ──────────────────────────────────────────────────

    #[test]
    fn test_lz4_empty_input() {
        let c = lz4_style_compress(&[]);
        assert!(c.is_empty());
        let d = lz4_style_decompress(&c).unwrap();
        assert!(d.is_empty());
    }

    #[test]
    fn test_lz4_single_byte() {
        let data = [0xFFu8];
        let c = lz4_style_compress(&data);
        let d = lz4_style_decompress(&c).unwrap();
        assert_eq!(d, data);
    }

    #[test]
    fn test_lz4_round_trip_short() {
        let data = b"hello world";
        let c = lz4_style_compress(data);
        let d = lz4_style_decompress(&c).unwrap();
        assert_eq!(d.as_slice(), data.as_ref());
    }

    #[test]
    fn test_lz4_round_trip_repeated() {
        let data = repeated_data(0xAA, 512);
        let c = lz4_style_compress(&data);
        let d = lz4_style_decompress(&c).unwrap();
        assert_eq!(d, data);
    }

    #[test]
    fn test_lz4_compresses_repetitive_data() {
        let data = repeated_data(0xBB, 1024);
        let c = lz4_style_compress(&data);
        // Must be smaller than original
        assert!(c.len() < data.len());
    }

    #[test]
    fn test_lz4_round_trip_alternating() {
        let data = alternating_data(300);
        let c = lz4_style_compress(&data);
        let d = lz4_style_decompress(&c).unwrap();
        assert_eq!(d, data);
    }

    #[test]
    fn test_lz4_round_trip_binary() {
        let data: Vec<u8> = (0..=255u8).cycle().take(800).collect();
        let c = lz4_style_compress(&data);
        let d = lz4_style_decompress(&c).unwrap();
        assert_eq!(d, data);
    }

    // ── detect_hint ──────────────────────────────────────────────────────────

    #[test]
    fn test_detect_hint_empty() {
        assert_eq!(
            StorageCompressionPipeline::detect_hint(&[]),
            CompressionHint::Unknown
        );
    }

    #[test]
    fn test_detect_hint_text() {
        assert_eq!(
            StorageCompressionPipeline::detect_hint(b"hello world"),
            CompressionHint::Text
        );
    }

    #[test]
    fn test_detect_hint_structured_object() {
        assert_eq!(
            StorageCompressionPipeline::detect_hint(b"{\"key\":\"value\"}"),
            CompressionHint::Structured
        );
    }

    #[test]
    fn test_detect_hint_structured_array() {
        assert_eq!(
            StorageCompressionPipeline::detect_hint(b"[1,2,3]"),
            CompressionHint::Structured
        );
    }

    #[test]
    fn test_detect_hint_binary() {
        let data = vec![0xFF, 0xFE, 0xFD, 0x00, 0x01];
        assert_eq!(
            StorageCompressionPipeline::detect_hint(&data),
            CompressionHint::Binary
        );
    }

    // ── compress_with_algo round-trips ────────────────────────────────────────

    #[test]
    fn test_compress_with_algo_none() {
        let p = make_pipeline();
        let data = b"no compression";
        let c = p.compress_with_algo(data, &CompressionAlgo::None);
        assert_eq!(c.as_slice(), data.as_ref());
    }

    #[test]
    fn test_compress_with_algo_rle() {
        let p = make_pipeline();
        let data = repeated_data(0x55, 200);
        let c = p.compress_with_algo(&data, &CompressionAlgo::Rle);
        let d = p.decompress(&c, &CompressionAlgo::Rle).unwrap();
        assert_eq!(d, data);
    }

    #[test]
    fn test_compress_with_algo_lz4() {
        let p = make_pipeline();
        let data = alternating_data(400);
        let c = p.compress_with_algo(&data, &CompressionAlgo::Lz4);
        let d = p.decompress(&c, &CompressionAlgo::Lz4).unwrap();
        assert_eq!(d, data);
    }

    #[test]
    fn test_compress_with_algo_snappy() {
        let p = make_pipeline();
        let data = repeated_data(0x11, 512);
        let c = p.compress_with_algo(&data, &CompressionAlgo::Snappy);
        let d = p.decompress(&c, &CompressionAlgo::Snappy).unwrap();
        assert_eq!(d, data);
    }

    #[test]
    fn test_compress_with_algo_zstd_low() {
        // level <= 3 → dispatches to RLE
        let p = make_pipeline();
        let data = repeated_data(0x22, 300);
        let c = p.compress_with_algo(&data, &CompressionAlgo::Zstd { level: 2 });
        let d = p
            .decompress(&c, &CompressionAlgo::Zstd { level: 2 })
            .unwrap();
        assert_eq!(d, data);
    }

    #[test]
    fn test_compress_with_algo_zstd_high() {
        // level >= 4 → dispatches to LZ4-style
        let p = make_pipeline();
        let data = alternating_data(600);
        let c = p.compress_with_algo(&data, &CompressionAlgo::Zstd { level: 5 });
        let d = p
            .decompress(&c, &CompressionAlgo::Zstd { level: 5 })
            .unwrap();
        assert_eq!(d, data);
    }

    // ── Pipeline compress / auto_compress ─────────────────────────────────────

    #[test]
    fn test_compress_small_data_uses_fallback() {
        // Data smaller than the smallest stage threshold (64) → fallback (None)
        let mut p = make_pipeline();
        let data = b"small";
        let result = p.compress(data, CompressionHint::Binary);
        assert_eq!(result.algo, CompressionAlgo::None);
        assert_eq!(result.original_size, data.len());
    }

    #[test]
    fn test_compress_medium_data_rle() {
        // 100 identical bytes should pass the RLE stage (min=64, expected ratio < 0.9)
        let mut p = make_pipeline();
        let data = repeated_data(0xAB, 100);
        let result = p.compress(&data, CompressionHint::Binary);
        // RLE on uniform data produces 2 bytes → ratio ≈ 0.02 << 0.9
        assert_eq!(result.algo, CompressionAlgo::Rle);
    }

    #[test]
    fn test_compress_round_trip_pipeline() {
        let mut p = make_pipeline();
        let data = repeated_data(0xCD, 200);
        let result = p.compress(&data, CompressionHint::Binary);
        let recovered = p.decompress(&result.data, &result.algo).unwrap();
        assert_eq!(recovered, data);
    }

    #[test]
    fn test_auto_compress_json() {
        let mut p = make_pipeline();
        let json =
            br#"{"key":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}"#;
        let result = p.auto_compress(json);
        assert_eq!(result.algo, CompressionAlgo::Rle); // JSON starts with '{' → Structured hint, but algo selected by pipeline
        let recovered = p.decompress(&result.data, &result.algo).unwrap();
        assert_eq!(recovered.as_slice(), json.as_ref());
    }

    #[test]
    fn test_auto_compress_round_trip() {
        let mut p = make_pipeline();
        let data = repeated_data(0xEE, 600);
        let result = p.auto_compress(&data);
        let recovered = p.decompress(&result.data, &result.algo).unwrap();
        assert_eq!(recovered, data);
    }

    #[test]
    fn test_compress_ratio_field() {
        let data = repeated_data(0x99, 200);
        let compressed = rle_compress(&data);
        let r = CompressionResult::new(CompressionAlgo::Rle, data.len(), compressed.clone());
        let expected = compressed.len() as f64 / data.len() as f64;
        assert!((r.ratio - expected).abs() < 1e-12);
    }

    #[test]
    fn test_compress_ratio_zero_len() {
        let r = CompressionResult::new(CompressionAlgo::None, 0, vec![]);
        assert_eq!(r.ratio, 1.0);
    }

    // ── Pipeline statistics ───────────────────────────────────────────────────

    #[test]
    fn test_stats_initial() {
        let p = make_pipeline();
        let stats = p.pipeline_stats();
        assert_eq!(stats.total_compressed, 0);
        assert_eq!(stats.total_bytes_in, 0);
        assert_eq!(stats.total_bytes_out, 0);
        assert_eq!(stats.avg_ratio, 1.0);
        assert!(stats.algo_usage.is_empty());
    }

    #[test]
    fn test_stats_after_compress() {
        let mut p = make_pipeline();
        let data = repeated_data(0x77, 200);
        p.compress(&data, CompressionHint::Binary);
        let stats = p.pipeline_stats();
        assert_eq!(stats.total_compressed, 1);
        assert_eq!(stats.total_bytes_in, 200);
        assert!(stats.total_bytes_out > 0);
        assert!(stats.avg_ratio > 0.0);
    }

    #[test]
    fn test_stats_algo_counter() {
        let mut p = make_pipeline();
        // Compress twice with RLE-friendly data
        let data = repeated_data(0x33, 200);
        p.compress(&data, CompressionHint::Binary);
        p.compress(&data, CompressionHint::Binary);
        let stats = p.pipeline_stats();
        // Both should use RLE
        let rle_count = stats.algo_usage.get("rle").copied().unwrap_or(0);
        assert_eq!(rle_count, 2);
    }

    #[test]
    fn test_stats_multiple_algos() {
        let mut p = make_pipeline();
        // Small → fallback (None)
        p.compress(b"small", CompressionHint::Binary);
        // Large uniform → RLE
        p.compress(&repeated_data(0x44, 200), CompressionHint::Binary);
        let stats = p.pipeline_stats();
        assert_eq!(stats.total_compressed, 2);
        assert!(stats.algo_usage.contains_key("none"));
        assert!(stats.algo_usage.contains_key("rle"));
    }

    // ── PipelineConfig / PipelineStage ────────────────────────────────────────

    #[test]
    fn test_custom_config_stages() {
        let config = PipelineConfig {
            stages: vec![PipelineStage {
                algo: CompressionAlgo::Lz4,
                min_input_size: 10,
                max_ratio: 0.99,
            }],
            fallback_algo: CompressionAlgo::None,
            target_ratio: 0.8,
            enable_ratio_check: true,
        };
        let mut p = StorageCompressionPipeline::new(config);
        let data = repeated_data(0x55, 512);
        let result = p.compress(&data, CompressionHint::Binary);
        let recovered = p.decompress(&result.data, &result.algo).unwrap();
        assert_eq!(recovered, data);
    }

    #[test]
    fn test_ratio_check_disabled() {
        // With ratio check disabled, every stage that meets size requirement is accepted
        let config = PipelineConfig {
            stages: vec![PipelineStage {
                algo: CompressionAlgo::Rle,
                min_input_size: 4,
                max_ratio: 0.01, // Extremely tight — would be rejected if check is on
            }],
            fallback_algo: CompressionAlgo::None,
            target_ratio: 1.0,
            enable_ratio_check: false,
        };
        let mut p = StorageCompressionPipeline::new(config);
        let data = b"abcd"; // Not compressible by RLE (4 distinct bytes)
        let result = p.compress(data, CompressionHint::Binary);
        // With ratio check disabled and size met, RLE should be used
        assert_eq!(result.algo, CompressionAlgo::Rle);
    }

    #[test]
    fn test_fallback_when_ratio_too_high() {
        // Only stage has a very tight ratio; random-ish data won't pass → fallback
        let config = PipelineConfig {
            stages: vec![PipelineStage {
                algo: CompressionAlgo::Rle,
                min_input_size: 1,
                max_ratio: 0.001, // Nearly impossible to achieve
            }],
            fallback_algo: CompressionAlgo::None,
            target_ratio: 1.0,
            enable_ratio_check: true,
        };
        let mut p = StorageCompressionPipeline::new(config);
        let data: Vec<u8> = (0u8..=127).collect();
        let result = p.compress(&data, CompressionHint::Binary);
        assert_eq!(result.algo, CompressionAlgo::None);
    }

    // ── CompressionAlgo helpers ───────────────────────────────────────────────

    #[test]
    fn test_algo_name_none() {
        assert_eq!(CompressionAlgo::None.name(), "none");
    }

    #[test]
    fn test_algo_name_lz4() {
        assert_eq!(CompressionAlgo::Lz4.name(), "lz4");
    }

    #[test]
    fn test_algo_name_zstd() {
        assert_eq!(CompressionAlgo::Zstd { level: 7 }.name(), "zstd(7)");
    }

    #[test]
    fn test_algo_name_snappy() {
        assert_eq!(CompressionAlgo::Snappy.name(), "snappy");
    }

    #[test]
    fn test_algo_name_rle() {
        assert_eq!(CompressionAlgo::Rle.name(), "rle");
    }

    // ── PipelineStats struct ──────────────────────────────────────────────────

    #[test]
    fn test_pipeline_stats_avg_ratio() {
        let mut p = make_pipeline();
        let data = repeated_data(0xBB, 1000);
        p.compress(&data, CompressionHint::Binary);
        let stats = p.pipeline_stats();
        let expected = stats.total_bytes_out as f64 / stats.total_bytes_in as f64;
        assert!((stats.avg_ratio - expected).abs() < 1e-12);
    }

    #[test]
    fn test_pipeline_stats_type() {
        let stats = PipelineStats {
            total_compressed: 5,
            total_bytes_in: 1000,
            total_bytes_out: 500,
            avg_ratio: 0.5,
            algo_usage: HashMap::new(),
        };
        assert_eq!(stats.total_compressed, 5);
        assert_eq!(stats.avg_ratio, 0.5);
    }

    // ── Large data round-trips ────────────────────────────────────────────────

    #[test]
    fn test_large_uniform_rle_round_trip() {
        let mut p = make_pipeline();
        let data = repeated_data(0xDE, 8192);
        let result = p.compress(&data, CompressionHint::Binary);
        let recovered = p.decompress(&result.data, &result.algo).unwrap();
        assert_eq!(recovered, data);
    }

    #[test]
    fn test_large_alternating_lz4_round_trip() {
        let p = make_pipeline();
        let data: Vec<u8> = (0u8..=255).cycle().take(4096).collect();
        let c = p.compress_with_algo(&data, &CompressionAlgo::Lz4);
        let d = p.decompress(&c, &CompressionAlgo::Lz4).unwrap();
        assert_eq!(d, data);
    }

    #[test]
    fn test_pipeline_three_calls_stats() {
        let mut p = make_pipeline();
        for _ in 0..3 {
            let data = repeated_data(0xAA, 500);
            p.compress(&data, CompressionHint::Binary);
        }
        let stats = p.pipeline_stats();
        assert_eq!(stats.total_compressed, 3);
        assert_eq!(stats.total_bytes_in, 1500);
    }
}
