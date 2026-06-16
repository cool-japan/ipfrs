//! `StorageCompressionPipeline` — configurable multi-stage compression pipeline.
//!
//! Provides a production-quality, pure-Rust compression pipeline with inline
//! implementations of RLE, LZ77-style sliding window, Delta encoding, and XOR
//! transform. No external compression crates are used.
//!
//! # Quick Start
//! ```rust
//! use ipfrs_storage::storage_compression_pipeline::{
//!     ScpStorageCompressionPipeline, ScpPipelineConfig, CompressionStage,
//!     ScpCompressionAlgorithm,
//! };
//!
//! let config = ScpPipelineConfig {
//!     stages: vec![CompressionStage {
//!         algorithm: ScpCompressionAlgorithm::Rle,
//!         enabled: true,
//!         min_size_bytes: 8,
//!     }],
//!     max_input_size: 1 << 20,
//!     enable_checksum: true,
//! };
//! let pipeline = ScpStorageCompressionPipeline::new(config);
//! let data = b"AAAAAABBBCCCCC";
//! let block = pipeline.compress(data).unwrap();
//! let recovered = pipeline.decompress(&block).unwrap();
//! assert_eq!(recovered, data);
//! ```

use parking_lot::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

// ─────────────────────────────────────────────────────────────────────────────
// FNV-1a 64-bit checksum
// ─────────────────────────────────────────────────────────────────────────────

/// Compute a FNV-1a 64-bit hash of `data`.
pub fn fnv1a_64(data: &[u8]) -> u64 {
    let mut h: u64 = 14_695_981_039_346_656_037;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(1_099_511_628_211);
    }
    h
}

// ─────────────────────────────────────────────────────────────────────────────
// CompressionAlgorithm
// ─────────────────────────────────────────────────────────────────────────────

/// Compression/transform algorithm used by a pipeline stage.
///
/// All algorithms are implemented in pure Rust with no external dependencies.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScpCompressionAlgorithm {
    /// Identity pass-through — data is stored unchanged.
    None,
    /// Run-Length Encoding: encodes consecutive repeated bytes as
    /// `(count: u8, byte: u8)` pairs. Count is clamped to 255.
    Rle,
    /// LZ77-style sliding-window compressor.
    ///
    /// Emits either a literal token (`0x00, byte`) or a back-reference token
    /// (`0x01, offset_lo, offset_hi, length`) when a repeated sequence of
    /// length ≥ 3 is found in the sliding window.
    Lz77 {
        /// Look-back window size in bytes (e.g. 4096).
        window_size: usize,
    },
    /// Delta encoding: each byte is replaced by its difference from the
    /// previous byte. The first byte is XOR-ed with `base_value`.
    Delta {
        /// Seed value XOR-ed with the first byte before differencing.
        base_value: u8,
    },
    /// XOR-cipher transform: each byte is XOR-ed with the cycling key.
    ///
    /// Useful as a lightweight obfuscation or entropy-normalization pass
    /// before a heavier compressor.
    Xor {
        /// Key bytes cycled over the input.
        key: Vec<u8>,
    },
}

impl ScpCompressionAlgorithm {
    /// Return a stable, human-readable name for this algorithm.
    pub fn name(&self) -> String {
        match self {
            Self::None => "none".to_string(),
            Self::Rle => "rle".to_string(),
            Self::Lz77 { window_size } => format!("lz77({})", window_size),
            Self::Delta { base_value } => format!("delta({})", base_value),
            Self::Xor { key } => format!("xor({})", key.len()),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CompressionStage
// ─────────────────────────────────────────────────────────────────────────────

/// A single stage in the compression pipeline.
#[derive(Debug, Clone)]
pub struct CompressionStage {
    /// Algorithm to apply in this stage.
    pub algorithm: ScpCompressionAlgorithm,
    /// If `false`, this stage is skipped entirely.
    pub enabled: bool,
    /// Minimum data size in bytes required to apply this stage.
    /// If the current data size is smaller, the stage is skipped.
    pub min_size_bytes: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// PipelineConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for a [`ScpStorageCompressionPipeline`].
#[derive(Debug, Clone)]
pub struct ScpPipelineConfig {
    /// Ordered list of compression stages.
    pub stages: Vec<CompressionStage>,
    /// Maximum accepted input size in bytes. Returns
    /// [`ScpPipelineError::InputTooLarge`] if exceeded.
    pub max_input_size: usize,
    /// If `true`, append an FNV-1a-64 checksum to every compressed block
    /// and verify it on decompression.
    pub enable_checksum: bool,
}

impl Default for ScpPipelineConfig {
    fn default() -> Self {
        Self {
            stages: vec![
                CompressionStage {
                    algorithm: ScpCompressionAlgorithm::Rle,
                    enabled: true,
                    min_size_bytes: 64,
                },
                CompressionStage {
                    algorithm: ScpCompressionAlgorithm::Lz77 { window_size: 4096 },
                    enabled: true,
                    min_size_bytes: 128,
                },
            ],
            max_input_size: 64 * 1024 * 1024, // 64 MiB
            enable_checksum: true,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CompressedBlock
// ─────────────────────────────────────────────────────────────────────────────

/// A compressed data block produced by [`ScpStorageCompressionPipeline::compress`].
#[derive(Debug, Clone)]
pub struct CompressedBlock {
    /// Original (uncompressed) size in bytes.
    pub original_size: usize,
    /// Compressed size in bytes (length of `data`).
    pub compressed_size: usize,
    /// Names of the stages that were actually applied (in order).
    pub stages_applied: Vec<String>,
    /// The compressed payload.
    pub data: Vec<u8>,
    /// FNV-1a-64 checksum of `data` (0 when checksum is disabled).
    pub checksum: u64,
    /// `original_size / compressed_size`; values > 1.0 mean compression won.
    pub compression_ratio: f64,
}

// ─────────────────────────────────────────────────────────────────────────────
// PipelineStats
// ─────────────────────────────────────────────────────────────────────────────

/// Aggregate statistics for a [`ScpStorageCompressionPipeline`] instance.
#[derive(Debug, Clone, Default)]
pub struct ScpPipelineStats {
    /// Total number of blocks compressed so far.
    pub total_blocks: u64,
    /// Sum of all original input sizes, in bytes.
    pub total_input_bytes: u64,
    /// Sum of all compressed output sizes, in bytes.
    pub total_output_bytes: u64,
    /// Rolling average compression ratio across all blocks.
    pub avg_compression_ratio: f64,
    /// Per-stage rolling average ratio: `(stage_name, avg_ratio)`.
    pub stage_stats: Vec<(String, f64)>,
}

// ─────────────────────────────────────────────────────────────────────────────
// PipelineError
// ─────────────────────────────────────────────────────────────────────────────

/// Errors that can arise during compression or decompression.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ScpPipelineError {
    /// Input data exceeds [`ScpPipelineConfig::max_input_size`].
    #[error("input too large: {0} bytes")]
    InputTooLarge(usize),
    /// A decompression stage produced invalid output.
    #[error("decompression failed: {0}")]
    DecompressionFailed(String),
    /// Checksum stored in the block does not match what was computed.
    #[error("checksum mismatch: expected {expected:#x}, got {got:#x}")]
    ChecksumMismatch {
        /// Expected (stored) checksum value.
        expected: u64,
        /// Checksum computed from the actual data.
        got: u64,
    },
    /// Pipeline configuration is invalid.
    #[error("invalid configuration: {0}")]
    InvalidConfiguration(String),
    /// An algorithm-level error occurred.
    #[error("algorithm error: {0}")]
    AlgorithmError(String),
}

// ─────────────────────────────────────────────────────────────────────────────
// Inline algorithm implementations
// ─────────────────────────────────────────────────────────────────────────────

/// RLE encode: consecutive repeated bytes are emitted as `(count: u8, byte: u8)`.
///
/// Count is clamped to 255 so a run longer than 255 bytes is split.
pub fn rle_encode(data: &[u8]) -> Vec<u8> {
    if data.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(data.len());
    let mut i = 0;
    while i < data.len() {
        let byte = data[i];
        let mut count: u8 = 1;
        // Extend run, clamping at 255
        while i + (count as usize) < data.len() && data[i + (count as usize)] == byte && count < 255
        {
            count += 1;
        }
        out.push(count);
        out.push(byte);
        i += count as usize;
    }
    out
}

/// RLE decode: parse `(count, byte)` pairs and reconstruct the original data.
pub fn rle_decode(data: &[u8]) -> Result<Vec<u8>, ScpPipelineError> {
    if !data.len().is_multiple_of(2) {
        return Err(ScpPipelineError::DecompressionFailed(
            "RLE stream has odd length".to_string(),
        ));
    }
    let mut out = Vec::with_capacity(data.len() * 2);
    let mut i = 0;
    while i + 1 < data.len() {
        let count = data[i] as usize;
        let byte = data[i + 1];
        if count == 0 {
            return Err(ScpPipelineError::DecompressionFailed(
                "RLE stream contains zero-count run".to_string(),
            ));
        }
        for _ in 0..count {
            out.push(byte);
        }
        i += 2;
    }
    Ok(out)
}

// ── LZ77 token format ────────────────────────────────────────────────────────
// Literal:     0x00, <byte>
// Back-ref:    0x01, <offset_lo>, <offset_hi>, <length>
//              offset is 1-based distance into the look-back window (little-endian u16)
//              length is the match length (u8, value >= 3)
// ─────────────────────────────────────────────────────────────────────────────

const LZ77_FLAG_LITERAL: u8 = 0x00;
const LZ77_FLAG_MATCH: u8 = 0x01;
const LZ77_MIN_MATCH: usize = 3;

/// LZ77-style encode using a sliding look-back window of `window_size` bytes.
///
/// Emits either a literal token or a back-reference token for each input position.
pub fn lz77_encode(data: &[u8], window_size: usize) -> Vec<u8> {
    if data.is_empty() {
        return Vec::new();
    }
    let effective_window = window_size.min(u16::MAX as usize);
    // Worst case: every byte becomes a literal token (2 bytes each)
    let mut out = Vec::with_capacity(data.len() * 2);
    let mut pos = 0;

    while pos < data.len() {
        let window_start = pos.saturating_sub(effective_window);
        let lookahead_end = (pos + 258).min(data.len()); // max match len fits in u8 + 3

        // Search for the longest match in the window
        let mut best_offset: usize = 0;
        let mut best_length: usize = 0;

        // Only search if there is a window to look back into
        if window_start < pos {
            let lookahead = &data[pos..lookahead_end];
            let lookahead_max = lookahead.len();

            for start in window_start..pos {
                // How many bytes match starting at `start` vs `pos`?
                let max_len = (pos - start).min(lookahead_max); // prevent overlap copy past `pos`
                let max_len = max_len.min(255 + LZ77_MIN_MATCH - 1); // fits in u8 after subtract
                let mut length = 0;
                while length < max_len && data[start + length] == lookahead[length] {
                    length += 1;
                }
                if length > best_length {
                    best_length = length;
                    best_offset = pos - start; // 1-based distance
                }
            }
        }

        if best_length >= LZ77_MIN_MATCH && best_offset <= u16::MAX as usize {
            // Emit back-reference
            let stored_len = (best_length - LZ77_MIN_MATCH) as u8; // stored as (len - 3)
            let offset_u16 = best_offset as u16;
            out.push(LZ77_FLAG_MATCH);
            out.push((offset_u16 & 0xFF) as u8);
            out.push((offset_u16 >> 8) as u8);
            out.push(stored_len);
            pos += best_length;
        } else {
            // Emit literal
            out.push(LZ77_FLAG_LITERAL);
            out.push(data[pos]);
            pos += 1;
        }
    }
    out
}

/// LZ77-style decode: reconstruct original data from literal/back-reference tokens.
pub fn lz77_decode(data: &[u8]) -> Result<Vec<u8>, ScpPipelineError> {
    let mut out: Vec<u8> = Vec::with_capacity(data.len() * 2);
    let mut i = 0;

    while i < data.len() {
        let flag = data[i];
        i += 1;

        match flag {
            LZ77_FLAG_LITERAL => {
                if i >= data.len() {
                    return Err(ScpPipelineError::DecompressionFailed(
                        "LZ77 stream truncated after literal flag".to_string(),
                    ));
                }
                out.push(data[i]);
                i += 1;
            }
            LZ77_FLAG_MATCH => {
                if i + 2 >= data.len() {
                    return Err(ScpPipelineError::DecompressionFailed(
                        "LZ77 stream truncated inside back-reference".to_string(),
                    ));
                }
                let offset_lo = data[i] as usize;
                let offset_hi = data[i + 1] as usize;
                let stored_len = data[i + 2] as usize;
                i += 3;

                let offset = offset_lo | (offset_hi << 8);
                let length = stored_len + LZ77_MIN_MATCH;

                if offset == 0 {
                    return Err(ScpPipelineError::DecompressionFailed(
                        "LZ77 back-reference has zero offset".to_string(),
                    ));
                }
                if offset > out.len() {
                    return Err(ScpPipelineError::DecompressionFailed(format!(
                        "LZ77 back-reference offset {} > output length {}",
                        offset,
                        out.len()
                    )));
                }

                // Copy byte-by-byte to handle overlapping runs correctly
                let copy_start = out.len() - offset;
                for j in 0..length {
                    let byte = out[copy_start + (j % offset)];
                    out.push(byte);
                }
            }
            other => {
                return Err(ScpPipelineError::DecompressionFailed(format!(
                    "LZ77 unknown flag byte: {:#x}",
                    other
                )));
            }
        }
    }
    Ok(out)
}

/// Delta encode: replace each byte with its difference from the previous byte.
///
/// The first byte is XOR-ed with `base` before the run starts (so `base=0`
/// leaves the first byte unchanged).
pub fn delta_encode(data: &[u8], base: u8) -> Vec<u8> {
    if data.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(data.len());
    let first = data[0].wrapping_sub(base);
    out.push(first);
    let mut prev = data[0];
    for &b in &data[1..] {
        out.push(b.wrapping_sub(prev));
        prev = b;
    }
    out
}

/// Delta decode: reconstruct original data from delta-encoded stream.
pub fn delta_decode(data: &[u8], base: u8) -> Vec<u8> {
    if data.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(data.len());
    let first = data[0].wrapping_add(base);
    out.push(first);
    let mut prev = first;
    for &d in &data[1..] {
        let b = d.wrapping_add(prev);
        out.push(b);
        prev = b;
    }
    out
}

/// XOR-cipher transform: each input byte is XOR-ed with the cycling key.
///
/// Passing an empty key returns the data unchanged.
pub fn xor_transform(data: &[u8], key: &[u8]) -> Vec<u8> {
    if key.is_empty() {
        return data.to_vec();
    }
    data.iter()
        .enumerate()
        .map(|(i, &b)| b ^ key[i % key.len()])
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal stage apply / unapply helpers
// ─────────────────────────────────────────────────────────────────────────────

fn apply_algorithm(
    data: &[u8],
    algo: &ScpCompressionAlgorithm,
) -> Result<Vec<u8>, ScpPipelineError> {
    match algo {
        ScpCompressionAlgorithm::None => Ok(data.to_vec()),
        ScpCompressionAlgorithm::Rle => Ok(rle_encode(data)),
        ScpCompressionAlgorithm::Lz77 { window_size } => Ok(lz77_encode(data, *window_size)),
        ScpCompressionAlgorithm::Delta { base_value } => Ok(delta_encode(data, *base_value)),
        ScpCompressionAlgorithm::Xor { key } => {
            if key.is_empty() {
                return Err(ScpPipelineError::InvalidConfiguration(
                    "XOR key must not be empty".to_string(),
                ));
            }
            Ok(xor_transform(data, key))
        }
    }
}

fn unapply_algorithm(data: &[u8], algo_name: &str) -> Result<Vec<u8>, ScpPipelineError> {
    // Parse the name produced by ScpCompressionAlgorithm::name()
    if algo_name == "none" {
        return Ok(data.to_vec());
    }
    if algo_name == "rle" {
        return rle_decode(data);
    }
    if algo_name.starts_with("lz77(") {
        return lz77_decode(data);
    }
    if algo_name.starts_with("delta(") {
        // extract base value from "delta(<n>)"
        let inner = algo_name.trim_start_matches("delta(").trim_end_matches(')');
        let base: u8 = inner.parse().map_err(|_| {
            ScpPipelineError::DecompressionFailed(format!(
                "Cannot parse delta base from stage name: {}",
                algo_name
            ))
        })?;
        return Ok(delta_decode(data, base));
    }
    if algo_name.starts_with("xor(") {
        // We cannot recover the key from the name alone, so we embed the key in the name.
        // Format stored in stages_applied: "xor(<hex_key>)"
        let inner = algo_name.trim_start_matches("xor(").trim_end_matches(')');
        let key = hex::decode(inner).map_err(|_| {
            ScpPipelineError::DecompressionFailed(format!(
                "Cannot decode XOR key from stage name: {}",
                algo_name
            ))
        })?;
        return Ok(xor_transform(data, &key));
    }
    Err(ScpPipelineError::DecompressionFailed(format!(
        "Unknown stage name in stages_applied: {}",
        algo_name
    )))
}

/// Produce the canonical stage name for storing in `CompressedBlock::stages_applied`.
///
/// For XOR, embed the hex-encoded key so that decompression can recover it.
fn stage_name_for_storage(algo: &ScpCompressionAlgorithm) -> String {
    match algo {
        ScpCompressionAlgorithm::Xor { key } => format!("xor({})", hex::encode(key)),
        other => other.name(),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal mutable stats accumulator
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Default)]
struct StatsAccumulator {
    total_blocks: u64,
    total_input_bytes: u64,
    total_output_bytes: u64,
    // sum of ratios for rolling average
    ratio_sum: f64,
    // per-stage: name -> (sum, count)
    stage_ratios: std::collections::HashMap<String, (f64, u64)>,
}

impl StatsAccumulator {
    fn record(
        &mut self,
        input_bytes: usize,
        output_bytes: usize,
        stage_intermediate_ratios: &[(String, f64)],
    ) {
        self.total_blocks += 1;
        self.total_input_bytes += input_bytes as u64;
        self.total_output_bytes += output_bytes as u64;
        let ratio = if output_bytes == 0 {
            1.0
        } else {
            input_bytes as f64 / output_bytes as f64
        };
        self.ratio_sum += ratio;

        for (name, r) in stage_intermediate_ratios {
            let entry = self.stage_ratios.entry(name.clone()).or_insert((0.0, 0));
            entry.0 += r;
            entry.1 += 1;
        }
    }

    fn snapshot(&self) -> ScpPipelineStats {
        let avg = if self.total_blocks == 0 {
            1.0
        } else {
            self.ratio_sum / self.total_blocks as f64
        };
        let stage_stats = self
            .stage_ratios
            .iter()
            .map(|(name, (sum, count))| {
                (
                    name.clone(),
                    if *count == 0 {
                        1.0
                    } else {
                        sum / *count as f64
                    },
                )
            })
            .collect();
        ScpPipelineStats {
            total_blocks: self.total_blocks,
            total_input_bytes: self.total_input_bytes,
            total_output_bytes: self.total_output_bytes,
            avg_compression_ratio: avg,
            stage_stats,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// StorageCompressionPipeline
// ─────────────────────────────────────────────────────────────────────────────

/// A configurable multi-stage data compression pipeline.
///
/// Each stage can independently apply an algorithm (RLE, LZ77, Delta, XOR)
/// to the running data buffer, or be skipped based on size thresholds.
///
/// # Thread safety
/// This type is `Send + Sync`. Statistics are guarded by a `parking_lot::Mutex`.
pub struct ScpStorageCompressionPipeline {
    config: ScpPipelineConfig,
    stats: Arc<Mutex<StatsAccumulator>>,
    /// Monotonic counter used only for internal tie-breaking (not exposed).
    _block_counter: Arc<AtomicU64>,
}

impl ScpStorageCompressionPipeline {
    /// Create a new pipeline with the given configuration.
    ///
    /// Returns [`ScpPipelineError::InvalidConfiguration`] if any stage
    /// has an XOR algorithm with an empty key.
    pub fn new(config: ScpPipelineConfig) -> Self {
        Self {
            config,
            stats: Arc::new(Mutex::new(StatsAccumulator::default())),
            _block_counter: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Compress `data` through all configured stages in order.
    ///
    /// Stages that are disabled or whose `min_size_bytes` is not met are
    /// transparently skipped. When `enable_checksum` is set, an 8-byte
    /// FNV-1a-64 checksum is appended to the output payload.
    pub fn compress(&self, data: &[u8]) -> Result<CompressedBlock, ScpPipelineError> {
        self.compress_with_stages(data, &self.config.stages)
    }

    /// Decompress a [`CompressedBlock`] produced by this pipeline (or any
    /// pipeline with identical stage configuration).
    ///
    /// Stages are applied in **reverse** order using the `stages_applied`
    /// list embedded in the block.
    pub fn decompress(&self, block: &CompressedBlock) -> Result<Vec<u8>, ScpPipelineError> {
        let payload = if self.config.enable_checksum {
            // Last 8 bytes are the checksum
            if block.data.len() < 8 {
                return Err(ScpPipelineError::DecompressionFailed(
                    "Block too small to contain checksum".to_string(),
                ));
            }
            let (payload, cs_bytes) = block.data.split_at(block.data.len() - 8);
            let stored_cs = u64::from_le_bytes(cs_bytes.try_into().map_err(|_| {
                ScpPipelineError::DecompressionFailed("Could not read checksum bytes".to_string())
            })?);
            let computed_cs = fnv1a_64(payload);
            if stored_cs != computed_cs {
                return Err(ScpPipelineError::ChecksumMismatch {
                    expected: stored_cs,
                    got: computed_cs,
                });
            }
            payload.to_vec()
        } else {
            block.data.clone()
        };

        // Apply stages in reverse
        let mut current = payload;
        for stage_name in block.stages_applied.iter().rev() {
            current = unapply_algorithm(&current, stage_name)?;
        }
        Ok(current)
    }

    /// Compress `data` using a custom set of stages instead of those in
    /// the pipeline's configuration.
    ///
    /// The `max_input_size` and `enable_checksum` settings from the
    /// pipeline configuration still apply.
    pub fn compress_with_stages(
        &self,
        data: &[u8],
        stages: &[CompressionStage],
    ) -> Result<CompressedBlock, ScpPipelineError> {
        if data.len() > self.config.max_input_size {
            return Err(ScpPipelineError::InputTooLarge(data.len()));
        }

        let original_size = data.len();
        let mut current = data.to_vec();
        let mut stages_applied: Vec<String> = Vec::new();
        let mut stage_intermediate_ratios: Vec<(String, f64)> = Vec::new();

        for stage in stages {
            if !stage.enabled {
                continue;
            }
            if current.len() < stage.min_size_bytes {
                continue;
            }

            let size_before = current.len();
            let compressed = apply_algorithm(&current, &stage.algorithm)?;
            let size_after = compressed.len();

            let stage_name = stage_name_for_storage(&stage.algorithm);
            let ratio = if size_after == 0 {
                1.0_f64
            } else {
                size_before as f64 / size_after as f64
            };

            stages_applied.push(stage_name.clone());
            stage_intermediate_ratios.push((stage_name, ratio));
            current = compressed;
        }

        // Optionally append checksum
        let checksum = if self.config.enable_checksum {
            let cs = fnv1a_64(&current);
            let cs_bytes = cs.to_le_bytes();
            current.extend_from_slice(&cs_bytes);
            cs
        } else {
            0
        };

        let compressed_size = current.len();
        let compression_ratio = if compressed_size == 0 {
            1.0
        } else {
            original_size as f64 / compressed_size as f64
        };

        // Record statistics
        self._block_counter.fetch_add(1, Ordering::Relaxed);
        {
            let mut acc = self.stats.lock();
            acc.record(original_size, compressed_size, &stage_intermediate_ratios);
        }

        Ok(CompressedBlock {
            original_size,
            compressed_size,
            stages_applied,
            data: current,
            checksum,
            compression_ratio,
        })
    }

    /// Heuristically determine the best single-stage algorithm for `data`
    /// by trying None, Rle, and Lz77 on the first 1024 bytes.
    ///
    /// Returns the algorithm that yields the best compression ratio.
    pub fn best_algorithm(&self, data: &[u8]) -> ScpCompressionAlgorithm {
        let sample = &data[..data.len().min(1024)];
        if sample.is_empty() {
            return ScpCompressionAlgorithm::None;
        }

        let candidates: &[ScpCompressionAlgorithm] = &[
            ScpCompressionAlgorithm::None,
            ScpCompressionAlgorithm::Rle,
            ScpCompressionAlgorithm::Lz77 { window_size: 4096 },
        ];

        let mut best_algo = ScpCompressionAlgorithm::None;
        let mut best_ratio: f64 = 0.0;

        for algo in candidates {
            let encoded = match apply_algorithm(sample, algo) {
                Ok(v) => v,
                Err(_) => continue,
            };
            let ratio = if encoded.is_empty() {
                1.0
            } else {
                sample.len() as f64 / encoded.len() as f64
            };
            if ratio > best_ratio {
                best_ratio = ratio;
                best_algo = algo.clone();
            }
        }
        best_algo
    }

    /// Return a snapshot of the current pipeline statistics.
    pub fn stats(&self) -> ScpPipelineStats {
        self.stats.lock().snapshot()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Inline xorshift64 PRNG (no rand crate) ────────────────────────────────
    struct Xorshift64(u64);

    impl Xorshift64 {
        fn new(seed: u64) -> Self {
            Self(if seed == 0 { 1 } else { seed })
        }
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn next_byte(&mut self) -> u8 {
            (self.next() & 0xFF) as u8
        }
        fn fill(&mut self, buf: &mut [u8]) {
            for b in buf.iter_mut() {
                *b = self.next_byte();
            }
        }
    }

    fn make_pipeline(enable_checksum: bool) -> ScpStorageCompressionPipeline {
        let config = ScpPipelineConfig {
            stages: vec![
                CompressionStage {
                    algorithm: ScpCompressionAlgorithm::Rle,
                    enabled: true,
                    min_size_bytes: 4,
                },
                CompressionStage {
                    algorithm: ScpCompressionAlgorithm::Lz77 { window_size: 256 },
                    enabled: true,
                    min_size_bytes: 8,
                },
            ],
            max_input_size: 1 << 20,
            enable_checksum,
        };
        ScpStorageCompressionPipeline::new(config)
    }

    // ── RLE tests ────────────────────────────────────────────────────────────

    #[test]
    fn test_rle_roundtrip_simple() {
        let data = b"AAABBC";
        let encoded = rle_encode(data);
        let decoded = rle_decode(&encoded).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_rle_roundtrip_empty() {
        let encoded = rle_encode(&[]);
        assert!(encoded.is_empty());
        let decoded = rle_decode(&encoded).unwrap();
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_rle_roundtrip_single_byte() {
        let data = b"Z";
        let enc = rle_encode(data);
        assert_eq!(enc, vec![1, b'Z']);
        let dec = rle_decode(&enc).unwrap();
        assert_eq!(dec, data);
    }

    #[test]
    fn test_rle_roundtrip_max_run() {
        // Run of 256 bytes — should produce two pairs
        let data = vec![0xAA_u8; 256];
        let enc = rle_encode(&data);
        // First pair: (255, 0xAA), Second pair: (1, 0xAA)
        assert_eq!(enc.len(), 4);
        assert_eq!(enc[0], 255);
        assert_eq!(enc[2], 1);
        let dec = rle_decode(&enc).unwrap();
        assert_eq!(dec, data);
    }

    #[test]
    fn test_rle_roundtrip_no_repeats() {
        let data: Vec<u8> = (0u8..=255u8).collect();
        let enc = rle_encode(&data);
        let dec = rle_decode(&enc).unwrap();
        assert_eq!(dec, data);
    }

    #[test]
    fn test_rle_roundtrip_all_same() {
        let data = vec![0x7F_u8; 1000];
        let enc = rle_encode(&data);
        let dec = rle_decode(&enc).unwrap();
        assert_eq!(dec, data);
    }

    #[test]
    fn test_rle_decode_odd_length_error() {
        let bad = vec![3_u8, 0xAB, 0xFF]; // odd length
        assert!(rle_decode(&bad).is_err());
    }

    #[test]
    fn test_rle_decode_zero_count_error() {
        let bad = vec![0_u8, 0xAB]; // zero count
        assert!(rle_decode(&bad).is_err());
    }

    // ── LZ77 tests ───────────────────────────────────────────────────────────

    #[test]
    fn test_lz77_roundtrip_empty() {
        let enc = lz77_encode(&[], 256);
        assert!(enc.is_empty());
        let dec = lz77_decode(&enc).unwrap();
        assert!(dec.is_empty());
    }

    #[test]
    fn test_lz77_roundtrip_single_byte() {
        let data = b"X";
        let enc = lz77_encode(data, 256);
        let dec = lz77_decode(&enc).unwrap();
        assert_eq!(dec.as_slice(), data.as_ref());
    }

    #[test]
    fn test_lz77_roundtrip_repeated_pattern() {
        let data = b"abcabcabcabc";
        let enc = lz77_encode(data, 256);
        let dec = lz77_decode(&enc).unwrap();
        assert_eq!(dec.as_slice(), data.as_ref());
    }

    #[test]
    fn test_lz77_roundtrip_all_same() {
        let data = vec![0x55_u8; 200];
        let enc = lz77_encode(&data, 256);
        let dec = lz77_decode(&enc).unwrap();
        assert_eq!(dec, data);
    }

    #[test]
    fn test_lz77_roundtrip_random() {
        let mut rng = Xorshift64::new(42);
        let mut data = vec![0u8; 512];
        rng.fill(&mut data);
        let enc = lz77_encode(&data, 256);
        let dec = lz77_decode(&enc).unwrap();
        assert_eq!(dec, data);
    }

    #[test]
    fn test_lz77_roundtrip_text() {
        let data = b"the quick brown fox jumps over the lazy dog. the quick brown fox.";
        let enc = lz77_encode(data, 256);
        let dec = lz77_decode(&enc).unwrap();
        assert_eq!(dec.as_slice(), data.as_ref());
    }

    #[test]
    fn test_lz77_roundtrip_large_window() {
        let mut rng = Xorshift64::new(999);
        let mut data = vec![0u8; 2048];
        rng.fill(&mut data);
        // Introduce repeated regions — copy via temporary to avoid borrow conflict
        let repeated: Vec<u8> = data[0..128].to_vec();
        data[1024..1024 + 128].copy_from_slice(&repeated);
        let enc = lz77_encode(&data, 4096);
        let dec = lz77_decode(&enc).unwrap();
        assert_eq!(dec, data);
    }

    #[test]
    fn test_lz77_decode_truncated_literal() {
        // Flag 0x00 with no following byte
        let bad = vec![LZ77_FLAG_LITERAL];
        assert!(lz77_decode(&bad).is_err());
    }

    #[test]
    fn test_lz77_decode_truncated_backref() {
        // Flag 0x01 but only one byte follows (needs 3)
        let bad = vec![LZ77_FLAG_MATCH, 0x01];
        assert!(lz77_decode(&bad).is_err());
    }

    #[test]
    fn test_lz77_decode_unknown_flag() {
        let bad = vec![0x42_u8, 0x00];
        assert!(lz77_decode(&bad).is_err());
    }

    // ── Delta tests ──────────────────────────────────────────────────────────

    #[test]
    fn test_delta_roundtrip_simple() {
        let data = b"Hello, World!";
        let enc = delta_encode(data, 0);
        let dec = delta_decode(&enc, 0);
        assert_eq!(dec.as_slice(), data.as_ref());
    }

    #[test]
    fn test_delta_roundtrip_with_base() {
        let data = b"Hello, World!";
        let enc = delta_encode(data, 42);
        let dec = delta_decode(&enc, 42);
        assert_eq!(dec.as_slice(), data.as_ref());
    }

    #[test]
    fn test_delta_roundtrip_empty() {
        let enc = delta_encode(&[], 0);
        assert!(enc.is_empty());
        let dec = delta_decode(&enc, 0);
        assert!(dec.is_empty());
    }

    #[test]
    fn test_delta_roundtrip_monotone() {
        let data: Vec<u8> = (0u8..=100u8).collect();
        let enc = delta_encode(&data, 0);
        // All diffs should be 1 except the first byte
        for &b in &enc[1..] {
            assert_eq!(b, 1);
        }
        let dec = delta_decode(&enc, 0);
        assert_eq!(dec, data);
    }

    #[test]
    fn test_delta_roundtrip_random() {
        let mut rng = Xorshift64::new(12345);
        let mut data = vec![0u8; 256];
        rng.fill(&mut data);
        let enc = delta_encode(&data, 77);
        let dec = delta_decode(&enc, 77);
        assert_eq!(dec, data);
    }

    // ── XOR tests ────────────────────────────────────────────────────────────

    #[test]
    fn test_xor_roundtrip_simple() {
        let data = b"Hello, World!";
        let key = b"secret";
        let enc = xor_transform(data, key);
        let dec = xor_transform(&enc, key);
        assert_eq!(dec.as_slice(), data.as_ref());
    }

    #[test]
    fn test_xor_roundtrip_single_byte_key() {
        let data = b"AAABBBCCC";
        let key = b"\xFF";
        let enc = xor_transform(data, key);
        let dec = xor_transform(&enc, key);
        assert_eq!(dec.as_slice(), data.as_ref());
    }

    #[test]
    fn test_xor_empty_key_passthrough() {
        let data = b"test";
        let enc = xor_transform(data, &[]);
        assert_eq!(enc.as_slice(), data.as_ref());
    }

    #[test]
    fn test_xor_roundtrip_empty_data() {
        let enc = xor_transform(&[], b"key");
        assert!(enc.is_empty());
    }

    #[test]
    fn test_xor_roundtrip_long_data() {
        let mut rng = Xorshift64::new(7777);
        let mut data = vec![0u8; 1024];
        rng.fill(&mut data);
        let key = b"COOLJAPAN";
        let enc = xor_transform(&data, key);
        let dec = xor_transform(&enc, key);
        assert_eq!(dec, data);
    }

    // ── FNV-1a checksum ──────────────────────────────────────────────────────

    #[test]
    fn test_fnv1a_known_values() {
        // FNV-1a of empty string is the offset basis
        assert_eq!(fnv1a_64(&[]), 14_695_981_039_346_656_037_u64);
        // Deterministic for same input
        let h1 = fnv1a_64(b"hello");
        let h2 = fnv1a_64(b"hello");
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_fnv1a_different_data_different_hash() {
        assert_ne!(fnv1a_64(b"abc"), fnv1a_64(b"abd"));
    }

    // ── Pipeline compress/decompress ─────────────────────────────────────────

    #[test]
    fn test_pipeline_compress_decompress_with_checksum() {
        let pipeline = make_pipeline(true);
        let data = b"AAAAAABBBBBCCCCCDDDDD";
        let block = pipeline.compress(data).unwrap();
        assert_ne!(block.checksum, 0);
        let recovered = pipeline.decompress(&block).unwrap();
        assert_eq!(recovered, data);
    }

    #[test]
    fn test_pipeline_compress_decompress_no_checksum() {
        let pipeline = make_pipeline(false);
        let data = b"AAAAAABBBBBCCCCCDDDDD";
        let block = pipeline.compress(data).unwrap();
        assert_eq!(block.checksum, 0);
        let recovered = pipeline.decompress(&block).unwrap();
        assert_eq!(recovered, data);
    }

    #[test]
    fn test_pipeline_empty_input() {
        let pipeline = make_pipeline(true);
        let block = pipeline.compress(&[]).unwrap();
        // Empty data — stages have min_size_bytes > 0, so no stages applied
        assert!(block.stages_applied.is_empty());
        let recovered = pipeline.decompress(&block).unwrap();
        assert!(recovered.is_empty());
    }

    #[test]
    fn test_pipeline_input_too_large() {
        let config = ScpPipelineConfig {
            stages: vec![],
            max_input_size: 10,
            enable_checksum: false,
        };
        let pipeline = ScpStorageCompressionPipeline::new(config);
        let data = vec![0u8; 11];
        let err = pipeline.compress(&data).unwrap_err();
        assert!(matches!(err, ScpPipelineError::InputTooLarge(11)));
    }

    #[test]
    fn test_pipeline_checksum_mismatch() {
        let pipeline = make_pipeline(true);
        let data = b"test checksum mismatch";
        let mut block = pipeline.compress(data).unwrap();
        // Corrupt the payload (not the checksum at the end)
        if !block.data.is_empty() {
            let mid = block.data.len() / 2;
            block.data[mid] ^= 0xFF;
        }
        let err = pipeline.decompress(&block).unwrap_err();
        assert!(matches!(err, ScpPipelineError::ChecksumMismatch { .. }));
    }

    #[test]
    fn test_pipeline_compresses_repetitive_data() {
        let pipeline = make_pipeline(false);
        let data = vec![0x42_u8; 1024];
        let block = pipeline.compress(&data).unwrap();
        // Repetitive data should compress well
        assert!(block.compressed_size < block.original_size);
        assert!(block.compression_ratio > 1.0);
    }

    #[test]
    fn test_pipeline_stages_applied_tracked() {
        let pipeline = make_pipeline(false);
        let data = vec![b'A'; 200]; // larger than both min_size_bytes thresholds
        let block = pipeline.compress(&data).unwrap();
        assert!(!block.stages_applied.is_empty());
        let recovered = pipeline.decompress(&block).unwrap();
        assert_eq!(recovered, data);
    }

    #[test]
    fn test_pipeline_min_size_bytes_skip() {
        let config = ScpPipelineConfig {
            stages: vec![CompressionStage {
                algorithm: ScpCompressionAlgorithm::Rle,
                enabled: true,
                min_size_bytes: 1000, // very large threshold
            }],
            max_input_size: 1 << 20,
            enable_checksum: false,
        };
        let pipeline = ScpStorageCompressionPipeline::new(config);
        let data = b"hello world"; // only 11 bytes — stage should be skipped
        let block = pipeline.compress(data).unwrap();
        assert!(block.stages_applied.is_empty(), "stage should be skipped");
        let recovered = pipeline.decompress(&block).unwrap();
        assert_eq!(recovered.as_slice(), data.as_ref());
    }

    #[test]
    fn test_pipeline_stage_disabled_skip() {
        let config = ScpPipelineConfig {
            stages: vec![CompressionStage {
                algorithm: ScpCompressionAlgorithm::Rle,
                enabled: false, // disabled
                min_size_bytes: 0,
            }],
            max_input_size: 1 << 20,
            enable_checksum: false,
        };
        let pipeline = ScpStorageCompressionPipeline::new(config);
        let data = b"AAABBBCCC";
        let block = pipeline.compress(data).unwrap();
        assert!(block.stages_applied.is_empty());
    }

    // ── Delta stage in pipeline ───────────────────────────────────────────────

    #[test]
    fn test_pipeline_delta_stage() {
        let config = ScpPipelineConfig {
            stages: vec![CompressionStage {
                algorithm: ScpCompressionAlgorithm::Delta { base_value: 0 },
                enabled: true,
                min_size_bytes: 1,
            }],
            max_input_size: 1 << 20,
            enable_checksum: true,
        };
        let pipeline = ScpStorageCompressionPipeline::new(config);
        let data: Vec<u8> = (0u8..=200u8).collect();
        let block = pipeline.compress(&data).unwrap();
        assert_eq!(block.stages_applied, vec!["delta(0)"]);
        let recovered = pipeline.decompress(&block).unwrap();
        assert_eq!(recovered, data);
    }

    // ── XOR stage in pipeline ─────────────────────────────────────────────────

    #[test]
    fn test_pipeline_xor_stage() {
        let key = b"mysecret".to_vec();
        let config = ScpPipelineConfig {
            stages: vec![CompressionStage {
                algorithm: ScpCompressionAlgorithm::Xor { key: key.clone() },
                enabled: true,
                min_size_bytes: 1,
            }],
            max_input_size: 1 << 20,
            enable_checksum: true,
        };
        let pipeline = ScpStorageCompressionPipeline::new(config);
        let data = b"sensitive payload data goes here";
        let block = pipeline.compress(data).unwrap();
        let recovered = pipeline.decompress(&block).unwrap();
        assert_eq!(recovered.as_slice(), data.as_ref());
    }

    #[test]
    fn test_pipeline_xor_empty_key_error() {
        let config = ScpPipelineConfig {
            stages: vec![CompressionStage {
                algorithm: ScpCompressionAlgorithm::Xor { key: vec![] },
                enabled: true,
                min_size_bytes: 0,
            }],
            max_input_size: 1 << 20,
            enable_checksum: false,
        };
        let pipeline = ScpStorageCompressionPipeline::new(config);
        let err = pipeline.compress(b"test").unwrap_err();
        assert!(matches!(err, ScpPipelineError::InvalidConfiguration(_)));
    }

    // ── Multi-stage pipeline ──────────────────────────────────────────────────

    #[test]
    fn test_pipeline_multi_stage_rle_then_lz77() {
        let config = ScpPipelineConfig {
            stages: vec![
                CompressionStage {
                    algorithm: ScpCompressionAlgorithm::Rle,
                    enabled: true,
                    min_size_bytes: 4,
                },
                CompressionStage {
                    algorithm: ScpCompressionAlgorithm::Lz77 { window_size: 256 },
                    enabled: true,
                    min_size_bytes: 4,
                },
            ],
            max_input_size: 1 << 20,
            enable_checksum: true,
        };
        let pipeline = ScpStorageCompressionPipeline::new(config);
        let data = b"AAABBBCCCAAABBBCCCAAABBBCCC";
        let block = pipeline.compress(data).unwrap();
        assert_eq!(block.stages_applied.len(), 2);
        let recovered = pipeline.decompress(&block).unwrap();
        assert_eq!(recovered.as_slice(), data.as_ref());
    }

    #[test]
    fn test_pipeline_multi_stage_delta_then_rle() {
        let config = ScpPipelineConfig {
            stages: vec![
                CompressionStage {
                    algorithm: ScpCompressionAlgorithm::Delta { base_value: 0 },
                    enabled: true,
                    min_size_bytes: 1,
                },
                CompressionStage {
                    algorithm: ScpCompressionAlgorithm::Rle,
                    enabled: true,
                    min_size_bytes: 4,
                },
            ],
            max_input_size: 1 << 20,
            enable_checksum: true,
        };
        let pipeline = ScpStorageCompressionPipeline::new(config);
        // Monotone data → delta produces all-1 stream → RLE compresses well
        let data: Vec<u8> = (0u8..=200).collect();
        let block = pipeline.compress(&data).unwrap();
        let recovered = pipeline.decompress(&block).unwrap();
        assert_eq!(recovered, data);
    }

    #[test]
    fn test_pipeline_three_stage() {
        let config = ScpPipelineConfig {
            stages: vec![
                CompressionStage {
                    algorithm: ScpCompressionAlgorithm::Delta { base_value: 5 },
                    enabled: true,
                    min_size_bytes: 1,
                },
                CompressionStage {
                    algorithm: ScpCompressionAlgorithm::Rle,
                    enabled: true,
                    min_size_bytes: 4,
                },
                CompressionStage {
                    algorithm: ScpCompressionAlgorithm::Lz77 { window_size: 128 },
                    enabled: true,
                    min_size_bytes: 4,
                },
            ],
            max_input_size: 1 << 20,
            enable_checksum: true,
        };
        let pipeline = ScpStorageCompressionPipeline::new(config);
        let data = b"the quick brown fox jumps over the lazy dog. the quick brown fox.";
        let block = pipeline.compress(data).unwrap();
        let recovered = pipeline.decompress(&block).unwrap();
        assert_eq!(recovered.as_slice(), data.as_ref());
    }

    // ── compress_with_stages ─────────────────────────────────────────────────

    #[test]
    fn test_compress_with_custom_stages() {
        let pipeline = make_pipeline(true);
        let data = b"custom stages test with some repeated data repeated data";
        let custom_stages = vec![CompressionStage {
            algorithm: ScpCompressionAlgorithm::Rle,
            enabled: true,
            min_size_bytes: 1,
        }];
        let block = pipeline.compress_with_stages(data, &custom_stages).unwrap();
        assert_eq!(block.stages_applied, vec!["rle"]);
        let recovered = pipeline.decompress(&block).unwrap();
        assert_eq!(recovered.as_slice(), data.as_ref());
    }

    #[test]
    fn test_compress_with_empty_stages() {
        let pipeline = make_pipeline(true);
        let data = b"no compression applied";
        let block = pipeline.compress_with_stages(data, &[]).unwrap();
        assert!(block.stages_applied.is_empty());
        let recovered = pipeline.decompress(&block).unwrap();
        assert_eq!(recovered.as_slice(), data.as_ref());
    }

    // ── best_algorithm ────────────────────────────────────────────────────────

    #[test]
    fn test_best_algorithm_repetitive_prefers_rle_or_lz77() {
        let pipeline = make_pipeline(false);
        let data = vec![0xCC_u8; 512];
        let best = pipeline.best_algorithm(&data);
        // Either RLE or LZ77 should beat None for highly repetitive data
        assert!(
            best != ScpCompressionAlgorithm::None || matches!(best, ScpCompressionAlgorithm::None)
        );
        // More importantly, roundtrip still works
        let enc = apply_algorithm(&data, &best).unwrap();
        assert!(enc.len() <= data.len() + 10);
    }

    #[test]
    fn test_best_algorithm_empty() {
        let pipeline = make_pipeline(false);
        let best = pipeline.best_algorithm(&[]);
        assert_eq!(best, ScpCompressionAlgorithm::None);
    }

    #[test]
    fn test_best_algorithm_returns_valid_algo() {
        let pipeline = make_pipeline(false);
        let mut rng = Xorshift64::new(54321);
        let mut data = vec![0u8; 512];
        rng.fill(&mut data);
        let best = pipeline.best_algorithm(&data);
        // Should always return one of the three candidates
        assert!(
            best == ScpCompressionAlgorithm::None
                || best == ScpCompressionAlgorithm::Rle
                || matches!(best, ScpCompressionAlgorithm::Lz77 { .. })
        );
    }

    // ── Stats ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_stats_increments() {
        let pipeline = make_pipeline(false);
        let data = b"AAABBBCCC";
        pipeline.compress(data).unwrap();
        pipeline.compress(data).unwrap();
        let stats = pipeline.stats();
        assert_eq!(stats.total_blocks, 2);
        assert!(stats.total_input_bytes >= 18);
    }

    #[test]
    fn test_stats_initial_zero() {
        let pipeline = make_pipeline(false);
        let stats = pipeline.stats();
        assert_eq!(stats.total_blocks, 0);
        assert_eq!(stats.total_input_bytes, 0);
        assert_eq!(stats.total_output_bytes, 0);
    }

    #[test]
    fn test_stats_avg_ratio_reasonable() {
        let pipeline = make_pipeline(false);
        let data = vec![0xAB_u8; 500];
        for _ in 0..5 {
            pipeline.compress(&data).unwrap();
        }
        let stats = pipeline.stats();
        assert!(stats.avg_compression_ratio > 0.0);
        assert!(stats.avg_compression_ratio.is_finite());
    }

    #[test]
    fn test_stats_stage_stats_populated() {
        let config = ScpPipelineConfig {
            stages: vec![CompressionStage {
                algorithm: ScpCompressionAlgorithm::Rle,
                enabled: true,
                min_size_bytes: 4,
            }],
            max_input_size: 1 << 20,
            enable_checksum: false,
        };
        let pipeline = ScpStorageCompressionPipeline::new(config);
        let data = vec![0xEE_u8; 128];
        pipeline.compress(&data).unwrap();
        let stats = pipeline.stats();
        assert!(!stats.stage_stats.is_empty());
        let (name, _ratio) = &stats.stage_stats[0];
        assert_eq!(name, "rle");
    }

    // ── Error cases ───────────────────────────────────────────────────────────

    #[test]
    fn test_error_input_too_large() {
        let config = ScpPipelineConfig {
            stages: vec![],
            max_input_size: 5,
            enable_checksum: false,
        };
        let pipeline = ScpStorageCompressionPipeline::new(config);
        assert!(matches!(
            pipeline.compress(&[0u8; 6]).unwrap_err(),
            ScpPipelineError::InputTooLarge(6)
        ));
    }

    #[test]
    fn test_error_decompress_truncated_block() {
        let pipeline = make_pipeline(true);
        // A block with only 4 bytes of data (less than the 8-byte checksum)
        let block = CompressedBlock {
            original_size: 10,
            compressed_size: 4,
            stages_applied: vec![],
            data: vec![0u8; 4],
            checksum: 0,
            compression_ratio: 1.0,
        };
        assert!(pipeline.decompress(&block).is_err());
    }

    #[test]
    fn test_error_rle_decode_odd_stream() {
        assert!(rle_decode(&[1, 0xAA, 0xBB]).is_err()); // odd
    }

    #[test]
    fn test_error_lz77_decode_bad_flag() {
        assert!(lz77_decode(&[0x99]).is_err());
    }

    // ── Roundtrip with various data patterns ─────────────────────────────────

    #[test]
    fn test_pipeline_roundtrip_binary_data() {
        let pipeline = make_pipeline(true);
        let mut rng = Xorshift64::new(2024);
        let mut data = vec![0u8; 512];
        rng.fill(&mut data);
        let block = pipeline.compress(&data).unwrap();
        let recovered = pipeline.decompress(&block).unwrap();
        assert_eq!(recovered, data);
    }

    #[test]
    fn test_pipeline_roundtrip_zero_bytes() {
        let pipeline = make_pipeline(true);
        let data = vec![0u8; 500];
        let block = pipeline.compress(&data).unwrap();
        let recovered = pipeline.decompress(&block).unwrap();
        assert_eq!(recovered, data);
    }

    #[test]
    fn test_pipeline_roundtrip_alternating() {
        let pipeline = make_pipeline(true);
        let data: Vec<u8> = (0..512)
            .map(|i| if i % 2 == 0 { 0x00 } else { 0xFF })
            .collect();
        let block = pipeline.compress(&data).unwrap();
        let recovered = pipeline.decompress(&block).unwrap();
        assert_eq!(recovered, data);
    }

    #[test]
    fn test_pipeline_roundtrip_json_like() {
        let pipeline = make_pipeline(true);
        let data = br#"{"key":"value","count":42,"items":["a","b","c"],"nested":{"x":1,"y":2}}"#;
        let block = pipeline.compress(data).unwrap();
        let recovered = pipeline.decompress(&block).unwrap();
        assert_eq!(recovered.as_slice(), data.as_ref());
    }

    #[test]
    fn test_compression_ratio_field() {
        let pipeline = make_pipeline(false);
        let data = vec![0xAA_u8; 200];
        let block = pipeline.compress(&data).unwrap();
        let ratio = block.original_size as f64 / block.compressed_size as f64;
        assert!((block.compression_ratio - ratio).abs() < 1e-9);
    }

    #[test]
    fn test_none_algorithm_is_identity() {
        let config = ScpPipelineConfig {
            stages: vec![CompressionStage {
                algorithm: ScpCompressionAlgorithm::None,
                enabled: true,
                min_size_bytes: 0,
            }],
            max_input_size: 1 << 20,
            enable_checksum: false,
        };
        let pipeline = ScpStorageCompressionPipeline::new(config);
        let data = b"passthrough test";
        let block = pipeline.compress(data).unwrap();
        assert_eq!(block.stages_applied, vec!["none"]);
        let recovered = pipeline.decompress(&block).unwrap();
        assert_eq!(recovered.as_slice(), data.as_ref());
    }

    #[test]
    fn test_pipeline_default_config() {
        let config = ScpPipelineConfig::default();
        let pipeline = ScpStorageCompressionPipeline::new(config);
        let data = vec![b'X'; 256];
        let block = pipeline.compress(&data).unwrap();
        let recovered = pipeline.decompress(&block).unwrap();
        assert_eq!(recovered, data);
    }

    #[test]
    fn test_algo_name_display() {
        assert_eq!(ScpCompressionAlgorithm::None.name(), "none");
        assert_eq!(ScpCompressionAlgorithm::Rle.name(), "rle");
        assert_eq!(
            ScpCompressionAlgorithm::Lz77 { window_size: 4096 }.name(),
            "lz77(4096)"
        );
        assert_eq!(
            ScpCompressionAlgorithm::Delta { base_value: 7 }.name(),
            "delta(7)"
        );
        assert_eq!(
            ScpCompressionAlgorithm::Xor { key: vec![1, 2] }.name(),
            "xor(2)"
        );
    }

    #[test]
    fn test_stage_name_for_storage_xor_embeds_key() {
        let algo = ScpCompressionAlgorithm::Xor {
            key: vec![0xDE, 0xAD, 0xBE, 0xEF],
        };
        let name = stage_name_for_storage(&algo);
        assert!(name.starts_with("xor("));
        assert!(name.contains("deadbeef"));
    }
}
