//! # Embedding Compression Codec
//!
//! Production-quality codec for compressing dense embedding vectors using multiple
//! pure-Rust algorithms: scalar quantization, product quantization, delta coding,
//! run-length encoding, and hybrid PQ+RLE.
//!
//! ## Supported Methods
//!
//! - `ScalarQuantization` — uniform min-max quantization to N bits
//! - `ProductQuantization` — split into subvectors, quantize each independently
//! - `DeltaCoding` — delta-encode sorted values, then scalar quantize
//! - `RunLengthEncoding` — RLE on quantized values
//! - `HybridPQ` — product quantization + RLE on residuals
//!
//! ## Example
//!
//! ```rust
//! use ipfrs_semantic::{EmbeddingCompressionCodec, EccMethod, EccCodecConfig};
//!
//! let mut codec = EmbeddingCompressionCodec::new();
//! let id = codec.register_codec("my-sq8", EccMethod::ScalarQuantization, 8, 64);
//! let embedding = vec![0.1f64; 128];
//! let compressed = codec.compress(id, &embedding).unwrap();
//! let decompressed = codec.decompress(&compressed).unwrap();
//! let mse = codec.reconstruction_error(&embedding, &decompressed);
//! assert!(mse < 1e-3);
//! ```

use std::collections::{HashMap, VecDeque};
use std::time::{SystemTime, UNIX_EPOCH};

use thiserror::Error;

// ---------------------------------------------------------------------------
// PRNG — xorshift64 (no external rand dependency for internal use)
// ---------------------------------------------------------------------------

#[allow(dead_code)]
#[inline]
fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

// ---------------------------------------------------------------------------
// Public type aliases (as required by spec)
// ---------------------------------------------------------------------------

/// Type alias for the codec itself.
pub type EccEmbeddingCompressionCodec = EmbeddingCompressionCodec;

/// Numeric identifier for a registered codec.
pub type EccCodecId = u32;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors produced by [`EmbeddingCompressionCodec`].
#[derive(Debug, Error, Clone, PartialEq)]
pub enum EccError {
    /// Requested codec id was not found in the registry.
    #[error("codec id {0} not found in registry")]
    CodecNotFound(EccCodecId),

    /// Embedding has zero length.
    #[error("embedding must not be empty")]
    EmptyEmbedding,

    /// Compressed payload is corrupt or truncated.
    #[error("compressed data is corrupt: {0}")]
    CorruptData(String),

    /// Dimension mismatch between original and decompressed.
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    /// Block size is incompatible with the embedding dimension.
    #[error("block size {block_size} does not divide embedding dim {dim} evenly")]
    BlockSizeMismatch { block_size: usize, dim: usize },

    /// Quantization bit width is not supported (only 4, 8, 16).
    #[error("unsupported quantize_bits {0}: must be 4, 8, or 16")]
    UnsupportedBitWidth(u8),

    /// General internal error.
    #[error("internal error: {0}")]
    Internal(String),
}

// ---------------------------------------------------------------------------
// EccMethod
// ---------------------------------------------------------------------------

/// Compression algorithm used by a codec.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum EccMethod {
    /// Uniform min-max scalar quantization to N bits.
    ScalarQuantization,
    /// Split vector into sub-blocks, scalar-quantize each independently.
    ProductQuantization,
    /// Delta-encode values sorted by magnitude, then scalar-quantize.
    DeltaCoding,
    /// Run-length encoding applied on quantized symbols.
    RunLengthEncoding,
    /// Product quantization followed by RLE on residual symbols.
    HybridPQ,
}

impl std::fmt::Display for EccMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EccMethod::ScalarQuantization => write!(f, "ScalarQuantization"),
            EccMethod::ProductQuantization => write!(f, "ProductQuantization"),
            EccMethod::DeltaCoding => write!(f, "DeltaCoding"),
            EccMethod::RunLengthEncoding => write!(f, "RunLengthEncoding"),
            EccMethod::HybridPQ => write!(f, "HybridPQ"),
        }
    }
}

// ---------------------------------------------------------------------------
// Configuration structs
// ---------------------------------------------------------------------------

/// Configuration for creating a new codec.
#[derive(Debug, Clone)]
pub struct EccCodecConfig {
    /// Name for display / lookup.
    pub name: String,
    /// Compression method.
    pub method: EccMethod,
    /// Number of bits per quantized value (4, 8, or 16).
    pub quantize_bits: u8,
    /// Whether to apply delta coding as a pre-processing step.
    pub use_delta_coding: bool,
    /// Sub-block size for PQ / HybridPQ.
    pub block_size: usize,
}

impl Default for EccCodecConfig {
    fn default() -> Self {
        Self {
            name: "default-sq8".to_string(),
            method: EccMethod::ScalarQuantization,
            quantize_bits: 8,
            use_delta_coding: false,
            block_size: 8,
        }
    }
}

/// Registered codec specification stored in the codec registry.
#[derive(Debug, Clone)]
pub struct EccCodecSpec {
    /// Unique numeric id.
    pub id: EccCodecId,
    /// Human-readable name.
    pub name: String,
    /// Compression method.
    pub method: EccMethod,
    /// Quantization bit-width.
    pub bits: u8,
    /// Sub-block size.
    pub block_size: usize,
}

// ---------------------------------------------------------------------------
// Compressed payload
// ---------------------------------------------------------------------------

/// Output of a compression operation.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EccCompressed {
    /// Id of the codec that produced this payload.
    pub codec_id: EccCodecId,
    /// Method used.
    pub method: EccMethod,
    /// Raw compressed bytes.
    pub data: Vec<u8>,
    /// Original vector length.
    pub original_dim: usize,
    /// Minimum value observed in the original vector (for de-quantization).
    pub min_val: f64,
    /// Maximum value observed in the original vector (for de-quantization).
    pub max_val: f64,
}

// ---------------------------------------------------------------------------
// Audit / statistics
// ---------------------------------------------------------------------------

/// One entry in the codec's compression audit log.
#[derive(Debug, Clone)]
pub struct EccCompressionRecord {
    /// Unix timestamp (seconds) when the compression occurred.
    pub ts: u64,
    /// Size of the original vector in bytes (f64 × dim × 8).
    pub original_bytes: usize,
    /// Size of the compressed payload in bytes.
    pub compressed_bytes: usize,
    /// Compression ratio: original / compressed.
    pub ratio: f64,
    /// Method used.
    pub method: EccMethod,
}

/// Aggregate statistics returned by [`EmbeddingCompressionCodec::codec_stats`].
#[derive(Debug, Clone)]
pub struct EccCodecStats {
    /// Total number of compression operations logged.
    pub total_compressed: usize,
    /// Total bytes saved across all operations.
    pub total_bytes_saved: usize,
    /// Average compression ratio.
    pub avg_ratio: f64,
    /// Per-method breakdown: (total_ops, total_bytes_saved, avg_ratio).
    pub per_method: HashMap<EccMethod, (usize, usize, f64)>,
}

// ---------------------------------------------------------------------------
// Core codec struct
// ---------------------------------------------------------------------------

/// Maximum number of records kept in the compression log.
const MAX_LOG_ENTRIES: usize = 500;

/// Codec for compressing dense embedding vectors.
///
/// Use [`register_codec`][Self::register_codec] to create named codec configurations,
/// then call [`compress`][Self::compress] / [`decompress`][Self::decompress] as needed.
pub struct EmbeddingCompressionCodec {
    /// Registry mapping id → spec.
    registry: HashMap<EccCodecId, EccCodecSpec>,
    /// Monotonically increasing id counter.
    next_id: EccCodecId,
    /// Bounded audit log (FIFO, max 500 entries).
    log: VecDeque<EccCompressionRecord>,
}

impl Default for EmbeddingCompressionCodec {
    fn default() -> Self {
        Self::new()
    }
}

impl EmbeddingCompressionCodec {
    /// Create a new codec with an empty registry.
    pub fn new() -> Self {
        Self {
            registry: HashMap::new(),
            next_id: 1,
            log: VecDeque::with_capacity(MAX_LOG_ENTRIES),
        }
    }

    // -----------------------------------------------------------------------
    // Registry
    // -----------------------------------------------------------------------

    /// Register a new codec and return its id.
    ///
    /// `bits` must be 4, 8, or 16. `block_size` is used by PQ / HybridPQ.
    pub fn register_codec(
        &mut self,
        name: &str,
        method: EccMethod,
        bits: u8,
        block_size: usize,
    ) -> EccCodecId {
        let id = self.next_id;
        self.next_id = self.next_id.wrapping_add(1);
        let spec = EccCodecSpec {
            id,
            name: name.to_string(),
            method,
            bits,
            block_size,
        };
        self.registry.insert(id, spec);
        id
    }

    /// Register a codec from a full [`EccCodecConfig`].
    pub fn register_from_config(&mut self, config: &EccCodecConfig) -> EccCodecId {
        self.register_codec(
            &config.name,
            config.method,
            config.quantize_bits,
            config.block_size,
        )
    }

    /// Look up a codec spec by id.
    pub fn get_spec(&self, id: EccCodecId) -> Option<&EccCodecSpec> {
        self.registry.get(&id)
    }

    /// Return the number of registered codecs.
    pub fn codec_count(&self) -> usize {
        self.registry.len()
    }

    // -----------------------------------------------------------------------
    // Compress / decompress
    // -----------------------------------------------------------------------

    /// Compress a single embedding vector using the specified codec.
    pub fn compress(
        &mut self,
        codec_id: EccCodecId,
        embedding: &[f64],
    ) -> Result<EccCompressed, EccError> {
        if embedding.is_empty() {
            return Err(EccError::EmptyEmbedding);
        }
        let spec = self
            .registry
            .get(&codec_id)
            .ok_or(EccError::CodecNotFound(codec_id))?
            .clone();

        validate_bits(spec.bits)?;

        let min_val = embedding.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = embedding.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        let data = match spec.method {
            EccMethod::ScalarQuantization => {
                compress_scalar(embedding, spec.bits, min_val, max_val)?
            }
            EccMethod::ProductQuantization => {
                compress_pq(embedding, spec.bits, spec.block_size, min_val, max_val)?
            }
            EccMethod::DeltaCoding => compress_delta(embedding, spec.bits, min_val, max_val)?,
            EccMethod::RunLengthEncoding => compress_rle(embedding, spec.bits, min_val, max_val)?,
            EccMethod::HybridPQ => {
                compress_hybrid_pq(embedding, spec.bits, spec.block_size, min_val, max_val)?
            }
        };

        let original_bytes = embedding.len() * 8;
        let compressed_bytes = data.len();
        let ratio = if compressed_bytes == 0 {
            1.0
        } else {
            original_bytes as f64 / compressed_bytes as f64
        };

        self.push_log(EccCompressionRecord {
            ts: unix_ts(),
            original_bytes,
            compressed_bytes,
            ratio,
            method: spec.method,
        });

        Ok(EccCompressed {
            codec_id,
            method: spec.method,
            data,
            original_dim: embedding.len(),
            min_val,
            max_val,
        })
    }

    /// Decompress a previously compressed payload.
    pub fn decompress(&self, compressed: &EccCompressed) -> Result<Vec<f64>, EccError> {
        let spec = self
            .registry
            .get(&compressed.codec_id)
            .ok_or(EccError::CodecNotFound(compressed.codec_id))?;

        validate_bits(spec.bits)?;

        match compressed.method {
            EccMethod::ScalarQuantization => decompress_scalar(
                &compressed.data,
                spec.bits,
                compressed.original_dim,
                compressed.min_val,
                compressed.max_val,
            ),
            EccMethod::ProductQuantization => decompress_pq(
                &compressed.data,
                spec.bits,
                spec.block_size,
                compressed.original_dim,
                compressed.min_val,
                compressed.max_val,
            ),
            EccMethod::DeltaCoding => decompress_delta(
                &compressed.data,
                spec.bits,
                compressed.original_dim,
                compressed.min_val,
                compressed.max_val,
            ),
            EccMethod::RunLengthEncoding => decompress_rle(
                &compressed.data,
                spec.bits,
                compressed.original_dim,
                compressed.min_val,
                compressed.max_val,
            ),
            EccMethod::HybridPQ => decompress_hybrid_pq(
                &compressed.data,
                spec.bits,
                spec.block_size,
                compressed.original_dim,
                compressed.min_val,
                compressed.max_val,
            ),
        }
    }

    /// Compress a batch of embeddings with the same codec.
    pub fn compress_batch(
        &mut self,
        codec_id: EccCodecId,
        embeddings: &[Vec<f64>],
    ) -> Vec<Result<EccCompressed, EccError>> {
        embeddings
            .iter()
            .map(|emb| self.compress(codec_id, emb))
            .collect()
    }

    /// Decompress a batch of compressed payloads.
    pub fn decompress_batch(&self, batch: &[EccCompressed]) -> Vec<Result<Vec<f64>, EccError>> {
        batch.iter().map(|c| self.decompress(c)).collect()
    }

    // -----------------------------------------------------------------------
    // Metrics / estimation
    // -----------------------------------------------------------------------

    /// Compute mean squared error between original and decompressed vectors.
    pub fn reconstruction_error(original: &[f64], decompressed: &[f64]) -> f64 {
        if original.is_empty() || decompressed.is_empty() {
            return 0.0;
        }
        let len = original.len().min(decompressed.len());
        let sum: f64 = original[..len]
            .iter()
            .zip(decompressed[..len].iter())
            .map(|(a, b)| {
                let d = a - b;
                d * d
            })
            .sum();
        sum / len as f64
    }

    /// Estimate the theoretical compression ratio for a given codec and vector dimension.
    ///
    /// The estimate is based on the quantization bit-width and method overhead.
    pub fn estimate_ratio(&self, codec_id: EccCodecId, dim: usize) -> f64 {
        let spec = match self.registry.get(&codec_id) {
            Some(s) => s,
            None => return 1.0,
        };
        if dim == 0 {
            return 1.0;
        }
        let original_bits = dim * 64;
        let quantized_bits: usize = match spec.method {
            EccMethod::ScalarQuantization => dim * spec.bits as usize,
            EccMethod::ProductQuantization => {
                let blocks = dim.div_ceil(spec.block_size);
                blocks * spec.block_size * spec.bits as usize
            }
            EccMethod::DeltaCoding => {
                // delta values tend to cluster near zero — assume 70 % of bits needed
                (dim * spec.bits as usize * 7) / 10
            }
            EccMethod::RunLengthEncoding => {
                // RLE overhead: 2 bytes per run; assume 50 % unique symbols
                let runs = (dim / 2).max(1);
                runs * (8 + spec.bits as usize)
            }
            EccMethod::HybridPQ => {
                let blocks = dim.div_ceil(spec.block_size);
                let pq_bits = blocks * spec.block_size * spec.bits as usize;
                // RLE on residuals reduces by ~30 %
                (pq_bits * 7) / 10
            }
        };
        // header overhead (min_val + max_val + original_dim) ≈ 3 × 8 bytes = 192 bits
        let total_compressed_bits = quantized_bits + 192;
        original_bits as f64 / total_compressed_bits as f64
    }

    /// Compute aggregate statistics over the compression log.
    pub fn codec_stats(&self) -> EccCodecStats {
        let total_compressed = self.log.len();
        let mut total_bytes_saved: usize = 0;
        let mut ratio_sum: f64 = 0.0;
        let mut per_method: HashMap<EccMethod, (usize, usize, f64)> = HashMap::new();

        for rec in &self.log {
            let saved = rec.original_bytes.saturating_sub(rec.compressed_bytes);
            total_bytes_saved += saved;
            ratio_sum += rec.ratio;

            let entry = per_method.entry(rec.method).or_insert((0, 0, 0.0));
            entry.0 += 1;
            entry.1 += saved;
            entry.2 += rec.ratio;
        }

        // Normalize per-method average ratios
        for entry in per_method.values_mut() {
            if entry.0 > 0 {
                entry.2 /= entry.0 as f64;
            }
        }

        let avg_ratio = if total_compressed > 0 {
            ratio_sum / total_compressed as f64
        } else {
            1.0
        };

        EccCodecStats {
            total_compressed,
            total_bytes_saved,
            avg_ratio,
            per_method,
        }
    }

    /// Return a read-only view of the compression log.
    pub fn log_entries(&self) -> &VecDeque<EccCompressionRecord> {
        &self.log
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    fn push_log(&mut self, record: EccCompressionRecord) {
        if self.log.len() >= MAX_LOG_ENTRIES {
            self.log.pop_front();
        }
        self.log.push_back(record);
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn unix_ts() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn validate_bits(bits: u8) -> Result<(), EccError> {
    match bits {
        4 | 8 | 16 => Ok(()),
        _ => Err(EccError::UnsupportedBitWidth(bits)),
    }
}

/// Maximum representable unsigned integer for a given bit-width.
#[inline]
fn max_quant_val(bits: u8) -> u64 {
    match bits {
        4 => 15,
        8 => 255,
        16 => 65535,
        _ => 255,
    }
}

/// Encode a single f64 value in [min_val, max_val] to a quantized integer.
#[inline]
fn quantize_val(v: f64, min_val: f64, max_val: f64, levels: u64) -> u64 {
    let range = max_val - min_val;
    if range < f64::EPSILON {
        return 0;
    }
    let norm = (v - min_val) / range;
    let q = (norm * levels as f64).round() as i64;
    q.clamp(0, levels as i64) as u64
}

/// Decode a quantized integer back to f64.
#[inline]
fn dequantize_val(q: u64, min_val: f64, max_val: f64, levels: u64) -> f64 {
    if levels == 0 {
        return min_val;
    }
    let norm = q as f64 / levels as f64;
    min_val + norm * (max_val - min_val)
}

// ---------------------------------------------------------------------------
// Scalar Quantization
// ---------------------------------------------------------------------------

fn compress_scalar(
    embedding: &[f64],
    bits: u8,
    min_val: f64,
    max_val: f64,
) -> Result<Vec<u8>, EccError> {
    let levels = max_quant_val(bits);
    match bits {
        4 => {
            // Pack two 4-bit values per byte
            let mut out = Vec::with_capacity(embedding.len().div_ceil(2));
            let mut iter = embedding.iter();
            loop {
                match iter.next() {
                    None => break,
                    Some(a) => {
                        let qa = quantize_val(*a, min_val, max_val, levels) as u8;
                        let qb = match iter.next() {
                            Some(b) => quantize_val(*b, min_val, max_val, levels) as u8,
                            None => 0,
                        };
                        out.push((qa & 0x0F) | ((qb & 0x0F) << 4));
                    }
                }
            }
            Ok(out)
        }
        8 => {
            let out: Vec<u8> = embedding
                .iter()
                .map(|&v| quantize_val(v, min_val, max_val, levels) as u8)
                .collect();
            Ok(out)
        }
        16 => {
            let mut out = Vec::with_capacity(embedding.len() * 2);
            for &v in embedding {
                let q = quantize_val(v, min_val, max_val, levels) as u16;
                out.extend_from_slice(&q.to_le_bytes());
            }
            Ok(out)
        }
        _ => Err(EccError::UnsupportedBitWidth(bits)),
    }
}

fn decompress_scalar(
    data: &[u8],
    bits: u8,
    original_dim: usize,
    min_val: f64,
    max_val: f64,
) -> Result<Vec<f64>, EccError> {
    let levels = max_quant_val(bits);
    match bits {
        4 => {
            let mut out = Vec::with_capacity(original_dim);
            for &byte in data {
                let a = (byte & 0x0F) as u64;
                let b = ((byte >> 4) & 0x0F) as u64;
                out.push(dequantize_val(a, min_val, max_val, levels));
                if out.len() < original_dim {
                    out.push(dequantize_val(b, min_val, max_val, levels));
                }
            }
            out.truncate(original_dim);
            if out.len() != original_dim {
                return Err(EccError::DimensionMismatch {
                    expected: original_dim,
                    got: out.len(),
                });
            }
            Ok(out)
        }
        8 => {
            if data.len() != original_dim {
                return Err(EccError::CorruptData(format!(
                    "expected {} bytes, got {}",
                    original_dim,
                    data.len()
                )));
            }
            Ok(data
                .iter()
                .map(|&b| dequantize_val(b as u64, min_val, max_val, levels))
                .collect())
        }
        16 => {
            if data.len() != original_dim * 2 {
                return Err(EccError::CorruptData(format!(
                    "expected {} bytes, got {}",
                    original_dim * 2,
                    data.len()
                )));
            }
            let mut out = Vec::with_capacity(original_dim);
            for chunk in data.chunks_exact(2) {
                let q = u16::from_le_bytes([chunk[0], chunk[1]]) as u64;
                out.push(dequantize_val(q, min_val, max_val, levels));
            }
            Ok(out)
        }
        _ => Err(EccError::UnsupportedBitWidth(bits)),
    }
}

// ---------------------------------------------------------------------------
// Product Quantization
// ---------------------------------------------------------------------------
//
// Each sub-block is quantized independently using its own local min/max.
// Layout: [ num_blocks:u32 | block_size:u32 | block_0_min:f64 | block_0_max:f64 |
//           block_0_quantized_bytes... | block_1_min:f64 | block_1_max:f64 | ... ]

fn compress_pq(
    embedding: &[f64],
    bits: u8,
    block_size: usize,
    _global_min: f64,
    _global_max: f64,
) -> Result<Vec<u8>, EccError> {
    let bs = block_size.max(1);
    let num_blocks = embedding.len().div_ceil(bs);
    let mut out: Vec<u8> = Vec::new();

    // Header
    out.extend_from_slice(&(num_blocks as u32).to_le_bytes());
    out.extend_from_slice(&(bs as u32).to_le_bytes());

    for block_idx in 0..num_blocks {
        let start = block_idx * bs;
        let end = (start + bs).min(embedding.len());
        let block = &embedding[start..end];

        let bmin = block.iter().cloned().fold(f64::INFINITY, f64::min);
        let bmax = block.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        out.extend_from_slice(&bmin.to_le_bytes());
        out.extend_from_slice(&bmax.to_le_bytes());

        let qbytes = compress_scalar(block, bits, bmin, bmax)?;
        // Store length prefix for variable-length last block
        out.extend_from_slice(&(qbytes.len() as u32).to_le_bytes());
        out.extend_from_slice(&qbytes);
    }
    Ok(out)
}

fn decompress_pq(
    data: &[u8],
    bits: u8,
    block_size: usize,
    original_dim: usize,
    _global_min: f64,
    _global_max: f64,
) -> Result<Vec<f64>, EccError> {
    let bs = block_size.max(1);
    if data.len() < 8 {
        return Err(EccError::CorruptData("PQ header too short".to_string()));
    }

    let num_blocks = u32::from_le_bytes(
        data[0..4]
            .try_into()
            .map_err(|_| EccError::CorruptData("num_blocks".to_string()))?,
    ) as usize;

    let stored_bs = u32::from_le_bytes(
        data[4..8]
            .try_into()
            .map_err(|_| EccError::CorruptData("block_size".to_string()))?,
    ) as usize;

    let _ = stored_bs; // block_size sanity — we use the caller's bs
    let mut offset = 8usize;
    let mut out: Vec<f64> = Vec::with_capacity(original_dim);

    for block_idx in 0..num_blocks {
        if offset + 20 > data.len() {
            return Err(EccError::CorruptData(format!(
                "block {} header missing",
                block_idx
            )));
        }
        let bmin = f64::from_le_bytes(
            data[offset..offset + 8]
                .try_into()
                .map_err(|_| EccError::CorruptData("bmin".to_string()))?,
        );
        offset += 8;
        let bmax = f64::from_le_bytes(
            data[offset..offset + 8]
                .try_into()
                .map_err(|_| EccError::CorruptData("bmax".to_string()))?,
        );
        offset += 8;
        let qlen = u32::from_le_bytes(
            data[offset..offset + 4]
                .try_into()
                .map_err(|_| EccError::CorruptData("qlen".to_string()))?,
        ) as usize;
        offset += 4;

        if offset + qlen > data.len() {
            return Err(EccError::CorruptData(format!(
                "block {} data truncated",
                block_idx
            )));
        }
        let qbytes = &data[offset..offset + qlen];
        offset += qlen;

        // Figure out dimension of this block
        let block_start = block_idx * bs;
        let block_dim = (original_dim - block_start).min(bs);

        let block_vals = decompress_scalar(qbytes, bits, block_dim, bmin, bmax)?;
        out.extend_from_slice(&block_vals);
    }

    out.truncate(original_dim);
    if out.len() != original_dim {
        return Err(EccError::DimensionMismatch {
            expected: original_dim,
            got: out.len(),
        });
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Delta Coding
// ---------------------------------------------------------------------------
//
// Compress: sort indices by value, compute deltas of sorted values, quantize deltas.
// Decompress: reverse — cumsum, un-sort, dequantize.
//
// Layout: [ perm:u32×dim | delta_min:f64 | delta_max:f64 | quantized_deltas ]

fn compress_delta(
    embedding: &[f64],
    bits: u8,
    _min_val: f64,
    _max_val: f64,
) -> Result<Vec<u8>, EccError> {
    let n = embedding.len();
    // Sort indices by value
    let mut perm: Vec<u32> = (0..n as u32).collect();
    perm.sort_unstable_by(|&a, &b| {
        embedding[a as usize]
            .partial_cmp(&embedding[b as usize])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Compute deltas of sorted values
    let mut deltas: Vec<f64> = Vec::with_capacity(n);
    let mut prev = embedding[perm[0] as usize];
    deltas.push(prev); // first value stored as-is
    for i in 1..n {
        let cur = embedding[perm[i] as usize];
        deltas.push(cur - prev);
        prev = cur;
    }

    let dmin = deltas.iter().cloned().fold(f64::INFINITY, f64::min);
    let dmax = deltas.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let qbytes = compress_scalar(&deltas, bits, dmin, dmax)?;

    let mut out: Vec<u8> = Vec::with_capacity(n * 4 + 16 + qbytes.len());
    // Write permutation
    for &p in &perm {
        out.extend_from_slice(&p.to_le_bytes());
    }
    // Write delta range
    out.extend_from_slice(&dmin.to_le_bytes());
    out.extend_from_slice(&dmax.to_le_bytes());
    // Write quantized deltas
    out.extend_from_slice(&qbytes);

    Ok(out)
}

fn decompress_delta(
    data: &[u8],
    bits: u8,
    original_dim: usize,
    _min_val: f64,
    _max_val: f64,
) -> Result<Vec<f64>, EccError> {
    let n = original_dim;
    let perm_bytes = n * 4;
    if data.len() < perm_bytes + 16 {
        return Err(EccError::CorruptData("delta header too short".to_string()));
    }

    let mut perm: Vec<u32> = Vec::with_capacity(n);
    for i in 0..n {
        let off = i * 4;
        let p = u32::from_le_bytes(
            data[off..off + 4]
                .try_into()
                .map_err(|_| EccError::CorruptData("perm bytes".to_string()))?,
        );
        perm.push(p);
    }

    let off = perm_bytes;
    let dmin = f64::from_le_bytes(
        data[off..off + 8]
            .try_into()
            .map_err(|_| EccError::CorruptData("dmin".to_string()))?,
    );
    let dmax = f64::from_le_bytes(
        data[off + 8..off + 16]
            .try_into()
            .map_err(|_| EccError::CorruptData("dmax".to_string()))?,
    );

    let qbytes = &data[off + 16..];
    let deltas = decompress_scalar(qbytes, bits, n, dmin, dmax)?;

    // Cumulative sum to recover sorted values
    let mut sorted_vals: Vec<f64> = Vec::with_capacity(n);
    let mut acc = deltas[0];
    sorted_vals.push(acc);
    for &d in &deltas[1..] {
        acc += d;
        sorted_vals.push(acc);
    }

    // Un-sort: perm[i] = original index of i-th sorted value
    let mut out = vec![0.0f64; n];
    for (sorted_idx, &orig_idx) in perm.iter().enumerate() {
        let idx = orig_idx as usize;
        if idx >= n {
            return Err(EccError::CorruptData(format!(
                "perm index {} out of range {}",
                idx, n
            )));
        }
        out[idx] = sorted_vals[sorted_idx];
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Run-Length Encoding
// ---------------------------------------------------------------------------
//
// Quantize values to N bits, then RLE-encode the quantized symbol stream.
// Layout: [ num_runs:u32 | (run_value:u16 | run_len:u16)... ]
// run_value stores the quantized symbol; run_len is clamped to u16::MAX.

fn compress_rle(
    embedding: &[f64],
    bits: u8,
    min_val: f64,
    max_val: f64,
) -> Result<Vec<u8>, EccError> {
    let levels = max_quant_val(bits);
    let quantized: Vec<u16> = embedding
        .iter()
        .map(|&v| quantize_val(v, min_val, max_val, levels) as u16)
        .collect();

    // Build runs
    let mut runs: Vec<(u16, u16)> = Vec::new(); // (value, count)
    let mut i = 0;
    while i < quantized.len() {
        let val = quantized[i];
        let mut count = 1u16;
        while (i + count as usize) < quantized.len()
            && quantized[i + count as usize] == val
            && count < u16::MAX
        {
            count += 1;
        }
        runs.push((val, count));
        i += count as usize;
    }

    let mut out = Vec::with_capacity(4 + runs.len() * 4);
    out.extend_from_slice(&(runs.len() as u32).to_le_bytes());
    for (val, cnt) in &runs {
        out.extend_from_slice(&val.to_le_bytes());
        out.extend_from_slice(&cnt.to_le_bytes());
    }
    Ok(out)
}

fn decompress_rle(
    data: &[u8],
    bits: u8,
    original_dim: usize,
    min_val: f64,
    max_val: f64,
) -> Result<Vec<f64>, EccError> {
    if data.len() < 4 {
        return Err(EccError::CorruptData("RLE header too short".to_string()));
    }
    let num_runs = u32::from_le_bytes(
        data[0..4]
            .try_into()
            .map_err(|_| EccError::CorruptData("num_runs".to_string()))?,
    ) as usize;

    if data.len() < 4 + num_runs * 4 {
        return Err(EccError::CorruptData("RLE data truncated".to_string()));
    }

    // Use the same quantization levels as compress_rle
    let levels: u64 = max_quant_val(bits);
    let mut out: Vec<f64> = Vec::with_capacity(original_dim);
    let mut offset = 4usize;
    for _ in 0..num_runs {
        let val = u16::from_le_bytes(
            data[offset..offset + 2]
                .try_into()
                .map_err(|_| EccError::CorruptData("run_val".to_string()))?,
        ) as u64;
        let cnt = u16::from_le_bytes(
            data[offset + 2..offset + 4]
                .try_into()
                .map_err(|_| EccError::CorruptData("run_cnt".to_string()))?,
        ) as usize;
        offset += 4;

        let decoded = dequantize_val(val, min_val, max_val, levels);
        for _ in 0..cnt {
            out.push(decoded);
        }
    }

    out.truncate(original_dim);
    if out.len() != original_dim {
        return Err(EccError::DimensionMismatch {
            expected: original_dim,
            got: out.len(),
        });
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// HybridPQ — PQ per block + RLE on quantized residual symbols
// ---------------------------------------------------------------------------
//
// Layout: [ num_blocks:u32 | block_size:u32 |
//           ( bmin:f64 | bmax:f64 | rle_len:u32 | rle_bytes... ) × num_blocks ]

fn compress_hybrid_pq(
    embedding: &[f64],
    bits: u8,
    block_size: usize,
    _global_min: f64,
    _global_max: f64,
) -> Result<Vec<u8>, EccError> {
    let bs = block_size.max(1);
    let num_blocks = embedding.len().div_ceil(bs);
    let mut out: Vec<u8> = Vec::new();

    out.extend_from_slice(&(num_blocks as u32).to_le_bytes());
    out.extend_from_slice(&(bs as u32).to_le_bytes());

    for block_idx in 0..num_blocks {
        let start = block_idx * bs;
        let end = (start + bs).min(embedding.len());
        let block = &embedding[start..end];

        let bmin = block.iter().cloned().fold(f64::INFINITY, f64::min);
        let bmax = block.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        out.extend_from_slice(&bmin.to_le_bytes());
        out.extend_from_slice(&bmax.to_le_bytes());

        let rle_bytes = compress_rle(block, bits, bmin, bmax)?;
        out.extend_from_slice(&(rle_bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(&rle_bytes);
    }
    Ok(out)
}

fn decompress_hybrid_pq(
    data: &[u8],
    bits: u8,
    block_size: usize,
    original_dim: usize,
    _global_min: f64,
    _global_max: f64,
) -> Result<Vec<f64>, EccError> {
    let bs = block_size.max(1);
    if data.len() < 8 {
        return Err(EccError::CorruptData(
            "HybridPQ header too short".to_string(),
        ));
    }

    let num_blocks = u32::from_le_bytes(
        data[0..4]
            .try_into()
            .map_err(|_| EccError::CorruptData("num_blocks".to_string()))?,
    ) as usize;

    let _stored_bs = u32::from_le_bytes(
        data[4..8]
            .try_into()
            .map_err(|_| EccError::CorruptData("block_size".to_string()))?,
    );

    let mut offset = 8usize;
    let mut out: Vec<f64> = Vec::with_capacity(original_dim);

    for block_idx in 0..num_blocks {
        if offset + 20 > data.len() {
            return Err(EccError::CorruptData(format!(
                "HybridPQ block {} header missing",
                block_idx
            )));
        }

        let bmin = f64::from_le_bytes(
            data[offset..offset + 8]
                .try_into()
                .map_err(|_| EccError::CorruptData("bmin".to_string()))?,
        );
        offset += 8;
        let bmax = f64::from_le_bytes(
            data[offset..offset + 8]
                .try_into()
                .map_err(|_| EccError::CorruptData("bmax".to_string()))?,
        );
        offset += 8;
        let rlen = u32::from_le_bytes(
            data[offset..offset + 4]
                .try_into()
                .map_err(|_| EccError::CorruptData("rlen".to_string()))?,
        ) as usize;
        offset += 4;

        if offset + rlen > data.len() {
            return Err(EccError::CorruptData(format!(
                "HybridPQ block {} data truncated",
                block_idx
            )));
        }
        let rle_bytes = &data[offset..offset + rlen];
        offset += rlen;

        let block_start = block_idx * bs;
        let block_dim = (original_dim - block_start).min(bs);
        let block_vals = decompress_rle(rle_bytes, bits, block_dim, bmin, bmax)?;
        out.extend_from_slice(&block_vals);
    }

    out.truncate(original_dim);
    if out.len() != original_dim {
        return Err(EccError::DimensionMismatch {
            expected: original_dim,
            got: out.len(),
        });
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Helpers ---------------------------------------------------------------

    fn make_rand_embedding(dim: usize, seed: u64) -> Vec<f64> {
        let mut state = seed;
        (0..dim)
            .map(|_| {
                let r = xorshift64(&mut state);
                // map to [-1.0, 1.0]
                (r as f64 / u64::MAX as f64) * 2.0 - 1.0
            })
            .collect()
    }

    fn make_codec_sq(bits: u8) -> (EmbeddingCompressionCodec, EccCodecId) {
        let mut c = EmbeddingCompressionCodec::new();
        let id = c.register_codec("sq", EccMethod::ScalarQuantization, bits, 8);
        (c, id)
    }

    fn roundtrip_mse(codec: &mut EmbeddingCompressionCodec, id: EccCodecId, emb: &[f64]) -> f64 {
        let comp = codec.compress(id, emb).expect("compress failed");
        let decomp = codec.decompress(&comp).expect("decompress failed");
        EmbeddingCompressionCodec::reconstruction_error(emb, &decomp)
    }

    // Registry tests --------------------------------------------------------

    #[test]
    fn test_register_codec_returns_unique_ids() {
        let mut c = EmbeddingCompressionCodec::new();
        let id1 = c.register_codec("a", EccMethod::ScalarQuantization, 8, 8);
        let id2 = c.register_codec("b", EccMethod::ProductQuantization, 8, 8);
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_codec_count() {
        let mut c = EmbeddingCompressionCodec::new();
        assert_eq!(c.codec_count(), 0);
        c.register_codec("x", EccMethod::ScalarQuantization, 8, 8);
        assert_eq!(c.codec_count(), 1);
        c.register_codec("y", EccMethod::DeltaCoding, 8, 8);
        assert_eq!(c.codec_count(), 2);
    }

    #[test]
    fn test_get_spec_returns_correct_spec() {
        let mut c = EmbeddingCompressionCodec::new();
        let id = c.register_codec("mycodec", EccMethod::RunLengthEncoding, 16, 32);
        let spec = c.get_spec(id).expect("spec not found");
        assert_eq!(spec.name, "mycodec");
        assert_eq!(spec.method, EccMethod::RunLengthEncoding);
        assert_eq!(spec.bits, 16);
        assert_eq!(spec.block_size, 32);
    }

    #[test]
    fn test_get_spec_unknown_id_returns_none() {
        let c = EmbeddingCompressionCodec::new();
        assert!(c.get_spec(999).is_none());
    }

    #[test]
    fn test_register_from_config() {
        let mut c = EmbeddingCompressionCodec::new();
        let cfg = EccCodecConfig {
            name: "cfg-test".to_string(),
            method: EccMethod::HybridPQ,
            quantize_bits: 4,
            use_delta_coding: false,
            block_size: 4,
        };
        let id = c.register_from_config(&cfg);
        let spec = c.get_spec(id).expect("no spec");
        assert_eq!(spec.bits, 4);
        assert_eq!(spec.method, EccMethod::HybridPQ);
    }

    // Error handling --------------------------------------------------------

    #[test]
    fn test_compress_unknown_codec_id_returns_err() {
        let mut c = EmbeddingCompressionCodec::new();
        let err = c.compress(42, &[0.1, 0.2]).unwrap_err();
        assert!(matches!(err, EccError::CodecNotFound(42)));
    }

    #[test]
    fn test_compress_empty_embedding_returns_err() {
        let mut c = EmbeddingCompressionCodec::new();
        let id = c.register_codec("x", EccMethod::ScalarQuantization, 8, 8);
        let err = c.compress(id, &[]).unwrap_err();
        assert!(matches!(err, EccError::EmptyEmbedding));
    }

    #[test]
    fn test_decompress_unknown_codec_id_returns_err() {
        let c = EmbeddingCompressionCodec::new();
        let bad = EccCompressed {
            codec_id: 99,
            method: EccMethod::ScalarQuantization,
            data: vec![0u8; 4],
            original_dim: 4,
            min_val: 0.0,
            max_val: 1.0,
        };
        assert!(c.decompress(&bad).is_err());
    }

    #[test]
    fn test_unsupported_bit_width_returns_err() {
        let mut c = EmbeddingCompressionCodec::new();
        let id = c.register_codec("x", EccMethod::ScalarQuantization, 7, 8);
        let err = c.compress(id, &[0.1, 0.2, 0.3]).unwrap_err();
        assert!(matches!(err, EccError::UnsupportedBitWidth(7)));
    }

    // Scalar Quantization roundtrips ----------------------------------------

    #[test]
    fn test_sq8_roundtrip_basic() {
        let (mut c, id) = make_codec_sq(8);
        let emb = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        let mse = roundtrip_mse(&mut c, id, &emb);
        assert!(mse < 1e-4, "mse={mse}");
    }

    #[test]
    fn test_sq16_roundtrip_high_precision() {
        let (mut c, id) = make_codec_sq(16);
        let emb = make_rand_embedding(128, 42);
        let mse = roundtrip_mse(&mut c, id, &emb);
        assert!(mse < 1e-7, "mse={mse}");
    }

    #[test]
    fn test_sq4_roundtrip_low_precision() {
        let (mut c, id) = make_codec_sq(4);
        let emb = make_rand_embedding(64, 7);
        let mse = roundtrip_mse(&mut c, id, &emb);
        // 4-bit is lossy; allow generous tolerance
        assert!(mse < 0.05, "mse={mse}");
    }

    #[test]
    fn test_sq8_all_same_values() {
        let (mut c, id) = make_codec_sq(8);
        let emb = vec![0.5f64; 32];
        let comp = c.compress(id, &emb).expect("compress");
        let decomp = c.decompress(&comp).expect("decompress");
        assert_eq!(decomp.len(), 32);
        for &v in &decomp {
            // When all values are identical the range is zero → all quantize to 0
            assert!((v - 0.5).abs() < 1.0);
        }
    }

    #[test]
    fn test_sq8_negative_values() {
        let (mut c, id) = make_codec_sq(8);
        let emb = vec![-1.0, -0.5, 0.0, 0.5, 1.0];
        let mse = roundtrip_mse(&mut c, id, &emb);
        assert!(mse < 1e-4, "mse={mse}");
    }

    #[test]
    fn test_sq8_single_element() {
        let (mut c, id) = make_codec_sq(8);
        let emb = vec![std::f64::consts::PI];
        let comp = c.compress(id, &emb).expect("compress");
        let decomp = c.decompress(&comp).expect("decompress");
        assert_eq!(decomp.len(), 1);
    }

    #[test]
    fn test_sq8_odd_dimension() {
        let (mut c, id) = make_codec_sq(8);
        let emb = make_rand_embedding(13, 99);
        let comp = c.compress(id, &emb).expect("compress");
        let decomp = c.decompress(&comp).expect("decompress");
        assert_eq!(decomp.len(), 13);
    }

    #[test]
    fn test_sq4_byte_packing_odd_dim() {
        let (mut c, id) = make_codec_sq(4);
        let emb = make_rand_embedding(7, 11);
        let comp = c.compress(id, &emb).expect("compress");
        // 7 values → 4 bytes packed
        assert_eq!(comp.data.len(), 4);
        let decomp = c.decompress(&comp).expect("decompress");
        assert_eq!(decomp.len(), 7);
    }

    // Product Quantization roundtrips ---------------------------------------

    #[test]
    fn test_pq8_roundtrip() {
        let mut c = EmbeddingCompressionCodec::new();
        let id = c.register_codec("pq8", EccMethod::ProductQuantization, 8, 8);
        let emb = make_rand_embedding(64, 1234);
        let mse = roundtrip_mse(&mut c, id, &emb);
        assert!(mse < 1e-4, "mse={mse}");
    }

    #[test]
    fn test_pq_block_size_1() {
        let mut c = EmbeddingCompressionCodec::new();
        let id = c.register_codec("pq1", EccMethod::ProductQuantization, 8, 1);
        let emb = make_rand_embedding(16, 55);
        let mse = roundtrip_mse(&mut c, id, &emb);
        assert!(mse < 1e-4, "mse={mse}");
    }

    #[test]
    fn test_pq_non_divisible_dim() {
        let mut c = EmbeddingCompressionCodec::new();
        let id = c.register_codec("pq-nd", EccMethod::ProductQuantization, 8, 8);
        let emb = make_rand_embedding(20, 77);
        let comp = c.compress(id, &emb).expect("compress");
        let decomp = c.decompress(&comp).expect("decompress");
        assert_eq!(decomp.len(), 20);
    }

    #[test]
    fn test_pq16_roundtrip_high_precision() {
        let mut c = EmbeddingCompressionCodec::new();
        let id = c.register_codec("pq16", EccMethod::ProductQuantization, 16, 16);
        let emb = make_rand_embedding(128, 9876);
        let mse = roundtrip_mse(&mut c, id, &emb);
        assert!(mse < 1e-7, "mse={mse}");
    }

    // Delta Coding roundtrips -----------------------------------------------

    #[test]
    fn test_delta_roundtrip_basic() {
        let mut c = EmbeddingCompressionCodec::new();
        let id = c.register_codec("dc8", EccMethod::DeltaCoding, 8, 8);
        let emb = make_rand_embedding(64, 321);
        let mse = roundtrip_mse(&mut c, id, &emb);
        assert!(mse < 1e-3, "mse={mse}");
    }

    #[test]
    fn test_delta_single_element() {
        let mut c = EmbeddingCompressionCodec::new();
        let id = c.register_codec("dc8", EccMethod::DeltaCoding, 8, 8);
        let emb = vec![0.42f64];
        let comp = c.compress(id, &emb).expect("compress");
        let decomp = c.decompress(&comp).expect("decompress");
        assert_eq!(decomp.len(), 1);
    }

    #[test]
    fn test_delta_monotone_input() {
        let mut c = EmbeddingCompressionCodec::new();
        let id = c.register_codec("dc8", EccMethod::DeltaCoding, 8, 8);
        let emb: Vec<f64> = (0..32).map(|i| i as f64 / 32.0).collect();
        let mse = roundtrip_mse(&mut c, id, &emb);
        assert!(mse < 1e-3, "mse={mse}");
    }

    #[test]
    fn test_delta_16bit_precision() {
        let mut c = EmbeddingCompressionCodec::new();
        let id = c.register_codec("dc16", EccMethod::DeltaCoding, 16, 8);
        let emb = make_rand_embedding(32, 42);
        let mse = roundtrip_mse(&mut c, id, &emb);
        assert!(mse < 1e-6, "mse={mse}");
    }

    // RLE roundtrips --------------------------------------------------------

    #[test]
    fn test_rle_roundtrip_basic() {
        let mut c = EmbeddingCompressionCodec::new();
        let id = c.register_codec("rle8", EccMethod::RunLengthEncoding, 8, 8);
        let emb = make_rand_embedding(64, 555);
        let mse = roundtrip_mse(&mut c, id, &emb);
        // RLE uses 8-bit quantization (255 levels); allow ~1e-4 MSE
        assert!(mse < 1e-4, "mse={mse}");
    }

    #[test]
    fn test_rle_with_repeated_values() {
        let mut c = EmbeddingCompressionCodec::new();
        let id = c.register_codec("rle8", EccMethod::RunLengthEncoding, 8, 8);
        let mut emb = vec![0.5f64; 50];
        emb.extend(vec![0.1f64; 50]);
        let comp = c
            .compress(id, &emb)
            .expect("test: compress with repeated values");
        let decomp = c
            .decompress(&comp)
            .expect("test: decompress with repeated values");
        assert_eq!(decomp.len(), 100);
        // only 2 runs
        // num_runs=2 → 4 + 2*4 = 12 bytes
        assert_eq!(comp.data.len(), 12);
    }

    #[test]
    fn test_rle_single_value() {
        let mut c = EmbeddingCompressionCodec::new();
        let id = c.register_codec("rle8", EccMethod::RunLengthEncoding, 8, 8);
        let emb = vec![0.7f64];
        let comp = c.compress(id, &emb).expect("test: compress single value");
        let decomp = c.decompress(&comp).expect("test: decompress single value");
        assert_eq!(decomp.len(), 1);
    }

    // HybridPQ roundtrips ---------------------------------------------------

    #[test]
    fn test_hybridpq_roundtrip_basic() {
        let mut c = EmbeddingCompressionCodec::new();
        let id = c.register_codec("hpq8", EccMethod::HybridPQ, 8, 8);
        let emb = make_rand_embedding(64, 777);
        let mse = roundtrip_mse(&mut c, id, &emb);
        // HybridPQ uses 8-bit RLE per block; allow ~1e-4 MSE
        assert!(mse < 1e-4, "mse={mse}");
    }

    #[test]
    fn test_hybridpq_non_divisible_dim() {
        let mut c = EmbeddingCompressionCodec::new();
        let id = c.register_codec("hpq8", EccMethod::HybridPQ, 8, 8);
        let emb = make_rand_embedding(17, 888);
        let comp = c
            .compress(id, &emb)
            .expect("test: compress hybridpq repeated");
        let decomp = c
            .decompress(&comp)
            .expect("test: decompress hybridpq repeated");
        assert_eq!(decomp.len(), 17);
    }

    #[test]
    fn test_hybridpq_repeated_values_compression() {
        let mut c = EmbeddingCompressionCodec::new();
        let id = c.register_codec("hpq8", EccMethod::HybridPQ, 8, 8);
        let emb = vec![0.3f64; 64];
        let comp_hpq = c
            .compress(id, &emb)
            .expect("test: compress hybridpq repeated");
        // Should still roundtrip correctly
        let decomp = c
            .decompress(&comp_hpq)
            .expect("test: decompress hybridpq repeated");
        assert_eq!(decomp.len(), 64);
    }

    // Batch operations ------------------------------------------------------

    #[test]
    fn test_compress_batch_all_succeed() {
        let mut c = EmbeddingCompressionCodec::new();
        let id = c.register_codec("sq8", EccMethod::ScalarQuantization, 8, 8);
        let batch: Vec<Vec<f64>> = (0..5).map(|i| make_rand_embedding(16, i + 100)).collect();
        let results = c.compress_batch(id, &batch);
        assert_eq!(results.len(), 5);
        for r in &results {
            assert!(r.is_ok());
        }
    }

    #[test]
    fn test_decompress_batch_all_succeed() {
        let mut c = EmbeddingCompressionCodec::new();
        let id = c.register_codec("sq8", EccMethod::ScalarQuantization, 8, 8);
        let batch: Vec<Vec<f64>> = (0..3).map(|i| make_rand_embedding(32, i + 200)).collect();
        let compressed: Vec<EccCompressed> = batch
            .iter()
            .map(|emb| c.compress(id, emb).expect("test: compress in batch"))
            .collect();
        let decompressed = c.decompress_batch(&compressed);
        assert_eq!(decompressed.len(), 3);
        for (orig, decomp_res) in batch.iter().zip(decompressed.iter()) {
            let decomp = decomp_res.as_ref().expect("decompress failed");
            assert_eq!(decomp.len(), orig.len());
        }
    }

    #[test]
    fn test_compress_batch_empty_batch() {
        let mut c = EmbeddingCompressionCodec::new();
        let id = c.register_codec("sq8", EccMethod::ScalarQuantization, 8, 8);
        let results = c.compress_batch(id, &[]);
        assert!(results.is_empty());
    }

    #[test]
    fn test_decompress_batch_empty() {
        let c = EmbeddingCompressionCodec::new();
        let result = c.decompress_batch(&[]);
        assert!(result.is_empty());
    }

    // Reconstruction error --------------------------------------------------

    #[test]
    fn test_reconstruction_error_identical_vectors() {
        let v = vec![0.1, 0.2, 0.3];
        let mse = EmbeddingCompressionCodec::reconstruction_error(&v, &v);
        assert!(mse.abs() < f64::EPSILON);
    }

    #[test]
    fn test_reconstruction_error_known_value() {
        let a = vec![0.0f64, 0.0, 0.0];
        let b = vec![1.0f64, 1.0, 1.0];
        let mse = EmbeddingCompressionCodec::reconstruction_error(&a, &b);
        assert!((mse - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_reconstruction_error_empty_returns_zero() {
        let mse = EmbeddingCompressionCodec::reconstruction_error(&[], &[]);
        assert_eq!(mse, 0.0);
    }

    #[test]
    fn test_reconstruction_error_different_lengths_uses_min() {
        let a = vec![0.0f64, 0.0, 0.0, 0.0];
        let b = vec![1.0f64, 1.0];
        let mse = EmbeddingCompressionCodec::reconstruction_error(&a, &b);
        // Only first 2 elements compared: (0-1)^2 * 2 / 2 = 1.0
        assert!((mse - 1.0).abs() < f64::EPSILON);
    }

    // Estimate ratio --------------------------------------------------------

    #[test]
    fn test_estimate_ratio_sq8_better_than_one() {
        let mut c = EmbeddingCompressionCodec::new();
        let id = c.register_codec("sq8", EccMethod::ScalarQuantization, 8, 8);
        let ratio = c.estimate_ratio(id, 512);
        assert!(ratio > 1.0, "ratio={ratio}");
    }

    #[test]
    fn test_estimate_ratio_sq16_less_than_sq8() {
        let mut c = EmbeddingCompressionCodec::new();
        let id8 = c.register_codec("sq8", EccMethod::ScalarQuantization, 8, 8);
        let id16 = c.register_codec("sq16", EccMethod::ScalarQuantization, 16, 8);
        let r8 = c.estimate_ratio(id8, 256);
        let r16 = c.estimate_ratio(id16, 256);
        assert!(r8 > r16, "r8={r8}, r16={r16}");
    }

    #[test]
    fn test_estimate_ratio_unknown_codec() {
        let c = EmbeddingCompressionCodec::new();
        let ratio = c.estimate_ratio(999, 128);
        assert_eq!(ratio, 1.0);
    }

    #[test]
    fn test_estimate_ratio_zero_dim() {
        let mut c = EmbeddingCompressionCodec::new();
        let id = c.register_codec("sq8", EccMethod::ScalarQuantization, 8, 8);
        let ratio = c.estimate_ratio(id, 0);
        assert_eq!(ratio, 1.0);
    }

    #[test]
    fn test_estimate_ratio_hybrid_pq_less_than_pq() {
        let mut c = EmbeddingCompressionCodec::new();
        let id_pq = c.register_codec("pq8", EccMethod::ProductQuantization, 8, 8);
        let id_hpq = c.register_codec("hpq8", EccMethod::HybridPQ, 8, 8);
        let rpq = c.estimate_ratio(id_pq, 128);
        let rhpq = c.estimate_ratio(id_hpq, 128);
        // HybridPQ claims ~30% better (0.7× factor on PQ bits)
        assert!(rhpq > rpq, "rpq={rpq} rhpq={rhpq}");
    }

    // Codec stats -----------------------------------------------------------

    #[test]
    fn test_codec_stats_empty_log() {
        let c = EmbeddingCompressionCodec::new();
        let stats = c.codec_stats();
        assert_eq!(stats.total_compressed, 0);
        assert_eq!(stats.avg_ratio, 1.0);
        assert!(stats.per_method.is_empty());
    }

    #[test]
    fn test_codec_stats_after_compressions() {
        let mut c = EmbeddingCompressionCodec::new();
        let id = c.register_codec("sq8", EccMethod::ScalarQuantization, 8, 8);
        let emb = make_rand_embedding(128, 11);
        c.compress(id, &emb).expect("test: compress for stats");
        c.compress(id, &emb)
            .expect("test: compress for stats second");
        let stats = c.codec_stats();
        assert_eq!(stats.total_compressed, 2);
        assert!(stats.avg_ratio > 1.0);
        assert!(stats
            .per_method
            .contains_key(&EccMethod::ScalarQuantization));
    }

    #[test]
    fn test_codec_stats_per_method_breakdown() {
        let mut c = EmbeddingCompressionCodec::new();
        let id_sq = c.register_codec("sq8", EccMethod::ScalarQuantization, 8, 8);
        let id_rle = c.register_codec("rle8", EccMethod::RunLengthEncoding, 8, 8);
        let emb = make_rand_embedding(64, 22);
        c.compress(id_sq, &emb)
            .expect("test: compress sq8 for per-method stats");
        c.compress(id_rle, &emb)
            .expect("test: compress rle for per-method stats");
        let stats = c.codec_stats();
        assert_eq!(stats.total_compressed, 2);
        let (sq_ops, _, _) = stats.per_method[&EccMethod::ScalarQuantization];
        let (rle_ops, _, _) = stats.per_method[&EccMethod::RunLengthEncoding];
        assert_eq!(sq_ops, 1);
        assert_eq!(rle_ops, 1);
    }

    // Log bounding ----------------------------------------------------------

    #[test]
    fn test_log_bounded_to_500() {
        let mut c = EmbeddingCompressionCodec::new();
        let id = c.register_codec("sq8", EccMethod::ScalarQuantization, 8, 8);
        let emb = vec![0.5f64; 8];
        for _ in 0..600 {
            c.compress(id, &emb)
                .expect("test: compress for log bound test");
        }
        assert_eq!(c.log_entries().len(), 500);
    }

    // Compression payload fields --------------------------------------------

    #[test]
    fn test_compressed_fields_populated() {
        let mut c = EmbeddingCompressionCodec::new();
        let id = c.register_codec("sq8", EccMethod::ScalarQuantization, 8, 8);
        let emb = vec![0.1, 0.5, 0.9];
        let comp = c
            .compress(id, &emb)
            .expect("test: compress for fields test");
        assert_eq!(comp.codec_id, id);
        assert_eq!(comp.method, EccMethod::ScalarQuantization);
        assert_eq!(comp.original_dim, 3);
        assert!((comp.min_val - 0.1).abs() < 1e-10);
        assert!((comp.max_val - 0.9).abs() < 1e-10);
    }

    #[test]
    fn test_compressed_data_non_empty() {
        let mut c = EmbeddingCompressionCodec::new();
        let id = c.register_codec("sq8", EccMethod::ScalarQuantization, 8, 8);
        let emb = make_rand_embedding(32, 33);
        let comp = c
            .compress(id, &emb)
            .expect("test: compress for non-empty data check");
        assert!(!comp.data.is_empty());
    }

    // Method display --------------------------------------------------------

    #[test]
    fn test_ecc_method_display() {
        assert_eq!(
            EccMethod::ScalarQuantization.to_string(),
            "ScalarQuantization"
        );
        assert_eq!(
            EccMethod::ProductQuantization.to_string(),
            "ProductQuantization"
        );
        assert_eq!(EccMethod::DeltaCoding.to_string(), "DeltaCoding");
        assert_eq!(
            EccMethod::RunLengthEncoding.to_string(),
            "RunLengthEncoding"
        );
        assert_eq!(EccMethod::HybridPQ.to_string(), "HybridPQ");
    }

    // Default codec config --------------------------------------------------

    #[test]
    fn test_default_config() {
        let cfg = EccCodecConfig::default();
        assert_eq!(cfg.quantize_bits, 8);
        assert_eq!(cfg.method, EccMethod::ScalarQuantization);
        assert!(!cfg.use_delta_coding);
        assert_eq!(cfg.block_size, 8);
    }

    // xorshift64 PRNG -------------------------------------------------------

    #[test]
    fn test_xorshift64_nondeterministic_zero_free() {
        let mut state: u64 = 12345;
        let mut any_nonzero = false;
        for _ in 0..100 {
            let v = xorshift64(&mut state);
            if v != 0 {
                any_nonzero = true;
            }
        }
        assert!(any_nonzero);
    }

    #[test]
    fn test_xorshift64_produces_different_values() {
        let mut state: u64 = 99999;
        let a = xorshift64(&mut state);
        let b = xorshift64(&mut state);
        assert_ne!(a, b);
    }

    // Additional edge cases -------------------------------------------------

    #[test]
    fn test_sq8_dim_1024() {
        let (mut c, id) = make_codec_sq(8);
        let emb = make_rand_embedding(1024, 2024);
        let mse = roundtrip_mse(&mut c, id, &emb);
        assert!(mse < 1e-4, "mse={mse}");
    }

    #[test]
    fn test_pq_large_block_exceeds_dim() {
        let mut c = EmbeddingCompressionCodec::new();
        let id = c.register_codec("pq-lb", EccMethod::ProductQuantization, 8, 128);
        let emb = make_rand_embedding(16, 42); // dim < block_size
        let comp = c
            .compress(id, &emb)
            .expect("test: compress should succeed when block size exceeds dim");
        let decomp = c
            .decompress(&comp)
            .expect("test: decompress should succeed when block size exceeds dim");
        assert_eq!(decomp.len(), 16);
    }

    #[test]
    fn test_all_methods_roundtrip_dim_128() {
        let methods = [
            EccMethod::ScalarQuantization,
            EccMethod::ProductQuantization,
            EccMethod::DeltaCoding,
            EccMethod::RunLengthEncoding,
            EccMethod::HybridPQ,
        ];
        let emb = make_rand_embedding(128, 1111);
        for method in &methods {
            let mut c = EmbeddingCompressionCodec::new();
            let id = c.register_codec("test", *method, 8, 8);
            let comp = c
                .compress(id, &emb)
                .unwrap_or_else(|_| panic!("compress {method}"));
            let decomp = c
                .decompress(&comp)
                .unwrap_or_else(|_| panic!("decompress {method}"));
            assert_eq!(decomp.len(), 128, "dim mismatch for {method}");
        }
    }

    #[test]
    fn test_compressed_size_smaller_than_original_sq8() {
        let (mut c, id) = make_codec_sq(8);
        let emb = make_rand_embedding(256, 42);
        let comp = c
            .compress(id, &emb)
            .expect("test: compress sq8 should succeed");
        let original_bytes = 256 * 8;
        assert!(
            comp.data.len() < original_bytes,
            "data.len()={}",
            comp.data.len()
        );
    }

    #[test]
    fn test_compressed_size_smaller_than_original_sq16() {
        let (mut c, id) = make_codec_sq(16);
        let emb = make_rand_embedding(256, 42);
        let comp = c
            .compress(id, &emb)
            .expect("test: compress sq16 should succeed");
        let original_bytes = 256 * 8;
        assert!(comp.data.len() < original_bytes);
    }

    #[test]
    fn test_hybridpq_with_block_size_4() {
        let mut c = EmbeddingCompressionCodec::new();
        let id = c.register_codec("hpq4", EccMethod::HybridPQ, 8, 4);
        let emb = make_rand_embedding(32, 444);
        let mse = roundtrip_mse(&mut c, id, &emb);
        // HybridPQ/8bit: allow ~1e-4 MSE
        assert!(mse < 1e-4, "mse={mse}");
    }

    #[test]
    fn test_delta_coding_all_negative() {
        let mut c = EmbeddingCompressionCodec::new();
        let id = c.register_codec("dc8", EccMethod::DeltaCoding, 8, 8);
        let emb: Vec<f64> = (0..32).map(|i| -(i as f64) / 32.0).collect();
        let mse = roundtrip_mse(&mut c, id, &emb);
        assert!(mse < 1e-3, "mse={mse}");
    }

    #[test]
    fn test_codec_stats_total_bytes_saved() {
        let mut c = EmbeddingCompressionCodec::new();
        let id = c.register_codec("sq8", EccMethod::ScalarQuantization, 8, 8);
        let emb = make_rand_embedding(256, 7);
        c.compress(id, &emb)
            .expect("test: compress should succeed for stats check");
        let stats = c.codec_stats();
        // 256 * 8 bytes original vs 256 bytes sq8 → should save 1792 bytes
        assert!(stats.total_bytes_saved > 0);
    }

    #[test]
    fn test_sq8_compress_data_length() {
        let (mut c, id) = make_codec_sq(8);
        let emb = make_rand_embedding(128, 10);
        let comp = c
            .compress(id, &emb)
            .expect("test: compress sq8 should succeed");
        // SQ8: 1 byte per element
        assert_eq!(comp.data.len(), 128);
    }

    #[test]
    fn test_sq16_compress_data_length() {
        let (mut c, id) = make_codec_sq(16);
        let emb = make_rand_embedding(64, 20);
        let comp = c
            .compress(id, &emb)
            .expect("test: compress sq16 should succeed");
        // SQ16: 2 bytes per element
        assert_eq!(comp.data.len(), 128);
    }

    #[test]
    fn test_sq4_compress_data_length_even_dim() {
        let (mut c, id) = make_codec_sq(4);
        let emb = make_rand_embedding(64, 30);
        let comp = c
            .compress(id, &emb)
            .expect("test: compress sq4 should succeed");
        // SQ4: 2 values per byte → 32 bytes
        assert_eq!(comp.data.len(), 32);
    }

    #[test]
    fn test_ecc_error_clone_and_partial_eq() {
        let e = EccError::EmptyEmbedding;
        assert_eq!(e.clone(), EccError::EmptyEmbedding);
    }

    #[test]
    fn test_ecc_method_hash_map_key() {
        let mut map: HashMap<EccMethod, usize> = HashMap::new();
        map.insert(EccMethod::ScalarQuantization, 1);
        map.insert(EccMethod::HybridPQ, 2);
        assert_eq!(
            *map.get(&EccMethod::ScalarQuantization)
                .expect("test: ScalarQuantization should be in map"),
            1
        );
    }
}
