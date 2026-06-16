//! Compression support for block data
//!
//! This module provides compression and decompression capabilities for block data,
//! supporting multiple algorithms for different use cases.
//!
//! # Supported Algorithms
//!
//! - **None**: No compression (passthrough)
//! - **Zstd**: Zstandard compression (high ratio, good speed)
//! - **Lz4**: LZ4 compression (very fast, moderate ratio)
//!
//! # Example
//!
//! ```rust
//! use ipfrs_core::compression::{CompressionAlgorithm, compress, decompress};
//! use bytes::Bytes;
//!
//! let data = Bytes::from_static(b"Hello, World! This is some data to compress.");
//! let level = 3;
//!
//! // Compress with Zstd
//! let compressed = compress(&data, CompressionAlgorithm::Zstd, level).unwrap();
//! println!("Original: {} bytes, Compressed: {} bytes", data.len(), compressed.len());
//!
//! // Decompress
//! let decompressed = decompress(&compressed, CompressionAlgorithm::Zstd).unwrap();
//! assert_eq!(data, decompressed);
//! ```

use crate::error::{Error, Result};
use bytes::Bytes;

/// Compression algorithms supported by ipfrs-core
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum CompressionAlgorithm {
    /// No compression (passthrough)
    #[default]
    None,
    /// Zstandard compression (high ratio, good speed)
    Zstd,
    /// LZ4 compression (very fast, moderate ratio)
    Lz4,
}

impl CompressionAlgorithm {
    /// Get the name of the compression algorithm
    #[inline]
    pub fn name(&self) -> &'static str {
        match self {
            CompressionAlgorithm::None => "none",
            CompressionAlgorithm::Zstd => "zstd",
            CompressionAlgorithm::Lz4 => "lz4",
        }
    }

    /// Check if this algorithm actually compresses data
    #[inline]
    pub fn is_compressed(&self) -> bool {
        !matches!(self, CompressionAlgorithm::None)
    }

    /// Get all available compression algorithms
    pub fn all() -> &'static [CompressionAlgorithm] {
        &[
            CompressionAlgorithm::None,
            CompressionAlgorithm::Zstd,
            CompressionAlgorithm::Lz4,
        ]
    }
}

impl std::fmt::Display for CompressionAlgorithm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Compress data using the specified algorithm and level
///
/// # Arguments
///
/// * `data` - The data to compress
/// * `algorithm` - The compression algorithm to use
/// * `level` - Compression level (0-9, where 0 is fastest and 9 is best compression)
///
/// # Returns
///
/// Compressed data as `Bytes`, or the original data if algorithm is `None`
///
/// # Example
///
/// ```rust
/// use ipfrs_core::compression::{CompressionAlgorithm, compress};
/// use bytes::Bytes;
///
/// let data = Bytes::from_static(b"Hello, World!");
/// let compressed = compress(&data, CompressionAlgorithm::Zstd, 3).unwrap();
/// ```
pub fn compress(data: &Bytes, algorithm: CompressionAlgorithm, level: u8) -> Result<Bytes> {
    if level > 9 {
        return Err(Error::InvalidInput(format!(
            "Invalid compression level {}, must be 0-9",
            level
        )));
    }

    match algorithm {
        CompressionAlgorithm::None => Ok(data.clone()),
        CompressionAlgorithm::Zstd => compress_zstd(data, level),
        CompressionAlgorithm::Lz4 => compress_lz4(data, level),
    }
}

/// Decompress data using the specified algorithm
///
/// # Arguments
///
/// * `data` - The compressed data
/// * `algorithm` - The compression algorithm that was used
///
/// # Returns
///
/// Decompressed data as `Bytes`
///
/// # Example
///
/// ```rust
/// use ipfrs_core::compression::{CompressionAlgorithm, compress, decompress};
/// use bytes::Bytes;
///
/// let data = Bytes::from_static(b"Hello, World!");
/// let compressed = compress(&data, CompressionAlgorithm::Lz4, 3).unwrap();
/// let decompressed = decompress(&compressed, CompressionAlgorithm::Lz4).unwrap();
/// assert_eq!(data, decompressed);
/// ```
pub fn decompress(data: &Bytes, algorithm: CompressionAlgorithm) -> Result<Bytes> {
    match algorithm {
        CompressionAlgorithm::None => Ok(data.clone()),
        CompressionAlgorithm::Zstd => decompress_zstd(data),
        CompressionAlgorithm::Lz4 => decompress_lz4(data),
    }
}

/// Compress data using Zstd
fn compress_zstd(data: &Bytes, level: u8) -> Result<Bytes> {
    // Convert level 0-9 to zstd level range (-7 to 22)
    // 0 -> 1 (fastest), 9 -> 22 (best compression)
    let zstd_level = if level == 0 {
        1
    } else {
        1 + (level as i32 * 21 / 9)
    };

    let compressed = oxiarc_zstd::compress_with_level(data, zstd_level)
        .map_err(|e| Error::Internal(format!("Zstd compression failed: {}", e)))?;
    Ok(Bytes::from(compressed))
}

/// Decompress data using Zstd
fn decompress_zstd(data: &Bytes) -> Result<Bytes> {
    // OxiARC zstd frames are self-describing, so no output capacity hint is needed.
    let decompressed = oxiarc_zstd::decompress(data)
        .map_err(|e| Error::Internal(format!("Zstd decompression failed: {}", e)))?;
    Ok(Bytes::from(decompressed))
}

/// Compress data using LZ4
///
/// The output is a self-describing block: a 4-byte little-endian original size
/// header followed by an OxiARC LZ4 frame. The size header lets [`decompress_lz4`]
/// allocate an accurate output buffer without an external length hint.
fn compress_lz4(data: &Bytes, _level: u8) -> Result<Bytes> {
    // OxiARC LZ4 does not expose compression levels in the simple frame API.
    let payload = oxiarc_lz4::compress(data)
        .map_err(|e| Error::Internal(format!("LZ4 compression failed: {}", e)))?;
    let orig_len = data.len() as u32;
    let mut out = Vec::with_capacity(4 + payload.len());
    out.extend_from_slice(&orig_len.to_le_bytes());
    out.extend_from_slice(&payload);
    Ok(Bytes::from(out))
}

/// Decompress data using LZ4
fn decompress_lz4(data: &Bytes) -> Result<Bytes> {
    if data.len() < 4 {
        return Err(Error::Internal(
            "LZ4 block too short (missing original size header)".to_string(),
        ));
    }
    let orig_size = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
    // Generous upper bound: 2x original size avoids truncation for edge cases.
    let max_output = orig_size.saturating_mul(2).max(orig_size + 64);
    let decompressed = oxiarc_lz4::decompress(&data[4..], max_output)
        .map_err(|e| Error::Internal(format!("LZ4 decompression failed: {}", e)))?;
    Ok(Bytes::from(decompressed))
}

/// Estimate compression ratio for given data
///
/// Returns a value between 0.0 and 1.0, where lower values indicate better compression.
/// A value of 0.5 means the data compressed to 50% of its original size.
///
/// # Example
///
/// ```rust
/// use ipfrs_core::compression::{CompressionAlgorithm, compression_ratio};
/// use bytes::Bytes;
///
/// let data = Bytes::from_static(b"Hello, World! Hello, World! Hello, World!");
/// let ratio = compression_ratio(&data, CompressionAlgorithm::Zstd, 5).unwrap();
/// assert!(ratio < 1.0); // Should compress well due to repetition
/// ```
pub fn compression_ratio(data: &Bytes, algorithm: CompressionAlgorithm, level: u8) -> Result<f64> {
    if data.is_empty() {
        return Ok(0.0);
    }

    let compressed = compress(data, algorithm, level)?;
    Ok(compressed.len() as f64 / data.len() as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compression_none() {
        let data = Bytes::from_static(b"Hello, World!");
        let compressed = compress(&data, CompressionAlgorithm::None, 5).unwrap();
        assert_eq!(data, compressed);

        let decompressed = decompress(&compressed, CompressionAlgorithm::None).unwrap();
        assert_eq!(data, decompressed);
    }

    #[test]
    fn test_compression_zstd() {
        // Use highly compressible data (repetitive pattern)
        let data = Bytes::from("Hello, World! ".repeat(100));
        let compressed = compress(&data, CompressionAlgorithm::Zstd, 5).unwrap();

        // Should compress well with repetitive data
        assert!(compressed.len() < data.len());

        let decompressed = decompress(&compressed, CompressionAlgorithm::Zstd).unwrap();
        assert_eq!(data, decompressed);
    }

    #[test]
    fn test_compression_lz4() {
        // Use highly compressible data (repetitive pattern)
        let data = Bytes::from("Hello, World! ".repeat(100));
        let compressed = compress(&data, CompressionAlgorithm::Lz4, 5).unwrap();

        // Should compress well with repetitive data
        assert!(compressed.len() < data.len());

        let decompressed = decompress(&compressed, CompressionAlgorithm::Lz4).unwrap();
        assert_eq!(data, decompressed);
    }

    #[test]
    fn test_compression_levels() {
        let data = Bytes::from_static(b"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"); // Highly compressible

        // Level 0 (fastest)
        let compressed_0 = compress(&data, CompressionAlgorithm::Zstd, 0).unwrap();

        // Level 9 (best compression)
        let compressed_9 = compress(&data, CompressionAlgorithm::Zstd, 9).unwrap();

        // Both should decompress correctly
        let decompressed_0 = decompress(&compressed_0, CompressionAlgorithm::Zstd).unwrap();
        let decompressed_9 = decompress(&compressed_9, CompressionAlgorithm::Zstd).unwrap();

        assert_eq!(data, decompressed_0);
        assert_eq!(data, decompressed_9);

        // Higher level should generally compress better
        assert!(compressed_9.len() <= compressed_0.len());
    }

    #[test]
    fn test_invalid_compression_level() {
        let data = Bytes::from_static(b"Hello");
        let result = compress(&data, CompressionAlgorithm::Zstd, 10);
        assert!(result.is_err());
    }

    #[test]
    fn test_compression_ratio() {
        let data = Bytes::from("a".repeat(1000));
        let ratio = compression_ratio(&data, CompressionAlgorithm::Zstd, 5).unwrap();
        assert!(ratio < 0.1); // Should compress very well
    }

    #[test]
    fn test_compression_algorithm_name() {
        assert_eq!(CompressionAlgorithm::None.name(), "none");
        assert_eq!(CompressionAlgorithm::Zstd.name(), "zstd");
        assert_eq!(CompressionAlgorithm::Lz4.name(), "lz4");
    }

    #[test]
    fn test_compression_algorithm_is_compressed() {
        assert!(!CompressionAlgorithm::None.is_compressed());
        assert!(CompressionAlgorithm::Zstd.is_compressed());
        assert!(CompressionAlgorithm::Lz4.is_compressed());
    }

    #[test]
    fn test_compression_algorithm_all() {
        let all = CompressionAlgorithm::all();
        assert_eq!(all.len(), 3);
        assert!(all.contains(&CompressionAlgorithm::None));
        assert!(all.contains(&CompressionAlgorithm::Zstd));
        assert!(all.contains(&CompressionAlgorithm::Lz4));
    }

    #[test]
    fn test_empty_data() {
        let data = Bytes::new();
        let compressed = compress(&data, CompressionAlgorithm::Zstd, 5).unwrap();
        let decompressed = decompress(&compressed, CompressionAlgorithm::Zstd).unwrap();
        assert_eq!(data, decompressed);
    }

    #[test]
    fn test_large_data() {
        let data = Bytes::from(vec![42u8; 1_000_000]); // 1MB of same data
        let compressed = compress(&data, CompressionAlgorithm::Zstd, 5).unwrap();

        // Should compress very well (highly redundant data)
        assert!(compressed.len() < data.len() / 100);

        let decompressed = decompress(&compressed, CompressionAlgorithm::Zstd).unwrap();
        assert_eq!(data, decompressed);
    }

    #[test]
    fn test_algorithm_display() {
        assert_eq!(CompressionAlgorithm::None.to_string(), "none");
        assert_eq!(CompressionAlgorithm::Zstd.to_string(), "zstd");
        assert_eq!(CompressionAlgorithm::Lz4.to_string(), "lz4");
    }

    #[test]
    fn test_all_algorithms_roundtrip() {
        let data = Bytes::from_static(b"The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs.");

        for algorithm in CompressionAlgorithm::all() {
            for level in 0..=9 {
                let compressed = compress(&data, *algorithm, level).unwrap();
                let decompressed = decompress(&compressed, *algorithm).unwrap();
                assert_eq!(
                    data, decompressed,
                    "Failed for {:?} at level {}",
                    algorithm, level
                );
            }
        }
    }
}
