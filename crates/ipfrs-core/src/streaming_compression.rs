//! Streaming compression and decompression support
//!
//! This module provides streaming compression and decompression capabilities for large data
//! that cannot or should not be loaded entirely into memory. It integrates with the async
//! streaming infrastructure and supports all compression algorithms.
//!
//! # Features
//!
//! - **Async streaming compression** - Compress data on-the-fly as it's streamed
//! - **Async streaming decompression** - Decompress data on-the-fly as it's streamed
//! - **All algorithms supported** - Zstd, LZ4, and passthrough (None)
//! - **Configurable buffer sizes** - Tune memory usage vs performance
//! - **Statistics tracking** - Monitor bytes processed and compression ratios
//!
//! # Example
//!
//! ```rust
//! use ipfrs_core::streaming_compression::{CompressingStream, CompressionAlgorithm};
//! use tokio::io::AsyncReadExt;
//! use bytes::Bytes;
//!
//! # #[tokio::main]
//! # async fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create data to compress
//! let data = Bytes::from(b"Hello, world! ".repeat(1000));
//! let cursor = std::io::Cursor::new(data.to_vec());
//!
//! // Create a compressing stream
//! let mut stream = CompressingStream::new(cursor, CompressionAlgorithm::Zstd, 3)?;
//!
//! // Read compressed data
//! let mut compressed = Vec::new();
//! stream.read_to_end(&mut compressed).await?;
//!
//! println!("Original: {} bytes", data.len());
//! println!("Compressed: {} bytes", compressed.len());
//! println!("Ratio: {:.2}%", (compressed.len() as f64 / data.len() as f64) * 100.0);
//! # Ok(())
//! # }
//! ```

use crate::error::{Error, Result};
use bytes::{Bytes, BytesMut};
use std::io::Cursor;
use std::pin::Pin;
use std::task::{Context, Poll};
use tokio::io::{AsyncRead, ReadBuf};

// Re-export CompressionAlgorithm for convenience in doc tests
pub use crate::compression::CompressionAlgorithm;

/// Buffer size for streaming operations (64KB)
const DEFAULT_BUFFER_SIZE: usize = 64 * 1024;

/// A streaming compressor that compresses data on-the-fly
///
/// This struct wraps an `AsyncRead` source and compresses the data as it's read,
/// providing an efficient way to compress large files without loading them entirely
/// into memory.
///
/// # Example
///
/// ```rust
/// use ipfrs_core::streaming_compression::{CompressingStream, CompressionAlgorithm};
/// use tokio::io::AsyncReadExt;
/// use bytes::Bytes;
///
/// # #[tokio::main]
/// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let data = b"Hello, streaming compression!".repeat(100);
/// let cursor = std::io::Cursor::new(data.clone());
///
/// let mut stream = CompressingStream::new(cursor, CompressionAlgorithm::Zstd, 5)?;
///
/// let mut compressed = Vec::new();
/// stream.read_to_end(&mut compressed).await?;
///
/// let stats = stream.stats();
/// println!("Compressed {} bytes to {} bytes", stats.bytes_read, stats.bytes_written);
/// # Ok(())
/// # }
/// ```
pub struct CompressingStream<R: AsyncRead + Unpin> {
    reader: R,
    algorithm: CompressionAlgorithm,
    level: u8,
    buffer: BytesMut,
    compressed_buffer: Cursor<Vec<u8>>,
    stats: StreamingStats,
    finished: bool,
    buffer_size: usize,
}

impl<R: AsyncRead + Unpin> CompressingStream<R> {
    /// Create a new compressing stream
    ///
    /// # Arguments
    ///
    /// * `reader` - The source to read uncompressed data from
    /// * `algorithm` - The compression algorithm to use
    /// * `level` - Compression level (0-9)
    ///
    /// # Returns
    ///
    /// A new `CompressingStream` instance
    pub fn new(reader: R, algorithm: CompressionAlgorithm, level: u8) -> Result<Self> {
        if level > 9 {
            return Err(Error::InvalidInput(format!(
                "compression level must be 0-9, got {}",
                level
            )));
        }

        Ok(Self {
            reader,
            algorithm,
            level,
            buffer: BytesMut::with_capacity(DEFAULT_BUFFER_SIZE),
            compressed_buffer: Cursor::new(Vec::new()),
            stats: StreamingStats::default(),
            finished: false,
            buffer_size: DEFAULT_BUFFER_SIZE,
        })
    }

    /// Create a new compressing stream with a custom buffer size
    ///
    /// # Arguments
    ///
    /// * `reader` - The source to read uncompressed data from
    /// * `algorithm` - The compression algorithm to use
    /// * `level` - Compression level (0-9)
    /// * `buffer_size` - Size of the internal buffer in bytes
    pub fn with_buffer_size(
        reader: R,
        algorithm: CompressionAlgorithm,
        level: u8,
        buffer_size: usize,
    ) -> Result<Self> {
        if level > 9 {
            return Err(Error::InvalidInput(format!(
                "compression level must be 0-9, got {}",
                level
            )));
        }

        Ok(Self {
            reader,
            algorithm,
            level,
            buffer: BytesMut::with_capacity(buffer_size),
            compressed_buffer: Cursor::new(Vec::new()),
            stats: StreamingStats::default(),
            finished: false,
            buffer_size,
        })
    }

    /// Get statistics about the compression operation
    pub fn stats(&self) -> &StreamingStats {
        &self.stats
    }
}

impl<R: AsyncRead + Unpin> AsyncRead for CompressingStream<R> {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<std::io::Result<()>> {
        // Try to read from compressed buffer first
        let pos = self.compressed_buffer.position() as usize;
        let available = self.compressed_buffer.get_ref().len() - pos;

        if available > 0 {
            let to_copy = available.min(buf.remaining());
            buf.put_slice(&self.compressed_buffer.get_ref()[pos..pos + to_copy]);
            self.compressed_buffer.set_position((pos + to_copy) as u64);
            return Poll::Ready(Ok(()));
        }

        // If finished and no more data, return EOF
        if self.finished {
            return Poll::Ready(Ok(()));
        }

        // Need to read more data from source
        // Get self as mut ref to avoid borrow checker issues
        let this = &mut *self;

        this.buffer.resize(this.buffer_size, 0);
        let mut read_buf = ReadBuf::new(&mut this.buffer[..]);

        match Pin::new(&mut this.reader).poll_read(cx, &mut read_buf) {
            Poll::Ready(Ok(())) => {
                let n = read_buf.filled().len();

                if n == 0 {
                    this.finished = true;
                    return Poll::Ready(Ok(()));
                }

                this.stats.bytes_read += n as u64;

                // Compress the data
                let data = Bytes::from(this.buffer[..n].to_vec());
                let compressed =
                    match crate::compression::compress(&data, this.algorithm, this.level) {
                        Ok(c) => c,
                        Err(e) => return Poll::Ready(Err(std::io::Error::other(e.to_string()))),
                    };

                this.stats.bytes_written += compressed.len() as u64;
                this.compressed_buffer = Cursor::new(compressed.to_vec());

                // Now read from the compressed buffer
                let pos = this.compressed_buffer.position() as usize;
                let available = this.compressed_buffer.get_ref().len() - pos;

                if available > 0 {
                    let to_copy = available.min(buf.remaining());
                    buf.put_slice(&this.compressed_buffer.get_ref()[pos..pos + to_copy]);
                    this.compressed_buffer.set_position((pos + to_copy) as u64);
                }

                Poll::Ready(Ok(()))
            }
            Poll::Ready(Err(e)) => Poll::Ready(Err(e)),
            Poll::Pending => Poll::Pending,
        }
    }
}

/// A streaming decompressor that decompresses data on-the-fly
///
/// This struct wraps an `AsyncRead` source and decompresses the data as it's read,
/// providing an efficient way to decompress large files without loading them entirely
/// into memory.
///
/// # Example
///
/// ```rust
/// use ipfrs_core::streaming_compression::{CompressingStream, DecompressingStream, CompressionAlgorithm};
/// use tokio::io::{AsyncReadExt, AsyncWriteExt};
/// use bytes::Bytes;
///
/// # #[tokio::main]
/// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
/// // First compress some data
/// let original = b"Hello, streaming decompression!".repeat(100);
/// let cursor = std::io::Cursor::new(original.clone());
/// let mut compressor = CompressingStream::new(cursor, CompressionAlgorithm::Zstd, 5)?;
///
/// let mut compressed = Vec::new();
/// compressor.read_to_end(&mut compressed).await?;
///
/// // Now decompress it
/// let cursor = std::io::Cursor::new(compressed);
/// let mut decompressor = DecompressingStream::new(cursor, CompressionAlgorithm::Zstd)?;
///
/// let mut decompressed = Vec::new();
/// decompressor.read_to_end(&mut decompressed).await?;
///
/// assert_eq!(original, decompressed);
/// # Ok(())
/// # }
/// ```
pub struct DecompressingStream<R: AsyncRead + Unpin> {
    reader: R,
    algorithm: CompressionAlgorithm,
    buffer: BytesMut,
    decompressed_buffer: Cursor<Vec<u8>>,
    stats: StreamingStats,
    finished: bool,
    buffer_size: usize,
}

impl<R: AsyncRead + Unpin> DecompressingStream<R> {
    /// Create a new decompressing stream
    ///
    /// # Arguments
    ///
    /// * `reader` - The source to read compressed data from
    /// * `algorithm` - The compression algorithm that was used
    pub fn new(reader: R, algorithm: CompressionAlgorithm) -> Result<Self> {
        Ok(Self {
            reader,
            algorithm,
            buffer: BytesMut::with_capacity(DEFAULT_BUFFER_SIZE),
            decompressed_buffer: Cursor::new(Vec::new()),
            stats: StreamingStats::default(),
            finished: false,
            buffer_size: DEFAULT_BUFFER_SIZE,
        })
    }

    /// Create a new decompressing stream with a custom buffer size
    ///
    /// # Arguments
    ///
    /// * `reader` - The source to read compressed data from
    /// * `algorithm` - The compression algorithm that was used
    /// * `buffer_size` - Size of the internal buffer in bytes
    pub fn with_buffer_size(
        reader: R,
        algorithm: CompressionAlgorithm,
        buffer_size: usize,
    ) -> Result<Self> {
        Ok(Self {
            reader,
            algorithm,
            buffer: BytesMut::with_capacity(buffer_size),
            decompressed_buffer: Cursor::new(Vec::new()),
            stats: StreamingStats::default(),
            finished: false,
            buffer_size,
        })
    }

    /// Get statistics about the decompression operation
    pub fn stats(&self) -> &StreamingStats {
        &self.stats
    }
}

impl<R: AsyncRead + Unpin> AsyncRead for DecompressingStream<R> {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<std::io::Result<()>> {
        // Try to read from decompressed buffer first
        let pos = self.decompressed_buffer.position() as usize;
        let available = self.decompressed_buffer.get_ref().len() - pos;

        if available > 0 {
            let to_copy = available.min(buf.remaining());
            buf.put_slice(&self.decompressed_buffer.get_ref()[pos..pos + to_copy]);
            self.decompressed_buffer
                .set_position((pos + to_copy) as u64);
            return Poll::Ready(Ok(()));
        }

        // If finished and no more data, return EOF
        if self.finished {
            return Poll::Ready(Ok(()));
        }

        // Need to read more data from source
        // Get self as mut ref to avoid borrow checker issues
        let this = &mut *self;

        this.buffer.resize(this.buffer_size, 0);
        let mut read_buf = ReadBuf::new(&mut this.buffer[..]);

        match Pin::new(&mut this.reader).poll_read(cx, &mut read_buf) {
            Poll::Ready(Ok(())) => {
                let n = read_buf.filled().len();

                if n == 0 {
                    this.finished = true;
                    return Poll::Ready(Ok(()));
                }

                this.stats.bytes_read += n as u64;

                // Decompress the data
                let data = Bytes::from(this.buffer[..n].to_vec());
                let decompressed = match crate::compression::decompress(&data, this.algorithm) {
                    Ok(d) => d,
                    Err(e) => return Poll::Ready(Err(std::io::Error::other(e.to_string()))),
                };

                this.stats.bytes_written += decompressed.len() as u64;
                this.decompressed_buffer = Cursor::new(decompressed.to_vec());

                // Now read from the decompressed buffer
                let pos = this.decompressed_buffer.position() as usize;
                let available = this.decompressed_buffer.get_ref().len() - pos;

                if available > 0 {
                    let to_copy = available.min(buf.remaining());
                    buf.put_slice(&this.decompressed_buffer.get_ref()[pos..pos + to_copy]);
                    this.decompressed_buffer
                        .set_position((pos + to_copy) as u64);
                }

                Poll::Ready(Ok(()))
            }
            Poll::Ready(Err(e)) => Poll::Ready(Err(e)),
            Poll::Pending => Poll::Pending,
        }
    }
}

/// Statistics for streaming compression/decompression
#[derive(Debug, Clone, Default)]
pub struct StreamingStats {
    /// Total bytes read from source
    pub bytes_read: u64,
    /// Total bytes written (compressed/decompressed)
    pub bytes_written: u64,
}

impl StreamingStats {
    /// Calculate the compression ratio (compressed_size / original_size)
    ///
    /// Returns 1.0 if no data has been processed.
    pub fn compression_ratio(&self) -> f64 {
        if self.bytes_read == 0 {
            1.0
        } else {
            self.bytes_written as f64 / self.bytes_read as f64
        }
    }

    /// Calculate space saved (in bytes)
    pub fn bytes_saved(&self) -> i64 {
        self.bytes_read as i64 - self.bytes_written as i64
    }

    /// Calculate space saved as a percentage
    pub fn savings_percent(&self) -> f64 {
        if self.bytes_read == 0 {
            0.0
        } else {
            (self.bytes_saved() as f64 / self.bytes_read as f64) * 100.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::AsyncReadExt;

    #[tokio::test]
    async fn test_compressing_stream_zstd() {
        let data = b"Hello, world! ".repeat(100);
        let cursor = std::io::Cursor::new(data.clone());

        let mut stream = CompressingStream::new(cursor, CompressionAlgorithm::Zstd, 3).unwrap();

        let mut compressed = Vec::new();
        stream.read_to_end(&mut compressed).await.unwrap();

        assert!(compressed.len() < data.len());
        let stats = stream.stats();
        assert_eq!(stats.bytes_read, data.len() as u64);
        assert!(stats.compression_ratio() < 1.0);
    }

    #[tokio::test]
    async fn test_compressing_stream_lz4() {
        let data = b"Test data for LZ4 compression! ".repeat(100);
        let cursor = std::io::Cursor::new(data.clone());

        let mut stream = CompressingStream::new(cursor, CompressionAlgorithm::Lz4, 5).unwrap();

        let mut compressed = Vec::new();
        stream.read_to_end(&mut compressed).await.unwrap();

        assert!(compressed.len() < data.len());
    }

    #[tokio::test]
    async fn test_compressing_stream_none() {
        let data = b"No compression applied".repeat(10);
        let cursor = std::io::Cursor::new(data.clone());

        let mut stream = CompressingStream::new(cursor, CompressionAlgorithm::None, 0).unwrap();

        let mut output = Vec::new();
        stream.read_to_end(&mut output).await.unwrap();

        assert_eq!(output, data);
        let stats = stream.stats();
        assert_eq!(stats.compression_ratio(), 1.0);
    }

    #[tokio::test]
    async fn test_decompressing_stream_roundtrip() {
        let original = b"Roundtrip test data! ".repeat(100);

        // Compress
        let cursor = std::io::Cursor::new(original.clone());
        let mut compressor = CompressingStream::new(cursor, CompressionAlgorithm::Zstd, 5).unwrap();

        let mut compressed = Vec::new();
        compressor.read_to_end(&mut compressed).await.unwrap();

        // Decompress
        let cursor = std::io::Cursor::new(compressed);
        let mut decompressor =
            DecompressingStream::new(cursor, CompressionAlgorithm::Zstd).unwrap();

        let mut decompressed = Vec::new();
        decompressor.read_to_end(&mut decompressed).await.unwrap();

        assert_eq!(original, decompressed.as_slice());
    }

    #[tokio::test]
    async fn test_streaming_stats() {
        let data = vec![0u8; 10000];
        let cursor = std::io::Cursor::new(data.clone());

        let mut stream = CompressingStream::new(cursor, CompressionAlgorithm::Zstd, 6).unwrap();

        let mut compressed = Vec::new();
        stream.read_to_end(&mut compressed).await.unwrap();

        let stats = stream.stats();
        assert_eq!(stats.bytes_read, 10000);
        assert!(stats.bytes_written < 10000);
        assert!(stats.compression_ratio() < 1.0);
        assert!(stats.bytes_saved() > 0);
        assert!(stats.savings_percent() > 0.0);
    }

    #[tokio::test]
    async fn test_custom_buffer_size() {
        let data = b"Custom buffer size test".repeat(50);
        let cursor = std::io::Cursor::new(data.clone());

        let mut stream =
            CompressingStream::with_buffer_size(cursor, CompressionAlgorithm::Lz4, 3, 1024)
                .unwrap();

        let mut compressed = Vec::new();
        stream.read_to_end(&mut compressed).await.unwrap();

        assert!(compressed.len() < data.len());
    }

    #[tokio::test]
    async fn test_invalid_compression_level() {
        let data = b"test";
        let cursor = std::io::Cursor::new(data);

        let result = CompressingStream::new(cursor, CompressionAlgorithm::Zstd, 10);
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_empty_stream() {
        let data: Vec<u8> = vec![];
        let cursor = std::io::Cursor::new(data);

        let mut stream = CompressingStream::new(cursor, CompressionAlgorithm::Zstd, 3).unwrap();

        let mut compressed = Vec::new();
        stream.read_to_end(&mut compressed).await.unwrap();

        let stats = stream.stats();
        assert_eq!(stats.bytes_read, 0);
        assert_eq!(stats.bytes_written, 0);
    }

    #[tokio::test]
    async fn test_large_data_streaming() {
        // Test with 1MB of repetitive data
        let data = vec![42u8; 1024 * 1024];
        let cursor = std::io::Cursor::new(data.clone());

        let mut stream = CompressingStream::new(cursor, CompressionAlgorithm::Zstd, 9).unwrap();

        let mut compressed = Vec::new();
        stream.read_to_end(&mut compressed).await.unwrap();

        // Should compress very well due to repetitive data
        assert!(compressed.len() < data.len() / 10);

        let stats = stream.stats();
        assert_eq!(stats.bytes_read, 1024 * 1024);
        assert!(stats.compression_ratio() < 0.1);
    }

    #[tokio::test]
    async fn test_decompression_stats() {
        let original = vec![1u8; 5000];

        // Compress
        let cursor = std::io::Cursor::new(original.clone());
        let mut compressor = CompressingStream::new(cursor, CompressionAlgorithm::Lz4, 5).unwrap();

        let mut compressed = Vec::new();
        compressor.read_to_end(&mut compressed).await.unwrap();

        // Decompress
        let cursor = std::io::Cursor::new(compressed.clone());
        let mut decompressor = DecompressingStream::new(cursor, CompressionAlgorithm::Lz4).unwrap();

        let mut decompressed = Vec::new();
        decompressor.read_to_end(&mut decompressed).await.unwrap();

        let stats = decompressor.stats();
        assert_eq!(stats.bytes_read, compressed.len() as u64);
        assert_eq!(stats.bytes_written, original.len() as u64);
        assert!(stats.compression_ratio() > 1.0); // Decompression expands
    }
}
