//! CAR (Content Addressable aRchive) format support.
//!
//! This module provides utilities for reading and writing CAR files, which are
//! used to package and transfer IPLD blocks in the IPFS ecosystem.
//!
//! CAR (CARv1) format structure:
//! - Header: CBOR-encoded with version and root CIDs
//! - Blocks: Sequence of length-prefixed blocks (varint length + CID + data)
//!
//! # Examples
//!
//! ```rust
//! use ipfrs_core::{Block, car::{CarWriter, CarReader}};
//! use bytes::Bytes;
//!
//! // Create some blocks
//! let block1 = Block::new(Bytes::from_static(b"Hello, CAR!")).unwrap();
//! let block2 = Block::new(Bytes::from_static(b"CAR format test")).unwrap();
//!
//! // Write to CAR format
//! let mut car_data = Vec::new();
//! let mut writer = CarWriter::new(&mut car_data, vec![*block1.cid()]).unwrap();
//! writer.write_block(&block1).unwrap();
//! writer.write_block(&block2).unwrap();
//! writer.finish().unwrap();
//!
//! // Read from CAR format
//! let reader = CarReader::new(&car_data[..]).unwrap();
//! let roots = reader.roots();
//! assert_eq!(roots.len(), 1);
//! assert_eq!(roots[0], *block1.cid());
//! ```

use crate::block::Block;
use crate::cid::{Cid, SerializableCid};
use crate::compression::CompressionAlgorithm;
use crate::error::{Error, Result};
use bytes::Bytes;
use std::io::{Read, Write};

/// CAR format version 1.
const CAR_VERSION: u64 = 1;

/// Maximum varint size (10 bytes for 64-bit values).
const MAX_VARINT_SIZE: usize = 10;

/// CAR file header containing version and root CIDs.
#[derive(Debug, Clone)]
pub struct CarHeader {
    /// CAR format version (always 1 for CARv1).
    pub version: u64,
    /// Root CIDs that represent the entry points into the DAG.
    pub roots: Vec<Cid>,
}

impl CarHeader {
    /// Create a new CAR header with the given root CIDs.
    ///
    /// # Arguments
    ///
    /// * `roots` - Vector of root CIDs
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ipfrs_core::{CidBuilder, car::CarHeader};
    ///
    /// let cid = CidBuilder::new().build(b"root data").unwrap();
    /// let header = CarHeader::new(vec![cid]);
    /// assert_eq!(header.version, 1);
    /// assert_eq!(header.roots.len(), 1);
    /// ```
    pub fn new(roots: Vec<Cid>) -> Self {
        Self {
            version: CAR_VERSION,
            roots,
        }
    }

    /// Encode the header to CBOR bytes.
    fn encode(&self) -> Result<Bytes> {
        use crate::ipld::Ipld;
        use std::collections::BTreeMap;

        let mut map = BTreeMap::new();
        map.insert("version".to_string(), Ipld::Integer(self.version as i128));

        let roots: Vec<Ipld> = self
            .roots
            .iter()
            .map(|cid| Ipld::Link(SerializableCid(*cid)))
            .collect();
        map.insert("roots".to_string(), Ipld::List(roots));

        let ipld = Ipld::Map(map);
        ipld.to_dag_cbor().map(Bytes::from)
    }

    /// Decode a header from CBOR bytes.
    fn decode(data: &[u8]) -> Result<Self> {
        use crate::ipld::Ipld;

        let ipld = Ipld::from_dag_cbor(data)?;

        let map = match ipld {
            Ipld::Map(m) => m,
            _ => {
                return Err(Error::Deserialization(
                    "CAR header must be a map".to_string(),
                ))
            }
        };

        let version = match map.get("version") {
            Some(Ipld::Integer(v)) => *v as u64,
            _ => {
                return Err(Error::Deserialization(
                    "CAR header missing version".to_string(),
                ))
            }
        };

        if version != CAR_VERSION {
            return Err(Error::Deserialization(format!(
                "Unsupported CAR version: {}",
                version
            )));
        }

        let roots = match map.get("roots") {
            Some(Ipld::List(list)) => list
                .iter()
                .map(|item| match item {
                    Ipld::Link(SerializableCid(cid)) => Ok(*cid),
                    _ => Err(Error::Deserialization(
                        "Invalid root CID in header".to_string(),
                    )),
                })
                .collect::<Result<Vec<Cid>>>()?,
            _ => {
                return Err(Error::Deserialization(
                    "CAR header missing roots".to_string(),
                ))
            }
        };

        Ok(Self { version, roots })
    }
}

/// Compression statistics for CAR operations.
#[derive(Debug, Clone, Default)]
pub struct CarCompressionStats {
    /// Total number of blocks processed.
    pub blocks_processed: usize,
    /// Total uncompressed bytes written.
    pub uncompressed_bytes: usize,
    /// Total compressed bytes written.
    pub compressed_bytes: usize,
    /// Number of blocks that were compressed.
    pub blocks_compressed: usize,
}

impl CarCompressionStats {
    /// Create new empty compression statistics.
    pub fn new() -> Self {
        Self::default()
    }

    /// Calculate the compression ratio (compressed / uncompressed).
    ///
    /// Returns 1.0 if no compression occurred.
    pub fn compression_ratio(&self) -> f64 {
        if self.uncompressed_bytes == 0 {
            1.0
        } else {
            self.compressed_bytes as f64 / self.uncompressed_bytes as f64
        }
    }

    /// Calculate bytes saved through compression.
    pub fn bytes_saved(&self) -> usize {
        self.uncompressed_bytes
            .saturating_sub(self.compressed_bytes)
    }

    /// Calculate compression percentage (0-100).
    pub fn compression_percentage(&self) -> f64 {
        if self.uncompressed_bytes == 0 {
            0.0
        } else {
            (self.bytes_saved() as f64 / self.uncompressed_bytes as f64) * 100.0
        }
    }
}

/// Builder for creating a CarWriter with optional compression.
///
/// # Examples
///
/// ```rust
/// use ipfrs_core::{CidBuilder, car::CarWriterBuilder, compression::CompressionAlgorithm};
///
/// let cid = CidBuilder::new().build(b"root").unwrap();
/// let mut output = Vec::new();
/// let writer = CarWriterBuilder::new(vec![cid])
///     .with_compression(CompressionAlgorithm::Zstd, 3)
///     .build(&mut output)
///     .unwrap();
/// ```
pub struct CarWriterBuilder {
    roots: Vec<Cid>,
    compression: Option<(CompressionAlgorithm, i32)>,
}

impl CarWriterBuilder {
    /// Create a new CarWriter builder with the given root CIDs.
    pub fn new(roots: Vec<Cid>) -> Self {
        Self {
            roots,
            compression: None,
        }
    }

    /// Enable compression with the specified algorithm and level.
    ///
    /// # Arguments
    ///
    /// * `algorithm` - The compression algorithm to use
    /// * `level` - Compression level (0-9 for Zstd, 0-12 for Lz4)
    pub fn with_compression(mut self, algorithm: CompressionAlgorithm, level: i32) -> Self {
        self.compression = Some((algorithm, level));
        self
    }

    /// Build the CarWriter with the configured options.
    pub fn build<W: Write>(self, writer: W) -> Result<CarWriter<W>> {
        CarWriter::new_with_options(writer, self.roots, self.compression)
    }
}

/// Write blocks to CAR format.
///
/// CAR files contain a CBOR-encoded header followed by length-prefixed blocks.
/// Optionally supports block compression for reduced archive sizes.
pub struct CarWriter<W: Write> {
    writer: W,
    header_written: bool,
    compression: Option<(CompressionAlgorithm, i32)>,
    stats: CarCompressionStats,
}

impl<W: Write> CarWriter<W> {
    /// Create a new CAR writer with the given root CIDs.
    ///
    /// For compression support, use `CarWriterBuilder` instead.
    ///
    /// # Arguments
    ///
    /// * `writer` - The writer to output CAR data to
    /// * `roots` - Vector of root CIDs for the CAR file
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ipfrs_core::{CidBuilder, car::CarWriter};
    ///
    /// let cid = CidBuilder::new().build(b"root").unwrap();
    /// let mut output = Vec::new();
    /// let writer = CarWriter::new(&mut output, vec![cid]).unwrap();
    /// ```
    pub fn new(writer: W, roots: Vec<Cid>) -> Result<Self> {
        Self::new_with_options(writer, roots, None)
    }

    /// Create a new CAR writer with optional compression.
    ///
    /// This is used internally by `CarWriterBuilder`.
    ///
    /// # Arguments
    ///
    /// * `writer` - The writer to output CAR data to
    /// * `roots` - Vector of root CIDs for the CAR file
    /// * `compression` - Optional compression algorithm and level
    fn new_with_options(
        writer: W,
        roots: Vec<Cid>,
        compression: Option<(CompressionAlgorithm, i32)>,
    ) -> Result<Self> {
        let mut car_writer = Self {
            writer,
            header_written: false,
            compression,
            stats: CarCompressionStats::new(),
        };
        car_writer.write_header(&CarHeader::new(roots))?;
        Ok(car_writer)
    }

    /// Write the CAR header.
    fn write_header(&mut self, header: &CarHeader) -> Result<()> {
        let header_bytes = header.encode()?;
        let header_len = header_bytes.len();

        // Write header length as varint
        write_varint(&mut self.writer, header_len as u64)?;

        // Write header data
        self.writer.write_all(&header_bytes)?;

        self.header_written = true;
        Ok(())
    }

    /// Write a block to the CAR file.
    ///
    /// If compression is enabled, the block data will be compressed before writing.
    ///
    /// # Arguments
    ///
    /// * `block` - The block to write
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ipfrs_core::{Block, car::CarWriter};
    /// use bytes::Bytes;
    ///
    /// let block = Block::new(Bytes::from_static(b"test data")).unwrap();
    /// let mut output = Vec::new();
    /// let mut writer = CarWriter::new(&mut output, vec![*block.cid()]).unwrap();
    /// writer.write_block(&block).unwrap();
    /// ```
    pub fn write_block(&mut self, block: &Block) -> Result<()> {
        if !self.header_written {
            return Err(Error::InvalidData("CAR header not written".to_string()));
        }

        // Encode CID to bytes
        let cid_bytes = block.cid().to_bytes();
        let data = block.data();

        // Update statistics
        self.stats.blocks_processed += 1;
        self.stats.uncompressed_bytes += data.len();

        // Compress data if compression is enabled
        let (final_data, was_compressed) = if let Some((algorithm, level)) = self.compression {
            let compressed = crate::compression::compress(data, algorithm, level as u8)?;
            // Only mark as compressed if algorithm is not None
            let is_compressed = algorithm != CompressionAlgorithm::None;
            if is_compressed {
                self.stats.blocks_compressed += 1;
            }
            self.stats.compressed_bytes += compressed.len();
            (compressed, is_compressed)
        } else {
            self.stats.compressed_bytes += data.len();
            (data.clone(), false)
        };

        // Total length: CID bytes + compression flag (1 byte) + block data
        let total_len = cid_bytes.len() + 1 + final_data.len();

        // Write length as varint
        write_varint(&mut self.writer, total_len as u64)?;

        // Write CID
        self.writer.write_all(&cid_bytes)?;

        // Write compression flag (0 = uncompressed, 1 = compressed)
        self.writer.write_all(&[was_compressed as u8])?;

        // Write block data (compressed or uncompressed)
        self.writer.write_all(&final_data)?;

        Ok(())
    }

    /// Get the compression statistics for this writer.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ipfrs_core::{Block, car::CarWriterBuilder, compression::CompressionAlgorithm};
    /// use bytes::Bytes;
    ///
    /// let block = Block::new(Bytes::from(vec![0u8; 1000])).unwrap();
    /// let mut output = Vec::new();
    /// let mut writer = CarWriterBuilder::new(vec![*block.cid()])
    ///     .with_compression(CompressionAlgorithm::Zstd, 3)
    ///     .build(&mut output)
    ///     .unwrap();
    /// writer.write_block(&block).unwrap();
    /// let stats = writer.stats();
    /// assert_eq!(stats.blocks_processed, 1);
    /// ```
    pub fn stats(&self) -> &CarCompressionStats {
        &self.stats
    }

    /// Finish writing and flush the writer.
    pub fn finish(mut self) -> Result<()> {
        self.writer.flush()?;
        Ok(())
    }
}

/// Read blocks from CAR format.
///
/// CAR files are read sequentially, yielding blocks one at a time.
pub struct CarReader<R: Read> {
    reader: R,
    header: CarHeader,
}

impl<R: Read> CarReader<R> {
    /// Create a new CAR reader.
    ///
    /// This reads and parses the CAR header immediately.
    ///
    /// # Arguments
    ///
    /// * `reader` - The reader to read CAR data from
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ipfrs_core::{Block, car::{CarWriter, CarReader}};
    /// use bytes::Bytes;
    ///
    /// let block = Block::new(Bytes::from_static(b"test")).unwrap();
    /// let mut data = Vec::new();
    /// let mut writer = CarWriter::new(&mut data, vec![*block.cid()]).unwrap();
    /// writer.write_block(&block).unwrap();
    /// writer.finish().unwrap();
    ///
    /// let reader = CarReader::new(&data[..]).unwrap();
    /// assert_eq!(reader.roots().len(), 1);
    /// ```
    pub fn new(mut reader: R) -> Result<Self> {
        // Read header length
        let header_len = read_varint(&mut reader)?;

        // Read header data
        let mut header_bytes = vec![0u8; header_len as usize];
        reader.read_exact(&mut header_bytes)?;

        // Decode header
        let header = CarHeader::decode(&header_bytes)?;

        Ok(Self { reader, header })
    }

    /// Get the root CIDs from the CAR header.
    pub fn roots(&self) -> &[Cid] {
        &self.header.roots
    }

    /// Read the next block from the CAR file.
    ///
    /// Returns `None` when there are no more blocks.
    /// Automatically decompresses blocks if they were compressed during writing.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ipfrs_core::{Block, car::{CarWriter, CarReader}};
    /// use bytes::Bytes;
    ///
    /// let block = Block::new(Bytes::from_static(b"data")).unwrap();
    /// let mut data = Vec::new();
    /// let mut writer = CarWriter::new(&mut data, vec![*block.cid()]).unwrap();
    /// writer.write_block(&block).unwrap();
    /// writer.finish().unwrap();
    ///
    /// let mut reader = CarReader::new(&data[..]).unwrap();
    /// let read_block = reader.read_block().unwrap().unwrap();
    /// assert_eq!(read_block.cid(), block.cid());
    /// ```
    pub fn read_block(&mut self) -> Result<Option<Block>> {
        // Try to read length varint
        let total_len = match read_varint_opt(&mut self.reader) {
            Ok(Some(len)) => len,
            Ok(None) => return Ok(None), // EOF
            Err(e) => return Err(e),
        };

        // Read CID + compression flag + data
        let mut block_bytes = vec![0u8; total_len as usize];
        self.reader.read_exact(&mut block_bytes)?;

        let mut cursor = &block_bytes[..];

        // Parse CID
        let cid = Cid::read_bytes(&mut cursor)
            .map_err(|e| Error::Cid(format!("Failed to parse CID: {}", e)))?;

        // Read compression flag (may not exist for legacy CAR files)
        let (is_compressed, data_start) = if !cursor.is_empty() {
            let flag = cursor[0];
            if flag == 0 || flag == 1 {
                // Valid compression flag found
                (flag == 1, 1)
            } else {
                // No compression flag (legacy CAR file)
                (false, 0)
            }
        } else {
            return Err(Error::Deserialization("Empty block data".to_string()));
        };

        // Get the block data (after compression flag, if present)
        let raw_data = &cursor[data_start..];

        // Decompress if necessary
        let final_data = if is_compressed {
            let raw_bytes = Bytes::from(raw_data.to_vec());
            // Try both compression algorithms since we don't store which one was used
            // Try Zstd first (most common), then Lz4
            crate::compression::decompress(&raw_bytes, CompressionAlgorithm::Zstd).or_else(
                |_| crate::compression::decompress(&raw_bytes, CompressionAlgorithm::Lz4),
            )?
        } else {
            Bytes::from(raw_data.to_vec())
        };

        // Create block
        let block = Block::new(final_data)?;

        // Verify CID matches
        if block.cid() != &cid {
            return Err(Error::InvalidData(format!(
                "Block CID mismatch: expected {}, got {}",
                cid,
                block.cid()
            )));
        }

        Ok(Some(block))
    }

    /// Read all blocks from the CAR file.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ipfrs_core::{Block, car::{CarWriter, CarReader}};
    /// use bytes::Bytes;
    ///
    /// let block1 = Block::new(Bytes::from_static(b"data1")).unwrap();
    /// let block2 = Block::new(Bytes::from_static(b"data2")).unwrap();
    /// let mut data = Vec::new();
    /// let mut writer = CarWriter::new(&mut data, vec![*block1.cid()]).unwrap();
    /// writer.write_block(&block1).unwrap();
    /// writer.write_block(&block2).unwrap();
    /// writer.finish().unwrap();
    ///
    /// let mut reader = CarReader::new(&data[..]).unwrap();
    /// let blocks = reader.read_all_blocks().unwrap();
    /// assert_eq!(blocks.len(), 2);
    /// ```
    pub fn read_all_blocks(&mut self) -> Result<Vec<Block>> {
        let mut blocks = Vec::new();

        while let Some(block) = self.read_block()? {
            blocks.push(block);
        }

        Ok(blocks)
    }
}

/// Write a varint-encoded unsigned integer.
fn write_varint<W: Write>(writer: &mut W, mut value: u64) -> Result<()> {
    let mut buf = [0u8; MAX_VARINT_SIZE];
    let mut i = 0;

    loop {
        let mut byte = (value & 0x7F) as u8;
        value >>= 7;

        if value != 0 {
            byte |= 0x80;
        }

        buf[i] = byte;
        i += 1;

        if value == 0 {
            break;
        }
    }

    writer.write_all(&buf[..i])?;
    Ok(())
}

/// Read a varint-encoded unsigned integer.
fn read_varint<R: Read>(reader: &mut R) -> Result<u64> {
    read_varint_opt(reader)?
        .ok_or_else(|| Error::Deserialization("Unexpected EOF reading varint".to_string()))
}

/// Read a varint-encoded unsigned integer, returning None on EOF.
fn read_varint_opt<R: Read>(reader: &mut R) -> Result<Option<u64>> {
    let mut result = 0u64;
    let mut shift = 0;
    let mut buf = [0u8; 1];

    for _ in 0..MAX_VARINT_SIZE {
        match reader.read_exact(&mut buf) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof && shift == 0 => {
                return Ok(None); // EOF at start
            }
            Err(e) => return Err(Error::from(e)),
        }

        let byte = buf[0];
        result |= ((byte & 0x7F) as u64) << shift;

        if byte & 0x80 == 0 {
            return Ok(Some(result));
        }

        shift += 7;

        if shift >= 64 {
            return Err(Error::Deserialization("Varint too large".to_string()));
        }
    }

    Err(Error::Deserialization(
        "Varint exceeds maximum size".to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::Block;
    use bytes::Bytes;

    #[test]
    fn test_car_header_encode_decode() {
        use crate::cid::CidBuilder;

        let cid1 = CidBuilder::new().build(b"test1").unwrap();
        let cid2 = CidBuilder::new().build(b"test2").unwrap();

        let header = CarHeader::new(vec![cid1, cid2]);
        let encoded = header.encode().unwrap();
        let decoded = CarHeader::decode(&encoded).unwrap();

        assert_eq!(decoded.version, 1);
        assert_eq!(decoded.roots.len(), 2);
        assert_eq!(decoded.roots[0], cid1);
        assert_eq!(decoded.roots[1], cid2);
    }

    #[test]
    fn test_car_write_read() {
        let block1 = Block::new(Bytes::from_static(b"Hello, CAR!")).unwrap();
        let block2 = Block::new(Bytes::from_static(b"CAR format test")).unwrap();

        // Write
        let mut car_data = Vec::new();
        let mut writer = CarWriter::new(&mut car_data, vec![*block1.cid()]).unwrap();
        writer.write_block(&block1).unwrap();
        writer.write_block(&block2).unwrap();
        writer.finish().unwrap();

        // Read
        let mut reader = CarReader::new(&car_data[..]).unwrap();

        assert_eq!(reader.roots().len(), 1);
        assert_eq!(reader.roots()[0], *block1.cid());

        let read_block1 = reader.read_block().unwrap().unwrap();
        assert_eq!(read_block1.cid(), block1.cid());
        assert_eq!(read_block1.data(), block1.data());

        let read_block2 = reader.read_block().unwrap().unwrap();
        assert_eq!(read_block2.cid(), block2.cid());
        assert_eq!(read_block2.data(), block2.data());

        assert!(reader.read_block().unwrap().is_none());
    }

    #[test]
    fn test_car_read_all_blocks() {
        let blocks: Vec<Block> = (0..5)
            .map(|i| Block::new(Bytes::from(format!("Block {}", i))).unwrap())
            .collect();

        let mut car_data = Vec::new();
        let mut writer = CarWriter::new(&mut car_data, vec![*blocks[0].cid()]).unwrap();

        for block in &blocks {
            writer.write_block(block).unwrap();
        }
        writer.finish().unwrap();

        let mut reader = CarReader::new(&car_data[..]).unwrap();
        let read_blocks = reader.read_all_blocks().unwrap();

        assert_eq!(read_blocks.len(), blocks.len());

        for (i, block) in read_blocks.iter().enumerate() {
            assert_eq!(block.cid(), blocks[i].cid());
            assert_eq!(block.data(), blocks[i].data());
        }
    }

    #[test]
    fn test_varint_roundtrip() {
        let test_values = vec![0, 1, 127, 128, 255, 256, 65535, 65536, u64::MAX];

        for value in test_values {
            let mut buf = Vec::new();
            write_varint(&mut buf, value).unwrap();

            let mut cursor = &buf[..];
            let decoded = read_varint(&mut cursor).unwrap();

            assert_eq!(decoded, value);
        }
    }

    #[test]
    fn test_car_empty_roots() {
        let block = Block::new(Bytes::from_static(b"test")).unwrap();

        let mut car_data = Vec::new();
        let mut writer = CarWriter::new(&mut car_data, vec![]).unwrap();
        writer.write_block(&block).unwrap();
        writer.finish().unwrap();

        let reader = CarReader::new(&car_data[..]).unwrap();
        assert_eq!(reader.roots().len(), 0);
    }

    #[test]
    fn test_car_multiple_roots() {
        use crate::cid::CidBuilder;

        let cid1 = CidBuilder::new().build(b"root1").unwrap();
        let cid2 = CidBuilder::new().build(b"root2").unwrap();
        let cid3 = CidBuilder::new().build(b"root3").unwrap();

        let block = Block::new(Bytes::from_static(b"data")).unwrap();

        let mut car_data = Vec::new();
        let mut writer = CarWriter::new(&mut car_data, vec![cid1, cid2, cid3]).unwrap();
        writer.write_block(&block).unwrap();
        writer.finish().unwrap();

        let reader = CarReader::new(&car_data[..]).unwrap();
        let roots = reader.roots();
        assert_eq!(roots.len(), 3);
        assert_eq!(roots[0], cid1);
        assert_eq!(roots[1], cid2);
        assert_eq!(roots[2], cid3);
    }

    #[test]
    fn test_car_large_blocks() {
        // Test with blocks larger than typical sizes
        let large_data = vec![0x42u8; 1_000_000]; // 1MB block
        let block = Block::new(Bytes::from(large_data.clone())).unwrap();

        let mut car_data = Vec::new();
        let mut writer = CarWriter::new(&mut car_data, vec![*block.cid()]).unwrap();
        writer.write_block(&block).unwrap();
        writer.finish().unwrap();

        let mut reader = CarReader::new(&car_data[..]).unwrap();
        let read_block = reader.read_block().unwrap().unwrap();

        assert_eq!(read_block.cid(), block.cid());
        assert_eq!(read_block.data().len(), large_data.len());
    }

    #[test]
    fn test_car_compression_zstd() {
        use crate::compression::CompressionAlgorithm;

        let block1 = Block::new(Bytes::from(vec![0x42u8; 1000])).unwrap();
        let block2 = Block::new(Bytes::from(vec![0xAAu8; 2000])).unwrap();

        // Write with Zstd compression
        let mut car_data = Vec::new();
        let mut writer = CarWriterBuilder::new(vec![*block1.cid()])
            .with_compression(CompressionAlgorithm::Zstd, 3)
            .build(&mut car_data)
            .unwrap();

        writer.write_block(&block1).unwrap();
        writer.write_block(&block2).unwrap();

        let stats = writer.stats();
        assert_eq!(stats.blocks_processed, 2);
        assert_eq!(stats.blocks_compressed, 2);
        assert_eq!(stats.uncompressed_bytes, 3000);
        assert!(stats.compressed_bytes < stats.uncompressed_bytes);
        assert!(stats.compression_ratio() < 1.0);

        writer.finish().unwrap();

        // Read and verify
        let mut reader = CarReader::new(&car_data[..]).unwrap();
        let read_block1 = reader.read_block().unwrap().unwrap();
        let read_block2 = reader.read_block().unwrap().unwrap();

        assert_eq!(read_block1.cid(), block1.cid());
        assert_eq!(read_block1.data(), block1.data());
        assert_eq!(read_block2.cid(), block2.cid());
        assert_eq!(read_block2.data(), block2.data());
    }

    #[test]
    fn test_car_compression_lz4() {
        use crate::compression::CompressionAlgorithm;

        let block = Block::new(Bytes::from(vec![0x11u8; 5000])).unwrap();

        // Write with LZ4 compression
        let mut car_data = Vec::new();
        let mut writer = CarWriterBuilder::new(vec![*block.cid()])
            .with_compression(CompressionAlgorithm::Lz4, 1)
            .build(&mut car_data)
            .unwrap();

        writer.write_block(&block).unwrap();

        let stats = writer.stats();
        assert_eq!(stats.blocks_processed, 1);
        assert_eq!(stats.blocks_compressed, 1);
        assert!(stats.compressed_bytes < stats.uncompressed_bytes);

        writer.finish().unwrap();

        // Read and verify
        let mut reader = CarReader::new(&car_data[..]).unwrap();
        let read_block = reader.read_block().unwrap().unwrap();

        assert_eq!(read_block.cid(), block.cid());
        assert_eq!(read_block.data(), block.data());
    }

    #[test]
    fn test_car_compression_none() {
        use crate::compression::CompressionAlgorithm;

        let block = Block::new(Bytes::from_static(b"test data")).unwrap();

        // Write with None compression (passthrough)
        let mut car_data = Vec::new();
        let mut writer = CarWriterBuilder::new(vec![*block.cid()])
            .with_compression(CompressionAlgorithm::None, 0)
            .build(&mut car_data)
            .unwrap();

        writer.write_block(&block).unwrap();

        let stats = writer.stats();
        assert_eq!(stats.blocks_processed, 1);
        assert_eq!(stats.blocks_compressed, 0); // None algorithm doesn't count as compressed
        assert_eq!(stats.uncompressed_bytes, stats.compressed_bytes);
        assert_eq!(stats.compression_ratio(), 1.0);

        writer.finish().unwrap();

        // Read and verify
        let mut reader = CarReader::new(&car_data[..]).unwrap();
        let read_block = reader.read_block().unwrap().unwrap();

        assert_eq!(read_block.cid(), block.cid());
        assert_eq!(read_block.data(), block.data());
    }

    #[test]
    fn test_car_compression_stats() {
        use crate::compression::CompressionAlgorithm;

        let blocks: Vec<Block> = (0..10)
            .map(|_| Block::new(Bytes::from(vec![0x42u8; 500])).unwrap())
            .collect();

        let mut car_data = Vec::new();
        let mut writer = CarWriterBuilder::new(vec![*blocks[0].cid()])
            .with_compression(CompressionAlgorithm::Zstd, 5)
            .build(&mut car_data)
            .unwrap();

        for block in &blocks {
            writer.write_block(block).unwrap();
        }

        let stats = writer.stats();
        assert_eq!(stats.blocks_processed, 10);
        assert_eq!(stats.blocks_compressed, 10);
        assert_eq!(stats.uncompressed_bytes, 5000);
        assert!(stats.bytes_saved() > 0);
        assert!(stats.compression_percentage() > 0.0);

        writer.finish().unwrap();
    }

    #[test]
    fn test_car_mixed_compression_backward_compat() {
        // Test that uncompressed CAR files can still be read
        let block = Block::new(Bytes::from_static(b"legacy data")).unwrap();

        // Write without compression (legacy format)
        let mut car_data = Vec::new();
        let mut writer = CarWriter::new(&mut car_data, vec![*block.cid()]).unwrap();
        writer.write_block(&block).unwrap();
        writer.finish().unwrap();

        // Read should work fine
        let mut reader = CarReader::new(&car_data[..]).unwrap();
        let read_block = reader.read_block().unwrap().unwrap();

        assert_eq!(read_block.cid(), block.cid());
        assert_eq!(read_block.data(), block.data());
    }

    #[test]
    fn test_car_compression_large_file() {
        use crate::compression::CompressionAlgorithm;

        // Simulate a large file with repetitive data (compresses well)
        let large_block = Block::new(Bytes::from(vec![0x55u8; 100_000])).unwrap();

        let mut car_data = Vec::new();
        let mut writer = CarWriterBuilder::new(vec![*large_block.cid()])
            .with_compression(CompressionAlgorithm::Zstd, 6)
            .build(&mut car_data)
            .unwrap();

        writer.write_block(&large_block).unwrap();

        let stats = writer.stats();
        assert_eq!(stats.uncompressed_bytes, 100_000);
        // Repetitive data should compress very well
        assert!(stats.compressed_bytes < 1_000);
        assert!(stats.compression_ratio() < 0.01);
        assert!(stats.compression_percentage() > 99.0);

        writer.finish().unwrap();

        // Verify decompression works
        let mut reader = CarReader::new(&car_data[..]).unwrap();
        let read_block = reader.read_block().unwrap().unwrap();

        assert_eq!(read_block.cid(), large_block.cid());
        assert_eq!(read_block.data().len(), 100_000);
    }

    #[test]
    fn test_car_builder_without_compression() {
        let block = Block::new(Bytes::from_static(b"test")).unwrap();

        // Test builder without compression
        let mut car_data = Vec::new();
        let mut writer = CarWriterBuilder::new(vec![*block.cid()])
            .build(&mut car_data)
            .unwrap();

        writer.write_block(&block).unwrap();
        writer.finish().unwrap();

        let mut reader = CarReader::new(&car_data[..]).unwrap();
        let read_block = reader.read_block().unwrap().unwrap();

        assert_eq!(read_block.cid(), block.cid());
        assert_eq!(read_block.data(), block.data());
    }
}
