//! CAR (Content Addressable aRchive) format support.
//!
//! CAR files are the standard format for packaging content-addressed data.
//! They contain a header with root CIDs followed by a series of blocks.
//!
//! # Format
//!
//! ```text
//! | Header (CBOR) | Block 1 | Block 2 | ... | Block N |
//! ```
//!
//! Each block is prefixed with a varint length and contains:
//! - CID bytes
//! - Block data
//!
//! # Example
//!
//! ```rust,ignore
//! use ipfrs_storage::car::{CarWriter, CarReader};
//!
//! // Export blocks to CAR file
//! let mut writer = CarWriter::new(&path, roots).await?;
//! for block in blocks {
//!     writer.write_block(&block).await?;
//! }
//! writer.finish().await?;
//!
//! // Import blocks from CAR file
//! let mut reader = CarReader::open(&path).await?;
//! while let Some(block) = reader.read_block().await? {
//!     store.put(&block).await?;
//! }
//! ```

use crate::traits::BlockStore;
use bytes::Bytes;
use ipfrs_core::{Block, Cid, Error, Result};
use std::path::Path;
use tokio::fs::File;
use tokio::io::{AsyncReadExt, AsyncWriteExt, BufReader, BufWriter};

/// CAR file version
pub const CAR_VERSION: u64 = 1;

/// CAR file header
#[derive(Debug, Clone)]
pub struct CarHeader {
    /// CAR version (currently 1)
    pub version: u64,
    /// Root CIDs
    pub roots: Vec<Cid>,
}

impl CarHeader {
    /// Create a new CAR header
    pub fn new(roots: Vec<Cid>) -> Self {
        Self {
            version: CAR_VERSION,
            roots,
        }
    }

    /// Encode header to CBOR bytes
    pub fn to_cbor(&self) -> Result<Vec<u8>> {
        // Simple CBOR encoding for header
        // { "version": 1, "roots": [<cid bytes>...] }
        let mut buf = Vec::new();

        // Map with 2 entries
        buf.push(0xa2); // map(2)

        // Key: "version"
        buf.push(0x67); // text(7)
        buf.extend_from_slice(b"version");
        // Value: version number
        buf.push(0x01); // unsigned(1)

        // Key: "roots"
        buf.push(0x65); // text(5)
        buf.extend_from_slice(b"roots");

        // Value: array of CIDs
        let roots_len = self.roots.len();
        if roots_len < 24 {
            buf.push(0x80 | roots_len as u8); // array(n)
        } else if roots_len < 256 {
            buf.push(0x98); // array(1 byte)
            buf.push(roots_len as u8);
        } else {
            return Err(Error::InvalidData("Too many roots".to_string()));
        }

        for root in &self.roots {
            let cid_bytes = root.to_bytes();
            // CID as byte string (tag 42 for CID)
            buf.push(0xd8); // tag(42)
            buf.push(0x2a);
            // Byte string
            if cid_bytes.len() < 24 {
                buf.push(0x40 | cid_bytes.len() as u8);
            } else if cid_bytes.len() < 256 {
                buf.push(0x58);
                buf.push(cid_bytes.len() as u8);
            } else {
                buf.push(0x59);
                buf.extend_from_slice(&(cid_bytes.len() as u16).to_be_bytes());
            }
            buf.extend_from_slice(&cid_bytes);
        }

        Ok(buf)
    }

    /// Decode header from CBOR bytes
    pub fn from_cbor(data: &[u8]) -> Result<Self> {
        // Simple CBOR parsing - just enough for CAR headers

        // Expect map
        if data.is_empty() || (data[0] & 0xe0) != 0xa0 {
            return Err(Error::InvalidData("Expected CBOR map".to_string()));
        }

        let (map_len, mut pos) = if data[0] == 0xa2 {
            (2, 1)
        } else {
            return Err(Error::InvalidData("Expected map(2)".to_string()));
        };

        let mut version = 1u64;
        let mut roots = Vec::new();

        for _ in 0..map_len {
            // Read key (text string)
            let (key, new_pos) = read_cbor_text(&data[pos..])?;
            pos += new_pos;

            match key.as_str() {
                "version" => {
                    // Read unsigned int
                    if pos >= data.len() {
                        return Err(Error::InvalidData("Unexpected end".to_string()));
                    }
                    let (v, new_pos) = read_cbor_uint(&data[pos..])?;
                    version = v;
                    pos += new_pos;
                }
                "roots" => {
                    // Read array of CIDs
                    let (r, new_pos) = read_cbor_roots(&data[pos..])?;
                    roots = r;
                    pos += new_pos;
                }
                _ => {
                    // Skip unknown keys
                    let new_pos = skip_cbor_value(&data[pos..])?;
                    pos += new_pos;
                }
            }
        }

        Ok(Self { version, roots })
    }
}

/// Read CBOR text string
fn read_cbor_text(data: &[u8]) -> Result<(String, usize)> {
    if data.is_empty() {
        return Err(Error::InvalidData("Unexpected end".to_string()));
    }

    let major = data[0] >> 5;
    if major != 3 {
        // text string
        return Err(Error::InvalidData("Expected text string".to_string()));
    }

    let (len, header_len) = read_cbor_len(data)?;
    let total_len = header_len + len;

    if data.len() < total_len {
        return Err(Error::InvalidData("Text string too short".to_string()));
    }

    let text = String::from_utf8(data[header_len..total_len].to_vec())
        .map_err(|e| Error::InvalidData(format!("Invalid UTF-8: {e}")))?;

    Ok((text, total_len))
}

/// Read CBOR unsigned int
fn read_cbor_uint(data: &[u8]) -> Result<(u64, usize)> {
    if data.is_empty() {
        return Err(Error::InvalidData("Unexpected end".to_string()));
    }

    let major = data[0] >> 5;
    if major != 0 {
        return Err(Error::InvalidData("Expected unsigned int".to_string()));
    }

    let (val, len) = read_cbor_len(data)?;
    Ok((val as u64, len))
}

/// Read CBOR length prefix
fn read_cbor_len(data: &[u8]) -> Result<(usize, usize)> {
    if data.is_empty() {
        return Err(Error::InvalidData("Unexpected end".to_string()));
    }

    let additional = data[0] & 0x1f;

    match additional {
        0..=23 => Ok((additional as usize, 1)),
        24 => {
            if data.len() < 2 {
                return Err(Error::InvalidData("Length too short".to_string()));
            }
            Ok((data[1] as usize, 2))
        }
        25 => {
            if data.len() < 3 {
                return Err(Error::InvalidData("Length too short".to_string()));
            }
            Ok((u16::from_be_bytes([data[1], data[2]]) as usize, 3))
        }
        26 => {
            if data.len() < 5 {
                return Err(Error::InvalidData("Length too short".to_string()));
            }
            Ok((
                u32::from_be_bytes([data[1], data[2], data[3], data[4]]) as usize,
                5,
            ))
        }
        _ => Err(Error::InvalidData(
            "Unsupported length encoding".to_string(),
        )),
    }
}

/// Read array of CIDs
fn read_cbor_roots(data: &[u8]) -> Result<(Vec<Cid>, usize)> {
    if data.is_empty() {
        return Err(Error::InvalidData("Unexpected end".to_string()));
    }

    let major = data[0] >> 5;
    if major != 4 {
        // array
        return Err(Error::InvalidData("Expected array".to_string()));
    }

    let (arr_len, header_len) = read_cbor_len(data)?;
    let mut pos = header_len;
    let mut roots = Vec::with_capacity(arr_len);

    for _ in 0..arr_len {
        // Skip tag if present (tag 42 for CID)
        if pos < data.len() && data[pos] == 0xd8 {
            pos += 2; // Skip tag
        }

        // Read byte string
        if pos >= data.len() {
            return Err(Error::InvalidData("Unexpected end in roots".to_string()));
        }

        let major = data[pos] >> 5;
        if major != 2 {
            // byte string
            return Err(Error::InvalidData(
                "Expected byte string for CID".to_string(),
            ));
        }

        let (len, header) = read_cbor_len(&data[pos..])?;
        pos += header;

        if pos + len > data.len() {
            return Err(Error::InvalidData("CID bytes too short".to_string()));
        }

        let cid = Cid::try_from(data[pos..pos + len].to_vec())
            .map_err(|e| Error::Cid(format!("Invalid CID: {e}")))?;
        roots.push(cid);
        pos += len;
    }

    Ok((roots, pos))
}

/// Skip a CBOR value
fn skip_cbor_value(data: &[u8]) -> Result<usize> {
    if data.is_empty() {
        return Err(Error::InvalidData("Unexpected end".to_string()));
    }

    let major = data[0] >> 5;
    let (len, header_len) = read_cbor_len(data)?;

    match major {
        0 | 1 => Ok(header_len),       // unsigned/negative int
        2 | 3 => Ok(header_len + len), // byte/text string
        4 => {
            // array
            let mut pos = header_len;
            for _ in 0..len {
                pos += skip_cbor_value(&data[pos..])?;
            }
            Ok(pos)
        }
        5 => {
            // map
            let mut pos = header_len;
            for _ in 0..len {
                pos += skip_cbor_value(&data[pos..])?; // key
                pos += skip_cbor_value(&data[pos..])?; // value
            }
            Ok(pos)
        }
        6 => {
            // tag
            Ok(header_len + skip_cbor_value(&data[header_len..])?)
        }
        7 => Ok(header_len), // simple/float
        _ => Err(Error::InvalidData("Unknown CBOR major type".to_string())),
    }
}

/// Encode unsigned varint
fn encode_varint(mut value: u64) -> Vec<u8> {
    let mut buf = Vec::new();
    while value >= 0x80 {
        buf.push((value as u8) | 0x80);
        value >>= 7;
    }
    buf.push(value as u8);
    buf
}

/// Decode unsigned varint
fn decode_varint(data: &[u8]) -> Result<(u64, usize)> {
    let mut result: u64 = 0;
    let mut shift = 0;

    for (i, &byte) in data.iter().enumerate() {
        result |= ((byte & 0x7f) as u64) << shift;
        if byte & 0x80 == 0 {
            return Ok((result, i + 1));
        }
        shift += 7;
        if shift >= 64 {
            return Err(Error::InvalidData("Varint too long".to_string()));
        }
    }

    Err(Error::InvalidData("Incomplete varint".to_string()))
}

/// CAR file writer
pub struct CarWriter {
    writer: BufWriter<File>,
    blocks_written: u64,
    bytes_written: u64,
}

impl CarWriter {
    /// Create a new CAR writer
    pub async fn create(path: &Path, roots: Vec<Cid>) -> Result<Self> {
        let file = File::create(path)
            .await
            .map_err(|e| Error::Storage(format!("Failed to create CAR file: {e}")))?;

        let mut writer = BufWriter::new(file);

        // Write header
        let header = CarHeader::new(roots);
        let header_bytes = header.to_cbor()?;
        let header_len = encode_varint(header_bytes.len() as u64);

        writer
            .write_all(&header_len)
            .await
            .map_err(|e| Error::Storage(format!("Failed to write header length: {e}")))?;
        writer
            .write_all(&header_bytes)
            .await
            .map_err(|e| Error::Storage(format!("Failed to write header: {e}")))?;

        let bytes_written = (header_len.len() + header_bytes.len()) as u64;

        Ok(Self {
            writer,
            blocks_written: 0,
            bytes_written,
        })
    }

    /// Write a block to the CAR file
    pub async fn write_block(&mut self, block: &Block) -> Result<()> {
        let cid_bytes = block.cid().to_bytes();
        let data = block.data();

        // Block format: varint(cid_len + data_len) | cid | data
        let block_len = cid_bytes.len() + data.len();
        let len_bytes = encode_varint(block_len as u64);

        self.writer
            .write_all(&len_bytes)
            .await
            .map_err(|e| Error::Storage(format!("Failed to write block length: {e}")))?;
        self.writer
            .write_all(&cid_bytes)
            .await
            .map_err(|e| Error::Storage(format!("Failed to write CID: {e}")))?;
        self.writer
            .write_all(data)
            .await
            .map_err(|e| Error::Storage(format!("Failed to write block data: {e}")))?;

        self.blocks_written += 1;
        self.bytes_written += (len_bytes.len() + block_len) as u64;

        Ok(())
    }

    /// Finish writing and close the file
    pub async fn finish(mut self) -> Result<CarWriteStats> {
        self.writer
            .flush()
            .await
            .map_err(|e| Error::Storage(format!("Failed to flush CAR file: {e}")))?;

        Ok(CarWriteStats {
            blocks_written: self.blocks_written,
            bytes_written: self.bytes_written,
        })
    }

    /// Get current statistics
    pub fn stats(&self) -> CarWriteStats {
        CarWriteStats {
            blocks_written: self.blocks_written,
            bytes_written: self.bytes_written,
        }
    }
}

/// Statistics from CAR writing
#[derive(Debug, Clone)]
pub struct CarWriteStats {
    pub blocks_written: u64,
    pub bytes_written: u64,
}

/// CAR file reader
pub struct CarReader {
    reader: BufReader<File>,
    header: CarHeader,
    blocks_read: u64,
    bytes_read: u64,
}

impl CarReader {
    /// Open a CAR file for reading
    pub async fn open(path: &Path) -> Result<Self> {
        let file = File::open(path)
            .await
            .map_err(|e| Error::Storage(format!("Failed to open CAR file: {e}")))?;

        let mut reader = BufReader::new(file);

        // Read header length (varint)
        let mut header_len_buf = [0u8; 10];
        let mut header_len_size = 0;

        for i in 0..10 {
            reader
                .read_exact(&mut header_len_buf[i..i + 1])
                .await
                .map_err(|e| Error::Storage(format!("Failed to read header length: {e}")))?;
            header_len_size = i + 1;
            if header_len_buf[i] & 0x80 == 0 {
                break;
            }
        }

        let (header_len, _) = decode_varint(&header_len_buf[..header_len_size])?;

        // Read header
        let mut header_bytes = vec![0u8; header_len as usize];
        reader
            .read_exact(&mut header_bytes)
            .await
            .map_err(|e| Error::Storage(format!("Failed to read header: {e}")))?;

        let header = CarHeader::from_cbor(&header_bytes)?;

        let bytes_read = (header_len_size + header_len as usize) as u64;

        Ok(Self {
            reader,
            header,
            blocks_read: 0,
            bytes_read,
        })
    }

    /// Get the CAR header
    pub fn header(&self) -> &CarHeader {
        &self.header
    }

    /// Get root CIDs
    pub fn roots(&self) -> &[Cid] {
        &self.header.roots
    }

    /// Read the next block from the CAR file
    pub async fn read_block(&mut self) -> Result<Option<Block>> {
        // Read block length (varint)
        let mut len_buf = [0u8; 10];
        let mut len_size = 0;

        #[allow(clippy::needless_range_loop)]
        for i in 0..10 {
            let mut byte_buf = [0u8; 1];
            match self.reader.read(&mut byte_buf).await {
                Ok(0) => {
                    if i == 0 {
                        return Ok(None); // End of file
                    }
                    return Err(Error::Storage("Incomplete block length".to_string()));
                }
                Ok(_) => {
                    len_buf[i] = byte_buf[0];
                }
                Err(e) => return Err(Error::Storage(format!("Failed to read block length: {e}"))),
            }
            len_size = i + 1;
            if len_buf[i] & 0x80 == 0 {
                break;
            }
        }

        let (block_len, _) = decode_varint(&len_buf[..len_size])?;

        // Read block data (CID + data)
        let mut block_data = vec![0u8; block_len as usize];
        self.reader
            .read_exact(&mut block_data)
            .await
            .map_err(|e| Error::Storage(format!("Failed to read block data: {e}")))?;

        // Parse CID from beginning of block data
        let cid = Cid::try_from(block_data.clone())
            .map_err(|e| Error::Cid(format!("Invalid CID in CAR: {e}")))?;

        let cid_len = cid.to_bytes().len();
        let data = Bytes::copy_from_slice(&block_data[cid_len..]);

        self.blocks_read += 1;
        self.bytes_read += (len_size + block_len as usize) as u64;

        Ok(Some(Block::from_parts(cid, data)))
    }

    /// Get read statistics
    pub fn stats(&self) -> CarReadStats {
        CarReadStats {
            blocks_read: self.blocks_read,
            bytes_read: self.bytes_read,
        }
    }
}

/// Statistics from CAR reading
#[derive(Debug, Clone)]
pub struct CarReadStats {
    pub blocks_read: u64,
    pub bytes_read: u64,
}

/// Export blocks to a CAR file
pub async fn export_to_car<S: BlockStore>(
    store: &S,
    path: &Path,
    roots: Vec<Cid>,
) -> Result<CarWriteStats> {
    let mut writer = CarWriter::create(path, roots.clone()).await?;

    // Get all CIDs reachable from roots
    let all_cids = store.list_cids()?;

    for cid in all_cids {
        if let Some(block) = store.get(&cid).await? {
            writer.write_block(&block).await?;
        }
    }

    writer.finish().await
}

/// Import blocks from a CAR file
pub async fn import_from_car<S: BlockStore>(store: &S, path: &Path) -> Result<CarReadStats> {
    let mut reader = CarReader::open(path).await?;

    while let Some(block) = reader.read_block().await? {
        store.put(&block).await?;
    }

    Ok(reader.stats())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blockstore::{BlockStoreConfig, SledBlockStore};

    fn make_test_block(data: &[u8]) -> Block {
        Block::new(Bytes::copy_from_slice(data)).expect("test data is valid block content")
    }

    #[test]
    fn test_varint_encode_decode() {
        let test_values = [0, 1, 127, 128, 255, 256, 16383, 16384, 1000000];

        for &val in &test_values {
            let encoded = encode_varint(val);
            let (decoded, _) = decode_varint(&encoded).expect("test: varint decode should succeed");
            assert_eq!(val, decoded, "Failed for value {}", val);
        }
    }

    #[test]
    fn test_car_header_roundtrip() {
        let block1 = make_test_block(b"test1");
        let block2 = make_test_block(b"test2");
        let roots = vec![*block1.cid(), *block2.cid()];

        let header = CarHeader::new(roots.clone());
        let cbor = header
            .to_cbor()
            .expect("test: header CBOR encoding should succeed");
        let decoded = CarHeader::from_cbor(&cbor).expect("test: CBOR decoding should succeed");

        assert_eq!(decoded.version, CAR_VERSION);
        assert_eq!(decoded.roots.len(), 2);
        assert_eq!(decoded.roots[0], roots[0]);
        assert_eq!(decoded.roots[1], roots[1]);
    }

    #[tokio::test]
    async fn test_car_write_read() {
        let path = std::env::temp_dir().join("test-car.car");
        let _ = std::fs::remove_file(&path);

        let block1 = make_test_block(b"hello world");
        let block2 = make_test_block(b"goodbye world");
        let roots = vec![*block1.cid()];

        // Write CAR
        {
            let mut writer = CarWriter::create(&path, roots.clone())
                .await
                .expect("test: CarWriter::create should succeed");
            writer
                .write_block(&block1)
                .await
                .expect("test: write block1 should succeed");
            writer
                .write_block(&block2)
                .await
                .expect("test: write block2 should succeed");
            let stats = writer.finish().await.expect("test: finish should succeed");
            assert_eq!(stats.blocks_written, 2);
        }

        // Read CAR
        {
            let mut reader = CarReader::open(&path)
                .await
                .expect("test: CarReader::open should succeed");
            assert_eq!(reader.roots().len(), 1);
            assert_eq!(reader.roots()[0], *block1.cid());

            let read_block1 = reader
                .read_block()
                .await
                .expect("test: read_block should succeed")
                .expect("test: first block should be present");
            assert_eq!(read_block1.cid(), block1.cid());
            assert_eq!(read_block1.data(), block1.data());

            let read_block2 = reader
                .read_block()
                .await
                .expect("test: read_block should succeed")
                .expect("test: second block should be present");
            assert_eq!(read_block2.cid(), block2.cid());
            assert_eq!(read_block2.data(), block2.data());

            // No more blocks
            assert!(reader
                .read_block()
                .await
                .expect("test: read_block at EOF should succeed")
                .is_none());
        }

        let _ = std::fs::remove_file(&path);
    }

    #[tokio::test]
    async fn test_export_import_car() {
        let store_path = std::env::temp_dir().join("ipfrs-test-car-store");
        let car_path = std::env::temp_dir().join("test-export.car");
        let _ = std::fs::remove_dir_all(&store_path);
        let _ = std::fs::remove_file(&car_path);

        let config = BlockStoreConfig {
            path: store_path.clone(),
            cache_size: 1024 * 1024,
        };
        let store = SledBlockStore::new(config).expect("test: SledBlockStore::new should succeed");

        // Add blocks
        let block1 = make_test_block(b"block1");
        let block2 = make_test_block(b"block2");
        store
            .put(&block1)
            .await
            .expect("test: put block1 should succeed");
        store
            .put(&block2)
            .await
            .expect("test: put block2 should succeed");

        // Export
        let write_stats = export_to_car(&store, &car_path, vec![*block1.cid()])
            .await
            .expect("test: export_to_car should succeed");
        assert_eq!(write_stats.blocks_written, 2);

        // Create new store and import
        let store_path2 = std::env::temp_dir().join("ipfrs-test-car-store2");
        let _ = std::fs::remove_dir_all(&store_path2);
        let config2 = BlockStoreConfig {
            path: store_path2.clone(),
            cache_size: 1024 * 1024,
        };
        let store2 = SledBlockStore::new(config2)
            .expect("test: SledBlockStore::new for store2 should succeed");

        let read_stats = import_from_car(&store2, &car_path)
            .await
            .expect("test: import_from_car should succeed");
        assert_eq!(read_stats.blocks_read, 2);

        // Verify blocks
        assert!(store2
            .has(block1.cid())
            .await
            .expect("test: has block1 should succeed"));
        assert!(store2
            .has(block2.cid())
            .await
            .expect("test: has block2 should succeed"));

        let _ = std::fs::remove_dir_all(&store_path);
        let _ = std::fs::remove_dir_all(&store_path2);
        let _ = std::fs::remove_file(&car_path);
    }
}
