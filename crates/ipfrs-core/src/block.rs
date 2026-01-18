//! Content-addressed data blocks.
//!
//! This module provides the [`Block`] type, which represents a piece of data
//! along with its content identifier ([`Cid`]). Blocks are the fundamental
//! storage unit in IPFS-style systems.
//!
//! # Example
//!
//! ```rust
//! use ipfrs_core::Block;
//! use bytes::Bytes;
//!
//! // Create a block from data
//! let block = Block::new(Bytes::from_static(b"Hello!")).unwrap();
//!
//! // Access the CID and data
//! println!("CID: {}", block.cid());
//! assert_eq!(block.data().as_ref(), b"Hello!");
//!
//! // Verify the block's integrity
//! assert!(block.verify().unwrap());
//! ```
//!
//! # Size Limits
//!
//! Blocks have size constraints for network efficiency:
//! - Minimum: 1 byte ([`MIN_BLOCK_SIZE`])
//! - Maximum: 2 MiB ([`MAX_BLOCK_SIZE`])

use crate::cid::{Cid, CidBuilder, HashAlgorithm};
use crate::error::{Error, Result};
use crate::types::BlockSize;
use bytes::Bytes;
use serde::{Deserialize, Serialize};

/// Maximum block size in bytes (2 MiB, same as IPFS default)
pub const MAX_BLOCK_SIZE: usize = 2 * 1024 * 1024;

/// Minimum block size in bytes
pub const MIN_BLOCK_SIZE: usize = 1;

/// A content-addressed data block.
///
/// A `Block` consists of raw data and its content identifier (CID).
/// The CID is cryptographically derived from the data, ensuring
/// that blocks can be verified independently.
///
/// # Creating Blocks
///
/// Use [`Block::new`] for simple block creation with default settings,
/// or [`Block::builder`] for custom hash algorithms or codecs.
///
/// # Thread Safety
///
/// `Block` is `Clone` and can be safely shared across threads.
/// The underlying data uses reference counting ([`Bytes`]).
#[derive(Debug, Clone)]
pub struct Block {
    /// Content identifier
    cid: Cid,
    /// Raw data
    data: Bytes,
}

impl Block {
    /// Create a new block with computed CID using default settings
    ///
    /// # Errors
    /// Returns an error if the block size is outside the valid range
    pub fn new(data: Bytes) -> Result<Self> {
        Self::validate_size(data.len())?;
        let cid = CidBuilder::new().build(&data)?;
        Ok(Self { cid, data })
    }

    /// Validate block size
    fn validate_size(size: usize) -> Result<()> {
        if size < MIN_BLOCK_SIZE {
            return Err(Error::InvalidData(format!(
                "Block size {} is below minimum {}",
                size, MIN_BLOCK_SIZE
            )));
        }
        if size > MAX_BLOCK_SIZE {
            return Err(Error::InvalidData(format!(
                "Block size {} exceeds maximum {} (2 MiB)",
                size, MAX_BLOCK_SIZE
            )));
        }
        Ok(())
    }

    /// Create a block from existing CID and data (without verification)
    pub fn from_parts(cid: Cid, data: Bytes) -> Self {
        Self { cid, data }
    }

    /// Get the CID of this block
    #[inline]
    pub fn cid(&self) -> &Cid {
        &self.cid
    }

    /// Get the data of this block
    #[inline]
    pub fn data(&self) -> &Bytes {
        &self.data
    }

    /// Get the size of this block in bytes
    #[inline]
    pub fn size(&self) -> BlockSize {
        self.data.len() as BlockSize
    }

    /// Verify that the CID matches the data
    pub fn verify(&self) -> Result<bool> {
        let computed_cid = CidBuilder::new().build(&self.data)?;
        Ok(computed_cid == self.cid)
    }

    /// Consume the block and return its parts
    #[inline]
    pub fn into_parts(self) -> (Cid, Bytes) {
        (self.cid, self.data)
    }

    /// Get a zero-copy slice of the block data
    ///
    /// This creates a new `Bytes` instance that references the same
    /// underlying data without copying. The range must be within bounds.
    ///
    /// # Arguments
    ///
    /// * `range` - The byte range to slice (e.g., `0..100` or `10..`)
    ///
    /// # Panics
    ///
    /// Panics if the range is out of bounds.
    ///
    /// # Example
    ///
    /// ```rust
    /// use ipfrs_core::Block;
    /// use bytes::Bytes;
    ///
    /// let block = Block::new(Bytes::from_static(b"Hello, World!")).unwrap();
    /// let slice = block.slice(0..5);  // "Hello"
    /// assert_eq!(slice.as_ref(), b"Hello");
    /// ```
    pub fn slice(&self, range: impl std::ops::RangeBounds<usize>) -> Bytes {
        use std::ops::Bound;

        let len = self.data.len();
        let start = match range.start_bound() {
            Bound::Included(&n) => n,
            Bound::Excluded(&n) => n + 1,
            Bound::Unbounded => 0,
        };
        let end = match range.end_bound() {
            Bound::Included(&n) => n + 1,
            Bound::Excluded(&n) => n,
            Bound::Unbounded => len,
        };

        self.data.slice(start..end)
    }

    /// Get a zero-copy view of the entire data as a byte slice
    ///
    /// This provides direct access to the underlying bytes without
    /// any allocation or copying.
    ///
    /// # Example
    ///
    /// ```rust
    /// use ipfrs_core::Block;
    /// use bytes::Bytes;
    ///
    /// let block = Block::new(Bytes::from_static(b"data")).unwrap();
    /// let view = block.as_bytes();
    /// assert_eq!(view, b"data");
    /// ```
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    /// Clone the underlying `Bytes` reference (zero-copy)
    ///
    /// Since `Bytes` uses reference counting, cloning is cheap and
    /// does not copy the underlying data.
    ///
    /// # Example
    ///
    /// ```rust
    /// use ipfrs_core::Block;
    /// use bytes::Bytes;
    ///
    /// let block = Block::new(Bytes::from_static(b"data")).unwrap();
    /// let data_clone = block.clone_data();
    /// // data_clone shares the same underlying memory
    /// ```
    #[inline]
    pub fn clone_data(&self) -> Bytes {
        self.data.clone()
    }

    /// Check if two blocks share the same underlying data buffer
    ///
    /// Returns `true` if both blocks reference the same memory,
    /// even if they have different CIDs (which shouldn't happen
    /// in normal usage).
    #[inline]
    #[must_use]
    pub fn shares_data(&self, other: &Block) -> bool {
        self.data.as_ptr() == other.data.as_ptr()
    }

    /// Get metadata about the block
    pub fn metadata(&self) -> BlockMetadata {
        BlockMetadata::new(self.cid, self.size())
    }

    /// Create a builder for constructing blocks with custom settings
    pub fn builder() -> BlockBuilder {
        BlockBuilder::new()
    }

    /// Get the length of the block data in bytes
    ///
    /// This is an alias for [`Block::size`] for better Rust convention compliance.
    ///
    /// # Example
    ///
    /// ```rust
    /// use ipfrs_core::Block;
    /// use bytes::Bytes;
    ///
    /// let block = Block::new(Bytes::from_static(b"data")).unwrap();
    /// assert_eq!(block.len(), 4);
    /// ```
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if the block is empty
    ///
    /// Note: Due to the MIN_BLOCK_SIZE constraint, blocks created through
    /// [`Block::new`] can never be empty. This method is provided for
    /// completeness and for blocks created through other means.
    ///
    /// # Example
    ///
    /// ```rust
    /// use ipfrs_core::Block;
    /// use bytes::Bytes;
    ///
    /// let block = Block::new(Bytes::from_static(b"data")).unwrap();
    /// assert!(!block.is_empty());
    /// ```
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Create a block from a byte vector with default settings
    ///
    /// This is a convenience method that takes ownership of the vector.
    ///
    /// # Errors
    ///
    /// Returns an error if the size is outside the valid range.
    ///
    /// # Example
    ///
    /// ```rust
    /// use ipfrs_core::Block;
    ///
    /// let data = vec![1, 2, 3, 4];
    /// let block = Block::from_vec(data).unwrap();
    /// assert_eq!(block.len(), 4);
    /// ```
    pub fn from_vec(vec: Vec<u8>) -> Result<Self> {
        Self::new(Bytes::from(vec))
    }

    /// Create a block from a byte slice with default settings
    ///
    /// This copies the slice data.
    ///
    /// # Errors
    ///
    /// Returns an error if the size is outside the valid range.
    ///
    /// # Example
    ///
    /// ```rust
    /// use ipfrs_core::Block;
    ///
    /// let block = Block::from_slice(b"hello").unwrap();
    /// assert_eq!(block.as_bytes(), b"hello");
    /// ```
    pub fn from_slice(slice: &[u8]) -> Result<Self> {
        Self::new(Bytes::copy_from_slice(slice))
    }

    /// Create a block from a static byte slice (zero-copy)
    ///
    /// This is a zero-copy operation for static data.
    ///
    /// # Errors
    ///
    /// Returns an error if the size is outside the valid range.
    ///
    /// # Example
    ///
    /// ```rust
    /// use ipfrs_core::Block;
    ///
    /// let block = Block::from_static(b"static data").unwrap();
    /// assert_eq!(block.len(), 11);
    /// ```
    pub fn from_static(slice: &'static [u8]) -> Result<Self> {
        Self::new(Bytes::from_static(slice))
    }
}

/// Builder for creating blocks with custom CID settings.
///
/// Use this when you need to specify a custom hash algorithm,
/// CID version, or codec.
///
/// # Example
///
/// ```rust
/// use ipfrs_core::{Block, HashAlgorithm, cid::codec};
/// use bytes::Bytes;
///
/// let block = Block::builder()
///     .hash_algorithm(HashAlgorithm::Sha3_256)
///     .codec(codec::DAG_CBOR)
///     .build(Bytes::from_static(b"data"))
///     .unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct BlockBuilder {
    cid_builder: CidBuilder,
}

impl Default for BlockBuilder {
    fn default() -> Self {
        Self {
            cid_builder: CidBuilder::new(),
        }
    }
}

impl BlockBuilder {
    /// Creates a new `BlockBuilder` with default settings.
    ///
    /// Uses SHA2-256 hash algorithm and CIDv1 by default.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the hash algorithm to use
    pub fn hash_algorithm(mut self, algorithm: HashAlgorithm) -> Self {
        self.cid_builder = self.cid_builder.hash_algorithm(algorithm);
        self
    }

    /// Set the CID version
    pub fn cid_version(mut self, version: cid::Version) -> Self {
        self.cid_builder = self.cid_builder.version(version);
        self
    }

    /// Set the codec
    pub fn codec(mut self, codec: u64) -> Self {
        self.cid_builder = self.cid_builder.codec(codec);
        self
    }

    /// Build a block from data
    pub fn build(self, data: Bytes) -> Result<Block> {
        Block::validate_size(data.len())?;
        let cid = self.cid_builder.build(&data)?;
        Ok(Block { cid, data })
    }

    /// Build a block from data slice (convenience method)
    pub fn build_from_slice(self, data: &[u8]) -> Result<Block> {
        self.build(Bytes::copy_from_slice(data))
    }
}

impl From<&Block> for Cid {
    fn from(block: &Block) -> Self {
        block.cid
    }
}

impl AsRef<[u8]> for Block {
    fn as_ref(&self) -> &[u8] {
        self.as_bytes()
    }
}

impl std::ops::Deref for Block {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        self.as_bytes()
    }
}

impl TryFrom<Vec<u8>> for Block {
    type Error = Error;

    fn try_from(vec: Vec<u8>) -> Result<Self> {
        Block::from_vec(vec)
    }
}

impl TryFrom<&[u8]> for Block {
    type Error = Error;

    fn try_from(slice: &[u8]) -> Result<Self> {
        Block::from_slice(slice)
    }
}

impl TryFrom<Bytes> for Block {
    type Error = Error;

    fn try_from(bytes: Bytes) -> Result<Self> {
        Block::new(bytes)
    }
}

impl PartialEq for Block {
    /// Two blocks are equal if they have the same CID.
    ///
    /// Since CIDs are derived from data, this also implies equal data
    /// (assuming no hash collisions).
    fn eq(&self, other: &Self) -> bool {
        self.cid == other.cid
    }
}

impl Eq for Block {}

impl std::hash::Hash for Block {
    /// Hash based on the CID for use in HashMaps and HashSets.
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.cid.to_bytes().hash(state);
    }
}

impl PartialOrd for Block {
    /// Compare blocks by their CID.
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Block {
    /// Compare blocks by their CID for use in BTreeMaps and BTreeSets.
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.cid.to_bytes().cmp(&other.cid.to_bytes())
    }
}

/// Metadata about a block for indexing and queries.
///
/// This lightweight structure contains summary information about a block
/// without holding the actual data. Useful for block indices and caches.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockMetadata {
    /// Content identifier of the block
    pub cid: crate::cid::SerializableCid,
    /// Size of the block data in bytes
    pub size: BlockSize,
    /// Number of CID links to other blocks (for DAG nodes)
    pub links: usize,
}

impl BlockMetadata {
    /// Creates new block metadata with the given CID and size.
    ///
    /// The number of links is initialized to 0.
    pub fn new(cid: Cid, size: BlockSize) -> Self {
        Self {
            cid: crate::cid::SerializableCid(cid),
            size,
            links: 0,
        }
    }

    /// Sets the number of CID links in this block.
    ///
    /// This is useful for DAG nodes that reference other blocks.
    pub fn with_links(mut self, links: usize) -> Self {
        self.links = links;
        self
    }
}
