//! Error types for IPFRS operations.
//!
//! This module provides a unified error type ([`enum@Error`]) and result alias
//! ([`Result`]) used throughout the IPFRS crate family.
//!
//! # Error Categories
//!
//! Errors are categorized by their source:
//! - **I/O errors** - File system and network I/O failures
//! - **Data errors** - Invalid blocks, CIDs, or serialization issues
//! - **Not found** - Missing blocks, peers, or resources
//! - **Protocol errors** - IPFS protocol violations
//!
//! # Example
//!
//! ```rust
//! use ipfrs_core::{Error, Result, Block};
//! use bytes::Bytes;
//!
//! fn process_block(data: &[u8]) -> Result<Block> {
//!     if data.is_empty() {
//!         return Err(Error::InvalidInput("empty data".to_string()));
//!     }
//!     Block::new(Bytes::copy_from_slice(data))
//! }
//! ```

use thiserror::Error;

/// Convenient result type for IPFRS operations.
pub type Result<T> = std::result::Result<T, Error>;

/// Unified error type for IPFRS operations.
///
/// This enum captures all error conditions that can occur in IPFRS,
/// providing detailed context through error messages.
#[derive(Debug, Error)]
pub enum Error {
    /// File system or network I/O error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Requested block was not found in storage
    #[error("Block not found: {0}")]
    BlockNotFound(String),

    /// CID parsing, generation, or validation error
    #[error("CID error: {0}")]
    Cid(String),

    /// Data serialization error (CBOR, JSON, etc.)
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// Data deserialization error (CBOR, JSON, etc.)
    #[error("Deserialization error: {0}")]
    Deserialization(String),

    /// Network communication error
    #[error("Network error: {0}")]
    Network(String),

    /// Storage backend error (disk, memory, S3, etc.)
    #[error("Storage error: {0}")]
    Storage(String),

    /// Encryption or decryption error
    #[error("Encryption error: {0}")]
    Encryption(String),

    /// Data validation error (malformed blocks, invalid sizes, etc.)
    #[error("Invalid data: {0}")]
    InvalidData(String),

    /// Invalid user input or parameters
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Requested resource not found
    #[error("Not found: {0}")]
    NotFound(String),

    /// IPFS protocol violation or incompatibility
    #[error("Protocol error: {0}")]
    Protocol(String),

    /// Feature not yet implemented
    #[error("Not implemented: {0}")]
    NotImplemented(String),

    /// Unexpected internal error (possible bug)
    #[error("Internal error: {0}")]
    Internal(String),

    /// System initialization or setup error
    #[error("Initialization error: {0}")]
    Initialization(String),

    /// Signature or cryptographic verification error
    #[error("Verification error: {0}")]
    Verification(String),
}

impl Error {
    /// Check if this is an I/O error
    #[inline]
    pub const fn is_io(&self) -> bool {
        matches!(self, Error::Io(_))
    }

    /// Check if this is a block not found error
    #[inline]
    pub const fn is_not_found(&self) -> bool {
        matches!(self, Error::BlockNotFound(_) | Error::NotFound(_))
    }

    /// Check if this is a serialization/deserialization error
    #[inline]
    pub const fn is_serialization(&self) -> bool {
        matches!(self, Error::Serialization(_) | Error::Deserialization(_))
    }

    /// Check if this is a network error
    #[inline]
    pub const fn is_network(&self) -> bool {
        matches!(self, Error::Network(_))
    }

    /// Check if this is a storage error
    #[inline]
    pub const fn is_storage(&self) -> bool {
        matches!(self, Error::Storage(_))
    }

    /// Check if this is a validation error
    #[inline]
    pub const fn is_validation(&self) -> bool {
        matches!(self, Error::InvalidData(_) | Error::InvalidInput(_))
    }

    /// Check if this is a CID-related error
    #[inline]
    pub const fn is_cid(&self) -> bool {
        matches!(self, Error::Cid(_))
    }

    /// Check if this is a verification error
    #[inline]
    pub const fn is_verification(&self) -> bool {
        matches!(self, Error::Verification(_))
    }

    /// Get a human-readable error category name
    pub const fn category(&self) -> &'static str {
        match self {
            Error::Io(_) => "io",
            Error::BlockNotFound(_) => "not_found",
            Error::Cid(_) => "cid",
            Error::Serialization(_) => "serialization",
            Error::Deserialization(_) => "deserialization",
            Error::Network(_) => "network",
            Error::Storage(_) => "storage",
            Error::Encryption(_) => "encryption",
            Error::InvalidData(_) => "invalid_data",
            Error::InvalidInput(_) => "invalid_input",
            Error::NotFound(_) => "not_found",
            Error::Protocol(_) => "protocol",
            Error::NotImplemented(_) => "not_implemented",
            Error::Internal(_) => "internal",
            Error::Initialization(_) => "initialization",
            Error::Verification(_) => "verification",
        }
    }

    /// Check if this error is recoverable (retrying might help)
    pub const fn is_recoverable(&self) -> bool {
        matches!(self, Error::Io(_) | Error::Network(_) | Error::Storage(_))
    }

    /// Check if this error indicates a bug or unexpected condition
    pub const fn is_internal(&self) -> bool {
        matches!(self, Error::Internal(_))
    }
}
