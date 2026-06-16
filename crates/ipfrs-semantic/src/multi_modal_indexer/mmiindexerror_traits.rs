//! # `MmiIndexError` - Trait Implementations
//!
//! This module contains trait implementations for `MmiIndexError`.
//!
//! ## Implemented Traits
//!
//! - `Display`
//! - `Error`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::fmt;

use super::types::MmiIndexError;

impl fmt::Display for MmiIndexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DocumentNotFound(id) => write!(f, "document not found: {id}"),
            Self::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
            Self::MaxDocumentsExceeded => write!(f, "maximum document limit exceeded"),
            Self::InvalidModality(name) => write!(f, "invalid modality: {name}"),
            Self::ConfigurationError(msg) => write!(f, "configuration error: {msg}"),
        }
    }
}

impl std::error::Error for MmiIndexError {}
