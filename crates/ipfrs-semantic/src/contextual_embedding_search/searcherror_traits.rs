//! # `SearchError` - Trait Implementations
//!
//! This module contains trait implementations for `SearchError`.
//!
//! ## Implemented Traits
//!
//! - `Display`
//! - `Error`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::SearchError;

impl std::fmt::Display for SearchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IndexEmpty => write!(f, "index is empty"),
            Self::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
            Self::InsufficientResults(n) => write!(f, "only {n} results available"),
            Self::ConfigurationError(msg) => write!(f, "configuration error: {msg}"),
        }
    }
}

impl std::error::Error for SearchError {}
