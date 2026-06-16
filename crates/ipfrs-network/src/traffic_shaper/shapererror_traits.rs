//! # `ShaperError` - Trait Implementations
//!
//! This module contains trait implementations for `ShaperError`.
//!
//! ## Implemented Traits
//!
//! - `Display`
//! - `Error`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::ShaperError;

impl std::fmt::Display for ShaperError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ShaperError::QueueFull(d) => write!(f, "queue full at depth {d}"),
            ShaperError::RateLimitExceeded(r) => {
                write!(f, "rate limit exceeded ({r} bps)")
            }
            ShaperError::InvalidConfig(msg) => write!(f, "invalid config: {msg}"),
            ShaperError::EntryNotFound(id) => write!(f, "entry {id} not found"),
        }
    }
}

impl std::error::Error for ShaperError {}
