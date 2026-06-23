//! # `TracerError` - Trait Implementations
//!
//! This module contains trait implementations for `TracerError`.
//!
//! ## Implemented Traits
//!
//! - `Display`
//! - `Error`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::TracerError;

impl std::fmt::Display for TracerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NodeNotFound(id) => write!(f, "node not found: {id}"),
            Self::CycleDetected { path } => {
                write!(f, "cycle detected: {}", path.join(" -> "))
            }
            Self::QueryTooExpensive(n) => write!(f, "query too expensive: {n} nodes"),
            Self::InvalidStrength(s) => {
                write!(f, "invalid strength {s}: must be in [0.0, 1.0]")
            }
            Self::MaxDepthExceeded => write!(f, "max depth exceeded"),
        }
    }
}

impl std::error::Error for TracerError {}
