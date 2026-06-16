//! # `BuilderError` - Trait Implementations
//!
//! This module contains trait implementations for `BuilderError`.
//!
//! ## Implemented Traits
//!
//! - `Display`
//! - `Error`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::BuilderError;

impl std::fmt::Display for BuilderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BuilderError::NodeNotFound(id) => write!(f, "node not found: {id}"),
            BuilderError::DuplicateNode(id) => write!(f, "duplicate node id: {id}"),
            BuilderError::SelfLoop(id) => write!(f, "self-loop on node: {id}"),
            BuilderError::InvalidWeight(w) => write!(f, "invalid edge weight: {w}"),
            BuilderError::GraphTooLarge(n) => write!(f, "graph too large: {n} nodes"),
        }
    }
}

impl std::error::Error for BuilderError {}
