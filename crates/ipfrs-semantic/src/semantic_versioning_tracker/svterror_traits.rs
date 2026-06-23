//! # `SvtError` - Trait Implementations
//!
//! This module contains trait implementations for `SvtError`.
//!
//! ## Implemented Traits
//!
//! - `Display`
//! - `Error`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::SvtError;

impl std::fmt::Display for SvtError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SvtError::VersionNotFound(id) => write!(f, "version {id} not found"),
            SvtError::AnchorNotFound {
                concept,
                version_id,
            } => {
                write!(f, "anchor '{concept}' not found for version {version_id}")
            }
            SvtError::DimMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
            SvtError::InsufficientAnchors { found, required } => {
                write!(
                    f,
                    "insufficient anchors: {found} found, {required} required"
                )
            }
            SvtError::VersionDeprecated(id) => write!(f, "version {id} is deprecated"),
            SvtError::InvalidConcept => write!(f, "concept name must not be empty"),
        }
    }
}

impl std::error::Error for SvtError {}
