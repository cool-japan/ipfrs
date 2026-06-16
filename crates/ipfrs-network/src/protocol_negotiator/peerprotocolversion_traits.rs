//! # `PeerProtocolVersion` - Trait Implementations
//!
//! This module contains trait implementations for `PeerProtocolVersion`.
//!
//! ## Implemented Traits
//!
//! - `Display`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::PeerProtocolVersion;

impl std::fmt::Display for PeerProtocolVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/{}.{}", self.name, self.major, self.minor)
    }
}
