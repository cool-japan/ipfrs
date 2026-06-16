//! # `NegotiatorConfig` - Trait Implementations
//!
//! This module contains trait implementations for `NegotiatorConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::NegotiatorConfig;

impl Default for NegotiatorConfig {
    fn default() -> Self {
        Self {
            local_min_version: 1,
            local_max_version: 3,
            required_features: Vec::new(),
            local_chunk_size: 65_536,
        }
    }
}
