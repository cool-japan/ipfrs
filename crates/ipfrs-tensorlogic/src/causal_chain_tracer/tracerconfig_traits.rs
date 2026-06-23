//! # `TracerConfig` - Trait Implementations
//!
//! This module contains trait implementations for `TracerConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::TracerConfig;

impl Default for TracerConfig {
    fn default() -> Self {
        Self {
            max_chain_depth: 32,
            min_edge_strength: 0.0,
            max_nodes: 100_000,
            enable_cycle_detection: true,
            confidence_threshold: 0.0,
        }
    }
}
