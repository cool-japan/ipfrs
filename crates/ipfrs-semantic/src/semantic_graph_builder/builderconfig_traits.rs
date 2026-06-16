//! # `BuilderConfig` - Trait Implementations
//!
//! This module contains trait implementations for `BuilderConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::BuilderConfig;

impl Default for BuilderConfig {
    fn default() -> Self {
        Self {
            similarity_threshold: 0.85,
            max_edges_per_node: 50,
            enable_transitive_closure: false,
            cooccurrence_window: 5,
        }
    }
}
