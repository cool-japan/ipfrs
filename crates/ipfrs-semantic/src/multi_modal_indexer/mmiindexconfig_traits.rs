//! # `MmiIndexConfig` - Trait Implementations
//!
//! This module contains trait implementations for `MmiIndexConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::MmiIndexConfig;

impl Default for MmiIndexConfig {
    fn default() -> Self {
        Self {
            enable_text_index: true,
            enable_vector_index: true,
            enable_structured_index: true,
            vector_dim: None,
            text_similarity_threshold: 0.0,
            vector_similarity_threshold: 0.0,
            max_documents: usize::MAX,
        }
    }
}
