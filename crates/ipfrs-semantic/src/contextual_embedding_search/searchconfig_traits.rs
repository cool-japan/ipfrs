//! # `SearchConfig` - Trait Implementations
//!
//! This module contains trait implementations for `SearchConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{DiversityStrategy, SearchConfig};

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            top_k: 10,
            diversity_strategy: DiversityStrategy::MaxMarginalRelevance(0.5),
            expansion_alpha: 0.3,
            use_negative_examples: true,
            rerank_top_n: 50,
            min_relevance: 0.0,
        }
    }
}
