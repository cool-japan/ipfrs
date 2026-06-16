//! # `OptimizerConfig` - Trait Implementations
//!
//! This module contains trait implementations for `OptimizerConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::HashMap;

use super::types::OptimizationRule;
use super::types_4::{IndexHints, OptimizerConfig};

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            max_terms: 64,
            max_embedding_dim: 4096,
            apply_rules: vec![
                OptimizationRule::ConstantFolding,
                OptimizationRule::DeduplicateTerms,
                OptimizationRule::FlattenNested,
                OptimizationRule::PushDownFilters,
                OptimizationRule::ReorderBySelectivity,
                OptimizationRule::EmbeddingCaching,
            ],
            index_stats: IndexHints {
                total_docs: 100_000,
                term_frequencies: HashMap::new(),
                avg_doc_length: 100.0,
            },
        }
    }
}
