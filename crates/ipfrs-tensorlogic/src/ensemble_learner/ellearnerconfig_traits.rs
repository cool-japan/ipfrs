//! # `ElLearnerConfig` - Trait Implementations
//!
//! This module contains trait implementations for `ElLearnerConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{ElLearnerConfig, ElMethod};

impl Default for ElLearnerConfig {
    fn default() -> Self {
        Self {
            method: ElMethod::Bagging,
            n_estimators: 100,
            learning_rate: 0.1,
            max_depth: 1,
            seed: 42,
            subsample: 1.0,
        }
    }
}
