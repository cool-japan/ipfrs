//! # `SolverConfig` - Trait Implementations
//!
//! This module contains trait implementations for `SolverConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{SolverConfig, SolverType};

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            gamma: 0.99,
            epsilon: 1e-6,
            max_iterations: 1000,
            solver: SolverType::ValueIteration,
        }
    }
}
