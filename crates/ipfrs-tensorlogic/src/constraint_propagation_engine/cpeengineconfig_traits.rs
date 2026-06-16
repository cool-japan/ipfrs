//! # `CpeEngineConfig` - Trait Implementations
//!
//! This module contains trait implementations for `CpeEngineConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{AcLevel, CpeEngineConfig};

impl Default for CpeEngineConfig {
    fn default() -> Self {
        Self {
            max_iterations: 10_000,
            arc_consistency_level: AcLevel::Ac3,
            use_bounds_propagation: true,
            fail_first: true,
        }
    }
}
