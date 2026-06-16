//! # `EngineConfig` - Trait Implementations
//!
//! This module contains trait implementations for `EngineConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::EngineConfig;

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            default_alpha: 0.05,
            min_sample_size: 2,
            enable_power_calculation: true,
        }
    }
}
