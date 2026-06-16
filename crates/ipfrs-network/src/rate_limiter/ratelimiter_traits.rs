//! # `RateLimiter` - Trait Implementations
//!
//! This module contains trait implementations for `RateLimiter`.
//!
//! ## Implemented Traits
//!
//! - `Debug`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types_7::RateLimiter;

impl std::fmt::Debug for RateLimiter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RateLimiter")
            .field("default_capacity", &self.default_capacity)
            .field("default_rate", &self.default_rate)
            .finish()
    }
}
