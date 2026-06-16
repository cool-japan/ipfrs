//! # `AtomicTokenBucket` - Trait Implementations
//!
//! This module contains trait implementations for `AtomicTokenBucket`.
//!
//! ## Implemented Traits
//!
//! - `Debug`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::AtomicTokenBucket;

impl std::fmt::Debug for AtomicTokenBucket {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AtomicTokenBucket")
            .field("capacity", &self.capacity)
            .field("tokens", &self.available_tokens())
            .field("refill_rate", &self.refill_rate)
            .finish()
    }
}
