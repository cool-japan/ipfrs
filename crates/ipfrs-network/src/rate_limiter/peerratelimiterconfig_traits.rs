//! # `PeerRateLimiterConfig` - Trait Implementations
//!
//! This module contains trait implementations for `PeerRateLimiterConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::PeerRateLimiterConfig;

impl Default for PeerRateLimiterConfig {
    fn default() -> Self {
        Self {
            per_peer_max_tokens: 1_000,
            per_peer_refill_rate: 100,
            global_max_tokens: 10_000,
            global_refill_rate: 1_000,
            auto_block_threshold: 10,
        }
    }
}
