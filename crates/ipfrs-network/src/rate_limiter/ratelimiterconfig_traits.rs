//! # `RateLimiterConfig` - Trait Implementations
//!
//! This module contains trait implementations for `RateLimiterConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::Duration;

use super::types::RateLimiterConfig;

impl Default for RateLimiterConfig {
    fn default() -> Self {
        Self {
            max_rate: 10.0,
            burst_size: 20,
            enable_per_peer_limits: true,
            max_per_peer_rate: 2.0,
            enable_adaptive: false,
            adaptive_factor: 0.1,
            min_rate: 1.0,
            max_adaptive_rate: 100.0,
            enable_queuing: true,
            max_queue_size: 100,
            peer_window: Duration::from_secs(60),
        }
    }
}
