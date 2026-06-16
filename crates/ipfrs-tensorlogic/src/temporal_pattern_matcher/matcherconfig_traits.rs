//! # `MatcherConfig` - Trait Implementations
//!
//! This module contains trait implementations for `MatcherConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::MatcherConfig;

impl Default for MatcherConfig {
    fn default() -> Self {
        Self {
            max_window_us: 10_000_000,
            max_events_buffered: 4096,
            enable_overlapping_matches: true,
        }
    }
}
