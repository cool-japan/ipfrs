//! # `EventLogConfig` - Trait Implementations
//!
//! This module contains trait implementations for `EventLogConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{EventLogConfig, RetentionPolicy};

impl Default for EventLogConfig {
    fn default() -> Self {
        Self {
            max_events: 100_000,
            retention_policy: RetentionPolicy::KeepLast(10_000),
            enable_checksums: true,
            batch_flush_size: 1_000,
        }
    }
}
