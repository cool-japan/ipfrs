//! # `PnNegotiatorConfig` - Trait Implementations
//!
//! This module contains trait implementations for `PnNegotiatorConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::PnNegotiatorConfig;

impl Default for PnNegotiatorConfig {
    fn default() -> Self {
        Self {
            max_sessions: 1024,
            session_ttl_secs: 300,
            prefer_latest: true,
            strict_compat: false,
        }
    }
}
