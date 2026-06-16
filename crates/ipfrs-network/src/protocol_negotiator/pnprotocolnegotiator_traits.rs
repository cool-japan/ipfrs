//! # `PnProtocolNegotiator` - Trait Implementations
//!
//! This module contains trait implementations for `PnProtocolNegotiator`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{PnNegotiatorConfig, PnProtocolNegotiator};

impl Default for PnProtocolNegotiator {
    fn default() -> Self {
        Self::new(PnNegotiatorConfig::default())
    }
}
