//! # `SvtTrackerConfig` - Trait Implementations
//!
//! This module contains trait implementations for `SvtTrackerConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::SvtTrackerConfig;

impl Default for SvtTrackerConfig {
    fn default() -> Self {
        Self {
            drift_threshold: 0.15,
            min_anchors: 1,
            window_size: 20,
            auto_deprecate: false,
        }
    }
}
