//! # `OverlayConfig` - Trait Implementations
//!
//! This module contains trait implementations for `OverlayConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{OverlayConfig, OverlayTopology, RoutingPolicy};

impl Default for OverlayConfig {
    fn default() -> Self {
        Self {
            topology: OverlayTopology::Custom,
            max_nodes: 1024,
            max_hops: 32,
            routing_policy: RoutingPolicy::ShortestPath,
            region_aware: false,
        }
    }
}
