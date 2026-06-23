//! # `TraceQuery` - Trait Implementations
//!
//! This module contains trait implementations for `TraceQuery`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::TraceQuery;

impl Default for TraceQuery {
    fn default() -> Self {
        Self {
            root_event_id: None,
            event_types: Vec::new(),
            max_depth: 16,
            min_strength: 0.0,
            time_window_us: None,
            include_relations: Vec::new(),
        }
    }
}
