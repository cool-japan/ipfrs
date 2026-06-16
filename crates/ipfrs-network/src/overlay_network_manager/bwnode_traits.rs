//! # `BwNode` - Trait Implementations
//!
//! This module contains trait implementations for `BwNode`.
//!
//! ## Implemented Traits
//!
//! - `Eq`
//! - `Ord`
//! - `PartialOrd`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::cmp::Ordering;

use super::types::BwNode;

impl Eq for BwNode {}

impl Ord for BwNode {
    fn cmp(&self, other: &Self) -> Ordering {
        self.bw
            .partial_cmp(&other.bw)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.id.cmp(&self.id))
    }
}

impl PartialOrd for BwNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
