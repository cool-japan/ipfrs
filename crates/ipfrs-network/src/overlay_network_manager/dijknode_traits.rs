//! # `DijkNode` - Trait Implementations
//!
//! This module contains trait implementations for `DijkNode`.
//!
//! ## Implemented Traits
//!
//! - `Eq`
//! - `Ord`
//! - `PartialOrd`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::cmp::Ordering;

use super::types::DijkNode;

impl Eq for DijkNode {}

impl Ord for DijkNode {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .cost
            .partial_cmp(&self.cost)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.id.cmp(&other.id))
    }
}

impl PartialOrd for DijkNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
