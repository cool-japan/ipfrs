//! # `EventLogError` - Trait Implementations
//!
//! This module contains trait implementations for `EventLogError`.
//!
//! ## Implemented Traits
//!
//! - `Display`
//! - `Error`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::EventLogError;

impl std::fmt::Display for EventLogError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EventLogError::EventNotFound(id) => write!(f, "event not found: {id}"),
            EventLogError::QueryTooExpensive { estimated } => {
                write!(f, "query too expensive (estimated scan: {estimated})")
            }
            EventLogError::RetentionError(msg) => write!(f, "retention error: {msg}"),
            EventLogError::CorruptedEvent(id) => write!(f, "corrupted event: {id}"),
        }
    }
}

impl std::error::Error for EventLogError {}
