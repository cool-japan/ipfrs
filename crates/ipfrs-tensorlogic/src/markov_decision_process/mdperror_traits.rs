//! # `MdpError` - Trait Implementations
//!
//! This module contains trait implementations for `MdpError`.
//!
//! ## Implemented Traits
//!
//! - `Display`
//! - `Error`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::MdpError;

impl std::fmt::Display for MdpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MdpError::StateOutOfRange(s) => write!(f, "state index {s} is out of range"),
            MdpError::ActionOutOfRange(a) => {
                write!(f, "action index {a} is out of range")
            }
            MdpError::InvalidProbability(p) => {
                write!(f, "probability {p} is not in the valid range [0, 1]")
            }
            MdpError::NoTransitions { state, action } => {
                write!(
                    f,
                    "no transitions defined for state={state}, action={action}"
                )
            }
        }
    }
}

impl std::error::Error for MdpError {}
