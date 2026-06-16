//! # `TestError` - Trait Implementations
//!
//! This module contains trait implementations for `TestError`.
//!
//! ## Implemented Traits
//!
//! - `Display`
//! - `Error`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::TestError;

impl std::fmt::Display for TestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InsufficientData { needed, got } => {
                write!(f, "Insufficient data: need {needed}, got {got}")
            }
            Self::InvalidAlpha(a) => write!(f, "Invalid alpha {a}: must be in (0, 1)"),
            Self::NumericalError(msg) => write!(f, "Numerical error: {msg}"),
            Self::HypothesisNotFound(id) => write!(f, "Hypothesis not found: {id}"),
            Self::InvalidContingency(msg) => {
                write!(f, "Invalid contingency table: {msg}")
            }
        }
    }
}

impl std::error::Error for TestError {}
