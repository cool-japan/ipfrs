//! # `OptimizerError` - Trait Implementations
//!
//! This module contains trait implementations for `OptimizerError`.
//!
//! ## Implemented Traits
//!
//! - `Display`
//! - `Error`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::OptimizerError;

impl std::fmt::Display for OptimizerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ParseError(m) => write!(f, "ParseError: {m}"),
            Self::InvalidQuery(m) => write!(f, "InvalidQuery: {m}"),
            Self::OptimizationFailed(m) => write!(f, "OptimizationFailed: {m}"),
            Self::ConfigurationError(m) => write!(f, "ConfigurationError: {m}"),
        }
    }
}

impl std::error::Error for OptimizerError {}
