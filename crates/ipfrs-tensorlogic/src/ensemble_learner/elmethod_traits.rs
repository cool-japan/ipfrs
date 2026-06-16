//! # `ElMethod` - Trait Implementations
//!
//! This module contains trait implementations for `ElMethod`.
//!
//! ## Implemented Traits
//!
//! - `Display`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::ElMethod;

impl std::fmt::Display for ElMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ElMethod::Bagging => write!(f, "Bagging"),
            ElMethod::AdaBoost => write!(f, "AdaBoost"),
            ElMethod::GradientBoosting => write!(f, "GradientBoosting"),
            ElMethod::RandomForest => write!(f, "RandomForest"),
            ElMethod::Stacking => write!(f, "Stacking"),
        }
    }
}
