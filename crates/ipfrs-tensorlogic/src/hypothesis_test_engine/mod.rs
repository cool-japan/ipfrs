//! Statistical Hypothesis Testing Engine for Logical Assertions
//!
//! Provides production-quality statistical hypothesis testing with:
//! - One-sample and two-sample z-tests and t-tests
//! - Chi-square goodness-of-fit and independence tests
//! - Proportion tests (one-sample and two-sample)
//! - Confidence interval computation
//! - Effect size (Cohen's d, Cramér's V)
//! - Monte Carlo power simulation via xorshift64 PRNG

pub mod engineconfig_traits;
pub mod functions;
pub mod hypothesistestengine_traits;
pub mod testerror_traits;
pub mod types;

// Re-export all types
pub use functions::*;
pub use types::*;

#[cfg(test)]
mod tests;
