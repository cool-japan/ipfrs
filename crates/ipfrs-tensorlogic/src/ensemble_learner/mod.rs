//! EnsembleLearner — production-quality ensemble learning system.
//!
//! Implements Bagging, AdaBoost, Gradient Boosting, Random Forest, and Stacking
//! strategies over decision stumps and perceptron base models.
//!
//! # Design
//!
//! - Bagging: bootstrap samples, parallel independent base models, majority/average vote.
//! - AdaBoost: sequential stump fitting, exponential weight update (`alpha`), sample
//!   re-weighting after each round.
//! - Gradient Boosting: pseudo-residual fitting, shrinkage via `learning_rate`.
//! - Random Forest: bagging + feature sub-sampling per stump (sqrt(n_features)).
//! - Stacking: train diverse base models and a linear meta-learner on their outputs.
//! - All operations use `xorshift64` for bootstrap/sub-sampling — no `rand` crate.
//! - Training history is kept in a `VecDeque<ElTrainingRecord>` capped at 100 entries.
//! - No `unwrap()` anywhere — all fallible operations use `?` / `ok_or`.

pub mod ellearnerconfig_traits;
pub mod elmethod_traits;
pub mod functions;
pub mod type_aliases;
pub mod types;

// Re-export all types
pub use type_aliases::*;
pub use types::*;

#[cfg(test)]
mod tests;
