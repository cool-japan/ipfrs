//! Temporal Pattern Matcher — NFA-based temporal sequence pattern matching.
//!
//! Provides production-quality matching of event streams against multi-step
//! temporal patterns with per-step timing constraints, repetition specs,
//! negation, and configurable NFA state management.

pub mod eventlabel_traits;
pub mod functions;
pub mod matcherconfig_traits;
pub mod types;

// Re-export all types
pub use functions::*;
pub use types::*;

#[cfg(test)]
mod tests;
