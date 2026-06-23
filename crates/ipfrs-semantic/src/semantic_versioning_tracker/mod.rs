//! Semantic drift tracker across model/embedding versions.
//!
//! [`SemanticVersioningTracker`] detects and quantifies how concept embeddings
//! shift between successive model versions, enabling data-driven migration
//! recommendations and compatibility analysis.

pub mod constants;
pub mod functions;
pub mod svterror_traits;
pub mod svttrackerconfig_traits;
pub mod type_aliases;
pub mod types;

// Re-export all types
pub use type_aliases::*;
pub use types::*;

#[cfg(test)]
mod tests;
