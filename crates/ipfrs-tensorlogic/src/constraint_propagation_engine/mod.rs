//! Constraint Propagation Engine for IPFRS TensorLogic
//!
//! Implements a full constraint propagation engine supporting:
//! - Arc consistency (AC3/AC4/AC6)
//! - Bounds propagation over integer intervals
//! - Interval arithmetic
//! - DFS backtracking with MRV (Minimum Remaining Values) heuristic
//! - AllDifferent, Linear, Sum, Abs, and relational constraints

pub mod cpeengineconfig_traits;
pub mod functions;
pub mod type_aliases;
pub mod types;
pub mod types_3;

// Re-export all types
pub use type_aliases::*;
pub use types::*;
pub use types_3::*;

#[cfg(test)]
mod tests;
