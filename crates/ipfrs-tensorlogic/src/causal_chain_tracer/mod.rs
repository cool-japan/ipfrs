//! Causal Chain Tracer — production-quality causal chain tracing for event sequences.
//!
//! This module provides:
//! - Directed causal graph with configurable relation types
//! - BFS/DFS chain tracing with depth, strength, time-window, and relation filters
//! - Shortest-path (hop count) and strongest-path (max product of edge strengths)
//! - Root-cause identification and downstream-effect enumeration
//! - DFS-based cycle detection on edge insertion

pub mod functions;
pub mod tracequery_traits;
pub mod tracerconfig_traits;
pub mod tracererror_traits;
pub mod types;

// Re-export all types
pub use functions::*;
pub use types::*;

#[cfg(test)]
mod tests;
