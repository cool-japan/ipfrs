//! Virtual overlay network topology manager.
//!
//! This module provides [`OverlayNetworkManager`], which models a logical
//! virtual network layered on top of physical peer connections.  Unlike the
//! simpler [`crate::overlay_network`] module (which uses numeric node IDs and
//! DHT-style routing), this implementation works with human-readable string IDs
//! and virtual IP-style addresses, supports multiple topology templates
//! (FullMesh, Ring, Star, Hypercube, Custom), and provides Dijkstra-based
//! multi-policy routing as well as k-shortest paths via Yen's algorithm.
//!
//! # Design goals
//! - Zero I/O, zero `unwrap()`.
//! - Deterministic PRNG (xorshift64) and FNV-1a hashing for internal use.
//! - All public methods return `Result<_, OverlayError>` or a plain value.
//! - Union-find for connected-components, Dijkstra for single-source routing,
//!   Yen's algorithm for k-shortest paths.

pub mod bwnode_traits;
pub mod dijknode_traits;
pub mod functions;
pub mod overlayconfig_traits;
pub mod types;

// Re-export all types
pub use functions::*;
pub use types::*;

#[cfg(test)]
mod tests;
