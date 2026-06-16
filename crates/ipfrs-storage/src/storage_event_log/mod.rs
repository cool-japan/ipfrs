//! Production-grade structured event log for storage operations.
//!
//! Provides append-only, queryable, FNV-1a-checksummed event records covering
//! the full lifecycle of objects: creation, reads, updates, deletions, copies,
//! moves, versioning, compression, encryption, tier migrations, batch
//! boundaries, and system errors.  Supports rich queries (by type, object,
//! user, time range, correlation), per-type aggregation, integrity
//! verification, and multiple retention policies.

pub mod eventid_traits;
pub mod eventlogconfig_traits;
pub mod eventlogerror_traits;
pub mod functions;
pub mod types;

// Re-export all types
pub use functions::*;
pub use types::*;

#[cfg(test)]
mod tests;
