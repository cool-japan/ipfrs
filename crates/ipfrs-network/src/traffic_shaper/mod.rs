//! Network traffic shaping with multiple queuing disciplines.
//!
//! Provides production-quality traffic shaping including FIFO, priority queuing,
//! weighted fair queuing, token bucket, leaky bucket, and DiffServ disciplines.
//! Also retains the original peer-level shaper types for backwards compatibility.

pub mod functions;
pub mod shapererror_traits;
pub mod types;
pub mod types_3;

// Re-export all types
pub use functions::*;
pub use types::*;
pub use types_3::*;

#[cfg(test)]
mod tests;
