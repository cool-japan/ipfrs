//! Connection rate limiting for preventing connection storms and resource exhaustion.
//!
//! This module provides sophisticated rate limiting capabilities to control the rate
//! of connection establishment, preventing connection storms, respecting peer limits,
//! and protecting against resource exhaustion.
//!
//! # Features
//!
//! - **Token Bucket Algorithm**: Classic token bucket with configurable rate and burst
//! - **Per-Peer Limits**: Individual rate limits for each peer
//! - **Global Limits**: System-wide connection rate limits
//! - **Priority-based Limiting**: Different limits for different priority levels
//! - **Adaptive Rate Limiting**: Adjust rates based on success/failure patterns
//! - **Backpressure Support**: Queue connections when rate limit is exceeded
//!
//! # Example
//!
//! ```rust
//! use ipfrs_network::rate_limiter::{ConnectionRateLimiter, RateLimiterConfig};
//! use std::time::Duration;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Create rate limiter allowing 10 connections per second with burst of 20
//! let mut limiter = ConnectionRateLimiter::new(RateLimiterConfig {
//!     max_rate: 10.0,
//!     burst_size: 20,
//!     enable_per_peer_limits: true,
//!     ..Default::default()
//! });
//!
//! // Check if connection is allowed
//! let peer_id = "QmExample".to_string();
//! if limiter.allow_connection(&peer_id).await {
//!     println!("Connection allowed");
//!     // Establish connection...
//! } else {
//!     println!("Rate limit exceeded, queuing...");
//! }
//! # Ok(())
//! # }
//! ```

pub mod atomictokenbucket_traits;
pub mod backpressurecontroller_traits;
pub mod constants;
pub mod peerratelimiterconfig_traits;
pub mod ratelimiter_traits;
pub mod ratelimiterconfig_traits;
pub mod types;
pub mod types_7;

// Re-export all types
pub use types::*;
pub use types_7::*;

#[cfg(test)]
mod tests;
