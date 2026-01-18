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

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use thiserror::Error;

/// Errors that can occur during rate limiting operations
#[derive(Debug, Error)]
pub enum RateLimiterError {
    #[error("Rate limit exceeded")]
    RateLimitExceeded,

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("Peer blocked: {0}")]
    PeerBlocked(String),
}

/// Priority level for connections
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConnectionPriority {
    /// Critical connections (bootstrap nodes, important peers)
    Critical,
    /// High priority connections
    High,
    /// Normal priority connections
    Normal,
    /// Low priority connections
    Low,
}

impl ConnectionPriority {
    /// Get the rate multiplier for this priority level
    pub fn rate_multiplier(&self) -> f64 {
        match self {
            Self::Critical => 2.0, // 2x the base rate
            Self::High => 1.5,     // 1.5x the base rate
            Self::Normal => 1.0,   // Base rate
            Self::Low => 0.5,      // Half the base rate
        }
    }
}

/// Configuration for the connection rate limiter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimiterConfig {
    /// Maximum connection rate (connections per second)
    pub max_rate: f64,

    /// Maximum burst size (tokens)
    pub burst_size: usize,

    /// Enable per-peer rate limiting
    pub enable_per_peer_limits: bool,

    /// Maximum connections per peer per second
    pub max_per_peer_rate: f64,

    /// Enable adaptive rate limiting
    pub enable_adaptive: bool,

    /// Adjustment factor for adaptive limiting (0.0 to 1.0)
    pub adaptive_factor: f64,

    /// Minimum rate (connections per second) when adapting
    pub min_rate: f64,

    /// Maximum rate (connections per second) when adapting
    pub max_adaptive_rate: f64,

    /// Enable connection queuing when rate limited
    pub enable_queuing: bool,

    /// Maximum queue size for pending connections
    pub max_queue_size: usize,

    /// Time window for per-peer tracking
    pub peer_window: Duration,
}

impl Default for RateLimiterConfig {
    fn default() -> Self {
        Self {
            max_rate: 10.0, // 10 connections/sec
            burst_size: 20, // 20 token burst
            enable_per_peer_limits: true,
            max_per_peer_rate: 2.0, // 2 connections/sec per peer
            enable_adaptive: false,
            adaptive_factor: 0.1,     // 10% adjustment
            min_rate: 1.0,            // Min 1 connection/sec
            max_adaptive_rate: 100.0, // Max 100 connections/sec
            enable_queuing: true,
            max_queue_size: 100,
            peer_window: Duration::from_secs(60), // 1 minute window
        }
    }
}

impl RateLimiterConfig {
    /// Configuration for aggressive rate limiting (low rates)
    pub fn conservative() -> Self {
        Self {
            max_rate: 5.0,
            burst_size: 10,
            max_per_peer_rate: 1.0,
            max_queue_size: 50,
            ..Default::default()
        }
    }

    /// Configuration for permissive rate limiting (high rates)
    pub fn permissive() -> Self {
        Self {
            max_rate: 50.0,
            burst_size: 100,
            max_per_peer_rate: 10.0,
            max_queue_size: 200,
            ..Default::default()
        }
    }

    /// Configuration with adaptive rate limiting enabled
    pub fn adaptive() -> Self {
        Self {
            enable_adaptive: true,
            adaptive_factor: 0.2,
            min_rate: 2.0,
            max_adaptive_rate: 50.0,
            ..Default::default()
        }
    }
}

/// Per-peer connection tracking
#[derive(Debug, Clone)]
struct PeerTracking {
    /// Connection attempts in current window
    attempts: Vec<Instant>,
    /// Successful connections
    successes: u64,
    /// Failed connections
    failures: u64,
    /// Last connection timestamp
    last_connection: Option<Instant>,
}

impl PeerTracking {
    fn new() -> Self {
        Self {
            attempts: Vec::new(),
            successes: 0,
            failures: 0,
            last_connection: None,
        }
    }

    /// Clean up old attempts outside the window
    fn cleanup(&mut self, window: Duration) {
        let cutoff = Instant::now() - window;
        self.attempts.retain(|&t| t > cutoff);
    }

    /// Record a connection attempt
    fn record_attempt(&mut self) {
        self.attempts.push(Instant::now());
        self.last_connection = Some(Instant::now());
    }

    /// Get current rate (attempts per second)
    fn current_rate(&self, window: Duration) -> f64 {
        if self.attempts.is_empty() {
            return 0.0;
        }

        let now = Instant::now();
        let recent = self
            .attempts
            .iter()
            .filter(|&&t| now.duration_since(t) < window)
            .count();

        recent as f64 / window.as_secs_f64()
    }
}

/// Token bucket for rate limiting
#[derive(Debug)]
struct TokenBucket {
    /// Current number of tokens
    tokens: f64,
    /// Maximum tokens (burst size)
    capacity: f64,
    /// Token refill rate (tokens per second)
    rate: f64,
    /// Last refill timestamp
    last_refill: Instant,
}

impl TokenBucket {
    fn new(rate: f64, capacity: usize) -> Self {
        Self {
            tokens: capacity as f64,
            capacity: capacity as f64,
            rate,
            last_refill: Instant::now(),
        }
    }

    /// Refill tokens based on elapsed time
    fn refill(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_refill).as_secs_f64();
        let new_tokens = elapsed * self.rate;

        self.tokens = (self.tokens + new_tokens).min(self.capacity);
        self.last_refill = now;
    }

    /// Try to consume a token
    fn try_consume(&mut self, count: f64) -> bool {
        self.refill();

        if self.tokens >= count {
            self.tokens -= count;
            true
        } else {
            false
        }
    }

    /// Get current token count
    fn available(&mut self) -> f64 {
        self.refill();
        self.tokens
    }

    /// Update rate dynamically
    fn update_rate(&mut self, new_rate: f64) {
        self.refill(); // Refill with old rate first
        self.rate = new_rate;
    }
}

/// Statistics tracked by the rate limiter
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RateLimiterStats {
    /// Total connection attempts
    pub total_attempts: u64,

    /// Connections allowed
    pub allowed: u64,

    /// Connections rate limited
    pub rate_limited: u64,

    /// Connections queued
    pub queued: u64,

    /// Current queue size
    pub current_queue_size: usize,

    /// Average rate (connections per second)
    pub avg_rate: f64,

    /// Current rate limit
    pub current_limit: f64,

    /// Tokens available
    pub tokens_available: f64,
}

/// Connection rate limiter
pub struct ConnectionRateLimiter {
    config: RateLimiterConfig,
    bucket: Arc<RwLock<TokenBucket>>,
    peer_tracking: Arc<RwLock<HashMap<String, PeerTracking>>>,
    stats: Arc<RwLock<RateLimiterStats>>,
    queue: Arc<RwLock<Vec<(String, ConnectionPriority, Instant)>>>,
}

impl ConnectionRateLimiter {
    /// Create a new connection rate limiter
    pub fn new(config: RateLimiterConfig) -> Self {
        let bucket = TokenBucket::new(config.max_rate, config.burst_size);

        Self {
            config,
            bucket: Arc::new(RwLock::new(bucket)),
            peer_tracking: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(RateLimiterStats::default())),
            queue: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Check if a connection is allowed
    pub async fn allow_connection(&self, peer_id: &str) -> bool {
        self.allow_connection_with_priority(peer_id, ConnectionPriority::Normal)
            .await
    }

    /// Check if a connection is allowed with specific priority
    pub async fn allow_connection_with_priority(
        &self,
        peer_id: &str,
        priority: ConnectionPriority,
    ) -> bool {
        let mut stats = self.stats.write();
        stats.total_attempts += 1;

        // Check per-peer limits
        if self.config.enable_per_peer_limits {
            let mut tracking = self.peer_tracking.write();
            let peer_track = tracking
                .entry(peer_id.to_string())
                .or_insert_with(PeerTracking::new);

            peer_track.cleanup(self.config.peer_window);

            let current_rate = peer_track.current_rate(self.config.peer_window);
            if current_rate >= self.config.max_per_peer_rate {
                stats.rate_limited += 1;
                return false;
            }
        }

        // Check global token bucket
        let cost = 1.0 / priority.rate_multiplier();
        let mut bucket = self.bucket.write();

        if bucket.try_consume(cost) {
            // Update tracking
            if self.config.enable_per_peer_limits {
                let mut tracking = self.peer_tracking.write();
                if let Some(peer_track) = tracking.get_mut(peer_id) {
                    peer_track.record_attempt();
                }
            }

            stats.allowed += 1;
            stats.tokens_available = bucket.available();
            true
        } else {
            stats.rate_limited += 1;

            // Queue if enabled
            if self.config.enable_queuing {
                let mut queue = self.queue.write();
                if queue.len() < self.config.max_queue_size {
                    queue.push((peer_id.to_string(), priority, Instant::now()));
                    stats.queued += 1;
                    stats.current_queue_size = queue.len();
                }
            }

            false
        }
    }

    /// Record a successful connection
    pub fn record_success(&self, peer_id: &str) {
        if !self.config.enable_per_peer_limits {
            return;
        }

        let mut tracking = self.peer_tracking.write();
        if let Some(peer_track) = tracking.get_mut(peer_id) {
            peer_track.successes += 1;

            // Adapt rate if enabled
            if self.config.enable_adaptive {
                self.adapt_rate_on_success();
            }
        }
    }

    /// Record a failed connection
    pub fn record_failure(&self, peer_id: &str) {
        if !self.config.enable_per_peer_limits {
            return;
        }

        let mut tracking = self.peer_tracking.write();
        if let Some(peer_track) = tracking.get_mut(peer_id) {
            peer_track.failures += 1;

            // Adapt rate if enabled
            if self.config.enable_adaptive {
                self.adapt_rate_on_failure();
            }
        }
    }

    /// Adapt rate upward on success
    fn adapt_rate_on_success(&self) {
        let mut bucket = self.bucket.write();
        let current_rate = bucket.rate;
        let new_rate =
            (current_rate * (1.0 + self.config.adaptive_factor)).min(self.config.max_adaptive_rate);

        if new_rate != current_rate {
            bucket.update_rate(new_rate);

            let mut stats = self.stats.write();
            stats.current_limit = new_rate;
        }
    }

    /// Adapt rate downward on failure
    fn adapt_rate_on_failure(&self) {
        let mut bucket = self.bucket.write();
        let current_rate = bucket.rate;
        let new_rate =
            (current_rate * (1.0 - self.config.adaptive_factor)).max(self.config.min_rate);

        if new_rate != current_rate {
            bucket.update_rate(new_rate);

            let mut stats = self.stats.write();
            stats.current_limit = new_rate;
        }
    }

    /// Process queued connections
    pub async fn process_queue(&self) -> Vec<String> {
        let mut queue = self.queue.write();
        let mut bucket = self.bucket.write();
        let mut allowed = Vec::new();

        // Sort by priority (Critical first, then by timestamp)
        queue.sort_by(|a, b| match (a.1, b.1) {
            (ConnectionPriority::Critical, ConnectionPriority::Critical) => a.2.cmp(&b.2),
            (ConnectionPriority::Critical, _) => std::cmp::Ordering::Less,
            (_, ConnectionPriority::Critical) => std::cmp::Ordering::Greater,
            (ConnectionPriority::High, ConnectionPriority::High) => a.2.cmp(&b.2),
            (ConnectionPriority::High, _) => std::cmp::Ordering::Less,
            (_, ConnectionPriority::High) => std::cmp::Ordering::Greater,
            _ => a.2.cmp(&b.2),
        });

        // Process as many as possible
        queue.retain(|(peer_id, priority, _)| {
            let cost = 1.0 / priority.rate_multiplier();
            if bucket.try_consume(cost) {
                allowed.push(peer_id.clone());
                false // Remove from queue
            } else {
                true // Keep in queue
            }
        });

        let mut stats = self.stats.write();
        stats.current_queue_size = queue.len();

        allowed
    }

    /// Get current statistics
    pub fn stats(&self) -> RateLimiterStats {
        let mut stats = self.stats.read().clone();

        // Update dynamic stats
        let bucket = self.bucket.write();
        stats.current_limit = bucket.rate;
        stats.tokens_available = bucket.tokens;

        if stats.total_attempts > 0 {
            stats.avg_rate = stats.allowed as f64 / (stats.total_attempts as f64 / bucket.rate);
        }

        stats
    }

    /// Get per-peer statistics
    pub fn peer_stats(&self, peer_id: &str) -> Option<(u64, u64, f64)> {
        let tracking = self.peer_tracking.read();
        tracking.get(peer_id).map(|track| {
            (
                track.successes,
                track.failures,
                track.current_rate(self.config.peer_window),
            )
        })
    }

    /// Reset rate limiter state
    pub fn reset(&self) {
        let mut bucket = self.bucket.write();
        bucket.tokens = bucket.capacity;
        bucket.last_refill = Instant::now();

        self.peer_tracking.write().clear();
        self.queue.write().clear();

        let mut stats = self.stats.write();
        *stats = RateLimiterStats::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::sleep;

    #[tokio::test]
    async fn test_rate_limiter_creation() {
        let limiter = ConnectionRateLimiter::new(RateLimiterConfig::default());
        let stats = limiter.stats();
        assert_eq!(stats.total_attempts, 0);
    }

    #[tokio::test]
    async fn test_allow_connection() {
        let limiter = ConnectionRateLimiter::new(RateLimiterConfig::default());

        let allowed = limiter.allow_connection("peer1").await;
        assert!(allowed);

        let stats = limiter.stats();
        assert_eq!(stats.allowed, 1);
    }

    #[tokio::test]
    async fn test_rate_limiting() {
        let config = RateLimiterConfig {
            max_rate: 10.0,
            burst_size: 5,
            ..Default::default()
        };
        let limiter = ConnectionRateLimiter::new(config);

        // Use up burst
        for _ in 0..5 {
            assert!(limiter.allow_connection("peer1").await);
        }

        // Next should be rate limited
        let allowed = limiter.allow_connection("peer1").await;
        assert!(!allowed);
    }

    #[tokio::test]
    async fn test_per_peer_limits() {
        let config = RateLimiterConfig {
            max_rate: 100.0,
            burst_size: 100,
            enable_per_peer_limits: true,
            max_per_peer_rate: 5.0, // Allow 5 connections per second
            peer_window: Duration::from_secs(1), // 1 second window for testing
            ..Default::default()
        };
        let limiter = ConnectionRateLimiter::new(config);

        // First 5 connections should succeed (rate = 5/sec)
        for _ in 0..5 {
            assert!(limiter.allow_connection("peer1").await);
        }

        // Sixth should be rate limited (would exceed 5 connections/sec)
        let allowed = limiter.allow_connection("peer1").await;
        assert!(!allowed);

        // Different peer should still work
        assert!(limiter.allow_connection("peer2").await);
    }

    #[tokio::test]
    async fn test_priority() {
        let config = RateLimiterConfig {
            max_rate: 10.0,
            burst_size: 2,
            ..Default::default()
        };
        let limiter = ConnectionRateLimiter::new(config);

        // Critical priority should cost less
        assert!(
            limiter
                .allow_connection_with_priority("peer1", ConnectionPriority::Critical)
                .await
        );
        assert!(
            limiter
                .allow_connection_with_priority("peer2", ConnectionPriority::Critical)
                .await
        );

        // Should still have some tokens due to lower cost
        let stats = limiter.stats();
        assert!(stats.tokens_available > 0.0);
    }

    #[tokio::test]
    async fn test_queuing() {
        let config = RateLimiterConfig {
            max_rate: 1.0,
            burst_size: 1,
            enable_queuing: true,
            max_queue_size: 10,
            ..Default::default()
        };
        let limiter = ConnectionRateLimiter::new(config);

        // First connection allowed
        assert!(limiter.allow_connection("peer1").await);

        // Next connections should be queued
        assert!(!limiter.allow_connection("peer2").await);
        assert!(!limiter.allow_connection("peer3").await);

        let stats = limiter.stats();
        assert_eq!(stats.queued, 2);
    }

    #[tokio::test]
    async fn test_process_queue() {
        let config = RateLimiterConfig {
            max_rate: 10.0,
            burst_size: 1,
            enable_queuing: true,
            ..Default::default()
        };
        let limiter = ConnectionRateLimiter::new(config);

        // Fill bucket
        limiter.allow_connection("peer1").await;

        // Queue some connections
        limiter.allow_connection("peer2").await;
        limiter.allow_connection("peer3").await;

        // Wait for refill
        sleep(Duration::from_millis(200)).await;

        // Process queue
        let allowed = limiter.process_queue().await;
        assert!(!allowed.is_empty());
    }

    #[tokio::test]
    async fn test_success_failure_recording() {
        let config = RateLimiterConfig {
            enable_per_peer_limits: true,
            ..Default::default()
        };
        let limiter = ConnectionRateLimiter::new(config);

        limiter.allow_connection("peer1").await;
        limiter.record_success("peer1");

        let (successes, failures, _) = limiter.peer_stats("peer1").unwrap();
        assert_eq!(successes, 1);
        assert_eq!(failures, 0);
    }

    #[tokio::test]
    async fn test_config_presets() {
        let conservative = RateLimiterConfig::conservative();
        assert!(conservative.max_rate < 10.0);

        let permissive = RateLimiterConfig::permissive();
        assert!(permissive.max_rate > 10.0);

        let adaptive = RateLimiterConfig::adaptive();
        assert!(adaptive.enable_adaptive);
    }

    #[tokio::test]
    async fn test_reset() {
        let limiter = ConnectionRateLimiter::new(RateLimiterConfig::default());

        limiter.allow_connection("peer1").await;
        assert_eq!(limiter.stats().allowed, 1);

        limiter.reset();
        assert_eq!(limiter.stats().allowed, 0);
    }

    #[tokio::test]
    async fn test_token_refill() {
        let config = RateLimiterConfig {
            max_rate: 10.0,
            burst_size: 5,
            ..Default::default()
        };
        let limiter = ConnectionRateLimiter::new(config);

        // Use all tokens
        for _ in 0..5 {
            limiter.allow_connection("peer1").await;
        }

        // Wait for refill
        sleep(Duration::from_millis(200)).await;

        // Should be able to connect again
        assert!(limiter.allow_connection("peer1").await);
    }
}
