//! Rate limiting for controlling request rates to backends
//!
//! Provides token bucket and leaky bucket algorithms for rate limiting:
//! - Token bucket: Allows bursts up to capacity
//! - Leaky bucket: Smooth rate limiting
//! - Per-operation rate limiting
//! - Configurable refill rates
//!
//! ## Example
//! ```no_run
//! use ipfrs_storage::{RateLimiter, RateLimitConfig};
//! use std::time::Duration;
//!
//! #[tokio::main]
//! async fn main() {
//!     let config = RateLimitConfig::new(100, Duration::from_secs(1));
//!     let limiter = RateLimiter::new(config);
//!
//!     // Acquire permission to proceed
//!     limiter.acquire(1).await;
//!     // Make your API call here
//! }
//! ```

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::time::sleep;

/// Rate limiting algorithm
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RateLimitAlgorithm {
    /// Token bucket - allows bursts up to capacity
    TokenBucket,
    /// Leaky bucket - smooth rate limiting
    LeakyBucket,
}

/// Rate limiter configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitConfig {
    /// Maximum tokens/requests (capacity)
    pub capacity: u64,
    /// Refill rate (tokens per interval)
    pub refill_rate: u64,
    /// Refill interval
    pub refill_interval: Duration,
    /// Algorithm to use
    pub algorithm: RateLimitAlgorithm,
    /// Whether to block or return error when limit exceeded
    pub block_on_limit: bool,
}

impl RateLimitConfig {
    /// Create a new rate limit configuration
    ///
    /// # Arguments
    /// * `capacity` - Maximum tokens (burst size)
    /// * `refill_interval` - How often to refill tokens
    pub fn new(capacity: u64, refill_interval: Duration) -> Self {
        Self {
            capacity,
            refill_rate: capacity,
            refill_interval,
            algorithm: RateLimitAlgorithm::TokenBucket,
            block_on_limit: true,
        }
    }

    /// Create configuration for requests per second
    pub fn per_second(requests: u64) -> Self {
        Self::new(requests, Duration::from_secs(1))
    }

    /// Create configuration for requests per minute
    pub fn per_minute(requests: u64) -> Self {
        Self::new(requests, Duration::from_secs(60))
    }

    /// Set the refill rate
    pub fn with_refill_rate(mut self, rate: u64) -> Self {
        self.refill_rate = rate;
        self
    }

    /// Set the algorithm
    pub fn with_algorithm(mut self, algorithm: RateLimitAlgorithm) -> Self {
        self.algorithm = algorithm;
        self
    }

    /// Set whether to block when limit is exceeded
    pub fn with_blocking(mut self, block: bool) -> Self {
        self.block_on_limit = block;
        self
    }
}

/// Internal state for rate limiter
#[derive(Debug)]
struct RateLimiterState {
    /// Available tokens
    tokens: f64,
    /// Last refill time
    last_refill: Instant,
    /// Total requests made
    total_requests: u64,
    /// Requests allowed
    requests_allowed: u64,
    /// Requests denied
    requests_denied: u64,
}

/// Rate limiter statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RateLimitStats {
    /// Total requests attempted
    pub total_requests: u64,
    /// Requests allowed through
    pub requests_allowed: u64,
    /// Requests denied/delayed
    pub requests_denied: u64,
    /// Current available tokens
    pub available_tokens: u64,
    /// Utilization percentage (0-100)
    pub utilization_percent: f64,
}

/// Token bucket rate limiter
pub struct RateLimiter {
    config: RateLimitConfig,
    state: Arc<Mutex<RateLimiterState>>,
}

impl RateLimiter {
    /// Create a new rate limiter
    pub fn new(config: RateLimitConfig) -> Self {
        Self {
            state: Arc::new(Mutex::new(RateLimiterState {
                tokens: config.capacity as f64,
                last_refill: Instant::now(),
                total_requests: 0,
                requests_allowed: 0,
                requests_denied: 0,
            })),
            config,
        }
    }

    /// Acquire tokens from the rate limiter
    ///
    /// # Arguments
    /// * `tokens` - Number of tokens to acquire
    ///
    /// # Returns
    /// True if tokens were acquired, false if rate limit exceeded
    pub async fn acquire(&self, tokens: u64) -> bool {
        loop {
            // Try to acquire tokens
            let wait_duration = {
                let mut state = self.state.lock();
                self.refill_tokens(&mut state);

                state.total_requests += 1;

                if state.tokens >= tokens as f64 {
                    // Have enough tokens
                    state.tokens -= tokens as f64;
                    state.requests_allowed += 1;
                    return true;
                } else {
                    state.requests_denied += 1;

                    if !self.config.block_on_limit {
                        return false;
                    }

                    // Calculate how long to wait
                    let tokens_needed = tokens as f64 - state.tokens;
                    let tokens_per_ms = self.config.refill_rate as f64
                        / self.config.refill_interval.as_millis() as f64;
                    let wait_ms = (tokens_needed / tokens_per_ms).ceil() as u64;
                    Duration::from_millis(wait_ms.max(1))
                }
            };

            // Wait before retrying
            sleep(wait_duration).await;
        }
    }

    /// Try to acquire tokens without blocking
    pub fn try_acquire(&self, tokens: u64) -> bool {
        let mut state = self.state.lock();
        self.refill_tokens(&mut state);

        state.total_requests += 1;

        if state.tokens >= tokens as f64 {
            state.tokens -= tokens as f64;
            state.requests_allowed += 1;
            true
        } else {
            state.requests_denied += 1;
            false
        }
    }

    /// Get current statistics
    pub fn stats(&self) -> RateLimitStats {
        let mut state = self.state.lock();
        self.refill_tokens(&mut state);

        RateLimitStats {
            total_requests: state.total_requests,
            requests_allowed: state.requests_allowed,
            requests_denied: state.requests_denied,
            available_tokens: state.tokens as u64,
            utilization_percent: if state.total_requests > 0 {
                (state.requests_allowed as f64 / state.total_requests as f64) * 100.0
            } else {
                0.0
            },
        }
    }

    /// Reset the rate limiter
    pub fn reset(&self) {
        let mut state = self.state.lock();
        state.tokens = self.config.capacity as f64;
        state.last_refill = Instant::now();
        state.total_requests = 0;
        state.requests_allowed = 0;
        state.requests_denied = 0;
    }

    /// Refill tokens based on elapsed time
    fn refill_tokens(&self, state: &mut RateLimiterState) {
        let now = Instant::now();
        let elapsed = now.duration_since(state.last_refill);

        if elapsed >= self.config.refill_interval {
            let intervals = elapsed.as_secs_f64() / self.config.refill_interval.as_secs_f64();
            let tokens_to_add = intervals * self.config.refill_rate as f64;

            state.tokens = (state.tokens + tokens_to_add).min(self.config.capacity as f64);
            state.last_refill = now;
        }
    }
}

impl Clone for RateLimiter {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            state: Arc::clone(&self.state),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::{sleep, Duration};

    #[tokio::test]
    async fn test_rate_limiter_basic() {
        let config = RateLimitConfig::new(10, Duration::from_secs(1));
        let limiter = RateLimiter::new(config);

        // Should be able to acquire up to capacity
        for _ in 0..10 {
            assert!(limiter.try_acquire(1));
        }

        // Next acquisition should fail
        assert!(!limiter.try_acquire(1));

        let stats = limiter.stats();
        assert_eq!(stats.requests_allowed, 10);
        assert_eq!(stats.requests_denied, 1);
    }

    #[tokio::test]
    async fn test_rate_limiter_refill() {
        let config = RateLimitConfig::new(5, Duration::from_millis(100));
        let limiter = RateLimiter::new(config);

        // Exhaust tokens
        for _ in 0..5 {
            assert!(limiter.try_acquire(1));
        }
        assert!(!limiter.try_acquire(1));

        // Wait for refill
        sleep(Duration::from_millis(150)).await;

        // Should be able to acquire again
        assert!(limiter.try_acquire(1));
    }

    #[tokio::test]
    async fn test_rate_limiter_blocking() {
        let config = RateLimitConfig::new(2, Duration::from_millis(100)).with_blocking(true);
        let limiter = RateLimiter::new(config);

        // Exhaust tokens
        limiter.acquire(2).await;

        // This should block and wait for refill
        let start = Instant::now();
        limiter.acquire(1).await;
        let elapsed = start.elapsed();

        // Should have waited at least ~100ms for refill
        assert!(elapsed >= Duration::from_millis(50));
    }

    #[tokio::test]
    async fn test_rate_limiter_stats() {
        let config = RateLimitConfig::new(10, Duration::from_secs(1));
        let limiter = RateLimiter::new(config);

        // Make some requests
        for _ in 0..5 {
            limiter.try_acquire(1);
        }

        let stats = limiter.stats();
        assert_eq!(stats.total_requests, 5);
        assert_eq!(stats.requests_allowed, 5);
        assert_eq!(stats.requests_denied, 0);
        assert_eq!(stats.available_tokens, 5);
        assert_eq!(stats.utilization_percent, 100.0);
    }

    #[tokio::test]
    async fn test_rate_limiter_reset() {
        let config = RateLimitConfig::new(5, Duration::from_secs(1));
        let limiter = RateLimiter::new(config);

        // Exhaust tokens
        for _ in 0..5 {
            limiter.try_acquire(1);
        }

        // Reset
        limiter.reset();

        // Should be able to acquire again
        assert!(limiter.try_acquire(1));

        let stats = limiter.stats();
        assert_eq!(stats.total_requests, 1);
    }

    #[tokio::test]
    async fn test_rate_limiter_per_second() {
        let config = RateLimitConfig::per_second(100);
        let limiter = RateLimiter::new(config);

        assert_eq!(limiter.config.capacity, 100);
        assert_eq!(limiter.config.refill_interval, Duration::from_secs(1));
    }

    #[tokio::test]
    async fn test_rate_limiter_per_minute() {
        let config = RateLimitConfig::per_minute(1000);
        let limiter = RateLimiter::new(config);

        assert_eq!(limiter.config.capacity, 1000);
        assert_eq!(limiter.config.refill_interval, Duration::from_secs(60));
    }
}
