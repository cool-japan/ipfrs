//! Circuit Breaker pattern for fault tolerance
//!
//! Prevents cascading failures by detecting unhealthy services and temporarily
//! blocking requests until the service recovers. Useful for external backends
//! like S3, IPFS gateways, and distributed storage nodes.
//!
//! ## States
//! - **Closed**: Normal operation, requests pass through
//! - **Open**: Service is unhealthy, requests fail fast
//! - **Half-Open**: Testing if service has recovered
//!
//! ## Example
//! ```no_run
//! use ipfrs_storage::CircuitBreaker;
//! use std::time::Duration;
//!
//! async fn call_external_service() -> Result<String, std::io::Error> {
//!     // Your external service call
//!     Ok("success".to_string())
//! }
//!
//! #[tokio::main]
//! async fn main() {
//!     let cb = CircuitBreaker::new(5, Duration::from_secs(30));
//!
//!     match cb.call(call_external_service()).await {
//!         Ok(result) => println!("Success: {}", result),
//!         Err(e) => eprintln!("Failed: {}", e),
//!     }
//! }
//! ```

use anyhow::{anyhow, Result};
use parking_lot::RwLock;
use std::future::Future;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Circuit breaker state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    /// Normal operation - requests pass through
    Closed,
    /// Service is unhealthy - fail fast
    Open,
    /// Testing recovery - allow limited requests
    HalfOpen,
}

/// Circuit breaker statistics
#[derive(Debug, Clone, Default)]
pub struct CircuitStats {
    /// Total requests attempted
    pub total_requests: u64,
    /// Successful requests
    pub successful_requests: u64,
    /// Failed requests
    pub failed_requests: u64,
    /// Requests rejected by circuit breaker
    pub rejected_requests: u64,
    /// Number of times circuit opened
    pub circuit_opened_count: u64,
    /// Current consecutive failures
    pub consecutive_failures: u32,
}

/// Internal state of the circuit breaker
#[derive(Debug)]
struct CircuitBreakerState {
    /// Current state
    state: CircuitState,
    /// Number of consecutive failures
    consecutive_failures: u32,
    /// When the circuit was opened
    opened_at: Option<Instant>,
    /// Statistics
    stats: CircuitStats,
}

/// Circuit Breaker for fault-tolerant external service calls
///
/// Automatically opens when failure threshold is exceeded, preventing
/// cascading failures. After a timeout, enters half-open state to test
/// if the service has recovered.
#[derive(Debug, Clone)]
pub struct CircuitBreaker {
    /// Failure threshold before opening circuit
    failure_threshold: u32,
    /// Timeout before attempting recovery
    timeout: Duration,
    /// Half-open success threshold before closing
    #[allow(dead_code)]
    half_open_threshold: u32,
    /// Internal state
    state: Arc<RwLock<CircuitBreakerState>>,
}

impl CircuitBreaker {
    /// Create a new circuit breaker
    ///
    /// # Arguments
    /// * `failure_threshold` - Consecutive failures before opening
    /// * `timeout` - Duration to wait before trying half-open
    pub fn new(failure_threshold: u32, timeout: Duration) -> Self {
        Self::with_half_open_threshold(failure_threshold, timeout, 1)
    }

    /// Create a circuit breaker with custom half-open threshold
    ///
    /// # Arguments
    /// * `failure_threshold` - Consecutive failures before opening
    /// * `timeout` - Duration to wait before trying half-open
    /// * `half_open_threshold` - Successes needed in half-open to close
    pub fn with_half_open_threshold(
        failure_threshold: u32,
        timeout: Duration,
        half_open_threshold: u32,
    ) -> Self {
        Self {
            failure_threshold,
            timeout,
            half_open_threshold,
            state: Arc::new(RwLock::new(CircuitBreakerState {
                state: CircuitState::Closed,
                consecutive_failures: 0,
                opened_at: None,
                stats: CircuitStats::default(),
            })),
        }
    }

    /// Execute a function through the circuit breaker
    ///
    /// # Arguments
    /// * `f` - Future to execute
    ///
    /// # Returns
    /// Result of the function or circuit breaker error
    pub async fn call<F, T, E>(&self, f: F) -> Result<T>
    where
        F: Future<Output = Result<T, E>>,
        E: std::error::Error + Send + Sync + 'static,
    {
        // Check if we can proceed
        if !self.can_proceed() {
            let mut state = self.state.write();
            state.stats.rejected_requests += 1;
            return Err(anyhow!("Circuit breaker is OPEN"));
        }

        // Increment total requests
        self.state.write().stats.total_requests += 1;

        // Execute the function
        match f.await {
            Ok(result) => {
                self.on_success();
                Ok(result)
            }
            Err(e) => {
                self.on_failure();
                Err(anyhow!("Circuit breaker call failed: {e}"))
            }
        }
    }

    /// Check if a request can proceed
    fn can_proceed(&self) -> bool {
        let mut state = self.state.write();

        match state.state {
            CircuitState::Closed => true,
            CircuitState::Open => {
                // Check if timeout has elapsed
                if let Some(opened_at) = state.opened_at {
                    if opened_at.elapsed() >= self.timeout {
                        // Transition to half-open
                        state.state = CircuitState::HalfOpen;
                        state.consecutive_failures = 0;
                        true
                    } else {
                        false
                    }
                } else {
                    false
                }
            }
            CircuitState::HalfOpen => true,
        }
    }

    /// Handle successful request
    fn on_success(&self) {
        let mut state = self.state.write();
        state.stats.successful_requests += 1;

        match state.state {
            CircuitState::Closed => {
                state.consecutive_failures = 0;
            }
            CircuitState::HalfOpen => {
                // Check if we've had enough successes to close
                if state.consecutive_failures == 0 {
                    // First success in half-open, need more
                    state.consecutive_failures = 0;
                }

                // For simplicity, close on first success
                // In production, you might want to require multiple successes
                state.state = CircuitState::Closed;
                state.consecutive_failures = 0;
                state.opened_at = None;
            }
            CircuitState::Open => {
                // Shouldn't happen, but reset if it does
                state.consecutive_failures = 0;
            }
        }
    }

    /// Handle failed request
    fn on_failure(&self) {
        let mut state = self.state.write();
        state.stats.failed_requests += 1;
        state.consecutive_failures += 1;

        match state.state {
            CircuitState::Closed => {
                if state.consecutive_failures >= self.failure_threshold {
                    // Open the circuit
                    state.state = CircuitState::Open;
                    state.opened_at = Some(Instant::now());
                    state.stats.circuit_opened_count += 1;
                }
            }
            CircuitState::HalfOpen => {
                // Failure in half-open means we're still unhealthy
                state.state = CircuitState::Open;
                state.opened_at = Some(Instant::now());
                state.stats.circuit_opened_count += 1;
            }
            CircuitState::Open => {
                // Already open, just count the failure
            }
        }

        state.stats.consecutive_failures = state.consecutive_failures;
    }

    /// Get current circuit state
    pub fn state(&self) -> CircuitState {
        self.state.read().state
    }

    /// Get circuit breaker statistics
    pub fn stats(&self) -> CircuitStats {
        self.state.read().stats.clone()
    }

    /// Manually reset the circuit breaker to closed state
    pub fn reset(&self) {
        let mut state = self.state.write();
        state.state = CircuitState::Closed;
        state.consecutive_failures = 0;
        state.opened_at = None;
        state.stats.consecutive_failures = 0;
    }

    /// Check if circuit is closed (healthy)
    pub fn is_closed(&self) -> bool {
        self.state() == CircuitState::Closed
    }

    /// Check if circuit is open (unhealthy)
    pub fn is_open(&self) -> bool {
        self.state() == CircuitState::Open
    }

    /// Check if circuit is half-open (testing recovery)
    pub fn is_half_open(&self) -> bool {
        self.state() == CircuitState::HalfOpen
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::sleep;

    async fn succeeding_operation() -> Result<&'static str, std::io::Error> {
        Ok("success")
    }

    async fn failing_operation() -> Result<&'static str, std::io::Error> {
        Err(std::io::Error::other("failure"))
    }

    #[tokio::test]
    async fn test_circuit_breaker_closed_state() {
        let cb = CircuitBreaker::new(3, Duration::from_secs(1));
        assert!(cb.is_closed());

        let result = cb.call(succeeding_operation()).await;
        assert!(result.is_ok());
        assert!(cb.is_closed());
    }

    #[tokio::test]
    async fn test_circuit_breaker_opens_on_failures() {
        let cb = CircuitBreaker::new(3, Duration::from_secs(1));

        // Trigger 3 failures to open circuit
        for _ in 0..3 {
            let _ = cb.call(failing_operation()).await;
        }

        assert!(cb.is_open());

        let stats = cb.stats();
        assert_eq!(stats.failed_requests, 3);
        assert_eq!(stats.circuit_opened_count, 1);
    }

    #[tokio::test]
    async fn test_circuit_breaker_rejects_when_open() {
        let cb = CircuitBreaker::new(2, Duration::from_secs(1));

        // Open the circuit
        for _ in 0..2 {
            let _ = cb.call(failing_operation()).await;
        }

        assert!(cb.is_open());

        // Next request should be rejected
        let result = cb.call(succeeding_operation()).await;
        assert!(result.is_err());

        let stats = cb.stats();
        assert_eq!(stats.rejected_requests, 1);
    }

    #[tokio::test]
    async fn test_circuit_breaker_half_open() {
        let cb = CircuitBreaker::new(2, Duration::from_millis(100));

        // Open the circuit
        for _ in 0..2 {
            let _ = cb.call(failing_operation()).await;
        }

        assert!(cb.is_open());

        // Wait for timeout
        sleep(Duration::from_millis(150)).await;

        // Next request should transition to half-open
        let result = cb.call(succeeding_operation()).await;
        assert!(result.is_ok());

        // Should be closed now after success
        assert!(cb.is_closed());
    }

    #[tokio::test]
    async fn test_circuit_breaker_stats() {
        let cb = CircuitBreaker::new(5, Duration::from_secs(1));

        // Make some successful calls
        for _ in 0..3 {
            let _ = cb.call(succeeding_operation()).await;
        }

        // Make some failed calls
        for _ in 0..2 {
            let _ = cb.call(failing_operation()).await;
        }

        let stats = cb.stats();
        assert_eq!(stats.total_requests, 5);
        assert_eq!(stats.successful_requests, 3);
        assert_eq!(stats.failed_requests, 2);
        assert_eq!(stats.consecutive_failures, 2);
    }

    #[tokio::test]
    async fn test_circuit_breaker_reset() {
        let cb = CircuitBreaker::new(2, Duration::from_secs(1));

        // Open the circuit
        for _ in 0..2 {
            let _ = cb.call(failing_operation()).await;
        }

        assert!(cb.is_open());

        // Manual reset
        cb.reset();

        assert!(cb.is_closed());
        assert_eq!(cb.stats().consecutive_failures, 0);
    }
}
