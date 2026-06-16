//! Error recovery mechanisms
//!
//! This module provides error recovery patterns including retry logic,
//! exponential backoff, and circuit breakers for resilient operations.

use std::future::Future;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::time::sleep;

/// Retry policy configuration
#[derive(Debug, Clone)]
pub struct RetryPolicy {
    /// Maximum number of retry attempts
    pub max_attempts: usize,
    /// Initial delay between retries
    pub initial_delay: Duration,
    /// Maximum delay between retries
    pub max_delay: Duration,
    /// Backoff multiplier
    pub backoff_multiplier: f64,
}

impl RetryPolicy {
    /// Create a new retry policy with exponential backoff
    ///
    /// # Example
    /// ```rust
    /// use ipfrs::recovery::RetryPolicy;
    /// use std::time::Duration;
    ///
    /// let policy = RetryPolicy::exponential(3, Duration::from_millis(100));
    /// ```
    pub fn exponential(max_attempts: usize, initial_delay: Duration) -> Self {
        Self {
            max_attempts,
            initial_delay,
            max_delay: Duration::from_secs(60),
            backoff_multiplier: 2.0,
        }
    }

    /// Create a retry policy with fixed delay
    pub fn fixed(max_attempts: usize, delay: Duration) -> Self {
        Self {
            max_attempts,
            initial_delay: delay,
            max_delay: delay,
            backoff_multiplier: 1.0,
        }
    }

    /// Calculate delay for a given attempt number
    pub fn delay_for_attempt(&self, attempt: usize) -> Duration {
        if attempt == 0 {
            return self.initial_delay;
        }

        let multiplier = self.backoff_multiplier.powi(attempt as i32);
        let delay_ms = (self.initial_delay.as_millis() as f64 * multiplier) as u64;
        let delay = Duration::from_millis(delay_ms);

        std::cmp::min(delay, self.max_delay)
    }
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self::exponential(3, Duration::from_millis(100))
    }
}

/// Retry an async operation with the given policy
///
/// # Arguments
/// * `policy` - Retry policy configuration
/// * `operation` - Async function to retry
///
/// # Returns
/// Result of the operation or the last error encountered
///
/// # Example
/// ```rust,no_run
/// use ipfrs::recovery::{retry_async, RetryPolicy};
/// use std::time::Duration;
///
/// async fn example() -> Result<String, String> {
///     let policy = RetryPolicy::exponential(3, Duration::from_millis(100));
///     retry_async(policy, || async {
///         // Your async operation here
///         Ok("Success".to_string())
///     }).await
/// }
/// ```
pub async fn retry_async<F, Fut, T, E>(policy: RetryPolicy, mut operation: F) -> Result<T, E>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T, E>>,
    E: std::fmt::Display,
{
    let mut attempts = 0;
    let mut last_error = None;

    while attempts < policy.max_attempts {
        match operation().await {
            Ok(result) => return Ok(result),
            Err(error) => {
                attempts += 1;
                if attempts >= policy.max_attempts {
                    last_error = Some(error);
                    break;
                }

                let delay = policy.delay_for_attempt(attempts - 1);
                tracing::warn!(
                    "Operation failed (attempt {}/{}): {}. Retrying in {:?}",
                    attempts,
                    policy.max_attempts,
                    error,
                    delay
                );

                sleep(delay).await;
                last_error = Some(error);
            }
        }
    }

    Err(last_error.expect("loop ran at least max_attempts times so last_error is set"))
}

/// Circuit breaker state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    /// Circuit is closed (normal operation)
    Closed,
    /// Circuit is open (failing, rejecting requests)
    Open,
    /// Circuit is half-open (testing if service recovered)
    HalfOpen,
}

/// Circuit breaker for preventing cascading failures
///
/// Implements the circuit breaker pattern to protect against repeated failures.
pub struct CircuitBreaker {
    /// Current state
    state: Arc<AtomicUsize>,
    /// Failure count
    failure_count: Arc<AtomicU64>,
    /// Success count in half-open state
    success_count: Arc<AtomicU64>,
    /// Last state change time
    last_state_change: Arc<parking_lot::Mutex<Instant>>,
    /// Failure threshold to open circuit
    failure_threshold: u64,
    /// Success threshold to close circuit from half-open
    success_threshold: u64,
    /// Timeout before moving to half-open
    timeout: Duration,
}

impl CircuitBreaker {
    /// Create a new circuit breaker
    ///
    /// # Arguments
    /// * `failure_threshold` - Number of failures before opening circuit
    /// * `success_threshold` - Number of successes needed to close circuit
    /// * `timeout` - Time to wait before attempting recovery
    ///
    /// # Example
    /// ```rust
    /// use ipfrs::recovery::CircuitBreaker;
    /// use std::time::Duration;
    ///
    /// let breaker = CircuitBreaker::new(5, 2, Duration::from_secs(60));
    /// ```
    pub fn new(failure_threshold: u64, success_threshold: u64, timeout: Duration) -> Self {
        Self {
            state: Arc::new(AtomicUsize::new(CircuitState::Closed as usize)),
            failure_count: Arc::new(AtomicU64::new(0)),
            success_count: Arc::new(AtomicU64::new(0)),
            last_state_change: Arc::new(parking_lot::Mutex::new(Instant::now())),
            failure_threshold,
            success_threshold,
            timeout,
        }
    }

    /// Get current circuit state
    pub fn state(&self) -> CircuitState {
        let state_value = self.state.load(Ordering::Relaxed);
        match state_value {
            0 => CircuitState::Closed,
            1 => CircuitState::Open,
            2 => CircuitState::HalfOpen,
            _ => CircuitState::Closed,
        }
    }

    /// Check if circuit allows requests
    pub fn is_available(&self) -> bool {
        let current_state = self.state();

        match current_state {
            CircuitState::Closed => true,
            CircuitState::HalfOpen => true,
            CircuitState::Open => {
                // Check if timeout has elapsed
                let last_change = *self.last_state_change.lock();
                if last_change.elapsed() >= self.timeout {
                    // Move to half-open
                    self.transition_to(CircuitState::HalfOpen);
                    true
                } else {
                    false
                }
            }
        }
    }

    /// Record a successful operation
    pub fn record_success(&self) {
        let current_state = self.state();

        match current_state {
            CircuitState::Closed => {
                // Reset failure count on success
                self.failure_count.store(0, Ordering::Relaxed);
            }
            CircuitState::HalfOpen => {
                let successes = self.success_count.fetch_add(1, Ordering::Relaxed) + 1;
                if successes >= self.success_threshold {
                    // Close the circuit
                    self.transition_to(CircuitState::Closed);
                    self.success_count.store(0, Ordering::Relaxed);
                    self.failure_count.store(0, Ordering::Relaxed);
                }
            }
            CircuitState::Open => {
                // Shouldn't happen, but reset if it does
                self.transition_to(CircuitState::Closed);
                self.failure_count.store(0, Ordering::Relaxed);
            }
        }
    }

    /// Record a failed operation
    pub fn record_failure(&self) {
        let current_state = self.state();

        match current_state {
            CircuitState::Closed => {
                let failures = self.failure_count.fetch_add(1, Ordering::Relaxed) + 1;
                if failures >= self.failure_threshold {
                    // Open the circuit
                    self.transition_to(CircuitState::Open);
                }
            }
            CircuitState::HalfOpen => {
                // Failed in half-open, go back to open
                self.transition_to(CircuitState::Open);
                self.success_count.store(0, Ordering::Relaxed);
            }
            CircuitState::Open => {
                // Already open, nothing to do
            }
        }
    }

    /// Transition to a new state
    fn transition_to(&self, new_state: CircuitState) {
        let old_state = self.state();
        if old_state != new_state {
            self.state.store(new_state as usize, Ordering::Relaxed);
            *self.last_state_change.lock() = Instant::now();
            tracing::info!(
                "Circuit breaker state changed: {:?} -> {:?}",
                old_state,
                new_state
            );
        }
    }

    /// Execute an operation through the circuit breaker
    ///
    /// # Arguments
    /// * `operation` - Async function to execute
    ///
    /// # Returns
    /// Result of the operation or circuit open error
    pub async fn call<F, Fut, T, E>(&self, operation: F) -> Result<T, CircuitBreakerError<E>>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<T, E>>,
    {
        if !self.is_available() {
            return Err(CircuitBreakerError::CircuitOpen);
        }

        match operation().await {
            Ok(result) => {
                self.record_success();
                Ok(result)
            }
            Err(error) => {
                self.record_failure();
                Err(CircuitBreakerError::OperationFailed(error))
            }
        }
    }
}

impl Default for CircuitBreaker {
    fn default() -> Self {
        Self::new(5, 2, Duration::from_secs(60))
    }
}

/// Circuit breaker error
#[derive(Debug)]
pub enum CircuitBreakerError<E> {
    /// Circuit is open, rejecting requests
    CircuitOpen,
    /// Operation failed
    OperationFailed(E),
}

impl<E: std::fmt::Display> std::fmt::Display for CircuitBreakerError<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CircuitBreakerError::CircuitOpen => write!(f, "Circuit breaker is open"),
            CircuitBreakerError::OperationFailed(e) => write!(f, "Operation failed: {}", e),
        }
    }
}

impl<E: std::error::Error> std::error::Error for CircuitBreakerError<E> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_retry_policy_exponential() {
        let policy = RetryPolicy::exponential(3, Duration::from_millis(100));
        assert_eq!(policy.delay_for_attempt(0), Duration::from_millis(100));
        assert_eq!(policy.delay_for_attempt(1), Duration::from_millis(200));
        assert_eq!(policy.delay_for_attempt(2), Duration::from_millis(400));
    }

    #[test]
    fn test_retry_policy_fixed() {
        let policy = RetryPolicy::fixed(3, Duration::from_millis(500));
        assert_eq!(policy.delay_for_attempt(0), Duration::from_millis(500));
        assert_eq!(policy.delay_for_attempt(1), Duration::from_millis(500));
        assert_eq!(policy.delay_for_attempt(2), Duration::from_millis(500));
    }

    #[tokio::test]
    async fn test_retry_async_success() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        let policy = RetryPolicy::fixed(3, Duration::from_millis(10));
        let attempts = Arc::new(AtomicUsize::new(0));
        let attempts_clone = Arc::clone(&attempts);

        let result = retry_async(policy, move || {
            let attempts = Arc::clone(&attempts_clone);
            async move {
                let count = attempts.fetch_add(1, Ordering::Relaxed) + 1;
                if count < 2 {
                    Err("Temporary failure")
                } else {
                    Ok("Success")
                }
            }
        })
        .await;

        assert!(result.is_ok());
        assert_eq!(result.expect("test: retry should succeed"), "Success");
        assert_eq!(attempts.load(Ordering::Relaxed), 2);
    }

    #[tokio::test]
    async fn test_retry_async_failure() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        let policy = RetryPolicy::fixed(3, Duration::from_millis(10));
        let attempts = Arc::new(AtomicUsize::new(0));
        let attempts_clone = Arc::clone(&attempts);

        let result = retry_async(policy, move || {
            let attempts = Arc::clone(&attempts_clone);
            async move {
                attempts.fetch_add(1, Ordering::Relaxed);
                Err::<(), _>("Always fails")
            }
        })
        .await;

        assert!(result.is_err());
        assert_eq!(attempts.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn test_circuit_breaker_creation() {
        let breaker = CircuitBreaker::new(5, 2, Duration::from_secs(60));
        assert_eq!(breaker.state(), CircuitState::Closed);
        assert!(breaker.is_available());
    }

    #[test]
    fn test_circuit_breaker_opens_on_failures() {
        let breaker = CircuitBreaker::new(3, 2, Duration::from_secs(60));
        assert_eq!(breaker.state(), CircuitState::Closed);

        // Record failures
        breaker.record_failure();
        assert_eq!(breaker.state(), CircuitState::Closed);

        breaker.record_failure();
        assert_eq!(breaker.state(), CircuitState::Closed);

        breaker.record_failure();
        assert_eq!(breaker.state(), CircuitState::Open);
        assert!(!breaker.is_available());
    }

    #[test]
    fn test_circuit_breaker_half_open_to_closed() {
        let breaker = CircuitBreaker::new(3, 2, Duration::from_millis(10));

        // Open the circuit
        breaker.record_failure();
        breaker.record_failure();
        breaker.record_failure();
        assert_eq!(breaker.state(), CircuitState::Open);

        // Wait for timeout
        std::thread::sleep(Duration::from_millis(20));

        // Check availability (should transition to half-open)
        assert!(breaker.is_available());
        assert_eq!(breaker.state(), CircuitState::HalfOpen);

        // Record successes to close
        breaker.record_success();
        assert_eq!(breaker.state(), CircuitState::HalfOpen);

        breaker.record_success();
        assert_eq!(breaker.state(), CircuitState::Closed);
    }

    #[test]
    fn test_circuit_breaker_half_open_to_open() {
        let breaker = CircuitBreaker::new(3, 2, Duration::from_millis(10));

        // Open the circuit
        breaker.record_failure();
        breaker.record_failure();
        breaker.record_failure();

        // Wait and transition to half-open
        std::thread::sleep(Duration::from_millis(20));
        assert!(breaker.is_available());
        assert_eq!(breaker.state(), CircuitState::HalfOpen);

        // Fail in half-open state
        breaker.record_failure();
        assert_eq!(breaker.state(), CircuitState::Open);
    }

    #[tokio::test]
    async fn test_circuit_breaker_call_success() {
        let breaker = CircuitBreaker::new(3, 2, Duration::from_secs(60));

        let result = breaker.call(|| async { Ok::<_, String>("Success") }).await;

        assert!(result.is_ok());
        assert_eq!(
            result.expect("test: circuit breaker call should succeed"),
            "Success"
        );
    }

    #[tokio::test]
    async fn test_circuit_breaker_call_failure() {
        let breaker = CircuitBreaker::new(2, 2, Duration::from_secs(60));

        // First failure
        let result = breaker.call(|| async { Err::<(), _>("Error 1") }).await;
        assert!(matches!(
            result,
            Err(CircuitBreakerError::OperationFailed(_))
        ));

        // Second failure - should open circuit
        let result = breaker.call(|| async { Err::<(), _>("Error 2") }).await;
        assert!(matches!(
            result,
            Err(CircuitBreakerError::OperationFailed(_))
        ));

        // Circuit should be open now
        let result = breaker
            .call(|| async { Ok::<_, String>("Should not execute") })
            .await;
        assert!(matches!(result, Err(CircuitBreakerError::CircuitOpen)));
    }
}
