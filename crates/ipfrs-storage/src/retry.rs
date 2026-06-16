//! Retry logic with exponential backoff and jitter
//!
//! Provides sophisticated retry strategies for handling transient failures:
//! - Exponential backoff to prevent overwhelming failing services
//! - Jitter to prevent thundering herd problems
//! - Configurable max attempts and timeouts
//! - Retry condition predicates
//!
//! ## Example
//! ```no_run
//! use ipfrs_storage::RetryPolicy;
//! use std::time::Duration;
//!
//! async fn flaky_operation() -> Result<String, std::io::Error> {
//!     // Your operation that might fail transiently
//!     Ok("success".to_string())
//! }
//!
//! #[tokio::main]
//! async fn main() {
//!     let policy = RetryPolicy::exponential(
//!         Duration::from_millis(100),
//!         3
//!     );
//!
//!     let result = policy.retry(|| flaky_operation()).await;
//!     println!("Result: {:?}", result);
//! }
//! ```

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::future::Future;
use std::time::Duration;
use tokio::time::sleep;

/// Backoff strategy for retries
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BackoffStrategy {
    /// Fixed delay between retries
    Fixed,
    /// Exponential backoff (delay doubles each retry)
    Exponential,
    /// Linear backoff (delay increases linearly)
    Linear,
}

/// Jitter type to add randomness to backoff
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum JitterType {
    /// No jitter
    None,
    /// Full jitter (0 to computed delay)
    Full,
    /// Equal jitter (half computed delay + random half)
    Equal,
    /// Decorrelated jitter (AWS recommended)
    Decorrelated,
}

/// Retry policy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryPolicy {
    /// Maximum number of attempts (including initial attempt)
    pub max_attempts: u32,
    /// Base delay for backoff
    pub base_delay: Duration,
    /// Maximum delay between retries
    pub max_delay: Duration,
    /// Backoff strategy
    pub strategy: BackoffStrategy,
    /// Jitter type
    pub jitter: JitterType,
    /// Multiplier for exponential backoff (default: 2.0)
    pub backoff_multiplier: f64,
    /// Overall timeout for all retry attempts
    pub total_timeout: Option<Duration>,
}

impl RetryPolicy {
    /// Create a new retry policy with exponential backoff
    ///
    /// # Arguments
    /// * `base_delay` - Initial delay between retries
    /// * `max_attempts` - Maximum number of attempts
    pub fn exponential(base_delay: Duration, max_attempts: u32) -> Self {
        Self {
            max_attempts,
            base_delay,
            max_delay: Duration::from_secs(60),
            strategy: BackoffStrategy::Exponential,
            jitter: JitterType::Equal,
            backoff_multiplier: 2.0,
            total_timeout: None,
        }
    }

    /// Create a retry policy with fixed delays
    pub fn fixed(delay: Duration, max_attempts: u32) -> Self {
        Self {
            max_attempts,
            base_delay: delay,
            max_delay: delay,
            strategy: BackoffStrategy::Fixed,
            jitter: JitterType::None,
            backoff_multiplier: 1.0,
            total_timeout: None,
        }
    }

    /// Create a retry policy with linear backoff
    pub fn linear(base_delay: Duration, max_attempts: u32) -> Self {
        Self {
            max_attempts,
            base_delay,
            max_delay: Duration::from_secs(60),
            strategy: BackoffStrategy::Linear,
            jitter: JitterType::Equal,
            backoff_multiplier: 1.0,
            total_timeout: None,
        }
    }

    /// Set maximum delay
    pub fn with_max_delay(mut self, max_delay: Duration) -> Self {
        self.max_delay = max_delay;
        self
    }

    /// Set jitter type
    pub fn with_jitter(mut self, jitter: JitterType) -> Self {
        self.jitter = jitter;
        self
    }

    /// Set total timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.total_timeout = Some(timeout);
        self
    }

    /// Set backoff multiplier
    pub fn with_multiplier(mut self, multiplier: f64) -> Self {
        self.backoff_multiplier = multiplier;
        self
    }

    /// Calculate delay for a given attempt number
    fn calculate_delay(&self, attempt: u32) -> Duration {
        if attempt == 0 {
            return Duration::from_secs(0);
        }

        let base_ms = self.base_delay.as_millis() as f64;

        let computed_delay_ms = match self.strategy {
            BackoffStrategy::Fixed => base_ms,
            BackoffStrategy::Exponential => {
                base_ms * self.backoff_multiplier.powi(attempt as i32 - 1)
            }
            BackoffStrategy::Linear => base_ms * attempt as f64,
        };

        // Cap at max delay
        let capped_ms = computed_delay_ms.min(self.max_delay.as_millis() as f64);

        // Apply jitter
        let final_ms = match self.jitter {
            JitterType::None => capped_ms,
            JitterType::Full => {
                // Random value between 0 and computed delay
                fastrand::f64() * capped_ms
            }
            JitterType::Equal => {
                // Half of computed delay + random half
                capped_ms / 2.0 + (fastrand::f64() * capped_ms / 2.0)
            }
            JitterType::Decorrelated => {
                // AWS recommended: min(max_delay, random(base, last_delay * 3))
                let last_delay = if attempt > 1 {
                    self.calculate_delay(attempt - 1).as_millis() as f64
                } else {
                    base_ms
                };
                let random_delay = base_ms + (fastrand::f64() * (last_delay * 3.0 - base_ms));
                random_delay.min(self.max_delay.as_millis() as f64)
            }
        };

        Duration::from_millis(final_ms as u64)
    }

    /// Execute a function with retry logic
    ///
    /// # Arguments
    /// * `f` - Function to retry
    ///
    /// # Returns
    /// Result of the function or last error
    pub async fn retry<F, Fut, T, E>(&self, mut f: F) -> Result<T>
    where
        F: FnMut() -> Fut,
        Fut: Future<Output = Result<T, E>>,
        E: std::error::Error + Send + Sync + 'static,
    {
        let start_time = std::time::Instant::now();
        let mut last_error = None;

        for attempt in 0..self.max_attempts {
            // Check total timeout
            if let Some(timeout) = self.total_timeout {
                if start_time.elapsed() >= timeout {
                    return Err(anyhow!("Retry timeout exceeded after {attempt} attempts"));
                }
            }

            // Try the operation
            match f().await {
                Ok(result) => return Ok(result),
                Err(e) => {
                    last_error = Some(e);

                    // Don't sleep after the last attempt
                    if attempt + 1 < self.max_attempts {
                        let delay = self.calculate_delay(attempt + 1);
                        sleep(delay).await;
                    }
                }
            }
        }

        // All attempts failed
        if let Some(e) = last_error {
            Err(anyhow!(
                "Operation failed after {} attempts: {}",
                self.max_attempts,
                e
            ))
        } else {
            Err(anyhow!(
                "Operation failed after {} attempts",
                self.max_attempts
            ))
        }
    }
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self::exponential(Duration::from_millis(100), 3)
    }
}

/// Trait for retryable operations
pub trait Retryable<T, E> {
    /// Execute with retry policy
    fn with_retry(self, policy: RetryPolicy) -> impl Future<Output = Result<T>>;
}

/// Retry statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RetryStats {
    /// Total retry attempts made
    pub total_attempts: u64,
    /// Successful operations
    pub successful_ops: u64,
    /// Failed operations (after all retries)
    pub failed_ops: u64,
    /// Total delay time spent in retries
    pub total_delay_ms: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Arc;

    #[tokio::test]
    async fn test_retry_success_first_attempt() {
        let policy = RetryPolicy::exponential(Duration::from_millis(10), 3);

        let result = policy
            .retry(|| async { Ok::<_, std::io::Error>("success") })
            .await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "success");
    }

    #[tokio::test]
    async fn test_retry_success_after_failures() {
        let counter = Arc::new(AtomicU32::new(0));
        let policy = RetryPolicy::exponential(Duration::from_millis(10), 3);

        let counter_clone = counter.clone();
        let result = policy
            .retry(|| {
                let c = counter_clone.clone();
                async move {
                    let count = c.fetch_add(1, Ordering::SeqCst);
                    if count < 2 {
                        Err(std::io::Error::other("Transient failure"))
                    } else {
                        Ok("success")
                    }
                }
            })
            .await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "success");
        assert_eq!(counter.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn test_retry_all_attempts_fail() {
        let policy = RetryPolicy::exponential(Duration::from_millis(10), 3);

        let result = policy
            .retry(|| async { Err::<&str, std::io::Error>(std::io::Error::other("Always fails")) })
            .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_fixed_backoff() {
        let policy = RetryPolicy::fixed(Duration::from_millis(50), 3);

        for i in 1..=3 {
            let delay = policy.calculate_delay(i);
            assert_eq!(delay.as_millis(), 50);
        }
    }

    #[tokio::test]
    async fn test_exponential_backoff() {
        let policy = RetryPolicy::exponential(Duration::from_millis(100), 4);

        // Exponential growth (without jitter for this test)
        let policy_no_jitter = policy.with_jitter(JitterType::None);
        let d1 = policy_no_jitter.calculate_delay(1).as_millis();
        let d2 = policy_no_jitter.calculate_delay(2).as_millis();
        let d3 = policy_no_jitter.calculate_delay(3).as_millis();

        assert_eq!(d1, 100);
        assert_eq!(d2, 200);
        assert_eq!(d3, 400);
    }

    #[tokio::test]
    async fn test_linear_backoff() {
        let policy =
            RetryPolicy::linear(Duration::from_millis(100), 4).with_jitter(JitterType::None);

        let d1 = policy.calculate_delay(1).as_millis();
        let d2 = policy.calculate_delay(2).as_millis();
        let d3 = policy.calculate_delay(3).as_millis();

        assert_eq!(d1, 100);
        assert_eq!(d2, 200);
        assert_eq!(d3, 300);
    }

    #[tokio::test]
    async fn test_max_delay_cap() {
        let policy = RetryPolicy::exponential(Duration::from_millis(100), 10)
            .with_max_delay(Duration::from_millis(500))
            .with_jitter(JitterType::None);

        let delay = policy.calculate_delay(5);
        assert!(delay.as_millis() <= 500);
    }

    #[tokio::test]
    async fn test_jitter_full() {
        let policy =
            RetryPolicy::exponential(Duration::from_millis(100), 3).with_jitter(JitterType::Full);

        // With full jitter, delay should be between 0 and computed delay
        for _ in 0..10 {
            let delay = policy.calculate_delay(1);
            assert!(delay.as_millis() <= 100);
        }
    }

    #[tokio::test]
    async fn test_jitter_equal() {
        let policy =
            RetryPolicy::exponential(Duration::from_millis(100), 3).with_jitter(JitterType::Equal);

        // With equal jitter, delay should be between 50 and 100
        for _ in 0..10 {
            let delay = policy.calculate_delay(1);
            let ms = delay.as_millis();
            assert!((50..=100).contains(&ms));
        }
    }

    #[tokio::test]
    async fn test_timeout() {
        let policy = RetryPolicy::exponential(Duration::from_millis(50), 10)
            .with_timeout(Duration::from_millis(150));

        let start = std::time::Instant::now();
        let result = policy
            .retry(|| async { Err::<&str, std::io::Error>(std::io::Error::other("Always fails")) })
            .await;

        let elapsed = start.elapsed();
        assert!(result.is_err());
        // Should timeout before all retries complete, but allow some margin
        assert!(elapsed < Duration::from_millis(500));
        assert!(elapsed >= Duration::from_millis(150));
    }
}
