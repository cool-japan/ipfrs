//! Bandwidth throttling for network traffic control
//!
//! This module provides bandwidth limiting capabilities using a token bucket algorithm.
//! It's designed for edge devices, mobile networks, and scenarios where bandwidth
//! needs to be controlled for power efficiency or network congestion management.

use parking_lot::Mutex;
use std::sync::Arc;
use std::time::{Duration, Instant};
use thiserror::Error;

/// Errors that can occur during throttling operations
#[derive(Error, Debug, Clone)]
pub enum ThrottleError {
    #[error("Bandwidth limit exceeded, retry after {0:?}")]
    RateLimitExceeded(Duration),

    #[error("Invalid throttle configuration: {0}")]
    InvalidConfig(String),

    #[error("Throttle disabled")]
    Disabled,
}

/// Configuration for bandwidth throttling
#[derive(Debug, Clone)]
pub struct ThrottleConfig {
    /// Maximum bytes per second for upload (None = unlimited)
    pub max_upload_bytes_per_sec: Option<u64>,

    /// Maximum bytes per second for download (None = unlimited)
    pub max_download_bytes_per_sec: Option<u64>,

    /// Burst size in bytes (allows temporary exceeding of rate limit)
    /// Default: 2x the per-second limit
    pub burst_size_bytes: Option<u64>,

    /// Whether throttling is enabled
    pub enabled: bool,

    /// Refill interval for token bucket (smaller = more granular)
    pub refill_interval: Duration,
}

impl Default for ThrottleConfig {
    fn default() -> Self {
        Self {
            max_upload_bytes_per_sec: None,
            max_download_bytes_per_sec: None,
            burst_size_bytes: None,
            enabled: false,
            refill_interval: Duration::from_millis(100), // 100ms refill interval
        }
    }
}

impl ThrottleConfig {
    /// Create a configuration for mobile/cellular networks
    /// Limits: 1 MB/s upload, 5 MB/s download
    pub fn mobile() -> Self {
        Self {
            max_upload_bytes_per_sec: Some(1_000_000),   // 1 MB/s
            max_download_bytes_per_sec: Some(5_000_000), // 5 MB/s
            burst_size_bytes: Some(2_000_000),           // 2 MB burst
            enabled: true,
            refill_interval: Duration::from_millis(100),
        }
    }

    /// Create a configuration for IoT/edge devices with limited bandwidth
    /// Limits: 128 KB/s upload, 512 KB/s download
    pub fn iot() -> Self {
        Self {
            max_upload_bytes_per_sec: Some(128_000),   // 128 KB/s
            max_download_bytes_per_sec: Some(512_000), // 512 KB/s
            burst_size_bytes: Some(256_000),           // 256 KB burst
            enabled: true,
            refill_interval: Duration::from_millis(100),
        }
    }

    /// Create a configuration for low-power mode
    /// Very conservative limits: 64 KB/s upload, 256 KB/s download
    pub fn low_power() -> Self {
        Self {
            max_upload_bytes_per_sec: Some(64_000),    // 64 KB/s
            max_download_bytes_per_sec: Some(256_000), // 256 KB/s
            burst_size_bytes: Some(128_000),           // 128 KB burst
            enabled: true,
            refill_interval: Duration::from_millis(200), // Longer interval = less frequent checks
        }
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<(), ThrottleError> {
        if self.refill_interval.is_zero() {
            return Err(ThrottleError::InvalidConfig(
                "Refill interval must be > 0".to_string(),
            ));
        }

        if let Some(burst) = self.burst_size_bytes {
            if burst == 0 {
                return Err(ThrottleError::InvalidConfig(
                    "Burst size must be > 0 if specified".to_string(),
                ));
            }
        }

        Ok(())
    }
}

/// Token bucket state for rate limiting
#[derive(Debug)]
struct TokenBucket {
    /// Current number of tokens (bytes available)
    tokens: f64,

    /// Maximum number of tokens (burst capacity)
    capacity: f64,

    /// Token refill rate (bytes per second)
    refill_rate: f64,

    /// Last refill time
    last_refill: Instant,

    /// Refill interval
    refill_interval: Duration,
}

impl TokenBucket {
    fn new(rate_bytes_per_sec: u64, burst_bytes: u64, refill_interval: Duration) -> Self {
        Self {
            tokens: burst_bytes as f64,
            capacity: burst_bytes as f64,
            refill_rate: rate_bytes_per_sec as f64,
            last_refill: Instant::now(),
            refill_interval,
        }
    }

    /// Refill tokens based on time elapsed
    fn refill(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_refill);

        if elapsed >= self.refill_interval {
            let tokens_to_add = self.refill_rate * elapsed.as_secs_f64();
            self.tokens = (self.tokens + tokens_to_add).min(self.capacity);
            self.last_refill = now;
        }
    }

    /// Try to consume tokens
    /// Returns Ok(()) if successful, or Err with retry duration
    fn consume(&mut self, bytes: u64) -> Result<(), Duration> {
        self.refill();

        if self.tokens >= bytes as f64 {
            self.tokens -= bytes as f64;
            Ok(())
        } else {
            // Calculate how long to wait
            let tokens_needed = bytes as f64 - self.tokens;
            let wait_time = Duration::from_secs_f64(tokens_needed / self.refill_rate);
            Err(wait_time)
        }
    }

    /// Get current available tokens
    fn available_tokens(&mut self) -> u64 {
        self.refill();
        self.tokens as u64
    }
}

/// Direction of traffic for throttling
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrafficDirection {
    Upload,
    Download,
}

/// Bandwidth throttler using token bucket algorithm
#[derive(Clone)]
pub struct BandwidthThrottle {
    config: ThrottleConfig,
    upload_bucket: Arc<Mutex<Option<TokenBucket>>>,
    download_bucket: Arc<Mutex<Option<TokenBucket>>>,
}

impl BandwidthThrottle {
    /// Create a new bandwidth throttle with the given configuration
    pub fn new(config: ThrottleConfig) -> Result<Self, ThrottleError> {
        config.validate()?;

        let upload_bucket = config.max_upload_bytes_per_sec.map(|rate| {
            let burst = config.burst_size_bytes.unwrap_or(rate * 2);
            TokenBucket::new(rate, burst, config.refill_interval)
        });

        let download_bucket = config.max_download_bytes_per_sec.map(|rate| {
            let burst = config.burst_size_bytes.unwrap_or(rate * 2);
            TokenBucket::new(rate, burst, config.refill_interval)
        });

        Ok(Self {
            config: config.clone(),
            upload_bucket: Arc::new(Mutex::new(upload_bucket)),
            download_bucket: Arc::new(Mutex::new(download_bucket)),
        })
    }

    /// Check if data transfer is allowed
    /// Returns Ok(()) if allowed, or Err with retry duration
    pub fn check_and_consume(
        &self,
        direction: TrafficDirection,
        bytes: u64,
    ) -> Result<(), ThrottleError> {
        if !self.config.enabled {
            return Err(ThrottleError::Disabled);
        }

        let bucket = match direction {
            TrafficDirection::Upload => &self.upload_bucket,
            TrafficDirection::Download => &self.download_bucket,
        };

        let mut guard = bucket.lock();
        if let Some(bucket) = guard.as_mut() {
            bucket
                .consume(bytes)
                .map_err(ThrottleError::RateLimitExceeded)
        } else {
            // No limit configured for this direction
            Ok(())
        }
    }

    /// Get available bandwidth in bytes for the given direction
    pub fn available_bandwidth(&self, direction: TrafficDirection) -> Option<u64> {
        if !self.config.enabled {
            return None;
        }

        let bucket = match direction {
            TrafficDirection::Upload => &self.upload_bucket,
            TrafficDirection::Download => &self.download_bucket,
        };

        let mut guard = bucket.lock();
        guard.as_mut().map(|b| b.available_tokens())
    }

    /// Enable or disable throttling
    pub fn set_enabled(&mut self, enabled: bool) {
        Arc::make_mut(&mut Arc::new(self.config.clone())).enabled = enabled;
    }

    /// Check if throttling is enabled
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Get the current configuration
    pub fn config(&self) -> &ThrottleConfig {
        &self.config
    }

    /// Update throttle configuration
    pub fn update_config(&mut self, config: ThrottleConfig) -> Result<(), ThrottleError> {
        config.validate()?;

        // Recreate buckets with new config
        let upload_bucket = config.max_upload_bytes_per_sec.map(|rate| {
            let burst = config.burst_size_bytes.unwrap_or(rate * 2);
            TokenBucket::new(rate, burst, config.refill_interval)
        });

        let download_bucket = config.max_download_bytes_per_sec.map(|rate| {
            let burst = config.burst_size_bytes.unwrap_or(rate * 2);
            TokenBucket::new(rate, burst, config.refill_interval)
        });

        *self.upload_bucket.lock() = upload_bucket;
        *self.download_bucket.lock() = download_bucket;
        self.config = config;

        Ok(())
    }
}

/// Statistics for throttling
#[derive(Debug, Clone, Default)]
pub struct ThrottleStats {
    /// Total bytes allowed (upload)
    pub upload_bytes_allowed: u64,

    /// Total bytes throttled/delayed (upload)
    pub upload_bytes_throttled: u64,

    /// Total bytes allowed (download)
    pub download_bytes_allowed: u64,

    /// Total bytes throttled/delayed (download)
    pub download_bytes_throttled: u64,

    /// Number of times upload was throttled
    pub upload_throttle_count: u64,

    /// Number of times download was throttled
    pub download_throttle_count: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_throttle_config_default() {
        let config = ThrottleConfig::default();
        assert!(!config.enabled);
        assert!(config.max_upload_bytes_per_sec.is_none());
        assert!(config.max_download_bytes_per_sec.is_none());
    }

    #[test]
    fn test_throttle_config_mobile() {
        let config = ThrottleConfig::mobile();
        assert!(config.enabled);
        assert_eq!(config.max_upload_bytes_per_sec, Some(1_000_000));
        assert_eq!(config.max_download_bytes_per_sec, Some(5_000_000));
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_throttle_config_iot() {
        let config = ThrottleConfig::iot();
        assert!(config.enabled);
        assert_eq!(config.max_upload_bytes_per_sec, Some(128_000));
        assert_eq!(config.max_download_bytes_per_sec, Some(512_000));
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_throttle_config_low_power() {
        let config = ThrottleConfig::low_power();
        assert!(config.enabled);
        assert_eq!(config.max_upload_bytes_per_sec, Some(64_000));
        assert_eq!(config.max_download_bytes_per_sec, Some(256_000));
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_throttle_disabled() {
        let config = ThrottleConfig::default();
        let throttle =
            BandwidthThrottle::new(config).expect("test: default config should create throttle");

        // Should fail when disabled
        let result = throttle.check_and_consume(TrafficDirection::Upload, 1000);
        assert!(matches!(result, Err(ThrottleError::Disabled)));
    }

    #[test]
    fn test_throttle_upload_within_limit() {
        let config = ThrottleConfig {
            enabled: true,
            max_upload_bytes_per_sec: Some(1000),
            burst_size_bytes: Some(2000),
            ..Default::default()
        };

        let throttle = BandwidthThrottle::new(config)
            .expect("test: upload within limit config should create throttle");

        // Should succeed within burst limit
        let result = throttle.check_and_consume(TrafficDirection::Upload, 1500);
        assert!(result.is_ok());
    }

    #[test]
    fn test_throttle_upload_exceeds_limit() {
        let config = ThrottleConfig {
            enabled: true,
            max_upload_bytes_per_sec: Some(1000),
            burst_size_bytes: Some(2000),
            ..Default::default()
        };

        let throttle = BandwidthThrottle::new(config)
            .expect("test: upload exceeds limit config should create throttle");

        // Consume all burst
        let _ = throttle.check_and_consume(TrafficDirection::Upload, 2000);

        // Next request should be throttled
        let result = throttle.check_and_consume(TrafficDirection::Upload, 100);
        assert!(matches!(result, Err(ThrottleError::RateLimitExceeded(_))));
    }

    #[test]
    fn test_throttle_download_within_limit() {
        let config = ThrottleConfig {
            enabled: true,
            max_download_bytes_per_sec: Some(5000),
            burst_size_bytes: Some(10000),
            ..Default::default()
        };

        let throttle = BandwidthThrottle::new(config)
            .expect("test: download within limit config should create throttle");

        // Should succeed within burst limit
        let result = throttle.check_and_consume(TrafficDirection::Download, 8000);
        assert!(result.is_ok());
    }

    #[test]
    fn test_throttle_refill() {
        let config = ThrottleConfig {
            enabled: true,
            max_upload_bytes_per_sec: Some(1000),
            burst_size_bytes: Some(1000),
            refill_interval: Duration::from_millis(100),
            ..Default::default()
        };

        let throttle =
            BandwidthThrottle::new(config).expect("test: refill config should create throttle");

        // Consume all tokens
        let _ = throttle.check_and_consume(TrafficDirection::Upload, 1000);

        // Wait for refill
        thread::sleep(Duration::from_millis(150));

        // Should have some tokens available now
        let available = throttle.available_bandwidth(TrafficDirection::Upload);
        assert!(available.is_some());
        assert!(available.expect("test: bandwidth should be available after refill") > 0);
    }

    #[test]
    fn test_throttle_available_bandwidth() {
        let config = ThrottleConfig {
            enabled: true,
            max_upload_bytes_per_sec: Some(1000),
            burst_size_bytes: Some(2000),
            ..Default::default()
        };

        let throttle = BandwidthThrottle::new(config)
            .expect("test: available bandwidth config should create throttle");

        let available = throttle.available_bandwidth(TrafficDirection::Upload);
        assert_eq!(available, Some(2000)); // Should equal burst size initially
    }

    #[test]
    fn test_throttle_independent_directions() {
        let config = ThrottleConfig {
            enabled: true,
            max_upload_bytes_per_sec: Some(1000),
            max_download_bytes_per_sec: Some(5000),
            burst_size_bytes: Some(2000),
            ..Default::default()
        };

        let throttle = BandwidthThrottle::new(config)
            .expect("test: independent directions config should create throttle");

        // Consume upload tokens
        let _ = throttle.check_and_consume(TrafficDirection::Upload, 2000);

        // Download should still work
        let result = throttle.check_and_consume(TrafficDirection::Download, 2000);
        assert!(result.is_ok());
    }

    #[test]
    fn test_throttle_update_config() {
        let config = ThrottleConfig {
            enabled: true,
            max_upload_bytes_per_sec: Some(1000),
            ..Default::default()
        };

        let mut throttle = BandwidthThrottle::new(config)
            .expect("test: update config initial config should create throttle");

        // Update to higher limit
        let new_config = ThrottleConfig {
            enabled: true,
            max_upload_bytes_per_sec: Some(5000),
            burst_size_bytes: Some(10000),
            ..Default::default()
        };

        throttle
            .update_config(new_config)
            .expect("test: update_config with valid new config should succeed");

        // Should have more bandwidth available
        let available = throttle.available_bandwidth(TrafficDirection::Upload);
        assert_eq!(available, Some(10000));
    }

    #[test]
    fn test_throttle_config_validation() {
        let config = ThrottleConfig {
            refill_interval: Duration::from_secs(0),
            ..Default::default()
        };

        let result = BandwidthThrottle::new(config);
        assert!(matches!(result, Err(ThrottleError::InvalidConfig(_))));
    }

    #[test]
    fn test_throttle_no_limit_direction() {
        let config = ThrottleConfig {
            enabled: true,
            max_upload_bytes_per_sec: Some(1000),
            // No download limit
            ..Default::default()
        };

        let throttle = BandwidthThrottle::new(config)
            .expect("test: no limit direction config should create throttle");

        // Download should succeed without limit
        let result = throttle.check_and_consume(TrafficDirection::Download, 1_000_000);
        assert!(result.is_ok());
    }
}
