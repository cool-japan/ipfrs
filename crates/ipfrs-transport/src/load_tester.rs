//! Load testing utilities for transport layer
//!
//! This module provides tools to test the transport layer under various load conditions.
//!
//! # Example
//!
//! ```
//! use ipfrs_transport::load_tester::{LoadTester, LoadTestConfig, LoadPattern};
//!
//! let config = LoadTestConfig {
//!     duration_secs: 10,
//!     pattern: LoadPattern::Constant(100),
//!     block_size_bytes: 1024,
//!     concurrent_requests: 10,
//! };
//!
//! let tester = LoadTester::new(config);
//! let stats = tester.stats();
//! assert_eq!(stats.total_requests, 0);
//! ```

use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Load pattern for testing
#[derive(Debug, Clone)]
pub enum LoadPattern {
    /// Constant rate (requests per second)
    Constant(usize),
    /// Linear ramp from min to max requests per second
    Ramp { min: usize, max: usize },
    /// Step pattern with different rates
    Step {
        steps: Vec<(usize, Duration)>, // (requests_per_sec, duration)
    },
    /// Spike pattern (burst followed by normal)
    Spike {
        normal_rate: usize,
        spike_rate: usize,
        spike_duration: Duration,
        spike_interval: Duration,
    },
    /// Random rate between min and max
    Random { min: usize, max: usize },
}

/// Configuration for load testing
#[derive(Debug, Clone)]
pub struct LoadTestConfig {
    /// Test duration in seconds
    pub duration_secs: u64,
    /// Load pattern to use
    pub pattern: LoadPattern,
    /// Size of blocks to request (bytes)
    pub block_size_bytes: usize,
    /// Number of concurrent requests
    pub concurrent_requests: usize,
}

impl Default for LoadTestConfig {
    fn default() -> Self {
        Self {
            duration_secs: 60,
            pattern: LoadPattern::Constant(100),
            block_size_bytes: 1024,
            concurrent_requests: 10,
        }
    }
}

/// Statistics from a load test
#[derive(Debug, Clone)]
pub struct LoadTestStats {
    /// Total number of requests sent
    pub total_requests: usize,
    /// Total number of successful responses
    pub successful_responses: usize,
    /// Total number of failures
    pub failures: usize,
    /// Total bytes transferred
    pub bytes_transferred: u64,
    /// Test duration
    pub duration: Duration,
    /// Average latency (milliseconds)
    pub avg_latency_ms: f64,
    /// p50 latency (milliseconds)
    pub p50_latency_ms: f64,
    /// p95 latency (milliseconds)
    pub p95_latency_ms: f64,
    /// p99 latency (milliseconds)
    pub p99_latency_ms: f64,
    /// Requests per second achieved
    pub requests_per_second: f64,
    /// Throughput in bytes per second
    pub throughput_bps: f64,
}

impl Default for LoadTestStats {
    fn default() -> Self {
        Self {
            total_requests: 0,
            successful_responses: 0,
            failures: 0,
            bytes_transferred: 0,
            duration: Duration::from_secs(0),
            avg_latency_ms: 0.0,
            p50_latency_ms: 0.0,
            p95_latency_ms: 0.0,
            p99_latency_ms: 0.0,
            requests_per_second: 0.0,
            throughput_bps: 0.0,
        }
    }
}

impl std::fmt::Display for LoadTestStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Load Test Results:")?;
        writeln!(f, "  Duration: {:?}", self.duration)?;
        writeln!(f, "  Total Requests: {}", self.total_requests)?;
        writeln!(f, "  Successful: {}", self.successful_responses)?;
        writeln!(f, "  Failures: {}", self.failures)?;
        writeln!(f, "  Bytes Transferred: {}", self.bytes_transferred)?;
        writeln!(f, "  Requests/sec: {:.2}", self.requests_per_second)?;
        writeln!(
            f,
            "  Throughput: {:.2} MB/s",
            self.throughput_bps / 1_000_000.0
        )?;
        writeln!(f, "  Avg Latency: {:.2}ms", self.avg_latency_ms)?;
        writeln!(f, "  p50 Latency: {:.2}ms", self.p50_latency_ms)?;
        writeln!(f, "  p95 Latency: {:.2}ms", self.p95_latency_ms)?;
        writeln!(f, "  p99 Latency: {:.2}ms", self.p99_latency_ms)?;
        Ok(())
    }
}

/// Load tester for transport layer
pub struct LoadTester {
    config: LoadTestConfig,
    stats: LoadTestStats,
    latencies: VecDeque<u64>,
    start_time: Option<Instant>,
}

impl LoadTester {
    /// Create a new load tester
    pub fn new(config: LoadTestConfig) -> Self {
        Self {
            config,
            stats: LoadTestStats::default(),
            latencies: VecDeque::new(),
            start_time: None,
        }
    }

    /// Start the load test
    pub fn start(&mut self) {
        self.start_time = Some(Instant::now());
        self.stats = LoadTestStats::default();
        self.latencies.clear();
    }

    /// Record a successful request
    pub fn record_success(&mut self, latency_ms: u64, bytes: usize) {
        self.stats.total_requests += 1;
        self.stats.successful_responses += 1;
        self.stats.bytes_transferred += bytes as u64;
        self.latencies.push_back(latency_ms);

        // Keep only recent latencies (for memory efficiency)
        if self.latencies.len() > 10000 {
            self.latencies.pop_front();
        }
    }

    /// Record a failed request
    pub fn record_failure(&mut self) {
        self.stats.total_requests += 1;
        self.stats.failures += 1;
    }

    /// Get current statistics
    pub fn stats(&self) -> &LoadTestStats {
        &self.stats
    }

    /// Calculate and finalize statistics
    pub fn finalize(&mut self) -> LoadTestStats {
        if let Some(start) = self.start_time {
            self.stats.duration = start.elapsed();
        }

        // Calculate latency percentiles
        if !self.latencies.is_empty() {
            let mut sorted: Vec<u64> = self.latencies.iter().copied().collect();
            sorted.sort_unstable();

            let sum: u64 = sorted.iter().sum();
            self.stats.avg_latency_ms = sum as f64 / sorted.len() as f64;

            let p50_idx = (sorted.len() as f64 * 0.50) as usize;
            let p95_idx = (sorted.len() as f64 * 0.95) as usize;
            let p99_idx = (sorted.len() as f64 * 0.99) as usize;

            self.stats.p50_latency_ms = sorted.get(p50_idx).copied().unwrap_or(0) as f64;
            self.stats.p95_latency_ms = sorted.get(p95_idx).copied().unwrap_or(0) as f64;
            self.stats.p99_latency_ms = sorted.get(p99_idx).copied().unwrap_or(0) as f64;
        }

        // Calculate throughput
        let duration_secs = self.stats.duration.as_secs_f64();
        if duration_secs > 0.0 {
            self.stats.requests_per_second = self.stats.total_requests as f64 / duration_secs;
            self.stats.throughput_bps = self.stats.bytes_transferred as f64 / duration_secs;
        }

        self.stats.clone()
    }

    /// Get the target request rate at a given time offset
    pub fn get_target_rate(&self, elapsed: Duration) -> usize {
        match &self.config.pattern {
            LoadPattern::Constant(rate) => *rate,
            LoadPattern::Ramp { min, max } => {
                let progress = elapsed.as_secs_f64() / self.config.duration_secs as f64;
                let range = (*max - *min) as f64;
                (*min as f64 + range * progress) as usize
            }
            LoadPattern::Step { steps } => {
                let mut accumulated = Duration::from_secs(0);
                for (rate, duration) in steps {
                    accumulated += *duration;
                    if elapsed < accumulated {
                        return *rate;
                    }
                }
                steps.last().map(|(rate, _)| *rate).unwrap_or(0)
            }
            LoadPattern::Spike {
                normal_rate,
                spike_rate,
                spike_duration,
                spike_interval,
            } => {
                let cycle_time = elapsed.as_secs_f64() % spike_interval.as_secs_f64();
                if cycle_time < spike_duration.as_secs_f64() {
                    *spike_rate
                } else {
                    *normal_rate
                }
            }
            LoadPattern::Random { min, max } => {
                // Simple pseudo-random (not cryptographically secure)
                let seed = elapsed.as_millis() as usize;
                min + (seed % (max - min + 1))
            }
        }
    }

    /// Get configuration
    pub fn config(&self) -> &LoadTestConfig {
        &self.config
    }

    /// Reset the tester
    pub fn reset(&mut self) {
        self.stats = LoadTestStats::default();
        self.latencies.clear();
        self.start_time = None;
    }
}

/// Builder for load test configuration
pub struct LoadTestConfigBuilder {
    config: LoadTestConfig,
}

impl LoadTestConfigBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            config: LoadTestConfig::default(),
        }
    }

    /// Set test duration
    pub fn duration_secs(mut self, secs: u64) -> Self {
        self.config.duration_secs = secs;
        self
    }

    /// Set load pattern
    pub fn pattern(mut self, pattern: LoadPattern) -> Self {
        self.config.pattern = pattern;
        self
    }

    /// Set block size
    pub fn block_size_bytes(mut self, bytes: usize) -> Self {
        self.config.block_size_bytes = bytes;
        self
    }

    /// Set concurrent requests
    pub fn concurrent_requests(mut self, count: usize) -> Self {
        self.config.concurrent_requests = count;
        self
    }

    /// Build the configuration
    pub fn build(self) -> LoadTestConfig {
        self.config
    }
}

impl Default for LoadTestConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_tester_creation() {
        let config = LoadTestConfig::default();
        let tester = LoadTester::new(config);
        assert_eq!(tester.stats().total_requests, 0);
    }

    #[test]
    fn test_record_success() {
        let config = LoadTestConfig::default();
        let mut tester = LoadTester::new(config);

        tester.start();
        tester.record_success(50, 1024);

        assert_eq!(tester.stats().total_requests, 1);
        assert_eq!(tester.stats().successful_responses, 1);
        assert_eq!(tester.stats().bytes_transferred, 1024);
    }

    #[test]
    fn test_record_failure() {
        let config = LoadTestConfig::default();
        let mut tester = LoadTester::new(config);

        tester.start();
        tester.record_failure();

        assert_eq!(tester.stats().total_requests, 1);
        assert_eq!(tester.stats().failures, 1);
    }

    #[test]
    fn test_finalize_stats() {
        let config = LoadTestConfig::default();
        let mut tester = LoadTester::new(config);

        tester.start();
        tester.record_success(50, 1024);
        tester.record_success(60, 1024);
        tester.record_success(70, 1024);

        // Ensure measurable elapsed time for throughput calculation
        std::thread::sleep(Duration::from_millis(1));

        let stats = tester.finalize();
        assert_eq!(stats.total_requests, 3);
        assert!(stats.avg_latency_ms > 0.0);
        assert!(stats.throughput_bps > 0.0);
    }

    #[test]
    fn test_constant_load_pattern() {
        let config = LoadTestConfig {
            pattern: LoadPattern::Constant(100),
            ..Default::default()
        };
        let tester = LoadTester::new(config);

        assert_eq!(tester.get_target_rate(Duration::from_secs(0)), 100);
        assert_eq!(tester.get_target_rate(Duration::from_secs(30)), 100);
    }

    #[test]
    fn test_ramp_load_pattern() {
        let config = LoadTestConfig {
            duration_secs: 10,
            pattern: LoadPattern::Ramp { min: 10, max: 100 },
            ..Default::default()
        };
        let tester = LoadTester::new(config);

        let rate_start = tester.get_target_rate(Duration::from_secs(0));
        let rate_end = tester.get_target_rate(Duration::from_secs(10));

        assert_eq!(rate_start, 10);
        assert_eq!(rate_end, 100);
    }

    #[test]
    fn test_step_load_pattern() {
        let config = LoadTestConfig {
            pattern: LoadPattern::Step {
                steps: vec![
                    (10, Duration::from_secs(5)),
                    (50, Duration::from_secs(5)),
                    (100, Duration::from_secs(5)),
                ],
            },
            ..Default::default()
        };
        let tester = LoadTester::new(config);

        assert_eq!(tester.get_target_rate(Duration::from_secs(2)), 10);
        assert_eq!(tester.get_target_rate(Duration::from_secs(7)), 50);
        assert_eq!(tester.get_target_rate(Duration::from_secs(12)), 100);
    }

    #[test]
    fn test_spike_load_pattern() {
        let config = LoadTestConfig {
            pattern: LoadPattern::Spike {
                normal_rate: 10,
                spike_rate: 100,
                spike_duration: Duration::from_secs(2),
                spike_interval: Duration::from_secs(10),
            },
            ..Default::default()
        };
        let tester = LoadTester::new(config);

        assert_eq!(tester.get_target_rate(Duration::from_secs(1)), 100); // In spike
        assert_eq!(tester.get_target_rate(Duration::from_secs(5)), 10); // Normal
    }

    #[test]
    fn test_config_builder() {
        let config = LoadTestConfigBuilder::new()
            .duration_secs(30)
            .pattern(LoadPattern::Constant(50))
            .block_size_bytes(2048)
            .concurrent_requests(20)
            .build();

        assert_eq!(config.duration_secs, 30);
        assert_eq!(config.block_size_bytes, 2048);
        assert_eq!(config.concurrent_requests, 20);
    }

    #[test]
    fn test_reset() {
        let config = LoadTestConfig::default();
        let mut tester = LoadTester::new(config);

        tester.start();
        tester.record_success(50, 1024);

        assert_eq!(tester.stats().total_requests, 1);

        tester.reset();
        assert_eq!(tester.stats().total_requests, 0);
    }

    #[test]
    fn test_percentile_calculation() {
        let config = LoadTestConfig::default();
        let mut tester = LoadTester::new(config);

        tester.start();
        for i in 1..=100 {
            tester.record_success(i, 1024);
        }

        let stats = tester.finalize();
        assert!(stats.p50_latency_ms >= 45.0 && stats.p50_latency_ms <= 55.0);
        assert!(stats.p95_latency_ms >= 90.0 && stats.p95_latency_ms <= 100.0);
        assert!(stats.p99_latency_ms >= 95.0 && stats.p99_latency_ms <= 100.0);
    }

    #[test]
    fn test_stats_display() {
        let stats = LoadTestStats {
            total_requests: 100,
            successful_responses: 95,
            failures: 5,
            bytes_transferred: 102400,
            duration: Duration::from_secs(10),
            avg_latency_ms: 50.0,
            p50_latency_ms: 45.0,
            p95_latency_ms: 90.0,
            p99_latency_ms: 95.0,
            requests_per_second: 10.0,
            throughput_bps: 10240.0,
        };

        let display = format!("{}", stats);
        assert!(display.contains("Total Requests: 100"));
        assert!(display.contains("Successful: 95"));
    }
}
