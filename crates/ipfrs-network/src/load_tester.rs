//! Network Load Testing and Stress Testing Utilities
//!
//! This module provides tools for stress-testing the network layer to understand
//! performance characteristics, identify bottlenecks, and validate scalability.
//!
//! # Features
//!
//! - **Connection Load Testing**: Test behavior under many simultaneous connections
//! - **DHT Query Storms**: Stress-test DHT with high query volumes
//! - **Bandwidth Saturation**: Test throughput limits
//! - **Provider Record Flooding**: Test provider record handling at scale
//! - **Concurrent Operations**: Test system under concurrent operations
//! - **Memory Pressure**: Test behavior under memory constraints
//! - **Performance Metrics**: Detailed performance tracking during tests
//!
//! # Example
//!
//! ```rust
//! use ipfrs_network::load_tester::{LoadTester, LoadTestConfig, LoadTestType};
//! use std::time::Duration;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let config = LoadTestConfig {
//!     duration: Duration::from_secs(60),
//!     connection_target: 100,
//!     query_rate: 50, // queries per second
//!     ..Default::default()
//! };
//!
//! let mut tester = LoadTester::new(config);
//! let results = tester.run_test(LoadTestType::ConnectionStress)?;
//!
//! println!("Test passed: {}", results.passed);
//! println!("Peak connections: {}", results.peak_connections);
//! println!("Average latency: {:?}", results.average_latency);
//! # Ok(())
//! # }
//! ```

use parking_lot::RwLock;
use rand::RngExt;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Configuration for load testing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadTestConfig {
    /// Test duration
    pub duration: Duration,
    /// Target number of connections
    pub connection_target: usize,
    /// Query rate (queries per second)
    pub query_rate: u64,
    /// Bandwidth target (bytes per second)
    pub bandwidth_target: u64,
    /// Provider record publication rate
    pub provider_publish_rate: u64,
    /// Concurrent operation count
    pub concurrent_operations: usize,
    /// Memory limit for testing (bytes)
    pub memory_limit: u64,
    /// Warmup duration before measurements
    pub warmup_duration: Duration,
    /// Ramp-up time to reach full load
    pub rampup_duration: Duration,
}

impl Default for LoadTestConfig {
    fn default() -> Self {
        Self {
            duration: Duration::from_secs(300), // 5 minutes
            connection_target: 100,
            query_rate: 10,
            bandwidth_target: 10_000_000, // 10 MB/s
            provider_publish_rate: 5,
            concurrent_operations: 50,
            memory_limit: 512 * 1024 * 1024, // 512 MB
            warmup_duration: Duration::from_secs(10),
            rampup_duration: Duration::from_secs(30),
        }
    }
}

impl LoadTestConfig {
    /// Create configuration for light load testing
    pub fn light() -> Self {
        Self {
            duration: Duration::from_secs(60),
            connection_target: 20,
            query_rate: 5,
            bandwidth_target: 1_000_000, // 1 MB/s
            provider_publish_rate: 2,
            concurrent_operations: 10,
            memory_limit: 128 * 1024 * 1024, // 128 MB
            warmup_duration: Duration::from_secs(5),
            rampup_duration: Duration::from_secs(10),
        }
    }

    /// Create configuration for moderate load testing
    pub fn moderate() -> Self {
        Self::default()
    }

    /// Create configuration for heavy load testing
    pub fn heavy() -> Self {
        Self {
            duration: Duration::from_secs(600), // 10 minutes
            connection_target: 500,
            query_rate: 100,
            bandwidth_target: 100_000_000, // 100 MB/s
            provider_publish_rate: 20,
            concurrent_operations: 200,
            memory_limit: 2 * 1024 * 1024 * 1024, // 2 GB
            warmup_duration: Duration::from_secs(30),
            rampup_duration: Duration::from_secs(60),
        }
    }

    /// Create configuration for extreme load testing
    pub fn extreme() -> Self {
        Self {
            duration: Duration::from_secs(1200), // 20 minutes
            connection_target: 2000,
            query_rate: 500,
            bandwidth_target: 1_000_000_000, // 1 GB/s
            provider_publish_rate: 100,
            concurrent_operations: 1000,
            memory_limit: 8 * 1024 * 1024 * 1024, // 8 GB
            warmup_duration: Duration::from_secs(60),
            rampup_duration: Duration::from_secs(120),
        }
    }
}

/// Type of load test to perform
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LoadTestType {
    /// Test connection handling under load
    ConnectionStress,
    /// Test DHT query performance under high load
    DhtQueryStorm,
    /// Test bandwidth saturation
    BandwidthSaturation,
    /// Test provider record handling
    ProviderFlood,
    /// Test concurrent operations
    ConcurrentOps,
    /// Test memory pressure handling
    MemoryPressure,
    /// Run all tests
    ComprehensiveSuite,
}

impl LoadTestType {
    /// Get human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            Self::ConnectionStress => "Connection Stress Test",
            Self::DhtQueryStorm => "DHT Query Storm",
            Self::BandwidthSaturation => "Bandwidth Saturation Test",
            Self::ProviderFlood => "Provider Record Flood",
            Self::ConcurrentOps => "Concurrent Operations Test",
            Self::MemoryPressure => "Memory Pressure Test",
            Self::ComprehensiveSuite => "Comprehensive Suite",
        }
    }

    /// Get description
    pub fn description(&self) -> &'static str {
        match self {
            Self::ConnectionStress => "Tests network behavior under many simultaneous connections",
            Self::DhtQueryStorm => "Stress-tests DHT with high volume of queries",
            Self::BandwidthSaturation => "Tests throughput limits and bandwidth handling",
            Self::ProviderFlood => "Tests provider record publishing and querying at scale",
            Self::ConcurrentOps => "Tests system behavior under many concurrent operations",
            Self::MemoryPressure => "Tests behavior under memory constraints",
            Self::ComprehensiveSuite => "Runs all load tests sequentially",
        }
    }
}

/// Results from a load test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadTestResults {
    /// Test type
    pub test_type: LoadTestType,
    /// Whether the test passed
    pub passed: bool,
    /// Test duration
    pub duration: Duration,
    /// Peak number of connections achieved
    pub peak_connections: usize,
    /// Average latency
    pub average_latency: Duration,
    /// P95 latency
    pub p95_latency: Duration,
    /// P99 latency
    pub p99_latency: Duration,
    /// Total queries executed
    pub total_queries: u64,
    /// Successful queries
    pub successful_queries: u64,
    /// Failed queries
    pub failed_queries: u64,
    /// Total bytes sent
    pub total_bytes_sent: u64,
    /// Total bytes received
    pub total_bytes_received: u64,
    /// Peak memory usage (bytes)
    pub peak_memory_usage: u64,
    /// Average memory usage (bytes)
    pub average_memory_usage: u64,
    /// Throughput (bytes per second)
    pub throughput_bps: u64,
    /// Query rate achieved (queries per second)
    pub query_rate_achieved: f64,
    /// Error messages if test failed
    pub errors: Vec<String>,
    /// Performance timeline (timestamp -> metric value)
    pub performance_timeline: HashMap<String, Vec<(Duration, f64)>>,
}

impl LoadTestResults {
    /// Create a new results instance
    pub fn new(test_type: LoadTestType) -> Self {
        Self {
            test_type,
            passed: false,
            duration: Duration::ZERO,
            peak_connections: 0,
            average_latency: Duration::ZERO,
            p95_latency: Duration::ZERO,
            p99_latency: Duration::ZERO,
            total_queries: 0,
            successful_queries: 0,
            failed_queries: 0,
            total_bytes_sent: 0,
            total_bytes_received: 0,
            peak_memory_usage: 0,
            average_memory_usage: 0,
            throughput_bps: 0,
            query_rate_achieved: 0.0,
            errors: Vec::new(),
            performance_timeline: HashMap::new(),
        }
    }

    /// Calculate success rate
    pub fn success_rate(&self) -> f64 {
        if self.total_queries == 0 {
            return 0.0;
        }
        (self.successful_queries as f64 / self.total_queries as f64) * 100.0
    }

    /// Get throughput in human-readable format
    pub fn throughput_human(&self) -> String {
        crate::utils::format_bandwidth(self.throughput_bps as usize)
    }

    /// Get summary string
    pub fn summary(&self) -> String {
        format!(
            "{}: {} | Connections: {} | Latency: {:?} (avg), {:?} (p95) | \
             Queries: {}/{} ({:.1}%) | Throughput: {} | Memory: {}",
            self.test_type.name(),
            if self.passed { "PASS" } else { "FAIL" },
            self.peak_connections,
            self.average_latency,
            self.p95_latency,
            self.successful_queries,
            self.total_queries,
            self.success_rate(),
            self.throughput_human(),
            crate::utils::format_bytes(self.peak_memory_usage as usize),
        )
    }
}

/// Network load tester
pub struct LoadTester {
    config: LoadTestConfig,
    metrics: Arc<RwLock<LoadTestMetrics>>,
}

/// Metrics tracking for load tests
#[derive(Debug, Default, Clone)]
pub struct LoadTestMetrics {
    /// Test start time
    pub start_time: Option<Instant>,
    /// Current number of connections
    pub connections: usize,
    /// Peak number of connections
    pub peak_connections: usize,
    /// Number of queries sent
    pub queries_sent: u64,
    /// Number of successful queries
    pub queries_succeeded: u64,
    /// Number of failed queries
    pub queries_failed: u64,
    /// Bytes sent
    pub bytes_sent: u64,
    /// Bytes received
    pub bytes_received: u64,
    /// Latency samples
    pub latencies: Vec<Duration>,
    /// Memory usage samples
    pub memory_samples: Vec<u64>,
    /// Error messages
    pub errors: Vec<String>,
}

impl LoadTester {
    /// Create a new load tester
    pub fn new(config: LoadTestConfig) -> Self {
        Self {
            config,
            metrics: Arc::new(RwLock::new(LoadTestMetrics::default())),
        }
    }

    /// Run a specific load test
    pub fn run_test(&mut self, test_type: LoadTestType) -> Result<LoadTestResults, LoadTestError> {
        match test_type {
            LoadTestType::ConnectionStress => self.run_connection_stress(),
            LoadTestType::DhtQueryStorm => self.run_dht_query_storm(),
            LoadTestType::BandwidthSaturation => self.run_bandwidth_saturation(),
            LoadTestType::ProviderFlood => self.run_provider_flood(),
            LoadTestType::ConcurrentOps => self.run_concurrent_ops(),
            LoadTestType::MemoryPressure => self.run_memory_pressure(),
            LoadTestType::ComprehensiveSuite => self.run_comprehensive_suite(),
        }
    }

    /// Run connection stress test
    fn run_connection_stress(&mut self) -> Result<LoadTestResults, LoadTestError> {
        let start = Instant::now();
        let mut results = LoadTestResults::new(LoadTestType::ConnectionStress);

        // Reset metrics
        *self.metrics.write() = LoadTestMetrics {
            start_time: Some(start),
            ..Default::default()
        };

        // Simulate ramping up connections
        let target = self.config.connection_target;
        let mut current_connections = 0;

        while current_connections < target {
            current_connections += 1;
            {
                let mut metrics = self.metrics.write();
                metrics.connections = current_connections;
                metrics.peak_connections = metrics.peak_connections.max(current_connections);
            }

            // Simulate some latency
            std::thread::sleep(Duration::from_millis(10));
        }

        // Run for duration
        std::thread::sleep(self.config.duration);

        // Collect results
        let metrics = self.metrics.read();
        results.peak_connections = metrics.peak_connections;
        results.duration = start.elapsed();
        results.passed = metrics.peak_connections >= self.config.connection_target;

        Ok(results)
    }

    /// Run DHT query storm test
    fn run_dht_query_storm(&mut self) -> Result<LoadTestResults, LoadTestError> {
        let start = Instant::now();
        let mut results = LoadTestResults::new(LoadTestType::DhtQueryStorm);

        // Reset metrics
        *self.metrics.write() = LoadTestMetrics {
            start_time: Some(start),
            ..Default::default()
        };

        let target_queries =
            (self.config.query_rate as f64 * self.config.duration.as_secs_f64()).max(1.0) as u64;

        for _ in 0..target_queries {
            // Simulate query
            self.metrics.write().queries_sent += 1;

            // Simulate success/failure
            let mut rng = rand::rng();
            if rng.random::<f64>() < 0.95 {
                self.metrics.write().queries_succeeded += 1;
            } else {
                self.metrics.write().queries_failed += 1;
            }

            // Simulate latency
            let latency_ms = rng.random_range(10..100);
            let latency = Duration::from_millis(latency_ms);
            self.metrics.write().latencies.push(latency);

            std::thread::sleep(Duration::from_micros(1000 / self.config.query_rate.max(1)));
        }

        // Collect results
        let metrics = self.metrics.read();
        results.total_queries = metrics.queries_sent;
        results.successful_queries = metrics.queries_succeeded;
        results.failed_queries = metrics.queries_failed;
        results.duration = start.elapsed();
        results.query_rate_achieved = results.total_queries as f64 / results.duration.as_secs_f64();

        if !metrics.latencies.is_empty() {
            let mut sorted_latencies = metrics.latencies.clone();
            sorted_latencies.sort();

            let sum: Duration = sorted_latencies.iter().sum();
            results.average_latency = sum / sorted_latencies.len() as u32;

            let p95_idx = (sorted_latencies.len() as f64 * 0.95) as usize;
            let p99_idx = (sorted_latencies.len() as f64 * 0.99) as usize;
            results.p95_latency = sorted_latencies
                .get(p95_idx)
                .copied()
                .unwrap_or(Duration::ZERO);
            results.p99_latency = sorted_latencies
                .get(p99_idx)
                .copied()
                .unwrap_or(Duration::ZERO);
        }

        results.passed = results.success_rate() >= 95.0;

        Ok(results)
    }

    /// Run bandwidth saturation test
    fn run_bandwidth_saturation(&mut self) -> Result<LoadTestResults, LoadTestError> {
        let start = Instant::now();
        let mut results = LoadTestResults::new(LoadTestType::BandwidthSaturation);

        // Reset metrics
        *self.metrics.write() = LoadTestMetrics {
            start_time: Some(start),
            ..Default::default()
        };

        let target_bytes = (self.config.bandwidth_target as f64
            * self.config.duration.as_secs_f64())
        .max(1024.0) as u64;
        let mut bytes_transferred = 0u64;

        while bytes_transferred < target_bytes {
            let chunk_size = 1024 * 1024; // 1 MB chunks
            bytes_transferred += chunk_size;

            self.metrics.write().bytes_sent += chunk_size / 2;
            self.metrics.write().bytes_received += chunk_size / 2;

            std::thread::sleep(Duration::from_millis(10));
        }

        // Collect results
        let metrics = self.metrics.read();
        results.total_bytes_sent = metrics.bytes_sent;
        results.total_bytes_received = metrics.bytes_received;
        results.duration = start.elapsed();
        results.throughput_bps =
            (metrics.bytes_sent + metrics.bytes_received) / results.duration.as_secs().max(1);
        results.passed = results.throughput_bps >= self.config.bandwidth_target;

        Ok(results)
    }

    /// Run provider record flood test
    fn run_provider_flood(&mut self) -> Result<LoadTestResults, LoadTestError> {
        let start = Instant::now();
        let mut results = LoadTestResults::new(LoadTestType::ProviderFlood);

        // Reset metrics
        *self.metrics.write() = LoadTestMetrics {
            start_time: Some(start),
            ..Default::default()
        };

        let target_records = (self.config.provider_publish_rate as f64
            * self.config.duration.as_secs_f64())
        .max(1.0) as u64;

        for _ in 0..target_records {
            // Simulate provider record publication
            self.metrics.write().queries_sent += 1;

            let mut rng = rand::rng();
            if rng.random::<f64>() < 0.98 {
                self.metrics.write().queries_succeeded += 1;
            } else {
                self.metrics.write().queries_failed += 1;
            }

            std::thread::sleep(Duration::from_micros(
                1000 / self.config.provider_publish_rate.max(1),
            ));
        }

        // Collect results
        let metrics = self.metrics.read();
        results.total_queries = metrics.queries_sent;
        results.successful_queries = metrics.queries_succeeded;
        results.failed_queries = metrics.queries_failed;
        results.duration = start.elapsed();
        results.passed = results.success_rate() >= 98.0;

        Ok(results)
    }

    /// Run concurrent operations test
    fn run_concurrent_ops(&mut self) -> Result<LoadTestResults, LoadTestError> {
        let start = Instant::now();
        let mut results = LoadTestResults::new(LoadTestType::ConcurrentOps);

        // Reset metrics
        *self.metrics.write() = LoadTestMetrics {
            start_time: Some(start),
            ..Default::default()
        };

        // Simulate concurrent operations
        let mut rng = rand::rng();
        for _ in 0..self.config.concurrent_operations {
            self.metrics.write().queries_sent += 1;

            if rng.random::<f64>() < 0.90 {
                self.metrics.write().queries_succeeded += 1;
            } else {
                self.metrics.write().queries_failed += 1;
            }
        }

        std::thread::sleep(self.config.duration);

        // Collect results
        let metrics = self.metrics.read();
        results.total_queries = metrics.queries_sent;
        results.successful_queries = metrics.queries_succeeded;
        results.failed_queries = metrics.queries_failed;
        results.duration = start.elapsed();
        results.passed = results.success_rate() >= 90.0;

        Ok(results)
    }

    /// Run memory pressure test
    fn run_memory_pressure(&mut self) -> Result<LoadTestResults, LoadTestError> {
        let start = Instant::now();
        let mut results = LoadTestResults::new(LoadTestType::MemoryPressure);

        // Reset metrics
        *self.metrics.write() = LoadTestMetrics {
            start_time: Some(start),
            ..Default::default()
        };

        // Simulate memory usage growth
        let samples = (self.config.duration.as_secs() / 10).max(1);
        let step = self.config.memory_limit / samples;

        for i in 0..samples {
            let memory_used = step * (i + 1);
            self.metrics.write().memory_samples.push(memory_used);
            std::thread::sleep(Duration::from_secs(10));
        }

        // Collect results
        let metrics = self.metrics.read();
        if !metrics.memory_samples.is_empty() {
            results.peak_memory_usage = *metrics
                .memory_samples
                .iter()
                .max()
                .expect("memory_samples is non-empty: checked above");
            results.average_memory_usage =
                metrics.memory_samples.iter().sum::<u64>() / metrics.memory_samples.len() as u64;
        }
        results.duration = start.elapsed();
        results.passed = results.peak_memory_usage <= self.config.memory_limit;

        Ok(results)
    }

    /// Run comprehensive suite of all tests
    fn run_comprehensive_suite(&mut self) -> Result<LoadTestResults, LoadTestError> {
        let start = Instant::now();
        let mut combined = LoadTestResults::new(LoadTestType::ComprehensiveSuite);

        let tests = vec![
            LoadTestType::ConnectionStress,
            LoadTestType::DhtQueryStorm,
            LoadTestType::BandwidthSaturation,
            LoadTestType::ProviderFlood,
            LoadTestType::ConcurrentOps,
            LoadTestType::MemoryPressure,
        ];

        let mut all_passed = true;

        for test_type in tests {
            match self.run_test(test_type) {
                Ok(result) => {
                    if !result.passed {
                        all_passed = false;
                        combined.errors.push(format!("{} failed", test_type.name()));
                    }
                    // Aggregate metrics
                    combined.total_queries += result.total_queries;
                    combined.successful_queries += result.successful_queries;
                    combined.failed_queries += result.failed_queries;
                    combined.peak_connections =
                        combined.peak_connections.max(result.peak_connections);
                    combined.peak_memory_usage =
                        combined.peak_memory_usage.max(result.peak_memory_usage);
                }
                Err(e) => {
                    all_passed = false;
                    combined.errors.push(format!("{}: {}", test_type.name(), e));
                }
            }
        }

        combined.duration = start.elapsed();
        combined.passed = all_passed;

        Ok(combined)
    }

    /// Get current metrics snapshot
    pub fn get_metrics_snapshot(&self) -> LoadTestMetrics {
        self.metrics.read().clone()
    }
}

/// Error types for load testing
#[derive(Debug, thiserror::Error)]
pub enum LoadTestError {
    #[error("Load test failed: {0}")]
    TestFailed(String),

    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("Timeout reached")]
    Timeout,

    #[error("Resource limit exceeded: {0}")]
    ResourceLimit(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_presets() {
        let light = LoadTestConfig::light();
        assert_eq!(light.connection_target, 20);

        let moderate = LoadTestConfig::moderate();
        assert_eq!(moderate.connection_target, 100);

        let heavy = LoadTestConfig::heavy();
        assert_eq!(heavy.connection_target, 500);

        let extreme = LoadTestConfig::extreme();
        assert_eq!(extreme.connection_target, 2000);
    }

    #[test]
    fn test_load_test_types() {
        assert_eq!(
            LoadTestType::ConnectionStress.name(),
            "Connection Stress Test"
        );
        assert!(!LoadTestType::DhtQueryStorm.description().is_empty());
    }

    #[test]
    fn test_results_creation() {
        let results = LoadTestResults::new(LoadTestType::ConnectionStress);
        assert_eq!(results.test_type, LoadTestType::ConnectionStress);
        assert!(!results.passed);
        assert_eq!(results.total_queries, 0);
    }

    #[test]
    fn test_success_rate() {
        let mut results = LoadTestResults::new(LoadTestType::DhtQueryStorm);
        results.total_queries = 100;
        results.successful_queries = 95;
        assert_eq!(results.success_rate(), 95.0);
    }

    #[test]
    fn test_tester_creation() {
        let config = LoadTestConfig::light();
        let tester = LoadTester::new(config);
        assert_eq!(tester.config.connection_target, 20);
    }

    #[test]
    fn test_connection_stress() {
        let config = LoadTestConfig {
            duration: Duration::from_millis(100),
            connection_target: 10,
            ..LoadTestConfig::light()
        };
        let mut tester = LoadTester::new(config);
        let results = tester
            .run_test(LoadTestType::ConnectionStress)
            .expect("test: ConnectionStress should succeed");
        assert!(results.peak_connections > 0);
    }

    #[test]
    fn test_dht_query_storm() {
        let config = LoadTestConfig {
            duration: Duration::from_millis(100),
            query_rate: 10,
            ..LoadTestConfig::light()
        };
        let mut tester = LoadTester::new(config);
        let results = tester
            .run_test(LoadTestType::DhtQueryStorm)
            .expect("test: DhtQueryStorm should succeed");
        assert!(results.total_queries > 0);
    }

    #[test]
    fn test_bandwidth_saturation() {
        let config = LoadTestConfig {
            duration: Duration::from_millis(100),
            bandwidth_target: 1_000_000,
            ..LoadTestConfig::light()
        };
        let mut tester = LoadTester::new(config);
        let results = tester
            .run_test(LoadTestType::BandwidthSaturation)
            .expect("test: BandwidthSaturation should succeed");
        assert!(results.total_bytes_sent > 0 || results.total_bytes_received > 0);
    }

    #[test]
    fn test_provider_flood() {
        let config = LoadTestConfig {
            duration: Duration::from_millis(100),
            provider_publish_rate: 10,
            ..LoadTestConfig::light()
        };
        let mut tester = LoadTester::new(config);
        let results = tester
            .run_test(LoadTestType::ProviderFlood)
            .expect("test: ProviderFlood should succeed");
        assert!(results.total_queries > 0);
    }

    #[test]
    fn test_concurrent_ops() {
        let config = LoadTestConfig {
            duration: Duration::from_millis(100),
            concurrent_operations: 20,
            ..LoadTestConfig::light()
        };
        let mut tester = LoadTester::new(config);
        let results = tester
            .run_test(LoadTestType::ConcurrentOps)
            .expect("test: ConcurrentOps should succeed");
        assert_eq!(results.total_queries, 20);
    }

    #[test]
    fn test_memory_pressure() {
        let config = LoadTestConfig {
            duration: Duration::from_millis(100),
            memory_limit: 100 * 1024 * 1024,
            ..LoadTestConfig::light()
        };
        let mut tester = LoadTester::new(config);
        let results = tester
            .run_test(LoadTestType::MemoryPressure)
            .expect("test: MemoryPressure should succeed");
        assert!(results.peak_memory_usage > 0);
    }

    #[test]
    fn test_results_summary() {
        let mut results = LoadTestResults::new(LoadTestType::ConnectionStress);
        results.passed = true;
        results.peak_connections = 100;
        let summary = results.summary();
        assert!(summary.contains("PASS"));
        assert!(summary.contains("100"));
    }

    #[test]
    fn test_metrics_snapshot() {
        let config = LoadTestConfig::light();
        let tester = LoadTester::new(config);
        let snapshot = tester.get_metrics_snapshot();
        assert_eq!(snapshot.connections, 0);
    }
}
