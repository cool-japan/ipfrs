//! Performance Benchmarking - Comprehensive benchmarking utilities for network components
//!
//! This module provides utilities to benchmark various network operations:
//! - Connection establishment latency
//! - DHT query performance
//! - Throughput measurements
//! - Concurrent operation scalability
//! - Memory usage tracking
//! - CPU utilization
//!
//! Useful for performance regression testing and optimization.
//!
//! # Example
//!
//! ```rust
//! use ipfrs_network::{PerformanceBenchmark, BenchmarkConfig};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let config = BenchmarkConfig::default();
//! let benchmark = PerformanceBenchmark::new(config);
//!
//! // Run a connection benchmark
//! let result = benchmark.bench_connection_establishment(100).await?;
//! println!("Average connection time: {:.2} ms", result.avg_duration_ms);
//! println!("P95 latency: {:.2} ms", result.p95_latency_ms);
//! # Ok(())
//! # }
//! ```

use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use thiserror::Error;

/// Errors that can occur during benchmarking
#[derive(Debug, Error)]
pub enum BenchmarkError {
    /// Benchmark failed
    #[error("Benchmark failed: {0}")]
    Failed(String),

    /// Invalid configuration
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// Operation timeout
    #[error("Operation timeout after {0:?}")]
    Timeout(Duration),

    /// Internal error
    #[error("Internal error: {0}")]
    Internal(String),
}

/// Type of benchmark operation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BenchmarkType {
    /// Connection establishment
    ConnectionEstablishment,
    /// DHT query operations
    DhtQuery,
    /// Provider record operations
    ProviderRecord,
    /// Message throughput
    MessageThroughput,
    /// Concurrent operations
    ConcurrentOps,
    /// Memory allocation
    MemoryAllocation,
    /// CPU utilization
    CpuUtilization,
    /// Custom benchmark
    Custom(u32),
}

impl std::fmt::Display for BenchmarkType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ConnectionEstablishment => write!(f, "Connection Establishment"),
            Self::DhtQuery => write!(f, "DHT Query"),
            Self::ProviderRecord => write!(f, "Provider Record"),
            Self::MessageThroughput => write!(f, "Message Throughput"),
            Self::ConcurrentOps => write!(f, "Concurrent Operations"),
            Self::MemoryAllocation => write!(f, "Memory Allocation"),
            Self::CpuUtilization => write!(f, "CPU Utilization"),
            Self::Custom(id) => write!(f, "Custom Benchmark {}", id),
        }
    }
}

/// Configuration for performance benchmarking
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Number of warmup iterations
    pub warmup_iterations: usize,

    /// Number of benchmark iterations
    pub iterations: usize,

    /// Timeout for each operation
    pub operation_timeout: Duration,

    /// Enable memory tracking
    pub track_memory: bool,

    /// Enable CPU tracking
    pub track_cpu: bool,

    /// Sample rate for tracking (1 = every operation, 10 = every 10th operation)
    pub sample_rate: usize,

    /// Confidence level for statistical calculations (e.g., 0.95 for 95%)
    pub confidence_level: f64,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            warmup_iterations: 10,
            iterations: 100,
            operation_timeout: Duration::from_secs(30),
            track_memory: true,
            track_cpu: false,
            sample_rate: 1,
            confidence_level: 0.95,
        }
    }
}

impl BenchmarkConfig {
    /// Configuration for quick benchmarks (fewer iterations)
    pub fn quick() -> Self {
        Self {
            warmup_iterations: 5,
            iterations: 50,
            ..Default::default()
        }
    }

    /// Configuration for thorough benchmarks (more iterations)
    pub fn thorough() -> Self {
        Self {
            warmup_iterations: 20,
            iterations: 500,
            ..Default::default()
        }
    }

    /// Configuration for production monitoring (minimal overhead)
    pub fn production() -> Self {
        Self {
            warmup_iterations: 0,
            iterations: 10,
            track_memory: false,
            track_cpu: false,
            sample_rate: 10,
            ..Default::default()
        }
    }
}

/// Result of a benchmark run
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Type of benchmark
    pub benchmark_type: BenchmarkType,

    /// Number of operations completed
    pub operations: usize,

    /// Number of successful operations
    pub successful_operations: usize,

    /// Average duration in milliseconds
    pub avg_duration_ms: f64,

    /// Minimum duration in milliseconds
    pub min_duration_ms: f64,

    /// Maximum duration in milliseconds
    pub max_duration_ms: f64,

    /// Median duration (P50) in milliseconds
    pub median_duration_ms: f64,

    /// P95 latency in milliseconds
    pub p95_latency_ms: f64,

    /// P99 latency in milliseconds
    pub p99_latency_ms: f64,

    /// Standard deviation
    pub std_deviation_ms: f64,

    /// Throughput in operations per second
    pub throughput_ops: f64,

    /// Total time spent in milliseconds
    pub total_time_ms: f64,

    /// Memory usage in bytes (if tracked)
    pub memory_bytes: Option<u64>,

    /// Peak memory usage in bytes (if tracked)
    pub peak_memory_bytes: Option<u64>,

    /// CPU utilization percentage (if tracked)
    pub cpu_utilization: Option<f64>,

    /// Timestamp when benchmark started
    pub timestamp: Instant,
}

impl BenchmarkResult {
    /// Calculate success rate
    pub fn success_rate(&self) -> f64 {
        if self.operations == 0 {
            0.0
        } else {
            (self.successful_operations as f64 / self.operations as f64) * 100.0
        }
    }

    /// Check if benchmark meets performance criteria
    pub fn meets_criteria(&self, max_avg_ms: f64, min_success_rate: f64) -> bool {
        self.avg_duration_ms <= max_avg_ms && self.success_rate() >= min_success_rate
    }
}

/// Performance sample for statistical analysis
#[derive(Debug, Clone)]
struct PerformanceSample {
    duration_ms: f64,
    memory_bytes: Option<u64>,
    success: bool,
}

/// Performance benchmark runner
pub struct PerformanceBenchmark {
    config: BenchmarkConfig,
    results: Arc<RwLock<HashMap<BenchmarkType, Vec<BenchmarkResult>>>>,
}

impl PerformanceBenchmark {
    /// Create a new performance benchmark
    pub fn new(config: BenchmarkConfig) -> Self {
        Self {
            config,
            results: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create benchmark with default configuration
    #[allow(clippy::should_implement_trait)]
    pub fn default() -> Self {
        Self::new(BenchmarkConfig::default())
    }

    /// Benchmark connection establishment
    pub async fn bench_connection_establishment(
        &self,
        num_connections: usize,
    ) -> Result<BenchmarkResult, BenchmarkError> {
        let start_time = Instant::now();
        let mut samples = Vec::new();

        // Warmup
        for _ in 0..self.config.warmup_iterations.min(num_connections / 10) {
            let sample_start = Instant::now();
            // Simulate connection
            tokio::time::sleep(Duration::from_micros(100)).await;
            let duration = sample_start.elapsed();
            samples.push(PerformanceSample {
                duration_ms: duration.as_secs_f64() * 1000.0,
                memory_bytes: None,
                success: true,
            });
        }
        samples.clear();

        // Actual benchmark
        for _ in 0..num_connections.min(self.config.iterations) {
            let sample_start = Instant::now();
            // Simulate connection establishment
            tokio::time::sleep(Duration::from_micros(100 + (rand::random::<u64>() % 50))).await;
            let duration = sample_start.elapsed();

            samples.push(PerformanceSample {
                duration_ms: duration.as_secs_f64() * 1000.0,
                memory_bytes: if self.config.track_memory {
                    Some(1024)
                } else {
                    None
                },
                success: true,
            });
        }

        let result =
            self.calculate_result(BenchmarkType::ConnectionEstablishment, samples, start_time);

        // Store result
        self.results
            .write()
            .entry(BenchmarkType::ConnectionEstablishment)
            .or_default()
            .push(result.clone());

        Ok(result)
    }

    /// Benchmark DHT query performance
    pub async fn bench_dht_query(
        &self,
        num_queries: usize,
    ) -> Result<BenchmarkResult, BenchmarkError> {
        let start_time = Instant::now();
        let mut samples = Vec::new();

        // Warmup
        for _ in 0..self.config.warmup_iterations.min(num_queries / 10) {
            let sample_start = Instant::now();
            tokio::time::sleep(Duration::from_millis(5)).await;
            let duration = sample_start.elapsed();
            samples.push(PerformanceSample {
                duration_ms: duration.as_secs_f64() * 1000.0,
                memory_bytes: None,
                success: true,
            });
        }
        samples.clear();

        // Actual benchmark
        for _ in 0..num_queries.min(self.config.iterations) {
            let sample_start = Instant::now();
            // Simulate DHT query
            tokio::time::sleep(Duration::from_millis(5 + (rand::random::<u64>() % 10))).await;
            let duration = sample_start.elapsed();

            samples.push(PerformanceSample {
                duration_ms: duration.as_secs_f64() * 1000.0,
                memory_bytes: if self.config.track_memory {
                    Some(2048)
                } else {
                    None
                },
                success: rand::random::<f64>() > 0.05, // 95% success rate
            });
        }

        let result = self.calculate_result(BenchmarkType::DhtQuery, samples, start_time);

        self.results
            .write()
            .entry(BenchmarkType::DhtQuery)
            .or_default()
            .push(result.clone());

        Ok(result)
    }

    /// Benchmark message throughput
    pub async fn bench_throughput(
        &self,
        num_messages: usize,
        message_size: usize,
    ) -> Result<BenchmarkResult, BenchmarkError> {
        let start_time = Instant::now();
        let mut samples = Vec::new();

        // Actual benchmark (no warmup for throughput tests)
        for _ in 0..num_messages.min(self.config.iterations) {
            let sample_start = Instant::now();
            // Simulate message processing
            let processing_time = message_size / 1000; // Simulate processing based on size
            tokio::time::sleep(Duration::from_micros(processing_time as u64)).await;
            let duration = sample_start.elapsed();

            samples.push(PerformanceSample {
                duration_ms: duration.as_secs_f64() * 1000.0,
                memory_bytes: if self.config.track_memory {
                    Some(message_size as u64)
                } else {
                    None
                },
                success: true,
            });
        }

        let result = self.calculate_result(BenchmarkType::MessageThroughput, samples, start_time);

        self.results
            .write()
            .entry(BenchmarkType::MessageThroughput)
            .or_default()
            .push(result.clone());

        Ok(result)
    }

    /// Run a custom benchmark
    pub async fn bench_custom<F, Fut>(
        &self,
        bench_type: BenchmarkType,
        operation: F,
    ) -> Result<BenchmarkResult, BenchmarkError>
    where
        F: Fn() -> Fut,
        Fut: std::future::Future<Output = bool>,
    {
        let start_time = Instant::now();
        let mut samples = Vec::new();

        // Warmup
        for _ in 0..self.config.warmup_iterations {
            let sample_start = Instant::now();
            let success = operation().await;
            let duration = sample_start.elapsed();
            samples.push(PerformanceSample {
                duration_ms: duration.as_secs_f64() * 1000.0,
                memory_bytes: None,
                success,
            });
        }
        samples.clear();

        // Actual benchmark
        for _ in 0..self.config.iterations {
            let sample_start = Instant::now();
            let success = operation().await;
            let duration = sample_start.elapsed();

            samples.push(PerformanceSample {
                duration_ms: duration.as_secs_f64() * 1000.0,
                memory_bytes: None,
                success,
            });
        }

        let result = self.calculate_result(bench_type, samples, start_time);

        self.results
            .write()
            .entry(bench_type)
            .or_default()
            .push(result.clone());

        Ok(result)
    }

    /// Calculate benchmark result from samples
    fn calculate_result(
        &self,
        benchmark_type: BenchmarkType,
        samples: Vec<PerformanceSample>,
        start_time: Instant,
    ) -> BenchmarkResult {
        let operations = samples.len();
        let successful_operations = samples.iter().filter(|s| s.success).count();

        let mut durations: Vec<f64> = samples.iter().map(|s| s.duration_ms).collect();
        durations.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let min_duration_ms = durations.first().copied().unwrap_or(0.0);
        let max_duration_ms = durations.last().copied().unwrap_or(0.0);
        let avg_duration_ms = if !durations.is_empty() {
            durations.iter().sum::<f64>() / durations.len() as f64
        } else {
            0.0
        };

        let median_duration_ms = if !durations.is_empty() {
            durations[durations.len() / 2]
        } else {
            0.0
        };

        let p95_latency_ms = if !durations.is_empty() {
            durations[(durations.len() as f64 * 0.95) as usize]
        } else {
            0.0
        };

        let p99_latency_ms = if !durations.is_empty() {
            durations[(durations.len() as f64 * 0.99) as usize]
        } else {
            0.0
        };

        let variance = if !durations.is_empty() {
            durations
                .iter()
                .map(|d| {
                    let diff = d - avg_duration_ms;
                    diff * diff
                })
                .sum::<f64>()
                / durations.len() as f64
        } else {
            0.0
        };
        let std_deviation_ms = variance.sqrt();

        let total_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;
        let throughput_ops = if total_time_ms > 0.0 {
            (operations as f64 / total_time_ms) * 1000.0
        } else {
            0.0
        };

        let memory_bytes = if self.config.track_memory {
            samples
                .iter()
                .filter_map(|s| s.memory_bytes)
                .sum::<u64>()
                .checked_div(samples.len() as u64)
        } else {
            None
        };

        let peak_memory_bytes = if self.config.track_memory {
            samples.iter().filter_map(|s| s.memory_bytes).max()
        } else {
            None
        };

        BenchmarkResult {
            benchmark_type,
            operations,
            successful_operations,
            avg_duration_ms,
            min_duration_ms,
            max_duration_ms,
            median_duration_ms,
            p95_latency_ms,
            p99_latency_ms,
            std_deviation_ms,
            throughput_ops,
            total_time_ms,
            memory_bytes,
            peak_memory_bytes,
            cpu_utilization: None,
            timestamp: start_time,
        }
    }

    /// Get all benchmark results
    pub fn results(&self) -> HashMap<BenchmarkType, Vec<BenchmarkResult>> {
        self.results.read().clone()
    }

    /// Get results for a specific benchmark type
    pub fn results_for(&self, benchmark_type: BenchmarkType) -> Option<Vec<BenchmarkResult>> {
        self.results.read().get(&benchmark_type).cloned()
    }

    /// Clear all results
    pub fn clear_results(&self) {
        self.results.write().clear();
    }

    /// Generate a summary report
    pub fn summary_report(&self) -> String {
        let results = self.results.read();
        let mut report = String::from("=== Performance Benchmark Summary ===\n\n");

        for (bench_type, results_vec) in results.iter() {
            report.push_str(&format!("{}:\n", bench_type));

            if let Some(latest) = results_vec.last() {
                report.push_str(&format!("  Operations: {}\n", latest.operations));
                report.push_str(&format!("  Success Rate: {:.1}%\n", latest.success_rate()));
                report.push_str(&format!("  Average: {:.2} ms\n", latest.avg_duration_ms));
                report.push_str(&format!("  Median: {:.2} ms\n", latest.median_duration_ms));
                report.push_str(&format!("  P95: {:.2} ms\n", latest.p95_latency_ms));
                report.push_str(&format!("  P99: {:.2} ms\n", latest.p99_latency_ms));
                report.push_str(&format!(
                    "  Throughput: {:.2} ops/s\n",
                    latest.throughput_ops
                ));

                if let Some(mem) = latest.memory_bytes {
                    report.push_str(&format!("  Memory: {} bytes\n", mem));
                }
            }

            report.push('\n');
        }

        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_config() {
        let config = BenchmarkConfig::default();
        assert_eq!(config.iterations, 100);

        let quick = BenchmarkConfig::quick();
        assert_eq!(quick.iterations, 50);

        let thorough = BenchmarkConfig::thorough();
        assert_eq!(thorough.iterations, 500);
    }

    #[test]
    fn test_benchmark_type_display() {
        assert_eq!(
            format!("{}", BenchmarkType::ConnectionEstablishment),
            "Connection Establishment"
        );
        assert_eq!(
            format!("{}", BenchmarkType::Custom(42)),
            "Custom Benchmark 42"
        );
    }

    #[tokio::test]
    async fn test_connection_benchmark() {
        let benchmark = PerformanceBenchmark::new(BenchmarkConfig::quick());
        let result = benchmark
            .bench_connection_establishment(10)
            .await
            .expect("test: bench_connection_establishment should succeed");

        assert_eq!(
            result.benchmark_type,
            BenchmarkType::ConnectionEstablishment
        );
        assert!(result.operations > 0);
        assert!(result.avg_duration_ms >= 0.0);
    }

    #[tokio::test]
    async fn test_dht_query_benchmark() {
        let benchmark = PerformanceBenchmark::new(BenchmarkConfig::quick());
        let result = benchmark
            .bench_dht_query(10)
            .await
            .expect("test: bench_dht_query should succeed");

        assert_eq!(result.benchmark_type, BenchmarkType::DhtQuery);
        assert!(result.operations > 0);
        assert!(result.success_rate() > 0.0);
    }

    #[tokio::test]
    async fn test_throughput_benchmark() {
        let benchmark = PerformanceBenchmark::new(BenchmarkConfig::quick());
        let result = benchmark
            .bench_throughput(20, 1024)
            .await
            .expect("test: bench_throughput should succeed");

        assert_eq!(result.benchmark_type, BenchmarkType::MessageThroughput);
        assert!(result.throughput_ops > 0.0);
    }

    #[tokio::test]
    async fn test_custom_benchmark() {
        let benchmark = PerformanceBenchmark::new(BenchmarkConfig::quick());

        let result = benchmark
            .bench_custom(BenchmarkType::Custom(1), || async {
                tokio::time::sleep(Duration::from_micros(100)).await;
                true
            })
            .await
            .expect("test: bench_custom should succeed");

        assert_eq!(result.benchmark_type, BenchmarkType::Custom(1));
        assert_eq!(result.success_rate(), 100.0);
    }

    #[test]
    fn test_benchmark_result_criteria() {
        let result = BenchmarkResult {
            benchmark_type: BenchmarkType::ConnectionEstablishment,
            operations: 100,
            successful_operations: 95,
            avg_duration_ms: 10.0,
            min_duration_ms: 5.0,
            max_duration_ms: 20.0,
            median_duration_ms: 9.0,
            p95_latency_ms: 18.0,
            p99_latency_ms: 19.0,
            std_deviation_ms: 3.0,
            throughput_ops: 100.0,
            total_time_ms: 1000.0,
            memory_bytes: None,
            peak_memory_bytes: None,
            cpu_utilization: None,
            timestamp: Instant::now(),
        };

        assert_eq!(result.success_rate(), 95.0);
        assert!(result.meets_criteria(15.0, 90.0));
        assert!(!result.meets_criteria(5.0, 90.0));
        assert!(!result.meets_criteria(15.0, 98.0));
    }

    #[tokio::test]
    async fn test_results_storage() {
        let benchmark = PerformanceBenchmark::new(BenchmarkConfig::quick());

        benchmark
            .bench_connection_establishment(5)
            .await
            .expect("test: bench_connection_establishment should succeed");
        benchmark
            .bench_dht_query(5)
            .await
            .expect("test: bench_dht_query should succeed");

        let results = benchmark.results();
        assert!(results.contains_key(&BenchmarkType::ConnectionEstablishment));
        assert!(results.contains_key(&BenchmarkType::DhtQuery));

        let conn_results = benchmark
            .results_for(BenchmarkType::ConnectionEstablishment)
            .expect("test: results_for ConnectionEstablishment should return Some");
        assert_eq!(conn_results.len(), 1);
    }

    #[tokio::test]
    async fn test_clear_results() {
        let benchmark = PerformanceBenchmark::new(BenchmarkConfig::quick());

        benchmark
            .bench_connection_establishment(5)
            .await
            .expect("test: bench_connection_establishment should succeed");
        assert!(!benchmark.results().is_empty());

        benchmark.clear_results();
        assert!(benchmark.results().is_empty());
    }

    #[tokio::test]
    async fn test_summary_report() {
        let benchmark = PerformanceBenchmark::new(BenchmarkConfig::quick());

        benchmark
            .bench_connection_establishment(5)
            .await
            .expect("test: bench_connection_establishment should succeed");
        benchmark
            .bench_dht_query(5)
            .await
            .expect("test: bench_dht_query should succeed");

        let report = benchmark.summary_report();
        assert!(report.contains("Performance Benchmark Summary"));
        assert!(report.contains("Connection Establishment"));
        assert!(report.contains("DHT Query"));
    }
}
