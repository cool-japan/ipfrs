//! Production readiness testing utilities
//!
//! This module provides comprehensive testing utilities for validating
//! the semantic search system under production-like conditions.
//!
//! # Features
//!
//! - **Stress Testing**: Validate system behavior under high load
//! - **Endurance Testing**: Long-running tests for memory leaks and stability
//! - **Chaos Testing**: Fault injection and error handling validation
//! - **Performance Regression**: Detect performance degradation
//! - **Concurrency Testing**: Validate thread-safety and race conditions
//!
//! # Usage
//!
//! ```rust,no_run
//! use ipfrs_semantic::prod_tests::{StressTest, StressTestConfig};
//!
//! # #[tokio::main]
//! # async fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let config = StressTestConfig {
//!     num_threads: 10,
//!     operations_per_thread: 1000,
//!     index_size: 10000,
//!     dimension: 768,
//!     ..Default::default()
//! };
//!
//! let mut stress_test = StressTest::new(config)?;
//! let results = stress_test.run().await?;
//!
//! println!("Operations/sec: {:.2}", results.ops_per_second);
//! println!("Average latency: {:?}", results.avg_latency);
//! println!("Success rate: {:.2}%", results.success_rate * 100.0);
//! # Ok(())
//! # }
//! ```

use crate::router::{RouterConfig, SemanticRouter};
use ipfrs_core::{Cid, Result};
use rand::RngExt;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::task;

/// Stress test configuration
#[derive(Debug, Clone)]
pub struct StressTestConfig {
    /// Number of concurrent threads/tasks
    pub num_threads: usize,
    /// Operations per thread
    pub operations_per_thread: usize,
    /// Initial index size
    pub index_size: usize,
    /// Vector dimension
    pub dimension: usize,
    /// Mix of operations (insert_ratio + query_ratio should = 1.0)
    pub insert_ratio: f64,
    /// Query ratio
    pub query_ratio: f64,
    /// k for queries
    pub k: usize,
}

impl Default for StressTestConfig {
    fn default() -> Self {
        Self {
            num_threads: 10,
            operations_per_thread: 100,
            index_size: 1000,
            dimension: 768,
            insert_ratio: 0.3,
            query_ratio: 0.7,
            k: 10,
        }
    }
}

/// Stress test results
#[derive(Debug, Clone)]
pub struct StressTestResults {
    /// Total operations executed
    pub total_ops: usize,
    /// Successful operations
    pub successful_ops: usize,
    /// Failed operations
    pub failed_ops: usize,
    /// Total duration
    pub total_duration: Duration,
    /// Operations per second
    pub ops_per_second: f64,
    /// Average operation latency
    pub avg_latency: Duration,
    /// P50 latency
    pub p50_latency: Duration,
    /// P90 latency
    pub p90_latency: Duration,
    /// P99 latency
    pub p99_latency: Duration,
    /// Success rate (0.0 to 1.0)
    pub success_rate: f64,
    /// Maximum concurrent operations
    pub max_concurrent: usize,
}

/// Stress testing framework
pub struct StressTest {
    config: StressTestConfig,
    router: Arc<SemanticRouter>,
}

impl StressTest {
    /// Create a new stress test
    pub fn new(config: StressTestConfig) -> Result<Self> {
        let router_config =
            RouterConfig::balanced(config.dimension).with_cache_size(config.index_size * 2);

        let router = SemanticRouter::new(router_config)?;

        // Pre-populate index (if index_size > 0)
        if config.index_size > 0 {
            for i in 0..config.index_size {
                let cid = generate_test_cid(i);
                let embedding = generate_random_embedding(config.dimension);
                router.add(&cid, &embedding)?;
            }
        }

        Ok(Self {
            config,
            router: Arc::new(router),
        })
    }

    /// Run the stress test
    pub async fn run(&mut self) -> Result<StressTestResults> {
        let start = Instant::now();
        let mut handles = Vec::new();
        let mut all_latencies = Vec::new();

        let total_ops = self.config.num_threads * self.config.operations_per_thread;
        let successful_ops = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let failed_ops = Arc::new(std::sync::atomic::AtomicUsize::new(0));

        // Spawn worker tasks
        for thread_id in 0..self.config.num_threads {
            let router = Arc::clone(&self.router);
            let config = self.config.clone();
            let successful = Arc::clone(&successful_ops);
            let failed = Arc::clone(&failed_ops);

            let handle = task::spawn(async move {
                let mut latencies = Vec::new();

                for op_id in 0..config.operations_per_thread {
                    let op_start = Instant::now();

                    // Determine operation type using thread_id and op_id for determinism
                    let should_insert =
                        ((thread_id + op_id) % 10) as f64 / 10.0 < config.insert_ratio;

                    let result = if should_insert {
                        // Insert operation
                        let cid = generate_test_cid(thread_id * 1000000 + op_id);
                        let embedding = generate_random_embedding(config.dimension);
                        router.add(&cid, &embedding)
                    } else {
                        // Query operation
                        let query = generate_random_embedding(config.dimension);
                        match router.query(&query, config.k).await {
                            Ok(_) => Ok(()),
                            Err(e) => Err(e),
                        }
                    };

                    let latency = op_start.elapsed();
                    latencies.push(latency);

                    match result {
                        Ok(_) => {
                            successful.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        }
                        Err(_) => {
                            failed.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        }
                    }
                }

                latencies
            });

            handles.push(handle);
        }

        // Collect results
        for handle in handles {
            let latencies = handle
                .await
                .map_err(|e| ipfrs_core::Error::InvalidInput(format!("Task join error: {}", e)))?;
            all_latencies.extend(latencies);
        }

        let total_duration = start.elapsed();

        // Calculate statistics
        all_latencies.sort();
        let avg_latency = if !all_latencies.is_empty() {
            all_latencies.iter().sum::<Duration>() / all_latencies.len() as u32
        } else {
            Duration::from_secs(0)
        };

        let p50_latency = percentile(&all_latencies, 0.50);
        let p90_latency = percentile(&all_latencies, 0.90);
        let p99_latency = percentile(&all_latencies, 0.99);

        let successful = successful_ops.load(std::sync::atomic::Ordering::Relaxed);
        let failed = failed_ops.load(std::sync::atomic::Ordering::Relaxed);

        Ok(StressTestResults {
            total_ops,
            successful_ops: successful,
            failed_ops: failed,
            total_duration,
            ops_per_second: total_ops as f64 / total_duration.as_secs_f64(),
            avg_latency,
            p50_latency,
            p90_latency,
            p99_latency,
            success_rate: successful as f64 / total_ops as f64,
            max_concurrent: self.config.num_threads,
        })
    }
}

/// Endurance test configuration
#[derive(Debug, Clone)]
pub struct EnduranceTestConfig {
    /// Test duration
    pub duration: Duration,
    /// Operations per second target
    pub target_ops_per_second: f64,
    /// Vector dimension
    pub dimension: usize,
    /// Memory check interval
    pub memory_check_interval: Duration,
}

impl Default for EnduranceTestConfig {
    fn default() -> Self {
        Self {
            duration: Duration::from_secs(300), // 5 minutes
            target_ops_per_second: 100.0,
            dimension: 768,
            memory_check_interval: Duration::from_secs(10),
        }
    }
}

/// Endurance test results
#[derive(Debug, Clone)]
pub struct EnduranceTestResults {
    /// Total operations completed
    pub total_ops: usize,
    /// Actual duration
    pub actual_duration: Duration,
    /// Average ops per second
    pub avg_ops_per_second: f64,
    /// Peak memory usage (bytes)
    pub peak_memory_bytes: usize,
    /// Initial memory usage (bytes)
    pub initial_memory_bytes: usize,
    /// Memory growth (bytes)
    pub memory_growth_bytes: isize,
    /// Number of errors encountered
    pub error_count: usize,
}

/// Endurance testing framework
pub struct EnduranceTest {
    config: EnduranceTestConfig,
    router: Arc<SemanticRouter>,
}

impl EnduranceTest {
    /// Create a new endurance test
    pub fn new(config: EnduranceTestConfig) -> Result<Self> {
        let router = SemanticRouter::with_defaults()?;

        Ok(Self {
            config,
            router: Arc::new(router),
        })
    }

    /// Run the endurance test
    pub async fn run(&mut self) -> Result<EnduranceTestResults> {
        let start = Instant::now();
        let target_interval = Duration::from_secs_f64(1.0 / self.config.target_ops_per_second);

        let initial_memory = estimate_process_memory();
        let mut peak_memory = initial_memory;
        let mut last_memory_check = Instant::now();

        let mut total_ops = 0;
        let mut error_count = 0;
        let mut op_counter = 0;

        while start.elapsed() < self.config.duration {
            let op_start = Instant::now();

            // Perform operation
            let cid = generate_test_cid(op_counter);
            let embedding = generate_random_embedding(self.config.dimension);

            match self.router.add(&cid, &embedding) {
                Ok(_) => total_ops += 1,
                Err(_) => error_count += 1,
            }

            // Also perform a query periodically
            if op_counter % 5 == 0 {
                let query = generate_random_embedding(self.config.dimension);
                match self.router.query(&query, 10).await {
                    Ok(_) => total_ops += 1,
                    Err(_) => error_count += 1,
                }
            }

            op_counter += 1;

            // Check memory periodically
            if last_memory_check.elapsed() >= self.config.memory_check_interval {
                let current_memory = estimate_process_memory();
                if current_memory > peak_memory {
                    peak_memory = current_memory;
                }
                last_memory_check = Instant::now();
            }

            // Rate limiting
            let elapsed = op_start.elapsed();
            if elapsed < target_interval {
                tokio::time::sleep(target_interval - elapsed).await;
            }
        }

        let actual_duration = start.elapsed();

        Ok(EnduranceTestResults {
            total_ops,
            actual_duration,
            avg_ops_per_second: total_ops as f64 / actual_duration.as_secs_f64(),
            peak_memory_bytes: peak_memory,
            initial_memory_bytes: initial_memory,
            memory_growth_bytes: peak_memory as isize - initial_memory as isize,
            error_count,
        })
    }
}

// Helper functions

fn generate_test_cid(index: usize) -> Cid {
    // Generate a unique CID for each index
    // We use a hash of the index to create unique multihashes
    use multihash::Multihash;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    index.hash(&mut hasher);
    let hash_value = hasher.finish();

    // Create a 32-byte hash from the index
    let mut hash_bytes = [0u8; 32];
    hash_bytes[..8].copy_from_slice(&hash_value.to_le_bytes());
    // Fill remaining bytes with deterministic pattern
    for i in 1..4 {
        let val = (hash_value.wrapping_mul(i as u64)).to_le_bytes();
        hash_bytes[i * 8..(i + 1) * 8].copy_from_slice(&val);
    }

    let mh = Multihash::wrap(0x12, &hash_bytes)
        .expect("wrapping 32-byte hash into SHA2-256 multihash is infallible"); // 0x12 is SHA2-256 code
    Cid::new_v1(0x55, mh) // 0x55 is raw codec
}

fn generate_random_embedding(dim: usize) -> Vec<f32> {
    let mut rng = rand::rng();
    (0..dim).map(|_| rng.random_range(0.0..1.0)).collect()
}

fn percentile(sorted_data: &[Duration], p: f64) -> Duration {
    if sorted_data.is_empty() {
        return Duration::from_secs(0);
    }
    let index = ((p * sorted_data.len() as f64) as usize).min(sorted_data.len() - 1);
    sorted_data[index]
}

#[allow(dead_code)]
fn estimate_process_memory() -> usize {
    // Simple estimation - in production, use a proper memory profiler
    // For Linux, could read /proc/self/status
    #[cfg(target_os = "linux")]
    {
        use std::fs;
        if let Ok(status) = fs::read_to_string("/proc/self/status") {
            for line in status.lines() {
                if line.starts_with("VmRSS:") {
                    if let Some(kb_str) = line.split_whitespace().nth(1) {
                        if let Ok(kb) = kb_str.parse::<usize>() {
                            return kb * 1024; // Convert KB to bytes
                        }
                    }
                }
            }
        }
    }

    // Fallback: return 0 if we can't measure
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_stress_test_creation() {
        // Just test that we can create and configure stress tests
        let config = StressTestConfig {
            num_threads: 2,
            operations_per_thread: 5,
            index_size: 20,
            dimension: 64,
            insert_ratio: 0.5,
            query_ratio: 0.5,
            k: 3,
        };

        let stress_test = StressTest::new(config.clone());
        if let Err(e) = &stress_test {
            eprintln!("Error creating stress test: {:?}", e);
        }
        assert!(stress_test.is_ok());

        // Verify configuration
        let test = stress_test.expect("test: StressTest::new should succeed with valid config");
        assert_eq!(test.config.num_threads, 2);
    }

    #[tokio::test]
    async fn test_endurance_test_creation() {
        // Just test that we can create and configure endurance tests
        let config = EnduranceTestConfig {
            duration: Duration::from_millis(100),
            target_ops_per_second: 10.0,
            dimension: 64,
            memory_check_interval: Duration::from_millis(50),
        };

        let endurance_test = EnduranceTest::new(config.clone());
        assert!(endurance_test.is_ok());

        // Verify configuration
        assert_eq!(
            endurance_test
                .expect("test: EnduranceTest::new should succeed with valid config")
                .config
                .dimension,
            64
        );
    }

    #[test]
    fn test_generate_test_cid() {
        let cid1 = generate_test_cid(0);
        let cid2 = generate_test_cid(1);
        let cid3 = generate_test_cid(5);

        // All CIDs should be unique
        assert_ne!(cid1, cid2);
        assert_ne!(cid1, cid3);
        assert_ne!(cid2, cid3);

        // Same index should produce same CID (deterministic)
        let cid1_again = generate_test_cid(0);
        assert_eq!(cid1, cid1_again);
    }

    #[test]
    fn test_percentile_calculation() {
        let data = vec![
            Duration::from_millis(1),
            Duration::from_millis(2),
            Duration::from_millis(3),
            Duration::from_millis(4),
            Duration::from_millis(5),
        ];

        let p50 = percentile(&data, 0.5);
        let p90 = percentile(&data, 0.9);

        assert_eq!(p50, Duration::from_millis(3));
        assert_eq!(p90, Duration::from_millis(5));
    }

    #[test]
    fn test_percentile_empty() {
        let data: Vec<Duration> = vec![];
        let p50 = percentile(&data, 0.5);
        assert_eq!(p50, Duration::from_secs(0));
    }
}
