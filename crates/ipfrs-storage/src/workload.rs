//! Workload simulation and generation for testing and benchmarking
//!
//! This module provides utilities for generating realistic storage workloads
//! for testing, benchmarking, and capacity planning.

use crate::traits::BlockStore;
use crate::utils::create_block;
use ipfrs_core::{Block, Cid, Result};
use rand::{Rng, RngExt};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::time::sleep;

/// Workload pattern for simulation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WorkloadPattern {
    /// Uniform random access
    Uniform,
    /// Zipf distribution (80/20 rule)
    Zipfian { alpha: f64 },
    /// Sequential access pattern
    Sequential,
    /// Burst pattern with periods of high activity
    Bursty {
        burst_duration: Duration,
        idle_duration: Duration,
    },
    /// Time-series pattern (recent blocks more likely)
    TimeSeries { decay_factor: f64 },
}

/// Operation mix for workload
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationMix {
    /// Percentage of put operations (0.0 - 1.0)
    pub put_ratio: f64,
    /// Percentage of get operations (0.0 - 1.0)
    pub get_ratio: f64,
    /// Percentage of has operations (0.0 - 1.0)
    pub has_ratio: f64,
    /// Percentage of delete operations (0.0 - 1.0)
    pub delete_ratio: f64,
}

impl OperationMix {
    /// Create a read-heavy workload (80% reads, 20% writes)
    pub fn read_heavy() -> Self {
        Self {
            put_ratio: 0.15,
            get_ratio: 0.70,
            has_ratio: 0.10,
            delete_ratio: 0.05,
        }
    }

    /// Create a write-heavy workload (80% writes, 20% reads)
    pub fn write_heavy() -> Self {
        Self {
            put_ratio: 0.60,
            get_ratio: 0.15,
            has_ratio: 0.05,
            delete_ratio: 0.20,
        }
    }

    /// Create a balanced workload
    pub fn balanced() -> Self {
        Self {
            put_ratio: 0.25,
            get_ratio: 0.50,
            has_ratio: 0.15,
            delete_ratio: 0.10,
        }
    }

    /// Create a cache-like workload (mostly reads, few writes)
    pub fn cache() -> Self {
        Self {
            put_ratio: 0.10,
            get_ratio: 0.85,
            has_ratio: 0.04,
            delete_ratio: 0.01,
        }
    }

    /// Validate that ratios sum to 1.0
    pub fn validate(&self) -> bool {
        let sum = self.put_ratio + self.get_ratio + self.has_ratio + self.delete_ratio;
        (sum - 1.0).abs() < 0.001
    }
}

/// Block size distribution for workload
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SizeDistribution {
    /// Fixed size blocks
    Fixed { size: usize },
    /// Uniform distribution between min and max
    Uniform { min: usize, max: usize },
    /// Normal distribution
    Normal { mean: usize, stddev: usize },
    /// Mixed sizes (small, medium, large with percentages)
    Mixed {
        small_size: usize,
        small_pct: f64,
        medium_size: usize,
        medium_pct: f64,
        large_size: usize,
        large_pct: f64,
    },
}

/// Workload configuration
#[derive(Debug, Clone)]
pub struct WorkloadConfig {
    /// Total number of operations to perform
    pub total_operations: usize,
    /// Number of unique blocks in the dataset
    pub dataset_size: usize,
    /// Operation mix
    pub operation_mix: OperationMix,
    /// Access pattern
    pub pattern: WorkloadPattern,
    /// Block size distribution
    pub size_distribution: SizeDistribution,
    /// Concurrency level (number of parallel tasks)
    pub concurrency: usize,
    /// Rate limit (operations per second, 0 = unlimited)
    pub rate_limit: usize,
    /// Percentage of compressible blocks (0.0 - 1.0)
    pub compressible_ratio: f64,
}

impl Default for WorkloadConfig {
    fn default() -> Self {
        Self {
            total_operations: 10_000,
            dataset_size: 1_000,
            operation_mix: OperationMix::balanced(),
            pattern: WorkloadPattern::Uniform,
            size_distribution: SizeDistribution::Uniform {
                min: 1024,
                max: 65536,
            },
            concurrency: 4,
            rate_limit: 0,
            compressible_ratio: 0.5,
        }
    }
}

/// Workload execution results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkloadResult {
    /// Total operations executed
    pub total_operations: usize,
    /// Operations per second
    pub ops_per_second: f64,
    /// Total duration
    pub duration: Duration,
    /// Per-operation breakdown
    pub operation_counts: HashMap<String, usize>,
    /// Per-operation latencies (microseconds)
    pub operation_latencies: HashMap<String, Vec<u64>>,
    /// Errors encountered
    pub errors: usize,
    /// Throughput in bytes per second
    pub throughput_bps: f64,
}

impl WorkloadResult {
    /// Calculate average latency for an operation
    pub fn avg_latency(&self, operation: &str) -> Option<f64> {
        self.operation_latencies.get(operation).map(|latencies| {
            if latencies.is_empty() {
                0.0
            } else {
                latencies.iter().sum::<u64>() as f64 / latencies.len() as f64
            }
        })
    }

    /// Calculate P95 latency for an operation
    pub fn p95_latency(&self, operation: &str) -> Option<u64> {
        self.operation_latencies
            .get(operation)
            .and_then(|latencies| {
                if latencies.is_empty() {
                    None
                } else {
                    let mut sorted = latencies.clone();
                    sorted.sort_unstable();
                    let idx = (sorted.len() as f64 * 0.95) as usize;
                    Some(sorted[idx.min(sorted.len() - 1)])
                }
            })
    }

    /// Calculate P99 latency for an operation
    pub fn p99_latency(&self, operation: &str) -> Option<u64> {
        self.operation_latencies
            .get(operation)
            .and_then(|latencies| {
                if latencies.is_empty() {
                    None
                } else {
                    let mut sorted = latencies.clone();
                    sorted.sort_unstable();
                    let idx = (sorted.len() as f64 * 0.99) as usize;
                    Some(sorted[idx.min(sorted.len() - 1)])
                }
            })
    }
}

/// Workload simulator for generating and executing storage workloads
pub struct WorkloadSimulator {
    config: WorkloadConfig,
    dataset: Vec<Block>,
    cids: Vec<Cid>,
}

impl WorkloadSimulator {
    /// Create a new workload simulator with the given configuration
    pub fn new(config: WorkloadConfig) -> Self {
        Self {
            config,
            dataset: Vec::new(),
            cids: Vec::new(),
        }
    }

    /// Generate the initial dataset
    pub fn generate_dataset(&mut self) {
        let mut rng = rand::rng();
        self.dataset.clear();
        self.cids.clear();

        for _ in 0..self.config.dataset_size {
            let size = self.generate_block_size(&mut rng);
            let data: Vec<u8> = (0..size).map(|_| rng.random::<u8>()).collect();
            let block = create_block(data).expect("Failed to create block");
            self.cids.push(*block.cid());
            self.dataset.push(block);
        }
    }

    /// Generate a block size according to the distribution
    fn generate_block_size(&self, rng: &mut impl Rng) -> usize {
        match &self.config.size_distribution {
            SizeDistribution::Fixed { size } => *size,
            SizeDistribution::Uniform { min, max } => rng.random_range(*min..=*max),
            SizeDistribution::Normal { mean, stddev } => {
                // Box-Muller transform for normal distribution
                let u1: f64 = rng.random();
                let u2: f64 = rng.random();
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                let size = *mean as f64 + z * (*stddev as f64);
                size.max(1.0) as usize
            }
            SizeDistribution::Mixed {
                small_size,
                small_pct,
                medium_size,
                medium_pct,
                large_size,
                large_pct: _,
            } => {
                let r: f64 = rng.random();
                if r < *small_pct {
                    *small_size
                } else if r < *small_pct + *medium_pct {
                    *medium_size
                } else {
                    *large_size
                }
            }
        }
    }

    /// Select a block index according to the access pattern
    #[allow(dead_code)]
    fn select_block_index(&self, rng: &mut impl Rng, operation_num: usize) -> usize {
        match &self.config.pattern {
            WorkloadPattern::Uniform => rng.random_range(0..self.dataset.len()),
            WorkloadPattern::Zipfian { alpha } => {
                // Zipfian distribution using rejection sampling
                let n = self.dataset.len() as f64;
                loop {
                    let u: f64 = rng.random();
                    let v: f64 = rng.random();
                    let x = ((n.powf(1.0 - alpha) - 1.0) * u + 1.0).powf(1.0 / (1.0 - alpha));
                    if x <= n && v * x.powf(*alpha) <= 1.0 {
                        return (x - 1.0) as usize;
                    }
                }
            }
            WorkloadPattern::Sequential => operation_num % self.dataset.len(),
            WorkloadPattern::Bursty { .. } => rng.random_range(0..self.dataset.len()),
            WorkloadPattern::TimeSeries { decay_factor } => {
                // Exponential decay for time-series access
                let r: f64 = rng.random();
                let idx = (-r.ln() / decay_factor) as usize;
                idx.min(self.dataset.len() - 1)
            }
        }
    }

    /// Select an operation according to the operation mix
    #[allow(dead_code)]
    fn select_operation(&self, rng: &mut impl Rng) -> &str {
        let r: f64 = rng.random();
        let mix = &self.config.operation_mix;

        if r < mix.put_ratio {
            "put"
        } else if r < mix.put_ratio + mix.get_ratio {
            "get"
        } else if r < mix.put_ratio + mix.get_ratio + mix.has_ratio {
            "has"
        } else {
            "delete"
        }
    }

    /// Run the workload against a block store
    pub async fn run<S: BlockStore + Send + Sync + 'static>(
        &self,
        store: Arc<S>,
    ) -> Result<WorkloadResult> {
        let start = Instant::now();
        let mut operation_counts: HashMap<String, usize> = HashMap::new();
        let mut operation_latencies: HashMap<String, Vec<u64>> = HashMap::new();
        let mut errors = 0usize;
        let mut total_bytes = 0usize;

        // Divide operations among concurrent tasks
        let ops_per_task = self.config.total_operations / self.config.concurrency;
        let mut tasks = Vec::new();

        for task_id in 0..self.config.concurrency {
            let store = store.clone();
            let dataset = self.dataset.clone();
            let cids = self.cids.clone();
            let config = self.config.clone();
            let start_op = task_id * ops_per_task;
            let end_op = if task_id == self.config.concurrency - 1 {
                self.config.total_operations
            } else {
                (task_id + 1) * ops_per_task
            };

            let task = tokio::spawn(async move {
                // Use a simple deterministic RNG for the task
                use rand::SeedableRng;
                let mut rng = rand::rngs::SmallRng::seed_from_u64(task_id as u64);
                let mut task_counts: HashMap<String, usize> = HashMap::new();
                let mut task_latencies: HashMap<String, Vec<u64>> = HashMap::new();
                let mut task_errors = 0usize;
                let mut task_bytes = 0usize;

                for op_num in start_op..end_op {
                    // Rate limiting
                    if config.rate_limit > 0 {
                        let delay = Duration::from_secs_f64(1.0 / config.rate_limit as f64);
                        sleep(delay).await;
                    }

                    let idx = if dataset.is_empty() {
                        0
                    } else {
                        op_num % dataset.len()
                    };
                    let operation = if dataset.is_empty() {
                        "get"
                    } else {
                        let r: f64 = rng.random();
                        let mix = &config.operation_mix;
                        if r < mix.put_ratio {
                            "put"
                        } else if r < mix.put_ratio + mix.get_ratio {
                            "get"
                        } else if r < mix.put_ratio + mix.get_ratio + mix.has_ratio {
                            "has"
                        } else {
                            "delete"
                        }
                    };

                    let op_start = Instant::now();
                    let result = match operation {
                        "put" => {
                            if idx < dataset.len() {
                                task_bytes += dataset[idx].data().len();
                                store.put(&dataset[idx]).await
                            } else {
                                Ok(())
                            }
                        }
                        "get" => {
                            if idx < cids.len() {
                                match store.get(&cids[idx]).await {
                                    Ok(Some(block)) => {
                                        task_bytes += block.data().len();
                                        Ok(())
                                    }
                                    Ok(None) => Ok(()),
                                    Err(e) => Err(e),
                                }
                            } else {
                                Ok(())
                            }
                        }
                        "has" => {
                            if idx < cids.len() {
                                store.has(&cids[idx]).await.map(|_| ())
                            } else {
                                Ok(())
                            }
                        }
                        "delete" => {
                            if idx < cids.len() {
                                store.delete(&cids[idx]).await
                            } else {
                                Ok(())
                            }
                        }
                        _ => Ok(()),
                    };

                    let latency = op_start.elapsed().as_micros() as u64;

                    *task_counts.entry(operation.to_string()).or_insert(0) += 1;
                    task_latencies
                        .entry(operation.to_string())
                        .or_default()
                        .push(latency);

                    if result.is_err() {
                        task_errors += 1;
                    }
                }

                (task_counts, task_latencies, task_errors, task_bytes)
            });

            tasks.push(task);
        }

        // Collect results from all tasks
        for task in tasks {
            let (task_counts, task_latencies, task_errors, task_bytes) =
                task.await.expect("workload task should not panic");

            for (op, count) in task_counts {
                *operation_counts.entry(op).or_insert(0) += count;
            }

            for (op, latencies) in task_latencies {
                operation_latencies.entry(op).or_default().extend(latencies);
            }

            errors += task_errors;
            total_bytes += task_bytes;
        }

        let duration = start.elapsed();
        let ops_per_second = self.config.total_operations as f64 / duration.as_secs_f64();
        let throughput_bps = total_bytes as f64 / duration.as_secs_f64();

        Ok(WorkloadResult {
            total_operations: self.config.total_operations,
            ops_per_second,
            duration,
            operation_counts,
            operation_latencies,
            errors,
            throughput_bps,
        })
    }
}

/// Workload presets for common scenarios
pub struct WorkloadPresets;

impl WorkloadPresets {
    /// Light testing workload (1K operations, 100 blocks)
    #[must_use]
    pub fn light_test() -> WorkloadConfig {
        WorkloadConfig {
            total_operations: 1_000,
            dataset_size: 100,
            operation_mix: OperationMix::balanced(),
            pattern: WorkloadPattern::Uniform,
            size_distribution: SizeDistribution::Uniform {
                min: 1024,
                max: 4096,
            },
            concurrency: 2,
            rate_limit: 0,
            compressible_ratio: 0.5,
        }
    }

    /// Medium stress test (100K operations, 10K blocks)
    #[must_use]
    pub fn medium_stress() -> WorkloadConfig {
        WorkloadConfig {
            total_operations: 100_000,
            dataset_size: 10_000,
            operation_mix: OperationMix::balanced(),
            pattern: WorkloadPattern::Zipfian { alpha: 1.1 },
            size_distribution: SizeDistribution::Mixed {
                small_size: 1024,
                small_pct: 0.5,
                medium_size: 16384,
                medium_pct: 0.3,
                large_size: 65536,
                large_pct: 0.2,
            },
            concurrency: 8,
            rate_limit: 0,
            compressible_ratio: 0.7,
        }
    }

    /// Heavy stress test (1M operations, 100K blocks)
    #[must_use]
    pub fn heavy_stress() -> WorkloadConfig {
        WorkloadConfig {
            total_operations: 1_000_000,
            dataset_size: 100_000,
            operation_mix: OperationMix::balanced(),
            pattern: WorkloadPattern::Zipfian { alpha: 1.1 },
            size_distribution: SizeDistribution::Mixed {
                small_size: 1024,
                small_pct: 0.4,
                medium_size: 32768,
                medium_pct: 0.4,
                large_size: 262144,
                large_pct: 0.2,
            },
            concurrency: 16,
            rate_limit: 0,
            compressible_ratio: 0.6,
        }
    }

    /// CDN cache simulation (read-heavy, Zipfian access)
    #[must_use]
    pub fn cdn_cache() -> WorkloadConfig {
        WorkloadConfig {
            total_operations: 50_000,
            dataset_size: 5_000,
            operation_mix: OperationMix::cache(),
            pattern: WorkloadPattern::Zipfian { alpha: 1.2 },
            size_distribution: SizeDistribution::Mixed {
                small_size: 4096,
                small_pct: 0.3,
                medium_size: 65536,
                medium_pct: 0.5,
                large_size: 1048576,
                large_pct: 0.2,
            },
            concurrency: 12,
            rate_limit: 0,
            compressible_ratio: 0.8,
        }
    }

    /// Data ingestion pipeline (write-heavy)
    #[must_use]
    pub fn ingestion_pipeline() -> WorkloadConfig {
        WorkloadConfig {
            total_operations: 100_000,
            dataset_size: 50_000,
            operation_mix: OperationMix::write_heavy(),
            pattern: WorkloadPattern::Sequential,
            size_distribution: SizeDistribution::Normal {
                mean: 32768,
                stddev: 8192,
            },
            concurrency: 8,
            rate_limit: 1000, // 1000 ops/sec
            compressible_ratio: 0.9,
        }
    }

    /// Time-series data access
    #[must_use]
    pub fn time_series() -> WorkloadConfig {
        WorkloadConfig {
            total_operations: 20_000,
            dataset_size: 10_000,
            operation_mix: OperationMix::read_heavy(),
            pattern: WorkloadPattern::TimeSeries { decay_factor: 0.1 },
            size_distribution: SizeDistribution::Fixed { size: 8192 },
            concurrency: 4,
            rate_limit: 0,
            compressible_ratio: 0.6,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MemoryBlockStore;

    #[test]
    fn test_operation_mix_validation() {
        let mix = OperationMix::balanced();
        assert!(mix.validate());

        let invalid_mix = OperationMix {
            put_ratio: 0.5,
            get_ratio: 0.3,
            has_ratio: 0.1,
            delete_ratio: 0.05,
        };
        assert!(!invalid_mix.validate());
    }

    #[test]
    fn test_dataset_generation() {
        let config = WorkloadPresets::light_test();
        let mut simulator = WorkloadSimulator::new(config);
        simulator.generate_dataset();

        assert_eq!(simulator.dataset.len(), 100);
        assert_eq!(simulator.cids.len(), 100);
    }

    #[tokio::test]
    async fn test_light_workload() {
        let config = WorkloadPresets::light_test();
        let mut simulator = WorkloadSimulator::new(config);
        simulator.generate_dataset();

        let store = Arc::new(MemoryBlockStore::new());
        let result = simulator.run(store).await.unwrap();

        assert_eq!(result.total_operations, 1_000);
        assert!(result.ops_per_second > 0.0);
        assert!(!result.operation_counts.is_empty());
    }

    #[tokio::test]
    async fn test_workload_latencies() {
        let config = WorkloadPresets::light_test();
        let mut simulator = WorkloadSimulator::new(config);
        simulator.generate_dataset();

        let store = Arc::new(MemoryBlockStore::new());
        let result = simulator.run(store).await.unwrap();

        // Check that latencies are recorded
        for latencies in result.operation_latencies.values() {
            assert!(!latencies.is_empty());
        }

        // Check percentile calculations
        // Note: p95 can be 0 for very fast in-memory operations
        // Just verify that p95 calculation works (returns Some)
        assert!(result.p95_latency("get").is_some());
    }

    #[test]
    fn test_workload_presets() {
        let _light = WorkloadPresets::light_test();
        let _medium = WorkloadPresets::medium_stress();
        let _heavy = WorkloadPresets::heavy_stress();
        let _cdn = WorkloadPresets::cdn_cache();
        let _ingestion = WorkloadPresets::ingestion_pipeline();
        let _timeseries = WorkloadPresets::time_series();
    }
}
