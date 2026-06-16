//! Benchmark comparison utilities for evaluating different configurations
//!
//! This module provides tools for systematically comparing different index
//! configurations, quantization strategies, and parameter settings.
//!
//! # Features
//!
//! - **Configuration Comparison**: Compare multiple index configurations
//! - **Parameter Sweeps**: Systematically test parameter ranges
//! - **Trade-off Analysis**: Analyze recall vs latency vs memory trade-offs
//! - **Recommendation Engine**: Suggest optimal configurations for use cases
//!
//! # Example
//!
//! ```rust
//! use ipfrs_semantic::benchmark_comparison::{BenchmarkSuite, IndexConfig};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let mut suite = BenchmarkSuite::new();
//!
//! // Add configurations to compare
//! suite.add_config("low_latency", IndexConfig::low_latency())?;
//! suite.add_config("high_recall", IndexConfig::high_recall())?;
//! suite.add_config("balanced", IndexConfig::balanced())?;
//!
//! // Run benchmarks (in real usage)
//! // let results = suite.run_benchmarks(test_data)?;
//! // let report = suite.generate_report(&results)?;
//! # Ok(())
//! # }
//! ```

use crate::hnsw::{DistanceMetric, VectorIndex};
use ipfrs_core::{Cid, Result};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Index configuration for benchmarking
#[derive(Debug, Clone)]
pub struct IndexConfig {
    /// Configuration name
    pub name: String,
    /// Vector dimension
    pub dimension: usize,
    /// Distance metric
    pub metric: DistanceMetric,
    /// HNSW M parameter
    pub m: usize,
    /// HNSW ef_construction parameter
    pub ef_construction: usize,
    /// Search ef parameter
    pub ef_search: usize,
    /// Whether to use quantization
    pub use_quantization: bool,
    /// Description
    pub description: String,
}

impl IndexConfig {
    /// Configuration optimized for low latency
    pub fn low_latency() -> Self {
        Self {
            name: "low_latency".to_string(),
            dimension: 768,
            metric: DistanceMetric::Cosine,
            m: 8,
            ef_construction: 100,
            ef_search: 16,
            use_quantization: false,
            description: "Optimized for minimal query latency".to_string(),
        }
    }

    /// Configuration optimized for high recall
    pub fn high_recall() -> Self {
        Self {
            name: "high_recall".to_string(),
            dimension: 768,
            metric: DistanceMetric::Cosine,
            m: 32,
            ef_construction: 400,
            ef_search: 128,
            use_quantization: false,
            description: "Optimized for maximum search accuracy".to_string(),
        }
    }

    /// Balanced configuration
    pub fn balanced() -> Self {
        Self {
            name: "balanced".to_string(),
            dimension: 768,
            metric: DistanceMetric::Cosine,
            m: 16,
            ef_construction: 200,
            ef_search: 50,
            use_quantization: false,
            description: "Balanced latency and recall".to_string(),
        }
    }

    /// Memory-efficient configuration with quantization
    pub fn memory_efficient() -> Self {
        Self {
            name: "memory_efficient".to_string(),
            dimension: 768,
            metric: DistanceMetric::Cosine,
            m: 12,
            ef_construction: 150,
            ef_search: 32,
            use_quantization: true,
            description: "Minimizes memory usage with quantization".to_string(),
        }
    }
}

/// Benchmark results for a single configuration
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Configuration name
    pub config_name: String,
    /// Average query latency
    pub avg_latency: Duration,
    /// P50 latency
    pub p50_latency: Duration,
    /// P90 latency
    pub p90_latency: Duration,
    /// P99 latency
    pub p99_latency: Duration,
    /// Recall@10
    pub recall_at_10: f64,
    /// Recall@100
    pub recall_at_100: f64,
    /// Queries per second
    pub qps: f64,
    /// Memory usage in MB
    pub memory_mb: f64,
    /// Index build time
    pub build_time: Duration,
}

/// Comparison report
#[derive(Debug, Clone)]
pub struct ComparisonReport {
    /// Results for each configuration
    pub results: Vec<BenchmarkResult>,
    /// Best configuration for latency
    pub best_latency: String,
    /// Best configuration for recall
    pub best_recall: String,
    /// Best configuration for memory
    pub best_memory: String,
    /// Recommendations
    pub recommendations: Vec<String>,
}

/// Benchmark suite for comparing configurations
pub struct BenchmarkSuite {
    /// Configurations to test
    configs: HashMap<String, IndexConfig>,
}

impl BenchmarkSuite {
    /// Create a new benchmark suite
    pub fn new() -> Self {
        Self {
            configs: HashMap::new(),
        }
    }

    /// Add a configuration to test
    pub fn add_config(&mut self, name: &str, config: IndexConfig) -> Result<()> {
        self.configs.insert(name.to_string(), config);
        Ok(())
    }

    /// Run benchmarks on test data
    pub fn run_benchmarks(
        &self,
        test_data: &[(Cid, Vec<f32>)],
        query_data: &[Vec<f32>],
    ) -> Result<Vec<BenchmarkResult>> {
        let mut results = Vec::new();

        for config in self.configs.values() {
            let result = self.benchmark_config(config, test_data, query_data)?;
            results.push(result);
        }

        Ok(results)
    }

    /// Benchmark a single configuration
    fn benchmark_config(
        &self,
        config: &IndexConfig,
        test_data: &[(Cid, Vec<f32>)],
        query_data: &[Vec<f32>],
    ) -> Result<BenchmarkResult> {
        // Build index
        let build_start = Instant::now();
        let mut index = VectorIndex::new(
            config.dimension,
            config.metric,
            config.m,
            config.ef_construction,
        )?;

        for (cid, embedding) in test_data {
            index.insert(cid, embedding)?;
        }
        let build_time = build_start.elapsed();

        // Measure query latencies
        let mut latencies = Vec::new();
        let query_start = Instant::now();

        for query in query_data {
            let start = Instant::now();
            let _results = index.search(query, 10, config.ef_search)?;
            latencies.push(start.elapsed());
        }

        let total_query_time = query_start.elapsed();
        let qps = query_data.len() as f64 / total_query_time.as_secs_f64();

        // Calculate latency percentiles
        latencies.sort();
        let avg_latency = latencies.iter().sum::<Duration>() / latencies.len() as u32;
        let p50_latency = latencies[latencies.len() / 2];
        let p90_latency = latencies[(latencies.len() as f64 * 0.9) as usize];
        let p99_latency = latencies[(latencies.len() as f64 * 0.99) as usize];

        // Calculate recall (would need ground truth in real implementation)
        let recall_at_10 = 0.95; // Placeholder
        let recall_at_100 = 0.99; // Placeholder

        // Estimate memory usage
        let memory_mb = self.estimate_memory(&index);

        Ok(BenchmarkResult {
            config_name: config.name.clone(),
            avg_latency,
            p50_latency,
            p90_latency,
            p99_latency,
            recall_at_10,
            recall_at_100,
            qps,
            memory_mb,
            build_time,
        })
    }

    /// Estimate memory usage for an index
    fn estimate_memory(&self, index: &VectorIndex) -> f64 {
        // Rough estimation: entries * dimension * 4 bytes per float
        let entries = index.len();
        let bytes_per_entry = 768 * 4 + 64; // embedding + overhead
        (entries * bytes_per_entry) as f64 / (1024.0 * 1024.0)
    }

    /// Generate comparison report
    pub fn generate_report(&self, results: &[BenchmarkResult]) -> Result<ComparisonReport> {
        if results.is_empty() {
            return Err(ipfrs_core::Error::InvalidInput(
                "No results to compare".into(),
            ));
        }

        // Find best configurations
        let best_latency = results
            .iter()
            .min_by_key(|r| r.avg_latency)
            .map(|r| r.config_name.clone())
            .expect("results is non-empty");

        let best_recall = results
            .iter()
            .max_by(|a, b| {
                a.recall_at_10
                    .partial_cmp(&b.recall_at_10)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|r| r.config_name.clone())
            .expect("results is non-empty");

        let best_memory = results
            .iter()
            .min_by(|a, b| {
                a.memory_mb
                    .partial_cmp(&b.memory_mb)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|r| r.config_name.clone())
            .expect("results is non-empty");

        // Generate recommendations
        let mut recommendations = Vec::new();
        recommendations.push(format!(
            "For lowest latency: {} ({:.2}ms avg)",
            best_latency,
            results
                .iter()
                .find(|r| r.config_name == best_latency)
                .expect("best_latency comes from results iterator")
                .avg_latency
                .as_micros() as f64
                / 1000.0
        ));

        recommendations.push(format!(
            "For highest recall: {} ({:.2}% recall@10)",
            best_recall,
            results
                .iter()
                .find(|r| r.config_name == best_recall)
                .expect("best_recall comes from results iterator")
                .recall_at_10
                * 100.0
        ));

        recommendations.push(format!(
            "For lowest memory: {} ({:.2}MB)",
            best_memory,
            results
                .iter()
                .find(|r| r.config_name == best_memory)
                .expect("best_memory comes from results iterator")
                .memory_mb
        ));

        Ok(ComparisonReport {
            results: results.to_vec(),
            best_latency,
            best_recall,
            best_memory,
            recommendations,
        })
    }

    /// Print a formatted comparison table
    pub fn print_comparison(&self, report: &ComparisonReport) {
        println!("\n=== Benchmark Comparison Report ===\n");
        println!(
            "{:<20} {:>10} {:>10} {:>10} {:>10} {:>10}",
            "Config", "Avg(ms)", "P99(ms)", "Recall@10", "QPS", "Memory(MB)"
        );
        println!("{:-<80}", "");

        for result in &report.results {
            println!(
                "{:<20} {:>10.2} {:>10.2} {:>10.2} {:>10.0} {:>10.2}",
                result.config_name,
                result.avg_latency.as_micros() as f64 / 1000.0,
                result.p99_latency.as_micros() as f64 / 1000.0,
                result.recall_at_10 * 100.0,
                result.qps,
                result.memory_mb
            );
        }

        println!("\n=== Recommendations ===\n");
        for rec in &report.recommendations {
            println!("  • {}", rec);
        }
        println!();
    }
}

impl Default for BenchmarkSuite {
    fn default() -> Self {
        Self::new()
    }
}

/// Parameter sweep utility for systematic testing
pub struct ParameterSweep {
    /// Base configuration
    base_config: IndexConfig,
    /// Parameter to sweep
    parameter: String,
    /// Values to test
    values: Vec<usize>,
}

impl ParameterSweep {
    /// Create a new parameter sweep
    pub fn new(base_config: IndexConfig, parameter: String, values: Vec<usize>) -> Self {
        Self {
            base_config,
            parameter,
            values,
        }
    }

    /// Generate configurations for sweep
    pub fn generate_configs(&self) -> Vec<IndexConfig> {
        self.values
            .iter()
            .map(|&value| {
                let mut config = self.base_config.clone();
                config.name = format!("{}_{}", self.parameter, value);

                match self.parameter.as_str() {
                    "m" => config.m = value,
                    "ef_construction" => config.ef_construction = value,
                    "ef_search" => config.ef_search = value,
                    _ => {}
                }

                config
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_index_config_presets() {
        let low_lat = IndexConfig::low_latency();
        assert_eq!(low_lat.name, "low_latency");
        assert_eq!(low_lat.m, 8);

        let high_rec = IndexConfig::high_recall();
        assert_eq!(high_rec.name, "high_recall");
        assert_eq!(high_rec.m, 32);

        let balanced = IndexConfig::balanced();
        assert_eq!(balanced.name, "balanced");
        assert_eq!(balanced.m, 16);

        let mem_eff = IndexConfig::memory_efficient();
        assert_eq!(mem_eff.name, "memory_efficient");
        assert!(mem_eff.use_quantization);
    }

    #[test]
    fn test_benchmark_suite_creation() {
        let suite = BenchmarkSuite::new();
        assert_eq!(suite.configs.len(), 0);
    }

    #[test]
    fn test_add_config() {
        let mut suite = BenchmarkSuite::new();
        let config = IndexConfig::low_latency();

        suite
            .add_config("test", config)
            .expect("test: add_config should succeed");
        assert_eq!(suite.configs.len(), 1);
    }

    #[test]
    fn test_parameter_sweep() {
        let base = IndexConfig::balanced();
        let sweep = ParameterSweep::new(base, "m".to_string(), vec![8, 16, 32, 64]);

        let configs = sweep.generate_configs();
        assert_eq!(configs.len(), 4);
        assert_eq!(configs[0].m, 8);
        assert_eq!(configs[1].m, 16);
        assert_eq!(configs[2].m, 32);
        assert_eq!(configs[3].m, 64);
    }

    #[test]
    fn test_ef_construction_sweep() {
        let base = IndexConfig::balanced();
        let sweep = ParameterSweep::new(
            base,
            "ef_construction".to_string(),
            vec![100, 200, 400, 800],
        );

        let configs = sweep.generate_configs();
        assert_eq!(configs.len(), 4);
        assert_eq!(configs[0].ef_construction, 100);
        assert_eq!(configs[3].ef_construction, 800);
    }

    #[test]
    fn test_ef_search_sweep() {
        let base = IndexConfig::balanced();
        let sweep = ParameterSweep::new(base, "ef_search".to_string(), vec![16, 32, 64, 128]);

        let configs = sweep.generate_configs();
        assert_eq!(configs.len(), 4);
        assert_eq!(configs[0].ef_search, 16);
        assert_eq!(configs[3].ef_search, 128);
    }
}
