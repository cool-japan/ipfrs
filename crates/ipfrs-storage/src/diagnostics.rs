//! Storage diagnostics and health monitoring utilities
//!
//! Provides comprehensive tools for analyzing storage performance,
//! health, and identifying potential issues.

use crate::traits::BlockStore;
use ipfrs_core::{Block, Cid, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use sysinfo::{ProcessesToUpdate, System};

/// Comprehensive storage diagnostics report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnosticsReport {
    /// Storage backend name
    pub backend: String,
    /// Total blocks tested
    pub total_blocks: usize,
    /// Performance metrics
    pub performance: PerformanceMetrics,
    /// Health check results
    pub health: HealthMetrics,
    /// Recommendations for optimization
    pub recommendations: Vec<String>,
    /// Overall health score (0-100)
    pub health_score: u8,
}

/// Performance metrics for storage operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Average write latency
    pub avg_write_latency: Duration,
    /// Average read latency
    pub avg_read_latency: Duration,
    /// Average batch write latency
    pub avg_batch_write_latency: Duration,
    /// Average batch read latency
    pub avg_batch_read_latency: Duration,
    /// Write throughput (blocks/sec)
    pub write_throughput: f64,
    /// Read throughput (blocks/sec)
    pub read_throughput: f64,
    /// Peak memory usage (bytes)
    pub peak_memory_usage: usize,
}

/// Health metrics for storage backend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthMetrics {
    /// Number of successful operations
    pub successful_ops: usize,
    /// Number of failed operations
    pub failed_ops: usize,
    /// Success rate (0.0 - 1.0)
    pub success_rate: f64,
    /// Data integrity check passed
    pub integrity_ok: bool,
    /// Storage is responsive
    pub responsive: bool,
}

/// Memory usage tracker for diagnostics
struct MemoryTracker {
    system: System,
    pid: sysinfo::Pid,
    peak_memory: usize,
}

impl MemoryTracker {
    /// Create a new memory tracker
    fn new() -> Self {
        let mut system = System::new();
        system.refresh_processes(ProcessesToUpdate::All, true);
        let pid = sysinfo::get_current_pid().expect("current process always has a PID");

        Self {
            system,
            pid,
            peak_memory: 0,
        }
    }

    /// Update peak memory usage
    fn update(&mut self) {
        self.system.refresh_processes(ProcessesToUpdate::All, true);
        if let Some(process) = self.system.process(self.pid) {
            let current_memory = process.memory() as usize;
            if current_memory > self.peak_memory {
                self.peak_memory = current_memory;
            }
        }
    }

    /// Get peak memory usage in bytes
    fn peak_memory_bytes(&self) -> usize {
        self.peak_memory
    }
}

/// Storage diagnostics runner
pub struct StorageDiagnostics<S: BlockStore> {
    store: S,
    backend_name: String,
}

impl<S: BlockStore> StorageDiagnostics<S> {
    /// Create a new diagnostics runner
    pub fn new(store: S, backend_name: String) -> Self {
        Self {
            store,
            backend_name,
        }
    }

    /// Run comprehensive diagnostics
    ///
    /// Tests include:
    /// - Write/read latency measurements
    /// - Batch operation performance
    /// - Data integrity verification
    /// - Storage responsiveness
    /// - Memory usage tracking
    pub async fn run(&mut self) -> Result<DiagnosticsReport> {
        let mut successful_ops = 0;
        let mut failed_ops = 0;

        // Initialize memory tracker
        let mut memory_tracker = MemoryTracker::new();
        memory_tracker.update();

        // Test data
        let test_blocks = self.generate_test_data()?;
        memory_tracker.update();

        // Measure write performance
        let write_start = Instant::now();
        for block in &test_blocks {
            match self.store.put(block).await {
                Ok(_) => successful_ops += 1,
                Err(_) => failed_ops += 1,
            }
        }
        let write_duration = write_start.elapsed();
        let avg_write_latency = write_duration / test_blocks.len() as u32;
        memory_tracker.update();

        // Measure read performance
        let read_start = Instant::now();
        let mut integrity_ok = true;
        for block in &test_blocks {
            match self.store.get(block.cid()).await {
                Ok(Some(retrieved)) => {
                    if retrieved.data() != block.data() {
                        integrity_ok = false;
                    }
                    successful_ops += 1;
                }
                Ok(None) => {
                    integrity_ok = false;
                    failed_ops += 1;
                }
                Err(_) => failed_ops += 1,
            }
        }
        let read_duration = read_start.elapsed();
        let avg_read_latency = read_duration / test_blocks.len() as u32;
        memory_tracker.update();

        // Measure batch write performance
        let batch_write_start = Instant::now();
        let batch_result = self.store.put_many(&test_blocks).await;
        let avg_batch_write_latency = batch_write_start.elapsed();
        if batch_result.is_ok() {
            successful_ops += test_blocks.len();
        } else {
            failed_ops += test_blocks.len();
        }
        memory_tracker.update();

        // Measure batch read performance
        let cids: Vec<Cid> = test_blocks.iter().map(|b| *b.cid()).collect();
        let batch_read_start = Instant::now();
        let _batch_read_result = self.store.get_many(&cids).await;
        let avg_batch_read_latency = batch_read_start.elapsed();
        memory_tracker.update();

        // Calculate throughput
        let write_throughput = test_blocks.len() as f64 / write_duration.as_secs_f64();
        let read_throughput = test_blocks.len() as f64 / read_duration.as_secs_f64();

        // Calculate success rate
        let total_ops = successful_ops + failed_ops;
        let success_rate = if total_ops > 0 {
            successful_ops as f64 / total_ops as f64
        } else {
            0.0
        };

        // Check responsiveness
        let responsive = avg_write_latency < Duration::from_secs(1)
            && avg_read_latency < Duration::from_millis(500);

        // Generate recommendations
        let recommendations = self.generate_recommendations(
            &avg_write_latency,
            &avg_read_latency,
            write_throughput,
            read_throughput,
            integrity_ok,
            responsive,
        );

        // Calculate health score
        let health_score = self.calculate_health_score(
            success_rate,
            integrity_ok,
            responsive,
            write_throughput,
            read_throughput,
        );

        // Get peak memory usage
        let peak_memory_usage = memory_tracker.peak_memory_bytes();

        Ok(DiagnosticsReport {
            backend: self.backend_name.clone(),
            total_blocks: test_blocks.len(),
            performance: PerformanceMetrics {
                avg_write_latency,
                avg_read_latency,
                avg_batch_write_latency,
                avg_batch_read_latency,
                write_throughput,
                read_throughput,
                peak_memory_usage,
            },
            health: HealthMetrics {
                successful_ops,
                failed_ops,
                success_rate,
                integrity_ok,
                responsive,
            },
            recommendations,
            health_score,
        })
    }

    /// Run quick health check (minimal overhead)
    pub async fn quick_health_check(&mut self) -> Result<bool> {
        // Test with a single small block
        let test_data = vec![0u8; 1024];
        let cid = crate::utils::compute_cid(&test_data);
        let block = Block::from_parts(cid, test_data.into());

        // Try write
        self.store.put(&block).await?;

        // Try read
        let retrieved = self.store.get(&cid).await?;

        // Verify
        Ok(retrieved.is_some_and(|r| r.cid() == &cid))
    }

    /// Generate test data for diagnostics
    fn generate_test_data(&self) -> Result<Vec<Block>> {
        crate::utils::generate_mixed_size_blocks(5, 3, 2)
    }

    /// Generate recommendations based on metrics
    #[allow(clippy::too_many_arguments)]
    fn generate_recommendations(
        &self,
        avg_write_latency: &Duration,
        avg_read_latency: &Duration,
        write_throughput: f64,
        read_throughput: f64,
        integrity_ok: bool,
        responsive: bool,
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        if *avg_write_latency > Duration::from_millis(100) {
            recommendations.push(
                "High write latency detected. Consider enabling write coalescing or batch operations.".to_string()
            );
        }

        if *avg_read_latency > Duration::from_millis(50) {
            recommendations.push(
                "High read latency detected. Consider enabling caching or bloom filters."
                    .to_string(),
            );
        }

        if write_throughput < 100.0 {
            recommendations.push(
                "Low write throughput. Consider using ParityDB backend or enabling compression."
                    .to_string(),
            );
        }

        if read_throughput < 200.0 {
            recommendations.push(
                "Low read throughput. Consider increasing cache size or using tiered caching."
                    .to_string(),
            );
        }

        if !integrity_ok {
            recommendations.push(
                "Data integrity issues detected! This is critical and should be investigated immediately.".to_string()
            );
        }

        if !responsive {
            recommendations.push(
                "Storage backend is not responsive. Check system resources and backend configuration.".to_string()
            );
        }

        if recommendations.is_empty() {
            recommendations.push("Storage is performing well. No issues detected.".to_string());
        }

        recommendations
    }

    /// Calculate overall health score (0-100)
    fn calculate_health_score(
        &self,
        success_rate: f64,
        integrity_ok: bool,
        responsive: bool,
        write_throughput: f64,
        read_throughput: f64,
    ) -> u8 {
        let mut score = 0u32;

        // Success rate (40 points)
        score += (success_rate * 40.0) as u32;

        // Integrity (30 points)
        if integrity_ok {
            score += 30;
        }

        // Responsiveness (15 points)
        if responsive {
            score += 15;
        }

        // Write throughput (7.5 points)
        if write_throughput >= 100.0 {
            score += 7;
        } else {
            score += (write_throughput / 100.0 * 7.0) as u32;
        }

        // Read throughput (7.5 points)
        if read_throughput >= 200.0 {
            score += 8;
        } else {
            score += (read_throughput / 200.0 * 8.0) as u32;
        }

        score.min(100) as u8
    }
}

/// Benchmark comparison between different storage backends
pub struct BenchmarkComparison {
    results: HashMap<String, DiagnosticsReport>,
}

impl BenchmarkComparison {
    /// Create a new benchmark comparison
    pub fn new() -> Self {
        Self {
            results: HashMap::new(),
        }
    }

    /// Add a benchmark result
    pub fn add_result(&mut self, name: String, report: DiagnosticsReport) {
        self.results.insert(name, report);
    }

    /// Get the fastest backend for writes
    pub fn fastest_write_backend(&self) -> Option<(&str, &DiagnosticsReport)> {
        self.results
            .iter()
            .min_by_key(|(_, r)| r.performance.avg_write_latency)
            .map(|(name, report)| (name.as_str(), report))
    }

    /// Get the fastest backend for reads
    pub fn fastest_read_backend(&self) -> Option<(&str, &DiagnosticsReport)> {
        self.results
            .iter()
            .min_by_key(|(_, r)| r.performance.avg_read_latency)
            .map(|(name, report)| (name.as_str(), report))
    }

    /// Get the healthiest backend
    pub fn healthiest_backend(&self) -> Option<(&str, &DiagnosticsReport)> {
        self.results
            .iter()
            .max_by_key(|(_, r)| r.health_score)
            .map(|(name, report)| (name.as_str(), report))
    }

    /// Generate a comparison summary
    pub fn summary(&self) -> String {
        let mut summary = String::from("=== Storage Backend Comparison ===\n\n");

        for (name, report) in &self.results {
            summary.push_str(&format!(
                "{}: Health Score = {}/100\n",
                name, report.health_score
            ));
            summary.push_str(&format!(
                "  Write Latency: {:?}, Read Latency: {:?}\n",
                report.performance.avg_write_latency, report.performance.avg_read_latency
            ));
            summary.push_str(&format!(
                "  Write Throughput: {:.2} blocks/s, Read Throughput: {:.2} blocks/s\n\n",
                report.performance.write_throughput, report.performance.read_throughput
            ));
        }

        if let Some((name, _)) = self.fastest_write_backend() {
            summary.push_str(&format!("Fastest for writes: {name}\n"));
        }

        if let Some((name, _)) = self.fastest_read_backend() {
            summary.push_str(&format!("Fastest for reads: {name}\n"));
        }

        if let Some((name, _)) = self.healthiest_backend() {
            summary.push_str(&format!("Healthiest overall: {name}\n"));
        }

        summary
    }
}

impl Default for BenchmarkComparison {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MemoryBlockStore;

    #[tokio::test]
    async fn test_diagnostics_run() {
        let store = MemoryBlockStore::new();
        let mut diagnostics = StorageDiagnostics::new(store, "MemoryStore".to_string());

        let report = diagnostics.run().await.unwrap();
        assert_eq!(report.backend, "MemoryStore");
        assert!(report.health_score > 0);
        assert!(report.health.integrity_ok);
    }

    #[tokio::test]
    async fn test_quick_health_check() {
        let store = MemoryBlockStore::new();
        let mut diagnostics = StorageDiagnostics::new(store, "MemoryStore".to_string());

        let healthy = diagnostics.quick_health_check().await.unwrap();
        assert!(healthy);
    }

    #[tokio::test]
    async fn test_benchmark_comparison() {
        let mut comparison = BenchmarkComparison::new();

        let store1 = MemoryBlockStore::new();
        let mut diag1 = StorageDiagnostics::new(store1, "Memory1".to_string());
        let report1 = diag1.run().await.unwrap();
        comparison.add_result("Memory1".to_string(), report1);

        let store2 = MemoryBlockStore::new();
        let mut diag2 = StorageDiagnostics::new(store2, "Memory2".to_string());
        let report2 = diag2.run().await.unwrap();
        comparison.add_result("Memory2".to_string(), report2);

        assert!(comparison.fastest_write_backend().is_some());
        assert!(comparison.fastest_read_backend().is_some());
        assert!(comparison.healthiest_backend().is_some());

        let summary = comparison.summary();
        assert!(summary.contains("Storage Backend Comparison"));
    }
}
