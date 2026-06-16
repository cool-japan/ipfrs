//! Advanced performance profiling utilities
//!
//! Provides detailed performance profiling with histograms, percentiles,
//! and latency distributions for in-depth performance analysis.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::time::Duration;

/// Latency histogram for tracking operation latencies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyHistogram {
    /// Bucket boundaries in microseconds
    buckets: Vec<u64>,
    /// Counts for each bucket
    counts: Vec<u64>,
    /// Total samples
    total_samples: u64,
    /// Minimum latency observed
    min_latency_us: u64,
    /// Maximum latency observed
    max_latency_us: u64,
    /// Sum of all latencies for average calculation
    sum_latency_us: u64,
}

impl LatencyHistogram {
    /// Create a new latency histogram with default buckets
    ///
    /// Default buckets: [10, 50, 100, 500, 1000, 5000, 10000, 50000] microseconds
    pub fn new() -> Self {
        Self::with_buckets(vec![10, 50, 100, 500, 1000, 5000, 10000, 50000])
    }

    /// Create a histogram with custom bucket boundaries
    pub fn with_buckets(mut buckets: Vec<u64>) -> Self {
        buckets.sort_unstable();
        let counts = vec![0; buckets.len() + 1];

        Self {
            buckets,
            counts,
            total_samples: 0,
            min_latency_us: u64::MAX,
            max_latency_us: 0,
            sum_latency_us: 0,
        }
    }

    /// Record a latency sample
    pub fn record(&mut self, latency: Duration) {
        let latency_us = latency.as_micros() as u64;

        // Update min/max
        self.min_latency_us = self.min_latency_us.min(latency_us);
        self.max_latency_us = self.max_latency_us.max(latency_us);

        // Update sum and count
        self.sum_latency_us += latency_us;
        self.total_samples += 1;

        // Find bucket and increment
        let bucket_idx = self
            .buckets
            .iter()
            .position(|&b| latency_us < b)
            .unwrap_or(self.buckets.len());
        self.counts[bucket_idx] += 1;
    }

    /// Get average latency
    pub fn avg(&self) -> Duration {
        Duration::from_micros(
            self.sum_latency_us
                .checked_div(self.total_samples)
                .unwrap_or(0),
        )
    }

    /// Get minimum latency
    pub fn min(&self) -> Duration {
        if self.min_latency_us == u64::MAX {
            Duration::from_micros(0)
        } else {
            Duration::from_micros(self.min_latency_us)
        }
    }

    /// Get maximum latency
    pub fn max(&self) -> Duration {
        Duration::from_micros(self.max_latency_us)
    }

    /// Get percentile value (0.0 - 1.0)
    ///
    /// Example: percentile(0.99) returns the 99th percentile latency
    pub fn percentile(&self, p: f64) -> Duration {
        if self.total_samples == 0 {
            return Duration::from_micros(0);
        }

        let target_count = (self.total_samples as f64 * p) as u64;
        let mut cumulative = 0u64;

        for (idx, &count) in self.counts.iter().enumerate() {
            cumulative += count;
            if cumulative >= target_count {
                // Return upper bound of this bucket
                let latency_us = if idx < self.buckets.len() {
                    self.buckets[idx]
                } else {
                    self.max_latency_us
                };
                return Duration::from_micros(latency_us);
            }
        }

        Duration::from_micros(self.max_latency_us)
    }

    /// Get p50 (median)
    pub fn p50(&self) -> Duration {
        self.percentile(0.50)
    }

    /// Get p90
    pub fn p90(&self) -> Duration {
        self.percentile(0.90)
    }

    /// Get p95
    pub fn p95(&self) -> Duration {
        self.percentile(0.95)
    }

    /// Get p99
    pub fn p99(&self) -> Duration {
        self.percentile(0.99)
    }

    /// Get p999 (99.9th percentile)
    pub fn p999(&self) -> Duration {
        self.percentile(0.999)
    }

    /// Get total number of samples
    pub fn count(&self) -> u64 {
        self.total_samples
    }

    /// Generate a summary report
    pub fn summary(&self) -> String {
        format!(
            "Samples: {}, Min: {:?}, Max: {:?}, Avg: {:?}, P50: {:?}, P90: {:?}, P95: {:?}, P99: {:?}",
            self.total_samples,
            self.min(),
            self.max(),
            self.avg(),
            self.p50(),
            self.p90(),
            self.p95(),
            self.p99()
        )
    }
}

impl Default for LatencyHistogram {
    fn default() -> Self {
        Self::new()
    }
}

/// Performance profiler for tracking multiple operation types
#[derive(Debug, Clone, Default)]
pub struct PerformanceProfiler {
    /// Histograms for different operation types
    histograms: BTreeMap<String, LatencyHistogram>,
}

impl PerformanceProfiler {
    /// Create a new performance profiler
    pub fn new() -> Self {
        Self {
            histograms: BTreeMap::new(),
        }
    }

    /// Record a latency for an operation
    pub fn record(&mut self, operation: &str, latency: Duration) {
        self.histograms
            .entry(operation.to_string())
            .or_default()
            .record(latency);
    }

    /// Get histogram for a specific operation
    pub fn get_histogram(&self, operation: &str) -> Option<&LatencyHistogram> {
        self.histograms.get(operation)
    }

    /// Get all histograms
    pub fn histograms(&self) -> &BTreeMap<String, LatencyHistogram> {
        &self.histograms
    }

    /// Generate a comprehensive report
    pub fn report(&self) -> String {
        let mut report = String::from("=== Performance Profile ===\n\n");

        for (operation, histogram) in &self.histograms {
            report.push_str(&format!("Operation: {operation}\n"));
            report.push_str(&format!("  {}\n\n", histogram.summary()));
        }

        report
    }

    /// Reset all histograms
    pub fn reset(&mut self) {
        self.histograms.clear();
    }
}

/// Throughput tracker for measuring operations per second
#[derive(Debug, Clone)]
pub struct ThroughputTracker {
    /// Operation name
    operation: String,
    /// Total operations completed
    total_ops: u64,
    /// Total bytes processed (if applicable)
    total_bytes: u64,
    /// Start time
    start_time: std::time::Instant,
}

impl ThroughputTracker {
    /// Create a new throughput tracker
    pub fn new(operation: String) -> Self {
        Self {
            operation,
            total_ops: 0,
            total_bytes: 0,
            start_time: std::time::Instant::now(),
        }
    }

    /// Record an operation completion
    pub fn record_op(&mut self) {
        self.total_ops += 1;
    }

    /// Record bytes processed
    pub fn record_bytes(&mut self, bytes: u64) {
        self.total_bytes += bytes;
    }

    /// Get operations per second
    pub fn ops_per_second(&self) -> f64 {
        let elapsed = self.start_time.elapsed().as_secs_f64();
        if elapsed > 0.0 {
            self.total_ops as f64 / elapsed
        } else {
            0.0
        }
    }

    /// Get bytes per second
    pub fn bytes_per_second(&self) -> f64 {
        let elapsed = self.start_time.elapsed().as_secs_f64();
        if elapsed > 0.0 {
            self.total_bytes as f64 / elapsed
        } else {
            0.0
        }
    }

    /// Get megabytes per second
    pub fn megabytes_per_second(&self) -> f64 {
        self.bytes_per_second() / (1024.0 * 1024.0)
    }

    /// Get elapsed time
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Generate a summary report
    pub fn summary(&self) -> String {
        format!(
            "{}: {} ops in {:?} ({:.2} ops/s, {:.2} MB/s)",
            self.operation,
            self.total_ops,
            self.elapsed(),
            self.ops_per_second(),
            self.megabytes_per_second()
        )
    }
}

/// Batch profiler for analyzing batch operation efficiency
#[derive(Debug, Clone, Default)]
pub struct BatchProfiler {
    /// Total batch operations
    total_batches: u64,
    /// Total individual items in batches
    total_items: u64,
    /// Batch sizes histogram
    batch_sizes: LatencyHistogram,
    /// Batch latencies
    batch_latencies: LatencyHistogram,
}

impl BatchProfiler {
    /// Create a new batch profiler
    pub fn new() -> Self {
        Self {
            total_batches: 0,
            total_items: 0,
            batch_sizes: LatencyHistogram::with_buckets(vec![1, 10, 50, 100, 500, 1000]),
            batch_latencies: LatencyHistogram::new(),
        }
    }

    /// Record a batch operation
    pub fn record_batch(&mut self, batch_size: usize, latency: Duration) {
        self.total_batches += 1;
        self.total_items += batch_size as u64;

        // Record batch size as "latency" for histogram purposes
        self.batch_sizes
            .record(Duration::from_micros(batch_size as u64));
        self.batch_latencies.record(latency);
    }

    /// Get average batch size
    pub fn avg_batch_size(&self) -> f64 {
        if self.total_batches == 0 {
            0.0
        } else {
            self.total_items as f64 / self.total_batches as f64
        }
    }

    /// Get average latency per item
    pub fn avg_latency_per_item(&self) -> Duration {
        let total_latency_us = self.batch_latencies.sum_latency_us;
        Duration::from_micros(total_latency_us.checked_div(self.total_items).unwrap_or(0))
    }

    /// Generate a summary report
    pub fn summary(&self) -> String {
        format!(
            "Batches: {}, Items: {}, Avg Batch Size: {:.2}, Avg Latency: {:?}, Avg per Item: {:?}",
            self.total_batches,
            self.total_items,
            self.avg_batch_size(),
            self.batch_latencies.avg(),
            self.avg_latency_per_item()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_latency_histogram_basic() {
        let mut hist = LatencyHistogram::new();

        hist.record(Duration::from_micros(50));
        hist.record(Duration::from_micros(100));
        hist.record(Duration::from_micros(150));

        assert_eq!(hist.count(), 3);
        assert!(hist.min() <= Duration::from_micros(50));
        assert!(hist.max() >= Duration::from_micros(150));
    }

    #[test]
    fn test_latency_histogram_percentiles() {
        let mut hist = LatencyHistogram::new();

        // Use a wider range of values to ensure they fall into different buckets
        for i in 1..=100 {
            hist.record(Duration::from_micros(i * 100));
        }

        assert_eq!(hist.count(), 100);

        let p50 = hist.p50();
        let p90 = hist.p90();
        let p99 = hist.p99();

        // P90 should be >= P50, P99 should be >= P90
        assert!(p50 <= p90);
        assert!(p90 <= p99);
    }

    #[test]
    fn test_performance_profiler() {
        let mut profiler = PerformanceProfiler::new();

        profiler.record("put", Duration::from_micros(100));
        profiler.record("put", Duration::from_micros(150));
        profiler.record("get", Duration::from_micros(50));

        assert!(profiler.get_histogram("put").is_some());
        assert!(profiler.get_histogram("get").is_some());
        assert!(profiler.get_histogram("delete").is_none());

        let put_hist = profiler.get_histogram("put").unwrap();
        assert_eq!(put_hist.count(), 2);

        let report = profiler.report();
        assert!(report.contains("put"));
        assert!(report.contains("get"));
    }

    #[test]
    fn test_throughput_tracker() {
        let mut tracker = ThroughputTracker::new("test".to_string());

        for _ in 0..100 {
            tracker.record_op();
            tracker.record_bytes(1024);
        }

        assert_eq!(tracker.total_ops, 100);
        assert_eq!(tracker.total_bytes, 102400);
        assert!(tracker.ops_per_second() > 0.0);

        let summary = tracker.summary();
        assert!(summary.contains("test"));
        assert!(summary.contains("100 ops"));
    }

    #[test]
    fn test_batch_profiler() {
        let mut profiler = BatchProfiler::new();

        profiler.record_batch(10, Duration::from_micros(1000));
        profiler.record_batch(20, Duration::from_micros(2000));
        profiler.record_batch(30, Duration::from_micros(3000));

        assert_eq!(profiler.total_batches, 3);
        assert_eq!(profiler.total_items, 60);
        assert_eq!(profiler.avg_batch_size(), 20.0);

        let summary = profiler.summary();
        assert!(summary.contains("Batches: 3"));
        assert!(summary.contains("Items: 60"));
    }
}
