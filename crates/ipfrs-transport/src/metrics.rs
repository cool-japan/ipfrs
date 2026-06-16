//! Performance Metrics and Latency Distribution Tracking
//!
//! Provides utilities for tracking latency distributions, percentiles (p50, p99, p99.9),
//! and other performance metrics for transport operations.
//!
//! # Example
//!
//! ```
//! use ipfrs_transport::{LatencyTracker, Timer};
//! use std::time::Duration;
//!
//! // Create a latency tracker
//! let tracker = LatencyTracker::new();
//!
//! // Record some latencies
//! tracker.record(Duration::from_millis(10));
//! tracker.record(Duration::from_millis(20));
//! tracker.record(Duration::from_millis(15));
//!
//! // Get statistics
//! let stats = tracker.stats();
//! println!("p50 latency: {:?}", stats.p50);
//! println!("p99 latency: {:?}", stats.p99);
//! println!("Mean latency: {:?}", stats.mean);
//!
//! // Use Timer for automatic measurement
//! let timer = Timer::start();
//! // ... do some work ...
//! let elapsed = timer.elapsed();
//! tracker.record(elapsed);
//! ```

use parking_lot::RwLock;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Configuration for metrics collection
#[derive(Debug, Clone)]
pub struct MetricsConfig {
    /// Maximum number of samples to keep in the histogram
    pub max_samples: usize,
    /// Enable percentile tracking
    pub enable_percentiles: bool,
    /// Sample rate (1.0 = all samples, 0.1 = 10% of samples)
    pub sample_rate: f64,
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            max_samples: 10_000,
            enable_percentiles: true,
            sample_rate: 1.0,
        }
    }
}

/// Latency distribution tracker
pub struct LatencyTracker {
    config: MetricsConfig,
    samples: Arc<RwLock<Vec<Duration>>>,
    total_count: Arc<RwLock<u64>>,
    total_duration: Arc<RwLock<Duration>>,
}

impl LatencyTracker {
    /// Create a new latency tracker with default configuration
    pub fn new() -> Self {
        Self::with_config(MetricsConfig::default())
    }

    /// Create a new latency tracker with custom configuration
    pub fn with_config(config: MetricsConfig) -> Self {
        Self {
            config,
            samples: Arc::new(RwLock::new(Vec::new())),
            total_count: Arc::new(RwLock::new(0)),
            total_duration: Arc::new(RwLock::new(Duration::ZERO)),
        }
    }

    /// Record a latency sample
    pub fn record(&self, latency: Duration) {
        // Apply sampling
        if self.config.sample_rate < 1.0 {
            use rand::RngExt;
            let mut rng = rand::rng();
            if rng.random_range(0.0..1.0) > self.config.sample_rate {
                return;
            }
        }

        *self.total_count.write() += 1;
        *self.total_duration.write() += latency;

        if self.config.enable_percentiles {
            let mut samples = self.samples.write();
            samples.push(latency);

            // Limit sample size using reservoir sampling
            if samples.len() > self.config.max_samples {
                use rand::RngExt;
                let mut rng = rand::rng();
                let remove_idx = rng.random_range(0..samples.len());
                samples.swap_remove(remove_idx);
            }
        }
    }

    /// Get latency statistics
    pub fn stats(&self) -> LatencyStats {
        let samples = self.samples.read();
        let total_count = *self.total_count.read();
        let total_duration = *self.total_duration.read();

        if samples.is_empty() {
            return LatencyStats::default();
        }

        // Sort samples for percentile calculation
        let mut sorted = samples.clone();
        sorted.sort();

        let min = *sorted
            .first()
            .expect("sorted is non-empty: early return above");
        let max = *sorted
            .last()
            .expect("sorted is non-empty: early return above");
        let mean = if total_count > 0 {
            total_duration / total_count as u32
        } else {
            Duration::ZERO
        };

        // Calculate percentiles
        let p50 = percentile(&sorted, 50.0);
        let p90 = percentile(&sorted, 90.0);
        let p95 = percentile(&sorted, 95.0);
        let p99 = percentile(&sorted, 99.0);
        let p99_9 = percentile(&sorted, 99.9);

        LatencyStats {
            count: total_count,
            min,
            max,
            mean,
            p50,
            p90,
            p95,
            p99,
            p99_9,
        }
    }

    /// Reset all collected samples
    pub fn reset(&self) {
        self.samples.write().clear();
        *self.total_count.write() = 0;
        *self.total_duration.write() = Duration::ZERO;
    }
}

impl Default for LatencyTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Latency statistics
#[derive(Debug, Clone)]
pub struct LatencyStats {
    /// Total number of samples
    pub count: u64,
    /// Minimum latency
    pub min: Duration,
    /// Maximum latency
    pub max: Duration,
    /// Mean (average) latency
    pub mean: Duration,
    /// 50th percentile (median)
    pub p50: Duration,
    /// 90th percentile
    pub p90: Duration,
    /// 95th percentile
    pub p95: Duration,
    /// 99th percentile
    pub p99: Duration,
    /// 99.9th percentile
    pub p99_9: Duration,
}

impl Default for LatencyStats {
    fn default() -> Self {
        Self {
            count: 0,
            min: Duration::ZERO,
            max: Duration::ZERO,
            mean: Duration::ZERO,
            p50: Duration::ZERO,
            p90: Duration::ZERO,
            p95: Duration::ZERO,
            p99: Duration::ZERO,
            p99_9: Duration::ZERO,
        }
    }
}

impl std::fmt::Display for LatencyStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Latency Stats (n={}): min={:?}, max={:?}, mean={:?}, p50={:?}, p90={:?}, p95={:?}, p99={:?}, p99.9={:?}",
            self.count, self.min, self.max, self.mean, self.p50, self.p90, self.p95, self.p99, self.p99_9
        )
    }
}

/// Calculate percentile from sorted samples
fn percentile(sorted: &[Duration], p: f64) -> Duration {
    if sorted.is_empty() {
        return Duration::ZERO;
    }

    let index = (p / 100.0 * (sorted.len() - 1) as f64).round() as usize;
    sorted[index.min(sorted.len() - 1)]
}

/// Timer for measuring operation duration
pub struct Timer {
    start: Instant,
}

impl Timer {
    /// Start a new timer
    pub fn start() -> Self {
        Self {
            start: Instant::now(),
        }
    }

    /// Get elapsed time
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }

    /// Stop the timer and record to a tracker
    pub fn stop_and_record(self, tracker: &LatencyTracker) {
        tracker.record(self.elapsed());
    }
}

/// Memory usage tracker
#[derive(Debug, Clone, Default)]
pub struct MemoryStats {
    /// Current allocated bytes
    pub allocated_bytes: u64,
    /// Peak allocated bytes
    pub peak_bytes: u64,
    /// Total allocations
    pub total_allocations: u64,
    /// Total deallocations
    pub total_deallocations: u64,
}

/// Memory tracker
pub struct MemoryTracker {
    stats: Arc<RwLock<MemoryStats>>,
}

impl MemoryTracker {
    /// Create a new memory tracker
    pub fn new() -> Self {
        Self {
            stats: Arc::new(RwLock::new(MemoryStats::default())),
        }
    }

    /// Record an allocation
    pub fn record_allocation(&self, size: u64) {
        let mut stats = self.stats.write();
        stats.allocated_bytes += size;
        stats.total_allocations += 1;
        if stats.allocated_bytes > stats.peak_bytes {
            stats.peak_bytes = stats.allocated_bytes;
        }
    }

    /// Record a deallocation
    pub fn record_deallocation(&self, size: u64) {
        let mut stats = self.stats.write();
        stats.allocated_bytes = stats.allocated_bytes.saturating_sub(size);
        stats.total_deallocations += 1;
    }

    /// Get current memory statistics
    pub fn stats(&self) -> MemoryStats {
        self.stats.read().clone()
    }

    /// Reset statistics
    pub fn reset(&self) {
        *self.stats.write() = MemoryStats::default();
    }
}

impl Default for MemoryTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Throughput tracker
pub struct ThroughputTracker {
    bytes_transferred: Arc<RwLock<u64>>,
    start_time: Instant,
}

impl ThroughputTracker {
    /// Create a new throughput tracker
    pub fn new() -> Self {
        Self {
            bytes_transferred: Arc::new(RwLock::new(0)),
            start_time: Instant::now(),
        }
    }

    /// Record bytes transferred
    pub fn record_bytes(&self, bytes: u64) {
        *self.bytes_transferred.write() += bytes;
    }

    /// Get current throughput in bytes per second
    pub fn throughput_bps(&self) -> f64 {
        let bytes = *self.bytes_transferred.read();
        let elapsed = self.start_time.elapsed().as_secs_f64();
        if elapsed > 0.0 {
            bytes as f64 / elapsed
        } else {
            0.0
        }
    }

    /// Get total bytes transferred
    pub fn total_bytes(&self) -> u64 {
        *self.bytes_transferred.read()
    }

    /// Reset tracker
    pub fn reset(&self) {
        *self.bytes_transferred.write() = 0;
    }
}

impl Default for ThroughputTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_latency_tracker_basic() {
        let tracker = LatencyTracker::new();

        tracker.record(Duration::from_millis(10));
        tracker.record(Duration::from_millis(20));
        tracker.record(Duration::from_millis(30));

        let stats = tracker.stats();
        assert_eq!(stats.count, 3);
        assert_eq!(stats.min, Duration::from_millis(10));
        assert_eq!(stats.max, Duration::from_millis(30));
        assert_eq!(stats.mean, Duration::from_millis(20));
    }

    #[test]
    fn test_latency_percentiles() {
        let tracker = LatencyTracker::new();

        // Add 100 samples from 1ms to 100ms
        for i in 1..=100 {
            tracker.record(Duration::from_millis(i));
        }

        let stats = tracker.stats();
        assert_eq!(stats.count, 100);

        // p50 should be around 50ms
        assert!((stats.p50.as_millis() as i64 - 50).abs() <= 1);

        // p99 should be around 99ms
        assert!((stats.p99.as_millis() as i64 - 99).abs() <= 1);
    }

    #[test]
    fn test_latency_reset() {
        let tracker = LatencyTracker::new();

        tracker.record(Duration::from_millis(10));
        tracker.record(Duration::from_millis(20));

        let stats1 = tracker.stats();
        assert_eq!(stats1.count, 2);

        tracker.reset();

        let stats2 = tracker.stats();
        assert_eq!(stats2.count, 0);
    }

    #[test]
    fn test_timer() {
        let tracker = LatencyTracker::new();
        let timer = Timer::start();

        std::thread::sleep(Duration::from_millis(10));

        timer.stop_and_record(&tracker);

        let stats = tracker.stats();
        assert_eq!(stats.count, 1);
        assert!(stats.min >= Duration::from_millis(10));
    }

    #[test]
    fn test_memory_tracker() {
        let tracker = MemoryTracker::new();

        tracker.record_allocation(1000);
        tracker.record_allocation(2000);

        let stats = tracker.stats();
        assert_eq!(stats.allocated_bytes, 3000);
        assert_eq!(stats.peak_bytes, 3000);
        assert_eq!(stats.total_allocations, 2);

        tracker.record_deallocation(1000);

        let stats = tracker.stats();
        assert_eq!(stats.allocated_bytes, 2000);
        assert_eq!(stats.peak_bytes, 3000); // Peak remains
        assert_eq!(stats.total_deallocations, 1);
    }

    #[test]
    fn test_throughput_tracker() {
        let tracker = ThroughputTracker::new();

        tracker.record_bytes(1000);
        tracker.record_bytes(2000);

        let total = tracker.total_bytes();
        assert_eq!(total, 3000);

        // Ensure measurable elapsed time for throughput calculation
        std::thread::sleep(Duration::from_millis(1));

        let throughput = tracker.throughput_bps();
        assert!(throughput > 0.0);
    }

    #[test]
    fn test_percentile_calculation() {
        let samples = vec![
            Duration::from_millis(1),
            Duration::from_millis(2),
            Duration::from_millis(3),
            Duration::from_millis(4),
            Duration::from_millis(5),
        ];

        assert_eq!(percentile(&samples, 0.0), Duration::from_millis(1));
        assert_eq!(percentile(&samples, 50.0), Duration::from_millis(3));
        assert_eq!(percentile(&samples, 100.0), Duration::from_millis(5));
    }

    #[test]
    fn test_sampling() {
        let config = MetricsConfig {
            sample_rate: 0.5, // 50% sampling
            ..Default::default()
        };
        let tracker = LatencyTracker::with_config(config);

        // Record 1000 samples
        for _ in 0..1000 {
            tracker.record(Duration::from_millis(10));
        }

        let stats = tracker.stats();
        // With 50% sampling, we should have roughly 500 samples
        // Allow for variance (300-700 range)
        assert!(stats.count >= 300 && stats.count <= 700);
    }

    #[test]
    fn test_max_samples_limit() {
        let config = MetricsConfig {
            max_samples: 100,
            ..Default::default()
        };
        let tracker = LatencyTracker::with_config(config);

        // Record 1000 samples
        for i in 0..1000 {
            tracker.record(Duration::from_millis(i));
        }

        let samples_len = tracker.samples.read().len();
        assert_eq!(samples_len, 100); // Should be capped at max_samples
    }

    #[test]
    fn test_latency_stats_display() {
        let stats = LatencyStats {
            count: 100,
            min: Duration::from_millis(1),
            max: Duration::from_millis(100),
            mean: Duration::from_millis(50),
            p50: Duration::from_millis(50),
            p90: Duration::from_millis(90),
            p95: Duration::from_millis(95),
            p99: Duration::from_millis(99),
            p99_9: Duration::from_millis(100),
        };

        let display = format!("{}", stats);
        assert!(display.contains("n=100"));
        assert!(display.contains("p50"));
        assert!(display.contains("p99"));
    }
}
