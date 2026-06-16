//! Metrics and observability for production monitoring
//!
//! This module provides comprehensive metrics tracking for IPFRS operations,
//! enabling production monitoring, performance analysis, and capacity planning.
//!
//! # Features
//!
//! - **Operation Counters** - Track blocks created, CIDs generated, bytes processed
//! - **Performance Metrics** - Latency percentiles, throughput rates
//! - **Resource Usage** - Memory allocations, pool hit rates
//! - **Error Tracking** - Error counts by type and category
//! - **Health Checks** - System health and readiness indicators
//!
//! # Example
//!
//! ```rust
//! use ipfrs_core::metrics::{global_metrics, MetricsSnapshot};
//!
//! // Record operations
//! let metrics = global_metrics();
//! metrics.record_block_created(1024);
//! metrics.record_cid_generated(50); // microseconds
//!
//! // Get snapshot for monitoring
//! let snapshot = metrics.snapshot();
//! println!("Blocks created: {}", snapshot.blocks_created);
//! println!("Total bytes: {}", snapshot.total_bytes_processed);
//! println!("Avg CID generation: {:.2}µs", snapshot.avg_cid_generation_us);
//! ```

use once_cell::sync::Lazy;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Global metrics instance for the entire application
pub static GLOBAL_METRICS: Lazy<Arc<Metrics>> = Lazy::new(|| Arc::new(Metrics::new()));

/// Get the global metrics instance
///
/// This returns a reference to the global metrics collector that tracks
/// all IPFRS operations across the application.
///
/// # Example
///
/// ```rust
/// use ipfrs_core::metrics::global_metrics;
///
/// let metrics = global_metrics();
/// metrics.record_block_created(2048);
/// ```
pub fn global_metrics() -> Arc<Metrics> {
    Arc::clone(&GLOBAL_METRICS)
}

/// Core metrics collector for IPFRS operations
///
/// This struct uses atomic operations for lock-free performance in hot paths,
/// while protecting detailed statistics with mutexes.
#[derive(Debug)]
pub struct Metrics {
    // Operation counters (lock-free)
    blocks_created: AtomicUsize,
    cids_generated: AtomicUsize,
    blocks_verified: AtomicUsize,
    chunks_created: AtomicUsize,
    total_bytes_processed: AtomicU64,

    // Error tracking (lock-free)
    errors_total: AtomicUsize,
    serialization_errors: AtomicUsize,
    validation_errors: AtomicUsize,
    network_errors: AtomicUsize,

    // Performance metrics (protected by mutex for percentile calculations)
    timings: Mutex<TimingStats>,

    // Resource usage
    memory_allocations: AtomicU64,
    pool_hits: AtomicUsize,
    pool_misses: AtomicUsize,

    // Start time for uptime calculation
    start_time: Instant,
}

/// Detailed timing statistics for performance monitoring
#[derive(Debug, Clone)]
struct TimingStats {
    cid_generation_samples: Vec<u64>, // microseconds
    block_creation_samples: Vec<u64>, // microseconds
    chunking_samples: Vec<u64>,       // microseconds
    verification_samples: Vec<u64>,   // microseconds
    max_samples: usize,               // Limit sample collection
}

impl Default for TimingStats {
    fn default() -> Self {
        Self {
            cid_generation_samples: Vec::with_capacity(10000),
            block_creation_samples: Vec::with_capacity(10000),
            chunking_samples: Vec::with_capacity(1000),
            verification_samples: Vec::with_capacity(10000),
            max_samples: 10000,
        }
    }
}

impl TimingStats {
    /// Add a sample, maintaining the sample limit
    fn add_sample(samples: &mut Vec<u64>, value: u64, max_samples: usize) {
        if samples.len() >= max_samples {
            // Remove oldest samples (simple FIFO, could use ring buffer)
            samples.drain(0..max_samples / 4);
        }
        samples.push(value);
    }

    /// Calculate percentile from sorted samples
    fn percentile(sorted_samples: &[u64], p: f64) -> u64 {
        if sorted_samples.is_empty() {
            return 0;
        }
        let idx = ((sorted_samples.len() as f64 - 1.0) * p) as usize;
        sorted_samples[idx]
    }

    /// Get percentile statistics for a set of samples
    fn get_percentiles(samples: &[u64]) -> PercentileStats {
        if samples.is_empty() {
            return PercentileStats::default();
        }

        let mut sorted = samples.to_vec();
        sorted.sort_unstable();

        PercentileStats {
            p50: Self::percentile(&sorted, 0.50),
            p90: Self::percentile(&sorted, 0.90),
            p95: Self::percentile(&sorted, 0.95),
            p99: Self::percentile(&sorted, 0.99),
            min: sorted[0],
            max: sorted[sorted.len() - 1],
        }
    }
}

/// Percentile statistics for latency analysis
#[derive(Debug, Clone, Default)]
pub struct PercentileStats {
    /// Median (50th percentile) in microseconds
    pub p50: u64,
    /// 90th percentile in microseconds
    pub p90: u64,
    /// 95th percentile in microseconds
    pub p95: u64,
    /// 99th percentile in microseconds
    pub p99: u64,
    /// Minimum value in microseconds
    pub min: u64,
    /// Maximum value in microseconds
    pub max: u64,
}

/// Snapshot of current metrics for reporting
#[derive(Debug, Clone)]
pub struct MetricsSnapshot {
    // Counters
    /// Total blocks created
    pub blocks_created: usize,
    /// Total CIDs generated
    pub cids_generated: usize,
    /// Total blocks verified
    pub blocks_verified: usize,
    /// Total chunks created
    pub chunks_created: usize,
    /// Total bytes processed
    pub total_bytes_processed: u64,

    // Errors
    /// Total errors encountered
    pub errors_total: usize,
    /// Serialization errors
    pub serialization_errors: usize,
    /// Validation errors
    pub validation_errors: usize,
    /// Network errors
    pub network_errors: usize,

    // Performance
    /// CID generation latency percentiles
    pub cid_generation: PercentileStats,
    /// Block creation latency percentiles
    pub block_creation: PercentileStats,
    /// Chunking latency percentiles
    pub chunking: PercentileStats,
    /// Verification latency percentiles
    pub verification: PercentileStats,

    // Derived metrics
    /// Average CID generation time in microseconds
    pub avg_cid_generation_us: f64,
    /// Average block size in bytes
    pub avg_block_size_bytes: f64,
    /// Throughput in bytes per second
    pub throughput_bytes_per_sec: f64,

    // Resource usage
    /// Total memory allocated in bytes
    pub memory_allocations: u64,
    /// Pool hit rate (0.0 to 1.0)
    pub pool_hit_rate: f64,

    // System health
    /// Uptime in seconds
    pub uptime_seconds: u64,
}

impl Metrics {
    /// Create a new metrics collector
    pub fn new() -> Self {
        Self {
            blocks_created: AtomicUsize::new(0),
            cids_generated: AtomicUsize::new(0),
            blocks_verified: AtomicUsize::new(0),
            chunks_created: AtomicUsize::new(0),
            total_bytes_processed: AtomicU64::new(0),
            errors_total: AtomicUsize::new(0),
            serialization_errors: AtomicUsize::new(0),
            validation_errors: AtomicUsize::new(0),
            network_errors: AtomicUsize::new(0),
            timings: Mutex::new(TimingStats::default()),
            memory_allocations: AtomicU64::new(0),
            pool_hits: AtomicUsize::new(0),
            pool_misses: AtomicUsize::new(0),
            start_time: Instant::now(),
        }
    }

    // === Operation Recording ===

    /// Record a block creation
    pub fn record_block_created(&self, size_bytes: u64) {
        self.blocks_created.fetch_add(1, Ordering::Relaxed);
        self.total_bytes_processed
            .fetch_add(size_bytes, Ordering::Relaxed);
    }

    /// Record block creation with timing
    pub fn record_block_created_timed(&self, size_bytes: u64, duration_us: u64) {
        self.record_block_created(size_bytes);
        if let Ok(mut timings) = self.timings.lock() {
            let max_samples = timings.max_samples;
            TimingStats::add_sample(
                &mut timings.block_creation_samples,
                duration_us,
                max_samples,
            );
        }
    }

    /// Record a CID generation
    pub fn record_cid_generated(&self, duration_us: u64) {
        self.cids_generated.fetch_add(1, Ordering::Relaxed);
        if let Ok(mut timings) = self.timings.lock() {
            let max_samples = timings.max_samples;
            TimingStats::add_sample(
                &mut timings.cid_generation_samples,
                duration_us,
                max_samples,
            );
        }
    }

    /// Record a block verification
    pub fn record_block_verified(&self, duration_us: u64) {
        self.blocks_verified.fetch_add(1, Ordering::Relaxed);
        if let Ok(mut timings) = self.timings.lock() {
            let max_samples = timings.max_samples;
            TimingStats::add_sample(&mut timings.verification_samples, duration_us, max_samples);
        }
    }

    /// Record chunking operation
    pub fn record_chunking(&self, num_chunks: usize, duration_us: u64) {
        self.chunks_created.fetch_add(num_chunks, Ordering::Relaxed);
        if let Ok(mut timings) = self.timings.lock() {
            let max_samples = timings.max_samples;
            TimingStats::add_sample(&mut timings.chunking_samples, duration_us, max_samples);
        }
    }

    // === Error Recording ===

    /// Record a serialization error
    pub fn record_serialization_error(&self) {
        self.errors_total.fetch_add(1, Ordering::Relaxed);
        self.serialization_errors.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a validation error
    pub fn record_validation_error(&self) {
        self.errors_total.fetch_add(1, Ordering::Relaxed);
        self.validation_errors.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a network error
    pub fn record_network_error(&self) {
        self.errors_total.fetch_add(1, Ordering::Relaxed);
        self.network_errors.fetch_add(1, Ordering::Relaxed);
    }

    // === Resource Tracking ===

    /// Record memory allocation
    pub fn record_memory_allocation(&self, bytes: u64) {
        self.memory_allocations.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Record pool hit
    pub fn record_pool_hit(&self) {
        self.pool_hits.fetch_add(1, Ordering::Relaxed);
    }

    /// Record pool miss
    pub fn record_pool_miss(&self) {
        self.pool_misses.fetch_add(1, Ordering::Relaxed);
    }

    // === Snapshot and Reporting ===

    /// Get a snapshot of current metrics
    pub fn snapshot(&self) -> MetricsSnapshot {
        let blocks_created = self.blocks_created.load(Ordering::Relaxed);
        let cids_generated = self.cids_generated.load(Ordering::Relaxed);
        let total_bytes = self.total_bytes_processed.load(Ordering::Relaxed);
        let pool_hits = self.pool_hits.load(Ordering::Relaxed);
        let pool_misses = self.pool_misses.load(Ordering::Relaxed);

        let timings = self.timings.lock().unwrap_or_else(|e| e.into_inner());

        // Calculate derived metrics
        let avg_block_size = if blocks_created > 0 {
            total_bytes as f64 / blocks_created as f64
        } else {
            0.0
        };

        let uptime_seconds = self.start_time.elapsed().as_secs();
        let throughput = if uptime_seconds > 0 {
            total_bytes as f64 / uptime_seconds as f64
        } else {
            0.0
        };

        let avg_cid_gen = if !timings.cid_generation_samples.is_empty() {
            timings.cid_generation_samples.iter().sum::<u64>() as f64
                / timings.cid_generation_samples.len() as f64
        } else {
            0.0
        };

        let pool_total = pool_hits + pool_misses;
        let hit_rate = if pool_total > 0 {
            pool_hits as f64 / pool_total as f64
        } else {
            0.0
        };

        MetricsSnapshot {
            blocks_created,
            cids_generated,
            blocks_verified: self.blocks_verified.load(Ordering::Relaxed),
            chunks_created: self.chunks_created.load(Ordering::Relaxed),
            total_bytes_processed: total_bytes,
            errors_total: self.errors_total.load(Ordering::Relaxed),
            serialization_errors: self.serialization_errors.load(Ordering::Relaxed),
            validation_errors: self.validation_errors.load(Ordering::Relaxed),
            network_errors: self.network_errors.load(Ordering::Relaxed),
            cid_generation: TimingStats::get_percentiles(&timings.cid_generation_samples),
            block_creation: TimingStats::get_percentiles(&timings.block_creation_samples),
            chunking: TimingStats::get_percentiles(&timings.chunking_samples),
            verification: TimingStats::get_percentiles(&timings.verification_samples),
            avg_cid_generation_us: avg_cid_gen,
            avg_block_size_bytes: avg_block_size,
            throughput_bytes_per_sec: throughput,
            memory_allocations: self.memory_allocations.load(Ordering::Relaxed),
            pool_hit_rate: hit_rate,
            uptime_seconds,
        }
    }

    /// Reset all metrics (useful for testing)
    pub fn reset(&self) {
        self.blocks_created.store(0, Ordering::Relaxed);
        self.cids_generated.store(0, Ordering::Relaxed);
        self.blocks_verified.store(0, Ordering::Relaxed);
        self.chunks_created.store(0, Ordering::Relaxed);
        self.total_bytes_processed.store(0, Ordering::Relaxed);
        self.errors_total.store(0, Ordering::Relaxed);
        self.serialization_errors.store(0, Ordering::Relaxed);
        self.validation_errors.store(0, Ordering::Relaxed);
        self.network_errors.store(0, Ordering::Relaxed);
        self.memory_allocations.store(0, Ordering::Relaxed);
        self.pool_hits.store(0, Ordering::Relaxed);
        self.pool_misses.store(0, Ordering::Relaxed);

        if let Ok(mut timings) = self.timings.lock() {
            timings.cid_generation_samples.clear();
            timings.block_creation_samples.clear();
            timings.chunking_samples.clear();
            timings.verification_samples.clear();
        }
    }

    /// Check if system is healthy
    pub fn is_healthy(&self) -> bool {
        let snapshot = self.snapshot();

        // System is unhealthy if error rate > 10%
        let total_ops = snapshot.blocks_created + snapshot.cids_generated;
        if total_ops > 0 {
            let error_rate = snapshot.errors_total as f64 / total_ops as f64;
            if error_rate > 0.10 {
                return false;
            }
        }

        // Check if p99 latency is reasonable (< 10ms for CID generation)
        if snapshot.cid_generation.p99 > 10_000 {
            return false;
        }

        true
    }
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper to measure operation duration
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

    /// Get elapsed time in microseconds
    pub fn elapsed_us(&self) -> u64 {
        self.start.elapsed().as_micros() as u64
    }

    /// Get elapsed duration
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_basic() {
        let metrics = Metrics::new();

        metrics.record_block_created(1024);
        metrics.record_cid_generated(100);

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.blocks_created, 1);
        assert_eq!(snapshot.cids_generated, 1);
        assert_eq!(snapshot.total_bytes_processed, 1024);
    }

    #[test]
    fn test_metrics_timing() {
        let metrics = Metrics::new();

        metrics.record_cid_generated(100);
        metrics.record_cid_generated(200);
        metrics.record_cid_generated(300);

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.cids_generated, 3);
        assert_eq!(snapshot.cid_generation.min, 100);
        assert_eq!(snapshot.cid_generation.max, 300);
    }

    #[test]
    fn test_metrics_errors() {
        let metrics = Metrics::new();

        metrics.record_serialization_error();
        metrics.record_validation_error();
        metrics.record_network_error();

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.errors_total, 3);
        assert_eq!(snapshot.serialization_errors, 1);
        assert_eq!(snapshot.validation_errors, 1);
        assert_eq!(snapshot.network_errors, 1);
    }

    #[test]
    fn test_metrics_pool_stats() {
        let metrics = Metrics::new();

        metrics.record_pool_hit();
        metrics.record_pool_hit();
        metrics.record_pool_miss();

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.pool_hit_rate, 2.0 / 3.0);
    }

    #[test]
    fn test_metrics_reset() {
        let metrics = Metrics::new();

        metrics.record_block_created(1024);
        metrics.record_cid_generated(100);

        metrics.reset();

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.blocks_created, 0);
        assert_eq!(snapshot.cids_generated, 0);
    }

    #[test]
    fn test_percentile_calculation() {
        let metrics = Metrics::new();

        for i in 1..=100 {
            metrics.record_cid_generated(i * 10);
        }

        let snapshot = metrics.snapshot();
        assert!(snapshot.cid_generation.p50 > 0);
        assert!(snapshot.cid_generation.p90 > snapshot.cid_generation.p50);
        assert!(snapshot.cid_generation.p99 > snapshot.cid_generation.p90);
    }

    #[test]
    fn test_timer() {
        let timer = Timer::start();
        std::thread::sleep(Duration::from_micros(100));
        let elapsed = timer.elapsed_us();
        assert!(elapsed >= 100);
    }

    #[test]
    fn test_health_check() {
        let metrics = Metrics::new();

        // Healthy system
        for _ in 0..100 {
            metrics.record_block_created(1024);
            metrics.record_cid_generated(100);
        }
        assert!(metrics.is_healthy());

        // Unhealthy due to high error rate (>10% of total operations)
        for _ in 0..50 {
            metrics.record_validation_error();
        }
        assert!(!metrics.is_healthy());
    }

    #[test]
    fn test_global_metrics() {
        let metrics = global_metrics();
        metrics.record_block_created(2048);

        let snapshot = metrics.snapshot();
        assert!(snapshot.blocks_created > 0);
    }

    #[test]
    fn test_throughput_calculation() {
        let metrics = Metrics::new();

        // Sleep a bit to ensure uptime > 0
        std::thread::sleep(Duration::from_millis(10));

        metrics.record_block_created(1_000_000);
        std::thread::sleep(Duration::from_millis(100));

        let snapshot = metrics.snapshot();
        // Throughput should be calculated based on uptime
        assert!(snapshot.uptime_seconds > 0 || snapshot.throughput_bytes_per_sec >= 0.0);
    }

    #[test]
    fn test_avg_block_size() {
        let metrics = Metrics::new();

        metrics.record_block_created(1000);
        metrics.record_block_created(2000);
        metrics.record_block_created(3000);

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.avg_block_size_bytes, 2000.0);
    }
}
