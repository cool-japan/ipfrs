//! Index statistics and monitoring
//!
//! This module provides comprehensive statistics collection and monitoring
//! for vector indexes, enabling performance analysis and optimization.

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Index statistics collector
#[derive(Default)]
pub struct IndexStats {
    /// Total number of inserts
    insert_count: AtomicU64,
    /// Total number of deletes
    delete_count: AtomicU64,
    /// Total number of searches
    search_count: AtomicU64,
    /// Search latency histogram
    search_latencies: Arc<RwLock<LatencyHistogram>>,
    /// Insert latency histogram
    insert_latencies: Arc<RwLock<LatencyHistogram>>,
    /// Cache hit count
    cache_hits: AtomicU64,
    /// Cache miss count
    cache_misses: AtomicU64,
    /// Timestamp when stats started collecting
    start_time: u64,
    /// Recent query log for analysis
    recent_queries: Arc<RwLock<VecDeque<QueryRecord>>>,
    /// Maximum recent queries to keep
    max_recent_queries: usize,
}

impl IndexStats {
    /// Create a new stats collector
    pub fn new() -> Self {
        Self {
            insert_count: AtomicU64::new(0),
            delete_count: AtomicU64::new(0),
            search_count: AtomicU64::new(0),
            search_latencies: Arc::new(RwLock::new(LatencyHistogram::new())),
            insert_latencies: Arc::new(RwLock::new(LatencyHistogram::new())),
            cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
            start_time: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            recent_queries: Arc::new(RwLock::new(VecDeque::new())),
            max_recent_queries: 1000,
        }
    }

    /// Record an insert operation
    pub fn record_insert(&self, duration: Duration) {
        self.insert_count.fetch_add(1, Ordering::Relaxed);
        self.insert_latencies
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .record(duration.as_micros() as u64);
    }

    /// Record a delete operation
    pub fn record_delete(&self) {
        self.delete_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a search operation
    pub fn record_search(&self, duration: Duration, k: usize, result_count: usize) {
        self.search_count.fetch_add(1, Ordering::Relaxed);
        self.search_latencies
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .record(duration.as_micros() as u64);

        // Record query details
        let mut queries = self
            .recent_queries
            .write()
            .unwrap_or_else(|e| e.into_inner());
        if queries.len() >= self.max_recent_queries {
            queries.pop_front();
        }
        queries.push_back(QueryRecord {
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            latency_us: duration.as_micros() as u64,
            k,
            result_count,
        });
    }

    /// Record a cache hit
    pub fn record_cache_hit(&self) {
        self.cache_hits.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a cache miss
    pub fn record_cache_miss(&self) {
        self.cache_misses.fetch_add(1, Ordering::Relaxed);
    }

    /// Get a snapshot of current statistics
    pub fn snapshot(&self) -> StatsSnapshot {
        let search_latencies = self
            .search_latencies
            .read()
            .unwrap_or_else(|e| e.into_inner());
        let insert_latencies = self
            .insert_latencies
            .read()
            .unwrap_or_else(|e| e.into_inner());

        let cache_hits = self.cache_hits.load(Ordering::Relaxed);
        let cache_misses = self.cache_misses.load(Ordering::Relaxed);
        let total_cache = cache_hits + cache_misses;

        StatsSnapshot {
            insert_count: self.insert_count.load(Ordering::Relaxed),
            delete_count: self.delete_count.load(Ordering::Relaxed),
            search_count: self.search_count.load(Ordering::Relaxed),
            search_latency_p50: search_latencies.percentile(50),
            search_latency_p90: search_latencies.percentile(90),
            search_latency_p99: search_latencies.percentile(99),
            search_latency_avg: search_latencies.average(),
            insert_latency_avg: insert_latencies.average(),
            cache_hit_rate: if total_cache > 0 {
                cache_hits as f64 / total_cache as f64
            } else {
                0.0
            },
            uptime_seconds: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs()
                - self.start_time,
        }
    }

    /// Reset all statistics
    pub fn reset(&self) {
        self.insert_count.store(0, Ordering::Relaxed);
        self.delete_count.store(0, Ordering::Relaxed);
        self.search_count.store(0, Ordering::Relaxed);
        self.search_latencies
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .reset();
        self.insert_latencies
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .reset();
        self.cache_hits.store(0, Ordering::Relaxed);
        self.cache_misses.store(0, Ordering::Relaxed);
        self.recent_queries
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
    }

    /// Get recent query records
    pub fn recent_queries(&self) -> Vec<QueryRecord> {
        self.recent_queries
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .cloned()
            .collect()
    }

    /// Calculate queries per second (QPS)
    pub fn qps(&self) -> f64 {
        let uptime = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
            - self.start_time;

        if uptime > 0 {
            self.search_count.load(Ordering::Relaxed) as f64 / uptime as f64
        } else {
            0.0
        }
    }
}

/// Statistics snapshot at a point in time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatsSnapshot {
    /// Total insert operations
    pub insert_count: u64,
    /// Total delete operations
    pub delete_count: u64,
    /// Total search operations
    pub search_count: u64,
    /// Search latency P50 (microseconds)
    pub search_latency_p50: u64,
    /// Search latency P90 (microseconds)
    pub search_latency_p90: u64,
    /// Search latency P99 (microseconds)
    pub search_latency_p99: u64,
    /// Average search latency (microseconds)
    pub search_latency_avg: u64,
    /// Average insert latency (microseconds)
    pub insert_latency_avg: u64,
    /// Cache hit rate (0.0 to 1.0)
    pub cache_hit_rate: f64,
    /// Uptime in seconds
    pub uptime_seconds: u64,
}

impl StatsSnapshot {
    /// Format latency as human-readable string
    pub fn format_latency(us: u64) -> String {
        if us < 1000 {
            format!("{}µs", us)
        } else if us < 1_000_000 {
            format!("{:.2}ms", us as f64 / 1000.0)
        } else {
            format!("{:.2}s", us as f64 / 1_000_000.0)
        }
    }

    /// Get a summary string
    pub fn summary(&self) -> String {
        format!(
            "Searches: {} (P50: {}, P99: {}), Inserts: {}, Cache: {:.1}%",
            self.search_count,
            Self::format_latency(self.search_latency_p50),
            Self::format_latency(self.search_latency_p99),
            self.insert_count,
            self.cache_hit_rate * 100.0
        )
    }
}

/// Query record for analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryRecord {
    /// Unix timestamp
    pub timestamp: u64,
    /// Latency in microseconds
    pub latency_us: u64,
    /// K parameter (number of results requested)
    pub k: usize,
    /// Actual result count
    pub result_count: usize,
}

/// Latency histogram for percentile calculations
#[derive(Default)]
pub struct LatencyHistogram {
    /// Sorted latencies (in microseconds)
    values: Vec<u64>,
    /// Sum for average calculation
    sum: u64,
    /// Count
    count: u64,
}

impl LatencyHistogram {
    /// Create a new histogram
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a latency value
    pub fn record(&mut self, value_us: u64) {
        // Keep sorted for percentile calculation
        let pos = self.values.binary_search(&value_us).unwrap_or_else(|i| i);
        self.values.insert(pos, value_us);

        self.sum += value_us;
        self.count += 1;

        // Keep bounded to avoid memory growth
        if self.values.len() > 10000 {
            // Remove oldest values (this is approximate)
            self.values.drain(0..1000);
        }
    }

    /// Get percentile value
    pub fn percentile(&self, p: u8) -> u64 {
        if self.values.is_empty() {
            return 0;
        }

        let idx = ((p as usize) * self.values.len() / 100).min(self.values.len() - 1);
        self.values[idx]
    }

    /// Get average value
    pub fn average(&self) -> u64 {
        if self.count == 0 {
            return 0;
        }
        self.sum / self.count
    }

    /// Reset the histogram
    pub fn reset(&mut self) {
        self.values.clear();
        self.sum = 0;
        self.count = 0;
    }

    /// Get total count
    pub fn count(&self) -> u64 {
        self.count
    }
}

/// Index health metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexHealth {
    /// Index size (number of vectors)
    pub size: usize,
    /// Estimated memory usage (bytes)
    pub memory_bytes: usize,
    /// Vector dimension
    pub dimension: usize,
    /// Average connectivity (HNSW specific)
    pub avg_connectivity: Option<f32>,
    /// Search recall estimate (if available)
    pub recall_estimate: Option<f32>,
    /// Overall health score (0.0 to 1.0)
    pub health_score: f32,
    /// Issues detected
    pub issues: Vec<HealthIssue>,
}

/// Health issue description
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthIssue {
    /// Issue severity (0 = info, 1 = warning, 2 = error)
    pub severity: u8,
    /// Issue description
    pub message: String,
    /// Recommendation
    pub recommendation: String,
}

impl IndexHealth {
    /// Create health metrics for an index
    pub fn analyze(size: usize, dimension: usize, stats: Option<&StatsSnapshot>) -> Self {
        let mut issues = Vec::new();
        let mut health_score = 1.0;

        // Estimate memory usage (HNSW overhead ~= 4 * dimension * M bytes per vector)
        let memory_bytes = size * dimension * 4 + size * dimension * 4 * 16;

        // Check for potential issues
        if size == 0 {
            issues.push(HealthIssue {
                severity: 0,
                message: "Index is empty".to_string(),
                recommendation: "Add vectors to enable semantic search".to_string(),
            });
            health_score *= 0.9;
        }

        if let Some(s) = stats {
            // Check latency
            if s.search_latency_p99 > 100_000 {
                // > 100ms
                issues.push(HealthIssue {
                    severity: 2,
                    message: format!(
                        "High P99 search latency: {}",
                        StatsSnapshot::format_latency(s.search_latency_p99)
                    ),
                    recommendation: "Consider reducing ef_search or optimizing index parameters"
                        .to_string(),
                });
                health_score *= 0.7;
            } else if s.search_latency_p99 > 10_000 {
                // > 10ms
                issues.push(HealthIssue {
                    severity: 1,
                    message: format!(
                        "Elevated P99 search latency: {}",
                        StatsSnapshot::format_latency(s.search_latency_p99)
                    ),
                    recommendation: "Monitor latency trends".to_string(),
                });
                health_score *= 0.9;
            }

            // Check cache hit rate
            if s.cache_hit_rate < 0.5 && s.search_count > 100 {
                issues.push(HealthIssue {
                    severity: 1,
                    message: format!("Low cache hit rate: {:.1}%", s.cache_hit_rate * 100.0),
                    recommendation: "Consider increasing cache size".to_string(),
                });
                health_score *= 0.95;
            }
        }

        // Check size for performance
        if size > 1_000_000 {
            issues.push(HealthIssue {
                severity: 1,
                message: format!("Large index size: {} vectors", size),
                recommendation:
                    "Consider using DiskANN or quantization for better memory efficiency"
                        .to_string(),
            });
        }

        Self {
            size,
            memory_bytes,
            dimension,
            avg_connectivity: None,
            recall_estimate: None,
            health_score,
            issues,
        }
    }
}

/// Performance timer for measuring operation latencies
pub struct PerfTimer {
    start: Instant,
}

impl PerfTimer {
    /// Start a new timer
    pub fn start() -> Self {
        Self {
            start: Instant::now(),
        }
    }

    /// Get elapsed duration
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }

    /// Stop and return duration
    pub fn stop(self) -> Duration {
        self.start.elapsed()
    }
}

/// Memory usage tracker
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryUsage {
    /// Vector data memory (bytes)
    pub vectors_bytes: usize,
    /// Index structure memory (bytes)
    pub index_bytes: usize,
    /// Metadata memory (bytes)
    pub metadata_bytes: usize,
    /// Cache memory (bytes)
    pub cache_bytes: usize,
    /// Total memory (bytes)
    pub total_bytes: usize,
}

impl MemoryUsage {
    /// Estimate memory usage
    pub fn estimate(
        num_vectors: usize,
        dimension: usize,
        metadata_count: usize,
        cache_size: usize,
    ) -> Self {
        // Vector storage: num_vectors * dimension * 4 bytes (f32)
        let vectors_bytes = num_vectors * dimension * 4;

        // HNSW index overhead: approximately M * num_vectors * 4 * 2 bytes for graph
        // Assuming M = 16
        let index_bytes = 16 * num_vectors * 4 * 2;

        // Metadata: rough estimate of 200 bytes per entry
        let metadata_bytes = metadata_count * 200;

        // Cache: cached vectors + overhead
        let cache_bytes = cache_size * dimension * 4 * 2;

        let total_bytes = vectors_bytes + index_bytes + metadata_bytes + cache_bytes;

        Self {
            vectors_bytes,
            index_bytes,
            metadata_bytes,
            cache_bytes,
            total_bytes,
        }
    }

    /// Format as human-readable string
    pub fn format_bytes(bytes: usize) -> String {
        if bytes < 1024 {
            format!("{} B", bytes)
        } else if bytes < 1024 * 1024 {
            format!("{:.2} KB", bytes as f64 / 1024.0)
        } else if bytes < 1024 * 1024 * 1024 {
            format!("{:.2} MB", bytes as f64 / (1024.0 * 1024.0))
        } else {
            format!("{:.2} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
        }
    }

    /// Get formatted summary
    pub fn summary(&self) -> String {
        format!(
            "Total: {} (Vectors: {}, Index: {}, Metadata: {}, Cache: {})",
            Self::format_bytes(self.total_bytes),
            Self::format_bytes(self.vectors_bytes),
            Self::format_bytes(self.index_bytes),
            Self::format_bytes(self.metadata_bytes),
            Self::format_bytes(self.cache_bytes),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stats_recording() {
        let stats = IndexStats::new();

        // Record some operations
        stats.record_insert(Duration::from_micros(100));
        stats.record_insert(Duration::from_micros(200));
        stats.record_search(Duration::from_micros(50), 10, 10);
        stats.record_search(Duration::from_micros(150), 10, 8);
        stats.record_cache_hit();
        stats.record_cache_miss();

        let snapshot = stats.snapshot();

        assert_eq!(snapshot.insert_count, 2);
        assert_eq!(snapshot.search_count, 2);
        assert!(snapshot.cache_hit_rate > 0.4 && snapshot.cache_hit_rate < 0.6);
    }

    #[test]
    fn test_latency_histogram() {
        let mut histogram = LatencyHistogram::new();

        for i in 1..=100 {
            histogram.record(i);
        }

        assert_eq!(histogram.count(), 100);
        // Percentile 50 should be around 50-51 (0-indexed array, so idx 50 = value 51)
        let p50 = histogram.percentile(50);
        assert!((50..=52).contains(&p50), "P50 was {}", p50);
        assert!(histogram.percentile(99) >= 99);
        // Average of 1..=100 is 50.5, rounded to 50
        assert!(histogram.average() >= 50 && histogram.average() <= 51);
    }

    #[test]
    fn test_index_health() {
        let health = IndexHealth::analyze(1000, 768, None);

        assert!(health.health_score > 0.0);
        assert_eq!(health.size, 1000);
        assert_eq!(health.dimension, 768);
    }

    #[test]
    fn test_memory_usage() {
        let usage = MemoryUsage::estimate(10000, 768, 10000, 1000);

        // Should be in MB range for this size
        assert!(usage.total_bytes > 1024 * 1024);
        assert!(usage.vectors_bytes > 0);
    }

    #[test]
    fn test_perf_timer() {
        let timer = PerfTimer::start();
        std::thread::sleep(Duration::from_millis(10));
        let elapsed = timer.stop();

        assert!(elapsed >= Duration::from_millis(10));
    }
}
