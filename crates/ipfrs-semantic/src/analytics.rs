//! Query analytics and performance tracking
//!
//! This module provides analytics capabilities to track query patterns,
//! performance metrics, and usage statistics for semantic search operations.

use crate::hnsw::DistanceMetric;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Query performance metrics
#[derive(Debug, Clone)]
pub struct QueryMetrics {
    /// Query duration
    pub duration: Duration,
    /// Number of results returned
    pub result_count: usize,
    /// Whether query was served from cache
    pub cache_hit: bool,
    /// Distance metric used
    pub metric: DistanceMetric,
    /// ef_search parameter used
    pub ef_search: usize,
    /// k (number of results requested)
    pub k: usize,
}

/// Aggregated analytics for a time window
#[derive(Debug, Clone)]
pub struct AnalyticsSummary {
    /// Total number of queries
    pub total_queries: usize,
    /// Number of cache hits
    pub cache_hits: usize,
    /// Cache hit rate (0.0-1.0)
    pub cache_hit_rate: f32,
    /// Average query duration
    pub avg_duration: Duration,
    /// P50 latency
    pub p50_latency: Duration,
    /// P90 latency
    pub p90_latency: Duration,
    /// P99 latency
    pub p99_latency: Duration,
    /// Most common k values
    pub top_k_values: Vec<(usize, usize)>, // (k_value, count)
    /// Queries per second
    pub qps: f32,
}

/// Detected query pattern for analytics
#[derive(Debug, Clone)]
pub struct DetectedPattern {
    /// Hash of the query embedding (for pattern matching)
    pub embedding_hash: u64,
    /// Frequency of this pattern
    pub frequency: usize,
    /// Average duration for this pattern
    pub avg_duration: Duration,
}

/// Analytics tracker
pub struct AnalyticsTracker {
    /// Query history
    query_history: Arc<RwLock<Vec<(Instant, QueryMetrics)>>>,
    /// Query patterns (embedding hash -> pattern)
    query_patterns: Arc<RwLock<HashMap<u64, DetectedPattern>>>,
    /// Maximum history size
    max_history_size: usize,
    /// Start time for QPS calculation
    start_time: Instant,
}

impl AnalyticsTracker {
    /// Create a new analytics tracker
    pub fn new(max_history_size: usize) -> Self {
        Self {
            query_history: Arc::new(RwLock::new(Vec::new())),
            query_patterns: Arc::new(RwLock::new(HashMap::new())),
            max_history_size,
            start_time: Instant::now(),
        }
    }

    /// Create a tracker with default settings
    pub fn with_defaults() -> Self {
        Self::new(10000) // Keep last 10k queries
    }

    /// Record a query
    pub fn record_query(&self, embedding: &[f32], metrics: QueryMetrics) {
        let now = Instant::now();
        let hash = Self::hash_embedding(embedding);

        // Update history
        {
            let mut history = self.query_history.write();
            history.push((now, metrics.clone()));

            // Trim if needed
            if history.len() > self.max_history_size {
                let remove_count = history.len() - self.max_history_size;
                history.drain(0..remove_count);
            }
        }

        // Update patterns
        {
            let mut patterns = self.query_patterns.write();
            patterns
                .entry(hash)
                .and_modify(|pattern| {
                    pattern.frequency += 1;
                    // Update running average
                    let total = pattern.avg_duration.as_nanos() as f64
                        * (pattern.frequency - 1) as f64
                        + metrics.duration.as_nanos() as f64;
                    pattern.avg_duration =
                        Duration::from_nanos((total / pattern.frequency as f64) as u64);
                })
                .or_insert(DetectedPattern {
                    embedding_hash: hash,
                    frequency: 1,
                    avg_duration: metrics.duration,
                });
        }
    }

    /// Get analytics summary for a time window
    pub fn get_summary(&self, window: Option<Duration>) -> AnalyticsSummary {
        let history = self.query_history.read();

        // Filter by time window
        let now = Instant::now();
        let filtered: Vec<&QueryMetrics> = if let Some(duration) = window {
            history
                .iter()
                .filter(|(timestamp, _)| now.duration_since(*timestamp) <= duration)
                .map(|(_, metrics)| metrics)
                .collect()
        } else {
            history.iter().map(|(_, metrics)| metrics).collect()
        };

        if filtered.is_empty() {
            return AnalyticsSummary {
                total_queries: 0,
                cache_hits: 0,
                cache_hit_rate: 0.0,
                avg_duration: Duration::from_secs(0),
                p50_latency: Duration::from_secs(0),
                p90_latency: Duration::from_secs(0),
                p99_latency: Duration::from_secs(0),
                top_k_values: Vec::new(),
                qps: 0.0,
            };
        }

        let total_queries = filtered.len();
        let cache_hits = filtered.iter().filter(|m| m.cache_hit).count();
        let cache_hit_rate = cache_hits as f32 / total_queries as f32;

        // Calculate average duration
        let total_duration: u128 = filtered.iter().map(|m| m.duration.as_nanos()).sum();
        let avg_duration = Duration::from_nanos((total_duration / total_queries as u128) as u64);

        // Calculate percentiles
        let mut durations: Vec<Duration> = filtered.iter().map(|m| m.duration).collect();
        durations.sort();

        let p50_latency = durations[total_queries * 50 / 100];
        let p90_latency = durations[total_queries * 90 / 100];
        let p99_latency = durations[total_queries * 99 / 100];

        // Calculate top k values
        let mut k_counts: HashMap<usize, usize> = HashMap::new();
        for metrics in &filtered {
            *k_counts.entry(metrics.k).or_insert(0) += 1;
        }
        let mut top_k_values: Vec<(usize, usize)> = k_counts.into_iter().collect();
        top_k_values.sort_by_key(|a| std::cmp::Reverse(a.1)); // Sort by count descending
        top_k_values.truncate(5); // Top 5

        // Calculate QPS
        let elapsed = self.start_time.elapsed().as_secs_f32();
        let qps = if elapsed > 0.0 {
            total_queries as f32 / elapsed
        } else {
            0.0
        };

        AnalyticsSummary {
            total_queries,
            cache_hits,
            cache_hit_rate,
            avg_duration,
            p50_latency,
            p90_latency,
            p99_latency,
            top_k_values,
            qps,
        }
    }

    /// Get top query patterns
    pub fn get_top_patterns(&self, limit: usize) -> Vec<DetectedPattern> {
        let patterns = self.query_patterns.read();
        let mut sorted: Vec<DetectedPattern> = patterns.values().cloned().collect();
        sorted.sort_by_key(|a| std::cmp::Reverse(a.frequency));
        sorted.truncate(limit);
        sorted
    }

    /// Clear all analytics data
    pub fn clear(&self) {
        self.query_history.write().clear();
        self.query_patterns.write().clear();
    }

    /// Get total number of queries tracked
    pub fn total_queries(&self) -> usize {
        self.query_history.read().len()
    }

    /// Hash an embedding for pattern detection
    fn hash_embedding(embedding: &[f32]) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        // Sample embedding to reduce hash computation
        for (i, &val) in embedding.iter().enumerate().step_by(8) {
            (i, (val * 1000.0) as i32).hash(&mut hasher);
        }
        hasher.finish()
    }
}

/// Query timer for automatic metrics collection
pub struct QueryTimer {
    start: Instant,
    embedding: Vec<f32>,
    k: usize,
    ef_search: usize,
    metric: DistanceMetric,
    cache_hit: bool,
}

impl QueryTimer {
    /// Start a new query timer
    pub fn start(embedding: Vec<f32>, k: usize, ef_search: usize, metric: DistanceMetric) -> Self {
        Self {
            start: Instant::now(),
            embedding,
            k,
            ef_search,
            metric,
            cache_hit: false,
        }
    }

    /// Mark query as cache hit
    pub fn set_cache_hit(&mut self, hit: bool) {
        self.cache_hit = hit;
    }

    /// Finish the timer and record metrics
    pub fn finish(self, tracker: &AnalyticsTracker, result_count: usize) {
        let duration = self.start.elapsed();
        let metrics = QueryMetrics {
            duration,
            result_count,
            cache_hit: self.cache_hit,
            metric: self.metric,
            ef_search: self.ef_search,
            k: self.k,
        };
        tracker.record_query(&self.embedding, metrics);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tracker_creation() {
        let tracker = AnalyticsTracker::with_defaults();
        assert_eq!(tracker.total_queries(), 0);
    }

    #[test]
    fn test_record_query() {
        let tracker = AnalyticsTracker::with_defaults();
        let embedding = vec![0.5; 128];

        let metrics = QueryMetrics {
            duration: Duration::from_millis(10),
            result_count: 5,
            cache_hit: false,
            metric: DistanceMetric::Cosine,
            ef_search: 50,
            k: 10,
        };

        tracker.record_query(&embedding, metrics);
        assert_eq!(tracker.total_queries(), 1);
    }

    #[test]
    fn test_analytics_summary() {
        let tracker = AnalyticsTracker::with_defaults();
        let embedding = vec![0.5; 128];

        // Record multiple queries
        for i in 0..10 {
            let metrics = QueryMetrics {
                duration: Duration::from_millis(i * 10),
                result_count: 5,
                cache_hit: i % 2 == 0, // 50% cache hit rate
                metric: DistanceMetric::Cosine,
                ef_search: 50,
                k: 10,
            };
            tracker.record_query(&embedding, metrics);
        }

        let summary = tracker.get_summary(None);
        assert_eq!(summary.total_queries, 10);
        assert_eq!(summary.cache_hits, 5);
        assert!((summary.cache_hit_rate - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_query_patterns() {
        let tracker = AnalyticsTracker::with_defaults();

        // Record same pattern multiple times
        let embedding1 = vec![0.5; 128];
        for _ in 0..5 {
            let metrics = QueryMetrics {
                duration: Duration::from_millis(10),
                result_count: 5,
                cache_hit: false,
                metric: DistanceMetric::Cosine,
                ef_search: 50,
                k: 10,
            };
            tracker.record_query(&embedding1, metrics);
        }

        // Record different pattern
        let embedding2 = vec![0.8; 128];
        for _ in 0..3 {
            let metrics = QueryMetrics {
                duration: Duration::from_millis(20),
                result_count: 5,
                cache_hit: false,
                metric: DistanceMetric::Cosine,
                ef_search: 50,
                k: 10,
            };
            tracker.record_query(&embedding2, metrics);
        }

        let patterns = tracker.get_top_patterns(2);
        assert_eq!(patterns.len(), 2);
        assert_eq!(patterns[0].frequency, 5); // Most frequent pattern first
    }

    #[test]
    fn test_query_timer() {
        let tracker = AnalyticsTracker::with_defaults();
        let embedding = vec![0.5; 128];

        let timer = QueryTimer::start(embedding, 10, 50, DistanceMetric::Cosine);
        std::thread::sleep(Duration::from_millis(10));
        timer.finish(&tracker, 5);

        assert_eq!(tracker.total_queries(), 1);
        let summary = tracker.get_summary(None);
        assert!(summary.avg_duration >= Duration::from_millis(10));
    }

    #[test]
    fn test_top_k_values() {
        let tracker = AnalyticsTracker::with_defaults();
        let embedding = vec![0.5; 128];

        // Record queries with different k values
        for k in &[5, 10, 10, 10, 20] {
            let metrics = QueryMetrics {
                duration: Duration::from_millis(10),
                result_count: 5,
                cache_hit: false,
                metric: DistanceMetric::Cosine,
                ef_search: 50,
                k: *k,
            };
            tracker.record_query(&embedding, metrics);
        }

        let summary = tracker.get_summary(None);
        assert_eq!(summary.top_k_values[0].0, 10); // k=10 is most common
        assert_eq!(summary.top_k_values[0].1, 3); // appeared 3 times
    }

    #[test]
    fn test_clear_analytics() {
        let tracker = AnalyticsTracker::with_defaults();
        let embedding = vec![0.5; 128];

        let metrics = QueryMetrics {
            duration: Duration::from_millis(10),
            result_count: 5,
            cache_hit: false,
            metric: DistanceMetric::Cosine,
            ef_search: 50,
            k: 10,
        };

        tracker.record_query(&embedding, metrics);
        assert_eq!(tracker.total_queries(), 1);

        tracker.clear();
        assert_eq!(tracker.total_queries(), 0);
    }

    #[test]
    fn test_time_window_filtering() {
        let tracker = AnalyticsTracker::with_defaults();
        let embedding = vec![0.5; 128];

        // Record a query
        let metrics = QueryMetrics {
            duration: Duration::from_millis(10),
            result_count: 5,
            cache_hit: false,
            metric: DistanceMetric::Cosine,
            ef_search: 50,
            k: 10,
        };
        tracker.record_query(&embedding, metrics);

        // Get summary for last 1 second (should include the query)
        let summary = tracker.get_summary(Some(Duration::from_secs(1)));
        assert_eq!(summary.total_queries, 1);

        // Sleep and get summary for a very short window (should not include old query)
        std::thread::sleep(Duration::from_millis(100));
        let summary = tracker.get_summary(Some(Duration::from_millis(10)));
        assert_eq!(summary.total_queries, 0);
    }
}
