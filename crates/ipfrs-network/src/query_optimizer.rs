//! DHT Query Optimization Module
//!
//! Provides query optimization features including:
//! - Early termination based on result quality
//! - Query pipelining for sequential operations
//! - Query performance tracking
//! - Adaptive query strategies

use libp2p::PeerId;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Query optimization configuration
#[derive(Debug, Clone)]
pub struct QueryOptimizerConfig {
    /// Enable early termination
    pub enable_early_termination: bool,
    /// Minimum acceptable result quality (0.0-1.0)
    pub min_result_quality: f64,
    /// Enable query pipelining
    pub enable_pipelining: bool,
    /// Maximum pipelined queries
    pub max_pipelined_queries: usize,
    /// Query timeout
    pub query_timeout: Duration,
}

impl Default for QueryOptimizerConfig {
    fn default() -> Self {
        Self {
            enable_early_termination: true,
            min_result_quality: 0.7,
            enable_pipelining: true,
            max_pipelined_queries: 5,
            query_timeout: Duration::from_secs(10),
        }
    }
}

/// Query result with quality score
#[derive(Debug, Clone)]
pub struct QueryResult {
    /// Result peers
    pub peers: Vec<PeerId>,
    /// Quality score (0.0-1.0)
    pub quality: f64,
    /// Query duration
    pub duration: Duration,
    /// Number of peers queried
    pub peers_queried: usize,
}

/// Query performance metrics
#[derive(Debug, Clone, Default)]
pub struct QueryMetrics {
    /// Total queries
    pub total_queries: u64,
    /// Queries with early termination
    pub early_terminated: u64,
    /// Average query duration
    pub avg_duration: Duration,
    /// Average result quality
    pub avg_quality: f64,
    /// Queries that timed out
    pub timeouts: u64,
}

/// Query optimizer for DHT operations
pub struct QueryOptimizer {
    config: QueryOptimizerConfig,
    /// Query performance history (query_id -> metrics)
    query_history: Arc<RwLock<HashMap<String, QueryPerformance>>>,
    /// Global metrics
    metrics: Arc<RwLock<QueryMetrics>>,
}

/// Performance data for a single query
#[derive(Debug, Clone)]
struct QueryPerformance {
    started_at: Instant,
    duration: Option<Duration>,
    quality: Option<f64>,
    peers_queried: usize,
    early_terminated: bool,
}

impl QueryOptimizer {
    /// Create a new query optimizer
    pub fn new(config: QueryOptimizerConfig) -> Self {
        Self {
            config,
            query_history: Arc::new(RwLock::new(HashMap::new())),
            metrics: Arc::new(RwLock::new(QueryMetrics::default())),
        }
    }

    /// Start tracking a new query
    pub fn start_query(&self, query_id: String) {
        let mut history = self.query_history.write();
        history.insert(
            query_id,
            QueryPerformance {
                started_at: Instant::now(),
                duration: None,
                quality: None,
                peers_queried: 0,
                early_terminated: false,
            },
        );
    }

    /// Check if query should terminate early based on result quality
    pub fn should_terminate_early(&self, query_id: &str, current_results: &[PeerId]) -> bool {
        if !self.config.enable_early_termination {
            return false;
        }

        // Calculate result quality based on number of results and response time
        let quality = self.calculate_result_quality(query_id, current_results);

        quality >= self.config.min_result_quality
    }

    /// Calculate result quality score
    fn calculate_result_quality(&self, query_id: &str, results: &[PeerId]) -> f64 {
        let history = self.query_history.read();

        if let Some(perf) = history.get(query_id) {
            let elapsed = perf.started_at.elapsed();
            let timeout = self.config.query_timeout;

            // Quality factors:
            // 1. Number of results (more is better, up to a point)
            let result_score = (results.len() as f64 / 20.0).min(1.0);

            // 2. Response time (faster is better)
            let time_score = 1.0 - (elapsed.as_secs_f64() / timeout.as_secs_f64()).min(1.0);

            // Weighted average
            (result_score * 0.7) + (time_score * 0.3)
        } else {
            0.0
        }
    }

    /// Complete a query and record metrics
    pub fn complete_query(&self, query_id: &str, result: QueryResult) {
        let mut history = self.query_history.write();
        let mut metrics = self.metrics.write();

        if let Some(perf) = history.get_mut(query_id) {
            perf.duration = Some(result.duration);
            perf.quality = Some(result.quality);
            perf.peers_queried = result.peers_queried;

            // Update global metrics
            metrics.total_queries += 1;
            if perf.early_terminated {
                metrics.early_terminated += 1;
            }

            // Update average duration (running average)
            if metrics.total_queries == 1 {
                metrics.avg_duration = result.duration;
                metrics.avg_quality = result.quality;
            } else {
                let count = metrics.total_queries as f64;
                let old_avg = metrics.avg_duration.as_secs_f64();
                let new_avg = (old_avg * (count - 1.0) + result.duration.as_secs_f64()) / count;
                metrics.avg_duration = Duration::from_secs_f64(new_avg);

                metrics.avg_quality =
                    (metrics.avg_quality * (count - 1.0) + result.quality) / count;
            }
        }

        // Clean up old history (keep last 1000 queries)
        if history.len() > 1000 {
            let oldest_keys: Vec<String> = history
                .iter()
                .filter_map(|(k, v)| {
                    if v.started_at.elapsed() > Duration::from_secs(3600) {
                        Some(k.clone())
                    } else {
                        None
                    }
                })
                .collect();

            for key in oldest_keys {
                history.remove(&key);
            }
        }
    }

    /// Mark a query as early terminated
    pub fn mark_early_terminated(&self, query_id: &str) {
        let mut history = self.query_history.write();
        if let Some(perf) = history.get_mut(query_id) {
            perf.early_terminated = true;
        }
    }

    /// Record a query timeout
    pub fn record_timeout(&self, query_id: &str) {
        let mut metrics = self.metrics.write();
        metrics.timeouts += 1;

        // Also remove from history
        let mut history = self.query_history.write();
        history.remove(query_id);
    }

    /// Get query metrics
    pub fn get_metrics(&self) -> QueryMetrics {
        self.metrics.read().clone()
    }

    /// Get early termination rate
    pub fn early_termination_rate(&self) -> f64 {
        let metrics = self.metrics.read();
        if metrics.total_queries == 0 {
            0.0
        } else {
            metrics.early_terminated as f64 / metrics.total_queries as f64
        }
    }

    /// Check if we can pipeline a new query
    pub fn can_pipeline_query(&self) -> bool {
        if !self.config.enable_pipelining {
            return false;
        }

        let history = self.query_history.read();
        let active_queries = history.values().filter(|p| p.duration.is_none()).count();

        active_queries < self.config.max_pipelined_queries
    }

    /// Get active query count
    pub fn active_query_count(&self) -> usize {
        let history = self.query_history.read();
        history.values().filter(|p| p.duration.is_none()).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_query_optimizer_creation() {
        let config = QueryOptimizerConfig::default();
        let optimizer = QueryOptimizer::new(config);

        assert_eq!(optimizer.active_query_count(), 0);
        assert_eq!(optimizer.get_metrics().total_queries, 0);
    }

    #[test]
    fn test_start_query() {
        let optimizer = QueryOptimizer::new(QueryOptimizerConfig::default());
        optimizer.start_query("test_query".to_string());

        assert_eq!(optimizer.active_query_count(), 1);
    }

    #[test]
    fn test_complete_query() {
        let optimizer = QueryOptimizer::new(QueryOptimizerConfig::default());
        optimizer.start_query("test_query".to_string());

        let result = QueryResult {
            peers: vec![],
            quality: 0.8,
            duration: Duration::from_millis(100),
            peers_queried: 5,
        };

        optimizer.complete_query("test_query", result);

        let metrics = optimizer.get_metrics();
        assert_eq!(metrics.total_queries, 1);
        assert_eq!(metrics.avg_quality, 0.8);
    }

    #[test]
    fn test_early_termination() {
        let config = QueryOptimizerConfig {
            min_result_quality: 0.5,
            ..Default::default()
        };

        let optimizer = QueryOptimizer::new(config);
        optimizer.start_query("test_query".to_string());

        // Simulate getting many results quickly (should have high quality)
        let peers: Vec<PeerId> = (0..20).map(|_| PeerId::random()).collect();

        let should_terminate = optimizer.should_terminate_early("test_query", &peers);
        assert!(should_terminate);
    }

    #[test]
    fn test_pipelining() {
        let config = QueryOptimizerConfig {
            max_pipelined_queries: 3,
            ..Default::default()
        };

        let optimizer = QueryOptimizer::new(config);

        // Start multiple queries
        optimizer.start_query("query1".to_string());
        optimizer.start_query("query2".to_string());
        optimizer.start_query("query3".to_string());

        assert_eq!(optimizer.active_query_count(), 3);
        assert!(!optimizer.can_pipeline_query()); // At limit

        // Complete one query
        optimizer.complete_query(
            "query1",
            QueryResult {
                peers: vec![],
                quality: 0.8,
                duration: Duration::from_millis(100),
                peers_queried: 5,
            },
        );

        assert!(optimizer.can_pipeline_query()); // Now we can pipeline
    }

    #[test]
    fn test_metrics_tracking() {
        let optimizer = QueryOptimizer::new(QueryOptimizerConfig::default());

        optimizer.start_query("query1".to_string());
        optimizer.mark_early_terminated("query1");
        optimizer.complete_query(
            "query1",
            QueryResult {
                peers: vec![],
                quality: 0.9,
                duration: Duration::from_millis(50),
                peers_queried: 10,
            },
        );

        let metrics = optimizer.get_metrics();
        assert_eq!(metrics.total_queries, 1);
        assert_eq!(metrics.early_terminated, 1);
        assert_eq!(optimizer.early_termination_rate(), 1.0);
    }

    #[test]
    fn test_timeout_recording() {
        let optimizer = QueryOptimizer::new(QueryOptimizerConfig::default());

        optimizer.start_query("slow_query".to_string());
        optimizer.record_timeout("slow_query");

        let metrics = optimizer.get_metrics();
        assert_eq!(metrics.timeouts, 1);
    }

    #[test]
    fn test_query_quality_calculation() {
        let optimizer = QueryOptimizer::new(QueryOptimizerConfig::default());
        optimizer.start_query("test_query".to_string());

        // With many results, quality should be high
        let many_peers: Vec<PeerId> = (0..20).map(|_| PeerId::random()).collect();
        let quality_high = optimizer.calculate_result_quality("test_query", &many_peers);

        // With few results, quality should be lower
        let few_peers: Vec<PeerId> = (0..2).map(|_| PeerId::random()).collect();
        let quality_low = optimizer.calculate_result_quality("test_query", &few_peers);

        assert!(quality_high > quality_low);
    }

    #[test]
    fn test_pipelining_disabled() {
        let config = QueryOptimizerConfig {
            enable_pipelining: false,
            ..Default::default()
        };

        let optimizer = QueryOptimizer::new(config);
        assert!(!optimizer.can_pipeline_query());
    }

    #[test]
    fn test_early_termination_disabled() {
        let config = QueryOptimizerConfig {
            enable_early_termination: false,
            ..Default::default()
        };

        let optimizer = QueryOptimizer::new(config);
        optimizer.start_query("test_query".to_string());

        let peers: Vec<PeerId> = (0..20).map(|_| PeerId::random()).collect();
        assert!(!optimizer.should_terminate_early("test_query", &peers));
    }
}
