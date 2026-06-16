//! DHT query batching and frequency optimization
//!
//! This module provides batching and rate limiting for DHT queries to:
//! - Reduce network traffic by batching similar queries
//! - Control query frequency to prevent network flooding
//! - Merge duplicate queries
//! - Implement adaptive query delays based on network conditions

use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use thiserror::Error;
use tokio::sync::mpsc;

/// Errors that can occur during query batching
#[derive(Error, Debug, Clone)]
pub enum QueryBatcherError {
    #[error("Batch queue is full")]
    QueueFull,

    #[error("Query rate limit exceeded")]
    RateLimitExceeded,

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
}

/// Configuration for query batching
#[derive(Debug, Clone)]
pub struct QueryBatcherConfig {
    /// Maximum batch size (queries per batch)
    pub max_batch_size: usize,

    /// Batch window duration (wait time before sending batch)
    pub batch_window: Duration,

    /// Maximum queries per second (rate limit)
    pub max_queries_per_second: u64,

    /// Enable query deduplication
    pub enable_deduplication: bool,

    /// Deduplication window (merge queries within this window)
    pub dedup_window: Duration,

    /// Maximum pending queries in queue
    pub max_pending_queries: usize,

    /// Enable adaptive rate limiting
    pub enable_adaptive_rate: bool,

    /// Target success rate for adaptive limiting (0.0-1.0)
    pub target_success_rate: f64,
}

impl Default for QueryBatcherConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 10,
            batch_window: Duration::from_millis(100),
            max_queries_per_second: 100,
            enable_deduplication: true,
            dedup_window: Duration::from_secs(5),
            max_pending_queries: 1000,
            enable_adaptive_rate: true,
            target_success_rate: 0.8,
        }
    }
}

impl QueryBatcherConfig {
    /// Configuration for low-power mode (minimal queries)
    pub fn low_power() -> Self {
        Self {
            max_batch_size: 20,
            batch_window: Duration::from_millis(500),
            max_queries_per_second: 10,
            enable_deduplication: true,
            dedup_window: Duration::from_secs(10),
            max_pending_queries: 100,
            enable_adaptive_rate: true,
            target_success_rate: 0.7,
        }
    }

    /// Configuration for mobile devices
    pub fn mobile() -> Self {
        Self {
            max_batch_size: 15,
            batch_window: Duration::from_millis(200),
            max_queries_per_second: 50,
            enable_deduplication: true,
            dedup_window: Duration::from_secs(5),
            max_pending_queries: 500,
            enable_adaptive_rate: true,
            target_success_rate: 0.75,
        }
    }

    /// Configuration for high-performance mode
    pub fn high_performance() -> Self {
        Self {
            max_batch_size: 5,
            batch_window: Duration::from_millis(50),
            max_queries_per_second: 500,
            enable_deduplication: false,
            dedup_window: Duration::from_secs(1),
            max_pending_queries: 5000,
            enable_adaptive_rate: false,
            target_success_rate: 0.9,
        }
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<(), QueryBatcherError> {
        if self.max_batch_size == 0 {
            return Err(QueryBatcherError::InvalidConfig(
                "max_batch_size must be > 0".to_string(),
            ));
        }

        if self.max_queries_per_second == 0 {
            return Err(QueryBatcherError::InvalidConfig(
                "max_queries_per_second must be > 0".to_string(),
            ));
        }

        if self.target_success_rate < 0.0 || self.target_success_rate > 1.0 {
            return Err(QueryBatcherError::InvalidConfig(
                "target_success_rate must be in [0.0, 1.0]".to_string(),
            ));
        }

        Ok(())
    }
}

/// Type of DHT query
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum QueryType {
    /// Find providers for a CID
    FindProviders(String),
    /// Find a specific peer
    FindPeer(String),
    /// Get value for a key
    GetValue(String),
    /// Put value for a key
    PutValue(String),
}

impl QueryType {
    /// Get the query key for deduplication
    pub fn key(&self) -> String {
        match self {
            QueryType::FindProviders(cid) => format!("providers:{}", cid),
            QueryType::FindPeer(peer) => format!("peer:{}", peer),
            QueryType::GetValue(key) => format!("get:{}", key),
            QueryType::PutValue(key) => format!("put:{}", key),
        }
    }
}

/// A pending query in the batch queue
#[derive(Debug, Clone)]
pub struct PendingQuery {
    /// Query type
    pub query_type: QueryType,
    /// Timestamp when query was added
    pub added_at: Instant,
    /// Response channel (if needed)
    pub response_tx: Option<mpsc::UnboundedSender<QueryBatchResult>>,
}

/// Result of a batched query
#[derive(Debug, Clone)]
pub struct QueryBatchResult {
    /// Whether the query succeeded
    pub success: bool,
    /// Number of results found
    pub result_count: usize,
    /// Time taken
    pub duration: Duration,
}

/// Query batching state
#[derive(Debug)]
struct BatcherState {
    /// Current batch being assembled
    current_batch: Vec<PendingQuery>,
    /// Last batch send time
    last_batch_sent: Instant,
    /// Query count in current second
    queries_this_second: u64,
    /// Current second start
    second_start: Instant,
    /// Query history for deduplication
    recent_queries: HashMap<String, Instant>,
    /// Adaptive rate limit multiplier (1.0 = normal)
    rate_multiplier: f64,
    /// Recent success rate (for adaptive limiting)
    recent_success_rate: f64,
}

impl BatcherState {
    fn new() -> Self {
        let now = Instant::now();
        Self {
            current_batch: Vec::new(),
            last_batch_sent: now,
            queries_this_second: 0,
            second_start: now,
            recent_queries: HashMap::new(),
            rate_multiplier: 1.0,
            recent_success_rate: 1.0,
        }
    }
}

/// DHT query batcher
pub struct QueryBatcher {
    config: QueryBatcherConfig,
    state: Arc<RwLock<BatcherState>>,
    stats: Arc<RwLock<QueryBatcherStats>>,
}

impl QueryBatcher {
    /// Create a new query batcher
    pub fn new(config: QueryBatcherConfig) -> Result<Self, QueryBatcherError> {
        config.validate()?;

        Ok(Self {
            config,
            state: Arc::new(RwLock::new(BatcherState::new())),
            stats: Arc::new(RwLock::new(QueryBatcherStats::default())),
        })
    }

    /// Add a query to the batch
    pub fn add_query(&self, query: QueryType) -> Result<(), QueryBatcherError> {
        let mut state = self.state.write();
        let mut stats = self.stats.write();

        // Check queue size
        if state.current_batch.len() >= self.config.max_pending_queries {
            stats.queries_dropped += 1;
            return Err(QueryBatcherError::QueueFull);
        }

        // Check rate limit
        let now = Instant::now();
        if now.duration_since(state.second_start) >= Duration::from_secs(1) {
            state.queries_this_second = 0;
            state.second_start = now;
        }

        let effective_rate_limit =
            (self.config.max_queries_per_second as f64 * state.rate_multiplier) as u64;

        if state.queries_this_second >= effective_rate_limit {
            stats.queries_rate_limited += 1;
            return Err(QueryBatcherError::RateLimitExceeded);
        }

        // Deduplication check
        if self.config.enable_deduplication {
            let key = query.key();
            if let Some(&last_query) = state.recent_queries.get(&key) {
                if now.duration_since(last_query) < self.config.dedup_window {
                    stats.queries_deduplicated += 1;
                    return Ok(()); // Skip duplicate
                }
            }
            state.recent_queries.insert(key, now);
        }

        // Add to batch
        let pending = PendingQuery {
            query_type: query,
            added_at: now,
            response_tx: None,
        };

        state.current_batch.push(pending);
        state.queries_this_second += 1;
        stats.queries_batched += 1;

        Ok(())
    }

    /// Check if batch is ready to send
    pub fn should_send_batch(&self) -> bool {
        let state = self.state.read();

        if state.current_batch.is_empty() {
            return false;
        }

        // Send if batch is full
        if state.current_batch.len() >= self.config.max_batch_size {
            return true;
        }

        // Send if batch window expired
        let now = Instant::now();
        if now.duration_since(state.last_batch_sent) >= self.config.batch_window {
            return true;
        }

        false
    }

    /// Get the current batch and clear it
    pub fn take_batch(&self) -> Vec<PendingQuery> {
        let mut state = self.state.write();
        let mut stats = self.stats.write();

        let batch = std::mem::take(&mut state.current_batch);
        state.last_batch_sent = Instant::now();

        if !batch.is_empty() {
            stats.batches_sent += 1;
            stats.total_queries_sent += batch.len() as u64;
        }

        batch
    }

    /// Record query result for adaptive rate limiting
    pub fn record_result(&self, result: QueryBatchResult) {
        let mut state = self.state.write();
        let mut stats = self.stats.write();

        if result.success {
            stats.successful_queries += 1;
        } else {
            stats.failed_queries += 1;
        }

        // Update adaptive rate limiter
        if self.config.enable_adaptive_rate {
            let total = stats.successful_queries + stats.failed_queries;
            if total > 0 {
                state.recent_success_rate = stats.successful_queries as f64 / total as f64;

                // Adjust rate multiplier based on success rate
                if state.recent_success_rate < self.config.target_success_rate {
                    // Too many failures, slow down
                    state.rate_multiplier = (state.rate_multiplier * 0.9).max(0.1);
                    stats.rate_adjustments += 1;
                } else if state.recent_success_rate > self.config.target_success_rate + 0.1 {
                    // High success rate, can speed up
                    state.rate_multiplier = (state.rate_multiplier * 1.1).min(2.0);
                    stats.rate_adjustments += 1;
                }
            }
        }
    }

    /// Get current statistics
    pub fn stats(&self) -> QueryBatcherStats {
        self.stats.read().clone()
    }

    /// Get current rate multiplier (for adaptive rate limiting)
    pub fn rate_multiplier(&self) -> f64 {
        self.state.read().rate_multiplier
    }

    /// Get current success rate
    pub fn success_rate(&self) -> f64 {
        self.state.read().recent_success_rate
    }

    /// Clean up old deduplication entries
    pub fn cleanup_dedup_cache(&self) {
        let mut state = self.state.write();
        let now = Instant::now();

        state.recent_queries.retain(|_, &mut last_query| {
            now.duration_since(last_query) < self.config.dedup_window * 2
        });
    }

    /// Reset statistics
    pub fn reset_stats(&self) {
        *self.stats.write() = QueryBatcherStats::default();
    }
}

/// Statistics for query batching
#[derive(Debug, Clone, Default)]
pub struct QueryBatcherStats {
    /// Total queries added to batches
    pub queries_batched: u64,
    /// Queries dropped due to full queue
    pub queries_dropped: u64,
    /// Queries skipped due to rate limiting
    pub queries_rate_limited: u64,
    /// Queries deduplicated
    pub queries_deduplicated: u64,
    /// Number of batches sent
    pub batches_sent: u64,
    /// Total queries actually sent (after batching/dedup)
    pub total_queries_sent: u64,
    /// Successful queries
    pub successful_queries: u64,
    /// Failed queries
    pub failed_queries: u64,
    /// Number of rate adjustments made
    pub rate_adjustments: u64,
}

impl QueryBatcherStats {
    /// Calculate the deduplication ratio
    pub fn dedup_ratio(&self) -> f64 {
        if self.queries_batched == 0 {
            return 0.0;
        }
        self.queries_deduplicated as f64 / self.queries_batched as f64
    }

    /// Calculate the batching efficiency (queries saved)
    pub fn batching_efficiency(&self) -> f64 {
        if self.queries_batched == 0 {
            return 0.0;
        }
        let saved = self.queries_batched - self.total_queries_sent;
        saved as f64 / self.queries_batched as f64
    }

    /// Calculate success rate
    pub fn success_rate(&self) -> f64 {
        let total = self.successful_queries + self.failed_queries;
        if total == 0 {
            return 0.0;
        }
        self.successful_queries as f64 / total as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = QueryBatcherConfig::default();
        assert!(config.validate().is_ok());
        assert_eq!(config.max_batch_size, 10);
        assert!(config.enable_deduplication);
    }

    #[test]
    fn test_config_low_power() {
        let config = QueryBatcherConfig::low_power();
        assert!(config.validate().is_ok());
        assert_eq!(config.max_queries_per_second, 10);
    }

    #[test]
    fn test_config_mobile() {
        let config = QueryBatcherConfig::mobile();
        assert!(config.validate().is_ok());
        assert_eq!(config.max_queries_per_second, 50);
    }

    #[test]
    fn test_config_high_performance() {
        let config = QueryBatcherConfig::high_performance();
        assert!(config.validate().is_ok());
        assert!(!config.enable_deduplication);
    }

    #[test]
    fn test_config_validation() {
        let config = QueryBatcherConfig {
            max_batch_size: 0,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_query_type_key() {
        let q1 = QueryType::FindProviders("QmTest".to_string());
        let q2 = QueryType::FindProviders("QmTest".to_string());
        assert_eq!(q1.key(), q2.key());
    }

    #[test]
    fn test_add_query() {
        let config = QueryBatcherConfig::default();
        let batcher = QueryBatcher::new(config)
            .expect("test: QueryBatcher::new should succeed with valid config");

        let query = QueryType::FindProviders("QmTest".to_string());
        let result = batcher.add_query(query);
        assert!(result.is_ok());

        let stats = batcher.stats();
        assert_eq!(stats.queries_batched, 1);
    }

    #[test]
    fn test_deduplication() {
        let config = QueryBatcherConfig::default();
        let batcher = QueryBatcher::new(config)
            .expect("test: QueryBatcher::new should succeed with valid config");

        let query = QueryType::FindProviders("QmTest".to_string());

        batcher
            .add_query(query.clone())
            .expect("test: add_query should succeed");
        batcher
            .add_query(query)
            .expect("test: add_query duplicate should succeed (deduplication path)"); // Duplicate

        let stats = batcher.stats();
        assert_eq!(stats.queries_deduplicated, 1);
        assert_eq!(stats.queries_batched, 1); // Only one unique query
    }

    #[test]
    fn test_batch_ready_when_full() {
        let config = QueryBatcherConfig {
            max_batch_size: 3,
            ..Default::default()
        };
        let batcher = QueryBatcher::new(config)
            .expect("test: QueryBatcher::new should succeed with max_batch_size=3");

        for i in 0..3 {
            let query = QueryType::FindProviders(format!("QmTest{}", i));
            batcher
                .add_query(query)
                .expect("test: add_query should succeed");
        }

        assert!(batcher.should_send_batch());
    }

    #[test]
    fn test_take_batch() {
        let config = QueryBatcherConfig::default();
        let batcher = QueryBatcher::new(config)
            .expect("test: QueryBatcher::new should succeed with valid config");

        for i in 0..5 {
            let query = QueryType::FindProviders(format!("QmTest{}", i));
            batcher
                .add_query(query)
                .expect("test: add_query should succeed");
        }

        let batch = batcher.take_batch();
        assert_eq!(batch.len(), 5);

        let batch2 = batcher.take_batch();
        assert_eq!(batch2.len(), 0);
    }

    #[test]
    fn test_rate_limit() {
        let config = QueryBatcherConfig {
            max_queries_per_second: 5,
            ..Default::default()
        };
        let batcher = QueryBatcher::new(config)
            .expect("test: QueryBatcher::new should succeed with max_queries_per_second=5");

        // Add 5 queries (should succeed)
        for i in 0..5 {
            let query = QueryType::FindProviders(format!("QmTest{}", i));
            assert!(batcher.add_query(query).is_ok());
        }

        // 6th query should be rate limited
        let query = QueryType::FindProviders("QmTest6".to_string());
        let result = batcher.add_query(query);
        assert!(matches!(result, Err(QueryBatcherError::RateLimitExceeded)));
    }

    #[test]
    fn test_adaptive_rate_limiting() {
        let config = QueryBatcherConfig::default();
        let batcher = QueryBatcher::new(config)
            .expect("test: QueryBatcher::new should succeed with valid config");

        let initial_rate = batcher.rate_multiplier();

        // Record failures
        for _ in 0..10 {
            batcher.record_result(QueryBatchResult {
                success: false,
                result_count: 0,
                duration: Duration::from_millis(100),
            });
        }

        let rate_after_failures = batcher.rate_multiplier();
        assert!(rate_after_failures < initial_rate);
    }

    #[test]
    fn test_stats_dedup_ratio() {
        let stats = QueryBatcherStats {
            queries_batched: 100,
            queries_deduplicated: 20,
            ..Default::default()
        };

        assert_eq!(stats.dedup_ratio(), 0.2);
    }

    #[test]
    fn test_stats_batching_efficiency() {
        let stats = QueryBatcherStats {
            queries_batched: 100,
            total_queries_sent: 60,
            ..Default::default()
        };

        assert_eq!(stats.batching_efficiency(), 0.4);
    }

    #[test]
    fn test_cleanup_dedup_cache() {
        let config = QueryBatcherConfig::default();
        let batcher = QueryBatcher::new(config)
            .expect("test: QueryBatcher::new should succeed with valid config");

        let query = QueryType::FindProviders("QmTest".to_string());
        batcher
            .add_query(query)
            .expect("test: add_query should succeed");

        // Cache should have entry
        {
            let state = batcher.state.read();
            assert_eq!(state.recent_queries.len(), 1);
        }

        batcher.cleanup_dedup_cache();

        // Cache should still have entry (not old enough)
        {
            let state = batcher.state.read();
            assert_eq!(state.recent_queries.len(), 1);
        }
    }
}
