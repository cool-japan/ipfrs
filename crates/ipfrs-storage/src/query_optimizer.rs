//! Query optimizer for storage operations
//!
//! This module provides query optimization and planning for complex storage operations.
//! It analyzes query patterns and suggests optimal execution strategies.
//!
//! # Example
//!
//! ```rust
//! use ipfrs_storage::{QueryOptimizer, QueryPlan, MemoryBlockStore};
//!
//! let store = MemoryBlockStore::new();
//! let optimizer = QueryOptimizer::new();
//!
//! // Optimize a batch get operation
//! let cids = vec![/* ... */];
//! let plan = optimizer.optimize_batch_get(&cids);
//! println!("Optimal batch size: {}", plan.batch_size);
//! ```

use ipfrs_core::Cid;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Query execution plan
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryPlan {
    /// Estimated execution time in microseconds
    pub estimated_duration_us: u64,
    /// Recommended batch size for operations
    pub batch_size: usize,
    /// Whether to use parallel execution
    pub use_parallel: bool,
    /// Estimated memory usage in bytes
    pub estimated_memory_bytes: usize,
    /// Strategy to use
    pub strategy: QueryStrategy,
    /// Additional optimization hints
    pub hints: Vec<String>,
}

/// Query execution strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QueryStrategy {
    /// Sequential execution
    Sequential,
    /// Parallel batch execution
    ParallelBatch,
    /// Streaming execution
    Streaming,
    /// Cache-first strategy
    CacheFirst,
    /// Hybrid approach
    Hybrid,
}

/// Query optimizer for storage operations
#[derive(Debug, Clone)]
pub struct QueryOptimizer {
    /// Historical query statistics
    stats: QueryStats,
    /// Configuration
    config: OptimizerConfig,
}

/// Query statistics for optimization
#[derive(Debug, Clone, Default)]
struct QueryStats {
    /// Average block size in bytes
    avg_block_size: usize,
    /// Cache hit rate (0.0 to 1.0)
    cache_hit_rate: f64,
    /// Average batch operation latency
    #[allow(dead_code)]
    avg_batch_latency_us: u64,
    /// Number of queries analyzed
    query_count: u64,
}

/// Optimizer configuration
#[derive(Debug, Clone)]
pub struct OptimizerConfig {
    /// Maximum batch size
    pub max_batch_size: usize,
    /// Minimum batch size
    pub min_batch_size: usize,
    /// Parallel execution threshold (number of items)
    pub parallel_threshold: usize,
    /// Streaming threshold (total bytes)
    pub streaming_threshold_bytes: usize,
    /// Memory limit for operations
    pub memory_limit_bytes: usize,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 1000,
            min_batch_size: 10,
            parallel_threshold: 100,
            streaming_threshold_bytes: 100 * 1024 * 1024, // 100MB
            memory_limit_bytes: 1024 * 1024 * 1024,       // 1GB
        }
    }
}

impl QueryOptimizer {
    /// Create a new query optimizer with default configuration
    pub fn new() -> Self {
        Self::with_config(OptimizerConfig::default())
    }

    /// Create a new query optimizer with custom configuration
    pub fn with_config(config: OptimizerConfig) -> Self {
        Self {
            stats: QueryStats::default(),
            config,
        }
    }

    /// Update statistics with query feedback
    pub fn update_stats(&mut self, avg_block_size: usize, cache_hit_rate: f64) {
        self.stats.avg_block_size = avg_block_size;
        self.stats.cache_hit_rate = cache_hit_rate;
        self.stats.query_count += 1;
    }

    /// Optimize a batch get operation
    pub fn optimize_batch_get(&self, cids: &[Cid]) -> QueryPlan {
        let count = cids.len();

        if count == 0 {
            return QueryPlan {
                estimated_duration_us: 0,
                batch_size: 0,
                use_parallel: false,
                estimated_memory_bytes: 0,
                strategy: QueryStrategy::Sequential,
                hints: vec!["Empty query".to_string()],
            };
        }

        // Estimate memory usage
        let estimated_memory_bytes = count * self.stats.avg_block_size;

        // Determine strategy based on size and cache hit rate
        let strategy = if estimated_memory_bytes > self.config.streaming_threshold_bytes {
            QueryStrategy::Streaming
        } else if self.stats.cache_hit_rate > 0.8 {
            QueryStrategy::CacheFirst
        } else if count >= self.config.parallel_threshold {
            QueryStrategy::ParallelBatch
        } else {
            QueryStrategy::Sequential
        };

        // Calculate optimal batch size
        let batch_size = self.calculate_batch_size(count, estimated_memory_bytes);

        // Estimate duration (simplified model)
        let base_latency_us = 500; // Base per-item latency
        let cache_speedup = 1.0 - (self.stats.cache_hit_rate * 0.7);
        let parallel_speedup = if strategy == QueryStrategy::ParallelBatch {
            0.3
        } else {
            1.0
        };
        let estimated_duration_us =
            ((count as f64) * base_latency_us as f64 * cache_speedup * parallel_speedup) as u64;

        let mut hints = Vec::new();
        if estimated_memory_bytes > self.config.memory_limit_bytes / 2 {
            hints.push("High memory usage - consider streaming".to_string());
        }
        if count > self.config.max_batch_size {
            hints.push(format!(
                "Large query - split into {} batches",
                count.div_ceil(self.config.max_batch_size)
            ));
        }
        if self.stats.cache_hit_rate < 0.3 {
            hints.push("Low cache hit rate - consider cache warming".to_string());
        }

        QueryPlan {
            estimated_duration_us,
            batch_size,
            use_parallel: strategy == QueryStrategy::ParallelBatch,
            estimated_memory_bytes,
            strategy,
            hints,
        }
    }

    /// Optimize a batch put operation
    pub fn optimize_batch_put(&self, block_count: usize, total_bytes: usize) -> QueryPlan {
        if block_count == 0 {
            return QueryPlan {
                estimated_duration_us: 0,
                batch_size: 0,
                use_parallel: false,
                estimated_memory_bytes: 0,
                strategy: QueryStrategy::Sequential,
                hints: vec!["Empty operation".to_string()],
            };
        }

        // Determine strategy
        let strategy = if total_bytes > self.config.streaming_threshold_bytes {
            QueryStrategy::Streaming
        } else if block_count >= self.config.parallel_threshold {
            QueryStrategy::ParallelBatch
        } else {
            QueryStrategy::Sequential
        };

        // Calculate optimal batch size
        let batch_size = self.calculate_batch_size(block_count, total_bytes);

        // Estimate duration (write is typically slower than read)
        let base_latency_us = 1000; // Base per-item latency for writes
        let parallel_speedup = if strategy == QueryStrategy::ParallelBatch {
            0.4
        } else {
            1.0
        };
        let estimated_duration_us =
            ((block_count as f64) * base_latency_us as f64 * parallel_speedup) as u64;

        let mut hints = Vec::new();
        if total_bytes > self.config.memory_limit_bytes {
            hints.push("Very large write - use streaming".to_string());
        }
        if block_count > self.config.max_batch_size * 2 {
            hints.push("Consider write coalescing".to_string());
        }

        QueryPlan {
            estimated_duration_us,
            batch_size,
            use_parallel: strategy == QueryStrategy::ParallelBatch,
            estimated_memory_bytes: total_bytes,
            strategy,
            hints,
        }
    }

    /// Calculate optimal batch size
    fn calculate_batch_size(&self, item_count: usize, estimated_bytes: usize) -> usize {
        // Start with max batch size
        let mut batch_size = self.config.max_batch_size;

        // Adjust based on memory constraints
        if estimated_bytes > 0 {
            let bytes_per_item = estimated_bytes / item_count;
            let memory_based_limit = self.config.memory_limit_bytes / bytes_per_item;
            batch_size = batch_size.min(memory_based_limit);
        }

        // Ensure minimum
        batch_size = batch_size.max(self.config.min_batch_size);

        // Don't exceed item count
        batch_size.min(item_count)
    }

    /// Analyze query patterns and provide recommendations
    pub fn analyze_patterns(&self, query_log: &[QueryLogEntry]) -> Vec<Recommendation> {
        let mut recommendations = Vec::new();

        if query_log.is_empty() {
            return recommendations;
        }

        // Analyze access patterns
        let mut cid_access_count: HashMap<String, usize> = HashMap::new();
        let mut total_items = 0;
        let mut large_queries = 0;

        for entry in query_log {
            for cid in &entry.cids {
                *cid_access_count.entry(cid.to_string()).or_insert(0) += 1;
            }
            total_items += entry.cids.len();
            if entry.cids.len() > self.config.parallel_threshold {
                large_queries += 1;
            }
        }

        // Hot data detection
        let hot_threshold = query_log.len() / 4; // Top 25%
        let hot_cids: Vec<_> = cid_access_count
            .iter()
            .filter(|(_, &count)| count >= hot_threshold)
            .collect();

        if !hot_cids.is_empty() {
            recommendations.push(Recommendation {
                priority: RecommendationPriority::High,
                category: RecommendationCategory::Caching,
                description: format!(
                    "Detected {} hot CIDs (accessed {}+ times). Consider pinning or caching.",
                    hot_cids.len(),
                    hot_threshold
                ),
                impact: "Improved cache hit rate by 20-40%".to_string(),
            });
        }

        // Large query detection
        if large_queries > query_log.len() / 2 {
            recommendations.push(Recommendation {
                priority: RecommendationPriority::Medium,
                category: RecommendationCategory::Performance,
                description: format!(
                    "{}% of queries are large (>{} items). Enable parallel execution.",
                    (large_queries * 100) / query_log.len(),
                    self.config.parallel_threshold
                ),
                impact: "Reduced query latency by 30-50%".to_string(),
            });
        }

        // Average query size
        let avg_query_size = total_items / query_log.len();
        if avg_query_size < self.config.min_batch_size {
            recommendations.push(Recommendation {
                priority: RecommendationPriority::Low,
                category: RecommendationCategory::Efficiency,
                description: format!(
                    "Average query size is {avg_query_size} items. Consider batching small queries."
                ),
                impact: "Reduced overhead by 10-20%".to_string(),
            });
        }

        recommendations
    }
}

impl Default for QueryOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Query log entry for pattern analysis
#[derive(Debug, Clone)]
pub struct QueryLogEntry {
    /// CIDs accessed in this query
    pub cids: Vec<Cid>,
    /// Duration of the query
    pub duration: Duration,
    /// Whether the query hit the cache
    pub cache_hit: bool,
}

/// Optimization recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recommendation {
    /// Priority of this recommendation
    pub priority: RecommendationPriority,
    /// Category
    pub category: RecommendationCategory,
    /// Description
    pub description: String,
    /// Estimated impact
    pub impact: String,
}

/// Recommendation priority
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecommendationPriority {
    /// Critical - address immediately
    Critical,
    /// High priority
    High,
    /// Medium priority
    Medium,
    /// Low priority
    Low,
}

/// Recommendation category
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecommendationCategory {
    /// Performance optimization
    Performance,
    /// Caching strategy
    Caching,
    /// Resource efficiency
    Efficiency,
    /// Reliability
    Reliability,
}

#[cfg(test)]
mod tests {
    use super::*;
    use ipfrs_core::Block;

    #[test]
    fn test_query_optimizer_basic() {
        let optimizer = QueryOptimizer::new();

        let block = Block::new(vec![0u8; 1024].into()).unwrap();
        let cids = vec![*block.cid(); 100];

        let plan = optimizer.optimize_batch_get(&cids);
        assert!(plan.batch_size > 0);
        assert!(plan.estimated_duration_us > 0);
    }

    #[test]
    fn test_optimize_empty_query() {
        let optimizer = QueryOptimizer::new();
        let plan = optimizer.optimize_batch_get(&[]);

        assert_eq!(plan.batch_size, 0);
        assert_eq!(plan.estimated_duration_us, 0);
        assert_eq!(plan.strategy, QueryStrategy::Sequential);
    }

    #[test]
    fn test_optimize_large_query() {
        let optimizer = QueryOptimizer::new();
        let block = Block::new(vec![0u8; 1024].into()).unwrap();
        let cids = vec![*block.cid(); 1000];

        let plan = optimizer.optimize_batch_get(&cids);
        assert_eq!(plan.strategy, QueryStrategy::ParallelBatch);
        assert!(plan.use_parallel);
    }

    #[test]
    fn test_optimize_streaming_query() {
        let config = OptimizerConfig {
            streaming_threshold_bytes: 1024,
            ..OptimizerConfig::default()
        }; // Very low threshold for testing

        let mut optimizer = QueryOptimizer::with_config(config);
        optimizer.update_stats(2048, 0.5); // Set avg block size to ensure streaming threshold is met

        let block = Block::new(vec![0u8; 1024].into()).unwrap();
        let cids = vec![*block.cid(); 100];

        let plan = optimizer.optimize_batch_get(&cids);
        assert_eq!(plan.strategy, QueryStrategy::Streaming);
    }

    #[test]
    fn test_optimize_batch_put() {
        let optimizer = QueryOptimizer::new();
        let plan = optimizer.optimize_batch_put(100, 100 * 1024);

        assert!(plan.batch_size > 0);
        assert!(plan.estimated_duration_us > 0);
    }

    #[test]
    fn test_pattern_analysis() {
        let optimizer = QueryOptimizer::new();
        let block = Block::new(vec![0u8; 1024].into()).unwrap();
        let cid = *block.cid();

        // Create log with repeated accesses
        let log = vec![
            QueryLogEntry {
                cids: vec![cid],
                duration: Duration::from_millis(10),
                cache_hit: false,
            };
            10
        ];

        let recommendations = optimizer.analyze_patterns(&log);
        assert!(!recommendations.is_empty());
    }

    #[test]
    fn test_update_stats() {
        let mut optimizer = QueryOptimizer::new();
        optimizer.update_stats(1024, 0.9);

        assert_eq!(optimizer.stats.avg_block_size, 1024);
        assert_eq!(optimizer.stats.cache_hit_rate, 0.9);
        assert_eq!(optimizer.stats.query_count, 1);
    }

    #[test]
    fn test_cache_first_strategy() {
        let mut optimizer = QueryOptimizer::new();
        optimizer.update_stats(1024, 0.95); // High cache hit rate

        let block = Block::new(vec![0u8; 1024].into()).unwrap();
        let cids = vec![*block.cid(); 50]; // Below parallel threshold

        let plan = optimizer.optimize_batch_get(&cids);
        assert_eq!(plan.strategy, QueryStrategy::CacheFirst);
    }
}
