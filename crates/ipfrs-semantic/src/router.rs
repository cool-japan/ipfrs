//! Semantic router for content discovery
//!
//! This module provides semantic routing capabilities that combine
//! CID-based lookups with vector similarity search for intelligent
//! content discovery.

use crate::hnsw::{DistanceMetric, SearchResult, VectorIndex};
use ipfrs_core::{Cid, Result};
use lru::LruCache;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::num::NonZeroUsize;
use std::sync::{Arc, RwLock};

/// Configuration for semantic router
#[derive(Debug, Clone)]
pub struct RouterConfig {
    /// Vector dimension for embeddings
    pub dimension: usize,
    /// Distance metric to use
    pub metric: DistanceMetric,
    /// Maximum connections per HNSW layer
    pub max_connections: usize,
    /// HNSW construction parameter
    pub ef_construction: usize,
    /// Default search parameter
    pub ef_search: usize,
    /// Query result cache size (number of queries to cache)
    pub cache_size: usize,
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            dimension: 768, // Common dimension for sentence transformers
            metric: DistanceMetric::Cosine,
            max_connections: 16,
            ef_construction: 200,
            ef_search: 50,
            cache_size: 1000, // Cache up to 1000 recent queries
        }
    }
}

impl RouterConfig {
    /// Create configuration optimized for low latency queries
    ///
    /// Best for: Real-time applications, interactive search, chat systems
    /// Trade-offs: Slightly lower recall (~90%), faster queries
    ///
    /// # Example
    /// ```
    /// use ipfrs_semantic::RouterConfig;
    ///
    /// let config = RouterConfig::low_latency(768);
    /// assert_eq!(config.dimension, 768);
    /// // Optimized for speed with reasonable accuracy
    /// ```
    pub fn low_latency(dimension: usize) -> Self {
        Self {
            dimension,
            metric: DistanceMetric::Cosine,
            max_connections: 12,
            ef_construction: 150,
            ef_search: 32,
            cache_size: 2000, // Larger cache for frequently accessed queries
        }
    }

    /// Create configuration optimized for high recall (accuracy)
    ///
    /// Best for: Research applications, critical retrieval, high-quality recommendations
    /// Trade-offs: Slower queries, higher memory usage
    ///
    /// # Example
    /// ```
    /// use ipfrs_semantic::RouterConfig;
    ///
    /// let config = RouterConfig::high_recall(768);
    /// assert_eq!(config.dimension, 768);
    /// // Optimized for accuracy with acceptable latency
    /// ```
    pub fn high_recall(dimension: usize) -> Self {
        Self {
            dimension,
            metric: DistanceMetric::Cosine,
            max_connections: 32,
            ef_construction: 400,
            ef_search: 200,
            cache_size: 1000,
        }
    }

    /// Create configuration optimized for memory efficiency
    ///
    /// Best for: Edge devices, constrained environments, large datasets with limited RAM
    /// Trade-offs: Lower recall (~85%), smaller cache
    ///
    /// # Example
    /// ```
    /// use ipfrs_semantic::RouterConfig;
    ///
    /// let config = RouterConfig::memory_efficient(384);
    /// assert_eq!(config.dimension, 384);
    /// // Smaller connections and cache for low memory footprint
    /// ```
    pub fn memory_efficient(dimension: usize) -> Self {
        Self {
            dimension,
            metric: DistanceMetric::Cosine,
            max_connections: 8,
            ef_construction: 100,
            ef_search: 50,
            cache_size: 500, // Smaller cache to save memory
        }
    }

    /// Create configuration optimized for large-scale datasets (100k+ vectors)
    ///
    /// Best for: Production systems, large knowledge bases, document collections
    /// Trade-offs: Higher memory usage, optimized for throughput
    ///
    /// # Example
    /// ```
    /// use ipfrs_semantic::RouterConfig;
    ///
    /// let config = RouterConfig::large_scale(768);
    /// assert_eq!(config.dimension, 768);
    /// // Balanced for large datasets with good performance
    /// ```
    pub fn large_scale(dimension: usize) -> Self {
        Self {
            dimension,
            metric: DistanceMetric::Cosine,
            max_connections: 24,
            ef_construction: 300,
            ef_search: 100,
            cache_size: 5000, // Larger cache for diverse queries
        }
    }

    /// Create configuration for balanced performance (alias for default)
    ///
    /// Best for: General purpose, getting started, typical applications
    /// Trade-offs: Balanced recall (~95%) and latency
    ///
    /// # Example
    /// ```
    /// use ipfrs_semantic::RouterConfig;
    ///
    /// let config = RouterConfig::balanced(768);
    /// assert_eq!(config.dimension, 768);
    /// // Good all-around configuration
    /// ```
    pub fn balanced(dimension: usize) -> Self {
        Self {
            dimension,
            metric: DistanceMetric::Cosine,
            max_connections: 16,
            ef_construction: 200,
            ef_search: 50,
            cache_size: 1000,
        }
    }

    /// Create configuration with custom distance metric
    ///
    /// # Example
    /// ```
    /// use ipfrs_semantic::{RouterConfig, DistanceMetric};
    ///
    /// let config = RouterConfig::with_metric(768, DistanceMetric::L2);
    /// assert_eq!(config.dimension, 768);
    /// ```
    pub fn with_metric(dimension: usize, metric: DistanceMetric) -> Self {
        Self {
            dimension,
            metric,
            ..Self::balanced(dimension)
        }
    }

    /// Set the query result cache size
    ///
    /// # Example
    /// ```
    /// use ipfrs_semantic::RouterConfig;
    ///
    /// let config = RouterConfig::balanced(768).with_cache_size(5000);
    /// assert_eq!(config.cache_size, 5000);
    /// ```
    pub fn with_cache_size(mut self, size: usize) -> Self {
        self.cache_size = size;
        self
    }

    /// Set the ef_search parameter for query-time search
    ///
    /// Higher values improve recall but increase latency
    ///
    /// # Example
    /// ```
    /// use ipfrs_semantic::RouterConfig;
    ///
    /// let config = RouterConfig::balanced(768).with_ef_search(100);
    /// assert_eq!(config.ef_search, 100);
    /// ```
    pub fn with_ef_search(mut self, ef_search: usize) -> Self {
        self.ef_search = ef_search;
        self
    }
}

/// Query filter for semantic search
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct QueryFilter {
    /// Minimum similarity score threshold
    pub min_score: Option<f32>,
    /// Maximum similarity score threshold (for range queries)
    pub max_score: Option<f32>,
    /// Maximum number of results
    pub max_results: Option<usize>,
    /// Specific CID prefix to filter by
    pub cid_prefix: Option<String>,
}

impl Default for QueryFilter {
    fn default() -> Self {
        Self {
            min_score: None,
            max_score: None,
            max_results: Some(10),
            cid_prefix: None,
        }
    }
}

impl QueryFilter {
    /// Create a range filter for scores
    pub fn range(min: f32, max: f32) -> Self {
        Self {
            min_score: Some(min),
            max_score: Some(max),
            max_results: None,
            cid_prefix: None,
        }
    }

    /// Create a threshold filter (minimum score)
    pub fn threshold(min: f32) -> Self {
        Self {
            min_score: Some(min),
            max_score: None,
            max_results: None,
            cid_prefix: None,
        }
    }

    /// Create a prefix filter (CID prefix matching)
    pub fn prefix(prefix: String) -> Self {
        Self {
            min_score: None,
            max_score: None,
            max_results: None,
            cid_prefix: Some(prefix),
        }
    }

    /// Combine with another filter (AND operation)
    pub fn and(mut self, other: QueryFilter) -> Self {
        if let Some(min) = other.min_score {
            self.min_score = Some(self.min_score.unwrap_or(f32::MIN).max(min));
        }
        if let Some(max) = other.max_score {
            self.max_score = Some(self.max_score.unwrap_or(f32::MAX).min(max));
        }
        if let Some(max_results) = other.max_results {
            self.max_results = Some(self.max_results.unwrap_or(usize::MAX).min(max_results));
        }
        if other.cid_prefix.is_some() {
            self.cid_prefix = other.cid_prefix;
        }
        self
    }

    /// Set maximum results
    pub fn limit(mut self, max: usize) -> Self {
        self.max_results = Some(max);
        self
    }
}

/// Cache key for query results
type QueryCacheKey = u64;

/// Semantic router combining CID-based and vector-based search
///
/// Provides intelligent content discovery through vector similarity
/// search over content embeddings.
pub struct SemanticRouter {
    /// Vector index for semantic search
    index: Arc<RwLock<VectorIndex>>,
    /// Router configuration
    config: RouterConfig,
    /// Query result cache (LRU)
    query_cache: Arc<RwLock<LruCache<QueryCacheKey, Vec<SearchResult>>>>,
}

impl SemanticRouter {
    /// Create a new semantic router with the given configuration
    pub fn new(config: RouterConfig) -> Result<Self> {
        let index = VectorIndex::new(
            config.dimension,
            config.metric,
            config.max_connections,
            config.ef_construction,
        )?;

        let cache_size =
            NonZeroUsize::new(config.cache_size).unwrap_or(NonZeroUsize::new(1000).unwrap());
        let query_cache = LruCache::new(cache_size);

        Ok(Self {
            index: Arc::new(RwLock::new(index)),
            config,
            query_cache: Arc::new(RwLock::new(query_cache)),
        })
    }

    /// Create a new router with default configuration
    pub fn with_defaults() -> Result<Self> {
        Self::new(RouterConfig::default())
    }

    /// Add content with its embedding to the router
    ///
    /// # Arguments
    /// * `cid` - Content identifier
    /// * `embedding` - Vector embedding of the content
    pub fn add(&self, cid: &Cid, embedding: &[f32]) -> Result<()> {
        self.index.write().unwrap().insert(cid, embedding)
    }

    /// Add multiple content items in batch
    ///
    /// More efficient than adding one by one
    ///
    /// # Arguments
    /// * `items` - Vector of (CID, embedding) pairs
    pub fn add_batch(&self, items: &[(Cid, Vec<f32>)]) -> Result<()> {
        self.index.write().unwrap().insert_batch(items)
    }

    /// Remove content from the router
    pub fn remove(&self, cid: &Cid) -> Result<()> {
        self.index.write().unwrap().delete(cid)
    }

    /// Check if content exists in the router
    pub fn contains(&self, cid: &Cid) -> bool {
        self.index.read().unwrap().contains(cid)
    }

    /// Query for content by semantic similarity
    ///
    /// # Arguments
    /// * `query_embedding` - Query vector
    /// * `k` - Number of results to return
    pub async fn query(&self, query_embedding: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        self.query_with_filter(query_embedding, k, QueryFilter::default())
            .await
    }

    /// Query with auto-tuned ef_search parameter
    ///
    /// Automatically determines the optimal ef_search based on k and index size
    ///
    /// # Arguments
    /// * `query_embedding` - Query vector
    /// * `k` - Number of results to return
    pub async fn query_auto(&self, query_embedding: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        let optimal_ef_search = self.index.read().unwrap().compute_optimal_ef_search(k);
        self.query_with_ef(query_embedding, k, optimal_ef_search)
            .await
    }

    /// Query with custom ef_search parameter
    ///
    /// # Arguments
    /// * `query_embedding` - Query vector
    /// * `k` - Number of results to return
    /// * `ef_search` - Search parameter (higher = more accurate but slower)
    pub async fn query_with_ef(
        &self,
        query_embedding: &[f32],
        k: usize,
        ef_search: usize,
    ) -> Result<Vec<SearchResult>> {
        // Generate cache key
        let cache_key = Self::compute_cache_key(query_embedding, k, &QueryFilter::default());

        // Try cache first
        if let Some(cached) = self.query_cache.write().unwrap().get(&cache_key) {
            return Ok(cached.clone());
        }

        // Search with custom ef_search
        let results = self
            .index
            .read()
            .unwrap()
            .search(query_embedding, k, ef_search)?;

        // Cache the results
        self.query_cache
            .write()
            .unwrap()
            .put(cache_key, results.clone());

        Ok(results)
    }

    /// Query with filtering options
    ///
    /// # Arguments
    /// * `query_embedding` - Query vector
    /// * `k` - Number of results to return
    /// * `filter` - Query filter options
    pub async fn query_with_filter(
        &self,
        query_embedding: &[f32],
        k: usize,
        filter: QueryFilter,
    ) -> Result<Vec<SearchResult>> {
        // Generate cache key from query parameters
        let cache_key = Self::compute_cache_key(query_embedding, k, &filter);

        // Try cache first (only if no filtering, as filtered results may vary)
        if filter.min_score.is_none() && filter.cid_prefix.is_none() {
            if let Some(cached) = self.query_cache.write().unwrap().get(&cache_key) {
                return Ok(cached.clone());
            }
        }

        // Determine actual k to fetch (might need more if filtering)
        let fetch_k = if filter.min_score.is_some() || filter.cid_prefix.is_some() {
            k * 2 // Fetch more to account for filtering
        } else {
            k
        };

        // Search the vector index
        let mut results =
            self.index
                .read()
                .unwrap()
                .search(query_embedding, fetch_k, self.config.ef_search)?;

        // Apply filters
        if let Some(min_score) = filter.min_score {
            results.retain(|r| r.score >= min_score);
        }

        if let Some(max_score) = filter.max_score {
            results.retain(|r| r.score <= max_score);
        }

        if let Some(ref prefix) = filter.cid_prefix {
            results.retain(|r| r.cid.to_string().starts_with(prefix));
        }

        // Apply max results limit
        if let Some(max_results) = filter.max_results {
            results.truncate(max_results);
        }

        // Cache the results (only unfiltered queries)
        if filter.min_score.is_none() && filter.cid_prefix.is_none() {
            self.query_cache
                .write()
                .unwrap()
                .put(cache_key, results.clone());
        }

        Ok(results)
    }

    /// Compute a cache key from query parameters
    fn compute_cache_key(embedding: &[f32], k: usize, filter: &QueryFilter) -> QueryCacheKey {
        let mut hasher = DefaultHasher::new();

        // Hash embedding values (sample to avoid too much computation)
        for (i, &val) in embedding.iter().enumerate().step_by(8) {
            (i, (val * 1000.0) as i32).hash(&mut hasher);
        }

        k.hash(&mut hasher);
        filter.max_results.hash(&mut hasher);

        hasher.finish()
    }

    /// Clear the query result cache
    pub fn clear_cache(&self) {
        self.query_cache.write().unwrap().clear();
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> CacheStats {
        let cache = self.query_cache.read().unwrap();
        CacheStats {
            size: cache.len(),
            capacity: cache.cap().get(),
        }
    }

    /// Get statistics about the router
    pub fn stats(&self) -> RouterStats {
        let index = self.index.read().unwrap();
        RouterStats {
            num_vectors: index.len(),
            dimension: index.dimension(),
            metric: index.metric(),
        }
    }

    /// Get optimization recommendations
    ///
    /// Returns recommended HNSW parameters for the current index size
    pub fn optimization_recommendations(&self) -> OptimizationRecommendations {
        let index = self.index.read().unwrap();
        let (m, ef_construction) = index.compute_optimal_parameters();

        OptimizationRecommendations {
            recommended_m: m,
            recommended_ef_construction: ef_construction,
            current_size: index.len(),
        }
    }

    /// Save the semantic index to a file
    ///
    /// Serializes the entire HNSW index including all vectors and CID mappings
    /// to a file for later loading.
    ///
    /// # Arguments
    /// * `path` - Path to save the index file
    pub async fn save_index<P: AsRef<std::path::Path>>(&self, path: P) -> Result<()> {
        self.index.read().unwrap().save(path.as_ref())
    }

    /// Load a semantic index from a file
    ///
    /// Loads a previously saved HNSW index from disk, replacing the current index.
    ///
    /// # Arguments
    /// * `path` - Path to the saved index file
    pub async fn load_index<P: AsRef<std::path::Path>>(&self, path: P) -> Result<()> {
        let loaded_index = VectorIndex::load(path.as_ref())?;
        *self.index.write().unwrap() = loaded_index;
        // Clear cache after loading new index
        self.clear_cache();
        Ok(())
    }

    /// Clear all content from the router
    pub fn clear(&self) -> Result<()> {
        // Create new empty index
        let new_index = VectorIndex::new(
            self.config.dimension,
            self.config.metric,
            self.config.max_connections,
            self.config.ef_construction,
        )?;

        *self.index.write().unwrap() = new_index;

        // Clear the cache as well
        self.query_cache.write().unwrap().clear();

        Ok(())
    }

    /// Query with aggregations
    ///
    /// Returns both results and aggregated statistics
    ///
    /// # Arguments
    /// * `query_embedding` - Query vector
    /// * `k` - Number of results to return
    /// * `filter` - Query filter options
    pub async fn query_with_aggregations(
        &self,
        query_embedding: &[f32],
        k: usize,
        filter: QueryFilter,
    ) -> Result<(Vec<SearchResult>, SearchAggregations)> {
        let results = self.query_with_filter(query_embedding, k, filter).await?;
        let aggregations = SearchAggregations::from_results(&results);
        Ok((results, aggregations))
    }

    /// Batch query for multiple embeddings at once
    ///
    /// More efficient than querying one by one due to parallelization
    /// and amortized overhead.
    ///
    /// # Arguments
    /// * `query_embeddings` - Multiple query vectors
    /// * `k` - Number of results to return per query
    ///
    /// # Returns
    /// Vector of search results, one for each query in the same order
    pub async fn query_batch(
        &self,
        query_embeddings: &[Vec<f32>],
        k: usize,
    ) -> Result<Vec<Vec<SearchResult>>> {
        self.query_batch_with_filter(query_embeddings, k, QueryFilter::default())
            .await
    }

    /// Batch query with filtering options
    ///
    /// Processes multiple queries in parallel with filtering applied to each.
    ///
    /// # Arguments
    /// * `query_embeddings` - Multiple query vectors
    /// * `k` - Number of results to return per query
    /// * `filter` - Query filter options (applied to all queries)
    ///
    /// # Returns
    /// Vector of search results, one for each query in the same order
    pub async fn query_batch_with_filter(
        &self,
        query_embeddings: &[Vec<f32>],
        k: usize,
        filter: QueryFilter,
    ) -> Result<Vec<Vec<SearchResult>>> {
        use rayon::prelude::*;

        // Process queries in parallel using rayon
        let results: Result<Vec<Vec<SearchResult>>> = query_embeddings
            .par_iter()
            .map(|embedding| {
                // Generate cache key
                let cache_key = Self::compute_cache_key(embedding, k, &filter);

                // Try cache first (only if no filtering)
                if filter.min_score.is_none() && filter.cid_prefix.is_none() {
                    if let Some(cached) = self.query_cache.write().unwrap().get(&cache_key) {
                        return Ok(cached.clone());
                    }
                }

                // Determine actual k to fetch (might need more if filtering)
                let fetch_k = if filter.min_score.is_some() || filter.cid_prefix.is_some() {
                    k * 2 // Fetch more to account for filtering
                } else {
                    k
                };

                // Search the vector index
                let mut results =
                    self.index
                        .read()
                        .unwrap()
                        .search(embedding, fetch_k, self.config.ef_search)?;

                // Apply filters
                if let Some(min_score) = filter.min_score {
                    results.retain(|r| r.score >= min_score);
                }

                if let Some(max_score) = filter.max_score {
                    results.retain(|r| r.score <= max_score);
                }

                if let Some(ref prefix) = filter.cid_prefix {
                    results.retain(|r| r.cid.to_string().starts_with(prefix));
                }

                // Apply max results limit
                if let Some(max_results) = filter.max_results {
                    results.truncate(max_results);
                }

                // Cache the results (only unfiltered queries)
                if filter.min_score.is_none() && filter.cid_prefix.is_none() {
                    self.query_cache
                        .write()
                        .unwrap()
                        .put(cache_key, results.clone());
                }

                Ok(results)
            })
            .collect();

        results
    }

    /// Batch query with custom ef_search parameter
    ///
    /// Processes multiple queries in parallel with custom search parameter.
    ///
    /// # Arguments
    /// * `query_embeddings` - Multiple query vectors
    /// * `k` - Number of results to return per query
    /// * `ef_search` - Search parameter (higher = more accurate but slower)
    ///
    /// # Returns
    /// Vector of search results, one for each query in the same order
    pub async fn query_batch_with_ef(
        &self,
        query_embeddings: &[Vec<f32>],
        k: usize,
        ef_search: usize,
    ) -> Result<Vec<Vec<SearchResult>>> {
        use rayon::prelude::*;

        // Process queries in parallel using rayon
        let results: Result<Vec<Vec<SearchResult>>> = query_embeddings
            .par_iter()
            .map(|embedding| {
                // Generate cache key
                let cache_key = Self::compute_cache_key(embedding, k, &QueryFilter::default());

                // Try cache first
                if let Some(cached) = self.query_cache.write().unwrap().get(&cache_key) {
                    return Ok(cached.clone());
                }

                // Search with custom ef_search
                let results = self.index.read().unwrap().search(embedding, k, ef_search)?;

                // Cache the results
                self.query_cache
                    .write()
                    .unwrap()
                    .put(cache_key, results.clone());

                Ok(results)
            })
            .collect();

        results
    }

    /// Get batch query statistics
    ///
    /// Returns aggregated statistics across a batch of queries
    pub fn batch_stats(&self, batch_results: &[Vec<SearchResult>]) -> BatchStats {
        let total_queries = batch_results.len();
        let total_results: usize = batch_results.iter().map(|r| r.len()).sum();
        let avg_results_per_query = if total_queries > 0 {
            total_results as f32 / total_queries as f32
        } else {
            0.0
        };

        let all_scores: Vec<f32> = batch_results
            .iter()
            .flat_map(|results| results.iter().map(|r| r.score))
            .collect();

        let avg_score = if !all_scores.is_empty() {
            all_scores.iter().sum::<f32>() / all_scores.len() as f32
        } else {
            0.0
        };

        BatchStats {
            total_queries,
            total_results,
            avg_results_per_query,
            avg_score,
        }
    }
}

/// Router statistics
#[derive(Debug, Clone)]
pub struct RouterStats {
    /// Number of indexed vectors
    pub num_vectors: usize,
    /// Vector dimension
    pub dimension: usize,
    /// Distance metric
    pub metric: DistanceMetric,
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    /// Current number of cached entries
    pub size: usize,
    /// Maximum cache capacity
    pub capacity: usize,
}

/// Batch query statistics
#[derive(Debug, Clone)]
pub struct BatchStats {
    /// Total number of queries in batch
    pub total_queries: usize,
    /// Total number of results across all queries
    pub total_results: usize,
    /// Average number of results per query
    pub avg_results_per_query: f32,
    /// Average similarity score across all results
    pub avg_score: f32,
}

/// Optimization recommendations for HNSW index
#[derive(Debug, Clone)]
pub struct OptimizationRecommendations {
    /// Recommended M parameter (max connections per layer)
    pub recommended_m: usize,
    /// Recommended ef_construction parameter
    pub recommended_ef_construction: usize,
    /// Current index size
    pub current_size: usize,
}

/// Search result aggregations
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SearchAggregations {
    /// Total number of results
    pub total_count: usize,
    /// Average similarity score
    pub avg_score: f32,
    /// Minimum score in results
    pub min_score: f32,
    /// Maximum score in results
    pub max_score: f32,
    /// Score distribution by buckets
    pub score_buckets: Vec<ScoreBucket>,
}

/// Score bucket for distribution analysis
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ScoreBucket {
    /// Bucket range (min, max)
    pub range: (f32, f32),
    /// Count of results in this bucket
    pub count: usize,
}

impl SearchAggregations {
    /// Compute aggregations from search results
    pub fn from_results(results: &[SearchResult]) -> Self {
        if results.is_empty() {
            return Self {
                total_count: 0,
                avg_score: 0.0,
                min_score: 0.0,
                max_score: 0.0,
                score_buckets: Vec::new(),
            };
        }

        let total_count = results.len();
        let sum: f32 = results.iter().map(|r| r.score).sum();
        let avg_score = sum / total_count as f32;
        let min_score = results
            .iter()
            .map(|r| r.score)
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        let max_score = results
            .iter()
            .map(|r| r.score)
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();

        // Create 10 buckets for score distribution
        let bucket_count = 10;
        let range = max_score - min_score;
        let bucket_size = if range > 0.0 {
            range / bucket_count as f32
        } else {
            1.0
        };

        let mut buckets = vec![0; bucket_count];
        for result in results {
            let bucket_idx = if range > 0.0 {
                ((result.score - min_score) / bucket_size).floor() as usize
            } else {
                0
            };
            let bucket_idx = bucket_idx.min(bucket_count - 1);
            buckets[bucket_idx] += 1;
        }

        let score_buckets = buckets
            .into_iter()
            .enumerate()
            .map(|(i, count)| ScoreBucket {
                range: (
                    min_score + i as f32 * bucket_size,
                    min_score + (i + 1) as f32 * bucket_size,
                ),
                count,
            })
            .collect();

        Self {
            total_count,
            avg_score,
            min_score,
            max_score,
            score_buckets,
        }
    }
}

impl Default for SemanticRouter {
    fn default() -> Self {
        Self::with_defaults().expect("Failed to create default SemanticRouter")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_router_creation() {
        let router = SemanticRouter::with_defaults();
        assert!(router.is_ok());
    }

    #[tokio::test]
    async fn test_add_and_query() {
        let router = SemanticRouter::with_defaults().unwrap();

        let cid1 = "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi"
            .parse::<Cid>()
            .unwrap();
        let embedding1 = vec![0.5; 768];

        router.add(&cid1, &embedding1).unwrap();

        let results = router.query(&embedding1, 1).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].cid, cid1);
    }

    #[tokio::test]
    async fn test_filtering() {
        let router = SemanticRouter::with_defaults().unwrap();

        let cid1 = "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi"
            .parse::<Cid>()
            .unwrap();
        let embedding1 = vec![0.5; 768];

        router.add(&cid1, &embedding1).unwrap();

        // Query with score filter
        let filter = QueryFilter {
            min_score: Some(0.9),
            max_score: None,
            max_results: Some(10),
            cid_prefix: None,
        };

        let results = router
            .query_with_filter(&embedding1, 10, filter)
            .await
            .unwrap();

        // Should find the exact match
        assert!(!results.is_empty());
    }

    #[tokio::test]
    async fn test_integration_with_blocks() {
        use bytes::Bytes;
        use ipfrs_core::Block;

        // Create router with dimension 3 for this test
        let router = SemanticRouter::new(RouterConfig {
            dimension: 3,
            ..Default::default()
        })
        .unwrap();

        // Create test blocks
        let data1 = Bytes::from_static(b"Hello, semantic search!");
        let data2 = Bytes::from_static(b"Goodbye, semantic search!");
        let data3 = Bytes::from_static(b"Hello, world!");

        let block1 = Block::new(data1).unwrap();
        let block2 = Block::new(data2).unwrap();
        let block3 = Block::new(data3).unwrap();

        // Generate simple embeddings based on content
        // In real use, these would come from an embedding model
        let embedding1 = vec![1.0, 0.0, 0.0]; // "Hello" cluster
        let embedding2 = vec![0.0, 1.0, 0.0]; // "Goodbye" cluster
        let embedding3 = vec![0.9, 0.1, 0.0]; // Close to "Hello" cluster

        // Index blocks with their embeddings
        router.add(block1.cid(), &embedding1).unwrap();
        router.add(block2.cid(), &embedding2).unwrap();
        router.add(block3.cid(), &embedding3).unwrap();

        // Query for blocks similar to "Hello"
        let query_embedding = vec![1.0, 0.0, 0.0];
        let results = router.query(&query_embedding, 2).await.unwrap();

        // Should return block1 and block3 (both in "Hello" cluster)
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].cid, *block1.cid());
    }

    #[tokio::test]
    async fn test_integration_with_tensor_metadata() {
        use ipfrs_core::{TensorDtype, TensorMetadata, TensorShape};

        let router = SemanticRouter::new(RouterConfig {
            dimension: 2,
            ..Default::default()
        })
        .unwrap();

        // Create tensor metadata
        let shape1 = TensorShape::new(vec![1, 768]);
        let mut metadata1 = TensorMetadata::new(shape1, TensorDtype::F32);
        metadata1.name = Some("vision_embedding".to_string());
        metadata1
            .metadata
            .insert("semantic_tag".to_string(), "vision".to_string());

        let shape2 = TensorShape::new(vec![1, 768]);
        let mut metadata2 = TensorMetadata::new(shape2, TensorDtype::F32);
        metadata2.name = Some("text_embedding".to_string());
        metadata2
            .metadata
            .insert("semantic_tag".to_string(), "text".to_string());

        // Generate embeddings for tensor types
        let vision_embedding = vec![1.0, 0.0];
        let text_embedding = vec![0.0, 1.0];

        // Create CIDs for metadata (in real use, these would be the tensor CIDs)
        let cid1 = "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi"
            .parse::<Cid>()
            .unwrap();
        let cid2 = "bafybeibazl2z6vqxqqzmhmvx2hfpxqtwggqgbbyy3sxkq4vzq6cqsvwbjy"
            .parse::<Cid>()
            .unwrap();

        // Index tensors by their semantic embeddings
        router.add(&cid1, &vision_embedding).unwrap();
        router.add(&cid2, &text_embedding).unwrap();

        // Search for vision-type tensors
        let results = router.query(&vision_embedding, 1).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].cid, cid1);
    }

    #[tokio::test]
    async fn test_large_scale_indexing() {
        use rand::Rng;

        let dimension = 128;

        // Create router with dimension 128 for this test
        let router = SemanticRouter::new(RouterConfig {
            dimension,
            ..Default::default()
        })
        .unwrap();

        // Generate 1000 random embeddings and index them
        let mut rng = rand::rng();
        let num_items = 1000;

        let mut indexed_cids = Vec::new();

        for i in 0..num_items {
            // Generate unique CID
            use multihash_codetable::{Code, MultihashDigest};
            let data = format!("large_scale_test_{}", i);
            let hash = Code::Sha2_256.digest(data.as_bytes());
            let cid = Cid::new_v1(0x55, hash);

            // Generate random embedding
            let embedding: Vec<f32> = (0..dimension)
                .map(|_| rng.random_range(-1.0..1.0))
                .collect();

            router.add(&cid, &embedding).unwrap();
            indexed_cids.push((cid, embedding));
        }

        // Verify index size
        let stats = router.stats();
        assert_eq!(stats.num_vectors, num_items);

        // Test query on a known embedding
        let (test_cid, test_embedding) = &indexed_cids[42];
        let results = router.query(test_embedding, 1).await.unwrap();

        // Should return the exact match as the top result
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].cid, *test_cid);
    }

    #[tokio::test]
    async fn test_cache_effectiveness() {
        let router = SemanticRouter::with_defaults().unwrap();

        let cid1 = "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi"
            .parse::<Cid>()
            .unwrap();
        let embedding1 = vec![0.5; 768];

        router.add(&cid1, &embedding1).unwrap();

        // Perform same query multiple times
        for _ in 0..10 {
            let _ = router.query(&embedding1, 1).await.unwrap();
        }

        // Check cache stats - should have cached the query
        let cache_stats = router.cache_stats();
        assert_eq!(cache_stats.size, 1, "Cache should have 1 unique query");
        assert!(cache_stats.capacity > 0, "Cache should have capacity");
    }

    #[tokio::test]
    async fn test_batch_query() {
        use rand::Rng;

        let dimension = 128;

        // Create router with dimension 128 for this test
        let router = SemanticRouter::new(RouterConfig {
            dimension,
            ..Default::default()
        })
        .unwrap();

        // Generate and index 100 random embeddings
        let mut rng = rand::rng();
        let num_items = 100;

        for i in 0..num_items {
            // Generate unique CID
            use multihash_codetable::{Code, MultihashDigest};
            let data = format!("batch_test_{}", i);
            let hash = Code::Sha2_256.digest(data.as_bytes());
            let cid = Cid::new_v1(0x55, hash);

            // Generate random embedding
            let embedding: Vec<f32> = (0..dimension)
                .map(|_| rng.random_range(-1.0..1.0))
                .collect();

            router.add(&cid, &embedding).unwrap();
        }

        // Create batch of query embeddings
        let batch_size = 10;
        let query_batch: Vec<Vec<f32>> = (0..batch_size)
            .map(|_| {
                (0..dimension)
                    .map(|_| rng.random_range(-1.0..1.0))
                    .collect()
            })
            .collect();

        // Execute batch query
        let results = router.query_batch(&query_batch, 5).await.unwrap();

        // Verify results
        assert_eq!(results.len(), batch_size);
        for result in &results {
            assert!(!result.is_empty());
            assert!(result.len() <= 5);
        }

        // Get batch statistics
        let stats = router.batch_stats(&results);
        assert_eq!(stats.total_queries, batch_size);
        assert!(stats.total_results > 0);
        assert!(stats.avg_results_per_query > 0.0);
    }

    #[tokio::test]
    async fn test_batch_query_with_filter() {
        use rand::Rng;

        let dimension = 64;

        let router = SemanticRouter::new(RouterConfig {
            dimension,
            ..Default::default()
        })
        .unwrap();

        // Generate and index embeddings
        let mut rng = rand::rng();
        let num_items = 50;

        for i in 0..num_items {
            use multihash_codetable::{Code, MultihashDigest};
            let data = format!("filter_batch_test_{}", i);
            let hash = Code::Sha2_256.digest(data.as_bytes());
            let cid = Cid::new_v1(0x55, hash);

            let embedding: Vec<f32> = (0..dimension)
                .map(|_| rng.random_range(-1.0..1.0))
                .collect();

            router.add(&cid, &embedding).unwrap();
        }

        // Create batch queries
        let batch_size = 5;
        let query_batch: Vec<Vec<f32>> = (0..batch_size)
            .map(|_| {
                (0..dimension)
                    .map(|_| rng.random_range(-1.0..1.0))
                    .collect()
            })
            .collect();

        // Execute batch query with filter
        let filter = QueryFilter {
            min_score: Some(0.0),
            max_results: Some(3),
            ..Default::default()
        };

        let results = router
            .query_batch_with_filter(&query_batch, 5, filter)
            .await
            .unwrap();

        // Verify results
        assert_eq!(results.len(), batch_size);
        for result in &results {
            assert!(result.len() <= 3); // Max results filter applied
        }
    }

    #[tokio::test]
    async fn test_batch_query_with_ef() {
        use rand::Rng;

        let dimension = 64;

        let router = SemanticRouter::new(RouterConfig {
            dimension,
            ..Default::default()
        })
        .unwrap();

        // Generate and index embeddings
        let mut rng = rand::rng();
        let num_items = 50;

        for i in 0..num_items {
            use multihash_codetable::{Code, MultihashDigest};
            let data = format!("ef_batch_test_{}", i);
            let hash = Code::Sha2_256.digest(data.as_bytes());
            let cid = Cid::new_v1(0x55, hash);

            let embedding: Vec<f32> = (0..dimension)
                .map(|_| rng.random_range(-1.0..1.0))
                .collect();

            router.add(&cid, &embedding).unwrap();
        }

        // Create batch queries
        let batch_size = 5;
        let query_batch: Vec<Vec<f32>> = (0..batch_size)
            .map(|_| {
                (0..dimension)
                    .map(|_| rng.random_range(-1.0..1.0))
                    .collect()
            })
            .collect();

        // Execute batch query with custom ef_search
        let results = router
            .query_batch_with_ef(&query_batch, 3, 100)
            .await
            .unwrap();

        // Verify results
        assert_eq!(results.len(), batch_size);
        for result in &results {
            assert!(!result.is_empty());
            assert!(result.len() <= 3);
        }
    }
}
