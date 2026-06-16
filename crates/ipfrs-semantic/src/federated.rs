//! Federated Query Support for Multi-Index Search
//!
//! This module enables querying multiple semantic indices simultaneously,
//! supporting heterogeneous distance metrics and privacy-preserving search.
//!
//! Use cases:
//! - Multi-organization search
//! - Cross-domain semantic search
//! - Privacy-preserving federated learning
//! - Hybrid cloud/edge deployments

use crate::hnsw::{DistanceMetric, SearchResult};
use ipfrs_core::{Cid, Error, Result};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

/// Configuration for federated queries
#[derive(Debug, Clone)]
pub struct FederatedConfig {
    /// Maximum number of concurrent index queries
    pub max_concurrent_queries: usize,
    /// Timeout for each index query in milliseconds
    pub query_timeout_ms: u64,
    /// Enable privacy-preserving query mode
    pub privacy_preserving: bool,
    /// Noise level for differential privacy (0.0 = no noise)
    pub privacy_noise_level: f32,
    /// Result aggregation strategy
    pub aggregation_strategy: AggregationStrategy,
    /// Normalize scores across indices
    pub normalize_scores: bool,
}

impl Default for FederatedConfig {
    fn default() -> Self {
        Self {
            max_concurrent_queries: 10,
            query_timeout_ms: 5000,
            privacy_preserving: false,
            privacy_noise_level: 0.0,
            aggregation_strategy: AggregationStrategy::RankFusion,
            normalize_scores: true,
        }
    }
}

/// Result aggregation strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregationStrategy {
    /// Simple concatenation and re-ranking
    Simple,
    /// Reciprocal rank fusion (recommended for heterogeneous metrics)
    RankFusion,
    /// Score normalization and merging
    ScoreNormalization,
    /// Borda count voting
    BordaCount,
}

/// A queryable index interface
#[async_trait::async_trait]
pub trait QueryableIndex: Send + Sync {
    /// Query the index with an embedding
    async fn query(&self, embedding: &[f32], k: usize) -> Result<Vec<SearchResult>>;

    /// Get the distance metric used by this index
    fn distance_metric(&self) -> DistanceMetric;

    /// Get index identifier
    fn index_id(&self) -> String;

    /// Get index size (number of entries)
    fn size(&self) -> usize;
}

/// Wrapper for VectorIndex to make it queryable
pub struct LocalIndexAdapter {
    index: Arc<RwLock<crate::hnsw::VectorIndex>>,
    index_id: String,
}

impl LocalIndexAdapter {
    /// Create a new local index adapter
    pub fn new(index: Arc<RwLock<crate::hnsw::VectorIndex>>, index_id: String) -> Self {
        Self { index, index_id }
    }
}

#[async_trait::async_trait]
impl QueryableIndex for LocalIndexAdapter {
    async fn query(&self, embedding: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        let index = self.index.read();
        let ef_search = k * 10; // Heuristic
        index.search(embedding, k, ef_search)
    }

    fn distance_metric(&self) -> DistanceMetric {
        let index = self.index.read();
        index.metric()
    }

    fn index_id(&self) -> String {
        self.index_id.clone()
    }

    fn size(&self) -> usize {
        let index = self.index.read();
        index.len()
    }
}

/// Federated search result with index provenance
#[derive(Debug, Clone)]
pub struct FederatedSearchResult {
    /// The CID
    pub cid: Cid,
    /// Aggregated score
    pub score: f32,
    /// Index ID where this result was found
    pub source_index_id: String,
    /// Original rank in source index
    pub source_rank: usize,
    /// Distance metric used by source
    pub source_metric: DistanceMetric,
}

/// Federated query executor
pub struct FederatedQueryExecutor {
    /// Configuration
    config: FederatedConfig,
    /// Registered indices
    indices: Arc<RwLock<HashMap<String, Arc<dyn QueryableIndex>>>>,
    /// Query statistics
    stats: Arc<RwLock<FederatedQueryStats>>,
}

/// Statistics for federated queries
#[derive(Debug, Clone, Default)]
pub struct FederatedQueryStats {
    /// Total queries executed
    pub total_queries: u64,
    /// Total indices queried
    pub total_indices_queried: u64,
    /// Average query latency (ms)
    pub avg_latency_ms: f64,
    /// Average results per query
    pub avg_results_per_query: f64,
    /// Number of timeouts
    pub timeouts: u64,
}

impl FederatedQueryExecutor {
    /// Create a new federated query executor
    pub fn new(config: FederatedConfig) -> Self {
        Self {
            config,
            indices: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(FederatedQueryStats::default())),
        }
    }

    /// Register an index for federated queries
    pub fn register_index(&self, index: Arc<dyn QueryableIndex>) -> Result<()> {
        let index_id = index.index_id();
        let mut indices = self.indices.write();

        if indices.contains_key(&index_id) {
            return Err(Error::InvalidInput(format!(
                "Index '{}' is already registered",
                index_id
            )));
        }

        indices.insert(index_id.clone(), index);
        tracing::info!("Registered index '{}' for federated queries", index_id);
        Ok(())
    }

    /// Unregister an index
    pub fn unregister_index(&self, index_id: &str) -> Result<()> {
        let mut indices = self.indices.write();
        if indices.remove(index_id).is_some() {
            tracing::info!("Unregistered index '{}'", index_id);
            Ok(())
        } else {
            Err(Error::NotFound(format!("Index '{}' not found", index_id)))
        }
    }

    /// Execute a federated query across all registered indices
    pub async fn query(&self, embedding: &[f32], k: usize) -> Result<Vec<FederatedSearchResult>> {
        let start = std::time::Instant::now();

        // Get snapshot of indices
        let indices = {
            let indices_lock = self.indices.read();
            indices_lock
                .iter()
                .map(|(id, idx)| (id.clone(), Arc::clone(idx)))
                .collect::<Vec<_>>()
        };

        if indices.is_empty() {
            return Err(Error::InvalidInput(
                "No indices registered for federated query".to_string(),
            ));
        }

        // Apply privacy noise if enabled
        let query_embedding = if self.config.privacy_preserving {
            self.apply_privacy_noise(embedding)
        } else {
            embedding.to_vec()
        };

        // Query all indices concurrently
        let mut tasks = Vec::new();
        for (index_id, index) in indices {
            let query_emb = query_embedding.clone();
            let task = tokio::spawn(async move {
                let result = index.query(&query_emb, k).await;
                (index_id, index.distance_metric(), result)
            });
            tasks.push(task);
        }

        // Collect results with timeout handling
        let mut all_results = Vec::new();
        let mut indices_queried = 0;
        let mut timeouts = 0;

        for task in tasks {
            match tokio::time::timeout(
                std::time::Duration::from_millis(self.config.query_timeout_ms),
                task,
            )
            .await
            {
                Ok(Ok((index_id, metric, Ok(results)))) => {
                    indices_queried += 1;
                    for (rank, result) in results.into_iter().enumerate() {
                        all_results.push((index_id.clone(), metric, rank, result));
                    }
                }
                Ok(Ok((index_id, _, Err(e)))) => {
                    tracing::warn!("Query failed for index '{}': {:?}", index_id, e);
                }
                Ok(Err(e)) => {
                    tracing::warn!("Task panicked: {:?}", e);
                }
                Err(_) => {
                    timeouts += 1;
                    tracing::warn!("Query timeout for an index");
                }
            }
        }

        // Aggregate results
        let aggregated = self.aggregate_results(all_results, k)?;

        // Update statistics
        let latency = start.elapsed().as_millis() as f64;
        self.update_stats(indices_queried, aggregated.len(), latency, timeouts);

        Ok(aggregated)
    }

    /// Query specific indices only
    pub async fn query_indices(
        &self,
        embedding: &[f32],
        k: usize,
        index_ids: &[String],
    ) -> Result<Vec<FederatedSearchResult>> {
        let start = std::time::Instant::now();

        // Get requested indices
        let indices = {
            let indices_lock = self.indices.read();
            index_ids
                .iter()
                .filter_map(|id| {
                    indices_lock
                        .get(id)
                        .map(|idx| (id.clone(), Arc::clone(idx)))
                })
                .collect::<Vec<_>>()
        };

        if indices.is_empty() {
            return Err(Error::InvalidInput(
                "None of the requested indices are registered".to_string(),
            ));
        }

        // Apply privacy noise if enabled
        let query_embedding = if self.config.privacy_preserving {
            self.apply_privacy_noise(embedding)
        } else {
            embedding.to_vec()
        };

        // Query specified indices concurrently
        let mut tasks = Vec::new();
        for (index_id, index) in indices {
            let query_emb = query_embedding.clone();
            let task = tokio::spawn(async move {
                let result = index.query(&query_emb, k).await;
                (index_id, index.distance_metric(), result)
            });
            tasks.push(task);
        }

        // Collect and aggregate results
        let mut all_results = Vec::new();
        let mut indices_queried = 0;
        let mut timeouts = 0;

        for task in tasks {
            match tokio::time::timeout(
                std::time::Duration::from_millis(self.config.query_timeout_ms),
                task,
            )
            .await
            {
                Ok(Ok((index_id, metric, Ok(results)))) => {
                    indices_queried += 1;
                    for (rank, result) in results.into_iter().enumerate() {
                        all_results.push((index_id.clone(), metric, rank, result));
                    }
                }
                Ok(Ok((index_id, _, Err(e)))) => {
                    tracing::warn!("Query failed for index '{}': {:?}", index_id, e);
                }
                Ok(Err(e)) => {
                    tracing::warn!("Task panicked: {:?}", e);
                }
                Err(_) => {
                    timeouts += 1;
                    tracing::warn!("Query timeout for an index");
                }
            }
        }

        let aggregated = self.aggregate_results(all_results, k)?;

        let latency = start.elapsed().as_millis() as f64;
        self.update_stats(indices_queried, aggregated.len(), latency, timeouts);

        Ok(aggregated)
    }

    /// Apply differential privacy noise to query embedding
    fn apply_privacy_noise(&self, embedding: &[f32]) -> Vec<f32> {
        use rand::RngExt;
        let mut rng = rand::rng();

        embedding
            .iter()
            .map(|&x| {
                let noise = rng.random_range(
                    -self.config.privacy_noise_level..self.config.privacy_noise_level,
                );
                x + noise
            })
            .collect()
    }

    /// Aggregate results from multiple indices
    fn aggregate_results(
        &self,
        results: Vec<(String, DistanceMetric, usize, SearchResult)>,
        k: usize,
    ) -> Result<Vec<FederatedSearchResult>> {
        match self.config.aggregation_strategy {
            AggregationStrategy::Simple => self.aggregate_simple(results, k),
            AggregationStrategy::RankFusion => self.aggregate_rank_fusion(results, k),
            AggregationStrategy::ScoreNormalization => {
                self.aggregate_score_normalization(results, k)
            }
            AggregationStrategy::BordaCount => self.aggregate_borda_count(results, k),
        }
    }

    /// Simple concatenation and re-ranking
    fn aggregate_simple(
        &self,
        results: Vec<(String, DistanceMetric, usize, SearchResult)>,
        k: usize,
    ) -> Result<Vec<FederatedSearchResult>> {
        let mut federated: Vec<_> = results
            .into_iter()
            .map(|(index_id, metric, rank, result)| FederatedSearchResult {
                cid: result.cid,
                score: result.score,
                source_index_id: index_id,
                source_rank: rank,
                source_metric: metric,
            })
            .collect();

        // Sort by score and take top k
        federated.sort_by(|a, b| {
            a.score
                .partial_cmp(&b.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        federated.truncate(k);

        Ok(federated)
    }

    /// Reciprocal rank fusion (RRF) - works well with heterogeneous metrics
    fn aggregate_rank_fusion(
        &self,
        results: Vec<(String, DistanceMetric, usize, SearchResult)>,
        k: usize,
    ) -> Result<Vec<FederatedSearchResult>> {
        let mut scores: HashMap<Cid, (f32, String, usize, DistanceMetric)> = HashMap::new();
        const RRF_K: f32 = 60.0;

        for (index_id, metric, rank, result) in results {
            let rrf_score = 1.0 / (RRF_K + rank as f32);

            scores
                .entry(result.cid)
                .and_modify(|(score, _, _, _)| *score += rrf_score)
                .or_insert((rrf_score, index_id.clone(), rank, metric));
        }

        let mut federated: Vec<_> = scores
            .into_iter()
            .map(
                |(cid, (score, index_id, rank, metric))| FederatedSearchResult {
                    cid,
                    score,
                    source_index_id: index_id,
                    source_rank: rank,
                    source_metric: metric,
                },
            )
            .collect();

        // Sort by RRF score (higher is better)
        federated.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        federated.truncate(k);

        Ok(federated)
    }

    /// Score normalization across indices
    fn aggregate_score_normalization(
        &self,
        results: Vec<(String, DistanceMetric, usize, SearchResult)>,
        k: usize,
    ) -> Result<Vec<FederatedSearchResult>> {
        // Group by index to compute normalization
        let mut by_index: HashMap<String, Vec<(DistanceMetric, usize, SearchResult)>> =
            HashMap::new();

        for (index_id, metric, rank, result) in results {
            by_index
                .entry(index_id)
                .or_default()
                .push((metric, rank, result));
        }

        // Normalize scores per index
        let mut normalized = Vec::new();
        for (index_id, index_results) in by_index {
            if index_results.is_empty() {
                continue;
            }

            // Find min/max scores
            let scores: Vec<f32> = index_results.iter().map(|(_, _, r)| r.score).collect();
            let min_score = scores.iter().copied().fold(f32::INFINITY, f32::min);
            let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let range = max_score - min_score;

            for (metric, rank, result) in index_results {
                let normalized_score = if range > 1e-6 {
                    (result.score - min_score) / range
                } else {
                    0.5 // All scores are equal
                };

                normalized.push(FederatedSearchResult {
                    cid: result.cid,
                    score: normalized_score,
                    source_index_id: index_id.clone(),
                    source_rank: rank,
                    source_metric: metric,
                });
            }
        }

        // Sort by normalized score and take top k
        normalized.sort_by(|a, b| {
            a.score
                .partial_cmp(&b.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        normalized.truncate(k);

        Ok(normalized)
    }

    /// Borda count voting method
    fn aggregate_borda_count(
        &self,
        results: Vec<(String, DistanceMetric, usize, SearchResult)>,
        k: usize,
    ) -> Result<Vec<FederatedSearchResult>> {
        let mut borda_scores: HashMap<Cid, (usize, String, usize, DistanceMetric)> = HashMap::new();

        // Maximum rank across all results
        let max_rank = results
            .iter()
            .map(|(_, _, rank, _)| *rank)
            .max()
            .unwrap_or(0);

        for (index_id, metric, rank, result) in results {
            let borda_points = max_rank.saturating_sub(rank);

            borda_scores
                .entry(result.cid)
                .and_modify(|(points, _, _, _)| *points += borda_points)
                .or_insert((borda_points, index_id.clone(), rank, metric));
        }

        let mut federated: Vec<_> = borda_scores
            .into_iter()
            .map(
                |(cid, (points, index_id, rank, metric))| FederatedSearchResult {
                    cid,
                    score: points as f32,
                    source_index_id: index_id,
                    source_rank: rank,
                    source_metric: metric,
                },
            )
            .collect();

        // Sort by Borda score (higher is better)
        federated.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        federated.truncate(k);

        Ok(federated)
    }

    /// Update query statistics
    fn update_stats(&self, indices_queried: u64, num_results: usize, latency: f64, timeouts: u64) {
        let mut stats = self.stats.write();
        stats.total_queries += 1;
        stats.total_indices_queried += indices_queried;
        stats.timeouts += timeouts;

        // Exponential moving average
        let alpha = 0.1;
        stats.avg_latency_ms = alpha * latency + (1.0 - alpha) * stats.avg_latency_ms;
        stats.avg_results_per_query =
            alpha * num_results as f64 + (1.0 - alpha) * stats.avg_results_per_query;
    }

    /// Get query statistics
    pub fn stats(&self) -> FederatedQueryStats {
        self.stats.read().clone()
    }

    /// Get list of registered index IDs
    pub fn registered_indices(&self) -> Vec<String> {
        let indices = self.indices.read();
        indices.keys().cloned().collect()
    }

    /// Get total size across all registered indices
    pub fn total_size(&self) -> usize {
        let indices = self.indices.read();
        indices.values().map(|idx| idx.size()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hnsw::VectorIndex;
    use multihash_codetable::{Code, MultihashDigest};

    #[tokio::test]
    async fn test_federated_executor_creation() {
        let config = FederatedConfig::default();
        let executor = FederatedQueryExecutor::new(config);
        assert_eq!(executor.registered_indices().len(), 0);
    }

    #[tokio::test]
    async fn test_register_and_unregister_index() {
        let executor = FederatedQueryExecutor::new(FederatedConfig::default());

        let index = VectorIndex::new(128, DistanceMetric::Cosine, 16, 200)
            .expect("test: create cosine index");
        let adapter =
            LocalIndexAdapter::new(Arc::new(RwLock::new(index)), "test_index".to_string());

        executor
            .register_index(Arc::new(adapter))
            .expect("test: register test index");
        assert_eq!(executor.registered_indices().len(), 1);

        executor
            .unregister_index("test_index")
            .expect("test: unregister test index");
        assert_eq!(executor.registered_indices().len(), 0);
    }

    #[tokio::test]
    async fn test_federated_query_single_index() {
        let executor = FederatedQueryExecutor::new(FederatedConfig::default());

        // Create and populate an index
        let index = VectorIndex::new(128, DistanceMetric::Cosine, 16, 200)
            .expect("test: create cosine index for single query");
        let index_lock = Arc::new(RwLock::new(index));

        // Insert some vectors
        for i in 0..100 {
            let data = format!("vector_{}", i);
            let hash = Code::Sha2_256.digest(data.as_bytes());
            let cid = Cid::new_v1(0x55, hash);
            let embedding: Vec<f32> = (0..128).map(|j| (i + j) as f32 * 0.01).collect();
            index_lock
                .write()
                .insert(&cid, &embedding)
                .expect("test: insert vector into index");
        }

        let adapter = LocalIndexAdapter::new(Arc::clone(&index_lock), "index1".to_string());
        executor
            .register_index(Arc::new(adapter))
            .expect("test: register index1");

        // Query
        let query_emb: Vec<f32> = (0..128).map(|i| i as f32 * 0.01).collect();
        let results = executor
            .query(&query_emb, 10)
            .await
            .expect("test: federated query single index");

        assert!(!results.is_empty());
        assert!(results.len() <= 10);
    }

    #[tokio::test]
    async fn test_federated_query_multiple_indices() {
        let config = FederatedConfig {
            aggregation_strategy: AggregationStrategy::RankFusion,
            ..Default::default()
        };
        let executor = FederatedQueryExecutor::new(config);

        // Create two indices with different metrics
        let index1 = VectorIndex::new(128, DistanceMetric::Cosine, 16, 200)
            .expect("test: create cosine index1");
        let index2 =
            VectorIndex::new(128, DistanceMetric::L2, 16, 200).expect("test: create l2 index2");

        let lock1 = Arc::new(RwLock::new(index1));
        let lock2 = Arc::new(RwLock::new(index2));

        // Populate both indices
        for i in 0..50 {
            let data = format!("vector_a_{}", i);
            let hash = Code::Sha2_256.digest(data.as_bytes());
            let cid = Cid::new_v1(0x55, hash);
            let embedding: Vec<f32> = (0..128).map(|j| (i + j) as f32 * 0.01).collect();
            lock1
                .write()
                .insert(&cid, &embedding)
                .expect("test: insert into index1");
        }

        for i in 25..75 {
            // Overlapping range
            let data = format!("vector_b_{}", i);
            let hash = Code::Sha2_256.digest(data.as_bytes());
            let cid = Cid::new_v1(0x55, hash);
            let embedding: Vec<f32> = (0..128).map(|j| (i + j) as f32 * 0.01).collect();
            lock2
                .write()
                .insert(&cid, &embedding)
                .expect("test: insert into index2");
        }

        executor
            .register_index(Arc::new(LocalIndexAdapter::new(
                Arc::clone(&lock1),
                "index1".to_string(),
            )))
            .expect("test: register index1 for multi");
        executor
            .register_index(Arc::new(LocalIndexAdapter::new(
                Arc::clone(&lock2),
                "index2".to_string(),
            )))
            .expect("test: register index2 for multi");

        // Query
        let query_emb: Vec<f32> = (0..128).map(|i| i as f32 * 0.02).collect();
        let results = executor
            .query(&query_emb, 10)
            .await
            .expect("test: federated query multiple indices");

        assert!(!results.is_empty());
        assert!(results.len() <= 10);

        // Check stats
        let stats = executor.stats();
        assert_eq!(stats.total_queries, 1);
        assert!(stats.total_indices_queried >= 1);
    }

    #[tokio::test]
    async fn test_different_aggregation_strategies() {
        for strategy in &[
            AggregationStrategy::Simple,
            AggregationStrategy::RankFusion,
            AggregationStrategy::ScoreNormalization,
            AggregationStrategy::BordaCount,
        ] {
            let config = FederatedConfig {
                aggregation_strategy: *strategy,
                ..Default::default()
            };
            let executor = FederatedQueryExecutor::new(config);

            let index = VectorIndex::new(128, DistanceMetric::Cosine, 16, 200)
                .expect("test: create index for strategy test");
            let lock = Arc::new(RwLock::new(index));

            // Populate
            for i in 0..20 {
                let data = format!("vec_{}", i);
                let hash = Code::Sha2_256.digest(data.as_bytes());
                let cid = Cid::new_v1(0x55, hash);
                let embedding: Vec<f32> = (0..128).map(|j| (i + j) as f32 * 0.01).collect();
                lock.write()
                    .insert(&cid, &embedding)
                    .expect("test: insert vector for strategy test");
            }

            executor
                .register_index(Arc::new(LocalIndexAdapter::new(
                    lock,
                    format!("index_{:?}", strategy),
                )))
                .expect("test: register index for strategy");

            let query_emb: Vec<f32> = (0..128).map(|i| i as f32 * 0.01).collect();
            let results = executor
                .query(&query_emb, 5)
                .await
                .expect("test: strategy query");

            assert!(!results.is_empty(), "Strategy {:?} failed", strategy);
        }
    }

    #[tokio::test]
    async fn test_privacy_preserving_mode() {
        let config = FederatedConfig {
            privacy_preserving: true,
            privacy_noise_level: 0.1,
            ..Default::default()
        };

        let executor = FederatedQueryExecutor::new(config);

        let index = VectorIndex::new(128, DistanceMetric::Cosine, 16, 200)
            .expect("test: create cosine index for privacy test");
        let lock = Arc::new(RwLock::new(index));

        for i in 0..30 {
            let data = format!("private_vec_{}", i);
            let hash = Code::Sha2_256.digest(data.as_bytes());
            let cid = Cid::new_v1(0x55, hash);
            let embedding: Vec<f32> = (0..128).map(|j| (i + j) as f32 * 0.01).collect();
            lock.write()
                .insert(&cid, &embedding)
                .expect("test: insert private vector");
        }

        executor
            .register_index(Arc::new(LocalIndexAdapter::new(
                lock,
                "private_index".to_string(),
            )))
            .expect("test: register private index");

        let query_emb: Vec<f32> = (0..128).map(|i| i as f32 * 0.01).collect();
        let results = executor
            .query(&query_emb, 5)
            .await
            .expect("test: privacy preserving query");

        // Results should still be returned (with noise applied to query)
        assert!(!results.is_empty());
    }

    #[tokio::test]
    async fn test_query_specific_indices() {
        let executor = FederatedQueryExecutor::new(FederatedConfig::default());

        // Register three indices
        for idx_num in 0..3 {
            let index = VectorIndex::new(128, DistanceMetric::Cosine, 16, 200)
                .expect("test: create cosine index for specific indices test");
            let lock = Arc::new(RwLock::new(index));

            for i in 0..20 {
                let data = format!("vec_{}_{}", idx_num, i);
                let hash = Code::Sha2_256.digest(data.as_bytes());
                let cid = Cid::new_v1(0x55, hash);
                let embedding: Vec<f32> =
                    (0..128).map(|j| (i + j + idx_num) as f32 * 0.01).collect();
                lock.write()
                    .insert(&cid, &embedding)
                    .expect("test: insert vector into specific index");
            }

            executor
                .register_index(Arc::new(LocalIndexAdapter::new(
                    lock,
                    format!("index_{}", idx_num),
                )))
                .expect("test: register specific index");
        }

        // Query only specific indices
        let query_emb: Vec<f32> = (0..128).map(|i| i as f32 * 0.01).collect();
        let results = executor
            .query_indices(
                &query_emb,
                10,
                &["index_0".to_string(), "index_2".to_string()],
            )
            .await
            .expect("test: query specific indices");

        assert!(!results.is_empty());

        // Results should only come from index_0 and index_2
        for result in results {
            assert!(result.source_index_id == "index_0" || result.source_index_id == "index_2");
        }
    }
}
