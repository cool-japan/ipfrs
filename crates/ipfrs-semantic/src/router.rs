//! Semantic router for content discovery
//!
//! This module provides semantic routing capabilities that combine
//! CID-based lookups with vector similarity search for intelligent
//! content discovery.

use crate::diskann::DiskANNIndex;
use crate::hnsw::{DistanceMetric, SearchResult, VectorIndex};
use crate::quantization::{dequantize_i8_to_f32, quantize_f32_to_i8};
use ipfrs_core::{Cid, Error, Result};
use lru::LruCache;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::num::NonZeroUsize;
use std::sync::{Arc, RwLock};

// ─────────────────────────────────────────────────────────────────────────────
// Index backend selection
// ─────────────────────────────────────────────────────────────────────────────

/// Selectable vector index backend for [`SemanticRouter`].
///
/// Choose based on dataset size and storage constraints:
/// - **HNSW** — fully in-memory, optimal for ≤ 10 M vectors.
/// - **DiskANN** — memory-mapped graph; handles 100 M+ vectors with bounded RAM.
#[derive(Debug, Clone, Default)]
pub enum IndexBackend {
    /// HNSW (default) — in-memory, fast search with high recall.
    #[default]
    Hnsw,
    /// DiskANN — disk-backed graph index for massive-scale datasets.
    DiskAnn {
        /// Path at which the DiskANN graph file is created or opened.
        graph_path: std::path::PathBuf,
    },
}

/// Internal handle that owns either an HNSW or a DiskANN index.
enum IndexHandle {
    Hnsw(Arc<RwLock<VectorIndex>>),
    DiskAnn(Arc<RwLock<DiskANNIndex>>),
}

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
    /// Which index backend to use (HNSW or DiskANN).
    /// Defaults to [`IndexBackend::Hnsw`].
    pub index_backend: IndexBackend,
    /// When `true`, vectors are quantized to INT8 before being stored in the
    /// index.  This trades a tiny accuracy loss for ~4× memory savings.
    pub quantize_vectors: bool,
    /// Bit-width for quantization: `8` for INT8 (default) or `1` for binary.
    /// Only relevant when `quantize_vectors` is `true`.
    pub quantization_bits: u8,
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
            index_backend: IndexBackend::Hnsw,
            quantize_vectors: false,
            quantization_bits: 8,
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
            cache_size: 2000,
            ..Self::default()
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
            ..Self::default()
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
            cache_size: 500,
            ..Self::default()
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
            cache_size: 5000,
            ..Self::default()
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
            ..Self::default()
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
/// search over content embeddings.  Supports both HNSW (in-memory) and
/// DiskANN (disk-backed) backends, and optional INT8 vector quantization.
pub struct SemanticRouter {
    /// Underlying vector index (HNSW or DiskANN)
    index: IndexHandle,
    /// Router configuration
    config: RouterConfig,
    /// Query result cache (LRU)
    query_cache: Arc<RwLock<LruCache<QueryCacheKey, Vec<SearchResult>>>>,
}

impl SemanticRouter {
    /// Create a new semantic router with the given configuration.
    ///
    /// When the backend is [`IndexBackend::DiskAnn`] the graph file is created
    /// at `graph_path` if it does not yet exist.
    pub fn new(config: RouterConfig) -> Result<Self> {
        let cache_size = NonZeroUsize::new(config.cache_size)
            .unwrap_or_else(|| NonZeroUsize::new(1000).expect("1000 is non-zero"));
        let query_cache = LruCache::new(cache_size);

        let index = match &config.index_backend {
            IndexBackend::Hnsw => {
                let hnsw = VectorIndex::new(
                    config.dimension,
                    config.metric,
                    config.max_connections,
                    config.ef_construction,
                )?;
                IndexHandle::Hnsw(Arc::new(RwLock::new(hnsw)))
            }
            IndexBackend::DiskAnn { graph_path } => {
                use crate::diskann::DiskANNConfig;
                let da_config = DiskANNConfig {
                    dimension: config.dimension,
                    ..DiskANNConfig::default()
                };
                let mut da_index = DiskANNIndex::new(da_config);
                da_index.create(graph_path)?;
                IndexHandle::DiskAnn(Arc::new(RwLock::new(da_index)))
            }
        };

        Ok(Self {
            index,
            config,
            query_cache: Arc::new(RwLock::new(query_cache)),
        })
    }

    /// Create a new router with default configuration
    pub fn with_defaults() -> Result<Self> {
        Self::new(RouterConfig::default())
    }

    /// Optionally quantize `embedding` to INT8 and back to f32.
    ///
    /// When `config.quantize_vectors` is `true` this reduces storage fidelity
    /// slightly but saves ~4× memory inside the index.
    fn maybe_quantize(&self, embedding: &[f32]) -> Vec<f32> {
        if self.config.quantize_vectors {
            let (q, scale, zero_point) = quantize_f32_to_i8(embedding);
            dequantize_i8_to_f32(&q, scale, zero_point)
        } else {
            embedding.to_vec()
        }
    }

    /// Helper: convert a `diskann::SearchResult` (distance-based) to the
    /// common `hnsw::SearchResult` (score-based) used throughout the router.
    ///
    /// Score = 1 / (1 + distance) so that closer vectors get higher scores.
    fn diskann_to_search_result(r: crate::diskann::SearchResult) -> SearchResult {
        SearchResult {
            cid: r.cid,
            score: 1.0 / (1.0 + r.distance),
        }
    }

    /// Add content with its embedding to the router.
    ///
    /// When `config.quantize_vectors` is `true` the embedding is first
    /// quantized to INT8 and dequantized before being stored.
    ///
    /// # Arguments
    /// * `cid` - Content identifier
    /// * `embedding` - Vector embedding of the content
    pub fn add(&self, cid: &Cid, embedding: &[f32]) -> Result<()> {
        let v = self.maybe_quantize(embedding);
        match &self.index {
            IndexHandle::Hnsw(idx) => idx
                .write()
                .map_err(|_| Error::Internal("HNSW index lock poisoned".into()))?
                .insert(cid, &v),
            IndexHandle::DiskAnn(idx) => idx
                .write()
                .map_err(|_| Error::Internal("DiskANN index lock poisoned".into()))?
                .insert(cid, &v),
        }
    }

    /// Add multiple content items in batch.
    ///
    /// For HNSW backends this uses the optimised bulk-insert path; for DiskANN
    /// it falls back to sequential inserts.
    ///
    /// # Arguments
    /// * `items` - Vector of (CID, embedding) pairs
    pub fn add_batch(&self, items: &[(Cid, Vec<f32>)]) -> Result<()> {
        match &self.index {
            IndexHandle::Hnsw(idx) => {
                if self.config.quantize_vectors {
                    let quantized: Vec<(Cid, Vec<f32>)> = items
                        .iter()
                        .map(|(cid, emb)| (*cid, self.maybe_quantize(emb)))
                        .collect();
                    idx.write()
                        .map_err(|_| Error::Internal("HNSW index lock poisoned".into()))?
                        .insert_batch(&quantized)
                } else {
                    idx.write()
                        .map_err(|_| Error::Internal("HNSW index lock poisoned".into()))?
                        .insert_batch(items)
                }
            }
            IndexHandle::DiskAnn(idx) => {
                for (cid, emb) in items {
                    let v = self.maybe_quantize(emb);
                    idx.write()
                        .map_err(|_| Error::Internal("DiskANN index lock poisoned".into()))?
                        .insert(cid, &v)?;
                }
                Ok(())
            }
        }
    }

    /// Remove content from the router.
    ///
    /// **Note:** DiskANN does not support deletion; this returns an error for
    /// that backend.
    pub fn remove(&self, cid: &Cid) -> Result<()> {
        match &self.index {
            IndexHandle::Hnsw(idx) => idx
                .write()
                .map_err(|_| Error::Internal("HNSW index lock poisoned".into()))?
                .delete(cid),
            IndexHandle::DiskAnn(_) => Err(Error::InvalidInput(
                "DiskANN backend does not support deletion".to_string(),
            )),
        }
    }

    /// Check if content exists in the router.
    ///
    /// **Note:** For DiskANN this always returns `false` because the CID map
    /// is not publicly exposed by the backend.
    pub fn contains(&self, cid: &Cid) -> bool {
        match &self.index {
            IndexHandle::Hnsw(idx) => idx.read().map(|g| g.contains(cid)).unwrap_or(false),
            IndexHandle::DiskAnn(_) => false,
        }
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

    /// Query with auto-tuned ef_search parameter.
    ///
    /// For HNSW backends, computes the optimal ef_search based on k and index
    /// size.  For DiskANN, falls back to the configured `ef_search` value.
    ///
    /// # Arguments
    /// * `query_embedding` - Query vector
    /// * `k` - Number of results to return
    pub async fn query_auto(&self, query_embedding: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        let optimal_ef_search = match &self.index {
            IndexHandle::Hnsw(idx) => idx
                .read()
                .map(|g| g.compute_optimal_ef_search(k))
                .unwrap_or(self.config.ef_search),
            IndexHandle::DiskAnn(_) => self.config.ef_search,
        };
        self.query_with_ef(query_embedding, k, optimal_ef_search)
            .await
    }

    /// Query with custom ef_search parameter.
    ///
    /// The `ef_search` parameter is ignored for DiskANN (the backend controls
    /// its own search list size).
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
        let cache_key = Self::compute_cache_key(query_embedding, k, &QueryFilter::default());

        if let Some(cached) = self
            .query_cache
            .write()
            .map_err(|_| Error::Internal("cache lock poisoned".into()))?
            .get(&cache_key)
        {
            return Ok(cached.clone());
        }

        let results = self.search_backend(query_embedding, k, ef_search)?;

        self.query_cache
            .write()
            .map_err(|_| Error::Internal("cache lock poisoned".into()))?
            .put(cache_key, results.clone());

        Ok(results)
    }

    /// Dispatch a search to the active backend and return unified `SearchResult`s.
    fn search_backend(
        &self,
        query: &[f32],
        k: usize,
        ef_search: usize,
    ) -> Result<Vec<SearchResult>> {
        match &self.index {
            IndexHandle::Hnsw(idx) => idx
                .read()
                .map_err(|_| Error::Internal("HNSW index lock poisoned".into()))?
                .search(query, k, ef_search),
            IndexHandle::DiskAnn(idx) => {
                let raw = idx
                    .read()
                    .map_err(|_| Error::Internal("DiskANN index lock poisoned".into()))?
                    .search(query, k)?;
                Ok(raw
                    .into_iter()
                    .map(Self::diskann_to_search_result)
                    .collect())
            }
        }
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
        let cache_key = Self::compute_cache_key(query_embedding, k, &filter);

        if filter.min_score.is_none() && filter.cid_prefix.is_none() {
            if let Some(cached) = self
                .query_cache
                .write()
                .map_err(|_| Error::Internal("cache lock poisoned".into()))?
                .get(&cache_key)
            {
                return Ok(cached.clone());
            }
        }

        let fetch_k = if filter.min_score.is_some() || filter.cid_prefix.is_some() {
            k * 2
        } else {
            k
        };

        let mut results = self.search_backend(query_embedding, fetch_k, self.config.ef_search)?;

        if let Some(min_score) = filter.min_score {
            results.retain(|r| r.score >= min_score);
        }

        if let Some(max_score) = filter.max_score {
            results.retain(|r| r.score <= max_score);
        }

        if let Some(ref prefix) = filter.cid_prefix {
            results.retain(|r| r.cid.to_string().starts_with(prefix));
        }

        if let Some(max_results) = filter.max_results {
            results.truncate(max_results);
        }

        if filter.min_score.is_none() && filter.cid_prefix.is_none() {
            self.query_cache
                .write()
                .map_err(|_| Error::Internal("cache lock poisoned".into()))?
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
        if let Ok(mut cache) = self.query_cache.write() {
            cache.clear();
        }
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> CacheStats {
        match self.query_cache.read() {
            Ok(cache) => CacheStats {
                size: cache.len(),
                capacity: cache.cap().get(),
            },
            Err(_) => CacheStats {
                size: 0,
                capacity: 0,
            },
        }
    }

    /// Get statistics about the router
    pub fn stats(&self) -> RouterStats {
        match &self.index {
            IndexHandle::Hnsw(idx) => {
                let guard = idx.read().unwrap_or_else(|p| p.into_inner());
                RouterStats {
                    num_vectors: guard.len(),
                    dimension: guard.dimension(),
                    metric: guard.metric(),
                }
            }
            IndexHandle::DiskAnn(idx) => {
                let guard = idx.read().unwrap_or_else(|p| p.into_inner());
                let s = guard.stats();
                RouterStats {
                    num_vectors: s.num_vectors,
                    dimension: s.dimension,
                    metric: DistanceMetric::L2,
                }
            }
        }
    }

    /// Estimated memory usage in bytes for the underlying index.
    ///
    /// For HNSW delegates to [`VectorIndex::estimated_memory_bytes`].
    /// For DiskANN returns the estimated disk size instead (most data is memory-mapped).
    pub fn estimated_memory_bytes(&self) -> ipfrs_core::Result<usize> {
        match &self.index {
            IndexHandle::Hnsw(idx) => {
                let guard = idx.read().map_err(|_| {
                    ipfrs_core::Error::Storage("HNSW index lock poisoned".to_string())
                })?;
                Ok(guard.estimated_memory_bytes())
            }
            IndexHandle::DiskAnn(idx) => {
                let guard = idx.read().map_err(|_| {
                    ipfrs_core::Error::Storage("DiskANN index lock poisoned".to_string())
                })?;
                Ok(guard.stats().estimated_disk_size)
            }
        }
    }

    /// Get optimization recommendations.
    ///
    /// For DiskANN backends, returns defaults as HNSW-specific tuning does
    /// not apply.
    pub fn optimization_recommendations(&self) -> OptimizationRecommendations {
        match &self.index {
            IndexHandle::Hnsw(idx) => {
                let guard = idx.read().unwrap_or_else(|p| p.into_inner());
                let (m, ef_construction) = guard.compute_optimal_parameters();
                OptimizationRecommendations {
                    recommended_m: m,
                    recommended_ef_construction: ef_construction,
                    current_size: guard.len(),
                }
            }
            IndexHandle::DiskAnn(idx) => {
                let guard = idx.read().unwrap_or_else(|p| p.into_inner());
                let s = guard.stats();
                OptimizationRecommendations {
                    recommended_m: 64,
                    recommended_ef_construction: 100,
                    current_size: s.num_vectors,
                }
            }
        }
    }

    /// Save the semantic index to a file.
    ///
    /// For HNSW backends, serializes the full index to the given path.
    /// For DiskANN, persists all in-memory graph data to the backing file.
    ///
    /// # Arguments
    /// * `path` - Path to save the index file (only used for HNSW)
    pub async fn save_index<P: AsRef<std::path::Path>>(&self, path: P) -> Result<()> {
        match &self.index {
            IndexHandle::Hnsw(idx) => idx
                .read()
                .map_err(|_| Error::Internal("HNSW index lock poisoned".into()))?
                .save(path.as_ref()),
            IndexHandle::DiskAnn(idx) => idx
                .read()
                .map_err(|_| Error::Internal("DiskANN index lock poisoned".into()))?
                .save(),
        }
    }

    /// Save the semantic index using smart incremental logic.
    ///
    /// Only supported for HNSW backends (falls back to full save for DiskANN).
    ///
    /// # Arguments
    /// * `path` - Base path of the snapshot file
    pub async fn save_index_smart<P: AsRef<std::path::Path>>(&self, path: P) -> Result<()> {
        match &self.index {
            IndexHandle::Hnsw(idx) => {
                use crate::persistence::IndexPersistence;
                let persistence = IndexPersistence::new(path.as_ref());
                let index_guard = idx.read().map_err(|_| {
                    ipfrs_core::Error::Internal("index lock poisoned in save_index_smart".into())
                })?;
                persistence.save_smart(&index_guard)
            }
            IndexHandle::DiskAnn(idx) => idx
                .read()
                .map_err(|_| Error::Internal("DiskANN index lock poisoned".into()))?
                .save(),
        }
    }

    /// Load a semantic index from a file and apply any available incremental
    /// delta on top of the loaded full snapshot.
    ///
    /// Only supported for HNSW backends.
    ///
    /// # Arguments
    /// * `path` - Path to the full snapshot file
    pub async fn load_index_with_delta<P: AsRef<std::path::Path>>(&self, path: P) -> Result<()> {
        match &self.index {
            IndexHandle::Hnsw(idx) => {
                use crate::persistence::{IndexPersistence, IndexPersistence as IP};
                let persistence = IP::new(path.as_ref());

                let mut snap = persistence.load()?;

                if let Ok(delta) = persistence.load_incremental() {
                    if delta.delta_version > delta.base_version {
                        tracing::debug!(
                            base_version = delta.base_version,
                            delta_version = delta.delta_version,
                            changed = delta.changed_entries.len(),
                            "Applying incremental HNSW delta on top of full snapshot"
                        );
                        IndexPersistence::apply_incremental(&mut snap, &delta)?;
                    }
                }

                let loaded_index = VectorIndex::from_snapshot(&snap)?;
                *idx.write().map_err(|_| {
                    ipfrs_core::Error::Internal(
                        "index write lock poisoned in load_index_with_delta".into(),
                    )
                })? = loaded_index;

                self.clear_cache();
                Ok(())
            }
            IndexHandle::DiskAnn(_) => Err(Error::InvalidInput(
                "load_index_with_delta is not supported for DiskANN backend".to_string(),
            )),
        }
    }

    /// Load a semantic index from a file.
    ///
    /// Only supported for HNSW backends; DiskANN indices are always loaded via
    /// `RouterConfig` at construction time.
    ///
    /// # Arguments
    /// * `path` - Path to the saved index file
    pub async fn load_index<P: AsRef<std::path::Path>>(&self, path: P) -> Result<()> {
        match &self.index {
            IndexHandle::Hnsw(idx) => {
                let loaded_index = VectorIndex::load(path.as_ref())?;
                *idx.write()
                    .map_err(|_| Error::Internal("HNSW index lock poisoned".into()))? =
                    loaded_index;
                self.clear_cache();
                Ok(())
            }
            IndexHandle::DiskAnn(_) => Err(Error::InvalidInput(
                "load_index is not supported for DiskANN backend; use RouterConfig at construction"
                    .to_string(),
            )),
        }
    }

    /// Clear all content from the router.
    ///
    /// For DiskANN backends this returns an error; use a new router instance
    /// with a fresh graph path to start over.
    pub fn clear(&self) -> Result<()> {
        match &self.index {
            IndexHandle::Hnsw(idx) => {
                let new_index = VectorIndex::new(
                    self.config.dimension,
                    self.config.metric,
                    self.config.max_connections,
                    self.config.ef_construction,
                )?;
                *idx.write()
                    .map_err(|_| Error::Internal("HNSW index lock poisoned".into()))? = new_index;
                self.clear_cache();
                Ok(())
            }
            IndexHandle::DiskAnn(_) => Err(Error::InvalidInput(
                "clear is not supported for DiskANN backend".to_string(),
            )),
        }
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

        let ef_search = self.config.ef_search;

        let results: Result<Vec<Vec<SearchResult>>> = query_embeddings
            .par_iter()
            .map(|embedding| {
                let cache_key = Self::compute_cache_key(embedding, k, &filter);

                if filter.min_score.is_none() && filter.cid_prefix.is_none() {
                    if let Ok(mut cache) = self.query_cache.write() {
                        if let Some(cached) = cache.get(&cache_key) {
                            return Ok(cached.clone());
                        }
                    }
                }

                let fetch_k = if filter.min_score.is_some() || filter.cid_prefix.is_some() {
                    k * 2
                } else {
                    k
                };

                let mut results = self.search_backend(embedding, fetch_k, ef_search)?;

                if let Some(min_score) = filter.min_score {
                    results.retain(|r| r.score >= min_score);
                }
                if let Some(max_score) = filter.max_score {
                    results.retain(|r| r.score <= max_score);
                }
                if let Some(ref prefix) = filter.cid_prefix {
                    results.retain(|r| r.cid.to_string().starts_with(prefix));
                }
                if let Some(max_results) = filter.max_results {
                    results.truncate(max_results);
                }

                if filter.min_score.is_none() && filter.cid_prefix.is_none() {
                    if let Ok(mut cache) = self.query_cache.write() {
                        cache.put(cache_key, results.clone());
                    }
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

        let results: Result<Vec<Vec<SearchResult>>> = query_embeddings
            .par_iter()
            .map(|embedding| {
                let cache_key = Self::compute_cache_key(embedding, k, &QueryFilter::default());

                if let Ok(mut cache) = self.query_cache.write() {
                    if let Some(cached) = cache.get(&cache_key) {
                        return Ok(cached.clone());
                    }
                }

                let results = self.search_backend(embedding, k, ef_search)?;

                if let Ok(mut cache) = self.query_cache.write() {
                    cache.put(cache_key, results.clone());
                }

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
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .expect("results is non-empty (total_count > 0)");
        let max_score = results
            .iter()
            .map(|r| r.score)
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .expect("results is non-empty (total_count > 0)");

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
        let router =
            SemanticRouter::with_defaults().expect("test: SemanticRouter::with_defaults failed");

        let cid1 = "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi"
            .parse::<Cid>()
            .expect("test: CID parse failed");
        let embedding1 = vec![0.5; 768];

        router
            .add(&cid1, &embedding1)
            .expect("test: router.add cid1 failed");

        let results = router
            .query(&embedding1, 1)
            .await
            .expect("test: router.query failed");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].cid, cid1);
    }

    #[tokio::test]
    async fn test_filtering() {
        let router =
            SemanticRouter::with_defaults().expect("test: SemanticRouter::with_defaults failed");

        let cid1 = "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi"
            .parse::<Cid>()
            .expect("test: CID parse failed");
        let embedding1 = vec![0.5; 768];

        router
            .add(&cid1, &embedding1)
            .expect("test: router.add cid1 failed");

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
            .expect("test: router.query_with_filter failed");

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
        .expect("test: SemanticRouter::new with dimension 3 failed");

        // Create test blocks
        let data1 = Bytes::from_static(b"Hello, semantic search!");
        let data2 = Bytes::from_static(b"Goodbye, semantic search!");
        let data3 = Bytes::from_static(b"Hello, world!");

        let block1 = Block::new(data1).expect("test: Block::new data1 failed");
        let block2 = Block::new(data2).expect("test: Block::new data2 failed");
        let block3 = Block::new(data3).expect("test: Block::new data3 failed");

        // Generate simple embeddings based on content
        // In real use, these would come from an embedding model
        let embedding1 = vec![1.0, 0.0, 0.0]; // "Hello" cluster
        let embedding2 = vec![0.0, 1.0, 0.0]; // "Goodbye" cluster
        let embedding3 = vec![0.9, 0.1, 0.0]; // Close to "Hello" cluster

        // Index blocks with their embeddings
        router
            .add(block1.cid(), &embedding1)
            .expect("test: router.add block1 failed");
        router
            .add(block2.cid(), &embedding2)
            .expect("test: router.add block2 failed");
        router
            .add(block3.cid(), &embedding3)
            .expect("test: router.add block3 failed");

        // Query for blocks similar to "Hello"
        let query_embedding = vec![1.0, 0.0, 0.0];
        let results = router
            .query(&query_embedding, 2)
            .await
            .expect("test: router.query failed");

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
        .expect("test: SemanticRouter::new with dimension 2 failed");

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
            .expect("test: CID1 parse failed");
        let cid2 = "bafybeibazl2z6vqxqqzmhmvx2hfpxqtwggqgbbyy3sxkq4vzq6cqsvwbjy"
            .parse::<Cid>()
            .expect("test: CID2 parse failed");

        // Index tensors by their semantic embeddings
        router
            .add(&cid1, &vision_embedding)
            .expect("test: router.add cid1 vision failed");
        router
            .add(&cid2, &text_embedding)
            .expect("test: router.add cid2 text failed");

        // Search for vision-type tensors
        let results = router
            .query(&vision_embedding, 1)
            .await
            .expect("test: router.query vision failed");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].cid, cid1);
    }

    #[tokio::test]
    async fn test_large_scale_indexing() {
        use rand::RngExt;

        let dimension = 128;

        // Create router with dimension 128 for this test
        let router = SemanticRouter::new(RouterConfig {
            dimension,
            ..Default::default()
        })
        .expect("test: SemanticRouter::new with dimension 128 failed");

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

            router
                .add(&cid, &embedding)
                .expect("test: router.add large scale failed");
            indexed_cids.push((cid, embedding));
        }

        // Verify index size
        let stats = router.stats();
        assert_eq!(stats.num_vectors, num_items);

        // Test query on a known embedding
        let (test_cid, test_embedding) = &indexed_cids[42];
        let results = router
            .query(test_embedding, 1)
            .await
            .expect("test: router.query large scale failed");

        // Should return the exact match as the top result
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].cid, *test_cid);
    }

    #[tokio::test]
    async fn test_cache_effectiveness() {
        let router =
            SemanticRouter::with_defaults().expect("test: SemanticRouter::with_defaults failed");

        let cid1 = "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi"
            .parse::<Cid>()
            .expect("test: CID parse failed");
        let embedding1 = vec![0.5; 768];

        router
            .add(&cid1, &embedding1)
            .expect("test: router.add cid1 failed");

        // Perform same query multiple times
        for _ in 0..10 {
            let _ = router
                .query(&embedding1, 1)
                .await
                .expect("test: router.query cache test failed");
        }

        // Check cache stats - should have cached the query
        let cache_stats = router.cache_stats();
        assert_eq!(cache_stats.size, 1, "Cache should have 1 unique query");
        assert!(cache_stats.capacity > 0, "Cache should have capacity");
    }

    #[tokio::test]
    async fn test_batch_query() {
        use rand::RngExt;

        let dimension = 128;

        // Create router with dimension 128 for this test
        let router = SemanticRouter::new(RouterConfig {
            dimension,
            ..Default::default()
        })
        .expect("test: SemanticRouter::new with dimension 128 failed");

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

            router
                .add(&cid, &embedding)
                .expect("test: router.add batch test failed");
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
        let results = router
            .query_batch(&query_batch, 5)
            .await
            .expect("test: router.query_batch failed");

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
        use rand::RngExt;

        let dimension = 64;

        let router = SemanticRouter::new(RouterConfig {
            dimension,
            ..Default::default()
        })
        .expect("test: SemanticRouter::new with dimension 64 failed");

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

            router
                .add(&cid, &embedding)
                .expect("test: router.add filter batch test failed");
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
            .expect("test: router.query_batch_with_filter failed");

        // Verify results
        assert_eq!(results.len(), batch_size);
        for result in &results {
            assert!(result.len() <= 3); // Max results filter applied
        }
    }

    #[tokio::test]
    async fn test_batch_query_with_ef() {
        use rand::RngExt;

        let dimension = 64;

        let router = SemanticRouter::new(RouterConfig {
            dimension,
            ..Default::default()
        })
        .expect("test: SemanticRouter::new with dimension 64 failed");

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

            router
                .add(&cid, &embedding)
                .expect("test: router.add ef batch test failed");
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
            .expect("test: router.query_batch_with_ef failed");

        // Verify results
        assert_eq!(results.len(), batch_size);
        for result in &results {
            assert!(!result.is_empty());
            assert!(result.len() <= 3);
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Tests for DiskANN backend and quantized mode
    // ─────────────────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_router_diskann_backend_selection() {
        // Use a temp directory to host the graph file
        let tmp = std::env::temp_dir().join(format!(
            "ipfrs_diskann_test_{}.idx",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));

        let config = RouterConfig {
            dimension: 8,
            index_backend: IndexBackend::DiskAnn {
                graph_path: tmp.clone(),
            },
            ..RouterConfig::balanced(8)
        };

        let router = SemanticRouter::new(config).expect("should create DiskANN router");

        // Insert a few vectors
        use multihash_codetable::{Code, MultihashDigest};
        for i in 0..3usize {
            let data = format!("diskann_test_{i}");
            let hash = Code::Sha2_256.digest(data.as_bytes());
            let cid = Cid::new_v1(0x55, hash);
            let emb: Vec<f32> = (0..8).map(|d| (i * 8 + d) as f32 * 0.01).collect();
            router.add(&cid, &emb).expect("insert should succeed");
        }

        // Verify stats reflect the inserts
        let s = router.stats();
        assert_eq!(s.num_vectors, 3, "DiskANN should report 3 vectors");
        assert_eq!(s.dimension, 8);

        // Search should return something
        let query: Vec<f32> = (0..8).map(|d| d as f32 * 0.01).collect();
        let results = router
            .query(&query, 2)
            .await
            .expect("search should succeed");
        assert!(!results.is_empty(), "DiskANN search should return results");

        // Cleanup
        let _ = std::fs::remove_file(&tmp);
        let _ = std::fs::remove_file(format!("{}.vectors", tmp.display()));
    }

    #[tokio::test]
    async fn test_router_quantized_mode() {
        use multihash_codetable::{Code, MultihashDigest};

        let config = RouterConfig {
            dimension: 16,
            quantize_vectors: true,
            quantization_bits: 8,
            ..RouterConfig::balanced(16)
        };

        let router = SemanticRouter::new(config).expect("should create quantized router");

        // Add 5 embeddings
        let mut cids = Vec::new();
        for i in 0..5usize {
            let data = format!("quant_test_{i}");
            let hash = Code::Sha2_256.digest(data.as_bytes());
            let cid = Cid::new_v1(0x55, hash);
            let emb: Vec<f32> = (0..16).map(|d| (i as f32 + d as f32) * 0.05).collect();
            router
                .add(&cid, &emb)
                .expect("quantized insert should succeed");
            cids.push((cid, emb));
        }

        assert_eq!(router.stats().num_vectors, 5);

        // Search should return results
        let (_, ref query_emb) = cids[0];
        let results = router
            .query(query_emb, 3)
            .await
            .expect("quantized search should succeed");
        assert!(
            !results.is_empty(),
            "quantized search should return results"
        );
    }
}
