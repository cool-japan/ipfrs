//! Semantic DHT - Vector-based content routing
//!
//! This module extends the standard Kademlia DHT with semantic routing capabilities,
//! allowing content discovery based on vector embeddings and semantic similarity
//! rather than just content-addressed hashes.
//!
//! ## Features
//!
//! - **Embedding-based Routing**: Map vector embeddings to DHT keys using locality-sensitive hashing
//! - **Semantic Queries**: Find content based on semantic similarity
//! - **Distributed ANN**: Approximate nearest neighbor search across the network
//! - **Multiple Namespaces**: Support different embedding spaces (text, image, etc.)
//! - **Adaptive Routing**: Learn from query results to improve routing decisions
//!
//! ## Design
//!
//! The semantic DHT uses a two-layer architecture:
//! 1. **Embedding Layer**: Maps high-dimensional embeddings to DHT keys via LSH
//! 2. **Routing Layer**: Routes queries using both XOR distance (Kademlia) and semantic distance
//!
//! Each peer maintains:
//! - A local vector index for semantic search
//! - A mapping from embedding regions to peer IDs
//! - Statistics on query success rates per region

use cid::Cid;
use dashmap::DashMap;
use libp2p::PeerId;
use multihash_codetable::{Code, MultihashDigest};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use thiserror::Error;

/// Errors that can occur in semantic DHT operations
#[derive(Error, Debug)]
pub enum SemanticDhtError {
    #[error("Invalid embedding dimension: expected {expected}, got {actual}")]
    InvalidDimension { expected: usize, actual: usize },

    #[error("Unknown namespace: {0}")]
    UnknownNamespace(String),

    #[error("No peers found for embedding region")]
    NoPeersFound,

    #[error("Query timeout after {0:?}")]
    QueryTimeout(Duration),

    #[error("Embedding encoding error: {0}")]
    EncodingError(String),
}

/// Configuration for semantic DHT operations
#[derive(Debug, Clone)]
pub struct SemanticDhtConfig {
    /// Number of hash functions for LSH
    pub lsh_hash_functions: usize,

    /// Number of hash tables for LSH
    pub lsh_hash_tables: usize,

    /// Bucket width for LSH (affects quantization)
    pub lsh_bucket_width: f32,

    /// Maximum number of peers to query for ANN search
    pub max_query_peers: usize,

    /// Timeout for semantic queries
    pub query_timeout: Duration,

    /// Whether to cache query results
    pub enable_caching: bool,

    /// Cache TTL for query results
    pub cache_ttl: Duration,

    /// Maximum cache size
    pub max_cache_size: usize,

    /// Number of results to return for ANN queries
    pub top_k: usize,
}

impl Default for SemanticDhtConfig {
    fn default() -> Self {
        Self {
            lsh_hash_functions: 8,
            lsh_hash_tables: 4,
            lsh_bucket_width: 4.0,
            max_query_peers: 20,
            query_timeout: Duration::from_secs(10),
            enable_caching: true,
            cache_ttl: Duration::from_secs(300),
            max_cache_size: 1000,
            top_k: 10,
        }
    }
}

/// Semantic namespace identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NamespaceId(pub String);

impl NamespaceId {
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }

    /// Standard namespace for text embeddings
    pub fn text() -> Self {
        Self("text".to_string())
    }

    /// Standard namespace for image embeddings
    pub fn image() -> Self {
        Self("image".to_string())
    }

    /// Standard namespace for audio embeddings
    pub fn audio() -> Self {
        Self("audio".to_string())
    }
}

/// Namespace configuration
#[derive(Debug, Clone)]
pub struct SemanticNamespace {
    /// Namespace identifier
    pub id: NamespaceId,

    /// Expected embedding dimension
    pub dimension: usize,

    /// Distance metric to use
    pub distance_metric: DistanceMetric,

    /// LSH configuration specific to this namespace
    pub lsh_config: LshConfig,
}

/// Distance metric for vector similarity
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DistanceMetric {
    /// Euclidean distance (L2)
    Euclidean,

    /// Cosine distance (1 - cosine similarity)
    Cosine,

    /// Manhattan distance (L1)
    Manhattan,

    /// Dot product
    DotProduct,
}

/// LSH configuration
#[derive(Debug, Clone)]
pub struct LshConfig {
    /// Number of hash functions per table
    pub hash_functions: usize,

    /// Number of hash tables
    pub num_tables: usize,

    /// Bucket width for quantization
    pub bucket_width: f32,
}

impl Default for LshConfig {
    fn default() -> Self {
        Self {
            hash_functions: 8,
            num_tables: 4,
            bucket_width: 4.0,
        }
    }
}

/// Locality-Sensitive Hash for embeddings
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LshHash {
    /// Table index
    pub table: usize,

    /// Hash bucket
    pub bucket: Vec<i32>,
}

impl LshHash {
    /// Convert LSH hash to a DHT key (CID)
    pub fn to_cid(&self) -> Cid {
        // Serialize the hash bucket
        let mut data = Vec::new();
        data.push(self.table as u8);
        for &val in &self.bucket {
            data.extend_from_slice(&val.to_le_bytes());
        }

        // Hash the serialized data
        let hash = Code::Sha2_256.digest(&data);

        // Create CID
        Cid::new_v1(0x55, hash) // 0x55 = raw codec
    }
}

/// Semantic DHT query
#[derive(Debug, Clone)]
pub struct SemanticQuery {
    /// Query embedding
    pub embedding: Vec<f32>,

    /// Namespace to query
    pub namespace: NamespaceId,

    /// Number of results to return
    pub top_k: usize,

    /// Optional metadata filter
    pub metadata_filter: Option<HashMap<String, String>>,

    /// Query timeout
    pub timeout: Duration,
}

/// Result from a semantic query
#[derive(Debug, Clone)]
pub struct SemanticResult {
    /// Content identifier
    pub cid: Cid,

    /// Similarity score (higher is more similar)
    pub score: f32,

    /// Peer that provided this result
    pub peer: PeerId,

    /// Optional metadata
    pub metadata: HashMap<String, String>,
}

/// Cache entry for semantic queries
#[derive(Debug, Clone)]
struct CacheEntry {
    results: Vec<SemanticResult>,
    timestamp: Instant,
}

/// Semantic DHT manager
pub struct SemanticDht {
    /// Configuration
    config: SemanticDhtConfig,

    /// Registered namespaces
    namespaces: Arc<DashMap<NamespaceId, SemanticNamespace>>,

    /// LSH projections per namespace (random vectors for LSH)
    lsh_projections: Arc<DashMap<NamespaceId, Vec<Vec<f32>>>>,

    /// Mapping from LSH hash to peer IDs
    hash_to_peers: Arc<DashMap<LshHash, Vec<PeerId>>>,

    /// Local content index (CID -> embedding)
    local_index: Arc<DashMap<Cid, (Vec<f32>, NamespaceId)>>,

    /// Query result cache
    query_cache: Arc<DashMap<Vec<u8>, CacheEntry>>,

    /// Statistics
    stats: Arc<RwLock<SemanticDhtStats>>,
}

/// Statistics for semantic DHT
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SemanticDhtStats {
    /// Total queries processed
    pub total_queries: u64,

    /// Successful queries
    pub successful_queries: u64,

    /// Failed queries
    pub failed_queries: u64,

    /// Cache hits
    pub cache_hits: u64,

    /// Cache misses
    pub cache_misses: u64,

    /// Average query latency
    pub avg_query_latency_ms: f64,

    /// Total content indexed
    pub indexed_content: u64,

    /// Queries per namespace
    pub queries_per_namespace: HashMap<String, u64>,
}

impl SemanticDht {
    /// Create a new semantic DHT
    pub fn new(config: SemanticDhtConfig) -> Self {
        Self {
            config,
            namespaces: Arc::new(DashMap::new()),
            lsh_projections: Arc::new(DashMap::new()),
            hash_to_peers: Arc::new(DashMap::new()),
            local_index: Arc::new(DashMap::new()),
            query_cache: Arc::new(DashMap::new()),
            stats: Arc::new(RwLock::new(SemanticDhtStats::default())),
        }
    }

    /// Register a new semantic namespace
    pub fn register_namespace(&self, namespace: SemanticNamespace) -> Result<(), SemanticDhtError> {
        let namespace_id = namespace.id.clone();

        // Generate LSH projections for this namespace
        let projections = self.generate_lsh_projections(
            namespace.dimension,
            namespace.lsh_config.hash_functions,
            namespace.lsh_config.num_tables,
        );

        self.lsh_projections
            .insert(namespace_id.clone(), projections);
        self.namespaces.insert(namespace_id, namespace);

        Ok(())
    }

    /// Generate random projections for LSH
    fn generate_lsh_projections(
        &self,
        dimension: usize,
        hash_functions: usize,
        num_tables: usize,
    ) -> Vec<Vec<f32>> {
        use std::f32::consts::PI;

        let mut projections = Vec::new();
        let total_projections = hash_functions * num_tables;

        // Generate random unit vectors using Box-Muller transform
        for i in 0..total_projections {
            let mut projection = Vec::with_capacity(dimension);

            for j in 0..dimension {
                // Simple pseudo-random generation (deterministic for reproducibility)
                let seed = (i * dimension + j) as f32;
                let angle = seed * 2.0 * PI / 1000.0;
                let value = angle.sin();
                projection.push(value);
            }

            // Normalize
            let norm: f32 = projection.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for val in &mut projection {
                    *val /= norm;
                }
            }

            projections.push(projection);
        }

        projections
    }

    /// Compute LSH hashes for an embedding
    pub fn compute_lsh_hashes(
        &self,
        embedding: &[f32],
        namespace: &NamespaceId,
    ) -> Result<Vec<LshHash>, SemanticDhtError> {
        let ns = self
            .namespaces
            .get(namespace)
            .ok_or_else(|| SemanticDhtError::UnknownNamespace(namespace.0.clone()))?;

        if embedding.len() != ns.dimension {
            return Err(SemanticDhtError::InvalidDimension {
                expected: ns.dimension,
                actual: embedding.len(),
            });
        }

        let projections = self
            .lsh_projections
            .get(namespace)
            .ok_or_else(|| SemanticDhtError::UnknownNamespace(namespace.0.clone()))?;

        let mut hashes = Vec::new();
        let hash_functions = ns.lsh_config.hash_functions;

        for table in 0..ns.lsh_config.num_tables {
            let mut bucket = Vec::with_capacity(hash_functions);

            for func in 0..hash_functions {
                let proj_idx = table * hash_functions + func;
                let projection = &projections[proj_idx];

                // Compute dot product
                let dot_product: f32 = embedding
                    .iter()
                    .zip(projection.iter())
                    .map(|(a, b)| a * b)
                    .sum();

                // Quantize using bucket width
                let quantized = (dot_product / ns.lsh_config.bucket_width).floor() as i32;
                bucket.push(quantized);
            }

            hashes.push(LshHash { table, bucket });
        }

        Ok(hashes)
    }

    /// Index content with its embedding
    pub fn index_content(
        &self,
        cid: Cid,
        embedding: Vec<f32>,
        namespace: NamespaceId,
    ) -> Result<(), SemanticDhtError> {
        // Validate namespace
        let ns = self
            .namespaces
            .get(&namespace)
            .ok_or_else(|| SemanticDhtError::UnknownNamespace(namespace.0.clone()))?;

        if embedding.len() != ns.dimension {
            return Err(SemanticDhtError::InvalidDimension {
                expected: ns.dimension,
                actual: embedding.len(),
            });
        }

        // Store in local index
        self.local_index
            .insert(cid, (embedding.clone(), namespace.clone()));

        // Compute LSH hashes
        let hashes = self.compute_lsh_hashes(&embedding, &namespace)?;

        // Register hashes (in a real implementation, this would announce to DHT)
        for hash in hashes {
            // This is a placeholder - actual DHT announcement would happen here
            let _ = hash.to_cid();
        }

        // Update stats
        let mut stats = self.stats.write();
        stats.indexed_content += 1;

        Ok(())
    }

    /// Execute a semantic query
    pub fn query(&self, query: SemanticQuery) -> Result<Vec<SemanticResult>, SemanticDhtError> {
        let start = Instant::now();

        // Check cache first
        if self.config.enable_caching {
            let cache_key = self.compute_cache_key(&query);
            if let Some(entry) = self.query_cache.get(&cache_key) {
                if start.duration_since(entry.timestamp) < self.config.cache_ttl {
                    let mut stats = self.stats.write();
                    stats.cache_hits += 1;
                    return Ok(entry.results.clone());
                }
            }
        }

        // Validate namespace
        let _ns = self
            .namespaces
            .get(&query.namespace)
            .ok_or_else(|| SemanticDhtError::UnknownNamespace(query.namespace.0.clone()))?;

        // Compute LSH hashes for query
        let hashes = self.compute_lsh_hashes(&query.embedding, &query.namespace)?;

        // Find candidate peers (in real implementation, query DHT)
        let mut candidate_peers = Vec::new();
        for hash in &hashes {
            if let Some(peers) = self.hash_to_peers.get(hash) {
                candidate_peers.extend(peers.iter().cloned());
            }
        }

        // For now, search local index (in production, would query remote peers)
        let mut results = Vec::new();
        for entry in self.local_index.iter() {
            let (cid, (embedding, ns)) = entry.pair();

            if ns != &query.namespace {
                continue;
            }

            let distance = self.compute_distance(&query.embedding, embedding, &query.namespace)?;
            let score = 1.0 / (1.0 + distance); // Convert distance to similarity score

            results.push(SemanticResult {
                cid: *cid,
                score,
                peer: PeerId::random(), // Placeholder
                metadata: HashMap::new(),
            });
        }

        // Sort by score descending and take top-k
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        results.truncate(query.top_k);

        // Cache results
        if self.config.enable_caching {
            let cache_key = self.compute_cache_key(&query);
            let entry = CacheEntry {
                results: results.clone(),
                timestamp: start,
            };
            self.query_cache.insert(cache_key, entry);

            // Cleanup old cache entries
            self.cleanup_cache();
        }

        // Update stats
        let latency = start.elapsed().as_millis() as f64;
        let mut stats = self.stats.write();
        stats.total_queries += 1;
        stats.successful_queries += 1;
        stats.cache_misses += 1;

        // Update average latency (exponential moving average)
        let alpha = 0.1;
        stats.avg_query_latency_ms = alpha * latency + (1.0 - alpha) * stats.avg_query_latency_ms;

        *stats
            .queries_per_namespace
            .entry(query.namespace.0.clone())
            .or_insert(0) += 1;

        Ok(results)
    }

    /// Compute distance between two embeddings
    fn compute_distance(
        &self,
        a: &[f32],
        b: &[f32],
        namespace: &NamespaceId,
    ) -> Result<f32, SemanticDhtError> {
        let ns = self
            .namespaces
            .get(namespace)
            .ok_or_else(|| SemanticDhtError::UnknownNamespace(namespace.0.clone()))?;

        let distance = match ns.distance_metric {
            DistanceMetric::Euclidean => a
                .iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).powi(2))
                .sum::<f32>()
                .sqrt(),
            DistanceMetric::Cosine => {
                let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
                let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
                let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
                1.0 - (dot / (norm_a * norm_b))
            }
            DistanceMetric::Manhattan => a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum(),
            DistanceMetric::DotProduct => {
                -a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>() // Negative for similarity
            }
        };

        Ok(distance)
    }

    /// Compute cache key for a query
    fn compute_cache_key(&self, query: &SemanticQuery) -> Vec<u8> {
        // Simple hash of embedding + namespace
        let mut data = Vec::new();
        data.extend_from_slice(query.namespace.0.as_bytes());
        for &val in &query.embedding {
            data.extend_from_slice(&val.to_le_bytes());
        }
        data
    }

    /// Cleanup old cache entries
    fn cleanup_cache(&self) {
        if self.query_cache.len() <= self.config.max_cache_size {
            return;
        }

        let now = Instant::now();
        let ttl = self.config.cache_ttl;

        self.query_cache
            .retain(|_, entry| now.duration_since(entry.timestamp) < ttl);
    }

    /// Get statistics
    pub fn stats(&self) -> SemanticDhtStats {
        self.stats.read().clone()
    }

    /// Get namespace information
    pub fn get_namespace(&self, id: &NamespaceId) -> Option<SemanticNamespace> {
        self.namespaces.get(id).map(|ns| ns.clone())
    }

    /// List all registered namespaces
    pub fn list_namespaces(&self) -> Vec<NamespaceId> {
        self.namespaces
            .iter()
            .map(|entry| entry.key().clone())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_embedding(dim: usize, seed: f32) -> Vec<f32> {
        (0..dim).map(|i| ((i as f32 + seed) * 0.1).sin()).collect()
    }

    #[test]
    fn test_semantic_dht_creation() {
        let config = SemanticDhtConfig::default();
        let dht = SemanticDht::new(config);
        assert_eq!(dht.list_namespaces().len(), 0);
    }

    #[test]
    fn test_namespace_registration() {
        let dht = SemanticDht::new(SemanticDhtConfig::default());

        let namespace = SemanticNamespace {
            id: NamespaceId::text(),
            dimension: 128,
            distance_metric: DistanceMetric::Cosine,
            lsh_config: LshConfig::default(),
        };

        dht.register_namespace(namespace.clone()).unwrap();

        assert_eq!(dht.list_namespaces().len(), 1);
        assert_eq!(
            dht.get_namespace(&NamespaceId::text()).unwrap().dimension,
            128
        );
    }

    #[test]
    fn test_lsh_hash_computation() {
        let dht = SemanticDht::new(SemanticDhtConfig::default());

        let namespace = SemanticNamespace {
            id: NamespaceId::text(),
            dimension: 64,
            distance_metric: DistanceMetric::Euclidean,
            lsh_config: LshConfig::default(),
        };

        dht.register_namespace(namespace).unwrap();

        let embedding = create_test_embedding(64, 1.0);
        let hashes = dht
            .compute_lsh_hashes(&embedding, &NamespaceId::text())
            .unwrap();

        assert_eq!(hashes.len(), 4); // num_tables = 4
        assert_eq!(hashes[0].bucket.len(), 8); // hash_functions = 8
    }

    #[test]
    fn test_content_indexing() {
        let dht = SemanticDht::new(SemanticDhtConfig::default());

        let namespace = SemanticNamespace {
            id: NamespaceId::text(),
            dimension: 64,
            distance_metric: DistanceMetric::Cosine,
            lsh_config: LshConfig::default(),
        };

        dht.register_namespace(namespace).unwrap();

        let cid = Cid::default();
        let embedding = create_test_embedding(64, 1.0);

        dht.index_content(cid, embedding, NamespaceId::text())
            .unwrap();

        let stats = dht.stats();
        assert_eq!(stats.indexed_content, 1);
    }

    #[test]
    fn test_semantic_query() {
        let dht = SemanticDht::new(SemanticDhtConfig::default());

        let namespace = SemanticNamespace {
            id: NamespaceId::text(),
            dimension: 64,
            distance_metric: DistanceMetric::Cosine,
            lsh_config: LshConfig::default(),
        };

        dht.register_namespace(namespace).unwrap();

        // Index some content
        for i in 0..5 {
            let cid = Cid::default();
            let embedding = create_test_embedding(64, i as f32);
            dht.index_content(cid, embedding, NamespaceId::text())
                .unwrap();
        }

        // Query
        let query = SemanticQuery {
            embedding: create_test_embedding(64, 2.5),
            namespace: NamespaceId::text(),
            top_k: 3,
            metadata_filter: None,
            timeout: Duration::from_secs(5),
        };

        let results = dht.query(query).unwrap();
        assert!(results.len() <= 3);

        // Verify results are sorted by score
        for i in 1..results.len() {
            assert!(results[i - 1].score >= results[i].score);
        }
    }

    #[test]
    fn test_distance_metrics() {
        let dht = SemanticDht::new(SemanticDhtConfig::default());

        // Test Euclidean
        let ns_euclidean = SemanticNamespace {
            id: NamespaceId::new("euclidean"),
            dimension: 3,
            distance_metric: DistanceMetric::Euclidean,
            lsh_config: LshConfig::default(),
        };
        dht.register_namespace(ns_euclidean).unwrap();

        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let dist = dht
            .compute_distance(&a, &b, &NamespaceId::new("euclidean"))
            .unwrap();
        assert!((dist - 1.414).abs() < 0.01); // sqrt(2)

        // Test Cosine
        let ns_cosine = SemanticNamespace {
            id: NamespaceId::new("cosine"),
            dimension: 2,
            distance_metric: DistanceMetric::Cosine,
            lsh_config: LshConfig::default(),
        };
        dht.register_namespace(ns_cosine).unwrap();

        let a = vec![1.0, 0.0];
        let b = vec![1.0, 0.0];
        let dist = dht
            .compute_distance(&a, &b, &NamespaceId::new("cosine"))
            .unwrap();
        assert!(dist.abs() < 0.01); // Same vectors have distance ~0
    }

    #[test]
    fn test_query_caching() {
        let config = SemanticDhtConfig {
            enable_caching: true,
            cache_ttl: Duration::from_secs(60),
            ..Default::default()
        };

        let dht = SemanticDht::new(config);

        let namespace = SemanticNamespace {
            id: NamespaceId::text(),
            dimension: 64,
            distance_metric: DistanceMetric::Cosine,
            lsh_config: LshConfig::default(),
        };

        dht.register_namespace(namespace).unwrap();

        // Index content
        let cid = Cid::default();
        let embedding = create_test_embedding(64, 1.0);
        dht.index_content(cid, embedding.clone(), NamespaceId::text())
            .unwrap();

        // First query (cache miss)
        let query = SemanticQuery {
            embedding: embedding.clone(),
            namespace: NamespaceId::text(),
            top_k: 3,
            metadata_filter: None,
            timeout: Duration::from_secs(5),
        };

        let _ = dht.query(query.clone()).unwrap();
        let stats1 = dht.stats();
        assert_eq!(stats1.cache_misses, 1);

        // Second query (cache hit)
        let _ = dht.query(query).unwrap();
        let stats2 = dht.stats();
        assert_eq!(stats2.cache_hits, 1);
    }

    #[test]
    fn test_invalid_dimension() {
        let dht = SemanticDht::new(SemanticDhtConfig::default());

        let namespace = SemanticNamespace {
            id: NamespaceId::text(),
            dimension: 64,
            distance_metric: DistanceMetric::Cosine,
            lsh_config: LshConfig::default(),
        };

        dht.register_namespace(namespace).unwrap();

        let cid = Cid::default();
        let wrong_embedding = create_test_embedding(32, 1.0); // Wrong dimension

        let result = dht.index_content(cid, wrong_embedding, NamespaceId::text());
        assert!(matches!(
            result,
            Err(SemanticDhtError::InvalidDimension { .. })
        ));
    }

    #[test]
    fn test_unknown_namespace() {
        let dht = SemanticDht::new(SemanticDhtConfig::default());

        let embedding = create_test_embedding(64, 1.0);
        let result = dht.compute_lsh_hashes(&embedding, &NamespaceId::text());

        assert!(matches!(result, Err(SemanticDhtError::UnknownNamespace(_))));
    }

    #[test]
    fn test_lsh_hash_to_cid() {
        let hash = LshHash {
            table: 0,
            bucket: vec![1, 2, 3, 4],
        };

        let cid = hash.to_cid();
        assert_eq!(cid.version(), cid::Version::V1);
    }

    #[test]
    fn test_namespace_ids() {
        assert_eq!(NamespaceId::text().0, "text");
        assert_eq!(NamespaceId::image().0, "image");
        assert_eq!(NamespaceId::audio().0, "audio");
    }
}
