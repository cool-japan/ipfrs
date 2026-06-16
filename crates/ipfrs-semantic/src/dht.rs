//! Distributed Semantic DHT
//!
//! This module provides a distributed hash table optimized for semantic search:
//! - Embedding-based routing to nearest peers in vector space
//! - Clustering of similar nodes for locality optimization
//! - Distributed k-NN search across multiple peers
//! - Replication for fault tolerance
//! - Load balancing and query routing optimization

use crate::hnsw::{DistanceMetric, SearchResult};
use ipfrs_core::{Cid, Error, Result};
use ipfrs_network::libp2p::PeerId;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// Configuration for the semantic DHT
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticDHTConfig {
    /// Embedding dimension for routing
    pub embedding_dim: usize,
    /// Number of replicas for each entry
    pub replication_factor: usize,
    /// Number of closest peers to consider for routing
    pub routing_table_size: usize,
    /// Distance metric for peer similarity
    pub distance_metric: DistanceMetric,
    /// Number of hops for multi-hop search
    pub max_hops: usize,
    /// Timeout for peer queries in milliseconds
    pub query_timeout_ms: u64,
}

impl Default for SemanticDHTConfig {
    fn default() -> Self {
        Self {
            embedding_dim: 768,
            replication_factor: 3,
            routing_table_size: 20,
            distance_metric: DistanceMetric::Cosine,
            max_hops: 5,
            query_timeout_ms: 5000,
        }
    }
}

/// Represents a peer in the semantic DHT with its embedding
#[derive(Debug, Clone)]
pub struct SemanticPeer {
    /// Peer identifier
    pub peer_id: PeerId,
    /// Embedding representing this peer's data distribution
    pub embedding: Vec<f32>,
    /// Cluster ID this peer belongs to
    pub cluster_id: Option<usize>,
    /// Last seen timestamp
    pub last_seen: u64,
    /// Load metric (0.0 = idle, 1.0 = overloaded)
    pub load: f32,
}

impl SemanticPeer {
    /// Create a new semantic peer
    pub fn new(peer_id: PeerId, embedding: Vec<f32>) -> Self {
        Self {
            peer_id,
            embedding,
            cluster_id: None,
            last_seen: current_timestamp(),
            load: 0.0,
        }
    }

    /// Update the last seen timestamp
    pub fn update_last_seen(&mut self) {
        self.last_seen = current_timestamp();
    }

    /// Update the load metric
    pub fn update_load(&mut self, load: f32) {
        self.load = load.clamp(0.0, 1.0);
    }
}

/// Routing table for semantic DHT
#[derive(Debug)]
pub struct SemanticRoutingTable {
    /// Configuration
    config: SemanticDHTConfig,
    /// Known peers with their embeddings
    peers: Arc<RwLock<HashMap<PeerId, SemanticPeer>>>,
    /// Cluster assignments
    clusters: Arc<RwLock<HashMap<usize, Vec<PeerId>>>>,
    /// Local peer's embedding
    local_embedding: Arc<RwLock<Vec<f32>>>,
    /// Route cache: maps embedding hash to best peers (for query routing optimization)
    route_cache: Arc<RwLock<lru::LruCache<u64, Vec<PeerId>>>>,
}

impl SemanticRoutingTable {
    /// Create a new semantic routing table
    pub fn new(config: SemanticDHTConfig) -> Self {
        let local_embedding = vec![0.0; config.embedding_dim];
        Self {
            config,
            peers: Arc::new(RwLock::new(HashMap::new())),
            clusters: Arc::new(RwLock::new(HashMap::new())),
            local_embedding: Arc::new(RwLock::new(local_embedding)),
            route_cache: Arc::new(RwLock::new(lru::LruCache::new(
                std::num::NonZeroUsize::new(1000).expect("1000 > 0"),
            ))),
        }
    }

    /// Update local peer's embedding based on stored data
    pub fn update_local_embedding(&self, embedding: Vec<f32>) -> Result<()> {
        if embedding.len() != self.config.embedding_dim {
            return Err(Error::InvalidInput(format!(
                "Expected embedding dimension {}, got {}",
                self.config.embedding_dim,
                embedding.len()
            )));
        }
        *self.local_embedding.write() = embedding;
        Ok(())
    }

    /// Add or update a peer in the routing table
    pub fn add_peer(&self, peer: SemanticPeer) -> Result<()> {
        if peer.embedding.len() != self.config.embedding_dim {
            return Err(Error::InvalidInput(format!(
                "Expected embedding dimension {}, got {}",
                self.config.embedding_dim,
                peer.embedding.len()
            )));
        }
        self.peers.write().insert(peer.peer_id, peer);
        Ok(())
    }

    /// Remove a peer from the routing table
    pub fn remove_peer(&self, peer_id: &PeerId) {
        self.peers.write().remove(peer_id);
    }

    /// Find k nearest peers to a given embedding (greedy routing)
    pub fn find_nearest_peers(&self, embedding: &[f32], k: usize) -> Vec<(PeerId, f32)> {
        // Check route cache first
        if let Some(cached_peers) = self.get_cached_route(embedding) {
            // Return cached peers with recomputed distances for accuracy
            let peers = self.peers.read();
            let result: Vec<(PeerId, f32)> = cached_peers
                .iter()
                .filter_map(|peer_id| {
                    peers.get(peer_id).map(|peer| {
                        let distance = self.compute_distance(embedding, &peer.embedding);
                        (*peer_id, distance)
                    })
                })
                .take(k)
                .collect();

            if result.len() == k {
                return result;
            }
            // Cache was stale, fall through to recompute
        }

        let peers = self.peers.read();
        let mut distances: Vec<(PeerId, f32)> = peers
            .values()
            .map(|peer| {
                let distance = self.compute_distance(embedding, &peer.embedding);
                (peer.peer_id, distance)
            })
            .collect();

        // Sort by distance (ascending for L2, descending for cosine similarity)
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let result: Vec<(PeerId, f32)> = distances.into_iter().take(k).collect();

        // Cache the routing decision
        let peer_ids: Vec<PeerId> = result.iter().map(|(id, _)| *id).collect();
        drop(peers);
        self.cache_route(embedding, peer_ids);

        result
    }

    /// Find k nearest peers with load balancing consideration
    pub fn find_nearest_peers_balanced(&self, embedding: &[f32], k: usize) -> Vec<(PeerId, f32)> {
        let peers = self.peers.read();
        let mut scored_peers: Vec<(PeerId, f32)> = peers
            .values()
            .map(|peer| {
                let distance = self.compute_distance(embedding, &peer.embedding);
                // Penalize overloaded peers: score = distance * (1 + load)
                let score = distance * (1.0 + peer.load);
                (peer.peer_id, score)
            })
            .collect();

        scored_peers.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        scored_peers.into_iter().take(k).collect()
    }

    /// Get peers in a specific cluster
    pub fn get_cluster_peers(&self, cluster_id: usize) -> Vec<PeerId> {
        self.clusters
            .read()
            .get(&cluster_id)
            .cloned()
            .unwrap_or_default()
    }

    /// Get number of peers
    pub fn num_peers(&self) -> usize {
        self.peers.read().len()
    }

    /// Get number of clusters
    pub fn num_clusters(&self) -> usize {
        self.clusters.read().len()
    }

    /// Hash an embedding for route caching
    fn hash_embedding(embedding: &[f32]) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        // Hash first 8 dimensions for efficiency (representative sample)
        for &val in embedding.iter().take(8) {
            // Convert to bits for consistent hashing
            val.to_bits().hash(&mut hasher);
        }
        hasher.finish()
    }

    /// Check route cache for cached routing decision
    pub fn get_cached_route(&self, embedding: &[f32]) -> Option<Vec<PeerId>> {
        let hash = Self::hash_embedding(embedding);
        self.route_cache.write().get(&hash).cloned()
    }

    /// Cache a routing decision for future queries
    pub fn cache_route(&self, embedding: &[f32], peers: Vec<PeerId>) {
        let hash = Self::hash_embedding(embedding);
        self.route_cache.write().put(hash, peers);
    }

    /// Clear the route cache (useful when network topology changes significantly)
    pub fn clear_route_cache(&self) {
        self.route_cache.write().clear();
    }

    /// Get route cache statistics
    pub fn route_cache_stats(&self) -> (usize, usize) {
        let cache = self.route_cache.read();
        (cache.len(), cache.cap().get())
    }

    /// Update peer clusters using k-means clustering
    pub fn update_clusters(&self, num_clusters: usize) -> Result<()> {
        let peers = self.peers.read();
        if peers.is_empty() {
            return Ok(());
        }

        let embeddings: Vec<Vec<f32>> = peers.values().map(|p| p.embedding.clone()).collect();
        let peer_ids: Vec<PeerId> = peers.keys().cloned().collect();
        drop(peers);

        // Simple k-means clustering
        let assignments = self.kmeans_clustering(&embeddings, num_clusters);

        // Update peer cluster assignments
        let mut peers_write = self.peers.write();
        let mut clusters_write = self.clusters.write();
        clusters_write.clear();

        for (peer_id, cluster_id) in peer_ids.iter().zip(assignments.iter()) {
            if let Some(peer) = peers_write.get_mut(peer_id) {
                peer.cluster_id = Some(*cluster_id);
            }
            clusters_write
                .entry(*cluster_id)
                .or_default()
                .push(*peer_id);
        }

        Ok(())
    }

    /// Compute distance between two embeddings
    fn compute_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        match self.config.distance_metric {
            DistanceMetric::L2 => crate::simd::l2_distance(a, b),
            DistanceMetric::Cosine => crate::simd::cosine_distance(a, b),
            DistanceMetric::DotProduct => -crate::simd::dot_product(a, b), // Negative for similarity
        }
    }

    /// Simple k-means clustering implementation
    fn kmeans_clustering(&self, embeddings: &[Vec<f32>], k: usize) -> Vec<usize> {
        if embeddings.is_empty() || k == 0 {
            return Vec::new();
        }

        let k = k.min(embeddings.len());
        let dim = embeddings[0].len();

        // Initialize centroids randomly
        let mut centroids: Vec<Vec<f32>> = (0..k)
            .map(|i| embeddings[i % embeddings.len()].clone())
            .collect();

        let mut assignments = vec![0; embeddings.len()];
        let max_iterations = 10;

        for _ in 0..max_iterations {
            // Assignment step
            for (i, embedding) in embeddings.iter().enumerate() {
                let mut min_dist = f32::MAX;
                let mut best_cluster = 0;

                for (cluster_id, centroid) in centroids.iter().enumerate() {
                    let dist = self.compute_distance(embedding, centroid);
                    if dist < min_dist {
                        min_dist = dist;
                        best_cluster = cluster_id;
                    }
                }
                assignments[i] = best_cluster;
            }

            // Update step
            let mut new_centroids = vec![vec![0.0; dim]; k];
            let mut counts = vec![0; k];

            for (embedding, &cluster_id) in embeddings.iter().zip(assignments.iter()) {
                for (j, &val) in embedding.iter().enumerate() {
                    new_centroids[cluster_id][j] += val;
                }
                counts[cluster_id] += 1;
            }

            for (cluster_id, count) in counts.iter().enumerate() {
                if *count > 0 {
                    for val in new_centroids[cluster_id].iter_mut().take(dim) {
                        *val /= *count as f32;
                    }
                }
            }

            centroids = new_centroids;
        }

        assignments
    }
}

/// DHT query for distributed search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DHTQuery {
    /// Query embedding
    pub embedding: Vec<f32>,
    /// Number of results requested
    pub k: usize,
    /// Query ID for tracking
    pub query_id: String,
    /// TTL (time to live) for query propagation
    pub ttl: usize,
    /// Peers already visited (to prevent loops) - serialized as strings
    #[serde(skip)]
    pub visited: HashSet<PeerId>,
}

/// DHT query response
#[derive(Debug, Clone)]
pub struct DHTQueryResponse {
    /// Query ID
    pub query_id: String,
    /// Results from this peer
    pub results: Vec<SearchResult>,
    /// Responding peer ID
    pub peer_id: PeerId,
}

/// Replication strategy for fault tolerance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReplicationStrategy {
    /// Replicate to k nearest peers
    NearestPeers(usize),
    /// Replicate to peers in same cluster
    SameCluster,
    /// Replicate to peers across different clusters
    CrossCluster(usize),
}

/// Entry in the distributed index
#[derive(Debug, Clone)]
pub struct DHTEntry {
    /// Content ID
    pub cid: Cid,
    /// Embedding
    pub embedding: Vec<f32>,
    /// Primary peer responsible for this entry
    pub primary_peer: PeerId,
    /// Replica peers
    pub replicas: Vec<PeerId>,
}

/// Statistics for the semantic DHT
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticDHTStats {
    /// Number of known peers
    pub num_peers: usize,
    /// Number of clusters
    pub num_clusters: usize,
    /// Number of entries in local index
    pub num_local_entries: usize,
    /// Number of queries processed
    pub queries_processed: u64,
    /// Average query latency in milliseconds
    pub avg_query_latency_ms: f64,
    /// Number of multi-hop queries
    pub multi_hop_queries: u64,
}

/// Get current timestamp in seconds
fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system time is after UNIX epoch")
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_routing_table_creation() {
        let config = SemanticDHTConfig::default();
        let table = SemanticRoutingTable::new(config);

        let local_emb = vec![0.5; 768];
        assert!(table.update_local_embedding(local_emb).is_ok());
    }

    #[test]
    fn test_add_peer() {
        let config = SemanticDHTConfig::default();
        let table = SemanticRoutingTable::new(config);

        let peer_id = PeerId::random();
        let embedding = vec![0.5; 768];
        let peer = SemanticPeer::new(peer_id, embedding);

        assert!(table.add_peer(peer).is_ok());
    }

    #[test]
    fn test_find_nearest_peers() {
        let config = SemanticDHTConfig::default();
        let table = SemanticRoutingTable::new(config);

        // Add some peers
        for i in 0..10 {
            let peer_id = PeerId::random();
            let embedding = vec![i as f32 * 0.1; 768];
            let peer = SemanticPeer::new(peer_id, embedding);
            table
                .add_peer(peer)
                .expect("test: add_peer with valid embedding should succeed");
        }

        let query_embedding = vec![0.5; 768];
        let nearest = table.find_nearest_peers(&query_embedding, 3);

        assert_eq!(nearest.len(), 3);
    }

    #[test]
    fn test_clustering() {
        let config = SemanticDHTConfig::default();
        let table = SemanticRoutingTable::new(config);

        // Add peers with distinct embeddings
        for i in 0..20 {
            let peer_id = PeerId::random();
            let mut embedding = vec![0.0; 768];
            // Create two clusters
            if i < 10 {
                embedding[0] = 1.0;
            } else {
                embedding[0] = -1.0;
            }
            let peer = SemanticPeer::new(peer_id, embedding);
            table
                .add_peer(peer)
                .expect("test: add_peer with valid embedding should succeed");
        }

        assert!(table.update_clusters(2).is_ok());

        // Check that clusters were assigned
        let cluster0 = table.get_cluster_peers(0);
        let cluster1 = table.get_cluster_peers(1);

        assert!(!cluster0.is_empty() || !cluster1.is_empty());
    }

    #[test]
    fn test_load_balancing() {
        let config = SemanticDHTConfig::default();
        let table = SemanticRoutingTable::new(config);

        // Add peers with different loads
        for i in 0..5 {
            let peer_id = PeerId::random();
            let embedding = vec![0.5; 768];
            let mut peer = SemanticPeer::new(peer_id, embedding);
            peer.update_load(i as f32 * 0.2); // Load: 0.0, 0.2, 0.4, 0.6, 0.8
            table
                .add_peer(peer)
                .expect("test: add_peer with valid embedding should succeed");
        }

        let query_embedding = vec![0.5; 768];
        let balanced = table.find_nearest_peers_balanced(&query_embedding, 3);

        assert_eq!(balanced.len(), 3);
        // Lower load peers should be preferred
    }

    #[test]
    fn test_route_caching() {
        let config = SemanticDHTConfig::default();
        let table = SemanticRoutingTable::new(config);

        // Add some peers
        for i in 0..10 {
            let peer_id = PeerId::random();
            let embedding = vec![i as f32 * 0.1; 768];
            let peer = SemanticPeer::new(peer_id, embedding);
            table
                .add_peer(peer)
                .expect("test: add_peer with valid embedding should succeed");
        }

        let query_embedding = vec![0.5; 768];

        // First query should not be cached
        let (cache_size_before, _) = table.route_cache_stats();
        assert_eq!(cache_size_before, 0);

        let result1 = table.find_nearest_peers(&query_embedding, 3);
        assert_eq!(result1.len(), 3);

        // After first query, should be cached
        let (cache_size_after, _) = table.route_cache_stats();
        assert_eq!(cache_size_after, 1);

        // Second query with same embedding should use cache
        let result2 = table.find_nearest_peers(&query_embedding, 3);
        assert_eq!(result2.len(), 3);

        // Results should be the same peer IDs
        let ids1: Vec<_> = result1.iter().map(|(id, _)| id).collect();
        let ids2: Vec<_> = result2.iter().map(|(id, _)| id).collect();
        assert_eq!(ids1, ids2);

        // Test cache clearing
        table.clear_route_cache();
        let (cache_size_cleared, _) = table.route_cache_stats();
        assert_eq!(cache_size_cleared, 0);
    }
}
