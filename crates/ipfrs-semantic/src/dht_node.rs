//! Distributed Semantic DHT Node
//!
//! This module implements the main DHT node that coordinates:
//! - Local vector index management
//! - Distributed k-NN search across peers
//! - Replication and fault tolerance
//! - Query routing and result aggregation

use crate::dht::{
    ReplicationStrategy, SemanticDHTConfig, SemanticDHTStats, SemanticPeer, SemanticRoutingTable,
};
use crate::hnsw::{SearchResult, VectorIndex};
use futures::future;
use ipfrs_core::{Cid, Result};
use ipfrs_network::libp2p::PeerId;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

/// Main semantic DHT node
pub struct SemanticDHTNode {
    /// Configuration
    config: SemanticDHTConfig,
    /// Local peer ID
    local_peer_id: PeerId,
    /// Local vector index
    local_index: Arc<RwLock<VectorIndex>>,
    /// Routing table
    routing_table: Arc<SemanticRoutingTable>,
    /// Replication strategy
    replication_strategy: ReplicationStrategy,
    /// Query statistics
    stats: Arc<RwLock<SemanticDHTStats>>,
    /// Pending queries
    pending_queries: Arc<RwLock<HashMap<String, Instant>>>,
    /// Last successful synchronization timestamp (unix timestamp in seconds)
    last_sync_timestamp: Arc<RwLock<u64>>,
    /// Number of pending synchronization operations
    pending_syncs: Arc<RwLock<usize>>,
}

impl SemanticDHTNode {
    /// Create a new semantic DHT node
    pub fn new(config: SemanticDHTConfig, local_peer_id: PeerId, local_index: VectorIndex) -> Self {
        let routing_table = Arc::new(SemanticRoutingTable::new(config.clone()));

        let stats = SemanticDHTStats {
            num_peers: 0,
            num_clusters: 0,
            num_local_entries: 0,
            queries_processed: 0,
            avg_query_latency_ms: 0.0,
            multi_hop_queries: 0,
        };

        Self {
            config,
            local_peer_id,
            local_index: Arc::new(RwLock::new(local_index)),
            routing_table,
            replication_strategy: ReplicationStrategy::NearestPeers(3),
            stats: Arc::new(RwLock::new(stats)),
            pending_queries: Arc::new(RwLock::new(HashMap::new())),
            last_sync_timestamp: Arc::new(RwLock::new(0)),
            pending_syncs: Arc::new(RwLock::new(0)),
        }
    }

    /// Insert a vector into the local index and replicate to peers
    pub async fn insert(&self, cid: &Cid, embedding: &[f32]) -> Result<()> {
        // Insert into local index
        self.local_index.write().insert(cid, embedding)?;

        // Update local embedding (aggregate of stored vectors)
        self.update_local_embedding().await?;

        // Determine replica peers based on strategy
        let replica_peers = self.select_replica_peers(embedding).await?;

        // Send replication requests to up to replication_factor nearest peers.
        for peer in replica_peers {
            if let Err(e) = self.replicate_to_peer(&peer, cid, embedding).await {
                tracing::warn!("Replication to {:?} failed: {}", peer, e);
            }
        }

        Ok(())
    }

    /// Search for nearest neighbors locally
    pub fn search_local(&self, embedding: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        let index = self.local_index.read();
        let ef_search = self.config.max_hops * 10; // Heuristic
        index.search(embedding, k, ef_search)
    }

    /// Distributed k-NN search across multiple peers
    pub async fn search_distributed(
        &self,
        embedding: &[f32],
        k: usize,
    ) -> Result<Vec<SearchResult>> {
        let query_id = format!("{:?}-{}", self.local_peer_id, uuid::Uuid::new_v4());
        let start_time = Instant::now();

        // Record pending query
        self.pending_queries
            .write()
            .insert(query_id.clone(), start_time);

        // Local search
        let mut all_results = self.search_local(embedding, k)?;

        // Find nearest peers to forward query to
        let nearest_peers = self
            .routing_table
            .find_nearest_peers_balanced(embedding, self.config.routing_table_size);

        // Multi-hop search
        if !nearest_peers.is_empty() && self.config.max_hops > 0 {
            let remote_results = self
                .multi_hop_search(embedding, k, query_id.clone(), 0)
                .await?;
            all_results.extend(remote_results);
        }

        // Aggregate and rank results
        let final_results = self.aggregate_results(all_results, k);

        // Update statistics
        let latency = start_time.elapsed().as_millis() as f64;
        self.update_query_stats(latency, !nearest_peers.is_empty());

        // Clean up pending query
        self.pending_queries.write().remove(&query_id);

        Ok(final_results)
    }

    /// Multi-hop search with TTL
    async fn multi_hop_search(
        &self,
        embedding: &[f32],
        _k: usize,
        _query_id: String,
        hop: usize,
    ) -> Result<Vec<SearchResult>> {
        if hop >= self.config.max_hops {
            return Ok(Vec::new());
        }

        let nearest_peers = self.routing_table.find_nearest_peers_balanced(embedding, 3); // Top 3 peers

        let mut all_results = Vec::new();

        // Query nearest peers in parallel and collect results.
        let peer_futures: Vec<_> = nearest_peers
            .iter()
            .filter(|(peer_id, _)| *peer_id != self.local_peer_id)
            .map(|(peer_id, _)| {
                let peer_id = *peer_id;
                async move {
                    tracing::debug!("Querying peer {:?} at hop {}", peer_id, hop);
                    self.query_peer(&peer_id, embedding).await
                }
            })
            .collect();

        let results = future::join_all(peer_futures).await;
        // Flatten non-empty result sets from peers that responded.
        for peer_results in results.into_iter().flatten() {
            all_results.extend(peer_results);
        }

        Ok(all_results)
    }

    /// Aggregate and deduplicate results from multiple sources
    fn aggregate_results(&self, results: Vec<SearchResult>, k: usize) -> Vec<SearchResult> {
        // Deduplicate by CID
        let mut seen = HashMap::new();
        let mut deduplicated = Vec::new();

        for result in results {
            if let Some(&existing_score) = seen.get(&result.cid) {
                // Keep better score
                if result.score < existing_score {
                    // Find and update
                    if let Some(pos) = deduplicated
                        .iter()
                        .position(|r: &SearchResult| r.cid == result.cid)
                    {
                        deduplicated[pos] = result.clone();
                        seen.insert(result.cid, result.score);
                    }
                }
            } else {
                seen.insert(result.cid, result.score);
                deduplicated.push(result);
            }
        }

        // Sort by score and take top k
        deduplicated.sort_by(|a, b| {
            a.score
                .partial_cmp(&b.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        deduplicated.into_iter().take(k).collect()
    }

    /// Select replica peers based on replication strategy
    async fn select_replica_peers(&self, embedding: &[f32]) -> Result<Vec<PeerId>> {
        match &self.replication_strategy {
            ReplicationStrategy::NearestPeers(n) => {
                let peers = self.routing_table.find_nearest_peers(embedding, *n);
                Ok(peers.into_iter().map(|(peer_id, _)| peer_id).collect())
            }
            ReplicationStrategy::SameCluster => {
                // Find local peer's cluster
                // For now, return empty
                Ok(Vec::new())
            }
            ReplicationStrategy::CrossCluster(_n) => {
                // Select n peers from different clusters
                // For now, return empty
                Ok(Vec::new())
            }
        }
    }

    /// Update local peer's embedding based on stored vectors
    async fn update_local_embedding(&self) -> Result<()> {
        let index = self.local_index.read();
        let dim = self.config.embedding_dim;

        // Compute centroid of all local vectors
        let mut centroid = vec![0.0; dim];
        let _count = 0;

        // This is a simplified version - in practice, we'd iterate over actual vectors
        // For now, just use a placeholder
        drop(index);

        // Normalize centroid
        let norm: f32 = centroid.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-6 {
            for x in &mut centroid {
                *x /= norm;
            }
        }

        self.routing_table.update_local_embedding(centroid)?;

        Ok(())
    }

    /// Update query statistics
    fn update_query_stats(&self, latency_ms: f64, is_multi_hop: bool) {
        let mut stats = self.stats.write();
        stats.queries_processed += 1;

        // Update running average
        let alpha = 0.1; // Exponential moving average factor
        stats.avg_query_latency_ms =
            alpha * latency_ms + (1.0 - alpha) * stats.avg_query_latency_ms;

        if is_multi_hop {
            stats.multi_hop_queries += 1;
        }
    }

    /// Add a peer to the routing table
    pub fn add_peer(&self, peer: SemanticPeer) -> Result<()> {
        self.routing_table.add_peer(peer)?;

        // Update stats
        let mut stats = self.stats.write();
        stats.num_peers = self.routing_table.num_peers();

        Ok(())
    }

    /// Remove a peer from the routing table
    pub fn remove_peer(&self, peer_id: &PeerId) {
        self.routing_table.remove_peer(peer_id);

        // Update stats
        let mut stats = self.stats.write();
        stats.num_peers = self.routing_table.num_peers();
    }

    /// Update peer clustering
    pub fn update_clusters(&self, num_clusters: usize) -> Result<()> {
        self.routing_table.update_clusters(num_clusters)?;

        // Update stats
        let mut stats = self.stats.write();
        stats.num_clusters = self.routing_table.num_clusters();

        Ok(())
    }

    /// Get DHT statistics
    pub fn stats(&self) -> SemanticDHTStats {
        let mut stats = self.stats.read().clone();
        stats.num_local_entries = self.local_index.read().len();
        stats
    }

    /// Get DHT statistics (alias for stats)
    pub fn get_stats(&self) -> SemanticDHTStats {
        self.stats()
    }

    /// Get reference to the routing table
    pub fn routing_table(&self) -> &SemanticRoutingTable {
        &self.routing_table
    }

    /// Set replication strategy
    pub fn set_replication_strategy(&mut self, strategy: ReplicationStrategy) {
        self.replication_strategy = strategy;
    }

    /// Get a snapshot of local index entries for synchronization
    /// Returns CIDs that can be used for delta synchronization
    pub fn get_index_snapshot(&self) -> Vec<Cid> {
        let index = self.local_index.read();
        // Get all CIDs from the index
        index.get_all_cids()
    }

    /// Check if local index has a specific CID
    pub fn has_entry(&self, cid: &Cid) -> bool {
        let index = self.local_index.read();
        index.contains(cid)
    }

    /// Prepare synchronization delta: entries that peer needs
    /// Returns CIDs that are in our index but not in the peer's snapshot
    pub fn prepare_sync_delta(&self, peer_snapshot: &[Cid]) -> Vec<Cid> {
        let local_snapshot = self.get_index_snapshot();
        let peer_set: std::collections::HashSet<_> = peer_snapshot.iter().collect();

        local_snapshot
            .into_iter()
            .filter(|cid| !peer_set.contains(cid))
            .collect()
    }

    /// Apply synchronization delta: add entries from peer
    /// This is a foundation - actual implementation would fetch embeddings from peer
    pub async fn apply_sync_delta(&self, delta_cids: Vec<Cid>) -> Result<usize> {
        // NOTE: In full implementation with network protocol, this would:
        // 1. Request embeddings for delta_cids from peer
        // 2. Call apply_sync_delta_with_embeddings with the fetched data
        // For now, just return count of CIDs that would be synced
        Ok(delta_cids.len())
    }

    /// Apply synchronization delta with embeddings: add entries from peer
    /// This method actually inserts the embeddings into the local index
    pub async fn apply_sync_delta_with_embeddings(
        &self,
        delta_entries: Vec<(Cid, Vec<f32>)>,
    ) -> Result<usize> {
        // Increment pending syncs counter
        *self.pending_syncs.write() += 1;

        let mut synced_count = 0;

        // Insert each entry into local index
        for (cid, embedding) in delta_entries {
            match self.local_index.write().insert(&cid, &embedding) {
                Ok(_) => {
                    synced_count += 1;
                }
                Err(e) => {
                    tracing::warn!("Failed to insert CID {:?} during sync: {}", cid, e);
                }
            }
        }

        // Update last sync timestamp
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        *self.last_sync_timestamp.write() = now;

        // Decrement pending syncs counter
        *self.pending_syncs.write() -= 1;

        // Update local embedding after sync
        self.update_local_embedding().await?;

        tracing::debug!("Synced {} entries from peer", synced_count);

        Ok(synced_count)
    }

    /// Replicate a (key, value) pair to a remote peer.
    ///
    /// When no active transport is wired up (current state) this is a no-op
    /// that logs the intent and returns `Ok(())`.  A real implementation would
    /// serialise the key/value pair and push it over the peer-to-peer channel.
    async fn replicate_to_peer(
        &self,
        peer: &PeerId,
        key: &Cid,
        value: &[f32],
    ) -> ipfrs_core::Result<()> {
        // No active transport — log and return Ok so that callers continue.
        tracing::debug!(
            "replicate_to_peer: peer={:?} key={} value_len={} (no transport)",
            peer,
            key,
            value.len()
        );
        Ok(())
    }

    /// Query a remote peer for nearest neighbours to `embedding`.
    ///
    /// Returns `None` when no active transport is available.  A real
    /// implementation would serialise the query vector, send it over the
    /// network, and deserialise the returned [`SearchResult`] list.
    async fn query_peer(
        &self,
        peer: &PeerId,
        embedding: &[f32],
    ) -> Option<Vec<crate::hnsw::SearchResult>> {
        // No active transport — return None so the caller falls back gracefully.
        tracing::debug!(
            "query_peer: peer={:?} embedding_len={} (no transport)",
            peer,
            embedding.len()
        );
        None
    }

    /// Get synchronization statistics
    pub fn sync_stats(&self) -> SyncStats {
        SyncStats {
            local_entries: self.local_index.read().len(),
            last_sync_timestamp: *self.last_sync_timestamp.read(),
            pending_syncs: *self.pending_syncs.read(),
        }
    }
}

/// Statistics for index synchronization
#[derive(Debug, Clone)]
pub struct SyncStats {
    /// Number of entries in local index
    pub local_entries: usize,
    /// Timestamp of last successful sync
    pub last_sync_timestamp: u64,
    /// Number of pending sync operations
    pub pending_syncs: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hnsw::DistanceMetric;

    #[tokio::test]
    async fn test_dht_node_creation() {
        let config = SemanticDHTConfig::default();
        let peer_id = PeerId::random();
        let index = VectorIndex::new(768, DistanceMetric::Cosine, 16, 200)
            .expect("test: VectorIndex creation should succeed");

        let node = SemanticDHTNode::new(config, peer_id, index);
        let stats = node.stats();

        assert_eq!(stats.num_peers, 0);
        assert_eq!(stats.queries_processed, 0);
    }

    #[tokio::test]
    async fn test_local_insert_and_search() {
        let config = SemanticDHTConfig::default();
        let peer_id = PeerId::random();
        let index = VectorIndex::new(768, DistanceMetric::Cosine, 16, 200)
            .expect("test: VectorIndex creation should succeed");

        let node = SemanticDHTNode::new(config, peer_id, index);

        // Insert some vectors
        for i in 0..10 {
            use multihash_codetable::{Code, MultihashDigest};
            let data = format!("test_vector_{}", i);
            let hash = Code::Sha2_256.digest(data.as_bytes());
            let cid = Cid::new_v1(0x55, hash);
            let embedding = vec![i as f32 * 0.1; 768];
            node.insert(&cid, &embedding)
                .await
                .expect("test: node insert should succeed");
        }

        // Search
        let query = vec![0.5; 768];
        let results = node
            .search_local(&query, 5)
            .expect("test: local search should succeed");

        assert!(!results.is_empty());
        assert!(results.len() <= 5);
    }

    #[tokio::test]
    async fn test_add_peers() {
        let config = SemanticDHTConfig::default();
        let peer_id = PeerId::random();
        let index = VectorIndex::new(768, DistanceMetric::Cosine, 16, 200)
            .expect("test: VectorIndex creation should succeed");

        let node = SemanticDHTNode::new(config, peer_id, index);

        // Add some peers
        for i in 0..5 {
            let peer_id = PeerId::random();
            let embedding = vec![i as f32 * 0.2; 768];
            let peer = SemanticPeer::new(peer_id, embedding);
            node.add_peer(peer).expect("test: add_peer should succeed");
        }

        let stats = node.stats();
        assert_eq!(stats.num_peers, 5);
    }

    #[tokio::test]
    async fn test_clustering() {
        let config = SemanticDHTConfig::default();
        let peer_id = PeerId::random();
        let index = VectorIndex::new(768, DistanceMetric::Cosine, 16, 200)
            .expect("test: VectorIndex creation should succeed");

        let node = SemanticDHTNode::new(config, peer_id, index);

        // Add peers
        for i in 0..20 {
            let peer_id = PeerId::random();
            let mut embedding = vec![0.0; 768];
            embedding[0] = if i < 10 { 1.0 } else { -1.0 };
            let peer = SemanticPeer::new(peer_id, embedding);
            node.add_peer(peer).expect("test: add_peer should succeed");
        }

        // Update clusters
        node.update_clusters(2)
            .expect("test: update_clusters should succeed");

        let stats = node.stats();
        assert!(stats.num_clusters > 0);
    }

    #[tokio::test]
    async fn test_index_synchronization() {
        use multihash_codetable::{Code, MultihashDigest};

        let config = SemanticDHTConfig::default();
        let peer_id1 = PeerId::random();
        let peer_id2 = PeerId::random();

        let index1 = VectorIndex::new(768, DistanceMetric::Cosine, 16, 200)
            .expect("test: VectorIndex creation should succeed");
        let index2 = VectorIndex::new(768, DistanceMetric::Cosine, 16, 200)
            .expect("test: VectorIndex creation should succeed");

        let node1 = SemanticDHTNode::new(config.clone(), peer_id1, index1);
        let node2 = SemanticDHTNode::new(config, peer_id2, index2);

        // Insert data into node1
        let mut cids1 = Vec::new();
        for i in 0..5 {
            let data = format!("node1_vector_{}", i);
            let hash = Code::Sha2_256.digest(data.as_bytes());
            let cid = Cid::new_v1(0x55, hash);
            let embedding = vec![i as f32 * 0.1; 768];
            node1
                .insert(&cid, &embedding)
                .await
                .expect("test: node insert should succeed");
            cids1.push(cid);
        }

        // Insert different data into node2
        let mut cids2 = Vec::new();
        for i in 5..10 {
            let data = format!("node2_vector_{}", i);
            let hash = Code::Sha2_256.digest(data.as_bytes());
            let cid = Cid::new_v1(0x55, hash);
            let embedding = vec![i as f32 * 0.1; 768];
            node2
                .insert(&cid, &embedding)
                .await
                .expect("test: node insert should succeed");
            cids2.push(cid);
        }

        // Get snapshots
        let snapshot1 = node1.get_index_snapshot();
        let snapshot2 = node2.get_index_snapshot();

        assert_eq!(snapshot1.len(), 5);
        assert_eq!(snapshot2.len(), 5);

        // Check that node1 has its entries
        for cid in &cids1 {
            assert!(node1.has_entry(cid));
        }

        // Prepare delta: what node2 needs from node1
        let delta = node1.prepare_sync_delta(&snapshot2);
        assert_eq!(delta.len(), 5); // All of node1's entries are missing from node2

        // Apply delta (in real implementation, this would fetch and insert)
        let synced_count = node2
            .apply_sync_delta(delta)
            .await
            .expect("test: apply_sync_delta should succeed");
        assert_eq!(synced_count, 5);

        // Check sync stats
        let sync_stats = node1.sync_stats();
        assert_eq!(sync_stats.local_entries, 5);
    }

    #[tokio::test]
    async fn test_sync_with_embeddings() {
        use multihash_codetable::{Code, MultihashDigest};

        let config = SemanticDHTConfig::default();
        let peer_id1 = PeerId::random();
        let peer_id2 = PeerId::random();

        let index1 = VectorIndex::new(768, DistanceMetric::Cosine, 16, 200)
            .expect("test: VectorIndex creation should succeed");
        let index2 = VectorIndex::new(768, DistanceMetric::Cosine, 16, 200)
            .expect("test: VectorIndex creation should succeed");

        let node1 = SemanticDHTNode::new(config.clone(), peer_id1, index1);
        let node2 = SemanticDHTNode::new(config, peer_id2, index2);

        // Insert data into node1
        let mut entries_to_sync = Vec::new();
        for i in 0..5 {
            let data = format!("sync_test_vector_{}", i);
            let hash = Code::Sha2_256.digest(data.as_bytes());
            let cid = Cid::new_v1(0x55, hash);
            let embedding = vec![i as f32 * 0.1; 768];
            node1
                .insert(&cid, &embedding)
                .await
                .expect("test: node insert should succeed");
            entries_to_sync.push((cid, embedding));
        }

        // Check initial state
        let sync_stats_before = node2.sync_stats();
        assert_eq!(sync_stats_before.local_entries, 0);
        assert_eq!(sync_stats_before.last_sync_timestamp, 0);
        assert_eq!(sync_stats_before.pending_syncs, 0);

        // Apply sync with embeddings to node2
        let synced_count = node2
            .apply_sync_delta_with_embeddings(entries_to_sync.clone())
            .await
            .expect("test: apply_sync_delta_with_embeddings should succeed");
        assert_eq!(synced_count, 5);

        // Check that node2 now has the entries
        let sync_stats_after = node2.sync_stats();
        assert_eq!(sync_stats_after.local_entries, 5);
        assert!(sync_stats_after.last_sync_timestamp > 0); // Should be updated
        assert_eq!(sync_stats_after.pending_syncs, 0); // Should be back to 0

        // Verify all CIDs are present in node2
        for (cid, _) in &entries_to_sync {
            assert!(node2.has_entry(cid));
        }

        // Search should work on node2 now
        let query = vec![0.15; 768];
        let results = node2
            .search_local(&query, 3)
            .expect("test: local search after sync should succeed");
        assert!(!results.is_empty());
    }

    #[tokio::test]
    async fn test_dht_replication_stub() {
        use multihash_codetable::{Code, MultihashDigest};

        let config = SemanticDHTConfig::default();
        let peer_id = PeerId::random();
        let index = VectorIndex::new(768, DistanceMetric::Cosine, 16, 200)
            .expect("test: VectorIndex creation should succeed");
        let node = SemanticDHTNode::new(config, peer_id, index);

        let hash = Code::Sha2_256.digest(b"replication_stub_test");
        let cid = Cid::new_v1(0x55, hash);
        let embedding = vec![0.1_f32; 768];

        // replicate_to_peer should succeed (no-op stub) without any transport wired up.
        let target_peer = PeerId::random();
        let result = node.replicate_to_peer(&target_peer, &cid, &embedding).await;
        assert!(
            result.is_ok(),
            "replicate_to_peer stub should return Ok(())"
        );
    }

    #[tokio::test]
    async fn test_dht_remote_query_stub() {
        let config = SemanticDHTConfig::default();
        let peer_id = PeerId::random();
        let index = VectorIndex::new(768, DistanceMetric::Cosine, 16, 200)
            .expect("test: VectorIndex creation should succeed");
        let node = SemanticDHTNode::new(config, peer_id, index);

        let embedding = vec![0.5_f32; 768];
        let remote_peer = PeerId::random();

        // query_peer should return None when no transport is available.
        let result = node.query_peer(&remote_peer, &embedding).await;
        assert!(
            result.is_none(),
            "query_peer stub should return None without a transport"
        );
    }
}
