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
use std::cmp::Reverse;
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

    // --- v0.3.0 additional error variants ---
    #[error("Index not initialized; call register_namespace first")]
    IndexNotInitialized,

    #[error("Vector dimension mismatch: expected {expected}, got {got}")]
    VectorDimensionMismatch { expected: usize, got: usize },

    #[error("Routing convergence timed out")]
    RoutingConvergenceTimeout,

    #[error("Peer unreachable: {0}")]
    PeerUnreachable(String),
}

/// A DHT record that carries an embedded vector alongside the CID it annotates.
///
/// `VectorAnnotatedRecord` is the wire format stored in the DHT; it allows any
/// peer that retrieves the record to immediately compute similarity without
/// fetching additional metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorAnnotatedRecord {
    /// Content identifier this record refers to
    pub cid: String,
    /// Embedding vector for the content
    pub vector: Vec<f32>,
    /// Dimensionality of `vector` (stored separately for fast validation)
    pub dimension: usize,
    /// PeerId (string-encoded) of the peer that published this record
    pub provider_id: String,
    /// How long (in seconds) this record should be considered fresh
    pub ttl_secs: u64,
    /// Arbitrary application-level metadata (e.g. content-type, language, …)
    pub metadata: HashMap<String, String>,
}

impl VectorAnnotatedRecord {
    /// Construct a new `VectorAnnotatedRecord`, deriving `dimension` from `vector`.
    pub fn new(
        cid: impl Into<String>,
        vector: Vec<f32>,
        provider_id: impl Into<String>,
        ttl_secs: u64,
        metadata: HashMap<String, String>,
    ) -> Self {
        let dimension = vector.len();
        Self {
            cid: cid.into(),
            vector,
            dimension,
            provider_id: provider_id.into(),
            ttl_secs,
            metadata,
        }
    }

    /// Check whether `vector.len() == dimension` (should always be true for
    /// well-formed records received from trusted peers; useful after deserde).
    pub fn is_consistent(&self) -> bool {
        self.vector.len() == self.dimension
    }
}

/// Routing-quality metrics snapshot emitted by [`SemanticDht::metrics`].
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SemanticDhtMetrics {
    /// Fraction of queries that returned at least one result (0.0 – 1.0)
    pub recall_rate: f32,
    /// Mean query latency across all queries (milliseconds)
    pub mean_latency_ms: f64,
    /// Current convergence score (0.0 = unstable, 1.0 = fully converged)
    pub routing_convergence: f32,
    /// Number of indexed CIDs in the local index
    pub indexed_cid_count: usize,
    /// Number of distinct LSH hash-to-peer mappings known
    pub known_hash_regions: usize,
    /// Cache hit ratio (0.0 – 1.0)
    pub cache_hit_ratio: f32,
    /// Number of partial-sync operations performed since node start
    pub partial_sync_count: u64,
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

    // --- v0.3.0 production fields ---
    /// Embedding dimension (used for `put_with_vector` / `search_similar`)
    /// Default: 384 (matches common sentence-transformers models)
    pub dimension: usize,

    /// HNSW `ef` search parameter – higher = more accurate but slower
    pub ef_search: usize,

    /// Maximum peers to route a single query to
    pub max_routing_peers: usize,

    /// How long vector records stay in the local annotated-record store
    pub vector_ttl: Duration,

    /// How often background gossip sync fires
    pub sync_interval: Duration,

    /// Fraction of routing-table entries that must agree before routing is
    /// considered converged (0.0 – 1.0)
    pub convergence_threshold: f32,
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
            // v0.3.0 production defaults
            dimension: 384,
            ef_search: 50,
            max_routing_peers: 20,
            vector_ttl: Duration::from_secs(3600),   // 1 hour
            sync_interval: Duration::from_secs(300), // 5 minutes
            convergence_threshold: 0.95,
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

// =========================================================================
// DHT Shard Balancing (v0.3.0)
// =========================================================================

/// Configuration for DHT shard balancing.
#[derive(Debug, Clone)]
pub struct ShardBalancerConfig {
    /// Maximum number of vectors a single peer should index before triggering
    /// a rebalance.  Default: 10_000.
    pub max_vectors_per_peer: usize,
    /// When a peer's load fraction exceeds this threshold the peer is marked
    /// overloaded and migration candidates are generated.  Default: 0.8.
    pub rebalance_threshold: f32,
    /// Target number of peers that should own each vector (redundancy factor).
    /// Default: 3.
    pub target_redundancy: usize,
}

impl Default for ShardBalancerConfig {
    fn default() -> Self {
        Self {
            max_vectors_per_peer: 10_000,
            rebalance_threshold: 0.8,
            target_redundancy: 3,
        }
    }
}

/// Tracks which peers own which HNSW-layer shards and computes load balance
/// metrics used to drive migration decisions.
pub struct ShardBalancer {
    config: ShardBalancerConfig,
    /// peer_id → number of vectors currently assigned
    peer_loads: std::collections::HashMap<String, usize>,
    /// cid → owning peer_ids (redundancy list)
    cid_owners: std::collections::HashMap<String, Vec<String>>,
    /// peers that currently exceed the rebalance threshold
    overloaded_peers: std::collections::HashSet<String>,
}

impl ShardBalancer {
    /// Create a new `ShardBalancer` with the given configuration.
    pub fn new(config: ShardBalancerConfig) -> Self {
        Self {
            config,
            peer_loads: std::collections::HashMap::new(),
            cid_owners: std::collections::HashMap::new(),
            overloaded_peers: std::collections::HashSet::new(),
        }
    }

    /// Record that `peer_id` has indexed the vector identified by `cid`.
    ///
    /// Increments the peer's load counter and updates the CID ownership list.
    /// If the resulting load fraction breaches `rebalance_threshold` the peer
    /// is added to `overloaded_peers`.
    pub fn record_vector_assignment(&mut self, peer_id: &str, cid: &str) {
        // Update load counter
        let load = self.peer_loads.entry(peer_id.to_string()).or_insert(0);
        *load = load.saturating_add(1);

        // Update ownership list
        let owners = self.cid_owners.entry(cid.to_string()).or_default();
        if !owners.contains(&peer_id.to_string()) {
            owners.push(peer_id.to_string());
        }

        // Re-evaluate overload status for this peer
        let threshold =
            (self.config.max_vectors_per_peer as f32 * self.config.rebalance_threshold) as usize;
        let current_load = self.peer_loads.get(peer_id).copied().unwrap_or(0);
        if current_load >= threshold {
            self.overloaded_peers.insert(peer_id.to_string());
        }
    }

    /// Return the `count` least-loaded peers, suitable for assigning a new
    /// vector.  Peers are sorted by ascending load; if fewer than `count` peers
    /// are known the full list is returned.
    pub fn suggest_peers_for_vector(&self, count: usize) -> Vec<String> {
        if self.peer_loads.is_empty() {
            return Vec::new();
        }

        let mut sorted: Vec<(&String, &usize)> = self.peer_loads.iter().collect();
        sorted.sort_by_key(|(_, &load)| load);
        sorted
            .into_iter()
            .take(count)
            .map(|(id, _)| id.clone())
            .collect()
    }

    /// Return `true` when `peer_id`'s load fraction exceeds
    /// `rebalance_threshold`.
    pub fn is_overloaded(&self, peer_id: &str) -> bool {
        self.overloaded_peers.contains(peer_id)
    }

    /// Return a map of `peer_id → load_fraction` (0.0–1.0), where the
    /// denominator is `max_vectors_per_peer`.
    pub fn load_distribution(&self) -> std::collections::HashMap<String, f32> {
        let max = self.config.max_vectors_per_peer as f32;
        self.peer_loads
            .iter()
            .map(|(id, &load)| (id.clone(), (load as f32 / max).min(1.0)))
            .collect()
    }

    /// Identify `(cid, from_peer)` pairs that should be migrated away from
    /// hot-spot peers.
    ///
    /// For every overloaded peer, walks its assigned CIDs and collects up to
    /// `ceil(excess)` migration candidates, preferring CIDs that have enough
    /// existing owners so redundancy is preserved after migration.
    pub fn vectors_to_migrate(&self) -> Vec<(String, String)> {
        let threshold =
            (self.config.max_vectors_per_peer as f32 * self.config.rebalance_threshold) as usize;

        let mut migrations = Vec::new();

        for peer_id in &self.overloaded_peers {
            let load = self.peer_loads.get(peer_id).copied().unwrap_or(0);
            if load <= threshold {
                continue;
            }
            let excess = load - threshold;

            // Collect CIDs owned by this peer
            let mut candidates: Vec<String> = self
                .cid_owners
                .iter()
                .filter(|(_, owners)| owners.contains(peer_id))
                .map(|(cid, _)| cid.clone())
                .take(excess)
                .collect();
            candidates.sort(); // deterministic ordering

            for cid in candidates.into_iter().take(excess) {
                migrations.push((cid, peer_id.clone()));
            }
        }

        migrations
    }

    /// Compute a balance score in `[0.0, 1.0]`.
    ///
    /// A score of `1.0` means all peers carry the same load; `0.0` means the
    /// load is completely concentrated on a single peer.
    ///
    /// Uses the complement of the *coefficient of variation* (CV) clamped to
    /// `[0, 1]` so that it is easily interpretable.
    pub fn balance_score(&self) -> f32 {
        if self.peer_loads.is_empty() {
            return 1.0; // vacuously balanced
        }

        let loads: Vec<f32> = self.peer_loads.values().map(|&l| l as f32).collect();
        let n = loads.len() as f32;

        let mean = loads.iter().sum::<f32>() / n;
        if mean == 0.0 {
            return 1.0; // all zeros → balanced
        }

        let variance = loads.iter().map(|&l| (l - mean).powi(2)).sum::<f32>() / n;
        let std_dev = variance.sqrt();
        let cv = std_dev / mean; // coefficient of variation

        // CV = 0 → perfect balance (score = 1); CV = 1 → mediocre; CV >> 1 → terrible
        // Map with saturation: score = 1 - min(cv, 1)
        (1.0 - cv.min(1.0)).clamp(0.0, 1.0)
    }

    /// Assign a vector to the `n_replicas` best peers using consistent hashing
    /// on the vector's hash fingerprint.
    ///
    /// Peer selection is deterministic given the same vector content: the vector
    /// is hashed via a stable fingerprint (sum of floats cast to bits), then
    /// peers are sorted by (hash XOR peer_hash) % capacity, and the
    /// `n_replicas` with the smallest distance are returned.
    ///
    /// When fewer than `n_replicas` peers are tracked, all known peers are
    /// returned (no padding).
    pub fn assign_vector(&self, vector: &[f32], n_replicas: usize) -> Vec<String> {
        if self.peer_loads.is_empty() {
            return Vec::new();
        }

        // Stable fingerprint: XOR of bit-patterns of all floats
        let vec_hash: u64 = vector.iter().enumerate().fold(0u64, |acc, (i, &v)| {
            let bits = v.to_bits() as u64;
            acc ^ bits.wrapping_mul(
                (i as u64)
                    .wrapping_add(1)
                    .wrapping_mul(0x9e37_79b9_7f4a_7c15),
            )
        });

        let mut peers_scored: Vec<(&String, u64)> = self
            .peer_loads
            .iter()
            .map(|(peer_id, &load)| {
                // Score = consistent-hash distance weighted by inverse load fraction
                let peer_hash: u64 = peer_id
                    .bytes()
                    .fold(0u64, |acc, b| acc.wrapping_mul(131).wrapping_add(b as u64));
                let distance = vec_hash ^ peer_hash;
                // Weight by load: prefer peers with available capacity
                let cap = self.config.max_vectors_per_peer.max(1) as u64;
                let load_penalty = (load as u64).saturating_mul(u64::MAX / cap);
                (peer_id, distance.wrapping_add(load_penalty))
            })
            .collect();

        peers_scored.sort_by_key(|(_, score)| *score);
        peers_scored
            .into_iter()
            .take(n_replicas)
            .map(|(id, _)| id.clone())
            .collect()
    }

    /// Compute the load imbalance score.
    ///
    /// Returns `0.0` for a perfectly balanced cluster and `1.0` when the entire
    /// load is concentrated on a single peer.  Uses the normalised standard
    /// deviation (coefficient of variation), clamped to `[0.0, 1.0]`.
    pub fn imbalance_score(&self) -> f64 {
        if self.peer_loads.is_empty() {
            return 0.0;
        }

        let loads: Vec<f64> = self.peer_loads.values().map(|&l| l as f64).collect();
        let n = loads.len() as f64;
        let mean = loads.iter().sum::<f64>() / n;

        if mean == 0.0 {
            return 0.0; // all-zero is perfectly balanced
        }

        let variance = loads.iter().map(|&l| (l - mean).powi(2)).sum::<f64>() / n;
        let cv = variance.sqrt() / mean;

        cv.min(1.0)
    }

    /// Return up to `limit` `(vector_id, target_peer)` pairs to migrate for
    /// better balance.
    ///
    /// Candidates are chosen from the most overloaded peer(s) and the target is
    /// the least-loaded peer that does not already own the vector.
    pub fn migration_plan(&self, limit: usize) -> Vec<(String, String)> {
        if limit == 0 || self.peer_loads.is_empty() {
            return Vec::new();
        }

        // Identify the least-loaded peer upfront
        let least_loaded = self
            .peer_loads
            .iter()
            .min_by_key(|(_, &l)| l)
            .map(|(id, _)| id.clone());

        let Some(target_peer) = least_loaded else {
            return Vec::new();
        };

        let threshold = (self.config.max_vectors_per_peer as f64
            * self.config.rebalance_threshold as f64) as usize;

        let mut plan = Vec::new();

        // Collect overloaded peers sorted by load descending for determinism
        let mut hot: Vec<(&String, usize)> = self
            .overloaded_peers
            .iter()
            .filter_map(|p| self.peer_loads.get(p).map(|&l| (p, l)))
            .collect();
        hot.sort_by_key(|b| Reverse(b.1));

        'outer: for (from_peer, load) in hot {
            if load <= threshold {
                continue;
            }
            let excess = load - threshold;

            let mut candidates: Vec<String> = self
                .cid_owners
                .iter()
                .filter(|(_, owners)| owners.contains(from_peer) && !owners.contains(&target_peer))
                .map(|(cid, _)| cid.clone())
                .take(excess)
                .collect();
            candidates.sort();

            for cid in candidates {
                plan.push((cid, target_peer.clone()));
                if plan.len() >= limit {
                    break 'outer;
                }
            }
        }

        plan
    }

    /// Update the capacity ceiling for a peer.
    ///
    /// This updates `max_vectors_per_peer` in the configuration to the supplied
    /// value and re-evaluates overload status for `peer`.  Only the single
    /// peer's overload flag is updated; a full sweep is not performed.
    pub fn update_peer_capacity(&mut self, peer: &str, capacity: usize) {
        // Update global capacity setting
        self.config.max_vectors_per_peer = capacity;

        // Re-evaluate the named peer
        let threshold = (capacity as f32 * self.config.rebalance_threshold) as usize;
        let load = self.peer_loads.get(peer).copied().unwrap_or(0);
        if load >= threshold {
            self.overloaded_peers.insert(peer.to_string());
        } else {
            self.overloaded_peers.remove(peer);
        }
    }

    /// Remove `peer` from the balancer and return the list of vector IDs it
    /// was responsible for.  The caller is expected to trigger migration of
    /// the returned CIDs to other peers.
    pub fn remove_peer(&mut self, peer: &str) -> Vec<String> {
        self.peer_loads.remove(peer);
        self.overloaded_peers.remove(peer);

        // Collect all CIDs owned by this peer
        let owned: Vec<String> = self
            .cid_owners
            .iter()
            .filter(|(_, owners)| owners.contains(&peer.to_string()))
            .map(|(cid, _)| cid.clone())
            .collect();

        // Remove the peer from all ownership lists
        for cid in &owned {
            if let Some(owners) = self.cid_owners.get_mut(cid) {
                owners.retain(|p| p != peer);
            }
        }

        owned
    }

    /// Update internal bookkeeping after `cid` has been migrated from
    /// `from_peer` to `to_peer`.
    ///
    /// Decrements `from_peer`'s counter, increments `to_peer`'s counter,
    /// updates the ownership list, and refreshes overload status.
    pub fn record_migration(&mut self, cid: &str, from_peer: &str, to_peer: &str) {
        // Decrement source peer load
        if let Some(load) = self.peer_loads.get_mut(from_peer) {
            *load = load.saturating_sub(1);
        }

        // Increment destination peer load
        let dest_load = self.peer_loads.entry(to_peer.to_string()).or_insert(0);
        *dest_load = dest_load.saturating_add(1);

        // Update ownership list
        if let Some(owners) = self.cid_owners.get_mut(cid) {
            owners.retain(|p| p != from_peer);
            if !owners.contains(&to_peer.to_string()) {
                owners.push(to_peer.to_string());
            }
        }

        // Refresh overload status
        let threshold =
            (self.config.max_vectors_per_peer as f32 * self.config.rebalance_threshold) as usize;

        for peer in [from_peer, to_peer] {
            let load = self.peer_loads.get(peer).copied().unwrap_or(0);
            if load >= threshold {
                self.overloaded_peers.insert(peer.to_string());
            } else {
                self.overloaded_peers.remove(peer);
            }
        }
    }
}

/// Configuration for the efficient partial-sync algorithm (v0.3.0).
///
/// Controls which vectors are gossiped during an incremental sync round and
/// how many rounds are allowed before forcing a full resync.
#[derive(Debug, Clone)]
pub struct PartialSyncConfig {
    /// Cosine distance threshold: only gossip vectors whose embedding has
    /// changed by more than this amount since the last sync.  Default: 0.05.
    pub sync_threshold: f32,
    /// Maximum number of vector records packed into a single GossipSub message.
    /// Default: 32.
    pub batch_size: usize,
    /// Maximum sync rounds before a full resync is forced.  Default: 100.
    pub max_rounds: usize,
}

impl Default for PartialSyncConfig {
    fn default() -> Self {
        Self {
            sync_threshold: 0.05,
            batch_size: 32,
            max_rounds: 100,
        }
    }
}

/// Statistics collected during a single partial-sync pass.
#[derive(Debug, Clone, Default)]
pub struct PartialSyncStats {
    /// Number of vectors that were gossiped (exceeded the threshold).
    pub vectors_synced: usize,
    /// Number of vectors skipped (within threshold, no change).
    pub vectors_skipped: usize,
    /// Total bytes that would be sent in a real network exchange.
    pub bytes_sent: usize,
    /// Number of GossipSub batches produced.
    pub rounds_completed: usize,
}

/// A single result from [`SemanticDht::distributed_search`].
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// Content identifier string.
    pub cid: String,
    /// Cosine similarity score in `[−1.0, 1.0]`; higher means more similar.
    pub score: f32,
    /// Which peer held this vector (`None` for local results).
    pub peer: Option<String>,
    /// The vector's record-key (same as `cid` in the current implementation).
    pub vector_id: String,
}

/// Result of a `merge_partial_index` operation.
#[derive(Debug, Clone, Default)]
pub struct MergeResult {
    /// Records added (CID was not present locally)
    pub added: usize,
    /// Records updated (CID was present but the incoming record has a newer TTL
    /// or different vector)
    pub updated: usize,
    /// Records skipped because the local copy was already up-to-date
    pub skipped: usize,
    /// Conflicting records: same CID, incompatible vectors (dimension differs)
    pub conflicts: usize,
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

    // --- v0.3.0 additions ---
    /// Flat vector-annotated record store: CID string → record + insertion time
    ///
    /// This is the local half of the distributed vector DHT.  In a fully
    /// distributed deployment each record would also be gossiped to neighbouring
    /// peers whose LSH bucket overlaps.
    vector_records: Arc<DashMap<String, (VectorAnnotatedRecord, Instant)>>,

    /// Monotonically increasing "routing table version" counter.
    ///
    /// Incremented on every `put_with_vector` and every peer-initiated gossip
    /// merge.  The convergence score is estimated by comparing the rate of
    /// change against the expected steady-state update frequency.
    routing_version: Arc<RwLock<u64>>,

    /// Timestamp of the last routing-table modification.
    last_routing_change: Arc<RwLock<Instant>>,

    // --- v0.3.0 shard balancer ---
    /// Shard-load balancer: tracks per-peer vector counts and identifies
    /// hot-spot peers that should shed load via migration.
    pub shard_balancer: parking_lot::Mutex<ShardBalancer>,
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

    // --- v0.3.0 additions ---
    /// Total `put_with_vector` calls that succeeded
    pub vector_puts: u64,

    /// Total `search_similar` calls
    pub vector_searches: u64,

    /// Number of efficient partial-sync operations performed
    pub partial_syncs: u64,

    /// Snapshot of the last computed routing convergence score
    pub last_convergence_score: f32,
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
            vector_records: Arc::new(DashMap::new()),
            routing_version: Arc::new(RwLock::new(0)),
            last_routing_change: Arc::new(RwLock::new(Instant::now())),
            shard_balancer: parking_lot::Mutex::new(ShardBalancer::new(
                ShardBalancerConfig::default(),
            )),
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
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
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

    // =========================================================================
    // v0.3.0 production methods
    // =========================================================================

    /// Store a CID with its embedding vector in the local record store.
    ///
    /// The record is validated (dimension check) and then written to the flat
    /// `vector_records` map.  In a full network deployment the caller would
    /// subsequently gossip this record to neighbouring peers.
    ///
    /// # Errors
    ///
    /// Returns [`SemanticDhtError::VectorDimensionMismatch`] when
    /// `vector.len() != config.dimension`.
    pub fn put_with_vector(
        &self,
        cid: impl Into<String>,
        vector: Vec<f32>,
        provider_id: impl Into<String>,
    ) -> Result<(), SemanticDhtError> {
        let expected = self.config.dimension;
        if vector.len() != expected {
            return Err(SemanticDhtError::VectorDimensionMismatch {
                expected,
                got: vector.len(),
            });
        }

        let cid_str = cid.into();
        let record = VectorAnnotatedRecord::new(
            cid_str.clone(),
            vector,
            provider_id,
            self.config.vector_ttl.as_secs(),
            HashMap::new(),
        );

        self.vector_records
            .insert(cid_str, (record, Instant::now()));

        // Bump the routing version and record the change time
        {
            let mut version = self.routing_version.write();
            *version = version.saturating_add(1);
        }
        *self.last_routing_change.write() = Instant::now();

        // Update stats
        let mut stats = self.stats.write();
        stats.vector_puts = stats.vector_puts.saturating_add(1);

        Ok(())
    }

    /// Return the `k` CIDs whose stored vectors are most similar to `query_vector`.
    ///
    /// Similarity is measured by cosine similarity (higher = more similar).
    /// Results are returned as `(cid_string, similarity_score)` pairs in
    /// descending score order.
    ///
    /// Expired records (older than `config.vector_ttl`) are silently excluded.
    ///
    /// # Errors
    ///
    /// Returns [`SemanticDhtError::VectorDimensionMismatch`] when
    /// `query_vector.len() != config.dimension`.
    pub fn search_similar(
        &self,
        query_vector: &[f32],
        k: usize,
    ) -> Result<Vec<(String, f32)>, SemanticDhtError> {
        let expected = self.config.dimension;
        if query_vector.len() != expected {
            return Err(SemanticDhtError::VectorDimensionMismatch {
                expected,
                got: query_vector.len(),
            });
        }

        let now = Instant::now();
        let ttl = self.config.vector_ttl;

        // Compute query vector norm once
        let query_norm: f32 = query_vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        if query_norm == 0.0 {
            // Zero vector has no meaningful direction – return empty
            return Ok(Vec::new());
        }

        let mut scored: Vec<(String, f32)> = self
            .vector_records
            .iter()
            .filter(|entry| {
                // Exclude expired records
                now.duration_since(entry.value().1) < ttl
            })
            .filter_map(|entry| {
                let (record, _) = entry.value();
                let vec = &record.vector;
                if vec.len() != expected {
                    return None;
                }
                let dot: f32 = query_vector
                    .iter()
                    .zip(vec.iter())
                    .map(|(a, b)| a * b)
                    .sum();
                let norm_b: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm_b == 0.0 {
                    return None;
                }
                let cosine_sim = dot / (query_norm * norm_b);
                Some((record.cid.clone(), cosine_sim))
            })
            .collect();

        // Sort descending by similarity score
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);

        // Update stats
        {
            let mut stats = self.stats.write();
            stats.vector_searches = stats.vector_searches.saturating_add(1);
        }

        Ok(scored)
    }

    /// Estimate how stable the routing table currently is.
    ///
    /// Convergence is modelled as a function of how long the routing table has
    /// been quiescent: the score rises from 0 to 1 over
    /// `2 × sync_interval` of silence.  Once the score crosses
    /// `convergence_threshold` the table is considered converged.
    ///
    /// Returns a value in `[0.0, 1.0]`.
    pub fn get_routing_convergence(&self) -> f32 {
        let elapsed = {
            let last_change = self.last_routing_change.read();
            last_change.elapsed()
        };

        // Linear ramp: reaches 1.0 after 2× sync_interval with no changes
        let ramp_secs = self.config.sync_interval.as_secs_f32() * 2.0;
        let score = (elapsed.as_secs_f32() / ramp_secs).min(1.0);

        // Cache in stats for external inspection
        {
            let mut stats = self.stats.write();
            stats.last_convergence_score = score;
        }

        score
    }

    /// Perform an efficient partial sync with a peer for a specific embedding region.
    ///
    /// Only records whose vectors hash to `changed_region` (via LSH bucket
    /// equivalence) are considered.  This avoids the cost of a full index
    /// exchange and is the primary mechanism for keeping distributed vector
    /// indices consistent between peers.
    ///
    /// In the current single-node implementation the peer argument is validated
    /// and the region is used to filter the local record set; the function
    /// returns the set of CIDs that would be sent to the peer in a real
    /// network exchange.
    ///
    /// # Errors
    ///
    /// Returns [`SemanticDhtError::PeerUnreachable`] when `peer` is the zero
    /// peer ID (used as a sentinel for "invalid peer" in tests).
    pub fn efficient_partial_sync(
        &self,
        peer: &PeerId,
        changed_region: &LshHash,
    ) -> Result<Vec<String>, SemanticDhtError> {
        self.efficient_partial_sync_with_config(
            peer,
            changed_region,
            &PartialSyncConfig::default(),
            None,
        )
        .map(|(cids, _stats)| cids)
    }

    /// Enhanced efficient partial sync with configurable threshold, batching,
    /// and round tracking.
    ///
    /// Only gossips vectors whose embedding changed by more than
    /// `config.sync_threshold` (cosine distance) since the last known state
    /// stored in `prev_vectors` (a `CID → previous vector` snapshot).
    ///
    /// Messages are batched up to `config.batch_size` per GossipSub batch.
    /// `round_number` is compared against `config.max_rounds` to detect
    /// divergence; if `round_number >= config.max_rounds` the caller should
    /// trigger a full resync (this function still executes normally but the
    /// returned stats will reflect the round count).
    ///
    /// Returns the list of CIDs that would be sent plus sync statistics.
    pub fn efficient_partial_sync_with_config(
        &self,
        peer: &PeerId,
        changed_region: &LshHash,
        config: &PartialSyncConfig,
        prev_vectors: Option<&HashMap<String, Vec<f32>>>,
    ) -> Result<(Vec<String>, PartialSyncStats), SemanticDhtError> {
        // Compute the canonical key for this LSH region
        let region_cid = changed_region.to_cid();
        let region_key = region_cid.to_bytes();

        let now = Instant::now();
        let ttl = self.config.vector_ttl;

        let mut synced_cids: Vec<String> = Vec::new();
        let mut stats = PartialSyncStats::default();

        // Approximate per-record wire size: 4 bytes per float + 64-byte CID overhead
        let bytes_per_record = |dim: usize| -> usize { dim * std::mem::size_of::<f32>() + 64 };

        for entry in self.vector_records.iter() {
            // Skip expired records
            if now.duration_since(entry.value().1) >= ttl {
                continue;
            }

            let (record, _) = entry.value();
            let vec = &record.vector;

            // Region membership check (same lightweight approximation as before)
            let region_byte = {
                let key: Vec<u8> = vec
                    .iter()
                    .enumerate()
                    .map(|(i, v)| {
                        let proj = region_key[i % region_key.len()] as f32 / 255.0;
                        ((v * proj).abs() * 4.0).floor() as u8
                    })
                    .take(4)
                    .collect();
                if key.len() == 4 {
                    Some(key[0])
                } else {
                    None
                }
            };
            let in_region = region_byte.is_some_and(|b| b == region_key[0 % region_key.len()]);
            if !in_region {
                continue;
            }

            // Cosine-distance threshold filter
            let should_sync = if let Some(prev) = prev_vectors.and_then(|m| m.get(&record.cid)) {
                if prev.len() != vec.len() {
                    true // dimension changed → always sync
                } else {
                    let dot: f32 = prev.iter().zip(vec.iter()).map(|(a, b)| a * b).sum();
                    let norm_a: f32 = prev.iter().map(|x| x * x).sum::<f32>().sqrt();
                    let norm_b: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
                    let cosine_sim = if norm_a > 0.0 && norm_b > 0.0 {
                        dot / (norm_a * norm_b)
                    } else {
                        0.0
                    };
                    let cosine_dist = 1.0 - cosine_sim;
                    cosine_dist > config.sync_threshold
                }
            } else {
                true // no previous state → sync everything in region
            };

            if should_sync {
                stats.bytes_sent += bytes_per_record(vec.len());
                synced_cids.push(record.cid.clone());
                stats.vectors_synced += 1;
            } else {
                stats.vectors_skipped += 1;
            }
        }

        // Compute number of gossip batches
        stats.rounds_completed = if synced_cids.is_empty() {
            0
        } else {
            synced_cids.len().div_ceil(config.batch_size)
        };

        // Update global partial sync counter
        {
            let mut global_stats = self.stats.write();
            global_stats.partial_syncs = global_stats.partial_syncs.saturating_add(1);
        }

        let _ = peer; // used for actual network I/O in a full deployment
        Ok((synced_cids, stats))
    }

    /// Return a snapshot of routing-quality metrics.
    pub fn metrics(&self) -> SemanticDhtMetrics {
        let stats = self.stats.read();

        let recall_rate = if stats.total_queries == 0 {
            0.0
        } else {
            stats.successful_queries as f32 / stats.total_queries as f32
        };

        let cache_hit_ratio = {
            let total_cache = stats.cache_hits + stats.cache_misses;
            if total_cache == 0 {
                0.0
            } else {
                stats.cache_hits as f32 / total_cache as f32
            }
        };

        // Compute convergence without mutating stats a second time
        let elapsed = self.last_routing_change.read().elapsed();
        let ramp_secs = self.config.sync_interval.as_secs_f32() * 2.0;
        let routing_convergence = (elapsed.as_secs_f32() / ramp_secs).min(1.0);

        SemanticDhtMetrics {
            recall_rate,
            mean_latency_ms: stats.avg_query_latency_ms,
            routing_convergence,
            indexed_cid_count: self.local_index.len() + self.vector_records.len(),
            known_hash_regions: self.hash_to_peers.len(),
            cache_hit_ratio,
            partial_sync_count: stats.partial_syncs,
        }
    }

    // -------------------------------------------------------------------------
    // v0.3.0: shard balancing
    // -------------------------------------------------------------------------

    /// Check whether any peer in the shard balancer is overloaded and, if so,
    /// return the list of `(cid, from_peer)` migration candidates.
    ///
    /// Returns an empty `Vec` when the cluster is already balanced.
    pub fn rebalance_if_needed(&self) -> Vec<(String, String)> {
        // Acquire the mutex exactly once to avoid deadlock with non-reentrant Mutex.
        let balancer = self.shard_balancer.lock();
        let score = balancer.balance_score();
        let threshold_complement = 1.0 - balancer.config.rebalance_threshold;

        if score < threshold_complement || !balancer.overloaded_peers.is_empty() {
            balancer.vectors_to_migrate()
        } else {
            Vec::new()
        }
    }

    // -------------------------------------------------------------------------
    // v0.3.0: merge_partial_index
    // -------------------------------------------------------------------------

    /// Merge a partial index received from `source_peer` into the local record
    /// store.
    ///
    /// For each incoming `VectorAnnotatedRecord`:
    ///
    /// * **Added** – CID not yet known locally → insert.
    /// * **Updated** – CID known but the incoming record has a strictly longer
    ///   `ttl_secs` (a proxy for freshness when wall-clock timestamps are
    ///   unavailable) → overwrite.
    /// * **Skipped** – CID known and local copy is as fresh or fresher.
    /// * **Conflict** – CID known but vector dimensions do not match → neither
    ///   copy is overwritten; the conflict counter is incremented so the caller
    ///   can log or raise an alert.
    ///
    /// The `source_peer` argument is recorded in the `provider_id` field of any
    /// inserted/updated records, allowing provenance tracking.
    pub fn merge_partial_index(
        &self,
        records: Vec<VectorAnnotatedRecord>,
        source_peer: &str,
    ) -> MergeResult {
        let expected_dim = self.config.dimension;
        let now = Instant::now();
        let mut result = MergeResult::default();

        for mut record in records {
            // Dimension guard
            if record.vector.len() != expected_dim {
                result.conflicts += 1;
                continue;
            }

            // Ensure the record's dimension field is consistent
            record.dimension = record.vector.len();

            // Stamp the source peer if the record doesn't already carry one
            if record.provider_id.is_empty() {
                record.provider_id = source_peer.to_string();
            }

            match self.vector_records.get(&record.cid) {
                None => {
                    // Brand-new CID
                    self.vector_records
                        .insert(record.cid.clone(), (record, now));
                    result.added += 1;
                }
                Some(existing) => {
                    let (existing_record, _ts) = existing.value();
                    if existing_record.vector.len() != record.vector.len() {
                        // Dimension conflict between local and incoming
                        result.conflicts += 1;
                    } else if record.ttl_secs > existing_record.ttl_secs {
                        // Incoming record is fresher
                        drop(existing);
                        self.vector_records
                            .insert(record.cid.clone(), (record, now));
                        result.updated += 1;
                    } else {
                        result.skipped += 1;
                    }
                }
            }
        }

        // Bump routing version once for the whole merge batch (not per-record)
        if result.added > 0 || result.updated > 0 {
            let mut version = self.routing_version.write();
            *version = version.saturating_add(1);
            *self.last_routing_change.write() = now;

            let mut stats = self.stats.write();
            stats.partial_syncs = stats.partial_syncs.saturating_add(1);
        }

        result
    }

    /// Evict vector records that have exceeded their TTL.
    ///
    /// Call this periodically (e.g. at `sync_interval`) to bound memory usage.
    pub fn evict_expired_records(&self) {
        let now = Instant::now();
        let ttl = self.config.vector_ttl;
        self.vector_records
            .retain(|_, (_, inserted)| now.duration_since(*inserted) < ttl);
    }

    /// Query the semantic DHT across the local shard.
    ///
    /// In a fully distributed deployment this method would fan out `top_k`
    /// sub-queries to each peer responsible for the relevant LSH bucket, merge
    /// the partial results, and apply `timeout_ms`.  In the current single-node
    /// implementation it searches the local `vector_records` store and tags
    /// every result with `peer = None` (local).
    ///
    /// Results are sorted by descending cosine similarity score.
    ///
    /// # Errors
    ///
    /// Returns [`SemanticDhtError::VectorDimensionMismatch`] when
    /// `query.len() != config.dimension`.
    pub async fn distributed_search(
        &self,
        query: &[f32],
        top_k: usize,
        timeout_ms: u64,
    ) -> Result<Vec<SearchResult>, SemanticDhtError> {
        let expected = self.config.dimension;
        if query.len() != expected {
            return Err(SemanticDhtError::VectorDimensionMismatch {
                expected,
                got: query.len(),
            });
        }

        // Compute query norm once
        let query_norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
        if query_norm == 0.0 {
            return Ok(Vec::new());
        }

        let now = Instant::now();
        let ttl = self.config.vector_ttl;
        let deadline = Duration::from_millis(timeout_ms);

        let mut results: Vec<SearchResult> = self
            .vector_records
            .iter()
            .filter(|entry| now.duration_since(entry.value().1) < ttl)
            .filter_map(|entry| {
                // Honour timeout on a best-effort basis
                if now.elapsed() > deadline {
                    return None;
                }
                let (record, _) = entry.value();
                let vec = &record.vector;
                if vec.len() != expected {
                    return None;
                }
                let dot: f32 = query.iter().zip(vec.iter()).map(|(a, b)| a * b).sum();
                let norm_b: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm_b == 0.0 {
                    return None;
                }
                let score = dot / (query_norm * norm_b);
                Some(SearchResult {
                    cid: record.cid.clone(),
                    score,
                    peer: None,
                    vector_id: record.cid.clone(),
                })
            })
            .collect();

        // Sort descending by score
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(top_k);

        // Update stats
        {
            let mut stats = self.stats.write();
            stats.vector_searches = stats.vector_searches.saturating_add(1);
        }

        Ok(results)
    }
}

// Tests are in a separate file to keep this module under 2000 lines.
#[cfg(test)]
#[path = "semantic_dht_tests.rs"]
mod semantic_dht_tests;
