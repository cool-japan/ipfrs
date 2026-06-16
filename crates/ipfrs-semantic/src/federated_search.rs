//! Federated Search Coordinator — Cross-Node Vector Similarity Search
//!
//! This module enables cross-node vector similarity search by fanning out queries to
//! multiple registered peer nodes and merging the ranked results into a single
//! deduplicated, re-ranked response.
//!
//! # Design
//!
//! - Peers are managed through a `RwLock<Vec<SearchPeer>>` for concurrent read access.
//! - Results are cached keyed by `QueryKey` (FNV-1a hash of query vector, top_k, ef).
//! - Merged result lists are deduplicated by CID and re-ranked by similarity score.
//! - Atomic stats counters track query counts, cache hits, peers queried, etc.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// QueryKey
// ---------------------------------------------------------------------------

/// Cache key derived from the query vector (FNV-1a hash), top_k, and ef.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct QueryKey {
    /// FNV-1a hash of the raw bytes of the query vector.
    pub query_hash: u64,
    /// Number of top results requested.
    pub top_k: usize,
    /// ef parameter for HNSW search.
    pub ef: usize,
}

impl QueryKey {
    /// Construct a new `QueryKey` by hashing `query_vec` with FNV-1a.
    pub fn new(query_vec: &[f32], top_k: usize, ef: usize) -> Self {
        let query_hash = fnv1a_hash_f32_slice(query_vec);
        Self {
            query_hash,
            top_k,
            ef,
        }
    }
}

/// FNV-1a hash over the raw bytes of a `&[f32]` slice.
fn fnv1a_hash_f32_slice(values: &[f32]) -> u64 {
    const FNV_OFFSET_BASIS: u64 = 14_695_981_039_346_656_037;
    const FNV_PRIME: u64 = 1_099_511_628_211;

    let mut hash = FNV_OFFSET_BASIS;
    for &v in values {
        for byte in v.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
    }
    hash
}

// ---------------------------------------------------------------------------
// SearchResult
// ---------------------------------------------------------------------------

/// A single result returned from a peer node's vector index.
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// Content identifier of the indexed item.
    pub cid: String,
    /// Similarity score — higher means more similar.
    pub score: f32,
    /// The peer node that produced this result.
    pub peer_id: String,
    /// Arbitrary metadata attached to the result.
    pub metadata: HashMap<String, String>,
}

// ---------------------------------------------------------------------------
// SearchPeer
// ---------------------------------------------------------------------------

/// A registered remote (or local) peer that holds a vector index.
#[derive(Debug, Clone)]
pub struct SearchPeer {
    /// Unique identifier for this peer.
    pub peer_id: String,
    /// Multiaddr string (e.g. `/ip4/127.0.0.1/tcp/4001`).
    pub multiaddr: String,
    /// Estimated round-trip latency to this peer in milliseconds.
    pub estimated_latency_ms: u64,
    /// Number of vectors indexed by this peer.
    pub index_size: u64,
    /// Last time this peer was observed as alive.
    pub last_seen: Instant,
}

impl SearchPeer {
    /// Create a new `SearchPeer` with sensible defaults.
    pub fn new(peer_id: impl Into<String>, multiaddr: impl Into<String>) -> Self {
        Self {
            peer_id: peer_id.into(),
            multiaddr: multiaddr.into(),
            estimated_latency_ms: 50,
            index_size: 0,
            last_seen: Instant::now(),
        }
    }
}

// ---------------------------------------------------------------------------
// CachedSearchResult
// ---------------------------------------------------------------------------

/// A cached response to a query.
#[derive(Debug, Clone)]
pub struct CachedSearchResult {
    /// The cached result list.
    pub results: Vec<SearchResult>,
    /// Wall-clock time when the entry was stored.
    pub cached_at: Instant,
    /// How long the cached entry remains valid.
    pub ttl: Duration,
}

impl CachedSearchResult {
    /// Returns `true` when the entry has outlived its TTL.
    pub fn is_expired(&self, now: Instant) -> bool {
        now.duration_since(self.cached_at) > self.ttl
    }
}

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------

/// Atomic statistics counters for the coordinator.
pub struct FederatedSearchStats {
    /// Total number of searches dispatched (cache misses included).
    pub total_queries: AtomicU64,
    /// Number of queries that were served from cache.
    pub total_cache_hits: AtomicU64,
    /// Cumulative number of peer node queries issued.
    pub total_peers_queried: AtomicU64,
    /// Total number of `SearchResult` entries merged across all queries.
    pub total_results_merged: AtomicU64,
    /// Total number of duplicate entries removed during merges.
    pub total_deduped: AtomicU64,
}

impl FederatedSearchStats {
    fn new() -> Self {
        Self {
            total_queries: AtomicU64::new(0),
            total_cache_hits: AtomicU64::new(0),
            total_peers_queried: AtomicU64::new(0),
            total_results_merged: AtomicU64::new(0),
            total_deduped: AtomicU64::new(0),
        }
    }

    /// Take a consistent snapshot of the current counters.
    pub fn snapshot(&self) -> FederatedSearchStatsSnapshot {
        FederatedSearchStatsSnapshot {
            total_queries: self.total_queries.load(Ordering::Relaxed),
            total_cache_hits: self.total_cache_hits.load(Ordering::Relaxed),
            total_peers_queried: self.total_peers_queried.load(Ordering::Relaxed),
            total_results_merged: self.total_results_merged.load(Ordering::Relaxed),
            total_deduped: self.total_deduped.load(Ordering::Relaxed),
        }
    }
}

impl std::fmt::Debug for FederatedSearchStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FederatedSearchStats")
            .field("total_queries", &self.total_queries.load(Ordering::Relaxed))
            .field(
                "total_cache_hits",
                &self.total_cache_hits.load(Ordering::Relaxed),
            )
            .field(
                "total_peers_queried",
                &self.total_peers_queried.load(Ordering::Relaxed),
            )
            .field(
                "total_results_merged",
                &self.total_results_merged.load(Ordering::Relaxed),
            )
            .field("total_deduped", &self.total_deduped.load(Ordering::Relaxed))
            .finish()
    }
}

/// Non-atomic copy of the coordinator stats for external consumption.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FederatedSearchStatsSnapshot {
    pub total_queries: u64,
    pub total_cache_hits: u64,
    pub total_peers_queried: u64,
    pub total_results_merged: u64,
    pub total_deduped: u64,
}

// ---------------------------------------------------------------------------
// FederatedSearchCoordinator
// ---------------------------------------------------------------------------

/// Coordinates cross-node vector similarity searches by fanning out queries to
/// multiple registered [`SearchPeer`]s and merging the returned ranked lists.
pub struct FederatedSearchCoordinator {
    /// All known peer nodes.
    known_peers: RwLock<Vec<SearchPeer>>,
    /// LRU-style result cache keyed by [`QueryKey`].
    result_cache: Mutex<HashMap<QueryKey, CachedSearchResult>>,
    /// How long a cached entry remains valid.
    pub cache_ttl: Duration,
    /// Upper bound on the number of peers queried per search.
    pub max_peers_per_query: usize,
    /// Atomic operational statistics.
    stats: Arc<FederatedSearchStats>,
}

impl FederatedSearchCoordinator {
    /// Construct a new coordinator with default settings.
    ///
    /// - `cache_ttl` = 30 seconds
    /// - `max_peers_per_query` = 8
    pub fn new() -> Self {
        Self {
            known_peers: RwLock::new(Vec::new()),
            result_cache: Mutex::new(HashMap::new()),
            cache_ttl: Duration::from_secs(30),
            max_peers_per_query: 8,
            stats: Arc::new(FederatedSearchStats::new()),
        }
    }

    /// Create a coordinator with custom TTL and max-peers settings.
    pub fn with_config(cache_ttl: Duration, max_peers_per_query: usize) -> Self {
        Self {
            known_peers: RwLock::new(Vec::new()),
            result_cache: Mutex::new(HashMap::new()),
            cache_ttl,
            max_peers_per_query,
            stats: Arc::new(FederatedSearchStats::new()),
        }
    }

    // ------------------------------------------------------------------
    // Peer management
    // ------------------------------------------------------------------

    /// Register a peer node so it is eligible to receive fan-out queries.
    pub fn register_peer(&self, peer: SearchPeer) {
        let mut peers = match self.known_peers.write() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        // Replace an existing entry with the same ID (upsert semantics).
        if let Some(existing) = peers.iter_mut().find(|p| p.peer_id == peer.peer_id) {
            *existing = peer;
        } else {
            peers.push(peer);
        }
    }

    /// Remove a peer by its ID. Does nothing if the peer is not registered.
    pub fn unregister_peer(&self, peer_id: &str) {
        let mut peers = match self.known_peers.write() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        peers.retain(|p| p.peer_id != peer_id);
    }

    /// Return the number of currently registered peers.
    pub fn peer_count(&self) -> usize {
        let peers = match self.known_peers.read() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        peers.len()
    }

    // ------------------------------------------------------------------
    // Peer selection
    // ------------------------------------------------------------------

    /// Select up to `max_peers_per_query` peers for a query, sorted by
    /// ascending `estimated_latency_ms` (lowest latency first).
    ///
    /// `top_k` is accepted for future routing heuristics but is not currently
    /// used to filter peers beyond the `max_peers_per_query` cap.
    pub fn select_peers_for_query(&self, _top_k: usize) -> Vec<SearchPeer> {
        let peers = match self.known_peers.read() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };

        let mut selected: Vec<SearchPeer> = peers.clone();
        selected.sort_by_key(|p| p.estimated_latency_ms);
        selected.truncate(self.max_peers_per_query);
        selected
    }

    // ------------------------------------------------------------------
    // Result merging
    // ------------------------------------------------------------------

    /// Merge N per-peer result lists into a single deduplicated, sorted list.
    ///
    /// Deduplication is by CID; when duplicates exist, the entry with the
    /// **highest** score is kept.  The final list is sorted by score descending
    /// and truncated to `top_k`.
    pub fn merge_results(
        &self,
        results: Vec<Vec<SearchResult>>,
        top_k: usize,
    ) -> Vec<SearchResult> {
        // Flatten into a single map keyed by CID, keeping the best score.
        let mut best: HashMap<String, SearchResult> = HashMap::new();
        let mut total_before_dedup: u64 = 0;

        for list in results {
            for item in list {
                total_before_dedup += 1;
                match best.get(&item.cid) {
                    Some(existing) if existing.score >= item.score => {
                        // Keep the already-stored higher-score entry.
                    }
                    _ => {
                        best.insert(item.cid.clone(), item);
                    }
                }
            }
        }

        let total_after_dedup = best.len() as u64;
        let deduped = total_before_dedup.saturating_sub(total_after_dedup);

        self.stats
            .total_results_merged
            .fetch_add(total_before_dedup, Ordering::Relaxed);
        self.stats
            .total_deduped
            .fetch_add(deduped, Ordering::Relaxed);

        // Sort by score descending, then truncate to top_k.
        let mut merged: Vec<SearchResult> = best.into_values().collect();
        merged.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        merged.truncate(top_k);
        merged
    }

    // ------------------------------------------------------------------
    // Cache operations
    // ------------------------------------------------------------------

    /// Store a result list in the cache under `key`.
    pub fn cache_result(&self, key: QueryKey, results: Vec<SearchResult>) {
        let entry = CachedSearchResult {
            results,
            cached_at: Instant::now(),
            ttl: self.cache_ttl,
        };
        let mut cache = match self.result_cache.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        cache.insert(key, entry);
    }

    /// Retrieve a cached result list, returning `None` if the key is absent
    /// or the entry has expired.
    pub fn get_cached(&self, key: &QueryKey) -> Option<Vec<SearchResult>> {
        let now = Instant::now();
        let cache = match self.result_cache.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        cache
            .get(key)
            .filter(|entry| !entry.is_expired(now))
            .map(|entry| {
                self.stats.total_cache_hits.fetch_add(1, Ordering::Relaxed);
                entry.results.clone()
            })
    }

    /// Remove all expired entries from the result cache.
    pub fn evict_expired_cache(&self) {
        let now = Instant::now();
        let mut cache = match self.result_cache.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        cache.retain(|_key, entry| !entry.is_expired(now));
    }

    // ------------------------------------------------------------------
    // High-level fan-out search (synchronous stub for library usage)
    // ------------------------------------------------------------------

    /// Fan out a query, merging results from selected peers.
    ///
    /// This method drives the coordinator's own bookkeeping (query counter,
    /// cache checks, peer-query counter).  Actual network transport is the
    /// caller's responsibility — `fetch_fn` is invoked once per selected peer
    /// and must return the peer's result list.
    ///
    /// ```rust
    /// # use ipfrs_semantic::federated_search::{FederatedSearchCoordinator, SearchPeer, SearchResult, QueryKey};
    /// # use std::collections::HashMap;
    /// # let coord = FederatedSearchCoordinator::new();
    /// # let peer = SearchPeer::new("p1", "/ip4/127.0.0.1/tcp/4001");
    /// # coord.register_peer(peer);
    /// let query = vec![0.1_f32; 8];
    /// let top_k = 5;
    /// let ef    = 64;
    /// let key   = QueryKey::new(&query, top_k, ef);
    ///
    /// let results = coord.search(
    ///     &query, top_k, ef,
    ///     |_peer| Vec::<SearchResult>::new(),
    /// );
    /// assert!(results.len() <= top_k);
    /// ```
    pub fn search<F>(
        &self,
        query: &[f32],
        top_k: usize,
        ef: usize,
        fetch_fn: F,
    ) -> Vec<SearchResult>
    where
        F: Fn(&SearchPeer) -> Vec<SearchResult>,
    {
        self.stats.total_queries.fetch_add(1, Ordering::Relaxed);

        let key = QueryKey::new(query, top_k, ef);

        // Cache check.
        if let Some(cached) = self.get_cached(&key) {
            return cached;
        }

        let peers = self.select_peers_for_query(top_k);
        self.stats
            .total_peers_queried
            .fetch_add(peers.len() as u64, Ordering::Relaxed);

        let per_peer_results: Vec<Vec<SearchResult>> = peers.iter().map(fetch_fn).collect();

        let merged = self.merge_results(per_peer_results, top_k);
        self.cache_result(key, merged.clone());
        merged
    }

    // ------------------------------------------------------------------
    // Stats
    // ------------------------------------------------------------------

    /// Return an immutable snapshot of the coordinator's statistics.
    pub fn stats(&self) -> FederatedSearchStatsSnapshot {
        self.stats.snapshot()
    }
}

impl Default for FederatedSearchCoordinator {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for FederatedSearchCoordinator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let peer_count = self.peer_count();
        let cache_len = self.result_cache.lock().map(|g| g.len()).unwrap_or(0);
        f.debug_struct("FederatedSearchCoordinator")
            .field("peer_count", &peer_count)
            .field("cache_entries", &cache_len)
            .field("cache_ttl", &self.cache_ttl)
            .field("max_peers_per_query", &self.max_peers_per_query)
            .field("stats", &self.stats)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    fn make_peer(id: &str, latency: u64) -> SearchPeer {
        let mut p = SearchPeer::new(id, format!("/ip4/127.0.0.1/tcp/{}", 4000 + latency));
        p.estimated_latency_ms = latency;
        p
    }

    fn make_result(cid: &str, score: f32, peer_id: &str) -> SearchResult {
        SearchResult {
            cid: cid.to_string(),
            score,
            peer_id: peer_id.to_string(),
            metadata: HashMap::new(),
        }
    }

    // ------------------------------------------------------------------
    // 1. Peer registration
    // ------------------------------------------------------------------

    #[test]
    fn test_register_peer_increases_count() {
        let coord = FederatedSearchCoordinator::new();
        assert_eq!(coord.peer_count(), 0);
        coord.register_peer(make_peer("p1", 10));
        assert_eq!(coord.peer_count(), 1);
        coord.register_peer(make_peer("p2", 20));
        assert_eq!(coord.peer_count(), 2);
    }

    // ------------------------------------------------------------------
    // 2. Peer unregistration
    // ------------------------------------------------------------------

    #[test]
    fn test_unregister_peer_decreases_count() {
        let coord = FederatedSearchCoordinator::new();
        coord.register_peer(make_peer("p1", 10));
        coord.register_peer(make_peer("p2", 20));
        coord.unregister_peer("p1");
        assert_eq!(coord.peer_count(), 1);
    }

    #[test]
    fn test_unregister_nonexistent_peer_noop() {
        let coord = FederatedSearchCoordinator::new();
        coord.register_peer(make_peer("p1", 10));
        coord.unregister_peer("does_not_exist");
        assert_eq!(coord.peer_count(), 1);
    }

    // ------------------------------------------------------------------
    // 3. peer_count reflects register/unregister
    // ------------------------------------------------------------------

    #[test]
    fn test_peer_count_reflects_all_operations() {
        let coord = FederatedSearchCoordinator::new();
        for i in 0..5u64 {
            coord.register_peer(make_peer(&format!("p{}", i), i * 5));
        }
        assert_eq!(coord.peer_count(), 5);
        coord.unregister_peer("p2");
        coord.unregister_peer("p4");
        assert_eq!(coord.peer_count(), 3);
    }

    // ------------------------------------------------------------------
    // 4. select_peers_for_query respects max_peers_per_query
    // ------------------------------------------------------------------

    #[test]
    fn test_select_peers_respects_max() {
        let coord = FederatedSearchCoordinator::with_config(Duration::from_secs(30), 3);
        for i in 0..10u64 {
            coord.register_peer(make_peer(&format!("p{}", i), i * 5));
        }
        let selected = coord.select_peers_for_query(10);
        assert_eq!(selected.len(), 3);
    }

    // ------------------------------------------------------------------
    // 5. select_peers_for_query sorts by latency ascending
    // ------------------------------------------------------------------

    #[test]
    fn test_select_peers_sorted_by_latency() {
        let coord = FederatedSearchCoordinator::with_config(Duration::from_secs(30), 5);
        // Register in reverse latency order.
        coord.register_peer(make_peer("slow", 200));
        coord.register_peer(make_peer("medium", 100));
        coord.register_peer(make_peer("fast", 10));

        let selected = coord.select_peers_for_query(5);
        assert_eq!(selected[0].peer_id, "fast");
        assert_eq!(selected[1].peer_id, "medium");
        assert_eq!(selected[2].peer_id, "slow");
    }

    // ------------------------------------------------------------------
    // 6. merge_results deduplicates by CID
    // ------------------------------------------------------------------

    #[test]
    fn test_merge_results_deduplicates_by_cid() {
        let coord = FederatedSearchCoordinator::new();
        let list_a = vec![
            make_result("cid1", 0.9, "p1"),
            make_result("cid2", 0.7, "p1"),
        ];
        let list_b = vec![
            make_result("cid1", 0.8, "p2"), // duplicate cid1 with lower score
            make_result("cid3", 0.6, "p2"),
        ];
        let merged = coord.merge_results(vec![list_a, list_b], 10);
        let cid1_results: Vec<_> = merged.iter().filter(|r| r.cid == "cid1").collect();
        // Exactly one entry for cid1.
        assert_eq!(cid1_results.len(), 1);
        // The higher-score entry is kept.
        assert!((cid1_results[0].score - 0.9).abs() < f32::EPSILON);
    }

    // ------------------------------------------------------------------
    // 7. merge_results returns top_k sorted by score descending
    // ------------------------------------------------------------------

    #[test]
    fn test_merge_results_top_k_sorted_descending() {
        let coord = FederatedSearchCoordinator::new();
        let list = vec![
            make_result("c1", 0.3, "p1"),
            make_result("c2", 0.9, "p1"),
            make_result("c3", 0.6, "p1"),
            make_result("c4", 0.1, "p1"),
            make_result("c5", 0.8, "p1"),
        ];
        let merged = coord.merge_results(vec![list], 3);
        assert_eq!(merged.len(), 3);
        assert!(merged[0].score >= merged[1].score);
        assert!(merged[1].score >= merged[2].score);
        // Top result must be c2 (0.9).
        assert_eq!(merged[0].cid, "c2");
    }

    // ------------------------------------------------------------------
    // 8. merge_results with empty input list
    // ------------------------------------------------------------------

    #[test]
    fn test_merge_results_empty_input() {
        let coord = FederatedSearchCoordinator::new();
        let merged = coord.merge_results(vec![], 10);
        assert!(merged.is_empty());
    }

    // ------------------------------------------------------------------
    // 9. merge_results with all empty per-peer lists
    // ------------------------------------------------------------------

    #[test]
    fn test_merge_results_all_empty_lists() {
        let coord = FederatedSearchCoordinator::new();
        let merged = coord.merge_results(vec![vec![], vec![], vec![]], 10);
        assert!(merged.is_empty());
    }

    // ------------------------------------------------------------------
    // 10. QueryKey hash stability (same input → same hash)
    // ------------------------------------------------------------------

    #[test]
    fn test_query_key_hash_stability() {
        let vec_a = vec![0.1_f32, 0.2, 0.3, 0.4];
        let key1 = QueryKey::new(&vec_a, 5, 64);
        let key2 = QueryKey::new(&vec_a, 5, 64);
        assert_eq!(key1, key2);
        assert_eq!(key1.query_hash, key2.query_hash);
    }

    #[test]
    fn test_query_key_different_inputs_differ() {
        let vec_a = vec![0.1_f32, 0.2, 0.3];
        let vec_b = vec![0.9_f32, 0.8, 0.7];
        let key_a = QueryKey::new(&vec_a, 5, 64);
        let key_b = QueryKey::new(&vec_b, 5, 64);
        assert_ne!(key_a.query_hash, key_b.query_hash);
    }

    // ------------------------------------------------------------------
    // 11. Cache hit on repeated query
    // ------------------------------------------------------------------

    #[test]
    fn test_cache_hit_on_repeated_query() {
        let coord = FederatedSearchCoordinator::new();
        let results = vec![make_result("cid1", 0.95, "p1")];
        let key = QueryKey::new(&[0.5_f32; 4], 5, 64);

        assert!(coord.get_cached(&key).is_none());
        coord.cache_result(key.clone(), results);
        let cached = coord.get_cached(&key);
        assert!(cached.is_some());
        let list = cached.expect("cache entry must exist");
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].cid, "cid1");
    }

    // ------------------------------------------------------------------
    // 12. Cache eviction on TTL expiry
    // ------------------------------------------------------------------

    #[test]
    fn test_cache_eviction_on_ttl_expiry() {
        // TTL of 0 — effectively expired immediately.
        let coord = FederatedSearchCoordinator::with_config(Duration::from_nanos(1), 8);
        let key = QueryKey::new(&[0.1_f32; 4], 5, 64);
        coord.cache_result(key.clone(), vec![make_result("cid_expired", 0.5, "p1")]);

        // Spin until at least 1ns has passed.
        let deadline = std::time::Instant::now() + Duration::from_millis(10);
        while std::time::Instant::now() < deadline {}

        // A get_cached on an expired entry should return None.
        assert!(coord.get_cached(&key).is_none());
    }

    #[test]
    fn test_evict_expired_cache_removes_stale_entries() {
        let coord = FederatedSearchCoordinator::with_config(Duration::from_nanos(1), 8);
        let key1 = QueryKey::new(&[0.1_f32; 4], 5, 64);
        let key2 = QueryKey::new(&[0.2_f32; 4], 5, 64);
        coord.cache_result(key1.clone(), vec![make_result("c1", 0.5, "p1")]);
        coord.cache_result(key2.clone(), vec![make_result("c2", 0.6, "p1")]);

        // Wait for TTL to elapse.
        let deadline = std::time::Instant::now() + Duration::from_millis(10);
        while std::time::Instant::now() < deadline {}

        coord.evict_expired_cache();

        let cache_len = coord
            .result_cache
            .lock()
            .map(|g| g.len())
            .unwrap_or(usize::MAX);
        assert_eq!(cache_len, 0);
    }

    // ------------------------------------------------------------------
    // 13. Stats accumulation via search()
    // ------------------------------------------------------------------

    #[test]
    fn test_stats_accumulation() {
        let coord = FederatedSearchCoordinator::new();
        coord.register_peer(make_peer("p1", 10));
        coord.register_peer(make_peer("p2", 20));

        let query = vec![0.5_f32; 4];

        // First search — cache miss, both peers queried.
        coord.search(&query, 3, 64, |_peer| {
            vec![make_result("c1", 0.9, "p1"), make_result("c2", 0.5, "p2")]
        });

        // Second search — same query key → cache hit.
        coord.search(&query, 3, 64, |_peer| vec![]);

        let snap = coord.stats();
        assert_eq!(snap.total_queries, 2);
        assert_eq!(snap.total_cache_hits, 1);
        // First search queried 2 peers.
        assert_eq!(snap.total_peers_queried, 2);
        // First search merged 4 results (2 peers × 2 results).
        assert_eq!(snap.total_results_merged, 4);
    }

    // ------------------------------------------------------------------
    // 14. Upsert semantics on register_peer
    // ------------------------------------------------------------------

    #[test]
    fn test_register_peer_upsert_does_not_duplicate() {
        let coord = FederatedSearchCoordinator::new();
        coord.register_peer(make_peer("p1", 50));
        // Re-register the same peer with updated latency.
        let mut updated = make_peer("p1", 10);
        updated.index_size = 9999;
        coord.register_peer(updated);
        // Still only one peer.
        assert_eq!(coord.peer_count(), 1);
        // And the updated peer is stored.
        let selected = coord.select_peers_for_query(1);
        assert_eq!(selected[0].estimated_latency_ms, 10);
        assert_eq!(selected[0].index_size, 9999);
    }

    // ------------------------------------------------------------------
    // 15. select_peers_for_query returns fewer than max when not enough peers
    // ------------------------------------------------------------------

    #[test]
    fn test_select_peers_fewer_than_max() {
        let coord = FederatedSearchCoordinator::with_config(Duration::from_secs(30), 8);
        coord.register_peer(make_peer("only_peer", 15));
        let selected = coord.select_peers_for_query(10);
        assert_eq!(selected.len(), 1);
    }

    // ------------------------------------------------------------------
    // 16. CachedSearchResult::is_expired logic
    // ------------------------------------------------------------------

    #[test]
    fn test_cached_result_is_expired() {
        let entry = CachedSearchResult {
            results: vec![],
            cached_at: Instant::now() - Duration::from_secs(60),
            ttl: Duration::from_secs(30),
        };
        assert!(entry.is_expired(Instant::now()));

        let fresh = CachedSearchResult {
            results: vec![],
            cached_at: Instant::now(),
            ttl: Duration::from_secs(30),
        };
        assert!(!fresh.is_expired(Instant::now()));
    }
}
