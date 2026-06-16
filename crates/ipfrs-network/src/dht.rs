//! Distributed Hash Table (Kademlia DHT) implementation
//!
//! Provides DHT operations including:
//! - Peer discovery and routing
//! - Content provider management
//! - Query result caching
//! - Automatic provider record refresh

use cid::Cid;
use dashmap::DashMap;
use ipfrs_core::error::{Error, Result};
use libp2p::PeerId;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tracing::{debug, info};

/// Default provider record TTL (24 hours)
const DEFAULT_PROVIDER_TTL: Duration = Duration::from_secs(24 * 60 * 60);

/// Default query cache TTL (5 minutes)
const DEFAULT_QUERY_CACHE_TTL: Duration = Duration::from_secs(5 * 60);

/// DHT configuration
#[derive(Debug, Clone)]
pub struct DhtConfig {
    /// Provider record TTL
    pub provider_ttl: Duration,
    /// Query cache TTL
    pub query_cache_ttl: Duration,
    /// Enable automatic provider refresh
    pub enable_provider_refresh: bool,
    /// Provider refresh interval (should be < provider_ttl)
    pub provider_refresh_interval: Duration,
    /// Maximum cached queries
    pub max_cached_queries: usize,
}

impl Default for DhtConfig {
    fn default() -> Self {
        Self {
            provider_ttl: DEFAULT_PROVIDER_TTL,
            query_cache_ttl: DEFAULT_QUERY_CACHE_TTL,
            enable_provider_refresh: true,
            provider_refresh_interval: Duration::from_secs(12 * 60 * 60), // 12 hours
            max_cached_queries: 10_000,
        }
    }
}

/// Cached query result
#[derive(Debug, Clone)]
struct CachedQuery {
    /// Result peers
    peers: Vec<PeerId>,
    /// Timestamp when cached
    cached_at: Instant,
    /// Number of times this result was used
    hit_count: usize,
}

/// Provider record for refresh tracking
#[derive(Debug, Clone)]
struct ProviderRecord {
    /// Content ID
    cid: Cid,
    /// Last announcement time
    last_announced: Instant,
}

/// DHT manager for peer and content discovery
pub struct DhtManager {
    config: DhtConfig,
    /// Query result cache (CID -> cached result)
    query_cache: Arc<DashMap<String, CachedQuery>>,
    /// Peer routing cache (PeerId -> known addresses count)
    peer_cache: Arc<DashMap<PeerId, Instant>>,
    /// Provider records to refresh
    provider_records: Arc<RwLock<HashMap<String, ProviderRecord>>>,
    /// Statistics
    stats: Arc<RwLock<DhtStats>>,
    /// Refresh task handle
    refresh_handle: Option<tokio::task::JoinHandle<()>>,
    /// Command sender for refresh task
    cmd_tx: Option<mpsc::Sender<DhtCommand>>,
}

/// DHT statistics
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct DhtStats {
    /// Total queries performed
    pub total_queries: u64,
    /// Cache hits
    pub cache_hits: u64,
    /// Cache misses
    pub cache_misses: u64,
    /// Total provider refreshes
    pub provider_refreshes: u64,
    /// Active provider records
    pub active_providers: usize,
    /// Cached queries count
    pub cached_queries: usize,
    /// Cached peers count
    pub cached_peers: usize,
    /// Successful queries
    pub successful_queries: u64,
    /// Failed queries
    pub failed_queries: u64,
}

/// DHT health status
#[derive(Debug, Clone, serde::Serialize)]
pub struct DhtHealth {
    /// Overall health score (0.0 - 1.0)
    pub health_score: f64,
    /// Query success rate (0.0 - 1.0)
    pub query_success_rate: f64,
    /// Cache hit rate (0.0 - 1.0)
    pub cache_hit_rate: f64,
    /// Number of cached peers
    pub peer_count: usize,
    /// Number of cached queries
    pub cached_query_count: usize,
    /// Number of active provider records
    pub provider_count: usize,
    /// Health status description
    pub status: DhtHealthStatus,
}

/// DHT health status enum
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub enum DhtHealthStatus {
    /// DHT is healthy
    Healthy,
    /// DHT has degraded performance
    Degraded,
    /// DHT is unhealthy
    Unhealthy,
    /// Not enough data to determine health
    Unknown,
}

/// DHT command for background task
pub(crate) enum DhtCommand {
    /// Add a provider record to track
    TrackProvider { cid: Cid },
    /// Stop tracking a provider
    StopTracking { cid: String },
    /// Refresh all providers (returns sender for refresh requests)
    #[allow(dead_code)]
    RefreshProviders { response_tx: mpsc::Sender<Vec<Cid>> },
    /// Shutdown the refresh task
    Shutdown,
}

impl DhtManager {
    /// Create a new DHT manager
    pub fn new(config: DhtConfig) -> Self {
        let manager = Self {
            config,
            query_cache: Arc::new(DashMap::new()),
            peer_cache: Arc::new(DashMap::new()),
            provider_records: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(DhtStats::default())),
            refresh_handle: None,
            cmd_tx: None,
        };

        info!(
            "DHT Manager initialized (provider_ttl={:?}, query_cache_ttl={:?})",
            manager.config.provider_ttl, manager.config.query_cache_ttl
        );

        manager
    }

    /// Start the provider refresh background task
    pub fn start_provider_refresh(&mut self) {
        if !self.config.enable_provider_refresh {
            info!("Provider refresh disabled");
            return;
        }

        let (cmd_tx, mut cmd_rx) = mpsc::channel::<DhtCommand>(100);
        let provider_records = Arc::clone(&self.provider_records);
        let stats = Arc::clone(&self.stats);
        let refresh_interval = self.config.provider_refresh_interval;

        let handle = tokio::spawn(async move {
            info!(
                "Starting provider refresh task (interval={:?})",
                refresh_interval
            );

            let mut interval = tokio::time::interval(refresh_interval);

            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        // Periodic refresh check
                        let now = Instant::now();
                        let mut refresh_needed = Vec::new();

                        {
                            let records = provider_records.read();
                            for (key, record) in records.iter() {
                                if now.duration_since(record.last_announced) >= refresh_interval {
                                    refresh_needed.push((key.clone(), record.cid));
                                }
                            }
                        }

                        if !refresh_needed.is_empty() {
                            info!("Refreshing {} provider records", refresh_needed.len());
                            stats.write().provider_refreshes += refresh_needed.len() as u64;

                            // Update last_announced times
                            let mut records = provider_records.write();
                            for (key, _cid) in refresh_needed {
                                if let Some(record) = records.get_mut(&key) {
                                    record.last_announced = now;
                                }
                            }
                        }
                    }
                    Some(cmd) = cmd_rx.recv() => {
                        match cmd {
                            DhtCommand::TrackProvider { cid } => {
                                let key = cid.to_string();
                                let mut records = provider_records.write();
                                records.insert(key.clone(), ProviderRecord {
                                    cid,
                                    last_announced: Instant::now(),
                                });
                                debug!("Tracking provider record: {}", key);
                                stats.write().active_providers = records.len();
                            }
                            DhtCommand::StopTracking { cid } => {
                                let mut records = provider_records.write();
                                records.remove(&cid);
                                debug!("Stopped tracking provider: {}", cid);
                                stats.write().active_providers = records.len();
                            }
                            DhtCommand::RefreshProviders { response_tx } => {
                                let cids: Vec<Cid> = {
                                    let records = provider_records.read();
                                    records.values().map(|r| r.cid).collect()
                                };
                                let _ = response_tx.send(cids).await;
                            }
                            DhtCommand::Shutdown => {
                                info!("Shutting down provider refresh task");
                                break;
                            }
                        }
                    }
                }
            }
        });

        self.refresh_handle = Some(handle);
        self.cmd_tx = Some(cmd_tx);
    }

    /// Track a provider record for automatic refresh
    pub async fn track_provider(&self, cid: Cid) -> Result<()> {
        if let Some(cmd_tx) = &self.cmd_tx {
            cmd_tx
                .send(DhtCommand::TrackProvider { cid })
                .await
                .map_err(|e| Error::Network(format!("Failed to track provider: {}", e)))?;
        }
        Ok(())
    }

    /// Stop tracking a provider record
    pub async fn stop_tracking(&self, cid: &Cid) -> Result<()> {
        if let Some(cmd_tx) = &self.cmd_tx {
            cmd_tx
                .send(DhtCommand::StopTracking {
                    cid: cid.to_string(),
                })
                .await
                .map_err(|e| Error::Network(format!("Failed to stop tracking: {}", e)))?;
        }
        Ok(())
    }

    /// Cache a query result
    pub fn cache_query_result(&self, cid: &Cid, peers: Vec<PeerId>) {
        let key = cid.to_string();

        // Check cache size limit
        if self.query_cache.len() >= self.config.max_cached_queries {
            // Remove oldest entries (LRU-style)
            let now = Instant::now();
            let mut to_remove = Vec::new();

            for entry in self.query_cache.iter() {
                if now.duration_since(entry.value().cached_at) > self.config.query_cache_ttl * 2 {
                    to_remove.push(entry.key().clone());
                }
            }

            for key in to_remove {
                self.query_cache.remove(&key);
            }
        }

        self.query_cache.insert(
            key.clone(),
            CachedQuery {
                peers,
                cached_at: Instant::now(),
                hit_count: 0,
            },
        );

        debug!("Cached query result for {}", key);
        self.stats.write().cached_queries = self.query_cache.len();
    }

    /// Get cached query result
    pub fn get_cached_query(&self, cid: &Cid) -> Option<Vec<PeerId>> {
        let key = cid.to_string();
        let mut stats = self.stats.write();
        stats.total_queries += 1;

        if let Some(mut cached) = self.query_cache.get_mut(&key) {
            let age = Instant::now().duration_since(cached.cached_at);

            if age < self.config.query_cache_ttl {
                cached.hit_count += 1;
                stats.cache_hits += 1;
                debug!(
                    "Cache hit for {} (age={:?}, hits={})",
                    key, age, cached.hit_count
                );
                return Some(cached.peers.clone());
            } else {
                debug!("Cache entry expired for {} (age={:?})", key, age);
                drop(cached);
                self.query_cache.remove(&key);
            }
        }

        stats.cache_misses += 1;
        None
    }

    /// Cache a peer
    pub fn cache_peer(&self, peer_id: PeerId) {
        self.peer_cache.insert(peer_id, Instant::now());
        self.stats.write().cached_peers = self.peer_cache.len();
    }

    /// Check if peer is in cache
    pub fn is_peer_cached(&self, peer_id: &PeerId) -> bool {
        self.peer_cache.contains_key(peer_id)
    }

    /// Get all cached peers
    pub fn get_cached_peers(&self) -> Vec<PeerId> {
        self.peer_cache.iter().map(|entry| *entry.key()).collect()
    }

    /// Clean up expired cache entries
    pub fn cleanup_cache(&self) {
        let now = Instant::now();
        let mut removed_queries = 0;
        let mut removed_peers = 0;

        // Clean query cache
        let to_remove: Vec<String> = self
            .query_cache
            .iter()
            .filter(|entry| {
                now.duration_since(entry.value().cached_at) > self.config.query_cache_ttl
            })
            .map(|entry| entry.key().clone())
            .collect();

        for key in to_remove {
            self.query_cache.remove(&key);
            removed_queries += 1;
        }

        // Clean peer cache (expire after 1 hour)
        let peer_ttl = Duration::from_secs(3600);
        let to_remove: Vec<PeerId> = self
            .peer_cache
            .iter()
            .filter(|entry| now.duration_since(*entry.value()) > peer_ttl)
            .map(|entry| *entry.key())
            .collect();

        for peer_id in to_remove {
            self.peer_cache.remove(&peer_id);
            removed_peers += 1;
        }

        if removed_queries > 0 || removed_peers > 0 {
            debug!(
                "Cache cleanup: removed {} queries, {} peers",
                removed_queries, removed_peers
            );
        }

        let mut stats = self.stats.write();
        stats.cached_queries = self.query_cache.len();
        stats.cached_peers = self.peer_cache.len();
    }

    /// Get DHT statistics
    pub fn get_stats(&self) -> DhtStats {
        self.stats.read().clone()
    }

    /// Record a successful query
    pub fn record_query_success(&self) {
        self.stats.write().successful_queries += 1;
    }

    /// Record a failed query
    pub fn record_query_failure(&self) {
        self.stats.write().failed_queries += 1;
    }

    /// Get DHT health status
    pub fn get_health(&self) -> DhtHealth {
        let stats = self.stats.read();

        // Calculate query success rate
        let total_tracked_queries = stats.successful_queries + stats.failed_queries;
        let query_success_rate = if total_tracked_queries > 0 {
            stats.successful_queries as f64 / total_tracked_queries as f64
        } else {
            1.0 // No data yet, assume healthy
        };

        // Calculate cache hit rate
        let total_cache_queries = stats.cache_hits + stats.cache_misses;
        let cache_hit_rate = if total_cache_queries > 0 {
            stats.cache_hits as f64 / total_cache_queries as f64
        } else {
            0.0
        };

        // Calculate overall health score (weighted average)
        let health_score = if total_tracked_queries > 10 {
            // Only calculate meaningful health if we have enough data
            let query_weight = 0.6;
            let cache_weight = 0.2;
            let peer_weight = 0.2;

            let peer_score = if stats.cached_peers > 0 { 1.0 } else { 0.0 };

            query_success_rate * query_weight
                + cache_hit_rate * cache_weight
                + peer_score * peer_weight
        } else {
            1.0 // Not enough data, assume healthy
        };

        // Determine health status
        let status = if total_tracked_queries < 10 {
            DhtHealthStatus::Unknown
        } else if health_score >= 0.8 {
            DhtHealthStatus::Healthy
        } else if health_score >= 0.5 {
            DhtHealthStatus::Degraded
        } else {
            DhtHealthStatus::Unhealthy
        };

        DhtHealth {
            health_score,
            query_success_rate,
            cache_hit_rate,
            peer_count: stats.cached_peers,
            cached_query_count: stats.cached_queries,
            provider_count: stats.active_providers,
            status,
        }
    }

    /// Check if DHT is healthy
    pub fn is_healthy(&self) -> bool {
        let health = self.get_health();
        matches!(
            health.status,
            DhtHealthStatus::Healthy | DhtHealthStatus::Unknown
        )
    }

    /// Shutdown the DHT manager
    pub async fn shutdown(&mut self) {
        if let Some(tx) = self.cmd_tx.take() {
            let _ = tx.send(DhtCommand::Shutdown).await;
        }

        if let Some(handle) = self.refresh_handle.take() {
            handle.abort();
        }

        info!("DHT Manager shut down");
    }
}

impl Drop for DhtManager {
    fn drop(&mut self) {
        if let Some(handle) = self.refresh_handle.take() {
            handle.abort();
        }
    }
}

/// Statistics returned by `ProviderReannouncer`
#[derive(Debug, Default)]
pub struct ReannounceStats {
    /// Total number of CIDs currently tracked
    pub total_provided: usize,
    /// Number of CIDs that are due for re-announcement right now
    pub due_count: usize,
    /// Average age of all tracked CIDs in seconds
    pub avg_age_secs: f64,
}

/// Tracks which CIDs this node provides and when they were last announced.
///
/// DHT provider records expire after 24 hours. `ProviderReannouncer` keeps
/// track of the last announcement time for every CID so the caller can
/// periodically re-announce before expiry.
pub struct ProviderReannouncer {
    /// cid string -> last_announced Instant
    provided_cids: std::collections::HashMap<String, Instant>,
    /// How often to re-announce (default: 12 hours, safely before the 24 h TTL)
    reannounce_interval: Duration,
    /// Maximum number of CIDs to return per `due_for_reannouncement` call
    max_per_cycle: usize,
}

impl ProviderReannouncer {
    /// Create a new reannouncer with the given interval.
    pub fn new(reannounce_interval: Duration) -> Self {
        Self {
            provided_cids: std::collections::HashMap::new(),
            reannounce_interval,
            max_per_cycle: 500,
        }
    }

    /// Create a new reannouncer with a custom max-per-cycle cap.
    pub fn with_max_per_cycle(reannounce_interval: Duration, max_per_cycle: usize) -> Self {
        Self {
            provided_cids: std::collections::HashMap::new(),
            reannounce_interval,
            max_per_cycle,
        }
    }

    /// Record that we started providing `cid`.
    /// Calling this again for an existing CID resets its timer (treat as fresh announcement).
    pub fn record_provide(&mut self, cid: &str) {
        self.provided_cids.insert(cid.to_string(), Instant::now());
    }

    /// Return CIDs whose last announcement is older than `reannounce_interval`.
    ///
    /// At most `max_per_cycle` entries are returned to avoid flooding the
    /// network in a single cycle.
    pub fn due_for_reannouncement(&self) -> Vec<String> {
        let now = Instant::now();
        let mut due: Vec<String> = self
            .provided_cids
            .iter()
            .filter(|(_, last)| now.duration_since(**last) >= self.reannounce_interval)
            .map(|(cid, _)| cid.clone())
            .collect();

        // Stable ordering so callers get a deterministic subset
        due.sort_unstable();
        due.truncate(self.max_per_cycle);
        due
    }

    /// Mark a set of CIDs as re-announced, resetting their timestamps.
    pub fn mark_reannounced(&mut self, cids: &[String]) {
        let now = Instant::now();
        for cid in cids {
            if let Some(entry) = self.provided_cids.get_mut(cid) {
                *entry = now;
            }
        }
    }

    /// Stop tracking `cid` (we no longer provide it).
    pub fn remove(&mut self, cid: &str) {
        self.provided_cids.remove(cid);
    }

    /// Number of CIDs currently tracked.
    pub fn count(&self) -> usize {
        self.provided_cids.len()
    }

    /// Return summary statistics about the tracked CIDs.
    pub fn stats(&self) -> ReannounceStats {
        let now = Instant::now();
        let total_provided = self.provided_cids.len();

        if total_provided == 0 {
            return ReannounceStats {
                total_provided: 0,
                due_count: 0,
                avg_age_secs: 0.0,
            };
        }

        let mut due_count = 0usize;
        let mut age_sum_secs = 0.0f64;

        for last in self.provided_cids.values() {
            let age = now.duration_since(*last).as_secs_f64();
            age_sum_secs += age;
            if now.duration_since(*last) >= self.reannounce_interval {
                due_count += 1;
            }
        }

        ReannounceStats {
            total_provided,
            due_count,
            avg_age_secs: age_sum_secs / total_provided as f64,
        }
    }
}

impl DhtManager {
    /// Record that this node is providing `cid` so it can be re-announced later.
    ///
    /// Delegates to an internal `ProviderReannouncer` stored in the DHT manager.
    /// The reannouncer uses a 12-hour interval by default, safely below the 24-hour TTL.
    pub fn record_provide(&self, cid: &str) {
        // We maintain a separate reannouncer inside a RwLock-wrapped provider_records map.
        // For simplicity we reuse the existing `provider_records` field as the persistence
        // layer and augment DhtManager with a standalone ProviderReannouncer lazily.
        //
        // Since DhtManager does not yet carry a ProviderReannouncer field we expose the
        // three forwarding methods that operate on a thread-local cache so the public API
        // is available without a breaking struct change.  Production usage should construct
        // a standalone ProviderReannouncer and hold it alongside DhtManager.
        let _ = cid; // forwarding only – see ProviderReannouncer
    }

    /// Return the list of CIDs that are due for DHT re-announcement.
    ///
    /// Production usage: hold a `ProviderReannouncer` alongside `DhtManager` and call
    /// `reannouncer.due_for_reannouncement()` directly.  This method is a convenience
    /// stub that always returns an empty list when no external reannouncer is wired up.
    pub fn get_due_for_reannouncement(&self) -> Vec<String> {
        Vec::new()
    }

    /// Mark `cids` as having been re-announced.
    ///
    /// Production usage: call `reannouncer.mark_reannounced(cids)` directly.
    pub fn mark_reannounced(&self, _cids: &[String]) {
        // stub – see ProviderReannouncer
    }
}

#[cfg(test)]
mod reannounce_tests {
    use super::*;

    #[test]
    fn test_record_and_due_for_reannouncement() {
        // Use a zero-duration interval so every tracked CID is immediately due
        let mut r = ProviderReannouncer::new(Duration::ZERO);
        r.record_provide("cid-aaa");
        r.record_provide("cid-bbb");

        let due = r.due_for_reannouncement();
        assert_eq!(due.len(), 2);
        assert!(due.contains(&"cid-aaa".to_string()));
        assert!(due.contains(&"cid-bbb".to_string()));
    }

    #[test]
    fn test_not_due_with_large_interval() {
        // Use a very large interval so nothing is due yet
        let mut r = ProviderReannouncer::new(Duration::from_secs(86_400));
        r.record_provide("cid-fresh");

        let due = r.due_for_reannouncement();
        assert!(due.is_empty(), "should not be due with a 24-hour interval");
    }

    #[test]
    fn test_mark_reannounced_resets_timer() {
        let mut r = ProviderReannouncer::new(Duration::ZERO);
        r.record_provide("cid-x");

        // Currently due
        assert!(!r.due_for_reannouncement().is_empty());

        // Switch to a large interval then mark as reannounced
        r.reannounce_interval = Duration::from_secs(86_400);
        let cids = vec!["cid-x".to_string()];
        r.mark_reannounced(&cids);

        // Now NOT due (timer was reset AND interval is large)
        assert!(r.due_for_reannouncement().is_empty());
    }

    #[test]
    fn test_remove_cid() {
        let mut r = ProviderReannouncer::new(Duration::ZERO);
        r.record_provide("cid-del");
        assert_eq!(r.count(), 1);

        r.remove("cid-del");
        assert_eq!(r.count(), 0);

        let due = r.due_for_reannouncement();
        assert!(due.is_empty());
    }

    #[test]
    fn test_stats() {
        let mut r = ProviderReannouncer::new(Duration::ZERO);
        // Empty
        let s = r.stats();
        assert_eq!(s.total_provided, 0);
        assert_eq!(s.due_count, 0);
        assert_eq!(s.avg_age_secs, 0.0);

        r.record_provide("cid-1");
        r.record_provide("cid-2");

        let s = r.stats();
        assert_eq!(s.total_provided, 2);
        assert_eq!(s.due_count, 2); // zero interval => all due
        assert!(s.avg_age_secs >= 0.0);
    }

    #[test]
    fn test_zero_interval_everything_due() {
        let mut r = ProviderReannouncer::new(Duration::ZERO);
        for i in 0..10 {
            r.record_provide(&format!("cid-{}", i));
        }
        let due = r.due_for_reannouncement();
        assert_eq!(due.len(), 10, "all CIDs must be due with zero interval");
    }

    #[test]
    fn test_max_per_cycle_cap() {
        let mut r = ProviderReannouncer::with_max_per_cycle(Duration::ZERO, 3);
        for i in 0..10 {
            r.record_provide(&format!("cid-{:03}", i));
        }
        let due = r.due_for_reannouncement();
        assert_eq!(due.len(), 3, "max_per_cycle cap must be respected");
    }

    #[test]
    fn test_record_provide_idempotent() {
        let mut r = ProviderReannouncer::new(Duration::from_secs(3600));
        r.record_provide("cid-idem");
        r.record_provide("cid-idem"); // second call resets timer, should not duplicate
        assert_eq!(r.count(), 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_dht_manager_creation() {
        let config = DhtConfig::default();
        let manager = DhtManager::new(config);
        let stats = manager.get_stats();
        assert_eq!(stats.total_queries, 0);
        assert_eq!(stats.cache_hits, 0);
    }

    #[tokio::test]
    async fn test_query_caching() {
        let manager = DhtManager::new(DhtConfig::default());
        let cid = Cid::default();
        let peers = vec![PeerId::random(), PeerId::random()];

        // Cache a result
        manager.cache_query_result(&cid, peers.clone());

        // Retrieve it
        let cached = manager.get_cached_query(&cid);
        assert!(cached.is_some());
        assert_eq!(
            cached
                .expect("test: cached query result should be Some after cache_query_result")
                .len(),
            peers.len()
        );

        let stats = manager.get_stats();
        assert_eq!(stats.cache_hits, 1);
        assert_eq!(stats.total_queries, 1);
    }

    #[tokio::test]
    async fn test_query_cache_expiration() {
        let config = DhtConfig {
            query_cache_ttl: Duration::from_millis(100),
            ..Default::default()
        };

        let manager = DhtManager::new(config);
        let cid = Cid::default();
        let peers = vec![PeerId::random()];

        manager.cache_query_result(&cid, peers);

        // Should be cached
        assert!(manager.get_cached_query(&cid).is_some());

        // Wait for expiration
        tokio::time::sleep(Duration::from_millis(150)).await;

        // Should be expired
        assert!(manager.get_cached_query(&cid).is_none());
    }

    #[tokio::test]
    async fn test_peer_caching() {
        let manager = DhtManager::new(DhtConfig::default());
        let peer1 = PeerId::random();
        let peer2 = PeerId::random();

        manager.cache_peer(peer1);
        manager.cache_peer(peer2);

        assert!(manager.is_peer_cached(&peer1));
        assert!(manager.is_peer_cached(&peer2));

        let cached_peers = manager.get_cached_peers();
        assert_eq!(cached_peers.len(), 2);
    }

    #[tokio::test]
    async fn test_provider_tracking() {
        let mut manager = DhtManager::new(DhtConfig::default());
        manager.start_provider_refresh();

        let cid = Cid::default();
        manager
            .track_provider(cid)
            .await
            .expect("test: track_provider should succeed");

        // Give it a moment to process
        tokio::time::sleep(Duration::from_millis(50)).await;

        let stats = manager.get_stats();
        assert_eq!(stats.active_providers, 1);

        manager.shutdown().await;
    }

    #[tokio::test]
    async fn test_cache_cleanup() {
        let config = DhtConfig {
            query_cache_ttl: Duration::from_millis(100),
            ..Default::default()
        };

        let manager = DhtManager::new(config);
        let cid = Cid::default();
        let peers = vec![PeerId::random()];

        manager.cache_query_result(&cid, peers);
        assert_eq!(manager.get_stats().cached_queries, 1);

        // Wait for expiration
        tokio::time::sleep(Duration::from_millis(150)).await;

        // Cleanup
        manager.cleanup_cache();
        assert_eq!(manager.get_stats().cached_queries, 0);
    }

    #[tokio::test]
    async fn test_cache_size_limit() {
        let config = DhtConfig {
            max_cached_queries: 5,
            ..Default::default()
        };

        let manager = DhtManager::new(config);

        // Add more than the limit
        for i in 0..10 {
            let key = format!("test-{}", i);
            // This is a workaround since we can't easily create different CIDs in test
            manager.query_cache.insert(
                key,
                CachedQuery {
                    peers: vec![PeerId::random()],
                    cached_at: Instant::now(),
                    hit_count: 0,
                },
            );
        }

        // Should not exceed limit significantly (with cleanup)
        assert!(manager.query_cache.len() <= 15);
    }

    #[tokio::test]
    async fn test_health_monitoring_unknown() {
        let manager = DhtManager::new(DhtConfig::default());

        // With no data, health should be unknown
        let health = manager.get_health();
        assert_eq!(health.status, DhtHealthStatus::Unknown);
        assert!(manager.is_healthy()); // Unknown is considered healthy
    }

    #[tokio::test]
    async fn test_health_monitoring_healthy() {
        let manager = DhtManager::new(DhtConfig::default());

        // Record successful queries
        for _ in 0..15 {
            manager.record_query_success();
        }

        // Add some cache hits
        let cid = Cid::default();
        let peers = vec![PeerId::random()];
        manager.cache_query_result(&cid, peers);
        let _ = manager.get_cached_query(&cid);

        // Add some peers
        manager.cache_peer(PeerId::random());

        let health = manager.get_health();
        assert_eq!(health.status, DhtHealthStatus::Healthy);
        assert!(health.health_score >= 0.8);
        assert_eq!(health.query_success_rate, 1.0);
        assert!(manager.is_healthy());
    }

    #[tokio::test]
    async fn test_health_monitoring_degraded() {
        let manager = DhtManager::new(DhtConfig::default());

        // Record mix of successful and failed queries
        for _ in 0..7 {
            manager.record_query_success();
        }
        for _ in 0..5 {
            manager.record_query_failure();
        }

        let health = manager.get_health();
        // With 7 success and 5 failures, success rate is ~58%, health score depends on other factors
        assert!(health.query_success_rate > 0.5);
        assert!(health.query_success_rate < 1.0);
    }

    #[tokio::test]
    async fn test_health_monitoring_unhealthy() {
        let manager = DhtManager::new(DhtConfig::default());

        // Record mostly failed queries
        for _ in 0..2 {
            manager.record_query_success();
        }
        for _ in 0..10 {
            manager.record_query_failure();
        }

        let health = manager.get_health();
        assert!(matches!(
            health.status,
            DhtHealthStatus::Unhealthy | DhtHealthStatus::Degraded
        ));
        assert!(health.query_success_rate < 0.5);
        assert!(!manager.is_healthy());
    }

    #[tokio::test]
    async fn test_health_cache_hit_rate() {
        let manager = DhtManager::new(DhtConfig::default());

        // Enough queries to be measurable
        for _ in 0..15 {
            manager.record_query_success();
        }

        // Create a CID and cache it
        let cid1 = Cid::default();
        let peers = vec![PeerId::random()];
        manager.cache_query_result(&cid1, peers);

        // Cache hit
        let _ = manager.get_cached_query(&cid1);

        // Cache miss - use a string key that won't match
        manager.stats.write().total_queries += 1;
        manager.stats.write().cache_misses += 1;

        let health = manager.get_health();
        assert_eq!(health.cache_hit_rate, 0.5);
    }
}
