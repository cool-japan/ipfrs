//! Storage Pool Manager for multi-backend routing
//!
//! This module provides a storage pool that manages multiple backends with
//! intelligent routing strategies including:
//! - Load balancing across backends
//! - Size-based routing (small blocks to fast storage, large to cold)
//! - Cost-aware routing for cloud storage
//! - Automatic failover and redundancy
//! - Backend health monitoring

use crate::traits::BlockStore;
use async_trait::async_trait;
use dashmap::DashMap;
use ipfrs_core::{Block, Cid, Error, Result};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{debug, warn};

/// Backend identifier
pub type BackendId = String;

/// Routing strategy for selecting backends
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RoutingStrategy {
    /// Round-robin load balancing
    RoundRobin,
    /// Route based on block size (small to fast, large to slow)
    SizeBased,
    /// Route to least loaded backend
    LeastLoaded,
    /// Route to lowest cost backend
    CostAware,
    /// Route to geographically closest backend
    LatencyAware,
    /// Replicate to all backends
    Replicated,
    /// Hash-based consistent hashing
    ConsistentHash,
}

/// Backend configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendConfig {
    /// Unique backend identifier
    pub id: BackendId,
    /// Backend priority (higher = preferred)
    pub priority: u8,
    /// Maximum capacity in bytes (0 = unlimited)
    pub capacity: u64,
    /// Current used bytes
    pub used: u64,
    /// Cost per GB per month (for cost-aware routing)
    pub cost_per_gb: f64,
    /// Average read latency in milliseconds
    pub avg_latency_ms: f64,
    /// Block size threshold (route blocks larger than this)
    pub size_threshold: Option<u64>,
    /// Whether this backend is healthy
    pub healthy: bool,
    /// Whether to use for reads
    pub read_enabled: bool,
    /// Whether to use for writes
    pub write_enabled: bool,
}

impl Default for BackendConfig {
    fn default() -> Self {
        Self {
            id: "default".to_string(),
            priority: 100,
            capacity: 0,
            used: 0,
            cost_per_gb: 0.0,
            avg_latency_ms: 10.0,
            size_threshold: None,
            healthy: true,
            read_enabled: true,
            write_enabled: true,
        }
    }
}

/// Backend statistics
#[derive(Debug, Default)]
pub struct BackendStats {
    /// Total read operations
    pub reads: AtomicU64,
    /// Total write operations
    pub writes: AtomicU64,
    /// Total bytes read
    pub bytes_read: AtomicU64,
    /// Total bytes written
    pub bytes_written: AtomicU64,
    /// Total errors
    pub errors: AtomicU64,
    /// Last health check time
    pub last_health_check: parking_lot::Mutex<Option<Instant>>,
}

impl BackendStats {
    fn record_read(&self, bytes: u64) {
        self.reads.fetch_add(1, Ordering::Relaxed);
        self.bytes_read.fetch_add(bytes, Ordering::Relaxed);
    }

    fn record_write(&self, bytes: u64) {
        self.writes.fetch_add(1, Ordering::Relaxed);
        self.bytes_written.fetch_add(bytes, Ordering::Relaxed);
    }

    fn record_error(&self) {
        self.errors.fetch_add(1, Ordering::Relaxed);
    }

    fn update_health_check(&self) {
        *self.last_health_check.lock() = Some(Instant::now());
    }
}

/// Backend wrapper with metadata
struct Backend<S: BlockStore> {
    store: Arc<S>,
    config: parking_lot::RwLock<BackendConfig>,
    stats: BackendStats,
}

/// Storage pool configuration
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// Routing strategy
    pub strategy: RoutingStrategy,
    /// Replication factor (for replicated strategy)
    pub replication_factor: usize,
    /// Health check interval
    pub health_check_interval: Duration,
    /// Enable automatic failover
    pub auto_failover: bool,
    /// Minimum healthy backends required
    pub min_healthy_backends: usize,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            strategy: RoutingStrategy::RoundRobin,
            replication_factor: 1,
            health_check_interval: Duration::from_secs(30),
            auto_failover: true,
            min_healthy_backends: 1,
        }
    }
}

/// Storage pool statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolStats {
    /// Total backends
    pub total_backends: usize,
    /// Healthy backends
    pub healthy_backends: usize,
    /// Total capacity
    pub total_capacity: u64,
    /// Total used
    pub total_used: u64,
    /// Total reads
    pub total_reads: u64,
    /// Total writes
    pub total_writes: u64,
    /// Total errors
    pub total_errors: u64,
    /// Average cost per GB
    pub avg_cost_per_gb: f64,
    /// Average latency
    pub avg_latency_ms: f64,
}

/// Storage Pool Manager
///
/// Manages multiple storage backends with intelligent routing
pub struct StoragePool<S: BlockStore> {
    backends: DashMap<BackendId, Backend<S>>,
    config: parking_lot::RwLock<PoolConfig>,
    round_robin_counter: AtomicUsize,
    /// CID to backend mapping (for tracking where blocks are stored)
    cid_map: DashMap<Cid, Vec<BackendId>>,
}

impl<S: BlockStore> StoragePool<S> {
    /// Create a new storage pool
    pub fn new(config: PoolConfig) -> Self {
        Self {
            backends: DashMap::new(),
            config: parking_lot::RwLock::new(config),
            round_robin_counter: AtomicUsize::new(0),
            cid_map: DashMap::new(),
        }
    }

    /// Add a backend to the pool
    pub fn add_backend(&self, config: BackendConfig, store: Arc<S>) {
        let id = config.id.clone();
        let backend = Backend {
            store,
            config: parking_lot::RwLock::new(config),
            stats: BackendStats::default(),
        };
        self.backends.insert(id.clone(), backend);
        debug!("Added backend to pool: {}", id);
    }

    /// Remove a backend from the pool
    pub fn remove_backend(&self, id: &str) -> Option<Arc<S>> {
        self.backends.remove(id).map(|(_, backend)| backend.store)
    }

    /// Get backend configuration
    pub fn get_backend_config(&self, id: &str) -> Option<BackendConfig> {
        self.backends
            .get(id)
            .map(|backend| backend.config.read().clone())
    }

    /// Update backend configuration
    pub fn update_backend_config(&self, id: &str, config: BackendConfig) -> Result<()> {
        let backend = self
            .backends
            .get(id)
            .ok_or_else(|| Error::Storage(format!("Backend not found: {}", id)))?;
        *backend.config.write() = config;
        Ok(())
    }

    /// Mark backend as healthy or unhealthy
    pub fn set_backend_health(&self, id: &str, healthy: bool) -> Result<()> {
        let backend = self
            .backends
            .get(id)
            .ok_or_else(|| Error::Storage(format!("Backend not found: {}", id)))?;
        backend.config.write().healthy = healthy;
        backend.stats.update_health_check();
        debug!("Backend {} health set to: {}", id, healthy);
        Ok(())
    }

    /// Get pool statistics
    pub fn stats(&self) -> PoolStats {
        let mut total_capacity = 0u64;
        let mut total_used = 0u64;
        let mut total_reads = 0u64;
        let mut total_writes = 0u64;
        let mut total_errors = 0u64;
        let mut total_cost = 0.0;
        let mut total_latency = 0.0;
        let mut healthy_count = 0;
        let total_count = self.backends.len();

        for backend in self.backends.iter() {
            let config = backend.config.read();
            let stats = &backend.stats;

            if config.healthy {
                healthy_count += 1;
            }

            total_capacity += config.capacity;
            total_used += config.used;
            total_reads += stats.reads.load(Ordering::Relaxed);
            total_writes += stats.writes.load(Ordering::Relaxed);
            total_errors += stats.errors.load(Ordering::Relaxed);
            total_cost += config.cost_per_gb;
            total_latency += config.avg_latency_ms;
        }

        let avg_cost_per_gb = if total_count > 0 {
            total_cost / total_count as f64
        } else {
            0.0
        };

        let avg_latency_ms = if total_count > 0 {
            total_latency / total_count as f64
        } else {
            0.0
        };

        PoolStats {
            total_backends: total_count,
            healthy_backends: healthy_count,
            total_capacity,
            total_used,
            total_reads,
            total_writes,
            total_errors,
            avg_cost_per_gb,
            avg_latency_ms,
        }
    }

    /// Select backends for a write operation
    #[allow(dead_code)]
    fn select_backends_for_write(&self, cid: &Cid, data_size: usize) -> Vec<BackendId> {
        let config = self.config.read();
        let strategy = config.strategy;
        let replication_factor = config.replication_factor;

        match strategy {
            RoutingStrategy::RoundRobin => self.select_round_robin(replication_factor),
            RoutingStrategy::SizeBased => self.select_size_based(data_size, replication_factor),
            RoutingStrategy::LeastLoaded => self.select_least_loaded(replication_factor),
            RoutingStrategy::CostAware => self.select_cost_aware(replication_factor),
            RoutingStrategy::LatencyAware => self.select_latency_aware(replication_factor),
            RoutingStrategy::Replicated => self.select_all_healthy(),
            RoutingStrategy::ConsistentHash => self.select_consistent_hash(cid, replication_factor),
        }
    }

    /// Select backends using round-robin
    fn select_round_robin(&self, count: usize) -> Vec<BackendId> {
        let healthy: Vec<_> = self
            .backends
            .iter()
            .filter(|b| b.config.read().healthy && b.config.read().write_enabled)
            .map(|b| b.config.read().id.clone())
            .collect();

        if healthy.is_empty() {
            return Vec::new();
        }

        let mut selected = Vec::new();
        for _ in 0..count.min(healthy.len()) {
            let idx = self.round_robin_counter.fetch_add(1, Ordering::Relaxed) % healthy.len();
            selected.push(healthy[idx].clone());
        }
        selected
    }

    /// Select backends based on size
    fn select_size_based(&self, data_size: usize, count: usize) -> Vec<BackendId> {
        let mut candidates: Vec<_> = self
            .backends
            .iter()
            .filter_map(|b| {
                let config = b.config.read();
                if !config.healthy || !config.write_enabled {
                    return None;
                }

                let matches_size = if let Some(threshold) = config.size_threshold {
                    if data_size >= threshold as usize {
                        config.priority >= 50 // Low priority for large blocks
                    } else {
                        config.priority > 50 // High priority for small blocks
                    }
                } else {
                    true
                };

                if matches_size {
                    Some((config.id.clone(), config.priority))
                } else {
                    None
                }
            })
            .collect();

        candidates.sort_by_key(|a| std::cmp::Reverse(a.1));
        candidates
            .into_iter()
            .take(count)
            .map(|(id, _)| id)
            .collect()
    }

    /// Select least loaded backends
    fn select_least_loaded(&self, count: usize) -> Vec<BackendId> {
        let mut candidates: Vec<_> = self
            .backends
            .iter()
            .filter_map(|b| {
                let config = b.config.read();
                if !config.healthy || !config.write_enabled {
                    return None;
                }

                let load = if config.capacity > 0 {
                    (config.used as f64 / config.capacity as f64 * 100.0) as u64
                } else {
                    0
                };

                Some((config.id.clone(), load))
            })
            .collect();

        candidates.sort_by_key(|(_, load)| *load);
        candidates
            .into_iter()
            .take(count)
            .map(|(id, _)| id)
            .collect()
    }

    /// Select lowest cost backends
    fn select_cost_aware(&self, count: usize) -> Vec<BackendId> {
        let mut candidates: Vec<_> = self
            .backends
            .iter()
            .filter_map(|b| {
                let config = b.config.read();
                if !config.healthy || !config.write_enabled {
                    return None;
                }
                Some((config.id.clone(), (config.cost_per_gb * 1000.0) as u64))
            })
            .collect();

        candidates.sort_by_key(|(_, cost)| *cost);
        candidates
            .into_iter()
            .take(count)
            .map(|(id, _)| id)
            .collect()
    }

    /// Select lowest latency backends
    fn select_latency_aware(&self, count: usize) -> Vec<BackendId> {
        let mut candidates: Vec<_> = self
            .backends
            .iter()
            .filter_map(|b| {
                let config = b.config.read();
                if !config.healthy || !config.read_enabled {
                    return None;
                }
                Some((config.id.clone(), (config.avg_latency_ms * 1000.0) as u64))
            })
            .collect();

        candidates.sort_by_key(|(_, latency)| *latency);
        candidates
            .into_iter()
            .take(count)
            .map(|(id, _)| id)
            .collect()
    }

    /// Select all healthy backends
    fn select_all_healthy(&self) -> Vec<BackendId> {
        self.backends
            .iter()
            .filter_map(|b| {
                let config = b.config.read();
                if config.healthy && config.write_enabled {
                    Some(config.id.clone())
                } else {
                    None
                }
            })
            .collect()
    }

    /// Select backends using consistent hashing
    fn select_consistent_hash(&self, cid: &Cid, count: usize) -> Vec<BackendId> {
        let healthy: Vec<_> = self
            .backends
            .iter()
            .filter_map(|b| {
                let config = b.config.read();
                if config.healthy && config.write_enabled {
                    Some(config.id.clone())
                } else {
                    None
                }
            })
            .collect();

        if healthy.is_empty() {
            return Vec::new();
        }

        // Use CID hash to determine backend
        let cid_bytes = cid.to_bytes();
        let hash = cid_bytes
            .iter()
            .fold(0u64, |acc, &b| acc.wrapping_mul(31).wrapping_add(b as u64));

        let mut selected = Vec::new();
        for i in 0..count.min(healthy.len()) {
            let idx = ((hash + i as u64) % healthy.len() as u64) as usize;
            selected.push(healthy[idx].clone());
        }
        selected
    }

    /// Get backends where a CID is stored
    fn get_backends_for_cid(&self, cid: &Cid) -> Vec<BackendId> {
        self.cid_map
            .get(cid)
            .map(|backends| backends.clone())
            .unwrap_or_default()
    }

    /// Record that a CID is stored in a backend
    fn record_cid_location(&self, cid: Cid, backend_id: BackendId) {
        self.cid_map.entry(cid).or_default().push(backend_id);
    }
}

#[async_trait]
impl<S: BlockStore + Send + Sync + 'static> BlockStore for StoragePool<S> {
    async fn get(&self, cid: &Cid) -> Result<Option<Block>> {
        // Try to get from known backends first
        let known_backends = self.get_backends_for_cid(cid);

        for backend_id in known_backends {
            if let Some(backend) = self.backends.get(&backend_id) {
                let (healthy, read_enabled) = {
                    let config = backend.config.read();
                    (config.healthy, config.read_enabled)
                };

                if !healthy || !read_enabled {
                    continue;
                }

                match backend.store.get(cid).await {
                    Ok(Some(block)) => {
                        backend.stats.record_read(block.data().len() as u64);
                        return Ok(Some(block));
                    }
                    Ok(None) => continue,
                    Err(e) => {
                        warn!("Backend {} failed to get CID: {}", backend_id, e);
                        backend.stats.record_error();
                    }
                }
            }
        }

        // Fallback: try all healthy backends
        for backend in self.backends.iter() {
            let (healthy, read_enabled, backend_id) = {
                let config = backend.config.read();
                (config.healthy, config.read_enabled, config.id.clone())
            };

            if !healthy || !read_enabled {
                continue;
            }

            match backend.store.get(cid).await {
                Ok(Some(block)) => {
                    backend.stats.record_read(block.data().len() as u64);
                    // Update mapping
                    self.record_cid_location(*cid, backend_id);
                    return Ok(Some(block));
                }
                Ok(None) => continue,
                Err(_) => {
                    backend.stats.record_error();
                }
            }
        }

        Ok(None)
    }

    async fn put(&self, block: &Block) -> Result<()> {
        let cid = block.cid();
        let data_size = block.data().len();
        let backends = self.select_backends_for_write(cid, data_size);

        if backends.is_empty() {
            return Err(Error::Storage(
                "No healthy backends available for write".to_string(),
            ));
        }

        let mut errors = Vec::new();
        let mut success_count = 0;

        for backend_id in &backends {
            if let Some(backend) = self.backends.get(backend_id) {
                match backend.store.put(block).await {
                    Ok(()) => {
                        backend.stats.record_write(data_size as u64);
                        self.record_cid_location(*cid, backend_id.clone());
                        success_count += 1;
                    }
                    Err(e) => {
                        backend.stats.record_error();
                        errors.push((backend_id.clone(), e));
                    }
                }
            }
        }

        if success_count == 0 {
            return Err(Error::Storage(format!(
                "Failed to write to any backend: {} errors",
                errors.len()
            )));
        }

        Ok(())
    }

    async fn has(&self, cid: &Cid) -> Result<bool> {
        // Check known backends first
        let known_backends = self.get_backends_for_cid(cid);

        for backend_id in known_backends {
            if let Some(backend) = self.backends.get(&backend_id) {
                if !backend.config.read().healthy {
                    continue;
                }

                if let Ok(true) = backend.store.has(cid).await {
                    return Ok(true);
                }
            }
        }

        // Fallback: check all healthy backends
        for backend in self.backends.iter() {
            if !backend.config.read().healthy {
                continue;
            }

            if let Ok(true) = backend.store.has(cid).await {
                // Update mapping
                self.record_cid_location(*cid, backend.config.read().id.clone());
                return Ok(true);
            }
        }

        Ok(false)
    }

    async fn delete(&self, cid: &Cid) -> Result<()> {
        let backends = self.get_backends_for_cid(cid);

        if backends.is_empty() {
            // Try all backends if we don't know where it's stored
            for backend in self.backends.iter() {
                let _ = backend.store.delete(cid).await;
            }
        } else {
            for backend_id in &backends {
                if let Some(backend) = self.backends.get(backend_id) {
                    let _ = backend.store.delete(cid).await;
                }
            }
        }

        // Remove from mapping
        self.cid_map.remove(cid);
        Ok(())
    }

    fn list_cids(&self) -> Result<Vec<Cid>> {
        // Get unique CIDs from the mapping
        let cids: Vec<Cid> = self.cid_map.iter().map(|entry| *entry.key()).collect();
        Ok(cids)
    }

    fn len(&self) -> usize {
        self.cid_map.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::MemoryBlockStore;
    use bytes::Bytes;

    #[tokio::test]
    async fn test_pool_basic() {
        let pool = StoragePool::new(PoolConfig::default());

        let backend1 = Arc::new(MemoryBlockStore::new());
        let config1 = BackendConfig {
            id: "backend1".to_string(),
            ..Default::default()
        };

        pool.add_backend(config1, backend1);

        let data = Bytes::from_static(b"test data");
        let block = Block::new(data).unwrap();
        let cid = block.cid();

        pool.put(&block).await.unwrap();
        assert!(pool.has(cid).await.unwrap());

        let retrieved = pool.get(cid).await.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().data(), block.data());
    }

    #[tokio::test]
    async fn test_pool_replicated() {
        let config = PoolConfig {
            strategy: RoutingStrategy::Replicated,
            ..Default::default()
        };
        let pool = StoragePool::new(config);

        let backend1 = Arc::new(MemoryBlockStore::new());
        let backend2 = Arc::new(MemoryBlockStore::new());

        pool.add_backend(
            BackendConfig {
                id: "backend1".to_string(),
                ..Default::default()
            },
            backend1.clone(),
        );

        pool.add_backend(
            BackendConfig {
                id: "backend2".to_string(),
                ..Default::default()
            },
            backend2.clone(),
        );

        let data = Bytes::from_static(b"test data");
        let block = Block::new(data).unwrap();
        let cid = block.cid();

        pool.put(&block).await.unwrap();

        // Should be in both backends
        assert!(backend1.has(cid).await.unwrap());
        assert!(backend2.has(cid).await.unwrap());
    }

    #[tokio::test]
    async fn test_pool_stats() {
        let pool = StoragePool::new(PoolConfig::default());

        let backend1 = Arc::new(MemoryBlockStore::new());
        pool.add_backend(
            BackendConfig {
                id: "backend1".to_string(),
                capacity: 1000,
                cost_per_gb: 0.023,
                ..Default::default()
            },
            backend1,
        );

        let stats = pool.stats();
        assert_eq!(stats.total_backends, 1);
        assert_eq!(stats.healthy_backends, 1);
        assert_eq!(stats.total_capacity, 1000);
    }
}
