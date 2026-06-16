//! Quota Management for per-tenant storage limits
//!
//! This module provides quota enforcement and tracking for multi-tenant storage:
//! - Per-tenant storage quotas
//! - Block count limits
//! - Bandwidth quotas (reads/writes per period)
//! - Quota enforcement with soft/hard limits
//! - Usage tracking and reporting
//! - Quota alerts and notifications

use crate::traits::BlockStore;
use async_trait::async_trait;
use dashmap::DashMap;
use ipfrs_core::{Block, Cid, Error, Result};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tracing::{debug, warn};

/// Tenant identifier
pub type TenantId = String;

/// Quota configuration for a tenant
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuotaConfig {
    /// Maximum storage in bytes (0 = unlimited)
    pub max_bytes: u64,
    /// Maximum number of blocks (0 = unlimited)
    pub max_blocks: u64,
    /// Maximum read bandwidth per period (bytes/sec, 0 = unlimited)
    pub max_read_bandwidth: u64,
    /// Maximum write bandwidth per period (bytes/sec, 0 = unlimited)
    pub max_write_bandwidth: u64,
    /// Soft limit threshold (percentage, e.g., 80 for 80%)
    pub soft_limit_percent: u8,
    /// Hard limit enforcement (reject on exceed)
    pub hard_limit_enabled: bool,
}

impl Default for QuotaConfig {
    fn default() -> Self {
        Self {
            max_bytes: 0,
            max_blocks: 0,
            max_read_bandwidth: 0,
            max_write_bandwidth: 0,
            soft_limit_percent: 80,
            hard_limit_enabled: true,
        }
    }
}

/// Quota usage statistics
#[derive(Debug)]
pub struct QuotaUsage {
    /// Current storage used in bytes
    pub bytes_used: AtomicU64,
    /// Current number of blocks
    pub blocks_count: AtomicU64,
    /// Total bytes read in current period
    pub bytes_read: AtomicU64,
    /// Total bytes written in current period
    pub bytes_written: AtomicU64,
    /// Number of quota violations
    pub violations: AtomicU64,
    /// Last reset time for bandwidth tracking
    pub last_reset: parking_lot::Mutex<SystemTime>,
}

impl QuotaUsage {
    fn new() -> Self {
        Self {
            bytes_used: AtomicU64::new(0),
            blocks_count: AtomicU64::new(0),
            bytes_read: AtomicU64::new(0),
            bytes_written: AtomicU64::new(0),
            violations: AtomicU64::new(0),
            last_reset: parking_lot::Mutex::new(SystemTime::now()),
        }
    }

    fn record_write(&self, bytes: u64) {
        self.bytes_used.fetch_add(bytes, Ordering::Relaxed);
        self.blocks_count.fetch_add(1, Ordering::Relaxed);
        self.bytes_written.fetch_add(bytes, Ordering::Relaxed);
    }

    fn record_read(&self, bytes: u64) {
        self.bytes_read.fetch_add(bytes, Ordering::Relaxed);
    }

    fn record_delete(&self, bytes: u64) {
        self.bytes_used.fetch_sub(bytes, Ordering::Relaxed);
        self.blocks_count.fetch_sub(1, Ordering::Relaxed);
    }

    fn record_violation(&self) {
        self.violations.fetch_add(1, Ordering::Relaxed);
    }

    fn reset_bandwidth(&self) {
        self.bytes_read.store(0, Ordering::Relaxed);
        self.bytes_written.store(0, Ordering::Relaxed);
        *self.last_reset.lock() = SystemTime::now();
    }

    fn should_reset(&self, period: Duration) -> bool {
        let last = *self.last_reset.lock();
        SystemTime::now().duration_since(last).unwrap_or_default() > period
    }
}

/// Quota status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuotaStatus {
    /// Within limits
    Ok,
    /// Exceeded soft limit (warning)
    SoftLimitExceeded,
    /// Exceeded hard limit (rejected)
    HardLimitExceeded,
}

/// Quota violation type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ViolationType {
    /// Storage bytes exceeded
    StorageBytes,
    /// Block count exceeded
    BlockCount,
    /// Read bandwidth exceeded
    ReadBandwidth,
    /// Write bandwidth exceeded
    WriteBandwidth,
}

/// Quota manager configuration
#[derive(Debug, Clone)]
pub struct QuotaManagerConfig {
    /// Default quota for new tenants
    pub default_quota: QuotaConfig,
    /// Bandwidth tracking period
    pub bandwidth_period: Duration,
    /// Enable quota enforcement
    pub enforcement_enabled: bool,
}

impl Default for QuotaManagerConfig {
    fn default() -> Self {
        Self {
            default_quota: QuotaConfig::default(),
            bandwidth_period: Duration::from_secs(60),
            enforcement_enabled: true,
        }
    }
}

/// Tenant quota information
struct TenantQuota {
    config: parking_lot::RwLock<QuotaConfig>,
    usage: QuotaUsage,
}

/// Quota Manager
///
/// Manages per-tenant storage quotas with enforcement
pub struct QuotaManager {
    tenants: DashMap<TenantId, TenantQuota>,
    config: parking_lot::RwLock<QuotaManagerConfig>,
    /// Mapping of CID to (tenant_id, size)
    cid_map: DashMap<Cid, (TenantId, u64)>,
}

impl QuotaManager {
    /// Create a new quota manager
    pub fn new(config: QuotaManagerConfig) -> Self {
        Self {
            tenants: DashMap::new(),
            config: parking_lot::RwLock::new(config),
            cid_map: DashMap::new(),
        }
    }

    /// Set quota for a tenant
    pub fn set_quota(&self, tenant_id: TenantId, config: QuotaConfig) {
        self.tenants
            .entry(tenant_id.clone())
            .and_modify(|tenant| *tenant.config.write() = config.clone())
            .or_insert_with(|| TenantQuota {
                config: parking_lot::RwLock::new(config),
                usage: QuotaUsage::new(),
            });
        debug!("Set quota for tenant: {}", tenant_id);
    }

    /// Get quota configuration for a tenant
    pub fn get_quota(&self, tenant_id: &str) -> Option<QuotaConfig> {
        self.tenants
            .get(tenant_id)
            .map(|tenant| tenant.config.read().clone())
    }

    /// Get quota usage for a tenant
    pub fn get_usage(&self, tenant_id: &str) -> Option<QuotaUsageSnapshot> {
        self.tenants
            .get(tenant_id)
            .map(|tenant| QuotaUsageSnapshot {
                bytes_used: tenant.usage.bytes_used.load(Ordering::Relaxed),
                blocks_count: tenant.usage.blocks_count.load(Ordering::Relaxed),
                bytes_read: tenant.usage.bytes_read.load(Ordering::Relaxed),
                bytes_written: tenant.usage.bytes_written.load(Ordering::Relaxed),
                violations: tenant.usage.violations.load(Ordering::Relaxed),
            })
    }

    /// Check if a write operation is allowed
    pub fn check_write_quota(
        &self,
        tenant_id: &str,
        data_size: u64,
    ) -> std::result::Result<QuotaStatus, ViolationType> {
        let (enforcement_enabled, bandwidth_period) = {
            let config_guard = self.config.read();
            (
                config_guard.enforcement_enabled,
                config_guard.bandwidth_period,
            )
        };

        if !enforcement_enabled {
            return Ok(QuotaStatus::Ok);
        }

        let tenant = match self.tenants.get(tenant_id) {
            Some(t) => t,
            None => {
                // Create tenant with default quota
                let default_quota = self.config.read().default_quota.clone();
                self.set_quota(tenant_id.to_string(), default_quota);
                self.tenants
                    .get(tenant_id)
                    .expect("tenant just inserted via set_quota")
            }
        };

        let quota_config = tenant.config.read();
        let usage = &tenant.usage;

        // Reset bandwidth if period expired
        if usage.should_reset(bandwidth_period) {
            usage.reset_bandwidth();
        }

        // Check storage bytes
        if quota_config.max_bytes > 0 {
            let current = usage.bytes_used.load(Ordering::Relaxed);
            let projected = current + data_size;
            let soft_limit =
                (quota_config.max_bytes * quota_config.soft_limit_percent as u64) / 100;

            if projected > quota_config.max_bytes {
                if quota_config.hard_limit_enabled {
                    usage.record_violation();
                    return Err(ViolationType::StorageBytes);
                }
                return Ok(QuotaStatus::HardLimitExceeded);
            } else if projected > soft_limit {
                warn!(
                    "Tenant {} exceeded soft storage limit: {} / {}",
                    tenant_id, projected, quota_config.max_bytes
                );
                return Ok(QuotaStatus::SoftLimitExceeded);
            }
        }

        // Check block count
        if quota_config.max_blocks > 0 {
            let current = usage.blocks_count.load(Ordering::Relaxed);
            let soft_limit =
                (quota_config.max_blocks * quota_config.soft_limit_percent as u64) / 100;

            if current + 1 > quota_config.max_blocks {
                if quota_config.hard_limit_enabled {
                    usage.record_violation();
                    return Err(ViolationType::BlockCount);
                }
                return Ok(QuotaStatus::HardLimitExceeded);
            } else if current + 1 > soft_limit {
                return Ok(QuotaStatus::SoftLimitExceeded);
            }
        }

        // Check write bandwidth
        if quota_config.max_write_bandwidth > 0 {
            let current = usage.bytes_written.load(Ordering::Relaxed);
            if current + data_size > quota_config.max_write_bandwidth {
                if quota_config.hard_limit_enabled {
                    usage.record_violation();
                    return Err(ViolationType::WriteBandwidth);
                }
                return Ok(QuotaStatus::HardLimitExceeded);
            }
        }

        Ok(QuotaStatus::Ok)
    }

    /// Check if a read operation is allowed
    pub fn check_read_quota(
        &self,
        tenant_id: &str,
        data_size: u64,
    ) -> std::result::Result<QuotaStatus, ViolationType> {
        let (enforcement_enabled, bandwidth_period) = {
            let config_guard = self.config.read();
            (
                config_guard.enforcement_enabled,
                config_guard.bandwidth_period,
            )
        };

        if !enforcement_enabled {
            return Ok(QuotaStatus::Ok);
        }

        let tenant = match self.tenants.get(tenant_id) {
            Some(t) => t,
            None => return Ok(QuotaStatus::Ok), // Allow reads for unknown tenants
        };

        let quota_config = tenant.config.read();
        let usage = &tenant.usage;

        // Reset bandwidth if period expired
        if usage.should_reset(bandwidth_period) {
            usage.reset_bandwidth();
        }

        // Check read bandwidth
        if quota_config.max_read_bandwidth > 0 {
            let current = usage.bytes_read.load(Ordering::Relaxed);
            if current + data_size > quota_config.max_read_bandwidth {
                if quota_config.hard_limit_enabled {
                    usage.record_violation();
                    return Err(ViolationType::ReadBandwidth);
                }
                return Ok(QuotaStatus::HardLimitExceeded);
            }
        }

        Ok(QuotaStatus::Ok)
    }

    /// Record a write operation
    pub fn record_write(&self, tenant_id: &str, cid: Cid, data_size: u64) {
        if let Some(tenant) = self.tenants.get(tenant_id) {
            tenant.usage.record_write(data_size);
            self.cid_map.insert(cid, (tenant_id.to_string(), data_size));
        }
    }

    /// Record a read operation
    pub fn record_read(&self, tenant_id: &str, data_size: u64) {
        if let Some(tenant) = self.tenants.get(tenant_id) {
            tenant.usage.record_read(data_size);
        }
    }

    /// Record a delete operation
    pub fn record_delete(&self, cid: &Cid) {
        if let Some((_, (tenant_id, data_size))) = self.cid_map.remove(cid) {
            if let Some(tenant) = self.tenants.get(&tenant_id) {
                tenant.usage.record_delete(data_size);
            }
        }
    }

    /// Get all tenants
    pub fn list_tenants(&self) -> Vec<TenantId> {
        self.tenants
            .iter()
            .map(|entry| entry.key().clone())
            .collect()
    }

    /// Get quota report for a tenant
    pub fn get_quota_report(&self, tenant_id: &str) -> Option<QuotaReport> {
        let tenant = self.tenants.get(tenant_id)?;
        let config = tenant.config.read().clone();
        let usage_snapshot = QuotaUsageSnapshot {
            bytes_used: tenant.usage.bytes_used.load(Ordering::Relaxed),
            blocks_count: tenant.usage.blocks_count.load(Ordering::Relaxed),
            bytes_read: tenant.usage.bytes_read.load(Ordering::Relaxed),
            bytes_written: tenant.usage.bytes_written.load(Ordering::Relaxed),
            violations: tenant.usage.violations.load(Ordering::Relaxed),
        };

        let storage_percent = if config.max_bytes > 0 {
            usage_snapshot.bytes_used as f64 / config.max_bytes as f64 * 100.0
        } else {
            0.0
        };

        let blocks_percent = if config.max_blocks > 0 {
            usage_snapshot.blocks_count as f64 / config.max_blocks as f64 * 100.0
        } else {
            0.0
        };

        Some(QuotaReport {
            tenant_id: tenant_id.to_string(),
            config,
            usage: usage_snapshot,
            storage_utilization_percent: storage_percent,
            blocks_utilization_percent: blocks_percent,
        })
    }
}

/// Snapshot of quota usage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuotaUsageSnapshot {
    pub bytes_used: u64,
    pub blocks_count: u64,
    pub bytes_read: u64,
    pub bytes_written: u64,
    pub violations: u64,
}

/// Quota report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuotaReport {
    pub tenant_id: TenantId,
    pub config: QuotaConfig,
    pub usage: QuotaUsageSnapshot,
    pub storage_utilization_percent: f64,
    pub blocks_utilization_percent: f64,
}

/// Quota-enforced block store
pub struct QuotaBlockStore<S: BlockStore> {
    inner: Arc<S>,
    quota_manager: Arc<QuotaManager>,
    tenant_id: TenantId,
}

impl<S: BlockStore> QuotaBlockStore<S> {
    /// Create a new quota-enforced block store
    pub fn new(inner: Arc<S>, quota_manager: Arc<QuotaManager>, tenant_id: TenantId) -> Self {
        Self {
            inner,
            quota_manager,
            tenant_id,
        }
    }

    /// Get the quota manager
    pub fn quota_manager(&self) -> &Arc<QuotaManager> {
        &self.quota_manager
    }

    /// Get the tenant ID
    pub fn tenant_id(&self) -> &str {
        &self.tenant_id
    }
}

#[async_trait]
impl<S: BlockStore + Send + Sync + 'static> BlockStore for QuotaBlockStore<S> {
    async fn get(&self, cid: &Cid) -> Result<Option<Block>> {
        let block_opt = self.inner.get(cid).await?;

        if let Some(ref block) = block_opt {
            // Check read quota
            match self
                .quota_manager
                .check_read_quota(&self.tenant_id, block.data().len() as u64)
            {
                Ok(QuotaStatus::Ok) => {
                    self.quota_manager
                        .record_read(&self.tenant_id, block.data().len() as u64);
                }
                Ok(QuotaStatus::SoftLimitExceeded) => {
                    warn!("Tenant {} exceeded soft read quota limit", self.tenant_id);
                    self.quota_manager
                        .record_read(&self.tenant_id, block.data().len() as u64);
                }
                Ok(QuotaStatus::HardLimitExceeded) | Err(_) => {
                    return Err(Error::InvalidInput(format!(
                        "Tenant {} exceeded read quota",
                        self.tenant_id
                    )))
                }
            }
        }

        Ok(block_opt)
    }

    async fn put(&self, block: &Block) -> Result<()> {
        // Check write quota before writing
        match self
            .quota_manager
            .check_write_quota(&self.tenant_id, block.data().len() as u64)
        {
            Ok(QuotaStatus::Ok) => {}
            Ok(QuotaStatus::SoftLimitExceeded) => {
                warn!(
                    "Tenant {} exceeded soft storage quota limit",
                    self.tenant_id
                );
            }
            Ok(QuotaStatus::HardLimitExceeded) | Err(_) => {
                return Err(Error::InvalidInput(format!(
                    "Tenant {} exceeded storage quota",
                    self.tenant_id
                )))
            }
        }

        self.inner.put(block).await?;
        self.quota_manager
            .record_write(&self.tenant_id, *block.cid(), block.data().len() as u64);
        Ok(())
    }

    async fn has(&self, cid: &Cid) -> Result<bool> {
        self.inner.has(cid).await
    }

    async fn delete(&self, cid: &Cid) -> Result<()> {
        self.inner.delete(cid).await?;
        self.quota_manager.record_delete(cid);
        Ok(())
    }

    fn list_cids(&self) -> Result<Vec<Cid>> {
        self.inner.list_cids()
    }

    fn len(&self) -> usize {
        self.inner.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::MemoryBlockStore;
    use bytes::Bytes;

    #[tokio::test]
    async fn test_quota_enforcement() {
        let config = QuotaManagerConfig {
            default_quota: QuotaConfig {
                max_bytes: 1000,
                max_blocks: 10,
                hard_limit_enabled: true,
                ..Default::default()
            },
            enforcement_enabled: true,
            ..Default::default()
        };

        let manager = Arc::new(QuotaManager::new(config));
        let store = Arc::new(MemoryBlockStore::new());
        let quota_store = QuotaBlockStore::new(store, manager.clone(), "tenant1".to_string());

        let data = vec![0u8; 100];
        let block = Block::new(Bytes::from(data)).unwrap();

        // Should succeed
        quota_store.put(&block).await.unwrap();

        // Check usage
        let usage = manager.get_usage("tenant1").unwrap();
        assert_eq!(usage.bytes_used, 100);
        assert_eq!(usage.blocks_count, 1);
    }

    #[tokio::test]
    async fn test_quota_exceeded() {
        let config = QuotaManagerConfig {
            default_quota: QuotaConfig {
                max_bytes: 50,
                hard_limit_enabled: true,
                ..Default::default()
            },
            enforcement_enabled: true,
            ..Default::default()
        };

        let manager = Arc::new(QuotaManager::new(config));
        let store = Arc::new(MemoryBlockStore::new());
        let quota_store = QuotaBlockStore::new(store, manager, "tenant1".to_string());

        let data = vec![0u8; 100];
        let block = Block::new(Bytes::from(data)).unwrap();

        // Should fail (exceeds quota)
        let result = quota_store.put(&block).await;
        assert!(result.is_err());
    }

    #[test]
    fn test_quota_report() {
        let manager = QuotaManager::new(QuotaManagerConfig::default());
        manager.set_quota(
            "tenant1".to_string(),
            QuotaConfig {
                max_bytes: 1000,
                max_blocks: 100,
                ..Default::default()
            },
        );

        let cid = cid::Cid::default();
        manager.record_write("tenant1", cid, 500);

        let report = manager.get_quota_report("tenant1").unwrap();
        assert_eq!(report.usage.bytes_used, 500);
        assert_eq!(report.storage_utilization_percent, 50.0);
    }
}
