//! Storage Quota Manager — per-namespace quota tracking and enforcement
//!
//! Tracks per-namespace storage quotas and enforces limits before write operations.
//! Supports hard limits (reject), soft limits (warn), and unlimited namespaces.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

use thiserror::Error;
use tracing::warn;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors that can occur during quota operations.
#[derive(Debug, Error)]
pub enum QuotaError {
    /// The requested write would exceed the hard limit for a namespace.
    #[error("quota exceeded for namespace '{namespace}': used {used} bytes, limit {limit} bytes")]
    QuotaExceeded {
        namespace: String,
        used: u64,
        limit: u64,
    },

    /// No quota entry exists for the given namespace.
    #[error("namespace not found: '{0}'")]
    NamespaceNotFound(String),

    /// A quota limit of zero is invalid (must be > 0).
    #[error("invalid limit: limit must be greater than 0")]
    InvalidLimit,
}

// ---------------------------------------------------------------------------
// QuotaPolicy
// ---------------------------------------------------------------------------

/// Enforcement policy applied to a namespace quota.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QuotaPolicy {
    /// Reject writes that would cause the namespace to exceed its limit.
    HardLimit,
    /// Allow writes that would exceed the limit, but emit a tracing warning.
    SoftLimit,
    /// No enforcement — writes are always allowed regardless of usage.
    NoLimit,
}

impl QuotaPolicy {
    /// Returns a human-readable string representation used in [`QuotaStats`].
    pub fn as_str(&self) -> &'static str {
        match self {
            QuotaPolicy::HardLimit => "hard",
            QuotaPolicy::SoftLimit => "soft",
            QuotaPolicy::NoLimit => "none",
        }
    }
}

// ---------------------------------------------------------------------------
// NamespaceQuota
// ---------------------------------------------------------------------------

/// Quota state for a single namespace.
pub struct NamespaceQuota {
    /// Identifier for this namespace.
    pub namespace: String,
    /// Maximum bytes allowed (only meaningful for HardLimit / SoftLimit).
    pub limit_bytes: u64,
    /// Currently consumed bytes (atomic, shared across threads).
    pub used_bytes: Arc<AtomicU64>,
    /// Enforcement policy.
    pub policy: QuotaPolicy,
    /// Number of stored blocks (atomic, shared across threads).
    pub block_count: Arc<AtomicU64>,
}

impl NamespaceQuota {
    /// Create a new `NamespaceQuota`.
    pub fn new(namespace: String, limit_bytes: u64, policy: QuotaPolicy) -> Self {
        Self {
            namespace,
            limit_bytes,
            used_bytes: Arc::new(AtomicU64::new(0)),
            policy,
            block_count: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Returns how many bytes are still available before the limit is reached.
    ///
    /// Uses saturating subtraction, so returns `0` when already over limit.
    pub fn available_bytes(&self) -> u64 {
        self.limit_bytes
            .saturating_sub(self.used_bytes.load(Ordering::Relaxed))
    }

    /// Returns the fraction of the limit currently used (range `[0.0, ∞)`).
    ///
    /// Returns `0.0` when `limit_bytes` is `0` to avoid division by zero.
    pub fn usage_ratio(&self) -> f64 {
        if self.limit_bytes == 0 {
            return 0.0;
        }
        self.used_bytes.load(Ordering::Relaxed) as f64 / self.limit_bytes as f64
    }

    /// Returns `true` when `usage_ratio()` exceeds `threshold`.
    pub fn is_over_soft_threshold(&self, threshold: f64) -> bool {
        self.usage_ratio() > threshold
    }
}

// ---------------------------------------------------------------------------
// QuotaStats  (snapshot — no atomics)
// ---------------------------------------------------------------------------

/// A plain-data snapshot of quota state for a namespace.
#[derive(Debug, Clone)]
pub struct QuotaStats {
    /// Namespace identifier.
    pub namespace: String,
    /// Maximum bytes allowed.
    pub limit_bytes: u64,
    /// Currently consumed bytes.
    pub used_bytes: u64,
    /// Bytes remaining before the limit.
    pub available_bytes: u64,
    /// Number of stored blocks.
    pub block_count: u64,
    /// Usage as a percentage (`used_bytes / limit_bytes * 100`).
    pub usage_pct: f64,
    /// Policy string: `"hard"`, `"soft"`, or `"none"`.
    pub policy: String,
}

impl QuotaStats {
    fn from_quota(q: &NamespaceQuota) -> Self {
        let used = q.used_bytes.load(Ordering::Relaxed);
        let limit = q.limit_bytes;
        let available = limit.saturating_sub(used);
        let usage_pct = if limit == 0 {
            0.0
        } else {
            used as f64 / limit as f64 * 100.0
        };
        Self {
            namespace: q.namespace.clone(),
            limit_bytes: limit,
            used_bytes: used,
            available_bytes: available,
            block_count: q.block_count.load(Ordering::Relaxed),
            usage_pct,
            policy: q.policy.as_str().to_owned(),
        }
    }
}

// ---------------------------------------------------------------------------
// StorageQuotaManager
// ---------------------------------------------------------------------------

/// Thread-safe manager that tracks per-namespace storage quotas.
///
/// # Example
/// ```rust
/// use ipfrs_storage::quota_manager::{StorageQuotaManager, QuotaPolicy};
///
/// let mgr = StorageQuotaManager::new(0.8);
/// mgr.register_namespace("ns1".to_owned(), 1024, QuotaPolicy::HardLimit).unwrap();
/// mgr.check_write("ns1", 512).unwrap();
/// mgr.record_write("ns1", 512).unwrap();
/// let stats = mgr.stats("ns1").unwrap();
/// assert_eq!(stats.used_bytes, 512);
/// ```
pub struct StorageQuotaManager {
    quotas: RwLock<HashMap<String, NamespaceQuota>>,
    soft_threshold: f64,
}

impl StorageQuotaManager {
    /// Create a new `StorageQuotaManager`.
    ///
    /// `soft_threshold` is a ratio in `[0.0, 1.0]` — namespaces whose usage
    /// exceeds this fraction of their limit are considered "over threshold".
    /// The recommended default is `0.8` (80 %).
    pub fn new(soft_threshold: f64) -> Self {
        Self {
            quotas: RwLock::new(HashMap::new()),
            soft_threshold,
        }
    }

    /// Register a new namespace with the given limit and policy.
    ///
    /// Returns [`QuotaError::InvalidLimit`] if `limit_bytes` is `0` and
    /// the policy is not [`QuotaPolicy::NoLimit`].
    pub fn register_namespace(
        &self,
        namespace: String,
        limit_bytes: u64,
        policy: QuotaPolicy,
    ) -> Result<(), QuotaError> {
        if limit_bytes == 0 && policy != QuotaPolicy::NoLimit {
            return Err(QuotaError::InvalidLimit);
        }
        let quota = NamespaceQuota::new(namespace.clone(), limit_bytes, policy);
        let mut map = self.quotas.write().unwrap_or_else(|e| e.into_inner());
        map.insert(namespace, quota);
        Ok(())
    }

    /// Check whether a write of `size_bytes` is permitted for `namespace`.
    ///
    /// - [`QuotaPolicy::HardLimit`]: returns `Err(QuotaExceeded)` when
    ///   `used + size > limit`.
    /// - [`QuotaPolicy::SoftLimit`]: logs a warning when the limit would be
    ///   exceeded, but still returns `Ok(())`.
    /// - [`QuotaPolicy::NoLimit`]: always returns `Ok(())`.
    ///
    /// Returns [`QuotaError::NamespaceNotFound`] for unknown namespaces.
    pub fn check_write(&self, namespace: &str, size_bytes: u64) -> Result<(), QuotaError> {
        let map = self.quotas.read().unwrap_or_else(|e| e.into_inner());
        let quota = map
            .get(namespace)
            .ok_or_else(|| QuotaError::NamespaceNotFound(namespace.to_owned()))?;

        match quota.policy {
            QuotaPolicy::NoLimit => Ok(()),
            QuotaPolicy::HardLimit => {
                let used = quota.used_bytes.load(Ordering::Relaxed);
                let projected = used.saturating_add(size_bytes);
                if projected > quota.limit_bytes {
                    Err(QuotaError::QuotaExceeded {
                        namespace: namespace.to_owned(),
                        used,
                        limit: quota.limit_bytes,
                    })
                } else {
                    Ok(())
                }
            }
            QuotaPolicy::SoftLimit => {
                let used = quota.used_bytes.load(Ordering::Relaxed);
                let projected = used.saturating_add(size_bytes);
                if projected > quota.limit_bytes {
                    warn!(
                        namespace = %namespace,
                        used = used,
                        size = size_bytes,
                        limit = quota.limit_bytes,
                        "soft quota exceeded — write allowed but usage is over limit"
                    );
                }
                Ok(())
            }
        }
    }

    /// Record that `size_bytes` have been successfully written to `namespace`.
    ///
    /// Atomically increments `used_bytes` and `block_count`.
    pub fn record_write(&self, namespace: &str, size_bytes: u64) -> Result<(), QuotaError> {
        let map = self.quotas.read().unwrap_or_else(|e| e.into_inner());
        let quota = map
            .get(namespace)
            .ok_or_else(|| QuotaError::NamespaceNotFound(namespace.to_owned()))?;
        quota.used_bytes.fetch_add(size_bytes, Ordering::Relaxed);
        quota.block_count.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Record that `size_bytes` have been deleted from `namespace`.
    ///
    /// Uses saturating subtraction so that underflow is safe.
    /// `block_count` is also decremented using saturating subtraction.
    pub fn record_delete(&self, namespace: &str, size_bytes: u64) {
        let map = self.quotas.read().unwrap_or_else(|e| e.into_inner());
        if let Some(quota) = map.get(namespace) {
            // Saturating subtract for used_bytes
            let prev =
                quota
                    .used_bytes
                    .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                        Some(current.saturating_sub(size_bytes))
                    });
            // fetch_update only errors if closure returns None, which ours never does.
            let _ = prev;

            // Saturating subtract for block_count
            let _ =
                quota
                    .block_count
                    .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                        Some(current.saturating_sub(1))
                    });
        }
    }

    /// Return a snapshot of quota statistics for `namespace`.
    pub fn stats(&self, namespace: &str) -> Result<QuotaStats, QuotaError> {
        let map = self.quotas.read().unwrap_or_else(|e| e.into_inner());
        map.get(namespace)
            .map(QuotaStats::from_quota)
            .ok_or_else(|| QuotaError::NamespaceNotFound(namespace.to_owned()))
    }

    /// Return snapshots for all registered namespaces, sorted by namespace name.
    pub fn all_stats(&self) -> Vec<QuotaStats> {
        let map = self.quotas.read().unwrap_or_else(|e| e.into_inner());
        let mut stats: Vec<QuotaStats> = map.values().map(QuotaStats::from_quota).collect();
        stats.sort_by(|a, b| a.namespace.cmp(&b.namespace));
        stats
    }

    /// Return namespaces whose usage ratio exceeds `self.soft_threshold`.
    pub fn namespaces_over_threshold(&self) -> Vec<String> {
        let map = self.quotas.read().unwrap_or_else(|e| e.into_inner());
        let mut over: Vec<String> = map
            .values()
            .filter(|q| q.is_over_soft_threshold(self.soft_threshold))
            .map(|q| q.namespace.clone())
            .collect();
        over.sort();
        over
    }

    /// Zero out `used_bytes` and `block_count` for `namespace`.
    pub fn reset_namespace(&self, namespace: &str) -> Result<(), QuotaError> {
        let map = self.quotas.read().unwrap_or_else(|e| e.into_inner());
        let quota = map
            .get(namespace)
            .ok_or_else(|| QuotaError::NamespaceNotFound(namespace.to_owned()))?;
        quota.used_bytes.store(0, Ordering::Relaxed);
        quota.block_count.store(0, Ordering::Relaxed);
        Ok(())
    }

    /// Remove `namespace` from the manager.
    ///
    /// Returns `true` if the namespace existed, `false` otherwise.
    pub fn remove_namespace(&self, namespace: &str) -> bool {
        let mut map = self.quotas.write().unwrap_or_else(|e| e.into_inner());
        map.remove(namespace).is_some()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_mgr() -> StorageQuotaManager {
        StorageQuotaManager::new(0.8)
    }

    // 1. new() with custom soft_threshold
    #[test]
    fn test_new_custom_soft_threshold() {
        let mgr = StorageQuotaManager::new(0.5);
        assert_eq!(mgr.soft_threshold, 0.5);
    }

    // 2. register_namespace with HardLimit
    #[test]
    fn test_register_hard_limit() {
        let mgr = make_mgr();
        mgr.register_namespace("ns".to_owned(), 1024, QuotaPolicy::HardLimit)
            .expect("should register");
        let stats = mgr.stats("ns").expect("should have stats");
        assert_eq!(stats.limit_bytes, 1024);
        assert_eq!(stats.policy, "hard");
    }

    // 3. register_namespace with SoftLimit
    #[test]
    fn test_register_soft_limit() {
        let mgr = make_mgr();
        mgr.register_namespace("ns".to_owned(), 2048, QuotaPolicy::SoftLimit)
            .expect("should register");
        let stats = mgr.stats("ns").expect("should have stats");
        assert_eq!(stats.policy, "soft");
    }

    // 3b. register_namespace with NoLimit
    #[test]
    fn test_register_no_limit() {
        let mgr = make_mgr();
        mgr.register_namespace("ns".to_owned(), 0, QuotaPolicy::NoLimit)
            .expect("NoLimit with 0 bytes should succeed");
        let stats = mgr.stats("ns").expect("should have stats");
        assert_eq!(stats.policy, "none");
    }

    // 4. check_write under limit returns Ok
    #[test]
    fn test_check_write_under_limit_ok() {
        let mgr = make_mgr();
        mgr.register_namespace("ns".to_owned(), 1000, QuotaPolicy::HardLimit)
            .unwrap();
        mgr.check_write("ns", 500).expect("should be under limit");
    }

    // 5. check_write over limit HardLimit returns QuotaExceeded
    #[test]
    fn test_check_write_over_hard_limit_err() {
        let mgr = make_mgr();
        mgr.register_namespace("ns".to_owned(), 100, QuotaPolicy::HardLimit)
            .unwrap();
        mgr.record_write("ns", 90).unwrap(); // used = 90
        let err = mgr
            .check_write("ns", 20)
            .expect_err("should exceed hard limit");
        matches!(err, QuotaError::QuotaExceeded { .. });
    }

    // 6. check_write over limit SoftLimit returns Ok
    #[test]
    fn test_check_write_over_soft_limit_ok() {
        let mgr = make_mgr();
        mgr.register_namespace("ns".to_owned(), 100, QuotaPolicy::SoftLimit)
            .unwrap();
        mgr.record_write("ns", 90).unwrap(); // used = 90
        mgr.check_write("ns", 50)
            .expect("SoftLimit should return Ok even over limit");
    }

    // 7. check_write NoLimit always Ok
    #[test]
    fn test_check_write_no_limit_always_ok() {
        let mgr = make_mgr();
        mgr.register_namespace("ns".to_owned(), 0, QuotaPolicy::NoLimit)
            .unwrap();
        mgr.check_write("ns", u64::MAX)
            .expect("NoLimit should always be Ok");
    }

    // 8. check_write unknown namespace returns NamespaceNotFound
    #[test]
    fn test_check_write_unknown_namespace() {
        let mgr = make_mgr();
        let err = mgr
            .check_write("missing", 100)
            .expect_err("unknown namespace should error");
        matches!(err, QuotaError::NamespaceNotFound(_));
    }

    // 9. record_write increments used_bytes and block_count
    #[test]
    fn test_record_write_increments() {
        let mgr = make_mgr();
        mgr.register_namespace("ns".to_owned(), 10000, QuotaPolicy::HardLimit)
            .unwrap();
        mgr.record_write("ns", 300).unwrap();
        mgr.record_write("ns", 200).unwrap();
        let stats = mgr.stats("ns").unwrap();
        assert_eq!(stats.used_bytes, 500);
        assert_eq!(stats.block_count, 2);
    }

    // 10. record_delete decrements used_bytes and block_count
    #[test]
    fn test_record_delete_decrements() {
        let mgr = make_mgr();
        mgr.register_namespace("ns".to_owned(), 10000, QuotaPolicy::HardLimit)
            .unwrap();
        mgr.record_write("ns", 1000).unwrap();
        mgr.record_write("ns", 500).unwrap();
        mgr.record_delete("ns", 300);
        let stats = mgr.stats("ns").unwrap();
        assert_eq!(stats.used_bytes, 1200);
        assert_eq!(stats.block_count, 1);
    }

    // 11. record_delete underflow is safe (saturating)
    #[test]
    fn test_record_delete_underflow_safe() {
        let mgr = make_mgr();
        mgr.register_namespace("ns".to_owned(), 10000, QuotaPolicy::HardLimit)
            .unwrap();
        // Delete more than recorded — should not panic, should saturate at 0
        mgr.record_delete("ns", 999_999);
        let stats = mgr.stats("ns").unwrap();
        assert_eq!(stats.used_bytes, 0);
        assert_eq!(stats.block_count, 0);
    }

    // 12. stats() returns correct snapshot
    #[test]
    fn test_stats_correct_snapshot() {
        let mgr = make_mgr();
        mgr.register_namespace("ns".to_owned(), 1000, QuotaPolicy::SoftLimit)
            .unwrap();
        mgr.record_write("ns", 400).unwrap();
        let s = mgr.stats("ns").unwrap();
        assert_eq!(s.namespace, "ns");
        assert_eq!(s.limit_bytes, 1000);
        assert_eq!(s.used_bytes, 400);
        assert_eq!(s.available_bytes, 600);
        assert_eq!(s.block_count, 1);
        assert!((s.usage_pct - 40.0).abs() < 1e-9);
        assert_eq!(s.policy, "soft");
    }

    // 13. all_stats() returns sorted list
    #[test]
    fn test_all_stats_sorted() {
        let mgr = make_mgr();
        mgr.register_namespace("zebra".to_owned(), 500, QuotaPolicy::HardLimit)
            .unwrap();
        mgr.register_namespace("alpha".to_owned(), 500, QuotaPolicy::HardLimit)
            .unwrap();
        mgr.register_namespace("mango".to_owned(), 500, QuotaPolicy::HardLimit)
            .unwrap();
        let all = mgr.all_stats();
        assert_eq!(all.len(), 3);
        assert_eq!(all[0].namespace, "alpha");
        assert_eq!(all[1].namespace, "mango");
        assert_eq!(all[2].namespace, "zebra");
    }

    // 14. namespaces_over_threshold filtered correctly
    #[test]
    fn test_namespaces_over_threshold() {
        let mgr = StorageQuotaManager::new(0.8);
        mgr.register_namespace("low".to_owned(), 1000, QuotaPolicy::HardLimit)
            .unwrap();
        mgr.register_namespace("high".to_owned(), 1000, QuotaPolicy::SoftLimit)
            .unwrap();
        // Push "high" above 80 %
        mgr.record_write("high", 900).unwrap();
        // Keep "low" well under
        mgr.record_write("low", 100).unwrap();
        let over = mgr.namespaces_over_threshold();
        assert_eq!(over, vec!["high".to_owned()]);
    }

    // 15. reset_namespace zeroes counters
    #[test]
    fn test_reset_namespace() {
        let mgr = make_mgr();
        mgr.register_namespace("ns".to_owned(), 10000, QuotaPolicy::HardLimit)
            .unwrap();
        mgr.record_write("ns", 5000).unwrap();
        mgr.reset_namespace("ns").unwrap();
        let stats = mgr.stats("ns").unwrap();
        assert_eq!(stats.used_bytes, 0);
        assert_eq!(stats.block_count, 0);
    }

    // 16. remove_namespace removes entry
    #[test]
    fn test_remove_namespace() {
        let mgr = make_mgr();
        mgr.register_namespace("ns".to_owned(), 1000, QuotaPolicy::HardLimit)
            .unwrap();
        assert!(mgr.remove_namespace("ns"));
        assert!(!mgr.remove_namespace("ns")); // second call returns false
        let err = mgr.stats("ns").expect_err("should be gone");
        matches!(err, QuotaError::NamespaceNotFound(_));
    }

    // 17. InvalidLimit for limit=0 with HardLimit
    #[test]
    fn test_invalid_limit_zero() {
        let mgr = make_mgr();
        let err = mgr
            .register_namespace("ns".to_owned(), 0, QuotaPolicy::HardLimit)
            .expect_err("limit=0 with HardLimit should fail");
        matches!(err, QuotaError::InvalidLimit);
    }

    // 17b. InvalidLimit for limit=0 with SoftLimit
    #[test]
    fn test_invalid_limit_zero_soft() {
        let mgr = make_mgr();
        let err = mgr
            .register_namespace("ns".to_owned(), 0, QuotaPolicy::SoftLimit)
            .expect_err("limit=0 with SoftLimit should fail");
        matches!(err, QuotaError::InvalidLimit);
    }
}
