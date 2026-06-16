//! Storage Quota Enforcer — per-namespace soft/hard limit enforcement
//!
//! Provides per-namespace storage quota management with:
//! - Soft limits: warn but allow writes (QuotaLevel::Warning)
//! - Hard limits: block writes when exceeded (QuotaLevel::Exceeded)
//! - Usage tracking with byte-level and block-count accounting
//! - Utilization metrics and over-quota namespace detection
//! - Statistics for monitoring check/rejection counts

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// QuotaLevel
// ---------------------------------------------------------------------------

/// Describes the current quota state for a namespace.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuotaLevel {
    /// Usage is below the soft limit — everything is fine.
    Ok,
    /// Usage is between the soft limit (inclusive) and hard limit — a warning.
    Warning,
    /// Usage is at or above the hard limit — writes should be rejected.
    Exceeded,
}

// ---------------------------------------------------------------------------
// NamespaceQuota
// ---------------------------------------------------------------------------

/// Per-namespace quota configuration and current usage.
#[derive(Debug, Clone)]
pub struct NamespaceQuota {
    /// Namespace identifier.
    pub namespace: String,
    /// Byte threshold for soft limit warnings.
    pub soft_limit_bytes: u64,
    /// Byte threshold for hard limit enforcement.
    pub hard_limit_bytes: u64,
    /// Current bytes consumed in this namespace.
    pub current_usage_bytes: u64,
    /// Number of blocks currently stored in this namespace.
    pub block_count: u64,
}

impl NamespaceQuota {
    /// Compute the [`QuotaLevel`] for the current usage.
    fn level(&self) -> QuotaLevel {
        Self::level_for(
            self.current_usage_bytes,
            self.soft_limit_bytes,
            self.hard_limit_bytes,
        )
    }

    /// Compute the [`QuotaLevel`] for a hypothetical usage value.
    fn level_for(usage: u64, soft: u64, hard: u64) -> QuotaLevel {
        if usage >= hard {
            QuotaLevel::Exceeded
        } else if usage >= soft {
            QuotaLevel::Warning
        } else {
            QuotaLevel::Ok
        }
    }
}

// ---------------------------------------------------------------------------
// QuotaEnforcerConfig
// ---------------------------------------------------------------------------

/// Configuration for [`StorageQuotaEnforcer`] default limits.
#[derive(Debug, Clone)]
pub struct QuotaEnforcerConfig {
    /// Default soft limit in bytes (used when `None` is passed to
    /// [`StorageQuotaEnforcer::register_namespace`]).
    pub default_soft_limit: u64,
    /// Default hard limit in bytes (used when `None` is passed to
    /// [`StorageQuotaEnforcer::register_namespace`]).
    pub default_hard_limit: u64,
}

impl Default for QuotaEnforcerConfig {
    fn default() -> Self {
        Self {
            default_soft_limit: 1_000_000_000, // 1 GB
            default_hard_limit: 2_000_000_000, // 2 GB
        }
    }
}

// ---------------------------------------------------------------------------
// QuotaEnforcerStats
// ---------------------------------------------------------------------------

/// Aggregate statistics for the quota enforcer.
#[derive(Debug, Clone)]
pub struct QuotaEnforcerStats {
    /// Number of registered namespaces.
    pub namespace_count: usize,
    /// Total number of quota checks performed.
    pub total_checks: u64,
    /// Total number of rejections (quota exceeded).
    pub total_rejections: u64,
    /// Number of namespaces currently at [`QuotaLevel::Exceeded`].
    pub over_quota_count: usize,
}

// ---------------------------------------------------------------------------
// StorageQuotaEnforcer
// ---------------------------------------------------------------------------

/// Enforces per-namespace storage quotas with soft and hard limits.
///
/// All operations are synchronous and single-threaded. Callers that need
/// concurrent access should wrap the enforcer in a `Mutex` or `RwLock`.
pub struct StorageQuotaEnforcer {
    config: QuotaEnforcerConfig,
    namespaces: HashMap<String, NamespaceQuota>,
    total_checks: u64,
    total_rejections: u64,
}

impl StorageQuotaEnforcer {
    /// Creates a new enforcer with the given configuration.
    pub fn new(config: QuotaEnforcerConfig) -> Self {
        Self {
            config,
            namespaces: HashMap::new(),
            total_checks: 0,
            total_rejections: 0,
        }
    }

    /// Registers a namespace with optional soft/hard limits.
    ///
    /// When `soft_limit` or `hard_limit` is `None`, the corresponding default
    /// from [`QuotaEnforcerConfig`] is used. If the namespace already exists,
    /// its limits are updated while preserving current usage.
    pub fn register_namespace(
        &mut self,
        namespace: &str,
        soft_limit: Option<u64>,
        hard_limit: Option<u64>,
    ) {
        let soft = soft_limit.unwrap_or(self.config.default_soft_limit);
        let hard = hard_limit.unwrap_or(self.config.default_hard_limit);

        let entry = self
            .namespaces
            .entry(namespace.to_owned())
            .or_insert_with(|| NamespaceQuota {
                namespace: namespace.to_owned(),
                soft_limit_bytes: soft,
                hard_limit_bytes: hard,
                current_usage_bytes: 0,
                block_count: 0,
            });
        // Always update limits (even if namespace already existed).
        entry.soft_limit_bytes = soft;
        entry.hard_limit_bytes = hard;
    }

    /// Checks the quota level that would result if `additional_bytes` were
    /// added to `namespace`.
    ///
    /// This is a read-only check — it does **not** modify usage. Returns an
    /// error if the namespace is not registered.
    pub fn check_quota(
        &mut self,
        namespace: &str,
        additional_bytes: u64,
    ) -> Result<QuotaLevel, String> {
        self.total_checks = self.total_checks.saturating_add(1);

        let quota = self
            .namespaces
            .get(namespace)
            .ok_or_else(|| format!("unknown namespace: {namespace}"))?;

        let hypothetical = quota.current_usage_bytes.saturating_add(additional_bytes);
        let level =
            NamespaceQuota::level_for(hypothetical, quota.soft_limit_bytes, quota.hard_limit_bytes);

        if level == QuotaLevel::Exceeded {
            self.total_rejections = self.total_rejections.saturating_add(1);
        }

        Ok(level)
    }

    /// Records `bytes` of usage in `namespace` and returns the new quota level.
    ///
    /// Returns an error if the namespace is not registered.
    pub fn record_usage(&mut self, namespace: &str, bytes: u64) -> Result<QuotaLevel, String> {
        let quota = self
            .namespaces
            .get_mut(namespace)
            .ok_or_else(|| format!("unknown namespace: {namespace}"))?;

        quota.current_usage_bytes = quota.current_usage_bytes.saturating_add(bytes);
        quota.block_count = quota.block_count.saturating_add(1);

        Ok(quota.level())
    }

    /// Releases `bytes` of usage from `namespace` using saturating subtraction.
    ///
    /// Returns an error if the namespace is not registered.
    pub fn release_usage(&mut self, namespace: &str, bytes: u64) -> Result<(), String> {
        let quota = self
            .namespaces
            .get_mut(namespace)
            .ok_or_else(|| format!("unknown namespace: {namespace}"))?;

        quota.current_usage_bytes = quota.current_usage_bytes.saturating_sub(bytes);
        quota.block_count = quota.block_count.saturating_sub(1);

        Ok(())
    }

    /// Returns the current [`QuotaLevel`] for `namespace`, or `None` if it is
    /// not registered.
    pub fn get_level(&self, namespace: &str) -> Option<QuotaLevel> {
        self.namespaces.get(namespace).map(|q| q.level())
    }

    /// Returns a reference to the [`NamespaceQuota`] for `namespace`, or `None`
    /// if it is not registered.
    pub fn get_quota(&self, namespace: &str) -> Option<&NamespaceQuota> {
        self.namespaces.get(namespace)
    }

    /// Returns the utilization ratio (`current_usage / hard_limit`) for
    /// `namespace`, or `None` if it is not registered.
    ///
    /// When `hard_limit_bytes` is zero, returns `f64::INFINITY`.
    pub fn utilization(&self, namespace: &str) -> Option<f64> {
        self.namespaces.get(namespace).map(|q| {
            if q.hard_limit_bytes == 0 {
                f64::INFINITY
            } else {
                q.current_usage_bytes as f64 / q.hard_limit_bytes as f64
            }
        })
    }

    /// Returns all namespaces currently at [`QuotaLevel::Exceeded`].
    pub fn over_quota_namespaces(&self) -> Vec<&NamespaceQuota> {
        self.namespaces
            .values()
            .filter(|q| q.level() == QuotaLevel::Exceeded)
            .collect()
    }

    /// Returns aggregate statistics for the enforcer.
    pub fn stats(&self) -> QuotaEnforcerStats {
        let over_quota_count = self
            .namespaces
            .values()
            .filter(|q| q.level() == QuotaLevel::Exceeded)
            .count();

        QuotaEnforcerStats {
            namespace_count: self.namespaces.len(),
            total_checks: self.total_checks,
            total_rejections: self.total_rejections,
            over_quota_count,
        }
    }
}

impl Default for StorageQuotaEnforcer {
    fn default() -> Self {
        Self::new(QuotaEnforcerConfig::default())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> QuotaEnforcerConfig {
        QuotaEnforcerConfig::default()
    }

    fn small_config() -> QuotaEnforcerConfig {
        QuotaEnforcerConfig {
            default_soft_limit: 500,
            default_hard_limit: 1000,
        }
    }

    // --- register_namespace with defaults ---

    #[test]
    fn register_with_defaults_uses_config() {
        let mut e = StorageQuotaEnforcer::new(default_config());
        e.register_namespace("ns1", None, None);
        let q = e.get_quota("ns1").expect("namespace should exist");
        assert_eq!(q.soft_limit_bytes, 1_000_000_000);
        assert_eq!(q.hard_limit_bytes, 2_000_000_000);
        assert_eq!(q.current_usage_bytes, 0);
        assert_eq!(q.block_count, 0);
    }

    #[test]
    fn register_with_custom_soft() {
        let mut e = StorageQuotaEnforcer::new(default_config());
        e.register_namespace("ns2", Some(100), None);
        let q = e.get_quota("ns2").expect("namespace should exist");
        assert_eq!(q.soft_limit_bytes, 100);
        assert_eq!(q.hard_limit_bytes, 2_000_000_000);
    }

    #[test]
    fn register_with_custom_hard() {
        let mut e = StorageQuotaEnforcer::new(default_config());
        e.register_namespace("ns3", None, Some(500));
        let q = e.get_quota("ns3").expect("namespace should exist");
        assert_eq!(q.soft_limit_bytes, 1_000_000_000);
        assert_eq!(q.hard_limit_bytes, 500);
    }

    #[test]
    fn register_with_both_custom() {
        let mut e = StorageQuotaEnforcer::new(default_config());
        e.register_namespace("ns4", Some(200), Some(400));
        let q = e.get_quota("ns4").expect("namespace should exist");
        assert_eq!(q.soft_limit_bytes, 200);
        assert_eq!(q.hard_limit_bytes, 400);
    }

    #[test]
    fn register_updates_limits_preserves_usage() {
        let mut e = StorageQuotaEnforcer::new(small_config());
        e.register_namespace("ns5", Some(100), Some(1000));
        e.record_usage("ns5", 50).expect("record should succeed");
        // Re-register with new limits.
        e.register_namespace("ns5", Some(200), Some(2000));
        let q = e.get_quota("ns5").expect("namespace should exist");
        assert_eq!(q.soft_limit_bytes, 200);
        assert_eq!(q.hard_limit_bytes, 2000);
        assert_eq!(q.current_usage_bytes, 50, "usage should be preserved");
        assert_eq!(q.block_count, 1, "block count should be preserved");
    }

    // --- check_quota levels ---

    #[test]
    fn check_quota_ok() {
        let mut e = StorageQuotaEnforcer::new(small_config());
        e.register_namespace("cq", Some(500), Some(1000));
        let level = e.check_quota("cq", 100).expect("should succeed");
        assert_eq!(level, QuotaLevel::Ok);
    }

    #[test]
    fn check_quota_warning() {
        let mut e = StorageQuotaEnforcer::new(small_config());
        e.register_namespace("cq", Some(500), Some(1000));
        let level = e.check_quota("cq", 600).expect("should succeed");
        assert_eq!(level, QuotaLevel::Warning);
    }

    #[test]
    fn check_quota_exceeded() {
        let mut e = StorageQuotaEnforcer::new(small_config());
        e.register_namespace("cq", Some(500), Some(1000));
        let level = e.check_quota("cq", 1000).expect("should succeed");
        assert_eq!(level, QuotaLevel::Exceeded);
    }

    #[test]
    fn check_quota_exactly_at_soft_is_warning() {
        let mut e = StorageQuotaEnforcer::new(small_config());
        e.register_namespace("cq", Some(500), Some(1000));
        let level = e.check_quota("cq", 500).expect("should succeed");
        assert_eq!(
            level,
            QuotaLevel::Warning,
            "exactly at soft limit is Warning"
        );
    }

    #[test]
    fn check_quota_exactly_at_hard_is_exceeded() {
        let mut e = StorageQuotaEnforcer::new(small_config());
        e.register_namespace("cq", Some(500), Some(1000));
        let level = e.check_quota("cq", 1000).expect("should succeed");
        assert_eq!(
            level,
            QuotaLevel::Exceeded,
            "exactly at hard limit is Exceeded"
        );
    }

    #[test]
    fn check_quota_does_not_modify_usage() {
        let mut e = StorageQuotaEnforcer::new(small_config());
        e.register_namespace("cq", Some(500), Some(1000));
        let _ = e.check_quota("cq", 999);
        let q = e.get_quota("cq").expect("namespace should exist");
        assert_eq!(q.current_usage_bytes, 0, "check_quota should be read-only");
    }

    #[test]
    fn check_quota_unknown_namespace_error() {
        let mut e = StorageQuotaEnforcer::new(small_config());
        let result = e.check_quota("nonexistent", 10);
        assert!(result.is_err());
        assert!(result
            .expect_err("should be error")
            .contains("unknown namespace"));
    }

    // --- record_usage transitions ---

    #[test]
    fn record_usage_ok_level() {
        let mut e = StorageQuotaEnforcer::new(small_config());
        e.register_namespace("ru", Some(500), Some(1000));
        let level = e.record_usage("ru", 100).expect("should succeed");
        assert_eq!(level, QuotaLevel::Ok);
        let q = e.get_quota("ru").expect("namespace should exist");
        assert_eq!(q.current_usage_bytes, 100);
        assert_eq!(q.block_count, 1);
    }

    #[test]
    fn record_usage_transitions_to_warning() {
        let mut e = StorageQuotaEnforcer::new(small_config());
        e.register_namespace("ru", Some(500), Some(1000));
        e.record_usage("ru", 400).expect("ok");
        let level = e.record_usage("ru", 200).expect("should succeed");
        assert_eq!(level, QuotaLevel::Warning);
        assert_eq!(e.get_quota("ru").expect("exists").current_usage_bytes, 600);
    }

    #[test]
    fn record_usage_transitions_to_exceeded() {
        let mut e = StorageQuotaEnforcer::new(small_config());
        e.register_namespace("ru", Some(500), Some(1000));
        e.record_usage("ru", 900).expect("ok");
        let level = e.record_usage("ru", 200).expect("should succeed");
        assert_eq!(level, QuotaLevel::Exceeded);
        assert_eq!(e.get_quota("ru").expect("exists").current_usage_bytes, 1100);
    }

    #[test]
    fn record_usage_unknown_namespace_error() {
        let mut e = StorageQuotaEnforcer::new(small_config());
        let result = e.record_usage("ghost", 10);
        assert!(result.is_err());
    }

    #[test]
    fn record_usage_increments_block_count() {
        let mut e = StorageQuotaEnforcer::new(small_config());
        e.register_namespace("bc", Some(500), Some(1000));
        e.record_usage("bc", 10).expect("ok");
        e.record_usage("bc", 20).expect("ok");
        e.record_usage("bc", 30).expect("ok");
        let q = e.get_quota("bc").expect("exists");
        assert_eq!(q.block_count, 3);
        assert_eq!(q.current_usage_bytes, 60);
    }

    // --- release_usage ---

    #[test]
    fn release_usage_subtracts_bytes() {
        let mut e = StorageQuotaEnforcer::new(small_config());
        e.register_namespace("rel", Some(500), Some(1000));
        e.record_usage("rel", 300).expect("ok");
        e.release_usage("rel", 100).expect("should succeed");
        let q = e.get_quota("rel").expect("exists");
        assert_eq!(q.current_usage_bytes, 200);
        assert_eq!(q.block_count, 0, "block count saturates to 0");
    }

    #[test]
    fn release_usage_saturating() {
        let mut e = StorageQuotaEnforcer::new(small_config());
        e.register_namespace("rel2", Some(500), Some(1000));
        e.record_usage("rel2", 50).expect("ok");
        e.release_usage("rel2", 9999).expect("should succeed");
        let q = e.get_quota("rel2").expect("exists");
        assert_eq!(q.current_usage_bytes, 0, "should not underflow");
    }

    #[test]
    fn release_usage_unknown_namespace_error() {
        let mut e = StorageQuotaEnforcer::new(small_config());
        let result = e.release_usage("phantom", 10);
        assert!(result.is_err());
    }

    // --- utilization ---

    #[test]
    fn utilization_calculation() {
        let mut e = StorageQuotaEnforcer::new(small_config());
        e.register_namespace("util", Some(500), Some(1000));
        e.record_usage("util", 250).expect("ok");
        let u = e.utilization("util").expect("should exist");
        assert!((u - 0.25).abs() < 1e-9, "250/1000 = 0.25, got {u}");
    }

    #[test]
    fn utilization_zero_hard_limit() {
        let mut e = StorageQuotaEnforcer::new(small_config());
        e.register_namespace("zero", Some(0), Some(0));
        let u = e.utilization("zero").expect("should exist");
        assert!(u.is_infinite(), "zero hard limit should yield infinity");
    }

    #[test]
    fn utilization_unknown_namespace() {
        let e = StorageQuotaEnforcer::new(small_config());
        assert!(e.utilization("nope").is_none());
    }

    #[test]
    fn utilization_full() {
        let mut e = StorageQuotaEnforcer::new(small_config());
        e.register_namespace("full", Some(500), Some(1000));
        e.record_usage("full", 1000).expect("ok");
        let u = e.utilization("full").expect("should exist");
        assert!((u - 1.0).abs() < 1e-9, "1000/1000 = 1.0, got {u}");
    }

    // --- over_quota_namespaces ---

    #[test]
    fn over_quota_namespaces_empty_when_none_exceeded() {
        let mut e = StorageQuotaEnforcer::new(small_config());
        e.register_namespace("ok1", Some(500), Some(1000));
        e.register_namespace("ok2", Some(500), Some(1000));
        assert!(e.over_quota_namespaces().is_empty());
    }

    #[test]
    fn over_quota_namespaces_returns_exceeded() {
        let mut e = StorageQuotaEnforcer::new(small_config());
        e.register_namespace("ok", Some(500), Some(1000));
        e.register_namespace("bad", Some(500), Some(1000));
        e.record_usage("bad", 1500).expect("ok");
        let over = e.over_quota_namespaces();
        assert_eq!(over.len(), 1);
        assert_eq!(over[0].namespace, "bad");
    }

    #[test]
    fn over_quota_namespaces_multiple() {
        let mut e = StorageQuotaEnforcer::new(small_config());
        e.register_namespace("a", Some(50), Some(100));
        e.register_namespace("b", Some(50), Some(100));
        e.register_namespace("c", Some(50), Some(100));
        e.record_usage("a", 200).expect("ok");
        e.record_usage("c", 150).expect("ok");
        let over = e.over_quota_namespaces();
        assert_eq!(over.len(), 2);
        let names: Vec<&str> = over.iter().map(|q| q.namespace.as_str()).collect();
        assert!(names.contains(&"a"));
        assert!(names.contains(&"c"));
    }

    // --- get_level ---

    #[test]
    fn get_level_ok() {
        let mut e = StorageQuotaEnforcer::new(small_config());
        e.register_namespace("gl", Some(500), Some(1000));
        assert_eq!(e.get_level("gl"), Some(QuotaLevel::Ok));
    }

    #[test]
    fn get_level_warning() {
        let mut e = StorageQuotaEnforcer::new(small_config());
        e.register_namespace("gl", Some(500), Some(1000));
        e.record_usage("gl", 700).expect("ok");
        assert_eq!(e.get_level("gl"), Some(QuotaLevel::Warning));
    }

    #[test]
    fn get_level_exceeded() {
        let mut e = StorageQuotaEnforcer::new(small_config());
        e.register_namespace("gl", Some(500), Some(1000));
        e.record_usage("gl", 1500).expect("ok");
        assert_eq!(e.get_level("gl"), Some(QuotaLevel::Exceeded));
    }

    #[test]
    fn get_level_unknown_namespace() {
        let e = StorageQuotaEnforcer::new(small_config());
        assert_eq!(e.get_level("missing"), None);
    }

    // --- stats ---

    #[test]
    fn stats_initial() {
        let e = StorageQuotaEnforcer::new(small_config());
        let s = e.stats();
        assert_eq!(s.namespace_count, 0);
        assert_eq!(s.total_checks, 0);
        assert_eq!(s.total_rejections, 0);
        assert_eq!(s.over_quota_count, 0);
    }

    #[test]
    fn stats_after_operations() {
        let mut e = StorageQuotaEnforcer::new(small_config());
        e.register_namespace("s1", Some(50), Some(100));
        e.register_namespace("s2", Some(50), Some(100));

        // 3 checks: Ok, Warning, Exceeded
        let _ = e.check_quota("s1", 10); // Ok
        let _ = e.check_quota("s1", 60); // Warning
        let _ = e.check_quota("s1", 200); // Exceeded -> rejection

        let s = e.stats();
        assert_eq!(s.namespace_count, 2);
        assert_eq!(s.total_checks, 3);
        assert_eq!(s.total_rejections, 1);
        assert_eq!(s.over_quota_count, 0, "no actual usage recorded");
    }

    #[test]
    fn stats_over_quota_count() {
        let mut e = StorageQuotaEnforcer::new(small_config());
        e.register_namespace("x", Some(50), Some(100));
        e.register_namespace("y", Some(50), Some(100));
        e.record_usage("x", 200).expect("ok");
        let s = e.stats();
        assert_eq!(s.over_quota_count, 1);
    }

    // --- zero limits edge case ---

    #[test]
    fn zero_limits_everything_exceeded() {
        let mut e = StorageQuotaEnforcer::new(small_config());
        e.register_namespace("zero", Some(0), Some(0));
        // Even 0 additional bytes: usage 0 >= hard 0 is Exceeded.
        let level = e.check_quota("zero", 0).expect("should succeed");
        assert_eq!(level, QuotaLevel::Exceeded);
    }

    #[test]
    fn zero_limits_any_usage_exceeded() {
        let mut e = StorageQuotaEnforcer::new(small_config());
        e.register_namespace("zero2", Some(0), Some(0));
        let level = e.record_usage("zero2", 1).expect("should succeed");
        assert_eq!(level, QuotaLevel::Exceeded);
    }

    // --- Default trait ---

    #[test]
    fn default_enforcer() {
        let e = StorageQuotaEnforcer::default();
        let s = e.stats();
        assert_eq!(s.namespace_count, 0);
        assert_eq!(s.total_checks, 0);
    }

    // --- get_quota ---

    #[test]
    fn get_quota_returns_none_for_unknown() {
        let e = StorageQuotaEnforcer::new(small_config());
        assert!(e.get_quota("nope").is_none());
    }

    #[test]
    fn get_quota_returns_correct_data() {
        let mut e = StorageQuotaEnforcer::new(small_config());
        e.register_namespace("data", Some(100), Some(200));
        e.record_usage("data", 75).expect("ok");
        let q = e.get_quota("data").expect("should exist");
        assert_eq!(q.namespace, "data");
        assert_eq!(q.soft_limit_bytes, 100);
        assert_eq!(q.hard_limit_bytes, 200);
        assert_eq!(q.current_usage_bytes, 75);
        assert_eq!(q.block_count, 1);
    }

    // --- check_quota with existing usage ---

    #[test]
    fn check_quota_considers_existing_usage() {
        let mut e = StorageQuotaEnforcer::new(small_config());
        e.register_namespace("eu", Some(500), Some(1000));
        e.record_usage("eu", 400).expect("ok");
        // 400 + 200 = 600 -> Warning (>= soft 500, < hard 1000)
        let level = e.check_quota("eu", 200).expect("should succeed");
        assert_eq!(level, QuotaLevel::Warning);
    }

    #[test]
    fn check_quota_with_usage_near_hard() {
        let mut e = StorageQuotaEnforcer::new(small_config());
        e.register_namespace("eu2", Some(500), Some(1000));
        e.record_usage("eu2", 900).expect("ok");
        // 900 + 200 = 1100 -> Exceeded
        let level = e.check_quota("eu2", 200).expect("should succeed");
        assert_eq!(level, QuotaLevel::Exceeded);
    }

    // --- rejection counting ---

    #[test]
    fn check_quota_counts_rejections() {
        let mut e = StorageQuotaEnforcer::new(small_config());
        e.register_namespace("rej", Some(50), Some(100));
        let _ = e.check_quota("rej", 200); // Exceeded
        let _ = e.check_quota("rej", 300); // Exceeded
        let _ = e.check_quota("rej", 10); // Ok
        let s = e.stats();
        assert_eq!(s.total_rejections, 2);
        assert_eq!(s.total_checks, 3);
    }
}
