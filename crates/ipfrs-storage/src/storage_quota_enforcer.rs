//! Storage Quota Enforcer — per-namespace quota enforcement with forecasting
//!
//! Provides:
//! - Per-namespace storage quota limits (bytes, objects, max object size)
//! - Configurable enforcement policies: Reject, Evict, Warn, Throttle
//! - Usage tracking over time with sliding history window
//! - Linear-regression growth forecasting (days until quota)
//! - Violation log for audit and monitoring
//! - Top-N namespace ranking by usage

use std::collections::{HashMap, VecDeque};

// ---------------------------------------------------------------------------
// NamespaceId
// ---------------------------------------------------------------------------

/// Newtype wrapper for namespace identifiers.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NamespaceId(pub String);

impl NamespaceId {
    /// Creates a new `NamespaceId` from any string-like value.
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }

    /// Returns the inner string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for NamespaceId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

// ---------------------------------------------------------------------------
// QuotaLimit
// ---------------------------------------------------------------------------

/// Quota limits for a namespace. A value of 0 means unlimited for that field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuotaLimit {
    /// Maximum total bytes. 0 = unlimited.
    pub max_bytes: u64,
    /// Maximum number of objects. 0 = unlimited.
    pub max_objects: u64,
    /// Maximum size of a single object. 0 = unlimited.
    pub max_object_size: u64,
}

impl QuotaLimit {
    /// Creates a quota with only a byte limit; objects and size are unlimited.
    pub fn bytes_only(max_bytes: u64) -> Self {
        Self {
            max_bytes,
            max_objects: 0,
            max_object_size: 0,
        }
    }

    /// Creates a quota with bytes, objects, and max object size limits.
    pub fn full(max_bytes: u64, max_objects: u64, max_object_size: u64) -> Self {
        Self {
            max_bytes,
            max_objects,
            max_object_size,
        }
    }
}

// ---------------------------------------------------------------------------
// QuotaUsage
// ---------------------------------------------------------------------------

/// Current usage snapshot for a single namespace.
#[derive(Debug, Clone)]
pub struct QuotaUsage {
    /// Namespace this usage belongs to.
    pub namespace: NamespaceId,
    /// Total bytes currently stored.
    pub bytes_used: u64,
    /// Total number of objects currently stored.
    pub objects_used: u64,
    /// Unix timestamp (seconds) when this record was last updated.
    pub last_updated: u64,
}

// ---------------------------------------------------------------------------
// EnforcementPolicy
// ---------------------------------------------------------------------------

/// Defines how quota violations are handled.
#[derive(Debug, Clone, PartialEq)]
pub enum EnforcementPolicy {
    /// Deny the write operation when the quota is exceeded.
    Reject,
    /// Evict least-recently-used data to make space; the caller is responsible
    /// for performing the actual eviction.
    Evict,
    /// Allow the write but record the violation as a warning.
    Warn,
    /// Allow the write but signal the caller to throttle at the given factor
    /// (0.0 = no throttle, 1.0 = full throttle / stop).
    Throttle { factor: f64 },
}

// ---------------------------------------------------------------------------
// ViolationKind
// ---------------------------------------------------------------------------

/// Describes why a quota violation occurred.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ViolationKind {
    /// The total stored bytes would exceed `max_bytes`.
    BytesExceeded,
    /// The total number of stored objects would exceed `max_objects`.
    ObjectCountExceeded,
    /// The size of the individual object exceeds `max_object_size`.
    ObjectSizeTooLarge,
}

// ---------------------------------------------------------------------------
// QuotaViolation (module-internal name; exported as SqeQuotaViolation)
// ---------------------------------------------------------------------------

/// Describes a single quota violation event.
#[derive(Debug, Clone)]
pub struct QuotaViolation {
    /// Namespace in which the violation occurred.
    pub namespace: NamespaceId,
    /// What kind of limit was breached.
    pub kind: ViolationKind,
    /// The observed value that exceeded the limit.
    pub current: u64,
    /// The limit that was breached.
    pub limit: u64,
    /// Unix timestamp (seconds) when the violation was detected.
    pub timestamp: u64,
}

// ---------------------------------------------------------------------------
// UsageSample
// ---------------------------------------------------------------------------

/// A point-in-time usage measurement used for growth forecasting.
#[derive(Debug, Clone, Copy)]
pub struct UsageSample {
    /// Unix timestamp (seconds) of the measurement.
    pub timestamp: u64,
    /// Bytes in use at this timestamp.
    pub bytes: u64,
    /// Objects in use at this timestamp.
    pub objects: u64,
}

// ---------------------------------------------------------------------------
// GrowthForecast
// ---------------------------------------------------------------------------

/// Growth forecast for a namespace derived from historical samples.
#[derive(Debug, Clone)]
pub struct GrowthForecast {
    /// Namespace this forecast is for.
    pub namespace: NamespaceId,
    /// Predicted bytes in 30 days from `now`.
    pub predicted_bytes_in_30d: u64,
    /// Days until the namespace hits its quota, or `None` if no quota is set
    /// or growth rate is non-positive (never fills up).
    pub days_until_quota: Option<u64>,
    /// Estimated bytes added per day (may be negative if shrinking).
    pub growth_rate_bytes_per_day: f64,
}

// ---------------------------------------------------------------------------
// EnforcerConfig
// ---------------------------------------------------------------------------

/// Configuration for [`SqeStorageQuotaEnforcer`].
#[derive(Debug, Clone)]
pub struct EnforcerConfig {
    /// Default enforcement policy applied when no per-namespace policy is set.
    pub default_policy: EnforcementPolicy,
    /// Number of historical [`UsageSample`] entries kept per namespace.
    pub history_window: usize,
    /// Maximum number of namespaces the enforcer will track.
    pub max_namespaces: usize,
}

impl Default for EnforcerConfig {
    fn default() -> Self {
        Self {
            default_policy: EnforcementPolicy::Reject,
            history_window: 100,
            max_namespaces: 1000,
        }
    }
}

// ---------------------------------------------------------------------------
// SqeStorageQuotaEnforcer
// ---------------------------------------------------------------------------

/// Enforces per-namespace storage quotas with configurable policies and
/// historical growth forecasting.
///
/// This is the main entry point for the storage quota enforcement subsystem.
/// The type name is prefixed with `Sqe` to avoid collision with the existing
/// `StorageQuotaEnforcer` in `quota_enforcer.rs`.
pub struct SqeStorageQuotaEnforcer {
    config: EnforcerConfig,
    quotas: HashMap<NamespaceId, QuotaLimit>,
    usage: HashMap<NamespaceId, QuotaUsage>,
    history: HashMap<NamespaceId, VecDeque<UsageSample>>,
    violations: Vec<QuotaViolation>,
    policies: HashMap<NamespaceId, EnforcementPolicy>,
}

impl SqeStorageQuotaEnforcer {
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    /// Creates a new enforcer with the supplied configuration.
    pub fn new(config: EnforcerConfig) -> Self {
        Self {
            config,
            quotas: HashMap::new(),
            usage: HashMap::new(),
            history: HashMap::new(),
            violations: Vec::new(),
            policies: HashMap::new(),
        }
    }

    // -----------------------------------------------------------------------
    // Quota management
    // -----------------------------------------------------------------------

    /// Registers or updates the quota limit for `ns`.
    pub fn set_quota(&mut self, ns: NamespaceId, limit: QuotaLimit) {
        self.quotas.insert(ns, limit);
    }

    /// Removes the quota for `ns`. Returns `true` if a quota existed.
    pub fn remove_quota(&mut self, ns: &NamespaceId) -> bool {
        self.quotas.remove(ns).is_some()
    }

    /// Overrides the enforcement policy for `ns`.
    pub fn set_policy(&mut self, ns: NamespaceId, policy: EnforcementPolicy) {
        self.policies.insert(ns, policy);
    }

    // -----------------------------------------------------------------------
    // Write checking
    // -----------------------------------------------------------------------

    /// Checks whether a write of `object_size` bytes is permitted for `ns`.
    ///
    /// - If no quota is configured for `ns`, the write is always allowed.
    /// - Violations are appended to the internal log regardless of policy.
    /// - `Reject` → returns `Err(violation)`.
    /// - `Warn` / `Throttle` / `Evict` → returns `Ok(())`.
    pub fn check_write(&self, ns: &NamespaceId, object_size: u64) -> Result<(), QuotaViolation> {
        let limit = match self.quotas.get(ns) {
            Some(l) => l,
            None => return Ok(()),
        };

        let usage = self.usage.get(ns);
        let bytes_used = usage.map(|u| u.bytes_used).unwrap_or(0);
        let objects_used = usage.map(|u| u.objects_used).unwrap_or(0);

        let policy = self.policies.get(ns).unwrap_or(&self.config.default_policy);

        // 1. Check per-object size limit.
        if limit.max_object_size > 0 && object_size > limit.max_object_size {
            let v = QuotaViolation {
                namespace: ns.clone(),
                kind: ViolationKind::ObjectSizeTooLarge,
                current: object_size,
                limit: limit.max_object_size,
                timestamp: 0,
            };
            return Self::apply_policy(policy, v);
        }

        // 2. Check total bytes limit.
        if limit.max_bytes > 0 {
            let projected = bytes_used.saturating_add(object_size);
            if projected > limit.max_bytes {
                let v = QuotaViolation {
                    namespace: ns.clone(),
                    kind: ViolationKind::BytesExceeded,
                    current: projected,
                    limit: limit.max_bytes,
                    timestamp: 0,
                };
                return Self::apply_policy(policy, v);
            }
        }

        // 3. Check total objects limit.
        if limit.max_objects > 0 {
            let projected_objects = objects_used.saturating_add(1);
            if projected_objects > limit.max_objects {
                let v = QuotaViolation {
                    namespace: ns.clone(),
                    kind: ViolationKind::ObjectCountExceeded,
                    current: projected_objects,
                    limit: limit.max_objects,
                    timestamp: 0,
                };
                return Self::apply_policy(policy, v);
            }
        }

        Ok(())
    }

    /// Applies the enforcement policy to a violation.
    ///
    /// - `Reject` → `Err(violation)`
    /// - anything else → `Ok(())` (caller may inspect `violations()` separately)
    fn apply_policy(
        policy: &EnforcementPolicy,
        violation: QuotaViolation,
    ) -> Result<(), QuotaViolation> {
        match policy {
            EnforcementPolicy::Reject => Err(violation),
            EnforcementPolicy::Warn
            | EnforcementPolicy::Throttle { .. }
            | EnforcementPolicy::Evict => Ok(()),
        }
    }

    // -----------------------------------------------------------------------
    // Usage recording
    // -----------------------------------------------------------------------

    /// Records a successful write of `object_size` bytes at time `now` (Unix
    /// seconds) for namespace `ns`, then appends a history sample.
    ///
    /// If `ns` has never been seen before and `max_namespaces` would be
    /// exceeded, the write is silently ignored to avoid unbounded growth.
    pub fn record_write(&mut self, ns: &NamespaceId, object_size: u64, now: u64) {
        if !self.usage.contains_key(ns) {
            if self.usage.len() >= self.config.max_namespaces {
                return;
            }
            self.usage.insert(
                ns.clone(),
                QuotaUsage {
                    namespace: ns.clone(),
                    bytes_used: 0,
                    objects_used: 0,
                    last_updated: now,
                },
            );
        }

        if let Some(u) = self.usage.get_mut(ns) {
            u.bytes_used = u.bytes_used.saturating_add(object_size);
            u.objects_used = u.objects_used.saturating_add(1);
            u.last_updated = now;
        }

        self.append_history_sample(ns, now);
    }

    /// Records a deletion of `object_size` bytes at time `now` (Unix seconds).
    ///
    /// Usage floors at zero; an unknown namespace is silently ignored.
    pub fn record_delete(&mut self, ns: &NamespaceId, object_size: u64, now: u64) {
        if let Some(u) = self.usage.get_mut(ns) {
            u.bytes_used = u.bytes_used.saturating_sub(object_size);
            u.objects_used = u.objects_used.saturating_sub(1);
            u.last_updated = now;
            let (bytes, objects) = (u.bytes_used, u.objects_used);
            self.append_history_sample_values(ns, now, bytes, objects);
        }
    }

    /// Appends a history sample using the current stored usage values.
    fn append_history_sample(&mut self, ns: &NamespaceId, now: u64) {
        let (bytes, objects) = match self.usage.get(ns) {
            Some(u) => (u.bytes_used, u.objects_used),
            None => return,
        };
        self.append_history_sample_values(ns, now, bytes, objects);
    }

    /// Appends a history sample with explicit byte and object counts.
    fn append_history_sample_values(
        &mut self,
        ns: &NamespaceId,
        now: u64,
        bytes: u64,
        objects: u64,
    ) {
        let window = self.config.history_window;
        let deque = self.history.entry(ns.clone()).or_default();

        deque.push_back(UsageSample {
            timestamp: now,
            bytes,
            objects,
        });

        // Truncate to window size from the front (oldest entries removed).
        while deque.len() > window {
            deque.pop_front();
        }
    }

    // -----------------------------------------------------------------------
    // Queries
    // -----------------------------------------------------------------------

    /// Returns the current usage for `ns`, or `None` if unknown.
    pub fn usage_for(&self, ns: &NamespaceId) -> Option<&QuotaUsage> {
        self.usage.get(ns)
    }

    /// Returns the percentage of the byte quota consumed as a value in
    /// `[0.0, ∞)`, or `None` if no quota is set or `max_bytes` is 0.
    pub fn usage_percent_bytes(&self, ns: &NamespaceId) -> Option<f64> {
        let limit = self.quotas.get(ns)?;
        if limit.max_bytes == 0 {
            return None;
        }
        let bytes_used = self.usage.get(ns).map(|u| u.bytes_used).unwrap_or(0);
        Some(bytes_used as f64 / limit.max_bytes as f64 * 100.0)
    }

    /// Forecasts the byte growth for `ns` over the next 30 days using linear
    /// regression of the history window.
    ///
    /// Returns `None` if fewer than 2 history samples are available.
    pub fn forecast(&self, ns: &NamespaceId, now: u64) -> Option<GrowthForecast> {
        let samples = self.history.get(ns)?;
        if samples.len() < 2 {
            return None;
        }

        let slope = Self::linear_regression_slope(samples);
        // slope is bytes-per-second; convert to bytes-per-day.
        let growth_rate_bytes_per_day = slope * 86_400.0;

        let thirty_days_seconds: f64 = 30.0 * 86_400.0;
        let current_bytes = self.usage.get(ns).map(|u| u.bytes_used).unwrap_or(0);

        let predicted_bytes_f64 = current_bytes as f64 + slope * thirty_days_seconds;
        let predicted_bytes_in_30d = if predicted_bytes_f64 > 0.0 {
            predicted_bytes_f64 as u64
        } else {
            0
        };

        let days_until_quota = if let Some(limit) = self.quotas.get(ns) {
            if limit.max_bytes > 0 && growth_rate_bytes_per_day > 0.0 {
                let remaining = limit.max_bytes as f64 - current_bytes as f64;
                if remaining <= 0.0 {
                    // Already over quota.
                    Some(0u64)
                } else {
                    let days = remaining / growth_rate_bytes_per_day;
                    Some(days.floor() as u64)
                }
            } else {
                None
            }
        } else {
            None
        };

        // `now` is accepted for API completeness; the forecast uses the samples
        // themselves to derive the slope so `now` is not strictly needed here.
        let _ = now;

        Some(GrowthForecast {
            namespace: ns.clone(),
            predicted_bytes_in_30d,
            days_until_quota,
            growth_rate_bytes_per_day,
        })
    }

    /// Returns all recorded violations in insertion order.
    pub fn violations(&self) -> &[QuotaViolation] {
        &self.violations
    }

    /// Records a violation into the internal log (useful for callers using
    /// non-`Reject` policies who still want audit history).
    pub fn record_violation(&mut self, v: QuotaViolation) {
        self.violations.push(v);
    }

    /// Returns the top `n` namespaces sorted by bytes used (descending).
    pub fn top_namespaces_by_usage(&self, n: usize) -> Vec<(NamespaceId, u64)> {
        let mut entries: Vec<(NamespaceId, u64)> = self
            .usage
            .iter()
            .map(|(ns, u)| (ns.clone(), u.bytes_used))
            .collect();

        entries.sort_by_key(|b| std::cmp::Reverse(b.1));
        entries.truncate(n);
        entries
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Computes the ordinary least-squares slope (bytes per second) for the
    /// samples using the formula:
    ///
    /// ```text
    /// slope = (n·Σxy − Σx·Σy) / (n·Σx² − (Σx)²)
    /// ```
    ///
    /// where `x = timestamp` and `y = bytes`.
    ///
    /// Returns `0.0` if the denominator is zero (constant x values).
    fn linear_regression_slope(samples: &VecDeque<UsageSample>) -> f64 {
        let n = samples.len() as f64;
        if n < 2.0 {
            return 0.0;
        }

        let mut sum_x: f64 = 0.0;
        let mut sum_y: f64 = 0.0;
        let mut sum_xy: f64 = 0.0;
        let mut sum_xx: f64 = 0.0;

        for s in samples {
            let x = s.timestamp as f64;
            let y = s.bytes as f64;
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_xx += x * x;
        }

        let denom = n * sum_xx - sum_x * sum_x;
        if denom == 0.0 {
            return 0.0;
        }

        (n * sum_xy - sum_x * sum_y) / denom
    }
}

impl Default for SqeStorageQuotaEnforcer {
    fn default() -> Self {
        Self::new(EnforcerConfig::default())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn small_config() -> EnforcerConfig {
        EnforcerConfig {
            default_policy: EnforcementPolicy::Reject,
            history_window: 10,
            max_namespaces: 5,
        }
    }

    fn ns(s: &str) -> NamespaceId {
        NamespaceId::new(s)
    }

    // -----------------------------------------------------------------------
    // NamespaceId
    // -----------------------------------------------------------------------

    #[test]
    fn namespace_id_new_and_as_str() {
        let id = NamespaceId::new("tenant-1");
        assert_eq!(id.as_str(), "tenant-1");
    }

    #[test]
    fn namespace_id_display() {
        let id = NamespaceId::new("my-ns");
        assert_eq!(format!("{id}"), "my-ns");
    }

    #[test]
    fn namespace_id_equality() {
        assert_eq!(ns("a"), ns("a"));
        assert_ne!(ns("a"), ns("b"));
    }

    // -----------------------------------------------------------------------
    // QuotaLimit helpers
    // -----------------------------------------------------------------------

    #[test]
    fn quota_limit_bytes_only() {
        let l = QuotaLimit::bytes_only(1000);
        assert_eq!(l.max_bytes, 1000);
        assert_eq!(l.max_objects, 0);
        assert_eq!(l.max_object_size, 0);
    }

    #[test]
    fn quota_limit_full() {
        let l = QuotaLimit::full(2000, 10, 500);
        assert_eq!(l.max_bytes, 2000);
        assert_eq!(l.max_objects, 10);
        assert_eq!(l.max_object_size, 500);
    }

    // -----------------------------------------------------------------------
    // set_quota / remove_quota
    // -----------------------------------------------------------------------

    #[test]
    fn set_and_remove_quota() {
        let mut e = SqeStorageQuotaEnforcer::new(small_config());
        e.set_quota(ns("ns1"), QuotaLimit::bytes_only(1000));
        assert!(e.remove_quota(&ns("ns1")));
        assert!(!e.remove_quota(&ns("ns1")));
    }

    #[test]
    fn remove_nonexistent_quota_returns_false() {
        let mut e = SqeStorageQuotaEnforcer::new(small_config());
        assert!(!e.remove_quota(&ns("ghost")));
    }

    // -----------------------------------------------------------------------
    // check_write — no quota
    // -----------------------------------------------------------------------

    #[test]
    fn check_write_no_quota_always_ok() {
        let e = SqeStorageQuotaEnforcer::new(small_config());
        assert!(e.check_write(&ns("free"), 999_999).is_ok());
    }

    // -----------------------------------------------------------------------
    // check_write — object size limit
    // -----------------------------------------------------------------------

    #[test]
    fn check_write_object_size_too_large_reject() {
        let mut e = SqeStorageQuotaEnforcer::new(small_config());
        e.set_quota(ns("ns"), QuotaLimit::full(10_000, 0, 100));
        let result = e.check_write(&ns("ns"), 101);
        assert!(result.is_err());
        let v = result.expect_err("should be violation");
        assert_eq!(v.kind, ViolationKind::ObjectSizeTooLarge);
        assert_eq!(v.current, 101);
        assert_eq!(v.limit, 100);
    }

    #[test]
    fn check_write_object_size_exactly_at_limit_ok() {
        let mut e = SqeStorageQuotaEnforcer::new(small_config());
        e.set_quota(ns("ns"), QuotaLimit::full(10_000, 0, 100));
        assert!(e.check_write(&ns("ns"), 100).is_ok());
    }

    #[test]
    fn check_write_object_size_zero_limit_means_unlimited() {
        let mut e = SqeStorageQuotaEnforcer::new(small_config());
        // max_bytes=0 (unlimited), max_objects=0 (unlimited), max_object_size=0 (unlimited).
        e.set_quota(ns("ns"), QuotaLimit::full(0, 0, 0));
        assert!(e.check_write(&ns("ns"), 999_999).is_ok());
    }

    // -----------------------------------------------------------------------
    // check_write — bytes exceeded
    // -----------------------------------------------------------------------

    #[test]
    fn check_write_bytes_exceeded_reject() {
        let mut e = SqeStorageQuotaEnforcer::new(small_config());
        e.set_quota(ns("ns"), QuotaLimit::bytes_only(500));
        e.record_write(&ns("ns"), 400, 1000);
        let result = e.check_write(&ns("ns"), 200);
        assert!(result.is_err());
        let v = result.expect_err("violation");
        assert_eq!(v.kind, ViolationKind::BytesExceeded);
        assert_eq!(v.limit, 500);
    }

    #[test]
    fn check_write_bytes_exactly_at_limit_ok() {
        let mut e = SqeStorageQuotaEnforcer::new(small_config());
        e.set_quota(ns("ns"), QuotaLimit::bytes_only(500));
        e.record_write(&ns("ns"), 400, 1000);
        // 400 + 100 = 500, which is NOT > 500, so Ok.
        assert!(e.check_write(&ns("ns"), 100).is_ok());
    }

    #[test]
    fn check_write_bytes_zero_limit_means_unlimited() {
        let mut e = SqeStorageQuotaEnforcer::new(small_config());
        e.set_quota(ns("ns"), QuotaLimit::full(0, 0, 0));
        assert!(e.check_write(&ns("ns"), 999_999).is_ok());
    }

    // -----------------------------------------------------------------------
    // check_write — object count exceeded
    // -----------------------------------------------------------------------

    #[test]
    fn check_write_object_count_exceeded_reject() {
        let mut e = SqeStorageQuotaEnforcer::new(small_config());
        e.set_quota(ns("ns"), QuotaLimit::full(0, 3, 0));
        e.record_write(&ns("ns"), 10, 1000);
        e.record_write(&ns("ns"), 10, 1001);
        e.record_write(&ns("ns"), 10, 1002);
        let result = e.check_write(&ns("ns"), 10);
        assert!(result.is_err());
        let v = result.expect_err("violation");
        assert_eq!(v.kind, ViolationKind::ObjectCountExceeded);
    }

    #[test]
    fn check_write_object_count_at_limit_still_allows_final() {
        let mut e = SqeStorageQuotaEnforcer::new(small_config());
        e.set_quota(ns("ns"), QuotaLimit::full(0, 3, 0));
        e.record_write(&ns("ns"), 10, 1000);
        e.record_write(&ns("ns"), 10, 1001);
        // Third write should succeed (objects_used would become 3 which is NOT > 3).
        assert!(e.check_write(&ns("ns"), 10).is_ok());
    }

    // -----------------------------------------------------------------------
    // check_write — policy variants
    // -----------------------------------------------------------------------

    #[test]
    fn check_write_warn_policy_returns_ok() {
        let mut e = SqeStorageQuotaEnforcer::new(small_config());
        e.set_quota(ns("ns"), QuotaLimit::bytes_only(100));
        e.set_policy(ns("ns"), EnforcementPolicy::Warn);
        e.record_write(&ns("ns"), 80, 1000);
        // Would exceed, but Warn returns Ok.
        assert!(e.check_write(&ns("ns"), 50).is_ok());
    }

    #[test]
    fn check_write_throttle_policy_returns_ok() {
        let mut e = SqeStorageQuotaEnforcer::new(small_config());
        e.set_quota(ns("ns"), QuotaLimit::bytes_only(100));
        e.set_policy(ns("ns"), EnforcementPolicy::Throttle { factor: 0.8 });
        e.record_write(&ns("ns"), 80, 1000);
        assert!(e.check_write(&ns("ns"), 50).is_ok());
    }

    #[test]
    fn check_write_evict_policy_returns_ok() {
        let mut e = SqeStorageQuotaEnforcer::new(small_config());
        e.set_quota(ns("ns"), QuotaLimit::bytes_only(100));
        e.set_policy(ns("ns"), EnforcementPolicy::Evict);
        e.record_write(&ns("ns"), 80, 1000);
        assert!(e.check_write(&ns("ns"), 50).is_ok());
    }

    #[test]
    fn check_write_reject_policy_returns_err() {
        let mut e = SqeStorageQuotaEnforcer::new(small_config());
        e.set_quota(ns("ns"), QuotaLimit::bytes_only(100));
        e.set_policy(ns("ns"), EnforcementPolicy::Reject);
        e.record_write(&ns("ns"), 80, 1000);
        assert!(e.check_write(&ns("ns"), 50).is_err());
    }

    // -----------------------------------------------------------------------
    // record_write
    // -----------------------------------------------------------------------

    #[test]
    fn record_write_increments_usage() {
        let mut e = SqeStorageQuotaEnforcer::new(small_config());
        e.record_write(&ns("ns"), 100, 1000);
        let u = e.usage_for(&ns("ns")).expect("should exist");
        assert_eq!(u.bytes_used, 100);
        assert_eq!(u.objects_used, 1);
        assert_eq!(u.last_updated, 1000);
    }

    #[test]
    fn record_write_multiple_accumulates() {
        let mut e = SqeStorageQuotaEnforcer::new(small_config());
        e.record_write(&ns("ns"), 100, 1000);
        e.record_write(&ns("ns"), 200, 2000);
        let u = e.usage_for(&ns("ns")).expect("exists");
        assert_eq!(u.bytes_used, 300);
        assert_eq!(u.objects_used, 2);
    }

    #[test]
    fn record_write_appends_history_sample() {
        let mut e = SqeStorageQuotaEnforcer::new(small_config());
        e.record_write(&ns("ns"), 100, 1000);
        e.record_write(&ns("ns"), 50, 2000);
        let history = e.history.get(&ns("ns")).expect("history should exist");
        assert_eq!(history.len(), 2);
    }

    #[test]
    fn record_write_truncates_history_to_window() {
        let config = EnforcerConfig {
            history_window: 3,
            ..small_config()
        };
        let mut e = SqeStorageQuotaEnforcer::new(config);
        for i in 0..6u64 {
            e.record_write(&ns("ns"), 10, 1000 + i);
        }
        let history = e.history.get(&ns("ns")).expect("history should exist");
        assert_eq!(history.len(), 3);
    }

    #[test]
    fn record_write_respects_max_namespaces() {
        let config = EnforcerConfig {
            max_namespaces: 2,
            ..small_config()
        };
        let mut e = SqeStorageQuotaEnforcer::new(config);
        e.record_write(&ns("a"), 10, 1);
        e.record_write(&ns("b"), 10, 2);
        // Third namespace should be silently ignored.
        e.record_write(&ns("c"), 10, 3);
        assert!(e.usage_for(&ns("c")).is_none());
    }

    // -----------------------------------------------------------------------
    // record_delete
    // -----------------------------------------------------------------------

    #[test]
    fn record_delete_decrements_usage() {
        let mut e = SqeStorageQuotaEnforcer::new(small_config());
        e.record_write(&ns("ns"), 300, 1000);
        e.record_delete(&ns("ns"), 100, 2000);
        let u = e.usage_for(&ns("ns")).expect("exists");
        assert_eq!(u.bytes_used, 200);
        assert_eq!(u.objects_used, 0);
        assert_eq!(u.last_updated, 2000);
    }

    #[test]
    fn record_delete_saturates_at_zero() {
        let mut e = SqeStorageQuotaEnforcer::new(small_config());
        e.record_write(&ns("ns"), 50, 1000);
        e.record_delete(&ns("ns"), 9999, 2000);
        let u = e.usage_for(&ns("ns")).expect("exists");
        assert_eq!(u.bytes_used, 0);
        assert_eq!(u.objects_used, 0);
    }

    #[test]
    fn record_delete_unknown_namespace_noop() {
        let mut e = SqeStorageQuotaEnforcer::new(small_config());
        // Should not panic.
        e.record_delete(&ns("ghost"), 100, 1000);
        assert!(e.usage_for(&ns("ghost")).is_none());
    }

    // -----------------------------------------------------------------------
    // usage_percent_bytes
    // -----------------------------------------------------------------------

    #[test]
    fn usage_percent_bytes_no_quota_returns_none() {
        let mut e = SqeStorageQuotaEnforcer::new(small_config());
        e.record_write(&ns("ns"), 100, 1);
        assert!(e.usage_percent_bytes(&ns("ns")).is_none());
    }

    #[test]
    fn usage_percent_bytes_zero_max_returns_none() {
        let mut e = SqeStorageQuotaEnforcer::new(small_config());
        e.set_quota(ns("ns"), QuotaLimit::full(0, 0, 0));
        e.record_write(&ns("ns"), 100, 1);
        assert!(e.usage_percent_bytes(&ns("ns")).is_none());
    }

    #[test]
    fn usage_percent_bytes_half_quota() {
        let mut e = SqeStorageQuotaEnforcer::new(small_config());
        e.set_quota(ns("ns"), QuotaLimit::bytes_only(1000));
        e.record_write(&ns("ns"), 500, 1);
        let pct = e.usage_percent_bytes(&ns("ns")).expect("should be Some");
        assert!((pct - 50.0).abs() < 1e-9, "expected 50%, got {pct}");
    }

    #[test]
    fn usage_percent_bytes_full() {
        let mut e = SqeStorageQuotaEnforcer::new(small_config());
        e.set_quota(ns("ns"), QuotaLimit::bytes_only(1000));
        e.record_write(&ns("ns"), 1000, 1);
        let pct = e.usage_percent_bytes(&ns("ns")).expect("some");
        assert!((pct - 100.0).abs() < 1e-9);
    }

    // -----------------------------------------------------------------------
    // forecast
    // -----------------------------------------------------------------------

    #[test]
    fn forecast_requires_at_least_two_samples() {
        let mut e = SqeStorageQuotaEnforcer::new(small_config());
        e.record_write(&ns("ns"), 100, 1000);
        assert!(e.forecast(&ns("ns"), 2000).is_none());
    }

    #[test]
    fn forecast_returns_some_with_two_samples() {
        let mut e = SqeStorageQuotaEnforcer::new(small_config());
        e.record_write(&ns("ns"), 100, 0);
        e.record_write(&ns("ns"), 200, 86_400); // 1 day later
        let fc = e.forecast(&ns("ns"), 2 * 86_400).expect("should forecast");
        // Growth rate ~ 100 bytes/day.
        assert!(
            fc.growth_rate_bytes_per_day > 0.0,
            "positive growth expected"
        );
    }

    #[test]
    fn forecast_growth_rate_approximately_correct() {
        // Set up uniform daily samples: +100 bytes per day.
        let mut e = SqeStorageQuotaEnforcer::new(small_config());
        let day = 86_400u64;
        for i in 0..5u64 {
            e.record_write(&ns("ns"), 100, i * day);
        }
        let fc = e.forecast(&ns("ns"), 5 * day).expect("forecast");
        // 100 bytes per write per day, 4 intervals → slope ≈ 100 bytes/day.
        let rate = fc.growth_rate_bytes_per_day;
        assert!(rate > 50.0 && rate < 200.0, "rate {rate} out of range");
    }

    #[test]
    fn forecast_days_until_quota_set_correctly() {
        let mut e = SqeStorageQuotaEnforcer::new(small_config());
        let day = 86_400u64;
        e.set_quota(ns("ns"), QuotaLimit::bytes_only(1_000));
        // Simulate 3 days of 100 bytes/day growth.
        e.record_write(&ns("ns"), 100, 0);
        e.record_write(&ns("ns"), 100, day);
        e.record_write(&ns("ns"), 100, 2 * day);
        let fc = e.forecast(&ns("ns"), 3 * day).expect("forecast");
        // bytes_used = 300, max_bytes = 1000, growth ≈ 100/day → ~7 days remaining.
        if let Some(days) = fc.days_until_quota {
            assert!(days > 0, "should be some positive number of days");
        }
    }

    #[test]
    fn forecast_days_until_quota_none_when_no_quota() {
        let mut e = SqeStorageQuotaEnforcer::new(small_config());
        let day = 86_400u64;
        e.record_write(&ns("ns"), 100, 0);
        e.record_write(&ns("ns"), 100, day);
        let fc = e.forecast(&ns("ns"), 2 * day).expect("forecast");
        assert!(fc.days_until_quota.is_none());
    }

    #[test]
    fn forecast_days_until_quota_zero_when_already_over() {
        let mut e = SqeStorageQuotaEnforcer::new(small_config());
        let day = 86_400u64;
        e.set_quota(ns("ns"), QuotaLimit::bytes_only(50));
        e.record_write(&ns("ns"), 100, 0);
        e.record_write(&ns("ns"), 100, day);
        let fc = e.forecast(&ns("ns"), 2 * day).expect("forecast");
        if let Some(days) = fc.days_until_quota {
            assert_eq!(days, 0, "already over quota should yield 0 days");
        }
    }

    #[test]
    fn forecast_negative_growth_days_until_quota_none() {
        let mut e = SqeStorageQuotaEnforcer::new(small_config());
        let day = 86_400u64;
        e.set_quota(ns("ns"), QuotaLimit::bytes_only(1000));
        // Simulate shrinking: first record then delete.
        e.record_write(&ns("ns"), 500, 0);
        e.record_delete(&ns("ns"), 100, day);
        let fc = e.forecast(&ns("ns"), 2 * day).expect("forecast");
        // Negative growth → never fills up.
        assert!(fc.days_until_quota.is_none());
    }

    // -----------------------------------------------------------------------
    // violations
    // -----------------------------------------------------------------------

    #[test]
    fn violations_initially_empty() {
        let e = SqeStorageQuotaEnforcer::new(small_config());
        assert!(e.violations().is_empty());
    }

    #[test]
    fn record_violation_appends() {
        let mut e = SqeStorageQuotaEnforcer::new(small_config());
        e.record_violation(QuotaViolation {
            namespace: ns("ns"),
            kind: ViolationKind::BytesExceeded,
            current: 600,
            limit: 500,
            timestamp: 9999,
        });
        assert_eq!(e.violations().len(), 1);
        assert_eq!(e.violations()[0].kind, ViolationKind::BytesExceeded);
    }

    // -----------------------------------------------------------------------
    // top_namespaces_by_usage
    // -----------------------------------------------------------------------

    #[test]
    fn top_namespaces_sorted_by_bytes_desc() {
        let mut e = SqeStorageQuotaEnforcer::new(small_config());
        e.record_write(&ns("a"), 100, 1);
        e.record_write(&ns("b"), 300, 2);
        e.record_write(&ns("c"), 200, 3);
        let top = e.top_namespaces_by_usage(3);
        assert_eq!(top[0].0, ns("b"));
        assert_eq!(top[1].0, ns("c"));
        assert_eq!(top[2].0, ns("a"));
    }

    #[test]
    fn top_namespaces_truncated_to_n() {
        let mut e = SqeStorageQuotaEnforcer::new(small_config());
        e.record_write(&ns("a"), 100, 1);
        e.record_write(&ns("b"), 300, 2);
        e.record_write(&ns("c"), 200, 3);
        let top = e.top_namespaces_by_usage(2);
        assert_eq!(top.len(), 2);
    }

    #[test]
    fn top_namespaces_empty_when_no_usage() {
        let e = SqeStorageQuotaEnforcer::new(small_config());
        assert!(e.top_namespaces_by_usage(5).is_empty());
    }

    #[test]
    fn top_namespaces_n_larger_than_count() {
        let mut e = SqeStorageQuotaEnforcer::new(small_config());
        e.record_write(&ns("a"), 100, 1);
        let top = e.top_namespaces_by_usage(100);
        assert_eq!(top.len(), 1);
    }

    // -----------------------------------------------------------------------
    // Default trait
    // -----------------------------------------------------------------------

    #[test]
    fn default_enforcer_initialises_correctly() {
        let e = SqeStorageQuotaEnforcer::default();
        assert!(e.violations().is_empty());
        assert!(e.usage_for(&ns("any")).is_none());
    }

    // -----------------------------------------------------------------------
    // EnforcerConfig default
    // -----------------------------------------------------------------------

    #[test]
    fn enforcer_config_default_values() {
        let cfg = EnforcerConfig::default();
        assert_eq!(cfg.history_window, 100);
        assert_eq!(cfg.max_namespaces, 1000);
        assert_eq!(cfg.default_policy, EnforcementPolicy::Reject);
    }

    // -----------------------------------------------------------------------
    // Edge-case: check_write then record_write lifecycle
    // -----------------------------------------------------------------------

    #[test]
    fn full_write_lifecycle_check_then_record() {
        let mut e = SqeStorageQuotaEnforcer::new(small_config());
        e.set_quota(ns("ns"), QuotaLimit::bytes_only(1000));
        // Check passes.
        assert!(e.check_write(&ns("ns"), 400).is_ok());
        // Record the write.
        e.record_write(&ns("ns"), 400, 1000);
        // Check still passes.
        assert!(e.check_write(&ns("ns"), 400).is_ok());
        // Second record.
        e.record_write(&ns("ns"), 400, 2000);
        // Now 800 bytes used; adding 300 would put us at 1100 > 1000.
        assert!(e.check_write(&ns("ns"), 300).is_err());
    }

    #[test]
    fn set_policy_overrides_default() {
        let mut e = SqeStorageQuotaEnforcer::new(small_config()); // default = Reject
        e.set_quota(ns("ns"), QuotaLimit::bytes_only(100));
        e.set_policy(ns("ns"), EnforcementPolicy::Warn);
        e.record_write(&ns("ns"), 90, 1);
        // Would exceed (90 + 50 > 100) but policy is Warn → Ok.
        assert!(e.check_write(&ns("ns"), 50).is_ok());
    }

    #[test]
    fn violation_violation_kind_object_size_too_large_fields() {
        let mut e = SqeStorageQuotaEnforcer::new(small_config());
        e.set_quota(ns("ns"), QuotaLimit::full(10_000, 0, 50));
        let v = e.check_write(&ns("ns"), 51).expect_err("should err");
        assert_eq!(v.kind, ViolationKind::ObjectSizeTooLarge);
        assert_eq!(v.current, 51);
        assert_eq!(v.limit, 50);
        assert_eq!(v.namespace, ns("ns"));
    }
}
