//! Storage Quota Registry
//!
//! A centralized registry that tracks storage usage per user/namespace/project
//! and enforces configurable limits with grace periods and notifications.

use std::collections::HashMap;

/// Identifies the kind of quota entity being tracked.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum QuotaKind {
    /// Per-user storage quota
    User,
    /// Per-namespace storage quota
    Namespace,
    /// Per-project storage quota
    Project,
    /// Global (system-wide) storage quota
    Global,
}

/// A single quota entry tracking limits and usage for one entity.
#[derive(Clone, Debug)]
pub struct QuotaEntry {
    /// Unique identifier for this quota
    pub quota_id: u64,
    /// Kind of entity this quota applies to
    pub kind: QuotaKind,
    /// Name of the user/namespace/project
    pub name: String,
    /// Soft limit in bytes — triggers a warning when exceeded
    pub soft_limit_bytes: u64,
    /// Hard limit in bytes — triggers enforcement when exceeded (plus grace)
    pub hard_limit_bytes: u64,
    /// Current usage in bytes
    pub used_bytes: u64,
    /// Additional bytes allowed beyond hard_limit before blocking
    pub grace_bytes: u64,
}

impl QuotaEntry {
    /// Returns `used_bytes / hard_limit_bytes`, or 0.0 when hard_limit is zero.
    pub fn usage_ratio(&self) -> f64 {
        if self.hard_limit_bytes == 0 {
            0.0
        } else {
            self.used_bytes as f64 / self.hard_limit_bytes as f64
        }
    }

    /// Returns true when usage has reached or exceeded the soft limit.
    pub fn is_soft_exceeded(&self) -> bool {
        self.used_bytes >= self.soft_limit_bytes
    }

    /// Returns true when usage exceeds `hard_limit_bytes + grace_bytes`.
    pub fn is_hard_exceeded(&self) -> bool {
        self.used_bytes > self.hard_limit_bytes.saturating_add(self.grace_bytes)
    }

    /// Returns the number of bytes still available before the hard+grace ceiling.
    ///
    /// Saturates at 0 rather than underflowing.
    pub fn available_bytes(&self) -> u64 {
        self.hard_limit_bytes
            .saturating_add(self.grace_bytes)
            .saturating_sub(self.used_bytes)
    }
}

/// The type of quota violation detected.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ViolationType {
    /// Usage has met or exceeded the soft (warning) limit.
    SoftLimitExceeded,
    /// Usage has exceeded the hard limit plus any grace bytes.
    HardLimitExceeded,
}

/// Describes a single quota violation.
#[derive(Clone, Debug)]
pub struct QuotaViolation {
    /// ID of the quota that was violated
    pub quota_id: u64,
    /// Name of the entity whose quota was violated
    pub name: String,
    /// Kind of entity
    pub kind: QuotaKind,
    /// The specific type of violation
    pub violation_type: ViolationType,
}

/// Aggregate statistics across all registered quotas.
#[derive(Clone, Debug, PartialEq)]
pub struct RegistryStats {
    /// Total number of registered quotas
    pub total_quotas: usize,
    /// Number of quotas where the soft limit has been exceeded
    pub soft_exceeded: usize,
    /// Number of quotas where the hard limit has been exceeded
    pub hard_exceeded: usize,
    /// Sum of `used_bytes` across all quotas
    pub total_used_bytes: u64,
    /// Sum of `hard_limit_bytes` across all quotas
    pub total_capacity_bytes: u64,
}

/// Central registry for all storage quotas.
///
/// Tracks usage per user/namespace/project and enforces configurable limits
/// including soft limits (warnings), hard limits (enforcement), and grace
/// periods.
pub struct StorageQuotaRegistry {
    /// All registered quota entries, keyed by quota_id
    pub quotas: HashMap<u64, QuotaEntry>,
    /// Secondary index: (QuotaKind, name) → quota_id
    pub name_index: HashMap<(QuotaKind, String), u64>,
    /// Monotonically increasing ID counter
    pub next_quota_id: u64,
}

impl StorageQuotaRegistry {
    /// Creates a new, empty registry.
    pub fn new() -> Self {
        Self {
            quotas: HashMap::new(),
            name_index: HashMap::new(),
            next_quota_id: 1,
        }
    }

    /// Registers a new quota and returns its assigned `quota_id`.
    ///
    /// If a quota for the same `(kind, name)` pair already exists its entry
    /// is overwritten and the old entry is replaced under the new id.
    pub fn register(
        &mut self,
        kind: QuotaKind,
        name: &str,
        soft_limit_bytes: u64,
        hard_limit_bytes: u64,
        grace_bytes: u64,
    ) -> u64 {
        let quota_id = self.next_quota_id;
        self.next_quota_id += 1;

        // Remove any stale name-index entry that pointed to an old id
        let key = (kind, name.to_string());
        if let Some(old_id) = self.name_index.get(&key).copied() {
            self.quotas.remove(&old_id);
        }

        let entry = QuotaEntry {
            quota_id,
            kind,
            name: name.to_string(),
            soft_limit_bytes,
            hard_limit_bytes,
            used_bytes: 0,
            grace_bytes,
        };

        self.name_index.insert(key, quota_id);
        self.quotas.insert(quota_id, entry);
        quota_id
    }

    /// Overwrites the `used_bytes` field for the given quota.
    ///
    /// Returns `false` when no quota with that id exists.
    pub fn update_usage(&mut self, quota_id: u64, used_bytes: u64) -> bool {
        match self.quotas.get_mut(&quota_id) {
            Some(entry) => {
                entry.used_bytes = used_bytes;
                true
            }
            None => false,
        }
    }

    /// Adds `delta_bytes` to the current `used_bytes` using saturating addition.
    ///
    /// Returns `false` when no quota with that id exists.
    pub fn add_usage(&mut self, quota_id: u64, delta_bytes: u64) -> bool {
        match self.quotas.get_mut(&quota_id) {
            Some(entry) => {
                entry.used_bytes = entry.used_bytes.saturating_add(delta_bytes);
                true
            }
            None => false,
        }
    }

    /// Looks up a quota by its `(kind, name)` pair.
    pub fn get_by_name(&self, kind: QuotaKind, name: &str) -> Option<&QuotaEntry> {
        let id = self.name_index.get(&(kind, name.to_string())).copied()?;
        self.quotas.get(&id)
    }

    /// Looks up a quota by its numeric id.
    pub fn get(&self, quota_id: u64) -> Option<&QuotaEntry> {
        self.quotas.get(&quota_id)
    }

    /// Returns all current violations, sorted ascending by `quota_id`.
    ///
    /// Both soft and hard violations are reported; a single quota may produce
    /// two entries if both thresholds are breached.
    pub fn check_violations(&self) -> Vec<QuotaViolation> {
        let mut violations: Vec<QuotaViolation> = self
            .quotas
            .values()
            .flat_map(|entry| {
                let mut v = Vec::new();
                if entry.is_soft_exceeded() {
                    v.push(QuotaViolation {
                        quota_id: entry.quota_id,
                        name: entry.name.clone(),
                        kind: entry.kind,
                        violation_type: ViolationType::SoftLimitExceeded,
                    });
                }
                if entry.is_hard_exceeded() {
                    v.push(QuotaViolation {
                        quota_id: entry.quota_id,
                        name: entry.name.clone(),
                        kind: entry.kind,
                        violation_type: ViolationType::HardLimitExceeded,
                    });
                }
                v
            })
            .collect();

        violations.sort_by_key(|v| (v.quota_id, v.violation_type as u8));
        violations
    }

    /// Removes the quota entry with the given id.
    ///
    /// Also removes the corresponding `name_index` entry.  Returns `false`
    /// when no quota with that id exists.
    pub fn remove(&mut self, quota_id: u64) -> bool {
        match self.quotas.remove(&quota_id) {
            Some(entry) => {
                self.name_index.remove(&(entry.kind, entry.name));
                true
            }
            None => false,
        }
    }

    /// Computes aggregate statistics across all registered quotas.
    pub fn stats(&self) -> RegistryStats {
        let mut soft_exceeded = 0usize;
        let mut hard_exceeded = 0usize;
        let mut total_used_bytes = 0u64;
        let mut total_capacity_bytes = 0u64;

        for entry in self.quotas.values() {
            if entry.is_soft_exceeded() {
                soft_exceeded += 1;
            }
            if entry.is_hard_exceeded() {
                hard_exceeded += 1;
            }
            total_used_bytes = total_used_bytes.saturating_add(entry.used_bytes);
            total_capacity_bytes = total_capacity_bytes.saturating_add(entry.hard_limit_bytes);
        }

        RegistryStats {
            total_quotas: self.quotas.len(),
            soft_exceeded,
            hard_exceeded,
            total_used_bytes,
            total_capacity_bytes,
        }
    }
}

impl Default for StorageQuotaRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ──────────────────────────────────────────────────────────────

    fn make_registry() -> StorageQuotaRegistry {
        StorageQuotaRegistry::new()
    }

    // ── basic construction ────────────────────────────────────────────────────

    #[test]
    fn test_new_starts_empty() {
        let reg = make_registry();
        assert!(reg.quotas.is_empty());
        assert!(reg.name_index.is_empty());
        assert_eq!(reg.next_quota_id, 1);
    }

    // ── register ─────────────────────────────────────────────────────────────

    #[test]
    fn test_register_stores_quota() {
        let mut reg = make_registry();
        let id = reg.register(QuotaKind::User, "alice", 800, 1000, 0);
        let entry = reg.get(id).expect("quota should exist after register");
        assert_eq!(entry.quota_id, id);
        assert_eq!(entry.kind, QuotaKind::User);
        assert_eq!(entry.name, "alice");
        assert_eq!(entry.soft_limit_bytes, 800);
        assert_eq!(entry.hard_limit_bytes, 1000);
        assert_eq!(entry.used_bytes, 0);
        assert_eq!(entry.grace_bytes, 0);
    }

    #[test]
    fn test_register_returns_incrementing_ids() {
        let mut reg = make_registry();
        let id1 = reg.register(QuotaKind::User, "alice", 800, 1000, 0);
        let id2 = reg.register(QuotaKind::User, "bob", 800, 1000, 0);
        let id3 = reg.register(QuotaKind::Namespace, "team-a", 1000, 2000, 100);
        assert!(id1 < id2);
        assert!(id2 < id3);
        assert_eq!(id1 + 1, id2);
        assert_eq!(id2 + 1, id3);
    }

    #[test]
    fn test_register_indexes_by_kind_and_name() {
        let mut reg = make_registry();
        let id = reg.register(QuotaKind::Namespace, "team-x", 500, 1000, 0);
        let entry = reg
            .get_by_name(QuotaKind::Namespace, "team-x")
            .expect("should be found by name index");
        assert_eq!(entry.quota_id, id);
    }

    // ── update_usage ──────────────────────────────────────────────────────────

    #[test]
    fn test_update_usage_sets_used_bytes() {
        let mut reg = make_registry();
        let id = reg.register(QuotaKind::User, "alice", 800, 1000, 0);
        assert!(reg.update_usage(id, 500));
        assert_eq!(reg.get(id).expect("exists").used_bytes, 500);
    }

    #[test]
    fn test_update_usage_returns_false_for_unknown() {
        let mut reg = make_registry();
        assert!(!reg.update_usage(999, 100));
    }

    // ── add_usage ─────────────────────────────────────────────────────────────

    #[test]
    fn test_add_usage_increments_correctly() {
        let mut reg = make_registry();
        let id = reg.register(QuotaKind::Project, "proj-1", 800, 1000, 0);
        reg.update_usage(id, 300);
        assert!(reg.add_usage(id, 200));
        assert_eq!(reg.get(id).expect("exists").used_bytes, 500);
    }

    #[test]
    fn test_add_usage_saturates_at_u64_max() {
        let mut reg = make_registry();
        let id = reg.register(QuotaKind::Global, "global", 0, u64::MAX, 0);
        reg.update_usage(id, u64::MAX);
        assert!(reg.add_usage(id, 1));
        // saturating_add should stay at u64::MAX, not wrap
        assert_eq!(reg.get(id).expect("exists").used_bytes, u64::MAX);
    }

    // ── get_by_name ───────────────────────────────────────────────────────────

    #[test]
    fn test_get_by_name_some() {
        let mut reg = make_registry();
        reg.register(QuotaKind::User, "dave", 100, 200, 10);
        assert!(reg.get_by_name(QuotaKind::User, "dave").is_some());
    }

    #[test]
    fn test_get_by_name_none_for_wrong_kind() {
        let mut reg = make_registry();
        reg.register(QuotaKind::User, "dave", 100, 200, 10);
        assert!(reg.get_by_name(QuotaKind::Namespace, "dave").is_none());
    }

    #[test]
    fn test_get_by_name_none_for_unknown() {
        let reg = make_registry();
        assert!(reg.get_by_name(QuotaKind::User, "nobody").is_none());
    }

    // ── get ───────────────────────────────────────────────────────────────────

    #[test]
    fn test_get_some() {
        let mut reg = make_registry();
        let id = reg.register(QuotaKind::User, "eve", 100, 200, 0);
        assert!(reg.get(id).is_some());
    }

    #[test]
    fn test_get_none_for_unknown() {
        let reg = make_registry();
        assert!(reg.get(42).is_none());
    }

    // ── usage_ratio ───────────────────────────────────────────────────────────

    #[test]
    fn test_usage_ratio_computed_correctly() {
        let mut reg = make_registry();
        let id = reg.register(QuotaKind::User, "frank", 800, 1000, 0);
        reg.update_usage(id, 250);
        let ratio = reg.get(id).expect("exists").usage_ratio();
        assert!((ratio - 0.25).abs() < f64::EPSILON);
    }

    #[test]
    fn test_usage_ratio_zero_when_hard_limit_zero() {
        let entry = QuotaEntry {
            quota_id: 1,
            kind: QuotaKind::User,
            name: "test".to_string(),
            soft_limit_bytes: 0,
            hard_limit_bytes: 0,
            used_bytes: 100,
            grace_bytes: 0,
        };
        assert_eq!(entry.usage_ratio(), 0.0);
    }

    // ── is_soft_exceeded ──────────────────────────────────────────────────────

    #[test]
    fn test_is_soft_exceeded_true() {
        let mut reg = make_registry();
        let id = reg.register(QuotaKind::User, "grace", 500, 1000, 0);
        reg.update_usage(id, 500);
        assert!(reg.get(id).expect("exists").is_soft_exceeded());
    }

    #[test]
    fn test_is_soft_exceeded_false() {
        let mut reg = make_registry();
        let id = reg.register(QuotaKind::User, "grace", 500, 1000, 0);
        reg.update_usage(id, 499);
        assert!(!reg.get(id).expect("exists").is_soft_exceeded());
    }

    // ── is_hard_exceeded ─────────────────────────────────────────────────────

    #[test]
    fn test_is_hard_exceeded_without_grace() {
        let mut reg = make_registry();
        let id = reg.register(QuotaKind::User, "henry", 500, 1000, 0);
        // exactly at hard limit — NOT exceeded
        reg.update_usage(id, 1000);
        assert!(!reg.get(id).expect("exists").is_hard_exceeded());
        // one byte over — exceeded
        reg.update_usage(id, 1001);
        assert!(reg.get(id).expect("exists").is_hard_exceeded());
    }

    #[test]
    fn test_is_hard_exceeded_with_grace_bytes() {
        let mut reg = make_registry();
        let id = reg.register(QuotaKind::User, "iris", 500, 1000, 100);
        // within grace period — not exceeded
        reg.update_usage(id, 1050);
        assert!(!reg.get(id).expect("exists").is_hard_exceeded());
        // at exact grace ceiling — not exceeded
        reg.update_usage(id, 1100);
        assert!(!reg.get(id).expect("exists").is_hard_exceeded());
        // one byte beyond grace ceiling — exceeded
        reg.update_usage(id, 1101);
        assert!(reg.get(id).expect("exists").is_hard_exceeded());
    }

    // ── available_bytes ───────────────────────────────────────────────────────

    #[test]
    fn test_available_bytes_with_grace() {
        let mut reg = make_registry();
        let id = reg.register(QuotaKind::User, "jack", 500, 1000, 200);
        reg.update_usage(id, 800);
        // (1000 + 200) - 800 = 400
        assert_eq!(reg.get(id).expect("exists").available_bytes(), 400);
    }

    #[test]
    fn test_available_bytes_saturates_at_zero() {
        let mut reg = make_registry();
        let id = reg.register(QuotaKind::User, "kate", 500, 1000, 0);
        reg.update_usage(id, 1500);
        assert_eq!(reg.get(id).expect("exists").available_bytes(), 0);
    }

    // ── check_violations ─────────────────────────────────────────────────────

    #[test]
    fn test_check_violations_finds_soft_exceeded() {
        let mut reg = make_registry();
        let id = reg.register(QuotaKind::User, "leo", 500, 1000, 0);
        reg.update_usage(id, 600); // soft exceeded, hard not
        let violations = reg.check_violations();
        assert_eq!(violations.len(), 1);
        assert_eq!(
            violations[0].violation_type,
            ViolationType::SoftLimitExceeded
        );
        assert_eq!(violations[0].quota_id, id);
    }

    #[test]
    fn test_check_violations_finds_hard_exceeded() {
        let mut reg = make_registry();
        let id = reg.register(QuotaKind::User, "mia", 500, 1000, 0);
        reg.update_usage(id, 1001); // both soft and hard exceeded
        let violations = reg.check_violations();
        let types: Vec<ViolationType> = violations.iter().map(|v| v.violation_type).collect();
        assert!(types.contains(&ViolationType::SoftLimitExceeded));
        assert!(types.contains(&ViolationType::HardLimitExceeded));
    }

    #[test]
    fn test_check_violations_sorted_by_quota_id() {
        let mut reg = make_registry();
        // Register three quotas, all exceeding soft limit
        let id1 = reg.register(QuotaKind::User, "u1", 100, 1000, 0);
        let id2 = reg.register(QuotaKind::User, "u2", 100, 1000, 0);
        let id3 = reg.register(QuotaKind::User, "u3", 100, 1000, 0);
        reg.update_usage(id3, 200);
        reg.update_usage(id1, 200);
        reg.update_usage(id2, 200);
        let violations = reg.check_violations();
        assert!(!violations.is_empty());
        let ids: Vec<u64> = violations.iter().map(|v| v.quota_id).collect();
        let mut sorted = ids.clone();
        sorted.sort_unstable();
        assert_eq!(ids, sorted, "violations must be sorted by quota_id");
    }

    // ── remove ────────────────────────────────────────────────────────────────

    #[test]
    fn test_remove_returns_true_and_cleans_up() {
        let mut reg = make_registry();
        let id = reg.register(QuotaKind::User, "nina", 100, 200, 0);
        assert!(reg.remove(id));
        assert!(reg.get(id).is_none());
        assert!(reg.get_by_name(QuotaKind::User, "nina").is_none());
    }

    #[test]
    fn test_remove_returns_false_for_unknown() {
        let mut reg = make_registry();
        assert!(!reg.remove(9999));
    }

    // ── stats ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_stats_total_quotas_and_exceeded_counts() {
        let mut reg = make_registry();
        let id1 = reg.register(QuotaKind::User, "s1", 500, 1000, 0);
        let id2 = reg.register(QuotaKind::User, "s2", 500, 1000, 0);
        let _id3 = reg.register(QuotaKind::Namespace, "ns1", 500, 1000, 0);
        reg.update_usage(id1, 600); // soft only
        reg.update_usage(id2, 1001); // soft + hard
        let stats = reg.stats();
        assert_eq!(stats.total_quotas, 3);
        assert_eq!(stats.soft_exceeded, 2);
        assert_eq!(stats.hard_exceeded, 1);
    }

    #[test]
    fn test_stats_total_used_and_capacity_bytes() {
        let mut reg = make_registry();
        let id1 = reg.register(QuotaKind::User, "t1", 0, 1000, 0);
        let id2 = reg.register(QuotaKind::User, "t2", 0, 2000, 0);
        reg.update_usage(id1, 300);
        reg.update_usage(id2, 700);
        let stats = reg.stats();
        assert_eq!(stats.total_used_bytes, 1000);
        assert_eq!(stats.total_capacity_bytes, 3000);
    }
}
