//! Per-namespace storage quota enforcement with usage tracking and eviction triggers.
//!
//! `StorageQuotaManager` tracks allocated objects per namespace, enforces configurable
//! hard/soft byte and object-count limits, and provides multiple eviction strategies
//! (Oldest, LRU, LFU, SizeDescending) for reclaiming space.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Newtype wrapper for namespace identifiers (e.g. user ID, app ID).
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct QuotaNamespace(pub String);

impl QuotaNamespace {
    /// Create a new `QuotaNamespace` from any string-like value.
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }

    /// Return the inner string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for QuotaNamespace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

// ---------------------------------------------------------------------------
// Eviction strategy
// ---------------------------------------------------------------------------

/// Selects which objects are evicted first when a namespace is over quota.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SqmEvictionStrategy {
    /// Evict the object with the smallest `created_at` timestamp first.
    Oldest,
    /// Least Recently Used — evict the object with the smallest `last_accessed` timestamp.
    Lru,
    /// Least Frequently Used — evict the object with the smallest `access_count`;
    /// ties broken by smallest `last_accessed`.
    Lfu,
    /// Evict the largest object (by `size_bytes`) first; ties broken by object_id.
    SizeDescending,
}

// ---------------------------------------------------------------------------
// QuotaPolicy
// ---------------------------------------------------------------------------

/// Configuration for a single namespace's quota.
#[derive(Clone, Debug)]
pub struct QuotaPolicy {
    /// Hard byte ceiling; allocations that would exceed this are rejected.
    pub max_bytes: u64,
    /// Hard object-count ceiling.
    pub max_objects: u64,
    /// Fraction of `max_bytes` at which a `SoftLimitWarning` is emitted.
    /// Must be in `(0.0, 1.0]`; defaults to `0.8`.
    pub soft_limit_fraction: f64,
    /// Strategy used by `evict_candidates`.
    pub eviction_strategy: SqmEvictionStrategy,
}

impl QuotaPolicy {
    /// Create a policy with default soft-limit fraction (0.8) and LRU eviction.
    pub fn new(max_bytes: u64, max_objects: u64) -> Self {
        Self {
            max_bytes,
            max_objects,
            soft_limit_fraction: 0.8,
            eviction_strategy: SqmEvictionStrategy::Lru,
        }
    }

    /// Builder: set the soft-limit fraction.
    pub fn with_soft_limit_fraction(mut self, fraction: f64) -> Self {
        self.soft_limit_fraction = fraction;
        self
    }

    /// Builder: set the eviction strategy.
    pub fn with_eviction_strategy(mut self, strategy: SqmEvictionStrategy) -> Self {
        self.eviction_strategy = strategy;
        self
    }
}

// ---------------------------------------------------------------------------
// QuotaEntry
// ---------------------------------------------------------------------------

/// Live accounting entry for a registered namespace.
#[derive(Clone, Debug)]
pub struct SqmQuotaEntry {
    /// Namespace this entry belongs to.
    pub namespace: QuotaNamespace,
    /// Current total bytes allocated across all objects.
    pub bytes_used: u64,
    /// Current number of allocated objects.
    pub object_count: u64,
    /// Unix-epoch timestamp (seconds) of the most recent successful `allocate`.
    pub last_write: u64,
    /// The policy governing this namespace.
    pub policy: QuotaPolicy,
}

// ---------------------------------------------------------------------------
// ObjectRecord
// ---------------------------------------------------------------------------

/// Per-object tracking record.
#[derive(Clone, Debug)]
pub struct ObjectRecord {
    /// Unique identifier for this object.
    pub object_id: String,
    /// Namespace to which this object belongs.
    pub namespace: QuotaNamespace,
    /// Size of the object in bytes.
    pub size_bytes: u64,
    /// Unix-epoch timestamp when the object was created.
    pub created_at: u64,
    /// Unix-epoch timestamp of the most recent `access_object` call.
    pub last_accessed: u64,
    /// Total number of `access_object` calls.
    pub access_count: u64,
}

// ---------------------------------------------------------------------------
// QuotaViolation
// ---------------------------------------------------------------------------

/// Describes a quota-related event returned from `allocate`.
#[derive(Clone, Debug, PartialEq)]
pub enum SqmQuotaViolation {
    /// The namespace's hard byte limit would be exceeded.
    HardLimitExceeded {
        /// Namespace name string.
        namespace: String,
        /// Current usage before this allocation.
        used: u64,
        /// Configured hard limit.
        limit: u64,
    },
    /// The namespace's hard object-count limit would be exceeded.
    ObjectLimitExceeded {
        /// Namespace name string.
        namespace: String,
        /// Current object count before this allocation.
        count: u64,
        /// Configured hard object limit.
        limit: u64,
    },
    /// Usage has crossed the soft-limit threshold (warning, not an error).
    SoftLimitWarning {
        /// Namespace name string.
        namespace: String,
        /// Current fraction (bytes_used / max_bytes) after allocation.
        fraction: f64,
    },
}

// ---------------------------------------------------------------------------
// QuotaError
// ---------------------------------------------------------------------------

/// Errors returned by `StorageQuotaManager` operations.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum QuotaError {
    /// A namespace with this identifier is already registered.
    #[error("namespace already exists: {0}")]
    NamespaceAlreadyExists(String),
    /// No namespace with this identifier is registered.
    #[error("namespace not found: {0}")]
    NamespaceNotFound(String),
    /// No object with this identifier is tracked.
    #[error("object not found: {0}")]
    ObjectNotFound(String),
    /// An object with this identifier is already tracked.
    #[error("object already exists: {0}")]
    ObjectAlreadyExists(String),
    /// This allocation would exceed the global byte ceiling.
    #[error("global limit exceeded: used={used}, limit={limit}")]
    GlobalLimitExceeded {
        /// Bytes that would be used after this allocation.
        used: u64,
        /// Configured global limit.
        limit: u64,
    },
    /// Namespace-level hard limit exceeded (byte or object count).
    #[error("namespace hard limit exceeded in {namespace}")]
    HardLimitExceeded {
        /// Namespace name.
        namespace: String,
    },
}

// ---------------------------------------------------------------------------
// QuotaStats
// ---------------------------------------------------------------------------

/// Aggregate statistics snapshot for the whole manager.
#[derive(Clone, Debug)]
pub struct QuotaStats {
    /// Number of registered namespaces.
    pub namespace_count: usize,
    /// Sum of `bytes_used` across all namespaces.
    pub total_bytes_used: u64,
    /// Sum of `object_count` across all namespaces.
    pub total_objects: usize,
    /// `total_bytes_used / global_max_bytes` (0.0 when global_max_bytes == 0).
    pub global_utilization: f64,
    /// Namespaces where `usage_fraction ≥ soft_limit_fraction`.
    pub namespaces_at_soft_limit: usize,
    /// Namespaces where `usage_fraction ≥ 1.0`.
    pub namespaces_at_hard_limit: usize,
}

// ---------------------------------------------------------------------------
// StorageQuotaManager
// ---------------------------------------------------------------------------

/// Per-namespace storage quota enforcement manager.
///
/// Maintains a registry of namespaces with their policies and accounting entries,
/// plus a flat index of all tracked objects for O(1) lookup.
pub struct StorageQuotaManager {
    /// Namespace entries keyed by namespace identifier.
    pub entries: HashMap<QuotaNamespace, SqmQuotaEntry>,
    /// Object records keyed by object_id.
    pub objects: HashMap<String, ObjectRecord>,
    /// Global byte ceiling across all namespaces combined.
    pub global_max_bytes: u64,
}

impl StorageQuotaManager {
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    /// Create a new manager with the given global byte ceiling.
    ///
    /// The global ceiling is an additional guard on top of per-namespace limits.
    /// Pass `u64::MAX` to effectively disable it.
    pub fn new(global_max_bytes: u64) -> Self {
        Self {
            entries: HashMap::new(),
            objects: HashMap::new(),
            global_max_bytes,
        }
    }

    // -----------------------------------------------------------------------
    // Namespace management
    // -----------------------------------------------------------------------

    /// Register a new namespace with the given policy.
    ///
    /// Returns [`QuotaError::NamespaceAlreadyExists`] if `ns` is already registered.
    pub fn register_namespace(
        &mut self,
        ns: QuotaNamespace,
        policy: QuotaPolicy,
    ) -> Result<(), QuotaError> {
        if self.entries.contains_key(&ns) {
            return Err(QuotaError::NamespaceAlreadyExists(ns.0));
        }
        let entry = SqmQuotaEntry {
            namespace: ns.clone(),
            bytes_used: 0,
            object_count: 0,
            last_write: 0,
            policy,
        };
        self.entries.insert(ns, entry);
        Ok(())
    }

    /// Unregister a namespace and remove all objects that belong to it.
    ///
    /// Returns [`QuotaError::NamespaceNotFound`] if `ns` is not registered.
    pub fn unregister_namespace(&mut self, ns: &QuotaNamespace) -> Result<(), QuotaError> {
        if !self.entries.contains_key(ns) {
            return Err(QuotaError::NamespaceNotFound(ns.0.clone()));
        }
        // Collect and remove all objects belonging to this namespace.
        let to_remove: Vec<String> = self
            .objects
            .values()
            .filter(|o| &o.namespace == ns)
            .map(|o| o.object_id.clone())
            .collect();
        for oid in to_remove {
            self.objects.remove(&oid);
        }
        self.entries.remove(ns);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Object lifecycle
    // -----------------------------------------------------------------------

    /// Allocate a new object in the given namespace.
    ///
    /// Checks the global limit and the namespace's hard limits before inserting.
    /// Returns a (possibly empty) vector of warnings (e.g. soft-limit crossed).
    /// Returns `Err` if any hard limit or the global limit would be exceeded, or
    /// if the object already exists.
    pub fn allocate(
        &mut self,
        object_id: String,
        ns: &QuotaNamespace,
        size_bytes: u64,
        now: u64,
    ) -> Result<Vec<SqmQuotaViolation>, QuotaError> {
        // Existence check
        if self.objects.contains_key(&object_id) {
            return Err(QuotaError::ObjectAlreadyExists(object_id));
        }

        // Namespace must exist
        let entry = self
            .entries
            .get(ns)
            .ok_or_else(|| QuotaError::NamespaceNotFound(ns.0.clone()))?;

        // Global limit check
        let current_global = self.total_bytes_used();
        let new_global = current_global.saturating_add(size_bytes);
        if new_global > self.global_max_bytes {
            return Err(QuotaError::GlobalLimitExceeded {
                used: new_global,
                limit: self.global_max_bytes,
            });
        }

        // Namespace byte hard limit
        let new_bytes = entry.bytes_used.saturating_add(size_bytes);
        if new_bytes > entry.policy.max_bytes {
            return Err(QuotaError::HardLimitExceeded {
                namespace: ns.0.clone(),
            });
        }

        // Namespace object count hard limit
        let new_count = entry.object_count.saturating_add(1);
        if new_count > entry.policy.max_objects {
            return Err(QuotaError::HardLimitExceeded {
                namespace: ns.0.clone(),
            });
        }

        // Insert the object record
        let record = ObjectRecord {
            object_id: object_id.clone(),
            namespace: ns.clone(),
            size_bytes,
            created_at: now,
            last_accessed: now,
            access_count: 0,
        };
        self.objects.insert(object_id, record);

        // Update namespace accounting — re-borrow mutably
        let entry_mut = self
            .entries
            .get_mut(ns)
            .ok_or_else(|| QuotaError::NamespaceNotFound(ns.0.clone()))?;
        entry_mut.bytes_used = new_bytes;
        entry_mut.object_count = new_count;
        entry_mut.last_write = now;

        // Emit any warnings
        let mut warnings = Vec::new();
        let fraction = if entry_mut.policy.max_bytes > 0 {
            entry_mut.bytes_used as f64 / entry_mut.policy.max_bytes as f64
        } else {
            0.0
        };
        if fraction >= entry_mut.policy.soft_limit_fraction {
            warnings.push(SqmQuotaViolation::SoftLimitWarning {
                namespace: ns.0.clone(),
                fraction,
            });
        }

        Ok(warnings)
    }

    /// Remove an object and return the number of bytes freed.
    ///
    /// Updates the namespace accounting entry accordingly.
    pub fn deallocate(&mut self, object_id: &str, _now: u64) -> Result<u64, QuotaError> {
        let record = self
            .objects
            .remove(object_id)
            .ok_or_else(|| QuotaError::ObjectNotFound(object_id.to_string()))?;

        let freed = record.size_bytes;
        let ns = &record.namespace.clone();

        if let Some(entry) = self.entries.get_mut(ns) {
            entry.bytes_used = entry.bytes_used.saturating_sub(freed);
            entry.object_count = entry.object_count.saturating_sub(1);
        }

        Ok(freed)
    }

    /// Update `last_accessed` and increment `access_count` for LRU/LFU eviction.
    pub fn access_object(&mut self, object_id: &str, now: u64) -> Result<(), QuotaError> {
        let record = self
            .objects
            .get_mut(object_id)
            .ok_or_else(|| QuotaError::ObjectNotFound(object_id.to_string()))?;
        record.last_accessed = now;
        record.access_count = record.access_count.saturating_add(1);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Usage queries
    // -----------------------------------------------------------------------

    /// Return `bytes_used / max_bytes` for the namespace, or `None` if not registered.
    pub fn usage_fraction(&self, ns: &QuotaNamespace) -> Option<f64> {
        let entry = self.entries.get(ns)?;
        if entry.policy.max_bytes == 0 {
            Some(0.0)
        } else {
            Some(entry.bytes_used as f64 / entry.policy.max_bytes as f64)
        }
    }

    /// Return `true` if the namespace's usage fraction is ≥ 1.0 (at or over hard limit).
    pub fn needs_eviction(&self, ns: &QuotaNamespace) -> bool {
        self.usage_fraction(ns).is_some_and(|f| f >= 1.0)
    }

    /// Return a reference to the accounting entry for the namespace.
    pub fn namespace_usage(&self, ns: &QuotaNamespace) -> Option<&SqmQuotaEntry> {
        self.entries.get(ns)
    }

    /// Return all registered namespace identifiers, sorted alphabetically.
    pub fn all_namespaces(&self) -> Vec<&QuotaNamespace> {
        let mut keys: Vec<&QuotaNamespace> = self.entries.keys().collect();
        keys.sort();
        keys
    }

    /// Sum of `bytes_used` across all namespaces.
    pub fn total_bytes_used(&self) -> u64 {
        self.entries.values().map(|e| e.bytes_used).sum()
    }

    /// Sum of `object_count` across all namespaces.
    pub fn total_objects(&self) -> usize {
        self.entries.values().map(|e| e.object_count as usize).sum()
    }

    // -----------------------------------------------------------------------
    // Eviction
    // -----------------------------------------------------------------------

    /// Return a list of object IDs to evict according to the namespace's strategy.
    ///
    /// Objects are selected until removing them would free at least `target_free_bytes`.
    /// The returned list is never longer than the number of objects in the namespace.
    /// Objects are **not** removed by this call; use `force_evict` or call `deallocate`
    /// for each candidate.
    pub fn evict_candidates(&self, ns: &QuotaNamespace, target_free_bytes: u64) -> Vec<String> {
        let entry = match self.entries.get(ns) {
            Some(e) => e,
            None => return Vec::new(),
        };

        // Collect objects in this namespace
        let mut candidates: Vec<&ObjectRecord> = self
            .objects
            .values()
            .filter(|o| &o.namespace == ns)
            .collect();

        if candidates.is_empty() {
            return Vec::new();
        }

        // Sort according to the eviction strategy (ascending = evicted first)
        match entry.policy.eviction_strategy {
            SqmEvictionStrategy::Oldest => {
                candidates.sort_by_key(|o| (o.created_at, o.object_id.as_str().to_string()));
            }
            SqmEvictionStrategy::Lru => {
                candidates.sort_by_key(|o| (o.last_accessed, o.object_id.as_str().to_string()));
            }
            SqmEvictionStrategy::Lfu => {
                candidates.sort_by_key(|o| {
                    (
                        o.access_count,
                        o.last_accessed,
                        o.object_id.as_str().to_string(),
                    )
                });
            }
            SqmEvictionStrategy::SizeDescending => {
                // Largest first → sort descending by size, then by object_id for stability
                candidates.sort_by(|a, b| {
                    b.size_bytes
                        .cmp(&a.size_bytes)
                        .then_with(|| a.object_id.cmp(&b.object_id))
                });
            }
        }

        // Greedily pick until we have covered `target_free_bytes`
        let mut freed: u64 = 0;
        let mut result = Vec::new();
        for obj in candidates {
            if freed >= target_free_bytes {
                break;
            }
            freed = freed.saturating_add(obj.size_bytes);
            result.push(obj.object_id.clone());
        }

        result
    }

    /// Evict objects from the namespace until at least `target_free_bytes` are freed.
    ///
    /// Calls `evict_candidates` then `deallocate` on each candidate.
    /// Returns the total number of bytes actually freed.
    pub fn force_evict(
        &mut self,
        ns: &QuotaNamespace,
        target_free_bytes: u64,
        now: u64,
    ) -> Result<u64, QuotaError> {
        // We must not hold a borrow when calling deallocate, so collect first.
        let candidates = self.evict_candidates(ns, target_free_bytes);

        if candidates.is_empty() {
            // Return 0 if namespace exists but has no objects; error if unknown.
            if self.entries.contains_key(ns) {
                return Ok(0);
            } else {
                return Err(QuotaError::NamespaceNotFound(ns.0.clone()));
            }
        }

        let mut total_freed: u64 = 0;
        for oid in candidates {
            match self.deallocate(&oid, now) {
                Ok(freed) => total_freed = total_freed.saturating_add(freed),
                Err(QuotaError::ObjectNotFound(_)) => {
                    // Concurrent removal is acceptable; skip.
                }
                Err(e) => return Err(e),
            }
        }

        Ok(total_freed)
    }

    // -----------------------------------------------------------------------
    // Aggregate stats
    // -----------------------------------------------------------------------

    /// Return an aggregate statistics snapshot.
    pub fn stats(&self) -> QuotaStats {
        let namespace_count = self.entries.len();
        let total_bytes_used = self.total_bytes_used();
        let total_objects = self.total_objects();

        let global_utilization = if self.global_max_bytes > 0 {
            total_bytes_used as f64 / self.global_max_bytes as f64
        } else {
            0.0
        };

        let mut namespaces_at_soft_limit = 0usize;
        let mut namespaces_at_hard_limit = 0usize;

        for entry in self.entries.values() {
            if entry.policy.max_bytes == 0 {
                continue;
            }
            let frac = entry.bytes_used as f64 / entry.policy.max_bytes as f64;
            if frac >= 1.0 {
                namespaces_at_hard_limit += 1;
                // A namespace at the hard limit is also at the soft limit
                namespaces_at_soft_limit += 1;
            } else if frac >= entry.policy.soft_limit_fraction {
                namespaces_at_soft_limit += 1;
            }
        }

        QuotaStats {
            namespace_count,
            total_bytes_used,
            total_objects,
            global_utilization,
            namespaces_at_soft_limit,
            namespaces_at_hard_limit,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use crate::storage_quota_manager::{
        QuotaError, QuotaNamespace, QuotaPolicy, QuotaStats, SqmEvictionStrategy,
        SqmQuotaViolation, StorageQuotaManager,
    };

    fn make_ns(s: &str) -> QuotaNamespace {
        QuotaNamespace::new(s)
    }

    fn make_policy(max_bytes: u64, max_objects: u64) -> QuotaPolicy {
        QuotaPolicy::new(max_bytes, max_objects)
    }

    // --- Construction ---

    #[test]
    fn test_new_manager_empty() {
        let mgr = StorageQuotaManager::new(1024 * 1024);
        assert_eq!(mgr.total_bytes_used(), 0);
        assert_eq!(mgr.total_objects(), 0);
        assert!(mgr.all_namespaces().is_empty());
    }

    // --- Namespace registration ---

    #[test]
    fn test_register_namespace_ok() {
        let mut mgr = StorageQuotaManager::new(u64::MAX);
        let ns = make_ns("alice");
        let policy = make_policy(1000, 10);
        assert!(mgr.register_namespace(ns.clone(), policy).is_ok());
        assert!(mgr.namespace_usage(&ns).is_some());
    }

    #[test]
    fn test_register_namespace_duplicate_error() {
        let mut mgr = StorageQuotaManager::new(u64::MAX);
        let ns = make_ns("alice");
        mgr.register_namespace(ns.clone(), make_policy(1000, 10))
            .unwrap();
        let result = mgr.register_namespace(ns.clone(), make_policy(2000, 20));
        assert!(matches!(result, Err(QuotaError::NamespaceAlreadyExists(_))));
    }

    #[test]
    fn test_unregister_namespace_ok() {
        let mut mgr = StorageQuotaManager::new(u64::MAX);
        let ns = make_ns("bob");
        mgr.register_namespace(ns.clone(), make_policy(1000, 10))
            .unwrap();
        assert!(mgr.unregister_namespace(&ns).is_ok());
        assert!(mgr.namespace_usage(&ns).is_none());
    }

    #[test]
    fn test_unregister_namespace_not_found() {
        let mut mgr = StorageQuotaManager::new(u64::MAX);
        let ns = make_ns("ghost");
        let result = mgr.unregister_namespace(&ns);
        assert!(matches!(result, Err(QuotaError::NamespaceNotFound(_))));
    }

    #[test]
    fn test_unregister_removes_objects() {
        let mut mgr = StorageQuotaManager::new(u64::MAX);
        let ns = make_ns("carol");
        mgr.register_namespace(ns.clone(), make_policy(10_000, 100))
            .unwrap();
        mgr.allocate("obj-1".to_string(), &ns, 100, 1).unwrap();
        mgr.allocate("obj-2".to_string(), &ns, 200, 2).unwrap();
        assert_eq!(mgr.total_objects(), 2);
        mgr.unregister_namespace(&ns).unwrap();
        assert_eq!(mgr.total_objects(), 0);
        assert!(!mgr.objects.contains_key("obj-1"));
        assert!(!mgr.objects.contains_key("obj-2"));
    }

    // --- Allocation ---

    #[test]
    fn test_allocate_basic() {
        let mut mgr = StorageQuotaManager::new(u64::MAX);
        let ns = make_ns("dave");
        mgr.register_namespace(ns.clone(), make_policy(1000, 10))
            .unwrap();
        let warnings = mgr.allocate("obj-a".to_string(), &ns, 100, 10).unwrap();
        // 100/1000 = 10%, below soft limit 0.8
        assert!(warnings.is_empty());
        let entry = mgr.namespace_usage(&ns).unwrap();
        assert_eq!(entry.bytes_used, 100);
        assert_eq!(entry.object_count, 1);
        assert_eq!(entry.last_write, 10);
    }

    #[test]
    fn test_allocate_soft_limit_warning() {
        let mut mgr = StorageQuotaManager::new(u64::MAX);
        let ns = make_ns("eve");
        // soft_limit_fraction = 0.8, max_bytes = 100
        let policy = make_policy(100, 100).with_soft_limit_fraction(0.8);
        mgr.register_namespace(ns.clone(), policy).unwrap();
        // Allocate 85 bytes → 85%
        let warnings = mgr.allocate("obj".to_string(), &ns, 85, 1).unwrap();
        assert!(!warnings.is_empty());
        assert!(warnings
            .iter()
            .any(|w| matches!(w, SqmQuotaViolation::SoftLimitWarning { .. })));
    }

    #[test]
    fn test_allocate_hard_byte_limit_error() {
        let mut mgr = StorageQuotaManager::new(u64::MAX);
        let ns = make_ns("frank");
        mgr.register_namespace(ns.clone(), make_policy(100, 100))
            .unwrap();
        let result = mgr.allocate("big".to_string(), &ns, 200, 1);
        assert!(matches!(result, Err(QuotaError::HardLimitExceeded { .. })));
    }

    #[test]
    fn test_allocate_object_count_limit_error() {
        let mut mgr = StorageQuotaManager::new(u64::MAX);
        let ns = make_ns("grace");
        // max 2 objects, plenty of bytes
        mgr.register_namespace(ns.clone(), make_policy(100_000, 2))
            .unwrap();
        mgr.allocate("o1".to_string(), &ns, 10, 1).unwrap();
        mgr.allocate("o2".to_string(), &ns, 10, 2).unwrap();
        let result = mgr.allocate("o3".to_string(), &ns, 10, 3);
        assert!(matches!(result, Err(QuotaError::HardLimitExceeded { .. })));
    }

    #[test]
    fn test_allocate_duplicate_object_error() {
        let mut mgr = StorageQuotaManager::new(u64::MAX);
        let ns = make_ns("henry");
        mgr.register_namespace(ns.clone(), make_policy(10_000, 100))
            .unwrap();
        mgr.allocate("dup".to_string(), &ns, 10, 1).unwrap();
        let result = mgr.allocate("dup".to_string(), &ns, 10, 2);
        assert!(matches!(result, Err(QuotaError::ObjectAlreadyExists(_))));
    }

    #[test]
    fn test_allocate_unknown_namespace_error() {
        let mut mgr = StorageQuotaManager::new(u64::MAX);
        let ns = make_ns("unknown");
        let result = mgr.allocate("x".to_string(), &ns, 10, 1);
        assert!(matches!(result, Err(QuotaError::NamespaceNotFound(_))));
    }

    #[test]
    fn test_allocate_global_limit_error() {
        let mut mgr = StorageQuotaManager::new(50);
        let ns = make_ns("ivy");
        mgr.register_namespace(ns.clone(), make_policy(1_000_000, 1_000_000))
            .unwrap();
        let result = mgr.allocate("big".to_string(), &ns, 100, 1);
        assert!(matches!(
            result,
            Err(QuotaError::GlobalLimitExceeded { .. })
        ));
    }

    // --- Deallocation ---

    #[test]
    fn test_deallocate_ok() {
        let mut mgr = StorageQuotaManager::new(u64::MAX);
        let ns = make_ns("jack");
        mgr.register_namespace(ns.clone(), make_policy(10_000, 100))
            .unwrap();
        mgr.allocate("o1".to_string(), &ns, 300, 1).unwrap();
        let freed = mgr.deallocate("o1", 2).unwrap();
        assert_eq!(freed, 300);
        assert_eq!(mgr.namespace_usage(&ns).unwrap().bytes_used, 0);
        assert_eq!(mgr.namespace_usage(&ns).unwrap().object_count, 0);
    }

    #[test]
    fn test_deallocate_not_found() {
        let mut mgr = StorageQuotaManager::new(u64::MAX);
        assert!(matches!(
            mgr.deallocate("nope", 1),
            Err(QuotaError::ObjectNotFound(_))
        ));
    }

    #[test]
    fn test_deallocate_updates_total() {
        let mut mgr = StorageQuotaManager::new(u64::MAX);
        let ns = make_ns("kate");
        mgr.register_namespace(ns.clone(), make_policy(10_000, 100))
            .unwrap();
        mgr.allocate("a".to_string(), &ns, 100, 1).unwrap();
        mgr.allocate("b".to_string(), &ns, 200, 2).unwrap();
        assert_eq!(mgr.total_bytes_used(), 300);
        mgr.deallocate("a", 3).unwrap();
        assert_eq!(mgr.total_bytes_used(), 200);
    }

    // --- Access tracking ---

    #[test]
    fn test_access_object_updates_fields() {
        let mut mgr = StorageQuotaManager::new(u64::MAX);
        let ns = make_ns("leo");
        mgr.register_namespace(ns.clone(), make_policy(10_000, 100))
            .unwrap();
        mgr.allocate("obj".to_string(), &ns, 50, 1).unwrap();
        mgr.access_object("obj", 10).unwrap();
        mgr.access_object("obj", 20).unwrap();
        let rec = mgr.objects.get("obj").unwrap();
        assert_eq!(rec.last_accessed, 20);
        assert_eq!(rec.access_count, 2);
    }

    #[test]
    fn test_access_object_not_found() {
        let mut mgr = StorageQuotaManager::new(u64::MAX);
        assert!(matches!(
            mgr.access_object("missing", 1),
            Err(QuotaError::ObjectNotFound(_))
        ));
    }

    // --- Usage fraction ---

    #[test]
    fn test_usage_fraction_zero_when_empty() {
        let mut mgr = StorageQuotaManager::new(u64::MAX);
        let ns = make_ns("mia");
        mgr.register_namespace(ns.clone(), make_policy(1000, 10))
            .unwrap();
        assert_eq!(mgr.usage_fraction(&ns), Some(0.0));
    }

    #[test]
    fn test_usage_fraction_correct() {
        let mut mgr = StorageQuotaManager::new(u64::MAX);
        let ns = make_ns("ned");
        mgr.register_namespace(ns.clone(), make_policy(1000, 10))
            .unwrap();
        mgr.allocate("o".to_string(), &ns, 500, 1).unwrap();
        assert!((mgr.usage_fraction(&ns).unwrap() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_usage_fraction_none_for_unknown() {
        let mgr = StorageQuotaManager::new(u64::MAX);
        let ns = make_ns("nobody");
        assert_eq!(mgr.usage_fraction(&ns), None);
    }

    // --- needs_eviction ---

    #[test]
    fn test_needs_eviction_false_when_under_limit() {
        let mut mgr = StorageQuotaManager::new(u64::MAX);
        let ns = make_ns("olivia");
        mgr.register_namespace(ns.clone(), make_policy(1000, 10))
            .unwrap();
        mgr.allocate("o".to_string(), &ns, 500, 1).unwrap();
        assert!(!mgr.needs_eviction(&ns));
    }

    #[test]
    fn test_needs_eviction_true_when_at_limit() {
        let mut mgr = StorageQuotaManager::new(u64::MAX);
        let ns = make_ns("peter");
        mgr.register_namespace(ns.clone(), make_policy(100, 100))
            .unwrap();
        mgr.allocate("o".to_string(), &ns, 100, 1).unwrap();
        assert!(mgr.needs_eviction(&ns));
    }

    // --- all_namespaces sorted ---

    #[test]
    fn test_all_namespaces_sorted() {
        let mut mgr = StorageQuotaManager::new(u64::MAX);
        mgr.register_namespace(make_ns("z-ns"), make_policy(100, 10))
            .unwrap();
        mgr.register_namespace(make_ns("a-ns"), make_policy(100, 10))
            .unwrap();
        mgr.register_namespace(make_ns("m-ns"), make_policy(100, 10))
            .unwrap();
        let namespaces: Vec<String> = mgr.all_namespaces().iter().map(|n| n.0.clone()).collect();
        assert_eq!(namespaces, vec!["a-ns", "m-ns", "z-ns"]);
    }

    // --- Eviction candidates ---

    #[test]
    fn test_evict_candidates_oldest() {
        let mut mgr = StorageQuotaManager::new(u64::MAX);
        let ns = make_ns("quinn");
        let policy = make_policy(10_000, 100).with_eviction_strategy(SqmEvictionStrategy::Oldest);
        mgr.register_namespace(ns.clone(), policy).unwrap();
        mgr.allocate("old".to_string(), &ns, 100, 1).unwrap();
        mgr.allocate("new".to_string(), &ns, 100, 100).unwrap();
        let candidates = mgr.evict_candidates(&ns, 100);
        assert_eq!(candidates.first().map(String::as_str), Some("old"));
    }

    #[test]
    fn test_evict_candidates_lru() {
        let mut mgr = StorageQuotaManager::new(u64::MAX);
        let ns = make_ns("rose");
        let policy = make_policy(10_000, 100).with_eviction_strategy(SqmEvictionStrategy::Lru);
        mgr.register_namespace(ns.clone(), policy).unwrap();
        mgr.allocate("a".to_string(), &ns, 100, 1).unwrap();
        mgr.allocate("b".to_string(), &ns, 100, 1).unwrap();
        // Access "a" more recently
        mgr.access_object("a", 100).unwrap();
        let candidates = mgr.evict_candidates(&ns, 100);
        // "b" was last accessed earlier
        assert_eq!(candidates.first().map(String::as_str), Some("b"));
    }

    #[test]
    fn test_evict_candidates_lfu() {
        let mut mgr = StorageQuotaManager::new(u64::MAX);
        let ns = make_ns("sam");
        let policy = make_policy(10_000, 100).with_eviction_strategy(SqmEvictionStrategy::Lfu);
        mgr.register_namespace(ns.clone(), policy).unwrap();
        mgr.allocate("freq".to_string(), &ns, 100, 1).unwrap();
        mgr.allocate("rare".to_string(), &ns, 100, 1).unwrap();
        // Access "freq" many times
        for t in 2..12_u64 {
            mgr.access_object("freq", t).unwrap();
        }
        let candidates = mgr.evict_candidates(&ns, 100);
        // "rare" has access_count=0 → evicted first
        assert_eq!(candidates.first().map(String::as_str), Some("rare"));
    }

    #[test]
    fn test_evict_candidates_size_descending() {
        let mut mgr = StorageQuotaManager::new(u64::MAX);
        let ns = make_ns("tara");
        let policy =
            make_policy(10_000, 100).with_eviction_strategy(SqmEvictionStrategy::SizeDescending);
        mgr.register_namespace(ns.clone(), policy).unwrap();
        mgr.allocate("small".to_string(), &ns, 50, 1).unwrap();
        mgr.allocate("large".to_string(), &ns, 5000, 1).unwrap();
        mgr.allocate("medium".to_string(), &ns, 500, 1).unwrap();
        let candidates = mgr.evict_candidates(&ns, 1);
        assert_eq!(candidates.first().map(String::as_str), Some("large"));
    }

    #[test]
    fn test_evict_candidates_covers_target() {
        let mut mgr = StorageQuotaManager::new(u64::MAX);
        let ns = make_ns("ulrich");
        let policy = make_policy(10_000, 100).with_eviction_strategy(SqmEvictionStrategy::Oldest);
        mgr.register_namespace(ns.clone(), policy).unwrap();
        for i in 0..10_u64 {
            mgr.allocate(format!("obj-{i}"), &ns, 100, i).unwrap();
        }
        // Need to free 350 bytes → 4 objects (each 100 bytes covers after 4: 400 ≥ 350)
        let candidates = mgr.evict_candidates(&ns, 350);
        let total: u64 = candidates
            .iter()
            .map(|id| mgr.objects.get(id).map_or(0, |o| o.size_bytes))
            .sum();
        assert!(total >= 350, "Should cover at least 350 bytes, got {total}");
    }

    #[test]
    fn test_evict_candidates_empty_namespace() {
        let mut mgr = StorageQuotaManager::new(u64::MAX);
        let ns = make_ns("vera");
        mgr.register_namespace(ns.clone(), make_policy(1000, 10))
            .unwrap();
        assert!(mgr.evict_candidates(&ns, 100).is_empty());
    }

    #[test]
    fn test_evict_candidates_unknown_namespace() {
        let mgr = StorageQuotaManager::new(u64::MAX);
        let ns = make_ns("nobody");
        assert!(mgr.evict_candidates(&ns, 100).is_empty());
    }

    // --- force_evict ---

    #[test]
    fn test_force_evict_frees_bytes() {
        let mut mgr = StorageQuotaManager::new(u64::MAX);
        let ns = make_ns("will");
        let policy = make_policy(10_000, 100).with_eviction_strategy(SqmEvictionStrategy::Oldest);
        mgr.register_namespace(ns.clone(), policy).unwrap();
        mgr.allocate("a".to_string(), &ns, 200, 1).unwrap();
        mgr.allocate("b".to_string(), &ns, 200, 2).unwrap();
        mgr.allocate("c".to_string(), &ns, 200, 3).unwrap();
        let freed = mgr.force_evict(&ns, 300, 10).unwrap();
        assert!(freed >= 300, "freed={freed}");
    }

    #[test]
    fn test_force_evict_updates_accounting() {
        let mut mgr = StorageQuotaManager::new(u64::MAX);
        let ns = make_ns("xena");
        let policy =
            make_policy(10_000, 100).with_eviction_strategy(SqmEvictionStrategy::SizeDescending);
        mgr.register_namespace(ns.clone(), policy).unwrap();
        mgr.allocate("big".to_string(), &ns, 1000, 1).unwrap();
        mgr.allocate("small".to_string(), &ns, 50, 2).unwrap();
        let before = mgr.namespace_usage(&ns).unwrap().bytes_used;
        let freed = mgr.force_evict(&ns, 500, 10).unwrap();
        let after = mgr.namespace_usage(&ns).unwrap().bytes_used;
        assert_eq!(before - after, freed);
    }

    #[test]
    fn test_force_evict_unknown_namespace_error() {
        let mut mgr = StorageQuotaManager::new(u64::MAX);
        let ns = make_ns("nobody");
        let result = mgr.force_evict(&ns, 100, 1);
        assert!(matches!(result, Err(QuotaError::NamespaceNotFound(_))));
    }

    // --- stats ---

    #[test]
    fn test_stats_empty() {
        let mgr = StorageQuotaManager::new(1024);
        let s: QuotaStats = mgr.stats();
        assert_eq!(s.namespace_count, 0);
        assert_eq!(s.total_bytes_used, 0);
        assert_eq!(s.total_objects, 0);
        assert_eq!(s.global_utilization, 0.0);
        assert_eq!(s.namespaces_at_soft_limit, 0);
        assert_eq!(s.namespaces_at_hard_limit, 0);
    }

    #[test]
    fn test_stats_counts_correctly() {
        let mut mgr = StorageQuotaManager::new(10_000);
        let ns1 = make_ns("y-ns");
        let ns2 = make_ns("z-ns");
        mgr.register_namespace(ns1.clone(), make_policy(500, 10))
            .unwrap();
        mgr.register_namespace(ns2.clone(), make_policy(500, 10))
            .unwrap();
        // Bring ns1 to 90% (above soft limit 0.8)
        mgr.allocate("obj".to_string(), &ns1, 450, 1).unwrap();
        let s = mgr.stats();
        assert_eq!(s.namespace_count, 2);
        assert_eq!(s.total_bytes_used, 450);
        assert_eq!(s.total_objects, 1);
        assert_eq!(s.namespaces_at_soft_limit, 1);
        assert_eq!(s.namespaces_at_hard_limit, 0);
    }

    #[test]
    fn test_stats_hard_limit_count() {
        let mut mgr = StorageQuotaManager::new(u64::MAX);
        let ns = make_ns("zara");
        mgr.register_namespace(ns.clone(), make_policy(100, 100))
            .unwrap();
        mgr.allocate("full".to_string(), &ns, 100, 1).unwrap();
        let s = mgr.stats();
        assert_eq!(s.namespaces_at_hard_limit, 1);
        assert_eq!(s.namespaces_at_soft_limit, 1);
    }

    // --- QuotaNamespace helpers ---

    #[test]
    fn test_quota_namespace_as_str() {
        let ns = QuotaNamespace::new("test-ns");
        assert_eq!(ns.as_str(), "test-ns");
    }

    #[test]
    fn test_quota_namespace_display() {
        let ns = QuotaNamespace::new("display-me");
        assert_eq!(format!("{ns}"), "display-me");
    }

    // --- QuotaPolicy builder ---

    #[test]
    fn test_policy_builder_defaults() {
        let p = make_policy(1000, 20);
        assert!((p.soft_limit_fraction - 0.8).abs() < 1e-9);
        assert_eq!(p.eviction_strategy, SqmEvictionStrategy::Lru);
    }

    #[test]
    fn test_policy_builder_custom() {
        let p = make_policy(1000, 20)
            .with_soft_limit_fraction(0.5)
            .with_eviction_strategy(SqmEvictionStrategy::Oldest);
        assert!((p.soft_limit_fraction - 0.5).abs() < 1e-9);
        assert_eq!(p.eviction_strategy, SqmEvictionStrategy::Oldest);
    }

    // --- Global utilization ---

    #[test]
    fn test_global_utilization_in_stats() {
        let mut mgr = StorageQuotaManager::new(1000);
        let ns = make_ns("util-test");
        mgr.register_namespace(ns.clone(), make_policy(1000, 100))
            .unwrap();
        mgr.allocate("o".to_string(), &ns, 250, 1).unwrap();
        let s = mgr.stats();
        assert!((s.global_utilization - 0.25).abs() < 1e-9);
    }

    // --- Multiple namespaces isolation ---

    #[test]
    fn test_namespaces_are_isolated() {
        let mut mgr = StorageQuotaManager::new(u64::MAX);
        let ns_a = make_ns("iso-a");
        let ns_b = make_ns("iso-b");
        mgr.register_namespace(ns_a.clone(), make_policy(500, 10))
            .unwrap();
        mgr.register_namespace(ns_b.clone(), make_policy(500, 10))
            .unwrap();
        mgr.allocate("a1".to_string(), &ns_a, 300, 1).unwrap();
        mgr.allocate("b1".to_string(), &ns_b, 200, 1).unwrap();
        assert_eq!(mgr.namespace_usage(&ns_a).unwrap().bytes_used, 300);
        assert_eq!(mgr.namespace_usage(&ns_b).unwrap().bytes_used, 200);
    }

    // --- Re-use object id after deallocate ---

    #[test]
    fn test_reallocate_after_deallocate() {
        let mut mgr = StorageQuotaManager::new(u64::MAX);
        let ns = make_ns("reuse");
        mgr.register_namespace(ns.clone(), make_policy(10_000, 100))
            .unwrap();
        mgr.allocate("slot".to_string(), &ns, 100, 1).unwrap();
        mgr.deallocate("slot", 2).unwrap();
        // Should succeed now that "slot" is gone
        assert!(mgr.allocate("slot".to_string(), &ns, 100, 3).is_ok());
    }

    // --- Soft limit exactly at boundary ---

    #[test]
    fn test_soft_limit_exactly_at_boundary() {
        let mut mgr = StorageQuotaManager::new(u64::MAX);
        let ns = make_ns("boundary");
        let policy = make_policy(100, 100).with_soft_limit_fraction(0.9);
        mgr.register_namespace(ns.clone(), policy).unwrap();
        // 90 bytes = exactly at soft limit
        let warnings = mgr.allocate("o".to_string(), &ns, 90, 1).unwrap();
        assert!(!warnings.is_empty());
    }

    // --- total_objects accuracy ---

    #[test]
    fn test_total_objects_across_namespaces() {
        let mut mgr = StorageQuotaManager::new(u64::MAX);
        let ns1 = make_ns("t1");
        let ns2 = make_ns("t2");
        mgr.register_namespace(ns1.clone(), make_policy(10_000, 100))
            .unwrap();
        mgr.register_namespace(ns2.clone(), make_policy(10_000, 100))
            .unwrap();
        mgr.allocate("a".to_string(), &ns1, 10, 1).unwrap();
        mgr.allocate("b".to_string(), &ns1, 10, 2).unwrap();
        mgr.allocate("c".to_string(), &ns2, 10, 3).unwrap();
        assert_eq!(mgr.total_objects(), 3);
        mgr.deallocate("b", 4).unwrap();
        assert_eq!(mgr.total_objects(), 2);
    }
}
