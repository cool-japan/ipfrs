//! Object Lifecycle Manager — Policy-driven object lifecycle management
//!
//! Provides TTL expiry, retention rules, and automated tier transitions
//! for objects stored in IPFRS. Rules are evaluated in priority-descending order
//! and produce lifecycle actions that are applied atomically to managed objects.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

// ── Error type ───────────────────────────────────────────────────────────────

/// Errors produced by [`ObjectLifecycleManager`].
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum LifecycleError {
    /// Object with the given ID already exists in the registry.
    #[error("object already exists: {0}")]
    ObjectAlreadyExists(String),

    /// No object with the given ID found.
    #[error("object not found: {0}")]
    ObjectNotFound(String),

    /// Object is in an unexpected state for the requested operation.
    #[error("invalid state for object {object_id}: current={current}, expected={expected}")]
    InvalidState {
        object_id: String,
        current: String,
        expected: String,
    },

    /// Referenced rule name does not exist.
    #[error("rule not found: {0}")]
    RuleNotFound(String),
}

// ── LifecycleState ────────────────────────────────────────────────────────────

/// The current lifecycle state of a managed object.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum OlmLifecycleState {
    /// Object is live and accessible.
    Active,
    /// Object will expire at `expires_at` (Unix ms).
    Expiring { expires_at: u64 },
    /// Object has been moved to archive storage.
    Archived,
    /// Object is scheduled for deletion.
    MarkedForDeletion,
    /// Object has been logically deleted; may still linger until purge.
    Deleted,
}

impl OlmLifecycleState {
    /// Short name used for filtering / stats.
    pub fn name(&self) -> &'static str {
        match self {
            OlmLifecycleState::Active => "Active",
            OlmLifecycleState::Expiring { .. } => "Expiring",
            OlmLifecycleState::Archived => "Archived",
            OlmLifecycleState::MarkedForDeletion => "MarkedForDeletion",
            OlmLifecycleState::Deleted => "Deleted",
        }
    }
}

// ── RetentionRule ─────────────────────────────────────────────────────────────

/// A named rule that governs when and how objects transition or expire.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OlmRetentionRule {
    /// Unique human-readable name for this rule.
    pub name: String,

    /// Object must be at least this many milliseconds old for the rule to match.
    pub min_age_ms: u64,

    /// Objects older than this (ms) are auto-expired.  `None` = never auto-expire.
    pub max_age_ms: Option<u64>,

    /// Tier to transition to when the rule fires.  `None` = delete the object.
    pub transition_to: Option<String>,

    /// Higher priority wins when multiple rules match the same object.
    pub priority: u32,
}

// ── ManagedObject ─────────────────────────────────────────────────────────────

/// An object tracked by the lifecycle manager.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManagedObject {
    /// Unique identifier.
    pub object_id: String,

    /// Logical namespace / tenant.
    pub namespace: String,

    /// Size on disk in bytes.
    pub size_bytes: u64,

    /// Unix timestamp (ms) when the object was created.
    pub created_at: u64,

    /// Unix timestamp (ms) of the most recent access.
    pub last_accessed: u64,

    /// Current lifecycle state.
    pub state: OlmLifecycleState,

    /// Name of the storage tier this object currently lives on.
    pub current_tier: String,

    /// Arbitrary user-defined tags.
    pub tags: HashMap<String, String>,

    /// Name of the retention rule currently applied to this object, if any.
    pub retention_rule: Option<String>,
}

// ── LifecycleAction ───────────────────────────────────────────────────────────

/// An action produced by [`ObjectLifecycleManager::apply_rules`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum OlmLifecycleAction {
    /// Move object from one tier to another.
    Transition {
        object_id: String,
        from_tier: String,
        to_tier: String,
    },
    /// Permanently delete the object.
    Delete { object_id: String, reason: String },
    /// Move object to archive storage.
    Archive { object_id: String },
    /// Mark object as expiring at the given timestamp.
    SetExpiring { object_id: String, expires_at: u64 },
}

impl OlmLifecycleAction {
    /// Returns the object ID this action targets.
    pub fn object_id(&self) -> &str {
        match self {
            OlmLifecycleAction::Transition { object_id, .. } => object_id,
            OlmLifecycleAction::Delete { object_id, .. } => object_id,
            OlmLifecycleAction::Archive { object_id } => object_id,
            OlmLifecycleAction::SetExpiring { object_id, .. } => object_id,
        }
    }
}

// ── LifecycleStats ────────────────────────────────────────────────────────────

/// Aggregate statistics for the lifecycle manager.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OlmLifecycleStats {
    pub total_objects: usize,
    pub active_count: usize,
    pub expiring_count: usize,
    pub archived_count: usize,
    pub marked_for_deletion_count: usize,
    pub deleted_count: usize,
    pub total_active_bytes: u64,
    pub rules_count: usize,
}

// ── ObjectLifecycleManager ───────────────────────────────────────────────────

/// Policy-driven object lifecycle manager.
///
/// Objects are registered and then evaluated against a set of [`OlmRetentionRule`]s
/// every time [`apply_rules`](Self::apply_rules) is called.  The manager updates
/// object states internally and returns the list of actions that must be executed
/// by the caller (e.g., actually moving bytes between tiers or removing data).
pub struct ObjectLifecycleManager {
    objects: HashMap<String, ManagedObject>,
    rules: Vec<OlmRetentionRule>,
    default_ttl_ms: Option<u64>,
    /// Insertion-order counter used to break priority ties deterministically.
    rule_insertion_order: Vec<String>,
}

impl ObjectLifecycleManager {
    // ── Construction ──────────────────────────────────────────────────────────

    /// Create a new manager with an optional default TTL (milliseconds).
    ///
    /// When `default_ttl_ms` is `Some(n)`, objects that match no rule and are
    /// older than `n` ms are deleted automatically.
    pub fn new(default_ttl_ms: Option<u64>) -> Self {
        Self {
            objects: HashMap::new(),
            rules: Vec::new(),
            default_ttl_ms,
            rule_insertion_order: Vec::new(),
        }
    }

    // ── Rule management ───────────────────────────────────────────────────────

    /// Append a rule.  Rules are evaluated in **priority-descending** order;
    /// equal-priority rules are evaluated in insertion order (earlier first).
    pub fn add_rule(&mut self, rule: OlmRetentionRule) {
        self.rule_insertion_order.push(rule.name.clone());
        self.rules.push(rule);
    }

    // ── Object registration ───────────────────────────────────────────────────

    /// Register a new object.  Returns [`LifecycleError::ObjectAlreadyExists`]
    /// if an object with the same ID already exists.
    pub fn register_object(&mut self, object: ManagedObject) -> Result<(), LifecycleError> {
        if self.objects.contains_key(&object.object_id) {
            return Err(LifecycleError::ObjectAlreadyExists(
                object.object_id.clone(),
            ));
        }
        self.objects.insert(object.object_id.clone(), object);
        Ok(())
    }

    /// Remove and return a registered object.
    pub fn unregister_object(&mut self, object_id: &str) -> Result<ManagedObject, LifecycleError> {
        self.objects
            .remove(object_id)
            .ok_or_else(|| LifecycleError::ObjectNotFound(object_id.to_owned()))
    }

    /// Update `last_accessed` to `now`.
    ///
    /// Returns [`LifecycleError::InvalidState`] when the object is in
    /// `Deleted` state (deleted objects cannot be accessed).
    pub fn access_object(&mut self, object_id: &str, now: u64) -> Result<(), LifecycleError> {
        let obj = self
            .objects
            .get_mut(object_id)
            .ok_or_else(|| LifecycleError::ObjectNotFound(object_id.to_owned()))?;

        if obj.state == OlmLifecycleState::Deleted {
            return Err(LifecycleError::InvalidState {
                object_id: object_id.to_owned(),
                current: "Deleted".to_owned(),
                expected: "not Deleted".to_owned(),
            });
        }

        obj.last_accessed = now;
        Ok(())
    }

    // ── Core rule evaluation ─────────────────────────────────────────────────

    /// Evaluate all active objects against registered rules and the default TTL.
    ///
    /// Returns a list of actions.  The manager **also updates object states
    /// internally**, so callers do not need to call `execute_action` to keep
    /// the manager consistent — but they should use the returned actions to
    /// perform the actual data movement / deletion.
    pub fn apply_rules(&mut self, now: u64) -> Vec<OlmLifecycleAction> {
        // Build a sorted rule index: higher priority first, then insertion order.
        let sorted_indices = self.sorted_rule_indices();

        let mut actions: Vec<OlmLifecycleAction> = Vec::new();

        // Collect IDs first to avoid borrow conflicts.
        let ids: Vec<String> = self.objects.keys().cloned().collect();

        for id in ids {
            let obj = match self.objects.get(&id) {
                Some(o) => o,
                None => continue,
            };

            // Skip already-deleted objects.
            if obj.state == OlmLifecycleState::Deleted {
                continue;
            }

            let age_ms = now.saturating_sub(obj.created_at);
            let mut matched_rule: Option<usize> = None; // index into self.rules

            // Find highest-priority matching rule (age >= min_age_ms).
            for &rule_idx in &sorted_indices {
                let rule = &self.rules[rule_idx];
                if age_ms >= rule.min_age_ms {
                    matched_rule = Some(rule_idx);
                    break; // sorted by priority, first match wins
                }
            }

            if let Some(rule_idx) = matched_rule {
                let rule = self.rules[rule_idx].clone();

                // Only fire when max_age_ms is set and the object is old enough.
                if let Some(max_age) = rule.max_age_ms {
                    if age_ms >= max_age {
                        let action = self.build_rule_action(id.clone(), &rule);
                        self.apply_action_to_object(&id, &action);
                        actions.push(action);
                        continue;
                    }
                }

                // Rule matched but max_age not reached — record the rule association.
                if let Some(obj_mut) = self.objects.get_mut(&id) {
                    obj_mut.retention_rule = Some(rule.name.clone());
                }
            } else {
                // No rule matched; check default TTL.
                if let Some(ttl) = self.default_ttl_ms {
                    if age_ms >= ttl {
                        let action = OlmLifecycleAction::Delete {
                            object_id: id.clone(),
                            reason: "default_ttl_expired".to_owned(),
                        };
                        self.apply_action_to_object(&id, &action);
                        actions.push(action);
                    }
                }
            }
        }

        actions
    }

    /// Apply the state change for a single action.
    ///
    /// This is idempotent for re-applying the same action and can be called
    /// after the fact if needed.  Returns an error if the referenced object
    /// does not exist.
    pub fn execute_action(
        &mut self,
        action: &OlmLifecycleAction,
        _now: u64,
    ) -> Result<(), LifecycleError> {
        let id = action.object_id();
        if !self.objects.contains_key(id) {
            return Err(LifecycleError::ObjectNotFound(id.to_owned()));
        }
        self.apply_action_to_object(id, action);
        Ok(())
    }

    // ── Query helpers ─────────────────────────────────────────────────────────

    /// Return all objects whose state name matches `state_name`.
    ///
    /// Valid names: `"Active"`, `"Expiring"`, `"Archived"`, `"MarkedForDeletion"`,
    /// `"Deleted"`.
    pub fn objects_in_state(&self, state_name: &str) -> Vec<&ManagedObject> {
        self.objects
            .values()
            .filter(|o| o.state.name() == state_name)
            .collect()
    }

    /// Return all objects in `Expiring` state whose `expires_at` ≤ `timestamp`.
    pub fn objects_expiring_before(&self, timestamp: u64) -> Vec<&ManagedObject> {
        self.objects
            .values()
            .filter(|o| {
                matches!(
                    &o.state,
                    OlmLifecycleState::Expiring { expires_at }
                    if *expires_at <= timestamp
                )
            })
            .collect()
    }

    /// Return all objects on `tier`, sorted by `created_at` ascending.
    pub fn objects_by_tier(&self, tier: &str) -> Vec<&ManagedObject> {
        let mut result: Vec<&ManagedObject> = self
            .objects
            .values()
            .filter(|o| o.current_tier == tier)
            .collect();
        result.sort_by_key(|o| o.created_at);
        result
    }

    /// Remove logically deleted objects that were created more than 1 hour ago.
    ///
    /// Returns the number of objects removed.
    pub fn purge_deleted(&mut self, now: u64) -> usize {
        const ONE_HOUR_MS: u64 = 3_600_000;
        let threshold = now.saturating_sub(ONE_HOUR_MS);

        let to_remove: Vec<String> = self
            .objects
            .iter()
            .filter(|(_, o)| o.state == OlmLifecycleState::Deleted && o.created_at <= threshold)
            .map(|(id, _)| id.clone())
            .collect();

        let count = to_remove.len();
        for id in to_remove {
            self.objects.remove(&id);
        }
        count
    }

    /// Sum of `size_bytes` for all `Active` and `Expiring` objects.
    pub fn total_active_bytes(&self) -> u64 {
        self.objects
            .values()
            .filter(|o| {
                matches!(
                    o.state,
                    OlmLifecycleState::Active | OlmLifecycleState::Expiring { .. }
                )
            })
            .map(|o| o.size_bytes)
            .sum()
    }

    /// Aggregate statistics snapshot.
    pub fn stats(&self) -> OlmLifecycleStats {
        let mut s = OlmLifecycleStats {
            total_objects: self.objects.len(),
            rules_count: self.rules.len(),
            ..Default::default()
        };

        for obj in self.objects.values() {
            match &obj.state {
                OlmLifecycleState::Active => {
                    s.active_count += 1;
                    s.total_active_bytes += obj.size_bytes;
                }
                OlmLifecycleState::Expiring { .. } => {
                    s.expiring_count += 1;
                    s.total_active_bytes += obj.size_bytes;
                }
                OlmLifecycleState::Archived => s.archived_count += 1,
                OlmLifecycleState::MarkedForDeletion => s.marked_for_deletion_count += 1,
                OlmLifecycleState::Deleted => s.deleted_count += 1,
            }
        }

        s
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /// Returns rule indices sorted: higher priority first, then by insertion order.
    fn sorted_rule_indices(&self) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..self.rules.len()).collect();
        indices.sort_by(|&a, &b| {
            let pa = self.rules[a].priority;
            let pb = self.rules[b].priority;
            if pa != pb {
                return pb.cmp(&pa); // descending priority
            }
            // Equal priority → earlier insertion first.
            let ia = self
                .rule_insertion_order
                .iter()
                .position(|n| n == &self.rules[a].name)
                .unwrap_or(usize::MAX);
            let ib = self
                .rule_insertion_order
                .iter()
                .position(|n| n == &self.rules[b].name)
                .unwrap_or(usize::MAX);
            ia.cmp(&ib)
        });
        indices
    }

    /// Build the appropriate action for a rule firing on an object.
    fn build_rule_action(&self, object_id: String, rule: &OlmRetentionRule) -> OlmLifecycleAction {
        match &rule.transition_to {
            Some(to_tier) => {
                let from_tier = self
                    .objects
                    .get(&object_id)
                    .map(|o| o.current_tier.clone())
                    .unwrap_or_default();
                OlmLifecycleAction::Transition {
                    object_id,
                    from_tier,
                    to_tier: to_tier.clone(),
                }
            }
            None => OlmLifecycleAction::Delete {
                object_id,
                reason: format!("rule:{}", rule.name),
            },
        }
    }

    /// Apply the state change implied by an action directly to the object map.
    fn apply_action_to_object(&mut self, object_id: &str, action: &OlmLifecycleAction) {
        let obj = match self.objects.get_mut(object_id) {
            Some(o) => o,
            None => return,
        };

        match action {
            OlmLifecycleAction::Transition { to_tier, .. } => {
                obj.current_tier = to_tier.clone();
                // Keep state Active unless it was already something else.
                if obj.state == OlmLifecycleState::Active {
                    // state remains Active
                }
            }
            OlmLifecycleAction::Delete { .. } => {
                obj.state = OlmLifecycleState::Deleted;
            }
            OlmLifecycleAction::Archive { .. } => {
                obj.state = OlmLifecycleState::Archived;
            }
            OlmLifecycleAction::SetExpiring { expires_at, .. } => {
                obj.state = OlmLifecycleState::Expiring {
                    expires_at: *expires_at,
                };
            }
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::{
        LifecycleError, ManagedObject, ObjectLifecycleManager, OlmLifecycleAction,
        OlmLifecycleState, OlmRetentionRule,
    };
    use std::collections::HashMap;

    // ── Fixtures ──────────────────────────────────────────────────────────────

    fn obj(id: &str, created_at: u64) -> ManagedObject {
        ManagedObject {
            object_id: id.to_owned(),
            namespace: "test".to_owned(),
            size_bytes: 1024,
            created_at,
            last_accessed: created_at,
            state: OlmLifecycleState::Active,
            current_tier: "hot".to_owned(),
            tags: HashMap::new(),
            retention_rule: None,
        }
    }

    fn rule(
        name: &str,
        min_age_ms: u64,
        max_age_ms: Option<u64>,
        transition_to: Option<&str>,
        priority: u32,
    ) -> OlmRetentionRule {
        OlmRetentionRule {
            name: name.to_owned(),
            min_age_ms,
            max_age_ms,
            transition_to: transition_to.map(|s| s.to_owned()),
            priority,
        }
    }

    // ── new / basic construction ──────────────────────────────────────────────

    #[test]
    fn test_new_no_ttl() {
        let mgr = ObjectLifecycleManager::new(None);
        let s = mgr.stats();
        assert_eq!(s.total_objects, 0);
        assert_eq!(s.rules_count, 0);
    }

    #[test]
    fn test_new_with_ttl() {
        let mgr = ObjectLifecycleManager::new(Some(60_000));
        let s = mgr.stats();
        assert_eq!(s.total_objects, 0);
    }

    // ── register / unregister ─────────────────────────────────────────────────

    #[test]
    fn test_register_object_ok() {
        let mut mgr = ObjectLifecycleManager::new(None);
        assert!(mgr.register_object(obj("a", 0)).is_ok());
        assert_eq!(mgr.stats().total_objects, 1);
    }

    #[test]
    fn test_register_duplicate_error() {
        let mut mgr = ObjectLifecycleManager::new(None);
        mgr.register_object(obj("a", 0)).unwrap();
        let err = mgr.register_object(obj("a", 0)).unwrap_err();
        assert!(matches!(err, LifecycleError::ObjectAlreadyExists(id) if id == "a"));
    }

    #[test]
    fn test_unregister_ok() {
        let mut mgr = ObjectLifecycleManager::new(None);
        mgr.register_object(obj("a", 0)).unwrap();
        let removed = mgr.unregister_object("a").unwrap();
        assert_eq!(removed.object_id, "a");
        assert_eq!(mgr.stats().total_objects, 0);
    }

    #[test]
    fn test_unregister_not_found() {
        let mut mgr = ObjectLifecycleManager::new(None);
        let err = mgr.unregister_object("missing").unwrap_err();
        assert!(matches!(err, LifecycleError::ObjectNotFound(_)));
    }

    // ── access_object ─────────────────────────────────────────────────────────

    #[test]
    fn test_access_updates_last_accessed() {
        let mut mgr = ObjectLifecycleManager::new(None);
        mgr.register_object(obj("a", 100)).unwrap();
        mgr.access_object("a", 999).unwrap();
        let o = mgr.unregister_object("a").unwrap();
        assert_eq!(o.last_accessed, 999);
    }

    #[test]
    fn test_access_deleted_returns_error() {
        let mut mgr = ObjectLifecycleManager::new(None);
        let mut o = obj("a", 0);
        o.state = OlmLifecycleState::Deleted;
        mgr.register_object(o).unwrap();
        let err = mgr.access_object("a", 1).unwrap_err();
        assert!(matches!(err, LifecycleError::InvalidState { .. }));
    }

    #[test]
    fn test_access_not_found() {
        let mut mgr = ObjectLifecycleManager::new(None);
        let err = mgr.access_object("nope", 1).unwrap_err();
        assert!(matches!(err, LifecycleError::ObjectNotFound(_)));
    }

    // ── apply_rules — default TTL ─────────────────────────────────────────────

    #[test]
    fn test_default_ttl_fires_delete() {
        let mut mgr = ObjectLifecycleManager::new(Some(1000));
        mgr.register_object(obj("a", 0)).unwrap();
        let actions = mgr.apply_rules(2000);
        assert_eq!(actions.len(), 1);
        assert!(matches!(
            &actions[0],
            OlmLifecycleAction::Delete { object_id, reason }
            if object_id == "a" && reason == "default_ttl_expired"
        ));
    }

    #[test]
    fn test_default_ttl_not_fired_when_young() {
        let mut mgr = ObjectLifecycleManager::new(Some(5000));
        mgr.register_object(obj("a", 0)).unwrap();
        let actions = mgr.apply_rules(100);
        assert!(actions.is_empty());
    }

    #[test]
    fn test_no_default_ttl_no_action() {
        let mut mgr = ObjectLifecycleManager::new(None);
        mgr.register_object(obj("a", 0)).unwrap();
        let actions = mgr.apply_rules(999_999);
        assert!(actions.is_empty());
    }

    // ── apply_rules — rule-based transitions ──────────────────────────────────

    #[test]
    fn test_rule_transition_fires() {
        let mut mgr = ObjectLifecycleManager::new(None);
        mgr.add_rule(rule("to_warm", 100, Some(500), Some("warm"), 10));
        mgr.register_object(obj("a", 0)).unwrap();
        let actions = mgr.apply_rules(600);
        assert_eq!(actions.len(), 1);
        assert!(matches!(
            &actions[0],
            OlmLifecycleAction::Transition { object_id, to_tier, .. }
            if object_id == "a" && to_tier == "warm"
        ));
    }

    #[test]
    fn test_rule_delete_fires_when_no_transition() {
        let mut mgr = ObjectLifecycleManager::new(None);
        mgr.add_rule(rule("delete_old", 0, Some(100), None, 5));
        mgr.register_object(obj("a", 0)).unwrap();
        let actions = mgr.apply_rules(200);
        assert_eq!(actions.len(), 1);
        assert!(matches!(
            &actions[0],
            OlmLifecycleAction::Delete { object_id, .. }
            if object_id == "a"
        ));
    }

    #[test]
    fn test_rule_not_fired_before_max_age() {
        let mut mgr = ObjectLifecycleManager::new(None);
        mgr.add_rule(rule("to_cold", 0, Some(10_000), Some("cold"), 1));
        mgr.register_object(obj("a", 0)).unwrap();
        let actions = mgr.apply_rules(5_000);
        assert!(actions.is_empty());
    }

    #[test]
    fn test_rule_not_matched_before_min_age() {
        let mut mgr = ObjectLifecycleManager::new(None);
        mgr.add_rule(rule("old_only", 10_000, Some(20_000), Some("cold"), 1));
        mgr.register_object(obj("a", 0)).unwrap();
        // age = 5000 < min_age_ms = 10000 → rule doesn't match at all
        let actions = mgr.apply_rules(5_000);
        assert!(actions.is_empty());
    }

    // ── apply_rules — priority ordering ───────────────────────────────────────

    #[test]
    fn test_higher_priority_rule_wins() {
        let mut mgr = ObjectLifecycleManager::new(None);
        // Lower priority rule: delete
        mgr.add_rule(rule("delete_low", 0, Some(100), None, 1));
        // Higher priority rule: transition
        mgr.add_rule(rule("transit_high", 0, Some(100), Some("warm"), 10));
        mgr.register_object(obj("a", 0)).unwrap();
        let actions = mgr.apply_rules(200);
        assert_eq!(actions.len(), 1);
        assert!(matches!(
            &actions[0],
            OlmLifecycleAction::Transition { to_tier, .. }
            if to_tier == "warm"
        ));
    }

    #[test]
    fn test_equal_priority_insertion_order() {
        let mut mgr = ObjectLifecycleManager::new(None);
        // Both same priority — first inserted wins
        mgr.add_rule(rule("first", 0, Some(100), Some("warm"), 5));
        mgr.add_rule(rule("second", 0, Some(100), Some("cold"), 5));
        mgr.register_object(obj("a", 0)).unwrap();
        let actions = mgr.apply_rules(200);
        assert_eq!(actions.len(), 1);
        assert!(matches!(
            &actions[0],
            OlmLifecycleAction::Transition { to_tier, .. }
            if to_tier == "warm"
        ));
    }

    // ── apply_rules — state updates ───────────────────────────────────────────

    #[test]
    fn test_apply_rules_marks_object_deleted() {
        let mut mgr = ObjectLifecycleManager::new(Some(1000));
        mgr.register_object(obj("a", 0)).unwrap();
        mgr.apply_rules(2000);
        let objs = mgr.objects_in_state("Deleted");
        assert_eq!(objs.len(), 1);
        assert_eq!(objs[0].object_id, "a");
    }

    #[test]
    fn test_deleted_object_skipped_on_second_run() {
        let mut mgr = ObjectLifecycleManager::new(Some(500));
        mgr.register_object(obj("a", 0)).unwrap();
        let a1 = mgr.apply_rules(1000);
        let a2 = mgr.apply_rules(2000);
        assert_eq!(a1.len(), 1);
        assert!(a2.is_empty(), "Deleted objects must be skipped");
    }

    // ── execute_action ────────────────────────────────────────────────────────

    #[test]
    fn test_execute_action_delete() {
        let mut mgr = ObjectLifecycleManager::new(None);
        mgr.register_object(obj("a", 0)).unwrap();
        let action = OlmLifecycleAction::Delete {
            object_id: "a".to_owned(),
            reason: "manual".to_owned(),
        };
        mgr.execute_action(&action, 100).unwrap();
        assert_eq!(mgr.objects_in_state("Deleted").len(), 1);
    }

    #[test]
    fn test_execute_action_archive() {
        let mut mgr = ObjectLifecycleManager::new(None);
        mgr.register_object(obj("a", 0)).unwrap();
        let action = OlmLifecycleAction::Archive {
            object_id: "a".to_owned(),
        };
        mgr.execute_action(&action, 100).unwrap();
        assert_eq!(mgr.objects_in_state("Archived").len(), 1);
    }

    #[test]
    fn test_execute_action_set_expiring() {
        let mut mgr = ObjectLifecycleManager::new(None);
        mgr.register_object(obj("a", 0)).unwrap();
        let action = OlmLifecycleAction::SetExpiring {
            object_id: "a".to_owned(),
            expires_at: 5000,
        };
        mgr.execute_action(&action, 100).unwrap();
        assert_eq!(mgr.objects_in_state("Expiring").len(), 1);
    }

    #[test]
    fn test_execute_action_transition_updates_tier() {
        let mut mgr = ObjectLifecycleManager::new(None);
        mgr.register_object(obj("a", 0)).unwrap();
        let action = OlmLifecycleAction::Transition {
            object_id: "a".to_owned(),
            from_tier: "hot".to_owned(),
            to_tier: "cold".to_owned(),
        };
        mgr.execute_action(&action, 100).unwrap();
        let tiers = mgr.objects_by_tier("cold");
        assert_eq!(tiers.len(), 1);
    }

    #[test]
    fn test_execute_action_not_found() {
        let mut mgr = ObjectLifecycleManager::new(None);
        let action = OlmLifecycleAction::Delete {
            object_id: "ghost".to_owned(),
            reason: "test".to_owned(),
        };
        let err = mgr.execute_action(&action, 0).unwrap_err();
        assert!(matches!(err, LifecycleError::ObjectNotFound(_)));
    }

    // ── objects_in_state ──────────────────────────────────────────────────────

    #[test]
    fn test_objects_in_state_active() {
        let mut mgr = ObjectLifecycleManager::new(None);
        mgr.register_object(obj("a", 0)).unwrap();
        mgr.register_object(obj("b", 0)).unwrap();
        assert_eq!(mgr.objects_in_state("Active").len(), 2);
        assert_eq!(mgr.objects_in_state("Deleted").len(), 0);
    }

    #[test]
    fn test_objects_in_state_unknown_returns_empty() {
        let mut mgr = ObjectLifecycleManager::new(None);
        mgr.register_object(obj("a", 0)).unwrap();
        assert!(mgr.objects_in_state("NonExistent").is_empty());
    }

    // ── objects_expiring_before ───────────────────────────────────────────────

    #[test]
    fn test_objects_expiring_before() {
        let mut mgr = ObjectLifecycleManager::new(None);
        mgr.register_object(obj("a", 0)).unwrap();
        mgr.register_object(obj("b", 0)).unwrap();
        let a1 = OlmLifecycleAction::SetExpiring {
            object_id: "a".to_owned(),
            expires_at: 100,
        };
        let a2 = OlmLifecycleAction::SetExpiring {
            object_id: "b".to_owned(),
            expires_at: 9999,
        };
        mgr.execute_action(&a1, 0).unwrap();
        mgr.execute_action(&a2, 0).unwrap();

        let expiring = mgr.objects_expiring_before(500);
        assert_eq!(expiring.len(), 1);
        assert_eq!(expiring[0].object_id, "a");
    }

    #[test]
    fn test_objects_expiring_at_exact_boundary() {
        let mut mgr = ObjectLifecycleManager::new(None);
        mgr.register_object(obj("a", 0)).unwrap();
        let action = OlmLifecycleAction::SetExpiring {
            object_id: "a".to_owned(),
            expires_at: 100,
        };
        mgr.execute_action(&action, 0).unwrap();
        // Boundary: expires_at == timestamp → should be included (≤)
        assert_eq!(mgr.objects_expiring_before(100).len(), 1);
        assert_eq!(mgr.objects_expiring_before(99).len(), 0);
    }

    // ── objects_by_tier ───────────────────────────────────────────────────────

    #[test]
    fn test_objects_by_tier_sorted_by_created_at() {
        let mut mgr = ObjectLifecycleManager::new(None);
        mgr.register_object(obj("c", 300)).unwrap();
        mgr.register_object(obj("a", 100)).unwrap();
        mgr.register_object(obj("b", 200)).unwrap();
        let result = mgr.objects_by_tier("hot");
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].object_id, "a");
        assert_eq!(result[1].object_id, "b");
        assert_eq!(result[2].object_id, "c");
    }

    #[test]
    fn test_objects_by_tier_filter_by_tier_name() {
        let mut mgr = ObjectLifecycleManager::new(None);
        let mut warm_obj = obj("w", 0);
        warm_obj.current_tier = "warm".to_owned();
        mgr.register_object(obj("h", 0)).unwrap();
        mgr.register_object(warm_obj).unwrap();
        assert_eq!(mgr.objects_by_tier("hot").len(), 1);
        assert_eq!(mgr.objects_by_tier("warm").len(), 1);
        assert_eq!(mgr.objects_by_tier("cold").len(), 0);
    }

    // ── purge_deleted ─────────────────────────────────────────────────────────

    #[test]
    fn test_purge_deleted_removes_old_deleted_objects() {
        let mut mgr = ObjectLifecycleManager::new(None);
        let mut d = obj("a", 0); // created_at=0; with now=4_000_000, age > 1 hour
        d.state = OlmLifecycleState::Deleted;
        mgr.register_object(d).unwrap();
        let count = mgr.purge_deleted(4_000_000);
        assert_eq!(count, 1);
        assert_eq!(mgr.stats().total_objects, 0);
    }

    #[test]
    fn test_purge_deleted_keeps_recent_deleted_objects() {
        let mut mgr = ObjectLifecycleManager::new(None);
        let now = 3_600_000_u64; // exactly 1 hour
        let mut d = obj("a", now - 100); // created 100ms ago → not old enough
        d.state = OlmLifecycleState::Deleted;
        mgr.register_object(d).unwrap();
        let count = mgr.purge_deleted(now);
        assert_eq!(count, 0);
        assert_eq!(mgr.stats().total_objects, 1);
    }

    #[test]
    fn test_purge_deleted_ignores_active_objects() {
        let mut mgr = ObjectLifecycleManager::new(None);
        mgr.register_object(obj("a", 0)).unwrap(); // Active, old
        let count = mgr.purge_deleted(9_999_999);
        assert_eq!(count, 0);
        assert_eq!(mgr.stats().total_objects, 1);
    }

    // ── total_active_bytes ────────────────────────────────────────────────────

    #[test]
    fn test_total_active_bytes_sums_active_and_expiring() {
        let mut mgr = ObjectLifecycleManager::new(None);
        let mut o1 = obj("a", 0);
        o1.size_bytes = 500;
        let mut o2 = obj("b", 0);
        o2.size_bytes = 300;
        o2.state = OlmLifecycleState::Expiring { expires_at: 9999 };
        let mut o3 = obj("c", 0);
        o3.size_bytes = 200;
        o3.state = OlmLifecycleState::Archived;
        mgr.register_object(o1).unwrap();
        mgr.register_object(o2).unwrap();
        mgr.register_object(o3).unwrap();
        assert_eq!(mgr.total_active_bytes(), 800);
    }

    #[test]
    fn test_total_active_bytes_excludes_deleted() {
        let mut mgr = ObjectLifecycleManager::new(None);
        let mut o = obj("a", 0);
        o.size_bytes = 1000;
        o.state = OlmLifecycleState::Deleted;
        mgr.register_object(o).unwrap();
        assert_eq!(mgr.total_active_bytes(), 0);
    }

    // ── stats ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_stats_counts_all_states() {
        let mut mgr = ObjectLifecycleManager::new(None);
        mgr.register_object(obj("active", 0)).unwrap();
        let mut exp = obj("exp", 0);
        exp.state = OlmLifecycleState::Expiring { expires_at: 1 };
        mgr.register_object(exp).unwrap();
        let mut arch = obj("arch", 0);
        arch.state = OlmLifecycleState::Archived;
        mgr.register_object(arch).unwrap();
        let mut mfd = obj("mfd", 0);
        mfd.state = OlmLifecycleState::MarkedForDeletion;
        mgr.register_object(mfd).unwrap();
        let mut del = obj("del", 0);
        del.state = OlmLifecycleState::Deleted;
        mgr.register_object(del).unwrap();

        let s = mgr.stats();
        assert_eq!(s.total_objects, 5);
        assert_eq!(s.active_count, 1);
        assert_eq!(s.expiring_count, 1);
        assert_eq!(s.archived_count, 1);
        assert_eq!(s.marked_for_deletion_count, 1);
        assert_eq!(s.deleted_count, 1);
    }

    #[test]
    fn test_stats_rules_count() {
        let mut mgr = ObjectLifecycleManager::new(None);
        mgr.add_rule(rule("r1", 0, None, None, 1));
        mgr.add_rule(rule("r2", 0, None, None, 2));
        assert_eq!(mgr.stats().rules_count, 2);
    }

    // ── add_rule (no max_age, no firing) ──────────────────────────────────────

    #[test]
    fn test_rule_without_max_age_does_not_fire() {
        let mut mgr = ObjectLifecycleManager::new(None);
        // No max_age → rule never fires (just records the rule association)
        mgr.add_rule(rule("forever", 0, None, Some("warm"), 1));
        mgr.register_object(obj("a", 0)).unwrap();
        let actions = mgr.apply_rules(999_999_999);
        assert!(actions.is_empty());
    }

    // ── multiple objects ──────────────────────────────────────────────────────

    #[test]
    fn test_multiple_objects_independent_rules() {
        let mut mgr = ObjectLifecycleManager::new(Some(1000));
        mgr.register_object(obj("young", 500)).unwrap(); // age at now=1000 → 500ms
        mgr.register_object(obj("old", 0)).unwrap(); // age at now=1000 → 1000ms
        let actions = mgr.apply_rules(1000);
        // Only "old" should be deleted (age == ttl)
        assert_eq!(actions.len(), 1);
        assert!(matches!(
            &actions[0],
            OlmLifecycleAction::Delete { object_id, .. }
            if object_id == "old"
        ));
    }

    #[test]
    fn test_apply_rules_multiple_objects_multiple_actions() {
        let mut mgr = ObjectLifecycleManager::new(None);
        mgr.add_rule(rule("to_warm", 0, Some(100), Some("warm"), 1));
        for i in 0..5_u64 {
            mgr.register_object(obj(&format!("o{i}"), 0)).unwrap();
        }
        let actions = mgr.apply_rules(200);
        assert_eq!(actions.len(), 5);
        for action in &actions {
            assert!(
                matches!(action, OlmLifecycleAction::Transition { to_tier, .. } if to_tier == "warm")
            );
        }
    }

    // ── LifecycleState::name ──────────────────────────────────────────────────

    #[test]
    fn test_state_name_variants() {
        assert_eq!(OlmLifecycleState::Active.name(), "Active");
        assert_eq!(
            OlmLifecycleState::Expiring { expires_at: 0 }.name(),
            "Expiring"
        );
        assert_eq!(OlmLifecycleState::Archived.name(), "Archived");
        assert_eq!(
            OlmLifecycleState::MarkedForDeletion.name(),
            "MarkedForDeletion"
        );
        assert_eq!(OlmLifecycleState::Deleted.name(), "Deleted");
    }

    // ── OlmLifecycleAction::object_id ────────────────────────────────────────

    #[test]
    fn test_action_object_id() {
        let t = OlmLifecycleAction::Transition {
            object_id: "x".to_owned(),
            from_tier: "a".to_owned(),
            to_tier: "b".to_owned(),
        };
        assert_eq!(t.object_id(), "x");

        let d = OlmLifecycleAction::Delete {
            object_id: "y".to_owned(),
            reason: "r".to_owned(),
        };
        assert_eq!(d.object_id(), "y");

        let ar = OlmLifecycleAction::Archive {
            object_id: "z".to_owned(),
        };
        assert_eq!(ar.object_id(), "z");

        let se = OlmLifecycleAction::SetExpiring {
            object_id: "w".to_owned(),
            expires_at: 0,
        };
        assert_eq!(se.object_id(), "w");
    }

    // ── LifecycleError variants ───────────────────────────────────────────────

    #[test]
    fn test_error_display_already_exists() {
        let e = LifecycleError::ObjectAlreadyExists("abc".to_owned());
        assert!(e.to_string().contains("abc"));
    }

    #[test]
    fn test_error_display_not_found() {
        let e = LifecycleError::ObjectNotFound("xyz".to_owned());
        assert!(e.to_string().contains("xyz"));
    }

    #[test]
    fn test_error_display_invalid_state() {
        let e = LifecycleError::InvalidState {
            object_id: "o1".to_owned(),
            current: "Deleted".to_owned(),
            expected: "Active".to_owned(),
        };
        let s = e.to_string();
        assert!(s.contains("o1"));
        assert!(s.contains("Deleted"));
        assert!(s.contains("Active"));
    }

    #[test]
    fn test_error_display_rule_not_found() {
        let e = LifecycleError::RuleNotFound("r99".to_owned());
        assert!(e.to_string().contains("r99"));
    }

    // ── purge_deleted — boundary at exactly 1 hour ────────────────────────────

    #[test]
    fn test_purge_deleted_exactly_one_hour_boundary() {
        let mut mgr = ObjectLifecycleManager::new(None);
        let one_hour = 3_600_000_u64;
        let now = 2 * one_hour;
        // Object created exactly one hour ago: created_at = now - one_hour
        let mut d = obj("a", now - one_hour);
        d.state = OlmLifecycleState::Deleted;
        mgr.register_object(d).unwrap();
        // threshold = now - one_hour = created_at → created_at <= threshold → purged
        let count = mgr.purge_deleted(now);
        assert_eq!(count, 1);
    }

    // ── tags and namespace ────────────────────────────────────────────────────

    #[test]
    fn test_managed_object_with_tags() {
        let mut o = obj("tagged", 0);
        o.tags.insert("env".to_owned(), "prod".to_owned());
        o.namespace = "billing".to_owned();
        let mut mgr = ObjectLifecycleManager::new(None);
        mgr.register_object(o).unwrap();
        let retrieved = mgr.unregister_object("tagged").unwrap();
        assert_eq!(retrieved.tags.get("env").map(String::as_str), Some("prod"));
        assert_eq!(retrieved.namespace, "billing");
    }

    // ── retention_rule association ────────────────────────────────────────────

    #[test]
    fn test_rule_association_recorded_when_matched_but_not_expired() {
        let mut mgr = ObjectLifecycleManager::new(None);
        // Rule: matches when age >= 50, but expires only at >= 10_000
        mgr.add_rule(rule("slow_expire", 50, Some(10_000), Some("warm"), 1));
        mgr.register_object(obj("a", 0)).unwrap();
        // age = 100: matches min_age but not max_age yet
        let actions = mgr.apply_rules(100);
        assert!(actions.is_empty());
        // The rule name should be recorded on the object
        let o = mgr.unregister_object("a").unwrap();
        assert_eq!(o.retention_rule.as_deref(), Some("slow_expire"));
    }
}
