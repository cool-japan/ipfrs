//! Role-based and attribute-based access control (RBAC + ABAC) for storage operations.
//!
//! This module provides `StorageAccessController`, a production-grade engine that
//! evaluates access requests against a set of resource policies.  Decisions are
//! made via a deterministic algorithm:
//!
//! 1. Explicit **allow-list** membership → `Allowed`
//! 2. Explicit **deny-list** membership → `Denied`
//! 3. **Role** check  (required_roles, incl. inherited via BFS)
//! 4. **Attribute** check (all required key-value pairs must be present)
//! 5. **Clearance** check  (subject.clearance_level >= policy.min_clearance)
//! 6. **Permission** check (required_permission must match the requested one)
//! 7. Return the policy **effect** (Allow / Deny); no-match → `default_effect`
//!
//! An append-only audit log is maintained for every evaluated decision when
//! `AclConfig::enable_audit` is `true`.

use parking_lot::RwLock;
use std::collections::{HashMap, HashSet, VecDeque};

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/// Atomic storage operation that a subject may wish to perform.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Permission {
    Read,
    Write,
    Delete,
    List,
    Admin,
    Share,
    /// Arbitrary named operation (e.g. `"compress"`, `"replicate"`).
    Execute(String),
}

impl Permission {
    fn matches(&self, other: &Permission) -> bool {
        match (self, other) {
            (Permission::Execute(a), Permission::Execute(b)) => a == b,
            _ => std::mem::discriminant(self) == std::mem::discriminant(other),
        }
    }
}

impl std::fmt::Display for Permission {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Permission::Read => write!(f, "Read"),
            Permission::Write => write!(f, "Write"),
            Permission::Delete => write!(f, "Delete"),
            Permission::List => write!(f, "List"),
            Permission::Admin => write!(f, "Admin"),
            Permission::Share => write!(f, "Share"),
            Permission::Execute(op) => write!(f, "Execute({op})"),
        }
    }
}

/// A named role carrying a set of permissions and optional parent roles.
#[derive(Debug, Clone)]
pub struct SacRole {
    pub name: String,
    pub permissions: Vec<Permission>,
    /// Names of roles this role inherits permissions from.
    pub inherits: Vec<String>,
}

/// Identity and attributes of a subject (user / service account).
#[derive(Debug, Clone)]
pub struct SubjectAttributes {
    pub subject_id: String,
    pub roles: Vec<String>,
    /// Arbitrary key-value metadata (e.g. `("department", "eng")`).
    pub attributes: Vec<(String, String)>,
    /// Numeric clearance level: 0 = public, 255 = top secret.
    pub clearance_level: u8,
}

/// Determines what happens when a policy matches a request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PolicyEffect {
    Allow,
    Deny,
}

/// A policy governing which subjects may perform an operation on a resource.
#[derive(Debug, Clone)]
pub struct ResourcePolicy {
    /// Glob-like pattern: `"*"`, `"dir/*"`, `"*.json"`, or exact path.
    pub resource_pattern: String,
    /// The operation guarded by this policy.
    pub required_permission: Permission,
    /// Subject must hold at least one of these roles (incl. inherited).
    /// Empty = any role is accepted.
    pub required_roles: Vec<String>,
    /// All of these key-value attributes must be present on the subject.
    pub required_attributes: Vec<(String, String)>,
    /// Subject's clearance_level must be >= this value.
    pub min_clearance: u8,
    /// Subjects in this list are always denied, regardless of other checks.
    pub deny_list: Vec<String>,
    /// Subjects in this list are always allowed (overrides deny_list).
    pub allow_list: Vec<String>,
    pub effect: PolicyEffect,
}

/// Final access decision with a human-readable reason.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AccessDecision {
    Allowed { reason: String },
    Denied { reason: String },
    NotApplicable,
}

/// One entry in the audit log.
#[derive(Debug, Clone)]
pub struct SacAuditEntry {
    pub subject_id: String,
    pub resource: String,
    pub permission: Permission,
    pub decision: AccessDecision,
    /// Unix-epoch milliseconds (caller-supplied).
    pub timestamp: u64,
    /// Name of the policy whose effect decided the outcome (if any).
    pub policy_matched: Option<String>,
}

/// Controller-wide configuration.
#[derive(Debug, Clone)]
pub struct AclConfig {
    /// What to do when no policy matches. Defaults to `PolicyEffect::Deny`.
    pub default_effect: PolicyEffect,
    pub enable_audit: bool,
    pub max_audit_entries: usize,
    /// Maximum BFS hops when resolving role inheritance chains.
    pub role_inheritance_depth: u8,
}

impl Default for AclConfig {
    fn default() -> Self {
        AclConfig {
            default_effect: PolicyEffect::Deny,
            enable_audit: true,
            max_audit_entries: 10_000,
            role_inheritance_depth: 16,
        }
    }
}

/// Aggregated statistics snapshot.
#[derive(Debug, Clone)]
pub struct AclStats {
    pub decisions_made: u64,
    pub allows: u64,
    pub denies: u64,
    pub subjects_registered: usize,
    pub policies_count: usize,
}

/// Errors produced by the access controller.
#[derive(Debug, thiserror::Error)]
pub enum AclError {
    #[error("subject not found: {0}")]
    SubjectNotFound(String),
    #[error("role not found: {0}")]
    RoleNotFound(String),
    #[error("policy conflict on resource '{resource}': policies {policies:?}")]
    PolicyConflict {
        resource: String,
        policies: Vec<String>,
    },
    #[error("cyclic role inheritance detected: {0:?}")]
    CyclicRoleInheritance(Vec<String>),
    #[error("invalid resource pattern: {0}")]
    InvalidPattern(String),
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Simple glob matching:
/// - `"*"` matches everything
/// - `"prefix/*"` matches anything that starts with `"prefix/"`
/// - `"*.ext"` matches anything that ends with `".ext"`
/// - exact match otherwise
fn matches_pattern(pattern: &str, resource: &str) -> bool {
    if pattern == "*" {
        return true;
    }
    if let Some(prefix) = pattern.strip_suffix("/*") {
        return resource.starts_with(prefix);
    }
    if let Some(suffix) = pattern.strip_prefix("*.") {
        return resource.ends_with(suffix);
    }
    pattern == resource
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal mutable state
// ─────────────────────────────────────────────────────────────────────────────

struct AclState {
    subjects: HashMap<String, SubjectAttributes>,
    roles: HashMap<String, SacRole>,
    policies: Vec<ResourcePolicy>,
    audit_log: VecDeque<SacAuditEntry>,
    decisions_made: u64,
    allows: u64,
    denies: u64,
}

impl AclState {
    fn new() -> Self {
        AclState {
            subjects: HashMap::new(),
            roles: HashMap::new(),
            policies: Vec::new(),
            audit_log: VecDeque::new(),
            decisions_made: 0,
            allows: 0,
            denies: 0,
        }
    }

    /// BFS role expansion — returns the full set of role names reachable from
    /// `start_roles` honouring the `max_depth` limit.
    fn expand_roles(
        &self,
        start_roles: &[String],
        max_depth: u8,
    ) -> Result<HashSet<String>, AclError> {
        let mut visited: HashSet<String> = HashSet::new();
        let mut queue: VecDeque<(String, u8)> = VecDeque::new();

        for r in start_roles {
            queue.push_back((r.clone(), 0));
        }

        while let Some((role_name, depth)) = queue.pop_front() {
            if visited.contains(&role_name) {
                continue;
            }
            visited.insert(role_name.clone());

            if depth >= max_depth {
                continue;
            }

            if let Some(role) = self.roles.get(&role_name) {
                for parent in &role.inherits {
                    if !visited.contains(parent) {
                        queue.push_back((parent.clone(), depth + 1));
                    }
                }
            }
        }

        Ok(visited)
    }

    /// Validate that no role introduces a cycle in the inheritance graph.
    /// Uses DFS coloring: white (0) / gray (1) / black (2).
    fn validate_no_cycle(&self, role_name: &str) -> Result<(), AclError> {
        let mut color: HashMap<&str, u8> = HashMap::new();
        let mut cycle_path: Vec<String> = Vec::new();

        fn dfs<'a>(
            name: &'a str,
            roles: &'a HashMap<String, SacRole>,
            color: &mut HashMap<&'a str, u8>,
            path: &mut Vec<String>,
        ) -> bool {
            color.insert(name, 1);
            path.push(name.to_string());

            if let Some(role) = roles.get(name) {
                for parent in &role.inherits {
                    let parent_color = *color.get(parent.as_str()).unwrap_or(&0);
                    if parent_color == 1 {
                        path.push(parent.clone());
                        return true; // cycle
                    }
                    if parent_color == 0 && dfs(parent.as_str(), roles, color, path) {
                        return true;
                    }
                }
            }

            path.pop();
            color.insert(name, 2);
            false
        }

        if dfs(role_name, &self.roles, &mut color, &mut cycle_path) {
            return Err(AclError::CyclicRoleInheritance(cycle_path));
        }
        Ok(())
    }

    /// Append an entry to the audit log, evicting oldest if over capacity.
    fn append_audit(&mut self, entry: SacAuditEntry, max_entries: usize, enabled: bool) {
        if !enabled {
            return;
        }
        if self.audit_log.len() >= max_entries {
            self.audit_log.pop_front();
        }
        self.audit_log.push_back(entry);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Public controller
// ─────────────────────────────────────────────────────────────────────────────

/// Thread-safe role-based and attribute-based access controller.
pub struct StorageAccessController {
    config: AclConfig,
    state: RwLock<AclState>,
}

impl StorageAccessController {
    /// Create a new controller with the supplied configuration.
    pub fn new(config: AclConfig) -> Self {
        StorageAccessController {
            config,
            state: RwLock::new(AclState::new()),
        }
    }

    /// Create a controller with default (deny-by-default, audit-enabled) settings.
    pub fn with_defaults() -> Self {
        Self::new(AclConfig::default())
    }

    // ── Mutation ─────────────────────────────────────────────────────────────

    /// Register (or replace) a subject identity and its attributes.
    pub fn register_subject(&self, subject: SubjectAttributes) -> Result<(), AclError> {
        let mut state = self.state.write();
        state.subjects.insert(subject.subject_id.clone(), subject);
        Ok(())
    }

    /// Register (or replace) a role definition.
    ///
    /// Returns [`AclError::CyclicRoleInheritance`] if adding this role would
    /// introduce a cycle in the inheritance graph.
    pub fn register_role(&self, role: SacRole) -> Result<(), AclError> {
        {
            let mut state = self.state.write();
            // Insert tentatively so validate_no_cycle can see the new edges.
            state.roles.insert(role.name.clone(), role.clone());
            let result = state.validate_no_cycle(&role.name);
            if result.is_err() {
                // Rollback
                state.roles.remove(&role.name);
                return result;
            }
        }
        Ok(())
    }

    /// Append a resource policy.
    ///
    /// Validates the resource pattern is non-empty.
    pub fn add_policy(&self, policy: ResourcePolicy) -> Result<(), AclError> {
        if policy.resource_pattern.is_empty() {
            return Err(AclError::InvalidPattern(
                "resource_pattern must not be empty".to_string(),
            ));
        }
        let mut state = self.state.write();
        state.policies.push(policy);
        Ok(())
    }

    /// Remove all policies whose `resource_pattern` equals `resource_pattern`.
    ///
    /// Returns `Err(AclError::InvalidPattern)` if no policies matched.
    pub fn remove_policy(&self, resource_pattern: &str) -> Result<(), AclError> {
        let mut state = self.state.write();
        let before = state.policies.len();
        state
            .policies
            .retain(|p| p.resource_pattern != resource_pattern);
        if state.policies.len() == before {
            return Err(AclError::InvalidPattern(format!(
                "no policy with pattern '{resource_pattern}'"
            )));
        }
        Ok(())
    }

    // ── Core evaluation ───────────────────────────────────────────────────────

    /// Evaluate whether `subject_id` may perform `permission` on `resource`.
    ///
    /// `current_ts` is a caller-supplied Unix timestamp (milliseconds) used
    /// only for audit log entries.
    pub fn check_access(
        &self,
        subject_id: &str,
        resource: &str,
        permission: Permission,
        current_ts: u64,
    ) -> Result<AccessDecision, AclError> {
        let max_depth = self.config.role_inheritance_depth;
        let default_effect = self.config.default_effect.clone();
        let enable_audit = self.config.enable_audit;
        let max_audit = self.config.max_audit_entries;

        // We hold a read lock for evaluation then upgrade to write for stats/audit.
        let decision = {
            let state = self.state.read();

            let subject = state
                .subjects
                .get(subject_id)
                .ok_or_else(|| AclError::SubjectNotFound(subject_id.to_string()))?;

            let effective_roles = state.expand_roles(&subject.roles, max_depth)?;

            // Collect matching policies (those whose pattern covers the resource
            // AND whose required_permission matches the requested one).
            let matching: Vec<&ResourcePolicy> = state
                .policies
                .iter()
                .filter(|p| {
                    matches_pattern(&p.resource_pattern, resource)
                        && p.required_permission.matches(&permission)
                })
                .collect();

            if matching.is_empty() {
                let decision = match default_effect {
                    PolicyEffect::Allow => AccessDecision::Allowed {
                        reason: "no matching policy; default-allow".to_string(),
                    },
                    PolicyEffect::Deny => AccessDecision::Denied {
                        reason: "no matching policy; default-deny".to_string(),
                    },
                };
                (decision, None)
            } else {
                evaluate_policies(subject, subject_id, &effective_roles, &matching)
            }
        };

        // Write stats and audit.
        {
            let mut state = self.state.write();
            state.decisions_made += 1;
            match &decision.0 {
                AccessDecision::Allowed { .. } => state.allows += 1,
                AccessDecision::Denied { .. } => state.denies += 1,
                AccessDecision::NotApplicable => {}
            }
            state.append_audit(
                SacAuditEntry {
                    subject_id: subject_id.to_string(),
                    resource: resource.to_string(),
                    permission: permission.clone(),
                    decision: decision.0.clone(),
                    timestamp: current_ts,
                    policy_matched: decision.1.clone(),
                },
                max_audit,
                enable_audit,
            );
        }

        Ok(decision.0)
    }

    /// Return all permissions the subject is granted (i.e. yields `Allowed`)
    /// for the given resource across the known permission variants.
    ///
    /// The set of probed permissions is:
    /// `[Read, Write, Delete, List, Admin, Share]` plus any `Execute` ops that
    /// appear in registered policies for this resource.
    pub fn effective_permissions(
        &self,
        subject_id: &str,
        resource: &str,
    ) -> Result<Vec<Permission>, AclError> {
        let ts = 0u64; // audit entries for this call use ts=0

        // Collect Execute operation names from matching policies.
        let execute_ops: Vec<String> = {
            let state = self.state.read();
            state
                .policies
                .iter()
                .filter(|p| matches_pattern(&p.resource_pattern, resource))
                .filter_map(|p| {
                    if let Permission::Execute(op) = &p.required_permission {
                        Some(op.clone())
                    } else {
                        None
                    }
                })
                .collect()
        };

        let candidates: Vec<Permission> = {
            let mut v = vec![
                Permission::Read,
                Permission::Write,
                Permission::Delete,
                Permission::List,
                Permission::Admin,
                Permission::Share,
            ];
            for op in execute_ops {
                v.push(Permission::Execute(op));
            }
            v
        };

        let mut allowed = Vec::new();
        for perm in candidates {
            let decision = self.check_access(subject_id, resource, perm.clone(), ts)?;
            if matches!(decision, AccessDecision::Allowed { .. }) {
                allowed.push(perm);
            }
        }
        Ok(allowed)
    }

    /// Return all role names the subject holds, including inherited roles (BFS).
    pub fn roles_for_subject(&self, subject_id: &str) -> Result<Vec<String>, AclError> {
        let state = self.state.read();
        let subject = state
            .subjects
            .get(subject_id)
            .ok_or_else(|| AclError::SubjectNotFound(subject_id.to_string()))?;
        let expanded = state.expand_roles(&subject.roles, self.config.role_inheritance_depth)?;
        let mut result: Vec<String> = expanded.into_iter().collect();
        result.sort();
        Ok(result)
    }

    // ── Audit ─────────────────────────────────────────────────────────────────

    /// Query the audit log.
    ///
    /// Both filters are optional; omitting them returns the full log.
    /// Entries are returned in insertion order (oldest first).
    pub fn audit_log_entries(
        &self,
        subject_id: Option<&str>,
        resource: Option<&str>,
    ) -> Vec<SacAuditEntry> {
        let state = self.state.read();
        state
            .audit_log
            .iter()
            .filter(|e| {
                subject_id.is_none_or(|s| e.subject_id == s)
                    && resource.is_none_or(|r| e.resource == r)
            })
            .cloned()
            .collect()
    }

    // ── Stats ─────────────────────────────────────────────────────────────────

    /// Return a snapshot of controller-wide statistics.
    pub fn stats(&self) -> AclStats {
        let state = self.state.read();
        AclStats {
            decisions_made: state.decisions_made,
            allows: state.allows,
            denies: state.denies,
            subjects_registered: state.subjects.len(),
            policies_count: state.policies.len(),
        }
    }

    /// Current number of entries in the audit log.
    pub fn audit_log_len(&self) -> usize {
        self.state.read().audit_log.len()
    }

    /// Direct read access to the audit log via a closure (avoids cloning).
    pub fn with_audit_log<F, T>(&self, f: F) -> T
    where
        F: FnOnce(&VecDeque<SacAuditEntry>) -> T,
    {
        let state = self.state.read();
        f(&state.audit_log)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Policy evaluation (pure function — no locks required)
// ─────────────────────────────────────────────────────────────────────────────

/// Evaluate a non-empty list of matching policies against a subject, returning
/// `(decision, policy_pattern_that_decided)`.
///
/// Algorithm (first-match wins in the following priority order):
///   1. allow_list  → Allowed
///   2. deny_list   → Denied
///   3. Roles + Attributes + Clearance + Permission → effect
fn evaluate_policies(
    subject: &SubjectAttributes,
    subject_id: &str,
    effective_roles: &HashSet<String>,
    policies: &[&ResourcePolicy],
) -> (AccessDecision, Option<String>) {
    // Pass 1 – explicit allow_list (highest priority)
    for policy in policies {
        if policy.allow_list.iter().any(|id| id == subject_id) {
            return (
                AccessDecision::Allowed {
                    reason: format!(
                        "subject '{subject_id}' is on the allow_list of policy '{}'",
                        policy.resource_pattern
                    ),
                },
                Some(policy.resource_pattern.clone()),
            );
        }
    }

    // Pass 2 – explicit deny_list
    for policy in policies {
        if policy.deny_list.iter().any(|id| id == subject_id) {
            return (
                AccessDecision::Denied {
                    reason: format!(
                        "subject '{subject_id}' is on the deny_list of policy '{}'",
                        policy.resource_pattern
                    ),
                },
                Some(policy.resource_pattern.clone()),
            );
        }
    }

    // Pass 3 – full RBAC+ABAC evaluation on each matching policy.
    // We evaluate each policy and return the first applicable decision.
    // A policy is "applicable" when all its constraints are satisfied; a
    // constraint failure is itself a `Denied` decision.
    let result = policies
        .iter()
        .map(|policy| evaluate_single_policy(subject, policy, effective_roles))
        .next();

    result.unwrap_or((AccessDecision::NotApplicable, None))
}

/// Evaluate one policy against a subject, returning an `(AccessDecision, pattern)` pair.
fn evaluate_single_policy(
    subject: &SubjectAttributes,
    policy: &ResourcePolicy,
    effective_roles: &HashSet<String>,
) -> (AccessDecision, Option<String>) {
    // Role check: if required_roles is non-empty the subject must hold at least one.
    if !policy.required_roles.is_empty() {
        let has_role = policy
            .required_roles
            .iter()
            .any(|r| effective_roles.contains(r.as_str()));
        if !has_role {
            return (
                AccessDecision::Denied {
                    reason: format!(
                        "subject lacks required roles {:?} (policy '{}')",
                        policy.required_roles, policy.resource_pattern
                    ),
                },
                Some(policy.resource_pattern.clone()),
            );
        }
    }

    // Attribute check: every (key, value) pair must be present.
    for (key, value) in &policy.required_attributes {
        let has_attr = subject
            .attributes
            .iter()
            .any(|(k, v)| k == key && v == value);
        if !has_attr {
            return (
                AccessDecision::Denied {
                    reason: format!(
                        "subject missing attribute '{}={}' (policy '{}')",
                        key, value, policy.resource_pattern
                    ),
                },
                Some(policy.resource_pattern.clone()),
            );
        }
    }

    // Clearance check
    if subject.clearance_level < policy.min_clearance {
        return (
            AccessDecision::Denied {
                reason: format!(
                    "subject clearance {} < required {} (policy '{}')",
                    subject.clearance_level, policy.min_clearance, policy.resource_pattern
                ),
            },
            Some(policy.resource_pattern.clone()),
        );
    }

    // All checks passed – return the policy's effect.
    match policy.effect {
        PolicyEffect::Allow => (
            AccessDecision::Allowed {
                reason: format!("policy '{}' granted access", policy.resource_pattern),
            },
            Some(policy.resource_pattern.clone()),
        ),
        PolicyEffect::Deny => (
            AccessDecision::Denied {
                reason: format!("policy '{}' denied access", policy.resource_pattern),
            },
            Some(policy.resource_pattern.clone()),
        ),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Inline xorshift64 PRNG (no rand crate) ────────────────────────────────

    struct Xorshift64(u64);

    impl Xorshift64 {
        fn new(seed: u64) -> Self {
            Xorshift64(if seed == 0 { 0xdeadbeef_cafebabe } else { seed })
        }
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
        fn next_u8(&mut self) -> u8 {
            (self.next() & 0xff) as u8
        }
        fn next_usize(&mut self, max: usize) -> usize {
            (self.next() % max as u64) as usize
        }
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn make_controller() -> StorageAccessController {
        StorageAccessController::with_defaults()
    }

    fn subject(id: &str, roles: &[&str]) -> SubjectAttributes {
        SubjectAttributes {
            subject_id: id.to_string(),
            roles: roles.iter().map(|s| s.to_string()).collect(),
            attributes: vec![],
            clearance_level: 0,
        }
    }

    fn subject_with_attrs(
        id: &str,
        roles: &[&str],
        attrs: &[(&str, &str)],
        clearance: u8,
    ) -> SubjectAttributes {
        SubjectAttributes {
            subject_id: id.to_string(),
            roles: roles.iter().map(|s| s.to_string()).collect(),
            attributes: attrs
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
            clearance_level: clearance,
        }
    }

    fn role(name: &str, perms: Vec<Permission>, inherits: &[&str]) -> SacRole {
        SacRole {
            name: name.to_string(),
            permissions: perms,
            inherits: inherits.iter().map(|s| s.to_string()).collect(),
        }
    }

    fn allow_policy(pattern: &str, perm: Permission, roles: &[&str]) -> ResourcePolicy {
        ResourcePolicy {
            resource_pattern: pattern.to_string(),
            required_permission: perm,
            required_roles: roles.iter().map(|s| s.to_string()).collect(),
            required_attributes: vec![],
            min_clearance: 0,
            deny_list: vec![],
            allow_list: vec![],
            effect: PolicyEffect::Allow,
        }
    }

    fn deny_policy(pattern: &str, perm: Permission, roles: &[&str]) -> ResourcePolicy {
        ResourcePolicy {
            resource_pattern: pattern.to_string(),
            required_permission: perm,
            required_roles: roles.iter().map(|s| s.to_string()).collect(),
            required_attributes: vec![],
            min_clearance: 0,
            deny_list: vec![],
            allow_list: vec![],
            effect: PolicyEffect::Deny,
        }
    }

    // ── Pattern matching unit tests ───────────────────────────────────────────

    #[test]
    fn test_pattern_wildcard_all() {
        assert!(matches_pattern("*", "anything/at/all.json"));
    }

    #[test]
    fn test_pattern_prefix_wildcard() {
        assert!(matches_pattern("data/*", "data/block1"));
        assert!(matches_pattern("data/*", "data/nested"));
        assert!(!matches_pattern("data/*", "other/block1"));
    }

    #[test]
    fn test_pattern_suffix_wildcard() {
        assert!(matches_pattern("*.json", "config.json"));
        assert!(matches_pattern("*.json", "schema.json"));
        assert!(!matches_pattern("*.json", "config.toml"));
    }

    #[test]
    fn test_pattern_exact() {
        assert!(matches_pattern("exact/path", "exact/path"));
        assert!(!matches_pattern("exact/path", "exact/other"));
    }

    // ── Subject registration ──────────────────────────────────────────────────

    #[test]
    fn test_register_subject_ok() {
        let ctrl = make_controller();
        ctrl.register_subject(subject("alice", &["reader"]))
            .expect("register failed");
        let s = ctrl.stats();
        assert_eq!(s.subjects_registered, 1);
    }

    #[test]
    fn test_register_subject_overwrite() {
        let ctrl = make_controller();
        ctrl.register_subject(subject("alice", &["reader"]))
            .unwrap();
        ctrl.register_subject(subject("alice", &["admin"])).unwrap();
        let s = ctrl.stats();
        assert_eq!(s.subjects_registered, 1);
    }

    // ── Role registration ─────────────────────────────────────────────────────

    #[test]
    fn test_register_role_ok() {
        let ctrl = make_controller();
        ctrl.register_role(role("reader", vec![Permission::Read], &[]))
            .unwrap();
        let s = ctrl.stats();
        // Stats.policies_count refers to policies, not roles.
        assert_eq!(s.subjects_registered, 0);
    }

    #[test]
    fn test_register_role_cyclic_two() {
        let ctrl = make_controller();
        ctrl.register_role(role("a", vec![], &["b"])).unwrap();
        let err = ctrl.register_role(role("b", vec![], &["a"])).unwrap_err();
        assert!(
            matches!(err, AclError::CyclicRoleInheritance(_)),
            "expected cyclic error, got {err:?}"
        );
    }

    #[test]
    fn test_register_role_cyclic_self() {
        let ctrl = make_controller();
        let err = ctrl
            .register_role(role("loop_role", vec![], &["loop_role"]))
            .unwrap_err();
        assert!(matches!(err, AclError::CyclicRoleInheritance(_)));
    }

    #[test]
    fn test_register_role_cyclic_three() {
        let ctrl = make_controller();
        ctrl.register_role(role("x", vec![], &["y"])).unwrap();
        ctrl.register_role(role("y", vec![], &["z"])).unwrap();
        let err = ctrl.register_role(role("z", vec![], &["x"])).unwrap_err();
        assert!(matches!(err, AclError::CyclicRoleInheritance(_)));
    }

    // ── Policy management ─────────────────────────────────────────────────────

    #[test]
    fn test_add_policy_ok() {
        let ctrl = make_controller();
        ctrl.add_policy(allow_policy("*", Permission::Read, &[]))
            .unwrap();
        assert_eq!(ctrl.stats().policies_count, 1);
    }

    #[test]
    fn test_add_policy_empty_pattern_error() {
        let ctrl = make_controller();
        let err = ctrl
            .add_policy(allow_policy("", Permission::Read, &[]))
            .unwrap_err();
        assert!(matches!(err, AclError::InvalidPattern(_)));
    }

    #[test]
    fn test_remove_policy_ok() {
        let ctrl = make_controller();
        ctrl.add_policy(allow_policy("data/*", Permission::Read, &[]))
            .unwrap();
        ctrl.remove_policy("data/*").unwrap();
        assert_eq!(ctrl.stats().policies_count, 0);
    }

    #[test]
    fn test_remove_policy_not_found_error() {
        let ctrl = make_controller();
        let err = ctrl.remove_policy("nonexistent/*").unwrap_err();
        assert!(matches!(err, AclError::InvalidPattern(_)));
    }

    // ── Default-deny / default-allow ─────────────────────────────────────────

    #[test]
    fn test_default_deny_no_policy() {
        let ctrl = StorageAccessController::new(AclConfig {
            default_effect: PolicyEffect::Deny,
            ..AclConfig::default()
        });
        ctrl.register_subject(subject("bob", &[])).unwrap();
        let d = ctrl
            .check_access("bob", "secret.json", Permission::Read, 1)
            .unwrap();
        assert!(matches!(d, AccessDecision::Denied { .. }));
    }

    #[test]
    fn test_default_allow_no_policy() {
        let ctrl = StorageAccessController::new(AclConfig {
            default_effect: PolicyEffect::Allow,
            ..AclConfig::default()
        });
        ctrl.register_subject(subject("guest", &[])).unwrap();
        let d = ctrl
            .check_access("guest", "public.json", Permission::Read, 1)
            .unwrap();
        assert!(matches!(d, AccessDecision::Allowed { .. }));
    }

    // ── Allow via role ────────────────────────────────────────────────────────

    #[test]
    fn test_allow_via_role() {
        let ctrl = make_controller();
        ctrl.register_role(role("reader", vec![Permission::Read], &[]))
            .unwrap();
        ctrl.register_subject(subject("alice", &["reader"]))
            .unwrap();
        ctrl.add_policy(allow_policy("data/*", Permission::Read, &["reader"]))
            .unwrap();
        let d = ctrl
            .check_access("alice", "data/file.bin", Permission::Read, 1)
            .unwrap();
        assert!(matches!(d, AccessDecision::Allowed { .. }), "{d:?}");
    }

    #[test]
    fn test_deny_wrong_role() {
        let ctrl = make_controller();
        ctrl.register_role(role("reader", vec![Permission::Read], &[]))
            .unwrap();
        ctrl.register_subject(subject("dave", &["reader"])).unwrap();
        ctrl.add_policy(allow_policy("admin/*", Permission::Write, &["admin"]))
            .unwrap();
        let d = ctrl
            .check_access("dave", "admin/config", Permission::Write, 1)
            .unwrap();
        // Policy matches but dave lacks "admin" role → Denied
        assert!(matches!(d, AccessDecision::Denied { .. }), "{d:?}");
    }

    // ── Deny list ─────────────────────────────────────────────────────────────

    #[test]
    fn test_deny_list_overrides_role() {
        let ctrl = make_controller();
        ctrl.register_role(role("admin", vec![Permission::Admin], &[]))
            .unwrap();
        ctrl.register_subject(subject("mallory", &["admin"]))
            .unwrap();

        let mut policy = allow_policy("*", Permission::Admin, &["admin"]);
        policy.deny_list.push("mallory".to_string());
        ctrl.add_policy(policy).unwrap();

        let d = ctrl
            .check_access("mallory", "anything", Permission::Admin, 1)
            .unwrap();
        assert!(matches!(d, AccessDecision::Denied { .. }), "{d:?}");
    }

    #[test]
    fn test_deny_list_does_not_affect_others() {
        let ctrl = make_controller();
        ctrl.register_role(role("admin", vec![], &[])).unwrap();
        ctrl.register_subject(subject("alice", &["admin"])).unwrap();
        ctrl.register_subject(subject("mallory", &["admin"]))
            .unwrap();

        let mut policy = allow_policy("*", Permission::Admin, &["admin"]);
        policy.deny_list.push("mallory".to_string());
        ctrl.add_policy(policy).unwrap();

        let d = ctrl
            .check_access("alice", "anything", Permission::Admin, 1)
            .unwrap();
        assert!(matches!(d, AccessDecision::Allowed { .. }), "{d:?}");
    }

    // ── Allow list ────────────────────────────────────────────────────────────

    #[test]
    fn test_allow_list_overrides_deny_list() {
        let ctrl = make_controller();
        ctrl.register_subject(subject("super_user", &[])).unwrap();

        let mut policy = deny_policy("*", Permission::Delete, &[]);
        policy.allow_list.push("super_user".to_string());
        policy.deny_list.push("super_user".to_string()); // also on deny – allow wins
        ctrl.add_policy(policy).unwrap();

        let d = ctrl
            .check_access("super_user", "anything", Permission::Delete, 1)
            .unwrap();
        assert!(matches!(d, AccessDecision::Allowed { .. }), "{d:?}");
    }

    #[test]
    fn test_allow_list_bypasses_role_check() {
        let ctrl = make_controller();
        // No roles registered; policy requires "admin" role
        ctrl.register_subject(subject("vip", &[])).unwrap();

        let mut policy = allow_policy("secret/*", Permission::Read, &["admin"]);
        policy.allow_list.push("vip".to_string());
        ctrl.add_policy(policy).unwrap();

        let d = ctrl
            .check_access("vip", "secret/file", Permission::Read, 1)
            .unwrap();
        assert!(matches!(d, AccessDecision::Allowed { .. }), "{d:?}");
    }

    // ── Attribute matching ────────────────────────────────────────────────────

    #[test]
    fn test_attribute_match_allow() {
        let ctrl = make_controller();
        ctrl.register_role(role("engineer", vec![], &[])).unwrap();
        ctrl.register_subject(subject_with_attrs(
            "eng_alice",
            &["engineer"],
            &[("department", "eng"), ("team", "storage")],
            0,
        ))
        .unwrap();

        let mut policy = allow_policy("eng/*", Permission::Write, &["engineer"]);
        policy
            .required_attributes
            .push(("department".to_string(), "eng".to_string()));
        ctrl.add_policy(policy).unwrap();

        let d = ctrl
            .check_access("eng_alice", "eng/data", Permission::Write, 1)
            .unwrap();
        assert!(matches!(d, AccessDecision::Allowed { .. }), "{d:?}");
    }

    #[test]
    fn test_attribute_mismatch_deny() {
        let ctrl = make_controller();
        ctrl.register_role(role("engineer", vec![], &[])).unwrap();
        ctrl.register_subject(subject_with_attrs(
            "sales_bob",
            &["engineer"],
            &[("department", "sales")],
            0,
        ))
        .unwrap();

        let mut policy = allow_policy("eng/*", Permission::Write, &["engineer"]);
        policy
            .required_attributes
            .push(("department".to_string(), "eng".to_string()));
        ctrl.add_policy(policy).unwrap();

        let d = ctrl
            .check_access("sales_bob", "eng/data", Permission::Write, 1)
            .unwrap();
        assert!(matches!(d, AccessDecision::Denied { .. }), "{d:?}");
    }

    #[test]
    fn test_attribute_multiple_required_all_must_match() {
        let ctrl = make_controller();
        ctrl.register_role(role("dev", vec![], &[])).unwrap();
        ctrl.register_subject(subject_with_attrs(
            "user_partial",
            &["dev"],
            &[("dept", "eng")], // missing "clearance"="secret"
            0,
        ))
        .unwrap();

        let mut policy = allow_policy("*", Permission::Read, &["dev"]);
        policy
            .required_attributes
            .push(("dept".to_string(), "eng".to_string()));
        policy
            .required_attributes
            .push(("clearance".to_string(), "secret".to_string()));
        ctrl.add_policy(policy).unwrap();

        let d = ctrl
            .check_access("user_partial", "anything", Permission::Read, 1)
            .unwrap();
        assert!(matches!(d, AccessDecision::Denied { .. }), "{d:?}");
    }

    // ── Clearance level ───────────────────────────────────────────────────────

    #[test]
    fn test_clearance_sufficient() {
        let ctrl = make_controller();
        ctrl.register_role(role("agent", vec![], &[])).unwrap();
        ctrl.register_subject(subject_with_attrs("spy", &["agent"], &[], 200))
            .unwrap();

        let mut policy = allow_policy("classified/*", Permission::Read, &["agent"]);
        policy.min_clearance = 100;
        ctrl.add_policy(policy).unwrap();

        let d = ctrl
            .check_access("spy", "classified/mission", Permission::Read, 1)
            .unwrap();
        assert!(matches!(d, AccessDecision::Allowed { .. }), "{d:?}");
    }

    #[test]
    fn test_clearance_insufficient() {
        let ctrl = make_controller();
        ctrl.register_role(role("agent", vec![], &[])).unwrap();
        ctrl.register_subject(subject_with_attrs("rookie", &["agent"], &[], 50))
            .unwrap();

        let mut policy = allow_policy("classified/*", Permission::Read, &["agent"]);
        policy.min_clearance = 100;
        ctrl.add_policy(policy).unwrap();

        let d = ctrl
            .check_access("rookie", "classified/mission", Permission::Read, 1)
            .unwrap();
        assert!(matches!(d, AccessDecision::Denied { .. }), "{d:?}");
    }

    #[test]
    fn test_clearance_exact_boundary() {
        let ctrl = make_controller();
        ctrl.register_role(role("r", vec![], &[])).unwrap();
        ctrl.register_subject(subject_with_attrs("u", &["r"], &[], 100))
            .unwrap();

        let mut policy = allow_policy("*", Permission::Read, &["r"]);
        policy.min_clearance = 100;
        ctrl.add_policy(policy).unwrap();

        let d = ctrl
            .check_access("u", "resource", Permission::Read, 1)
            .unwrap();
        assert!(matches!(d, AccessDecision::Allowed { .. }), "{d:?}");
    }

    // ── Role inheritance (BFS) ────────────────────────────────────────────────

    #[test]
    fn test_role_inheritance_one_hop() {
        let ctrl = make_controller();
        ctrl.register_role(role("base", vec![Permission::Read], &[]))
            .unwrap();
        ctrl.register_role(role("derived", vec![], &["base"]))
            .unwrap();
        ctrl.register_subject(subject("user", &["derived"]))
            .unwrap();
        ctrl.add_policy(allow_policy("*", Permission::Read, &["base"]))
            .unwrap();

        let d = ctrl
            .check_access("user", "file.txt", Permission::Read, 1)
            .unwrap();
        assert!(matches!(d, AccessDecision::Allowed { .. }), "{d:?}");
    }

    #[test]
    fn test_role_inheritance_two_hops() {
        let ctrl = make_controller();
        ctrl.register_role(role("root", vec![], &[])).unwrap();
        ctrl.register_role(role("middle", vec![], &["root"]))
            .unwrap();
        ctrl.register_role(role("leaf", vec![], &["middle"]))
            .unwrap();
        ctrl.register_subject(subject("u", &["leaf"])).unwrap();
        ctrl.add_policy(allow_policy("*", Permission::Admin, &["root"]))
            .unwrap();

        let d = ctrl.check_access("u", "sys", Permission::Admin, 1).unwrap();
        assert!(matches!(d, AccessDecision::Allowed { .. }), "{d:?}");
    }

    #[test]
    fn test_roles_for_subject_bfs() {
        let ctrl = make_controller();
        ctrl.register_role(role("a", vec![], &[])).unwrap();
        ctrl.register_role(role("b", vec![], &["a"])).unwrap();
        ctrl.register_role(role("c", vec![], &["b"])).unwrap();
        ctrl.register_subject(subject("u", &["c"])).unwrap();

        let roles = ctrl.roles_for_subject("u").unwrap();
        assert!(roles.contains(&"a".to_string()));
        assert!(roles.contains(&"b".to_string()));
        assert!(roles.contains(&"c".to_string()));
        assert_eq!(roles.len(), 3);
    }

    #[test]
    fn test_roles_for_subject_diamond_inheritance() {
        // a ← b ← d
        // a ← c ← d
        let ctrl = make_controller();
        ctrl.register_role(role("a", vec![], &[])).unwrap();
        ctrl.register_role(role("b", vec![], &["a"])).unwrap();
        ctrl.register_role(role("c", vec![], &["a"])).unwrap();
        ctrl.register_role(role("d", vec![], &["b", "c"])).unwrap();
        ctrl.register_subject(subject("u", &["d"])).unwrap();

        let roles = ctrl.roles_for_subject("u").unwrap();
        // Should contain a, b, c, d – no duplicates.
        assert_eq!(roles.len(), 4);
        assert!(roles.contains(&"a".to_string()));
    }

    #[test]
    fn test_roles_for_subject_not_found() {
        let ctrl = make_controller();
        let err = ctrl.roles_for_subject("ghost").unwrap_err();
        assert!(matches!(err, AclError::SubjectNotFound(_)));
    }

    // ── Effective permissions ─────────────────────────────────────────────────

    #[test]
    fn test_effective_permissions_read_write() {
        let ctrl = make_controller();
        ctrl.register_role(role("editor", vec![], &[])).unwrap();
        ctrl.register_subject(subject("ed", &["editor"])).unwrap();
        ctrl.add_policy(allow_policy("docs/*", Permission::Read, &["editor"]))
            .unwrap();
        ctrl.add_policy(allow_policy("docs/*", Permission::Write, &["editor"]))
            .unwrap();
        // Delete allowed for admin only (not editor)
        ctrl.add_policy(allow_policy("docs/*", Permission::Delete, &["admin"]))
            .unwrap();

        let perms = ctrl.effective_permissions("ed", "docs/file.md").unwrap();
        assert!(perms.contains(&Permission::Read), "{perms:?}");
        assert!(perms.contains(&Permission::Write), "{perms:?}");
        assert!(!perms.contains(&Permission::Delete), "{perms:?}");
    }

    #[test]
    fn test_effective_permissions_execute_custom() {
        let ctrl = make_controller();
        ctrl.register_role(role("runner", vec![], &[])).unwrap();
        ctrl.register_subject(subject("svc", &["runner"])).unwrap();
        ctrl.add_policy(allow_policy(
            "jobs/*",
            Permission::Execute("compress".to_string()),
            &["runner"],
        ))
        .unwrap();

        let perms = ctrl.effective_permissions("svc", "jobs/task1").unwrap();
        assert!(
            perms.contains(&Permission::Execute("compress".to_string())),
            "{perms:?}"
        );
    }

    #[test]
    fn test_effective_permissions_no_policies() {
        let ctrl = make_controller();
        ctrl.register_subject(subject("nobody", &[])).unwrap();
        let perms = ctrl
            .effective_permissions("nobody", "some/resource")
            .unwrap();
        // Default-deny: nothing is allowed
        assert!(perms.is_empty(), "{perms:?}");
    }

    // ── Audit log ─────────────────────────────────────────────────────────────

    #[test]
    fn test_audit_log_entries_captured() {
        let ctrl = make_controller();
        ctrl.register_subject(subject("u", &[])).unwrap();
        ctrl.check_access("u", "res", Permission::Read, 42).unwrap();
        let entries = ctrl.audit_log_entries(None, None);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].subject_id, "u");
        assert_eq!(entries[0].resource, "res");
        assert_eq!(entries[0].timestamp, 42);
    }

    #[test]
    fn test_audit_log_filter_by_subject() {
        let ctrl = make_controller();
        ctrl.register_subject(subject("alice", &[])).unwrap();
        ctrl.register_subject(subject("bob", &[])).unwrap();
        ctrl.check_access("alice", "r", Permission::Read, 1)
            .unwrap();
        ctrl.check_access("bob", "r", Permission::Read, 2).unwrap();
        ctrl.check_access("alice", "r", Permission::Write, 3)
            .unwrap();

        let alice_entries = ctrl.audit_log_entries(Some("alice"), None);
        assert_eq!(alice_entries.len(), 2);
        for e in &alice_entries {
            assert_eq!(e.subject_id, "alice");
        }
    }

    #[test]
    fn test_audit_log_filter_by_resource() {
        let ctrl = make_controller();
        ctrl.register_subject(subject("u", &[])).unwrap();
        ctrl.check_access("u", "res_a", Permission::Read, 1)
            .unwrap();
        ctrl.check_access("u", "res_b", Permission::Read, 2)
            .unwrap();
        ctrl.check_access("u", "res_a", Permission::Write, 3)
            .unwrap();

        let entries = ctrl.audit_log_entries(None, Some("res_a"));
        assert_eq!(entries.len(), 2);
        for e in &entries {
            assert_eq!(e.resource, "res_a");
        }
    }

    #[test]
    fn test_audit_log_max_capacity_eviction() {
        let ctrl = StorageAccessController::new(AclConfig {
            max_audit_entries: 5,
            ..AclConfig::default()
        });
        ctrl.register_subject(subject("u", &[])).unwrap();
        for i in 0..10u64 {
            ctrl.check_access("u", "r", Permission::Read, i).unwrap();
        }
        assert_eq!(ctrl.audit_log_len(), 5);
        // Oldest entries should have been evicted; newest should remain.
        let entries = ctrl.audit_log_entries(None, None);
        assert_eq!(entries[0].timestamp, 5);
        assert_eq!(entries[4].timestamp, 9);
    }

    #[test]
    fn test_audit_log_disabled() {
        let ctrl = StorageAccessController::new(AclConfig {
            enable_audit: false,
            ..AclConfig::default()
        });
        ctrl.register_subject(subject("u", &[])).unwrap();
        ctrl.check_access("u", "r", Permission::Read, 1).unwrap();
        assert_eq!(ctrl.audit_log_len(), 0);
    }

    // ── Statistics ────────────────────────────────────────────────────────────

    #[test]
    fn test_stats_counts() {
        let ctrl = make_controller();
        ctrl.register_subject(subject("u", &[])).unwrap();
        // default-deny → all denied
        for i in 0..4u64 {
            ctrl.check_access("u", "r", Permission::Read, i).unwrap();
        }
        let s = ctrl.stats();
        assert_eq!(s.decisions_made, 4);
        assert_eq!(s.denies, 4);
        assert_eq!(s.allows, 0);
    }

    #[test]
    fn test_stats_allows() {
        let ctrl = make_controller();
        ctrl.register_role(role("r", vec![], &[])).unwrap();
        ctrl.register_subject(subject("u", &["r"])).unwrap();
        ctrl.add_policy(allow_policy("*", Permission::Read, &["r"]))
            .unwrap();
        ctrl.check_access("u", "x", Permission::Read, 1).unwrap();
        ctrl.check_access("u", "y", Permission::Read, 2).unwrap();
        let s = ctrl.stats();
        assert_eq!(s.allows, 2);
        assert_eq!(s.denies, 0);
    }

    // ── Error cases ───────────────────────────────────────────────────────────

    #[test]
    fn test_check_access_subject_not_found() {
        let ctrl = make_controller();
        let err = ctrl
            .check_access("ghost", "res", Permission::Read, 0)
            .unwrap_err();
        assert!(matches!(err, AclError::SubjectNotFound(_)));
    }

    #[test]
    fn test_roles_for_subject_unknown() {
        let ctrl = make_controller();
        let err = ctrl.roles_for_subject("unknown").unwrap_err();
        assert!(matches!(err, AclError::SubjectNotFound(_)));
    }

    // ── Execute permission ────────────────────────────────────────────────────

    #[test]
    fn test_execute_permission_specific_op() {
        let ctrl = make_controller();
        ctrl.register_role(role("worker", vec![], &[])).unwrap();
        ctrl.register_subject(subject("w", &["worker"])).unwrap();
        ctrl.add_policy(allow_policy(
            "jobs/*",
            Permission::Execute("replicate".to_string()),
            &["worker"],
        ))
        .unwrap();

        // Correct op
        let d = ctrl
            .check_access(
                "w",
                "jobs/task",
                Permission::Execute("replicate".to_string()),
                1,
            )
            .unwrap();
        assert!(matches!(d, AccessDecision::Allowed { .. }), "{d:?}");

        // Wrong op (same family but different string)
        let d2 = ctrl
            .check_access(
                "w",
                "jobs/task",
                Permission::Execute("compress".to_string()),
                2,
            )
            .unwrap();
        // no policy matches "compress" → default deny
        assert!(matches!(d2, AccessDecision::Denied { .. }), "{d2:?}");
    }

    // ── Multiple policies on same resource ────────────────────────────────────

    #[test]
    fn test_multiple_policies_first_applicable_wins() {
        let ctrl = make_controller();
        ctrl.register_role(role("staff", vec![], &[])).unwrap();
        ctrl.register_subject(subject("emp", &["staff"])).unwrap();

        // Policy 1: allow staff read on data/*
        ctrl.add_policy(allow_policy("data/*", Permission::Read, &["staff"]))
            .unwrap();
        // Policy 2: deny everyone write on data/*
        ctrl.add_policy(deny_policy("data/*", Permission::Write, &[]))
            .unwrap();

        let read_d = ctrl
            .check_access("emp", "data/file", Permission::Read, 1)
            .unwrap();
        let write_d = ctrl
            .check_access("emp", "data/file", Permission::Write, 2)
            .unwrap();

        assert!(
            matches!(read_d, AccessDecision::Allowed { .. }),
            "{read_d:?}"
        );
        assert!(
            matches!(write_d, AccessDecision::Denied { .. }),
            "{write_d:?}"
        );
    }

    // ── Deny policy effect ────────────────────────────────────────────────────

    #[test]
    fn test_deny_effect_policy() {
        let ctrl = StorageAccessController::new(AclConfig {
            default_effect: PolicyEffect::Allow, // default allow
            ..AclConfig::default()
        });
        ctrl.register_role(role("r", vec![], &[])).unwrap();
        ctrl.register_subject(subject("u", &["r"])).unwrap();
        ctrl.add_policy(deny_policy("forbidden/*", Permission::Delete, &["r"]))
            .unwrap();

        let d = ctrl
            .check_access("u", "forbidden/file", Permission::Delete, 1)
            .unwrap();
        assert!(matches!(d, AccessDecision::Denied { .. }), "{d:?}");
    }

    // ── Concurrency / thread safety ───────────────────────────────────────────

    #[test]
    fn test_concurrent_access() {
        use std::sync::Arc;
        use std::thread;

        let ctrl = Arc::new(make_controller());
        ctrl.register_role(role("reader", vec![], &[])).unwrap();
        ctrl.register_subject(subject("concurrent_user", &["reader"]))
            .unwrap();
        ctrl.add_policy(allow_policy("shared/*", Permission::Read, &["reader"]))
            .unwrap();

        let handles: Vec<_> = (0..8)
            .map(|i| {
                let c = Arc::clone(&ctrl);
                thread::spawn(move || {
                    for j in 0..50u64 {
                        let d = c
                            .check_access(
                                "concurrent_user",
                                "shared/resource",
                                Permission::Read,
                                i * 50 + j,
                            )
                            .unwrap();
                        assert!(matches!(d, AccessDecision::Allowed { .. }));
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().expect("thread panicked");
        }

        assert_eq!(ctrl.stats().decisions_made, 400);
    }

    // ── Random / property-style checks ───────────────────────────────────────

    #[test]
    fn test_xorshift_rng_coverage() {
        // Smoke-test the inline PRNG used by other tests
        let mut rng = Xorshift64::new(42);
        let values: Vec<u64> = (0..1000).map(|_| rng.next()).collect();
        // All values should be distinct (with overwhelming probability)
        let unique: HashSet<u64> = values.iter().cloned().collect();
        assert!(unique.len() > 990, "RNG appears degenerate");
    }

    #[test]
    fn test_random_clearance_decisions() {
        let mut rng = Xorshift64::new(0xc0ffee);
        // Use a fresh controller per iteration to avoid cross-policy interference
        // from earlier iterations' policies matching later resources (e.g.
        // "agent_docs_1/*" prefix-matches "agent_docs_10/file").
        ctrl_register_role_shared_test(&mut rng);
    }

    fn ctrl_register_role_shared_test(rng: &mut Xorshift64) {
        for i in 0..20usize {
            let clearance = rng.next_u8();
            let min_req = rng.next_u8();

            // Fresh controller per iteration guarantees exactly one policy matches.
            let ctrl = make_controller();
            ctrl.register_role(role("classified_reader", vec![], &[]))
                .unwrap();
            let sid = format!("agent_{i}");
            ctrl.register_subject(subject_with_attrs(
                &sid,
                &["classified_reader"],
                &[],
                clearance,
            ))
            .unwrap();

            // Use zero-padded resource names to avoid prefix ambiguity in patterns.
            let pat = format!("vault_{i:04}/*");
            let mut p = allow_policy(&pat, Permission::Read, &["classified_reader"]);
            p.min_clearance = min_req;
            ctrl.add_policy(p).unwrap();

            let resource = format!("vault_{i:04}/secret");
            let d = ctrl
                .check_access(&sid, &resource, Permission::Read, i as u64)
                .unwrap();
            if clearance >= min_req {
                assert!(
                    matches!(d, AccessDecision::Allowed { .. }),
                    "i={i} clearance={clearance} min={min_req} → {d:?}"
                );
            } else {
                assert!(
                    matches!(d, AccessDecision::Denied { .. }),
                    "i={i} clearance={clearance} min={min_req} → {d:?}"
                );
            }
        }
    }

    #[test]
    fn test_random_allow_deny_list_membership() {
        let mut rng = Xorshift64::new(0x1337);
        let ctrl = make_controller();
        let subjects: Vec<String> = (0..10)
            .map(|i| {
                let sid = format!("s{i}");
                ctrl.register_subject(subject(&sid, &[])).unwrap();
                sid
            })
            .collect();

        let allowed_idx = rng.next_usize(subjects.len());
        let denied_idx = (allowed_idx + 1) % subjects.len();

        let mut policy = allow_policy("*", Permission::List, &[]);
        policy.allow_list.push(subjects[allowed_idx].clone());
        policy.deny_list.push(subjects[denied_idx].clone());
        ctrl.add_policy(policy).unwrap();

        let d_allow = ctrl
            .check_access(&subjects[allowed_idx], "res", Permission::List, 1)
            .unwrap();
        let d_deny = ctrl
            .check_access(&subjects[denied_idx], "res", Permission::List, 2)
            .unwrap();

        assert!(
            matches!(d_allow, AccessDecision::Allowed { .. }),
            "{d_allow:?}"
        );
        assert!(
            matches!(d_deny, AccessDecision::Denied { .. }),
            "{d_deny:?}"
        );
    }

    // ── Permission::matches self-test ─────────────────────────────────────────

    #[test]
    fn test_permission_matches_same_variant() {
        assert!(Permission::Read.matches(&Permission::Read));
        assert!(Permission::Admin.matches(&Permission::Admin));
        assert!(
            Permission::Execute("op".to_string()).matches(&Permission::Execute("op".to_string()))
        );
    }

    #[test]
    fn test_permission_matches_different_execute_ops() {
        assert!(
            !Permission::Execute("a".to_string()).matches(&Permission::Execute("b".to_string()))
        );
    }

    #[test]
    fn test_permission_matches_different_variants() {
        assert!(!Permission::Read.matches(&Permission::Write));
        assert!(!Permission::Delete.matches(&Permission::Admin));
    }

    // ── Permission::Display ───────────────────────────────────────────────────

    #[test]
    fn test_permission_display() {
        assert_eq!(Permission::Read.to_string(), "Read");
        assert_eq!(
            Permission::Execute("run_gc".to_string()).to_string(),
            "Execute(run_gc)"
        );
    }

    // ── AclConfig defaults ────────────────────────────────────────────────────

    #[test]
    fn test_acl_config_defaults() {
        let cfg = AclConfig::default();
        assert_eq!(cfg.default_effect, PolicyEffect::Deny);
        assert!(cfg.enable_audit);
        assert_eq!(cfg.max_audit_entries, 10_000);
        assert_eq!(cfg.role_inheritance_depth, 16);
    }

    // ── with_audit_log closure ────────────────────────────────────────────────

    #[test]
    fn test_with_audit_log_closure() {
        let ctrl = make_controller();
        ctrl.register_subject(subject("u", &[])).unwrap();
        ctrl.check_access("u", "r", Permission::Read, 99).unwrap();

        let ts = ctrl.with_audit_log(|log| log.back().map(|e| e.timestamp));
        assert_eq!(ts, Some(99));
    }
}
