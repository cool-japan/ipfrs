//! Rule set versioning and conflict resolution for distributed knowledge federation.
//!
//! When multiple IPFRS nodes merge rule sets, conflicts must be detected and resolved
//! deterministically. This module provides the infrastructure for versioning rule sets,
//! computing diffs, and resolving conflicts according to configurable strategies.

use std::collections::HashSet;
use std::fmt;

// ---------------------------------------------------------------------------
// FNV-1a hash (pure Rust, no external crate needed)
// ---------------------------------------------------------------------------

/// Compute a 64-bit FNV-1a hash over an arbitrary byte slice.
fn fnv1a_64(data: &[u8]) -> u64 {
    const FNV_OFFSET_BASIS: u64 = 14_695_981_039_346_656_037;
    const FNV_PRIME: u64 = 1_099_511_628_211;

    let mut hash = FNV_OFFSET_BASIS;
    for &byte in data {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Compute the deterministic fingerprint of a rule-set.
///
/// Rules are sorted lexicographically before hashing so that
/// two rule-sets with the same contents (regardless of insertion order)
/// always produce the same fingerprint.
fn fingerprint_rules(rules: &[String]) -> u64 {
    let mut sorted: Vec<&str> = rules.iter().map(String::as_str).collect();
    sorted.sort_unstable();

    // Hash each rule separated by a NUL byte so that "ab" + "c" ≠ "a" + "bc".
    let mut combined: Vec<u8> = Vec::new();
    for rule in sorted {
        combined.extend_from_slice(rule.as_bytes());
        combined.push(0u8);
    }
    fnv1a_64(&combined)
}

/// Return current Unix time in milliseconds.
///
/// Falls back to 0 if the system clock is unavailable (e.g. in no-std / WASM).
fn unix_millis_now() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// RuleSetVersion
// ---------------------------------------------------------------------------

/// Versioning metadata for a rule set.
///
/// The `fingerprint` is an FNV-1a hash of the sorted rule bodies, providing
/// a fast, deterministic content identity check independent of ordering.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuleSetVersion {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
    /// FNV-1a hash over all sorted rule bodies.
    pub fingerprint: u64,
    /// Unix timestamp in milliseconds when this version was created.
    pub created_at: u64,
    /// Peer ID of the node that authored this version.
    pub author_peer_id: String,
}

impl RuleSetVersion {
    /// Create a new `RuleSetVersion`.
    ///
    /// `created_at` is set to the current wall-clock time.
    /// `fingerprint` is computed from `rules` (sorted for determinism).
    pub fn new(major: u32, minor: u32, patch: u32, rules: &[String], author: &str) -> Self {
        Self {
            major,
            minor,
            patch,
            fingerprint: fingerprint_rules(rules),
            created_at: unix_millis_now(),
            author_peer_id: author.to_string(),
        }
    }

    /// Returns `true` when `self` and `other` share the same major version number,
    /// indicating that they are expected to be compatible.
    pub fn is_compatible_with(&self, other: &Self) -> bool {
        self.major == other.major
    }
}

impl fmt::Display for RuleSetVersion {
    /// Formats as `"major.minor.patch+<fingerprint_hex>"`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}.{}.{}+{:016x}",
            self.major, self.minor, self.patch, self.fingerprint
        )
    }
}

// ---------------------------------------------------------------------------
// RuleSetDiff
// ---------------------------------------------------------------------------

/// Diff between two rule sets.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuleSetDiff {
    /// Rules present in the new set but absent in the old set.
    pub added: Vec<String>,
    /// Rules present in the old set but absent in the new set.
    pub removed: Vec<String>,
    /// Number of rules that appear in both sets.
    pub unchanged: usize,
}

impl RuleSetDiff {
    /// Compute the diff between `old_rules` and `new_rules`.
    ///
    /// The `added` and `removed` vectors are sorted for determinism.
    pub fn diff(old_rules: &[String], new_rules: &[String]) -> Self {
        let old_set: HashSet<&str> = old_rules.iter().map(String::as_str).collect();
        let new_set: HashSet<&str> = new_rules.iter().map(String::as_str).collect();

        let mut added: Vec<String> = new_set
            .difference(&old_set)
            .map(|s| (*s).to_string())
            .collect();
        let mut removed: Vec<String> = old_set
            .difference(&new_set)
            .map(|s| (*s).to_string())
            .collect();

        added.sort_unstable();
        removed.sort_unstable();

        let unchanged = old_set.intersection(&new_set).count();

        Self {
            added,
            removed,
            unchanged,
        }
    }

    /// Returns `true` when there are no additions and no removals.
    pub fn is_empty(&self) -> bool {
        self.added.is_empty() && self.removed.is_empty()
    }

    /// Returns a compact summary string in the form `"+A/-R/=U"`.
    pub fn summary(&self) -> String {
        format!(
            "+{}/-{}/={}",
            self.added.len(),
            self.removed.len(),
            self.unchanged
        )
    }
}

impl fmt::Display for RuleSetDiff {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.summary())
    }
}

// ---------------------------------------------------------------------------
// ConflictStrategy
// ---------------------------------------------------------------------------

/// Strategy to use when resolving a conflict between two `VersionedRuleSet`s.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConflictStrategy {
    /// Choose the rule set whose `created_at` timestamp is larger (more recent).
    LastWriteWins,
    /// Choose the rule set with the higher semantic version (major → minor → patch).
    HigherVersionWins,
    /// Merge all rules from both sets, deduplicated and sorted.
    Union,
    /// Keep only rules that appear in both sets.
    Intersection,
    /// Named custom strategy — falls back to `Union` semantics at runtime.
    Custom(String),
}

impl fmt::Display for ConflictStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LastWriteWins => write!(f, "LastWriteWins"),
            Self::HigherVersionWins => write!(f, "HigherVersionWins"),
            Self::Union => write!(f, "Union"),
            Self::Intersection => write!(f, "Intersection"),
            Self::Custom(name) => write!(f, "Custom({})", name),
        }
    }
}

// ---------------------------------------------------------------------------
// VersionedRuleSet
// ---------------------------------------------------------------------------

/// A rule set with associated version metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VersionedRuleSet {
    pub version: RuleSetVersion,
    pub rules: Vec<String>,
}

impl VersionedRuleSet {
    /// Create a new `VersionedRuleSet`.
    pub fn new(version: RuleSetVersion, rules: Vec<String>) -> Self {
        Self { version, rules }
    }

    /// Returns the number of rules in this set.
    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }
}

// ---------------------------------------------------------------------------
// ResolvedRuleSet
// ---------------------------------------------------------------------------

/// The result of conflict resolution between two `VersionedRuleSet`s.
#[derive(Debug, Clone)]
pub struct ResolvedRuleSet {
    /// The merged/chosen rules after resolution.
    pub rules: Vec<String>,
    /// Human-readable name of the strategy that was applied.
    pub strategy_used: String,
    /// `true` when the two source versions had different fingerprints,
    /// indicating that a genuine conflict existed.
    pub conflict_detected: bool,
    /// The two source versions that were resolved.
    pub source_versions: (RuleSetVersion, RuleSetVersion),
}

// ---------------------------------------------------------------------------
// ConflictResolver
// ---------------------------------------------------------------------------

/// Resolves conflicts between two `VersionedRuleSet`s according to a chosen
/// [`ConflictStrategy`].
pub struct ConflictResolver {
    pub strategy: ConflictStrategy,
}

impl ConflictResolver {
    /// Create a new `ConflictResolver` with the given strategy.
    pub fn new(strategy: ConflictStrategy) -> Self {
        Self { strategy }
    }

    /// Resolve a conflict between `local` and `remote`, returning a
    /// [`ResolvedRuleSet`] that describes the outcome.
    pub fn resolve(&self, local: &VersionedRuleSet, remote: &VersionedRuleSet) -> ResolvedRuleSet {
        let conflict_detected = local.version.fingerprint != remote.version.fingerprint;
        let strategy_used = self.strategy.to_string();

        let rules = match &self.strategy {
            ConflictStrategy::LastWriteWins => {
                if remote.version.created_at > local.version.created_at {
                    remote.rules.clone()
                } else {
                    local.rules.clone()
                }
            }

            ConflictStrategy::HigherVersionWins => {
                let local_ver = (
                    local.version.major,
                    local.version.minor,
                    local.version.patch,
                );
                let remote_ver = (
                    remote.version.major,
                    remote.version.minor,
                    remote.version.patch,
                );
                if remote_ver > local_ver {
                    remote.rules.clone()
                } else {
                    local.rules.clone()
                }
            }

            ConflictStrategy::Union | ConflictStrategy::Custom(_) => {
                // Union semantics: merge all rules, deduplicate, sort.
                let mut merged: Vec<String> = local
                    .rules
                    .iter()
                    .chain(remote.rules.iter())
                    .cloned()
                    .collect::<HashSet<String>>()
                    .into_iter()
                    .collect();
                merged.sort_unstable();
                merged
            }

            ConflictStrategy::Intersection => {
                let local_set: HashSet<&str> = local.rules.iter().map(String::as_str).collect();
                let remote_set: HashSet<&str> = remote.rules.iter().map(String::as_str).collect();
                let mut intersection: Vec<String> = local_set
                    .intersection(&remote_set)
                    .map(|s| (*s).to_string())
                    .collect();
                intersection.sort_unstable();
                intersection
            }
        };

        ResolvedRuleSet {
            rules,
            strategy_used,
            conflict_detected,
            source_versions: (local.version.clone(), remote.version.clone()),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: build a VersionedRuleSet quickly.
    fn make_vrs(
        major: u32,
        minor: u32,
        patch: u32,
        rules: &[&str],
        author: &str,
        created_at_override: Option<u64>,
    ) -> VersionedRuleSet {
        let rule_strings: Vec<String> = rules.iter().map(|s| s.to_string()).collect();
        let mut ver = RuleSetVersion::new(major, minor, patch, &rule_strings, author);
        if let Some(ts) = created_at_override {
            ver.created_at = ts;
        }
        VersionedRuleSet::new(ver, rule_strings)
    }

    // -----------------------------------------------------------------------
    // RuleSetVersion Display
    // -----------------------------------------------------------------------

    #[test]
    fn test_version_display_format() {
        let rules = vec!["rule_a".to_string(), "rule_b".to_string()];
        let v = RuleSetVersion::new(1, 2, 3, &rules, "peer-1");
        let s = v.to_string();
        // Must start with "1.2.3+"
        assert!(s.starts_with("1.2.3+"), "got: {s}");
        // Must contain exactly one '+' followed by 16 hex chars
        let parts: Vec<&str> = s.splitn(2, '+').collect();
        assert_eq!(parts.len(), 2);
        assert_eq!(
            parts[1].len(),
            16,
            "fingerprint hex should be 16 chars: {s}"
        );
        assert!(
            parts[1].chars().all(|c| c.is_ascii_hexdigit()),
            "fingerprint not hex: {s}"
        );
    }

    #[test]
    fn test_version_display_zero_patch() {
        let rules: Vec<String> = vec![];
        let v = RuleSetVersion::new(0, 0, 0, &rules, "peer");
        assert!(v.to_string().starts_with("0.0.0+"));
    }

    // -----------------------------------------------------------------------
    // is_compatible_with
    // -----------------------------------------------------------------------

    #[test]
    fn test_compatible_same_major() {
        let rules: Vec<String> = vec!["r1".to_string()];
        let v1 = RuleSetVersion::new(2, 0, 0, &rules, "a");
        let v2 = RuleSetVersion::new(2, 5, 3, &rules, "b");
        assert!(v1.is_compatible_with(&v2));
        assert!(v2.is_compatible_with(&v1));
    }

    #[test]
    fn test_incompatible_different_major() {
        let rules: Vec<String> = vec!["r1".to_string()];
        let v1 = RuleSetVersion::new(1, 9, 9, &rules, "a");
        let v2 = RuleSetVersion::new(2, 0, 0, &rules, "b");
        assert!(!v1.is_compatible_with(&v2));
        assert!(!v2.is_compatible_with(&v1));
    }

    #[test]
    fn test_compatible_same_version() {
        let rules: Vec<String> = vec![];
        let v = RuleSetVersion::new(3, 1, 4, &rules, "peer");
        assert!(v.is_compatible_with(&v));
    }

    // -----------------------------------------------------------------------
    // Fingerprint determinism
    // -----------------------------------------------------------------------

    #[test]
    fn test_fingerprint_order_independent() {
        let rules_ab = vec!["rule_a".to_string(), "rule_b".to_string()];
        let rules_ba = vec!["rule_b".to_string(), "rule_a".to_string()];
        let v1 = RuleSetVersion::new(1, 0, 0, &rules_ab, "peer");
        let v2 = RuleSetVersion::new(1, 0, 0, &rules_ba, "peer");
        assert_eq!(v1.fingerprint, v2.fingerprint);
    }

    // -----------------------------------------------------------------------
    // RuleSetDiff
    // -----------------------------------------------------------------------

    #[test]
    fn test_diff_added_removed_unchanged() {
        let old = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let new = vec!["b".to_string(), "c".to_string(), "d".to_string()];
        let diff = RuleSetDiff::diff(&old, &new);
        assert_eq!(diff.added, vec!["d".to_string()]);
        assert_eq!(diff.removed, vec!["a".to_string()]);
        assert_eq!(diff.unchanged, 2);
    }

    #[test]
    fn test_diff_no_change() {
        let rules = vec!["x".to_string(), "y".to_string()];
        let diff = RuleSetDiff::diff(&rules, &rules);
        assert!(diff.is_empty());
        assert_eq!(diff.unchanged, 2);
    }

    #[test]
    fn test_diff_all_added() {
        let old: Vec<String> = vec![];
        let new = vec!["r1".to_string(), "r2".to_string()];
        let diff = RuleSetDiff::diff(&old, &new);
        assert_eq!(diff.added.len(), 2);
        assert!(diff.removed.is_empty());
        assert_eq!(diff.unchanged, 0);
    }

    #[test]
    fn test_diff_all_removed() {
        let old = vec!["r1".to_string(), "r2".to_string()];
        let new: Vec<String> = vec![];
        let diff = RuleSetDiff::diff(&old, &new);
        assert!(diff.added.is_empty());
        assert_eq!(diff.removed.len(), 2);
        assert_eq!(diff.unchanged, 0);
    }

    #[test]
    fn test_diff_is_empty_false() {
        let old = vec!["a".to_string()];
        let new = vec!["b".to_string()];
        let diff = RuleSetDiff::diff(&old, &new);
        assert!(!diff.is_empty());
    }

    #[test]
    fn test_diff_summary_format() {
        let old = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let new = vec![
            "b".to_string(),
            "c".to_string(),
            "d".to_string(),
            "e".to_string(),
        ];
        let diff = RuleSetDiff::diff(&old, &new);
        // +2 added (d, e), -1 removed (a), =2 unchanged (b, c)
        assert_eq!(diff.summary(), "+2/-1/=2");
    }

    #[test]
    fn test_diff_summary_no_change() {
        let rules = vec!["x".to_string()];
        let diff = RuleSetDiff::diff(&rules, &rules);
        assert_eq!(diff.summary(), "+0/-0/=1");
    }

    // -----------------------------------------------------------------------
    // ConflictResolver — LastWriteWins
    // -----------------------------------------------------------------------

    #[test]
    fn test_last_write_wins_picks_higher_timestamp() {
        let local = make_vrs(1, 0, 0, &["rule_local"], "local", Some(100));
        let remote = make_vrs(1, 0, 0, &["rule_remote"], "remote", Some(200));
        let resolver = ConflictResolver::new(ConflictStrategy::LastWriteWins);
        let result = resolver.resolve(&local, &remote);
        assert_eq!(result.rules, vec!["rule_remote".to_string()]);
        assert_eq!(result.strategy_used, "LastWriteWins");
    }

    #[test]
    fn test_last_write_wins_picks_local_when_newer() {
        let local = make_vrs(1, 0, 0, &["rule_local"], "local", Some(999));
        let remote = make_vrs(1, 0, 0, &["rule_remote"], "remote", Some(1));
        let resolver = ConflictResolver::new(ConflictStrategy::LastWriteWins);
        let result = resolver.resolve(&local, &remote);
        assert_eq!(result.rules, vec!["rule_local".to_string()]);
    }

    // -----------------------------------------------------------------------
    // ConflictResolver — HigherVersionWins
    // -----------------------------------------------------------------------

    #[test]
    fn test_higher_version_wins_remote_major() {
        let local = make_vrs(1, 9, 9, &["local_rule"], "local", None);
        let remote = make_vrs(2, 0, 0, &["remote_rule"], "remote", None);
        let resolver = ConflictResolver::new(ConflictStrategy::HigherVersionWins);
        let result = resolver.resolve(&local, &remote);
        assert_eq!(result.rules, vec!["remote_rule".to_string()]);
    }

    #[test]
    fn test_higher_version_wins_local_minor() {
        let local = make_vrs(1, 5, 0, &["local_rule"], "local", None);
        let remote = make_vrs(1, 3, 0, &["remote_rule"], "remote", None);
        let resolver = ConflictResolver::new(ConflictStrategy::HigherVersionWins);
        let result = resolver.resolve(&local, &remote);
        assert_eq!(result.rules, vec!["local_rule".to_string()]);
    }

    #[test]
    fn test_higher_version_wins_patch_tiebreak() {
        let local = make_vrs(2, 1, 3, &["local_rule"], "local", None);
        let remote = make_vrs(2, 1, 7, &["remote_rule"], "remote", None);
        let resolver = ConflictResolver::new(ConflictStrategy::HigherVersionWins);
        let result = resolver.resolve(&local, &remote);
        assert_eq!(result.rules, vec!["remote_rule".to_string()]);
    }

    // -----------------------------------------------------------------------
    // ConflictResolver — Union
    // -----------------------------------------------------------------------

    #[test]
    fn test_union_merges_and_deduplicates() {
        let local = make_vrs(1, 0, 0, &["a", "b", "c"], "local", None);
        let remote = make_vrs(1, 0, 0, &["b", "c", "d"], "remote", None);
        let resolver = ConflictResolver::new(ConflictStrategy::Union);
        let result = resolver.resolve(&local, &remote);
        let expected = vec!["a", "b", "c", "d"]
            .into_iter()
            .map(String::from)
            .collect::<Vec<_>>();
        assert_eq!(result.rules, expected);
    }

    #[test]
    fn test_union_identical_sets() {
        let local = make_vrs(1, 0, 0, &["x", "y"], "local", None);
        let remote = make_vrs(1, 0, 0, &["x", "y"], "remote", None);
        let resolver = ConflictResolver::new(ConflictStrategy::Union);
        let result = resolver.resolve(&local, &remote);
        assert_eq!(result.rules, vec!["x".to_string(), "y".to_string()]);
    }

    // -----------------------------------------------------------------------
    // ConflictResolver — Intersection
    // -----------------------------------------------------------------------

    #[test]
    fn test_intersection_returns_common_rules() {
        let local = make_vrs(1, 0, 0, &["a", "b", "c"], "local", None);
        let remote = make_vrs(1, 0, 0, &["b", "c", "d"], "remote", None);
        let resolver = ConflictResolver::new(ConflictStrategy::Intersection);
        let result = resolver.resolve(&local, &remote);
        let expected = vec!["b".to_string(), "c".to_string()];
        assert_eq!(result.rules, expected);
    }

    #[test]
    fn test_intersection_empty_when_disjoint() {
        let local = make_vrs(1, 0, 0, &["a", "b"], "local", None);
        let remote = make_vrs(1, 0, 0, &["c", "d"], "remote", None);
        let resolver = ConflictResolver::new(ConflictStrategy::Intersection);
        let result = resolver.resolve(&local, &remote);
        assert!(result.rules.is_empty());
    }

    // -----------------------------------------------------------------------
    // ConflictResolver — Custom (Union fallback)
    // -----------------------------------------------------------------------

    #[test]
    fn test_custom_strategy_union_fallback() {
        let local = make_vrs(1, 0, 0, &["p", "q"], "local", None);
        let remote = make_vrs(1, 0, 0, &["q", "r"], "remote", None);
        let resolver = ConflictResolver::new(ConflictStrategy::Custom("my-policy".to_string()));
        let result = resolver.resolve(&local, &remote);
        let expected = vec!["p".to_string(), "q".to_string(), "r".to_string()];
        assert_eq!(result.rules, expected);
        assert_eq!(result.strategy_used, "Custom(my-policy)");
    }

    // -----------------------------------------------------------------------
    // conflict_detected flag
    // -----------------------------------------------------------------------

    #[test]
    fn test_conflict_detected_different_fingerprints() {
        let local = make_vrs(1, 0, 0, &["rule_a"], "local", None);
        let remote = make_vrs(1, 0, 0, &["rule_b"], "remote", None);
        // Different rule bodies → different fingerprints.
        assert_ne!(local.version.fingerprint, remote.version.fingerprint);
        let resolver = ConflictResolver::new(ConflictStrategy::Union);
        let result = resolver.resolve(&local, &remote);
        assert!(result.conflict_detected);
    }

    #[test]
    fn test_conflict_not_detected_identical_fingerprints() {
        let rules = vec!["rule_a".to_string(), "rule_b".to_string()];
        // Build two versioned rule sets with the same rule bodies.
        let ver_local = RuleSetVersion::new(1, 0, 0, &rules, "local");
        let ver_remote = RuleSetVersion::new(1, 0, 1, &rules, "remote");
        // Same rules ⇒ same fingerprint, no conflict.
        assert_eq!(ver_local.fingerprint, ver_remote.fingerprint);
        let local = VersionedRuleSet::new(ver_local, rules.clone());
        let remote = VersionedRuleSet::new(ver_remote, rules);
        let resolver = ConflictResolver::new(ConflictStrategy::Union);
        let result = resolver.resolve(&local, &remote);
        assert!(!result.conflict_detected);
    }

    // -----------------------------------------------------------------------
    // VersionedRuleSet
    // -----------------------------------------------------------------------

    #[test]
    fn test_versioned_rule_set_rule_count() {
        let vrs = make_vrs(1, 0, 0, &["a", "b", "c", "d", "e"], "peer", None);
        assert_eq!(vrs.rule_count(), 5);
    }

    // -----------------------------------------------------------------------
    // Source versions are preserved in ResolvedRuleSet
    // -----------------------------------------------------------------------

    #[test]
    fn test_source_versions_preserved() {
        let local = make_vrs(1, 0, 0, &["l"], "local-peer", None);
        let remote = make_vrs(2, 0, 0, &["r"], "remote-peer", None);
        let resolver = ConflictResolver::new(ConflictStrategy::HigherVersionWins);
        let result = resolver.resolve(&local, &remote);
        assert_eq!(result.source_versions.0.author_peer_id, "local-peer");
        assert_eq!(result.source_versions.1.author_peer_id, "remote-peer");
    }
}
