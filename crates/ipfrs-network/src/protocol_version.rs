//! Protocol versioning and compatibility negotiation for P2P connections.
//!
//! This module provides:
//! - Semantic version representation for protocols (`ProtocolVersion`)
//! - Compatibility classification between local and remote protocol versions
//! - Protocol descriptor with feature advertisement
//! - Multi-protocol version negotiation (`ProtocolVersionManager`)
//! - Statistics tracking for negotiation outcomes
//!
//! ## Design Rationale
//!
//! Compatibility is evaluated against a per-protocol `min_compatible` floor.
//! A remote peer speaking version `R` is:
//!
//! - **FullyCompatible**   — `R == local`
//! - **BackwardCompatible** — `R < local` but `R >= min_compatible` (we speak down to them)
//! - **ForwardCompatible**  — `R > local` (they are newer; we use our feature set)
//! - **Incompatible**       — `R < min_compatible` (wire format too old to speak safely)
//!
//! Feature negotiation produces the intersection of advertised feature sets.
//! Features present locally but absent from the intersection are recorded in
//! `NegotiationResult::dropped_features`.

use std::collections::HashMap;
use std::fmt;

// ────────────────────────────────────────────────────────────────────────────
// ProtocolVersion
// ────────────────────────────────────────────────────────────────────────────

/// A semantic version triple for a protocol (`major.minor.patch`).
///
/// Ordering follows standard semver lexicographic comparison: major is most
/// significant, then minor, then patch.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ProtocolVersion {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}

impl ProtocolVersion {
    /// Construct a new version triple.
    pub fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }

    /// Parse a dotted version string such as `"1.2.3"`.
    ///
    /// Returns `None` if the string does not contain exactly three
    /// dot-separated numeric components.
    pub fn parse(s: &str) -> Option<Self> {
        let parts: Vec<&str> = s.split('.').collect();
        if parts.len() != 3 {
            return None;
        }
        let major = parts[0].parse::<u32>().ok()?;
        let minor = parts[1].parse::<u32>().ok()?;
        let patch = parts[2].parse::<u32>().ok()?;
        Some(Self::new(major, minor, patch))
    }

    /// Determine the `CompatibilityLevel` of `self` (the local version) with
    /// respect to `other` (the remote version), given `min_compatible` as the
    /// oldest version we are willing to speak with.
    ///
    /// Rules (evaluated in priority order):
    ///
    /// 1. `other < min_compatible` → `Incompatible`
    /// 2. `other == self`          → `FullyCompatible`
    /// 3. `other < self`           → `BackwardCompatible` (remote is older)
    /// 4. `other > self`           → `ForwardCompatible`  (remote is newer)
    pub fn is_compatible_with(&self, other: &Self, min_compatible: &Self) -> CompatibilityLevel {
        if other < min_compatible {
            return CompatibilityLevel::Incompatible;
        }
        if other == self {
            CompatibilityLevel::FullyCompatible
        } else if other < self {
            CompatibilityLevel::BackwardCompatible
        } else {
            CompatibilityLevel::ForwardCompatible
        }
    }

    /// Return the canonical string representation `"major.minor.patch"`.
    pub fn to_string_repr(&self) -> String {
        format!("{}.{}.{}", self.major, self.minor, self.patch)
    }
}

impl fmt::Display for ProtocolVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

// ────────────────────────────────────────────────────────────────────────────
// CompatibilityLevel
// ────────────────────────────────────────────────────────────────────────────

/// Classification of the compatibility relationship between two protocol
/// version endpoints.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompatibilityLevel {
    /// Both peers run the identical version — full feature parity guaranteed.
    FullyCompatible,
    /// The remote peer is older than us but at or above the minimum floor.
    /// We adapt downward to serve them.
    BackwardCompatible,
    /// The remote peer is newer than us.  We can still communicate using our
    /// own (older) feature set, but some of their features will be unavailable.
    ForwardCompatible,
    /// The remote version is below our `min_compatible` floor.
    /// Communication is not safe and negotiation should be rejected.
    Incompatible,
}

// ────────────────────────────────────────────────────────────────────────────
// ProtocolDescriptor
// ────────────────────────────────────────────────────────────────────────────

/// Full description of a protocol as advertised by one side of a connection.
///
/// A `ProtocolDescriptor` captures both the current version and the oldest
/// version that the advertising peer can still speak (`min_compatible`), plus
/// the list of optional features it supports.
#[derive(Debug, Clone)]
pub struct ProtocolDescriptor {
    /// Human-readable protocol name, e.g. `"ipfrs/bitswap"`.
    pub name: String,
    /// The version this peer is currently running.
    pub version: ProtocolVersion,
    /// The oldest version this peer can still inter-operate with.
    pub min_compatible: ProtocolVersion,
    /// Optional feature strings this peer advertises.
    pub features: Vec<String>,
}

impl ProtocolDescriptor {
    /// Convenience constructor.
    pub fn new(
        name: impl Into<String>,
        version: ProtocolVersion,
        min_compatible: ProtocolVersion,
        features: Vec<String>,
    ) -> Self {
        Self {
            name: name.into(),
            version,
            min_compatible,
            features,
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// NegotiationResult
// ────────────────────────────────────────────────────────────────────────────

/// The outcome of a successful protocol version negotiation between two peers.
///
/// When negotiation fails entirely (e.g. unknown protocol or incompatible
/// version), `ProtocolVersionManager::negotiate` returns `None`.
#[derive(Debug, Clone)]
pub struct NegotiationResult {
    /// The protocol name that was negotiated.
    pub protocol: String,
    /// The version that both sides agreed to operate at.
    ///
    /// This is the *lower* of the two versions (local min or remote version)
    /// ensuring backward compatibility.
    pub agreed_version: ProtocolVersion,
    /// Overall compatibility classification.
    pub compatibility: CompatibilityLevel,
    /// Features present on both sides — the effective feature set.
    pub common_features: Vec<String>,
    /// Features present locally but absent on the remote — unavailable for
    /// this session.
    pub dropped_features: Vec<String>,
}

// ────────────────────────────────────────────────────────────────────────────
// VersionStats
// ────────────────────────────────────────────────────────────────────────────

/// Running statistics for all negotiation attempts handled by a
/// `ProtocolVersionManager`.
#[derive(Debug, Clone, Default)]
pub struct VersionStats {
    /// Total negotiation attempts (success + failure).
    pub negotiations: u64,
    /// Negotiations that produced a `NegotiationResult`.
    pub successful: u64,
    /// Negotiations that failed (unknown protocol or incompatible versions).
    pub failed: u64,
    /// Successful negotiations that were `BackwardCompatible`.
    pub backward_compat: u64,
    /// Successful negotiations that were `ForwardCompatible`.
    pub forward_compat: u64,
}

// ────────────────────────────────────────────────────────────────────────────
// ProtocolVersionManager
// ────────────────────────────────────────────────────────────────────────────

/// Central registry and negotiation engine for protocol versions.
///
/// Peers register their local `ProtocolDescriptor`s and then call `negotiate`
/// when a remote peer's descriptor arrives.  The manager updates internal
/// statistics for every attempt.
pub struct ProtocolVersionManager {
    supported: HashMap<String, ProtocolDescriptor>,
    stats: VersionStats,
}

impl ProtocolVersionManager {
    /// Create an empty manager with no registered protocols.
    pub fn new() -> Self {
        Self {
            supported: HashMap::new(),
            stats: VersionStats::default(),
        }
    }

    /// Register a local `ProtocolDescriptor`.
    ///
    /// Returns `true` if this is a new registration, `false` if a descriptor
    /// with the same name already existed and was overwritten.
    pub fn register(&mut self, descriptor: ProtocolDescriptor) -> bool {
        let is_new = !self.supported.contains_key(&descriptor.name);
        self.supported.insert(descriptor.name.clone(), descriptor);
        is_new
    }

    /// Attempt to negotiate with a remote peer's `ProtocolDescriptor`.
    ///
    /// Returns `Some(NegotiationResult)` on success, `None` when:
    /// - The protocol is not registered locally, or
    /// - The versions are `Incompatible` in both directions.
    ///
    /// **Compatibility check flow:**
    ///
    /// 1. We check the remote version against our local descriptor's
    ///    `min_compatible` floor.
    /// 2. We *also* check our local version against the remote descriptor's
    ///    `min_compatible` floor — we must be acceptable to them too.
    /// 3. If both checks pass, a `NegotiationResult` is produced.
    ///
    /// **Agreed version selection:**
    /// The session runs at `min(local.version, remote.version)` — the older of
    /// the two — ensuring that neither side uses features unavailable to the
    /// other.
    pub fn negotiate(
        &mut self,
        protocol: &str,
        remote: &ProtocolDescriptor,
    ) -> Option<NegotiationResult> {
        self.stats.negotiations += 1;

        let local = match self.supported.get(protocol) {
            Some(d) => d.clone(),
            None => {
                self.stats.failed += 1;
                return None;
            }
        };

        // Check remote version against our minimum floor.
        let compat_local_pov = local
            .version
            .is_compatible_with(&remote.version, &local.min_compatible);

        if compat_local_pov == CompatibilityLevel::Incompatible {
            self.stats.failed += 1;
            return None;
        }

        // Check our version against the remote's minimum floor.
        let compat_remote_pov = remote
            .version
            .is_compatible_with(&local.version, &remote.min_compatible);

        if compat_remote_pov == CompatibilityLevel::Incompatible {
            self.stats.failed += 1;
            return None;
        }

        // Choose the lower version for the session.
        let agreed_version = if local.version <= remote.version {
            local.version.clone()
        } else {
            remote.version.clone()
        };

        let common_features = Self::find_common_features(&local.features, &remote.features);

        let dropped_features: Vec<String> = local
            .features
            .iter()
            .filter(|f| !common_features.contains(f))
            .cloned()
            .collect();

        // Update directional stats.
        match &compat_local_pov {
            CompatibilityLevel::BackwardCompatible => self.stats.backward_compat += 1,
            CompatibilityLevel::ForwardCompatible => self.stats.forward_compat += 1,
            _ => {}
        }
        self.stats.successful += 1;

        Some(NegotiationResult {
            protocol: protocol.to_owned(),
            agreed_version,
            compatibility: compat_local_pov,
            common_features,
            dropped_features,
        })
    }

    /// Compute the intersection of two feature sets, preserving the order of
    /// the local side.
    pub fn find_common_features(local: &[String], remote: &[String]) -> Vec<String> {
        let remote_set: std::collections::HashSet<&String> = remote.iter().collect();
        local
            .iter()
            .filter(|f| remote_set.contains(f))
            .cloned()
            .collect()
    }

    /// Look up the registered descriptor for a protocol by name.
    pub fn get_descriptor(&self, protocol: &str) -> Option<&ProtocolDescriptor> {
        self.supported.get(protocol)
    }

    /// Return `true` if a descriptor for `protocol` has been registered.
    pub fn is_registered(&self, protocol: &str) -> bool {
        self.supported.contains_key(protocol)
    }

    /// Return all registered protocol names as string slices.
    pub fn supported_protocols(&self) -> Vec<&str> {
        self.supported.keys().map(String::as_str).collect()
    }

    /// Read-only view of the accumulated negotiation statistics.
    pub fn stats(&self) -> &VersionStats {
        &self.stats
    }
}

impl Default for ProtocolVersionManager {
    fn default() -> Self {
        Self::new()
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── ProtocolVersion construction & display ────────────────────────────

    #[test]
    fn test_version_new_stores_components() {
        let v = ProtocolVersion::new(1, 2, 3);
        assert_eq!(v.major, 1);
        assert_eq!(v.minor, 2);
        assert_eq!(v.patch, 3);
    }

    #[test]
    fn test_version_display() {
        let v = ProtocolVersion::new(2, 0, 1);
        assert_eq!(format!("{}", v), "2.0.1");
    }

    #[test]
    fn test_version_to_string_repr() {
        let v = ProtocolVersion::new(0, 10, 99);
        assert_eq!(v.to_string_repr(), "0.10.99");
    }

    #[test]
    fn test_version_roundtrip() {
        let original = ProtocolVersion::new(3, 14, 7);
        let s = original.to_string_repr();
        let parsed = ProtocolVersion::parse(&s).expect("roundtrip must succeed");
        assert_eq!(original, parsed);
    }

    // ── ProtocolVersion parsing ───────────────────────────────────────────

    #[test]
    fn test_parse_valid() {
        let v = ProtocolVersion::parse("1.2.3").expect("valid parse");
        assert_eq!(v, ProtocolVersion::new(1, 2, 3));
    }

    #[test]
    fn test_parse_zeros() {
        let v = ProtocolVersion::parse("0.0.0").expect("valid zeros");
        assert_eq!(v, ProtocolVersion::new(0, 0, 0));
    }

    #[test]
    fn test_parse_invalid_not_enough_parts() {
        assert!(ProtocolVersion::parse("1.2").is_none());
    }

    #[test]
    fn test_parse_invalid_too_many_parts() {
        assert!(ProtocolVersion::parse("1.2.3.4").is_none());
    }

    #[test]
    fn test_parse_invalid_non_numeric() {
        assert!(ProtocolVersion::parse("a.b.c").is_none());
    }

    #[test]
    fn test_parse_empty_string() {
        assert!(ProtocolVersion::parse("").is_none());
    }

    // ── ProtocolVersion ordering ──────────────────────────────────────────

    #[test]
    fn test_version_ordering_major_dominates() {
        assert!(ProtocolVersion::new(2, 0, 0) > ProtocolVersion::new(1, 9, 9));
    }

    #[test]
    fn test_version_ordering_minor_secondary() {
        assert!(ProtocolVersion::new(1, 3, 0) > ProtocolVersion::new(1, 2, 99));
    }

    #[test]
    fn test_version_ordering_patch_tertiary() {
        assert!(ProtocolVersion::new(1, 2, 4) > ProtocolVersion::new(1, 2, 3));
    }

    #[test]
    fn test_version_equality() {
        assert_eq!(ProtocolVersion::new(1, 2, 3), ProtocolVersion::new(1, 2, 3));
    }

    // ── Compatibility levels ──────────────────────────────────────────────

    #[test]
    fn test_compat_fully_compatible() {
        let local = ProtocolVersion::new(1, 2, 3);
        let remote = ProtocolVersion::new(1, 2, 3);
        let min = ProtocolVersion::new(1, 0, 0);
        assert_eq!(
            local.is_compatible_with(&remote, &min),
            CompatibilityLevel::FullyCompatible
        );
    }

    #[test]
    fn test_compat_backward_compatible_remote_older() {
        let local = ProtocolVersion::new(2, 0, 0);
        let remote = ProtocolVersion::new(1, 5, 0); // older than local
        let min = ProtocolVersion::new(1, 0, 0);
        assert_eq!(
            local.is_compatible_with(&remote, &min),
            CompatibilityLevel::BackwardCompatible
        );
    }

    #[test]
    fn test_compat_forward_compatible_remote_newer() {
        let local = ProtocolVersion::new(1, 0, 0);
        let remote = ProtocolVersion::new(2, 0, 0); // newer than local
        let min = ProtocolVersion::new(1, 0, 0);
        assert_eq!(
            local.is_compatible_with(&remote, &min),
            CompatibilityLevel::ForwardCompatible
        );
    }

    #[test]
    fn test_compat_incompatible_below_min() {
        let local = ProtocolVersion::new(2, 0, 0);
        let remote = ProtocolVersion::new(0, 5, 0); // below min floor
        let min = ProtocolVersion::new(1, 0, 0);
        assert_eq!(
            local.is_compatible_with(&remote, &min),
            CompatibilityLevel::Incompatible
        );
    }

    #[test]
    fn test_compat_exactly_at_min_floor() {
        let local = ProtocolVersion::new(2, 0, 0);
        let remote = ProtocolVersion::new(1, 0, 0); // exactly at min
        let min = ProtocolVersion::new(1, 0, 0);
        // remote == min, but remote < local → BackwardCompatible
        assert_eq!(
            local.is_compatible_with(&remote, &min),
            CompatibilityLevel::BackwardCompatible
        );
    }

    // ── find_common_features ──────────────────────────────────────────────

    #[test]
    fn test_common_features_full_intersection() {
        let local = vec!["a".to_owned(), "b".to_owned(), "c".to_owned()];
        let remote = vec!["b".to_owned(), "a".to_owned(), "c".to_owned()];
        let common = ProtocolVersionManager::find_common_features(&local, &remote);
        // Order follows local
        assert_eq!(common, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_common_features_partial_intersection() {
        let local = vec!["x".to_owned(), "y".to_owned(), "z".to_owned()];
        let remote = vec!["y".to_owned(), "w".to_owned()];
        let common = ProtocolVersionManager::find_common_features(&local, &remote);
        assert_eq!(common, vec!["y"]);
    }

    #[test]
    fn test_common_features_empty_intersection() {
        let local = vec!["a".to_owned()];
        let remote = vec!["b".to_owned()];
        let common = ProtocolVersionManager::find_common_features(&local, &remote);
        assert!(common.is_empty());
    }

    #[test]
    fn test_common_features_empty_inputs() {
        let common = ProtocolVersionManager::find_common_features(&[], &[]);
        assert!(common.is_empty());
    }

    // ── ProtocolVersionManager registration ──────────────────────────────

    #[test]
    fn test_register_new_returns_true() {
        let mut mgr = ProtocolVersionManager::new();
        let desc = ProtocolDescriptor::new(
            "proto/a",
            ProtocolVersion::new(1, 0, 0),
            ProtocolVersion::new(1, 0, 0),
            vec![],
        );
        assert!(mgr.register(desc));
    }

    #[test]
    fn test_register_duplicate_returns_false() {
        let mut mgr = ProtocolVersionManager::new();
        let mk = || {
            ProtocolDescriptor::new(
                "proto/a",
                ProtocolVersion::new(1, 0, 0),
                ProtocolVersion::new(1, 0, 0),
                vec![],
            )
        };
        mgr.register(mk());
        assert!(!mgr.register(mk()));
    }

    #[test]
    fn test_is_registered() {
        let mut mgr = ProtocolVersionManager::new();
        assert!(!mgr.is_registered("proto/b"));
        let desc = ProtocolDescriptor::new(
            "proto/b",
            ProtocolVersion::new(1, 0, 0),
            ProtocolVersion::new(1, 0, 0),
            vec![],
        );
        mgr.register(desc);
        assert!(mgr.is_registered("proto/b"));
    }

    #[test]
    fn test_get_descriptor() {
        let mut mgr = ProtocolVersionManager::new();
        let desc = ProtocolDescriptor::new(
            "proto/c",
            ProtocolVersion::new(2, 1, 0),
            ProtocolVersion::new(1, 0, 0),
            vec!["compression".to_owned()],
        );
        mgr.register(desc);
        let retrieved = mgr.get_descriptor("proto/c").expect("must exist");
        assert_eq!(retrieved.name, "proto/c");
        assert_eq!(retrieved.version, ProtocolVersion::new(2, 1, 0));
    }

    #[test]
    fn test_supported_protocols_lists_names() {
        let mut mgr = ProtocolVersionManager::new();
        mgr.register(ProtocolDescriptor::new(
            "p1",
            ProtocolVersion::new(1, 0, 0),
            ProtocolVersion::new(1, 0, 0),
            vec![],
        ));
        mgr.register(ProtocolDescriptor::new(
            "p2",
            ProtocolVersion::new(1, 0, 0),
            ProtocolVersion::new(1, 0, 0),
            vec![],
        ));
        let mut names = mgr.supported_protocols();
        names.sort();
        assert_eq!(names, vec!["p1", "p2"]);
    }

    // ── Negotiation ───────────────────────────────────────────────────────

    fn make_mgr_with(
        name: &str,
        local_ver: ProtocolVersion,
        min: ProtocolVersion,
        features: Vec<String>,
    ) -> ProtocolVersionManager {
        let mut mgr = ProtocolVersionManager::new();
        mgr.register(ProtocolDescriptor::new(name, local_ver, min, features));
        mgr
    }

    #[test]
    fn test_negotiate_unknown_protocol_returns_none() {
        let mut mgr = ProtocolVersionManager::new();
        let remote = ProtocolDescriptor::new(
            "unknown",
            ProtocolVersion::new(1, 0, 0),
            ProtocolVersion::new(1, 0, 0),
            vec![],
        );
        assert!(mgr.negotiate("unknown", &remote).is_none());
        assert_eq!(mgr.stats().failed, 1);
        assert_eq!(mgr.stats().negotiations, 1);
    }

    #[test]
    fn test_negotiate_fully_compatible() {
        let mut mgr = make_mgr_with(
            "proto/x",
            ProtocolVersion::new(1, 0, 0),
            ProtocolVersion::new(1, 0, 0),
            vec!["feat-a".to_owned()],
        );
        let remote = ProtocolDescriptor::new(
            "proto/x",
            ProtocolVersion::new(1, 0, 0),
            ProtocolVersion::new(1, 0, 0),
            vec!["feat-a".to_owned()],
        );
        let result = mgr.negotiate("proto/x", &remote).expect("should succeed");
        assert_eq!(result.compatibility, CompatibilityLevel::FullyCompatible);
        assert_eq!(result.agreed_version, ProtocolVersion::new(1, 0, 0));
        assert_eq!(result.common_features, vec!["feat-a"]);
        assert!(result.dropped_features.is_empty());
    }

    #[test]
    fn test_negotiate_backward_compat_remote_older() {
        let mut mgr = make_mgr_with(
            "proto/y",
            ProtocolVersion::new(2, 0, 0),
            ProtocolVersion::new(1, 0, 0),
            vec!["new-feat".to_owned(), "old-feat".to_owned()],
        );
        let remote = ProtocolDescriptor::new(
            "proto/y",
            ProtocolVersion::new(1, 5, 0), // older than local 2.0.0
            ProtocolVersion::new(1, 0, 0),
            vec!["old-feat".to_owned()],
        );
        let result = mgr.negotiate("proto/y", &remote).expect("should succeed");
        assert_eq!(result.compatibility, CompatibilityLevel::BackwardCompatible);
        // agreed at the remote (lower) version
        assert_eq!(result.agreed_version, ProtocolVersion::new(1, 5, 0));
        assert_eq!(result.common_features, vec!["old-feat"]);
        assert_eq!(result.dropped_features, vec!["new-feat"]);
        assert_eq!(mgr.stats().backward_compat, 1);
    }

    #[test]
    fn test_negotiate_forward_compat_remote_newer() {
        let mut mgr = make_mgr_with(
            "proto/z",
            ProtocolVersion::new(1, 0, 0),
            ProtocolVersion::new(1, 0, 0),
            vec!["base".to_owned()],
        );
        let remote = ProtocolDescriptor::new(
            "proto/z",
            ProtocolVersion::new(2, 0, 0), // newer than local 1.0.0
            ProtocolVersion::new(1, 0, 0),
            vec!["base".to_owned(), "advanced".to_owned()],
        );
        let result = mgr.negotiate("proto/z", &remote).expect("should succeed");
        assert_eq!(result.compatibility, CompatibilityLevel::ForwardCompatible);
        // agreed at local (lower) version
        assert_eq!(result.agreed_version, ProtocolVersion::new(1, 0, 0));
        assert_eq!(result.common_features, vec!["base"]);
        assert_eq!(mgr.stats().forward_compat, 1);
    }

    #[test]
    fn test_negotiate_incompatible_remote_below_local_min() {
        let mut mgr = make_mgr_with(
            "proto/q",
            ProtocolVersion::new(3, 0, 0),
            ProtocolVersion::new(2, 0, 0), // min floor is 2.0.0
            vec![],
        );
        let remote = ProtocolDescriptor::new(
            "proto/q",
            ProtocolVersion::new(1, 9, 9), // below our min
            ProtocolVersion::new(1, 0, 0),
            vec![],
        );
        assert!(mgr.negotiate("proto/q", &remote).is_none());
        assert_eq!(mgr.stats().failed, 1);
    }

    #[test]
    fn test_negotiate_incompatible_local_below_remote_min() {
        // Remote requires at least 3.0.0; local only runs 1.0.0.
        let mut mgr = make_mgr_with(
            "proto/r",
            ProtocolVersion::new(1, 0, 0),
            ProtocolVersion::new(1, 0, 0),
            vec![],
        );
        let remote = ProtocolDescriptor::new(
            "proto/r",
            ProtocolVersion::new(3, 0, 0),
            ProtocolVersion::new(3, 0, 0), // remote won't talk < 3.0.0
            vec![],
        );
        assert!(mgr.negotiate("proto/r", &remote).is_none());
        assert_eq!(mgr.stats().failed, 1);
    }

    #[test]
    fn test_negotiate_stats_accumulate() {
        let mut mgr = make_mgr_with(
            "proto/s",
            ProtocolVersion::new(1, 0, 0),
            ProtocolVersion::new(1, 0, 0),
            vec![],
        );
        let good_remote = ProtocolDescriptor::new(
            "proto/s",
            ProtocolVersion::new(1, 0, 0),
            ProtocolVersion::new(1, 0, 0),
            vec![],
        );
        mgr.negotiate("proto/s", &good_remote);
        mgr.negotiate("proto/s", &good_remote);
        mgr.negotiate("unknown", &good_remote);

        let s = mgr.stats();
        assert_eq!(s.negotiations, 3);
        assert_eq!(s.successful, 2);
        assert_eq!(s.failed, 1);
    }

    #[test]
    fn test_negotiate_feature_dropping() {
        let mut mgr = make_mgr_with(
            "proto/t",
            ProtocolVersion::new(1, 0, 0),
            ProtocolVersion::new(1, 0, 0),
            vec!["alpha".to_owned(), "beta".to_owned(), "gamma".to_owned()],
        );
        let remote = ProtocolDescriptor::new(
            "proto/t",
            ProtocolVersion::new(1, 0, 0),
            ProtocolVersion::new(1, 0, 0),
            vec!["alpha".to_owned(), "gamma".to_owned()], // beta missing
        );
        let result = mgr.negotiate("proto/t", &remote).expect("success");
        assert_eq!(result.common_features, vec!["alpha", "gamma"]);
        assert_eq!(result.dropped_features, vec!["beta"]);
    }

    #[test]
    fn test_protocol_name_in_result() {
        let mut mgr = make_mgr_with(
            "ipfrs/custom",
            ProtocolVersion::new(1, 0, 0),
            ProtocolVersion::new(1, 0, 0),
            vec![],
        );
        let remote = ProtocolDescriptor::new(
            "ipfrs/custom",
            ProtocolVersion::new(1, 0, 0),
            ProtocolVersion::new(1, 0, 0),
            vec![],
        );
        let result = mgr.negotiate("ipfrs/custom", &remote).expect("success");
        assert_eq!(result.protocol, "ipfrs/custom");
    }

    #[test]
    fn test_default_manager_has_no_protocols() {
        let mgr = ProtocolVersionManager::default();
        assert!(mgr.supported_protocols().is_empty());
        assert_eq!(mgr.stats().negotiations, 0);
    }
}
