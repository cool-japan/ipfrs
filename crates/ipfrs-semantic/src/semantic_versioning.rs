//! Semantic document/embedding versioning with change detection,
//! compatibility analysis, and migration paths.
//!
//! Provides [`SemanticVersioningEngine`] for managing versioned artifacts with
//! full SemVer 2.0 parsing, compatibility matrices, and changelog tracking.

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors produced by the semantic versioning subsystem.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SemVerError {
    /// A version string could not be parsed.
    ParseError(String),
    /// No artifact with the given ID was found.
    ArtifactNotFound(String),
    /// An artifact with the given ID already exists.
    ArtifactAlreadyExists(String),
    /// A version value is logically invalid.
    InvalidVersion(String),
}

impl fmt::Display for SemVerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SemVerError::ParseError(s) => write!(f, "SemVer parse error: {s}"),
            SemVerError::ArtifactNotFound(s) => write!(f, "artifact not found: {s}"),
            SemVerError::ArtifactAlreadyExists(s) => write!(f, "artifact already exists: {s}"),
            SemVerError::InvalidVersion(s) => write!(f, "invalid version: {s}"),
        }
    }
}

impl std::error::Error for SemVerError {}

// ---------------------------------------------------------------------------
// SemVer
// ---------------------------------------------------------------------------

/// A Semantic Versioning 2.0 version triple plus optional pre-release and
/// build-metadata labels.
///
/// Ordering and equality are based solely on `(major, minor, patch)`.
#[derive(Debug, Clone, Eq)]
pub struct SemVer {
    /// Major version — breaking changes.
    pub major: u32,
    /// Minor version — backward-compatible new features.
    pub minor: u32,
    /// Patch version — backward-compatible bug fixes.
    pub patch: u32,
    /// Optional pre-release identifier (e.g. `"alpha"`, `"beta.1"`).
    pub pre_release: Option<String>,
    /// Optional build metadata (e.g. `"build.1"`, `"20240101"`).
    pub build_metadata: Option<String>,
}

impl SemVer {
    /// Construct a new version without pre-release or build metadata.
    pub fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
            pre_release: None,
            build_metadata: None,
        }
    }

    /// Parse a SemVer string such as `"1.2.3"`, `"1.2.3-alpha"`,
    /// `"1.2.3+build.1"`, or `"1.2.3-alpha+build"`.
    pub fn parse(s: &str) -> Result<SemVer, SemVerError> {
        // Split off build metadata first (after '+')
        let (version_and_pre, build_metadata) = match s.split_once('+') {
            Some((left, right)) => (left, Some(right.to_string())),
            None => (s, None),
        };

        // Split off pre-release (after '-')
        let (version_part, pre_release) = match version_and_pre.split_once('-') {
            Some((left, right)) => (left, Some(right.to_string())),
            None => (version_and_pre, None),
        };

        // Parse X.Y.Z
        let parts: Vec<&str> = version_part.split('.').collect();
        if parts.len() != 3 {
            return Err(SemVerError::ParseError(format!(
                "expected X.Y.Z, got {version_part:?}"
            )));
        }

        let parse_u32 = |raw: &str| -> Result<u32, SemVerError> {
            raw.parse::<u32>().map_err(|_| {
                SemVerError::ParseError(format!("non-numeric version component: {raw:?}"))
            })
        };

        Ok(SemVer {
            major: parse_u32(parts[0])?,
            minor: parse_u32(parts[1])?,
            patch: parse_u32(parts[2])?,
            pre_release,
            build_metadata,
        })
    }

    /// Increment the major version and zero out minor and patch.
    /// Clears pre-release and build metadata.
    pub fn bump_major(&self) -> SemVer {
        SemVer::new(self.major + 1, 0, 0)
    }

    /// Increment the minor version and zero out patch.
    /// Clears pre-release and build metadata.
    pub fn bump_minor(&self) -> SemVer {
        SemVer::new(self.major, self.minor + 1, 0)
    }

    /// Increment the patch version.
    /// Clears pre-release and build metadata.
    pub fn bump_patch(&self) -> SemVer {
        SemVer::new(self.major, self.minor, self.patch + 1)
    }

    /// Returns `true` when `self` is compatible with `other`:
    /// same major version AND `self >= other`.
    pub fn is_compatible_with(&self, other: &SemVer) -> bool {
        self.major == other.major && self >= other
    }
}

impl PartialEq for SemVer {
    fn eq(&self, other: &Self) -> bool {
        (self.major, self.minor, self.patch) == (other.major, other.minor, other.patch)
    }
}

impl PartialOrd for SemVer {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SemVer {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (self.major, self.minor, self.patch).cmp(&(other.major, other.minor, other.patch))
    }
}

impl fmt::Display for SemVer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)?;
        if let Some(pre) = &self.pre_release {
            write!(f, "-{pre}")?;
        }
        if let Some(build) = &self.build_metadata {
            write!(f, "+{build}")?;
        }
        Ok(())
    }
}

impl Default for SemVer {
    fn default() -> Self {
        SemVer::new(0, 1, 0)
    }
}

// ---------------------------------------------------------------------------
// ChangeType / BumpType
// ---------------------------------------------------------------------------

/// The semantic category of a change.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChangeType {
    /// A breaking, backward-incompatible change.
    Breaking,
    /// A new backward-compatible feature.
    Feature,
    /// A backward-compatible bug fix.
    Fix,
    /// Documentation-only change.
    Documentation,
    /// Code restructuring without observable behaviour change.
    Refactor,
}

/// The kind of SemVer bump required to publish a [`ChangeType`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BumpType {
    /// Increment the major component.
    Major,
    /// Increment the minor component.
    Minor,
    /// Increment the patch component.
    Patch,
}

impl ChangeType {
    /// Map this change category to the minimum required version bump.
    pub fn required_bump(self) -> BumpType {
        match self {
            ChangeType::Breaking => BumpType::Major,
            ChangeType::Feature => BumpType::Minor,
            ChangeType::Fix | ChangeType::Documentation | ChangeType::Refactor => BumpType::Patch,
        }
    }
}

// ---------------------------------------------------------------------------
// ChangeRecord
// ---------------------------------------------------------------------------

/// A single entry in an artifact's changelog.
#[derive(Debug, Clone)]
pub struct ChangeRecord {
    /// The version after this change was applied.
    pub version: SemVer,
    /// The semantic category of the change.
    pub change_type: ChangeType,
    /// Human-readable description of the change.
    pub description: String,
    /// Unix timestamp (seconds) when the change was recorded.
    pub timestamp: u64,
}

// ---------------------------------------------------------------------------
// VersionedArtifact
// ---------------------------------------------------------------------------

/// An artifact (e.g. a document or embedding index) tracked by the versioning
/// engine.
#[derive(Debug, Clone)]
pub struct VersionedArtifact {
    /// Stable identifier for this artifact.
    pub id: String,
    /// Current version.
    pub version: SemVer,
    /// FNV-1a hash of the artifact's content bytes.
    pub content_hash: u64,
    /// Dimensionality of the associated embedding, if any.
    pub embedding_dim: Option<usize>,
    /// Unix timestamp (seconds) when the artifact was first registered.
    pub created_at: u64,
    /// Full history of changes applied to this artifact.
    pub changelog: Vec<ChangeRecord>,
}

impl VersionedArtifact {
    /// Compute the FNV-1a 64-bit hash of arbitrary bytes.
    pub fn fnv1a(data: &[u8]) -> u64 {
        const FNV_OFFSET: u64 = 14_695_981_039_346_656_037;
        const FNV_PRIME: u64 = 1_099_511_628_211;
        data.iter().fold(FNV_OFFSET, |acc, &b| {
            (acc ^ (b as u64)).wrapping_mul(FNV_PRIME)
        })
    }
}

// ---------------------------------------------------------------------------
// CompatibilityLevel / CompatibilityMatrix
// ---------------------------------------------------------------------------

/// The degree of compatibility between two major versions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompatibilityLevel {
    /// Both directions are fully compatible.
    FullyCompatible,
    /// Newer code can read data written by older code.
    BackwardCompatible,
    /// Older code can read data written by newer code.
    ForwardCompatible,
    /// No compatibility guarantee.
    Incompatible,
}

/// Stores pairwise compatibility levels keyed by `(from_major, to_major)`.
#[derive(Debug, Clone, Default)]
pub struct CompatibilityMatrix {
    /// Explicit entries; missing pairs default to [`CompatibilityLevel::Incompatible`].
    pub entries: HashMap<(u32, u32), CompatibilityLevel>,
}

impl CompatibilityMatrix {
    /// Return the compatibility level for the given major version pair,
    /// defaulting to `Incompatible` when no entry is found.
    pub fn get(&self, from_major: u32, to_major: u32) -> CompatibilityLevel {
        *self
            .entries
            .get(&(from_major, to_major))
            .unwrap_or(&CompatibilityLevel::Incompatible)
    }

    /// Insert a compatibility entry.
    pub fn set(&mut self, from_major: u32, to_major: u32, level: CompatibilityLevel) {
        self.entries.insert((from_major, to_major), level);
    }
}

// ---------------------------------------------------------------------------
// VersioningStats
// ---------------------------------------------------------------------------

/// Aggregate statistics produced by [`SemanticVersioningEngine::stats`].
#[derive(Debug, Clone)]
pub struct VersioningStats {
    /// Number of artifacts currently registered.
    pub total_artifacts: usize,
    /// Total change records across all artifacts.
    pub total_changes: usize,
    /// Total breaking changes across all artifacts.
    pub breaking_changes: usize,
    /// Mean `(major, minor, patch)` formatted as a SemVer string.
    pub avg_version: String,
    /// Highest version seen across all artifacts, if any artifacts exist.
    pub latest_version: Option<SemVer>,
}

// ---------------------------------------------------------------------------
// SemanticVersioningEngine
// ---------------------------------------------------------------------------

/// Manages versioned artifacts with SemVer-based change detection,
/// compatibility analysis, and migration-path computation.
pub struct SemanticVersioningEngine {
    /// Registered artifacts keyed by their ID.
    pub artifacts: HashMap<String, VersionedArtifact>,
    /// Rules governing cross-version compatibility.
    pub compatibility_rules: CompatibilityMatrix,
}

impl SemanticVersioningEngine {
    /// Create a new engine with sensible default compatibility rules:
    ///
    /// * Same major (distance 0) → [`CompatibilityLevel::FullyCompatible`].
    /// * Adjacent major (distance 1) → [`CompatibilityLevel::BackwardCompatible`].
    /// * All other pairs → [`CompatibilityLevel::Incompatible`].
    ///
    /// Rules are pre-populated for majors `0..=9`.
    pub fn new() -> Self {
        let mut matrix = CompatibilityMatrix::default();
        // Pre-populate for major versions 0..=9
        for m in 0u32..=9 {
            // Same major: fully compatible
            matrix.set(m, m, CompatibilityLevel::FullyCompatible);
            // Adjacent major: backward compatible
            if m > 0 {
                matrix.set(m - 1, m, CompatibilityLevel::BackwardCompatible);
                matrix.set(m, m - 1, CompatibilityLevel::BackwardCompatible);
            }
        }
        Self {
            artifacts: HashMap::new(),
            compatibility_rules: matrix,
        }
    }

    // -----------------------------------------------------------------------
    // Registration
    // -----------------------------------------------------------------------

    /// Register a new artifact.
    ///
    /// Returns [`SemVerError::ArtifactAlreadyExists`] if an artifact with the
    /// same ID is already registered.
    pub fn register_artifact(&mut self, artifact: VersionedArtifact) -> Result<(), SemVerError> {
        if self.artifacts.contains_key(&artifact.id) {
            return Err(SemVerError::ArtifactAlreadyExists(artifact.id.clone()));
        }
        self.artifacts.insert(artifact.id.clone(), artifact);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Mutation
    // -----------------------------------------------------------------------

    /// Apply a change to an artifact, auto-bumping its version according to
    /// [`ChangeType::required_bump`], appending a [`ChangeRecord`], and
    /// returning the new version.
    ///
    /// Returns [`SemVerError::ArtifactNotFound`] when `artifact_id` is
    /// unknown.
    pub fn publish_change(
        &mut self,
        artifact_id: &str,
        change_type: ChangeType,
        description: String,
        now: u64,
    ) -> Result<SemVer, SemVerError> {
        let artifact = self
            .artifacts
            .get_mut(artifact_id)
            .ok_or_else(|| SemVerError::ArtifactNotFound(artifact_id.to_string()))?;

        let new_version = match change_type.required_bump() {
            BumpType::Major => artifact.version.bump_major(),
            BumpType::Minor => artifact.version.bump_minor(),
            BumpType::Patch => artifact.version.bump_patch(),
        };

        let record = ChangeRecord {
            version: new_version.clone(),
            change_type,
            description,
            timestamp: now,
        };

        artifact.version = new_version.clone();
        artifact.changelog.push(record);
        Ok(new_version)
    }

    // -----------------------------------------------------------------------
    // Queries
    // -----------------------------------------------------------------------

    /// Return the current version of an artifact, or `None` if not found.
    pub fn get_version(&self, artifact_id: &str) -> Option<&SemVer> {
        self.artifacts.get(artifact_id).map(|a| &a.version)
    }

    /// Return all changelog entries for an artifact in registration order.
    pub fn version_history(&self, artifact_id: &str) -> Vec<&ChangeRecord> {
        self.artifacts
            .get(artifact_id)
            .map(|a| a.changelog.iter().collect())
            .unwrap_or_default()
    }

    /// Determine the compatibility level between the current versions of two
    /// artifacts.
    ///
    /// Returns errors when either artifact ID is unknown.
    pub fn check_compatibility(
        &self,
        from_id: &str,
        to_id: &str,
    ) -> Result<CompatibilityLevel, SemVerError> {
        let from = self
            .artifacts
            .get(from_id)
            .ok_or_else(|| SemVerError::ArtifactNotFound(from_id.to_string()))?;
        let to = self
            .artifacts
            .get(to_id)
            .ok_or_else(|| SemVerError::ArtifactNotFound(to_id.to_string()))?;

        Ok(self
            .compatibility_rules
            .get(from.version.major, to.version.major))
    }

    /// Return all changelog entries with [`ChangeType::Breaking`] whose
    /// version is strictly greater than `since`.
    pub fn find_breaking_changes(&self, artifact_id: &str, since: &SemVer) -> Vec<&ChangeRecord> {
        self.artifacts
            .get(artifact_id)
            .map(|a| {
                a.changelog
                    .iter()
                    .filter(|r| r.change_type == ChangeType::Breaking && &r.version > since)
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Compute a migration path between two SemVer values.
    ///
    /// Rules:
    /// * Same version → `[from]`
    /// * Same major, different minor → `[from, intermediate, to]` where
    ///   `intermediate` is `(from.major, to.minor, 0)` (i.e. from's major
    ///   with to's minor bumped in).
    /// * Adjacent major (|delta| == 1) → `[from, to]`
    /// * Otherwise → `[]` (incompatible gap, no path)
    pub fn migration_path(&self, from: &SemVer, to: &SemVer) -> Vec<SemVer> {
        if from == to {
            return vec![from.clone()];
        }

        if from.major == to.major {
            if from.minor == to.minor {
                // Only patch differs — direct path
                return vec![from.clone(), to.clone()];
            }
            // Different minor within same major — route through intermediate
            let intermediate = SemVer::new(from.major, to.minor, 0);
            return vec![from.clone(), intermediate, to.clone()];
        }

        let major_delta = (to.major as i64 - from.major as i64).unsigned_abs();
        if major_delta == 1 {
            return vec![from.clone(), to.clone()];
        }

        // Incompatible — no migration path
        vec![]
    }

    /// Return the IDs of all artifacts whose current version equals `version`.
    pub fn artifacts_at_version(&self, version: &SemVer) -> Vec<&str> {
        self.artifacts
            .values()
            .filter(|a| &a.version == version)
            .map(|a| a.id.as_str())
            .collect()
    }

    // -----------------------------------------------------------------------
    // Statistics
    // -----------------------------------------------------------------------

    /// Compute aggregate statistics over all registered artifacts.
    pub fn stats(&self) -> VersioningStats {
        let total_artifacts = self.artifacts.len();

        let mut total_changes = 0usize;
        let mut breaking_changes = 0usize;
        let mut latest_version: Option<SemVer> = None;

        // Accumulators for average version computation
        let mut sum_major = 0u64;
        let mut sum_minor = 0u64;
        let mut sum_patch = 0u64;

        for artifact in self.artifacts.values() {
            let cl = artifact.changelog.len();
            total_changes += cl;
            breaking_changes += artifact
                .changelog
                .iter()
                .filter(|r| r.change_type == ChangeType::Breaking)
                .count();

            sum_major += artifact.version.major as u64;
            sum_minor += artifact.version.minor as u64;
            sum_patch += artifact.version.patch as u64;

            match &latest_version {
                None => latest_version = Some(artifact.version.clone()),
                Some(current) if artifact.version > *current => {
                    latest_version = Some(artifact.version.clone());
                }
                _ => {}
            }
        }

        let avg_version = if total_artifacts == 0 {
            "0.0.0".to_string()
        } else {
            let n = total_artifacts as u64;
            let avg_maj = (sum_major / n) as u32;
            let avg_min = (sum_minor / n) as u32;
            let avg_pat = (sum_patch / n) as u32;
            SemVer::new(avg_maj, avg_min, avg_pat).to_string()
        };

        VersioningStats {
            total_artifacts,
            total_changes,
            breaking_changes,
            avg_version,
            latest_version,
        }
    }
}

impl Default for SemanticVersioningEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use crate::semantic_versioning::{
        BumpType, ChangeRecord, ChangeType, CompatibilityLevel, CompatibilityMatrix, SemVer,
        SemVerError, SemanticVersioningEngine, VersionedArtifact,
    };

    // ------------------------------------------------------------------
    // SemVer::parse
    // ------------------------------------------------------------------

    #[test]
    fn parse_simple() {
        let v = SemVer::parse("1.2.3").expect("valid");
        assert_eq!((v.major, v.minor, v.patch), (1, 2, 3));
        assert!(v.pre_release.is_none());
        assert!(v.build_metadata.is_none());
    }

    #[test]
    fn parse_with_pre_release() {
        let v = SemVer::parse("1.2.3-alpha").expect("valid");
        assert_eq!((v.major, v.minor, v.patch), (1, 2, 3));
        assert_eq!(v.pre_release.as_deref(), Some("alpha"));
        assert!(v.build_metadata.is_none());
    }

    #[test]
    fn parse_with_build_metadata() {
        let v = SemVer::parse("1.2.3+build.1").expect("valid");
        assert_eq!((v.major, v.minor, v.patch), (1, 2, 3));
        assert!(v.pre_release.is_none());
        assert_eq!(v.build_metadata.as_deref(), Some("build.1"));
    }

    #[test]
    fn parse_with_pre_and_build() {
        let v = SemVer::parse("1.2.3-alpha+build").expect("valid");
        assert_eq!((v.major, v.minor, v.patch), (1, 2, 3));
        assert_eq!(v.pre_release.as_deref(), Some("alpha"));
        assert_eq!(v.build_metadata.as_deref(), Some("build"));
    }

    #[test]
    fn parse_zero_version() {
        let v = SemVer::parse("0.0.0").expect("valid");
        assert_eq!((v.major, v.minor, v.patch), (0, 0, 0));
    }

    #[test]
    fn parse_error_missing_parts() {
        assert!(matches!(
            SemVer::parse("1.2"),
            Err(SemVerError::ParseError(_))
        ));
    }

    #[test]
    fn parse_error_non_numeric() {
        assert!(matches!(
            SemVer::parse("a.b.c"),
            Err(SemVerError::ParseError(_))
        ));
    }

    #[test]
    fn parse_error_extra_dots() {
        assert!(matches!(
            SemVer::parse("1.2.3.4"),
            Err(SemVerError::ParseError(_))
        ));
    }

    // ------------------------------------------------------------------
    // SemVer::Display
    // ------------------------------------------------------------------

    #[test]
    fn display_simple() {
        assert_eq!(SemVer::new(1, 2, 3).to_string(), "1.2.3");
    }

    #[test]
    fn display_with_pre() {
        let v = SemVer::parse("2.0.0-beta").expect("valid");
        assert_eq!(v.to_string(), "2.0.0-beta");
    }

    #[test]
    fn display_with_build() {
        let v = SemVer::parse("2.0.0+sha.abc").expect("valid");
        assert_eq!(v.to_string(), "2.0.0+sha.abc");
    }

    #[test]
    fn display_with_pre_and_build() {
        let v = SemVer::parse("2.0.0-rc.1+exp.sha.5114f85").expect("valid");
        assert_eq!(v.to_string(), "2.0.0-rc.1+exp.sha.5114f85");
    }

    // ------------------------------------------------------------------
    // SemVer ordering (ignores pre-release/build)
    // ------------------------------------------------------------------

    #[test]
    fn ordering_major() {
        assert!(SemVer::new(2, 0, 0) > SemVer::new(1, 9, 9));
    }

    #[test]
    fn ordering_minor() {
        assert!(SemVer::new(1, 5, 0) > SemVer::new(1, 4, 99));
    }

    #[test]
    fn ordering_patch() {
        assert!(SemVer::new(1, 0, 2) > SemVer::new(1, 0, 1));
    }

    #[test]
    fn ordering_equal() {
        let a = SemVer::parse("1.2.3-alpha").expect("valid");
        let b = SemVer::parse("1.2.3+build").expect("valid");
        assert_eq!(a, b); // pre/build ignored
    }

    // ------------------------------------------------------------------
    // SemVer bumps
    // ------------------------------------------------------------------

    #[test]
    fn bump_major_resets_minor_patch() {
        let v = SemVer::new(1, 5, 3).bump_major();
        assert_eq!((v.major, v.minor, v.patch), (2, 0, 0));
    }

    #[test]
    fn bump_minor_resets_patch() {
        let v = SemVer::new(1, 5, 3).bump_minor();
        assert_eq!((v.major, v.minor, v.patch), (1, 6, 0));
    }

    #[test]
    fn bump_patch_increments() {
        let v = SemVer::new(1, 5, 3).bump_patch();
        assert_eq!((v.major, v.minor, v.patch), (1, 5, 4));
    }

    #[test]
    fn bump_clears_pre_release() {
        let v = SemVer::parse("1.0.0-alpha").expect("valid");
        let bumped = v.bump_patch();
        assert!(bumped.pre_release.is_none());
    }

    // ------------------------------------------------------------------
    // is_compatible_with
    // ------------------------------------------------------------------

    #[test]
    fn compatible_same_major_newer() {
        let a = SemVer::new(1, 5, 0);
        let b = SemVer::new(1, 3, 0);
        assert!(a.is_compatible_with(&b));
    }

    #[test]
    fn not_compatible_different_major() {
        let a = SemVer::new(2, 0, 0);
        let b = SemVer::new(1, 9, 0);
        assert!(!a.is_compatible_with(&b));
    }

    #[test]
    fn not_compatible_older_than_target() {
        let a = SemVer::new(1, 1, 0);
        let b = SemVer::new(1, 5, 0);
        assert!(!a.is_compatible_with(&b));
    }

    // ------------------------------------------------------------------
    // ChangeType::required_bump
    // ------------------------------------------------------------------

    #[test]
    fn change_type_bump_breaking() {
        assert_eq!(ChangeType::Breaking.required_bump(), BumpType::Major);
    }

    #[test]
    fn change_type_bump_feature() {
        assert_eq!(ChangeType::Feature.required_bump(), BumpType::Minor);
    }

    #[test]
    fn change_type_bump_fix() {
        assert_eq!(ChangeType::Fix.required_bump(), BumpType::Patch);
    }

    #[test]
    fn change_type_bump_documentation() {
        assert_eq!(ChangeType::Documentation.required_bump(), BumpType::Patch);
    }

    #[test]
    fn change_type_bump_refactor() {
        assert_eq!(ChangeType::Refactor.required_bump(), BumpType::Patch);
    }

    // ------------------------------------------------------------------
    // VersionedArtifact::fnv1a
    // ------------------------------------------------------------------

    #[test]
    fn fnv1a_empty() {
        // FNV-1a of empty bytes == FNV offset basis
        assert_eq!(VersionedArtifact::fnv1a(b""), 14_695_981_039_346_656_037);
    }

    #[test]
    fn fnv1a_deterministic() {
        let h1 = VersionedArtifact::fnv1a(b"hello world");
        let h2 = VersionedArtifact::fnv1a(b"hello world");
        assert_eq!(h1, h2);
    }

    #[test]
    fn fnv1a_different_inputs() {
        let h1 = VersionedArtifact::fnv1a(b"foo");
        let h2 = VersionedArtifact::fnv1a(b"bar");
        assert_ne!(h1, h2);
    }

    // ------------------------------------------------------------------
    // CompatibilityMatrix
    // ------------------------------------------------------------------

    #[test]
    fn matrix_default_incompatible() {
        let m = CompatibilityMatrix::default();
        assert_eq!(m.get(0, 5), CompatibilityLevel::Incompatible);
    }

    #[test]
    fn matrix_set_and_get() {
        let mut m = CompatibilityMatrix::default();
        m.set(1, 2, CompatibilityLevel::BackwardCompatible);
        assert_eq!(m.get(1, 2), CompatibilityLevel::BackwardCompatible);
    }

    // ------------------------------------------------------------------
    // SemanticVersioningEngine::register_artifact
    // ------------------------------------------------------------------

    fn make_artifact(id: &str) -> VersionedArtifact {
        VersionedArtifact {
            id: id.to_string(),
            version: SemVer::new(0, 1, 0),
            content_hash: 0,
            embedding_dim: None,
            created_at: 1_000,
            changelog: vec![],
        }
    }

    #[test]
    fn register_new_artifact() {
        let mut engine = SemanticVersioningEngine::new();
        assert!(engine.register_artifact(make_artifact("doc-a")).is_ok());
        assert!(engine.get_version("doc-a").is_some());
    }

    #[test]
    fn register_duplicate_returns_error() {
        let mut engine = SemanticVersioningEngine::new();
        engine
            .register_artifact(make_artifact("doc-a"))
            .expect("first");
        let err = engine
            .register_artifact(make_artifact("doc-a"))
            .unwrap_err();
        assert!(matches!(err, SemVerError::ArtifactAlreadyExists(_)));
    }

    // ------------------------------------------------------------------
    // publish_change
    // ------------------------------------------------------------------

    #[test]
    fn publish_breaking_bumps_major() {
        let mut engine = SemanticVersioningEngine::new();
        engine.register_artifact(make_artifact("a")).expect("ok");
        let v = engine
            .publish_change("a", ChangeType::Breaking, "Removed old API".into(), 2000)
            .expect("ok");
        assert_eq!(v, SemVer::new(1, 0, 0));
    }

    #[test]
    fn publish_feature_bumps_minor() {
        let mut engine = SemanticVersioningEngine::new();
        engine.register_artifact(make_artifact("a")).expect("ok");
        let v = engine
            .publish_change("a", ChangeType::Feature, "New endpoint".into(), 2000)
            .expect("ok");
        assert_eq!(v, SemVer::new(0, 2, 0));
    }

    #[test]
    fn publish_fix_bumps_patch() {
        let mut engine = SemanticVersioningEngine::new();
        engine.register_artifact(make_artifact("a")).expect("ok");
        let v = engine
            .publish_change("a", ChangeType::Fix, "Null ptr fix".into(), 2000)
            .expect("ok");
        assert_eq!(v, SemVer::new(0, 1, 1));
    }

    #[test]
    fn publish_unknown_artifact_errors() {
        let mut engine = SemanticVersioningEngine::new();
        let err = engine
            .publish_change("missing", ChangeType::Fix, "x".into(), 0)
            .unwrap_err();
        assert!(matches!(err, SemVerError::ArtifactNotFound(_)));
    }

    #[test]
    fn publish_appends_changelog() {
        let mut engine = SemanticVersioningEngine::new();
        engine.register_artifact(make_artifact("a")).expect("ok");
        engine
            .publish_change("a", ChangeType::Feature, "feat 1".into(), 1000)
            .expect("ok");
        engine
            .publish_change("a", ChangeType::Fix, "fix 1".into(), 2000)
            .expect("ok");
        let history = engine.version_history("a");
        assert_eq!(history.len(), 2);
    }

    // ------------------------------------------------------------------
    // version_history
    // ------------------------------------------------------------------

    #[test]
    fn history_of_unknown_artifact_is_empty() {
        let engine = SemanticVersioningEngine::new();
        assert!(engine.version_history("ghost").is_empty());
    }

    // ------------------------------------------------------------------
    // check_compatibility
    // ------------------------------------------------------------------

    #[test]
    fn same_major_fully_compatible() {
        let mut engine = SemanticVersioningEngine::new();
        engine.register_artifact(make_artifact("a")).expect("ok");
        engine.register_artifact(make_artifact("b")).expect("ok");
        assert_eq!(
            engine.check_compatibility("a", "b").expect("ok"),
            CompatibilityLevel::FullyCompatible
        );
    }

    #[test]
    fn adjacent_major_backward_compatible() {
        let mut engine = SemanticVersioningEngine::new();
        // Bump "b" to major 1
        engine.register_artifact(make_artifact("a")).expect("ok");
        let mut b = make_artifact("b");
        b.version = SemVer::new(1, 0, 0);
        engine.register_artifact(b).expect("ok");
        assert_eq!(
            engine.check_compatibility("a", "b").expect("ok"),
            CompatibilityLevel::BackwardCompatible
        );
    }

    #[test]
    fn check_compatibility_unknown_from() {
        let mut engine = SemanticVersioningEngine::new();
        engine.register_artifact(make_artifact("b")).expect("ok");
        assert!(matches!(
            engine.check_compatibility("ghost", "b"),
            Err(SemVerError::ArtifactNotFound(_))
        ));
    }

    #[test]
    fn check_compatibility_unknown_to() {
        let mut engine = SemanticVersioningEngine::new();
        engine.register_artifact(make_artifact("a")).expect("ok");
        assert!(matches!(
            engine.check_compatibility("a", "ghost"),
            Err(SemVerError::ArtifactNotFound(_))
        ));
    }

    // ------------------------------------------------------------------
    // find_breaking_changes
    // ------------------------------------------------------------------

    #[test]
    fn find_breaking_changes_empty_when_no_breaking() {
        let mut engine = SemanticVersioningEngine::new();
        engine.register_artifact(make_artifact("a")).expect("ok");
        engine
            .publish_change("a", ChangeType::Feature, "f".into(), 100)
            .expect("ok");
        let bc = engine.find_breaking_changes("a", &SemVer::new(0, 1, 0));
        assert!(bc.is_empty());
    }

    #[test]
    fn find_breaking_changes_returns_breaking_after_since() {
        let mut engine = SemanticVersioningEngine::new();
        engine.register_artifact(make_artifact("a")).expect("ok");
        let since = engine.get_version("a").expect("ok").clone();
        engine
            .publish_change("a", ChangeType::Breaking, "removed X".into(), 100)
            .expect("ok");
        let bc = engine.find_breaking_changes("a", &since);
        assert_eq!(bc.len(), 1);
    }

    #[test]
    fn find_breaking_changes_not_included_at_or_before_since() {
        let mut engine = SemanticVersioningEngine::new();
        engine.register_artifact(make_artifact("a")).expect("ok");
        engine
            .publish_change("a", ChangeType::Breaking, "removed X".into(), 100)
            .expect("ok");
        let since = engine.get_version("a").expect("ok").clone();
        // No new breaking changes after `since`
        let bc = engine.find_breaking_changes("a", &since);
        assert!(bc.is_empty());
    }

    // ------------------------------------------------------------------
    // migration_path
    // ------------------------------------------------------------------

    #[test]
    fn migration_path_same_version() {
        let engine = SemanticVersioningEngine::new();
        let v = SemVer::new(1, 2, 3);
        let path = engine.migration_path(&v, &v);
        assert_eq!(path, vec![v]);
    }

    #[test]
    fn migration_path_same_major_different_minor() {
        let engine = SemanticVersioningEngine::new();
        let from = SemVer::new(1, 1, 0);
        let to = SemVer::new(1, 4, 0);
        let path = engine.migration_path(&from, &to);
        assert_eq!(path.len(), 3);
        assert_eq!(path[1], SemVer::new(1, 4, 0)); // intermediate with to's minor
    }

    #[test]
    fn migration_path_adjacent_major() {
        let engine = SemanticVersioningEngine::new();
        let from = SemVer::new(1, 5, 0);
        let to = SemVer::new(2, 0, 0);
        let path = engine.migration_path(&from, &to);
        assert_eq!(path, vec![from, to]);
    }

    #[test]
    fn migration_path_incompatible_gap() {
        let engine = SemanticVersioningEngine::new();
        let from = SemVer::new(1, 0, 0);
        let to = SemVer::new(5, 0, 0);
        let path = engine.migration_path(&from, &to);
        assert!(path.is_empty());
    }

    #[test]
    fn migration_path_same_minor_different_patch() {
        let engine = SemanticVersioningEngine::new();
        let from = SemVer::new(2, 3, 0);
        let to = SemVer::new(2, 3, 5);
        let path = engine.migration_path(&from, &to);
        assert_eq!(path, vec![from, to]);
    }

    // ------------------------------------------------------------------
    // artifacts_at_version
    // ------------------------------------------------------------------

    #[test]
    fn artifacts_at_version_matches() {
        let mut engine = SemanticVersioningEngine::new();
        engine.register_artifact(make_artifact("a")).expect("ok");
        engine.register_artifact(make_artifact("b")).expect("ok");
        let mut ids = engine.artifacts_at_version(&SemVer::new(0, 1, 0));
        ids.sort_unstable();
        assert_eq!(ids, vec!["a", "b"]);
    }

    #[test]
    fn artifacts_at_version_empty_when_none_match() {
        let mut engine = SemanticVersioningEngine::new();
        engine.register_artifact(make_artifact("a")).expect("ok");
        let ids = engine.artifacts_at_version(&SemVer::new(9, 9, 9));
        assert!(ids.is_empty());
    }

    // ------------------------------------------------------------------
    // stats
    // ------------------------------------------------------------------

    #[test]
    fn stats_empty_engine() {
        let engine = SemanticVersioningEngine::new();
        let s = engine.stats();
        assert_eq!(s.total_artifacts, 0);
        assert_eq!(s.total_changes, 0);
        assert_eq!(s.breaking_changes, 0);
        assert_eq!(s.avg_version, "0.0.0");
        assert!(s.latest_version.is_none());
    }

    #[test]
    fn stats_single_artifact_no_changes() {
        let mut engine = SemanticVersioningEngine::new();
        engine.register_artifact(make_artifact("a")).expect("ok");
        let s = engine.stats();
        assert_eq!(s.total_artifacts, 1);
        assert_eq!(s.total_changes, 0);
        assert_eq!(s.breaking_changes, 0);
        assert_eq!(s.latest_version, Some(SemVer::new(0, 1, 0)));
    }

    #[test]
    fn stats_counts_breaking_changes() {
        let mut engine = SemanticVersioningEngine::new();
        engine.register_artifact(make_artifact("a")).expect("ok");
        engine
            .publish_change("a", ChangeType::Breaking, "b1".into(), 100)
            .expect("ok");
        engine
            .publish_change("a", ChangeType::Feature, "f1".into(), 200)
            .expect("ok");
        let s = engine.stats();
        assert_eq!(s.total_changes, 2);
        assert_eq!(s.breaking_changes, 1);
    }

    #[test]
    fn stats_latest_version_is_max() {
        let mut engine = SemanticVersioningEngine::new();
        engine.register_artifact(make_artifact("a")).expect("ok");
        let mut b = make_artifact("b");
        b.version = SemVer::new(3, 0, 0);
        engine.register_artifact(b).expect("ok");
        let s = engine.stats();
        assert_eq!(s.latest_version, Some(SemVer::new(3, 0, 0)));
    }

    #[test]
    fn stats_avg_version_two_artifacts() {
        let mut engine = SemanticVersioningEngine::new();
        // versions 0.1.0 and 2.3.4 → avg = 1.2.2
        engine.register_artifact(make_artifact("a")).expect("ok");
        let mut b = make_artifact("b");
        b.version = SemVer::new(2, 3, 4);
        engine.register_artifact(b).expect("ok");
        let s = engine.stats();
        // floor(0+2/2)=1, floor(1+3/2)=2, floor(0+4/2)=2
        assert_eq!(s.avg_version, "1.2.2");
    }

    // ------------------------------------------------------------------
    // ChangeRecord presence in history
    // ------------------------------------------------------------------

    #[test]
    fn change_record_fields_preserved() {
        let mut engine = SemanticVersioningEngine::new();
        engine.register_artifact(make_artifact("a")).expect("ok");
        engine
            .publish_change(
                "a",
                ChangeType::Documentation,
                "Update readme".into(),
                42_000,
            )
            .expect("ok");
        let history = engine.version_history("a");
        let rec: &ChangeRecord = history[0];
        assert_eq!(rec.change_type, ChangeType::Documentation);
        assert_eq!(rec.description, "Update readme");
        assert_eq!(rec.timestamp, 42_000);
    }

    // ------------------------------------------------------------------
    // Default impl
    // ------------------------------------------------------------------

    #[test]
    fn default_engine_is_empty() {
        let engine = SemanticVersioningEngine::default();
        assert!(engine.artifacts.is_empty());
    }

    // ------------------------------------------------------------------
    // SemVerError Display
    // ------------------------------------------------------------------

    #[test]
    fn error_display_parse() {
        let e = SemVerError::ParseError("bad".into());
        assert!(e.to_string().contains("bad"));
    }

    #[test]
    fn error_display_not_found() {
        let e = SemVerError::ArtifactNotFound("x".into());
        assert!(e.to_string().contains("x"));
    }

    #[test]
    fn error_display_already_exists() {
        let e = SemVerError::ArtifactAlreadyExists("x".into());
        assert!(e.to_string().contains("x"));
    }

    #[test]
    fn error_display_invalid_version() {
        let e = SemVerError::InvalidVersion("0.0.0".into());
        assert!(e.to_string().contains("0.0.0"));
    }
}
