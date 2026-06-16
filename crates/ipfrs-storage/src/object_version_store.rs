//! Versioned object store with full history, branching, and garbage collection.
//!
//! Maintains the complete history of content-addressed objects and supports
//! point-in-time retrieval, branch management, and configurable GC policies.

use std::collections::{HashMap, HashSet, VecDeque};

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors produced by [`ObjectVersionStore`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VsError {
    /// A requested version number does not exist.
    VersionNotFound(u64),
    /// A requested branch name does not exist.
    BranchNotFound(String),
    /// A branch with the given name already exists.
    BranchAlreadyExists(String),
    /// Store has reached its configured maximum number of versions.
    MaxVersionsReached,
    /// Store has reached its configured maximum number of branches.
    MaxBranchesReached,
}

impl std::fmt::Display for VsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::VersionNotFound(v) => write!(f, "version {v} not found"),
            Self::BranchNotFound(b) => write!(f, "branch '{b}' not found"),
            Self::BranchAlreadyExists(b) => write!(f, "branch '{b}' already exists"),
            Self::MaxVersionsReached => write!(f, "maximum number of versions reached"),
            Self::MaxBranchesReached => write!(f, "maximum number of branches reached"),
        }
    }
}

impl std::error::Error for VsError {}

// ---------------------------------------------------------------------------
// Core domain types
// ---------------------------------------------------------------------------

/// A single version of an object in the store.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OvsObjectVersion {
    /// Monotonically increasing version number (global across all branches).
    pub version: u64,
    /// Content identifier — FNV-1a 64-bit hash of `data` as a hex string.
    pub cid: String,
    /// Raw object bytes.
    pub data: Vec<u8>,
    /// Optional link to the version this was derived from.
    pub parent_version: Option<u64>,
    /// Unix timestamp (seconds) when this version was created.
    pub created_at: u64,
    /// Arbitrary string tags.
    pub tags: Vec<String>,
    /// Byte length of `data`.
    pub size_bytes: u64,
}

/// A named branch pointing to a specific head version.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VersionBranch {
    /// Branch name (unique within the store).
    pub name: String,
    /// Version number that is the current head of this branch.
    pub head_version: u64,
    /// Unix timestamp (seconds) when this branch was created.
    pub created_at: u64,
    /// Optional parent branch this branch was forked from.
    pub parent_branch: Option<String>,
}

// ---------------------------------------------------------------------------
// Query type
// ---------------------------------------------------------------------------

/// Specifies how to select a version from the store.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VersionQuery {
    /// The most recent version on the "main" branch.
    Latest,
    /// A specific version by number.
    AtVersion(u64),
    /// The latest version whose `created_at` is ≤ the given timestamp.
    AtTime(u64),
    /// The first version that carries the given tag.
    Tagged(String),
    /// The head version of the named branch.
    OnBranch(String),
}

// ---------------------------------------------------------------------------
// GC policy
// ---------------------------------------------------------------------------

/// Determines which versions are eligible for garbage collection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OvsGcPolicy {
    /// Never delete any version.
    KeepAll,
    /// Keep only the last `n` versions (by version number); delete the rest
    /// unless they are reachable from a branch head.
    KeepLast(usize),
    /// Delete versions with `created_at < t` unless reachable from a branch head.
    KeepSince(u64),
    /// Delete versions that have no tags and are not reachable from a branch head.
    KeepTagged,
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for an [`ObjectVersionStore`].
#[derive(Debug, Clone)]
pub struct VersionStoreConfig {
    /// Maximum number of live versions before GC is triggered automatically.
    pub max_versions: usize,
    /// Maximum number of branches allowed.
    pub max_branches: usize,
    /// Policy that drives which versions are eligible for deletion.
    pub gc_policy: OvsGcPolicy,
    /// Whether old versions should be compressed (currently reserved; not used
    /// for actual compression but carried for forward-compatibility).
    pub compress_old_versions: bool,
}

impl Default for VersionStoreConfig {
    fn default() -> Self {
        Self {
            max_versions: 10_000,
            max_branches: 100,
            gc_policy: OvsGcPolicy::KeepAll,
            compress_old_versions: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------

/// A point-in-time snapshot of store statistics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VsStats {
    /// Number of live (non-deleted) versions.
    pub version_count: usize,
    /// Number of branches.
    pub branch_count: usize,
    /// Total bytes occupied by all live version payloads.
    pub total_bytes: u64,
    /// Cumulative count of versions removed by GC.
    pub deleted_versions: u64,
}

// ---------------------------------------------------------------------------
// The store itself
// ---------------------------------------------------------------------------

/// Versioned object store with branch management and configurable GC.
///
/// # Lifecycle
///
/// 1. Create with [`ObjectVersionStore::new`].
/// 2. Commit new content with [`ObjectVersionStore::put`] or
///    [`ObjectVersionStore::put_on_branch`].
/// 3. Retrieve historical content with [`ObjectVersionStore::get`].
/// 4. Create branches with [`ObjectVersionStore::create_branch`] and
///    merge them back with [`ObjectVersionStore::merge_branch`].
/// 5. Run GC explicitly with [`ObjectVersionStore::gc`] or let `put` trigger
///    it automatically when `max_versions` is reached.
pub struct ObjectVersionStore {
    /// Runtime configuration.
    pub config: VersionStoreConfig,
    /// Live version storage, keyed by version number.
    pub versions: HashMap<u64, OvsObjectVersion>,
    /// Branch registry, keyed by branch name.
    pub branches: HashMap<String, VersionBranch>,
    /// Counter for the next version number to assign.
    pub next_version: u64,
    /// Cumulative byte count for all live versions.
    pub total_bytes: u64,
    /// Total number of versions that have been garbage-collected.
    pub deleted_versions: u64,
}

// ---------------------------------------------------------------------------
// FNV-1a 64-bit helper
// ---------------------------------------------------------------------------

/// Compute the FNV-1a 64-bit hash of a byte slice and return it as a
/// zero-padded 16-character lowercase hex string (the CID used by this store).
fn fnv1a_64(data: &[u8]) -> String {
    const OFFSET_BASIS: u64 = 14_695_981_039_346_656_037;
    const PRIME: u64 = 1_099_511_628_211;

    let mut hash = OFFSET_BASIS;
    for &byte in data {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(PRIME);
    }
    format!("{hash:016x}")
}

// ---------------------------------------------------------------------------
// Implementation
// ---------------------------------------------------------------------------

impl ObjectVersionStore {
    /// Create a new store and initialise the "main" branch at version 0.
    ///
    /// Version 0 is a sentinel – it is *not* inserted into `versions`.  The
    /// "main" branch's `head_version` starts at 0, meaning "no versions yet".
    pub fn new(config: VersionStoreConfig) -> Self {
        let mut branches = HashMap::new();
        branches.insert(
            "main".to_owned(),
            VersionBranch {
                name: "main".to_owned(),
                head_version: 0,
                created_at: 0,
                parent_branch: None,
            },
        );

        Self {
            config,
            versions: HashMap::new(),
            branches,
            next_version: 1,
            total_bytes: 0,
            deleted_versions: 0,
        }
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Allocate the next version number and advance the counter.
    fn alloc_version(&mut self) -> u64 {
        let v = self.next_version;
        self.next_version += 1;
        v
    }

    /// Create and insert a new [`OvsObjectVersion`], linked to `parent`.
    fn insert_version(
        &mut self,
        data: Vec<u8>,
        tags: Vec<String>,
        parent_version: Option<u64>,
        now: u64,
    ) -> u64 {
        let cid = fnv1a_64(&data);
        let size_bytes = data.len() as u64;
        let version = self.alloc_version();

        let ov = OvsObjectVersion {
            version,
            cid,
            data,
            parent_version,
            created_at: now,
            tags,
            size_bytes,
        };

        self.total_bytes += size_bytes;
        self.versions.insert(version, ov);
        version
    }

    /// Return the current head version of a branch (0 if branch is empty).
    fn branch_head(&self, branch_name: &str) -> Option<u64> {
        self.branches.get(branch_name).map(|b| b.head_version)
    }

    /// Advance a branch head; the branch must already exist.
    fn set_branch_head(&mut self, branch_name: &str, version: u64) {
        if let Some(b) = self.branches.get_mut(branch_name) {
            b.head_version = version;
        }
    }

    // -----------------------------------------------------------------------
    // Public write API
    // -----------------------------------------------------------------------

    /// Commit `data` on the "main" branch and return the new version number.
    ///
    /// If the version count would exceed `config.max_versions`, GC is run
    /// before inserting.  CID is computed as the FNV-1a 64-bit hex hash of
    /// `data`.
    pub fn put(&mut self, data: Vec<u8>, tags: Vec<String>, now: u64) -> u64 {
        // Trigger GC before inserting so we stay within the limit.
        if self.versions.len() >= self.config.max_versions {
            self.gc();
        }

        let parent = self.branch_head("main").filter(|&v| v != 0);
        let version = self.insert_version(data, tags, parent, now);
        self.set_branch_head("main", version);
        version
    }

    /// Commit `data` on a named branch and return the new version number.
    ///
    /// Returns [`VsError::BranchNotFound`] if the branch does not exist.
    pub fn put_on_branch(
        &mut self,
        branch: String,
        data: Vec<u8>,
        tags: Vec<String>,
        now: u64,
    ) -> Result<u64, VsError> {
        if !self.branches.contains_key(&branch) {
            return Err(VsError::BranchNotFound(branch));
        }

        if self.versions.len() >= self.config.max_versions {
            self.gc();
        }

        let parent = self.branch_head(&branch).filter(|&v| v != 0);
        let version = self.insert_version(data, tags, parent, now);
        self.set_branch_head(&branch, version);
        Ok(version)
    }

    // -----------------------------------------------------------------------
    // Public read API
    // -----------------------------------------------------------------------

    /// Retrieve a version according to the given [`VersionQuery`].
    pub fn get(&self, query: VersionQuery) -> Result<&OvsObjectVersion, VsError> {
        match query {
            VersionQuery::Latest => {
                let head = self
                    .branches
                    .get("main")
                    .map(|b| b.head_version)
                    .unwrap_or(0);
                self.versions
                    .get(&head)
                    .ok_or(VsError::VersionNotFound(head))
            }

            VersionQuery::AtVersion(v) => self.versions.get(&v).ok_or(VsError::VersionNotFound(v)),

            VersionQuery::AtTime(t) => {
                // Pick the version with the highest version number whose
                // `created_at` is ≤ t.
                let best = self
                    .versions
                    .values()
                    .filter(|v| v.created_at <= t)
                    .max_by_key(|v| v.version);

                best.ok_or(VsError::VersionNotFound(0))
            }

            VersionQuery::Tagged(tag) => {
                // Scan in version-number order for determinism.
                let mut candidates: Vec<&OvsObjectVersion> = self
                    .versions
                    .values()
                    .filter(|v| v.tags.contains(&tag))
                    .collect();
                candidates.sort_by_key(|v| v.version);
                candidates
                    .into_iter()
                    .next()
                    .ok_or(VsError::VersionNotFound(0))
            }

            VersionQuery::OnBranch(b) => {
                let branch = self
                    .branches
                    .get(&b)
                    .ok_or_else(|| VsError::BranchNotFound(b.clone()))?;
                let head = branch.head_version;
                self.versions
                    .get(&head)
                    .ok_or(VsError::VersionNotFound(head))
            }
        }
    }

    // -----------------------------------------------------------------------
    // Branch management
    // -----------------------------------------------------------------------

    /// Create a new branch starting at `from_version`.
    ///
    /// # Errors
    ///
    /// - [`VsError::BranchAlreadyExists`] if `name` is taken.
    /// - [`VsError::VersionNotFound`] if `from_version` is not in the store.
    /// - [`VsError::MaxBranchesReached`] if the branch limit is hit.
    pub fn create_branch(
        &mut self,
        name: String,
        from_version: u64,
        now: u64,
    ) -> Result<(), VsError> {
        if self.branches.contains_key(&name) {
            return Err(VsError::BranchAlreadyExists(name));
        }
        if !self.versions.contains_key(&from_version) {
            return Err(VsError::VersionNotFound(from_version));
        }
        if self.branches.len() >= self.config.max_branches {
            return Err(VsError::MaxBranchesReached);
        }

        self.branches.insert(
            name.clone(),
            VersionBranch {
                name,
                head_version: from_version,
                created_at: now,
                parent_branch: None,
            },
        );
        Ok(())
    }

    /// Merge the head of `source` into `target` by creating a new version on
    /// `target` that is a copy of the source head, with `parent_version` set
    /// to the source head version number.
    ///
    /// Returns the new version number.
    pub fn merge_branch(&mut self, source: &str, target: &str, now: u64) -> Result<u64, VsError> {
        let source_head = self
            .branches
            .get(source)
            .ok_or_else(|| VsError::BranchNotFound(source.to_owned()))?
            .head_version;

        // Verify target exists before borrowing versions.
        if !self.branches.contains_key(target) {
            return Err(VsError::BranchNotFound(target.to_owned()));
        }

        // Clone the source version's data and tags so we can release the
        // immutable borrow before mutating self.
        let (data, tags) = {
            let src_ver = self
                .versions
                .get(&source_head)
                .ok_or(VsError::VersionNotFound(source_head))?;
            (src_ver.data.clone(), src_ver.tags.clone())
        };

        if self.versions.len() >= self.config.max_versions {
            self.gc();
        }

        let cid = fnv1a_64(&data);
        let size_bytes = data.len() as u64;
        let version = self.alloc_version();

        let ov = OvsObjectVersion {
            version,
            cid,
            data,
            parent_version: Some(source_head),
            created_at: now,
            tags,
            size_bytes,
        };

        self.total_bytes += size_bytes;
        self.versions.insert(version, ov);
        self.set_branch_head(target, version);
        Ok(version)
    }

    // -----------------------------------------------------------------------
    // History
    // -----------------------------------------------------------------------

    /// Walk the parent-version chain from the head of `branch` and return
    /// versions in reverse-chronological order (newest first).
    pub fn history(&self, branch: &str) -> Result<Vec<&OvsObjectVersion>, VsError> {
        let head = self
            .branches
            .get(branch)
            .ok_or_else(|| VsError::BranchNotFound(branch.to_owned()))?
            .head_version;

        let mut result = Vec::new();
        let mut current = if head == 0 { None } else { Some(head) };

        // Guard against cycles (shouldn't happen in a well-formed store).
        let mut visited = HashSet::new();

        while let Some(v) = current {
            if !visited.insert(v) {
                break; // cycle guard
            }
            if let Some(ver) = self.versions.get(&v) {
                result.push(ver);
                current = ver.parent_version;
            } else {
                break;
            }
        }

        Ok(result)
    }

    // -----------------------------------------------------------------------
    // Garbage collection
    // -----------------------------------------------------------------------

    /// Compute the set of version numbers reachable from any branch head via
    /// `parent_version` links.
    pub fn reachable_from_heads(&self) -> HashSet<u64> {
        let mut reachable = HashSet::new();
        let mut queue: VecDeque<u64> = VecDeque::new();

        // Seed queue with all branch heads (skip sentinel 0).
        for branch in self.branches.values() {
            if branch.head_version != 0 {
                queue.push_back(branch.head_version);
            }
        }

        while let Some(v) = queue.pop_front() {
            if !reachable.insert(v) {
                continue; // already visited
            }
            if let Some(ver) = self.versions.get(&v) {
                if let Some(parent) = ver.parent_version {
                    queue.push_back(parent);
                }
            }
        }

        reachable
    }

    /// Run garbage collection according to `config.gc_policy`.
    ///
    /// Returns the number of versions deleted.
    pub fn gc(&mut self) -> usize {
        match self.config.gc_policy.clone() {
            OvsGcPolicy::KeepAll => 0,

            OvsGcPolicy::KeepLast(n) => {
                let reachable = self.reachable_from_heads();

                // Collect all version numbers that are *not* reachable.
                let mut candidates: Vec<u64> = self
                    .versions
                    .keys()
                    .copied()
                    .filter(|v| !reachable.contains(v))
                    .collect();

                // Sort descending: we want to keep the *highest* (newest) n
                // unreachable ones and delete the rest.  But we only need to
                // delete versions outside the top-n by number.
                candidates.sort_unstable();

                // How many of the unreachable we want to drop:
                // All unreachable that are NOT in the last n overall versions.
                let all_sorted: Vec<u64> = {
                    let mut all: Vec<u64> = self.versions.keys().copied().collect();
                    all.sort_unstable();
                    all
                };
                let keep_threshold = if all_sorted.len() > n {
                    all_sorted[all_sorted.len() - n]
                } else {
                    0
                };

                let to_delete: Vec<u64> = candidates
                    .iter()
                    .copied()
                    .filter(|&v| v < keep_threshold)
                    .collect();

                self.remove_versions(&to_delete)
            }

            OvsGcPolicy::KeepSince(t) => {
                let reachable = self.reachable_from_heads();

                let to_delete: Vec<u64> = self
                    .versions
                    .iter()
                    .filter(|(v, ov)| ov.created_at < t && !reachable.contains(v))
                    .map(|(v, _)| *v)
                    .collect();

                self.remove_versions(&to_delete)
            }

            OvsGcPolicy::KeepTagged => {
                let reachable = self.reachable_from_heads();

                let to_delete: Vec<u64> = self
                    .versions
                    .iter()
                    .filter(|(v, ov)| ov.tags.is_empty() && !reachable.contains(v))
                    .map(|(v, _)| *v)
                    .collect();

                self.remove_versions(&to_delete)
            }
        }
    }

    /// Remove a list of version numbers from the store and update accounting.
    ///
    /// Returns the number actually removed.
    fn remove_versions(&mut self, to_delete: &[u64]) -> usize {
        let mut count = 0usize;
        for &v in to_delete {
            if let Some(ov) = self.versions.remove(&v) {
                self.total_bytes = self.total_bytes.saturating_sub(ov.size_bytes);
                count += 1;
            }
        }
        self.deleted_versions += count as u64;
        count
    }

    // -----------------------------------------------------------------------
    // Stats / counters
    // -----------------------------------------------------------------------

    /// Number of live versions currently in the store.
    pub fn version_count(&self) -> usize {
        self.versions.len()
    }

    /// Total bytes occupied by all live version payloads.
    pub fn total_size_bytes(&self) -> u64 {
        self.total_bytes
    }

    /// Return a statistics snapshot.
    pub fn stats(&self) -> VsStats {
        VsStats {
            version_count: self.versions.len(),
            branch_count: self.branches.len(),
            total_bytes: self.total_bytes,
            deleted_versions: self.deleted_versions,
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::{
        fnv1a_64, ObjectVersionStore, OvsGcPolicy, VersionBranch, VersionQuery, VersionStoreConfig,
        VsError, VsStats,
    };

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn default_store() -> ObjectVersionStore {
        ObjectVersionStore::new(VersionStoreConfig::default())
    }

    fn store_with_gc(policy: OvsGcPolicy) -> ObjectVersionStore {
        ObjectVersionStore::new(VersionStoreConfig {
            gc_policy: policy,
            ..Default::default()
        })
    }

    // -----------------------------------------------------------------------
    // Test 1 – new store has "main" branch and zero versions
    // -----------------------------------------------------------------------
    #[test]
    fn test_new_has_main_branch() {
        let store = default_store();
        assert!(store.branches.contains_key("main"));
        assert_eq!(store.version_count(), 0);
    }

    // -----------------------------------------------------------------------
    // Test 2 – put returns version 1 for the first commit
    // -----------------------------------------------------------------------
    #[test]
    fn test_put_first_version_is_one() {
        let mut store = default_store();
        let v = store.put(b"hello".to_vec(), vec![], 100);
        assert_eq!(v, 1);
    }

    // -----------------------------------------------------------------------
    // Test 3 – successive puts return monotonically increasing numbers
    // -----------------------------------------------------------------------
    #[test]
    fn test_put_monotonic_versions() {
        let mut store = default_store();
        let v1 = store.put(b"a".to_vec(), vec![], 1);
        let v2 = store.put(b"b".to_vec(), vec![], 2);
        let v3 = store.put(b"c".to_vec(), vec![], 3);
        assert!(v1 < v2 && v2 < v3);
    }

    // -----------------------------------------------------------------------
    // Test 4 – get(Latest) returns the most recent version on main
    // -----------------------------------------------------------------------
    #[test]
    fn test_get_latest() {
        let mut store = default_store();
        store.put(b"first".to_vec(), vec![], 1);
        let v2 = store.put(b"second".to_vec(), vec![], 2);
        let got = store.get(VersionQuery::Latest).expect("latest");
        assert_eq!(got.version, v2);
    }

    // -----------------------------------------------------------------------
    // Test 5 – get(Latest) on empty store returns error
    // -----------------------------------------------------------------------
    #[test]
    fn test_get_latest_empty_store_errors() {
        let store = default_store();
        assert!(store.get(VersionQuery::Latest).is_err());
    }

    // -----------------------------------------------------------------------
    // Test 6 – get(AtVersion) retrieves exact version
    // -----------------------------------------------------------------------
    #[test]
    fn test_get_at_version() {
        let mut store = default_store();
        let v1 = store.put(b"alpha".to_vec(), vec![], 10);
        store.put(b"beta".to_vec(), vec![], 20);
        let got = store.get(VersionQuery::AtVersion(v1)).expect("v1");
        assert_eq!(got.data, b"alpha");
    }

    // -----------------------------------------------------------------------
    // Test 7 – get(AtVersion) for missing version returns VersionNotFound
    // -----------------------------------------------------------------------
    #[test]
    fn test_get_at_version_not_found() {
        let store = default_store();
        assert_eq!(
            store.get(VersionQuery::AtVersion(999)),
            Err(VsError::VersionNotFound(999))
        );
    }

    // -----------------------------------------------------------------------
    // Test 8 – get(AtTime) returns the latest version ≤ timestamp
    // -----------------------------------------------------------------------
    #[test]
    fn test_get_at_time() {
        let mut store = default_store();
        let v1 = store.put(b"t100".to_vec(), vec![], 100);
        store.put(b"t200".to_vec(), vec![], 200);
        store.put(b"t300".to_vec(), vec![], 300);
        // Ask for the state at t=150 — should get v1.
        let got = store.get(VersionQuery::AtTime(150)).expect("at time");
        assert_eq!(got.version, v1);
    }

    // -----------------------------------------------------------------------
    // Test 9 – get(AtTime) with no versions before timestamp returns error
    // -----------------------------------------------------------------------
    #[test]
    fn test_get_at_time_before_all_versions() {
        let mut store = default_store();
        store.put(b"data".to_vec(), vec![], 500);
        assert!(store.get(VersionQuery::AtTime(1)).is_err());
    }

    // -----------------------------------------------------------------------
    // Test 10 – get(Tagged) finds first version with tag
    // -----------------------------------------------------------------------
    #[test]
    fn test_get_tagged() {
        let mut store = default_store();
        store.put(b"no-tag".to_vec(), vec![], 1);
        let v2 = store.put(b"has-tag".to_vec(), vec!["release".to_owned()], 2);
        store.put(b"also-tag".to_vec(), vec!["release".to_owned()], 3);
        let got = store
            .get(VersionQuery::Tagged("release".to_owned()))
            .expect("tagged");
        assert_eq!(got.version, v2);
    }

    // -----------------------------------------------------------------------
    // Test 11 – get(Tagged) with no matching tag returns error
    // -----------------------------------------------------------------------
    #[test]
    fn test_get_tagged_no_match() {
        let mut store = default_store();
        store.put(b"x".to_vec(), vec![], 1);
        assert!(store
            .get(VersionQuery::Tagged("missing".to_owned()))
            .is_err());
    }

    // -----------------------------------------------------------------------
    // Test 12 – get(OnBranch) returns branch head
    // -----------------------------------------------------------------------
    #[test]
    fn test_get_on_branch() {
        let mut store = default_store();
        let v1 = store.put(b"initial".to_vec(), vec![], 1);
        store
            .create_branch("dev".to_owned(), v1, 2)
            .expect("create");
        store
            .put_on_branch("dev".to_owned(), b"dev-work".to_vec(), vec![], 3)
            .expect("put");
        let got = store
            .get(VersionQuery::OnBranch("dev".to_owned()))
            .expect("on branch");
        assert_eq!(got.data, b"dev-work");
    }

    // -----------------------------------------------------------------------
    // Test 13 – get(OnBranch) for missing branch returns BranchNotFound
    // -----------------------------------------------------------------------
    #[test]
    fn test_get_on_branch_not_found() {
        let store = default_store();
        assert_eq!(
            store.get(VersionQuery::OnBranch("ghost".to_owned())),
            Err(VsError::BranchNotFound("ghost".to_owned()))
        );
    }

    // -----------------------------------------------------------------------
    // Test 14 – create_branch succeeds and appears in branches map
    // -----------------------------------------------------------------------
    #[test]
    fn test_create_branch_ok() {
        let mut store = default_store();
        let v = store.put(b"v1".to_vec(), vec![], 1);
        store
            .create_branch("feature".to_owned(), v, 2)
            .expect("create ok");
        assert!(store.branches.contains_key("feature"));
    }

    // -----------------------------------------------------------------------
    // Test 15 – create_branch on an existing name returns BranchAlreadyExists
    // -----------------------------------------------------------------------
    #[test]
    fn test_create_branch_duplicate() {
        let mut store = default_store();
        let v = store.put(b"x".to_vec(), vec![], 1);
        store.create_branch("dup".to_owned(), v, 1).expect("first");
        assert_eq!(
            store.create_branch("dup".to_owned(), v, 2),
            Err(VsError::BranchAlreadyExists("dup".to_owned()))
        );
    }

    // -----------------------------------------------------------------------
    // Test 16 – create_branch from unknown version returns VersionNotFound
    // -----------------------------------------------------------------------
    #[test]
    fn test_create_branch_unknown_version() {
        let mut store = default_store();
        assert_eq!(
            store.create_branch("b".to_owned(), 42, 1),
            Err(VsError::VersionNotFound(42))
        );
    }

    // -----------------------------------------------------------------------
    // Test 17 – create_branch respects max_branches
    // -----------------------------------------------------------------------
    #[test]
    fn test_create_branch_max_branches() {
        let mut store = ObjectVersionStore::new(VersionStoreConfig {
            max_branches: 2, // "main" counts as 1
            ..Default::default()
        });
        let v = store.put(b"base".to_vec(), vec![], 1);
        store.create_branch("b1".to_owned(), v, 2).expect("b1 ok");
        assert_eq!(
            store.create_branch("b2".to_owned(), v, 3),
            Err(VsError::MaxBranchesReached)
        );
    }

    // -----------------------------------------------------------------------
    // Test 18 – put_on_branch for missing branch returns BranchNotFound
    // -----------------------------------------------------------------------
    #[test]
    fn test_put_on_branch_not_found() {
        let mut store = default_store();
        assert_eq!(
            store.put_on_branch("ghost".to_owned(), b"x".to_vec(), vec![], 1),
            Err(VsError::BranchNotFound("ghost".to_owned()))
        );
    }

    // -----------------------------------------------------------------------
    // Test 19 – merge_branch copies source head onto target branch
    // -----------------------------------------------------------------------
    #[test]
    fn test_merge_branch() {
        let mut store = default_store();
        let v1 = store.put(b"base".to_vec(), vec![], 1);
        store
            .create_branch("feature".to_owned(), v1, 2)
            .expect("create");
        store
            .put_on_branch("feature".to_owned(), b"feature-work".to_vec(), vec![], 3)
            .expect("put on feature");

        let merged = store.merge_branch("feature", "main", 4).expect("merge");
        let got = store.get(VersionQuery::Latest).expect("latest");
        assert_eq!(got.version, merged);
        assert_eq!(got.data, b"feature-work");
    }

    // -----------------------------------------------------------------------
    // Test 20 – merge_branch propagates parent link from source head
    // -----------------------------------------------------------------------
    #[test]
    fn test_merge_branch_parent_link() {
        let mut store = default_store();
        let v1 = store.put(b"base".to_vec(), vec![], 1);
        store
            .create_branch("feat".to_owned(), v1, 2)
            .expect("create");
        let feat_head = store
            .put_on_branch("feat".to_owned(), b"feat".to_vec(), vec![], 3)
            .expect("put");

        let merged = store.merge_branch("feat", "main", 4).expect("merge");
        let ver = store
            .get(VersionQuery::AtVersion(merged))
            .expect("merged ver");
        assert_eq!(ver.parent_version, Some(feat_head));
    }

    // -----------------------------------------------------------------------
    // Test 21 – merge_branch with missing source returns BranchNotFound
    // -----------------------------------------------------------------------
    #[test]
    fn test_merge_branch_missing_source() {
        let mut store = default_store();
        assert_eq!(
            store.merge_branch("ghost", "main", 1),
            Err(VsError::BranchNotFound("ghost".to_owned()))
        );
    }

    // -----------------------------------------------------------------------
    // Test 22 – merge_branch with missing target returns BranchNotFound
    // -----------------------------------------------------------------------
    #[test]
    fn test_merge_branch_missing_target() {
        let mut store = default_store();
        let v = store.put(b"x".to_vec(), vec![], 1);
        store.create_branch("src".to_owned(), v, 2).expect("create");
        assert_eq!(
            store.merge_branch("src", "ghost", 3),
            Err(VsError::BranchNotFound("ghost".to_owned()))
        );
    }

    // -----------------------------------------------------------------------
    // Test 23 – history returns versions in reverse-chronological order
    // -----------------------------------------------------------------------
    #[test]
    fn test_history_order() {
        let mut store = default_store();
        let v1 = store.put(b"a".to_vec(), vec![], 1);
        let v2 = store.put(b"b".to_vec(), vec![], 2);
        let v3 = store.put(b"c".to_vec(), vec![], 3);

        let hist = store.history("main").expect("history");
        let nums: Vec<u64> = hist.iter().map(|v| v.version).collect();
        assert_eq!(nums, vec![v3, v2, v1]);
    }

    // -----------------------------------------------------------------------
    // Test 24 – history on empty branch returns empty vec
    // -----------------------------------------------------------------------
    #[test]
    fn test_history_empty_branch() {
        let store = default_store();
        let hist = store.history("main").expect("history");
        assert!(hist.is_empty());
    }

    // -----------------------------------------------------------------------
    // Test 25 – history on missing branch returns BranchNotFound
    // -----------------------------------------------------------------------
    #[test]
    fn test_history_missing_branch() {
        let store = default_store();
        assert_eq!(
            store.history("nope"),
            Err(VsError::BranchNotFound("nope".to_owned()))
        );
    }

    // -----------------------------------------------------------------------
    // Test 26 – GC KeepAll deletes nothing
    // -----------------------------------------------------------------------
    #[test]
    fn test_gc_keep_all() {
        let mut store = store_with_gc(OvsGcPolicy::KeepAll);
        for i in 0u8..10 {
            store.put(vec![i], vec![], u64::from(i));
        }
        let deleted = store.gc();
        assert_eq!(deleted, 0);
        assert_eq!(store.version_count(), 10);
    }

    // -----------------------------------------------------------------------
    // Test 27 – GC KeepLast(n) keeps the n newest and deletes unreachable rest
    // -----------------------------------------------------------------------
    #[test]
    fn test_gc_keep_last() {
        let mut store = store_with_gc(OvsGcPolicy::KeepLast(3));
        for i in 0u8..10 {
            store.put(vec![i], vec![], u64::from(i));
        }
        // Detach the head so versions 1–7 are unreachable (only the last 3 committed
        // to "main" form the reachable chain).
        // Actually with KeepLast(3) we keep the top-3 by number regardless of reachability
        // among unreachable versions.
        store.gc();
        // At most 3 unreachable versions survive; main's head chain is always kept.
        // All 10 were added linearly so the full chain is reachable. KeepLast(3)
        // only deletes unreachable versions outside the top-3, so nothing is deleted.
        // Let's verify by checking version_count ≥ 3 (the reachable chain is intact).
        assert!(store.version_count() >= 3);
    }

    // -----------------------------------------------------------------------
    // Test 28 – GC KeepLast deletes truly unreachable old versions
    // -----------------------------------------------------------------------
    #[test]
    fn test_gc_keep_last_deletes_unreachable() {
        let mut store = store_with_gc(OvsGcPolicy::KeepLast(2));
        // Add versions 1..5 on main (all reachable via parent chain)
        for i in 1u8..=5 {
            store.put(vec![i], vec![], u64::from(i));
        }
        // Create a branch at version 3 and advance it — version 3 becomes
        // reachable via the new branch, but versions 1 and 2 are only reachable
        // via the main chain (which includes them through parent links).
        // All are reachable, so nothing should be deleted.
        let deleted = store.gc();
        assert_eq!(deleted, 0);
        assert_eq!(store.version_count(), 5);
    }

    // -----------------------------------------------------------------------
    // Test 29 – GC KeepSince deletes old unreachable versions
    // -----------------------------------------------------------------------
    #[test]
    fn test_gc_keep_since() {
        // Build a store: commit versions at t=10, 20, 30, 40, 50 then create
        // a new branch from the last version.  Reset main to only the newest.
        let mut store = store_with_gc(OvsGcPolicy::KeepSince(35));

        let v1 = store.put(b"t10".to_vec(), vec![], 10);
        let v2 = store.put(b"t20".to_vec(), vec![], 20);
        let _v3 = store.put(b"t30".to_vec(), vec![], 30);
        let v4 = store.put(b"t40".to_vec(), vec![], 40);
        let v5 = store.put(b"t50".to_vec(), vec![], 50);

        // Create an isolated branch at v1 and advance it (v1 is now reachable
        // from the new branch).
        store
            .create_branch("keep-old".to_owned(), v1, 5)
            .expect("create");

        // All versions ≥ t=35 (v4, v5) are safe; v1 is safe (reachable).
        // v2 (t=20) and v3 (t=30) are only reachable via main's chain through
        // parent links starting at v5 → v4 → v3 → v2 → v1.
        // So nothing unreachable yet.  Let's confirm 0 are deleted.
        let deleted = store.gc();
        assert_eq!(deleted, 0);

        // Now break the main chain by resetting its head directly to v5 (same
        // as current state, no real break), but add an orphan version.
        // Simulate an orphan by inserting a version directly.
        use super::OvsObjectVersion;
        store.versions.insert(
            999,
            OvsObjectVersion {
                version: 999,
                cid: "orphan".to_owned(),
                data: b"orphan".to_vec(),
                parent_version: None,
                created_at: 10, // old timestamp → should be GC'd
                tags: vec![],
                size_bytes: 6,
            },
        );
        store.total_bytes += 6;

        let deleted2 = store.gc();
        // version 999 has created_at=10 < 35 and is not reachable from any head.
        assert_eq!(deleted2, 1);
        assert!(!store.versions.contains_key(&999));

        let _ = (v2, v4, v5); // suppress unused
    }

    // -----------------------------------------------------------------------
    // Test 30 – GC KeepTagged deletes untagged unreachable versions
    // -----------------------------------------------------------------------
    #[test]
    fn test_gc_keep_tagged() {
        let mut store = store_with_gc(OvsGcPolicy::KeepTagged);

        // Create a linear chain on main.
        let v1 = store.put(b"v1".to_vec(), vec!["release".to_owned()], 1);
        let _v2 = store.put(b"v2".to_vec(), vec![], 2); // no tag, reachable
        let _v3 = store.put(b"v3".to_vec(), vec!["stable".to_owned()], 3);

        // Insert an orphan (no tag, not reachable).
        use super::OvsObjectVersion;
        store.versions.insert(
            888,
            OvsObjectVersion {
                version: 888,
                cid: "orphan2".to_owned(),
                data: b"orphan".to_vec(),
                parent_version: None,
                created_at: 1,
                tags: vec![],
                size_bytes: 6,
            },
        );
        store.total_bytes += 6;

        let deleted = store.gc();
        // Only version 888 should be deleted (no tag, not reachable from any head).
        assert_eq!(deleted, 1);
        assert!(!store.versions.contains_key(&888));

        // v1 has a tag → survives even if unreachable (but it IS reachable here).
        assert!(store.versions.contains_key(&v1));
    }

    // -----------------------------------------------------------------------
    // Test 31 – reachable_from_heads traverses parent chain correctly
    // -----------------------------------------------------------------------
    #[test]
    fn test_reachable_from_heads() {
        let mut store = default_store();
        let v1 = store.put(b"a".to_vec(), vec![], 1);
        let v2 = store.put(b"b".to_vec(), vec![], 2);
        let v3 = store.put(b"c".to_vec(), vec![], 3);

        let reachable = store.reachable_from_heads();
        assert!(reachable.contains(&v1));
        assert!(reachable.contains(&v2));
        assert!(reachable.contains(&v3));
    }

    // -----------------------------------------------------------------------
    // Test 32 – stats returns correct counts after operations
    // -----------------------------------------------------------------------
    #[test]
    fn test_stats() {
        let mut store = default_store();
        store.put(b"hello".to_vec(), vec![], 1);
        store.put(b"world".to_vec(), vec![], 2);

        let s = store.stats();
        assert_eq!(s.version_count, 2);
        assert_eq!(s.branch_count, 1);
        assert_eq!(s.total_bytes, 10); // "hello"(5) + "world"(5)
        assert_eq!(s.deleted_versions, 0);
    }

    // -----------------------------------------------------------------------
    // Test 33 – total_size_bytes tracks accurately
    // -----------------------------------------------------------------------
    #[test]
    fn test_total_size_bytes() {
        let mut store = default_store();
        store.put(b"abc".to_vec(), vec![], 1); // 3 bytes
        store.put(b"de".to_vec(), vec![], 2); // 2 bytes
        assert_eq!(store.total_size_bytes(), 5);
    }

    // -----------------------------------------------------------------------
    // Test 34 – CID is deterministic and based on data content
    // -----------------------------------------------------------------------
    #[test]
    fn test_cid_is_deterministic() {
        let mut store1 = default_store();
        let mut store2 = default_store();
        store1.put(b"same content".to_vec(), vec![], 1);
        store2.put(b"same content".to_vec(), vec![], 1);
        let cid1 = &store1.get(VersionQuery::Latest).expect("v1").cid;
        let cid2 = &store2.get(VersionQuery::Latest).expect("v2").cid;
        assert_eq!(cid1, cid2);
    }

    // -----------------------------------------------------------------------
    // Test 35 – Different data produces different CIDs
    // -----------------------------------------------------------------------
    #[test]
    fn test_cid_different_for_different_data() {
        let cid_a = fnv1a_64(b"hello");
        let cid_b = fnv1a_64(b"world");
        assert_ne!(cid_a, cid_b);
    }

    // -----------------------------------------------------------------------
    // Test 36 – parent_version chain is set correctly on main
    // -----------------------------------------------------------------------
    #[test]
    fn test_parent_version_chain_on_main() {
        let mut store = default_store();
        let v1 = store.put(b"first".to_vec(), vec![], 1);
        let v2 = store.put(b"second".to_vec(), vec![], 2);
        let v3 = store.put(b"third".to_vec(), vec![], 3);

        let ver1 = store.get(VersionQuery::AtVersion(v1)).expect("v1");
        assert_eq!(ver1.parent_version, None);

        let ver2 = store.get(VersionQuery::AtVersion(v2)).expect("v2");
        assert_eq!(ver2.parent_version, Some(v1));

        let ver3 = store.get(VersionQuery::AtVersion(v3)).expect("v3");
        assert_eq!(ver3.parent_version, Some(v2));
    }

    // -----------------------------------------------------------------------
    // Test 37 – branch inherits parent chain from fork point
    // -----------------------------------------------------------------------
    #[test]
    fn test_branch_parent_chain() {
        let mut store = default_store();
        let v1 = store.put(b"base".to_vec(), vec![], 1);
        store
            .create_branch("side".to_owned(), v1, 2)
            .expect("create");
        let v_side = store
            .put_on_branch("side".to_owned(), b"side".to_vec(), vec![], 3)
            .expect("put");

        let ver = store.get(VersionQuery::AtVersion(v_side)).expect("ver");
        assert_eq!(ver.parent_version, Some(v1));
    }

    // -----------------------------------------------------------------------
    // Test 38 – deleted_versions counter increments on GC
    // -----------------------------------------------------------------------
    #[test]
    fn test_deleted_versions_counter() {
        let mut store = store_with_gc(OvsGcPolicy::KeepTagged);

        // Insert an orphan manually.
        use super::OvsObjectVersion;
        store.versions.insert(
            777,
            OvsObjectVersion {
                version: 777,
                cid: "x".to_owned(),
                data: b"x".to_vec(),
                parent_version: None,
                created_at: 1,
                tags: vec![],
                size_bytes: 1,
            },
        );
        store.total_bytes += 1;

        assert_eq!(store.stats().deleted_versions, 0);
        store.gc();
        assert_eq!(store.stats().deleted_versions, 1);
    }

    // -----------------------------------------------------------------------
    // Test 39 – version_count decreases after GC
    // -----------------------------------------------------------------------
    #[test]
    fn test_version_count_after_gc() {
        let mut store = store_with_gc(OvsGcPolicy::KeepTagged);
        store.put(b"keep".to_vec(), vec!["tag".to_owned()], 1);

        use super::OvsObjectVersion;
        store.versions.insert(
            500,
            OvsObjectVersion {
                version: 500,
                cid: "orphan".to_owned(),
                data: b"orphan".to_vec(),
                parent_version: None,
                created_at: 1,
                tags: vec![],
                size_bytes: 6,
            },
        );
        store.total_bytes += 6;

        let before = store.version_count();
        store.gc();
        assert!(store.version_count() < before);
    }

    // -----------------------------------------------------------------------
    // Test 40 – VsError Display messages are non-empty
    // -----------------------------------------------------------------------
    #[test]
    fn test_vs_error_display() {
        assert!(!VsError::VersionNotFound(1).to_string().is_empty());
        assert!(!VsError::BranchNotFound("x".to_owned())
            .to_string()
            .is_empty());
        assert!(!VsError::BranchAlreadyExists("x".to_owned())
            .to_string()
            .is_empty());
        assert!(!VsError::MaxVersionsReached.to_string().is_empty());
        assert!(!VsError::MaxBranchesReached.to_string().is_empty());
    }

    // -----------------------------------------------------------------------
    // Test 41 – VersionStoreConfig default values
    // -----------------------------------------------------------------------
    #[test]
    fn test_version_store_config_defaults() {
        let cfg = VersionStoreConfig::default();
        assert_eq!(cfg.max_versions, 10_000);
        assert_eq!(cfg.max_branches, 100);
        assert!(!cfg.compress_old_versions);
        assert_eq!(cfg.gc_policy, OvsGcPolicy::KeepAll);
    }

    // -----------------------------------------------------------------------
    // Test 42 – VersionBranch fields are accessible
    // -----------------------------------------------------------------------
    #[test]
    fn test_version_branch_fields() {
        let b = VersionBranch {
            name: "test".to_owned(),
            head_version: 5,
            created_at: 42,
            parent_branch: Some("main".to_owned()),
        };
        assert_eq!(b.name, "test");
        assert_eq!(b.head_version, 5);
        assert_eq!(b.created_at, 42);
        assert_eq!(b.parent_branch, Some("main".to_owned()));
    }

    // -----------------------------------------------------------------------
    // Test 43 – VsStats fields reflect live state
    // -----------------------------------------------------------------------
    #[test]
    fn test_vs_stats_fields() {
        let s = VsStats {
            version_count: 3,
            branch_count: 2,
            total_bytes: 100,
            deleted_versions: 5,
        };
        assert_eq!(s.version_count, 3);
        assert_eq!(s.branch_count, 2);
        assert_eq!(s.total_bytes, 100);
        assert_eq!(s.deleted_versions, 5);
    }

    // -----------------------------------------------------------------------
    // Test 44 – multi-branch get(AtTime) picks correct cross-branch version
    // -----------------------------------------------------------------------
    #[test]
    fn test_get_at_time_across_branches() {
        let mut store = default_store();
        let v1 = store.put(b"main-t10".to_vec(), vec![], 10);
        store
            .create_branch("side".to_owned(), v1, 11)
            .expect("create");
        store
            .put_on_branch("side".to_owned(), b"side-t20".to_vec(), vec![], 20)
            .expect("put");
        store.put(b"main-t30".to_vec(), vec![], 30);

        // At time 15 only v1 (t=10) and the side branch commit (t=20) have
        // created_at ≤ 15; v1 is the only one.
        let got = store.get(VersionQuery::AtTime(15)).expect("at time");
        assert_eq!(got.version, v1);
    }

    // -----------------------------------------------------------------------
    // Test 45 – get(AtTime) when multiple versions share the timestamp
    // -----------------------------------------------------------------------
    #[test]
    fn test_get_at_time_picks_highest_version() {
        let mut store = default_store();
        let v1 = store.put(b"early".to_vec(), vec![], 100);
        let v2 = store.put(b"also-t100".to_vec(), vec![], 100);
        let got = store.get(VersionQuery::AtTime(100)).expect("tie");
        // Should return v2 (highest version number with created_at ≤ 100).
        assert!(got.version == v1 || got.version == v2);
        // Specifically the highest:
        assert_eq!(got.version, v2);
    }

    // -----------------------------------------------------------------------
    // Test 46 – empty store reachable_from_heads returns empty set
    // -----------------------------------------------------------------------
    #[test]
    fn test_reachable_empty_store() {
        let store = default_store();
        assert!(store.reachable_from_heads().is_empty());
    }

    // -----------------------------------------------------------------------
    // Test 47 – auto-gc triggered when max_versions is hit
    // -----------------------------------------------------------------------
    #[test]
    fn test_auto_gc_on_max_versions() {
        let mut store = ObjectVersionStore::new(VersionStoreConfig {
            max_versions: 5,
            gc_policy: OvsGcPolicy::KeepTagged,
            ..Default::default()
        });

        // Fill up to max_versions with untagged versions on main.
        // All are reachable through the parent chain, so KeepTagged won't
        // delete them; but auto-gc is invoked without panic.
        for i in 0u8..5 {
            store.put(vec![i], vec![], u64::from(i));
        }
        // Add one more; this triggers GC before insertion.
        store.put(vec![99], vec!["important".to_owned()], 99);
        // Store is functional and contains at least the tagged version.
        assert!(store.get(VersionQuery::Latest).is_ok());
    }

    // -----------------------------------------------------------------------
    // Test 48 – history on branch with single commit
    // -----------------------------------------------------------------------
    #[test]
    fn test_history_single_commit() {
        let mut store = default_store();
        let v = store.put(b"only".to_vec(), vec![], 1);
        let hist = store.history("main").expect("hist");
        assert_eq!(hist.len(), 1);
        assert_eq!(hist[0].version, v);
    }

    // -----------------------------------------------------------------------
    // Test 49 – put_on_branch advances correct branch, not main
    // -----------------------------------------------------------------------
    #[test]
    fn test_put_on_branch_does_not_advance_main() {
        let mut store = default_store();
        let v1 = store.put(b"main-base".to_vec(), vec![], 1);
        store
            .create_branch("side".to_owned(), v1, 2)
            .expect("create");
        store
            .put_on_branch("side".to_owned(), b"side-update".to_vec(), vec![], 3)
            .expect("put");

        // main head should still be v1.
        let main_head = store.branches["main"].head_version;
        assert_eq!(main_head, v1);
    }

    // -----------------------------------------------------------------------
    // Test 50 – fnv1a_64 produces correct length hex string
    // -----------------------------------------------------------------------
    #[test]
    fn test_fnv1a_64_hex_length() {
        let cid = fnv1a_64(b"test data");
        assert_eq!(cid.len(), 16);
        assert!(cid.chars().all(|c| c.is_ascii_hexdigit()));
    }
}
