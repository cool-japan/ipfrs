//! Bidirectional sync between storage mirrors with conflict detection, resolution, and audit.
//!
//! # Overview
//!
//! `StorageMirrorSync` manages synchronization between a local mirror and one or more remote
//! mirrors. It computes diff plans, resolves conflicts according to configurable policies,
//! applies operations, and maintains a capped audit log for observability.
//!
//! # Example
//!
//! ```rust
//! use ipfrs_storage::mirror_sync::{
//!     StorageMirrorSync, MirrorId, SyncItem, MsConflictResolution,
//! };
//!
//! let local_id = MirrorId("local".to_string());
//! let remote_id = MirrorId("remote-1".to_string());
//! let mut sync = StorageMirrorSync::new(local_id, MsConflictResolution::TakeNewest);
//! sync.register_mirror(remote_id.clone());
//!
//! let item = SyncItem::new("Qm1234".to_string(), 100, 0, MirrorId("local".to_string()));
//! sync.update_local(item);
//! let plan = sync.diff(&remote_id);
//! assert_eq!(plan.operations.len(), 1);
//! ```

use std::collections::{HashMap, VecDeque};

// ── Audit-log capacity ────────────────────────────────────────────────────────
const AUDIT_LOG_CAPACITY: usize = 1000;

// ── FNV-1a 64-bit hash ────────────────────────────────────────────────────────

/// Computes the FNV-1a 64-bit checksum of a byte slice.
#[inline]
pub fn fnv1a_64(data: &[u8]) -> u64 {
    const OFFSET: u64 = 14_695_981_039_346_656_037;
    const PRIME: u64 = 1_099_511_628_211;
    data.iter()
        .fold(OFFSET, |acc, &b| (acc ^ u64::from(b)).wrapping_mul(PRIME))
}

// ── MirrorId ──────────────────────────────────────────────────────────────────

/// Newtype wrapper for mirror identifiers.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct MirrorId(pub String);

impl MirrorId {
    /// Creates a new `MirrorId` from a string.
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Returns the inner string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for MirrorId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

// ── SyncItem ──────────────────────────────────────────────────────────────────

/// A single item tracked for synchronisation across mirrors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyncItem {
    /// Content Identifier (CID) of the block.
    pub cid: String,
    /// FNV-1a-64 checksum of the CID bytes.
    pub checksum: u64,
    /// Block size in bytes.
    pub size_bytes: u64,
    /// Monotonically increasing version number.
    pub version: u64,
    /// UNIX timestamp (ms) of the last modification.
    pub last_modified: u64,
    /// The mirror that owns / sourced this item.
    pub mirror_id: MirrorId,
}

impl SyncItem {
    /// Creates a `SyncItem`, computing the FNV-1a-64 checksum from the CID bytes.
    pub fn new(cid: String, size_bytes: u64, last_modified: u64, mirror_id: MirrorId) -> Self {
        let checksum = fnv1a_64(cid.as_bytes());
        Self {
            cid,
            checksum,
            size_bytes,
            version: 1,
            last_modified,
            mirror_id,
        }
    }

    /// Creates a `SyncItem` with all fields specified explicitly.
    pub fn with_version(
        cid: String,
        size_bytes: u64,
        version: u64,
        last_modified: u64,
        mirror_id: MirrorId,
    ) -> Self {
        let checksum = fnv1a_64(cid.as_bytes());
        Self {
            cid,
            checksum,
            size_bytes,
            version,
            last_modified,
            mirror_id,
        }
    }

    /// Creates a `SyncItem` with an explicitly provided checksum (for testing / deserialization).
    pub fn with_checksum(
        cid: String,
        checksum: u64,
        size_bytes: u64,
        version: u64,
        last_modified: u64,
        mirror_id: MirrorId,
    ) -> Self {
        Self {
            cid,
            checksum,
            size_bytes,
            version,
            last_modified,
            mirror_id,
        }
    }
}

// ── ConflictType ──────────────────────────────────────────────────────────────

/// Classifies the nature of a synchronisation conflict.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConflictType {
    /// Local and remote have different version numbers.
    VersionConflict,
    /// Local and remote have different checksums.
    ChecksumMismatch,
    /// Local and remote have different sizes.
    SizeConflict,
    /// One side deleted the item while the other modified it.
    DeleteVsModify,
}

// ── MsConflictResolution ──────────────────────────────────────────────────────

/// Strategy for automatically resolving a sync conflict.
///
/// Prefixed with `Ms` to avoid clashing with the crate-level re-export of
/// `eventual_consistency::ConflictResolution`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MsConflictResolution {
    /// Keep the local copy; upload to remote.
    TakeLocal,
    /// Keep the remote copy; download to local.
    TakeRemote,
    /// Keep whichever has the newer `last_modified` timestamp.
    TakeNewest,
    /// Keep whichever has the larger `size_bytes`.
    TakeLargest,
    /// Do nothing — leave the conflict unresolved in the plan.
    Skip,
}

// ── SyncConflict ─────────────────────────────────────────────────────────────

/// Describes a conflict between local and remote versions of the same CID.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyncConflict {
    /// The conflicting CID.
    pub cid: String,
    /// Local copy of the item.
    pub local_item: SyncItem,
    /// Remote copy of the item.
    pub remote_item: SyncItem,
    /// Classification of the conflict.
    pub conflict_type: ConflictType,
}

impl SyncConflict {
    /// Classifies a conflict between two `SyncItem`s that share the same CID.
    pub fn classify(local_item: SyncItem, remote_item: SyncItem) -> Self {
        let conflict_type = if local_item.checksum != remote_item.checksum {
            ConflictType::ChecksumMismatch
        } else if local_item.version != remote_item.version {
            ConflictType::VersionConflict
        } else if local_item.size_bytes != remote_item.size_bytes {
            ConflictType::SizeConflict
        } else {
            ConflictType::ChecksumMismatch
        };
        let cid = local_item.cid.clone();
        Self {
            cid,
            local_item,
            remote_item,
            conflict_type,
        }
    }
}

// ── SyncOperation ─────────────────────────────────────────────────────────────

/// A single operation within a `SyncPlan`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SyncOperation {
    /// Send a block from local to a remote mirror.
    Upload {
        /// CID of the block to send.
        cid: String,
        /// Destination mirror.
        to_mirror: MirrorId,
    },
    /// Fetch a block from a remote mirror to local.
    Download {
        /// CID of the block to fetch.
        cid: String,
        /// Source mirror.
        from_mirror: MirrorId,
    },
    /// Remove a block from a mirror.
    Delete {
        /// CID of the block to remove.
        cid: String,
        /// Mirror to remove it from.
        from_mirror: MirrorId,
    },
    /// Resolve a conflict according to a specified strategy.
    Resolve {
        /// The detected conflict.
        conflict: SyncConflict,
        /// The chosen resolution strategy.
        resolution: MsConflictResolution,
    },
}

// ── SyncPlan ─────────────────────────────────────────────────────────────────

/// The result of a `diff` computation — a list of operations and unresolved conflicts.
#[derive(Debug, Clone)]
pub struct SyncPlan {
    /// Ordered list of operations to execute.
    pub operations: Vec<SyncOperation>,
    /// Conflicts that were left unresolved (e.g. `Skip` strategy).
    pub conflicts: Vec<SyncConflict>,
    /// Total bytes that would be transferred if the plan is applied.
    pub estimated_bytes: u64,
}

impl SyncPlan {
    /// Returns `true` when there are no operations and no unresolved conflicts.
    pub fn is_empty(&self) -> bool {
        self.operations.is_empty() && self.conflicts.is_empty()
    }
}

// ── MsSyncResult ─────────────────────────────────────────────────────────────

/// Summary of what happened after applying a `SyncPlan`.
///
/// Prefixed with `Ms` to avoid clashing with the crate-level re-export of
/// `replication::SyncResult`.
#[derive(Debug, Clone, Default)]
pub struct MsSyncResult {
    /// Number of operations that were executed.
    pub operations_executed: usize,
    /// Total bytes transferred.
    pub bytes_transferred: u64,
    /// Number of conflicts that were resolved.
    pub conflicts_resolved: usize,
    /// Non-fatal errors encountered during execution.
    pub errors: Vec<String>,
    /// Wall-clock duration of the apply call (in milliseconds).
    pub duration_ms: u64,
}

// ── MirrorSyncStats ───────────────────────────────────────────────────────────

/// Aggregate statistics tracked across all `apply_plan` calls.
#[derive(Debug, Clone, Default)]
pub struct MirrorSyncStats {
    /// Number of items in the local state.
    pub local_items: usize,
    /// Number of registered remote mirrors.
    pub remote_mirrors: usize,
    /// Total number of conflicts detected across all diffs.
    pub total_conflicts_detected: u64,
    /// Total number of operations executed across all apply calls.
    pub total_operations: u64,
    /// Total bytes transferred across all apply calls.
    pub bytes_transferred: u64,
}

// ── StorageMirrorSync ─────────────────────────────────────────────────────────

/// Bidirectional synchronisation between storage mirrors.
///
/// Maintains the local state, one or more remote mirror states, resolves
/// conflicts, executes sync plans, and retains a bounded audit log.
pub struct StorageMirrorSync {
    /// The identifier for the local mirror.
    pub local_id: MirrorId,
    /// Current state of the local mirror (CID → SyncItem).
    pub local_state: HashMap<String, SyncItem>,
    /// States of all registered remote mirrors.
    pub remote_states: HashMap<MirrorId, HashMap<String, SyncItem>>,
    /// Default strategy for automatically resolving conflicts.
    pub default_resolution: MsConflictResolution,
    /// Bounded audit log of executed operations (newest at the back).
    pub audit_log: VecDeque<SyncOperation>,

    // ── aggregate counters ──
    total_conflicts_detected: u64,
    total_operations: u64,
    total_bytes_transferred: u64,
}

impl StorageMirrorSync {
    // ── Constructors ─────────────────────────────────────────────────────────

    /// Creates a new `StorageMirrorSync` with the given local identifier and default conflict
    /// resolution strategy.
    pub fn new(local_id: MirrorId, default_resolution: MsConflictResolution) -> Self {
        Self {
            local_id,
            local_state: HashMap::new(),
            remote_states: HashMap::new(),
            default_resolution,
            audit_log: VecDeque::new(),
            total_conflicts_detected: 0,
            total_operations: 0,
            total_bytes_transferred: 0,
        }
    }

    // ── Mirror registration ───────────────────────────────────────────────────

    /// Registers a remote mirror.  If the mirror is already registered this is a no-op.
    pub fn register_mirror(&mut self, mirror_id: MirrorId) {
        self.remote_states.entry(mirror_id).or_default();
    }

    // ── Local state mutations ─────────────────────────────────────────────────

    /// Inserts or updates an item in the local state.
    ///
    /// Returns `true` if the CID was not previously present (new insertion).
    pub fn update_local(&mut self, item: SyncItem) -> bool {
        let is_new = !self.local_state.contains_key(&item.cid);
        self.local_state.insert(item.cid.clone(), item);
        is_new
    }

    /// Removes an item from the local state by CID.
    ///
    /// Returns `true` if the item existed and was removed.
    pub fn remove_local(&mut self, cid: &str) -> bool {
        self.local_state.remove(cid).is_some()
    }

    // ── Remote state mutations ────────────────────────────────────────────────

    /// Inserts or updates an item in a remote mirror's state.
    ///
    /// Returns `true` if the CID was not previously present in that mirror.
    /// Returns `false` if the mirror is not registered.
    pub fn update_remote(&mut self, mirror_id: &MirrorId, item: SyncItem) -> bool {
        match self.remote_states.get_mut(mirror_id) {
            Some(state) => {
                let is_new = !state.contains_key(&item.cid);
                state.insert(item.cid.clone(), item);
                is_new
            }
            None => false,
        }
    }

    // ── Diff ──────────────────────────────────────────────────────────────────

    /// Computes the set of operations needed to synchronise the local mirror with `mirror_id`.
    ///
    /// Conflict resolution follows `self.default_resolution`.
    pub fn diff(&self, mirror_id: &MirrorId) -> SyncPlan {
        let remote_state = match self.remote_states.get(mirror_id) {
            Some(s) => s,
            None => {
                return SyncPlan {
                    operations: Vec::new(),
                    conflicts: Vec::new(),
                    estimated_bytes: 0,
                }
            }
        };

        let mut operations: Vec<SyncOperation> = Vec::new();
        let mut conflicts: Vec<SyncConflict> = Vec::new();
        let mut estimated_bytes: u64 = 0;

        // Items in local but not remote → Upload
        for (cid, local_item) in &self.local_state {
            if !remote_state.contains_key(cid) {
                estimated_bytes = estimated_bytes.saturating_add(local_item.size_bytes);
                operations.push(SyncOperation::Upload {
                    cid: cid.clone(),
                    to_mirror: mirror_id.clone(),
                });
            }
        }

        // Items in remote but not local → Download
        // Items in both with different checksums → conflict
        for (cid, remote_item) in remote_state {
            match self.local_state.get(cid) {
                None => {
                    estimated_bytes = estimated_bytes.saturating_add(remote_item.size_bytes);
                    operations.push(SyncOperation::Download {
                        cid: cid.clone(),
                        from_mirror: mirror_id.clone(),
                    });
                }
                Some(local_item) => {
                    if local_item.checksum != remote_item.checksum {
                        let conflict =
                            SyncConflict::classify(local_item.clone(), remote_item.clone());
                        // (conflict counter is updated in apply_plan, not here)
                        let resolution = self.resolve_conflict(&conflict);
                        match &resolution {
                            MsConflictResolution::TakeLocal => {
                                estimated_bytes =
                                    estimated_bytes.saturating_add(local_item.size_bytes);
                                operations.push(SyncOperation::Resolve {
                                    conflict: conflict.clone(),
                                    resolution: MsConflictResolution::TakeLocal,
                                });
                            }
                            MsConflictResolution::TakeRemote => {
                                estimated_bytes =
                                    estimated_bytes.saturating_add(remote_item.size_bytes);
                                operations.push(SyncOperation::Resolve {
                                    conflict: conflict.clone(),
                                    resolution: MsConflictResolution::TakeRemote,
                                });
                            }
                            MsConflictResolution::TakeNewest => {
                                let winner_bytes =
                                    if local_item.last_modified >= remote_item.last_modified {
                                        local_item.size_bytes
                                    } else {
                                        remote_item.size_bytes
                                    };
                                estimated_bytes = estimated_bytes.saturating_add(winner_bytes);
                                operations.push(SyncOperation::Resolve {
                                    conflict: conflict.clone(),
                                    resolution: MsConflictResolution::TakeNewest,
                                });
                            }
                            MsConflictResolution::TakeLargest => {
                                let winner_bytes =
                                    local_item.size_bytes.max(remote_item.size_bytes);
                                estimated_bytes = estimated_bytes.saturating_add(winner_bytes);
                                operations.push(SyncOperation::Resolve {
                                    conflict: conflict.clone(),
                                    resolution: MsConflictResolution::TakeLargest,
                                });
                            }
                            MsConflictResolution::Skip => {
                                conflicts.push(conflict);
                            }
                        }
                    }
                }
            }
        }

        SyncPlan {
            operations,
            conflicts,
            estimated_bytes,
        }
    }

    /// Chooses a `MsConflictResolution` for a conflict, honouring `default_resolution`.
    fn resolve_conflict(&self, _conflict: &SyncConflict) -> MsConflictResolution {
        self.default_resolution.clone()
    }

    // ── Apply plan ────────────────────────────────────────────────────────────

    /// Simulates executing all operations in `plan`, updating local/remote states and audit log.
    ///
    /// `now` is the UNIX timestamp (ms) to stamp on newly created items.
    pub fn apply_plan(&mut self, plan: &SyncPlan, now: u64) -> MsSyncResult {
        let start = now;
        let mut result = MsSyncResult::default();

        for op in &plan.operations {
            match op {
                SyncOperation::Upload { cid, to_mirror } => {
                    if let Some(local_item) = self.local_state.get(cid).cloned() {
                        let size = local_item.size_bytes;
                        let mut remote_item = local_item;
                        remote_item.mirror_id = to_mirror.clone();
                        if let Some(remote_state) = self.remote_states.get_mut(to_mirror) {
                            remote_state.insert(cid.clone(), remote_item);
                        } else {
                            result.errors.push(format!(
                                "Upload failed: mirror '{}' not registered",
                                to_mirror
                            ));
                            continue;
                        }
                        result.bytes_transferred = result.bytes_transferred.saturating_add(size);
                    } else {
                        result
                            .errors
                            .push(format!("Upload failed: CID '{}' not in local state", cid));
                        continue;
                    }
                }

                SyncOperation::Download { cid, from_mirror } => {
                    // Retrieve from remote and store locally
                    let remote_item_opt = self
                        .remote_states
                        .get(from_mirror)
                        .and_then(|s| s.get(cid))
                        .cloned();

                    if let Some(remote_item) = remote_item_opt {
                        let size = remote_item.size_bytes;
                        let mut local_item = remote_item;
                        local_item.mirror_id = self.local_id.clone();
                        self.local_state.insert(cid.clone(), local_item);
                        result.bytes_transferred = result.bytes_transferred.saturating_add(size);
                    } else {
                        result.errors.push(format!(
                            "Download failed: CID '{}' not in remote mirror '{}'",
                            cid, from_mirror
                        ));
                        continue;
                    }
                }

                SyncOperation::Delete { cid, from_mirror } => {
                    let removed = if *from_mirror == self.local_id {
                        self.local_state.remove(cid).is_some()
                    } else if let Some(remote_state) = self.remote_states.get_mut(from_mirror) {
                        remote_state.remove(cid).is_some()
                    } else {
                        false
                    };
                    if !removed {
                        result.errors.push(format!(
                            "Delete failed: CID '{}' not found in mirror '{}'",
                            cid, from_mirror
                        ));
                        continue;
                    }
                }

                SyncOperation::Resolve {
                    conflict,
                    resolution,
                } => {
                    match resolution {
                        MsConflictResolution::TakeLocal => {
                            // Upload local to remote (all remotes that have this CID)
                            let local_opt = self.local_state.get(&conflict.cid).cloned();
                            if let Some(local_item) = local_opt {
                                let size = local_item.size_bytes;
                                let remote_mirror = conflict.remote_item.mirror_id.clone();
                                if let Some(remote_state) =
                                    self.remote_states.get_mut(&remote_mirror)
                                {
                                    let mut item = local_item;
                                    item.mirror_id = remote_mirror.clone();
                                    remote_state.insert(conflict.cid.clone(), item);
                                }
                                result.bytes_transferred =
                                    result.bytes_transferred.saturating_add(size);
                            }
                        }
                        MsConflictResolution::TakeRemote => {
                            // Download remote to local
                            let remote_item_opt = self
                                .remote_states
                                .get(&conflict.remote_item.mirror_id)
                                .and_then(|s| s.get(&conflict.cid))
                                .cloned();
                            if let Some(remote_item) = remote_item_opt {
                                let size = remote_item.size_bytes;
                                let mut item = remote_item;
                                item.mirror_id = self.local_id.clone();
                                self.local_state.insert(conflict.cid.clone(), item);
                                result.bytes_transferred =
                                    result.bytes_transferred.saturating_add(size);
                            }
                        }
                        MsConflictResolution::TakeNewest => {
                            let local_item = self.local_state.get(&conflict.cid).cloned();
                            let remote_item = self
                                .remote_states
                                .get(&conflict.remote_item.mirror_id)
                                .and_then(|s| s.get(&conflict.cid))
                                .cloned();

                            match (local_item, remote_item) {
                                (Some(l), Some(r)) => {
                                    if l.last_modified >= r.last_modified {
                                        // local wins → push to remote
                                        let size = l.size_bytes;
                                        let remote_mirror = r.mirror_id.clone();
                                        if let Some(rs) = self.remote_states.get_mut(&remote_mirror)
                                        {
                                            let mut item = l;
                                            item.mirror_id = remote_mirror.clone();
                                            rs.insert(conflict.cid.clone(), item);
                                        }
                                        result.bytes_transferred =
                                            result.bytes_transferred.saturating_add(size);
                                    } else {
                                        // remote wins → pull to local
                                        let size = r.size_bytes;
                                        let mut item = r;
                                        item.mirror_id = self.local_id.clone();
                                        self.local_state.insert(conflict.cid.clone(), item);
                                        result.bytes_transferred =
                                            result.bytes_transferred.saturating_add(size);
                                    }
                                }
                                _ => {
                                    result.errors.push(format!(
                                        "TakeNewest: item '{}' missing from one side",
                                        conflict.cid
                                    ));
                                    continue;
                                }
                            }
                        }
                        MsConflictResolution::TakeLargest => {
                            let local_item = self.local_state.get(&conflict.cid).cloned();
                            let remote_item = self
                                .remote_states
                                .get(&conflict.remote_item.mirror_id)
                                .and_then(|s| s.get(&conflict.cid))
                                .cloned();

                            match (local_item, remote_item) {
                                (Some(l), Some(r)) => {
                                    if l.size_bytes >= r.size_bytes {
                                        // local wins → push to remote
                                        let size = l.size_bytes;
                                        let remote_mirror = r.mirror_id.clone();
                                        if let Some(rs) = self.remote_states.get_mut(&remote_mirror)
                                        {
                                            let mut item = l;
                                            item.mirror_id = remote_mirror.clone();
                                            rs.insert(conflict.cid.clone(), item);
                                        }
                                        result.bytes_transferred =
                                            result.bytes_transferred.saturating_add(size);
                                    } else {
                                        // remote wins → pull to local
                                        let size = r.size_bytes;
                                        let mut item = r;
                                        item.mirror_id = self.local_id.clone();
                                        self.local_state.insert(conflict.cid.clone(), item);
                                        result.bytes_transferred =
                                            result.bytes_transferred.saturating_add(size);
                                    }
                                }
                                _ => {
                                    result.errors.push(format!(
                                        "TakeLargest: item '{}' missing from one side",
                                        conflict.cid
                                    ));
                                    continue;
                                }
                            }
                        }
                        MsConflictResolution::Skip => {
                            // No state change; still record in audit
                        }
                    }

                    result.conflicts_resolved += 1;
                }
            }

            // Append to audit log, capping at AUDIT_LOG_CAPACITY
            self.audit_log.push_back(op.clone());
            if self.audit_log.len() > AUDIT_LOG_CAPACITY {
                self.audit_log.pop_front();
            }
            result.operations_executed += 1;
        }

        // Accumulate global counters
        self.total_conflicts_detected = self
            .total_conflicts_detected
            .saturating_add(plan.conflicts.len() as u64);
        self.total_operations = self
            .total_operations
            .saturating_add(result.operations_executed as u64);
        self.total_bytes_transferred = self
            .total_bytes_transferred
            .saturating_add(result.bytes_transferred);

        result.duration_ms = now.saturating_sub(start); // Simulated; single-threaded ≡ 0
        result
    }

    // ── Conflict detection ────────────────────────────────────────────────────

    /// Returns all CIDs present in both local and `mirror_id` that have differing checksums.
    pub fn detect_conflicts(&self, mirror_id: &MirrorId) -> Vec<SyncConflict> {
        let remote_state = match self.remote_states.get(mirror_id) {
            Some(s) => s,
            None => return Vec::new(),
        };

        let mut result: Vec<SyncConflict> = Vec::new();
        for (cid, remote_item) in remote_state {
            if let Some(local_item) = self.local_state.get(cid) {
                if local_item.checksum != remote_item.checksum {
                    result.push(SyncConflict::classify(
                        local_item.clone(),
                        remote_item.clone(),
                    ));
                }
            }
        }
        result.sort_by(|a, b| a.cid.cmp(&b.cid));
        result
    }

    // ── CID queries ───────────────────────────────────────────────────────────

    /// Returns all CIDs present only in the local state (not in `mirror_id`), sorted.
    pub fn local_only_cids(&self, mirror_id: &MirrorId) -> Vec<&str> {
        let remote_state = match self.remote_states.get(mirror_id) {
            Some(s) => s,
            None => {
                let mut all: Vec<&str> = self.local_state.keys().map(|s| s.as_str()).collect();
                all.sort_unstable();
                return all;
            }
        };

        let mut result: Vec<&str> = self
            .local_state
            .keys()
            .filter(|cid| !remote_state.contains_key(*cid))
            .map(|s| s.as_str())
            .collect();
        result.sort_unstable();
        result
    }

    /// Returns all CIDs present only in `mirror_id` (not in local state), sorted.
    pub fn remote_only_cids(&self, mirror_id: &MirrorId) -> Vec<&str> {
        let remote_state = match self.remote_states.get(mirror_id) {
            Some(s) => s,
            None => return Vec::new(),
        };

        let mut result: Vec<&str> = remote_state
            .keys()
            .filter(|cid| !self.local_state.contains_key(*cid))
            .map(|s| s.as_str())
            .collect();
        result.sort_unstable();
        result
    }

    // ── Sync-state checks ─────────────────────────────────────────────────────

    /// Returns `true` when every CID in both local and `mirror_id` has the same checksum.
    ///
    /// Returns `false` for unregistered mirrors.
    pub fn in_sync_with(&self, mirror_id: &MirrorId) -> bool {
        let remote_state = match self.remote_states.get(mirror_id) {
            Some(s) => s,
            None => return false,
        };

        if self.local_state.len() != remote_state.len() {
            return false;
        }

        for (cid, local_item) in &self.local_state {
            match remote_state.get(cid) {
                Some(remote_item) if remote_item.checksum == local_item.checksum => {}
                _ => return false,
            }
        }
        true
    }

    // ── Audit log ─────────────────────────────────────────────────────────────

    /// Returns up to the `n` most-recently logged operations (newest last).
    pub fn audit_log_tail(&self, n: usize) -> Vec<&SyncOperation> {
        let len = self.audit_log.len();
        let skip = len.saturating_sub(n);
        self.audit_log.iter().skip(skip).collect()
    }

    // ── Stats ─────────────────────────────────────────────────────────────────

    /// Returns aggregate statistics for this instance.
    pub fn stats(&self) -> MirrorSyncStats {
        MirrorSyncStats {
            local_items: self.local_state.len(),
            remote_mirrors: self.remote_states.len(),
            total_conflicts_detected: self.total_conflicts_detected,
            total_operations: self.total_operations,
            bytes_transferred: self.total_bytes_transferred,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use crate::mirror_sync::{
        fnv1a_64, ConflictType, MirrorId, MirrorSyncStats, MsConflictResolution, MsSyncResult,
        StorageMirrorSync, SyncConflict, SyncItem, SyncOperation, SyncPlan,
    };

    // ── helpers ───────────────────────────────────────────────────────────────

    fn local_id() -> MirrorId {
        MirrorId::new("local")
    }

    fn remote_id() -> MirrorId {
        MirrorId::new("remote-1")
    }

    fn make_item(cid: &str, size: u64, ts: u64, mirror: MirrorId) -> SyncItem {
        SyncItem::new(cid.to_string(), size, ts, mirror)
    }

    fn make_item_v(cid: &str, size: u64, version: u64, ts: u64, mirror: MirrorId) -> SyncItem {
        SyncItem::with_version(cid.to_string(), size, version, ts, mirror)
    }

    fn default_sync() -> StorageMirrorSync {
        let mut s = StorageMirrorSync::new(local_id(), MsConflictResolution::TakeNewest);
        s.register_mirror(remote_id());
        s
    }

    // ── 1. fnv1a_64 ───────────────────────────────────────────────────────────

    #[test]
    fn test_fnv1a_empty() {
        let h = fnv1a_64(b"");
        assert_eq!(h, 14_695_981_039_346_656_037);
    }

    #[test]
    fn test_fnv1a_known_value() {
        let h = fnv1a_64(b"hello");
        // FNV-1a 64 of "hello" = 0xa430d84680aabd0b
        assert_eq!(h, 0xa430d84680aabd0b);
    }

    #[test]
    fn test_fnv1a_differs_by_input() {
        assert_ne!(fnv1a_64(b"abc"), fnv1a_64(b"xyz"));
    }

    // ── 2. MirrorId ───────────────────────────────────────────────────────────

    #[test]
    fn test_mirror_id_new_and_as_str() {
        let id = MirrorId::new("mirror-a");
        assert_eq!(id.as_str(), "mirror-a");
    }

    #[test]
    fn test_mirror_id_display() {
        let id = MirrorId::new("m1");
        assert_eq!(format!("{}", id), "m1");
    }

    #[test]
    fn test_mirror_id_equality() {
        assert_eq!(MirrorId::new("x"), MirrorId::new("x"));
        assert_ne!(MirrorId::new("x"), MirrorId::new("y"));
    }

    // ── 3. SyncItem ───────────────────────────────────────────────────────────

    #[test]
    fn test_sync_item_checksum_matches_fnv1a() {
        let item = make_item("QmTest", 128, 0, local_id());
        assert_eq!(item.checksum, fnv1a_64(b"QmTest"));
    }

    #[test]
    fn test_sync_item_with_version() {
        let item = make_item_v("Qm1", 64, 5, 100, local_id());
        assert_eq!(item.version, 5);
        assert_eq!(item.last_modified, 100);
    }

    #[test]
    fn test_sync_item_with_checksum() {
        let item = SyncItem::with_checksum("Qm2".to_string(), 999, 32, 2, 200, local_id());
        assert_eq!(item.checksum, 999);
    }

    // ── 4. register_mirror ────────────────────────────────────────────────────

    #[test]
    fn test_register_mirror_creates_empty_state() {
        let mut s = StorageMirrorSync::new(local_id(), MsConflictResolution::TakeLocal);
        let rid = MirrorId::new("remote");
        s.register_mirror(rid.clone());
        assert!(s.remote_states.contains_key(&rid));
        assert!(s.remote_states[&rid].is_empty());
    }

    #[test]
    fn test_register_mirror_idempotent() {
        let mut s = default_sync();
        s.update_remote(&remote_id(), make_item("Qm1", 10, 0, remote_id()));
        s.register_mirror(remote_id()); // should not clear existing state
        assert!(s.remote_states[&remote_id()].contains_key("Qm1"));
    }

    // ── 5. update_local / remove_local ────────────────────────────────────────

    #[test]
    fn test_update_local_returns_true_for_new() {
        let mut s = default_sync();
        let item = make_item("Qm1", 10, 0, local_id());
        assert!(s.update_local(item));
    }

    #[test]
    fn test_update_local_returns_false_for_update() {
        let mut s = default_sync();
        s.update_local(make_item("Qm1", 10, 0, local_id()));
        assert!(!s.update_local(make_item("Qm1", 20, 1, local_id())));
    }

    #[test]
    fn test_remove_local_returns_true_when_present() {
        let mut s = default_sync();
        s.update_local(make_item("Qm1", 10, 0, local_id()));
        assert!(s.remove_local("Qm1"));
        assert!(!s.local_state.contains_key("Qm1"));
    }

    #[test]
    fn test_remove_local_returns_false_when_absent() {
        let mut s = default_sync();
        assert!(!s.remove_local("nonexistent"));
    }

    // ── 6. update_remote ─────────────────────────────────────────────────────

    #[test]
    fn test_update_remote_returns_false_for_unknown_mirror() {
        let mut s = default_sync();
        let unknown = MirrorId::new("ghost");
        assert!(!s.update_remote(&unknown, make_item("Qm1", 10, 0, unknown.clone())));
    }

    #[test]
    fn test_update_remote_inserts_item() {
        let mut s = default_sync();
        s.update_remote(&remote_id(), make_item("Qm1", 10, 0, remote_id()));
        assert!(s.remote_states[&remote_id()].contains_key("Qm1"));
    }

    // ── 7. diff – upload ──────────────────────────────────────────────────────

    #[test]
    fn test_diff_local_only_generates_upload() {
        let mut s = default_sync();
        s.update_local(make_item("QmA", 50, 0, local_id()));
        let plan = s.diff(&remote_id());
        assert_eq!(plan.operations.len(), 1);
        assert!(matches!(
            &plan.operations[0],
            SyncOperation::Upload { cid, .. } if cid == "QmA"
        ));
        assert_eq!(plan.estimated_bytes, 50);
    }

    // ── 8. diff – download ────────────────────────────────────────────────────

    #[test]
    fn test_diff_remote_only_generates_download() {
        let mut s = default_sync();
        s.update_remote(&remote_id(), make_item("QmB", 80, 0, remote_id()));
        let plan = s.diff(&remote_id());
        assert_eq!(plan.operations.len(), 1);
        assert!(matches!(
            &plan.operations[0],
            SyncOperation::Download { cid, .. } if cid == "QmB"
        ));
        assert_eq!(plan.estimated_bytes, 80);
    }

    // ── 9. diff – in-sync produces no ops ────────────────────────────────────

    #[test]
    fn test_diff_in_sync_produces_no_ops() {
        let mut s = default_sync();
        s.update_local(make_item("QmC", 10, 1, local_id()));
        s.update_remote(&remote_id(), make_item("QmC", 10, 1, remote_id()));
        let plan = s.diff(&remote_id());
        assert!(plan.is_empty());
    }

    // ── 10. diff – conflict TakeNewest (local newer) ──────────────────────────

    #[test]
    fn test_diff_conflict_take_newest_local_wins() {
        let mut s = StorageMirrorSync::new(local_id(), MsConflictResolution::TakeNewest);
        s.register_mirror(remote_id());

        let local = SyncItem::with_checksum("Qm1".to_string(), 11, 100, 1, 200, local_id());
        let remote = SyncItem::with_checksum("Qm1".to_string(), 22, 80, 1, 100, remote_id());
        s.update_local(local);
        s.update_remote(&remote_id(), remote);

        let plan = s.diff(&remote_id());
        assert_eq!(plan.operations.len(), 1);
        assert!(matches!(
            &plan.operations[0],
            SyncOperation::Resolve { resolution, .. }
            if *resolution == MsConflictResolution::TakeNewest
        ));
    }

    // ── 11. diff – conflict Skip produces no op ───────────────────────────────

    #[test]
    fn test_diff_conflict_skip() {
        let mut s = StorageMirrorSync::new(local_id(), MsConflictResolution::Skip);
        s.register_mirror(remote_id());

        let local = SyncItem::with_checksum("Qm2".to_string(), 11, 100, 1, 0, local_id());
        let remote = SyncItem::with_checksum("Qm2".to_string(), 22, 80, 1, 0, remote_id());
        s.update_local(local);
        s.update_remote(&remote_id(), remote);

        let plan = s.diff(&remote_id());
        assert!(plan.operations.is_empty());
        assert_eq!(plan.conflicts.len(), 1);
    }

    // ── 12. diff – unknown mirror returns empty plan ──────────────────────────

    #[test]
    fn test_diff_unknown_mirror_returns_empty() {
        let s = default_sync();
        let plan = s.diff(&MirrorId::new("ghost"));
        assert!(plan.is_empty());
    }

    // ── 13. apply_plan – upload ───────────────────────────────────────────────

    #[test]
    fn test_apply_plan_upload_updates_remote() {
        let mut s = default_sync();
        s.update_local(make_item("QmU", 64, 0, local_id()));
        let plan = s.diff(&remote_id());
        let res = s.apply_plan(&plan, 0);
        assert_eq!(res.operations_executed, 1);
        assert_eq!(res.bytes_transferred, 64);
        assert!(s.remote_states[&remote_id()].contains_key("QmU"));
    }

    // ── 14. apply_plan – download ─────────────────────────────────────────────

    #[test]
    fn test_apply_plan_download_updates_local() {
        let mut s = default_sync();
        s.update_remote(&remote_id(), make_item("QmD", 32, 0, remote_id()));
        let plan = s.diff(&remote_id());
        let res = s.apply_plan(&plan, 0);
        assert_eq!(res.operations_executed, 1);
        assert_eq!(res.bytes_transferred, 32);
        assert!(s.local_state.contains_key("QmD"));
    }

    // ── 15. apply_plan – in-sync after upload ─────────────────────────────────

    #[test]
    fn test_in_sync_after_apply() {
        let mut s = default_sync();
        s.update_local(make_item("QmE", 16, 0, local_id()));
        let plan = s.diff(&remote_id());
        s.apply_plan(&plan, 0);
        assert!(s.in_sync_with(&remote_id()));
    }

    // ── 16. apply_plan – audit log grows ─────────────────────────────────────

    #[test]
    fn test_apply_plan_audit_log_grows() {
        let mut s = default_sync();
        s.update_local(make_item("Qm1", 10, 0, local_id()));
        s.update_local(make_item("Qm2", 10, 0, local_id()));
        let plan = s.diff(&remote_id());
        s.apply_plan(&plan, 0);
        assert_eq!(s.audit_log.len(), 2);
    }

    // ── 17. audit_log_tail ────────────────────────────────────────────────────

    #[test]
    fn test_audit_log_tail_n_zero() {
        let s = default_sync();
        assert!(s.audit_log_tail(0).is_empty());
    }

    #[test]
    fn test_audit_log_tail_returns_last_n() {
        let mut s = default_sync();
        for i in 0..5u64 {
            let cid = format!("Qm{}", i);
            s.update_local(make_item(&cid, 10, i, local_id()));
        }
        let plan = s.diff(&remote_id());
        s.apply_plan(&plan, 0);
        let tail = s.audit_log_tail(3);
        assert_eq!(tail.len(), 3);
    }

    // ── 18. audit log cap ─────────────────────────────────────────────────────

    #[test]
    fn test_audit_log_capped_at_1000() {
        let mut s = default_sync();
        // Insert 1100 items and apply to fill beyond cap
        for i in 0..1100u64 {
            let cid = format!("QmCap{}", i);
            s.update_local(make_item(&cid, 1, i, local_id()));
            s.update_remote(&remote_id(), make_item(&cid, 999, i, remote_id()));
            // create conflicts
        }
        // Only test the cap: apply a fresh plan for items not in remote
        let mut s2 = default_sync();
        for i in 0..1100u64 {
            let cid = format!("QmFresh{}", i);
            s2.update_local(make_item(&cid, 1, i, local_id()));
        }
        let plan = s2.diff(&remote_id());
        s2.apply_plan(&plan, 0);
        assert!(s2.audit_log.len() <= 1000);
    }

    // ── 19. detect_conflicts ──────────────────────────────────────────────────

    #[test]
    fn test_detect_conflicts_finds_checksum_mismatch() {
        let mut s = StorageMirrorSync::new(local_id(), MsConflictResolution::Skip);
        s.register_mirror(remote_id());

        let local = SyncItem::with_checksum("QmX".to_string(), 1, 100, 1, 0, local_id());
        let remote = SyncItem::with_checksum("QmX".to_string(), 2, 80, 1, 0, remote_id());
        s.update_local(local);
        s.update_remote(&remote_id(), remote);

        let conflicts = s.detect_conflicts(&remote_id());
        assert_eq!(conflicts.len(), 1);
        assert_eq!(conflicts[0].cid, "QmX");
        assert_eq!(conflicts[0].conflict_type, ConflictType::ChecksumMismatch);
    }

    #[test]
    fn test_detect_conflicts_none_when_in_sync() {
        let mut s = default_sync();
        s.update_local(make_item("Qm1", 10, 0, local_id()));
        s.update_remote(&remote_id(), make_item("Qm1", 10, 0, remote_id()));
        assert!(s.detect_conflicts(&remote_id()).is_empty());
    }

    #[test]
    fn test_detect_conflicts_sorted_by_cid() {
        let mut s = StorageMirrorSync::new(local_id(), MsConflictResolution::Skip);
        s.register_mirror(remote_id());

        for ch in ["zzz", "aaa", "mmm"] {
            let local = SyncItem::with_checksum(ch.to_string(), 1, 100, 1, 0, local_id());
            let remote = SyncItem::with_checksum(ch.to_string(), 2, 80, 1, 0, remote_id());
            s.update_local(local);
            s.update_remote(&remote_id(), remote);
        }
        let conflicts = s.detect_conflicts(&remote_id());
        assert_eq!(conflicts[0].cid, "aaa");
        assert_eq!(conflicts[1].cid, "mmm");
        assert_eq!(conflicts[2].cid, "zzz");
    }

    // ── 20. local_only_cids ───────────────────────────────────────────────────

    #[test]
    fn test_local_only_cids_sorted() {
        let mut s = default_sync();
        s.update_local(make_item("z", 1, 0, local_id()));
        s.update_local(make_item("a", 1, 0, local_id()));
        s.update_local(make_item("m", 1, 0, local_id()));
        let cids = s.local_only_cids(&remote_id());
        assert_eq!(cids, vec!["a", "m", "z"]);
    }

    #[test]
    fn test_local_only_cids_excludes_shared() {
        let mut s = default_sync();
        s.update_local(make_item("shared", 1, 0, local_id()));
        s.update_local(make_item("local-only", 1, 0, local_id()));
        s.update_remote(&remote_id(), make_item("shared", 1, 0, remote_id()));
        let cids = s.local_only_cids(&remote_id());
        assert_eq!(cids, vec!["local-only"]);
    }

    // ── 21. remote_only_cids ──────────────────────────────────────────────────

    #[test]
    fn test_remote_only_cids_sorted() {
        let mut s = default_sync();
        s.update_remote(&remote_id(), make_item("z", 1, 0, remote_id()));
        s.update_remote(&remote_id(), make_item("a", 1, 0, remote_id()));
        let cids = s.remote_only_cids(&remote_id());
        assert_eq!(cids, vec!["a", "z"]);
    }

    #[test]
    fn test_remote_only_cids_unknown_mirror() {
        let s = default_sync();
        assert!(s.remote_only_cids(&MirrorId::new("ghost")).is_empty());
    }

    // ── 22. in_sync_with ──────────────────────────────────────────────────────

    #[test]
    fn test_in_sync_with_empty_states() {
        let s = default_sync();
        assert!(s.in_sync_with(&remote_id()));
    }

    #[test]
    fn test_in_sync_with_false_different_size_sets() {
        let mut s = default_sync();
        s.update_local(make_item("Qm1", 10, 0, local_id()));
        assert!(!s.in_sync_with(&remote_id()));
    }

    #[test]
    fn test_in_sync_with_false_for_unknown_mirror() {
        let s = default_sync();
        assert!(!s.in_sync_with(&MirrorId::new("ghost")));
    }

    // ── 23. stats ─────────────────────────────────────────────────────────────

    #[test]
    fn test_stats_initial() {
        let s = default_sync();
        let st = s.stats();
        assert_eq!(st.local_items, 0);
        assert_eq!(st.remote_mirrors, 1);
        assert_eq!(st.total_operations, 0);
        assert_eq!(st.bytes_transferred, 0);
    }

    #[test]
    fn test_stats_after_apply() {
        let mut s = default_sync();
        s.update_local(make_item("Qm1", 50, 0, local_id()));
        let plan = s.diff(&remote_id());
        s.apply_plan(&plan, 0);
        let st = s.stats();
        assert_eq!(st.total_operations, 1);
        assert_eq!(st.bytes_transferred, 50);
        assert_eq!(st.local_items, 1);
    }

    #[test]
    fn test_stats_conflicts_tracked() {
        let mut s = StorageMirrorSync::new(local_id(), MsConflictResolution::Skip);
        s.register_mirror(remote_id());

        let local = SyncItem::with_checksum("Qm1".to_string(), 1, 100, 1, 0, local_id());
        let remote = SyncItem::with_checksum("Qm1".to_string(), 2, 80, 1, 0, remote_id());
        s.update_local(local);
        s.update_remote(&remote_id(), remote);

        let plan = s.diff(&remote_id());
        s.apply_plan(&plan, 0);
        let st = s.stats();
        assert_eq!(st.total_conflicts_detected, 1);
    }

    // ── 24. conflict resolution – TakeLocal ───────────────────────────────────

    #[test]
    fn test_resolve_take_local_pushes_to_remote() {
        let mut s = StorageMirrorSync::new(local_id(), MsConflictResolution::TakeLocal);
        s.register_mirror(remote_id());

        let local = SyncItem::with_checksum("Qm1".to_string(), 1, 100, 1, 0, local_id());
        let remote = SyncItem::with_checksum("Qm1".to_string(), 2, 80, 1, 0, remote_id());
        s.update_local(local);
        s.update_remote(&remote_id(), remote);

        let plan = s.diff(&remote_id());
        s.apply_plan(&plan, 0);

        // After TakeLocal, remote should have local's checksum
        let remote_item = &s.remote_states[&remote_id()]["Qm1"];
        assert_eq!(remote_item.checksum, 1);
    }

    // ── 25. conflict resolution – TakeRemote ──────────────────────────────────

    #[test]
    fn test_resolve_take_remote_pulls_to_local() {
        let mut s = StorageMirrorSync::new(local_id(), MsConflictResolution::TakeRemote);
        s.register_mirror(remote_id());

        let local = SyncItem::with_checksum("Qm1".to_string(), 1, 100, 1, 0, local_id());
        let remote = SyncItem::with_checksum("Qm1".to_string(), 2, 80, 1, 0, remote_id());
        s.update_local(local);
        s.update_remote(&remote_id(), remote);

        let plan = s.diff(&remote_id());
        s.apply_plan(&plan, 0);

        let local_item = &s.local_state["Qm1"];
        assert_eq!(local_item.checksum, 2);
    }

    // ── 26. conflict resolution – TakeLargest ─────────────────────────────────

    #[test]
    fn test_resolve_take_largest_selects_bigger() {
        let mut s = StorageMirrorSync::new(local_id(), MsConflictResolution::TakeLargest);
        s.register_mirror(remote_id());

        // Local is larger
        let local = SyncItem::with_checksum("Qm1".to_string(), 1, 200, 1, 0, local_id());
        let remote = SyncItem::with_checksum("Qm1".to_string(), 2, 100, 1, 0, remote_id());
        s.update_local(local);
        s.update_remote(&remote_id(), remote);

        let plan = s.diff(&remote_id());
        s.apply_plan(&plan, 0);

        // Remote should be updated to local's (larger) item
        let remote_item = &s.remote_states[&remote_id()]["Qm1"];
        assert_eq!(remote_item.size_bytes, 200);
    }

    // ── 27. Delete operation ──────────────────────────────────────────────────

    #[test]
    fn test_apply_delete_from_remote() {
        let mut s = default_sync();
        s.update_remote(&remote_id(), make_item("QmDel", 10, 0, remote_id()));

        let plan = SyncPlan {
            operations: vec![SyncOperation::Delete {
                cid: "QmDel".to_string(),
                from_mirror: remote_id(),
            }],
            conflicts: Vec::new(),
            estimated_bytes: 0,
        };
        let res = s.apply_plan(&plan, 0);
        assert_eq!(res.operations_executed, 1);
        assert!(!s.remote_states[&remote_id()].contains_key("QmDel"));
    }

    #[test]
    fn test_apply_delete_from_local() {
        let mut s = default_sync();
        s.update_local(make_item("QmDelLocal", 10, 0, local_id()));

        let plan = SyncPlan {
            operations: vec![SyncOperation::Delete {
                cid: "QmDelLocal".to_string(),
                from_mirror: local_id(),
            }],
            conflicts: Vec::new(),
            estimated_bytes: 0,
        };
        s.apply_plan(&plan, 0);
        assert!(!s.local_state.contains_key("QmDelLocal"));
    }

    // ── 28. multiple mirrors ──────────────────────────────────────────────────

    #[test]
    fn test_multiple_mirrors_independent() {
        let mut s = StorageMirrorSync::new(local_id(), MsConflictResolution::TakeLocal);
        let r2 = MirrorId::new("remote-2");
        s.register_mirror(remote_id());
        s.register_mirror(r2.clone());

        s.update_local(make_item("QmM", 10, 0, local_id()));
        s.update_remote(&r2, make_item("QmM", 10, 0, r2.clone()));

        let plan1 = s.diff(&remote_id());
        let plan2 = s.diff(&r2);
        assert_eq!(plan1.operations.len(), 1); // upload to remote-1
        assert!(plan2.is_empty()); // already in r2
    }

    // ── 29. SyncConflict::classify ────────────────────────────────────────────

    #[test]
    fn test_classify_checksum_mismatch() {
        let local = SyncItem::with_checksum("Qm".to_string(), 10, 50, 1, 0, local_id());
        let remote = SyncItem::with_checksum("Qm".to_string(), 20, 50, 1, 0, remote_id());
        let c = SyncConflict::classify(local, remote);
        assert_eq!(c.conflict_type, ConflictType::ChecksumMismatch);
    }

    #[test]
    fn test_classify_version_conflict() {
        // Same checksum, different versions
        let local = SyncItem::with_checksum("Qm".to_string(), 10, 50, 1, 0, local_id());
        let mut remote = SyncItem::with_checksum("Qm".to_string(), 10, 50, 2, 0, remote_id());
        remote.checksum = local.checksum; // make checksums equal
        let c = SyncConflict::classify(local, remote);
        assert_eq!(c.conflict_type, ConflictType::VersionConflict);
    }

    // ── 30. Plan is_empty ─────────────────────────────────────────────────────

    #[test]
    fn test_plan_is_empty_true() {
        let plan = SyncPlan {
            operations: Vec::new(),
            conflicts: Vec::new(),
            estimated_bytes: 0,
        };
        assert!(plan.is_empty());
    }

    #[test]
    fn test_plan_is_empty_false_when_ops() {
        let plan = SyncPlan {
            operations: vec![SyncOperation::Upload {
                cid: "x".to_string(),
                to_mirror: remote_id(),
            }],
            conflicts: Vec::new(),
            estimated_bytes: 0,
        };
        assert!(!plan.is_empty());
    }

    // ── 31. MsSyncResult default ──────────────────────────────────────────────

    #[test]
    fn test_ms_sync_result_default() {
        let r = MsSyncResult::default();
        assert_eq!(r.operations_executed, 0);
        assert_eq!(r.bytes_transferred, 0);
        assert!(r.errors.is_empty());
    }

    // ── 32. MirrorSyncStats default ───────────────────────────────────────────

    #[test]
    fn test_mirror_sync_stats_default() {
        let st = MirrorSyncStats::default();
        assert_eq!(st.local_items, 0);
        assert_eq!(st.remote_mirrors, 0);
    }
}
