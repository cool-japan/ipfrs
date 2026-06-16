//! `StorageSnapshotManager` — Point-in-time snapshot management for storage
//! state with incremental deltas and restoration.
//!
//! # Overview
//!
//! This module provides two distinct APIs:
//!
//! 1. **New production-grade API** (`StorageSnapshotManager`, `SnapshotId`,
//!    `SnapshotEntry`, `SnapshotDelta`, `SsmSnapshot`, `StorageState`,
//!    `SnapshotError`, `SnapshotStats`) — full point-in-time snapshot
//!    management with FNV-1a checksums, incremental deltas, and chain-based
//!    restoration.
//!
//! 2. **Legacy block-snapshot API** (`LegacySnapshotEntry`, `fnv1a_64`,
//!    `LegacySnapshot`, `SnapshotKind`, `StorageSnapshot`, `SnapshotConfig`,
//!    `SnapshotManagerStats`) — retained for backwards compatibility.

use std::collections::{HashMap, VecDeque};
use thiserror::Error;

// ---------------------------------------------------------------------------
// SnapshotId
// ---------------------------------------------------------------------------

/// Monotonically increasing snapshot identifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SnapshotId(pub u64);

impl std::fmt::Display for SnapshotId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SnapshotId({})", self.0)
    }
}

// ---------------------------------------------------------------------------
// SnapshotEntry
// ---------------------------------------------------------------------------

/// A key-value record at snapshot time.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SnapshotEntry {
    /// The storage key.
    pub key: String,
    /// The raw value bytes.
    pub value: Vec<u8>,
    /// Logical version counter, incremented on each write to the same key.
    pub version: u64,
}

// ---------------------------------------------------------------------------
// SnapshotDelta
// ---------------------------------------------------------------------------

/// Changes relative to the previous snapshot.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SnapshotDelta {
    /// Entries that were added since the previous snapshot.
    pub added: Vec<SnapshotEntry>,
    /// Entries that were modified since the previous snapshot.
    pub modified: Vec<SnapshotEntry>,
    /// Keys that were deleted since the previous snapshot.
    pub deleted: Vec<String>,
}

impl SnapshotDelta {
    /// Returns `true` if the delta contains no changes at all.
    pub fn is_empty(&self) -> bool {
        self.added.is_empty() && self.modified.is_empty() && self.deleted.is_empty()
    }
}

// ---------------------------------------------------------------------------
// SsmSnapshot (new Snapshot type)
// ---------------------------------------------------------------------------

/// An immutable point-in-time snapshot of the storage state.
#[derive(Clone, Debug)]
pub struct SsmSnapshot {
    /// Unique monotonically-increasing identifier.
    pub id: SnapshotId,
    /// Unix timestamp (or logical clock) at which the snapshot was taken.
    pub created_at: u64,
    /// Number of live entries captured.
    pub entry_count: usize,
    /// Sum of all value byte lengths at snapshot time.
    pub total_bytes: u64,
    /// `None` for the first (full) snapshot; `Some` for incremental snapshots.
    pub delta: Option<SnapshotDelta>,
    /// FNV-1a checksum computed over all key+value pairs sorted by key.
    pub checksum: u64,
    /// Optional human-readable label.
    pub label: Option<String>,
    /// The full reconstructed state at this snapshot point (used for chain
    /// restoration without having to replay from the very beginning each time).
    ///
    /// Stored inline so that `restore_snapshot` is O(1) after chain walk.
    pub(crate) state_at_snapshot: HashMap<String, SnapshotEntry>,
}

// ---------------------------------------------------------------------------
// StorageState
// ---------------------------------------------------------------------------

/// The current live mutable storage state.
#[derive(Clone, Debug, Default)]
pub struct StorageState {
    /// Active key-value entries.
    pub entries: HashMap<String, SnapshotEntry>,
}

// ---------------------------------------------------------------------------
// SnapshotError
// ---------------------------------------------------------------------------

/// Errors returned by `StorageSnapshotManager` operations.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum SnapshotError {
    /// The requested snapshot id was not found.
    #[error("snapshot not found: {0}")]
    SnapshotNotFound(u64),

    /// Attempted to delete a snapshot that is not the oldest.
    #[error("only the oldest snapshot may be deleted")]
    CannotDeleteNonOldest,

    /// Operation requires at least one snapshot but none exist.
    #[error("snapshot chain is empty")]
    EmptySnapshotChain,

    /// The delta chain is broken (an incremental snapshot references a missing
    /// predecessor).
    #[error("delta chain is broken")]
    DeltaChainBroken,
}

// ---------------------------------------------------------------------------
// SnapshotStats
// ---------------------------------------------------------------------------

/// Aggregate statistics for `StorageSnapshotManager`.
#[derive(Clone, Debug, Default)]
pub struct SnapshotStats {
    /// Total number of snapshots retained.
    pub snapshot_count: usize,
    /// Id of the oldest retained snapshot, if any.
    pub oldest_snapshot_id: Option<u64>,
    /// Id of the newest retained snapshot, if any.
    pub newest_snapshot_id: Option<u64>,
    /// Sum of `total_bytes` across all retained snapshots.
    pub total_snapshot_bytes: u64,
    /// Number of entries in the current live state.
    pub live_entries: usize,
    /// Sum of value sizes in the current live state.
    pub live_bytes: u64,
}

// ---------------------------------------------------------------------------
// StorageSnapshotManager
// ---------------------------------------------------------------------------

/// Production-grade point-in-time snapshot manager with incremental deltas
/// and chain-based restoration.
pub struct StorageSnapshotManager {
    /// The current live mutable state.
    state: StorageState,
    /// Ordered queue of retained snapshots (oldest first).
    snapshots: VecDeque<SsmSnapshot>,
    /// Maximum number of snapshots to retain before evicting the oldest.
    max_snapshots: usize,
    /// Next snapshot id to assign.
    next_id: u64,
    /// Next version counter to assign when writing entries.
    next_version: u64,
}

// ---------------------------------------------------------------------------
// Internal FNV-1a checksum helper
// ---------------------------------------------------------------------------

/// Compute an FNV-1a 64-bit checksum over all entries sorted by key.
fn compute_checksum(entries: &HashMap<String, SnapshotEntry>) -> u64 {
    let mut keys: Vec<&str> = entries.keys().map(|s| s.as_str()).collect();
    keys.sort_unstable();
    let mut h: u64 = 14_695_981_039_346_656_037;
    for k in keys {
        for b in k.bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(1_099_511_628_211);
        }
        if let Some(e) = entries.get(k) {
            for b in &e.value {
                h ^= u64::from(*b);
                h = h.wrapping_mul(1_099_511_628_211);
            }
        }
    }
    h
}

/// Reconstruct a `HashMap<String, SnapshotEntry>` state from a snapshot.
/// For full snapshots (no delta), returns the stored state directly.
/// For incremental snapshots, the state was already pre-computed and stored.
fn reconstructed_state(snap: &SsmSnapshot) -> &HashMap<String, SnapshotEntry> {
    &snap.state_at_snapshot
}

impl StorageSnapshotManager {
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    /// Create a new, empty manager.
    ///
    /// `max_snapshots` controls how many snapshots are retained before the
    /// oldest is evicted to make room for a new one.
    pub fn new(max_snapshots: usize) -> Self {
        Self {
            state: StorageState::default(),
            snapshots: VecDeque::new(),
            max_snapshots: max_snapshots.max(1),
            next_id: 1,
            next_version: 1,
        }
    }

    // -----------------------------------------------------------------------
    // Live-state mutations
    // -----------------------------------------------------------------------

    /// Insert or update a key in the live state.
    ///
    /// The entry's `version` is set to `self.next_version` (then incremented).
    pub fn put(&mut self, key: String, value: Vec<u8>, _now: u64) {
        let version = self.next_version;
        self.next_version += 1;
        self.state.entries.insert(
            key.clone(),
            SnapshotEntry {
                key,
                value,
                version,
            },
        );
    }

    /// Remove a key from the live state.
    ///
    /// Returns `false` if the key was not present.
    pub fn delete(&mut self, key: &str) -> bool {
        self.state.entries.remove(key).is_some()
    }

    /// Look up the value bytes for a key in the live state.
    pub fn get(&self, key: &str) -> Option<&[u8]> {
        self.state.entries.get(key).map(|e| e.value.as_slice())
    }

    // -----------------------------------------------------------------------
    // Snapshot creation
    // -----------------------------------------------------------------------

    /// Capture the current live state as a new snapshot.
    ///
    /// - If this is the first snapshot, it is a *full* snapshot (`delta` is
    ///   `None`).
    /// - Otherwise, a delta relative to the preceding snapshot is computed.
    ///
    /// If the queue is already at `max_snapshots`, the oldest snapshot is
    /// evicted before appending the new one.
    pub fn take_snapshot(&mut self, label: Option<String>, now: u64) -> SnapshotId {
        let id = SnapshotId(self.next_id);
        self.next_id += 1;

        // Clone the current live state for the snapshot.
        let current_state = self.state.entries.clone();

        // Compute delta vs. the previous snapshot (if any).
        let delta: Option<SnapshotDelta> = if let Some(prev) = self.snapshots.back() {
            let prev_state = reconstructed_state(prev);
            Some(compute_delta(prev_state, &current_state))
        } else {
            None
        };

        let checksum = compute_checksum(&current_state);
        let entry_count = current_state.len();
        let total_bytes: u64 = current_state.values().map(|e| e.value.len() as u64).sum();

        let snap = SsmSnapshot {
            id,
            created_at: now,
            entry_count,
            total_bytes,
            delta,
            checksum,
            label,
            state_at_snapshot: current_state,
        };

        // Evict oldest if at capacity.
        if self.snapshots.len() >= self.max_snapshots {
            self.snapshots.pop_front();
        }

        self.snapshots.push_back(snap);
        id
    }

    // -----------------------------------------------------------------------
    // Snapshot restoration
    // -----------------------------------------------------------------------

    /// Restore the live state to the point captured by snapshot `id`.
    ///
    /// The state stored inline in the snapshot is used directly (O(1) after
    /// locating the snapshot in the deque).
    pub fn restore_snapshot(&mut self, id: SnapshotId) -> Result<(), SnapshotError> {
        let snap = self
            .snapshots
            .iter()
            .find(|s| s.id == id)
            .ok_or(SnapshotError::SnapshotNotFound(id.0))?;

        self.state.entries = snap.state_at_snapshot.clone();
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Snapshot queries
    // -----------------------------------------------------------------------

    /// Return all retained snapshots ordered oldest to newest.
    pub fn list_snapshots(&self) -> Vec<&SsmSnapshot> {
        self.snapshots.iter().collect()
    }

    /// Look up a snapshot by id.
    pub fn get_snapshot(&self, id: SnapshotId) -> Option<&SsmSnapshot> {
        self.snapshots.iter().find(|s| s.id == id)
    }

    /// Remove the oldest snapshot from the queue.
    ///
    /// Returns `CannotDeleteNonOldest` if `id` does not refer to the oldest
    /// snapshot, and `SnapshotNotFound` if no snapshot with `id` exists.
    pub fn delete_snapshot(&mut self, id: SnapshotId) -> Result<(), SnapshotError> {
        // Must exist.
        let exists = self.snapshots.iter().any(|s| s.id == id);
        if !exists {
            return Err(SnapshotError::SnapshotNotFound(id.0));
        }

        // Only allow deleting the oldest.
        let oldest_id = self
            .snapshots
            .front()
            .map(|s| s.id)
            .ok_or(SnapshotError::EmptySnapshotChain)?;

        if oldest_id != id {
            return Err(SnapshotError::CannotDeleteNonOldest);
        }

        self.snapshots.pop_front();
        Ok(())
    }

    /// Compute the full state diff between two snapshots.
    ///
    /// Reconstructs both states from the stored inline state, then compares.
    pub fn diff_snapshots(
        &self,
        a: SnapshotId,
        b: SnapshotId,
    ) -> Result<SnapshotDelta, SnapshotError> {
        let snap_a = self
            .snapshots
            .iter()
            .find(|s| s.id == a)
            .ok_or(SnapshotError::SnapshotNotFound(a.0))?;
        let snap_b = self
            .snapshots
            .iter()
            .find(|s| s.id == b)
            .ok_or(SnapshotError::SnapshotNotFound(b.0))?;

        let state_a = reconstructed_state(snap_a);
        let state_b = reconstructed_state(snap_b);

        Ok(compute_delta(state_a, state_b))
    }

    // -----------------------------------------------------------------------
    // Counts and statistics
    // -----------------------------------------------------------------------

    /// Number of snapshots currently retained.
    pub fn snapshot_count(&self) -> usize {
        self.snapshots.len()
    }

    /// Number of entries in the current live state.
    pub fn live_entry_count(&self) -> usize {
        self.state.entries.len()
    }

    /// Sum of value byte sizes in the current live state.
    pub fn live_total_bytes(&self) -> u64 {
        self.state
            .entries
            .values()
            .map(|e| e.value.len() as u64)
            .sum()
    }

    /// Return aggregate statistics.
    pub fn stats(&self) -> SnapshotStats {
        let oldest_snapshot_id = self.snapshots.front().map(|s| s.id.0);
        let newest_snapshot_id = self.snapshots.back().map(|s| s.id.0);
        let total_snapshot_bytes: u64 = self.snapshots.iter().map(|s| s.total_bytes).sum();
        SnapshotStats {
            snapshot_count: self.snapshots.len(),
            oldest_snapshot_id,
            newest_snapshot_id,
            total_snapshot_bytes,
            live_entries: self.live_entry_count(),
            live_bytes: self.live_total_bytes(),
        }
    }
}

// ---------------------------------------------------------------------------
// Internal delta helper
// ---------------------------------------------------------------------------

/// Compute a `SnapshotDelta` describing changes from `prev` to `curr`.
fn compute_delta(
    prev: &HashMap<String, SnapshotEntry>,
    curr: &HashMap<String, SnapshotEntry>,
) -> SnapshotDelta {
    let mut added = Vec::new();
    let mut modified = Vec::new();
    let mut deleted = Vec::new();

    // Keys in curr that are new or changed.
    for (key, curr_entry) in curr {
        match prev.get(key) {
            None => added.push(curr_entry.clone()),
            Some(prev_entry) => {
                if prev_entry.version != curr_entry.version || prev_entry.value != curr_entry.value
                {
                    modified.push(curr_entry.clone());
                }
            }
        }
    }

    // Keys in prev that are absent in curr.
    for key in prev.keys() {
        if !curr.contains_key(key) {
            deleted.push(key.clone());
        }
    }

    SnapshotDelta {
        added,
        modified,
        deleted,
    }
}

// ===========================================================================
// Legacy block-snapshot API — kept for backwards compatibility
// ===========================================================================

// ---------------------------------------------------------------------------
// SnapshotKind
// ---------------------------------------------------------------------------

/// Describes which blocks are recorded in a legacy snapshot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SnapshotKind {
    /// A complete snapshot of every known block.
    Full,
    /// Only the blocks that changed since the immediately preceding snapshot.
    Incremental,
    /// All blocks that changed since the most recent `Full` snapshot.
    Differential,
}

// ---------------------------------------------------------------------------
// LegacySnapshotEntry
// ---------------------------------------------------------------------------

/// A single content-addressed block reference stored inside a legacy snapshot.
#[derive(Clone, Debug, PartialEq)]
pub struct LegacySnapshotEntry {
    /// Content identifier of the block (e.g. a CIDv1 string).
    pub cid: String,
    /// Raw byte size of the block.
    pub size_bytes: u64,
    /// FNV-1a (64-bit) hash of `cid`, used for fast deduplication lookups.
    pub hash: u64,
}

impl LegacySnapshotEntry {
    /// Construct a new entry, computing the FNV-1a hash automatically.
    pub fn new(cid: impl Into<String>, size_bytes: u64) -> Self {
        let cid = cid.into();
        let hash = fnv1a_64(cid.as_bytes());
        Self {
            cid,
            size_bytes,
            hash,
        }
    }
}

// ---------------------------------------------------------------------------
// FNV-1a helpers (public)
// ---------------------------------------------------------------------------

/// FNV-1a 64-bit hash of an arbitrary byte slice.
#[inline]
pub fn fnv1a_64(data: &[u8]) -> u64 {
    const OFFSET_BASIS: u64 = 14_695_981_039_346_656_037;
    const PRIME: u64 = 1_099_511_628_211;
    let mut hash = OFFSET_BASIS;
    for &byte in data {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}

// ---------------------------------------------------------------------------
// LegacySnapshot
// ---------------------------------------------------------------------------

/// An immutable point-in-time view of a set of storage blocks (legacy API).
#[derive(Clone, Debug)]
pub struct LegacySnapshot {
    /// Unique, monotonically-increasing identifier assigned by the manager.
    pub snapshot_id: u64,
    /// What subset of blocks this snapshot captures.
    pub kind: SnapshotKind,
    /// Unix timestamp (seconds since epoch) at which this snapshot was taken.
    pub created_at_secs: u64,
    /// The block entries recorded in this snapshot.
    pub entries: Vec<LegacySnapshotEntry>,
    /// For `Incremental` and `Differential` snapshots, the id of the parent
    /// snapshot this one is relative to.  Always `None` for `Full` snapshots.
    pub parent_id: Option<u64>,
}

impl LegacySnapshot {
    /// Sum of `size_bytes` across all entries.
    pub fn total_size(&self) -> u64 {
        self.entries.iter().map(|e| e.size_bytes).sum()
    }

    /// Number of block entries in this snapshot.
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }
}

// ---------------------------------------------------------------------------
// SnapshotState
// ---------------------------------------------------------------------------

/// Lifecycle state of a managed storage snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SnapshotState {
    /// Snapshot is being assembled (not yet queryable).
    Creating,
    /// Snapshot is complete and ready for restore / diff.
    Ready,
    /// Snapshot exceeded its TTL and is no longer restorable.
    Expired,
    /// Snapshot has been explicitly deleted by the user.
    Deleted,
}

// ---------------------------------------------------------------------------
// StorageSnapshot
// ---------------------------------------------------------------------------

/// A managed, stateful snapshot of storage blocks.
#[derive(Debug, Clone)]
pub struct StorageSnapshot {
    /// Unique id assigned by the manager.
    pub id: u64,
    /// Human-readable label for this snapshot.
    pub label: String,
    /// Current lifecycle state.
    pub state: SnapshotState,
    /// Tick at which this snapshot was created.
    pub created_tick: u64,
    /// Number of blocks in this snapshot.
    pub block_count: usize,
    /// Total byte size of all blocks.
    pub total_bytes: u64,
    /// CIDs of the blocks captured in this snapshot.
    pub block_cids: Vec<String>,
}

// ---------------------------------------------------------------------------
// SnapshotConfig
// ---------------------------------------------------------------------------

/// Configuration for `LegacyStorageSnapshotManager`.
#[derive(Debug, Clone)]
pub struct SnapshotConfig {
    /// Maximum number of non-deleted snapshots allowed at once.
    pub max_snapshots: usize,
    /// How many ticks a snapshot survives before being expired.
    pub ttl_ticks: u64,
    /// Whether `tick_cleanup` should automatically remove `Deleted` snapshots.
    pub auto_cleanup: bool,
}

impl Default for SnapshotConfig {
    fn default() -> Self {
        Self {
            max_snapshots: 10,
            ttl_ticks: 1000,
            auto_cleanup: true,
        }
    }
}

// ---------------------------------------------------------------------------
// LegacySnapshotDiff
// ---------------------------------------------------------------------------

/// The difference between two legacy snapshots.
#[derive(Clone, Debug, Default)]
pub struct LegacySnapshotDiff {
    /// CIDs present in snapshot B but absent from snapshot A.
    pub added: Vec<String>,
    /// CIDs present in snapshot A but absent from snapshot B.
    pub removed: Vec<String>,
    /// Number of CIDs common to both snapshots.
    pub common: usize,
}

impl LegacySnapshotDiff {
    /// Returns `true` when no blocks were added or removed.
    pub fn is_empty(&self) -> bool {
        self.added.is_empty() && self.removed.is_empty()
    }
}

// ---------------------------------------------------------------------------
// SnapshotManagerStats
// ---------------------------------------------------------------------------

/// Aggregate statistics reported by `LegacyStorageSnapshotManager::stats`.
#[derive(Clone, Debug, Default)]
pub struct SnapshotManagerStats {
    /// Total number of snapshots currently held by the manager.
    pub total_snapshots: usize,
    /// Number of snapshots in `Ready` state.
    pub ready_count: usize,
    /// Number of snapshots in `Expired` state.
    pub expired_count: usize,
    /// Lifetime count of snapshots created.
    pub total_created: u64,
    /// Lifetime count of snapshots deleted.
    pub total_deleted: u64,
}

// ---------------------------------------------------------------------------
// LegacyStorageSnapshotManager
// ---------------------------------------------------------------------------

/// Legacy snapshot manager (kept for backwards compatibility).
///
/// Manages point-in-time storage snapshots with TTL, lifecycle states,
/// auto-cleanup, and diff computation.
pub struct LegacyStorageSnapshotManager {
    config: SnapshotConfig,
    snapshots: HashMap<u64, StorageSnapshot>,
    next_id: u64,
    current_tick: u64,
    total_created: u64,
    total_deleted: u64,
}

impl LegacyStorageSnapshotManager {
    /// Create a new manager with the given configuration.
    pub fn new(config: SnapshotConfig) -> Self {
        Self {
            config,
            snapshots: HashMap::new(),
            next_id: 1,
            current_tick: 0,
            total_created: 0,
            total_deleted: 0,
        }
    }

    /// Create a snapshot with the given label and block data.
    ///
    /// Returns the assigned snapshot id on success.
    /// Returns an error if the maximum number of active (non-deleted)
    /// snapshots has been reached.
    pub fn create_snapshot(
        &mut self,
        label: &str,
        block_cids: Vec<String>,
        total_bytes: u64,
    ) -> Result<u64, String> {
        let active_count = self
            .snapshots
            .values()
            .filter(|s| s.state != SnapshotState::Deleted)
            .count();
        if active_count >= self.config.max_snapshots {
            return Err(format!(
                "max snapshots reached ({})",
                self.config.max_snapshots
            ));
        }

        let id = self.next_id;
        self.next_id += 1;
        self.total_created += 1;

        let block_count = block_cids.len();
        let snapshot = StorageSnapshot {
            id,
            label: label.to_owned(),
            state: SnapshotState::Ready,
            created_tick: self.current_tick,
            block_count,
            total_bytes,
            block_cids,
        };
        self.snapshots.insert(id, snapshot);
        Ok(id)
    }

    /// Look up a snapshot by id.
    pub fn get_snapshot(&self, id: u64) -> Option<&StorageSnapshot> {
        self.snapshots.get(&id)
    }

    /// Mark a snapshot as `Deleted`.
    pub fn delete_snapshot(&mut self, id: u64) -> Result<(), String> {
        let snap = self
            .snapshots
            .get_mut(&id)
            .ok_or_else(|| format!("snapshot {} not found", id))?;
        if snap.state == SnapshotState::Deleted {
            return Err(format!("snapshot {} already deleted", id));
        }
        snap.state = SnapshotState::Deleted;
        self.total_deleted += 1;
        Ok(())
    }

    /// Restore a snapshot, returning a clone of its block CIDs.
    pub fn restore_snapshot(&self, id: u64) -> Result<Vec<String>, String> {
        let snap = self
            .snapshots
            .get(&id)
            .ok_or_else(|| format!("snapshot {} not found", id))?;
        match snap.state {
            SnapshotState::Ready => Ok(snap.block_cids.clone()),
            SnapshotState::Expired => Err(format!("snapshot {} is expired", id)),
            SnapshotState::Deleted => Err(format!("snapshot {} is deleted", id)),
            SnapshotState::Creating => Err(format!("snapshot {} is still being created", id)),
        }
    }

    /// List all snapshots as `(id, label, state)` tuples.
    pub fn list_snapshots(&self) -> Vec<(u64, String, SnapshotState)> {
        let mut list: Vec<(u64, String, SnapshotState)> = self
            .snapshots
            .values()
            .map(|s| (s.id, s.label.clone(), s.state))
            .collect();
        list.sort_by_key(|(id, _, _)| *id);
        list
    }

    /// Advance the internal tick, expire TTL-exceeded snapshots, and optionally
    /// remove `Deleted` snapshots if `auto_cleanup` is enabled.
    pub fn tick_cleanup(&mut self) {
        self.current_tick += 1;

        let ttl = self.config.ttl_ticks;
        let current = self.current_tick;

        for snap in self.snapshots.values_mut() {
            if snap.state == SnapshotState::Ready && current.saturating_sub(snap.created_tick) > ttl
            {
                snap.state = SnapshotState::Expired;
            }
        }

        if self.config.auto_cleanup {
            self.snapshots
                .retain(|_, s| s.state != SnapshotState::Deleted);
        }
    }

    /// Count snapshots currently in `Ready` state.
    pub fn ready_count(&self) -> usize {
        self.snapshots
            .values()
            .filter(|s| s.state == SnapshotState::Ready)
            .count()
    }

    /// Compute the diff between two snapshots identified by their ids.
    pub fn diff_snapshots(&self, id_a: u64, id_b: u64) -> Result<LegacySnapshotDiff, String> {
        use std::collections::HashSet;

        let snap_a = self
            .snapshots
            .get(&id_a)
            .ok_or_else(|| format!("snapshot {} not found", id_a))?;
        let snap_b = self
            .snapshots
            .get(&id_b)
            .ok_or_else(|| format!("snapshot {} not found", id_b))?;

        let set_a: HashSet<&str> = snap_a.block_cids.iter().map(|s| s.as_str()).collect();
        let set_b: HashSet<&str> = snap_b.block_cids.iter().map(|s| s.as_str()).collect();

        let mut added: Vec<String> = set_b.difference(&set_a).map(|s| (*s).to_owned()).collect();
        let mut removed: Vec<String> = set_a.difference(&set_b).map(|s| (*s).to_owned()).collect();
        added.sort_unstable();
        removed.sort_unstable();
        let common = set_a.intersection(&set_b).count();

        Ok(LegacySnapshotDiff {
            added,
            removed,
            common,
        })
    }

    /// Return aggregate statistics.
    pub fn stats(&self) -> SnapshotManagerStats {
        let mut ready_count = 0usize;
        let mut expired_count = 0usize;
        for snap in self.snapshots.values() {
            match snap.state {
                SnapshotState::Ready => ready_count += 1,
                SnapshotState::Expired => expired_count += 1,
                _ => {}
            }
        }
        SnapshotManagerStats {
            total_snapshots: self.snapshots.len(),
            ready_count,
            expired_count,
            total_created: self.total_created,
            total_deleted: self.total_deleted,
        }
    }
}

impl Default for LegacyStorageSnapshotManager {
    fn default() -> Self {
        Self::new(SnapshotConfig::default())
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::snapshot_manager::SnapshotEntry;
    use crate::snapshot_manager::{
        compute_checksum, fnv1a_64, LegacySnapshotDiff, LegacyStorageSnapshotManager,
        SnapshotConfig, SnapshotError, SnapshotId, SnapshotState, StorageSnapshotManager,
    };

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn build_mgr(max: usize) -> StorageSnapshotManager {
        StorageSnapshotManager::new(max)
    }

    // -----------------------------------------------------------------------
    // 1. New empty manager has zero entries and zero snapshots
    // -----------------------------------------------------------------------
    #[test]
    fn test_new_manager_is_empty() {
        let mgr = build_mgr(10);
        assert_eq!(mgr.live_entry_count(), 0);
        assert_eq!(mgr.live_total_bytes(), 0);
        assert_eq!(mgr.snapshot_count(), 0);
        assert!(mgr.list_snapshots().is_empty());
    }

    // -----------------------------------------------------------------------
    // 2. put inserts an entry and get retrieves it
    // -----------------------------------------------------------------------
    #[test]
    fn test_put_and_get() {
        let mut mgr = build_mgr(10);
        mgr.put("alpha".into(), b"hello".to_vec(), 1);
        assert_eq!(mgr.get("alpha"), Some(b"hello".as_slice()));
    }

    // -----------------------------------------------------------------------
    // 3. get returns None for unknown key
    // -----------------------------------------------------------------------
    #[test]
    fn test_get_missing_key() {
        let mgr = build_mgr(10);
        assert!(mgr.get("nope").is_none());
    }

    // -----------------------------------------------------------------------
    // 4. delete removes an existing key
    // -----------------------------------------------------------------------
    #[test]
    fn test_delete_existing_key() {
        let mut mgr = build_mgr(10);
        mgr.put("k".into(), b"v".to_vec(), 0);
        let removed = mgr.delete("k");
        assert!(removed);
        assert!(mgr.get("k").is_none());
    }

    // -----------------------------------------------------------------------
    // 5. delete returns false for non-existent key
    // -----------------------------------------------------------------------
    #[test]
    fn test_delete_missing_key() {
        let mut mgr = build_mgr(10);
        assert!(!mgr.delete("ghost"));
    }

    // -----------------------------------------------------------------------
    // 6. live_entry_count tracks additions and deletions
    // -----------------------------------------------------------------------
    #[test]
    fn test_live_entry_count() {
        let mut mgr = build_mgr(10);
        assert_eq!(mgr.live_entry_count(), 0);
        mgr.put("a".into(), b"1".to_vec(), 0);
        mgr.put("b".into(), b"2".to_vec(), 0);
        assert_eq!(mgr.live_entry_count(), 2);
        mgr.delete("a");
        assert_eq!(mgr.live_entry_count(), 1);
    }

    // -----------------------------------------------------------------------
    // 7. live_total_bytes is sum of value lengths
    // -----------------------------------------------------------------------
    #[test]
    fn test_live_total_bytes() {
        let mut mgr = build_mgr(10);
        mgr.put("x".into(), vec![0u8; 10], 0);
        mgr.put("y".into(), vec![0u8; 20], 0);
        assert_eq!(mgr.live_total_bytes(), 30);
    }

    // -----------------------------------------------------------------------
    // 8. take_snapshot on empty state produces full snapshot with no delta
    // -----------------------------------------------------------------------
    #[test]
    fn test_first_snapshot_is_full() {
        let mut mgr = build_mgr(10);
        let id = mgr.take_snapshot(None, 100);
        let snap = mgr.get_snapshot(id).expect("must exist");
        assert!(
            snap.delta.is_none(),
            "first snapshot must be full (no delta)"
        );
    }

    // -----------------------------------------------------------------------
    // 9. take_snapshot returns monotonically increasing ids
    // -----------------------------------------------------------------------
    #[test]
    fn test_snapshot_ids_monotonic() {
        let mut mgr = build_mgr(10);
        let id1 = mgr.take_snapshot(None, 0);
        let id2 = mgr.take_snapshot(None, 1);
        let id3 = mgr.take_snapshot(None, 2);
        assert!(id1 < id2);
        assert!(id2 < id3);
    }

    // -----------------------------------------------------------------------
    // 10. take_snapshot captures correct entry_count and total_bytes
    // -----------------------------------------------------------------------
    #[test]
    fn test_snapshot_entry_count_and_bytes() {
        let mut mgr = build_mgr(10);
        mgr.put("k1".into(), vec![1u8; 8], 0);
        mgr.put("k2".into(), vec![2u8; 16], 0);
        let id = mgr.take_snapshot(None, 0);
        let snap = mgr.get_snapshot(id).expect("must exist");
        assert_eq!(snap.entry_count, 2);
        assert_eq!(snap.total_bytes, 24);
    }

    // -----------------------------------------------------------------------
    // 11. take_snapshot captures label
    // -----------------------------------------------------------------------
    #[test]
    fn test_snapshot_label() {
        let mut mgr = build_mgr(10);
        let id = mgr.take_snapshot(Some("my-label".to_owned()), 0);
        let snap = mgr.get_snapshot(id).expect("must exist");
        assert_eq!(snap.label.as_deref(), Some("my-label"));
    }

    // -----------------------------------------------------------------------
    // 12. Second snapshot has a delta with added entries
    // -----------------------------------------------------------------------
    #[test]
    fn test_second_snapshot_delta_added() {
        let mut mgr = build_mgr(10);
        mgr.take_snapshot(None, 0); // full (empty)
        mgr.put("new".into(), b"data".to_vec(), 1);
        let id2 = mgr.take_snapshot(None, 2);
        let snap2 = mgr.get_snapshot(id2).expect("must exist");
        let delta = snap2
            .delta
            .as_ref()
            .expect("second snapshot must have delta");
        assert_eq!(delta.added.len(), 1);
        assert_eq!(delta.added[0].key, "new");
        assert!(delta.modified.is_empty());
        assert!(delta.deleted.is_empty());
    }

    // -----------------------------------------------------------------------
    // 13. Delta captures modified entries
    // -----------------------------------------------------------------------
    #[test]
    fn test_snapshot_delta_modified() {
        let mut mgr = build_mgr(10);
        mgr.put("k".into(), b"v1".to_vec(), 0);
        mgr.take_snapshot(None, 0);
        mgr.put("k".into(), b"v2".to_vec(), 1);
        let id2 = mgr.take_snapshot(None, 1);
        let snap2 = mgr.get_snapshot(id2).expect("must exist");
        let delta = snap2.delta.as_ref().expect("must have delta");
        assert!(delta.added.is_empty());
        assert_eq!(delta.modified.len(), 1);
        assert_eq!(delta.modified[0].key, "k");
        assert_eq!(delta.modified[0].value, b"v2");
        assert!(delta.deleted.is_empty());
    }

    // -----------------------------------------------------------------------
    // 14. Delta captures deleted entries
    // -----------------------------------------------------------------------
    #[test]
    fn test_snapshot_delta_deleted() {
        let mut mgr = build_mgr(10);
        mgr.put("gone".into(), b"bye".to_vec(), 0);
        mgr.take_snapshot(None, 0);
        mgr.delete("gone");
        let id2 = mgr.take_snapshot(None, 1);
        let snap2 = mgr.get_snapshot(id2).expect("must exist");
        let delta = snap2.delta.as_ref().expect("must have delta");
        assert!(delta.added.is_empty());
        assert!(delta.modified.is_empty());
        assert_eq!(delta.deleted, vec!["gone".to_owned()]);
    }

    // -----------------------------------------------------------------------
    // 15. restore_snapshot rebuilds live state
    // -----------------------------------------------------------------------
    #[test]
    fn test_restore_snapshot_basic() {
        let mut mgr = build_mgr(10);
        mgr.put("a".into(), b"original".to_vec(), 0);
        let id = mgr.take_snapshot(None, 0);

        // Mutate live state.
        mgr.put("a".into(), b"changed".to_vec(), 1);
        mgr.put("b".into(), b"new".to_vec(), 1);
        assert_eq!(mgr.get("a"), Some(b"changed".as_slice()));

        // Restore to snapshot.
        mgr.restore_snapshot(id).expect("restore should succeed");
        assert_eq!(mgr.get("a"), Some(b"original".as_slice()));
        assert!(mgr.get("b").is_none(), "b was not in snapshot");
    }

    // -----------------------------------------------------------------------
    // 16. restore_snapshot returns SnapshotNotFound for unknown id
    // -----------------------------------------------------------------------
    #[test]
    fn test_restore_snapshot_not_found() {
        let mut mgr = build_mgr(10);
        let result = mgr.restore_snapshot(SnapshotId(999));
        assert_eq!(result, Err(SnapshotError::SnapshotNotFound(999)));
    }

    // -----------------------------------------------------------------------
    // 17. get_snapshot returns None for unknown id
    // -----------------------------------------------------------------------
    #[test]
    fn test_get_snapshot_unknown_id() {
        let mgr = build_mgr(10);
        assert!(mgr.get_snapshot(SnapshotId(42)).is_none());
    }

    // -----------------------------------------------------------------------
    // 18. list_snapshots returns ordered oldest-to-newest
    // -----------------------------------------------------------------------
    #[test]
    fn test_list_snapshots_order() {
        let mut mgr = build_mgr(10);
        let id1 = mgr.take_snapshot(None, 0);
        let id2 = mgr.take_snapshot(None, 1);
        let id3 = mgr.take_snapshot(None, 2);
        let list = mgr.list_snapshots();
        assert_eq!(list.len(), 3);
        assert_eq!(list[0].id, id1);
        assert_eq!(list[1].id, id2);
        assert_eq!(list[2].id, id3);
    }

    // -----------------------------------------------------------------------
    // 19. Oldest snapshot is evicted when max_snapshots is exceeded
    // -----------------------------------------------------------------------
    #[test]
    fn test_max_snapshots_eviction() {
        let mut mgr = build_mgr(3);
        let id1 = mgr.take_snapshot(None, 0);
        let id2 = mgr.take_snapshot(None, 1);
        let id3 = mgr.take_snapshot(None, 2);
        // id1 should still be present before overflow.
        assert!(mgr.get_snapshot(id1).is_some());
        // Add a 4th — id1 should be evicted.
        let id4 = mgr.take_snapshot(None, 3);
        assert!(mgr.get_snapshot(id1).is_none(), "id1 must be evicted");
        assert!(mgr.get_snapshot(id2).is_some());
        assert!(mgr.get_snapshot(id3).is_some());
        assert!(mgr.get_snapshot(id4).is_some());
        assert_eq!(mgr.snapshot_count(), 3);
    }

    // -----------------------------------------------------------------------
    // 20. delete_snapshot removes the oldest snapshot
    // -----------------------------------------------------------------------
    #[test]
    fn test_delete_oldest_snapshot() {
        let mut mgr = build_mgr(10);
        let id1 = mgr.take_snapshot(None, 0);
        let _id2 = mgr.take_snapshot(None, 1);
        mgr.delete_snapshot(id1)
            .expect("delete oldest should succeed");
        assert!(mgr.get_snapshot(id1).is_none());
        assert_eq!(mgr.snapshot_count(), 1);
    }

    // -----------------------------------------------------------------------
    // 21. delete_snapshot CannotDeleteNonOldest for non-oldest id
    // -----------------------------------------------------------------------
    #[test]
    fn test_delete_non_oldest_snapshot_errors() {
        let mut mgr = build_mgr(10);
        let _id1 = mgr.take_snapshot(None, 0);
        let id2 = mgr.take_snapshot(None, 1);
        let result = mgr.delete_snapshot(id2);
        assert_eq!(result, Err(SnapshotError::CannotDeleteNonOldest));
    }

    // -----------------------------------------------------------------------
    // 22. delete_snapshot SnapshotNotFound for unknown id
    // -----------------------------------------------------------------------
    #[test]
    fn test_delete_snapshot_not_found() {
        let mut mgr = build_mgr(10);
        let result = mgr.delete_snapshot(SnapshotId(404));
        assert_eq!(result, Err(SnapshotError::SnapshotNotFound(404)));
    }

    // -----------------------------------------------------------------------
    // 23. diff_snapshots correctly identifies changes between two snapshots
    // -----------------------------------------------------------------------
    #[test]
    fn test_diff_snapshots() {
        let mut mgr = build_mgr(10);
        mgr.put("common".into(), b"same".to_vec(), 0);
        mgr.put("will-change".into(), b"old".to_vec(), 0);
        mgr.put("will-delete".into(), b"bye".to_vec(), 0);
        let id_a = mgr.take_snapshot(None, 0);

        mgr.put("will-change".into(), b"new".to_vec(), 1);
        mgr.delete("will-delete");
        mgr.put("added".into(), b"fresh".to_vec(), 1);
        let id_b = mgr.take_snapshot(None, 1);

        let delta = mgr.diff_snapshots(id_a, id_b).expect("diff should succeed");
        assert_eq!(delta.added.len(), 1);
        assert_eq!(delta.added[0].key, "added");
        assert_eq!(delta.modified.len(), 1);
        assert_eq!(delta.modified[0].key, "will-change");
        assert_eq!(delta.deleted, vec!["will-delete".to_owned()]);
    }

    // -----------------------------------------------------------------------
    // 24. diff_snapshots returns SnapshotNotFound for bad ids
    // -----------------------------------------------------------------------
    #[test]
    fn test_diff_snapshots_not_found() {
        let mut mgr = build_mgr(10);
        let id = mgr.take_snapshot(None, 0);
        assert_eq!(
            mgr.diff_snapshots(id, SnapshotId(999)),
            Err(SnapshotError::SnapshotNotFound(999))
        );
        assert_eq!(
            mgr.diff_snapshots(SnapshotId(999), id),
            Err(SnapshotError::SnapshotNotFound(999))
        );
    }

    // -----------------------------------------------------------------------
    // 25. stats() reflects accurate counts
    // -----------------------------------------------------------------------
    #[test]
    fn test_stats_accuracy() {
        let mut mgr = build_mgr(10);
        mgr.put("a".into(), vec![0u8; 5], 0);
        mgr.put("b".into(), vec![0u8; 10], 0);
        let id1 = mgr.take_snapshot(None, 0);
        let _id2 = mgr.take_snapshot(None, 1);

        let stats = mgr.stats();
        assert_eq!(stats.snapshot_count, 2);
        assert_eq!(stats.oldest_snapshot_id, Some(id1.0));
        assert_eq!(stats.live_entries, 2);
        assert_eq!(stats.live_bytes, 15);
    }

    // -----------------------------------------------------------------------
    // 26. stats() on empty manager
    // -----------------------------------------------------------------------
    #[test]
    fn test_stats_empty() {
        let mgr = build_mgr(10);
        let stats = mgr.stats();
        assert_eq!(stats.snapshot_count, 0);
        assert!(stats.oldest_snapshot_id.is_none());
        assert!(stats.newest_snapshot_id.is_none());
        assert_eq!(stats.total_snapshot_bytes, 0);
        assert_eq!(stats.live_entries, 0);
        assert_eq!(stats.live_bytes, 0);
    }

    // -----------------------------------------------------------------------
    // 27. checksum is deterministic
    // -----------------------------------------------------------------------
    #[test]
    fn test_checksum_deterministic() {
        let mut map1 = HashMap::new();
        map1.insert(
            "k1".to_owned(),
            SnapshotEntry {
                key: "k1".into(),
                value: b"val1".to_vec(),
                version: 1,
            },
        );
        map1.insert(
            "k2".to_owned(),
            SnapshotEntry {
                key: "k2".into(),
                value: b"val2".to_vec(),
                version: 2,
            },
        );

        let mut map2 = HashMap::new();
        map2.insert(
            "k2".to_owned(),
            SnapshotEntry {
                key: "k2".into(),
                value: b"val2".to_vec(),
                version: 2,
            },
        );
        map2.insert(
            "k1".to_owned(),
            SnapshotEntry {
                key: "k1".into(),
                value: b"val1".to_vec(),
                version: 1,
            },
        );

        assert_eq!(
            compute_checksum(&map1),
            compute_checksum(&map2),
            "checksum must be order-independent"
        );
    }

    // -----------------------------------------------------------------------
    // 28. checksum changes when value changes
    // -----------------------------------------------------------------------
    #[test]
    fn test_checksum_changes_on_mutation() {
        let mut map1 = HashMap::new();
        map1.insert(
            "k".to_owned(),
            SnapshotEntry {
                key: "k".into(),
                value: b"original".to_vec(),
                version: 1,
            },
        );
        let c1 = compute_checksum(&map1);

        let mut map2 = HashMap::new();
        map2.insert(
            "k".to_owned(),
            SnapshotEntry {
                key: "k".into(),
                value: b"changed!".to_vec(),
                version: 2,
            },
        );
        let c2 = compute_checksum(&map2);

        assert_ne!(c1, c2);
    }

    // -----------------------------------------------------------------------
    // 29. empty delta is_empty() returns true
    // -----------------------------------------------------------------------
    #[test]
    fn test_snapshot_delta_is_empty() {
        let mut mgr = build_mgr(10);
        mgr.put("x".into(), b"same".to_vec(), 0);
        let id1 = mgr.take_snapshot(None, 0);
        // No changes between id1 and id2.
        let id2 = mgr.take_snapshot(None, 1);
        let delta = mgr.diff_snapshots(id1, id2).expect("ok");
        assert!(delta.is_empty());
    }

    // -----------------------------------------------------------------------
    // 30. restore to an earlier snapshot then re-snapshot preserves history
    // -----------------------------------------------------------------------
    #[test]
    fn test_restore_and_resnap() {
        let mut mgr = build_mgr(10);
        mgr.put("key".into(), b"v1".to_vec(), 0);
        let id1 = mgr.take_snapshot(Some("snap-1".into()), 0);
        mgr.put("key".into(), b"v2".to_vec(), 1);
        mgr.take_snapshot(Some("snap-2".into()), 1);

        // Restore to snap-1.
        mgr.restore_snapshot(id1).expect("restore");
        assert_eq!(mgr.get("key"), Some(b"v1".as_slice()));

        // Take a new snapshot after restore.
        let id3 = mgr.take_snapshot(Some("snap-3".into()), 2);
        let snap3 = mgr.get_snapshot(id3).expect("must exist");
        // snap3 is relative to snap-2 (the previous head in the deque).
        assert_eq!(snap3.entry_count, 1);
    }

    // -----------------------------------------------------------------------
    // 31. Multiple puts to same key preserve latest version
    // -----------------------------------------------------------------------
    #[test]
    fn test_put_overwrites_value() {
        let mut mgr = build_mgr(10);
        mgr.put("k".into(), b"first".to_vec(), 0);
        mgr.put("k".into(), b"second".to_vec(), 0);
        mgr.put("k".into(), b"third".to_vec(), 0);
        assert_eq!(mgr.get("k"), Some(b"third".as_slice()));
        assert_eq!(mgr.live_entry_count(), 1);
    }

    // -----------------------------------------------------------------------
    // 32. snapshot_count returns correct value
    // -----------------------------------------------------------------------
    #[test]
    fn test_snapshot_count() {
        let mut mgr = build_mgr(10);
        assert_eq!(mgr.snapshot_count(), 0);
        mgr.take_snapshot(None, 0);
        assert_eq!(mgr.snapshot_count(), 1);
        mgr.take_snapshot(None, 1);
        assert_eq!(mgr.snapshot_count(), 2);
    }

    // -----------------------------------------------------------------------
    // 33. diff_snapshots on same id returns empty delta
    // -----------------------------------------------------------------------
    #[test]
    fn test_diff_same_snapshot() {
        let mut mgr = build_mgr(10);
        mgr.put("k".into(), b"v".to_vec(), 0);
        let id = mgr.take_snapshot(None, 0);
        let delta = mgr.diff_snapshots(id, id).expect("ok");
        assert!(delta.is_empty());
    }

    // -----------------------------------------------------------------------
    // 34. stats newest_snapshot_id updated after multiple snapshots
    // -----------------------------------------------------------------------
    #[test]
    fn test_stats_newest_id() {
        let mut mgr = build_mgr(10);
        mgr.take_snapshot(None, 0);
        let id_last = mgr.take_snapshot(None, 1);
        let stats = mgr.stats();
        assert_eq!(stats.newest_snapshot_id, Some(id_last.0));
    }

    // -----------------------------------------------------------------------
    // 35. snapshot checksum stored in SsmSnapshot
    // -----------------------------------------------------------------------
    #[test]
    fn test_snapshot_checksum_stored() {
        let mut mgr = build_mgr(10);
        mgr.put("ck".into(), b"data".to_vec(), 0);
        let id = mgr.take_snapshot(None, 0);
        let snap = mgr.get_snapshot(id).expect("must exist");
        assert_ne!(
            snap.checksum, 0,
            "checksum must be non-zero for non-empty state"
        );
    }

    // -----------------------------------------------------------------------
    // 36. Snapshot created_at stores the provided timestamp
    // -----------------------------------------------------------------------
    #[test]
    fn test_snapshot_created_at() {
        let mut mgr = build_mgr(10);
        let id = mgr.take_snapshot(None, 1234567890);
        let snap = mgr.get_snapshot(id).expect("must exist");
        assert_eq!(snap.created_at, 1234567890);
    }

    // -----------------------------------------------------------------------
    // Legacy API tests
    // -----------------------------------------------------------------------

    fn default_legacy_config() -> SnapshotConfig {
        SnapshotConfig::default()
    }

    fn small_legacy_config(max: usize, ttl: u64) -> SnapshotConfig {
        SnapshotConfig {
            max_snapshots: max,
            ttl_ticks: ttl,
            auto_cleanup: true,
        }
    }

    // -----------------------------------------------------------------------
    // L1. Legacy create snapshot
    // -----------------------------------------------------------------------
    #[test]
    fn test_legacy_create_snapshot_fields() {
        let mut mgr = LegacyStorageSnapshotManager::new(default_legacy_config());
        let cids = vec!["cid_a".to_owned(), "cid_b".to_owned()];
        let id = mgr
            .create_snapshot("backup-1", cids.clone(), 300)
            .expect("should succeed");

        let snap = mgr.get_snapshot(id).expect("snapshot must exist");
        assert_eq!(snap.id, id);
        assert_eq!(snap.label, "backup-1");
        assert_eq!(snap.state, SnapshotState::Ready);
        assert_eq!(snap.block_count, 2);
        assert_eq!(snap.total_bytes, 300);
        assert_eq!(snap.block_cids, cids);
    }

    // -----------------------------------------------------------------------
    // L2. Legacy restore
    // -----------------------------------------------------------------------
    #[test]
    fn test_legacy_restore_snapshot() {
        let mut mgr = LegacyStorageSnapshotManager::new(default_legacy_config());
        let cids = vec!["a".into(), "b".into(), "c".into()];
        let id = mgr
            .create_snapshot("restore-me", cids.clone(), 100)
            .expect("ok");
        let restored = mgr.restore_snapshot(id).expect("restore should succeed");
        assert_eq!(restored, cids);
    }

    // -----------------------------------------------------------------------
    // L3. Legacy diff — added
    // -----------------------------------------------------------------------
    #[test]
    fn test_legacy_diff_snapshots_added() {
        let mut mgr = LegacyStorageSnapshotManager::new(default_legacy_config());
        let id_a = mgr
            .create_snapshot("a", vec!["common".into()], 10)
            .expect("ok");
        let id_b = mgr
            .create_snapshot("b", vec!["common".into(), "new".into()], 20)
            .expect("ok");
        let diff: LegacySnapshotDiff = mgr.diff_snapshots(id_a, id_b).expect("ok");
        assert_eq!(diff.added, vec!["new".to_owned()]);
        assert!(diff.removed.is_empty());
        assert_eq!(diff.common, 1);
    }

    // -----------------------------------------------------------------------
    // L4. Legacy TTL expiration
    // -----------------------------------------------------------------------
    #[test]
    fn test_legacy_ttl_expiration() {
        let mut mgr = LegacyStorageSnapshotManager::new(small_legacy_config(10, 2));
        let id = mgr.create_snapshot("ttl-test", vec![], 0).expect("ok");
        mgr.tick_cleanup();
        mgr.tick_cleanup();
        assert_eq!(
            mgr.get_snapshot(id).expect("exists").state,
            SnapshotState::Ready,
        );
        mgr.tick_cleanup();
        assert_eq!(
            mgr.get_snapshot(id).expect("exists").state,
            SnapshotState::Expired,
        );
    }

    // -----------------------------------------------------------------------
    // L5. Legacy max snapshots limit
    // -----------------------------------------------------------------------
    #[test]
    fn test_legacy_max_snapshots() {
        let mut mgr = LegacyStorageSnapshotManager::new(small_legacy_config(2, 1000));
        mgr.create_snapshot("s1", vec![], 0).expect("ok");
        mgr.create_snapshot("s2", vec![], 0).expect("ok");
        let result = mgr.create_snapshot("s3", vec![], 0);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // 37. fnv1a_64 is deterministic
    // -----------------------------------------------------------------------
    #[test]
    fn test_fnv1a_deterministic() {
        let h1 = fnv1a_64(b"hello");
        let h2 = fnv1a_64(b"hello");
        assert_eq!(h1, h2);
        assert_ne!(h1, fnv1a_64(b"world"));
    }

    // -----------------------------------------------------------------------
    // 38. Snapshot with label=None stores None
    // -----------------------------------------------------------------------
    #[test]
    fn test_snapshot_no_label() {
        let mut mgr = build_mgr(10);
        let id = mgr.take_snapshot(None, 0);
        let snap = mgr.get_snapshot(id).expect("must exist");
        assert!(snap.label.is_none());
    }

    // -----------------------------------------------------------------------
    // 39. delete_snapshot on single-element deque leaves it empty
    // -----------------------------------------------------------------------
    #[test]
    fn test_delete_only_snapshot() {
        let mut mgr = build_mgr(10);
        let id = mgr.take_snapshot(None, 0);
        mgr.delete_snapshot(id).expect("should succeed");
        assert_eq!(mgr.snapshot_count(), 0);
        // Now trying to delete again fails with NotFound.
        let result = mgr.delete_snapshot(id);
        assert_eq!(result, Err(SnapshotError::SnapshotNotFound(id.0)));
    }

    // -----------------------------------------------------------------------
    // 40. SnapshotId Display formatting
    // -----------------------------------------------------------------------
    #[test]
    fn test_snapshot_id_display() {
        let sid = SnapshotId(42);
        assert_eq!(format!("{}", sid), "SnapshotId(42)");
    }
}
