//! Peer Sync Protocol — bidirectional state synchronisation using vector clocks and CRDTs.
//!
//! [`PeerSyncProtocol`] implements a production-grade, conflict-free replicated data type
//! (CRDT) engine.  Each node maintains a [`VectorClock`] that it increments on every local
//! write.  Remote operations are applied through [`PeerSyncProtocol::apply_remote_op`], which
//! resolves write–write conflicts according to a pluggable [`ConflictPolicy`].
//!
//! ## Design goals
//! - **Convergence**: any two nodes that have received the same set of operations end up in
//!   identical state regardless of delivery order.
//! - **No data loss by default**: `LastWriteWins` and `HighestClock` policies always keep
//!   one version; `MergeBytes` retains both.
//! - **Tombstoning**: deleted keys are permanently suppressed so late-arriving puts for the
//!   same key are silently discarded.
//! - **Delta sync**: [`PeerSyncProtocol::generate_delta`] returns only the entries that a
//!   remote peer is missing, batched into a single [`SyncOperation::Merge`].
//!
//! ## Example
//! ```rust
//! use ipfrs_network::peer_sync_protocol::{
//!     PeerSyncProtocol, ConflictPolicy, SyncOperation, VectorClock,
//! };
//!
//! let mut node_a = PeerSyncProtocol::new("node-a".to_string(), ConflictPolicy::LastWriteWins);
//! let entry = node_a.local_put("greeting".to_string(), b"hello".to_vec(), 1000);
//!
//! let mut node_b = PeerSyncProtocol::new("node-b".to_string(), ConflictPolicy::LastWriteWins);
//! let op = SyncOperation::Put { entry };
//! node_b.apply_remote_op("node-a", op, 1001).unwrap();
//!
//! assert_eq!(node_b.get("greeting").unwrap().value, b"hello");
//! ```

use std::collections::{HashMap, HashSet, VecDeque};

use thiserror::Error;

// ─── Error type ─────────────────────────────────────────────────────────────

/// Errors returned by [`PeerSyncProtocol`] operations.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum SyncError {
    /// The key has already been deleted and its tombstone is permanent.
    #[error("key '{0}' is tombstoned and cannot be re-inserted")]
    Tombstoned(String),

    /// A conflict was detected but the active policy is [`ConflictPolicy::RejectConflict`].
    #[error("conflict rejected for key '{key}'")]
    ConflictRejected { key: String },

    /// The operation is structurally invalid (e.g. an empty Merge).
    #[error("invalid sync operation")]
    InvalidOperation,
}

// ─── VectorClock ─────────────────────────────────────────────────────────────

/// A logical clock that tracks causality per node.
///
/// Each node maintains a counter that is incremented whenever the node produces
/// a new event.  The happened-before relation (→) is defined by Lamport/Mattern
/// vector clock rules.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct VectorClock {
    /// Map from `node_id` to that node's logical counter.
    pub entries: HashMap<String, u64>,
}

impl VectorClock {
    /// Create a new, zeroed vector clock.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// Increment the counter for `node_id` by one.
    pub fn increment(&mut self, node_id: &str) {
        let counter = self.entries.entry(node_id.to_string()).or_insert(0);
        *counter = counter.saturating_add(1);
    }

    /// Element-wise maximum of `self` and `other`.
    ///
    /// The result is the smallest vector clock that *dominates* both inputs,
    /// i.e. `self.merge(other).dominates(self) && self.merge(other).dominates(other)`.
    #[must_use]
    pub fn merge(&self, other: &VectorClock) -> VectorClock {
        let mut result = self.entries.clone();
        for (node, &count) in &other.entries {
            let entry = result.entry(node.clone()).or_insert(0);
            if count > *entry {
                *entry = count;
            }
        }
        VectorClock { entries: result }
    }

    /// `self → other`: every component of `self` is ≤ the corresponding component
    /// of `other`, and at least one is strictly less.
    #[must_use]
    pub fn happens_before(&self, other: &VectorClock) -> bool {
        let mut strictly_less = false;
        for (node, &self_count) in &self.entries {
            let other_count = other.entries.get(node).copied().unwrap_or(0);
            if self_count > other_count {
                return false;
            }
            if self_count < other_count {
                strictly_less = true;
            }
        }
        // Also check nodes that appear only in `other`.
        for (node, &other_count) in &other.entries {
            if !self.entries.contains_key(node) && other_count > 0 {
                strictly_less = true;
            }
        }
        strictly_less
    }

    /// Two clocks are concurrent when neither happens-before the other.
    #[must_use]
    pub fn concurrent_with(&self, other: &VectorClock) -> bool {
        !self.happens_before(other) && !other.happens_before(self)
    }

    /// `self` dominates `other`: every component of `self` is ≥ the corresponding
    /// component of `other` (i.e. `other → self` or `other == self`).
    ///
    /// Note: this is not strict; equal clocks both dominate each other.
    #[must_use]
    pub fn dominates(&self, other: &VectorClock) -> bool {
        for (node, &other_count) in &other.entries {
            let self_count = self.entries.get(node).copied().unwrap_or(0);
            if self_count < other_count {
                return false;
            }
        }
        true
    }

    /// The maximum counter value across all nodes, or `0` if the clock is empty.
    #[must_use]
    pub fn max_value(&self) -> u64 {
        self.entries.values().copied().max().unwrap_or(0)
    }

    /// Return the counter for `node_id`, or `0` if not present.
    #[must_use]
    pub fn get(&self, node_id: &str) -> u64 {
        self.entries.get(node_id).copied().unwrap_or(0)
    }
}

// ─── SyncEntry ───────────────────────────────────────────────────────────────

/// A key-value record annotated with causal metadata.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SyncEntry {
    /// The logical key for this entry.
    pub key: String,
    /// Arbitrary payload bytes.
    pub value: Vec<u8>,
    /// The vector clock at the time of the last write.
    pub clock: VectorClock,
    /// The node that produced this entry.
    pub node_id: String,
    /// Wall-clock timestamp (milliseconds since Unix epoch, supplied by the caller).
    pub timestamp: u64,
}

// ─── SyncOperation ───────────────────────────────────────────────────────────

/// An atomic operation that can be applied to a [`PeerSyncProtocol`].
#[derive(Clone, Debug)]
pub enum SyncOperation {
    /// Insert or update a single entry.
    Put { entry: SyncEntry },
    /// Delete a key, recording the causal context at which deletion occurred.
    Delete { key: String, clock: VectorClock },
    /// Apply a batch of entries (used for delta and full sync).
    Merge { entries: Vec<SyncEntry> },
}

// ─── SyncState ───────────────────────────────────────────────────────────────

/// The replicated state held by a single node.
#[derive(Clone, Debug)]
pub struct SyncState {
    /// Live entries indexed by key.
    pub entries: HashMap<String, SyncEntry>,
    /// Keys that have been deleted; these keys can never be re-inserted.
    pub tombstones: HashSet<String>,
    /// This node's current vector clock (incremented on every local write/delete).
    pub local_clock: VectorClock,
    /// Stable identifier for this node.
    pub node_id: String,
}

impl SyncState {
    fn new(node_id: String) -> Self {
        Self {
            entries: HashMap::new(),
            tombstones: HashSet::new(),
            local_clock: VectorClock::new(),
            node_id,
        }
    }
}

// ─── ConflictPolicy ──────────────────────────────────────────────────────────

/// Strategy used to resolve write–write conflicts (concurrent updates to the same key).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ConflictPolicy {
    /// Keep the entry with the higher `timestamp`; on tie, prefer the incoming entry.
    LastWriteWins,
    /// Keep the entry whose clock *dominates* the other.  When both clocks are
    /// concurrent, prefer the incoming entry.
    HighestClock,
    /// Concatenate both values separated by `separator`.
    MergeBytes { separator: u8 },
    /// Reject the incoming entry and keep the existing one unchanged.
    RejectConflict,
}

// ─── PeerSyncProtocol ────────────────────────────────────────────────────────

/// Bidirectional state-synchronisation protocol backed by vector clocks and CRDTs.
pub struct PeerSyncProtocol {
    /// The local replicated state.
    pub state: SyncState,
    /// How concurrent writes to the same key are resolved.
    pub conflict_policy: ConflictPolicy,
    /// Operations that have been enqueued but not yet dispatched to remote peers.
    pub pending_ops: VecDeque<SyncOperation>,
    /// Audit log: `(peer_id, operation, wall_timestamp)`.
    pub sync_log: VecDeque<(String, SyncOperation, u64)>,
}

// ─── SyncStats ───────────────────────────────────────────────────────────────

/// A snapshot of key metrics for a [`PeerSyncProtocol`] instance.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PspSyncStats {
    /// Number of live (non-tombstoned) entries.
    pub total_entries: usize,
    /// Number of tombstoned (deleted) keys.
    pub tombstone_count: usize,
    /// Number of pending outbound operations.
    pub pending_ops: usize,
    /// Number of entries in the sync audit log.
    pub sync_log_size: usize,
    /// The highest counter value in the local vector clock.
    pub local_clock_max: u64,
}

// ─── Implementation ──────────────────────────────────────────────────────────

impl PeerSyncProtocol {
    /// Create a new protocol instance for the given node.
    ///
    /// # Arguments
    /// * `node_id` — a stable, globally-unique identifier for this node.
    /// * `conflict_policy` — the policy applied when concurrent writes collide.
    pub fn new(node_id: String, conflict_policy: ConflictPolicy) -> Self {
        Self {
            state: SyncState::new(node_id),
            conflict_policy,
            pending_ops: VecDeque::new(),
            sync_log: VecDeque::new(),
        }
    }

    // ── Local mutations ───────────────────────────────────────────────────

    /// Write `value` at `key` and return the resulting [`SyncEntry`].
    ///
    /// The local vector clock is incremented before the entry is stored so that
    /// the returned entry strictly succeeds every previously stored entry from
    /// this node.
    pub fn local_put(&mut self, key: String, value: Vec<u8>, now: u64) -> SyncEntry {
        self.state
            .local_clock
            .increment(&self.state.node_id.clone());
        let entry = SyncEntry {
            key: key.clone(),
            value,
            clock: self.state.local_clock.clone(),
            node_id: self.state.node_id.clone(),
            timestamp: now,
        };
        self.state.entries.insert(key, entry.clone());
        self.pending_ops.push_back(SyncOperation::Put {
            entry: entry.clone(),
        });
        entry
    }

    /// Mark `key` as deleted.
    ///
    /// The key is added to the tombstone set and any live entry for it is removed.
    /// The local clock is incremented so that the deletion causally succeeds any
    /// previous entry.
    pub fn local_delete(&mut self, key: String, _now: u64) {
        self.state
            .local_clock
            .increment(&self.state.node_id.clone());
        let clock = self.state.local_clock.clone();
        self.state.tombstones.insert(key.clone());
        self.state.entries.remove(&key);
        self.pending_ops.push_back(SyncOperation::Delete {
            key: key.clone(),
            clock,
        });
    }

    // ── Remote operations ─────────────────────────────────────────────────

    /// Apply an operation received from a remote peer.
    ///
    /// The sync audit log is updated regardless of whether the operation caused
    /// a state change.
    ///
    /// # Errors
    /// - [`SyncError::Tombstoned`] — a `Put` arrived for a tombstoned key.
    /// - [`SyncError::ConflictRejected`] — [`ConflictPolicy::RejectConflict`] rejected a
    ///   conflicting `Put`.
    /// - [`SyncError::InvalidOperation`] — a `Merge` with zero entries was supplied.
    pub fn apply_remote_op(
        &mut self,
        peer_id: &str,
        op: SyncOperation,
        now: u64,
    ) -> Result<(), SyncError> {
        self.sync_log
            .push_back((peer_id.to_string(), op.clone(), now));

        match op {
            SyncOperation::Put { ref entry } => {
                self.apply_put(entry)?;
            }
            SyncOperation::Delete { ref key, ref clock } => {
                self.apply_delete(key, clock);
            }
            SyncOperation::Merge { ref entries } => {
                if entries.is_empty() {
                    return Err(SyncError::InvalidOperation);
                }
                for entry in entries {
                    // Individual tombstone / conflict errors are silently skipped
                    // during a bulk merge to preserve convergence.
                    let _ = self.apply_put(entry);
                }
            }
        }
        Ok(())
    }

    // ── Internal apply helpers ────────────────────────────────────────────

    fn apply_put(&mut self, incoming: &SyncEntry) -> Result<(), SyncError> {
        let key = &incoming.key;

        // Tombstoned keys are permanently suppressed.
        if self.state.tombstones.contains(key) {
            return Err(SyncError::Tombstoned(key.clone()));
        }

        match self.state.entries.get(key) {
            None => {
                // Fresh key — insert directly and advance local clock.
                self.state.local_clock = self.state.local_clock.merge(&incoming.clock);
                self.state.entries.insert(key.clone(), incoming.clone());
            }
            Some(existing) => {
                // Identical clock → idempotent no-op.
                if existing.clock == incoming.clock {
                    return Ok(());
                }

                // Incoming is strictly older than existing → discard.
                if incoming.clock.happens_before(&existing.clock) {
                    return Ok(());
                }

                // Existing is strictly older → replace with incoming.
                if existing.clock.happens_before(&incoming.clock) {
                    self.state.local_clock = self.state.local_clock.merge(&incoming.clock);
                    self.state.entries.insert(key.clone(), incoming.clone());
                    return Ok(());
                }

                // Concurrent writes — apply conflict policy.
                let existing_clone = existing.clone();
                let resolved = self.resolve_conflict_inner(&existing_clone, incoming)?;
                self.state.local_clock = self.state.local_clock.merge(&resolved.clock);
                self.state.entries.insert(key.clone(), resolved);
            }
        }
        Ok(())
    }

    fn apply_delete(&mut self, key: &str, clock: &VectorClock) {
        self.state.tombstones.insert(key.to_string());
        self.state.entries.remove(key);
        self.state.local_clock = self.state.local_clock.merge(clock);
    }

    // ── Conflict resolution ───────────────────────────────────────────────

    /// Resolve a write–write conflict between `existing` and `incoming` according
    /// to the active [`ConflictPolicy`].
    ///
    /// # Errors
    /// Returns [`SyncError::ConflictRejected`] when the policy is
    /// [`ConflictPolicy::RejectConflict`].
    pub fn resolve_conflict(
        &self,
        existing: &SyncEntry,
        incoming: &SyncEntry,
    ) -> Result<SyncEntry, SyncError> {
        self.resolve_conflict_inner(existing, incoming)
    }

    fn resolve_conflict_inner(
        &self,
        existing: &SyncEntry,
        incoming: &SyncEntry,
    ) -> Result<SyncEntry, SyncError> {
        match &self.conflict_policy {
            ConflictPolicy::LastWriteWins => {
                // Higher timestamp wins; on tie prefer incoming.
                if existing.timestamp > incoming.timestamp {
                    Ok(existing.clone())
                } else {
                    Ok(incoming.clone())
                }
            }
            ConflictPolicy::HighestClock => {
                if existing.clock.dominates(&incoming.clock)
                    && !incoming.clock.dominates(&existing.clock)
                {
                    // existing strictly dominates
                    Ok(existing.clone())
                } else {
                    // incoming dominates or they are concurrent → prefer incoming
                    Ok(incoming.clone())
                }
            }
            ConflictPolicy::MergeBytes { separator } => {
                let sep = *separator;
                let mut merged_value = existing.value.clone();
                merged_value.push(sep);
                merged_value.extend_from_slice(&incoming.value);
                let merged_clock = existing.clock.merge(&incoming.clock);
                Ok(SyncEntry {
                    key: existing.key.clone(),
                    value: merged_value,
                    clock: merged_clock,
                    node_id: incoming.node_id.clone(),
                    timestamp: existing.timestamp.max(incoming.timestamp),
                })
            }
            ConflictPolicy::RejectConflict => Err(SyncError::ConflictRejected {
                key: existing.key.clone(),
            }),
        }
    }

    // ── Delta / full sync ─────────────────────────────────────────────────

    /// Generate the set of local entries that a peer is missing.
    ///
    /// An entry is considered missing from the peer when its clock does **not**
    /// happen-before `since_clock` (i.e. the peer has not yet observed it).
    /// All matching entries are returned as a single [`SyncOperation::Merge`].
    /// An empty `Vec` is returned when nothing needs to be sent.
    pub fn generate_delta(&self, since_clock: &VectorClock) -> Vec<SyncOperation> {
        let missing: Vec<SyncEntry> = self
            .state
            .entries
            .values()
            .filter(|e| !e.clock.happens_before(since_clock))
            .cloned()
            .collect();

        if missing.is_empty() {
            return Vec::new();
        }
        vec![SyncOperation::Merge { entries: missing }]
    }

    /// Return all live (non-tombstoned) entries as a single
    /// [`SyncOperation::Merge`] suitable for a full-state transfer.
    pub fn full_sync(&self) -> SyncOperation {
        SyncOperation::Merge {
            entries: self.state.entries.values().cloned().collect(),
        }
    }

    // ── Read accessors ────────────────────────────────────────────────────

    /// Look up a live entry by key.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&SyncEntry> {
        self.state.entries.get(key)
    }

    /// Returns `true` if `key` exists as a live entry.
    #[must_use]
    pub fn contains(&self, key: &str) -> bool {
        self.state.entries.contains_key(key)
    }

    /// Returns `true` if `key` has been deleted (tombstoned).
    #[must_use]
    pub fn is_tombstoned(&self, key: &str) -> bool {
        self.state.tombstones.contains(key)
    }

    /// Number of live (non-tombstoned) entries.
    #[must_use]
    pub fn entry_count(&self) -> usize {
        self.state.entries.len()
    }

    /// The current vector-clock counter for `node_id`, or `0` if unknown.
    #[must_use]
    pub fn clock_value(&self, node_id: &str) -> u64 {
        self.state.local_clock.get(node_id)
    }

    /// Return a statistics snapshot for this instance.
    #[must_use]
    pub fn stats(&self) -> PspSyncStats {
        PspSyncStats {
            total_entries: self.state.entries.len(),
            tombstone_count: self.state.tombstones.len(),
            pending_ops: self.pending_ops.len(),
            sync_log_size: self.sync_log.len(),
            local_clock_max: self.state.local_clock.max_value(),
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use crate::peer_sync_protocol::{
        ConflictPolicy, PeerSyncProtocol, PspSyncStats, SyncEntry, SyncError, SyncOperation,
        VectorClock,
    };

    // ── VectorClock ──────────────────────────────────────────────────────

    #[test]
    fn vc_new_is_empty() {
        let vc = VectorClock::new();
        assert!(vc.entries.is_empty());
        assert_eq!(vc.max_value(), 0);
    }

    #[test]
    fn vc_increment_creates_entry() {
        let mut vc = VectorClock::new();
        vc.increment("a");
        assert_eq!(vc.get("a"), 1);
    }

    #[test]
    fn vc_increment_twice() {
        let mut vc = VectorClock::new();
        vc.increment("a");
        vc.increment("a");
        assert_eq!(vc.get("a"), 2);
    }

    #[test]
    fn vc_increment_different_nodes() {
        let mut vc = VectorClock::new();
        vc.increment("a");
        vc.increment("b");
        assert_eq!(vc.get("a"), 1);
        assert_eq!(vc.get("b"), 1);
    }

    #[test]
    fn vc_merge_element_wise_max() {
        let mut a = VectorClock::new();
        a.increment("x");
        a.increment("x"); // x=2
        a.increment("y"); // y=1

        let mut b = VectorClock::new();
        b.increment("x"); // x=1
        b.increment("z"); // z=1

        let merged = a.merge(&b);
        assert_eq!(merged.get("x"), 2);
        assert_eq!(merged.get("y"), 1);
        assert_eq!(merged.get("z"), 1);
    }

    #[test]
    fn vc_merge_is_commutative() {
        let mut a = VectorClock::new();
        a.increment("a");
        a.increment("a");
        let mut b = VectorClock::new();
        b.increment("a");
        b.increment("b");

        assert_eq!(a.merge(&b), b.merge(&a));
    }

    #[test]
    fn vc_happens_before_strict() {
        let mut old = VectorClock::new();
        old.increment("n");
        let mut new = VectorClock::new();
        new.increment("n");
        new.increment("n");

        assert!(old.happens_before(&new));
        assert!(!new.happens_before(&old));
    }

    #[test]
    fn vc_equal_clocks_not_happens_before() {
        let mut a = VectorClock::new();
        a.increment("n");
        let b = a.clone();
        assert!(!a.happens_before(&b));
        assert!(!b.happens_before(&a));
    }

    #[test]
    fn vc_concurrent_with() {
        let mut a = VectorClock::new();
        a.increment("a");
        let mut b = VectorClock::new();
        b.increment("b");

        assert!(a.concurrent_with(&b));
        assert!(b.concurrent_with(&a));
    }

    #[test]
    fn vc_not_concurrent_when_ordered() {
        let mut early = VectorClock::new();
        early.increment("n");
        let mut late = early.clone();
        late.increment("n");

        assert!(!early.concurrent_with(&late));
        assert!(!late.concurrent_with(&early));
    }

    #[test]
    fn vc_dominates_self() {
        let mut vc = VectorClock::new();
        vc.increment("n");
        assert!(vc.dominates(&vc.clone()));
    }

    #[test]
    fn vc_dominates_older() {
        let mut old = VectorClock::new();
        old.increment("n");
        let mut new = old.clone();
        new.increment("n");

        assert!(new.dominates(&old));
        assert!(!old.dominates(&new));
    }

    #[test]
    fn vc_get_missing_node_returns_zero() {
        let vc = VectorClock::new();
        assert_eq!(vc.get("nonexistent"), 0);
    }

    #[test]
    fn vc_max_value_multiple_nodes() {
        let mut vc = VectorClock::new();
        vc.increment("a");
        vc.increment("b");
        vc.increment("b");
        vc.increment("b");
        assert_eq!(vc.max_value(), 3);
    }

    // ── PeerSyncProtocol — local operations ──────────────────────────────

    #[test]
    fn local_put_stores_entry() {
        let mut proto = PeerSyncProtocol::new("node-a".to_string(), ConflictPolicy::LastWriteWins);
        proto.local_put("k1".to_string(), b"hello".to_vec(), 100);
        assert!(proto.contains("k1"));
        assert_eq!(
            proto.get("k1").expect("test: k1 entry must exist").value,
            b"hello"
        );
    }

    #[test]
    fn local_put_increments_clock() {
        let mut proto = PeerSyncProtocol::new("node-a".to_string(), ConflictPolicy::LastWriteWins);
        proto.local_put("k".to_string(), b"v".to_vec(), 1);
        assert_eq!(proto.clock_value("node-a"), 1);
        proto.local_put("k2".to_string(), b"v2".to_vec(), 2);
        assert_eq!(proto.clock_value("node-a"), 2);
    }

    #[test]
    fn local_put_returns_correct_entry() {
        let mut proto = PeerSyncProtocol::new("n".to_string(), ConflictPolicy::LastWriteWins);
        let entry = proto.local_put("key".to_string(), b"val".to_vec(), 42);
        assert_eq!(entry.key, "key");
        assert_eq!(entry.value, b"val");
        assert_eq!(entry.timestamp, 42);
        assert_eq!(entry.node_id, "n");
    }

    #[test]
    fn local_put_enqueues_pending_op() {
        let mut proto = PeerSyncProtocol::new("n".to_string(), ConflictPolicy::LastWriteWins);
        proto.local_put("k".to_string(), b"v".to_vec(), 1);
        assert_eq!(proto.pending_ops.len(), 1);
    }

    #[test]
    fn local_delete_tombstones_key() {
        let mut proto = PeerSyncProtocol::new("n".to_string(), ConflictPolicy::LastWriteWins);
        proto.local_put("k".to_string(), b"v".to_vec(), 1);
        proto.local_delete("k".to_string(), 2);
        assert!(!proto.contains("k"));
        assert!(proto.is_tombstoned("k"));
    }

    #[test]
    fn local_delete_increments_clock() {
        let mut proto = PeerSyncProtocol::new("n".to_string(), ConflictPolicy::LastWriteWins);
        proto.local_delete("k".to_string(), 5);
        assert_eq!(proto.clock_value("n"), 1);
    }

    // ── apply_remote_op — Put ────────────────────────────────────────────

    #[test]
    fn apply_remote_put_fresh_key() {
        let mut node_a = PeerSyncProtocol::new("a".to_string(), ConflictPolicy::LastWriteWins);
        let entry = node_a.local_put("x".to_string(), b"data".to_vec(), 10);

        let mut node_b = PeerSyncProtocol::new("b".to_string(), ConflictPolicy::LastWriteWins);
        node_b
            .apply_remote_op("a", SyncOperation::Put { entry }, 11)
            .expect("test: apply remote put for fresh key x should succeed");
        assert_eq!(
            node_b
                .get("x")
                .expect("test: x entry must exist after remote put")
                .value,
            b"data"
        );
    }

    #[test]
    fn apply_remote_put_tombstoned_returns_error() {
        let mut node = PeerSyncProtocol::new("a".to_string(), ConflictPolicy::LastWriteWins);
        node.local_delete("k".to_string(), 1);

        let mut vc = VectorClock::new();
        vc.increment("b");
        let entry = SyncEntry {
            key: "k".to_string(),
            value: b"late".to_vec(),
            clock: vc,
            node_id: "b".to_string(),
            timestamp: 5,
        };
        let result = node.apply_remote_op("b", SyncOperation::Put { entry }, 6);
        assert_eq!(result, Err(SyncError::Tombstoned("k".to_string())));
    }

    #[test]
    fn apply_remote_put_older_clock_discarded() {
        let mut node = PeerSyncProtocol::new("n".to_string(), ConflictPolicy::LastWriteWins);
        // Insert a newer entry locally.
        node.local_put("k".to_string(), b"new".to_vec(), 100);

        // Simulate an older remote entry.
        let mut old_vc = VectorClock::new();
        old_vc.increment("remote");
        let old_entry = SyncEntry {
            key: "k".to_string(),
            value: b"old".to_vec(),
            clock: old_vc,
            node_id: "remote".to_string(),
            timestamp: 1,
        };
        node.apply_remote_op("remote", SyncOperation::Put { entry: old_entry }, 2)
            .expect("test: apply older remote put should succeed without error");
        // Value must not be overwritten by an older entry.
        assert_eq!(
            node.get("k")
                .expect("test: k entry must still exist after discarded old entry")
                .value,
            b"new"
        );
    }

    #[test]
    fn apply_remote_delete_removes_entry() {
        let mut node_a = PeerSyncProtocol::new("a".to_string(), ConflictPolicy::LastWriteWins);
        node_a.local_put("x".to_string(), b"hi".to_vec(), 1);

        let mut del_clock = VectorClock::new();
        del_clock.increment("b");
        node_a
            .apply_remote_op(
                "b",
                SyncOperation::Delete {
                    key: "x".to_string(),
                    clock: del_clock,
                },
                2,
            )
            .expect("test: apply remote delete for key x should succeed");
        assert!(!node_a.contains("x"));
        assert!(node_a.is_tombstoned("x"));
    }

    #[test]
    fn apply_remote_merge_applies_all_entries() {
        let mut node_a = PeerSyncProtocol::new("a".to_string(), ConflictPolicy::LastWriteWins);
        let e1 = node_a.local_put("k1".to_string(), b"v1".to_vec(), 10);
        let e2 = node_a.local_put("k2".to_string(), b"v2".to_vec(), 11);

        let mut node_b = PeerSyncProtocol::new("b".to_string(), ConflictPolicy::LastWriteWins);
        node_b
            .apply_remote_op(
                "a",
                SyncOperation::Merge {
                    entries: vec![e1, e2],
                },
                12,
            )
            .expect("test: apply remote merge with two entries should succeed");
        assert_eq!(node_b.entry_count(), 2);
        assert!(node_b.contains("k1"));
        assert!(node_b.contains("k2"));
    }

    #[test]
    fn apply_remote_merge_empty_returns_invalid() {
        let mut node = PeerSyncProtocol::new("n".to_string(), ConflictPolicy::LastWriteWins);
        let result = node.apply_remote_op("peer", SyncOperation::Merge { entries: vec![] }, 1);
        assert_eq!(result, Err(SyncError::InvalidOperation));
    }

    // ── Conflict resolution policies ─────────────────────────────────────

    #[test]
    fn conflict_last_write_wins_picks_higher_timestamp() {
        let mut proto = PeerSyncProtocol::new("n".to_string(), ConflictPolicy::LastWriteWins);
        // Insert existing entry.
        proto.local_put("k".to_string(), b"existing".to_vec(), 50);

        // Incoming has higher timestamp → should win.
        let mut incoming_vc = VectorClock::new();
        incoming_vc.increment("peer");
        let incoming = SyncEntry {
            key: "k".to_string(),
            value: b"incoming".to_vec(),
            clock: incoming_vc,
            node_id: "peer".to_string(),
            timestamp: 100,
        };
        proto
            .apply_remote_op("peer", SyncOperation::Put { entry: incoming }, 101)
            .expect("test: apply remote put with higher timestamp should succeed");
        assert_eq!(
            proto
                .get("k")
                .expect("test: k entry must exist after LastWriteWins conflict")
                .value,
            b"incoming"
        );
    }

    #[test]
    fn conflict_last_write_wins_keeps_existing_on_higher_ts() {
        let mut proto = PeerSyncProtocol::new("n".to_string(), ConflictPolicy::LastWriteWins);
        proto.local_put("k".to_string(), b"existing".to_vec(), 200);

        let mut incoming_vc = VectorClock::new();
        incoming_vc.increment("peer");
        let incoming = SyncEntry {
            key: "k".to_string(),
            value: b"stale".to_vec(),
            clock: incoming_vc,
            node_id: "peer".to_string(),
            timestamp: 100,
        };
        proto
            .apply_remote_op("peer", SyncOperation::Put { entry: incoming }, 201)
            .expect("test: apply remote put with lower timestamp should succeed");
        assert_eq!(
            proto
                .get("k")
                .expect("test: k entry must exist; existing higher-ts entry retained")
                .value,
            b"existing"
        );
    }

    #[test]
    fn conflict_highest_clock_incoming_dominates() {
        let mut proto = PeerSyncProtocol::new("n".to_string(), ConflictPolicy::HighestClock);
        proto.local_put("k".to_string(), b"old".to_vec(), 1);

        // Build an incoming clock that dominates local clock.
        let mut big_vc = proto.state.local_clock.clone();
        big_vc.increment("peer");
        big_vc.increment("peer");
        let incoming = SyncEntry {
            key: "k".to_string(),
            value: b"newer".to_vec(),
            clock: big_vc,
            node_id: "peer".to_string(),
            timestamp: 1,
        };
        proto
            .apply_remote_op("peer", SyncOperation::Put { entry: incoming }, 2)
            .expect("test: apply remote put with dominating clock should succeed");
        assert_eq!(
            proto
                .get("k")
                .expect("test: k entry must exist after HighestClock conflict")
                .value,
            b"newer"
        );
    }

    #[test]
    fn conflict_merge_bytes_concatenates() {
        let mut proto = PeerSyncProtocol::new(
            "n".to_string(),
            ConflictPolicy::MergeBytes { separator: b'|' },
        );
        proto.local_put("k".to_string(), b"left".to_vec(), 1);

        let mut peer_vc = VectorClock::new();
        peer_vc.increment("peer");
        let incoming = SyncEntry {
            key: "k".to_string(),
            value: b"right".to_vec(),
            clock: peer_vc,
            node_id: "peer".to_string(),
            timestamp: 1,
        };
        proto
            .apply_remote_op("peer", SyncOperation::Put { entry: incoming }, 2)
            .expect("test: apply remote put for MergeBytes conflict should succeed");
        let merged = &proto
            .get("k")
            .expect("test: k entry must exist after MergeBytes conflict")
            .value;
        assert!(merged.contains(&b'|'));
        assert!(merged.windows(4).any(|w| w == b"left"));
        assert!(merged.windows(5).any(|w| w == b"right"));
    }

    #[test]
    fn conflict_reject_returns_error() {
        let mut proto = PeerSyncProtocol::new("n".to_string(), ConflictPolicy::RejectConflict);
        proto.local_put("k".to_string(), b"existing".to_vec(), 1);

        let mut peer_vc = VectorClock::new();
        peer_vc.increment("peer");
        let incoming = SyncEntry {
            key: "k".to_string(),
            value: b"conflict".to_vec(),
            clock: peer_vc,
            node_id: "peer".to_string(),
            timestamp: 2,
        };
        let result = proto.apply_remote_op("peer", SyncOperation::Put { entry: incoming }, 3);
        assert!(matches!(result, Err(SyncError::ConflictRejected { .. })));
        // Existing value is preserved.
        assert_eq!(
            proto
                .get("k")
                .expect("test: k entry must exist; RejectConflict keeps existing")
                .value,
            b"existing"
        );
    }

    // ── Delta / full sync ────────────────────────────────────────────────

    #[test]
    fn generate_delta_returns_missing_entries() {
        let mut node_a = PeerSyncProtocol::new("a".to_string(), ConflictPolicy::LastWriteWins);
        node_a.local_put("k1".to_string(), b"v1".to_vec(), 1);
        node_a.local_put("k2".to_string(), b"v2".to_vec(), 2);

        // Empty peer clock — peer has seen nothing.
        let peer_clock = VectorClock::new();
        let delta = node_a.generate_delta(&peer_clock);
        assert_eq!(delta.len(), 1);
        if let SyncOperation::Merge { entries } = &delta[0] {
            assert_eq!(entries.len(), 2);
        } else {
            panic!("expected Merge");
        }
    }

    #[test]
    fn generate_delta_empty_when_peer_up_to_date() {
        let mut node_a = PeerSyncProtocol::new("a".to_string(), ConflictPolicy::LastWriteWins);
        node_a.local_put("k".to_string(), b"v".to_vec(), 1);

        // Peer clock that dominates all entries.
        let current_clock = node_a.state.local_clock.clone();
        // A clock that is strictly greater than the entry's clock.
        let mut ahead_clock = current_clock.clone();
        ahead_clock.increment("a");

        let delta = node_a.generate_delta(&ahead_clock);
        assert!(delta.is_empty());
    }

    #[test]
    fn full_sync_returns_all_live_entries() {
        let mut node = PeerSyncProtocol::new("n".to_string(), ConflictPolicy::LastWriteWins);
        node.local_put("a".to_string(), b"1".to_vec(), 1);
        node.local_put("b".to_string(), b"2".to_vec(), 2);
        node.local_delete("b".to_string(), 3);

        if let SyncOperation::Merge { entries } = node.full_sync() {
            // Only "a" is live; "b" was tombstoned.
            assert_eq!(entries.len(), 1);
            assert_eq!(entries[0].key, "a");
        } else {
            panic!("expected Merge");
        }
    }

    // ── Stats ────────────────────────────────────────────────────────────

    #[test]
    fn stats_initial_state() {
        let proto = PeerSyncProtocol::new("n".to_string(), ConflictPolicy::LastWriteWins);
        let s = proto.stats();
        assert_eq!(
            s,
            PspSyncStats {
                total_entries: 0,
                tombstone_count: 0,
                pending_ops: 0,
                sync_log_size: 0,
                local_clock_max: 0,
            }
        );
    }

    #[test]
    fn stats_after_operations() {
        let mut proto = PeerSyncProtocol::new("n".to_string(), ConflictPolicy::LastWriteWins);
        proto.local_put("k1".to_string(), b"v1".to_vec(), 1);
        proto.local_put("k2".to_string(), b"v2".to_vec(), 2);
        proto.local_delete("k1".to_string(), 3);
        let s = proto.stats();
        assert_eq!(s.total_entries, 1);
        assert_eq!(s.tombstone_count, 1);
        assert_eq!(s.pending_ops, 3); // put, put, delete
        assert_eq!(s.local_clock_max, 3);
    }

    #[test]
    fn stats_sync_log_grows_on_remote_ops() {
        let mut node_a = PeerSyncProtocol::new("a".to_string(), ConflictPolicy::LastWriteWins);
        let entry = node_a.local_put("k".to_string(), b"v".to_vec(), 1);

        let mut node_b = PeerSyncProtocol::new("b".to_string(), ConflictPolicy::LastWriteWins);
        node_b
            .apply_remote_op("a", SyncOperation::Put { entry }, 2)
            .expect("test: apply remote put for sync log growth test should succeed");
        assert_eq!(node_b.stats().sync_log_size, 1);
    }

    // ── Convergence / CRDT invariants ────────────────────────────────────

    #[test]
    fn convergence_two_nodes_put_same_key() {
        // Both nodes write concurrently; after exchanging ops they should converge.
        let mut a = PeerSyncProtocol::new("a".to_string(), ConflictPolicy::LastWriteWins);
        let mut b = PeerSyncProtocol::new("b".to_string(), ConflictPolicy::LastWriteWins);

        let ea = a.local_put("k".to_string(), b"from-a".to_vec(), 10);
        let eb = b.local_put("k".to_string(), b"from-b".to_vec(), 20);

        // Exchange
        let _ = a.apply_remote_op("b", SyncOperation::Put { entry: eb }, 21);
        let _ = b.apply_remote_op("a", SyncOperation::Put { entry: ea }, 21);

        // Both should agree on the same value (higher timestamp wins).
        assert_eq!(
            a.get("k")
                .expect("test: node a must have k after convergence exchange")
                .value,
            b.get("k")
                .expect("test: node b must have k after convergence exchange")
                .value
        );
    }

    #[test]
    fn idempotent_apply_put_same_entry_twice() {
        let mut a = PeerSyncProtocol::new("a".to_string(), ConflictPolicy::LastWriteWins);
        let entry = a.local_put("k".to_string(), b"v".to_vec(), 1);

        let mut b = PeerSyncProtocol::new("b".to_string(), ConflictPolicy::LastWriteWins);
        b.apply_remote_op(
            "a",
            SyncOperation::Put {
                entry: entry.clone(),
            },
            2,
        )
        .expect("test: first idempotent apply of entry should succeed");
        b.apply_remote_op("a", SyncOperation::Put { entry }, 3)
            .expect("test: second idempotent apply of same entry should succeed");
        // Entry count must be 1, not 2.
        assert_eq!(b.entry_count(), 1);
    }

    #[test]
    fn tombstone_blocks_late_arriving_put_in_merge() {
        let mut node = PeerSyncProtocol::new("n".to_string(), ConflictPolicy::LastWriteWins);
        node.local_delete("k".to_string(), 1);

        // A merge containing a put for the tombstoned key should be silently skipped.
        let mut vc = VectorClock::new();
        vc.increment("peer");
        let late = SyncEntry {
            key: "k".to_string(),
            value: b"late".to_vec(),
            clock: vc,
            node_id: "peer".to_string(),
            timestamp: 5,
        };
        // Merge silently skips tombstoned entries (does not return error for batch).
        let result = node.apply_remote_op(
            "peer",
            SyncOperation::Merge {
                entries: vec![late],
            },
            6,
        );
        assert!(result.is_ok());
        assert!(!node.contains("k"));
        assert!(node.is_tombstoned("k"));
    }
}
